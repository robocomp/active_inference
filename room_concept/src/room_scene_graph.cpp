/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */

#include <algorithm>
#include <ranges>
#include "room_scene_graph.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

#include <QString>
#include <QtCore/qdebug.h>
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc
{

namespace
{
// A furniture concept node now lives as a generic type()=="object" node whose class is carried in the
// object_subtype string attribute; its NAME prefix (table_*, chair_*, …) is unchanged. Match either, so a
// producer that publishes only the name (or only the subtype) is still recognised as that class.
bool node_is_object_class(DSR::DSRGraph& G, const DSR::Node& n, std::string_view cls)
{
    if (std::string_view(n.name()).starts_with(cls))
        return true;
    if (const auto s = G.get_attrib_by_name<object_subtype_att>(n); s.has_value())
        return s.value() == cls;
    return false;
}

// ── Floor display mesh ────────────────────────────────────────────────────────────────────────
// The floor's shape IS the room layout, so unlike the wall panel it cannot be a fixed asset: it is
// generated from the live polygon and rewritten whenever that polygon is authored. Everything else
// about it follows the shared contract — a unit-box OBJ (x,y in [-0.5,0.5]) that a consumer rescales
// by width_m/depth_m.

// Ear clipping, because nominal_room_polygon() may return an arbitrary CONFIGURED polygon
// (room_concept.h: init_polygon_vertices_), not only the default rectangle — a centroid fan would
// emit triangles outside a non-convex room.
std::vector<std::array<int, 3>> triangulate_polygon(std::vector<Eigen::Vector2f> p)
{
    std::vector<std::array<int, 3>> tris;
    const int n = static_cast<int>(p.size());
    if (n < 3)
        return tris;

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    // Work counter-clockwise so the "convex vertex" test has a fixed sense.
    float area2 = 0.f;
    for (int i = 0; i < n; ++i)
    {
        const auto& a = p[i]; const auto& b = p[(i + 1) % n];
        area2 += a.x() * b.y() - b.x() * a.y();
    }
    if (area2 < 0.f)
        std::reverse(idx.begin(), idx.end());

    const auto cross = [](const Eigen::Vector2f& o, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    { return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x()); };

    int guard = 0;
    while (idx.size() > 3 && guard++ < 4 * n)
    {
        bool clipped = false;
        const int m = static_cast<int>(idx.size());
        for (int i = 0; i < m; ++i)
        {
            const int ia = idx[(i + m - 1) % m], ib = idx[i], ic = idx[(i + 1) % m];
            if (cross(p[ia], p[ib], p[ic]) <= 0.f)
                continue;   // reflex vertex — not an ear
            bool contains = false;
            for (const int j : idx)
            {
                if (j == ia || j == ib || j == ic)
                    continue;
                if (cross(p[ia], p[ib], p[j]) >= 0.f && cross(p[ib], p[ic], p[j]) >= 0.f
                    && cross(p[ic], p[ia], p[j]) >= 0.f)
                { contains = true; break; }
            }
            if (contains)
                continue;
            tris.push_back({ia, ib, ic});
            idx.erase(idx.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped)
            break;   // degenerate/self-intersecting input — emit what we have rather than spin
    }
    if (idx.size() == 3)
        tris.push_back({idx[0], idx[1], idx[2]});
    return tris;
}

// Writes the floor OBJ and returns its bounding-box extents (metres) for width_m/depth_m.
// `out` is relative to the agent's run dir, matching how the published path resolves for consumers.
bool write_floor_obj(const std::vector<Eigen::Vector2f>& poly, const std::filesystem::path& out,
                     float& out_w, float& out_d)
{
    if (poly.size() < 3)
        return false;
    float minx = poly[0].x(), maxx = minx, miny = poly[0].y(), maxy = miny;
    for (const auto& v : poly)
    {
        minx = std::min(minx, v.x()); maxx = std::max(maxx, v.x());
        miny = std::min(miny, v.y()); maxy = std::max(maxy, v.y());
    }
    out_w = std::max(maxx - minx, 1e-3f);
    out_d = std::max(maxy - miny, 1e-3f);
    const float cx = 0.5f * (minx + maxx), cy = 0.5f * (miny + maxy);

    const auto tris = triangulate_polygon(poly);
    if (tris.empty())
        return false;

    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);
    std::ofstream o(out);
    if (!o)
        return false;
    o << "# room_concept floor — GENERATED from the live room polygon; do not hand-edit.\n"
      << "# Rewritten whenever the layout is authored, which a fixed asset could never track.\n"
      << "# Unit-box convention: x,y in [-0.5,0.5] about the polygon's bbox centre, z = 0 (flat);\n"
      << "# true size comes from the floor node's width_m/depth_m.\n";
    for (const auto& v : poly)
        o << "v " << (v.x() - cx) / out_w << ' ' << (v.y() - cy) / out_d << " 0.000000\n";
    for (const auto& v : poly)
        o << "vt " << (v.x() - minx) / out_w << ' ' << (v.y() - miny) / out_d << '\n';
    for (const auto& t : tris)
        o << "f " << t[0] + 1 << '/' << t[0] + 1 << ' '
                  << t[1] + 1 << '/' << t[1] + 1 << ' '
                  << t[2] + 1 << '/' << t[2] + 1 << '\n';
    return static_cast<bool>(o);
}

constexpr std::string_view kFloorMeshRel = "room_concept/meshes/floor.obj";
constexpr std::string_view kFloorMeshOut = "meshes/floor.obj";   // relative to the run dir
}  // namespace

void RoomSceneGraph::monitor_affordance()
{
    if (G_)
        affordance_manager_.monitor_execution(G_);
}

void RoomSceneGraph::on_controller_lost()
{
    if (!G_)
        return;
    if (affordance_manager_.release_execution_claim(G_))
    {
        if (epistemic_)
            epistemic_->epistemic_planner().clear_target();
        qWarning() << "[Presence] released stale afford_room execution claim after controller loss";
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::update(const rc::RoomConcept::UpdateResult& res, float adv, float side, float rot,
                           bool write_rt)
{
    last_adv_  = adv;
    last_side_ = side;
    last_rot_  = rot;

    const float sdf_mse = res.sdf_mse;
    const float cov_tt  = (res.covariance.rows() > 2 && res.covariance.cols() > 2)
                          ? res.covariance(2, 2) : 1.f;
    const bool stable   = (res.iterations_used == 0)
                          && sdf_mse < params_->STABLE_SDF_MSE_MAX
                          && cov_tt  < params_->STABLE_COV_TT_MAX;

    // Steady-state stability, for the object-anchor pin guard. Maintained on EVERY frame — unlike
    // stable_frames_ below, which stops being updated the moment the room node exists.
    anchor_stable_frames_ = stable ? anchor_stable_frames_ + 1 : 0;

    if (!room_node_created_)
    {
        stable_frames_ = stable ? stable_frames_ + 1 : 0;
        if (stable_frames_ >= params_->STABLE_FRAMES_REQUIRED)
            dsr_create_room_and_reparent(res);
        else if (write_rt)
            dsr_update_pose(res);   // world->robot RT while waiting for stable room creation
    }
    else
    {
        if (write_rt)
            dsr_update_pose(res);   // robot->room RT (skipped when the odometry publisher owns it)
        if (params_->PUBLISH_AFFORDANCE)
            dsr_update_affordance(res); // publish epistemic target affordance (off ⇒ room never competes)
        // ★NOT GATED ON PUBLISH_AFFORDANCE, and not gated on CalibPivotEnabled either. The PASSIVE
        // half of this — watching how well the odometry's rotation scale is known — commands nothing
        // and costs one window per second; free data is free whether or not anyone acts on it, and a
        // run that never offers the manoeuvre still ends knowing whether it should have.
        dsr_update_calibration(res);
    }
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_update_pose(const rc::RoomConcept::UpdateResult& res)
{
    write_robot_room_rt(res.robot_pose, res.covariance, static_cast<std::uint64_t>(res.timestamp_ms));
}

// High-rate predicted pose (dead-reckoned between lidar corrections): same RT edge, same frame/cov
// conversion as a corrected pose — only the source and the (grown) covariance + validity stamp differ.
void RoomSceneGraph::dsr_publish_predicted_pose(const Eigen::Affine2f& robot_pose,
                                                const Eigen::Matrix3f& covariance,
                                                std::uint64_t timestamp_ms)
{
    write_robot_room_rt(robot_pose, covariance, timestamp_ms);
}

// Shared writer for the robot↔room RT edge. `robot_pose`/`covariance` are the room←robot estimate
// (room frame); when the room is a child of the robot we invert to robot→room (pose AND cov Jacobian)
// before the timestamped ring-buffer write. Called by both the corrected and predicted paths.
void RoomSceneGraph::write_robot_room_rt(const Eigen::Affine2f& robot_pose,
                                         const Eigen::Matrix3f& covariance,
                                         std::uint64_t timestamp_ms)
{
    if (!G_ || !rt_api_) return;

    // ── SAFETY BARRIER: never publish a non-finite pose to the graph ─────────────────────────────
    // This RT edge is what every other agent — including the controller that drives the wheels —
    // reads as "where the robot is". A NaN/Inf here does not stay inside this process: it propagates
    // into the RT tree, poisons every inner_eigen transform chained through it, and reaches motion
    // control. On a real robot around people that is not an acceptable failure mode, so the write is
    // refused outright and the last good edge is left in place (consumers then see it age, which the
    // freshness-as-precision path already treats as growing uncertainty). Observed 2026-07-21: a
    // singular Hessian (cond_num sentinel 1e8) produced a NaN pose that reached the graph and made
    // the robot vanish from the canvas; downstream agents had no way to tell.
    const bool pose_finite = robot_pose.matrix().allFinite() and covariance.allFinite();
    if (not pose_finite)
    {
        static std::uint64_t nan_rt_k = 0;
        if ((nan_rt_k++ % 20) == 0)
            qCritical() << "[SAFETY] refusing to publish NON-FINITE robot RT (pose_finite="
                        << robot_pose.matrix().allFinite() << "cov_finite=" << covariance.allFinite()
                        << ") — keeping the last good edge; localizer is diverged, count=" << nan_rt_k;
        return;
    }

    const auto describe_node = [this](uint64_t id) -> QString
    {
        if (id == 0)
            return "0:<unset>";
        if (const auto node = G_->get_node(id); node.has_value())
            return QString::number(id) + ":" + QString::fromStdString(node->name());
        return QString::number(id) + ":<missing>";
    };

    const auto log_cached_ids = [this, &describe_node](const char *reason, uint64_t missing_child_id)
    {
        const char *missing_child_kind = "other";
        if (missing_child_id == dsr_room_id_)
            missing_child_kind = "room";
        else if (missing_child_id == dsr_robot_id_)
            missing_child_kind = "robot";
        else if (missing_child_id == dsr_world_id_)
            missing_child_kind = "world";

        qWarning() << "dsr_update_pose:" << reason
                   << "missing_child_kind=" << missing_child_kind
                   << "missing_child_id=" << missing_child_id
                   << "world=" << describe_node(dsr_world_id_)
                   << "robot=" << describe_node(dsr_robot_id_)
                   << "room=" << describe_node(dsr_room_id_);
    };

    if ((dsr_world_id_ == 0 || !G_->get_node(dsr_world_id_).has_value()) ||
        (dsr_robot_id_ == 0 || !G_->get_node(dsr_robot_id_).has_value()))
    {
        check_init_graph_is_valid();
    }

    if (room_node_created_ && !G_->get_node(dsr_room_id_).has_value())
    {
        qWarning() << "dsr_update_pose: cached room node missing, resetting room state"
                   << "room_id=" << dsr_room_id_;
        log_cached_ids("cached room node missing", dsr_room_id_);
        room_node_created_ = false;
        dsr_room_id_ = 0;
        affordance_manager_.reset();
        stable_frames_ = 0;
    }

    const Eigen::Matrix2f R = robot_pose.linear();
    const Eigen::Vector2f t = robot_pose.translation();
    const float theta_room_to_robot = std::atan2(R(1, 0), R(0, 0));

    // Convert room->robot estimate into robot->room when the room is a child of the robot.
    const Eigen::Vector2f t_robot_to_room = -(R.transpose() * t);
    const float theta_robot_to_room = -theta_room_to_robot;

    const uint64_t parent_id = room_node_created_ ? dsr_robot_id_ : dsr_world_id_;
    const uint64_t child_id  = room_node_created_ ? dsr_room_id_  : dsr_robot_id_;

    auto parent_opt = G_->get_node(parent_id);
    if (!parent_opt.has_value()) return;

    if (!G_->get_node(child_id).has_value())
    {
        qWarning() << "dsr_update_pose: destination node missing, skipping RT update"
                   << "parent_id=" << parent_id
                   << "child_id=" << child_id
                   << "room_node_created=" << room_node_created_;
        log_cached_ids("destination node missing", child_id);
        if (room_node_created_ && child_id == dsr_room_id_)
        {
            room_node_created_ = false;
            dsr_room_id_ = 0;
            affordance_manager_.reset();
            stable_frames_ = 0;
        }
        return;
    }

    const float x     = room_node_created_ ? t_robot_to_room.x() : t.x();
    const float y     = room_node_created_ ? t_robot_to_room.y() : t.y();
    const float theta = room_node_created_ ? theta_robot_to_room : theta_room_to_robot;

    // ── Covariance (SE2 3×3 packed into 6×6 flat row-major) ───────────────
    Eigen::Matrix3f cov_se2 = Eigen::Matrix3f::Identity();
    if (covariance.rows() >= 3 && covariance.cols() >= 3)
        cov_se2 = covariance.topLeftCorner<3, 3>();

    if (room_node_created_)
    {
        const float c = std::cos(theta_room_to_robot);
        const float s = std::sin(theta_room_to_robot);
        Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
        J(0, 0) = -c;  J(0, 1) = -s;  J(0, 2) =  s * t.x() - c * t.y();
        J(1, 0) =  s;  J(1, 1) = -c;  J(1, 2) =  c * t.x() + s * t.y();
        J(2, 2) = -1.f;
        cov_se2 = J * cov_se2 * J.transpose();
    }

    // ── SE(2) -> SE3 SLOTS, AND THE MAPPING IS THE WHOLE POINT ───────────────────────────────────
    // rt_covariance is a 6x6 ROW-MAJOR **SE3** covariance ordered [x, y, z, rx, ry, rz]
    // (RT_COVARIANCE_BLOCK_SIZE = 36 in cortex's dsr_rt_api.cpp). This used to write the SE(2) 3x3
    // into the top-left 3x3, i.e. treat the block as a PADDED 3x3 — which puts the heading variance
    // at (2,2), the slot that means var_Z, and leaves yaw at (5,5) permanently ZERO.
    //
    // ★ MEASURED CONSEQUENCE, 2026-08-16. The controller reads yaw from (5,5) — its own comment
    // records fixing that from (2,2) precisely because "the rotation throttle was being driven by the
    // robot's HEIGHT uncertainty". So it now reads a slot this agent never wrote: over two complete
    // tours, 20817 moving samples, `pose_theta_std_rad` in the controller's profile.csv is
    // IDENTICALLY 0.0000 — median, p75, p90, p95 and max alike. Its rotation throttle has been inert,
    // and anything else consuming yaw uncertainty from this edge has been reading a zero.
    // ★ Note how it arose: the consumer was corrected to cortex's convention and the producer was not,
    // so a one-sided fix turned an accidentally-consistent pair into a silently-broken one. Before it,
    // both used (2,2) and the wrong slot cancelled out.
    //
    // Index map: SE(2) (x, y, theta) -> SE3 (x, y, yaw) = rows/cols 0, 1, 5. Cross terms come along,
    // so the xy<->theta correlation the preintegrated motion model produces survives the transport
    // instead of being dropped on the floor.
    static constexpr int se3_of_se2[3] = {0, 1, 5};
    std::vector<float> cov_flat(36, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov_flat[se3_of_se2[r] * 6 + se3_of_se2[c]] = cov_se2(r, c);

    // ── Body-frame twist, written BEFORE the RT block ───────────────────────
    // Consumers read velocity DIRECTLY from here instead of differentiating the pose (which is what
    // produced correction-induced velocity spikes). Layout: rt_translation_velocity = [adv, side, 0],
    // rt_rotation_euler_xyz_velocity = [0,0,rot]; last_adv_/side_/rot_ are the measured odometry set
    // in update(). Covariance is a config diagonal.
    //
    // ★★ THAT IS ARRAY ORDER, NOT FRAME AXES. This comment used to read "[adv(fwd,+x), side(lat,+y)]"
    // and the parentheticals were WRONG: the robot body frame these rates live in is **+Y FORWARD,
    // +X lateral**. A consumer composing them into a transform must therefore put `adv` on the frame's
    // y axis and `side` on its x. Getting that backwards rotates the predicted displacement by 90°, so
    // it lands √2·|motion| from the truth — WORSE than assuming the robot never moved, which is how it
    // was found (controller twist compensation, 2026-08-04). Measured on 421 logged forward-driving
    // cycles, predicting the next pose from the true body twist: adv→x p50 25.86 mm, adv→y p50 0.07 mm.
    // The same frame is stated by the controller's safety-gate integrator (`x += adv*sin(th);
    // y += adv*cos(th)`) and by its carrot bearing being atan2(x, y) rather than atan2(y, x).
    //
    // ★ ORDER IS LOAD-BEARING. This used to run AFTER insert_or_assign_edge_RT, as a get_edge →
    // add attributes → insert_or_assign_edge(copy) write-back. That copy is a snapshot of the edge
    // INCLUDING its rt_timestamps / rt_head_index ring buffer, so assigning it back re-publishes the
    // ring state as of the fetch — the edge-level form of the write-back hazard already documented on
    // nodes. The write site's own comment records that an earlier version of this froze the position
    // outright and was patched by re-fetching; re-fetching narrows the window but does not close it.
    //
    // Measured symptom: room_concept publishes at ~19.6 Hz (pose_trace inter-publish p50 51 ms, 3.3%
    // over 100 ms, 0% over 200 ms) while the controller observes the RT stamp advancing at only 9.0 Hz
    // (p50 100 ms, 47% over 100 ms, 10% over 200 ms) — a factor of 2.2, i.e. about every second update
    // invisible. Writing the twist FIRST and letting the timestamped RT write land LAST means the
    // ring-buffer advance is the final word on this edge for this frame and nothing can write over it.
    if (auto edge = G_->get_edge(parent_opt.value().id(), child_id, "RT"); edge.has_value())
    {
        G_->add_or_modify_attrib_local<rt_translation_velocity_att>(
            edge.value(), std::vector<float>{last_adv_, last_side_, 0.f});
        G_->add_or_modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(
            edge.value(), std::vector<float>{0.f, 0.f, last_rot_});
        // 6×6 row-major SE3 [x,y,z,rx,ry,rz], same convention as the pose covariance above — the
        // velocity twist beside it is already written that way (rt_rotation_euler_xyz_velocity puts
        // yaw rate in the THIRD slot of an xyz triple, i.e. rz). The yaw-rate variance therefore
        // belongs at (5,5) = [35], not at [14] = var_Z. The old comment claimed it matched the pose
        // covariance and it did — both were wrong in the same way, which is exactly what made the
        // pair look self-consistent.
        std::vector<float> vel_cov(36, 0.f);
        if (params_)
        {
            vel_cov[0]  = params_->ROBOT_VEL_COV_ADV;    // (0,0) var_x
            vel_cov[7]  = params_->ROBOT_VEL_COV_SIDE;   // (1,1) var_y
            vel_cov[35] = params_->ROBOT_VEL_COV_ROT;    // (5,5) var_yaw — was [14] = var_Z
        }
        // Written raw rather than through RT_API::insert_or_assign_edge_RT_covariance so all three
        // twist attributes ride ONE get_edge/insert_or_assign_edge round trip: the API call would
        // fetch and re-publish the edge a second time per odometry cycle, and every extra edge
        // write-back on this path is another chance to re-publish stale ring state (see above).
        G_->add_or_modify_attrib_local<rt_covariance_velocity_att>(edge.value(), vel_cov);
        G_->insert_or_assign_edge(edge.value());
    }

    // ── Timestamped RT write ────────────────────────────────────────────────
    // Write via the RT_API covariance+timestamp overload so the edge keeps a proper TIMESTAMPED
    // HISTORY ring buffer (rt_timestamps / rt_head_index). The previous code overwrote a single
    // untimestamped block, so consumers' InterpolatedRT could never interpolate to a requested
    // time — it always returned the latest pose (RTdelta=0), freezing the controller's lidar
    // overlay between the ~5 Hz pose publishes. res.timestamp_ms is the pose's validity time (the
    // localization stamp). DSR interpolates pose between blocks; it does NOT extrapolate the velocity
    // attrs (written below) — those are for consumers to read directly.
    try
    {
        rt_api_->insert_or_assign_edge_RT(parent_opt.value(), child_id,
                                          std::vector<float>{x, y, 0.f},
                                          std::vector<float>{0.f, 0.f, theta},
                                          cov_flat,
                                          timestamp_ms);
    }
    catch (const std::exception &error)
    {
        qWarning() << "dsr_update_pose: insert_or_assign_edge_RT failed:" << error.what();
        log_cached_ids("insert_or_assign_edge_RT failed", child_id);
        if (room_node_created_ && child_id == dsr_room_id_)
        {
            room_node_created_ = false;
            dsr_room_id_ = 0;
            affordance_manager_.reset();
            stable_frames_ = 0;
        }
        return;
    }

}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res)
{
    if (!G_) return;

    const auto room_polygon = room_concept_->nominal_room_polygon();
    std::vector<float> polygon_x;
    std::vector<float> polygon_y;
    polygon_x.reserve(room_polygon.size());
    polygon_y.reserve(room_polygon.size());
    for (const auto& vertex : room_polygon)
    {
        polygon_x.push_back(vertex.x());
        polygon_y.push_back(vertex.y());
    }

    if (const auto room_nodes = G_->get_nodes_by_type("room"); !room_nodes.empty())
    {
        dsr_room_id_ = room_nodes.front().id();
        room_node_created_ = true;
        stable_frames_ = 0;
        // Idempotent polygon write: a room node ADOPTED here (persisted from a prior session, or created bare by
        // another agent) may LACK the delimiting polygon — it is only set on the create path below. Every
        // consumer's room-containment prior (chair_concept, cabinet_concept) then silently goes inert. So if the
        // adopted node has no usable polygon, author it now from the nominal model. Mirrors the create-path write.
        if (auto adopted = G_->get_node(dsr_room_id_); adopted.has_value() and polygon_x.size() >= 3)
        {
            auto& rn = adopted.value();
            const auto px = G_->get_attrib_by_name<delimiting_polygon_x_att>(rn);
            const bool has_poly = px.has_value() and px->get().size() >= 3;
            if (not has_poly)
            {
                rn.attrs()[delimiting_polygon_x_str.data()] = DSR::Attribute{polygon_x, 0, 0};
                rn.attrs()[delimiting_polygon_y_str.data()] = DSR::Attribute{polygon_y, 0, 0};
                G_->update_node(rn);
                qWarning() << "RoomSceneGraph: authored missing delimiting_polygon on adopted room node"
                           << dsr_room_id_ << "(" << static_cast<int>(polygon_x.size()) << "verts)";
            }
        }
        dsr_update_pose(res);
        return;
    }

    DSR::Node room_node = DSR::Node::create<room_node_type>("room");
    room_node.attrs()[delimiting_polygon_x_str.data()] = DSR::Attribute{polygon_x, 0, 0};
    room_node.attrs()[delimiting_polygon_y_str.data()] = DSR::Attribute{polygon_y, 0, 0};
    room_node.attrs()[room_height_str.data()] = DSR::Attribute{params_->room_height, 0, 0};
    // TODO: change to add_or_modify_attrib_local once available

    rc::provenance::stamp_creation(*G_, room_node);   // birth stamp: epoch ms + local ISO-8601
    const auto room_id_opt = G_->insert_node(room_node);
    if (!room_id_opt.has_value())
    {
        qWarning() << "DSR: failed to create room node";
        return;
    }

    dsr_room_id_ = room_id_opt.value();
    room_node_created_ = true;
    stable_frames_ = 0;
    trigger_layout_();

    dsr_update_pose(res);
    dsr_create_wall_nodes();

    // Seed the epistemic planner with room geometry so it can generate candidates.
    if (!room_polygon.empty() && epistemic_)
    {
        Eigen::Vector2f pmin = room_polygon.front();
        Eigen::Vector2f pmax = room_polygon.front();
        for (const auto& v : room_polygon)
        {
            pmin = pmin.cwiseMin(v);
            pmax = pmax.cwiseMax(v);
        }
        epistemic_->set_room_bounds(pmin, pmax);
        epistemic_->set_room_polygon(room_polygon);
    }
}

///////////////////////////////////////////////////////////////////////////////
// ★ SIGNAL AS THE WAKE-UP, LEVEL AS THE TRUTH. The graph tells us the instant epistemic_pending or
// aff_outcome changes on our own affordance node, so a consumer that claims and finishes inside one of
// its cycles can no longer slip between two of our polls. We still only LATCH here and let the compute
// loop consume it: DSR updates can coalesce under CRDT churn, so an edge-only design would be racy in
// the other direction — monitor_execution re-reads the node and decides from the level.
void RoomSceneGraph::on_affordance_attr_changed(std::uint64_t node_id)
{
    if (G_ == nullptr or node_id == 0) return;
    if (node_id == affordance_manager_.managed_node_id()) affordance_manager_.monitor_execution(G_);
    // ★THE SECOND NODE NEEDS THE SAME WAKE-UP. Without this the calibration channel would poll only,
    // and a consumer that claims and finishes inside one of our cycles slips between two polls —
    // which is the exact race that wedged the first pair.
    else if (node_id == calib_manager_.managed_node_id()) calib_manager_.monitor_execution(G_);
}

///////////////////////////////////////////////////////////////////////////////
// afford_calib — the calibration pivot, offered as a sequence of ordinary Orient affordances.
//
// ★THE PASSIVE HALF RUNS UNCONDITIONALLY. Every tour the robot makes serving a standpoint turns, and
// the estimator does not care WHY the robot moved, so ordinary exploration is free calibration data.
// That half commands nothing; it exists so the agent can answer, at any moment, "how well does this
// robot currently know its own motion model" — and so the deliberate manoeuvre can be priced against
// the free data rather than offered blind.
//
// ★THE ACTIVE HALF IS OFF UNLESS ASKED FOR, and even then it is not triggered by a threshold. The
// offer carries its expected information gain in nats and competes in the controller's EFE selection
// with every exploration cell; it wins only when knowing the motion model better is worth more than
// the next thing the robot could look at, and it extinguishes itself as the posterior sharpens
// because the quantity it advertises genuinely falls to zero. Measured in the unit test: 3.46 nats
// cold, 0.01 nats after 400 s of ordinary turning, 3.70 nats after 400 s of driving straight.
void RoomSceneGraph::dsr_update_calibration(const rc::RoomConcept::UpdateResult& res)
{
    if (G_ == nullptr or not room_node_created_ or params_ == nullptr or room_concept_ == nullptr)
        return;

    // ── 1. WATCH THE MOTION THE ROBOT IS ALREADY MAKING ─────────────────────────────────────────
    const double t_s = static_cast<double>(res.timestamp_ms) * 1e-3;
    const auto &odo = room_concept_->last_measured_prior();
    const auto &R = res.robot_pose.linear();
    const double theta = std::atan2(static_cast<double>(R(1, 0)), static_cast<double>(R(0, 0)));
    calib_.note_motion(t_s, res.robot_pose.translation().x(), res.robot_pose.translation().y(),
                       theta, static_cast<double>(odo.delta_pose.z()),
                       odo.valid and odo.fresh and odo.is_measured);
    calib_.note_robot_pos(res.robot_pose.translation().x(), res.robot_pose.translation().y());
    calib_last_t_s_ = t_s;

    // A throttled readout of what the passive half believes, so a run that never offers the manoeuvre
    // still ends knowing whether it should have. This is the whole product of the passive half.
    if (++calib_dbg_ % 1200 == 0)
    {
        const auto post = calib_.posterior();
        std::print("[calib] rotation scale {:+.5f} +/- {:.5f} ({} windows, {} discarded, density "
                   "{:.5f} rad/sqrt(s), {}) | diet {:.3f} rad/s | a pivot would be worth {:.3f} nats\n",
                   post.s, post.s_std, post.windows, calib_.poisoned_windows(), post.sigma,
                   post.identifiable() ? "MEASURED" : "still the prior",
                   calib_.passive_rate_rad_s(), calib_.marginal_gain_nats());
        std::fflush(stdout);
    }

    if (not params_->CALIB_PIVOT_ENABLED) return;

    // ── 2. DID THE CONSUMER ANSWER THE STANDING OFFER? ──────────────────────────────────────────
    calib_manager_.monitor_execution(G_);
    if (calib_manager_.consume_completion_event())
    {
        const auto outcome = calib_manager_.last_outcome();
        calib_.on_outcome(outcome, theta);
        const auto cl = calib_.closure();
        std::print("[calib] pivot step -> {} | {} step(s), {:.1f} deg accumulated\n",
                   rc::affordance::to_string(outcome), calib_.pivot().steps_issued(),
                   calib_.pivot().accumulated_rad() * 180.0 / M_PI);
        if (cl.usable)
            std::print("[calib] ★CLOSURE: odometry accumulated {:.1f} deg against {:.1f} deg of TRUTH "
                       "=> s_omega {:+.4f} ({:+.2f}%), resolved to {:.2f}%. No map, no survey, no "
                       "localiser in that number.\n",
                       cl.turned_rad * 180.0 / M_PI, cl.truth_rad * 180.0 / M_PI, cl.s_omega,
                       cl.s_omega * 100.0, cl.resolution * 100.0);
        else if (calib_.pivot().state() == rc::calib::PivotAffordance::State::Closed)
            std::print("[calib] the pivot closed but the scale it measured ({:+.4f}) is finer than "
                       "the closure resolves ({:.2f}%) — not a measurement, and not quoted as one.\n",
                       cl.s_omega, cl.resolution * 100.0);
        std::fflush(stdout);
        return;
    }
    if (calib_manager_.is_executing(G_)) return;   // the consumer owns it; do not rewrite the offer

    // ── 3. IS THERE A STEP TO OFFER? ────────────────────────────────────────────────────────────
    const auto bearing = calib_.offer(theta);
    if (not bearing.has_value()) return;
    calib_bearing_rad_ = static_cast<float>(*bearing);

    // ★THE CONTRACT IS AUTHORED ONCE, ON THE NODE, AT BIRTH. Orient with NO completion predicate:
    // "rotate in place to this bearing" is the whole affordance, and the executor completes it on
    // reaching the bearing.
    // ★★★THE TIMEOUT IS NOT DERIVED FROM THE TURN RATE, BECAUSE THE PRODUCER DOES NOT KNOW IT. The
    // first version computed 2 x (120 deg / 0.12 rad/s) = 35 s, taking 0.12 from the code default —
    // and the live config sets LockOnMaxYawRps = 0.06, at which one step takes 35 s exactly. Every
    // step would have timed out at the instant it completed, and the pivot would never have advanced
    // once. A number derived from an assumption about the other agent is not derived, it is guessed
    // with extra steps.
    // ★So state the assumption instead, at the only level the producer can honestly make it: a base
    // that cannot turn 120 degrees within two minutes is failing in a way worth reporting. Being
    // generous costs nothing here — the sequence advances on MEASURED heading, never on the steps it
    // issued, so a step that times out simply leaves the pivot where it was.
    const auto write_calib_contract = [this]
    {
        if (calib_contract_written_) return;
        auto n = G_->get_node(calib_manager_.managed_node_id());
        if (not n.has_value()) return;
        constexpr float kStepPatienceS = 120.0f;
        rc::affordance::write_contract(*G_, n.value(),
            rc::affordance::Contract::orient().stable(2).timeout_s(kStepPatienceS));
        G_->update_node(n.value());
        calib_contract_written_ = true;
        std::print("[calib] afford_calib authored: orient, no predicate — the rotation IS the goal; "
                   "{:.0f}s per step, which is patience rather than a prediction (this side cannot "
                   "know the consumer's turn rate)\n", kStepPatienceS);
    };

    const bool published = calib_manager_.publish_target(
        G_, dsr_room_id_,
        res.robot_pose.translation().x(),        // its own pose: an orient does not navigate, and
        res.robot_pose.translation().y(),        // publishing somewhere else would be a claim we cannot make
        calib_bearing_rad_,
        static_cast<float>(calib_.marginal_gain_nats()),
        [this, write_calib_contract]() { write_calib_contract(); trigger_layout_(); },
        [this]() { trigger_layout_(); });
    write_calib_contract();                      // also on the re-arm path, where no node is inserted

    if (published)
    {
        calib_armed_at_ms_ = static_cast<std::uint64_t>(res.timestamp_ms);
        std::print("[calib] offering step {} of {}: turn to {:.0f} deg, worth {:.3f} nats\n",
                   calib_.pivot().steps_issued() + 1, 3 * 4 /* turns x steps-per-turn */,
                   calib_bearing_rad_ * 180.0 / M_PI, calib_.marginal_gain_nats());
        std::fflush(stdout);
    }
}

void RoomSceneGraph::dsr_update_affordance(const rc::RoomConcept::UpdateResult& res)
{
    if (!G_ || !room_node_created_ || !epistemic_) return;

    auto& planner = epistemic_->epistemic_planner();

    // Always update robot state so mark_and_refresh uses the correct position.
    epistemic_->set_robot_state(res.robot_pose, res.covariance);

    // Mirror the exploration DRIVE into the localiser CSV every cycle, whether or not anything
    // completed. The drive is a continuous quantity (the scoring prior's precision decaying as
    // exp(-age/belief_forget_time)) and the completions are rare events punctuating it, so the only
    // way to see whether gating refresh_belief on Satisfied actually keeps the drive alive is to have
    // BOTH on the same time axis. Logging only at completion would sample the curve exactly where it
    // is discontinuous and nowhere else.
    if (room_concept_ != nullptr)
    {
        const auto tgt = planner.current_target();
        room_concept_->note_exploration_drive(planner.belief_age_s(), planner.belief_decay(),
                                              last_outcome_code_, aff_completions_,
                                              tgt ? tgt->position.x() : std::numeric_limits<float>::quiet_NaN(),
                                              tgt ? tgt->position.y() : std::numeric_limits<float>::quiet_NaN(),
                                              pub_tx_, pub_ty_, pub_ok_);
    }

    if (affordance_manager_.consume_completion_event())
    {
        // ★ COMPLETED IS NOT NEUTRAL. refresh_belief() resets ONE GLOBAL clock, and that clock feeds
        // exactly one thing: the exponential decay of the scoring prior's precision,
        // Y_info *= exp(-dt / belief_forget_time) (epistemic_planner.cpp). It is this agent asserting
        // "I have just explored, so my pose belief is sharp — stop inflating it". That is only true if
        // the look SUCCEEDED. A contract that ran to timeout arrives at the standpoint and completes
        // identically, and refreshing on it lets an attempt that observed NOTHING switch the
        // exploration drive back off.
        // ★ IT DOES NOT STEER THE ROBOT. Which cell comes next is the visit grid's job
        // (mark_target_finished → IoR), which de-prioritises the just-attempted cell on BOTH outcomes
        // by design — otherwise the planner re-proposes the cell the robot is standing on — and lets
        // it back in after ior_decay_time either way. Do not read this gate as "go back and retry
        // there"; it only keeps the drive alive instead of having it reset by a failure.
        // clear_target() runs either way — the affordance is over regardless of how it ended, and
        // holding a finished target would wedge the planner.
        const auto outcome = affordance_manager_.last_outcome();
        const bool observed = rc::affordance::observation_happened(outcome);
        using rc::affordance::Outcome;
        last_outcome_code_ = outcome == Outcome::Satisfied   ? 1
                           : outcome == Outcome::Timeout     ? 2
                           : outcome == Outcome::Refused     ? 3
                           : outcome == Outcome::Abandoned   ? 4
                           : outcome == Outcome::Infeasible  ? 5
                           : outcome == Outcome::Unreachable ? 6
                           : outcome == Outcome::OutsideRoom ? 7 : 0;
        ++aff_completions_;
        std::print("[planner] completion consumed (outcome={}) -> target cleared, {}\n",
                   rc::affordance::to_string(outcome),
                   observed ? "belief refreshed" : "belief NOT refreshed (nothing was observed)");
        std::fflush(stdout);
        // ★ A REFUSAL MUST CHANGE THE NEXT CHOICE, or it is only a label. The consumer refuses when it
        // was ALREADY standing on the proposed standpoint — no approach, nothing observed — and if we
        // merely clear the target, the same cell wins the next selection and the pair loops: offer,
        // instant refusal, offer. mark_target_finished() is the operator that exists for this: it
        // stamps the cell into the visit grid so its neglect drops to ~0 and recovers over
        // ior_decay_time, so the cell is de-prioritised for a while and then genuinely retried.
        // ★No blacklist and no threshold: a cell refused only because the robot happened to be parked
        // on it comes back into play on its own once the robot has moved on.
        // ★ GUARD ON HAVING A TARGET, NOT ON pub_ok_. pub_ok_ answers "did the LAST publish call
        // re-arm the node", and once the node IS armed every subsequent publish correctly declines as
        // a no-op — so pub_ok_ reads false on ~97% of cycles, including the one where the refusal
        // arrives. Guarding on it skipped the de-prioritisation exactly when it was needed: the
        // planner kept re-choosing the refused cell, publish_target declined it as unchanged, nothing
        // was ever Offered again, and the refusal protocol's second half — "so the producer sends a
        // NEW one" — never happened. The condition that matters is simply that we know which cell was
        // refused.
        // ★KEYED ON THE ARMED CELL. pub_tx_/pub_ty_ move on every publish ATTEMPT, declined ones
        // included, so by the time a refusal is consumed they may name a cell the consumer was never
        // offered — de-prioritising the wrong one and leaving the refused cell at full score.
        // (5535c6f's point survives: the guard is on KNOWING the cell, never on pub_ok_.)
        // ★★★A FACT ABOUT THE APPROACH IS NOT A FACT ABOUT THE WORLD. Infeasible ("the body does not
        // fit at that pose") and Unreachable ("no route from where I am") are the consumer's own
        // measurements, added 2026-08-19 to replace the silent standpoint substitution that reported
        // them as SATISFIED. They are handled exactly as a refusal here, and that is deliberate: the
        // planner wants the same thing from all three — stop proposing this cell for a while — and
        // NOTHING WAS OBSERVED in any of them, so `observed` stays false and refresh_belief() is not
        // called. The cell keeps its neglect: it is still unexplored, because it still is.
        // ★The de-prioritisation is note_attempt/attempt_suppressor — a decaying SCORE term, never a
        // stamp in the visit grid and never a blacklist — so a cell that was unreachable only because
        // a door was shut or the robot was parked badly returns on its own once the suppressor decays.
        // ★A cost, not an information update: this is rule 5 of the protocol design. A reachability
        // failure may change what is cheap to look at; it must never change what is believed to be seen.
        const bool approach_failed = outcome == Outcome::Refused
                                  or outcome == Outcome::Infeasible
                                  or outcome == Outcome::Unreachable
                                  or outcome == Outcome::OutsideRoom;
        // ★OUTSIDE_ROOM IS OUR OWN BUG, AND IT SAYS SO. The consumer is reporting that this agent
        // proposed a standpoint that is not inside the room polygon this agent published. It is
        // handled like the others so the run continues, but it is worth its own line: no amount of
        // driving fixes a cell that does not exist in the layout.
        if (outcome == Outcome::OutsideRoom)
            std::print("[planner] ★the consumer says ({:.2f},{:.2f}) is OUTSIDE the room layout — "
                       "that cell should never have been offered; check the exploration grid extent "
                       "against the room polygon\n", armed_tx_, armed_ty_);
        if (approach_failed and not std::isnan(armed_tx_) and not std::isnan(armed_ty_))
        {
            planner.mark_target_finished(Eigen::Vector2f(armed_tx_, armed_ty_));
            std::print("[planner] {} at ({:.2f},{:.2f}) — nothing observed there; cell de-prioritised "
                       "(decaying attempt suppressor, neglect untouched), selecting elsewhere\n",
                       rc::affordance::to_string(outcome), armed_tx_, armed_ty_);
        }
        planner.clear_target();
        planner.mark_and_refresh();   // keep path trail live in viewer
        if (observed)
            planner.refresh_belief(); // just finished exploring → belief fresh (restart forget clock)
        return;
    }

    if (affordance_manager_.is_executing(G_))
    {
        planner.mark_and_refresh();   // stamp path + refresh IoR overlay during navigation
        planner.refresh_belief();     // actively exploring → hold belief fresh; forgetting only when idle

        // Liveness: the planner is idle for as long as the claim is held, so a controller that can
        // never reach this target would park the whole run here. Break the claim if the robot stops
        // closing on it. Falls through to a fresh selection on the same cycle.
        if (!break_execution_stall(planner.robot_pos()))
        {
            // Throttled progress report while the controller owns the claim. Counts THIS execution
            // episode only (reset below when no claim is held) and reports the quantities the
            // watchdog actually decides on — current distance, best approach so far, and seconds
            // since that best improved. A previous version used a never-reset static cycle counter,
            // which made a perfectly normal drive look identical to a deadlock.
            if (++exec_hold_cycles_ % 90 == 0)
            {
                const auto& t = planner.current_target();
                const float dist = t ? (t->position - planner.robot_pos()).norm() : -1.f;
                const float idle_s = stall_tracking_
                    ? std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - stall_last_progress_).count()
                    : 0.f;
                std::print("[planner] afford_room EXECUTING (controller-claimed) — target "
                           "({:.2f},{:.2f}) d={:.2f}m best={:.2f}m no_progress={:.1f}s/{:.0f}s "
                           "episode_cycles={}\n",
                           t ? t->position.x() : 0.f, t ? t->position.y() : 0.f,
                           dist, stall_best_dist_, idle_s, params_->EXEC_STALL_TIMEOUT_S,
                           exec_hold_cycles_);
                std::fflush(stdout);
            }
            return;
        }
        exec_hold_cycles_ = 0;
    }
    else
    {
        stall_tracking_   = false;   // no claim in flight → nothing to watch
        exec_hold_cycles_ = 0;       // per-episode counter, so a normal drive can't look like a stall

        // ---- LEVEL-TRIGGERED completion ------------------------------------------------------
        // consume_completion_event() above is EDGE-triggered off monitor_affordance(), which only
        // polls every 200 ms — miss the Executing→Completed transition once and the planner keeps
        // its target forever. That is not a benign miss: with the target still set, update_target()
        // never reaches select_target(), so the planner re-publishes the SAME target, and since the
        // executor's goal tolerance is looser than our arrival_distance the robot is already inside
        // it — the affordance is claimed and completed instantly, on the spot, over and over, and
        // the robot never moves again. (Symptom: zero [planner] selection lines and a stream of
        // "reached -> REACH (consume immediately)" from the consumer.)
        //
        // Reading the Completed state directly is immune to the poll race: the executor's
        // completion is authoritative about arrival, so retire the target on it and let the next
        // cycle select a genuinely new one. Do NOT gate this on our own arrival_distance — the
        // whole failure is that the two thresholds disagree.
        if (const auto n = G_->get_node("afford_room"); n.has_value() and planner.current_target())
        {
            const bool a = G_->get_attrib_by_name<active_att>(n.value()).value_or(false);
            const bool p = G_->get_attrib_by_name<epistemic_pending_att>(n.value()).value_or(true);
            // ★★★RULE 3: EVERY CLAIM IS A LEASE, AND A LEASE DOES NOT ASK PERMISSION.
            // `break_execution_stall` already watches a claimed affordance — but it needs the planner
            // to hold a target and it measures approach PROGRESS, so it cannot fire when the consumer
            // has dropped its target and plan while still holding the claim. That is the measured
            // failure: 12% of live records read `Executing` with the consumer holding no target at
            // all, and the producer had no escape whatsoever. TLA+ agrees — with a stuttering consumer
            // `ProducerLive` is VIOLATED even with the unclaimed-offer timeout, because that only
            // covers Offered-and-unclaimed. An independent review named it before seeing either.
            // ★So: an unconditional cap on how long a claim may be held, on OUR clock, regardless of
            // what the consumer's internal state is or whether we can see it. Generous — well above
            // any real approach and above EXEC_STALL_TIMEOUT_S, so it is a backstop for the case the
            // progress watchdog structurally cannot see, never a competitor to it.
            constexpr std::uint64_t kExecutionLeaseMs = 45000;
            if (a and p and armed_at_ms_ != 0)
            {
                const auto now_ms = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                if (now_ms - armed_at_ms_ > kExecutionLeaseMs)
                {
                    std::print("[planner] afford_room claim held {:.1f}s without completing — LEASE "
                               "EXPIRED, reclaiming ({:.2f},{:.2f}) and selecting elsewhere\n",
                               (now_ms - armed_at_ms_) / 1000.f, armed_tx_, armed_ty_);
                    std::fflush(stdout);
                    armed_at_ms_ = 0;
                    if (planner.current_target())
                        planner.mark_target_finished(planner.current_target()->position);
                }
            }

            // ★UNCLAIMED-OFFER TIMEOUT. `a` is the consumer's claim; if the node is still merely pending
            // after this long, nobody is coming. Retire it and let select_target choose elsewhere —
            // otherwise room waits on a consumer that has already declined, for up to 100 s (measured).
            constexpr std::uint64_t kOfferUnclaimedMs = 5000;
            if (p and not a and armed_at_ms_ != 0)
            {
                const auto now_ms = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                if (now_ms - armed_at_ms_ > kOfferUnclaimedMs and planner.current_target())
                {
                    std::print("[planner] afford_room OFFERED for {:.1f}s and never claimed — retiring "
                               "({:.2f},{:.2f}) and selecting elsewhere\n",
                               (now_ms - armed_at_ms_) / 1000.f, armed_tx_, armed_ty_);
                    std::fflush(stdout);
                    armed_at_ms_ = 0;
                    planner.mark_target_finished(planner.current_target()->position);
                }
            }
            if (a or p) armed_seen_live_ = true;   // the arming is real: a Completed reading now means it
            if (not a and not p and armed_seen_live_)   // Completed, and we saw it live first
            {
                const auto done = planner.current_target()->position;
                // ★A COMPLETION IS EVIDENCE ABOUT THE CELL THAT WAS ARMED. This branch exists to catch
                // a completion whose edge the 200 ms poll missed, but it reads a LEVEL, and the level
                // says only "not pending" — it stays true until something re-arms. Retiring whatever
                // the planner happens to hold on that basis charged a completion to a cell that was
                // never offered, once per cycle. Measured 2026-08-19: (-1.50,-3.38) retired at ~20 Hz
                // with the robot 1.32 m away.
                // ★⚠DO NOT ADD AN `else clear_target()` HERE. I tried exactly that and it re-selected
                // every cycle from candidates separated by less than the noise, flipping the published
                // cell ~10x/s — 373 distinct cells in 97 s, far worse than the loop it replaced.
                // Holding the target is correct; the publish path re-arms it when it differs.
                const bool is_armed_target =
                    not std::isnan(armed_tx_) and not std::isnan(armed_ty_)
                    and (done - Eigen::Vector2f(armed_tx_, armed_ty_)).norm() < 0.05f;
                if (is_armed_target and not armed_retired_)
                {
                    armed_retired_ = true;
                    std::print("[planner] afford_room COMPLETED (level-triggered) — retiring target "
                               "({:.2f},{:.2f}) at d={:.2f}m; re-selecting\n",
                               done.x(), done.y(), (done - planner.robot_pos()).norm());
                    std::fflush(stdout);
                    planner.mark_target_finished(done);
                    planner.refresh_belief();
                }
            }
        }
    }

    // Refresh obstacle exclusion zones from DSR graph before selecting the target.
    update_planner_obstacle_footprints();

    // Refresh the localizer's object anchors (validated objects → SE(2) pose landmarks).
    refresh_object_anchors();

    // Ask the planner for the current best target (handles dwell / arrival internally)
    const auto target_opt = planner.update_target();

    if (not target_opt.has_value())
    {
        // update_target() declined to produce one. Silent until now, which made this
        // indistinguishable from "the publish path is not running at all". Two ways to get here:
        // the planner is mid-dwell with a cleared target, or evaluate_targets bailed before the
        // STARVED diagnostic (room bounds / robot state not yet set).
        if (++no_target_dbg_ % 90 == 0)
            std::print("[planner] update_target() returned NOTHING — no affordance published "
                       "(dwell or unset room/robot state); {} cycles\n", no_target_dbg_);
        std::fflush(stdout);
        return;
    }
    no_target_dbg_ = 0;

    const float tx   = target_opt->position.x();
    const float ty   = target_opt->position.y();
    // Advertise the GROUNDED epistemic value in nats, so afford_room's gain is in the same currency
    // as an object concept's ΔH and the controller can compare them as one EFE term. NOT
    // target.score, which additionally folds in the distance tie-break used only for the room's own
    // internal ranking.
    //
    // The published quantity is MARGINAL pose information (what this vantage adds over the one the
    // robot already occupies) + the destination's neglect information. Marginal rather than
    // absolute because the absolute ΔH reads ~4 nats from anywhere in a room whose layout is a
    // fixed prior — it never falls, so it would advertise "there is something to learn here"
    // forever and permanently out-bid every object affordance. Recomputed LIVE every publish cycle
    // rather than frozen at selection time, so it tracks the pose precision as the robot drives in.
    // The neglect part is unbounded in neglect age, so the gain can never collapse and exploration
    // cannot stall. rotate_in_place switches to the absolute reading — see live_total_epistemic_gain.
    const float gain = planner.live_total_epistemic_gain(target_opt->position,
                                                         target_opt->rotate_in_place);

    // Heading: face toward room centre so the robot maximises wall/corner visibility.
    const float cx  = (planner.room_min().x() + planner.room_max().x()) * 0.5f;
    const float cy  = (planner.room_min().y() + planner.room_max().y()) * 0.5f;
    float yaw = std::atan2(cy - ty, cx - tx);

    // ---- Rotate-in-place (heading recovery) needs a yaw the robot does NOT already hold ----
    // This target's POSITION is the robot's own, so the only executable part of it is the yaw: the
    // consumer's arrival rotation turns in place until the heading matches, then completes. Aiming
    // it at the room centre is therefore a no-op whenever the robot already happens to face that
    // way — the consumer reports "reached" instantly, no rotation occurs, no heading information is
    // gathered, and the next cycle re-proposes a byte-identical target (same position, same yaw,
    // same gain). That identical re-proposal is fatal, not merely wasteful: publish_target refuses
    // to re-arm a Completed affordance whose target is unchanged, so the affordance stays Completed
    // forever and the consumer sits at "no eligible affordance" — the exploration stops dead.
    //
    // Publishing a substantial turn relative to the CURRENT heading fixes both halves: the rotation
    // actually happens (which is the entire point of the recovery), and the commanded yaw advances
    // every cycle, so successive recoveries sweep the full circle instead of colliding as "the same
    // target". kRecoveryTurnRad is a fraction of a full turn, large enough to be unambiguously a
    // real motion and to clear the consumer's heading tolerance.
    if (target_opt->rotate_in_place)
    {
        constexpr float kRecoveryTurnRad = 2.0f;   // ~115°, so three legs cover a full sweep
        yaw = std::atan2(std::sin(planner.robot_theta() + kRecoveryTurnRad),
                         std::cos(planner.robot_theta() + kRecoveryTurnRad));
    }

    const bool published = affordance_manager_.publish_target(
        G_,
        dsr_room_id_,
        tx,
        ty,
        yaw,
        gain,
        [this]() { trigger_layout_(); },
        [this]() { trigger_layout_(); });

    pub_tx_ = tx; pub_ty_ = ty; pub_ok_ = published;
    // A new arming is a new affordance instance: latch what is really on the node and re-open the
    // right to retire it exactly once. `published` is the only signal a consumer will ever see it.
    if (published)
    {
        armed_tx_ = tx; armed_ty_ = ty; armed_retired_ = false; armed_seen_live_ = false;
        armed_at_ms_ = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    // ---- Publish trace: the ONLY place that shows why exploration stops ----------------------
    // Everything upstream of here can look healthy while the affordance never returns to Offered:
    // publish_target silently declines when the node is Completed and the proposed target is
    // unchanged, and the planner path that re-proposes an unchanged target prints nothing (it never
    // reaches select_target, because current_target_ is still set). So log the decision itself:
    // the proposed target, whether publish_target accepted it, and the resulting protocol state read
    // back from the graph. Throttled, plus an immediate line whenever the state is not Offered so a
    // stuck Completed/Invalid shows up at once instead of up to 90 cycles later.
    bool active = false, pending = true;
    if (const auto n = G_->get_node("afford_room"); n.has_value())
    {
        active  = G_->get_attrib_by_name<active_att>(n.value()).value_or(false);
        pending = G_->get_attrib_by_name<epistemic_pending_att>(n.value()).value_or(true);
    }
    const char* state = (not active and pending) ? "Offered"
                      : (    active and pending) ? "Executing"
                      : (not active and not pending) ? "COMPLETED(stuck?)" : "INVALID";
    const bool offered = (not active and pending);
    if (++publish_dbg_ % 90 == 0 or not offered)
        std::print("[planner] publish target=({:.2f},{:.2f}) yaw={:.2f} gain={:.3f} rot_in_place={} "
                   "accepted={} -> afford_room state={}\n",
                   tx, ty, yaw, gain, target_opt->rotate_in_place, published, state);
    std::fflush(stdout);
}

///////////////////////////////////////////////////////////////////////////////
// break_execution_stall — see the header for why this watchdog is a threshold on purpose.
bool RoomSceneGraph::break_execution_stall(const Eigen::Vector2f& robot_pos)
{
    if (!epistemic_ || !params_ || params_->EXEC_STALL_TIMEOUT_S <= 0.f)
        return false;

    auto& planner = epistemic_->epistemic_planner();
    const auto  now = std::chrono::steady_clock::now();
    const auto& target_opt = planner.current_target();

    // A claim held while the planner has NO target of its own is already inconsistent: nothing is
    // being driven toward, so no amount of waiting can produce a completion. This is reachable right
    // after a completion is consumed (which clears the target) if the claim is not fully torn down,
    // and it is the nastier deadlock of the two because the distance-progress test below has nothing
    // to measure. Time it out on the same clock and release.
    if (!target_opt.has_value())
    {
        if (not stall_tracking_)
        {
            stall_tracking_      = true;
            stall_target_        = Eigen::Vector2f::Zero();
            stall_best_dist_     = std::numeric_limits<float>::infinity();
            stall_last_progress_ = now;
            return false;
        }
        if (std::chrono::duration<float>(now - stall_last_progress_).count()
            < params_->EXEC_STALL_TIMEOUT_S)
            return false;

        const bool released = affordance_manager_.release_execution_claim(G_);
        qWarning() << "[planner] afford_room claimed for" << params_->EXEC_STALL_TIMEOUT_S
                   << "s with NO planner target — inconsistent claim"
                   << (released ? "— execution claim released" : "— no claim to release")
                   << "; re-selecting";
        stall_tracking_ = false;
        return true;
    }

    // A rotate-in-place recovery does not close any distance by construction — it would trip the
    // watchdog every time. It is also self-limiting (it ends when the heading covariance drops).
    if (target_opt->rotate_in_place)
    {
        stall_tracking_ = false;
        return false;
    }

    const float dist = (target_opt->position - robot_pos).norm();

    // (Re)arm on a fresh claim or a target that moved.
    if (!stall_tracking_ || (target_opt->position - stall_target_).squaredNorm() > 1e-6f)
    {
        stall_tracking_      = true;
        stall_target_        = target_opt->position;
        stall_best_dist_     = dist;
        stall_last_progress_ = now;
        return false;
    }

    // Progress = a new closest approach. Requires a real improvement so pose noise alone cannot
    // keep resetting the clock (which would make the watchdog never fire — the failure it exists
    // to catch is precisely one where the robot jitters in place).
    if (dist < stall_best_dist_ - params_->EXEC_STALL_PROGRESS_M)
    {
        stall_best_dist_     = dist;
        stall_last_progress_ = now;
        return false;
    }

    const float idle_s = std::chrono::duration<float>(now - stall_last_progress_).count();
    if (idle_s < params_->EXEC_STALL_TIMEOUT_S)
        return false;

    // Give up on this target: release the controller's claim and fold the failure into the visit
    // grid, so the neglect drive that chose this cell reads ~0 there and the next selection goes
    // somewhere else. The cell is NOT blacklisted — its neglect recovers over IorDecayTime, so a
    // target that was only transiently blocked is retried later on its own.
    const bool released = affordance_manager_.release_execution_claim(G_);
    qWarning() << "[planner] afford_room target (" << target_opt->position.x() << ","
               << target_opt->position.y() << ") abandoned: no approach progress for"
               << idle_s << "s (closest" << stall_best_dist_ << "m)"
               << (released ? "— execution claim released" : "— no claim to release")
               << "; marking visited and re-selecting";

    planner.mark_target_finished(stall_target_);
    stall_tracking_ = false;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::update_planner_obstacle_footprints()
{
    if (!G_ || !rt_api_ || !room_node_created_ || !epistemic_) return;

    std::vector<rc::EpistemicPlanner::ObstacleFootprint> footprints;

    auto collect = [&](const std::string& node_type)
    {
        for (const auto& node : G_->get_nodes_by_type(node_type))
        {
            const auto w_opt = G_->get_attrib_by_name<width_m_att>(node);
            const auto d_opt = G_->get_attrib_by_name<depth_m_att>(node);
            if (!w_opt.has_value() || !d_opt.has_value()) continue;
            const float half_w = w_opt.value() * 0.5f;
            const float half_d = d_opt.value() * 0.5f;
            if (half_w <= 0.f || half_d <= 0.f) continue;

            const auto rt_opt = rt_api_->get_RT_pose_from_parent(node);
            if (!rt_opt.has_value()) continue;

            const Eigen::Vector3d t = rt_opt->translation();
            const float yaw = static_cast<float>(
                std::atan2(rt_opt->linear()(1, 0), rt_opt->linear()(0, 0)));

            footprints.push_back({
                .center = {static_cast<float>(t.x()), static_cast<float>(t.y())},
                .half_w = half_w,
                .half_d = half_d,
                .yaw    = yaw
            });
        }
    };

    collect("object");   // tables/chairs/bottles are now generic "object" nodes (class in object_subtype)
    collect("obstacle");

    epistemic_->epistemic_planner().set_obstacle_footprints(std::move(footprints));

}

///////////////////////////////////////////////////////////////////////////////
// Observe-only table-landmark logging (pure reads, no anchor/pin side effects). Tracks how the detected
// table coordinates evolve as the robot moves: the ROOM-frame pose (from the room→table RT) should stay
// STABLE if localization is consistent; the ROBOT-frame observation (obj_obs_robot, table_concept's raw
// per-frame detection) naturally moves with the robot. Robot pose logged for correlation.
void RoomSceneGraph::log_table_landmarks()
{
    // Robot pose in room frame (independent of the table) + localization timestamp.
    float rx = 0.f, ry = 0.f, rth = 0.f; std::uint64_t ts_ms = 0;
    if (const auto lr = room_concept_->get_last_result(); lr.has_value())
    {
        rx  = lr->robot_pose.translation().x();
        ry  = lr->robot_pose.translation().y();
        rth = std::atan2(lr->robot_pose.linear()(1, 0), lr->robot_pose.linear()(0, 0));
        ts_ms = static_cast<std::uint64_t>(lr->timestamp_ms);
    }

    const bool throttle_log = (table_landmark_log_k_++ % 10) == 0;   // stdout ~2/s; CSV gets every frame

    for (const auto& n : G_->get_nodes_by_type("object"))
    {
        if (not node_is_object_class(*G_, n, "table")) continue;   // furniture is now type "object"

        // ROBOT-frame detection (raw): obj_obs_robot = [x, y]. Absent ⇒ table not detected this frame.
        const auto obs = G_->get_attrib_by_name<obj_obs_robot_att>(n);
        const bool has_obs = obs.has_value() and obs->get().size() >= 2;
        const float ox = has_obs ? obs->get()[0] : std::numeric_limits<float>::quiet_NaN();
        const float oy = has_obs ? obs->get()[1] : std::numeric_limits<float>::quiet_NaN();

        // Anisotropic R_o published by the producer as obj_obs_robot_cov=[Rxx,Ryy,Rxy] (ray-loose): log its
        // anisotropy (σ_along²/σ_perp², large ⇒ working) and whether its loose axis aligns with the viewing
        // ray (|cos| between R_o's big eigenvector and obs_robot dir; ~1 ⇒ correct). See TABLE_TRIANGULATION.
        float cov_ratio = std::numeric_limits<float>::quiet_NaN();
        float ray_align = std::numeric_limits<float>::quiet_NaN();
        if (const auto cov = G_->get_attrib_by_name<obj_obs_robot_cov_att>(n);
            cov.has_value() and cov->get().size() >= 3 and has_obs)
        {
            const auto& c = cov->get();
            Eigen::Matrix2f R; R << c[0], c[2], c[2], c[1];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(R);
            const float lo = std::max(1e-9f, es.eigenvalues()(0)), hi = es.eigenvalues()(1);
            cov_ratio = hi / lo;
            const Eigen::Vector2f big = es.eigenvectors().col(1);
            const float on = std::hypot(ox, oy);
            if (on > 1e-6f)
                ray_align = std::abs((big.x() * ox + big.y() * oy) / on);
        }

        // ROOM-frame table pose from the RT edge (parent should be 'room'; log parent to catch frame bugs).
        float wx = std::numeric_limits<float>::quiet_NaN(), wy = wx, wyaw = wx;
        std::string parent_name = "?";
        if (const auto p = G_->get_attrib_by_name<parent_att>(n); p.has_value())
            if (const auto pn = G_->get_node(p.value()); pn.has_value())
                parent_name = pn->name();
        if (const auto rt = rt_api_->get_RT_pose_from_parent(n); rt.has_value())
        {
            const Eigen::Vector3d t = rt->translation();
            wx = static_cast<float>(t.x()); wy = static_cast<float>(t.y());
            wyaw = static_cast<float>(std::atan2(rt->linear()(1, 0), rt->linear()(0, 0)));
        }

        bool alive = true;
        int  age   = 0;   // table_frames_since_detection: 0 = fresh this cycle, grows while unseen
        {
            // TYPE-ATTRIBUTED reads (CLAUDE.md): checked against the REGISTER_TYPE in dsr_attr_name.h at
            // COMPILE time. The previous form looked the attribute up by string in the raw attrs() map and
            // called Attribute::dec(), which consults the declared type not at all — so when the producer
            // changed table_detection_alive from int to bool this threw "INT is not selected, selected is
            // BOOL" at runtime instead of failing the build here.
            if (const auto v = G_->get_attrib_by_name<table_detection_alive_att>(n); v.has_value())
                alive = v.value();
            if (const auto v = G_->get_attrib_by_name<table_frames_since_detection_att>(n); v.has_value())
                age = std::max(0, v.value());
        }
        // Freshness-as-precision weight actually applied to this obs by object_anchor_source:
        //   R_o ← R_o·(1+age/scale)²  ⇒  relative precision (weight) = 1/(1+age/scale)². 1=fresh, →0 stale.
        const float age_scale = std::max(1e-3f, params_->OBJECT_ANCHOR_FRESHNESS_AGE_SCALE);
        const float f = 1.0f + static_cast<float>(age) / age_scale;
        const float fresh_weight = 1.0f / (f * f);

        // "Detected" = has a live robot-frame observation. Skip nodes with no measurement this frame.
        if (not has_obs) continue;

        if (throttle_log)
            std::print("[table-landmark] #{} room=({:.3f},{:.3f},{:.2f}) obs_robot=({:.3f},{:.3f}) "
                       "robot=({:.2f},{:.2f},{:.2f}) alive={} age={} fresh_w={:.3f} cov_ratio={:.1f} "
                       "ray_align={:.2f} parent='{}'\n",
                       n.id(), wx, wy, wyaw, ox, oy, rx, ry, rth, alive, age, fresh_weight,
                       cov_ratio, ray_align, parent_name);

        if (!table_landmark_csv_open_attempted_)
        {
            table_landmark_csv_open_attempted_ = true;
            ::mkdir("tmp", 0755);
            ::mkdir("tmp/sdf_localizer", 0755);
            const std::time_t tt = std::time(nullptr);
            std::tm tm_local{}; localtime_r(&tt, &tm_local);
            char ts_buf[32]; std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
            table_landmark_csv_.open(std::string("tmp/sdf_localizer/table_landmark_") + ts_buf + ".csv",
                                     std::ios::out | std::ios::trunc);
            if (table_landmark_csv_.is_open())
                table_landmark_csv_ << "ts_ms,node_id,room_x,room_y,room_yaw,obs_robot_x,obs_robot_y,"
                                       "robot_x,robot_y,robot_th,alive,frames_since_detection,fresh_weight,"
                                       "cov_ratio,ray_align,parent\n";
        }
        if (table_landmark_csv_.is_open())
        {
            table_landmark_csv_ << ts_ms << ',' << n.id() << ','
                                << wx << ',' << wy << ',' << wyaw << ','
                                << ox << ',' << oy << ','
                                << rx << ',' << ry << ',' << rth << ','
                                << (alive ? 1 : 0) << ',' << age << ',' << fresh_weight << ','
                                << cov_ratio << ',' << ray_align << ',' << parent_name << '\n';
            table_landmark_csv_.flush();
        }
    }
}

// Gather validated modelled objects (tables, …) from the graph and hand them to the localizer
// as SE(2) pose landmarks. MAIN THREAD (get_RT_pose_from_parent uses the ts==0 InnerEigen cache).
void RoomSceneGraph::refresh_object_anchors()
{
    if (!G_ || !rt_api_ || !room_concept_ || !room_node_created_)
        return;

    // Observe-only table-landmark logging: pure graph reads, ZERO anchor/pin side effects, runs even when
    // the anchor FACTOR is disabled — so the detected table coordinates can be studied as the robot moves
    // (register-then-pull) before deciding to let the landmark pull the pose.
    log_table_landmarks();

    if (!params_->OBJECT_ANCHOR_ENABLE)
        return;

    if (!inner_gaussian_)                       // lazily create on the main thread (Qt event loop present)
        inner_gaussian_ = G_->get_inner_gaussian_api();

    rc::ObjectAnchorSource::Config cfg;
    cfg.enable             = params_->OBJECT_ANCHOR_ENABLE;
    cfg.subtypes           = params_->OBJECT_ANCHOR_SUBTYPES;
    cfg.optimize_landmark  = params_->OBJECT_ANCHOR_OPTIMIZE_LANDMARK;
    // A pin is FOREVER (for the life of the process): it snapshots the object's world pose and uses that
    // fixed value as p_o thereafter. Capturing one while the localizer is mid-flip or mid-delocalization
    // bakes the wrong place in, and the anchor then fights the correct pose for the rest of the run.
    // So new pins require the localizer to have been settled for STABLE_FRAMES_REQUIRED consecutive
    // frames. Existing pins are unaffected.
    cfg.allow_pin          = anchor_stable_frames_ >= params_->STABLE_FRAMES_REQUIRED;
    cfg.meas_sigma_xy      = params_->OBJECT_ANCHOR_MEAS_SIG_XY;
    cfg.meas_sigma_yaw     = params_->OBJECT_ANCHOR_MEAS_SIG_YAW;
    cfg.validate_sigma     = params_->OBJECT_ANCHOR_VALIDATE_SIGMA;
    cfg.freshness_enable   = params_->OBJECT_ANCHOR_FRESHNESS_ENABLE;
    cfg.freshness_age_scale= params_->OBJECT_ANCHOR_FRESHNESS_AGE_SCALE;

    auto anchors = object_anchor_source_.gather(*G_, *inner_gaussian_, cfg);

    // Diagnostic (~1/s): pinpoint where the chain breaks —
    //   tables=0            → room doesn't see the table node
    //   with_obs=0          → table_concept isn't publishing obj_obs_robot (producer side)
    //   used<with_obs       → reader skipped (no RT pose / gaussian transform)
    //   used>0 but loss=0   → factor/slot-copy issue on the localizer side
    static std::uint64_t anchor_dbg_k = 0;
    if ((anchor_dbg_k++ % 30) == 0)
    {
        int n_nodes = 0, with_obs = 0;
        for (const auto& n : G_->get_nodes_by_type("object"))
        {
            const bool admitted = std::ranges::any_of(cfg.subtypes,
                [&](const std::string& c) { return node_is_object_class(*G_, n, c); });
            if (not admitted) continue;
            ++n_nodes;
            if (const auto o = G_->get_attrib_by_name<obj_obs_robot_att>(n);
                o.has_value() and o->get().size() >= 2)
                ++with_obs;
        }
        std::string cls_list;
        for (const auto& c : cfg.subtypes) { if (not cls_list.empty()) cls_list += ','; cls_list += c; }
        std::print("[room][anchors] classes=[{}] nodes={} with_obj_obs_robot={} anchors_used={}\n",
                   cls_list, n_nodes, with_obs, anchors.size());
        if (not anchors.empty())
        {
            const auto& a = anchors.front();
            // Λdiag≈0 ⇒ map cov Σ_o came back huge (factor muted). p_o vs z_o wildly apart ⇒ frame bug;
            // near-equal (after the R(-θ) transform) ⇒ anchor satisfied (0 loss is then correct).
            std::print("[room][anchors]   a0 type={} p_o=({:.2f},{:.2f},{:.2f}) z_o=({:.2f},{:.2f}) "
                       "Λdiag=({:.2f},{:.2f}) has_yaw={}\n",
                       a.type, a.pose_world.x(), a.pose_world.y(), a.pose_world.z(),
                       a.obs_robot.x(), a.obs_robot.y(), a.information(0, 0), a.information(1, 1),
                       a.has_orientation);
        }
        // Cross-check: the table's RAW room→table RT + its actual parent frame. If parent!='room' or the
        // raw translation differs from p_o above, the map pose is being read in the wrong frame.
        for (const auto& n : G_->get_nodes_by_type("object"))
        {
            if (not node_is_object_class(*G_, n, "table")) continue;
            std::string parent_name = "?";
            if (const auto p = G_->get_attrib_by_name<parent_att>(n); p.has_value())
                if (const auto pn = G_->get_node(p.value()); pn.has_value())
                    parent_name = pn->name();
            if (const auto rt = rt_api_->get_RT_pose_from_parent(n); rt.has_value())
                std::print("[room][anchors]   RAW {} #{} parent='{}' t=({:.2f},{:.2f})\n",
                           n.name(), n.id(), parent_name,
                           rt->translation().x(), rt->translation().y());
            break;   // first table only
        }
    }

    // Cache the pinned landmark world positions (+ live "being measured" flag) for the viewer
    // (robot→landmark sight lines). "Measured" = the producer's per-frame detection flag; for tables
    // that is `table_detection_alive` (type-attributed read, compile-checked). Absent flag ⇒
    // default to measured, so an object type that doesn't publish it still shows its sight line.
    latest_pinned_landmarks_.clear();
    latest_pinned_measured_.clear();
    latest_pinned_landmarks_.reserve(anchors.size());
    latest_pinned_measured_.reserve(anchors.size());
    for (const auto& a : anchors)
    {
        latest_pinned_landmarks_.emplace_back(a.pose_world.x(), a.pose_world.y());
        bool measured = true;
        if (const auto n = G_->get_node(a.node_id); n.has_value())
        {
            // Type-attributed read — compile-time checked against dsr_attr_name.h (see note above).
            if (const auto v = G_->get_attrib_by_name<table_detection_alive_att>(n.value()); v.has_value())
                measured = v.value();
        }
        latest_pinned_measured_.push_back(measured ? 1 : 0);
    }

    room_concept_->set_object_anchors(std::move(anchors));
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::load_robot_body_dimensions_from_graph()
{
    if (!G_)
        return;

    std::optional<DSR::Node> body_node = std::nullopt;

    const auto body_nodes = G_->get_nodes_by_type("body");
    if (!body_nodes.empty())
        body_node = body_nodes.front();
    else
        body_node = G_->get_node("body");

    if (!body_node.has_value())
    {
        qWarning() << "dsr_init_graph: no 'body' node found; keeping default robot dimensions"
                   << params_->ROBOT_WIDTH << params_->ROBOT_LENGTH << params_->ROBOT_HEIGHT;
        return;
    }

    dsr_body_id_ = body_node->id();

    if (const auto width_value = G_->get_attrib_by_name<width_m_att>(body_node.value()); width_value.has_value())
        params_->ROBOT_WIDTH = width_value.value();
    if (const auto depth_value = G_->get_attrib_by_name<depth_m_att>(body_node.value()); depth_value.has_value())
        params_->ROBOT_LENGTH = depth_value.value();
    if (const auto height_value = G_->get_attrib_by_name<height_m_att>(body_node.value()); height_value.has_value())
        params_->ROBOT_HEIGHT = height_value.value();

    if (epistemic_)
        epistemic_->set_robot_footprint(params_->ROBOT_WIDTH, params_->ROBOT_LENGTH);

    qInfo() << "Robot dimensions from body node: width depth height ="
            << params_->ROBOT_WIDTH << params_->ROBOT_LENGTH << params_->ROBOT_HEIGHT;
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::dsr_create_wall_nodes()
{
    if (!G_ || !rt_api_) return;

    auto room_node_opt = G_->get_node(dsr_room_id_);
    if (!room_node_opt.has_value()) { qWarning() << "dsr_create_wall_nodes: room node missing"; return; }

    const auto polygon = room_concept_->nominal_room_polygon();
    const int n = static_cast<int>(polygon.size());
    if (n < 3) { qWarning() << "dsr_create_wall_nodes: polygon has fewer than 3 vertices"; return; }

    // Guard — idempotent: if wall nodes already exist under this room, skip.
    if (const auto existing = G_->get_nodes_by_type("wall"); !existing.empty())
    {
        // ...but walls and the floor persist in the shared graph, so ones created before the display
        // meshes existed would never acquire them and every viewer would silently keep falling back
        // to a generic box. Backfill here, exactly as the adopted room node's delimiting_polygon is
        // backfilled above.
        for (auto wall : existing)
            if (const auto mp = G_->get_attrib_by_name<mesh_path_att>(wall);
                !mp.has_value() || mp.value().empty())
            {
                G_->add_or_modify_attrib_local<mesh_path_att>(wall, std::string("room_concept/meshes/wall.obj"));
                G_->update_node(wall);
                qInfo() << "dsr_create_wall_nodes: backfilled mesh_path on existing"
                        << QString::fromStdString(wall.name());
            }
        // The floor mesh is REGENERATED unconditionally: it encodes the layout, so an adopted floor
        // may be carrying a mesh built from a different polygon than the one in force now.
        if (auto f = G_->get_node("floor"); f.has_value())
        {
            if (float fw = 0.f, fd = 0.f;
                write_floor_obj(polygon, std::filesystem::path(kFloorMeshOut), fw, fd))
            {
                G_->add_or_modify_attrib_local<mesh_path_att>(f.value(), std::string(kFloorMeshRel));
                G_->add_or_modify_attrib_local<width_m_att>(f.value(), fw);
                G_->add_or_modify_attrib_local<depth_m_att>(f.value(), fd);
                G_->update_node(f.value());
                qInfo() << "dsr_create_wall_nodes: regenerated floor display mesh"
                        << fw << "x" << fd << "m";
            }
        }
        return;
    }

    const float half_h = params_->room_height * 0.5f;

    // ── Walls ────────────────────────────────────────────────────────────────
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& p0 = polygon[i];
        const Eigen::Vector2f& p1 = polygon[(i + 1) % n];
        const float L = (p1 - p0).norm();
        if (L < 0.1f)
        {
            qWarning() << "dsr_create_wall_nodes: skipping degenerate wall" << i << "(length" << L << "m)";
            continue;
        }

        const Eigen::Vector2f dir = (p1 - p0) / L;
        const float yaw = std::atan2(dir.y(), dir.x());
        const Eigen::Vector2f mid = (p0 + p1) * 0.5f;

        DSR::Node wall_node = DSR::Node::create<wall_node_type>("wall_" + std::to_string(i));
        G_->add_or_modify_attrib_local<width_m_att>(wall_node, L);
        G_->add_or_modify_attrib_local<height_m_att>(wall_node, params_->room_height);
        // Display mesh, same contract the furniture concepts use: a 2-triangle panel authored to the
        // shared unit-box convention (local x ALONG the wall, z = height, y = the normal, no
        // thickness), which a viewer rescales by width_m/height_m. The agent owns its appearance.
        G_->add_or_modify_attrib_local<mesh_path_att>(wall_node, std::string("room_concept/meshes/wall.obj"));
        G_->add_or_modify_attrib_local<parent_att>(wall_node, dsr_room_id_);
        G_->add_or_modify_attrib_local<level_att>(wall_node, 4);

        rc::provenance::stamp_creation(*G_, wall_node);   // birth stamp: epoch ms + local ISO-8601
        const auto wall_id = G_->insert_node(wall_node);
        if (!wall_id.has_value())
        {
            qWarning() << "dsr_create_wall_nodes: failed to insert wall_" + QString::number(i);
            continue;
        }

        rt_api_->insert_or_assign_edge_RT(room_node_opt.value(),
                                          wall_id.value(),
                                          {mid.x(), mid.y(), half_h},
                                          {0.f, 0.f, yaw});
    }

    // ── Floor ─────────────────────────────────────────────────────────────────
    // Purely semantic parent for floor-attached objects; placed at room origin.
    DSR::Node floor_node = DSR::Node::create<floor_node_type>("floor");
    G_->add_or_modify_attrib_local<parent_att>(floor_node, dsr_room_id_);
    G_->add_or_modify_attrib_local<level_att>(floor_node, 4);
    // Display mesh generated from THIS polygon, so a viewer draws the actual layout instead of a
    // stand-in rectangle. width_m/depth_m carry the bbox so the unit-box OBJ can be rescaled.
    if (float fw = 0.f, fd = 0.f; write_floor_obj(polygon, std::filesystem::path(kFloorMeshOut), fw, fd))
    {
        G_->add_or_modify_attrib_local<mesh_path_att>(floor_node, std::string(kFloorMeshRel));
        G_->add_or_modify_attrib_local<width_m_att>(floor_node, fw);
        G_->add_or_modify_attrib_local<depth_m_att>(floor_node, fd);
    }
    else
        qWarning() << "dsr_create_wall_nodes: could not write the floor display mesh";

    rc::provenance::stamp_creation(*G_, floor_node);   // birth stamp: epoch ms + local ISO-8601
    const auto floor_id = G_->insert_node(floor_node);
    if (!floor_id.has_value())
        qWarning() << "dsr_create_wall_nodes: failed to insert floor node";
    else
        rt_api_->insert_or_assign_edge_RT(room_node_opt.value(),
                                          floor_id.value(),
                                          {0.f, 0.f, 0.f},
                                          {0.f, 0.f, 0.f});

    trigger_layout_();
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::cleanup_room_graph_nodes()
{
    if (!G_) return;
    // Delete affordance nodes hanging from room via "has_intention" edges,
    // then delete the room nodes themselves.
    for (const auto& room_node : G_->get_nodes_by_type("room"))
    {
        for (const auto& edge : G_->get_node_edges_by_type(room_node, "has_intention"))
            if (G_->get_node(edge.to()).has_value())
                G_->delete_node(edge.to());
        G_->delete_node(room_node);
    }
    // Delete wall and floor nodes owned by this agent.
    for (const auto& n : G_->get_nodes_by_type("wall"))
        G_->delete_node(n);
    if (auto n = G_->get_node("floor"); n.has_value())
        G_->delete_node(n.value());
    // Fallback: delete the affordance nodes by their known names in case they are orphaned.
    for (const char *name : {"afford_room", "afford_calib"})
        if (auto n = G_->get_node(name); n.has_value())
            G_->delete_node(n.value());
    room_node_created_ = false;
    dsr_room_id_ = 0;
    affordance_manager_.reset();
    calib_manager_.reset();
    calib_contract_written_ = false;
    stable_frames_ = 0;
}

///////////////////////////////////////////////////////////////////////////////
void RoomSceneGraph::check_init_graph_is_valid()
{
    if (!G_) { qWarning() << "dsr_init_graph: DSR graph not available"; return; }

    // Resolve the root/world node by type (name may vary, e.g. "root", "world")
    const auto root_nodes = G_->get_nodes_by_type("root");
    if (!root_nodes.empty())
        dsr_world_id_ = root_nodes.front().id();
    else { qWarning() << "dsr_init_graph: no 'root' type node found in graph"; return; }

    // Resolve the robot node by type
    const auto robot_nodes = G_->get_nodes_by_type("robot");
    if (!robot_nodes.empty())
        dsr_robot_id_ = robot_nodes.front().id();
    else { qWarning() << "dsr_init_graph: no 'robot' type node found in graph"; return; }

    // Optimise in the SAME frame we publish the robot↔room RT onto — the type-"robot" node itself.
    // The RT edge is written parent=<robot node>→child=room, so the SDF must fit the lidar in THAT
    // node's frame; otherwise the published pose is off by the fixed <robot>→base offset (e.g. Shadow→
    // body ≈ 4.5 cm z here) — a z-only error today, but wrong to publish. Deriving the frame from the
    // robot node (instead of a hardcoded "body") locks the optimisation frame to the write target so
    // they can never drift. Empty LIDAR_ROBOT_FRAME = auto-derive; a non-empty config value overrides.
    if (params_ != nullptr)
    {
        const std::string robot_name = robot_nodes.front().name();
        if (params_->LIDAR_ROBOT_FRAME.empty())
        {
            params_->LIDAR_ROBOT_FRAME = robot_name;
            qInfo() << "[room] optimisation/lidar frame derived from robot node ="
                    << QString::fromStdString(robot_name)
                    << "(robot↔room RT now published in this frame — no base offset)";
        }
        else if (params_->LIDAR_ROBOT_FRAME != robot_name)
            qWarning() << "[room] LIDAR_ROBOT_FRAME =" << QString::fromStdString(params_->LIDAR_ROBOT_FRAME)
                       << "differs from the robot node" << QString::fromStdString(robot_name)
                       << "— the published robot↔room RT will carry that frame's offset. Set it empty to auto-derive.";
    }

    load_robot_body_dimensions_from_graph();
}

}  // namespace rc
