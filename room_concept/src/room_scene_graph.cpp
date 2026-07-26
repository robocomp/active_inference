/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */

#include "room_scene_graph.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <print>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

#include <QString>
#include <QtCore/qdebug.h>

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

    std::vector<float> cov_flat(36, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov_flat[r * 6 + c] = cov_se2(r, c);

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

    // Publish the robot's BODY-FRAME twist + covariance on the SAME RT edge, so the controller reads
    // velocity DIRECTLY (no pose differentiation → no correction-induced velocity spikes). Convention:
    // rt_translation_velocity = [adv(fwd,+x), side(lat,+y), 0], rt_rotation_euler_xyz_velocity = [0,0,rot].
    // last_adv_/side_/rot_ are the measured odometry (set in update()). Covariance = diagonal (config).
    // Re-fetch the edge FRESH from the graph (NOT from the stale parent_opt node captured before the RT
    // write) — otherwise we'd write back the old pose ring buffer and freeze the position.
    if (auto edge = G_->get_edge(parent_opt.value().id(), child_id, "RT"); edge.has_value())
    {
        G_->add_or_modify_attrib_local<rt_translation_velocity_att>(
            edge.value(), std::vector<float>{last_adv_, last_side_, 0.f});
        G_->add_or_modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(
            edge.value(), std::vector<float>{0.f, 0.f, last_rot_});
        std::vector<float> vel_cov(36, 0.f);   // 6×6 row-major; SE2 diag at [0],[7],[14] (matches pose cov)
        if (params_)
        {
            vel_cov[0]  = params_->ROBOT_VEL_COV_ADV;
            vel_cov[7]  = params_->ROBOT_VEL_COV_SIDE;
            vel_cov[14] = params_->ROBOT_VEL_COV_ROT;
        }
        G_->add_or_modify_attrib_local<rt_se2_covariance_velocity_att>(edge.value(), vel_cov);
        G_->insert_or_assign_edge(edge.value());
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
void RoomSceneGraph::dsr_update_affordance(const rc::RoomConcept::UpdateResult& res)
{
    if (!G_ || !room_node_created_ || !epistemic_) return;

    auto& planner = epistemic_->epistemic_planner();

    // Always update robot state so mark_and_refresh uses the correct position.
    epistemic_->set_robot_state(res.robot_pose, res.covariance);

    if (affordance_manager_.consume_completion_event())
    {
        planner.clear_target();
        planner.mark_and_refresh();   // keep path trail live in viewer
        planner.refresh_belief();     // just finished exploring → belief fresh (restart forget clock)
        return;
    }

    if (affordance_manager_.is_executing(G_))
    {
        planner.mark_and_refresh();   // stamp path + refresh IoR overlay during navigation
        planner.refresh_belief();     // actively exploring → hold belief fresh; forgetting only when idle
        // Throttled: a stuck claim (controller Executing but never completing) keeps the planner idle
        // here forever — no reselect, no [planner] line. Reveals the OTHER "robot won't move on" path.
        static int exec_hold_dbg = 0;
        if (++exec_hold_dbg % 90 == 0)
            std::print("[planner] afford_room EXECUTING (controller-claimed) — planner idle, holding "
                       "target ({:.2f},{:.2f}); {} cycles\n",
                       planner.current_target() ? planner.current_target()->position.x() : 0.f,
                       planner.current_target() ? planner.current_target()->position.y() : 0.f,
                       exec_hold_dbg);
        return;
    }

    // Refresh obstacle exclusion zones from DSR graph before selecting the target.
    update_planner_obstacle_footprints();

    // Refresh the localizer's object anchors (validated objects → SE(2) pose landmarks).
    refresh_object_anchors();

    // Ask the planner for the current best target (handles dwell / arrival internally)
    const auto target_opt = planner.update_target();

    if (!target_opt.has_value()) return;

    const float tx   = target_opt->position.x();
    const float ty   = target_opt->position.y();
    // Advertise the GROUNDED epistemic value in nats: the raw FIM D-optimality gain
    // ΔH = ½·log det(I + Y_prior⁻¹·I_pred) = expected pose-entropy reduction from observing
    // corners/walls at this viewpoint. NOT target.score (which folds in IoR-staleness × path
    // heuristics used only for the room's own target ranking). This puts afford_room's gain in the
    // same currency as afford_table's ΔH so the controller can compare them as one EFE term.
    //
    // Recompute LIVE against the current pose precision (do not reuse the value frozen at target
    // selection): as the robot drives in and localization tightens, Y_prior grows ⇒ the pose-FIM
    // part decays toward 0. We publish the TOTAL gain (pose-FIM ΔH + IoR patrol-staleness drive):
    // the IoR term is NON-saturating, so once the pose is localized the advertised gain stays
    // positive on long-unvisited cells instead of falling through the consumer's withdrawal
    // threshold and stopping exploration after a single affordance.
    // The recompute at target_opt->position also gives the rotate-in-place recovery a real gain.
    const float gain = planner.live_total_epistemic_gain(target_opt->position);

    // Heading: face toward room centre so the robot maximises wall/corner visibility
    const float cx  = (planner.room_min().x() + planner.room_max().x()) * 0.5f;
    const float cy  = (planner.room_min().y() + planner.room_max().y()) * 0.5f;
    const float yaw = std::atan2(cy - ty, cx - tx);

    affordance_manager_.publish_target(
        G_,
        dsr_room_id_,
        tx,
        ty,
        yaw,
        gain,
        [this]() { trigger_layout_(); },
        [this]() { trigger_layout_(); });
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
        int n_tables = 0, with_obs = 0;
        for (const auto& n : G_->get_nodes_by_type("object"))
        {
            if (not node_is_object_class(*G_, n, "table")) continue;
            ++n_tables;
            if (const auto o = G_->get_attrib_by_name<obj_obs_robot_att>(n);
                o.has_value() and o->get().size() >= 2)
                ++with_obs;
        }
        std::print("[room][anchors] table_nodes={} with_obj_obs_robot={} anchors_used={}\n",
                   n_tables, with_obs, anchors.size());
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

    // Guard — idempotent: if wall nodes already exist under this room, skip.
    if (!G_->get_nodes_by_type("wall").empty())
        return;

    const auto polygon = room_concept_->nominal_room_polygon();
    const int n = static_cast<int>(polygon.size());
    if (n < 3) { qWarning() << "dsr_create_wall_nodes: polygon has fewer than 3 vertices"; return; }

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
        G_->add_or_modify_attrib_local<parent_att>(wall_node, dsr_room_id_);
        G_->add_or_modify_attrib_local<level_att>(wall_node, 4);

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
    // Fallback: delete the affordance node by its known name in case it is orphaned.
    if (auto n = G_->get_node("afford_room"); n.has_value())
        G_->delete_node(n.value());
    room_node_created_ = false;
    dsr_room_id_ = 0;
    affordance_manager_.reset();
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
