/*
 * bottle_scene_graph.cpp — DSR node/RT I/O for bottle_concept.
 */

#include "bottle_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <utility>

#include <QDebug>

BottleSceneGraph::BottleSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                   DSR::RT_API* rt_api,
                                   DSR::InnerEigenAPI* inner_eigen,
                                   AgentConfig& cfg)
    : G_(std::move(graph)), rt_api_(rt_api), inner_eigen_(inner_eigen), cfg_(cfg)
{}

std::optional<DSR::Node> BottleSceneGraph::find_table_node(float bx, float by) const
{
    if (not inner_eigen_)
        return std::nullopt;
    const auto t = G_->get_node("table");           // bootstrap/room-created static surface
    if (not t.has_value())
        return std::nullopt;
    const float w = G_->get_attrib_by_name<width_m_att> (t.value()).value_or(0.0f);
    const float d = G_->get_attrib_by_name<depth_m_att> (t.value()).value_or(0.0f);
    const float h = G_->get_attrib_by_name<height_m_att>(t.value()).value_or(0.0f);
    if (h <= 0.0f)
        return std::nullopt;
    // Table origin (its base) expressed in the room frame — a Vector3d, so no Transform-block hazard.
    const auto org = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), "table", 0);
    if (not org.has_value())
        return std::nullopt;
    // Footprint gate (loose, yaw-agnostic): the bottle centre must lie over the table.
    const float half = 0.5f * std::max(w, d) + 0.10f;
    if (std::hypot(bx - static_cast<float>(org->x()), by - static_cast<float>(org->y())) > half)
        return std::nullopt;
    return t;
}

std::optional<float> BottleSceneGraph::find_table_top(float bx, float by) const
{
    const auto t = find_table_node(bx, by);
    if (not t.has_value())
        return std::nullopt;
    const float h = G_->get_attrib_by_name<height_m_att>(t.value()).value_or(0.0f);
    const auto org = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), "table", 0);
    if (not org.has_value())
        return std::nullopt;
    return static_cast<float>(org->z()) + h;       // table top = base origin z + height
}

void BottleSceneGraph::scaffold_missing_bottle_nodes(const std::vector<BottlePrior>& priors,
                                                     const BottlePerception::MasksPacket& masks,
                                                     std::uint64_t room_node_id)
{
    if (not masks.valid or priors.empty())
        return;

    std::vector<bool> mask_used(masks.slices.size(), false);

    for (const auto& p : priors)
    {
        if (G_->get_node(p.node_name).has_value())
            continue;

        int best_mask_idx = -1;
        int fallback_mask_idx = -1;
        float best_dist_xy = std::numeric_limits<float>::max();
        const float max_match_dist_xy = std::max(0.5f, 3.0f * p.sigma_pose + p.radius_m);
        for (std::size_t i = 0; i < masks.slices.size(); ++i)
        {
            if (mask_used[i])
                continue;

            const auto& slice = masks.slices[i];
            if (slice.label != "bottle")
                continue;
            if (slice.support_end <= slice.support_begin)
                continue;

            if (fallback_mask_idx < 0)
                fallback_mask_idx = static_cast<int>(i);

            const float dx = slice.centroid.x() - p.room_x_m;
            const float dy = slice.centroid.y() - p.room_y_m;
            const float dist_xy = std::hypot(dx, dy);
            if (dist_xy <= max_match_dist_xy and dist_xy < best_dist_xy)
            {
                best_dist_xy = dist_xy;
                best_mask_idx = static_cast<int>(i);
            }
        }

        if (best_mask_idx < 0)
            best_mask_idx = fallback_mask_idx;
        if (best_mask_idx < 0)
            continue;

        const auto& matched_slice = masks.slices[static_cast<std::size_t>(best_mask_idx)];
        mask_used[static_cast<std::size_t>(best_mask_idx)] = true;

        auto room_opt = G_->get_node(room_node_id);
        if (not room_opt.has_value())
        {
            qWarning() << "bottle_concept: room node missing, cannot scaffold" << p.node_name.c_str();
            continue;
        }

        DSR::Node bottle_node = DSR::Node::create<cylinder_node_type>(p.node_name);
        G_->add_or_modify_attrib_local<width_m_att> (bottle_node, 2.0f * p.radius_m);
        G_->add_or_modify_attrib_local<depth_m_att> (bottle_node, 2.0f * p.radius_m);
        G_->add_or_modify_attrib_local<height_m_att>(bottle_node, p.height_m);

        // Hang the bottle from the TABLE it sits on (room→bottle ⇒ table→bottle) when one is under it;
        // else from the room. insert_or_assign_edge_RT (below) sets the bottle's parent + level
        // (= anchor.level + 1) authoritatively from the RT edge, so we don't bookkeep them here.
        const float cz = matched_slice.centroid.z() > 1e-3f ? matched_slice.centroid.z() : p.room_z_m;
        const auto table = find_table_node(matched_slice.centroid.x(), matched_slice.centroid.y());
        DSR::Node parent_node = table.value_or(room_opt.value());
        const int parent_level = G_->get_node_level(parent_node).value_or(-1);   // for the log line only
        {
            const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G_->add_or_modify_attrib_local<pos_x_att>(bottle_node, rpx + 180.f);
            G_->add_or_modify_attrib_local<pos_y_att>(bottle_node, rpy +  80.f);
        }

        const auto id_opt = G_->insert_node(bottle_node);
        if (not id_opt.has_value())
        {
            qWarning() << "bottle_concept: failed to insert node" << p.node_name.c_str();
            continue;
        }

        // Centroid is room-frame; express it in the parent's frame for the RT edge (identity if room).
        Mat::Vector3d c_parent(matched_slice.centroid.x(), matched_slice.centroid.y(), cz);
        if (const auto cp = inner_eigen_->transform(parent_node.name(), c_parent, "room", 0); cp.has_value())
            c_parent = cp.value();
        rt_api_->insert_or_assign_edge_RT(parent_node, id_opt.value(),
                                          {static_cast<float>(c_parent.x()), static_cast<float>(c_parent.y()),
                                           static_cast<float>(c_parent.z())},
                                          {0.0f, 0.0f, 0.0f});

        std::print("bottle_concept: created node '{}' id={} parent='{}' (lvl {}) from masks frame={} at room ({:.2f}, {:.2f}, {:.2f})\n",
                   p.node_name, id_opt.value(), parent_node.name(), parent_level + 1, masks.frame_id,
                   matched_slice.centroid.x(), matched_slice.centroid.y(), cz);
    }
}

void BottleSceneGraph::step_write_model(BottleInstance& inst, DSR::Node& node, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish to the graph only when the model meaningfully changed. Once the fit is stable this
    // stops the per-cycle rewrite of the node (incl. the large mesh + the ever-incrementing
    // model_generation) and the RT edge — sparing the DSR network and every agent's graph viewer.
    // FE jitter is deliberately NOT a trigger (it never fully settles), only pose/size do.
    constexpr float kPosEps  = 0.005f;   // 5 mm
    constexpr float kSizeEps = 0.002f;   // 2 mm
    const bool changed =
        std::abs(s.radius - inst.last_pub_radius) > kSizeEps or
        std::abs(s.height - inst.last_pub_height) > kSizeEps or
        std::abs(s.cx     - inst.last_pub_cx)     > kPosEps  or
        std::abs(s.cy     - inst.last_pub_cy)     > kPosEps  or
        std::abs(s.cz     - inst.last_pub_cz)     > kPosEps;
    if (not changed)
        return;

    inst.last_pub_radius = s.radius;
    inst.last_pub_height = s.height;
    inst.last_pub_cx     = s.cx;
    inst.last_pub_cy     = s.cy;
    inst.last_pub_cz     = s.cz;

    G_->add_or_modify_attrib_local<width_m_att> (node, 2.0f * s.radius);
    G_->add_or_modify_attrib_local<depth_m_att> (node, 2.0f * s.radius);
    G_->add_or_modify_attrib_local<height_m_att>(node, s.height);
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, make_cylinder_mesh(s));

    // Export the historical RFE queue (remembered evidence) as XYZ triples.
    {
        const auto qpts = inst.queue.points();
        std::vector<float> qflat;
        qflat.reserve(qpts.size() * 3);
        for (const auto& p : qpts) { qflat.push_back(p.x()); qflat.push_back(p.y()); qflat.push_back(p.z()); }
        G_->runtime_checked_add_or_modify_attrib_local(node, "rfe_pts", qflat);
    }

    G_->update_node(node);

    write_rt_pose(inst);
}

Eigen::Matrix2f BottleSceneGraph::read_robot_covariance(std::uint64_t room_node_id) const
{
    const auto robots = G_->get_nodes_by_type("robot");
    if (not robots.empty() and room_node_id != 0)
    {
        const auto edge = G_->get_edge(room_node_id, robots.front().id(), "RT");
        if (edge.has_value())
        {
            const auto cov_opt = G_->get_attrib_by_name<rt_covariance_att>(edge.value());
            if (cov_opt.has_value())
            {
                const auto& c = cov_opt.value().get();
                // rt_covariance is the 6×6 SE3 covariance (row-major); the XY
                // block sits at indices (0,0),(0,1),(1,0),(1,1) with stride 6.
                if (c.size() >= 8)
                {
                    Eigen::Matrix2f m;
                    m << c[0], c[1], c[6], c[7];
                    return m;
                }
            }
        }
    }
    return Eigen::Matrix2f::Identity() * 0.01f;
}

void BottleSceneGraph::write_rt_pose(BottleInstance& inst)
{
    if (inst.parent_id == 0 or not rt_api_ or not inner_eigen_)
        return;

    const auto& s = inst.model.state();

    // Dead-band: suppress RT updates below ~1 cm to avoid pos churn from jitter (room-frame state).
    constexpr float kMinWriteDistSq = 0.01f * 0.01f;
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (dx*dx + dy*dy < kMinWriteDistSq)
        return;

    auto parent_opt = G_->get_node(inst.parent_id);
    if (not parent_opt.has_value())
        return;

    // Express the room-frame fit in the PARENT frame for the RT edge (identity if parent==room;
    // table→bottle when hung from the table). The table is a pure translation from room, so the
    // pose_covariance (room frame) carries unchanged to the table-frame edge below.
    Mat::Vector3d p_parent(s.cx, s.cy, s.cz);
    if (const auto pp = inner_eigen_->transform(inst.parent_name, p_parent, "room", 0); pp.has_value())
        p_parent = pp.value();
    rt_api_->insert_or_assign_edge_RT(parent_opt.value(), inst.node_id,
                                      {static_cast<float>(p_parent.x()), static_cast<float>(p_parent.y()),
                                       static_cast<float>(p_parent.z())},
                                      {0.0f, 0.0f, 0.0f});

    // Attach P_bottle (Laplace covariance) on the RT edge as the 6×6 SE3
    // covariance (rt_covariance_att, row-major [x,y,z,rx,ry,rz]). The 3×3
    // translation block comes from the model; orientation is unobservable for a
    // symmetric upright cylinder, so its diagonal is large. The controller reads
    // this to set its closed→open-loop look-up rate.
    if (auto edge = G_->get_edge(inst.parent_id, inst.node_id, "RT"); edge.has_value())
    {
        const auto fit_pts = inst.queue.points();
        const Eigen::Matrix3f cov = inst.model.pose_covariance(fit_pts, inst.queue.weights());
        std::vector<float> cov_flat(36, 0.0f);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov_flat[r * 6 + c] = cov(r, c);
        cov_flat[3 * 6 + 3] = cfg_.yaw_variance;   // rx
        cov_flat[4 * 6 + 4] = cfg_.yaw_variance;   // ry
        cov_flat[5 * 6 + 5] = cfg_.yaw_variance;   // rz
        G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov_flat);
        G_->insert_or_assign_edge(edge.value());
    }

    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

std::vector<float> BottleSceneGraph::make_cylinder_mesh(const BottleState& s, int segments)
{
    // Flat triangle list (room frame): side wall + top/bottom caps.
    std::vector<float> verts;
    const int n = std::max(6, segments);
    verts.reserve(static_cast<std::size_t>(n) * 4 * 9);

    const float r  = s.radius;
    const float zt = s.cz + s.height * 0.5f;
    const float zb = s.cz - s.height * 0.5f;
    const float two_pi = 2.0f * std::numbers::pi_v<float>;

    auto push = [&](float x, float y, float z) { verts.push_back(x); verts.push_back(y); verts.push_back(z); };

    for (int i = 0; i < n; ++i)
    {
        const float a0 = two_pi * static_cast<float>(i)     / static_cast<float>(n);
        const float a1 = two_pi * static_cast<float>(i + 1) / static_cast<float>(n);
        const float x0 = s.cx + r * std::cos(a0), y0 = s.cy + r * std::sin(a0);
        const float x1 = s.cx + r * std::cos(a1), y1 = s.cy + r * std::sin(a1);

        // Side wall (two triangles).
        push(x0, y0, zb); push(x1, y1, zb); push(x1, y1, zt);
        push(x0, y0, zb); push(x1, y1, zt); push(x0, y0, zt);
        // Top cap.
        push(s.cx, s.cy, zt); push(x0, y0, zt); push(x1, y1, zt);
        // Bottom cap.
        push(s.cx, s.cy, zb); push(x1, y1, zb); push(x0, y0, zb);
    }

    return verts;
}
