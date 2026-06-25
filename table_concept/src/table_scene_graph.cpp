/*
 * table_scene_graph.cpp — DSR node/RT I/O for table_concept.
 */

#include "table_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <utility>

#include <QDebug>

#include "table_model.h"   // rc::TableModel statics (TOP_THICKNESS, LEG_RADIUS)

namespace rc {

TableSceneGraph::TableSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 const TableConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), cfg_(cfg), relayout_(std::move(relayout))
{}

void TableSceneGraph::scaffold_missing_table_nodes(const std::vector<TablePrior>& priors,
                                                   const MaskIngestor::MasksPacket& masks,
                                                   std::uint64_t room_node_id)
{
    if (!masks.valid || priors.empty())
        return;

    std::vector<bool> mask_used(masks.slices.size(), false);

    for (const auto& p : priors)
    {
        if (G_->get_node(p.node_name).has_value())
            continue;

        int best_mask_idx = -1;
        int fallback_mask_idx = -1;
        float best_dist_xy = std::numeric_limits<float>::max();
        const float prior_half_diag = 0.5f * std::sqrt(p.width_m * p.width_m + p.depth_m * p.depth_m);
        const float max_match_dist_xy = std::max(1.0f, 3.0f * p.sigma_pose + prior_half_diag);
        for (std::size_t i = 0; i < masks.slices.size(); ++i)
        {
            if (mask_used[i])
                continue;

            const auto& slice = masks.slices[i];
            if (slice.label != "table")
                continue;
            if (slice.support_end <= slice.support_begin)
                continue;

            if (fallback_mask_idx < 0)
                fallback_mask_idx = static_cast<int>(i);

            const float dx = slice.centroid.x() - p.room_x_m;
            const float dy = slice.centroid.y() - p.room_y_m;
            const float dist_xy = std::hypot(dx, dy);
            if (dist_xy <= max_match_dist_xy && dist_xy < best_dist_xy)
            {
                best_dist_xy = dist_xy;
                best_mask_idx = static_cast<int>(i);
            }
        }

        if (best_mask_idx < 0)
            best_mask_idx = fallback_mask_idx;

        if (best_mask_idx < 0)
        {
            std::print("table_concept: no usable table mask for '{}' yet\n", p.node_name);
            continue;
        }

        const auto& matched_slice = masks.slices[static_cast<std::size_t>(best_mask_idx)];
        mask_used[static_cast<std::size_t>(best_mask_idx)] = true;

        auto room_opt = G_->get_node(room_node_id);
        if (not room_opt.has_value())
        {
            qWarning() << "table_concept: room node missing, cannot scaffold" << p.node_name.c_str();
            continue;
        }

        DSR::Node table_node = DSR::Node::create<table_node_type>(p.node_name);
        G_->add_or_modify_attrib_local<width_m_att> (table_node, p.width_m);
        G_->add_or_modify_attrib_local<depth_m_att> (table_node, p.depth_m);
        G_->add_or_modify_attrib_local<height_m_att>(table_node, p.height_m);
        G_->add_or_modify_attrib_local<level_att>   (table_node, 3);
        G_->add_or_modify_attrib_local<parent_att>  (table_node, room_node_id);
        // Canvas position: derive from room node + fixed offset so the viewer doesn't randomize
        // pos_x/pos_y on every render tick.
        {
            const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G_->add_or_modify_attrib_local<pos_x_att>(table_node, rpx + 150.f);
            G_->add_or_modify_attrib_local<pos_y_att>(table_node, rpy +  50.f);
        }

        const auto id_opt = G_->insert_node(table_node);
        if (not id_opt.has_value())
        {
            qWarning() << "table_concept: failed to insert node" << p.node_name.c_str();
            continue;
        }

        const float z = p.height_m * 0.5f;
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                          {matched_slice.centroid.x(), matched_slice.centroid.y(), z},
                                          {0.0f, 0.0f, p.yaw_rad});

        if (relayout_)
            relayout_();

        std::print("table_concept: created node '{}' id={} from masks frame={} at ({:.2f}, {:.2f})\n",
                   p.node_name, id_opt.value(), masks.frame_id,
                   matched_slice.centroid.x(), matched_slice.centroid.y());
    }
}

bool TableSceneGraph::persist_table_belief(TableInstance& inst, std::uint64_t node_id,
                                           std::uint64_t room_id, float free_energy)
{
    auto node_opt = G_->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_write_model(inst, node_opt.value(), room_id, free_energy);
    return true;
}

void TableSceneGraph::step_write_model(TableInstance& inst, DSR::Node& node,
                                       std::uint64_t room_id, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish gate: only (re)write the geometry + mesh when the fitted dims/pose moved beyond a few
    // mm / mrad since the last publish. The voxelizer renders the MESH attribute (not the coarsely
    // dead-banded RT pose), rebuilt here every cycle from the raw fit — so writing it from the
    // sub-cm oscillating state each frame makes the viewer table JITTER even though room→table is
    // static. Freezing the mesh once settled removes the jitter. Mirrors bottle_concept's pub gate.
    constexpr float kPosEps = 0.003f;   // 3 mm
    constexpr float kYawEps = 0.005f;   // ~0.3°
    const bool geometry_changed =
        std::abs(s.cx - inst.last_pub_cx) > kPosEps or
        std::abs(s.cy - inst.last_pub_cy) > kPosEps or
        std::abs(s.w  - inst.last_pub_w)  > kPosEps or
        std::abs(s.h  - inst.last_pub_h)  > kPosEps or
        std::abs(s.table_height - inst.last_pub_H) > kPosEps or
        std::abs(static_cast<float>(std::remainder(s.yaw - inst.last_pub_yaw, 2.0 * M_PI))) > kYawEps;

    // Geometry attributes + mesh (gated to kill the viewer jitter once settled).
    if (geometry_changed)
    {
        G_->add_or_modify_attrib_local<width_m_att> (node, s.w);
        G_->add_or_modify_attrib_local<depth_m_att> (node, s.h);
        G_->add_or_modify_attrib_local<height_m_att>(node, s.table_height);
        G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
        write_table_mesh(inst, node);   // mesh for the voxelizer 3D viewer
        inst.last_pub_cx = s.cx; inst.last_pub_cy = s.cy;
        inst.last_pub_w  = s.w;  inst.last_pub_h  = s.h;
        inst.last_pub_H  = s.table_height; inst.last_pub_yaw = s.yaw;
    }
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);

    // Export the current historical RFE queue (remembered evidence) as XYZ triples.
    {
        const auto qpts = inst.queue.points();
        std::vector<float> qflat;
        qflat.reserve(qpts.size() * 3);
        for (const auto& p : qpts) { qflat.push_back(p.x()); qflat.push_back(p.y()); qflat.push_back(p.z()); }
        G_->runtime_checked_add_or_modify_attrib_local(node, "rfe_pts", qflat);
    }

    // Export full table-owned voxel memory (room frame) as XYZ triples.
    {
        std::vector<float> bank_flat;
        bank_flat.reserve(inst.voxel_bank_pts.size() * 3);
        for (const auto& p : inst.voxel_bank_pts) { bank_flat.push_back(p.x()); bank_flat.push_back(p.y()); bank_flat.push_back(p.z()); }
        G_->runtime_checked_add_or_modify_attrib_local(node, "table_voxel_bank_pts", bank_flat);
    }

    // Latest residual points (model-unexplained) for the voxelizer's residual layer — it reads
    // residual_pts_att but nothing was writing it, so that layer was always empty.
    {
        std::vector<float> res_flat;
        res_flat.reserve(inst.last_residual_pts.size() * 3);
        for (const auto& p : inst.last_residual_pts) { res_flat.push_back(p.x()); res_flat.push_back(p.y()); res_flat.push_back(p.z()); }
        G_->runtime_checked_add_or_modify_attrib_local(node, "residual_pts", res_flat);
    }

    // Active-perception channel for the controller's local lock-on search:
    //  - table_roi_offset [ox, oy]: normalised image-centre offset of the projected model
    //    (drive →0 to centre the table in the frame).
    //  - table_roi_fill: projected extent fraction (drive toward a sweet-spot for stand-off/scale).
    //  - table_roi_valid: model currently projects in front of the camera.
    //  - table_detection_alive / _confidence / _frames_since: is YOLO firing here, and how strongly.
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_roi_offset",
        std::vector<float>{inst.roi_offset_x, inst.roi_offset_y});
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_roi_fill", inst.roi_fill);
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_roi_valid", inst.roi_valid ? 1 : 0);
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_detection_alive", inst.detection_alive ? 1 : 0);
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_detection_confidence", inst.last_mask_confidence);
    G_->runtime_checked_add_or_modify_attrib_local(node, "table_frames_since_detection", inst.frames_since_detection);

    G_->update_node(node);

    write_rt_pose(room_id, inst);
    // Upload the table pose covariance after the pose write (so a rare >5 cm RT recreate doesn't
    // clobber it). `force` on a geometry republish; otherwise the write self-gates on a meaningful
    // uncertainty change, so a stationary-but-tightening table stays current without edge churn.
    write_rt_covariance(room_id, inst, geometry_changed);
}

void TableSceneGraph::write_rt_covariance(std::uint64_t room_id, TableInstance& inst, bool force)
{
    if (not cfg_.rt_cov_upload or room_id == 0)
        return;

    // Per-DOF accumulated precision from the Fisher filter: [cx,cy,w,h,H,leg,yaw,inset].
    const auto& Y = inst.fisher_info_raw;
    const float scale = std::max(1e-6f, cfg_.rt_cov_scale);
    constexpr float big = 1e3f;   // unobservable / never-seen DOF → large variance
    const auto var = [&](int j) -> float { return Y[j] > 1e-6f ? scale / Y[j] : big; };

    const float vx = var(0), vy = var(1), vz = 0.25f * var(4), vyaw = var(6);   // data-driven DOFs

    // Self-gate on the data-driven trace: write on a geometry republish (force) OR when the
    // uncertainty moved >5 % since the last publish (covers a frozen pose whose belief keeps
    // tightening / loosening). Avoids rewriting the edge every cycle once everything has settled.
    const float trace = vx + vy + vz + vyaw;
    const float prev  = inst.last_pub_cov_trace;
    const bool cov_changed = not std::isfinite(prev) or prev <= 0.0f or
                             std::abs(trace - prev) > 0.05f * prev;
    if (not force and not cov_changed)
        return;

    auto edge = G_->get_edge(room_id, inst.node_id, "RT");
    if (not edge.has_value())
        return;

    // 6×6 SE3 covariance, row-major [x,y,z,rx,ry,rz]; diagonal only (cross-terms unmodelled).
    std::vector<float> cov(36, 0.0f);
    cov[0 * 6 + 0] = vx;     // x   ← cx
    cov[1 * 6 + 1] = vy;     // y   ← cy
    cov[2 * 6 + 2] = vz;     // z   ← table_height (z = H/2 ⇒ var_z = var_H/4)
    cov[3 * 6 + 3] = big;   // roll  (unobservable)
    cov[4 * 6 + 4] = big;   // pitch (unobservable)
    cov[5 * 6 + 5] = vyaw;   // yaw ← ψ

    G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov);
    G_->insert_or_assign_edge(edge.value());
    inst.last_pub_cov_trace = trace;
}

std::vector<float> TableSceneGraph::make_table_mesh(const TableState& s)
{
    // Flat triangle list (room frame): 1 top slab + 4 square legs = 5 boxes × 12 tri.
    std::vector<float> verts;
    verts.reserve(5 * 108);

    const float cy = std::cos(s.yaw);
    const float sy = std::sin(s.yaw);

    auto push_box = [&](float bx, float by, float bz, float hw, float hd, float hh)
    {
        auto push = [&](float lx, float ly, float lz)
        {
            verts.push_back(bx + cy * lx - sy * ly);
            verts.push_back(by + sy * lx + cy * ly);
            verts.push_back(bz + lz);
        };
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw, hd,-hh);  // bottom
        push(-hw,-hd,-hh); push( hw, hd,-hh); push(-hw, hd,-hh);
        push(-hw,-hd, hh); push( hw, hd, hh); push( hw,-hd, hh);  // top
        push(-hw,-hd, hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw,-hd, hh);  // front -y
        push(-hw,-hd,-hh); push( hw,-hd, hh); push(-hw,-hd, hh);
        push( hw, hd,-hh); push(-hw, hd,-hh); push(-hw, hd, hh);  // back  +y
        push( hw, hd,-hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push(-hw,-hd, hh); push(-hw, hd, hh);  // left  -x
        push(-hw,-hd,-hh); push(-hw, hd, hh); push(-hw, hd,-hh);
        push( hw,-hd,-hh); push( hw, hd,-hh); push( hw, hd, hh);  // right +x
        push( hw,-hd,-hh); push( hw, hd, hh); push( hw,-hd, hh);
    };

    // Top slab — centred at floor + leg_length + half slab thickness
    const float ht  = TableModel::TOP_THICKNESS * 0.5f;
    push_box(s.cx, s.cy, s.leg_length + ht, s.w * 0.5f, s.h * 0.5f, ht);

    // 4 legs — square cross-section (2×LEG_RADIUS), inset from table edges by the model's
    // leg_inset (the top overhang), matching the SDF leg placement.
    const float lr  = TableModel::LEG_RADIUS;
    const float lhz = s.leg_length * 0.5f;
    for (int ix : {-1, 1})
        for (int iy : {-1, 1})
        {
            const float lx = ix * (s.w * 0.5f - s.leg_inset);
            const float ly = iy * (s.h * 0.5f - s.leg_inset);
            const float rx = s.cx + cy * lx - sy * ly;
            const float ry = s.cy + sy * lx + cy * ly;
            push_box(rx, ry, lhz, lr, lr, lhz);
        }

    return verts;
}

void TableSceneGraph::write_table_mesh(TableInstance& inst, DSR::Node& node)
{
    const std::vector<float> verts = make_table_mesh(inst.model.state());
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, verts);
}

Eigen::Matrix2f TableSceneGraph::read_robot_covariance(std::uint64_t room_id) const
{
    const auto robots = G_->get_nodes_by_type("robot");
    if (not robots.empty() and room_id != 0)
    {
        const auto edge = G_->get_edge(room_id, robots.front().id(), "RT");
        if (edge.has_value())
        {
            const auto cov_opt = G_->get_attrib_by_name<rt_se2_covariance_att>(edge.value());
            if (cov_opt.has_value())
            {
                const auto& c = cov_opt.value().get();
                // rt_se2_covariance is a 9-vector (3×3 row-major for [x,y,θ]); take the XY block.
                if (c.size() >= 4)
                {
                    Eigen::Matrix2f m;
                    m << c[0], c[1], c[3], c[4];
                    return m;
                }
            }
        }
    }
    return Eigen::Matrix2f::Identity() * 0.01f;   // fallback: small identity (high confidence)
}

void TableSceneGraph::write_rt_pose(std::uint64_t room_id, TableInstance& inst)
{
    if (room_id == 0 or not rt_api_)
        return;

    const auto& s = inst.model.state();

    // Dead-band: suppress RT edge updates below ~5 cm to avoid pos churn from gradient oscillations.
    constexpr float kMinWriteDistSq = 0.05f * 0.05f;
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (dx*dx + dy*dy < kMinWriteDistSq)
        return;

    auto room_opt = G_->get_node(room_id);
    if (not room_opt.has_value())
        return;

    // Table node origin = BASE on the floor (z=0), NOT the mid-height. Every consumer assumes a
    // base origin: the voxelizer box (z∈[origin, origin+height]), and bottle_concept's table-top
    // lookup + support decision (top = origin.z + height). Publishing z=table_height/2 put the
    // origin at mid-height → table_top came out as 1.5·height → bottles failed the support test,
    // parented to the room and floated. Keep the base on the floor so top = 0 + height = height.
    const float z = 0.0f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {s.cx, s.cy, z},
                                      {0.0f, 0.0f, s.yaw});
    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

void TableSceneGraph::write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop)
{
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>  (node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>  (node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>        (node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>     (node, true);
    G_->update_node(node);
}

}  // namespace rc
