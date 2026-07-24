/*
 * table_scene_graph.cpp — DSR node/RT I/O for table_concept.
 *
 * Implements TableSceneGraph: births "table_N" nodes from tracker detections and, each cycle, writes the
 * fitted model back to the graph — geometry attrs + mesh + residual/voxel-bank export + free energy, the
 * room→table RT pose (dead-banded), and the 6×6 pose covariance mapped from the belief Σ (with a flat
 * roll/pitch prior). Writes self-gate on a meaningful geometry/uncertainty change to avoid per-cycle edge
 * churn. All graph access is on the main thread (SpecificWorker's compute path).
 */

#include "table_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <print>
#include <utility>

#include <QDebug>

#include "table_model.h"   // rc::TableModel statics (TOP_THICKNESS, LEG_RADIUS)
#include "../../common/object_anchor/object_anchor_contract.h"

namespace rc {

TableSceneGraph::TableSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 const TableConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), cfg_(cfg), relayout_(std::move(relayout))
{}

// ─── Node birth ──────────────────────────────────────────────────────────────────────────────────

std::uint64_t TableSceneGraph::create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                              std::uint64_t room_node_id)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return 0;

    // Auto-name: one past the highest existing "table_<N>".
    int max_n = 0;
    for (const auto& n : G_->get_nodes_by_type("table"))
        if (n.name().rfind("table_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(6))); } catch (...) {}
    const std::string name = "table_" + std::to_string(max_n + 1);

    DSR::Node table_node = DSR::Node::create<table_node_type>(name);
    G_->add_or_modify_attrib_local<width_m_att> (table_node, cfg_.tracker_birth_width_m);
    G_->add_or_modify_attrib_local<depth_m_att> (table_node, cfg_.tracker_birth_depth_m);
    G_->add_or_modify_attrib_local<height_m_att>(table_node, cfg_.tracker_birth_height_m);
    G_->add_or_modify_attrib_local<level_att>   (table_node, 3);
    G_->add_or_modify_attrib_local<parent_att>  (table_node, room_node_id);
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(table_node, rpx + 150.f);
        G_->add_or_modify_attrib_local<pos_y_att>(table_node, rpy +  50.f);
    }

    const auto id_opt = G_->insert_node(table_node);
    if (not id_opt.has_value())
        return 0;

    const float z = cfg_.tracker_birth_height_m * 0.5f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                      {centroid_room.x(), centroid_room.y(), z}, {0.0f, 0.0f, 0.0f});

    if (relayout_)
        relayout_();

    std::print("table_concept: [tracker] BIRTH '{}' id={} at room ({:.2f},{:.2f})\n",
               name, id_opt.value(), centroid_room.x(), centroid_room.y());
    return id_opt.value();
}

// ─── Model publish (geometry · mesh · residual/voxel export · RT · covariance) ───────────────────

// Resolve the table node by id and write the fitted model to it (false if the node is gone).
bool TableSceneGraph::persist_table_belief(TableInstance& inst, std::uint64_t node_id,
                                           std::uint64_t room_id, float free_energy)
{
    auto node_opt = G_->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_write_model(inst, node_opt.value(), room_id, free_energy);
    return true;
}

// Write the full fitted model onto the node + RT edge: geometry attrs + mesh (gated on a real geometry move),
// free energy, voxel-bank + residual point exports, the active-perception ROI/detection channel, then the RT
// pose and pose covariance. The mesh gate freezes the voxelizer render once settled to stop viewer jitter.
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
    // Inferred shape subtype (round/square), chosen by free-energy model evidence in TableFitter::evaluate_shape.
    // The voxelizer reads this to render the matching mesh (disc vs box).
    G_->add_or_modify_attrib_local<object_subtype_att>(node, inst.subtype);

    // Export full table-owned voxel memory (room frame) as XYZ triples.
    {
        std::vector<float> bank_flat;
        bank_flat.reserve(inst.voxel_bank_pts.size() * 3);
        for (const auto& p : inst.voxel_bank_pts) { bank_flat.push_back(p.x()); bank_flat.push_back(p.y()); bank_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<table_voxel_bank_pts_att>(node, bank_flat);
    }

    // DIAGNOSTIC one-shot: dump this table's accumulated voxel-bank cloud (room frame) to a file for the
    // offline square-vs-round model comparison (tests/compare_models). Gated on a config path + a populated
    // bank; fires exactly once per instance. No effect unless TableConcept.DumpCloudPath is set.
    if (not cfg_.dump_cloud_path.empty() and not inst.cloud_dumped and inst.voxel_bank_pts.size() > 200)
    {
        if (std::ofstream f{cfg_.dump_cloud_path}; f)
        {
            for (const auto& p : inst.voxel_bank_pts) f << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
            inst.cloud_dumped = true;
            std::print("[{}] dumped {} voxel-bank pts -> {}\n",
                       inst.node_name, inst.voxel_bank_pts.size(), cfg_.dump_cloud_path);
        }
    }

    // Latest residual points (model-unexplained) for the voxelizer's residual layer — it reads
    // residual_pts_att but nothing was writing it, so that layer was always empty.
    {
        std::vector<float> res_flat;
        res_flat.reserve(inst.last_residual_pts.size() * 3);
        for (const auto& p : inst.last_residual_pts) { res_flat.push_back(p.x()); res_flat.push_back(p.y()); res_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<residual_pts_att>(node, res_flat);
    }

    // Active-perception channel for the controller's local lock-on search:
    //  - table_roi_offset [ox, oy]: normalised image-centre offset of the projected model
    //    (drive →0 to centre the table in the frame).
    //  - table_roi_fill: projected extent fraction (drive toward a sweet-spot for stand-off/scale).
    //  - table_roi_valid: model currently projects in front of the camera.
    //  - table_detection_alive / _confidence / _frames_since: is YOLO firing here, and how strongly.
    G_->add_or_modify_attrib_local<table_roi_offset_att>(node,
        std::vector<float>{inst.roi_offset_x, inst.roi_offset_y});
    G_->add_or_modify_attrib_local<table_roi_fill_att>(node, inst.roi_fill);
    G_->add_or_modify_attrib_local<table_roi_valid_att>(node, inst.roi_valid);
    G_->add_or_modify_attrib_local<table_detection_alive_att>(node, inst.detection_alive);
    G_->add_or_modify_attrib_local<table_detection_confidence_att>(node, inst.last_mask_confidence);
    G_->add_or_modify_attrib_local<table_frames_since_detection_att>(node, inst.frames_since_detection);

    // Object-anchor observation z_o (raw camera-frame centroid → static body extrinsic) for the room
    // localizer's landmark factor. Position-only (single-view yaw is biased). Batched into this node
    // write; only present when the fitter has it enabled (OFF by default).
    if (inst.obs_robot_valid)
        rc::object_anchor::write_observation_aniso(
            *G_, node, inst.obs_robot.head<2>(), inst.obs_robot_cov);   // anisotropic R_o (ray-loose)

    G_->update_node(node);

    write_rt_pose(room_id, inst);
    // Upload the table pose covariance after the pose write (so a rare >5 cm RT recreate doesn't
    // clobber it). `force` on a geometry republish; otherwise the write self-gates on a meaningful
    // uncertainty change, so a stationary-but-tightening table stays current without edge churn.
    write_rt_covariance(room_id, inst, geometry_changed);
}

// Map the belief's 6×6 Σ [cx,cy,H,w,h,yaw] onto the room→table RT edge covariance (+ chain term), with
// roll/pitch pinned small by the flat-on-the-floor prior and yaw as the marginal (mode-entropy) variance.
void TableSceneGraph::write_rt_covariance(std::uint64_t room_id, TableInstance& inst, bool force)
{
    if (room_id == 0)
        return;

    const float scale = std::max(1e-6f, cfg_.rt_cov_scale);
    // A table rests FLAT on the floor: roll/pitch are pinned to ~0 by that physical prior — CONFIDENTLY known,
    // not unknown. The old 1e3 ("unobservable → huge variance") was backwards: it told the controller the table
    // might be tilted AND it dominated the published-covariance plot scale so the real sub-1 DOF read as ~0.
    // Publish a small flat-table variance instead (≈1.3° std), consistent with the geometry.
    constexpr float flat_rp_var = 5e-4f;   // roll/pitch: confidently flat on the floor (std ≈ 1.3°)

    if (not inst.ai2_initialized)
        return;   // belief not seeded yet — nothing calibrated to publish

    // The belief carries a full 6×6 Σ over [cx,cy,H,w,h,yaw] — publish it directly. Σ already folds the
    // per-frame common-mode floor (incl. chain), but across-frame accumulation can tighten cx,cy below it,
    // so we still ADD the localization/chain term below — the safe direction for the controller's governor.
    const auto& S = inst.ai2_belief.covariance();
    float vx   = scale * S(0, 0);          // cx
    float vy   = scale * S(1, 1);          // cy
    float vz   = scale * 0.25f * S(2, 2);  // z = H/2 ⇒ var_z = var_H/4
    // MARGINAL yaw variance: within-mode Σ(5,5) + discrete-mode entropy p(1−p)(π/2)². A still-ambiguous
    // near-square table publishes an honest ~45° (not the overconfident ~1° within-mode width) so the
    // controller's uncertainty governor + epistemic planner know yaw is unresolved. Collapses as evidence
    // resolves the mode. See TABLE.md.
    float vyaw = scale * inst.ai2_belief.yaw_marginal_var();
    // Localization/chain covariance J·Σ_chain·Jᵀ (computed in the fitter) — the table's room-frame
    // position is conditional on the robot pose, so its published uncertainty must include it.
    vx += inst.chain_cov_xx;
    vy += inst.chain_cov_yy;

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
    cov[3 * 6 + 3] = flat_rp_var;   // roll  (pinned flat on the floor)
    cov[4 * 6 + 4] = flat_rp_var;   // pitch (pinned flat on the floor)
    cov[5 * 6 + 5] = vyaw;   // yaw ← ψ

    G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov);
    G_->insert_or_assign_edge(edge.value());
    inst.last_pub_cov_trace = trace;

    // Verification readout: the controller-visible pose uncertainty published on the RT edge — ~cm
    // (Σ-mapped). Naturally throttled — the cov write self-gates above (>5 % trace change or republish).
    std::print("[{}] RT-cov σ x={:.1f}cm y={:.1f}cm z={:.1f}cm yaw={:.2f}° (src=Σ)\n",
               inst.node_name,
               100.0f * std::sqrt(std::max(0.0f, vx)), 100.0f * std::sqrt(std::max(0.0f, vy)),
               100.0f * std::sqrt(std::max(0.0f, vz)),
               57.2958f * std::sqrt(std::max(0.0f, vyaw)));
}

// ─── Mesh ────────────────────────────────────────────────────────────────────────────────────────

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

// ─── RT pose ─────────────────────────────────────────────────────────────────────────────────────

// Write the room→table RT edge (origin = base on the floor, z=0; yaw only), dead-banded below ~5 cm.
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

// ─── Epistemic proposal ──────────────────────────────────────────────────────────────────────────

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
