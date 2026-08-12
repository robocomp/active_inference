/*
 * refrigerator_scene_graph.cpp — DSR node/RT I/O for refrigerator_concept.
 *
 * Implements RefrigeratorSceneGraph: births "refrigerator_N" nodes from tracker detections and, each cycle, writes the
 * fitted model back to the graph — geometry attrs + mesh + residual/voxel-bank export + free energy, the
 * room→refrigerator RT pose (dead-banded), and the 6×6 pose covariance mapped from the belief Σ (with a flat
 * roll/pitch prior). Writes self-gate on a meaningful geometry/uncertainty change to avoid per-cycle edge
 * churn. All graph access is on the main thread (SpecificWorker's compute path).
 */

#include "refrigerator_scene_graph.h"

#include "../../common/rt_covariance/rt_covariance.h"   // rc::rtcov::publish (SHARED)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <print>
#include <utility>

#include <QDebug>

#include "refrigerator_model.h"   // rc::RefrigeratorModel statics (TOP_THICKNESS, LEG_RADIUS)
#include "../../common/object_anchor/object_anchor_contract.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc {

RefrigeratorSceneGraph::RefrigeratorSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 const RefrigeratorConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), cfg_(cfg), relayout_(std::move(relayout))
{}

// ─── Node birth ──────────────────────────────────────────────────────────────────────────────────

std::uint64_t RefrigeratorSceneGraph::create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                              std::uint64_t room_node_id)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return 0;

    // Auto-name: one past the highest existing "refrigerator_<N>".
        // ★NAMES ARE NEVER RECYCLED. Taking the max over LIVE nodes means a removal LOWERS the ceiling and
    // the next birth is handed the dead object's number — so one name denotes two different objects
    // within a single run, and any history keyed on the name (phantom logs, existence traces, dashboard
    // series, a level-2 rig's membership) silently splices them together. It reads as a TELEPORT rather
    // than an identity swap, which is what makes it expensive to notice. Same defect and fix as
    // chair_concept 9aef57a / table_concept; door_concept solves it with a persistent registry instead.
    // ⚠Per-RUN only: across a restart the mark is rebuilt from the live graph, so a name freed before
    // the restart can still be re-issued after it (door persists etc/door_identities.csv for that).
    int max_n = name_high_water_;
    // DSR node type is the GENERIC "object" (cortex REGISTER_NODE_TYPE(object)); the CLASS discriminator is the
    // name prefix "refrigerator_" + the object_subtype string attribute (mirrors cabinet_concept's box/cabinet_*).
    for (const auto& n : G_->get_nodes_by_type("object"))
        if (n.name().rfind("refrigerator_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(13))); } catch (...) {}
    name_high_water_ = max_n + 1;
    const std::string name = "refrigerator_" + std::to_string(max_n + 1);

    DSR::Node refrigerator_node = DSR::Node::create<object_node_type>(name);
    G_->add_or_modify_attrib_local<object_subtype_att>(refrigerator_node, std::string("refrigerator"));
    // Display asset for the voxelizer 3D viewer (relative to its meshes/ root). The agent owns its
    // appearance; the viewer loads & scales this to the fitted box (see cortex mesh_path contract).
    G_->add_or_modify_attrib_local<mesh_path_att>(refrigerator_node, std::string("refrigerator_concept/meshes/fridge.obj"));
    G_->add_or_modify_attrib_local<mesh_texture_path_att>(refrigerator_node, std::string("refrigerator_concept/meshes/fridge_basecolor.jpg"));
    G_->add_or_modify_attrib_local<width_m_att> (refrigerator_node, cfg_.tracker_birth_width_m);
    G_->add_or_modify_attrib_local<depth_m_att> (refrigerator_node, cfg_.tracker_birth_depth_m);
    G_->add_or_modify_attrib_local<height_m_att>(refrigerator_node, cfg_.tracker_birth_height_m);
    G_->add_or_modify_attrib_local<level_att>   (refrigerator_node, 3);
    G_->add_or_modify_attrib_local<parent_att>  (refrigerator_node, room_node_id);
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(refrigerator_node, rpx + 150.f);
        G_->add_or_modify_attrib_local<pos_y_att>(refrigerator_node, rpy +  50.f);
    }

    rc::provenance::stamp_creation(*G_, refrigerator_node);   // birth stamp: epoch ms + local ISO-8601
    const auto id_opt = G_->insert_node(refrigerator_node);
    if (not id_opt.has_value())
        return 0;

    const float z = cfg_.tracker_birth_height_m * 0.5f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                      {centroid_room.x(), centroid_room.y(), z}, {0.0f, 0.0f, 0.0f});

    if (relayout_)
        relayout_();

    std::print("refrigerator_concept: [tracker] BIRTH '{}' id={} at room ({:.2f},{:.2f})\n",
               name, id_opt.value(), centroid_room.x(), centroid_room.y());
    return id_opt.value();
}

// ─── Model publish (geometry · mesh · residual/voxel export · RT · covariance) ───────────────────

// Resolve the refrigerator node by id and write the fitted model to it (false if the node is gone).
bool RefrigeratorSceneGraph::persist_refrigerator_belief(RefrigeratorInstance& inst, std::uint64_t node_id,
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
void RefrigeratorSceneGraph::step_write_model(RefrigeratorInstance& inst, DSR::Node& node,
                                       std::uint64_t room_id, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish gate: only (re)write the geometry + mesh when the fitted dims/pose moved beyond a few
    // mm / mrad since the last publish. The voxelizer renders the MESH attribute (not the coarsely
    // dead-banded RT pose), rebuilt here every cycle from the raw fit — so writing it from the
    // sub-cm oscillating state each frame makes the viewer refrigerator JITTER even though room→refrigerator is
    // static. Freezing the mesh once settled removes the jitter. Mirrors bottle_concept's pub gate.
    constexpr float kPosEps = 0.003f;   // 3 mm
    constexpr float kYawEps = 0.005f;   // ~0.3°
    const bool geometry_changed =
        std::abs(s.cx - inst.last_pub_cx) > kPosEps or
        std::abs(s.cy - inst.last_pub_cy) > kPosEps or
        std::abs(s.w  - inst.last_pub_w)  > kPosEps or
        std::abs(s.h  - inst.last_pub_h)  > kPosEps or
        std::abs(s.refrigerator_height - inst.last_pub_H) > kPosEps or
        std::abs(static_cast<float>(std::remainder(s.yaw - inst.last_pub_yaw, 2.0 * M_PI))) > kYawEps;

    // Geometry attributes + mesh (gated to kill the viewer jitter once settled).
    if (geometry_changed)
    {
        G_->add_or_modify_attrib_local<width_m_att> (node, s.w);
        G_->add_or_modify_attrib_local<depth_m_att> (node, s.h);
        G_->add_or_modify_attrib_local<height_m_att>(node, s.refrigerator_height);
        G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
        write_refrigerator_mesh(inst, node);   // mesh for the voxelizer 3D viewer
        inst.last_pub_cx = s.cx; inst.last_pub_cy = s.cy;
        inst.last_pub_w  = s.w;  inst.last_pub_h  = s.h;
        inst.last_pub_H  = s.refrigerator_height; inst.last_pub_yaw = s.yaw;
    }
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    // Inferred shape subtype (round/square), chosen by free-energy model evidence in RefrigeratorFitter::evaluate_shape.
    // The voxelizer reads this to render the matching mesh (disc vs box).
    G_->add_or_modify_attrib_local<object_subtype_att>(node, inst.subtype);

    // Export full refrigerator-owned voxel memory (room frame) as XYZ triples.
    {
        std::vector<float> bank_flat;
        bank_flat.reserve(inst.voxel_bank_pts.size() * 3);
        for (const auto& p : inst.voxel_bank_pts) { bank_flat.push_back(p.x()); bank_flat.push_back(p.y()); bank_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<table_voxel_bank_pts_att>(node, bank_flat);
    }

    // DIAGNOSTIC one-shot: dump this refrigerator's accumulated voxel-bank cloud (room frame) to a file for the
    // offline square-vs-round model comparison (tests/compare_models). Gated on a config path + a populated
    // bank; fires exactly once per instance. No effect unless RefrigeratorConcept.DumpCloudPath is set.
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
    //    (drive →0 to centre the refrigerator in the frame).
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
    // Upload the refrigerator pose covariance after the pose write (so a rare >5 cm RT recreate doesn't
    // clobber it). `force` on a geometry republish; otherwise the write self-gates on a meaningful
    // uncertainty change, so a stationary-but-tightening refrigerator stays current without edge churn.
    write_rt_covariance(room_id, inst, geometry_changed);
}

// Map the belief's 6×6 Σ [cx,cy,H,w,h,yaw] onto the room→refrigerator RT edge covariance (+ chain term), with
// roll/pitch pinned small by the flat-on-the-floor prior and yaw as the marginal (mode-entropy) variance.
void RefrigeratorSceneGraph::write_rt_covariance(std::uint64_t room_id, RefrigeratorInstance& inst, bool force)
{
    if (room_id == 0)
        return;

    const float scale = std::max(1e-6f, cfg_.rt_cov_scale);
    // A refrigerator rests FLAT on the floor: roll/pitch are pinned to ~0 by that physical prior — CONFIDENTLY known,
    // not unknown. The old 1e3 ("unobservable → huge variance") was backwards: it told the controller the refrigerator
    // might be tilted AND it dominated the published-covariance plot scale so the real sub-1 DOF read as ~0.
    // Publish a small flat-refrigerator variance instead (≈1.3° std), consistent with the geometry.

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
    // near-square refrigerator publishes an honest ~45° (not the overconfident ~1° within-mode width) so the
    // controller's uncertainty governor + epistemic planner know yaw is unresolved. Collapses as evidence
    // resolves the mode. See REFRIGERATOR.md.
    float vyaw = scale * inst.ai2_belief.yaw_marginal_var();
    // Localization/chain covariance J·Σ_chain·Jᵀ (computed in the fitter) — the refrigerator's room-frame
    // position is conditional on the robot pose, so its published uncertainty must include it.
    vx += inst.chain_cov_xx;
    vy += inst.chain_cov_yy;

    // Everything from here — the self-gate, the 6×6 block layout, the edge write, the trace bookkeeping
    // and the readout — is the SHARED publisher (common/rt_covariance/rt_covariance.h). What stays above is
    // the only part that is genuinely this object's: the mapping from its own DOF to the six variances.
    const rc::rtcov::Se3Var v{vx, vy, vz, rc::rtcov::kFlatRollPitchVar, rc::rtcov::kFlatRollPitchVar, vyaw};
    rc::rtcov::publish(*G_, room_id, inst.node_id, v, inst.last_pub_cov_trace, force, inst.node_name);
}

// ─── Mesh ────────────────────────────────────────────────────────────────────────────────────────

std::vector<float> RefrigeratorSceneGraph::make_refrigerator_mesh(const RefrigeratorState& s)
{
    // Flat triangle list (room frame): ONE solid floor-anchored box (12 tri). A refrigerator is a single
    // cuboid — no top slab, no legs.
    std::vector<float> verts;
    verts.reserve(108);

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

    // Single solid box: centred at (cx, cy, H/2), half extents (w/2, h/2, H/2) — spans floor→H.
    const float halfH = s.refrigerator_height * 0.5f;
    push_box(s.cx, s.cy, halfH, s.w * 0.5f, s.h * 0.5f, halfH);

    return verts;
}

void RefrigeratorSceneGraph::write_refrigerator_mesh(RefrigeratorInstance& inst, DSR::Node& node)
{
    const std::vector<float> verts = make_refrigerator_mesh(inst.model.state());
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, verts);
}

// ─── RT pose ─────────────────────────────────────────────────────────────────────────────────────

// Write the room→refrigerator RT edge (origin = base on the floor, z=0; yaw only), dead-banded below ~5 cm.
void RefrigeratorSceneGraph::write_rt_pose(std::uint64_t room_id, RefrigeratorInstance& inst)
{
    if (room_id == 0 or not rt_api_)
        return;

    const auto& s = inst.model.state();

    // Dead-band: suppress RT edge updates below ~5 cm to avoid pos churn from gradient oscillations.
    // Dead-band: suppress RT edge updates below ~5 cm of motion AT THE FOOTPRINT CORNER — one statement
    // covering translation AND rotation, so an object that turns in place republishes its yaw. The centre-only
    // form was blind to pure rotation: hood_concept sat 9° off its wall while its centre drifted 5 mm, and no
    // consumer ever saw the corrected heading. See rc::rtcov::rt_pose_moved.
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (not rc::rtcov::rt_pose_moved(dx, dy, s.yaw - inst.last_written_yaw,
                                     0.5f * std::hypot(s.w, s.h)))
        return;

    auto room_opt = G_->get_node(room_id);
    if (not room_opt.has_value())
        return;

    // Refrigerator node origin = BASE on the floor (z=0), NOT the mid-height. Every consumer assumes a
    // base origin: the voxelizer box (z∈[origin, origin+height]), and bottle_concept's refrigerator-top
    // lookup + support decision (top = origin.z + height). Publishing z=refrigerator_height/2 put the
    // origin at mid-height → refrigerator_top came out as 1.5·height → bottles failed the support test,
    // parented to the room and floated. Keep the base on the floor so top = 0 + height = height.
    const float z = 0.0f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {s.cx, s.cy, z},
                                      {0.0f, 0.0f, s.yaw});
    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
    inst.last_written_yaw = s.yaw;
}

// ─── Epistemic proposal ──────────────────────────────────────────────────────────────────────────

void RefrigeratorSceneGraph::write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop)
{
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>  (node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>  (node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>        (node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>     (node, true);
    G_->update_node(node);
}

}  // namespace rc
