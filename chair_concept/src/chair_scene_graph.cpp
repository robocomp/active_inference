/*
 * chair_scene_graph.cpp — DSR node/RT I/O for chair_concept.
 */

#include "chair_scene_graph.h"

#include "../../common/rt_covariance/rt_covariance.h"   // rc::rtcov::publish (SHARED)

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <utility>

#include <QDebug>

#include "chair_model.h"   // rc::ChairModel statics (TOP_THICKNESS, LEG_RADIUS)
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc {

ChairSceneGraph::ChairSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 const ChairConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), cfg_(cfg), relayout_(std::move(relayout))
{}

std::uint64_t ChairSceneGraph::create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                              std::uint64_t room_node_id)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return 0;

    // Auto-name: one past the highest existing "chair_<N>". Chairs are now generic `object` nodes
    // named "chair_*" (schema migration), so scan get_nodes_by_type("object") + name-prefix filter.
    // ★NAMES ARE NEVER RECYCLED. Taking the max over LIVE nodes means a removal LOWERS the ceiling and the
    // next birth is handed the dead chair's number — so one name denotes two different objects, in one run.
    // Measured 2026-08-11: "chair_3" was born at (-2.16,-4.43), removed at L = -3.07, and the freed name was
    // re-issued to a birth at (-1.61,-2.80). In the CSV that reads as a single chair TELEPORTING 1.8 m, and
    // it cost three steps of analysis to notice the object had been swapped underneath the label. Any
    // history keyed on the name — the phantom log, the existence trace, the dashboard series — silently
    // splices two objects together.
    //
    // The high-water mark only ever rises, so a freed number stays retired. (Same defect and same reasoning
    // as door_concept's identity registry; see [[door-identity-name-recycling]].)
    // ⚠REMAINING GAP: this is per-RUN. Across a restart the mark is rebuilt from the live graph, so a name
    // freed before the restart can still be re-issued after it. door_concept persists its registry to
    // etc/door_identities.csv for exactly that reason; chair does not yet.
    int max_n = name_high_water_;
    for (const auto& n : G_->get_nodes_by_type("object"))
        if (n.name().rfind("chair_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(6))); } catch (...) {}
    name_high_water_ = max_n + 1;
    const std::string name = "chair_" + std::to_string(max_n + 1);

    // Generic `object` node named "chair_*"; class carried in object_subtype ("chair"). Every
    // get_nodes_by_type("object") MUST therefore be paired with a starts_with("chair") filter.
    DSR::Node chair_node = DSR::Node::create<object_node_type>(name);
    // Display asset for the voxelizer 3D viewer (relative to its meshes/ root); the viewer loads & scales it
    // to the fitted box (cortex mesh_path contract — the agent owns its appearance). Empty/missing asset
    // falls back to the fitted box.
    G_->add_or_modify_attrib_local<mesh_path_att>(chair_node, std::string("chair_concept/meshes/chair.obj"));
    // No base-colour texture → the viewer renders the chair in its flat class colour (color_for_category),
    // keeping it visually distinct from the table (which wears the gray mesa_1 wood texture).
    G_->add_or_modify_attrib_local<mesh_texture_path_att>(chair_node, std::string(""));
    G_->add_or_modify_attrib_local<width_m_att> (chair_node, cfg_.tracker_birth_seat_w);
    G_->add_or_modify_attrib_local<depth_m_att> (chair_node, cfg_.tracker_birth_seat_d);
    G_->add_or_modify_attrib_local<height_m_att>(chair_node, cfg_.tracker_birth_seat_h + cfg_.tracker_birth_back_h);
    G_->add_or_modify_attrib_local<object_subtype_att>(chair_node, std::string("chair"));  // type-agnostic consumers
    G_->add_or_modify_attrib_local<level_att>   (chair_node, 3);
    G_->add_or_modify_attrib_local<parent_att>  (chair_node, room_node_id);
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(chair_node, rpx + 150.f);
        G_->add_or_modify_attrib_local<pos_y_att>(chair_node, rpy +  50.f);
    }

    rc::provenance::stamp_creation(*G_, chair_node);   // birth stamp: epoch ms + local ISO-8601
    const auto id_opt = G_->insert_node(chair_node);
    if (not id_opt.has_value())
        return 0;

    rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                      {centroid_room.x(), centroid_room.y(), 0.0f}, {0.0f, 0.0f, 0.0f});

    if (relayout_)
        relayout_();

    std::print("chair_concept: [tracker] BIRTH '{}' id={} at room ({:.2f},{:.2f})\n",
               name, id_opt.value(), centroid_room.x(), centroid_room.y());
    return id_opt.value();
}

bool ChairSceneGraph::persist_chair_belief(ChairInstance& inst, std::uint64_t node_id,
                                           std::uint64_t room_id, float free_energy)
{
    auto node_opt = G_->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_write_model(inst, node_opt.value(), room_id, free_energy);
    return true;
}

void ChairSceneGraph::step_write_model(ChairInstance& inst, DSR::Node& node,
                                       std::uint64_t room_id, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish gate: only (re)write the geometry + mesh when the fitted dims/pose moved beyond a few
    // mm / mrad since the last publish. The voxelizer renders the MESH attribute (not the coarsely
    // dead-banded RT pose), rebuilt here every cycle from the raw fit — so writing it from the
    // sub-cm oscillating state each frame makes the viewer chair JITTER even though room→chair is
    // static. Freezing the mesh once settled removes the jitter. Mirrors bottle_concept's pub gate.
    constexpr float kPosEps = 0.003f;   // 3 mm
    constexpr float kYawEps = 0.005f;   // ~0.3°
    const bool geometry_changed =
        std::abs(s.cx - inst.last_pub_cx) > kPosEps or
        std::abs(s.cy - inst.last_pub_cy) > kPosEps or
        std::abs(s.seat_w - inst.last_pub_w) > kPosEps or
        std::abs(s.seat_d - inst.last_pub_h) > kPosEps or
        std::abs((s.seat_h + s.back_h) - inst.last_pub_H) > kPosEps or
        std::abs(static_cast<float>(std::remainder(s.yaw - inst.last_pub_yaw, 2.0 * M_PI))) > kYawEps;

    // Geometry attributes + mesh (gated to kill the viewer jitter once settled).
    // Map chair fields to the standard DSR geometry attrs consumers read:
    //   width_m←seat_w, depth_m←seat_d, height_m←overall height (seat_h + back_h).
    if (geometry_changed)
    {
        G_->add_or_modify_attrib_local<width_m_att> (node, s.seat_w);
        G_->add_or_modify_attrib_local<depth_m_att> (node, s.seat_d);
        G_->add_or_modify_attrib_local<height_m_att>(node, s.seat_h + s.back_h);
        G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
        write_chair_mesh(inst, node);   // mesh for the voxelizer 3D viewer
        inst.last_pub_cx = s.cx; inst.last_pub_cy = s.cy;
        inst.last_pub_w  = s.seat_w;  inst.last_pub_h  = s.seat_d;
        inst.last_pub_H  = s.seat_h + s.back_h; inst.last_pub_yaw = s.yaw;
    }
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);

    // The accumulated support bank is NOT published. It never was voxels: the points are the 3-D support of
    // a segmentation mask, split by this model's own SDF into on- and off-surface. The graph export was up to
    // SupportBankMaxPoints (4000) points x 3 floats, per object, per publish, into a CRDT graph that nothing
    // read — the same shape as the unbounded dot cloud that pinned an agent at 100% CPU. Gated off 2026-08-14,
    // and the *_voxel_bank_pts attribute registrations were removed from cortex on 2026-08-14, so the gate,
    // the knob and the attribute are all gone rather than lying dormant.
    //
    // THE BANK ITSELF STAYS and is load-bearing: evaluate_shape() fits the round hypothesis to
    // inst.support_bank_pts for the round-vs-square model selection, and DumpCloudPath exports it for the
    // offline harness. Both are local reads. It was only ever the graph traffic that had no consumer.


    // Latest residual points (model-unexplained) for the voxelizer's residual layer — it reads
    // residual_pts_att but nothing was writing it, so that layer was always empty.
    {
        std::vector<float> res_flat;
        res_flat.reserve(inst.last_residual_pts.size() * 3);
        for (const auto& p : inst.last_residual_pts) { res_flat.push_back(p.x()); res_flat.push_back(p.y()); res_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<residual_pts_att>(node, res_flat);
    }

    // Active-perception channel for the controller's local lock-on search:
    //  - chair_roi_offset [ox, oy]: normalised image-centre offset of the projected model
    //    (drive →0 to centre the chair in the frame).
    //  - chair_roi_fill: projected extent fraction (drive toward a sweet-spot for stand-off/scale).
    //  - chair_roi_valid: model currently projects in front of the camera.
    //  - chair_detection_alive / _confidence / _frames_since: is YOLO firing here, and how strongly.
    G_->add_or_modify_attrib_local<chair_roi_offset_att>(node,
        std::vector<float>{inst.roi_offset_x, inst.roi_offset_y});
    G_->add_or_modify_attrib_local<chair_roi_fill_att>(node, inst.roi_fill);
    G_->add_or_modify_attrib_local<chair_roi_valid_att>(node, inst.roi_valid);
    G_->add_or_modify_attrib_local<chair_detection_alive_att>(node, inst.detection_alive);
    G_->add_or_modify_attrib_local<chair_detection_confidence_att>(node, inst.last_mask_confidence);
    G_->add_or_modify_attrib_local<chair_frames_since_detection_att>(node, inst.frames_since_detection);

    // Inferred albedo tint for the display mesh (DISPLAY ONLY — see common/appearance_belief). Published
    // as CHROMATICITY, pre-scaled by the belief's confidence toward neutral grey, so the colour fades IN
    // as evidence accumulates instead of popping to whatever the first frame's lighting happened to be.
    // Not published at all until the belief has seen a frame, which leaves the viewer on the asset's
    // authored colours — the correct default when we know nothing. Written here (per-cycle) rather than
    // at node creation because the belief keeps moving as new views arrive.
    if (inst.appearance.initialized())
    {
        const float           c       = inst.appearance.confidence();
        const Eigen::Vector3f neutral = Eigen::Vector3f::Constant(1.0f / 3.0f);
        const Eigen::Vector3f tint    = neutral + c * (inst.appearance.map() - neutral);
        G_->add_or_modify_attrib_local<mesh_color_rgb_att>(node,
            std::vector<float>{tint.x(), tint.y(), tint.z()});
    }

    G_->update_node(node);

    write_rt_pose(room_id, inst);
    // Upload the chair pose covariance after the pose write (so a rare >5 cm RT recreate doesn't
    // clobber it). `force` on a geometry republish; otherwise the write self-gates on a meaningful
    // uncertainty change, so a stationary-but-tightening chair stays current without edge churn.
    write_rt_covariance(room_id, inst, geometry_changed);
}

void ChairSceneGraph::write_rt_covariance(std::uint64_t room_id, ChairInstance& inst, bool force)
{
    if (not cfg_.rt_cov_upload or room_id == 0)
        return;

    const float scale = std::max(1e-6f, cfg_.rt_cov_scale);

    if (not inst.ai2_initialized)
        return;   // belief not seeded yet — nothing calibrated to publish

    // The belief carries a 3×3 Σ over [cx,cy,yaw] (pose-only; size is a fixed template). z is pinned to the
    // floor → its uncertainty is the floor-height std (not a DOF). Use covariance_REPORTED so the yaw term
    // folds in the discrete orientation-mode entropy — a side-on chair (backrest unseen) advertises an honest
    // large σ_yaw to the controller instead of a falsely-confident heading (mirrors table's reported cov).
    const auto S = inst.ai2_belief.covariance_reported();
    float vx   = scale * S(0, 0);   // cx
    float vy   = scale * S(1, 1);   // cy
    float vz   = inst.ai2_belief.params().floor_std * inst.ai2_belief.params().floor_std;   // cz pinned
    float vyaw = scale * S(2, 2);   // yaw
    // Localization/chain covariance J·Σ_chain·Jᵀ — the chair's room-frame position is conditional on the
    // robot pose, so its published uncertainty must include it.
    vx += inst.chain_cov_xx;
    vy += inst.chain_cov_yy;

    // Everything from here — the self-gate, the 6×6 block layout, the edge write, the trace bookkeeping
    // and the readout — is the SHARED publisher (common/rt_covariance/rt_covariance.h). What stays above is
    // the only part that is genuinely this object's: the mapping from its own DOF to the six variances.
    const rc::rtcov::Se3Var v{vx, vy, vz, rc::rtcov::kFlatRollPitchVar, rc::rtcov::kFlatRollPitchVar, vyaw};
    rc::rtcov::publish(*G_, room_id, inst.node_id, v, inst.last_pub_cov_trace, force, inst.node_name);
}

std::vector<float> ChairSceneGraph::make_chair_mesh(const ChairState& s)
{
    // Flat triangle list (room frame): seat slab + backrest + 4 square legs = 6 boxes × 12 tri.
    std::vector<float> verts;
    verts.reserve(6 * 108);

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

    const float hw = s.seat_w * 0.5f;
    const float hd = s.seat_d * 0.5f;
    const float ht = ChairModel::SEAT_THICKNESS * 0.5f;

    // Seat slab — its TOP face sits at cz + seat_h, so its centre is half a thickness below.
    push_box(s.cx, s.cy, s.cz + s.seat_h - ht, hw, hd, ht);

    // Backrest — a thin slab along the -y edge (local), rising back_h above the seat top.
    const float back_cy = -hd + ht;               // local y of backrest centre
    const float back_cz = s.cz + s.seat_h + s.back_h * 0.5f;
    {
        const float brx = s.cx + cy * 0.0f - sy * back_cy;
        const float bry = s.cy + sy * 0.0f + cy * back_cy;
        push_box(brx, bry, back_cz, hw, ht, s.back_h * 0.5f);
    }

    // 4 legs — square cross-section (2×LEG_HALF), at the seat corners, floor → seat underside.
    const float lr  = ChairModel::LEG_HALF;
    const float leg_top = s.seat_h - ChairModel::SEAT_THICKNESS;   // underside of the seat slab
    const float lhz = std::max(0.01f, leg_top) * 0.5f;
    for (int ix : {-1, 1})
        for (int iy : {-1, 1})
        {
            const float lx = ix * (hw - lr);
            const float ly = iy * (hd - lr);
            const float rx = s.cx + cy * lx - sy * ly;
            const float ry = s.cy + sy * lx + cy * ly;
            push_box(rx, ry, s.cz + lhz, lr, lr, lhz);
        }

    return verts;
}

void ChairSceneGraph::write_chair_mesh(ChairInstance& inst, DSR::Node& node)
{
    const std::vector<float> verts = make_chair_mesh(inst.model.state());
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, verts);
}

void ChairSceneGraph::write_rt_pose(std::uint64_t room_id, ChairInstance& inst)
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

    // Chair node origin = BASE on the floor (z=0), NOT the mid-height. Every consumer assumes a
    // base origin: the voxelizer box (z∈[origin, origin+height]), and bottle_concept's chair-top
    // lookup + support decision (top = origin.z + height). Publishing z=chair_height/2 put the
    // origin at mid-height → chair_top came out as 1.5·height → bottles failed the support test,
    // parented to the room and floated. Keep the base on the floor so top = 0 + height = height.
    const float z = 0.0f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {s.cx, s.cy, z},
                                      {0.0f, 0.0f, s.yaw});
    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

void ChairSceneGraph::write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop)
{
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>  (node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>  (node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>        (node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>     (node, true);
    G_->update_node(node);
}

}  // namespace rc
