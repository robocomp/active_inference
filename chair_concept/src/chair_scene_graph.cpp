/*
 * chair_scene_graph.cpp — DSR node/RT I/O for chair_concept.
 */

#include "chair_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <utility>

#include <QDebug>

#include "chair_model.h"   // rc::ChairModel statics (TOP_THICKNESS, LEG_RADIUS)

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

    // Auto-name: one past the highest existing "chair_<N>".
    int max_n = 0;
    for (const auto& n : G_->get_nodes_by_type("chair"))
        if (n.name().rfind("chair_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(6))); } catch (...) {}
    const std::string name = "chair_" + std::to_string(max_n + 1);

    DSR::Node chair_node = DSR::Node::create<chair_node_type>(name);
    G_->add_or_modify_attrib_local<width_m_att> (chair_node, cfg_.tracker_birth_seat_w);
    G_->add_or_modify_attrib_local<depth_m_att> (chair_node, cfg_.tracker_birth_seat_d);
    G_->add_or_modify_attrib_local<height_m_att>(chair_node, cfg_.tracker_birth_seat_h + cfg_.tracker_birth_back_h);
    G_->add_or_modify_attrib_local<level_att>   (chair_node, 3);
    G_->add_or_modify_attrib_local<parent_att>  (chair_node, room_node_id);
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(chair_node, rpx + 150.f);
        G_->add_or_modify_attrib_local<pos_y_att>(chair_node, rpy +  50.f);
    }

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

    // Export full chair-owned voxel memory (room frame) as XYZ triples.
    {
        std::vector<float> bank_flat;
        bank_flat.reserve(inst.voxel_bank_pts.size() * 3);
        for (const auto& p : inst.voxel_bank_pts) { bank_flat.push_back(p.x()); bank_flat.push_back(p.y()); bank_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<chair_voxel_bank_pts_att>(node, bank_flat);
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
    constexpr float big = 1e3f;   // unobservable / never-seen DOF → large variance

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
    cov[2 * 6 + 2] = vz;     // z   ← cz (floor height)
    cov[3 * 6 + 3] = big;   // roll  (unobservable)
    cov[4 * 6 + 4] = big;   // pitch (unobservable)
    cov[5 * 6 + 5] = vyaw;   // yaw ← ψ

    G_->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov);
    G_->insert_or_assign_edge(edge.value());
    inst.last_pub_cov_trace = trace;
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
