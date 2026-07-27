/*
 * door_scene_graph.cpp — DSR node/RT I/O for door_concept.
 */

#include "door_scene_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <print>
#include <utility>

#include <QDebug>

#include "door_model.h"   // rc::DoorModel statics (TOP_THICKNESS, LEG_RADIUS)

namespace rc {

DoorSceneGraph::DoorSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                                 DSR::RT_API* rt_api,
                                 const DoorConfig& cfg,
                                 std::function<void()> relayout)
    : G_(std::move(graph)), rt_api_(rt_api), cfg_(cfg), relayout_(std::move(relayout))
{}

namespace
{
// Door pose (room frame: centre cx,cy; base on the floor z=0; heading yaw_room) expressed in the wall
// node's local frame (origin = wall midpoint at height z0, x = wall tangent yaw_w). Returns the RT-edge
// translation (along-wall, across-wall, base-below-origin) + the door's yaw relative to the wall.
struct LocalPose { float x, y, z, yaw; };
LocalPose door_in_wall(float cx, float cy, float yaw_room,
                       const Eigen::Vector2f& mid, float yaw_w, float z0)
{
    const float c = std::cos(yaw_w), s = std::sin(yaw_w);
    const float rx = cx - mid.x(), ry = cy - mid.y();
    const float lyaw = std::remainder(yaw_room - yaw_w, 2.0f * static_cast<float>(M_PI));
    return { c * rx + s * ry, -s * rx + c * ry, 0.0f - z0, lyaw };
}
}  // namespace

// Pick the "wall_*" node whose segment is nearest the door centre. Each wall node carries a room→wall RT
// edge (midpoint + tangent yaw) and width_m (length); we clamp-project the door centre onto every wall
// segment and keep the closest. ok=false ⇒ no wall nodes yet (room_concept not up) → caller hangs from room.
DoorSceneGraph::WallRef DoorSceneGraph::resolve_wall(std::uint64_t room_id, const Eigen::Vector2f& door_xy) const
{
    WallRef best;
    float best_d2 = std::numeric_limits<float>::max();
    for (const auto& wn : G_->get_nodes_by_type("wall"))
    {
        const auto edge = G_->get_edge(room_id, wn.id(), "RT");
        if (not edge.has_value()) continue;
        const auto tr  = G_->get_attrib_by_name<rt_translation_att>(edge.value());
        const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value());
        if (not tr.has_value() or not rot.has_value()) continue;
        const auto& t = tr.value().get();
        const auto& r = rot.value().get();
        if (t.size() < 3 or r.size() < 3) continue;
        const Eigen::Vector2f mid(t[0], t[1]);
        const float yaw = r[2];
        float L = 1.0f;
        if (const auto w = G_->get_attrib_by_name<width_m_att>(wn); w.has_value()) L = w.value();
        const Eigen::Vector2f dir(std::cos(yaw), std::sin(yaw));
        const float proj = std::clamp((door_xy - mid).dot(dir), -0.5f * L, 0.5f * L);
        const float d2 = (door_xy - (mid + proj * dir)).squaredNorm();
        if (d2 < best_d2)
        {
            best_d2 = d2;
            best = WallRef{wn.id(), mid, yaw, t[2], L, true};
        }
    }
    return best;
}

std::uint64_t DoorSceneGraph::create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                              std::uint64_t room_node_id)
{
    auto room_opt = G_->get_node(room_node_id);
    if (not room_opt.has_value())
        return 0;

    // Auto-name: one past the highest existing "door_<N>". Doors are now generic `object` nodes
    // named "door_*" (schema migration), so scan get_nodes_by_type("object") + name-prefix filter.
    int max_n = 0;
    for (const auto& n : G_->get_nodes_by_type("object"))
        if (n.name().rfind("door_", 0) == 0)
            try { max_n = std::max(max_n, std::stoi(n.name().substr(5))); } catch (...) {}   // "door_" = 5 chars
    const std::string name = "door_" + std::to_string(max_n + 1);

    // Generic `object` node named "door_*"; class carried in object_subtype ("door"). Every
    // get_nodes_by_type("object") MUST therefore be paired with a starts_with("door") filter.
    DSR::Node door_node = DSR::Node::create<object_node_type>(name);
    // Display asset for the voxelizer 3D viewer (relative to its meshes/ root); the viewer loads & scales it
    // to the fitted box (cortex mesh_path contract — the agent owns its appearance). Empty/missing asset
    // falls back to the fitted box.
    G_->add_or_modify_attrib_local<mesh_path_att>(door_node, std::string("door_concept/meshes/door.obj"));
    G_->add_or_modify_attrib_local<mesh_texture_path_att>(door_node, std::string("door_concept/meshes/door_basecolor.jpg"));
    G_->add_or_modify_attrib_local<width_m_att> (door_node, cfg_.door_prior_w_m);
    G_->add_or_modify_attrib_local<depth_m_att> (door_node, cfg_.door_thickness_m);
    G_->add_or_modify_attrib_local<height_m_att>(door_node, cfg_.door_prior_h_m);
    G_->add_or_modify_attrib_local<object_subtype_att>(door_node, std::string("door"));  // type-agnostic consumers

    // A door HANGS FROM ITS WALL: parent the node to the nearest "wall_*" node (its RT pose is then the door
    // in the wall frame), falling back to the room only if no wall nodes exist yet. write_rt_pose keeps it
    // on the right wall each cycle.
    const WallRef wall = resolve_wall(room_node_id, {centroid_room.x(), centroid_room.y()});
    const std::uint64_t parent_id = wall.ok ? wall.id : room_node_id;
    G_->add_or_modify_attrib_local<level_att> (door_node, wall.ok ? 5 : 3);
    G_->add_or_modify_attrib_local<parent_att>(door_node, parent_id);
    {
        const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
        const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
        G_->add_or_modify_attrib_local<pos_x_att>(door_node, rpx + 150.f);
        G_->add_or_modify_attrib_local<pos_y_att>(door_node, rpy +  50.f);
    }

    const auto id_opt = G_->insert_node(door_node);
    if (not id_opt.has_value())
        return 0;

    if (wall.ok)
    {
        if (auto wnode = G_->get_node(wall.id); wnode.has_value())
        {
            // At birth the heading is unknown → align the door to the wall (lyaw = 0); the fit refines it.
            const auto lp = door_in_wall(centroid_room.x(), centroid_room.y(), wall.yaw, wall.mid, wall.yaw, wall.z0);
            rt_api_->insert_or_assign_edge_RT(wnode.value(), id_opt.value(),
                                              {lp.x, lp.y, lp.z}, {0.0f, 0.0f, 0.0f});
        }
    }
    else
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                          {centroid_room.x(), centroid_room.y(), 0.0f}, {0.0f, 0.0f, 0.0f});

    if (relayout_)
        relayout_();

    std::print("door_concept: [tracker] BIRTH '{}' id={} at room ({:.2f},{:.2f}) parent={} ({})\n",
               name, id_opt.value(), centroid_room.x(), centroid_room.y(),
               parent_id, wall.ok ? "wall" : "room");
    return id_opt.value();
}

bool DoorSceneGraph::persist_door_belief(DoorInstance& inst, std::uint64_t node_id,
                                           std::uint64_t room_id, float free_energy)
{
    auto node_opt = G_->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_write_model(inst, node_opt.value(), room_id, free_energy);
    return true;
}

void DoorSceneGraph::step_write_model(DoorInstance& inst, DSR::Node& node,
                                       std::uint64_t room_id, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish gate: only (re)write the geometry + mesh when the fitted dims/pose moved beyond a few
    // mm / mrad since the last publish. The voxelizer renders the MESH attribute (not the coarsely
    // dead-banded RT pose), rebuilt here every cycle from the raw fit — so writing it from the
    // sub-cm oscillating state each frame makes the viewer door JITTER even though room→door is
    // static. Freezing the mesh once settled removes the jitter. Mirrors bottle_concept's pub gate.
    constexpr float kPosEps = 0.003f;   // 3 mm
    constexpr float kYawEps = 0.005f;   // ~0.3°
    const bool geometry_changed =
        std::abs(s.cx - inst.last_pub_cx) > kPosEps or
        std::abs(s.cy - inst.last_pub_cy) > kPosEps or
        std::abs(s.w - inst.last_pub_w) > kPosEps or
        std::abs(s.thickness - inst.last_pub_h) > kPosEps or
        std::abs(s.h - inst.last_pub_H) > kPosEps or
        std::abs(static_cast<float>(std::remainder(s.yaw - inst.last_pub_yaw, 2.0 * M_PI))) > kYawEps;

    // Geometry attributes + mesh (gated to kill the viewer jitter once settled).
    // Map the door panel to the standard DSR geometry attrs consumers read:
    //   width_m←w (along wall), depth_m←thickness (across wall), height_m←h.
    if (geometry_changed)
    {
        G_->add_or_modify_attrib_local<width_m_att> (node, s.w);
        G_->add_or_modify_attrib_local<depth_m_att> (node, s.thickness);
        G_->add_or_modify_attrib_local<height_m_att>(node, s.h);
        G_->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
        write_door_mesh(inst, node);   // mesh for the voxelizer 3D viewer
        inst.last_pub_cx = s.cx; inst.last_pub_cy = s.cy;
        inst.last_pub_w  = s.w;  inst.last_pub_h  = s.thickness;
        inst.last_pub_H  = s.h;  inst.last_pub_yaw = s.yaw;
    }
    G_->add_or_modify_attrib_local<free_energy_att>(node, free_energy);

    // Export full door-owned voxel memory (room frame) as XYZ triples.
    {
        std::vector<float> bank_flat;
        bank_flat.reserve(inst.voxel_bank_pts.size() * 3);
        for (const auto& p : inst.voxel_bank_pts) { bank_flat.push_back(p.x()); bank_flat.push_back(p.y()); bank_flat.push_back(p.z()); }
        G_->add_or_modify_attrib_local<door_voxel_bank_pts_att>(node, bank_flat);
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
    //  - door_roi_offset [ox, oy]: normalised image-centre offset of the projected model
    //    (drive →0 to centre the door in the frame).
    //  - door_roi_fill: projected extent fraction (drive toward a sweet-spot for stand-off/scale).
    //  - door_roi_valid: model currently projects in front of the camera.
    //  - door_detection_alive / _confidence / _frames_since: is YOLO firing here, and how strongly.
    G_->add_or_modify_attrib_local<door_roi_offset_att>(node,
        std::vector<float>{inst.roi_offset_x, inst.roi_offset_y});
    G_->add_or_modify_attrib_local<door_roi_fill_att>(node, inst.roi_fill);
    G_->add_or_modify_attrib_local<door_roi_valid_att>(node, inst.roi_valid);
    G_->add_or_modify_attrib_local<door_detection_alive_att>(node, inst.detection_alive);
    G_->add_or_modify_attrib_local<door_detection_confidence_att>(node, inst.last_mask_confidence);
    G_->add_or_modify_attrib_local<door_frames_since_detection_att>(node, inst.frames_since_detection);

    G_->update_node(node);

    write_rt_pose(room_id, inst);
    // Upload the door pose covariance after the pose write (so a rare >5 cm RT recreate doesn't
    // clobber it). `force` on a geometry republish; otherwise the write self-gates on a meaningful
    // uncertainty change, so a stationary-but-tightening door stays current without edge churn.
    write_rt_covariance(room_id, inst, geometry_changed);
}

void DoorSceneGraph::write_rt_covariance(std::uint64_t room_id, DoorInstance& inst, bool force)
{
    if (not cfg_.rt_cov_upload or room_id == 0)
        return;

    const float scale = std::max(1e-6f, cfg_.rt_cov_scale);
    constexpr float big = 1e3f;   // unobservable / never-seen DOF → large variance

    if (not inst.ai2_initialized)
        return;   // belief not seeded yet — nothing calibrated to publish

    // The belief carries a 3×3 Σ over the WALL-FRAME [s,w,h]. Only s (the along-wall offset) is a position
    // DOF; yaw is FIXED by the (trusted, nominal) wall (small floor variance), z is pinned to the floor,
    // and w,h are SIZE (not pose) so they don't enter the RT-edge pose covariance. The RT edge is expressed
    // in the PARENT frame, so when the door hangs from its wall the covariance goes in WALL-LOCAL axes
    // (s = local x along the wall); in the room-frame fallback it is the rank-1 σ_s²·(u uᵀ) along u.
    const std::uint64_t parent = inst.wall_node_id != 0 ? inst.wall_node_id : room_id;
    const auto  S  = inst.ai2_belief.covariance();
    const float vs = scale * S(0, 0);   // along-wall position variance σ_s² (m²)
    const Eigen::Vector2f u = inst.ai2_belief.params().wall_u;
    const float chain_along = u.x() * u.x() * inst.chain_cov_xx + u.y() * u.y() * inst.chain_cov_yy;
    const float vz   = inst.ai2_belief.params().floor_std * inst.ai2_belief.params().floor_std;   // cz pinned
    constexpr float kWallYawVar = 3.0e-4f;   // (~1°)²: the wall fixes yaw; the nominal polygon is trusted
    float vx, vy, vyaw = kWallYawVar;
    if (inst.wall_node_id != 0)
    {
        vx = vs + chain_along;   // along wall (local x)
        vy = 1.0e-4f;            // across wall (local y) — the door lies in the wall plane
    }
    else
    {
        vx = vs * u.x() * u.x() + inst.chain_cov_xx;
        vy = vs * u.y() * u.y() + inst.chain_cov_yy;
    }

    // Self-gate on the data-driven trace: write on a geometry republish (force) OR when the
    // uncertainty moved >5 % since the last publish (covers a frozen pose whose belief keeps
    // tightening / loosening). Avoids rewriting the edge every cycle once everything has settled.
    const float trace = vx + vy + vz + vyaw;
    const float prev  = inst.last_pub_cov_trace;
    const bool cov_changed = not std::isfinite(prev) or prev <= 0.0f or
                             std::abs(trace - prev) > 0.05f * prev;
    if (not force and not cov_changed)
        return;

    auto edge = G_->get_edge(parent, inst.node_id, "RT");
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

std::vector<float> DoorSceneGraph::make_door_mesh(const DoorState& s)
{
    // Flat triangle list (room frame): ONE thin panel box (12 triangles). Local x = along the wall (width),
    // local y = across the wall (thickness), local z = up. Centred horizontally at (cx,cy), spanning
    // [cz, cz+h] vertically.
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

    // Panel: half-extents (w/2 along wall, thickness/2 across, h/2 up); centre at (cx,cy,cz+h/2).
    push_box(s.cx, s.cy, s.cz + 0.5f * s.h, 0.5f * s.w, 0.5f * s.thickness, 0.5f * s.h);
    return verts;
}

void DoorSceneGraph::write_door_mesh(DoorInstance& inst, DSR::Node& node)
{
    const std::vector<float> verts = make_door_mesh(inst.model.state());
    G_->add_or_modify_attrib_local<mesh_vertices_att>(node, verts);
}

void DoorSceneGraph::write_rt_pose(std::uint64_t room_id, DoorInstance& inst)
{
    if (room_id == 0 or not rt_api_)
        return;

    const auto& s = inst.model.state();

    // A door hangs from its WALL: resolve the nearest wall node and publish the pose in that wall's frame.
    // Falls back to the room only while no wall nodes exist. If the parent changes (first association, or the
    // door moved onto another wall) we RE-PARENT: drop the stale RT edge and update parent/level.
    const WallRef wall = resolve_wall(room_id, {s.cx, s.cy});
    const std::uint64_t new_parent = wall.ok ? wall.id : room_id;

    auto door_opt = G_->get_node(inst.node_id);
    if (not door_opt.has_value())
        return;
    const std::uint64_t old_parent = G_->get_attrib_by_name<parent_att>(door_opt.value()).value_or(room_id);
    const bool reparent = new_parent != old_parent;

    // Dead-band: suppress RT edge updates below ~5 cm to avoid pos churn from gradient oscillations — but
    // NEVER skip a re-parent (the door must move onto its wall even when it hasn't translated).
    constexpr float kMinWriteDistSq = 0.05f * 0.05f;
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (not reparent and dx * dx + dy * dy < kMinWriteDistSq)
        return;

    // Door node origin = BASE on the floor (z=0 in room). In the wall frame that base sits z0 below the wall
    // origin (which is at half room height). Every consumer assumes a base origin: the voxelizer box
    // (z∈[origin, origin+height]) and bottle_concept's door-top support test (top = origin.z + height).
    float lx = s.cx, ly = s.cy, lz = 0.0f, lyaw = s.yaw;   // room-frame fallback
    std::optional<DSR::Node> parent_node;
    if (wall.ok)
    {
        parent_node = G_->get_node(wall.id);
        if (parent_node.has_value())
        {
            const auto lp = door_in_wall(s.cx, s.cy, s.yaw, wall.mid, wall.yaw, wall.z0);
            lx = lp.x; ly = lp.y; lz = lp.z; lyaw = lp.yaw;
        }
    }
    if (not parent_node.has_value())
        parent_node = G_->get_node(room_id);
    if (not parent_node.has_value())
        return;

    if (reparent)
    {
        G_->delete_edge(old_parent, inst.node_id, "RT");   // a node must have exactly one RT parent
        G_->add_or_modify_attrib_local<parent_att>(door_opt.value(), new_parent);
        G_->add_or_modify_attrib_local<level_att>(door_opt.value(), wall.ok ? 5 : 3);
        G_->update_node(door_opt.value());
        if (relayout_) relayout_();
    }
    inst.wall_node_id = wall.ok ? wall.id : 0;

    rt_api_->insert_or_assign_edge_RT(parent_node.value(), inst.node_id, {lx, ly, lz}, {0.0f, 0.0f, lyaw});
    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

void DoorSceneGraph::write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop)
{
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>  (node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>  (node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>        (node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>     (node, true);
    G_->update_node(node);
}

}  // namespace rc
