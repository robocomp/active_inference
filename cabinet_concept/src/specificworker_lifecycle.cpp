/*
 * specificworker_lifecycle.cpp — SpecificWorker's instance-lifecycle + ricoh-attention methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit) to keep the orchestrator lean. Owns:
 *  - merge_overlapping_instances : physical-exclusion collapse of two instances fitted to one cabinet,
 *  - run_instance_tracker        : the ONLY birth/associate/merge path (shared rc::InstanceTracker),
 *  - process_ricoh_bearings      : ricoh-360 bearing-only PERIPHERAL ATTENTION (never births/fits).
 */

#include "specificworker.h"
#include "cabinet_geometry.h"        // rc::geom::footprint_overlap_ratio, point_in_footprint
#include "cabinet_residual_birth.h"  // rc::find_residual_birth (residual-cluster → new-run seed)
#include "cabinet_lshape_split.h"    // rc::lshape_split (L-corner mask → two arms)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <format>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─── Room geometry (wall-flush factor + C2v yaw reference) ───────────────────────────────────────

// Read the room's delimiting polygon (room frame) + its centroid, and hand both to the fitter. The
// polygon feeds the per-frame wall-flush factor (the term that makes a run's unobservable depth
// identifiable); the centroid is the interior reference canonicalize() folds the 180° yaw against.
// It is a NOMINAL model room_concept authors from apartamento_layout.svg and localizes AGAINST — it
// never fits the polygon to LiDAR returns, so there is no circularity between "the wall" and the
// cabinet front face (verified 2026-07-21). Empty polygon ⇒ the fitter treats every run as
// free-standing (the wall factor goes inert), which is the correct fallback for an unknown room.
void SpecificWorker::refresh_room_geometry()
{
    if (not G or room_node_id_ == 0) return;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return;
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 2) return;

    std::vector<Eigen::Vector2f> poly;
    poly.reserve(n);
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        poly.emplace_back(xs[i], ys[i]);
        centroid += poly.back();
    }
    centroid /= static_cast<float>(n);
    fitter_->set_room_geometry(centroid, std::move(poly));
}

// ─── Instance lifecycle: merge / tracker / birth ─────────────────────────────────────────────────

// Retire one instance: drop its affordance node, forget it in the fitter, delete its graph node. The single
// teardown path shared by every lifecycle exit (merge / death), keeping the affordance+fitter+graph invariant.
void SpecificWorker::retire_instance(std::uint64_t id)
{
    if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
        it->second.affordance.remove();
    fitter_->forget_node(id);
    G->delete_node(id);
}


// Data-driven multi-instance lifecycle (mirrors bottle_concept::run_instance_tracker). Cabinets are large
// static furniture, so birth_min_sep is wide, death is off, and overlaps merge. The only path that
// creates/associates cabinet instances (the prior-scaffold + greedy nearest-mask were removed in Stage 2).
// Collapse instances whose footprints overlap (same physical cabinet fitted twice): keep the one with more
// integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a duplicate is
// gone before it is fed a mask. v1 keeps-best; precision-weighted DOF pooling is a later refinement.
void SpecificWorker::merge_overlapping_instances()
{
    if (cfg_.tracker_merge_overlap <= 0.0f)
        return;
    auto& insts = fitter_->instances();
    if (insts.size() < 2)
        return;

    std::vector<std::uint64_t> ids;
    ids.reserve(insts.size());
    for (auto& [id, _] : insts) ids.push_back(id);

    // Per-instance joint std helper: √Σᵢᵢ + a sensor floor, for the collinear-merge acceptance widths.
    const auto sd = [](const rc::CabinetInstance& in, int i)
    { return in.ai2_initialized ? std::sqrt(std::max(0.0f, in.ai2_belief.covariance()(i, i))) : 1.0f; };

    std::unordered_set<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (removed.count(ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (removed.count(ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;

            // MERGE NEEDS YAW EVIDENCE ON BOTH SIDES. The collinear test asserts "these two runs share an
            // axis"; that claim is only supportable once each run's axis has actually been MEASURED. A
            // just-born fragment is not ai2_initialized, so sd() would hand the parallel gate an
            // uninformative yaw std (1.0 rad ≈ 57°) → n_sigma·s_yaw ≫ π/2 → the gate can NEVER reject, and
            // the two PERPENDICULAR arms of an L-corner (dyaw≈90°) get fused into one tilted, oscillating
            // run every cycle (then the dropped arm re-births → endless birth→merge churn). Requiring both
            // instances fit before they are merge-eligible lets the split's two arms persist as separate
            // runs and re-associate to their own tracks next cycle. This is evidence-gating, not a threshold:
            // no measured axis ⇒ no support for "same axis" ⇒ not mergeable yet.
            if (not ia->second.ai2_initialized or not ib->second.ai2_initialized) continue;

            const auto& sa = ia->second.model.state();
            const auto& sb = ib->second.model.state();

            // A base run and the wall units above it are DISTINCT runs that happen to share a footprint,
            // so an oriented-rectangle overlap would wrongly fuse them. Separate them by their vertical
            // band first: only merge instances whose carcasses actually overlap in z. (Two base runs meeting
            // at an L-corner have z overlap and different yaw — handled by the collinear test's angle gate.)
            const float z_overlap = std::min(sa.z1, sb.z1) - std::max(sa.z0, sb.z0);
            if (z_overlap <= 0.0f) continue;

            // WALL-KEYED IDENTITY (re-key Stage 3). Two runs committed to the SAME wall + same tier, both
            // FLUSH-anchored, ARE the same physical run — merge them BY CONSTRUCTION (take the union
            // interval), with NO σ-gated collinear test that an unconverged covariance could defeat (the
            // root cause behind the whole merge-churn patch chain). A free-standing or not-yet-committed
            // run has no reliable wall id, so it falls back to the geometric collinear test below.
            const auto& A = ia->second;
            const auto& B = ib->second;
            const bool same_cell = A.committed_wall_seg_id >= 0
                                   and A.committed_wall_seg_id == B.committed_wall_seg_id
                                   and A.ai2_belief.tier() == B.ai2_belief.tier();
            const bool both_flush = std::abs(A.ai2_belief.last_wall_gap()) < cfg_.wall_reach_m
                                    and std::abs(B.ai2_belief.last_wall_gap()) < cfg_.wall_reach_m;

            rc::geom::RunMerge m;
            if (same_cell and both_flush)
            {
                // Identity ⇒ gates inert (huge σ) so collinear_merge just returns the union geometry.
                m = rc::geom::collinear_merge(sa, sb, 1.0e3f, 1.0e3f, 1.0e3f, 1.0f);
            }
            else
            {
                // COLLINEAR-RUN merge: turn "a run glimpsed in pieces" into one object. Oriented-rectangle
                // overlap can never join two co-linear NON-overlapping fragments, so use the segment test
                // against the pair's joint uncertainty (sigma_L floored: fragments seen seconds apart abut
                // with a gap on the order of the mask's coarseness).
                const float s_yaw = std::hypot(sd(A, 2), sd(B, 2)) + 0.05f;   // yaw index 2
                const float s_lat = std::hypot(sd(A, 1), sd(B, 1)) + 0.10f;   // lateral floor
                const float s_L   = std::hypot(sd(A, 3), sd(B, 3)) + cfg_.merge_gap_floor_m;
                m = rc::geom::collinear_merge(sa, sb, s_yaw, s_lat, s_L, cfg_.merge_n_sigma);
            }
            if (not m.merge) continue;

            // Keep the more-observed instance and GRAFT the fused length onto it, so the union interval
            // (extent the other fragment proved) is not thrown away — the extent channel is grow-only, so
            // re-observing it from scratch would be slow. Position/yaw follow to the fused axis midpoint.
            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            auto& kinst = insts.at(keep);
            rc::CabinetBeliefState fused = kinst.ai2_belief.state();
            fused.cx = m.cx; fused.cy = m.cy; fused.yaw = m.yaw; fused.L = m.L;
            kinst.ai2_belief.set_state(fused);
            { auto ms = kinst.model.state(); ms.cx = m.cx; ms.cy = m.cy; ms.yaw = m.yaw; ms.L = m.L;
              kinst.model.set_state(ms); }
            std::print("cabinet_concept: [tracker] COLLINEAR MERGE id={} into id={} "
                       "(gap={:.2f} dlat={:.2f} dyaw={:.3f} → L={:.2f})\n",
                       drop, keep, m.gap, m.d_lat, m.d_yaw, m.L);
            ++ev_g_.merges; ++ev_g_.merges_cum;   // EvidenceMonitor counter
            retire_instance(drop);
            removed.insert(drop);
            if (drop == ids[i]) break;   // this i is gone; advance to the next i
        }
    }
}

// One tracker cycle: merge overlaps, build tracks from live instances (Mahalanobis gate on belief Σ) and
// detections from this frame's ZED "cabinet" slices, then apply the result (death / associate / birth). Ricoh
// slices are excluded here (bearing-only). This is the ONLY path that creates/associates cabinet instances.
// Kitchen-model Stage 0 SHADOW (instrument-only, no belief/fit change): build the (wall_id, tier) cell table
// from the trusted room polygon and soft-route this cycle's cabinet/counter/chest mask points into it, then
// log the per-cell evidence mass. Validates the routing foundation live before any behaviour depends on it.
void SpecificWorker::shadow_route_kitchen_cells()
{
    if (not fitter_ or not mask_ingestor_) return;
    if (kitchen_routing_.cells.empty())                     // (re)build when we first have a polygon
    {
        const auto walls = fitter_->kitchen_walls();
        if (walls.empty()) return;
        const rc::KitchenTier tiers[2] = { {0.0f, 0.87f, 0.55f, "base"}, {1.40f, 2.10f, 0.35f, "upper"} };
        kitchen_routing_.build(walls, tiers);
    }
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid) return;
    std::vector<Eigen::Vector3f> pts;
    for (const auto& sl : pkt.slices)
    {
        const bool run_label = sl.label == "cabinet" or sl.label == "chest of drawers"
                               or sl.label == "counter" or sl.label == "countertop";
        if (not run_label or sl.depth_var > 0.0f) continue;   // ZED-depth run masks only
        for (std::size_t k = std::min(sl.support_begin, pkt.support_points.size());
             k < std::min(sl.support_end, pkt.support_points.size()); k += 4)   // subsample
            pts.push_back(pkt.support_points[k]);
    }
    if (pts.empty()) return;
    kitchen_routing_.route(pts, 0.15f, 0.20f, 0.05f);

    static int kdbg = 0;
    if (++kdbg % 30 == 0)   // Stage-0 shadow: one low-frequency diagnostic line (not gated by verbose_log)
    {
        std::string s;
        for (const auto& c : kitchen_routing_.cells)
            if (c.mass > 0.02 * static_cast<double>(pts.size()))
                s += std::format(" {}={:.0f}", c.id, c.mass);
        std::print("[kitchen-cells] pts={} clutter={:.0f} |{}\n", pts.size(), kitchen_routing_.clutter_mass, s);
    }
}

// Transform-chain camera ego-motion (PRODUCER-INDEPENDENT, aligned with chair_concept::update_ego_motion): the
// current room→zed pose vs the previous cycle's → linear + angular speed. ts==0 latest pose on the main thread ⇒
// safe. Used by the stillness/VOR common-mode: geometry updates freeze continuously as motion×periphery grows.
void SpecificWorker::update_kitchen_ego_motion()
{
    const auto now = std::chrono::steady_clock::now();
    if (not inner_eigen_) { have_prev_cam_ = false; return; }
    const auto T = inner_eigen_->get_transformation_matrix("room", "zed", 0);
    if (not T.has_value()) { have_prev_cam_ = false; return; }   // pose chain unavailable → reset baseline
    const Eigen::Vector3f pos = T->translation().cast<float>();
    const Eigen::Vector3f fwd = T->linear().col(1).cast<float>();   // zed +y is the depth/forward axis
    if (have_prev_cam_)
    {
        const float dt = std::max(1e-3f, std::chrono::duration<float>(now - prev_cam_tp_).count());
        ego_lin_mps_ = (pos - prev_cam_pos_).norm() / dt;
        const float fa = prev_cam_fwd_.norm(), fb = fwd.norm();
        const float cang = (fa > 1e-6f and fb > 1e-6f)
            ? std::clamp(prev_cam_fwd_.dot(fwd) / (fa * fb), -1.0f, 1.0f) : 1.0f;
        ego_ang_radps_ = std::acos(cang) / dt;
    }
    prev_cam_pos_ = pos; prev_cam_fwd_ = fwd; prev_cam_tp_ = now; have_prev_cam_ = true;
}

// ─── Stage 2: kitchen-of-runs model ─────────────────────────────────────────────────────────────
// The (wall,tier) cells own the WallRunBeliefs. Identity IS the cell, so there is no birth/associate/merge
// and the disasters (crossings, 10 cm slivers, ceiling boxes) are unrepresentable — the wall chart fixes yaw
// to the wall, pins the back on the wall, and bounds t∈[0,W], z∈[0,H_room]. Per cycle: gather the run-mask
// support points, soft-route them to cells (responsibility → per-point R), let each active cell run one AI2
// update, and reconcile the active set with DSR box nodes. Free-standing islands (peninsulas) route to no wall
// and fall through here — that is Stage 4 (generic tracker), deliberately not handled yet.
void SpecificWorker::run_kitchen_model()
{
    if (not fitter_ or not mask_ingestor_) return;
    if (not kitchen_mgr_.built())                                   // build the cell table once we have a polygon
    {
        const auto walls = fitter_->kitchen_walls();
        if (walls.empty()) return;
        const rc::KitchenTier tiers[2] = { {0.0f, 0.87f, 0.55f, "base"}, {1.40f, 2.10f, 0.35f, "upper"} };
        const rc::WallTierPrior tp[2] = {
            { cfg_.base_depth_m, cfg_.base_depth_std, cfg_.base_z0_m, cfg_.base_z0_std, cfg_.base_z1_m, cfg_.base_z1_std },
            { cfg_.wall_depth_m, cfg_.wall_depth_std, cfg_.wall_z0_m, cfg_.wall_z0_std, cfg_.wall_z1_m, cfg_.wall_z1_std },
        };
        rc::CabinetBeliefParams bp;                                 // only the fields WallRunBelief reads
        bp.sigma_base_m         = cfg_.ai2_sigma_base_m;
        bp.clutter_frac         = cfg_.ai2_clutter_frac;
        bp.clutter_scale_m      = cfg_.ai2_clutter_scale_m;
        bp.prior_L_std          = cfg_.ai2_prior_size_std;
        bp.process_std_m        = cfg_.ai2_process_std_m;
        bp.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        bp.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        bp.gn_iters             = cfg_.ai2_gn_iters;
        bp.extent_precision     = cfg_.extent_precision;
        bp.free_space_precision = cfg_.free_space_precision;
        kitchen_mgr_.build(walls, tiers, tp, bp, cfg_.ceiling_height_m);
        std::print("cabinet_concept: [kitchen] built {} cells over {} walls (H_room={:.2f})\n",
                   kitchen_mgr_.cells().size(), walls.size(), cfg_.ceiling_height_m);
    }

    const auto& pkt = mask_ingestor_->packet();                    // ZED-depth run masks (base + upper labels)
    std::vector<Eigen::Vector3f> pts, island_pts;                  // wall-run points vs free-standing island points
    double radius_wsum = 0.0, dotd_wsum = 0.0, w_sum = 0.0;         // point-weighted mean off-centring + mask motion
    if (pkt.valid)
        for (const auto& sl : pkt.slices)
        {
            const bool is_island = sl.label == "kitchen_island" or sl.label == "kitchen island" or sl.label == "island";
            const bool run_label = sl.label == "cabinet" or sl.label == "chest of drawers"
                                   or sl.label == "counter" or sl.label == "countertop";
            if (sl.depth_var > 0.0f or not (run_label or is_island)) continue;   // ZED-depth cabinet/island masks
            const std::size_t b = std::min(sl.support_begin, pkt.support_points.size());
            const std::size_t e = std::min(sl.support_end,   pkt.support_points.size());
            const double wn = static_cast<double>(e - b);
            radius_wsum += static_cast<double>(sl.centroid_radius)     * wn;
            dotd_wsum   += static_cast<double>(std::abs(sl.motion_dotd)) * wn;
            w_sum       += wn;
            std::vector<Eigen::Vector3f>& dst = is_island ? island_pts : pts;
            for (std::size_t k = b; k < e; k += 2) dst.push_back(pkt.support_points[k]);
        }

    // STILLNESS / VOR gate (be still to UPDATE, else CONFIRM) — ALIGNED with chair_concept. The frame's authority
    // to MOVE the geometry is capped by a common-mode covariance (NOT a hard gate) that the belief's Woodbury
    // marginalisation saturates on. It grows with MOTION × OFF-AXIS position: motion = max(mask motion_dotd,
    // ego_lin + ang_lever·ego_ang); periph = (radius/periph_ref)². Still (motion→0) OR well-centred (periph→0) ⇒
    // ~0 common-mode ⇒ full authority; moving AND peripheral ⇒ confirm-only. A centred mask stays trusted while
    // moving (the "pure-translation" exception, emergent from the periphery factor). WallRun has no yaw DOF.
    update_kitchen_ego_motion();
    const float radius     = (w_sum > 0.0) ? static_cast<float>(radius_wsum / w_sum) : 0.0f;
    const float mean_dotd  = (w_sum > 0.0) ? static_cast<float>(dotd_wsum   / w_sum) : 0.0f;
    const float motion_mag = std::max(mean_dotd, ego_lin_mps_ + cfg_.kitchen_ang_lever_m * ego_ang_radps_);
    const float rr         = radius / std::max(1e-6f, cfg_.kitchen_periph_ref);
    const float periph     = std::clamp(rr * rr, 0.0f, 1.0f);
    const float mv         = cfg_.kitchen_motion_cm_gain * motion_mag;
    rc::KitchenManagerParams mp;
    rc::CabinetFrame tmpl;
    tmpl.ego_motion_pos_var = mv * mv * periph;                     // fed to WallRunBelief::common_mode_inv_diag

    // FREE-SPACE CARVE (evidence of absence) — the complement to the grow-only extent. Feed the room-frame LiDAR
    // sweep into the frame; WallRunBelief::accumulate_freespace marches each ray against the run's box and, for a
    // THROUGH-beam exiting an END/TOP face (p_through gating separates hits from through-beams), RETRACTS that
    // face. So an over-grown end proven empty by LiDAR shrinks back, and corner-fill becomes evidence-gated (a
    // ray seeing through the corner overrides the fill). The reader is live because FreeSpacePrecision>0.
    std::size_t sweep_n = 0;
    if (lidar_ingestor_)
    {
        lidar_ingestor_->pump();                                   // main-thread; pins the sweep to room frame
        if (lidar_ingestor_->helios_fresh())
        {
            tmpl.lidar_freespace.origin    = lidar_ingestor_->origin_room();
            tmpl.lidar_freespace.endpoints = lidar_ingestor_->sweep_room();
            sweep_n = tmpl.lidar_freespace.endpoints.size();
        }
    }

    static int mdbg = 0;
    kitchen_mgr_.set_corner_fill_log(cfg_.verbose_log and (mdbg % 30) == 29);   // corner-fill state dump (verbose only)
    kitchen_mgr_.update(pts, mp, tmpl);
    kitchen_mgr_.update_island(island_pts, mp, tmpl);              // the free-standing 4th cabinet (peninsula)
    publish_kitchen_boxes();

    if (++mdbg % 30 == 0)                                           // low-rate stillness diagnostic
        std::print("cabinet_concept: [kitchen] ego_lin={:.2f} ego_ang={:.2f} dotd={:.2f} radius={:.2f} periph={:.2f} sweep={} → pos_var={:.4f}\n",
                   ego_lin_mps_, ego_ang_radps_, mean_dotd, radius, periph, sweep_n, tmpl.ego_motion_pos_var);
}

// Reconcile the active cells with their DSR box nodes: create on activation, update size+RT while active,
// delete on retirement. Nodes are `box` typed named "cabinet_w<seg>_<tier>" (stable per cell) — the same
// reuse-a-primitive + "cabinet" prefix convention the classic path uses, so every consumer/cleanup still
// matches them. Box convention (mirrors CabinetSceneGraph): width=L, depth=d, height=z1−z0, RT=(cx,cy,z0),(0,0,yaw).
void SpecificWorker::publish_kitchen_boxes()
{
    if (not G or room_node_id_ == 0 or not rt_api_) return;
    auto room_opt = G->get_node(room_node_id_);
    if (not room_opt.has_value()) return;
    const float rpx = G->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
    const float rpy = G->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);

    const auto boxes = kitchen_mgr_.active_boxes();
    std::unordered_set<std::string> live;
    bool any_birth = false;
    int slot = 0;
    for (const auto& b : boxes)
    {
        live.insert(b.id);
        const float H = std::max(0.05f, b.z1 - b.z0);
        auto it = kitchen_nodes_.find(b.id);
        if (it == kitchen_nodes_.end())
        {
            const std::string name = (b.wall_seg_id < 0)
                ? std::string("cabinet_island")
                : "cabinet_w" + std::to_string(b.wall_seg_id) + (b.tier == 0 ? "_base" : "_up");
            DSR::Node node = DSR::Node::create<object_node_type>(name);
            // Display asset for the voxelizer 3D viewer (relative to its meshes/ root); the viewer loads &
            // scales it to the fitted run box (cortex mesh_path contract — the agent owns its appearance).
            G->add_or_modify_attrib_local<mesh_path_att>(node, std::string("cabinet_concept/meshes/cabinet.obj"));
            G->add_or_modify_attrib_local<mesh_texture_path_att>(node, std::string("cabinet_concept/meshes/cabinet_basecolor.jpg"));
            G->add_or_modify_attrib_local<width_m_att> (node, b.L);
            G->add_or_modify_attrib_local<depth_m_att> (node, b.d);
            G->add_or_modify_attrib_local<height_m_att>(node, H);
            G->add_or_modify_attrib_local<object_subtype_att>(node, std::string("cabinet"));  // type-agnostic consumers
            G->add_or_modify_attrib_local<level_att>   (node, 3);
            G->add_or_modify_attrib_local<parent_att>  (node, room_node_id_);
            G->add_or_modify_attrib_local<pos_x_att>   (node, rpx + 150.f + 40.f * static_cast<float>(slot));
            G->add_or_modify_attrib_local<pos_y_att>   (node, rpy + 50.f);
            const auto id_opt = G->insert_node(node);
            if (not id_opt.has_value()) { ++slot; continue; }
            kitchen_nodes_[b.id] = id_opt.value();
            rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(), {b.cx, b.cy, b.z0}, {0.f, 0.f, b.yaw});
            any_birth = true;
            std::print("cabinet_concept: [kitchen] BIRTH '{}' id={} run=({:.2f},{:.2f}) L={:.2f} d={:.2f} z=[{:.2f},{:.2f}]\n",
                       name, id_opt.value(), b.cx, b.cy, b.L, b.d, b.z0, b.z1);
        }
        else
        {
            if (auto n = G->get_node(it->second); n.has_value())
            {
                // Backfill the display asset on nodes that predate this attribute (idempotent).
                if (not G->get_attrib_by_name<mesh_path_att>(n.value()).has_value())
                {
                    G->add_or_modify_attrib_local<mesh_path_att>(n.value(), std::string("cabinet_concept/meshes/cabinet.obj"));
                    G->add_or_modify_attrib_local<mesh_texture_path_att>(n.value(), std::string("cabinet_concept/meshes/cabinet_basecolor.jpg"));
                }
                G->add_or_modify_attrib_local<width_m_att> (n.value(), b.L);
                G->add_or_modify_attrib_local<depth_m_att> (n.value(), b.d);
                G->add_or_modify_attrib_local<height_m_att>(n.value(), H);
                G->update_node(n.value());
                rt_api_->insert_or_assign_edge_RT(room_opt.value(), it->second, {b.cx, b.cy, b.z0}, {0.f, 0.f, b.yaw});
            }
        }
        ++slot;
    }
    for (auto it = kitchen_nodes_.begin(); it != kitchen_nodes_.end(); )
    {
        if (not live.contains(it->first))
        {
            G->delete_node(it->second);
            std::print("cabinet_concept: [kitchen] RETIRE cell '{}' (node {})\n", it->first, it->second);
            it = kitchen_nodes_.erase(it);
        }
        else ++it;
    }
    if (any_birth) trigger_graph_layout_twopi();
}

void SpecificWorker::run_instance_tracker()
{
    shadow_route_kitchen_cells();     // Stage 0: shadow routing diagnostic (no behaviour change)
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = cfg_.tracker_death_frames;
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.multi_det_per_track = true;   // fuse multiple ZED slices of one cabinet (one belief update per slice)
    tp.z_gate_m         = cfg_.tracker_z_gate_m;   // keep WALL-unit masks (z≈1.7) off BASE tracks (z≈0.35)
    tracker_.set_params(tp);

    // Tracks ← live instances: centre from the fit, XY cov from the belief's position covariance Σ.
    std::vector<rc::TrackView> tracks;
    tracks.reserve(fitter_->instances().size());
    for (auto& [id, inst] : fitter_->instances())
    {
        rc::TrackView t;
        t.id = id;
        const auto& s = inst.model.state();
        t.xy = {s.cx, s.cy};
        t.z  = s.zc();                    // box-centre height → the z_gate keeps tiers separate
        if (inst.ai2_initialized)
        {
            // Gate on the belief's position covariance (+ localization chain) so association uses the
            // Mahalanobis innovation S = P + R²I, not the Euclidean fallback (matters for multi-instance).
            const auto& S = inst.ai2_belief.covariance();   // Σ over [cx,cy,yaw,L,d,z0,z1]
            Eigen::Matrix2f P = Eigen::Matrix2f::Zero();
            P(0, 0) = S(0, 0) + inst.chain_cov_xx;
            P(1, 1) = S(1, 1) + inst.chain_cov_yy;
            P(0, 1) = P(1, 0) = S(0, 1);                    // keep the fitted xy correlation
            // ★ RUN association: a mask fragment appears ANYWHERE along the run, so its centroid's
            // along-axis coordinate is ~uniform over L (variance L²/12) and carries almost no identity
            // information — only the lateral offset does. Inflating P along the run axis makes the shared
            // Mahalanobis gate permissive ALONG the run and unchanged ACROSS it, so a fragment near one end
            // still associates to the run centred metres away instead of birthing a duplicate. Without this
            // the centre-distance gate would reject it and the run would fragment into cabinet_1..N — the
            // exact per-module individuation the run model exists to avoid.
            t.cov = rc::geom::run_position_cov(s, P);
            t.has_cov = true;
        }
        tracks.push_back(t);
        inst.assigned_mask_idxs.clear();   // cleared; re-filled below with every slice associated this cycle
    }

    // Detections ← the current "cabinet" mask slices (carry the slice index for the assignment). Built EVERY
    // cycle from the packet (even a stale one): the tracker needs continuous detections so its birth CANDIDATES
    // mature over birth_frames consecutive cycles and so association stays live — gating this on mask freshness
    // wiped the candidates every stale cycle and NOTHING ever birthed. Re-fitting a stale mask is prevented
    // separately, in process_cabinet_node (per-instance frame_id gate), NOT here.
    // Only ZED-depth slices (depth_var==0) reach the tracker; ricoh (depth_var>0) is bearing-only and drives
    // the attention path (process_ricoh_bearings), never birth/associate/fit. Every ZED det is birthable.
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
            // COUNTERTOP AS RUN EVIDENCE. The 'counter' mask is the base run's TOP FACE seen from above:
            // the highest-completeness observation of its along-axis span + axis (spans the whole run, far
            // less occluded than the carcass front) and the ONLY channel that sees depth d directly (front
            // edge → wall). It is a deterministic function of the base run (a slab at z1), NOT its own
            // object, so it REFINES a base run and never BIRTHS one (birthable=false). It rides the normal
            // association: the z_gate (|0.8−base_zc|≈0.4 < ZGateM) binds it to the base run below and keeps
            // it off the upper tier (|0.8−1.7|>gate); observe_slice then feeds its points to the belief,
            // where they land on the top face (SDF≈0 → candidate) and drive accumulate_extent's full span.
            // A 'chest of drawers' is standalone storage furniture — a cabinet run in its own right, so it
            // is a BIRTHABLE cabinet label just like 'cabinet' (fits a short run; wall-flush handles whether
            // it stands against a wall or free). Its carcass reaches the floor, so it is a base-tier run.
            const bool is_cabinet = sl.label == "cabinet" or sl.label == "chest of drawers";
            const bool is_counter = cfg_.counter_evidence_enabled
                                    and (sl.label == "counter" or sl.label == "countertop");
            if ((not is_cabinet and not is_counter) or sl.support_end <= sl.support_begin)
                continue;
            if (sl.depth_var > 0.0f)   // RICOH is BEARING-ONLY: never births/associates/fits — the tracker sees
                continue;              // ZED masks only. Ricoh drives the attention path (process_ricoh_bearings).
            rc::DetectionView dv;
            dv.xy = {sl.centroid.x(), sl.centroid.y()};
            dv.z  = sl.centroid.z();   // slice height → the z_gate keeps WALL masks off BASE tracks
            dv.slice_index = i;
            dv.birthable = is_cabinet; // a countertop refines a base run; only a 'cabinet' mask births one
            // FUSED BIRTH: raise this detection's per-frame birth evidence by the residual SURPRISE MASS under it
            // (independent geometric corroboration). Saturating m/(m+ref) ∈ [0,1) → a real, unexplained-occupancy
            // detection matures in ~1-2 frames; a phantom (m≈0 → evidence 1.0) still serves the full debounce.
            // Off unless cfg_.birth_fusion AND the residual field was read this cycle. See birth_surprise_probe.h.
            if (cfg_.birth_fusion and residual_field_.valid())
            {
                const float m = rc::BirthSurpriseProbe::residual_mass_near(residual_field_, dv.xy.x(), dv.xy.y(),
                                                                           cfg_.birth_fusion_radius_m);
                const float s = m / (m + std::max(1e-3f, cfg_.birth_fusion_mass_ref));   // half-saturated at ref
                dv.birth_evidence = 1.0f + cfg_.birth_fusion_gain * s;
            }
            dets.push_back(dv);
        }

    // Snapshot the ZED cabinet-detection centroids (room frame) for the birth-surprise fusion probe (read at the
    // compute() tail in log_birth_surprise). Cheap; only used when cfg_.birth_surprise_probe is on.
    last_cabinet_dets_xy_.clear();
    for (const auto& d : dets) last_cabinet_dets_xy_.push_back(d.xy);

    const auto res = tracker_.update(tracks, dets);

    // Diagnostic (throttled, plus on any birth/death): reveals "1 slice" (upstream) vs "N slices, no birth".
    static int dbg = 0;
    const int n_assigned = static_cast<int>(std::count_if(res.assignment.begin(), res.assignment.end(),
                                                          [](int a){ return a >= 0; }));

    // EvidenceMonitor mask/tracker counters (merges are added in merge_overlapping_instances above).
    ev_g_.mask_frame_id = pkt.valid ? pkt.frame_id : -1;
    ev_g_.total_slices  = pkt.valid ? static_cast<int>(pkt.slices.size()) : 0;
    ev_g_.class_dets      = static_cast<int>(dets.size());
                                                             // (generic "detections of this agent's class"; name is
                                                             // historical — do NOT rename, it is used by table/chair too)
    ev_g_.assigned      = n_assigned;
    ev_g_.discarded     = static_cast<int>(dets.size()) - n_assigned;
    ev_g_.births       += static_cast<int>(res.births.size());
    ev_g_.births_cum   += static_cast<long>(res.births.size());
    if (cfg_.verbose_log and (++dbg % 30 == 0 or not res.births.empty() or not res.deaths.empty()))
    {
        std::print("[tracker] instances={} cabinet_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& t : tracks)
            std::print("[tracker]   track id={} xy=({:.2f},{:.2f}) has_cov={}\n",
                       t.id, t.xy.x(), t.xy.y(), t.has_cov);
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
        // FULL slice dump — EVERY mask this cycle (all labels), so we can see what YOLO-sem actually produced
        // vs what became a cabinet detection: label, confidence, z-band (base ~0–0.9 vs wall ~1.4–2.0), support.
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& s = pkt.slices[i];
            std::print("[slice {}] '{}' conf={:.2f} depth={} c=({:.2f},{:.2f},{:.2f}) z=[{:.2f},{:.2f}] "
                       "xy=[{:.2f},{:.2f}]x[{:.2f},{:.2f}] support={}\n",
                       i, s.label, s.confidence, s.has_depth ? "Y" : "N",
                       s.centroid.x(), s.centroid.y(), s.centroid.z(), s.bbox_min.z(), s.bbox_max.z(),
                       s.bbox_min.x(), s.bbox_max.x(), s.bbox_min.y(), s.bbox_max.y(),
                       s.support_end - s.support_begin);
        }
    }

    // DEATH: a cabinet is rigid, persistent furniture, so a long occlusion (no mask for many frames) is NOT
    // absence and must never retire it. The ONLY way a cabinet is removed is the MERGE operator (two cabinets
    // cannot share space) or the existence-removal channel. res.deaths is deliberately ignored.

    // ASSOCIATE: route every matched detection's mask slice to its instance (read in observe_slice()). With
    // multi_det_per_track a track may collect SEVERAL ZED slices → fused as sequential updates.
    for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        if (res.assignment[d] >= 0)
        {
            const std::uint64_t id = tracks[res.assignment[d]].id;
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.assigned_mask_idxs.push_back(dets[d].slice_index);
        }

    // FOOTPRINT CLAIM (residual-born runs only): a run split off from a corner mask shares that mask's
    // slice with its parent, so the single-assignment above never feeds it. Give each residual-born
    // instance any cabinet slice whose support points substantially fall on ITS footprint. observe_slice
    // then re-splits the shared cloud per-model, so parent and child each keep their own arm (they are
    // perpendicular, so the collinear-merge guard never fuses them). Scoped to residual_born instances,
    // so the established cabinet_1/2 assignment is untouched.
    if (cfg_.residual_birth_enabled)
        for (auto& [id, inst] : fitter_->instances())
        {
            if (not inst.residual_born) continue;
            const auto& st = inst.model.state();
            for (int d = 0; d < static_cast<int>(dets.size()); ++d)
            {
                const int si = dets[d].slice_index;
                if (std::find(inst.assigned_mask_idxs.begin(), inst.assigned_mask_idxs.end(), si)
                    != inst.assigned_mask_idxs.end())
                    continue;   // already assigned by the tracker
                const auto& sl = pkt.slices[si];
                const std::size_t b = std::min(sl.support_begin, pkt.support_points.size());
                const std::size_t e = std::min(sl.support_end,   pkt.support_points.size());
                int inside = 0, total = 0;
                for (std::size_t k = b; k < e; k += 4)   // subsample: a fraction estimate needs no full scan
                {
                    ++total;
                    const Eigen::Vector2f q(pkt.support_points[k].x(), pkt.support_points[k].y());
                    if (rc::geom::point_in_footprint(st, q, cfg_.residual_claim_margin_m)) ++inside;
                }
                if (total > 0 and static_cast<float>(inside) / static_cast<float>(total) >= cfg_.residual_claim_frac)
                    inst.assigned_mask_idxs.push_back(si);   // dbg_n_zed_slices recomputed at the tracker tail
            }
        }

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the cabinet (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
            fitter_->note_birth(new_id, c);   // full XYZ: z seeds the tier so a WALL unit is born high
    }

    // Per-instance ZED-slice count for the EvidenceMonitor. Every assigned slice is a ZED detection now
    // (ricoh is bearing-only and never reaches assigned_mask_idxs), so this is just the assignment count.
    for (auto& [id, inst] : fitter_->instances())
        inst.dbg_n_zed_slices = static_cast<int>(inst.assigned_mask_idxs.size());
}


// ─── L-corner mask split ────────────────────────────────────────────────────────────────────────

// A cabinet RUN cannot be L-shaped, so one 'cabinet' mask that wraps a room corner is TWO runs. Split each
// such mask into its two perpendicular room-axis arms IN PLACE: partition its support points (cabinet_lshape_
// split.h), reorder them so each arm is contiguous, and replace the one slice with two single-arm sub-slices.
// The tracker then sees two clean detections and the ordinary single-run path births/fits each — no per-frame
// per-instance corner arbitration. Runs after refresh(), before the tracker, on the mutable packet.
void SpecificWorker::split_lshaped_cabinet_masks()
{
    if (not cfg_.lshape_split_enabled or not mask_ingestor_)
        return;
    auto& pkt = mask_ingestor_->mutable_packet();
    if (not pkt.valid or pkt.slices.empty())
        return;

    rc::LShapeParams lp;
    lp.min_arm_pts     = cfg_.lshape_min_arm_pts;
    lp.bin_m           = cfg_.lshape_bin_m;
    lp.arm_halfwidth_m = cfg_.lshape_arm_halfwidth_m;

    std::vector<rc::MaskIngestor::MaskSlice> out;
    out.reserve(pkt.slices.size() + 2);
    bool any_split = false;

    // Recompute a sub-slice's centroid + z-band from its (reordered, contiguous) support range.
    const auto sub_slice = [&](const rc::MaskIngestor::MaskSlice& parent, std::size_t sb, std::size_t se)
    {
        rc::MaskIngestor::MaskSlice s = parent;     // inherit label/conf/motion/etc.
        s.support_begin = sb; s.support_end = se;
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        float zmin = 1e9f, zmax = -1e9f;
        for (std::size_t i = sb; i < se; ++i)
        {
            sum += pkt.support_points[i];
            zmin = std::min(zmin, pkt.support_points[i].z());
            zmax = std::max(zmax, pkt.support_points[i].z());
        }
        const float n = static_cast<float>(std::max<std::size_t>(1, se - sb));
        s.centroid = sum / n;
        s.bbox_min.z() = zmin; s.bbox_max.z() = zmax;
        return s;
    };

    for (const auto& s : pkt.slices)
    {
        // 'counter' masks split too: a countertop that wraps the U spans multiple runs and must decompose
        // per-arm just like the carcass mask (a straight counter yields one arm ⇒ unchanged).
        const bool label_ok = s.label == "cabinet" or s.label == "chest of drawers"
                              or (cfg_.counter_evidence_enabled and (s.label == "counter" or s.label == "countertop"));
        const bool splittable = label_ok and s.has_depth and s.support_end > s.support_begin;
        if (not splittable) { out.push_back(s); continue; }

        const auto r = rc::lshape_split(pkt.support_points, s.support_begin, s.support_end, lp);
        if (not r.is_split) { out.push_back(s); continue; }

        // Reorder this slice's support points arm-by-arm so each arm's sub-slice is a contiguous range,
        // then emit one sub-slice per arm (2 for an L, 3 for a U, …). Points that fell to a dropped tiny
        // arm are left past the last bound and simply go unreferenced (no sub-slice covers them).
        const std::size_t b = s.support_begin, e = s.support_end;
        std::vector<Eigen::Vector3f> reordered;
        reordered.reserve(e - b);
        std::vector<std::size_t> bounds{b};
        for (const auto& arm : r.arms)
        {
            for (const int idx : arm.idx) reordered.push_back(pkt.support_points[idx]);
            bounds.push_back(b + reordered.size());
        }
        std::copy(reordered.begin(), reordered.end(), pkt.support_points.begin() + static_cast<std::ptrdiff_t>(b));

        std::string dbg;
        for (std::size_t k = 0; k < r.arms.size(); ++k)
        {
            out.push_back(sub_slice(s, bounds[k], bounds[k + 1]));
            dbg += std::format(" | {}={} c=({:.2f},{:.2f})",
                               r.arms[k].axis == 0 ? "y-run" : "x-run",
                               r.arms[k].idx.size(), r.arms[k].c.x(), r.arms[k].c.y());
        }
        any_split = true;
        if (cfg_.verbose_log)
            std::print("cabinet_concept: [split] '{}' mask ({} pts) → {} arms{}\n",
                       s.label, e - b, r.arms.size(), dbg);
    }
    if (any_split)
        pkt.slices = std::move(out);
}

// ─── Residual-driven birth ────────────────────────────────────────────────────────────────────────

// Cluster the pooled model-unexplained (residual) points across all instances; a coherent, elongated,
// separated arm that no believed run covers (e.g. the perpendicular arm of an L-shaped corner mask)
// matures over residual_birth_frames cycles into its own axis-aligned "cabinet_N", pre-seeded from the
// arm so it commits to it (not the shared slice's dominant/parent arm). Runs AFTER the fits, when
// last_residual_pts is current. The clustering/seed/separation logic is pure (cabinet_residual_birth.h).
void SpecificWorker::birth_from_residual()
{
    if (not cfg_.residual_birth_enabled)
        return;
    auto& insts = fitter_->instances();
    if (insts.empty())
    {
        residual_cand_active_ = false; residual_cand_hits_ = 0;
        return;
    }

    // Pool residuals + snapshot the current footprints (to exclude points already on a believed run).
    std::vector<Eigen::Vector3f>  residual;
    std::vector<rc::CabinetState> existing;
    existing.reserve(insts.size());
    for (auto& [id, inst] : insts)
    {
        existing.push_back(inst.model.state());
        residual.insert(residual.end(), inst.last_residual_pts.begin(), inst.last_residual_pts.end());
    }

    rc::ResidualBirthParams p;
    p.min_cluster_pts = cfg_.residual_birth_min_pts;
    p.sep_m           = cfg_.residual_birth_sep_m;
    const auto cand = rc::find_residual_birth(residual, existing, p);

    // Per-cycle tuning diagnostic (throttled; always on a candidate/near-miss so a rejected arm is visible).
    // Fields: residual pool (all instances) → free pool (off believed footprints) → top cluster mass;
    // then the seed's length/elongation/separation and WHY it was rejected, plus the debounce progress.
    static int rdbg = 0;
    if (cfg_.verbose_log and (++rdbg % 30 == 0 or cand.ok or cand.pts > 0 /* a cluster formed */ or residual_cand_hits_ > 0))
        std::print("[residual-birth] pool={} free={} topcluster={} (min={}) L={:.2f} aniso={:.2f} sep={:.2f} "
                   "reason={} hits={}/{}\n",
                   static_cast<int>(residual.size()), cand.pool, cand.pts, cfg_.residual_birth_min_pts,
                   cand.seed.L, cand.seed.aniso, cand.nearest_sep, cand.reason,
                   residual_cand_hits_, cfg_.residual_birth_frames);

    if (not cand.ok)
    {
        if (--residual_cand_hits_ <= 0) { residual_cand_active_ = false; residual_cand_hits_ = 0; }
        return;
    }

    // Debounce: the same arm must recur near the same place for residual_birth_frames cycles.
    const Eigen::Vector2f xy(cand.seed.cx, cand.seed.cy);
    if (residual_cand_active_ and (xy - residual_cand_xy_).norm() < cfg_.residual_birth_match_m)
        ++residual_cand_hits_;
    else
    { residual_cand_active_ = true; residual_cand_hits_ = 1; }
    residual_cand_xy_   = xy;
    residual_cand_seed_ = cand.seed;
    if (residual_cand_hits_ < cfg_.residual_birth_frames)
        return;

    // Mature → birth. The full seed (via note_birth_seed) makes ensure_instance/lazy-init commit the box
    // to this arm; run_instance_tracker's footprint-claim then feeds it from the shared slice.
    const Eigen::Vector3f c3(cand.seed.cx, cand.seed.cy, 0.5f * (cand.seed.z0 + cand.seed.z1));
    const auto new_id = scene_graph_->create_instance_from_detection(c3, room_node_id_);
    if (new_id != 0)
    {
        fitter_->note_birth_seed(new_id, cand.seed);
        std::print("cabinet_concept: [residual-birth] new run id={} at ({:.2f},{:.2f}) yaw={:.1f}° L={:.2f} pts={}\n",
                   new_id, cand.seed.cx, cand.seed.cy, cand.seed.yaw * 57.2958f, cand.seed.L, cand.pts);
        ++ev_g_.births; ++ev_g_.births_cum;
    }
    residual_cand_active_ = false; residual_cand_hits_ = 0;
}

// ─── Ricoh 360 peripheral attention (bearing-only) ───────────────────────────────────────────────

// Ricoh 360 as PERIPHERAL ATTENTION. A ricoh detection's DIRECTION (bearing from the robot) is reliable even
// when its centroid/range is biased, so we use ONLY that: associate a ricoh bearing to an existing cabinet if the
// bearing falls within that cabinet's angular extent (its circumscribed radius over its range — a physical size,
// not a tuned gate). A ricoh bearing that NO cabinet explains is UNASSIGNED → an attention target ("there is
// something cabinet-like in that direction that the ZED hasn't confirmed; seek a ZED view there"). Never births,
// never fits — that is the ZED's job. This is step 1–3 of the peripheral→saccade→foveal design; the controller
// consuming the target (turn the ZED to the bearing) is step 4.
void SpecificWorker::process_ricoh_bearings()
{
    ricoh_attention_targets_.clear();
    if (not inner_eigen_)   // ricoh peripheral attention is always on (validated live); it never fits — ZED's job
        return;
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);   // robot pose in room
    if (not rtb.has_value())
        return;
    const auto& Tm = rtb.value();
    const float bx = static_cast<float>(Tm(0, 3)), by = static_cast<float>(Tm(1, 3));   // body XY in room
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };

    for (const auto& sl : pkt.slices)
    {
        if (sl.label != "cabinet" or sl.depth_var <= 0.0f)          // ricoh (depth-carrying 360) detections only
            continue;
        if (sl.confidence < cfg_.ricoh_attention_conf)            // only reasonably confident peripheral blobs
            continue;

        // Robust position from the mask's 3D points: component-wise MEDIAN (resists the see-through/outlier tail
        // that biases the centroid). Foreground-gated upstream, so the median sits near the tabletop surface.
        // Falls back to the centroid if too few points. Used for the ATTENTION target's bearing + rough range
        // ONLY — never the fit (a partial-view range is biased, so it decides WHERE TO LOOK / WHICH cabinet, not
        // where the cabinet IS; the ZED measures that). Bearing is reliable; range is indicative.
        float mx = sl.centroid.x(), my = sl.centroid.y();
        const std::size_t b = std::min(sl.support_begin, pkt.support_points.size());
        const std::size_t e = std::min(sl.support_end,   pkt.support_points.size());
        if (e > b + 8)
        {
            std::vector<float> xs, ys; xs.reserve(e - b); ys.reserve(e - b);
            for (std::size_t i = b; i < e; ++i) { xs.push_back(pkt.support_points[i].x()); ys.push_back(pkt.support_points[i].y()); }
            const std::size_t k = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + k, xs.end()); mx = xs[k];
            std::nth_element(ys.begin(), ys.begin() + k, ys.end()); my = ys[k];
        }
        const float br  = std::atan2(my - by, mx - bx);                            // bearing to the robust point
        const float rng = std::sqrt((mx - bx) * (mx - bx) + (my - by) * (my - by));// rough range (biased, indicative)

        bool assigned = false;
        for (const auto& [id, inst] : fitter_->instances())
        {
            const auto& s = inst.model.state();
            const float dx = s.cx - bx, dy = s.cy - by;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 1e-3f) { assigned = true; break; }
            const float tb   = std::atan2(dy, dx);                                  // bearing to the cabinet
            const float rad  = 0.5f * std::sqrt(s.L * s.L + s.d * s.d);             // circumscribed footprint radius
            const float tol  = std::atan2(rad, dist) + cfg_.ricoh_attention_angle_margin_rad;  // TIGHT: angular half-size
            const float band = rad + cfg_.ricoh_attention_range_band_m;              // GENEROUS: ricoh range is rough
            // Assign only if the detection matches this cabinet in BOTH direction AND (rough) range — so a new,
            // unconfirmed cabinet hiding along the SAME bearing as a known one (different range) stays UNASSIGNED
            // and still raises attention, instead of collapsing onto the known cabinet (bearing-only would miss it).
            if (std::abs(wrap(br - tb)) < tol and std::abs(rng - dist) < band) { assigned = true; break; }
        }
        if (not assigned)
            ricoh_attention_targets_.push_back({br, rng, sl.confidence, {mx, my}});
    }

    ev_g_.ricoh_attention = static_cast<int>(ricoh_attention_targets_.size());
    static int rb_dbg = 0;
    if (not ricoh_attention_targets_.empty() and (rb_dbg++ % 10 == 0))
        for (const auto& t : ricoh_attention_targets_)
            std::print("[ricoh-attention] UNASSIGNED cabinet bearing={:+.0f}° range≈{:.1f}m conf={:.2f} → seek ZED view\n",
                       t.bearing_rad * 180.0f / static_cast<float>(M_PI), t.range_m, t.confidence);
}

