/*
 * specificworker_lifecycle.cpp — SpecificWorker's instance-lifecycle + ricoh-attention methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit) to keep the orchestrator lean. Owns:
 *  - merge_overlapping_instances : physical-exclusion collapse of two instances fitted to one cabinet,
 *  - run_instance_tracker        : the ONLY birth/associate/merge path (shared rc::InstanceTracker),
 *  - process_ricoh_bearings      : ricoh-360 bearing-only PERIPHERAL ATTENTION (never births/fits).
 */

#include "specificworker.h"

#include "../../common/peripheral_channel/peripheral_channel.h"   // THE shared ricoh path
#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include "../../common/existence_belief/existence_belief.h"   // rc::exist — peripheral CONFIRM-ONLY
#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth:: the shared CREATE policy
#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "cabinet_geometry.h"        // rc::geom::footprint_overlap_ratio, point_in_footprint
#include "cabinet_residual_birth.h"  // rc::find_residual_birth (residual-cluster → new-run seed)
#include "cabinet_lshape_split.h"    // rc::lshape_split (L-corner mask → two arms)

#include <algorithm>
#include <cmath>
#include <limits>   // numeric_limits<int>::max — the disabled tracker death counter
#include <cstdint>
#include <print>
#include <format>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

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

// Read OTHER agents' furniture from the shared graph as room-frame OBBs (mirrors retina build_graph_object_box):
// type "object", SKIP our own cabinet_* runs, need width/depth/height + a room->object RT transform. A min-footprint
// guard drops small clutter (bottles/cups). ts==0 transform on the main thread ⇒ cache-safe. Empty if the feature
// is off / graph unavailable. Consumed by the exclusion factor + engulfment retirement.
std::vector<rc::SceneObjectBox> SpecificWorker::read_scene_objects() const
{
    std::vector<rc::SceneObjectBox> out;
    if (not G or not inner_eigen_) return out;
    constexpr float kMinFootprintM2 = 0.06f;
    for (const auto& node : G->get_nodes_by_type("object"))
    {
        if (node.name().starts_with("cabinet")) continue;                 // our own kitchen runs — never self-exclude
        const auto w = G->get_attrib_by_name<width_m_att>(node);
        const auto d = G->get_attrib_by_name<depth_m_att>(node);
        const auto h = G->get_attrib_by_name<height_m_att>(node);
        if (not w or not d or not h) continue;
        if (w.value() <= 0.0f or d.value() <= 0.0f or h.value() <= 0.0f) continue;
        if (w.value() * d.value() < kMinFootprintM2) continue;            // drop bottles/cups (class-agnostic)
        const auto T = inner_eigen_->get_transformation_matrix("room", node.name(), 0);
        if (not T.has_value()) continue;
        rc::SceneObjectBox b;
        b.cx  = static_cast<float>(T->translation().x());
        b.cy  = static_cast<float>(T->translation().y());
        b.z0  = static_cast<float>(T->translation().z());                  // base origin (0 for a floor object)
        b.z1  = b.z0 + h.value();
        b.yaw = static_cast<float>(std::atan2(T->linear()(1, 0), T->linear()(0, 0)));
        b.w   = w.value(); b.d = d.value();
        out.push_back(b);
    }
    return out;
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
        bp.tier_prior_gain      = cfg_.tier_prior_gain;   // STANDING d/z0/z1 prior — see the param's note
        bp.object_exclusion_precision = cfg_.object_exclusion_precision;   // retract a run's crossing end
        bp.object_exclusion_margin_m  = cfg_.object_exclusion_margin_m;
        kitchen_mgr_.build(walls, tiers, tp, bp, cfg_.ceiling_height_m);
        std::print("cabinet_concept: [kitchen] built {} cells over {} walls (H_room={:.2f})\n",
                   kitchen_mgr_.cells().size(), walls.size(), cfg_.ceiling_height_m);
    }

    const auto& pkt = mask_ingestor_->packet();                    // ZED-depth run masks (base + upper labels)
    std::vector<Eigen::Vector3f> pts, island_pts;                  // wall-run points vs free-standing island points
    std::vector<std::uint8_t>    pt_labels;                        // 0 = carcass mask, 1 = worktop mask
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
            // Keep WHICH mask each point came from. The run pool deliberately mixes carcass and
            // worktop labels (a counter is evidence for the run beneath it), but a countertop mask can
            // also take in the overhang and whatever stands on it — so when a run fits an impossible
            // depth, the label of the far points is the thing that identifies the culprit.
            const bool worktop_label = sl.label == "counter" or sl.label == "countertop";
            std::vector<Eigen::Vector3f>& dst = is_island ? island_pts : pts;
            for (std::size_t k = b; k < e; k += 2)
            {
                dst.push_back(pkt.support_points[k]);
                if (not is_island) pt_labels.push_back(worktop_label ? 1 : 0);
            }
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
    mp.object_exclusion_enabled = cfg_.object_exclusion_precision > 0.0f;   // gate the engulfment-retirement pass
    mp.engulf_frac              = cfg_.object_engulf_frac;
    // RETIREMENT: LiDAR evidence of absence on the born cells (a beam reaching the wall BEHIND a run refutes it).
    // Same physical sensor rates as the classic existence path — one calibrated model, two consumers.
    mp.lidar_existence_enabled  = cfg_.kitchen_lidar_existence;
    mp.sensor_sigma_m           = cfg_.existence_sensor_sigma_m;
    mp.detection_prob           = cfg_.existence_detection_prob;
    mp.clutter_prob             = cfg_.existence_clutter_prob;
    mp.absence_range_ref_m      = cfg_.existence_absence_range_ref_m;
    mp.absence_range_power      = cfg_.existence_absence_range_power;
    mp.retire_frames            = cfg_.existence_remove_frames;
    rc::CabinetFrame tmpl;
    tmpl.ego_motion_pos_var = mv * mv * periph;                     // fed to WallRunBelief::common_mode_inv_diag

    // SCENE-OBJECT NON-PENETRATION: read OTHER agents' furniture (fridge/table/…) so a run does not cross them.
    // The exclusion factor retracts a crossing end; a run engulfed by an object is retired (the on-fridge false
    // cabinet the LiDAR carve can't reach). Same frame flows to both update() and update_island().
    if (cfg_.object_exclusion_precision > 0.0f)
        tmpl.scene_objects = read_scene_objects();

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

    // ── DO THE MASK POINTS AGREE WITH THE LiDAR? ────────────────────────────────────────────────
    // Every fit today has been faithful to its points, and the points keep being wrong the same way:
    // a run built from countertop masks reached 0.91 m off the wall, and the peninsula's box sits
    // where ~6x more LiDAR beams pass through than stop. Both are what over-reading depth looks like.
    // This measures it directly: bin the sweep by azimuth, take the NEAREST return per bin (the first
    // surface along that ray), and compare each mask point's range from the same origin. A positive
    // median means the masks de-project BEYOND the surface the LiDAR sees — points in mid-air.
    // One number per cycle; costs a pass over the sweep and the points.
    float mask_lidar_dr = 0.0f;
    int   mask_lidar_n  = 0;
    if (sweep_n > 0 and not pts.empty())
    {
        // ★Match HEIGHT as well as bearing. A first version binned by azimuth alone and compared each
        // mask point against the nearest return in that direction — which is usually the floor a metre
        // from the robot, not the cabinet. It produced a rock-steady +2.09 m that measured the gap to
        // the floor, not any depth error. A surface comparison has to be against the surface.
        constexpr int   kBins  = 720;                      // 0.5 deg in azimuth
        constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;
        constexpr float kDz    = 0.15f;                    // same-surface height window (m)
        const Eigen::Vector3f org = tmpl.lidar_freespace.origin;
        const auto bin_of = [&](const Eigen::Vector3f& v)
        { return std::clamp(static_cast<int>((std::atan2(v.y(), v.x()) + std::numbers::pi_v<float>)
                                             / kTwoPi * kBins), 0, kBins - 1); };
        std::vector<std::vector<std::pair<float, float>>> ret(kBins);   // (range, z) per azimuth bin
        for (const auto& e : tmpl.lidar_freespace.endpoints)
        {
            const Eigen::Vector3f v = e - org;
            ret[static_cast<std::size_t>(bin_of(v))].emplace_back(v.head<2>().norm(), e.z());
        }
        std::vector<float> dr;
        dr.reserve(pts.size());
        for (const auto& p : pts)
        {
            const Eigen::Vector3f v = p - org;
            float nearest = -1.0f;
            for (const auto& [r, z] : ret[static_cast<std::size_t>(bin_of(v))])
                if (std::abs(z - p.z()) < kDz and (nearest < 0.0f or r < nearest)) nearest = r;
            if (nearest < 0.0f) continue;                  // nothing at this bearing AND height
            dr.push_back(v.head<2>().norm() - nearest);
        }
        if (not dr.empty())
        {
            std::ranges::nth_element(dr, dr.begin() + static_cast<long>(dr.size() / 2));
            mask_lidar_dr = dr[dr.size() / 2];              // MEDIAN: robust to the tail of stray points
            mask_lidar_n  = static_cast<int>(dr.size());
        }
    }

    static int mdbg = 0;
    kitchen_mgr_.set_corner_fill_log(cfg_.verbose_log and (mdbg % 30) == 29);   // corner-fill state dump (verbose only)
    // Level-2 first: the arrangement's end targets must be in place BEFORE the fit, or they would
    // only take effect a cycle late and always chase the data.
    apply_arrangement_end_priors();

    // ★A POINT ANOTHER OBJECT ALREADY EXPLAINS IS NOT EVIDENCE FOR A KITCHEN CELL EITHER (SHARED,
    // common/exclusion). The instance path got this in 8b74282; the KITCHEN path is where cabinet actually
    // lives — its belief unit is a (wall, tier) cell, not a fitter instance — so the drop has to happen here
    // too or the rule simply never runs for this agent. Measured 2026-08-16: cell w-42_-5_d18_t0 grew along
    // the same wall as refrigerator_1, same yaw, until its near end sat ~4 cm from the fridge's footprint.
    //
    // Filtered HERE rather than inside KitchenManager::update so the routing stays untouched, and pts and
    // pt_labels are filtered TOGETHER — they are parallel arrays and a lone filter would silently mis-label
    // every point after the first drop.
    //
    // ★AND THE POINT'S z GOES IN (2026-08-16). Without it this was the most destructive form of the rule, not
    // the safest: a kitchen is built out of things stacked over one another, so hood_1 — measured 100.0%
    // inside cabinet_w13_base's footprint and 1.2 m above it — was silently deleting every return along
    // ~0.97 m of the cabinet's own wall run. A 2-D containment test cannot tell "somebody else is here" from
    // "somebody else is overhead", and in this room the second is the common case.
    if (not foreign_claims_.empty() and not pts.empty())
    {
        const bool labelled = (pt_labels.size() == pts.size());
        std::vector<Eigen::Vector3f> keep_pts;
        std::vector<std::uint8_t>    keep_labels;
        keep_pts.reserve(pts.size());
        if (labelled) keep_labels.reserve(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i)
        {
            if (rc::exclusion::explained_by_other(pts[i].x(), pts[i].y(), pts[i].z(), foreign_claims_))
                continue;
            keep_pts.push_back(pts[i]);
            if (labelled) keep_labels.push_back(pt_labels[i]);
        }
        const std::size_t dropped = pts.size() - keep_pts.size();
        if (dropped > 0)
        {
            // NOT gated on verbose_log. The sibling exclusion reports (birth, occupancy) are not, and this is
            // the one that says whether the rule is running at all — the first thing anyone asks. Throttled
            // rather than silenced: a drop happening every cycle should say so periodically, not never.
            static int excl_log = 0;
            if ((excl_log++ % 30) == 0)
                std::print("cabinet_concept: [exclusion] {} of {} kitchen points dropped — already explained "
                           "by another object\n", dropped, pts.size());
            pts.swap(keep_pts);
            if (labelled) pt_labels.swap(keep_labels);
        }
    }

    kitchen_mgr_.update(pts, mp, tmpl, &pt_labels);
    kitchen_mgr_.update_island(island_pts, mp, tmpl);              // the free-standing 4th cabinet (peninsula)
    publish_kitchen_boxes();
    log_kitchen_cells(sweep_n, mask_lidar_dr, mask_lidar_n);   // per-cycle cell CSV

    if (++mdbg % 30 == 0)                                           // low-rate stillness diagnostic
        std::print("cabinet_concept: [kitchen] ego_lin={:.2f} ego_ang={:.2f} dotd={:.2f} radius={:.2f} periph={:.2f} sweep={} objs={} → pos_var={:.4f}\n",
                   ego_lin_mps_, ego_ang_radps_, mean_dotd, radius, periph, sweep_n, tmpl.scene_objects.size(), tmpl.ego_motion_pos_var);
}

// Read the arrangement's end prior for every live run. Room-frame targets are projected onto each
// run's own chart here — the metaconcept deliberately knows nothing about t0/t1, so the projection
// belongs on this side.
void SpecificWorker::apply_arrangement_end_priors()
{
    kitchen_mgr_.clear_end_priors();          // a silent frame must stop steering, not linger
    if (not cfg_.arrangement_prior_enabled or not G)
        return;
    const auto now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());

    for (const auto& [cell_id, node_id] : kitchen_nodes_)
    {
        const rc::WallChart* chart = kitchen_mgr_.cell_chart(cell_id);
        if (not chart) continue;
        for (const auto& e : G->get_edges_to_id(node_id))
        {
            if (e.type() != "group_member") continue;
            // Staleness: a frame that stopped publishing stops steering. 0 disables the check.
            if (cfg_.arrangement_stale_ms > 0)
                if (const auto st = G->get_attrib_by_name<rig_stamp_ms_att>(e); st.has_value())
                    if (now > st.value() and now - st.value()
                        > static_cast<std::uint64_t>(cfg_.arrangement_stale_ms))
                        continue;

            const auto lo_i = G->get_attrib_by_name<rig_end_lo_info_att>(e);
            const auto hi_i = G->get_attrib_by_name<rig_end_hi_info_att>(e);
            const auto lo_x = G->get_attrib_by_name<rig_end_lo_x_att>(e);
            const auto lo_y = G->get_attrib_by_name<rig_end_lo_y_att>(e);
            const auto hi_x = G->get_attrib_by_name<rig_end_hi_x_att>(e);
            const auto hi_y = G->get_attrib_by_name<rig_end_hi_y_att>(e);
            // Project each room-frame target onto this chart's along-wall coordinate.
            const auto to_t = [&](float x, float y)
            { return (Eigen::Vector2f(x, y) - chart->A).dot(chart->u); };
            float t0 = 0.0f, t0_info = 0.0f, t1 = 0.0f, t1_info = 0.0f;
            if (lo_i and lo_i.value() > 0.0f and lo_x and lo_y)
            { t0 = to_t(lo_x.value(), lo_y.value());
              t0_info = std::min(lo_i.value(), cfg_.arrangement_end_info_max); }
            if (hi_i and hi_i.value() > 0.0f and hi_x and hi_y)
            { t1 = to_t(hi_x.value(), hi_y.value());
              t1_info = std::min(hi_i.value(), cfg_.arrangement_end_info_max); }
            if (t0_info > 0.0f or t1_info > 0.0f)
            {
                kitchen_mgr_.set_cell_end_prior(cell_id, t0, t0_info, t1, t1_info);
                if (cfg_.verbose_log)
                    std::print("cabinet_concept: [kitchen] '{}' end prior t0={:.2f}({:.0f}) t1={:.2f}({:.0f})\n",
                               cell_id, t0, t0_info, t1, t1_info);
            }
            break;   // one arrangement per run
        }
    }
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

    // Pin the chain covariance to the CAPTURE stamp, not "now" — the run was fit from this packet's points,
    // so the localization uncertainty that applies is the one at capture time (CLAUDE.md: real ts ⇒ no cache).
    const std::uint64_t mask_stamp_ms = mask_ingestor_ ? mask_ingestor_->packet().timestamp_ms : 0;

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
            // ★Name it for what it IS. Only the mask label sends a run down the island path; the
            // geometry decides whether it is free-standing or a PENINSULA (short side against a
            // wall — which is what the apartment actually has). Calling a wall-attached cabinet an
            // "island" misdescribes it in the graph and hides which branch of derive_island_chart
            // fired, so the name now carries the answer.
            const std::string name = (b.wall_seg_id < 0)
                ? (b.anchored ? std::string("cabinet_peninsula") : std::string("cabinet_island"))
                : "cabinet_w" + std::to_string(b.wall_seg_id) + (b.tier == 0 ? "_base" : "_up");
            DSR::Node node = DSR::Node::create<object_node_type>(name);
            // Display asset for the retina 3D viewer (relative to its meshes/ root); the viewer loads &
            // scales it to the fitted run box (cortex mesh_path contract — the agent owns its appearance).
            G->add_or_modify_attrib_local<mesh_path_att>(node, std::string("cabinet_concept/meshes/cabinet.obj"));
            G->add_or_modify_attrib_local<mesh_texture_path_att>(node, std::string("cabinet_concept/meshes/cabinet_basecolor.jpg"));
            G->add_or_modify_attrib_local<width_m_att> (node, b.L);
            G->add_or_modify_attrib_local<depth_m_att> (node, b.d);
            G->add_or_modify_attrib_local<height_m_att>(node, H);
            G->add_or_modify_attrib_local<object_subtype_att>(node, std::string("cabinet"));  // type-agnostic consumers
            // SIZE uncertainty alongside the sizes themselves (cortex `object_size_variance`,
            // registered 2026-08-10). Without it a consumer must treat every producer's dimensions as
            // equally certain; see the note on that REGISTER_TYPE.
            G->add_or_modify_attrib_local<object_size_variance_att>(node,
                std::vector<float>{b.var_width, b.var_depth, b.var_height});
            G->add_or_modify_attrib_local<level_att>   (node, 3);
            G->add_or_modify_attrib_local<parent_att>  (node, room_node_id_);
            G->add_or_modify_attrib_local<pos_x_att>   (node, rpx + 150.f + 40.f * static_cast<float>(slot));
            G->add_or_modify_attrib_local<pos_y_att>   (node, rpy + 50.f);
            rc::provenance::stamp_creation(*G, node);   // birth stamp: epoch ms + local ISO-8601
            const auto id_opt = G->insert_node(node);
            if (not id_opt.has_value()) { ++slot; continue; }
            kitchen_nodes_[b.id] = id_opt.value();
            rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(), {b.cx, b.cy, b.z0}, {0.f, 0.f, b.yaw});
            write_kitchen_rt_covariance(room_node_id_, id_opt.value(), b, mask_stamp_ms);
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
                G->add_or_modify_attrib_local<object_size_variance_att>(n.value(),
                    std::vector<float>{b.var_width, b.var_depth, b.var_height});
                G->update_node(n.value());
                rt_api_->insert_or_assign_edge_RT(room_opt.value(), it->second, {b.cx, b.cy, b.z0}, {0.f, 0.f, b.yaw});
                // After the pose write, so a re-created edge cannot clobber the covariance (same ordering
                // rationale as CabinetSceneGraph::persist_cabinet_belief).
                write_kitchen_rt_covariance(room_node_id_, it->second, b, mask_stamp_ms);
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
            kitchen_cov_trace_.erase(it->first);
            it = kitchen_nodes_.erase(it);
        }
        else ++it;
    }
    if (any_birth) trigger_graph_layout_twopi();
}

// Publish one run's room-frame pose covariance on the room→run RT edge.
//
// WHY THIS EXISTS: the classic (per-instance) path has always written rt_covariance
// (CabinetSceneGraph::write_rt_covariance), but the kitchen path wrote pose ONLY. A consumer that weights
// peers by their published Σ — the controller, and any level-2 metaconcept, whose entire "no confidence
// gate" argument rests on the producer's own covariance being there — saw nothing, and had to fall back to
// treating a 2.2 m well-observed run and a barely-glimpsed sliver as equally certain.
//
// The numbers come from KitchenManager::fill_box_covariance, which maps the belief's wall-chart Σ over
// [t0,t1,d,z0,z1] into the room frame. Two things differ from the classic path, both improvements the
// chart makes free:
//   · the XY block is ANISOTROPIC and correctly rotated (along-wall vs depth uncertainty are different
//     quantities and the chart knows which way each points), not a diagonal;
//   · the yaw variance states the CHART's provenance — tight for a polygon-pinned wall run, measured and
//     typically much looser for the self-derived island. That contrast is the useful signal.
void SpecificWorker::write_kitchen_rt_covariance(std::uint64_t room_id, std::uint64_t node_id,
                                                 const rc::KitchenBox& box, std::uint64_t stamp_ms)
{
    if (not G or room_id == 0 or node_id == 0)
        return;

    float vxx = box.cov_xx, vyy = box.cov_yy;
    const float vxy = box.cov_xy;
    // The run is fit in ROOM but its pose stays conditional on the robot pose (camera→robot→room), so the
    // localization/chain term belongs in what we publish. Without it the run advertises the belief's
    // internal precision as if the robot knew exactly where it was standing.
    if (fitter_)
    {
        float cxx = 0.0f, cyy = 0.0f;
        if (fitter_->chain_cov_at({box.cx, box.cy}, stamp_ms, cxx, cyy))
        { vxx += cxx; vyy += cyy; }
    }

    // A cabinet run rests FLAT on the floor: roll/pitch are pinned near 0 by that physical prior — they are
    // CONFIDENTLY known, not unknown. Same value and reasoning as the classic path (≈1.3° std).
    constexpr float flat_rp_var = 5e-4f;

    // Self-gate on the trace so a settled run stops rewriting its edge every cycle. Write when the run is
    // new to us or the uncertainty moved >5 % — matching CabinetSceneGraph::write_rt_covariance.
    const float trace = vxx + vyy + box.var_z + box.var_yaw;
    if (const auto prev = kitchen_cov_trace_.find(box.id); prev != kitchen_cov_trace_.end())
    {
        const float p = prev->second;
        if (std::isfinite(p) and p > 0.0f and std::abs(trace - p) <= 0.05f * p)
            return;
    }

    auto edge = G->get_edge(room_id, node_id, "RT");
    if (not edge.has_value())
        return;

    // 6×6 SE3 covariance, row-major [x,y,z,rx,ry,rz] — the layout every consumer reads (xx at 0, yy at 7,
    // yaw at 35). The XY cross-term IS filled here: unlike the classic diagonal-only write, the chart gives
    // us the real error ellipse, and a grazing run's is strongly off-axis.
    std::vector<float> cov(36, 0.0f);
    cov[0 * 6 + 0] = vxx;
    cov[0 * 6 + 1] = vxy;
    cov[1 * 6 + 0] = vxy;
    cov[1 * 6 + 1] = vyy;
    cov[2 * 6 + 2] = std::max(1e-9f, box.var_z);
    cov[3 * 6 + 3] = flat_rp_var;   // roll  (pinned flat on the floor)
    cov[4 * 6 + 4] = flat_rp_var;   // pitch (pinned flat on the floor)
    cov[5 * 6 + 5] = std::max(1e-9f, box.var_yaw);

    G->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov);
    G->insert_or_assign_edge(edge.value());
    kitchen_cov_trace_[box.id] = trace;

    if (cfg_.verbose_log)
        std::print("cabinet_concept: [kitchen] RT-cov '{}' σ x={:.1f}cm y={:.1f}cm z={:.1f}cm yaw={:.2f}°\n",
                   box.id, 100.0f * std::sqrt(std::max(0.0f, vxx)), 100.0f * std::sqrt(std::max(0.0f, vyy)),
                   100.0f * std::sqrt(std::max(0.0f, box.var_z)),
                   57.2958f * std::sqrt(std::max(0.0f, box.var_yaw)));
}

// Per-cycle CSV of the KITCHEN CELLS. In kitchen mode the fitter holds no instances, so ai2_log.csv is never
// written and birth_surprise.csv's cabinet columns are identically 0 (it reads fitter_->instances()) — this file
// is the only place the cells' state is observable. One row per cell that is ACTIVE or has any evidence this
// cycle (existence ≠ 0 / routed points), so an idle 30-cell table costs nothing. Columns: the derived room box,
// the two birth signals (existence log-odds, coverage EMA), and the LiDAR absence evidence that drives
// retirement (ex_n = beams that probed it; 0 ⇒ HOLD). Off unless cfg_.kitchen_cells_csv_path is set.
void SpecificWorker::log_kitchen_cells(std::size_t sweep_n, float mask_lidar_dr, int mask_lidar_n)
{
    if (cfg_.kitchen_cells_csv_path.empty()) return;
    if (not kitchen_cells_csv_.is_open())
    {
        // ROTATE, do not truncate: an intermittent fault is usually chased AFTER the restart it
        // provoked, and truncating loses exactly the run worth reading.
        if (not rc::diag::open_rotating(kitchen_cells_csv_, cfg_.kitchen_cells_csv_path,
                "cycle,cell,wall_seg,tier,active,node_id,existence,cov_ema,n_route,sweep_n,"
                "cx,cy,yaw,L,d,z0,z1,fe,ex_n,ex_occ,ex_free,ex_dL,retire_streak,"
                "std_d,std_z1,lat_min,lat_mean,lat_max,span,"
                "n_carcass,latmax_carcass,n_counter,latmax_counter,t0,t1,n_far,far_t_min,far_t_max,"
                "anchored,attach_seg,wall_gap,mask_lidar_dr,mask_lidar_n\n"))
        { cfg_.kitchen_cells_csv_path.clear(); return; }
    }
    const long cyc = ++kitchen_cells_cycle_;
    const auto row = [&](const std::string& id, int seg, int tier, bool active, std::uint64_t node_id,
                         float existence, float cov_ema, const rc::KitchenCellDiag& dg,
                         const rc::WallRunBelief* b, float fe)
    {
        float cx = 0, cy = 0, yaw = 0, L = 0, d = 0, z0 = 0, z1 = 0;
        if (b) b->room_box(cx, cy, yaw, L, d, z0, z1);
        kitchen_cells_csv_ << cyc << ',' << id << ',' << seg << ',' << tier << ',' << (active ? 1 : 0) << ','
                           << node_id << ',' << existence << ',' << cov_ema << ',' << dg.n_route << ',' << sweep_n << ','
                           << cx << ',' << cy << ',' << yaw << ',' << L << ',' << d << ',' << z0 << ',' << z1 << ','
                           << fe << ',' << dg.ex_n << ',' << dg.ex_occ << ',' << dg.ex_free << ',' << dg.ex_dL << ','
                           << dg.retire_streak << ','
                           // ── the belief's OWN uncertainty: how fast it locks on ──────────────
                           // Measured 2026-08-10: a run reaches its final depth by frame ~19 and then
                           // moves 2 mm in 730 cycles. Without sigma in the log you cannot see whether
                           // that is the belief becoming certain or the data simply agreeing.
                           << (b ? std::sqrt(std::max(0.0f, b->covariance()(2, 2))) : 0.0f) << ','
                           << (b ? std::sqrt(std::max(0.0f, b->covariance()(4, 4))) : 0.0f) << ','
                           // ── what the routed points actually said (see KitchenCellDiag) ─────
                           << dg.lat_min << ',' << dg.lat_mean << ',' << dg.lat_max << ',' << dg.span << ','
                           // which mask label put points where — the far points identify the culprit
                           << dg.n_carcass << ',' << dg.latmax_carcass << ','
                           << dg.n_counter << ',' << dg.latmax_counter << ','
                           // the run's own interval, so far_t_* can be read against it directly
                           << (b ? b->state().t0 : 0.0f) << ',' << (b ? b->state().t1 : 0.0f) << ','
                           << dg.n_far << ',' << dg.far_t_min << ',' << dg.far_t_max << ','
                           // ── island/peninsula chart provenance (island row only) ────────────
                           << (dg.anchored ? 1 : 0) << ',' << dg.attach_seg << ',' << dg.wall_gap << ','
                           << mask_lidar_dr << ',' << mask_lidar_n << '\n';
    };
    for (const auto& c : kitchen_mgr_.cells())
    {
        if (not c.active() and c.diag.n_route == 0 and std::abs(c.existence) < 1e-3f) continue;   // silent cell
        const auto it = kitchen_nodes_.find(c.geom.id);
        row(c.geom.id, c.geom.wall_seg_id, c.geom.tier, c.active(),
            it != kitchen_nodes_.end() ? it->second : 0, c.existence, c.cov_ema, c.diag, c.belief.get(), c.fe);
    }
    if (const auto* isl = kitchen_mgr_.island(); isl or kitchen_mgr_.island_diag().n_route > 0)
    {
        const auto it = kitchen_nodes_.find("island");
        row("island", -1, 0, isl != nullptr, it != kitchen_nodes_.end() ? it->second : 0,
            kitchen_mgr_.island_existence(), kitchen_mgr_.island_cov_ema(),
            kitchen_mgr_.island_diag(), isl, 0.0f);
    }
    kitchen_cells_csv_.flush();
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
    // ★Invariant 5: removal is a Bayesian decision on the existence log-odds, NEVER a miss counter. An
    //    armed death counter beside a live existence channel is a SECOND removal authority, and it is the
    //    one that carries no evidence and leaves no attributable record — a phantom analysis cannot tell a
    //    reasoned removal from a timeout. Tying it to the existence flag makes the two mutually exclusive
    //    by construction, and keeps them A/B-able: turn the channel off and the counter comes back exactly.
    //    cabinet's existence removal is OFF by default (CabinetModel.ExistenceRemovalEnabled=false), so TODAY
    //    the counter is its ONLY authority and stays armed — this changes nothing until that flag is turned
    //    on, at which point the counter stands down automatically instead of silently competing.
    tp.death_frames     = cfg_.existence_removal_enabled ? std::numeric_limits<int>::max()
                                            : cfg_.tracker_death_frames;
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

    // ★BIRTH EVIDENCE — the shared CREATE policy (common/instance_tracker/birth_evidence.h). This agent fed
    // the tracker `birth_evidence = 1.0` on EVERY compute cycle, so `birth_frames` counted cycles rather than
    // observations: at ~10 Hz compute against a ~9.5 Hz mask stream one mask frame was counted several times,
    // and "N frames" became well under a second of a single unchanging view — which is how a YOLO false
    // positive on a wall panel becomes furniture. Three rules, none of them a threshold: an OBSERVATION not a
    // cycle; birth admitted by the UPDATE rule (frame_admissible — a frame the fit would refuse may not create
    // an object); and an admissible observation still worth only its reliability (confidence x range).
    const bool birth_new_obs = pkt.valid and static_cast<long>(pkt.frame_id) > last_birth_mask_frame_;
    if (birth_new_obs)
        last_birth_mask_frame_ = static_cast<long>(pkt.frame_id);
    const rc::birth::Detectability birth_detect{cfg_.kitchen_periph_ref, cfg_.existence_absence_range_ref_m, cfg_.existence_absence_range_power};

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
            // ★★ONLY THE FRONT RGB-D CAMERA MAY CREATE OR UPDATE AN OBJECT. `has_depth` is NOT that
            // question: once the producer began depth-filling ricoh masks from reprojected LiDAR it
            // publishes them as full 3D slices with has_depth = 1, so a 360° detection from BEHIND the
            // robot passed every guard written as `if (has_depth)`. Reported live on bottle_concept —
            // moving and cloning with the robot facing away, 3 m off. mask_source says which camera,
            // unambiguously, and the retina has been publishing it all along. A ricoh slice may
            // still CONFIRM a live instance or raise a proto-object to go and look
                // at; it may not move one. ★THE MECHANISM HERE IS THIS AGENT'S OWN
                // process_ricoh_bearings, NOT common/bearing_confirm — that module is used by
                // bottle/chair/door, which consume BEARING-ONLY slices (has_depth=false, azimuth).
                // This agent instead consumes ricoh slices that carry 3D points from the LiDAR
                // depth-fill. Both are the same channel; naming the wrong one sent an audit
                // looking for a bearing_confirm call that was never going to be here.
            // See MaskIngestor::MaskSlice::may_fit_geometry.
            // cabinet alone had this rule, via the depth_var > 0 PROXY — right in effect but indirect, and
            // it breaks the day a zed mask carries a range variance. Ask the source directly.
            if (not sl.may_fit_geometry())
                continue;              // Ricoh drives the attention path (process_ricoh_bearings), not the fit.
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
                // ★STRICTLY WORSE THAN NOT ADOPTING, BEFORE THIS. The residual-corroboration boost was
                // applied raw, so a repeated (stale) mask from an inadmissible view advanced the streak by
                // MORE than one ideal observation — BirthFrames could be crossed in ~2 compute cycles of a
                // single smeared, low-confidence, distant detection. The boost is a corroboration factor and
                // belongs ON TOP of the shared per-observation weight, not instead of it.
                dv.birth_evidence = (1.0f + cfg_.birth_fusion_gain * s)
                                  * rc::birth::evidence({sl.confidence, sl.range}, birth_detect,
                                                        birth_new_obs, fitter_->frame_admissible(sl));

                // ★MUTUAL EXCLUSION — no two objects occupy the same space (SHARED, common/exclusion).
                // A continuous support multiplied into the birth evidence exactly like the others above, so a
                // candidate condensing onto ANOTHER CONCEPT's object never accrues enough to mature: it is not
                // vetoed, it is unsupported. Every agent already refused to fit two of its OWN instances to one
                // object; none ever asked what a different concept had claimed, which is how a refrigerator was
                // created on top of door_3 (16 cm apart, same width, same yaw) and then could not die.
                if (not foreign_claims_.empty())
                {
                    const rc::exclusion::Claim* who = nullptr;
                    // The candidate's own vertical band, derived the SAME way CabinetSceneGraph derives the
                    // newborn's z0 — centred on the slice, not planted on the floor, because a wall unit is
                    // born high. Without a band, a base candidate under a hood is charged for space it does
                    // not occupy, and a wall unit is charged for the base run beneath it.
                    const float cand_z0 = std::max(0.0f, dv.z - 0.5f * cfg_.tracker_birth_height_m);
                    const float unclaimed = rc::exclusion::p_unclaimed(
                        {dv.xy.x(), dv.xy.y(), cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, 0.0f}, foreign_claims_, &who,
                        cand_z0, cand_z0 + cfg_.tracker_birth_height_m);
                    dv.birth_evidence *= unclaimed;
                    if (unclaimed < 0.99f)
                        std::print("[cabinet] birth cand CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f}\n",
                                   who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed);
                }
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
        {
            fitter_->note_birth(new_id, c);   // full XYZ: z seeds the tier so a WALL unit is born high
            // ★SENIORITY IS OBSERVED AT BIRTH, not inferred later (common/exclusion). Birth is the only moment
            // at which "who was here first" is actually seen: resolving it on a later cycle would have a real
            // carcass, standing legitimately beside its neighbour in the run, wake up after a RESTART, find the
            // neighbour present, and declare ITSELF the junior. Anything not created this run stays senior.
            if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
            {
                // Same band CabinetSceneGraph gives the node it just created (z0 centred on the detection,
                // clamped to the floor). Seniority is recorded ONCE and never revisited, so a wall unit
                // stamped junior here for the base run below it would carry that for life.
                const float bz0 = std::max(0.0f, c.z() - 0.5f * cfg_.tracker_birth_height_m);
                it->second.exclusion.resolve_at_birth({c.x(), c.y(), cfg_.tracker_birth_width_m,
                                                      cfg_.tracker_birth_depth_m, 0.0f}, foreign_claims_,
                                                      bz0, bz0 + cfg_.tracker_birth_height_m);
            }
            // Shadow-mode birth record (CONCEPT_AGENT_LIFECYCLE.md §4.2): captures the place AND the
            // viewpoint that produced it, so a phantom that dies young is attributable to both.
            log_phantom_event("BIRTH", new_id, "", c.x(), c.y(), nullptr, "");
        }
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
    // Peripheral (ricoh-360) channel. The 94-line copy that used to live here — in four agents, with
    // three different behaviours across the seven — is now common/peripheral_channel. See that header
    // for why: same sensor, same question, three answers, and a comment in each copy naming a module
    // none of them called.
    ricoh_attention_targets_.clear();
    if (not inner_eigen_ or not mask_ingestor_)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);   // robot pose in room
    if (not rtb.has_value())
        return;
    const Eigen::Vector2f robot_xy{static_cast<float>(rtb.value()(0, 3)),
                                  static_cast<float>(rtb.value()(1, 3))};

    const auto dets = rc::peripheral::gather(mask_ingestor_->packet(), "cabinet", robot_xy,
                                             cfg_.ricoh_attention_conf);
    std::vector<rc::peripheral::TrackRef> tracks;
    tracks.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& st = inst.model.state();
        // A cabinet RUN is a long box: L along the run axis, d deep. Half-diagonal is the same
        // circumscribed radius the other agents use, just from this model's own field names.
        tracks.push_back({id, {st.cx, st.cy}, 0.5f * std::sqrt(st.L * st.L + st.d * st.d)});
    }

    rc::peripheral::Params pp;
    pp.angular_margin_rad = cfg_.ricoh_attention_angle_margin_rad;
    pp.range_band_m       = cfg_.ricoh_attention_range_band_m;
    const auto res = rc::peripheral::associate(tracks, dets, robot_xy, pp);

    // A MATCH IS EVIDENCE, not a no-op: confirm-only, e_free hard 0, so this channel can only push L up.
    // A ricoh miss charges nothing — the 360 detector's p_detect at a given range is uncharacterised, and
    // absence weighted by an unknown p_detect is the ratchet that has bitten this fleet before.
    if (cfg_.ricoh_confirm_enabled)
        for (const auto& cf : res.confirms)
        {
            auto& insts = fitter_->instances();
            if (const auto it = insts.find(cf.track_id); it != insts.end())
            {
                rc::exist::SensorModel sm;
                sm.detection_prob = cfg_.ricoh_confirm_detection_prob;
                sm.clutter_prob   = cfg_.ricoh_confirm_clutter_prob;
                const auto ev = rc::exist::mask_evidence(std::clamp(cf.confidence, 0.0f, 1.0f),
                                                         /*e_free=*/0.0f, /*n_detectable=*/1, sm);
                it->second.existence.integrate(ev, sm.detection_prob);
            }
        }

    for (const auto& at : res.attention)
        ricoh_attention_targets_.push_back({at.azimuth_rad, at.range_m, at.confidence, {at.xy.x(), at.xy.y()}});
    ev_g_.ricoh_attention = static_cast<int>(ricoh_attention_targets_.size());
}

