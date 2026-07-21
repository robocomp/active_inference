/*
 * specificworker_lifecycle.cpp — SpecificWorker's instance-lifecycle + ricoh-attention methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit) to keep the orchestrator lean. Owns:
 *  - merge_overlapping_instances : physical-exclusion collapse of two instances fitted to one cabinet,
 *  - run_instance_tracker        : the ONLY birth/associate/merge path (shared rc::InstanceTracker),
 *  - process_ricoh_bearings      : ricoh-360 bearing-only PERIPHERAL ATTENTION (never births/fits).
 */

#include "specificworker.h"
#include "cabinet_geometry.h"   // rc::geom::footprint_overlap_ratio

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
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
            const auto& sa = ia->second.model.state();
            const auto& sb = ib->second.model.state();

            // A base run and the wall units above it are DISTINCT runs that happen to share a footprint,
            // so an oriented-rectangle overlap would wrongly fuse them. Separate them by their vertical
            // band first: only merge instances whose carcasses actually overlap in z. (Two base runs meeting
            // at an L-corner have z overlap and different yaw — handled by the collinear test's angle gate.)
            const float z_overlap = std::min(sa.z1, sb.z1) - std::max(sa.z0, sb.z0);
            if (z_overlap <= 0.0f) continue;

            // COLLINEAR-RUN merge: the operator that turns "a run glimpsed in pieces" into one object.
            // Oriented-rectangle overlap (still used by table/chair) can never join two co-linear
            // NON-overlapping fragments — the dominant duplicate mode for a run — so use the segment test
            // against the pair's joint uncertainty. sigma_L gets a generous floor: fragments seen seconds
            // apart legitimately abut with a gap on the order of the mask's own coarseness.
            const float s_yaw = std::hypot(sd(ia->second, 2), sd(ib->second, 2)) + 0.05f;   // yaw index 2
            const float s_lat = std::hypot(sd(ia->second, 1), sd(ib->second, 1)) + 0.10f;   // lateral floor
            const float s_L   = std::hypot(sd(ia->second, 3), sd(ib->second, 3)) + cfg_.merge_gap_floor_m;
            const auto m = rc::geom::collinear_merge(sa, sb, s_yaw, s_lat, s_L, cfg_.merge_n_sigma);
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
void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = cfg_.tracker_death_frames;
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.multi_det_per_track = true;   // fuse multiple ZED slices of one cabinet (one belief update per slice)
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
            if (sl.label != "cabinet" or sl.support_end <= sl.support_begin)
                continue;
            if (sl.depth_var > 0.0f)   // RICOH is BEARING-ONLY: never births/associates/fits — the tracker sees
                continue;              // ZED masks only. Ricoh drives the attention path (process_ricoh_bearings).
            rc::DetectionView dv;
            dv.xy = {sl.centroid.x(), sl.centroid.y()};
            dv.slice_index = i;
            dv.birthable = true;       // ZED only reaches here → always birthable
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
    ev_g_.table_dets      = static_cast<int>(dets.size());   // shared field in common/dashboard/evidence_monitor.h
                                                             // (generic "detections of this agent's class"; name is
                                                             // historical — do NOT rename, it is used by table/chair too)
    ev_g_.assigned      = n_assigned;
    ev_g_.discarded     = static_cast<int>(dets.size()) - n_assigned;
    ev_g_.births       += static_cast<int>(res.births.size());
    ev_g_.births_cum   += static_cast<long>(res.births.size());
    if (++dbg % 30 == 0 or not res.births.empty() or not res.deaths.empty())
    {
        std::print("[tracker] instances={} cabinet_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& t : tracks)
            std::print("[tracker]   track id={} xy=({:.2f},{:.2f}) has_cov={}\n",
                       t.id, t.xy.x(), t.xy.y(), t.has_cov);
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
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

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the cabinet (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
    }

    // Per-instance ZED-slice count for the EvidenceMonitor. Every assigned slice is a ZED detection now
    // (ricoh is bearing-only and never reaches assigned_mask_idxs), so this is just the assignment count.
    for (auto& [id, inst] : fitter_->instances())
        inst.dbg_n_zed_slices = static_cast<int>(inst.assigned_mask_idxs.size());
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

