/*
 * specificworker_lifecycle.cpp — SpecificWorker's instance-lifecycle + ricoh-attention methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit) to keep the orchestrator lean. Owns:
 *  - merge_overlapping_instances : physical-exclusion collapse of two instances fitted to one table,
 *  - run_instance_tracker        : the ONLY birth/associate/merge path (shared rc::InstanceTracker),
 *  - process_ricoh_bearings      : ricoh-360 bearing-only PERIPHERAL ATTENTION (never births/fits).
 */

#include <limits>
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule
#include "specificworker.h"

#include "../../common/peripheral_channel/peripheral_channel.h"   // THE shared ricoh path

#include "../../common/existence_belief/existence_belief.h"   // rc::exist — peripheral CONFIRM-ONLY
#include "table_geometry.h"   // rc::geom::footprint_overlap_ratio
#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth — the SHARED CREATE policy

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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


// Data-driven multi-instance lifecycle (mirrors bottle_concept::run_instance_tracker). Tables are large
// static furniture, so birth_min_sep is wide, death is off, and overlaps merge. The only path that
// creates/associates table instances (the prior-scaffold + greedy nearest-mask were removed in Stage 2).
// Collapse instances whose footprints overlap (same physical table fitted twice): keep the one with more
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

    std::unordered_set<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (removed.count(ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (removed.count(ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;

            const float ratio = rc::geom::footprint_overlap_ratio(ia->second.model.state(), ib->second.model.state());
            if (ratio < cfg_.tracker_merge_overlap) continue;

            // Keep the more-observed instance (more integrated fresh frames); retire the other.
            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("table_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
                       drop, keep, ratio);
            ++ev_g_.merges; ++ev_g_.merges_cum;   // EvidenceMonitor counter
            retire_instance(drop);
            removed.insert(drop);
            if (drop == ids[i]) break;   // this i is gone; advance to the next i
        }
    }
}

// One tracker cycle: merge overlaps, build tracks from live instances (Mahalanobis gate on belief Σ) and
// detections from this frame's ZED "table" slices, then apply the result (death / associate / birth). Ricoh
// slices are excluded here (bearing-only). This is the ONLY path that creates/associates table instances.
void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = std::numeric_limits<int>::max();   // never retire by occlusion (see table_config.h)
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.multi_det_per_track = true;   // fuse multiple ZED slices of one table (one belief update per slice)
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
            const auto& S = inst.ai2_belief.covariance();   // Σ over [cx,cy,H,w,h,yaw]
            t.cov = Eigen::Matrix2f::Zero();
            t.cov(0, 0) = S(0, 0) + inst.chain_cov_xx;
            t.cov(1, 1) = S(1, 1) + inst.chain_cov_yy;
            t.has_cov = true;
        }
        tracks.push_back(t);
        inst.assigned_mask_idxs.clear();   // cleared; re-filled below with every slice associated this cycle
    }

    // Detections ← the current "table" mask slices (carry the slice index for the assignment). Built EVERY
    // cycle from the packet (even a stale one): the tracker needs continuous detections so its birth CANDIDATES
    // mature over birth_frames consecutive cycles and so association stays live — gating this on mask freshness
    // wiped the candidates every stale cycle and NOTHING ever birthed. Re-fitting a stale mask is prevented
    // separately, in process_table_node (per-instance frame_id gate), NOT here.
    // Only ZED-depth slices (depth_var==0) reach the tracker; ricoh (depth_var>0) is bearing-only and drives
    // the attention path (process_ricoh_bearings), never birth/associate/fit. Every ZED det is birthable.
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    // ── Is this cycle a genuinely NEW observation? (rc::birth rule 1) ─────────────────────────────────
    // The detections below are rebuilt EVERY cycle on purpose (see the comment above): a pending candidate that
    // finds no matching detection expires, so skipping stale cycles would wipe every candidate and nothing would
    // ever birth. But PERSISTING a candidate and ACCRUING evidence into it are different things, and conflating
    // them made BirthFrames count COMPUTE CYCLES. At 10 Hz compute against a ~9.5 Hz mask stream one mask frame
    // was counted several times — which is how table_2 (a YOLO blip that appears in 4 of 146 logged detection
    // rows, cycles 202–205) became permanent furniture at cycle 203. A stale cycle now contributes 0: the
    // candidate stays alive, its streak simply does not grow, so BirthFrames means N distinct OBSERVATIONS.
    const bool new_observation = pkt.valid and static_cast<long>(pkt.frame_id) > last_tracker_mask_frame_;
    if (new_observation)
        last_tracker_mask_frame_ = static_cast<long>(pkt.frame_id);
    // This sensor's detectability — the only thing the shared birth policy needs from the agent. Deliberately
    // the SAME numbers the removal side weights absence by: an object the ZED cannot resolve well enough to
    // DELETE is one it cannot resolve well enough to CREATE.
    const rc::birth::Detectability birth_detectability{0.50f,
                                                       cfg_.existence_absence_range_ref_m,
                                                       cfg_.existence_absence_range_power};
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
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
            if (sl.label != "table" or sl.support_end <= sl.support_begin or not sl.may_fit_geometry())
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
            // MASK QUALITY → what this ONE observation is worth toward CREATING a table. The policy is shared
            // (common/instance_tracker/birth_evidence.h) so every concept agent births on the same terms; this
            // agent supplies only its sensor's detectability, built once above. Two rules, neither a threshold:
            // a repeat frame_id is worth 0 (rule 1), and a frame the FIT would refuse can never create an object
            // (rule 2 — birth is admitted by the UPDATE rule, here the fixation gate). On top of that the
            // observation is worth its reliability: confidence · (range_ref/range)^power (rule 3).
            //
            // ★Without this, corroboration alone decided birth: table_2's residual mass 42.3 gave
            // 1 + 6·(42.3/50.3) = 6.05 evidence/frame, so the nominal 8-frame debounce was satisfied in TWO
            // compute cycles — and the residual grid says "something unexplained is here", never "a table is
            // here" (a wall or a box scores the same). The quality factor is what makes the corroboration
            // bonus ride on a real, admissible, confident observation instead of substituting for one.
            const bool admissible = fitter_->frame_admissible(sl);
            const rc::birth::MaskQuality mq{sl.confidence, sl.range};
            dv.birth_evidence *= rc::birth::evidence(mq, birth_detectability, new_observation, admissible);

            // ★MUTUAL EXCLUSION — no two objects occupy the same space (SHARED, common/exclusion).
            // A continuous support multiplied into the birth evidence exactly like the others above, so a
            // candidate condensing onto ANOTHER CONCEPT's object never accrues enough to mature: it is not
            // vetoed, it is unsupported. Every agent already refused to fit two of its OWN instances to one
            // object; none ever asked what a different concept had claimed, which is how a refrigerator was
            // created on top of door_3 (16 cm apart, same width, same yaw) and then could not die.
            if (not foreign_claims_.empty())
            {
                const rc::exclusion::Claim* who = nullptr;
                // A table is born standing on the floor: band [0, birth height]. Anything ON it (a bottle) or
                // over it no longer reads as a claim on the space the table itself occupies.
                const float unclaimed = rc::exclusion::p_unclaimed(
                    {dv.xy.x(), dv.xy.y(), cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, 0.0f}, foreign_claims_, &who,
                    0.0f, cfg_.tracker_birth_height_m);
                dv.birth_evidence *= unclaimed;
                if (unclaimed < 0.99f)
                    std::print("[table] birth cand CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f}\n",
                               who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed);
            }
            dets.push_back(dv);
        }

    // Snapshot the ZED table-detection centroids (room frame) for the birth-surprise fusion probe (read at the
    // compute() tail in log_birth_surprise). Cheap; only used when cfg_.birth_surprise_probe is on.
    last_table_dets_xy_.clear();
    for (const auto& d : dets) last_table_dets_xy_.push_back(d.xy);

    const auto res = tracker_.update(tracks, dets);

    // Diagnostic (throttled, plus on any birth/death): reveals "1 slice" (upstream) vs "N slices, no birth".
    static int dbg = 0;
    const int n_assigned = static_cast<int>(std::count_if(res.assignment.begin(), res.assignment.end(),
                                                          [](int a){ return a >= 0; }));

    // EvidenceMonitor mask/tracker counters (merges are added in merge_overlapping_instances above).
    ev_g_.mask_frame_id = pkt.valid ? pkt.frame_id : -1;
    ev_g_.total_slices  = pkt.valid ? static_cast<int>(pkt.slices.size()) : 0;
    ev_g_.class_dets    = static_cast<int>(dets.size());
    ev_g_.assigned      = n_assigned;
    ev_g_.discarded     = static_cast<int>(dets.size()) - n_assigned;
    ev_g_.births       += static_cast<int>(res.births.size());
    ev_g_.births_cum   += static_cast<long>(res.births.size());
    if (++dbg % 30 == 0 or not res.births.empty() or not res.deaths.empty())
    {
        std::print("[tracker] instances={} table_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& t : tracks)
            std::print("[tracker]   track id={} xy=({:.2f},{:.2f}) has_cov={}\n",
                       t.id, t.xy.x(), t.xy.y(), t.has_cov);
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
    }

    // DEATH: a table is rigid, persistent furniture, so a long occlusion (no mask for many frames) is NOT
    // absence and must never retire it. The ONLY way a table is removed is the MERGE operator (two tables
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
    // fitter with the detection XY so the model starts AT the table (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
        {
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
            // ★SENIORITY IS OBSERVED AT BIRTH, not inferred later (common/exclusion). Birth is the only
            // moment at which "who was here first" is actually seen: resolving it on a later cycle would
            // have a real object, standing legitimately beside a real neighbour, wake up after a RESTART,
            // find the neighbour present, and declare ITSELF the junior. Anything not created this run
            // stays senior by default — the rule can fail to catch a collision, never invent one.
            if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
                it->second.exclusion.resolve_at_birth({c.x(), c.y(), cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, 0.0f},
                                                      foreign_claims_, 0.0f, cfg_.tracker_birth_height_m);
            // Shadow-mode record (§4.2): a phantom is a birth that dies young from a confident view, so the
            // birth half is where the place + viewpoint that produced it is captured.
            log_phantom_event("BIRTH", new_id, "", c.x(), c.y(), nullptr, "");
        }
    }

    // Per-instance ZED-slice count for the EvidenceMonitor. Every assigned slice is a ZED detection now
    // (ricoh is bearing-only and never reaches assigned_mask_idxs), so this is just the assignment count.
    for (auto& [id, inst] : fitter_->instances())
        inst.dbg_n_zed_slices = static_cast<int>(inst.assigned_mask_idxs.size());
}


// ─── Ricoh 360 peripheral attention (bearing-only) ───────────────────────────────────────────────

// Ricoh 360 as PERIPHERAL ATTENTION. A ricoh detection's DIRECTION (bearing from the robot) is reliable even
// when its centroid/range is biased, so we use ONLY that: associate a ricoh bearing to an existing table if the
// bearing falls within that table's angular extent (its circumscribed radius over its range — a physical size,
// not a tuned gate). A ricoh bearing that NO table explains is UNASSIGNED → an attention target ("there is
// something table-like in that direction that the ZED hasn't confirmed; seek a ZED view there"). Never births,
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

    const auto dets = rc::peripheral::gather(mask_ingestor_->packet(), "table", robot_xy,
                                             cfg_.ricoh_attention_conf);
    std::vector<rc::peripheral::TrackRef> tracks;
    tracks.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& st = inst.model.state();
        tracks.push_back({id, {st.cx, st.cy}, 0.5f * std::sqrt(st.w * st.w + st.h * st.h)});
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
    if (fitter_) fitter_->set_ricoh_attention(ev_g_.ricoh_attention);   // → ai2_log ricoh_attn
}

