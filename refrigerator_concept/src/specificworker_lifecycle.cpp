/*
 * specificworker_lifecycle.cpp — SpecificWorker's instance-lifecycle + ricoh-attention methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit) to keep the orchestrator lean. Owns:
 *  - merge_overlapping_instances : physical-exclusion collapse of two instances fitted to one refrigerator,
 *  - run_instance_tracker        : the ONLY birth/associate/merge path (shared rc::InstanceTracker),
 *  - process_ricoh_bearings      : ricoh-360 bearing-only PERIPHERAL ATTENTION (never births/fits).
 */

#include <limits>
#include "specificworker.h"

#include "../../common/peripheral_channel/peripheral_channel.h"   // THE shared ricoh path
#include "../../common/detect_probe/detect_probe.h"   // rc::probe — the detector's truth table (SHARED)

#include "../../common/existence_belief/existence_belief.h"   // rc::exist — peripheral CONFIRM-ONLY
#include "refrigerator_geometry.h"   // rc::geom::footprint_overlap_ratio
#include "../../common/track/merge_instances.h"   // rc::track::merge_overlapping — the SHARED merge sweep
#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth — the SHARED CREATE policy
#include "../../common/exclusion/exclusion.h"               // rc::exclusion — the SHARED no-two-objects rule

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─── Instance lifecycle: merge / tracker / birth ─────────────────────────────────────────────────

// One mask slice's room-frame support points. Used both to score a birth candidate per-frame and to bank it
// into the candidate's burst, so the two always see exactly the same cloud. Bounds-checked: a truncated
// packet must not index past support_points.
std::vector<Eigen::Vector3f> SpecificWorker::slice_cloud(const rc::MaskIngestor::MasksPacket& pkt, int slice_index)
{
    if (slice_index < 0 or slice_index >= static_cast<int>(pkt.slices.size()))
        return {};
    const auto& sl = pkt.slices[slice_index];
    const std::size_t b = std::min<std::size_t>(sl.support_begin, pkt.support_points.size());
    const std::size_t e = std::min<std::size_t>(sl.support_end,   pkt.support_points.size());
    if (e <= b)
        return {};
    return {pkt.support_points.begin() + static_cast<std::ptrdiff_t>(b),
            pkt.support_points.begin() + static_cast<std::ptrdiff_t>(e)};
}

// Retire one instance: drop its affordance node, forget it in the fitter, delete its graph node. The single
// teardown path shared by every lifecycle exit (merge / death), keeping the affordance+fitter+graph invariant.
void SpecificWorker::retire_instance(std::uint64_t id)
{
    if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
        it->second.affordance.remove();
    fitter_->forget_node(id);
    G->delete_node(id);
}


// Data-driven multi-instance lifecycle (mirrors bottle_concept::run_instance_tracker). Refrigerators are large
// static furniture, so birth_min_sep is wide, death is off, and overlaps merge. The only path that
// creates/associates refrigerator instances (the prior-scaffold + greedy nearest-mask were removed in Stage 2).
// Collapse instances whose footprints overlap (same physical refrigerator fitted twice): keep the one with more
// integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a duplicate is
// gone before it is fed a mask. v1 keeps-best; precision-weighted DOF pooling is a later refinement.
void SpecificWorker::merge_overlapping_instances()
{
    if (cfg_.tracker_merge_overlap <= 0.0f)
        return;

    // The sweep is SHARED (common/track); only the two per-object questions stay here — when are two
    // fitted rectangles the same refrigerator, and what does retiring one mean. The counter is filled by the
    // sweep itself, because three agents proved a call-site counter gets forgotten.
    rc::track::merge_overlapping(
        fitter_->instances(), ev_g_,
        [&](const auto& a, const auto& b) -> std::optional<float>
        {
            const float ratio = rc::geom::footprint_overlap_ratio(a.model.state(), b.model.state());
            return ratio >= cfg_.tracker_merge_overlap ? std::optional{ratio} : std::nullopt;
        },
        [](const auto& in) { return in.matched_frames; },      // keep the more-observed instance
        [&](std::uint64_t keep, std::uint64_t drop, auto&, const auto&, float ratio)
        {
            std::print("refrigerator_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
                       drop, keep, ratio);
            retire_instance(drop);
        });
}

// One tracker cycle: merge overlaps, build tracks from live instances (Mahalanobis gate on belief Σ) and
// detections from this frame's ZED "refrigerator" slices, then apply the result (death / associate / birth). Ricoh
// slices are excluded here (bearing-only). This is the ONLY path that creates/associates refrigerator instances.
void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = std::numeric_limits<int>::max();   // never retire by occlusion (see refrigerator_config.h)
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.multi_det_per_track = true;   // fuse multiple ZED slices of one refrigerator (one belief update per slice)
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

    // Detections ← the current "refrigerator" mask slices (carry the slice index for the assignment). Built EVERY
    // cycle from the packet (even a stale one): the tracker needs continuous detections so its birth CANDIDATES
    // mature over birth_frames consecutive cycles and so association stays live — gating this on mask freshness
    // wiped the candidates every stale cycle and NOTHING ever birthed. Re-fitting a stale mask is prevented
    // separately, in process_refrigerator_node (per-instance frame_id gate), NOT here.
    // Only ZED-depth slices (depth_var==0) reach the tracker; ricoh (depth_var>0) is bearing-only and drives
    // the attention path (process_ricoh_bearings), never birth/associate/fit. Every ZED det is birthable.
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    // Plausibility params for the BIRTH-GATE (candidate-cloud "is this really a fridge?" score). Built once.
    rc::RefrigeratorBeliefParams plaus_p;
    plaus_p.plaus_aspect_scale = cfg_.plaus_aspect_scale;
    plaus_p.plaus_size_scale   = cfg_.plaus_size_scale;
    plaus_p.plaus_height_min   = cfg_.plaus_height_min;
    plaus_p.plaus_height_soft  = cfg_.plaus_height_soft;
    plaus_p.plaus_fe_ref       = cfg_.plaus_fe_ref;
    plaus_p.plaus_fe_scale     = cfg_.plaus_fe_scale;
    plaus_p.prior_footprint_m  = cfg_.ai2_prior_footprint_m;
    // ── Is this cycle a genuinely NEW observation? ────────────────────────────────────────────────────
    // The detections below are rebuilt EVERY cycle on purpose: a pending birth candidate that finds no matching
    // detection expires (instance_tracker.h swaps next_cand in), so skipping stale cycles would wipe every
    // candidate and nothing would ever birth. But PERSISTING a candidate and ACCRUING evidence into it are two
    // different things, and the old code did both — so `BirthFrames` counted COMPUTE CYCLES. At 10 Hz compute
    // against a ~9.5 Hz mask stream that meant one mask frame could be counted several times, and 8 "frames" was
    // well under a second of a single unchanging view. A YOLO false positive on a wall panel or a door easily
    // survives that. Below, a stale cycle contributes birth_evidence 0: the candidate is still matched and kept
    // alive, its streak simply does not grow. BirthFrames now means N distinct observations.
    // This is the birth-side instance of "repetition is not independence" — see [[existence-policy-unification]].
    const bool new_observation = pkt.valid and static_cast<long>(pkt.frame_id) > last_tracker_mask_frame_;
    if (new_observation)
        last_tracker_mask_frame_ = static_cast<long>(pkt.frame_id);

    // This sensor's detectability, the ONLY thing the shared birth policy needs from the agent. Deliberately the
    // same numbers the removal side weights absence by: an object the ZED cannot resolve well enough to DELETE
    // is one it cannot resolve well enough to CREATE.
    const rc::birth::Detectability birth_detectability{cfg_.ai2_periph_ref,
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
                // process_ricoh_bearings. (It used to say "NOT common/bearing_confirm"; that module was DELETED on
                // 2026-08-17 with no caller left — the pointer itself had become the wild goose chase it warned
                // about. The shared path is common/peripheral_channel.) The distinction that remains is
                // bottle/chair/door, which consume BEARING-ONLY slices (has_depth=false, azimuth).
                // This agent instead consumes ricoh slices that carry 3D points from the LiDAR
                // depth-fill. Both are the same channel; naming the wrong one sent an audit
                // which SLICE KIND each agent consumes — see peripheral_channel.h.
                // See MaskIngestor::MaskSlice::may_fit_geometry.
            if (sl.label != "refrigerator" or sl.support_end <= sl.support_begin or not sl.may_fit_geometry())
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
            // BIRTH GATE — "is this really a fridge?" on the raw candidate cloud. Scale the per-frame birth
            // evidence by the candidate footprint/height plausibility (aspect + size + z-range height), so an
            // ELONGATED or SHORT candidate (the mis-detection, e.g. a 70 cm cabinet) accumulates slowly / never
            // births, while a fridge-shaped one births normally. Continuous (∈(0,1]), not a gate. See
            // RefrigeratorBelief::candidate_plausibility.
            if (cfg_.fridge_filter_enabled)
            {
                const auto cloud = slice_cloud(pkt, i);
                const float cand_plaus = rc::RefrigeratorBelief::candidate_plausibility(cloud, plaus_p);
                dv.birth_evidence *= cand_plaus;
                if (cfg_.fridge_filter_log)
                    std::print("[fridge-filter] birth cand slice={} plaus={:.3f} → birth_ev={:.3f}\n",
                               i, cand_plaus, dv.birth_evidence);
            }
            // ROOM ENVELOPE. A fridge occupies room INTERIOR; a detection at or beyond the layout boundary is
            // already accounted for by the room model (masonry / the space beyond), so it must not read as
            // unexplained evidence for a new object. Continuous support ∈[0,1] — 0 outside, →1 once a fridge's
            // centre could physically stand there — multiplied into the birth evidence exactly like the shape
            // plausibility above, so it delays/denies maturation rather than hard-vetoing the detection. This is
            // the birth-side twin of the belief's one-sided wall mixture component; 1.0 when no room polygon is
            // known. Fixes fridges being born OUTSIDE the layout. See [[refrigerator-room-envelope-birth]].
            const float support = fitter_->interior_support(dv.xy);
            dv.birth_evidence *= support;

            // ★MUTUAL EXCLUSION — no two objects occupy the same space (SHARED, common/exclusion). The room
            // envelope above asks "could a fridge stand here at all"; this asks "is somebody else ALREADY
            // standing here". Both are continuous supports multiplied into the birth evidence, so a candidate
            // condensing onto another concept's object simply never accrues enough to mature — it is not
            // vetoed, it is unsupported. This is the defect that produced refrigerator_2 on top of door_3
            // (16 cm apart, same width, same yaw): every agent refused to fit two of its OWN instances to one
            // object and none of them ever asked what a DIFFERENT concept had claimed.
            if (not foreign_claims_.empty())
            {
                const rc::geom::Footprint cand{dv.xy.x(), dv.xy.y(),
                                               cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, 0.0f};
                const rc::exclusion::Claim* who = nullptr;
                // A fridge is born standing on the floor, so its band is [0, birth height]. A hood or a wall
                // unit above the candidate shares its footprint and none of its volume.
                const float unclaimed = rc::exclusion::p_unclaimed(cand, foreign_claims_, &who,
                                                                   0.0f, cfg_.tracker_birth_height_m);
                dv.birth_evidence *= unclaimed;
                if (unclaimed < 0.99f)
                    std::print("[fridge-filter] birth cand slice={} CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f} -> {:.3f}\n",
                               i, who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed, dv.birth_evidence);
            }
            if (support < 0.99f and cfg_.fridge_filter_log)
                std::print("[fridge-filter] birth cand slice={} OUTSIDE/at layout edge: interior_support={:.3f} → birth_ev={:.3f}\n",
                           i, support, dv.birth_evidence);

            // MASK QUALITY → how much this ONE observation is worth toward creating an instance. The policy is
            // shared (common/instance_tracker/birth_evidence.h) so every concept agent births on the same terms;
            // this agent only supplies its sensor's detectability, built once above. A birth used to count every
            // mask equally, so the marginal `refrigerator 0.53` blob on a wall in fridge_1.png matured exactly as
            // fast as a clean 0.90 detection of the real fridge, and a 7 m detection as fast as a 1.5 m one.
            // ★Birth is admitted by the UPDATE rule: the very same predicate that decides whether this frame may
            // move an EXISTING fridge's geometry (untruncated + robot still, or the mask well centred). A frame
            // the fit would refuse can never create an object — no second, weaker birth-only notion of "good
            // enough". On top of that the observation is worth its reliability (confidence · range).
            const bool admissible = fitter_->frame_admissible(sl);
            const rc::birth::MaskQuality mq{sl.confidence, sl.range};
            const float quality = rc::birth::evidence(mq, birth_detectability, new_observation, admissible);
            dv.birth_evidence *= quality;

            if (cfg_.fridge_filter_log)
                std::print("[fridge-filter] birth cand slice={} weight={:.3f} (conf={:.2f} range={:.1f}m){}{} → birth_ev={:.3f}\n",
                           i, quality, sl.confidence, sl.range,
                           new_observation ? "" : " STALE(0)",
                           admissible ? "" : " INADMISSIBLE(0)", dv.birth_evidence);
            dets.push_back(dv);
        }

    // Snapshot the ZED refrigerator-detection centroids (room frame) for the birth-surprise fusion probe (read at the
    // compute() tail in log_birth_surprise). Cheap; only used when cfg_.birth_surprise_probe is on.
    last_refrigerator_dets_xy_.clear();
    for (const auto& d : dets) last_refrigerator_dets_xy_.push_back(d.xy);

    const auto res = tracker_.update(tracks, dets);

    // ── BIRTH FRAGMENT: bank this observation's cloud against the candidate it fed ────────────────────
    // Runs BEFORE the birth loop below, so a candidate promoted this cycle has the promoting frame in its
    // burst too. Only genuinely new observations are banked: on a repeat frame_id the same cloud would be
    // re-added (harmless — it dedups) but n_obs/span would count our compute cadence instead of the sensor's,
    // which is exactly the confusion birth_evidence.h rule 1 exists to prevent.
    if (cfg_.birth_frag_enabled and pkt.valid and new_observation)
        for (int d = 0; d < static_cast<int>(dets.size()); ++d)
            if (res.cand_of_det[d] != 0)
                birth_frag_.accumulate(res.cand_of_det[d], slice_cloud(pkt, dets[d].slice_index),
                                       pkt.timestamp_ms, cfg_.birth_frag_cell_m,
                                       static_cast<std::size_t>(std::max(1, cfg_.birth_frag_max_pts)));
    birth_frag_.expire(res.expired_candidates);   // unconditional: frees bursts even if banking is disabled

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
        std::print("[tracker] instances={} refrigerator_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& t : tracks)
            std::print("[tracker]   track id={} xy=({:.2f},{:.2f}) has_cov={}\n",
                       t.id, t.xy.x(), t.xy.y(), t.has_cov);
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
    }

    // DEATH: a refrigerator is rigid, persistent furniture, so a long occlusion (no mask for many frames) is NOT
    // absence and must never retire it. The ONLY way a refrigerator is removed is the MERGE operator (two refrigerators
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

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection — but only if its
    // accumulated probation burst still reads as a fridge once seen whole. The burst is used ONLY for that
    // decision; the model is seeded with the detection XY alone, because cold-start geometry is owned by
    // run_inference's lazy snap and seeding it from the burst measurably hurt (see the ★ note there).
    for (std::size_t k = 0; k < res.births.size(); ++k)
    {
        const int d = res.births[k];
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;

        std::optional<rc::Burst> burst;
        if (cfg_.birth_frag_enabled and k < res.birth_cand_ids.size())
        {
            burst = birth_frag_.take(res.birth_cand_ids[k]);
            // LOCAL CONSISTENCY δ: observations too far apart in time are not one view of one object, so the
            // union is not a surface. The birth still stands (the streak matured) — only the cloud is dropped.
            if (burst and not rc::BirthFragment::span_ok(*burst, cfg_.birth_frag_delta_ms))
            {
                std::print("[fridge-filter] burst SPAN {} ms > delta {} ms → not locally consistent, "
                           "falling back to default birth geometry\n",
                           burst->span_ms(), cfg_.birth_frag_delta_ms);
                burst.reset();
            }
        }

        // ADMISSION on the whole burst. The same plausibility already applied per-frame as an evidence
        // multiplier, but re-run on the union — where the z-range finally spans the object instead of one
        // partial view, so it stops penalising real fridges — and read as a DECISION. A refused birth never
        // reaches DSR; without this the mis-detection is created and must be retracted later by
        // apply_fridge_filter, and in this agent `res.deaths` is ignored so that is the only way back out.
        if (burst and cfg_.fridge_filter_enabled and cfg_.birth_admit_plausibility > 0.0f)
        {
            const float plaus = rc::RefrigeratorBelief::candidate_plausibility(burst->pts, plaus_p);
            if (plaus < cfg_.birth_admit_plausibility)
            {
                std::print("[fridge-filter] BIRTH REFUSED at ({:.2f},{:.2f}): burst plaus={:.3f} < {:.3f} "
                           "({} pts over {} obs, {} ms)\n",
                           c.x(), c.y(), plaus, cfg_.birth_admit_plausibility,
                           burst->pts.size(), burst->n_obs, burst->span_ms());
                ++ev_g_.births_refused;
                ++ev_g_.births_refused_cum;
                // Recorded like a birth so a refusal is attributable to the same place + viewpoint.
                log_phantom_event("BIRTH_REFUSED", 0, "", c.x(), c.y(), nullptr, "");
                continue;
            }
        }

        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
        {
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
            // ★SENIORITY IS OBSERVED AT BIRTH, not inferred later (common/exclusion). Birth is the only moment
            // at which "who was here first" is actually seen; resolving it on some later cycle would have a
            // real fridge, standing legitimately beside a real cabinet, wake up after a RESTART, find the
            // cabinet already present, and declare ITSELF the junior. An instance we did not create this run
            // stays senior by default — the rule can fail to catch a collision, never invent one.
            if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
                it->second.exclusion.resolve_at_birth({c.x(), c.y(), cfg_.tracker_birth_width_m,
                                                       cfg_.tracker_birth_depth_m, 0.0f}, foreign_claims_,
                                                      0.0f, cfg_.tracker_birth_height_m);
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


// ─── "Is this really a fridge?" soft singleton + plausibility→existence retire ────────────────────

// Fold each fridge's accumulated shape-plausibility evidence (Part 1) AND a soft SINGLETON inhibition (a
// kitchen has ~one fridge — a stronger-evidence fridge inhibits weaker duplicates) into its existence
// log-odds each cycle, then retire the instances the combined evidence drives below the removal boundary
// (debounced). Everything is continuous + bounded: a genuine, strongly-supported second fridge survives; the
// weaker mis-detection (elongated/short cabinet mislabelled "refrigerator") decays and is removed. The worker
// owns this because the singleton term needs to see ALL instances at once. See RefrigeratorBelief::
// singleton_existence_deltas / fridge_log_evidence_ratio. Runs regardless of the sensor-existence channel.
void SpecificWorker::apply_fridge_filter()
{
    if (not cfg_.fridge_filter_enabled)
        return;
    auto& insts = fitter_->instances();
    if (insts.empty())
        return;

    // Snapshot (stable order) the accumulated plausibility evidence + current existence probability per instance.
    std::vector<std::uint64_t> ids;
    std::vector<float>         pe, px;
    ids.reserve(insts.size()); pe.reserve(insts.size()); px.reserve(insts.size());
    for (auto& [id, inst] : insts)
    {
        if (not inst.ai2_initialized) continue;   // only fitted instances participate (shape evidence is meaningful)
        // Judge the CURRENT fitted shape, but ONLY on a cycle that actually MEASURED it — matched_frames advances
        // solely inside run_inference, so comparing against plaus_seen_frames admits exactly the cycles that took
        // a geometric update. Re-scoring a frozen belief every cycle is not sequential Bayes, it is the SAME
        // evidence counted N times: it drove the accumulator to −clamp on an unchanged shape and deleted
        // well-fitted fridges with no sensor evidence at all (live: L falling 1.1/cycle while sndet=socc=sfree=0).
        // Same defect class as [[chair-flip-acc-repetition-defect]].
        //
        // The increment is the shape LOG-EVIDENCE RATIO vs "other furniture" (nats), clamped per cycle to
        // ±plaus_clamp so a single frame can at most cancel a settled accumulator, never invert it.
        //
        // TRADE-OFF this replaces: the old every-cycle rule also kept decaying a mis-detection that had diverged
        // to a cabinet shape and then coasted OUT OF FoV, so it could never be immortal. That decay is gone — but
        // it was decay without observation, the same thing that killed real fridges. A diverged mis-detection now
        // sits frozen until looked at, and one measured frame is enough (its LLR ≈ −17 saturates the accumulator
        // immediately). Meanwhile the silhouette channel HOLDs out-of-FoV and raises wants_verification, which is
        // the designed answer: send the robot to LOOK, never delete what it has not seen.
        if (inst.matched_frames != inst.plaus_seen_frames and not inst.dbg_gated)
        {
            inst.plaus_seen_frames = inst.matched_frames;
            inst.last_plausibility = inst.ai2_belief.fridge_log_evidence_ratio();
            const float step = std::clamp(inst.last_plausibility, -cfg_.plaus_clamp, cfg_.plaus_clamp);
            inst.plaus_evidence = std::clamp(inst.plaus_evidence + step, -cfg_.plaus_clamp, cfg_.plaus_clamp);
        }
        ids.push_back(id);
        pe.push_back(inst.plaus_evidence);
        px.push_back(inst.existence.p_exists());
    }
    if (ids.empty())
        return;

    // Per-instance existence-logodds delta = shape support (tanh of plaus_evidence) − singleton inhibition.
    const auto deltas = rc::RefrigeratorBelief::singleton_existence_deltas(
        pe, px, cfg_.plaus_to_existence_gain, cfg_.singleton_inhibition, cfg_.plaus_clamp);

    // The SAME policy the sensor debounce uses (rc::exist::RemovalPolicy) — see the note at the debounce below.
    rc::exist::RemovalPolicy plaus_policy;
    plaus_policy.logodds_max   = cfg_.existence_logodds_max;
    plaus_policy.removal_prob  = cfg_.existence_removal_prob;
    plaus_policy.remove_frames = static_cast<float>(cfg_.existence_remove_frames);

    std::vector<std::uint64_t> doomed;
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        auto it = insts.find(ids[k]);
        if (it == insts.end()) continue;
        auto& inst = it->second;
        inst.existence.set_max(cfg_.existence_logodds_max);
        bool plaus_doomed = false;
        // ★Apply the shape/singleton delta ONLY on a cycle that actually measured this instance. I gated the
        // ACCUMULATOR to measured cycles earlier but left the DELTA firing every cycle, so a frozen
        // plaus_evidence kept pushing the existence log-odds at a constant rate with no new evidence — the same
        // repetition-is-not-independence defect, one level up. Live: with the belief frozen (gated=1, npts=0)
        // and BOTH sensor channels silent (sfree_eff≈0, lfree_eff=0), ex_L still fell a steady −0.65/cycle from
        // +1.41 to the −4 clamp and the fridge was removed. Whatever the shape says, saying it again on a frame
        // that observed nothing is not evidence. Same rule as CREATE, UPDATE and the sensor REMOVE channels.
        // ★THE DEBOUNCE OBEYS THE SAME RULE AS THE DELTA. It used to advance on ANY cycle where the decision
        // said remove, so once L was below the boundary — however it got there, including from the sensor
        // channel — this streak ran to RemoveFrames on cycles that measured NOTHING and deleted the fridge.
        // That also made it the weaker of the two authorities: it would fire before the sensor streak
        // (which now accumulates p_detect), so fixing that one alone would have changed almost nothing here.
        // Advancing only on a MEASURED, ungated cycle is the same "saying it again on a frame that observed
        // nothing is not evidence" rule the comment above states for the delta.
        const bool measured = (inst.matched_frames == inst.plaus_seen_frames and not inst.dbg_gated);
        if (measured)
        {
            inst.existence.set(inst.existence.logodds() + deltas[k]);   // clamped by L_max

            // Removal debounce (dedicated streak, independent of the sensor-existence streak). A freshly-born
            // instance is given the SAME warmup grace as the belief aging (matched_frames_before_aging) so a
            // young genuine fridge is never retired before it has accrued its own shape evidence.
            //
            // ★IT TESTS THE SAME L AS THE SENSOR DEBOUNCE, SO IT MUST USE THE SAME POLICY. existence_belief.h
            // names this exact hazard: a second, weaker streak on the same belief "fires first and silently
            // masks any fix applied here". Hand-written, it counted CYCLES against a fixed RemoveFrames while
            // the sensor path counted LOOKS against a confidence-scaled requirement — so the weaker one
            // decided. Routed through rc::exist::decide_removal it inherits both rules.
            //
            // ★★AND A CHANNEL MAY ONLY EXECUTE A REMOVAL IT HAS ARGUED FOR. `deltas[k] < 0` is this channel's
            // own evidence AGAINST the object this cycle — bad shape, or out-competed by a stronger sibling.
            // Without that condition the debounce merely watched L, which any OTHER channel may have driven
            // down, and deleted on a case it never made.
            //
            // That is not hypothetical. hood_concept set PlausToExistenceGain = 0.0 on 2026-08-11 because an
            // identity prior may not vote on a hypothesis it does not model — which correctly zeroed
            // deltas[k], and left this debounce untouched, because the only guard on the whole path is
            // FridgeFilterEnabled. So the filter kept its trigger after losing its vote: the LiDAR free-space
            // carve walked L to the clamp and THIS counter did the killing, 15 cycles later, every time.
            // Measured over one 1.5 h run: 18 births and 18 deaths, and on every single one the SENSOR
            // debounce stood at 3.1-5.4 of the 15 it needs — it never fired once. Turning a channel's gain to
            // zero must silence it completely, and now does.
            const bool warm   = inst.matched_frames >= cfg_.matched_frames_before_aging;
            const bool argued = deltas[k] < 0.0f;
            plaus_doomed = rc::exist::decide_removal(inst.existence, inst.plaus_debounce, plaus_policy,
                                                     (warm and argued) ? 1.0f : 0.0f).remove;
        }

        if (cfg_.fridge_filter_log)
            std::print("[fridge-filter] {} plaus_ev={:.2f} llr={:+.2f} ΔL={:+.2f} L={:.2f} p={:.2f} streak={:.1f}\n",
                       inst.node_name, inst.plaus_evidence, inst.last_plausibility, deltas[k],
                       inst.existence.logodds(), inst.existence.p_exists(), inst.plaus_debounce.streak);

        if (plaus_doomed)
            doomed.push_back(ids[k]);
    }

    ev_g_.removals     += static_cast<int>(doomed.size());
    ev_g_.removals_cum += static_cast<long>(doomed.size());
    for (const std::uint64_t id : doomed)
    {
        std::print("refrigerator_concept: [fridge-filter] removing id={} — implausible as a fridge "
                   "(elongated/short/out-competed by a stronger fridge)\n", id);
        retire_instance(id);
    }
}


// ─── Ricoh 360 peripheral attention (bearing-only) ───────────────────────────────────────────────

// Ricoh 360 as PERIPHERAL ATTENTION. A ricoh detection's DIRECTION (bearing from the robot) is reliable even
// when its centroid/range is biased, so we use ONLY that: associate a ricoh bearing to an existing refrigerator if the
// bearing falls within that refrigerator's angular extent (its circumscribed radius over its range — a physical size,
// not a tuned gate). A ricoh bearing that NO refrigerator explains is UNASSIGNED → an attention target ("there is
// something refrigerator-like in that direction that the ZED hasn't confirmed; seek a ZED view there"). Never births,
// never fits — that is the ZED's job. This is step 1–3 of the peripheral→saccade→foveal design; the controller
// consuming the target (turn the ZED to the bearing) is step 4.
void SpecificWorker::process_ricoh_bearings()
{
    // Peripheral (ricoh-360) channel. gather → associate → confirm → what is left is ATTENTION: the whole
    // sequence is SHARED (rc::peripheral::run_cycle). Only the track list and the confirm sink are refrigerator's.
    ricoh_attention_targets_.clear();
    if (not inner_eigen_ or not mask_ingestor_ or not fitter_)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);   // robot pose in room
    if (not rtb.has_value())
        return;
    const Eigen::Vector2f robot_xy{static_cast<float>(rtb.value()(0, 3)),
                                   static_cast<float>(rtb.value()(1, 3))};

    std::vector<rc::peripheral::TrackRef> tracks;
    tracks.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& st = inst.model.state();
        tracks.push_back({id, {st.cx, st.cy}, 0.5f * std::sqrt(st.w * st.w + st.h * st.h)});
    }

    const auto out = rc::peripheral::run_cycle(
        mask_ingestor_->packet(), "refrigerator", robot_xy, tracks,
        {.detect_conf        = cfg_.ricoh_attention_conf,
         .angular_margin_rad = cfg_.ricoh_attention_angle_margin_rad,
         .range_band_m       = cfg_.ricoh_attention_range_band_m},
        [&](const rc::peripheral::Confirm& cf)
        {
            // A MATCH IS EVIDENCE: confirm-only, e_free hard 0, so this channel can only push L up. A ricoh
            // miss charges nothing — see the note on run_cycle for why there is no on_miss.
            if (not cfg_.ricoh_confirm_enabled)
                return;
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
        });

    ricoh_attention_targets_ = out.attention;
    ev_g_.ricoh_attention    = static_cast<int>(ricoh_attention_targets_.size());
    // Both counts: gathered vs unassigned. Zero attention with non-zero dets means the channel is WORKING
    // and everything it saw matched — the opposite conclusion from zero of both.
    fitter_->set_ricoh_counts(static_cast<int>(out.n_dets), ev_g_.ricoh_attention);
}



// ─── Detector truth table (rc::probe) ─────────────────────────────────────────────────────────────
//
// One row per cycle per live instance: WHERE the robot was standing, HOW the object projected, and what
// the detector did. See common/detect_probe/detect_probe.h for why each column exists — in short, the
// existing ai2 log can say "no mask this cycle" but cannot say from how many DISTINCT viewpoints, nor
// with what score it fired, nor whether the detector actually saw the object and named it something
// else. Those three gaps are what make a miss unattributable today.
//
// ★WHY THE ROBOT POSE IS THE POINT. Absence is integrated as if every frame were an independent trial.
// It is not — a miss from a pose repeats at 20 Hz from that same pose, so standing still manufactures
// ~100 refutations of a single look. Without (rx,ry,rtheta) that hypothesis cannot even be TESTED, so
// this column is the one that unblocks the whole question.
void SpecificWorker::log_detect_probe()
{
    if (cfg_.detect_probe_csv_path.empty() or not fitter_ or not mask_ingestor_)
        return;
    if (not rc::probe::ensure_open(detect_probe_csv_, cfg_.detect_probe_csv_path))
        return;

    // Pose is REQUIRED, not optional-with-a-fallback: a row whose viewpoint silently defaulted to the
    // origin would cluster with every other failed lookup and invent a viewpoint that never existed.
    // No pose ⇒ no row.
    if (not inner_eigen_)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);
    if (not rtb.has_value())
        return;
    const auto& Tb = rtb.value();
    const float rx = static_cast<float>(Tb(0, 3)), ry = static_cast<float>(Tb(1, 3));
    const float rtheta = static_cast<float>(std::atan2(Tb(1, 0), Tb(0, 0)));
    float cam_z = static_cast<float>(Tb(2, 3));
    if (const auto rtz = inner_eigen_->get_transformation_matrix("room", "zed", 0); rtz.has_value())
        cam_z = static_cast<float>(rtz.value()(2, 3));

    const auto& pkt = mask_ingestor_->packet();
    int n_all = 0, n_my = 0;
    if (pkt.valid)
        for (const auto& sl : pkt.slices)
        {
            if (not sl.is_zed()) continue;   // the ZED is the detector this file is about
            ++n_all;
            if (sl.label == "refrigerator") ++n_my;
        }

    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& s = inst.model.state();
        rc::probe::Sample row;
        row.cycle    = inst.processed_cycles;
        row.node     = inst.node_name;
        row.stamp_ms = inst.dbg_pkt_ts_ms;
        row.rx = rx; row.ry = ry; row.rtheta = rtheta; row.cam_z = cam_z;
        row.ocx = s.cx; row.ocy = s.cy; row.ocz = 0.5f * s.refrigerator_height;
        row.range_m       = inst.last_range;
        row.roi_fill      = inst.roi_fill;
        row.roi_fill_h    = inst.roi_fill_h;
        row.roi_fill_v    = inst.roi_fill_v;
        row.roi_valid     = inst.roi_valid;
        row.obliquity_cos = inst.dbg_obliquity_cos;
        row.trunc_frac    = inst.last_trunc_frac;
        row.p_detect      = inst.dbg_ex_pdetect;
        row.p_exists_prior = inst.existence.p_exists();   // PRIOR: this runs before the update below
        // ★THE LABEL, and it is not det_alive. detection_alive LATCHES across frames (measured on hood:
        // 1 in 9432/9432 rows, still 1 at frames_since_det=6172), so as an outcome it is degenerate and
        // fit_envelope correctly refuses to fit it. The per-cycle outcome is "a mask arrived THIS cycle".
        row.fired = (inst.frames_since_detection == 0);
        row.conf  = row.fired ? inst.last_mask_confidence : 0.0f;
        row.n_my = n_my; row.n_all = n_all;
        row.sibling = rc::probe::nearest_other_label(pkt, "refrigerator", {s.cx, s.cy});
        rc::probe::append(detect_probe_csv_, row);
    }
}
