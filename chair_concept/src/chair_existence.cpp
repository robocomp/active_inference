/*
 * chair_existence.cpp — see chair_existence.h. Moved verbatim from SpecificWorker::update_existence_beliefs();
 * only the inputs (now passed in) and the removal teardown (now the worker's on_remove) changed.
 */

#include "chair_existence.h"

#include "chair_fitter.h"
#include "../../common/existence_belief/existence_belief.h"   // rc::exist:: (SHARED policy)
#include "../../common/contour_edge/contour_edge_check.h"     // rc::edges — classifier-free RGB contour check
#include "../../common/contour_edge/contour_depth_check.h"    // rc::edges — its metric (depth) half
#include "../../common/diag_log/rotating_csv.h"               // keep the previous run instead of wiping it

#include <algorithm>
#include <cmath>
#include <fstream>
#include <print>
#include <vector>

namespace rc {

void ChairExistence::update_and_remove(ChairFitter& fitter, const Inputs& in,
                                       const std::function<void(std::uint64_t, ChairInstance&)>& on_remove)
{
    // Integrate at the SENSOR rate, not the compute rate: only when a new mask frame arrived. Otherwise a fast
    // compute loop would decay a briefly-occluded real chair away between two sensor frames.
    if (not in.pkt)
        return;
    const auto& pkt = *in.pkt;
    const bool sensor_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != last_mask_frame_;
    if (not sensor_fresh)
        return;
    last_mask_frame_ = static_cast<int>(pkt.frame_id);

    if (support_scale_ <= 0.0f)                 // lazy-seed the expected-support scale C = E[npts·range²]
        support_scale_ = cfg_.exist_expected_support_c;

    const float g   = cfg_.exist_evidence_gain;
    const float ar  = cfg_.exist_adequacy_ref;
    const float cap = cfg_.exist_adequacy_cap;

    // "ZED removes": every mask frame is ZED-cadenced (upload_masks runs per ZED RGBD frame, ricoh slices appended),
    // so a fresh frame ALREADY means "ZED looked this cycle" — including a frame with ZERO ZED detections, which is
    // exactly ZED staring at empty space and finding nothing (the strongest vacate evidence). Do NOT gate on
    // "ZED produced a detection": that would refuse to remove a phantom precisely when ZED confirms it is gone.
    // ★ONE REMOVAL DECISION, THE SHARED ONE (rc::exist::decide_removal). chair was the last agent deciding
    // by hand — a bare `L < RemoveLogodds` with NO debounce, the only agent in the fleet able to delete an
    // instance on a single frame. The boundary is preserved EXACTLY: RemoveLogodds = -3.0 is the same test
    // as should_remove(p) with p = 1/(1+e^3) = 0.04743, so nothing about WHEN a chair is condemned changes.
    // What changes is that the decision must now be sustained, in the fleet's unit (IDEAL OBSERVATIONS,
    // Sum p_vis) rather than cycles, and that a condemned-but-unexecutable instance reports itself.
    //
    // ★MEASURED FIRST, on chair's own etc/chair_existence_log.csv, because a debounce chosen blind is how
    // this family of bugs starts. Two facts decided it:
    //   · on the cycles the chair was NOT detected, zed_pd (its p_vis) has median 0.000 and p90 0.000 —
    //     the SENSOR channel is very nearly inert for removal (chair_1/chair_2 sat pinned at L=+4.00 for
    //     1554 rows each with zed_pd median 0.000). Counting LOOKS therefore holds those chairs rather
    //     than deleting them, which is correct: a camera that cannot resolve the chair has not seen it go.
    //   · the ROOM PRIOR is the channel that actually fires, and it is visibility-INDEPENDENT by
    //     construction, drawing OutOfRoomGain = 1.5 nats every frame — so +4 to -3 is under FIVE frames.
    //     chair_3 reached L = -2.70 and recovered: it came within 0.30 nats of being deleted by a
    //     localization wobble. That path is where the debounce has to bite, and it does: a prior passes
    //     p_vis = 1.0, so RemoveFrames = 15 means 15 frames of sustained "outside the room".
    rc::exist::RemovalPolicy policy;
    policy.logodds_max   = cfg_.exist_max_logodds;
    policy.removal_prob  = 1.0f / (1.0f + std::exp(-cfg_.exist_remove_logodds));   // exact same boundary
    policy.remove_frames = static_cast<float>(cfg_.exist_remove_frames);

    // ── CONTOUR channel inputs for this cycle (classifier-free; common/contour_edge) ─────────────────────
    // Each plane is scored only on a frame it has not scored yet, and its contour is projected at THAT
    // frame's own capture stamp — the RGB and depth planes need not carry the same instant, and a depth
    // frame compared against an RGB-stamped projection would be judged at the wrong pose while turning.
    // The RGB gradient is prepared ONCE per frame and scored for every chair (prepare once, score many).
    const cv::Mat* ct_rgb   = (in.rgb   and not in.rgb->empty())   ? in.rgb   : nullptr;
    const cv::Mat* ct_depth = (in.depth and not in.depth->empty()) ? in.depth : nullptr;
    const std::uint64_t ct_rgb_stamp   = ct_rgb   ? in.rgb_stamp_ms   : 0;
    const std::uint64_t ct_depth_stamp = ct_depth ? in.depth_stamp_ms : 0;
    const bool ct_rgb_fresh   = cfg_.exist_contour_check and ct_rgb_stamp > 0 and ct_rgb_stamp != contour_last_stamp_ms_;
    const bool ct_depth_fresh = cfg_.exist_contour_check and ct_depth_stamp > 0 and ct_depth_stamp != contour_last_depth_stamp_ms_;
    rc::edges::PreparedFrame ct_prep;
    if (ct_rgb_fresh)
    {
        ct_prep = rc::edges::prepare_frame(*ct_rgb);
        contour_last_stamp_ms_ = ct_rgb_stamp;
    }
    if (ct_depth_fresh)
        contour_last_depth_stamp_ms_ = ct_depth_stamp;

    std::vector<std::uint64_t> to_remove;
    for (auto& [id, inst] : fitter.instances())
    {
        if (std::isnan(inst.exist_logodds))           // seed on first visit (fresh birth OR adopted graph node)
        {
            inst.existence.set_max(cfg_.exist_max_logodds);
            inst.existence.set(cfg_.exist_birth_logodds);
            inst.exist_logodds = inst.existence.logodds();
        }

        // ROOM-CONTAINMENT POSE PRIOR (runs BEFORE the frustum gate, so it reaches a chair a localization glitch
        // put OUTSIDE the walls / behind a wall where the sensor can never vacate it): P(chair outside) ≈ 0, so
        // an out-of-room centre draws a STRONG negative every frame regardless of visibility → removed in a few.
        if (cfg_.exist_room_prior and fitter.has_room_polygon() and not inst.is_bearing_hypothesis)
        {
            const auto& ms = inst.model.state();
            if (not fitter.point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m))
            {
                // A containment violation is fully resolvable — the polygon is a trusted nominal model —
                // so p_vis = 1 and the whole ratio applies. Through the shared policy like every other
                // channel, so it inherits the clamp and the correlation handling.
                inst.existence.integrate(1.0f, -cfg_.exist_out_of_room_gain);
                inst.exist_logodds = inst.existence.logodds();
                // A containment violation is fully resolvable — the polygon is a trusted nominal model — so
                // this cycle is worth one IDEAL observation and the debounce advances at full weight. Same
                // treatment as door_concept's two prior channels: the difference from a sensor channel is
                // argued, and expressed in the shared unit rather than by skipping the shared decision.
                if (rc::exist::decide_removal(inst.existence, inst.existence_debounce, policy, 1.0f).remove)
                    to_remove.push_back(id);
                continue;   // outside the room → no sensor evidence can rescue it; skip the normal channels
            }
        }

        // HOLD (no evidence) unless the instance is a depth chair in the camera frustum with a real belief.
        // roi_valid is one compute-cycle stale (fine for a frustum test); a bearing-only hypothesis carries no
        // depth (existence unjudgeable from support mass); an un-initialised newborn hasn't had a chance yet.
        if (inst.is_bearing_hypothesis or not inst.roi_valid or not inst.ai2_initialized)
            continue;

        // TWO evidence channels for whether a chair really occupies this in-frustum spot. Each sets the
        // pair the shared policy wants: p_vis (could this look have resolved it?) and the log-ratio.
        float llr;                 // the p_vis-weighted product, kept for the CSV/diagnostics
        float p_vis  = 1.0f;       // P(this probe could have resolved the chair | it exists)
        float ratio  = 0.0f;       // log[ P(outcome|exists) / P(outcome|¬exists) ], unweighted
        const bool won = inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(pkt.slices.size());
        if (won)
        {
            // WON a mask → how much of the EXPECTED chair silhouette does the model actually EXPLAIN?
            //   explanation = (support / expected_at_range) × (1 − clutter_frac).
            // A real chair explains a good fraction → POSITIVE. Two failure modes both explain ≈nothing → NEGATIVE:
            // a far-too-SPARSE won mask (chair_3's ~8 points), OR a big but ~ALL-CLUTTER blob (chair_3/chair_5's
            // 5994/1126 pts at clutter≈0.99 — support present, but the chair model fits none of it). Folding
            // clutter into the SIGN is the fix for the "won a garbage blob → scored neutral → never removed" gap.
            const auto& sl = pkt.slices[inst.assigned_mask_idx];
            const int npts = (sl.support_end > sl.support_begin) ? static_cast<int>(sl.support_end - sl.support_begin) : 0;
            const float range    = std::max(0.5f, inst.last_range);
            const float expected = support_scale_ / (range * range);
            const float adequacy = std::clamp(static_cast<float>(npts) / std::max(1.0f, expected), 0.0f, cap);
            const float explained = adequacy * std::clamp(1.0f - inst.last_clutter_frac, 0.0f, 1.0f);
            llr = (explained >= ar) ? g * (explained - ar) / std::max(1e-3f, cap - ar)
                                    : -g * (ar - explained) / std::max(1e-3f, ar);
            // Be-still invariant, CONTINUOUS: an unreliable frame (moving AND off-axis → smeared/high-clutter) only
            // CONFIRMS — scale its NEGATIVE evidence by the frame reliability ∈ [0,1] so it can hold/raise existence
            // (positive kept full) but not argue the chair away. Reliability→1 for a still or well-centred frame.
            // ★RICOH CONFIRMS, NEVER REMOVES: a won ricoh slice (depth_var>0, or bearing-only) has unreliable
            // depth/clutter → it may CONFIRM existence (its win already reset frames_since_detection) but must not
            // produce NEGATIVE evidence. Only a ZED win (depth_var==0) may argue a chair down.
            const bool won_zed = sl.has_depth and sl.depth_var == 0.0f;
            ratio = llr;                               // the unweighted evidence this mask carries
            if (llr < 0.0f)
            {
                // A refuting look is only worth what the frame could resolve — and a ricoh win may never
                // refute at all (unreliable depth/clutter), which is p_vis = 0, i.e. a HOLD.
                p_vis = won_zed ? fitter.frame_reliability(inst) : 0.0f;
                llr   = ratio * p_vis;
            }
        }
        else
        {
            // WON NOTHING while in the frustum, UNOCCLUDED, on a ZED-active frame → absence evidence, but weighted
            // TWO ways: (1) CONFIDENCE ramps with frames_since_detection (freshness-as-precision, anti-death-spiral
            // — a chair that just lost the slice or is briefly hidden barely moves and recovers on its next win);
            // (2) ZED EXPECTED-DETECTABILITY pd ∈ [0,1] — absence only removes to the degree ZED would RELIABLY
            // have detected a present chair (falls off toward the image edge + with range). A far/peripheral chair
            // that only ricoh can see has pd≈0 → it HOLDs (maintained by ricoh confirmations) and is removed only
            // once a clean, close, centred ZED look comes up empty. This is "ZED removes".
            const float conf = (cfg_.exist_vacate_confident_frames > 0)
                ? std::clamp(static_cast<float>(inst.frames_since_detection)
                             / static_cast<float>(cfg_.exist_vacate_confident_frames), 0.0f, 1.0f)
                : 0.0f;
            // pd FLOOR (clear line of sight): occlusion is already HELD above, so here the LoS is clear. Even a
            // peripheral chair (low pd) then vacates at ≥ the floor rate, so a glitch-stranded phantom the robot
            // never centres still dies over time; conf gates on staleness so a recently-seen chair is untouched.
            const float pd = std::max(fitter.zed_detectability(inst), cfg_.exist_zed_clear_los_floor);
            // ★OCCLUSION SCALES ABSENCE, IT NO LONGER SKIPS THE CYCLE. This used to `continue` on a boolean
            // los_occluded(), which is a HOLD with no way out: a phantom chair born inside the dining set is
            // permanently "occluded" by the very furniture it overlaps, so its absence was never charged and
            // L froze short of the removal floor. Measured live: chair_3 pinned at L = -1.94958 for 300+
            // cycles with won=0, since_det=1814 and occluded=1 — while zed_pd said 0.53, i.e. the two
            // visibility models flatly disagreed and the pessimistic one won by being a gate.
            //
            // Genuine occlusion still protects a real chair: strength → 1 leaves ~no absence evidence, which
            // is the behaviour the old branch was after. What changes is that a PARTIALLY or SPURIOUSLY
            // occluded phantom keeps vacating, slowly, instead of living for ever.
            const float occ = cfg_.exist_occlusion_check ? fitter.los_occlusion(inst) : 0.0f;
            // p_vis is precisely "how well could this look have resolved a chair that IS here":
            // staleness confidence x ZED detectability x how much line of sight is left.
            p_vis = std::clamp(conf * pd * std::clamp(1.0f - occ, 0.0f, 1.0f), 0.0f, 1.0f);
            ratio = -g;                                // absence, at full strength, before visibility
            llr   = ratio * p_vis;
        }

        // ★THROUGH THE SHARED POLICY. `llr` above is already the p_vis-weighted product (each branch
        // scales by conf / zed_pd / (1-occlusion) / frame_reliability), so p_vis is handed in separately
        // and the ratio recovered — the class then owns the clamp, the frame-correlation decorrelation
        // and the removal boundary, identically to refrigerator/table/cabinet/door.
        inst.existence.integrate(p_vis, ratio);
        inst.exist_logodds = inst.existence.logodds();

        // ── CONTOUR channel (classifier-free) — the YOLO-blind rescue ────────────────────────────────
        // ★THE ONLY EVIDENCE HERE THAT DOES NOT END AT THE DETECTOR. Both channels above are YOLO's: when
        // the network stops painting "chair" on a chair already in the model — a viewpoint it dislikes, a
        // chair half under a table, too close — the not-won branch charges that as ABSENCE OF THE CHAIR
        // rather than absence of the label. This asks the image itself, as door_concept and
        // refrigerator_concept do (common/contour_edge):
        //   RGB   — is there an intensity boundary along the chair's projected SOLID outline, more than
        //           along the same outline slid sideways along the floor? Relative (gradient has no
        //           absolute scale), so it NEEDS a surviving control and stays silent without one.
        //   DEPTH — is there a surface at the depth the belief predicts, with space behind its edge?
        //           Absolute (metres are metres), so it needs no null — and it is the half that can say
        //           "we are looking straight through where the chair is supposed to be".
        // Both are bounded and dimensionless, so they SUM, and the sum passes through ONE tanh: two
        // consequences of the same fact must not each buy a full observation.
        //
        // ★IN THIS AGENT'S UNIT, NOT THE FRIDGE'S. rc::exist::contour_evidence scales by log(pd/pc) ≈ 2.83
        // nats; chair's channels are denominated in Existence.EvidenceGain (0.15/frame — the not-won branch
        // charges exactly -g at full visibility). At 2.83 the contour would outweigh the chair's own mask
        // channel 19:1 and decide existence single-handed. So the cycle is worth g·tanh(verdict): at most
        // one chair observation, the same unit as the absence it argues against — enough to HOLD a chair
        // YOLO has gone blind on, not enough to keep a phantom alive against a working detector.
        // ★SYMMETRIC, weighted only by remaining line of sight: an occluder in front of the outline makes
        // both halves read the occluder, not the chair, so that look is worth (1 − occlusion) either way.
        float p_vis_ct = 0.0f;
        if (ct_rgb_fresh or ct_depth_fresh)
        {
            float verdict = 0.0f;
            int   n_used  = 0;
            inst.dbg_ct_edge_n = inst.dbg_ct_depth_n = 0;   // 0 = NOT MEASURED this frame, never "refuted"
            if (ct_rgb_fresh and ct_prep.valid())
            {
                const auto cs = fitter.compute_contour_set(inst, ct_rgb_stamp, ct_rgb->cols, ct_rgb->rows);
                if (cs.face.valid())
                {
                    std::vector<std::vector<cv::Point>> ctl;
                    ctl.reserve(cs.controls.size());
                    for (const auto& cc : cs.controls)
                        ctl.push_back(cc.px);
                    const auto es = rc::edges::contour_edge_support(ct_prep, cs.face.px, ctl);
                    // This half's own precondition: with no surviving control `excess` has no scale.
                    if (es.n_samples > 0 and es.n_controls > 0)
                    {
                        verdict += es.excess;
                        n_used   = std::max(n_used, es.n_samples);
                        inst.dbg_ct_edge_excess = es.excess;
                        inst.dbg_ct_edge_n      = es.n_samples;
                    }
                }
            }
            if (ct_depth_fresh)
            {
                // Projected straight into the DEPTH image's own grid (compute_contour_set rescales from the
                // intrinsics), so the scorer's RGB→depth scale factors stay 1.
                const auto cs = fitter.compute_contour_set(inst, ct_depth_stamp, ct_depth->cols, ct_depth->rows);
                if (cs.face.valid())
                {
                    rc::edges::ContourDepthParams dp;
                    // σ_d from the belief's OWN uncertainty, not a constant: position, plus yaw swinging the
                    // outline's lateral edges by up to half a seat width. A poorly known chair must not be
                    // judged against its predicted depth as if it were pinned.
                    const auto& P = inst.ai2_belief.covariance();
                    const float lever = 0.5f * inst.ai2_belief.seat_w();
                    dp.sigma_depth_m = std::sqrt(std::max(1e-4f, P(0, 0) + P(1, 1) + lever * lever * P(2, 2)));
                    const auto ds = rc::edges::contour_depth_support(*ct_depth, cs.face, cs.controls, dp);
                    if (ds.n_samples > 0)
                    {
                        verdict += ds.verdict;
                        n_used   = std::max(n_used, ds.n_samples);
                        inst.dbg_ct_depth_verdict = ds.verdict;
                        inst.dbg_ct_depth_bias_m  = ds.mean_bias_m;
                        inst.dbg_ct_depth_n       = ds.n_samples;
                    }
                }
            }
            inst.dbg_ct_dL = 0.0f;
            if (n_used > 0 and std::isfinite(verdict))     // n_used == 0 ⇒ nothing measured ⇒ HOLD
            {
                const float occ_ct = cfg_.exist_occlusion_check ? fitter.los_occlusion(inst) : 0.0f;
                p_vis_ct = std::clamp(1.0f - occ_ct, 0.0f, 1.0f);
                const float ratio_ct = g * std::tanh(verdict);
                inst.existence.integrate(p_vis_ct, ratio_ct);
                inst.exist_logodds = inst.existence.logodds();
                inst.dbg_ct_dL = p_vis_ct * ratio_ct;
            }
        }

        // p_vis is EXACTLY what one look was worth (staleness confidence x ZED detectability x remaining
        // line of sight), which is the debounce's unit — so it is passed straight through. A cycle that
        // could not have resolved the chair advances nothing and correctly HOLDS.
        // A cycle in which the contour channel also looked was worth the better of the two looks.
        const auto verdict = rc::exist::decide_removal(inst.existence, inst.existence_debounce, policy,
                                                       std::max(p_vis, p_vis_ct));
        if (verdict.remove)
            to_remove.push_back(id);
        if (verdict.stalled and verdict.starved == policy.stall_warn_cycles)   // once per episode
            std::print("chair_concept: {}\n", rc::exist::stall_note(inst.node_name, inst.existence,
                                                                    inst.existence_debounce, verdict));

        // ★AND THE MIRROR CASE, structurally unreportable until 2026-08-15: NOT condemned, and nothing
        // has resolved it for a long time. `stalled` is computed AFTER decide_removal's not-condemned
        // early return, so a belief pinned at the +4 clamp by evidence no look can contradict said
        // NOTHING — a phantom refrigerator held there by a DOOR's returns sat silent for an hour with
        // streak 0 on all 9714 rows. A REPORT only; nothing branches on it.
        if (verdict.unfalsifiable and verdict.starved == policy.stall_warn_cycles)   // once per episode
            std::print("chair_concept: {}\n",
                       rc::exist::unfalsifiable_note(inst.node_name, inst.existence, verdict));
    }

    // Throttled existence readout so a "why is this phantom still here?" case is diagnosable from the log.
    static int ex_dbg = 0;
    if (cfg_.chair_log_period_frames > 0 and ++ex_dbg % (2 * cfg_.chair_log_period_frames) == 0)
        for (const auto& [id, inst] : fitter.instances())
            if (not inst.is_bearing_hypothesis)
            {
                const auto& ms = inst.model.state();
                const bool has_poly = fitter.has_room_polygon();
                const bool inroom = (not has_poly)
                                    or fitter.point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m);
                const float occluded = (inst.roi_valid and inst.assigned_mask_idx < 0) ? fitter.los_occlusion(inst) : 0.0f;
                const float zed_pd = fitter.zed_detectability(inst);   // ZED expected-detectability that gates vacate
                std::print("chair_concept: [existence] {} L={:.2f} pos=({:.2f},{:.2f}) inroom={} roomprior={} roi={} won={} since_det={} occluded={} zed_pd={:.2f}\n",
                           inst.node_name, inst.exist_logodds, ms.cx, ms.cy, inroom ? 1 : 0,
                           has_poly ? 1 : 0, inst.roi_valid ? 1 : 0,
                           inst.assigned_mask_idx >= 0 ? 1 : 0, inst.frames_since_detection, occluded, zed_pd);
                // Same diagnostic to a CSV (you read those) — roomprior=0 ⇒ polygon NOT loaded; inroom=0 ⇒ outside walls;
                // zed_pd=0 ⇒ ZED can't reliably see it (far/edge) so absence won't remove it (ricoh-only-visible).
                static std::ofstream ex_csv = []{ std::ofstream f; rc::diag::open_rotating(f, "etc/chair_existence_log.csv");
                    f << "cycle,node,L,cx,cy,inroom,roomprior_loaded,roi,won,since_det,occluded,zed_pd,"
                         "ct_en,ct_ex,ct_dn,ct_dv,ct_bias,ct_dL\n"; return f; }();
                if (ex_csv)
                {
                    ex_csv << ex_dbg << ',' << inst.node_name << ',' << inst.exist_logodds << ',' << ms.cx << ',' << ms.cy
                           << ',' << (inroom ? 1 : 0) << ',' << (has_poly ? 1 : 0) << ',' << (inst.roi_valid ? 1 : 0)
                           << ',' << (inst.assigned_mask_idx >= 0 ? 1 : 0) << ',' << inst.frames_since_detection
                           << ',' << occluded << ',' << zed_pd
                           // contour channel: ct_en/ct_dn = 0 means NOT MEASURED, never refuted
                           << ',' << inst.dbg_ct_edge_n << ',' << inst.dbg_ct_edge_excess
                           << ',' << inst.dbg_ct_depth_n << ',' << inst.dbg_ct_depth_verdict
                           << ',' << inst.dbg_ct_depth_bias_m << ',' << inst.dbg_ct_dL << '\n';
                    ex_csv.flush();
                }
            }

    // Teardown is the worker's (on_remove): it owns the affordance → instance → node order.
    for (const std::uint64_t id : to_remove)
    {
        const auto it = fitter.instances().find(id);
        if (it == fitter.instances().end())
            continue;   // ids were taken from this same map this call; nothing in between erases one
        std::print("chair_concept: [existence] REMOVE id={} (log-odds {:.2f} < {:.2f}; unexplained/vacated in view)\n",
                   id, it->second.exist_logodds, cfg_.exist_remove_logodds);
        on_remove(id, it->second);
    }
}

}  // namespace rc
