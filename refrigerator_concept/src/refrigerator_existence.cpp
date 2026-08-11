/*
 * refrigerator_existence.cpp — see refrigerator_existence.h. Evidence-based refrigerator removal (existence log-odds + debounce).
 */

#include "refrigerator_existence.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <print>
#include <utility>
#include <vector>

#include "refrigerator_fitter.h"                                       // rc::RefrigeratorFitter (instances, silhouette)
#include "refrigerator_lidar_ingestor.h"                              // rc::RefrigeratorLidarIngestor (per-plane sweeps)
#include "refrigerator_model.h"                                       // RefrigeratorModel::TOP_THICKNESS / LEG_RADIUS
#include "../../common/existence_belief/existence_belief.h"    // rc::exist:: carve_box / mask_evidence / …
#include "../../common/detectability/detectability.h"          // rc::detect — the detector INVERSE model
#include "../../common/dashboard/evidence_monitor.h"           // rc::EvidenceGlobals

namespace rc {

// ─── Existence-based removal ─────────────────────────────────────────────────────────────────────

// Carve the LiDAR sweep(s) + silhouette against every refrigerator footprint, integrate the per-instance existence
// log-odds, and remove the refrigerators whose volume is demonstrably empty (debounced). See refrigerator_existence.h.
void RefrigeratorExistence::update_and_remove(RefrigeratorFitter& fitter, RefrigeratorLidarIngestor* lidar,
                                       bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                                       const std::function<void(std::uint64_t, const RefrigeratorInstance&)>& on_remove)
{
    // Decoupled cadence: the SILHOUETTE/mask channel integrates on a fresh MASK frame (camera clock), the LiDAR
    // free-space carve on a fresh SWEEP (LiDAR clock). Each accrues its evidence independently, so a camera-only
    // cycle still votes absence and a LiDAR-only cycle still carves free space. The removal debounce advances
    // only on an EVIDENCE cycle (when a channel actually integrated), so mixed sensor rates don't bias it.
    const std::vector<Eigen::Vector3f>* sweep = nullptr;
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    if (fresh_sweep and lidar)
    {
        const auto& s = lidar->sweep_room();
        if (not s.empty()) { sweep = &s; origin = lidar->origin_room(); }
    }
    // Low bpearl sweep (own origin) — a second occupancy-only ray-set for the box carve (it strikes the fridge
    // body low, where the high helios sweeps the upper part). Only staged while its feature is on.
    const std::vector<Eigen::Vector3f>* sweep_bp = nullptr;
    Eigen::Vector3f origin_bp = Eigen::Vector3f::Zero();
    if (fresh_sweep and lidar and cfg_.lidar_bpearl_precision > 0.0f)
    {
        const auto& sb = lidar->sweep_bpearl_room();
        if (not sb.empty()) { sweep_bp = &sb; origin_bp = lidar->origin_bpearl_room(); }
    }
    if (not fresh_masks and sweep == nullptr and sweep_bp == nullptr)   // no new evidence on any channel this cycle
        return;

    rc::exist::SensorModel sm;
    sm.sensor_sigma_m = cfg_.existence_sensor_sigma_m;
    sm.detection_prob = cfg_.existence_detection_prob;
    sm.clutter_prob   = cfg_.existence_clutter_prob;

    // Absence-confidence vs range: a miss at long range is weak evidence of removal (the sensor likely could
    // not resolve the object). c = (range_ref/range)^power capped at 1 → 1 up close, →0 far. Scales the FREE
    // (absence) half only; occupancy stays fully informative. power 0 disables (constant 1).
    const auto absence_range_conf = [&](float range_m) -> float
    {
        const float rr = cfg_.existence_absence_range_ref_m;
        if (rr <= 0.0f or cfg_.existence_absence_range_power <= 0.0f) return 1.0f;
        return std::min(1.0f, std::pow(rr / std::max(range_m, 1e-3f), cfg_.existence_absence_range_power));
    };
    // ONE removal policy, shared (rc::exist::RemovalPolicy) so the debounce cannot drift again — it had
    // drifted three ways by 2026-08-10, and door paid for it with twelve deaths at fixated = 0.
    rc::exist::RemovalPolicy policy;
    policy.logodds_max       = cfg_.existence_logodds_max;
    policy.removal_prob      = cfg_.existence_removal_prob;
    policy.frame_correlation = cfg_.existence_frame_correlation;
    policy.remove_frames     = static_cast<float>(cfg_.existence_remove_frames);


    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter.instances())
    {
        // How resolving THIS cycle's look was, in units of one ideal observation (0 => the sensor could not
        // have seen it from here even if it were present). Drives the removal debounce below.
        float cycle_p_detect = 0.0f;
        // Hoisted with it: the debounce advances only on an evidence cycle, but the DECISION is read
        // outside that block, so the two must not share a scope.
        bool  doomed_now = false;
        if (not inst.ai2_initialized) continue;
        rc::exist::arm(inst.existence, policy);
        const auto& bs = inst.ai2_belief.state();
        const auto& S  = inst.ai2_belief.covariance();
        const float surf_sigma = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));   // footprint position σ
        const bool observed = inst.frames_since_detection == 0;                      // fresh mask this cycle
        bool integrated = false;

        // SILHOUETTE / MASK channel (pixel-level) — CAMERA clock: project the refrigeratortop silhouette and compare,
        // per predicted-VISIBLE pixel, against the current YOLO foreground. Lit by a "refrigerator" mask ⇒ occupancy
        // (holds L up); lit by NOTHING ⇒ predicted-but-absent (the "gone" signal, present EVEN WITH NO YOLO MASK);
        // lit by a NON-refrigerator mask ⇒ occluded ⇒ HOLD (never false absence behind a nearer object). n_detectable==0
        // (out of FoV / fully occluded) ⇒ mask_evidence HOLDs.
        if (fresh_masks)
        {
            const auto sil = fitter.compute_silhouette_existence(inst);
            inst.dbg_ex_sil_occ = sil.e_occ; inst.dbg_ex_sil_free = sil.e_free; inst.dbg_ex_sil_ndet = sil.n_detectable;
            if (sil.n_detectable > 0)
            {
                // Observed-guard (mirrors the LiDAR hollow guard): if the refrigerator was DETECTED by any sensor this
                // frame it is not gone, so suppress silhouette ABSENCE — a ZED false-negative, or a refrigerator seen
                // only by the ricoh whose silhouette the ZED-based check can't confirm, must never vote it away.
                // Occupancy always counts. (The ricoh-projected silhouette, once the producer ships 360 mask
                // pixels, will let absence be judged in the ricoh view directly instead of relying on this guard.)
                // ★A frame that may not MOVE the geometry may not DESTROY the object either.
                // `dbg_gated` is the fit's own admissibility verdict for this frame (truncated mask, or the
                // robot moving with the mask off-centre). We already apply that rule to CREATE — birth accrues
                // only on admissible frames — and removal is the same decision with the opposite sign, so it
                // needs the same evidence standard. It did not have it, and that asymmetry is what deleted a
                // healthy fridge in 5 seconds (etc/ai2_log.csv 19:55): the robot drove 2.28 m → 0.82 m, the
                // motion gate froze the belief at cycle 16, and from cycle 30 the frozen box no longer projected
                // onto the mask (socc 600 → 0 with npts still ~55000, i.e. the fridge was plainly visible). Every
                // cycle that contributed removal evidence had gated=1. At 0.8 m a small pose-chain error is a
                // large pixel error, so a frozen belief cannot be checked against the image at all.
                //
                // ⚠Consequence to watch: a phantom is now only removable from an admissible viewpoint — still,
                // and reasonably framed. That is what the epistemic planner exists to produce, but it does mean
                // removal waits for a good look instead of happening while driving past.
                const float raw_free = (observed or inst.dbg_gated) ? 0.0f : sil.e_free;
                // P(detect | present, geometry): how confidently the ZED would resolve this refrigerator FROM HERE.
                // in_fov_frac folds the real FRUSTUM + occlusion; range_conf the angular-size drop; central_frac
                // whether the robot is actually LOOKING at it (a peripheral refrigerator clipping the wide FoV edge is
                // NOT a verifying view). A predicted-visible-but-absent observation is only evidence of REMOVAL in
                // proportion to P(detect); the REST is epistemic surprise — "I can't resolve this from here" — which
                // must make the robot GO VERIFY, never delete a refrigerator it never properly looked at.
                // A verifying view = CLOSE (range_conf) + in frustum & unoccluded (in_fov_frac) + the robot is
                // LOOKING at it (central_frac). NOT obliquity: this robot is chronically edge-on and an edge-on
                // refrigerator IS detectable up close (refrigerator_1), so obliquity would wrongly block legitimate removal.
                // ★P(detect) from the shared detector INVERSE model, not a far-range term alone. The old form
                // was monotonic in range, so it had no way to say that standing TOO CLOSE also destroys
                // detectability: at 0.82 m the fridge spans fill≈0.80 of the frame and the envelope gives
                // P(detect)=0.035 — a missing mask there is EXPECTED, not evidence the fridge is gone. That
                // asymmetry is what let a nose-to-nose approach read as absence. `sil.fill` is measured over all
                // forward-projecting samples including those off-frame, so an overflowing object reads fill>1.
                // Same model the epistemic planner uses to pick the viewpoint — one envelope, both directions.
                const float p_detect = rc::detect::p_detect(sil.fill, sil.in_fov_frac(), det_env_)
                                     * sil.central_frac();
                cycle_p_detect = p_detect;
                const float verify = raw_free * (1.0f - p_detect);        // un-confident absence → go-verify surprise

                // ★p_detect scales the SATURATED delta, not the raw pixel COUNT — ported from door_concept,
                // which documented this before I wrote the count-scaling form here. mask_evidence tanh-saturates
                // the summed log-odds, and with hundreds of silhouette samples the sum sits far past the knee, so
                // multiplying the COUNT by p_detect changes the delta almost not at all until p_detect hits
                // exactly 0. That is why the two long-range deletions happened DESPITE a low p_detect: the 7 m
                // through-a-wall removal ran at p_detect=0.126 and the 6 m one lower still, and both deleted at
                // full strength. The range term was not too weak — it was very nearly INERT over its whole range,
                // biting only at zero. Interpolating between the confirm-only delta and the full delta makes
                // p_detect act linearly end to end, and leaves confirmation (+) untouched.
                const float d_full = rc::exist::mask_evidence(sil.e_occ, raw_free, sil.n_detectable, sm).log_odds_delta;
                rc::exist::Evidence ev_sil;
                ev_sil.e_occ = sil.e_occ; ev_sil.e_free = raw_free; ev_sil.n_reached = sil.n_detectable;
                // ★INTERPOLATE FROM ZERO, NOT FROM THE OCCUPANCY-ONLY FLOOR. The old form was
                //     delta = d_conf + p_detect * (d_full - d_conf)
                // where d_conf was mask_evidence(e_occ, free=0, …) — the CONFIRM-ONLY delta. Its comment
                // claimed "p_detect -> 0 => pure HOLD"; it does not, because it leaves +d_conf standing, and
                // an occupancy-only likelihood is always positive. (d_conf is no longer computed at all — it
                // survived the fix as an unused variable, which is how the compiler kept pointing at the
                // deleted ratchet long after it stopped firing.) That is precisely the
                // RATCHET this header warns about — "an occupancy-only channel is not a likelihood ratio,
                // it can only push L to +L_max, where it sticks forever" — and it is what it did again:
                // a phantom refrigerator standing INSIDE the dining set collected occupancy from the real
                // table and chairs in its volume, sat at L = +4.00 with lfree = 483 against locc = 206, and
                // could not be removed while the robot stared straight at it.
                //
                // A probe that could not have resolved the object is not evidence EITHER WAY, so the honest
                // update is 0. Scaling the full likelihood ratio by p_detect gives exactly that, keeps the
                // update SYMMETRIC as this file requires, and still reproduces the old behaviour at
                // p_detect -> 1 where the two forms coincide.
                ev_sil.log_odds_delta = p_detect * d_full;
                inst.dbg_ex_sil_free_eff = raw_free * p_detect;   // diagnostic: absence mass actually admitted
                inst.dbg_ex_pdetect = p_detect; inst.dbg_ex_central = sil.central_frac();
                // p_detect passed EXPLICITLY: log_odds_delta already carries it, but the second argument is
                // also what the frame-correlation weighting reads (rho_eff = rho*(1-p_detect)). Defaulting it
                // to 1.0 tells the belief every miss was a maximally unambiguous look, so correlated misses
                // count as independent. Inert while ExistenceFrameCorrelation is 0; wrong once it is measured.
                inst.existence.integrate(ev_sil, p_detect);
                integrated = true;
                // Route the un-resolvable absence into an epistemic VERIFY pull (decayed accumulator). When it
                // builds up, the refrigerator is flagged for verification (the epistemic planner drives the robot to a
                // good ZED viewpoint). Removal then only ever fires from a HIGH-p_detect view — verification-gated.
                const float vss = cfg_.verify_surprise_smooth;
                inst.verify_surprise = (1.0f - vss) * inst.verify_surprise + vss * verify;
                inst.wants_verification = inst.verify_surprise > cfg_.existence_verify_surprise;
            }
        }

        // LiDAR occupancy carve of the WHOLE SOLID BOX z∈[0, H] — LiDAR clock. A refrigerator is one solid
        // floor-anchored cuboid (refrigerator_model.cpp), so every ring that reaches it strikes real surface at
        // whatever height it happens to sweep: a return inside the volume is occupancy, a beam through it is a
        // pass-through. The table-concept original carved only the [H−3cm, H] top slab plus four 2.5 cm corner
        // "leg" columns — a 3 cm sliver at 1.9 m that the horizontal rings almost never intersect, so the fridge's
        // massively-hit solid body contributed NO confirmation and ex_locc swung erratically between 0.03 and 70
        // depending on whether a ring happened to graze a fake leg. See [[refrigerator-table-geometry-churn]].
        // Both LiDARs are carved: the low bpearl strikes the fridge body low, the high helios mid/upper.
        if (sweep)
        {
            rc::exist::Evidence ev = rc::exist::carve_box(origin, *sweep, bs.cx, bs.cy, bs.yaw, bs.w, bs.h,
                                                          0.0f, bs.H, surf_sigma, sm);
            if (sweep_bp != nullptr)   // second ray-set, own origin (occlusion-aware first-hit) — occupancy only
            {
                const rc::exist::Evidence bp = rc::exist::carve_box(origin_bp, *sweep_bp, bs.cx, bs.cy, bs.yaw,
                                                                    bs.w, bs.h, 0.0f, bs.H, surf_sigma, sm);
                ev.e_occ += bp.e_occ; ev.n_reached += bp.n_reached;
            }
            inst.dbg_ex_lidar_occ = ev.e_occ; inst.dbg_ex_lidar_free = ev.e_free; inst.dbg_ex_lidar_n = ev.n_reached;
            if (ev.n_reached > 0)                                                    // 0 ⇒ not probed ⇒ HOLD
            {
                // ── P(detect) for the LiDAR, exactly as the silhouette has one ────────────────────────
                // Absence is evidence of removal only in proportion to how well this sensor could have
                // resolved the object FROM HERE. The camera has had that discipline all along (range_conf x
                // in_fov_frac x central_frac); the LiDAR did not, and once solid_delta made this channel
                // SYMMETRIC that omission became a deletion mechanism: LiDAR is 360°, so it votes every cycle
                // regardless of where the robot is looking. A fridge was removed with the robot 6 m away and
                // FACING AWAY — the camera correctly HELD (n_detectable==0, nothing integrated) while the
                // LiDAR walked L down on its own. See [[existence-policy-unification]].
                //
                //   coverage = n_reached / LidarCoverageN0 — how many beams actually reached the footprint
                //              against the count that constitutes a resolving look. That constant already
                //              exists and already means exactly this (the fit uses it the same way), so this
                //              introduces no new number. At 6 m the rings are sparse, a handful of beams
                //              graze the box, coverage ~0 ⇒ absence says nothing ⇒ HOLD. Up close a phantom
                //              is swept by many beams, coverage 1 ⇒ absence at full strength ⇒ it still dies.
                //
                // ★Applied to e_interior AS WELL as e_free. Only e_free was degraded before, so the interior
                // term — the one that makes a phantom-over-a-wall die — went in at FULL strength from any
                // distance. That was a plain bug, not a policy choice.
                const float lidar_range = (origin - Eigen::Vector3f(bs.cx, bs.cy, 0.5f * bs.H)).norm();
                const float coverage = (cfg_.lidar_coverage_n0 > 0.0f)
                                     ? std::min(1.0f, static_cast<float>(ev.n_reached) / cfg_.lidar_coverage_n0)
                                     : 1.0f;
                // ★Same admissibility rule as the silhouette: a frame that may not MOVE the geometry may not
                // DESTROY the object. I applied this to the camera channel first and the fridge died in exactly
                // the same 5 seconds, because the LiDAR half had no such gate — etc/ai2_log.csv 20:08 shows
                // ex_sfree_eff = 0 on every row (camera correctly silent) while ex_lfree_eff ran 714 → 971 with
                // ex_locc ≈ 0.017: ~900 beams passing straight THROUGH the believed box each cycle.
                //
                // Both sensors were saying the same true thing — the believed box is not where the fridge is —
                // and neither was entitled to act on it. The belief froze at cycle 16 (motion gate) while the
                // robot drove 2.3 m → 1.15 m; a stale box at 1 m is metres of ray error and a whole frame of
                // pixel error. The evidence is about OUR registration, not about the fridge's existence.
                // Occupancy is unaffected: a return that lands on the box can only ever confirm.
                const float admissible = inst.dbg_gated ? 0.0f : 1.0f;
                const float p_detect_lidar = absence_range_conf(lidar_range) * coverage * admissible;
                // A refrigerator IS a faithful opaque solid, so it declares itself as one and gets the symmetric
                // policy: a return at the near face is FOR; a through-beam and — the fix — a return metres deep
                // INSIDE the volume are AGAINST, because a solid fridge would have blocked them.
                //
                // This replaces occupancy-only, which was a RATCHET: it could only push L to the +4 clamp, and
                // being integrated after the silhouette it re-pinned L there every cycle, before should_remove
                // ran. Live: phantom refrigerator_2 accrued confident camera absence on 2885 of 3255 cycles
                // (~5 min, mean sfree_eff 85.5) and its ex_L NEVER went below its initial 0.00, streak never
                // left 0 — immortal. The wall behind it supplied ~54 interior returns/cycle that the old code
                // scored as proof it existed. See [[existence-policy-unification]].
                // Same interpolation as the silhouette above: scale the SATURATED delta, not the raw beam
                // counts. solid_delta tanh-saturates too, and with ~900 beams the sum is far past the knee, so
                // multiplying e_free/e_interior by p_detect barely moved the result — exactly the inertness
                // door_concept documented for the mask channel.
                // (The occupancy-only counterpart — solid_delta on an Evidence with e_free/e_interior
                //  zeroed — is deliberately NOT computed: interpolating from it is the ratchet above.)
                const float l_full = rc::exist::solid_delta(ev, sm);        // occupancy + absence
                inst.dbg_ex_lidar_free_eff = (ev.e_free + ev.e_interior) * p_detect_lidar;
                // Same correction as the silhouette channel above: interpolate from ZERO.
                ev.log_odds_delta = p_detect_lidar * l_full;
                inst.existence.integrate(ev, p_detect_lidar);   // see the note at the silhouette integrate
                integrated = true;
            }
        }

        // ★DEBOUNCE ON LOOKS, NOT CYCLES. `++` also counted the cycles where p_detect was 0 and the
        // channel had just, correctly, HELD — so an instance that dipped below the boundary once was
        // deleted RemoveFrames cycles later regardless of what happened in between. Measured in
        // door_concept, where 12 of 12 deaths had fixated=0: the `if (integrated)` guard does NOT
        // prevent this, because a channel that ran and resolved nothing still sets integrated.
        // Accumulating p_detect makes RemoveFrames a number of IDEAL observations (table/bottle unit).
        //
        // ★AND IT IS DECIDED EVERY CYCLE, NOT ONLY ON AN EVIDENCE CYCLE. The `if (integrated)` guard used
        // to wrap this call, which is what made a stalled debounce SILENT: on a cycle where no channel ran the
        // streak neither advanced nor reset, and the same guard suppressed the per-cycle log row, so the frozen
        // state left no trace at all. Measured on table_concept (2026-08-11): a condemned phantom sat at
        // p(exists)=0.055 with streak 1.53/15 and stopped logging 7 300 cycles before the run ended. The
        // arithmetic here is unchanged — a non-resolving cycle adds 0, and a belief above the boundary is
        // already reset — what changes is that the stall becomes observable, and that `required` now falls
        // with confidence (see rc::exist::required_observations).
        const auto verdict = rc::exist::decide_removal(
                inst.existence, inst.existence_debounce, policy, cycle_p_detect);
        doomed_now = verdict.remove;
        if (verdict.stalled)
            std::print("refrigerator_concept: {}\n", rc::exist::stall_note(inst.node_name, inst.existence,
                                                                     inst.existence_debounce, verdict));

        // The per-cycle READOUT still belongs to an evidence cycle: a cycle in which nothing was measured has
        // nothing new to report about the evidence, only about the decision (printed above when it stalls).
        if (integrated)
        {
            if (fitter.should_log(inst))
                std::print("[{}] [existence] L={:.2f} p={:.2f} | lidar occ={:.1f} free={:.1f} n={} | sil occ={:.0f} free={:.0f} ndet={} | {} streak={:.1f}/{:.1f}\n",
                           inst.node_name, inst.existence.logodds(), inst.existence.p_exists(),
                           inst.dbg_ex_lidar_occ, inst.dbg_ex_lidar_free, inst.dbg_ex_lidar_n,
                           inst.dbg_ex_sil_occ, inst.dbg_ex_sil_free, inst.dbg_ex_sil_ndet,
                           observed ? "obs" : "-", inst.existence_debounce.streak, verdict.required);

            // Persist the existence trajectory across the OUT-OF-FoV stretch that a fit row can't reach: an
            // out-of-view instance is not fit (no fresh mask) so no ai2_log row is written for it — yet this is
            // exactly where a spurious LiDAR-absence carve accrues and removes a real refrigerator. Emit a row here for
            // non-observed instances only (observed ones already got a fit row this cycle → no double-log).
            if (not observed)
                fitter.log_existence_cycle(inst);
        }
        if (doomed_now)
            doomed.push_back(id);
    }
    ev_g.removals     += static_cast<int>(doomed.size());   // EvidenceMonitor counters
    ev_g.removals_cum += static_cast<long>(doomed.size());
    for (const std::uint64_t id : doomed)
    {
        std::print("refrigerator_concept: [existence] removing refrigerator id={} — free-space evidence (empty volume)\n", id);
        if (auto it = fitter.instances().find(id); it != fitter.instances().end())
        {
            // Record BEFORE teardown, while the existence state that justified the kill is readable.
            if (on_remove) on_remove(id, it->second);
            it->second.affordance.remove();
        }
        fitter.forget_node(id);
        G_->delete_node(id);
    }
}

}  // namespace rc
