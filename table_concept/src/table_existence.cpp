/*
 * table_existence.cpp — see table_existence.h. Evidence-based table removal (existence log-odds + debounce).
 */

#include "table_existence.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <print>
#include <utility>
#include <vector>

#include "table_fitter.h"                                       // rc::TableFitter (instances, silhouette)
#include "table_lidar_ingestor.h"                              // rc::TableLidarIngestor (per-plane sweeps)
#include "table_model.h"                                       // TableModel::TOP_THICKNESS / LEG_RADIUS
#include "../../common/existence_belief/existence_belief.h"    // rc::exist:: carve_box / mask_evidence / …
#include "../../common/dashboard/evidence_monitor.h"           // rc::EvidenceGlobals

namespace rc {

// ─── Existence-based removal ─────────────────────────────────────────────────────────────────────

// Carve the LiDAR sweep(s) + silhouette against every table footprint, integrate the per-instance existence
// log-odds, and remove the tables whose volume is demonstrably empty (debounced). See table_existence.h.
void TableExistence::update_and_remove(TableFitter& fitter, TableLidarIngestor* lidar,
                                       bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                                       const std::function<void(std::uint64_t, const TableInstance&)>& on_remove)
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
    // Low bpearl sweep (own origin) — an occupancy-only source for the LEG carve (the low LiDAR is what actually
    // strikes the legs the high helios grazes over). Only staged while its feature is on.
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

    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter.instances())
    {
        if (not inst.ai2_initialized) continue;
        inst.existence.set_max(cfg_.existence_logodds_max);
        const auto& bs = inst.ai2_belief.state();
        const auto& S  = inst.ai2_belief.covariance();
        const float surf_sigma = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));   // footprint position σ
        const bool observed = inst.frames_since_detection == 0;                      // fresh mask this cycle
        bool integrated = false;
        // How resolving THIS cycle's look was, in units of one ideal observation (0 ⇒ the camera could not have
        // seen the table from here even if it were present). Drives the removal debounce below.
        float cycle_p_detect = 0.0f;

        // SILHOUETTE / MASK channel (pixel-level) — CAMERA clock: project the tabletop silhouette and compare,
        // per predicted-VISIBLE pixel, against the current YOLO foreground. Lit by a "table" mask ⇒ occupancy
        // (holds L up); lit by NOTHING ⇒ predicted-but-absent (the "gone" signal, present EVEN WITH NO YOLO MASK);
        // lit by a NON-table mask ⇒ occluded ⇒ HOLD (never false absence behind a nearer object). n_detectable==0
        // (out of FoV / fully occluded) ⇒ mask_evidence HOLDs.
        if (fresh_masks)
        {
            const auto sil = fitter.compute_silhouette_existence(inst);
            inst.dbg_ex_sil_occ = sil.e_occ; inst.dbg_ex_sil_free = sil.e_free; inst.dbg_ex_sil_ndet = sil.n_detectable;
            inst.dbg_ex_sil_ntotal = sil.n_total;   // → the real in-FoV fraction for the phantom log
            if (sil.n_detectable > 0)
            {
                // Observed-guard (mirrors the LiDAR hollow guard): if the table was DETECTED by any sensor this
                // frame it is not gone, so suppress silhouette ABSENCE — a ZED false-negative, or a table seen
                // only by the ricoh whose silhouette the ZED-based check can't confirm, must never vote it away.
                // Occupancy always counts. (The ricoh-projected silhouette, once the producer ships 360 mask
                // pixels, will let absence be judged in the ricoh view directly instead of relying on this guard.)
                // ★A frame that may not MOVE the geometry may not DESTROY the object either. `dbg_gated` is the
                // fit's own admissibility verdict for this frame (truncated mask, or the robot moving with the
                // mask off-centre): when it is set the belief was frozen, so the projected silhouette is a
                // STALE box compared against a LIVE image, and any mismatch measures our registration rather
                // than the table's existence.
                //
                // Ported from refrigerator_concept, where the absence of this rule deleted a healthy fridge in
                // 5 seconds: the robot drove 2.3 m → 1.15 m, the motion gate froze the belief, and the frozen
                // box stopped projecting onto a mask that was still arriving with ~55000 points (socc 570 → 0).
                // Every cycle that contributed removal evidence had gated=1. At ~1 m a small pose-chain error
                // is a whole frame of pixel error, so a frozen belief cannot be checked against the image at
                // all. Occupancy is untouched — a mask pixel that lights the silhouette can only ever confirm.
                //
                // ⚠Consequence: a phantom is now only removable from an admissible viewpoint — still, and
                // reasonably framed. That is what the epistemic planner exists to produce, but removal now
                // waits for a good look instead of happening while driving past.
                //
                // ★2026-08-06 — THE GUARD MUST READ THE GATE'S *REASON*, NOT THE GATE. `dbg_gated` is
                // `trunc_gated or not fixated`, and `fixated` requires npts ≥ FixationMinPts. A PHANTOM HAS NO
                // MASK BY DEFINITION, so npts = 0 ⇒ never fixated ⇒ permanently gated ⇒ its absence evidence was
                // permanently zeroed: the channel demanded a mask on the object in order to believe there is no
                // mask on the object. Live proof (table_2, a 2-frame YOLO blip at (-3.66,-0.07)): gated=1 on
                // 3239/3239 cycles, and on the one confident look — sndet 640, central 1.000, p_detect 0.545,
                // e_free 640, robot parked and staring — sfree_eff was still 0.00, with L pinned at the +4 clamp.
                // Only the VIEWPOINT reasons make a silhouette comparison measure registration instead of
                // existence; point-starvation is the observation itself. Second defect fixed here: the fit
                // returns early when no mask arrives, so on exactly the cycles this guard fires the flags were
                // STALE (from whenever the table was last seen) — dbg_gate_fresh says whether they mean anything.
                const bool view_untrustworthy = inst.dbg_gate_fresh
                                            and (inst.dbg_trunc_gated or not inst.dbg_fix_still
                                                                      or not inst.dbg_fix_centred);
                const float raw_free = (observed or view_untrustworthy) ? 0.0f : sil.e_free;
                // P(detect | present, geometry): how confidently the ZED would resolve this table FROM HERE.
                // in_fov_frac folds the real FRUSTUM + occlusion; range_conf the angular-size drop; central_frac
                // whether the robot is actually LOOKING at it (a peripheral table clipping the wide FoV edge is
                // NOT a verifying view). A predicted-visible-but-absent observation is only evidence of REMOVAL in
                // proportion to P(detect); the REST is epistemic surprise — "I can't resolve this from here" — which
                // must make the robot GO VERIFY, never delete a table it never properly looked at.
                // A verifying view = CLOSE (range_conf) + in frustum & unoccluded (in_fov_frac) + the robot is
                // LOOKING at it (central_frac). NOT obliquity: this robot is chronically edge-on and an edge-on
                // table IS detectable up close (table_1), so obliquity would wrongly block legitimate removal.
                // ★2026-08-06 — P(detect) IS UNIMODAL IN FRAMING, NOT MONOTONIC IN RANGE. The old form was
                // `absence_range_conf(range) · in_fov_frac · central_frac`, and absence_range_conf is
                // min(1, (ref/range)^power): ONE-SIDED. It discounts absence only when the table is FAR, and
                // returns FULL confidence up close — exactly where a ~0.9 m table overflows the ZED frame, YOLO
                // loses the context it needs, and no mask comes back. That is not evidence of removal, it is a
                // blind spot, and it deleted the REAL table on 2026-08-06: at range 0.46 m with npts=0 for ~35
                // cycles, p_detect climbed 0.16 → 0.83 as the robot CENTRED on the table, L fell +2.24 → −4.00
                // and the debounce ran to 15.16 (phantom_events DEATH row: range 0.88 m, in_fov 0.92,
                // central 0.63, fixated 0). Driving closer made the agent MORE confident the table was gone.
                //
                // rc::detect::p_detect is the shared model that already gets this right — two logistics on the
                // projected FILL fraction, so it → 0 both when the object is too few pixels AND when it crowds
                // the frame. `initialize()` has always claimed "ONE detector envelope, both directions"; until
                // now only the epistemic planner honoured it while removal kept its own one-sided range curve.
                // Same envelope instance, so the viewpoint the planner drives to is the argmax of the model
                // absence is weighted by. At the death geometry it returns ~0.002 instead of 0.83.
                // roi_fill is refreshed every cycle by compute_projected_roi, including on no-mask cycles.
                const float p_detect = cfg_.existence_absence_envelope
                    ? rc::detect::p_detect(inst.roi_fill, sil.in_fov_frac(), det_env_) * sil.central_frac()
                    : absence_range_conf(sil.mean_range_m) * sil.in_fov_frac() * sil.central_frac();
                cycle_p_detect = p_detect;
                const float verify = raw_free * (1.0f - p_detect);        // un-confident absence → go-verify surprise
                // p_detect must scale the SATURATED per-cycle log-odds, NOT the raw pixel count. Scaling the count
                // is a no-op in practice: e_free is thousands of pixels, so even p_detect=0.02 leaves the product
                // far above the tanh knee and the cycle still lands at the full ±llr_occ ceiling — a 2%-resolvable
                // view removed the table at the same rate as a clean one (live: table_5 p_detect=0.018 → L=−3.13,
                // table_1 at 5.8 m p_detect=0.064 → removed). Instead take the frame's OCCUPANCY-only saturated
                // delta as the floor and admit only p_detect of the additional absence swing, so one cycle of an
                // un-resolvable view is worth p_detect of one confident look. p_detect→1 (close, centred, in
                // frustum) reproduces the previous behaviour exactly; p_detect→0 ⇒ pure HOLD.
                const rc::exist::Evidence e_conf = rc::exist::mask_evidence(sil.e_occ, 0.0f, sil.n_detectable, sm);
                const rc::exist::Evidence e_full = rc::exist::mask_evidence(sil.e_occ, raw_free, sil.n_detectable, sm);
                rc::exist::Evidence e_use = e_full;
                e_use.log_odds_delta = e_conf.log_odds_delta
                                     + p_detect * (e_full.log_odds_delta - e_conf.log_odds_delta);
                inst.dbg_ex_sil_free_eff = raw_free * p_detect;   // diagnostic: absence mass actually admitted
                inst.dbg_ex_pdetect = p_detect; inst.dbg_ex_central = sil.central_frac();
                inst.existence.integrate(e_use);
                integrated = true;
                // Route the un-resolvable absence into an epistemic VERIFY pull (decayed accumulator). When it
                // builds up, the table is flagged for verification (the epistemic planner drives the robot to a
                // good ZED viewpoint). Removal then only ever fires from a HIGH-p_detect view — verification-gated.
                const float vss = cfg_.verify_surprise_smooth;
                inst.verify_surprise = (1.0f - vss) * inst.verify_surprise + vss * verify;
                inst.wants_verification = inst.verify_surprise > cfg_.existence_verify_surprise;
            }
        }

        // LiDAR occupancy carve of the TOP-SLAB z-band [H−t, H] — LiDAR clock (the slab spans the full footprint
        // there, so a beam hits rim/top = occupancy or passes through = gone — no hollow ambiguity between legs).
        if (sweep)
        {
            const float slab_lo = bs.H - rc::TableModel::TOP_THICKNESS;
            rc::exist::Evidence ev = rc::exist::carve_box(origin, *sweep, bs.cx, bs.cy, bs.yaw, bs.w, bs.h,
                                                          slab_lo, bs.H, surf_sigma, sm);
            // NEIGHBOURHOOD SAMPLE for the measured clutter prior: the SAME slab band with the footprint scaled
            // by √2, so the shell (outer minus this box) has exactly the box's own volume and the two return
            // rates are comparable samples of the same structure, gathered by the same beams. See
            // rc::exist::measure_clutter. Cheap: one extra carve of a box twice this one's footprint.
            rc::exist::LocalClutter clutter;
            if (cfg_.existence_local_clutter)
            {
                constexpr float kEqualVolumeDilation = 1.41421356f;   // √2 in x and y ⇒ 2× area ⇒ shell ≡ box
                const rc::exist::Evidence ev_outer = rc::exist::carve_box(
                    origin, *sweep, bs.cx, bs.cy, bs.yaw,
                    kEqualVolumeDilation * bs.w, kEqualVolumeDilation * bs.h, slab_lo, bs.H, surf_sigma, sm);
                clutter = rc::exist::measure_clutter(ev, ev_outer, sm);
            }
            // Leg OCCUPANCY (the 3D model has 4 legs; the rings hit them too). Occupancy-ONLY: legs are thin, so a
            // MISS is not evidence of absence and the hollow space between them must never vote free. Carve each leg's
            // tall volume [0, H−t] and add only its e_occ — robustifies CONFIRMATION when the thin top slab sits at an
            // awkward height for the horizontal rings (tall legs are far likelier struck). Cannot cause a false removal.
            if (cfg_.existence_leg_occupancy)
            {
                const float lr   = rc::TableModel::LEG_RADIUS;
                // 5 cm floor: a degenerate-geometry guard, not a tuned gate. If the belief's H collapses toward
                // the top thickness the leg volume would vanish/invert; floor it to a minimum plausible leg extent
                // so the carve stays valid. Physical minimum, no config key.
                const float legz = std::max(0.05f, bs.H - rc::TableModel::TOP_THICKNESS);
                const float ix   = 0.5f * bs.w - lr, iy = 0.5f * bs.h - lr;   // leg centres inset at the outer corners
                const float cyaw = std::cos(bs.yaw), syaw = std::sin(bs.yaw);
                // Carve the legs from EACH available LiDAR — the LOW bpearl is the one that actually strikes them
                // (the high helios grazes over), so this is where the per-device split pays off. Occupancy-only.
                const std::array<std::pair<const std::vector<Eigen::Vector3f>*, Eigen::Vector3f>, 2> leg_srcs = {{
                    {sweep, origin}, {sweep_bp, origin_bp} }};
                for (const auto& [swp, org] : leg_srcs)
                {
                    if (swp == nullptr) continue;
                    for (const auto& [llx, lly] : {std::pair{ix, iy}, std::pair{-ix, iy}, std::pair{ix, -iy}, std::pair{-ix, -iy}})
                    {
                        const float lx = bs.cx + cyaw * llx - syaw * lly;
                        const float ly = bs.cy + syaw * llx + cyaw * lly;
                        const rc::exist::Evidence le = rc::exist::carve_box(org, *swp, lx, ly, bs.yaw,
                                                                            2.0f * lr, 2.0f * lr, 0.0f, legz, surf_sigma, sm);
                        ev.e_occ += le.e_occ; ev.n_reached += le.n_reached;   // OCCUPANCY ONLY (discard leg e_free)
                    }
                }
            }
            inst.dbg_ex_lidar_occ = ev.e_occ; inst.dbg_ex_lidar_free = ev.e_free; inst.dbg_ex_lidar_n = ev.n_reached;
            if (ev.n_reached > 0)                                                    // 0 ⇒ not probed ⇒ HOLD
            {
                // Degrade the free/absence half by LiDAR range: far away the beams are sparse and the thin
                // top-slab subtends few of them, so a pass-through is weak evidence the tabletop is gone.
                const float lidar_range = (origin - Eigen::Vector3f(bs.cx, bs.cy, bs.H)).norm();
                ev.e_free *= absence_range_conf(lidar_range);
                // OCCUPANCY-ONLY, unconditionally: a solid-slab model vs a thin real tabletop makes LiDAR
                // "free" unreliable (beams passing under the top surface read as gone), so it must never drive
                // removal — only the camera silhouette does. The A/B that enabled LiDAR absence evidence was
                // REFUTED on live data (2026-07-12) and its flag has been removed; occupancy still counts and
                // holds L up. e_free is computed above only for the diagnostic columns.
                inst.dbg_ex_lidar_free_eff = 0.0f;
                // ★2026-08-06 — the occupancy half is now a CONTRAST against the measured neighbourhood, not a
                // count against an assumed-empty room. `hollow_guarded_delta` scored every returning beam at
                // log(pd/0.05), so ANY occupancy was worth a full confident hit and re-pinned L at the +4 clamp
                // every cycle, after the camera's absence and before should_remove — the RATCHET that
                // existence_belief.h names as a bug. Live proof: the phantom's box returned 6.56% of beams and
                // the REAL table's 1.96% (both saturating to the same +2.83), so the channel ranked the phantom
                // 3.3× above the truth. Replaying the whole run, the LiDAR contributed +15073 log-odds against
                // the camera's −710. A contrast is 0 where the object is indistinguishable from what surrounds
                // it, which is the whole of the phantom-on-a-wall case. Set ExistenceLocalClutter=false to A/B
                // back to the historic form.
                ev.log_odds_delta = cfg_.existence_local_clutter
                                  ? rc::exist::contrast_delta(ev, clutter, sm)
                                  : rc::exist::hollow_guarded_delta(ev, true, sm);
                inst.dbg_ex_clutter_q  = clutter.rate;
                inst.dbg_ex_clutter_n  = clutter.n_reached;
                //
                // ★★THE LiDAR IS IDENTITY-BLIND, SO IT DOES NOT VOTE ON TABLE-NESS (default; 2026-08-06).
                //
                // The contrast above fixes the phantom-over-a-WALL case: object and neighbourhood become
                // indistinguishable and the channel falls silent. It does NOT fix the case actually on the floor
                // here — a REAL, COMPACT object that YOLO mistook for a table. There the shell is empty air, the
                // contrast is large and positive, and the channel confirms the phantom forever. It is right to:
                // the beams really do return. They are simply not evidence for the proposition being tracked.
                //
                // The existence log-odds is about "is there a TABLE here", and range data cannot separate a table
                // from any other object of similar extent: P(return | table) ≈ P(return | that bin / box / chair),
                // so the likelihood ratio is ≈1 and the honest ΔL is 0 — for confirmation as much as for removal.
                // The measurement is unambiguous that this is not merely weak but INVERTED: over ~5400 cycles the
                // real table returned 1.96% of the beams reaching it and the phantom 6.56%, and both saturated to
                // the same +2.83. A channel that ranks the phantom 3.3× above the truth must not be allowed to
                // outvote the one sensor that can read a LABEL.
                //
                // What remains is exactly the user's test, and it is the silhouette channel's job: from a
                // viewpoint where we can infer YOLO SHOULD have detected a table (p_detect = range_conf ·
                // in_fov_frac · central_frac), it repeatedly did not — ExistenceRemoveFrames cycles in a row.
                // Nothing else gets a vote. LiDAR keeps its two real jobs untouched: the geometric fit
                // (LidarPrecision) and occlusion reasoning.
                //
                // A/B: ExistenceLidarConfirms=true restores a voting LiDAR — with ExistenceLocalClutter it votes
                // by contrast, without it by the historic ratchet that pinned L at the clamp forever.
                inst.dbg_ex_lidar_llr = cfg_.existence_lidar_confirms ? ev.log_odds_delta : 0.0f;
                if (cfg_.existence_lidar_confirms)
                {
                    inst.existence.integrate(ev);
                    integrated = true;
                }
            }
        }

        // Debounce advances ONLY on an evidence cycle, so it counts sustained EVIDENCE agreement (not wall-clock
        // cycles) regardless of the two sensors' rates. A transient hiccup still can't delete a real table.
        //
        // ★2026-08-06 — AND IT MUST COUNT LOOKS, NOT CYCLES. `++streak` on any cycle where should_remove(L) holds
        // made this a TIMER: once L dipped below the threshold, cycles carrying ZERO evidence (p_detect = 0 —
        // "I could not have seen it from here") kept the count climbing, so a table nudged under by two ambiguous
        // frames was deleted 15 cycles later by the mere passage of uninformative time. The LiDAR ratchet hid
        // this by pinning L at the clamp; removing that vote exposed it, and the replay deleted the REAL table at
        // cycle 98 — its cycles 96–132 read sndet 14/16/14/7 with p_detect 0.000 throughout.
        //
        // A cycle now contributes its own p_detect, so ExistenceRemoveFrames means "this many FULLY-RESOLVING
        // looks' worth of sustained absence" — which is exactly the stated test: from a viewpoint where we can
        // infer YOLO should have detected a table, it did not, and enough times over. An unresolvable cycle adds
        // 0 and HOLDS (it is neither evidence to delete nor evidence to forgive); occupancy still resets to 0.
        if (integrated)
        {
            if (inst.existence.should_remove(cfg_.existence_removal_prob))
                inst.existence_remove_streak += cycle_p_detect;
            else
                inst.existence_remove_streak = 0.0f;

            if (fitter.should_log(inst))
                std::print("[{}] [existence] L={:.2f} p={:.2f} | lidar occ={:.1f} free={:.1f} n={} | sil occ={:.0f} free={:.0f} ndet={} | {} streak={:.1f}\n",
                           inst.node_name, inst.existence.logodds(), inst.existence.p_exists(),
                           inst.dbg_ex_lidar_occ, inst.dbg_ex_lidar_free, inst.dbg_ex_lidar_n,
                           inst.dbg_ex_sil_occ, inst.dbg_ex_sil_free, inst.dbg_ex_sil_ndet,
                           observed ? "obs" : "-", inst.existence_remove_streak);

            // Persist the existence trajectory across the OUT-OF-FoV stretch that a fit row can't reach: an
            // out-of-view instance is not fit (no fresh mask) so no ai2_log row is written for it — yet this is
            // exactly where a spurious LiDAR-absence carve accrues and removes a real table. Emit a row here for
            // non-observed instances only (observed ones already got a fit row this cycle → no double-log).
            if (not observed)
                fitter.log_existence_cycle(inst);
        }
        if (inst.existence_remove_streak >= cfg_.existence_remove_frames)
            doomed.push_back(id);
    }
    ev_g.removals     += static_cast<int>(doomed.size());   // EvidenceMonitor counters
    ev_g.removals_cum += static_cast<long>(doomed.size());
    for (const std::uint64_t id : doomed)
    {
        std::print("table_concept: [existence] removing table id={} — free-space evidence (empty volume)\n", id);
        if (auto it = fitter.instances().find(id); it != fitter.instances().end())
        {
            // Record BEFORE teardown, while the existence state that justified the kill is still readable.
            // Shadow mode: this cannot alter the decision (see the header comment on on_remove).
            if (on_remove) on_remove(id, it->second);
            it->second.affordance.remove();
        }
        fitter.forget_node(id);
        G_->delete_node(id);
    }
}

}  // namespace rc
