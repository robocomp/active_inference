/*
 * cabinet_existence.cpp — see cabinet_existence.h. Evidence-based cabinet removal (existence log-odds + debounce).
 */

#include "cabinet_existence.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <print>
#include <utility>
#include <vector>

#include "cabinet_fitter.h"                                       // rc::CabinetFitter (instances, silhouette)
#include "cabinet_lidar_ingestor.h"                              // rc::CabinetLidarIngestor (per-plane sweeps)
#include "cabinet_model.h"                                       // CabinetModel::TOP_THICKNESS / LEG_RADIUS
#include "../../common/existence_belief/existence_belief.h"    // rc::exist:: carve_box / mask_evidence / …
#include "../../common/dashboard/evidence_monitor.h"           // rc::EvidenceGlobals

namespace rc {

// ─── Existence-based removal ─────────────────────────────────────────────────────────────────────

// Carve the LiDAR sweep(s) + silhouette against every cabinet footprint, integrate the per-instance existence
// log-odds, and remove the cabinets whose volume is demonstrably empty (debounced). See cabinet_existence.h.
void CabinetExistence::update_and_remove(CabinetFitter& fitter, CabinetLidarIngestor* lidar,
                                       bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                                       const std::function<void(std::uint64_t, const CabinetInstance&)>& on_remove)
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

        // SILHOUETTE / MASK channel (pixel-level) — CAMERA clock: project the tabletop silhouette and compare,
        // per predicted-VISIBLE pixel, against the current YOLO foreground. Lit by a "cabinet" mask ⇒ occupancy
        // (holds L up); lit by NOTHING ⇒ predicted-but-absent (the "gone" signal, present EVEN WITH NO YOLO MASK);
        // lit by a NON-cabinet mask ⇒ occluded ⇒ HOLD (never false absence behind a nearer object). n_detectable==0
        // (out of FoV / fully occluded) ⇒ mask_evidence HOLDs.
        if (fresh_masks)
        {
            const auto sil = fitter.compute_silhouette_existence(inst);
            inst.dbg_ex_sil_occ = sil.e_occ; inst.dbg_ex_sil_free = sil.e_free; inst.dbg_ex_sil_ndet = sil.n_detectable;
            if (sil.n_detectable > 0)
            {
                // Observed-guard (mirrors the LiDAR hollow guard): if the cabinet was DETECTED by any sensor this
                // frame it is not gone, so suppress silhouette ABSENCE — a ZED false-negative, or a cabinet seen
                // only by the ricoh whose silhouette the ZED-based check can't confirm, must never vote it away.
                // Occupancy always counts. (The ricoh-projected silhouette, once the producer ships 360 mask
                // pixels, will let absence be judged in the ricoh view directly instead of relying on this guard.)
                const float raw_free = observed ? 0.0f : sil.e_free;
                // P(detect | present, geometry): how confidently the ZED would resolve this cabinet FROM HERE.
                // in_fov_frac folds the real FRUSTUM + occlusion; range_conf the angular-size drop; central_frac
                // whether the robot is actually LOOKING at it (a peripheral cabinet clipping the wide FoV edge is
                // NOT a verifying view). A predicted-visible-but-absent observation is only evidence of REMOVAL in
                // proportion to P(detect); the REST is epistemic surprise — "I can't resolve this from here" — which
                // must make the robot GO VERIFY, never delete a cabinet it never properly looked at.
                // A verifying view = CLOSE (range_conf) + in frustum & unoccluded (in_fov_frac) + the robot is
                // LOOKING at it (central_frac). NOT obliquity: this robot is chronically edge-on and an edge-on
                // cabinet IS detectable up close (cabinet_1), so obliquity would wrongly block legitimate removal.
                const float p_detect = absence_range_conf(sil.mean_range_m) * sil.in_fov_frac() * sil.central_frac();
                cycle_p_detect = p_detect;
                const float sfree  = raw_free * p_detect;                 // confident absence → removal log-odds
                const float verify = raw_free * (1.0f - p_detect);        // un-confident absence → go-verify surprise
                inst.dbg_ex_sil_free_eff = sfree;
                inst.dbg_ex_pdetect = p_detect; inst.dbg_ex_central = sil.central_frac();
                inst.existence.integrate(rc::exist::mask_evidence(sil.e_occ, sfree, sil.n_detectable, sm));
                integrated = true;
                // Route the un-resolvable absence into an epistemic VERIFY pull (decayed accumulator). When it
                // builds up, the cabinet is flagged for verification (the epistemic planner drives the robot to a
                // good ZED viewpoint). Removal then only ever fires from a HIGH-p_detect view — verification-gated.
                inst.verify_surprise = 0.9f * inst.verify_surprise + 0.1f * verify;
                inst.wants_verification = inst.verify_surprise > cfg_.existence_verify_surprise;
            }
        }

        // LiDAR carve of the WHOLE carcass [z0, z1]. This is where a run differs fundamentally from a
        // table: a table is a thin slab on thin legs, so the model volume is mostly EMPTY and a
        // pass-through beam is not evidence the table is gone (hence the table's occupancy-only,
        // slab-band-only carve plus a separate leg pass). A cabinet run is a genuinely SOLID box, so
        // the carve is model-consistent over the full volume and free-space evidence is meaningful —
        // which is why ExistenceLidarAbsence can honestly be enabled here.
        if (sweep)
        {
            rc::exist::Evidence ev = rc::exist::carve_box(origin, *sweep, bs.cx, bs.cy, bs.yaw, bs.L, bs.d,
                                                          bs.z0, bs.z1, surf_sigma, sm);
            // Fold in the low bpearl as a second, independent probe of the same solid volume (it
            // strikes the base carcass squarely where the high helios grazes the worktop edge).
            if (sweep_bp)
            {
                const rc::exist::Evidence e2 = rc::exist::carve_box(origin_bp, *sweep_bp, bs.cx, bs.cy, bs.yaw,
                                                                    bs.L, bs.d, bs.z0, bs.z1, surf_sigma, sm);
                ev.e_occ += e2.e_occ; ev.e_free += e2.e_free; ev.n_reached += e2.n_reached;
            }
            inst.dbg_ex_lidar_occ = ev.e_occ; inst.dbg_ex_lidar_free = ev.e_free; inst.dbg_ex_lidar_n = ev.n_reached;
            if (ev.n_reached > 0)                                                    // 0 ⇒ not probed ⇒ HOLD
            {
                // Degrade the free/absence half by LiDAR range: far away the beams are sparse, so a
                // pass-through is weaker evidence that the run is gone.
                const float lidar_range = (origin - Eigen::Vector3f(bs.cx, bs.cy, bs.z1)).norm();
                ev.e_free *= absence_range_conf(lidar_range);
                // Unlike the table, a run IS a solid box, so LiDAR free-space is model-consistent and may
                // drive removal. Left config-gated (default OFF) so the first live runs prove it before it
                // can delete anything — the table's absence channel was refuted on live data, and although
                // the reason (thin slab vs solid model) does not apply here, that is an argument for
                // enabling it deliberately rather than assuming it.
                // ★★AND SUPPRESSED WHENEVER ANOTHER SENSOR IS HOLDING THE RUN THIS FRAME — the same rule the
                // silhouette channel applies at `raw_free` above, which the LiDAR carve never got. A
                // through-beam grazing the volume and a mask sitting on the object are not two opinions to
                // average: one resolved it, the other did not. Measured on the siblings 2026-08-11: hood's
                // LiDAR voted free > occupancy on 84% of probed cycles while its camera's admitted absence
                // summed to exactly zero over 50 983 rows, and bottle's ran 2.5x with YOLO holding the bottle
                // on 72% of them. Inert here while ExistenceLidarAbsence is off; correct the moment it is on.
                const bool suppress_free = observed or not cfg_.existence_lidar_absence;
                inst.dbg_ex_lidar_free_eff = suppress_free ? 0.0f : ev.e_free;
                ev.log_odds_delta = rc::exist::hollow_guarded_delta(ev, suppress_free, sm);
                inst.existence.integrate(ev);
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
            std::print("cabinet_concept: {}\n", rc::exist::stall_note(inst.node_name, inst.existence,
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
            // exactly where a spurious LiDAR-absence carve accrues and removes a real cabinet. Emit a row here for
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
        std::print("cabinet_concept: [existence] removing cabinet id={} — free-space evidence (empty volume)\n", id);
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
