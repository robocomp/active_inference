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
                                       bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g)
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

        // SILHOUETTE / MASK channel (pixel-level) — CAMERA clock: project the tabletop silhouette and compare,
        // per predicted-VISIBLE pixel, against the current YOLO foreground. Lit by a "table" mask ⇒ occupancy
        // (holds L up); lit by NOTHING ⇒ predicted-but-absent (the "gone" signal, present EVEN WITH NO YOLO MASK);
        // lit by a NON-table mask ⇒ occluded ⇒ HOLD (never false absence behind a nearer object). n_detectable==0
        // (out of FoV / fully occluded) ⇒ mask_evidence HOLDs.
        if (fresh_masks)
        {
            const auto sil = fitter.compute_silhouette_existence(inst);
            inst.dbg_ex_sil_occ = sil.e_occ; inst.dbg_ex_sil_free = sil.e_free; inst.dbg_ex_sil_ndet = sil.n_detectable;
            if (sil.n_detectable > 0)
            {
                // Observed-guard (mirrors the LiDAR hollow guard): if the table was DETECTED by any sensor this
                // frame it is not gone, so suppress silhouette ABSENCE — a ZED false-negative, or a table seen
                // only by the ricoh whose silhouette the ZED-based check can't confirm, must never vote it away.
                // Occupancy always counts. (The ricoh-projected silhouette, once the producer ships 360 mask
                // pixels, will let absence be judged in the ricoh view directly instead of relying on this guard.)
                const float raw_free = observed ? 0.0f : sil.e_free;
                // P(detect | present, geometry): how confidently the ZED would resolve this table FROM HERE.
                // in_fov_frac folds the real FRUSTUM + occlusion; range_conf the angular-size drop; central_frac
                // whether the robot is actually LOOKING at it (a peripheral table clipping the wide FoV edge is
                // NOT a verifying view). A predicted-visible-but-absent observation is only evidence of REMOVAL in
                // proportion to P(detect); the REST is epistemic surprise — "I can't resolve this from here" — which
                // must make the robot GO VERIFY, never delete a table it never properly looked at.
                // A verifying view = CLOSE (range_conf) + in frustum & unoccluded (in_fov_frac) + the robot is
                // LOOKING at it (central_frac). NOT obliquity: this robot is chronically edge-on and an edge-on
                // table IS detectable up close (table_1), so obliquity would wrongly block legitimate removal.
                const float p_detect = absence_range_conf(sil.mean_range_m) * sil.in_fov_frac() * sil.central_frac();
                const float sfree  = raw_free * p_detect;                 // confident absence → removal log-odds
                const float verify = raw_free * (1.0f - p_detect);        // un-confident absence → go-verify surprise
                inst.dbg_ex_sil_free_eff = sfree;
                inst.dbg_ex_pdetect = p_detect; inst.dbg_ex_central = sil.central_frac();
                inst.existence.integrate(rc::exist::mask_evidence(sil.e_occ, sfree, sil.n_detectable, sm));
                integrated = true;
                // Route the un-resolvable absence into an epistemic VERIFY pull (decayed accumulator). When it
                // builds up, the table is flagged for verification (the epistemic planner drives the robot to a
                // good ZED viewpoint). Removal then only ever fires from a HIGH-p_detect view — verification-gated.
                inst.verify_surprise = 0.9f * inst.verify_surprise + 0.1f * verify;
                inst.wants_verification = inst.verify_surprise > cfg_.existence_verify_surprise;
            }
        }

        // LiDAR occupancy carve of the TOP-SLAB z-band [H−t, H] — LiDAR clock (the slab spans the full footprint
        // there, so a beam hits rim/top = occupancy or passes through = gone — no hollow ambiguity between legs).
        if (sweep)
        {
            rc::exist::Evidence ev = rc::exist::carve_box(origin, *sweep, bs.cx, bs.cy, bs.yaw, bs.w, bs.h,
                                                          bs.H - rc::TableModel::TOP_THICKNESS, bs.H, surf_sigma, sm);
            // Leg OCCUPANCY (the 3D model has 4 legs; the rings hit them too). Occupancy-ONLY: legs are thin, so a
            // MISS is not evidence of absence and the hollow space between them must never vote free. Carve each leg's
            // tall volume [0, H−t] and add only its e_occ — robustifies CONFIRMATION when the thin top slab sits at an
            // awkward height for the horizontal rings (tall legs are far likelier struck). Cannot cause a false removal.
            if (cfg_.existence_leg_occupancy)
            {
                const float lr   = rc::TableModel::LEG_RADIUS;
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
                    for (const auto [llx, lly] : {std::pair{ix, iy}, std::pair{-ix, iy}, std::pair{ix, -iy}, std::pair{-ix, -iy}})
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
                // OCCUPANCY-ONLY by default: a solid-slab model vs a thin real tabletop makes LiDAR "free"
                // unreliable (beams under the top surface read as gone), so it must not drive removal — only
                // the camera silhouette does. Suppress free unless it was observed (hollow guard) OR the
                // ExistenceLidarAbsence override is on. Occupancy still counts, holding L up.
                const bool suppress_free = true;   // LiDAR absence-evidence REFUTED on live data (A/B 2026-07-12)
                inst.dbg_ex_lidar_free_eff = suppress_free ? 0.0f : ev.e_free;
                ev.log_odds_delta = rc::exist::hollow_guarded_delta(ev, suppress_free, sm);
                inst.existence.integrate(ev);
                integrated = true;
            }
        }

        // Debounce advances ONLY on an evidence cycle, so it counts sustained EVIDENCE agreement (not wall-clock
        // cycles) regardless of the two sensors' rates. A transient hiccup still can't delete a real table.
        if (integrated)
        {
            if (inst.existence.should_remove(cfg_.existence_removal_prob)) ++inst.existence_remove_streak;
            else                                                          inst.existence_remove_streak = 0;

            if (fitter.should_log(inst))
                std::print("[{}] [existence] L={:.2f} p={:.2f} | lidar occ={:.1f} free={:.1f} n={} | sil occ={:.0f} free={:.0f} ndet={} | {} streak={}\n",
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
            it->second.affordance.remove();
        fitter.forget_node(id);
        G_->delete_node(id);
    }
}

}  // namespace rc
