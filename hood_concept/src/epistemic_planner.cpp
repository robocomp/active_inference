/*
 * epistemic_planner.cpp  —  Σ-based D-optimal next-best-view over the hood's vertical faces,
 *                           weighted by the probability the detector fires from where we propose to stand.
 *
 * Only the four vertical faces (±X, ±Y in hood frame) are considered: a floor-navigating robot cannot
 * observe the top face from above or the bottom face at all. Each face is scored by the expected entropy
 * reduction ½·ln det(I₆ + Σ·ΔI) on the belief's full covariance — the information a detection WOULD yield —
 * and `rc::nbv` multiplies that by P(detect | framing) at that face's own stand-off, so the winner is the most
 * informative view we can actually GET. The scalar affordance value stays bounded by the adequacy gap to Σ*.
 *
 * ★WHAT WAS WRONG BEFORE (fixed 2026-08-07). The stand-off inverted a HORIZONTAL-ONLY framing model:
 * d = R_circ / (fill·tan(hfov/2)), with R_circ the footprint's circumscribed radius. But `roi_fill` — the
 * quantity the detector envelope is a function of, and the one hood_projection.cpp measures — is
 * max(Δcol/W, Δrow/H), a max over BOTH image axes, and for a 1.7 m box seen from a camera mounted at 0.945 m
 * the VERTICAL axis binds by ~4×. Measured on the live rig (ZED 1280×720, hfov 110°, fx=fy=448): the planner
 * proposed 1.21 m from the face, where the predicted fill is 0.87 and P(detect) = 0.010. The robot was being
 * sent to the one place a mask could not form, and the removal channel then read the resulting absence as
 * evidence the hood was gone. Two smaller errors pushed the same way — the fill inversion was
 * small-angle (~12 % high at fill≈0.4) and R_circ was added to a FACE-relative stand-off. All three now live
 * in common/nbv/viewpoint_score.h, which projects the box corners exactly as the projector does.
 */

#include "epistemic_planner.h"
#include "hood_dof.h"   // kHoodDofs: names/units/σ* — shared with the dashboard
#include "../../common/nbv/viewpoint_score.h"   // rc::nbv — the shared detection-weighted NBV core

#include <algorithm>
#include <array>
#include <cmath>
#include <print>
#include <vector>

// ─── Face sampler ────────────────────────────────────────────────────────────

namespace rc {

namespace
{
// Samples surface points on one vertical face of the 6-DOF HoodBeliefState [cx,cy,H,w,h,yaw], for ΔI.
std::vector<Eigen::Vector3f> sample_face_surface(const HoodBeliefState& s, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    const float cyaw = std::cos(s.yaw), syaw = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts; pts.reserve(kTangent * kVert);
    for (int m = 0; m < kTangent; ++m)
    {
        const float t = (kTangent > 1) ? (-1.0f + 2.0f * m / (kTangent - 1)) : 0.0f;
        float lx = 0.0f, ly = 0.0f;
        switch (face_idx)
        {
            case 0: lx =  hw;     ly =  t * hh; break;   // +x
            case 1: lx = -hw;     ly =  t * hh; break;   // -x
            case 2: lx =  t * hw; ly =  hh;     break;   // +y
            default: lx = t * hw; ly = -hh;     break;   // -y
        }
        const float rx = s.cx + cyaw * lx - syaw * ly;
        const float ry = s.cy + syaw * lx + cyaw * ly;
        for (int k = 0; k < kVert; ++k)
        {
            const float z = s.H * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(rx, ry, z);
        }
    }
    return pts;
}

}  // namespace

// ─── EpistemicPlanner ────────────────────────────────────────────────────────

// AI2-native Σ-based D-optimal next-best-view: score the four faces, return the best framed viewpoint.
EpistemicProposal EpistemicPlanner::compute(const HoodBelief& belief, float lat_rate, float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<EpistemicPlanner::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon) const
{
    constexpr float kRobotRadiusM = 0.30f;   // Shadow's footprint radius — a physical dimension, not a knob

    // The sensor arrives from the caller with the camera's real FoVs + mount height; the detector envelope is
    // ours (the same one the removal channel weights absence by, so create and remove agree on "could we have
    // seen it?"). One model, both directions.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;

    // ★NO SENSOR MODEL ⇒ NO PROPOSAL. A hood is a TALL box, so the VERTICAL axis is the one that binds:
    // without vfov the fill model silently reverts to horizontal-only and proposes ~0.9 m, which is precisely
    // the "no mask can form here" pose this whole rewrite exists to prevent. Degrading to a horizontal answer
    // looks graceful and is not — and it is WORSE than no answer, because hood_affordance freezes the
    // target pose for the duration of an executing claim, so one proposal made in the startup window (before
    // robot_concept has published cam_rgb_focaly, or before the room→zed chain resolves) is locked in for the
    // whole execution. "I do not know the camera yet" must stay silent, not guess.
    // The condition itself is rc::nbv::Sensor::complete() — ONE definition, shared with the choke point inside
    // plan_faces() that refuses for every agent. Kept here as well only to name the HOOD in the log and
    // to skip the belief work; it is deliberately the same predicate, not a second opinion.
    if (not sensor.complete())
    {
        static bool warned = false;
        if (not warned)
        {
            warned = true;
            std::print("[epistemic-NBV] NO PROPOSAL: incomplete sensor model (vfov={:.3f} rad, mount_z={:.3f} m). "
                       "Waiting for the ZED intrinsics + room→zed transform; a horizontal-only stand-off would "
                       "be locked in by the affordance freeze.\n", sensor.vfov_rad, sensor.height_m);
        }
        return {};
    }

    // REPORTED covariance: Σ with the yaw entry inflated by the discrete-mode entropy (p(1−p)(π/2)²), so a
    // near-square hood whose orientation mode is unresolved shows a large yaw variance → the D-optimal NBV
    // scores a mode-discriminating (side/leg) view highly and drives the orbit that resolves it.
    const Eigen::Matrix<float, 6, 6> S = belief.covariance_reported().topLeftCorner<6, 6>();   // [cx,cy,H,w,h,yaw] (drop the tilt-calibration DOF)
    const HoodBeliefState& s = belief.state();

    // ── Adequacy gap (active-perception step 1, HOOD.md) ─────────────────────────────────
    // Target precision Σ* and the gap formula now live in hood_dof.h /
    // common/ai_belief/dof_spec.h — ONE table shared with the dashboard, so the demands the planner acts
    // on are exactly the ones the BeliefInspector shows. Same values, same per-DOF clamp, same result.
    // A mode-ambiguous hood (marginal σyaw≈45°) stays inadequate on the yaw DOF until the mode
    // resolves, so the robot keeps gathering evidence on it before releasing.
    const float adequacy_gap = adequacy_gap_nats(kHoodDofs, [&](std::size_t j) { return S(j, j); });

    // ── The object as the belief holds it — estimate AND uncertainty ────────────────────────────
    // σ_pos/σ_extent are what make the FRAMING precision-aware: rc::nbv marginalises P(detect) over them, so a
    // poorly-known hood asks for a safer framing on its own. That is the model-level replacement for the
    // hand-picked stand-off margin this planner used to carry.
    rc::nbv::Target target;
    target.cx = s.cx; target.cy = s.cy; target.yaw = s.yaw;
    target.w  = s.w;  target.h  = s.h;
    target.z0 = 0.0f; target.z1 = s.H;   // the box stands on the floor; z1 is what binds the vertical framing
    target.sigma_pos_m    = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));
    target.sigma_extent_m = std::sqrt(std::max(0.0f, 0.5f * (S(3, 3) + S(4, 4))));

    // ── The belief half: what a detection from face i at range d would tell us ───────────────────
    // Rᵢ = σ_base² + (lat_rate·d)² — range-aware, so the same face is worth less from further away. rc::nbv
    // calls this at each face's own stand-off, so the trade "further ⇒ detectable but less informative" is
    // resolved with the real numbers on both sides instead of one shared range.
    const Eigen::Matrix<float, 6, 6> I6 = Eigen::Matrix<float, 6, 6>::Identity();
    const auto raw_gain = [&](int face_idx, float standoff) -> float
    {
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const Eigen::Matrix<float, 6, 6> dI =
            belief.predicted_information(sample_face_surface(s, face_idx), Ri).topLeftCorner<6, 6>();
        const float det = (I6 + S * dI).determinant();
        return std::max(0.0f, 0.5f * std::log(std::max(1e-9f, det)));   // single-view D-optimal info (nats)
    };

    // ★The room polygon is what kills the through-the-wall faces. A hood is wall-anchored, so its back
    // (and often one side) presents a perfectly VISIBLE face whose viewpoint is outside the room — and the raw
    // information term cannot tell front from back. Without the polygon `is_reachable` returns true for every
    // pose and such a face can win outright; nothing downstream refuses it either, because an unroutable goal
    // is REPAIRED, not rejected. Passing it turns "the target is behind the wall" into "that face is not a
    // viewpoint", which is a feasibility fact, not a preference.
    const rc::nbv::Plan plan = rc::nbv::plan_faces(target, sensor, kRobotRadiusM, obstacles, raw_gain,
                                                   room_polygon);
    if (not plan.valid)
        return {};
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. `plan.best_pos` is then the raw argmax, which for a
    // wall-anchored box is a pose behind the wall. Publishing it hands the controller an unroutable standpoint,
    // and the controller REPAIRS rather than rejects: nearest_reachable is measured from the robot, so the goal
    // snaps to the nearest occupiable cell — the floor right at the fridge. An invalid proposal makes the caller
    // hold_offered() instead, which is the honest answer: "no viewpoint from here". (door_concept already does
    // this; the hood did not, so the refusal path did not exist on this side.)
    if (not plan.any_usable)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("hood_concept: [NBV] no usable face — every viewpoint is inside an obstacle or "
                       "outside the room. REFUSING (the raw argmax would stand in the wall).\n");
        return {};
    }

    const int   bi        = plan.best_face;
    const float best_raw  = plan.face_raw_gains[bi];
    // SCALAR affordance value (epistemic_gain): the winning face's EXPECTED info (already P(detect)-weighted,
    // and 0 if the pose stands inside another object) BOUNDED by the adequacy gap to Σ*, so it → 0 as the belief
    // reaches the consumer's precision — a threshold-free "done" AND the cross-affordance EFE currency.
    // Information beyond Σ* is worthless to the consumer, so an adequate hood stops attracting.
    // The bound applies to the USEFUL fraction of the raw information; the detection weighting rides along
    // untouched, which is what makes the old p_observable=1.0 stub finally a real number.
    const float useful_frac = (best_raw > 1e-9f) ? std::min(best_raw, adequacy_gap) / best_raw : 0.0f;
    const float best_gain   = plan.face_gains[bi] * useful_frac;

    // Verification readout (throttled): the chosen face should be perpendicular to Σ's dominant
    // uncertainty direction — e.g. dom-unc = h ⇒ a ±y face wins. Confirms the NBV attacks the worst DOF.
    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        // Dominant-uncertainty DOF, normalised by a common scale (10 cm / 0.1 rad) so metres and radians
        // are comparable — "which DOF is least known relative to its natural scale". (The gain decision
        // itself uses the unit-invariant log-det; this is only a readable diagnostic label.)
        constexpr float kRef = 0.10f;   // common scale: 10 cm / 0.1 rad
        static const char* fn[4] = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        // Emit the ACTUAL stand-off, the framing it realises and the detection probability there, so
        // "the affordance is too close" is a readable number rather than something inferred from the viewer.
        // The stand-off is FACE-relative; dist_from_centre is what the viewer shows.
        const float fill = rc::nbv::predicted_fill(target, plan.best_pos, sensor, plan.best_yaw);
        std::print("[epistemic-NBV] standoff={:.2f}m band=[{:.2f},{:.2f}] fov=({:.0f},{:.0f})deg cam_z={:.2f} "
                   "fill={:.2f}→p_det={:.3f} vis={:.2f} target=({:.2f},{:.2f}) dist_from_centre={:.2f}m\n",
                   plan.best_standoff_m, plan.standoff_min_m, plan.standoff_max_m,
                   sensor.hfov_rad * 57.29578f, sensor.vfov_rad * 57.29578f, sensor.height_m,
                   fill, plan.face_p_detect[bi], plan.face_visible[bi],
                   plan.best_pos.x(), plan.best_pos.y(),
                   std::hypot(plan.best_pos.x() - s.cx, plan.best_pos.y() - s.cy));
        std::print("[epistemic-NBV] face={} gain={:.3f}(raw {:.3f}) adq_gap={:.3f} | Σ dom-unc={} σ={:.3f}{} | "
                   "expected +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f} (p_det {:.2f},{:.2f},{:.2f},{:.2f})\n",
                   fn[bi], best_gain, best_raw, adequacy_gap, kHoodDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kHoodDofs[dom].unit,
                   plan.face_gains[0], plan.face_gains[1], plan.face_gains[2], plan.face_gains[3],
                   plan.face_p_detect[0], plan.face_p_detect[1], plan.face_p_detect[2], plan.face_p_detect[3]);
    }

    EpistemicProposal proposal{plan.best_pos.x(), plan.best_pos.y(), plan.best_yaw, best_gain, true};
    // Object-relative viewpoint constraint (authoritative): publish ALL four faces with their EXPECTED per-face
    // gains (the ranking — P(detect)-weighted so an undetectable face cannot out-bid travel cost, and NOT
    // adequacy-clamped, else they tie) + the winning face's stand-off band + the framing the servo must drive
    // to + Σ*, so the controller picks the best FEASIBLE face itself. Face order matches [+x,-x,+y,-y].
    proposal.face_gains     = plan.face_gains;
    proposal.standoff_min_m = plan.standoff_min_m;
    proposal.standoff_max_m = plan.standoff_max_m;
    // The framing the servo advances to is the argmax of the SAME envelope the stand-off realises. Publishing a
    // different constant (this shipped 0.45) makes the servo drive the robot off the pose the planner chose.
    proposal.framing_fill   = plan.framing_fill;
    proposal.sigma_star     = sigma_star_array<6>(kHoodDofs);   // the SAME demands the adequacy gap used
    proposal.chosen_standoff_m = plan.best_standoff_m;
    proposal.chosen_p_detect   = plan.face_p_detect[bi];
    proposal.sensor_vfov_rad   = sensor.vfov_rad;
    if (not proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
