/*
 * epistemic_planner.cpp  —  Σ-based D-optimal next-best-view over the cabinet's vertical faces.
 *
 * Only the four vertical faces (±X, ±Y in cabinet frame) are considered: a floor-navigating robot cannot
 * observe the top face from above or the bottom face at all. Each face is scored by the expected entropy
 * reduction ½·ln det(I₆ + Σ·ΔI) on the belief's full covariance, bounded by the adequacy gap to Σ*, and
 * the winning face's framed viewpoint + heading is returned. See epistemic_planner.h for the formulas.
 */

#include "epistemic_planner.h"
#include "cabinet_dof.h"        // kCabinetDofs: names/units/σ* — one table shared with the dashboard

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <vector>

// ─── Face sampler ────────────────────────────────────────────────────────────

namespace rc {

namespace
{
// Samples surface points on one vertical face of the 7-DOF CabinetBeliefState
// [cx,cy,yaw,L,d,z0,z1], for ΔI. Face 0/1 are the run's END CAPS (±along the axis), 2/3 the FRONT
// and BACK faces. The back face is normally unviewable (it is against the wall) — the planner does
// not special-case that: a face the robot cannot stand off from simply never wins on feasibility.
std::vector<Eigen::Vector3f> sample_face_surface(const CabinetBeliefState& s, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const float hw = s.L * 0.5f, hh = s.d * 0.5f;
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
            const float f = (kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f;
            const float z = s.z0 + f * s.height();
            pts.emplace_back(rx, ry, z);
        }
    }
    return pts;
}


}  // namespace

// ─── EpistemicPlanner ────────────────────────────────────────────────────────

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// AI2-native Σ-based D-optimal next-best-view: score the four faces, return the best framed viewpoint.
EpistemicProposal EpistemicPlanner::compute(const CabinetBelief& belief, float lat_rate, float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            bool verbose,
                                            const std::vector<rc::nbv::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon) const
{
    // The camera geometry arrives from the caller (read fresh from the graph each cycle); the
    // detector envelope is ours. Merging here keeps the two concerns in their right owners.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;

    // REPORTED covariance: Σ with the (d,z0,z1) entries inflated by the discrete TIER-mode entropy
    // (p(1−p)Δ²), so a cabinet whose base-vs-wall tier is still unresolved shows a large σ_z → it stays
    // inadequate on those DOFs and the D-optimal NBV scores a tier-discriminating (height) view highly,
    // driving the view that resolves the mode.
    const Eigen::Matrix<float, 7, 7> S = belief.covariance_reported();   // [cx,cy,yaw,L,d,z0,z1] — full state, matching predicted_information's ordering
    const CabinetBeliefState& s = belief.state();

    // ── Adequacy gap (active-perception step 1, CABINET.md) ─────────────────────────────────
    // Target precision Σ* and the gap formula now live in cabinet_dof.h / common/ai_belief/dof_spec.h —
    // ONE table shared with the dashboard, so the demands the planner acts on are exactly the ones the
    // BeliefInspector shows. Same values (incl. the deliberately loose L), same per-DOF clamp, same
    // result as the old local kTargetStd. A mode-ambiguous cabinet (marginal σyaw≈45°) stays inadequate
    // on the yaw DOF until the mode resolves, so the robot keeps gathering evidence before releasing.
    const float adequacy_gap = adequacy_gap_nats(kCabinetDofs, [&](std::size_t j) { return S(j, j); });

    // ── The shared detection-weighted plan (common/nbv) ───────────────────────────────────
    // The cabinet as the BELIEF holds it — estimate AND uncertainty. The σ's make the framing precision-aware:
    // P(detect) is unimodal in projected fill, so a poorly-known run is framed further from the overflow
    // shoulder automatically. That is the model-level replacement for the 0.45 m margin constant.
    // z0/z1 are real DOFs here, so a wall-tier cabinet (z0≈1.4 m) is correctly framed on the VERTICAL axis.
    rc::nbv::Target target;
    target.cx = s.cx; target.cy = s.cy; target.yaw = s.yaw;
    target.w  = s.L;  target.h  = s.d;
    target.z0 = s.z0; target.z1 = s.z1;
    target.sigma_pos_m    = std::sqrt(std::max({0.0f, S(0, 0), S(1, 1)}));   // cx, cy
    target.sigma_extent_m = std::sqrt(std::max({0.0f, S(3, 3), S(4, 4)}));   // L, d

    // RAW per-face D-optimal gain (nats), UNBOUNDED — the information a DETECTION on that face would yield.
    // Not adequacy-clamped: each face's single-view info typically EXCEEDS the remaining gap, so clamping made
    // all four tie (the degenerate ranked-set that erased the controller's basis to prefer one face). The
    // adequacy bound belongs on the SCALAR value below.
    const Eigen::Matrix<float, 7, 7> I6 = Eigen::Matrix<float, 7, 7>::Identity();
    const auto raw_gain_of = [&](int i, float standoff)
    {
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const Eigen::Matrix<float, 7, 7> dI = belief.predicted_information(sample_face_surface(s, i), Ri);
        const float det = (I6 + S * dI).determinant();
        return std::max(0.0f, 0.5f * std::log(std::max(1e-9f, det)));
    };

    const rc::nbv::Plan plan = rc::nbv::plan_faces(target, sensor, robot_radius_m_, obstacles, raw_gain_of,
                                                   room_polygon);
    if (not plan.valid)
        return {};
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. plan.best_pos is then the raw argmax, which for a
    // wall-anchored object is a pose behind the wall. Publishing it hands the controller an unroutable
    // standpoint, and the controller REPAIRS rather than rejects: nearest_reachable is measured from the
    // robot, so the goal snaps to the nearest occupiable cell — the floor right at the object. An invalid
    // proposal makes the caller hold_offered() instead: "no viewpoint from here", which is the honest answer.
    if (not plan.any_usable)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("cabinet_concept: [NBV] no usable face — every viewpoint is inside an obstacle or "
                       "outside the room. REFUSING (the raw argmax would stand in the wall).\n");
        return {};
    }
    const int   best_idx  = plan.best_face;
    const float best_raw  = plan.face_gains[best_idx];   // DETECTION-WEIGHTED — the honest expected value
    const auto& face_raw  = plan.face_gains;
    if (not std::isfinite(best_raw))
        return {};

    // SCALAR affordance value (epistemic_gain): the winning face's info BOUNDED by the adequacy gap to Σ*, so
    // it → 0 as the belief reaches the consumer's precision — a threshold-free "done" AND the cross-affordance
    // EFE currency. Information beyond Σ* is worthless to the consumer, so an adequate cabinet stops attracting.
    // The old `p_observable = 1.0` placeholder ("the standoff is a framed viewpoint by construction") is gone:
    // P(detect) is now REAL and already folded into best_raw by plan_faces, so a cabinet we cannot actually get
    // a mask of publishes a near-zero gain and stops bidding for travel.
    const float best_gain = std::min(best_raw, adequacy_gap);

    // Verification readout (throttled): the chosen face should be perpendicular to Σ's dominant
    // uncertainty direction — e.g. dom-unc = h ⇒ a ±y face wins. Confirms the NBV attacks the worst DOF.
    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        // Dominant-uncertainty DOF, normalised by a common scale (10 cm / 0.1 rad) so metres and radians
        // are comparable — "which DOF is least known relative to its natural scale". (The gain decision
        // itself uses the unit-invariant log-det; this is only a readable diagnostic label.)
        constexpr float kRef = 0.10f;   // common scale: 10 cm / 0.1 rad
        static const char* fn[4] = {"end+", "end-", "front", "back"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)   // all N (the old loop stopped at 6 and never named `z1`)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        if (verbose)
        // `exp` vs `raw`: a large gap says the best-informing face is one the detector cannot fire from --
        // exactly the failure the weighting exists to catch, so both columns are printed.
        std::print("[epistemic-NBV] face={} gain={:.3f}(exp {:.3f}) adq_gap={:.3f} | Σ dom-unc={} σ={:.3f}{} | "
                   "d={:.2f}m fill*={:.2f} | exp {:.2f}/{:.2f}/{:.2f}/{:.2f} | raw {:.2f}/{:.2f}/{:.2f}/{:.2f} | "
                   "pdet {:.2f}/{:.2f}/{:.2f}/{:.2f}\n",
                   fn[best_idx], best_gain, best_raw, adequacy_gap, kCabinetDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kCabinetDofs[dom].unit,
                   plan.best_standoff_m, plan.framing_fill,
                   face_raw[0], face_raw[1], face_raw[2], face_raw[3],
                   plan.face_raw_gains[0], plan.face_raw_gains[1], plan.face_raw_gains[2], plan.face_raw_gains[3],
                   plan.face_p_detect[0], plan.face_p_detect[1], plan.face_p_detect[2], plan.face_p_detect[3]);
    }

    EpistemicProposal proposal{plan.best_pos.x(), plan.best_pos.y(), plan.best_yaw, best_gain, true};
    // Object-relative viewpoint constraint (authoritative): publish ALL four faces with their DETECTION-WEIGHTED
    // per-face gains (the ranking — NOT adequacy-clamped, else they tie) + the sensor-model stand-off band + framing
    // + Σ*, so the controller picks the best FEASIBLE face itself (a blocked argmax face falls back to the
    // next reachable one). The affordance's scalar epistemic_gain stays adequacy-bounded (cross-affordance
    // selection + "done"). Face order matches the [+x,-x,+y,-y] sampling order above.
    proposal.face_gains     = plan.face_gains;      // DETECTION-WEIGHTED, so the controller's own fallback to a
    proposal.standoff_min_m = plan.standoff_min_m;  // lower-ranked face reads a number that already accounts
    proposal.standoff_max_m = plan.standoff_max_m;  // for whether a mask is obtainable from there.
    // The framing the servo drives to is the SAME argmax the stand-off realises. It used to be a separate 0.45
    // literal, so the servo spent its approach undoing the planner's choice of range.
    proposal.framing_fill   = plan.framing_fill;
    proposal.sigma_star     = sigma_star_array<7>(kCabinetDofs);   // the SAME demands the adequacy gap used
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
