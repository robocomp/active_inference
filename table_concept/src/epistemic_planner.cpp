/*
 * epistemic_planner.cpp  —  Σ-based D-optimal next-best-view over the table's vertical faces.
 *
 * Only the four vertical faces (±X, ±Y in table frame) are considered: a floor-navigating robot cannot
 * observe the top face from above or the bottom face at all. Each face is scored by the expected entropy
 * reduction ½·ln det(I₆ + Σ·ΔI) on the belief's full covariance, bounded by the adequacy gap to Σ*, and * the winning face's framed viewpoint + heading is returned. See epistemic_planner.h for the formulas.
 */

#include "epistemic_planner.h"
#include "../../common/detectability/detectability.h"   // rc::detect — the detector INVERSE model
#include "table_dof.h"          // kTableDofs: names/units/σ* — one table shared with the dashboard

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
// Samples surface points on one vertical face of the 6-DOF TableBeliefState [cx,cy,H,w,h,yaw], for ΔI.
std::vector<Eigen::Vector3f> sample_face_surface(const TableBeliefState& s, int face_idx)
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

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// AI2-native Σ-based D-optimal next-best-view, WEIGHTED BY THE PROBABILITY OF ACTUALLY GETTING A DETECTION.
// The geometry (framing, stand-off, visibility, P(detect)) lives in common/nbv; this function supplies only
// what the belief knows — the information a detection on face i would yield.
EpistemicProposal EpistemicPlanner::compute(const TableBelief& belief, float lat_rate, float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<EpistemicPlanner::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon) const
{
    // The camera geometry arrives from the caller (read fresh from the graph each cycle); the detector
    // envelope is ours. Merging here keeps the two concerns with their right owners.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;
    // REPORTED covariance: Σ with the yaw entry inflated by the discrete-mode entropy (p(1−p)(π/2)²), so a
    // near-square table whose orientation mode is unresolved shows a large yaw variance → the D-optimal NBV
    // scores a mode-discriminating (side/leg) view highly and drives the orbit that resolves it.
    const Eigen::Matrix<float, 6, 6> S = belief.covariance_reported().topLeftCorner<6, 6>();   // [cx,cy,H,w,h,yaw] (drop the tilt-calibration DOF)
    const TableBeliefState& s = belief.state();

    // ── Adequacy gap (active-perception step 1, TABLE.md) ─────────────────────────────────
    // Target precision Σ* and the gap formula now live in table_dof.h / common/ai_belief/dof_spec.h —
    // ONE table shared with the dashboard, so the demands the planner acts on are exactly the ones the
    // BeliefInspector shows. Same values, same per-DOF clamp, same result as the old local kTargetStd:
    // `t` (depth tilt) declares no demand and is skipped, just as the old `j < 6` loop skipped it.
    // A mode-ambiguous table (marginal σyaw≈45°) stays inadequate on the yaw DOF until the mode resolves,
    // so the robot keeps gathering evidence on it before releasing.
    const float adequacy_gap = adequacy_gap_nats(kTableDofs, [&](std::size_t j) { return S(j, j); });

    // ── The shared detection-weighted plan (common/nbv) ───────────────────────────────────
    // The object as the BELIEF holds it — estimate AND uncertainty. The σ's are what make the framing
    // precision-aware: P(detect) is unimodal in projected fill, so a poorly-known table is framed further from
    // the overflow shoulder automatically. That is the model-level replacement for the 0.45 m margin constant
    // this function used to carry.
    rc::nbv::Target target;
    target.cx = s.cx; target.cy = s.cy; target.yaw = s.yaw;
    target.w  = s.w;  target.h  = s.h;
    target.z0 = 0.0f; target.z1 = s.H;                                   // a table occludes from the floor up
    target.sigma_pos_m    = std::sqrt(std::max({0.0f, S(0, 0), S(1, 1)}));   // cx, cy
    target.sigma_extent_m = std::sqrt(std::max({0.0f, S(3, 3), S(4, 4)}));   // w, h

    // RAW per-face D-optimal gain (nats), UNBOUNDED — the information a DETECTION on that face would yield.
    // Not adequacy-clamped: each face's single-view info typically EXCEEDS the remaining gap, so clamping made
    // all four tie (the degenerate ranked-set that erased the controller's basis to prefer one face). The
    // adequacy bound belongs on the SCALAR value below.
    const Eigen::Matrix<float, 6, 6> I6 = Eigen::Matrix<float, 6, 6>::Identity();
    const auto raw_gain_of = [&](int i, float standoff)
    {
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const Eigen::Matrix<float, 6, 6> dI = belief.predicted_information(sample_face_surface(s, i), Ri).topLeftCorner<6, 6>();
        const float det = (I6 + S * dI).determinant();
        return std::max(0.0f, 0.5f * std::log(std::max(1e-9f, det)));
    };

    const rc::nbv::Plan plan = rc::nbv::plan_faces(target, sensor, robot_radius_m_, obstacles, raw_gain_of,
                                                   room_polygon);
    if (not plan.valid)
        return {};
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. plan.best_pos is then the raw argmax, which for a
    // wall-adjacent object is a pose behind the wall. The controller REPAIRS rather than rejects an
    // unroutable standpoint — nearest_reachable is measured from the robot, so the goal snaps to the floor
    // right at the object. An invalid proposal makes the caller hold instead: "no viewpoint from here".
    if (not plan.any_usable)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("table_concept: [NBV] no usable face — every viewpoint is inside an obstacle or "
                       "outside the room. REFUSING (the raw argmax would stand in the wall).\n");
        return {};
    }

    const int   best_idx = plan.best_face;
    const float best_raw = plan.face_gains[best_idx];   // DETECTION-WEIGHTED — the honest expected value
    if (not std::isfinite(best_raw))
        return {};

    // SCALAR affordance value (epistemic_gain): the winning face's info BOUNDED by the adequacy gap to Σ*, so
    // it → 0 as the belief reaches the consumer's precision — a threshold-free "done" AND the cross-affordance
    // EFE currency. Information beyond Σ* is worthless to the consumer, so an adequate table stops attracting.
    // The old `p_observable = 1.0` placeholder ("the standoff is a framed viewpoint by construction") is gone:
    // P(detect) is now REAL and already folded into best_raw by plan_faces, so a table we cannot actually get
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
        static const char* fn[4] = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)   // S is the 6x6 slice the planner scores (the tilt DOF is dropped)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        // Both columns are printed on purpose: `raw` is what the belief wants, `exp` is what we can actually
        // expect to get. A large gap between them IS the diagnosis — it says the best-informing face is one the
        // detector cannot fire from, which is exactly the failure this weighting exists to catch.
        std::print("[epistemic-NBV] face={} gain={:.3f}(exp {:.3f}) adq_gap={:.3f} | Σ dom-unc={} σ={:.3f}{} | "
                   "d={:.2f}m fill*={:.2f} | exp +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f} | "
                   "raw {:.2f}/{:.2f}/{:.2f}/{:.2f} | pdet {:.2f}/{:.2f}/{:.2f}/{:.2f}\n",
                   fn[best_idx], best_gain, best_raw, adequacy_gap, kTableDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kTableDofs[dom].unit,
                   plan.best_standoff_m, plan.framing_fill,
                   plan.face_gains[0], plan.face_gains[1], plan.face_gains[2], plan.face_gains[3],
                   plan.face_raw_gains[0], plan.face_raw_gains[1], plan.face_raw_gains[2], plan.face_raw_gains[3],
                   plan.face_p_detect[0], plan.face_p_detect[1], plan.face_p_detect[2], plan.face_p_detect[3]);
    }

    EpistemicProposal proposal{plan.best_pos.x(), plan.best_pos.y(), plan.best_yaw, best_gain, true};
    // Object-relative viewpoint constraint (authoritative): publish ALL four faces with their DETECTION-
    // WEIGHTED per-face gains (the ranking — NOT adequacy-clamped, else they tie) + the stand-off band +
    // framing + Σ*, so the controller picks the best FEASIBLE face itself. Publishing the weighted gains is
    // what makes the controller's own fallback honest: when it steps down to a lower-ranked face it is now
    // reading a number that already accounts for whether a mask is obtainable from there.
    // Face order matches the [+x,-x,+y,-y] sampling order.
    proposal.face_gains     = plan.face_gains;
    proposal.standoff_min_m = plan.standoff_min_m;   // detectability band, not a hand-picked floor
    proposal.standoff_max_m = plan.standoff_max_m;
    // The framing the servo drives to is the SAME argmax the stand-off realises. It used to be a separate 0.45
    // literal, so the servo spent its approach undoing the planner's choice of range.
    proposal.framing_fill   = plan.framing_fill;
    proposal.chosen_standoff_m = plan.best_standoff_m;
    proposal.chosen_p_detect   = plan.face_p_detect[best_idx];
    proposal.chosen_fill       = rc::nbv::predicted_fill(target, plan.best_pos, sensor, plan.best_yaw);
    proposal.sensor_vfov_rad   = sensor.vfov_rad;
    proposal.sigma_star     = sigma_star_array<6>(kTableDofs);   // the SAME demands the adequacy gap used
    if (not proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
