/*
 * epistemic_planner.cpp  —  Σ-based D-optimal next-best-view over the table's vertical faces.
 *
 * Only the four vertical faces (±X, ±Y in table frame) are considered: a floor-navigating robot cannot
 * observe the top face from above or the bottom face at all. Each face is scored by the expected entropy
 * reduction ½·ln det(I₆ + Σ·ΔI) on the belief's full covariance, bounded by the adequacy gap to Σ*, and
 * the winning face's framed viewpoint + heading is returned. See epistemic_planner.h for the formulas.
 */

#include "epistemic_planner.h"

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

// AI2-native Σ-based D-optimal next-best-view: score the four faces, return the best framed viewpoint.
EpistemicProposal EpistemicPlanner::compute(const TableBelief& belief, float lat_rate, float sigma_base) const
{
    constexpr float kEffectiveHorizontalFovRad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    constexpr float kMinimumStandOffM = 1.15f;         // YOLO won't fire too close → keep a viewing gap
    constexpr float kStandOffSafetyMarginM = 0.45f;    // extra stand-off beyond the FoV-fit distance

    // REPORTED covariance: Σ with the yaw entry inflated by the discrete-mode entropy (p(1−p)(π/2)²), so a
    // near-square table whose orientation mode is unresolved shows a large yaw variance → the D-optimal NBV
    // scores a mode-discriminating (side/leg) view highly and drives the orbit that resolves it.
    const Eigen::Matrix<float, 6, 6> S = belief.covariance_reported().topLeftCorner<6, 6>();   // [cx,cy,H,w,h,yaw] (drop the tilt-calibration DOF)
    const TableBeliefState& s = belief.state();

    const float cy = std::cos(s.yaw), sy = std::sin(s.yaw);
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    struct Face { Eigen::Vector2f normal, centre; float half_span; };
    const std::array<Face, 4> faces = {{
        { { cy,  sy}, {s.cx + cy * hw, s.cy + sy * hw}, hh },   // +x
        { {-cy, -sy}, {s.cx - cy * hw, s.cy - sy * hw}, hh },   // -x
        { {-sy,  cy}, {s.cx - sy * hh, s.cy + cy * hh}, hw },   // +y
        { { sy, -cy}, {s.cx + sy * hh, s.cy - cy * hh}, hw },   // -y
    }};
    const float max_stand_off = std::max(kMinimumStandOffM, d_obs_);
    const auto standoff_for = [&](float half_span)
    { return std::clamp(half_span / std::tan(kEffectiveHorizontalFovRad * 0.5f) + kStandOffSafetyMarginM,
                        kMinimumStandOffM, max_stand_off); };

    // ── Adequacy gap (active-perception step 1, TABLE.md) ─────────────────────────────────
    // ★PLACEHOLDER target precision Σ* — the posterior covariance at which the table is "adequately
    // resolved" for its downstream consumer. It should be the physical manipulation tolerance (gripper
    // clearance / placement margin / approach-cone half-angle) pushed through ∂success/∂θ, and ultimately
    // PUBLISHED by the consuming grasp/place affordance. Hardcoded here until that channel exists — REPLACE.
    static constexpr std::array<float, 6> kTargetStd =   // [cx,cy,H,w,h,yaw]  (m / rad)
        {0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.05f};      // 2 cm pos/size, ~2.9° yaw
    // Remaining information (nats) to carry the belief down to Σ*, summed PER-DOF over the marginal
    // variances Σ_ii and clamped per DOF: ½·Σ_i max(0, ln(Σ_ii/Σ*_ii)). Per-DOF clamp (not the full
    // ½ln detΣ/detΣ*) on purpose — the consumer needs EACH of w,h,yaw within tolerance, so an over-resolved
    // DOF must NOT compensate for an under-resolved one (the full log-det lets a sub-mm position mask an
    // unresolved 45° yaw). ≤0 on every DOF ⇒ adequate ⇒ no epistemic value left ⇒ the affordance goes quiet
    // and the controller moves on — a threshold-free "done" set by the CONSUMER's precision demand, not a
    // tuned Σ bound. A mode-ambiguous table (marginal σyaw≈45°) stays inadequate on the yaw DOF until the
    // mode resolves, so the robot keeps gathering evidence on it before releasing.
    float adequacy_gap = 0.0f;
    for (int j = 0; j < 6; ++j)
        adequacy_gap += std::max(0.0f, 0.5f * std::log(S(j, j) / (kTargetStd[j] * kTargetStd[j])));

    // Score each face by the D-optimal expected entropy reduction on Σ, with a range-aware R per face.
    const Eigen::Matrix<float, 6, 6> I6 = Eigen::Matrix<float, 6, 6>::Identity();
    int   best_idx      = 0;
    float best_raw      = -std::numeric_limits<float>::max();
    float best_standoff = kMinimumStandOffM;
    // RAW per-face D-optimal gain (nats), UNBOUNDED — this is the RANKING the controller uses to choose a
    // feasible face, so it must NOT be adequacy-clamped. Each face's single-view info typically EXCEEDS the
    // remaining gap, so clamping every face at the gap made all four tie (the degenerate ranked-set that
    // erased the controller's basis to prefer one face). The adequacy bound belongs on the SCALAR value below.
    std::array<float, 4> face_raw{};
    for (int i = 0; i < 4; ++i)
    {
        const float standoff = standoff_for(faces[i].half_span);
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const Eigen::Matrix<float, 6, 6> dI = belief.predicted_information(sample_face_surface(s, i), Ri).topLeftCorner<6, 6>();
        const float det  = (I6 + S * dI).determinant();
        const float raw_gain = std::max(0.0f, 0.5f * std::log(std::max(1e-9f, det)));   // single-view D-optimal info (nats)
        face_raw[i] = raw_gain;
        if (raw_gain > best_raw) { best_raw = raw_gain; best_idx = i; best_standoff = standoff; }
    }
    if (!std::isfinite(best_raw))   // low-but-finite gain is NOT withdrawn (controller ranks it low)
        return {};

    // SCALAR affordance value (epistemic_gain): the winning face's info BOUNDED by the adequacy gap to Σ*, so
    // it → 0 as the belief reaches the consumer's precision — a threshold-free "done" AND the cross-affordance
    // EFE currency. Information beyond Σ* is worthless to the consumer, so an adequate table stops attracting.
    // p_observable hook = 1.0 (the standoff is a framed viewpoint by construction); wire reachability later.
    constexpr float p_observable = 1.0f;
    const float best_gain = p_observable * std::min(best_raw, adequacy_gap);

    // Verification readout (throttled): the chosen face should be perpendicular to Σ's dominant
    // uncertainty direction — e.g. dom-unc = h ⇒ a ±y face wins. Confirms the NBV attacks the worst DOF.
    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        // Dominant-uncertainty DOF, normalised by a common scale (10 cm / 0.1 rad) so metres and radians
        // are comparable — "which DOF is least known relative to its natural scale". (The gain decision
        // itself uses the unit-invariant log-det; this is only a readable diagnostic label.)
        static const float  ref[6] = {0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f};  // cx,cy,H,w,h (m); yaw (rad)
        static const char*  dof[6] = {"cx", "cy", "H", "w", "h", "yaw"};
        static const char*  fn[4]  = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < 6; ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / ref[j];
            if (n > best) { best = n; dom = j; }
        }
        std::print("[epistemic-NBV] face={} gain={:.3f}(raw {:.3f}) adq_gap={:.3f} | Σ dom-unc={} σ={:.3f}{} | raw +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f}\n",
                   fn[best_idx], best_gain, best_raw, adequacy_gap, dof[dom], std::sqrt(std::max(0.0f, S(dom, dom))), (dom == 5 ? "rad" : "m"),
                   face_raw[0], face_raw[1], face_raw[2], face_raw[3]);
    }

    const auto& f = faces[best_idx];
    const float vx = f.centre.x() + f.normal.x() * best_standoff;
    const float vy = f.centre.y() + f.normal.y() * best_standoff;
    const float yaw_to_face = std::atan2(s.cy - vy, s.cx - vx);

    EpistemicProposal proposal{vx, vy, yaw_to_face, best_gain, true};
    // Object-relative viewpoint constraint (authoritative): publish ALL four faces with their RAW per-face
    // gains (the ranking — NOT adequacy-clamped, else they tie) + the sensor-model stand-off band + framing
    // + Σ*, so the controller picks the best FEASIBLE face itself (a blocked argmax face falls back to the
    // next reachable one). The affordance's scalar epistemic_gain stays adequacy-bounded (cross-affordance
    // selection + "done"). Face order matches the [+x,-x,+y,-y] sampling order above.
    constexpr float kFramingFill = 0.45f;   // FoV framing sweet spot (matches the table servo contract's advance target)
    proposal.face_gains    = face_raw;
    proposal.standoff_min_m = kMinimumStandOffM;
    proposal.standoff_max_m = max_stand_off;
    proposal.framing_fill   = kFramingFill;
    proposal.sigma_star     = kTargetStd;
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
