/*
 * epistemic_planner.cpp  —  Σ-based D-optimal next-best-view over the refrigerator's vertical faces.
 *
 * Only the four vertical faces (±X, ±Y in refrigerator frame) are considered: a floor-navigating robot cannot
 * observe the top face from above or the bottom face at all. Each face is scored by the expected entropy
 * reduction ½·ln det(I₆ + Σ·ΔI) on the belief's full covariance, bounded by the adequacy gap to Σ*, and * the winning face's framed viewpoint + heading is returned. See epistemic_planner.h for the formulas.
 */

#include "epistemic_planner.h"
#include "refrigerator_dof.h"   // kRefrigeratorDofs: names/units/σ* — shared with the dashboard

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
// Samples surface points on one vertical face of the 6-DOF RefrigeratorBeliefState [cx,cy,H,w,h,yaw], for ΔI.
std::vector<Eigen::Vector3f> sample_face_surface(const RefrigeratorBeliefState& s, int face_idx)
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
EpistemicProposal EpistemicPlanner::compute(const RefrigeratorBelief& belief, float lat_rate, float sigma_base,
                                            float hfov_rad, const std::vector<EpistemicPlanner::Obstacle>& obstacles) const
{
    // HORIZONTAL FoV only, from the camera's REAL intrinsics when the caller supplies them (2·atan(W/2fx));
    // the 70° literal is only the fallback for a caller that has no CameraAPI yet.
    const float hfov = (hfov_rad > 0.1f and hfov_rad < 3.0f)
                     ? hfov_rad : (70.0f * std::numbers::pi_v<float> / 180.0f);
    constexpr float kRobotRadiusM = 0.30f;   // Shadow's footprint radius — a physical dimension, not a knob
    // Fraction of the half-FoV the object should subtend: 0.45 leaves ~55% of the half-frame as margin, so the
    // silhouette stays whole and trunc_frac stays low. Same constant the servo contract already publishes.
    constexpr float kFramingFill = 0.45f;

    // REPORTED covariance: Σ with the yaw entry inflated by the discrete-mode entropy (p(1−p)(π/2)²), so a
    // near-square refrigerator whose orientation mode is unresolved shows a large yaw variance → the D-optimal NBV
    // scores a mode-discriminating (side/leg) view highly and drives the orbit that resolves it.
    const Eigen::Matrix<float, 6, 6> S = belief.covariance_reported().topLeftCorner<6, 6>();   // [cx,cy,H,w,h,yaw] (drop the tilt-calibration DOF)
    const RefrigeratorBeliefState& s = belief.state();

    const float cy = std::cos(s.yaw), sy = std::sin(s.yaw);
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    struct Face { Eigen::Vector2f normal, centre; float half_span; };
    const std::array<Face, 4> faces = {{
        { { cy,  sy}, {s.cx + cy * hw, s.cy + sy * hw}, hh },   // +x
        { {-cy, -sy}, {s.cx - cy * hw, s.cy - sy * hw}, hh },   // -x
        { {-sy,  cy}, {s.cx - sy * hh, s.cy + cy * hh}, hw },   // +y
        { { sy, -cy}, {s.cx + sy * hh, s.cy - cy * hh}, hw },   // -y
    }};
    // ── Stand-off: DERIVED framing distance, not a constant ────────────────────────────────────────────
    // The old rule was half_span/tan(hfov/2) + 0.45, clamped into [1.15, d_obs]. Two faults: the "+0.45"
    // margin was arbitrary, and half_span/tan(hfov/2) is the distance at which the face EXACTLY FILLS the
    // frame width — the framing at which YOLO's mask is clipped by the image border, `trunc_frac` rises and
    // the frame stops being admissible. For a 0.6 m fridge that evaluated to 0.89 m and then hit the 1.15 m
    // floor, i.e. the floor was doing all the work and the geometry none. That is why standing close was
    // producing truncated masks. See [[refrigerator-standoff-and-obstacles]].
    //
    // Instead: place the object so its horizontal extent occupies `fill` of the half-FoV, leaving (1−fill) as
    // image margin, which is what keeps the silhouette whole:
    //     d = R / tan(fill · hfov/2)
    // R is the footprint's CIRCUMSCRIBED radius ½√(w²+h²), not the viewed face's half-width — the robot
    // approaches from an arbitrary bearing, so the projected width can reach the diagonal. Using R makes the
    // stand-off valid from every approach direction instead of only head-on.
    const float R_circ = 0.5f * std::sqrt(s.w * s.w + s.h * s.h);
    // Collision floor: the robot cannot stand closer than its own radius plus the object's. Geometry, not tuning.
    const float collision_floor = R_circ + kRobotRadiusM;
    const float framing_d = R_circ / std::max(1e-3f, std::tan(kFramingFill * 0.5f * hfov));
    const float framed_standoff = std::max(framing_d, collision_floor);
    // d_obs is a PREFERENCE for staying close enough to orbit; it must never pull the robot inside the
    // framing distance, or we are back to truncated masks.
    const float max_stand_off = std::max(framed_standoff, d_obs_);
    const auto standoff_for = [&](float) { return framed_standoff; };

    // ── Adequacy gap (active-perception step 1, REFRIGERATOR.md) ─────────────────────────────────
    // Target precision Σ* and the gap formula now live in refrigerator_dof.h /
    // common/ai_belief/dof_spec.h — ONE table shared with the dashboard, so the demands the planner acts
    // on are exactly the ones the BeliefInspector shows. Same values, same per-DOF clamp, same result.
    // A mode-ambiguous refrigerator (marginal σyaw≈45°) stays inadequate on the yaw DOF until the mode
    // resolves, so the robot keeps gathering evidence on it before releasing.
    const float adequacy_gap = adequacy_gap_nats(kRefrigeratorDofs, [&](std::size_t j) { return S(j, j); });

    // ── Would this viewpoint stand ON another object? ───────────────────────────────────────────────
    // The obstacles are every OTHER object the DSR graph knows about, already inflated by the robot radius by
    // the caller, so the test is a plain point-in-oriented-rectangle. A face whose framed viewpoint lands
    // inside one is skipped and the next-best face is taken — the ranking already exists for exactly this
    // fallback, it was simply never consulted. If EVERY face is blocked we keep the argmax anyway and let the
    // controller resolve it: this planner proposes, it does not own global occupancy.
    const auto blocked = [&](float vx, float vy)
    {
        for (const auto& o : obstacles)
        {
            const float c = std::cos(-o.yaw), sn = std::sin(-o.yaw);
            const float dx = vx - o.cx, dy = vy - o.cy;
            const float lx = c * dx - sn * dy, ly = sn * dx + c * dy;   // into the obstacle's own frame
            if (std::abs(lx) <= 0.5f * o.w and std::abs(ly) <= 0.5f * o.h)
                return true;
        }
        return false;
    };
    const auto viewpoint_of = [&](int i, float standoff)
    {
        return Eigen::Vector2f{faces[i].centre.x() + faces[i].normal.x() * standoff,
                               faces[i].centre.y() + faces[i].normal.y() * standoff};
    };

    // Score each face by the D-optimal expected entropy reduction on Σ, with a range-aware R per face.
    const Eigen::Matrix<float, 6, 6> I6 = Eigen::Matrix<float, 6, 6>::Identity();
    int   best_idx      = 0;
    float best_raw      = -std::numeric_limits<float>::max();
    float best_standoff = framed_standoff;
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
    if (not std::isfinite(best_raw))   // low-but-finite gain is NOT withdrawn (controller ranks it low)
        return {};

    // Re-pick the highest-gain face whose framed viewpoint is NOT inside another object.
    int   free_idx = -1;
    float free_raw = -std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i)
    {
        const auto v = viewpoint_of(i, framed_standoff);
        if (blocked(v.x(), v.y())) continue;
        if (face_raw[i] > free_raw) { free_raw = face_raw[i]; free_idx = i; }
    }
    if (free_idx >= 0 and free_idx != best_idx)
    {
        std::print("[epistemic-NBV] argmax face blocked by another object → falling back (gain {:.3f} → {:.3f})\n",
                   best_raw, free_raw);
        best_idx = free_idx; best_raw = free_raw;
    }
    else if (free_idx < 0 and not obstacles.empty())
        std::print("[epistemic-NBV] ALL four viewpoints blocked by other objects — keeping argmax, controller must resolve\n");

    // SCALAR affordance value (epistemic_gain): the winning face's info BOUNDED by the adequacy gap to Σ*, so
    // it → 0 as the belief reaches the consumer's precision — a threshold-free "done" AND the cross-affordance
    // EFE currency. Information beyond Σ* is worthless to the consumer, so an adequate refrigerator stops attracting.
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
        constexpr float kRef = 0.10f;   // common scale: 10 cm / 0.1 rad
        static const char* fn[4] = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        std::print("[epistemic-NBV] face={} gain={:.3f}(raw {:.3f}) adq_gap={:.3f} | Σ dom-unc={} σ={:.3f}{} | raw +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f}\n",
                   fn[best_idx], best_gain, best_raw, adequacy_gap, kRefrigeratorDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kRefrigeratorDofs[dom].unit,
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
    proposal.face_gains    = face_raw;
    proposal.standoff_min_m = collision_floor;
    proposal.standoff_max_m = max_stand_off;
    proposal.framing_fill   = kFramingFill;
    proposal.sigma_star     = sigma_star_array<6>(kRefrigeratorDofs);   // the SAME demands the adequacy gap used
    if (not proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
