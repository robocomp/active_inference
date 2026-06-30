/*
 * epistemic_planner.cpp
 *
 * Computes optimal viewpoint for the least-observed vertical table face.
 *
 * Only the four vertical faces (±X, ±Y in table frame) are considered because
 * a floor-navigating robot cannot observe the top face from above or the
 * bottom face at all.
 *
 * The gain proxy is:  ΔH = face_face_area / σ_obs²  (expected Fisher info)
 * scaled by the coverage deficit (1 − coverage / δ_min).
 */

#include "epistemic_planner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <vector>

// ─── EpistemicPlanner ────────────────────────────────────────────────────────


namespace rc {

namespace
{

// Synthetic samples on a vertical face plane (face f, floor → table-top), room frame. The Fisher
// evaluation robustly down-weights samples that miss solid surface (open air between legs), so this
// effectively predicts the information from the part of the face that is actually there to be seen.
std::vector<Eigen::Vector3f> sample_face_surface(const TableState& s, int face_idx)
{
    constexpr int kTangent = 10;   // samples along the face edge
    constexpr int kVert    = 6;    // samples from floor to table top
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    const float cyaw = std::cos(s.yaw), syaw = std::sin(s.yaw);

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(kTangent * kVert);
    for (int m = 0; m < kTangent; ++m)
    {
        const float t = (kTangent > 1) ? (-1.0f + 2.0f * m / (kTangent - 1)) : 0.0f;   // [-1,1]
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
            const float z = s.table_height * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(rx, ry, z);
        }
    }
    return pts;
}

// AI2 belief-state face sampler (same geometry as above, off the 6-DOF TableBeliefState [cx,cy,H,w,h,yaw]).
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

// Expected entropy reduction ΔH = Σ_j ½·log(1 + I_pred_j / Y_j), nats. Y_j is floored at a prior
// precision so a never-observed DOF gives a large-but-finite gain rather than ∞.
float expected_info_gain(const std::array<float, 8>& I_pred, const std::array<float, 8>& Y)
{
    constexpr float kPriorInfoFloor = 4.0f;   // ≈ prior std 0.5 (DOF units): avoids div-by-zero
    float dH = 0.0f;
    for (int j = 0; j < 8; ++j)
    {
        const float ratio = I_pred[j] / std::max(Y[j], kPriorInfoFloor);
        if (ratio > 0.0f)
            dH += 0.5f * std::log1p(ratio);
    }
    return dH;
}

}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

EpistemicProposal EpistemicPlanner::compute(const TableModel&  model,
                                             const SampleQueue<TableModel>& queue,
                                             const std::array<float, 8>& posterior_info) const
{
    constexpr float kEffectiveHorizontalFovRad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    constexpr float kMinimumStandOffM = 0.90f;
    constexpr float kStandOffSafetyMarginM = 0.25f;

    const auto coverage = queue.face_coverage(model);
    const auto& s       = model.state();

    // Four vertical face normals in table frame: +x, -x, +y, -y  (indices 0-3)
    // In room frame the normals are rotated by yaw
    const float cy = std::cos(s.yaw);
    const float sy = std::sin(s.yaw);

    // Outward normals (room frame) and face centres (room frame)
    struct Face
    {
        Eigen::Vector2f normal;    // Unit outward normal in room XY
        Eigen::Vector2f centre;    // Face centre in room XY
        float           half_span; // Half-length of face edge (for area)
        float           cov;       // Coverage weight sum
    };

    const float hw = s.w * 0.5f;
    const float hh = s.h * 0.5f;

    // Rotate local (±1,0) and (0,±1) by yaw into room frame
    //   +x face: local normal (+1,0) → room (cy, sy)
    //   -x face: local normal (-1,0) → room (-cy, -sy)
    //   +y face: local normal (0,+1) → room (-sy, cy)
    //   -y face: local normal (0,-1) → room (sy, -cy)
    std::array<Face, 4> faces = {{
        // +x
        { { cy,  sy}, {s.cx + cy*hw - sy*0.0f, s.cy + sy*hw + cy*0.0f}, hh, coverage[0] },
        // -x
        { {-cy, -sy}, {s.cx - cy*hw,            s.cy - sy*hw           }, hh, coverage[1] },
        // +y
        { {-sy,  cy}, {s.cx - sy*hh,            s.cy + cy*hh           }, hw, coverage[2] },
        // -y
        { { sy, -cy}, {s.cx + sy*hh,            s.cy - cy*hh           }, hw, coverage[3] },
    }};

    // Score each face and keep the most informative one. The score is either the expected entropy
    // reduction ΔH from the Fisher posterior (information-seeking). P(detect|v) would scale ΔH here —
    // kept at 1.0 for now (hook for a camera-projection detectability model; see
    // [[affordance-contract-efe-selection]]).
    int   best_idx   = 0;
    float best_gain  = -std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i)
    {
        const auto I_pred = model.observation_information(sample_face_surface(s, i), {});
        const float p_detect = 1.0f;
        const float gain = p_detect * expected_info_gain(I_pred, posterior_info);
        if (gain > best_gain)
        {
            best_gain = gain;
            best_idx  = i;
        }
    }

    // Reject only a degenerate (non-finite) score. A LOW but finite ΔH is NOT withdrawn here: the
    // planner keeps returning the best-face proposal carrying its true gain so the affordance node
    // persists and refreshes, and the controller's EFE selection simply doesn't pick a low-nat target
    // (the belief→knowledge governor, expressed as a small gain rather than a deleted node).
    if (!std::isfinite(best_gain))
        return {};

    const auto& f = faces[best_idx];

    const float face_fit_distance = f.half_span /
        std::tan(kEffectiveHorizontalFovRad * 0.5f);
    const float max_stand_off = std::max(kMinimumStandOffM, d_obs_);
    const float stand_off = std::clamp(face_fit_distance + kStandOffSafetyMarginM,
                                       kMinimumStandOffM,
                                       max_stand_off);

    // Place the robot just outside the selected face, close enough to orbit the
    // table but far enough for the face to remain comfortably inside the ZED FOV.
    const float vx = f.centre.x() + f.normal.x() * stand_off;
    const float vy = f.centre.y() + f.normal.y() * stand_off;

    // Heading: point the robot at the table centre so the camera sees the full
    // table body rather than grazing only the selected face point.
    const float yaw_to_face = std::atan2(s.cy - vy, s.cx - vx);

    EpistemicProposal proposal{vx, vy, yaw_to_face, best_gain, true};
    if (!proposal.is_finite())
        return {};

    return proposal;
}

// ── AI2-native Σ-based D-optimal next-best-view ──────────────────────────────────────────────
EpistemicProposal EpistemicPlanner::compute(const TableBelief& belief, float lat_rate, float sigma_base) const
{
    constexpr float kEffectiveHorizontalFovRad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    constexpr float kMinimumStandOffM = 0.90f;
    constexpr float kStandOffSafetyMarginM = 0.25f;

    const Eigen::Matrix<float, 6, 6>& S = belief.covariance();   // full Σ over [cx,cy,H,w,h,yaw]
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

    // Score each face by the D-optimal expected entropy reduction on Σ, with a range-aware R per face.
    const Eigen::Matrix<float, 6, 6> I6 = Eigen::Matrix<float, 6, 6>::Identity();
    int   best_idx      = 0;
    float best_gain     = -std::numeric_limits<float>::max();
    float best_standoff = kMinimumStandOffM;
    std::array<float, 4> face_gain{};
    for (int i = 0; i < 4; ++i)
    {
        const float standoff = standoff_for(faces[i].half_span);
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const auto  dI = belief.predicted_information(sample_face_surface(s, i), Ri);
        const float det  = (I6 + S * dI).determinant();
        const float gain = 0.5f * std::log(std::max(1e-9f, det));   // P(detect|v) hook = 1.0 for now
        face_gain[i] = gain;
        if (gain > best_gain) { best_gain = gain; best_idx = i; best_standoff = standoff; }
    }
    if (!std::isfinite(best_gain))   // low-but-finite gain is NOT withdrawn (controller ranks it low)
        return {};

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
        std::print("[epistemic-NBV] face={} gain={:.3f} | Σ dom-unc={} σ={:.3f}{} | gains +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f}\n",
                   fn[best_idx], best_gain, dof[dom], std::sqrt(std::max(0.0f, S(dom, dom))), (dom == 5 ? "rad" : "m"),
                   face_gain[0], face_gain[1], face_gain[2], face_gain[3]);
    }

    const auto& f = faces[best_idx];
    const float vx = f.centre.x() + f.normal.x() * best_standoff;
    const float vy = f.centre.y() + f.normal.y() * best_standoff;
    const float yaw_to_face = std::atan2(s.cy - vy, s.cx - vx);

    EpistemicProposal proposal{vx, vy, yaw_to_face, best_gain, true};
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
