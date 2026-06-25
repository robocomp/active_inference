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

EpistemicPlanner::EpistemicPlanner(float delta_min, float gain_threshold, float d_obs,
                                   bool use_info_gain, float min_info_gain)
    : delta_min_(delta_min), gain_threshold_(gain_threshold), d_obs_(d_obs),
      use_info_gain_(use_info_gain), min_info_gain_(min_info_gain)
{}

EpistemicProposal EpistemicPlanner::compute(const TableModel&  model,
                                             const SampleQueue& queue,
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
    // reduction ΔH from the Fisher posterior (information-seeking), or the legacy coverage-deficit
    // proxy. P(detect|v) would scale ΔH here — kept at 1.0 for now (hook for a camera-projection
    // detectability model; see [[affordance-contract-efe-selection]]).
    const float sigma2_proxy = model.params().sigma_obs * model.params().sigma_obs;
    int   best_idx   = 0;
    float best_gain  = -std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i)
    {
        float gain;
        if (use_info_gain_)
        {
            const auto I_pred = model.observation_information(sample_face_surface(s, i), {});
            const float p_detect = 1.0f;
            gain = p_detect * expected_info_gain(I_pred, posterior_info);
        }
        else
        {
            const float face_area = 2.0f * faces[i].half_span * s.table_height;
            const float deficit   = 1.0f - std::min(1.0f, faces[i].cov / delta_min_);
            if (!std::isfinite(sigma2_proxy) || sigma2_proxy <= std::numeric_limits<float>::epsilon())
                return {};
            gain = face_area * deficit / sigma2_proxy;
        }
        if (gain > best_gain)
        {
            best_gain = gain;
            best_idx  = i;
        }
    }

    // Below threshold ⇒ nothing worth observing (ΔH→0 = belief has become knowledge).
    const float threshold = use_info_gain_ ? min_info_gain_ : gain_threshold_;
    if (!std::isfinite(best_gain) || best_gain < threshold)
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

}  // namespace rc
