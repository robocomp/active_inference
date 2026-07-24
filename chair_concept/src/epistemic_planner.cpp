/*
 * epistemic_planner.cpp
 *
 * AI2-native Σ-based D-optimal next-best-view for the chair-concept agent.
 *
 * Only the four vertical faces (±X, ±Y in chair frame) are considered because
 * a floor-navigating robot cannot observe the top face from above or the
 * bottom face at all.
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

// AI2 belief-state face sampler: synthetic samples on a vertical face plane (face f, floor → chair
// top), room frame. Dims come from the fixed template; only the pose (cx,cy,yaw) is a DOF.
std::vector<Eigen::Vector3f> sample_face_surface(const ChairBelief& belief, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const auto& s = belief.state();
    const float hw = belief.seat_w() * 0.5f, hh = belief.seat_d() * 0.5f;
    const float cyaw = std::cos(s.yaw), syaw = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts; pts.reserve(kTangent * kVert);
    for (int m = 0; m < kTangent; ++m)
    {
        const float t = (kTangent > 1) ? (-1.0f + 2.0f * m / (kTangent - 1)) : 0.0f;
        float lx = 0.0f, ly = 0.0f;
        switch (face_idx)
        {
            case 0: lx =  hw;     ly =  t * hh; break;
            case 1: lx = -hw;     ly =  t * hh; break;
            case 2: lx =  t * hw; ly =  hh;     break;
            default: lx = t * hw; ly = -hh;     break;
        }
        const float rx = s.cx + cyaw * lx - syaw * ly;
        const float ry = s.cy + syaw * lx + cyaw * ly;
        const float top = belief.seat_h() + belief.back_h();
        for (int k = 0; k < kVert; ++k)
        {
            const float z = belief.cz() + top * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(rx, ry, z);
        }
    }
    return pts;
}

}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// ── AI2-native Σ-based D-optimal next-best-view ──────────────────────────────────────────────
EpistemicProposal EpistemicPlanner::compute(const ChairBelief& belief, float lat_rate, float sigma_base) const
{
    constexpr float kEffectiveHorizontalFovRad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    constexpr float kStandOffSafetyMarginM = 0.45f;    // extra stand-off beyond the FoV-fit distance

    // REPORTED Σ: the yaw term folds in the discrete orientation-mode entropy, so a mode-ambiguous (side-on)
    // chair scores a backrest-revealing face highly and the orbit resolves the ambiguity. (Raw Σ would hide it.)
    const Eigen::Matrix<float, 3, 3> S = belief.covariance_reported();   // Σ over [cx,cy,yaw]
    const ChairBeliefState& s = belief.state();

    const float cy = std::cos(s.yaw), sy = std::sin(s.yaw);
    const float hw = belief.seat_w() * 0.5f, hh = belief.seat_d() * 0.5f;
    struct Face { Eigen::Vector2f normal, centre; float half_span; };
    const std::array<Face, 4> faces = {{
        { { cy,  sy}, {s.cx + cy * hw, s.cy + sy * hw}, hh },   // +x
        { {-cy, -sy}, {s.cx - cy * hw, s.cy - sy * hw}, hh },   // -x
        { {-sy,  cy}, {s.cx - sy * hh, s.cy + cy * hh}, hw },   // +y
        { { sy, -cy}, {s.cx + sy * hh, s.cy - cy * hh}, hw },   // -y
    }};
    const float max_stand_off = std::max(min_standoff_, d_obs_);
    const auto standoff_for = [&](float half_span)
    { return std::clamp(half_span / std::tan(kEffectiveHorizontalFovRad * 0.5f) + kStandOffSafetyMarginM,
                        min_standoff_, max_stand_off); };

    const Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    int   best_idx = 0;
    float best_gain = -std::numeric_limits<float>::max();
    float best_standoff = min_standoff_;
    std::array<float, 4> face_gain{};
    for (int i = 0; i < 4; ++i)
    {
        const float standoff = standoff_for(faces[i].half_span);
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const auto  dI = belief.predicted_information(sample_face_surface(belief, i), Ri);
        const float gain = 0.5f * std::log(std::max(1e-9f, (I3 + S * dI).determinant()));
        face_gain[i] = gain;
        if (gain > best_gain) { best_gain = gain; best_idx = i; best_standoff = standoff; }
    }
    if (!std::isfinite(best_gain))
        return {};

    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        static const float ref[3] = {0.10f, 0.10f, 0.10f};   // cx,cy (m); yaw (rad)
        static const char* dof[3] = {"cx", "cy", "yaw"};
        static const char* fn[4]  = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < 3; ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / ref[j];
            if (n > best) { best = n; dom = j; }
        }
        std::print("[epistemic-NBV] face={} gain={:.3f} | Σ dom-unc={} σ={:.3f}{} | gains +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f}\n",
                   fn[best_idx], best_gain, dof[dom], std::sqrt(std::max(0.0f, S(dom, dom))), (dom == 2 ? "rad" : "m"),
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
