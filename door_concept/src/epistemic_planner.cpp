/*
 * epistemic_planner.cpp
 *
 * AI2-native Σ-based D-optimal next-best-view for the door-concept agent.
 *
 * Only the four vertical faces (±X, ±Y in door frame) are considered because
 * a floor-navigating robot cannot observe the top face from above or the
 * bottom face at all.
 */

#include "epistemic_planner.h"
#include "door_dof.h"          // kDoorDofs: names/units (no σ* published for the door)

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

// Face sampler: synthetic samples on one vertical face of the LEAF (floor → door top), room frame.
// Geometry comes from rc::door::LeafPose (door_geometry.h), so the samples sit on the panel where it
// PHYSICALLY IS. That matters the moment phi is fitted: sampling the wall plane instead would evaluate
// predicted_information at points that are no longer on any surface, and rank the faces against a panel
// that isn't there. At phi = 0 the leaf is flush in the aperture and these are the same points as before.
std::vector<Eigen::Vector3f> sample_face_surface(const door::LeafPose& L, float floor_z, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const float hw = L.half_w, hh = L.half_t;
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
        const Eigen::Vector2f r = L.centre_xy + lx * L.ex + ly * L.ey;
        const float top = 2.0f * L.half_h;
        for (int k = 0; k < kVert; ++k)
        {
            const float z = floor_z + top * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(r.x(), r.y(), z);
        }
    }
    return pts;
}

}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// ── AI2-native Σ-based D-optimal next-best-view ──────────────────────────────────────────────
EpistemicProposal EpistemicPlanner::compute(const DoorBelief& belief, float lat_rate, float sigma_base) const
{
    constexpr float kEffectiveHorizontalFovRad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    constexpr float kStandOffSafetyMarginM = 0.45f;    // extra stand-off beyond the FoV-fit distance

    // Σ over the wall-frame [s,w,h]: the D-optimal score ranks which viewpoint most reduces the panel's
    // along-wall offset + size uncertainty. (No orientation entropy — the wall fixes yaw.)
    const Eigen::Matrix<float, 3, 3> S = belief.covariance_reported();   // Σ over [s,w,h]
    // Faces of the LEAF, from its actual axes — NOT reconstructed from the wall tangent. The old code
    // took its normals from belief.yaw(), which is the wall's, so once a leaf swings it would send the
    // robot to stand square to the WALL rather than to the panel it is trying to observe.
    const door::LeafPose L = belief.leaf_pose();
    const Eigen::Vector2f ctr = L.centre_xy;
    const float hw = L.half_w, hh = L.half_t;
    struct Face { Eigen::Vector2f normal, centre; float half_span; };
    const std::array<Face, 4> faces = {{
        {  L.ex, Eigen::Vector2f(ctr + hw * L.ex), hh },   // +x (free-edge side)
        { -L.ex, Eigen::Vector2f(ctr - hw * L.ex), hh },   // -x (hinge side)
        {  L.ey, Eigen::Vector2f(ctr + hh * L.ey), hw },   // +y (front panel face)
        { -L.ey, Eigen::Vector2f(ctr - hh * L.ey), hw },   // -y (back panel face)
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
        const auto  dI = belief.predicted_information(sample_face_surface(L, belief.cz(), i), Ri);
        const float gain = 0.5f * std::log(std::max(1e-9f, (I3 + S * dI).determinant()));
        face_gain[i] = gain;
        if (gain > best_gain) { best_gain = gain; best_idx = i; best_standoff = standoff; }
    }
    if (!std::isfinite(best_gain))
        return {};

    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        constexpr float kRef = 0.10f;   // common scale: 10 cm / 0.1 rad
        static const char* fn[4] = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        std::print("[epistemic-NBV] face={} gain={:.3f} | Σ dom-unc={} σ={:.3f}{} | gains +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f}\n",
                   fn[best_idx], best_gain, kDoorDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kDoorDofs[dom].unit,
                   face_gain[0], face_gain[1], face_gain[2], face_gain[3]);
    }

    const auto& f = faces[best_idx];
    const float vx = f.centre.x() + f.normal.x() * best_standoff;
    const float vy = f.centre.y() + f.normal.y() * best_standoff;
    const float yaw_to_face = std::atan2(ctr.y() - vy, ctr.x() - vx);
    EpistemicProposal proposal{vx, vy, yaw_to_face, best_gain, true};
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
