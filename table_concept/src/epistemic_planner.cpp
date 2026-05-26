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

#include <array>
#include <cmath>
#include <numbers>

// ─── EpistemicPlanner ────────────────────────────────────────────────────────

EpistemicPlanner::EpistemicPlanner(float delta_min, float gain_threshold, float d_obs)
    : delta_min_(delta_min), gain_threshold_(gain_threshold), d_obs_(d_obs)
{}

EpistemicProposal EpistemicPlanner::compute(const TableModel&  model,
                                             const SampleQueue& queue) const
{
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

    // Find the face with lowest coverage
    int   best_idx  = 0;
    float best_cov  = faces[0].cov;
    for (int i = 1; i < 4; ++i)
    {
        if (faces[i].cov < best_cov)
        {
            best_cov  = faces[i].cov;
            best_idx  = i;
        }
    }

    // If coverage is already sufficient on the best candidate, nothing to do
    if (best_cov >= delta_min_)
        return {};

    const auto& f = faces[best_idx];

    // Viewpoint: face centre + outward normal × d_obs
    const float vx = f.centre.x() + f.normal.x() * d_obs_;
    const float vy = f.centre.y() + f.normal.y() * d_obs_;

    // Heading: face the table (inverse of outward normal)
    const float yaw_to_face = std::atan2(-f.normal.y(), -f.normal.x());

    // Gain proxy: face area × coverage deficit / σ²
    const float face_area   = 2.0f * f.half_span * s.table_height;
    const float deficit     = 1.0f - std::min(1.0f, best_cov / delta_min_);
    const float sigma2      = model.params().sigma_obs * model.params().sigma_obs;
    const float gain        = face_area * deficit / sigma2;

    if (gain < gain_threshold_)
        return {};

    return EpistemicProposal{vx, vy, yaw_to_face, gain, true};
}
