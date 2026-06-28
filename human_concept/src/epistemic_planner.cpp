/*
 * epistemic_planner.cpp — reduce-occlusion next-best-view for human_concept.
 */

#include "epistemic_planner.h"

#include <algorithm>
#include <cmath>

namespace rc {

EpistemicPlanner::EpistemicPlanner(float d_obs, float view_info)
    : d_obs_(d_obs), view_info_(view_info)
{}

EpistemicProposal EpistemicPlanner::compute(const Eigen::Vector2f& person_xy,
                                            const Eigen::Vector2f& camera_xy,
                                            float worst_info) const
{
    EpistemicProposal p;

    const Eigen::Vector2f ray = person_xy - camera_xy;          // camera → person (horizontal)
    if (not std::isfinite(ray.x()) or not std::isfinite(ray.y()) or ray.norm() < 1e-3f)
        return p;                                                // degenerate: can't resolve the hidden side

    const Eigen::Vector2f dir    = ray.normalized();
    const Eigen::Vector2f target = person_xy + dir * d_obs_;     // far side, beyond the person
    p.epistemic_target_x_m     = target.x();
    p.epistemic_target_y_m     = target.y();
    p.epistemic_target_yaw_rad = std::atan2(person_xy.y() - target.y(),
                                            person_xy.x() - target.x());   // look back at the person

    // Expected entropy reduction in the worst-constrained DOF: ΔH = ½·log(1 + I_view/Y_worst).
    // ΔH→0 as the skeleton is well-observed (Y_worst grows), so a well-seen person yields a low gain.
    const float Y_worst = std::max(worst_info, 1e-3f);
    p.epistemic_gain = 0.5f * std::log(1.0f + view_info_ / Y_worst);

    p.valid = p.is_finite();
    return p;
}

}  // namespace rc
