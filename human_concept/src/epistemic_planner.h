/*
 * epistemic_planner.h
 *
 * Epistemic action proposal for human_concept: send the robot to a viewpoint that REDUCES OCCLUSION
 * of the tracked person, so the pose estimator can constrain its worst-seen DOFs (occluded limbs).
 * This is the prototype's "change viewpoint / reduce occlusion / move closer" action hint made into a
 * concrete next-best-view + an information gain the controller's EFE selection can rank:
 *
 *   dir    = normalize(person_xy − camera_xy)         (camera → person, horizontal)
 *   v*     = person_xy + dir · d_obs                  (stand d_obs beyond the person, far side)
 *   heading= atan2(person_y − v.y, person_x − v.x)    (look back at the person)
 *   ΔH     = ½·log(1 + I_view / Y_worst)              (entropy reduction in the worst-constrained DOF)
 *
 * Y_worst is the smallest accumulated posterior precision across the 11 angle DOFs (the most uncertain
 * joint), so ΔH→0 as the whole skeleton is well-observed and a well-seen person publishes a low gain
 * the controller won't pick.
 */

#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace rc {

struct EpistemicProposal
{
    float epistemic_target_x_m     = 0.0f;
    float epistemic_target_y_m     = 0.0f;
    float epistemic_target_yaw_rad = 0.0f;
    float epistemic_gain           = 0.0f;   // expected information gain ΔH (nats)
    bool  valid                    = false;

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) &&
               std::isfinite(epistemic_target_y_m) &&
               std::isfinite(epistemic_target_yaw_rad) &&
               std::isfinite(epistemic_gain);
    }
};

class EpistemicPlanner
{
public:
    /**
     * @param d_obs      Stand-off from the person at the look viewpoint (m).
     * @param view_info  Fisher precision a clearer view is expected to add to the worst DOF (ΔH scale).
     */
    explicit EpistemicPlanner(float d_obs = 1.5f, float view_info = 50.0f);

    /**
     * @param person_xy   Person root (pelvis) in the room frame, horizontal.
     * @param camera_xy   ZED origin in the room frame, horizontal.
     * @param worst_info  Smallest accumulated posterior precision across DOFs (the most uncertain joint).
     * Returns valid==false only if the camera→person ray is degenerate.
     */
    EpistemicProposal compute(const Eigen::Vector2f& person_xy,
                              const Eigen::Vector2f& camera_xy,
                              float worst_info) const;

private:
    float d_obs_;
    float view_info_;
};

}  // namespace rc
