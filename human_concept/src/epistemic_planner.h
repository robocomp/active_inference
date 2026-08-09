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

#include "../../common/nbv/viewpoint_score.h"   // rc::nbv — the shared DETECTION-WEIGHTED NBV core

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
    /// The returned gain is DETECTION-WEIGHTED: P(detect at the far-side viewpoint) · ΔH. An occlusion-
    /// reducing orbit we could not get a detection from is worth ~0 nats, so it cannot out-bid nav_dist in
    /// the controller's EFE — which matters here because this affordance asks the robot to walk AROUND a
    /// person, the most expensive and most socially costly move it makes.
    EpistemicProposal compute(const Eigen::Vector2f& person_xy,
                              const Eigen::Vector2f& camera_xy,
                              float worst_info,
                              const rc::nbv::Sensor& sensor_in,
                              const std::vector<rc::nbv::Obstacle>& obstacles = {}) const;

    // The detector's operating envelope. The stand-off is the argmax of THIS, retiring d_obs_ as the thing
    // that set the viewing distance.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }


    void set_robot_radius(float m) { robot_radius_m_ = m; }

private:
    float d_obs_;
    float view_info_;
    // The envelope is OWNED (config-driven); the camera GEOMETRY arrives per call, because the zed
    // intrinsics appear only once robot_concept starts publishing frames. See sensor_from_graph().
    rc::detect::DetectorEnvelope det_env_{};
    float robot_radius_m_ = 0.30f;   // Shadow
};

}  // namespace rc
