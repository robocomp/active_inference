/*
 * epistemic_planner.h
 *
 * Epistemic action proposal for the bottle-concept agent: send the robot to view the bottle's
 * HIDDEN FACE. A vertical cylinder seen by a single depth camera is observed only on its near arc;
 * the far arc is occluded, and that one-sidedness is exactly what makes the symmetric SDF
 * depth-degenerate (radius and the camera-ward XY centroid are poorly constrained — see
 * bottle-sdf-depth-bias). The most informative next view is therefore from the OPPOSITE side:
 *
 *   dir    = normalize(bottle_xy − camera_xy)          (camera → bottle, horizontal)
 *   v*     = bottle_xy + dir · d_obs                   (stand d_obs beyond the bottle, far side)
 *   heading= atan2(bottle_y − v.y, bottle_x − v.x)     (look back at the bottle)
 *   ΔH     = ½·log(1 + I_view / Y_radius)              (entropy reduction in the degenerate radius DOF)
 *
 * ΔH→0 as the radius is resolved, so a well-seen bottle publishes a low gain the controller's EFE
 * selection won't pick (belief→knowledge governor; the affordance node persists, it just isn't chosen).
 */

#pragma once

#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "bottle_model.h"
#include "bottle_belief.h"                           // AI2 belief: Σ + predicted_information for the NBV
#include "../../common/nbv/viewpoint_score.h"        // rc::nbv — the shared DETECTION-WEIGHTED NBV core

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
     * @param d_obs      Stand-off from the bottle at the far-side viewpoint (m).
     * @param view_info  Fisher precision a full back-view is expected to add to the radius DOF
     *                   (sets the ΔH scale; larger → higher gain while the radius is still uncertain).
     */
    explicit EpistemicPlanner(float d_obs = 0.9f, float view_info = 50.0f);

    /**
     * Hidden-face next-best-view. @p camera_xy is the ZED origin in the room frame (horizontal). The gain
     * is the D-optimal expected entropy reduction on the belief's FULL covariance Σ (mirrors table/chair):
     *   ΔH = ½·log det( I₅ + Σ · ΔI ),   ΔI = Σₚ (1/R) Jₚ Jₚᵀ  (BottleBelief::predicted_information)
     * over synthetic hidden-arc points, R = σ_base². ΔH→0 as the belief tightens (belief→knowledge).
     * Returns valid==false only if the camera→bottle ray is degenerate (can't tell which side is hidden).
     */
    /// The returned gain is DETECTION-WEIGHTED: P(detect at the far-side viewpoint) · ΔH. A hidden-face view
    /// we could not get a mask from is worth ~0 nats, so it cannot out-bid nav_dist in the controller's EFE —
    /// which matters most here, because this affordance asks the robot to drive all the way AROUND the bottle.
    EpistemicProposal compute(const BottleBelief& belief,
                              const Eigen::Vector2f& camera_xy,
                              float sigma_base,
                              const rc::nbv::Sensor& sensor_in,
                              const std::vector<rc::nbv::Obstacle>& obstacles = {},
                              /// The REACHABLE region. This planner scores ONE candidate of its own geometry rather than going
                              /// through plan_faces, so it does not inherit that function's is_reachable check — without the
                              /// polygon a far-side viewpoint outside the room is scored as perfectly good, and the controller
                              /// then REPAIRS the unroutable standpoint onto the bottle itself.
                              const std::vector<Eigen::Vector2f>& room_polygon = {}) const;

    // The detector's operating envelope. The stand-off is the argmax of THIS, which is what retires the
    // d_obs_ constructor constant as the thing that set the viewing distance.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }


    // Footprint radius of the robot: the geometric floor under the stand-off. A physical dimension.
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
