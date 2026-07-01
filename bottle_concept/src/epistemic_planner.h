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
#include "bottle_belief.h"            // AI2 belief: Σ + predicted_information for the Σ-based NBV

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
     * Compute the hidden-face viewpoint. @p camera_xy is the ZED origin in the room frame
     * (horizontal). @p posterior_info is BottleInstance::fisher_info_raw (DOF order
     * [cx,cy,cz,radius,height]); the radius precision (index 3) drives ΔH.
     * Returns valid==false only if the camera→bottle ray is degenerate (can't tell which side is hidden).
     */
    EpistemicProposal compute(const BottleModel& model,
                              const Eigen::Vector2f& camera_xy,
                              const std::array<float, 5>& posterior_info) const;

    /**
     * AI2-native next-best-view (cfg.use_ai2). Same hidden-face viewpoint geometry, but the gain is the
     * D-optimal expected entropy reduction on the belief's FULL covariance Σ (mirrors table/chair):
     *   ΔH = ½·log det( I₅ + Σ · ΔI ),   ΔI = Σₚ (1/R) Jₚ Jₚᵀ  (BottleBelief::predicted_information)
     * over synthetic hidden-arc points, R = σ_base². ΔH→0 as the belief tightens (belief→knowledge).
     */
    EpistemicProposal compute(const BottleBelief& belief,
                              const Eigen::Vector2f& camera_xy,
                              float sigma_base) const;

private:
    float d_obs_;
    float view_info_;
};

}  // namespace rc
