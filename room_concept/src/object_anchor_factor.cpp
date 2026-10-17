#include "object_anchor_factor.h"

#include <cmath>

namespace rc
{
    float ObjectAnchorFactor::loss(const std::vector<ObjectAnchorObs>& anchors,
                                   const Eigen::Vector3f& pose,
                                   const Params& params)
    {
        if (not params.enable or anchors.empty())
            return 0.f;

        // Robot yaw as a scalar; R(-θ) rotates a room-frame delta into the robot frame.
        const float theta = pose.z();
        const float cos_th = std::cos(theta), sin_th = std::sin(theta);
        float total = 0.f;
        for (const auto& a : anchors)
        {
            // --- Predicted observation of the landmark in the robot frame ---
            //   z_hat_xy  = R(-θ)·(p_o.xy − t)
            //   z_hat_yaw = wrap(p_o.yaw − θ)
            const float dwx = a.pose_world.x() - pose.x(), dwy = a.pose_world.y() - pose.y();
            const float pred_x   =  cos_th * dwx + sin_th * dwy;
            const float pred_y   = -sin_th * dwx + cos_th * dwy;
            const float pred_yaw = a.pose_world.z() - theta;

            // --- Residual r = z_o ⊖ z_hat (yaw wrapped to (−π, π]) ---
            const float r_yaw_raw = a.obs_robot.z() - pred_yaw;
            const Eigen::Vector3f r(a.obs_robot.x() - pred_x, a.obs_robot.y() - pred_y,
                                    std::atan2(std::sin(r_yaw_raw), std::cos(r_yaw_raw)));

            // --- Robust (Huber) Mahalanobis on Λ = S^{-1} (yaw row/col already zeroed if !has_orientation):
            //     whitened distance d = sqrt(rᵀΛr) ---
            const float m2 = r.dot(a.information * r);
            const float d  = std::sqrt(m2 + 1e-9f);
            const float w  = d <= params.huber_delta ? 1.f : params.huber_delta / d;
            total += params.weight * 0.5f * w * m2;
        }
        return total;
    }
} // namespace rc
