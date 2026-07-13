#include "object_anchor_factor.h"

namespace rc
{
    torch::Tensor ObjectAnchorFactor::loss(const std::vector<ObjectAnchorObs>& anchors,
                                           const torch::Tensor& pose_xy,
                                           const torch::Tensor& pose_theta,
                                           const Params& params,
                                           torch::Device device)
    {
        const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor total = torch::zeros({}, f32);
        if (not params.enable or anchors.empty())
            return total;

        // Robot yaw as a scalar; R(-θ) rotates a room-frame delta into the robot frame.
        const auto theta  = pose_theta.reshape({});
        const auto cos_th = torch::cos(theta);
        const auto sin_th = torch::sin(theta);

        for (const auto& a : anchors)
        {
            // --- Predicted observation of the landmark in the robot frame ---
            //   z_hat_xy  = R(-θ)·(p_o.xy − t)
            //   z_hat_yaw = wrap(p_o.yaw − θ)
            const auto c_w = torch::tensor({a.pose_world.x(), a.pose_world.y()}, f32);
            const auto dw  = c_w - pose_xy;                       // [2] room-frame delta
            const auto pred_x   =  cos_th * dw[0] + sin_th * dw[1];
            const auto pred_y   = -sin_th * dw[0] + cos_th * dw[1];
            const auto pred_yaw = a.pose_world.z() - theta;

            // --- Residual r = z_o ⊖ z_hat (yaw wrapped to (−π, π]) ---
            const auto obs = torch::tensor({a.obs_robot.x(), a.obs_robot.y(), a.obs_robot.z()}, f32);
            const auto r_x   = obs[0] - pred_x;
            const auto r_y   = obs[1] - pred_y;
            const auto r_yaw_raw = obs[2] - pred_yaw;
            const auto r_yaw = torch::atan2(torch::sin(r_yaw_raw), torch::cos(r_yaw_raw));
            const auto r = torch::stack({r_x, r_y, r_yaw});       // [3]

            // --- Information matrix Λ = S^{-1} (yaw row/col already zeroed if !has_orientation).
            // Eigen is column-major but Λ is symmetric, so a plain row-major view is identical.
            const auto lam = torch::from_blob(const_cast<float*>(a.information.data()),
                                              {3, 3}, torch::kFloat32).clone().to(device);

            // --- Robust (Huber) Mahalanobis: whitened distance d = sqrt(rᵀΛr) ---
            const auto m2 = torch::matmul(r.unsqueeze(0), torch::matmul(lam, r.unsqueeze(1))).reshape({});
            const auto d  = torch::sqrt(m2 + 1e-9f);
            const auto w  = torch::where(d <= params.huber_delta,
                                         torch::ones_like(d),
                                         params.huber_delta / d);
            total = total + params.weight * 0.5f * w * m2;
        }
        return total;
    }
} // namespace rc
