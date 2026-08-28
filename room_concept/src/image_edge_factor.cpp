/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#include <Eigen/Dense>
#include "image_edge_factor.h"

#include <cmath>

#include "image_edge_accumulate.h"
#include "image_edge_ops.h"

namespace rc
{
namespace
{
    /// Project a camera-frame point in TORCH, reproducing rc::img::project_with_model exactly.
    /// Autograd flows through this, which is the whole point: compute_posterior_covariance() takes
    /// an autograd Hessian of compute_rfe_loss, so a term that is not differentiable HERE is a term
    /// that is missing from the reported sigma — and that sigma is the pre-registered criterion.
    std::pair<torch::Tensor, torch::Tensor> project_t(const CameraModel& m, const torch::Tensor& p)
    {
        const auto X = p[0], Y = p[1], Z = p[2];
        if (m.kind == CameraModel::Kind::Pinhole)
            return { m.fx * X / Y + m.cx, m.cy - m.fy * Z / Y };

        const auto rho = torch::sqrt(X * X + Y * Y + 1e-18);
        const auto r   = torch::sqrt(X * X + Y * Y + Z * Z + 1e-18);
        const auto az  = m.azimuth_sign * torch::atan2(X, Y) + m.azimuth_offset;
        const auto u   = (az / (2.0 * M_PI) + 0.5) * m.width;      // NOT wrapped: see below
        const auto v   = (m.kind == CameraModel::Kind::Equirect)
                       ? (torch::asin(torch::clamp(-Z / r, -1.0, 1.0)) / M_PI + 0.5) * m.height
                       : 0.5 * m.height - (m.width / (2.0 * M_PI)) * (Z / rho);
        return { u, v };
    }
}   // namespace

torch::Tensor ImageEdgeFactor::loss(const ImageEdgeObs& obs,
                                    const torch::Tensor& pose_xy,
                                    const torch::Tensor& pose_theta,
                                    const Params& params,
                                    torch::Device device)
{
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor total = torch::zeros({}, f32);
    if (not params.enable or obs.empty() or not obs.cam.valid)
        return total;

    const auto theta = pose_theta.reshape({});
    const auto cth = torch::cos(theta), sth = torch::sin(theta);

    // The static camera<-robot extrinsic. A CONSTANT of the optimisation — never a graph read of
    // room<-robot, which is the variable (see image_edge_source.h on the circularity trap).
    const auto Rc = torch::from_blob(const_cast<float*>(obs.cam_R_robot.data()), {3, 3},
                                     torch::kFloat32).clone().t().to(device);   // Eigen is col-major
    const auto tc = torch::from_blob(const_cast<float*>(obs.cam_t_robot.data()), {3},
                                     torch::kFloat32).clone().to(device);

    // ── TRIPLE POINTS: the 0-D form of the same evidence ─────────────────────────────────────────
    // A 2-D projection residual per floor-wall-wall corner, weighted by the line-intersection
    // covariance. Chosen INSTEAD of the per-sample residuals, never alongside them: a triple point is
    // those same segment offsets re-expressed, so running both counts the same photons twice.
    //
    // ★ NO WOODBURY HERE, AND THAT IS NOT AN OVERSIGHT. The marginalisation exists to stop N samples
    //   along one wall claiming sqrt(N) precision about a position they share. A triple point is ONE
    //   measurement per corner; there is no N to over-count. The mount nuisances remain unmodelled in
    //   this branch, which is a real gap, but the measured mount residual is -0.4 px / +3 mm, an
    //   order below the corner's own sigma — so folding it in would be modelling a term smaller than
    //   the noise it sits in.
    //
    // ★ A ONE-CORNER FRAME STILL CONTRIBUTES. Two residual components on three DOF is a rank-2
    //   constraint, and a factor CONTRIBUTES information rather than having to be invertible: in a
    //   multi-modal graph which landmarks are visible changes frame to frame by design, and the SDF
    //   and LiDAR terms supply the rest. Nothing here gates on the corner count.
    if (params.use_triple_points)
    {
        for (const auto& t : obs.triple_points)
        {
            const auto px = torch::tensor({t.p_room.x(), t.p_room.y(), t.p_room.z()}, f32);
            const auto dx = px[0] - pose_xy[0];
            const auto dy = px[1] - pose_xy[1];
            const auto rx =  cth * dx + sth * dy;
            const auto ry = -sth * dx + cth * dy;
            const auto p_robot = torch::stack({rx, ry, px[2]});
            const auto p_cam   = torch::matmul(Rc, p_robot) + tc;
            const auto [u, v]  = project_t(obs.cam, p_cam);

            auto du = u - t.uv_meas.x();
            if (obs.cam.kind != CameraModel::Kind::Pinhole)
            {   // same detached-wrap trick as the sample path: exact, not an approximation
                const float wrap = std::round(du.item<float>() / obs.cam.width) * obs.cam.width;
                du = du - wrap;
            }
            const auto dv = v - t.uv_meas.y();

            // W = cov_uv^-1, inverted in double and only if it is actually invertible. A vertex seen
            // edge-on has a near-parallel line crossing and a huge covariance; that is TRUE and the
            // weight simply goes to zero, so no gate is needed.
            const Eigen::Matrix2d C = t.cov_uv.cast<double>();
            const double det = C.determinant();
            if (not (det > 1e-12) or not C.allFinite()) continue;
            const Eigen::Matrix2d W = C.inverse();
            if (not W.allFinite()) continue;

            const auto quad = static_cast<float>(W(0, 0)) * du * du
                            + 2.f * static_cast<float>(W(0, 1)) * du * dv
                            + static_cast<float>(W(1, 1)) * dv * dv;
            if (not std::isfinite(quad.item<float>())) continue;
            total = total + 0.5 * quad;
        }
        return total;
    }

    for (const auto& seg : obs.segments)
    {
        // Per-segment Woodbury accumulators, mirroring image_edge_accumulate.h term for term.
        // Held as plain doubles because the responsibilities gamma are FROZEN (an EM E-step) and the
        // nuisance block carries no gradient — only the residuals do.
        std::vector<torch::Tensor> r_list;
        std::vector<float>         w_list;
        std::vector<rc::img::NuisVec> h_list;
        r_list.reserve(seg.samples.size());

        for (const auto& s : seg.samples)
        {
            const auto px = torch::tensor({s.p_room.x(), s.p_room.y(), s.p_room.z()}, f32);
            const auto dx = px[0] - pose_xy[0];
            const auto dy = px[1] - pose_xy[1];
            // p_robot = R(-theta) * (p_room - t)
            const auto rx =  cth * dx + sth * dy;
            const auto ry = -sth * dx + cth * dy;
            const auto p_robot = torch::stack({rx, ry, px[2]});
            const auto p_cam   = torch::matmul(Rc, p_robot) + tc;

            const auto [u, v] = project_t(obs.cam, p_cam);
            auto du = u - s.uv_meas.x();
            if (obs.cam.kind != CameraModel::Kind::Pinhole)
            {
                // Fold the panorama wrap in a way autograd can follow: subtract a CONSTANT multiple
                // of W chosen from the current value. The derivative of a wrap is 1 almost
                // everywhere, so detaching the integer part is exact, not an approximation.
                const float duv = du.item<float>();
                const float wrap = std::round(duv / obs.cam.width) * obs.cam.width;
                du = du - wrap;
            }
            const auto dv = v - s.uv_meas.y();
            const auto r  = s.n_hat.x() * du + s.n_hat.y() * dv;

            const float rf = r.item<float>();
            if (not std::isfinite(rf) or not (s.sigma_px > 0.f)) continue;
            const float s2  = s.sigma_px * s.sigma_px;
            const float gam = rc::img::responsibility(rf, s2, s.pi_vis, s.search_L);
            if (not (gam > 1e-6f)) continue;

            r_list.push_back(r);
            w_list.push_back(gam / s2);
            h_list.push_back(s.h);
        }
        if (r_list.empty()) continue;

        // sum_k w_k r_k^2  -  Z^T K Z, with Z = sum_k w_k h_k r_k, K = (I + Q)^-1.
        // The second term is what stops N correlated samples on one wall claiming sqrt(N) precision.
        rc::img::NuisMat Q = rc::img::NuisMat::Zero();
        for (std::size_t k = 0; k < r_list.size(); ++k)
            Q.noalias() += w_list[k] * (h_list[k] * h_list[k].transpose());
        const rc::img::NuisMat K = (rc::img::NuisMat::Identity() + Q).inverse();

        auto sum_wr2 = torch::zeros({}, f32);
        std::vector<torch::Tensor> Zc;
        Zc.reserve(IMAGE_EDGE_NUISANCES);
        for (int m = 0; m < IMAGE_EDGE_NUISANCES; ++m) Zc.push_back(torch::zeros({}, f32));
        for (std::size_t k = 0; k < r_list.size(); ++k)
        {
            sum_wr2 = sum_wr2 + w_list[k] * r_list[k] * r_list[k];
            for (int m = 0; m < IMAGE_EDGE_NUISANCES; ++m)
                Zc[m] = Zc[m] + (w_list[k] * h_list[k](m)) * r_list[k];
        }
        auto Z = torch::stack(Zc);
        const auto Kt = torch::from_blob(const_cast<float*>(K.data()),
                                         {IMAGE_EDGE_NUISANCES, IMAGE_EDGE_NUISANCES},
                                         torch::kFloat32).clone().t().to(device);
        const auto quad = torch::dot(Z, torch::matmul(Kt, Z));
        total = total + 0.5 * (sum_wr2 - quad);
    }
    return total;
}

}  // namespace rc
