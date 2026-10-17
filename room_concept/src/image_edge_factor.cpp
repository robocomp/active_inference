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
    /// Room point -> pixel residual base (du, dv), panorama wrap folded into du. false = not projectable.
    bool project_residual(const ImageEdgeObs& obs, const Eigen::Vector3f& pose, const Eigen::Vector3f& p_room,
                          const Eigen::Vector2f& uv_meas, float& du, float& dv)
    {
        const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
        const float dx = p_room.x() - pose.x(), dy = p_room.y() - pose.y();
        // p_robot = R(-theta) * (p_room - t); then the STATIC camera<-robot extrinsic — a CONSTANT of the
        // optimisation, never a graph read of room<-robot (see image_edge_source.h on the circularity trap).
        const Eigen::Vector3f p_robot(cth * dx + sth * dy, -sth * dx + cth * dy, p_room.z());
        const Eigen::Vector3f p_cam = obs.cam_R_robot * p_robot + obs.cam_t_robot;
        Eigen::Vector2d uv;
        if (not rc::img::project_with_model(obs.cam, p_cam.cast<double>(), uv)) return false;
        du = static_cast<float>(uv.x()) - uv_meas.x();
        if (obs.cam.kind != CameraModel::Kind::Pinhole)
            du -= std::round(du / obs.cam.width) * obs.cam.width;
        dv = static_cast<float>(uv.y()) - uv_meas.y();
        return true;
    }
}   // namespace

float ImageEdgeFactor::loss(const ImageEdgeObs& obs, const Eigen::Vector3f& pose, const Params& params)
{
    if (not params.enable or obs.empty() or not obs.cam.valid)
        return 0.f;

    // ── TRIPLE POINTS: the 0-D form of the same evidence ─────────────────────────────────────────
    // A 2-D projection residual per floor-wall-wall corner, weighted by the line-intersection
    // covariance. Chosen INSTEAD of the per-sample residuals, never alongside them: a triple point is
    // those same segment offsets re-expressed, so running both counts the same photons twice.
    // ★ NO WOODBURY HERE, AND THAT IS NOT AN OVERSIGHT: one measurement per corner, no N to over-count.
    // ★ A ONE-CORNER FRAME STILL CONTRIBUTES: a factor CONTRIBUTES information rather than having to be
    //   invertible; the SDF and LiDAR terms supply the rest. Nothing here gates on the corner count.
    if (params.use_triple_points)
    {
        double total = 0.0;
        for (const auto& t : obs.triple_points)
        {
            float du = 0.f, dv = 0.f;
            if (not project_residual(obs, pose, t.p_room, t.uv_meas, du, dv)) continue;
            // W = cov_uv^-1, inverted in double and only if it is actually invertible. A vertex seen
            // edge-on has a huge covariance; that is TRUE and the weight simply goes to zero.
            const Eigen::Matrix2d C = t.cov_uv.cast<double>();
            const double det = C.determinant();
            if (not (det > 1e-12) or not C.allFinite()) continue;
            const Eigen::Matrix2d W = C.inverse();
            if (not W.allFinite()) continue;
            const double quad = W(0, 0) * du * du + 2.0 * W(0, 1) * du * dv + W(1, 1) * dv * dv;
            if (not std::isfinite(quad)) continue;
            total += 0.5 * quad;
        }
        return static_cast<float>(total);
    }

    double total = 0.0;
    for (const auto& seg : obs.segments)
    {
        // Per-segment Woodbury accumulators, mirroring image_edge_accumulate.h term for term. The
        // responsibilities gamma are FROZEN (an EM E-step).
        std::vector<float>            r_list, w_list;
        std::vector<rc::img::NuisVec> h_list;
        for (const auto& s : seg.samples)
        {
            float du = 0.f, dv = 0.f;
            if (not project_residual(obs, pose, s.p_room, s.uv_meas, du, dv)) continue;
            const float r = s.n_hat.x() * du + s.n_hat.y() * dv;
            if (not std::isfinite(r) or not (s.sigma_px > 0.f)) continue;
            const float s2  = s.sigma_px * s.sigma_px;
            const float gam = rc::img::responsibility(r, s2, s.pi_vis, s.search_L);
            if (not (gam > 1e-6f)) continue;
            r_list.push_back(r);
            w_list.push_back(gam / s2);
            h_list.push_back(s.h);
        }
        if (r_list.empty()) continue;

        // sum_k w_k r_k^2  -  Z^T K Z, with Z = sum_k w_k h_k r_k, K = (I + Q)^-1.
        // The second term is what stops N correlated samples on one wall claiming sqrt(N) precision.
        rc::img::NuisMat Q = rc::img::NuisMat::Zero();
        rc::img::NuisVec Z = rc::img::NuisVec::Zero();
        double sum_wr2 = 0.0;
        for (std::size_t k = 0; k < r_list.size(); ++k)
        {
            Q.noalias() += w_list[k] * (h_list[k] * h_list[k].transpose());
            Z.noalias() += (w_list[k] * r_list[k]) * h_list[k];
            sum_wr2 += static_cast<double>(w_list[k]) * r_list[k] * r_list[k];
        }
        const rc::img::NuisMat K = (rc::img::NuisMat::Identity() + Q).inverse();
        total += 0.5 * (sum_wr2 - static_cast<double>(Z.dot(K * Z)));
    }
    return static_cast<float>(total);
}

}  // namespace rc
