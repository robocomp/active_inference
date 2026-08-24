/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

/*
 *  image_edge_accumulate.h — ONE definition of the RGB edge term's normal equations, shared by the
 *  GN factor (room_gn_solver.cpp), the torch mirror (image_edge_factor.cpp) and gn_selftest.
 *
 *  It exists as its own unit for the same reason room_obs_weights.h does: if the two backends each
 *  had their own copy, the shadow comparison would be measuring their disagreement instead of the
 *  solvers'.
 *
 *  ═══ THE COMMON-MODE MARGINALISATION — the reason this term does not out-vote the LiDAR ═══
 *
 *  N samples along one straight junction are NOT N measurements. A line has 2 DOF, and every sample
 *  on it shares camera pitch, camera height, boresight yaw, exposure and the image/LiDAR time offset.
 *  Summing N independent pixel residuals makes the term claim ~sqrt(N) times the precision it has.
 *  The numbers here are not abstract: at 3 m one pixel is ~1.9 cm against the LiDAR's ObsSigma of
 *  0.05 m, so an uncapped 200-sample wall would simply overrule the LiDAR while every convergence
 *  diagnostic still looked healthy.
 *
 *  Model the residual covariance of ONE segment as R = D + H*H^T, D = diag(sigma_k^2 / gamma_k) and
 *  H the per-sample sensitivity to each UNIT-VARIANCE shared nuisance (the sigmas are already folded
 *  into ImageEdgeSample::h). With w_k = gamma_k / sigma_k^2, Woodbury gives, exactly:
 *
 *      U = sum_k w_k * J_k^T h_k^T     (3 x M)
 *      Q = sum_k w_k * h_k h_k^T       (M x M)
 *      Z = sum_k w_k * h_k r_k         (M)
 *      K = (I_M + Q)^-1                (M x M, M = 4)
 *
 *      H_seg = sum_k w_k J_k^T J_k  -  U K U^T
 *      b_seg = sum_k w_k J_k^T r_k  -  U K Z
 *      L_seg = 0.5 * ( sum_k w_k r_k^2  -  Z^T K Z )
 *
 *  ★ WHY RANK-M AND NOT ONE POSE-SPACE Sigma_c. For a floor line the height sensitivity is
 *    h_k ~ fy/d_k, which varies ~3x along a single wall. This form therefore caps the line's MEAN at
 *    roughly one nuisance-limited observation while leaving its INTERNAL CONTRAST — near-end vs
 *    far-end disagreement, which is real information about the range gradient — at full weight. A
 *    single capped covariance, or any sigma-floor, destroys both equally.
 *
 *  ★ L_seg MATTERS AS MUCH AS THE OTHER TWO. IFactor requires evaluate() to return exactly what
 *    linearize() linearizes, or Levenberg's accept/reject test measures the wrong function
 *    (room_gn_solver.cpp:32-37 documents that trap). The `- Z^T K Z` term is not optional.
 *
 *  ★★ THE SUBTRACTION IS A CANCELLATION, AND IT MUST BE ACCUMULATED IN DOUBLE.
 *    `Id` and `U K U^T` are both large and nearly equal — that is the whole point, since their
 *    difference is the capped information. Measured on a synthetic floor line with J ~ fy/d: at
 *    N = 200, Id.trace() = 2.6e7 while the true H.trace() = 1.2e4, a difference of THREE orders of
 *    magnitude; at N = 1000 the float32 version returned trace(H) = -1.0e4 and a minimum eigenvalue
 *    of -1.1e4, i.e. an INDEFINITE information matrix. Downstream that is a negative variance, which
 *    is precisely the failure recursive_laplace.h:119-135 records as the cause of the table_concept
 *    yaw/extent teleports.
 *    So every accumulator below is `double`, and only the finished 3x3 / 3x1 / scalar are cast back
 *    to float. This is a pure numerics choice — no clamp, no threshold, no model term. The assertion
 *    that keeps it honest is in gn_selftest: H must stay positive semi-definite and its trace must
 *    SATURATE as N grows, over a range of N that includes far more samples than a real segment has.
 */

#include <cmath>

#include <Eigen/Dense>

#include "image_edge_types.h"

namespace rc::img
{
    using NuisVec = Eigen::Matrix<float, IMAGE_EDGE_NUISANCES, 1>;
    using NuisMat = Eigen::Matrix<float, IMAGE_EDGE_NUISANCES, IMAGE_EDGE_NUISANCES>;

    /// One segment's contribution, in the solver's convention (L = 0.5*sum w r^2, b = sum w r J^T).
    struct SegmentAccum
    {
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();
        float loss = 0.f;
        float trace_raw = 0.f;   ///< trace(sum w J^T J) BEFORE the cap
        float trace_eff = 0.f;   ///< trace(H) AFTER it. raw/eff IS the fabrication factor — log it.
        int   n_used = 0;
        float sum_gamma = 0.f;
        float chi2 = 0.f;        ///< sum gamma_k * r_k^2 / (sigma_k^2 + sigma_pred^2-ish), for consistency
    };

    /// Mixture responsibility for one sample: inlier N(r; 0, s2) against a UNIFORM over the window
    /// that was actually searched.
    ///
    /// ★ WHY NOT THE SOLVER'S HUBER, which would be cheaper and is already there. The dominant
    ///   failure of a normal search is NOT a large residual — it is the WRONG PEAK AT A SMALL
    ///   RESIDUAL (a skirting board 4 px below the junction). Huber is a function of |r| alone, so it
    ///   downweights the honest large residual and fully trusts the confident wrong one. The mixture
    ///   sees it, because the inlier density is compared against a uniform whose width L_k is KNOWN.
    ///   Occlusion also has nowhere to enter under Huber; here it is exactly the prior pi_vis.
    inline float responsibility(float r, float s2, float pi_vis, float search_L)
    {
        if (not (s2 > 0.f) or not (search_L > 0.f)) return 0.f;
        const float pi_c = std::clamp(pi_vis, 1e-4f, 1.f - 1e-4f);
        const float inlier  = pi_c * std::exp(-0.5f * r * r / s2)
                            / std::sqrt(2.f * static_cast<float>(M_PI) * s2);
        const float outlier = (1.f - pi_c) / (2.f * search_L);
        const float den = inlier + outlier;
        return den > 0.f ? inlier / den : 0.f;
    }

    /// Accumulate ONE segment. `residual(k, J_row)` must fill J_row (1x3, d r_k / d x) and return r_k;
    /// return NaN to skip the sample. `sigma_pred2(k)` is the projected pose variance in px^2, which
    /// widens the inlier density so a merely-uncertain prediction is not judged an outlier.
    ///
    /// gamma is computed HERE and the caller must FREEZE it for the whole linearisation (an EM
    /// E-step). Because the loss is defined as that frozen-gamma surrogate, evaluate() and
    /// linearize() use one identical expression — which is how this factor avoids the IRLS-weight-
    /// vs-loss-weight split that Se2LandmarkFactor:274-286 records as a real past bug.
    template <class ResidualFn, class SigmaPredFn>
    SegmentAccum accumulate_segment(const ImageEdgeSegment& seg, ResidualFn&& residual,
                                    SigmaPredFn&& sigma_pred2)
    {
        using Vec3d = Eigen::Matrix<double, 3, 1>;
        using Mat3d = Eigen::Matrix<double, 3, 3>;
        using NuisVecD = Eigen::Matrix<double, IMAGE_EDGE_NUISANCES, 1>;
        using NuisMatD = Eigen::Matrix<double, IMAGE_EDGE_NUISANCES, IMAGE_EDGE_NUISANCES>;

        SegmentAccum out;
        // DOUBLE throughout: see the header note. The difference Id - U K U^T is a 3-4 decade
        // cancellation, and in float32 it goes indefinite on a segment that is not even unusually long.
        Eigen::Matrix<double, 3, IMAGE_EDGE_NUISANCES> U; U.setZero();
        NuisMatD Q; Q.setZero();
        NuisVecD Z; Z.setZero();
        double sum_wr2 = 0.0, chi2 = 0.0, sum_gamma = 0.0;
        Mat3d Id = Mat3d::Zero();
        Vec3d bd = Vec3d::Zero();

        for (std::size_t k = 0; k < seg.samples.size(); ++k)
        {
            const auto& s = seg.samples[k];
            Eigen::Matrix<float, 1, 3> Jf;
            const float rf = residual(k, Jf);
            if (not std::isfinite(rf) or not Jf.allFinite()) continue;
            if (not (s.sigma_px > 0.f) or not std::isfinite(s.sigma_px)) continue;

            const float s2  = s.sigma_px * s.sigma_px + std::max(0.f, sigma_pred2(k));
            const float gam = responsibility(rf, s2, s.pi_vis, s.search_L);
            if (not (gam > 1e-6f)) continue;

            const Eigen::Matrix<double, 1, 3> J = Jf.cast<double>();
            const NuisVecD h = s.h.cast<double>();
            const double r = static_cast<double>(rf);
            const double w = static_cast<double>(gam) /
                             (static_cast<double>(s.sigma_px) * static_cast<double>(s.sigma_px));

            Id.noalias() += w * (J.transpose() * J);
            bd.noalias() += (w * r) * J.transpose();
            U.noalias()  += w * (J.transpose() * h.transpose());
            Q.noalias()  += w * (h * h.transpose());
            Z.noalias()  += (w * r) * h;
            sum_wr2      += w * r * r;
            chi2         += static_cast<double>(gam) * r * r / static_cast<double>(s2);
            sum_gamma    += gam;
            ++out.n_used;
        }
        if (out.n_used == 0) return out;

        const NuisMatD K = (NuisMatD::Identity() + Q).inverse();
        const Eigen::Matrix<double, 3, IMAGE_EDGE_NUISANCES> UK = U * K;
        Mat3d Heff = Id - UK * U.transpose();
        Heff = 0.5 * (Heff + Heff.transpose());          // symmetric in exact arithmetic; enforce it
        const Vec3d beff = bd - UK * Z;
        const double loss = 0.5 * (sum_wr2 - Z.dot(K * Z));

        out.trace_raw = static_cast<float>(Id.trace());
        out.trace_eff = static_cast<float>(Heff.trace());
        out.H         = Heff.cast<float>();
        out.b         = beff.cast<float>();
        out.loss      = static_cast<float>(loss);
        out.chi2      = static_cast<float>(chi2);
        out.sum_gamma = static_cast<float>(sum_gamma);
        if (not std::isfinite(out.loss) or not out.H.allFinite() or not out.b.allFinite())
        { out.loss = 0.f; out.H.setZero(); out.b.setZero(); out.n_used = 0; }
        return out;
    }

}  // namespace rc::img
