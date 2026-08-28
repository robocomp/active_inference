/*
 *  Copyright (C) 2026 by Pablo Bustos
 *
 *  This program is free software: you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software Foundation, either version 3 or
 *  any later version. See <http://www.gnu.org/licenses/>.
 */
#pragma once

/**
 *  STAGE 2 of the online mount self-calibration: the camera against the LiDAR, with NO POSE IN IT.
 *
 *  ── The one idea ────────────────────────────────────────────────────────────────────────────────
 *  A room corner is seen by both sensors and it is the SAME physical (x, y): the LiDAR finds the
 *  wall-wall intersection at sensor height, the image finds the floor-wall-wall triple point at floor
 *  height. `CornerMatch::model_index` and `TriplePoint::vertex` are both indices into the ORIGINAL
 *  polygon vertex list, so the association is EXACT — a lookup, not a nearest-neighbour search with
 *  a gate. There is no misassociation failure mode to defend against, which matters: "pose jumps =
 *  MISASSOC" is already on record for corners in this codebase.
 *
 *  ── Why it is pose-free, and why that is the whole point ────────────────────────────────────────
 *  `CornerMatch::detected` is in the ROBOT frame. So the LiDAR's corner can be pushed through the
 *  camera<-robot extrinsic ALONE and predicted in the image without the localiser's pose appearing
 *  anywhere:
 *
 *      uv_lidar = project( cam_R_robot * p_robot + cam_t_robot )
 *      r        = uv_image - uv_lidar
 *
 *  Stage 1's residual was image-minus-MODEL, which carries the pose error: a heading error of dtheta
 *  enters as fx*dtheta and is indistinguishable from a boresight yaw within one view. That is exactly
 *  why measuring the boresight needed the Webots supervisor, and why the stage-1 yaw estimate still
 *  rests on the assumption that heading error is zero-mean across poses. Here the pose is not in the
 *  expression at all, so the assumption is not needed. What the residual contains is the CAMERA
 *  against the LIDAR — a hand-eye disagreement — plus each sensor's own corner-detection noise.
 *
 *  ── The second, less obvious dividend ───────────────────────────────────────────────────────────
 *  ★ It makes POOLING ACROSS WINDOWS legitimate, and that is what can break the pitch/height ridge.
 *    Stage 1 had to solve per window and report the BETWEEN-window scatter as the honest uncertainty,
 *    because the formal error ran 17-149x too small: consecutive windows disagreed far more than
 *    their own samples did. The dominant reason is that each window carries a DIFFERENT pose error,
 *    which is a per-window nuisance no amount of samples averages away. Remove the pose and the
 *    windows become genuinely comparable draws of one static quantity, so their information may be
 *    summed — and a pool over many poses spans the range diversity that one 5 s window never does.
 *    Pitch and height are told apart only by the fy/d covariate: within a window the visible corners
 *    span maybe 2-6 m, giving corr(1, fy/d) ~ 0.96 and cond ~ 50-100, which is what was measured.
 *    Across a drive the same columns span 1-8 m.
 *  ★ This is a CLAIM, not yet a result: it is right only if the pose really was the dominant
 *    between-window nuisance. The test is direct and is why both are logged — if the pooled
 *    between-window scatter here does NOT collapse relative to stage 1's, something else was driving
 *    it and pooling is still illegitimate.
 *
 *  ── What it cannot do ───────────────────────────────────────────────────────────────────────────
 *  It measures the camera RELATIVE TO THE LIDAR. A yaw error common to both mounts is invisible here
 *  and stays with the localiser's own gauge. That is the correct target anyway — the point is to make
 *  the two terms agree about the same room — but the number must not be reported as an absolute
 *  camera boresight. Naming it wrongly is the mistake this file exists partly to avoid repeating.
 */

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <vector>

#include "corner_detector.h"
#include "image_edge_ops.h"
#include "image_edge_types.h"

namespace rc::mount
{
    /// One corner seen by both sensors, with everything needed to form the residual.
    struct PairObs
    {
        int             vertex = -1;
        Eigen::Vector2f uv_image = Eigen::Vector2f::Zero();   ///< the triple point's measurement
        Eigen::Vector2f uv_lidar = Eigen::Vector2f::Zero();   ///< the LiDAR corner through the extrinsic
        Eigen::Vector2f r        = Eigen::Vector2f::Zero();   ///< uv_image - uv_lidar
        Eigen::Matrix2f cov      = Eigen::Matrix2f::Identity();
        Eigen::Matrix<float, 2, IMAGE_EDGE_NUISANCES> J =
            Eigen::Matrix<float, 2, IMAGE_EDGE_NUISANCES>::Zero();  ///< d(uv) / d(nuisance), prior-scaled
        float assoc_prob = 1.f;
        float range_m    = 0.f;
        bool  ok         = false;
    };

    /// Builds the pose-free pair for one (triple point, corner match) with the same model_index.
    ///
    /// `J` mirrors image_edge_source.cpp's hcol columns EXACTLY — pitch = rotation about the camera x
    /// axis, height = translation along camera z, yaw = rotation about camera z — but keeps the full
    /// 2-vector per nuisance instead of contracting onto a contour normal. A triple point has no
    /// aperture problem, so contracting would throw away half the information for no reason.
    /// The dt column is left at zero: it needs the body twist and the image/LiDAR offset, and unlike
    /// the other three it is not a property of the mount at all.
    inline PairObs make_pair(const TriplePoint& tp,
                             const rc::CornerDetector::CornerMatch& cm,
                             const CameraModel& cam,
                             const Eigen::Matrix3f& cam_R_robot,
                             const Eigen::Vector3f& cam_t_robot,
                             float sigma_pitch, float sigma_height, float sigma_yaw)
    {
        PairObs o;
        o.vertex = tp.vertex;
        o.assoc_prob = cm.assoc_prob;

        // ★ z = 0: the triple point is at FLOOR level, and this codebase's robot frame has its origin
        //   on the floor (the projection path forms e = p_room - pose with e.z() = p_room.z()). The
        //   LiDAR corner is a 2-D (x, y) at sensor height, but it is the SAME vertical edge, so its
        //   (x, y) is the triple point's (x, y). Taking z from the LiDAR would be wrong — it does not
        //   measure one.
        const Eigen::Vector3f p_robot(cm.detected.x(), cm.detected.y(), 0.f);
        const Eigen::Vector3f pc = cam_R_robot * p_robot + cam_t_robot;
        if (not (pc.norm() > 1e-3f)) return o;

        Eigen::Vector2d uv;
        if (not rc::img::project_with_model(cam, pc.cast<double>(), uv)) return o;
        Eigen::Matrix<double, 2, 3> P;
        if (not rc::img::project_jacobian_model(cam, pc.cast<double>(), P)) return o;

        o.uv_lidar = uv.cast<float>();
        o.uv_image = tp.uv_meas;
        o.r        = o.uv_image - o.uv_lidar;
        o.range_m  = pc.norm();

        const Eigen::Matrix<float, 2, 3> Pf = P.cast<float>();
        // ★ THE CAMERA'S OWN AXES, in camera coordinates — literally (1,0,0) and (0,0,1), exactly as
        //   image_edge_source.cpp:277 defines them. This previously used cam_R_robot.col(0) and
        //   .col(2), which are the ROBOT's axes expressed in camera coordinates: a different thing.
        //   For this mount the up-axes coincide, so the yaw and height columns were unaffected — but
        //   col(0) is the robot's FORWARD axis, so what was labelled "pitch" was a rotation about the
        //   optical axis, i.e. a ROLL. Any pitch figure from the pair fit before this is void; the
        //   yaw result (+0.014 deg) stands, which is why it is worth saying which is which rather
        //   than quietly re-running everything.
        const Eigen::Vector3f x_cam(1.f, 0.f, 0.f);
        const Eigen::Vector3f z_cam(0.f, 0.f, 1.f);
        o.J.col(0) = sigma_pitch  * (Pf * x_cam.cross(pc));
        o.J.col(1) = sigma_height * (Pf * z_cam);
        o.J.col(2) = sigma_yaw    * (Pf * z_cam.cross(pc));
        // column 3 (dt) deliberately zero — see the note above.

        // ── Covariance: BOTH sensors' uncertainty, in pixels ─────────────────────────────────────
        // The image half comes from the triple point's own line-intersection covariance. The LiDAR
        // half is its detection precision pushed through the same projection. Using only one of them
        // would hand the residual to whichever sensor was quietly the noisier.
        Eigen::Matrix2f cov_img = tp.cov_uv;
        Eigen::Matrix2f cov_lid = Eigen::Matrix2f::Zero();
        // Λ_det can be RANK-1 when the two walls are near-parallel — a shallow corner leaves the
        // bisector direction unconstrained (corner_detector.h). Inverting it blindly gives infinities;
        // an eigen-floor keeps the unconstrained direction merely very uncertain, which is true.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cm.information);
        if (es.info() == Eigen::Success)
        {
            Eigen::Vector2f ev = es.eigenvalues();
            for (int i = 0; i < 2; ++i) ev(i) = std::max(ev(i), 1e-4f);   // 1e-4 /m^2 -> sigma 100 m
            const Eigen::Matrix2f cov_xy =
                es.eigenvectors() * ev.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
            const Eigen::Matrix<float, 2, 2> Pxy = (Pf * cam_R_robot).leftCols<2>();
            cov_lid = Pxy * cov_xy * Pxy.transpose();
        }
        o.cov = cov_img + cov_lid + 0.01f * Eigen::Matrix2f::Identity();   // 0.1 px numerical floor
        o.ok  = o.cov.allFinite() and o.J.allFinite() and o.r.allFinite();
        return o;
    }

    /// The same 4-parameter normal-equation block stage 1 uses, so the two are directly comparable.
    /// Prior is the IDENTITY because J carries the prior sigma (see room_concept.h).
    struct Accum
    {
        Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
        Eigen::Vector4d b = Eigen::Vector4d::Zero();
        double          rTr = 0.0;
        long            n = 0;

        void add(const PairObs& o)
        {
            if (not o.ok) return;
            const Eigen::Matrix2d C = o.cov.cast<double>();
            const double det = C.determinant();
            if (not (det > 1e-12)) return;
            const Eigen::Matrix2d W = C.inverse();
            // ★ Weighted by assoc_prob: the detector's own posterior that this detection belongs to
            //   this model corner. A 0.6-probability association contributes 60% of a measurement,
            //   which is what it is — not a threshold, and not a full one either.
            const double w = std::clamp(static_cast<double>(o.assoc_prob), 0.0, 1.0);
            if (not (w > 1e-3)) return;
            // Columns 0-3 only: [4] is a per-contour map offset and a single paired corner carries
            // no information about it that is separable from the mount.
            const Eigen::Matrix<double, 2, 4> J = o.J.template leftCols<4>().template cast<double>();
            const Eigen::Vector2d r = o.r.cast<double>();
            H.noalias() += w * J.transpose() * W * J;
            b.noalias() += w * J.transpose() * W * r;
            rTr += w * r.dot(W * r);
            ++n;
        }
        void reset() { H.setZero(); b.setZero(); rTr = 0.0; n = 0; }

        /// Returns {parameters in units of prior sigma, posterior sigma, chi2/dof, cond, ok}.
        /// SIGN: r = uv_image - uv_lidar and J = d(uv_pred)/d(nuisance). A mount error of x makes the
        /// PREDICTION wrong by J*x while the image measurement is right, so r = -J*x and the fit
        /// returns minus the parameter — identical to stage 1, deliberately, so the two can be
        /// compared without a sign convention standing between them.
        struct Solution
        {
            Eigen::Vector4d p = Eigen::Vector4d::Zero();
            Eigen::Vector4d sigma = Eigen::Vector4d::Ones();
            double chi2_dof = 0.0, cond = 0.0, rho = 0.0;
            int    rho_i = 0, rho_j = 1, informed = 0;
            bool   ok = false;
        };
        [[nodiscard]] Solution solve(long min_n = 30) const
        {
            Solution s;
            if (n < min_n) return s;
            const Eigen::Matrix4d A = H + Eigen::Matrix4d::Identity();
            const Eigen::Matrix4d C = A.inverse();
            if (not C.allFinite()) return s;
            const Eigen::Vector4d x = C * b;
            const double chi2 = std::max(0.0, rTr - x.dot(b));
            s.chi2_dof = chi2 / std::max(1.0, 2.0 * static_cast<double>(n) - 4.0);
            const double infl = std::sqrt(std::max(1.0, s.chi2_dof));
            s.p = -x;
            for (int i = 0; i < 4; ++i)
            {
                s.sigma(i) = std::sqrt(std::max(0.0, C(i, i))) * infl;
                if (s.sigma(i) < 0.9) s.informed |= (1 << i);
            }
            Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    R(i, j) = C(i, j) / std::sqrt(std::max(1e-300, C(i, i) * C(j, j)));
            const Eigen::Vector4d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(R).eigenvalues();
            s.cond = ev(3) / std::max(1e-12, ev(0));
            for (int i = 0; i < 4; ++i)
                for (int j = i + 1; j < 4; ++j)
                    if (std::abs(R(i, j)) > std::abs(s.rho)) { s.rho = R(i, j); s.rho_i = i; s.rho_j = j; }
            s.ok = true;
            return s;
        }
    };
}   // namespace rc::mount
