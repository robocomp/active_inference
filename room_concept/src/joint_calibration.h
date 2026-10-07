/*  joint_calibration.h — ONE posterior over the odometry parameters, the helios mount and every camera
 *  mount, from the evidence the separate calibrators already keep.
 *
 *  WHY. The chain "LiDAR trusted -> odometry against it -> cameras against it" never questions the
 *  LiDAR, and each link reports itself certain of what the previous link assumed. The helios yaw is
 *  the parameter they share (spec §9.1): the motion block measures it as eps_yaw, and every camera yaw
 *  "relative to the LiDAR" contains it. Fusing through that one column gives camera mounts in the
 *  BODY frame (gauge G3: the wheels define body) with the LiDAR-yaw uncertainty carried, not dropped.
 *
 *  THE MODEL (plan docs/superpowers/plans/2026-10-05-joint-calibration.md, Task 3, r1).
 *  theta = [ motion P_COUNT | cam_1 4 | ... | cam_N 4 ], camera entries in the BODY frame. Camera c's
 *  DATA measured its mount relative to the LiDAR: y_c = A_c theta, A_c = identity on its own columns
 *  plus kCamPerLidarYaw / kEpsPerLidarYaw on the eps_yaw column of its yaw row. Its PRIOR is on its own
 *  body-frame mount (I_c), because the two mounts are physically independent -- a prior on the
 *  relative mount would correlate psi_c with psi_h at 0.94 before any data (Fable review Q2):
 *
 *      H = H_motion + sum_c [ A_c^T H_c,data A_c + I_c^T H_c,prior I_c ]
 *      b = b_motion + sum_c [ A_c^T b_c,data     + I_c^T b_c,prior     ]
 *
 *  Consequence (pinned by the selftest): with eps at its prior only and a precise y_rel, the relative
 *  measurement is SHARED between the two mounts in proportion to their prior variances; "joint ==
 *  chained" holds only in the limit sigma_h -> 0.
 *
 *  ⚠ A camera's 4 columns mean what rc::mount::Accum::Solution::p means: the error REMAINING relative
 *  to the mount that camera currently uses (mountApply folds each solve into `applied`), with the
 *  prior's pull toward the ORIGINAL graph mount carried in b_prior.
 *  ⚠ A planted/real helios LEVER also shifts every LiDAR corner the camera pairs use (~0.57 deg of
 *  bearing at 3 m for 3 cm), which the camera's four parameters cannot express; it is NOT modelled
 *  here (it would need lever columns in PairObs::J). Plan Task 6 grades the yaw coupling on a
 *  yaw-only leg for that reason.
 *
 *  Pure: no DSR, no Qt. MONITOR-ONLY (Task 4) until Task 6 validates application.
 */
#pragma once
#include "calibration_estimator.h"
#include "mount_lidar_pair.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace rc::joint
{
    /// eps_yaw = k * psi_h. eps acts on the odometry heading (room_concept.cpp: theta_mid + yaw_offset())
    /// and the localiser reads theta_true + psi_h, so the closed loop cancels iff eps = -psi_h
    /// (Fable review 1b). ⚠ TO BE CONFIRMED LIVE BY TASK 6 CHECK 2 (the yaw-only leg).
    inline constexpr float kEpsPerLidarYaw = -1.f;
    /// y_cam = psi_c + k * psi_h: the pair's yaw column gives p = psi_c - psi_h (plan Task 3; review 1b).
    inline constexpr float kCamPerLidarYaw = -1.f;
    inline constexpr int   kCamYawRow = 2;           ///< rc::camcal::P_YAW

    struct CameraBlock
    {
        std::string name;
        rc::mount::MarginalInfo info;
        /// The LiDAR yaw correction (rad, psi_app) the camera evidence is currently referenced to
        /// (plan Task 5). When LidarMountApply rotates the helios points by psi_app, a pair measures
        /// y = psi_c - (psi_h - psi_app), and the accumulated evidence is re-anchored to that frame
        /// (Accum::rebase). The motion block's eps stays a TOTAL (through p_applied), so solve() refers
        /// the camera data back to the ORIGINAL LiDAR frame: b_data -= H_data * psi_app * e_yaw.
        double lidar_yaw_applied = 0.0;
    };
    struct Solution
    {
        Eigen::VectorXd value, sigma;           ///< size P_COUNT + 4*N; camera entries in BODY frame
        Eigen::MatrixXd cov;
        std::vector<std::string> names;         ///< "k_v", ..., "lever_y", "<cam>.pitch", "<cam>.height", "<cam>.yaw", "<cam>.dt"
        bool ok = false;
    };

    inline Solution solve(const rc::calib::BatchEstimator::Information &m, const std::vector<CameraBlock> &cams)
    {
        constexpr int NM = rc::calib::P_COUNT;
        const int n = NM + 4 * static_cast<int>(cams.size());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        H.topLeftCorner(NM, NM) = m.H.cast<double>();
        b.head(NM)              = m.b.cast<double>();

        Solution s;
        for (int i = 0; i < NM; ++i) s.names.emplace_back(rc::calib::param_name(i));
        for (std::size_t c = 0; c < cams.size(); ++c)
        {
            const auto &ci = cams[c].info;
            for (const char *p : {"pitch", "height", "yaw", "dt"}) s.names.push_back(cams[c].name + "." + p);
            const int o = NM + 4 * static_cast<int>(c);
            // y_rel = A theta: identity on the camera's own columns, + coupling on eps_yaw for the yaw row.
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, n);
            A.block(0, o, 4, 4).setIdentity();
            A(kCamYawRow, rc::calib::P_EPS_YAW) = static_cast<double>(kCamPerLidarYaw / kEpsPerLidarYaw);
            // No (usable) evidence: a unit-information prior at zero, only so H is never singular. It
            // carries no coupling and leaves every other column exactly as it was without this camera.
            if (not ci.ok)
            { H.block(o, o, 4, 4) += Eigen::Matrix4d::Identity(); continue; }
            // DATA on the relative mount (coupled through eps); PRIOR on the camera's own body-frame mount.
            Eigen::Vector4d ref = Eigen::Vector4d::Zero();
            ref[kCamYawRow] = cams[c].lidar_yaw_applied;
            H += A.transpose() * ci.H_data * A;
            b += A.transpose() * (ci.b_data - ci.H_data * ref);
            H.block(o, o, 4, 4) += ci.H_prior;
            b.segment(o, 4)     += ci.b_prior;
        }
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() != Eigen::Success) return s;
        s.value = ldlt.solve(b);
        s.cov   = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
        s.sigma = s.cov.diagonal().cwiseMax(0.0).cwiseSqrt();
        s.ok    = s.value.allFinite() and s.cov.allFinite();
        return s;
    }
}
