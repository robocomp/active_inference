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
#include "mount_factors.h"
#include "mount_lidar_pair.h"
#include <array>
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

    /* ── r2: THE MOUNTS, DECOUPLED FROM THE MOTION CALIBRATOR (plan 2026-10-08, Task 5; r2.2 items 2, 7) ──────
     *  theta = [ helios 6 | bpearl 6 | cam_1 4 | ... ], each LiDAR a Pose6 error (dx, dy, dz, droll, dpitch, dyaw)
     *  in the BODY frame, rotation about ITS OWN sensor origin (lidar_mount.h). Each factor measures a known
     *  linear map of theta (y_f = A_f theta) and contributes A_f^T H_f A_f / A_f^T b_f -- its DATA part only; ONE
     *  prior per physical parameter is added here (helios_prior_sigma, bpearl_prior_sigma, the cameras' own).
     *    kinematic (KinematicMount)  (dx, dy, psi) of the BAND-EFFECTIVE helios, BODY-origin planar form (the
     *        2-D localiser absorbs z = X o M_eff). A helios tilt shifts every band point by (w_y dz, -w_x dz),
     *        dz = band_dz = z_band - z_helios, i.e. it IS a planar shift of the 2-D scan:
     *          dx_k = h.dx + band_dz h.dpitch - (J s_h)_x h.dyaw,   dy_k = h.dy - band_dz h.droll - (J s_h)_y h.dyaw,
     *          psi_k = h.dyaw,     J s = (-s_y, s_x): the Pose6 (sensor-origin) -> Planar (body-origin) map.
     *        ⚠ ADDITION to the plan: the same tilt->band coupling r2.2 item 7 pins for the bpearl acts on the
     *        kinematic lever (it is the same 2-D scan), so it is modelled here too. It is REQUIRED for item 7 to
     *        be right: with item 7's bpearl coupling alone the helios-tilt arm moves the BPEARL by ~8 mm, and with
     *        neither it moves the HELIOS planar instead (joint_mounts_selftest (d1), (d2)).
     *    floor (FloorPlaneMount)     (droll, dpitch, dz) of the bpearl, absolute (gauge G4).
     *    vert_helios (VerticalMount HeliosTilt)  helios (droll, dpitch), absolute; its other rows carry 0.
     *    vert_bpearl (VerticalMount BpearlPlanar) bpearl RELATIVE to the helios-built map, about s_b:
     *          M_rel = M_h,eff^-1 o M_b  =>  rel_t = t_b - t_h,eff - h.dyaw J (s_b - s_h),  rel_yaw = b.dyaw - h.dyaw
     *        with t_h,eff = (h.dx + band_dz h.dpitch, h.dy - band_dz h.droll): r2.2 item 7's coupling
     *        ("bpearl_rel_dx -= w_hy dz, bpearl_rel_dy += w_hx dz"), pinned by joint_mounts_selftest (d), plus
     *        the rotation-centre term h.dyaw J(s_b - s_h) (0.295 m: 5 mm/deg) the review's A = [I_b - I_h] omits.
     *    cameras  y_c = psi_c + kCamPerLidarYaw * psi_h on the yaw row, with psi_h = the helios dyaw column
     *        DIRECTLY (r2.2 item 2: eps_yaw = -psi_h, kEps = -1, so A = kCam = -1 -- NOT kCam/kEps again).
     *  SINGLE OWNER: the r1 motion block (eps_yaw, lever) is NOT an input -- it would double-anchor the helios
     *  (review §4(c)); joint_mounts_selftest (e) pins that it cannot reach this solve.
     *  GAUGE: helios dz is informed by nothing here (ceiling height is a nuisance, the floor is grazing) and
     *  stays at its prior -- the report prints it as "prior". The helios-bpearl vertical offset is therefore NOT
     *  calibrated by this plan.
     */
    struct MountBlocks
    {
        rc::mountf::Info3 kinematic;      ///< helios (dx, dy, psi), body-origin planar, band-effective
        rc::mountf::Info3 floor;          ///< bpearl (droll, dpitch, dz)
        rc::mountf::Info6 vert_helios;    ///< helios, tilt rows only
        rc::mountf::Info6 vert_bpearl;    ///< bpearl RELATIVE planar (dx, dy, dyaw) about s_bpearl
        Eigen::Vector3d s_helios{0.0, -0.155, 1.1075};   ///< sensor origins in the Shadow frame (r2.2 item 5)
        Eigen::Vector3d s_bpearl{0.0, 0.14, 0.7025};
        double band_dz = 0.6425;          ///< mean height of the helios band (1.5-2.0 m) above the helios origin
        bool kinematic_tilt_coupling = true;   ///< false ONLY for the selftest's control (d2)
        Eigen::Matrix<double, 6, 1> helios_prior_sigma =
            (Eigen::Matrix<double, 6, 1>() << 0.05, 0.05, 0.02, 0.0175, 0.0175, 0.0175).finished();
        Eigen::Matrix<double, 6, 1> bpearl_prior_sigma =
            (Eigen::Matrix<double, 6, 1>() << 0.05, 0.05, 0.02, 0.0175, 0.0175, 0.0175).finished();
    };
    inline constexpr int kHelios = 0, kBpearl = 6, kMountCols = 12;
    /// Per-parameter information share columns: kinematic, floor, vert_helios, vert_bpearl, cameras, prior.
    inline constexpr int kShareCols = 6;
    inline const std::array<const char *, kShareCols> kShareNames{"kin", "floor", "vert_h", "vert_b", "cam", "prior"};
    struct MountSolution
    {
        Eigen::VectorXd value, sigma;
        Eigen::MatrixXd cov;
        Eigen::MatrixXd share;            ///< n x kShareCols: diag(A_f^T H_f A_f)_i / H_ii
        std::vector<std::string> names;   ///< "helios.dx".."helios.dyaw", "bpearl.dx".., "<cam>.pitch".."<cam>.dt"
        bool ok = false;
    };

    inline MountSolution solve_mounts(const MountBlocks &B, const std::vector<CameraBlock> &cams)
    {
        const int n = kMountCols + 4 * static_cast<int>(cams.size());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        MountSolution s;
        s.share = Eigen::MatrixXd::Zero(n, kShareCols);
        for (const char *dev : {"helios", "bpearl"})
            for (const char *p : {"dx", "dy", "dz", "droll", "dpitch", "dyaw"}) s.names.push_back(std::string(dev) + "." + p);
        enum { DX, DY, DZ, ROLL, PITCH, YAW };
        auto add = [&](const Eigen::MatrixXd &A, const Eigen::MatrixXd &Hf, const Eigen::VectorXd &bf, int f)
        {
            const Eigen::MatrixXd At = A.transpose();
            const Eigen::MatrixXd HA = At * Hf * A;
            H += HA;  b += At * bf;
            s.share.col(f) += HA.diagonal();
        };
        // kinematic -> helios (band-effective, body-origin planar)
        {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, n);
            const Eigen::Vector2d Js(-B.s_helios.y(), B.s_helios.x());
            const double dzk = B.kinematic_tilt_coupling ? B.band_dz : 0.0;
            A(0, kHelios + DX) = 1.0; A(0, kHelios + PITCH) =  dzk; A(0, kHelios + YAW) = -Js.x();
            A(1, kHelios + DY) = 1.0; A(1, kHelios + ROLL)  = -dzk; A(1, kHelios + YAW) = -Js.y();
            A(2, kHelios + YAW) = 1.0;
            add(A, B.kinematic.H_data, B.kinematic.b_data, 0);
        }
        // floor -> bpearl (droll, dpitch, dz)
        {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, n);
            A(0, kBpearl + ROLL) = 1.0; A(1, kBpearl + PITCH) = 1.0; A(2, kBpearl + DZ) = 1.0;
            add(A, B.floor.H_data, B.floor.b_data, 1);
        }
        // vertical helios -> helios, identity (only its tilt rows are non-zero)
        {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, n);
            A.block(0, kHelios, 6, 6).setIdentity();
            add(A, B.vert_helios.H_data, B.vert_helios.b_data, 2);
        }
        // vertical bpearl -> RELATIVE to the band-effective helios, about the bpearl origin
        {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, n);
            A.block(0, kBpearl, 6, 6).setIdentity();
            const Eigen::Vector2d d = (B.s_bpearl - B.s_helios).head<2>();
            const Eigen::Vector2d Jd(-d.y(), d.x());
            A(0, kHelios + DX) = -1.0; A(0, kHelios + PITCH) = -B.band_dz; A(0, kHelios + YAW) = -Jd.x();
            A(1, kHelios + DY) = -1.0; A(1, kHelios + ROLL)  =  B.band_dz; A(1, kHelios + YAW) = -Jd.y();
            A(5, kHelios + YAW) = -1.0;
            add(A, B.vert_bpearl.H_data, B.vert_bpearl.b_data, 3);
        }
        // cameras (r1 model, coupling on the helios dyaw column DIRECTLY)
        for (std::size_t c = 0; c < cams.size(); ++c)
        {
            const auto &ci = cams[c].info;
            for (const char *p : {"pitch", "height", "yaw", "dt"}) s.names.push_back(cams[c].name + "." + p);
            const int o = kMountCols + 4 * static_cast<int>(c);
            if (not ci.ok) { H.block(o, o, 4, 4) += Eigen::Matrix4d::Identity(); s.share.block(o, 5, 4, 1).array() += 1.0; continue; }
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, n);
            A.block(0, o, 4, 4).setIdentity();
            A(kCamYawRow, kHelios + YAW) = static_cast<double>(kCamPerLidarYaw);
            Eigen::Vector4d ref = Eigen::Vector4d::Zero();
            ref[kCamYawRow] = cams[c].lidar_yaw_applied;
            add(A, ci.H_data, ci.b_data - ci.H_data * ref, 4);
            H.block(o, o, 4, 4) += ci.H_prior;
            b.segment(o, 4)     += ci.b_prior;
            s.share.block(o, 5, 4, 1) += ci.H_prior.diagonal();
        }
        // ONE prior per LiDAR parameter, at zero (the nominal mount)
        for (int k = 0; k < 6; ++k)
        {
            const double ph = 1.0 / (B.helios_prior_sigma[k] * B.helios_prior_sigma[k]);
            const double pb = 1.0 / (B.bpearl_prior_sigma[k] * B.bpearl_prior_sigma[k]);
            H(kHelios + k, kHelios + k) += ph;  s.share(kHelios + k, 5) += ph;
            H(kBpearl + k, kBpearl + k) += pb;  s.share(kBpearl + k, 5) += pb;
        }
        for (int i = 0; i < n; ++i) if (H(i, i) > 0.0) s.share.row(i) /= H(i, i);
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() != Eigen::Success) return s;
        s.value = ldlt.solve(b);
        s.cov   = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
        s.sigma = s.cov.diagonal().cwiseMax(0.0).cwiseSqrt();
        s.ok    = s.value.allFinite() and s.cov.allFinite();
        return s;
    }
}
