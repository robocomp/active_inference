// Offline check of rc::joint::solve (plan 2026-10-05 Task 3): the motion and camera blocks fused
// through the shared helios yaw. Blocks are built DIRECTLY (no episodes, no pairs), so this tests only
// the fusion algebra.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/joint_calib_selftest.cpp -o /tmp/joint_calib_selftest && /tmp/joint_calib_selftest
#include "../src/joint_calibration.h"
#include "../src/joint_calib_monitor.h"
#include <cstdio>
#include <fstream>
static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc;

int main()
{
    const double psi_h = 0.010, psi_c = -0.004;          // truth, rad, body frame
    const double eps   = joint::kEpsPerLidarYaw * psi_h; // what the motion block sees
    const double y_rel = psi_c + joint::kCamPerLidarYaw * psi_h;   // what the camera block sees

    // Motion block: eps_yaw measured to sigma 0.002, everything else at prior (H = prior only).
    calib::BatchEstimator::Information m;
    m.H.setZero(); m.H_prior.setZero(); m.b.setZero();
    const float sp[calib::P_COUNT] = {0.02f, 0.0175f, 0.02f, 5e-4f, 0.05f, 0.02f, 0.155f, 0.05f, 0.05f};
    for (int i = 0; i < calib::P_COUNT; ++i) m.H_prior(i, i) = m.H(i, i) = 1.f / (sp[i] * sp[i]);
    const float w_eps = 1.f / (0.002f * 0.002f);
    m.H(calib::P_EPS_YAW, calib::P_EPS_YAW) += w_eps;
    m.b[calib::P_EPS_YAW] = w_eps * float(eps);

    // Camera block: relative yaw measured to sigma 0.001, other 3 at prior.
    joint::CameraBlock cam{"ricoh", {}};
    const Eigen::Vector4d unit(0.0035, 0.010, 0.0035, 1.0);
    cam.info.H_prior = unit.cwiseInverse().cwiseAbs2().asDiagonal();
    cam.info.H_data.setZero(); cam.info.H_data(2, 2) = 1.0 / (0.001 * 0.001);
    cam.info.b_prior.setZero();
    cam.info.b_data.setZero(); cam.info.b_data[2] = cam.info.H_data(2, 2) * y_rel;
    cam.info.ok = true;

    const auto s = joint::solve(m, {cam});
    check(s.ok, "joint solve ok");
    const int iy = calib::P_COUNT + 2;   // ricoh.yaw (body frame)
    std::printf("  eps %+.5f (truth %+.5f) | ricoh.yaw body %+.5f (truth %+.5f) +- %.5f | corr %+.3f\n",
                s.value[calib::P_EPS_YAW], eps, s.value[iy], psi_c, s.sigma[iy],
                s.cov(calib::P_EPS_YAW, iy) / (s.sigma[calib::P_EPS_YAW] * s.sigma[iy]));
    // ⚠ DEVIATION from the plan's checks "|yaw - psi_c| < 1e-3" and "sigma > 0.0019": both ignore the
    // camera's own BODY-frame prior (sigma 0.0035 at 0) that r1 introduced. That prior correctly pulls
    // the yaw toward 0 (truth is 1.1 sigma out) and adds information, so a correct solve gives
    // -0.00294 +- 0.00188. Checked instead: (1) the EXACT Gaussian posterior of the two coupled
    // unknowns (cam yaw, eps), every other column being decoupled; (2) recovery of psi_c when the
    // camera prior is weak; (3) the joint sigma exceeds the CHAINED sigma (LiDAR-yaw uncertainty is
    // propagated, not dropped).
    {
        const double w_y = cam.info.H_data(2, 2), w_c = cam.info.H_prior(2, 2);
        const double w_e = double(m.H(calib::P_EPS_YAW, calib::P_EPS_YAW));
        Eigen::Matrix2d H2; H2 << w_y + w_c, w_y, w_y, w_y + w_e;          // A(yaw, eps) = +1
        const Eigen::Vector2d b2(cam.info.b_data[2], cam.info.b_data[2] + double(m.b[calib::P_EPS_YAW]));
        const Eigen::Vector2d x2 = H2.ldlt().solve(b2);
        const double sd2 = std::sqrt(H2.inverse()(0, 0));
        std::printf("  exact 2x2 posterior: cam yaw %+.5f +- %.5f\n", x2[0], sd2);
        check(std::abs(s.value[iy] - x2[0]) < 1e-9 and std::abs(s.sigma[iy] - sd2) < 1e-9,
              "joint == exact Gaussian posterior of (cam yaw, eps)");
    }
    {
        joint::CameraBlock weak = cam; weak.info.H_prior(2, 2) = 1.0;   // camera yaw prior sigma 1 rad
        const auto sw = joint::solve(m, {weak});
        std::printf("  weak camera prior: cam yaw %+.5f (truth %+.5f)\n", sw.value[iy], psi_c);
        check(std::abs(sw.value[iy] - psi_c) < 1e-3, "camera yaw recovered in the BODY frame (weak camera prior)");
    }
    // Chained (today): camera yaw taken as relative + LiDAR assumed perfect => error = psi_h.
    check(std::abs(y_rel - psi_c) > 5e-3, "chained estimate is off by the LiDAR yaw (the defect)");
    // The joint sigma on the body-frame camera yaw must include the eps uncertainty:
    const double chained_sd = std::sqrt((cam.info.H_data + cam.info.H_prior).inverse()(2, 2));
    std::printf("  joint sigma %.5f vs chained sigma %.5f\n", s.sigma[iy], chained_sd);
    check(s.sigma[iy] > 1.5 * chained_sd, "joint sigma > chained sigma (LiDAR-yaw uncertainty propagated)");
    check(s.names[iy] == "ricoh.yaw", "names line up");

    {   // REDUCTION: eps_yaw at its prior only — the precise relative measurement is SHARED between the
        // two mounts in proportion to their prior variances (sigma_c^2 : sigma_h^2).
        calib::BatchEstimator::Information m0 = m;
        m0.H(calib::P_EPS_YAW, calib::P_EPS_YAW) = m0.H_prior(calib::P_EPS_YAW, calib::P_EPS_YAW);
        m0.b.setZero();
        const auto s0 = joint::solve(m0, {cam});
        const double vc = 1.0 / cam.info.H_prior(2, 2), vh = 1.0 / m0.H_prior(calib::P_EPS_YAW, calib::P_EPS_YAW);
        const double expect_c = y_rel * vc / (vc + vh);
        std::printf("  reduction: cam yaw %+.5f (expect %+.5f), eps %+.5f\n", s0.value[iy], expect_c, s0.value[calib::P_EPS_YAW]);
        check(std::abs(s0.value[iy] - expect_c) < 0.1 * std::abs(y_rel), "reduction: split in proportion to the prior variances");
        // Limit sigma_h -> 0 (helios yaw known): the joint must equal the chained answer.
        calib::BatchEstimator::Information m1 = m0;
        m1.H(calib::P_EPS_YAW, calib::P_EPS_YAW) += 1e12f;
        const auto s1 = joint::solve(m1, {cam});
        const Eigen::Matrix4d Hc = cam.info.H_data + cam.info.H_prior;
        const double own = Hc.ldlt().solve(cam.info.b_data + cam.info.b_prior)[2];
        check(std::abs(s1.value[iy] - own) < 1e-6, "limit sigma_h -> 0: joint == chained");
    }
    {   // RE-ANCHORED CAMERA EVIDENCE (plan Task 5): with the LiDAR points rotated by psi_app, the camera's
        // data (rebased to the current LiDAR frame) reads y_rel + psi_app; the block says so through
        // lidar_yaw_applied and the joint must refer it back -- identical solution to the unrotated case.
        const double psi_app = 0.006;
        joint::CameraBlock moved = cam;
        moved.info.b_data[2] += moved.info.H_data(2, 2) * psi_app;   // what Accum::rebase(+psi_app/sigma) does to b_data
        moved.lidar_yaw_applied = psi_app;
        const auto sm = joint::solve(m, {moved});
        check(sm.ok and (sm.value - s.value).cwiseAbs().maxCoeff() < 1e-12,
              "re-anchored camera evidence + lidar_yaw_applied == the unrotated solution");
    }
    {   // A camera block that refused (ok = false) must enter as a unit prior and change nothing else.
        joint::CameraBlock none{"zed", {}};
        const auto s2 = joint::solve(m, {cam, none});
        check(s2.ok and std::abs(s2.value[iy] - s.value[iy]) < 1e-12, "a refused camera block changes nothing");
    }
    {   // MONITOR CSV (plan Task 4): one row per solve, header from Solution::names; graded offline by
        // tools/joint_calib_report.py on the file printed here (argv[1] = output directory).
        joint::Monitor mon("/tmp/joint_calib_selftest_out");
        joint::CameraBlock none{"zed", {}};
        for (int k = 0; k < 3; ++k) mon.observe(1000 * k, 100 + k, m, {cam, none});
        std::ifstream f(mon.path());
        std::string header, row; std::getline(f, header);
        int rows = 0; while (std::getline(f, row)) ++rows;
        std::printf("  monitor wrote %s (%d rows)\n", mon.path().c_str(), rows);
        check(rows == 3 and header.starts_with("ts_ms,episodes,k_v,k_v_sd,") and header.find("ricoh.yaw,ricoh.yaw_sd") != std::string::npos
              and header.ends_with("corr_eps_ricohyaw,corr_eps_zedyaw,chain_ricoh.yaw,chain_zed.yaw"),
              "monitor CSV: header and rows as the report expects");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
