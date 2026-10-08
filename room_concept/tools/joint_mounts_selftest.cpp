// Offline check of rc::joint::solve_mounts (plan docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md
// Task 5, r2.2 items 2 and 7): helios 6 + bpearl 6 + cameras 4*N from the mount-factor blocks. Blocks are
// built DIRECTLY from the physics each factor measures (no sweeps), so this tests only the fusion algebra:
// the frame maps, the tilt coupling, the camera yaw coupling and what stays at its prior.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/joint_mounts_selftest.cpp -o /tmp/joint_mounts_selftest && /tmp/joint_mounts_selftest
#include "../src/joint_calibration.h"
#include "../src/joint_calib_monitor.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc;
using V6 = Eigen::Matrix<double, 6, 1>;
static constexpr double kDegD = M_PI / 180.0;

/// The physics each factor measures, written out INDEPENDENTLY of solve_mounts (so a sign slip in one is
/// not mirrored in the other). h, b: the true helios / bpearl Pose6 errors (dx, dy, dz, droll, dpitch, dyaw).
struct Truth
{
    V6 h = V6::Zero(), b = V6::Zero();
    Eigen::Vector3d sh{0.0, -0.155, 1.1075}, sb{0.0, 0.14, 0.7025};
    double dz_band = 0.6425;
    /// the 2-D localiser absorbs the band-effective helios mount, body-origin form (z = X o M_eff)
    Eigen::Vector3d kinematic() const
    {
        const double tx = h[0] + dz_band * h[4], ty = h[1] - dz_band * h[3], psi = h[5];
        // Pose6 rotates about s, the localiser's planar M about the body origin: t_planar = t6 - psi J s
        return {tx - psi * (-sh.y()), ty - psi * sh.x(), psi};
    }
    /// the bpearl against the helios-built map: M_rel = M_h,eff^-1 o M_b, about the bpearl origin
    V6 bpearl_rel() const
    {
        const double thx = h[0] + dz_band * h[4], thy = h[1] - dz_band * h[3];
        const Eigen::Vector2d d = (sb - sh).head<2>();
        V6 r = V6::Zero();
        r[0] = b[0] - thx - h[5] * (-d.y());
        r[1] = b[1] - thy - h[5] * d.x();
        r[5] = b[5] - h[5];
        return r;
    }
};
static mountf::Info3 info3(const Eigen::Vector3d &y, const Eigen::Vector3d &sd)
{
    mountf::Info3 i; i.H_data = sd.cwiseAbs2().cwiseInverse().asDiagonal(); i.b_data = i.H_data * y; i.n = 1000; return i;
}
static mountf::Info6 info6(const V6 &y, const V6 &sd_or_zero)
{
    mountf::Info6 i;
    for (int k = 0; k < 6; ++k) if (sd_or_zero[k] > 0.0) i.H_data(k, k) = 1.0 / (sd_or_zero[k] * sd_or_zero[k]);
    i.b_data = i.H_data * y; i.n = 1000; return i;
}
static joint::MountBlocks blocks(const Truth &t, double sd_scale = 1.0)
{
    joint::MountBlocks B;
    B.s_helios = t.sh; B.s_bpearl = t.sb; B.band_dz = t.dz_band;
    B.kinematic = info3(t.kinematic(), sd_scale * Eigen::Vector3d(1e-4, 1e-4, 1e-5));
    B.floor     = info3(Eigen::Vector3d(t.b[3], t.b[4], t.b[2]), sd_scale * Eigen::Vector3d(1e-5, 1e-5, 1e-4));
    V6 vh = V6::Zero(); vh[3] = 1e-5 * sd_scale; vh[4] = 1e-5 * sd_scale;             // helios: tilt only
    B.vert_helios = info6(t.h, vh);
    V6 vb = V6::Zero(); vb[0] = vb[1] = 1e-4 * sd_scale; vb[5] = 1e-5 * sd_scale;    // bpearl: relative planar only
    B.vert_bpearl = info6(t.bpearl_rel(), vb);
    return B;
}
static int col(const joint::MountSolution &s, const std::string &n)
{
    for (std::size_t i = 0; i < s.names.size(); ++i) if (s.names[i] == n) return int(i);
    return -1;
}

int main()
{
    std::printf("joint_mounts_selftest: rc::joint::solve_mounts\n");
    {   // (a) a planted helios planar + bpearl planar: the bpearl's ABSOLUTE planar = relative + the helios (kinematic)
        Truth t; t.h << 0.03, -0.02, 0.0, 0.0, 0.0, 1.0 * kDegD;  t.b << 0.01, 0.02, 0.0, 0.0, 0.0, 0.5 * kDegD;
        const auto s = joint::solve_mounts(blocks(t), {});
        check(s.ok, "(a) solve ok");
        std::printf("    helios dx %+.2f dy %+.2f mm yaw %+.3f deg | bpearl dx %+.2f dy %+.2f mm yaw %+.3f deg\n",
                    s.value[col(s, "helios.dx")] * 1e3, s.value[col(s, "helios.dy")] * 1e3, s.value[col(s, "helios.dyaw")] / kDegD,
                    s.value[col(s, "bpearl.dx")] * 1e3, s.value[col(s, "bpearl.dy")] * 1e3, s.value[col(s, "bpearl.dyaw")] / kDegD);
        bool ok = true;
        for (int k : {0, 1, 5}) ok = ok and std::abs(s.value[k] - t.h[k]) < 2e-4 and std::abs(s.value[6 + k] - t.b[k]) < 4e-4;
        check(ok, "(a) helios (dx, dy, dyaw) and bpearl ABSOLUTE (dx, dy, dyaw) recovered (bpearl = relative + helios)");
        // the body-origin -> sensor-origin map: helios.dx is NOT the kinematic dx (which reads t6 - 0.155 psi)
        check(std::abs(t.kinematic()[0] - t.h[0]) > 2e-3 and std::abs(s.value[0] - t.h[0]) < 2e-4,
              "(a) the kinematic block's BODY-origin lever is mapped onto the sensor-origin Pose6 (2.7 mm/deg)");
    }
    {   // (b) camera body-frame yaw = relative + helios yaw (r1 sign): y_rel = psi_c + kCam * psi_h, kCam = -1
        Truth t; t.h << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 * kDegD;
        const double psi_c = -0.4 * kDegD;
        joint::CameraBlock cam{"ricoh", {}};
        cam.info.H_prior = Eigen::Vector4d(0.0035, 0.010, 1.0, 1.0).cwiseInverse().cwiseAbs2().asDiagonal();   // weak yaw prior
        cam.info.H_data.setZero(); cam.info.H_data(2, 2) = 1.0 / (1e-5 * 1e-5);
        cam.info.b_prior.setZero();
        cam.info.b_data.setZero(); cam.info.b_data[2] = cam.info.H_data(2, 2) * (psi_c + joint::kCamPerLidarYaw * t.h[5]);
        cam.info.ok = true;
        const auto s = joint::solve_mounts(blocks(t), {cam});
        const int iy = col(s, "ricoh.yaw");
        std::printf("    ricoh.yaw body %+.4f deg (truth %+.4f) | relative measured %+.4f deg\n", s.value[iy] / kDegD, psi_c / kDegD,
                    (psi_c - t.h[5]) / kDegD);
        check(iy >= 0 and std::abs(s.value[iy] - psi_c) < 1e-4, "(b) camera body-frame yaw = relative + helios yaw (kCam = -1 on the psi_h column, DIRECTLY)");
    }
    {   // (c) helios dz: no factor informs it -> it stays at its prior, and the report must say "prior"
        Truth t; t.h << 0.03, -0.02, 0.0, 0.7 * kDegD, -0.4 * kDegD, 1.0 * kDegD;
        const auto B = blocks(t);
        const auto s = joint::solve_mounts(B, {});
        const int iz = col(s, "helios.dz");
        std::printf("    helios.dz %+.4f +- %.4f m (prior sd %.4f), information share prior = %.3f\n", s.value[iz], s.sigma[iz],
                    B.helios_prior_sigma[2], s.share(iz, joint::kShareCols - 1));
        check(std::abs(s.sigma[iz] - B.helios_prior_sigma[2]) < 1e-9 and std::abs(s.value[iz]) < 1e-12,
              "(c) helios dz stays EXACTLY at its prior");
        check(std::abs(s.share(iz, joint::kShareCols - 1) - 1.0) < 1e-12, "(c) its information share is 100 % prior");
    }
    {   // (d) r2.2 item 7: plant a helios TILT only. The map walls AND the kinematic lever are where the tilt
        //     put them at band height; with the coupling modelled on BOTH blocks the bpearl planar stays at 0 and
        //     the helios planar stays at 0. Controls on the SAME data: (d1) no coupling anywhere -> the HELIOS
        //     planar reads the tilt's band shift; (d2) item 7's coupling on the bpearl block ALONE (the review's
        //     text) -> the BPEARL moves. Both couplings are needed; joint_calibration.h records the addition.
        Truth t; t.h << 0.0, 0.0, 0.0, 0.7 * kDegD, -0.4 * kDegD, 0.0;
        const auto B = blocks(t);
        const auto s = joint::solve_mounts(B, {});
        auto B1 = B; B1.band_dz = 0.0;
        const auto c1 = joint::solve_mounts(B1, {});
        auto B2 = B; B2.kinematic_tilt_coupling = false;
        const auto c2 = joint::solve_mounts(B2, {});
        std::printf("    tilt plant: bpearl dx %+.2f dy %+.2f mm, helios dx %+.2f dy %+.2f mm\n", s.value[6] * 1e3, s.value[7] * 1e3,
                    s.value[0] * 1e3, s.value[1] * 1e3);
        std::printf("    (d1) no coupling:            bpearl dx %+.2f dy %+.2f mm, helios dx %+.2f dy %+.2f mm\n", c1.value[6] * 1e3,
                    c1.value[7] * 1e3, c1.value[0] * 1e3, c1.value[1] * 1e3);
        std::printf("    (d2) bpearl coupling only:   bpearl dx %+.2f dy %+.2f mm, helios dx %+.2f dy %+.2f mm\n", c2.value[6] * 1e3,
                    c2.value[7] * 1e3, c2.value[0] * 1e3, c2.value[1] * 1e3);
        check(std::abs(s.value[6]) < 3e-4 and std::abs(s.value[7]) < 3e-4 and std::abs(s.value[0]) < 3e-4 and std::abs(s.value[1]) < 3e-4,
              "(d) helios tilt alone: bpearl planar AND helios planar stay at 0 (coupling sign right)");
        check(std::abs(s.value[3] - t.h[3]) < 1e-4 and std::abs(s.value[4] - t.h[4]) < 1e-4, "(d) and the tilt itself is recovered");
        check(std::hypot(c1.value[0], c1.value[1]) > 4e-3, "(d1) control: no coupling -> the helios planar reads ~0.64 m x tilt (the bite)");
        check(std::hypot(c2.value[6], c2.value[7]) > 4e-3, "(d2) control: item 7's bpearl coupling alone -> the bpearl moves (why the kinematic coupling is needed)");
    }
    {   // (e) single owner: the r1 MOTION block's eps/lever never reaches solve_mounts. Run r1's solve() with
        //     two different motion blocks next to solve_mounts on the same cameras: r1 changes, mounts do not.
        Truth t; t.h << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 * kDegD;
        joint::CameraBlock cam{"ricoh", {}};
        cam.info.H_prior = Eigen::Vector4d(0.0035, 0.010, 0.0035, 1.0).cwiseInverse().cwiseAbs2().asDiagonal();
        cam.info.H_data.setZero(); cam.info.H_data(2, 2) = 1e6; cam.info.b_prior.setZero();
        cam.info.b_data.setZero(); cam.info.b_data[2] = 1e6 * (-t.h[5]); cam.info.ok = true;
        calib::BatchEstimator::Information m1, m2;
        m1.H.setIdentity(); m1.H_prior.setIdentity(); m1.b.setZero();
        m2 = m1; m2.H(calib::P_EPS_YAW, calib::P_EPS_YAW) += 1e6f; m2.b[calib::P_EPS_YAW] = 1e6f * 0.03f;
        m2.H(calib::P_LEVER_X, calib::P_LEVER_X) += 1e6f; m2.b[calib::P_LEVER_X] = 1e6f * 0.05f;
        const auto r1a = joint::solve(m1, {cam}), r1b = joint::solve(m2, {cam});
        const auto sa = joint::solve_mounts(blocks(t), {cam}), sb = joint::solve_mounts(blocks(t), {cam});
        check(std::abs(r1a.value[calib::P_COUNT + 2] - r1b.value[calib::P_COUNT + 2]) > 1e-4,
              "(e) r1 solve(): the motion block's eps moves the camera yaw (the coupling it was built on)");
        check((sa.value - sb.value).norm() == 0.0, "(e) solve_mounts has NO motion-block input: eps/lever cannot double-anchor the helios");
    }
    {   // (f) the monitor row: names, sigma and per-factor information shares, through the classic locale
        Truth t; t.h << 0.03, -0.02, 0.0, 0.7 * kDegD, -0.4 * kDegD, 1.0 * kDegD;
        const std::string dir = (std::filesystem::temp_directory_path() / "joint_mounts_selftest_csv").string();
        std::filesystem::remove_all(dir);
        joint::MountMonitor mon(dir);
        joint::MountMonitor::Extra ex; ex.yaw_offset = 0.0;
        const auto s = mon.observe(1234, blocks(t), {}, ex);
        std::ifstream f(mon.path());
        std::string head, row;
        std::getline(f, head); std::getline(f, row);
        check(s.ok and head.starts_with("ts_ms,yaw_offset,") and head.find("helios.dz_sd") != std::string::npos
                  and head.find("bpearl.dyaw_sh_vert_b") != std::string::npos and row.starts_with("1234,0,"),
              "(f) mounts CSV: header carries value, sd and per-factor shares; yaw_offset logged per row");
        // shares of an informed parameter sum to 1
        const int ix = col(s, "helios.dx");
        // (the bpearl-relative block informs helios.dx too -- it measures b - h -- so with equally precise blocks
        //  the diagonal share splits ~50/50; the shares are a diagnostic, not "who determines it")
        check(std::abs(s.share.row(ix).sum() - 1.0) < 1e-9 and s.share(ix, 0) > 0.45 and s.share(ix, 5) < 1e-3,
              "(f) helios.dx information: kinematic share > 45 %, prior < 0.1 %, shares sum to 1");
        const int ir = col(s, "bpearl.droll");
        check(s.share(ir, 1) > 0.99, "(f) bpearl.droll information comes from the floor factor");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
