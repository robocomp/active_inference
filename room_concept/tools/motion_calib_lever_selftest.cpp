// Offline check of the MotionCalibrator side of the joint calibration (plan 2026-10-05 Tasks 1 and 5):
//   - the lever covariates are accumulated PER CORRECTION (sum sin / sum 1-cos of the rotation since the
//     previous correction), not on the episode's net turn;
//   - the lever is estimated but does NOT act (p_applied lever = 0) unless apply_lever;
//   - lidar_side_yaw zeroes the odometry-side yaw_offset() while acting(P_EPS_YAW) -- and so p_applied --
//     is unchanged.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/motion_calib_lever_selftest.cpp -o /tmp/motion_calib_lever_selftest && /tmp/motion_calib_lever_selftest
#include "../src/motion_calibration.h"
#include <charconv>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace rc::calib;
static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }

namespace
{
    std::mt19937 rng(7);
    float noise(float s) { return std::normal_distribution<float>(0.f, s)(rng); }

    // Drive: spins (corrected every `every` cycles) interleaved with straights. Residuals per correction:
    // lever (I - R(dth_k)^T) * L with dth_k the rotation since the previous correction, eps by distance.
    // ⚠ OPEN-LOOP generator: residuals do not shrink when a parameter starts acting, so any parameter
    // that ACTS here (eps with apply=true) is over-read through the p_applied undo -- its value is not
    // graded. The lever is graded only where it does NOT act.
    void drive(MotionCalibrator &mc, float Lx, float Ly, float eps, int every, int laps)
    {
        for (int lap = 0; lap < laps; ++lap)
        {
            const bool spin = lap % 2 == 0;
            float since = 0.f;
            for (int k = 0; k < 40; ++k)
            {
                const float dth = spin ? (lap % 4 == 0 ? 0.05f : -0.05f) : 0.f;
                const float dfw = spin ? 0.f : 0.04f;
                since += dth;
                const bool corrected = (k % every) == every - 1;
                float r_lat = 0.f, r_fwd = 0.f;
                if (corrected)
                {
                    const float s = std::sin(since), c = 1.f - std::cos(since);
                    r_lat = c * Lx - s * Ly - eps * dfw * static_cast<float>(every) + noise(0.0005f);
                    r_fwd = s * Lx + c * Ly + noise(0.0005f);
                    since = 0.f;
                }
                HeadingCovariates hc; hc.th_gyro = dth; hc.t_gyro = 0.05f;
                mc.observe(dfw, 0.f, dth, r_fwd, r_lat, 0.f, 1e-6f, 1e-6f, corrected, 0.01f, 0.05f, hc);
            }
        }
    }

    // Last E row of a saved window: returns {d_theta, lever_s, lever_c, p_applied[P_LEVER_X]}.
    std::vector<float> last_row(const std::string &path)
    {
        std::ifstream f(path);
        std::string line, last;
        while (std::getline(f, line)) if (not line.empty() and line[0] == 'E') last = line;
        std::vector<float> v;
        const char *p = last.data() + 2, *end = last.data() + last.size();
        while (p < end)
        {
            float x = 0.f;
            const auto [next, ec] = std::from_chars(p, end, x);
            if (ec != std::errc{}) break;
            v.push_back(x);
            p = (next < end and *next == ',') ? next + 1 : end;
        }
        if (v.size() != 13 + P_COUNT + 2) return {};
        return {v[2], v[13 + P_COUNT], v[14 + P_COUNT], v[13 + P_LEVER_X]};
    }
}

int main()
{
    Config cfg;
    cfg.enabled = true; cfg.apply = true; cfg.apply_mask = -1;
    cfg.episode_min_rot = 0.8f; cfg.episode_min_trans = 1.0f;
    cfg.rot_model_sigma = 0.f;            // isolate the lever (the model term is tested by its own switch)
    const float Lx = 0.03f, Ly = -0.02f, eps = 0.01f;

    {   // A: per-correction covariates. Corrections every 3 cycles of 0.05 rad -> dth_k = 0.15 rad.
        MotionCalibrator mc; mc.configure(cfg);
        drive(mc, Lx, Ly, eps, 3, 2);
        const std::string path = "/tmp/motion_calib_lever_selftest.csv";
        mc.save_state(path);
        const auto r = last_row(path);
        check(r.size() == 4, "A: a current-format episode row was written");
        if (r.size() == 4)
        {
            // a spin episode: lever_s / d_theta = sin(0.15)/0.15, lever_c / d_theta = (1-cos 0.15)/0.15
            // -- NOT sin(net)/net. Find a spin row: the straight laps write d_theta 0.
            std::printf("  A: last row d_theta %+.4f lever_s %+.5f lever_c %+.6f\n", r[0], r[1], r[2]);
        }
        // Re-run with only spins to read a spin row.
        MotionCalibrator ms; ms.configure(cfg);
        drive(ms, Lx, Ly, eps, 3, 1);
        ms.save_state(path);
        const auto q = last_row(path);
        if (q.size() == 4 and q[0] != 0.f)
        {
            std::printf("  A: spin row d_theta %+.4f lever_s/d_theta %.5f (expect %.5f) lever_c/d_theta %.5f (expect %.5f)\n",
                        q[0], q[1] / q[0], std::sin(0.15f) / 0.15f, q[2] / q[0], (1.f - std::cos(0.15f)) / 0.15f);
            check(std::abs(q[1] / q[0] - std::sin(0.15f) / 0.15f) < 1e-4f and
                  std::abs(q[2] / q[0] - (1.f - std::cos(0.15f)) / 0.15f) < 1e-4f,
                  "A: lever covariates accumulated PER CORRECTION (not on the net turn)");
        }
        else check(false, "A: a spin episode row was written");
    }
    {   // B: estimated, informed, but NOT acting without apply_lever; acting with it.
        MotionCalibrator mc; mc.configure(cfg);
        drive(mc, Lx, Ly, eps, 1, 40);
        const auto &s = mc.last_solve();
        std::printf("  B: lever x %+.4f +- %.4f (truth %+.4f) | y %+.4f +- %.4f (truth %+.4f) | eps %+.4f\n",
                    s.value[P_LEVER_X], s.sigma[P_LEVER_X], Lx, s.value[P_LEVER_Y], s.sigma[P_LEVER_Y], Ly, s.value[P_EPS_YAW]);
        check(mc.taught(P_LEVER_X) and mc.taught(P_LEVER_Y), "B: lever informed from spins");
        check(std::abs(s.value[P_LEVER_X] - Lx) < 3.f * s.sigma[P_LEVER_X] + 1e-3f
              and std::abs(s.value[P_LEVER_Y] - Ly) < 3.f * s.sigma[P_LEVER_Y] + 1e-3f, "B: lever recovered from live-style accumulation");
        check(not mc.acting(P_LEVER_X) and not mc.acting(P_LEVER_Y), "B: lever NOT acting without apply_lever");
        const std::string path = "/tmp/motion_calib_lever_selftest.csv";
        mc.save_state(path);
        const auto r = last_row(path);
        check(r.size() == 4 and r[3] == 0.f, "B: p_applied lever recorded as 0 while not acting");

        Config c2 = cfg; c2.apply_lever = true;
        MotionCalibrator ma; ma.configure(c2);
        drive(ma, Lx, Ly, eps, 1, 40);
        check(ma.acting(P_LEVER_X) and ma.acting(P_LEVER_Y), "B: lever acting with apply_lever");
    }
    {   // C: lidar_side_yaw -- yaw_offset() 0 while acting(P_EPS_YAW) stays true (p_applied keeps it).
        Config c3 = cfg; c3.lidar_side_yaw = true;
        MotionCalibrator mc; mc.configure(c3);
        drive(mc, Lx, Ly, eps, 1, 40);
        std::printf("  C: eps %+.5f +- %.5f taught %d acting %d yaw_offset %+.5f\n", mc.last_solve().value[P_EPS_YAW],
                    mc.last_solve().sigma[P_EPS_YAW], int(mc.taught(P_EPS_YAW)), int(mc.acting(P_EPS_YAW)), mc.yaw_offset());
        check(mc.acting(P_EPS_YAW) and mc.yaw_offset() == 0.f, "C: lidar_side_yaw: eps acting (p_applied) but yaw_offset() == 0");
        MotionCalibrator mo; mo.configure(cfg);
        drive(mo, Lx, Ly, eps, 1, 40);
        check(mo.acting(P_EPS_YAW) and mo.yaw_offset() != 0.f, "C: without it the odometry side applies the yaw");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
