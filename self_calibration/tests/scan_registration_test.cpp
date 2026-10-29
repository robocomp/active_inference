// scan_registration_test -- synthetic (notched 4x6 m room, 0.25 deg rays, 1 cm range noise), no DDS/DSR.
//   make -C build scan_registration_test && ../bin/scan_registration_test
#include "../../common/sensor_calib/scan_registration.h"
#include "synthetic_room.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

using namespace sc::calib;
using namespace sc::calib::testing;

static int g_fail = 0;
#define CHECK(c, ...) do { if (not (c)) { ++g_fail; std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static constexpr double kStep = 0.25 * std::numbers::pi / 180.0;
static constexpr double kDeg = std::numbers::pi / 180.0;

// ref at the room origin (identity), cur at `truth`: the truth pose of CUR in REF is `truth` itself.
static ScanRegResult run(const Room2D& room, const Eigen::Vector3d& truth, const Eigen::Vector3d& init, unsigned seed,
                         double clutter = 0.0, double max_range = 1e9, int normal_k = ScanRegConfig{}.normal_k)
{
    std::mt19937 rng(seed);
    const auto ref = raycast(room, {0, 0, 0}, kStep, 0.01, rng, max_range);
    auto cur = raycast(room, truth, kStep, 0.01, rng, max_range);
    if (clutter > 0.0)
    {
        std::uniform_real_distribution<float> ux(-2.f, 2.f), uy(-3.f, 3.f);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        for (auto& p : cur) if (u01(rng) < clutter) p = {ux(rng), uy(rng)};
    }
    ScanRegConfig c; c.normal_k = normal_k; return register_scans_2d(ref, cur, init, c);
}
static void check_pose(const char* nm, const ScanRegResult& r, const Eigen::Vector3d& truth)
{
    const double ex = r.xyt.x() - truth.x(), ey = r.xyt.y() - truth.y(), et = (r.xyt.z() - truth.z()) / kDeg;
    std::printf("%-20s ok=%d err=(%.2f mm, %.2f mm, %.4f deg) iters=%d share=%.3f\n", nm, int(r.ok), ex * 1e3, ey * 1e3, et, r.iters, r.inlier_share);
    CHECK(r.ok, "%s not ok", nm);
    CHECK(std::abs(ex) < 5e-3 and std::abs(ey) < 5e-3, "%s xy err %.2f %.2f mm", nm, ex * 1e3, ey * 1e3);
    CHECK(std::abs(et) < 0.05, "%s yaw err %.4f deg", nm, et);
}

int main()
{
    const Room2D room = notched_room();
    {
        const Eigen::Vector3d tr(0.30, 0.05, 2.0 * kDeg);
        check_pose("known_motion", run(room, tr, Eigen::Vector3d::Zero(), 7), tr);
    }
    {
        const Eigen::Vector3d tr(1.2, -0.1, 15.0 * kDeg);
        check_pose("init_from_odometry", run(room, tr, tr + Eigen::Vector3d(0.05, 0.05, 3.0 * kDeg), 7), tr);
    }
    {
        const Eigen::Vector3d tr(0.30, 0.05, 2.0 * kDeg);
        const auto r = run(room, tr, Eigen::Vector3d::Zero(), 7, 0.15);
        check_pose("outliers_do_not_bias", r, tr);
        CHECK(r.inlier_share >= 0.75 and r.inlier_share <= 0.95, "inlier_share %.3f", r.inlier_share);
    }
    {
        const Eigen::Vector3d tr(0.5, 0.0, 0.0);
        const auto r = run(corridor(200.0, 2.0), tr, Eigen::Vector3d::Zero(), 7, 0.0, 15.0);
        const double sx = std::sqrt(r.cov(0, 0)), sy = std::sqrt(r.cov(1, 1)), ex = r.xyt.x() - tr.x();
        std::printf("degenerate_corridor  sd_x=%.3f sd_y=%.5f ex=%.3f m ey=%.2f mm\n", sx, sy, ex, (r.xyt.y() - tr.y()) * 1e3);
        CHECK(sx > 0.05, "along-corridor sd %.4f m not wider than the 5 cm mount prior", sx);
        CHECK(std::abs(ex) < 3.0 * sx, "x err %.3f beyond 3 reported sigma %.3f", ex, sx);
        CHECK(sx > 10.0 * sy, "along-corridor sd %.4f not >> cross sd %.5f", sx, sy);
        CHECK(std::abs(r.xyt.y() - tr.y()) < 5e-3, "y err %.2f mm", (r.xyt.y() - tr.y()) * 1e3);
    }
    {
        // seed dependence of the corridor guarantee: 100 noise draws (reported, see below)
        const Eigen::Vector3d tr(0.5, 0.0, 0.0); int N = 100, wide = 0, covered = 0;
        for (int s = 0; s < N; ++s)
        {
            const auto r = run(corridor(200.0, 2.0), tr, Eigen::Vector3d::Zero(), 100u + unsigned(s), 0.0, 15.0);
            const double sx = std::sqrt(r.cov(0, 0));
            wide += sx > 0.05; covered += std::abs(r.xyt.x() - tr.x()) < 3.0 * sx;
        }
        std::printf("corridor over %d seeds: sd_x > 5 cm in %d, x within 3 sigma in %d\n", N, wide, covered);
        CHECK(wide >= 95, "wide %d/%d", wide, N);
        CHECK(covered >= 97, "covered %d/%d", covered, N);
    }
    {   // honest covariance: empirical sd / mean reported sd for x, y, yaw, in two geometries, 200 seeds each
        const Eigen::Vector3d trs[2] = {{0.30, 0.05, 2.0 * kDeg}, {1.2, -0.3, 10.0 * kDeg}};
        for (int sc = 0; sc < 2; ++sc)
        {
            const int N = 200; Eigen::Vector3d s1 = Eigen::Vector3d::Zero(), s2 = s1, rep = s1;
            for (int s = 0; s < N; ++s)
            {
                const auto r = run(room, trs[sc], sc == 0 ? Eigen::Vector3d(Eigen::Vector3d::Zero()) : Eigen::Vector3d(trs[sc] + Eigen::Vector3d(0.05, 0.05, 3.0 * kDeg)), 1000u + unsigned(s));
                const Eigen::Vector3d e = r.xyt - trs[sc];
                s1 += e; s2 += e.cwiseProduct(e); rep += r.cov.diagonal().cwiseSqrt();
            }
            const char* nm[3] = {"x", "y", "yaw"};
            for (int i = 0; i < 3; ++i)
            {
                const double m = s1[i] / N, sd = std::sqrt(s2[i] / N - m * m), q = sd / (rep[i] / N);
                std::printf("honest_sigma sc%d %-3s empirical sd=%.3e reported=%.3e ratio=%.3f\n", sc, nm[i], sd, rep[i] / N, q);
                CHECK(q >= 0.7 and q <= 1.5, "scenario %d %s ratio %.3f", sc, nm[i], q);
            }
        }
    }
    if (g_fail) std::printf("scan_registration_test: %d FAILURES\n", g_fail);
    else std::printf("scan_registration_test: ALL PASS\n");
    return g_fail ? 1 : 0;
}
