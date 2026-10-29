// helios_mount_test -- Task 4.4: planar Helios mount from registered keyframe pairs. Synthetic (notched 4x6 m room,
// 0.5 deg rays, 5 mm range noise), no DDS/DSR. The sweeps are generated in the TRUE sensor frame (true mount =
// nominal + planted error) and expressed in the robot frame through the NOMINAL mount, the odometry comes from the
// true motion; the stage must read the correction (dx, dy, psi) = -(planted error) (Shadow axes: X lateral, Y forward).
//   make -C build helios_mount_test && ../bin/helios_mount_test
#include "../src/helios_mount_stage.h"
#include "synthetic_room.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

using namespace sc;
using namespace sc::calib;
using namespace sc::calib::testing;

static int g_fail = 0;
#define CHECK(c, ...) do { if (not (c)) { ++g_fail; std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static constexpr double kPi = std::numbers::pi, kDeg = kPi / 180.0;
static constexpr std::int64_t T0 = 1'000'000;

struct Seg { double adv, rot, dur; };
struct Plant { double lx = 0, ly = 0, psi = 0; };   // true sensor vs nominal: lever in Shadow axes (X lateral, Y forward) [m], yaw [rad]
struct Opt { double gyro_bias = 0.0; double gyro_scale = 0.0; bool drop_keyframes = false; };

struct Out { Eigen::Vector3d m, s; long n = 0; double path = 0; std::string line; };

static HarvesterConfig hcfg() { HarvesterConfig c; c.ring_span_s = 10.0; return c; }   // a compute budget: pairs are committed while their sweeps are still held

// Lift 2-D hits to 3-D robot-frame points: wall hits inside the band, plus decoys outside it that must be ignored.
static std::vector<Eigen::Vector3f> lift(const std::vector<Eigen::Vector2f>& hits_h, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uz(1.55f, 1.95f), u(-3.f, 3.f);
    std::vector<Eigen::Vector3f> out;
    for (const auto& h : hits_h) out.emplace_back(-h.y(), h.x(), uz(rng));   // harvester (fwd, left) -> Shadow (X = -left, Y = fwd)
    for (int i = 0; i < 300; ++i) out.emplace_back(u(rng), u(rng), (i % 2) ? 0.3f : 1.0f);   // floor / furniture clutter, below the band
    return out;
}

static Out run(const Room2D& room, const Eigen::Vector3d& start, const std::vector<Seg>& segs, const Plant& pl, const Opt& o,
               unsigned seed, double gyro_bias_sigma = 0.002, double gyro_scale_sigma = 0.01)
{
    HeliosMountConfig c; c.log_dir = ""; c.gyro_bias_sigma = gyro_bias_sigma; c.gyro_scale_sigma = gyro_scale_sigma;
    TrajectoryHarvester h(hcfg());
    HeliosMountStage st(c, hcfg());
    std::mt19937 rng(seed);
    const Eigen::Vector2d l_h(pl.ly, -pl.lx);   // harvester axes: forward = Y, left = -X
    double x = start.x(), y = start.y(), th = start.z(), path = 0;
    std::int64_t t_ms = 0;
    const auto offer = [&]() {
        if (not h.add_keyframe(T0 + t_ms, false)) return;
        const Eigen::Vector2d sp = Eigen::Vector2d(x, y) + Eigen::Rotation2Dd(th).toRotationMatrix() * l_h;
        st.store_keyframe(T0 + t_ms, lift(raycast(room, {sp.x(), sp.y(), th + pl.psi}, 0.5 * kDeg, 0.005, rng), rng));
    };
    const auto drain = [&]() {   // as the agent does every compute(): take what the harvester committed
        for (const auto& p : h.take_pairs()) if (not o.drop_keyframes) st.add_pair(p);
    };
    h.add_odometry(T0, 0, 0, 0); h.add_gyro(T0, 0);
    offer();
    for (const auto& sg : segs)
        for (int k = 0; k < int(std::llround(sg.dur / 0.05)); ++k)
        {
            const double dt = 0.05, thm = th + 0.5 * sg.rot * dt;
            x += sg.adv * std::cos(thm) * dt; y += sg.adv * std::sin(thm) * dt; th += sg.rot * dt; path += std::abs(sg.adv) * dt;
            t_ms += 50;
            h.add_odometry(T0 + t_ms - 50, sg.adv, 0.0, sg.rot);   // piecewise-constant hold: the sample opens its own interval
            h.add_gyro(T0 + t_ms - 50, sg.rot * (1.0 + o.gyro_scale) + o.gyro_bias);
            if (t_ms % 100 == 0) { offer(); drain(); }
        }
    h.add_odometry(T0 + t_ms, 0, 0, 0); h.add_gyro(T0 + t_ms, 0);
    h.finish();
    drain();
    Out r; r.m = st.mean(); r.s = st.sigma(); r.n = st.pairs_used(); r.path = path; r.line = st.summary();
    return r;
}

// 36 m + of mixed driving: three loops of a 2.4 x 3.6 m rectangle, pivot corners, then a long arc.
static std::vector<Seg> mixed()
{
    std::vector<Seg> s;
    for (int i = 0; i < 3; ++i)
    {
        for (int side = 0; side < 4; ++side) { s.push_back({0.4, 0, side % 2 ? 9.0 : 6.0}); s.push_back({0, kPi / 4, 2.0}); }
    }
    s.push_back({0.3, 0.3, 14.0});
    return s;
}

// Three CCW loops of the same rectangle with ARC corners (radius 0.51 m): the rotation never changes sign.
static std::vector<Seg> ccw_arcs()
{
    std::vector<Seg> s;
    for (int i = 0; i < 3; ++i)
        for (int side = 0; side < 4; ++side) { s.push_back({0.4, 0, side % 2 ? 9.0 : 6.0}); s.push_back({0.4, kPi / 4, 2.0}); }
    return s;
}

int main()
{
    const Room2D room = notched_room();
    const double psi_prior = hcfg().prior_sigma_yaw, xy_prior = hcfg().prior_sigma_xy;

    {   // 1. planted_yaw_sign
        const Plant pl{0, 0, 1.84 * kDeg};
        const auto r = run(room, {-1.2, -2.2, 0.0}, mixed(), pl, {}, 11);
        std::printf("planted_yaw_sign: %.1f m  %s\n", r.path, r.line.c_str());
        CHECK(r.path > 35.0, "path %.1f m", r.path);
        CHECK(std::abs(r.m[2] + pl.psi) < 2 * r.s[2], "psi %.4f deg vs %.4f deg (2 sigma %.4f)", r.m[2] / kDeg, -pl.psi / kDeg, 2 * r.s[2] / kDeg);
        CHECK(r.s[2] < 0.1 * kDeg, "sigma psi %.4f deg", r.s[2] / kDeg);
    }
    {   // 2. lever_from_pivots
        const Plant pl{0.030, -0.020, 0.0};
        std::vector<Seg> s{{0, 0.8, 24.0}, {0, -0.8, 24.0}};
        const auto r = run(room, {0.3, -0.4, 0.2}, s, pl, {}, 12);
        std::printf("lever_from_pivots: %s\n", r.line.c_str());
        CHECK(std::abs(r.m[0] + pl.lx) < 2 * r.s[0] and std::abs(r.m[1] + pl.ly) < 2 * r.s[1],
              "(dx, dy) = (%.2f, %.2f) mm vs (%.1f, %.1f) mm, sigma (%.2f, %.2f)", r.m[0] * 1e3, r.m[1] * 1e3, -pl.lx * 1e3, -pl.ly * 1e3, r.s[0] * 1e3, r.s[1] * 1e3);
        CHECK(r.s[2] >= 0.9 * psi_prior, "psi sigma %.4f deg vs prior %.4f deg", r.s[2] / kDeg, psi_prior / kDeg);
    }
    {   // 3. yaw_from_straights
        const Plant pl{0, 0, 1.0 * kDeg};
        std::vector<Seg> s;
        for (int i = 0; i < 4; ++i) { s.push_back({0.5, 0, 6.0}); s.push_back({-0.5, 0, 6.0}); }
        const auto r = run(room, {0.0, -1.5, kPi / 2}, s, pl, {}, 13);
        std::printf("yaw_from_straights: %s\n", r.line.c_str());
        CHECK(r.s[2] < 0.2 * kDeg, "psi sigma %.4f deg", r.s[2] / kDeg);
        CHECK(std::abs(r.m[2] + pl.psi) < 2 * r.s[2], "psi %.4f deg vs %.4f deg", r.m[2] / kDeg, -pl.psi / kDeg);
        CHECK(r.s[0] >= 0.9 * xy_prior and r.s[1] >= 0.9 * xy_prior, "dx/dy sigma %.2f %.2f mm vs prior %.1f mm", r.s[0] * 1e3, r.s[1] * 1e3, xy_prior * 1e3);
    }
    {   // 4. gyro_bias_not_absorbed: one-direction CCW arc corners, a gyro that reads 3 mrad/s high, mount exact
        Opt o; o.gyro_bias = 0.003;
        const auto s = ccw_arcs();
        const auto r = run(room, {-1.2, -2.2, 0.0}, s, {}, o, 14);
        const auto c = run(room, {-1.2, -2.2, 0.0}, s, {}, o, 14, 1e-7, 1e-7);   // the control: nuisance removed
        std::printf("gyro_bias_not_absorbed: %s\n                        control(no nuisance): %s\n", r.line.c_str(), c.line.c_str());
        CHECK(std::abs(r.m[2]) < 2 * r.s[2], "psi %.4f deg, 2 sigma %.4f deg", r.m[2] / kDeg, 2 * r.s[2] / kDeg);
        CHECK(std::abs(c.m[2]) > 3 * c.s[2], "control psi %.4f deg, 3 sigma %.4f deg (must be confidently wrong)", c.m[2] / kDeg, 3 * c.s[2] / kDeg);
    }
    {   // 5. missing_sweep_adds_nothing
        HeliosMountConfig c; c.log_dir = "";
        HeliosMountStage st(c, hcfg());
        Pair p = make_pair_from_motion(0.5, 1.0, 0.0, 1.0, 0.5, hcfg());
        p.stamp_a_ms = T0; p.stamp_b_ms = T0 + 500;
        CHECK(not st.add_pair(p), "pair with no stored keyframes was accepted");
        st.store_keyframe(T0, {});                      // only one of the two, and empty
        CHECK(not st.add_pair(p), "pair with one missing sweep was accepted");
        CHECK(st.pairs_used() == 0, "pairs_used %ld", st.pairs_used());
        CHECK(st.sigma()[2] > 0.99 * psi_prior and st.mean().norm() == 0.0, "stage moved with no evidence");
    }
    std::printf(g_fail == 0 ? "helios_mount_test: PASS\n" : "helios_mount_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
