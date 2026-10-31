// helios_mount_test -- Task 4.4: planar Helios mount from registered keyframe pairs. Synthetic (notched 4x6 m room,
// 0.5 deg rays, 5 mm range noise), no DDS/DSR. The sweeps are generated in the TRUE sensor frame (true mount =
// nominal + planted error) and expressed in the robot frame through the NOMINAL mount, the odometry comes from the
// true motion; the stage must read the correction (dx, dy, psi) = -(planted error) (Shadow axes: X lateral, Y forward).
//   make -C build helios_mount_test && ../bin/helios_mount_test
#include "../src/helios_mount_stage.h"
#include "synthetic_room.h"

#include <cmath>
#include <limits>
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
struct Opt {
    double gyro_bias = 0.0, gyro_scale = 0.0, wheel_scale = 0.0;   // odometry errors: gyro rad/s bias, fractional gyro / wheel-translation scale
    bool drop_keyframes = false;
    double ring_span_s = 10.0;       // harvester ring (compute budget); 10 s keeps the older tests' pair sets
    int drain_every_ms = 100;        // how often the "agent tick" takes pairs from the harvester
    double max_range = std::numeric_limits<double>::infinity();   // lidar range limit (corridor: no end walls in view)
    double wheel_sigma = 0.03, outlier_prior = 0.05;
    bool cross_cov = true;
    int inject_k = -1;               // replace the registration of the inject_k-th pair with |d_fwd| > 1 m by a wrong-basin one
    double inject_err_m = 0.3;
};

struct Out { Eigen::Vector3d m, s; long n = 0, missing = 0, rejected = 0; double path = 0, n_eff = 0, gamma_injected = -1; std::string line; };

static HarvesterConfig hcfg(double span = 10.0) { HarvesterConfig c; c.ring_span_s = span; return c; }

// Lift 2-D hits to 3-D robot-frame points: wall hits inside the band, plus decoys outside it that must be ignored.
static std::vector<Eigen::Vector3f> lift(const std::vector<Eigen::Vector2f>& hits_h, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uz(1.55f, 1.95f), u(-3.f, 3.f);
    std::vector<Eigen::Vector3f> out;
    for (const auto& h : hits_h) out.emplace_back(-h.y(), h.x(), uz(rng));   // harvester (fwd, left) -> Shadow (X = -left, Y = fwd)
    for (int i = 0; i < 300; ++i) out.emplace_back(u(rng), u(rng), (i % 2) ? 0.3f : 1.0f);   // floor / furniture clutter, below the band
    return out;
}

static Eigen::Matrix2d rot(double a) { Eigen::Matrix2d R; R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a); return R; }

static Out run(const Room2D& room, const Eigen::Vector3d& start, const std::vector<Seg>& segs, const Plant& pl, const Opt& o,
               unsigned seed, double gyro_bias_sigma = 0.002, double gyro_scale_sigma = 0.01)
{
    HeliosMountConfig c; c.log_dir = ""; c.gyro_bias_sigma = gyro_bias_sigma; c.gyro_scale_sigma = gyro_scale_sigma;
    c.wheel_scale_sigma = o.wheel_sigma; c.heading_cross_cov = o.cross_cov; c.outlier_prior = o.outlier_prior;
    c.registration_budget_ms = 1e9;   // the tests drain completely every tick; the budget logic has its own test
    TrajectoryHarvester h(hcfg(o.ring_span_s));
    HeliosMountStage st(c, hcfg(o.ring_span_s));
    std::mt19937 rng(seed);
    const Eigen::Vector2d l_h(pl.ly, -pl.lx);   // harvester axes: forward = Y, left = -X
    double x = start.x(), y = start.y(), th = start.z(), path = 0;
    std::int64_t t_ms = 0;
    const auto offer = [&]() {
        if (not h.add_keyframe(T0 + t_ms, false)) return;
        const Eigen::Vector2d sp = Eigen::Vector2d(x, y) + Eigen::Rotation2Dd(th).toRotationMatrix() * l_h;
        st.store_keyframe(T0 + t_ms, lift(raycast(room, {sp.x(), sp.y(), th + pl.psi}, 0.5 * kDeg, 0.005, rng, o.max_range), rng));
    };
    int n_long = 0, inj_idx = -1;
    const auto drain = [&]() {   // as the agent does every compute(): take what the harvester committed, queue, process
        const auto pairs = h.take_pairs();
        if (o.drop_keyframes) {}
        else if (o.inject_k < 0) st.enqueue(pairs);
        else for (const auto& p : pairs)
        {
            auto reg = st.register_pair(p);
            if (not reg) continue;
            if (std::abs(p.d_fwd) > 1.0 and n_long++ == o.inject_k)
            {   // a wrong-basin registration that still reports a 1 mm covariance: the chord is off by inject_err_m laterally
                reg->xyt.head<2>() += rot(0.5 * p.dth) * Eigen::Vector2d(0.0, o.inject_err_m);
                reg->cov = Eigen::Vector3d(1e-6, 1e-6, 1e-8).asDiagonal();
                inj_idx = int(st.pairs_used());
            }
            st.add_registration(p, *reg);
        }
        st.process_queue(h.oldest_pending_stamp_ms());
    };
    h.add_odometry(T0, 0, 0, 0); h.add_gyro(T0, 0);
    offer();
    for (const auto& sg : segs)
        for (int k = 0; k < int(std::llround(sg.dur / 0.05)); ++k)
        {
            const double dt = 0.05, thm = th + 0.5 * sg.rot * dt;
            x += sg.adv * std::cos(thm) * dt; y += sg.adv * std::sin(thm) * dt; th += sg.rot * dt; path += std::abs(sg.adv) * dt;
            t_ms += 50;
            h.add_odometry(T0 + t_ms - 50, sg.adv * (1.0 + o.wheel_scale), 0.0, sg.rot);   // piecewise-constant hold: the sample opens its own interval
            h.add_gyro(T0 + t_ms - 50, sg.rot * (1.0 + o.gyro_scale) + o.gyro_bias);
            if (t_ms % 100 == 0) offer();
            if (t_ms % o.drain_every_ms == 0) drain();
        }
    h.add_odometry(T0 + t_ms, 0, 0, 0); h.add_gyro(T0 + t_ms, 0);
    h.finish();
    drain();
    Out r; r.m = st.mean(); r.s = st.sigma(); r.n = st.pairs_used(); r.path = path; r.line = st.summary();
    r.missing = st.missing(); r.rejected = st.rejected(); r.n_eff = st.effective_pairs();
    if (inj_idx >= 0) r.gamma_injected = st.responsibilities()[std::size_t(inj_idx)];
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
    {   // 4. gyro_bias_not_absorbed: one-direction CCW arc corners, a gyro that reads 8 mrad/s high, mount exact; 5 seeds
        Opt o; o.gyro_bias = 0.008;
        const auto s = ccw_arcs();
        for (unsigned seed = 14; seed < 19; ++seed)
        {
            const auto r = run(room, {-1.2, -2.2, 0.0}, s, {}, o, seed);
            Opt oc = o; oc.outlier_prior = 0.0; oc.wheel_sigma = 1e-7; oc.cross_cov = false;   // the control = the model WITHOUT nuisances (the original control): with the EM weight on, a mis-modelled run is called all-outlier (sigma back to the prior) instead; the wheel-scale nuisance would alias part of the gyro bias and the lat-rot cross term (minor 5) explains most of it away per pair; the lat-rot cross term (minor 5) would explain most of it away per pair
            const auto c = run(room, {-1.2, -2.2, 0.0}, s, {}, oc, seed, 1e-7, 1e-7);   // the control: nuisance removed
            std::printf("gyro_bias_not_absorbed[%u]: psi %.3f +- %.3f deg   control(no nuisance): %.3f +- %.3f deg (%.1f sigma)\n", seed,
                        r.m[2] / kDeg, r.s[2] / kDeg, c.m[2] / kDeg, c.s[2] / kDeg, std::abs(c.m[2]) / c.s[2]);
            CHECK(std::abs(r.m[2]) < 2 * r.s[2], "seed %u psi %.4f deg, 2 sigma %.4f deg", seed, r.m[2] / kDeg, 2 * r.s[2] / kDeg);
            CHECK(std::abs(c.m[2]) > 3 * c.s[2], "seed %u control psi %.4f deg, 3 sigma %.4f deg (must be confidently wrong)", seed, c.m[2] / kDeg, 3 * c.s[2] / kDeg);
        }
    }
    {   // 4b. wheel_scale_not_absorbed: odometry translation 2 % long, one-direction arcs, mount exact; 5 seeds
        Opt o; o.wheel_scale = 0.02;
        Opt oc = o; oc.wheel_sigma = 1e-7; oc.outlier_prior = 0.0;   // the control: the wheel-scale nuisance removed (and, to isolate it, the EM weight)
        const auto s = ccw_arcs();
        for (unsigned seed = 20; seed < 25; ++seed)
        {
            const auto r = run(room, {-1.2, -2.2, 0.0}, s, {}, o, seed);
            const auto c = run(room, {-1.2, -2.2, 0.0}, s, {}, oc, seed);
            std::printf("wheel_scale_not_absorbed[%u]: dx %.2f +- %.2f dy %.2f +- %.2f mm   control(no nuisance): dx %.2f +- %.2f (%.1f sigma) dy %.2f +- %.2f (%.1f sigma)\n", seed,
                        r.m[0] * 1e3, r.s[0] * 1e3, r.m[1] * 1e3, r.s[1] * 1e3, c.m[0] * 1e3, c.s[0] * 1e3, std::abs(c.m[0]) / c.s[0],
                        c.m[1] * 1e3, c.s[1] * 1e3, std::abs(c.m[1]) / c.s[1]);
            CHECK(std::abs(r.m[0]) < 2 * r.s[0] and std::abs(r.m[1]) < 2 * r.s[1], "seed %u (dx, dy) = (%.2f, %.2f) mm, 2 sigma (%.2f, %.2f)", seed,
                  r.m[0] * 1e3, r.m[1] * 1e3, 2 * r.s[0] * 1e3, 2 * r.s[1] * 1e3);
            CHECK(std::abs(c.m[0]) > 3 * c.s[0] or std::abs(c.m[1]) > 3 * c.s[1], "seed %u control (%.2f, %.2f) mm must be confidently wrong on the biased axis (3 sigma %.2f, %.2f)", seed,
                  c.m[0] * 1e3, c.m[1] * 1e3, 3 * c.s[0] * 1e3, 3 * c.s[1] * 1e3);
        }
    }
    {   // 4c. aliased_pair_downweighted: one wrong-basin registration (0.3 m chord error, 1 mm covariance) in a clean planted-yaw run
        const Plant pl{0, 0, 1.84 * kDeg};
        Opt o; o.inject_k = 3;
        Opt oc = o; oc.outlier_prior = 0.0;   // the control: every pair an inlier
        const auto r = run(room, {-1.2, -2.2, 0.0}, mixed(), pl, o, 11);
        const auto c = run(room, {-1.2, -2.2, 0.0}, mixed(), pl, oc, 11);
        std::printf("aliased_pair_downweighted: gamma(injected) %.2e  n_eff %.1f/%ld  psi err %.4f deg (%.2f sigma)   control(no EM weight): psi err %.4f deg (%.2f sigma)\n",
                    r.gamma_injected, r.n_eff, r.n, (r.m[2] + pl.psi) / kDeg, std::abs(r.m[2] + pl.psi) / r.s[2], (c.m[2] + pl.psi) / kDeg, std::abs(c.m[2] + pl.psi) / c.s[2]);
        CHECK(r.gamma_injected >= 0.0 and r.gamma_injected < 0.01, "injected pair responsibility %.3g", r.gamma_injected);
        CHECK(std::abs(r.m[2] + pl.psi) < 2 * r.s[2], "psi %.4f deg vs %.4f deg (2 sigma %.4f)", r.m[2] / kDeg, -pl.psi / kDeg, 2 * r.s[2] / kDeg);
        CHECK(std::abs(c.m[2] + pl.psi) > 3 * c.s[2], "control psi err %.4f deg must be visibly pulled (3 sigma %.4f deg)", (c.m[2] + pl.psi) / kDeg, 3 * c.s[2] / kDeg);
    }
    {   // 4d. batch_drain_no_missing: default ring (30 s), pairs taken once per 4 s batch of keyframes -> no sweep pruned before its pair
        const Plant pl{0, 0, 1.0 * kDeg};
        Opt o; o.ring_span_s = 30.0; o.drain_every_ms = 4000;
        const auto r = run(room, {-1.2, -2.2, 0.0}, mixed(), pl, o, 31);
        std::printf("batch_drain_no_missing: %s\n", r.line.c_str());
        CHECK(r.missing == 0, "%ld pairs reported missing sweeps", r.missing);
        CHECK(r.n > 10, "only %ld pairs", r.n);
    }
    {   // 4e. corridor_straights: scan range limited so the end walls are out of view: psi informed, dx stays at its prior
        const Plant pl{0, 0, 1.0 * kDeg};
        Opt o; o.max_range = 4.0;
        std::vector<Seg> s;
        for (int i = 0; i < 3; ++i) { s.push_back({0.5, 0, 16.0}); s.push_back({-0.5, 0, 16.0}); }
        const auto r = run(corridor(40.0, 2.0), {-10.0, 0.0, 0.0}, s, pl, o, 41);
        std::printf("corridor_straights: %s\n", r.line.c_str());
        CHECK(r.rejected == 0 and r.n == 12, "corridor pairs: %ld used, %ld rejected (the registration must converge in a corridor)", r.n, r.rejected);
        CHECK(r.s[2] < 0.2 * kDeg, "psi sigma %.4f deg", r.s[2] / kDeg);
        CHECK(std::abs(r.m[2] + pl.psi) < 2 * r.s[2], "psi %.4f deg vs %.4f deg (2 sigma %.4f)", r.m[2] / kDeg, -pl.psi / kDeg, 2 * r.s[2] / kDeg);
        CHECK(r.s[0] >= 0.9 * xy_prior and r.s[1] >= 0.9 * xy_prior, "dx/dy sigma %.2f %.2f mm vs prior %.1f mm", r.s[0] * 1e3, r.s[1] * 1e3, xy_prior * 1e3);
    }
    {   // 4f. rejected_pair_counted: both sweeps present but too sparse to register -> register_pair fails, rejected_ counts, nothing added
        HeliosMountConfig c; c.log_dir = "";
        HeliosMountStage st(c, hcfg());
        Pair p = make_pair_from_motion(0.5, 1.0, 0.0, 1.0, 0.5, hcfg());
        p.stamp_a_ms = T0; p.stamp_b_ms = T0 + 500;
        const std::vector<Eigen::Vector3f> sparse{{1.f, 1.f, 1.7f}, {2.f, 1.f, 1.7f}, {3.f, 1.f, 1.7f}};
        st.store_keyframe(T0, sparse); st.store_keyframe(T0 + 500, sparse);
        CHECK(not st.add_pair(p), "unregistrable pair was accepted");
        CHECK(st.rejected() == 1 and st.missing() == 0 and st.pairs_used() == 0, "rejected %ld missing %ld used %ld", st.rejected(), st.missing(), st.pairs_used());
    }
    {   // 4g. budget_queue: a burst of pairs drains one per call when the budget is spent at once, and sweeps are held until consumed
        HeliosMountConfig c; c.log_dir = ""; c.registration_budget_ms = 0.0;
        HeliosMountStage st(c, hcfg());
        std::vector<Pair> ps;
        for (int i = 0; i < 4; ++i)
        {
            Pair p = make_pair_from_motion(0.5, 1.0, 0.0, 1.0, 0.5, hcfg());
            p.stamp_a_ms = T0 + 1000 * i; p.stamp_b_ms = p.stamp_a_ms + 500;
            st.store_keyframe(p.stamp_a_ms, {}); st.store_keyframe(p.stamp_b_ms, {});
            ps.push_back(p);
        }
        st.enqueue(ps);
        CHECK(st.process_queue(T0 + 10'000) == 1 and st.queued() == 3, "queued %zu", st.queued());
        CHECK(st.sweeps_held() == 6, "sweeps held %zu while pairs still queued (the consumed pair's two are gone)", st.sweeps_held());
        st.process_queue(T0 + 10'000); st.process_queue(T0 + 10'000); st.process_queue(T0 + 10'000);
        CHECK(st.queued() == 0 and st.missing() == 4, "queued %zu missing %ld (empty sweeps)", st.queued(), st.missing());
        CHECK(st.sweeps_held() == 0, "sweeps held %zu after the queue drained with the ring front ahead", st.sweeps_held());
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
