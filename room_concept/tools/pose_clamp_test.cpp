// pose_clamp_test: rc::publish::clamp_correction (src/published_pose.h) — the kinematic clamp on the PUBLISHED pose.
// warm21 (2026-11-01) clamp #36..: the estimate held at (4.122, 6.353) but the frame's prediction was the DEFAULT (0,0,0);
// the "measured motion" pred - prev_est = -(the whole pose) went through unbounded and re-based the published pose by about
// (-4.1, -6.3) every frame — (0.03,0.04), (-4.07,-6.27), (-8.16,-12.58) … (-268,-458) — covariance 290+ m, controller blind.
// + rc::PosePropagator / plan_correction / record_rewrite (src/pose_propagator.h): the pose between corrections and the
//   blocks every correction writes with itself — spread over (t_k, t_c), re-integrated tail after t_c (2026-11-11).
// Build + run (from room_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -Isrc tools/pose_clamp_test.cpp -o /tmp/pose_clamp_test && /tmp/pose_clamp_test
#include "../src/published_pose.h"
#include "../src/pose_propagator.h"
#include <clocale>
#include <cmath>
#include <cstdio>
#include <map>

static int failures = 0;
static void check(bool ok, const char* what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

int main()
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: the agent runs under es_ES; so does the test
    const float dt = 0.05f, max_xy = 1.f * dt, max_th = 2.f * dt;   // the agent's limits: 1 m/s, 2 rad/s
    std::printf("warm21 runaway: a default (0,0,0) prediction must not move the published pose\n");
    {
        Eigen::Affine2f pub = Eigen::Translation2f(4.122f, 6.353f) * Eigen::Rotation2Df(-0.554f);
        const Eigen::Vector3f est(4.122f, 6.353f, -1.876f);
        std::optional<Eigen::Vector3f> prev_est = Eigen::Vector3f(4.122f, 6.353f, -0.554f);
        const Eigen::Affine2f now = Eigen::Translation2f(est.x(), est.y()) * Eigen::Rotation2Df(est.z());
        for (int k = 0; k < 6; ++k)
        {
            pub = rc::publish::clamp_correction(pub, prev_est, now, Eigen::Vector3f::Zero(), max_xy, max_th).pose;
            prev_est = est;
        }
        const float off = (pub.translation() - Eigen::Vector2f(4.122f, 6.353f)).norm();
        std::printf("  after 6 frames: published (%.3f,%.3f), %.3f m from the estimate\n", pub.translation().x(), pub.translation().y(), off);
        check(off < 0.01f, "the published pose stays at the estimate");
    }
    std::printf("a real measured motion still passes\n");
    {
        const Eigen::Affine2f p0 = Eigen::Translation2f(1.f, 1.f) * Eigen::Rotation2Df(0.f);
        const Eigen::Affine2f n1 = Eigen::Translation2f(1.03f, 1.f) * Eigen::Rotation2Df(0.f);
        const auto cl = rc::publish::clamp_correction(p0, Eigen::Vector3f(1.f, 1.f, 0.f), n1, Eigen::Vector3f(1.03f, 1.f, 0.f), max_xy, max_th);
        check(not cl.fired and std::abs(cl.pose.translation().x() - 1.03f) < 1e-5f, "a 3 cm step with a matching prediction passes unclamped");
    }
    using rc::PosePropagator;
    const rc::PropagatorParams pp{.odom_lag_ms = 0};   // the toml's OdomLagMs = 0
    const auto wrap = [](double a) { return std::remainder(a, 2.0 * M_PI); };
    std::printf("propagator: gyro trapezoid with the applied k_w, b_w (ramp => exact)\n");
    {
        PosePropagator P(pp);
        const double w0 = 0.6, ramp = 0.3;   // rad/s, rad/s^2
        for (std::int64_t t = 1000; t <= 1200; t += 8) P.add_imu(t, static_cast<float>(w0 + ramp * (t - 1000) * 1e-3));
        P.add_odometry(990, 0.f, 0.f, 0.5f);   // the wheels say turning: no ZUPT
        const rc::PropagatorCalib cal{.k_w = 1.02f, .b_w = 0.003f, .k_ww = 0.9f};
        P.reset(1003, 0.f, 0.f, 0.2f, cal);    // between two samples: the clip is exercised
        const auto s1 = P.at(1197);
        const double T0 = 0.003, T1 = 0.197;
        const double want = (w0 * (T1 - T0) + 0.5 * ramp * (T1 * T1 - T0 * T0)) * 1.02 - 0.003 * (T1 - T0);
        std::printf("  dth %.7f, analytic %.7f\n", s1->th - 0.2f, want);
        check(s1 and std::abs(wrap(s1->th - 0.2 - want)) < 2e-6, "heading = k_w * integral(gyro) - b_w * dt");
    }
    std::printf("propagator: translation = wheels along the mid heading (k_v, k_lat, yaw_offset)\n");
    {
        PosePropagator P(pp);
        const float w = 0.8f, v = 0.4f;
        for (std::int64_t t = 0; t <= 1000; t += 8) P.add_imu(t, w);
        P.add_odometry(0, v / 1.05f, 0.f, w);   // k_v = 1.05 restores v
        P.reset(0, 1.f, 2.f, 0.3f, {.k_v = 1.05f});
        const auto s1 = P.at(500);
        // Exact arc for a +Y-forward body: heading th, velocity (-v sin th, v cos th).
        const double th0 = 0.3, th1 = th0 + w * 0.5;
        const double x = 1.0 + v / w * (std::cos(th1) - std::cos(th0)), y = 2.0 + v / w * (std::sin(th1) - std::sin(th0));
        const double err = std::hypot(s1->x - x, s1->y - y);
        std::printf("  (%.5f, %.5f) vs arc (%.5f, %.5f): %.3f mm\n", s1->x, s1->y, x, y, err * 1e3);
        check(err < 1e-4, "on the arc within 0.1 mm over 0.5 s at 0.8 rad/s");
        PosePropagator Q(pp);
        for (std::int64_t t = 0; t <= 1000; t += 8) Q.add_imu(t, 0.f);
        Q.add_odometry(0, 0.f, 0.3f, 0.01f);   // pure strafe, with a mount yaw offset
        Q.reset(0, 0.f, 0.f, 0.f, {.k_lat = 1.f, .yaw_offset = 0.1f});
        const auto q = Q.at(1000);
        check(std::abs(q->x - 0.3f * std::cos(0.1f)) < 1e-5f and std::abs(q->y - 0.3f * std::sin(0.1f)) < 1e-5f,
              "lateral (+X) travel rotated by yaw_offset");
    }
    std::printf("propagator: ZUPT — wheels stopped, a biased gyro does not turn the pose\n");
    {
        PosePropagator P(pp);
        for (std::int64_t t = 0; t <= 2000; t += 8) P.add_imu(t, 0.01f);
        P.add_odometry(0, 0.f, 0.f, 0.f);
        P.reset(0, 0.f, 0.f, 1.f, {});
        check(std::abs(P.at(1900)->th - 1.f) < 1e-7f, "stopped: heading held");
        P.add_odometry(1950, 0.f, 0.f, 0.3f);   // moving again: the gyro is back
        const auto s2 = P.at(2000);
        check(std::abs(s2->th - 1.f - 0.01f * 0.05f) < 1e-6f, "moving: the gyro integrates from the first moving sample");
    }
    std::printf("propagator: reset integrates to the newest IMU sample\n");
    {
        PosePropagator P(pp);
        for (std::int64_t t = 0; t <= 96; t += 8) P.add_imu(t, 0.5f);
        P.add_odometry(0, 0.2f, 0.f, 0.5f);
        P.reset(51, 0.f, 0.f, 0.f, {});
        check(P.latest() and P.latest()->t_ms == 96 and std::abs(P.latest()->th - 0.5f * 0.045f) < 1e-6f, "latest() is at 96 ms, 45 ms of gyro");
        P.reset(120, 0.f, 0.f, 0.f, {});
        check(P.latest()->t_ms == 120, "no newer sample: latest() is the correction itself");
    }
    // A run: gyro 0.6 rad/s every 8 ms, wheels 0.4 m/s forward. Corrections C1 at 1004 and C2 at 1052, each landing
    // 4 ms after its stamp (Webots) — the IMU is fed up to the landing time before the plan.
    const auto feed = [](PosePropagator& P, std::int64_t& t_imu, std::int64_t until)
    { for (; t_imu <= until; t_imu += 8) P.add_imu(t_imu, 0.6f); };
    std::printf("mode (a): ticks are not written — the tail is the newest-IMU block, the spread moves the previous tail\n");
    {
        PosePropagator P(pp); std::deque<std::int64_t> written; std::int64_t t_imu = 800;
        P.add_odometry(800, 0.4f, 0.f, 0.6f);
        feed(P, t_imu, 1008);
        const PosePropagator::Pose C1{1004, 1.f, 1.f, 0.5f};
        auto plan1 = rc::plan_correction(P, written, C1, {}, true, 500);
        check(plan1.spread.empty() and plan1.tail.size() == 1 and plan1.tail[0].t_ms == 1008, "C1: no base yet, tail = one block at the newest sample");
        rc::record_rewrite(written, C1.t_ms, plan1);
        // independent prediction of C1's tail block, made now, before anything else is fed
        PosePropagator R1(pp); { std::int64_t ti = 800; R1.add_odometry(800, 0.4f, 0.f, 0.6f); feed(R1, ti, 1056); R1.reset(1004, 1.f, 1.f, 0.5f, {}); }
        feed(P, t_imu, 1056);
        const auto pred_c2 = R1.at(1052); const auto pred_t = R1.at(1008);
        const PosePropagator::Pose C2{1052, pred_c2->x + 0.012f, pred_c2->y - 0.006f, pred_c2->th + 0.004f};   // the jump
        auto plan2 = rc::plan_correction(P, written, C2, {}, true, 500);
        check(plan2.tail.size() == 1 and plan2.tail[0].t_ms == 1056, "C2: tail = ONE block, the newest IMU sample");
        check(plan2.spread.size() == 1 and plan2.spread[0].t_ms == 1008, "C2: spread = C1's tail block, the only written block in (t_k, t_c)");
        if (plan2.spread.size() == 1)
        {
            const float a = (1008.f - 1004.f) / (1052.f - 1004.f);
            const auto& s = plan2.spread[0];
            check(std::abs(s.x - (pred_t->x + a * 0.012f)) < 1e-5f and std::abs(s.y - (pred_t->y - a * 0.006f)) < 1e-5f
                  and std::abs(s.th - (pred_t->th + a * 0.004f)) < 1e-6f, "spread block = prediction + alpha * Delta, alpha = (t - t_k)/(t_c - t_k)");
        }
        rc::record_rewrite(written, C2.t_ms, plan2);
        check(written == std::deque<std::int64_t>{1008, 1056}, "the edge's non-correction blocks: C1's (spread) tail, C2's tail");
        // relocalisation: may_spread = false
        feed(P, t_imu, 1104);
        const PosePropagator::Pose C3{1100, 5.f, 5.f, 2.f};
        auto plan3 = rc::plan_correction(P, written, C3, {}, false, 500);
        check(plan3.spread.empty() and plan3.tail.size() == 1, "relocalisation: no spread, the tail still written");
        rc::record_rewrite(written, C3.t_ms, plan3);
        check(written == std::deque<std::int64_t>{1008, 1056, 1104}, "nothing before t_c rewritten without a spread");
    }
    std::printf("mode (b): every written tick after t_c is re-integrated from the correction\n");
    {
        PosePropagator P(pp); std::deque<std::int64_t> written; std::int64_t t_imu = 800;
        P.add_odometry(800, 0.4f, 0.f, 0.6f);
        feed(P, t_imu, 1000);
        P.reset(1000, 0.f, 0.f, 0.f, {});
        for (; t_imu <= 1096; t_imu += 8) { P.add_imu(t_imu, 0.6f); P.at(t_imu); written.push_back(t_imu); }   // ticks written
        const PosePropagator::Pose C{1050, 0.03f, 0.03f, 0.05f};
        auto plan = rc::plan_correction(P, written, C, {}, true, 500);
        std::size_t n_after = 0, n_between = 0;
        for (const auto t : written) { n_after += t > 1050; n_between += t > 1000 and t < 1050; }
        check(plan.tail.size() == n_after and plan.spread.size() == n_between, "tail = the ticks after t_c, spread = the ticks in (t_k, t_c)");
        PosePropagator R(pp); { std::int64_t ti = 800; R.add_odometry(800, 0.4f, 0.f, 0.6f); feed(R, ti, 1096); R.reset(1050, 0.03f, 0.03f, 0.05f, {}); }
        float worst = 0.f;
        for (const auto& s : plan.tail) { const auto r = R.at(s.t_ms); worst = std::max({worst, std::abs(s.x - r->x), std::abs(s.y - r->y), std::abs(s.th - r->th)}); }
        check(worst < 1e-6f and plan.tail.back().t_ms == 1096, "each re-integrated tick == the correction propagated to it; the last is the newest");
        // the spread at t -> t_c reaches the correction, at t -> t_k stays at the old prediction: no step at either end
        check(plan.spread.front().t_ms == 1008 and plan.spread.back().t_ms == 1048, "spread spans the ticks strictly inside (t_k, t_c)");
        rc::record_rewrite(written, C.t_ms, plan);
        check(written.size() == n_between + n_after and std::ranges::is_sorted(written), "ticks keep their stamps, ascending");
    }
    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    return failures == 0 ? 0 : 1;
}
