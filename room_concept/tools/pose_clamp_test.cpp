// pose_clamp_test: rc::publish::clamp_correction (src/published_pose.h) — the kinematic clamp on the PUBLISHED pose.
// warm21 (2026-11-01) clamp #36..: the estimate held at (4.122, 6.353) but the frame's prediction was the DEFAULT (0,0,0);
// the "measured motion" pred - prev_est = -(the whole pose) went through unbounded and re-based the published pose by about
// (-4.1, -6.3) every frame — (0.03,0.04), (-4.07,-6.27), (-8.16,-12.58) … (-268,-458) — covariance 290+ m, controller blind.
// + rc::publish::propagate_twist / predict_to_newest_imu: every correction's second RT block, the correction re-integrated
//   over the IMU to the newest sample (2026-11-11).
// Build + run (from room_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -Isrc tools/pose_clamp_test.cpp -o /tmp/pose_clamp_test && /tmp/pose_clamp_test
#include "../src/published_pose.h"
#include <cmath>
#include <cstdio>
#include <map>

static int failures = 0;
static void check(bool ok, const char* what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

int main()
{
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
    std::printf("propagate_twist == the predictor's original inline exp map (refactor regression)\n");
    {
        const auto legacy = [](const Eigen::Affine2f& base, float adv, float side, float rot, float dt_s)
        {   // publish_predicted_tick before 2026-11-11, verbatim
            const float base_th = std::atan2(base.linear()(1, 0), base.linear()(0, 0));
            const float phi = rot * dt_s, vx = side * dt_s, vy = adv * dt_s;
            float dx = vx, dy = vy;
            if (std::abs(phi) > 1e-6f) { const float s = std::sin(phi), c = std::cos(phi);
                dx = (s * vx - (1.f - c) * vy) / phi; dy = ((1.f - c) * vx + s * vy) / phi; }
            Eigen::Affine2f p = Eigen::Affine2f::Identity();
            p.linear() = Eigen::Rotation2Df(base_th + phi).toRotationMatrix();
            p.translation() = base.translation() + base.linear() * Eigen::Vector2f(dx, dy);
            return p;
        };
        const Eigen::Affine2f b = Eigen::Translation2f(2.f, -1.f) * Eigen::Rotation2Df(0.7f);
        float worst = 0.f;
        for (const float rot : {0.f, 0.8f, -1.9f})
            for (const float dt : {0.f, 0.008f, 0.048f, 0.2f})
                worst = std::max(worst, (rc::publish::propagate_twist(b, 0.5f, 0.05f, rot, dt).matrix()
                                         - legacy(b, 0.5f, 0.05f, rot, dt).matrix()).cwiseAbs().maxCoeff());
        check(worst == 0.f, "bit-identical on 12 (rot, dt) cases");
    }
    std::printf("every correction = TWO blocks: itself + itself re-integrated over the IMU to the newest sample\n");
    {
        // Webots-like clocks: the buffer key is SIM ms from 0; the wall stamp runs 6% faster (sim behind real time).
        const std::int64_t wall0 = 1'793'641'522'000;
        const double wall_per_sim = 1.06;
        const auto wall_of = [&](std::int64_t sim) { return wall0 + static_cast<std::int64_t>(std::llround(sim * wall_per_sim)); };
        const auto sim_of  = [&](std::int64_t wall) { return static_cast<std::int64_t>(std::llround((wall - wall0) / wall_per_sim)); };
        // Gyro ramping linearly 0.6 -> ~0.9 rad/s, sampled every 8 sim-ms: the trapezoid is exact for a ramp.
        const double w0 = 0.6, ramp = 0.3;   // rad/s, rad/s^2 (on the sim clock)
        std::vector<rc::ImuReading> imu;
        for (std::int64_t sim = 1000; sim <= 2048; sim += 8)
        {
            rc::ImuReading r;
            r.gyro_z = static_cast<float>(w0 + ramp * (sim - 1000) * 1e-3);
            r.sim_ts_ms = sim; r.simulated = true; r.source_ts_ms = wall_of(sim);
            imu.push_back(r);
        }
        // The ring before the correction: an old correction B0 and its carried block, plus a STALE prediction after t_c.
        const Eigen::Affine2f B0 = Eigen::Translation2f(1.f, 2.f) * Eigen::Rotation2Df(0.3f);
        const std::uint64_t t_c = static_cast<std::uint64_t>(wall_of(2003));   // between two samples: the clip is exercised
        std::map<std::uint64_t, Eigen::Affine2f> ring{{t_c - 50, B0}, {t_c - 10, B0}, {t_c + 20, B0}};
        const Eigen::Affine2f C = Eigen::Translation2f(1.03f, 2.01f) * Eigen::Rotation2Df(0.344f);
        const float adv = 0.45f, side = 0.01f, k_g = 1.02f, b_g = 0.003f;
        const std::int64_t t_c_key = sim_of(static_cast<std::int64_t>(t_c));
        const auto c = rc::publish::predict_to_newest_imu(C, static_cast<std::int64_t>(t_c), t_c_key, imu, true, adv, side, k_g, b_g);
        check(c.has_value(), "a correction with newer IMU samples is carried");
        if (c.has_value())
        {
            check(c->ts_ms == static_cast<std::uint64_t>(imu.back().source_ts_ms), "block 2 is stamped with the newest sample's WALL stamp");
            // Independent: integral of the ramp over (t_c_key, 2048] on the sim clock, calibrated.
            const double T0 = (t_c_key - 1000) * 1e-3, T1 = (2048 - 1000) * 1e-3;
            const double raw = w0 * (T1 - T0) + 0.5 * ramp * (T1 * T1 - T0 * T0);
            const double want = raw * k_g - b_g * (T1 - T0);
            const float yaw_c = std::atan2(C.linear()(1, 0), C.linear()(0, 0));
            const float yaw_p = std::atan2(c->pose.linear()(1, 0), c->pose.linear()(0, 0));
            std::printf("  dyaw: carried %.6f rad, analytic integrated gyro %.6f rad\n", std::remainder(yaw_p - yaw_c, 2.0 * M_PI), want);
            check(std::abs(std::remainder(yaw_p - yaw_c - want, 2.0 * M_PI)) < 1e-5, "block 2's yaw = the correction's + the integrated (calibrated) gyro");
            // Translation runs on WALL time with the cached twist: ~|v| * (t_imu - t_c)_wall.
            const float d = (c->pose.translation() - C.translation()).norm();
            const float want_d = std::hypot(adv, side) * static_cast<float>(c->ts_ms - t_c) * 1e-3f;
            std::printf("  translation %.5f m, |v| * wall gap %.5f m\n", d, want_d);
            check(std::abs(d - want_d) < 2e-4f, "translation = cached twist over the WALL gap (chord, within 0.2 mm)");
            // cortex's rewrite_edge_RT_since contract: drop every block >= t_c, write the two.
            std::erase_if(ring, [&](const auto& kv) { return kv.first >= t_c; });
            ring[t_c] = C;
            ring[c->ts_ms] = c->pose;
            check(ring.size() == 4 and ring.contains(t_c - 50) and ring.contains(t_c - 10)
                  and ring.at(t_c - 50).isApprox(B0) and not ring.contains(t_c + 20),
                  "the past is untouched, the stale prediction after t_c is gone, two blocks added");
        }
        // No IMU sample newer than the correction (the newest is at sim 2048): the correction goes alone.
        const std::uint64_t t_late = static_cast<std::uint64_t>(wall_of(2060));
        check(not rc::publish::predict_to_newest_imu(C, static_cast<std::int64_t>(t_late), sim_of(static_cast<std::int64_t>(t_late)),
                                                     imu, true, adv, side, k_g, b_g).has_value(),
              "a correction with no newer IMU sample gives ONE block");
        // History not reaching back to t_c: refused (partial coverage under-rotates), the correction goes alone.
        const std::vector<rc::ImuReading> short_imu(imu.end() - 3, imu.end());
        check(not rc::publish::predict_to_newest_imu(C, static_cast<std::int64_t>(t_c), t_c_key, short_imu, true, adv, side, k_g, b_g).has_value(),
              "an IMU history that does not bracket (t_c, t_imu] is refused");
    }
    std::printf("real robot (one clock), constant gyro: the piecewise carry == propagate_twist in one step\n");
    {
        std::vector<rc::ImuReading> imu;
        for (std::int64_t t = 5000; t <= 5100; t += 10)
        { rc::ImuReading r; r.gyro_z = 0.8f; r.source_ts_ms = t; imu.push_back(r); }
        const Eigen::Affine2f C = Eigen::Translation2f(-0.7f, 2.7f) * Eigen::Rotation2Df(2.2f);
        const auto c = rc::publish::predict_to_newest_imu(C, 5043, 5043, imu, false, 0.4f, -0.02f, 1.f, 0.f);
        const auto one = rc::publish::propagate_twist(C, 0.4f, -0.02f, 0.8f, 0.057f);
        check(c.has_value() and c->ts_ms == 5100 and (c->pose.matrix() - one.matrix()).cwiseAbs().maxCoeff() < 1e-5f,
              "constant twist: composing the 7 pieces equals one exp-map step");
    }
    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    return failures == 0 ? 0 : 1;
}
