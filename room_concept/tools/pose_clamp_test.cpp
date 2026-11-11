// pose_clamp_test: rc::publish::clamp_correction (src/published_pose.h) — the kinematic clamp on the PUBLISHED pose.
// warm21 (2026-11-01) clamp #36..: the estimate held at (4.122, 6.353) but the frame's prediction was the DEFAULT (0,0,0);
// the "measured motion" pred - prev_est = -(the whole pose) went through unbounded and re-based the published pose by about
// (-4.1, -6.3) every frame — (0.03,0.04), (-4.07,-6.27), (-8.16,-12.58) … (-268,-458) — covariance 290+ m, controller blind.
// + rc::publish::propagate_twist / superseded_tail: the late-correction rewrite of the predicted tail (2026-11-11).
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
    std::printf("a late correction leaves NO pre-correction prediction after t_c (rewrite_edge_RT_since contract)\n");
    {
        // The ring as the agent leaves it: correction B0 at 1000, then IMU ticks every 8 ms from B0, stamps 1008..1048.
        const Eigen::Affine2f B0 = Eigen::Translation2f(1.f, 2.f) * Eigen::Rotation2Df(0.3f);
        const float adv0 = 0.4f, side0 = 0.f, rot0 = 0.9f;
        std::map<std::uint64_t, Eigen::Affine2f> ring{{1000, B0}};
        for (std::uint64_t ts = 1008; ts <= 1048; ts += 8)
            ring[ts] = rc::publish::propagate_twist(B0, adv0, side0, rot0, static_cast<float>(ts - 1000) * 1e-3f);
        std::vector<std::uint64_t> stamps{0, 0};                    // two empty slots, as cortex leaves them
        for (const auto& [ts, _] : ring) stamps.push_back(ts);
        std::ranges::reverse(stamps);                               // ring order is not stamp order
        // The correction for 1010 arrives now, 3 cm and 2 deg off the prediction, with a new twist.
        const std::uint64_t t_c = 1010;
        const Eigen::Affine2f C = Eigen::Translation2f(1.03f, 2.01f) * Eigen::Rotation2Df(0.3f + 0.009f + 0.035f);
        const float adv1 = 0.45f, side1 = 0.01f, rot1 = 0.85f;
        const auto tail = rc::publish::superseded_tail(stamps, t_c, C, adv1, side1, rot1);
        check(tail.size() == 5 and tail.front().first == 1016 and tail.back().first == 1048, "tail = the 5 stamps after t_c, oldest first");
        // cortex's contract: drop every block at/after the earliest given, write the given ones.
        std::erase_if(ring, [&](const auto& kv) { return kv.first >= t_c; });
        ring[t_c] = C;
        for (const auto& [ts, p] : tail) ring[ts] = p;
        bool all_from_c = true, before_kept = ring.contains(1000) and ring.contains(1008);
        for (const auto& [ts, p] : ring)
            if (ts > t_c)
                all_from_c = all_from_c and (p.matrix() - rc::publish::propagate_twist(C, adv1, side1, rot1,
                                             static_cast<float>(ts - t_c) * 1e-3f).matrix()).cwiseAbs().maxCoeff() == 0.f;
        check(all_from_c, "every block after t_c is propagated from the correction");
        check(before_kept and ring.size() == 8, "blocks before t_c untouched, none lost");
        // The next tick (1056), computed from the new base, continues the rewritten tail with no step.
        const auto next = rc::publish::propagate_twist(C, adv1, side1, rot1, 0.046f);
        const float step = (next.translation() - ring.rbegin()->second.translation()).norm();
        std::printf("  newest rewritten -> next tick: %.4f m (one 8 ms step of 0.45 m/s = 0.0036)\n", step);
        check(step < 0.004f, "the next tick continues the rewritten tail");
        check(rc::publish::superseded_tail(stamps, 2000, C, adv1, side1, rot1).empty(), "a correction newer than the ring rewrites nothing");
    }
    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    return failures == 0 ? 0 : 1;
}
