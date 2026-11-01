// pose_clamp_test: rc::publish::clamp_correction (src/published_pose.h) — the kinematic clamp on the PUBLISHED pose.
// warm21 (2026-11-01) clamp #36..: the estimate held at (4.122, 6.353) but the frame's prediction was the DEFAULT (0,0,0);
// the "measured motion" pred - prev_est = -(the whole pose) went through unbounded and re-based the published pose by about
// (-4.1, -6.3) every frame — (0.03,0.04), (-4.07,-6.27), (-8.16,-12.58) … (-268,-458) — covariance 290+ m, controller blind.
// Build + run (from room_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -Isrc tools/pose_clamp_test.cpp -o /tmp/pose_clamp_test && /tmp/pose_clamp_test
#include "../src/published_pose.h"
#include <cmath>
#include <cstdio>

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
    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    return failures == 0 ? 0 : 1;
}
