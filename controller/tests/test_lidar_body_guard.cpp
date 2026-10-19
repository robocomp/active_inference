// lidar_body_guard.h (Task 14): the LiDAR safety net that does not trust the anchor — body-footprint self-filter, the
// swept-path translation ceiling and the corner-sweep rotation ceiling on the raw returns, all unfloored.
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_lidar_body_guard.cpp
//     ../common/robot_footprint/*.cpp -o /tmp/tlbg -lfastdds -lfastcdr -lQt5Core && /tmp/tlbg        (one line)
#include "../src/lidar_body_guard.h"
#include "../../common/robot_footprint/robot_footprint.h"

#include <cmath>
#include <print>
#include <string_view>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

constexpr float kSigma = 0.03f, kMargin = 0.05f, kAdec = 1.0f, kVmax = 0.35f, kWmax = 1.0f;
const Eigen::Vector2f kFwd(0.f, 1.f), kBack(0.f, -1.f);

rc::LidarGuardReturns returns_of(const std::vector<Eigen::Vector2f>& pts, std::span<const Eigen::Vector2f> body)
{ return rc::lidar_guard_returns(pts, body, kSigma, 10.f); }
}  // namespace

int main()
{
    const auto fp = rc::RobotFootprint::shadow();
    const auto& body = fp.polygon();
    // The front-left corner of the hull and its outward direction.
    const Eigen::Vector2f corner(-0.2536f, 0.2036f);

    // 1. A jamb point 0.05 m outside the footprint corner, AHEAD of it on the path: driving toward it stops.
    {
        const Eigen::Vector2f jamb = corner + Eigen::Vector2f(0.01f, 0.05f);   // ahead of the corner, inside the corridor
        CHECK(rc::anchor_signed_distance(body, jamb) > 0.04f);
        const auto r = returns_of({jamb}, body);
        CHECK(r.obstacles.size() == 1 and r.n_self == 0);
        CHECK(rc::lidar_return_dropped_by_disc(jamb, fp.circumscribed_radius()));   // the old filter deleted it
        const float v = rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax);
        std::println("  jamb 0.05 m ahead of the front-left corner: forward ceiling {:.4f} m/s (margin {:.2f})", v, kMargin);
        CHECK(v < 0.01f);
        // Without the standoff it is the stopping law on the 5 cm gap — and it never floors at 0.05.
        const float v0 = rc::lidar_return_speed_limit(r.obstacles, kFwd, body, 0.f, kAdec, kVmax);
        CHECK(v0 > 0.f and v0 <= std::sqrt(2.f * kAdec * 0.05f) + 1e-4f);
        // Closer (just outside the self-filter): slower without the standoff, exactly 0 with it — no 0.05 m/s floor.
        const auto rc0 = returns_of({corner + Eigen::Vector2f(0.01f, 0.04f)}, body);
        CHECK(rc0.obstacles.size() == 1);
        CHECK(rc::lidar_return_speed_limit(rc0.obstacles, kFwd, body, 0.f, kAdec, kVmax) < v0);
        CHECK(rc::lidar_return_speed_limit(rc0.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
        // Reversing away from it is free.
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
        // Far ahead on the same line: full speed.
        const auto far = returns_of({jamb + Eigen::Vector2f(0.f, 1.0f)}, body);
        CHECK(rc::lidar_return_speed_limit(far.obstacles, kFwd, body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
    }
    // 2. The same distance outside the body BESIDE it (off the swept path): forward motion is not stopped.
    {
        for (const float y : {-0.10f, 0.0f, 0.10f, 0.15f})
        {
            const Eigen::Vector2f beside(-0.2716f - 0.05f, y);
            const auto r = returns_of({beside}, body);
            CHECK(r.obstacles.size() == 1);
            const float v = rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax);
            std::println("  return 0.05 m beside the left side at y {:+.2f}: forward ceiling {:.3f} m/s", y, v);
            CHECK(v > 0.85f * kVmax);
        }
        // Behind the body, on the corridor: forward is free, reverse is capped.
        const auto r = returns_of({Eigen::Vector2f(0.f, -0.23f - 0.05f)}, body);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, body, kMargin, kAdec, kVmax) < 0.01f);
    }
    // 3. Body-own points inside the footprint are filtered: the r ≈ 0 population, a wheel's outer face, a point on the hull.
    {
        const std::vector<Eigen::Vector2f> self{{0.001f, 0.0005f}, {0.26f, 0.16f}, {-0.249f, -0.10f}, {0.f, 0.229f},
                                                {0.2716f + 0.02f, 0.0f}};   // the last: 2 cm out, inside σ_r
        const auto r = returns_of(self, body);
        CHECK(r.n_self == int(self.size()) and r.obstacles.empty());
        for (const auto& q : self) CHECK(rc::lidar_return_is_self(body, kSigma, q));
        CHECK(not rc::lidar_return_is_self(body, kSigma, Eigen::Vector2f(0.2716f + 0.04f, 0.f)));
    }
    // 4. Rotation: a turn cannot grind a corner into a real return; the turn away stays free.
    {
        // Just beside the front of the left side; a CCW turn swings the left side's front leftward onto it.
        const Eigen::Vector2f q(-0.2716f - 0.04f, 0.15f);
        const auto r = returns_of({q}, body);
        CHECK(r.obstacles.size() == 1);
        const float w_ccw = rc::lidar_return_rot_limit(r.obstacles, +1.f, body, kMargin, kAdec, kWmax);
        const float w_cw = rc::lidar_return_rot_limit(r.obstacles, -1.f, body, kMargin, kAdec, kWmax);
        std::println("  return 4 cm beside the left side front: |w| ceiling CCW {:.3f}, CW {:.3f} rad/s", w_ccw, w_cw);
        CHECK(w_ccw < 0.05f);
        CHECK(w_cw > 0.9f * kWmax);
        // Out of the corners' reach: no cap either way.
        const auto far = returns_of({Eigen::Vector2f(-0.6f, 0.15f)}, body);
        CHECK(rc::lidar_return_rot_limit(far.obstacles, +1.f, body, kMargin, kAdec, kWmax) == kWmax);
    }
    // 5. The try74 contact, synthetic: the front-left corner 1 cm short of the east jamb end of a 0.90 m opening, 1 cm
    //    sampling of the near wall face, the jamb faces, and body self-returns. Count what the old disc deleted.
    {
        std::vector<Eigen::Vector2f> pts;
        const float y_w = 0.2036f + 0.04f, x_j = -0.2536f + 0.02f, w = 0.90f, t = 0.10f;
        for (float x = -2.f; x <= x_j; x += 0.01f) pts.emplace_back(x, y_w);                  // left near face
        for (float x = x_j + w; x <= 2.f; x += 0.01f) pts.emplace_back(x, y_w);               // right near face
        for (float y = y_w; y <= y_w + t; y += 0.01f) { pts.emplace_back(x_j, y); pts.emplace_back(x_j + w, y); }   // jamb faces
        const int n_world = int(pts.size());
        for (const Eigen::Vector2f& s : {Eigen::Vector2f(0.001f, 0.f), Eigen::Vector2f(-0.0008f, 0.0012f),
                                        Eigen::Vector2f(0.249f, 0.16f), Eigen::Vector2f(-0.249f, 0.16f),
                                        Eigen::Vector2f(0.249f, -0.10f), Eigen::Vector2f(-0.249f, -0.10f)})
            pts.push_back(s);
        int old_dropped = 0, new_dropped = 0, kept_new_dropped_old = 0, world_old_dropped = 0, world_new_dropped = 0, self_new_kept = 0;
        for (int i = 0; i < int(pts.size()); ++i)
        {
            const bool od = rc::lidar_return_dropped_by_disc(pts[i], fp.circumscribed_radius());
            const bool nd = rc::lidar_return_is_self(body, kSigma, pts[i]);
            old_dropped += od; new_dropped += nd;
            kept_new_dropped_old += (od and not nd);
            if (i < n_world) { world_old_dropped += od; world_new_dropped += nd; } else self_new_kept += not nd;
        }
        std::println("  try74 synthetic frame: {} world + {} self returns; old disc drops {} ({} of them world), footprint "
                     "drops {} ({} world, within σ_r of the hull); KEPT by the footprint that the disc deleted: {}; self returns kept: {}",
                     n_world, int(pts.size()) - n_world, old_dropped, world_old_dropped, new_dropped, world_new_dropped,
                     kept_new_dropped_old, self_new_kept);
        CHECK(self_new_kept == 0);
        CHECK(kept_new_dropped_old == world_old_dropped - world_new_dropped and kept_new_dropped_old > 0);
        const auto r = rc::lidar_guard_returns(pts, body, kSigma, 0.6f);
        const float v = rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax);
        const float w_ccw = rc::lidar_return_rot_limit(r.obstacles, +1.f, body, kMargin, kAdec, kWmax);
        const float w_cw = rc::lidar_return_rot_limit(r.obstacles, -1.f, body, kMargin, kAdec, kWmax);
        std::println("  try74 synthetic frame: forward {:.4f} m/s; |w| CCW {:.3f} CW {:.3f}; {} returns in reach, {} merged cells",
                     v, w_ccw, w_cw, r.obstacles.size() + r.n_self, r.obstacles.size());
        CHECK(v < 0.01f);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
    }

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
