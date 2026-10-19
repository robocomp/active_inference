// lidar_body_guard.h (Task 14): the LiDAR safety net that does not trust the anchor — body-footprint self-filter, the
// swept-path translation ceiling and the corner-sweep rotation ceiling on the raw returns, all unfloored.
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_lidar_body_guard.cpp
//     -I/home/pbustos/robocomp/robocomp_core/classes ../common/robot_footprint/*.cpp -o /tmp/tlbg
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tlbg        (one line; the robocomp_core -I is the buffer's threadpool)
#include "../src/lidar_body_guard.h"
#include "../src/lidar_guard_buffer.h"
#include "../src/controller_runtime_types.h"
#include "../../common/robot_footprint/robot_footprint.h"

#include <chrono>
#include <cmath>
#include <thread>
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
        const std::vector<Eigen::Vector2f> self{{0.001f, 0.0005f}, {0.26f, 0.16f}, {-0.249f, -0.10f}, {0.f, 0.229f}};
        const auto r = returns_of(self, body);
        CHECK(r.n_self == int(self.size()) and r.obstacles.empty());
        for (const auto& q : self) CHECK(rc::lidar_return_is_self(body, q));
        // Monotone (fix round 1): 2 cm OUTSIDE the hull, inside σ_r, is KEPT — it is contact, not the body.
        CHECK(not rc::lidar_return_is_self(body, Eigen::Vector2f(0.2716f + 0.02f, 0.f)));
        const auto band = returns_of({Eigen::Vector2f(0.2716f + 0.02f, 0.f)}, body);
        CHECK(band.obstacles.size() == 1 and band.n_band == 1 and band.n_self == 0);
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
            const bool nd = rc::lidar_return_is_self(body, pts[i]);
            old_dropped += od; new_dropped += nd;
            kept_new_dropped_old += (od and not nd);
            if (i < n_world) { world_old_dropped += od; world_new_dropped += nd; } else self_new_kept += not nd;
        }
        std::println("  try74 synthetic frame: {} world + {} self returns; old disc drops {} ({} of them world), footprint "
                     "drops {} ({} world, inside the hull); KEPT by the footprint that the disc deleted: {}; self returns kept: {}",
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

    // 6. Fix round 1, item 1 — MONOTONE: a return 0.02 m outside the hull (inside σ_r), robot closing on it ⇒ adv 0; moving
    //    away stays free. Ahead of the front face, and ahead of the front-left corner.
    for (const Eigen::Vector2f& q : {Eigen::Vector2f(0.f, 0.2300f + 0.02f), Eigen::Vector2f(corner + Eigen::Vector2f(0.005f, 0.02f))})
    {
        CHECK(rc::anchor_signed_distance(body, q) > 0.f and rc::anchor_signed_distance(body, q) < kSigma);
        const auto r = returns_of({q}, body);
        CHECK(r.obstacles.size() == 1);
        for (const float m : {0.f, kMargin})
        {
            CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, m, kAdec, kVmax) < 0.03f);
            CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, body, m, kAdec, kVmax) > 0.99f * kVmax);
        }
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
        // And closing further (1 cm, 0 cm outside): still 0 — no release in the last centimetres.
        for (const float k : {0.5f, 0.01f})
        {
            const Eigen::Vector2f qq = q + (k - 1.f) * 0.02f * Eigen::Vector2f(0.f, 1.f);
            if (rc::anchor_signed_distance(body, qq) < 0.f) continue;
            const auto rr = returns_of({qq}, body);
            CHECK(rr.obstacles.size() == 1);
            CHECK(rc::lidar_return_speed_limit(rr.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
        }
    }
    // 7. Fix round 1, item 2 — the deadlock's way out: a return pinning the front-left corner and one beside the left
    //    side's front: forward, CCW and CW all capped to 0. The backoff opens the min clearance over 0.5 s at 20 Hz.
    {
        std::vector<Eigen::Vector2f> pts{corner + Eigen::Vector2f(0.005f, 0.02f), Eigen::Vector2f(-0.2716f - 0.04f, 0.15f),
                                         Eigen::Vector2f(-0.2716f - 0.04f, 0.05f)};
        auto r = returns_of(pts, body);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
        const float w_ccw = rc::lidar_return_rot_limit(r.obstacles, +1.f, body, kMargin, kAdec, kWmax);
        const float w_cw = rc::lidar_return_rot_limit(r.obstacles, -1.f, body, kMargin, kAdec, kWmax);
        std::println("  pinned corner: fwd 0, |w| CCW {:.3f} CW {:.3f}", w_ccw, w_cw);
        CHECK(w_ccw == 0.f and w_cw == 0.f);
        const auto min_sd = [&](const std::vector<Eigen::Vector2f>& q)
        { float m = 1e9f; for (const auto& p : q) m = std::min(m, rc::anchor_signed_distance(body, p)); return m; };
        const float sd0 = min_sd(pts);
        const float dt = 0.05f;
        rc::AnchorCommand first;
        for (int k = 0; k < 10; ++k)
        {
            r = returns_of(pts, body);
            const auto b = rc::lidar_return_backoff(r.obstacles, body, kMargin, kSigma, kAdec, kVmax, kWmax, false);
            if (not b.has_value()) break;
            if (k == 0) first = *b;
            for (auto& p : pts) p = rc::anchor_point_after(p, b->side * dt, b->adv * dt, b->rot * dt);
        }
        const float sd1 = min_sd(pts);
        std::println("  backoff: first command adv {:+.3f} side {:+.3f} rot {:+.3f}; min clearance {:.3f} -> {:.3f} m over 0.5 s",
                     first.adv, first.side, first.rot, sd0, sd1);
        CHECK(first.adv < 0.f or first.rot != 0.f);
        CHECK(sd1 > sd0 + 0.01f);
        CHECK(rc::lidar_return_speed_limit(returns_of(pts, body).obstacles, kBack, body, kMargin, kAdec, kVmax) > 0.f);
        // No penetration of the band ⇒ no backoff.
        CHECK(not rc::lidar_return_backoff(returns_of({Eigen::Vector2f(0.f, 1.f)}, body).obstacles, body, kMargin, kSigma,
                                           kAdec, kVmax, kWmax, false).has_value());
    }
    // 8. The census and the command cap.
    {
        const auto r = returns_of({Eigen::Vector2f(0.f, 0.001f), Eigen::Vector2f(0.f, 0.25f), Eigen::Vector2f(0.6f, 0.f)}, body);
        const auto c = rc::lidar_guard_census(r, body, kMargin, kAdec, kVmax, kWmax);
        CHECK(c.n_self == 1 and c.n_band == 1 and c.n_kept == 2);
        CHECK(std::abs(c.nearest_sd - 0.02f) < 2e-3f and std::abs(c.nearest_bearing_deg) < 1.f);
        CHECK(c.v_fwd == 0.f and c.v_back > 0.99f * kVmax);
        float adv = 0.3f, side = 0.f, rot = 0.f;
        const auto cap = rc::lidar_cap_command(r.obstacles, body, kMargin, kAdec, kVmax, kWmax, adv, side, rot);
        CHECK(cap.bound_lin and adv == 0.f and not cap.bound_rot);
    }

    // 9. try76 fix: THE SESSION'S OWN PATH — a room-frame cloud put into a real rc::LidarPointBuffer (the obstacle tracker's
    //    converter shape), read back with rc::lidar_guard_returns_from_buffer on ControllerRobotPose::as_transform(), the
    //    accessor and transform ControllerSession::live_returns runs. Robot at the try74 contact pose, heading not 0.
    {
        ControllerRobotPose pose;
        pose.pos = {-4.566f, 9.53f};
        pose.theta = 1.2f;
        const Eigen::Affine2f room_T_robot = pose.as_transform();
        const std::vector<Eigen::Vector2f> robot_pts{
            corner + Eigen::Vector2f(0.005f, 0.02f),   // the jamb end in the band: contact
            Eigen::Vector2f(-0.30f, 0.30f),             // near, kept
            Eigen::Vector2f(0.001f, 0.0005f),           // r ≈ 0: the body
            Eigen::Vector2f(0.0f, 3.0f)};               // a far wall: beyond reach
        rc::RawLidarPointVectors raw;
        for (const auto& q : robot_pts) { const Eigen::Vector2f w = room_T_robot * q; raw.xs.push_back(w.x()); raw.ys.push_back(w.y()); raw.zs.push_back(0.9f); }
        raw.xs.push_back(std::numeric_limits<float>::quiet_NaN()); raw.ys.push_back(0.f); raw.zs.push_back(0.9f);
        rc::LidarPointBuffer buffer{5};
        CHECK(not rc::lidar_guard_returns_from_buffer(buffer, room_T_robot, body, kSigma, 0.436f).has_value());   // empty
        buffer.put<0>(std::move(raw), 1000, [](rc::RawLidarPointVectors&& in, rc::LidarPointVectors& out)
                      { auto& [x, y, z] = out; x = std::move(in.xs); y = std::move(in.ys); z = std::move(in.zs); });
        // put() converts on the buffer's own worker thread, as live: poll the read the session makes until it lands.
        const auto read_n = [&](int n_expected)
        {
            for (int k = 0; k < 200; ++k)
            {
                if (auto rr = rc::lidar_guard_returns_from_buffer(buffer, room_T_robot, body, kSigma, 0.436f);
                    rr.has_value() and rr->n_read == n_expected) return rr;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return rc::lidar_guard_returns_from_buffer(buffer, room_T_robot, body, kSigma, 0.436f);
        };
        const auto r = read_n(5);
        CHECK(r.has_value());
        if (r.has_value())
        {
            const auto c = rc::lidar_guard_census(*r, body, kMargin, kAdec, kVmax, kWmax);
            std::println("  session path: read {} non-finite {} far {} self {} kept {} band {}; nearest kept {:.3f} m at {:+.0f} deg, "
                         "any {:.3f} m; fwd cap {:.3f}", c.n_read, c.n_nonfinite, c.n_far, c.n_self, c.n_kept, c.n_band,
                         c.nearest_sd, c.nearest_bearing_deg, c.nearest_any_sd, c.v_fwd);
            CHECK(c.n_read == 5 and c.n_nonfinite == 1 and c.n_far == 1 and c.n_self == 1 and c.n_kept == 2 and c.n_band == 1);
            // The transform round-trips: the kept returns ARE the robot-frame points.
            bool found = false;
            for (const auto& o : r->obstacles) found = found or (o.mean - robot_pts[0]).norm() < 1e-4f;
            CHECK(found);
            CHECK(c.v_fwd == 0.f);
        }
        // Parked in the open: nothing within reach — the census must still say where the world is, not "-1".
        rc::RawLidarPointVectors open;
        for (const auto& q : {Eigen::Vector2f(0.f, 1.5f), Eigen::Vector2f(-1.2f, 0.f)})
        { const Eigen::Vector2f w = room_T_robot * q; open.xs.push_back(w.x()); open.ys.push_back(w.y()); open.zs.push_back(0.9f); }
        buffer.put<0>(std::move(open), 2000, [](rc::RawLidarPointVectors&& in, rc::LidarPointVectors& out)
                      { auto& [x, y, z] = out; x = std::move(in.xs); y = std::move(in.ys); z = std::move(in.zs); });
        const auto ro = read_n(2);
        CHECK(ro.has_value());
        if (ro.has_value())
        {
            const auto c = rc::lidar_guard_census(*ro, body, kMargin, kAdec, kVmax, kWmax);
            CHECK(c.n_read == 2 and c.n_far == 2 and c.n_kept == 0);
            CHECK(std::abs(c.nearest_any_sd - (1.2f - 0.2716f)) < 0.01f and std::abs(c.nearest_any_bearing_deg + 90.f) < 1.f);
        }
    }

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
