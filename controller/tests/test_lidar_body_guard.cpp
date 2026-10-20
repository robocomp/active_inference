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
#include "../src/lidar_height_band.h"

#include <cstdlib>
#include <numbers>
#include <random>
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
    // ★ 2026-10-20: Shadow's body is now everything Shadow.proto draws or collides on (shadow.stl, the two drive wheels, the
    // 0.44 x 0.46 boundingObject box): side x = ±0.2840 over y ∈ ±0.10, then the box's square corner at (±0.22, ±0.23). The front-left
    // vertex is (-0.22, 0.23) (was (-0.2536, 0.2036) on the old four-mecanum hull, (-0.2141, 0.1901) on the mesh-only hull).
    const Eigen::Vector2f corner(-0.22f, 0.23f);

    // 1. A jamb point 0.05 m outside the footprint corner, AHEAD of it on the path: driving toward it stops.
    {
        const Eigen::Vector2f jamb = corner + Eigen::Vector2f(0.0f, 0.05f);   // 0.05 m straight ahead of the front-left vertex of the corner, inside the corridor
        CHECK(rc::anchor_signed_distance(body, jamb) > 0.03f);   // ★ was 0.04: the chamfer is 0.037 m off along its normal
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
        const auto rc0 = returns_of({corner + Eigen::Vector2f(0.0f, 0.04f)}, body);
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
            const Eigen::Vector2f beside(-0.2840f - 0.05f, y);
            const auto r = returns_of({beside}, body);
            CHECK(r.obstacles.size() == 1);
            const float v = rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax);
            std::println("  return 0.05 m beside the left side at y {:+.2f}: forward ceiling {:.3f} m/s", y, v);
            CHECK(v > 0.85f * kVmax);
        }
        // Behind the body, on the corridor: forward is free, reverse is capped.
        const auto r = returns_of({Eigen::Vector2f(0.f, -0.2292f - 0.045f)}, body);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
        CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, body, kMargin, kAdec, kVmax) < 0.01f);
    }
    // 3. Body-own points inside the footprint are filtered: the r ≈ 0 population, a wheel's outer face, a point on the hull.
    {
        const std::vector<Eigen::Vector2f> self{{0.001f, 0.0005f}, {0.28f, 0.05f}, {-0.249f, -0.10f}, {0.f, 0.229f}};
        const auto r = returns_of(self, body);
        CHECK(r.n_self == int(self.size()) and r.obstacles.empty());
        for (const auto& q : self) CHECK(rc::lidar_return_is_self(body, q));
        // Monotone (fix round 1): 2 cm OUTSIDE the hull, inside σ_r, is KEPT — it is contact, not the body.
        CHECK(not rc::lidar_return_is_self(body, Eigen::Vector2f(0.2840f + 0.02f, 0.f)));
        const auto band = returns_of({Eigen::Vector2f(0.2840f + 0.02f, 0.f)}, body);
        CHECK(band.obstacles.size() == 1 and band.n_band == 1 and band.n_self == 0);
    }
    // 4. Rotation: a turn cannot grind a corner into a real return; the turn away stays free.
    {
        // Just beside the front of the left side; a CCW turn swings the left side's front leftward onto it.
        const Eigen::Vector2f q(-0.2840f - 0.04f, 0.09f);
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
        const float y_w = 0.23f + 0.04f, x_j = -0.22f + 0.02f, w = 0.90f, t = 0.10f;
        for (float x = -2.f; x <= x_j; x += 0.01f) pts.emplace_back(x, y_w);                  // left near face
        for (float x = x_j + w; x <= 2.f; x += 0.01f) pts.emplace_back(x, y_w);               // right near face
        for (float y = y_w; y <= y_w + t; y += 0.01f) { pts.emplace_back(x_j, y); pts.emplace_back(x_j + w, y); }   // jamb faces
        const int n_world = int(pts.size());
        for (const Eigen::Vector2f& s : {Eigen::Vector2f(0.001f, 0.f), Eigen::Vector2f(-0.0008f, 0.0012f),
                                        Eigen::Vector2f(0.28f, 0.09f), Eigen::Vector2f(-0.28f, 0.09f),
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
        // ★ the pin sits at the WHEEL's front corner (-0.284, 0.10): the new hull is rounder above it, so a point on the chamfer is
        // tangent to a CCW swing and no rotation could open it (that is the body's shape, not a defect of the backoff).
        std::vector<Eigen::Vector2f> pts{Eigen::Vector2f(-0.2840f + 0.004f, 0.10f + 0.02f), Eigen::Vector2f(-0.2840f - 0.04f, 0.09f),
                                         Eigen::Vector2f(-0.2840f - 0.04f, 0.05f)};
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
        // ★ 0.01 -> 0.005 (2026-10-20): the two-wheel hull is rounder above the wheel, so the swing the backoff picks (rot only)
        // opens the pinned point by 0.007 m in 0.5 s rather than >0.01; it still opens, which is the invariant.
        CHECK(sd1 > sd0 + 0.005f);
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
            CHECK(std::abs(c.nearest_any_sd - (1.2f - 0.2840f)) < 0.01f and std::abs(c.nearest_any_bearing_deg + 90.f) < 1.f);
        }
    }

    // 10. Fix round 3 (try77) — LAG: the buffer is room-frame and is read on the CURRENT pose. A robot turning CW at 1 rad/s,
    //     its own front-face returns captured 80 ms ago: re-expressed on the current pose they land OUTSIDE the front face,
    //     in the contact band, and stop the robot (the old path). Dropped at capture in the rigid robot frame
    //     (rc::lidar_capture_is_self, the obstacle tracker's converter) they never reach the buffer: no contact. A real jamb
    //     2 cm outside the hull NOW, on the turn's side, is still contact.
    {
        ControllerRobotPose cap_pose, now_pose;
        cap_pose.pos = now_pose.pos = {-4.566f, 9.53f};
        cap_pose.theta = 1.2f;
        now_pose.theta = 1.2f - 1.0f * 0.080f;   // CW at 1 rad/s for 80 ms
        const Eigen::Affine2f T_cap = cap_pose.as_transform(), T_now = now_pose.as_transform();
        const std::vector<Eigen::Vector2f> self_cap{{0.10f, 0.225f}, {0.0f, 0.226f}, {-0.10f, 0.225f}};   // front face, inside
        for (const auto& q : self_cap) CHECK(rc::lidar_capture_is_self(body, fp.circumscribed_radius(), q));
        // The jamb, 2 cm ahead of the front face in the CURRENT frame (the turn swung the face toward it since capture).
        const Eigen::Vector2f jamb_now(0.05f, 0.2299f + 0.02f);
        const Eigen::Vector2f jamb_room = T_now * jamb_now;
        const Eigen::Vector2f jamb_cap = T_cap.inverse() * jamb_room;
        CHECK(not rc::lidar_capture_is_self(body, fp.circumscribed_radius(), jamb_cap));   // a real return survives capture
        const auto converter = [](rc::RawLidarPointVectors&& in, rc::LidarPointVectors& out)
        { auto& [x, y, z] = out; x = std::move(in.xs); y = std::move(in.ys); z = std::move(in.zs); };
        const auto sweep = [&](bool capture_filter, bool with_jamb)
        {
            rc::RawLidarPointVectors raw;
            auto add = [&](const Eigen::Vector2f& q_cap)
            {
                if (capture_filter and rc::lidar_capture_is_self(body, fp.circumscribed_radius(), q_cap)) return;
                const Eigen::Vector2f w = T_cap * q_cap;   // placed with the CAPTURE pose, as the tracker does
                raw.xs.push_back(w.x()); raw.ys.push_back(w.y()); raw.zs.push_back(0.9f);
            };
            for (const auto& q : self_cap) add(q);
            if (with_jamb) add(jamb_cap);
            rc::LidarPointBuffer buf{5};
            const int n = int(raw.xs.size());
            buf.put<0>(std::move(raw), 1000, converter);
            std::optional<rc::LidarGuardReturns> r;
            for (int k = 0; k < 200; ++k)
            {
                r = rc::lidar_guard_returns_from_buffer(buf, T_now, body, kSigma, 0.436f);   // read on the CURRENT pose
                if (r.has_value() and r->n_read == n) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return r;
        };
        const auto old_path = sweep(false, false);
        const auto new_path = sweep(true, false);
        const auto new_jamb = sweep(true, true);
        CHECK(old_path.has_value() and new_path.has_value() and new_jamb.has_value());
        if (old_path.has_value() and new_path.has_value() and new_jamb.has_value())
        {
            const float v_old = rc::lidar_return_speed_limit(old_path->obstacles, kFwd, body, kMargin, kAdec, kVmax);
            const float v_new = rc::lidar_return_speed_limit(new_path->obstacles, kFwd, body, kMargin, kAdec, kVmax);
            const float v_jamb = rc::lidar_return_speed_limit(new_jamb->obstacles, kFwd, body, kMargin, kAdec, kVmax);
            std::println("  lag, CW 1 rad/s x 80 ms: own front-face returns read on the current pose -> {} in the band, fwd {:.3f} "
                         "(old path); dropped at capture -> {} kept, fwd {:.3f}; + a real jamb 2 cm out -> band {}, fwd {:.3f}",
                         old_path->n_band, v_old, new_path->obstacles.size(), v_new, new_jamb->n_band, v_jamb);
            CHECK(old_path->n_band > 0 and v_old == 0.f);           // the try77 artefact, reproduced
            CHECK(new_path->obstacles.empty() and v_new == kVmax);  // no contact from the robot's own returns
            CHECK(new_jamb->n_band == 1 and v_jamb == 0.f);         // the real jamb still stops it
            CHECK(new_jamb->band_points.size() == 1 and (new_jamb->band_points[0] - jamb_now).norm() < 1e-4f);
        }
    }

    // 11. Fix round 4 (try79) — HEMMED IN: a wall 0.05 m off the left side (an infinite wall, 1 cm sampling) and a planner
    //     target BEHIND the robot. The net refuses both turns (the side's ends and the corners swing into the margin), so
    //     the tracker can only sit. The returns' backoff (turn / drive / turn-then-drive, contact standoff) opens clearance
    //     at 20 Hz until the band is clear; the body grown by the safety margin (the planner's own footprint test) is then
    //     free at the start pose, and the turn toward the target is no longer refused.
    {
        constexpr float kGap = 0.04f;   // inside the 0.05 m margin: the planner's footprint test refuses the start pose
        auto fpm = rc::RobotFootprint::shadow();
        fpm.set_safety_margin(kMargin);
        std::vector<Eigen::Vector2f> pts;
        for (float y = -1.5f; y <= 1.5f; y += 0.01f) pts.emplace_back(-0.2840f - kGap, y);
        const auto r0 = returns_of(pts, body);
        const float w0_ccw = rc::lidar_return_rot_limit(r0.obstacles, +1.f, body, kMargin, kAdec, kWmax);
        const float w0_cw = rc::lidar_return_rot_limit(r0.obstacles, -1.f, body, kMargin, kAdec, kWmax);
        bool start_blocked = false;
        for (const auto& q : pts) start_blocked = start_blocked or fpm.contains(q);
        CHECK(w0_ccw < 0.5f * kWmax and w0_cw < 0.5f * kWmax and start_blocked);
        const auto min_sd = [&](const std::vector<Eigen::Vector2f>& q)
        { float m = 1e9f; for (const auto& p : q) m = std::min(m, rc::anchor_signed_distance(body, p)); return m; };
        const float sd0 = min_sd(pts);
        float sd_lo = sd0;
        int steps = 0;
        for (; steps < 400; ++steps)   // 20 s at 20 Hz, receding horizon
        {
            const auto b = rc::lidar_return_backoff(returns_of(pts, body).obstacles, body, kMargin, kSigma, kAdec, kVmax, kWmax, false);
            if (not b.has_value()) break;
            for (auto& p : pts) p = rc::anchor_point_after(p, b->side * 0.05f, b->adv * 0.05f, b->rot * 0.05f);
            sd_lo = std::min(sd_lo, min_sd(pts));
        }
        const float sd1 = min_sd(pts);
        bool start_free = true;
        for (const auto& q : pts) start_free = start_free and not fpm.contains(q);
        const auto r1 = returns_of(pts, body);
        const float w1_ccw = rc::lidar_return_rot_limit(r1.obstacles, +1.f, body, kMargin, kAdec, kWmax);
        const float w1_cw = rc::lidar_return_rot_limit(r1.obstacles, -1.f, body, kMargin, kAdec, kWmax);
        std::println("  hemmed in (wall 4 cm off the side): turns CCW {:.3f} CW {:.3f}, start in collision {}; backoff {} steps "
                     "({:.1f} s): min clearance {:.3f} -> {:.3f} m (never below {:.3f}); start free {}, turns CCW {:.3f} CW {:.3f}",
                     w0_ccw, w0_cw, start_blocked, steps, steps * 0.05f, sd0, sd1, sd_lo, start_free, w1_ccw, w1_cw);
        CHECK(steps < 400);                       // it terminates: the band is clear
        CHECK(sd1 >= kMargin + kSigma - 1e-3f);   // the margin and the noise band given back
        CHECK(sd_lo > 0.f);                       // never touched the wall on the way
        CHECK(start_free);                        // the planner's footprint test accepts the start pose
        CHECK(w1_ccw > 0.f or w1_cw > 0.f);       // and the turn toward the target behind is allowed one way
    }

    // 11b. ★ 2026-10-20 (Task 1b): the file the SESSION now loads for the outline is the robot node's `path` = shadow_footprint.obj
    //      (shadow.stl + wheels + casters + housings + boundingObject boxes). Nothing in Shadow.proto reaches the +0.2471 front
    //      cover of the DISPLAY mesh shadow.obj, so in simulation this outline lies inside the planning hull and adds no front
    //      cover. This pins that, and that a return just outside the simulated front is contact for the guard.
    {
        rc::RobotFootprint::MeshReport rep;
        const auto sim = rc::RobotFootprint::from_obj("/home/pbustos/robocomp/components/webots-shadow/protos/meshes/shadow_footprint.obj", 0.f, rep);
        if (not sim.has_value()) std::println("  SKIP 11b: shadow_footprint.obj absent ({})", rep.reason);
        else
        {
            float fy = 0.f; for (const auto &q : sim->polygon()) fy = std::max(fy, q.y());
            float by = 0.f; for (const auto &q : body) by = std::max(by, q.y());
            std::println("  session outline (shadow_footprint.obj): front {:.4f} vs hull {:.4f}", fy, by);
            CHECK(fy <= by + 1e-3f);
            const auto r = returns_of({Eigen::Vector2f(0.f, by + 0.02f)}, body);
            CHECK(r.obstacles.size() == 1);
            CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
        }
    }

    // 12. Fix round 5 (try78/79/80): the hull misses the robot's own FRONT. ★ 2026-10-20: this block and 13 exercise the REAL
    //     robot's display mesh (shadow.obj, front cover at +0.2471), which Shadow.proto does not draw; they pin that the guard still
    //     handles a mesh that pokes out of the hull, so they keep loading shadow.obj on purpose. The self footprint from the REAL mesh
    //     (robot_concept/meshes/shadow.obj, read as the controller reads it) ∪ the planning hull, ⊕ σ_r on the mesh.
    {
        rc::RobotFootprint::MeshReport rep;
        const auto mesh = rc::RobotFootprint::from_obj("../robot_concept/meshes/shadow.obj", 0.f, rep);
        CHECK(mesh.has_value());
        if (mesh.has_value())
        {
            const auto ret_body = rc::lidar_returns_body(body, mesh->polygon());
            const auto self_fp = rc::lidar_self_footprint(body, mesh->polygon(), kSigma);
            const auto front = [](const std::vector<Eigen::Vector2f>& p) { float y = -1.f; for (const auto& v : p) y = std::max(y, v.y()); return y; };
            const auto side = [](const std::vector<Eigen::Vector2f>& p) { float x = 0.f; for (const auto& v : p) x = std::max(x, std::abs(v.x())); return x; };
            std::println("  footprints: hull front {:.4f} |x| {:.4f}; mesh front {:.4f} |x| {:.4f}; returns body front {:.4f} |x| {:.4f}; "
                         "self front {:.4f} |x| {:.4f}", front(body), side(body), front(mesh->polygon()), side(mesh->polygon()),
                         front(ret_body), side(ret_body), front(self_fp), side(self_fp));
            CHECK(front(mesh->polygon()) > front(body) + 0.015f);   // the hull is short at the front
            // try81: the outline was read with P3Bot's +90 deg mesh yaw and came out with its axes swapped. shadow.obj is
            // in the robot frame (x right, y forward): its front (0.2472) exceeds its half-width (0.2420) at yaw 0, and a
            // +90 read inverts that — the signature of the swap.
            CHECK(front(mesh->polygon()) > side(mesh->polygon()));
            {
                rc::RobotFootprint::MeshReport rep90;
                const auto m90 = rc::RobotFootprint::from_obj("../robot_concept/meshes/shadow.obj", 1.5707963f, rep90);
                CHECK(m90.has_value() and front(m90->polygon()) < side(m90->polygon()));
            }
            CHECK(side(self_fp) < side(body) + 0.005f);             // the self footprint does not eat the sides
            // The two front-centre returns logged in try78/try80 are SELF now. The try79 one at the front-RIGHT corner
            // (+0.252, +0.236) is NOT: it is 1.1 cm outside even the mesh ⊕ σ_r and ~6 cm outside the STL the simulator
            // renders — unexplained, and deliberately not absorbed by widening anything (see the fix report).
            for (const Eigen::Vector2f& q : {Eigen::Vector2f(0.007f, 0.260f), Eigen::Vector2f(0.123f, 0.258f)})
                CHECK(rc::anchor_point_in_convex(self_fp, q));
            CHECK(not rc::anchor_point_in_convex(self_fp, Eigen::Vector2f(0.252f, 0.236f)));
            // Returns at y +0.26 across the front width, in open space: self ⇒ forward free.
            std::vector<Eigen::Vector2f> front_pts;
            for (float x = -0.20f; x <= 0.2001f; x += 0.02f) front_pts.emplace_back(x, 0.26f);
            const auto r = rc::lidar_guard_returns(front_pts, ret_body, kSigma, 0.6f, self_fp);
            const float v = rc::lidar_return_speed_limit(r.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax);
            std::println("  front returns at y +0.26: {} self, {} kept; forward {:.3f}", r.n_self, r.obstacles.size(), v);
            CHECK(r.n_self == int(front_pts.size()) and r.obstacles.empty() and v == kVmax);
            // The old footprint (hull only) would have kept them and stopped the robot.
            const auto r_old = rc::lidar_guard_returns(front_pts, body, kSigma, 0.6f);
            CHECK(rc::lidar_return_speed_limit(r_old.obstacles, kFwd, body, kMargin, kAdec, kVmax) == 0.f);
            // A real obstacle at y +0.32 still blocks.
            const std::vector<Eigen::Vector2f> real_pts{Eigen::Vector2f(0.0f, 0.32f), Eigen::Vector2f(0.1f, 0.32f)};
            const auto rb = rc::lidar_guard_returns(real_pts, ret_body, kSigma, 0.6f, self_fp);
            const float vb = rc::lidar_return_speed_limit(rb.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax);
            std::println("  real obstacle at y +0.32: kept {}, forward {:.3f} (stopping law on a 7.3 cm gap)", rb.obstacles.size(), vb);
            CHECK(rb.obstacles.size() == 2 and vb < 0.5f * kVmax);   // capped by the stopping law on its 7 cm gap
            // Closer, inside the standoff but still outside the self footprint (y +0.29): stops.
            const std::vector<Eigen::Vector2f> near_pts{Eigen::Vector2f(0.0f, 0.29f)};
            const auto rn = rc::lidar_guard_returns(near_pts, ret_body, kSigma, 0.6f, self_fp);
            CHECK(rn.obstacles.size() == 1 and rc::lidar_return_speed_limit(rn.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax) == 0.f);
            // The contact standoff (margin from the RETURNS body) brakes before the self footprint's front.
            CHECK(front(ret_body) + kMargin > front(self_fp));
        }
    }

    // 13. try104 — "the robot repeatedly hits the wall when the target is to its left". The REAL Shadow bodies: the planning
    //     hull (what the simulator renders and collides, front 0.2300), the mesh outline (robot_concept/meshes/shadow.obj),
    //     the returns body and the round-5 self footprint built from them as the session builds them. TRUE geometry for
    //     contact = the hull ∪ the two differential drive wheels of webots-shadow/protos/Shadow.proto (axle |x| 0.259,
    //     R 0.10, width 0.05 ⇒ |x| ≤ 0.284, |y| ≤ 0.10; the compiled hull stops at 0.2716 — see the report). A wall
    //     (segment), sampled every 1 cm, returns drawn along the ray from the centre with N(0, σ) range noise (fixed seed),
    //     the cloud put in the ROOM frame and read back through lidar_guard_returns_room on the moving pose, as live. The
    //     robot asks for an in-place turn at the tracker's 0.58 rad/s; the command goes through the session's chain
    //     (rc::lidar_guard_command: the turn's room, then the caps; a command held whole goes to the returns' backoff);
    //     the base executes it ONE CYCLE LATE (65 ms, the measured control period). Contact = a noise-free wall point
    //     inside the TRUE body, checked at 5 sub-steps per cycle.
    {
        rc::RobotFootprint::MeshReport rep;
        const auto mesh = rc::RobotFootprint::from_obj("../robot_concept/meshes/shadow.obj", 0.f, rep);
        CHECK(mesh.has_value());
        if (mesh.has_value())
        {
            const auto ret_body = rc::lidar_returns_body(body, mesh->polygon());
            const auto self_fp = rc::lidar_self_footprint(body, mesh->polygon(), kSigma);
            std::vector<Eigen::Vector2f> true_pts(body.begin(), body.end());
            for (const float sx : {-1.f, 1.f})
                for (const float y : {-0.10f, 0.10f}) { true_pts.emplace_back(sx * 0.234f, y); true_pts.emplace_back(sx * 0.284f, y); }
            const auto true_body = rc::mesh::convex_hull_2d(std::move(true_pts));
            constexpr float kDt = 0.065f, kAsk = 0.58f;
            // A wall segment from a to b (room frame; the start pose is the room origin, heading +y).
            struct Wall { Eigen::Vector2f a, b; };
            struct Outcome { bool contact = false; float min_true_sd = 1e9f; float turned = 0.f; int held = 0, backoff = 0, room = 0;
                             int deleted_near = 0; };
            enum class Rule { Old, New };
            const auto run = [&](const Wall& wall, float sign, Rule rule, float sigma_gen, int cycles, float prime_back)
            {
                std::mt19937 rng(104);
                std::normal_distribution<float> noise(0.f, sigma_gen);
                rc::LidarWorldMemory memory;
                Eigen::Vector2f pos(0.f, 0.f);
                float th = 0.f;
                float pending_adv = 0.f, pending_rot = 0.f;   // last cycle's command: what the base executes now
                Outcome o;
                const auto T = [&] { return Eigen::Affine2f(Eigen::Translation2f(pos) * Eigen::Rotation2Df(th)); };
                std::vector<Eigen::Vector2f> wall_fine;
                const float len = (wall.b - wall.a).norm();
                for (float s = 0.f; s <= len; s += 0.005f) wall_fine.push_back(wall.a + (s / len) * (wall.b - wall.a));
                const auto true_contact = [&]
                {
                    const Eigen::Affine2f inv = T().inverse();
                    bool c = false;
                    for (const auto& w : wall_fine)
                    {
                        const float sd = rc::anchor_signed_distance(true_body, inv * w);
                        o.min_true_sd = std::min(o.min_true_sd, sd);
                        c = c or sd < 0.f;
                    }
                    return c;
                };
                const auto sweep = [&]
                {
                    rc::RawLidarPointVectors raw;
                    const Eigen::Affine2f Tn = T();
                    for (float s = 0.f; s <= len; s += 0.01f)
                    {
                        const Eigen::Vector2f w = wall.a + (s / len) * (wall.b - wall.a);
                        const Eigen::Vector2f q = Tn.inverse() * w;
                        const Eigen::Vector2f wn = Tn * Eigen::Vector2f(q * ((q.norm() + noise(rng)) / q.norm()));
                        raw.xs.push_back(wn.x()); raw.ys.push_back(wn.y()); raw.zs.push_back(0.9f);
                    }
                    return raw;
                };
                const auto read = [&](const rc::RawLidarPointVectors& raw)
                {
                    return rule == Rule::New
                        ? rc::lidar_guard_returns_room(raw.xs, raw.ys, T(), ret_body, kSigma, 0.6f, self_fp, body, &memory)
                        : rc::lidar_guard_returns_room(raw.xs, raw.ys, T(), ret_body, kSigma, 0.6f, self_fp);
                };
                // The approach: the robot saw this wall from `prime_back` further back (one sweep) before it got here.
                if (prime_back > 0.f)
                {
                    pos = Eigen::Vector2f(0.f, -prime_back);
                    (void)read(sweep());
                    pos = Eigen::Vector2f(0.f, 0.f);
                }
                o.contact = true_contact();
                for (int k = 0; k < cycles and not o.contact; ++k)
                {
                    for (int sub = 0; sub < 5 and not o.contact; ++sub)   // the base executes the PREVIOUS cycle's command
                    {
                        th += pending_rot * kDt / 5.f;
                        pos += Eigen::Rotation2Df(th) * Eigen::Vector2f(0.f, pending_adv * kDt / 5.f);
                        o.turned += pending_rot * kDt / 5.f;
                        o.contact = true_contact();
                    }
                    const auto r = read(sweep());
                    if (k == 0) o.deleted_near = r.n_self;
                    const auto g = rc::lidar_guard_command(r.obstacles, ret_body, kMargin, kAdec, kVmax, kWmax, false,
                                                           rc::AnchorCommand{0.f, 0.f, sign * kAsk, false});
                    float adv = g.cmd.adv, rot = g.cmd.rot;
                    o.room += g.turn_room;
                    if (adv == 0.f and rot == 0.f)
                    {
                        ++o.held;
                        if (const auto b = rc::lidar_return_backoff(r.obstacles, ret_body, kMargin, kSigma, kAdec, kVmax, kWmax, false);
                            b.has_value())
                        { ++o.backoff; adv = b->adv; rot = b->rot; }
                    }
                    pending_adv = adv; pending_rot = rot;
                    if (std::abs(o.turned) >= std::numbers::pi_v<float> / 2.f) break;   // turned to the target
                }
                return o;
            };
            const auto report = [](const char* what, float gap, float sg, float sign, const char* rule, const Outcome& o)
            {
                std::println("  try104 {} {:.3f} m [noise {:.2f}], turn {} — {}: contact {} (true clearance min {:+.3f} m), turned {:+.2f} rad, "
                             "deleted as self at first {}, held {}, turn-room {}, backoff {}", what, gap, sg, sign > 0 ? "LEFT " : "RIGHT",
                             rule, o.contact, o.min_true_sd, o.turned, o.deleted_near, o.held, o.room, o.backoff);
            };
            const float quarter = std::numbers::pi_v<float> / 2.f - 1e-3f;
            // (a) THE LIVE CASE: a surface ahead that the body's front has come to within 0.5–2 cm of — the try104 returns
            //     spanned body x −0.16..+0.16 at y 0.236–0.258. A 0.36 m panel, inside the round-5 self footprint (front
            //     0.2777) end to end, seen one sweep earlier from 6 cm back (the approach). The target 90° to the left, and
            //     to the right.
            for (const float sg : {0.01f, kSigma})
                for (const float gap : {0.005f, 0.01f, 0.02f})
                    for (const float sign : {+1.f, -1.f})
                    {
                        const float y = 0.2300f + gap;
                        const Wall w{Eigen::Vector2f(-0.18f, y), Eigen::Vector2f(0.18f, y)};
                        const auto o_old = run(w, sign, Rule::Old, sg, 400, 0.06f);
                        const auto o_new = run(w, sign, Rule::New, sg, 400, 0.06f);
                        report("panel AHEAD", gap, sg, sign, "OLD rule", o_old);
                        report("panel AHEAD", gap, sg, sign, "NEW rule", o_new);
                        if (sg < kSigma) CHECK(o_old.contact);   // the try104 failure, reproduced: the panel deleted as the robot
                        CHECK(not o_new.contact);                // never a turn into it
                        CHECK(std::abs(o_new.turned) >= quarter and o_new.turned * sign > 0.f);   // and the way out: it turns
                    }
            // (b) A LONG wall ahead (it runs past the body's sides: the world continues into the footprint), no approach seen.
            for (const float gap : {0.005f, 0.02f})
                for (const float sign : {+1.f, -1.f})
                {
                    const float y = 0.2300f + gap;
                    const auto o = run(Wall{Eigen::Vector2f(-1.5f, y), Eigen::Vector2f(1.5f, y)}, sign, Rule::New, 0.01f, 600, 0.f);
                    report("wall AHEAD", gap, 0.01f, sign, "NEW rule", o);
                    CHECK(not o.contact);
                    CHECK(std::abs(o.turned) >= quarter and o.turned * sign > 0.f);
                }
            // (c) A wall on the RIGHT, parallel to the side, at 1 cm, 3 cm and 0.2 m from the TRUE side (the wheel face), the
            //     target to the left (and right): the tail / nose swing into it is never commanded.
            for (const float gap : {0.01f, 0.03f, 0.20f})
                for (const float sign : {+1.f, -1.f})
                {
                    const float x = 0.284f + gap;
                    const auto o = run(Wall{Eigen::Vector2f(x, -1.5f), Eigen::Vector2f(x, 1.5f)}, sign, Rule::New, kSigma, 600, 0.f);
                    report("wall RIGHT", gap, kSigma, sign, "NEW rule", o);
                    CHECK(not o.contact);
                    if (gap >= 0.2f) CHECK(o.held == 0 and std::abs(o.turned) >= quarter);   // far enough: the turn is simply free
                }
            // (d) Round 5 is kept for what it was for: ISOLATED returns on the front cover (y +0.26, no world beside them, no
            //     memory) are still the robot under the new rule — forward stays free; and a wall at the same y that runs
            //     past the body's sides is the WORLD (it continues into returns outside the footprint) — forward stops.
            {
                std::vector<Eigen::Vector2f> cover, wall;
                for (float x = -0.20f; x <= 0.2001f; x += 0.02f) cover.emplace_back(x, 0.26f);
                for (float x = -0.60f; x <= 0.6001f; x += 0.02f) wall.emplace_back(x, 0.26f);
                const auto rc_ = rc::lidar_guard_returns(cover, ret_body, kSigma, 0.6f, self_fp, body);
                const auto rw = rc::lidar_guard_returns(wall, ret_body, kSigma, 0.6f, self_fp, body);
                const float vc = rc::lidar_return_speed_limit(rc_.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax);
                const float vw = rc::lidar_return_speed_limit(rw.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax);
                std::println("  new rule: isolated front-cover returns at y +0.26 -> {} self, fwd {:.3f}; a WALL at y +0.26 -> {} self, "
                             "{} kept, fwd {:.3f}", rc_.n_self, vc, rw.n_self, rw.obstacles.size(), vw);
                CHECK(rc_.n_self == int(cover.size()) and vc == kVmax);
                CHECK(rw.n_self == 0 and vw == 0.f);
                // Inside the TRUE body is the robot whatever is around it.
                const auto ri = rc::lidar_guard_returns(std::vector<Eigen::Vector2f>{{0.f, 0.20f}, {0.3f, 0.20f}}, ret_body, kSigma,
                                                        0.6f, self_fp, body);
                CHECK(ri.n_self == 1);
            }
            // (e) A world return the body already touches (inside the returns body, kept because the wall it belongs to runs
            //     past the sides) holds only the motion that deepens it: forward held, reverse free, and the backoff's
            //     rollout is not vetoed by the contact itself.
            {
                std::vector<Eigen::Vector2f> touch;
                for (float x = -0.60f; x <= 0.6001f; x += 0.02f) touch.emplace_back(x, 0.240f);
                const auto r = rc::lidar_guard_returns(touch, ret_body, kSigma, 0.6f, self_fp, body);
                CHECK(r.n_self == 0);
                CHECK(rc::lidar_return_speed_limit(r.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax) == 0.f);
                CHECK(rc::lidar_return_speed_limit(r.obstacles, kBack, ret_body, kMargin, kAdec, kVmax) > 0.99f * kVmax);
                const auto b = rc::lidar_return_backoff(r.obstacles, ret_body, kMargin, kSigma, kAdec, kVmax, kWmax, false);
                CHECK(b.has_value() and b->adv < 0.f);
                // The asked turn, either way, is refused in place and given its room: reverse while turning.
                for (const float sign : {+1.f, -1.f})
                {
                    const auto g = rc::lidar_guard_command(r.obstacles, ret_body, kMargin, kAdec, kVmax, kWmax, false,
                                                           rc::AnchorCommand{0.f, 0.f, sign * kAsk, false});
                    std::println("  touching a wall ahead, turn {} asked: turn-room {} ({} blockers) -> adv {:+.3f} rot {:+.3f}",
                                 sign > 0 ? "LEFT" : "RIGHT", g.turn_room, g.turn_blockers, g.cmd.adv, g.cmd.rot);
                    CHECK(g.turn_room and g.cmd.adv < 0.f);
                }
            }
        }
    }

    // low_returns_reach_policy (Task 7, local-motion policy): the policy's contact set starts just above the floor
    // (Lidar.ContactMinHeightM), not at the old consumers' 0.20 m band — a 0.10 m kerb, a robot-vacuum, a door sill or
    // a chair's foot-ring is CONTACT for the policy. The 0.20 band is kept for the old consumers until Task 10.
    {
        const ControllerParams p;
        const auto bands = rc::lidar_height_bands(p);
        CHECK(bands.legacy_min_m == p.temporary_obstacle_min_height_m);      // old consumers: unchanged
        CHECK(bands.legacy_min_m == 0.20f);
        CHECK(bands.contact_min_m < 0.10f);                                  // the low band reaches a 0.10 m return
        CHECK(bands.contact_min_m > 0.f);                                    // and still rejects the floor itself
        const auto low = rc::lidar_band_membership(0.10f, bands);
        std::println("  low_returns_reach_policy: z 0.10 m -> policy {} legacy {} (contact band from {:.3f} m, legacy from {:.2f} m)",
                     low.policy, low.legacy, bands.contact_min_m, bands.legacy_min_m);
        CHECK(low.policy and not low.legacy);
        const auto mid = rc::lidar_band_membership(0.50f, bands);            // above both floors: in both sets
        CHECK(mid.policy and mid.legacy);
        const auto floor = rc::lidar_band_membership(0.0f, bands);           // the floor return: in neither
        CHECK(not floor.policy and not floor.legacy);
        const auto high = rc::lidar_band_membership(p.temporary_obstacle_max_height_m + 0.1f, bands);   // above the body
        CHECK(not high.policy and not high.legacy);
        CHECK(not rc::lidar_band_membership(std::nanf(""), bands).policy);
    }

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
