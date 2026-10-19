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
#include "../src/self_return_map.h"

#include <cstdlib>
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
        for (float y = -1.5f; y <= 1.5f; y += 0.01f) pts.emplace_back(-0.2716f - kGap, y);
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

    // 12. Fix round 5 (try78/79/80): the hull misses the robot's own FRONT. The self footprint from the REAL mesh
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

    // 13. Fix round 6 (try79/try82): the self footprint LEARNED from motion (rc::SelfReturnMap). A body-frame-fixed point at
    //     the front-right corner (+0.246, +0.242) — outside the hull and the mesh ⊕ σ_r — becomes self while the robot
    //     drives and turns for 3 s; a WORLD-fixed jamb end approached at 0.1 m/s never does, and forward goes to 0 at contact.
    {
        rc::RobotFootprint::MeshReport rep;
        const auto mesh = rc::RobotFootprint::from_obj("../robot_concept/meshes/shadow.obj", 0.f, rep);
        CHECK(mesh.has_value());
        if (mesh.has_value())
        {
            const auto ret_body = rc::lidar_returns_body(body, mesh->polygon());
            const auto self_fp = rc::lidar_self_footprint(body, mesh->polygon(), kSigma);
            std::mt19937 rng(7);
            std::normal_distribution<float> noise(0.f, kSigma);
            // RANGE noise: along the beam from the bpearl (body (0, +0.14)), σ_r — a LiDAR return is uncertain in range, not
            // in bearing (0.4° azimuth step = 2 mm at these distances), so its scatter is a line, not a disc.
            const Eigen::Vector2f bpearl(0.f, 0.14f);
            const auto jitter = [&](Eigen::Vector2f q) { const Eigen::Vector2f u = (q - bpearl).normalized(); return Eigen::Vector2f(q + noise(rng) * u); };
            const auto weigh = [&](rc::LidarGuardReturns r, const rc::SelfReturnMap& m)
            { for (auto& o : r.obstacles) o.p = m.p_world(o.mean); return r; };
            const Eigen::Vector2f corner_pt(0.246f, 0.242f);
            CHECK(not rc::anchor_point_in_convex(self_fp, corner_pt));

            // (a) drive + turn for 3 s with the corner part on the body and a wall in the world.
            rc::SelfReturnMap map;
            map.configure(kSigma, ret_body, self_fp, kMargin);
            const float p0 = map.p_self(corner_pt);
            ControllerRobotPose pose;
            pose.pos = {1.f, 1.f};
            pose.theta = 0.3f;
            std::vector<Eigen::Vector2f> wall_world;   // a wall 1.5 m away: outside the domain, as in open space
            for (float x = -2.f; x <= 4.f; x += 0.05f) wall_world.emplace_back(x, 3.0f);
            for (int k = 0; k < 60; ++k)   // 3 s at 20 Hz: v 0.15 m/s, w 0.3 rad/s
            {
                const Eigen::Affine2f T = pose.as_transform();
                std::vector<Eigen::Vector2f> sweep{jitter(corner_pt)};
                for (const auto& w : wall_world) sweep.push_back(jitter(T.inverse() * w));
                const auto r = rc::lidar_guard_returns(sweep, ret_body, kSigma, 0.6f, self_fp);
                std::vector<Eigen::Vector2f> kept;
                for (const auto& o : r.obstacles) kept.push_back(o.mean);
                map.update(kept, T, 0.005f, k * 0.05);
                pose.theta += 0.3f * 0.05f;   // forward = robot +y = room direction theta + 90 deg
                pose.pos += 0.15f * 0.05f * Eigen::Vector2f(-std::sin(pose.theta), std::cos(pose.theta));
            }
            const float p1 = map.p_self(corner_pt);
            const auto body_eff = map.body_with_learned(ret_body);
            const auto rc_ = weigh(rc::lidar_guard_returns(std::vector<Eigen::Vector2f>{corner_pt}, body_eff, kSigma, 0.6f, self_fp), map);
            const float v_corner = rc::lidar_return_speed_limit(rc_.obstacles, kFwd, body_eff, kMargin, kAdec, kVmax);
            std::println("  learned self: corner part P(self) {:.3f} -> {:.3f} after 3 s driving+turning; forward {:.3f}", p0, p1, v_corner);
            CHECK(p0 < 0.05f and p1 > 0.5f and v_corner > 0.95f * kVmax);
            // What matters live: the part's NOISY returns, sweep after sweep — how many sweeps still leave forward free?
            {
                int free_sweeps = 0;
                for (int k = 0; k < 200; ++k)
                {
                    const auto rj = weigh(rc::lidar_guard_returns(std::vector<Eigen::Vector2f>{jitter(corner_pt)}, body_eff, kSigma, 0.6f, self_fp), map);
                    free_sweeps += rc::lidar_return_speed_limit(rj.obstacles, kFwd, body_eff, kMargin, kAdec, kVmax) > 0.95f * kVmax;
                }
                std::println("  the part's noisy returns after learning: forward free on {}/200 sweeps", free_sweeps);
                CHECK(free_sweeps >= 190);
            }
            // Parked, the same point teaches nothing (no motion separates the hypotheses).
            {
                rc::SelfReturnMap still;
                still.configure(kSigma, ret_body, self_fp, kMargin);
                for (int k = 0; k < 60; ++k)
                {
                    const std::vector<Eigen::Vector2f> sweep{jitter(corner_pt)};
                    still.update(sweep, Eigen::Affine2f::Identity(), 0.005f, k * 0.05);
                }
                rc::SelfReturnMap fresh;
                fresh.configure(kSigma, ret_body, self_fp, kMargin);
                CHECK(std::abs(still.logodds(corner_pt) - fresh.logodds(corner_pt)) < 1e-4f);
            }

            // (b) a world-fixed jamb end, approached at 0.1 m/s from 0.5 m ahead until the front touches it.
            rc::SelfReturnMap mj;
            mj.configure(kSigma, ret_body, self_fp, kMargin);
            ControllerRobotPose rp;   // theta 0: forward = room +y
            const Eigen::Vector2f jamb_room(0.1f, 0.25f + 0.5f);
            float p_max = 0.f, v_end = 1.f;
            int stopped = 0;
            for (int k = 0; k < 400 and stopped < 20; ++k)   // the robot OBEYS the cap (the net is in the loop)
            {
                const Eigen::Affine2f T = rp.as_transform();
                const Eigen::Vector2f q = T.inverse() * jamb_room;
                std::vector<Eigen::Vector2f> sweep{jitter(q)};
                const auto r = rc::lidar_guard_returns(sweep, ret_body, kSigma, 0.6f, self_fp);
                std::vector<Eigen::Vector2f> kept;
                for (const auto& o : r.obstacles) kept.push_back(o.mean);
                mj.update(kept, T, 0.005f, k * 0.05);
                p_max = std::max(p_max, mj.p_self(q));
                const auto rq = weigh(rc::lidar_guard_returns(std::vector<Eigen::Vector2f>{q}, ret_body, kSigma, 0.6f, self_fp), mj);
                v_end = rc::lidar_return_speed_limit(rq.obstacles, kFwd, ret_body, kMargin, kAdec, kVmax);
                const float v = std::min(0.1f, v_end);
                if (std::getenv("TLBG_DEBUG") and k % 10 == 0) std::println("    k {} q.y {:.3f} p_self {:.3f} L {:.2f} v {:.3f}", k, q.y(), mj.p_self(q), mj.logodds(q), v_end);
                stopped = v == 0.f ? stopped + 1 : 0;
                rp.pos.y() += v * 0.05f;
            }
            const float gap = (rp.as_transform().inverse() * jamb_room).y() - 0.2472f;
            std::println("  world jamb approached at 0.1 m/s: max P(self) {:.3f}; stopped {:.3f} m short of the front cover, "
                         "forward {:.3f}", p_max, gap, v_end);
            CHECK(p_max < 0.05f and v_end == 0.f and gap > 0.f);

            // (b2) The hard case: a world jamb end approached straight at the LEARNED corner (the map from (a), P(self) ≈ 1
            //      there). It slides through the body frame, the evidence turns, and the robot must still stop short of it.
            {
                rc::SelfReturnMap m2 = map;
                ControllerRobotPose rp2;
                const Eigen::Vector2f target = corner_pt + Eigen::Vector2f(0.f, 0.25f);   // 25 cm ahead of the part
                float p_first = -1.f, v_last = 1.f, min_gap = 1.f, min_gap_part = 1.f;
                int stopped2 = 0;
                for (int k = 0; k < 600 and stopped2 < 20; ++k)
                {
                    const Eigen::Affine2f T = rp2.as_transform();
                    const Eigen::Vector2f q = T.inverse() * target;
                    std::vector<Eigen::Vector2f> sweep{jitter(q), jitter(corner_pt)};   // the robot's own part still returns
                    const auto be = m2.body_with_learned(ret_body);
                    const auto r = rc::lidar_guard_returns(sweep, be, kSigma, 0.6f, self_fp);
                    std::vector<Eigen::Vector2f> kept;
                    for (const auto& o : r.obstacles) kept.push_back(o.mean);
                    m2.update(kept, T, 0.005f, 100.0 + k * 0.05);
                    if (p_first < 0.f and m2.in_domain(q)) p_first = m2.p_self(q);
                    const auto be2 = m2.body_with_learned(ret_body);
                    const auto rq = weigh(rc::lidar_guard_returns(std::vector<Eigen::Vector2f>{q}, be2, kSigma, 0.6f, self_fp), m2);
                    v_last = rq.obstacles.empty() ? kVmax : rc::lidar_return_speed_limit(rq.obstacles, kFwd, be2, kMargin, kAdec, kVmax);
                    const float v = std::min(0.1f, v_last);
                    stopped2 = v == 0.f ? stopped2 + 1 : 0;
                    rp2.pos.y() += v * 0.05f;
                    min_gap = std::min(min_gap, rc::anchor_signed_distance(ret_body, q));
                    min_gap_part = std::min(min_gap_part, (q - corner_pt).norm());
                }
                std::println("  world jamb driven at the LEARNED corner: P(self) on entering the domain {:.3f}; closest {:.3f} m from the "
                             "returns body, {:.3f} m from the part; forward at the end {:.3f}", p_first, min_gap, min_gap_part, v_last);
                CHECK(min_gap > 0.f and min_gap_part > kSigma and v_last == 0.f);
            }
            // (e) Fix round 7 (try83): driving 3 s PARALLEL to a wall 0.375 m to the left (1 cm sampling, range noise) must
            //     learn nothing — try83 learned 22 cells of exactly this wall as the robot. Read through the same handover
            //     the session uses: every return within the domain plus the ring's motion bound (1.2 m here).
            {
                rc::SelfReturnMap mw;
                mw.configure(kSigma, ret_body, self_fp, kMargin);
                ControllerRobotPose wp;   // theta 0: forward = room +y, left = room −x
                std::vector<Eigen::Vector2f> wall;
                for (float y = -3.f; y <= 6.f; y += 0.01f) wall.emplace_back(-0.375f, y);
                const float prior = mw.logodds(Eigen::Vector2f(-0.375f, 0.15f));
                for (int k = 0; k < 60; ++k)
                {
                    const Eigen::Affine2f T = wp.as_transform();
                    std::vector<Eigen::Vector2f> sweep;
                    for (const auto& w : wall)
                        if (const Eigen::Vector2f q = T.inverse() * w; q.norm() < 1.2f) sweep.push_back(jitter(q));
                    const auto r = rc::lidar_guard_returns(sweep, ret_body, kSigma, 1.2f, self_fp);
                    std::vector<Eigen::Vector2f> kept;
                    for (const auto& o : r.obstacles) kept.push_back(o.mean);
                    mw.update(kept, T, 0.005f, k * 0.05);
                    wp.pos.y() += 0.15f * 0.05f;
                }
                float pmax = 0.f, lmax = -1e9f;
                for (float y = -0.10f; y <= 0.30f; y += 0.01f)
                {
                    pmax = std::max(pmax, mw.p_self(Eigen::Vector2f(-0.375f, y)));
                    lmax = std::max(lmax, mw.logodds(Eigen::Vector2f(-0.375f, y)));
                }
                const auto cw = mw.census(self_fp);
                std::println("  driving 3 s beside a wall 0.375 m to the left: max P(self) on the wall {:.4f} (log-odds {:+.2f}, prior "
                             "there {:+.2f}); {} cells learned self", pmax, lmax, prior, cw.self_cells);
                CHECK(pmax < 0.02f and cw.self_cells == 0);
            }
            // (d) A localisation JUMP while parked (7.5 cm in one sweep): the world point beside the corner keeps its body
            //     position, which a jump would read as self — the ring restarts at the jump and nothing is learned.
            {
                rc::SelfReturnMap mjump;
                mjump.configure(kSigma, ret_body, self_fp, kMargin);
                mjump.set_capability(1.f, 2.f);
                const Eigen::Vector2f q(-0.30f, 0.10f);
                Eigen::Affine2f T = Eigen::Affine2f::Identity();
                for (int k = 0; k < 60; ++k)
                {
                    if (k == 20) T.translation() += Eigen::Vector2f(0.075f, 0.f);
                    const std::vector<Eigen::Vector2f> sweep{jitter(q)};
                    mjump.update(sweep, T, 0.005f, k * 0.05);
                }
                std::println("  localisation jump while parked: restarts {}, P(self) {:.3f}", mjump.restarts(), mjump.p_self(q));
                CHECK(mjump.restarts() == 1 and mjump.p_self(q) < 0.05f);
            }

            // (c) persistence round-trip, classic locale, from_chars.
            const std::string path = "/tmp/self_returns_test.csv";
            CHECK(map.save(path));
            rc::SelfReturnMap back;
            back.configure(kSigma, ret_body, self_fp, kMargin);
            CHECK(back.load(path) > 0 and std::abs(back.p_self(corner_pt) - p1) < 1e-3f);
            const auto cen = map.census(self_fp);
            std::println("  census: {} cells, {} self, {} learned beyond the outline (top at ({:+.3f},{:+.3f}))", cen.cells, cen.self_cells,
                         cen.learned_self, cen.top.empty() ? 0.f : cen.top[0].x(), cen.top.empty() ? 0.f : cen.top[0].y());
            CHECK(cen.learned_self >= 1);
        }
    }

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
