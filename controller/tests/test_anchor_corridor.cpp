// try101 leg 1 (2026-10-19): the anchor cross in room_1's CORRIDOR STUB below door_1. The robot entered the cross 55° off
// the door normal, 2.13 m before the wall, 0.21 m west of the opening's centre line, and never crossed: the servo turned it
// 138° to drive SIDEWAYS across the corridor, the turn back toward the door was then refused by the live returns of the
// corridor wall it now faced, and the guard's backoff and the servo chattered (CW backoff / CCW servo) until the cross
// contract timed out — three times (cross_unreachable x3).
//
// Closed-loop, kinematic, on the EXACT Webots DOOR_1 scene (door frame of door_concept/tests/test_door_phi_sides.cpp: x along
// the wall, the opening's centre at x = 0, room_1 at y < 0; room_1's wall face y = -0.24, the leaf open 90° INTO room_1 on
// the west jamb, the corridor's east wall at x ≈ 0.52, its west side 1.1-1.6 m out). Every cycle runs the session's chain
// (ControllerSession::step_anchor_cross): rc::anchor_cross_command → rc::anchor_jamb_guard → the live returns' caps
// (rc::lidar_cap_command, self-filtered by the returns body) → the returns' backoff when nothing is left — fed with
// synthetic returns of the scene's faces in the robot frame. The anchor is driven on the line the servo SAW in try101: the
// measured gap read the opening's centre 8 cm east of the truth.
//
// Checked: from the try101 entry pose AND from the pose the three retries started from (pinned 0.37 m from the corridor's
// east wall facing it at 15°), and in a plain 1.0 m corridor 2.6 m before a 0.9 m gap at −33°, the body goes through
// (s ≥ standoff on the anchor) with no scene point ever inside it, and the servo never turns the robot to face along the
// wall (|heading error| < 80° once it has started turning toward the door).
//
// BEFORE the fix (measured, 2026-10-19): build with -DANCHOR_BEFORE_FIX against 4e5115d8's controller_anchor_servo.h and
// lidar_body_guard.h — 3 failures: the entry pose turns to |ψ| 88° (sideways across the corridor, the 90° manoeuvre), and
// the retry pose never gets through (60 s at (+0.17, −1.96) heading 24-32°: the try101 deadlock, reproduced). After: 0.
// The plain 1 m corridor at −33° goes through before too on this ideal unicycle (no rate limit, no lag): it is the guard.
//
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_anchor_corridor.cpp ../common/robot_footprint/*.cpp
//     -o /tmp/tac -lfastdds -lfastcdr -lQt5Core && /tmp/tac          (one line; TRACE=1 prints every cycle)
#include "../src/controller_anchor_servo.h"
#include "../src/lidar_body_guard.h"
#include "../../common/robot_footprint/robot_footprint.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <numbers>
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

using V2 = Eigen::Vector2f;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kMargin = 0.05f, kSigma = 0.03f, kAdec = 1.0f, kVmax = 0.35f, kWmax = 1.0f, kStandoff = 2.0f;
constexpr float kDt = 0.1f;

// The scene's faces, sampled every 1 cm (door frame). Boxes as in test_door_phi_sides.cpp (heights dropped: everything here
// reaches the LiDAR's body band).
std::vector<V2> scene_points()
{
    std::vector<V2> pts;
    auto seg = [&](V2 a, V2 b)
    {
        const int n = std::max(1, int(std::ceil((b - a).norm() / 0.01f)));
        for (int i = 0; i <= n; ++i) pts.push_back(a + (b - a) * (float(i) / float(n)));
    };
    auto box = [&](float x0, float x1, float y0, float y1)
    { seg({x0, y0}, {x1, y0}); seg({x1, y0}, {x1, y1}); seg({x1, y1}, {x0, y1}); seg({x0, y1}, {x0, y0}); };
    box(-3.06f, -0.53f, -0.11f, 0.f); box(0.53f, 3.06f, -0.11f, 0.f);        // room_2's south wall
    box(-1.21f, -0.501f, -0.24f, -0.13f); box(0.499f, 0.67f, -0.24f, -0.13f); // room_1's wall, its 1.0 m opening
    box(-0.5f, -0.45f, -0.20f, 0.f); box(0.45f, 0.5f, -0.20f, 0.f);           // the casings
    box(-0.525f, -0.475f, -1.15f, -0.20f);                                    // the leaf, open 90° into room_1 (west jamb)
    {                                                                         // the corridor's east wall, 0.11 m thick
        const V2 a(0.574f, -0.25f), e(0.474f, -4.46f), o(0.11f, 0.f);
        seg(a, e); seg(a + o, e + o); seg(a, a + o); seg(e, e + o);
    }
    box(-1.22f, -1.11f, -1.47f, -0.24f); box(-1.67f, -1.11f, -1.58f, -1.47f);  // the west side and its jog
    box(-1.67f, -1.56f, -2.31f, -1.47f); box(-6.f, -1.15f, -2.42f, -2.31f);
    box(-3.17f, -3.06f, 0.f, 4.11f); box(3.06f, 3.17f, 0.f, 4.11f); box(-3.2f, 3.2f, 4.11f, 4.22f);   // room_2's other walls
    return pts;
}

// A plain corridor 1.0 m wide (walls x = ±0.5, 0.1 m thick, 4 m long) ending in room_1's wall with a 0.9 m gap (|x| < 0.45).
std::vector<V2> corridor_points()
{
    std::vector<V2> pts;
    auto seg = [&](V2 a, V2 b)
    {
        const int n = std::max(1, int(std::ceil((b - a).norm() / 0.01f)));
        for (int i = 0; i <= n; ++i) pts.push_back(a + (b - a) * (float(i) / float(n)));
    };
    seg({-0.5f, -0.24f}, {-0.5f, -4.24f}); seg({0.5f, -0.24f}, {0.5f, -4.24f});
    seg({-0.6f, -0.24f}, {-0.6f, -4.24f}); seg({0.6f, -0.24f}, {0.6f, -4.24f});
    seg({-2.f, -0.24f}, {-0.45f, -0.24f}); seg({0.45f, -0.24f}, {2.f, -0.24f});
    seg({-2.f, -0.13f}, {-0.45f, -0.13f}); seg({0.45f, -0.13f}, {2.f, -0.13f});
    seg({-0.45f, -0.24f}, {-0.45f, -0.13f}); seg({0.45f, -0.24f}, {0.45f, -0.13f});
    return pts;
}

struct Pose { V2 p; float th; };   // door frame; th = the FORWARD direction's angle from +x

V2 to_robot(const Pose& r, const V2& q)
{
    const V2 d = q - r.p, fwd(std::cos(r.th), std::sin(r.th)), right(std::sin(r.th), -std::cos(r.th));
    return {d.dot(right), d.dot(fwd)};
}

// The anchor door_concept hangs from the robot: the opening's centre (x = centre_x, on room_1's wall face) and the wall
// direction (+x) in the robot frame, with the covariance try101 logged (σ_lat 0.025 m).
rc::interaction::RelPose anchor_of(const Pose& r, float centre_x)
{
    rc::interaction::RelPose a;
    const V2 A = to_robot(r, V2(centre_x, -0.24f));
    a.x = {A.x(), A.y(), rc::interaction::wrap_pi(kPi / 2.f - r.th)};
    a.P = Eigen::Matrix3f::Zero();
    a.P(0, 0) = a.P(1, 1) = 0.025f * 0.025f;
    a.P(2, 2) = 0.02f * 0.02f;
    return a;
}

struct Outcome { bool through = false; int contacts = 0; float t = 0.f; Pose end; float min_sd = 1e9f, max_psi = 0.f; };

Outcome run(const std::vector<V2>& scene, Pose r, float line_x, float width, float t_max, bool trace)
{
    const auto fp = rc::RobotFootprint::shadow();
    const auto& hull = fp.polygon();
    // The returns body (lidar_body_guard.h fix round 5): the planning hull ∪ the display mesh's front (y 0.2472, |x| 0.242).
    std::vector<V2> mesh{{0.242f, 0.2472f}, {-0.242f, 0.2472f}, {0.242f, -0.20f}, {-0.242f, -0.20f}};
    const auto body = rc::lidar_returns_body(hull, mesh);
    const auto self = rc::lidar_self_footprint(hull, mesh, kSigma);
    float half_w = 0.f, front = 0.f;
    for (const auto& v : hull) { half_w = std::max(half_w, std::abs(v.x())); front = std::max(front, v.y()); }
    half_w += kMargin;   // approach_body_.support_radius(0, ±x / +y): the safety margin included
    front += kMargin;
    float r_body = 0.f;
    for (const auto& v : rc::anchor_grown_body(body, kMargin)) r_body = std::max(r_body, v.norm());
    const float reach = r_body + kVmax * kVmax / (2.f * kAdec);
    Outcome o;
    for (int k = 0; k * kDt < t_max; ++k)
    {
        o.t = float(k) * kDt;
        std::vector<V2> robot_pts;
        robot_pts.reserve(scene.size());
        for (const auto& q : scene)
        {
            const V2 qr = to_robot(r, q);
            if (rc::anchor_point_in_convex(body, qr)) ++o.contacts;
            o.min_sd = std::min(o.min_sd, rc::anchor_signed_distance(body, qr));
            robot_pts.push_back(qr);
        }
        const auto returns = rc::lidar_guard_returns(robot_pts, body, kSigma, reach, self);
        const auto pose = anchor_of(r, line_x);
#ifdef ANCHOR_BEFORE_FIX
        const auto c = rc::anchor_cross_command(pose, width, half_w, kStandoff, kVmax, kWmax, /*holonomic=*/false);
        if (c.through) { o.through = true; break; }
        struct { rc::AnchorCommand cmd; bool applied = false; } room{c};
#else
        // The advance the body, aligned with the normal, has before its path meets a return (rc::lidar_free_advance).
        const float free = rc::lidar_free_advance(robot_pts, rc::anchor_cross_geometry(pose).n, body);
        const auto c = rc::anchor_cross_command(pose, width, half_w, kStandoff, kVmax, kWmax, /*holonomic=*/false, {}, front,
                                                free);
        if (c.through) { o.through = true; break; }
        // The servo's turn given its room (rc::anchor_make_turn_room) against the anchor's own obstacles AND the live
        // returns, as ControllerSession::step_anchor_cross does before the guards.
        const auto model = rc::anchor_obstacles(pose, width, {});
        const std::array<rc::AnchorTurnBlockers, 2> sets{rc::AnchorTurnBlockers{model, hull},
                                                         rc::AnchorTurnBlockers{returns.obstacles, body}};
        const auto room = rc::anchor_make_turn_room(c, sets, kMargin, kAdec, kVmax, kWmax, false);
#endif
        if (o.t >= 1.f) o.max_psi = std::max(o.max_psi, std::abs(rc::anchor_cross_geometry(pose).heading_err));
        const auto guard = rc::anchor_jamb_guard(pose, width, room.cmd, false, hull, kMargin, kAdec, kVmax, kWmax);
        float adv = guard.cmd.adv, side = 0.f, rot = guard.cmd.rot;
        const bool had = adv != 0.f or rot != 0.f;
        const auto cap = rc::lidar_cap_command(returns.obstacles, body, kMargin, kAdec, kVmax, kWmax, adv, side, rot);
        bool backoff = false;
        if (had and adv == 0.f and rot == 0.f)
            if (const auto b = rc::lidar_return_backoff(returns.obstacles, body, kMargin, kSigma, kAdec, kVmax, kWmax, false);
                b.has_value())
            { adv = b->adv; rot = b->rot; backoff = true; }
        if (trace)
        {
            const auto g = rc::anchor_cross_geometry(pose);
            std::println("    t {:5.1f} p ({:+.3f},{:+.3f}) hd {:+6.1f} | e {:+.3f} s {:+.3f} psi {:+6.1f} | servo adv {:+.3f} rot {:+.3f}"
                         " | out adv {:+.3f} rot {:+.3f}{}{}{}{} | sd {:+.3f}", o.t, r.p.x(), r.p.y(), r.th * 180.f / kPi, g.e, g.s,
                         g.heading_err * 180.f / kPi, c.adv, c.rot, adv, rot, cap.bound_lin ? " LIN" : "",
                         cap.bound_rot ? " ROT" : "", backoff ? " BACKOFF" : "", room.applied ? " ROOM" : "", o.min_sd);
        }
        r.p += adv * kDt * V2(std::cos(r.th), std::sin(r.th));
        r.th += rot * kDt;
    }
    o.end = r;
    return o;
}

void report(std::string_view what, const Outcome& o)
{
    std::println("  {}: {} after {:.1f} s, end ({:+.3f},{:+.3f}) heading {:+.1f} deg, contacts {}, min clearance {:+.3f} m, "
                 "max |heading err| after 1 s {:.1f} deg", what, o.through ? "THROUGH" : "NOT through", o.t, o.end.p.x(),
                 o.end.p.y(), o.end.th * 180.f / kPi, o.contacts, o.min_sd, o.max_psi * 180.f / kPi);
}
}  // namespace

int main()
{
    const bool trace = std::getenv("TRACE") != nullptr;
    const auto door1 = scene_points(), corridor = corridor_points();
    const float width = 0.90f;   // DOOR_1's clear gap: the open leaf's face (x −0.475) to the east casing (x +0.45)
    const float kTurnedAlong = 80.f * kPi / 180.f;   // "faces along the wall": the 90° manoeuvre, with 10° of loop slack
    std::println("try101 leg 1 — the anchor cross from room_1's corridor stub (door frame; normal = +y, heading 90 deg)");
    // (a) The try101 entry pose: Webots ground truth at the cross's first cycle, door frame (−0.214, −2.128), heading 145°.
    //     The line the servo drove on: the measured gap, its centre read 8 cm east of the truth; and the true line.
    for (const float line : {0.08f, 0.f})
    {
        const auto o = run(door1, Pose{V2(-0.214f, -2.128f), 145.f * kPi / 180.f}, line, width, 60.f, trace);
        report(std::format("entry pose, line at x {:+.2f}", line), o);
        CHECK(o.through);
        CHECK(o.contacts == 0);
        CHECK(o.max_psi < kTurnedAlong);
    }
    // (b) The pose the three retries started from: pinned 0.37 m from the east wall, facing it at 15°. First, what the live
    //     guard sees there (try101: "~45 returns near the body, 0 within sigma_r outside the hull, 0 filtered as the body"):
    //     the corridor's east wall alone, inside the margin band, outside the contact band — a real wall, not the robot.
    {
        const auto fp = rc::RobotFootprint::shadow();
        std::vector<V2> mesh{{0.242f, 0.2472f}, {-0.242f, 0.2472f}, {0.242f, -0.20f}, {-0.242f, -0.20f}};
        const auto body = rc::lidar_returns_body(fp.polygon(), mesh);
        const Pose r{V2(0.161f, -1.963f), 15.f * kPi / 180.f};
        std::vector<V2> pts;
        for (const auto& q : door1) pts.push_back(to_robot(r, q));
        float r_body = 0.f;
        for (const auto& v : rc::anchor_grown_body(body, kMargin)) r_body = std::max(r_body, v.norm());
        const auto ret = rc::lidar_guard_returns(pts, body, kSigma, r_body + kVmax * kVmax / (2.f * kAdec),
                                                 rc::lidar_self_footprint(fp.polygon(), mesh, kSigma));
        const auto c = rc::lidar_guard_census(ret, body, kMargin, kAdec, kVmax, kWmax);
        float x_min = 1e9f, x_max = -1e9f;
        for (const auto& j : ret.obstacles) { const V2 w = r.p + j.mean.x() * V2(std::sin(r.th), -std::cos(r.th)) + j.mean.y() * V2(std::cos(r.th), std::sin(r.th)); x_min = std::min(x_min, w.x()); x_max = std::max(x_max, w.x()); }
        std::println("  retry pose census: {} returns kept within reach (door-frame x {:+.3f}..{:+.3f}: the east wall), {} in the "
                     "contact band, {} self; nearest {:.3f} m from the hull at {:+.0f} deg; caps fwd {:.3f} back {:.3f} m/s, "
                     "CCW {:.3f} CW {:.3f} rad/s; the grown body's reach {:.3f} m", c.n_kept, x_min, x_max, c.n_band, c.n_self,
                     c.nearest_sd, c.nearest_bearing_deg, c.v_fwd, c.v_back, c.w_ccw, c.w_cw, r_body);
        CHECK(c.n_band == 0 and c.n_self == 0 and c.n_kept > 0);
        CHECK(x_min > 0.47f);                 // every kept return is the corridor's east wall (inner face x 0.474..0.574)
        CHECK(c.nearest_sd < kMargin + kSigma);   // ...inside the margin band: the turn toward the door is refused
    }
    for (const float line : {0.08f, 0.f})
    {
        const auto o = run(door1, Pose{V2(0.161f, -1.963f), 15.f * kPi / 180.f}, line, width, 60.f, trace);
        report(std::format("retry pose, line at x {:+.2f}", line), o);
        CHECK(o.through);
        CHECK(o.contacts == 0);
    }
    // (c) A plain 1.0 m corridor, 2.6 m before the 0.9 m gap at −33° (try101's anchor reading), centred and off-centre, on
    //     the true line and on one read 8 cm off.
    for (const float x0 : {0.f, -0.1f, 0.1f})
        for (const float line : {0.f, 0.08f})
        {
            const auto o = run(corridor, Pose{V2(x0, -2.84f), (90.f + 33.f) * kPi / 180.f}, line, width, 60.f, trace);
            report(std::format("1 m corridor, 2.6 m out at -33 deg, x {:+.2f}, line {:+.2f}", x0, line), o);
            CHECK(o.through);
            CHECK(o.contacts == 0);
            CHECK(o.max_psi < kTurnedAlong);
        }
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
