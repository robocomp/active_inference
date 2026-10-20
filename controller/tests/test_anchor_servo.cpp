// anchor_cross_command (controller_anchor_servo.h): crossing a door by servoing on its robot-frame anchor.
// Build+run (pure Eigen + std, but relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_anchor_servo.cpp -o /tmp/tas
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tas          (one line)
// (run from controller/)
#include "../src/controller_anchor_servo.h"
#include "../src/stall_judge.h"

#include <cmath>
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
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using rc::interaction::RelPose;

constexpr float kWidth = 0.90f, kBody = 0.2938f, kStandoff = 2.0f, kVmax = 0.35f, kWmax = 1.0f;

// An anchor at (x, y, yaw) in the robot frame, isotropic position sigma, small yaw sigma.
RelPose anchor(float x, float y, float yaw, float sigma_xy)
{
    RelPose r;
    r.x = {x, y, yaw};
    r.P = Eigen::Matrix3f::Zero();
    r.P(0, 0) = r.P(1, 1) = sigma_xy * sigma_xy;
    r.P(2, 2) = 0.02f * 0.02f;
    return r;
}

rc::AnchorCommand cmd(const RelPose& r, bool holonomic = true)
{ return rc::anchor_cross_command(r, kWidth, kBody, kStandoff, kVmax, kWmax, holonomic); }

// Rectangular bodies, CCW, robot frame (x right, y forward).
std::vector<Eigen::Vector2f> rect(float hw, float hl) { return {{hw, -hl}, {hw, hl}, {-hw, hl}, {-hw, -hl}}; }
const auto kLong = rect(kBody, 0.35f);       // the long test body of the original jamb test
const auto kShadow = rect(kBody, 0.23f);     // Shadow-like: 0.59 m wide, 0.46 m long
bool touches(const std::vector<Eigen::Vector2f>& body, const Eigen::Vector2f& q)
{ return rc::anchor_point_in_convex(body, q); }

// Drive the command on a kinematic base for n steps of dt; returns the e trace and the s trace.
struct Trace { std::vector<float> e, s, psi; };
Trace drive(RelPose r, bool holonomic, int n, float dt)
{
    Trace t;
    for (int i = 0; i < n; ++i)
    {
        const auto g = rc::anchor_cross_geometry(r);
        t.e.push_back(g.e); t.s.push_back(g.s); t.psi.push_back(g.heading_err);
        const auto c = cmd(r, holonomic);
        if (not holonomic) CHECK(c.side == 0.f);
        r.x = rc::interaction::move_mean(r.x, c.side * dt, c.adv * dt, c.rot * dt);
    }
    return t;
}

void centred_goes_straight()
{
    const auto c = cmd(anchor(0.f, 2.f, 0.f, 0.03f));            // 2 m ahead, on the robot's forward axis
    CHECK(std::abs(c.side) < 1e-3f);
    CHECK(std::abs(c.rot) < 1e-3f);
    CHECK(c.adv > 0.8f * kVmax);
    CHECK(c.adv <= kVmax + 1e-5f);
    CHECK(not c.through);
}

void offset_slides_sideways()
{
    const auto centred = cmd(anchor(0.f, 2.f, 0.f, 0.03f));
    // Robot 0.3 m to the anchor's +x of the opening's normal line ⇒ the opening is at robot x = −0.3.
    const auto g = rc::anchor_cross_geometry(anchor(-0.3f, 2.f, 0.f, 0.03f));
    CHECK(std::abs(g.e - 0.3f) < 1e-4f);                        // e = robot's offset along the wall
    CHECK(std::abs(g.s + 2.f) < 1e-4f);                         // 2 m before the opening
    const auto c = cmd(anchor(-0.3f, 2.f, 0.f, 0.03f));
    CHECK(c.side < 0.f);                                        // toward the line (robot −x)
    CHECK(c.adv < centred.adv);                                 // 0.3 m off a 0.9 m door does not fit yet
    CHECK(std::abs(c.rot) < 1e-3f);
    // Mirror image: offset the other way ⇒ slide the other way, same speed.
    const auto m = cmd(anchor(0.3f, 2.f, 0.f, 0.03f));
    CHECK(m.side > 0.f);
    CHECK(std::abs(m.adv - c.adv) < 1e-5f);
}

void rotated_anchor_turns_and_slides()
{
    // The door normal points 30° to the robot's left (ψ = +30°, n = (−sin ψ, cos ψ)). The robot sits on that normal
    // line (opening at 2 m along n): e = 0, so the translation goes along n — including a side component — and the
    // robot turns CCW (positive rot) toward it.
    const float psi = std::numbers::pi_v<float> / 6.f;
    const Eigen::Vector2f n(-std::sin(psi), std::cos(psi));
    const Eigen::Vector2f A = 2.f * n;
    const auto g = rc::anchor_cross_geometry(anchor(A.x(), A.y(), psi, 0.03f));
    CHECK(std::abs(g.e) < 1e-4f);
    CHECK(std::abs(g.s + 2.f) < 1e-4f);
    CHECK(std::abs(g.heading_err - psi) < 1e-4f);
    const auto c = cmd(anchor(A.x(), A.y(), psi, 0.03f));
    CHECK(c.rot > 0.f);
    CHECK(c.rot <= kWmax + 1e-5f);
    // translation parallel to n
    CHECK(std::abs(c.side * n.y() - c.adv * n.x()) < 1e-4f);
    CHECK(c.side < 0.f and c.adv > 0.f);
}

void uncertain_slows()
{
    // Centred, with a body that leaves 0.15 m slack per side: a 0.3 m along-wall sigma has to slow it.
    const auto sure = cmd(anchor(0.f, 2.f, 0.f, 0.03f));
    const auto unsure = cmd(anchor(0.f, 2.f, 0.f, 0.3f));
    CHECK(unsure.adv < sure.adv);
    CHECK(unsure.adv > 0.f);
    // Only the ALONG-WALL variance matters: a large normal-only variance leaves the speed alone.
    RelPose r = anchor(0.f, 2.f, 0.f, 0.03f);
    r.P(1, 1) = 0.3f * 0.3f;
    CHECK(std::abs(cmd(r).adv - sure.adv) < 1e-4f);
}

void through_after_standoff()
{
    const auto c = cmd(anchor(0.f, -2.1f, 0.f, 0.03f));           // the opening 2.1 m BEHIND
    CHECK(c.through);
    CHECK(c.adv == 0.f and c.side == 0.f and c.rot == 0.f);
    const auto before = cmd(anchor(0.f, -1.9f, 0.f, 0.03f));      // 1.9 m behind: not yet
    CHECK(not before.through);
    CHECK(before.adv > 0.f);
}

void degenerate_inputs_do_not_move()
{
    // No width (absent attribute) ⇒ the body cannot be shown to fit ⇒ no advance.
    const auto c = rc::anchor_cross_command(anchor(0.f, 2.f, 0.f, 0.03f), 0.f, kBody, kStandoff, kVmax, kWmax, false);
    CHECK(c.adv < 1e-3f);
    // A non-finite pose ⇒ zero command.
    const auto z = cmd(anchor(std::nanf(""), 2.f, 0.f, 0.03f));
    CHECK(z.adv == 0.f and z.side == 0.f and z.rot == 0.f and not z.through);
}

void differential_converges_laterally()
{
    // e = 0.3 m, aligned with the normal, 2 m before the opening. side is never used.
    const auto t = drive(anchor(-0.3f, 2.f, 0.f, 0.03f), /*holonomic=*/false, 40, 0.1f);
    // Monotone after the first few steps (the turn toward the line), to within 3 mm: the heading arrives a little
    // late on a rate-limited base, so e may cross zero by a couple of millimetres before settling.
    bool monotone = true;
    for (std::size_t i = 4; i + 1 < t.e.size(); ++i)
        if (std::abs(t.e[i + 1]) > std::abs(t.e[i]) + 1e-4f and std::abs(t.e[i + 1]) > 0.003f) monotone = false;
    CHECK(std::abs(t.e.back()) < 0.05f);
    CHECK(monotone);
    std::size_t k = 0;                                   // first step with |e| < 0.05
    while (k < t.e.size() and std::abs(t.e[k]) >= 0.05f) ++k;
    CHECK(k < t.e.size());
    CHECK(k < t.e.size() and t.s[k] < 0.f);              // centred before reaching the opening
    std::println("  differential: |e| < 0.05 at step {} (s = {:+.2f}), final e {:+.3f}, psi {:+.3f}",
                 k, k < t.s.size() ? t.s[k] : 0.f, t.e.back(), t.psi.back());
    // With side dropped, a first command on a pure lateral offset must TURN toward the line, not ask for side.
    const auto c = cmd(anchor(-0.3f, 2.f, 0.f, 0.03f), false);
    CHECK(c.side == 0.f);
    CHECK(c.rot > 0.f);                                  // the line is to the robot's left (−x): CCW
}

void combined_offset_and_heading()
{
    // e = 0.25, ψ = +0.3 rad. Opening placed so that robot origin in the anchor frame is (0.25, −2.2).
    const float psi = 0.3f;
    const Eigen::Vector2f a(std::cos(psi), std::sin(psi)), n(-std::sin(psi), std::cos(psi));
    const Eigen::Vector2f A = -(0.25f * a + -2.2f * n);
    const auto r = anchor(A.x(), A.y(), psi, 0.03f);
    const auto g = rc::anchor_cross_geometry(r);
    CHECK(std::abs(g.e - 0.25f) < 1e-4f and std::abs(g.s + 2.2f) < 1e-4f);
    // Omni: the translation reduces e (its component along a is negative) and the rotation is toward ψ.
    const auto o = cmd(r, true);
    CHECK(Eigen::Vector2f(o.side, o.adv).dot(g.a) < 0.f);
    CHECK(o.rot > 0.f);
    // Both laws centre and align on the kinematic base.
    for (const bool holo : {true, false})
    {
        const auto t = drive(r, holo, 60, 0.1f);
        CHECK(std::abs(t.e.back()) < 0.05f);
        CHECK(std::abs(t.psi.back()) < 0.1f);
        std::println("  combined ({}): final e {:+.3f}, psi {:+.3f}, s {:+.2f}", holo ? "omni" : "diff",
                     t.e.back(), t.psi.back(), t.s.back());
    }
}

void jamb_stops_the_body()
{
    const Eigen::Vector2f fwd(0.f, 1.f);
    // ψ = 0, the jamb J+ = A + (w/2)·a placed 0.30 m straight ahead of the body centre.
    const auto r = anchor(-0.5f * kWidth, 0.30f, 0.f, 0.03f);
    const float v = rc::anchor_jamb_speed_limit(r, kWidth, fwd, kLong, 0.f, 1.0f, kVmax);
    const auto c = cmd(r, false);
    CHECK(std::min(c.adv, v) <= 0.01f);
    CHECK(v <= 0.01f);
    // Centred in a 0.9 m opening 1 m ahead: the jambs are beside the corridor, nothing limited.
    CHECK(rc::anchor_jamb_speed_limit(anchor(0.f, 1.f, 0.f, 0.03f), kWidth, fwd, kLong, 0.f, 1.0f, kVmax)
          > kVmax - 1e-3f);
    // The same jamb 1 m ahead in the corridor: a braking-distance ceiling, here above the cap.
    const float v3 = rc::anchor_jamb_speed_limit(anchor(-0.5f * kWidth, 1.0f, 0.f, 0.03f), kWidth, fwd, kLong, 0.f,
                                                 1.0f, kVmax);
    CHECK(v3 > 0.f and v3 <= kVmax);
    // A jamb 5 cm beside the corridor, 5 cm beyond the face: a sure opening leaves it alone, an unsure one slows.
    const auto beside = anchor(-0.5f * kWidth + kBody + 0.05f, 0.40f, 0.f, 0.01f);
    CHECK(rc::anchor_jamb_speed_limit(beside, kWidth, fwd, kLong, 0.f, 1.0f, kVmax) > kVmax - 1e-3f);
    const auto beside_unsure = anchor(-0.5f * kWidth + kBody + 0.05f, 0.40f, 0.f, 0.10f);
    CHECK(rc::anchor_jamb_speed_limit(beside_unsure, kWidth, fwd, kLong, 0.f, 1.0f, kVmax) < kVmax);
    // Moving AWAY (backwards) from the jamb dead ahead: not closing, not limited.
    CHECK(rc::anchor_jamb_speed_limit(r, kWidth, Eigen::Vector2f(0.f, -1.f), kLong, 0.f, 1.0f, kVmax) > kVmax - 1e-3f);
}

// The session's guard on one servo command: translation along the base's motion axis and rotation, both capped.
rc::AnchorCommand guarded(const RelPose& r, const std::vector<Eigen::Vector2f>& body, float a_dec)
{
    auto c = cmd(r, false);
    if (c.adv > 0.f)
        c.adv = std::min(c.adv, rc::anchor_jamb_speed_limit(r, kWidth, Eigen::Vector2f(0.f, 1.f), body, 0.f, a_dec, kVmax));
    if (c.rot != 0.f)
        c.rot = std::copysign(std::min(std::abs(c.rot),
                                       rc::anchor_jamb_rot_limit(r, kWidth, c.rot, body, 0.f, a_dec, kWmax)), c.rot);
    return c;
}

void jamb_does_not_lock_recentering()
{
    // Differential robot at the mouth (s = −0.4), e = 0.10, σ_lat = 0.05, a 0.9 m door, a Shadow-like body.
    RelPose r = anchor(-0.10f, 0.40f, 0.f, 0.05f);
    const float e0 = rc::anchor_cross_geometry(r).e;
    int moving = 0, contacts = 0;
    const int n = 30;
    for (int i = 0; i < n; ++i)
    {
        const auto c = guarded(r, kShadow, 1.0f);
        if (c.adv > 1e-3f) ++moving;
        // Sub-step the motion and check the TRUE jamb points against the body box on the way.
        for (int k = 0; k < 10; ++k)
        {
            r.x = rc::interaction::move_mean(r.x, 0.f, c.adv * 0.01f, c.rot * 0.01f);
            for (const auto& j : rc::anchor_jambs(r, kWidth))
                if (touches(kShadow, j.mean)) ++contacts;
        }
    }
    const auto g = rc::anchor_cross_geometry(r);
    std::println("  recentering at the mouth: moving {}/{} steps, e {:+.3f} -> {:+.3f}, s {:+.2f}, contacts {}",
                 moving, n, e0, g.e, g.s, contacts);
    CHECK(moving >= 2 * n / 3);
    CHECK(std::abs(g.e) < std::abs(e0));
    CHECK(contacts == 0);
    CHECK(g.s > 0.f);                                    // it went through
}

void rotation_near_jamb_is_capped()
{
    // Body centred in the opening (s = 0), heading off the normal by 0.5 rad, a body whose corners reach PAST the
    // half-opening (0.32 x 0.33: corner radius 0.46 m against 0.45 m), braking 0.5 m/s².
    const auto body = rect(0.32f, 0.33f);
    const float a_dec = 0.5f;
    const RelPose r0 = anchor(0.f, 0.f, 0.5f, 0.03f);
    // (a) The servo's own turn (toward the normal, CCW) OPENS clearance: the corner nearest a jamb moves away from it,
    //     so it is not capped — and no corner meets a true jamb over 0.5 s.
    const auto c = guarded(r0, body, a_dec);
    CHECK(c.rot > 0.f);
    std::println("  doorway turn toward the normal: rot {:+.3f} (w_max {:.2f})", c.rot, kWmax);
    // (b) The same pose turning the OTHER way closes on a jamb (the front-right corner is 0.28 rad short of J+):
    //     capped below w_max, and closed-loop the corner never crosses the jamb.
    const float w_cw = rc::anchor_jamb_rot_limit(r0, kWidth, -1.f, body, 0.f, a_dec, kWmax);
    CHECK(w_cw < kWmax);
    std::println("  doorway turn toward the jamb: |rot| cap {:.3f}", w_cw);
    for (const float dir : {+1.f, -1.f})
    {
        RelPose r = r0;
        int contacts = 0;
        for (int k = 0; k < 50; ++k)                        // 0.5 s at dt = 0.01, the cap re-evaluated each step
        {
            const float w = rc::anchor_jamb_rot_limit(r, kWidth, dir, body, 0.f, a_dec, kWmax);
            r.x = rc::interaction::move_mean(r.x, 0.f, 0.f, dir * w * 0.01f);
            for (const auto& j : rc::anchor_jambs(r, kWidth))
                if (touches(body, j.mean)) ++contacts;
        }
        CHECK(contacts == 0);
    }
}
// Task 9, fix C (try66 leg 2): a jamb inside the GROWN body (the safety margin breached, the body not touching), ahead-left:
// the RIGHT jamb of an opening that lies to the robot's left. The servo wants to go on — forward, turning LEFT toward the
// opening — and both close on that jamb: forward has no floor, and the old rotation cap gave Δ = 0 in BOTH directions,
// so the old chain commanded nothing, a deadlock the stall judge called a wedge. The guard must instead move the body so
// that the min clearance grows, without ever touching a jamb.
void jamb_deadlock_backs_off()
{
    const float margin = 0.05f, a_dec = 1.0f, dt = 0.01f;
    const auto grown = rc::anchor_grown_body(kShadow, margin);
    // J+ at (−0.31, 0.20): 1.6 cm outside the body's left side, inside its margin, 0.20 m ahead of the centre.
    RelPose r = anchor(-0.31f - 0.5f * kWidth, 0.20f, 0.f, 0.02f);
    auto clearance = [&](const RelPose& q) { return rc::anchor_min_clearance(q, kWidth, grown, 0.f, 0.f, 0.f); };
    const float c0 = clearance(r);
    bool touching = false;
    for (const auto& j : rc::anchor_jambs(r, kWidth)) touching = touching or touches(kShadow, j.mean);
    CHECK(c0 < 0.f and not touching);
    // The OLD chain on the servo's command: forward capped by the jamb ceiling, the turn by Δ = 0 — a deadlock.
    {
        auto c = cmd(r, false);
        const float v = rc::anchor_jamb_speed_limit(r, kWidth, Eigen::Vector2f(0.f, 1.f), kShadow, margin, a_dec, kVmax);
        std::println("  deadlock pose: servo asks adv {:+.3f} rot {:+.3f}; forward ceiling {:.4f}; min clearance {:+.3f} m",
                     c.adv, c.rot, v, c0);
        const float w_ccw = rc::anchor_jamb_rot_limit(r, kWidth, c.rot, kShadow, margin, a_dec, kWmax);
        std::println("  ...and its turn ({:+.3f}) is capped to {:.4f}", c.rot, w_ccw);
        CHECK(c.adv >= 0.f and v < 1e-3f);      // adv ≥ 0 and forward closes on the jamb: no translation
        CHECK(c.rot > 0.f and w_ccw < 1e-3f);   // the servo turns LEFT, into the jamb: capped to 0 — the deadlock
    }
    const auto first = rc::anchor_jamb_guard(r, kWidth, cmd(r, false), false, kShadow, margin, a_dec, kVmax, kWmax);
    std::println("  guard: backoff {} adv {:+.3f} side {:+.3f} rot {:+.3f}", first.backoff, first.cmd.adv, first.cmd.side, first.cmd.rot);
    CHECK(first.backoff);
    CHECK(first.cmd.side == 0.f);
    CHECK(std::abs(first.cmd.adv) > 1e-3f or std::abs(first.cmd.rot) > 1e-3f);
    // Closed loop for 0.5 s: the guard re-evaluated every step on a kinematic differential base, contacts checked.
    int contacts = 0;
    float c_min = c0;
    for (int k = 0; k < 50; ++k)
    {
        const auto g = rc::anchor_jamb_guard(r, kWidth, cmd(r, false), false, kShadow, margin, a_dec, kVmax, kWmax);
        r.x = rc::interaction::move_mean(r.x, 0.f, g.cmd.adv * dt, g.cmd.rot * dt);
        for (const auto& j : rc::anchor_jambs(r, kWidth)) if (touches(kShadow, j.mean)) ++contacts;
        c_min = std::min(c_min, clearance(r));
    }
    const float c1 = clearance(r);
    std::println("  after 0.5 s: min clearance {:+.3f} -> {:+.3f} m (lowest {:+.3f}), contacts {}", c0, c1, c_min, contacts);
    CHECK(c1 > c0);
    CHECK(c_min >= c0 - 1e-4f);
    CHECK(contacts == 0);
}

// Task 9, fix C: the rotation cap for a jamb already inside the grown body caps only the turn that CLOSES on it.
void inside_jamb_caps_only_the_closing_turn()
{
    const float margin = 0.05f;
    const RelPose r = anchor(-0.31f + 0.5f * kWidth, 0.20f, 0.f, 0.02f);   // J− ahead-left, inside the margin
    const float ccw = rc::anchor_jamb_rot_limit(r, kWidth, +1.f, kShadow, margin, 1.0f, kWmax);
    const float cw  = rc::anchor_jamb_rot_limit(r, kWidth, -1.f, kShadow, margin, 1.0f, kWmax);
    std::println("  inside jamb ahead-left: |rot| cap CCW {:.3f}, CW {:.3f}", ccw, cw);
    // Turning CCW swings the front-left corner INTO J− (closing); CW swings it away (opening).
    CHECK(ccw < 1e-3f);
    CHECK(cw > 0.5f * kWmax);
}
// Task 9 fix round 1: driving on a PREDICTED anchor, whose covariance grows with the motion — the servo slows by itself.
void adv_falls_as_sigma_lat_grows()
{
    float last = 1e9f;
    bool falling = true;
    for (const float s : {0.01f, 0.03f, 0.06f, 0.10f, 0.20f})
    {
        const auto c = cmd(anchor(-0.05f, 1.0f, 0.f, s), false);
        std::println("  sigma_lat {:.2f} m: adv {:.3f}", s, c.adv);
        falling = falling and c.adv < last + 1e-6f;
        last = c.adv;
    }
    CHECK(falling);
    // σ_lat 0.20: the advance is P(fit) of the sure one — read, on this differential base, at the offset PREDICTED where the
    // body enters the opening (try101: anchor_p_fit_at_entry; 0.76 here, 0.70 at the current offset).
    const float p20 = rc::anchor_p_fit_at_entry(kWidth, {}, kBody, 0.05f, 0.f, -1.0f, 0.f, kVmax, 0.20f);
    std::println("  sigma_lat 0.20 m: P(fit) at entry {:.3f}, adv ratio {:.3f}", p20, last / cmd(anchor(-0.05f, 1.0f, 0.f, 0.01f), false).adv);
    CHECK(p20 < 0.8f);
    CHECK(std::abs(last / cmd(anchor(-0.05f, 1.0f, 0.f, 0.01f), false).adv - p20) < 0.02f);
}
// Task 9 fix round 2 (try68): a stale anchor 90° to the robot's LEFT (robot −x side) — the wait turns CCW toward it; one to
// the right turns CW; one already ahead asks nothing; in the doorway the turn follows the normal.
void stale_anchor_left_turns_ccw()
{
    const float half_len = 0.30f;
    const float left = rc::anchor_wait_turn(anchor(-2.f, 0.f, float(M_PI_2), 0.02f), kWmax, half_len);
    const float right = rc::anchor_wait_turn(anchor(2.f, 0.f, -float(M_PI_2), 0.02f), kWmax, half_len);
    const float ahead = rc::anchor_wait_turn(anchor(0.f, 2.f, 0.f, 0.02f), kWmax, half_len);
    const float in_door = rc::anchor_wait_turn(anchor(0.f, 0.1f, 0.4f, 0.02f), kWmax, half_len);   // s = −0.1, ψ = 0.4
    std::println("  wait turn: anchor left {:+.3f}, right {:+.3f}, ahead {:+.3f}, in the doorway (ψ +0.4) {:+.3f}", left, right, ahead, in_door);
    CHECK(left > 0.5f);
    CHECK(right < -0.5f);
    CHECK(std::abs(ahead) < 1e-4f);
    CHECK(in_door > 0.f);
}
// Task 11 fix 3: the wait-turn is judged when its ask is resolvable, skipped only when it is ~0. Runs the session's gate
// (anchor_wait_turn_judged) in front of the real StallJudge, cycle by cycle at 10 Hz, heading frozen (a blocked turn).
void wait_turn_grinding_is_judged()
{
    const float window_s = 1.5f, theta_std = 0.01f;   // stuck_confirm_ms; a localiser heading σ of ~0.6°
    auto run = [&](float rot_rps) -> rc::StallVerdict
    {
        rc::StallJudge judge;
        rc::StallVerdict last = rc::StallVerdict::None;
        for (int k = 0; k <= 40; ++k)   // 4 s
        {
            const std::uint64_t t = 1000 + 100 * std::uint64_t(k);
            if (not rc::anchor_wait_turn_judged(rot_rps, window_s, theta_std)) { judge.reset(); last = rc::StallVerdict::None; continue; }
            const auto r = judge.note(true, 0.f, 0.f, 0.f, Eigen::Vector2f::Zero(), /*heading frozen*/0.f, rot_rps, t,
                                      {.slip_ratio = 0.25f, .confirm_ms = window_s * 1e3f});
            if (r.window_s > 0.f) last = r.verdict;
        }
        return last;
    };
    const auto grinding = run(0.5f);      // the wait-turn at half its cap, delivering nothing
    const auto settled  = run(0.003f);    // try68: facing the opening, tanh of a tiny bearing (0.0045 rad per window)
    std::println("  wait turn judged: 0.5 rad/s blocked → {}, 0.003 rad/s → {}; gate(0.5)={} gate(0.003)={} gate(σ unknown)={}",
                 int(grinding), int(settled), rc::anchor_wait_turn_judged(0.5f, window_s, theta_std),
                 rc::anchor_wait_turn_judged(0.003f, window_s, theta_std), rc::anchor_wait_turn_judged(0.5f, window_s, -1.f));
    CHECK(grinding == rc::StallVerdict::Spinning);   // a turn ground against a jamb is caught
    CHECK(settled == rc::StallVerdict::None);        // try68's false positive stays away
    CHECK(rc::anchor_wait_turn_judged(-0.5f, window_s, theta_std));   // either direction
    CHECK(not rc::anchor_wait_turn_judged(0.5f, window_s, -1.f));      // no heading σ ⇒ the ask cannot be weighed
}
}  // namespace

// ── Task 12 fix A: the swung leaf ──────────────────────────────────────────────────────────────────────────────────
// try70's door: the leaf hinged on one jamb, swung 90° TOWARD the robot (its side, anchor −y). Anchor frame: the hinge on
// the +x jamb's robot-side face, the leaf running back along −y for its length.
constexpr float kWallHalfT = 0.05f, kMargin = 0.05f;
rc::interaction::AnchorLeaf leaf_toward(float phi = std::numbers::pi_v<float> / 2.f, float sigma_phi = 0.05f)
{ return rc::interaction::make_anchor_leaf(+0.5f * kWidth, -kWallHalfT, -1.f, -1.f, kWidth, phi, sigma_phi); }

// The TRUE leaf slab against the TRUE body (no margin): its centre line sampled at 5 mm, each sample a disc of the leaf's
// half-thickness — a contact is any disc reaching inside the body box.
int leaf_contacts(const RelPose& r, const rc::interaction::AnchorLeaf& l, const std::vector<Eigen::Vector2f>& body)
{
    const float psi = r.x.z(), c = std::cos(psi), s = std::sin(psi);
    int n = 0;
    for (int i = 0; i <= 200; ++i)
    {
        const Eigen::Vector2f qa = l.hinge + (l.tip - l.hinge) * (float(i) / 200.f);
        const Eigen::Vector2f q = r.x.head<2>() + Eigen::Vector2f(c * qa.x() - s * qa.y(), s * qa.x() + c * qa.y());
        if (rc::anchor_signed_distance(body, q) < l.half_thick) ++n;
    }
    return n;
}

// The session's chain on a differential base: the servo (with or without the leaf), then the jamb+leaf guard.
using Leaves = std::span<const rc::interaction::AnchorLeaf>;
Leaves one(const rc::interaction::AnchorLeaf* l) { return l != nullptr ? Leaves(l, 1) : Leaves(); }
rc::AnchorCommand chain(const RelPose& r, Leaves l, const std::vector<Eigen::Vector2f>& body)
{
    const auto c = rc::anchor_cross_command(r, kWidth, kBody + kMargin, kStandoff, kVmax, kWmax, false, l);
    return rc::anchor_jamb_guard(r, kWidth, c, false, body, kMargin, 0.5f, kVmax, kWmax, l).cmd;
}

void clear_gap_beside_the_leaf()
{
    const auto l90 = leaf_toward();
    const auto g90 = rc::anchor_clear_gap(kWidth, &l90);
    // Hinged at +w/2: the gap runs from −w/2 to the leaf's −x face (thickness 2·0.02 off the hinge side).
    CHECK(std::abs(g90.width - (kWidth - 2.f * float(rc::interaction::kLeafHalfThickM))) < 1e-4f);   // the slab is inside the opening
    CHECK(g90.centre < 0.f and g90.centre > -0.03f);
    const auto l60 = leaf_toward(std::numbers::pi_v<float> / 3.f, 0.f);   // ajar: projects L·cos 60° = 0.45 m into the opening
    const auto g60 = rc::anchor_clear_gap(kWidth, &l60);
    CHECK(g60.width < 0.5f * kWidth and g60.width > 0.5f * kWidth - 4.f * float(rc::interaction::kLeafHalfThickM));
    CHECK(g60.centre < -0.2f);
    const auto l120 = leaf_toward(2.f * std::numbers::pi_v<float> / 3.f, 0.f);   // past 90°: folded back over the wall
    CHECK(rc::anchor_clear_gap(kWidth, &l120).width > kWidth - 2.f * float(rc::interaction::kLeafHalfThickM) - 1e-3f);
    CHECK(rc::anchor_clear_gap(kWidth, nullptr).width == kWidth and rc::anchor_clear_gap(kWidth, nullptr).centre == 0.f);
    // The encoding round-trips: hinge, free edge, σ.
    auto l2 = leaf_toward(); l2.p = 0.3f; auto l1 = l90; l1.p = 0.7f;
    const auto [xs, ys] = rc::interaction::encode_anchor_leaves({l1, l2});
    const auto d = rc::interaction::decode_anchor_leaves(xs, ys, l90.phi);
    CHECK(d.size() == 2 and (d[0].hinge - l90.hinge).norm() < 1e-6f and (d[0].tip - l90.tip).norm() < 1e-5f
          and std::abs(d[0].sigma_phi - 0.05f) < 1e-4f and std::abs(d[0].p - 0.7f) < 1e-6f and std::abs(d[1].p - 0.3f) < 1e-6f);
    CHECK(rc::interaction::decode_anchor_leaves({0.f, 0.f}, {0.f, 0.f}, 0.f).empty());
    CHECK(rc::interaction::decode_anchor_leaves({xs[0], xs[1], xs[2]}, {ys[0], ys[1], ys[2]}, 0.f).size() == 1);   // the 3-vertex form
    std::println("  clear gap: 90 deg {:.3f} m at {:+.3f}; 60 deg {:.3f} m at {:+.3f}", g90.width, g90.centre, g60.width, g60.centre);
}

void leaf_on_robot_side_not_contacted()
{
    // A differential robot 2 m before the opening, 0.30 m off-centre TOWARD THE HINGE side, the leaf at 90° on its side.
    const auto l = leaf_toward();
    // From 2 m, and from 1.2 m — the body's nose 2 cm short of the leaf's free edge, where the jamb-only law (the old one,
    // run alongside as the control) drives the body's right side into the leaf.
    for (const float s0 : {2.0f, 1.2f})
    for (const bool model_leaf : {true, false})
    {
        RelPose r = anchor(-0.30f, s0, 0.f, 0.02f);
        int leaf_hits = 0, jamb_hits = 0, steps = 0;
        bool through = false;
        float s_min_contact = 9.f;
        for (; steps < 600 and not through; ++steps)
        {
            const auto c = chain(r, one(model_leaf ? &l : nullptr), kShadow);
            if (c.through) { through = true; break; }
            for (int k = 0; k < 5; ++k)
            {
                r.x = rc::interaction::move_mean(r.x, 0.f, c.adv * 0.01f, c.rot * 0.01f);
                if (const int h = leaf_contacts(r, l, kShadow); h > 0) { ++leaf_hits; s_min_contact = std::min(s_min_contact, rc::anchor_cross_geometry(r).s); }
                for (const auto& j : rc::anchor_jambs(r, kWidth))
                    if (touches(kShadow, j.mean)) ++jamb_hits;
            }
        }
        const auto g = rc::anchor_cross_geometry(r);
        std::println("  from s {:+.1f}, leaf {}: {} steps, e {:+.3f} s {:+.2f}, leaf contacts {} (first at s {:+.2f}), jamb contacts {}",
                     -s0, model_leaf ? "MODELLED  " : "unmodelled", steps, g.e, g.s, leaf_hits, s_min_contact, jamb_hits);
        if (model_leaf)
        {
            CHECK(leaf_hits == 0);
            CHECK(jamb_hits == 0);
            CHECK(g.s > 1.0f);   // through: well past the plane (v_n tapers to 0 AT the standoff, so `through` itself is asymptotic)
        }
        else if (s0 < 1.5f)
            CHECK(leaf_hits > 0);   // the control: without the leaf this approach DOES hit it
    }
}

void rotation_near_leaf_capped()
{
    // Beside the leaf, 0.55 m before the plane: the 90° leaf runs past the body's right side 0.37 m from its centre, outside
    // the body (0.294) but inside its grown reach — any turn swings a right-hand corner (radius 0.373 m) out toward it.
    const auto l = leaf_toward();
    const RelPose r0 = anchor(-0.08f, 0.55f, 0.f, 0.02f);
    CHECK(leaf_contacts(r0, l, kShadow) == 0);
    for (const float dir : {+1.f, -1.f})
    {
        const float w_leaf = rc::anchor_jamb_rot_limit(r0, kWidth, dir, kShadow, kMargin, 0.5f, kWmax, one(&l));
        const float w_none = rc::anchor_jamb_rot_limit(r0, kWidth, dir, kShadow, kMargin, 0.5f, kWmax, Leaves());
        std::println("  beside the leaf, turn {}: |rot| cap {:.3f} with the leaf, {:.3f} without", dir > 0 ? "CCW" : "CW", w_leaf, w_none);
        CHECK(w_none > kWmax - 1e-3f);   // the jambs alone do not see it
        CHECK(w_leaf < 0.5f * kWmax);    // the leaf caps the sweep
        RelPose r = r0;
        int contacts = 0;
        for (int k = 0; k < 100; ++k)    // 1 s at dt = 0.01, the cap re-evaluated each step
        {
            const float w = rc::anchor_jamb_rot_limit(r, kWidth, dir, kShadow, kMargin, 0.5f, kWmax, one(&l));
            r.x = rc::interaction::move_mean(r.x, 0.f, 0.f, dir * w * 0.01f);
            contacts += leaf_contacts(r, l, kShadow);
        }
        CHECK(contacts == 0);
    }
    // ...and translation toward it is capped too: a sideways (holonomic) push into the leaf.
    const float v = rc::anchor_jamb_speed_limit(r0, kWidth, Eigen::Vector2f(1.f, 0.f), kShadow, kMargin, 0.5f, kVmax, one(&l));
    std::println("  beside the leaf, a push toward it: speed cap {:.3f}", v);
    CHECK(v < 0.05f);
}

// Task 12 fix C: the approach standpoint, asked in the anchor's frame, ends ON the cross line at the producer's depth.
void approach_standpoint_on_the_cross_line()
{
    // The opening 2.2 m ahead, its normal rotated 0.2 rad; the room-frame standpoint the producer published lands 0.31 m off
    // the normal (try70 legs 3 and 5) at 1.9 m before the plane.
    const RelPose r = anchor(0.4f, 2.2f, 0.2f, 0.02f);
    const auto g0 = rc::anchor_cross_geometry(r);
    const Eigen::Vector2f offered = r.x.head<2>() + g0.a * (-0.31f) + g0.n * (-1.9f);
    for (const bool with_leaf : {false, true})
    {
        const auto l = leaf_toward();
        const auto* lp = with_leaf ? &l : nullptr;
        const Eigen::Vector2f q = rc::anchor_standpoint(r, kWidth, one(lp), kBody + kMargin, offered);
        // The robot standing at q (heading along the normal): the anchor seen from there.
        RelPose seen = r;
        seen.x = rc::interaction::move_mean(r.x, q.x(), q.y(), 0.f);   // translate the robot to q (robot-frame step)
        const auto g = rc::anchor_cross_geometry(seen);
        const float line = rc::anchor_cross_line(kWidth, one(lp), kBody + kMargin, rc::anchor_cross_geometry(r).sigma_lat);
        std::println("  approach standpoint{}: offered e {:+.3f} s {:+.3f} -> e {:+.3f} s {:+.3f} (the line at {:+.3f})",
                     with_leaf ? " (leaf at 90 deg)" : "", g0.a.dot(offered - r.x.head<2>()), g0.n.dot(offered - r.x.head<2>()),
                     g.e, g.s, line);
        CHECK(std::abs(g.e - line) < 0.005f);       // on the cross line
        CHECK(std::abs(g.s - (-1.9f)) < 0.005f);    // at the producer's depth
    }
    CHECK((rc::anchor_standpoint(RelPose{}, 0.f, Leaves(), kBody, offered) - offered).norm() == 0.f);   // no width: as published
    // I-5: fused by precision with the room frame — a room σ equal to the anchor's lands halfway between line and offer.
    {
        const float sa = rc::anchor_cross_geometry(r).sigma_lat;
        const Eigen::Vector2f q = rc::anchor_standpoint(r, kWidth, Leaves(), kBody, offered, sa);
        RelPose seen = r; seen.x = rc::interaction::move_mean(r.x, q.x(), q.y(), 0.f);
        const auto g = rc::anchor_cross_geometry(seen);
        std::println("  fused with an equally sure room frame: e {:+.3f} (offered -0.310, line 0)", g.e);
        CHECK(std::abs(g.e - (-0.155f)) < 0.005f);
    }
}

// Fix round 1, I-1: a predicted pose carried while parked in the doorway grows its covariance (rc::interaction::grow_unseen),
// so the servo slows on an old sighting instead of driving it at full speed.
void predicted_pose_slows_while_parked()
{
    const auto l = leaf_toward();
    const auto lv = one(&l);
    // Seen at the mouth (s −0.3), 4 cm off the cross line, σ_lat 0.019 m (try70's claimed σ on an 8σ-wrong pose).
    const float line = rc::anchor_cross_line(kWidth, lv, kBody + kMargin, 0.019f);
    RelPose seen = anchor(-(line + 0.04f), 0.3f, 0.f, 0.019f);
    RelPose carried = seen;
    for (int k = 0; k < 50; ++k) rc::interaction::grow_unseen(carried, 0.1f, 3.0f);   // 5 s parked, unseen, kAnchorPredictMs 3 s
    const float s0 = rc::anchor_cross_geometry(seen).sigma_lat, s1 = rc::anchor_cross_geometry(carried).sigma_lat;
    const auto c0 = rc::anchor_cross_command(seen, kWidth, kBody + kMargin, kStandoff, kVmax, kWmax, false, lv);
    const auto c1 = rc::anchor_cross_command(carried, kWidth, kBody + kMargin, kStandoff, kVmax, kWmax, false, lv);
    const float p0 = rc::anchor_p_fit_leaves(kWidth, lv, kBody + kMargin, rc::anchor_cross_geometry(seen).e, s0);
    const float p1 = rc::anchor_p_fit_leaves(kWidth, lv, kBody + kMargin, rc::anchor_cross_geometry(carried).e, s1);
    std::println("  parked 5 s unseen: sigma_lat {:.3f} -> {:.3f} m, P(fits) {:.3f} -> {:.3f}, adv {:.3f} -> {:.3f}", s0, s1, p0, p1,
                 c0.adv, c1.adv);
    CHECK(s1 > 4.f * s0);
    CHECK(c1.adv < 0.75f * c0.adv);
    CHECK(c0.adv > 0.f);
}

// Fix round 1, I-4 / I-6: the leaf hypotheses.
void leaf_hypotheses_weigh_the_minority()
{
    // The hinge believed at P 0.6 (log-odds 0.405), the swing sure (toward the robot): two 90° leaves, one per jamb.
    const auto hyp = rc::interaction::anchor_leaf_hypotheses(+0.5f * kWidth, -kWallHalfT, -1.f, -1.f, kWidth,
                                                            std::numbers::pi_v<float> / 2.f, 0.05f, std::log(0.6f / 0.4f), 20.f);
    CHECK(hyp.size() == 4 and std::abs(hyp[0].p - 0.6f) < 1e-3f and std::abs(hyp[1].p - 0.4f) < 1e-3f);
    const RelPose r = anchor(0.f, 1.0f, 0.f, 0.02f);
    int left = 0, right = 0;
    float p_left = 0.f, p_right = 0.f;
    for (const auto& o : rc::anchor_obstacles(r, kWidth, hyp))
        if (o.radius > 0.f and o.p > 0.1f) { if (o.mean.x() > 0.f) { ++right; p_right = o.p; } else { ++left; p_left = o.p; } }
    const float line = rc::anchor_cross_line(kWidth, hyp, kBody + kMargin, 0.02f);
    std::println("  hinge P 0.6: leaf obstacles left {} (p {:.2f}), right {} (p {:.2f}); cross line {:+.3f} m (MAP alone -0.020)",
                 left, p_left, right, p_right, line);
    CHECK(left > 0 and right > 0);
    CHECK(std::abs(line) < 0.01f);
    // I-6: ajar 20° toward the robot, sure: the gap it leaves is a slit — the line stays on the opening's centre.
    const auto l20 = leaf_toward(20.f * std::numbers::pi_v<float> / 180.f, 0.f);
    const float line20 = rc::anchor_cross_line(kWidth, one(&l20), kBody + kMargin, 0.02f);
    const auto g20 = rc::anchor_clear_gap(kWidth, &l20);
    std::println("  ajar 20 deg: gap {:.3f} m at {:+.3f}; cross line {:+.4f} m", g20.width, g20.centre, line20);
    CHECK(std::abs(line20) < 0.005f);
    CHECK(std::abs(g20.centre) > 0.1f);   // the slit's centre is far off — the line does not go there
    // The guard weighs a leaf by its probability: a push into a 0.4-probable leaf is slowed, not stopped like a sure one.
    auto sure = leaf_toward(); sure.p = 1.f;
    auto unsure = leaf_toward(); unsure.p = 0.4f;
    const RelPose r0 = anchor(-0.08f, 0.55f, 0.f, 0.02f);
    const float v_sure = rc::anchor_jamb_speed_limit(r0, kWidth, Eigen::Vector2f(1.f, 0.f), kShadow, kMargin, 0.5f, kVmax, one(&sure));
    const float v_unsure = rc::anchor_jamb_speed_limit(r0, kWidth, Eigen::Vector2f(1.f, 0.f), kShadow, kMargin, 0.5f, kVmax, one(&unsure));
    std::println("  push into the leaf: cap {:.3f} (sure) vs {:.3f} (P 0.4)", v_sure, v_unsure);
    CHECK(v_sure < v_unsure and v_unsure < kVmax);
}

// Task 12 Fix E: the cross ends when the robot is THROUGH on the anchor, not at the room switch.
void cross_ends_through_not_at_the_room_switch()
{
    const RelPose at03 = anchor(0.f, -0.3f, 0.f, 0.02f), at20 = anchor(0.f, -2.0f, 0.f, 0.02f);   // s = +0.3, s = +2.0
    const bool t03 = rc::anchor_cross_command(at03, kWidth, kBody, kStandoff, kVmax, kWmax, false).through;
    const bool t20 = rc::anchor_cross_command(at20, kWidth, kBody, kStandoff, kVmax, kWmax, false).through;
    using E = rc::AnchorCrossEnd;
    CHECK(rc::anchor_cross_end(true, true, t03) == E::Continue);    // room switch at s +0.3 with an anchor: keep going
    CHECK(rc::anchor_cross_end(true, true, t20) == E::Through);     // s +2.0: through
    CHECK(rc::anchor_cross_end(true, false, t20) == E::Through);
    CHECK(rc::anchor_cross_end(false, true, t03) == E::RoomSwitch); // no anchor: the old room-switch rule
    CHECK(rc::anchor_cross_end(false, false, t03) == E::Continue);
    std::println("  cross end: switch at s +0.3 on the anchor -> continue (through {}), s +2.0 -> through ({})", t03, t20);
}

// Task 13 (try74 leg 1: 40 s at s +1.56 waiting for a far-side anchor): past the room switch the cross completes on the pose
// odometry carries — no anchor seen after the switch, the carried pose's covariance growing (motion + unseen time), the
// guards still in the chain — and ends `through` without waiting.
void clearing_completes_on_odometry()
{
    const auto l = leaf_toward();
    RelPose r = anchor(0.02f, -0.3f, 0.05f, 0.02f);   // the switch: 0.3 m past the plane, roughly centred
    float t = 0.f;
    bool through = false;
    int leaf_hits = 0, jamb_hits = 0;
    for (int k = 0; k < 600 and not through; ++k)   // 0.1-s cycles, 60 s cap
    {
        const auto c = chain(r, one(&l), kShadow);
        if (c.through) { through = true; break; }
        rc::rt::Twist tw;
        tw.linear = {c.side, c.adv, 0.f};
        tw.angular = {0.f, 0.f, c.rot};
        r = rc::interaction::predict(r, tw, Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f), 0.1f);   // odometry carries it (+ k_rot term)
        rc::interaction::grow_unseen(r, 0.1f, 3.0f);                                     // unseen all along
        t += 0.1f;
        leaf_hits += leaf_contacts(r, l, kShadow);
        for (const auto& j : rc::anchor_jambs(r, kWidth)) if (touches(kShadow, j.mean)) ++jamb_hits;
    }
    const auto g = rc::anchor_cross_geometry(r);
    std::println("  clearing on odometry: through {} after {:.1f} s at s {:+.2f} (sigma_lat {:.3f} m), contacts leaf {} jamb {}",
                 through, t, g.s, g.sigma_lat, leaf_hits, jamb_hits);
    CHECK(through);
    CHECK(t < 20.f);   // no 40-s wait
    CHECK(leaf_hits == 0 and jamb_hits == 0);
    CHECK(rc::anchor_cross_end(true, true, through) == rc::AnchorCrossEnd::Through);
}

// Task 16 (try76 leg 2): a stale anchor past the plane, before any room switch, is driven on to `through`; before it, waited for.
void stale_past_the_plane_continues()
{
    CHECK(rc::anchor_drive_on(false, false, true, anchor(0.f, -0.36f, 0.f, 0.03f)));   // seen, then stale at s +0.36: drive
    CHECK(not rc::anchor_drive_on(false, false, true, anchor(0.f, 0.30f, 0.f, 0.02f))); // s −0.30: wait for a fresh one
    CHECK(rc::anchor_drive_on(true, false, false, anchor(0.f, 0.30f, 0.f, 0.02f)));    // fresh: drive
    CHECK(rc::anchor_drive_on(false, true, false, anchor(0.f, 0.30f, 0.f, 0.02f)));    // clearing: drive
    // Task 16 fix (try81 leg 3): a NEVER-SEEN anchor (the retry's room initialisation, carried) at "s +0.20", σ_lat 0.38: waits.
    CHECK(not rc::anchor_drive_on(false, false, false, anchor(0.f, -0.20f, 0.f, 0.38f)));
    // From s +0.36, never seen again, carried by odometry with the guards: through, no wait.
    const auto l = leaf_toward();
    RelPose r = anchor(0.02f, -0.36f, 0.f, 0.02f);
    float t = 0.f; bool through = false; int waited = 0;
    for (int k = 0; k < 600 and not through; ++k)
    {
        if (not rc::anchor_drive_on(false, false, true, r)) { ++waited; t += 0.1f; continue; }
        const auto c = chain(r, one(&l), kShadow);
        if (c.through) { through = true; break; }
        rc::rt::Twist tw; tw.linear = {c.side, c.adv, 0.f}; tw.angular = {0.f, 0.f, c.rot};
        r = rc::interaction::predict(r, tw, Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f), 0.1f);
        rc::interaction::grow_unseen(r, 0.1f, 3.0f);
        t += 0.1f;
    }
    std::println("  stale past the plane: through {} after {:.1f} s, {} cycles waited", through, t, waited);
    CHECK(through and waited == 0 and t < 20.f);
}

// Final review I-1: the anchor gone past the room switch with no held gap ends the cross (never a room route on old-room
// coordinates), and the passage then takes its room-switch end — no unbounded `clearing`.
void anchor_gone_while_clearing_ends()
{
    using A = rc::AnchorGoneAction;
    CHECK(rc::anchor_gone_action(true, true) == A::CarryOnGap);     // held gap + near side seen: carry to through
    CHECK(rc::anchor_gone_action(true, false) == A::CarryOnGap);
    CHECK(rc::anchor_gone_action(false, false) == A::RoomRoute);    // before the switch: the room route is this room's
    CHECK(rc::anchor_gone_action(false, true) == A::EndClearing);   // past it, nothing to carry: end
    // The session drops `engaged` on EndClearing; the passage's decision then ends the cross by the room-switch rule.
    CHECK(rc::anchor_cross_end(/*anchored=*/false, /*clearing=*/true, /*through=*/false) == rc::AnchorCrossEnd::RoomSwitch);
    // Before the fix `engaged` stayed true: Continue for ever.
    CHECK(rc::anchor_cross_end(true, true, false) == rc::AnchorCrossEnd::Continue);
    // A gap that is held but whose near side was never observed does not carry (the retry-near-the-plane case).
    rc::CrossThroughGate gate;
    CHECK(not rc::cross_continues_without_anchor(true, gate));
    CHECK(rc::anchor_gone_action(rc::cross_continues_without_anchor(true, gate), true) == A::EndClearing);
}

// Task 8: the cross's COMPLETION, extracted from step_anchor_cross so the local policy's doorway drive and the servo share it:
// through only at s >= standoff AND observed on the near side in this claim; s >= standoff without that is not a crossing.
static void cross_progress_semantics()
{
    const float standoff = 2.0f, half_len = 0.3f;
    rc::interaction::RelPose r; r.x = Eigen::Vector3f(0.f, -2.1f, 0.f);   // centre 2.1 m BEHIND the robot: s = +2.1
    rc::CrossThroughGate gate;
    CHECK(rc::cross_progress(r, standoff, gate) == rc::CrossProgress::NotACrossing);   // never seen on the near side
    CHECK(not rc::cross_progress_complete(r, standoff, gate));
    rc::interaction::RelPose before; before.x = Eigen::Vector3f(0.f, 1.0f, 0.f);       // s = -1.0: wholly before the plane
    CHECK(rc::cross_progress(before, standoff, gate) == rc::CrossProgress::Going);
    CHECK(gate.observe(true, rc::anchor_cross_geometry(before).s, half_len));
    CHECK(rc::cross_progress(r, standoff, gate) == rc::CrossProgress::Through);
    CHECK(rc::cross_progress_complete(r, standoff, gate));
    rc::interaction::RelPose mid; mid.x = Eigen::Vector3f(0.f, -1.0f, 0.f);           // s = +1.0 < standoff
    CHECK(rc::cross_progress(mid, standoff, gate) == rc::CrossProgress::Going);
    rc::interaction::RelPose bad; bad.x = Eigen::Vector3f(NAN, 0.f, 0.f);
    CHECK(rc::cross_progress(bad, standoff, gate) == rc::CrossProgress::Going);         // nothing to judge: not through
    rc::interaction::RelPose badP = r; badP.P(0, 0) = NAN;                              // review M3: a non-finite covariance
    CHECK(rc::cross_progress(badP, standoff, gate) == rc::CrossProgress::Going);
}

int main()
{
    centred_goes_straight();
    offset_slides_sideways();
    rotated_anchor_turns_and_slides();
    uncertain_slows();
    through_after_standoff();
    degenerate_inputs_do_not_move();
    differential_converges_laterally();
    combined_offset_and_heading();
    jamb_stops_the_body();
    jamb_does_not_lock_recentering();
    rotation_near_jamb_is_capped();
    jamb_deadlock_backs_off();
    inside_jamb_caps_only_the_closing_turn();
    adv_falls_as_sigma_lat_grows();
    stale_anchor_left_turns_ccw();
    wait_turn_grinding_is_judged();
    clear_gap_beside_the_leaf();
    leaf_on_robot_side_not_contacted();
    rotation_near_leaf_capped();
    approach_standpoint_on_the_cross_line();
    predicted_pose_slows_while_parked();
    leaf_hypotheses_weigh_the_minority();
    cross_ends_through_not_at_the_room_switch();
    clearing_completes_on_odometry();
    stale_past_the_plane_continues();
    anchor_gone_while_clearing_ends();
    cross_progress_semantics();
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
