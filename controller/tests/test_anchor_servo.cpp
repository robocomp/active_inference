// anchor_cross_command (controller_anchor_servo.h): crossing a door by servoing on its robot-frame anchor.
// Build+run (pure Eigen + std, but relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_anchor_servo.cpp -o /tmp/tas
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tas          (one line)
// (run from controller/)
#include "../src/controller_anchor_servo.h"

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
    CHECK(last < 0.75f * cmd(anchor(-0.05f, 1.0f, 0.f, 0.01f), false).adv);   // σ_lat 0.20: P(fit) ≈ 0.70
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
}  // namespace

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
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
