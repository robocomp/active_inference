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

// A rectangular body: half-width kBody (x), half-length 0.35 (y). Its support along a robot-frame direction.
float rect_support(const Eigen::Vector2f& d)
{
    const Eigen::Vector2f u = d.normalized();
    return kBody * std::abs(u.x()) + 0.35f * std::abs(u.y());
}

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
    // ψ = 0, the jamb J+ = A + (w/2)·a placed 0.30 m straight ahead of the body centre.
    const auto r = anchor(-0.5f * kWidth, 0.30f, 0.f, 0.03f);
    const float v = rc::anchor_jamb_speed_limit(r, kWidth, Eigen::Vector2f(0.f, 1.f), rect_support, 1.0f, kVmax);
    const auto c = cmd(r, false);
    CHECK(std::min(c.adv, v) <= 0.01f);
    CHECK(v <= 0.01f);
    // Centred in a 0.9 m opening 1 m ahead: the jambs are beside the corridor, nothing limited.
    const auto r2 = anchor(0.f, 1.f, 0.f, 0.03f);
    CHECK(rc::anchor_jamb_speed_limit(r2, kWidth, Eigen::Vector2f(0.f, 1.f), rect_support, 1.0f, kVmax) == kVmax);
    // The same jamb 1 m ahead in the corridor: a finite, braking-distance ceiling.
    const auto r3 = anchor(-0.5f * kWidth, 1.0f, 0.f, 0.03f);
    const float v3 = rc::anchor_jamb_speed_limit(r3, kWidth, Eigen::Vector2f(0.f, 1.f), rect_support, 1.0f, kVmax);
    CHECK(v3 > 0.f and v3 <= kVmax);
    // A larger along-wall sigma inflates the jamb: a jamb just beside the corridor at small sigma enters it.
    const auto beside = anchor(-0.5f * kWidth + kBody + 0.05f, 0.5f, 0.f, 0.01f);
    CHECK(rc::anchor_jamb_speed_limit(beside, kWidth, Eigen::Vector2f(0.f, 1.f), rect_support, 1.0f, kVmax) == kVmax);
    const auto beside_unsure = anchor(-0.5f * kWidth + kBody + 0.05f, 0.5f, 0.f, 0.10f);
    CHECK(rc::anchor_jamb_speed_limit(beside_unsure, kWidth, Eigen::Vector2f(0.f, 1.f), rect_support, 1.0f, kVmax)
          < kVmax);
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
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
