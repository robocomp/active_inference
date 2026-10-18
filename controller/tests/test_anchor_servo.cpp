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

rc::AnchorCommand cmd(const RelPose& r)
{ return rc::anchor_cross_command(r, kWidth, kBody, kStandoff, kVmax, kWmax); }

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
    const auto c = rc::anchor_cross_command(anchor(0.f, 2.f, 0.f, 0.03f), 0.f, kBody, kStandoff, kVmax, kWmax);
    CHECK(c.adv < 1e-3f);
    // A non-finite pose ⇒ zero command.
    const auto z = cmd(anchor(std::nanf(""), 2.f, 0.f, 0.03f));
    CHECK(z.adv == 0.f and z.side == 0.f and z.rot == 0.f and not z.through);
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
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
