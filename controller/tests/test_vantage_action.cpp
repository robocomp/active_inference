// vantage_action + reaim_on_anchor (controller_anchor_servo.h; door-passable spec step 2, 2026-11-05): at the vantage of an
// open door the passage always has an action. warm61 leg 3 stood 9 min: the anchor put the opening 0.46 m to the side
// (within ±0.44 failed), the servo would not start, the route had arrived — nothing was commanded.
// Build+run (from controller/; relative_pose.h pulls dsr_api.h, hence Qt/fastdds — as test_anchor_servo.cpp):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_vantage_action.cpp -o /tmp/tva
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tva          (one line)
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
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

rc::interaction::RelPose anchor(float x, float y, float yaw)
{
    rc::interaction::RelPose r;
    r.x = {x, y, yaw};
    r.P = Eigen::Matrix3f::Identity() * 1e-4f;
    return r;
}
}  // namespace

int main()
{
    using rc::VantageAction;
    // warm61_reaims: open, seen, before the plane, 0.46 m off a 0.88 m opening, route arrived
    {
        const auto g = rc::anchor_cross_geometry(anchor(0.46f, 1.58f, 0.0f));
        const bool within = std::abs(g.e) < 0.5f * 0.88f;
        CHECK(not within);
        CHECK(rc::vantage_action(false, true, true, g.s < 0.f, within, true, true, true) == VantageAction::ReAim);
        const auto goal = rc::reaim_on_anchor(g);
        CHECK(std::abs(goal.p_robot.x() - 0.46f) < 1e-5f and std::abs(goal.p_robot.y()) < 1e-5f);
        CHECK(std::abs(goal.yaw_robot) < 1e-6f);
        // standing at the goal (the anchor seen from there) puts the robot on the axis
        const auto g2 = rc::anchor_cross_geometry(anchor(0.46f - goal.p_robot.x(), 1.58f - goal.p_robot.y(), 0.0f));
        CHECK(std::abs(g2.e) < 1e-5f);
    }
    // a rotated anchor: the goal heading is the normal's, the goal point is on the axis at the current progress
    {
        const auto g = rc::anchor_cross_geometry(anchor(0.3f, 1.5f, 0.2f));
        const auto goal = rc::reaim_on_anchor(g);
        CHECK(std::abs(goal.yaw_robot - 0.2f) < 1e-6f);
        const Eigen::Vector2f A(0.3f, 1.5f);
        CHECK(std::abs(g.a.dot(goal.p_robot - A)) < 1e-5f);          // on the normal line through the opening
        CHECK(std::abs(g.n.dot(goal.p_robot - A) - g.s) < 1e-5f);    // at the same progress s
    }
    // no_mouth_holds_only_when_placed (clean50, 2026-11-11): placed in front of the opening (near side, within, facing) the
    // robot holds for the mouth; off-axis or misfaced it re-aims first — holding where no view through the door is possible
    // waited 8m47s for a mouth that could not appear
    CHECK(rc::vantage_action(false, true, false, true, true, true, true, true) == VantageAction::HoldNoMouth);
    CHECK(rc::vantage_action(false, false, true, true, true, true, true, true) == VantageAction::HoldNoMouth);
    CHECK(rc::vantage_action(false, true, false, true, false, true, true, true) == VantageAction::ReAim);    // off-axis: re-aim
    CHECK(rc::vantage_action(false, true, false, true, false, false, true, true) == VantageAction::ReAim);   // clean50: e -0.56, 117 deg
    CHECK(rc::vantage_action(false, true, false, true, true, false, true, true) == VantageAction::ReAim);    // misfaced: re-aim
    CHECK(rc::vantage_action(false, true, false, true, false, true, true, false) == VantageAction::RouteDrives);   // stale anchor: no re-aim
    // past_plane_never_reaims
    CHECK(rc::vantage_action(false, true, true, false, false, true, true, true) == VantageAction::RouteDrives);
    // engaged_servo_wins
    CHECK(rc::vantage_action(true, false, false, false, false, false, true, true) == VantageAction::Servo);
    CHECK(rc::vantage_action(false, true, true, true, true, true, false, true) == VantageAction::Servo);
    // route_not_arrived_drives
    CHECK(rc::vantage_action(false, true, true, true, false, true, false, true) == VantageAction::RouteDrives);
    // stale_anchor_never_reaims (final review I2): a stale robot-frame anchor composed with the CURRENT pose drags the goal
    // with the robot — no re-aim until the anchor is fresh again
    CHECK(rc::vantage_action(false, true, true, true, false, true, true, false) != VantageAction::ReAim);
    // facing alone failing also re-aims once the route has arrived
    CHECK(rc::vantage_action(false, true, true, true, true, false, true, true) == VantageAction::ReAim);

    // ── FACE THE DOOR (2026-11-15; chat_e2e20 `cross door_25`: robot 1.45 m in front, heading -176 deg, 50+ s idle) ──────────
    // At the vantage with the route arrived, the robot faces AWAY: the anchor is stale, so vantage_action says RouteDrives and nothing
    // drove. face_decision is the passage's own turn: the goal is the DOOR's heading (robot forward + the anchor's heading_err), the
    // turn runs until the servo's start condition (|heading_err| < 90) holds, and a blocked arc is a REPORT, never a wait.
    using rc::FaceKind;
    constexpr float kPi = std::numbers::pi_v<float>;
    constexpr float kD = kPi / 180.f;
    const float door_yaw = -kPi / 2.f;                      // the door normal's forward heading
    const auto route_drives_stale = rc::vantage_action(false, true, false, true, true, false, true, false);   // the e2e20 verdict
    CHECK(route_drives_stale == VantageAction::RouteDrives);
    // start at -176 deg off the door: Turn, commanded to the door's heading
    {
        const float yaw = door_yaw + 176.f * kD;                       // the body's forward heading
        const float err = rc::interaction::wrap_pi(door_yaw - yaw);   // the anchor's heading_err (CCW from forward to the normal)
        const auto d = rc::face_decision(route_drives_stale, true, yaw, err, false, false);
        CHECK(d.kind == FaceKind::Turn);
        CHECK(std::abs(rc::interaction::wrap_pi(d.yaw - door_yaw)) < 1e-4f);
    }
    // the first cycle turned it to 80 deg off with the anchor still stale: STILL turning (the 90 deg boundary no longer stops it)
    {
        const float yaw = door_yaw + 80.f * kD, err = rc::interaction::wrap_pi(door_yaw - yaw);
        const auto d = rc::face_decision(route_drives_stale, true, yaw, err, true, false);
        CHECK(d.kind == FaceKind::Turn);
        CHECK(std::abs(rc::interaction::wrap_pi(d.yaw - door_yaw)) < 1e-4f);
        // ...but a turn that has not begun is not started for an error the servo already accepts
        CHECK(rc::face_decision(route_drives_stale, true, yaw, err, false, false).kind == FaceKind::Proceed);
    }
    // facing it with a fresh anchor: the servo / re-aim own the decision
    CHECK(rc::face_decision(VantageAction::Servo, true, 0.f, 0.1f, true, false).kind == FaceKind::Proceed);
    CHECK(rc::face_decision(VantageAction::ReAim, true, 0.f, 3.f, true, false).kind == FaceKind::Proceed);
    CHECK(rc::face_decision(VantageAction::HoldNoMouth, true, 0.f, 0.1f, true, false).kind == FaceKind::Proceed);
    // route not arrived: it drives
    CHECK(rc::face_decision(VantageAction::RouteDrives, false, 0.f, 3.f, false, false).kind == FaceKind::Proceed);
    // the arc is blocked: a REPORT toward the door's heading, not a wait (and not a turn), whether or not the turn had begun
    CHECK(rc::face_decision(VantageAction::RouteDrives, true, 0.f, 3.f, false, true).kind == FaceKind::Report);
    CHECK(rc::face_decision(VantageAction::RouteDrives, true, 0.f, 1.7f, true, true).kind == FaceKind::Report);

    std::println("{}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
