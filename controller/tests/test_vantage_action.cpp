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
    // At the vantage with the route arrived and the door believed open, the robot faces AWAY: the anchor is not fresh (it cannot
    // see the door behind it) so vantage_action says RouteDrives, and nothing drove. The passage turns the body to the door itself,
    // in the ROOM frame (the door's forward heading from the route's vantage vs the body's own), with the facing test the passage
    // already uses: |heading error| < 90 deg.
    constexpr float kPi = std::numbers::pi_v<float>;
    const float door_yaw = -kPi / 2.f;                       // the door is straight toward -y
    // heading -176 deg from the door normal: the policy yaw is door_yaw + 176 deg  =>  theta = yaw - pi/2
    const float yaw_m176 = door_yaw + 176.f * kPi / 180.f;
    const float err_m176 = rc::door_heading_err(yaw_m176 - kPi / 2.f, door_yaw);
    CHECK(std::abs(std::abs(err_m176) - 176.f * kPi / 180.f) < 1e-3f);
    // the e2e20 verdict: route arrived, mouth not seen, anchor stale -> RouteDrives -> face the door
    const auto v_stale = rc::vantage_action(false, true, false, true, true, false, true, false);
    CHECK(v_stale == VantageAction::RouteDrives);
    CHECK(rc::should_face_door(v_stale, true, err_m176));
    // the same with the door SEEN open but the anchor stale (open_seen true, facing false, anchor not fresh)
    CHECK(rc::should_face_door(rc::vantage_action(false, true, true, true, true, false, true, false), true, err_m176));
    // already facing it (within 90 deg either side): nothing to turn
    CHECK(not rc::should_face_door(VantageAction::RouteDrives, true, 0.3f));
    CHECK(not rc::should_face_door(VantageAction::RouteDrives, true, -1.5f));
    // the route has not arrived: the route drives, the body is not at the vantage to turn there
    CHECK(not rc::should_face_door(VantageAction::RouteDrives, false, err_m176));
    // the other actions own the decision: the servo drives, a fresh anchor re-aims, a placed hold stays a hold
    CHECK(not rc::should_face_door(VantageAction::Servo, true, err_m176));
    CHECK(not rc::should_face_door(VantageAction::ReAim, true, err_m176));
    CHECK(not rc::should_face_door(VantageAction::HoldNoMouth, true, err_m176));

    std::println("{}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
