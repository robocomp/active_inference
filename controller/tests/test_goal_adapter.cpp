// Task 8 (local-motion policy): the controller ↔ policy adapter (src/local_policy/goal_adapter.h).
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_goal_adapter.cpp -o /tmp/tga
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tga          (one line)
#include "../src/local_policy/goal_adapter.h"
#include "../src/controller_relative_motion.h"

#include <cmath>
#include <cstdio>
#include <numbers>

using namespace rc::policy;
constexpr float kPi = std::numbers::pi_v<float>;

static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) <= tol; }
static float wrapf(float a) { return std::remainder(a, 2.f * kPi); }

int main()
{
    // ---- 1. the ONE pose conversion: the controller's theta faces theta + pi/2 (GridPlanner / RelativeMotion)
    {
        const ControllerRobotPose r{.pos = {1.5f, -2.f}, .theta = 0.f};
        const Pose2 p = to_policy_pose(r);
        CHECK("pose: theta 0 -> yaw pi/2", near(p.yaw, kPi / 2.f));
        CHECK("pose: position copied", near(p.p.x(), 1.5f) and near(p.p.y(), -2.f));
        // the policy's forward (cos yaw, sin yaw) is the controller's forward (-sin theta, cos theta)
        const ControllerRobotPose r2{.pos = {0.f, 0.f}, .theta = 0.7f};
        const Pose2 p2 = to_policy_pose(r2);
        const Eigen::Vector2f f_pol(std::cos(p2.yaw), std::sin(p2.yaw));
        const Eigen::Vector2f f_ctl = rc::RelativeMotion::forward(r2.theta);
        CHECK("pose: the policy's forward == RelativeMotion::forward(theta)", (f_pol - f_ctl).norm() < 1e-5f);
    }

    // ---- 2. ROT SIGN: a CCW policy w turns the robot CCW in the session's command convention.
    //   The session's send_speed_command(adv, side, rot) is CCW-positive (theta increases): RelativeMotion's "turn +90"
    //   (left, CCW) sends rot > 0, step_orient sends k*(target - theta), and trajectory_controller.cpp's align branch states
    //   it. (The tracker's own out.rot is the opposite, hence the session's `rot = -control_output.rot`; that negation is
    //   the tracker's, not the base's.)
    {
        const auto cmd = to_session_command(0.2f, +0.5f);
        CHECK("sign: policy w +0.5 (CCW) -> session rot > 0", cmd.rot > 0.f);
        CHECK("sign: adv = v, side = 0 (differential)", near(cmd.adv, 0.2f) and cmd.side == 0.f);
        // cross-check against code that drives the live base: RelativeMotion's CCW (left) turn
        rc::RelativeMotion rm;
        const auto steps = rc::RelativeMotion::parse("turn 90");
        const auto free = [](const Eigen::Vector2f&, float) { return true; };
        const auto inside = [](const Eigen::Vector2f&) { return true; };
        CHECK("sign: RelativeMotion accepts 'turn 90'", steps.has_value() and not rm.start(*steps, {0.f, 0.f}, 0.f, 0, free, inside).has_value());
        const auto rcmd = rm.step({0.f, 0.f}, 0.f, 10, free);
        CHECK("sign: RelativeMotion's left (CCW) turn sends rot > 0, same sign as the policy's CCW w", rcmd.rot > 0.f and cmd.rot * rcmd.rot > 0.f);
        // kinematic closure: integrate the session convention (theta += rot dt) and read the policy yaw back
        const float dt = 0.1f;
        ControllerRobotPose r{.pos = {0.f, 0.f}, .theta = 0.3f};
        const float yaw0 = to_policy_pose(r).yaw;
        r.theta += cmd.rot * dt;
        CHECK("sign: the policy yaw advances by +w dt", near(wrapf(to_policy_pose(r).yaw - yaw0), 0.5f * dt, 1e-5f));
        CHECK("sign: a CW policy w gives session rot < 0", to_session_command(0.f, -0.3f).rot < 0.f);
    }

    // ---- 3. Doorway goal from the MEASURED gap (robot frame RelPose) -> room frame, goal points past_m beyond the centre
    {
        const ControllerRobotPose robot{.pos = {1.f, 2.f}, .theta = 0.3f};
        rc::interaction::RelPose gap;
        gap.x = Eigen::Vector3f(0.1f, 1.0f, 0.2f);        // centre 0.1 right, 1.0 ahead; wall yaw 0.2 rad
        const float width = 0.90f, r_ins = 0.23f;
        const RoomGap g = gap_in_room(gap, width, robot);
        // the gap centre in the room: robot.pos + R(theta) (x, y)
        const Eigen::Vector2f c_exp = robot.pos + Eigen::Rotation2Df(robot.theta) * Eigen::Vector2f(0.1f, 1.0f);
        CHECK("doorway: centre in the room frame", (g.centre - c_exp).norm() < 1e-5f);
        const Eigen::Vector2f n_robot(-std::sin(0.2f), std::cos(0.2f));
        CHECK("doorway: normal = R(theta) n_robot (unit)", (g.normal - Eigen::Rotation2Df(robot.theta) * n_robot).norm() < 1e-5f and near(g.normal.norm(), 1.f));
        GoalAdapterParams prm; prm.past_m = 2.0f; prm.inscribed_radius_m = r_ins;
        ControllerTargetInfo t; t.node_name = "aff_door_2_cross"; t.from_affordance = true; t.room_pos = {9.f, 9.f};
        const auto spec = goal_for_target(t, robot, g, GoalMode::Cross, false, prm);
        CHECK("doorway: a cross with a gap gives a goal", spec.has_value());
        if (spec)
        {
            CHECK("doorway: kind Doorway, p = gap centre, normal, width", spec->goal.kind == Goal::Kind::Doorway
                  and (spec->goal.p - g.centre).norm() < 1e-5f and (spec->goal.normal - g.normal).norm() < 1e-5f and near(spec->goal.width, width));
            bool all_past = not spec->points.empty(), lateral_ok = true;
            const Eigen::Vector2f a(g.normal.y(), -g.normal.x());
            for (const auto& q : spec->points)
            {
                all_past = all_past and near((q - g.centre).dot(g.normal), 2.0f, 1e-4f);
                lateral_ok = lateral_ok and std::abs((q - g.centre).dot(a)) <= 0.5f * width - r_ins + 1e-4f;
            }
            CHECK("doorway: every goal point lies 2 m past the gap centre along its normal", all_past);
            CHECK("doorway: goal points span the lateral room the body's centre has (half width - inscribed)", lateral_ok and spec->points.size() > 1);
        }
        // the corridor the doorway field may cross: the far side is NOT in the planner's room map (the door passage corridor
        // ends 2.0 m past the aperture), so the goal line 2.25 m past it would be unreachable without it
        const Eigen::Vector2f a_dir(g.normal.y(), -g.normal.x());
        CHECK("corridor: the goal line is inside it", in_doorway_corridor(g, g.centre + 2.25f * g.normal, 1.0f, 2.3f, r_ins));
        CHECK("corridor: the gap centre is inside it", in_doorway_corridor(g, g.centre, 1.0f, 2.3f, r_ins));
        CHECK("corridor: a point where the body's centre cannot pass (beside a jamb) is not",
              not in_doorway_corridor(g, g.centre + 0.30f * a_dir, 1.0f, 2.3f, r_ins));
        CHECK("corridor: beyond its far end is not", not in_doorway_corridor(g, g.centre + 2.5f * g.normal, 1.0f, 2.3f, r_ins));
        const RoomGap narrow{g.centre, g.normal, 0.40f};   // narrower than the body: no corridor at all
        CHECK("corridor: an opening narrower than the body has none", not in_doorway_corridor(narrow, g.centre, 1.0f, 2.3f, r_ins));
        // identity: a re-measured gap (centre 2 cm off, width 3 cm wider — a different number of seed points) is the same goal
        const RoomGap g2{g.centre + Eigen::Vector2f(0.02f, 0.f), g.normal, width + 0.03f};
        const auto spec2 = goal_for_target(t, robot, g2, GoalMode::Cross, false, prm);
        CHECK("identity: a re-measured gap is the same doorway goal", spec and spec2 and same_goal(*spec, *spec2));
        CHECK("doorway: a cross with no measured gap gives no goal", not goal_for_target(t, robot, std::nullopt, GoalMode::Cross, false, prm).has_value());
    }

    // ---- 4. Heading goal for an orient target; Place goals for a reach (with / without facing)
    {
        const ControllerRobotPose robot{.pos = {0.5f, 0.5f}, .theta = 0.f};
        ControllerTargetInfo t; t.room_pos = {3.f, 1.f}; t.yaw_rad = 0.4f;
        GoalAdapterParams prm;
        const auto o = goal_for_target(t, robot, std::nullopt, GoalMode::Orient, false, prm);
        CHECK("orient: Heading goal", o.has_value() and o->goal.kind == Goal::Kind::Heading);
        // step_orient aligns the controller's theta with target.yaw_rad: the policy yaw is that + pi/2
        CHECK("orient: yaw = target.yaw_rad + pi/2 (step_orient's convention)", o and o->goal.yaw and near(wrapf(*o->goal.yaw - (0.4f + kPi / 2.f)), 0.f));
        CHECK("orient: holds the robot's position", o and (o->goal.p - robot.pos).norm() < 1e-6f and o->points.size() == 1);
        const auto pf = goal_for_target(t, robot, std::nullopt, GoalMode::Place, true, prm);
        // a REACH standpoint's yaw_rad is a FACING yaw (GridPlanner: forward = +x at 0): no quarter turn
        CHECK("place: facing yaw = target.yaw_rad", pf and pf->goal.kind == Goal::Kind::Place and pf->goal.yaw and near(*pf->goal.yaw, 0.4f));
        CHECK("place: p = standpoint", pf and (pf->goal.p - t.room_pos).norm() < 1e-6f);
        const auto pn = goal_for_target(t, robot, std::nullopt, GoalMode::Place, false, prm);
        CHECK("place: no facing -> no yaw", pn and not pn->goal.yaw.has_value());
        const auto lk = lockon_heading_goal(t, robot, prm);
        CHECK("lock-on: Heading toward the object (the standpoint's facing yaw)", lk.goal.kind == Goal::Kind::Heading and lk.goal.yaw and near(*lk.goal.yaw, 0.4f));
    }

    // ---- 5. goal identity: a goal that moved less than its own tolerance is the same goal (no rebuild)
    {
        const auto a = place_goal({1.f, 1.f}, std::nullopt, 0.25f, 0.06f);
        auto b = place_goal({1.1f, 1.f}, std::nullopt, 0.25f, 0.06f);
        CHECK("identity: 0.10 m inside a 0.25 m tolerance is the same goal", same_goal(a, b));
        b = place_goal({1.3f, 1.f}, std::nullopt, 0.25f, 0.06f);
        CHECK("identity: 0.30 m is a different goal", not same_goal(a, b));
        const auto h1 = heading_goal({0.f, 0.f}, 1.0f, 0.06f), h2 = heading_goal({0.f, 0.f}, 1.2f, 0.06f);
        CHECK("identity: a heading change beyond tol_yaw is a different goal", not same_goal(h1, h2));
        CHECK("identity: a different kind is a different goal", not same_goal(a, h1));
    }

    // ---- 6. the commands IN FORCE over a tick, from the send history shifted by the command latency
    {
        // sent: t=0.00 (0.1, 0), t=0.05 (0.2, 0.3), t=0.12 (0, 0)  ; latency 0.06 ; tick [0.05, 0.15]
        const std::vector<SentCommand> sent{{0.00, 0.1f, 0.f}, {0.05, 0.2f, 0.3f}, {0.12, 0.f, 0.f}};
        const auto segs = commands_in_force(sent, 0.15, 0.10f, 0.06f);
        float total = 0.f; for (const auto& s : segs) total += s.dt;
        CHECK("in force: segments sum to dt", near(total, 0.10f, 1e-5f));
        // at the wheels (sent + 0.06): [0.05,0.06) nothing yet (zero); [0.06,0.11) the 0.00 command; [0.11,0.15) the 0.05
        // one; the 0.12 one arrives at 0.18, after the tick
        CHECK("in force: 3 segments", segs.size() == 3);
        if (segs.size() == 3)
        {
            CHECK("in force: first = zero until the first command lands (0.01 s)", near(segs[0].dt, 0.01f, 1e-5f) and segs[0].v == 0.f and segs[0].w == 0.f);
            CHECK("in force: second = the command sent at t 0.00, for 0.05 s", near(segs[1].dt, 0.05f, 1e-5f) and near(segs[1].v, 0.1f) and near(segs[1].w, 0.f));
            CHECK("in force: third = the one sent at t 0.05, for 0.04 s", near(segs[2].dt, 0.04f, 1e-5f) and near(segs[2].v, 0.2f) and near(segs[2].w, 0.3f));
        }
        const auto none = commands_in_force({}, 1.0, 0.1f, 0.06f);
        CHECK("in force: no history -> one zero command for the whole tick", none.size() == 1 and near(none[0].dt, 0.1f) and none[0].v == 0.f);
    }

    // ---- 7. base limits from the capability (deadband from capability, else the provisional config)
    {
        rc::policy::BaseLimitsInputs in;
        in.v_max_policy = 0.5f; in.w_max_policy = 0.8f;
        in.cap_v_max = 1.0f; in.cap_w_max = 2.0f; in.cap_a_lin = 0.5f; in.cap_d_lin = 1.0f; in.axes_length_m = 0.518f;
        in.cap_v_min = std::nullopt; in.cap_w_min = 0.12f; in.cfg_v_min = 0.04f; in.cfg_w_min = 0.10f;
        const BaseLimits b = base_limits(in);
        CHECK("limits: v_max = min(policy, capability)", near(b.v_max, 0.5f) and near(b.w_max, 0.8f));
        CHECK("limits: a_rot = a_lin / (axes/2) (the wheels' acceleration bound)", near(b.a_rot, 0.5f / 0.259f, 1e-4f) and near(b.d_rot, 1.0f / 0.259f, 1e-4f));
        CHECK("limits: deadband — capability when published, the provisional config otherwise", near(b.v_min_eff, 0.04f) and near(b.w_min_eff, 0.12f));
    }

    std::printf(fails == 0 ? "ALL PASS\n" : "FAILURES (%d failing)\n", fails);
    return fails == 0 ? 0 : 1;
}
