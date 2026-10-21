// Task 8 (local-motion policy): the controller ↔ policy adapter (src/local_policy/goal_adapter.h).
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_goal_adapter.cpp -o /tmp/tga
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tga          (one line)
#include "../src/local_policy/goal_adapter.h"
#include "../src/controller_relative_motion.h"
#include "../src/controller_anchor_servo.h"

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
        // I3: the controller places the line at standoff + 2 tol_xy — real slack beyond the protocol's through (s >= standoff):
        // a gap estimate drifting FORWARD by tol_xy/2 moves the line with it, and through still comes first (no deadlock)
        {
            const float standoff = 2.0f, tol = 0.25f;
            GoalAdapterParams q = prm; q.tol_xy = tol; q.past_m = standoff + 2.f * tol;
            const auto s0 = doorway_goal(g, q);
            const RoomGap drifted{g.centre + 0.5f * tol * g.normal, g.normal, g.width};
            const auto s1 = doorway_goal(drifted, q);
            const Eigen::Vector2f through_pt = drifted.centre + standoff * drifted.normal;   // where the protocol completes
            const float line_ahead = (s1.points.front() - through_pt).dot(g.normal);
            CHECK("I3: after a tol/2 forward drift the goal line is still >= 1.5 tol beyond the protocol's through point", line_ahead >= 1.5f * tol - 1e-4f);
            CHECK("I3: a tol/2 drift is the same goal (no rebuild)", same_goal(s0, s1));
        }
        CHECK("doorway: a cross with no measured gap gives no goal", not goal_for_target(t, robot, std::nullopt, GoalMode::Cross, false, prm).has_value());
    }

    // ---- 4. Heading goal for an orient target; Place goals for a reach (with / without facing)
    {
        const ControllerRobotPose robot{.pos = {2.9f, 1.05f}, .theta = 0.f};   // at the standpoint (Live fix 4: orient turns there)
        ControllerTargetInfo t; t.room_pos = {3.f, 1.f}; t.yaw_rad = 0.4f;
        GoalAdapterParams prm;
        const auto o = goal_for_target(t, robot, std::nullopt, GoalMode::Orient, false, prm);
        CHECK("orient: Heading goal", o.has_value() and o->goal.kind == Goal::Kind::Heading);
        // ONE convention (Task 8 ruling 2): target.yaw_rad is the FORWARD heading for orient and reach alike
        CHECK("orient: yaw = target.yaw_rad (the forward heading, as the contract writers fill it)", o and o->goal.yaw and near(wrapf(*o->goal.yaw - 0.4f), 0.f));
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

    // ---- 4b. ONE YAW CONVENTION, pinned on the contract writers' own formula: a "face the door" yaw is
    //   atan2(door - standpoint) (common/nbv/viewpoint_score.h:787/821, door_concept's NBV reach) or the room bearing of the
    //   ray toward the door (door_fitter.cpp:203/212, the Orient hypothesis azimuth). Either way the policy goal must point
    //   AT the door, and a robot whose controller theta faces the door (forward = RelativeMotion::forward(theta)) satisfies it.
    {
        const Eigen::Vector2f stand(1.f, -0.5f), door(3.f, 2.5f);
        const float contract_yaw = std::atan2(door.y() - stand.y(), door.x() - stand.x());
        ControllerTargetInfo t; t.room_pos = stand; t.yaw_rad = contract_yaw;
        const ControllerRobotPose robot{.pos = stand, .theta = 0.f};
        GoalAdapterParams prm;
        const Eigen::Vector2f to_door = (door - stand).normalized();
        for (const GoalMode m : {GoalMode::Orient, GoalMode::Place})
        {
            const auto g = goal_for_target(t, robot, std::nullopt, m, true, prm);
            const bool ok = g and g->goal.yaw and (Eigen::Vector2f(std::cos(*g->goal.yaw), std::sin(*g->goal.yaw)) - to_door).norm() < 1e-5f;
            CHECK(m == GoalMode::Orient ? "one yaw: an orient contract yaw for 'face the door' points the policy AT the door"
                                        : "one yaw: a reach facing yaw for 'face the door' points the policy AT the door", ok);
        }
        // the controller's theta that faces the door, through the ONE conversion, is the goal heading
        const float theta = facing_yaw_to_theta(contract_yaw);
        CHECK("one yaw: facing_yaw_to_theta -> RelativeMotion::forward(theta) points at the door",
              (rc::RelativeMotion::forward(theta) - to_door).norm() < 1e-5f);
        CHECK("one yaw: to_policy_pose(theta) recovers the contract yaw", near(wrapf(to_policy_pose({stand, theta}).yaw - contract_yaw), 0.f));
    }

    // ---- 4c. a continuous ROUTE through the policy: waypoints as successive Place goals (Task 8 ruling 1)
    {
        const std::vector<RouteWaypoint> wps{{{0.f, 0.f}, std::nullopt}, {{2.f, 0.f}, 0.7f}, {{2.f, 2.f}, 1.2f}};
        auto r = route_goal_step(wps, 1, 0, false, 0.25f, 0.06f);
        CHECK("route: index 0, not finished, a Place goal at waypoint 0", not r.finished and r.index == 0 and r.goal
              and r.goal->goal.kind == Goal::Kind::Place and (r.goal->goal.p - wps[0].pos).norm() < 1e-6f and near(r.goal->goal.tol_xy, 0.25f));
        CHECK("route: an intermediate waypoint holds no heading (even if it has one)", r.goal and not r.goal->goal.yaw);
        r = route_goal_step(wps, 1, 0, true, 0.25f, 0.06f);
        CHECK("route: Arrived advances to the next waypoint", r.index == 1 and r.goal and (r.goal->goal.p - wps[1].pos).norm() < 1e-6f
              and not r.goal->goal.yaw);
        r = route_goal_step(wps, 1, 1, true, 0.25f, 0.06f);
        CHECK("route: the FINAL waypoint keeps the route's end heading", r.index == 2 and r.goal and r.goal->goal.yaw and near(*r.goal->goal.yaw, 1.2f));
        r = route_goal_step(wps, 1, 2, true, 0.25f, 0.06f);
        CHECK("route: Arrived on the last one finishes the route (no goal)", r.finished and not r.goal);
        r = route_goal_step(wps, 2, 2, true, 0.25f, 0.06f);
        CHECK("route: with 2 laps the tour starts again at waypoint 0", not r.finished and r.index == 3 and r.goal
              and (r.goal->goal.p - wps[0].pos).norm() < 1e-6f);
        r = route_goal_step(wps, 2, 4, true, 0.25f, 0.06f);
        CHECK("route: the last waypoint of the LAST lap keeps the end heading", r.index == 5 and r.goal and r.goal->goal.yaw and near(*r.goal->goal.yaw, 1.2f));
        CHECK("route: an empty tour is finished at once", route_goal_step({}, 1, 0, false, 0.25f, 0.06f).finished);
    }

    // ---- 4d. I5: odometry turn sign vs the pose's heading change
    {
        OdoSignCheck c;
        const float dt = 0.1f, w = 0.4f, sig = 0.013f * std::sqrt(dt);
        for (int k = 0; k < 10; ++k) c.update(w, w * dt, dt, 0.1f, sig);
        CHECK("I5: consistent turning never contradicts", not c.contradicted());
        c.update(w, -w * dt, dt, 0.1f, sig); c.update(w, -w * dt, dt, 0.1f, sig);
        CHECK("I5: two opposite intervals (a pose jump + snap back) do not contradict", not c.contradicted());
        c.update(w, w * dt, dt, 0.1f, sig);
        c.update(w, -w * dt, dt, 0.1f, sig); c.update(w, 0.f, dt, 0.1f, sig); c.update(w, -w * dt, dt, 0.1f, sig);
        CHECK("I5: an interval the pose cannot resolve is neutral (run continues, not reset)", c.run() == 2 and not c.contradicted());
        c.update(0.05f, -0.05f, dt, 0.1f, sig);
        CHECK("I5: below the deadband nothing is judged", c.run() == 2);
        c.update(w, -w * dt, dt, 0.1f, sig);
        CHECK("I5: a third opposite turning interval in a row contradicts the odometry (latched)", c.contradicted());
        c.update(w, w * dt, dt, 0.1f, sig);
        CHECK("I5: latched until reset", c.contradicted());
    }

    // ---- 4d'. I5 TIME-ALIGNED: the pose's heading change over [t_a, t_b] (its own time base: the tick minus the pose age)
    //   against the odometry integrated over the SAME interval. A turn REVERSAL with a 250 ms pose lag, ramped at the
    //   capability's a_rot, must not trip; a true sign inversion must trip within 3 turning intervals.
    {
        const double tick = 0.05, lag = 0.25; const float a_rot = 0.5f / 0.259f, w0 = 0.5f;
        auto run = [&](float odo_sign, int* trip_after) {
            OdoSignCheck c; std::vector<OdoSample> hist; double yaw_true = 0.0; float w = w0; double prev_pose_t = -1.0; float prev_pose_yaw = 0.f;
            std::vector<std::pair<double, double>> yaw_hist;   // (t, true yaw)
            int turning_intervals = 0; *trip_after = -1;
            for (int k = 0; k < 120; ++k)
            {
                const double t = k * tick;
                const float target = t < 2.0 ? w0 : -w0;                 // the reversal at t = 2 s
                const float step = a_rot * float(tick);
                w = w < target ? std::min(target, w + step) : std::max(target, w - step);
                yaw_true += w * tick;
                yaw_hist.push_back({t, yaw_true});
                hist.push_back({t, odo_sign * w});                     // fresh odometry
                const double tp = t - lag;                             // the pose describes the robot at t - lag
                if (tp < 0.0) continue;
                float yp = 0.f;
                for (const auto& [ts, y] : yaw_hist) if (ts <= tp + 1e-9) yp = float(y);
                if (prev_pose_t >= 0.0)
                {
                    c.update_interval(hist, prev_pose_t, tp, yp - prev_pose_yaw, 0.1f, 0.013f * std::sqrt(float(tp - prev_pose_t)));
                    ++turning_intervals;
                    if (c.contradicted() and *trip_after < 0) *trip_after = turning_intervals;
                }
                prev_pose_t = tp; prev_pose_yaw = yp;
            }
            return c.contradicted();
        };
        int trip = -1;
        CHECK("I5 aligned: a reversal under a 250 ms pose lag does NOT trip", not run(+1.f, &trip));
        CHECK("I5 aligned: a true sign inversion trips", run(-1.f, &trip));
        CHECK("I5 aligned: ...within 3 turning intervals", trip > 0 and trip <= 3);
        // the unaligned comparison (current wheel rate vs the lagged pose) WOULD trip on the same reversal
        OdoSignCheck naive; double yaw_true = 0.0; float w = w0; std::vector<double> yaws;
        for (int k = 0; k < 120; ++k)
        {
            const double t = k * tick; const float target = t < 2.0 ? w0 : -w0; const float stepw = a_rot * float(tick);
            const float w_prev = w; w = w < target ? std::min(target, w + stepw) : std::max(target, w - stepw);
            yaw_true += w * tick; yaws.push_back(yaw_true);
            const int lagk = k - int(lag / tick + 0.5);
            if (lagk >= 1) naive.update(0.5f * (w + w_prev), float(yaws[lagk] - yaws[lagk - 1]), float(tick), 0.1f, 0.013f * std::sqrt(float(tick)));
        }
        // (information, not a check: with this ramp the unaligned comparison sees ~3 contradicting intervals — at the edge)
        std::printf("   I5: the unaligned comparison on the same reversal: run %d, %s\n", naive.run(), naive.contradicted() ? "TRIPS" : "does not trip");
    }

    // ---- 4f. review fix 2, item 1: a policy-tick HOLD is an interruption (PolicyTickGate)
    {
        PolicyTickGate gate;
        CHECK("gate: the first tick restarts", gate.on_tick(10.0));
        CHECK("gate: a tick 50 ms later continues", not gate.on_tick(10.05));
        gate.on_hold();                                               // a 0.1 s odometry-stale hold
        CHECK("gate: the tick after a 0.1 s hold interrupts (well under the 0.5 s gap rule)", gate.on_tick(10.15));
        CHECK("gate: and the next one continues", not gate.on_tick(10.2));
        CHECK("gate: a 0.6 s gap with no hold noted still interrupts", gate.on_tick(10.8));
    }

    // ---- 4h. Live fix 2 ruling 2: does a trap disc meet the measured doorway's corridor? (only an opening the body fits)
    {
        const RoomGap g{{0.f, 0.f}, {0.f, 1.f}, 0.80f};
        CHECK("trap/corridor: a disc in the doorway mouth meets it", disc_meets_doorway_corridor(g, {0.f, -0.3f}, 0.32f, 1.0f, 2.55f, 0.23f, 0.568f));
        CHECK("trap/corridor: a disc 1 m to the side does not", not disc_meets_doorway_corridor(g, {1.2f, -0.3f}, 0.32f, 1.0f, 2.55f, 0.23f, 0.568f));
        CHECK("trap/corridor: a disc touching the corridor's edge does", disc_meets_doorway_corridor(g, {0.17f + 0.30f, 0.5f}, 0.32f, 1.0f, 2.55f, 0.23f, 0.568f));
        const RoomGap narrow{{0.f, 0.f}, {0.f, 1.f}, 0.50f};   // narrower than the body: no evidence that it fits
        CHECK("trap/corridor: an opening narrower than the body vetoes nothing", not disc_meets_doorway_corridor(narrow, {0.f, -0.3f}, 0.32f, 1.0f, 2.55f, 0.23f, 0.568f));
    }

    // ---- 4i. Live fix 4 ruling 1: an ORIENT with a standpoint goes TO it first (Place WITH the contract's yaw) when the robot is
    //   farther than the arrival tolerance; only there does it turn in place (Heading). tourP8: aff_door_2_open at (0.51,
    //   0.86) yaw +1.57 was turned to at (-0.64, 1.89), 58 deg off the door normal.
    {
        ControllerTargetInfo t; t.room_pos = {0.51f, 0.86f}; t.yaw_rad = 1.5708f;
        GoalAdapterParams prm; prm.tol_xy = 0.25f;
        const ControllerRobotPose far{.pos = {-0.64f, 1.89f}, .theta = 0.f};
        const auto g1 = goal_for_target(t, far, std::nullopt, GoalMode::Orient, false, prm);
        CHECK("orient far: a Place goal at the standpoint WITH the contract's yaw", g1 and g1->goal.kind == Goal::Kind::Place
              and (g1->goal.p - t.room_pos).norm() < 1e-6f and g1->goal.yaw and near(*g1->goal.yaw, 1.5708f));
        const ControllerRobotPose there{.pos = {0.55f, 0.80f}, .theta = 0.f};
        const auto g2 = goal_for_target(t, there, std::nullopt, GoalMode::Orient, false, prm);
        CHECK("orient there: a Heading goal, turning in place", g2 and g2->goal.kind == Goal::Kind::Heading and near(*g2->goal.yaw, 1.5708f)
              and (g2->goal.p - there.pos).norm() < 1e-6f);
    }
    // ---- 4j. Live fix 4 ruling 4: a stale-anchor wait that has ARRIVED (facing where the anchor was published) and stays
    //   unpublished for one plan duration hands the cross back so the passage repositions (back to approach).
    {
        StaleAnchorWait w;
        CHECK("stale wait: not arrived -> no hand back", not w.on_tick(false, 100.0, 2.0));
        CHECK("stale wait: arrived just now -> waits", not w.on_tick(true, 101.0, 2.0));
        CHECK("stale wait: arrived 1.5 s -> still waits (plan 2.0 s)", not w.on_tick(true, 102.5, 2.0));
        CHECK("stale wait: arrived for a plan duration -> hand back", w.on_tick(true, 103.05, 2.0));
        w.reset();
        CHECK("stale wait: a turn in between restarts the clock", not w.on_tick(true, 200.0, 2.0) and not w.on_tick(false, 201.0, 2.0)
              and not w.on_tick(true, 202.5, 2.0));
    }

    // ---- 4g. Live fix 2 (tourP3 retry): a ROOM-FRAME CHANGE (room_concept moved the robot's RT into the next room's frame,
    //   pose (0.60,2.57) -> (-3.64,0.41) in one cycle) resets the policy, and it drives again only on returns registered
    //   in the NEW frame — the contact buffer still held sweeps registered in the old one ("9 returns inside the body").
    {
        RoomFrameGuard g;
        CHECK("frame: the first room latches without a reset", not g.on_room("room_1", 1000));
        CHECK("frame: ready once a scan exists", g.ready(900));
        CHECK("frame: the same room is no change", not g.on_room("room_1", 1050));
        CHECK("frame: a new room is a change (reset)", g.on_room("room_2", 1100));
        CHECK("frame: a scan accepted BEFORE the switch is not ready (old-frame registration)", not g.ready(1080));
        CHECK("frame: ...nor one accepted at the switch instant", not g.ready(1100));
        CHECK("frame: a scan accepted after the switch is", g.ready(1150));
        CHECK("frame: an empty room name (not yet known) is no change", not g.on_room("", 1200));
    }

    // ---- 4e. C1: the relative-motion path on a pose that MOVES between ticks (the session passes the fresh pose)
    {
        rc::RelativeMotion rm;
        const auto free = [](const Eigen::Vector2f&, float) { return true; };
        const auto inside = [](const Eigen::Vector2f&) { return true; };
        Eigen::Vector2f pos(1.f, 1.f); float theta = 0.3f;
        CHECK("C1: move 0.5 accepted", not rm.start(*rc::RelativeMotion::parse("move 0.5"), pos, theta, 0, free, inside).has_value());
        const auto g0 = rm.policy_goal();
        const Eigen::Vector2f fwd = rc::RelativeMotion::forward(theta);
        CHECK("C1: the move's goal is 0.5 m ahead, facing the start heading (to_policy_pose of it)", g0 and not g0->turn
              and (g0->pos - (pos + 0.5f * fwd)).norm() < 1e-5f
              and near(wrapf(to_policy_pose({g0->pos, g0->theta}).yaw - to_policy_pose({pos, theta}).yaw), 0.f));
        int ticks = 0;
        for (; ticks < 40 and rm.active(); ++ticks)
        {
            pos += 0.05f * fwd;                                       // the robot moves between ticks
            rm.policy_step(pos, theta, static_cast<std::uint64_t>(100 * (ticks + 1)), std::nullopt);
            if (rm.active()) CHECK("C1: the goal stays where it was set while the robot moves", (rm.policy_goal()->pos - g0->pos).norm() < 1e-5f);
        }
        CHECK("C1: the step completes on the MOVED pose (within the move's 3 cm)", rm.view().phase == rc::RelativeMotion::Phase::Done and ticks <= 10);
        // a 270-degree left turn on a turning pose: quarter-turn marks, never the short way round
        rc::RelativeMotion rt; theta = 0.f;
        CHECK("C1: turn 270 accepted", not rt.start(*rc::RelativeMotion::parse("turn 270"), pos, theta, 0, free, inside).has_value());
        float last_goal = theta; bool monotone = true;
        for (int k = 0; k < 200 and rt.active(); ++k)
        {
            const auto g = rt.policy_goal();
            monotone = monotone and g->theta >= last_goal - 1e-4f; last_goal = g->theta;
            theta += 0.05f;                                           // turning CCW as asked
            rt.policy_step(pos, std::remainder(theta, 2.f * kPi), static_cast<std::uint64_t>(100 * (k + 1)), std::nullopt);
        }
        CHECK("C1: a 270 deg turn completes with goals that only advance CCW", rt.view().phase == rc::RelativeMotion::Phase::Done and monotone);
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
        CHECK("limits: the axle length for the wheel-coupled rollout (Live fix 2)", near(b.track_m, 0.518f));
        in.measured_wheel_decel = 0.5f;
        const BaseLimits bm = base_limits(in);
        CHECK("limits: the measured wheel decel overrides the capability's (d_lin 0.5, d_rot 0.5/0.259)",
              near(bm.d_lin, 0.5f) and near(bm.d_rot, 0.5f / 0.259f, 1e-4f));
    }

    // ---- 8. Task 3b (rtD1c leg 5): the cross's GOAL LINE and TRAP VETO from the anchor's opening, the gap weighted by its
    //         score (doorway_corridor.h expected_opening) — a 0.03-score phantom locked on the leaf must not set the goal line
    {
        GoalAdapterParams gp; gp.past_m = 2.5f; gp.inscribed_radius_m = 0.23f;
        const RoomGap anchor{Eigen::Vector2f(0.f, 0.f), Eigen::Vector2f(0.f, 1.f), 0.87f};
        // the leaf + the west jamb seen obliquely: 0.62 m (wide enough to veto on its own), its normal 120 deg off the door's
        const Eigen::Vector2f ng(-std::sin(2.f * kPi / 3.f), std::cos(2.f * kPi / 3.f));
        const RoomGap phantom{Eigen::Vector2f(-0.45f, -0.6f), ng, 0.62f};
        const float score = 0.03f;
        const auto centroid = [](const GoalSpec& g) { Eigen::Vector2f c = Eigen::Vector2f::Zero(); for (const auto& q : g.points) c += q; return Eigen::Vector2f(c / float(g.points.size())); };
        const GoalSpec ga = doorway_goal(anchor, gp), gph = doorway_goal(phantom, gp);
        const GoalSpec gm = doorway_goal(expected_opening(anchor, phantom, score), gp);
        const float d_ph = (centroid(gph) - centroid(ga)).norm(), d_m = (centroid(gm) - centroid(ga)).norm();
        std::printf("   goal line centroid: phantom's %.2f m from the anchor's; the score-weighted one %.3f m\n", d_ph, d_m);
        CHECK("3b goal line: a score-0.03 phantom moves it by at most ~its weight x the distance (<= 1.5 x 0.03 x d)", d_m <= 1.5f * score * d_ph);
        bool past_anchor = true;
        for (const auto& q : gm.points)
            past_anchor = past_anchor and std::abs(q.y() - 2.5f) < 0.1f and std::abs(q.x()) <= 0.5f * 0.87f - 0.23f + 0.1f;
        CHECK("3b goal line: every seed point is past the ANCHOR's opening, inside its lateral span", past_anchor);
        CHECK("3b goal line: no gap / score 0 = the anchor's exactly",
              (centroid(doorway_goal(expected_opening(anchor, std::nullopt, 0.9f), gp)) - centroid(ga)).norm() < 1e-6f
              and (centroid(doorway_goal(expected_opening(anchor, phantom, 0.f), gp)) - centroid(ga)).norm() < 1e-6f);
        const RoomGap good{Eigen::Vector2f(0.08f, 0.02f), Eigen::Vector2f(0.f, -1.f), 0.85f};   // a reliable gap; opposite sign convention
        const RoomGap og = expected_opening(anchor, good, 1.f);
        CHECK("3b goal line: a score-1 gap IS the geometry (its normal taken in the anchor's sense)",
              (og.centre - good.centre).norm() < 1e-6f and (og.normal - Eigen::Vector2f(0.f, 1.f)).norm() < 1e-6f and near(og.width, 0.85f));
        // the trap veto on the same opening
        const float body_w = 0.568f, back = 1.0f, ahead = gp.past_m + 0.05f;
        const Eigen::Vector2f on_phantom = phantom.centre + 0.5f * phantom.normal;   // in the phantom's strip, off the door
        const Eigen::Vector2f on_door(0.f, -0.3f);
        const RoomGap om = expected_opening(anchor, phantom, score);
        CHECK("3b veto: (baseline) on the phantom alone a trap in its strip would be vetoed",
              disc_meets_doorway_corridor(phantom, on_phantom, 0.2f, back, ahead, 0.23f, body_w));
        CHECK("3b veto: score-weighted, a trap in the phantom's strip (on the leaf) is NOT vetoed",
              not disc_meets_doorway_corridor(om, on_phantom, 0.2f, back, ahead, 0.23f, body_w));
        CHECK("3b veto: a trap in the door's own corridor still is",
              disc_meets_doorway_corridor(om, on_door, 0.2f, back, ahead, 0.23f, body_w));
    }

    // ---- 9. Task 3b: the leaf hypotheses in the ROOM frame — the same points and covariances anchor_leaf_points gives in the
    //         robot frame, rotated by the robot's pose
    {
        const ControllerRobotPose robot{.pos = {1.0f, 2.0f}, .theta = 0.3f};
        rc::interaction::RelPose a;
        a.x = Eigen::Vector3f(0.2f, 1.5f, 0.1f);
        a.P = Eigen::Matrix3f::Zero(); a.P(0, 0) = 0.01f; a.P(1, 1) = 0.02f; a.P(0, 1) = a.P(1, 0) = 0.004f; a.P(2, 2) = 0.003f;
        auto l = rc::interaction::make_anchor_leaf(-0.435f, -0.06f, 1.f, -1.f, 0.82f, 1.4f, 0.12f);
        l.p = 0.7f;
        const std::vector<rc::interaction::AnchorLeaf> ls{l};
        const DoorLeaf dl = door_leaf_in_room(a, ls, robot);
        const auto pts = rc::anchor_leaf_points(a, l);
        const Eigen::Matrix2f R = Eigen::Rotation2Df(robot.theta).toRotationMatrix();
        bool same = dl.hyps.size() == 1 and not pts.empty();
        float worst_p = 0.f, worst_c = 0.f;
        for (std::size_t i = 0; same and i < pts.size(); ++i)
        {
            const float r = dl.hyps[0].length * float(i) / float(pts.size() - 1);
            const Eigen::Vector2f q_room = dl.hyps[0].hinge + r * dl.hyps[0].dir, q_ref = robot.pos + R * pts[i].mean;
            const Eigen::Matrix2f c_ref = R * pts[i].cov * R.transpose();
            worst_p = std::max(worst_p, (q_room - q_ref).norm());
            worst_c = std::max(worst_c, (dl.hyps[0].point_cov(r) - c_ref).cwiseAbs().maxCoeff());
        }
        std::printf("   leaf in the room frame vs anchor_leaf_points: worst point %.2e m, worst covariance entry %.2e m^2\n", worst_p, worst_c);
        CHECK("3b leaf: the room-frame centre line = anchor_leaf_points' means (rotated by the robot pose)", same and worst_p < 1e-5f);
        CHECK("3b leaf: and its covariance = theirs (P_xy + sigma_psi^2 g g^T + r^2 sigma_phi^2 t t^T)", same and worst_c < 1e-6f);
        CHECK("3b leaf: probability and half-thickness carried", same and near(dl.hyps[0].p, 0.7f) and near(dl.hyps[0].half_thick, l.half_thick));
        CHECK("3b leaf: no leaves -> empty", door_leaf_in_room(a, {}, robot).empty());
    }

    std::printf(fails == 0 ? "ALL PASS\n" : "FAILURES (%d failing)\n", fails);
    return fails == 0 ? 0 : 1;
}
