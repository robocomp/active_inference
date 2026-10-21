// goal_adapter.h — where the controller's types meet the local policy's (Task 8). Pure (Eigen + std + the two type
// headers), so every conversion here is unit-tested (tests/test_goal_adapter.cpp) and nowhere else re-derives one.
//
// ★ THE ONLY POSE CONVERSION. The controller's ControllerRobotPose::theta is the body angle whose robot frame has x right,
//   y forward: the robot FACES theta + pi/2 (GridPlanner, RelativeMotion::forward = (-sin theta, cos theta)). The policy's
//   Pose2::yaw is the forward heading itself (forward = (cos yaw, sin yaw)). to_policy_pose adds the quarter turn; nothing
//   else in the controller may.
// ★ THE ROT SIGN. The session's send_speed_command(adv, side, rot) is CCW-positive — a positive rot increases theta
//   (RelativeMotion's "turn +90" = left sends rot > 0; step_orient sends k (target - theta); trajectory_controller.cpp's
//   align branch says so). The policy's w is CCW-positive about the forward heading, and d(yaw)/dt = d(theta)/dt, so the
//   session rot is +w. (The session's `rot = -control_output.rot` at the tracker's send site negates the TRACKER's own
//   clockwise-positive out.rot — plain_tracker.cpp "FRAME conversion (2/2)" — not the base's convention.)
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include "policy_types.h"
#include "local_policy.h"
#include "unseen_contact.h"
#include "policy_tick_gate.h"
#include "doorway_corridor.h"
#include "door_leaf.h"
#include "../controller_runtime_types.h"
#include "../../../common/interaction_anchor/relative_pose.h"
#include "../../../common/interaction_anchor/anchor_leaf.h"

namespace rc::policy
{
inline Pose2 to_policy_pose(const ControllerRobotPose& r)
{
    return {r.pos, r.theta + std::numbers::pi_v<float> / 2.f};
}
// ★ ONE YAW CONVENTION (Task 8 ruling 2). An affordance's target yaw (epistemic_target_yaw_rad -> ControllerTargetInfo::
//   yaw_rad) is the FORWARD heading, room frame, +x at 0, CCW — for Reach and Orient alike. The writers: the NBV reach
//   pose, common/nbv/viewpoint_score.h:787 / :821 `yaw = atan2(t.cy - pos.y, t.cx - pos.x)` ("look back at the object
//   centre"), published by door_concept/src/door_scene_graph.cpp:530 (and every concept's scene graph); the Orient
//   hypothesis, door_concept/src/specificworker.cpp:3034 `epistemic_target_yaw_rad = inst.hypothesis_azimuth`, the room
//   bearing of the ray door_fitter.cpp:203/212 seeds as wall_u = (cos az, sin az). So it IS the policy's yaw; the
//   controller's body angle that faces it is yaw - pi/2 (the inverse of to_policy_pose).
inline float facing_yaw_to_theta(float yaw) { return yaw - std::numbers::pi_v<float> / 2.f; }

struct SessionCommand { float adv = 0.f, side = 0.f, rot = 0.f; };
// A differential base: side is always 0 (Shadow's bridge drops it). rot = +w (see the header note).
inline SessionCommand to_session_command(float v, float w) { return {v, 0.f, w}; }

inline RoomGap gap_in_room(const rc::interaction::RelPose& gap_robot, float width, const ControllerRobotPose& robot)
{
    const Eigen::Rotation2Df R(robot.theta);
    const float psi = gap_robot.x.z();
    RoomGap g;
    g.centre = robot.pos + R * Eigen::Vector2f(gap_robot.x.x(), gap_robot.x.y());
    g.normal = (R * Eigen::Vector2f(-std::sin(psi), std::cos(psi))).normalized();
    g.width = width;
    return g;
}

// ★ Task 3b: door_concept's leaf hypotheses (anchor frame, anchor_leaf.h) in the ROOM frame, with what door_leaf.h needs to
//   give every centre-line point anchor_leaf_points' covariance there: the anchor's centre (ψ rotates about it), its position
//   covariance rotated into the room, its yaw variance, the leaf angle's. Empty with no leaf or a non-finite anchor.
inline DoorLeaf door_leaf_in_room(const rc::interaction::RelPose& anchor_robot, std::span<const rc::interaction::AnchorLeaf> leaves,
                                  const ControllerRobotPose& robot)
{
    DoorLeaf out;
    if (leaves.empty() or not anchor_robot.x.allFinite() or not anchor_robot.P.allFinite()) return out;
    const Eigen::Rotation2Df Rr(robot.theta), Ra(anchor_robot.x.z());
    const Eigen::Matrix2f R = Rr.toRotationMatrix(), RA = R * Ra.toRotationMatrix();
    const Eigen::Vector2f A = robot.pos + R * anchor_robot.x.head<2>();
    const Eigen::Matrix2f P = R * anchor_robot.P.topLeftCorner<2, 2>() * R.transpose();
    for (const auto& l : leaves)
    {
        const float len = l.length();
        if (not (len > 0.f) or not (l.p > 0.f)) continue;
        LeafHypothesis h;
        h.hinge = A + RA * l.hinge;
        h.dir = RA * ((l.tip - l.hinge) / len);
        h.length = len;
        h.half_thick = l.half_thick;
        h.p = l.p;
        h.pivot = A;
        h.P_xy = P;
        h.var_psi = std::max(0.f, anchor_robot.P(2, 2));
        h.var_phi = l.sigma_phi * l.sigma_phi;
        out.hyps.push_back(h);
    }
    return out;
}

// A goal and the points its cost-to-go is seeded from.
struct GoalSpec { Goal goal{.kind = Goal::Kind::Place, .yaw = {}}; std::vector<Eigen::Vector2f> points; };

struct GoalAdapterParams
{
    float tol_xy = 0.25f;            // the controller's goal_threshold (Place arrival)
    float tol_yaw = 0.06f;           // the tracker's align_yaw_tol_rad (facing arrival)
    float past_m = 2.0f;             // a cross's goal line, this far past the gap centre along its normal
    float inscribed_radius_m = 0.23f;   // the body's: where its CENTRE can be across an opening
    float lateral_step_m = 0.05f;    // spacing of the goal line's seed points (the field's own cell)
};

inline GoalSpec place_goal(const Eigen::Vector2f& p, std::optional<float> yaw, float tol_xy, float tol_yaw)
{
    GoalSpec s;
    s.goal = Goal{.kind = Goal::Kind::Place, .p = p, .yaw = yaw, .tol_xy = tol_xy, .tol_yaw = tol_yaw};
    s.points = {p};
    return s;
}
// "Turn to face yaw, here": Heading goals hold position (round-4 ruling); the field is seeded at the robot.
// ★ CONTRACT: a Heading goal's p IS its hold position (`here`) — decide() charges displacement from it as negative progress
//   and, in a Release, picks the end pose closest to it (local_policy.cpp, Live fix 1). It always carries a yaw: a
//   Heading goal WITHOUT one reads p as the point to face (local_policy.h), which this helper never builds.
inline GoalSpec heading_goal(const Eigen::Vector2f& here, float yaw, float tol_yaw)
{
    GoalSpec s;
    s.goal = Goal{.kind = Goal::Kind::Heading, .p = here, .yaw = yaw, .tol_yaw = tol_yaw};
    s.points = {here};
    return s;
}
// "Distance to being through": a line across the opening past_m beyond its centre along the normal, over the lateral span
// the body's centre can occupy (half width minus the inscribed radius; the centre point when the opening has none).
inline GoalSpec doorway_goal(const RoomGap& g, const GoalAdapterParams& p)
{
    GoalSpec s;
    s.goal = Goal{.kind = Goal::Kind::Doorway, .p = g.centre, .yaw = {}, .normal = g.normal, .width = g.width, .tol_xy = p.tol_xy, .tol_yaw = p.tol_yaw};
    const Eigen::Vector2f lat(-g.normal.y(), g.normal.x());
    const float half = std::max(0.f, 0.5f * g.width - p.inscribed_radius_m);
    const float step = std::max(1e-3f, p.lateral_step_m);
    for (float t = -half; t <= half + 1e-4f; t += step) s.points.push_back(g.centre + p.past_m * g.normal + t * lat);
    if (s.points.empty()) s.points.push_back(g.centre + p.past_m * g.normal);
    return s;
}

// ★ The cost-to-go's FREE test on the planner's map (review I4: driving is gated on this field, not on the A*): the body's
//   CENTRE can be at p at some heading iff the planner's distance to the nearest occupied cell exceeds the BARE body's
//   inscribed radius. The A*'s footprint test (hull + FootprintSafetyMargin at 8 headings) is stricter on purpose — a
//   route preference — and must not stop a body that fits from driving.
//   distance_m is to the nearest occupied cell's CENTRE; the cell spans half a cell around it, so the obstacle's surface may be
//   cell/2 closer: free iff distance - cell/2 > inscribed radius.
inline bool field_free_from_distance(float distance_m, float cell_m, float inscribed_radius_m)
{
    return distance_m - 0.5f * cell_m > inscribed_radius_m;
}

// What the task layer is asking for.
//   Place  — a reach standpoint / click target / relative move: p = target.room_pos; yaw = target.yaw_rad when the contract
//            designed a facing (want_facing). A standpoint's yaw_rad is a FACING yaw (forward = +x at 0, GridPlanner's
//            convention), so no quarter turn.
//   Cross  — the doorway from the MEASURED gap (nullopt without one: nothing measured to drive through).
//   Orient — go to the standpoint (Place with yaw) while farther than tol_xy, then turn in place to target.yaw_rad (the
//            same forward-heading convention, facing_yaw_to_theta above).
enum class GoalMode { Place, Cross, Orient };
inline std::optional<GoalSpec> goal_for_target(const ControllerTargetInfo& t, const ControllerRobotPose& robot,
                                               const std::optional<RoomGap>& gap, GoalMode mode, bool want_facing,
                                               const GoalAdapterParams& p)
{
    switch (mode)
    {
        case GoalMode::Cross:
            if (not gap.has_value() or not (gap->width > 0.f)) return std::nullopt;
            return doorway_goal(*gap, p);
        case GoalMode::Orient:
            // ★ Live fix 4 ruling 1: an orient GOES TO its standpoint first — a Place WITH the contract's yaw while the robot
            //   is farther than the arrival tolerance from it; only there does it turn in place (tourP8: turned to +1.57 at a
            //   spot 1.2 m from the open standpoint, 58 deg off the door normal).
            if ((robot.pos - t.room_pos).norm() > p.tol_xy) return place_goal(t.room_pos, t.yaw_rad, p.tol_xy, p.tol_yaw);
            return heading_goal(robot.pos, t.yaw_rad, p.tol_yaw);
        case GoalMode::Place:
            return place_goal(t.room_pos, want_facing ? std::optional<float>(t.yaw_rad) : std::nullopt, p.tol_xy, p.tol_yaw);
    }
    return std::nullopt;
}
// The lock-on micro-search's MOTION as a goal: face the object from where the robot stands. The standpoint's facing yaw
// was computed toward the object (atan2 object - standpoint, build_planning_step), so it is the heading to hold.
inline GoalSpec lockon_heading_goal(const ControllerTargetInfo& t, const ControllerRobotPose& robot, const GoalAdapterParams& p)
{
    return heading_goal(robot.pos, t.yaw_rad, p.tol_yaw);
}

// ★ A CONTINUOUS ROUTE THROUGH THE POLICY (Task 8 ruling 1): the tour's waypoints (authored order, `laps` times) are
//   successive Place goals — tol_xy as for a place, the next one on Arrived (or when the caller skips one it cannot
//   reach); only the FINAL waypoint of the last lap keeps the route's end heading, if it has one (a tour does not stop to
//   face at every corner). index = the waypoint being driven to, counted across laps; finished = the tour is done.
struct RouteWaypoint { Eigen::Vector2f pos = Eigen::Vector2f::Zero(); std::optional<float> yaw; };
struct RouteGoalStep { int index = 0; bool finished = false; std::optional<GoalSpec> goal; };
inline RouteGoalStep route_goal_step(std::span<const RouteWaypoint> wps, int laps, int index, bool advance, float tol_xy, float tol_yaw)
{
    RouteGoalStep r;
    const int n = static_cast<int>(wps.size()), total = n * std::max(1, laps);
    r.index = std::max(0, index) + (advance ? 1 : 0);
    if (n == 0 or r.index >= total) { r.finished = true; r.index = std::min(r.index, total); return r; }
    const RouteWaypoint& w = wps[static_cast<std::size_t>(r.index % n)];
    r.goal = place_goal(w.pos, r.index == total - 1 ? w.yaw : std::nullopt, tol_xy, tol_yaw);
    return r;
}

// ★ Task 8 ruling I5 — IS THE ODOMETRY'S TURN SIGN THE POSE'S? The policy rolls out from the measured w and UnseenContact
//   predicts from it: a flipped robot_current_angular_speed would drive on a mirrored world. Each control interval in which
//   the base turns (|w| >= w_min_eff, the deadband) and the localised heading changed by more than its own 2 sigma (sigma =
//   the relative yaw sigma over the interval — below that the pose carries no sign) is evidence: agreeing signs reset the
//   run, opposite signs extend it. kConsecutive = 3 opposite intervals in a row CONTRADICT the odometry. Why 3: a single
//   relocalisation step of the pose touches at most TWO intervals (the jump, and the one after if it snaps back), so no
//   single pose event can trip it, while a truly flipped sign contradicts every turning interval. Latched until reset.
struct OdoSample { double t = 0.0; float w = 0.f, v = 0.f; };   // odometry turn rate + speed and the time they describe
// ★ Task 3d (rtD1e leg 5) — THE TURN SIGN IS A BELIEF, NOT A LATCH. The old check latched "contradicted" after 3 opposite
//   intervals in a row and was cleared only by policy_reset(), which selector-idle and hand-backs never call: a relocaliser
//   correction smoothed over >= 3 control intervals (439.98-440.17: odometry turning right, the pose's yaw RISING and x/y
//   jumping 0.25 m in 0.19 s) left the policy Blind for the rest of the run.
//   Two hidden states: S = the odometry's turn sign is flipped (a persistent fault), not-S. Per interval, with the odometry's
//   integrated turn theta and the pose's heading change dyaw (sigma_yaw: the relative heading sigma over the interval):
//     log N(dyaw; -theta, sigma_yaw) - log N(dyaw; +theta, sigma_yaw) = -2 dyaw theta / sigma_yaw^2,
//   weighted by the probability that the interval is NOT a pose correction. That comes from the TRANSLATION residual
//   e = |dp_pose| - |odometry displacement|: under no correction e ~ N(0, sigma_p^2) (the odometry's own displacement sigma),
//   under a correction e ~ N(0, sigma_p^2 + sigma_corr^2) (sigma_corr: the size of a correction the localiser can make — its
//   published xy std, else the pose's drift sigma), equal priors — the residual decides. An interval in which the pose moved
//   more than the odometry predicts plus that covariance is explained by the correction and carries ~no evidence for S.
//   Folded as a two-state chain: S switches with probability kSwitch per interval, which is also its prior — so agreeing
//   intervals decay it, nothing latches, and a genuine fault is detected again whenever it appears.
//   PROVISIONAL, stated (not fitted): kSwitch = 0.01 (the same order as UnseenContact::kPriorBlocked's base rate).
//   Blind iff P(S) > 1/2.
class TurnSignBelief
{
public:
    static constexpr float kSwitch = 0.01f;
    // theta_odo: odometry's integrated turn (rad); disp_odo: its integrated displacement (m); dyaw_pose, dp_pose: the pose's
    // heading change (rad) and displacement (m) over the same interval
    void update(float theta_odo, float disp_odo, float dyaw_pose, float dp_pose, float sigma_yaw, float sigma_p, float sigma_corr)
    {
        // the chain's switch step (also the prior on a fresh belief)
        p_ = p_ * (1.f - kSwitch) + (1.f - p_) * kSwitch;
        if (not (sigma_yaw > 0.f)) return;
        const float e = std::abs(dp_pose) - std::abs(disp_odo);
        const float sn2 = std::max(sigma_p * sigma_p, 1e-12f), sc2 = sn2 + sigma_corr * sigma_corr;
        // P(correction | e), equal priors: N(e; 0, sc2) / (N(e; 0, sc2) + N(e; 0, sn2)) in log space
        const float lc = -0.5f * e * e / sc2 - 0.5f * std::log(sc2), ln = -0.5f * e * e / sn2 - 0.5f * std::log(sn2);
        const float p_corr = 1.f / (1.f + std::exp(ln - lc));
        last_llr_ = (1.f - p_corr) * (-2.f * dyaw_pose * theta_odo / (sigma_yaw * sigma_yaw));
        last_p_corr_ = p_corr;
        const float L = std::log(p_ / (1.f - p_)) + last_llr_;
        p_ = L > 30.f ? 1.f - 1e-13f : L < -30.f ? 1e-13f : 1.f / (1.f + std::exp(-L));
    }
    // TIME-ALIGNED (review fix 2.2): the pose's change over [t_a, t_b] in the POSE's time base against the odometry INTEGRATED
    // over the same interval (samples held until the next). An interval the history does not cover judges nothing.
    void update_interval(std::span<const OdoSample> hist, double t_a, double t_b, float dyaw_pose, float dp_pose,
                         float sigma_yaw, float sigma_p, float sigma_corr)
    {
        const double dt = t_b - t_a;
        if (not (dt > 0.0) or hist.empty() or hist.front().t > t_a + 1e-9) return;
        double turn = 0.0, disp = 0.0;
        for (std::size_t i = 0; i < hist.size(); ++i)
        {
            const double s0 = hist[i].t, s1 = i + 1 < hist.size() ? hist[i + 1].t : t_b;
            const double lo = std::max(s0, t_a), hi = std::min(s1, t_b);
            if (hi > lo) { turn += double(hist[i].w) * (hi - lo); disp += double(hist[i].v) * (hi - lo); }
        }
        update(float(turn), float(disp), dyaw_pose, dp_pose, sigma_yaw, sigma_p, sigma_corr);
    }
    [[nodiscard]] bool flipped() const { return p_ > 0.5f; }
    [[nodiscard]] float p_flipped() const { return p_; }
    [[nodiscard]] float last_llr() const { return last_llr_; }
    [[nodiscard]] float last_p_correction() const { return last_p_corr_; }
    void reset() { *this = TurnSignBelief{}; }
private:
    float p_ = kSwitch, last_llr_ = 0.f, last_p_corr_ = 0.f;
};

// ★ GOAL IDENTITY — a goal that moved by less than its OWN tolerance is the same goal (its meaning of "there" did not
//   change), so the Navigator keeps its plan, traps and monitor; anything else is a new goal (set_goal). A measured gap
//   re-estimated every cycle moves by centimetres; rebuilding on each would reset the plan commitment every tick.
inline bool same_goal(const GoalSpec& a, const GoalSpec& b)
{
    if (a.goal.kind != b.goal.kind or a.points.empty() != b.points.empty()) return false;
    const float tol = std::min(a.goal.tol_xy, b.goal.tol_xy);
    // a goal LINE (a doorway's) is compared by its two ends: a re-measured width changes how many seed points it has, not
    // where it is; a point goal by its point
    if (a.points.size() != b.points.size() and a.goal.kind != Goal::Kind::Doorway) return false;
    if (not a.points.empty()
        and ((a.points.front() - b.points.front()).norm() > tol or (a.points.back() - b.points.back()).norm() > tol)) return false;
    if (a.points.size() == b.points.size())
        for (std::size_t i = 0; i < a.points.size(); ++i)
            if ((a.points[i] - b.points[i]).norm() > tol) return false;
    if (a.goal.kind != Goal::Kind::Heading and (a.goal.p - b.goal.p).norm() > tol) return false;
    if (a.goal.yaw.has_value() != b.goal.yaw.has_value()) return false;
    if (a.goal.yaw and std::abs(std::remainder(*a.goal.yaw - *b.goal.yaw, 2.f * std::numbers::pi_v<float>)) > std::min(a.goal.tol_yaw, b.goal.tol_yaw))
        return false;
    return true;
}

// ★ THE COMMANDS IN FORCE over the tick (t_end - dt, t_end] (UnseenContact, review fix round 3): the base executes each
//   command latency_s after it was sent, so the command at the wheels at time t is the last one SENT at or before
//   t - latency_s. Returned as consecutive {duration, v, w} sub-intervals summing to dt; zero before the first send.
struct SentCommand { double t = 0.0; float v = 0.f, w = 0.f; };
inline std::vector<UnseenContact::CmdSeg> commands_in_force(std::span<const SentCommand> sent, double t_end, float dt, float latency_s)
{
    std::vector<UnseenContact::CmdSeg> out;
    const double a = t_end - dt, b = t_end;
    // the command at the wheels at time x
    auto at = [&](double x) -> std::pair<float, float> {
        float v = 0.f, w = 0.f;
        for (const auto& s : sent) { if (s.t + latency_s <= x + 1e-9) { v = s.v; w = s.w; } else break; }
        return {v, w};
    };
    std::vector<double> cuts{a};
    for (const auto& s : sent) { const double c = s.t + latency_s; if (c > a + 1e-9 and c < b - 1e-9) cuts.push_back(c); }
    cuts.push_back(b);
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
    {
        const auto [v, w] = at(cuts[i]);
        const float d = float(cuts[i + 1] - cuts[i]);
        if (d <= 0.f) continue;
        if (not out.empty() and out.back().v == v and out.back().w == w) out.back().dt += d;
        else out.push_back({d, v, w});
    }
    return out;
}

// ★ BASE LIMITS. Speeds: the policy's (config) choice, never above what the base can do. Accelerations: the capability's.
//   Rotation: a differential base's wheels share one acceleration bound, so a_rot = a_lin / (axes_length / 2) — the
//   wheel's linear limit at the wheel's lever arm (Shadow: 0.5 / 0.259 = 1.93 rad/s^2). Deadband: the capability's
//   measured min speeds when robot_concept publishes them (RC_HAVE_ROBOT_DEADBAND_ATTRS), else the PROVISIONAL config
//   values ([Policy] MinLinearSpeedMps / MinRotSpeedRps — the same numbers robot_concept would publish).
struct BaseLimitsInputs
{
    float v_max_policy = 0.5f, w_max_policy = 0.8f;
    std::optional<float> cap_v_max, cap_w_max, cap_a_lin, cap_d_lin, axes_length_m, cap_v_min, cap_w_min;
    float cfg_v_min = 0.04f, cfg_w_min = 0.10f;
    std::optional<float> measured_wheel_decel;   // [Policy] WheelDecelMps2: overrides the capability's d_lin (Live fix 2)
    float fallback_a_lin = 0.5f, fallback_d_lin = 1.0f, fallback_axes_m = 0.518f;
};
inline BaseLimits base_limits(const BaseLimitsInputs& in)
{
    BaseLimits b;
    b.v_max = in.cap_v_max ? std::min(in.v_max_policy, *in.cap_v_max) : in.v_max_policy;
    b.w_max = in.cap_w_max ? std::min(in.w_max_policy, *in.cap_w_max) : in.w_max_policy;
    b.a_lin = in.cap_a_lin.value_or(in.fallback_a_lin);
    // ★ Live fix 2 ruling: the MEASURED per-wheel deceleration overrides the capability's (tourP3: ~0.5 against 1.0)
    b.d_lin = in.measured_wheel_decel.has_value() and *in.measured_wheel_decel > 0.f ? *in.measured_wheel_decel
                                                                                      : in.cap_d_lin.value_or(in.fallback_d_lin);
    const float lever = 0.5f * std::max(0.05f, in.axes_length_m.value_or(in.fallback_axes_m));
    b.a_rot = b.a_lin / lever;
    b.d_rot = b.d_lin / lever;
    b.track_m = 2.f * lever;   // Live fix 2: the axle the wheel-coupled rollout ramps the wheels on
    b.v_min_eff = in.cap_v_min.value_or(in.cfg_v_min);
    b.w_min_eff = in.cap_w_min.value_or(in.cfg_w_min);
    return b;
}
}  // namespace rc::policy
