#pragma once
/*
 * controller_anchor_servo.h — cross a doorway by servoing on its INTERACTION ANCHOR, in the robot frame (2026-10-18).
 *
 * While door_concept executes `cross`, it hangs a shallow copy of the door from the robot (`door_N@robot`); the RT
 * edge robot→copy is the opening's pose IN THE ROBOT FRAME with its covariance (interaction_anchor.h). Anchor frame:
 * origin = the opening's centre on the wall plane, x along the wall, y = the wall normal, robot on −y before crossing.
 * Room-frame door and robot poses were 0.4–0.5 m wrong and drove the robot into the wall; this law never touches a
 * room frame, so a relocalisation or a room switch cannot move the target under it.
 *
 * GEOMETRY. The anchor is at A = (ax, ay) with yaw ψ, all in the robot frame (forward = +y, side = +x).
 *   a = (cos ψ, sin ψ)    the wall direction (anchor +x) in the robot frame
 *   n = (−sin ψ, cos ψ)   the wall normal    (anchor +y) in the robot frame — the direction of travel
 * The robot origin in the anchor frame is Rᵀ(0 − A) = (−a·A, −n·A), so
 *   e = −a·A   the robot's signed offset ALONG THE WALL from the opening's normal line (0 = centred)
 *   s = −n·A   progress along the normal: < 0 before the opening, 0 on the wall plane, > 0 past it
 *   heading error = ψ: n is the robot's forward +y rotated by ψ CCW, so a positive ψ wants a CCW (positive) turn.
 *
 * DESIRED VELOCITY (robot frame), the same for both bases:  d = v_n·n + v_a·a
 *   v_a = −v_max·tanh(e / (v_max·τ_lat))                  nulls e (moves the robot back onto the normal line)
 *   v_n =  v_max·P_fit·tanh((standoff − s)/(v_max·τ_s))   advances, tapering to 0 at the standoff
 * tanh is a smooth saturation with unit slope at 0, so each term is a first-order loop with time constant τ that
 * never exceeds its rate limit. |d| is then scaled (not clipped per axis) to ≤ v_max.
 *
 * DIFFERENTIAL BASE (Shadow: the bridge drops `side`, rc::BaseCapability::holonomic = false). The base can only
 * move along its forward axis, so it TURNS TOWARD d and advances by d's forward projection:
 *   ε   = atan2(−d.x, d.y)                 angle from forward (+y) to d, CCW positive (ε = 0: d straight ahead)
 *   rot = w_max·tanh(ε / (w_max·τ_rot)),  adv = max(0, |d|·cos ε),  side = 0
 * As e → 0, d → n, so the heading converges to the wall normal on its own; with nothing to do (d = 0) it aligns
 * with the normal (ε = ψ).
 *
 * HOLONOMIC BASE. side = d.x, adv = d.y, rot = w_max·tanh(ψ/(w_max·τ_rot)).
 *   ⚠ THE SIGN OF +side ON A REAL BASE IS UNVERIFIED. This law takes +side = body +x (right), as the robot-frame
 *   convention says; trajectory_controller.h (clearance_at's comment) says the OmniRobot command frame has
 *   +side = LEFT, while webots-bridge odometry reports side = −velocity_local(0), i.e. body +x. No controller path
 *   has ever commanded a non-zero side. Verify on the first holonomic run (base_commands.jsonl `side` vs
 *   robot_current_side_speed) before trusting this branch.
 *
 * ★ HOW FAST = HOW LIKELY THE BODY FITS. P_fit = Φ((w/2 − body_half_width − |e|) / σ_lat), σ_lat² = aᵀ P_xy a: the
 *   variance of the opening's position along the wall, from the anchor's own covariance. Off-centre or unsure ⇒
 *   slow (the lateral loop keeps working, so it re-centres first); centred and sure ⇒ full speed. No gate.
 *
 * `through` = s ≥ standoff: the opening is `standoff` behind along the normal. FLAGGED decision level: it ends the
 * action (the robot has nothing left to do here), not a state of belief.
 *
 * Header-only, pure Eigen + std (relative_pose.h for the type).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "../../common/interaction_anchor/anchor_fit.h"
#include "../../common/interaction_anchor/anchor_leaf.h"
#include "../../common/interaction_anchor/relative_pose.h"

namespace rc
{

struct AnchorCommand { float adv = 0, side = 0, rot = 0; bool through = false; };

struct AnchorCrossGeometry
{
    float e = 0.f;              // signed along-wall offset of the robot from the opening's normal line (m)
    float s = 0.f;              // progress along the normal: < 0 before the opening (m)
    float heading_err = 0.f;    // angle from robot forward to the normal, CCW positive (rad)
    float sigma_lat = 0.f;      // sqrt(aᵀ P_xy a): the opening's along-wall position sigma (m)
    Eigen::Vector2f a{1.f, 0.f}, n{0.f, 1.f};   // wall direction and normal, robot frame
};

[[nodiscard]] inline AnchorCrossGeometry anchor_cross_geometry(const rc::interaction::RelPose& r)
{
    AnchorCrossGeometry g;
    const float psi = rc::interaction::wrap_pi(r.x.z());
    g.a = {std::cos(psi), std::sin(psi)};
    g.n = {-std::sin(psi), std::cos(psi)};
    const Eigen::Vector2f A = r.x.head<2>();
    g.e = -g.a.dot(A);
    g.s = -g.n.dot(A);
    g.heading_err = psi;
    g.sigma_lat = std::sqrt(std::max(0.f, g.a.dot(r.P.topLeftCorner<2, 2>() * g.a)));
    return g;
}

// ─── AT THE VANTAGE, ALWAYS AN ACTION (door-passable spec step 2, 2026-11-05) ─────────────────────────────────────────────
// warm61 leg 3 stood 9 minutes in front of an open door: the anchor put the opening 0.46 m to the side (the servo's start
// check |e| < w/2 = 0.44 failed), the room-frame route had ARRIVED (it believed itself aligned), so each rule handed the decision
// to the other and nothing was commanded. Each failing start check now maps to the action that reduces it: off-axis or
// misfacing with the route arrived ⇒ re-aim on the perceived opening (the same anchor the servo will drive on, so success
// makes `within` true by construction). No mouth ⇒ hold (warm24, unchanged). Past the plane ⇒ the engaged servo / the route.
enum class VantageAction { Servo, HoldNoMouth, ReAim, RouteDrives };
// `anchor_fresh` (final review I2): the re-aim goal is the anchor's robot-frame axis composed with the CURRENT pose, so it is
// built only from a fresh anchor — a stale one, stamped at an older pose, would drag the goal along with the robot.
[[nodiscard]] inline VantageAction vantage_action(bool servo_engaged, bool open_belief, bool open_seen, bool near_side,
                                                  bool within, bool facing, bool route_arrived, bool anchor_fresh)
{
    if (servo_engaged or (open_belief and open_seen and near_side and within and facing)) return VantageAction::Servo;
    // ★ HOLD ONLY WHERE A MOUTH COULD BE SEEN (clean50, 2026-11-11): the robot held 8m47s at the vantage "no mouth seen", 0.56 m
    //   off the opening's axis (of ±0.44) and 117 deg off-heading — from there no view through the door was possible, and the
    //   hold forbade the move that would have made one. Waiting is worth something only once the body is placed in front of the
    //   opening (near side, within it, facing it); off-axis the robot re-aims on the anchor first, as it does when a mouth is seen.
    if (not open_seen or not open_belief)
    {
        if (near_side and within and facing) return VantageAction::HoldNoMouth;
        if (near_side and route_arrived and anchor_fresh) return VantageAction::ReAim;
        return VantageAction::RouteDrives;
    }
    if (near_side and route_arrived and anchor_fresh and (not within or not facing)) return VantageAction::ReAim;
    return VantageAction::RouteDrives;
}
// ─── FACE THE DOOR WHEN THE ANCHOR CANNOT SAY SO (2026-11-15; chat_e2e20 `cross door_25`) ──────────────────────────────────
// The robot stood 1.45 m in front of an open door, heading -176 deg from its normal, for 50+ s: the route had ARRIVED (distance), the
// servo's start check failed on `facing`, and the re-aim needs a FRESH anchor — which a body facing away from the door cannot have.
// Each rule waited for the other. The crossing includes turning to face the door and the planner must not need to know, so the
// passage turns the body itself. THE GOAL IS THE DOOR'S HEADING (the anchor's normal: robot forward + heading_err), not "under 90 deg":
// the turn, once begun, runs until the servo's own start condition can hold (the anchor's `facing` is |heading_err| < 90 deg, the
// test the servo already uses) or the Heading goal reports reached — no hysteresis constant of its own. The swept arc is checked
// against the grid first; a blocked arc is REPORTED (the passage fails the route), never waited on.
enum class FaceKind { Proceed, Turn, Report };
struct FaceDecision { FaceKind kind = FaceKind::Proceed; float yaw = 0.f; };   // yaw: the forward heading to turn to (Turn / Report)
// robot_yaw: the body's forward heading (theta + pi/2); heading_err: the anchor's, CCW from forward to the door normal (it is propagated
// by odometry, so it is meaningful while stale); `turning`: this leg's turn has begun; arc_blocked: GridPlanner::why_cannot_sweep.
[[nodiscard]] inline FaceDecision face_decision(VantageAction vact, bool route_arrived, float robot_yaw, float heading_err, bool turning,
                                                bool arc_blocked)
{
    // the servo drives, a fresh anchor re-aims, a placed body holds; before the route has arrived it drives
    if (vact != VantageAction::RouteDrives or not route_arrived) return {};
    const bool facing = std::abs(heading_err) < 0.5f * static_cast<float>(M_PI);   // the servo's own `facing`
    if (facing and not turning) return {};
    const float yaw = rc::interaction::wrap_pi(robot_yaw + heading_err);
    return {arc_blocked ? FaceKind::Report : FaceKind::Turn, yaw};
}
// The re-aim goal in the ROBOT frame: the point on the anchor's axis at the robot's current progress s (robot − e·a), facing
// along the normal (the robot turns by heading_err). The anchor is already a filter (σ_lat); it is not re-filtered here.
struct ReAimGoal { Eigen::Vector2f p_robot = Eigen::Vector2f::Zero(); float yaw_robot = 0.f; };
[[nodiscard]] inline ReAimGoal reaim_on_anchor(const AnchorCrossGeometry& g)
{
    return {.p_robot = -g.e * g.a, .yaw_robot = g.heading_err};
}

// IS THE BODY AT THE VANTAGE (clean3 2026-11-08, tour21): the hand-over from the route to the doorway law asks whether the body is
// at least as close to the door as the VANTAGE THE ROUTE DROVE TO. It compared with door_concept's live approach standpoint, which
// moves after the route is built (1.43 m -> 0.58 m): the route held at its vantage waiting for the cross while the gate
// waited for the body to come closer, and neither could change. One vantage, the route's own.
[[nodiscard]] inline bool passage_at_vantage(const Eigen::Vector2f& robot, const Eigen::Vector2f& door_centre,
                                             const Eigen::Vector2f& vantage, float arrival_tol)
{
    return (robot - door_centre).norm() <= (vantage - door_centre).norm() + arrival_tol;
}

// HAS THE ROUTE REACHED ITS VANTAGE (clean8 2026-11-08): distance to the door is not "in front of it". Driving along the corridor to R2 the body passed
// 1.26 m from door_2 THROUGH THE CORRIDOR WALL, the distance test said "at the vantage", the doorway law took over, saw no opening and held it there.
// The vantage is a point ON THE ROUTE, so being at it is the route's own progress. A route with no vantage (the body already stands in the doorway) is there.
[[nodiscard]] inline bool route_reached_vantage(int goal_index, int vantage_goals, bool route_has_no_vantage)
{
    return route_has_no_vantage or goal_index >= vantage_goals;
}

// Loop time constants (s). Control-loop gains, not decision thresholds. Chosen on the differential base, where the
// heading must turn toward d and back while e closes: the lateral loop must be slow against the rate-limited turn
// (w_max = 1 rad/s) or the robot arrives on the line still pointing across it and overshoots. On a kinematic
// unicycle from e = 0.3 m (tests/test_anchor_servo.cpp): τ_lat 1.0 / τ_rot 0.5 overshoots 7 cm; 2.0 / 0.2
// overshoots 2 mm and is centred (< 5 cm) 1.8 m before the opening.
inline constexpr float kAnchorTauS   = 1.0f;   // approach to the standoff

// ── THE LEAF HYPOTHESES: one passage, judged under all of them (fix round 1, I-4 / I-6) ─────────────────────────────────
// The anchor carries every hinge × swing hypothesis with its probability p_k (anchor_leaf.h). Each leaves its own gap k
// (centre c_k, width w_k). Two quantities follow, both continuous:
//   P(body fits at e) = Σ_k p_k Φ((w_k/2 − body_half − |e − c_k|)/σ_lat)   — the opening's P_fit, under the leaf mixture;
//   the CROSS LINE    = Σ_k p_k u_k c_k,  u_k = Φ((w_k/2 − body_half)/σ_lat) = P(the body fits gap k, centred on it).
// The line moves toward a gap only as far as that gap is believed (p_k) AND usable (u_k): with the hinge at P 0.6 the two
// 90° gaps pull it ±2 cm and it stays near the centre; an ajar leaf leaving a slit (u ≈ 0) keeps the opening's own centre
// line (I-6: the robot is not aimed at a gap it cannot use — it waits on P_fit there). No leaves ⇒ the opening (line 0).
[[nodiscard]] inline float anchor_cross_line(float width, std::span<const rc::interaction::AnchorLeaf> leaves,
                                             float body_half_width, float sigma_lat)
{
    float line = 0.f;
    for (const auto& l : leaves)
    {
        const auto gap = anchor_clear_gap(width, &l);
        line += l.p * anchor_p_fit(gap.width, body_half_width, 0.f, sigma_lat) * gap.centre;
    }
    return line;
}

// ── THE APPROACH STANDPOINT ON THE ANCHOR (Task 12 fix C) ──────────────────────────────────────────────────────────────
// try70: the cross engaged at e −0.31 (legs 3 and 5) and −0.37 (leg 4). door_concept's approach standpoint is the ROOM-frame
// aperture's front pose; against Webots ground truth the robot truly stood 0.13–0.26 m off the opening's normal there (legs
// 3–5) — a room-frame error, the same 0.1–0.5 m the anchor exists to absorb. While the approach is executed the anchor
// already hangs from the robot (it serves approach, then cross), so the standpoint can be asked in ITS frame: keep the
// producer's depth along the normal (its standoff is a design of the producer's) and put it on the CROSS LINE — the clear
// gap's centre, anchor_clear_gap, so the cross starts on the line it will servo on. Robot frame in, robot frame out:
// q_a = R(ψ)ᵀ(q − A), q_a.x ← c_gap, q = A + R(ψ) q_a.
// Fix round 1 (I-5): not an override — the room frame and the anchor both say where the line is, each with its own σ.
// The along-wall coordinate of the standpoint is their PRECISION-WEIGHTED fusion: the room's (the offered point's own
// along-wall coordinate, σ_room = the room-frame door's along-wall σ) and the anchor's (the cross line, σ_anchor = the
// anchor's own σ_lat). σ_room unknown (≤ 0 or not finite) ⇒ the anchor's line alone; the depth stays the producer's.
[[nodiscard]] inline Eigen::Vector2f anchor_standpoint(const rc::interaction::RelPose& r, float width,
                                                       std::span<const rc::interaction::AnchorLeaf> leaves, float body_half_width,
                                                       const Eigen::Vector2f& offered, float sigma_room = -1.f)
{
    if (not r.x.allFinite() or not offered.allFinite() or not (width > 0.f)) return offered;
    const auto g = anchor_cross_geometry(r);
    const float psi = rc::interaction::wrap_pi(r.x.z()), c = std::cos(psi), s = std::sin(psi);
    const Eigen::Vector2f d = offered - r.x.head<2>();
    Eigen::Vector2f qa(c * d.x() + s * d.y(), -s * d.x() + c * d.y());
    const float line = anchor_cross_line(width, leaves, body_half_width, g.sigma_lat);
    if (std::isfinite(sigma_room) and sigma_room > 0.f)
    {
        const float wa = 1.f / std::max(1e-8f, g.sigma_lat * g.sigma_lat), wr = 1.f / (sigma_room * sigma_room);
        qa.x() = (wa * line + wr * qa.x()) / (wa + wr);
    }
    else
        qa.x() = line;
    return r.x.head<2>() + Eigen::Vector2f(c * qa.x() - s * qa.y(), s * qa.x() + c * qa.y());
}

// ── WHEN THE CROSS ENDS (Task 12 Fix E) ────────────────────────────────────────────────────────────────────────────────
// On the anchor the cross ends when the robot is THROUGH (the servo's `through`, s ≥ standoff along the normal) — a room
// switch on the way (room_concept moves the robot's RT ~0.3 m past the plane) does not end it. Without an anchor (the room
// route), the room switch ends it, as before. Pure: the door passage asks it every cycle.
enum class AnchorCrossEnd { Continue, Through, RoomSwitch };
[[nodiscard]] inline AnchorCrossEnd anchor_cross_end(bool anchored, bool room_switched, bool through)
{
    if (anchored) return through ? AnchorCrossEnd::Through : AnchorCrossEnd::Continue;
    return room_switched ? AnchorCrossEnd::RoomSwitch : AnchorCrossEnd::Continue;
}

// ── WAS THE ROBOT EVER ON THE NEAR SIDE? (try89 fix) ──────────────────────────────────────────────────────────────────────
// try89 leg 3: a leftover anchor with a FLIPPED normal (s +1.78, heading 176°) put a robot standing BEFORE the door "past" it,
// and the cross completed `through` at 2.00 m without crossing. `through` (s ≥ standoff) counts only if the robot was OBSERVED
// on the near side earlier in the same claim: s < −half_len (the body wholly before the plane) on a pose seen that cycle (a
// measured gap, or an anchor seen — not predicted, not carried). A claim that starts at s > 0 cannot complete as through.
// ⚑ −half_len is geometry (the body's own half-length), not a tuned level.
struct CrossThroughGate
{
    bool near_seen = false;
    // True on the cycle the near side is first observed.
    bool observe(bool observed, float s, float half_len)
    {
        if (near_seen or not observed or not std::isfinite(s) or not (s < -half_len)) return false;
        near_seen = true;
        return true;
    }
    [[nodiscard]] bool through(bool servo_through) const { return servo_through and near_seen; }
};

// ── IS THE CROSS COMPLETE? (Task 8: shared by the anchor servo and the local policy's doorway drive) ───────────────────────
// `through` = s >= standoff along the normal of the pose driven on (the measured gap, else the anchor); it COMPLETES only if
// the robot was observed on the near side in this claim (CrossThroughGate, try89); s >= standoff without that is NOT A
// CROSSING (a flipped / leftover frame) — the session hands the claim back. A non-finite pose judges nothing (Going).
enum class CrossProgress { Going, Through, NotACrossing };
[[nodiscard]] inline CrossProgress cross_progress(const rc::interaction::RelPose& drive_pose, float standoff, const CrossThroughGate& gate)
{
    if (not drive_pose.x.allFinite() or not drive_pose.P.allFinite()) return CrossProgress::Going;   // review M3
    if (not (anchor_cross_geometry(drive_pose).s >= standoff)) return CrossProgress::Going;
    return gate.through(true) ? CrossProgress::Through : CrossProgress::NotACrossing;
}
[[nodiscard]] inline bool cross_progress_complete(const rc::interaction::RelPose& drive_pose, float standoff, const CrossThroughGate& gate)
{
    return cross_progress(drive_pose, standoff, gate) == CrossProgress::Through;
}

// ── THE ANCHOR GONE MID-CROSS (try90) ───────────────────────────────────────────────────────────────────────────────────
// door_concept removes the anchor once the passage is over by ITS rule (Task 21: through the room switch, the predicted window
// spent — try90: ~0.8 m past the plane). The anchor was only the prior for WHICH gap; once the cross holds a measured (or
// carried) gap line and the robot was observed on its near side, the cross goes on on that line to `through` (s ≥ standoff),
// the returns net in the chain — no wait, and no fallback to a room route on old-room coordinates (review NB2).
[[nodiscard]] inline bool cross_continues_without_anchor(bool gap_held, const CrossThroughGate& gate)
{
    return gap_held and gate.near_seen;
}

// ── …AND WHEN NOTHING CAN CARRY IT (final review I-1) ─────────────────────────────────────────────────────────────────
// With the anchor gone and no held gap (no gap line, or the near side never observed in this claim), what the cross does
// depends on where it is. Before the room switch the room route stands (its coordinates are still this room's). Past it —
// the passage clearing — the cross target is a pose in the room JUST LEFT, read in the new room's frame: it must never be
// driven (review NB2). The cross ENDS instead: the session stops the base and drops `engaged`, so the passage takes its
// room-switch end (rc::anchor_cross_end(false, true, ·) == RoomSwitch) on the next cycle and reports the step's outcome.
// Without this the passage stayed `clearing` for ever (anchored_ still true, through_ never reached) — an unbounded hang.
// The door-route DRIVE-IN (UseRoute, past THROUGH: the body was already judged wholly past the plane) is the same case: its
// leg's cross pose is the room left's, so a lost anchor ENDS the drive-in (DoorPassage::note_drive_in_done, whose contract
// already lists "no anchor") and the passage completes by its room-switch + archive rule. names_ltsm2 (2026-11-15): the
// anchor was removed 1.1 m past the plane, the room route was built on room_1's cross pose in room_2's grid ("goal outside
// the grid") and the leg failed cross_unreachable with the robot in room_2.
enum class AnchorGoneAction { RoomRoute, CarryOnGap, EndClearing, EndDriveIn };
[[nodiscard]] inline AnchorGoneAction anchor_gone_action(bool gap_carries, bool clearing, bool driving_in = false)
{
    if (gap_carries) return AnchorGoneAction::CarryOnGap;
    if (driving_in) return AnchorGoneAction::EndDriveIn;
    return clearing ? AnchorGoneAction::EndClearing : AnchorGoneAction::RoomRoute;
}

// ── DRIVE ON A STALE ANCHOR? (Task 13 Fix E far side; Task 16: from the plane on) ───────────────────────────────────────
// A fresh anchor is driven on. A stale one is driven on — on the pose odometry carries, the guards in the chain — once the
// body's centre is PAST THE PLANE on that pose (s > 0: try76 leg 2 stood at s +0.36 for 40 s waiting for an anchor that
// could not be seen from inside the doorway), or past the room switch (clearing); before the plane the wait-for-fresh rule
// stands (the approach is where a sighting is both possible and needed). ⚑ s = 0 is the wall plane: geometry, the ruling's
// level, not a tuned one.
// ★ Task 16 fix (try81 leg 3): the past-the-plane branch needs the carried pose to DERIVE FROM A SIGHTING in this attempt
//   (`seen_in_attempt`: the servo drove on a fresh anchor since this claim began). After a retry, door_concept's not-seen pose
//   is the room initialisation carried by odometry: its "s +0.20" (σ_lat 0.38 m) put a robot that was really AT the jamb past
//   the plane, and the returns net then blocked every motion. A pose nobody has seen in this attempt says nothing about which
//   side of the plane the body is on — it waits, exactly as before the plane. (A fact, not a level: the covariance route —
//   P_fit at the carried σ — already throttles the servo, and does not by itself stop a never-seen pose.)
[[nodiscard]] inline bool anchor_drive_on(bool fresh, bool clearing, bool seen_in_attempt, const rc::interaction::RelPose& pose)
{
    if (fresh or clearing) return true;
    if (not seen_in_attempt or not pose.x.allFinite()) return false;
    return anchor_cross_geometry(pose).s > 0.f;
}

// ── WAITING FOR AN ANCHOR NOT SEEN YET: TURN SO THE SENSORS CAN SEE IT (Task 9) ────────────────────────────────────────
// Toward the opening's centre (its bearing from forward, CCW positive) while it lies ahead; once the body's centre is within
// half a body length of the wall plane (s > −half_len) — or past it — the centre may be beside or behind the robot, and
// facing along the crossing (the normal: heading_err) is what brings the doorway back into view. The servo's own heading
// loop (w_max·tanh(·/(w_max·τ_rot))). Pure: tests/test_anchor_servo.cpp.
// Task 8: the wait-turn's AIM as a bearing (rad from forward, CCW positive) — the local policy turns to it as a Heading goal.
[[nodiscard]] inline float anchor_wait_aim(const rc::interaction::RelPose& r, float half_len)
{
    if (not r.x.allFinite()) return 0.f;
    const auto g = anchor_cross_geometry(r);
    return g.s > -half_len ? g.heading_err : std::atan2(-r.x.x(), r.x.y());
}
// ── THE JAMBS AS OBSTACLES THE MODEL ALREADY KNOWS ────────────────────────────────────────────────────────────
// The live field cannot stop the robot at a jamb: the self-filter clears every return within ~0.4 m of the body
// centre, and the LiDAR ceiling is floored at 0.05 m/s. The anchor knows where the jambs are: J± = A ± (w/2)·a in
// the robot frame, with covariance Σ_J = P_xy + (w/2)²·σ_ψ²·n nᵀ (dJ/dψ = ±(w/2)·n). That is the SAME Gaussian P_fit
// reads (aᵀ P_xy a is its along-wall part), projected here on whichever axis a question is asked along, so the
// ceilings and P_fit agree about one opening. Each jamb limits a motion only as far as that motion CLOSES on it:
//
// TRANSLATION along unit u (p ⟂ u). The body sweeps a corridor of half-width support(±p); the jamb counts with
// P_in = w((support(±p) − |J·p|)/σ_p), σ_p² = pᵀΣ_J p — P_fit's own form, slack over sigma (w: anchor_contact_weight).
// Moving along u closes on it only while it is AHEAD of the centre, P_closing = Φ((J·u)/σ_u), σ_u² = uᵀΣ_J u; a jamb
// beside or behind the body is not approached by translating along u, so it limits nothing (re-centring past a jamb
// is not frozen). The gap to the leading face is g = J·u − support(u); a blocked jamb allows
// v_block = Φ(g/σ_u)·sqrt(2·a_dec·max(0, g)). The ceiling from that jamb is v_cap − P_in·P_closing·(v_cap − v_block).
// NO floor: a jamb the mean puts ahead and touching gives 0.
//
// ROTATION by sign(ω) about the centre. The jamb (radius r_J, angle φ_J) is reached by the body boundary points at
// that same radius — the polygon's crossings of the circle |x| = r_J. Δ = the smallest rotation, in the commanded
// direction, that brings one of them onto φ_J (0 if the jamb is already inside the body; no crossing and the body
// shorter than r_J ⇒ unreachable). It counts with P_reach = w((r_max − r_J)/σ_r), σ_r² = r̂ᵀΣ_J r̂; a blocked turn
// allows |ω| ≤ Φ(Δ/σ_θ)·sqrt(2·(a_dec/r_J)·max(0, Δ)), σ_θ = (t̂ᵀΣ_J t̂)^½ / r_J (a_dec as tangential deceleration at
// r_J). Same blend, no floor. A turn AWAY from a jamb sees it a long way round, so only a closing turn is capped.
//
// body = the footprint polygon in the robot frame (x right, y forward), margin = its outward safety margin.

// radius: the obstacle's own extent; p: the probability it is there at all (1 for a jamb; a leaf hypothesis's p_k, I-4).
struct AnchorJamb { Eigen::Vector2f mean; Eigen::Matrix2f cov; float radius = 0.f; float p = 1.f; };

// ── THE SWUNG LEAF AS OBSTACLES THE MODEL KNOWS (Task 12 fix A) ─────────────────────────────────────────────────────────
// try70: every failed crossing was on the side the 90° leaf swings toward. The LiDAR self-filter (~0.405 m) hides a leaf
// within reach of the body and the jamb guard modelled only the two jambs, so the servo turned a corner into the leaf
// (leg 3: 1.75 rad commanded per 3 s, 0.00 netted) or drove beside it. The anchor carries the leaf (anchor_leaf.h); here it
// becomes the same kind of obstacle as a jamb: points q along its centre line, each a disc of the leaf's half-thickness,
// with the Gaussian the anchor's own covariance and the leaf angle's σ give it. In the robot frame q = A + R(ψ)·q_a, so
//   Σ_q = P_xy + σ_ψ² g gᵀ + r² σ_φ² t tᵀ,   g = R(ψ)·(−q_a.y, q_a.x) (∂q/∂ψ),  t = R(ψ)·(the leaf's unit normal),
// r = |q_a − H| (a point r from the hinge moves r·dφ across the leaf when φ moves). For a jamb (q_a = (±w/2, 0)) g = ±(w/2)·n:
// exactly anchor_jambs' term, so jambs and leaf are judged on one rule. Spacing: the half-thickness, so the discs cover the
// slab's faces to within t(1 − √3/2) ≈ 0.13 t — ARITHMETIC (a discretisation), not a model level.
[[nodiscard]] inline std::vector<AnchorJamb> anchor_leaf_points(const rc::interaction::RelPose& r, const rc::interaction::AnchorLeaf& leaf)
{
    std::vector<AnchorJamb> out;
    const float len = leaf.length();
    if (not (len > 0.f) or not (leaf.half_thick > 0.f) or not r.x.allFinite()) return out;
    const float psi = rc::interaction::wrap_pi(r.x.z()), c = std::cos(psi), s = std::sin(psi);
    const Eigen::Matrix2f R{{c, -s}, {s, c}};
    const Eigen::Vector2f A = r.x.head<2>();
    const Eigen::Vector2f dir = (leaf.tip - leaf.hinge) / len, nrm_a(-dir.y(), dir.x());
    const Eigen::Vector2f t = R * nrm_a;
    const float var_psi = std::max(0.f, r.P(2, 2)), var_phi = leaf.sigma_phi * leaf.sigma_phi;
    const int n = 1 + int(std::ceil(len / leaf.half_thick));
    out.reserve(std::size_t(n));
    for (int i = 0; i < n; ++i)
    {
        const float rr = len * float(i) / float(n - 1);
        const Eigen::Vector2f qa = leaf.hinge + rr * dir;
        const Eigen::Vector2f g = R * Eigen::Vector2f(-qa.y(), qa.x());
        const Eigen::Matrix2f cov = r.P.topLeftCorner<2, 2>() + var_psi * (g * g.transpose()) + (rr * rr * var_phi) * (t * t.transpose());
        out.push_back(AnchorJamb{A + R * qa, cov, leaf.half_thick, std::clamp(leaf.p, 0.f, 1.f)});
    }
    return out;
}

// How much a jamb counts on an axis where `z` = mean slack / sigma (slack > 0: the MEAN geometry puts the jamb in
// the body's way). min(1, 2Φ(z)) = Φ(z)/Φ(0): the Gaussian's probability renormalised so the mean-contact case is
// CERTAIN. Plain Φ would make the expected allowance at a mean contact (1 − Φ(0))·v_cap = half speed into the jamb;
// this keeps the soft tail outside (a jamb probably beside the body slows it as much as it is probably in the way)
// while a jamb the mean says is in the way gets the full braking ceiling.
[[nodiscard]] inline float anchor_contact_weight(float z) { return std::min(1.f, 2.f * anchor_std_normal_cdf(z)); }

[[nodiscard]] inline float anchor_axis_sigma(const Eigen::Matrix2f& cov, const Eigen::Vector2f& axis)
{ return std::sqrt(std::max(1e-12f, axis.dot(cov * axis))); }

[[nodiscard]] inline float anchor_body_support(std::span<const Eigen::Vector2f> body, float margin,
                                               const Eigen::Vector2f& dir)
{
    float best = 0.f;
    for (const auto& v : body) best = std::max(best, v.dot(dir));
    return best + margin;
}

// The body polygon grown by `margin` (radially per vertex — the margin is small against the body).
[[nodiscard]] inline std::vector<Eigen::Vector2f> anchor_grown_body(std::span<const Eigen::Vector2f> body, float margin)
{
    std::vector<Eigen::Vector2f> out;
    out.reserve(body.size());
    for (const auto& v : body)
    {
        const float n = v.norm();
        out.push_back(n > 1e-6f ? Eigen::Vector2f(v * ((n + margin) / n)) : v);
    }
    return out;
}

[[nodiscard]] inline bool anchor_point_in_convex(std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& q)
{
    if (poly.size() < 3) return false;
    float sign = 0.f;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Eigen::Vector2f e = poly[(i + 1) % poly.size()] - poly[i], d = q - poly[i];
        const float c = e.x() * d.y() - e.y() * d.x();
        if (c == 0.f) continue;
        if (sign == 0.f) sign = c;
        else if ((c > 0.f) != (sign > 0.f)) return false;
    }
    return true;
}

// Signed distance from q to the boundary of a convex polygon: > 0 outside, < 0 inside (the penetration depth).
[[nodiscard]] inline float anchor_signed_distance(std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& q)
{
    if (poly.size() < 3) return std::numeric_limits<float>::infinity();
    float d = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Eigen::Vector2f a = poly[i], e = poly[(i + 1) % poly.size()] - a;
        const float t = std::clamp(e.squaredNorm() > 0.f ? (q - a).dot(e) / e.squaredNorm() : 0.f, 0.f, 1.f);
        d = std::min(d, (q - (a + t * e)).norm());
    }
    return anchor_point_in_convex(poly, q) ? -d : d;
}

// The body's motion as seen by a FIXED point q of the robot frame: translating the body by (side, adv) and turning it by
// dθ moves q to R(−dθ)(q − d) — the same algebra as relative_pose.h move_mean.
[[nodiscard]] inline Eigen::Vector2f anchor_point_after(const Eigen::Vector2f& q, float side, float adv, float dth)
{
    const float c = std::cos(dth), s = std::sin(dth);
    const Eigen::Vector2f p = q - Eigen::Vector2f(side, adv);
    return {c * p.x() + s * p.y(), -s * p.x() + c * p.y()};
}

// ⚑ The finite-difference step of the clearance gains below: 1 mm of body-point travel. ARITHMETIC (a derivative's step,
// far below every length in the problem), not a model level.
inline constexpr float kAnchorGainStepM = 1e-3f;

// The cap on |ω| for a turn in the direction of `rot` (rad/s, CCW positive), against any set of robot-frame obstacles
// (Task 14: the anchor's jambs and leaf points, or the LiDAR's own returns — lidar_body_guard.h — on one rule).
[[nodiscard]] inline float anchor_obstacles_rot_limit(std::span<const AnchorJamb> obstacles, float rot,
                                                      std::span<const Eigen::Vector2f> body, float margin, float a_dec,
                                                      float w_cap)
{
    if (rot == 0.f or body.size() < 3) return w_cap;
    const float two_pi = 2.f * std::numbers::pi_v<float>;
    const auto wrap_0_2pi = [two_pi](float x) { x = std::fmod(x, two_pi); return x < 0.f ? x + two_pi : x; };
    // The body grown by the margin plus each obstacle's own radius (jambs 0, the leaf its half-thickness): two polygons.
    float poly_radius = -1.f, r_max = 0.f;
    std::vector<Eigen::Vector2f> poly;
    float w = w_cap;
    for (const auto& j : obstacles)
    {
        if (j.radius != poly_radius)
        {
            poly_radius = j.radius;
            poly = anchor_grown_body(body, margin + j.radius);
            r_max = 0.f;
            for (const auto& v : poly) r_max = std::max(r_max, v.norm());
        }
        const float rj = j.mean.norm();
        if (rj < 1e-4f) { w = 0.f; continue; }                         // a jamb on the rotation centre: inside
        const Eigen::Vector2f rh = j.mean / rj, th(-rh.y(), rh.x());
        const float phi = std::atan2(j.mean.y(), j.mean.x());
        float delta = std::numeric_limits<float>::infinity();
        if (anchor_point_in_convex(poly, j.mean))
        {
            // Task 9, fix C: a jamb already INSIDE the grown body is reached by every turn (Δ = 0) — but a turn that moves
            // the body boundary AWAY from it opens clearance and is not closing on it. Only the closing direction is capped
            // (to 0); the other is this jamb's way out. (Before: both directions capped to 0, and with adv ≥ 0 the base
            // could neither turn nor move — a deadlock the stall judge then called a wedge.)
            const float h = kAnchorGainStepM / std::max(rj, 1e-3f);
            const float gain = anchor_signed_distance(poly, anchor_point_after(j.mean, 0.f, 0.f, std::copysign(h, rot)))
                             - anchor_signed_distance(poly, j.mean);
            if (gain > 0.f) continue;
            delta = 0.f;
        }
        else
            for (std::size_t i = 0; i < poly.size(); ++i)                 // crossings of |x| = rj with each edge
            {
                const Eigen::Vector2f p0 = poly[i], d = poly[(i + 1) % poly.size()] - p0;
                const float A = d.squaredNorm(), B = 2.f * p0.dot(d), C = p0.squaredNorm() - rj * rj;
                const float disc = B * B - 4.f * A * C;
                if (A <= 0.f or disc < 0.f) continue;
                for (const float t : {(-B - std::sqrt(disc)) / (2.f * A), (-B + std::sqrt(disc)) / (2.f * A)})
                {
                    if (t < 0.f or t > 1.f) continue;
                    const Eigen::Vector2f q = p0 + t * d;
                    const float th_q = std::atan2(q.y(), q.x());
                    delta = std::min(delta, rot > 0.f ? wrap_0_2pi(phi - th_q) : wrap_0_2pi(th_q - phi));
                }
            }
        if (not std::isfinite(delta)) continue;                            // the sweep never reaches this radius
        const float p_reach = anchor_contact_weight((r_max - rj) / anchor_axis_sigma(j.cov, rh));
        const float s_th = anchor_axis_sigma(j.cov, th) / rj;
        const float w_block = std::min(w_cap, anchor_std_normal_cdf(delta / s_th)
                                              * std::sqrt(2.f * std::max(0.f, a_dec) / rj * delta));
        w = std::min(w, w_cap - j.p * p_reach * (w_cap - w_block));   // weighted by P(it is there) (I-4)
    }
    return std::max(0.f, w);
}

// ── MAKING ROOM FOR THE TURN (try101) ────────────────────────────────────────────────────────────────────────────────────
// A differential base aligns by turning in place, and a turn is refused by every obstacle its sweep reaches: the grown
// body's corners reach R = max |vertex| (+ margin + the obstacle's radius) from the centre, so an obstacle at |q| < R that the
// turn closes on caps it (anchor_obstacles_rot_limit). The servo had no motion for that: its translation is d's forward
// projection, which is (near) zero or INTO the obstacle while the heading is far off. try101 leg 1 (and its two retries)
// stood 0.37 m from room_1's corridor wall facing it at 15-25°, the turn toward the door refused by that wall (R 0.396 m),
// forward refused by the same wall — and the returns' backoff (which minimises margin penetration, not the turn's need)
// turned it back the other way each cycle: a CW/CCW chatter until the contract timed out, three times.
// The motion that gives the turn its room is the translation that moves the obstacles it is blocked by OUT of the sweep:
// a blocker j (its own cap below the asked |rot|) has the reach deficit δ_j = R_j − |q_j| > 0, and translating the body by
// h along u changes it by h·(q_j·u)/|q_j|. Candidates: forward/reverse (and ±side on a holonomic base); the one that lowers
// the total P-weighted deficit Σ p_j δ_j fastest is taken (a SUM, as the returns' backoff: a face beside the body must not
// veto the reverse that frees the corner ahead). Speed: the worst deficit closed on the approach loop's time constant,
// v_max·tanh(max p_j δ_j / (v_max·τ_s)) — the deficit is a length the body still has to put between itself and the blocker.
// The servo's turn stays as asked (the guards cap it to what the room allows); its translation is REPLACED by this one
// unless the servo already moves at least as fast the same way. Nothing blocks the turn, or no candidate lowers the deficit
// ⇒ the servo's command, untouched. The result then goes through the SAME guard chain, so the translation that makes room
// cannot close on anything behind. Receding horizon: every cycle the blockers are found again, and once the turn has its
// room the servo's own translation is back.
struct AnchorTurnBlockers { std::span<const AnchorJamb> obstacles; std::span<const Eigen::Vector2f> body; };
struct AnchorTurnRoom { AnchorCommand cmd; bool applied = false; int blockers = 0; float deficit = 0.f; };
[[nodiscard]] inline AnchorTurnRoom anchor_make_turn_room(const AnchorCommand& c, std::span<const AnchorTurnBlockers> sets,
                                                          float margin, float a_dec, float v_max, float w_max, bool holonomic)
{
    AnchorTurnRoom out{.cmd = c};
    if (c.through or not (std::abs(c.rot) > 0.f) or not (v_max > 0.f)) return out;
    const float ask = std::abs(c.rot);
    struct Blocker { Eigen::Vector2f q; float deficit, p; };
    std::vector<Blocker> blk;
    for (const auto& set : sets)
    {
        if (set.body.size() < 3) continue;
        float r_body = 0.f;
        for (const auto& v : set.body) r_body = std::max(r_body, v.norm());
        for (const auto& j : set.obstacles)
        {
            const float rq = j.mean.norm();
            const float reach = r_body + margin + j.radius;   // anchor_grown_body grows radially: its max |vertex|
            if (not (rq > 1e-4f) or rq >= reach) continue;
            if (anchor_obstacles_rot_limit(std::span<const AnchorJamb>(&j, 1), c.rot, set.body, margin, a_dec, w_max) >= ask)
                continue;
            blk.push_back({j.mean, reach - rq, std::clamp(j.p, 0.f, 1.f)});
        }
    }
    out.blockers = int(blk.size());
    if (blk.empty()) return out;
    std::vector<Eigen::Vector2f> cand{{0.f, 1.f}, {0.f, -1.f}};
    if (holonomic) { cand.push_back({1.f, 0.f}); cand.push_back({-1.f, 0.f}); }
    float best_gain = 0.f;
    std::optional<Eigen::Vector2f> best;
    for (const auto& u : cand)
    {
        float gain = 0.f;   // −d(Σ p δ)/dh: moving the body along u moves each blocker by −u
        for (const auto& b : blk) gain -= b.p * b.q.dot(u) / b.q.norm();
        if (gain > best_gain) { best_gain = gain; best = u; }
    }
    if (not best.has_value()) return out;
    float worst = 0.f;
    for (const auto& b : blk) worst = std::max(worst, b.p * b.deficit);
    out.deficit = worst;
    const float v = v_max * std::tanh(worst / (v_max * kAnchorTauS));
    const float own = Eigen::Vector2f(c.side, c.adv).dot(*best);   // the servo's own translation along u
    if (own >= v) return out;                                       // it already makes the room, at least as fast
    out.cmd.side = holonomic ? v * best->x() : 0.f;
    out.cmd.adv = v * best->y();
    out.applied = true;
    return out;
}

}  // namespace rc
