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

// Φ, the standard normal CDF.
[[nodiscard]] inline float anchor_std_normal_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

// P(the body fits through the opening at the current lateral offset). σ_lat = 0 (a certain opening) is the limit
// of the expression: a step at slack 0.
[[nodiscard]] inline float anchor_p_fit(float width, float body_half_width, float e, float sigma_lat)
{
    const float slack = 0.5f * width - body_half_width - std::abs(e);
    if (sigma_lat <= 0.f) return slack > 0.f ? 1.f : (slack < 0.f ? 0.f : 0.5f);
    return anchor_std_normal_cdf(slack / sigma_lat);
}

// Loop time constants (s). Control-loop gains, not decision thresholds. Chosen on the differential base, where the
// heading must turn toward d and back while e closes: the lateral loop must be slow against the rate-limited turn
// (w_max = 1 rad/s) or the robot arrives on the line still pointing across it and overshoots. On a kinematic
// unicycle from e = 0.3 m (tests/test_anchor_servo.cpp): τ_lat 1.0 / τ_rot 0.5 overshoots 7 cm; 2.0 / 0.2
// overshoots 2 mm and is centred (< 5 cm) 1.8 m before the opening.
inline constexpr float kAnchorTauLat = 2.0f;   // lateral re-centring
inline constexpr float kAnchorTauS   = 1.0f;   // approach to the standoff
inline constexpr float kAnchorTauRot = 0.2f;   // heading alignment

// ── THE CLEAR GAP THE SWUNG LEAF LEAVES (Task 12 fix A) ──────────────────────────────────────────────────────────────
// The leaf hangs from one jamb; its centre line's footprint ALONG THE WALL, grown by its half-thickness, is
// [min(H.x, T.x) − t, max(H.x, T.x) + t] (anchor_leaf.h; anchor frame). Whatever side it swings to, the body passes through
// both half-spaces of the doorway, so the passage the robot can use is the part of the opening [−w/2, +w/2] that footprint
// does not cover: the interval beside the leaf, up to the far jamb. A leaf at 90° takes its thickness off the hinge side;
// an ajar leaf takes its projection L·cos φ; one swung past 90° folds back over the wall and takes only its thickness.
// The CROSS LINE is that interval's centre and P(body fits) its width — the same geometry, no new level. No leaf ⇒ the
// opening itself (centre 0, width w).
struct AnchorGap { float centre = 0.f, width = 0.f; };
[[nodiscard]] inline AnchorGap anchor_clear_gap(float width, const rc::interaction::AnchorLeaf* leaf)
{
    const float half = 0.5f * width;
    if (leaf == nullptr or not (width > 0.f)) return {0.f, std::max(0.f, width)};
    const float lo_leaf = std::min(leaf->hinge.x(), leaf->tip.x()) - leaf->half_thick;
    const float hi_leaf = std::max(leaf->hinge.x(), leaf->tip.x()) + leaf->half_thick;
    float lo = -half, hi = half;
    if (leaf->hinge.x() < 0.f) lo = std::clamp(hi_leaf, -half, half);   // hung from the −x jamb: the gap is to its +x
    else                       hi = std::clamp(lo_leaf, -half, half);   // hung from the +x jamb: the gap is to its −x
    return {0.5f * (lo + hi), std::max(0.f, hi - lo)};
}

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
[[nodiscard]] inline float anchor_p_fit_leaves(float width, std::span<const rc::interaction::AnchorLeaf> leaves,
                                               float body_half_width, float e, float sigma_lat)
{
    if (leaves.empty()) return anchor_p_fit(width, body_half_width, e, sigma_lat);
    float p = 0.f;
    for (const auto& l : leaves)
    {
        const auto gap = anchor_clear_gap(width, &l);
        p += l.p * anchor_p_fit(gap.width, body_half_width, e - gap.centre, sigma_lat);
    }
    return std::clamp(p, 0.f, 1.f);
}

// d nulls the lateral offset e to the CROSS LINE (anchor_cross_line) and advances along the normal at a speed scaled by
// P(body fits) under the leaf hypotheses (anchor_p_fit_leaves); `through` once the opening is `standoff` behind along it.
// holonomic: side/adv = d, rot aligns the heading with n. Differential: side = 0, rot turns toward d, adv = d's forward
// projection. Signs: adv = forward (+y) m/s, side = lateral (+x) m/s, rot = rad/s CCW positive (the value handed to the
// base). No leaves: the opening's own centre and width (the pre-Task-12 law).
[[nodiscard]] inline AnchorCommand anchor_cross_command(const rc::interaction::RelPose& r, float width,
                                                        float body_half_width, float standoff, float v_max,
                                                        float w_max, bool holonomic,
                                                        std::span<const rc::interaction::AnchorLeaf> leaves = {})
{
    AnchorCommand c;
    if (not r.x.allFinite() or not r.P.allFinite() or not (v_max > 0.f) or not (w_max > 0.f)) return c;
    const auto g = anchor_cross_geometry(r);
    if (g.s >= standoff) { c.through = true; return c; }

    const float e_line = g.e - anchor_cross_line(width, leaves, body_half_width, g.sigma_lat);
    const float p_fit = anchor_p_fit_leaves(width, leaves, body_half_width, g.e, g.sigma_lat);
    const float v_a = -v_max * std::tanh(e_line / (v_max * kAnchorTauLat));
    // Task 13: the loop aims one σ PAST the standoff (σ_s = the opening's σ along its normal), so `through` (s ≥ standoff) is
    // reached in finite time — aimed exactly at the level, the first-order loop only approaches it (a kinematic replay of
    // the far side sat at s 2.00 for 60 s). The overshoot is the pose's own uncertainty about where the standoff is.
    const float sigma_s = std::sqrt(std::max(0.f, g.n.dot(r.P.topLeftCorner<2, 2>() * g.n)));
    const float v_n = v_max * p_fit * std::tanh((standoff + sigma_s - g.s) / (v_max * kAnchorTauS));
    Eigen::Vector2f d = v_n * g.n + v_a * g.a;
    float speed = d.norm();
    if (speed > v_max) { d *= v_max / speed; speed = v_max; }
    if (holonomic)
    {
        c.side = d.x();
        c.adv = d.y();
        c.rot = w_max * std::tanh(g.heading_err / (w_max * kAnchorTauRot));
        return c;
    }
    const float eps = speed > 0.f ? std::atan2(-d.x(), d.y()) : g.heading_err;
    c.side = 0.f;
    c.adv = std::max(0.f, speed * std::cos(eps));
    c.rot = w_max * std::tanh(eps / (w_max * kAnchorTauRot));
    return c;
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
enum class AnchorGoneAction { RoomRoute, CarryOnGap, EndClearing };
[[nodiscard]] inline AnchorGoneAction anchor_gone_action(bool gap_carries, bool clearing)
{
    if (gap_carries) return AnchorGoneAction::CarryOnGap;
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
[[nodiscard]] inline float anchor_wait_turn(const rc::interaction::RelPose& r, float w_max, float half_len)
{
    if (not r.x.allFinite() or not (w_max > 0.f)) return 0.f;
    const auto g = anchor_cross_geometry(r);
    const float aim = g.s > -half_len ? g.heading_err : std::atan2(-r.x.x(), r.x.y());
    return w_max * std::tanh(aim / (w_max * kAnchorTauRot));
}

// ── IS THE WAIT-TURN'S ASK JUDGEABLE? (Task 11, fix 3) ──────────────────────────────────────────────────────────────────
// The stall judge weighs the heading a window was TOLD to sweep against the heading it NETTED. A wait-turn that already
// faces the opening asks ~0 (tanh of a tiny bearing), and try68 read every such 3-s window as SPINNING ("0.00 rad
// commanded"): an ask smaller than what the heading estimate can resolve cannot be failed, only mis-measured. Task 9's
// round-2 cure skipped the judge for EVERY wait cycle, so a turn ground against a jamb for up to the 40-s wait, with no
// LiDAR check on rotation. The line between the two is not a constant: the turn this rate asks over ONE judge window
// must exceed the localiser's own heading σ — below it "did it turn" has no answer. An unknown σ (≤ 0 or not finite)
// leaves the ask unweighable, and the wait is not judged (the old behaviour, bounded by the wait's own timeout).
[[nodiscard]] inline bool anchor_wait_turn_judged(float rot_rps, float window_s, float theta_std_rad)
{
    if (not std::isfinite(rot_rps) or not std::isfinite(theta_std_rad) or not (theta_std_rad > 0.f) or not (window_s > 0.f))
        return false;
    return std::abs(rot_rps) * window_s > theta_std_rad;
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

[[nodiscard]] inline std::array<AnchorJamb, 2> anchor_jambs(const rc::interaction::RelPose& r, float width)
{
    const auto g = anchor_cross_geometry(r);
    const Eigen::Vector2f A = r.x.head<2>();
    const Eigen::Matrix2f cov = r.P.topLeftCorner<2, 2>()
                              + (0.25f * width * width * std::max(0.f, r.P(2, 2))) * (g.n * g.n.transpose());
    return {AnchorJamb{A - 0.5f * width * g.a, cov}, AnchorJamb{A + 0.5f * width * g.a, cov}};
}

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

// The jambs and every leaf hypothesis: every obstacle the guards below judge, each with its probability.
[[nodiscard]] inline std::vector<AnchorJamb> anchor_obstacles(const rc::interaction::RelPose& r, float width,
                                                              std::span<const rc::interaction::AnchorLeaf> leaves)
{
    const auto j = anchor_jambs(r, width);
    std::vector<AnchorJamb> out(j.begin(), j.end());
    for (const auto& l : leaves)
        for (auto& q : anchor_leaf_points(r, l)) out.push_back(std::move(q));
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

[[nodiscard]] inline float anchor_jamb_speed_limit(const rc::interaction::RelPose& r, float width,
                                                   const Eigen::Vector2f& u, std::span<const Eigen::Vector2f> body,
                                                   float margin, float a_dec, float v_cap,
                                                   std::span<const rc::interaction::AnchorLeaf> leaves = {})
{
    if (not r.x.allFinite() or not r.P.allFinite() or not (u.squaredNorm() > 0.f) or not (width > 0.f) or body.empty())
        return v_cap;
    const Eigen::Vector2f un = u.normalized();
    const Eigen::Vector2f p(-un.y(), un.x());
    float v = v_cap;
    for (const auto& j : anchor_obstacles(r, width, leaves))
    {
        const float m = margin + j.radius;   // a disc obstacle = a point against the body grown by its radius
        const float lead = anchor_body_support(body, m, un);
        const float along = j.mean.dot(un), lat = j.mean.dot(p);
        const float s_u = anchor_axis_sigma(j.cov, un), s_p = anchor_axis_sigma(j.cov, p);
        const Eigen::Vector2f side = lat >= 0.f ? p : Eigen::Vector2f(-p);
        const float p_in = anchor_contact_weight((anchor_body_support(body, m, side) - std::abs(lat)) / s_p);
        const float p_closing = anchor_std_normal_cdf(along / s_u);
        const float gap = along - lead;
        const float v_block = std::min(v_cap, anchor_std_normal_cdf(gap / s_u)
                                              * std::sqrt(2.f * std::max(0.f, a_dec) * std::max(0.f, gap)));
        v = std::min(v, v_cap - j.p * p_in * p_closing * (v_cap - v_block));   // weighted by P(it is there) (I-4)
    }
    return std::max(0.f, v);
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

[[nodiscard]] inline float anchor_jamb_rot_limit(const rc::interaction::RelPose& r, float width, float rot,
                                                 std::span<const Eigen::Vector2f> body, float margin, float a_dec,
                                                 float w_cap, std::span<const rc::interaction::AnchorLeaf> leaves = {})
{
    if (rot == 0.f or not r.x.allFinite() or not r.P.allFinite() or not (width > 0.f) or body.size() < 3) return w_cap;
    const auto obs = anchor_obstacles(r, width, leaves);
    return anchor_obstacles_rot_limit(obs, rot, body, margin, a_dec, w_cap);
}

// ── THE JAMB DEADLOCK AND THE SERVO'S OWN WAY OUT (Task 9, fix C) ─────────────────────────────────────────────────────
// With a jamb mean INSIDE the grown body (the safety margin already breached, the body itself not yet touching) the
// ceilings above stop every motion that closes on it: forward when it is ahead (no floor), and the turn toward it. If the
// servo's own command is one of those, the base stands still in the doorway — and the stall judge, seeing a command with
// no motion, called that a wedge and escaped (try66 leg 2: "spinning verdict, translation asked", three escapes,
// Unreachable). That is the servo's problem to solve, with the geometry it already has. While the servo's own capped
// command opens clearance it stands; when the caps leave it nothing that does (both bind), command the motion that
// INCREASES the clearance. Candidates: reverse or advance along the heading (and ±side on a holonomic base), or turn either way; each
// scored by the gain of the MIN signed clearance of the two jambs to the grown body per metre of body-point travel (a turn
// by its arm, the jamb's radius), the largest positive gain taken. Its speed closes the deficit to "probably outside" —
// the penetration plus the jamb's own σ along the escape — on the approach loop's time constant (kAnchorTauS):
// v = v_max·tanh((pen + σ)/(v_max·τ_s)), a turn at the corner speed v/r_J. The result then goes through the same jamb
// ceilings, so it cannot close on the OTHER jamb. No gain anywhere ⇒ no command (the stall judge stays the backstop).
struct AnchorGuarded { AnchorCommand cmd; bool backoff = false; float min_clearance = std::numeric_limits<float>::infinity(); };

// grown: the body grown by its safety margin; an obstacle of radius ρ is clear of it by its signed distance − ρ. A breach
// (negative clearance) counts in proportion to P(the obstacle is there) — I-4: a minority leaf hypothesis inside the margin
// is a fraction of a breach, continuous in p, not a deadlock of its own.
[[nodiscard]] inline float anchor_obstacle_clearance(const AnchorJamb& j, std::span<const Eigen::Vector2f> grown,
                                                     float side, float adv, float dth)
{
    const float c = anchor_signed_distance(grown, anchor_point_after(j.mean, side, adv, dth)) - j.radius;
    return c >= 0.f ? c : j.p * c;
}
[[nodiscard]] inline float anchor_min_clearance(std::span<const AnchorJamb> obstacles, std::span<const Eigen::Vector2f> grown,
                                                float side, float adv, float dth)
{
    float c = std::numeric_limits<float>::infinity();
    for (const auto& j : obstacles) c = std::min(c, anchor_obstacle_clearance(j, grown, side, adv, dth));
    return c;
}
[[nodiscard]] inline float anchor_min_clearance(const rc::interaction::RelPose& r, float width,
                                                std::span<const Eigen::Vector2f> grown, float side, float adv, float dth,
                                                std::span<const rc::interaction::AnchorLeaf> leaves = {})
{
    const auto obs = anchor_obstacles(r, width, leaves);
    return anchor_min_clearance(obs, grown, side, adv, dth);
}

// The servo command `c` through the jamb ceilings (translation along the base's own motion axis, rotation by the corner
// sweep), with the deadlock's own way out. Differential: side is forced to 0.
[[nodiscard]] inline AnchorGuarded anchor_jamb_guard(const rc::interaction::RelPose& r, float width, AnchorCommand c,
                                                     bool holonomic, std::span<const Eigen::Vector2f> body, float margin,
                                                     float a_dec, float v_max, float w_max,
                                                     std::span<const rc::interaction::AnchorLeaf> leaves = {})
{
    AnchorGuarded g;
    if (not holonomic) c.side = 0.f;
    auto cap = [&](AnchorCommand& k)
    {
        if (const float lin = std::hypot(k.adv, k.side); lin > 0.f)
        {
            const Eigen::Vector2f u(k.side / lin, k.adv / lin);
            if (const float v = anchor_jamb_speed_limit(r, width, u, body, margin, a_dec, v_max, leaves); lin > v)
            { k.adv *= v / lin; k.side *= v / lin; }
        }
        if (k.rot != 0.f)
            k.rot = std::copysign(std::min(std::abs(k.rot), anchor_jamb_rot_limit(r, width, k.rot, body, margin, a_dec, w_max, leaves)), k.rot);
    };
    g.cmd = c;
    if (not r.x.allFinite() or not r.P.allFinite() or not (width > 0.f) or body.size() < 3) return g;
    const auto grown = anchor_grown_body(body, margin);
    const auto obs = anchor_obstacles(r, width, leaves);
    g.min_clearance = anchor_min_clearance(obs, grown, 0.f, 0.f, 0.f);
    cap(g.cmd);
    if (not (g.min_clearance < 0.f) or c.through) return g;   // nothing inside the grown body: the servo, capped
    // An obstacle (a jamb, or Task 12 a point of the leaf) inside the grown body. The servo's own (capped) command stands if
    // it already OPENS clearance — the caps leave it a way out. Only when the caps have bound it to a motion that does not
    // (the deadlock) does the guard step in.
    const AnchorJamb* bp = &obs.front();
    for (const auto& j : obs)
        if (anchor_obstacle_clearance(j, grown, 0.f, 0.f, 0.f) < anchor_obstacle_clearance(*bp, grown, 0.f, 0.f, 0.f)) bp = &j;
    const AnchorJamb& breached = *bp;
    const float arm = std::max(breached.mean.norm(), 1e-3f);
    const float h = kAnchorGainStepM;
    if (const float sp = std::hypot(g.cmd.adv, g.cmd.side) + std::abs(g.cmd.rot) * arm; sp > 0.f)
    {
        const float k = h / sp;   // the servo's command scaled to h of body-point travel
        if (anchor_min_clearance(obs, grown, k * g.cmd.side, k * g.cmd.adv, k * g.cmd.rot) > g.min_clearance) return g;
    }

    // The deadlock: the motion with the largest clearance gain per metre of body-point travel.
    struct Cand { float side, adv, rot; };
    std::vector<Cand> cand{{0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}};
    if (holonomic) { cand.push_back({1.f, 0.f, 0.f}); cand.push_back({-1.f, 0.f, 0.f}); }
    // The arm of a turn: the radius of the jamb the body has breached most.
    float best_gain = 0.f;
    std::optional<Cand> best;
    for (const auto& k : cand)
    {
        const float gain = (anchor_min_clearance(obs, grown, h * k.side, h * k.adv, h * k.rot / arm) - g.min_clearance) / h;
        if (gain > best_gain) { best_gain = gain; best = k; }
    }
    g.backoff = true;
    g.cmd = AnchorCommand{};
    if (not best.has_value()) return g;   // nothing opens clearance: stand still, the stall judge decides
    // The escape direction in the robot frame (translation: its unit vector; a turn: the breached jamb's tangent).
    const Eigen::Vector2f dir = best->rot != 0.f ? Eigen::Vector2f(Eigen::Vector2f(-breached.mean.y(), breached.mean.x()) / arm)
                                                 : Eigen::Vector2f(best->side, best->adv);
    const float sigma = anchor_axis_sigma(breached.cov, dir);
    const float v = v_max * std::tanh((-g.min_clearance + sigma) / (v_max * kAnchorTauS));
    g.cmd.side = best->side * v;
    g.cmd.adv  = best->adv * v;
    g.cmd.rot  = std::clamp(best->rot * v / arm, -w_max, w_max);
    cap(g.cmd);
    return g;
}

}  // namespace rc
