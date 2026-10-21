/*
 * door_pragmatics.h — the three PRAGMATIC door affordances: approach, open, cross.
 *
 * The epistemic affordance (`aff_door_N`, rc::ObjectAffordance) asks "where should I stand so the leaf
 * resolves?". These three ask what the robot can DO with the door, and each is offered only while the
 * world is in a state where the action is possible:
 *
 *   approach — ALWAYS, for any located door. Stand in the actuation zone, facing through the aperture.
 *              Its completion predicate is `door_reach_prob`, which is exactly the precondition `open`
 *              needs — so "approach succeeded" and "open became possible" are the SAME statement,
 *              measured once, rather than two numbers that can disagree.
 *   open     — while the robot is believed to be IN the actuation zone and the aperture is believed
 *              NOT passable. Claiming it makes door_concept ask a provider (Webots supervisor, home
 *              automation, a person over TTS) to move the leaf; see SpecificWorker::request_door_actuation.
 *   cross    — while the aperture is believed passable for this robot's body.
 *
 * ─── NO THRESHOLDS: WHAT EACH PRECONDITION ACTUALLY IS ────────────────────────────────────────────
 *
 * ★`door_open_prob` IS MARGINALISED OVER THE LEAF-ANGLE POSTERIOR. `DoorInstance::phi_curve` carries the
 * whole (phi, weight) curve precisely so a consumer does not have to condition on its argmax, and the
 * reason is measured: the peak support is 0.007-0.245, i.e. the curve is nearly FLAT and its argmax is
 * noise (see DoorInstance::phi_curve, and the same lesson in [[table-phantom-immortality]]). A flat curve
 * therefore yields a p_open near the prior and no affordance appears or disappears on it; a peaked curve
 * yields a decisive one. There is no angle threshold anywhere: the geometric question "does the robot's
 * body fit through the gap this leaf angle leaves?" is asked ONCE PER HYPOTHESIS and the answers are
 * averaged under their own weights.
 *
 * ★`door_reach_prob` IS A PROBABILITY OVER THE POSE, not a distance test. The actuation zone has a radius
 * that is a CAPABILITY (how close a robot has to be for a request to be unambiguously about THIS door and
 * for the manoeuvre that follows to make sense), and the robot's distance to the aperture is uncertain —
 * so what crosses the wire is P(d <= R) under the pose and aperture covariance. At d == R it is 0.5 by
 * construction, which is the honest answer, and the offer/withdraw decision is then a Bayesian decision on
 * that probability with a Schmitt band (rc::pragmatic::PragmaticAffordance::Policy) — the same shape as
 * Existence.RemovalProb, not a new kind of knob.
 *
 * ★`door_crossing_progress` IS SIGNED AGAINST THE SIDE HELD WHEN THE CLAIM WAS MADE. "Crossed" is not a
 * position, it is a CHANGE of side, so no static predicate on a coordinate can express it. The side the
 * robot was on when the consumer claimed `cross` is latched (DoorInstance::cross_side0) and progress runs
 * -1 (back where it started) → +1 (clear on the far side). A consumer can then hold a plain `>= 0.9`
 * clause against it and still be asking the right question.
 *
 * Header-only geometry + a builder for the three rc::pragmatic::Offer values. No DSR, no Qt: the
 * preconditions are computable — and testable — from a pose, an aperture and a phi posterior alone.
 */

#pragma once

#include <limits>
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../../common/affordance_protocol/affordance_protocol.h"
#include "../../common/pragmatic_affordance/pragmatic_affordance.h"
#include "door_config.h"      // rc::DoorPragmaticCfg — the tunables live with every other tunable
#include "door_geometry.h"
#include "door_passability.h"  // clear_width, passable_prob
#include "door_standpoint.h"   // the poses + which side of the aperture to stand on
#include "door_approach_pose.h" // kApproachCompleteProb — the approach contract's completion level

namespace rc::door
{

// What the agent believes about interacting with ONE door, this cycle. Every field carries its own
// `*_known`: "could not compute" and "computed and it is false" are different facts (rc::pragmatic rule 3).
struct InteractionState
{
    float phi_rad = 0.0f;
    bool  phi_known = false;

    float p_open = 0.0f;              // P(the aperture is passable for this body)
    bool  p_open_known = false;       // false ⇒ no leaf-angle posterior this cycle
    // WHERE the weights came from: the image's own phi curve, or the belief's persistence prior when no
    // curve was produced this cycle. ★Published beside the probability on purpose — the two are the same
    // marginalisation over different weights, and a log that cannot tell them apart cannot be audited.
    bool  p_open_from_curve = false;
    // Task 11 fix round 1: the same quantity as a LOG-ODDS, log P(passable) − log P(not), taken in log space from the
    // φ posterior (passable_logit). p_open = σ(p_open_logit). Finite on a collapsed curve, so nothing is absorbing.
    double p_open_logit = 0.0;

    // P(the robot is AT the door for whatever comes next), marginalised over the door's state:
    //   P(open)·P(d ≤ reach) + P(shut)·P(in band).
    // The band's near edge is the leaf's swept arc, a hazard only for a door that still has to be SWUNG. An
    // open door has nothing left to swing, so being closer than the arc is being at it, not short of it.
    // ★Before 2026-10-17 this was the band alone: try46 leg 2 stood 1.29 m from an OPEN door (band
    // [1.50, 2.00]), P(reach)=0.02, `approach` could never complete, `open` is never offered for an open
    // door, and the passage never goes approach→cross — so the robot stalled at the door for ever.
    float p_reach = 0.0f;
    bool  p_reach_known = false;      // false ⇒ no robot pose this cycle
    float p_band  = 0.0f;             // P(d_min ≤ d ≤ d_max): `open`'s half — it is the step that swings the leaf
    float p_within = 0.0f;            // P(d ≤ d_max): AT an open door — `cross`'s half (cross_offer_prob)
    // P(the opening is IN FRONT of the robot): its apparent span w·|cos θ| from where the robot stands admits the body
    // (rc::door::prob_facing_opening, tourP8). Multiplied into p_band and p_within (hence p_reach): in range but 58°
    // off the normal is not "at the door".
    float p_facing = 1.0f;
    float facing_deg = 0.0f;          // θ, the robot's angle off the aperture normal (diagnostic)

    float crossing_progress = 0.0f;   // -1 .. +1, signed against the latched side
    bool  crossing_known = false;

    float robot_side = 0.0f;          // signed distance robot→aperture plane along the aperture normal (m)
    // The side (±1 along the aperture normal) to STAND on: the side in the robot's own region of the room
    // polygon (rc::door::approach_side). Equals sign(robot_side) in a convex room; differs when the robot is
    // inside the room but beyond the plane of a door it can only reach from the other side. 0 = no pose.
    float approach_side = 0.0f;
    float approach_standoff_m = 0.0f; // middle of the actuation band [safe_standoff, reach] — the stand-off
    float robot_range_m = 0.0f;       // |robot - aperture centre| (m)
    float range_sigma_m = 0.0f;       // σ of that range (door Σ projected + pose chain, floored)
    float safe_standoff_m = 0.0f;     // near edge of the actuation band = the leaf's swept arc + the body
    float clear_width_m = 0.0f;       // along-wall clear span at the MODAL phi (diagnostic only)
};

// ─── geometry ────────────────────────────────────────────────────────────────────────────────────

// clear_width() and passable_prob() live in door_passability.h (dependency-free, so the offline phi harness
// tests/test_door_phi_stability.cpp marginalises with the SAME function the agent publishes).

// prob_within (P(d ≤ R)) lives in door_passability.h, beside cross_offer_prob (dependency-free, tested offline).

// P(d_min <= d <= d_max): the actuation zone is a BAND, not a disc.
// ★A DISC IS THE WRONG SHAPE AND IT FAILS TOWARD THE DANGER. "Within reach" stays true as the robot gets
// CLOSER, so `open` remained on offer right up against the door — inside the arc the leaf sweeps when it
// opens. The zone has a far edge (beyond it, a request is not unambiguously about THIS door) AND a near
// edge (inside it, the door the robot asked to open hits the robot). Both edges are real; only one was
// modelled. Still a probability over the pose, so there is no gate anywhere — just two CDFs instead of one.
inline float prob_in_band(float d, float d_min, float d_max, float sigma)
{
    if (d_max <= d_min) return 0.0f;   // an empty band can never be satisfied, and says so
    return std::max(0.0f, prob_within(d, d_max, sigma) - prob_within(d, d_min, sigma));
}

// The nearest a robot may stand to the aperture plane without entering the leaf's swept arc.
// Derived from the door and the body, never configured — see DoorPragmaticCfg::swing_clearance_m.
inline float safe_standoff(float aperture_w, const DoorPragmaticCfg& p)
{
    return aperture_w + 0.5f * p.robot_passage_width_m + p.swing_clearance_m;
}

// ─── the three contracts ─────────────────────────────────────────────────────────────────────────
// Each binds its completion predicate to a quantity door_concept publishes on the DOOR node (the
// affordance's parent, which is the executor's default feedback node). A contract bound to an attribute
// nobody writes never satisfies and times out on every attempt — the failure default_contract_for()
// documents for cabinet and door, so these names must stay in step with write_interaction_state().

// approach: drive to the actuation zone and end up FACING through the aperture. Servo, not Reach, because
// ControllerSession::wants_final_facing honours the affordance's target yaw only for Servo/Orient — a
// Reach would put the robot in the right place pointing along the wall.
inline rc::affordance::Contract approach_contract(const DoorPragmaticCfg& p)
{
    using enum rc::affordance::CompareOp;
    return rc::affordance::Contract::servo()
        .center("door_roi_offset")
        .valid ("door_roi_valid")
        .until ("door_reach_prob", GE, kApproachCompleteProb)   // ONE value, read by the pose solver too (door_approach_pose.h)
        .still (0.10f, 0.15f)
        .stable(2).timeout_s(p.approach_timeout_s).on_fail(rc::affordance::OnFail::Consume);
}

// ★THE TURN OF AN open/close IS NOT A LOOK, so this producer declares its rate (Contract::max_yaw_rate) instead of
//   inheriting the consumer's default. That default is the LockOn creep (0.06 rad/s), tuned to keep masks sharp while
//   a servo collects them; open/close observe nothing while they turn (the .still hold applies once aligned) and are
//   bounded by open_timeout_s, so the creep made any arrival more than ~41 deg off the leaf a guaranteed timeout
//   (try98 leg 1: 12 s at 0.06 rad/s, "orient GIVE_UP", door_wont_open). No rate of our own: the largest float asks
//   for "no cap from this producer" and the consumer clamps it to the base's own limit (max_rot_speed_rps).
inline constexpr float kTurnAtBaseLimit = std::numeric_limits<float>::max();

// open: the robot is ALREADY there — approach put it there, and that is the precondition. So this must not
// navigate: Orient turns in place to face the leaf and then WAITS for the world to change. A Servo would
// run its distance sweep and creep around while the leaf swings, which is both pointless and unsafe next
// to a moving door. Completion is PERCEPTUAL (door_open_prob), never the provider's own `Delivered` —
// a protocol event is not evidence.
inline rc::affordance::Contract open_contract(const DoorPragmaticCfg& p)
{
    using enum rc::affordance::CompareOp;
    return rc::affordance::Contract::orient()
        .until ("door_open_prob", GE, 0.70f)
        .still (0.05f, 0.10f)
        .yaw_rate(kTurnAtBaseLimit)
        .stable(2).timeout_s(p.open_timeout_s).on_fail(rc::affordance::OnFail::Consume);
}

// close: the mirror of open, after a crossing — Orient (no navigation: the robot stands clear of the swing, which is
// close's precondition) and WAIT until the leaf is seen shut. Completion is perception, never the provider's Delivered.
inline rc::affordance::Contract close_contract(const DoorPragmaticCfg& p)
{
    using enum rc::affordance::CompareOp;
    return rc::affordance::Contract::orient()
        .until ("door_open_prob", LE, 0.30f)
        .still (0.05f, 0.10f)
        .yaw_rate(kTurnAtBaseLimit)
        .stable(2).timeout_s(p.open_timeout_s).on_fail(rc::affordance::OnFail::Consume);
}

// cross: drive THROUGH. Reach, because there is nothing to servo on — the goal is a change of side, and the
// predicate says so directly. No .still: holding still in a doorway is the one thing this must not do.
inline rc::affordance::Contract cross_contract(const DoorPragmaticCfg& p)
{
    using enum rc::affordance::CompareOp;
    return rc::affordance::Contract::reach()
        .until ("door_crossing_progress", GE, 0.90f)
        .stable(2).timeout_s(p.cross_timeout_s).on_fail(rc::affordance::OnFail::Consume);
}

// ─── the poses ───────────────────────────────────────────────────────────────────────────────────
// Pose2, front_pose, through_pose and approach_side live in door_standpoint.h (pure geometry, tested offline).

}  // namespace rc::door
