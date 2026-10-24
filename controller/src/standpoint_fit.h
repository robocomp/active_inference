// standpoint_fit.h — does an affordance standpoint SERVE, or is it handed back? (Task 3c, 2026-10-21, rtD1d leg 4)
//
// ★ WHY. With the local policy driving, recheck_standpoint_on_approach used to RE-SITE an object (Servo) standpoint
//   whenever ANY live LiDAR return fell inside a 0.62 m disc around it (circumscribed hull + margin + goal_threshold,
//   tested as a disc because the terminal facing turn sweeps every heading). Live, rtD1d leg 4: door_concept offered
//   aff_door_1_approach at the middle of its reach band ([1.55, 2.00] m from the door); single returns moved it
//   0.44 m and then 0.58 m, to 2.09 m out — beyond the band. P(reach) stayed 0.17, the lock-on could only turn, and the
//   approach timed out twice. The consumer re-sited a producer's standpoint by a criterion the producer does not share,
//   on evidence (one return in a disc, existence 1) that the policy's own contact set barely held (0-1 returns).
//
// ★ THE RULE (ruling 2026-10-21). One predicate for "a standpoint that serves this affordance", and the consumer never
//   re-sites by any other: the standpoint serves iff the body can be there under the occupancy belief; otherwise it is HANDED
//   BACK (Unreachable) so the producer re-solves inside its own band — as Reach standpoints already were. Nothing is
//   moved, so the producer's arrival predicate (door_concept's P(reach)) is never judged at a pose it did not choose.
//
// ★ "FITS" IS THE PLANNER'S OCCUPANCY BELIEF, the one the policy's cost-to-go already reads (GridPlanner):
//     hard cells (room walls, object polygons, residual cells with no belief behind them or corroborated by a live
//     return in the body band — residual_risk.h harden_corroborated)  -> the body cannot be there;
//     soft residual cells -> -log Pr(cell mostly free | residual's Beta), summed over the cells the body covers.
//   So -log P(fits) = +inf if a hard cell is under the body, else the soft nats under it (GridPlanner::fit_nats_*).
//   A lone return with no residual cell behind it is not in that belief at all, and one outside the footprint is not
//   under the body — so a stray hit in a disc condemns nothing (the policy's contact model still sees every live return
//   while it drives, which is what keeps the body out of contact on the way in).
//   Which footprint: a target that ends in a facing turn sweeps EVERY heading (the union of the eight footprints, each
//   cell counted once — fit_nats_turning); one that does not stands at one heading, the cheapest (fit_nats_some_heading).
//
// ★ THE DECISION: hand back iff P(fits) = 0 under that belief — a HARD cell under the body. Soft cells are priced, never
//   a hand-back by themselves, exactly as the planner and the policy's free mask treat them ("an unconfirmed obstacle is
//   a toll, not a wall", residual_risk.h); -log P(fits) is still computed and logged.
//   ★ WHY NOT THE MAP BAR P(fits) < ½ (tried first, rejected on the test's numbers): residual's cell price is
//   -log Pr(theta < ½), and EVERY published cell carries occupancy evidence (P > ½), so Pr(free) < ½ for any one of them —
//   a barely-seen cell gives P(fits) 0.49, a one/two-hit latch 0.41, Jeffreys alone exactly ½. The MAP bar therefore
//   reduces to "any residual cell under the body", i.e. one stray hit again. The hard/soft partition is the belief's own
//   statement of what blocks: a residual cell becomes hard only with no belief behind it or when a LIVE return lands in
//   it this cycle (two independent observations, not one stray hit), and a confirmed object the robot is approaching is
//   seen by the LiDAR inside the recheck window, which is what hardens it — so the hand-back fires on that cycle.
//   A standpoint inside soft-only occupancy is served; if the body really cannot get there the policy's reachability
//   belief (F1) hands it back on realized progress, without a bar here.
//
// Pure (GridPlanner + std); tested in tests/test_standpoint_fit.cpp.
#pragma once
#include <cmath>
#include <Eigen/Dense>
#include "grid_planner.h"

namespace rc::standpoint
{
enum class Verdict { Serves, HandBack, Unknown };

struct Fit
{
    float nats = 0.f;          // -log P(the body fits): +inf when a hard cell is under it
    bool  turning = false;     // judged over every heading (a terminal facing turn) or at the cheapest one
    [[nodiscard]] float p() const { return std::exp(-nats); }
};

[[nodiscard]] inline Fit body_fit(const rc::GridPlanner& g, const Eigen::Vector2f& p, bool turning, bool persistent_only = false)
{
    return {turning ? g.fit_nats_turning(p, persistent_only) : g.fit_nats_some_heading(p, persistent_only), turning};
}

// P(fits) = 0 (a hard cell under the body) -> HandBack; otherwise Serves (header). Unknown before the planner has a world.
[[nodiscard]] inline Verdict verdict(const Fit& f) { return std::isfinite(f.nats) ? Verdict::Serves : Verdict::HandBack; }

[[nodiscard]] inline Verdict judge(const rc::GridPlanner& g, const Eigen::Vector2f& p, bool turning, bool persistent_only = false)
{
    if (not g.has_world()) return Verdict::Unknown;
    return verdict(body_fit(g, p, turning, persistent_only));
}

// ── THE ONE DECISION BOTH CALL SITES MAKE (fix round 1, I-2) ─────────────────────────────────────────────────────────
// Stage::Planning — build_planning_step, every cycle, at ANY range: hands back only for causes that do not come and go
//   (walls, object polygons, residual cells with no belief). A cell hard only because a live return lands in it THIS
//   cycle (a person walking past the standpoint while the robot is metres away) does not hand back here (I-1): it is
//   judged where the LiDAR measures the spot —
// Stage::ApproachWindow — recheck_standpoint_on_approach, inside its window: every hard cell counts.
// Actions: LegacyRepair = the old chain's repair runs unchanged (always with the policy OFF — the A/B arm; and a Reach
//   cell at the Planning stage, which room publishes as a 0.5 m CELL, so resolving inside it edits nobody's decision);
//   ServeAsPublished = keep the producer's pose exactly; HandBack = Unreachable to the producer. There is no action that
//   moves a pose: under the policy no branch re-sites.
// Which footprint (M-2): every non-Reach contract under the policy ends in a turn in place — Servo's facing turn or its
//   lock-on micro-search (controller_lockon.h: yaw centring and a yaw dither at the standpoint, to whatever heading the
//   detection needs, whether or not GoalFacingYawEnabled), Orient its whole purpose — so it is judged through the turn; a Reach only where it arrives (`rotate`: wants_final_facing).
//   An unknown contract under the policy is not assumed to be Reach: it is never re-sited either.
enum class Stage { Planning, ApproachWindow };
enum class Action { LegacyRepair, ServeAsPublished, HandBack };
struct Decision { Action action = Action::LegacyRepair; Fit fit{}; };

[[nodiscard]] inline Decision decide_standpoint(const rc::GridPlanner& g, const Eigen::Vector2f& published, bool rotate,
                                                bool policy_on, bool contract_known, bool is_reach, Stage stage)
{
    if (not policy_on) return {Action::LegacyRepair, {}};
    const bool reach = contract_known and is_reach;
    if (stage == Stage::Planning and reach) return {Action::LegacyRepair, {}};
    if (not g.has_world()) return {Action::ServeAsPublished, {}};      // no belief yet: no verdict, and never a move
    const Fit f = body_fit(g, published, rotate or not reach, /*persistent_only=*/stage == Stage::Planning);
    return {verdict(f) == Verdict::HandBack ? Action::HandBack : Action::ServeAsPublished, f};
}

// ── D4: A PLACE THE FIELD CANNOT REACH IS HANDED BACK, NOT DRIVEN AT ─────────────────────────────────────────────────────
// body_fit / decide_standpoint judge the standpoint against the planner's belief; the policy's cost-to-go is built on its own
// free mask (the inscribed disc must clear the distance field), and a place the two disagree on used to be driven at: gold13's
// epistemic look aff_door_1 sat inside Bed(1) — "goal free 0, field inf m from the robot" — and the controller spun in NO ROUTE.
// The goal is seeded in a non-free cell and the field stays infinite everywhere, the robot's cell included. So the verdict is
// the field's own: a place whose cost-to-go at the robot is not finite is a place the body cannot be taken to — Unreachable to
// the producer (object and Reach alike; the look, the approach and the cell all go through here), which re-sites it.
[[nodiscard]] inline bool place_unreachable(float cost_to_go_at_robot) { return not std::isfinite(cost_to_go_at_robot); }

[[nodiscard]] inline const char* action_name(Action a)
{
    return a == Action::LegacyRepair ? "LegacyRepair" : a == Action::ServeAsPublished ? "ServeAsPublished" : "HandBack";
}
}  // namespace rc::standpoint
