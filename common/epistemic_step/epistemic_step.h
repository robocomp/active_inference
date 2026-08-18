#pragma once
/*
 * common/epistemic_step/epistemic_step.h — the per-instance epistemic cycle, run once.
 *
 * WHAT WAS DUPLICATED. Seven copies of the same lifecycle around one per-object call: decrement the
 * cooldown, notice that the controller has completed the affordance and start a hold, ask the planner for a
 * proposal, suppress or floor its gain, publish it, and remember that a proposal is pending. The planner call
 * in the middle is genuinely per-object — every agent has its own `EpistemicPlanner` and its own
 * `EpistemicProposal` — but nothing around it was.
 *
 * ★★★THE DEFECT THIS EXISTS TO MAKE STRUCTURAL: "NO PROPOSAL THIS CYCLE" IS NOT A WITHDRAWAL.
 * A cycle can fail to produce a proposal for ordinary reasons — the belief has not seen its first frame, the
 * camera model is not complete yet, the fit is degenerate. On those cycles `update()` is skipped, and the
 * RE-OFFER lives inside `update()`, so an affordance the controller had just completed stayed Completed FOR
 * EVER: skipped by every selection branch, its gain frozen at whatever it last published. Measured live
 * 2026-08-07 — `aff_table_1` sat at gain=5, the verify-me maximum, and Completed, while afford_room's gain
 * ticked on beside it. `ObjectAffordance::hold_offered()` was added for exactly this and documents it, yet
 * FOUR agents still returned bare from SEVEN bail paths between them (bottle 2, cabinet 2, chair 2, door 1).
 * Here there is one bail path and it always calls hold_offered(), so the bug cannot be reintroduced by
 * forgetting a line. The protocol state ("do I still want this look?") and the target ("where, and how
 * much?") are different statements and must not share a code path.
 *
 * ★★THE PROPOSAL TYPE IS A TEMPLATE PARAMETER, ON PURPOSE. `EpistemicProposal` is per-agent and comes in
 * tiers: bottle/chair/door carry 5 fields, cabinet 10, hood/refrigerator 13, table 14. The optional ones are
 * copied onto the AffordanceTarget under `if constexpr (requires ...)`, so an agent publishes exactly the
 * fields its planner actually computes — and no agent starts publishing a default-constructed stand-off band
 * it never calculated, which is what a uniform 11-field copy would have done to bottle, chair and door.
 *
 * ⚠KNOWN, DELIBERATELY PRESERVED DIVERGENCE — `record` receives BOTH gains. hood/refrigerator/table mirror
 * `dbg_nbv_gain` BEFORE the cooldown suppression and the verification pull (so their ai2_log column is the
 * RAW belief gain); bottle/cabinet/chair mirror it AFTER, and their comment says "the value actually
 * published". The same column therefore means two different things across the fleet, and during a cooldown
 * hold table logs a high gain while publishing 0. Both readings are defensible, so this extraction does NOT
 * pick one — it hands `record` the final proposal and the pre-adjustment gain and lets each agent keep its
 * present meaning. ★Resolve it deliberately (log both, in two columns) rather than by accident.
 *
 * MAIN-THREAD ONLY: reads and writes the graph.
 */

#include <cstdint>
#include <functional>
#include <optional>
#include <print>

#include <dsr/api/dsr_api.h>

#include "../object_affordance/object_affordance.h"   // rc::AffordanceTarget, ObjectAffordance

namespace rc::epistemic
{

// The per-cycle numbers the agent decides.
struct StepInputs
{
    // Length of the post-completion hold. The affordance completes on a WEAK detection (contract goal
    // conf >= 0.20), which fires almost instantly — before ΔH has decayed — so without a hold a
    // just-completed object is re-offered while its gain is still high. The node is NOT deleted (that is
    // what made affordances vanish from the graph); it keeps refreshing and only its GAIN is suppressed,
    // so the controller's EFE selection simply stops picking it.
    int   cooldown_cycles = 0;

    // Verification pull, or 0 for none. An object whose predicted absence could NOT be resolved from recent
    // views (far / peripheral / edge-on "I don't see it") must not be deleted — it should be LOOKED at. A
    // floor under the gain sends the robot to get a verifying view. It OVERRIDES the cooldown on purpose: a
    // "might be gone" alarm is not anti-chatter-suppressible. Agents with no verification channel pass 0.
    float verify_gain_floor = 0.0f;

    // Orient rather than Servo. A BEARING-ONLY hypothesis has no depth, so the robot can only turn to face
    // it and let a real detection resolve it. Agents with no hypothesis state pass false.
    bool  orient_mode = false;
};

template <class Proposal>
struct StepHooks
{
    // THE per-object part: ask this agent's planner. Returning nullopt means "no proposal this cycle" —
    // belief not started, camera model incomplete, degenerate fit. ★It is NOT a withdrawal; see the header.
    std::function<std::optional<Proposal>()> compute;

    // Record the proposal: mirror it onto the instance for ai2_log, write it onto the object node.
    // `published` carries the FINAL gain (after suppression and the verification floor); `raw_gain` is the
    // planner's gain before either. See the divergence note in the header for why both are handed over.
    std::function<void(const Proposal& published, float raw_gain)> record;

    // The affordance node did not exist before this cycle and does now (re-layout the graph view).
    std::function<void()> on_affordance_created;

    // After the affordance has been refreshed — so anything reading its protocol state here sees it fresh.
    std::function<void(const Proposal& published)> on_published;
};

/*
 * Run one epistemic cycle for one instance.
 *
 * `Inst` must expose: epistemic_cooldown, epistemic_pending, node_name, affordance.
 */
template <class Inst, class Proposal>
void step(Inst& inst, DSR::DSRGraph& G, const StepInputs& in, const StepHooks<Proposal>& hooks)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold. `active` false AND `epistemic_pending` false is the controller saying it
    // finished with this look; the default for `pending` is TRUE so a node we have not read yet is never
    // mistaken for a completed one.
    if (const auto aid = inst.affordance.node_id(); aid != 0)
        if (auto an = G.get_node(aid); an.has_value())
        {
            const bool active  = G.get_attrib_by_name<active_att>(an.value()).value_or(false);
            const bool pending = G.get_attrib_by_name<epistemic_pending_att>(an.value()).value_or(true);
            if (not active and not pending and inst.epistemic_cooldown == 0)
            {
                inst.epistemic_cooldown = in.cooldown_cycles;
                std::print("[{}] controller completed affordance → hold {} cycles (node kept, gain suppressed)\n",
                           inst.node_name, in.cooldown_cycles);
            }
        }

    const auto proposal = hooks.compute();
    if (not proposal.has_value())
    {
        // ★THE ONE BAIL PATH, AND IT ALWAYS HOLDS THE OFFER. See the header: leaving the affordance as-is is
        // not neutral, because as-is is Completed, which the controller reads as a withdrawal and which never
        // recovers on its own. hold_offered() leaves an EXECUTING claim alone — that one is the controller's.
        inst.affordance.hold_offered();
        return;
    }
    Proposal prop = *proposal;
    const float raw_gain = prop.epistemic_gain;

    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;
    if (in.verify_gain_floor > 0.0f)
        prop.epistemic_gain = std::max(prop.epistemic_gain, in.verify_gain_floor);

    if (hooks.record)
        hooks.record(prop, raw_gain);

    const auto affordance_node_before = inst.affordance.node_id();

    // Planner internals stay in EpistemicProposal; the producer takes the shared view. The optional fields
    // are copied only where the agent's proposal HAS them — see the header note on the tiers.
    rc::AffordanceTarget tgt;
    tgt.x_m     = prop.epistemic_target_x_m;
    tgt.y_m     = prop.epistemic_target_y_m;
    tgt.yaw_rad = prop.epistemic_target_yaw_rad;
    tgt.gain    = prop.epistemic_gain;
    tgt.valid   = prop.valid;
    if constexpr (requires { prop.face_gains; })
        tgt.face_gains.assign(prop.face_gains.begin(), prop.face_gains.end());
    if constexpr (requires { prop.sigma_star; })
        tgt.sigma_star.assign(prop.sigma_star.begin(), prop.sigma_star.end());
    if constexpr (requires { prop.standoff_min_m; })
    {
        tgt.standoff_min_m = prop.standoff_min_m;
        tgt.standoff_max_m = prop.standoff_max_m;
    }
    if constexpr (requires { prop.framing_fill; })
        tgt.framing_fill = prop.framing_fill;

    inst.affordance.update(tgt, in.orient_mode);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0 and hooks.on_affordance_created)
        hooks.on_affordance_created();
    inst.epistemic_pending = true;

    if (hooks.on_published)
        hooks.on_published(prop);
}

}  // namespace rc::epistemic
