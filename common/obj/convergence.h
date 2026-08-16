/*
 * common/obj/convergence.h — "is this belief settled?", decided in one place. SHARED, header-only.
 *
 * ★CONVERGENCE IS ON THE STATE, NOT ON |ΔFE|. The free energy keeps jittering with queue churn and point
 * count even when the fitted geometry has stopped moving, so a |ΔFE| latch never fired. What settles is the
 * accepted STATE, so the rule counts consecutive frames whose state moved less than `state_eps`, and calls
 * the model stable at `k_stable` of them. Every one of the six agents that had this had exactly that rule.
 *
 * ★THE SEAM IS THE DELTA. Which DOFs enter the sum is the one genuinely per-object part — a hood adds
 * z_top, a refrigerator adds leg_inset, a chair its seat dimensions — so the caller computes its own
 * state_delta and hands over a number. Everything after it, including WHEN the graph is told, is the rule.
 *
 * Why this is worth sharing at 24 lines a copy: it is a LIFECYCLE DECISION with two graph writes hanging off
 * it, and the fleet has already watched a decision of exactly this shape (the removal debounce) drift three
 * ways while every copy still looked right. `model_stable` is consumed by other agents; six independent
 * answers to "stable since when?" is six chances for one of them to be a different question.
 */

#pragma once

#include <algorithm>

#include <dsr/api/dsr_api.h>

namespace rc::converge
{

struct Params
{
    float state_eps = 0.0f;   // a frame counts as "settled" when the state moved less than this
    int   k_stable  = 0;      // consecutive settled frames required before the model is called stable
};

// Advance the convergence counter and publish the verdict. `frames_converged` and `model_stable` are the
// caller's instance fields, updated in place. Returns true when the flag CHANGED, so a caller that wants to
// log the edge does not have to re-derive it.
//
// ⚠BEHAVIOUR PRESERVED EXACTLY, INCLUDING SOMETHING THAT LOOKS WRONG. `update_node` is called only on a
// model_stable TRANSITION, so the model_uncertainty_att written just above it reaches the graph only on
// those rare cycles — it is a continuous readout that is, in practice, published a handful of times in a
// run. All six copies did this, and the obvious "fix" is to update every cycle.
//
// ★I did not, and the reason is worth keeping next to the code: model_uncertainty_att has SEVEN writers and
// ZERO readers across the whole tree. Publishing it per cycle would add DSR traffic — and CRDT dot growth,
// which this fleet has already been bitten by — for an attribute nobody reads. The honest fix is to decide
// whether anything should consume it, and delete it if not; that is a call for the owner, not a side effect
// of an extraction. See the published-attrs-without-consumers audit (161 of 292 unread).
inline bool step(DSR::DSRGraph& G, DSR::Node& node,
                 float state_delta, float model_uncertainty,
                 int& frames_converged, bool& model_stable, const Params& p)
{
    if (state_delta < p.state_eps)
        frames_converged = std::min(frames_converged + 1, p.k_stable);
    else
    {
        frames_converged = 0;
        model_stable     = false;
    }

    G.add_or_modify_attrib_local<model_uncertainty_att>(node, model_uncertainty);

    const bool want = frames_converged >= p.k_stable;
    if (want == model_stable)
        return false;                       // no edge ⇒ no graph write, exactly as before
    model_stable = want;
    G.add_or_modify_attrib_local<model_stable_att>(node, want);
    G.update_node(node);
    return true;
}

}  // namespace rc::converge
