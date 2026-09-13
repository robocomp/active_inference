/*
 * pragmatic_affordance.h — a NAMED, PRECONDITIONED affordance node. SHARED.
 *
 * ★HOW THIS DIFFERS FROM rc::ObjectAffordance, which is the other producer in this directory.
 * ObjectAffordance publishes ONE node per object and that node means exactly one thing: "stand here and
 * look at me, it will shrink my uncertainty by ΔH nats". It is EPISTEMIC, it always exists while the
 * object does, and its value is an entropy reduction the producer computes from its own Σ.
 *
 * This one publishes SEVERAL nodes per object, each with a NAME the producer chooses ("approach", "open",
 * "cross"), and each PRECONDITIONED: it is on the wire only while the producer believes the world is in a
 * state where the action is possible at all. A door you cannot reach cannot be opened; a door that is shut
 * cannot be crossed. Those are not low-value offers, they are NOT OFFERS, and the difference matters to a
 * consumer that ranks whatever it finds: an impossible offer with a plausible price is worse than silence.
 *
 * Everything else is deliberately IDENTICAL — same node type (`affordance`), same parent edge
 * (`has_intention`), same wire protocol (epistemic_target_x_m/_y_m/_yaw_rad + epistemic_gain +
 * active/epistemic_pending + the aff_* Contract + aff_outcome). The executor is generic and stays generic:
 * it reads a Contract and runs it, and it never learns what a door is. `aff_kind` is a LABEL for humans,
 * logs and dashboards — never a dispatch key.
 *
 * ─── THE COMPLETION HALF IS THE SHARED PROTOCOL, NOT A LOCAL INVENTION ────────────────────────────
 *
 * ★★★A CLAIM IS AN EDGE AND AN EPOCH, NOT A BOOLEAN. This class first shipped reading only
 * `active` + `epistemic_pending`, which is precisely the return channel the affordance protocol
 * upgrade of 2026-08-23 existed to replace: content flowed one way and what came back was a bool, so
 * the consumer could say THAT it was executing and never WHAT. Two agents then held a variable each
 * called "the target" whose values differed, with NO TERM IN EITHER STATE able to reveal it — measured
 * live, the producer reporting a 4.92 m distance to its own latest publication while the consumer had
 * committed to a pose 2.55 m away, both correct and both blind.
 *
 * So the completion contract here is the fleet's, via rc::AffordanceManager:
 *   · PRODUCER (this class) writes `epistemic_target_epoch`, bumped ONLY on a CONTENT change — a new
 *     pose — and never on a re-arm. "A value that changes for PROTOCOL reasons is not a new proposal."
 *   · CONSUMER writes an edge IT owns: <consumer agent node> --[executing]--> <affordance node>,
 *     carrying executing_epoch + the pose it is ACTUALLY driving to after its own repairs.
 *   · This class reads that edge (`AffordanceManager::read_executing`) and hands the producer the
 *     CLAIMED pose, so a side effect fires against what is being executed rather than against our
 *     latest publication. That is protocol invariant 2 and it is the one that bites: an `open` request
 *     issued against a pose the consumer is not driving to is a request about a different doorway.
 *   · executing_epoch < epistemic_target_epoch ⇒ the consumer is on a STALE proposal. Invariant 4, and
 *     the term the boolean protocol lacked. Surfaced, never silently ignored.
 *   · The edge also self-cleans on a crash: cortex deletes a dead agent's node and its edges, so a
 *     consumer that dies cannot leave a claim standing — which `active` could and did.
 *
 * ⚠DEGRADATION IS NOT OPTIONAL. A pre-rollout consumer writes no edge, and "nobody claims it" and "a
 * consumer that cannot say" are DIFFERENT readings. `read_executing` reports which via `claimed`, and
 * this class falls back to the boolean pair on nullopt rather than concluding nothing is executing.
 *
 * ─── THE THREE RULES THIS CLASS EXISTS TO ENFORCE ──────────────────────────────────────────────────
 *
 * 1. ★A NODE THE CONSUMER IS EXECUTING IS NEVER REMOVED. The precondition can go false while the
 *    manoeuvre is in flight — the robot rolls back out of the actuation zone, the leaf swings shut
 *    again — and deleting the node then strands the consumer mid-execution holding a claim on something
 *    that no longer exists. That is the stranded-`Completed`-for-ever defect the fleet already paid for
 *    once (TIER B: four agents left affordances Completed FOR EVER). While Executing this class only
 *    logs; the node is retired after the claim resolves, by the consumer's own terminal transition or
 *    by the contract's timeout. The consumer owns the claim; the producer owns the offer.
 *
 * 2. ★THE PRECONDITION IS A PROBABILITY AND THE OFFER IS A DECISION ON IT, debounced. A precondition
 *    computed per cycle from a live belief will sit near its boundary and chatter, and a node that
 *    appears and disappears at 10 Hz is a CRDT write storm plus a consumer that can never finish
 *    selecting. So `Offer::p` is the probability, `offer_prob` / `withdraw_prob` are the one decision
 *    boundary stated as the probability it actually is (exactly how Existence.RemovalProb is written),
 *    and `stable_cycles` is how long the decision must hold. Two boundaries, not one, because a
 *    Schmitt band is what stops chatter without either side needing a timer.
 *
 * 3. ★"I COULD NOT COMPUTE THE PRECONDITION" IS NOT "THE PRECONDITION IS FALSE". A cycle with no
 *    robot pose, no leaf-angle posterior, no fitted aperture, tells you nothing about the world. Pass
 *    `Offer::known = false` on those cycles and a standing offer HOLDS rather than being withdrawn —
 *    the same distinction rc::exist draws between "looked at and empty" and "never looked at", and the
 *    same one ObjectAffordance::hold_offered() was added for. The withdrawal must be a MEASUREMENT.
 *
 * MAIN-THREAD ONLY (graph reads and writes throughout).
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <dsr/api/dsr_api.h>

#include "../affordance_manager/affordance_manager.h"     // rc::AffordanceManager::ExecutingClaim (reader)
#include "../affordance_protocol/affordance_protocol.h"   // rc::affordance::Contract / Outcome

namespace rc::pragmatic
{

// What the producer offers for ONE named affordance on ONE cycle.
struct Offer
{
    // ── the precondition ──────────────────────────────────────────────────────────────────────────
    // `known == false` ⇒ this cycle carries NO information about the precondition (see rule 3): `p` is
    // ignored, a standing offer holds, and an absent one stays absent.
    bool  known = false;
    float p     = 0.0f;      // P(the precondition holds). 1.0 for an unconditional affordance.

    // ── the proposal ──────────────────────────────────────────────────────────────────────────────
    float x_m = 0.0f, y_m = 0.0f, yaw_rad = 0.0f;   // the pose the executor is asked to reach
    float value = 0.0f;                              // what the consumer ranks on (same wire field as ΔH)
    rc::affordance::Contract contract{};             // how the consumer completes it

    // Why the precondition reads the way it does, in words. Logged on every state change — a withdrawn
    // affordance with no stated reason is indistinguishable from a producer that crashed.
    std::string why;

    bool is_finite() const;
};

class PragmaticAffordance
{
public:
    // absent → offered → executing → (completed | aborted) → offered/absent
    enum class State : std::uint8_t { absent, offered, executing, completed, aborted };

    // The two decision boundaries and the debounce. See rule 2. Defaults are a wide Schmitt band and a
    // short hold: they suppress chatter without adding perceptible latency at 10 Hz.
    struct Policy
    {
        float offer_prob    = 0.60f;   // offer once P(precondition) rises above this …
        float withdraw_prob = 0.35f;   // … and withdraw once it falls below THIS (hysteresis, not one line)
        int   stable_cycles = 3;       // … and the decision has held for this many MEASURED cycles
    };

    PragmaticAffordance() = default;

    /// Late initialisation — once, after the parent object's DSR node exists.
    /// `kind` is the affordance's name in the producer's vocabulary ("approach", "open", "cross"). It
    /// becomes both the node-name suffix (`aff_<parent>_<kind>`) and the `aff_kind` label.
    void init(std::shared_ptr<DSR::DSRGraph> G,
              std::uint64_t parent_node_id,
              std::string   parent_node_name,
              std::string   kind,
              // NO DEFAULT ARGUMENT. `= {}` here made gcc reject the declaration outright ("could not
              // convert '<brace-enclosed initializer list>()' ... to Policy") while completing the
              // enclosing class, and because this unit is in ai_common_affordance that broke the build
              // for every agent linking it, not just the one being edited. The policy is a decision the
              // producer must state anyway, so requiring it is the better contract as well as the one
              // that compiles.
              Policy        policy);

    /// ONE call per compute cycle, always — including on cycles where nothing is known (pass
    /// `Offer{.known = false}`). Creates, refreshes, or retires the node per the rules above.
    void update(const Offer& offer);

    /// Read the consumer-owned protocol flags and terminal outcome off our node. POLLED, not pushed —
    /// see rc::poll_protocol for the event-loop starvation that settled that question for the fleet.
    void poll();

    /// ★THE ONE-SHOT THE PRODUCER ACTS ON. True exactly once per claim: the cycle on which the consumer
    /// took this affordance. A producer whose affordance has a side effect in the world (asking a provider
    /// to move a door) fires it HERE, so the effect happens once per claim and not once per cycle while
    /// the claim stands.
    [[nodiscard]] bool consume_claim();

    /// WHAT the consumer says it is executing — the pose it is actually driving to after its own
    /// repairs, plus the epoch it accepted. nullopt when nothing claims this affordance OR when the
    /// consumer is pre-rollout and cannot say; `claimed()` distinguishes those, and they are not the
    /// same fact. ★A producer with a side effect must act on THIS, not on its own last publication.
    [[nodiscard]] const std::optional<rc::AffordanceManager::ExecutingClaim>& claim() const
    { return claim_; }
    /// True when SOMETHING claims this affordance — from the edge if the consumer writes one, else
    /// from the legacy boolean pair. The fallback is what keeps a pre-rollout consumer working.
    [[nodiscard]] bool claimed() const { return claimed_; }
    /// The consumer accepted an OLDER proposal than the one now published (invariant 4). The producer
    /// owns preemption: wait, or withdraw — but it must not measure progress against the new one.
    [[nodiscard]] bool claim_is_stale() const { return claim_stale_; }
    /// The epoch currently published on our node. Bumps only on a content change.
    [[nodiscard]] int  epoch() const { return epoch_; }

    /// True exactly once per terminal transition; `outcome()` says which. A producer needs this to tell
    /// "the predicate held" from "we gave up" — those are different beliefs (see aff_outcome in cortex).
    [[nodiscard]] bool consume_completion();

    /// Delete the node now, whatever its state. For instance teardown ONLY (the object is going away):
    /// the per-cycle path must go through update(), which honours rule 1.
    void remove();

    /// Call from del_node_slot: somebody else deleted our node.
    void on_node_deleted(std::uint64_t id);

    [[nodiscard]] State                   state()      const { return state_; }
    [[nodiscard]] std::uint64_t           node_id()    const { return node_id_; }
    [[nodiscard]] rc::affordance::Outcome outcome()    const { return outcome_; }
    [[nodiscard]] bool                    is_offered() const { return node_id_ != 0; }
    [[nodiscard]] const std::string&      kind()       const { return kind_; }
    [[nodiscard]] float                   last_p()     const { return last_p_; }

    static std::string_view state_name(State s);

private:
    void create_node(const Offer& offer);
    void refresh_node(const Offer& offer);
    void write_proposal(DSR::Node& node, const Offer& offer, bool with_flags);
    void refresh_edge();
    void reset_local();

    std::shared_ptr<DSR::DSRGraph> G_;
    std::uint64_t parent_node_id_ = 0;
    std::string   parent_node_name_;
    std::string   kind_;
    std::string   node_name_;
    Policy        policy_{};

    std::uint64_t node_id_ = 0;
    State         state_   = State::absent;
    rc::affordance::Outcome outcome_ = rc::affordance::Outcome::None;

    std::optional<rc::AffordanceManager::ExecutingClaim> claim_;   // what the consumer says it drives to
    bool  claimed_       = false;  // something claims it (edge, or the legacy bool pair)
    bool  claim_stale_   = false;  // executing_epoch < epistemic_target_epoch (invariant 4)
    int   epoch_         = 0;      // as last written by us; the NODE is the truth, this is a mirror
    int   want_streak_   = 0;      // consecutive MEASURED cycles saying "offer it"
    int   unwant_streak_ = 0;      // consecutive MEASURED cycles saying "withdraw it"
    float last_p_        = 0.0f;   // last MEASURED probability (for the log / the dashboard)
    bool  claim_pending_ = false;  // a claim the producer has not consumed yet
    bool  completion_pending_ = false;
    std::string last_why_;
};

}  // namespace rc::pragmatic
