/*
 * pragmatic_affordance.cpp — create / refresh / retire a named, preconditioned affordance node, and
 * track the consumer-owned protocol state on it. See the header for the three rules this enforces.
 */

#include "pragmatic_affordance.h"

#include <cmath>
#include <format>
#include <print>
#include <utility>

#include "../graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc::pragmatic
{

bool Offer::is_finite() const
{
    return std::isfinite(x_m) and std::isfinite(y_m) and std::isfinite(yaw_rad)
           and std::isfinite(value) and std::isfinite(p);
}

std::string_view PragmaticAffordance::state_name(State s)
{
    switch (s)
    {
        case State::absent:    return "absent";
        case State::offered:   return "offered";
        case State::executing: return "executing";
        case State::completed: return "completed";
        case State::aborted:   return "aborted";
    }
    return "unknown";
}

void PragmaticAffordance::init(std::shared_ptr<DSR::DSRGraph> G,
                               std::uint64_t parent_node_id,
                               std::string   parent_node_name,
                               std::string   kind,
                               Policy        policy)
{
    G_                = std::move(G);
    parent_node_id_   = parent_node_id;
    parent_node_name_ = std::move(parent_node_name);
    kind_             = std::move(kind);
    policy_           = policy;
    // ★The name keeps the `aff_<object-prefix>` shape on purpose: rc::owned::remove_stale_affordances
    // identifies an agent's affordances by that prefix, and it is the ORPHAN-SAFE half of the sweep (it
    // still fires after the parent object node is gone). A name like "approach_door_1" would have leaked
    // on every crash, with nothing else in the fleet ever deleting it.
    node_name_        = "aff_" + parent_node_name_ + "_" + kind_;
}

void PragmaticAffordance::update(const Offer& offer)
{
    if (not G_ or parent_node_id_ == 0)
        return;

    poll();   // learn the consumer's current verdict BEFORE deciding anything about the node

    // ── Rule 3: an unmeasured cycle is not evidence ───────────────────────────────────────────────
    // Neither streak advances and a standing offer is left exactly as it is. Its POSE is not refreshed
    // either: a target computed from inputs we do not have this cycle would be a guess published as a
    // destination.
    if (not offer.known)
        return;

    if (not offer.is_finite())
    {
        std::print("[{}] REFUSING a non-finite offer (p={} x={} y={} yaw={} value={})\n",
                   node_name_, offer.p, offer.x_m, offer.y_m, offer.yaw_rad, offer.value);
        return;
    }

    last_p_ = offer.p;
    const bool want   = offer.p >= policy_.offer_prob;
    const bool unwant = offer.p <  policy_.withdraw_prob;
    // Inside the Schmitt band neither streak grows, so a probability hovering there holds whatever
    // decision was last taken rather than flipping on noise.
    want_streak_   = want   ? want_streak_   + 1 : 0;
    unwant_streak_ = unwant ? unwant_streak_ + 1 : 0;

    if (node_id_ == 0)
    {
        if (want_streak_ >= policy_.stable_cycles)
            create_node(offer);
        return;
    }

    // ── Rule 1: never pull a node the consumer is executing ───────────────────────────────────────
    if (state_ == State::executing)
    {
        if (unwant_streak_ == policy_.stable_cycles)   // == so this says itself once per episode
            std::print("[{}] precondition has FAILED (p={:.3f}: {}) while the consumer is EXECUTING — "
                       "holding the node; the claim is the consumer's to resolve\n",
                       node_name_, offer.p, offer.why.empty() ? "no reason given" : offer.why);
        return;   // and do not rewrite the target under a live claim
    }

    if (unwant_streak_ >= policy_.stable_cycles)
    {
        std::print("[{}] WITHDRAWN after {} measured cycles below P={:.2f} (p={:.3f}: {})\n",
                   node_name_, unwant_streak_, policy_.withdraw_prob, offer.p,
                   offer.why.empty() ? "no reason given" : offer.why);
        remove();
        return;
    }

    refresh_node(offer);
}

void PragmaticAffordance::create_node(const Offer& offer)
{
    DSR::Node node = DSR::Node::create<affordance_node_type>(node_name_);
    G_->add_or_modify_attrib_local<level_att> (node, 4);
    G_->add_or_modify_attrib_local<parent_att>(node, parent_node_id_);
    // ★FOUR SIBLINGS UNDER ONE OBJECT NEED FOUR PLACES. ObjectAffordance plants its single node at a
    // fixed (300,200); three more at the same spot draw exactly on top of each other and the graph view
    // shows one affordance where there are four — which is the opposite of what a graph view is for.
    // The lane is derived from the kind so it is stable across restarts rather than birth-order.
    std::size_t lane = 0;
    for (const char ch : kind_) lane = lane * 31 + static_cast<unsigned char>(ch);
    G_->add_or_modify_attrib_local<pos_x_att> (node, 300.f + 90.f * static_cast<float>(lane % 4));
    G_->add_or_modify_attrib_local<pos_y_att> (node, 260.f + 70.f * static_cast<float>((lane / 4) % 3));
    G_->add_or_modify_attrib_local<aff_kind_att>(node, kind_);
    write_proposal(node, offer, /*with_flags=*/true);
    rc::affordance::write_contract(*G_, node, offer.contract);
    // ★NO ViewpointConstraint. That constraint says "resolve a collision-free pose on one of these object
    // faces yourself" — an EPISTEMIC request, where any view that shrinks Σ will do. A pragmatic pose is
    // not substitutable: the actuation zone in front of a door, or the far side of an aperture, is the
    // place or it is nothing. Leaving it unwritten is what makes the published target authoritative.
    rc::provenance::stamp_creation(*G_, node);

    const auto id = G_->insert_node(node);
    if (not id.has_value())
    {
        std::print("[{}] FAILED to insert the affordance node\n", node_name_);
        return;
    }
    node_id_ = id.value();
    state_   = State::offered;
    outcome_ = rc::affordance::Outcome::None;
    claim_pending_ = completion_pending_ = false;
    refresh_edge();
    last_why_ = offer.why;
    std::print("[{}] OFFERED id={} P={:.3f} target=({:.2f},{:.2f}) yaw={:+.2f} value={:.3f} policy={} ({})\n",
               node_name_, node_id_, offer.p, offer.x_m, offer.y_m, offer.yaw_rad, offer.value,
               rc::affordance::to_string(offer.contract.policy),
               offer.why.empty() ? "unconditional" : offer.why);
}

void PragmaticAffordance::refresh_node(const Offer& offer)
{
    auto node_opt = G_->get_node(node_id_);
    if (not node_opt.has_value())
    {
        std::print("[{}] node vanished from the graph — will re-offer next cycle\n", node_name_);
        reset_local();
        return;
    }
    auto& node = node_opt.value();

    // A terminal state the producer has not consumed yet is left alone: rewriting active/pending here
    // would erase the consumer's own verdict before anyone read it.
    const bool re_arm = (state_ == State::completed or state_ == State::aborted);
    write_proposal(node, offer, /*with_flags=*/re_arm or state_ == State::offered);
    if (re_arm and not completion_pending_)
    {
        // Re-offer: the precondition still holds, so the action is still possible. The CONTRACT is
        // rewritten too — a completed node's contract may have been for a different situation.
        rc::affordance::write_contract(*G_, node, offer.contract);
        G_->add_or_modify_attrib_local<aff_outcome_att>(node, std::string{});
        state_   = State::offered;
        outcome_ = rc::affordance::Outcome::None;
        std::print("[{}] RE-OFFERED (precondition still holds, P={:.3f})\n", node_name_, offer.p);
    }
    G_->update_node(node);
    refresh_edge();
    if (offer.why != last_why_)
    {
        last_why_ = offer.why;
        std::print("[{}] precondition now: {} (P={:.3f})\n", node_name_,
                   offer.why.empty() ? "unconditional" : offer.why, offer.p);
    }
}

void PragmaticAffordance::write_proposal(DSR::Node& node, const Offer& offer, bool with_flags)
{
    // ★THE EPOCH BUMPS ON A CONTENT CHANGE AND ONLY ON A CONTENT CHANGE. It is the one field that
    // distinguishes "here is a NEW proposal" from "I am re-arming the same one", and the consumer's
    // executing_epoch is compared against it to detect that it is driving to something we have since
    // replaced (invariant 4). A bump on every refresh would make every standing offer look stale and
    // every consumer look wrong; no bump at all would make a replaced pose undetectable — which is the
    // blindness this whole mechanism replaced.
    // ★Compared against what is ON THE NODE, not against our member: the node is the shared truth and
    // after a restart the two disagree. Mirrors AffordanceManager::publish_target exactly, including
    // the max(1, ...) so that "has an epoch" and "epoch is the default" are never the same reading.
    {
        const auto ox = G_->get_attrib_by_name<epistemic_target_x_m_att>(node);
        const auto oy = G_->get_attrib_by_name<epistemic_target_y_m_att>(node);
        const auto ow = G_->get_attrib_by_name<epistemic_target_yaw_rad_att>(node);
        const int  oe = G_->get_attrib_by_name<epistemic_target_epoch_att>(node).value_or(0);
        const bool content_changed =
            not ox.has_value() or not oy.has_value() or not ow.has_value()
            or std::abs(ox.value() - offer.x_m)     > 1e-4f
            or std::abs(oy.value() - offer.y_m)     > 1e-4f
            or std::abs(ow.value() - offer.yaw_rad) > 1e-4f;
        epoch_ = content_changed ? std::max(1, oe + 1) : std::max(1, oe);
        G_->add_or_modify_attrib_local<epistemic_target_epoch_att>(node, epoch_);
    }
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (node, offer.x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (node, offer.y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, offer.yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (node, offer.value);
    if (with_flags)
    {
        G_->add_or_modify_attrib_local<active_att>           (node, false);
        G_->add_or_modify_attrib_local<epistemic_pending_att>(node, true);
    }
}

void PragmaticAffordance::poll()
{
    if (not G_ or node_id_ == 0)
        return;
    auto node_opt = G_->get_node(node_id_);
    if (not node_opt.has_value())
    {
        reset_local();
        return;
    }
    const auto& node = node_opt.value();
    const bool active  = G_->get_attrib_by_name<active_att>(node).value_or(false);
    // Default TRUE: a node whose flag we have not read yet must never be mistaken for a completed one.
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
    epoch_ = G_->get_attrib_by_name<epistemic_target_epoch_att>(node).value_or(epoch_);

    // ── WHAT the consumer is executing, from the edge IT owns ─────────────────────────────────────
    // ⚠`claimed` distinguishes "nobody claims it" from "a pre-rollout consumer that cannot say", and
    // those are different facts. On nullopt we fall back to the boolean pair rather than concluding
    // nothing is executing — a protocol that half-breaks during a rollout is worse than the old one.
    bool edge_claimed = false;
    claim_ = rc::AffordanceManager::read_executing(G_, node_id_, &edge_claimed);
    claimed_ = edge_claimed or (active and pending);
    // Invariant 4: the consumer accepted an OLDER proposal than the one now on the node. Detectable by
    // both sides, which is the whole point; the producer owns what to do about it.
    const bool was_stale = claim_stale_;
    claim_stale_ = claim_.has_value() and epoch_ > 0 and claim_->epoch < epoch_;
    if (claim_stale_ and not was_stale)
        std::print("[{}] consumer is executing epoch {} while we now publish {} — it is driving to "
                   "({:.2f},{:.2f}) yaw {:+.2f}, NOT to our latest proposal\n",
                   node_name_, claim_->epoch, epoch_, claim_->x, claim_->y, claim_->yaw);

    if (claimed_ and pending)
    {
        if (state_ != State::executing)
        {
            state_ = State::executing;
            claim_pending_ = true;   // consume_claim() hands this to the producer exactly once
            if (claim_.has_value())
                std::print("[{}] → EXECUTING (consumer claimed epoch {}, driving to ({:.2f},{:.2f}) "
                           "yaw {:+.2f})\n", node_name_, claim_->epoch, claim_->x, claim_->y, claim_->yaw);
            else
                std::print("[{}] → EXECUTING (consumer claimed, but writes no `executing` edge — "
                           "pre-rollout peer, so WHAT it is driving to is unknown to us)\n", node_name_);
        }
        return;
    }
    if (not claimed_ and pending)
    {
        state_ = State::offered;
        return;
    }
    if (not claimed_ and not pending)
    {
        if (state_ != State::completed)
        {
            state_   = State::completed;
            outcome_ = rc::affordance::read_outcome(node);
            completion_pending_ = true;
            std::print("[{}] → COMPLETED, outcome = {}\n", node_name_,
                       rc::affordance::to_string(outcome_));
        }
        return;
    }
    if (state_ != State::aborted)
    {
        state_   = State::aborted;
        outcome_ = rc::affordance::read_outcome(node);
        completion_pending_ = true;
        std::print("[{}] → ABORTED (invalid protocol state active=1 pending=0)\n", node_name_);
    }
}

bool PragmaticAffordance::consume_claim()
{
    return std::exchange(claim_pending_, false);
}

bool PragmaticAffordance::consume_completion()
{
    return std::exchange(completion_pending_, false);
}

void PragmaticAffordance::remove()
{
    if (not G_ or node_id_ == 0)
        return;
    const auto id = node_id_;
    G_->delete_node(id);
    std::print("[{}] node {} deleted (state was {})\n", node_name_, id, state_name(state_));
    reset_local();
}

void PragmaticAffordance::on_node_deleted(std::uint64_t id)
{
    if (node_id_ == 0 or id != node_id_)
        return;
    std::print("[{}] node deleted externally (state was {}) → absent\n",
               node_name_, state_name(state_));
    reset_local();
}

void PragmaticAffordance::refresh_edge()
{
    auto edge = DSR::Edge::create<has_intention_edge_type>(parent_node_id_, node_id_);
    G_->insert_or_assign_edge(edge);
}

void PragmaticAffordance::reset_local()
{
    node_id_ = 0;
    state_   = State::absent;
    outcome_ = rc::affordance::Outcome::None;
    claim_pending_ = completion_pending_ = false;
    claim_.reset();
    claimed_ = claim_stale_ = false;
    epoch_ = 0;   // the node is gone, so our mirror of ITS epoch is meaningless
    // The streaks are NOT reset: the precondition's history is a fact about the world, independent of
    // whether our node happens to exist. Clearing them would add `stable_cycles` of latency to every
    // re-offer after the consumer completed one.
}

}  // namespace rc::pragmatic
