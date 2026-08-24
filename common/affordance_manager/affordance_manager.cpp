#include <chrono>
#include <source_location>
#include <locale>
#include <format>
#include <fstream>
#include "affordance_manager.h"

#include <dsr/api/dsr_api.h>
#include "../affordance_protocol/affordance_protocol.h"   // read_viewpoint (object-relative viewpoint debug read)

#include <QDebug>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <print>
#include <sstream>
#include <utility>
#include "../graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace
{
constexpr auto kAffordanceEdgeTypeStr = std::string_view("has_intention");
[[maybe_unused]] inline const bool kAffordanceEdgeTypeRegistered = edge_types::register_type(kAffordanceEdgeTypeStr);
using affordance_edge_type = EdgeType<kAffordanceEdgeTypeStr>;

bool finite_target_values(float tx, float ty, float yaw, float gain)
{
    return std::isfinite(tx) && std::isfinite(ty) && std::isfinite(yaw) && std::isfinite(gain);
}

// Dedup KEY: identity only (name + parent type). The log is throttled by comparing this against the
// previous key, so it must NOT embed gain/shape/pending — those wobble every cycle and would reprint.
std::string make_affordance_key(const rc::AffordanceManager::Target &target)
{
    std::ostringstream out;
    out << target.node_name << '|' << target.parent_node_type;
    return out.str();
}

// Display LINE: printed once per identity switch (keyed by make_affordance_key), but carries the live
// gain so the value at the moment of each switch is visible.
std::string make_affordance_debug_report(const rc::AffordanceManager::Target &target)
{
    std::ostringstream out;
    out << "[aff-select] -> '" << target.node_name << "' (" << target.parent_node_type << ")"
        << " gain=" << target.epistemic_gain << " nats"
        << " pending=" << (target.epistemic_pending ? 1 : 0);
    return out.str();
}
}

namespace rc
{

AffordanceManager::AffordanceManager(std::string managed_node_name)
    : managed_node_name_(std::move(managed_node_name))
{
}

void AffordanceManager::set_selection_params(float lambda_cost, float switch_margin)
{
    select_lambda_cost_ = std::max(0.0f, lambda_cost);
    select_switch_margin_ = std::max(0.0f, switch_margin);
}

void AffordanceManager::set_room_gain_scale(float scale)
{
    select_room_gain_scale_ = std::clamp(scale, 0.0f, 1.0f);
}

std::string_view AffordanceManager::state_name(State state)
{
    switch (state)
    {
        case State::Idle: return "Idle";
        case State::Searching: return "Searching";
        case State::Following: return "Following";
        case State::Claiming: return "Claiming";
        case State::Completing: return "Completing";
    }
    return "Unknown";
}

std::string_view AffordanceManager::protocol_state_name(ProtocolState state)
{
    switch (state)
    {
        case ProtocolState::Missing: return "Missing";
        case ProtocolState::Offered: return "Offered";
        case ProtocolState::Executing: return "Executing";
        case ProtocolState::Completed: return "Completed";
        case ProtocolState::Invalid: return "Invalid";
    }
    return "Unknown";
}

AffordanceManager::ProtocolState AffordanceManager::decode_protocol_state(bool active, bool pending)
{
    if (!active && pending)
        return ProtocolState::Offered;
    if (active && pending)
        return ProtocolState::Executing;
    if (!active && !pending)
        return ProtocolState::Completed;
    return ProtocolState::Invalid;
}

void AffordanceManager::transition_to(State next, std::string_view reason, std::uint64_t node_id, std::string_view node_name)
{
    (void)reason;
    (void)node_id;
    (void)node_name;
    if (state_ == next)
        return;

    state_ = next;
}

void AffordanceManager::log_observation(std::uint64_t node_id,
                                        std::string_view node_name,
                                        bool active,
                                        bool pending,
                                        float x,
                                        float y,
                                        float yaw,
                                        float gain)
{
    if (last_observed_active_.has_value() && last_observed_pending_.has_value()
        && last_observed_active_.value() == active && last_observed_pending_.value() == pending)
        return;

    (void)node_id;
    (void)node_name;
    (void)x;
    (void)y;
    (void)yaw;
    (void)gain;
    last_observed_active_ = active;
    last_observed_pending_ = pending;
}

void AffordanceManager::reset_observation()
{
    last_observed_active_.reset();
    last_observed_pending_.reset();
}

void AffordanceManager::reset()
{
    managed_node_id_ = 0;
    waiting_completion_ = false;
    completion_detected_ = false;
    last_managed_active_ = false;
    last_managed_pending_ = true;

    current_affordance_id_ = 0;
    current_affordance_name_.clear();
    // NOT selected_target_debug_report_: it is a PRINT-dedup key, not affordance state. Clearing it here
    // made [aff-select] re-print on every re-selection of the SAME affordance — and clear_current() runs
    // once per cycle whenever there is no target (or a mission owns the base), so the "print only on
    // change" intent degenerated into per-cycle spam.
    reset_observation();
    transition_to(State::Idle, "reset called");
}

bool AffordanceManager::consume_completion_event()
{
    if (!completion_detected_)
        return false;

    completion_detected_ = false;
    return true;
}

std::optional<AffordanceManager::Target> AffordanceManager::read_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                                                        const DSR::Node &node) const
{
    const auto tx = graph->get_attrib_by_name<epistemic_target_x_m_att>(node);
    const auto ty = graph->get_attrib_by_name<epistemic_target_y_m_att>(node);
    const auto yaw = graph->get_attrib_by_name<epistemic_target_yaw_rad_att>(node);
    const auto gain = graph->get_attrib_by_name<epistemic_gain_att>(node);
    const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node);
    const auto active = graph->get_attrib_by_name<active_att>(node);

    if (!tx.has_value() || !ty.has_value() || !yaw.has_value() || !gain.has_value() || !pending.has_value() || !active.has_value())
        return std::nullopt;

    if (!finite_target_values(tx.value(), ty.value(), yaw.value(), gain.value()))
        return std::nullopt;

    Target target;
    target.node_id = node.id();
    target.node_name = node.name();
    target.room_pos = Eigen::Vector2f(tx.value(), ty.value());
    target.yaw_rad = yaw.value();
    target.epistemic_gain = gain.value();
    target.epistemic_pending = pending.value();
    // The contract's policy, straight off the node. Absent ⇒ Reach, which is how a contract-less node
    // has always been executed, so this changes nothing for anyone who never writes one.
    if (const auto pol = graph->get_attrib_by_name<aff_policy_att>(node); pol.has_value())
        target.policy = rc::affordance::policy_from(pol.value());
    const auto parent_id = graph->get_attrib_by_name<parent_att>(node);
    if (parent_id.has_value())
    {
        target.parent_node_id = parent_id.value();
        if (const auto parent_node = graph->get_node(target.parent_node_id); parent_node.has_value())
        {
            target.parent_node_name = parent_node->name();
            target.parent_node_type = parent_node->type();
            const auto width_attr = graph->get_attrib_by_name<width_m_att>(parent_node.value());
            const auto depth_attr = graph->get_attrib_by_name<depth_m_att>(parent_node.value());
            if (width_attr.has_value() && depth_attr.has_value()
                && std::isfinite(width_attr.value()) && std::isfinite(depth_attr.value())
                && width_attr.value() > 0.f && depth_attr.value() > 0.f)
            {
                target.shape_width_m = width_attr.value();
                target.shape_depth_m = depth_attr.value();
                target.has_shape = true;
            }
        }
    }
    return target;
}

std::optional<DSR::Node> AffordanceManager::get_managed_node(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    if (!graph)
        return std::nullopt;

    if (managed_node_id_ != 0)
    {
        if (auto node_opt = graph->get_node(managed_node_id_); node_opt.has_value())
        {
            if (managed_node_name_.empty() || node_opt->name() == managed_node_name_)
                return node_opt.value();
        }
        managed_node_id_ = 0;
    }

    if (managed_node_name_.empty())
        return std::nullopt;

    if (auto node_opt = graph->get_node(managed_node_name_); node_opt.has_value())
    {
        managed_node_id_ = node_opt->id();
        return node_opt.value();
    }

    return std::nullopt;
}

bool AffordanceManager::read_managed_flags(const std::shared_ptr<DSR::DSRGraph> &graph, bool &active, bool &pending)
{
    active = false;
    pending = true;

    const auto node_opt = get_managed_node(graph);
    if (!node_opt.has_value())
        return false;

    const auto active_opt = graph->get_attrib_by_name<active_att>(node_opt.value());
    const auto pending_opt = graph->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
    active = active_opt.value_or(false);
    pending = pending_opt.value_or(true);
    return true;
}

void AffordanceManager::monitor_execution(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    bool active = false;
    bool pending = true;
    if (!read_managed_flags(graph, active, pending))
    {
        waiting_completion_ = false;
        last_managed_active_ = false;
        last_managed_pending_ = true;
        return;
    }

    if (active != last_managed_active_ || pending != last_managed_pending_)
    {
        last_managed_active_ = active;
        last_managed_pending_ = pending;
    }

    // ★★★ LEVEL-TRIGGERED, NOT EDGE-TRIGGERED. This used to require OBSERVING the node in
    // (active && pending) before it would accept a later completion — an edge signal implemented by
    // polling, which is a race by construction: it is correct only while the consumer is slow enough
    // to still be in that state when the producer next looks. It also CLEARED the wait on `!active`,
    // i.e. on the ordinary Offered state, so the flag survived only inside the claim window.
    //
    // Measured 2026-08-18: once the consumer began claiming and finishing inside ONE of its 50 ms
    // cycles (a standpoint repaired onto the robot's own pose, refused immediately), the producer's
    // 20 Hz poll never caught the window. Zero completions were detected in 549 s while the consumer
    // was completing continuously — so the producer never cleared its target, kept re-proposing the
    // same cell, publish_target declined it as unchanged-and-Completed, nothing was ever Offered
    // again, and both agents wedged. The 3 s dwell had been hiding this for as long as it existed.
    //
    // `pending` is the LEVEL: the producer sets it when it arms the node, the consumer clears it when
    // it finishes. Both sides own one edge of it and neither can miss the other's. `active` is the
    // consumer's claim and is none of the producer's business — waiting on it is what created the
    // race. publish_target() also sets waiting_completion_ directly, so a completion that happens
    // entirely between two polls is still seen.
    if (pending)
    {
        waiting_completion_ = true;
        pending_seen_ = true;          // the arming is real; a not-pending reading now means something
    }
    else if (waiting_completion_ and not pending_seen_)
    {
        // Armed, then read not-pending WITHOUT ever having been seen pending: that is a stale read of
        // our own write, not a completion. Say so once rather than silently inventing an arrival.
        std::print("[affordance] ignoring a not-pending reading on '{}' that was never seen pending — "
                   "stale read of our own arming, not a completion\n", managed_node_name_);
        std::fflush(stdout);
        waiting_completion_ = false;
    }
    else if (waiting_completion_)
    {
        waiting_completion_ = false;
        pending_seen_ = false;
        completion_detected_ = true;
        // Latch WHY it ended, at the same moment the completion is detected. Read here rather than
        // when the caller gets round to consume_completion_event(): by then the producer may have
        // re-published the affordance, which overwrites aff_outcome with the new attempt's value.
        last_outcome_ = rc::affordance::Outcome::None;
        if (const auto node = graph->get_node(managed_node_name_); node.has_value())
            last_outcome_ = rc::affordance::read_outcome(node.value());
    }
}

bool AffordanceManager::is_executing(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    bool active = false;
    bool pending = true;
    return read_managed_flags(graph, active, pending) && decode_protocol_state(active, pending) == ProtocolState::Executing;
}

bool AffordanceManager::publish_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                       std::uint64_t parent_id,
                                       float tx,
                                       float ty,
                                       float yaw,
                                       float gain,
                                       const std::function<void()> &on_node_inserted,
                                       const std::function<void()> &on_edge_inserted)
{
    if (!graph)
        return false;
    if (managed_node_name_.empty())
    {
        qWarning() << "AffordanceManager publish_target requires a managed node name";
        return false;
    }

    auto parent_node_opt = graph->get_node(parent_id);
    if (!parent_node_opt.has_value())
        return false;

    auto node_opt = get_managed_node(graph);
    if (!node_opt.has_value())
    {
        DSR::Node aff_node = DSR::Node::create<affordance_node_type>(managed_node_name_);
        graph->add_or_modify_attrib_local<level_att>(aff_node, 3);
        graph->add_or_modify_attrib_local<parent_att>(aff_node, parent_id);
        graph->add_or_modify_attrib_local<pos_x_att>(aff_node, 300.f);
        graph->add_or_modify_attrib_local<pos_y_att>(aff_node, 200.f);
        graph->add_or_modify_attrib_local<active_att>(aff_node, false);
        graph->add_or_modify_attrib_local<epistemic_target_x_m_att>(aff_node, tx);
        graph->add_or_modify_attrib_local<epistemic_target_y_m_att>(aff_node, ty);
        graph->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(aff_node, yaw);
        graph->add_or_modify_attrib_local<epistemic_gain_att>(aff_node, gain);
        graph->add_or_modify_attrib_local<epistemic_pending_att>(aff_node, true);
        // ★THE PROPOSAL EPOCH. A brand-new node is proposal 1. It identifies WHICH proposal a consumer
        // accepted, and it is the only thing that lets either side notice they have diverged — see
        // ExecutingClaim in the header for the deadlock this exists to end.
        graph->add_or_modify_attrib_local<epistemic_target_epoch_att>(aff_node, 1);

        rc::provenance::stamp_creation(*graph, aff_node);   // birth stamp: epoch ms + local ISO-8601
        const auto id_opt = graph->insert_node(aff_node);
        if (!id_opt.has_value())
        {
            qWarning() << "DSR: failed to create affordance node" << managed_node_name_.c_str();
            return false;
        }

        managed_node_id_ = id_opt.value();
        waiting_completion_ = true;   // armed on creation — see the note on the re-publish path
        pending_seen_ = false;        // not believed until observed pending
        node_opt = graph->get_node(managed_node_id_);
        if (!node_opt.has_value())
            return false;
        if (on_node_inserted)
            on_node_inserted();
    }

    auto node = node_opt.value();
    const auto current_target_opt = read_target(graph, node);
    const bool current_active = graph->get_attrib_by_name<active_att>(node).value_or(false);
    const bool current_pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);

    constexpr float kPosEps = 0.03f;      // 3 cm
    constexpr float kYawEps = 0.0872665f; // 5 deg
    constexpr float kGainEps = 1e-4f;

    const bool same_target = current_target_opt.has_value()
        && std::fabs(current_target_opt->room_pos.x() - tx) <= kPosEps
        && std::fabs(current_target_opt->room_pos.y() - ty) <= kPosEps
        && std::fabs(current_target_opt->yaw_rad - yaw) <= kYawEps
        && std::fabs(current_target_opt->epistemic_gain - gain) <= kGainEps;

    const bool has_intention_edge = graph->get_edge(parent_id, managed_node_id_, kAffordanceEdgeTypeStr.data()).has_value();

    // Execution state is owned by the sibling controller agent.
    // Never publish or rewrite while it has the affordance claimed.
    if (current_active && current_pending)
    {
        return false;
    }

    if (same_target && current_pending)
    {
        if (!has_intention_edge)
        {
            auto aff_edge = DSR::Edge::create<affordance_edge_type>(parent_id, managed_node_id_);
            graph->insert_or_assign_edge(aff_edge);
            if (on_edge_inserted)
                on_edge_inserted();
        }
        return true;
    }

    // ★★★RE-OFFERING AFTER A COMPLETION IS NEWS. This used to decline, on the reasoning that an
    // unchanged proposal says nothing new — and that is true while the node is still OFFERED (handled
    // above). Once it is COMPLETED the offer has been consumed, and declining leaves the pair deadlocked:
    // the producer waits for the consumer to take an offer it never re-armed, and the consumer's
    // selector rejects the node at its first test because it is not Offered — before the "just
    // completed" branch that would have set the yield's fallback. Observed live 2026-08-19:
    //     [aff-select] NONE ELIGIBLE — nothing is Offered or Executing. [afford_room (JustCompleted)]
    // repeated indefinitely, robot frozen with cmd_adv = cmd_rot = 0.000 and valid cells on the wire.
    // ★This is precisely the row AffordanceProtocol.tla calls DEADLOCK (RearmUnchanged = FALSE with a
    // reporting consumer). It was dismissed because LevelTriggered was believed to cover it; the model
    // could not see it because it has no notion of WHICH cell is proposed, so "the producer will pick a
    // different one" was an assumption baked into the abstraction rather than a checked property.
    // ★SAFE TO RE-ARM NOW, and only now, because the CONSUMER carries the rate limit: a standpoint it
    // refused is un-offerable to it for kRefusalRetryMs, keyed on the POSE. So the producer may always
    // re-offer, the consumer declines to re-take the same spot too soon, and the pair can neither
    // deadlock nor busy-loop. Neither half is sufficient alone — that is why both exist.
    // ★★BUMP THE EPOCH ON A CONTENT CHANGE, AND ONLY ON A CONTENT CHANGE. This is the same distinction
    // ControllerWorldModel::same_target_instance had to learn the hard way: a value that changes for
    // PROTOCOL reasons (a re-arm, a gain refresh) is not a new proposal, and treating it as one made a
    // re-offered standpoint look new, which made an already-reached goal look like an approach that
    // never happened, which refused it — for ever, with the robot parked on it.
    // ★Compared against what is ON THE NODE, not against a member: the node is the shared truth, a
    // member is this process's memory of it, and after a restart the two disagree.
    // ★A missing epoch (pre-rollout node) starts at 1 rather than 0, so "has an epoch" and "epoch is
    // the default" are never the same reading.
    {
        const auto ox = graph->get_attrib_by_name<epistemic_target_x_m_att>(node);
        const auto oy = graph->get_attrib_by_name<epistemic_target_y_m_att>(node);
        const auto ow = graph->get_attrib_by_name<epistemic_target_yaw_rad_att>(node);
        const int  oe = graph->get_attrib_by_name<epistemic_target_epoch_att>(node).value_or(0);
        const bool content_changed =
            not ox.has_value() or not oy.has_value() or not ow.has_value()
            or std::abs(ox.value() - tx)  > 1e-4f
            or std::abs(oy.value() - ty)  > 1e-4f
            or std::abs(ow.value() - yaw) > 1e-4f;
        graph->add_or_modify_attrib_local<epistemic_target_epoch_att>(node,
            content_changed ? std::max(1, oe + 1) : std::max(1, oe));
    }
    graph->add_or_modify_attrib_local<parent_att>(node, parent_id);
    graph->add_or_modify_attrib_local<epistemic_target_x_m_att>(node, tx);
    graph->add_or_modify_attrib_local<epistemic_target_y_m_att>(node, ty);
    graph->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, yaw);
    graph->add_or_modify_attrib_local<epistemic_gain_att>(node, gain);
    if (!current_pending)
        graph->add_or_modify_attrib_local<epistemic_pending_att>(node, true);
    graph->update_node(node);
    // ★ WE ARMED IT, SO WE ARE WAITING — set synchronously rather than waiting to OBSERVE the node
    // in some state. A consumer that claims and finishes between two of our polls would otherwise be
    // invisible, which is precisely the race that wedged both agents (see monitor_execution).
    waiting_completion_ = true;
    pending_seen_ = false;   // not believed until observed pending

    if (graph->get_edge(parent_id, managed_node_id_, "RT").has_value())
        graph->delete_edge(parent_id, managed_node_id_, "RT");

    if (!has_intention_edge)
    {
        auto aff_edge = DSR::Edge::create<affordance_edge_type>(parent_id, managed_node_id_);
        graph->insert_or_assign_edge(aff_edge);
        if (on_edge_inserted)
            on_edge_inserted();
    }

    waiting_completion_ = false;
    completion_detected_ = false;
    last_managed_active_ = false;
    last_managed_pending_ = true;
    return true;
}

// ══ THE EXECUTION CLAIM, AND WHAT IT IS A CLAIM TO ═══════════════════════════════════════════════
// See ExecutingClaim in the header for why this is an edge the consumer owns rather than attributes on
// the producer's node: `update_node` is a whole-node replace on BOTH cortex sync engines, so two agents
// read-modify-writing one node race, and the loser's attribute is deleted graph-wide. One writer per
// attribute is not enough on this engine; it takes one writer per node.

std::optional<std::uint64_t> AffordanceManager::own_agent_node_id(
    const std::shared_ptr<DSR::DSRGraph> &graph)
{
    if (!graph) return std::nullopt;
    // The same lookup cortex performs on participant loss (dsr_api.cpp): type "agent" + agent_id_att.
    // Deliberately NOT get_node("<agent_name> <id>") — that reconstructs a name the API already owns,
    // and a second spelling of an identity is how a cleanup misses the agent that spelled it differently.
    const auto want = graph->get_agent_id();
    for (auto &n : graph->get_nodes_by_type("agent"))
        if (const auto id = graph->get_attrib_by_name<agent_id_att>(n); id.has_value() and id.value() == want)
            return n.id();
    return std::nullopt;
}

std::optional<int> AffordanceManager::producer_epoch(const std::shared_ptr<DSR::DSRGraph> &graph,
                                                     std::uint64_t affordance_id)
{
    if (!graph or affordance_id == 0) return std::nullopt;
    const auto n = graph->get_node(affordance_id);
    if (!n.has_value()) return std::nullopt;
    // nullopt here means a PRE-ROLLOUT producer, not epoch 0. The caller must fall back, not assume.
    return graph->get_attrib_by_name<epistemic_target_epoch_att>(n.value());
}

bool AffordanceManager::publish_executing(const std::shared_ptr<DSR::DSRGraph> &graph,
                                          std::uint64_t affordance_id,
                                          int epoch, float x, float y, float yaw)
{
    if (!graph or affordance_id == 0) return false;
    const auto from = own_agent_node_id(graph);
    if (!from.has_value()) return false;          // no agent node yet (cold start) — say nothing false

    // ★IDEMPOTENT. This is called every cycle the target decision is made, at 20 Hz. Publishing an
    // unchanged edge would put a delta on the wire per cycle per consumer, which is exactly the kind of
    // per-cycle chatter that has bitten this graph before. Compare first; write only on change.
    if (const auto cur = graph->get_edge(from.value(), affordance_id, "executing"); cur.has_value())
    {
        const auto e  = graph->get_attrib_by_name<executing_epoch_att>(cur.value());
        const auto cx = graph->get_attrib_by_name<executing_target_x_m_att>(cur.value());
        const auto cy = graph->get_attrib_by_name<executing_target_y_m_att>(cur.value());
        const auto cw = graph->get_attrib_by_name<executing_target_yaw_rad_att>(cur.value());
        if (e.has_value() and cx.has_value() and cy.has_value() and cw.has_value()
            and e.value() == epoch
            and std::abs(cx.value() - x) < 1e-4f and std::abs(cy.value() - y) < 1e-4f
            and std::abs(cw.value() - yaw) < 1e-4f)
            return true;                          // nothing changed; the standing claim is still true
    }

    // ── INV-6: A CONSUMER CLAIMS AT MOST ONE AFFORDANCE, AND THAT IS STRUCTURAL ──────────────────
    // Selection can move from affordance A to B without either a completion or an explicit release —
    // A simply stops winning. Nothing then tore A's claim down, and A's producer would go on measuring
    // its watchdog against a pose this robot walked away from. Clearing every OTHER claim of ours here
    // makes "one claim per consumer" a property of the write rather than an assumption about callers,
    // which is the only version of it that survives a new call site.
    // get_edges(id) returns this node's OUTGOING edges keyed by (to, type) — our own fano, so the scan
    // is over edges we wrote and nobody else's.
    if (const auto self = from.value(); const auto fano = graph->get_edges(self))
    {
        std::vector<std::uint64_t> stale;
        for (const auto &[key, e] : fano.value())
            if (key.second == "executing" and key.first != affordance_id) stale.push_back(key.first);
        for (const auto to : stale) graph->delete_edge(self, to, "executing");
    }

    // ★ONE insert_or_assign_edge, so the four attributes can never be observed apart. INV-1 says the
    // edge's existence implies its contents; splitting this into two writes would make that false for
    // one cycle, and one cycle is all the producer needs to draw the wrong conclusion.
    auto edge = DSR::Edge::create<executing_edge_type>(from.value(), affordance_id);
    graph->add_or_modify_attrib_local<executing_epoch_att>(edge, epoch);
    graph->add_or_modify_attrib_local<executing_target_x_m_att>(edge, x);
    graph->add_or_modify_attrib_local<executing_target_y_m_att>(edge, y);
    graph->add_or_modify_attrib_local<executing_target_yaw_rad_att>(edge, yaw);
    return graph->insert_or_assign_edge(std::move(edge));
}

bool AffordanceManager::clear_executing(const std::shared_ptr<DSR::DSRGraph> &graph,
                                        std::uint64_t affordance_id)
{
    if (!graph or affordance_id == 0) return false;
    const auto from = own_agent_node_id(graph);
    if (!from.has_value()) return false;
    if (!graph->get_edge(from.value(), affordance_id, "executing").has_value()) return false;
    return graph->delete_edge(from.value(), affordance_id, "executing");
}

std::optional<AffordanceManager::ExecutingClaim> AffordanceManager::read_executing(
    const std::shared_ptr<DSR::DSRGraph> &graph, std::uint64_t affordance_id, bool *claimed)
{
    if (claimed) *claimed = false;
    if (!graph or affordance_id == 0) return std::nullopt;
    // Scanned by INCOMING edge, so the producer never has to know which agent is driving — it asks the
    // affordance who is holding it, which is the question it actually has.
    for (auto &e : graph->get_edges_to_id(affordance_id))
    {
        if (e.type() != "executing") continue;
        if (claimed) *claimed = true;             // somebody claims it...
        const auto ep = graph->get_attrib_by_name<executing_epoch_att>(e);
        const auto x  = graph->get_attrib_by_name<executing_target_x_m_att>(e);
        const auto y  = graph->get_attrib_by_name<executing_target_y_m_att>(e);
        const auto w  = graph->get_attrib_by_name<executing_target_yaw_rad_att>(e);
        // ...but a PRE-ROLLOUT consumer cannot say what. `claimed` true with nullopt returned is that
        // case, and it is NOT the same as "nobody is driving" — the caller must keep its old behaviour.
        if (!ep.has_value() or !x.has_value() or !y.has_value() or !w.has_value()) return std::nullopt;
        return ExecutingClaim{.epoch = ep.value(), .x = x.value(), .y = y.value(), .yaw = w.value(),
                              .agent_node_id = e.from()};
    }
    return std::nullopt;
}

bool AffordanceManager::refresh_gain(const std::shared_ptr<DSR::DSRGraph> &graph, float gain)
{
    if (graph == nullptr or managed_node_id_ == 0 or not std::isfinite(gain)) return false;
    auto node = graph->get_node(managed_node_id_);
    if (not node.has_value()) return false;
    // Only while the offer is actually STANDING. Rewriting the gain of a node the consumer has
    // claimed would change the price of something already being executed, and rewriting a Completed
    // node's gain would resurrect a number nobody is offering.
    const bool pending = graph->get_attrib_by_name<epistemic_pending_att>(node.value()).value_or(false);
    const bool active  = graph->get_attrib_by_name<active_att>(node.value()).value_or(false);
    if (not pending or active) return false;
    const auto cur = graph->get_attrib_by_name<epistemic_gain_att>(node.value());
    if (cur.has_value() and std::fabs(cur.value() - gain) <= 1e-4f) return false;   // no news
    graph->add_or_modify_attrib_local<epistemic_gain_att>(node.value(), gain);
    graph->update_node(node.value());
    return true;
}

bool AffordanceManager::release_execution_claim(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    if (!graph)
        return false;

    const auto node_opt = get_managed_node(graph);
    if (!node_opt.has_value())
        return false;

    auto node = node_opt.value();
    // ★DROP THE EDGE FIRST, AND UNCONDITIONALLY. The state check below early-returns whenever the node
    // does not READ Executing — a producer that flipped `pending`, a poll that raced the transition —
    // and on that path the claim edge used to survive a release. A leaked claim is worse than no claim:
    // it tells the producer we are driving to a pose we have abandoned, and its watchdog now BELIEVES
    // that pose (INV-2), so it would wait out the full timeout on a fiction. "Release" means "we are
    // not driving this", and that is true regardless of what the node currently reads.
    clear_executing(graph, node.id());
    const bool active = graph->get_attrib_by_name<active_att>(node).value_or(false);
    const bool pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
    if (decode_protocol_state(active, pending) != ProtocolState::Executing)
        return false;

    graph->add_or_modify_attrib_local<active_att>(node, false);
    graph->update_node(node);

    waiting_completion_ = false;
    completion_detected_ = false;
    last_managed_active_ = false;
    last_managed_pending_ = true;
    current_affordance_id_ = 0;
    current_affordance_name_.clear();
    // NOT selected_target_debug_report_: it is a PRINT-dedup key, not affordance state. Clearing it here
    // made [aff-select] re-print on every re-selection of the SAME affordance — and clear_current() runs
    // once per cycle whenever there is no target (or a mission owns the base), so the "print only on
    // change" intent degenerated into per-cycle spam.
    reset_observation();
    transition_to(State::Idle, "execution claim released", node.id(), node.name());
    return true;
}

// ── IS THIS THE SAME REQUEST THE ROBOT JUST FINISHED? ────────────────────────────────────────────
// The no-two-in-a-row rule forbids REPEATING a request, and what a request IS depends on the policy.
// For a Reach it is a place: go and stand there. Comparing standpoints is exactly right, and two
// offers of the same cell are the repetition the rule exists to stop.
//
// ★FOR AN ORIENT IT IS A BEARING, AND COMPARING PLACES IS THE WRONG QUESTION ENTIRELY. An Orient is
// published AT THE ROBOT'S OWN POSE and turns in place, so its standpoint is unchanged by
// construction -- every step of a sequence looks like a repeat of the one before, for ever. The
// calibration pivot is twelve consecutive 120-degree steps from one spot: measured 2026-08-24, it
// completed a step, was suppressed as "just-completed" on the very next cycle, and afford_room was
// claimed and committed the robot to a 5.4 m traversal. That is why the pivot has never once got past
// its opening steps, and why forcing its gain did not help -- it was excluded before it was scored.
// Two Orients from one spot asking for bearings 120 degrees apart are DIFFERENT REQUESTS, and the
// bearing is what has to be compared. The band is the executor's own "aligned": inside it the robot is
// already pointing there, so re-issuing really would be the repetition the rule forbids.
static bool is_same_request(const rc::AffordanceManager::Target &t, float last_x, float last_y,
                            float last_yaw, bool pose_known)
{
    if (not pose_known) return false;                       // fail OPEN: when in doubt the rule yields
    constexpr float kSameSpotM   = 0.30f;
    constexpr float kSameYawRad  = 0.05f;                   // ControllerSession::step_orient's band
    if (std::hypot(t.room_pos.x() - last_x, t.room_pos.y() - last_y) >= kSameSpotM) return false;
    if (t.policy != rc::affordance::Policy::Orient) return true;
    const float d = std::atan2(std::sin(t.yaw_rad - last_yaw), std::cos(t.yaw_rad - last_yaw));
    return std::abs(d) < kSameYawRad;
}

// How long a refused STANDPOINT stays un-offerable. Long enough that the situation can change,
// short enough that a genuine retry is not deferred noticeably.
static constexpr std::uint64_t kRefusalRetryMs = 3000;
std::optional<AffordanceManager::Target> AffordanceManager::select_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                                                          std::optional<Eigen::Vector2f> robot_pos)
{
    transition_to(State::Searching, "select_target called", current_affordance_id_, current_affordance_name_);
    suppressed_name_.clear();
    // One selection round has passed for every affordance the consumer could not reach.
    for (auto it = unreachable_rounds_.begin(); it != unreachable_rounds_.end();)
        if (--it->second <= 0) it = unreachable_rounds_.erase(it); else ++it;

    if (!graph)
    {
        transition_to(State::Idle, "graph unavailable");
        return std::nullopt;
    }

    // Grounded EFE selection: pick the affordance minimising G = λ_cost·nav_dist − epistemic_gain,
    // i.e. maximising the negated score below. epistemic_gain is now a common currency across
    // producers (expected entropy reduction in nats: afford_table = shape-belief ΔH, afford_room =
    // pose-belief ΔH). The previously-selected affordance gets a switch-margin bonus so the choice
    // commits and does not thrash (hysteresis).
    const auto affordances = graph->get_nodes_by_type("affordance");

    // ── IS ANY OBJECT AFFORDANCE IN THE RUNNING? ──────────────────────────────────────────────────
    // Decided over the ELIGIBLE set only (Offered or Executing): an object affordance its own producer
    // has retired is not a competitor, and demoting the room in favour of something nobody is offering
    // would leave the robot doing neither. Computed once, before scoring, because the demotion is a
    // property of the CONTEST and must be the same for every candidate in it.
    bool object_competing = false;
    for (const auto &node : affordances)
    {
        const auto t = read_target(graph, node);
        if (!t.has_value() || t->parent_node_type == "room")
            continue;
        const auto a = graph->get_attrib_by_name<active_att>(node).value_or(false);
        const auto pd = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
        if (const auto st = decode_protocol_state(a, pd);
            st == ProtocolState::Offered || st == ProtocolState::Executing)
        {
            object_competing = true;
            break;
        }
    }

    const auto neg_efe = [&](const Target &t) -> float
    {
        // No robot position ⇒ nav-cost disabled (rank by epistemic_gain + hysteresis only).
        const float nav_dist = robot_pos ? (t.room_pos - *robot_pos).norm() : 0.f;
        // The room's gain is discounted only while an object is actually competing for the robot —
        // see set_room_gain_scale. Applied to the GAIN, not to the final score, so the nav-cost stays
        // in metres-of-driving and the discount stays in units of information.
        // ★AND IT IS ABOUT RELOCALISATION TOURS, NOT ABOUT PARENTAGE. The preference above is argued
        // from one property: re-localising is a standing background need that recovers on its own and
        // can be paid at any time, so it must not starve the opportunistic looks. An in-place
        // manoeuvre published on the room node is not that -- afford_calib is parented to `room` only
        // because room_concept is the agent that produces it, and its chance is as perishable as an
        // object's (the robot has space to turn HERE, and will drive away). Discounting it by 0.35
        // charged it for a property it does not have.
        const bool relocalisation_tour = t.parent_node_type == "room"
                                     and t.policy != rc::affordance::Policy::Orient;
        const float gain = (relocalisation_tour and object_competing)
                               ? t.epistemic_gain * select_room_gain_scale_
                               : t.epistemic_gain;
        float score = gain - select_lambda_cost_ * nav_dist;
        if (t.node_id == last_selected_id_)
            score += select_switch_margin_;
        return score;
    };
    const auto better = [&](const Target &candidate, const Target &current) -> bool
    {
        const float dc = neg_efe(candidate), dk = neg_efe(current);
        if (dc != dk) return dc > dk;
        return candidate.node_id < current.node_id;   // stable tie-break
    };

    // Snapshot every evaluated affordance (gain + EFE score + eligibility) for the controller's EFE
    // plot — done up front so it is populated regardless of which selection branch returns below.
    last_candidates_.clear();
    for (const auto &node : affordances)
    {
        const auto target = read_target(graph, node);
        if (!target.has_value())
            continue;
        const auto active = graph->get_attrib_by_name<active_att>(node).value_or(false);
        const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
        const auto state = decode_protocol_state(active, pending);
        const bool just_done = node.id() == last_completed_id_
            and is_same_request(*target, last_completed_x_, last_completed_y_, last_completed_yaw_,
                                last_completed_pose_known_);
        last_candidates_.push_back({target->node_name, target->parent_node_type,
                                    target->epistemic_gain, neg_efe(*target),
                                    (state == ProtocolState::Offered
                                     || state == ProtocolState::Executing) && !just_done,
                                    just_done ? std::string("JustCompleted")
                                              : std::string(protocol_state_name(state))});
    }

    // One compact line listing EVERY affordance with its protocol state — shows WHY the eligible set
    // flips (e.g. the incumbent went Completed and dropped out), which gain/score alone can't reveal.
    std::ostringstream all_cands;
    for (const auto &c : last_candidates_)
        all_cands << " '" << c.node_name << "'(" << c.state << ") gain=" << c.gain
                  << " score=" << c.efe_score << " |";
    const std::string candidates_str = all_cands.str();

    if (current_affordance_id_ != 0)
    {
        for (const auto &node : affordances)
        {
            if (node.id() != current_affordance_id_)
                continue;

            const auto target = read_target(graph, node);
            const auto active = graph->get_attrib_by_name<active_att>(node).value_or(false);
            const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
            if (target.has_value() && decode_protocol_state(active, pending) == ProtocolState::Executing)
            {
                if (const auto key = make_affordance_key(*target); key != selected_target_debug_report_)
                {
                    selected_target_debug_report_ = key;
                    std::print("{}  [candidates:{} ]\n", make_affordance_debug_report(*target), candidates_str);
                    std::fflush(stdout);
                }
                log_observation(target->node_id,
                                target->node_name,
                                active,
                                pending,
                                target->room_pos.x(),
                                target->room_pos.y(),
                                target->yaw_rad,
                                target->epistemic_gain);
                transition_to(State::Following, "keeping current active affordance", target->node_id, target->node_name);
                last_selected_id_ = target->node_id;
                return target;
            }

            current_affordance_id_ = 0;
            current_affordance_name_.clear();
            reset_observation();
            break;
        }
    }

    std::optional<Target> resumed_target;
    for (const auto &node : affordances)
    {
        const auto target = read_target(graph, node);
        if (!target.has_value())
            continue;

        const auto active = graph->get_attrib_by_name<active_att>(node).value_or(false);
        const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
        if (decode_protocol_state(active, pending) != ProtocolState::Executing)
            continue;

        if (!resumed_target.has_value() || better(*target, *resumed_target))
            resumed_target = target;
    }
    if (resumed_target.has_value())
    {
        current_affordance_id_ = resumed_target->node_id;
        current_affordance_name_ = resumed_target->node_name;
        claimed_x_ = resumed_target->room_pos.x();
        claimed_y_ = resumed_target->room_pos.y();
        claimed_yaw_ = resumed_target->yaw_rad;
        claimed_pose_known_ = true;
        if (const auto key = make_affordance_key(*resumed_target); key != selected_target_debug_report_)
        {
            selected_target_debug_report_ = key;
            std::print("{}  [candidates:{} ]\n", make_affordance_debug_report(*resumed_target), candidates_str);
            std::fflush(stdout);
        }
        reset_observation();
        log_observation(resumed_target->node_id,
                        resumed_target->node_name,
                        true,
                        resumed_target->epistemic_pending,
                        resumed_target->room_pos.x(),
                        resumed_target->room_pos.y(),
                        resumed_target->yaw_rad,
                        resumed_target->epistemic_gain);
        transition_to(State::Following, "resuming executing affordance", resumed_target->node_id, resumed_target->node_name);
        last_selected_id_ = resumed_target->node_id;
        return resumed_target;
    }

    std::optional<Target> best_target;
    std::optional<Target> suppressed_target;
    for (const auto &node : affordances)
    {
        const auto target = read_target(graph, node);
        if (!target.has_value())
            continue;

        const auto active = graph->get_attrib_by_name<active_att>(node).value_or(false);
        const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
        if (decode_protocol_state(active, pending) != ProtocolState::Offered)
        { last_reject_reason_ = "not-offered"; continue; }

        // ★ALREADY DECIDED AGAINST, FROM THIS POSE, ON THIS MAP. Taking it again cannot end
        // differently: the two questions this side can answer — does the body fit, is there a route —
        // are functions of exactly (cell, pose, map), and none of them has changed. Skipping here is
        // what keeps the pair from completing standpoints at loop rate with the base at zero (21648
        // reports in 30 minutes, live). The producer was TOLD the first time, so this is not a silent
        // veto; and it lapses the instant the robot moves or the grid is rebuilt differently.
        if (has_map_verdict(target->room_pos, robot_pos))
        {
            last_reject_reason_ = "map-verdict";
            suppressed_name_ = target->node_name;
            continue;
        }

        // NOT THE ONE THAT JUST FINISHED. Its producer re-offers it immediately and its gain has not
        // had time to fall, so without this it wins again on the very next cycle and the robot works
        // one object forever. Reported rather than silent: an affordance that is Offered, top-scoring
        // and passed over is otherwise indistinguishable from a selector that is broken.
        // ★FAIL OPEN, NOT CLOSED. This read `not last_completed_pose_known_ or ...`, so a standpoint
        // whose pose could not be recovered suppressed EVERY candidate — reinstating the starvation it
        // was written to remove (measured: 201 of 838 records back to reject=just-completed, robot
        // parked with a valid offer 2.99 m away). The suppression is a POLITENESS rule; when in doubt
        // it must yield. Re-taking one affordance too soon is bounded by the refusal hold and the
        // yield; blacklisting the whole channel is not bounded by anything.
        const bool same_standpoint = is_same_request(*target, last_completed_x_, last_completed_y_,
                                                    last_completed_yaw_, last_completed_pose_known_);
        if (node.id() == last_completed_id_ and same_standpoint)
        {
            last_reject_reason_ = "just-completed";
            suppressed_name_ = target->node_name;
            suppressed_target = target;      // kept: the rule YIELDS rather than deadlocks — see below
            continue;
        }

        // NOT ONE THE ROBOT COULD NOT REACH. See suppress_target. Deliberately NOT kept as a
        // `suppressed_target` fallback: the no-two-in-a-row rule yields rather than deadlock because
        // taking that affordance again is merely impatient, whereas taking THIS one again is known to
        // end in the same wedge. Idling for a few rounds is the cheaper outcome, and the counter
        // expires on its own.
        if (const auto it = unreachable_rounds_.find(node.id());
            it != unreachable_rounds_.end() and it->second > 0)
        {
            last_reject_reason_ = "unreachable-rounds";   // ★NODE-KEYED: suppresses EVERY standpoint
            suppressed_name_ = target->node_name;
            continue;
        }

        // ★★★DO NOT RE-TAKE A SPOT THAT WAS JUST REFUSED. A refusal says "not from here", and nothing
        // can have changed in one 50 ms cycle — so taking it again is guaranteed to refuse again. This
        // is the loop the run showed: refuse -> producer re-arms -> selected normally -> refuse, at
        // ~104 completions per minute with the robot standing still. Keyed on the STANDPOINT so a
        // genuinely different cell on the same node is unaffected, and it expires by itself.
        if (const auto it = refused_at_ms_.find(node.id()); it != refused_at_ms_.end())
        {
            const auto now_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            const float dx = target->room_pos.x() - it->second.x;
            const float dy = target->room_pos.y() - it->second.y;
            constexpr float kSameSpotM = 0.30f;
            if (now_ms >= it->second.when_ms and now_ms - it->second.when_ms < kRefusalRetryMs
                and (dx * dx + dy * dy) < kSameSpotM * kSameSpotM)
            {
                last_reject_reason_ = "refused-recently";
                suppressed_name_ = target->node_name;
                continue;                       // same spot, too soon: let the producer offer elsewhere
            }
        }

        if (!best_target.has_value() || better(*target, *best_target))
            best_target = target;
    }

    // ── THE RULE YIELDS RATHER THAN DEADLOCKS ─────────────────────────────────────────────────────
    // "Not twice in a row" is a preference for spreading attention, not an instruction to stand still.
    // When the affordance just completed is the ONLY one on offer, refusing it leaves the robot with
    // nothing to do FOREVER — which is exactly what happened: three object affordances sat Completed
    // (their producers were not re-offering them) and the one live candidate was the one suppressed.
    // A rule that can bring the whole agent to a permanent halt is not a preference, it is a deadlock.
    // ★THE YIELD NEEDS A CLOCK. Retrying is right eventually and wrong immediately: if the only
    // candidate was REFUSED moments ago, taking it again cannot end differently, and the pair spins at
    // loop rate. Hold off until enough time has passed that the situation could have changed, then
    // yield exactly as before — so this still cannot deadlock, it just cannot busy-loop either.
    bool refused_too_recently = false;
    if (suppressed_target.has_value())
        if (const auto it = refused_at_ms_.find(suppressed_target->node_id); it != refused_at_ms_.end())
        {
            const auto now_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            refused_too_recently = now_ms >= it->second.when_ms
                                   and now_ms - it->second.when_ms < kRefusalRetryMs;
        }
    if (best_target.has_value()) last_reject_reason_.clear();
    if (!best_target.has_value() and suppressed_target.has_value() and refused_too_recently)
    {
        // Say it once per refusal, not once per cycle: the point is that we are deliberately idling.
        if (suppressed_name_ != suppressed_target->node_name)
        {
            std::print("[aff-select] '{}' was REFUSED moments ago — idling rather than re-offering it; "
                       "nothing can have changed yet.\n", suppressed_target->node_name);
            std::fflush(stdout);
        }
        suppressed_name_ = suppressed_target->node_name;
    }
    else if (!best_target.has_value() and suppressed_target.has_value())
    {
        std::print("[aff-select] '{}' was the only affordance on offer — taking it again rather than "
                   "idling (no-two-in-a-row yields when it would leave nothing)\n",
                   suppressed_target->node_name);
        std::fflush(stdout);
        best_target = suppressed_target;
        suppressed_name_.clear();
        last_completed_id_ = 0;
    }

    if (!best_target.has_value())
    {
        // ★SAY WHICH ONES WERE THERE AND WHAT STATE THEY WERE IN. This used to print the bare word
        // "none", which is the least informative thing it could say in the one situation where the
        // question is loudest: the EFE panel is showing several affordances with healthy gains and the
        // robot is going to none of them. Gain and score cannot answer that — only the protocol state
        // can, because a Completed or Invalid node is skipped by every selection branch above however
        // good its score is. Dedup on the STATES, so a set that is merely sitting there stays quiet and
        // any change speaks.
        if (const std::string key = "[aff-select] none" + candidates_str;
            key != selected_target_debug_report_)
        {
            selected_target_debug_report_ = key;
            std::print("[aff-select] NONE ELIGIBLE — nothing is Offered or Executing."
                       "  [candidates:{} ]\n", candidates_str);
            std::fflush(stdout);
        }
        transition_to(State::Idle, "no eligible affordance found");
        return std::nullopt;
    }

    if (auto node_opt = graph->get_node(best_target->node_id); node_opt.has_value())
    {
        log_observation(best_target->node_id,
                        best_target->node_name,
                        false,
                        best_target->epistemic_pending,
                        best_target->room_pos.x(),
                        best_target->room_pos.y(),
                        best_target->yaw_rad,
                        best_target->epistemic_gain);
        transition_to(State::Claiming, "claiming offered affordance", best_target->node_id, best_target->node_name);

        auto node = node_opt.value();
        graph->add_or_modify_attrib_local<active_att>(node, true);
        graph->update_node(node);

        current_affordance_id_ = best_target->node_id;
        claimed_x_ = best_target->room_pos.x();
        claimed_y_ = best_target->room_pos.y();
        claimed_yaw_ = best_target->yaw_rad;
        claimed_pose_known_ = true;
        current_affordance_name_ = best_target->node_name;
        // A DIFFERENT affordance has been chosen, so the previous one is no longer "in a row" — it may
        // win the next contest on its merits. This forbids repetition, not revisiting.
        last_completed_id_ = 0;
        if (const auto key = make_affordance_key(*best_target); key != selected_target_debug_report_)
        {
            selected_target_debug_report_ = key;
            std::print("{}  [candidates:{} ]\n", make_affordance_debug_report(*best_target), candidates_str);
            // Consumer-side debug read (Part B / object-relative resolution not built yet): decode the
            // viewpoint constraint the producer published on the claimed node, to confirm the aff_view_*
            // wire format survives the DSR boundary. Pure diagnostic — no resolution logic here yet.
            if (const auto vc = rc::affordance::read_viewpoint(node); vc.object_relative)
            {
                std::ostringstream gss; gss.setf(std::ios::fixed); gss.precision(2);
                for (std::size_t i = 0; i < vc.face_gains.size(); ++i) { if (i) gss << ','; gss << vc.face_gains[i]; }
                std::print("[affordance-viewpoint] '{}' object-relative: {} faces gains=[{}] standoff=[{:.2f},{:.2f}] "
                           "fill={:.2f} σ*({})\n",
                           best_target->node_name, vc.faces.size(), gss.str(),
                           vc.standoff_min_m, vc.standoff_max_m, vc.framing_fill, vc.sigma_star.size());
            }
            std::fflush(stdout);
        }
        reset_observation();
        log_observation(best_target->node_id,
                        best_target->node_name,
                        true,
                        best_target->epistemic_pending,
                        best_target->room_pos.x(),
                        best_target->room_pos.y(),
                        best_target->yaw_rad,
                        best_target->epistemic_gain);
        transition_to(State::Following, "affordance claimed", best_target->node_id, best_target->node_name);
        last_selected_id_ = best_target->node_id;
        return best_target;
    }

    transition_to(State::Idle, "eligible affordance disappeared before claim");
    return std::nullopt;
}

void AffordanceManager::mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph,
                                     rc::affordance::Outcome outcome,
                                     std::source_location loc)
{
    // ★EVERY completion, with its caller and how far the robot was from what it claimed. The phantom
    // completions (robot 3.11 m from the cell, counter climbing while it stood still) have to come
    // through here — this names which call site produces them.
    {
        static std::ofstream j;
        static bool ok = false;
        if (not ok) { j.open("completions.jsonl", std::ios::out | std::ios::trunc);
                      j.imbue(std::locale::classic()); ok = j.is_open(); }
        if (ok)
        {
            j << std::format(R"({{"outcome":"{}","line":{},"fn":"{}","claimed_x":{:.3f},)"
                             R"("claimed_y":{:.3f},"known":{}}})" "\n",
                             rc::affordance::to_string(outcome), loc.line(), loc.function_name(),
                             claimed_x_, claimed_y_, claimed_pose_known_ ? 1 : 0);
            j.flush();
        }
    }
    if (!graph || current_affordance_id_ == 0)
        return;

    transition_to(State::Completing,
                  std::string("mark_reached: ").append(rc::affordance::to_string(outcome)),
                  current_affordance_id_, current_affordance_name_);
    // The approach is over, whatever the outcome word says. INV-5 again: an outcome and a standing
    // claim cannot both be true, and leaving the edge up would tell the producer we are still driving
    // to a pose we have just finished with — or refused.
    clear_executing(graph, current_affordance_id_);
    // Remember it so the next selection cannot hand back the affordance that just finished.
    last_completed_id_ = current_affordance_id_;
    // ★FROM WHAT WE CLAIMED, not from the node — see claimed_x_ in the header.
    last_completed_pose_known_ = claimed_pose_known_;
    last_completed_yaw_ = claimed_yaw_;
    last_completed_x_ = claimed_x_;
    last_completed_y_ = claimed_y_;
    claimed_pose_known_ = false;
    // A REFUSAL IS NOT A COMPLETION. Stamp it, so the yield below cannot retry it this instant.
    if (outcome == rc::affordance::Outcome::Refused)
    {
        const auto now_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        RefusedSpot spot{.when_ms = now_ms};
        if (claimed_pose_known_) { spot.x = claimed_x_; spot.y = claimed_y_; }
        refused_at_ms_[current_affordance_id_] = spot;
    }

    if (auto node_opt = graph->get_node(current_affordance_id_); node_opt.has_value())
    {
        auto node = node_opt.value();
        // ★ THE OUTCOME GOES ON BEFORE THE FLAGS COME OFF, in the same update_node. Clearing
        // epistemic_pending is what the producer's monitor watches for, so writing the outcome after
        // it would race: the producer can observe the completion and read the PREVIOUS outcome (or
        // none at all) in the window between two graph updates. One node write, no window.
        rc::affordance::write_outcome(*graph, node, outcome);
        graph->add_or_modify_attrib_local<epistemic_pending_att>(node, false);
        graph->add_or_modify_attrib_local<active_att>(node, false);
        graph->update_node(node);
    }
    last_outcome_ = outcome;

    current_affordance_id_ = 0;
    current_affordance_name_.clear();
    reset_observation();
    transition_to(State::Idle, "affordance completed and cleared");
}

// ── THE MAP-ONLY VERDICT CACHE (see the header for why it is not a timer) ───────────────────────
// Quantised to 5 cm on both the cell and the pose: finer than the arrival band, coarser than the
// jitter of a stationary robot's localisation, so standing still does not manufacture new poses and
// therefore does not manufacture new questions.
namespace
{
constexpr float kVerdictQuantM = 0.05f;
std::pair<std::int64_t, std::int64_t> verdict_key(const Eigen::Vector2f &cell, const Eigen::Vector2f &robot)
{
    const auto q = [](float v) { return static_cast<std::int64_t>(std::llround(v / kVerdictQuantM)); };
    // Two 2-D points packed into one pair: the cell in the high half, the pose in the low half.
    return { (q(cell.x()) << 20) ^ q(cell.y()), (q(robot.x()) << 20) ^ q(robot.y()) };
}
}   // namespace

void AffordanceManager::note_map_verdict(const Eigen::Vector2f &cell, const Eigen::Vector2f &robot)
{
    if (map_identity_ == 0) return;                  // no rasterised world: nothing to remember it by
    if (map_identity_ != map_verdict_hash_)          // the world changed: every answer is unproven
    {
        map_verdicts_.clear();
        map_verdict_hash_ = map_identity_;
    }
    map_verdicts_.insert(verdict_key(cell, robot));
}

bool AffordanceManager::has_map_verdict(const Eigen::Vector2f &cell,
                                        const std::optional<Eigen::Vector2f> &robot) const
{
    if (map_identity_ == 0 or not robot.has_value()) return false;  // ★fails OPEN
    if (map_identity_ != map_verdict_hash_) return false;           // answers belong to the old world
    return map_verdicts_.contains(verdict_key(cell, *robot));
}

void AffordanceManager::suppress_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                       std::uint64_t node_id, int rounds)
{
    if (node_id == 0 or rounds <= 0)
        return;
    // NOTE: this records the refusal, it does not RETIRE the offer — that is the refusal protocol's
    // own step (REFUSAL_PLAN.md) and it changes what the producer must do. Recording is additive: a
    // producer that ignores epistemic_refused behaves exactly as it does now.
    // ★ A refusal is NOT a completion. Nothing was observed and nothing about the object changed —
    // the only thing learned is that this STANDPOINT is not reachable, which is a fact about the
    // approach, not about the belief.
    if (graph)
        if (auto node = graph->get_node(node_id); node.has_value())
        {
            auto n = node.value();
            graph->add_or_modify_attrib_local<epistemic_refused_att>(n, true);
            graph->update_node(n);
        }
    // Take the LONGER of any existing suppression: two independent reports that this one is
    // unreachable should not shorten the wait.
    auto &r = unreachable_rounds_[node_id];
    r = std::max(r, rounds);
    // If it is the one being held, stop holding it — otherwise the resume-executing branch above hands
    // it straight back on the next call and the suppression never gets a chance to apply.
    if (current_affordance_id_ == node_id)
        clear_current();
}

void AffordanceManager::clear_current()
{
    current_affordance_id_ = 0;
    current_affordance_name_.clear();
    // NOT selected_target_debug_report_: it is a PRINT-dedup key, not affordance state. Clearing it here
    // made [aff-select] re-print on every re-selection of the SAME affordance — and clear_current() runs
    // once per cycle whenever there is no target (or a mission owns the base), so the "print only on
    // change" intent degenerated into per-cycle spam.
    reset_observation();
    transition_to(State::Idle, "clear_current called");
}

bool AffordanceManager::has_current() const
{
    return current_affordance_id_ != 0;
}

std::string AffordanceManager::current_name() const
{
    return current_affordance_name_;
}

} // namespace rc