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

    if (active && pending)
        waiting_completion_ = true;
    else if (waiting_completion_ && !pending)
    {
        waiting_completion_ = false;
        completion_detected_ = true;
    }
    else if (!active)
        waiting_completion_ = false;
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

        rc::provenance::stamp_creation(*graph, aff_node);   // birth stamp: epoch ms + local ISO-8601
        const auto id_opt = graph->insert_node(aff_node);
        if (!id_opt.has_value())
        {
            qWarning() << "DSR: failed to create affordance node" << managed_node_name_.c_str();
            return false;
        }

        managed_node_id_ = id_opt.value();
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

    if (same_target && !current_pending)
    {
        return false;
    }
    graph->add_or_modify_attrib_local<parent_att>(node, parent_id);
    graph->add_or_modify_attrib_local<epistemic_target_x_m_att>(node, tx);
    graph->add_or_modify_attrib_local<epistemic_target_y_m_att>(node, ty);
    graph->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, yaw);
    graph->add_or_modify_attrib_local<epistemic_gain_att>(node, gain);
    if (!current_pending)
        graph->add_or_modify_attrib_local<epistemic_pending_att>(node, true);
    graph->update_node(node);

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

bool AffordanceManager::release_execution_claim(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    if (!graph)
        return false;

    const auto node_opt = get_managed_node(graph);
    if (!node_opt.has_value())
        return false;

    auto node = node_opt.value();
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

std::optional<AffordanceManager::Target> AffordanceManager::select_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                                                          std::optional<Eigen::Vector2f> robot_pos)
{
    transition_to(State::Searching, "select_target called", current_affordance_id_, current_affordance_name_);
    suppressed_name_.clear();

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
        const float gain = (t.parent_node_type == "room" and object_competing)
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
        const bool just_done = node.id() == last_completed_id_;
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
            continue;

        // NOT THE ONE THAT JUST FINISHED. Its producer re-offers it immediately and its gain has not
        // had time to fall, so without this it wins again on the very next cycle and the robot works
        // one object forever. Reported rather than silent: an affordance that is Offered, top-scoring
        // and passed over is otherwise indistinguishable from a selector that is broken.
        if (node.id() == last_completed_id_)
        {
            suppressed_name_ = target->node_name;
            suppressed_target = target;      // kept: the rule YIELDS rather than deadlocks — see below
            continue;
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
    if (!best_target.has_value() and suppressed_target.has_value())
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

void AffordanceManager::mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    if (!graph || current_affordance_id_ == 0)
        return;

    transition_to(State::Completing, "mark_reached called", current_affordance_id_, current_affordance_name_);
    // Remember it so the next selection cannot hand back the affordance that just finished.
    last_completed_id_ = current_affordance_id_;

    if (auto node_opt = graph->get_node(current_affordance_id_); node_opt.has_value())
    {
        auto node = node_opt.value();
        graph->add_or_modify_attrib_local<epistemic_pending_att>(node, false);
        graph->add_or_modify_attrib_local<active_att>(node, false);
        graph->update_node(node);
    }

    current_affordance_id_ = 0;
    current_affordance_name_.clear();
    reset_observation();
    transition_to(State::Idle, "affordance completed and cleared");
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