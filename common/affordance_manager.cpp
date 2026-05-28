#include "affordance_manager.h"

#include <dsr/api/dsr_api.h>

#include <QDebug>

#include <cmath>
#include <utility>

namespace
{
constexpr auto kAffordanceEdgeTypeStr = std::string_view("has_intention");
[[maybe_unused]] inline const bool kAffordanceEdgeTypeRegistered = edge_types::register_type(kAffordanceEdgeTypeStr);
using affordance_edge_type = EdgeType<kAffordanceEdgeTypeStr>;
}

namespace rc
{

AffordanceManager::AffordanceManager(std::string managed_node_name)
    : managed_node_name_(std::move(managed_node_name))
{
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

    Target target;
    target.node_id = node.id();
    target.node_name = node.name();
    target.room_pos = Eigen::Vector2f(tx.value(), ty.value());
    target.yaw_rad = yaw.value();
    target.epistemic_gain = gain.value();
    target.epistemic_pending = pending.value();
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

std::optional<AffordanceManager::Target> AffordanceManager::select_target(const std::shared_ptr<DSR::DSRGraph> &graph)
{
    transition_to(State::Searching, "select_target called", current_affordance_id_, current_affordance_name_);

    if (!graph)
    {
        transition_to(State::Idle, "graph unavailable");
        return std::nullopt;
    }

    const auto affordances = graph->get_nodes_by_type("affordance");

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
                log_observation(target->node_id,
                                target->node_name,
                                active,
                                pending,
                                target->room_pos.x(),
                                target->room_pos.y(),
                                target->yaw_rad,
                                target->epistemic_gain);
                transition_to(State::Following, "keeping current active affordance", target->node_id, target->node_name);
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

        if (!resumed_target.has_value() || target->epistemic_gain > resumed_target->epistemic_gain)
            resumed_target = target;
    }
    if (resumed_target.has_value())
    {
        current_affordance_id_ = resumed_target->node_id;
        current_affordance_name_ = resumed_target->node_name;
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
        return resumed_target;
    }

    std::optional<Target> best_target;
    for (const auto &node : affordances)
    {
        const auto target = read_target(graph, node);
        if (!target.has_value())
            continue;

        const auto active = graph->get_attrib_by_name<active_att>(node).value_or(false);
        const auto pending = graph->get_attrib_by_name<epistemic_pending_att>(node).value_or(true);
        if (decode_protocol_state(active, pending) != ProtocolState::Offered)
            continue;

        if (!best_target.has_value() || target->epistemic_gain > best_target->epistemic_gain)
            best_target = target;
    }

    if (!best_target.has_value())
    {
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
    reset_observation();
    transition_to(State::Idle, "clear_current called");
}

bool AffordanceManager::has_current() const
{
    return current_affordance_id_ != 0;
}

} // namespace rc