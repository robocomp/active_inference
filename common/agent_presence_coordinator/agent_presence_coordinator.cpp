#include "agent_presence_coordinator.h"

#include <stdexcept>
#include <utility>

void AgentPresenceCoordinator::configure(const ConfigLoader &config_loader,
                                        std::shared_ptr<DSR::DSRGraph> graph,
                                        std::uint32_t local_agent_id)
{
    stop();
    config_loader_ = &config_loader;
    graph_ = std::move(graph);
    local_agent_id_ = local_agent_id;
    // Start reporting our health on our own agent node. Presence is still Unknown here, which
    // publishes orange — correct: we are not known-healthy until we reach Operating.
    state_publisher_.configure(graph_);
}

void AgentPresenceCoordinator::set_transition_hooks(TransitionHooks hooks)
{
    transition_hooks_ = std::move(hooks);
}

void AgentPresenceCoordinator::set_peer_hooks(PeerHooks hooks)
{
    peer_hooks_ = std::move(hooks);
}

void AgentPresenceCoordinator::set_lifecycle_hooks(LifecycleHooks hooks)
{
    lifecycle_hooks_ = std::move(hooks);
}

void AgentPresenceCoordinator::set_policy(Policy policy)
{
    policy_ = policy;
}

void AgentPresenceCoordinator::require_configuration() const
{
    if (config_loader_ == nullptr)
        throw std::logic_error("AgentPresenceCoordinator is not configured with a ConfigLoader");
    if (!graph_)
        throw std::logic_error("AgentPresenceCoordinator is not configured with a DSR graph");
}

void AgentPresenceCoordinator::start()
{
    require_configuration();
    if (presence_monitor_)
        return;

    presence_monitor_ = std::make_unique<AgentPresenceMonitor>(*config_loader_, graph_, local_agent_id_);
    presence_monitor_->on_required_ready = [this]()
    {
        if (transition_hooks_.request_presence_ready)
            transition_hooks_.request_presence_ready();
    };
    presence_monitor_->on_required_lost = [this]()
    {
        if (transition_hooks_.request_presence_lost)
            transition_hooks_.request_presence_lost();
    };
    presence_monitor_->on_peer_restarted = [this](std::uint32_t id)
    {
        if (peer_hooks_.on_peer_restarted)
            peer_hooks_.on_peer_restarted(id);
    };
    presence_monitor_->on_optional_agent_lost = [this](std::string name, std::uint32_t id)
    {
        if (peer_hooks_.on_optional_peer_lost)
            peer_hooks_.on_optional_peer_lost(name, id);
    };
    presence_monitor_->on_optional_agent_ready = [this](std::string name, std::uint32_t id)
    {
        if (peer_hooks_.on_optional_peer_ready)
            peer_hooks_.on_optional_peer_ready(name, id);
    };
    presence_monitor_->start();
}

void AgentPresenceCoordinator::stop()
{
    state_publisher_.stop();

    if (!presence_monitor_)
        return;

    presence_monitor_->stop();
    presence_monitor_.reset();
}

bool AgentPresenceCoordinator::all_required_ready() const
{
    return presence_monitor_ && presence_monitor_->all_required_ready();
}

std::vector<std::string> AgentPresenceCoordinator::missing_required_names() const
{
    if (!presence_monitor_)
        return {};

    return presence_monitor_->missing_required_names();
}

void AgentPresenceCoordinator::set_local_ready(bool ready)
{
    if (presence_monitor_)
        presence_monitor_->set_local_ready(ready);
}

void AgentPresenceCoordinator::run_hook(const std::function<void()> &hook,
                                       HookOrder target_order,
                                       HookOrder active_order) const
{
    if (hook && target_order == active_order)
        hook();
}

void AgentPresenceCoordinator::waiting_enter()
{
    // Publish BEFORE the hooks: an agent's on_degraded_enter debounces for seconds before deciding
    // whether to die, and the graph must show the trouble for that whole window, not after it.
    state_publisher_.set_presence(rc::agent_status::Presence::Waiting);
    run_hook(lifecycle_hooks_.on_waiting_enter, policy_.waiting_enter_order, HookOrder::BeforeCoordinator);
    if (policy_.set_local_ready_false_on_waiting_enter && presence_monitor_)
        presence_monitor_->set_local_ready(false);
    run_hook(lifecycle_hooks_.on_waiting_enter, policy_.waiting_enter_order, HookOrder::AfterCoordinator);

    if (policy_.auto_request_presence_ready_from_waiting
        && presence_monitor_
        && presence_monitor_->all_required_ready()
        && transition_hooks_.request_presence_ready)
    {
        transition_hooks_.request_presence_ready();
    }
}

void AgentPresenceCoordinator::waiting_loop()
{
    if (lifecycle_hooks_.on_waiting_loop)
        lifecycle_hooks_.on_waiting_loop();
}

void AgentPresenceCoordinator::operating_enter()
{
    state_publisher_.set_presence(rc::agent_status::Presence::Operating);
    run_hook(lifecycle_hooks_.on_operating_enter, policy_.operating_enter_order, HookOrder::BeforeCoordinator);
    if (policy_.set_local_ready_true_on_operating_enter && presence_monitor_)
        presence_monitor_->set_local_ready(true);
    run_hook(lifecycle_hooks_.on_operating_enter, policy_.operating_enter_order, HookOrder::AfterCoordinator);
}

void AgentPresenceCoordinator::operating_loop()
{
    if (lifecycle_hooks_.on_operating_loop)
        lifecycle_hooks_.on_operating_loop();
}

void AgentPresenceCoordinator::degraded_enter()
{
    state_publisher_.set_presence(rc::agent_status::Presence::Degraded);
    run_hook(lifecycle_hooks_.on_degraded_enter, policy_.degraded_enter_order, HookOrder::BeforeCoordinator);
    if (policy_.set_local_ready_false_on_degraded_enter && presence_monitor_)
        presence_monitor_->set_local_ready(false);
    run_hook(lifecycle_hooks_.on_degraded_enter, policy_.degraded_enter_order, HookOrder::AfterCoordinator);
}

void AgentPresenceCoordinator::degraded_loop()
{
    if (lifecycle_hooks_.on_degraded_loop)
        lifecycle_hooks_.on_degraded_loop();
}

void AgentPresenceCoordinator::cleanup_owned_nodes()
{
    if (!presence_monitor_)
        return;

    // Stop monitor activity first (timer, graph signal callbacks and AgentInfoAPI
    // heartbeats) to avoid racing graph deletions during shutdown.
    presence_monitor_->stop();
    presence_monitor_->delete_owned_nodes();
    presence_monitor_.reset();
}