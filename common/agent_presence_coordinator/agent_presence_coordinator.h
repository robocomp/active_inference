#pragma once

#include <ConfigLoader/ConfigLoader.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../agent_presence_monitor/agent_presence_monitor.h"
#include "../agent_state_publisher/agent_state_publisher.h"

class QStateMachine;

namespace DSR {
class DSRGraph;
}

class AgentPresenceCoordinator
{
public:
    enum class HookOrder
    {
        BeforeCoordinator,
        AfterCoordinator,
    };

    struct TransitionHooks
    {
        std::function<void()> request_presence_ready;
        std::function<void()> request_presence_lost;
    };

    struct PeerHooks
    {
        std::function<void(std::uint32_t)> on_peer_restarted;
        std::function<void(const std::string &, std::uint32_t)> on_optional_peer_lost;
        std::function<void(const std::string &, std::uint32_t)> on_optional_peer_ready;
    };

    struct LifecycleHooks
    {
        std::function<void()> on_waiting_enter;
        std::function<void()> on_waiting_loop;
        std::function<void()> on_operating_enter;
        std::function<void()> on_operating_loop;
        std::function<void()> on_degraded_enter;
        std::function<void()> on_degraded_loop;
    };

    struct Policy
    {
        bool set_local_ready_false_on_waiting_enter = true;
        bool set_local_ready_true_on_operating_enter = true;
        bool set_local_ready_false_on_degraded_enter = true;
        bool auto_request_presence_ready_from_waiting = true;
        HookOrder waiting_enter_order = HookOrder::BeforeCoordinator;
        HookOrder operating_enter_order = HookOrder::BeforeCoordinator;
        HookOrder degraded_enter_order = HookOrder::BeforeCoordinator;
    };

    void configure(const ConfigLoader &config_loader,
                   std::shared_ptr<DSR::DSRGraph> graph,
                   std::uint32_t local_agent_id);

    void set_transition_hooks(TransitionHooks hooks);
    void set_peer_hooks(PeerHooks hooks);
    void set_lifecycle_hooks(LifecycleHooks hooks);
    void set_policy(Policy policy);

    void start();
    void stop();

    // Publish this agent's external FSM state (Initialize/Compute/Emergency/Restore) alongside the
    // presence lifecycle, so its node in the graph view is coloured by the WORSE of the two. Call
    // once from initialize(), right after configure(), passing GenericWorker's `statemachine`.
    // Presence transitions are published automatically — this only adds the FSM axis.
    void attach_state_machine(QStateMachine *sm) { state_publisher_.attach_state_machine(sm); }

    // Publish an FSM state the state machine cannot express. Needed for a LOCAL hold: a stalled input
    // stream must be visible fleet-wide (Emergency -> Bad -> red node) WITHOUT the teardown that a
    // Degraded presence transition triggers — the stream may come back, and killing the agent for a
    // transient media flap is the failure this codebase already learned to debounce.
    void publish_fsm(rc::agent_status::Fsm f) { state_publisher_.set_fsm(f); }

    [[nodiscard]] AgentPresenceMonitor *monitor() const { return presence_monitor_.get(); }
    [[nodiscard]] bool all_required_ready() const;
    [[nodiscard]] std::vector<std::string> missing_required_names() const;

    void set_local_ready(bool ready);

    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();

    void cleanup_owned_nodes();

private:
    void run_hook(const std::function<void()> &hook, HookOrder target_order, HookOrder active_order) const;
    void require_configuration() const;

    const ConfigLoader *config_loader_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    std::uint32_t local_agent_id_ = 0;
    TransitionHooks transition_hooks_;
    PeerHooks peer_hooks_;
    LifecycleHooks lifecycle_hooks_;
    Policy policy_;
    std::unique_ptr<AgentPresenceMonitor> presence_monitor_;
    rc::AgentStatePublisher state_publisher_;
};