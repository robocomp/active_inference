#pragma once
/*
 * agent_state_publisher.h — publishes THIS agent's live health onto its own DSR agent node, so the
 * whole system's state is readable at a glance in any graph view (robot_concept shows the complete
 * graph; every agent node hangs off "mind").
 *
 * It writes three attributes on the node AgentInfoAPI already maintains for us, named
 * "<agent_name> <agent_id>" and of type "agent":
 *
 *   agent_fsm_state       Initialize | Compute | Emergency | Restore
 *   agent_presence_state  Waiting | Operating | Degraded
 *   color                 worst-wins colour derived from the pair (see agent_status.h)
 *
 * `color` is derived HERE rather than in each viewer so the mapping lives in exactly one place and
 * plain dsr_gui paints the agents correctly without knowing anything about our presence protocol.
 *
 * Design notes:
 *  - No new nodes. Adding a child node per agent would double the "mind" subtree, add CRDT/DDS
 *    churn, and need [Owns] cleanup entries — stale state-nodes after a crash are exactly what the
 *    presence protocol exists to prevent.
 *  - Writes happen only on CHANGE, not periodically: an agent that is quietly healthy generates no
 *    graph traffic at all.
 *  - ...but a 1 s re-assert timer re-checks the node, because the agent node is created LAZILY by
 *    AgentInfoAPI's first heartbeat (~1 s after start) and can be re-created later after a
 *    peer-driven cleanup. A transition published before the node exists must not be lost.
 *  - Main thread only. Both sources (QState::entered and the presence coordinator's lifecycle
 *    transitions) fire on the main thread, which is also where AgentInfoAPI marshals its heartbeat
 *    write, so we never interleave with the other writer of this node.
 *  - Header-only on purpose: every agent already compiles agent_presence_coordinator.cpp, so
 *    nothing has to change in ten CMakeLists.
 */

#include <dsr/api/dsr_api.h>

#include <QAbstractState>
#include <QObject>
#include <QStateMachine>
#include <QTimer>

#include <memory>
#include <string>

#include "agent_status.h"

namespace rc
{

class AgentStatePublisher
{
public:
    AgentStatePublisher() = default;
    ~AgentStatePublisher() { stop(); }

    AgentStatePublisher(const AgentStatePublisher &) = delete;
    AgentStatePublisher &operator=(const AgentStatePublisher &) = delete;

    // Main thread. Safe to call before the agent node exists — the re-assert timer will pick it up.
    void configure(std::shared_ptr<DSR::DSRGraph> graph)
    {
        graph_ = std::move(graph);
        if (not graph_)
            return;
        node_name_ = graph_->get_agent_name() + " " + std::to_string(graph_->get_agent_id());

        if (not reassert_timer_)
        {
            reassert_timer_ = std::make_unique<QTimer>();
            QObject::connect(reassert_timer_.get(), &QTimer::timeout,
                             reassert_timer_.get(), [this]() { publish(); });
            reassert_timer_->start(kReassertPeriodMs);
        }
        publish();
    }

    // Observe the generated external FSM. Discovery is generic — each GRAFCETStep setObjectName()s
    // itself and QStateMachine::addState reparents it — so this keeps working across genericworker
    // regeneration, which is why we do NOT touch that file.
    void attach_state_machine(QStateMachine *sm)
    {
        if (sm == nullptr)
            return;
        for (auto *state : sm->findChildren<QAbstractState *>(Qt::FindDirectChildrenOnly))
        {
            const auto fsm = agent_status::fsm_from_name(state->objectName().toStdString());
            if (fsm == agent_status::Fsm::Unknown)
                continue;
            QObject::connect(state, &QAbstractState::entered, state, [this, fsm]() { set_fsm(fsm); });
        }
        // The machine may already be running (initialize() runs inside the Initialize step), so seed
        // from the current configuration instead of waiting for the next transition.
        for (const auto *state : sm->configuration())
            if (const auto fsm = agent_status::fsm_from_name(state->objectName().toStdString());
                fsm != agent_status::Fsm::Unknown)
                set_fsm(fsm);
    }

    void set_fsm(const agent_status::Fsm f)
    {
        if (f == fsm_)
            return;
        fsm_ = f;
        publish();
    }

    void set_presence(const agent_status::Presence p)
    {
        if (p == presence_)
            return;
        presence_ = p;
        publish();
    }

    void stop()
    {
        if (reassert_timer_)
            reassert_timer_->stop();
        reassert_timer_.reset();
    }

private:
    static constexpr int kReassertPeriodMs = 1000;

    // Idempotent: writes only when the node's attributes differ from what we want, so the steady
    // state costs one get_node() per second and produces no graph updates.
    void publish()
    {
        if (not graph_)
            return;
        auto node = graph_->get_node(node_name_);
        if (not node.has_value())
            return;   // AgentInfoAPI has not created it yet; the next tick will retry

        const std::string fsm{agent_status::name_of(fsm_)};
        const std::string presence{agent_status::name_of(presence_)};
        const std::string color{agent_status::color_for(fsm_, presence_)};

        bool dirty = false;
        auto &n = node.value();
        if (const auto cur = graph_->get_attrib_by_name<agent_fsm_state_att>(n);
            not cur.has_value() or cur.value() != fsm)
        {
            graph_->add_or_modify_attrib_local<agent_fsm_state_att>(n, fsm);
            dirty = true;
        }
        if (const auto cur = graph_->get_attrib_by_name<agent_presence_state_att>(n);
            not cur.has_value() or cur.value() != presence)
        {
            graph_->add_or_modify_attrib_local<agent_presence_state_att>(n, presence);
            dirty = true;
        }
        if (const auto cur = graph_->get_attrib_by_name<color_att>(n);
            not cur.has_value() or cur.value().get() != color)
        {
            graph_->add_or_modify_attrib_local<color_att>(n, color);
            dirty = true;
        }
        if (dirty)
            graph_->update_node(n);
    }

    std::shared_ptr<DSR::DSRGraph> graph_;
    std::string node_name_;
    agent_status::Fsm fsm_ = agent_status::Fsm::Unknown;
    agent_status::Presence presence_ = agent_status::Presence::Unknown;
    std::unique_ptr<QTimer> reassert_timer_;
};

}   // namespace rc
