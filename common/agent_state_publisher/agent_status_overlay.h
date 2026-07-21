#pragma once
/*
 * agent_status_overlay.h — observer side of the agent-health display.
 *
 * Each agent paints its OWN node via rc::AgentStatePublisher, which covers every state it can
 * report. It cannot report the one state you most need to see: "this agent died". A crashed or
 * hung agent stops writing, and its node simply freezes on whatever colour it last published —
 * green, if it died while healthy. So the observer has to decide that, from the age of the
 * heartbeat AgentInfoAPI keeps refreshing (timestamp_agent, nanoseconds).
 *
 * This overlay runs in whichever agent shows the full graph (robot_concept) and, once a second:
 *   - greys out agent nodes whose heartbeat has gone stale,
 *   - restores the published colour when they come back,
 *   - keeps each node's label as "<name> <id> — <fsm>/<presence>" so the state is readable as text
 *     and not only as a colour.
 *
 * It is the single authority for agent-node colour in this viewer (it re-applies the published
 * colour itself rather than relying on an update signal reaching the node), which keeps the stale
 * and live paths from fighting each other.
 *
 * Declaration-only on purpose: the dsr_gui headers define an AbstractGraphicViewer that collides
 * with the component-local one of the same name, so they must stay inside a single .cpp and never
 * leak through specificworker.h. Main thread only — it touches Qt graphics items.
 */

#include <QTimer>

#include <cstdint>
#include <memory>

namespace DSR
{
class DSRGraph;
class GraphViewer;
}

namespace rc
{

class AgentStatusOverlay
{
public:
    AgentStatusOverlay() = default;
    ~AgentStatusOverlay();

    AgentStatusOverlay(const AgentStatusOverlay &) = delete;
    AgentStatusOverlay &operator=(const AgentStatusOverlay &) = delete;

    // `stale_after_ms` is a DISPLAY horizon and is deliberately shorter than the presence protocol's
    // heartbeat_timeout_ms: greying a node early costs nothing and is undone as soon as the
    // heartbeat resumes, whereas acting on a short silence is what proved hypersensitive.
    void start(std::shared_ptr<DSR::DSRGraph> graph, DSR::GraphViewer *viewer, int stale_after_ms);
    void stop();

private:
    static constexpr int kPeriodMs = 1000;

    void refresh();

    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::GraphViewer *viewer_ = nullptr;
    std::uint64_t stale_after_ns_ = 3000ULL * 1000000ULL;
    std::unique_ptr<QTimer> timer_;
};

}   // namespace rc
