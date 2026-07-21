#include "agent_status_overlay.h"

#include "agent_status.h"

#include <dsr/api/dsr_api.h>
#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include <dsr/gui/viewers/graph_viewer/graph_node.h>

#include <QObject>

#include <string>

namespace rc
{

AgentStatusOverlay::~AgentStatusOverlay()
{
    stop();
}

void AgentStatusOverlay::start(std::shared_ptr<DSR::DSRGraph> graph, DSR::GraphViewer *viewer,
                               const int stale_after_ms)
{
    if (not graph or viewer == nullptr)
        return;
    graph_ = std::move(graph);
    viewer_ = viewer;
    stale_after_ns_ = static_cast<std::uint64_t>(stale_after_ms) * 1000000ULL;

    stop();
    timer_ = std::make_unique<QTimer>();
    QObject::connect(timer_.get(), &QTimer::timeout, timer_.get(), [this]() { refresh(); });
    timer_->start(kPeriodMs);
}

void AgentStatusOverlay::stop()
{
    if (timer_)
        timer_->stop();
    timer_.reset();
}

void AgentStatusOverlay::refresh()
{
    if (not graph_ or viewer_ == nullptr)
        return;

    const auto gmap = viewer_->getGMap();
    const std::uint64_t now = get_unix_timestamp();

    for (const auto &node : graph_->get_nodes_by_type("agent"))
    {
        const auto it = gmap.find(node.id());
        if (it == gmap.end() or it->second == nullptr)
            continue;   // not drawn (yet)
        auto *gnode = it->second;

        const auto beat = graph_->get_attrib_by_name<timestamp_agent_att>(node);
        // A node with no heartbeat at all has never been seen alive by this viewer — treat it as
        // stale rather than inventing a healthy colour for it.
        const bool stale = not beat.has_value() or now < beat.value()
                           or (now - beat.value()) > stale_after_ns_;

        const auto fsm = graph_->get_attrib_by_name<agent_fsm_state_att>(node);
        const auto presence = graph_->get_attrib_by_name<agent_presence_state_att>(node);

        const std::string color =
            stale ? std::string{agent_status::STALE_COLOR}
                  : std::string{agent_status::color_for(
                        agent_status::fsm_from_name(fsm.value_or("")),
                        agent_status::presence_from_name(presence.value_or("")))};

        // Only on change: set_color() restarts the pulse animation, so re-applying it every second
        // would leave every agent node permanently blinking.
        if (gnode->getColor() != color)
            gnode->set_color(color);

        std::string label = node.name();
        if (stale)
            label += "  —  stale";
        else
            label += "  —  " + fsm.value_or("?") + "/" + presence.value_or("?");
        gnode->setTag(label);   // idempotent (see GraphNode::setTag)
    }
}

}   // namespace rc
