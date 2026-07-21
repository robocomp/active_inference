#include "specificworker.h"

#include <format>
#include <print>

#include <QDateTime>

// ── LiDAR stream gate ───────────────────────────────────────────────────────────────────────────
// The localizer's ONLY evidence is the LiDAR sweep. With no stream it cannot stabilize, so the agent
// must advertise that honestly (Waiting + a log line) instead of sitting in Operating looking healthy
// while nothing converges. Two probes, because "no stream" has two distinct causes:
//   * the producer never advertised the plane  → graph-only descriptor probe, usable before any DDS;
//   * the plane is advertised but silent/dead   → frame-age probe, only meaningful once Operating.
bool SpecificWorker::lidar_stream_ready(std::string* why) const
{
    if (not lidar_ingestor_)
    {
        if (why) *why = "LidarIngestor not constructed yet";
        return false;
    }
    return lidar_ingestor_->stream_descriptor_available(why);
}

bool SpecificWorker::lidar_stream_stalled(std::int64_t* age_ms_out) const
{
    const int timeout = params.LIDAR_STALL_TIMEOUT_MS;
    if (timeout <= 0 or not lidar_ingestor_)
        return false;

    const std::int64_t age = lidar_ingestor_->ms_since_last_frame();
    if (age_ms_out) *age_ms_out = age;

    // Before the first sweep ever arrives, measure from Operating entry: subscriber discovery is
    // throttled and the producer may be mid-startup, so a cold Operating is not yet a stall.
    if (age < 0)
    {
        const std::int64_t since_entry = QDateTime::currentMSecsSinceEpoch() - operating_since_ms_;
        return operating_since_ms_ > 0 and since_entry > timeout;
    }
    return age > timeout;
}

void SpecificWorker::waiting_enter()
{
    presence_coordinator_.waiting_enter();
}

void SpecificWorker::waiting_loop()
{
    presence_coordinator_.waiting_loop();
}

void SpecificWorker::operating_enter()
{
    presence_coordinator_.operating_enter();
}

void SpecificWorker::operating_loop()
{
    presence_coordinator_.operating_loop();
}

void SpecificWorker::degraded_enter()
{
    presence_coordinator_.degraded_enter();
}

void SpecificWorker::degraded_loop()
{
    presence_coordinator_.degraded_loop();
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // In PreserveBootstrapRoom (static-room) mode the room/table are a pre-seeded prior this
    // agent does NOT own, so do not delete them on exit: skip both the type-based room cleanup
    // and the [Owns]-driven owned-node cleanup (which lists subtrees=["room"]). We still remove
    // our own agent presence node so peers detect our departure.
    if (params.PRESERVE_BOOTSTRAP_ROOM)
    {
        qInfo() << "[room] PreserveBootstrapRoom=true: keeping static room/table on exit (skipping owned-node deletion).";
    }
    else
    {
        scene_graph_->cleanup_room_graph_nodes();
        presence_coordinator_.cleanup_owned_nodes();
    }
    cleanup_self_agent_node();
}

void SpecificWorker::cleanup_self_agent_node()
{
    if (!G)
        return;

    const auto self_id = static_cast<std::uint32_t>(agent_id);
    bool deleted = false;

    // Preferred path: match by agent_id attribute on nodes of type "agent".
    for (const auto& node : G->get_nodes_by_type("agent"))
    {
        const auto id_att = G->get_attrib_by_name<agent_id_att>(node);
        if (!id_att.has_value() || id_att.value() != self_id)
            continue;

        if (G->delete_node(node))
        {
            qInfo() << "[Shutdown] removed self agent node by agent_id"
                    << "agent_id=" << self_id
                    << "node_name=" << QString::fromStdString(node.name());
        }
        else
        {
            qWarning() << "[Shutdown] failed to remove self agent node by agent_id"
                       << "agent_id=" << self_id
                       << "node_name=" << QString::fromStdString(node.name());
        }
        deleted = true;
        break;
    }

    if (deleted)
        return;

    // Fallback by legacy node names used in Owns.nodes.
    const std::string by_agent_name = std::string("room_concept ") + std::to_string(self_id);
    if (auto n = G->get_node(by_agent_name); n.has_value())
    {
        if (G->delete_node(n.value()))
            qInfo() << "[Shutdown] removed self agent node by name" << QString::fromStdString(by_agent_name);
        else
            qWarning() << "[Shutdown] failed to remove self agent node by name" << QString::fromStdString(by_agent_name);
    }
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);

    if (name != "controller" || !G)
        return;

    scene_graph_->on_controller_lost();
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
