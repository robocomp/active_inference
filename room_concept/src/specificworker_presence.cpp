#include "specificworker.h"

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

    cleanup_room_graph_nodes();
    presence_coordinator_.cleanup_owned_nodes();
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

    if (affordance_manager_.release_execution_claim(G))
    {
        epistemic_controller_.epistemic_planner().clear_target();
        qWarning() << "[Presence] released stale afford_room execution claim after controller loss";
    }
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
