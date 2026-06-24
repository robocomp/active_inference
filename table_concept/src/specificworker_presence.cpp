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

    if (G and fitter_)
    {
        std::vector<std::pair<std::uint64_t, std::string>> affordance_nodes;
        std::vector<std::pair<std::uint64_t, std::string>> table_nodes;

        affordance_nodes.reserve(fitter_->instances().size());
        table_nodes.reserve(fitter_->instances().size());

        for (const auto& [id, inst] : fitter_->instances())
        {
            // Delete the affordance node whenever it EXISTS (node_id != 0), not
            // only when the FSM is active: is_active() is (state != idle), so an
            // affordance that created its DSR node but has since returned to idle
            // would otherwise be left orphaned (its parent table node is deleted
            // below), leaving a dangling table_afford node + edge in the graph.
            if (inst.affordance.node_id() != 0)
                affordance_nodes.emplace_back(inst.affordance.node_id(), inst.node_name);
            table_nodes.emplace_back(id, inst.node_name);
        }

        // Remove affordance nodes first (children of table nodes)
        for (const auto& [affordance_id, node_name] : affordance_nodes)
        {
            G->delete_node(affordance_id);
            qInfo() << "[table_concept] removed affordance node for"
                    << QString::fromStdString(node_name);
        }

        // Remove table nodes themselves
        for (const auto& [table_id, node_name] : table_nodes)
        {
            G->delete_node(table_id);
            qInfo() << "[table_concept] removed table node"
                    << QString::fromStdString(node_name);
        }
    }

    presence_coordinator_.cleanup_owned_nodes();
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
