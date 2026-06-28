#include "specificworker.h"

#include <print>

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

void SpecificWorker::remove_owned_table_nodes()
{
    if (not G)
        return;

    // Table instances are DSR `table` nodes named "table_*". Deleting the node also drops its
    // room→table RT edge, so no separate edge cleanup is needed. Startup stale-sweep (mirrors bottle).
    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("table"))
        if (node.name().starts_with("table"))
            to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("table_concept: removed table node id={}\n", id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Sweep every affordance node parented to a table (robust to renames / orphans not tracked in
    // instances_), while the table parents still exist for the parent-type lookup.
    remove_stale_affordance_nodes();

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

// Remove every "affordance" node whose parent object is a table — i.e. this agent's affordances,
// including stale ones left by a crashed previous run (whatever their possibly-renamed node name).
// Keyed on the stable parent TYPE, not the affordance node name. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    for (const auto& aff : G->get_nodes_by_type("affordance"))
    {
        // Ours if EITHER (a) the name matches our deterministic "aff_table*" scheme — orphan-safe,
        // catches affordances whose parent table node was already deleted (retirement / crashed run) —
        // OR (b) it still hangs from a live table (backstop for a DSR collision-renamed node).
        bool ours = aff.name().starts_with("aff_table");
        const char* why = "name";
        if (not ours)
            if (const auto pid = G->get_attrib_by_name<parent_att>(aff); pid.has_value())
                if (const auto parent = G->get_node(pid.value());
                    parent.has_value() and parent->type() == "table")
                {
                    ours = true;
                    why = "parent table";
                }
        if (ours)
        {
            qInfo() << "[table_concept] removing affordance node"
                    << QString::fromStdString(aff.name()) << "id" << aff.id() << "(" << why << ")";
            G->delete_node(aff.id());
        }
    }
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
