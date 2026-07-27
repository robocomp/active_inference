#include "specificworker.h"

#include <print>
#include <vector>

#include <dsr/api/dsr_api.h>

// ── GRAFCET state hooks → the shared presence coordinator (copy the level-1 protocol) ──

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// Startup stale-sweep: remove any leftover dining_set nodes this agent owns from a previous
// (crashed) run so it always starts from a clean slate. Deleting the node also drops its edges.
// Main-thread only (graph access). Mirrors the level-1 agents' remove_owned_*_nodes().
//
// ★Keys on type()=="object" + the owned NAME prefix, NOT on a per-class node type. There is no
// `dining_set` node type registered in cortex (and there must not be — the schema convention is one
// generic `object` type with the class in object_subtype), so sweeping by type would be a silent
// no-op and every crashed run would leak its node into the shared graph.
void SpecificWorker::remove_owned_dining_set_nodes()
{
    if (not G)
        return;

    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("object"))
        if (node.name().rfind(cfg_.node_prefix, 0) == 0)
            to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("ring_metaconcept: removed stale {}* node id={}\n", cfg_.node_prefix, id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Remove every dining_set node this agent owns (M3+ births them; nothing to remove yet at M1).
    remove_owned_dining_set_nodes();

    presence_coordinator_.cleanup_owned_nodes();
}

void SpecificWorker::on_optional_peer_lost(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
