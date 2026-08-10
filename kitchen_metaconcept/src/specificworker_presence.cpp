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

// Startup stale-sweep: remove any leftover kitchen_* nodes this agent owns from a previous (crashed)
// run so it always starts from a clean slate. Deleting the node also drops its edges. Main-thread
// only (graph access). Mirrors the level-1 agents' remove_owned_*_nodes().
//
// ★At M2 this agent creates NOTHING, so the sweep should always find nothing — it is here from the
// start deliberately. A `kill -9` cannot be caught, so a future milestone's node would leak into the
// persistent graph with no process alive to own it, and this sweep is the only thing that reaps it
// (CLAUDE.md). Adding it after the first node is born is how you get an immortal phantom.
//
// Sweeps BOTH types keyed on the owned NAME prefix: `metaconcept` is where a rig node belongs (it
// must stay out of everyone's get_nodes_by_type("object") sweep), and `object` is swept too so a
// node born by a run from before any retype is still reaped.
void SpecificWorker::remove_owned_kitchen_nodes()
{
    if (not G)
        return;

    std::vector<std::uint64_t> to_delete;
    for (const std::string type : {"metaconcept", "object"})
        for (const auto& node : G->get_nodes_by_type(type))
            if (node.name().rfind(cfg_.node_prefix, 0) == 0)
                to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("kitchen_metaconcept: removed stale {}* node id={}\n", cfg_.node_prefix, id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Nothing is owned at M2; the sweep is the contract a later milestone inherits for free.
    remove_owned_kitchen_nodes();

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
