#include "specificworker.h"

#include "../../common/owned_nodes/owned_nodes.h"   // rc::owned:: (SHARED ownership sweep)

#include <print>
#include <vector>

// ─── Presence state-machine step hooks (delegated to the coordinator) ──────────

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }


// Delete this agent's instance nodes: the SHARED ownership sweep (common/owned_nodes) over the types
// this agent declares. Both the startup stale-sweep (recover from a crashed previous run) and the
// shutdown path go through it, so the two can never disagree. Main-thread only (graph access).
void SpecificWorker::remove_owned_person_nodes()
{
    if (not G)
        return;
    static const rc::owned::Spec kOwned{
        .agent = "human_concept",
        .name_prefix = "person",
        .node_types = {"person"},
        .legacy_parent_types = {"person"},
    };
    rc::owned::remove_instance_nodes(*G, kOwned);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    remove_stale_affordance_nodes();
    presence_coordinator_.cleanup_owned_nodes();
}


// Remove every "affordance" node this agent owns, including stale ones left by a crashed previous run.
// SHARED (common/owned_nodes): the deterministic "aff_person*" name test — orphan-safe, it still
// fires after the parent node is gone — OR the parent lookup. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    static const rc::owned::Spec kOwned{
        .agent = "human_concept",
        .name_prefix = "person",
        .node_types = {"person"},
        .legacy_parent_types = {"person"},
    };
    rc::owned::remove_stale_affordances(*G, kOwned);
}

// ─── Optional-peer notifications ───────────────────────────────────────────────

void SpecificWorker::on_optional_peer_lost(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
