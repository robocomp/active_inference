#include "specificworker.h"

#include <print>
#include <vector>

// ─── Presence state-machine step hooks (delegated to the coordinator) ──────────

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// ─── Owned-node teardown ───────────────────────────────────────────────────────

void SpecificWorker::remove_owned_person_nodes()
{
    if (not G)
        return;
    // Person instances are DSR `person` nodes named "person_*". Deleting the node also drops its
    // room→person RT edge, so no separate edge cleanup is needed.
    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("person"))
        if (node.name().starts_with("person"))
            to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("human_concept: removed person node id={}\n", id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    remove_stale_affordance_nodes();
    presence_coordinator_.cleanup_owned_nodes();
}

// Remove every "affordance" node whose parent object is a person — i.e. this agent's affordances,
// including stale ones left by a crashed previous run. Keyed on the stable parent TYPE.
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    for (const auto& aff : G->get_nodes_by_type("affordance"))
    {
        const auto pid = G->get_attrib_by_name<parent_att>(aff);
        if (not pid.has_value())
            continue;
        const auto parent = G->get_node(pid.value());
        if (parent.has_value() and parent->type() == "person")
        {
            qInfo() << "[human_concept] removing affordance node"
                    << QString::fromStdString(aff.name()) << "id" << aff.id() << "(parent person)";
            G->delete_node(aff.id());
        }
    }
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
