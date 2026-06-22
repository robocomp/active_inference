#include "specificworker.h"

#include <print>

// ─── Presence state-machine step hooks (delegated to the coordinator) ──────────

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

// ─── Owned-node teardown ───────────────────────────────────────────────────────

void SpecificWorker::remove_owned_bottle_nodes()
{
    if (not G)
        return;

    // Bottle instances are DSR `cylinder` nodes named "bottle_*". Deleting the node
    // also drops its room→bottle RT edge, so no separate edge cleanup is needed.
    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("cylinder"))
        if (node.name().starts_with("bottle"))
            to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("bottle_concept: removed bottle node id={}\n", id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Stops the presence monitor FIRST, then deletes the [Owns] nodes ("bottle*" + the
    // agent node) — so every bottle cylinder is torn down here, after monitoring is off
    // (no separate remove_owned_bottle_nodes(): that ran the deletes while the monitor was
    // still live, and is only needed for the startup stale-sweep in initialize()).
    presence_coordinator_.cleanup_owned_nodes();
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
