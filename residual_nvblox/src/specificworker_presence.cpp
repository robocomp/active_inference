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

void SpecificWorker::remove_owned_residual_nodes()
{
    if (not G)
        return;
    // Residual obstacles are DSR `obstacle` nodes named "residual_*". Deleting the node also drops its
    // room→node RT edge, so no separate edge cleanup is needed. (Only OUR "residual_*" ones — the
    // controller's own temp "obstacle" nodes are left alone during the hybrid phase.)
    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("obstacle"))
        if (node.name().rfind("residual_", 0) == 0)
            to_delete.push_back(node.id());
    bool removed_any = false;
    for (const auto id : to_delete)
        if (G->delete_node(id))
        {
            std::print("residual_concept: removed residual node id={}\n", id);
            removed_any = true;
        }
    if (removed_any)                          // keep the graph tidy after the batch removal (no-op if headless)
        trigger_graph_layout_twopi();
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    // Explicitly delete every "residual_*" obstacle node this run created — the [Owns]-based coordinator
    // sweep doesn't reliably catch them (insert_node does not always add an Owns edge), so leftover
    // residuals would pollute the shared graph after exit. Delete by name+type while G is still alive
    // (this publishes del-deltas so the controller/peers drop them); then the coordinator stops the
    // monitor and removes the agent node.
    remove_owned_residual_nodes();
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
