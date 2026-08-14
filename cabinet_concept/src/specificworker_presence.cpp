/*
 * specificworker_presence.cpp — presence protocol + owned-node cleanup for cabinet_concept.
 *
 * The Waiting/Operating/Degraded state hooks are thin delegators to AgentPresenceCoordinator (the shared
 * bottle_concept protocol: debounced degraded, unique [Agent] id). The rest sweeps this agent's own DSR
 * nodes — "cabinet_*" instances and their child "affordance" nodes — both as a one-time startup stale-sweep
 * (recover from a crashed previous run) and on shutdown, so the graph never accumulates orphans. Main-thread
 * only: all graph access here runs on the main thread. See CLAUDE.md (presence protocol / startup join).
 */

#include "specificworker.h"

#include "../../common/owned_nodes/owned_nodes.h"   // rc::owned:: (SHARED ownership sweep)

#include <cstdint>
#include <print>

#include <QDateTime>   // wall-clock ms for the cold-start stall grace

// ─── Presence state delegators ───────────────────────────────────────────────────────────────────

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

// ─── Primary-input (masks) stream gate ───────────────────────────────────────────────────────────
// Mirrors room_concept/specificworker_presence.cpp:14-41 (lidar_stream_ready / lidar_stream_stalled),
// keyed on cabinet's PRIMARY input (the voxelizer `masks` node) instead of a LiDAR media plane.

// Admission probe: the masks producer is reachable (node present + carrying a frame id). Usable from Waiting.
bool SpecificWorker::masks_stream_ready(std::string *detail) const
{
    if (not mask_ingestor_) { if (detail) *detail = "MaskIngestor not constructed yet"; return false; }
    return mask_ingestor_->stream_ready(detail);
}

// Operating stall predicate: no NEW masks frame for longer than the timeout. Before the first frame ever
// arrives (age < 0) the grace is measured from Operating entry, so producer startup isn't misread as a stall.
bool SpecificWorker::masks_stream_stalled(std::int64_t *age_ms_out) const
{
    const int timeout = cfg_.masks_stall_timeout_ms;
    if (timeout <= 0 or not mask_ingestor_) return false;   // 0 ⇒ gate disabled
    const std::int64_t age = mask_ingestor_->ms_since_last_frame();
    if (age_ms_out) *age_ms_out = age;
    if (age < 0)   // no frame ever — measure from Operating entry (the producer may be mid-startup)
    {
        const std::int64_t since_entry = QDateTime::currentMSecsSinceEpoch() - operating_since_ms_;
        return operating_since_ms_ > 0 and since_entry > timeout;
    }
    return age > timeout;
}

// Admission: the producer is currently LIVE (a fresh masks frame within the timeout window). Unlike the
// node-exists probe, this requires actual freshness — so a persisting-but-dead `masks` node cannot re-admit
// the agent into an instant re-stall. Cold start (age<0, no frame yet) reads as not-live until the first
// frame is ingested (on_waiting_loop pumps refresh() so that happens while Waiting). Gate off ⇒ node-exists.
bool SpecificWorker::masks_stream_live() const
{
    if (not mask_ingestor_) return false;
    const int timeout = cfg_.masks_stall_timeout_ms;
    if (timeout <= 0) return mask_ingestor_->stream_ready();
    const std::int64_t age = mask_ingestor_->ms_since_last_frame();
    return age >= 0 and age < timeout;
}


// Delete this agent's instance nodes: the SHARED ownership sweep (common/owned_nodes) over the types
// this agent declares. Both the startup stale-sweep (recover from a crashed previous run) and the
// shutdown path go through it, so the two can never disagree. Main-thread only (graph access).
void SpecificWorker::remove_owned_cabinet_nodes()
{
    if (not G)
        return;
    static const rc::owned::Spec kOwned{
        .agent = "cabinet_concept",
        .name_prefix = "cabinet",
        .node_types = {"object", "box"},
        .legacy_parent_types = {},
    };
    rc::owned::remove_instance_nodes(*G, kOwned);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Sweep every affordance node parented to a cabinet (robust to renames / orphans not tracked in
    // instances_), while the cabinet parents still exist for the parent-type lookup.
    remove_stale_affordance_nodes();

    if (G and fitter_)
    {
        std::vector<std::pair<std::uint64_t, std::string>> affordance_nodes;
        std::vector<std::pair<std::uint64_t, std::string>> cabinet_nodes;

        affordance_nodes.reserve(fitter_->instances().size());
        cabinet_nodes.reserve(fitter_->instances().size());

        for (const auto& [id, inst] : fitter_->instances())
        {
            // Delete the affordance node whenever it EXISTS (node_id != 0), not
            // only when the FSM is active: is_active() is (state != idle), so an
            // affordance that created its DSR node but has since returned to idle
            // would otherwise be left orphaned (its parent cabinet node is deleted
            // below), leaving a dangling cabinet_afford node + edge in the graph.
            if (inst.affordance.node_id() != 0)
                affordance_nodes.emplace_back(inst.affordance.node_id(), inst.node_name);
            cabinet_nodes.emplace_back(id, inst.node_name);
        }

        // Remove affordance nodes first (children of cabinet nodes)
        for (const auto& [affordance_id, node_name] : affordance_nodes)
        {
            G->delete_node(affordance_id);
            qInfo() << "[cabinet_concept] removed affordance node for"
                    << QString::fromStdString(node_name);
        }

        // Remove cabinet nodes themselves
        for (const auto& [cabinet_id, node_name] : cabinet_nodes)
        {
            G->delete_node(cabinet_id);
            qInfo() << "[cabinet_concept] removed cabinet node"
                    << QString::fromStdString(node_name);
        }

        // Stage 2 kitchen model owns its own set of box nodes (keyed by cell, not in fitter_->instances()).
        for (const auto& [cell_id, node_id] : kitchen_nodes_)
        {
            G->delete_node(node_id);
            qInfo() << "[cabinet_concept] removed kitchen cell node" << QString::fromStdString(cell_id);
        }
        kitchen_nodes_.clear();
    }

    presence_coordinator_.cleanup_owned_nodes();
}


// Remove every "affordance" node this agent owns, including stale ones left by a crashed previous run.
// SHARED (common/owned_nodes): the deterministic "aff_cabinet*" name test — orphan-safe, it still
// fires after the parent node is gone — OR the parent lookup. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    static const rc::owned::Spec kOwned{
        .agent = "cabinet_concept",
        .name_prefix = "cabinet",
        .node_types = {"object", "box"},
        .legacy_parent_types = {},
    };
    rc::owned::remove_stale_affordances(*G, kOwned);
}

// ─── Optional-peer hooks ─────────────────────────────────────────────────────────────────────────

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
