/*
 * specificworker_presence.cpp — presence protocol + owned-node cleanup for hood_concept.
 *
 * The Waiting/Operating/Degraded state hooks are thin delegators to AgentPresenceCoordinator (the shared
 * bottle_concept protocol: debounced degraded, unique [Agent] id). The rest sweeps this agent's own DSR
 * nodes — "hood_*" instances and their child "affordance" nodes — both as a one-time startup stale-sweep
 * (recover from a crashed previous run) and on shutdown, so the graph never accumulates orphans. Main-thread
 * only: all graph access here runs on the main thread. See CLAUDE.md (presence protocol / startup join).
 */

#include "specificworker.h"

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
// keyed on hood's PRIMARY input (the voxelizer `masks` node) instead of a LiDAR media plane.

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

// ─── Owned-node cleanup (startup stale-sweep + shutdown) ─────────────────────────────────────────

void SpecificWorker::remove_owned_hood_nodes()
{
    if (not G)
        return;

    // Hood instances are DSR `hood` nodes named "hood_*". Deleting the node also drops its
    // room→hood RT edge, so no separate edge cleanup is needed. Startup stale-sweep (mirrors bottle).
    std::vector<std::uint64_t> to_delete;
    for (const auto& node : G->get_nodes_by_type("object"))   // generic DSR type; name-prefix is the class filter
        if (node.name().starts_with("hood"))
            to_delete.push_back(node.id());

    for (const auto id : to_delete)
        if (G->delete_node(id))
            std::print("hood_concept: removed hood node id={}\n", id);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Sweep every affordance node parented to a hood (robust to renames / orphans not tracked in
    // instances_), while the hood parents still exist for the parent-type lookup.
    remove_stale_affordance_nodes();

    if (G and fitter_)
    {
        std::vector<std::pair<std::uint64_t, std::string>> affordance_nodes;
        std::vector<std::pair<std::uint64_t, std::string>> hood_nodes;

        affordance_nodes.reserve(fitter_->instances().size());
        hood_nodes.reserve(fitter_->instances().size());

        for (const auto& [id, inst] : fitter_->instances())
        {
            // Delete the affordance node whenever it EXISTS (node_id != 0), not
            // only when the FSM is active: is_active() is (state != idle), so an
            // affordance that created its DSR node but has since returned to idle
            // would otherwise be left orphaned (its parent hood node is deleted
            // below), leaving a dangling hood_afford node + edge in the graph.
            if (inst.affordance.node_id() != 0)
                affordance_nodes.emplace_back(inst.affordance.node_id(), inst.node_name);
            hood_nodes.emplace_back(id, inst.node_name);
        }

        // Remove affordance nodes first (children of hood nodes)
        for (const auto& [affordance_id, node_name] : affordance_nodes)
        {
            G->delete_node(affordance_id);
            qInfo() << "[hood_concept] removed affordance node for"
                    << QString::fromStdString(node_name);
        }

        // Remove hood nodes themselves
        for (const auto& [hood_id, node_name] : hood_nodes)
        {
            G->delete_node(hood_id);
            qInfo() << "[hood_concept] removed hood node"
                    << QString::fromStdString(node_name);
        }
    }

    presence_coordinator_.cleanup_owned_nodes();
}

// Remove every "affordance" node whose parent object is a hood — i.e. this agent's affordances,
// including stale ones left by a crashed previous run (whatever their possibly-renamed node name).
// Keyed on the stable parent TYPE, not the affordance node name. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    for (const auto& aff : G->get_nodes_by_type("affordance"))
    {
        // Ours if EITHER (a) the name matches our deterministic "aff_hood*" scheme — orphan-safe,
        // catches affordances whose parent hood node was already deleted (retirement / crashed run) —
        // OR (b) it still hangs from a live hood (backstop for a DSR collision-renamed node).
        bool ours = aff.name().starts_with("aff_hood");
        const char* why = "name";
        if (not ours)
            if (const auto pid = G->get_attrib_by_name<parent_att>(aff); pid.has_value())
                if (const auto parent = G->get_node(pid.value());
                    parent.has_value() and parent->type() == "object"
                    and parent->name().starts_with("hood"))
                {
                    ours = true;
                    why = "parent hood";
                }
        if (ours)
        {
            qInfo() << "[hood_concept] removing affordance node"
                    << QString::fromStdString(aff.name()) << "id" << aff.id() << "(" << why << ")";
            G->delete_node(aff.id());
        }
    }
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
