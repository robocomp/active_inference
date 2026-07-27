#include "specificworker.h"

#include <cstdint>
#include <print>

#include <QDateTime>   // wall-clock ms for the cold-start stall grace

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

// ─── Primary-input (masks) stream gate ───────────────────────────────────────────────────────────
// Mirrors table_concept/specificworker_presence.cpp (masks_stream_ready / masks_stream_stalled /
// masks_stream_live), keyed on bottle's PRIMARY input (the voxelizer `masks` node).

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

// ─── Owned-node teardown ───────────────────────────────────────────────────────

void SpecificWorker::remove_owned_bottle_nodes()
{
    if (not G)
        return;

    // Bottle instances are DSR `object` nodes named "bottle_*". Deleting the node
    // also drops its room→bottle RT edge, so no separate edge cleanup is needed.
    // Sweep the new "object" type AND the legacy "cylinder" type (transition reap of nodes
    // leaked by a pre-migration run) — both filtered to "bottle*".
    std::vector<std::uint64_t> to_delete;
    for (const char* ty : {"object", "cylinder"})
        for (const auto& node : G->get_nodes_by_type(ty))
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

    // Sweep affordance nodes parented to a bottle cylinder (robust to renames / orphans), while the
    // cylinder parents still exist for the parent-type lookup, before the [Owns] nodes are deleted.
    remove_stale_affordance_nodes();

    // Stops the presence monitor FIRST, then deletes the [Owns] nodes ("bottle*" + the
    // agent node) — so every bottle cylinder is torn down here, after monitoring is off
    // (no separate remove_owned_bottle_nodes(): that ran the deletes while the monitor was
    // still live, and is only needed for the startup stale-sweep in initialize()).
    presence_coordinator_.cleanup_owned_nodes();
}

// Remove every "affordance" node whose parent object is a bottle cylinder — i.e. this agent's
// affordances, including stale ones left by a crashed previous run (whatever their renamed name).
// Keyed on the stable parent TYPE, not the affordance node name. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    for (const auto& aff : G->get_nodes_by_type("affordance"))
    {
        // Ours if EITHER (a) the name matches our deterministic "aff_<bottle_*>" scheme — orphan-safe,
        // catches affordances whose parent cylinder was already deleted (tracker DEATH / crashed run) —
        // OR (b) it still hangs from a live cylinder (backstop for a DSR collision-renamed node).
        bool ours = aff.name().starts_with("aff_bottle");
        const char* why = "name";
        if (not ours)
            if (const auto pid = G->get_attrib_by_name<parent_att>(aff); pid.has_value())
                if (const auto parent = G->get_node(pid.value());
                    parent.has_value() and
                    (parent->type() == "cylinder" or
                     (parent->type() == "object" and parent->name().starts_with("bottle"))))
                {
                    ours = true;
                    why = "parent bottle";
                }
        if (ours)
        {
            qInfo() << "[bottle_concept] removing affordance node"
                    << QString::fromStdString(aff.name()) << "id" << aff.id() << "(" << why << ")";
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
