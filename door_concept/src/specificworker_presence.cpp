#include "specificworker.h"

#include "../../common/owned_nodes/owned_nodes.h"   // rc::owned:: (SHARED ownership sweep)
#include "../../common/stream_gate/stream_gate.h"   // rc::stream:: (SHARED primary-input gate)

#include <cstdint>
#include <print>

#include <QDateTime>   // wall-clock ms for the cold-start stall grace

// The one declaration of what this agent OWNS. It used to be repeated verbatim inside both sweep
// functions, which is two places for the node-type list to drift apart.
static const rc::owned::Spec kOwned{
    .agent = "door_concept",
    .name_prefix = "door",
    .node_types = {"object", "door"},
    .legacy_parent_types = {"door"},
};

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
// Mirrors table_concept/specificworker_presence.cpp, keyed on door's PRIMARY input (the retina
// `masks` node) instead of a LiDAR media plane.

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
    if (not mask_ingestor_) return false;
    const std::int64_t age = mask_ingestor_->ms_since_last_frame();
    if (age_ms_out) *age_ms_out = age;
    return rc::stream::stalled(age, cfg_.masks_stall_timeout_ms, presence_protocol_.operating_since_ms(),
                               QDateTime::currentMSecsSinceEpoch());
}

// Admission: the producer is currently LIVE (a fresh masks frame within the timeout window). Unlike the
// node-exists probe, this requires actual freshness — so a persisting-but-dead `masks` node cannot re-admit
// the agent into an instant re-stall. Cold start (age<0, no frame yet) reads as not-live until the first
// frame is ingested (on_waiting_loop pumps refresh() so that happens while Waiting). Gate off ⇒ node-exists.
bool SpecificWorker::masks_stream_live() const
{
    if (not mask_ingestor_) return false;
    // Gate off ⇒ defer to this agent's own node-exists probe; see the warning on rc::stream::live.
    if (not rc::stream::gate_enabled(cfg_.masks_stall_timeout_ms)) return mask_ingestor_->stream_ready();
    return rc::stream::live(mask_ingestor_->ms_since_last_frame(), cfg_.masks_stall_timeout_ms);
}

// Delete this agent's instance nodes: the SHARED ownership sweep (common/owned_nodes) over the types
// this agent declares. Both the startup stale-sweep (recover from a crashed previous run) and the
// shutdown path go through it, so the two can never disagree. Main-thread only (graph access).
void SpecificWorker::remove_owned_door_nodes()
{
    if (not G)
        return;
    rc::owned::remove_instance_nodes(*G, kOwned);
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;

    // Which door was which, written down before the nodes go. The graph cannot carry it (we delete our own
    // nodes on the way out by contract), so etc/door_identities.csv is what the next run reads to keep
    // door_1 on the same physical door instead of renumbering by whatever it sees first.
    remember_live_identities();

    // Sweep every affordance node parented to a door (robust to renames / orphans not tracked in
    // instances_), while the door parents still exist for the parent-type lookup.
    remove_stale_affordance_nodes();

    // ★THE PER-INSTANCE SWEEP THAT USED TO BE HERE IS GONE (2026-08-16). It walked fitter_->instances()
    // and deleted each node by id — but [Owns].nodes already carries a WILDCARD ("door_*") that
    // matches every instance this agent ever names, so the coordinator's sweep below deletes exactly the same set. Worse,
    // the loop ran those deletes while the presence MONITOR was still live; bottle_concept removed its copy
    // for that reason on 2026-07-31 and chair's own config comment says the same. Stopping the monitor first
    // and letting the shared sweep do it is the form all seven now share.

    presence_coordinator_.cleanup_owned_nodes();
}


// Remove every "affordance" node this agent owns, including stale ones left by a crashed previous run.
// SHARED (common/owned_nodes): the deterministic "aff_door*" name test — orphan-safe, it still
// fires after the parent node is gone — OR the parent lookup. Main-thread only (graph access).
void SpecificWorker::remove_stale_affordance_nodes()
{
    if (not G)
        return;
    rc::owned::remove_stale_affordances(*G, kOwned);
}

void SpecificWorker::on_optional_peer_lost(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string &name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
