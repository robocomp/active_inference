#include "specificworker.h"

#include <QString>
#include <QTimer>

#include <dsr/api/dsr_api.h>

// ── GRAFCET state hooks → the shared presence coordinator (copy the level-1 protocol) ──

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// [Owns] = ["language_agent 25", "chat_intention"]: the shared coordinator deletes both (it stops the monitor first).
// No bespoke loop in between — CONCEPT_AGENT_LIFECYCLE.md §5 (a bespoke sweep races the live monitor).
void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    presence_coordinator_.cleanup_owned_nodes();
}

void SpecificWorker::on_optional_peer_lost(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
    if (name == "controller" and session_) session_->set_controller_present(false);
    // Stale (node still there, same process) keeps the graph; gone or replaced leaves it (specificworker.h).
    for (auto* m : {&em_, &ltom_})
        if (name == m->peer) sync_memory(*m, false);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t id)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
    if (name == "controller" and session_) session_->set_controller_present(true);
    for (auto* m : {&em_, &ltom_})
        if (name == m->peer)
        {
            m->peer_id = id;
            sync_memory(*m, true);
        }
}

void SpecificWorker::on_peer_restarted(std::uint32_t id)
{
    qInfo() << "[Presence] peer" << id << "restarted";
    // A new process re-seeded its domain: sync_memory sees the new stamp, drops the old graph and joins the new one.
    // Deferred: this hook can run inside the presence monitor's tick, and the join blocks for ~1 s.
    for (auto* m : {&em_, &ltom_})
        if (m->peer_id != 0 and id == m->peer_id)
            QTimer::singleShot(0, this, [this, m]() { sync_memory(*m, true); });
}

// ── memory graphs: read-only, each only while its owner is present (specificworker.h) ──
std::optional<std::uint64_t> SpecificWorker::process_stamp(const LazyMemory& m) const
{
    if (not G or m.peer_id == 0) return std::nullopt;
    for (const auto& n : G->get_nodes_by_type("agent"))
        if (const auto id = G->get_attrib_by_name<agent_id_att>(n); id.has_value() and std::uint32_t(id.value()) == m.peer_id)
            if (const auto ts = G->get_attrib_by_name<timestamp_creation_att>(n); ts.has_value())
                return std::uint64_t(ts.value());
    return std::nullopt;
}

void SpecificWorker::sync_memory(LazyMemory& m, bool may_join)
{
    if (shutting_down_ or m.peer_id == 0) return;
    const auto stamp = process_stamp(m);
    if (m.graph and (not stamp or *stamp != m.joined_stamp))
    {
        qInfo("[%s] %s %s", m.tag.c_str(), m.peer.c_str(), stamp ? "restarted (new process)" : "node is gone");
        leave_memory(m);
    }
    if (m.graph or not may_join or not stamp) return;
    m.joined_stamp = *stamp;
    qInfo("[%s] %s present: joining its memory graph on domain %d", m.tag.c_str(), m.peer.c_str(), m.domain);
    // configFile "" = request the graph from the peer (blocks ~1 s). Same name/id as on domain 0: a different DDS
    // domain, and nothing is ever written here, so no CRDT actor clash.
    m.graph = std::make_shared<DSR::DSRGraph>(agent_name, static_cast<std::uint32_t>(agent_id), std::string{}, true,
                                              static_cast<std::int8_t>(m.domain));
    if (session_) m.hand_to_session(m.graph);
    qInfo("[%s] joined domain %d", m.tag.c_str(), m.domain);
}

void SpecificWorker::leave_memory(LazyMemory& m)
{
    if (not m.graph) return;
    if (session_) m.hand_to_session(nullptr);
    m.graph.reset();   // the session held the only other reference; ~DSRGraph removes the DDS participant
    qInfo("[%s] left domain %d (%s not available)", m.tag.c_str(), m.domain, m.peer.c_str());
}
