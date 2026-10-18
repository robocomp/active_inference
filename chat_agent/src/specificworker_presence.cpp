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

// [Owns] = ["chat_agent 25", "chat_intention"]: the shared coordinator deletes both (it stops the monitor first).
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
    if (name == "episodic_agent") sync_em(false);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t id)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
    if (name == "controller" and session_) session_->set_controller_present(true);
    if (name == "episodic_agent")
    {
        em_peer_id_ = id;
        sync_em(true);
    }
}

void SpecificWorker::on_peer_restarted(std::uint32_t id)
{
    qInfo() << "[Presence] peer" << id << "restarted";
    if (em_peer_id_ == 0 or id != em_peer_id_) return;
    // A new episodic_agent process re-seeded domain 4: sync_em sees the new stamp, drops the old graph and joins the
    // new one. Deferred: this hook can run inside the presence monitor's tick, and the join blocks for ~1 s.
    QTimer::singleShot(0, this, [this]() { sync_em(true); });
}

// ── episodic memory: a read-only domain-4 graph while episodic_agent is present (specificworker.h) ──
std::optional<std::uint64_t> SpecificWorker::em_process_stamp() const
{
    if (not G or em_peer_id_ == 0) return std::nullopt;
    for (const auto& n : G->get_nodes_by_type("agent"))
        if (const auto id = G->get_attrib_by_name<agent_id_att>(n); id.has_value() and std::uint32_t(id.value()) == em_peer_id_)
            if (const auto ts = G->get_attrib_by_name<timestamp_creation_att>(n); ts.has_value())
                return std::uint64_t(ts.value());
    return std::nullopt;
}

void SpecificWorker::sync_em(bool may_join)
{
    if (shutting_down_ or em_peer_id_ == 0) return;
    const auto stamp = em_process_stamp();
    if (em_G_ and (not stamp or *stamp != em_joined_stamp_))
    {
        qInfo(stamp ? "[EM] episodic_agent restarted (new process)" : "[EM] episodic_agent's node is gone");
        leave_em();
    }
    if (em_G_ or not may_join or not stamp) return;
    em_joined_stamp_ = *stamp;
    qInfo("[EM] episodic_agent present: joining its memory graph on domain %d", em_domain_);
    // configFile "" = request the graph from the peer (blocks ~1 s). Same name/id as on domain 0: a different DDS
    // domain, and nothing is ever written here, so no CRDT actor clash.
    em_G_ = std::make_shared<DSR::DSRGraph>(agent_name, static_cast<std::uint32_t>(agent_id), std::string{}, true,
                                            static_cast<std::int8_t>(em_domain_));
    if (session_) session_->set_em_graph(em_G_);
    qInfo("[EM] joined domain %d", em_domain_);
}

void SpecificWorker::leave_em()
{
    if (not em_G_) return;
    if (session_) session_->set_em_graph(nullptr);
    em_G_.reset();   // the session held the only other reference; ~DSRGraph removes the DDS participant
    qInfo("[EM] left domain %d (episodic memory not available)", em_domain_);
}
