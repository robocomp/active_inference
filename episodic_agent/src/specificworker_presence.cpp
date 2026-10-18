#include "specificworker.h"

#include <QString>

#include <dsr/api/dsr_api.h>

#include <print>

// ── GRAFCET state hooks → the shared presence coordinator (copied from chat_agent / the level-1 protocol) ──

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// [Owns] = ["episodic_agent 26"]: the shared coordinator deletes it (it stops the monitor first). The domain-4
// records are memory, not live beliefs, and stay; the archive reseeds them on the next start.
void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    presence_coordinator_.cleanup_owned_nodes();
}

// The journal poll notices the journal node disappearing (a graceful controller stop deletes it). A crash or kill -9
// leaves the node behind, so the presence monitor losing the `controller` peer closes what is open `lost` too
// (finding 5). last_wire_ is kept: the lingering wire must not be re-read as a live controller; a restarted
// controller writes a new epoch, which the poll sees as a changed wire.
void SpecificWorker::on_optional_peer_lost(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
    if (name != "controller" or shutting_down_ or not builder_) return;
    const auto lost = builder_->owner_lost(now_ms());
    std::println("[episodic] controller peer lost — {} open record(s) closed as lost", lost.size());
    write(lost);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}

// ── ltsm memory: a read-only domain-2 graph while ltsm_agent's participant is up (specificworker.h) ──
bool SpecificWorker::ltsm_connected() const
{
    return G and std::ranges::any_of(G->get_connected_agents(),
                                     [](const std::string& p) { return p.ends_with("( ltsm_agent )"); });
}

void SpecificWorker::sync_ltsm()
{
    if (shutting_down_) return;
    const bool up = ltsm_connected();
    if (G_ltsm_ and not up)
    {
        std::println("[episodic] ltsm_agent's participant is gone");
        leave_ltsm();
    }
    if (G_ltsm_ or not up) return;
    std::println("[episodic] ltsm_agent present: joining its memory graph on domain {}", ltsm_domain_);
    // configFile "" = request the graph from the peer (blocks ~1 s). Same name/id as on domains 0 and 4: a different
    // DDS domain, and nothing is ever written here, so no CRDT actor clash.
    G_ltsm_ = std::make_shared<DSR::DSRGraph>(agent_name, static_cast<std::uint32_t>(agent_id), std::string{}, true,
                                              static_cast<std::int8_t>(ltsm_domain_));
    if (facts_) facts_->set_ltsm(G_ltsm_);
    std::println("[episodic] joined domain {}: doors now carry their ltsm passage identity", ltsm_domain_);
}

void SpecificWorker::leave_ltsm()
{
    if (not G_ltsm_) return;
    if (facts_) facts_->set_ltsm(nullptr);
    G_ltsm_.reset();   // facts_ held the only other reference; ~DSRGraph removes the DDS participant
    std::println("[episodic] left domain {} (doors carry no ltsm_id until ltsm_agent returns)", ltsm_domain_);
}
