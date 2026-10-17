#include "specificworker.h"

#include <QString>

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

// Informational only: the journal poll itself notices the controller node coming and going (Builder::owner_lost).
void SpecificWorker::on_optional_peer_lost(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name);
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
}
