#include "specificworker.h"

#include <QString>

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
}

void SpecificWorker::on_optional_peer_ready(const std::string& name, std::uint32_t /*id*/)
{
    qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name);
    if (name == "controller" and session_) session_->set_controller_present(true);
}
