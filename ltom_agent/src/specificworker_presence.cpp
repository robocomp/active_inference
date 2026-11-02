#include "specificworker.h"

// -- GRAFCET state hooks -> the shared presence coordinator (same shape as episodic_agent / language_agent) --

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// [Owns] = ["ltom_agent 28", "ltom_recall_*"]: the agent node plus the recall nodes (trailing * = name prefix), so the
// shared coordinator's [Owns] deletion is the whole graceful cleanup (it stops the monitor first). The crash-recovery
// sweep of ltom_recall_* is sweep_stale_recalls() in specificworker.cpp.
void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    presence_coordinator_.cleanup_owned_nodes();
}
