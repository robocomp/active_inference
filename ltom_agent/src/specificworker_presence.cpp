#include "specificworker.h"

// -- GRAFCET state hooks -> the shared presence coordinator (same shape as episodic_agent / language_agent) --

void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// [Owns] = ["ltom_agent 28"]: ltom owns only its agent node (no instance prefix, no affordances), so the shared
// coordinator's [Owns] deletion is the whole cleanup (it stops the monitor first); rc::owned::Spec has nothing to sweep.
void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    presence_coordinator_.cleanup_owned_nodes();
}
