/*
 * SpecificWorker -- ltom_agent (2026-11-01). Long-term object memory. An OBSERVER: it never writes an object node,
 * never decides identity, and no concept agent requires it.
 *
 * Task 1 (skeleton): joins the live graph (domain 0), publishes its agent node "ltom_agent 28" through the shared
 * presence coordinator (nothing required), reads its [LTOM] config, computes nothing, and exits cleanly on
 * SIGTERM/SIGINT (never kill -9). Later tasks add the observe / consolidate / persist logic in compute().
 *
 * ONE graph: [Agent.dsr] (G). DDS domain 5 is reserved for the future memory graph; step 2 does not join it.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

#include "repertory.h"
#include "recall_publisher.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

private:
    // -- Presence protocol (CLAUDE.md "Presence protocol") --
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void request_shutdown();
    void terminal_shutdown();
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;

    // -- [LTOM] config (read in initialize()). The poll period is Period.Compute (one source of truth) --
    struct Params
    {
        int owner_grace_ms = 3000;
        int save_period_s = 30;
        std::string repertory_path = "etc/ltom_repertory.json";
        std::string csv_path = "etc/ltom_repertory.csv";
        std::vector<std::string> classes;
        std::vector<std::string> recall_classes;   // classes whose remembered, absent members are offered as recall nodes
    };
    Params params_;

    // -- observe / consolidate / persist (Tasks 2-4) --
    void save_now(const char* why);
    std::unique_ptr<ltom::Repertory> repertory_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;   // created ONCE in initialize(), on the main thread (owns the ts==0 cache)
    std::unique_ptr<ltom::RecallPublisher> recall_pub_;
    bool recall_swept_on_operating_ = false;            // the post-sync stale sweep runs once, on the first Operating entry
    void sweep_stale_recalls();
    bool save_blocked_ = false;                         // load could not move a bad file aside: never save over the good copy
    bool operating_ = false;                            // compute() polls only while the presence SM is Operating
    std::chrono::steady_clock::time_point last_save_{std::chrono::steady_clock::now()};

    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
