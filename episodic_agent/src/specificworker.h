/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * SpecificWorker — episodic_agent (2026-10-17). An OBSERVER: it never acts, never claims, is never required.
 *
 * Every [EM] poll_ms, on the main thread (a QTimer; no DSR signal is connected):
 *   1. the `controller` agent node's em_journal → journal_from_wire → Builder::on_journal; the node vanishing after
 *      having been seen → Builder::owner_lost;
 *   2. the structural diff of the live graph (LiveDiff) → Builder::on_events;
 *   3. every record those return → Recorder::write → Archive::append + the domain-4 graph (EmGraph).
 * At startup the Recorder recomputes the all-time stats from the archive and reseeds today's records into domain 4.
 * On a graceful stop (SIGTERM/SIGINT, never kill -9) Builder::shutdown closes what is open as
 * `interrupted_by_em_shutdown`, and those records are archived.
 *
 * Graphs (etc/config.toml): G == "dsr" (live, domain 0, read), "em" (domain 4, owned), "ltsm" (domain 2, read).
 * The logic lives in common/episodic/ (pure, tested); this file is the shell.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>

#include "dsr_fact_source.h"
#include "em_graph.h"
#include "em_recorder.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/episodic/em_archive.h"
#include "../../common/episodic/em_builder.h"

#include <QTimer>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

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
    // ── Presence protocol (copied from chat_agent / the level-1 agents; CLAUDE.md "Presence protocol") ──
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
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);

    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    // ── Episodic memory ─────────────────────────────────────────────────────────────────────────────────
    struct GraphSink : rc::em::RecordSink          // Recorder → domain-4 graph
    {
        rc::em::EmGraph* g = nullptr;
        void upsert(const rc::em::Record& r) override;
        void upsert(const rc::em::Stats& s) override;
        void remove(const std::string& id) override;
    };

    std::shared_ptr<DSR::DSRGraph> G_em_, G_ltsm_;  // G is the live graph
    std::unique_ptr<rc::em::DsrFactSource> facts_;
    std::unique_ptr<rc::em::EmGraph> em_graph_;
    GraphSink sink_;
    std::unique_ptr<rc::em::Archive> archive_;
    std::unique_ptr<rc::em::Recorder> recorder_;
    std::unique_ptr<rc::em::Builder> builder_;
    rc::em::LiveDiff diff_;
    QTimer poll_timer_;

    std::string last_wire_;                         // identical wire ⇒ not re-parsed
    bool controller_seen_ = false, controller_present_ = false;
    bool bad_wire_logged_ = false, absent_logged_ = false;

    void poll();
    /// The live agent node of the `controller` (cortex names it "<agent_name> <id>"; matched without the id).
    std::optional<DSR::Node> controller_node() const;
    static std::uint64_t now_ms();

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
