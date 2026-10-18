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
 *   1. the controller's journal node (`em_journal`, type intention — em_journal.h) → journal_from_wire →
 *      Builder::on_journal; the node vanishing after having been seen, or the presence monitor losing the
 *      `controller` peer (a crash / kill -9 leaves the node behind) → Builder::owner_lost;
 *   2. the structural diff of the live graph (LiveDiff) → Builder::on_events;
 *   3. every record those return → Recorder::write → Archive::append + the domain-4 graph (EmGraph);
 *   4. every 10 s a row of <archive_dir>/metrics_<YYYYMMDD>_<HHMMSS>.csv (em_metrics.h).
 * At startup the Recorder recomputes the all-time stats from the archive and reseeds today's records into domain 4.
 * On a graceful stop (SIGTERM/SIGINT, never kill -9) Builder::shutdown closes what is open as
 * `interrupted_by_em_shutdown`, and those records are archived.
 *
 * Graphs (etc/config.toml): G == "dsr" (live, domain 0, read), "em" (domain 4, owned).
 *
 * LTSM MEMORY (domain [EM].ltsm_domain = 2) is joined LAZILY, read-only, only while ltsm_agent is up — the scheme of
 * chat_agent's domain-4 join (chat_agent/src/specificworker.h). Not an [Agent.ltsm] sub-table on purpose: the generated
 * code joins every [Agent.<name>] graph at startup, and a DSRGraph built with configFile "" qFatal()s the whole process
 * after ~15 s when no peer answers — EM must start and run with ltsm_agent down; doors then carry no ltsm_id, and their
 * stats are keyed by action kind alone (spec §7).
 * HOW "UP" IS SEEN: ltsm_agent runs no presence coordinator, so it has no agent node to stamp (chat_agent's test). EM
 * reads cortex's DDS participant set on the live graph instead (G->get_connected_agents(), names
 * "Participant_<id> ( ltsm_agent )"), every poll. Its domain-0 participant exists ~1 s before its domain-2 graph is
 * seeded; harmless, the joiner re-sends its graph request every second for 15 s (cortex fullgraph_request_thread).
 * WHICH PROCESS: a graceful ltsm_agent exit removes its participant at once, so a restart (round_trips.sh wipes its
 * memory) is seen as a gap and the old graph is dropped before the new process's is joined.
 * Residual risks (not catchable): ltsm_agent dying within the ~1 s of the join aborts EM; a CRASHED ltsm_agent is only
 * dropped when its DDS lease lapses, and cortex keys the participant set by name, so a crash-restart inside that lease
 * can leave EM holding the old graph.
 * VIEW DATA: left-click a node in the em window → "View data" → a popup with that episode as text (episode_text.h);
 * cortex's GraphViewer emits view_data_signal for nodes with no built-in widget, as robot_concept uses it.
 * The logic lives in common/episodic/ (pure, tested); this file is the shell.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>

#include "dsr_fact_source.h"
#include "em_graph.h"
#include "em_metrics.h"
#include "em_recorder.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/episodic/em_archive.h"
#include "../../common/episodic/em_builder.h"

#include <QDialog>
#include <QPlainTextEdit>
#include <QPointer>
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

    // ── ltsm memory (domain 2), joined lazily — see the header comment ──
    /// ltsm_agent's DDS participant is on the live graph (domain 0).
    bool ltsm_connected() const;
    /// Leave if ltsm_agent's participant is gone; join when it is there and we are not joined.
    void sync_ltsm();
    void leave_ltsm();
    int ltsm_domain_ = 2;

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

    std::shared_ptr<DSR::DSRGraph> G_em_;           // G is the live graph
    std::shared_ptr<DSR::DSRGraph> G_ltsm_;         // nullptr = not joined (sync_ltsm)
    std::unique_ptr<rc::em::DsrFactSource> facts_;
    std::unique_ptr<rc::em::EmGraph> em_graph_;
    GraphSink sink_;
    std::unique_ptr<rc::em::Archive> archive_;
    std::unique_ptr<rc::em::Recorder> recorder_;
    std::unique_ptr<rc::em::Builder> builder_;
    rc::em::LiveDiff diff_;
    std::unique_ptr<rc::em::Metrics> metrics_;
    QTimer poll_timer_;

    std::string last_wire_;                         // identical wire ⇒ not re-parsed
    bool controller_seen_ = false, controller_present_ = false;
    bool bad_wire_logged_ = false, absent_logged_ = false;

    void poll();
    // ── "View data" popup on a domain-4 node (episode_text.h) ──
    void wire_view_data();                          // once, main thread; no-op without the em graph window
    void show_node_text(std::uint64_t id);          // one reusable, non-modal popup; size kept in QSettings
    QPointer<QDialog> text_popup_;
    QPointer<QPlainTextEdit> text_view_;
    /// Archive + graph + metrics for what the Builder returned.
    void write(const std::vector<rc::em::Record>& out);
    static std::uint64_t now_ms();

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
