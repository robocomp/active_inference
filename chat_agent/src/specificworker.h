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

/**
 * chat_agent — a person talks to the robot (2026-10-12, phase 1 MVP).
 *
 * A local LLM (ollama, qwen3-vl:8b-instruct; it can also look through the cameras, look()) answers questions grounded in a digest of the live graph and turns requests into
 * commands for the controller. The model PROPOSES; C++ decides: every action goes through CommandGate, and an accepted
 * one is sent at once (no confirmation card since 2026-10-18; "stop" / "para" halts the robot without the model).
 * Commands reach the controller through ONE owned node, `chat_intention` (intention_node.h); the controller mirrors
 * the progress back into it (controller/src/chat_inbox.h).
 *
 * ★SAFETY (CLAUDE.md): connects NO update_node/update_edge signal. The digest and the command status are POLLED on
 * the main thread by QTimers (chat_session.h). The InnerEigenAPI is created on the main thread and used only there.
 *
 * Presence: required [] — the chat window is useful (and says so) with nothing else running; controller and
 * room_concept are OPTIONAL peers, tracked for the status bar and the gate.
 *
 * EPISODIC MEMORY (2026-10-17): episodic_agent's graph on DDS domain [EM].domain (4) is joined LAZILY, read-only,
 * only while episodic_agent is present on the live graph (an optional peer). Not a config [Agent.em] sub-table on
 * purpose: the generated code joins every [Agent.<name>] graph at startup, and a DSRGraph built with configFile ""
 * waits for a peer to send it the graph and qFatal()s the whole process after ~15 s when none answers (cortex
 * dsr_api.cpp) — chat_agent must keep running when episodic_agent is not. episodic_agent being present on domain 0
 * means its domain-4 graph (built before its presence node) is already up to answer.
 * WHICH PROCESS: the graph belongs to one episodic_agent PROCESS, identified by its agent node's timestamp_creation.
 * It is KEPT through a stale period (heartbeat silent, node present, same stamp — the monitor reports that as
 * "optional peer lost" too): re-joining the same live process would make it answer our second graph request with
 * the "repeated participant" reply, a qFatal here. It is LEFT only when that agent node is gone from the live graph
 * or carries a new stamp (a new process; checked by sync_em on every compute tick and on each presence hook), and
 * joined again on the new process's ready/restart — so a restarted episodic_agent, which re-seeds domain 4, never
 * meets the old graph still held here (README's double-seeder note).
 * Residual risk: episodic_agent dying within the ~1.3 s of the join itself still aborts chat_agent (not catchable).
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <genericworker.h>

#include "chat_session.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

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
    rc::chat::SessionConfig load_session_config(const ConfigLoader& cfg) const;

    // ── Presence protocol (copied from the level-1 agents; see CLAUDE.md) ───────
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
    void on_peer_restarted(std::uint32_t id);

    // ── episodic memory (domain 4), joined lazily — see the header comment ──
    /// timestamp_creation of episodic_agent's agent node on the live graph; nullopt = no such node.
    std::optional<std::uint64_t> em_process_stamp() const;
    /// Leave if the process we joined is gone or replaced; with `may_join`, join the present process.
    void sync_em(bool may_join);
    void leave_em();
    int em_domain_ = 4;
    std::uint32_t em_peer_id_ = 0;             ///< episodic_agent's agent id once seen (0 = never)
    std::shared_ptr<DSR::DSRGraph> em_G_;      ///< nullptr = not joined
    std::uint64_t em_joined_stamp_ = 0;        ///< em_process_stamp() of the process em_G_ was joined from

    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    bool operating_sweep_done_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::chat::SessionConfig session_cfg_;
    std::unique_ptr<rc::chat::ChatSession> session_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
