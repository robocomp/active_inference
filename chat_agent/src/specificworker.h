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
 * A local LLM (ollama, qwen3:8b) answers questions grounded in a digest of the live graph and turns requests into
 * commands for the controller. The model PROPOSES; C++ decides: every action goes through CommandGate, and anything
 * that moves the robot needs the person's confirmation on a card (or a typed yes, matched here, never by the model).
 * Commands reach the controller through ONE owned node, `chat_intention` (intention_node.h); the controller mirrors
 * the progress back into it (controller/src/chat_inbox.h).
 *
 * ★SAFETY (CLAUDE.md): connects NO update_node/update_edge signal. The digest and the command status are POLLED on
 * the main thread by QTimers (chat_session.h). The InnerEigenAPI is created on the main thread and used only there.
 *
 * Presence: required [] — the chat window is useful (and says so) with nothing else running; controller and
 * room_concept are OPTIONAL peers, tracked for the status bar and the gate.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <cstdint>
#include <memory>
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
