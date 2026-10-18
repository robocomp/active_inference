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

/** SpecificWorker — chat_agent. See specificworker.h. */

#include "specificworker.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <QCoreApplication>
#include <QTimer>
#include <chrono>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <iostream>
#include <print>
#include <thread>

#include <dsr/api/dsr_api.h>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx), startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    session_cfg_ = load_session_config(configLoader);

    int period = 0;
    rc::cfg::Reader(configLoader, "chat_agent").req("Period.Compute", period,
            "GRAFCET step period (ms) for every state of this agent's state machine");
    em_domain_ = rc::cfg::Reader(configLoader, "chat_agent").i("EM.domain", 4,
            "DDS domain of episodic_agent's memory graph; joined read-only while episodic_agent is present");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this), std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this), std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this), std::bind(&SpecificWorker::degraded_enter, this));

    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

    statemachine.addState(states["Waiting"].get());
    statemachine.addState(states["Operating"].get());
    statemachine.addState(states["Degraded"].get());

    statemachine.setChildMode(QState::ExclusiveStates);
    statemachine.start();

    const auto err = statemachine.errorString();
    if (err.length() > 0)
    {
        qWarning() << err;
        throw err;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("chat_agent: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    save_window_settings();
    session_.reset();          // cancels the request in flight, closes the log, destroys the window
    em_G_.reset();             // leave domain 4 (nothing of ours lives there)
    cleanup_owned_nodes();     // chat_intention + our agent node, through the shared coordinator ([Owns])
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    // Crash-free terminal exit (matches the fleet): cleanup, clean DDS teardown, then _Exit instead of returning
    // into the Ice communicator teardown + static destruction, whose cross-TU order aborts.
    request_shutdown();
    if (G)
    {
        try { G->reset(); }
        catch (...) {}
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

rc::chat::SessionConfig SpecificWorker::load_session_config(const ConfigLoader& cfg) const
{
    rc::cfg::Reader r(cfg, "chat_agent");
    rc::chat::SessionConfig c;
    c.llm.url = QString::fromStdString(r.s("LLM.url", "http://localhost:11434", "ollama server base URL"));
    c.llm.model = QString::fromStdString(r.s("LLM.model", "qwen3:8b", "ollama model tag (must be pulled)"));
    c.llm.num_ctx = r.i("LLM.num_ctx", 8192, "context window passed to ollama (options.num_ctx)");
    c.llm.temperature = r.d("LLM.temperature", 0.2, "sampling temperature (low: grounded, repeatable answers)");
    c.llm.first_token_timeout_ms = r.i("LLM.first_token_timeout_ms", 60000,
            "abort a request with no token/tool call after this long (cold model load is the usual cause)");
    c.llm.transfer_timeout_ms = r.i("LLM.transfer_timeout_ms", 120000, "transport gives up after this long with no bytes");
    c.llm.keep_alive = QString::fromStdString(r.s("LLM.keep_alive", "30m", "how long ollama keeps the model loaded"));
    c.llm.num_predict = r.i("LLM.num_predict", -1,
            "cap on generated tokens per request (options.num_predict); <= 0 = no cap (a runaway answer is cut, not stopped)");
    c.max_history = r.i("Chat.max_history", 16, "messages of history sent with each request (tool pairs kept together)");
    c.digest_period_ms = r.i("Chat.digest_period_ms", 500, "period of the graph STATE digest (main thread poll)");
    c.status_period_ms = r.i("Chat.status_period_ms", 250, "period of the chat_intention state poll");
    c.log_dir = r.s("Chat.log_dir", "logs", "directory for logs/chat_<date>.jsonl");
    c.system_prompt_path = r.s("Chat.system_prompt_path", "etc/system_prompt.md",
            "robot description prepended to the system message (read once at start; missing => built-in)");
    c.limits.max_move_m = r.d("Chat.max_move_m", 3.0, "move_relative: longer distances are refused (controller cap 3 m)");
    c.default_step_m = r.d("Chat.default_step_m", 0.3, "\"advance a bit\": the step the model is told to use");
    c.robot_radius_m = r.d("Chat.robot_radius_m", 0.35, "approach_object: half the body width");
    c.approach_clearance_m = r.d("Chat.approach_clearance_m", 0.30,
            "approach_object: free space kept between the body and the object (room to turn)");
    c.objects_nearest = r.i("Chat.objects_nearest", 6, "OBJECTS section of the STATE: how many nearest objects are listed");
    c.object_change_period_ms = r.i("Chat.object_change_period_ms", 2000,
            "object births/removals are shown in the window at most this often (flaps inside the window cancel)");
    c.look_max_side = r.i("Chat.look_max_side", 1280,
            "look(): the front (ZED) frame is shrunk so its longest side is at most this (px) before JPEG "
            "(ollama resizes to ~1 Mpx itself: ~1050 tokens at <=1280 px, ~1800 at 1920)");
    c.look_max_side_360 = r.i("Chat.look_max_side_360", 1280,
            "look(): the same for the 2:1 Ricoh panorama (1280x640 ~1070 tokens; 1920x960 ~1800 and saw no more)");
    c.look_jpeg_quality = r.i("Chat.look_jpeg_quality", 80, "look(): JPEG quality 1..100 of the attached frame");
    return c;
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("chat_agent: initialize()\n");
    GenericWorker::initialize();
    if (not G)
    {
        qWarning() << "chat_agent: DSR graph not available in initialize()";
        return;
    }
    // Heavy payloads are never read here.
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [this](std::uint32_t id) { on_peer_restarted(id); },
        .on_optional_peer_lost  = [this](const std::string& n, std::uint32_t i) { on_optional_peer_lost(n, i); },
        .on_optional_peer_ready = [this](const std::string& n, std::uint32_t i) { on_optional_peer_ready(n, i); },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty()) qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing) m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating");
            // ★Second stale-sweep, once: the initialize() pass can run before a crashed run's leftovers have
            // synced from the persistent server (CLAUDE.md).
            if (not operating_sweep_done_ and session_)
            {
                operating_sweep_done_ = true;
                session_->sweep_stale();
            }
        },
        .on_operating_loop  = [this]() { compute(); },
        .on_degraded_enter  = [this]()
        {
            if (shutting_down_) return;
            // Debounce (CLAUDE.md): with required = [] this should never fire, but if [Presence] ever gains a
            // required peer, a transient flap must not tear the agent down.
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_) return;
                if (presence_coordinator_.all_required_ready())
                { qInfo("[SM] required peers recovered during grace — staying alive"); return; }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    // NO update_node/update_edge signal connects (CLAUDE.md): the session polls on the main thread.
    session_ = std::make_unique<rc::chat::ChatSession>(G, session_cfg_);
    session_->set_em_graph(em_G_);   // nullptr until episodic_agent is seen (join_em)
    session_->sweep_stale();   // a crashed run's chat_intention, before ours is born
    session_->start();

    // Every config key this agent reads goes through rc::cfg::Reader (check_registry_complete.sh chat_agent).
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("chat_agent");
    rc::cfg::Reader(configLoader, "chat_agent").publish("etc/config_effective.csv");
}

void SpecificWorker::compute()
{
    // Everything runs on the session's QTimers (main thread); the GRAFCET loop only keeps the presence protocol —
    // and notices episodic_agent's node vanishing or being replaced (no presence hook fires after a stale period).
    sync_em(false);
}

void SpecificWorker::emergency() { std::print("chat_agent: emergency()\n"); }
void SpecificWorker::restore()   { std::print("chat_agent: restore()\n"); }
int SpecificWorker::startup_check()
{
    std::print("chat_agent: startup_check()\n");
    return 0;
}
