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
 * SpecificWorker — scene_graph_viewer. See specificworker.h for what this view is and why it is
 * its own agent. This TU is the only one that includes the renderer, keeping the Qt OpenGL widget
 * stack (and dsr_gui, pulled in by genericworker.h) confined.
 */

#include "specificworker.h"

#include "../../common/viewers/gl_graph3d_viewer.h"

#include <QCoreApplication>
#include <QMainWindow>
#include <QSettings>
#include <QTimer>

#include <chrono>
#include <cmath>
#include <cstdlib>    // std::_Exit — crash-free terminal shutdown
#include <iostream>   // std::cout/cerr flush
#include <print>
#include <thread>     // brief DDS flush before _Exit

#include <dsr/api/dsr_api.h>

namespace {
// QSettings group for this agent's own window. GenericWorker::save/restore_window_settings only
// covers the windows IT creates (the cortex DSRViewer's), and this agent deliberately creates none
// — Agent.graph/2d/tree are all false. So the geometry of our own window is ours to persist.
QString settings_group(int agent_id)
{
    return QStringLiteral("windows/%1/graph3d").arg(agent_id);
}
}   // namespace

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader,
                               TuplePrx tprx,
                               bool startup_check)
    : GenericWorker(configLoader, tprx),
      startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    cfg_ = rc::load_viewer_config(configLoader);

    const int period = configLoader.get<int>("Period.Compute");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this),
        std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this),
        std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this),
        std::bind(&SpecificWorker::degraded_enter, this));

    // Compute → Waiting on start
    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    // Waiting → Operating when all required peers are ready (immediately: there are none)
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    // Operating → Degraded when a required peer is lost
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    // Degraded → Waiting immediately (self-kill scheduled inside degraded_enter)
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
    std::print("scene_graph_viewer: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    // Stop the rebuild BEFORE anything else: it is this agent's only recurring work, and a timer
    // still firing into build() while the graph is being torn down is the one way a pure observer
    // can crash on exit.
    if (refresh_timer_)
        refresh_timer_->stop();
    save_window_geometry();
    viewer_ = nullptr;
    window_.reset();

    save_window_settings();
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // Crash-free terminal exit (matches the level-1 agents). After our cleanup, hard-exit instead
    // of returning into the Ice communicator teardown + C++ static destruction, which run with
    // undefined cross-TU order and abort.
    request_shutdown();
    if (G)
    {
        try { G->reset(); }   // clean DDS participant/entity removal without touching Ice
        catch (...) { /* best-effort: we are exiting regardless */ }
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("scene_graph_viewer: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "scene_graph_viewer: DSR graph not available in initialize()";
        return;
    }

    // Never mirror the legacy blob attributes locally: this agent reads poses and geometry only,
    // and the media plane carries the pixels.
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's own node in the graph view by its live health, exactly as every other
    // agent does — the viewer should not be the one node in the graph that lies about itself.
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty())
                qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating");
        },
        .on_operating_loop = [this]()
        {
            compute();
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // Debounce: a transient required-peer flap (startup handshake, brief node churn) fires
            // presenceLost momentarily and then recovers; tearing down here would kill the agent on
            // a blip. Wait a grace period and only shut down if a required peer is STILL missing.
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                {
                    qInfo("[SM] required peers recovered during grace — staying alive");
                    return;
                }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    // NO update_node/update_edge signal connects. The scene is POLLED from a main-thread timer;
    // those signals fire on the FastDDS reader threads and connecting them buys nothing here.

    // Deferred so the graph has finished syncing from the persistent server — otherwise the first
    // frame is a near-empty scene the camera auto-frames onto, and it visibly snaps a moment later.
    QTimer::singleShot(cfg_.window_delay_ms, this, [this]()
    {
        if (shutting_down_)
            return;
        build_viewer_window();
    });
}

// ─── The view ────────────────────────────────────────────────────────────────

void SpecificWorker::build_viewer_window()
{
    if (window_)
        return;   // idempotent

    graph3d_builder_.set_graph(G);
    auto& bcfg = graph3d_builder_.config();
    bcfg.stratum_gap      = cfg_.stratum_gap;
    bcfg.stale_after_ms   = cfg_.stale_after_ms;
    bcfg.show_affordances = cfg_.show_affordances;
    bcfg.show_agents      = cfg_.show_agents;
    bcfg.floor_alpha      = cfg_.floor_alpha;
    bcfg.robot_mesh_path  = cfg_.robot_mesh_path;

    viewer_ = new rc::viewers::GLGraph3DViewer();
    viewer_->set_light_background(cfg_.light_background);
    viewer_->set_pick_callback([this](std::uint64_t id, const std::string& type)
                               { log_picked_node(id, type); });

    // Own top-level window rather than a dock in the cortex DSRViewer (this agent creates none:
    // Agent.graph/2d/tree are all false). Beyond the independence that motivated the split, it
    // keeps this heavy QOpenGLWidget off the GraphViewer's shared backing store — see the warning
    // on DSRViewer::add_custom_widget_in_own_window in dsr_gui.h.
    window_ = std::make_unique<QMainWindow>();
    window_->setWindowTitle(QString("%1-%2 | stratified DSR graph")
                                .arg(QString::fromStdString(agent_name))
                                .arg(agent_id));
    window_->resize(1000, 800);
    window_->setCentralWidget(viewer_);   // reparents: the window owns the viewer from here on
    restore_window_geometry();
    window_->show();
    // Give the GL widget the keyboard focus explicitly. A QMainWindow does NOT hand focus to its
    // central widget on show, so until something is clicked the window itself swallows every key
    // press and EVERY shortcut this view has (O/L/G/T/B/R/Esc) silently does nothing — which reads
    // exactly like an unimplemented feature.
    viewer_->setFocus();

    // POLLED, not signal-driven, on purpose: SceneBuilder::build() walks the InnerEigenAPI ts==0
    // transform cache, which is unlocked and therefore main-thread-only (CLAUDE.md), and a QTimer
    // guarantees that. A full rebuild over ~50 nodes at 5 Hz is nothing.
    //
    // The timer is deliberately INDEPENDENT of the GRAFCET state: a viewer that stops redrawing
    // when presence degrades is a viewer that hides exactly the moment you need to look at it.
    refresh_timer_ = new QTimer(this);
    QObject::connect(refresh_timer_, &QTimer::timeout, this, &SpecificWorker::refresh_scene);
    refresh_timer_->start(cfg_.refresh_ms);
    refresh_scene();

    qInfo() << "[graph3d] stratified 3D graph view up at" << 1000 / std::max(1, cfg_.refresh_ms) << "Hz";
}

void SpecificWorker::refresh_scene()
{
    if (shutting_down_.load() or not viewer_)
        return;
    const auto scene = graph3d_builder_.build();
    last_node_count_ = scene.nodes.size();

    // Wall bookkeeping for the liveness line. A wall only gets its lit interior face when it is
    // placed AND the room polygon exists to say which side "interior" is, and it only shows when the
    // published panel mesh actually loaded — a glyph fallback is single-sided. Counting all three
    // separately is the difference between "no inside face" and knowing WHY.
    last_wall_count_ = 0;
    last_wall_two_sided_ = 0;
    last_wall_meshed_ = 0;
    for (const auto &n : scene.nodes)
        if (n.cls == "wall")
        {
            ++last_wall_count_;
            if (n.inside_sign != 0.0f)    ++last_wall_two_sided_;
            if (not n.mesh_path.empty())  ++last_wall_meshed_;
        }
    last_ground_valid_ = scene.ground.valid;

    viewer_->set_scene(scene);
}

void SpecificWorker::log_picked_node(std::uint64_t node_id, const std::string& type)
{
    if (not G)
        return;
    const auto node = G->get_node(node_id);
    if (not node.has_value())
    {
        qInfo() << "[pick] node" << node_id << "(type" << QString::fromStdString(type)
                << ") is no longer in the graph";
        return;
    }
    const auto subtype = G->get_attrib_by_name<object_subtype_att>(node.value());
    const auto owner   = G->get_attrib_by_name<agent_id_att>(node.value());
    qInfo() << "[pick]" << QString::fromStdString(node->name())
            << "id" << node_id
            << "type" << QString::fromStdString(node->type())
            << "subtype" << QString::fromStdString(subtype.value_or(std::string{"-"}))
            << "written by agent" << (owner.has_value() ? QString::number(owner.value()) : QStringLiteral("?"));
}

void SpecificWorker::restore_window_geometry()
{
    if (not cfg_.remember_geometry or not window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
    settings.beginGroup(settings_group(agent_id));
    const QByteArray geometry = settings.value(QStringLiteral("geometry")).toByteArray();
    if (not geometry.isEmpty())
        window_->restoreGeometry(geometry);
    settings.endGroup();
}

void SpecificWorker::save_window_geometry() const
{
    if (not cfg_.remember_geometry or not window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QString::fromStdString(agent_name));
    settings.beginGroup(settings_group(agent_id));
    settings.setValue(QStringLiteral("geometry"), window_->saveGeometry());
    settings.endGroup();
}

// ─── GRAFCET compute ─────────────────────────────────────────────────────────

void SpecificWorker::compute()
{
    // The scene rebuild runs on its own timer (see build_viewer_window), so the GRAFCET loop has
    // no per-cycle work. It stays as the liveness channel: a throttled line proving the agent is
    // stepping and reporting how big the scene it is drawing actually is — zero nodes with the
    // window up would otherwise look identical to a healthy empty room.
    ++cycle_;
    if (cfg_.log_period_frames > 0 and cycle_ % static_cast<std::uint64_t>(cfg_.log_period_frames) == 0)
        std::print("scene_graph_viewer: cycle {} — {} nodes · walls {} ({} meshed, {} two-sided) · "
                   "room polygon {}\n",
                   cycle_, last_node_count_, last_wall_count_, last_wall_meshed_,
                   last_wall_two_sided_, last_ground_valid_ ? "yes" : "NO");
}

void SpecificWorker::emergency()
{
    std::print("scene_graph_viewer: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("scene_graph_viewer: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("scene_graph_viewer: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
