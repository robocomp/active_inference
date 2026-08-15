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

#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include "voxel_opengl_viewer.h"
#include "scene_feed.h"

#include <QCoreApplication>
#include <QMainWindow>
#include <QSettings>
#include <QTimer>

#include <algorithm>  // std::max — clamps the FPSCounter window
#include <chrono>
#include <cmath>
#include <cstdlib>    // std::_Exit — crash-free terminal shutdown
#include <format>
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
    return QStringLiteral("windows/%1/viewer3d").arg(agent_id);
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

    viewer_ = new rc::VoxelOpenGLViewer();
    feed_   = std::make_unique<rc::SceneFeed>(G, viewer_);
    feed_->set_verbose(cfg_.verbose);
    if (not cfg_.robot_mesh_path.empty())
        viewer_->load_robot_mesh(cfg_.robot_mesh_path);

    // Own top-level window rather than a dock in the cortex DSRViewer (this agent creates none:
    // Agent.graph/2d/tree are all false). Beyond the independence that motivated the split, it keeps
    // this heavy QOpenGLWidget off the GraphViewer's shared backing store — see the warning on
    // DSRViewer::add_custom_widget_in_own_window in dsr_gui.h. That hazard is exactly why this widget
    // was already in its own window inside the retina.
    window_ = std::make_unique<QMainWindow>();
    window_->setWindowTitle(QString("%1-%2 | metric 3D scene")
                                .arg(QString::fromStdString(agent_name))
                                .arg(agent_id));
    window_->resize(1200, 900);
    window_->setCentralWidget(build_layer_panel());   // reparents: the window owns panel + viewer
    restore_window_geometry();
    window_->show();
    // A QMainWindow does NOT hand focus to its central widget on show, so until something is clicked
    // the window swallows every key press and every shortcut this view has silently does nothing —
    // which reads exactly like an unimplemented feature.
    viewer_->setFocus();

    // The LiDAR is the one input that does NOT come through the graph. Its descriptor is authored by
    // robot_concept on the sensor node, so this can only succeed once that node exists — retried from
    // refresh_scene() rather than failed here.
    feed_->init_lidar(cfg_.lidar_topic);

    // POLLED, not signal-driven, on purpose: the feed walks InnerEigenAPI's ts==0 transform cache,
    // which is unlocked and therefore main-thread-only (CLAUDE.md), and a QTimer guarantees that.
    // Connecting update_node_signal instead would run it on a FastDDS reader thread — the crash this
    // codebase already paid for once.
    //
    // The timer is deliberately INDEPENDENT of the GRAFCET state: a viewer that stops redrawing when
    // presence degrades is a viewer that hides exactly the moment you need to look at it.
    refresh_timer_ = new QTimer(this);
    QObject::connect(refresh_timer_, &QTimer::timeout, this, &SpecificWorker::refresh_scene);
    refresh_timer_->start(cfg_.refresh_ms);
    refresh_scene();

    qInfo() << "[viewer3d] metric 3D scene view up at" << 1000 / std::max(1, cfg_.refresh_ms) << "Hz";
}


// ★THE LAYER TOGGLES CAME ACROSS WITH THE VIEW. They drove this GL widget, so they belong to this agent,
// not to the retina they used to sit in. Same labels, same default states and the same accent colours
// as before the split — each button is tinted with the colour of the thing it shows, so the association
// between a toggle and its geometry survives the move.
QWidget* SpecificWorker::build_layer_panel()
{
    auto* panel  = new QWidget(nullptr);
    auto* layout = new QHBoxLayout(panel);   // GL view takes the width it can, side pane on the RIGHT
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(6);

    // The pane is its own widget so it can carry a fixed width: buttons stacked in a column would
    // otherwise stretch to the widest caption and shift every time a label flips ON/OFF.
    auto* pane      = new QWidget(panel);
    auto* pane_col  = new QVBoxLayout(pane);
    pane_col->setContentsMargins(0, 0, 0, 0);
    pane_col->setSpacing(4);
    pane->setFixedWidth(140);

    const auto mk = [&](const char* text, bool on, const char* hex) -> QPushButton*
    {
        auto* b = new QPushButton(QString::fromUtf8(text), pane);
        b->setCheckable(true);
        b->setChecked(on);
        b->setCursor(Qt::PointingHandCursor);
        b->setStyleSheet(QString(
            "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 4px 8px; text-align: left; }"
            "QPushButton:checked { background-color: %1; color: #101010; }").arg(hex));
        return b;
    };

    auto* lidar_btn    = mk("Lidar: OFF",    false, "#8C9EC7");   // lidar: slate blue-gray
    auto* models_btn   = mk("Models: ON",    true,  "#FFC864");   // models: table-mesh amber
    auto* masks_btn    = mk("Masks: OFF",    false, "#EBEBF2");   // mask: white
    auto* residual_btn = mk("Residual: OFF", false, "#2633CC");   // residual: dark blue
    auto* grid_btn     = mk("Grid: ON",      true,  "#CC8C0D");   // occupancy grid: amber
    auto* field_btn    = mk("Field: ON",     true,  "#D64550");   // belief field: risk red
    auto* labels_btn   = mk("Labels: ON",    true,  "#DDDDDD");   // node-name labels: light grey

    // Group 1 — raw/perception geometry: the point clouds and the fitted models drawn from them.
    pane_col->addWidget(lidar_btn);
    pane_col->addWidget(models_btn);
    pane_col->addWidget(masks_btn);
    pane_col->addWidget(residual_btn);
    pane_col->addSpacing(10);   // the only thing separating the two groups now that rows are gone
    // Group 2 — derived belief layers and the text labels.
    pane_col->addWidget(grid_btn);
    pane_col->addWidget(field_btn);
    pane_col->addWidget(labels_btn);
    pane_col->addStretch(1);    // buttons stay at the TOP of the pane, not spread down its height

    // Push each button's INITIAL state into the viewer. Without this the widget's own defaults and the
    // buttons' captions can disagree from the first frame — a toggle that reads ON over a hidden layer,
    // which is the kind of thing that gets chased as a missing-data bug.
    const auto wire = [this](QPushButton* b, void (rc::VoxelOpenGLViewer::*setter)(bool), const char* label)
    {
        (viewer_->*setter)(b->isChecked());
        connect(b, &QPushButton::toggled, this, [b, setter, label, this](bool checked)
        {
            if (viewer_)
                (viewer_->*setter)(checked);
            b->setText(QString("%1: %2").arg(QString::fromUtf8(label), checked ? "ON" : "OFF"));
        });
    };
    wire(lidar_btn,    &rc::VoxelOpenGLViewer::set_show_lidar,    "Lidar");
    wire(models_btn,   &rc::VoxelOpenGLViewer::set_show_models,   "Models");
    wire(masks_btn,    &rc::VoxelOpenGLViewer::set_show_masks,    "Masks");
    wire(residual_btn, &rc::VoxelOpenGLViewer::set_show_residual, "Residual");
    wire(grid_btn,     &rc::VoxelOpenGLViewer::set_show_grid,     "Grid");
    wire(field_btn,    &rc::VoxelOpenGLViewer::set_show_field,    "Field");
    wire(labels_btn,   &rc::VoxelOpenGLViewer::set_show_labels,   "Labels");

    layout->addWidget(viewer_, 1);   // reparents the GL widget into the panel
    layout->addWidget(pane, 0);      // toggles sit to the RIGHT of the canvas
    return panel;
}

void SpecificWorker::refresh_scene()
{
    if (shutting_down_.load() or not viewer_ or not feed_)
        return;
    if (not lidar_ready_)
        lidar_ready_ = feed_->init_lidar(cfg_.lidar_topic);   // cheap no-op once it succeeds
    feed_->refresh();
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
    // stepping, what it is drawing, and what that costs in CPU and memory — zero nodes with the
    // window up would otherwise look identical to a healthy empty room.
    ++cycle_;
    if (cfg_.log_period_frames <= 0)
        return;

    // FPSCounter owns BOTH the throttle and the sampling, so it must be called on EVERY cycle (it
    // counts them) and the line must come out of print() alone: get_cpu_use() is a delta since its own
    // last call, so reading it directly here would eat the interval and halve the next reported figure.
    // The percentage is process-wide over all threads (times(2) sums them), so >100% is a real reading
    // on this multithreaded agent — GL redraw and the scene rebuild are separate costs.
    //
    // Its window is in MILLISECONDS while our knob is in FRAMES, hence the conversion. The 1000 ms
    // floor is NOT taste: print() computes fps as `cont/(msPeriod/1000)` in INTEGER arithmetic, so any
    // window under a second divides by zero and takes the agent down with SIGFPE.
    const auto window_ms = static_cast<unsigned int>(
        std::max(1000, cfg_.log_period_frames * std::max(1, getPeriod("Compute"))));
    fps_counter_.print(std::format("[viewer3d] cycle {} · window {} · lidar plane {}",
                                   cycle_, window_ ? "up" : "not built", lidar_ready_ ? "up" : "waiting"),
                       window_ms);
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
