/*
 *    Copyright (C) 2026 by YOUR NAME HERE
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
#include "specificworker.h"

#include "mount_calibrator.h"
#include "pose_publisher.h"
#include "calib_channels.h"
#include "ground_truth_log.h"
#include <limits>
#include "../../common/robot_capability/robot_capability.h"

#include "image_edge_ops.h"   // xyz_from_pixel_depth(): model-aware, unlike cortex's pinhole-only version
#include <filesystem>
#include <locale>

#include <algorithm>
#include <print>
#include <sys/resource.h>   // getrusage — process CPU% readout
#include <cstdio>   // std::remove — the Reset must delete the saved evidence
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <random>
#include <stdexcept>
#include <QElapsedTimer>
#include <QCoreApplication>
#include <fstream>
#include <unordered_set>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QMetaObject>
#include <QFileInfo>
#include <QThread>
#include <QVBoxLayout>

#include <variant>

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {
#ifdef HIBERNATION_ENABLED
        hibernationChecker.start(500);
#endif
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

        states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
        states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
        states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
        states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

        statemachine.addState(states["Waiting"].get());
        statemachine.addState(states["Operating"].get());
        statemachine.addState(states["Degraded"].get());

        statemachine.setChildMode(QState::ExclusiveStates);
        statemachine.start();
        auto error = statemachine.errorString();
        if (error.length() > 0) { qWarning() << error; throw error; }
    }
}

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::~SpecificWorker()
{
    request_shutdown();
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    if (viewer_)
        viewer_->persist_window_state();   // _Exit below skips ~RoomViewer, so save its state now
    save_robot_pose_once();

    // Drop the lidar media subscriber BEFORE tearing down RoomConcept (pump() calls
    // room_concept_.notify_new_lidar) and while G is still alive.
    if (calib_) calib_->stop();   // drop the RGB readers before the graph goes
    imu_ingestor_.reset();   // stop the IMU reader before the graph goes
    lidar_ingestor_.reset();

    room_concept_.stop();
    cleanup_owned_nodes();

    // Cleanly remove THIS agent's DDS participant + entities so peers free agent id 5 IMMEDIATELY.
    // A bare _Exit skips ~DSRGraph (which does exactly this), so the dead id-5 participant lingers in
    // live peers' discovery until the FastDDS liveliness lease expires — a quick reopen then hits
    // "There is already an agent connected with the id: 5". DSRGraph::reset() runs
    // remove_participant_and_entities() directly, WITHOUT touching the Ice communicator, so it doesn't
    // trip the static-destruction abort below. Mirrors bottle_concept's known-good shutdown. This must
    // run even if other shared_ptr copies of G survive — reset() is an explicit method, not refcount-
    // driven, so we don't need to chase down every holder (scene_graph_/camera_viz_/viewer_2d_).
    if (G)
    {
        try { G->reset(); }
        catch (...) { /* best-effort: exiting regardless */ }
    }

    // Crash-free terminal exit. After our cleanup (state saved, self agent node deleted, participant
    // removed, peers notified) we hard-exit instead of returning into Ice::Application's communicator
    // teardown + C++ static destruction. Those run with UNDEFINED cross-TU order: a global/DDS holder
    // copies a graph Node (e.g. type "mind") AFTER the node-type registry static is destroyed, so
    // Node::type() throws "<type> is not a valid node type" -> std::terminate/abort on every exit.
    // _Exit skips all of that; the OS reclaims memory/sockets/threads. Only reached on a real shutdown
    // (shutting_down_ latched above). Brief pause lets the removal deltas + participant departure reach
    // peers first.
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::_Exit(EXIT_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::initialize()
{
    GenericWorker::initialize();

    // ── STARTUP PHASES: TIMED, AND THE WINDOW STAYS ALIVE THROUGH THEM ───────────────────────────
    // initialize() runs on the GUI THREAD, so every millisecond spent in it is a millisecond the
    // window cannot paint, resize or respond — the agent looks hung from the moment it appears until
    // the first estimate lands. Two things here, and they are deliberately together:
    //   · each phase's cost is appended to tmp/startup_timing.csv, so "which part is slow" stops
    //     being a guess. Nothing in this agent measured startup at all; compute_timing.csv only
    //     begins once compute() is already ticking.
    //   · processEvents() is pumped at each boundary, so the window paints and stays responsive
    //     across the slow phases instead of after them. Safe at these points: the DSR update signals
    //     are not connected until later in this function, so no graph slot can re-enter here, and the
    //     compute timer has not started.
    QElapsedTimer init_timer, phase_timer;
    init_timer.start();
    phase_timer.start();
    {   // truncate once, then never hold the stream open — see the phase() lambda
        std::ofstream reset("tmp/startup_timing.csv", std::ios::out | std::ios::trunc);
        if (reset.is_open()) { reset.imbue(std::locale::classic()); reset << "phase,ms,cumulative_ms\n"; }
    }
    const auto phase = [&](const char* name)
    {
        const auto ms = phase_timer.restart();
        // ⚠ OPEN-APPEND-CLOSE per row, because RoomViewer's constructor appends its own sub-phases to
        // this same file while this function is still running. Holding a truncating stream open across
        // that made TWO writers with independent file offsets: this one's next write landed at its own
        // offset and overwrote the viewer's rows — which silently destroyed the two rows that mattered
        // (viewer:custom_widget and viewer:viewer_2d, the latter being the 16-second one) and left a
        // half-overwritten line in the file. An instrument that erases its own most important reading
        // is worse than no instrument.
        std::ofstream f("tmp/startup_timing.csv", std::ios::out | std::ios::app);
        if (f.is_open())
        {
            f.imbue(std::locale::classic());
            f << name << ',' << ms << ',' << init_timer.elapsed() << '\n';
        }
        if (ms > 200)
            qInfo().noquote() << QString("[startup] %1 took %2 ms (cumulative %3 ms) — the window is "
                                         "blocked for the duration of any phase on this thread")
                                     .arg(name).arg(ms).arg(init_timer.elapsed());
        QCoreApplication::processEvents();
    };
    phase("generic_worker");

    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    // ── Load all config (agent + RoomConcept + EpistemicController params) ──
    rc::load_room_config(configLoader, params, room_concept_, epistemic_controller_);
    // The mount calibrator borrows the viewer SLOT, not the viewer: it is constructed here, long
    // before RoomViewer exists, and reads through the pointer whenever it needs to display.
    mount_ = std::make_unique<rc::MountCalibrator>(G, params, &viewer_raw_slot_);
    pose_pub_ = std::make_unique<rc::PosePublisher>(G, rt_api_.get(), params, room_concept_,
                                                    &viewer_raw_slot_, shutting_down_);
    pose_pub_->set_on_corrected_published([this](const rc::RoomConcept::UpdateResult& res) { gt_log_->log_ground_truth(res); });
    calib_ = std::make_unique<rc::CalibChannels>(G, params, room_concept_, *mount_, &viewer_raw_slot_);
    gt_log_ = std::make_unique<rc::GroundTruthLog>(G, room_concept_, shutting_down_);
    phase("load_config");

    // ── Collaborators (constructor injection; worker owns rt_api + shared params) ──
    rt_api_ = G->get_rt_api();
    // 2026-09-09: widen the RT timestamped history from the library default (5 blocks) so consumers
    // querying a real acquisition timestamp (camera/lidar, PTP-anchored) more than ~50ms old do not
    // silently clamp to the newest/oldest retained sample (RT_API::bracketing_blocks never
    // extrapolates -- see dsr_rt_api.cpp). publish_predicted_tick writes this SAME edge at the IMU
    // rate (~100 Hz measured, [ImuInject] coverage log), so N slots cover roughly N*10ms of history.
    // 25 slots -> ~250ms, chosen as a settled production value (was bumped to 150 = ~1.5s only for
    // the one-off ricoh_omni_dds time_offset test, since reverted -- see debug_pose_sync_sensores.md).
    // Process-local: rt_api_ is this worker's own instance (DSRGraph::get_rt_api() is a factory,
    // never shared -- see dsr_api.h), and this agent is the sole writer of the robot<->room RT edge,
    // so this is the only place that needs it.
    rt_api_->HISTORY_SIZE = 25;
    scene_graph_ = std::make_unique<rc::RoomSceneGraph>(
        G, rt_api_.get(), params, room_concept_, epistemic_controller_,
        [this] { trigger_graph_layout_twopi(); });
    lidar_ingestor_ = std::make_unique<rc::LidarIngestor>(G, room_concept_, params);
    phase("ingestors");
    // No source switch any more: the producer no longer writes the imu_* attributes, so a "dsr"
    // option would select a path with nothing on it — a dead config of exactly the kind this
    // codebase keeps rediscovering. The escape hatch is git, not a flag that cannot work.
    imu_ingestor_ = std::make_unique<rc::ImuIngestor>(G, imu_buffer_, sim_clock_);
    // FIX 2026-09-04: publish a predicted pose on every IMU sample -- see publish_predicted_tick().
    // Runs on the imu-ingest thread; the callback itself takes publish_mutex_ before touching anything
    // shared with maybe_publish_corrected_pose().
    imu_ingestor_->set_on_new_sample([this](std::int64_t imu_ts_ms) { pose_pub_->publish_predicted_tick(imu_ts_ms); });
    // The RGB channels configure themselves — the 56 lines that stood here are CalibChannels::configure().
    calib_->configure();

    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.high_lidar_buffer = &lidar_ingestor_->buffer();
    // The ceiling the LiDAR measured travels with the model, not through a second wire: the scene
    // graph publishes it on the room node and the image-edge module projects the contour at it.
    // ⚠ NOT HERE ANY MORE. This ran in initialize(), reading an atomic that starts at 0 before the
    //   LiDAR thread has ever produced a scan, so RoomConcept held 0 for the life of the process and
    //   every consumer took its `> 1.5f` fallback: the room node's room_height was never published
    //   from the measurement, and the camera's wall-ceiling corners sat at the STATED 3.0 m for ever.
    //   Measured 2026-09-12: pz read exactly 3.0000 on all 16607 ceiling rows, and the camera's own
    //   floor-vs-ceiling disagreement put the real ceiling ~14 cm lower. The copy now happens per
    //   cycle in compute(), where the measurement can actually exist.
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    run_ctx.imu_buffer      = &imu_buffer_;
    run_ctx.sim_clock       = &sim_clock_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;

    // The scenario overlay CHOOSES the layout file, so it has to be resolved before the SVG is
    // read. It used to be applied at the end of check_init_graph_is_valid() (below), which is
    // after this line — the axis was inert and the log still said "applied".
    scene_graph_->resolve_overlays_from_graph();
    if (not scene_graph_->overlays_resolved() and not params.scenario_overlays.empty())
    {
        // The graph could not name the robot yet, so the scenario is unknown — and the very next
        // line would read whatever layout the shared section happens to hold. Stopping is the
        // only honest option: a wrong floor plan reads downstream as a localiser fault.
        qCritical() << "[room] REFUSING TO START: no type-\"robot\" node in the graph when the"
                    << "layout must be chosen. Start robot_concept first.";
        std::exit(EXIT_FAILURE);
    }
    // The other half of the overlays: values whose home is the localiser or the planner, not the
    // config struct. Same sections, same log line — split only because RoomConfig does not own those
    // objects. ★ This is where CmdNoise* finally lands: the parse existed, the application did not.
    {
        QStringList l;
        for (const auto& c : params.apply_platform_to(scene_graph_->overlay_robot_name(),
                                                      room_concept_, epistemic_controller_))
            l << QString::fromStdString(c);
        for (const auto& c : params.apply_scenario_to(scene_graph_->overlay_scenario_name(),
                                                      epistemic_controller_))
            l << QString::fromStdString(c);
        if (not l.isEmpty())
            qInfo() << "[room] overlay (localiser/planner params) applied:" << l.join(", ");
    }
    // The compute period is fixed into the GRAFCET steps at construction, before the graph can name
    // the robot — so a per-robot Period is applied here instead, by re-setting the steps.
    if (const auto it = params.platform_overlays.find(scene_graph_->overlay_robot_name());
        it != params.platform_overlays.end() and it->second.period_compute.has_value())
    {
        const int per = *it->second.period_compute;
        for (const auto& name : {"Waiting", "Operating", "Degraded", "Emergency", "Initialize", "Compute"})
            if (const auto st = states.find(name); st != states.end() and st->second)
                st->second->setPeriod(per);
        qInfo() << "[room] platform overlay: compute period set to" << per << "ms";
    }

    if (room_concept_.estimating())
    {
        // RoomShape.MapMode = estimate: no layout is loaded. The room is learnt from the LiDAR and
        // published once its polygon closes; the viewer draws the walls as they are born.
        room_concept_.configure_room_estimate();
        calib_->set_room_polygon({}, Eigen::Vector2f::Zero());
        room_initialized_from_svg_polygon_ = false;
        qInfo() << "[room] RoomShape.MapMode = estimate: no layout loaded; the room will be learnt from the LiDAR"
                << "(first pose = origin until the polygon closes and is re-anchored).";
    }
    else
        initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    phase("room_model");
    room_concept_.set_seed_pose_file(pose_path);
    phase("seed_pose");

    // The DSR graph viewer is OPTIONAL now: the layout GUI lives in its own top-level window
    // (see RoomViewer), so the agent runs with Agent.graph=false (no DSRViewer created). When a
    // viewer flag IS enabled we still use it for the graph relayout; every access is null-guarded.
    if (!find_graph_viewer(""))
        qInfo() << "[room] No DSR viewer (Agent.graph=false); layout window runs standalone.";

    // Room polygon for visualizations (viewer outline + camera-projection overlay). Reuse the one
    // the localizer got — re-loading the SVG here would skip the recentring and draw the outline in
    // the un-shifted frame.
    const std::vector<Eigen::Vector2f>& room_polygon_for_viz = calib_->room_polygon();

    // GUI / visualization (2-D viewer, FE plot, camera-projection window + RGB media plane).
    viewer_ = std::make_unique<rc::RoomViewer>(
        G, params, room_polygon_for_viz,
        room_initialized_from_svg_polygon_, room_concept_, epistemic_controller_);

    // AFTER the viewer exists. Registering this beside the ingestor setup (where the camera calib
    // is otherwise configured) put it before make_unique, so the `if (viewer_)` was false and the
    // handler was never installed — a Reset that silently cleared only half the evidence.
    phase("viewer_window");   // the window EXISTS from here; everything after this is visible hang time
    viewer_raw_slot_ = viewer_.get();   // collaborators built before the viewer read through this
    viewer_->set_camera_reset_handler([this]
    {
        mount_->reset_evidence();
        qInfo() << "[camcal] evidence cleared and etc/camera_calib.txt deleted";
    });

    if (auto* w = viewer_->widget())
    {
        connect(w->btn_camera_viz, &QPushButton::clicked, this, [this] { viewer_->show_camera(); });
        connect(w->btn_ricoh_viz,  &QPushButton::clicked, this, [this] { viewer_->show_ricoh(); });
        connect(w->btn_calib_viz, &QPushButton::clicked, this, [this] { viewer_->show_calibration(); });
        connect(w->btn_lidar_points_viz, &QPushButton::toggled, this, [this](bool on) { viewer_->toggle_lidar_points(on); });
        if (auto* v = viewer_->viewer())
            v->set_lidar_points_visible(w->btn_lidar_points_viz->isChecked());
    }

    // ── DSR scene-graph writer: resolve graph ids + body dims ──────────────
    scene_graph_->check_init_graph_is_valid();

    // Ensure a clean startup: if a stale room node exists from previous runs,
    // remove it so the room is recreated only after localization is stable.
    // In PreserveBootstrapRoom (static-room) mode, skip this: the room/table are a
    // pre-seeded static prior in the bootstrap graph and must NOT be deleted.
    if (params.PRESERVE_BOOTSTRAP_ROOM)
        qInfo() << "[room] PreserveBootstrapRoom=true: skipping start cleanup; adopting pre-seeded room/table as a static prior.";
    else
        scene_graph_->cleanup_room_graph_nodes();

    // ── Connect DSR signals ────────────────────────────────────────────────
    // NEVER Qt::DirectConnection on DSR update signals: it runs the slot on the raw
    // FastDDS reader thread and corrupts the heap under peer churn (smashed AgentInfo
    // heartbeat). modify_node_slot is empty now (LiDAR is media-only), so we simply do
    // NOT connect update_node_signal. The attrs slot below uses the default (Queued)
    // connection — it runs on the main thread and only reads velocity/odometry attrs.
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    // Publish corrections the INSTANT the localizer produces them, on the LOCALIZER thread itself.
    // FIX 2026-09-03: was marshalled to the main thread via Qt::QueuedConnection under the belief
    // that DSR writes must happen there. DSR's own API (dsr_api.h) guards the graph with an internal
    // std::shared_mutex and is written to be called from any thread/process concurrently -- the
    // marshal was a local caution, not a real requirement. Removed here to stop this publish waiting
    // behind whatever the main thread's Qt event loop is doing that tick (viewer redraw, image-edge
    // pumping, camera projection) before external consumers (controller, retina, ...) can see it.
    // compute()'s own redundant call to maybe_publish_corrected_pose() is REMOVED below (see FIX
    // 2026-09-03 there) -- keeping both would race on this callback's now-direct thread against
    // compute()'s main-thread one, hitting the same non-atomic dedup state and the same ofstream
    // members inside dsr_update_calibration/dsr_update_affordance without a lock.
    room_concept_.set_on_result_ready([this]() { pose_pub_->maybe_publish_corrected_pose(); });

    // LiDAR is pumped synchronously from compute() (no ingest thread); just start the localizer.
    room_concept_.start();
    phase("localiser_thread");

    // ── Presence coordinator ────────────────────────────────────────────────
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    AgentPresenceCoordinator::Policy presence_policy;
    presence_policy.set_local_ready_false_on_waiting_enter = false;
    presence_policy.set_local_ready_true_on_operating_enter = false;
    presence_policy.set_local_ready_false_on_degraded_enter = false;
    presence_coordinator_.set_policy(presence_policy);
    presence_coordinator_.set_transition_hooks({
        // Required peers being up is necessary but NOT sufficient: without the LiDAR media plane the
        // localizer has no evidence and can never stabilize, so advancing to Operating would just look
        // like a silent hang. Hold in Waiting until the stream is advertised; on_waiting_loop re-checks
        // every tick and advances the moment it appears (same shape as the retina's room gate).
        // Declines SILENTLY when the stream is missing — this fires on every presence event, so
        // logging here would spam. on_waiting_loop owns the (throttled) "why are we still waiting" line.
        .request_presence_ready = [this]()
        {
            if (lidar_stream_ready())
                emit presenceReady();
        },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            qInfo() << "[SM] -> Waiting";
            QTimer::singleShot(0, this, [this]() { presence_coordinator_.set_local_ready(false); });
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "  missing:" << m;
            }
        },
        .on_waiting_loop = [this]()
        {
            if (shutting_down_)
                return;
            const bool peers_ready = presence_coordinator_.all_required_ready();
            std::string why;
            const bool lidar_ready = lidar_stream_ready(&why);
            if (peers_ready and lidar_ready)
            {
                std::println("[SM] Waiting: peers ready and LiDAR stream '{}' advertised -> Operating", why);
                emit presenceReady();
                return;
            }
            // Say WHY we are stuck, on a throttle. This is the line that was missing: a room agent
            // with no lidar producer used to sit in Waiting printing nothing at all.
            const auto now = QDateTime::currentMSecsSinceEpoch();
            if (params.LIDAR_WAIT_LOG_PERIOD_MS > 0 and
                now - last_wait_log_ms_ >= params.LIDAR_WAIT_LOG_PERIOD_MS)
            {
                last_wait_log_ms_ = now;
                std::string missing;
                for (const auto& label : presence_coordinator_.missing_required_names())
                    missing += " " + label;
                std::println("[SM] Waiting — peers{}{} | lidar: {}",
                             peers_ready ? " OK" : " MISSING:",
                             peers_ready ? std::string{} : missing,
                             lidar_ready ? ("OK (" + why + ")") : why);
            }
        },
        .on_operating_enter = [this]()
        {
            operating_since_ms_   = QDateTime::currentMSecsSinceEpoch();
            lidar_stall_reported_ = false;
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            QTimer::singleShot(0, this, [this]() { presence_coordinator_.set_local_ready(true); });
            if (!room_concept_.is_running())
            {
                qWarning() << "[SM] Operating enter: RoomConcept thread was not running, starting it";
                room_concept_.start();
            }
            // Start the dedicated LiDAR ingest thread ONLY now (post graph-join): it reads the DSR graph
            // (subscriber discovery + inner_eigen transform), which is unsafe during the join. Idempotent.
            if (lidar_ingestor_)
                lidar_ingestor_->start();
                if (imu_ingestor_) imu_ingestor_->start();
                // Same rule as the LiDAR thread: only post graph-join, because bring-up reads the graph.
                if (calib_) calib_->start();
        },
        .on_operating_loop = [this]()
        {
            const auto run_operating_tick = [this]()
            {
                operating_compute_queued_.store(false, std::memory_order_release);
                // Stream-stall guard: acting on a dead LiDAR means integrating stale evidence, which
                // CLAUDE.md forbids. Drop back to Waiting (via Degraded, whose only transition is
                // ->Waiting) so the gate above can re-admit us when the producer returns.
                if (std::int64_t age = 0; not lidar_stall_reported_ and lidar_stream_stalled(&age))
                {
                    lidar_stall_reported_ = true;
                    degraded_from_lidar_  = true;
                    std::println("[SM] Operating -> Waiting: LiDAR stream STALLED ({}) — "
                                 "not localizing on stale evidence",
                                 age < 0 ? std::string("no sweep ever arrived")
                                         : std::format("last sweep {} ms ago", age));
                    emit presenceLost();
                    return;
                }
                compute();
                if (auto v = find_graph_viewer(""); v)
                    v->set_external_fps(states.at("Operating")->getActualFps());
            };

            if (QThread::currentThread() == this->thread())
            {
                run_operating_tick();
                return;
            }

            if (!operating_compute_queued_.exchange(true, std::memory_order_acq_rel))
                QMetaObject::invokeMethod(this, run_operating_tick, Qt::QueuedConnection);
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // DEBOUNCE — do NOT tear down on entry. A transient required-peer flap (startup
            // handshake, brief DSR node churn, a peer restarting) momentarily fires presenceLost and
            // then recovers; the old code deleted room's OWN agent node + stopped RoomConcept here,
            // then "recovered" to Operating in a half-broken state and exited anyway. Wait a grace
            // period and shut down only if a required peer is STILL genuinely missing.
            constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
            // A LiDAR stall also routes through here (Degraded is the only way back to Waiting), but it
            // is NOT a peer loss — say so, otherwise the log blames the wrong thing. The grace timer
            // below then finds all peers present and correctly declines to shut down.
            if (degraded_from_lidar_)
            {
                degraded_from_lidar_ = false;
                qInfo() << "[SM] -> Degraded (LiDAR stall, peers intact) — passing through to Waiting";
            }
            else
                qInfo() << "[SM] -> Degraded: required peer lost —" << REQUIRED_LOSS_GRACE_MS << "ms grace before shutdown";
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                {
                    qInfo() << "[SM] required peers recovered during grace — staying alive";
                    return;
                }
                qWarning() << "[SM] required peer still missing after grace — shutting down cleanly";
                request_shutdown();   // does cleanup + crash-free _Exit (terminal)
            });
        },
    });
    presence_coordinator_.start();

    // ── Wire mouse-driven pose reset ───────────────────────────────────────
    if (auto* v = viewer_->viewer())
    {
        connect(v, &rc::Viewer2D::robot_moved,  this, [this](QPointF p){ viewer_->on_robot_moved(p); });
        connect(v, &rc::Viewer2D::robot_rotate, this, [this](QPointF p){ viewer_->on_robot_rotated(p); });
    }


    restore_window_settings();
}

///////////////////////////////////////////////////////////////////////////////
// Compute the RT publish rate over a ~1 s window and show it in the custom widget (corrected +
// predicted blocks/s). Counters accumulate every compute tick; the readout refreshes once per second.
// The widget write is done only on the GUI thread (Qt requirement); counters still reset so the rate
// stays a true 1 s average.
// The pose publishing that lived here — corrected at LiDAR rate, predicted at IMU rate, the kinematic
// clamp, the jump log and the shared trace, 521 lines across two threads — is now rc::PosePublisher
// (src/pose_publisher.{h,cpp}).

// The ground-truth grading that lived here — including the self-checking heading-convention test — is
// now rc::GroundTruthLog (src/ground_truth_log.{h,cpp}).

void SpecificWorker::compute()
{
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    QElapsedTimer compute_timer;
    compute_timer.start();
    // ── The ceiling the LiDAR measured, into the model, EVERY CYCLE ──────────────────────────────
    // The producer is a LiDAR-thread atomic that only becomes valid once the ceiling check has seen
    // a scan, so a one-shot copy at start-up can only ever move a zero. From here the existing
    // machinery does the rest: room_scene_graph publishes it on the room node (debounced, so an
    // attribute is not rewritten every frame) and pump_image_edges projects the wall-ceiling contour
    // at it instead of at the stated constant.
    if (lidar_ingestor_)
    {
        room_concept_.set_measured_ceiling(
            lidar_ingestor_->measured_ceiling_z_.load(std::memory_order_relaxed));
        // The width travels with the value. A consumer that may REFINE this number needs to know how
        // much it is allowed to move, and one that only draws it can ignore the second number.
        room_concept_.set_measured_ceiling_sigma(
            lidar_ingestor_->measured_ceiling_sigma_.load(std::memory_order_relaxed));
    }
    auto init_time = std::chrono::steady_clock::now();
    // MICROSECONDS, not milliseconds. These were qint64 *_ms read off QElapsedTimer::elapsed(),
    // which is integer ms — and every stage here is sub-millisecond, so every section column in
    // etc/compute_timing.csv had been exactly 0 for the life of the file. The CSV could report that
    // compute() cost ~3 ms but never which stage, which is precisely the split any decoupling work
    // needs to judge itself on. nsecsElapsed() costs the same and resolves it.
    qint64 t_affordance_us = 0;
    qint64 t_loc_fetch_us = 0;
    qint64 t_viewer_us = 0;
    qint64 t_dsr_us = 0;
    qint64 t_ui_us = 0;
    qint64 t_health_us = 0;
    bool   did_publish = false;   // a corrected RT block was published this tick (for compute_timing.csv)

    if (last_affordance_monitor_ms_ == 0 || now_ms - last_affordance_monitor_ms_ >= 200)
    {
        QElapsedTimer section_timer;
        section_timer.start();
        scene_graph_->monitor_affordance();
        t_affordance_us = section_timer.nsecsElapsed() / 1000;
        last_affordance_monitor_ms_ = now_ms;
    }

    // LiDAR is drained by lidar_ingestor_'s dedicated ingest thread (started at Operating-enter), which
    // pumps a fresh scan to the localizer with ~0-2 ms latency instead of this ~16 ms tick. compute()
    // only reads the resulting buffer/result below.

    // ── WHERE THE CPU GOES WHEN NOTHING IS BEING OPTIMISED ──────────────────────────────────────
    // At 100% early exit the localiser update is ~1.6 ms at 20 Hz — 3% of a core — so the rest of
    // the CPU is not the thing being watched. Both pumps below run EVERY compute tick regardless of
    // whether the optimiser will: the driving camera's extraction, and then the SAME extraction
    // again for each entry in ImageEdge.calibCameras (zed + ricoh today, so three per tick).
    // Their only live consumers while the gate holds are the calibration accumulator and the canvas
    // — the loss sees them just on optimised frames. Reported as ms per second, i.e. percent of one
    // core, so the answer is a number rather than an argument about which pump is heavy.
    {
        QElapsedTimer t; t.start();
        calib_->pump_image_edges();
        const auto t_edge = t.nsecsElapsed();
        t.restart();
        calib_->pump_calib_channels();   // extra cameras: calibration only, never the pose
        const auto t_calib = t.nsecsElapsed();
        pump_ns_edge_ += t_edge; pump_ns_calib_ += t_calib; ++pump_ticks_;
        const auto now_ms = QDateTime::currentMSecsSinceEpoch();
        if (pump_report_ms_ == 0) pump_report_ms_ = now_ms;
        else if (now_ms - pump_report_ms_ > 5000)
        {
            const double secs = 1e-3 * static_cast<double>(now_ms - pump_report_ms_);
            // ★ The CONVERSION ratio belongs on the same line as the extraction cost, because the
            //   two answer one question between them. The pumps measured 5.5% of a core, which is
            //   NOT where this agent's CPU goes; the frames those pumps read are pulled and greyed
            //   on the per-camera ingest threads, and that is what minConvertIntervalMs throttles.
            //   Printing converted/delivered per camera is what makes the throttle falsifiable —
            //   a ratio of 1.00 means it never fired, and no CPU number would have said so.
            QString conv;
            const auto add_conv = [&conv](const std::string &name, std::pair<long, long> st)
            {
                conv += QString(" %1 %2/%3").arg(QString::fromStdString(name))
                            .arg(st.second).arg(st.first);   // converted / delivered
            };
            conv += QString::fromStdString(calib_->convert_stats_line());
            qInfo().nospace().noquote()
                << "[pumps] over " << secs << " s: image_edge " << (1e-6 * pump_ns_edge_ / secs)
                << " ms/s  calib_channels " << (1e-6 * pump_ns_calib_ / secs)
                << " ms/s  (= % of one core) | " << pump_ticks_ << " ticks | conv" << conv
                << " (converted/delivered)";
            pump_report_ms_ = now_ms; pump_ns_edge_ = pump_ns_calib_ = 0; pump_ticks_ = 0;
        }
    }

    QElapsedTimer section_timer;
    section_timer.start();
    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;
    t_loc_fetch_us = section_timer.nsecsElapsed() / 1000;

    const Eigen::Affine2f pose_for_draw = viewer_->best_available_pose(loc_res, have_loc);
    
    // ── Update 2-D viewer ─────────────────────────────────────────────────
    const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    const bool use_loc = have_loc && !loc_res->lidar_scan.empty();

    std::vector<Eigen::Vector3f> lidar_for_canvas;
    if (use_loc)
        lidar_for_canvas = loc_res->lidar_scan;
    else
    {
        const auto& [lidar_from_buffer] = lidar_ingestor_->buffer().read_last();
        if (lidar_from_buffer.has_value())
            lidar_for_canvas = lidar_from_buffer->first;
    }

    const bool on_gui_thread = (QThread::currentThread() == this->thread());
    if (on_gui_thread)
    {
        section_timer.restart();
        viewer_->update_viewer(loc_res, have_loc, pose_for_draw, lidar_for_canvas, loc_pose, use_loc);
        viewer_->draw_landmarks(scene_graph_->pinned_landmarks(), scene_graph_->pinned_measured(), pose_for_draw);
        t_viewer_us = section_timer.nsecsElapsed() / 1000;
    }

    // ── DSR graph update (only on fresh localization frames) ──────────────
    // In PreserveBootstrapRoom mode the room is a static prior: do NOT create/reparent the
    // room or write robot->room (that would clobber the fixed low-cov bootstrap edge). The
    // localizer still runs for the viewer; it just doesn't touch the graph.
    // Publish near the lidar rate (~60 ms) so the RT timestamped history is dense enough for
    // consumers to bracket a recent lidar-stamped query (e.g. the controller's overlay) instead of
    // clamping to a stale block. Steady RT updates on one edge — not join/leave churn — so low risk.
    {
        section_timer.restart();
        // FIX 2026-09-03: this redundant call REMOVED. The localizer's own on_result_ready callback
        // now calls maybe_publish_corrected_pose() directly (no more QueuedConnection marshal to this
        // thread — see the FIX note at set_on_result_ready), so it is no longer guaranteed to run on
        // the main thread. Calling it again from here would race the localizer thread's call on the
        // dedup state (last_dsr_publish_try_ms_/last_dsr_published_ts_ms_, not atomic) and on the
        // ofstream members touched inside dsr_update_calibration/dsr_update_affordance — exactly the
        // class of bug the old comment's "whichever fires first wins" was quietly relying on both
        // firing on the same thread to make safe. did_publish is no longer meaningful per compute()
        // tick now that publishing is fully decoupled from the compute cadence; see pose_trace.csv /
        // optimizer_timing.csv for the real publish timing.
        did_publish = false;
        t_dsr_us = section_timer.nsecsElapsed() / 1000;
    }

    // Visual RT-rate monitor: refresh the custom-widget readout once per second (low freq, cheap).
    pose_pub_->update_rt_rate_readout(now_ms, on_gui_thread);

    if (on_gui_thread)
    {
        section_timer.restart();
        viewer_->update_ui(loc_res);
        // Room-stabilization indicator: AMBER while a global grid search is relocating the robot,
        // GREEN once the room node exists in the graph (the very condition downstream consumers gate
        // on), RED while stable frames are still accumulating.
        viewer_->set_room_stable(scene_graph_->room_node_created(),
                                 scene_graph_->stable_frames(),
                                 params.STABLE_FRAMES_REQUIRED,
                                 room_concept_.is_grid_searching());
        t_ui_us = section_timer.nsecsElapsed() / 1000;
    }

    t_health_us = 0;

    const auto total_ms = compute_timer.elapsed();
    // Sub-millisecond resolution total. MICROSECONDS — named total_us everywhere it is emitted so it
    // cannot be mistaken for milliseconds. total_ms below is kept ONLY for the >50 ms stall trigger.
    const auto elapsed_since_init_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - init_time).count();

    // Per-tick compute-timing CSV — shows whether compute() holds the period (≈50 ms) or stalls, and
    // which section is responsible, explaining publish-rate loss vs the optimizer's production rate.
    if (not compute_csv_open_attempted_)
    {
        compute_csv_open_attempted_ = true;
        compute_csv_.open("etc/compute_timing.csv", std::ios::out | std::ios::trunc);
        if (compute_csv_.is_open())
            compute_csv_ << "wall_ms,total_us,affordance_us,loc_fetch_us,viewer_us,dsr_us,ui_us,did_publish,gui_thread\n";
    }
    if (compute_csv_.is_open())
    {
        compute_csv_ << now_ms << ',' << elapsed_since_init_us << ',' << t_affordance_us << ',' << t_loc_fetch_us
                     << ',' << t_viewer_us << ',' << t_dsr_us << ',' << t_ui_us << ','
                     << (did_publish ? 1 : 0) << ',' << (on_gui_thread ? 1 : 0) << '\n';
        compute_csv_.flush();
    }

    if (last_compute_timing_log_ms_ == 0 || now_ms - last_compute_timing_log_ms_ >= 3000 || total_ms > 50)
    {
        last_compute_timing_log_ms_ = now_ms;
        qInfo() << "[Timing][compute]"
                << "total_us=" << elapsed_since_init_us   // MICROSECONDS, as are every section below
                << "affordance_us=" << t_affordance_us
                << "loc_fetch_us=" << t_loc_fetch_us
                << "viewer_us=" << t_viewer_us
                << "dsr_us=" << t_dsr_us
                << "ui_us=" << t_ui_us
                << "health_us=" << t_health_us
                << "gui_thread=" << on_gui_thread;
    }
    fps_counter_.print("[Compute]", 3000);
}

void SpecificWorker::initialize_room_model_from_svg()
{
    // The scenario names the FILE; LayoutDir says where the layouts live. An explicit path in the
    // config (anything containing a separator) wins, so a one-off layout outside the folder still
    // works without a config-shape change.
    const std::string svg_path =
        (params.ROOM_LAYOUT_SVG.find('/') != std::string::npos or params.LAYOUT_DIR.empty())
            ? params.ROOM_LAYOUT_SVG
            : params.LAYOUT_DIR + "/" + params.ROOM_LAYOUT_SVG;
    auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        svg_path, "room_contour", false, true);
    if (room_polygon.size() < 3)
        qWarning() << "[room] layout" << QString::fromStdString(svg_path)
                   << "gave fewer than 3 points — check RoomConcept.LayoutDir and the scenario's"
                      " RoomLayoutSvg. The layouts moved to active_inference/layouts on 2026-08-29.";
    if (room_polygon.size() >= 3)
    {
        // Move the room-frame origin onto the layout's geometric centre. Everything downstream is
        // origin-agnostic (the grid search derives its box from the polygon bbox, the planner grid
        // carries its own origin, and the DSR `room` node republishes these very vertices as
        // delimiting_polygon_x/y), so the shift propagates on its own.
        const Eigen::Vector2f offset = params.RECENTER_ROOM_POLYGON
                                   ? rc::SvgRoomLoader::recenter_to_bbox_center(room_polygon)
                                   : Eigen::Vector2f::Zero();
        if (offset.norm() > 1e-3f)
            qInfo() << "[room] Room polygon recentred on its bbox centre: origin moved by"
                    << offset.x() << offset.y() << "m."
                    << "Poses in the room frame are shifted by that amount — a seed pose or object"
                    << "RT edges saved before this change are stale.";

        calib_->set_room_polygon(room_polygon, offset);
        room_concept_.configure_room_from_polygon(calib_->room_polygon());
        room_initialized_from_svg_polygon_ = true;
        return;
    }
    calib_->set_room_polygon({}, Eigen::Vector2f::Zero());
    room_concept_.configure_room_from_rect(params.GRID_MAX_DIM.width(), params.GRID_MAX_DIM.height());
    room_initialized_from_svg_polygon_ = false;
    qWarning() << "SVG polygon not loaded; using rectangular fallback.";
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::save_robot_pose_on_exit() const
{
    // An estimated map defines its own frame each run; a seed pose from it would mislead a later
    // Given-mode start against the layout (and an Estimate-mode start ignores seeds anyway).
    if (room_concept_.estimating()) return;
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();
    if (const auto loc = room_concept_.get_last_result(); loc.has_value() && loc->ok)
    {
        pose[0] = loc->state[2]; pose[1] = loc->state[3]; pose[2] = loc->state[4];
    }
    else if (room_concept_.is_initialized())
    {
        const auto state = room_concept_.get_current_state();
        pose[0] = state[2]; pose[1] = state[3]; pose[2] = state[4];
    }
    else return;

    const QString qpath = QString::fromStdString(pose_file_path());
    QDir().mkpath(QFileInfo(qpath).absolutePath());
    std::ofstream out(qpath.toStdString(), std::ios::trunc);
    if (!out.is_open()) { qWarning() << "Cannot open pose file:" << qpath; return; }
    out << pose[0] << ' ' << pose[1] << ' ' << pose[2] << '\n';
}

void SpecificWorker::save_robot_pose_once()
{
    if (pose_saved_.exchange(true)) return;
    save_robot_pose_on_exit();
}

std::string SpecificWorker::pose_file_path() const
{
    auto find_etc_upwards = [](const QString& start) -> QString {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth)
        {
            const QString etc_dir = dir.absoluteFilePath("etc");
            if (QDir(etc_dir).exists()) return etc_dir;
            if (!dir.cdUp()) break;
        }
        return {};
    };
    const QString from_app = find_etc_upwards(QCoreApplication::applicationDirPath());
    if (!from_app.isEmpty()) return (from_app + "/last_robot_pose.txt").toStdString();
    const QString from_cwd = find_etc_upwards(QDir::currentPath());
    if (!from_cwd.isEmpty()) return (from_cwd + "/last_robot_pose.txt").toStdString();
    return (QDir(QCoreApplication::applicationDirPath() + "/../etc").absolutePath()
            + "/last_robot_pose.txt").toStdString();
}


///////////////////////////////////////////////////////////////////////////////
/// SLOTS from GUI and DSR signals
///////////////////////////////////////////////////////////////////////////////
/// @brief ///////////DSR callback triggered when a node is modified. We check if it's the robot node and if the current speed attributes have been updated, then we read them and push them to the odometry buffer with some optional noise added.

// The DSR graph slots (modify_node_attrs_slot, modify_node_slot) live in specificworker_slots.cpp —
// same class, separate translation unit: they are 220 lines of dispatch, and dispatch is what this
// worker is for, so they stay its methods rather than becoming a class of their own.

void SpecificWorker::trigger_graph_layout_twopi()
{
    auto graph_viewer_owner = find_graph_viewer("");
    if (not graph_viewer_owner)
        return;

    QWidget* graph_widget = graph_viewer_owner->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (not graph_viewer)
        return;

    // Run now and once queued, so layout also happens after pending node/edge
    // update signals are processed by the viewer.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer]() { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}
///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
}

void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}


///////////////////////////////////////////////////////////////////////////////
/// ICE INTERFACE CALLBACKS
///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    rc::VelocityCommand cmd;
    float raw_adv_y = 0.f;
    float raw_rot = 0.f;
    for (const auto& axis : data.axes)
    {
        if      (axis.name == "rotate")
        {
            raw_rot = axis.value;
            cmd.rot = axis.value;
        }
        else if (axis.name == "advance")
        {
            raw_adv_y = axis.value / 1000.0f;
            cmd.adv_y = raw_adv_y;
        }
        else if (axis.name == "side")    cmd.adv_x = 0.0f;
    }
    cmd.timestamp  = std::chrono::high_resolution_clock::now();
    cmd.recv_ts_ms = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    room_concept_.record_command_ingress("joystick",
                                         raw_adv_y,
                                         raw_rot,
                                         cmd.adv_y,
                                         cmd.rot,
                                         cmd.recv_ts_ms);
    velocity_buffer_.put<0>(std::move(cmd), static_cast<std::uint64_t>(cmd.recv_ts_ms));
}
