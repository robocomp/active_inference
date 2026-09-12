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
    camera_ingestor_.reset();  // stop the RGB reader before the graph goes
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
    imu_ingestor_->set_on_new_sample([this](std::int64_t imu_ts_ms) { publish_predicted_tick(imu_ts_ms); });
    // RGB edge alignment. Constructed ONLY when enabled: with ImageEdge.enable = false there is no
    // subscriber, no thread and no extraction, so the feature is exactly free when off.
    if (params.IMAGE_EDGE_ENABLE)
    {
        camera_ingestor_ = std::make_unique<rc::CameraIngestor>(G, params.IMAGE_EDGE_CAMERA);
        // Set BEFORE the first bind_camera(): the correction is applied where the extrinsic is read.
        camera_ingestor_->set_mount_yaw_correction(params.IMAGE_EDGE_MOUNT_YAW_CORR);
        // Throttle the grey CONVERSION, never the drain — see CameraIngestor and the config note.
        camera_ingestor_->set_min_convert_interval_ms(params.IMAGE_EDGE_MIN_CONVERT_MS);

        image_edge_source_ = std::make_unique<rc::ImageEdgeSource>();
        rc::ImageEdgeSource::Config ic;
        ic.use_wall_corners    = params.IMAGE_EDGE_USE_WALL_CORNERS;
        ic.use_floor_junction  = params.IMAGE_EDGE_USE_FLOOR_JUNCTION;
        ic.use_wall_ceiling    = params.IMAGE_EDGE_USE_WALL_CEILING;
        ic.sample_spacing_m    = params.IMAGE_EDGE_SAMPLE_SPACING_M;
        ic.search_sigmas       = params.IMAGE_EDGE_SEARCH_SIGMAS;
        ic.max_search_px       = params.IMAGE_EDGE_MAX_SEARCH_PX;
        ic.mount_pitch_sigma   = params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
        ic.mount_height_sigma  = params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
        ic.mount_yaw_sigma     = params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
        ic.wall_position_sigma = params.IMAGE_EDGE_WALL_POS_SIGMA;
        ic.room_height         = params.room_height;
        image_edge_source_->set_config(ic);

        // ── Extra CALIBRATION channels ─────────────────────────────────────────────────────────
        // ★ AFTER image_edge_source_ is built and configured. This block sat BEFORE it and
        //   called image_edge_source_->config() on a null unique_ptr — a segfault on the
        //   very first start, before the graph was even up. Second time today that adding
        //   setup beside related code put it ahead of the thing it depends on; the
        //   dependency is on CONSTRUCTION ORDER and nothing in the surrounding code shows it.──
        // The driving camera is skipped: it already has an ingestor above, and running it twice
        // would feed its own evidence file from two extractions of the same frames — the exact
        // double-count the triple-point/segment choice exists to avoid.
        for (const auto &cam : params.CALIB_CAMERAS)
        {
            if (cam == params.IMAGE_EDGE_CAMERA) continue;
            auto ch = std::make_unique<CalibChannel>();
            ch->name = cam;
            ch->ingestor = std::make_unique<rc::CameraIngestor>(G, cam);
            // ★ The CALIB interval, not the driving one. A calibration channel's frames never
            //   reach the loss — they feed an estimator that accumulates H and b over minutes — so
            //   skipping frames here costs coverage, not evidence. Sharing one number with the
            //   driving camera would pay for this saving with the pose factor's rate.
            ch->ingestor->set_min_convert_interval_ms(params.IMAGE_EDGE_CALIB_MIN_CONVERT_MS);
            // ★ The yaw correction is per CAMERA and is NOT shared. It was measured for the zed; a
            //   second camera has its own mount and applying one camera's correction to another
            //   would be a fabricated extrinsic.
            ch->source = std::make_unique<rc::ImageEdgeSource>();
            ch->source->set_config(ic);   // the same config, from the local `ic` — see below
            calib_channels_.push_back(std::move(ch));
            qInfo() << "[camcal] calibration channel for" << QString::fromStdString(cam)
                    << "(does not drive the pose)";
        }
    }

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
        room_polygon_.clear();
        room_polygon_offset_ = Eigen::Vector2f::Zero();
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
    const std::vector<Eigen::Vector2f>& room_polygon_for_viz = room_polygon_;

    // GUI / visualization (2-D viewer, FE plot, camera-projection window + RGB media plane).
    viewer_ = std::make_unique<rc::RoomViewer>(
        G, params, room_polygon_for_viz,
        room_initialized_from_svg_polygon_, room_concept_, epistemic_controller_);

    // AFTER the viewer exists. Registering this beside the ingestor setup (where the camera calib
    // is otherwise configured) put it before make_unique, so the `if (viewer_)` was false and the
    // handler was never installed — a Reset that silently cleared only half the evidence.
    phase("viewer_window");   // the window EXISTS from here; everything after this is visible hang time
    viewer_->set_camera_reset_handler([this]
    {
        mp_pool_.reset();
        mp_win_.reset();
        std::remove(mp_pool_.path().c_str());
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
    room_concept_.set_on_result_ready([this]() { maybe_publish_corrected_pose(); });

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
                if (camera_ingestor_) camera_ingestor_->start();
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
void SpecificWorker::update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread)
{
    if (rt_rate_window_start_ms_ == 0)
    {
        rt_rate_window_start_ms_ = now_ms;
        return;
    }
    const std::int64_t elapsed = now_ms - rt_rate_window_start_ms_;
    if (elapsed < 1000)
        return;

    const float dt_s    = static_cast<float>(elapsed) * 0.001f;
    const float corr_hz = static_cast<float>(rt_corr_count_) / dt_s;   // RT publishes = corrected poses

    // Optimizer timing (loc thread): processing rate, mean cost/update, and early-exit fraction —
    // diagnoses whether localization is compute-bound (low Hz, high ms) and CUDA/window effects.
    const auto opt = room_concept_.take_optimizer_timing();
    const float opt_hz = static_cast<float>(opt.count) / dt_s;
    const float ee_pct = (opt.count > 0)
        ? 100.0f * static_cast<float>(opt.early_exits) / static_cast<float>(opt.count) : 0.0f;

    // Process CPU usage over the same window (sum across all threads — loc/GUI/ingest — so >100% on a
    // busy multithreaded frame is expected, exactly like `top`'s per-process %CPU). getrusage is portable
    // and needs no /proc parsing; delta CPU-seconds / wall-seconds × 100.
    float cpu_pct = 0.0f;
    {
        static double last_cpu_s = -1.0;
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
        {
            const double cpu_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
                               + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
            if (last_cpu_s >= 0.0)
                cpu_pct = static_cast<float>(100.0 * (cpu_s - last_cpu_s) / dt_s);
            last_cpu_s = cpu_s;
        }
    }

    // ── Cost, logged rather than only displayed ──────────────────────────────────────────────────
    // The experiment compares configurations that differ in what they compute — a second camera
    // channel, a corner factor in the loss — so CPU is a RESULT, not housekeeping: an accuracy gain
    // bought with 40% more CPU is a different claim from one that is free.
    // ★ The active configuration is written into the header. Runs are compared days apart from
    //   archived files, and a cost figure whose condition has to be reconstructed from memory is
    //   worth very little — this file says what produced it.
    {
        static std::ofstream res_csv;
        if (not res_csv.is_open())
        {
            res_csv.open("etc/resource_usage.csv", std::ios::out | std::ios::trunc);
            if (res_csv.is_open())
            {
                res_csv.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                std::string cams;
                for (const auto &c : params.CALIB_CAMERAS)
                { if (not cams.empty()) cams += "+"; cams += c; }
                res_csv << "# camera=" << params.IMAGE_EDGE_CAMERA
                        << " drive=" << (params.IMAGE_EDGE_DRIVE ? 1 : 0)
                        << " shadow=" << (params.IMAGE_EDGE_SHADOW ? 1 : 0)
                        << " triple=" << (params.IMAGE_EDGE_USE_TRIPLE_POINTS ? 1 : 0)
                        << " calib_cameras=" << (cams.empty() ? "none" : cams)
                        << " enable=" << (params.IMAGE_EDGE_ENABLE ? 1 : 0) << '\n';
                res_csv << "ts_ms,cpu_pct,rss_mb,opt_hz,corr_hz,early_exit_pct,avg_update_ms\n";
            }
        }
        if (res_csv.is_open())
        {
            struct rusage ru2;
            const double rss_mb = (getrusage(RUSAGE_SELF, &ru2) == 0)
                                      ? ru2.ru_maxrss / 1024.0 : 0.0;   // ru_maxrss is KiB on Linux
            res_csv << now_ms << ',' << cpu_pct << ',' << rss_mb << ',' << opt_hz << ','
                    << corr_hz << ',' << ee_pct << ',' << opt.avg_update_ms << '\n';
            res_csv.flush();
        }
    }

    if (on_gui_thread && viewer_)
    {
        // RT/opt rates → live plot (trend). The text readout keeps only the scalar optimizer-health
        // numbers that aren't plotted (ms per update, early-exit %, process CPU%).
        viewer_->add_rate_samples(corr_hz, opt_hz);
        viewer_->set_rt_rate_text(
            QString("%1 ms/upd · %2% early-exit · %3% CPU")
                .arg(opt.avg_update_ms, 0, 'f', 1)
                .arg(ee_pct, 0, 'f', 0)
                .arg(cpu_pct, 0, 'f', 0));
    }

    rt_corr_count_ = 0;
    rt_rate_window_start_ms_ = now_ms;
}

// Append one pose-trace row. type 0=corrected (optimizer, ~20 Hz), 1=predicted (dead-reckon, ~60 Hz).
// valid_ts_ms = the pose's validity stamp; wall_ms = QDateTime now. Lets us overlay the intermediate
// predicted poses on the corrected ones and see the dead-reckoning wander / correction snaps.
void SpecificWorker::log_pose_trace(int type, std::int64_t valid_ts_ms,
                                    const Eigen::Affine2f& pose, float innov_norm)
{
    if (not pose_trace_open_attempted_)
    {
        pose_trace_open_attempted_ = true;
        pose_trace_.open("etc/pose_trace.csv", std::ios::out | std::ios::trunc);
        if (pose_trace_.is_open())
            pose_trace_ << "wall_ms,type,valid_ts_ms,x,y,theta,innov_norm\n";
    }
    if (not pose_trace_.is_open())
        return;
    const float x = pose.translation().x();
    const float y = pose.translation().y();
    const float th = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
    const std::int64_t wall_ms = QDateTime::currentMSecsSinceEpoch();
    pose_trace_ << wall_ms << ',' << type << ',' << valid_ts_ms << ','
                << x << ',' << y << ',' << th << ',' << innov_norm << '\n';
    pose_trace_.flush();

    // ── AND, IF THIS STEP WAS PHYSICALLY IMPOSSIBLE, RECORD IT SOMEWHERE THAT SURVIVES ──────────
    if (type < 0 or type > 1)
        return;
    TracePoint now{wall_ms, valid_ts_ms, x, y, th, innov_norm, true};
    const TracePoint before = prev_trace_[type];
    prev_trace_[type] = now;
    if (not before.set)
        return;

    const double dt_s = static_cast<double>(wall_ms - before.wall_ms) * 1e-3;
    if (dt_s <= 1e-3 or dt_s > 1.0)     // first row after a gap: nothing to compare against
        return;

    // The bound is what the BASE SAYS IT CAN DO, read off the robot node. No capability published ⇒
    // say so once and stay quiet, rather than inventing a number and calling its output evidence.
    rc::BaseCapability cap;
    if (G != nullptr)
        if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
            cap = rc::read_base_capability(*G, robots.front().id());
    // ★EACH BOUND IS USED IF PUBLISHED, AND THE ABSENCE OF ONE DOES NOT DISABLE THE OTHER.
    // ⚠CORRECTION (2026-09-11): the commit that added this block, and an earlier version of this
    // comment, both asserted that the base declares no maximum linear speed. That was WRONG and the
    // error was mine: SVD48VBase config_diferential.toml has always carried `maxLinSpeed`, and a
    // grep pattern of mine failed to match the key. Both bounds have been published all along
    // (maxLinSpeed, now 1500 mm/s; maxRotSpeed 2 rad/s), so the linear check was never inert.
    // The independent-bounds design below is kept anyway, because it is right for a robot that
    // declares only one -- but it was not, as claimed, necessary here.
    // ★EITHER BOUND ALONE WOULD HAVE CAUGHT WHAT WAS SEEN. Both 2026-09-10 events carried 43.6 and
    // 89.2 degrees of heading step (89 deg in 34 ms is ~46 rad/s against a 2 rad/s base) as well as
    // impossible translation. The `trigger` column records which fired, so a reader never has to
    // assume both were checked.
    if (not cap.max_linear_speed_mps.has_value() and not cap.max_rot_speed_rps.has_value())
    {
        if (not pose_jump_no_capability_warned_)
        {
            pose_jump_no_capability_warned_ = true;
            qWarning() << "[pose_jumps] the base publishes NEITHER a linear nor a rotational speed"
                       << "capability, so no step can be called impossible — etc/pose_jumps.csv will"
                       << "stay empty. That is a MISSING CHANNEL, not an absence of jumps.";
        }
        return;
    }

    const double step_m      = std::hypot(now.x - before.x, now.y - before.y);
    const double implied_mps = step_m / dt_s;
    const double dtheta      = std::remainder(static_cast<double>(now.th - before.th), 2.0 * M_PI);
    const double implied_rps = std::abs(dtheta) / dt_s;
    const bool   too_fast    = cap.max_linear_speed_mps.has_value()
                               and implied_mps > cap.max_linear_speed_mps.value();
    const bool   too_spinny  = cap.max_rot_speed_rps.has_value()
                               and implied_rps > cap.max_rot_speed_rps.value();
    if (not too_fast and not too_spinny)
        return;
    const char *trigger = (too_fast and too_spinny) ? "both" : too_fast ? "linear" : "rotational";

    if (not pose_jump_log_attempted_)
    {
        pose_jump_log_attempted_ = true;
        pose_jump_run_id_ = wall_ms;
        // ★APPEND, never truncate. The header goes in only when the file is new, so a week of runs
        // accumulates into one file and run_id_ms separates them.
        const bool fresh = not std::filesystem::exists("etc/pose_jumps.csv")
                           or std::filesystem::file_size("etc/pose_jumps.csv") == 0;
        pose_jump_log_.open("etc/pose_jumps.csv", std::ios::out | std::ios::app);
        if (pose_jump_log_.is_open())
        {
            pose_jump_log_.imbue(std::locale::classic());   // never emit a decimal comma under es_ES
            if (fresh)
                pose_jump_log_ << "run_id_ms,wall_ms,type,dt_ms,trigger,"
                                  "x_before,y_before,th_before,valid_ts_before,innov_before,"
                                  "x_after,y_after,th_after,valid_ts_after,innov_after,"
                                  "step_m,implied_mps,dtheta_rad,implied_rps,cap_mps,cap_rps\n";
            qInfo() << "[pose_jumps] recording localiser discontinuities to etc/pose_jumps.csv"
                    << "(append-only; run id" << pose_jump_run_id_ << ")";
        }
    }
    if (not pose_jump_log_.is_open())
        return;

    // An unpublished bound is written EMPTY, never 0 — a zero here would read as a real capability
    // of zero, i.e. every step impossible, which is the opposite of what absence means.
    pose_jump_log_ << pose_jump_run_id_ << ',' << wall_ms << ',' << type << ','
                   << static_cast<std::int64_t>(dt_s * 1000.0) << ',' << trigger << ','
                   << before.x << ',' << before.y << ',' << before.th << ','
                   << before.valid_ts_ms << ',' << before.innov << ','
                   << now.x << ',' << now.y << ',' << now.th << ','
                   << now.valid_ts_ms << ',' << now.innov << ','
                   << step_m << ',' << implied_mps << ',' << dtheta << ',' << implied_rps << ',';
    if (cap.max_linear_speed_mps.has_value()) pose_jump_log_ << cap.max_linear_speed_mps.value();
    pose_jump_log_ << ',';
    if (cap.max_rot_speed_rps.has_value())    pose_jump_log_ << cap.max_rot_speed_rps.value();
    pose_jump_log_ << '\n';
    pose_jump_log_.flush();
}

bool SpecificWorker::maybe_publish_corrected_pose()
{
    // Called from the LOCALIZER thread directly (on_result_ready, no marshal since the 2026-09-03
    // concurrency fix -- see set_on_result_ready) and, redundantly, is NOT also called from compute()
    // any more (see the FIX note there) precisely because two threads landing here would race on the
    // dedup state below and on the ofstream members inside dsr_update_calibration/dsr_update_affordance.
    // In PreserveBootstrapRoom mode the room is a static prior — the localizer still runs for the
    // viewer, but we never touch the graph.
    if (shutting_down_.load() || params.PRESERVE_BOOTSTRAP_ROOM)
        return false;

    // FIX 2026-09-04: held for the rest of this function. Shared with publish_predicted_tick(), which
    // runs on the imu-ingest thread and reads last_published_pose_/last_pub_adv_/_side_/_rot_/
    // last_published_cov_ -- all written below -- and calls into the same scene_graph_.
    std::lock_guard<std::mutex> publish_lock(publish_mutex_);

    // NOT const: get_last_result() returns a by-value copy, and the kinematic clamp below rewrites
    // robot_pose/covariance in place before this frame is handed to the scene graph.
    auto loc_res = room_concept_.get_last_result();
    if (!loc_res.has_value() || !loc_res->ok)
        return false;

    // Dedup by the lidar stamp so the compute() path and the immediate-publish path are idempotent
    // (whichever fires first for a given frame wins; the other no-ops). The throttle is a small
    // anti-burst floor only — publishing runs at the optimizer/lidar rate, not the compute rate.
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    const bool fresh_correction = loc_res->timestamp_ms > 0
                                  && loc_res->timestamp_ms != last_dsr_published_ts_ms_;
    if (!fresh_correction
        || (last_dsr_publish_try_ms_ != 0 && now_ms - last_dsr_publish_try_ms_ < 15))
        return false;

    last_dsr_publish_try_ms_ = now_ms;

    // ---- KINEMATIC CLAMP on the published CORRECTION --------------------------------------------
    // The published stream must be a physically possible trajectory: between two published poses the
    // robot cannot have jumped further than its own speed limit allows. Measured on this agent's own
    // debug log (71k frames), it routinely did — 92.8% of frames take the prediction early-exit and
    // publish a drifting odometry prediction, then the optimizer runs on the remaining 7.2% and
    // discharges the whole accumulated error in ONE frame: |innovation| p99 96.7 mm, max 280.2 mm,
    // i.e. an implied 2.2 m/s at p99 and 7.2 m/s peak against a 0.70 m/s robot.
    //
    // This is not a tuning threshold — it is the support of the motion model. A delta implying 7.2 m/s
    // is not unlikely, it is impossible, so it cannot be a report about where the robot went.
    //
    // ★ WHAT IS BOUNDED IS THE CORRECTION, NOT THE POSE DELTA. The published delta is the sum of two
    // things with completely different supports: the motion the sensors MEASURED (bounded by the
    // robot's dynamics, and already a report about where it went), and the optimizer's correction on
    // top of it (a teleport, bounded by nothing). Bounding their SUM against W_MAX made the published
    // yaw rate a hard rate limiter at exactly W_MAX: measured 2026-08-28 on the Webots P3Bot, the
    // published |omega| maxed at 0.8003 rad/s across 60k frames while ground truth reached 3.51 rad/s,
    // 4.5% of frames sat pinned in [0.75, 0.80], and 2221 deg of real rotation was forbidden over one
    // run. The published pose then falls behind DURING a pivot and only catches up once the true rate
    // drops back under the cap — which is exactly the reported symptom: the controller's LiDAR cloud
    // and robot icon swing >100 deg off the walls while turning and snap back when the turn ends,
    // while room_concept's own viewer (which draws the UNCLAMPED last_result_) looks perfect
    // throughout. The clamp had zero headroom by construction, because W_MAX was set to the
    // controller's MaxRotSpeed — the very rate the robot is commanded to reach.
    //
    // So: let the measured motion (pred[k] - pred[k-1]) through untouched, and rate-limit only the
    // residual on top of it. A genuine relocalization still converges over a few frames; a pivot at
    // full speed is no longer misreported as a slower one.
    //
    // The correction itself is LEGITIMATE (real accumulated drift, not a bad fit), so it is not
    // rejected — only rate-limited.
    // Captured BEFORE the clamp rewrites robot_pose: this is the localiser's own estimate, which is
    // the base the NEXT prediction will be built on. It is what the next frame's measured-motion
    // difference must reference -- never the clamped pose, which no predictor ever sees.
    const Eigen::Affine2f est_pre_clamp = loc_res->robot_pose;

    bool clamp_fired = false;
    if (params.POSE_CLAMP_ENABLED and last_published_pose_.has_value()
        and last_published_ts_ms_ > 0 and loc_res->timestamp_ms > last_published_ts_ms_)
    {
        const float dt = static_cast<float>(loc_res->timestamp_ms - last_published_ts_ms_) / 1000.f;
        if (dt > 0.f and dt <= params.POSE_CLAMP_MAX_DT_S)
        {
            const auto& prev = last_published_pose_.value();
            const Eigen::Vector2f d_xy = loc_res->robot_pose.translation() - prev.translation();
            const float prev_th = std::atan2(prev.linear()(1, 0), prev.linear()(0, 0));
            const float new_th  = std::atan2(loc_res->robot_pose.linear()(1, 0),
                                             loc_res->robot_pose.linear()(0, 0));
            const float d_th = std::remainder(new_th - prev_th, 2.f * static_cast<float>(M_PI));

            // The motion the sensors MEASURED between the last published frame and this one.
            // build_motion_prior_selection() builds every prediction on last_update_result.robot_pose
            // — the previous CORRECTED estimate — so pred[k] - est[k-1] is exactly the preintegrated
            // increment for this interval, with nothing else in it. Verified on the live log after the
            // clamp fix: pred[k] - (est[k-1] + imu_dtheta) is 0.0002 deg at p50.
            //
            // ⚠ IT MUST BE est[k-1], NOT pred[k-1]. Differencing two consecutive PREDICTIONS looks
            // equivalent and is not: substituting pred[k-1] = est[k-2] + increment[k-1] gives
            // pred[k] - pred[k-1] = increment[k] + innovation[k-1], i.e. it hands the previous frame's
            // correction through the clamp UNBOUNDED, one frame late. That was the bug in the first
            // version of this fix, and it survived because a free-running predictor would make the two
            // forms identical — the log column it was checked against (est_theta) is written AFTER the
            // clamp, so it showed the clamp's own backlog and not the localiser's estimate.
            //
            // With no previous estimate to difference against (first publication after a reset) it is
            // zero, which degrades to the old total-delta behaviour for that single frame.
            //
            // Any correction the clamp did not apply on earlier frames is deliberately left INSIDE the
            // bounded part here, so a backlog still discharges at max_th per frame rather than
            // escaping as one jump.
            Eigen::Vector2f m_xy = Eigen::Vector2f::Zero();
            float           m_th = 0.f;
            if (last_published_est_.has_value())
            {
                const Eigen::Vector3f& pe = last_published_est_.value();
                m_xy = Eigen::Vector2f(loc_res->pred_x - pe.x(), loc_res->pred_y - pe.y());
                m_th = std::remainder(loc_res->pred_theta - pe.z(), 2.f * static_cast<float>(M_PI));
            }

            // What is left after the measured motion is accounted for IS the correction — the only
            // part of the published delta that has no dynamical support and can therefore be a
            // teleport. Bound that, and only that.
            const Eigen::Vector2f c_xy = d_xy - m_xy;
            const float           c_th = std::remainder(d_th - m_th, 2.f * static_cast<float>(M_PI));

            // Take the bound from the robot before using it. Cheap after the first success (one bool),
            // and placed HERE rather than in initialize() so it cannot silently fall back for the life
            // of the process because the robot node had not synced yet.
            apply_base_capability_to_pose_clamp();
            const float max_xy = params.POSE_CLAMP_V_MAX * dt;
            const float max_th = params.POSE_CLAMP_W_MAX * dt;
            const float n_xy = c_xy.norm();

            if (n_xy > max_xy or std::abs(c_th) > max_th)
            {
                const Eigen::Vector2f xy_c = m_xy + ((n_xy > max_xy and n_xy > 1e-9f)
                                             ? Eigen::Vector2f(c_xy * (max_xy / n_xy)) : c_xy);
                const float th_c = m_th + std::clamp(c_th, -max_th, max_th);

                // The part of the correction we did NOT apply is real, known error in the published
                // pose. Surfacing it as covariance is what keeps the clamp honest rather than a lie:
                // consumers learn the pose is deliberately behind the measurement, and it is the one
                // signal in this stream that is genuinely informative (it fires on ~1% of frames, so
                // it cannot shift the mean sigma the speed governor watches).
                const Eigen::Vector2f res_xy = d_xy - xy_c;
                const float res_th = d_th - th_c;
                loc_res->covariance(0, 0) += res_xy.x() * res_xy.x();
                loc_res->covariance(1, 1) += res_xy.y() * res_xy.y();
                loc_res->covariance(2, 2) += res_th * res_th;

                Eigen::Affine2f clamped = Eigen::Affine2f::Identity();
                clamped.translation() = prev.translation() + xy_c;
                clamped.linear() = Eigen::Rotation2Df(prev_th + th_c).toRotationMatrix();
                loc_res->robot_pose = clamped;

                clamp_fired = true;
                if (++pose_clamp_hits_ % 20 == 1)
                    qWarning() << "[pose-clamp] CORRECTION implied" << (n_xy / dt) << "m/s /"
                               << (std::abs(c_th) / dt) << "rad/s over dt" << dt << "s — limits"
                               << params.POSE_CLAMP_V_MAX << "/" << params.POSE_CLAMP_W_MAX
                               << "; measured motion passed through at" << (std::abs(m_th) / dt)
                               << "rad/s; carried" << res_xy.norm() * 1000.f
                               << "mm into covariance (" << pose_clamp_hits_ << " clamps)";
            }
        }
    }

    // Hand the ACTUALLY-PUBLISHED covariance back for the debug log. The clamp above runs downstream
    // of RoomConcept's own write, so cov_xx there is the PRE-clamp value while every consumer sees
    // this one; without recording it the published sigma is simply not observable from any log, which
    // is how a claim about it came to be made and then retracted.
    // ⚠ ONE-FRAME LAG BY CONSTRUCTION: RoomConcept writes its row before returning, so these values
    // land on the NEXT row. Join pub_cov_* to the frame BEFORE, not the one they appear on.
    room_concept_.note_published_covariance(loc_res->covariance(0, 0), loc_res->covariance(2, 2),
                                            clamp_fired);

    // FIX 2026-09-04: publish the CORRECTED twist, not the raw wheel-odometry snapshot.
    // last_robot_adv_speed_/_side_/_rot_ (kept as fallback below) is
    // -- the latest single sample of robot_concept's robot_current_speed attribute (wheel odometry,
    // ~10 Hz, uncorrected), cached in specificworker.cpp and forwarded here unchanged. Two problems
    // with that as the twist consumers extrapolate with (camera_visualizer.cpp, retina/scene_processor,
    // controller_obstacle_tracker): it can be up to ~100 ms stale (odometry at 10 Hz vs publish at
    // ~20 Hz), and it never received any of the corrections integrate_odometry_over_window() applies
    // to the localiser's OWN prediction -- gyro preference over wheel yaw (measured 8.2% over-report,
    // see room_concept.cpp:5991), the k_w/b_w scale+bias from the online calibrator, or the IMU linear
    // injection. This computes the same corrected quantity room_concept already trusts for its own
    // prediction, from the fields that quantity is already exposed through:
    //   loc_res->dy_local / dx_local -- BODY frame (dy=+Y forward, dx=+X lateral), the k_v/k_lat-scaled,
    //     IMU-injected translation accumulated THIS cycle (see room_concept.cpp:2916-2917/3512-3513).
    //   loc_res->imu_dtheta + wheel_dtheta -- the total rotation entered into the prior this cycle,
    //     whichever channel (gyro or wheel) supplied each segment (same fields motion_calib_.observe()
    //     already sums this way, room_concept.cpp:5732).
    // Divided by dt since the LAST PUBLISH (not a differentiated pose -- differentiating the published,
    // SDF-corrected pose is exactly the correction-induced velocity spike this twist channel was
    // introduced to avoid; see the frame-note in room_scene_graph.cpp). Falls back to the raw wheel
    // snapshot only when there is no previous publish to difference against (first frame) or the
    // interval is degenerate.
    float pub_adv = last_robot_adv_speed_, pub_side = last_robot_side_speed_, pub_rot = last_robot_rot_speed_;
    if (last_published_ts_ms_ > 0 and loc_res->timestamp_ms > last_published_ts_ms_)
    {
        const float dt_s = static_cast<float>(loc_res->timestamp_ms - last_published_ts_ms_) * 1e-3f;
        if (dt_s > 1e-4f)
        {
            pub_adv  = loc_res->dy_local / dt_s;                        // forward, body +Y
            pub_side = loc_res->dx_local / dt_s;                        // lateral, body +X
            pub_rot  = (loc_res->imu_dtheta + loc_res->wheel_dtheta) / dt_s;
        }
    }

    // Publish (corrected pose → robot↔room RT) at the optimizer rate.
    scene_graph_->update(*loc_res, pub_adv, pub_side, pub_rot);
    last_published_pose_ = loc_res->robot_pose;
    // Cache for publish_predicted_tick(): the twist and covariance an IMU-rate tick should extrapolate
    // with/report until the NEXT real publish replaces them. covariance is loc_res->covariance
    // (post-clamp, same one every other consumer of this publish sees -- see the note above on
    // note_published_covariance).
    last_pub_adv_       = pub_adv;
    last_pub_side_      = pub_side;
    last_pub_rot_        = pub_rot;
    last_published_cov_ = loc_res->covariance;
    // Anchor for the NEXT frame's measured-motion difference: the localiser's own estimate for this
    // frame, which is the base its next prediction is built on. Taken pre-clamp, so the difference
    // stays a pure sensor increment however hard the clamp bit here.
    last_published_est_ = Eigen::Vector3f(
        est_pre_clamp.translation().x(), est_pre_clamp.translation().y(),
        std::atan2(est_pre_clamp.linear()(1, 0), est_pre_clamp.linear()(0, 0)));
    last_published_ts_ms_ = loc_res->timestamp_ms;
    last_dsr_published_ts_ms_ = loc_res->timestamp_ms;

    ++rt_corr_count_;
    log_pose_trace(/*type=corrected*/0, loc_res->timestamp_ms,
                   loc_res->robot_pose, loc_res->innovation_norm);
    log_ground_truth(*loc_res);
    return true;
}

// FIX 2026-09-04: called from ImuIngestor's ingest thread, once per accepted IMU sample (~100 Hz on
// this robot, see imu_ingestor.cpp), so the graph is never more than one IMU period behind instead of
// waiting on the next lidar-paced correction (~20 Hz). Publishes a DEAD-RECKONED pose only -- it never
// touches last_published_pose_/_ts_ms_/_est_ (those anchor the kinematic clamp and the NEXT real
// prediction's measured-motion difference, and must only ever move on a real SDF-validated result) --
// so every tick extrapolates from the SAME last real publish with a growing dt, rather than compounding
// error by extrapolating from the previous tick's own guess.
//
// Deliberately narrow: calls dsr_publish_predicted_pose() (-> write_robot_room_rt(), the same
// NaN-guarded, ring-buffer-safe writer maybe_publish_corrected_pose() uses), NOT scene_graph_->update().
// update() also drives dsr_update_calibration()/dsr_update_affordance() -- CSV writers and DSR node
// churn sized for ~20 Hz -- which must stay on the real, lidar-paced path only.
void SpecificWorker::publish_predicted_tick(std::int64_t imu_ts_ms)
{
    if (shutting_down_.load() || params.PRESERVE_BOOTSTRAP_ROOM || !scene_graph_)
        return;

    std::lock_guard<std::mutex> publish_lock(publish_mutex_);

    if (!last_published_pose_.has_value() || last_published_ts_ms_ <= 0 || imu_ts_ms <= last_published_ts_ms_)
        return;   // nothing real published yet, or this sample predates it

    const float dt_s = static_cast<float>(imu_ts_ms - last_published_ts_ms_) * 1e-3f;
    // Reuses POSE_CLAMP_MAX_DT_S rather than a new config key: same question ("how long a gap is
    // guessing still better than silence for"), same answer. Past it, a stalled lidar feed means the
    // cached twist is itself stale, and ticking anyway would run the extrapolation away unbounded at
    // ~100 Hz -- worse than just not publishing until a real correction arrives.
    if (dt_s <= 0.f || dt_s > params.POSE_CLAMP_MAX_DT_S)
        return;

    const Eigen::Affine2f& base = last_published_pose_.value();
    const float base_th = std::atan2(base.linear()(1, 0), base.linear()(0, 0));
    const float phi = last_pub_rot_ * dt_s;

    // Body-frame twist -> SE(2) chord. Same exponential map as
    // controller_obstacle_tracker.cpp::twist_delta() (this codebase's original, verified-correct
    // reference for this exact composition, 2026-08-04) -- the left Jacobian V turns v*dt into the
    // CHORD of the arc actually swept instead of cutting the corner. +Y forward = adv (last_pub_adv_),
    // +X lateral = side (last_pub_side_); see the frame note in room_scene_graph.cpp.
    const float vx = last_pub_side_ * dt_s;
    const float vy = last_pub_adv_  * dt_s;
    float dx = vx, dy = vy;
    if (std::abs(phi) > 1e-6f)
    {
        const float s = std::sin(phi), c = std::cos(phi);
        dx = (s * vx - (1.f - c) * vy) / phi;
        dy = ((1.f - c) * vx + s * vy) / phi;
    }

    Eigen::Affine2f predicted = Eigen::Affine2f::Identity();
    predicted.linear()      = Eigen::Rotation2Df(base_th + phi).toRotationMatrix();
    predicted.translation() = base.translation() + base.linear() * Eigen::Vector2f(dx, dy);

    // last_published_cov_ reused as-is (not grown with dt_s) -- a known simplification. A consumer
    // reading this between two real corrections currently sees the same sigma the last real one
    // reported, not an inflated one reflecting the extra dead-reckoned distance since. Revisit if a
    // consumer needs "how much am I trusting this specific sample" rather than just the freshest pose.
    scene_graph_->dsr_publish_predicted_pose(predicted, last_published_cov_,
                                             static_cast<std::uint64_t>(imu_ts_ms));
    log_pose_trace(/*type=predicted*/1, imu_ts_ms, predicted, 0.f);
}

// Localiser pose beside the Webots supervisor pose, one row per published correction.
// robot_concept writes robot_gt_* onto the robot node ONLY while the producer reports
// simulated==true, so on real hardware the attributes are absent and this writes nothing and opens
// no file. Absence is the gate; there is no switch to misconfigure.
//
// Both poses are logged RAW, in their own frames — GT in world, the estimate in the room frame.
// They are deliberately NOT differenced here: the room frame's orientation is arbitrary, so a
// constant offset is expected and only its VARIATION is a defect. Fit offset+gain across the file
// and read the residual; a single pair of readings cannot tell those apart.
// Which convention is right is an EMPIRICAL question with a decisive answer: the correct one leaves
// a CONSTANT offset (the room frame's arbitrary orientation datum), the wrong one leaves an offset
// that swings with the robot's heading. Score both by circular concentration R = |mean unit vector|;
// R -> 1 is constant, R -> 0 is uniformly spread. Reported once, on the first few hundred samples.
// ── STAGE 2: pair each RGB triple point with the LiDAR corner of the SAME polygon vertex ─────────
// See mount_lidar_pair.h for why the residual has no pose in it and why that is the point.
void SpecificWorker::push_mount_correction(rc::CameraIngestor &ing, const Eigen::Vector4d &applied,
                                           const std::string &cam, const char *why)
{
    // Prior-sigma units -> radians and metres, on the camera's own axes (see set_mount_correction).
    const float pitch  = static_cast<float>(applied(0)) * params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    const float height = static_cast<float>(applied(1)) * params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    const float yaw    = static_cast<float>(applied(2)) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    if (std::abs(pitch) + std::abs(height) + std::abs(yaw) <= 0.f) return;
    ing.set_mount_correction(pitch, height, yaw);
    qInfo().nospace().noquote()
        << "[camcal] " << QString::fromStdString(cam) << " mount correction " << why << ": pitch "
        << QString::number(pitch * 180.0 / M_PI, 'f', 4) << " deg, height "
        << QString::number(height, 'f', 4) << " m, yaw "
        << QString::number(yaw * 180.0 / M_PI, 'f', 4) << " deg  (total against the graph extrinsic)";
}

/// Feed the pooled solve back into the camera mount.
///
/// ★ THERE IS NO GATE HERE, AND THAT IS THE DESIGN. The increment applied is the POSTERIOR MEAN,
///   which is already shrunk toward the prior by exactly how much the data informs the axis: the
///   zed's yaw is 98% data and applies nearly all of itself, the ricoh's is 45% and applies 45%,
///   and an axis the data says nothing about applies nothing. A threshold on "informed enough"
///   would be a second, cruder copy of a judgement the posterior has already made.
/// ★ Accum::apply_correction moves the evidence AND the prior's anchor together, so the total is
///   always the posterior mean of the error relative to the graph extrinsic and cannot ratchet.
/// ⚠ It REFUSES unless the solve was marginalised. An estimate taken with the per-vertex nuisance
///   off has absorbed each corner's own detection bias, and writing that into an extrinsic would
///   put a detector's error into the robot's geometry — silently, and with a tightening sigma to
///   go with it. That refusal, not a magnitude limit, is what makes this safe to leave running.
void SpecificWorker::apply_mount_solve(rc::camcal::Estimator &pool, rc::CameraIngestor &ing,
                                       const rc::mount::Accum::Solution &sol, const std::string &cam)
{
    if (not params.IMAGE_EDGE_MOUNT_APPLY or not sol.ok) return;
    if (not sol.marginalised)
    {
        if (not mount_apply_refused_logged_)
        {
            mount_apply_refused_logged_ = true;
            qWarning() << "[camcal] mountApply is on but the solve is NOT marginalised — set"
                       << "ImageEdge.mountVertexOffsetSigmaPx > 0, or delete evidence that carries"
                       << "no per-vertex partials. Refusing to write an unmarginalised estimate"
                       << "into the extrinsic.";
        }
        return;
    }
    // ⚠ THE SIGN. `Solution::p` is `-x`, and every place that reports it negates it again, so the
    //   correction that cancels the reported error is `-p` and not `p`. Applying `+p` runs the loop
    //   away linearly instead of converging (measured: -53 degrees in 12 cycles on a 1 degree truth).
    Eigen::Vector4d dp = -sol.p;
    dp(3) = 0.0;                       // the dt column is not a mount parameter and has no observation
    if (not dp.allFinite() or dp.isZero()) return;
    pool.apply_correction(dp);         // evidence first: the ingestor must never lead the evidence
    push_mount_correction(ing, pool.applied(), cam, "applied");
    // The shared extrinsic LAST, and only ever mirroring what the local correction already is.
    publish_mount_to_graph(pool, ing, cam);
}

/// Decide what the ingestor's base must be: the nominal the evidence was measured against, or the
/// extrinsic just read from the graph.
///
/// ★★★ THIS IS WHAT MAKES PUBLISHING AND RESUMING BOTH CORRECT. `applied` is a total relative to a
///     nominal. Binding takes the base from the graph, which is right only until something writes
///     that edge — and mountPublish writes exactly it. Without this the next restart binds to
///     `nominal ⊕ applied`, pushes `applied` again, and the correction lands TWICE with the prior
///     anchored at the doubled total: one correction per restart, which is the ratchet.
/// ★ The stored nominal WINS over the graph, and that is the point. It is also checked rather than
///   trusted: the graph should read either the nominal (nothing published, or robot_concept reseeded
///   the robot's JSON over it) or `nominal ⊕ applied` (our publish still standing). Anything else came
///   from OUTSIDE this loop.
/// ⚠ An external nominal is ADOPTED and the correction it absorbs is dropped; our own published value
///   never is. Re-centring the prior on the estimator's own last answer removes the prior's pull, and
///   publishes happen every window — that asymmetry is the whole reason this function is not two
///   lines.
void SpecificWorker::reconcile_mount_nominal(rc::camcal::Estimator &pool, rc::CameraIngestor &ing,
                                             const std::string &cam)
{
    const Eigen::Matrix3f graph_R = ing.base_R();       // as just bound, i.e. what the graph says
    const Eigen::Vector3f graph_t = ing.base_t();
    const Eigen::Vector4d ap = pool.applied();
    const bool has_corr = ap.head<3>().cwiseAbs().maxCoeff() > 1e-12;

    if (not pool.have_base())
    {
        // First run on this camera, or a pre-format-3 file. The graph is the only nominal there is.
        if (has_corr and params.IMAGE_EDGE_MOUNT_PUBLISH)
        {
            // ⚠ AMBIGUOUS, AND IT MUST NOT BE GUESSED. The file carries a correction, publishing is
            //   on, and nothing records whether the graph already contains it. Applying it again
            //   doubles the mount; not applying it discards a measurement. So the correction is
            //   dropped and the graph adopted: the evidence (H, b) is KEPT, only its anchor moves,
            //   and the next save records the nominal so this cannot recur.
            pool.adopt_external_nominal(graph_R, graph_t, params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
            qWarning().noquote() << QString::asprintf(
                "[camcal] %s: evidence predates the stored nominal (format < 3) and carries a"
                " correction of pitch %+.4f / height %+.4f / yaw %+.4f prior sigmas while mountPublish"
                " is ON. Nothing records whether the graph already holds it, so the correction is"
                " DROPPED and the current extrinsic adopted as the nominal. H and b are kept; the"
                " total re-converges from here and the nominal is saved from now on.",
                cam.c_str(), ap(0), ap(1), ap(2));
        }
        else
            pool.set_base(graph_R, graph_t);
        return;
    }

    // What the graph would read under each hypothesis.
    const Eigen::Matrix3f nom_R = pool.base_R();
    const Eigen::Vector3f nom_t = pool.base_t();
    const float d_nominal = (graph_R - nom_R).cwiseAbs().maxCoeff()
                          + (graph_t - nom_t).cwiseAbs().maxCoeff();
    // `nominal ⊕ applied`, built the way the ingestor builds it (rebuild_extrinsic_).
    const float pitch = static_cast<float>(ap(0)) * params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    const float height = static_cast<float>(ap(1)) * params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    const float yaw = static_cast<float>(ap(2)) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    const Eigen::Matrix3f Rc = (Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitX())
                              * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())).toRotationMatrix();
    const Eigen::Matrix3f pub_R = Rc * nom_R;
    const Eigen::Vector3f pub_t = Rc * nom_t + height * Eigen::Vector3f::UnitZ();
    const float d_published = (graph_R - pub_R).cwiseAbs().maxCoeff()
                            + (graph_t - pub_t).cwiseAbs().maxCoeff();

    constexpr float kTol = 1e-4f;      // 1e-4 rad is 0.006 deg, far below any correction here
    if (d_nominal < kTol or d_published < kTol)
    {
        // Either the graph still holds the nominal or it holds our own published total. Both mean the
        // stored nominal is the mount this evidence describes, so it becomes the base and the
        // correction is pushed on top exactly as it was measured.
        ing.set_base(nom_R, nom_t);
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s: base taken from the EVIDENCE's nominal, not the graph (graph %s;"
            " |graph-nominal| %.2e, |graph-published| %.2e). This is what stops the resumed"
            " correction being applied twice.", cam.c_str(),
            d_published < kTol ? "still holds our published total" : "holds the nominal",
            static_cast<double>(d_nominal), static_cast<double>(d_published));
        return;
    }
    // Neither: the extrinsic changed from outside this loop — the robot's JSON reseeded with a mount
    // that now carries a measurement, or someone edited it. Adopt it, and drop the correction it
    // absorbs rather than stacking ours on top of theirs.
    Eigen::Vector3f delta = Eigen::Vector3f::Zero();
    float res = 0.f;
    const bool rebased = pool.adopt_external_nominal(
        graph_R, graph_t, params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA, params.IMAGE_EDGE_MOUNT_YAW_SIGMA, &delta, &res);
    qWarning().noquote() << QString::asprintf(
        "[camcal] %s: the extrinsic changed OUTSIDE this loop (|graph-nominal| %.2e,"
        " |graph-published| %.2e, both over %.0e). Adopting it as the new nominal. The change is"
        " pitch %+.4f deg / height %+.4f m / yaw %+.4f deg with a family residual of %.2e, so the"
        " evidence was %s.",
        cam.c_str(), static_cast<double>(d_nominal), static_cast<double>(d_published),
        static_cast<double>(kTol), delta.x() * 180.0 / M_PI, delta.y(), delta.z() * 180.0 / M_PI,
        static_cast<double>(res),
        rebased ? "RE-ANCHORED onto it — b is untouched because the MOUNT did not move, only the point it is measured from, and no information is lost"
                : "RESET: the change is not expressible as a mount correction, so evidence measured"
                  " against the old mount is not evidence about this one");
}

/// Mirror the measured mount into the SHARED body->camera RT edge, so retina, the controller and
/// every other consumer read the corrected extrinsic instead of the nominal one. Until this existed
/// the loop corrected only room_concept's own copy, which was deliberate containment while the loop
/// was young and never the end state.
///
/// ★★★ THIS IS AN OUTPUT AND NOTHING BUT AN OUTPUT, AND THAT IS THE WHOLE SAFETY ARGUMENT. The
///     estimator keeps measuring against the extrinsic it read at bind: CameraIngestor freezes that
///     base, Accum's prior stays anchored on it, and neither is re-read after a write. So publishing
///     cannot move a single number the loop produces — it only tells the fleet what the loop already
///     decided. Re-reading the edge after writing it would make the graph both the input and the
///     output of one estimator and re-centre the prior on its own last answer, which is exactly the
///     ratchet the feedback path was designed to avoid (a weakly informed axis then walks away one
///     honest step at a time).
/// ⚠⚠ ACROSS A RESTART IT STOPS BEING ONLY AN OUTPUT, and that is the open end of this feature. A
///     new process binds to whatever the edge now holds, so today's published value becomes the next
///     session's prior anchor and its nominal. Within a session the loop is stationary; across
///     sessions it can random-walk by its own posterior sigma, because nothing outside the loop
///     still holds the mount's independently measured value. THE CURE IS NOT HERE: the nominal
///     belongs in the robot's JSON — the description of the physical mount — so that the prior is
///     anchored on a measurement the loop never writes. See ImageEdge.mountPublish in etc/config.toml
///     for the persistence step that closes it, which is NOT BUILT.
/// ★ Composes from the NOMINAL captured on the first write, never from the edge's current value: the
///   edge is always `nominal x correction`, so a second window cannot compound the first one's write.
/// ★ Refuses an unmarginalised solve for the same reason apply_mount_solve does, and refuses if
///   composing the edge does not reproduce cortex's own chain — a convention this file believed
///   rather than checked would land a pitch on a roll and still look plausible.
void SpecificWorker::publish_mount_to_graph(rc::camcal::Estimator &pool,
                                            const rc::CameraIngestor &ing, const std::string &cam)
{
    if (not params.IMAGE_EDGE_MOUNT_PUBLISH) return;
    const Eigen::Vector3f corr = ing.mount_correction();          // pitch rad, height m, yaw rad
    if (not corr.allFinite()) return;
    MountPublishState &st = mount_publish_[cam];
    // Nothing moved since the last write. An RT write is a graph signal to every peer, so a window
    // that changed the mount by less than a microradian must not cost one.
    if (st.have_last and (corr - st.last).cwiseAbs().maxCoeff() < 1e-6f) return;

    auto cn = G->get_node(cam);
    if (not cn.has_value()) return;                               // ALWAYS check (CLAUDE.md)
    const auto pid = G->get_attrib_by_name<parent_att>(cn.value());
    if (not pid.has_value()) return;
    auto pn = G->get_node(pid.value());
    if (not pn.has_value()) return;
    const auto e = G->get_edge(pn->id(), cn->id(), "RT");
    if (not e.has_value()) return;
    const auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
    const auto tr  = G->get_attrib_by_name<rt_translation_att>(e.value());
    if (not rot.has_value() or not tr.has_value()
        or rot.value().get().size() < 3 or tr.value().get().size() < 3) return;

    // Rx * Ry * Rz, the composition the whole fleet uses for rt_rotation_euler_xyz.
    const auto euler_to_R = [](const Eigen::Vector3f &r) -> Eigen::Matrix3f
    {
        return (Eigen::AngleAxisf(r.x(), Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(r.y(), Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(r.z(), Eigen::Vector3f::UnitZ())).toRotationMatrix();
    };

    if (not st.have_nominal and pool.have_edge_nominal())
    {
        // ★ FROM THE EVIDENCE, not from the edge. After a room_concept restart the edge holds LAST
        //   session's published total, so capturing it here would compose this session's correction
        //   on top of it and compound the publish once per restart. The nominal travels with the
        //   evidence for the same reason the base does.
        st.nominal_t = pool.edge_nominal_t();
        st.nominal_r = pool.edge_nominal_r();
        st.have_nominal = true;
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s: publish nominal restored from the evidence, t [%+.4f %+.4f %+.4f]"
            " euler [%+.5f %+.5f %+.5f] — the edge's current value is NOT used as a base.",
            cam.c_str(), st.nominal_t.x(), st.nominal_t.y(), st.nominal_t.z(),
            st.nominal_r.x(), st.nominal_r.y(), st.nominal_r.z());
    }
    if (not st.have_nominal)
    {
        st.nominal_t = Eigen::Vector3f(tr.value().get()[0],  tr.value().get()[1],  tr.value().get()[2]);
        st.nominal_r = Eigen::Vector3f(rot.value().get()[0], rot.value().get()[1], rot.value().get()[2]);
        // Persist it with the evidence so the next session composes from the same place.
        pool.set_edge_nominal(st.nominal_t, st.nominal_r);
        st.have_nominal = true;
    }
    // ★ THE CONVENTION CHECK RUNS ONCE PER CAMERA AND INDEPENDENTLY OF WHERE THE NOMINAL CAME FROM.
    //   It used to sit inside the capture branch, so a nominal restored from the evidence skipped it
    //   — i.e. it would only ever run on a camera's first-ever session, which is the one run where a
    //   convention error is least likely to have been introduced since.
    if (not st.checked)
    {
        st.checked = true;
        // ── The convention, checked against cortex's own chain instead of asserted ───────────────
        // ★ MAIN THREAD ONLY (ts == 0 touches InnerEigenAPI's unlocked cache — CLAUDE.md). Both call
        //   sites are in compute()'s pump; if that ever changes, skip the check rather than corrupt
        //   the cache, and say the check was skipped rather than let silence imply it passed.
        if (QThread::currentThread() == QCoreApplication::instance()->thread())
        {
            if (auto inner = G->get_inner_eigen_api())
            {
                const auto chain = inner->get_transformation_matrix(
                    cam, pn->name(), 0, "RT", DSR::RT_API::TimeQuery::Nearest);
                if (chain.has_value())                            // ALWAYS check the optional
                {
                    Eigen::Matrix4f parent_T_cam = Eigen::Matrix4f::Identity();
                    parent_T_cam.block<3, 3>(0, 0) = euler_to_R(st.nominal_r);
                    parent_T_cam.block<3, 1>(0, 3) = st.nominal_t;
                    const Eigen::Matrix4f prod =
                        chain.value().matrix().cast<float>() * parent_T_cam;
                    const float err = (prod - Eigen::Matrix4f::Identity()).cwiseAbs().maxCoeff();
                    if (err > 1e-3f)
                    {
                        if (not mount_publish_refused_logged_)
                        {
                            mount_publish_refused_logged_ = true;
                            qWarning().noquote() << QString::asprintf(
                                "[camcal] mountPublish REFUSES on %s: composing the body->camera edge "
                                "as Rx*Ry*Rz does not reproduce cortex's own chain (max residual "
                                "%.2e). A mount written under the wrong euler convention puts a pitch "
                                "on a roll and looks plausible. Fix the composition, do not relax "
                                "this.", cam.c_str(), static_cast<double>(err));
                        }
                        return;
                    }
                }
            }
        }
        else
            qInfo() << "[camcal] mountPublish: euler-convention check SKIPPED for"
                    << QString::fromStdString(cam) << "— not on the main thread, so the ts==0 chain"
                    << "query is unsafe here (CLAUDE.md). The write below is unverified.";
    }

    // ── The correction, as the ingestor applies it ───────────────────────────────────────────────
    // rebuild_extrinsic_(), verbatim: p_cam' = Rx(pitch)*Rz(yaw) * p_cam + height * z_cam. So the
    // camera<-parent transform is premultiplied by A, and parent->camera is postmultiplied by A^-1.
    // The correction lives entirely in the CAMERA frame, so it localises to this one link whatever
    // sits above it in the tree — which is why nothing here needs to know about Shadow->body.
    Eigen::Matrix4f A = Eigen::Matrix4f::Identity();
    A.block<3, 3>(0, 0) = (Eigen::AngleAxisf(corr.x(), Eigen::Vector3f::UnitX())
                         * Eigen::AngleAxisf(corr.z(), Eigen::Vector3f::UnitZ())).toRotationMatrix();
    A.block<3, 1>(0, 3) = corr.y() * Eigen::Vector3f::UnitZ();
    Eigen::Matrix4f T_nom = Eigen::Matrix4f::Identity();
    T_nom.block<3, 3>(0, 0) = euler_to_R(st.nominal_r);
    T_nom.block<3, 1>(0, 3) = st.nominal_t;
    const Eigen::Matrix4f T_new = T_nom * A.inverse();
    const Eigen::Vector3f t_new = T_new.block<3, 1>(0, 3);
    const Eigen::Matrix3f R_new = T_new.block<3, 3>(0, 0);
    // ★ NOT Eigen::eulerAngles(0,1,2). It constrains the MIDDLE angle to [0,pi], and both of this
    //   robot's cameras have a nominal Y of about -2e-5 rad, so it returns the other valid
    //   decomposition — roughly (pi, +y, pi). That composes to the same rotation and would pass any
    //   round-trip check, while writing a triple in which r[2] is no longer readable as a yaw. Every
    //   consumer that reads a component by index — robot_concept's boresight write does — would then
    //   be reading nonsense from a mathematically correct edge. Closed form instead, valid for
    //   |pitch| < pi/2, which a camera mount is:
    //     R = Rx(a)Ry(b)Rz(c)  =>  b = asin(R02), a = atan2(-R12, R22), c = atan2(-R01, R00)
    const Eigen::Vector3f r_new(std::atan2(-R_new(1, 2), R_new(2, 2)),
                                std::asin(std::clamp(R_new(0, 2), -1.f, 1.f)),
                                std::atan2(-R_new(0, 1), R_new(0, 0)));
    // Round-trip the extraction: a write whose own decomposition does not reproduce the matrix is a
    // different mount, not a rounding. (Measured on both cameras' real values: 4.7e-07.)
    if ((euler_to_R(r_new) - R_new).cwiseAbs().maxCoeff() > 1e-5f)
    {
        qWarning() << "[camcal] mountPublish: euler decomposition did not round-trip for"
                   << QString::fromStdString(cam) << "— refusing to write.";
        return;
    }
    if (not t_new.allFinite() or not r_new.allFinite()) return;

    if (auto rt = G->get_rt_api())
    {
        // ★ STATIC, NOT TIMESTAMPED. robot_concept's reasoning applies unchanged: this is a fixed
        //   physical mount whose ESTIMATE is being refined, so the newest value is the best answer
        //   for every frame including older ones — and a timestamped write would stamp the blocks
        //   with the moment of calibration, putting every later query through body->camera outside
        //   the ring for ever and reporting a clamp it did not earn.
        rt->insert_or_assign_edge_RT_static(pn.value(), cn->id(),
            std::vector<float>{t_new.x(), t_new.y(), t_new.z()},
            std::vector<float>{r_new.x(), r_new.y(), r_new.z()});
        st.last = corr;
        st.have_last = true;
        qInfo().noquote() << QString::asprintf(
            "[camcal] %s mount PUBLISHED into %s->%s: pitch %+.4f deg, height %+.4f m, yaw %+.4f deg "
            "on top of the nominal — t [%+.4f %+.4f %+.4f] euler [%+.5f %+.5f %+.5f]. Every agent "
            "reads this now.", cam.c_str(), pn->name().c_str(), cam.c_str(),
            corr.x() * 180.0 / M_PI, corr.y(), corr.z() * 180.0 / M_PI,
            t_new.x(), t_new.y(), t_new.z(), r_new.x(), r_new.y(), r_new.z());
    }
}

void SpecificWorker::open_pair_log(std::ofstream &csv, const std::string &cam,
                                   const rc::CameraIngestor &ing)
{
    // ★ KEYED BY CAMERA, like the evidence file beside it (camera_calib_<robot>_<camera>.txt).
    //   A fixed filename let a second camera's run destroy the first one's rows; the estimator's
    //   own evidence was already keyed, only this diagnostic was not.
    csv.open("etc/image_edge_pair_" + cam + ".csv", std::ios::out | std::ios::trunc);
    if (not csv.is_open()) return;
    csv.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
    // ★ FULL float precision, for the reason camera_calibration.h::save already gives about the
    //   evidence file — and this file needed it MORE, not less. The solve weights by cov^-1, and a
    //   near-singular 2x2 amplifies input error by its condition number. MEASURED on the 09-03 tour
    //   at the default 6 significant figures: typical cond 17 (harmless), but 259 of 91152 rows above
    //   1e6 and 9 rows that came back NOT positive-definite once rounded — those were dropped by a
    //   replay though they counted live, leaving H off by 6% on the yaw-height cross term while b and
    //   rTr agreed to 1e-2. 9 digits round-trips a float exactly, so the delta=0 replay can be exact
    //   rather than approximately right.
    csv << std::setprecision(std::numeric_limits<float>::max_digits10);
    csv << "ts_ms,camera,vertex,"
        // WHICH corner of the vertical edge: the loop closure keys on vertex*2 + ceiling, so a
        // replay without this column would difference a floor corner against a ceiling one and
        // recompute a closure the agent never measured.
           "ceiling,"
           "u_img,v_img,u_lidar,v_lidar,ru,rv,"
        // sigu/sigv are the DIAGONAL of the pair covariance; cuv is its off-diagonal. All three,
        // because the solve weights by the full 2x2 inverse: a replay handed only the diagonal
        // would compute a different H from the same rows, and then a bug in the replay could not be
        // told apart from that difference. With cuv the zero-injection replay must reproduce the
        // live solve exactly, which is the only self-check the replay has.
           "sigu,sigv,cuv,assoc_prob,range_m,angle_deg,assoc_chi2,"
        // ── the association's INPUTS, beside its verdict ──────────────────────────────────────
        // assoc_chi2 is TRUNCATED to [0, CornerDetector::Params::assoc_chi2] by the gate itself,
        // so its distribution cannot be used to judge the gate. These two can: n_rivals is how
        // many model corners were in gate for this detection (0 = no choice to get wrong), and
        // runnerup_chi2 is how far away the best loser sat. The MARGIN runnerup_chi2/assoc_chi2 is
        // the correspondence's real confidence — a match is trustworthy when the second-best
        // candidate is FAR, not when the best one is CLOSE.
           "n_rivals,runnerup_chi2,"
        // The LiDAR corner in the ROBOT frame — what uv_lidar was computed FROM. With it (and the
        // sidecar's mount) an extrinsic injection is a replay of this file rather than another
        // 200 m of driving (VALIDATION_THREE_DEVICE_CORNERS §2b, arm 7).
           "px_robot,py_robot,pz_robot,"
        // The self-calibration correction in force WHEN THIS ROW WAS WRITTEN. With mountApply on the
        // mount moves during a run, so the sidecar's single extrinsic describes only the moment the
        // file opened; a replay that recomputed uv_lidar from it would be reconstructing a mount the
        // later rows were never measured against. Per row, because that is where the truth is.
           "corr_pitch,corr_height,corr_yaw,"
        // The LIDAR HALF of the covariance, on its own. The other half (the image corner's) is the
        // remainder. A replay needs the split because the LiDAR half is the part that moves with the
        // mount: with it the injected weighting is rebuilt exactly instead of held fixed and
        // apologised for.
           "cl_uu,cl_uv,cl_vv\n";

    rc::mount::ReplayContext rc_ctx;
    rc_ctx.robot        = params.LIDAR_ROBOT_FRAME;
    rc_ctx.camera       = cam;
    rc_ctx.cam          = ing.model();
    rc_ctx.cam_R_robot  = ing.cam_R_robot();
    rc_ctx.cam_t_robot  = ing.cam_t_robot();
    rc_ctx.sigma_pitch  = params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    rc_ctx.sigma_height = params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    rc_ctx.sigma_yaw    = params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    rc_ctx.offset_sigma_px = params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX;
    rc_ctx.applied         = ing.mount_correction();   // cam_R_robot above already includes it
    // The LiDAR's origin in the robot frame: a LiDAR mount error rotates every corner about THAT
    // point, not about the robot origin, and the parallax that leaves is precisely what decides
    // whether a LiDAR injection really cancels in the camera-vs-camera closure. If the chain does
    // not resolve, the flag stays false and the replay REFUSES that leg rather than assuming zero.
    if (auto inner = G->get_inner_eigen_api())
        if (const auto T = inner->get_transformation_matrix(params.LIDAR_ROBOT_FRAME,
                                                            params.LIDAR_HELIOS_NAME, 0, "RT",
                                                            DSR::RT_API::TimeQuery::Nearest);
            T.has_value())   // ALWAYS check the optional (CLAUDE.md)
        {
            rc_ctx.lidar_t_robot = T.value().matrix().block<3, 1>(0, 3).cast<float>();
            rc_ctx.lidar_known   = true;
        }
    const std::string side = "etc/image_edge_replay_" + cam + ".txt";
    if (not rc::mount::write_replay_context(side, rc_ctx))
        qWarning() << "[camcal] could not write" << QString::fromStdString(side)
                   << "— the pair rows will not be replayable offline";
    else if (not rc_ctx.lidar_known)
        qWarning() << "[camcal]" << QString::fromStdString(params.LIDAR_HELIOS_NAME) << "<-"
                   << QString::fromStdString(params.LIDAR_ROBOT_FRAME)
                   << "did not resolve; the LiDAR-injection leg of arm 7 will refuse to run";
}

void SpecificWorker::write_pair_row(std::ofstream &csv, const std::string &cam, std::int64_t ts,
                                    const rc::mount::PairObs &pr, bool ceiling, float angle_deg,
                                    float assoc_chi2, int n_rivals, float runnerup_chi2,
                                    const Eigen::Vector3f &corr)
{
    if (not csv.is_open()) return;
    csv << ts << ',' << cam << ',' << pr.vertex << ',' << (ceiling ? 1 : 0) << ','
        << pr.uv_image.x() << ',' << pr.uv_image.y() << ','
        << pr.uv_lidar.x() << ',' << pr.uv_lidar.y() << ','
        << pr.r.x() << ',' << pr.r.y() << ','
        << std::sqrt(std::max(0.f, pr.cov(0, 0))) << ','
        << std::sqrt(std::max(0.f, pr.cov(1, 1))) << ','
        << pr.cov(0, 1) << ','
        << pr.assoc_prob << ',' << pr.range_m << ','
        << angle_deg << ',' << assoc_chi2 << ','
        << n_rivals << ','
        // A match with no rival has an INFINITE margin, not a huge finite one. Writing the 1e9
        // sentinel would put a number into an average that means "no rival".
        << (n_rivals > 0 ? runnerup_chi2 : -1.f) << ','
        << pr.p_robot.x() << ',' << pr.p_robot.y() << ',' << pr.p_robot.z() << ','
        << corr.x() << ',' << corr.y() << ',' << corr.z() << ','
        << pr.cov_lidar(0, 0) << ',' << pr.cov_lidar(0, 1) << ',' << pr.cov_lidar(1, 1) << '\n';
}

void SpecificWorker::mount_pair_update(const rc::ImageEdgeObs &obs,
                                       const std::vector<rc::CornerDetector::CornerMatch> &matches,
                                       std::int64_t timestamp_ms)
{
    // The driving camera. Auxiliary channels call feed_calib_pairs() with their own name and model.

    if (obs.triple_points.empty() or matches.empty() or not camera_ingestor_) return;
    if (mp_win_start_ms_ == 0) mp_win_start_ms_ = timestamp_ms;
    if (not mp_loaded_)
    {
        // Resume the measurement from the last session. Loading H and b is not a ratchet: the prior
        // is re-added at solve time, so this is the same evidence arriving earlier.
        mp_loaded_ = true;
        // ★ ONE FILE PER CAMERA. A single etc/camera_calib.txt accumulated 523844 pairs across a
        //   zed -> ricoh switch on 2026-08-29 before this was noticed: two different mounts summed
        //   into one information matrix, which estimates neither. The name keys the file AND is
        //   checked inside it, so a copy or a rename is caught too.
        mp_pool_.set_camera(params.IMAGE_EDGE_CAMERA, params.LIDAR_ROBOT_FRAME);
        // The per-vertex offset nuisance, on the pool AND on the per-window accumulator, so the two
        // columns on the same log line are produced by the same model. Set BEFORE load(), which
        // preserves it. 0 = off = the pre-2026-09-02 solve exactly.
        const double vox = params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX;
        mp_pool_.set_vertex_offset_sigma_px(vox);
        mp_win_.offset_sigma_px = vox;
        if (vox > 0.0)
            qInfo().nospace() << "[camcal] per-vertex offset nuisance ON, prior sigma " << vox
                              << " px — mount sigmas are now cluster-honest and will read LARGER; "
                                 "yaw approaches the between-vertex SEM by construction";
        const std::string path = mp_pool_.path();
        if (const std::size_t k = mp_pool_.load(path); k > 0)
            qInfo().nospace() << "[camcal] resumed from " << QString::fromStdString(path)
                              << " (" << k << " pairs, camera "
                              << QString::fromStdString(params.IMAGE_EDGE_CAMERA) << ")";
        // A correction restored from disk must reach the mount BEFORE the first frame is measured
        // against it, or this session's first window is referenced to an extrinsic the evidence
        // does not describe.
        reconcile_mount_nominal(mp_pool_, *camera_ingestor_, params.IMAGE_EDGE_CAMERA);
        push_mount_correction(*camera_ingestor_, mp_pool_.applied(), params.IMAGE_EDGE_CAMERA, "resumed");
    }

    for (const auto &tp : obs.triple_points)
    {
        ++mp_seen_;
        // EXACT association: both sides index the ORIGINAL polygon vertex list. No gate, no
        // nearest-neighbour, so no misassociation mode to defend against.
        const auto it = std::ranges::find_if(matches,
            [&](const auto &m) { return m.model_index == tp.vertex; });
        if (it == matches.end()) continue;
        // A SUPPRESSED match is a retired landmark: still detected, still carrying numbers, but
        // deliberately kept out of the loss (corner_detector.h). It must stay out of this one too.
        if (it->suppressed) continue;

        const auto pr = rc::mount::make_pair(tp, *it, camera_ingestor_->model(),
                                             camera_ingestor_->cam_R_robot(),
                                             camera_ingestor_->cam_t_robot(),
                                             params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                             params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                             params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
        if (not pr.ok) continue;
        ++mp_paired_;
        mp_win_.add(pr);
        mp_pool_.add(pr);
        // ★ IN RADIANS, not pixels. A pixel is 0.128 deg on the zed and 0.188 on the ricoh, so a
        //   pixel residual cannot be compared across cameras and an angular one can.
        {
            // ★ AT THE MEASUREMENT, not at the principal point. The closure differences the two
            //   cameras' residuals in radians; the panorama's scale is constant but the ZED is a
            //   pinhole, whose true local scale is fx·sec^2(theta). Measured on a synthetic drive of
            //   the real pair, that constant was 82% of the leakage a LiDAR error puts into the
            //   closure — a bias in the comparison, not in either camera (mount_replay --selftest).
            const Eigen::Vector2f ppr = rc::img::px_per_rad_at(obs.cam, pr.uv_image);
            if (ppr.x() > 0.f and ppr.y() > 0.f)
                loop_closure_observe(params.IMAGE_EDGE_CAMERA, tp.vertex,
                                     tp.from == rc::ContourClass::WallCeiling,
                                     static_cast<double>(pr.r.x() / ppr.x()),
                                     static_cast<double>(pr.r.y() / ppr.y()), timestamp_ms);
        }

        if (not mp_csv_.is_open()) open_pair_log(mp_csv_, params.IMAGE_EDGE_CAMERA, *camera_ingestor_);
        write_pair_row(mp_csv_, params.IMAGE_EDGE_CAMERA, timestamp_ms, pr,
                       tp.from == rc::ContourClass::WallCeiling,
                       it->angle_deg, it->assoc_chi2_val, it->n_rivals, it->runnerup_chi2,
                       camera_ingestor_->mount_correction());
    }

    if (timestamp_ms - mp_win_start_ms_ < 5000) return;
    mp_win_start_ms_ = timestamp_ms;
    if (mp_csv_.is_open()) mp_csv_.flush();

    const auto win  = mp_win_.solve();
    const auto pool = mp_pool_.solve();
    apply_mount_solve(mp_pool_, *camera_ingestor_, pool, params.IMAGE_EDGE_CAMERA);
    mp_pool_.save(mp_pool_.path());   // per (robot, camera); a kill -9 costs at most one window
    if (viewer_ and pool.ok)
    {
        Eigen::Matrix<float, rc::camcal::P_COUNT, 1> pv, sv;
        const double psig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
        for (int i = 0; i < rc::camcal::P_COUNT; ++i)
        {
            // The estimator works in units of the PRIOR SIGMA (h carries it), so convert back to
            // physical here — and scale the posterior sigma the same way, or the popup would show a
            // physical value with a dimensionless uncertainty beside it.
            pv(i) = static_cast<float>(pool.p(i)     * psig[i]);
            sv(i) = static_cast<float>(pool.sigma(i) * psig[i]);
        }
        viewer_->set_camera_calibration(pv, sv, pool.informed, static_cast<float>(pool.cond),
                                        mp_pool_.pairs(), params.IMAGE_EDGE_CAMERA);
        if (loop_n_ > 0)
        {
            const double R = 180.0 / M_PI;
            const double mu = loop_du_sum_ / loop_n_, mv = loop_dv_sum_ / loop_n_;
            viewer_->set_loop_closure(
                mu * R, mv * R,
                std::sqrt(std::max(0.0, loop_du_sq_ / loop_n_ - mu * mu)) * R,
                std::sqrt(std::max(0.0, loop_dv_sq_ / loop_n_ - mv * mv)) * R, loop_n_);
        }
    }
    mp_win_.reset();
    if (not win.ok) return;
    ++mp_wins_;
    mp_sum_  += win.p;
    mp_sum2_ += win.p.cwiseProduct(win.p);
    ++mp_sum_n_;

    const double sig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                           params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                           params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
    const char *nm[4] = {"pitch", "height", "yaw", "dt"};
    const auto phys = [&](double v, int i)
    { return (i == 0 or i == 2) ? v * sig[i] * 180.0 / M_PI : v * sig[i]; };

    QString body;
    for (int i = 0; i < 3; ++i)      // dt is not estimated here; do not print an unmoved prior
        body += QString(" | %1 %2 (%3%4)").arg(nm[i])
                    .arg(phys(win.p(i), i), 0, 'f', 4).arg(win.sigma(i), 0, 'f', 3)
                    .arg((win.informed >> i) & 1 ? ", INF" : "");
    qInfo().nospace().noquote()
        << "[mount/pair] window " << mp_wins_ << " (" << win.chi2_dof * 0 + mp_paired_
        << " pairs of " << mp_seen_ << " triple points"
        // ★ THE CLUSTER COUNT IS THE REAL SAMPLE SIZE FOR THE MOUNT'S LEVEL, and printing it beside
        //   the pair count is what stops "395171 pairs" being read as 395171 measurements.
        << ", " << win.clusters << " corners" << (win.marginalised ? ", offset marginalised" : "")
        << ")" << body
        << " | chi2/dof " << QString::number(win.chi2_dof, 'f', 2)
        << " | cond " << QString::number(win.cond, 'f', 1)
        << " (" << nm[win.rho_i] << "/" << nm[win.rho_j] << " rho "
        << QString::number(win.rho, 'f', 3) << ")";

    // ★ The pooled solve is the CLAIM under test: with the pose gone from the residual, windows
    //   should be comparable draws of one static quantity, so summing their information is legitimate
    //   and buys the range diversity one window never has. If the between-window scatter does NOT
    //   collapse toward the pooled sigma, the pose was not the dominant nuisance and pooling is still
    //   wrong — which is why both numbers are on the same line rather than only the flattering one.
    if (pool.ok and mp_sum_n_ >= 2)
    {
        QString pb;
        for (int i = 0; i < 3; ++i)
        {
            const double m = mp_sum_(i) / mp_sum_n_;
            const double sc = std::sqrt(std::max(0.0, mp_sum2_(i) / mp_sum_n_ - m * m));
            pb += QString(" | %1 %2 (%3%4) [scatter %5]").arg(nm[i])
                      .arg(phys(pool.p(i), i), 0, 'f', 4).arg(pool.sigma(i), 0, 'f', 3)
                      .arg((pool.informed >> i) & 1 ? ", INF" : "")
                      .arg(phys(sc, i), 0, 'f', 4);
        }
        qInfo().nospace().noquote()
            << "[mount/pool] " << mp_pool_.pairs() << " pairs over " << mp_wins_ << " windows" << pb
            << " | chi2/dof " << QString::number(pool.chi2_dof, 'f', 2)
            << " | cond " << QString::number(pool.cond, 'f', 1)
            << " (" << nm[pool.rho_i] << "/" << nm[pool.rho_j] << " rho "
            << QString::number(pool.rho, 'f', 3) << ")"
            << (pool.cond < 50.0 && win.cond > 50.0
                    ? "   <- POOLING BROKE THE RIDGE: the pair is separable across poses and was not"
                      " within one window" : "");
    }
}

// One camera's angular disagreement with the LiDAR, for one corner. When a SECOND camera reports the
// same corner close enough in time, the two are differenced: the LiDAR corner's own error is common
// to both and cancels, leaving camera-vs-camera.
void SpecificWorker::loop_closure_observe(const std::string &cam, int vertex, bool ceiling,
                                          double du_rad, double dv_rad, std::int64_t ts)
{
    const int key = vertex * 2 + (ceiling ? 1 : 0);
    for (auto &[k, v] : loop_last_)
    {
        if (k.second != key or k.first == cam) continue;
        // ★ NEAR-SIMULTANEOUS ONLY. The two cameras free-run at different rates, and differencing
        //   observations 200 ms apart would fold the robot's own motion into what is meant to be a
        //   static mount comparison. 60 ms is about one frame at these rates.
        if (std::abs(ts - v.ts) > 60) continue;
        const double ddu = du_rad - v.du_rad, ddv = dv_rad - v.dv_rad;
        loop_du_sum_ += ddu; loop_dv_sum_ += ddv;
        loop_du_sq_  += ddu * ddu; loop_dv_sq_ += ddv * ddv;
        ++loop_n_;
        if (not loop_csv_.is_open())
        {
            loop_csv_.open("etc/camera_loop.csv", std::ios::out | std::ios::trunc);
            if (loop_csv_.is_open())
            {
                loop_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                loop_csv_ << "ts_ms,vertex,ceiling,cam_a,cam_b,"
                             "du_a_deg,dv_a_deg,du_b_deg,dv_b_deg,ddu_deg,ddv_deg\n";
            }
        }
        if (loop_csv_.is_open())
        {
            const double R = 180.0 / M_PI;
            loop_csv_ << ts << ',' << vertex << ',' << (ceiling ? 1 : 0) << ','
                      << cam << ',' << k.first << ','
                      << du_rad * R << ',' << dv_rad * R << ','
                      << v.du_rad * R << ',' << v.dv_rad * R << ','
                      << ddu * R << ',' << ddv * R << '\n';
        }
        if (loop_n_ % 500 == 0)
        {
            const double R = 180.0 / M_PI;
            const double mu = loop_du_sum_ / loop_n_, mv = loop_dv_sum_ / loop_n_;
            const double su = std::sqrt(std::max(0.0, loop_du_sq_ / loop_n_ - mu * mu));
            const double sv = std::sqrt(std::max(0.0, loop_dv_sq_ / loop_n_ - mv * mv));
            qInfo().nospace().noquote()
                << "[loop] " << loop_n_ << " shared corners | camera-vs-camera du "
                << QString::number(mu * R, 'f', 4) << " +/- " << QString::number(su * R, 'f', 4)
                << " deg, dv " << QString::number(mv * R, 'f', 4) << " +/- "
                << QString::number(sv * R, 'f', 4) << " deg"
                << "   <- the LiDAR corner's own error CANCELS here; what is left is the two"
                   " cameras disagreeing with each other";
        }
        break;
    }
    loop_last_[{cam, key}] = CornerAngle{du_rad, dv_rad, ts};
}

void SpecificWorker::gt_convention_report(float est_th, float gt_th_raw)
{
    const double d = est_th - gt_th_raw, u = est_th + gt_th_raw;
    gt_sum_diff_c_ += std::cos(d); gt_sum_diff_s_ += std::sin(d);
    gt_sum_sum_c_  += std::cos(u); gt_sum_sum_s_  += std::sin(u);
    if (++gt_n_ != gt_report_at_) return;
    gt_report_at_ *= 10;                                   // 200, 2000, 20000 -- three checks, then quiet
    const double n = static_cast<double>(gt_n_);
    const double Rd = std::hypot(gt_sum_diff_c_, gt_sum_diff_s_) / n;
    const double Ru = std::hypot(gt_sum_sum_c_,  gt_sum_sum_s_)  / n;
    const double od = std::atan2(gt_sum_diff_s_, gt_sum_diff_c_) * 180.0 / M_PI;
    const double ou = std::atan2(gt_sum_sum_s_,  gt_sum_sum_c_)  * 180.0 / M_PI;
    qInfo().nospace().noquote()
        << "[gt] convention check over " << gt_n_ << " samples: "
        << "est-gt R=" << QString::number(Rd, 'f', 4) << " (offset " << QString::number(od, 'f', 2)
        << " deg) | est+gt R=" << QString::number(Ru, 'f', 4) << " (offset "
        << QString::number(ou, 'f', 2) << " deg)  ->  "
        << (Ru > Rd ? "producer sign is INVERTED, local negation is CORRECT"
                    : "producer sign looks RIGHT -- robot_concept may have been fixed; REMOVE the "
                      "local negation in log_ground_truth or every heading comparison inverts");
}

void SpecificWorker::apply_base_capability_to_pose_clamp()
{
    if (pose_clamp_from_capability_ or shutting_down_.load() or not G) return;
    const auto robots = G->get_nodes_by_type("robot");
    if (robots.empty()) return;                       // not synced yet: try again next cycle
    pose_clamp_from_capability_ = true;               // one shot, whatever it finds

    const auto cap = rc::read_base_capability(*G, robots.front().id());
    if (not cap.any())
    {
        std::cout << "[pose-clamp] the robot node publishes no base capability; keeping the configured "
                     "fallback (v " << params.POSE_CLAMP_V_MAX << " m/s, w " << params.POSE_CLAMP_W_MAX
                  << " rad/s). Producer: robot_concept + Agent.base_config_file." << std::endl;
        return;
    }
    // Absent stays absent: a field the base did not state leaves the fallback in force. Zero would
    // pin every correction to zero, which is not a conservative reading of "unknown".
    const float v_old = params.POSE_CLAMP_V_MAX, w_old = params.POSE_CLAMP_W_MAX;
    if (cap.max_linear_speed_mps) params.POSE_CLAMP_V_MAX = *cap.max_linear_speed_mps;
    if (cap.max_rot_speed_rps)    params.POSE_CLAMP_W_MAX = *cap.max_rot_speed_rps;
    std::cout << "[pose-clamp] bounded by what the ROBOT can do, not by a controller's preference: "
              << "v " << v_old << " -> " << params.POSE_CLAMP_V_MAX << " m/s, "
              << "w " << w_old << " -> " << params.POSE_CLAMP_W_MAX << " rad/s "
              << "(from robot_max_linear_speed / robot_max_rot_speed on '"
              << robots.front().name() << "')." << std::endl;
    if (params.POSE_CLAMP_W_MAX <= w_old)
        std::cout << "[pose-clamp] note: the capability is NOT above the previous value, so this clamp "
                     "was not the thing limiting the published yaw rate." << std::endl;
}

void SpecificWorker::log_ground_truth(const rc::RoomConcept::UpdateResult &res)
{
    if (shutting_down_.load() or not G)
        return;
    const auto robots = G->get_nodes_by_type("robot");
    if (robots.empty())
        return;
    const auto &rn = robots.front();
    const auto gx = G->get_attrib_by_name<robot_gt_x_att>(rn);
    const auto gy = G->get_attrib_by_name<robot_gt_y_att>(rn);
    const auto ga = G->get_attrib_by_name<robot_gt_angle_att>(rn);
    if (not gx.has_value() or not gy.has_value() or not ga.has_value())
        return;                                  // no ground truth -> real robot -> nothing to do

    if (not gt_csv_open_attempted_)
    {
        gt_csv_open_attempted_ = true;
        QDir().mkpath("tmp/sdf_localizer");
        // ★ ARCHIVE THE PREVIOUS RUN BEFORE TRUNCATING. This file is the only record of what the
        //   localiser actually did, and every restart used to erase it. That has already cost real
        //   evidence: an A/B comparison could not be re-checked because the baseline run's data was
        //   gone, and a stale copy of a DIFFERENT file was once analysed as if it were live. A
        //   comparison between two configurations is only possible if both survive.
        {
            const QString cur = "tmp/sdf_localizer/gt_error.csv";
            if (QFileInfo fi(cur); fi.exists() and fi.size() > 0)
            {
                const QString keep = QString("tmp/sdf_localizer/gt_error_%1.csv")
                                         .arg(fi.lastModified().toString("yyyy-MM-dd_HH-mm-ss"));
                if (QFile::rename(cur, keep))
                    qInfo() << "[gt] previous run archived as" << keep;
                else
                    qWarning() << "[gt] could NOT archive the previous run; it is about to be lost";
            }
        }
        gt_csv_.open("tmp/sdf_localizer/gt_error.csv", std::ios::out | std::ios::trunc);
        if (gt_csv_.is_open())
        {
            // Written through the CLASSIC locale: these machines run es_ES, where a comma is the
            // decimal separator, and a CSV whose fields contain commas is unparseable.
            gt_csv_.imbue(std::locale::classic());
            gt_csv_ << "ts_ms,gt_x,gt_y,gt_theta,est_x,est_y,est_theta,gt_theta_raw,"
                       "sdf_mse,iters,cov_tt,"
                       "imu_dtheta,wheel_dtheta,wheel_shadow_dtheta,imu_segs,wheel_segs,"
                       "pred_x,pred_y,pred_theta,dx_local,dy_local,"
                       "calib_k_v,calib_k_w,calib_yaw,calib_eps,calib_carried,calib_dropped,"
                       "calib_sig_kv,calib_sig_kw,calib_sig_yaw,calib_pos_var,"
                       "calib_b_omega,calib_informed,calib_cond,"
                       "imu_dvx,imu_dvy,wheel_dvx,wheel_dvy,imu_dpx,imu_dpy,imu_lin_segs,"
                       // ── FACTOR B of the camera-extrinsic experiment ─────────────────────────
                       // The same window solved twice in the shadow: under the mount as it is, and
                       // under the mount with the self-calibration removed. Both poses RAW and on
                       // the SAME ROW as the ground truth, so M1 is a subtraction here rather than
                       // a join across two files with two clocks. Never pre-differenced: a mean and
                       // its counterfactual travelling as one number is how a mismatch hides.
                       // fb_ts = 0 means the shadow did not produce a pair on this cycle.
                       "fb_ts,fb_cal_x,fb_cal_y,fb_cal_th,fb_nom_x,fb_nom_y,fb_nom_th,"
                       "fb_corr_pitch,fb_corr_height,fb_corr_yaw\n";
        }
        else
            qWarning() << "[gt] cannot open tmp/sdf_localizer/gt_error.csv";
    }
    if (not gt_csv_.is_open())
        return;

    const auto &p = res.robot_pose;
    const float est_th = std::atan2(p.linear()(1, 0), p.linear()(0, 0));
    // See the declaration in specificworker.h: the producer's sign is inverted, so the CSV carries
    // the corrected angle in gt_theta (what every downstream analysis wants) and the untouched value
    // in gt_theta_raw (so nothing is lost and the claim stays checkable from the file alone).
    const float gt_th_raw = ga.value();
    const float gt_th     = -gt_th_raw;
    gt_convention_report(est_th, gt_th_raw);
    // Fetched once and used raw: the pose error each implies is computed from this row offline,
    // because the subtraction is the analysis and not the measurement.
    const auto fb = room_concept_.get_factor_b();
    const std::int64_t fb_ts = fb.valid ? fb.ts_ms : 0;
    gt_csv_ << res.timestamp_ms
            << ',' << gx.value() << ',' << gy.value() << ',' << gt_th
            << ',' << p.translation().x() << ',' << p.translation().y() << ',' << est_th
            << ',' << gt_th_raw
            << ',' << res.sdf_mse << ',' << res.iterations_used
            << ',' << (res.covariance.rows() > 2 ? res.covariance(2, 2) : -1.f)
            // Heading-channel attribution: which sensor produced this cycle's predicted rotation.
            // imu_dtheta+wheel_dtheta is what entered the prior; wheel_shadow_dtheta is what the
            // wheels claimed over the SAME intervals the gyro overrode. With gt_theta alongside,
            // each channel's scale error can be regressed out separately instead of inferred from
            // their pooled effect.
            << ',' << res.imu_dtheta << ',' << res.wheel_dtheta << ',' << res.wheel_shadow_dtheta
            << ',' << res.imu_segs << ',' << res.wheel_segs
            // Pre-optimizer predicted pose: (est - pred) is the correction, i.e. one tooth measured
            // in the room frame with no GT and no frame fit. Equals est exactly when iters==0.
            << ',' << res.pred_x << ',' << res.pred_y << ',' << res.pred_theta
            // What the wheels claimed in the BODY frame: dx_local is lateral. Driving straight it
            // should be ~0; anything else is the predictor being told the robot slid sideways.
            << ',' << res.dx_local << ',' << res.dy_local
            // Learned motion-model parameters, so convergence is visible in the same file as the
            // error they are supposed to remove.
            << ',' << res.calib_k_v << ',' << res.calib_k_w << ',' << res.calib_yaw
            << ',' << res.calib_episodes
            // Spans that reached the trigger with nothing measured in them. See
            // CALIB_UNMEASURED_EPISODES.md: these used to be emitted as zeros.
            << ',' << res.calib_carried << ',' << res.calib_dropped
            // Sigmas + the R actually used: without these a replay of the filter from this file is
            // guesswork, and a parameter that stopped moving cannot be told from one never taught.
            << ',' << res.calib_sigma_k_v << ',' << res.calib_sigma_k_w
            << ',' << res.calib_sigma_yaw << ',' << res.calib_pos_var
            // Joint-solve outputs: the gyro bias it can now separate, which parameters this window
            // actually taught, and how collinear the window was.
            << ',' << res.calib_b_omega << ',' << res.calib_informed << ',' << res.calib_condition
            // Linear IMU channel. imu_dv vs wheel_dv is translation's first independent cross-check
            // logged before being fused, because a channel whose covariance is unknown (the ImuFrame
            // IDL has no acc_var) must be shown to agree with something before anything trusts it.
            << ',' << res.imu_dvx << ',' << res.imu_dvy
            << ',' << res.wheel_dvx << ',' << res.wheel_dvy
            << ',' << res.imu_dpx << ',' << res.imu_dpy << ',' << res.imu_lin_segs
            // ── Factor B: both poses RAW, on this row, never differenced here ────────────────────
            << ',' << fb_ts
            << ',' << fb.pose_calibrated.x() << ',' << fb.pose_calibrated.y() << ',' << fb.pose_calibrated.z()
            << ',' << fb.pose_nominal.x()    << ',' << fb.pose_nominal.y()    << ',' << fb.pose_nominal.z()
            << ',' << fb.correction.x() << ',' << fb.correction.y() << ',' << fb.correction.z()
            << '\n';
    gt_csv_.flush();
}


// ── RGB edge alignment: one extraction per compute() tick ────────────────────────────────────────
//
// Runs on the MAIN thread. That is safe and deliberate: everything it touches is plain data (a
// grayscale vector, the reduced CameraModel, the cached static extrinsic, the room polygon), so it
// makes no DSR call at all — in particular no room<-robot lookup, which is the circularity trap the
// whole design is built to avoid (see image_edge_source.h). The only graph access in this subsystem
// is CameraIngestor::bind_camera(), which is main-thread by construction because of the ts==0 cache.
// Every camera in ImageEdge.calibCameras that is NOT the driving one: bind, extract, pair against
// the same LiDAR corners, and feed its own evidence file. Deliberately a separate function from
// pump_image_edges(): the driving camera's path also produces the pose factor and the shadow, and
// none of that should run twice.
void SpecificWorker::place_triple_points_in_room(rc::ImageEdgeObs &obs,
                                                 const rc::CameraIngestor &ing,
                                                 const Eigen::Vector3f &pose)
{
    const Eigen::Matrix3f Rcr = obs.cam_R_robot;          // robot -> camera
    const Eigen::Matrix3f Rrc = Rcr.transpose();          // camera -> robot
    const Eigen::Vector3f o_rob = -Rrc * obs.cam_t_robot; // camera centre in robot coords
    const float c = std::cos(pose.z()), sn = std::sin(pose.z());
    const auto to_room = [&](const Eigen::Vector3f &v, bool is_point)
    {
        const Eigen::Vector3f r(c * v.x() - sn * v.y(), sn * v.x() + c * v.y(), v.z());
        return is_point ? Eigen::Vector3f(r.x() + pose.x(), r.y() + pose.y(), r.z()) : r;
    };
    const Eigen::Vector3f o_room = to_room(o_rob, true);

    // ONE map, pixel -> room plane, used both for the point and for its uncertainty. Writing it once
    // is the point: a covariance propagated through a SECOND, hand-written copy of this geometry would
    // be a model of the map rather than the map, and the two would drift apart silently.
    // The plane a corner lies in is its own height: 0 for a floor corner, room_height for a ceiling
    // one. A ray nearly parallel to that plane meets it nowhere useful — the intersection runs away to
    // infinity — so it is skipped rather than clamped.
    const auto intersect = [&](const rc::TriplePoint &tp, double u, double v,
                               Eigen::Vector3f &out) -> bool
    {
        const Eigen::Vector3f d_cam = ing.ray_from_pixel(u, v);
        if (not (d_cam.norm() > 1e-6f)) return false;
        const Eigen::Vector3f d_room = to_room(Rrc * d_cam, false);
        if (std::abs(d_room.z()) < 1e-3f) return false;
        const float tt = (tp.p_room.z() - o_room.z()) / d_room.z();
        if (not (tt > 0.f) or not std::isfinite(tt)) return false;   // behind the camera
        out = o_room + tt * d_room;
        return out.allFinite();
    };

    for (auto &t : obs.triple_points)
    {
        Eigen::Vector3f p;
        if (not intersect(t, t.uv_meas.x(), t.uv_meas.y(), p)) continue;
        t.p_room_meas = p;
        // Range along the ray, so a consumer has one whether or not a depth stream exists. NOT the
        // same quantity as depth_raw, which stays what the sensor said.
        t.range_m = (p - o_room).norm();

        // ── cov_uv (px^2) carried into the room plane (m^2) ─────────────────────────────────────
        // A CENTRED difference of the intersection itself, one pixel apart, gives d(x, y)/d(u, v)
        // directly in metres per pixel. No angular rule is assumed and no focal length is quoted:
        // whatever the camera model does — pinhole, equirectangular, cylindrical — and whatever the
        // grazing geometry does to the v axis (the d^2/h amplification a floor corner suffers and a
        // bearing does not) is already inside the map being differenced. Half a pixel each side stays
        // within the frame for any uv the extractor could have produced, and the map is smooth there.
        // If either probe fails the corner simply keeps cov_room zero, and the consumer must say so
        // rather than draw the width a zero covariance would imply.
        Eigen::Vector3f p_up, p_um, p_vp, p_vm;
        if (intersect(t, t.uv_meas.x() + 0.5, t.uv_meas.y(), p_up) and
            intersect(t, t.uv_meas.x() - 0.5, t.uv_meas.y(), p_um) and
            intersect(t, t.uv_meas.x(), t.uv_meas.y() + 0.5, p_vp) and
            intersect(t, t.uv_meas.x(), t.uv_meas.y() - 0.5, p_vm))
        {
            Eigen::Matrix2f J;                       // d(x, y) / d(u, v), metres per pixel
            J.col(0) = (p_up - p_um).head<2>();
            J.col(1) = (p_vp - p_vm).head<2>();
            const Eigen::Matrix2f C = J * t.cov_uv * J.transpose();
            if (C.allFinite()) t.cov_room = C;
        }
    }
}

void SpecificWorker::pump_calib_channels()
{
    const auto res = room_concept_.get_last_result();
    if (not res.has_value() or room_polygon_.size() < 3) return;

    for (auto &chp : calib_channels_)
    {
        auto &ch = *chp;
        if (not ch.bound)
        {
            if (not ch.ingestor->bind_camera(params.LIDAR_ROBOT_FRAME)) continue;
            ch.bound = true;
            ch.source->set_room_polygon(room_polygon_);
            auto ic = ch.source->config();
            ic.room_height = params.room_height;
            ch.source->set_config(ic);
            ch.ingestor->start();
            qInfo() << "[camcal] channel" << QString::fromStdString(ch.name) << "bound";
        }
        if (not ch.loaded)
        {
            ch.loaded = true;
            ch.calib.set_camera(ch.name, params.LIDAR_ROBOT_FRAME);
            // Same model on every camera, or the Calib window would show two mounts judged by two
            // different notions of uncertainty side by side.
            ch.calib.set_vertex_offset_sigma_px(params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX);
            const std::string path = ch.calib.path();
            const std::size_t k_aux = ch.calib.load(path);
            reconcile_mount_nominal(ch.calib, *ch.ingestor, ch.name);
            push_mount_correction(*ch.ingestor, ch.calib.applied(), ch.name, "resumed");
            if (k_aux > 0)
                qInfo().nospace() << "[camcal] " << QString::fromStdString(ch.name)
                                  << " resumed from " << QString::fromStdString(path)
                                  << " (" << k_aux << " pairs)";
        }

        // ── This channel's own column in the Calib window ────────────────────────────────────────
        // ★ THE SECOND ESTIMATOR HAD NO WAY OUT. Only the driving camera's pool ever reached the
        //   viewer (see mount_pair_update), so this channel's evidence accumulated in its own file
        //   with nothing on screen able to show it, while the window's single camera block was
        //   captioned with whichever camera reported last. One column per camera now, each fed by
        //   its own estimator.
        // ★ BEFORE take_latest(), not after the pairing loop. Placed after it, a channel that is
        //   bound but currently sees no corner — including one that has just RESUMED a full evidence
        //   file from disk — would never push, and its column would read "no evidence" while the
        //   evidence sat on disk. The column must describe the estimator, not this tick's luck.
        //   Wall clock, 5 s, matching the driving camera's window so both columns are the same age.
        if (const auto now_ms = QDateTime::currentMSecsSinceEpoch();
            viewer_ and now_ms - ch.viz_ms >= 5000)
        {
            ch.viz_ms = now_ms;
            if (const auto sol = ch.calib.solve(); sol.ok)
            {
                Eigen::Matrix<float, rc::camcal::P_COUNT, 1> pv, sv;
                // Same unit conversion as the driving camera: the estimator works in units of the
                // PRIOR SIGMA, so value AND sigma are both scaled back to physical here — scaling
                // only the value would put a dimensionless uncertainty beside a physical number.
                const double psig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
                for (int i = 0; i < rc::camcal::P_COUNT; ++i)
                {
                    pv(i) = static_cast<float>(sol.p(i)     * psig[i]);
                    sv(i) = static_cast<float>(sol.sigma(i) * psig[i]);
                }
                viewer_->set_camera_calibration(pv, sv, sol.informed,
                                                static_cast<float>(sol.cond),
                                                ch.calib.pairs(), ch.name);
            }
        }

        // ── The ceiling the contour is projected at, REFRESHED ──────────────────────────────────────
    // ★ It used to be set only inside the bind block above, so a ceiling measured one second after
    //   binding never reached the projection — and with the wire broken it never arrived at all. The
    //   room polygon beside it was already refreshed every tick for exactly this reason.
    // ⚠ A change here moves every wall-ceiling corner, which is what the mount's HEIGHT parameter
    //   reads, so it is logged when it moves rather than changing the geometry quietly.
    if (image_edge_bound_)
    {
        const float mz = room_concept_.measured_ceiling();
        const float want = mz > 1.5f ? mz : params.room_height;
        if (auto ic = image_edge_source_->config(); std::abs(ic.room_height - want) > 1e-4f)
        {
            qInfo().noquote() << QString::asprintf(
                "[imgedge] wall-ceiling contour re-projected: room_height %.4f -> %.4f m (%s;"
                " the scenario states %.4f)", ic.room_height, want,
                mz > 1.5f ? "MEASURED by the LiDAR" : "stated in the scenario", params.room_height);
            if (mz > 1.5f)
                qInfo().noquote() << QString::asprintf(
                    "         its uncertainty is +/- %.3f m — the ceiling corners' z now carries that,"
                    " and the mount's height can no longer silently absorb it",
                    room_concept_.measured_ceiling_sigma());
            ic.room_height = want;
            image_edge_source_->set_config(ic);
        }
    }

    rc::GrayFrame frame;
        if (not ch.ingestor->take_latest(frame)) continue;

        const Eigen::Affine2f &rp = res->robot_pose;
        const Eigen::Vector3f pose(rp.translation().x(), rp.translation().y(),
                                   std::atan2(rp.linear()(1, 0), rp.linear()(0, 0)));
        const std::int64_t dt_ms = static_cast<std::int64_t>(frame.stamp) - res->timestamp_ms;
        auto obs = ch.source->extract(frame, ch.ingestor->model(), ch.ingestor->cam_R_robot(),
                                      ch.ingestor->cam_t_robot(), pose, res->covariance,
                                      Eigen::Vector3f::Zero(), dt_ms, nullptr);
        place_triple_points_in_room(obs, *ch.ingestor, pose);
        // Each camera's corners go to ITS OWN window: the ZED popup shows the ZED's corners even
        // though the Ricoh is the one driving, which is the comparison worth being able to see.
        if (viewer_) viewer_->set_triple_points(obs.triple_points, ch.name);
        if (obs.triple_points.empty() or res->corner_matches.empty()) continue;

        for (const auto &tp : obs.triple_points)
        {
            const auto it = std::ranges::find_if(res->corner_matches,
                [&](const auto &m) { return m.model_index == tp.vertex; });
            if (it == res->corner_matches.end() or it->suppressed) continue;
            const auto pr = rc::mount::make_pair(tp, *it, ch.ingestor->model(),
                                                 ch.ingestor->cam_R_robot(),
                                                 ch.ingestor->cam_t_robot(),
                                                 params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                                 params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                                 params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
            if (not pr.ok) continue;
            ch.calib.add(pr);
            ++ch.pairs;
            // The SAME row the driving camera writes. Without it this channel produced evidence
            // nobody could re-derive: arm 7 needs both cameras' mounts re-solved under one
            // injection, and the closure recomputed from the two, all from a single drive.
            if (not ch.csv.is_open()) open_pair_log(ch.csv, ch.name, *ch.ingestor);
            write_pair_row(ch.csv, ch.name, static_cast<std::int64_t>(frame.stamp), pr,
                           tp.from == rc::ContourClass::WallCeiling,
                           it->angle_deg, it->assoc_chi2_val, it->n_rivals, it->runnerup_chi2,
                           ch.ingestor->mount_correction());
            // Per ROW, not per frame: see the note on the driving camera's call — the local
            // pixel-to-angle scale varies across a pinhole's field, and this channel IS the pinhole.
            const Eigen::Vector2f ppr = rc::img::px_per_rad_at(obs.cam, pr.uv_image);
            if (ppr.x() > 0.f and ppr.y() > 0.f)
                loop_closure_observe(ch.name, tp.vertex,
                                     tp.from == rc::ContourClass::WallCeiling,
                                     static_cast<double>(pr.r.x() / ppr.x()),
                                     static_cast<double>(pr.r.y() / ppr.y()),
                                     static_cast<std::int64_t>(frame.stamp));
        }
        if (ch.pairs % 2000 < 12 and ch.pairs > 0)
        {
            if (const auto sol_apply = ch.calib.solve(); sol_apply.ok)
                apply_mount_solve(ch.calib, *ch.ingestor, sol_apply, ch.name);
            ch.calib.save(ch.calib.path());
            if (ch.csv.is_open()) ch.csv.flush();   // same cadence as the evidence beside it
            if (const auto sol = ch.calib.solve(); sol.ok)
                qInfo().nospace().noquote()
                    << "[camcal] " << QString::fromStdString(ch.name) << " " << ch.pairs
                    << " pairs | yaw "
                    << QString::number(-sol.p(2) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA * 180.0 / M_PI,
                                       'f', 4)
                    << " deg | cond " << QString::number(sol.cond, 'f', 1);
        }
    }
}

void SpecificWorker::pump_image_edges()
{
    if (not camera_ingestor_ or not image_edge_source_) return;

    // Bind lazily and keep retrying: the camera node, its media descriptor and the RT chain all come
    // up asynchronously, and a miss here is normal for the first few seconds.
    if (not image_edge_bound_)
    {
        // The room polygon and the robot-frame name are both resolved asynchronously during
        // start-up, so this retries until all of it is available rather than binding once and
        // failing permanently.
        if (room_polygon_.size() < 3) return;
        if (not camera_ingestor_->bind_camera(params.LIDAR_ROBOT_FRAME)) return;
        image_edge_bound_ = true;
        image_edge_source_->set_room_polygon(room_polygon_);
        // The objects the room currently believes in, so the extraction can explain away wall
        // samples they stand in front of. Refreshed each tick below, not only at bind: furniture
        // appears, moves and is forgotten while the agent runs.
        image_edge_source_->set_object_anchors(room_concept_.object_anchors());
        // room_height is read from the graph after construction, so refresh it here too — and
        // PREFER THE MEASURED CEILING when the LiDAR has one. The startup geometry check already
        // locates the ceiling plane with a likelihood test (annulus vs wall-top) and found 3.01 m
        // against a stated 3.00; until 2026-09-03 that number only capped the wall band and was then
        // discarded, while the wall-ceiling contour this module projects used the hand-typed
        // constant. A stated ceiling that is a few cm wrong is a pure SCALE error on every range the
        // contour implies, and a Manhattan estimator cannot see it.
        auto ic = image_edge_source_->config();
        ic.room_height = params.room_height;
        const float measured_ceiling = room_concept_.measured_ceiling();
        if (measured_ceiling > 1.5f)
        {
            ic.room_height = measured_ceiling;
            if (std::abs(measured_ceiling - params.room_height) > 0.05f)
                qWarning() << "[imgedge] the LiDAR measures the ceiling at" << measured_ceiling
                           << "m but the scenario states" << params.room_height
                           << "m; using the measurement. A stated ceiling that is wrong scales every"
                           << "range the wall-ceiling contour implies.";
        }
        image_edge_source_->set_config(ic);
        qInfo() << "[imgedge] bound to" << QString::fromStdString(params.IMAGE_EDGE_CAMERA)
                << "in frame" << QString::fromStdString(params.LIDAR_ROBOT_FRAME)
                << "| polygon" << room_polygon_.size() << "pts, room_height" << ic.room_height
                << (measured_ceiling > 1.5f ? "(measured by the LiDAR)" : "(stated in the scenario)");
    }

    rc::GrayFrame frame;
    image_edge_source_->set_object_anchors(room_concept_.object_anchors());
    if (not camera_ingestor_->take_latest(frame)) return;    // no fresh image this tick

    const auto res = room_concept_.get_last_result();
    if (not res.has_value()) return;

    // The pose at which to PREDICT and SEARCH. This is the localizer's own current estimate, which is
    // correct and is NOT the circularity: the search only decides WHICH image edge each model contour
    // is matched to. The residual the factor then minimises is a function of the state VARIABLE, and
    // uv_meas is held fixed while it does — which is also what LM's accept/reject test requires.
    const Eigen::Affine2f& rp = res->robot_pose;
    const Eigen::Vector3f pose(rp.translation().x(), rp.translation().y(),
                               std::atan2(rp.linear()(1, 0), rp.linear()(0, 0)));

    const std::int64_t dt_ms = static_cast<std::int64_t>(frame.stamp) - res->timestamp_ms;

    // Body twist, for the image/LiDAR dt nuisance column. A VARIANCE, never a correction: dead-
    // reckoning the pose forward here would reintroduce exactly the graph-pose dependency we removed.
    // Measured by differencing two published localizer results, so it needs no graph read and no
    // command channel (the commanded velocity is not what the robot did).
    Eigen::Vector3f twist = Eigen::Vector3f::Zero();
    if (image_edge_prev_ts_ > 0 and res->timestamp_ms > image_edge_prev_ts_)
    {
        const float dt_s = 1e-3f * static_cast<float>(res->timestamp_ms - image_edge_prev_ts_);
        if (dt_s > 1e-3f and dt_s < 1.0f)
        {
            const Eigen::Vector3f d = pose - image_edge_prev_pose_;
            const float dth = std::atan2(std::sin(d.z()), std::cos(d.z()));
            // Rotate the room-frame displacement into the robot frame: the nuisance is expressed
            // there, because that is where the lever arm to the camera lives.
            const float c = std::cos(pose.z()), sn = std::sin(pose.z());
            twist = Eigen::Vector3f(( c * d.x() + sn * d.y()) / dt_s,
                                    (-sn * d.x() +  c * d.y()) / dt_s,
                                    dth / dt_s);
        }
    }
    image_edge_prev_pose_ = pose;
    image_edge_prev_ts_   = res->timestamp_ms;

    rc::ImageEdgeSource::Stats st;
    auto obs = image_edge_source_->extract(frame, camera_ingestor_->model(),
                                           camera_ingestor_->cam_R_robot(),
                                           camera_ingestor_->cam_t_robot(),
                                           pose, res->covariance, twist, dt_ms, &st);
    // The correction that was inside the extrinsic used above. Recorded on the observation so a
    // shadow solve can remove it and re-create the nominal mount for THESE measurements — factor B.
    obs.mount_correction = camera_ingestor_->mount_correction();
    // Provenance travels WITH the evidence from here on: every downstream consumer (the pair log,
    // the triple log, the viewer overlay) then reports the camera this observation actually came
    // from, not the one the config names at the moment it is asked.
    obs.camera = params.IMAGE_EDGE_CAMERA;

    // ── Range for the triple points, from the ZED depth plane ────────────────────────────────────
    // Zero-copy: the pixel list is known now (the corners were detected from the RGB frame above),
    // so probe_depth reads exactly these pixels inside the loaned SHM view and copies no frame.
    // The depth frame is whatever is newest at this instant; its stamp is recorded beside the RGB
    // stamp rather than assumed equal, because they are different streams from different threads.
    if (not obs.triple_points.empty())
    {
        std::vector<Eigen::Vector2f> uv;
        uv.reserve(obs.triple_points.size());
        for (const auto& t : obs.triple_points) uv.push_back(t.uv_meas);
        std::vector<float> dm;
        std::int64_t dstamp = 0;
        if (camera_ingestor_->probe_depth(uv, 2, dm, dstamp) > 0)
        {
            obs.depth_stamp_ms = dstamp;
            for (std::size_t k = 0; k < obs.triple_points.size(); ++k)
            {
                auto& t = obs.triple_points[k];
                t.depth_raw = dm[k];
                Eigen::Vector3d xyz;
                if (not rc::img::xyz_from_pixel_depth(camera_ingestor_->model(),
                                                      t.uv_meas.x(), t.uv_meas.y(), dm[k], xyz))
                    continue;
                t.p_cam_meas = xyz.cast<float>();
                t.range_m    = static_cast<float>(xyz.norm());
                // camera -> robot -> room. cam_R_robot maps robot into camera, so its transpose
                // brings the point back; then the pose rotation, which is R(+theta) — the inverse of
                // the R(-theta) the projection path applies.
                const Eigen::Vector3f p_rb = camera_ingestor_->cam_R_robot().transpose()
                                           * (t.p_cam_meas - camera_ingestor_->cam_t_robot());
                const float cs = std::cos(pose.z()), sn = std::sin(pose.z());
                t.p_room_meas = Eigen::Vector3f(pose.x() + cs * p_rb.x() - sn * p_rb.y(),
                                                pose.y() + sn * p_rb.x() + cs * p_rb.y(),
                                                p_rb.z());
                // ★ The sigma is a PLACEHOLDER and is marked as one. A depth sigma is a property of
                //   the sensor at that range and this camera's has not been measured here; the
                //   LiDAR-anchored depth-correction work in retina is where that number should come
                //   from. Writing a plausible constant and treating it as measured is how a term
                //   acquires unearned authority, so nothing may consume this until it is real.
                t.range_sigma = -1.f;
            }
        }
    }
    // Place the corners in the room BEFORE anything consumes them: the 2-D canvas needs it, and it
    // works for a camera with no depth at all, which the panorama is.
    place_triple_points_in_room(obs, *camera_ingestor_, pose);
    mount_pair_update(obs, res->corner_matches, static_cast<std::int64_t>(frame.stamp));
    // The overlay draws these on whichever window IS this camera; the name travels with them so a
    // ricoh corner can never be painted onto a zed frame at a plausible-looking wrong position.
    if (viewer_) viewer_->set_triple_points(obs.triple_points, params.IMAGE_EDGE_CAMERA);
    room_concept_.set_image_edges(std::move(obs));

    if (const auto [polls, hits] = camera_ingestor_->depth_stats();
        polls > 0 and polls % 100 == 0)
        qInfo().nospace().noquote()
            << "[depth] " << hits << "/" << polls << " polls delivered a frame ("
            << QString::number(100.0 * hits / polls, 'f', 1) << "%)"
            << (hits == 0 ? "  <- subscriber exists and NOTHING arrives: check the producer is "
                            "publishing depth and that the frame is under MAX_IMAGE_BYTES (3.69 MB), "
                            "which the plane drops SILENTLY" : "");

    // ~1/s liveness line. n_searched == 0 with n_projected > 0 means the contours project but carry
    // no gradient — a real answer (blank walls), not a plumbing failure, and the two are worth being
    // able to tell apart from the log alone.
    const auto now = QDateTime::currentMSecsSinceEpoch();
    if (now - last_image_edge_log_ms_ > 1000)
    {
        last_image_edge_log_ms_ = now;
        qInfo() << "[imgedge] contours" << st.n_contours << "projected" << st.n_projected
                << "visible" << st.n_visible << "searched" << st.n_searched
                << "occluded" << st.n_occluded
                << "| corners" << st.n_triple << "(" << st.n_triple_occl << "behind a wall)"
                << "| med sigma" << st.med_sigma_px << "px"
                << "med L" << st.med_search_px << "px sigma_i" << st.sigma_i
                << "| dt_img_lidar" << dt_ms << "ms";
    }
}

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
        pump_image_edges();
        const auto t_edge = t.nsecsElapsed();
        t.restart();
        pump_calib_channels();   // extra cameras: calibration only, never the pose
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
            if (camera_ingestor_) add_conv(params.IMAGE_EDGE_CAMERA, camera_ingestor_->convert_stats());
            for (auto &chp : calib_channels_)
                if (chp->ingestor) add_conv(chp->name, chp->ingestor->convert_stats());
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
    update_rt_rate_readout(now_ms, on_gui_thread);

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
        room_polygon_offset_ = params.RECENTER_ROOM_POLYGON
                                   ? rc::SvgRoomLoader::recenter_to_bbox_center(room_polygon)
                                   : Eigen::Vector2f::Zero();
        if (room_polygon_offset_.norm() > 1e-3f)
            qInfo() << "[room] Room polygon recentred on its bbox centre: origin moved by"
                    << room_polygon_offset_.x() << room_polygon_offset_.y() << "m."
                    << "Poses in the room frame are shifted by that amount — a seed pose or object"
                    << "RT edges saved before this change are stale.";

        room_polygon_ = room_polygon;
        room_concept_.configure_room_from_polygon(room_polygon_);
        room_initialized_from_svg_polygon_ = true;
        return;
    }
    room_polygon_.clear();
    room_polygon_offset_ = Eigen::Vector2f::Zero();
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

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{
    if (!G || id == 0)
        return;

    const auto touches_any = [&att_names](std::initializer_list<const char*> names)
    {
        return std::ranges::any_of(names, [&att_names](const char* name)
        {
            return std::find(att_names.begin(), att_names.end(), name) != att_names.end();
        });
    };

    // LiDAR is read from the media plane (pumped in compute), not the DSR graph laser_* attrs.

    // ★ OUR OWN AFFORDANCE NODE — handled BEFORE the robot-id filter below, which would otherwise
    // discard it. The consumer clears epistemic_pending when it finishes; being told the moment that
    // happens is what removes the missed-edge race that wedged both agents (549 s with zero
    // completions detected while the consumer was completing continuously).
    if (std::ranges::any_of(att_names, [](const std::string& n)
                            { return n == "epistemic_pending" or n == "aff_outcome"; }))
        scene_graph_->on_affordance_attr_changed(id);

    if (const auto robot_id = scene_graph_->robot_id(); robot_id != 0 && id != robot_id)
        return;

    const bool touches_current_speed = touches_any({
        "robot_current_advance_speed",
        "robot_current_side_speed",
        "robot_current_angular_speed",
        "robot_current_speed_timestamp"
    });
    const bool touches_ref_speed = touches_any({
        "robot_ref_adv_speed",
        "robot_ref_side_speed",
        "robot_ref_rot_speed",
        "robot_ref_speed_timestamp"
    });
    if (not touches_current_speed and not touches_ref_speed)
        return;

    const auto node_opt = G->get_node(id);
    if (!node_opt.has_value())
        return;

    const auto &robot_node = node_opt.value();

    // No inertial branch here: the IMU rides the media plane (rc/imu/data) and never touches the
    // graph, so this slot has nothing to do for it. ImuIngestor owns that channel; the commentary
    // about why the gyro matters for yaw now lives with it, next to the code that uses it.

    if (touches_current_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_current_advance_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_current_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_current_angular_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_current_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_current_speed_timestamp_)
                        {
                            static std::mt19937 gen{std::random_device{}()};
                            const float nf = params.ODOMETRY_NOISE_FACTOR;

                            auto add_noise = [&](float value) -> float {
                                if (nf <= 0.f || value == 0.f) return value;
                                std::normal_distribution<float> dist(0.f, std::abs(value) * nf);
                                return value + dist(gen);
                            };

                            rc::OdometryReading odom;
                            odom.adv = add_noise(adv_value.value());
                            odom.side = add_noise(side_value.value());
                            odom.rot = add_noise(rot_value.value());
                            odom.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            // The producer's own clock, where it has one. A simulator's velocities are
                            // per SIMULATION second, so integrating them over wall-clock intervals
                            // under-counts by the sim/wall ratio; this is the stamp that makes the
                            // interval agree with the rate. Absent on real hardware, where the two
                            // clocks are the same thing and integration_ts_ms() falls back cleanly.
                            if (const auto sim_ts = G->get_attrib_by_name<robot_current_speed_sim_timestamp_att>(robot_node);
                                sim_ts.has_value())
                                odom.sim_ts_ms = static_cast<std::int64_t>(sim_ts.value());
                            if (const auto sim_flag = G->get_attrib_by_name<robot_current_speed_simulated_att>(robot_node);
                                sim_flag.has_value())
                                odom.simulated = sim_flag.value();
                            // ── The producer's OWN per-sample velocity variance ────────────────────
                            // robot_concept forwards FullPoseEuler::velCov here as {adv, side, rot},
                            // already de-crossed from the body indices (m11 = forward on this +Y robot).
                            // Absent attribute = this producer never published it; a NEGATIVE entry =
                            // it published and said "unknown" for that channel. Both leave the fields
                            // at -1 and the preintegrator falls back to its asserted constant, so a
                            // producer that says nothing is not silently treated as a perfect one.
                            //
                            // NOT NOISE-FACTOR-ADJUSTED on purpose: ODOMETRY_NOISE_FACTOR above injects
                            // SYNTHETIC noise for robustness tests, and pretending the sensor reported
                            // that would let a test knob masquerade as a measurement.
                            if (const auto vvar = G->get_attrib_by_name<robot_current_speed_variance_att>(robot_node);
                                vvar.has_value())
                            {
                                const auto& v = vvar.value().get();
                                if (v.size() >= 3)
                                {
                                    const auto stated = [](float x) { return x >= 0.f ? x : -1.f; };
                                    odom.var_adv  = stated(v[0]);
                                    odom.var_side = stated(v[1]);
                                    odom.var_rot  = stated(v[2]);
                                }
                            }
                            if (odom.simulated and odom.sim_ts_ms > 0)
                                sim_clock_.observe(odom.source_ts_ms, odom.sim_ts_ms);
                            odom.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            odom.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_odometry_ingress("dsr_current_speed",
                                                                  adv_value.value(),
                                                                  rot_value.value(),
                                                                  odom.adv,
                                                                  odom.rot,
                                                                  odom.source_ts_ms);
                            // ── PER-SAMPLE log, for measuring the stream's own statistics ──────
                            // Written HERE, where samples arrive, and not from the per-cycle debug
                            // row: that row keeps only the latest sample and is emitted once per
                            // lidar sweep, so it aliases this stream onto a slower one. An
                            // autocorrelation computed on aliased rows answers a different question
                            // than the one asked, and looks perfectly reasonable doing it.
                            //
                            // `seq` is the arrival counter and `source_ts_ms` the producer's stamp:
                            // together they make a DROPPED sample visible as a stamp gap without a
                            // seq gap, which matters because a dropped sample changes the effective
                            // sampling interval and therefore the very quantity being measured.
                            if (params.ODOM_SAMPLE_LOG)
                            {
                                if (not odom_sample_log_.is_open())
                                {
                                    odom_sample_log_.open("etc/odom_samples.csv",
                                                          std::ios::out | std::ios::trunc);
                                    // Locale-proof the WRITE side: these machines run es_ES, where a
                                    // stray comma separator would silently turn every value into a
                                    // column break. See CLAUDE.md.
                                    odom_sample_log_.imbue(std::locale::classic());
                                    odom_sample_log_ << "seq,recv_ts_ms,source_ts_ms,sim_ts_ms,simulated,"
                                                        "adv,side,rot,var_adv,var_side,var_rot,"
                                                        "cmd_adv,cmd_side,cmd_rot,cmd_ts_ms\n";
                                }
                                odom_sample_log_ << odom_sample_seq_++ << ','
                                                 << odom.recv_ts_ms   << ','
                                                 << odom.source_ts_ms << ','
                                                 << odom.sim_ts_ms    << ','
                                                 << (odom.simulated ? 1 : 0) << ','
                                                 << odom.adv << ',' << odom.side << ',' << odom.rot << ','
                                                 << odom.var_adv << ',' << odom.var_side << ',' << odom.var_rot << ','
                                                 << last_cmd_adv_ << ',' << last_cmd_side_ << ','
                                                 << last_cmd_rot_ << ',' << last_cmd_ts_ms_ << '\n';
                                odom_sample_log_.flush();   // 50 lines/s; a lost tail costs more
                            }
                            odometry_buffer_.put<0>(std::move(odom), static_cast<std::uint64_t>(odom.recv_ts_ms));
                            last_robot_current_speed_timestamp_ = source_ts;
                            last_robot_adv_speed_  = adv_value.value();
                            last_robot_side_speed_ = side_value.value();
                            last_robot_rot_speed_  = rot_value.value();
                        }
                    }
                }
            }
        }
    }

    if (touches_ref_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_ref_adv_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_ref_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_ref_rot_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_ref_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_ref_speed_timestamp_)
                        {
                            rc::VelocityCommand cmd;
                            cmd.adv_y = adv_value.value();
                            cmd.adv_x = side_value.value();
                            cmd.rot = rot_value.value();
                            cmd.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            cmd.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            cmd.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_command_ingress("dsr_ref",
                                                                 adv_value.value(),
                                                                 rot_value.value(),
                                                                 cmd.adv_y,
                                                                 cmd.rot,
                                                                 cmd.source_ts_ms);
                            last_cmd_adv_    = cmd.adv_y;
                            last_cmd_side_   = cmd.adv_x;
                            last_cmd_rot_    = cmd.rot;
                            last_cmd_ts_ms_  = cmd.source_ts_ms;
                            velocity_buffer_.put<0>(std::move(cmd), source_ts);
                            last_robot_ref_speed_timestamp_ = source_ts;
                        }
                    }
                }
            }
        }
    }
}


void SpecificWorker::modify_node_slot(std::uint64_t /*id*/, const std::string& /*type*/)
{
    // LiDAR now arrives via the media plane (pumped in compute); the graph 'laser' node is not read.
}

// Re-run the graph viewer's twopi layout now and again once queued, so it also settles after the
// pending node/edge update signals are processed. No-op when Agent.graph is off (no viewer widget).
// Lives here (not in the regenerated genericworker) so robocompdsl regeneration cannot clobber it;
// mirrors robot_concept. find_graph_viewer() is the inherited GenericWorker helper.
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
