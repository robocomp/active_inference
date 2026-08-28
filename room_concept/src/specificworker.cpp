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

#include "image_edge_ops.h"   // xyz_from_pixel_depth(): model-aware, unlike cortex's pinhole-only version
#include <locale>

#include <algorithm>
#include <print>
#include <sys/resource.h>   // getrusage — process CPU% readout
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <random>
#include <stdexcept>
#include <fstream>
#include <unordered_set>
#include <QDir>
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

    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    // ── Load all config (agent + RoomConcept + EpistemicController params) ──
    rc::load_room_config(configLoader, params, room_concept_, epistemic_controller_);

    // ── Collaborators (constructor injection; worker owns rt_api + shared params) ──
    rt_api_ = G->get_rt_api();
    scene_graph_ = std::make_unique<rc::RoomSceneGraph>(
        G, rt_api_.get(), params, room_concept_, epistemic_controller_,
        [this] { trigger_graph_layout_twopi(); });
    lidar_ingestor_ = std::make_unique<rc::LidarIngestor>(G, room_concept_, params);
    // No source switch any more: the producer no longer writes the imu_* attributes, so a "dsr"
    // option would select a path with nothing on it — a dead config of exactly the kind this
    // codebase keeps rediscovering. The escape hatch is git, not a flag that cannot work.
    imu_ingestor_ = std::make_unique<rc::ImuIngestor>(G, imu_buffer_, sim_clock_);
    // RGB edge alignment. Constructed ONLY when enabled: with ImageEdge.enable = false there is no
    // subscriber, no thread and no extraction, so the feature is exactly free when off.
    if (params.IMAGE_EDGE_ENABLE)
    {
        camera_ingestor_ = std::make_unique<rc::CameraIngestor>(G, params.IMAGE_EDGE_CAMERA);
        // Set BEFORE the first bind_camera(): the correction is applied where the extrinsic is read.
        camera_ingestor_->set_mount_yaw_correction(params.IMAGE_EDGE_MOUNT_YAW_CORR);
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
    }

    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.high_lidar_buffer = &lidar_ingestor_->buffer();
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    run_ctx.imu_buffer      = &imu_buffer_;
    run_ctx.sim_clock       = &sim_clock_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;

    initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    room_concept_.set_seed_pose_file(pose_path);

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

    if (auto* w = viewer_->widget())
    {
        connect(w->btn_camera_viz, &QPushButton::clicked, this, [this] { viewer_->show_camera(); });
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

    // Publish corrections the INSTANT the localizer produces them, not on the next compute() tick.
    // The callback runs on the LOCALIZER thread, so it only marshals the actual graph write to the
    // MAIN thread via a Qt::QueuedConnection (maybe_publish_corrected_pose touches the DSR graph and
    // must stay on the main thread). Removes ~one compute-period of lidar→RT-publish latency; the
    // timestamp dedup keeps it idempotent with compute()'s own publish call.
    room_concept_.set_on_result_ready([this]()
    {
        QMetaObject::invokeMethod(this, [this]() { maybe_publish_corrected_pose(); }, Qt::QueuedConnection);
    });

    // LiDAR is pumped synchronously from compute() (no ingest thread); just start the localizer.
    room_concept_.start();

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
    pose_trace_ << QDateTime::currentMSecsSinceEpoch() << ',' << type << ',' << valid_ts_ms << ','
                << x << ',' << y << ',' << th << ',' << innov_norm << '\n';
    pose_trace_.flush();
}

bool SpecificWorker::maybe_publish_corrected_pose()
{
    // Main-thread only (compute() directly, or the localizer's on_result_ready callback marshalled here
    // via QueuedConnection). In PreserveBootstrapRoom mode the room is a static prior — the localizer
    // still runs for the viewer, but we never touch the graph.
    if (shutting_down_.load() || params.PRESERVE_BOOTSTRAP_ROOM)
        return false;

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

    // Publish (corrected pose → robot↔room RT) at the optimizer rate.
    scene_graph_->update(*loc_res, last_robot_adv_speed_, last_robot_side_speed_, last_robot_rot_speed_);
    last_published_pose_ = loc_res->robot_pose;
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
void SpecificWorker::mount_pair_update(const rc::ImageEdgeObs &obs,
                                       const std::vector<rc::CornerDetector::CornerMatch> &matches,
                                       std::int64_t timestamp_ms)
{
    if (obs.triple_points.empty() or matches.empty() or not camera_ingestor_) return;
    if (mp_win_start_ms_ == 0) mp_win_start_ms_ = timestamp_ms;

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

        if (not mp_csv_.is_open())
        {
            mp_csv_.open("etc/image_edge_pair.csv", std::ios::out | std::ios::trunc);
            if (mp_csv_.is_open())
            {
                mp_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                mp_csv_ << "ts_ms,vertex,u_img,v_img,u_lidar,v_lidar,ru,rv,"
                           "sigu,sigv,assoc_prob,range_m,angle_deg,assoc_chi2\n";
            }
        }
        if (mp_csv_.is_open())
            mp_csv_ << timestamp_ms << ',' << pr.vertex << ','
                    << pr.uv_image.x() << ',' << pr.uv_image.y() << ','
                    << pr.uv_lidar.x() << ',' << pr.uv_lidar.y() << ','
                    << pr.r.x() << ',' << pr.r.y() << ','
                    << std::sqrt(std::max(0.f, pr.cov(0, 0))) << ','
                    << std::sqrt(std::max(0.f, pr.cov(1, 1))) << ','
                    << pr.assoc_prob << ',' << pr.range_m << ','
                    << it->angle_deg << ',' << it->assoc_chi2_val << '\n';
    }

    if (timestamp_ms - mp_win_start_ms_ < 5000) return;
    mp_win_start_ms_ = timestamp_ms;
    if (mp_csv_.is_open()) mp_csv_.flush();

    const auto win  = mp_win_.solve();
    const auto pool = mp_pool_.solve();
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
        << " pairs of " << mp_seen_ << " triple points)" << body
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
            << "[mount/pool] " << mp_pool_.n << " pairs over " << mp_wins_ << " windows" << pb
            << " | chi2/dof " << QString::number(pool.chi2_dof, 'f', 2)
            << " | cond " << QString::number(pool.cond, 'f', 1)
            << " (" << nm[pool.rho_i] << "/" << nm[pool.rho_j] << " rho "
            << QString::number(pool.rho, 'f', 3) << ")"
            << (pool.cond < 50.0 && win.cond > 50.0
                    ? "   <- POOLING BROKE THE RIDGE: the pair is separable across poses and was not"
                      " within one window" : "");
    }
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
                       "calib_k_v,calib_k_w,calib_yaw,calib_eps,"
                       "calib_sig_kv,calib_sig_kw,calib_sig_yaw,calib_pos_var,"
                       "calib_b_omega,calib_informed,calib_cond,"
                       "imu_dvx,imu_dvy,wheel_dvx,wheel_dvy,imu_dpx,imu_dpy,imu_lin_segs\n";
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
            // Sigmas + the R actually used: without these a replay of the filter from this file is
            // guesswork, and a parameter that stopped moving cannot be told from one never taught.
            << ',' << res.calib_sigma_k_v << ',' << res.calib_sigma_k_w
            << ',' << res.calib_sigma_yaw << ',' << res.calib_pos_var
            // Joint-solve outputs: the gyro bias it can now separate, which parameters this window
            // actually taught, and how collinear the window was.
            << ',' << res.calib_b_omega << ',' << res.calib_informed << ',' << res.calib_condition
            // Linear IMU channel. imu_dv vs wheel_dv is translation's first independent cross-check;
            // logged before being fused, because a channel whose covariance is unknown (the ImuFrame
            // IDL has no acc_var) must be shown to agree with something before anything trusts it.
            << ',' << res.imu_dvx << ',' << res.imu_dvy
            << ',' << res.wheel_dvx << ',' << res.wheel_dvy
            << ',' << res.imu_dpx << ',' << res.imu_dpy << ',' << res.imu_lin_segs
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
        // room_height is read from the graph after construction, so refresh it here too.
        auto ic = image_edge_source_->config();
        ic.room_height = params.room_height;
        image_edge_source_->set_config(ic);
        qInfo() << "[imgedge] bound to" << QString::fromStdString(params.IMAGE_EDGE_CAMERA)
                << "in frame" << QString::fromStdString(params.LIDAR_ROBOT_FRAME)
                << "| polygon" << room_polygon_.size() << "pts, room_height" << params.room_height;
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
    mount_pair_update(obs, res->corner_matches, static_cast<std::int64_t>(frame.stamp));
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
                << "occluded" << st.n_occluded << "| med sigma" << st.med_sigma_px << "px"
                << "med L" << st.med_search_px << "px sigma_i" << st.sigma_i
                << "| dt_img_lidar" << dt_ms << "ms";
    }
}

void SpecificWorker::compute()
{
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    QElapsedTimer compute_timer;
    compute_timer.start();
    auto init_time = std::chrono::steady_clock::now();
    qint64 t_affordance_ms = 0;
    qint64 t_loc_fetch_ms = 0;
    qint64 t_viewer_ms = 0;
    qint64 t_dsr_ms = 0;
    qint64 t_ui_ms = 0;
    qint64 t_health_ms = 0;
    bool   did_publish = false;   // a corrected RT block was published this tick (for compute_timing.csv)

    if (last_affordance_monitor_ms_ == 0 || now_ms - last_affordance_monitor_ms_ >= 200)
    {
        QElapsedTimer section_timer;
        section_timer.start();
        scene_graph_->monitor_affordance();
        t_affordance_ms = section_timer.elapsed();
        last_affordance_monitor_ms_ = now_ms;
    }

    // LiDAR is drained by lidar_ingestor_'s dedicated ingest thread (started at Operating-enter), which
    // pumps a fresh scan to the localizer with ~0-2 ms latency instead of this ~16 ms tick. compute()
    // only reads the resulting buffer/result below.

    pump_image_edges();   // no-op unless ImageEdge.enable

    QElapsedTimer section_timer;
    section_timer.start();
    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;
    t_loc_fetch_ms = section_timer.elapsed();

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
        t_viewer_ms = section_timer.elapsed();
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
        // Publish the fresh correction now (idempotent by timestamp). The localizer also triggers this
        // the instant it finishes, via a QueuedConnection, so whichever fires first wins and this call
        // usually no-ops — but it stays here so a compute() tick still publishes if the immediate hop
        // was ever missed.
        did_publish = maybe_publish_corrected_pose();
        t_dsr_ms = section_timer.elapsed();
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
        t_ui_ms = section_timer.elapsed();
    }

    t_health_ms = 0;

    const auto total_ms = compute_timer.elapsed();
    // Sub-millisecond resolution total (compute() typically ~0.3 ms → the integer-ms section timers above
    // all read 0). This is MICROSECONDS — printed as total_us so it's not mistaken for milliseconds.
    const auto elapsed_since_init_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - init_time).count();

    // Per-tick compute-timing CSV — shows whether compute() holds the period (≈50 ms) or stalls, and
    // which section is responsible, explaining publish-rate loss vs the optimizer's production rate.
    if (not compute_csv_open_attempted_)
    {
        compute_csv_open_attempted_ = true;
        compute_csv_.open("etc/compute_timing.csv", std::ios::out | std::ios::trunc);
        if (compute_csv_.is_open())
            compute_csv_ << "wall_ms,total_ms,affordance_ms,loc_fetch_ms,viewer_ms,dsr_ms,ui_ms,did_publish,gui_thread\n";
    }
    if (compute_csv_.is_open())
    {
        compute_csv_ << now_ms << ',' << total_ms << ',' << t_affordance_ms << ',' << t_loc_fetch_ms
                     << ',' << t_viewer_ms << ',' << t_dsr_ms << ',' << t_ui_ms << ','
                     << (did_publish ? 1 : 0) << ',' << (on_gui_thread ? 1 : 0) << '\n';
        compute_csv_.flush();
    }

    if (last_compute_timing_log_ms_ == 0 || now_ms - last_compute_timing_log_ms_ >= 3000 || total_ms > 50)
    {
        last_compute_timing_log_ms_ = now_ms;
        qInfo() << "[Timing][compute]"
                << "total_us=" << elapsed_since_init_us   // MICROSECONDS (≈0.3 ms); sections below are integer ms
                << "affordance_ms=" << t_affordance_ms
                << "loc_fetch_ms=" << t_loc_fetch_ms
                << "viewer_ms=" << t_viewer_ms
                << "dsr_ms=" << t_dsr_ms
                << "ui_ms=" << t_ui_ms
                << "health_ms=" << t_health_ms
                << "gui_thread=" << on_gui_thread;
    }
    fps_counter_.print("[Compute]", 3000);
}

void SpecificWorker::initialize_room_model_from_svg()
{
    auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        params.ROOM_LAYOUT_SVG, "room_contour", false, true);
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
