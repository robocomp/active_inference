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

#include <algorithm>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <random>
#include <stdexcept>
#include <fstream>
#include <unordered_set>
#include <QDir>
#include <QDateTime>
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
    save_robot_pose_once();

    // Stop the lidar ingestion thread BEFORE tearing down RoomConcept (it calls
    // room_concept_.notify_new_lidar) and while G is still alive.
    lidar_ingestor_.stop();

    room_concept_.stop();
    cleanup_owned_nodes();

    // Crash-free terminal exit. After our cleanup (state saved, self agent node deleted, peers
    // notified) we hard-exit instead of returning into Ice::Application's communicator teardown +
    // C++ static destruction. Those run with UNDEFINED cross-TU order: a global/DDS holder copies a
    // graph Node (e.g. type "mind") AFTER the node-type registry static is destroyed, so Node::type()
    // throws "<type> is not a valid node type" -> std::terminate/abort on every exit. _Exit skips all
    // of that; the OS reclaims memory/sockets/threads and peers detect departure via the presence
    // protocol. Only reached on a real shutdown (shutting_down_ latched above). Brief pause lets the
    // self-agent-node deletion delta reach peers first.
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
    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.high_lidar_buffer = &lidar_ingestor_.buffer();
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;

    initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    room_concept_.set_seed_pose_file(pose_path);

    auto default_viewer = find_graph_viewer("");
    if (!default_viewer)
        throw std::runtime_error("SpecificWorker requires a default DSR viewer. Enable at least one Agent viewer flag for the default graph.");

    // Load room polygon for visualizations (viewer outline + camera-projection overlay).
    std::vector<Eigen::Vector2f> room_polygon_for_viz;
    if (room_initialized_from_svg_polygon_)
        room_polygon_for_viz = rc::SvgRoomLoader::load_polygon_points(
            params.ROOM_LAYOUT_SVG, "room_contour", false, true);

    // GUI / visualization (2-D viewer, FE plot, camera-projection window + RGB media plane).
    rc::RoomViewController::Config view_cfg;
    view_cfg.grid_max_dim          = params.GRID_MAX_DIM;
    view_cfg.robot_width           = params.ROBOT_WIDTH;
    view_cfg.robot_length          = params.ROBOT_LENGTH;
    view_cfg.max_lidar_draw_points = params.MAX_LIDAR_DRAW_POINTS;
    view_cfg.has_room_polygon      = room_initialized_from_svg_polygon_;
    view_cfg.media_domain_id       = static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID);
    view_cfg.media_rgb_topic       = params.MEDIA_RGB_TOPIC;
    view_.setup(default_viewer.get(), G, view_cfg, room_polygon_for_viz, &room_concept_, &epistemic_controller_);

    if (auto* w = view_.widget())
    {
        connect(w->btn_camera_viz, &QPushButton::clicked, this, [this] { view_.show_camera(); });
        connect(w->btn_lidar_points_viz, &QPushButton::toggled, this, [this](bool on) { view_.toggle_lidar_points(on); });
        if (auto* v = view_.viewer())
            v->set_lidar_points_visible(w->btn_lidar_points_viz->isChecked());
    }

    // ── DSR scene-graph writer ─────────────────────────────────────────────
    rc::RoomGraphPublisher::Config gp_cfg;
    gp_cfg.stable_frames_required = params.STABLE_FRAMES_REQUIRED;
    gp_cfg.stable_sdf_mse_max     = params.STABLE_SDF_MSE_MAX;
    gp_cfg.stable_cov_tt_max      = params.STABLE_COV_TT_MAX;
    gp_cfg.room_height            = params.room_height;
    gp_cfg.robot_width            = params.ROBOT_WIDTH;
    gp_cfg.robot_length           = params.ROBOT_LENGTH;
    gp_cfg.robot_height           = params.ROBOT_HEIGHT;
    graph_publisher_.init(gp_cfg, {
        .graph                = G,
        .room_concept         = &room_concept_,
        .epistemic_controller = &epistemic_controller_,
        .trigger_layout       = [this] { trigger_graph_layout_twopi(); },
    });
    graph_publisher_.check_init_graph_is_valid();

    // Ensure a clean startup: if a stale room node exists from previous runs,
    // remove it so the room is recreated only after localization is stable.
    // In PreserveBootstrapRoom (static-room) mode, skip this: the room/table are a
    // pre-seeded static prior in the bootstrap graph and must NOT be deleted.
    if (params.PRESERVE_BOOTSTRAP_ROOM)
        qInfo() << "[room] PreserveBootstrapRoom=true: skipping start cleanup; adopting pre-seeded room/table as a static prior.";
    else
        graph_publisher_.cleanup_room_graph_nodes();

    // ── Connect DSR signals ────────────────────────────────────────────────
    // Laser updates use a DirectConnection so the trigger runs on the DDS emitter
    // thread (a cheap CV notify) instead of being queued onto the GUI thread. The
    // heavy point-cloud decode is done on the dedicated lidar ingestion thread
    // (lidar_ingest_loop), keeping the localization inner loop at full rate even when
    // the GUI thread is busy rendering.
    connect(G.get(), &DSR::DSRGraph::update_node_signal,      this, &SpecificWorker::modify_node_slot, Qt::DirectConnection);
    // connect(G.get(), &DSR::DSRGraph::update_edge_signal,      this, &SpecificWorker::modify_edge_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    // Start the decoupled lidar ingestion thread before the localizer so the buffer
    // is filled as soon as data arrives.
    lidar_ingestor_.init({.lidar_name = params.LIDAR_NAME, .high_min_height = params.LIDAR_HIGH_MIN_HEIGHT,
                          .media_domain_id = static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID)},
                         {.graph = G, .room_concept = &room_concept_, .qt_owner = this});
    lidar_ingestor_.start();

    room_concept_.start();

    // ── Presence coordinator ────────────────────────────────────────────────
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    AgentPresenceCoordinator::Policy presence_policy;
    presence_policy.set_local_ready_false_on_waiting_enter = false;
    presence_policy.set_local_ready_true_on_operating_enter = false;
    presence_policy.set_local_ready_false_on_degraded_enter = false;
    presence_coordinator_.set_policy(presence_policy);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
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
        .on_operating_enter = [this]()
        {
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            QTimer::singleShot(0, this, [this]() { presence_coordinator_.set_local_ready(true); });
            if (!room_concept_.is_running())
            {
                qWarning() << "[SM] Operating enter: RoomConcept thread was not running, starting it";
                room_concept_.start();
            }
        },
        .on_operating_loop = [this]()
        {
            const auto run_operating_tick = [this]()
            {
                operating_compute_queued_.store(false, std::memory_order_release);
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
    if (auto* v = view_.viewer())
    {
        connect(v, &rc::Viewer2D::robot_moved,  this, [this](QPointF p){ view_.on_robot_moved(p); });
        connect(v, &rc::Viewer2D::robot_rotate, this, [this](QPointF p){ view_.on_robot_rotated(p); });
    }

    restore_window_settings();
}

///////////////////////////////////////////////////////////////////////////////
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

    if (last_affordance_monitor_ms_ == 0 || now_ms - last_affordance_monitor_ms_ >= 200)
    {
        QElapsedTimer section_timer;
        section_timer.start();
        graph_publisher_.monitor_affordance();
        t_affordance_ms = section_timer.elapsed();
        last_affordance_monitor_ms_ = now_ms;
    }

    QElapsedTimer section_timer;
    section_timer.start();
    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;
    t_loc_fetch_ms = section_timer.elapsed();

    const Eigen::Affine2f pose_for_draw = view_.best_available_pose(loc_res, have_loc);
    
    // ── Update 2-D viewer ─────────────────────────────────────────────────
    const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    const bool use_loc = have_loc && !loc_res->lidar_scan.empty();

    std::vector<Eigen::Vector3f> lidar_for_canvas;
    if (use_loc)
        lidar_for_canvas = loc_res->lidar_scan;
    else
    {
        const auto& [lidar_from_buffer] = lidar_ingestor_.buffer().read_last();
        if (lidar_from_buffer.has_value())
            lidar_for_canvas = lidar_from_buffer->first;
    }

    const bool on_gui_thread = (QThread::currentThread() == this->thread());
    if (on_gui_thread)
    {
        section_timer.restart();
        view_.update_viewer(loc_res, have_loc, pose_for_draw, lidar_for_canvas, loc_pose, use_loc);
        t_viewer_ms = section_timer.elapsed();
    }

    // ── DSR graph update (only on fresh localization frames) ──────────────
    // In PreserveBootstrapRoom mode the room is a static prior: do NOT create/reparent the
    // room or write robot->room (that would clobber the fixed low-cov bootstrap edge). The
    // localizer still runs for the viewer; it just doesn't touch the graph.
    if (!params.PRESERVE_BOOTSTRAP_ROOM
        && have_loc && loc_res->timestamp_ms > 0 && loc_res->timestamp_ms != last_dsr_published_ts_ms_
        && (last_dsr_publish_try_ms_ == 0 || now_ms - last_dsr_publish_try_ms_ >= 200))
    {
        section_timer.restart();
        last_dsr_publish_try_ms_ = now_ms;
        graph_publisher_.update(*loc_res, last_robot_adv_speed_, last_robot_side_speed_, last_robot_rot_speed_);
        last_dsr_published_ts_ms_ = loc_res->timestamp_ms;
        t_dsr_ms = section_timer.elapsed();
    }

    if (on_gui_thread)
    {
        section_timer.restart();
        view_.update_ui(loc_res);
        t_ui_ms = section_timer.elapsed();
    }

    section_timer.restart();
    lidar_ingestor_.tick_health(now_ms, on_gui_thread);
    t_health_ms = section_timer.elapsed();

    const auto total_ms = compute_timer.elapsed();
    const auto elapsed_since_init_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - init_time).count();
    if (last_compute_timing_log_ms_ == 0 || now_ms - last_compute_timing_log_ms_ >= 3000 || total_ms > 50)
    {
        last_compute_timing_log_ms_ = now_ms;
        qInfo() << "[Timing][compute]"
                << "total_ms=" << elapsed_since_init_ms
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
    const auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        params.ROOM_LAYOUT_SVG, "room_contour", false, true);
    if (room_polygon.size() >= 3)
    {
        room_concept_.configure_room_from_polygon(room_polygon);
        room_initialized_from_svg_polygon_ = true;
        return;
    }
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

    const bool touches_lidar_payload = touches_any({
        "laser_X",
        "laser_Y",
        "laser_Z",
        "laser_timestamp"
    });

    if (touches_lidar_payload)
        lidar_ingestor_.on_laser_attrs_signal(id);

    if (const auto robot_id = graph_publisher_.robot_id(); robot_id != 0 && id != robot_id)
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
                            velocity_buffer_.put<0>(std::move(cmd), source_ts);
                            last_robot_ref_speed_timestamp_ = source_ts;
                        }
                    }
                }
            }
        }
    }
}


void SpecificWorker::modify_node_slot(std::uint64_t /*id*/, const std::string &type)
{
    if (type != "laser")
        return;
    lidar_ingestor_.on_laser_node_signal();
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
