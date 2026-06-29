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
#include <fstream>   // [perf-probe] CSV timing logs (remove with the probes)
#include "scene_processor.h"
#include "voxel_processor.h"
#include "yolo_processor.h"
#include "yolo_human.h"
#include "graph_publisher.h"
#ifdef emit
#undef emit
#endif
#include "unified_voxel_grid.h"
#include "voxel_opengl_viewer.h"
#include "yolo_viewer.h"
#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <sstream>
#include <unordered_map>

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
        if (error.length() > 0)
        {
            qWarning() << error;
            throw error;
        }
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    qInfo() << "Destroying SpecificWorker";
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_external_window_geometry();
    scene_processor.reset();
    cleanup_owned_nodes();
}

void SpecificWorker::initialize()
{
    qInfo() << "initialize voxelizer worker";
    GenericWorker::initialize();

    // --- Configuration ---
    params = load_voxelizer_params(configLoader);
    verbose_debug_ = params.VERBOSE_DEBUG;

    // Cap OpenCV's implicit thread pool. The per-cycle preprocessing (resize/split/convertTo of the
    // YOLO frames) otherwise fans out across ALL cores in short bursts, inflating the process's
    // core-count for negligible latency gain on a 640² tensor. 2 threads is plenty here.
    cv::setNumThreads(2);

    // --- YOLO ---
    yolo_processor = std::make_unique<YoloProcessor>();
    YoloProcessor::Config yolo_config;
    yolo_config.model_path          = params.YOLO_MODEL_PATH;
    yolo_config.conf_thresh         = params.YOLO_CONF_THRESH;
    yolo_config.iou_thresh          = params.YOLO_IOU_THRESH;
    yolo_config.input_size          = params.YOLO_INPUT_SIZE;
    yolo_config.use_gpu             = params.YOLO_USE_GPU;
    yolo_config.use_trt             = params.YOLO_USE_TRT;
    yolo_config.mask_erode_kernel   = params.YOLO_MASK_ERODE_KERNEL;
    yolo_config.mask_tray           = params.YOLO_MASK_TRAY;
    yolo_config.tray_mask_ref_width = params.YOLO_TRAY_MASK_REF_WIDTH;
    yolo_config.tray_mask_ref_height= params.YOLO_TRAY_MASK_REF_HEIGHT;
    yolo_config.tray_mask_polygon_px= params.YOLO_TRAY_MASK_POLYGON_PX;
    yolo_config.tray_drop_fraction  = params.YOLO_TRAY_DROP_FRACTION;
    yolo_config.accepted_labels     = params.YOLO_ACCEPTED_LABELS;
    yolo_config.verbose_debug       = verbose_debug_;
    yolo_processor->configure(yolo_config);

    // --- Human pose (optional second model → BODY_18 skeletons on the 'skeleton' node) ---
    if (params.HUMAN_POSE_ENABLED)
    {
        try
        {
            yolo_human_processor = std::make_unique<rc::human_pose::YoloHumanProcessor>();
            rc::human_pose::YoloHumanProcessor::Config pose_config;
            pose_config.model_path    = params.HUMAN_POSE_MODEL_PATH;
            pose_config.conf_thresh   = params.HUMAN_POSE_CONF_THRESH;
            pose_config.iou_thresh    = params.HUMAN_POSE_IOU_THRESH;
            pose_config.input_size    = params.HUMAN_POSE_INPUT_SIZE;
            pose_config.use_gpu       = params.HUMAN_POSE_USE_GPU;
            pose_config.use_trt       = params.HUMAN_POSE_USE_TRT;
            pose_config.verbose_debug = verbose_debug_;
            yolo_human_processor->configure(pose_config);
        }
        catch (const std::exception& e)
        {
            qWarning() << "[HumanPose] disabled — failed to load model" << params.HUMAN_POSE_MODEL_PATH.c_str()
                       << ":" << e.what();
            yolo_human_processor.reset();
        }
    }

    // --- Lidar secondary attributor (silent no-op when lidar3D data is absent) ---
    lidar_track_attributor = std::make_unique<LidarTrackAttributor>();

    // Custom drawing windows (Voxel3D GL + YOLO raster), each in its own top-level window — see
    // specificworker_viewers.cpp. Both config-gated (Voxel.show_voxel_viewer / show_yolo_viewer).
    setup_custom_viewers();

    // --- Voxel grid + processor ---
    UnifiedGridConfig voxel_grid_config;
    voxel_grid_config.max_display_voxels = static_cast<int>(params.VOXEL_VIEWER_MAX_RENDERED_VOXELS);
    voxel_grid = std::make_unique<UnifiedVoxelGrid>(voxel_grid_config);
    voxel_processor = std::make_unique<VoxelProcessor>(*voxel_grid);
    VoxelProcessor::Config voxel_processor_config;
    voxel_processor_config.voxel_decimation_factor         = params.VOXEL_DECIMATION_FACTOR;
    voxel_processor_config.viewer_max_rendered_voxels      = params.VOXEL_VIEWER_MAX_RENDERED_VOXELS;
    voxel_processor_config.track_association_max_distance_m= params.TRACK_ASSOCIATION_MAX_DISTANCE_M;
    voxel_processor_config.track_max_missed_frames         = params.TRACK_MAX_MISSED_FRAMES;
    voxel_processor_config.viewer_voxel_fps                = params.VOXEL_VIEWER_FPS;
    voxel_processor_config.z_lift_m                        = params.VOXEL_Z_LIFT_M;
    voxel_processor_config.mask_surface_band_m             = params.MASK_SURFACE_BAND_M;
    voxel_processor_config.verbose_debug                   = verbose_debug_;
    voxel_processor->configure(voxel_processor_config);

    // --- Scene processor (DSR-only: no proxies) ---
    inner_eigen_api = G->get_inner_eigen_api();
    scene_processor = std::make_unique<SceneProcessor>(G);
    scene_processor->configure(inner_eigen_api.get(), voxel_viewer_gl.get(),
                               params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_,
                               params.MASK_POSE_EXTRAPOLATE, params.MASK_POSE_EXTRAP_MAX_DT_S);
    scene_processor->init_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                      params.MEDIA_RGB_TOPIC, params.MEDIA_DEPTH_TOPIC);
    scene_processor->init_lidar_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                            params.MEDIA_LIDAR_TOPIC, params.LIDAR_USE_MEDIA);

    // All DSR semantic_grid exports (masks always; tracks/voxels config-gated). Relayout is injected
    // so the publisher stays decoupled from the GUI (graph_viewers).
    graph_publisher_ = std::make_unique<GraphPublisher>(
        G, params, voxel_processor.get(), voxel_grid.get(),
        [this]() { trigger_graph_layout_twopi(); });

    graph_publisher_->cleanup_semantic_grid_nodes();

    // Decoupled render timer: only worth running when the 3D viewer exists. 50 ms (20 Hz) matches the
    // viewer's repaint throttle, giving a stable 20 Hz refresh without re-running perception faster.
    // Created stopped; started in Operating (so graph reads happen only after the join completes).
    if (voxel_viewer_gl)
    {
        render_timer_ = std::make_unique<QTimer>(this);
        render_timer_->setInterval(50);
        QObject::connect(render_timer_.get(), &QTimer::timeout, this, &SpecificWorker::on_render_tick);
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { Q_EMIT presenceReady(); },
        .request_presence_lost  = [this]() { Q_EMIT presenceLost(); },
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
            if (render_timer_)
                render_timer_->start();   // start fluid viewer refresh only once the graph is joined
        },
        .on_operating_loop = [this]()
        {
            compute();
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = [this]()
        {
            qInfo() << "[SM] -> Degraded: a required peer or node is no longer available";
            if (render_timer_)
                render_timer_->stop();   // no graph access outside Operating
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    restore_window_settings();
}

// Decoupled viewer refresh (GUI thread, Operating only). Reads the LATEST robot pose from the graph
// and pushes it to the 3D viewer, independent of the perception/camera pipeline, so robot motion is
// fluid. set_robot_pose() triggers the viewer's own throttled repaint — no explicit update() here.
void SpecificWorker::on_render_tick()
{
    if (shutting_down_ || !scene_processor)
        return;
    scene_processor->refresh_viewer_robot_pose_latest();
}

void SpecificWorker::compute()
{
    if (!scene_processor or !voxel_processor)
        return;

    const auto frame = process_scene_frame(fps_counter_);

    // Budget regulation is a heartbeat: run it every cycle (self-throttled), even when no RGBD frame
    // arrived, so the cap/gauge keep working while the robot explores away from the table.
    if (compute_voxels_)
        regulate_voxel_budget(fps_counter_.get_frequency());

    // No RGBD frame this cycle (sensor not ready / gated transforms). Everything below dereferences
    // `frame`, so we MUST stop here — a fall-through hit frame->rgbd on an empty optional → SEGV.
    if (!frame.has_value())
        return;

    // Follow the RGB stream's REAL rate: the media cache repeats the last frame when nothing new
    // arrived this cycle. Skip the RGB-derived work (YOLO, pose, viewer, mask publish) on stale
    // repeats so the displayed FPS and published masks track the camera's actual delivery rate. The
    // lidar/robot-pose viewer updates already ran in process_scene_frame and keep the compute cadence.
    if (frame->frame_ts_ms == last_rgb_ts_)
        return;
    last_rgb_ts_ = frame->frame_ts_ms;

    // Tray-mask the frame ONCE and reuse it for both detection and the viewer overlay (one full-frame
    // clone/cycle instead of two). update_frame() clones internally and only reads, so sharing is safe.
    const cv::Mat masked_rgb = yolo_processor
        ? yolo_processor->apply_tray_mask(frame->rgbd.rgb)
        : frame->rgbd.rgb;
    const auto detections = yolo_processor
        ? yolo_processor->detect_segmentation_on(masked_rgb)
        : std::vector<SegDetection>{};

    // Human-pose: the MODEL runs decimated (every kPoseDecimation-th cycle — people don't move at
    // 20 Hz), but the viewer redraws the LAST cached detection every frame so the skeleton overlay
    // doesn't flicker between detections.
    const bool run_pose = yolo_human_processor and yolo_human_processor->ready()
                          and (pose_frame_counter_++ % kPoseDecimation == 0);
    if (run_pose)
        yolo_human_processor->detect_poses(frame->rgbd.rgb);   // refresh the cache (~6-7 Hz)
    static const std::vector<rc::human_pose::PoseDetection> kNoPoses;
    const auto& poses = (yolo_human_processor and yolo_human_processor->ready())
        ? yolo_human_processor->last_poses()
        : kNoPoses;

    // Voxel computation gated OFF for now — we only run the masks pipeline (YOLO detections →
    // graph_publisher publishes the "masks" support points, consumed by table/bottle_concept). The
    // voxel grid build + lidar→voxel fusion are skipped. Flip compute_voxels_ to re-enable.
    if (compute_voxels_)
    {
        voxel_processor->process_rgbd_frame(frame->rgbd, detections,
                                             frame->room_T_robot, frame->room_T_zed,
                                             frame->graph_object_boxes, voxel_viewer_gl.get());

        if (include_lidar3d_in_voxels_ && lidar_track_attributor && !frame->lidar_points_room.empty())
        {
            std::vector<LidarTrackAttributor::TrackCandidate> track_candidates;
            const auto& current_tracks = voxel_processor->last_track_candidates();
            track_candidates.reserve(current_tracks.size());
            for (const auto& track : current_tracks)
            {
                track_candidates.push_back(LidarTrackAttributor::TrackCandidate{
                    .track_id = track.track_id,
                    .category = track.category,
                    .min = track.min,
                    .max = track.max,
                    .centroid = track.centroid
                });
            }

            auto attributed = lidar_track_attributor->attribute_points(frame->lidar_points_room, track_candidates);
            voxel_processor->fuse_lidar_support_points(attributed);
        }
    }

    if (yolo_viewer_)
    {
        cv::Mat viewer_rgb = masked_rgb;   // already tray-masked above — no extra clone
        // Draw the detected skeletons (green bones, red joints, orange bbox) under the seg overlay.
        if (yolo_human_processor and not poses.empty())
            viewer_rgb = yolo_human_processor->compose_pose_canvas(viewer_rgb, poses);
        yolo_viewer_->update_frame(viewer_rgb, detections);
        // Size the RGB window to the image once (only when no saved geometry was restored).
        if (yolo_window_needs_image_size_ and yolo_window_ != nullptr and not viewer_rgb.empty())
        {
            yolo_window_->resize(viewer_rgb.cols, viewer_rgb.rows);
            yolo_window_needs_image_size_ = false;
        }
    }

    graph_publisher_->publish(frame->rgbd, frame->room_T_zed, detections, frame->frame_ts_ms);

    // Human-pose branch: BODY_18 skeletons (camera frame) on the 'skeleton' node for human_concept.
    // Only on cycles we actually ran the pose model (decimated above).
    if (run_pose)
        graph_publisher_->publish_skeletons(frame->rgbd, poses, frame->frame_ts_ms);

    fps_counter_.print("[Compute]", 3000);
}

// Homeostatic FPS regulation: once per control period, read the measured compute
// FPS and let the AIMD regulator resize the voxel-grid population cap so the
// frame-rate holds its target band. Self-throttled — safe to call every cycle.
void SpecificWorker::regulate_voxel_budget(float fps)
{
    if (!voxel_grid)
        return;

    // Self-throttle to one control tick per period (regardless of compute rate).
    constexpr auto kControlPeriod = std::chrono::seconds(2);
    const auto now = std::chrono::steady_clock::now();
    if (last_budget_tick_.time_since_epoch().count() != 0 &&
        now - last_budget_tick_ < kControlPeriod)
        return;
    last_budget_tick_ = now;

    const int live = voxel_grid->size();

    if (!std::isfinite(fps) || fps <= 0.0f)   // FPS not measured yet (first ~3 s) → hold, but show life
    {
        std::println("[VoxelBudget] warming up (no fps yet)  cap={} live={}",
                     voxel_grid->max_voxels(), live);
        return;
    }

    const int new_cap = voxel_budget_.update(fps);
    voxel_grid->set_max_voxels(new_cap);

    std::println("[VoxelBudget] fps={:.1f} target={:.1f} cap={} live={} [{}..{}]",
                 fps, voxel_budget_.target_fps, new_cap, live,
                 voxel_budget_.floor, voxel_budget_.ceiling);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<SpecificWorker::SceneFrame> SpecificWorker::process_scene_frame(FPSCounter& compute_fps)
{
    scene_processor->check_input_stream_startup_status();
    const auto [room_name, robot_name] = scene_processor->get_room_robot_names_for_compute();

    // Diagnostic: throttled (every ~2s) report of which gate drops the frame.
    static auto last_gate_report = std::chrono::steady_clock::now();
    const auto gate_log = [&](const char* gate)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_gate_report >= std::chrono::seconds(2))
        {
            std::println("[SceneGate] dropped at '{}' (room='{}' robot='{}')", gate, room_name, robot_name);
            last_gate_report = now;
        }
    };

    const auto rgbd_opt = scene_processor->get_rgbd_frame_from_dsr();
    if (!rgbd_opt.has_value())
        { gate_log("get_rgbd_frame"); return std::nullopt; }
    const std::uint64_t frame_ts_ms = scene_processor->get_frame_timestamp_ms();

    if (!scene_processor->ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
        { gate_log("ensure_room_and_robot_ready"); return std::nullopt; }

    const auto room_T_robot = scene_processor->get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
    if (!room_T_robot.has_value())
        { gate_log("get_room_robot_transform"); return std::nullopt; }
    const auto room_T_zed = scene_processor->get_room_zed_transform(compute_fps, robot_name, room_T_robot.value());
    if (!room_T_zed.has_value())
        { gate_log("get_room_zed_transform"); return std::nullopt; }

    scene_processor->log_room_robot_pose_periodic(room_T_robot.value());
    scene_processor->mark_room_rt_ready();
    const auto graph_object_boxes = scene_processor->get_graph_object_boxes(room_name, frame_ts_ms);

    std::vector<Eigen::Vector3f> lidar_points_room;
    if (auto lidar_data = scene_processor->get_lidar3D(); lidar_data.has_value())
    {
        Mat::RTMat room_T_robot_lidar = room_T_robot.value();
        if (inner_eigen_api != nullptr && lidar_data->timestamp_ms > 0)
        {
            const auto time_query = params.TRANSFORMS_INTERPOLATE_RT
                ? DSR::RT_API::TimeQuery::Interpolated
                : DSR::RT_API::TimeQuery::Nearest;
            if (auto interpolated = inner_eigen_api->get_transformation_matrix(
                    room_name,
                    robot_name,
                    lidar_data->timestamp_ms,
                    "RT",
                    time_query); interpolated.has_value())
            {
                room_T_robot_lidar = interpolated.value();
            }
        }

        const std::size_t count = std::min({lidar_data->xs.size(), lidar_data->ys.size(), lidar_data->zs.size()});
        lidar_points_room.reserve(count);
        const Eigen::Matrix3f room_rotation = room_T_robot_lidar.linear().cast<float>();
        const Eigen::Vector3f room_translation = room_T_robot_lidar.translation().cast<float>();
        for (std::size_t i = 0; i < count; ++i)
        {
            const Eigen::Vector3f point_robot(lidar_data->xs[i], lidar_data->ys[i], lidar_data->zs[i]);
            lidar_points_room.emplace_back(room_rotation * point_robot + room_translation);
        }
    }

    scene_processor->update_viewer_robot_pose(room_T_robot.value());
    scene_processor->update_viewer_lidar_points(room_name, robot_name, room_T_robot.value());
    scene_processor->update_viewer_graph_object_boxes(graph_object_boxes);
    scene_processor->update_viewer_object_meshes();
    scene_processor->update_viewer_person_skeletons();
    scene_processor->update_viewer_table_rfe_points();
    scene_processor->update_viewer_mask_points();
    scene_processor->update_room_polygon_periodic();

    return SceneFrame{rgbd_opt.value(),
                      room_T_robot.value(),
                      room_T_zed.value(),
                      std::move(lidar_points_room),
                      graph_object_boxes,
                      frame_ts_ms};
}

void SpecificWorker::emergency()
{
    qInfo() << "Emergency worker";
}

void SpecificWorker::restore()
{
    qInfo() << "Restore worker";
}

int SpecificWorker::startup_check()
{
    qInfo() << "Startup check";
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}

void SpecificWorker::trigger_graph_layout_twopi()
{
    const auto it = graph_viewers.find("");
    if (it == graph_viewers.end() || !it->second)
        return;

    QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (!graph_viewer)
        return;

    // Run now and once queued, so layout also happens after pending node/edge
    // update signals are processed by the viewer.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer]() { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}


