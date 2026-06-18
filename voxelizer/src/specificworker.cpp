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
#include "scene_processor.h"
#include "voxel_processor.h"
#include "yolo_processor.h"
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
    scene_processor.reset();
    cleanup_owned_nodes();
}

void SpecificWorker::initialize()
{
    qInfo() << "initialize voxelizer worker";
    GenericWorker::initialize();

    // --- Configuration ---
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.model_path", params.YOLO_MODEL_PATH);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.accepted_labels", params.YOLO_ACCEPTED_LABELS);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Yolo.conf_thresh", params.YOLO_CONF_THRESH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Yolo.iou_thresh", params.YOLO_IOU_THRESH);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.use_gpu", params.YOLO_USE_GPU);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.use_trt", params.YOLO_USE_TRT);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.mask_erode_kernel", params.YOLO_MASK_ERODE_KERNEL);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.mask_tray", params.YOLO_MASK_TRAY);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.tray_mask_ref_width", params.YOLO_TRAY_MASK_REF_WIDTH);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.tray_mask_ref_height", params.YOLO_TRAY_MASK_REF_HEIGHT);
    rc::ConfigLoaderUtils::load_optional_apply<std::vector<int>>(configLoader, "Yolo.tray_mask_polygon",
        [&](const std::vector<int>& flat)
        {
            if (flat.size() >= 6 && flat.size() % 2 == 0)
            {
                params.YOLO_TRAY_MASK_POLYGON_PX.clear();
                for (std::size_t i = 0; i + 1 < flat.size(); i += 2)
                    params.YOLO_TRAY_MASK_POLYGON_PX.emplace_back(flat[i], flat[i + 1]);
            }
        });
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Yolo.track_association_max_distance_m", params.TRACK_ASSOCIATION_MAX_DISTANCE_M);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Yolo.track_max_missed_frames", params.TRACK_MAX_MISSED_FRAMES);
    rc::ConfigLoaderUtils::load_optional<std::size_t, int>(configLoader, "Voxel.viewer_max_rendered_voxels", params.VOXEL_VIEWER_MAX_RENDERED_VOXELS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Voxel.viewer_fps", params.VOXEL_VIEWER_FPS);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.z_lift_m", params.VOXEL_Z_LIFT_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.mask_depth_gate_band_m", params.MASK_DEPTH_GATE_BAND_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.mask_outlier_radius_m", params.MASK_OUTLIER_RADIUS_M);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "Voxel.mask_outlier_min_neighbors", params.MASK_OUTLIER_MIN_NEIGHBORS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Transforms.interpolate_rt", params.TRANSFORMS_INTERPOLATE_RT);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.domain_id", params.MEDIA_DOMAIN_ID);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.rgb_topic", params.MEDIA_RGB_TOPIC);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.depth_topic", params.MEDIA_DEPTH_TOPIC);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Component.Debug.Verbose", verbose_debug_);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Debug.verbose", verbose_debug_);

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
    yolo_config.accepted_labels     = params.YOLO_ACCEPTED_LABELS;
    yolo_config.verbose_debug       = verbose_debug_;
    yolo_processor->configure(yolo_config);

    // --- Lidar secondary attributor (silent no-op when lidar3D data is absent) ---
    lidar_track_attributor = std::make_unique<LidarTrackAttributor>();

    // --- Voxel viewer (attaches to DSR GUI if available) ---
    if (!graph_viewers.empty())
    {
        const std::string viewer_key = graph_viewers.contains("")
            ? std::string("") : graph_viewers.begin()->first;

        auto* voxel_panel = new QWidget(nullptr);
        auto* panel_layout = new QVBoxLayout(voxel_panel);
        panel_layout->setContentsMargins(6, 6, 6, 6);
        panel_layout->setSpacing(6);

        auto* controls_layout = new QHBoxLayout();
        controls_layout->setContentsMargins(0, 0, 0, 0);
        controls_layout->setSpacing(8);

        auto* lidar_btn = new QPushButton("Lidar: OFF", voxel_panel);
        lidar_btn->setCheckable(true);
        lidar_btn->setCursor(Qt::PointingHandCursor);

        auto* lidar_voxels_btn = new QPushButton("Lidar3D Voxels: ON", voxel_panel);
        lidar_voxels_btn->setCheckable(true);
        lidar_voxels_btn->setChecked(include_lidar3d_in_voxels_);
        lidar_voxels_btn->setCursor(Qt::PointingHandCursor);

        auto* masks_btn = new QPushButton("Masks: OFF", voxel_panel);
        masks_btn->setCheckable(true);
        masks_btn->setCursor(Qt::PointingHandCursor);

        auto* clear_voxels_btn = new QPushButton("Clear Voxels", voxel_panel);
        clear_voxels_btn->setCursor(Qt::PointingHandCursor);

        controls_layout->addWidget(lidar_btn);
        controls_layout->addWidget(lidar_voxels_btn);
        controls_layout->addWidget(masks_btn);
        controls_layout->addWidget(clear_voxels_btn);
        controls_layout->addStretch(1);

        voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(nullptr);
        voxel_viewer_gl->load_robot_mesh("meshes/shadow.obj");

        panel_layout->addLayout(controls_layout);
        panel_layout->addWidget(voxel_viewer_gl.get(), 1);

        connect(lidar_btn, &QPushButton::toggled, this, [this, lidar_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_lidar(checked);
            lidar_btn->setText(checked ? "Lidar: ON" : "Lidar: OFF");
        });

        connect(lidar_voxels_btn, &QPushButton::toggled, this, [this, lidar_voxels_btn](bool checked)
        {
            include_lidar3d_in_voxels_ = checked;
            lidar_voxels_btn->setText(checked ? "Lidar3D Voxels: ON" : "Lidar3D Voxels: OFF");
        });

        connect(masks_btn, &QPushButton::toggled, this, [this, masks_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_masks(checked);
            masks_btn->setText(checked ? "Masks: ON" : "Masks: OFF");
        });

        connect(clear_voxels_btn, &QPushButton::clicked, this, [this]
        {
            if (!voxel_processor || !voxel_viewer_gl)
                return;
            voxel_processor->clear_state(voxel_viewer_gl.get());
            ensure_voxels_node_in_dsr();
            upload_voxel_grid_to_dsr();
        });

        graph_viewers.at(viewer_key)->add_custom_widget_to_dock("Voxel3D", voxel_panel);
        yolo_viewer_ = std::make_unique<rc::YoloViewer>(nullptr);
        graph_viewers.at(viewer_key)->add_custom_widget_to_dock("YOLO", yolo_viewer_.get());
        qInfo() << __FUNCTION__ << "Voxel3D viewer attached to DSR graph viewer";
    }

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
    voxel_processor_config.verbose_debug                   = verbose_debug_;
    voxel_processor->configure(voxel_processor_config);

    // --- Scene processor (DSR-only: no proxies) ---
    inner_eigen_api = G->get_inner_eigen_api();
    scene_processor = std::make_unique<SceneProcessor>(G);
    scene_processor->configure(inner_eigen_api.get(), voxel_viewer_gl.get(),
                               params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_);
    scene_processor->init_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                      params.MEDIA_RGB_TOPIC, params.MEDIA_DEPTH_TOPIC);

    cleanup_semantic_grid_nodes();

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
        .on_operating_enter = []()
        {
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
        },
        .on_operating_loop = [this]()
        {
            compute();
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = []()
        {
            qInfo() << "[SM] -> Degraded: a required peer or node is no longer available";
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    restore_window_settings();
}

void SpecificWorker::compute()
{
    if (!scene_processor or !voxel_processor)
        return;

    const auto frame = process_scene_frame(fps_counter_);

    // Budget regulation is a heartbeat: run it every cycle (self-throttled),
    // even when no RGBD frame arrived, so the cap/gauge keep working while the
    // robot explores away from the table. FPS comes from the shared counter,
    // which scene_processor also ticks on the not-ready path.
    qInfo() << fps_counter_.get_frequency();
    regulate_voxel_budget(fps_counter_.get_frequency());

    // No RGBD frame this cycle (sensor not ready / gated transforms). The budget heartbeat
    // above already ran; everything below dereferences `frame`, so we MUST stop here.
    // Falling through accessed frame->rgbd on an empty optional → garbage cv::Mat → SEGV in
    // cv::Mat::release() and heap corruption that later crashed paintAndFlush.
    if (!frame.has_value())
    {
        qWarning() << __FUNCTION__ << "frame has no value";
        return;
    }

    const auto detections = yolo_processor
        ? yolo_processor->detect_segmentation(frame->rgbd.rgb)
        : std::vector<SegDetection>{};
  
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

    if (yolo_viewer_) 
    {    
        const cv::Mat viewer_rgb = yolo_processor
            ? yolo_processor->apply_tray_mask(frame->rgbd.rgb)
            : frame->rgbd.rgb;
        yolo_viewer_->update_frame(viewer_rgb, detections);
    }

    ensure_masks_node_in_dsr();
    upload_masks_to_dsr(*frame, detections);

    ensure_tracks_node_in_dsr();
    publish_tracks_to_dsr();

    ensure_voxels_node_in_dsr();
    upload_voxel_grid_to_dsr();

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
    if (auto lidar_data = scene_processor->get_lidar3D_from_dsr(); lidar_data.has_value())
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
    scene_processor->update_viewer_table_rfe_points();
    scene_processor->update_viewer_mask_points();
    scene_processor->update_room_polygon_periodic();

    return SceneFrame{rgbd_opt.value(),
                      room_T_robot.value(),
                      room_T_zed.value(),
                      std::move(lidar_points_room),
                      graph_object_boxes};
}

void SpecificWorker::ensure_tracks_node_in_dsr()
{
    if (tracks_node_ready_)
        return;

    if (G->get_nodes_by_type("room").empty() or G->get_nodes_by_type("robot").empty())
        return;

    if (G->get_node("tracks").has_value())
    {
        tracks_node_ready_ = true;
        return;
    }

    auto zed_node = G->get_node("zed");
    if (!zed_node.has_value())
    {
        qWarning() << "[Tracks] WARNING: 'zed' node not found in graph — cannot create RT edge";
        return;
    }

    auto tracks_node = DSR::Node::create<semantic_grid_node_type>("tracks");
    G->add_or_modify_attrib_local<color_att>(tracks_node, std::string{"SteelBlue"});
    G->add_or_modify_attrib_local<level_att>(tracks_node, 4);
    G->add_or_modify_attrib_local<parent_att>(tracks_node, zed_node.value().id());
    G->add_or_modify_attrib_local<pos_x_att>(tracks_node, 105.849792f);
    G->add_or_modify_attrib_local<pos_y_att>(tracks_node, 291.904266f);

    if (const auto id = G->insert_node(tracks_node); id.has_value())
    {
        tracks_node_ready_ = true;
        qInfo() << "[Tracks] Created 'tracks' node id=" << *id << " (semantic_grid) under room '" << zed_node.value().name().c_str() << "'";

        auto rt_api = G->get_rt_api();
        rt_api->insert_or_assign_edge_RT(zed_node.value(), *id,
                                             {0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f});
        qInfo() << "[Tracks] RT edge inserted from 'zed' -> 'tracks'";
        // NOTE: deliberately NOT calling trigger_graph_layout_twopi() here. A full
        // twopi relayout reshuffles every graph item and, when it overlaps with
        // another agent (table_concept) inserting/removing nodes, the graph viewer
        // can paint a freed QGraphicsItem → SIGSEGV in QRegion/flush. Cosmetic only.
    }
    else
        qWarning() << "[Tracks] ERROR: failed to insert 'tracks' node into DSR graph";
}

// Feed-forward, class-agnostic publisher. Exports ALL current voxel tracks (every
// category) into the 'tracks' node, room frame, as a struct-of-arrays keyed by
// persistent track id. The voxelizer no longer reads any concept's model box;
// concept agents read these tracks and do their own instance assignment.
// Uses runtime/dynamic attributes (like the 'masks' node) — no REGISTER_TYPE.
void SpecificWorker::publish_tracks_to_dsr()
{
    if (!tracks_node_ready_)
        return;

    auto node_opt = G->get_node("tracks");
    if (!node_opt.has_value())
        return;
    auto& node = node_opt.value();

    const auto& tracks        = voxel_processor->last_track_candidates();
    const int   sensing_frame = voxel_processor->last_frame_id();
    constexpr int kMaxPtsPerTrack = 400;

    std::vector<float> ids, label_ids, confidences, last_seen, voxel_counts;
    std::vector<float> centroids_xyz, bbox_min_xyz, bbox_max_xyz;
    std::vector<float> support_offsets{0.0f};   // prefix offsets, in POINTS
    std::vector<float> support_points;
    std::ostringstream labels_joined;
    int published = 0;

    for (const auto& t : tracks)
    {
        const auto pts = voxel_processor->get_flat_pts_for_track(t.track_id, kMaxPtsPerTrack);
        if (pts.empty())
            continue;

        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << t.category;

        ids.push_back(static_cast<float>(t.track_id));
        label_ids.push_back(0.0f);                 // class id slot (category string carries the label)
        confidences.push_back(1.0f);               // label-vote confidence (placeholder)
        last_seen.push_back(static_cast<float>(t.last_seen_frame));
        voxel_counts.push_back(static_cast<float>(t.voxel_count));

        centroids_xyz.insert(centroids_xyz.end(), {t.centroid.x(), t.centroid.y(), t.centroid.z()});
        bbox_min_xyz.insert(bbox_min_xyz.end(),   {t.min.x(), t.min.y(), t.min.z()});
        bbox_max_xyz.insert(bbox_max_xyz.end(),   {t.max.x(), t.max.y(), t.max.z()});

        support_points.insert(support_points.end(), pts.begin(), pts.end());
        support_offsets.push_back(static_cast<float>(support_points.size() / 3));
        ++published;
    }

    G->runtime_checked_add_or_modify_attrib_local(node, "track_frame_id",      sensing_frame);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_count",         published);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_ids",           ids);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_labels",        labels_joined.str());
    G->runtime_checked_add_or_modify_attrib_local(node, "track_label_ids",     label_ids);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_confidences",   confidences);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_last_seen",     last_seen);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_voxel_counts",  voxel_counts);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_centroids_xyz", centroids_xyz);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_bbox_min_xyz",  bbox_min_xyz);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_bbox_max_xyz",  bbox_max_xyz);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_support_offsets", support_offsets);
    G->runtime_checked_add_or_modify_attrib_local(node, "track_support_points",  support_points);
    G->update_node(std::move(node));

    if (sensing_frame % 30 == 0)
        std::println("[Tracks] frame={} published={} support_pts={}",
                     sensing_frame, published, support_points.size() / 3);
}

void SpecificWorker::ensure_voxels_node_in_dsr()
{
    if (voxels_node_ready_)
        return;

    // Require both room and robot nodes to be present before creating the voxels node.
    if (G->get_nodes_by_type("room").empty() or G->get_nodes_by_type("robot").empty())
        return;

    // If the node already exists (created by another agent or a previous run), mark ready and return.
    if (G->get_node("voxels").has_value())
    {
        voxels_node_ready_ = true;
        return;
    }

    auto zed_node = G->get_node("zed");
    if (!zed_node.has_value())
    {
        qWarning() << "[Voxels] WARNING: 'zed' node not found in graph — cannot create RT edge";
        return;
    }

    auto voxels_node = DSR::Node::create<semantic_grid_node_type>("voxels");
    G->add_or_modify_attrib_local<color_att>(voxels_node, std::string{"Khaki"});
    G->add_or_modify_attrib_local<level_att>(voxels_node, 4);
    G->add_or_modify_attrib_local<parent_att>(voxels_node, zed_node.value().id());
    G->add_or_modify_attrib_local<pos_x_att>(voxels_node, 105.849792f);
    G->add_or_modify_attrib_local<pos_y_att>(voxels_node, 291.904266f);

    if (const auto id = G->insert_node(voxels_node); id.has_value())
    {
        voxels_node_ready_ = true;
        qInfo() << "[Voxels] Created 'voxels' node id=" << *id << " (semantic_grid) under room '" << zed_node.value().name().c_str() << "'";

        // Add an RT edge from "zed" to the new "voxels" node (identity transform).
        auto rt_api = G->get_rt_api();
        rt_api->insert_or_assign_edge_RT(zed_node.value(), *id,
                                             {0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f});
        qInfo() << "[Voxels] RT edge inserted from 'zed' -> 'voxels'";
        trigger_graph_layout_twopi();
    }
    else
        qWarning() << "[Voxels] ERROR: failed to insert 'voxels' node into DSR graph";
}

void SpecificWorker::ensure_masks_node_in_dsr()
{
    if (masks_node_ready_)
        return;

    if (G->get_nodes_by_type("room").empty() or G->get_nodes_by_type("robot").empty())
        return;

    if (G->get_node("masks").has_value())
    {
        masks_node_ready_ = true;
        return;
    }

    auto zed_node = G->get_node("zed");
    if (!zed_node.has_value())
    {
        qWarning() << "[Masks] WARNING: 'zed' node not found in graph — cannot create RT edge";
        return;
    }

    auto masks_node = DSR::Node::create<semantic_grid_node_type>("masks");
    G->add_or_modify_attrib_local<color_att>(masks_node, std::string{"Plum"});
    G->add_or_modify_attrib_local<level_att>(masks_node, 4);
    G->add_or_modify_attrib_local<parent_att>(masks_node, zed_node.value().id());
    G->add_or_modify_attrib_local<pos_x_att>(masks_node, 105.849792f);
    G->add_or_modify_attrib_local<pos_y_att>(masks_node, 291.904266f);

    if (const auto id = G->insert_node(masks_node); id.has_value())
    {
        masks_node_ready_ = true;
        qInfo() << "[Masks] Created 'masks' node id=" << *id << " (semantic_grid) under room '" << zed_node.value().name().c_str() << "'";

        auto rt_api = G->get_rt_api();
        rt_api->insert_or_assign_edge_RT(zed_node.value(), *id,
                                             {0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f});
        qInfo() << "[Masks] RT edge inserted from 'zed' -> 'masks'";
        trigger_graph_layout_twopi();
    }
    else
        qWarning() << "[Masks] ERROR: failed to insert 'masks' node into DSR graph";
}

void SpecificWorker::upload_voxel_grid_to_dsr()
{
    if (!voxels_node_ready_)
        return;

    const int sensing_frame = voxel_processor->last_frame_id();
    if (sensing_frame % 30 != 0)
        return;

    auto voxels_node_opt = G->get_node("voxels");
    if (!voxels_node_opt.has_value())
        return;
    auto& voxels_node = voxels_node_opt.value();

    const auto exported = voxel_grid->export_semantic_voxels();

    // Stride-5 encoding per voxel: [x, y, z, prob, track_id].
    // Receivers must interpret with stride 5.
    const std::size_t n = exported.points.size();
    std::vector<float> flat_pts;
    flat_pts.reserve(n * 5);
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& pt = exported.points[i];
        flat_pts.push_back(pt.x());
        flat_pts.push_back(pt.y());
        flat_pts.push_back(pt.z());
        flat_pts.push_back(i < exported.probs.size()    ? exported.probs[i]                    : 0.0f);
        flat_pts.push_back(i < exported.track_ids.size() ? static_cast<float>(exported.track_ids[i]) : -1.0f);
    }

    G->add_or_modify_attrib_local<candidate_pts_att>(voxels_node, flat_pts);
    G->add_or_modify_attrib_local<last_sensing_frame_att>(voxels_node, sensing_frame);
    G->update_node(std::move(voxels_node));   // last use: move the voxel-grid blob into the engine (no deep copy under _mutex)

    if (verbose_debug_)
        qInfo() << "[Voxels] Uploaded" << n << "voxels (stride-5) to DSR (frame=" << sensing_frame << ")";
}

void SpecificWorker::upload_masks_to_dsr(const SceneFrame& frame, const std::vector<SegDetection>& detections)
{
    if (!masks_node_ready_)
        return;

    // Keep masks stream progressing even when voxel grid processing is paused.
    const std::uint64_t sensing_frame = ++masks_publish_seq_;
    if (sensing_frame == last_masks_uploaded_frame_)
        return;
    last_masks_uploaded_frame_ = sensing_frame;

    auto masks_node_opt = G->get_node("masks");
    if (!masks_node_opt.has_value())
        return;
    auto& masks_node = masks_node_opt.value();

    const auto& rgbd = frame.rgbd;
    if (rgbd.depth.empty() || rgbd.width <= 0 || rgbd.height <= 0 || rgbd.focal_x <= 0.f || rgbd.focal_y <= 0.f)
    {
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_id", static_cast<int>(sensing_frame));
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_count", 0);
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_labels", std::string{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_label_ids", std::vector<float>{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_confidences", std::vector<float>{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_offsets", std::vector<float>{0.0f});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points", std::vector<float>{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", std::vector<float>{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", std::vector<float>{});
        G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", std::vector<float>{});
        G->update_node(std::move(masks_node));
        return;
    }

    const Eigen::Matrix3f room_rotation = frame.room_T_zed.linear().cast<float>();
    const Eigen::Vector3f room_translation = frame.room_T_zed.translation().cast<float>();
    const float z_lift_m = params.VOXEL_Z_LIFT_M;
    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    const float cx = static_cast<float>(rgbd.width) * 0.5f;
    const float cy = static_cast<float>(rgbd.height) * 0.5f;

    std::vector<float> label_ids;
    std::vector<float> confidences;
    std::vector<float> support_offsets;
    std::vector<float> support_points;
    std::vector<float> centroids_xyz;
    std::vector<float> bbox_min_xyz;
    std::vector<float> bbox_max_xyz;
    std::ostringstream labels_joined;
    std::size_t total_support_points = 0;
    support_offsets.push_back(0.0f);

    const std::size_t mask_stride = std::max<std::size_t>(1, params.VOXEL_DECIMATION_FACTOR);

    for (std::size_t det_idx = 0; det_idx < detections.size(); ++det_idx)
    {
        const auto& det = detections[det_idx];
        if (det.mask.empty())
            continue;

        cv::Mat mask_bin;
        cv::threshold(det.mask, mask_bin, 127, 255, cv::THRESH_BINARY);
        const int min_x = std::max(0, det.bbox.x);
        const int min_y = std::max(0, det.bbox.y);
        const int max_x = std::min(rgbd.width, det.bbox.x + det.bbox.width);
        const int max_y = std::min(rgbd.height, det.bbox.y + det.bbox.height);

        // Pass 1: gather valid masked (depth, room point) candidates.
        std::vector<std::pair<float, Eigen::Vector3f>> candidates;
        candidates.reserve(static_cast<std::size_t>(det.bbox.area() / std::max<int>(1, static_cast<int>(mask_stride * mask_stride))));

        for (int row = min_y; row < max_y; row += static_cast<int>(mask_stride))
        {
            for (int col = min_x; col < max_x; col += static_cast<int>(mask_stride))
            {
                if (mask_bin.at<std::uint8_t>(row, col) == 0)
                    continue;

                const std::size_t depth_idx = static_cast<std::size_t>(row * rgbd.width + col);
                if (depth_idx >= rgbd.depth.size())
                    continue;

                const float depth = rgbd.depth[depth_idx];
                if (!std::isfinite(depth) || depth <= 0.0f)
                    continue;

                const float px = (static_cast<float>(col) - cx) * depth / fx;
                const float py = depth;
                const float pz = (cy - static_cast<float>(row)) * depth / fy;
                Eigen::Vector3f point_room = room_rotation * Eigen::Vector3f(px, py, pz) + room_translation;
                point_room.z() += z_lift_m;

                candidates.emplace_back(depth, point_room);
            }
        }

        if (candidates.empty())
            continue;

        // Robust near-surface depth (20th percentile): the object is the NEAREST surface, so
        // this picks it even when most pixels drop through to the background. Reject pixels
        // beyond near + band — the transparent-object dropout line that the 2D mask hides.
        float depth_gate = std::numeric_limits<float>::max();
        if (params.MASK_DEPTH_GATE_BAND_M > 0.0f)
        {
            std::vector<float> depths;
            depths.reserve(candidates.size());
            for (const auto& [d, p] : candidates) depths.push_back(d);
            const std::size_t k = depths.size() / 5;   // 20th percentile
            std::nth_element(depths.begin(), depths.begin() + k, depths.end());
            depth_gate = depths[k] + params.MASK_DEPTH_GATE_BAND_M;
        }

        std::vector<Eigen::Vector3f> gated;
        gated.reserve(candidates.size());
        for (const auto& [d, point_room] : candidates)
            if (d <= depth_gate)
                gated.push_back(point_room);

        if (gated.empty())
            continue;

        // Radius outlier removal: keep points that have enough neighbours nearby. The dense
        // object body survives; the sparse silhouette-edge tail is trimmed.
        std::vector<Eigen::Vector3f> mask_points_room;
        if (params.MASK_OUTLIER_MIN_NEIGHBORS > 0 and params.MASK_OUTLIER_RADIUS_M > 0.0f
            and gated.size() > static_cast<std::size_t>(params.MASK_OUTLIER_MIN_NEIGHBORS))
        {
            const float r2 = params.MASK_OUTLIER_RADIUS_M * params.MASK_OUTLIER_RADIUS_M;
            mask_points_room.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
            {
                int neighbours = 0;
                for (std::size_t j = 0; j < gated.size() and neighbours < params.MASK_OUTLIER_MIN_NEIGHBORS; ++j)
                    if (i != j and (gated[i] - gated[j]).squaredNorm() <= r2)
                        ++neighbours;
                if (neighbours >= params.MASK_OUTLIER_MIN_NEIGHBORS)
                    mask_points_room.push_back(gated[i]);
            }
        }
        else
            mask_points_room = std::move(gated);

        if (mask_points_room.empty())
            continue;

        Eigen::Vector3f min_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f max_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        Eigen::Vector3f sum_pt = Eigen::Vector3f::Zero();
        for (const auto& point_room : mask_points_room)
        {
            sum_pt += point_room;
            min_pt = min_pt.cwiseMin(point_room);
            max_pt = max_pt.cwiseMax(point_room);
        }

        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << det.label;

        label_ids.push_back(static_cast<float>(det.class_id));
        confidences.push_back(det.confidence);
        support_offsets.push_back(static_cast<float>(support_offsets.back() + static_cast<float>(mask_points_room.size())));

        const Eigen::Vector3f centroid = sum_pt / static_cast<float>(mask_points_room.size());
        centroids_xyz.push_back(centroid.x());
        centroids_xyz.push_back(centroid.y());
        centroids_xyz.push_back(centroid.z());

        bbox_min_xyz.push_back(min_pt.x());
        bbox_min_xyz.push_back(min_pt.y());
        bbox_min_xyz.push_back(min_pt.z());
        bbox_max_xyz.push_back(max_pt.x());
        bbox_max_xyz.push_back(max_pt.y());
        bbox_max_xyz.push_back(max_pt.z());

        for (const auto& point : mask_points_room)
        {
            support_points.push_back(point.x());
            support_points.push_back(point.y());
            support_points.push_back(point.z());
        }

        total_support_points += mask_points_room.size();
    }

    const int mask_count = static_cast<int>(label_ids.size());
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_id", static_cast<int>(sensing_frame));
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_count", mask_count);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_labels", labels_joined.str());
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_label_ids", label_ids);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_confidences", confidences);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_offsets", support_offsets);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points", support_points);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", centroids_xyz);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", bbox_min_xyz);
    G->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", bbox_max_xyz);
    G->update_node(std::move(masks_node));

    if (verbose_debug_)
        qInfo() << "[Masks] Uploaded" << mask_count << "masks and" << total_support_points << "support points to DSR (frame=" << sensing_frame << ")";
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

void SpecificWorker::cleanup_semantic_grid_nodes()
{
    const auto nodes = G->get_nodes_by_type("semantic_grid");
    for (const auto& node : nodes)
    {
        qInfo() << "[Voxels] Removing stale '" << node.name().c_str() << "' node (id=" << node.id() << ") from DSR graph";
        G->delete_node(node.id());
    }
    voxels_node_ready_ = false;
    masks_node_ready_  = false;
    tracks_node_ready_ = false;
}

