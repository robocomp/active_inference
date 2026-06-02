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
    qInfo() << "Destroying SpecificWorker";
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
    rc::ConfigLoaderUtils::load_optional(configLoader, "Transforms.interpolate_rt", params.TRANSFORMS_INTERPOLATE_RT);
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

        auto* clear_voxels_btn = new QPushButton("Clear Voxels", voxel_panel);
        clear_voxels_btn->setCursor(Qt::PointingHandCursor);

        controls_layout->addWidget(lidar_btn);
        controls_layout->addWidget(lidar_voxels_btn);
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
                     this, &SpecificWorker::cleanup_owned_nodes, Qt::UniqueConnection);
}

void SpecificWorker::compute()
{
    if (!scene_processor or !voxel_processor)
        return;

    const auto frame = process_scene_frame(fps_counter_);
    if (!frame.has_value())
        return;

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

    update_table_nodes_from_tracks(frame->graph_object_boxes, frame->lidar_points_room);

    ensure_voxels_node_in_dsr();
    upload_voxel_grid_to_dsr();

    fps_counter_.print("[Compute]", 3000);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<SpecificWorker::SceneFrame> SpecificWorker::process_scene_frame(FPSCounter& compute_fps)
{
    scene_processor->check_input_stream_startup_status();
    const auto [room_name, robot_name] = scene_processor->get_room_robot_names_for_compute();

    const auto rgbd_opt = scene_processor->get_rgbd_frame_from_dsr();
    if (!rgbd_opt.has_value())
        return std::nullopt;
    const std::uint64_t frame_ts_ms = scene_processor->get_frame_timestamp_ms();

    if (!scene_processor->ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
        return std::nullopt;

    const auto room_T_robot = scene_processor->get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
    if (!room_T_robot.has_value())
        return std::nullopt;
    const auto room_T_zed = scene_processor->get_room_zed_transform(compute_fps, room_name, frame_ts_ms);
    if (!room_T_zed.has_value())
        return std::nullopt;

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
    scene_processor->update_room_polygon_periodic();

    return SceneFrame{rgbd_opt.value(),
                      room_T_robot.value(),
                      room_T_zed.value(),
                      std::move(lidar_points_room),
                      graph_object_boxes};
}

void SpecificWorker::update_table_nodes_from_tracks(const std::vector<GraphObjectBox>& graph_object_boxes,
                                                    std::span<const Eigen::Vector3f> lidar_points_room)
{
    const auto& track_cands   = voxel_processor->last_track_candidates();
    const int   sensing_frame = voxel_processor->last_frame_id();
    const int   table_cand_count = static_cast<int>(std::count_if(
        track_cands.begin(), track_cands.end(), [](const auto& c){ return c.category == "table"; }));
    const int   model_box_count  = static_cast<int>(std::count_if(
        graph_object_boxes.begin(), graph_object_boxes.end(), [](const auto& b){ return b.category == "model_table"; }));

    if (sensing_frame % 30 == 0)
        qInfo() << "[TableCapture] frame=" << sensing_frame << " model_boxes=" << model_box_count << " table_tracks=" << table_cand_count;

    // Index model boxes by DSR node name so each table node finds its own box in O(1).
    struct ModelBox
    {
        Eigen::Vector3f min, max, centroid, half_extents;
        float yaw_rad = 0.f;
    };
    std::unordered_map<std::string, ModelBox> model_box_map;
    for (const auto& box : graph_object_boxes)
    {
        if (box.category != "model_table" or box.node_name.empty())
            continue;
        model_box_map.emplace(box.node_name, ModelBox{
            box.min, box.max, (box.min + box.max) * 0.5f, box.half_extents, box.yaw_rad
        });
    }
    if (model_box_map.empty())
        return;

    for (auto table_node : G->get_nodes_by_type("table"))
    {
        const auto mb_it = model_box_map.find(table_node.name());
        if (mb_it == model_box_map.end())
            continue;
        const auto& mb = mb_it->second;

        // OBB inside check — defined before the matching loop so it can be
        // reused for both candidate scoring (50-pt sample) and explanation_ratio
        // (400-pt full fetch).
        const float cy = std::cos(-mb.yaw_rad);
        const float sy = std::sin(-mb.yaw_rad);
        auto inside_obb = [&](const Eigen::Vector3f& p) -> bool
        {
            const Eigen::Vector3f rel = p - mb.centroid;
            const float lx = rel.x() * cy - rel.y() * sy;
            const float ly = rel.x() * sy + rel.y() * cy;
            return std::abs(lx) <= mb.half_extents.x()
               and std::abs(ly) <= mb.half_extents.y()
               and std::abs(rel.z()) <= mb.half_extents.z();
        };

        const float kTrackMatchPadding = 0.10f;
        const Eigen::Vector3f match_min = mb.min.array() - kTrackMatchPadding;
        const Eigen::Vector3f match_max = mb.max.array() + kTrackMatchPadding;
        auto aabb_intersection_volume = [](const Eigen::Vector3f& min_a,
                                           const Eigen::Vector3f& max_a,
                                           const Eigen::Vector3f& min_b,
                                           const Eigen::Vector3f& max_b) -> float
        {
            const Eigen::Vector3f overlap = (max_a.cwiseMin(max_b) - min_a.cwiseMax(min_b)).cwiseMax(0.0f);
            return overlap.x() * overlap.y() * overlap.z();
        };

        // Primary match: pick the table track whose AABB overlaps the model box
        // the most. This remains meaningful even when model-explained points are
        // suppressed before track construction.
        int best_track_id  = -1;
        int best_obb_score = -1;
        float best_overlap_volume = -1.0f;
        float best_centroid_dist = std::numeric_limits<float>::max();
        for (const auto& cand : track_cands)
        {
            if (cand.category != "table") continue;

            const float overlap_volume = aabb_intersection_volume(cand.min, cand.max, match_min, match_max);
            const auto sample = voxel_processor->get_flat_pts_for_track(cand.track_id, 50);
            int score = 0;
            for (std::size_t k = 0; k < sample.size() / 3; ++k)
                if (inside_obb({sample[k*3], sample[k*3+1], sample[k*3+2]}))
                    ++score;

            const float dx = cand.centroid.x() - mb.centroid.x();
            const float dy = cand.centroid.y() - mb.centroid.y();
            const float dist = std::sqrt(dx*dx + dy*dy);

            if (overlap_volume > best_overlap_volume
                || (std::abs(overlap_volume - best_overlap_volume) < 1e-6f && score > best_obb_score)
                || (std::abs(overlap_volume - best_overlap_volume) < 1e-6f
                    && score == best_obb_score
                    && dist < best_centroid_dist))
            {
                best_overlap_volume = overlap_volume;
                best_obb_score = score;
                best_track_id  = cand.track_id;
                best_centroid_dist = dist;
            }
        }
        // Fallback to XY centroid distance only when no table track intersects the
        // model neighbourhood at all.
        if (best_overlap_volume <= 0.0f)
        {
            float best_dist = std::numeric_limits<float>::max();
            best_track_id   = -1;
            for (const auto& cand : track_cands)
            {
                if (cand.category != "table") continue;
                const float dx = cand.centroid.x() - mb.centroid.x();
                const float dy = cand.centroid.y() - mb.centroid.y();
                const float dist = std::sqrt(dx*dx + dy*dy);
                if (dist < best_dist) { best_dist = dist; best_track_id = cand.track_id; }
            }
        }
        if (sensing_frame % 30 == 0)
        {
            if (best_track_id >= 0)
                qInfo() << "[TableCapture] node='" << table_node.name().c_str()
                        << "' model_centroid=(" << mb.centroid.x() << "," << mb.centroid.y() << "," << mb.centroid.z()
                        << ") best_track=" << best_track_id << " overlap=" << best_overlap_volume << " obb_score=" << best_obb_score;
            else
                qInfo() << "[TableCapture] node='" << table_node.name().c_str() << "' no table track visible this frame";
        }
        std::vector<float> flat_pts;
        if (best_track_id >= 0)
            flat_pts = voxel_processor->get_flat_pts_for_track(best_track_id, 400);

        const std::size_t n_cands = flat_pts.size() / 3;
        int inside_count = 0;
        for (std::size_t k = 0; k < n_cands; ++k)
        {
            if (inside_obb(Eigen::Vector3f(flat_pts[k*3], flat_pts[k*3+1], flat_pts[k*3+2])))
                ++inside_count;
        }
        const float explanation_ratio = n_cands > 0
            ? static_cast<float>(inside_count) / static_cast<float>(n_cands)
            : 0.0f;

        // residual_pts: lidar points NOT explained by the current model OBB,
        // constrained to an expanded model neighbourhood. Capped at 150 points.
        constexpr float kNeighbourhoodMargin = 0.3f;
        const Eigen::Vector3f nb_min = mb.min.array() - kNeighbourhoodMargin;
        const Eigen::Vector3f nb_max = mb.max.array() + kNeighbourhoodMargin;
        std::vector<float> residual_flat;
        residual_flat.reserve(450);
        std::size_t lidar_in_neighbourhood = 0;

        if (!lidar_points_room.empty())
        {
            for (const auto& p : lidar_points_room)
            {
                if (residual_flat.size() >= 450)
                    break;
                if (!((p.array() >= nb_min.array()).all() and (p.array() <= nb_max.array()).all()))
                    continue;
                ++lidar_in_neighbourhood;
                if (inside_obb(p))
                    continue;
                residual_flat.push_back(p.x());
                residual_flat.push_back(p.y());
                residual_flat.push_back(p.z());
            }
        }
        else
        {
            // Fallback when no lidar frame is available.
            for (std::size_t k = 0; k < n_cands && residual_flat.size() < 450; ++k)
            {
                const Eigen::Vector3f p(flat_pts[k*3], flat_pts[k*3+1], flat_pts[k*3+2]);
                if (inside_obb(p))
                    continue;
                if ((p.array() >= nb_min.array()).all() and (p.array() <= nb_max.array()).all())
                {
                    residual_flat.push_back(p.x());
                    residual_flat.push_back(p.y());
                    residual_flat.push_back(p.z());
                }
            }
        }
        const int residual_mass = static_cast<int>(residual_flat.size() / 3);

        G->add_or_modify_attrib_local<candidate_pts_att>(table_node, flat_pts);
        G->add_or_modify_attrib_local<residual_pts_att>(table_node, residual_flat);
        G->add_or_modify_attrib_local<residual_mass_att>(table_node, residual_mass);
        G->add_or_modify_attrib_local<last_sensing_frame_att>(table_node, sensing_frame);
        G->add_or_modify_attrib_local<explanation_ratio_att>(table_node, explanation_ratio);
        G->update_node(table_node);
        if (sensing_frame % 30 == 0)
            qInfo() << "[TableCapture] WROTE node='" << table_node.name().c_str()
                    << "' frame=" << sensing_frame << " cands=" << n_cands
                    << " resid=" << residual_mass << " expl=" << explanation_ratio << " lidar_in_nb=" << lidar_in_neighbourhood;
    }
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
    G->update_node(voxels_node);

    if (verbose_debug_)
        qInfo() << "[Voxels] Uploaded" << n << "voxels (stride-5) to DSR (frame=" << sensing_frame << ")";
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
}

