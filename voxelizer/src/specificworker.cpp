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

#include <algorithm>
#include <limits>
#include <print>

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
        statemachine.setChildMode(QState::ExclusiveStates);
        statemachine.start();
        const auto error = statemachine.errorString();
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
    try { params.YOLO_MODEL_PATH = configLoader.get<std::string>("Yolo.model_path"); } catch (...) {}
    try { params.YOLO_ACCEPTED_LABELS = configLoader.get<std::vector<std::string>>("Yolo.accepted_labels"); } catch (...) {}
    try { params.YOLO_CONF_THRESH = static_cast<float>(configLoader.get<double>("Yolo.conf_thresh")); } catch (...) {}
    try { params.YOLO_IOU_THRESH = static_cast<float>(configLoader.get<double>("Yolo.iou_thresh")); } catch (...) {}
    try { params.YOLO_USE_GPU = configLoader.get<bool>("Yolo.use_gpu"); } catch (...) {}
    try { params.YOLO_USE_TRT = configLoader.get<bool>("Yolo.use_trt"); } catch (...) {}
    try { params.YOLO_MASK_ERODE_KERNEL = configLoader.get<int>("Yolo.mask_erode_kernel"); } catch (...) {}
    try { params.TRACK_ASSOCIATION_MAX_DISTANCE_M = static_cast<float>(configLoader.get<double>("Yolo.track_association_max_distance_m")); } catch (...) {}
    try { params.TRACK_MAX_MISSED_FRAMES = configLoader.get<int>("Yolo.track_max_missed_frames"); } catch (...) {}
    try { params.VOXEL_VIEWER_MAX_RENDERED_VOXELS = static_cast<std::size_t>(configLoader.get<int>("Voxel.viewer_max_rendered_voxels")); } catch (...) {}
    try { params.VOXEL_VIEWER_FPS = configLoader.get<int>("Voxel.viewer_fps"); } catch (...) {}
    try { params.TRANSFORMS_INTERPOLATE_RT = configLoader.get<bool>("Transforms.interpolate_rt"); } catch (...) {}
    try { verbose_debug_ = configLoader.get<bool>("Debug.verbose"); }
    catch (...) { verbose_debug_ = false; }

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

    // --- Voxel viewer (attaches to DSR GUI if available) ---
    if (!graph_viewers.empty())
    {
        const std::string viewer_key = graph_viewers.contains("")
            ? std::string("") : graph_viewers.begin()->first;
        voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(nullptr);
        graph_viewers.at(viewer_key)->add_custom_widget_to_dock("Voxel3D", voxel_viewer_gl.get());
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
    voxel_processor_config.verbose_debug                   = verbose_debug_;
    voxel_processor->configure(voxel_processor_config);

    // --- Scene processor (DSR-only: no proxies) ---
    inner_eigen_api = G->get_inner_eigen_api();
    scene_processor = std::make_unique<SceneProcessor>(G);
    scene_processor->configure(inner_eigen_api.get(), voxel_viewer_gl.get(),
                               params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_);
}

void SpecificWorker::compute()
{
    static FPSCounter compute_fps;
    if (!scene_processor or !voxel_processor)
        return;

    const auto frame = process_scene_frame(compute_fps);
    if (!frame.has_value())
        return;

    const auto detections = yolo_processor
        ? yolo_processor->detect_segmentation(frame->rgbd.rgb)
        : std::vector<SegDetection>{};
  
    voxel_processor->process_rgbd_frame(frame->rgbd, detections,
                                        frame->room_T_robot, frame->room_T_zed,
                                        frame->graph_object_boxes, voxel_viewer_gl.get());

    update_table_nodes_from_tracks(frame->graph_object_boxes);

    if (verbose_debug_)
        compute_fps.print("[Compute]", 2000);
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

    scene_processor->update_viewer_robot_pose(room_T_robot.value());
    scene_processor->update_viewer_lidar_points(room_name, robot_name, room_T_robot.value());
    scene_processor->update_viewer_graph_object_boxes(graph_object_boxes);
    scene_processor->update_room_polygon_periodic();

    return SceneFrame{rgbd_opt.value(), room_T_robot.value(), room_T_zed.value(), graph_object_boxes};
}

void SpecificWorker::update_table_nodes_from_tracks(const std::vector<GraphObjectBox>& graph_object_boxes)
{
    const auto& track_cands   = voxel_processor->last_track_candidates();
    const int   sensing_frame = voxel_processor->last_frame_id();
    const int   table_cand_count = static_cast<int>(std::count_if(
        track_cands.begin(), track_cands.end(), [](const auto& c){ return c.category == "table"; }));
    const int   model_box_count  = static_cast<int>(std::count_if(
        graph_object_boxes.begin(), graph_object_boxes.end(), [](const auto& b){ return b.category == "model_table"; }));

    if (sensing_frame % 30 == 0)
        std::println("[TableCapture] frame={} model_boxes={} table_tracks={}", sensing_frame, model_box_count, table_cand_count);

    for (auto table_node : G->get_nodes_by_type("table"))
    {
        Eigen::Vector3f model_centroid = Eigen::Vector3f::Zero();
        bool has_model_box = false;
        for (const auto& box : graph_object_boxes)
        {
            if (box.category == "model_table")
            {
                model_centroid = (box.min + box.max) * 0.5f;
                has_model_box = true;
                break;
            }
        }
        if (!has_model_box)
            continue;

        int   best_track_id = -1;
        float best_dist     = std::numeric_limits<float>::max();
        for (const auto& cand : track_cands)
        {
            if (cand.category != "table")
                continue;
            const float dist = (cand.centroid - model_centroid).norm();
            if (dist < best_dist)
            {
                best_dist     = dist;
                best_track_id = cand.track_id;
            }
        }
        if (sensing_frame % 30 == 0)
            std::println("[TableCapture] node='{}' model_centroid=({:.2f},{:.2f},{:.2f}) best_track={} best_dist={:.2f}",
                         table_node.name(), model_centroid.x(), model_centroid.y(), model_centroid.z(),
                         best_track_id, best_dist);
        if (best_track_id < 0)
            continue;

        auto flat_pts = voxel_processor->get_flat_pts_for_track(best_track_id, 400);
        if (flat_pts.empty())
            continue;

        G->add_or_modify_attrib_local<candidate_pts_att>(table_node, flat_pts);
        G->add_or_modify_attrib_local<last_sensing_frame_att>(table_node, sensing_frame);
        G->add_or_modify_attrib_local<explanation_ratio_att>(table_node, 0.0f);
        G->update_node(table_node);
        if (sensing_frame % 30 == 0)
            std::println("[TableCapture] WROTE node='{}' frame={} pts={}",
                         table_node.name(), sensing_frame, flat_pts.size() / 3);
    }
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

