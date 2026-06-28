/*
 * voxelizer_params.h
 *
 * Configuration struct for the voxelizer agent + a loader that fills it from a
 * RoboComp ConfigLoader (all keys optional, with the defaults below). Mirrors the
 * agent_config pattern used by the concept agents: keeps config parsing out of
 * SpecificWorker::initialize().
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core/types.hpp>
#include <ConfigLoader/ConfigLoader.h>

struct VoxelizerParams
{
    std::string YOLO_MODEL_PATH         = "yolo26l-seg.onnx";
    std::vector<std::string> YOLO_ACCEPTED_LABELS;
    float       YOLO_CONF_THRESH        = 0.25f;
    float       YOLO_IOU_THRESH         = 0.45f;
    int         YOLO_INPUT_SIZE         = 640;
    bool        YOLO_USE_GPU            = true;
    bool        YOLO_USE_TRT            = true;
    int         YOLO_MASK_ERODE_KERNEL  = 0;
    bool        YOLO_MASK_TRAY          = true;
    int         YOLO_TRAY_MASK_REF_WIDTH  = 1280;
    int         YOLO_TRAY_MASK_REF_HEIGHT = 720;
    // Default crescent polygon (outer arc + image bottom) for a 1280×720 robot camera.
    // Tunable via config.toml: Yolo.tray_mask_polygon = [x0,y0, x1,y1, ...]
    std::vector<cv::Point> YOLO_TRAY_MASK_POLYGON_PX = {
        {195, 720}, {230, 694}, {286, 664}, {353, 640}, {428, 621},
        {510, 608}, {596, 601}, {640, 600}, {684, 601}, {770, 608},
        {852, 621}, {927, 640}, {994, 664}, {1050, 694}, {1086, 720},
        {1280, 720}, {0, 720}
    };
    float       TRACK_ASSOCIATION_MAX_DISTANCE_M = 0.7f;
    int         TRACK_MAX_MISSED_FRAMES          = 10;
    std::size_t VOXEL_VIEWER_MAX_RENDERED_VOXELS = 30'000;
    int         VOXEL_VIEWER_FPS                 = 10;
    std::size_t VOXEL_DECIMATION_FACTOR          = 2;
    float       VOXEL_Z_LIFT_M                   = 0.0f;
    bool        TRANSFORMS_INTERPOLATE_RT        = true;
    // Per-mask depth gate: reject mask pixels whose depth exceeds the mask's near
    // surface by more than this band (m). Kills transparent-object depth dropout —
    // pixels that see THROUGH the object to the background and deproject into a line.
    // <= 0 disables the gate.
    float       MASK_DEPTH_GATE_BAND_M           = 0.20f;
    // Voxel-grid surface band (m): voxel_processor keeps masked points whose range is within
    // this distance of the detection's MEDIAN range. (The mask-support-point export in
    // graph_publisher applies no depth/range gate — it keeps every masked pixel.)
    float       MASK_SURFACE_BAND_M              = 0.35f;
    // Per-mask radius outlier removal: drop points with fewer than MIN_NEIGHBORS
    // others within RADIUS_M. Trims the sparse silhouette-edge "tail" the depth gate
    // leaves behind, while the dense object body survives. <= 0 disables.
    float       MASK_OUTLIER_RADIUS_M            = 0.03f;
    int         MASK_OUTLIER_MIN_NEIGHBORS       = 4;

    // Human-pose branch (yolo_human): a second YOLO-pose model on the same RGB frame, published as
    // BODY_18 3D skeletons (camera frame) on the 'skeleton' node for human_concept. Default OFF.
    bool        HUMAN_POSE_ENABLED      = false;            // HumanPose.enabled
    std::string HUMAN_POSE_MODEL_PATH   = "yolo26m-pose.onnx";
    float       HUMAN_POSE_CONF_THRESH  = 0.30f;            // person-detection confidence floor
    float       HUMAN_POSE_IOU_THRESH   = 0.45f;
    int         HUMAN_POSE_INPUT_SIZE   = 640;
    bool        HUMAN_POSE_USE_GPU      = true;
    bool        HUMAN_POSE_USE_TRT      = false;
    // Per-joint confidence floor below which a keypoint is dropped (NaN) from the skeleton node.
    float       SKELETON_KP_CONF_MIN    = 0.30f;
    // Half-window (px) for the median-depth patch sampled at each keypoint (0 = single pixel).
    int         SKELETON_DEPTH_PATCH    = 2;

    // Media plane (zero-copy DDS) for RGBD pixels carried OUT of the graph.
    int         MEDIA_DOMAIN_ID   = 0;
    std::string MEDIA_RGB_TOPIC   = "rc/zed/rgb";
    std::string MEDIA_DEPTH_TOPIC = "rc/zed/depth";
    // LiDAR over the media plane (robot_concept's LidarFrame stream). LIDAR_USE_MEDIA=false ⇒ DSR
    // graph 'lidar3D' node only.
    std::string MEDIA_LIDAR_TOPIC = "rc/lidar3d/points";
    bool        LIDAR_USE_MEDIA   = true;

    // Optional DSR exports beyond 'masks' (the live product). Both default OFF: 'voxels' has no
    // consumer and 'tracks' is read only by the (deferred) table_concept — publishing them every
    // cycle is wasted graph churn. Flip on if a consumer is revived. ('masks' is always published.)
    bool        PUBLISH_TRACKS = false;   // Voxel.publish_tracks
    bool        PUBLISH_VOXELS = false;   // Voxel.publish_voxels

    // Custom drawing windows (attach to the DSR GUI if available). Both default ON.
    bool        SHOW_VOXEL_VIEWER = true; // Voxel.show_voxel_viewer
    bool        SHOW_YOLO_VIEWER  = true; // Voxel.show_yolo_viewer

    bool        VERBOSE_DEBUG = false;
};

// Fill a VoxelizerParams from the RoboComp ConfigLoader (every key optional).
VoxelizerParams load_voxelizer_params(const ConfigLoader& configLoader);
