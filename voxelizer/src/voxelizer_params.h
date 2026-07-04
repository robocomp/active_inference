/*
 * voxelizer_params.h
 *
 * Configuration struct for the voxelizer agent + a loader that fills it from a
 * RoboComp ConfigLoader (all keys optional, with the defaults below). Mirrors the
 * agent_config pattern used by the concept agents: keeps config parsing out of
 * SpecificWorker::initialize().
 */

#pragma once

#include <cstdint>
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
    // Drop a detection whose ROI intersects the tray when the overlap covers ≥ this fraction of the
    // bbox. 0 ⇒ drop on ANY intersection (Yolo.tray_drop_fraction).
    float       YOLO_TRAY_DROP_FRACTION = 0.0f;
    std::size_t VOXEL_DECIMATION_FACTOR          = 2;
    float       VOXEL_Z_LIFT_M                   = 0.0f;
    bool        TRANSFORMS_INTERPOLATE_RT        = true;
    // Per-mask depth gate: reject mask pixels whose depth exceeds the mask's near
    // surface by more than this band (m). Kills transparent-object depth dropout —
    // pixels that see THROUGH the object to the background and deproject into a line.
    // <= 0 disables the gate.
    float       MASK_DEPTH_GATE_BAND_M           = 0.20f;
    // Per-mask radius outlier removal: drop points with fewer than MIN_NEIGHBORS
    // others within RADIUS_M. Trims the sparse silhouette-edge "tail" the depth gate
    // leaves behind, while the dense object body survives. <= 0 disables.
    float       MASK_OUTLIER_RADIUS_M            = 0.03f;
    int         MASK_OUTLIER_MIN_NEIGHBORS       = 4;

    // Ego-motion mask-corruption annotation (saccadic-suppression / VOR producer side). When ON, the
    // 'masks' node carries per-mask (dot_d, bias, variance, truncation, centroid radius) plus the
    // capture-time camera twist, so a consumer can downweight/gate masks taken while the camera moves.
    // Pure annotation — consumers that ignore the attrs are unaffected. See common/motion_corruption.
    bool        MASK_MOTION_ENABLED        = true;    // MaskMotion.enabled
    float       MASK_MOTION_EXPOSURE_S     = 0.005f;  // MaskMotion.exposure_s  (T_exp, blur window)
    float       MASK_MOTION_TIMING_JITTER_S = 0.010f; // MaskMotion.timing_jitter_s (σ_t, zero-mean)
    float       MASK_MOTION_TIMING_OFFSET_S = 0.0f;   // MaskMotion.timing_offset_s (δt, known lag → bias)
    // Diagnostic: append per-mask motion-corruption rows to etc/mask_motion_log.csv (verify dot_d≈0 when
    // static, spikes on pan). Default ON during bring-up; turn off in production. MaskMotion.csv_log
    bool        MASK_MOTION_CSV_LOG        = true;

    // Extrapolate the robot pose to the camera CAPTURE stamp using the body-frame velocity on the
    // robot→room RT edge. DSR's InterpolatedRT clamps to the newest block (poses lag the camera by
    // ~100 ms), so masks deproject against a stale pose; this fills that residual lag at the consumer
    // (efference-copy applied to perception) — no producer rate change. MaskMotion.pose_extrapolate
    bool        MASK_POSE_EXTRAPOLATE      = true;
    float       MASK_POSE_EXTRAP_MAX_DT_S  = 0.2f;   // clamp the extrapolation horizon (s)

    // Human-pose branch (yolo_human): a second YOLO-pose model on the same RGB frame, published as
    // BODY_18 3D skeletons (camera frame) on the 'skeleton' node for human_concept. Default OFF.
    bool        HUMAN_POSE_ENABLED      = false;            // HumanPose.enabled
    std::string HUMAN_POSE_MODEL_PATH   = "yolo26m-pose.onnx";
    float       HUMAN_POSE_CONF_THRESH  = 0.30f;            // person-detection confidence floor
    float       HUMAN_POSE_IOU_THRESH   = 0.45f;
    int         HUMAN_POSE_INPUT_SIZE   = 640;
    bool        HUMAN_POSE_USE_GPU      = true;
    bool        HUMAN_POSE_USE_TRT      = false;
    // Keep the last good skeleton on the viewer for up to this long after a detection miss (anti-flicker).
    std::uint64_t HUMAN_POSE_HOLD_MS    = 500;              // HumanPose.hold_ms
    // Run the pose model every Nth compute cycle. 1 = every frame (skeleton stays glued to the moving
    // person, no freeze-then-snap stutter). Raise to trade skeleton smoothness for GPU if the box is
    // inference-bound; the hold above still bridges any resulting gaps.
    int         HUMAN_POSE_DECIMATION   = 1;               // HumanPose.decimation
    // Per-joint confidence floor below which a keypoint is dropped (NaN) from the skeleton node.
    float       SKELETON_KP_CONF_MIN    = 0.30f;
    // Half-window (px) for the median-depth patch sampled at each keypoint (0 = single pixel).
    int         SKELETON_DEPTH_PATCH    = 2;

    // Semantic-segmentation branch (yolo_semantic): a YOLO26 *-sem model on the same RGB frame
    // producing a DENSE per-pixel ADE20K-150 class map (vs the sparse per-instance seg masks).
    // Currently a viewer overlay only — no DSR publish. Default OFF.
    bool        SEMANTIC_SEG_ENABLED    = false;                    // Semantic.enabled
    std::string SEMANTIC_SEG_MODEL_PATH = "yolo26l-sem-ade20k.onnx";
    float       SEMANTIC_SEG_CONF_THRESH= 0.25f;                    // per-pixel argmax-softmax floor
    int         SEMANTIC_SEG_INPUT_SIZE = 640;
    bool        SEMANTIC_SEG_USE_GPU    = true;
    bool        SEMANTIC_SEG_USE_TRT    = false;
    // Run the (heavy dense) model every Nth cycle; the last map is reused on skipped cycles.
    int         SEMANTIC_SEG_DECIMATION = 1;

    // Media plane (zero-copy DDS) for RGBD pixels carried OUT of the graph.
    int         MEDIA_DOMAIN_ID   = 0;
    std::string MEDIA_RGB_TOPIC   = "rc/zed/rgb";
    std::string MEDIA_DEPTH_TOPIC = "rc/zed/depth";
    // LiDAR over the media plane (robot_concept's LidarFrame stream). LIDAR_USE_MEDIA=false ⇒ DSR
    // graph 'lidar3D' node only.
    std::string MEDIA_LIDAR_TOPIC = "rc/lidar3d/points";
    bool        LIDAR_USE_MEDIA   = true;
    // RGBD_360 panorama (Image360Frame plane), display-only in the Ricoh popup.
    std::string MEDIA_RICOH_TOPIC = "rc/ricoh/rgb";

    // Ricoh 360 peripheral detection: OWN YOLO model/session (rc::RicohYoloWorker), run over the
    // panorama split into RICOH_YOLO_N_STRIPS vertical strips (see YoloProcessor::detect_segmentation_360)
    // on a DEDICATED thread — decoupled from compute()'s budget, paced to RICOH_YOLO_THREAD_PERIOD_MS
    // (~20 Hz default) rather than traded against the main perception rate via decimation. A coarse,
    // no-depth "glance" signal — NOT wired to DSR yet (see RICOH_360_PERIPHERAL_DETECTION.md). Default OFF.
    bool        RICOH_YOLO_ENABLED         = false;   // Ricoh.yolo_enabled
    int         RICOH_YOLO_THREAD_PERIOD_MS= 50;      // Ricoh.yolo_thread_period_ms — target worker-thread cycle (~20 Hz)
    int         RICOH_YOLO_N_STRIPS        = 3;       // Ricoh.yolo_n_strips
    int         RICOH_YOLO_STRIP_OVERLAP_PX= 80;      // Ricoh.yolo_strip_overlap_px
    float       RICOH_YOLO_MERGE_IOU       = 0.5f;    // Ricoh.yolo_merge_iou — cross-strip dedup threshold
    // Publish the ricoh 360 detections into the shared "masks" node as NO-DEPTH bearing slices (Part B,
    // RICOH_360_PERIPHERAL_DETECTION.md). azimuth_offset trims the panorama-column→room-bearing mapping for
    // the ricoh's mounting yaw + the panorama's 0-column convention — PROVISIONAL, verify live vs the
    // descriptor projection model before any consumer relies on the bearing (Part C is not built yet).
    bool        RICOH_PUBLISH_MASKS        = true;    // Ricoh.publish_masks
    float       RICOH_AZIMUTH_OFFSET_RAD   = 0.0f;    // Ricoh.azimuth_offset_rad — additive zero correction
    float       RICOH_AZIMUTH_SIGN         = 1.0f;    // Ricoh.azimuth_sign — +1/−1: flip if the panorama is mirrored
    bool        RICOH_LOG_BEARINGS         = false;   // Ricoh.log_bearings — print each 360 detection's
                                                      // label + computed bearings (azimuth calibration)

    // Custom drawing windows (attach to the DSR GUI if available). Both default ON.
    bool        SHOW_VOXEL_VIEWER = true; // Voxel.show_voxel_viewer
    bool        SHOW_YOLO_VIEWER  = true; // Voxel.show_yolo_viewer
    bool        SHOW_RICOH_VIEWER = true; // Voxel.show_ricoh_viewer — creates the button+popup (hidden until toggled)
    bool        PERF_LOG          = false; // Voxel.perf_log — per-frame compute/yolo/pose timing → etc/viewer_perf.csv
    float       TARGET_HZ         = 20.0f;  // target compute rate for the perception-rate regulator (+ warn floor)
    int         POSE_DECIM_MAX    = 4;       // RateRegulator.pose_decim_max — regulator won't skip pose more than this
    // Input-stream publish-hold watchdog: if the RGB feed goes stale (producer stall), stop publishing
    // perception (better nothing than stale masks). Debounced with hysteresis.
    float       HOLD_ENTER_S      = 1.5f;    // StreamWatchdog.hold_enter_s — RGB stale this long → enter hold
    float       HOLD_RECOVER_S    = 1.0f;    // StreamWatchdog.recover_s — sustained freshness before resuming

    bool        VERBOSE_DEBUG = false;
};

// Fill a VoxelizerParams from the RoboComp ConfigLoader (every key optional).
VoxelizerParams load_voxelizer_params(const ConfigLoader& configLoader);
