/*
 * voxelizer_params.cpp — load_voxelizer_params (config → VoxelizerParams).
 */

#include "voxelizer_params.h"

VoxelizerParams load_voxelizer_params(const ConfigLoader& configLoader)
{
    VoxelizerParams params;

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
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Yolo.tray_drop_fraction", params.YOLO_TRAY_DROP_FRACTION);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.z_lift_m", params.VOXEL_Z_LIFT_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.mask_depth_gate_band_m", params.MASK_DEPTH_GATE_BAND_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.mask_outlier_radius_m", params.MASK_OUTLIER_RADIUS_M);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "Voxel.mask_outlier_min_neighbors", params.MASK_OUTLIER_MIN_NEIGHBORS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Transforms.interpolate_rt", params.TRANSFORMS_INTERPOLATE_RT);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.domain_id", params.MEDIA_DOMAIN_ID);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.rgb_topic", params.MEDIA_RGB_TOPIC);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.depth_topic", params.MEDIA_DEPTH_TOPIC);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.lidar_topic", params.MEDIA_LIDAR_TOPIC);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.lidar_use_media", params.LIDAR_USE_MEDIA);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Media.ricoh_topic", params.MEDIA_RICOH_TOPIC);

    // Ricoh 360 peripheral detection (default OFF — see header).
    rc::ConfigLoaderUtils::load_optional(configLoader, "Ricoh.yolo_enabled", params.RICOH_YOLO_ENABLED);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Ricoh.yolo_thread_period_ms", params.RICOH_YOLO_THREAD_PERIOD_MS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Ricoh.yolo_n_strips", params.RICOH_YOLO_N_STRIPS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Ricoh.yolo_strip_overlap_px", params.RICOH_YOLO_STRIP_OVERLAP_PX);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Ricoh.yolo_merge_iou", params.RICOH_YOLO_MERGE_IOU);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Ricoh.publish_masks", params.RICOH_PUBLISH_MASKS);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Ricoh.azimuth_offset_rad", params.RICOH_AZIMUTH_OFFSET_RAD);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Ricoh.azimuth_sign", params.RICOH_AZIMUTH_SIGN);

    // Ego-motion mask-corruption annotation (default ON — pure producer-side metadata).
    rc::ConfigLoaderUtils::load_optional(configLoader, "MaskMotion.enabled", params.MASK_MOTION_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "MaskMotion.exposure_s", params.MASK_MOTION_EXPOSURE_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "MaskMotion.timing_jitter_s", params.MASK_MOTION_TIMING_JITTER_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "MaskMotion.timing_offset_s", params.MASK_MOTION_TIMING_OFFSET_S);
    rc::ConfigLoaderUtils::load_optional(configLoader, "MaskMotion.csv_log", params.MASK_MOTION_CSV_LOG);
    rc::ConfigLoaderUtils::load_optional(configLoader, "MaskMotion.pose_extrapolate", params.MASK_POSE_EXTRAPOLATE);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "MaskMotion.pose_extrap_max_dt_s", params.MASK_POSE_EXTRAP_MAX_DT_S);

    // Human-pose branch (default OFF — see header).
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.enabled", params.HUMAN_POSE_ENABLED);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.model_path", params.HUMAN_POSE_MODEL_PATH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "HumanPose.conf_thresh", params.HUMAN_POSE_CONF_THRESH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "HumanPose.iou_thresh", params.HUMAN_POSE_IOU_THRESH);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.input_size", params.HUMAN_POSE_INPUT_SIZE);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.use_gpu", params.HUMAN_POSE_USE_GPU);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.use_trt", params.HUMAN_POSE_USE_TRT);
    rc::ConfigLoaderUtils::load_optional<std::uint64_t, int>(configLoader, "HumanPose.hold_ms", params.HUMAN_POSE_HOLD_MS);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.decimation", params.HUMAN_POSE_DECIMATION);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "HumanPose.kp_conf_min", params.SKELETON_KP_CONF_MIN);
    rc::ConfigLoaderUtils::load_optional(configLoader, "HumanPose.depth_patch", params.SKELETON_DEPTH_PATCH);

    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.enabled", params.SEMANTIC_SEG_ENABLED);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.model_path", params.SEMANTIC_SEG_MODEL_PATH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Semantic.conf_thresh", params.SEMANTIC_SEG_CONF_THRESH);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.input_size", params.SEMANTIC_SEG_INPUT_SIZE);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.use_gpu", params.SEMANTIC_SEG_USE_GPU);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.use_trt", params.SEMANTIC_SEG_USE_TRT);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Semantic.decimation", params.SEMANTIC_SEG_DECIMATION);

    // Custom drawing windows (default ON).
    rc::ConfigLoaderUtils::load_optional(configLoader, "Voxel.show_voxel_viewer", params.SHOW_VOXEL_VIEWER);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Voxel.show_yolo_viewer", params.SHOW_YOLO_VIEWER);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Voxel.show_ricoh_viewer", params.SHOW_RICOH_VIEWER);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Voxel.perf_log", params.PERF_LOG);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "Voxel.target_hz", params.TARGET_HZ);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RateRegulator.target_hz", params.TARGET_HZ);
    rc::ConfigLoaderUtils::load_optional(configLoader, "RateRegulator.pose_decim_max", params.POSE_DECIM_MAX);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "StreamWatchdog.hold_enter_s", params.HOLD_ENTER_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "StreamWatchdog.recover_s", params.HOLD_RECOVER_S);

    rc::ConfigLoaderUtils::load_optional(configLoader, "Component.Debug.Verbose", params.VERBOSE_DEBUG);
    rc::ConfigLoaderUtils::load_optional(configLoader, "Debug.verbose", params.VERBOSE_DEBUG);

    return params;
}
