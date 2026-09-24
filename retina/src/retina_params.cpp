/*
 * retina_params.cpp — load_retina_params (config → RetinaParams).
 */

#include "retina_params.h"
#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <filesystem>
#include <print>
#include <string>
#include <vector>

RetinaParams load_retina_params(const ConfigLoader& configLoader)
{
    // Every read registers its key, its CODE DEFAULT and a one-line description, so the
    // startup table can say where each value came from - not just what it is.
    rc::cfg::Reader cfgr(configLoader, "retina");
    RetinaParams params;

    cfgr.opt("Yolo.model_path", params.YOLO_MODEL_PATH,
            "ADE20K semantic masks — whitelisting them here too would hand those agents two masks per object per frame");
    cfgr.opt("Yolo.accepted_labels", params.YOLO_ACCEPTED_LABELS,
            "");
    cfgr.opt<float, double>("Yolo.conf_thresh", params.YOLO_CONF_THRESH,
            "min class confidence to ACCEPT a seg detection/mask (argmax score floor); raise to reject weak masks");
    cfgr.opt<float, double>("Yolo.iou_thresh", params.YOLO_IOU_THRESH,
            "NMS IoU: dedupe overlapping same-class boxes above this");
    cfgr.opt<float, double>("Yolo.second_best_margin", params.YOLO_SECOND_BEST_MARGIN,
            "recover a rejected top class via an accepted runner-up within this margin (0 = off)");
    cfgr.opt("Yolo.use_gpu", params.YOLO_USE_GPU,
            "");
    cfgr.opt("Yolo.use_trt", params.YOLO_USE_TRT,
            "same masks, no TRT speedup — if the TRT libs can't be reconciled (e.g. after an apt upgrade)");
    cfgr.opt("Yolo.mask_erode_kernel", params.YOLO_MASK_ERODE_KERNEL,
            "");
    cfgr.opt("Yolo.mask_tray", params.YOLO_MASK_TRAY,
            "");
    cfgr.opt("Yolo.tray_mask_ref_width", params.YOLO_TRAY_MASK_REF_WIDTH,
            "");
    cfgr.opt("Yolo.tray_mask_ref_height", params.YOLO_TRAY_MASK_REF_HEIGHT,
            "");
    const auto reader_tray_polygon = cfgr.v<int>("Yolo.tray_mask_polygon", {},
            "flat x,y,... polygon (image pixels) masking out the serving tray; needs >= 3 points");
    const auto cfgr_apply_tray_mask = [](const std::vector<int>& flat, auto&& fn) { fn(flat); };
    cfgr_apply_tray_mask(reader_tray_polygon,
        [&](const std::vector<int>& flat)
        {
            if (flat.size() >= 6 && flat.size() % 2 == 0)
            {
                params.YOLO_TRAY_MASK_POLYGON_PX.clear();
                for (std::size_t i = 0; i + 1 < flat.size(); i += 2)
                    params.YOLO_TRAY_MASK_POLYGON_PX.emplace_back(flat[i], flat[i + 1]);
            }
        });
    cfgr.opt<float, double>("Yolo.tray_drop_fraction", params.YOLO_TRAY_DROP_FRACTION,
            "Drop a detection whose ROI intersects the tray when the overlap covers ≥ this fraction of the bbox. 0 ⇒ drop on ANY intersection…");
    cfgr.opt("Yolo.zed_thread_period_ms", params.ZED_THREAD_PERIOD_MS,
            "Yolo.zed_thread_period_ms");
    cfgr.opt<float, double>("Voxel.mask_depth_gate_band_m", params.MASK_DEPTH_GATE_BAND_M,
            "Per-mask FOREGROUND depth gate (Voxel.mask_depth_gate_band_m): anchor on the mask's NEAR surface (low percentile of camera-frame depth) and drop…");
    cfgr.opt<float, double>("Voxel.mask_outlier_radius_m", params.MASK_OUTLIER_RADIUS_M,
            "Per-mask radius outlier removal: drop points with fewer than MIN_NEIGHBORS others within RADIUS_M");
    cfgr.opt<int>("Voxel.mask_outlier_min_neighbors", params.MASK_OUTLIER_MIN_NEIGHBORS,
            "");
    cfgr.opt("Transforms.interpolate_rt", params.TRANSFORMS_INTERPOLATE_RT,
            "(Voxel.z_lift_m REMOVED) It offset mask points along ROOM z, so it could only ever be applied to the room-frame array — the camera-frame array the…");
    cfgr.opt("Media.domain_id", params.MEDIA_DOMAIN_ID,
            "Media plane (zero-copy DDS) for RGBD pixels carried OUT of the graph");
    cfgr.opt("Media.rgb_topic", params.MEDIA_RGB_TOPIC,
            "");
    cfgr.opt("Media.depth_topic", params.MEDIA_DEPTH_TOPIC,
            "");
    cfgr.opt("Media.lidar_topic", params.MEDIA_LIDAR_TOPIC,
            "LiDAR over the media plane (robot_concept's LidarFrame stream)");
    cfgr.opt("Media.lidar_use_media", params.LIDAR_USE_MEDIA,
            "false ⇒ DSR graph 'lidar3D' node only");
    cfgr.opt("Media.ricoh_topic", params.MEDIA_RICOH_TOPIC,
            "RGBD_360 panorama (Image360Frame plane), display-only in the Ricoh popup");

    // Ricoh 360 peripheral detection (default OFF — see header).
    cfgr.opt("Ricoh.yolo_enabled", params.RICOH_YOLO_ENABLED,
            "Ricoh.yolo_enabled");
    cfgr.opt("Ricoh.yolo_thread_period_ms", params.RICOH_YOLO_THREAD_PERIOD_MS,
            "Ricoh.yolo_thread_period_ms — target worker-thread cycle (~20 Hz)");
    cfgr.opt("Ricoh.yolo_n_strips", params.RICOH_YOLO_N_STRIPS,
            "Ricoh.yolo_n_strips");
    cfgr.opt("Ricoh.yolo_strips_per_frame", params.RICOH_YOLO_STRIPS_PER_FRAME,
            "Strips segmented PER FRAME (Ricoh.yolo_strips_per_frame). 0 or >= n_strips = all, the original behaviour. 1 = round-robin: a third of the cost, full…");
    cfgr.opt("Ricoh.semantic_enabled", params.RICOH_SEMANTIC_ENABLED,
            "ADE20K on the scheduled panorama strip (Ricoh.semantic_*)");
    cfgr.opt("Ricoh.semantic_decimation", params.RICOH_SEMANTIC_DECIMATION,
            "run every ricoh frame; raise if the second session costs too much");
    cfgr.opt("Ricoh.yolo_strip_overlap_px", params.RICOH_YOLO_STRIP_OVERLAP_PX,
            "Ricoh.yolo_strip_overlap_px");
    cfgr.opt<float, double>("Ricoh.yolo_merge_iou", params.RICOH_YOLO_MERGE_IOU,
            "Ricoh.yolo_merge_iou — cross-strip dedup threshold");
    cfgr.opt("Ricoh.publish_masks", params.RICOH_PUBLISH_MASKS,
            "Ricoh.publish_masks");
    cfgr.opt("Ricoh.mask_depth", params.RICOH_MASK_DEPTH,
            "Ricoh.mask_depth");
    cfgr.opt("Ricoh.mask_depth_helios_only", params.RICOH_MASK_DEPTH_HELIOS_ONLY,
            "Ricoh.mask_depth_helios_only — helios (co-located) only, exclude bpearl");
    cfgr.opt<float, double>("Ricoh.mask_fg_band_m", params.RICOH_MASK_FG_BAND_M,
            "Ricoh.mask_fg_band_m");
    cfgr.opt<float, double>("Ricoh.azimuth_tune_deg", params.RICOH_AZIMUTH_TUNE_DEG,
            "Ricoh.azimuth_tune_deg (DEGREES, not radians)");

    // [RicohDepth] — monocular depth on the panorama (default OFF; display-only, see depth_processor.h).
    // ── [PlaceMemory] ────────────────────────────────────────────────────────────────────────────
    cfgr.opt("PlaceMemory.enabled", params.PLACE_ENABLED,
            "PlaceMemory.enabled");
    cfgr.opt("PlaceMemory.model_path", params.PLACE_MODEL_PATH,
            "");
    cfgr.opt("PlaceMemory.use_gpu", params.PLACE_USE_GPU,
            "");
    cfgr.opt("PlaceMemory.use_trt", params.PLACE_USE_TRT,
            "first run BUILDS a TRT engine: 30-60 s stall");
    cfgr.opt("PlaceMemory.input_w", params.PLACE_INPUT_W,
            "asserted against the session, not just documented");
    cfgr.opt("PlaceMemory.input_h", params.PLACE_INPUT_H,
            "");
    cfgr.opt("PlaceMemory.n_sectors", params.PLACE_N_SECTORS,
            "22.5 deg; finer than the 90 deg lattice it seeds");
    cfgr.opt<float, double>("PlaceMemory.pool_p", params.PLACE_POOL_P,
            "MEASURED (see place_encoder.h), not chosen");
    cfgr.opt("PlaceMemory.band_lo", params.PLACE_BAND_LO,
            "patch rows kept: excludes ceiling AND the robot");
    cfgr.opt("PlaceMemory.band_hi", params.PLACE_BAND_HI,
            "elevation is CONSTANT around the circle, so carries no bearing info):");
    cfgr.opt("PlaceMemory.center", params.PLACE_CENTER,
            "per-frame mean subtraction; rotation-invariant");
    cfgr.opt<float, double>("PlaceMemory.sector_soft", params.PLACE_SECTOR_SOFT,
            "0 = hard bins; 2.0 = raised-cosine overlap");
    cfgr.opt("PlaceMemory.decimation", params.PLACE_DECIMATION,
            "~2 Hz on the 50 ms ricoh worker");
    cfgr.opt("PlaceMemory.build_map", params.PLACE_BUILD_MAP,
            "ON only for a mapping run");
    cfgr.opt<float, double>("PlaceMemory.insert_min_dist_m", params.PLACE_INSERT_MIN_DIST_M,
            "map DENSITY, not a belief gate");
    cfgr.opt("PlaceMemory.save_every_n", params.PLACE_SAVE_EVERY_N,
            "never write the map from inside a frame");
    cfgr.opt("PlaceMemory.map_path", params.PLACE_MAP_PATH,
            "");
    cfgr.opt("PlaceMemory.map_blob_path", params.PLACE_MAP_BLOB_PATH,
            "");
    cfgr.opt("PlaceMemory.log_queries", params.PLACE_LOG_QUERIES,
            "turn this on FIRST — it is what feeds tools/place_eval");
    cfgr.opt("PlaceMemory.log_grid", params.PLACE_LOG_GRID,
            "raw 16x32x384 fp16, 393 KB/frame; capture only");
    cfgr.opt("PlaceMemory.log_stride", params.PLACE_LOG_STRIDE,
            "per option. Capture sessions only");
    cfgr.opt("PlaceMemory.log_dir", params.PLACE_LOG_DIR,
            "");

    cfgr.opt("RicohDepth.enabled", params.RICOH_DEPTH_ENABLED,
            "RicohDepth.enabled");
    cfgr.opt("RicohDepth.model_path", params.RICOH_DEPTH_MODEL_PATH,
            "RicohDepth.model_path");
    cfgr.opt("RicohDepth.input_size", params.RICOH_DEPTH_INPUT_SIZE,
            "RicohDepth.input_size — must match the exported imgsz");
    cfgr.opt("RicohDepth.use_gpu", params.RICOH_DEPTH_USE_GPU,
            "RicohDepth.use_gpu");
    cfgr.opt("RicohDepth.use_trt", params.RICOH_DEPTH_USE_TRT,
            "RicohDepth.use_trt");
    cfgr.opt("RicohDepth.n_strips", params.RICOH_DEPTH_N_STRIPS,
            "RicohDepth.n_strips — 6 ⇒ 75° views (gnomonic wants narrow)");
    cfgr.opt("RicohDepth.overlap_px", params.RICOH_DEPTH_OVERLAP_PX,
            "RicohDepth.overlap_px — equirect path only; context, never blended");
    cfgr.opt<float, double>("RicohDepth.band_half_elev_deg", params.RICOH_DEPTH_BAND_HALF_ELEV_DEG,
            "RicohDepth.band_half_elev_deg — equirect path only");
    cfgr.opt("RicohDepth.gnomonic", params.RICOH_DEPTH_GNOMONIC,
            "RicohDepth.gnomonic");
    cfgr.opt<float, double>("RicohDepth.gnomonic_fov_deg", params.RICOH_DEPTH_GNOMONIC_FOV_DEG,
            "RicohDepth.gnomonic_fov_deg — 0 ⇒ 360/n_strips*1.25, cap 140");
    cfgr.opt("RicohDepth.zdepth_to_range", params.RICOH_DEPTH_ZDEPTH_TO_RANGE,
            "RicohDepth.zdepth_to_range — additive log correction");
    cfgr.opt("RicohDepth.lidar_diag", params.RICOH_DEPTH_LIDAR_DIAG,
            "RicohDepth.lidar_diag");
    cfgr.opt("RicohDepth.sample_stride", params.RICOH_DEPTH_SAMPLE_STRIDE,
            "RicohDepth.sample_stride");
    cfgr.opt("ZedDepth.enabled", params.ZED_ROOM_DEPTH_ENABLED,
            "ZedDepth.enabled");
    cfgr.opt("ZedDepth.decimate", params.ZED_ROOM_DEPTH_DECIMATE,
            "ZedDepth.decimate — ray-cast on a 1/N grid");
    cfgr.opt("ZedDepth.yolo_depth_enabled", params.ZED_DEPTH_ENABLED,
            "ZedDepth.yolo_depth_enabled");
    cfgr.opt<float, double>("ZedDepth.diff_span_m", params.ZED_DEPTH_DIFF_SPAN_M,
            "ZedDepth.diff_span_m — |error| that saturates the diff colour");
    cfgr.opt("RicohDepth.info_select", params.RICOH_DEPTH_INFO_SELECT,
            "RicohDepth.info_select");
    cfgr.opt<float, double>("RicohDepth.min_gain_nats", params.RICOH_DEPTH_MIN_GAIN_NATS,
            "RicohDepth.min_gain_nats");
    cfgr.opt<float, double>("RicohDepth.suspect_resid_mult", params.RICOH_DEPTH_SUSPECT_RESID_MULT,
            "RicohDepth.suspect_resid_mult");
    cfgr.opt("RicohDepth.save_frames", params.RICOH_DEPTH_SAVE_FRAMES,
            "RicohDepth.save_frames");
    cfgr.opt("RicohDepth.frames_dir", params.RICOH_DEPTH_FRAMES_DIR,
            "RicohDepth.frames_dir");
    cfgr.opt("RicohDepth.frame_jpeg_quality", params.RICOH_DEPTH_FRAME_QUALITY,
            "RicohDepth.frame_jpeg_quality");
    cfgr.opt<float, double>("RicohDepth.metric_lo_m", params.RICOH_DEPTH_METRIC_LO_M,
            "RicohDepth.metric_lo_m");
    cfgr.opt<float, double>("RicohDepth.metric_hi_m", params.RICOH_DEPTH_METRIC_HI_M,
            "RicohDepth.metric_hi_m");
    cfgr.opt("RicohDepth.decimation", params.RICOH_DEPTH_DECIMATION,
            "RicohDepth.decimation — run every Nth worker frame");
    cfgr.opt<float, double>("RicohDepth.overlay_alpha", params.RICOH_DEPTH_OVERLAY_ALPHA,
            "RicohDepth.overlay_alpha — popup blend weight");
    // Ricoh azimuth calibration is no longer a config knob — it lives in the graph (ricoh node's
    // cam_equirect_azimuth_sign/offset), applied by CameraAPI. See retina_params.h.

    // Ego-motion mask-corruption annotation (default ON — pure producer-side metadata).
    cfgr.opt("MaskMotion.enabled", params.MASK_MOTION_ENABLED,
            "MaskMotion.enabled");
    cfgr.opt<float, double>("MaskMotion.exposure_s", params.MASK_MOTION_EXPOSURE_S,
            "MaskMotion.exposure_s (T_exp, blur window)");
    cfgr.opt<float, double>("MaskMotion.timing_jitter_s", params.MASK_MOTION_TIMING_JITTER_S,
            "MaskMotion.timing_jitter_s (σ_t, zero-mean)");
    cfgr.opt<float, double>("MaskMotion.timing_offset_s", params.MASK_MOTION_TIMING_OFFSET_S,
            "MaskMotion.timing_offset_s (δt, known lag → bias)");
    cfgr.opt("MaskMotion.csv_log", params.MASK_MOTION_CSV_LOG,
            "Diagnostic: append per-mask motion-corruption rows to etc/mask_motion_log.csv (verify dot_d≈0 when static, spikes on pan)");
    cfgr.opt("MaskColor.enabled", params.MASK_COLOR_ENABLED,
            "MaskColor.enabled");
    cfgr.opt("MaskColor.cell_px", params.MASK_COLOR_CELL_PX,
            "MaskColor.cell_px");
    cfgr.opt("MaskMotion.pose_extrapolate", params.MASK_POSE_EXTRAPOLATE,
            "Extrapolate the robot pose to the camera CAPTURE stamp using the body-frame velocity on the robot→room RT edge");
    cfgr.opt<float, double>("MaskMotion.pose_extrap_max_dt_s", params.MASK_POSE_EXTRAP_MAX_DT_S,
            "clamp the extrapolation horizon (s)");

    // Human-pose branch (default OFF — see header).
    cfgr.opt("HumanPose.enabled", params.HUMAN_POSE_ENABLED,
            "HumanPose.enabled");
    cfgr.opt("HumanPose.model_path", params.HUMAN_POSE_MODEL_PATH,
            "");
    cfgr.opt<float, double>("HumanPose.conf_thresh", params.HUMAN_POSE_CONF_THRESH,
            "person-detection confidence floor");
    cfgr.opt<float, double>("HumanPose.iou_thresh", params.HUMAN_POSE_IOU_THRESH,
            "");
    cfgr.opt("HumanPose.input_size", params.HUMAN_POSE_INPUT_SIZE,
            "");
    cfgr.opt("HumanPose.use_gpu", params.HUMAN_POSE_USE_GPU,
            "");
    cfgr.opt("HumanPose.use_trt", params.HUMAN_POSE_USE_TRT,
            "pose model is cheap; CUDA EP is fine and avoids the TRT version lock");
    cfgr.opt<std::uint64_t, int>("HumanPose.hold_ms", params.HUMAN_POSE_HOLD_MS,
            "HumanPose.hold_ms");
    cfgr.opt("HumanPose.decimation", params.HUMAN_POSE_DECIMATION,
            "HumanPose.decimation");
    cfgr.opt<float, double>("HumanPose.kp_conf_min", params.SKELETON_KP_CONF_MIN,
            "Per-joint confidence floor below which a keypoint is dropped (NaN) from the skeleton node");
    cfgr.opt("HumanPose.depth_patch", params.SKELETON_DEPTH_PATCH,
            "Half-window (px) for the median-depth patch sampled at each keypoint (0 = single pixel)");

    cfgr.opt("Semantic.enabled", params.SEMANTIC_SEG_ENABLED,
            "Semantic.enabled");
    cfgr.opt("Semantic.model_path", params.SEMANTIC_SEG_MODEL_PATH,
            "graded build: exposes class_probs [1,5,80,80]");
    cfgr.opt<float, double>("Semantic.conf_thresh", params.SEMANTIC_SEG_CONF_THRESH,
            "per-pixel argmax-softmax floor");
    cfgr.opt("Semantic.input_size", params.SEMANTIC_SEG_INPUT_SIZE,
            "");
    cfgr.opt("Semantic.use_gpu", params.SEMANTIC_SEG_USE_GPU,
            "");
    cfgr.opt("Semantic.use_trt", params.SEMANTIC_SEG_USE_TRT,
            "dense model; CUDA EP avoids the TRT version lock");
    cfgr.opt("Semantic.decimation", params.SEMANTIC_SEG_DECIMATION,
            "Run the (heavy dense) model every Nth cycle; the last map is reused on skipped cycles");
    cfgr.opt("Semantic.publish_node", params.SEMANTIC_PUBLISH_NODE,
            "publish the dense label map to a 'semantic' DSR node under 'zed'");
    cfgr.opt("Semantic.publish_probs", params.SEMANTIC_PUBLISH_PROBS,
            "Publish the GRADED CLASS POSTERIOR field on the same 'semantic' node (semantic_class_probs + ids/size)");
    cfgr.opt("DoorApproach.enabled", params.DOOR_APPROACH_LOG,
            "Follow ONE semantic class's evidence against RANGE, one row per semantic frame INCLUDING the frames where nothing was found — see door_approach_log.h…");
    cfgr.opt("DoorApproach.path", params.DOOR_APPROACH_LOG_PATH,
            "");
    cfgr.opt("DoorApproach.label", params.DOOR_APPROACH_LABEL,
            "must be an ADE20K name the loaded model exposes");
    cfgr.opt("DoorSpecialist.enabled", params.DOOR_SPECIALIST_ENABLED,
            "A model trained on doors, run only on frames where the ADE20K path produced no door mask");
    cfgr.opt("DoorSpecialist.model_path", params.DOOR_SPECIALIST_MODEL,
            "");
    cfgr.opt("DoorSpecialist.input_size", params.DOOR_SPECIALIST_INPUT_SIZE,
            "");
    cfgr.opt("DoorSpecialist.use_gpu", params.DOOR_SPECIALIST_USE_GPU,
            "⚠an 8th CUDA session aborts retina — see door_specialist.cpp");
    cfgr.opt("DoorSpecialist.use_trt", params.DOOR_SPECIALIST_USE_TRT,
            "");
    cfgr.opt<float, double>("DoorSpecialist.score_floor", params.DOOR_SPECIALIST_SCORE_FLOOR,
            "a reporting floor, never a decision");
    cfgr.opt("DoorSpecialist.decimation", params.DOOR_SPECIALIST_DECIMATION,
            "Counts SILENT frames only. 1 = every silent frame (~37% duty measured); 4 ≈ 12% duty with the first call inside 0.62 s at 4.8 Hz — well within…");
    cfgr.opt<float, double>("Semantic.publish_min_interval_s", params.SEMANTIC_PUBLISH_MIN_INTERVAL_S,
            "rate cap for the (large) semantic-node publish");
    cfgr.opt("Semantic.publish_masks", params.SEMANTIC_PUBLISH_MASKS,
            "Semantic.publish_masks");
    cfgr.opt("Semantic.accepted_labels", params.SEMANTIC_ACCEPTED_LABELS,
            "");
    cfgr.opt<float, double>("Yolo.probe_floor", params.YOLO_PROBE_FLOOR,
            "Near-miss probe floor: detections in [this, conf_thresh) are recorded to etc/detect_drops.csv instead of vanishing. 0 disables");
    cfgr.opt<float, double>("Semantic.mask_min_area_frac", params.SEMANTIC_MASK_MIN_AREA_FRAC,
            "drop components smaller than this fraction of the frame");
    cfgr.opt<float, double>("Semantic.mask_overlap_drop_frac", params.SEMANTIC_MASK_OVERLAP_DROP_FRAC,
            "drop a component ≥this covered by a YOLO-seg mask (priority)");
    cfgr.opt("Semantic.mask_morph_kernel", params.SEMANTIC_MASK_MORPH_KERNEL,
            "open+close kernel (px) to denoise the class field; ≤1 = off");
    cfgr.opt<float, double>("Semantic.mask_score_default", params.SEMANTIC_MASK_SCORE_DEFAULT,
            "confidence used when per-pixel scores are unavailable");

    cfgr.opt("Sam2.enabled", params.SAM2_ENABLED,
            "Sam2.enabled");
    cfgr.opt("Sam2.encoder_path", params.SAM2_ENCODER_PATH,
            "");
    cfgr.opt("Sam2.decoder_path", params.SAM2_DECODER_PATH,
            "");
    cfgr.opt("Sam2.use_gpu", params.SAM2_USE_GPU,
            "");
    cfgr.opt("Sam2.encoder_use_trt", params.SAM2_ENCODER_USE_TRT,
            "fixed-shape encoder → TRT FP16 win (slow first build, cached)");
    cfgr.opt("Sam2.decoder_use_trt", params.SAM2_DECODER_USE_TRT,
            "dynamic-shape decoder → CUDA (TRT needs profiles); cheap anyway");
    cfgr.opt("Sam2.decimation", params.SAM2_DECIMATION,
            "run every Nth enabled frame");
    cfgr.opt("Sam2.mask_prior", params.SAM2_MASK_PRIOR,
            "feed the YOLO mask as the decoder mask_input prior");
    cfgr.opt<float, double>("Sam2.mask_prior_logit", params.SAM2_MASK_PRIOR_LOGIT,
            "±logit strength of that prior (high → SAM2 echoes YOLO; low → free to tighten)");
    cfgr.opt("Sam2.metrics_log", params.SAM2_METRICS_LOG,
            "write etc/sam2_refine_metrics.csv (YOLO vs SAM2 depth-bleed)");
    cfgr.opt("Sam2.publish_refined", params.SAM2_PUBLISH_REFINED,
            "route refined masks into the published 'masks' node (→ fitters)");
    cfgr.opt("Sam2.refine_labels", params.SAM2_REFINE_LABELS,
            "which classes to refine ([] → all detections)");

    // Custom drawing windows (default ON).
    cfgr.opt("Voxel.show_voxel_viewer", params.SHOW_VOXEL_VIEWER,
            "Voxel.show_voxel_viewer");
    cfgr.opt("Voxel.show_yolo_viewer", params.SHOW_YOLO_VIEWER,
            "Voxel.show_yolo_viewer");
    cfgr.opt("Voxel.show_ricoh_viewer", params.SHOW_RICOH_VIEWER,
            "Voxel.show_ricoh_viewer — creates the button+popup (hidden until toggled)");
    cfgr.opt("Voxel.perf_log", params.PERF_LOG,
            "Voxel.perf_log — per-frame compute/yolo/pose timing → etc/viewer_perf.csv");
    cfgr.opt<float, double>("StreamWatchdog.hold_enter_s", params.HOLD_ENTER_S,
            "StreamWatchdog.hold_enter_s — RGB stale this long → enter hold");
    cfgr.opt<float, double>("StreamWatchdog.recover_s", params.HOLD_RECOVER_S,
            "StreamWatchdog.recover_s — sustained freshness before resuming");

    cfgr.opt("Component.Debug.Verbose", params.VERBOSE_DEBUG,
            "");

    return params;
}

bool preflight_models(const RetinaParams& params)
{
    // One row per model the configuration actually asks for. `required` mirrors the gate in
    // SpecificWorker::initialize() that constructs the stage -- if the two ever disagree, this check
    // is either blocking a valid config or waving a broken one through, so they are worth re-reading
    // together when a stage moves.
    struct Need { bool required; const std::string& path; const char* flag; const char* purpose; };
    const bool ricoh = params.RICOH_YOLO_ENABLED;
    const std::vector<Need> needs{
        { true,
          params.YOLO_MODEL_PATH,        "(always)",
          "instance segmentation - the ZED and ricoh seg stages" },
        { params.HUMAN_POSE_ENABLED,
          params.HUMAN_POSE_MODEL_PATH,  "HumanPose.enabled",
          "human pose" },
        { params.SEMANTIC_SEG_ENABLED or (ricoh and params.RICOH_SEMANTIC_ENABLED),
          params.SEMANTIC_SEG_MODEL_PATH,"Semantic.enabled / Ricoh.semantic_enabled",
          "ADE20K semantic segmentation" },
        { params.SAM2_ENABLED,
          params.SAM2_ENCODER_PATH,      "Sam2.enabled",
          "SAM2 mask refinement (encoder)" },
        { params.SAM2_ENABLED,
          params.SAM2_DECODER_PATH,      "Sam2.enabled",
          "SAM2 mask refinement (decoder)" },
        { params.ZED_DEPTH_ENABLED or (ricoh and params.RICOH_DEPTH_ENABLED),
          params.RICOH_DEPTH_MODEL_PATH, "ZedDepth.yolo_depth_enabled / RicohDepth.enabled",
          "monocular depth" },
        { ricoh and params.PLACE_ENABLED,
          params.PLACE_MODEL_PATH,       "PlaceMemory.enabled",
          "DINOv2 panoramic place memory" },
    };

    std::vector<const Need*> missing;
    for (const auto& n : needs)
    {
        if (not n.required) continue;
        if (n.path.empty())                          // configured ON with no path at all
        { missing.push_back(&n); continue; }
        std::error_code ec;
        if (not std::filesystem::is_regular_file(n.path, ec)) missing.push_back(&n);
    }
    if (missing.empty()) return true;

    std::error_code ec;
    const auto cwd = std::filesystem::current_path(ec);
    std::print("\n"
        "[models] ══════════════════════════════════════════════════════════════════════════════════\n"
        "[models] ★★ retina REFUSES TO START: {} required model file(s) are missing.\n"
        "[models]\n", missing.size());
    for (const auto* n : missing)
    {
        std::print("[models]   {}\n"
                   "[models]       needed by : {}\n"
                   "[models]       purpose   : {}\n",
                   n->path.empty() ? "<no path configured>" : n->path, n->flag, n->purpose);
        if (not n->path.empty())
        {
            const std::filesystem::path p(n->path);
            std::print("[models]       looked at : {}\n",
                       (p.is_absolute() ? p : cwd / p).lexically_normal().string());
        }
    }
    std::print(
        "[models]\n"
        "[models]   Models are NOT shipped in this repository -- you download the upstream weights and\n"
        "[models]   export them to ONNX yourself. See the \"Models\" section of README.md for the\n"
        "[models]   directory layout, the flag -> file table, and the export commands.\n"
        "[models]\n"
        "[models]   ★ Paths are relative to the WORKING DIRECTORY you launched from, not to the binary.\n"
        "[models]     Current working directory: {}\n"
        "[models]     If the files exist but are listed above, you are almost certainly running from\n"
        "[models]     the wrong directory -- launch from the component root.\n"
        "[models]\n"
        "[models]   Or set the corresponding flag to false, and the capability stays off ON PURPOSE\n"
        "[models]   rather than by accident.\n"
        "[models] ══════════════════════════════════════════════════════════════════════════════════\n\n",
        cwd.string());
    return false;
}
