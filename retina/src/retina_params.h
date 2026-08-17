/*
 * retina_params.h
 *
 * Configuration struct for the retina agent + a loader that fills it from a
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

struct RetinaParams
{
    std::string YOLO_MODEL_PATH         = "yolo26l-seg.onnx";
    std::vector<std::string> YOLO_ACCEPTED_LABELS;
    float       YOLO_CONF_THRESH        = 0.25f;
    float       YOLO_IOU_THRESH         = 0.45f;
    float       YOLO_SECOND_BEST_MARGIN = 0.0f;   // recover a rejected top class via an accepted runner-up within this margin (0 = off)
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
    // (Voxel.z_lift_m REMOVED) It offset mask points along ROOM z, so it could only ever be applied to the
    // room-frame array — the camera-frame array the consumers now read could not carry it, and any non-zero
    // value would have silently desynced the two. It was 0.0 and set in no config, i.e. dead.
    bool        TRANSFORMS_INTERPOLATE_RT        = true;
    // Per-mask FOREGROUND depth gate (Voxel.mask_depth_gate_band_m): anchor on the mask's NEAR surface
    // (low percentile of camera-frame depth) and drop points more than this band (m) behind it. The
    // object occludes the background, so those far returns can't be the object — they are the WALL BEHIND
    // that bleeds in when the robot moves (RGB mask ↔ depth skew: silhouette-edge pixels sample the
    // background) or a transparent object's see-through dropout. Coherent, so the radius filter can't
    // catch it. <= 0 disables. Applied in graph_publisher::upload_masks.
    // ⚠ THIS IS A HARD THRESHOLD (a cliff) and it is TEMPORARY. It silently amputates any object deeper
    // than the band (e.g. a 2 m table viewed along its length loses its far ~0.5 m → biased-short fit).
    // Widened to 3.0 m so realistic objects survive while still catching gross wall-behind bleed; SAM2's
    // tighter silhouette already removes most edge bleed at the source. REMOVE this gate entirely once the
    // model-conditioned association (fitter accepts/rejects points by its own surface hypothesis) is tested
    // and live — see [[table-concept-fit-ai-rewrite]] / the mask-bleed→precision plan. Do NOT re-tune the
    // number; the fix is to delete it, not to grow it further.
    float       MASK_DEPTH_GATE_BAND_M           = 3.0f;
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

    // Per-mask APPEARANCE summary (median chromaticity + between-cell variance + effective sample count),
    // published on the 'masks' node for the concept agents' display-mesh tint. Pure annotation: consumers
    // that ignore the attrs are unaffected, and nothing in any geometric fit reads it. MaskColor.enabled
    bool        MASK_COLOR_ENABLED    = true;    // MaskColor.enabled
    // Grid cell size in PIXELS. This is a RESOLUTION choice, not a gate: it sets the spatial scale at which
    // mask pixels are treated as independent samples. Neighbouring pixels on one surface are massively
    // correlated, so aggregating per cell (rather than per pixel) is what keeps the reported variance
    // honest instead of collapsing it like 1/sqrt(N). Bigger cells ⇒ fewer, more-independent samples and
    // heavier boundary erosion; smaller ⇒ more samples that are more correlated. MaskColor.cell_px
    int         MASK_COLOR_CELL_PX    = 8;       // MaskColor.cell_px

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
    bool        SEMANTIC_PUBLISH_NODE   = false;   // publish the dense label map to a 'semantic' DSR node under 'zed'
    float       SEMANTIC_PUBLISH_MIN_INTERVAL_S = 0.5f;   // rate cap for the (large) semantic-node publish

    // Semantic-derived instance masks (semantic_mask_stage): turn the dense ADE20K class field into per-
    // connected-region SegDetections for furniture classes YOLO-seg misses (cabinet/hood/shelf/door),
    // append them to the 'masks' node (SAM2-refined). Default OFF. The master switch also force-runs the
    // semantic model (regardless of the viewer toggle) and enables its per-pixel score map.
    bool        SEMANTIC_PUBLISH_MASKS = false;                                 // Semantic.publish_masks
    std::vector<std::string> SEMANTIC_ACCEPTED_LABELS{"cabinet", "hood", "shelf", "door"};  // ADE20K names
    // Near-miss probe floor: detections in [this, conf_thresh) are recorded to etc/detect_drops.csv
    // instead of vanishing. 0 disables. This is DIAGNOSTIC ONLY — nothing below conf_thresh is ever
    // published; the point is to be able to tell "the object was not there" from "the detector nearly
    // fired", which no log in the fleet can distinguish today.
    float       YOLO_PROBE_FLOOR                = 0.05f;
    float       SEMANTIC_MASK_MIN_AREA_FRAC     = 0.003f;  // drop components smaller than this fraction of the frame
    float       SEMANTIC_MASK_OVERLAP_DROP_FRAC = 0.5f;    // drop a component ≥this covered by a YOLO-seg mask (priority)
    int         SEMANTIC_MASK_MORPH_KERNEL      = 5;       // open+close kernel (px) to denoise the class field; ≤1 = off
    float       SEMANTIC_MASK_SCORE_DEFAULT     = 0.5f;    // confidence used when per-pixel scores are unavailable

    // SAM2.1 mask REFINEMENT (sam2_stage): a two-model ONNX pipeline that sharpens the mask of an
    // ALREADY-LOCATED YOLO object (bbox → box+centre prompt → tight SAM2 mask). Viewer overlay only for
    // now (own PerceptionResult slot; the mask_ingestor route comes later). Default OFF; the heavy 1024²
    // encoder only fires when the ZED-window "SAM2" toggle is on AND a target object is present.
    bool        SAM2_ENABLED       = false;                             // Sam2.enabled
    std::string SAM2_ENCODER_PATH  = "sam2.1_hiera_tiny.encoder.onnx";
    std::string SAM2_DECODER_PATH  = "sam2.1_hiera_tiny.decoder.onnx";
    bool        SAM2_USE_GPU       = true;
    bool        SAM2_ENCODER_USE_TRT = false;   // fixed-shape encoder → TRT FP16 win (slow first build, cached)
    bool        SAM2_DECODER_USE_TRT = false;   // dynamic-shape decoder → CUDA (TRT needs profiles); cheap anyway
    int         SAM2_DECIMATION    = 1;                                 // run every Nth enabled frame
    bool        SAM2_MASK_PRIOR    = true;    // feed the YOLO mask as the decoder mask_input prior
    float       SAM2_MASK_PRIOR_LOGIT = 4.0f; // ±logit strength of that prior (high → SAM2 echoes YOLO; low → free to tighten)
    bool        SAM2_METRICS_LOG   = false;   // write etc/sam2_refine_metrics.csv (YOLO vs SAM2 depth-bleed)
    bool        SAM2_PUBLISH_REFINED = false; // route refined masks into the published 'masks' node (→ fitters);
                                              // also keeps the SAM2 stage running regardless of the overlay toggle
    std::vector<std::string> SAM2_REFINE_LABELS{"table", "chair", "monitor", "tv", "bottle"};   // empty → all classes

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

    // ── ZED worker pull pacing (Yolo.zed_thread_period_ms) ────────────────────────────────────────
    // How often the ZED PerceptionWorker asks its source for a frame when the last ask came up empty.
    // This is a POLL GRANULARITY, not a rate limit: a cycle that finds a frame runs at its own cost and
    // never sleeps, so lowering this cannot make perception run faster than the models allow.
    //
    // ★IT WAS 15 ms AND THAT WAS THE BINDING CONSTRAINT once SAM2 came off. Measured (468 frames,
    // etc/viewer_perf_zed_worker.csv): real work 25.3 ms against a camera period of 25 ms, yet the loop
    // ran 49.1 ms — processing exactly 1 frame in 1.96. The worker finished just as the next frame was
    // landing, asked a moment too early, missed, and then slept 15 ms — losing a whole frame to poll
    // granularity, not to compute. The loop-period histogram showed the two modes (32-39 and 40-47 ms).
    // 4 ms shrinks that miss window to ~4 ms. The cost is a more frequent idle poll (250 Hz vs 66 Hz) on
    // the cheap null-fetch path; raise it again if that ever shows up as idle CPU.
    int         ZED_THREAD_PERIOD_MS       = 4;       // Yolo.zed_thread_period_ms

    // Ricoh 360 peripheral detection: OWN YOLO model/session (rc::RicohYoloWorker), run over the
    // panorama split into RICOH_YOLO_N_STRIPS vertical strips (see YoloProcessor::detect_segmentation_360)
    // on a DEDICATED thread — decoupled from compute()'s budget, paced to RICOH_YOLO_THREAD_PERIOD_MS
    // (~20 Hz default) rather than traded against the main perception rate via decimation. A coarse,
    // no-depth "glance" signal — NOT wired to DSR yet (see RICOH_360_PERIPHERAL_DETECTION.md). Default OFF.
    bool        RICOH_YOLO_ENABLED         = false;   // Ricoh.yolo_enabled
    int         RICOH_YOLO_THREAD_PERIOD_MS= 50;      // Ricoh.yolo_thread_period_ms — target worker-thread cycle (~20 Hz)
    int         RICOH_YOLO_N_STRIPS        = 3;       // Ricoh.yolo_n_strips
    // Strips segmented PER FRAME (Ricoh.yolo_strips_per_frame). 0 or >= n_strips = all, the original
    // behaviour. 1 = round-robin: a third of the cost, full 360 coverage every 3 frames.
    int         RICOH_YOLO_STRIPS_PER_FRAME = 0;
    // ADE20K on the scheduled panorama strip (Ricoh.semantic_*). OFF by default: it is a second ONNX
    // session on the ricoh worker, and it is the ONLY way cabinet/hood ever see a peripheral detection.
    bool        RICOH_SEMANTIC_ENABLED     = false;
    int         RICOH_SEMANTIC_DECIMATION  = 1;
    int         RICOH_YOLO_STRIP_OVERLAP_PX= 80;      // Ricoh.yolo_strip_overlap_px
    float       RICOH_YOLO_MERGE_IOU       = 0.5f;    // Ricoh.yolo_merge_iou — cross-strip dedup threshold
    // Publish the ricoh 360 detections into the shared "masks" node as NO-DEPTH bearing slices (Part B,
    // RICOH_360_PERIPHERAL_DETECTION.md). azimuth_offset trims the panorama-column→room-bearing mapping for
    // the ricoh's mounting yaw + the panorama's 0-column convention — PROVISIONAL, verify live vs the
    // descriptor projection model before any consumer relies on the bearing (Part C is not built yet).
    bool        RICOH_PUBLISH_MASKS        = true;    // Ricoh.publish_masks
    // ── [RicohDepth] — monocular depth (yolo26l-depth) on the 360 panorama, same strip trick ────────
    // Its OWN slicing, deliberately not shared with the seg strips above: depth needs an ELEVATION BAND
    // (a perspective-trained depth head degrades toward the equirect poles) and a much smaller overlap
    // (the overlap is context only — nothing is merged across it, because the model's scale is
    // per-strip arbitrary). The defaults make a strip exactly 768 px wide = the model input, so there
    // is NO horizontal resampling at all. DISPLAY-ONLY — read depth_processor.h before changing that.
    bool        RICOH_DEPTH_ENABLED        = false;   // RicohDepth.enabled
    std::string RICOH_DEPTH_MODEL_PATH     = "models/yolo26/yolo26l-depth.onnx";  // RicohDepth.model_path
    int         RICOH_DEPTH_INPUT_SIZE     = 768;     // RicohDepth.input_size — must match the exported imgsz
    bool        RICOH_DEPTH_USE_GPU        = true;    // RicohDepth.use_gpu
    bool        RICOH_DEPTH_USE_TRT        = false;   // RicohDepth.use_trt
    int         RICOH_DEPTH_N_STRIPS       = 6;       // RicohDepth.n_strips — 6 ⇒ 75° views (gnomonic wants narrow)
    int         RICOH_DEPTH_OVERLAP_PX     = 64;      // RicohDepth.overlap_px — equirect path only; context, never blended
    float       RICOH_DEPTH_BAND_HALF_ELEV_DEG = 60.f;// RicohDepth.band_half_elev_deg — equirect path only
    // Render each strip as a virtual PINHOLE view before inference (and map the depth back after).
    // Raw equirect crops measured ANTI-CORRELATED with geometry — see depth_processor.h.
    bool        RICOH_DEPTH_GNOMONIC       = true;    // RicohDepth.gnomonic
    float       RICOH_DEPTH_GNOMONIC_FOV_DEG = 0.f;   // RicohDepth.gnomonic_fov_deg — 0 ⇒ 360/n_strips*1.25, cap 140
    bool        RICOH_DEPTH_ZDEPTH_TO_RANGE = true;   // RicohDepth.zdepth_to_range — additive log correction
    // Per-view Pearson r of model log-depth vs helios LiDAR range at the same panorama pixels, plus the
    // least-squares (a,b) that would make the views metric. etc/ricoh_depth_lidar.csv. Cheap: one pass
    // over the already-reprojected cloud.
    bool        RICOH_DEPTH_LIDAR_DIAG     = true;    // RicohDepth.lidar_diag
    // Keep every Nth LiDAR-anchored sample when recording. Neighbouring returns on one surface are
    // redundant, so 1-in-16 loses no information the fit can use and keeps the file ~40x smaller
    // (635 MB for 406 frames at stride 1). The CORRELATION still uses every hit.
    int         RICOH_DEPTH_SAMPLE_STRIDE  = 16;      // RicohDepth.sample_stride
    // Save the panorama alongside each ADMITTED dataset frame, keyed by the frame's own capture stamp
    // (`<stamp_ms>.jpg`, the same stamp the CSV row carries — no schema change needed to join them).
    // Needed ONLY by the offline enrichment pass: YOLO-sem needs pixels to segment ceiling/floor/wall,
    // and the CSV holds numbers. ~250-400 kB per frame, so a few hundred frames is ~150 MB.
    // ── Epistemic frame admission (rc::depth::InfoSelector) ───────────────────────────────────────
    // Admit a frame by the MUTUAL INFORMATION it carries about the map parameters instead of by how
    // far the robot moved. Saturates by itself: as Λ grows, repeated views of the same geometry score
    // ~0 nats and are dropped, while an under-constrained view (a long sightline) still scores high.
    bool        RICOH_DEPTH_INFO_SELECT    = true;    // RicohDepth.info_select
    // Minimum information a frame must add to be worth storing, in NATS — an information unit, not a
    // geometric cutoff. Raise it to keep only strongly novel views; 0 admits everything scored.
    float       RICOH_DEPTH_MIN_GAIN_NATS  = 0.50f;   // RicohDepth.min_gain_nats
    // Consistency guard. A frame whose median residual under the current map exceeds this multiple of
    // the map's own anchored residual is treated as SUSPECT and dropped unscored — D-optimality rates
    // a corrupted frame as maximally informative, so novelty must be earned by data that agrees.
    float       RICOH_DEPTH_SUSPECT_RESID_MULT = 3.0f; // RicohDepth.suspect_resid_mult
    // ── ZED depth-model overlay (validation, not navigation) ──────────────────────────────────────
    // Runs the SAME depth model on the perspective ZED frame so its output can be compared against
    // the camera's own MEASURED depth. The ZED does not need the model — it measures — but it is the
    // only dense, full-frame ground truth available, where the LiDAR gives a horizon stripe. Gated by
    // the ZED window's ZDepth button, so the model runs only while the comparison is on screen.
    // The ZED window's ZDepth overlay compares the ROOM BELIEF's predicted depth (floor/walls/
    // ceiling ray-cast) against the camera's MEASURED depth. No neural model: the ZED measures depth
    // and room_concept predicts it, so the difference is the belief's own per-pixel residual.
    bool        ZED_ROOM_DEPTH_ENABLED     = true;    // ZedDepth.enabled
    int         ZED_ROOM_DEPTH_DECIMATE    = 2;       // ZedDepth.decimate — ray-cast on a 1/N grid
    // ZED yolo-depth stage: NOT needed (the camera measures depth). Kept off; the validation it gave
    // is superseded by comparing against the room belief instead of against a monocular guess.
    bool        ZED_DEPTH_ENABLED          = false;   // ZedDepth.yolo_depth_enabled
    float       ZED_DEPTH_DIFF_SPAN_M      = 1.0f;    // ZedDepth.diff_span_m — |error| that saturates the diff colour
    bool        RICOH_DEPTH_SAVE_FRAMES    = true;    // RicohDepth.save_frames
    std::string RICOH_DEPTH_FRAMES_DIR     = "etc/depth_frames";  // RicohDepth.frames_dir
    int         RICOH_DEPTH_FRAME_QUALITY  = 92;      // RicohDepth.frame_jpeg_quality
    // Fixed metres span of the corrected (metric) overlay ramp — used only once a map + anchor exist.
    float       RICOH_DEPTH_METRIC_LO_M    = 0.4f;    // RicohDepth.metric_lo_m
    float       RICOH_DEPTH_METRIC_HI_M    = 8.0f;    // RicohDepth.metric_hi_m
    int         RICOH_DEPTH_DECIMATION     = 1;       // RicohDepth.decimation — run every Nth worker frame
    float       RICOH_DEPTH_OVERLAY_ALPHA  = 0.65f;   // RicohDepth.overlay_alpha — popup blend weight
    // Ricoh.mask_depth — reproject the lidar into the panorama and publish the 360 masks as FULL 3D
    // masks (has_depth=1, room+zed support points, mask_depth_var) instead of bearing-only. Default OFF:
    // flip ON only once the concept agents read mask_depth_var into R (else sparse ricoh masks are
    // over-trusted). See common/depth_projection.
    bool        RICOH_MASK_DEPTH           = false;   // Ricoh.mask_depth
    bool        RICOH_MASK_DEPTH_HELIOS_ONLY = true;  // Ricoh.mask_depth_helios_only — helios (co-located) only, exclude bpearl
    // Ricoh.mask_fg_band_m — FOREGROUND depth band (m) for the 360 mask depth-fill. Lidar hits inside a
    // detection's silhouette are anchored on the NEAREST surface (the object facing the ricoh, which
    // occludes the floor behind it); returns farther than this band are the occluded floor / under-object
    // floor and are dropped. It's a physical object-depth-extent prior (~a table's along-ray depth), not a
    // tuning cutoff — flagged per [[no-threshold-patches]]. Larger = admits deeper objects (+ more floor).
    float       RICOH_MASK_FG_BAND_M       = 0.60f;   // Ricoh.mask_fg_band_m
    // Ricoh.azimuth_tune_deg — LIVE azimuth fine-tune in DEGREES, applied as an extra yaw on room_T_ricoh
    // on top of the graph's cam_equirect_* intrinsics. Affects BOTH the projection overlay and the
    // detection→bearing path (one transform). Dial out the last few degrees of residual against the
    // RGB360 Lidar overlay (e.g. 2 or -3), then bake the total into the ricoh node's
    // cam_equirect_azimuth_offset in shadow.json and set this to 0. Restart retina only to change it.
    float       RICOH_AZIMUTH_TUNE_DEG     = 0.0f;    // Ricoh.azimuth_tune_deg (DEGREES, not radians)
    // NOTE: the panorama azimuth calibration (mirror sign + seam zero) now lives in the GRAPH as the
    // ricoh node's cam_equirect_azimuth_sign / cam_equirect_azimuth_offset intrinsics, applied by the
    // shared CameraAPI equirectangular model. Both the 360 projection overlay AND the detection→bearing
    // publisher go through CameraAPI (project / ray_from_pixel), so there are no RICOH_AZIMUTH_* config knobs.

    // Custom drawing windows (attach to the DSR GUI if available). Both default ON.
    bool        SHOW_VOXEL_VIEWER = true; // Voxel.show_voxel_viewer
    bool        SHOW_YOLO_VIEWER  = true; // Voxel.show_yolo_viewer
    bool        SHOW_RICOH_VIEWER = true; // Voxel.show_ricoh_viewer — creates the button+popup (hidden until toggled)
    bool        PERF_LOG          = false; // Voxel.perf_log — per-frame compute/yolo/pose timing → etc/viewer_perf.csv
    // Input-stream publish-hold watchdog: if the RGB feed goes stale (producer stall), stop publishing
    // perception (better nothing than stale masks). Debounced with hysteresis.
    float       HOLD_ENTER_S      = 1.5f;    // StreamWatchdog.hold_enter_s — RGB stale this long → enter hold
    float       HOLD_RECOVER_S    = 1.0f;    // StreamWatchdog.recover_s — sustained freshness before resuming

    bool        VERBOSE_DEBUG = false;
};

// Fill a RetinaParams from the RoboComp ConfigLoader (every key optional).
RetinaParams load_retina_params(const ConfigLoader& configLoader);
