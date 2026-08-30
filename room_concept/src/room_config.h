/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

// RoomConfig — the room_concept agent's worker-level configuration, plus a
// single entry point that loads EVERY config block (the agent's own params, the
// RoomConcept localizer params and the EpistemicController/planner params) from
// the ConfigLoader. Keeps the ~150 lines of load_* boilerplate out of
// SpecificWorker::initialize().

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <QRectF>

class ConfigLoader;

namespace rc
{
class RoomConcept;
class EpistemicController;

struct RoomConfig
{
    float ROBOT_WIDTH  = 0.460f;   // m
    float ROBOT_LENGTH = 0.480f;   // m
    float ROBOT_HEIGHT = 1.6f;     // m, obstacle cloud ceiling

    // Lidar
    // Per-device high LiDAR plane: points arrive in the DEVICE frame (metres) and must be
    // transformed device->robot via the DSR RT tree. robot_concept publishes onto this same
    // node when it bridges from Ice, so there is one name either way.
    std::string LIDAR_HELIOS_NAME     = "helios";
    // Destination frame for the device->robot transform (the mount RT edge parent, e.g. body->helios).
    std::string LIDAR_ROBOT_FRAME     = "";   // empty ⇒ auto-derived from the type-"robot" node at init

    // ── PLATFORM OVERLAY: one config for every robot ────────────────────────────────────────────
    // Most of this file is POLICY and is the same everywhere. A handful of values are PHYSICAL —
    // measured on one machine and meaningless on another — and keeping two whole files apart so
    // those few can differ is what let the rest drift: on 2026-08-29 the two platform files had 14
    // keys present in only one of them, four thresholds quietly different, and
    // OdomVarianceInjection true on one and false on the other, so a producer fix landed and was
    // ignored. The cost of the split was paid by the 95% that should never have differed.
    //
    // So: ONE file, with the physical few in a per-robot section applied by NAME:
    //     [Platform.Shadow]  mountPitchSigma = 0.0035
    //     [Platform.P3Bot]   mountPitchSigma = 0.0017
    //
    // ★ READ at load time, APPLIED at init. The robot's identity comes from the graph (the
    //   type-"robot" node), which is not up when the config is parsed — the same constraint that
    //   makes LIDAR_ROBOT_FRAME auto-derive rather than be configured. Every section is therefore
    //   parsed up front and the matching one applied once the name is known.
    // ★ A section that does not match ANY robot is not an error; a robot with NO section is not one
    //   either — it simply keeps the shared defaults. What IS an error is a value that must be
    //   measured per machine silently inheriting another machine's number, which is why the
    //   physical ones are listed here explicitly rather than "whatever happens to be in the file".
    struct PlatformOverlay
    {
        // Camera mount — measured on one machine, meaningless on another.
        std::optional<float> mount_pitch_sigma, mount_height_sigma, mount_yaw_sigma;
        std::optional<float> mount_yaw_correction, wall_position_sigma;
        std::optional<std::string> image_edge_camera;
        // Odometry noise — the wheels, the encoders and the floor they run on. Every one of these
        // was MEASURED on its robot (see the ZUPT and velCov work); inheriting another machine's
        // number is exactly the silent failure this overlay exists to prevent.
        std::optional<float> cmd_noise_rot, cmd_noise_trans;
        std::optional<float> zupt_density_v, zupt_density_omega;
        std::optional<float> odom_preint_sigma_omega, odom_preint_scale_omega;
        // Body geometry: the safety distances scale with the footprint.
        std::optional<float> sdf_safe, sdf_danger;
        // Which channels this machine runs, and how fast it runs them.
        std::optional<bool>  use_command_velocity_prior, calib_pivot_enabled, odom_sample_log;
        std::optional<int>   period_compute;

        // ── DRIFT, not physics ──────────────────────────────────────────────────────────────
        // These are solver settings and thresholds that SHOULD be the same on every machine.
        // They are per-robot only because the two config files diverged while they were apart:
        // shadow's file received later work that p3bot's never did, so p3bot has been running
        // the older code defaults with nothing in its file to say so. They live here to make the
        // merge change no running behaviour and to make the divergence VISIBLE side by side —
        // not because anyone decided a robot needs its own value. Collapse them into the shared
        // section once each pair has been judged; every one removed is a real simplification.
        std::optional<float> recovery_loss_threshold;
        std::optional<float> prediction_trust_factor;
        std::optional<float> rotation_sdf_coupling;
        std::optional<float> boundary_hessian_quality_threshold;
        std::optional<float> boundary_mu_quality_threshold;
        std::optional<float> symmetry_good_fit_mse;
        std::optional<float> gn_loss_rel_tol;
        std::optional<int> gn_max_iters;
        std::optional<int> torch_num_threads;
        std::optional<bool> adaptive_cov_enabled;
        std::optional<bool> window_stride_enabled;
        std::optional<bool> boundary_fej_schur;
        std::optional<bool> hier_prec_boundary_enabled;
        std::optional<bool> corner_early_exit_check;
        std::optional<float> w_ior;
        std::optional<float> belief_forget_time;
        std::optional<float> object_anchor_meas_sigma_xy;
        std::optional<float> stable_sdf_mse_max;
    };
    std::map<std::string, PlatformOverlay> platform_overlays;

    // ── SCENARIO OVERLAY: a robot is not a place ────────────────────────────────────────────────
    // The same P3Bot runs in the WAF room and in the apartment; the same apartment hosts P3Bot and
    // Shadow. Robot identity and scenario identity are INDEPENDENT axes and a config keyed on one
    // cannot express the other, which is why there are two overlays rather than one.
    // Keyed on the `scenario_name` graph attribute, published by robot_concept — the component that
    // exists in simulation and in reality alike, unlike the bridge.
    /// Where the layout SVGs live. Shared by every scenario, so a scenario section names only the
    /// FILE and the directory is stated once. Relative to the component's working directory, which
    /// is the component folder when launched as `bin/room_concept etc/config`.
    /// ★ The layouts moved OUT of room_concept's own folder on 2026-08-29: a floor plan is not a
    ///   property of the agent that happens to load it — every concept agent localises against the
    ///   same room, and one of them owning the file made it look otherwise.
    std::string LAYOUT_DIR = "../layouts";

    struct ScenarioOverlay
    {
        std::optional<std::string> room_layout_svg;
        std::optional<float>       room_height;
        std::optional<bool>        recenter_room_polygon;
        /// Upper bound of the LiDAR "high" band. It is a property of the CEILING, not of the robot:
        /// the band must exclude the ceiling return, so a 3 m room and a 2.4 m room need different
        /// caps for the same LiDAR.
        std::optional<float>       lidar_high_max_height;
        /// How close a viewpoint may be planned to a wall. A cramped apartment and an open lab room
        /// do not admit the same margin, and the robot is unchanged between them.
        std::optional<float>       target_wall_margin;
        /// WHICH object subtypes anchor the pose — a property of what stands in the room, not of the
        /// robot looking at it. The apartment anchors on the refrigerator (deliberately: the table's
        /// pin is suspect); the WAF room has no refrigerator at all, so anchoring on one there would
        /// leave the channel with nothing to find, which is not the same thing as turning it off.
        std::optional<std::vector<std::string>> object_anchor_subtypes;
    };
    std::map<std::string, ScenarioOverlay> scenario_overlays;
    std::vector<std::string> apply_scenario(const std::string& scenario);
    /// Apply the section matching `robot`, if there is one. Returns what it changed, for the log:
    /// a silent overlay is how a machine ends up running another machine's constants.
    std::vector<std::string> apply_platform(const std::string& robot);
    /// The half of the overlays whose targets do NOT live in this struct — the localiser's and the
    /// planner's own params, which are loaded into their objects directly. Split out rather than
    /// merged because RoomConfig does not own those objects; the caller does.
    /// Returns what it changed, for the same log line as apply_platform.
    std::vector<std::string> apply_platform_to(const std::string& robot,
                                               rc::RoomConcept& room_concept,
                                               rc::EpistemicController& epistemic);
    std::vector<std::string> apply_scenario_to(const std::string& scenario,
                                               rc::EpistemicController& epistemic);
                                              // (so the SDF optimises in the SAME frame the robot↔room RT
                                              // is published onto). Set explicitly only to override.
    float MAX_LIDAR_HIGH_RANGE        = 100.f;  // m
    int   LIDAR_LOW_DECIMATION_FACTOR = 1;
    float LIDAR_HIGH_MIN_HEIGHT       = 1.5f;   // m
    float LIDAR_HIGH_MAX_HEIGHT       = 2.0f;   // m — upper bound of the high band (excludes the ceiling)
    float LIDAR_HIGH_FLOOR_HEIGHT     = 0.15f;  // m
    // Startup geometry self-check: from the first few sweeps, detect the floor plane (warn if it
    // disagrees with the robot mount geometry -> a mis-set LiDAR mount height) and the ceiling plane
    // (cap the high band at ceiling - margin so only upper-wall points feed the localizer).
    bool  LIDAR_STARTUP_GEOMETRY_CHECK = true;
    int   LIDAR_STARTUP_CHECK_SWEEPS   = 15;    // sweeps to accumulate before running the check
    float LIDAR_FLOOR_TOLERANCE        = 0.06f; // m — warn if the measured floor is off by more
    float LIDAR_CEILING_MARGIN         = 0.15f; // m — keep wall points this far below the ceiling

    // View
    QRectF GRID_MAX_DIM{-5, -5, 10, 10};
    int    MAX_LIDAR_DRAW_POINTS = 500;

    // Localizer
    bool        PREDICTION_EARLY_EXIT = true;
    std::string OptimizerType         = "LBFGS";
    std::string ROOM_LAYOUT_SVG       = "beta_layout.svg";  // config: RoomConcept.RoomLayoutSvg
    // Put the room-frame origin on the layout's geometric (bounding-box) centre instead of wherever
    // the SVG author happened to place (0,0). Only affects plans traced from a corner
    // (apartamento_layout); layouts already drawn about their middle barely move.
    // NOTE when flipping this: the room frame shifts, so etc/last_robot_pose.txt (seed pose, in room
    // coords) and any object RT edges persisted under the DSR `room` node are stale by the offset.
    bool        RECENTER_ROOM_POLYGON = true;   // config: RoomConcept.RecenterRoomPolygon
    float       ODOMETRY_NOISE_FACTOR = 0.0f;
    // One CSV line per ARRIVING odometry sample (etc/odom_samples.csv), for measuring the stream's
    // own statistics. Deliberately NOT the per-cycle debug log: that one is written once per lidar
    // sweep and records only the LATEST odometry sample, which aliases a 50 Hz stream onto ~20 Hz
    // rows -- fatal for any autocorrelation question, since the aliasing invents correlation at one
    // rate and destroys it at another. Off by default; it is an instrument, not telemetry.
    bool        ODOM_SAMPLE_LOG = false;        // config: RoomConcept.OdomSampleLog
    /// Where the calibration WINDOW is kept between runs. Evidence (episodes + closed pivots), never
    /// the fitted parameters — restoring a fitted value as a prior mean is the ratchet that walked
    /// the gyro bias to the wrong sign. Delete the file, or press Reset in the calibration window,
    /// to return to the priors.
    std::string CALIB_STATE_FILE = "etc/motion_calib_state.csv";  // config: RoomConcept.CalibStateFile



    // DSR stabilization: this many consecutive "stable" frames before creating the
    // room node and re-parenting the robot under it.
    int   STABLE_FRAMES_REQUIRED = 30;
    float STABLE_SDF_MSE_MAX     = 0.06f;
    float STABLE_COV_TT_MAX      = 0.001f;

    // Static-room mode: ADOPT a pre-seeded room/table from the bootstrap graph
    // instead of deleting+recreating it, and do NOT write room pose / robot->room.
    bool  PRESERVE_BOOTSTRAP_ROOM = false;

    // ODOMETRY-DRIVEN COMPLEMENTARY-FILTER PUBLISH (v2). The robot↔room RT edge is published from the
    // ODOMETRY callback (event-driven, at the odometry rate, e.g. 20-30 Hz) — decoupled from the bursty
    // ~15 Hz lidar/compute path. Each odometry sample integrates a SMALL dt step (no catch-up jump) and
    // publishes a monotonic now-stamped block with a GROWING covariance. Lidar corrections are BLENDED
    // into the filter (complementary, gain below) and NOT written to RT directly (no past-stamped block
    // interleaving). This is the smooth replacement for the v1 QTimer predict-publish (which injected
    // 17 m/s spikes via per-correction catch-up + hard snap + block interleaving — see git history).
    // Robot BODY-FRAME velocity (adv=fwd, side=lat, rot=yaw-rate) is published on the robot↔room RT
    // edge (rt_translation_velocity / rt_rotation_euler_xyz_velocity) so the controller reads velocity
    // DIRECTLY instead of differentiating the pose (which would turn lidar-correction jumps into spikes).
    // A diagonal velocity covariance (variances below) goes on rt_covariance_velocity.
    float ROBOT_VEL_COV_ADV            = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_SIDE           = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_ROT            = 0.01f;   // (0.1 rad/s)²

    // ---- The calibration pivot (afford_calib) ----
    // OFF. This affordance has never driven a robot: turning on the spot for four full turns is a real
    // motion with a real footprint, and it goes live on a watched run or not at all. The PASSIVE half
    // — watching how well the odometry's rotation scale is known — runs regardless and commands
    // nothing, because free data is free whether or not anyone acts on it.
    bool  CALIB_PIVOT_ENABLED          = false;   // RoomConcept.CalibPivotEnabled
    // ★TESTING SCAFFOLD. >0 forces the advertised gain of afford_calib to this many nats so it wins
    // the controller's EFE selection on demand. The manoeuvre has never run to closure because a
    // calibrated robot prices it honestly at fractions of a nat and it loses every contest — so the
    // four-turn sequence downstream of selection is untested. This buys those tests and nothing else:
    // the true valuation is still computed and logged beside the forced one, and no result obtained
    // with this set may be quoted as a valuation. Back to 0 when the pivot works.
    float CALIB_FORCED_GAIN_NATS       = 0.f;     // RoomConcept.CalibForcedGainNats
    bool  PUBLISH_AFFORDANCE           = true;    // EpistemicController.PublishAffordance — publish the room
                                                  // exploration affordance. false ⇒ room never offers an
                                                  // affordance (so it can't out-compete object affordances in
                                                  // the controller's EFE selection — e.g. the 360-glance test).
    // ---- Kinematic clamp on the published pose ----
    // Bounds the pose delta between two published RT blocks to what the robot can physically do in the
    // elapsed time. Not a tuning threshold: it is the support of the motion model, so a delta outside
    // it cannot be a report about where the robot went. The un-applied residual is added to the
    // published covariance so the lag is visible rather than hidden.
    // ★ The limits must be the CONSUMER's, not this agent's own beliefs — room_concept's MaxAdvSpeed /
    // MaxRotSpeed (0.9 / 0.75) disagree with what the controller actually commands (0.7 / 0.8) and
    // neither is a superset. These default to the controller's and must be kept tracking it.
    bool  POSE_CLAMP_ENABLED           = true;    // RoomConcept.PoseClampEnabled
    // ── THE POSE CLAMP IS BOUNDED BY THE HARDWARE, NOT BY A CONTROLLER'S PREFERENCE ──────────────
    // ★★★THESE WERE COPIES OF THE CONTROLLER'S MaxAdvSpeed/MaxRotSpeed, and that was the defect: a
    // COMFORT SETTING became a hard rate limiter on the PUBLISHED yaw. Measured 2026-08-28: the clamp
    // held the published yaw rate at exactly 0.8 rad/s while the robot turned at 3.5.
    // ★WHAT ACTUALLY BOUNDS A PLAUSIBLE POSE CORRECTION is what the ROBOT CAN PHYSICALLY DO. The clamp
    // does not bound motion — it bounds the part of a published delta left over AFTER measured motion
    // is accounted for, i.e. the part with no dynamical support. A correction implying more rotation
    // than the hardware can produce in dt is a teleport whatever any consumer's speed preference is.
    // So these are overwritten at runtime from robot_max_linear_speed / robot_max_rot_speed on the
    // robot node, which robot_concept publishes from the base component's own config.
    // ★The values below are the FALLBACK, used when the robot node says nothing — absent means
    // "unknown, keep what you have", never zero, which would pin every correction to zero.
    // ⚠They are no longer the controller's numbers and must not be re-synced to them: raising the
    // controller's MaxRotSpeed is a policy change and must not touch this.
    float POSE_CLAMP_V_MAX             = 0.7f;    // m/s   — fallback; capability supersedes it
    float POSE_CLAMP_W_MAX             = 0.8f;    // rad/s — fallback; capability supersedes it
    // Gap beyond which the clamp is skipped entirely (stream restart, agent stall, relocalization after
    // a long silence). Clamping across a long gap would slew the pose in at v_max for many frames.
    float POSE_CLAMP_MAX_DT_S          = 0.5f;    // s

    // ---- Execution-stall watchdog (RoomSceneGraph::break_execution_stall) ----
    // While the controller holds the afford_room execution claim the planner is idle by design, so
    // a target the controller can never reach parks the whole run. If the robot's closest approach
    // to the claimed target fails to improve by EXEC_STALL_PROGRESS_M for EXEC_STALL_TIMEOUT_S, the
    // claim is released and the target is stamped visited so a different one is selected. This is a
    // LIVENESS guard on the affordance handshake, not a modelling threshold — see the header note.
    // Set EXEC_STALL_TIMEOUT_S = 0 to disable.
    float EXEC_STALL_TIMEOUT_S         = 25.0f;   // EpistemicController.ExecStallTimeout (s)
    float EXEC_STALL_PROGRESS_M        = 0.10f;   // EpistemicController.ExecStallProgress (m)

    bool  PREDICT_PUBLISH_ENABLED      = true;    // PredictPublish.enabled (drives RT from odometry)
    float PREDICT_PUBLISH_MAX_COAST_S  = 1.0f;    // stop publishing if no lidar correction for this long
    float PREDICT_PROCESS_NOISE_XY     = 0.04f;   // (m/√s)² → variance growth m²/s on x,y while coasting
    float PREDICT_PROCESS_NOISE_THETA  = 0.05f;   // (rad/√s)² → variance growth rad²/s on theta
    float PREDICT_MAX_DT_S             = 0.1f;    // clamp per-sample integration dt (a missed/late odom
                                                  // sample must not dead-reckon a big jump); also caps
                                                  // the lidar-lag used to extrapolate the correction
    // Correction is NOT applied as a position jump (that injects velocity spikes the MPPI differentiates
    // into garbage — measured post-blend 3.46 m/s vs pure-odom 0.84). Instead each lidar correction sets
    // a residual that is BLED IN smoothly over the 30 Hz odometry ticks, slew-limited, so it's a small
    // continuous velocity contribution. Per tick: step = min(blend_gain·residual, max_blend_step).
    float PREDICT_BLEND_GAIN           = 0.2f;    // fraction of the residual applied per odometry tick
    float PREDICT_MAX_BLEND_STEP_M     = 0.01f;   // max position bleed per tick (→ ≤0.3 m/s @30Hz)
    float PREDICT_MAX_BLEND_STEP_RAD   = 0.03f;   // max heading bleed per tick
    // Outlier-aware: a residual bigger than these is a relocalization/jump (the gentle slew would take
    // ~30 s to catch up → the pose strands metres from truth). SNAP to it immediately instead — bounds
    // divergence to this threshold; the rare spike on a real relocalization is legitimate. Small
    // residuals (normal drift) are still slewed smoothly.
    float PREDICT_SNAP_THRESH_M        = 0.30f;
    float PREDICT_SNAP_THRESH_RAD      = 0.40f;

    float room_height = 2.4f;  // m, room DSR node attribute

    // Debug bootstrap table hanging from the room node
    bool  BOOTSTRAP_TABLE_ENABLED = true;
    float BOOTSTRAP_TABLE_X       = 0.f;
    float BOOTSTRAP_TABLE_Y       = 0.f;
    float BOOTSTRAP_TABLE_YAW     = 0.f;
    float BOOTSTRAP_TABLE_WIDTH   = 1.5f;
    float BOOTSTRAP_TABLE_DEPTH   = 1.4f;
    float BOOTSTRAP_TABLE_HEIGHT  = 0.74f;

    // Media plane (zero-copy DDS) — RGB (camera window) + LiDAR (LidarIngestor).
    // DDS domain + topics are NOT configured: they are read from the media descriptor
    // JSON the producer authors on the "zed"/"helios" nodes, so the consumer always
    // uses the producer's dedicated domain. Subscribers are created lazily once those
    // nodes + descriptors exist.
    bool        LIDAR_USE_MEDIA   = true;   // false ⇒ DSR graph laser_* only

    // Waiting/Operating gate on the LiDAR media plane. Without LiDAR the localizer can never
    // stabilize, so the agent stays in (or falls back to) Waiting instead of sitting in Operating
    // doing nothing. 0 on either knob disables that half of the gate.
    int LIDAR_STALL_TIMEOUT_MS = 3000;   // Operating: no sweep for this long ⇒ back to Waiting
    int LIDAR_WAIT_LOG_PERIOD_MS = 2000; // Waiting: how often to reprint why we are still waiting

    // Camera-overlay object projection: DSR node TYPES whose oriented boxes are projected on
    // the live RGB image (alongside the always-drawn walls). Concept agents now publish all
    // furniture as generic "object" nodes (class in object_subtype), so "object" alone covers
    // tables/chairs/bottles/…. Config Overlay.ObjectTypes overrides this comma-separated list.
    std::vector<std::string> OVERLAY_OBJECT_TYPES = {"object"};

    // ── Object anchors (validated modelled objects as SE(2) pose landmarks for localization) ──
    // OFF by default. Precision-weighted by each object's own belief covariance (no threshold).
    bool  OBJECT_ANCHOR_ENABLE      = false;  // ObjectAnchor.enable
    bool  OBJECT_ANCHOR_OPTIMIZE_LANDMARK = false;  // ObjectAnchor.optimizeLandmark — p_o as a private
                                              // VARIABLE (birth prior = producer's belief) instead of a pin
    // ObjectAnchor.subtypes — which object CLASSES become landmarks. Default {"table"} keeps the
    // historical behaviour; the classes are deliberately selectable one at a time (see the header of
    // ObjectAnchorSource::Config for why table and refrigerator are not interchangeable).
    std::vector<std::string> OBJECT_ANCHOR_SUBTYPES = {"table"};
    float OBJECT_ANCHOR_WEIGHT      = 1.0f;   // ObjectAnchor.weight  (keep < walls)
    float OBJECT_ANCHOR_HUBER       = 3.0f;   // ObjectAnchor.huber   (whitened σ units)
    int   OBJECT_ANCHOR_MAX_SLOTS   = 3;      // ObjectAnchor.maxSlots
    float OBJECT_ANCHOR_MEAS_SIG_XY = 0.05f;  // ObjectAnchor.measSigmaXY  (m)  fallback R_o
    float OBJECT_ANCHOR_MEAS_SIG_YAW= 0.15f;  // ObjectAnchor.measSigmaYaw (rad) fallback R_o
    float OBJECT_ANCHOR_EARLY_EXIT_SIGMA = 2.0f;  // ObjectAnchor.earlyExitSigma — whitened anchor-residual
                                                  // σ-cutoff that forces the optimizer to run (drift catch)
    float OBJECT_ANCHOR_VALIDATE_SIGMA  = 0.10f;  // ObjectAnchor.validateSigma — map-pose σ (m) below which
                                                  // room PINS the table's world pose (breaks the circularity)
    bool  OBJECT_ANCHOR_FRESHNESS_ENABLE    = true;  // ObjectAnchor.freshnessEnable — grow R_o with obs age
    float OBJECT_ANCHOR_FRESHNESS_AGE_SCALE = 3.0f;  // ObjectAnchor.freshnessAgeScale — frames→σ doubles

    // ── RGB edge alignment (structural contours vs image gradient) ─────────────────────────────
    // OFF by default, at THREE independent levels, mirroring how GnShadow/OptimizerType were staged:
    //   ENABLE  — nothing constructed: no subscriber, no ingest thread, no factor. ZERO cost.
    //   SHADOW  — extract + evaluate the term into a DISCARDED system and log it. Pose UNTOUCHED.
    //   DRIVE   — the factor enters build_factors() and moves the published pose.
    // There is deliberately NO weight key: if the term needs a hand-set scalar to behave, its
    // covariance model is wrong, which is the whole point of the common-mode cap and the CRB-derived
    // per-sample precision. Do not add one.
    bool  IMAGE_EDGE_ENABLE = false;   // ImageEdge.enable
    bool  IMAGE_EDGE_SHADOW = false;   // ImageEdge.shadow
    bool  IMAGE_EDGE_DRIVE  = false;   // ImageEdge.drive   (refused unless OptimizerType == "GN")
    std::string IMAGE_EDGE_CAMERA = "zed";   // ImageEdge.camera — DSR node name ("zed" | "ricoh")
    /// Convert at most one camera frame per this many ms, per ingestor. 0 = every frame.
    /// The DRAIN is never throttled (a reliable reader whose pool backs up stops the producer);
    /// only the grey conversion of frames nobody would have read. See CameraIngestor.
    int   IMAGE_EDGE_MIN_CONVERT_MS = 0;     // ImageEdge.minConvertIntervalMs
    /// The same throttle for the CALIBRATION channels, which is a different question: their frames
    /// never reach the loss, so a skipped one costs coverage of a fit that accumulates over minutes,
    /// not evidence for this tick's pose. Kept SEPARATE from the driving camera on purpose — one
    /// number for both would buy the calibration's saving with the pose factor's rate.
    int   IMAGE_EDGE_CALIB_MIN_CONVERT_MS = 100;   // ImageEdge.calibMinConvertIntervalMs
    // Which structural contours are measured. Vertical wall-wall corners carry BEARING on a
    // horizontal normal and are nearly immune to the mount pitch/height nuisances; the floor-wall
    // junction carries RANGE and is exposed to them (δd = θ_pitch·d²/h ⇒ 1° ≈ 14 cm at 3 m).
    // That asymmetry is why they are separately switchable and why corners come first.
    bool  IMAGE_EDGE_USE_WALL_CORNERS  = true;   // ImageEdge.useWallCorners
    bool  IMAGE_EDGE_USE_FLOOR_JUNCTION = false; // ImageEdge.useFloorJunction
    bool  IMAGE_EDGE_USE_WALL_CEILING  = false;  // ImageEdge.useWallCeiling
    float IMAGE_EDGE_SAMPLE_SPACING_M  = 0.10f;  // ImageEdge.sampleSpacingM — arc-length in 3-D, NOT
                                                 // in pixels (uniform-in-pixels biases density to the
                                                 // near end, silently reweighting the estimator)
    float IMAGE_EDGE_SEARCH_SIGMAS     = 3.0f;   // ImageEdge.searchSigmas — Gaussian truncation point of
                                                 // a DERIVED window, not a pixel constant. Testable as
                                                 // such: sweep 2/3/4, the residual distribution and
                                                 // chi2_per_dof must not move. If they do, the mixture
                                                 // is mis-specified.
    int   IMAGE_EDGE_MAX_SEARCH_PX     = 64;     // ImageEdge.maxSearchPx — a COMPUTE bound only; the CSV
                                                 // logs n_clamped so it is visible if it ever binds
    int   IMAGE_EDGE_MAX_SLOTS         = 1;      // ImageEdge.maxSlots — newest slot only, like corners
    // Common-mode (shared-nuisance) priors. These are PHYSICAL: how well the mount is known. They are
    // what stops N correlated samples on one wall from claiming √N precision and out-voting the LiDAR.
    float IMAGE_EDGE_MOUNT_PITCH_SIGMA = 0.0035f; // ImageEdge.mountPitchSigma (rad, ~0.2°)
    float IMAGE_EDGE_MOUNT_HEIGHT_SIGMA= 0.010f;  // ImageEdge.mountHeightSigma (m)
    float IMAGE_EDGE_MOUNT_YAW_SIGMA   = 0.0035f; // ImageEdge.mountYawSigma (rad)
    // Local boresight correction applied to the graph's camera<-robot extrinsic. NOT a tuning knob:
    // a measured physical angle. 0 = use the graph's extrinsic unchanged.
    float IMAGE_EDGE_MOUNT_YAW_CORR    = 0.0f;    // ImageEdge.mountYawCorrection (rad)
    // Per-CONTOUR map-position uncertainty — nuisance column [4]. Not a mount property: how well any
    // single wall's place in the room polygon is known. MEASURED, see image_edge_types.h.
    float IMAGE_EDGE_WALL_POS_SIGMA    = 0.015f;  // ImageEdge.wallPositionSigma (m)
    // Evidence form: per-sample contour residuals (false) or triple points (true). INSTEAD, not as
    // well — a triple point is the same segment offsets re-expressed, so both would double-count.
    bool  IMAGE_EDGE_USE_TRIPLE_POINTS = false;   // ImageEdge.useTriplePoints
    // Cameras run for CALIBRATION, which need not be the one that drives the pose. Each gets its own
    // ingestor, its own extraction and its own evidence file; IMAGE_EDGE_CAMERA stays the one whose
    // factor may enter the loss. Two of them make the sensor triangle close — see camera_calibration.h.
    std::vector<std::string> CALIB_CAMERAS = {};   // ImageEdge.calibCameras
    std::string IMAGE_EDGE_CSV = "etc/image_edge.csv";  // ImageEdge.csv
};

// Load the agent params + RoomConcept params + EpistemicController/planner params,
// and seed the planner's robot footprint. Call once from initialize().
void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      RoomConcept& room_concept, EpistemicController& epistemic);

}  // namespace rc
