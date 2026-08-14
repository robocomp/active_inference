/*
 * bottle_config.h
 *
 * Plain-data configuration for the bottle_concept agent, plus a loader that
 * fills it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a
 * new concept agent can copy this file and edit only the keys it needs.
 */

#pragma once

#include <string>

#include "bottle_model.h"   // RobustLossType

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct BottleConfig
{
    float fe_eps           = 1e-3f;
    int   K_stable         = 30;
    int   diverged_retire_frames = 20;   // retire an instance after this many consecutive unexplained fits; 0 = off
    // Divergence sentinel (replaces the old surface-energy==0 all-clutter test): a frame is UNEXPLAINED when
    // its mean clutter responsibility exceeds this — the honest "the cylinder explains none of its data"
    // signal on the clutter-inclusive free energy. A healthy fit sits well below (most points on-surface).
    float clutter_diverge_frac = 0.90f;
    // FE-surprise attention baseline (TABLE.md §9): asymmetric EMA (down fast = consolidate a better fit; up
    // slow = a sustained rise, the bottle moved, stays surprising) + a smoothed positive gap = the surprise.
    float fe_baseline_adapt_down = 0.05f;
    float fe_baseline_adapt_up   = 0.005f;
    float fe_surprise_smooth     = 0.10f;
    float max_step_m       = 0.5f;   // reject a frame whose net centre move exceeds this (m); a bottle can't teleport — a corrupted cloud can. 0 = off
    float write_threshold  = 1e-3f;
    int   log_period_frames = 30;
    int   voxel_bank_max_points     = 4000;
    float voxel_bank_quantization_m = 0.01f;
    float voxel_select_radius_margin_m = 0.10f;
    float voxel_select_height_margin_m = 0.10f;

    // ── Primary-input stream gate (readiness + staleness) — LIFECYCLE, not a belief knob ──────────────
    // Demote Operating→Degraded→Waiting when the voxelizer's `masks` node stops advancing its mask_frame_id
    // for this long (producer dead/stalled) — don't integrate stale evidence; re-admit when it returns.
    // Orthogonal to ai2_age_nominal_dt_s (belief-axis Σ-aging). MUST exceed the voxelizer HOLD_ENTER_S so a
    // legitimately empty scene (still-advancing counter) never trips it. 0 = disable the gate.
    int   masks_stall_timeout_ms       = 3000;

    // BottleModel prior geometry (forwarded to BottleModelParams — the model is the SDF + state carrier).
    float prior_radius      = 0.035f;
    float prior_height      = 0.20f;
    float prior_size_std    = 0.03f;
    float mask_precision    = 0.0f;   // occluding-contour silhouette weight (per-ray precision; 0 = off)
    // ── YOLO-INDEPENDENT LiDAR first-hit range factor (see common/ai_belief/lidar_ray_factor.h) ──────
    // A second evidence channel: LiDAR returns falling on the bottle pin cx,cy IN METRIC DEPTH + radius
    // along the viewing ray — the direction the monocular mask SDF is blind to (see bottle-sdf-depth-bias).
    float lidar_precision      = 0.0f;   // per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF
    float lidar_robust_c_m     = 0.05f;  // Cauchy scale (m): returns this far off the surface fade out
    float lidar_select_margin_m = 0.06f; // pre-select returns within radius+margin (horiz) and h/2+margin (vert)
    std::string lidar_frame_node = "lidar3D"; // DSR node whose frame the raw sweep is in (room←this transform)
    // ── Range-scaled precision: perceive with precision when CLOSE (manipulation), leave UNTOUCHED when far ──
    // Observation precision fades continuously with the camera→object sensing distance (NOT a range gate): R
    // grows as (range/near)^power beyond `near`, so a distant/receding object's frames carry ~no information
    // and the belief holds its last good estimate; inside `near` (grasp range) it gets full precision and
    // sharpens. Mirrors the table-concept range-covariance philosophy. Also down-weight sparse LiDAR (few noisy
    // returns shouldn't swing radius). 0 near = disable range weighting.
    float range_near_m          = 0.6f;   // full precision within this sensing distance (m); ~manipulation reach
    float range_precision_power = 2.0f;   // how fast precision fades beyond `near`: R *= (range/near)^power
    float lidar_coverage_n0     = 25.0f;  // LiDAR ray count for FULL weight; fewer → proportionally down-weighted
    // ── YOLO detection-score → silhouette reliability weight ────────────────────────────────────────
    // The detector's per-mask confidence is a per-observation RELIABILITY (a noisy mask is noisy evidence):
    // w = clamp01((conf−floor)/(ref−floor))^power scales the silhouette precision so a weak mask can't
    // over-tighten radius. floor keeps a low-but-real detection contributing rather than zeroing it.
    bool  mask_conf_weight  = true;   // false → score ignored; w≡1
    float mask_conf_floor   = 0.2f;   // conf ≤ floor → minimal weight
    float mask_conf_ref     = 0.5f;   // conf ≥ ref   → full weight (w=1)
    float mask_conf_power    = 2.0f;  // shaping exponent on the normalised score
    // ── AI2 belief (the fit; mirrors table_concept [TableModel].AI2*/chair [ChairModel].AI2*) ────────
    // Full-covariance recursive-Laplace belief over [cx,cy,cz,radius,height] on the shared engine
    // (bottle_belief.h) — the ONLY fit path. Static (no yaw, single cylinder primitive); a moved bottle is
    // re-acquired via the tracker gate widening on stale predicts.
    float ai2_sigma_base_m         = 0.02f;  // base on-surface obs noise std (m); R = σ²
    // ── DETECTOR ENVELOPE (common/detectability) — the YOLO inverse model ─────────────────────────
    // ONE model, two consumers: the planner puts the stand-off at its argmax and the removal channel
    // weights absence by it. Defaults are the fleet PRIOR, not a measurement, so behaviour is unchanged
    // until etc/config.toml sets them. Genuinely object-dependent (measured: fridge max_fill 1.32 vs
    // table 0.677), and a bottle is far smaller than either — fit it from this agent's own ai2_log
    // with common/detectability/tools/fit_envelope (--label fresh).
    float detect_min_fill  = 0.10f;
    float detect_max_fill  = 0.60f;
    float detect_soft      = 0.06f;
    // ── EXISTENCE (bottle_existence.h) — the shared rc::exist policy ──────────────────────────────
    // Bottle was the last object agent with no existence channel: it retired instances on a miss counter
    // (Tracker.DeathFrames), which invariant 5 forbids. With this ON, removal is a decision on L and the
    // miss counter is disabled; OFF restores the counter exactly, so the two can be A/B'd.
    // ★No confirm/absence GAIN knobs: the two log-likelihood ratios are log(pd/pc) and log((1−pd)/(1−pc)),
    // derived from the same two sensor rates, so the channel cannot drift into a one-way ratchet.
    bool  existence_enabled           = true;
    float existence_birth_logodds     = 0.0f;   // a newborn bottle is a 50/50 hypothesis, not a fact
    float existence_logodds_max       = 4.0f;   // |L| clamp — also the recantation budget (2·L_max nats)
    float existence_removal_prob      = 0.12f;  // remove below P(exists) = this
    float existence_frame_correlation = 0.0f;   // ρ: opt-in, MEASURE it from this agent's log before setting
    float existence_detection_prob    = 0.85f;  // P(detect | exists & observable)
    float existence_clutter_prob      = 0.05f;  // P(detect | ¬exists) — the false-positive rate
    float existence_sensor_sigma_m    = 0.03f;  // LiDAR carve surface blur σ (m)
    int   existence_remove_frames     = 15;     // debounce in IDEAL OBSERVATIONS (Σ p_detect), not cycles
    float ai2_clutter_frac         = 0.10f;  // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m      = 0.08f;  // a point further than ~this from the surface is likely clutter
    float ai2_prior_pos_std        = 0.30f;  // broad position prior std (m) on cx,cy,cz
    float ai2_prior_size_std       = 0.03f;  // broad size prior std (m) on radius,height
    float ai2_process_std_m        = 0.005f; // predict process-noise std, POSITION (cx,cy,cz) (m/frame)
    // ★MOTION REQUIRES A CAUSE. A bottle does not move by itself, so its position process noise is a
    // MIXTURE over "is being carried", weighted by P(a mover is in contact). With no mover in the room the
    // position is as static as the size, and an apparent jump is read as a MIS-ASSOCIATION rather than
    // motion. Off (false) reproduces the previous always-volatile behaviour exactly, so the two are A/B-able.
    bool  motion_requires_cause    = true;
    float mover_reach_m            = 0.75f;  // human arm's reach — a PHYSICAL length, not a tuned radius
    float ai2_process_std_moved_m  = 0.05f;  // position std/frame while a mover IS in contact (~0.5 m/s @10Hz)
    float ai2_process_std_size_m   = 0.001f; // predict process-noise std, SIZE (radius,height) — tiny: rigid size sticks
    // Stale-belief aging (measurement-age → covariance). Nominal mask-stream period (s): with >0, an unseen
    // bottle's POSITION Σ inflates by Q·(dt/this) on the agent's clock (predict_stale) so a dead feed reads as
    // growing position uncertainty downstream. <=0 keeps the historic one-Q-per-unseen-cycle behaviour.
    float ai2_age_nominal_dt_s     = 0.0f;
    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out). The frame's
    // information saturates here, so N correlated points can't collapse σ → calibrated posterior.
    float ai2_common_mode_pos_std  = 0.02f;  // shared position error (m); pose-chain cov adds to it
    float ai2_common_mode_size_std = 0.01f;  // shared size error radius,height (m)
    // Ego-motion → COMMON-MODE fixation ("be still to UPDATE, else CONFIRM"; mirrors table_concept MotionCm*Gain).
    // A moving frame's mask is smeared/displaced by ego-motion by ≈ effective-lag·speed — a per-mask SHARED error
    // that per-point R averages away. Route it into the frame's common-mode (chain_cov) so the Woodbury cap freezes
    // how far a moving frame can move/reshape the GEOMETRY (existence/association don't read this). CONTINUOUS, no
    // gate: std growth per unit motion_dotd (m/s); at dotd→0 it vanishes (a still frame updates fully); 0 disables.
    float motion_cm_pos_gain       = 0.10f;  // position (cx,cy) shared-error std per m/s of motion_dotd
    float motion_cm_size_gain      = 0.20f;  // size (radius,height) shared-error std per m/s — the anti-RESHAPE lever
    // Ego-motion DISCRETE "confirm-only" gate (mirrors refrigerator/chair confirm_only). The continuous common-
    // mode above is the graceful backstop; this gate takes a PREDICT-ONLY branch (Σ inflates, geometry mean HELD)
    // when the robot is clearly MOVING, so a motion-smeared mask may only CONFIRM a converged bottle, never
    // move/reshape it. motion_magnitude(inst) = max(|motion_dotd|, ego_lin_mps + ai2_ang_lever_m·ego_ang_radps).
    bool  ai2_motion_confirm_only = true;    // master switch for the discrete gate (false = continuous common-mode only)
    float ai2_still_lin_mps       = 0.05f;   // camera linear speed (m/s) below which the robot counts as "still"
    float ai2_still_ang_radps     = 0.10f;   // camera angular speed (rad/s) below which the robot counts as "still"
    float ai2_still_dotd          = 0.05f;   // per-mask ego-motion corruption speed (m/s) still-level
    float ai2_ang_lever_m         = 2.0f;    // rad/s → m/s lever (tangential speed of a bottle ~this far away)
    int   ai2_gn_iters             = 4;      // Gauss-Newton iterations per frame
    std::string ai2_csv_path       = "";     // non-empty → append a per-cycle AI2 belief CSV (state + Σ diag)
    // ── Support-surface decision (room vs table parent) ───────────────────────────────────────────
    // The bottle hangs from the surface it RESTS ON, chosen by MAP over {room, every table_N}: the
    // centre must lie inside the table's oriented footprint AND the observed base must sit at its top
    // (vertical support). Robust to the perceived table-top z-bias (σ_z ≥ that bias) and to flicker
    // (commit a re-parent only after support_commit_cycles favouring the challenger). Replaces the
    // naive XY-only single-"table" gate.
    float support_sigma_z          = 0.04f;   // base vertical-support std (m); ≥ the table-top z-bias (~2.5 cm)
    float support_footprint_margin = 0.05f;   // m: slack added to the table half-extents for the footprint gate
    float support_lambda_xy        = 50.0f;   // penalty weight (1/m²) for the centre lying OUTSIDE the footprint
    float support_decision_margin  = 2.0f;    // log-evidence a table must beat the room/floor by to win
    int   support_commit_cycles    = 8;       // consecutive cycles a challenger must win before re-parenting
    // ── Multi-instance birth/associate/death tracker (shared rc::InstanceTracker) ──────────────────
    // The only instance-lifecycle path (mirrors table/chair): masks are associated to instances by a
    // covariance-gated global 1-to-1, a persistently-unexplained mask spawns a new bottle, and an
    // unsupported instance is retired.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match (when cov known)
    float tracker_gate_fallback_m  = 0.30f;   // metric XY gate (m) when an instance has no usable cov yet
    int   tracker_birth_frames     = 6;       // frames a mask must stay unexplained before spawning a bottle
    int   tracker_death_frames     = 90;      // frames an instance may go unsupported before retirement
    float tracker_birth_min_sep_m  = 0.20f;   // a birth must be ≥ this from every existing bottle (anti-dup)
    float tracker_detection_noise_m = 0.05f;  // R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)
    float tracker_merge_overlap    = 0.30f;   // merge two instances whose footprints (circles) overlap ≥ this
                                              // fraction of the SMALLER one, keeping the more-observed. 0 disables.
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
    // ── RGB-360 bearing confirmation (Part C confirm; RICOH_360_PERIPHERAL_DETECTION.md) ────────────
    // A ricoh no-depth bearing slice that lines up (in azimuth from the robot) with a live bottle is
    // evidence it is STILL THERE even when the zed missed it → HOLD its death-miss this cycle (peripheral
    // "glance"). No fit, no birth. Default OFF: the published bearing/azimuth convention is still
    // PROVISIONAL (see Part A step 8) — enable + watch the [bearing] log to verify it before relying on it.
    bool  bearing_confirm_enabled  = false;   // Bearing.ConfirmEnabled
    float bearing_confirm_gate_rad = 0.17f;   // Bearing.ConfirmGateRad — 1-D angular gate (~10°)
    // ── Part B: masks in camera frame + chain covariance ───────────────────────────────────────────
    // The voxelizer publishes support points in CAMERA frame (mask_support_points_cam); MaskIngestor
    // transforms them to the fit frame via inner_eigen pinned to the capture stamp. That is now the
    // DEFAULT for every agent (see the FRAME CONTRACT in common/mask_ingestor), so this flag no longer
    // SELECTS the frame — it only governs whether the published RT-edge covariance gains the
    // localization term J·Σ_chain·Jᵀ from InnerGaussianAPI (the chain source→target uncertainty).
    // To force the legacy room-frame array for an A/B, set MASK_INGESTOR_LEGACY_ROOM=1 in the environment.
    bool        masks_use_camera_frame = true;
    std::string masks_source_frame     = "zed";    // producer frame of mask_support_points_cam
    std::string masks_target_frame     = "room";   // bottle's fit frame
    bool        rt_cov_add_chain       = true;      // add J·Σ_chain·Jᵀ localization cov to the published RT cov
    // ── Epistemic "hidden-face" affordance ────────────────────────────────────────────────────────
    // The agent advertises a far-side viewpoint (opposite the camera) so the controller can observe the
    // bottle's occluded back arc and resolve the depth-degenerate radius. ΔH = ½·log(1 + view_info/Y_r).
    float epistemic_obs_distance   = 0.9f;    // stand-off (m) from the bottle at the far-side viewpoint
    float epistemic_view_info      = 50.0f;   // Fisher precision a back-view is expected to add to the radius DOF (ΔH scale)
    int   epistemic_cooldown_cycles = 200;    // post-completion hold: cycles the gain is suppressed so it isn't re-claimed
    std::string epistemic_csv_path  = "";     // non-empty → append a per-cycle epistemic/affordance CSV (debug/monitor)

    // Near-surface band (m): observe() splits a mask point into candidate (|SDF|<this) vs residual.
    float sdf_threshold_for_storage    = 0.03f;

    // Covariance write
    float yaw_variance = 9.87f;   // ≈π² — yaw is unobservable for a symmetric cylinder

    // Static ground-truth evaluation (Webots)
    // The bottle is stationary during perception, so its Webots pose is a constant expressed in the
    // room frame (DEF bottle → Shadow→room). When enabled, the tracker logs per-cycle position/size
    // error and NEES (covariance calibration). Consumed by BottleEvaluator.
    bool        eval_enabled = false;
    std::string eval_log_path = "etc/bottle_eval.csv";
    std::string eval_gt_source = "webots";   // "webots" (live getObjectPose) | "config"
    std::string eval_bottle_def = "bottle";  // Webots DEF of the bottle
    std::string eval_robot_def  = "shadow";  // Webots DEF of the Shadow robot (== DSR body frame)
    float gt_cx = 0.0f, gt_cy = 0.0f, gt_cz = 0.0f;   // cylinder CENTRE, room frame
    float gt_radius = 0.0f, gt_height = 0.0f;

    // One-shot bottle placement on start (Scene.*)
    // setObjectPose the real bottle to a fixed Webots-WORLD x,y ONCE at startup (z and orientation
    // kept), so the arm approaches from its own side and occludes the camera less.
    bool  place_on_start = false;   // Scene.PlaceBottleOnStart
    float place_world_x  = 0.0f;    // Scene.PlaceBottleWorldX (Webots world metres, +X front)
    float place_world_y  = 0.0f;    // Scene.PlaceBottleWorldY (Webots world metres, +Y right)

    // Moving-bottle validation experiment
    bool  move_experiment   = false;   // Eval.MoveExperiment
    int   move_settle_cycles = 25;     // cycles held at each grid pose before stepping
    float move_step_m        = 0.06f;  // grid spacing over the table (metres, world frame)
    int   move_grid_n        = 5;      // grid is move_grid_n × move_grid_n positions
    // Absolute-world grid (Eval.MoveAbsolute): sweep [xmin,xmax]×[ymin,ymax] in WORLD coords at
    // move_step_m spacing, instead of a home-centred N×N (reaches the +y/right side too).
    bool  move_absolute = false;       // Eval.MoveAbsolute
    float move_xmin = 0.0f, move_xmax = 0.0f, move_ymin = 0.0f, move_ymax = 0.0f;   // world bounds (m)

    // Static-restart validation
    // Validate the fit at independent static positions, restarting the agent per pose (fresh voxel
    // bank). One run = move the bottle to grid pose BOTTLE_TEST_POSE (env), fit from scratch, log.
    bool  static_pose_test  = false;   // Eval.StaticPoseTest
    int   static_pose_index = 0;       // grid index for THIS run (env BOTTLE_TEST_POSE)
};

// Fill an BottleConfig from a RoboComp ConfigLoader (all keys + defaults).
BottleConfig load_bottle_config(const ConfigLoader& cfg);

}  // namespace rc
