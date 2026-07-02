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
    std::string priors_path = "etc/object_priors.toml";

    float fe_eps           = 1e-3f;
    int   K_stable         = 30;
    float write_threshold  = 1e-3f;
    int   log_period_frames = 30;
    int   voxel_bank_max_points     = 4000;
    float voxel_bank_quantization_m = 0.01f;
    float voxel_select_radius_margin_m = 0.10f;
    float voxel_select_height_margin_m = 0.10f;

    // BottleModel prior geometry (forwarded to BottleModelParams — the model is the SDF + state carrier).
    float prior_radius      = 0.035f;
    float prior_height      = 0.20f;
    float prior_size_std    = 0.03f;
    float mask_precision    = 0.0f;   // occluding-contour silhouette weight (per-ray precision; 0 = off)
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
    float ai2_clutter_frac         = 0.10f;  // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m      = 0.08f;  // a point further than ~this from the surface is likely clutter
    float ai2_prior_pos_std        = 0.30f;  // broad position prior std (m) on cx,cy,cz
    float ai2_prior_size_std       = 0.03f;  // broad size prior std (m) on radius,height
    float ai2_process_std_m        = 0.005f; // predict process-noise std, all 5 length DOFs (m/frame)
    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out). The frame's
    // information saturates here, so N correlated points can't collapse σ → calibrated posterior.
    float ai2_common_mode_pos_std  = 0.02f;  // shared position error (m); pose-chain cov adds to it
    float ai2_common_mode_size_std = 0.01f;  // shared size error radius,height (m)
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
    // OFF → legacy prior-scaffold + greedy nearest-mask. ON → data-driven: masks are associated to
    // instances by a covariance-gated global 1-to-1, a persistently-unexplained mask spawns a new
    // bottle, and an unsupported instance is retired. Pure-tracker mode (priors give only the default
    // birth size); enable when the scene has an unknown / changing number of bottles.
    bool  tracker_enabled          = false;
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match (when cov known)
    float tracker_gate_fallback_m  = 0.30f;   // metric XY gate (m) when an instance has no usable cov yet
    int   tracker_birth_frames     = 6;       // frames a mask must stay unexplained before spawning a bottle
    int   tracker_death_frames     = 90;      // frames an instance may go unsupported before retirement
    float tracker_birth_min_sep_m  = 0.20f;   // a birth must be ≥ this from every existing bottle (anti-dup)
    float tracker_detection_noise_m = 0.05f;  // R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)
    float tracker_merge_overlap    = 0.30f;   // merge two instances whose footprints (circles) overlap ≥ this
                                              // fraction of the SMALLER one, keeping the more-observed. 0 disables.
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
    // ── Part B: masks in camera frame + chain covariance ───────────────────────────────────────────
    // The voxelizer dual-publishes support points in CAMERA frame (mask_support_points_cam). When
    // enabled, MaskIngestor transforms them to the fit frame via inner_eigen pinned to the capture
    // stamp, and the published RT-edge covariance gains the localization term J·Σ_chain·Jᵀ from
    // InnerGaussianAPI (the chain source→target uncertainty). false → legacy room-frame masks.
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
