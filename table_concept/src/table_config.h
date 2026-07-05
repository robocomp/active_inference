/*
 * table_config.h
 *
 * Plain-data configuration for the table_concept agent, plus a loader that fills
 * it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a new
 * concept agent can copy this file and edit only the keys it needs (mirrors
 * bottle_concept/bottle_config.h).
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct TableConfig
{
    // Agent convergence
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   detection_alive_max_frames = 40; // cycles without a fresh table mask before detection_alive=false
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    int   epistemic_cooldown_cycles = 200;    // min cycles withdrawn after satisfaction
    int   table_log_period_frames = 30;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;

    // Top/leg SDF split band (forwarded to TableModelParams.sigma_obs): a mask point within
    // TOP_THICKNESS + sigma_obs below the top face is attributed to the slab (candidate) vs a leg.
    float sigma_obs         = 0.05f;
    // On-surface membership for the candidate/residual split in TableFitter::observe.
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (TABLE_FIT_AI2.md) — full-covariance recursive filter ──────────────────────────
    float ai2_sigma_base_m     = 0.03f;  // base on-surface obs noise std (m); R = σ² (+ motion_var + …)
    float ai2_clutter_frac     = 0.10f;  // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m  = 0.12f;  // a point further than ~this from every surface is likely clutter
    float ai2_prior_size_std   = 0.30f;  // broad size prior std (m) on w,h,H
    float ai2_process_std_m    = 0.005f; // predict process-noise std, length DOFs (m/frame)
    float ai2_process_std_yaw  = 0.01f;  // predict process-noise std, yaw (rad/frame)
    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out). The frame's
    // information saturates here, so N≈10⁴ correlated points can't collapse σ → calibrated posterior.
    float ai2_common_mode_pos_std  = 0.03f; // shared position error (m); pose-chain cov adds to it
    float ai2_common_mode_size_std = 0.02f; // shared size error w,h,H (m)
    float ai2_common_mode_yaw_std  = 0.03f; // shared yaw error (rad)
    // STATIC range weighting (motion-free): the common-mode error grows with view distance, so a far, vague
    // mask cannot resolve pose — orientation least of all (a far view confirms existence, can't rotate the
    // table). Continuous, no gate. lat feeds R + position common-mode (m per m of range); yaw feeds the yaw
    // common-mode (rad per m), the binding term. Set 0 to disable. Range Z comes from the voxelizer (mask_range).
    float ai2_range_noise_lat_per_m = 0.02f; // lateral deprojection std growth (m per m of range)
    float ai2_range_noise_yaw_per_m = 0.03f; // yaw common-mode std growth (rad per m of range)
    float ai2_trunc_gate_frac  = 0.10f;  // skip the geometric update (predict only) when this fraction of
                                         // the mask silhouette is on the image border (truncated → biased)
    int   ai2_gn_iters         = 4;      // Gauss-Newton iterations per frame
    std::string ai2_csv_path   = "";     // if non-empty, append per-cycle AI2 belief (state + Σ diag + mask R) to this CSV

    // ── YOLO-INDEPENDENT LiDAR first-hit range factor (see common/ai_belief/lidar_ray_factor.h) ──────
    // Second, sensor-independent evidence channel: lidar3D returns landing on the table (legs + rim) pin the
    // extent and centre in METRIC range, ALONG the viewing ray — an error mechanism uncorrelated with the
    // YOLO segmentation, so it attacks the mask-erosion under-size the mask cannot self-correct. Consumes
    // the lidar3D media plane via TableLidarIngestor; dormant (no DDS participant) while precision == 0.
    float lidar_precision      = 0.0f;   // per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF
    float lidar_robust_c_m     = 0.05f;  // Cauchy scale (m): returns this far off the surface fade out
    float lidar_select_margin_m = 0.10f; // pre-select returns within (birth half-extent + margin), all z up to top
    std::string lidar_frame_node = "lidar3D"; // DSR node whose frame the raw sweep is in (room←this transform)
    float lidar_coverage_n0     = 60.0f; // LiDAR ray count for FULL weight; fewer → proportionally down-weighted
    // Angular-coverage weighting: precision ×= (1−R)^p, R = mean-resultant length of return bearings about the
    // centre. p=0 disables (flat), p=1 = pure circular variance. Down-weights ONE-SIDED sweeps (which over-
    // commit the near-square w↔h mode from a single view); the recursive belief still accumulates over an orbit.
    float lidar_coverage_ang_power = 1.0f;

    // Divergence safety net (mirrors bottle_concept). A static table cannot physically move this far in one
    // frame, so a GN step whose centre jump exceeds it is an OUTLIER frame (corrupted mask cloud / one-sided
    // LiDAR runaway) → reject the update, restore state+Σ, widen Σ. 0 disables. Prevents the cx=−200m runaway.
    float max_step_m            = 1.0f;

    // Upload the table pose covariance onto the room→table RT edge (rt_covariance_att, 6×6 SE3),
    // mapped from the belief's full Σ over [cx,cy,H,w,h,yaw]: x←cx, y←cy, z←H/2, yaw←ψ; roll/pitch are
    // unobservable (large). rt_cov_scale calibrates the raw variance toward NEES≈1.
    float rt_cov_scale  = 1.0f;
    bool  rt_cov_add_chain = true;   // Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov

    // ── Multi-instance birth/associate/death (shared rc::InstanceTracker) ─────────────────────────
    // "table" masks are associated to instances by a covariance-gated global 1-to-1; an unexplained mask
    // spawns a new table; an instance unobserved for tracker_death_frames is retired (death off by default).
    // Tables are large static furniture: keep birth_min_sep large (no two table centres that close) and
    // death_frames LARGE so a long occlusion doesn't drop a table.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.50f;   // metric XY gate (m) before an instance has a usable covariance
    // Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I). For a table
    // the YOLO mask CENTROID sits well off the fitted geometric CENTRE under a partial view (~0.2–0.5 m),
    // so R must cover that offset or the overconfident fit (σ ~mm) rejects its own real detection.
    float tracker_detection_noise_m = 0.35f;
    int   tracker_birth_frames     = 8;       // frames a mask must stay unexplained before spawning a table
    int   tracker_death_frames     = 300;     // frames an instance may go unobserved before retirement (large)
    bool  tracker_death_enabled    = false;   // OFF: a table is rigid, persistent furniture — never retired by
                                              // a miss timer. Removal is by the MERGE operator only.
    float tracker_birth_min_sep_m  = 0.60f;   // a birth must be ≥ this (m) from every existing table
    // Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of
    // the SMALLER footprint (two tables cannot share space). 0 disables the merge.
    float tracker_merge_overlap    = 0.05f;
    // Default geometry for a table BORN from a mask (no prior to seed it); the belief refines from here.
    float tracker_birth_width_m    = 1.0f;
    float tracker_birth_depth_m    = 0.6f;
    float tracker_birth_height_m   = 0.75f;
    // Prior PRECISION (size std, m) for a born table: the Birth* dims seed the belief size prior at THIS σ.
    float tracker_birth_size_std   = 0.15f;
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
};

// Fill a TableConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
TableConfig load_table_config(const ConfigLoader& cfg);

}  // namespace rc
