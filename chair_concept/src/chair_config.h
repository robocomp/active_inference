/*
 * chair_config.h
 *
 * Plain-data configuration for the chair_concept agent, plus a loader that fills
 * it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a new
 * concept agent can copy this file and edit only the keys it needs (mirrors
 * bottle_concept/bottle_config.h).
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct ChairConfig
{
    // Agent convergence
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   detection_alive_max_frames = 40; // cycles without a fresh chair mask before detection_alive=false
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float min_standoff_m    = 1.8f;    // min stand-off floor for epistemic viewpoints (YOLO misses too-close chairs)
    int   epistemic_cooldown_cycles = 200;    // min cycles withdrawn after satisfaction
    int   chair_log_period_frames = 30;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;

    // ChairModel geometry / mask split: on-surface membership for the candidate/residual split in
    // ChairFitter::observe (a mask point within sdf_threshold_for_storage of the compound SDF is a candidate).
    float sigma_obs         = 0.05f;
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (mirrors table_concept [TableModel].AI2*): recursive-Laplace full-covariance filter ──
    float ai2_sigma_base_m       = 0.03f;
    float ai2_clutter_frac       = 0.10f;
    float ai2_clutter_scale_m    = 0.12f;
    float ai2_clutter_structure_gain = 1.0f;  // density-aware clutter: shrink the clutter prior for seat-coplanar points (closes the escape valve → fixes seat_d collapse + its overconfidence). 0 → flat clutter
    float ai2_prior_size_std     = 0.15f;
    float ai2_process_std_m      = 0.005f;
    float ai2_process_std_yaw    = 0.01f;
    float ai2_process_std_size   = 0.0005f;  // rigid size DOFs ≪ pose (Tier-1: kills the vertical random walk)
    float ai2_floor_z            = 0.0f;     // room-frame floor; cz pinned here (Tier-1: removes cz gauge freedom)
    float ai2_floor_std          = 0.03f;    // floor-height uncertainty (m) → common-mode z
    float ai2_seat_anchor_std    = 0.04f;    // seat-layer height anchor obs noise (m); 0 → off. Fixes seat_h gauge runaway
    float ai2_seat_anchor_band   = 0.12f;    // seat-layer mean-shift bandwidth (m) = seat vertical scale
    float ai2_seat_extent_std    = 0.02f;    // seat-layer footprint (seat_w/seat_d) span anchor obs noise (m); 0 → off. Fixes seat_d collapse
    float ai2_common_mode_pos_std  = 0.03f;
    float ai2_common_mode_size_std = 0.02f;
    float ai2_common_mode_yaw_std  = 0.03f;
    float ai2_range_noise_lat_per_m = 0.02f;   // static range → R + position common-mode (m per m)
    float ai2_range_noise_yaw_per_m = 0.03f;   // static range → yaw common-mode (rad per m)
    float ai2_trunc_gate_frac    = 0.10f;
    int   ai2_gn_iters           = 4;
    float ai2_extent_std         = 0.05f;   // extent-observation noise (m) for the coverage/extent likelihood
    std::string ai2_csv_path     = "";

    // Upload the chair pose covariance onto the room→chair RT edge (rt_covariance_att, 6×6 SE3), built
    // from the belief's full Σ over [cx,cy,cz,yaw,...]: x←cx, y←cy, z←cz, yaw←ψ; roll/pitch are
    // unobservable (large). rt_cov_scale calibrates the raw variance toward NEES≈1.
    bool  rt_cov_upload = true;
    float rt_cov_scale  = 1.0f;
    bool  rt_cov_add_chain = true;   // Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov

    // ── Multi-instance birth/associate/death tracker (shared rc::InstanceTracker) ──────────────────
    // "chair" masks are associated to instances by a covariance-gated 1-to-1 (cov from the belief Σ),
    // a persistently-unexplained mask spawns a new chair, and (if death_enabled) an unobserved instance
    // is retired. Chairs are persistent furniture: death OFF by default; birth_min_sep wide. Mirrors table.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.40f;   // metric XY gate (m) before an instance has a usable covariance
    float tracker_detection_noise_m = 0.20f;  // R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)
    int   tracker_birth_frames     = 8;       // frames a mask must stay unexplained before spawning a chair
    int   tracker_death_frames     = 300;     // frames an instance may go unobserved before retirement
    bool  tracker_death_enabled    = false;   // OFF: a chair is persistent furniture — removed only by MERGE
    float tracker_birth_min_sep_m  = 0.70f;   // a birth must be ≥ this (m) from every existing chair (anti-dup)
    float tracker_merge_overlap    = 0.20f;   // merge two instances whose seat footprints overlap ≥ this
                                              // fraction of the smaller, keeping the more-observed. 0 disables.
    bool  tracker_prune_enabled        = true; // stillbirth prune of phantom duplicates born from churn
    int   tracker_prune_maturity_cycles = 90;  // probation window; older instances are permanent furniture
    int   tracker_prune_patience       = 30;   // consecutive tracker-unassigned cycles in probation → prune
    float tracker_birth_seat_w     = 0.45f;   // seed seat width/depth/heights for a freshly born chair node
    float tracker_birth_seat_d     = 0.45f;
    float tracker_birth_seat_h     = 0.45f;
    float tracker_birth_back_h     = 0.45f;
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
};

// Fill a ChairConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
ChairConfig load_chair_config(const ConfigLoader& cfg);

}  // namespace rc
