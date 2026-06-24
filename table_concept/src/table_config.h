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

#include "../../common/robust_metrics/robust_metrics.h"   // RobustLossType

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct TableConfig
{
    // Paths
    std::string priors_path = "etc/object_priors.toml";

    // Agent convergence
    float fe_eps            = 1e-3f;   // |ΔFE| threshold for convergence (legacy)
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   max_direct_fit_points = 400; // strided live points fed straight into the gradient each frame (0=off)
    int   detection_alive_max_frames = 40; // cycles without a fresh table mask before detection_alive=false
    int   M_diverge         = 20;
    float staleness_frames  = 90.0f;
    float explanation_ratio_thresh = 0.3f;
    float write_threshold   = 1e-3f;   // min ‖Δθ‖ before writing RT to DSR
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float delta_min         = 20.0f;   // min face coverage count
    float gain_threshold    = 0.1f;    // min ΔH for epistemic proposal
    int   table_log_period_frames = 30;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;

    // TableModel parameters (forwarded to TableModelParams)
    float sigma_obs         = 0.05f;
    float lambda_size       = 0.15f;
    float lambda_pos        = 0.05f;
    float lambda_state      = 0.02f;
    float lambda_angle      = 0.01f;
    float lambda_extent     = 2.0f;   // footprint-extent term: fit top rectangle to point span
    float prior_size_std    = 0.30f;
    int   optimization_iters = 10;
    float optimization_lr   = 0.05f;
    float grad_clip         = 2.0f;
    std::string optimizer_type = "adam";
    float sgd_momentum     = 0.9f;
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.10f;
    float robust_gnc_start_scale = 0.80f;   // GNC: wide initial robust scale annealed to robust_loss_scale
    float mask_precision = 0.30f;           // RGB-mask silhouette term weight (0 disables)
    int   sil_tangent_samples = 8;          // >0 = height-agnostic occluding-contour (samples/ray); 0 = top-plane only
    float sil_reopen_residual_m = 0.15f;    // a fresh mask with silhouette residual above this re-opens a converged fit
    int   robust_gnc_decay_cycles = 20;     // GNC start scale ramps to target over this many cycles, then off

    // SampleQueue parameters (forwarded to SampleQueueParams)
    int   num_angle_bins               = 24;
    int   num_z_bins                   = 10;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.08f;
    int   min_frames_before_historical = 10;
    int   historical_warmup_frames     = 50;
    int   max_new_points_per_frame     = 5;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float rfe_weight_gain              = 0.25f;
    float min_anchor_weight            = 0.12f;
    float edge_bonus_weight            = 0.3f;
    float edge_proximity_threshold     = 0.05f;

    // Observability-aware warm-start acceptance
    float warm_pts_min                 = 12.0f;
    float warm_pts_max                 = 30.0f;
    float warm_coverage_min_side       = 2.0f;
    float warm_rho_freeze              = 0.25f;
    float warm_size_pts_release        = 0.60f;   // rho_pts above which w/h trust points & ignore the side-coverage freeze
    int   warm_settle_cycles           = 40;      // cycles for acceptance gain to decay full→floor after a new-evidence burst
    int   warm_reopen_admit            = 8;       // net new queue anchors in a cycle that count as a fresh viewpoint (reset maturity)
    float warm_settle_floor            = 0.05f;   // acceptance-gain multiplier once mature (lower = stiffer lock; recovers on new evidence)
    float warm_info_half               = 20.0f;   // accumulated per-face view-info at which the w/h gain halves (lower = hardens faster)
    float warm_lambda_pos_base         = 0.15f;
    float warm_lambda_pos_gain         = 0.45f;
    float warm_lambda_size_base        = 0.02f;
    float warm_lambda_size_gain        = 0.18f;
    float warm_lambda_yaw_base         = 0.01f;
    float warm_lambda_yaw_gain         = 0.12f;
    float warm_confidence_decay         = 0.70f;
    float warm_confidence_coverage_gain = 0.35f;
    float warm_confidence_residual_gain = 0.65f;
};

// Fill a TableConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
TableConfig load_table_config(const ConfigLoader& cfg);

}  // namespace rc
