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

    // BottleModel parameters (forwarded to BottleModelParams)
    float sigma_obs         = 0.02f;
    float lambda_size       = 0.5f;
    float lambda_pos        = 0.05f;
    float lambda_state      = 0.02f;
    float prior_radius      = 0.035f;
    float prior_height      = 0.20f;
    float prior_size_std    = 0.03f;
    int   optimization_iters = 15;
    float optimization_lr   = 0.005f;   // cm-scale object: 10× smaller than the table's
    float grad_clip         = 2.0f;
    std::string optimizer_type = "adam";
    float sgd_momentum      = 0.9f;
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.05f;
    float mask_precision    = 0.0f;   // RGB-mask silhouette likelihood weight (0 = off)
    float cov_eff_scale     = 1.0f;   // covariance calibration: N_eff = N·scale (NEES → ~3)
    // Cold-start seed de-projection: the visible (front-arc-only) point centroid sits ~one radius
    // toward the camera from the true cylinder axis, so snapping the seed to it biases the model
    // camera-ward (perceived "closer than real"; the observed points fall behind its bbox). Push the
    // seed AWAY from the camera by frac·radius in the horizontal plane to recover the axis. 0 = off.
    float seed_deproject_frac = 1.0f;
    // Free-space / visibility FE term (BottleModelParams): penalise the model occupying the space
    // between the camera and the observed points. The source cure for the camera-ward depth bias.
    float lambda_freespace = 0.0f;     // 0 = off
    float freespace_margin = 0.01f;    // m: carve-sample offset in front of each point

    // SampleQueue parameters (forwarded to SampleQueueParams)
    int   num_angle_bins               = 16;
    int   num_z_bins                   = 6;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.03f;
    int   min_frames_before_historical = 10;
    int   historical_warmup_frames     = 5;
    int   max_new_points_per_frame     = 20;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float rfe_weight_gain              = 0.25f;
    float min_anchor_weight            = 0.12f;
    float edge_bonus_weight            = 0.3f;
    float edge_proximity_threshold     = 0.01f;
    float z_bin_size                   = 0.04f;

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
