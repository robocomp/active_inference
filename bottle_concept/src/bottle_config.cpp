/*
 * bottle_config.cpp — fill BottleConfig from a RoboComp ConfigLoader.
 */

#include "bottle_config.h"

#include <cstdlib>   // std::getenv, std::atoi
#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

BottleConfig load_bottle_config(const ConfigLoader& cfg)
{
    BottleConfig out;

    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };

    out.priors_path = gets("BottleConcept.PriorsPath", "etc/object_priors.toml");

    out.fe_eps            = getf("BottleConcept.FEps",            1e-3f);
    out.K_stable          = geti("BottleConcept.KStable",         30);
    out.write_threshold   = getf("BottleConcept.WriteThreshold",  1e-3f);
    out.log_period_frames = geti("BottleConcept.LogPeriodFrames", 30);
    out.voxel_bank_max_points        = geti("BottleConcept.VoxelBankMaxPoints",        4000);
    out.voxel_bank_quantization_m    = getf("BottleConcept.VoxelBankQuantizationM",    0.01f);
    out.voxel_select_radius_margin_m = getf("BottleConcept.VoxelSelectRadiusMarginM",  0.10f);
    out.voxel_select_height_margin_m = getf("BottleConcept.VoxelSelectHeightMarginM",  0.10f);

    out.prior_radius       = getf("BottleModel.PriorRadius",       0.035f);
    out.prior_height       = getf("BottleModel.PriorHeight",       0.20f);
    out.prior_size_std     = getf("BottleModel.PriorSizeStd",      0.03f);
    out.mask_precision     = getf("BottleModel.MaskPrecision",     0.0f);
    out.mask_conf_weight   = getb("BottleModel.MaskConfWeight",    true);
    out.mask_conf_floor    = getf("BottleModel.MaskConfFloor",     0.2f);
    out.mask_conf_ref      = getf("BottleModel.MaskConfRef",       0.5f);
    out.mask_conf_power    = getf("BottleModel.MaskConfPower",     2.0f);

    out.ai2_sigma_base_m         = getf("BottleModel.AI2SigmaBaseM",          0.02f);
    out.ai2_clutter_frac         = getf("BottleModel.AI2ClutterFrac",         0.10f);
    out.ai2_clutter_scale_m      = getf("BottleModel.AI2ClutterScaleM",       0.08f);
    out.ai2_prior_pos_std        = getf("BottleModel.AI2PriorPosStd",         0.30f);
    out.ai2_prior_size_std       = getf("BottleModel.AI2PriorSizeStd",        0.03f);
    out.ai2_process_std_m        = getf("BottleModel.AI2ProcessStdM",         0.005f);
    out.ai2_common_mode_pos_std  = getf("BottleModel.AI2CommonModePosStd",    0.02f);
    out.ai2_common_mode_size_std = getf("BottleModel.AI2CommonModeSizeStd",   0.01f);
    out.ai2_gn_iters             = geti("BottleModel.AI2GnIters",             4);
    out.ai2_csv_path             = gets("BottleModel.AI2CsvPath",             "");
    out.support_sigma_z          = getf("Support.SigmaZ",            0.04f);
    out.support_footprint_margin = getf("Support.FootprintMargin",   0.05f);
    out.support_lambda_xy        = getf("Support.LambdaXY",          50.0f);
    out.support_decision_margin  = getf("Support.DecisionMargin",    2.0f);
    out.support_commit_cycles    = geti("Support.CommitCycles",      8);
    out.tracker_enabled          = getb("Tracker.Enabled",            false);
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",    9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",      0.30f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",        6);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",        90);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",       0.20f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM",   0.05f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",       0.30f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",            false);
    out.masks_use_camera_frame = getb("Masks.UseCameraFrame", true);
    out.masks_source_frame     = gets("Masks.SourceFrame",    "zed");
    out.masks_target_frame     = gets("Masks.TargetFrame",    "room");
    out.rt_cov_add_chain       = getb("Masks.RtCovAddChain",  true);
    out.epistemic_obs_distance    = getf("Epistemic.ObsDistance",     0.9f);
    out.epistemic_view_info       = getf("Epistemic.ViewInfo",        50.0f);
    out.epistemic_cooldown_cycles = geti("Epistemic.CooldownCycles",  200);
    out.epistemic_csv_path        = gets("Epistemic.CsvPath",         "");

    out.sdf_threshold_for_storage    = getf("SampleQueue.SdfThresholdForStorage",    0.03f);

    out.yaw_variance = getf("BottleConcept.YawVariance", 9.87f);

    // Static Webots ground-truth evaluation (room frame; cylinder centre).
    out.eval_enabled    = getb("Eval.Enabled",  false);
    out.eval_log_path   = gets("Eval.LogPath",  "etc/bottle_eval.csv");
    out.eval_gt_source  = gets("Eval.GtSource", "webots");
    out.eval_bottle_def = gets("Eval.BottleDef", "bottle");
    out.eval_robot_def  = gets("Eval.RobotDef",  "shadow");
    out.gt_cx     = getf("Eval.GtCx",     0.0f);
    out.gt_cy     = getf("Eval.GtCy",     0.0f);
    out.gt_cz     = getf("Eval.GtCz",     0.0f);
    out.gt_radius = getf("Eval.GtRadius", 0.0f);
    out.gt_height = getf("Eval.GtHeight", 0.0f);

    // One-shot bottle placement on start (Webots world coords; setObjectPose's native frame).
    out.place_on_start = getb("Scene.PlaceBottleOnStart", false);
    out.place_world_x  = getf("Scene.PlaceBottleWorldX", 0.0f);
    out.place_world_y  = getf("Scene.PlaceBottleWorldY", 0.0f);

    // Moving-bottle validation experiment.
    out.move_experiment    = getb("Eval.MoveExperiment", false);
    out.move_settle_cycles = geti("Eval.MoveSettleCycles", 25);
    out.move_step_m        = getf("Eval.MoveStep", 0.06f);
    out.move_grid_n        = geti("Eval.MoveGridN", 5);
    out.move_absolute      = getb("Eval.MoveAbsolute", false);
    out.move_xmin          = getf("Eval.MoveXmin", 0.0f);
    out.move_xmax          = getf("Eval.MoveXmax", 0.0f);
    out.move_ymin          = getf("Eval.MoveYmin", 0.0f);
    out.move_ymax          = getf("Eval.MoveYmax", 0.0f);
    // Static-restart validation (one grid pose per run; index from env BOTTLE_TEST_POSE).
    out.static_pose_test   = getb("Eval.StaticPoseTest", false);
    if (const char* p = std::getenv("BOTTLE_TEST_POSE"))
        out.static_pose_index = std::atoi(p);
    if (out.static_pose_test)
        out.move_experiment = false;          // static-restart takes precedence
    if (out.move_experiment or out.static_pose_test)
        out.eval_enabled = true;              // the experiment is pointless without logging

    std::print("bottle_concept: configuration loaded. tracker_enabled={} (Tracker.Enabled)\n",
               out.tracker_enabled);
    return out;
}

}  // namespace rc
