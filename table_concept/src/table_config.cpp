/*
 * table_config.cpp — fill TableConfig from a RoboComp ConfigLoader.
 */

#include "table_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

TableConfig load_table_config(const ConfigLoader& cfg)
{
    TableConfig out;

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
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

    // Paths

    // Agent convergence
    out.state_eps                = getf("TableConcept.StateEps",               0.04f);
    out.K_stable                 = geti("TableConcept.KStable",                30);
    out.max_direct_fit_points    = geti("TableConcept.MaxDirectFitPoints",     400);
    out.detection_alive_max_frames = geti("TableConcept.DetectionAliveMaxFrames", 40);
    out.M_diverge                = geti("TableConcept.MDiverge",               20);
    out.explanation_ratio_thresh = getf("TableConcept.ExplanationRatioThresh", 0.3f);
    out.obs_distance             = getf("TableConcept.ObsDistance",            1.8f);
    out.delta_min                = getf("TableConcept.DeltaMin",               20.0f);
    out.epistemic_cooldown_cycles= geti("TableConcept.EpistemicCooldownCycles", 200);
    out.table_log_period_frames  = geti("TableConcept.TableLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("TableConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("TableConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("TableConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("TableConcept.VoxelSelectHeightMarginM", 0.25f);
    out.top_band_gate_enabled    = getb("TableConcept.TopBandGate",            false);
    out.top_band_m               = getf("TableConcept.TopBandM",               0.08f);

    // TableModel
    out.sigma_obs          = getf("TableModel.SigmaObs",          0.05f);
    out.lambda_size        = getf("TableModel.LambdaSize",        0.15f);
    out.lambda_extent      = getf("TableModel.LambdaExtent",      2.0f);
    out.extent_pct_lo      = getf("TableModel.ExtentPctLo",       0.02f);
    out.extent_pct_hi      = getf("TableModel.ExtentPctHi",       0.98f);
    out.lambda_pos         = getf("TableModel.LambdaPos",         0.05f);
    out.lambda_state       = getf("TableModel.LambdaState",       0.02f);
    out.lambda_angle       = getf("TableModel.LambdaAngle",       0.01f);
    out.prior_size_std     = getf("TableModel.PriorSizeStd",      0.30f);
    out.evidence_sigma_m   = getf("TableModel.EvidenceSigmaM",    0.12f);
    out.evidence_ema_alpha = getf("TableModel.EvidenceEmaAlpha",  0.9f);
    out.process_std_pos_m  = getf("WarmStart.ProcessStdPosM",     0.001f);
    out.view_novelty_scale_m = getf("TableModel.ViewNoveltyScaleM", 0.15f);
    out.view_novelty_floor   = getf("TableModel.ViewNoveltyFloor",  0.05f);
    out.view_novelty_gate    = getb("TableModel.ViewNoveltyGate",   true);
    out.torch_threads      = geti("TableModel.TorchThreads",     2);
    out.use_ai2              = getb("TableModel.UseAI2",              false);
    out.ai2_sigma_base_m     = getf("TableModel.AI2SigmaBaseM",       0.03f);
    out.ai2_clutter_frac     = getf("TableModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("TableModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("TableModel.AI2PriorSizeStd",     0.30f);
    out.ai2_process_std_m    = getf("TableModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("TableModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_common_mode_pos_std  = getf("TableModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("TableModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("TableModel.AI2CommonModeYawStd",  0.03f);
    out.ai2_range_noise_lat_per_m = getf("TableModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("TableModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_trunc_gate_frac    = getf("TableModel.AI2TruncGateFrac",   0.10f);
    out.ai2_gn_iters         = geti("TableModel.AI2GnIters",          4);
    out.optimization_iters = geti("TableModel.OptimizationIters", 10);
    out.optimization_lr    = getf("TableModel.OptimizationLr",    0.05f);
    out.grad_clip          = getf("TableModel.GradClip",          2.0f);
    out.optimizer_type     = gets("TableModel.OptimizerType",     "adam");
    out.sgd_momentum       = getf("TableModel.SgdMomentum",       0.9f);
    {
        const auto loss_name = gets("TableModel.RobustLoss", "quadratic");
        const auto loss_type = robust_loss_type_from_string(loss_name);
        if (loss_type.has_value())
            out.robust_loss = loss_type.value();
        else
        {
            std::print("table_concept: unknown robust loss '{}' - using quadratic\n", loss_name);
            out.robust_loss = RobustLossType::Quadratic;
        }
    }
    out.robust_loss_scale  = getf("TableModel.RobustLossScale",  0.10f);
    out.robust_gnc_start_scale = getf("TableModel.RobustGncStartScale", 0.80f);
    out.mask_precision     = getf("TableModel.MaskPrecision",     0.30f);
    out.sil_tangent_samples = geti("TableModel.MaskSilhouetteSamples", 8);
    out.robust_gnc_decay_cycles = geti("TableModel.RobustGncDecayCycles", 20);

    // SampleQueue
    out.num_angle_bins               = geti("SampleQueue.NumAngleBins",              24);
    out.num_z_bins                   = geti("SampleQueue.NumZBins",                  10);
    out.max_per_bin                  = geti("SampleQueue.MaxPerBin",                 2);
    out.sdf_threshold_for_storage    = getf("SampleQueue.SdfThresholdForStorage",    0.30f);
    out.min_frames_before_historical = geti("SampleQueue.MinFramesBeforeHistorical", 10);
    out.historical_warmup_frames     = geti("SampleQueue.HistoricalWarmupFrames",    5);
    out.max_new_points_per_frame     = geti("SampleQueue.MaxNewPointsPerFrame",      30);
    out.rfe_alpha                    = getf("SampleQueue.RfeAlpha",                  0.98f);
    out.rfe_max_threshold            = getf("SampleQueue.RfeMaxThreshold",           2.0f);
    out.rfe_weight_gain              = getf("SampleQueue.RfeWeightGain",             0.25f);
    out.min_anchor_weight            = getf("SampleQueue.MinAnchorWeight",           0.12f);
    out.edge_bonus_weight            = getf("SampleQueue.EdgeBonusWeight",           0.3f);
    out.edge_proximity_threshold     = getf("SampleQueue.EdgeProximityThreshold",    0.05f);

    // WarmStart
    out.warm_pts_min                  = getf("WarmStart.PtsMin",                  12.0f);
    out.warm_pts_max                  = getf("WarmStart.PtsMax",                  30.0f);
    out.fisher_info_decay             = getf("WarmStart.FisherInfoDecay",          0.95f);
    out.fisher_peak_decay             = getf("WarmStart.FisherPeakDecay",          1.0f);
    out.fisher_peak_ratchet           = getf("WarmStart.FisherPeakRatchet",        2.5f);
    out.fisher_peak_ema               = getf("WarmStart.FisherPeakEma",            0.30f);
    out.fisher_process_std_m          = getf("WarmStart.FisherProcessStdM",        0.005f);
    out.fisher_process_std_yaw        = getf("WarmStart.FisherProcessStdYaw",      0.01f);
    out.bad_fit_fe_ratio              = getf("WarmStart.BadFitFeRatio",            4.0f);
    out.fe_baseline_ema               = getf("WarmStart.FeBaselineEma",            0.10f);
    out.fisher_grad_clamp             = getf("WarmStart.FisherGradClamp",          2.0f);
    out.fisher_views_half             = getf("WarmStart.FisherViewsHalf",          4.0f);
    out.mask_conf_floor               = getf("WarmStart.MaskConfFloor",           0.2f);
    out.mask_conf_ref                 = getf("WarmStart.MaskConfRef",             0.5f);
    out.mask_conf_power               = getf("WarmStart.MaskConfPower",           2.0f);
    out.fisher_csv_path               = gets("WarmStart.FisherCsvPath",            "");
    out.ai2_csv_path                  = gets("TableModel.AI2CsvPath",              "");
    out.rt_cov_scale                  = getf("WarmStart.RtCovScale",              1.0f);
    out.rt_cov_add_chain              = getb("WarmStart.RtCovAddChain",          true);
    out.warm_confidence_decay         = getf("WarmStart.ConfidenceDecay",         0.70f);
    out.warm_confidence_coverage_gain = getf("WarmStart.ConfidenceCoverageGain",  0.35f);
    out.warm_confidence_residual_gain = getf("WarmStart.ConfidenceResidualGain",  0.65f);

    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.50f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",      300);
    out.tracker_death_enabled    = getb("Tracker.DeathEnabled",     false);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.60f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.05f);
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM",      1.0f);
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM",      0.6f);
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM",     0.75f);
    out.tracker_birth_size_std   = getf("Tracker.BirthSizeStd",     0.15f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",          false);

    std::print("table_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
