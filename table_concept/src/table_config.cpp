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
    out.priors_path = gets("TableConcept.PriorsPath", "etc/object_priors.toml");

    // Agent convergence
    out.fe_eps                   = getf("TableConcept.FEps",                   1e-3f);
    out.state_eps                = getf("TableConcept.StateEps",               0.04f);
    out.K_stable                 = geti("TableConcept.KStable",                30);
    out.max_direct_fit_points    = geti("TableConcept.MaxDirectFitPoints",     400);
    out.detection_alive_max_frames = geti("TableConcept.DetectionAliveMaxFrames", 40);
    out.M_diverge                = geti("TableConcept.MDiverge",               20);
    out.staleness_frames         = getf("TableConcept.StalenessFrames",        90.0f);
    out.explanation_ratio_thresh = getf("TableConcept.ExplanationRatioThresh", 0.3f);
    out.write_threshold          = getf("TableConcept.WriteThreshold",         1e-3f);
    out.obs_distance             = getf("TableConcept.ObsDistance",            1.8f);
    out.delta_min                = getf("TableConcept.DeltaMin",               20.0f);
    out.gain_threshold           = getf("TableConcept.GainThreshold",          0.1f);
    out.table_log_period_frames  = geti("TableConcept.TableLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("TableConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("TableConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("TableConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("TableConcept.VoxelSelectHeightMarginM", 0.25f);

    // TableModel
    out.sigma_obs          = getf("TableModel.SigmaObs",          0.05f);
    out.lambda_size        = getf("TableModel.LambdaSize",        0.15f);
    out.lambda_extent      = getf("TableModel.LambdaExtent",      2.0f);
    out.lambda_pos         = getf("TableModel.LambdaPos",         0.05f);
    out.lambda_state       = getf("TableModel.LambdaState",       0.02f);
    out.lambda_angle       = getf("TableModel.LambdaAngle",       0.01f);
    out.prior_size_std     = getf("TableModel.PriorSizeStd",      0.30f);
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
    out.sil_reopen_residual_m = getf("TableModel.SilReopenResidualM", 0.15f);
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
    out.warm_coverage_min_side        = getf("WarmStart.CoverageMinSide",         2.0f);
    out.warm_rho_freeze               = getf("WarmStart.RhoFreeze",               0.25f);
    out.warm_size_pts_release         = getf("WarmStart.SizePtsRelease",          0.60f);
    out.warm_settle_cycles            = geti("WarmStart.SettleCycles",            40);
    out.warm_reopen_admit             = geti("WarmStart.ReopenAdmit",             8);
    out.warm_settle_floor             = getf("WarmStart.SettleFloor",             0.05f);
    out.warm_info_half                = getf("WarmStart.InfoHalf",                20.0f);
    out.fisher_filter_enabled         = getb("WarmStart.FisherFilterEnabled",     true);
    out.fisher_kalman_stiffness       = getb("WarmStart.KalmanGainStiffness",     false);
    out.fisher_info_decay             = getf("WarmStart.FisherInfoDecay",          1.0f);
    out.fisher_process_std_m          = getf("WarmStart.FisherProcessStdM",        0.005f);
    out.fisher_process_std_yaw        = getf("WarmStart.FisherProcessStdYaw",      0.01f);
    out.fisher_csv_path               = gets("WarmStart.FisherCsvPath",            "");
    out.warm_lambda_pos_base          = getf("WarmStart.LambdaPosBase",           0.15f);
    out.warm_lambda_pos_gain          = getf("WarmStart.LambdaPosGain",           0.45f);
    out.warm_lambda_size_base         = getf("WarmStart.LambdaSizeBase",          0.02f);
    out.warm_lambda_size_gain         = getf("WarmStart.LambdaSizeGain",          0.18f);
    out.warm_lambda_yaw_base          = getf("WarmStart.LambdaYawBase",           0.01f);
    out.warm_lambda_yaw_gain          = getf("WarmStart.LambdaYawGain",           0.12f);
    out.warm_confidence_decay         = getf("WarmStart.ConfidenceDecay",         0.70f);
    out.warm_confidence_coverage_gain = getf("WarmStart.ConfidenceCoverageGain",  0.35f);
    out.warm_confidence_residual_gain = getf("WarmStart.ConfidenceResidualGain",  0.65f);

    std::print("table_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
