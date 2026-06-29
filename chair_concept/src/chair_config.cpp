/*
 * chair_config.cpp — fill ChairConfig from a RoboComp ConfigLoader.
 */

#include "chair_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

ChairConfig load_chair_config(const ConfigLoader& cfg)
{
    ChairConfig out;

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
    out.priors_path = gets("ChairConcept.PriorsPath", "etc/object_priors.toml");

    // Agent convergence
    out.fe_eps                   = getf("ChairConcept.FEps",                   1e-3f);
    out.state_eps                = getf("ChairConcept.StateEps",               0.04f);
    out.K_stable                 = geti("ChairConcept.KStable",                30);
    out.max_direct_fit_points    = geti("ChairConcept.MaxDirectFitPoints",     400);
    out.detection_alive_max_frames = geti("ChairConcept.DetectionAliveMaxFrames", 40);
    out.M_diverge                = geti("ChairConcept.MDiverge",               20);
    out.staleness_frames         = getf("ChairConcept.StalenessFrames",        90.0f);
    out.explanation_ratio_thresh = getf("ChairConcept.ExplanationRatioThresh", 0.3f);
    out.write_threshold          = getf("ChairConcept.WriteThreshold",         1e-3f);
    out.obs_distance             = getf("ChairConcept.ObsDistance",            1.8f);
    out.delta_min                = getf("ChairConcept.DeltaMin",               20.0f);
    out.gain_threshold           = getf("ChairConcept.GainThreshold",          0.1f);
    out.epistemic_use_info_gain  = getb("ChairConcept.EpistemicInfoGain",      true);
    out.epistemic_min_info_gain  = getf("ChairConcept.EpistemicMinInfoGain",   0.3f);
    out.epistemic_rearm_info_gain= getf("ChairConcept.EpistemicRearmInfoGain", 5.0f);
    out.epistemic_cooldown_cycles= geti("ChairConcept.EpistemicCooldownCycles", 200);
    out.chair_log_period_frames  = geti("ChairConcept.ChairLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("ChairConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("ChairConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("ChairConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("ChairConcept.VoxelSelectHeightMarginM", 0.25f);
    out.top_band_gate_enabled    = getb("ChairConcept.TopBandGate",            false);
    out.top_band_m               = getf("ChairConcept.TopBandM",               0.08f);

    // ChairModel
    out.sigma_obs          = getf("ChairModel.SigmaObs",          0.05f);
    // A chair is a RIGID, standardised object (Webots WoodenChair ≈ 0.45×0.45). Hold the size
    // strongly to the prior so the fit refines mostly position/yaw and the flat seat slab does NOT
    // balloon to engulf the (mask-contaminated) point spread. LambdaSize↑ + PriorSizeStd↓ stiffen
    // the size prior; LambdaExtent=0 disables the extent term that otherwise pins the seat to the
    // observed (contaminated) point span — the backrest already breaks the symmetry for position.
    out.lambda_size        = getf("ChairModel.LambdaSize",        1.5f);
    out.lambda_extent      = getf("ChairModel.LambdaExtent",      0.0f);
    out.extent_pct_lo      = getf("ChairModel.ExtentPctLo",       0.05f);
    out.extent_pct_hi      = getf("ChairModel.ExtentPctHi",       0.95f);
    out.lambda_pos         = getf("ChairModel.LambdaPos",         0.05f);
    out.lambda_state       = getf("ChairModel.LambdaState",       0.02f);
    out.lambda_angle       = getf("ChairModel.LambdaAngle",       0.01f);
    out.prior_size_std     = getf("ChairModel.PriorSizeStd",      0.06f);
    out.optimization_iters = geti("ChairModel.OptimizationIters", 10);
    out.optimization_lr    = getf("ChairModel.OptimizationLr",    0.05f);
    out.grad_clip          = getf("ChairModel.GradClip",          2.0f);
    out.optimizer_type     = gets("ChairModel.OptimizerType",     "adam");
    out.sgd_momentum       = getf("ChairModel.SgdMomentum",       0.9f);
    {
        const auto loss_name = gets("ChairModel.RobustLoss", "quadratic");
        const auto loss_type = robust_loss_type_from_string(loss_name);
        if (loss_type.has_value())
            out.robust_loss = loss_type.value();
        else
        {
            std::print("chair_concept: unknown robust loss '{}' - using quadratic\n", loss_name);
            out.robust_loss = RobustLossType::Quadratic;
        }
    }
    out.robust_loss_scale  = getf("ChairModel.RobustLossScale",  0.10f);
    out.robust_gnc_start_scale = getf("ChairModel.RobustGncStartScale", 0.80f);
    out.mask_precision     = getf("ChairModel.MaskPrecision",     0.30f);
    out.sil_tangent_samples = geti("ChairModel.MaskSilhouetteSamples", 8);
    out.sil_reopen_residual_m = getf("ChairModel.SilReopenResidualM", 0.15f);
    out.robust_gnc_decay_cycles = geti("ChairModel.RobustGncDecayCycles", 20);

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
    out.fisher_quality_rescale        = getb("WarmStart.FisherQualityRescale",     true);
    out.fisher_peak_decay             = getf("WarmStart.FisherPeakDecay",          1.0f);
    out.fisher_peak_ratchet           = getf("WarmStart.FisherPeakRatchet",        2.5f);
    out.fisher_peak_ema               = getf("WarmStart.FisherPeakEma",            0.30f);
    out.fisher_process_std_m          = getf("WarmStart.FisherProcessStdM",        0.005f);
    out.fisher_process_std_yaw        = getf("WarmStart.FisherProcessStdYaw",      0.01f);
    out.freeze_belief_on_stale        = getb("WarmStart.FreezeBeliefOnStale",      true);
    out.fisher_grad_clamp             = getf("WarmStart.FisherGradClamp",          2.0f);
    out.fisher_maturity_stiffness     = getb("WarmStart.FisherMaturityStiffness",  true);
    out.fisher_views_half             = getf("WarmStart.FisherViewsHalf",          4.0f);
    out.counter_evidence_gate         = getb("WarmStart.CounterEvidenceGate",      false);
    out.counter_evidence_band_m       = getf("WarmStart.CounterEvidenceBandM",     0.02f);
    out.counter_evidence_band_rad     = getf("WarmStart.CounterEvidenceBandRad",   0.05f);
    out.counter_evidence_base_pos     = getf("WarmStart.CounterEvidenceBasePos",    6.0f);
    out.counter_evidence_lambda_pos   = getf("WarmStart.CounterEvidenceLambdaPos",  1.0f);
    out.counter_evidence_base_size    = getf("WarmStart.CounterEvidenceBaseSize",   3.0f);
    out.counter_evidence_lambda_size  = getf("WarmStart.CounterEvidenceLambdaSize", 0.5f);
    out.counter_evidence_decay        = getf("WarmStart.CounterEvidenceDecay",     0.8f);
    out.counter_evidence_unlock_deflate = getf("WarmStart.CounterEvidenceUnlockDeflate", 0.3f);
    out.counter_evidence_step_cap     = getf("WarmStart.CounterEvidenceStepCap",  1.0f);
    out.mask_conf_weight              = getb("WarmStart.MaskConfWeight",          true);
    out.mask_conf_floor               = getf("WarmStart.MaskConfFloor",           0.2f);
    out.mask_conf_ref                 = getf("WarmStart.MaskConfRef",             0.5f);
    out.mask_conf_power               = getf("WarmStart.MaskConfPower",           2.0f);
    out.fisher_csv_path               = gets("WarmStart.FisherCsvPath",            "");
    out.rt_cov_upload                 = getb("WarmStart.RtCovUpload",              true);
    out.rt_cov_scale                  = getf("WarmStart.RtCovScale",              1.0f);
    out.rt_cov_add_chain              = getb("WarmStart.RtCovAddChain",          true);
    out.warm_lambda_pos_base          = getf("WarmStart.LambdaPosBase",           0.15f);
    out.warm_lambda_pos_gain          = getf("WarmStart.LambdaPosGain",           0.45f);
    out.warm_lambda_size_base         = getf("WarmStart.LambdaSizeBase",          0.02f);
    out.warm_lambda_size_gain         = getf("WarmStart.LambdaSizeGain",          0.18f);
    out.warm_lambda_yaw_base          = getf("WarmStart.LambdaYawBase",           0.01f);
    out.warm_lambda_yaw_gain          = getf("WarmStart.LambdaYawGain",           0.12f);
    out.warm_confidence_decay         = getf("WarmStart.ConfidenceDecay",         0.70f);
    out.warm_confidence_coverage_gain = getf("WarmStart.ConfidenceCoverageGain",  0.35f);
    out.warm_confidence_residual_gain = getf("WarmStart.ConfidenceResidualGain",  0.65f);

    out.tracker_enabled          = getb("Tracker.Enabled",          false);
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.40f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.20f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",      300);
    out.tracker_death_enabled    = getb("Tracker.DeathEnabled",     false);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.70f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.20f);
    out.tracker_prune_enabled         = getb("Tracker.PruneEnabled",        true);
    out.tracker_prune_maturity_cycles = geti("Tracker.PruneMaturityCycles", 90);
    out.tracker_prune_patience        = geti("Tracker.PrunePatience",       30);
    out.tracker_birth_seat_w     = getf("Tracker.BirthSeatW",       0.45f);
    out.tracker_birth_seat_d     = getf("Tracker.BirthSeatD",       0.45f);
    out.tracker_birth_seat_h     = getf("Tracker.BirthSeatH",       0.45f);
    out.tracker_birth_back_h     = getf("Tracker.BirthBackH",       0.45f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",          false);

    std::print("chair_concept: configuration loaded. tracker_enabled={} (Tracker.Enabled)\n", out.tracker_enabled);
    return out;
}

}  // namespace rc
