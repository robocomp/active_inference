/*
 * residual_config.cpp — fill ResidualConfig from a RoboComp ConfigLoader.
 */

#include "residual_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc
{

ResidualConfig load_residual_config(const ConfigLoader& cfg)
{
    ResidualConfig out;

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

    // ── agent loop ──
    out.fe_eps                 = getf("ResidualConcept.FEps",                 5e-3f);
    out.K_stable               = geti("ResidualConcept.KStable",             20);
    out.diverged_retire_frames = geti("ResidualConcept.DivergedRetireFrames", 0);
    out.log_period_frames      = geti("ResidualConcept.LogPeriodFrames",     30);
    out.write_threshold_m      = getf("ResidualConcept.WriteThresholdM",     0.02f);

    // ── belief ──
    out.belief.sigma_base_m         = getf("ResidualModel.AI2SigmaBaseM",        0.03f);
    out.belief.clutter_frac         = getf("ResidualModel.AI2ClutterFrac",       0.15f);
    out.belief.clutter_scale_m      = getf("ResidualModel.AI2ClutterScaleM",     0.12f);
    out.belief.prior_pos_std        = getf("ResidualModel.AI2PriorPosStd",       0.50f);
    out.belief.prior_yaw_std        = getf("ResidualModel.AI2PriorYawStd",       1.57f);
    out.belief.prior_size_std       = getf("ResidualModel.AI2PriorSizeStd",      0.50f);
    out.belief.process_std_pos_m    = getf("ResidualModel.AI2ProcessStdPosM",    0.02f);
    out.belief.process_std_yaw      = getf("ResidualModel.AI2ProcessStdYaw",     0.10f);
    out.belief.process_std_size_m   = getf("ResidualModel.AI2ProcessStdSizeM",   0.06f);
    out.belief.common_mode_pos_std  = getf("ResidualModel.AI2CommonModePosStd",  0.03f);
    out.belief.common_mode_yaw_std  = getf("ResidualModel.AI2CommonModeYawStd",  0.20f);
    out.belief.common_mode_size_std = getf("ResidualModel.AI2CommonModeSizeStd", 0.03f);
    out.belief.gn_iters             = geti("ResidualModel.AI2GnIters",           5);
    out.belief.min_size             = getf("ResidualModel.MinSize",              0.05f);
    out.belief.max_size             = getf("ResidualModel.MaxSize",              4.0f);
    out.ai2_csv_path                = gets("ResidualModel.AI2CsvPath",           "");
    out.diag_csv_path               = gets("ResidualConcept.DiagCsvPath",        "etc/residual_diag.csv");

    // ── clusterer (perception front-end) ──
    out.cluster.floor_z0         = getf("Clusterer.FloorZ0",         0.06f);
    out.cluster.floor_slope      = getf("Clusterer.FloorSlope",      0.04f);
    out.cluster.ceil_z           = getf("Clusterer.CeilZ",           1.80f);
    out.cluster.min_vertical_extent_m = getf("Clusterer.MinVerticalExtentM", 0.10f);
    out.cluster.min_top_z_m           = getf("Clusterer.MinTopZM",           0.30f);
    out.cluster.robot_radius_m   = getf("Clusterer.RobotRadiusM",    0.40f);
    out.cluster.wall_margin_m    = getf("Clusterer.WallMarginM",     0.30f);
    out.cluster.bpearl_floor_z0  = getf("Clusterer.BpearlFloorZ0",   0.12f);
    out.cluster.explain_margin_m = getf("Clusterer.ExplainMarginM",  0.04f);
    out.cluster.explain_sensor_sigma_m = getf("Clusterer.ExplainSensorSigmaM", 0.03f);
    out.cluster.explain_fit_margin_m   = getf("Clusterer.ExplainFitMarginM",   0.10f);  // under-fit slack (< clearance)
    out.cluster.dbscan_eps_m     = getf("Clusterer.DbscanEpsM",      0.20f);
    out.cluster.dbscan_min_pts   = geti("Clusterer.DbscanMinPts",    5);
    out.cluster.dbscan_z_scale   = getf("Clusterer.DbscanZScale",    0.35f);
    out.cluster.min_cluster_pts  = geti("Clusterer.MinClusterPts",   12);
    // Navigation fragment-merge gap = robot width + an excess: fragments closer than the robot can pass
    // between are merged into ONE obstacle (coalesces a LiDAR-shattered tabletop). 0 width → merge off.
    out.cluster.merge_gap_m      = getf("Clusterer.RobotWidthM", 0.50f) + getf("Clusterer.MergeExcessM", 0.10f);

    // ── tracker ──
    out.tracker_gate_mahalanobis  = getf("Tracker.GateMahalanobis",   9.0f);
    out.tracker_gate_fallback_m   = getf("Tracker.GateFallbackM",     0.35f);
    out.tracker_birth_frames      = geti("Tracker.BirthFrames",       4);
    out.tracker_death_frames      = geti("Tracker.DeathFrames",       30);
    out.tracker_birth_min_sep_m   = getf("Tracker.BirthMinSepM",      0.25f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM",   0.08f);
    out.tracker_merge_overlap     = getf("Tracker.MergeOverlap",      0.30f);
    out.tracker_nll_cost          = getb("Tracker.NllCost",           false);
    out.tracker_death_enabled     = getb("Tracker.DeathEnabled",      false);
    out.tracker_expected_visible_range_m = getf("Tracker.ExpectedVisibleRangeM", 6.0f);

    // ── occupancy-evidence removal (ray-carving) ──
    out.carve.sensor_sigma_m = getf("Carve.SensorSigmaM",  0.03f);   // LiDAR range noise σ
    out.carve.detection_prob = getf("Carve.DetectionProb", 0.85f);   // P(beam through occupied box returns)
    out.carve.clutter_prob   = getf("Carve.ClutterProb",   0.05f);   // P(beam through empty box returns inside)
    out.removal_prob          = getf("Carve.RemovalProb",       0.12f);  // remove when P(occupied) < this
    out.occupancy_logodds_max = getf("Carve.OccupancyLogOddsMax", 4.0f); // ± clamp on the accumulated log-odds

    // ── per-point reliability covariates ──
    out.range_noise_coeff = getf("ResidualModel.RangeNoiseCoeff", 0.01f);
    out.ego_motion_lag_s  = getf("ResidualModel.EgoMotionLagS",   0.05f);

    // ── ZED dense-depth boost ──
    out.zed_boost_enabled       = getb("ZedBoost.Enabled",       true);
    out.zed_detection_enabled   = getb("ZedBoost.FeedDetection", false);   // dense ZED → clustering
    out.grid_zed_enabled        = getb("ZedBoost.FeedGrid",      true);    // dense ZED → occupancy grid
    out.grid_field_ema_up       = getf("Grid.FieldEmaUp",        1.0f);    // published-field temporal low-pass (rise)
    out.grid_field_ema_down     = getf("Grid.FieldEmaDown",      0.30f);   // ...and fall (slow = stable)
    out.motion_vel0_mps         = getf("Grid.MotionVel0",        0.5f);    // linear speed (m/s) that halves sweep trust
    out.motion_omega0_rps       = getf("Grid.MotionOmega0",      0.6f);    // yaw rate (rad/s) that halves sweep trust
    out.grid_forget_half_life_s = getf("Grid.ForgetHalfLifeS",   10.0f);   // evidence half-life in an UNOBSERVED cell
    out.grid_self_body_radius_m = getf("Grid.SelfBodyRadiusM",   0.55f);   // body envelope for the sensor-model term
    out.grid_self_body_sigma_m  = getf("Grid.SelfBodySigmaM",    0.08f);   // its positional uncertainty
    // Robust ZED infrastructure subtraction (reuse the floor/ceiling heights; ZED depth-noise band σ0+q·r²).
    out.zed_infra.floor_z0      = out.cluster.floor_z0;
    out.zed_infra.ceil_z        = out.cluster.ceil_z;
    out.zed_infra.wall_margin_m = getf("ZedBoost.InfraWallMarginM", 0.10f);
    out.zed_infra.sigma0_m      = getf("ZedBoost.InfraSigma0M",     0.03f);
    out.zed_infra.sigma_quad    = getf("ZedBoost.InfraSigmaQuad",   0.006f);   // stereo depth noise ∝ range²
    out.zed_infra.k             = getf("ZedBoost.InfraK",           3.0f);
    out.zed_sigma_m             = getf("ZedBoost.SigmaM",        0.02f);
    out.zed_boost.subsample     = geti("ZedBoost.Subsample",     3);
    out.zed_boost.xy_margin_m   = getf("ZedBoost.XyMarginM",     0.10f);
    out.zed_boost.z_margin_m    = getf("ZedBoost.ZMarginM",      0.10f);
    out.zed_boost.min_depth_m   = getf("ZedBoost.MinDepthM",     0.30f);
    out.zed_boost.max_depth_m   = getf("ZedBoost.MaxDepthM",     5.0f);
    out.zed_boost.max_points    = geti("ZedBoost.MaxPoints",     1500);

    // ── RGB-semantic floor down-weighting (second, uncorrelated cue vs ZED floor phantoms) ──
    out.semantic_floor.enabled          = getb("Semantic.DownweightFloor", false);   // master flag (OFF by default)
    out.semantic_floor.floor_suppress   = getf("Semantic.FloorSuppress",   0.60f);
    out.semantic_floor.height_scale_m   = getf("Semantic.HeightScaleM",    0.15f);
    out.semantic_floor.fresh_half_life_s = getf("Semantic.FreshHalfLifeS", 0.50f);
    out.semantic_floor.floor_z0         = out.cluster.floor_z0;   // share the geometric nav band
    out.semantic_floor.floor_slope      = out.cluster.floor_slope;

    // ── data-driven floor plane (grid floor explainer follows an offset/tilted floor) ──
    out.floor_plane.enabled          = getb("FloorPlane.Enabled",         false);
    out.floor_plane.candidate_band_m = getf("FloorPlane.CandidateBandM",  0.25f);
    out.floor_plane.trim_k           = getf("FloorPlane.TrimK",           2.5f);
    out.floor_plane.iters            = geti("FloorPlane.Iters",           2);
    out.floor_plane.min_candidates   = geti("FloorPlane.MinCandidates",   80);
    out.floor_plane.ema              = getf("FloorPlane.Ema",             0.10f);
    out.floor_plane.max_offset_m     = getf("FloorPlane.MaxOffsetM",      0.30f);
    out.floor_plane.max_tilt         = getf("FloorPlane.MaxTilt",         0.15f);
    // Optional override of the ADE20K walkable-ground class set (comma/space-separated ids). Empty ⇒ the built-in
    // default (floor/road/grass/sidewalk/earth/rug/field/sand/path/runway/dirt-track/land). Lets the set be tuned
    // to the deployed segmenter without a rebuild.
    if (const std::string s = gets("Semantic.FloorClassIds", ""); not s.empty())
    {
        std::vector<std::uint8_t> ids; std::string tok;
        for (const char c : s + ",")
        {
            if (c == ',' or c == ' ' or c == ';' or c == '\t')
            { if (not tok.empty()) { ids.push_back(static_cast<std::uint8_t>(std::stoi(tok))); tok.clear(); } }
            else tok.push_back(c);
        }
        if (not ids.empty()) out.semantic_floor.floor_class_ids = std::move(ids);
    }

    // ── residual-specific ──
    out.dissolve_explained_frac = getf("ResidualConcept.DissolveExplainedFrac", 0.60f);

    // ── conservative publish (safe-clearance footprint) ──
    out.publish_extent_sigma_k  = getf("ResidualConcept.PublishExtentSigmaK",  2.0f);
    out.publish_safety_margin_m = getf("ResidualConcept.PublishSafetyMarginM", 0.05f);

    // ── media / cov ──
    out.lidar_frame_node  = gets("ResidualModel.LidarFrameNode",  "lidar3D");
    out.use_bpearl        = getb("ResidualModel.UseBpearl",       true);
    out.rt_cov_add_chain  = getb("ResidualConcept.RtCovAddChain", true);
    out.unobservable_var  = getf("ResidualConcept.UnobservableVar", 9.87f);

    std::print("residual_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
