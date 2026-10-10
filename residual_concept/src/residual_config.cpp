/*
 * residual_config.cpp — fill ResidualConfig from a RoboComp ConfigLoader.
 */

#include "residual_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc
{

ResidualConfig load_residual_config(const ConfigLoader& cfg)
{
    ResidualConfig out;

    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "residual_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ── agent loop ──
    out.fe_eps                 = getf("ResidualConcept.FEps", 5e-3f,
            "|ΔFE| convergence band");
    out.K_stable               = geti("ResidualConcept.KStable", 20,
            "consecutive converged frames before 'settled'");
    out.diverged_retire_frames = geti("ResidualConcept.DivergedRetireFrames", 0,
            "legacy no-data-fit counter; 0 = OFF (removal is now evidence-based)");
    out.log_period_frames      = geti("ResidualConcept.LogPeriodFrames", 30,
            "no-data-fit counter (failing to fit ≠ compelling evidence the obstacle is gone)");
    out.grid_publish_every_n   = geti("ResidualConcept.GridPublishEveryN", 1,
            "Compute cycles per publish of `grid_occupied_cells` (Period.Compute = 100 ms, so 1 = 10 Hz, 5 = 2 Hz)");
    out.write_threshold_m      = getf("ResidualConcept.WriteThresholdM", 0.02f,
            "geometry dead-band: republish only past this centre/size change");

    // ── belief ──
    out.belief.sigma_base_m         = getf("ResidualModel.AI2SigmaBaseM", 0.03f,
            "base on-surface noise std (m); R = σ² (+ chain + …) per point");
    out.belief.clutter_frac         = getf("ResidualModel.AI2ClutterFrac", 0.15f,
            "ε: prior weight of the uniform clutter component");
    out.belief.clutter_scale_m      = getf("ResidualModel.AI2ClutterScaleM", 0.12f,
            "a point further than ~this from the box edge is likely clutter");
    out.belief.prior_pos_std        = getf("ResidualModel.AI2PriorPosStd", 0.50f,
            "position prior std (m) on cx,cy");
    out.belief.prior_yaw_std        = getf("ResidualModel.AI2PriorYawStd", 1.57f,
            "orientation prior std (rad) — essentially uninformative");
    out.belief.prior_size_std       = getf("ResidualModel.AI2PriorSizeStd", 0.50f,
            "size prior std (m) on w,d — broad");
    out.belief.process_std_pos_m    = getf("ResidualModel.AI2ProcessStdPosM", 0.02f,
            "Temporal transition (predict): a physical obstacle is RIGID + STATIC, so shape process noise is SMALL and evidence ACCUMULATES across frames…");
    out.belief.process_std_yaw      = getf("ResidualModel.AI2ProcessStdYaw", 0.10f,
            "");
    out.belief.process_std_size_m   = getf("ResidualModel.AI2ProcessStdSizeM", 0.06f,
            "≈table/bottle; a residual envelope refines as more is seen, but");
    out.belief.common_mode_pos_std  = getf("ResidualModel.AI2CommonModePosStd", 0.03f,
            "shared position error (m)");
    out.belief.common_mode_yaw_std  = getf("ResidualModel.AI2CommonModeYawStd", 0.20f,
            "shared orientation error (rad)");
    out.belief.common_mode_size_std = getf("ResidualModel.AI2CommonModeSizeStd", 0.03f,
            "shared size error w,d (m)");
    out.belief.gn_iters             = geti("ResidualModel.AI2GnIters", 5,
            "Gauss-Newton iterations per frame");
    out.belief.min_size             = getf("ResidualModel.MinSize", 0.05f,
            "Geometry (physical floors/ceilings on the footprint extents; a residual can be small (a mug) or large (a wall segment), so the size prior is broad —…");
    out.belief.max_size             = getf("ResidualModel.MaxSize", 4.0f,
            "");
    out.ai2_csv_path                = gets("ResidualModel.AI2CsvPath", "",
            "non-empty → per-cycle belief CSV (state + Σ diag)");
    out.diag_csv_path               = gets("ResidualConcept.DiagCsvPath", "etc/residual_diag.csv",
            "Per-cluster diagnostic CSV (default ON): sweep/specialists/room_poly counts + each cluster's geometry — so wall/subtraction problems are diagnosable…");

    // ── clusterer (perception front-end) ──
    out.cluster.floor_z0         = getf("Clusterer.FloorZ0", 0.06f,
            "floor band at the sensor (tuned for the HIGH helios lidar)");
    out.cluster.floor_slope      = getf("Clusterer.FloorSlope", 0.04f,
            "+ this per metre of horizontal range (≈beam grazing at distance)");
    out.cluster.ceil_z           = getf("Clusterer.CeilZ", 1.80f,
            "drop returns above this (ceiling / overhead)");
    out.cluster.min_vertical_extent_m = getf("Clusterer.MinVerticalExtentM", 0.10f,
            "A residual OBSTACLE must have vertical extent OR be an elevated surface");
    out.cluster.min_top_z_m           = getf("Clusterer.MinTopZM", 0.30f,
            "...an elevated flat sheet with top ≥ this (a tabletop) is a REAL obstacle → KEPT");
    out.cluster.robot_radius_m   = getf("Clusterer.RobotRadiusM", 0.40f,
            "drop returns within this of the robot centre (self-returns)");
    out.cluster.wall_margin_m    = getf("Clusterer.WallMarginM", 0.30f,
            "drop returns within this of the room delimiting polygon boundary");
    out.cluster.bpearl_floor_z0  = getf("Clusterer.BpearlFloorZ0", 0.12f,
            "floor band for bpearl (device-specific; > floor_z0)");
    out.cluster.explain_margin_m = getf("Clusterer.ExplainMarginM", 0.04f,
            "a point within this of a specialist SURFACE is claimed (≈sensor noise");
    out.cluster.explain_sensor_sigma_m = getf("Clusterer.ExplainSensorSigmaM", 0.03f,
            "irreducible LiDAR range noise: the explained boundary is fuzzy by ≥ this");
    out.cluster.explain_fit_margin_m   = getf("Clusterer.ExplainFitMarginM", 0.10f,
            "claim M past each object box edge (0 = model-exact; < inflate_radius_m)");  // under-fit slack (< clearance)
    out.cluster.dbscan_eps_m     = getf("Clusterer.DbscanEpsM", 0.20f,
            "neighbourhood radius (m)");
    out.cluster.dbscan_min_pts   = geti("Clusterer.DbscanMinPts", 5,
            "core-point density");
    out.cluster.dbscan_z_scale   = getf("Clusterer.DbscanZScale", 0.35f,
            "Anisotropic z: z-distance is scaled by this before clustering, so vertically-separated LiDAR layers of ONE object (a pillar's rings, a surface…");
    out.cluster.min_cluster_pts  = geti("Clusterer.MinClusterPts", 12,
            "drop clusters smaller than this (margin leakage / sparse noise)");
    // Navigation fragment-merge gap = robot width + an excess: fragments closer than the robot can pass
    // between are merged into ONE obstacle (coalesces a LiDAR-shattered tabletop). 0 width → merge off.
    out.cluster.merge_gap_m      = getf("Clusterer.RobotWidthM", 0.50f,
            "0 = off") + getf("Clusterer.MergeExcessM", 0.10f,
            "extra bridge beyond the base gap");

    // ── tracker ──
    out.tracker_gate_mahalanobis  = getf("Tracker.GateMahalanobis", 9.0f,
            "~4σ (looser than bottle's 9) — residual centroids are noisy");
    out.tracker_gate_fallback_m   = getf("Tracker.GateFallbackM", 0.35f,
            "");
    out.tracker_birth_frames      = geti("Tracker.BirthFrames", 4,
            "↑ from 6 — a cluster must persist to birth (kills motion-transient births)");
    out.tracker_death_frames      = geti("Tracker.DeathFrames", 30,
            "↓ from 90 — transients clear in ~3 s; real obstacles are re-observed each pass");
    out.tracker_birth_min_sep_m   = getf("Tracker.BirthMinSepM", 0.25f,
            "↑ from 0.10 — residuals are large; suppress near-duplicate births");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.08f,
            "≥ cluster-centroid-vs-fit-centre offset");
    out.tracker_merge_overlap     = getf("Tracker.MergeOverlap", 0.30f,
            "box-IoU merge threshold; 0 disables");
    out.tracker_nll_cost          = getb("Tracker.NllCost", false,
            "");
    out.tracker_death_enabled     = getb("Tracker.DeathEnabled", false,
            "OFF: removal is now the evidence-based occupancy carve, not a");
    out.tracker_expected_visible_range_m = getf("Tracker.ExpectedVisibleRangeM", 6.0f,
            "frame counter (a miss = failed association ≠ evidence of absence) Negative-information death gate: an instance only accrues a death 'miss' when it…");

    // ── occupancy-evidence removal (ray-carving) ──
    out.carve.sensor_sigma_m = getf("Carve.SensorSigmaM", 0.03f,
            "LiDAR range noise σ (m) — surface-localisation blur along the beam");   // LiDAR range noise σ
    out.carve.detection_prob = getf("Carve.DetectionProb", 0.85f,
            "P(a beam through an OCCUPIED box returns from it) — sensor hit rate");   // P(beam through occupied box returns)
    out.carve.clutter_prob   = getf("Carve.ClutterProb", 0.05f,
            "P(a beam through an EMPTY box returns from inside anyway) — spurious rate");   // P(beam through empty box returns inside)
    out.removal_prob          = getf("Carve.RemovalProb", 0.12f,
            "Decision boundary: remove when P(occupied) < removal_prob (i.e. log-odds < log(p/(1−p)))");  // remove when P(occupied) < this
    out.occupancy_logodds_max = getf("Carve.OccupancyLogOddsMax", 4.0f,
            "clamp |log-odds| (±) so evidence stays finite AND recoverable"); // ± clamp on the accumulated log-odds

    // ── per-point reliability covariates ──
    out.range_noise_coeff = getf("ResidualModel.RangeNoiseCoeff", 0.01f,
            "m of extra std per m of range (LiDAR range noise + deprojection)");
    out.ego_motion_lag_s  = getf("ResidualModel.EgoMotionLagS", 0.05f,
            "effective registration lag (s): rotation smear = lag·rot_rate·range");

    // ── ZED dense-depth boost ──
    out.zed_boost_enabled       = getb("ZedBoost.Enabled", true,
            "── ZED dense-depth boost: when a residual box is in the ZED FoV, backproject the depth in its image region and add those dense points to the belief…");
    out.zed_detection_enabled   = getb("ZedBoost.FeedDetection", false,
            "Feed the dense ZED FoV cloud into DETECTION (clustering)");   // dense ZED → clustering
    out.grid_zed_enabled        = getb("ZedBoost.FeedGrid", true,
            "Feed the dense ZED FoV cloud into the OCCUPANCY GRID (as a second sensor after LiDAR)");    // dense ZED → occupancy grid
    out.grid_field_ema_up       = getf("Grid.FieldEmaUp", 1.0f,
            "blend toward a HIGHER risk (instant — never lag a new obstacle)");    // published-field temporal low-pass (rise)
    out.grid_field_ema_down     = getf("Grid.FieldEmaDown", 0.30f,
            "blend toward a LOWER risk (slow — stabilises flicker; ↓ = stickier)");   // ...and fall (slow = stable)
    out.motion_vel0_mps         = getf("Grid.MotionVel0", 0.5f,
            "linear speed (m/s) at which sweep trust halves");    // linear speed (m/s) that halves sweep trust
    out.motion_omega0_rps       = getf("Grid.MotionOmega0", 0.6f,
            "yaw rate (rad/s) at which sweep trust halves");    // yaw rate (rad/s) that halves sweep trust
    out.grid_forget_half_life_s = getf("Grid.ForgetHalfLifeS", 10.0f,
            "half-life of evidence in an UNOBSERVED cell (0 ⇒ never forget)");   // evidence half-life in an UNOBSERVED cell
    out.grid_forget_occupied_only = getb("Grid.ForgetOccupiedOnly", true,
            "...applied to OCCUPANCY evidence only"); // ...decay OCCUPANCY only
    out.grid_forget_visible_only  = getb("Grid.ForgetVisibleOnly", true,
            "...and only where a beam actually REACHED this cycle"); // ...and only where we LOOKED
    out.grid_forget_range_weighted = getb("Grid.ForgetRangeWeighted", true,
            "...and scaled by the precision that failed to confirm"); // ...at the precision we had
    out.grid_forget_can_unlatch   = getb("Grid.ForgetCanUnlatch", false,
            "...and the decay may never un-latch on its own: removal needs evidence of FREENESS, not absence of observation"); // ...and never un-latches by itself
    out.grid_bin_span_m           = getf("Grid.BinSpanM", 2.0f,
            "height covered by the 64 voxel bins ⇒ 3.125 cm voxels. Matched to the ~3 cm registration");  // ⇒ 64 voxels of 3.125 cm
    out.grid_clear_stop_max_m     = getf("Grid.ClearStopMaxM", 0.10f,
            "A/B on the probe: 1337 → 442 releases inside the footprint at 3.125 cm. 0 disables BOTH (textbook VoxelLayer)"); // do not clear close to the hit
    out.grid_collision_band_top_m = getf("Grid.CollisionBandTopM", 0.50f,
            "reported most of the map as unknown. 0 = no discount (the blind cone reads free, as costmap_2d does)"); // free needs the collision band seen
    out.grid_lidar_clearance_m    = getf("Grid.LidarClearanceM", 0.55f,
            "no cell inside this radius of the robot may be cleared"); // nothing inside this radius is cleared
    out.helios_min_range_m        = getf("LidarModel.HeliosMinRangeM", 0.40f,
            "RS-Helios datasheet minimum"); // dead shells: no returns closer
    out.bpearl_min_range_m        = getf("LidarModel.BpearlMinRangeM", 0.10f,
            "RS-Bpearl is a near-field dome");
    out.zed_min_range_m           = getf("LidarModel.ZedMinRangeM", 0.30f,
            "matches ZedBoost.MinDepthM");
    out.helios_beam_spacing_rad   = getf("LidarModel.HeliosBeamSpacingRad", 0.0394f,
            "RS-Helios-32: 70 deg over 31 gaps");  // vertical sampling
    out.bpearl_beam_spacing_rad   = getf("LidarModel.BpearlBeamSpacingRad", 0.0506f,
            "RS-Bpearl-32: ~90 deg dome over 31 gaps");  // interval per device
    out.zed_beam_spacing_rad      = getf("LidarModel.ZedBeamSpacingRad", 0.0068f,
            "dense depth at stride 4 — effectively a pencil");
    out.grid_speckle_min_neighbours = geti("Grid.SpeckleMinNeighbours", 1,
            "0 ⇒ off");   // lone-cell filter (a size
    out.grid_speckle_grace_cycles   = geti("Grid.SpeckleGraceCycles", 10,
            "cycles a lone cell keeps shipping after its last return");   // threshold — see the header)
    out.grid_speckle_min_component_cells = geti("Grid.SpeckleMinComponentCells", 3,
            "isolated-clump filter: cells a connected clump needs; 0 ⇒ off");  // isolated-clump filter
    out.release_csv_path      = gets("Grid.ReleaseCsvPath", "etc/residual_releases.csv",
            "per-released-cell trace; empty = off");
    out.birth_csv_path        = gets("Grid.BirthCsvPath", "etc/residual_births.csv",
            "per-latched-cell trace; empty = off");
    out.sweep_csv_path        = gets("Grid.SweepCsvPath", "etc/residual_sweep.csv",
            "periodic RAW helios sweep dump (room frame); empty = off");
    out.sweep_dump_every_n    = static_cast<int>(getf("Grid.SweepDumpEveryN", 300.0f,
            ""));
    out.cluster_helios_floor_z0 = getf("Clusterer.HeliosFloorZ0", 0.20f,
            "Robot body envelope used by the SENSOR model to discount returns off our own body at integration time");   // helios grazes: unusable near floor
    out.grid_inflate_radius_m   = getf("Grid.InflateRadiusM", 0.0f,
            "C-space inflation of the PUBLISHED hulls. 0 because the controller collides its real footprint against the grid, so any inflation here is added twice");    // 0: controller does exact footprint
    out.grid_self_body_radius_m = getf("Grid.SelfBodyRadiusM", 0.55f,
            "");   // body envelope for the sensor-model term
    out.grid_self_body_sigma_m  = getf("Grid.SelfBodySigmaM", 0.08f,
            "positional uncertainty of the body surface (probit width)");   // its positional uncertainty
    out.grid_pose_precision     = getb("Grid.PosePrecision", true,
            "weight evidence by the localiser's own σ (see OccGridParams)");    // localiser σ as an evidence weight
    out.grid_floor_responsibility = getb("Grid.FloorResponsibility", true,
            "hit weight ×= P(obstacle | z) in the {floor, obstacle} mixture"); // floor in the per-return mixture
    out.grid_floor_return_clears  = getb("Grid.FloorReturnClears", true,
            "a below-band return marks its OWN cell free (breaks the ratchet)"); // a floor return frees its own cell
    out.grid_floor_sigma_min_m    = getf("Grid.FloorSigmaMinM", 0.03f,
            "irreducible sensor range noise (m) — σ floor of the floor model");// sensor range noise (σ floor)
    // ── Stage 1 (2026-08-19): separate the MARKING filter from the CLEARING filter — the costmap_2d rule ──
    out.grid_floor_band_in_grid = getb("Grid.FloorBandInGrid", true,
            "hand device_sweep's near-floor returns to the grid and let the");  // per-device band INTO the grid instead of
                                                                       // deleting the returns before it sees them
    out.grid_zed_infra_clears   = getb("Grid.ZedInfraClears", true,
            "ZED floor/ceiling/wall returns are raytraced for clearing; only");  // ZED floor/ceiling/wall returns still
                                                                       // raytrace; only their MARK is suppressed
    out.grid_cell_dump_every_n  = geti("Grid.CellDumpEveryN", 600,
            "their endpoint MARK is suppressed (mark_mask) Per-cell geometry dump of the published residual set (etc/residual_cells.csv), every N cycles. 0 = off");   // per-cell residual geometry → CSV (0=off)
    // Robust ZED infrastructure subtraction (reuse the floor/ceiling heights; ZED depth-noise band σ0+q·r²).
    out.zed_infra.floor_z0      = out.cluster.floor_z0;
    out.zed_infra.ceil_z        = out.cluster.ceil_z;
    out.zed_infra.wall_margin_m = getf("ZedBoost.InfraWallMarginM", 0.10f,
            "drop returns within this of the room delimiting polygon boundary");
    out.zed_infra.sigma0_m      = getf("ZedBoost.InfraSigma0M", 0.03f,
            "depth-noise floor (m) — near-range + calibration offset");
    out.zed_infra.sigma_quad    = getf("ZedBoost.InfraSigmaQuad", 0.006f,
            "depth-noise range² coeff (m/m²): σ = σ0 + sigma_quad·r² (stereo physics)");   // stereo depth noise ∝ range²
    out.zed_infra.k             = getf("ZedBoost.InfraK", 3.0f,
            "how many σ of band (confidence)");
    out.zed_sigma_m             = getf("ZedBoost.SigmaM", 0.02f,
            "per-ZED-point obs noise std (m) → R (ZED close depth is crisp)");
    out.zed_boost.subsample     = geti("ZedBoost.Subsample", 3,
            "take every Nth pixel in u and v (dense depth → keep the count sane)");
    out.zed_boost.xy_margin_m   = getf("ZedBoost.XyMarginM", 0.10f,
            "keep a backprojected point if within this of the footprint edge");
    out.zed_boost.z_margin_m    = getf("ZedBoost.ZMarginM", 0.10f,
            "...and within this of the [z_min,z_max] band");
    out.zed_boost.min_depth_m   = getf("ZedBoost.MinDepthM", 0.30f,
            "ZED near limit");
    out.zed_boost.max_depth_m   = getf("ZedBoost.MaxDepthM", 5.0f,
            "ZED depth degrades with range² — ignore beyond this");
    out.zed_boost.max_points    = geti("ZedBoost.MaxPoints", 1500,
            "cap per instance (keeps the GN fit bounded)");

    // ── RGB-semantic floor down-weighting (second, uncorrelated cue vs ZED floor phantoms) ──
    out.semantic_floor.enabled          = getb("Semantic.DownweightFloor", false,
            "master flag (config FloorPlane.Enabled). When off the estimator still RUNS");   // master flag (OFF by default)
    out.semantic_floor.floor_suppress   = getf("Semantic.FloorSuppress", 0.60f,
            "max hit-weight reduction for a floor-class NEAR-floor point (weight = 1−this)");
    out.semantic_floor.height_scale_m   = getf("Semantic.HeightScaleM", 0.15f,
            "a floor label's authority decays as exp(−(h/scale)²) over height h above the");
    out.semantic_floor.fresh_half_life_s = getf("Semantic.FreshHalfLifeS", 0.50f,
            "suppression strength halves per this much map age (freshness-as-precision)");
    out.semantic_floor.floor_z0         = out.cluster.floor_z0;   // share the geometric nav band
    out.semantic_floor.floor_slope      = out.cluster.floor_slope;

    // ── data-driven floor plane (grid floor explainer follows an offset/tilted floor) ──
    out.floor_plane.enabled          = getb("FloorPlane.Enabled", false,
            "master flag (config FloorPlane.Enabled). When off the estimator still RUNS");
    out.floor_plane.candidate_band_m = getf("FloorPlane.CandidateBandM", 0.25f,
            "a return is a floor CANDIDATE if within this of the current band top (above");
    out.floor_plane.trim_k           = getf("FloorPlane.TrimK", 2.5f,
            "after each fit, drop candidates whose residual > trim_k·MAD (legs/clutter)");
    out.floor_plane.iters            = geti("FloorPlane.Iters", 2,
            "robust reweighting rounds");
    out.floor_plane.min_candidates   = geti("FloorPlane.MinCandidates", 80,
            "fewer than this ⇒ keep the previous plane (don't fit on noise)");
    out.floor_plane.ema              = getf("FloorPlane.Ema", 0.10f,
            "temporal smoothing toward the new fit (slow → stable; 1 = no smoothing)");
    out.floor_plane.max_offset_m     = getf("FloorPlane.MaxOffsetM", 0.30f,
            "|c| sanity clamp — reject a wild fit (keep previous) beyond this");
    out.floor_plane.max_tilt         = getf("FloorPlane.MaxTilt", 0.15f,
            "|a|,|b| sanity clamp (≈8.5° over 1 m) — reject a wild tilt");
    // Optional override of the ADE20K walkable-ground class set (comma/space-separated ids). Empty ⇒ the built-in
    // default (floor/road/grass/sidewalk/earth/rug/field/sand/path/runway/dirt-track/land). Lets the set be tuned
    // to the deployed segmenter without a rebuild.
    if (const std::string s = gets("Semantic.FloorClassIds", "",
            ""); not s.empty())
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
    out.dissolve_explained_frac = getf("ResidualConcept.DissolveExplainedFrac", 0.60f,
            "Dissolve-on-explained-interior (the fission policy): if a specialist explains ≥ this fraction of an instance's cluster support, invalidate it and let…");

    // ── conservative publish (safe-clearance footprint) ──
    out.publish_extent_sigma_k  = getf("ResidualConcept.PublishExtentSigmaK", 2.0f,
            "grow published w,d by k·√Σ_ww, k·√Σ_dd (occlusion-aware)");
    out.publish_safety_margin_m = getf("ResidualConcept.PublishSafetyMarginM", 0.05f,
            "+ this clearance on EACH footprint side (robot clearance)");

    // ── media / cov ──
    out.lidar_frame_node  = gets("ResidualModel.LidarFrameNode", "helios",
            "DSR node whose frame the raw sweep arrives in");
    out.use_bpearl        = getb("ResidualModel.UseBpearl", true,
            "fuse the low 'bpearl' plane with 'helios' (legs/low obstacles). Toggle off");
    out.rt_cov_add_chain  = getb("ResidualConcept.RtCovAddChain", true,
            "add J·Σ_chain·Jᵀ localization cov to the published RT cov");
    out.unobservable_var  = getf("ResidualConcept.UnobservableVar", 9.87f,
            "≈π² — roll/pitch (rx,ry) are unobservable for a footprint box");

    std::print("residual_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
