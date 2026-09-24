/*
 * cabinet_config.cpp  —  fill CabinetConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in cabinet_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "cabinet_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

CabinetConfig load_cabinet_config(const ConfigLoader& cfg)
{
    CabinetConfig out;


    // The one place this path is written. Relative to the agent's CWD (<agent>/), not to src/ — the same
    // string was right in one place and wrong in the other for a week, and the manifest was inert the whole time.
    static constexpr const char* kManifestPath = "../common/concept_manifest/cabinet.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL — rc::manifest::provenance_ok. `from = "inherited"` means the number
    // arrived by a rename and nobody chose it for THIS object; hood shipped ten such defects in a week with
    // several of them declared, in writing, in its manifest. A note stopped nothing, so this stops the agent.
    if (not rc::manifest::provenance_ok(kManifestPath, "cabinet"))
        std::exit(EXIT_FAILURE);

    // ★NO CLASS-LEVEL z-SPAN TO ADOPT, AND THAT IS A DECLARATION, NOT A GAP. cabinet's manifest states
    // support = "resolved": the belief unit is a (wall, tier) CELL, and each cell's span comes from its own
    // WallTierPrior (base ~[0, 0.87], upper ~[1.40, 2.10]) rather than from any class constant. Declaring one
    // enum here would be a WRONG statement about the object, so rc::manifest::Geometry::z_span() returns an
    // EMPTY band for `resolved` deliberately — a caller that built a band from it would get nothing rather
    // than a plausible-looking floor-referenced guess. band_contains_body() is therefore checked PER CELL by
    // the kitchen model, not once here. Same reasoning applies to bottle (`resolved` per support surface).

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "cabinet_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("CabinetConcept.StateEps", 0.04f,
            "Σ|Δstate| convergence threshold between cycles (m+rad)");
    out.K_stable                 = geti("CabinetConcept.KStable", 30,
            "consecutive converged cycles before model_stable");
    out.detection_alive_max_frames = geti("CabinetConcept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh mask before detection_alive=false");
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "Demote Operating→Degraded→Waiting when the retina's `masks` node stops advancing its mask_frame_id for this long (producer dead/stalled) — don't…");
    out.obs_distance             = getf("CabinetConcept.ObsDistance", 1.8f,
            "d_obs for the epistemic planner");
    out.epistemic_cooldown_cycles= geti("CabinetConcept.EpistemicCooldownCycles", 200,
            "min cycles withdrawn after satisfaction");
    out.cabinet_log_period_frames  = geti("CabinetConcept.CabinetLogPeriodFrames", 30,
            "per-cycle log throttle");
    out.show_dashboard            = getb("CabinetConcept.ShowDashboard", true,
            "Show the GUI (belief strip + the dashboard behind its 'details' button). false => no GUI windows are built at all (headless); the compute feed no-ops…");
    out.verbose_log                = getb("CabinetConcept.VerboseLog", false,
            "false = quiet terminal (only births/merges/removals)");
    out.support_bank_max_points    = geti("CabinetConcept.SupportBankMaxPoints", 4000,
            "cap on the cabinet-owned support-point memory bank");
    out.support_bank_quantization_m= getf("CabinetConcept.SupportBankQuantizationM", 0.02f,
            "support-bank dedup grid (m)");
    out.support_select_radius_margin_m = getf("CabinetConcept.SupportSelectRadiusMarginM", 0.50f,
            "XY margin (m) around the model for support-bank selection");
    out.support_select_height_margin_m = getf("CabinetConcept.SupportSelectHeightMarginM", 0.25f,
            "Z margin (m) around the model for support-bank selection");

    // ─── CabinetModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("CabinetModel.SigmaObs", 0.05f,
            "Forwarded to CabinetModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top face is attributed to the slab (candidate) vs…");
    out.sdf_threshold_for_storage = getf("CabinetModel.SdfThresholdForStorage", 0.08f,
            "On-surface membership for the candidate/residual split in CabinetFitter::observe");

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("CabinetModel.AI2SigmaBaseM", 0.03f,
            "base on-surface obs noise std (m); R = σ² (+ motion_var + …)");
    out.ai2_clutter_frac     = getf("CabinetModel.AI2ClutterFrac", 0.10f,
            "ε: prior weight of the uniform clutter mixture component");
    out.ai2_clutter_scale_m  = getf("CabinetModel.AI2ClutterScaleM", 0.12f,
            "a point further than ~this from every surface is likely clutter");
    out.ai2_prior_size_std   = getf("CabinetModel.AI2PriorSizeStd", 0.30f,
            "broad size prior std (m) on w,h,H");
    out.ai2_process_std_m    = getf("CabinetModel.AI2ProcessStdM", 0.005f,
            "predict process-noise std, length DOFs (m/frame)");
    out.ai2_process_std_yaw  = getf("CabinetModel.AI2ProcessStdYaw", 0.01f,
            "predict process-noise std, yaw (rad/frame)");
    out.ai2_age_nominal_dt_s = getf("CabinetModel.AI2AgeNominalDtS", 0.0f,
            "Stale-belief aging (measurement-age → covariance)");
    out.ai2_common_mode_pos_std  = getf("CabinetModel.AI2CommonModePosStd", 0.03f,
            "shared position error (m); pose-chain cov adds to it");
    out.ai2_common_mode_size_std = getf("CabinetModel.AI2CommonModeSizeStd", 0.02f,
            "shared size error w,h,H (m)");
    out.ai2_common_mode_yaw_std  = getf("CabinetModel.AI2CommonModeYawStd", 0.03f,
            "shared yaw error (rad)");
    out.ai2_common_mode_motion_gain = getf("CabinetModel.AI2CommonModeMotionGain", 3.0f,
            "strong: even slow driving (motion_dotd~0.1) freezes geometry");
    out.ai2_range_noise_lat_per_m = getf("CabinetModel.AI2RangeNoiseLatPerM", 0.02f,
            "lateral deprojection std growth (m per m of range)");
    out.ai2_range_noise_yaw_per_m = getf("CabinetModel.AI2RangeNoiseYawPerM", 0.03f,
            "yaw common-mode std growth (rad per m of range)");
    out.ai2_range_noise_size_per_m = getf("CabinetModel.AI2RangeNoiseSizePerM", 0.08f,
            "SIZE (w,h,H) common-mode std growth (m per m of range): a distant");
    out.ai2_trunc_gate_frac    = getf("CabinetModel.AI2TruncGateFrac", 0.10f,
            "mask can't reshape/inflate a converged cabinet (freezes geometry afar) THRESHOLD (truncation gate): skip the geometric update (predict only) when…");
    out.ai2_gn_iters         = geti("CabinetModel.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_csv_path         = gets("CabinetModel.AI2CsvPath", "",
            "if non-empty, append per-cycle belief (state + Σ diag + mask R) to CSV");
    out.birth_surprise_probe = getb("CabinetModel.BirthSurpriseProbe", false,
            "EXPERIMENTAL birth-surprise probe (read-only): read residual_concept's `grid` node as an unexplained- occupancy (surprise) field, cluster it, and LOG…");
    out.pixel_sigma_over_f     = getf("CabinetModel.PixelSigmaOverF", 0.0015f,
            "σ_px/f → transverse std per m of range");
    out.depth_sigma0_m         = getf("CabinetModel.DepthSigma0M", 0.006f,
            "depth std floor (m)");
    out.depth_sigma_range_coef = getf("CabinetModel.DepthSigmaRangeCoef", 0.004f,
            "depth std growth (m per m² of range)");
    out.model_sigma_m          = getf("CabinetModel.ModelSigmaM", 0.010f,
            "residual model std floor (m)");
    out.footprint_residual     = getb("CabinetModel.FootprintResidual", false,
            "a1′+a2′: weighted 2-D footprint residual + shared depth-affine nuisance (replaces the moment channel; the real grazing-yaw fix). depth_bias/scale_std…");
    out.quotient_chart         = getb("CabinetModel.QuotientChart", false,
            "C2v symmetry-quotient chart (PRECISION_AS_INFORMATION.md Stage 3): optimise the footprint in [s,a₁,a₂] so the 4 box representatives collapse to one…");
    out.erosion_px_std = getf("CabinetModel.ErosionPxStd", 2.0f,
            "shared per-frame depth tilt prior std (m/rad) — the yaw nuisance");
    out.depth_bias_std         = getf("CabinetModel.DepthBiasStd", 0.015f,
            "shared per-frame depth bias prior std (m)");
    out.depth_scale_std        = getf("CabinetModel.DepthScaleStd", 0.010f,
            "shared per-frame depth scale prior std (fraction)");

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("CabinetConcept.RtCovScale", 1.0f,
            "Upload the cabinet pose covariance onto the room→cabinet RT edge (rt_covariance_att, 6×6 SE3), mapped from the belief's full Σ over…");
    out.publish_object_obs            = getb("CabinetConcept.PublishObjectObs", false,
            "CabinetConcept.PublishObjectObs");
    out.object_obs_frame              = gets("CabinetConcept.ObjectObsFrame", "body",
            "CabinetConcept.ObjectObsFrame (localizer base node)");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "χ²₂ gate (~3σ) for a mask↔instance match once it has a cov");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.50f,
            "metric XY gate (m) before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f,
            "Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I)");
    out.tracker_birth_frames     = geti("Tracker.BirthFrames", 8,
            "frames a mask must stay unexplained before spawning a cabinet");
    out.birth_fusion             = getb("Tracker.BirthFusion", false,
            "FUSED BIRTH (EXPERIMENTAL, off by default): let residual-grid SURPRISE MASS under a detection accelerate its birth");
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain", 6.0f,
            "max extra evidence/frame at full corroboration (0 ⇒ baseline)");
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef", 8.0f,
            "residual mass at which corroboration is half-saturated (m/(m+ref))");
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM", 0.50f,
            "window (m) for the residual-mass sample under the detection");
    out.tracker_death_frames     = geti("Tracker.DeathFrames", 300,
            "frames an instance may go unobserved before retirement (large)");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.60f,
            "a birth must be ≥ this (m) from every existing cabinet");
    out.tracker_z_gate_m         = getf("Tracker.ZGateM", 0.60f,
            "association VERTICAL gate: a mask can't bind a track >this in z");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.05f,
            "apart → WALL units (z≈1.7) never fuse with BASE runs (z≈0.35) Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this…");
    out.merge_n_sigma            = getf("Tracker.MergeNSigma", 3.0f,
            "Collinear-run merge (the operator that fuses fragments of one run): acceptance width in joint-σ, and a floor on the along-axis gap tolerance (m)…");
    out.merge_gap_floor_m        = getf("Tracker.MergeGapFloorM", 0.30f,
            "");
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM", 1.0f,
            "Default geometry for a cabinet BORN from a mask (no prior to seed it); the belief refines from here");
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM", 0.6f,
            "");
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM", 0.75f,
            "");
    out.residual_birth_enabled   = getb("Tracker.ResidualBirthEnabled", true,
            "Cluster the pooled model-unexplained (residual) points; a coherent, elongated, separated arm that no believed run covers (e.g. the perpendicular arm…");
    out.residual_birth_frames    = geti("Tracker.ResidualBirthFrames", 4,
            "consecutive cycles a candidate must persist before birth");
    out.residual_birth_match_m   = getf("Tracker.ResidualBirthMatchM", 0.40f,
            "candidate↔candidate match radius across cycles (debounce)");
    out.residual_birth_min_pts   = geti("Tracker.ResidualBirthMinPts", 600,
            "min residual cluster mass to consider");
    out.residual_birth_sep_m     = getf("Tracker.ResidualBirthSepM", 0.60f,
            "min separation of the arm from every existing run");
    out.residual_claim_frac      = getf("Tracker.ResidualClaimFrac", 0.15f,
            "slice claimed by a residual-born run if ≥ this fraction of its");
    out.residual_claim_margin_m  = getf("Tracker.ResidualClaimMarginM", 0.20f,
            "footprint expansion for the claim test");
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f,
            "min YOLO confidence for a ricoh detection to raise attention");
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f,
            "extra angular tolerance on the cabinet's angular half-size");
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM", 1.0f,
            "extra range band (ricoh range is rough/indicative)");

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("CabinetModel.LidarPrecision", 0.0f,
            "per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF");
    out.lidar_bpearl_precision = getf("CabinetModel.LidarBpearlPrecision", 0.0f,
            "Low 'bpearl' LiDAR as a SEPARATE per-device ray-set (own origin, occlusion-aware first-hit; sees the legs the high helios grazes over)");
    out.lidar_robust_c_m     = getf("CabinetModel.LidarRobustCM", 0.05f,
            "Cauchy scale (m): returns this far off the surface fade out");
    out.lidar_select_margin_m = getf("CabinetModel.LidarSelectMarginM", 0.10f,
            "pre-select returns within (birth half-extent + margin), all z up to top");
    out.lidar_coverage_n0     = getf("CabinetModel.LidarCoverageN0", 60.0f,
            "LiDAR ray count for FULL weight; fewer → proportionally down-weighted");
    out.lidar_coverage_ang_power = getf("CabinetModel.LidarCoverageAngPower", 1.0f,
            "Angular-coverage weighting: precision ×= (1−R)^p, R = mean-resultant length of return bearings about the centre. p=0 disables (flat), p=1 = pure…");
    out.max_step_m            = getf("CabinetModel.MaxStepM", 1.0f,
            "THRESHOLD (outlier step guard): a static cabinet cannot physically move this far in one frame, so a GN step whose centre jump exceeds it is an…");
    out.coverage_precision    = getf("CabinetModel.CoveragePrecision", 0.0f,
            "Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a large mask grows to explain it (fixes…");
    out.coverage_robust_c_m   = getf("CabinetModel.CoverageRobustCM", 0.15f,
            "Cauchy scale (m): on-plane points this far outside fade out (stray-point guard)");
    out.free_space_precision  = getf("CabinetModel.FreeSpacePrecision", 0.0f,
            "1/m^2 per through-beam (shrink-only)");
    out.tier_prior_gain      = getf("CabinetModel.TierPriorGain", 1.0f,
            "The shared recursive-Laplace engine sets `prior_mean = state.vec()` at the end of EVERY predict and uses `P0 = Sigma.inverse()` — i.e. after the…");
    out.arrangement_prior_enabled = getb("CabinetModel.ArrangementPrior", true,
            "The only prior t0/t1 ever receive");
    out.arrangement_end_info_max  = getf("CabinetModel.ArrangementEndInfoMax", 400.0f,
            "m⁻² ⇒ σ ≥ 5 cm; the frame may not claim better");
    out.arrangement_stale_ms      = geti("CabinetModel.ArrangementStaleMs", 5000,
            "a frame that stopped publishing stops steering");
    out.object_exclusion_precision = getf("CabinetModel.ObjectExclusionPrecision", 0.0f,
            "1/m^2 per penetrating end (retract-only)");
    out.object_exclusion_margin_m  = getf("CabinetModel.ObjectExclusionMarginM", 0.03f,
            "flush clearance kept off the object boundary");
    out.object_engulf_frac         = getf("CabinetModel.ObjectEngulfFrac", 0.60f,
            "retire when along-wall AND depth AND z overlap all exceed this");
    out.footprint_moment_precision = getf("CabinetModel.FootprintMomentPrecision", 0.0f,
            "Measures (w,h,yaw) from the top-band cloud's 2D inertia tensor and folds it as a linear Gaussian factor — the escape from the clutter-trap that…");
    out.footprint_moment_range_per_m = getf("CabinetModel.FootprintMomentRangePerM", 0.03f,
            "GENTLE range term (m per m of range): shared per-frame moment variance grows as (this·range)²");
    out.fe_baseline_adapt_down       = getf("CabinetModel.FeBaselineAdaptDown", 0.05f,
            "Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the cabinet moved — stays surprising long enough to…");
    out.fe_baseline_adapt_up         = getf("CabinetModel.FeBaselineAdaptUp", 0.005f,
            "");
    out.fe_surprise_smooth           = getf("CabinetModel.FeSurpriseSmooth", 0.10f,
            "");
    out.footprint_moment_motion_gain = getf("CabinetModel.FootprintMomentMotionGain", 0.30f,
            "Ego-motion (motion_dotd) coupling: moment variance grows by (this·motion_dotd)² so a 'going-away/rotation' frame (degraded/split mask) can't reshape…");
    out.orientation_motion_ref       = getf("CabinetModel.OrientationMotionRef", 0.50f,
            "");
    out.obliquity_moment_gain        = getf("CabinetModel.ObliquityMomentGain", 0.0f,
            "Grazing-view yaw stability (CSV rogue-rotation fix)");
    out.footprint_moment_completeness_gain = getf("CabinetModel.FootprintMomentCompletenessGain", 0.0f,
            "footprint_moment_completeness_gain / _min_completeness: forwarded to CabinetBeliefParams — inflate the moment measurement variance as the observed…");
    out.footprint_moment_min_completeness  = getf("CabinetModel.FootprintMomentMinCompleteness", 0.02f,
            "");

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("CabinetModel.ExistenceRemovalEnabled", false,
            "Each cycle carve the LiDAR sweep against the cabinet footprint → occupancy/free-space log-odds; remove when P(occupied) < removal_prob");
    out.existence_removal_prob    = getf("CabinetModel.ExistenceRemovalProb", 0.12f,
            "decision boundary: remove when L < log(p/(1−p))");
    out.existence_frame_correlation = getf("CabinetModel.ExistenceFrameCorrelation", 0.0f,
            "rho: frame-to-frame correlation of MISSES. 0 = independent trials (the historic behaviour and the default); opt in only with a value MEASURED from…");
    out.existence_logodds_max     = getf("CabinetModel.ExistenceLogoddsMax", 4.0f,
            "clamp |L| so evidence stays finite AND recoverable");
    out.existence_detection_prob  = getf("CabinetModel.ExistenceDetectionProb", 0.85f,
            "P(beam through OCCUPIED footprint returns from it)");
    out.existence_clutter_prob    = getf("CabinetModel.ExistenceClutterProb", 0.05f,
            "P(beam through EMPTY footprint returns anyway) — spurious rate");
    out.existence_sensor_sigma_m  = getf("CabinetModel.ExistenceSensorSigmaM", 0.03f,
            "LiDAR range σ (m) for the soft occ/free surface split");
    out.existence_remove_frames   = geti("CabinetModel.ExistenceRemoveFrames", 15,
            "debounce: require the removal decision this many consecutive");
    out.existence_absence_range_ref_m = getf("CabinetModel.ExistenceAbsenceRangeRefM", 2.5f,
            "range (m) below which absence is trusted at full weight");
    out.existence_absence_range_power = getf("CabinetModel.ExistenceAbsenceRangePower", 2.0f,
            "decay exponent (2 ≈ angular-area ∝ 1/range²); 0 disables");
    out.existence_verify_surprise     = getf("CabinetModel.ExistenceVerifySurprise", 20.0f,
            "decayed go-verify surprise (un-resolvable absence) above which");
    out.existence_verify_gain         = getf("CabinetModel.ExistenceVerifyGain", 5.0f,
            "epistemic gain (nats) a wants_verification cabinet gets, so the");
    out.wall_precision                = getf("CabinetModel.WallPrecision", 400.0f,
            "1/m^2 at zero gap (~1/sigma^2 with sigma=5 cm)");
    out.wall_reach_m                  = getf("CabinetModel.WallReachM", 0.35f,
            "gap scale over which the flush hypothesis loses its weight");
    out.wall_sigma_m                  = getf("CabinetModel.WallSigmaM", 0.02f,
            "room-model wall position uncertainty (m)");
    out.wall_parallel_precision       = getf("CabinetModel.WallParallelPrecision", 200.0f,
            "on sin(angle between run axis and wall) — 0 = OFF");
    out.room_axis_precision           = getf("CabinetModel.RoomAxisPrecision", 300.0f,
            "1/rad^2 on the yaw→nearest-axis residual");
    out.room_axis_capture_rad         = getf("CabinetModel.RoomAxisCaptureRad", 0.0f,
            "|Δyaw| beyond which the pull is released (0 ⇒ always on)");
    out.seed_room_axis_snap           = getb("CabinetModel.SeedRoomAxisSnap", true,
            "Birth on the dominant room axis instead of the raw PCA diagonal — stops an L-shaped corner mask from birthing one oblique box spanning both walls…");
    out.kitchen_model                 = getb("CabinetModel.KitchenModel", false,
            "Wall-split: attribute each mask point to its nearest room wall; points flush to a DIFFERENT wall than the run they feed are pulled out of the fit (→…");
    out.ceiling_height_m              = getf("CabinetModel.CeilingHeightM", 2.6f,
            "z-domain upper bound for wall-run tops (H_room)");
    out.kitchen_motion_cm_gain        = getf("CabinetModel.KitchenMotionCmGain", 0.30f,
            "position shared-error std per (m/s) of motion (chair MotionCmPosGain)");
    out.kitchen_ang_lever_m           = getf("CabinetModel.KitchenAngLeverM", 2.0f,
            "rad/s → m/s lever for camera rotation (chair AI2AngLeverM)");
    out.kitchen_periph_ref            = getf("CabinetModel.KitchenPeriphRef", 0.50f,
            "centroid radius (focal-norm) at which periphery saturates (chair AI2PeriphRef)");
    out.kitchen_lidar_existence       = getb("CabinetModel.KitchenLidarExistence", false,
            "Kitchen RETIREMENT channel: LiDAR evidence of absence on the BORN cells");
    out.kitchen_cells_csv_path        = gets("CabinetModel.KitchenCellsCsvPath", "",
            "Per-cycle CSV of the kitchen cells (state + existence + the absence evidence that drives retirement)");
    out.counter_evidence_enabled      = getb("CabinetModel.CounterEvidence", true,
            "ingest 'counter'/'countertop' masks as top-face run evidence");
    out.lshape_split_enabled          = getb("CabinetModel.LShapeSplitEnabled", true,
            "");
    out.lshape_min_arm_pts            = geti("CabinetModel.LShapeMinArmPts", 500,
            "an arm (and the peeling residue) must exceed this to split");
    out.lshape_bin_m                  = getf("CabinetModel.LShapeBinM", 0.15f,
            "histogram bin for locating each arm's line");
    out.lshape_arm_halfwidth_m        = getf("CabinetModel.LShapeArmHalfwidthM", 0.45f,
            "half-width of an arm's footprint band (≈ half a carcass depth)");
    out.extent_precision              = getf("CabinetModel.ExtentPrecision", 800.0f,
            "1/m^2 on the end-containment residuals");
    out.existence_lidar_absence       = getb("CabinetModel.ExistenceLidarAbsence", false,
            "controller drives to a resolving ZED view (confirm-or-remove) LiDAR removal reliability: the model's top slab is a SOLID band, but a real tabletop is…");

    std::print("cabinet_concept: configuration loaded.\n");
    // Peripheral (ricoh) existence confirmation — see cabinet_config.h. OFF by default.
    out.ricoh_confirm_enabled        = getb("CabinetConcept.RicohConfirmEnabled", out.ricoh_confirm_enabled,
            "CabinetConcept.RicohConfirmEnabled");
    out.ricoh_confirm_detection_prob = getf("CabinetConcept.RicohConfirmDetectionProb", out.ricoh_confirm_detection_prob,
            "CabinetConcept.RicohConfirmDetectionProb");
    out.ricoh_confirm_clutter_prob   = getf("CabinetConcept.RicohConfirmClutterProb", out.ricoh_confirm_clutter_prob,
            "CabinetConcept.RicohConfirmClutterProb");

    return out;
}

}  // namespace rc
