/*
 * hood_config.cpp  —  fill HoodConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in hood_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "hood_config.h"

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>
#include <string>

#include <genericworker.h>   // ConfigLoader

namespace rc {

HoodConfig load_hood_config(const ConfigLoader& cfg)
{
    HoodConfig out;

    // The one place this path is written. It is relative to the agent's CWD (<agent>/), not to src/.
    static constexpr const char* kManifestPath = "../common/concept_manifest/hood.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL. The manifest has carried `from = measured|fitted|nominal|inherited`
    // since 2026-08-11, with a note saying "inherited is worse than absent — an absent value gets asked
    // about, an inherited one gets trusted". A note stopped nothing: hood_concept shipped TEN cloned defects
    // that week, several of them DECLARED as inherited, in writing, in this very file. The declaration is
    // now the enforcement — see rc::manifest::provenance_ok. Refusing to start is the whole point: a comment
    // cannot fail a build, and everything that only warned was fixed around rather than fixed.
    if (not rc::manifest::provenance_ok(kManifestPath, "hood"))
        std::exit(EXIT_FAILURE);

    // Loaded ONCE and handed to every resolve() below — the manifest is now read for values, not just
    // cross-checked. An unreadable manifest leaves `man` empty, so every resolve() falls through to the
    // config/agent default exactly as before, and load_geometry() prints the loud NOT LOADED banner.
    ConfigLoader man;
    try { man.load(kManifestPath); } catch (...) {}

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

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("HoodConcept.StateEps",               0.04f);
    out.K_stable                 = geti("HoodConcept.KStable",                30);
    out.detection_alive_max_frames = geti("HoodConcept.DetectionAliveMaxFrames", 40);
    out.matched_frames_before_aging = geti("HoodConcept.MatchedFramesBeforeAging", 5);
    out.central_region_frac      = getf("HoodConcept.CentralRegionFrac",     0.25f);
    out.epistemic_cooldown_cycles= geti("HoodConcept.EpistemicCooldownCycles", 200);
    out.hood_log_period_frames  = geti("HoodConcept.HoodLogPeriodFrames",   30);
    out.support_bank_max_points    = geti("HoodConcept.SupportBankMaxPoints",     4000);
    out.support_bank_quantization_m= getf("HoodConcept.SupportBankQuantizationM", 0.02f);
    out.support_select_radius_margin_m = getf("HoodConcept.SupportSelectRadiusMarginM", 0.50f);
    out.support_select_height_margin_m = getf("HoodConcept.SupportSelectHeightMarginM", 0.25f);

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.show_dashboard           = getb("HoodConcept.ShowDashboard",          true);
    out.shape_eval_period        = geti("HoodConcept.ShapeEvalPeriod",        30);
    out.shape_eval_min_points    = geti("HoodConcept.ShapeEvalMinPoints",     300);
    out.shape_evidence_clamp     = getf("HoodConcept.ShapeEvidenceClamp",     8.0f);
    out.dump_cloud_path          = gets("HoodConcept.DumpCloudPath",          "");

    // ─── HoodModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("HoodModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("HoodModel.SdfThresholdForStorage", 0.08f);

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("HoodModel.AI2SigmaBaseM",       0.03f);
    out.ai2_clutter_frac     = getf("HoodModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("HoodModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("HoodModel.AI2PriorSizeStd",     0.30f);
    // ★THE MANIFEST IS AUTHORITATIVE FOR WHAT A HOOD *IS*, not only for how it hangs. Every one of these was
    // a cloned refrigerator fact that reached runtime and did damage: the footprint mean was 0.60 (a fridge is
    // square in plan) so a correctly-fitted 0.90x0.50 hood scored 0.027 on its own shape prior; the height was
    // 1.90 — the fridge's — stated in four places with three values, and the one that WON was the parent's.
    // rc::manifest::resolve applies one precedence everywhere: MANIFEST (the world fact) → config.toml only as
    // an explicit override, which PRINTS when it contradicts → the agent default if neither speaks.
    out.ai2_prior_footprint_m   = rc::manifest::resolve(cfg, "HoodModel.AI2PriorFootprintM",
                                      man, "prior.footprint.mean_m", 0.60f, "hood footprint mean");
    out.ai2_prior_depth_m       = rc::manifest::resolve(cfg, "HoodModel.AI2PriorDepthM",
                                      man, "prior.depth.mean_m", 0.50f, "hood depth mean");
    out.ai2_prior_footprint_std = rc::manifest::resolve(cfg, "HoodModel.AI2PriorFootprintStd",
                                      man, "prior.footprint.std_m", 0.08f, "hood footprint std");
    out.ai2_prior_height_m      = rc::manifest::resolve(cfg, "HoodModel.AI2PriorHeightM",
                                      man, "prior.height.mean_m", 2.05f, "hood height mean");
    out.vertical_extent_m = getf("HoodModel.VerticalExtentM", 0.50f);

    // ── STAGE 2B: the manifest is AUTHORITATIVE for geometry ──────────────────────────────────────
    // `support` states the vertical anchoring ONCE, as a fact about the object, and the span is derived
    // from it — instead of four sites (SDF, point-admission band, NBV Target, LiDAR carve) each restating
    // the same assumption in their own arithmetic, which is how the cloned floor-anchored box survived.
    // Precedence is deliberate: MANIFEST first (the world fact), config.toml only as an explicit override,
    // and the agent's own default only if neither speaks — the reverse of the order that let hood run on
    // the refrigerator's height while two corrected files said otherwise.
    // Fallback if the manifest cannot be read: a hood HANGS. Stated here as this agent's own claim about
    // its object, so the span below is still derived from an anchoring rather than assumed to be the floor.
    auto man_support = rc::manifest::Support::hangs;
    {
        const auto g = rc::manifest::load_geometry(kManifestPath, "hood");
        if (g.valid)
        {
            float z0 = 0.0f, z1 = 0.0f; g.z_span(z0, z1);
            const float man_extent = z1 - z0;
            if (not cfg.exists("HoodModel.VerticalExtentM")) out.vertical_extent_m = man_extent;
            else if (std::abs(out.vertical_extent_m - man_extent) > 1e-3f)
                std::print("[manifest] hood OVERRIDE VerticalExtentM: config={:.3f} manifest={:.3f} — the "
                           "config wins, but a world fact is being contradicted; say why in the manifest\n",
                           out.vertical_extent_m, man_extent);
            if (not cfg.exists("HoodModel.AI2PriorHeightM")) out.ai2_prior_height_m = z1;
            else if (std::abs(out.ai2_prior_height_m - z1) > 1e-3f)
                std::print("[manifest] hood OVERRIDE AI2PriorHeightM: config={:.3f} manifest={:.3f}\n",
                           out.ai2_prior_height_m, z1);
            // ★A hood that declares floor_anchored is a WRONG STATEMENT, not a missing one — which is the
            // whole point of naming the support. Refuse to be quiet about it.
            if (g.support == rc::manifest::Support::floor_anchored)
                std::print("[manifest] ★hood declares support=floor_anchored — a hood HANGS. This is the "
                           "exact inheritance that cost p_detect=0 at every range; fix the manifest.\n");
            man_support = g.support;
        }
    }
    // ★THE BODY'S VERTICAL SPAN, RESOLVED ONCE. Derived through the SAME rc::manifest::z_span() the
    // manifest uses, from the values precedence has just settled — so a config override of the height or
    // the extent moves the span with it, and no site re-states the anchoring in its own arithmetic.
    //
    // This exists because a SIXTH floor-anchored site survived the scaffold pass and was found live on
    // 2026-08-11: the LiDAR range channel pre-selected returns in z ∈ [−margin, BirthHeightM + margin] —
    // the fridge's floor-referenced band. For a hood that is [−0.10, 0.85] m against a body at
    // [1.79, 2.29] m: THE TWO DO NOT OVERLAP. Measured every cycle in etc/ai2_log.csv — 109 returns
    // selected as "on the hood", mean SDF residual 1.33 m, selected z spanning 0.19–0.78 while H = 2.285.
    // The floor and the worktop were being fed to the fit as hood surface, every real hood return was
    // filtered out before the factor could see it, and the channel still reported FULL coverage (109 rays
    // against LidarCoverageN0 = 60). The geometric fit was mask-only without saying so — which is what left
    // yaw 9° off the wall (80.9° vs ~90°, std 8.5°) at obliquity_cos 0.29, and with it the visible tilt and
    // the corner in the wall.
    {
        rc::manifest::Geometry span;
        span.support  = man_support;
        span.z_top_m  = out.ai2_prior_height_m;
        span.extent_m = out.vertical_extent_m;
        span.z_span(out.body_z0_m, out.body_z1_m);
        std::print("[manifest] hood body span = [{:.2f},{:.2f}] m (support={})\n",
                   out.body_z0_m, out.body_z1_m, rc::manifest::support_name(man_support));

        // ★EVERY DERIVED BAND IS CHECKED AGAINST THE BODY, HERE, BEFORE A SINGLE FRAME ARRIVES. Each of
        // these was independently re-derived from "the height" by the code that owns it, and each was
        // floor-referenced by inheritance. The LiDAR selection band came out DISJOINT from the body and
        // nothing said so for days. Two numbers, one subtraction — there is no excuse for finding this in a
        // log. See rc::manifest::band_contains_body.
        rc::manifest::Geometry decl;
        decl.support = man_support; decl.z_top_m = out.ai2_prior_height_m;
        decl.extent_m = out.vertical_extent_m; decl.valid = true;
        const float m = out.lidar_select_margin_m;
        bool bands_ok = true;
        bands_ok &= rc::manifest::band_contains_body("hood ", "lidar_select",
                        out.body_z0_m - m, out.body_z1_m + m, decl);
        bands_ok &= rc::manifest::band_contains_body("hood ", "existence_carve",
                        out.body_z0_m, out.body_z1_m, decl);
        bands_ok &= rc::manifest::band_contains_body("hood ", "point_ownership",
                        out.body_z0_m - out.support_select_height_margin_m,
                        out.body_z1_m + out.support_select_height_margin_m, decl);
        if (bands_ok)
            std::print("[manifest] hood ✓ every derived z-band contains the declared body\n");
    }
    out.ai2_prior_height_std    = getf("HoodModel.AI2PriorHeightStd",    0.50f);
    out.ai2_depth_unobs_precision = getf("HoodModel.AI2DepthUnobsPrecision", 1500.0f);
    out.ai2_depth_obs_band_m      = getf("HoodModel.AI2DepthObsBandM",       0.10f);
    out.ai2_top_no_float_precision = getf("HoodModel.AI2TopNoFloatPrecision", 10000.0f);
    out.ai2_top_no_float_margin_m  = getf("HoodModel.AI2TopNoFloatMarginM",   0.02f);
    out.ai2_top_overseg_sigma_per_m = getf("HoodModel.AI2TopOversegSigmaPerM", 2.0f);
    out.ai2_wall_precision          = getf("HoodModel.AI2WallPrecision",         400.0f);
    out.ai2_wall_parallel_precision = getf("HoodModel.AI2WallParallelPrecision", 200.0f);
    out.ai2_wall_reach_m            = getf("HoodModel.AI2WallReachM",             0.15f);
    out.ai2_door_clearance_gain     = getf("HoodModel.AI2DoorClearanceGain",       3.0f);
    // The envelope too — this is the block that carried the REFRIGERATOR's 2296-row approach/retreat tour
    // (0.0 / 1.32 / 0.26) into hood's config as if it were a measurement of a hood. Declared here, so the
    // manifest and the runtime cannot say different things again.
    out.detect_min_fill = rc::manifest::resolve(cfg, "HoodModel.DetectMinFill",
                              man, "detector.envelope.min_fill", 0.10f, "hood detect min_fill");
    out.detect_max_fill = rc::manifest::resolve(cfg, "HoodModel.DetectMaxFill",
                              man, "detector.envelope.max_fill", 0.60f, "hood detect max_fill");
    out.detect_soft     = rc::manifest::resolve(cfg, "HoodModel.DetectSoft",
                              man, "detector.envelope.soft", 0.06f, "hood detect soft");
    out.ai2_volatility_infer        = getb("HoodModel.AI2VolatilityInfer",        false);
    out.ai2_volatility_lr           = getf("HoodModel.AI2VolatilityLr",           0.02f);
    out.ai2_volatility_sigma        = getf("HoodModel.AI2VolatilitySigma",        2.0f);
    out.ai2_wall_explain_frac       = getf("HoodModel.AI2WallExplainFrac",        0.25f);
    out.ai2_wall_explain_sigma_m    = getf("HoodModel.AI2WallExplainSigmaM",      0.05f);
    out.ai2_wall_flush_prior        = rc::manifest::resolve(cfg, "HoodModel.AI2WallFlushPrior",
                                          man, "prior.attachment.flush_prior", 1.0f, "hood flush prior");
    out.ai2_wall_no_cross_precision = getf("HoodModel.WallNoCrossPrecision",     2000.0f);
    out.ai2_wall_no_cross_margin_m  = getf("HoodModel.WallNoCrossMarginM",       0.0f);
    out.ai2_process_std_m    = getf("HoodModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("HoodModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_age_nominal_dt_s = getf("HoodModel.AI2AgeNominalDtS",    0.0f);
    out.ai2_common_mode_pos_std  = getf("HoodModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("HoodModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("HoodModel.AI2CommonModeYawStd",  0.03f);
    out.motion_cm_pos_gain       = getf("HoodModel.MotionCmPosGain",      0.10f);
    out.motion_cm_size_gain      = getf("HoodModel.MotionCmSizeGain",     0.20f);
    out.motion_cm_yaw_gain       = getf("HoodModel.MotionCmYawGain",      0.12f);
    out.ai2_ang_lever_m           = getf("HoodModel.AI2AngLeverM",           2.0f);
    out.ai2_periph_ref            = getf("HoodModel.AI2PeriphRef",           0.50f);
    out.ai2_motion_ref_mps        = getf("HoodModel.AI2MotionRefMps",        0.60f);
    out.ai2_motion_confirm_only   = getb("HoodModel.AI2MotionConfirmOnly",   true);
    out.ai2_still_lin_mps         = getf("HoodModel.AI2StillLinMps",         0.05f);
    out.ai2_still_ang_radps       = getf("HoodModel.AI2StillAngRadps",       0.10f);
    out.ai2_still_dotd            = getf("HoodModel.AI2StillDotd",           0.05f);
    out.ai2_moving_update_center_radius = getf("HoodModel.AI2MovingUpdateCenterRadius", 0.35f);
    out.ai2_range_noise_lat_per_m = getf("HoodModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("HoodModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_range_noise_size_per_m = getf("HoodModel.AI2RangeNoiseSizePerM", 0.08f);
    out.ai2_trunc_gate_frac    = getf("HoodModel.AI2TruncGateFrac",   0.10f);
    out.ai2_gn_iters         = geti("HoodModel.AI2GnIters",          4);
    out.ai2_csv_path         = gets("HoodModel.AI2CsvPath",          "");
    out.birth_surprise_probe = getb("HoodModel.BirthSurpriseProbe",  false);
    out.pixel_sigma_over_f     = getf("HoodModel.PixelSigmaOverF",       0.0015f);
    out.depth_sigma0_m         = getf("HoodModel.DepthSigma0M",          0.006f);
    out.depth_sigma_range_coef = getf("HoodModel.DepthSigmaRangeCoef",   0.004f);
    out.model_sigma_m          = getf("HoodModel.ModelSigmaM",           0.010f);
    out.footprint_residual     = getb("HoodModel.FootprintResidual",     false);
    out.quotient_chart         = getb("HoodModel.QuotientChart",          false);
    out.depth_tilt_std         = getf("HoodModel.DepthTiltStd",          0.020f);
    out.depth_bias_std         = getf("HoodModel.DepthBiasStd",          0.015f);
    out.depth_scale_std        = getf("HoodModel.DepthScaleStd",         0.010f);

    // ─── "Is this really a fridge?" plausibility filter + soft singleton ───────
    out.fridge_filter_enabled   = getb("HoodConcept.FridgeFilterEnabled",   true);
    out.plaus_aspect_scale      = getf("HoodModel.AspectScale",             0.15f);
    out.plaus_size_scale        = getf("HoodModel.SizeScale",               0.15f);
    out.plaus_alt_size_scale    = getf("HoodModel.AltSizeScale",            0.60f);
    out.plaus_height_min        = getf("HoodModel.HeightPlausibleMin",      1.20f);
    out.plaus_height_soft       = getf("HoodModel.HeightSoft",              0.15f);
    out.plaus_fe_ref            = getf("HoodModel.FeRef",                    2.0f);
    out.plaus_fe_scale          = getf("HoodModel.FeScale",                 1.0f);
    out.plaus_clamp             = getf("HoodModel.PlausClamp",              8.0f);
    out.plaus_height_prior_gain = getf("HoodModel.PlausHeightPriorGain",    2000.0f);
    out.plaus_to_existence_gain = getf("HoodModel.PlausToExistenceGain",    1.5f);
    out.singleton_inhibition    = getf("HoodModel.SingletonInhibition",     1.0f);
    out.fridge_filter_log       = getb("HoodConcept.FridgeFilterLog",       false);

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("HoodConcept.RtCovScale",           1.0f);
    out.publish_object_obs            = getb("HoodConcept.PublishObjectObs",   false);
    out.object_obs_frame              = gets("HoodConcept.ObjectObsFrame",     "body");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.50f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.birth_fusion             = getb("Tracker.BirthFusion",       false);
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain",   6.0f);
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef",8.0f);
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM",0.50f);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.60f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.05f);
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM",      1.0f);
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM",      0.6f);
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM",     0.75f);
    // Birth fragment: keep the probation burst and admit the birth on it (see hood_config.h).
    out.birth_frag_enabled       = getb("Tracker.BirthFragment",          true);
    out.birth_frag_cell_m       = getf("Tracker.BirthFragmentVoxelM",    0.03f);
    out.birth_frag_max_pts       = geti("Tracker.BirthFragmentMaxPts",    20000);
    out.birth_frag_delta_ms      = static_cast<std::uint64_t>(
                                       std::max(0, geti("Tracker.BirthFragmentDeltaMs", 4000)));
    out.birth_admit_plausibility = getf("Tracker.BirthAdmitPlausibility", 0.35f);
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f);
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f);
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM",  1.0f);

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("HoodModel.LidarPrecision",      0.0f);
    out.lidar_bpearl_precision = getf("HoodModel.LidarBpearlPrecision", 0.0f);
    out.lidar_robust_c_m     = getf("HoodModel.LidarRobustCM",       0.05f);
    out.lidar_select_margin_m = getf("HoodModel.LidarSelectMarginM", 0.10f);
    out.lidar_coverage_n0     = getf("HoodModel.LidarCoverageN0",     60.0f);
    out.lidar_coverage_ang_power = getf("HoodModel.LidarCoverageAngPower", 1.0f);
    out.max_step_m            = getf("HoodModel.MaxStepM",            1.0f);
    out.coverage_precision    = getf("HoodModel.CoveragePrecision",  0.0f);
    out.coverage_robust_c_m   = getf("HoodModel.CoverageRobustCM",   0.15f);
    out.free_space_precision  = getf("HoodModel.FreeSpacePrecision", 0.0f);
    out.footprint_moment_precision = getf("HoodModel.FootprintMomentPrecision", 0.0f);
    out.footprint_moment_range_per_m = getf("HoodModel.FootprintMomentRangePerM", 0.03f);
    out.fe_baseline_adapt_down       = getf("HoodModel.FeBaselineAdaptDown", 0.05f);
    out.fe_baseline_adapt_up         = getf("HoodModel.FeBaselineAdaptUp",   0.005f);
    out.fe_surprise_smooth           = getf("HoodModel.FeSurpriseSmooth",    0.10f);
    out.footprint_moment_motion_gain = getf("HoodModel.FootprintMomentMotionGain", 0.30f);
    out.orientation_motion_ref       = getf("HoodModel.OrientationMotionRef", 0.50f);

    // ─── Appearance-based FRONT (door) detection + yaw resolver ────────────────
    out.front_detect_enabled   = getb("HoodConcept.FrontDetectEnabled",   true);
    out.front_min_face_area_px = getf("HoodConcept.FrontMinFaceAreaPx",    900.0f);
    out.front_min_confidence   = getf("HoodConcept.FrontMinConfidence",    0.10f);
    out.front_log              = getb("HoodConcept.FrontLog",              false);
    out.obliquity_moment_gain        = getf("HoodModel.ObliquityMomentGain", 0.0f);
    out.footprint_moment_completeness_gain = getf("HoodModel.FootprintMomentCompletenessGain", 0.0f);
    out.footprint_moment_min_completeness  = getf("HoodModel.FootprintMomentMinCompleteness",  0.02f);

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("HoodModel.ExistenceRemovalEnabled", false);
    out.existence_removal_prob    = getf("HoodModel.ExistenceRemovalProb",    0.12f);
    out.existence_frame_correlation = getf("HoodModel.ExistenceFrameCorrelation", 0.0f);
    out.existence_logodds_max     = getf("HoodModel.ExistenceLogoddsMax",     4.0f);
    // Through the manifest like every other world fact: this is a measured sensor rate for THIS object, and
    // the 0.85 default is the refrigerator's. See [sensing.lidar] in hood.concept.toml.
    out.existence_detection_prob  = rc::manifest::resolve(cfg, "HoodModel.ExistenceDetectionProb",
                                        man, "sensing.lidar.detection_prob", 0.85f, "hood lidar P(return|present)");
    out.existence_clutter_prob    = getf("HoodModel.ExistenceClutterProb",    0.05f);
    out.existence_sensor_sigma_m  = getf("HoodModel.ExistenceSensorSigmaM",   0.03f);
    out.existence_remove_frames   = geti("HoodModel.ExistenceRemoveFrames",   15);
    out.existence_absence_range_ref_m = getf("HoodModel.ExistenceAbsenceRangeRefM", 2.5f);
    out.existence_absence_range_power = getf("HoodModel.ExistenceAbsenceRangePower", 2.0f);
    out.existence_verify_surprise     = getf("HoodModel.ExistenceVerifySurprise",   20.0f);
    out.verify_surprise_smooth        = getf("HoodModel.VerifySurpriseSmooth",       0.10f);
    out.existence_verify_gain         = getf("HoodModel.ExistenceVerifyGain",       5.0f);

    std::print("hood_concept: configuration loaded.\n");
    return out;
}

// ─── Concept-manifest cross-check (declarative-priors experiment, step 1½) ────────────────────────
//
// common/concept_manifest/<concept>.concept.toml declares WHAT a hood IS — its priors as world facts,
// separate from the lifecycle knobs. It is not authoritative yet. This compares it against the priors the live
// etc/config.toml actually produced, so we learn whether a manifest can reproduce the running agent BEFORE
// anything is generated from it. Startup-only, read-only: it never changes a value, it only reports.
//
// A DIFFERS line is a finding, not a failure — it means the manifest and the running config disagree about a
// world fact, and one of them is wrong. A MISSING line means the manifest does not yet describe that prior.
bool verify_hood_manifest(const HoodConfig& out, const std::string& path)
{
    ConfigLoader man;
    try { man.load(path); }
    catch (...) { std::print("[manifest] not loaded ({}) — cross-check skipped\n", path); return false; }

    int agree = 0, differ = 0, missing = 0;
    const auto chk = [&](const char* key, float live, const char* what) {
        if (not man.exists(key)) { ++missing;
            std::print("[manifest] MISSING  {:<42} live={:<10.4g} ({})\n", key, live, what); return; }
        const float m = static_cast<float>(man.get<double>(key));
        const float tol = 1e-4f * std::max(1.0f, std::abs(live));
        if (std::abs(m - live) <= tol) { ++agree; }
        else { ++differ;
            std::print("[manifest] DIFFERS  {:<42} manifest={:<10.4g} live={:<10.4g} ({})\n", key, m, live, what); }
    };

    chk("prior.footprint.mean_m",              out.ai2_prior_footprint_m,      "fridge footprint mean");
    chk("prior.footprint.std_m",               out.ai2_prior_footprint_std,    "footprint prior std");
    chk("prior.height.mean_m",                 out.ai2_prior_height_m,         "height anchor mean");
    chk("prior.height.std_m",                  out.ai2_prior_height_std,       "height anchor std");
    chk("prior.depth_observability.precision", out.ai2_depth_unobs_precision,  "depth-unobserved precision");
    chk("prior.depth_observability.observed_band_m", out.ai2_depth_obs_band_m, "depth observed band");
    chk("prior.top.precision",                 out.ai2_top_no_float_precision, "top no-float anchor");
    chk("prior.top.margin_m",                  out.ai2_top_no_float_margin_m,  "top anchor margin");
    chk("prior.attachment.precision",          out.ai2_wall_precision,         "wall flush");
    chk("prior.attachment.parallel_precision", out.ai2_wall_parallel_precision,"wall parallel");
    chk("prior.attachment.reach_m",            out.ai2_wall_reach_m,           "flush reach");
    chk("prior.attachment.no_cross_precision", out.ai2_wall_no_cross_precision,"wall no-cross");
    chk("prior.identity.aspect_scale",         out.plaus_aspect_scale,         "identity aspect");
    chk("prior.identity.size_scale",           out.plaus_size_scale,           "identity size");
    chk("prior.identity.alt_size_scale",       out.plaus_alt_size_scale,       "identity alternative");
    chk("prior.identity.height_min_m",         out.plaus_height_min,           "identity height centre");
    chk("prior.identity.height_soft_m",        out.plaus_height_soft,          "identity height softness");
    chk("prior.clearance.gain_nats",           out.ai2_door_clearance_gain,    "door clearance prior");
    chk("prior.explaining_away.weight",        out.ai2_wall_explain_frac,      "wall explain-away weight");
    chk("prior.explaining_away.sigma_m",       out.ai2_wall_explain_sigma_m,   "wall explain-away sigma");
    chk("cue.door_seam.min_face_area_px",      out.front_min_face_area_px,     "door cue min face area");
    chk("cue.door_seam.min_confidence",        out.front_min_confidence,       "door cue min confidence");

    std::print("[manifest] {} — {} agree, {} DIFFER, {} missing\n",
               (differ == 0 and missing == 0) ? "reproduces the live config" : "does NOT yet reproduce the live config",
               agree, differ, missing);
    return differ == 0 and missing == 0;
}

}  // namespace rc
