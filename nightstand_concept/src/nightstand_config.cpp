/*
 * nightstand_config.cpp  —  fill NightstandConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in nightstand_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "nightstand_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)
#include <string>

#include <genericworker.h>   // ConfigLoader

namespace rc {

NightstandConfig load_nightstand_config(const ConfigLoader& cfg)
{
    NightstandConfig out;

    // The one place this path is written. It is relative to the agent's CWD (<agent>/), not to src/.
    static constexpr const char* kManifestPath = "../common/concept_manifest/nightstand.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL. The manifest has carried `from = measured|fitted|nominal|inherited`
    // since 2026-08-11, with a note saying "inherited is worse than absent — an absent value gets asked
    // about, an inherited one gets trusted". A note stopped nothing: hood_concept shipped TEN cloned defects
    // that week, several of them DECLARED as inherited, in writing, in this very file. The declaration is
    // now the enforcement — see rc::manifest::provenance_ok. Refusing to start is the whole point: a comment
    // cannot fail a build, and everything that only warned was fixed around rather than fixed.
    if (not rc::manifest::provenance_ok(kManifestPath, "nightstand"))
        std::exit(EXIT_FAILURE);

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "nightstand_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("NightstandConcept.StateEps", 0.04f,
            "Σ|Δstate| convergence threshold between cycles (m+rad)");
    out.K_stable                 = geti("NightstandConcept.KStable", 30,
            "consecutive converged cycles before model_stable");
    out.detection_alive_max_frames = geti("NightstandConcept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh mask before detection_alive=false");
    out.matched_frames_before_aging = geti("NightstandConcept.MatchedFramesBeforeAging", 5,
            "a barely-born belief (fewer matched frames than this) is NOT");
    out.central_region_frac      = getf("NightstandConcept.CentralRegionFrac", 0.25f,
            "central image box is [frac, 1-frac]×[frac, 1-frac]; a detectable");
    out.epistemic_cooldown_cycles= geti("NightstandConcept.EpistemicCooldownCycles", 200,
            "min cycles withdrawn after satisfaction");
    out.nightstand_log_period_frames  = geti("NightstandConcept.NightstandLogPeriodFrames", 30,
            "per-cycle log throttle");
    out.support_bank_max_points    = geti("NightstandConcept.SupportBankMaxPoints", 4000,
            "cap on the nightstand-owned support-point memory bank");
    out.support_bank_quantization_m= getf("NightstandConcept.SupportBankQuantizationM", 0.02f,
            "support-bank dedup grid (m)");
    out.support_select_radius_margin_m = getf("NightstandConcept.SupportSelectRadiusMarginM", 0.50f,
            "XY margin (m) around the model for support-bank selection");
    out.support_select_height_margin_m = getf("NightstandConcept.SupportSelectHeightMarginM", 0.25f,
            "Z margin (m) around the model for support-bank selection");

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "Demote Operating→Degraded→Waiting when the retina's `masks` node stops advancing its mask_frame_id for this long (producer dead/stalled) — don't…");
    out.show_dashboard           = getb("NightstandConcept.ShowDashboard", true,
            "Show the combined GUI window (belief timeseries dashboard + evidence monitor in one splitter). false ⇒ no GUI windows are built at all (headless);…");
    out.shape_eval_period        = geti("NightstandConcept.ShapeEvalPeriod", 30,
            "Every shape_eval_period cycles (once the support bank has ≥ shape_eval_min_points) fit a ROUND model to the accumulated cloud and accumulate a…");
    out.shape_eval_min_points    = geti("NightstandConcept.ShapeEvalMinPoints", 300,
            "");
    out.shape_evidence_clamp     = getf("NightstandConcept.ShapeEvidenceClamp", 8.0f,
            "");
    out.dump_cloud_path          = gets("NightstandConcept.DumpCloudPath", "",
            "DIAGNOSTIC one-shot: if non-empty, dump a fitted nightstand's accumulated support-bank point cloud (room frame, XYZ per line) to this path ONCE…");

    // ─── NightstandModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("NightstandModel.SigmaObs", 0.05f,
            "Forwarded to NightstandModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top face is attributed to the slab…");
    out.sdf_threshold_for_storage = getf("NightstandModel.SdfThresholdForStorage", 0.08f,
            "On-surface membership for the candidate/residual split in NightstandFitter::observe");

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("NightstandModel.AI2SigmaBaseM", 0.03f,
            "base on-surface obs noise std (m); R = σ² (+ motion_var + …)");
    out.ai2_clutter_frac     = getf("NightstandModel.AI2ClutterFrac", 0.10f,
            "ε: prior weight of the uniform clutter mixture component");
    out.ai2_clutter_scale_m  = getf("NightstandModel.AI2ClutterScaleM", 0.12f,
            "a point further than ~this from every surface is likely clutter");
    out.ai2_prior_size_std   = getf("NightstandModel.AI2PriorSizeStd", 0.30f,
            "(legacy; the size prior is now split into footprint vs height below)");
    out.ai2_prior_footprint_m   = getf("NightstandModel.AI2PriorFootprintM", 0.60f,
            "mean of the w & h(depth) prior (m)");
    out.ai2_prior_footprint_std = getf("NightstandModel.AI2PriorFootprintStd", 0.08f,
            "TIGHT → strong footprint prior (m)");
    out.ai2_prior_depth_m       = getf("NightstandModel.AI2PriorDepthM", -1.0f,
            "depth (local Y, front-to-back) mean (m); <=0 = same as the footprint (square)");
    // ★FALLBACK ALIGNED TO THE MANIFEST (2026-08-17). This read 1.70 while the manifest and config.toml
    // both say 1.90 — invisible while config is present, and a silently shorter nightstand the moment it is
    // not. A code default that contradicts the declaration is the same defect class as an inherited one.
    out.ai2_prior_height_m      = getf("NightstandModel.AI2PriorHeightM", 1.90f,
            "mean of the H (vertical) static anchor (m) — a standard tall nightstand");
    out.ai2_prior_height_std    = getf("NightstandModel.AI2PriorHeightStd", 0.50f,
            "anchor std (m); stops the box top floating above the cloud");
    out.ai2_depth_unobs_precision = getf("NightstandModel.AI2DepthUnobsPrecision", 1500.0f,
            "extra 1/m² when depth extent is unobserved (single face)");
    out.ai2_depth_obs_band_m      = getf("NightstandModel.AI2DepthObsBandM", 0.10f,
            "ly-spread (m) below which the cloud is a single depth face");
    out.ai2_top_no_float_precision = getf("NightstandModel.AI2TopNoFloatPrecision", 10000.0f,
            "firm two-sided anchor pinning H → observed robust cloud top (1/m²)");
    out.ai2_top_no_float_margin_m  = getf("NightstandModel.AI2TopNoFloatMarginM", 0.02f,
            "upward allowance (m) added to the observed top as the anchor target");
    out.ai2_top_overseg_sigma_per_m = getf("NightstandModel.AI2TopOversegSigmaPerM", 2.0f,
            "R inflation (σ per m) for points ABOVE the robust top — fades the");
    out.ai2_wall_precision          = getf("NightstandModel.AI2WallPrecision", 400.0f,
            "1/m² at zero gap (back face → wall)");
    out.ai2_wall_parallel_precision = getf("NightstandModel.AI2WallParallelPrecision", 200.0f,
            "back face parallel to the wall (width-axis · wall-normal → 0)");
    out.ai2_wall_reach_m            = getf("NightstandModel.AI2WallReachM", 0.15f,
            "gap scale over which the flush hypothesis loses its weight");
    out.ai2_door_clearance_gain     = getf("NightstandModel.AI2DoorClearanceGain", 3.0f,
            "WALL as a competing per-point explanation (explaining away — see NightstandBeliefParams). π_wall and the extra wall-plane std. 0 = OFF (mixture…");
    out.detect_min_fill             = getf("NightstandModel.DetectMinFill", 0.10f,
            "INFERRED VOLATILITY (MODEL_HISTORY.md §3) — the HISTORY stage");
    out.detect_max_fill             = getf("NightstandModel.DetectMaxFill", 0.60f,
            "");
    out.detect_soft                 = getf("NightstandModel.DetectSoft", 0.06f,
            "");
    out.ai2_volatility_infer        = getb("NightstandModel.AI2VolatilityInfer", false,
            "ON 08-03 for the A/B tour. Watch omega_w/tau_w in ai2_log.csv");
    out.ai2_volatility_lr           = getf("NightstandModel.AI2VolatilityLr", 0.02f,
            "gradient step on ω per MEASURED cycle (predict-only cycles never touch it)");
    out.ai2_volatility_sigma        = getf("NightstandModel.AI2VolatilitySigma", 2.0f,
            "hyperprior width σ_ω: how far evidence may pull ω from the configured Q");
    out.ai2_wall_explain_frac       = getf("NightstandModel.AI2WallExplainFrac", 0.25f,
            "π_wall: prior weight of 'this point belongs to a known room wall'");
    out.ai2_wall_explain_sigma_m    = getf("NightstandModel.AI2WallExplainSigmaM", 0.05f,
            "extra wall-plane std (m): room-polygon + localization slack (adds to WallRef σ)");
    out.ai2_wall_no_cross_precision = getf("NightstandModel.WallNoCrossPrecision", 2000.0f,
            "1/m² per m of penetration past the wall (>> the flush precision)");
    out.ai2_wall_no_cross_margin_m  = getf("NightstandModel.WallNoCrossMarginM", 0.0f,
            "interior offset (m) inside which the constraint activates");
    out.ai2_process_std_m    = getf("NightstandModel.AI2ProcessStdM", 0.005f,
            "predict process-noise std, length DOFs (m/frame)");
    out.ai2_process_std_yaw  = getf("NightstandModel.AI2ProcessStdYaw", 0.01f,
            "predict process-noise std, yaw (rad/frame)");
    out.ai2_age_nominal_dt_s = getf("NightstandModel.AI2AgeNominalDtS", 0.0f,
            "Stale-belief aging (measurement-age → covariance)");
    out.ai2_common_mode_pos_std  = getf("NightstandModel.AI2CommonModePosStd", 0.03f,
            "shared position error (m); pose-chain cov adds to it");
    out.ai2_common_mode_size_std = getf("NightstandModel.AI2CommonModeSizeStd", 0.02f,
            "shared size error w,h,H (m)");
    out.ai2_common_mode_yaw_std  = getf("NightstandModel.AI2CommonModeYawStd", 0.03f,
            "shared yaw error (rad)");
    out.motion_cm_pos_gain       = getf("NightstandModel.MotionCmPosGain", 0.10f,
            "position (cx,cy) shared-error std per m/s of motion_dotd");
    out.motion_cm_size_gain      = getf("NightstandModel.MotionCmSizeGain", 0.20f,
            "extent (w,h,H) shared-error std per m/s — the anti-RESHAPE lever");
    out.motion_cm_yaw_gain       = getf("NightstandModel.MotionCmYawGain", 0.12f,
            "yaw shared-error std (rad) per m/s — the anti-ROTATE lever");
    out.ai2_ang_lever_m           = getf("NightstandModel.AI2AngLeverM", 2.0f,
            "rad/s → m/s lever (tangential speed of a nightstand ~this far away) for ego-motion");
    out.ai2_periph_ref            = getf("NightstandModel.AI2PeriphRef", 0.50f,
            "centroid radius (focal-norm) at which the off-axis periphery penalty → 1");
    out.ai2_motion_ref_mps        = getf("NightstandModel.AI2MotionRefMps", 0.60f,
            "motion magnitude (m/s) at which a fully-peripheral frame is fully unreliable");
    out.ai2_motion_confirm_only   = getb("NightstandModel.AI2MotionConfirmOnly", true,
            "Discrete confirm-only gate ('only update if the robot is still'): true ⇒ the fitter CONFIRMS but does NOT move the geometry mean (predict-only…");
    out.ai2_still_lin_mps         = getf("NightstandModel.AI2StillLinMps", 0.05f,
            "camera linear speed (m/s) below which the robot counts as 'still'");
    out.ai2_still_ang_radps       = getf("NightstandModel.AI2StillAngRadps", 0.10f,
            "camera angular speed (rad/s) below which the robot counts as 'still'");
    out.ai2_still_dotd            = getf("NightstandModel.AI2StillDotd", 0.05f,
            "per-mask ego-motion corruption speed (m/s) still-level");
    out.ai2_moving_update_center_radius = getf("NightstandModel.AI2MovingUpdateCenterRadius", 0.35f,
            "mask centroid radius below which a moving update is still allowed");
    out.ai2_range_noise_lat_per_m = getf("NightstandModel.AI2RangeNoiseLatPerM", 0.02f,
            "lateral deprojection std growth (m per m of range)");
    out.ai2_range_noise_yaw_per_m = getf("NightstandModel.AI2RangeNoiseYawPerM", 0.03f,
            "yaw common-mode std growth (rad per m of range)");
    out.ai2_range_noise_size_per_m = getf("NightstandModel.AI2RangeNoiseSizePerM", 0.08f,
            "SIZE (w,h,H) common-mode std growth (m per m of range): a distant");
    out.ai2_trunc_gate_frac    = getf("NightstandModel.AI2TruncGateFrac", 0.10f,
            "mask can't reshape/inflate a converged nightstand (freezes geometry afar) THRESHOLD (truncation gate): skip the geometric update (predict only)…");
    out.ai2_gn_iters         = geti("NightstandModel.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_csv_path         = gets("NightstandModel.AI2CsvPath", "",
            "if non-empty, append per-cycle belief (state + Σ diag + mask R) to CSV");
    out.detect_probe_csv_path = gets("NightstandConcept.DetectProbeCsvPath", out.detect_probe_csv_path,
            "The DETECTOR's truth table (rc::probe) — one row per cycle per instance: viewpoint, framing, and what YOLO did about it");
    out.birth_surprise_probe = getb("NightstandModel.BirthSurpriseProbe", false,
            "EXPERIMENTAL birth-surprise probe (read-only): read residual_concept's `grid` node as an unexplained- occupancy (surprise) field, cluster it, and LOG…");
    out.pixel_sigma_over_f     = getf("NightstandModel.PixelSigmaOverF", 0.0015f,
            "(footprint-residual path — OFF)");
    out.depth_sigma0_m         = getf("NightstandModel.DepthSigma0M", 0.006f,
            "depth std floor (m)");
    out.depth_sigma_range_coef = getf("NightstandModel.DepthSigmaRangeCoef", 0.004f,
            "depth std growth (m per m² of range)");
    out.model_sigma_m          = getf("NightstandModel.ModelSigmaM", 0.010f,
            "residual model/surface std (m) added in quadrature (R floor)");
    out.footprint_residual     = getb("NightstandModel.FootprintResidual", false,
            "a1′+a2′: weighted 2-D footprint residual + shared depth-affine nuisance (replaces the moment channel; the real grazing-yaw fix). depth_bias/scale_std…");
    out.quotient_chart         = getb("NightstandModel.QuotientChart", false,
            "C2v symmetry-quotient chart (PRECISION_AS_INFORMATION.md Stage 3): optimise the footprint in [s,a₁,a₂] so the 4 box representatives collapse to one…");
    out.depth_tilt_std         = getf("NightstandModel.DepthTiltStd", 0.020f,
            "shared per-frame depth tilt prior std (m/rad) — the yaw nuisance");
    out.depth_bias_std         = getf("NightstandModel.DepthBiasStd", 0.015f,
            "shared per-frame depth bias prior std (m)");
    out.depth_scale_std        = getf("NightstandModel.DepthScaleStd", 0.010f,
            "shared per-frame depth scale prior std (fraction)");

    // ─── "Is this really a nightstand?" plausibility filter + soft singleton ───────
    out.nightstand_filter_enabled   = getb("NightstandConcept.NightstandFilterEnabled", true,
            "master switch for the whole plausibility filter");
    out.plaus_aspect_scale      = getf("NightstandModel.AspectScale", 0.15f,
            "aspect = |w−h|/(w+h); aspect_ok = exp(−(aspect/scale)²) → 1 square");
    out.plaus_size_scale        = getf("NightstandModel.SizeScale", 0.15f,
            "size_ok = exp(−((w−fp)²+(h−fp)²)/(2·scale²)), fp = prior_footprint_m");
    out.plaus_alt_size_scale    = getf("NightstandModel.AltSizeScale", 0.60f,
            "alternative footprint std (m) around prior_footprint_m");
    out.plaus_height_min        = getf("NightstandModel.HeightPlausibleMin", 1.20f,
            "logistic centre (m): height_ok ≈ 0.5 here, →1 tall, →0 below ~1 m");
    out.plaus_height_soft       = getf("NightstandModel.HeightSoft", 0.15f,
            "logistic softness (m) of the height_ok falloff");
    out.plaus_fe_ref            = getf("NightstandModel.FeRef", 2.0f,
            "'healthy nightstand fit' free-energy reference (NEEDS LIVE TUNING)");
    out.plaus_fe_scale          = getf("NightstandModel.FeScale", 1.0f,
            "fit_ok = exp(−max(0, FE−fe_ref)/fe_scale)");
    out.plaus_clamp             = getf("NightstandModel.PlausClamp", 8.0f,
            "±bound on the plaus_evidence accumulator (bounded memory: one bad frame can't kill a real nightstand)");
    out.plaus_height_prior_gain = getf("NightstandModel.PlausHeightPriorGain", 2000.0f,
            "precision (1/m²) per unit (deficit/soft) below plaus_height_min");
    out.plaus_to_existence_gain = getf("NightstandModel.PlausToExistenceGain", 1.5f,
            "how strongly tanh(plaus_evidence) pushes the existence log-odds per cycle");
    out.singleton_inhibition    = getf("NightstandModel.SingletonInhibition", 1.0f,
            "mutual-inhibition weight: a STRONGER nightstand subtracts this·P(exists) from a weaker one's existence L");
    out.nightstand_filter_log       = getb("NightstandConcept.NightstandFilterLog", false,
            "log per-instance plausibility / evidence / existence-delta each cycle");

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("NightstandConcept.RtCovScale", 1.0f,
            "Upload the nightstand pose covariance onto the room→nightstand RT edge (rt_covariance_att, 6×6 SE3), mapped from the belief's full Σ over…");
    out.publish_object_obs            = getb("NightstandConcept.PublishObjectObs", false,
            "NightstandConcept.PublishObjectObs");
    out.object_obs_frame              = gets("NightstandConcept.ObjectObsFrame", "body",
            "NightstandConcept.ObjectObsFrame (localizer base node)");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "χ²₂ gate (~3σ) for a mask↔instance match once it has a cov");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.50f,
            "metric XY gate (m) before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f,
            "Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I)");
    out.recall_pi_move = getf("Tracker.RecallPiMove", 0.05f,
            "Mixture weight of 'the remembered nightstand was moved' in the recall location likelihood (class prior; GUESS, nightstands almost never move)");
    out.recall_log_prior_new = getf("Tracker.RecallLogPriorNew", 0.0f,
            "Log prior of 'new nightstand' against one recalled candidate (0 = equal)");
    out.tracker_birth_frames     = geti("Tracker.BirthFrames", 8,
            "frames a mask must stay unexplained before spawning a nightstand");
    out.birth_fusion             = getb("Tracker.BirthFusion", false,
            "FUSED BIRTH (EXPERIMENTAL, off by default): let residual-grid SURPRISE MASS under a detection accelerate its birth");
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain", 6.0f,
            "max extra evidence/frame at full corroboration (0 ⇒ baseline)");
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef", 8.0f,
            "residual mass at which corroboration is half-saturated (m/(m+ref))");
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM", 0.50f,
            "window (m) for the residual-mass sample under the detection");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.60f,
            "a birth must be ≥ this (m) from every existing nightstand");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.05f,
            "Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of the SMALLER footprint (two nightstands cannot…");
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM", 1.0f,
            "Default geometry for a nightstand BORN from a mask, used only when no birth burst is available (see birth_frag_* below); the belief refines from…");
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM", 0.6f,
            "");
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM", 0.75f,
            "");
    // Birth fragment: keep the probation burst and admit the birth on it (see nightstand_config.h).
    out.birth_frag_enabled       = getb("Tracker.BirthFragment", true,
            "A candidate matures over tracker_birth_frames observations; until now every mask cloud seen on the way was discarded and the instance was seeded from…");
    out.birth_frag_cell_m       = getf("Tracker.BirthFragmentVoxelM", 0.03f,
            "dedup grid for the burst (finer than the bank: a burst is short)");
    out.birth_frag_max_pts       = geti("Tracker.BirthFragmentMaxPts", 20000,
            "hard cap per candidate; a lingering blob cannot grow without bound");
    out.birth_frag_delta_ms      = static_cast<std::uint64_t>(
                                       std::max(0, geti("Tracker.BirthFragmentDeltaMs", 4000,
                                               "")));
    out.birth_admit_plausibility = getf("Tracker.BirthAdmitPlausibility", 0.35f,
            "ADMISSION on the accumulated burst");
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f,
            "min YOLO confidence for a ricoh detection to raise attention");
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f,
            "extra angular tolerance on the nightstand's angular half-size");
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM", 1.0f,
            "extra range band (ricoh range is rough/indicative)");

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("NightstandModel.LidarPrecision", 0.0f,
            "per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF");
    out.lidar_bpearl_precision = getf("NightstandModel.LidarBpearlPrecision", 0.0f,
            "Low 'bpearl' LiDAR as a SEPARATE per-device ray-set (own origin, occlusion-aware first-hit; sees the legs the high helios grazes over)");
    out.lidar_robust_c_m     = getf("NightstandModel.LidarRobustCM", 0.05f,
            "Cauchy scale (m): returns this far off the surface fade out");
    out.lidar_select_margin_m = getf("NightstandModel.LidarSelectMarginM", 0.10f,
            "pre-select returns within (birth half-extent + margin), all z up to top");
    out.lidar_coverage_n0     = getf("NightstandModel.LidarCoverageN0", 60.0f,
            "LiDAR ray count for FULL weight; fewer → proportionally down-weighted");
    out.lidar_coverage_ang_power = getf("NightstandModel.LidarCoverageAngPower", 1.0f,
            "Angular-coverage weighting: precision ×= (1−R)^p, R = mean-resultant length of return bearings about the centre. p=0 disables (flat), p=1 = pure…");
    out.max_step_m            = getf("NightstandModel.MaxStepM", 1.0f,
            "THRESHOLD (outlier step guard): a static nightstand cannot physically move this far in one frame, so a GN step whose centre jump exceeds it is an…");
    out.coverage_precision    = getf("NightstandModel.CoveragePrecision", 0.0f,
            "Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a large mask grows to explain it (fixes…");
    out.coverage_robust_c_m   = getf("NightstandModel.CoverageRobustCM", 0.15f,
            "Cauchy scale (m): on-plane points this far outside fade out (stray-point guard)");
    out.free_space_precision  = getf("NightstandModel.FreeSpacePrecision", 0.0f,
            "The counter-force that BOUNDS coverage");
    out.footprint_moment_precision = getf("NightstandModel.FootprintMomentPrecision", 0.0f,
            "Measures (w,h,yaw) from the top-band cloud's 2D inertia tensor and folds it as a linear Gaussian factor — the escape from the clutter-trap that…");
    out.footprint_moment_range_per_m = getf("NightstandModel.FootprintMomentRangePerM", 0.03f,
            "GENTLE range term (m per m of range): shared per-frame moment variance grows as (this·range)²");
    out.fe_baseline_adapt_down       = getf("NightstandModel.FeBaselineAdaptDown", 0.05f,
            "Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the nightstand moved — stays surprising long enough to…");
    out.fe_baseline_adapt_up         = getf("NightstandModel.FeBaselineAdaptUp", 0.005f,
            "");
    out.fe_surprise_smooth           = getf("NightstandModel.FeSurpriseSmooth", 0.10f,
            "");
    out.footprint_moment_motion_gain = getf("NightstandModel.FootprintMomentMotionGain", 0.30f,
            "Ego-motion (motion_dotd) coupling: moment variance grows by (this·motion_dotd)² so a 'going-away/rotation' frame (degraded/split mask) can't reshape…");
    out.orientation_motion_ref       = getf("NightstandModel.OrientationMotionRef", 0.50f,
            "");

    // ─── Appearance-based FRONT (door) detection + yaw resolver ────────────────
    out.front_detect_enabled   = getb("NightstandConcept.FrontDetectEnabled", true,
            "master switch: subscribe to ZED RGB + run detect_front/resolve_front");
    out.contour_check_enabled  = getb("NightstandConcept.ContourCheckEnabled", true,
            "The classifier-free existence channel: score the believed silhouette against the RGB's own gradient and against the DEPTH plane's agreement with the…");
    out.front_min_face_area_px = getf("NightstandConcept.FrontMinFaceAreaPx", 900.0f,
            "min projected face area (px²) to score for door-ness");
    out.front_min_confidence   = getf("NightstandConcept.FrontMinConfidence", 0.10f,
            "min door-ness margin (max−second)/(max+eps) for detect_front to emit");
    out.front_log              = getb("NightstandConcept.FrontLog", false,
            "log each adopted door-mode flip / accepted cue");
    out.obliquity_moment_gain        = getf("NightstandModel.ObliquityMomentGain", 0.0f,
            "Grazing-view yaw stability (CSV rogue-rotation fix)");
    out.footprint_moment_completeness_gain = getf("NightstandModel.FootprintMomentCompletenessGain", 0.0f,
            "footprint_moment_completeness_gain / _min_completeness: forwarded to NightstandBeliefParams — inflate the moment measurement variance as the…");
    out.footprint_moment_min_completeness  = getf("NightstandModel.FootprintMomentMinCompleteness", 0.02f,
            "clamp floor for 1/completeness in the term above (caps the inflation of");

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("NightstandModel.ExistenceRemovalEnabled", false,
            "Each cycle carve the LiDAR sweep against the nightstand footprint → occupancy/free-space log-odds; remove when P(occupied) < removal_prob");
    out.existence_removal_prob    = getf("NightstandModel.ExistenceRemovalProb", 0.12f,
            "decision boundary: remove when L < log(p/(1−p))");
    out.existence_out_of_room_gain = getf("NightstandModel.ExistenceOutOfRoomGain", 1.5f,
            "|dL| per evidence cycle while the nightstand centre is outside the room outline by more than half its short side");
    out.existence_frame_correlation = getf("NightstandModel.ExistenceFrameCorrelation", 0.0f,
            "rho: frame-to-frame correlation of MISSES. 0 = independent trials (the historic behaviour and the default); opt in only with a value MEASURED from…");
    out.existence_logodds_max     = getf("NightstandModel.ExistenceLogoddsMax", 4.0f,
            "clamp |L| so evidence stays finite AND recoverable");
    out.existence_detection_prob  = getf("NightstandModel.ExistenceDetectionProb", 0.85f,
            "P(beam through OCCUPIED footprint returns from it)");
    out.existence_clutter_prob    = getf("NightstandModel.ExistenceClutterProb", 0.05f,
            "P(beam through EMPTY footprint returns anyway) — spurious rate");
    out.existence_sensor_sigma_m  = getf("NightstandModel.ExistenceSensorSigmaM", 0.03f,
            "LiDAR range σ (m) for the soft occ/free surface split");
    out.existence_remove_frames   = geti("NightstandModel.ExistenceRemoveFrames", 15,
            "debounce: require the removal decision this many consecutive");
    out.existence_absence_range_ref_m = getf("NightstandModel.ExistenceAbsenceRangeRefM", 2.5f,
            "range (m) below which absence is trusted at full weight");
    out.existence_absence_range_power = getf("NightstandModel.ExistenceAbsenceRangePower", 2.0f,
            "decay exponent (2 ≈ angular-area ∝ 1/range²); 0 disables");
    out.existence_verify_surprise     = getf("NightstandModel.ExistenceVerifySurprise", 20.0f,
            "decayed go-verify surprise (un-resolvable absence) above which");
    out.verify_surprise_smooth        = getf("NightstandModel.VerifySurpriseSmooth", 0.10f,
            "EMA weight of the go-verify surprise accumulator (new sample");
    out.existence_verify_gain         = getf("NightstandModel.ExistenceVerifyGain", 5.0f,
            "epistemic gain (nats) a wants_verification nightstand gets, so the");

    std::print("nightstand_concept: configuration loaded.\n");
    // Peripheral (ricoh) existence confirmation — see nightstand_config.h. OFF by default.
    out.ricoh_confirm_enabled        = getb("NightstandConcept.RicohConfirmEnabled", out.ricoh_confirm_enabled,
            "NightstandConcept.RicohConfirmEnabled");
    out.ricoh_confirm_detection_prob = getf("NightstandConcept.RicohConfirmDetectionProb", out.ricoh_confirm_detection_prob,
            "NightstandConcept.RicohConfirmDetectionProb");
    out.ricoh_confirm_clutter_prob   = getf("NightstandConcept.RicohConfirmClutterProb", out.ricoh_confirm_clutter_prob,
            "NightstandConcept.RicohConfirmClutterProb");

    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★z_top is the AI2 HEIGHT PRIOR (1.90), NOT Tracker.BirthHeightM. Birth height is deliberately 0.75 —
    // below plaus_height_min so every real nightstand grows INTO its size — so it is the wrong number for a
    // class-level span. Getting that backwards would declare a body half the height of the object.
    {
        const auto span = rc::manifest::adopt_span(kManifestPath, "nightstand",
                                                  rc::manifest::Support::floor_anchored,
                                                  out.ai2_prior_height_m, out.ai2_prior_height_m);
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = out.ai2_prior_height_m;
        decl.extent_m = out.ai2_prior_height_m;
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body("nightstand ", "lidar_select",
                        span.z0 - out.lidar_select_margin_m, span.z1 + out.lidar_select_margin_m, decl);
        ok_bands &= rc::manifest::band_contains_body("nightstand ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] nightstand ✓ every derived z-band contains the declared body\n");
    }

    return out;
}

// ─── Concept-manifest cross-check (declarative-priors experiment, step 1½) ────────────────────────
//
// common/concept_manifest/<concept>.concept.toml declares WHAT a nightstand IS — its priors as world facts,
// separate from the lifecycle knobs. It is not authoritative yet. This compares it against the priors the live
// etc/config.toml actually produced, so we learn whether a manifest can reproduce the running agent BEFORE
// anything is generated from it. Startup-only, read-only: it never changes a value, it only reports.
//
// A DIFFERS line is a finding, not a failure — it means the manifest and the running config disagree about a
// world fact, and one of them is wrong. A MISSING line means the manifest does not yet describe that prior.
bool verify_nightstand_manifest(const NightstandConfig& out, const std::string& path)
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

    chk("prior.footprint.mean_m",              out.ai2_prior_footprint_m,      "nightstand footprint mean");
    chk("prior.footprint.std_m",               out.ai2_prior_footprint_std,    "footprint prior std");
    if (out.ai2_prior_depth_m > 0.0f)
        chk("prior.footprint.depth_mean_m",    out.ai2_prior_depth_m,          "footprint depth mean");
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
