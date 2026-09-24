/*
 * table_config.cpp  —  fill TableConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in table_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "table_config.h"

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)
#include "../../common/config_report/config_read.h"           // rc::cfg::Reader (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

TableConfig load_table_config(const ConfigLoader& cfg)
{
    TableConfig out;

    // The one place this path is written. It is relative to the agent's CWD (<agent>/), not to src/.
    static constexpr const char* kManifestPath = "../common/concept_manifest/table.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL. The manifest has carried `from = measured|fitted|nominal|inherited`
    // since 2026-08-11, with a note saying "inherited is worse than absent — an absent value gets asked
    // about, an inherited one gets trusted". A note stopped nothing: hood_concept shipped TEN cloned defects
    // that week, several of them DECLARED as inherited, in writing, in this very file. The declaration is
    // now the enforcement — see rc::manifest::provenance_ok. Refusing to start is the whole point: a comment
    // cannot fail a build, and everything that only warned was fixed around rather than fixed.
    if (not rc::manifest::provenance_ok(kManifestPath, "table"))
        std::exit(EXIT_FAILURE);

    // ── EVERY READ GOES THROUGH THE REGISTRY, AND CARRIES ITS OWN EXPLANATION ──────────────────
    //
    // These four were a local copy of the same quartet twelve agents each wrote out again, and they
    // answered only "what is the value". The question that actually costs time is "where did the
    // value COME FROM": measured across the fleet, 375 of 2033 keys the code reads are absent from
    // the agent's own config.toml and quietly run the code default. Commit b6c40b4 said it exactly —
    // "absent is not off, but it reads as off" — and P3Bot ran six older solver defaults that way.
    //
    // ★The description belongs HERE and not in etc/config.toml. File prose can be edited to match a
    // paste and go on reading plausibly: door_concept's config carries four chair-shaped
    // Tracker.BirthSeat* keys whose comment was reworded to sound door-ish, and door_concept reads
    // none of them. A description next to the read cannot drift from the code, and it still exists
    // for a key the file never mentions — which is the case that matters.
    //
    // rc::cfg::experiment marks an A/B arm: it PRINTS even at its default (an arm sitting at its
    // default is still a fact about the run) and it enters the run fingerprint. rc::cfg::diagnostic
    // marks a log/CSV/GUI switch, which is deliberately EXCLUDED from the fingerprint so that
    // turning a CSV on does not change which arm a run claims to be.
    rc::cfg::Reader reader(cfg, "table_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("TableConcept.StateEps", 0.04f,
            "sum|d state| convergence threshold between cycles (m+rad)");
    out.K_stable                 = geti("TableConcept.KStable", 30,
            "consecutive converged cycles before model_stable is asserted");
    out.detection_alive_max_frames = geti("TableConcept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh mask before detection_alive goes false");
    out.matched_frames_before_aging = geti("TableConcept.MatchedFramesBeforeAging", 5,
            "a belief with fewer matched frames than this is too young to age");
    out.central_region_frac      = getf("TableConcept.CentralRegionFrac", 0.25f,
            "central image box is [frac,1-frac]^2; a detection inside it counts as centred");
    out.obs_distance             = getf("TableConcept.ObsDistance", 1.8f,
            "d_obs, the standoff distance (m) the epistemic planner aims for");
    out.epistemic_cooldown_cycles= geti("TableConcept.EpistemicCooldownCycles", 200,
            "min cycles an affordance stays withdrawn after satisfaction");
    out.table_log_period_frames  = geti("TableConcept.TableLogPeriodFrames", 30,
            "per-cycle log throttle (frames)", rc::cfg::diagnostic);
    out.support_bank_max_points    = geti("TableConcept.SupportBankMaxPoints", 4000,
            "cap on the table-owned support-point memory bank");
    out.support_bank_quantization_m= getf("TableConcept.SupportBankQuantizationM", 0.02f,
            "support-bank dedup grid (m)");
    out.support_select_radius_margin_m = getf("TableConcept.SupportSelectRadiusMarginM", 0.50f,
            "XY margin (m) around the model for support-bank selection");
    out.support_select_height_margin_m = getf("TableConcept.SupportSelectHeightMarginM", 0.25f,
            "Z margin (m) around the model for support-bank selection");

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "stop integrating masks after this long without one (producer dead or stalled)");
    out.masks_legacy_room_frame  = getb("Masks.legacy_room_frame", false,
            "A/B: read masks as ALREADY in the room frame (pre-2026-07 producer)", rc::cfg::experiment);
    out.show_dashboard           = getb("TableConcept.ShowDashboard", true,
            "build the GUI (belief timeseries + evidence monitor); false = fully headless", rc::cfg::diagnostic);
    out.shape_eval_period        = geti("TableConcept.ShapeEvalPeriod", 30,
            "cycles between round-vs-rectangular shape re-evaluations");
    out.shape_eval_min_points    = geti("TableConcept.ShapeEvalMinPoints", 300,
            "support-bank points required before a shape evaluation runs");
    out.shape_evidence_clamp     = getf("TableConcept.ShapeEvidenceClamp", 8.0f,
            "bound (nats) on the accumulated round-vs-rect log-Bayes factor");
    out.shape_fit_iters          = geti("TableConcept.ShapeFitIters", 20,
            "iterations per shape fit; both hypotheses score the SAME points, so the test is PAIRED");
    out.shape_fit_max_points     = geti("TableConcept.ShapeFitMaxPoints", 1000,
            "cap on points used per shape fit");
    out.shape_evidence_ema_alpha = getf("TableModel.ShapeEvidenceEmaAlpha", 0.0f,
            "EMA on shape evidence: the bank is largely the same points each time, so summing measures dwell, not information");
    out.dump_cloud_path          = gets("TableConcept.DumpCloudPath", "",
            "one-shot dump of the accumulated support cloud (room frame, XYZ per line); empty = off", rc::cfg::diagnostic);

    // ─── TableModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("TableModel.SigmaObs", 0.05f,
            "on-surface mask-point observation noise std (m) for the top/leg SDF split");
    out.sdf_threshold_for_storage = getf("TableModel.SdfThresholdForStorage", 0.08f,
            "on-surface membership band (m) for the candidate/residual split in observe()");

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("TableModel.AI2SigmaBaseM", 0.03f,
            "base on-surface observation noise std (m); R = sigma^2 + motion + range terms");
    out.detect_min_fill      = getf("TableModel.DetectMinFill", 0.10f,
            "detector envelope: mask fill fraction below which YOLO stops seeing the table");
    out.detect_max_fill      = getf("TableModel.DetectMaxFill", 0.60f,
            "detector envelope: fill fraction above which the table overflows the frame");
    out.detect_soft          = getf("TableModel.DetectSoft", 0.06f,
            "softness (fill fraction) of both envelope edges");
    out.ai2_clutter_frac     = getf("TableModel.AI2ClutterFrac", 0.10f,
            "epsilon: prior weight of the uniform clutter mixture component");
    out.ai2_clutter_scale_m  = getf("TableModel.AI2ClutterScaleM", 0.12f,
            "a point further than this (m) from every surface is probably clutter");
    out.ai2_prior_size_std   = getf("TableModel.AI2PriorSizeStd", 0.30f,
            "broad prior std (m) on w,h,H");
    out.ai2_process_std_m    = getf("TableModel.AI2ProcessStdM", 0.005f,
            "process-noise std on pose cx,cy (m per frame)");
    out.ai2_process_std_yaw  = getf("TableModel.AI2ProcessStdYaw", 0.01f,
            "process-noise std on yaw (rad per frame)");
    out.ai2_process_std_extent_m  = getf("TableModel.AI2ProcessStdExtentM", 0.0f,
            "process noise on extent w,h,H; 0 = a table's dimensions are CONSTANT");
    out.ai2_clamp_sigma_to_prior  = getb("TableModel.AI2ClampSigmaToPrior", true,
            "cap Sigma at the prior on predict (OU decay), so ageing cannot make a table more plastic than its prior", rc::cfg::experiment);
    out.ai2_age_nominal_dt_s = getf("TableModel.AI2AgeNominalDtS", 0.0f,
            "turn per-frame Q into a rate so Sigma inflates on the agent's own clock while the sensor is silent; <=0 = off");
    out.ai2_common_mode_pos_std  = getf("TableModel.AI2CommonModePosStd", 0.03f,
            "shared (common-mode) position error std (m); the pose-chain covariance adds to it");
    out.ai2_common_mode_size_std = getf("TableModel.AI2CommonModeSizeStd", 0.02f,
            "shared size error std (m) on w,h,H");
    out.ai2_common_mode_yaw_std  = getf("TableModel.AI2CommonModeYawStd", 0.03f,
            "shared yaw error std (rad)");
    out.motion_cm_pos_gain       = getf("TableModel.MotionCmPosGain", 0.10f,
            "shared position error std per m/s of ego-motion");
    out.motion_cm_size_gain      = getf("TableModel.MotionCmSizeGain", 0.20f,
            "shared extent error std per m/s of ego-motion - the anti-RESHAPE lever");
    out.motion_cm_yaw_gain       = getf("TableModel.MotionCmYawGain", 0.12f,
            "shared yaw error std (rad) per m/s of ego-motion - the anti-ROTATE lever");
    out.ai2_range_noise_lat_per_m = getf("TableModel.AI2RangeNoiseLatPerM", 0.02f,
            "lateral deprojection std growth (m per m of range)");
    out.ai2_range_noise_yaw_per_m = getf("TableModel.AI2RangeNoiseYawPerM", 0.03f,
            "yaw common-mode std growth (rad per m of range)");
    out.ai2_range_noise_size_per_m = getf("TableModel.AI2RangeNoiseSizePerM", 0.08f,
            "size common-mode std growth (m per m of range): a distant view cannot resize the box");
    out.ai2_coverage_size_gain     = getf("TableModel.AI2CoverageSizeGain", 0.30f,
            "size common-mode std (m) per unit of (1/c - 1) - anti-RESHAPE under partial coverage");
    out.ai2_coverage_yaw_gain      = getf("TableModel.AI2CoverageYawGain", 0.15f,
            "yaw common-mode std (rad) per unit of (1/c - 1) - anti-ROTATE");
    out.ai2_coverage_pos_gain      = getf("TableModel.AI2CoveragePosGain", 0.05f,
            "position common-mode std (m) per unit of (1/c - 1)");
    out.ai2_coverage_min           = getf("TableModel.AI2CoverageMin", 0.02f,
            "floor on coverage c so 1/c stays finite, capping the inflation");
    out.ai2_trunc_gate_frac    = getf("TableModel.AI2TruncGateFrac", 0.10f,
            "THRESHOLD: skip the geometric update when this fraction of the silhouette lies on the image border");
    out.fixation_enabled       = getb("TableModel.FixationEnabled", true,
            "only let a frame touch geometry when the table is resolvable, centred and still", rc::cfg::experiment);
    out.fixation_min_pts       = geti("TableModel.FixationMinPts", 300,
            "fixation RESOLVABLE: min mask points before a frame may touch the geometry");
    out.fixation_max_trunc     = getf("TableModel.FixationMaxTrunc", 0.10f,
            "fixation RESOLVABLE: max silhouette truncation");
    out.fixation_range_m       = getf("TableModel.FixationRangeM", 0.0f,
            "RETIRED as a fixation gate (0 = off); kept so an A/B revert is one edit", rc::cfg::experiment);   // retired as a gate
    out.fixation_centre_frac   = getf("TableModel.FixationCentreFrac", 0.60f,
            "fixation CENTRED: max normalised ROI offset (0 = centred, 1 = image edge)");
    out.fixation_still_dotd    = getf("TableModel.FixationStillDotd", 0.05f,
            "fixation STILL: max ego-motion mask smear |motion_dotd| (m/s)");
    out.ai2_gn_iters         = geti("TableModel.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_csv_path         = gets("TableModel.AI2CsvPath", "",
            "per-cycle belief CSV (state + Sigma diagonal + mask R); empty = off", rc::cfg::diagnostic);
    out.detect_probe_csv_path = gets("TableConcept.DetectProbeCsvPath", out.detect_probe_csv_path,
            "detector inverse-model probe CSV: what was predicted visible vs what YOLO did", rc::cfg::diagnostic);
    out.birth_surprise_probe = getb("TableModel.BirthSurpriseProbe", false,
            "log uncovered high-surprise regions beside each birth decision", rc::cfg::diagnostic);
    out.pixel_sigma_over_f     = getf("TableModel.PixelSigmaOverF", 0.0015f,
            "sigma_px/f: transverse std per m of range");
    out.depth_sigma0_m         = getf("TableModel.DepthSigma0M", 0.006f,
            "depth std floor (m)");
    out.depth_sigma_range_coef = getf("TableModel.DepthSigmaRangeCoef", 0.004f,
            "depth std growth (m per m^2 of range)");
    out.model_sigma_m          = getf("TableModel.ModelSigmaM", 0.010f,
            "residual model std floor (m)");
    out.footprint_residual     = getb("TableModel.FootprintResidual", false,
            "weighted 2-D footprint residual + shared depth-affine nuisance - the real grazing-yaw fix", rc::cfg::experiment);
    out.quotient_chart         = getb("TableModel.QuotientChart", false,
            "C2v symmetry-quotient chart: optimise the footprint so the 4 box representatives collapse to one point", rc::cfg::experiment);
    out.depth_tilt_std         = getf("TableModel.DepthTiltStd", 0.020f,
            "shared per-frame depth tilt prior std (m/rad) - the yaw nuisance");
    out.depth_bias_std         = getf("TableModel.DepthBiasStd", 0.015f,
            "shared per-frame depth bias prior std (m)");
    out.depth_scale_std        = getf("TableModel.DepthScaleStd", 0.010f,
            "shared per-frame depth scale prior std (fraction)");

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("TableConcept.RtCovScale", 1.0f,
            "scale on the pose covariance uploaded to the room->table RT edge (6x6 SE3)");
    out.publish_object_obs            = getb("TableConcept.PublishObjectObs", false,
            "publish per-cycle object observations");
    out.object_obs_frame              = gets("TableConcept.ObjectObsFrame", "body",
            "frame the published observations are expressed in (localizer base node)");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "chi^2_2 gate (~3 sigma) for a mask<->instance match once it has a covariance");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.50f,
            "metric XY gate (m) used before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f,
            "detection noise std R (m) added to the fit covariance in the gate: the mask centroid sits well off the geometric centre");
    out.tracker_birth_frames     = geti("Tracker.BirthFrames", 8,
            "frames a mask must stay unexplained before a table is born");
    out.birth_fusion             = getb("Tracker.BirthFusion", false,
            "fold residual-occupancy corroboration into the birth decision", rc::cfg::experiment);
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain", 6.0f,
            "max extra birth evidence per frame at full corroboration; 0 = baseline behaviour");
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef", 8.0f,
            "residual mass at which corroboration is half-saturated, m/(m+ref)");
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM", 0.50f,
            "window (m) for the residual-mass sample under the detection");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.60f,
            "a birth must be at least this far (m) from every existing table");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.05f,
            "collapse two instances overlapping by this fraction of the SMALLER footprint; 0 disables");
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM", 1.0f,
            "default width (m) of a table born from a mask, before the belief refines it");
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM", 0.6f,
            "default depth (m) of a table born from a mask");
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM", 0.75f,
            "default height (m) at birth - and, for a table, ALSO the class surface height the manifest z-span is derived from");
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f,
            "min YOLO confidence for a ricoh detection to raise attention");
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f,
            "extra angular tolerance (rad) on the table's angular half-size");
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM", 1.0f,
            "extra range band (m); ricoh range is rough and indicative only");

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("TableModel.LidarPrecision", 0.0f,
            "helios per-ray range precision (1/m^2); 0 = the LiDAR range factor is OFF");
    out.lidar_bpearl_precision = getf("TableModel.LidarBpearlPrecision", 0.0f,
            "bpearl per-ray range precision: a separate ray-set that sees the legs helios grazes over");
    out.lidar_robust_c_m     = getf("TableModel.LidarRobustCM", 0.05f,
            "Cauchy scale (m): returns this far off the surface fade out");
    out.lidar_select_margin_m = getf("TableModel.LidarSelectMarginM", 0.10f,
            "pre-select returns within birth half-extent + this margin (m)");
    out.lidar_coverage_n0     = getf("TableModel.LidarCoverageN0", 60.0f,
            "ray count for full weight; fewer rays are proportionally down-weighted");
    out.lidar_coverage_ang_power = getf("TableModel.LidarCoverageAngPower", 1.0f,
            "precision *= (1-R)^p over return bearings; 0 = flat, 1 = pure circular variance");
    out.max_step_m            = getf("TableModel.MaxStepM", 1.0f,
            "THRESHOLD: a GN step moving the centre further than this in one frame is an outlier");
    out.coverage_precision    = getf("TableModel.CoveragePrecision", 0.0f,
            "grow-only pull from on-plane mask points the mixture ceded to clutter; 0 = OFF");
    out.coverage_robust_c_m   = getf("TableModel.CoverageRobustCM", 0.15f,
            "Cauchy scale (m) for the coverage pull");
    out.free_space_precision  = getf("TableModel.FreeSpacePrecision", 0.0f,
            "shrink term from LiDAR beams that passed through the box; 0 = OFF");
    out.footprint_moment_precision = getf("TableModel.FootprintMomentPrecision", 0.0f,
            "footprint-moment pull on w,h,yaw - the escape from the clutter trap; 0 = OFF");
    out.footprint_moment_range_per_m = getf("TableModel.FootprintMomentRangePerM", 0.03f,
            "moment variance grows as (this*range)^2; gentler than the size range term on purpose");
    out.fe_baseline_adapt_down       = getf("TableModel.FeBaselineAdaptDown", 0.05f,
            "free-energy baseline EMA rate DOWN - fast, to consolidate a better fit");
    out.fe_baseline_adapt_up         = getf("TableModel.FeBaselineAdaptUp", 0.005f,
            "free-energy baseline EMA rate UP - slow, so surprise stays surprising");
    out.fe_surprise_smooth           = getf("TableModel.FeSurpriseSmooth", 0.10f,
            "EMA weight on the surprise signal");
    out.footprint_moment_motion_gain = getf("TableModel.FootprintMomentMotionGain", 0.30f,
            "moment variance grows by (this*motion_dotd)^2, so a smeared frame cannot reshape an established box");
    out.orientation_motion_ref       = getf("TableModel.OrientationMotionRef", 0.50f,
            "reference ego-motion (m/s) for the orientation motion coupling");
    out.obliquity_moment_gain        = getf("TableModel.ObliquityMomentGain", 0.0f,
            "moment variance grows by (gain*(1/|cos theta|-1))^2 - the oblique-view rogue-rotation guard");
    out.footprint_moment_completeness_gain = getf("TableModel.FootprintMomentCompletenessGain", 0.0f,
            "inflate moment variance as the observed footprint area falls short of the model's");
    out.footprint_moment_min_completeness  = getf("TableModel.FootprintMomentMinCompleteness", 0.02f,
            "floor on observed completeness so that inflation stays finite");

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("TableModel.ExistenceRemovalEnabled", false,
            "carve the LiDAR sweep against the footprint and remove on EVIDENCE, not on missed detections", rc::cfg::experiment);
    out.existence_removal_prob    = getf("TableModel.ExistenceRemovalProb", 0.12f,
            "decision boundary: remove when P(occupied) falls below this");
    out.existence_logodds_max     = getf("TableModel.ExistenceLogoddsMax", 4.0f,
            "clamp |L| so the evidence stays finite AND recoverable");
    out.existence_frame_correlation = getf("TableModel.ExistenceFrameCorrelation", 0.0f,
            "how correlated consecutive misses are: N misses at one bad framing are worth far less than N observations");
    out.existence_detection_prob  = getf("TableModel.ExistenceDetectionProb", 0.85f,
            "P(a beam through an OCCUPIED footprint returns from it)");
    out.existence_clutter_prob    = getf("TableModel.ExistenceClutterProb", 0.05f,
            "P(a beam through an EMPTY footprint returns anyway) - the spurious rate");
    out.existence_sensor_sigma_m  = getf("TableModel.ExistenceSensorSigmaM", 0.03f,
            "LiDAR range sigma (m) for the soft occupied/free split");
    out.existence_remove_frames   = geti("TableModel.ExistenceRemoveFrames", 15,
            "debounce: require the removal decision this many consecutive cycles");
    out.existence_absence_range_ref_m = getf("TableModel.ExistenceAbsenceRangeRefM", 2.5f,
            "range (m) below which an absence is trusted at full weight");
    out.existence_absence_range_power = getf("TableModel.ExistenceAbsenceRangePower", 2.0f,
            "absence-weight decay exponent (2 ~ angular area falling as 1/range^2); 0 disables");
    out.existence_los_margin_m        = getf("TableModel.ExistenceLosMarginM", 0.25f,
            "a sample must be this far (m) BEYOND the first hit to count as occluded");
    out.existence_los_azim_bins       = geti("TableModel.ExistenceLosAzimBins", 0,
            "azimuth bins over 2pi for the occlusion oracle (720 ~ 0.5 deg); 0 = OFF");
    out.existence_verify_surprise     = getf("TableModel.ExistenceVerifySurprise", 20.0f,
            "decayed unresolvable-absence surprise above which the table asks to be verified");
    out.verify_surprise_smooth        = getf("TableModel.VerifySurpriseSmooth", 0.10f,
            "EMA weight of the go-verify surprise accumulator");
    out.existence_verify_gain         = getf("TableModel.ExistenceVerifyGain", 5.0f,
            "epistemic gain (nats) granted to a table that wants verification");
    out.existence_leg_occupancy       = getb("TableModel.ExistenceLegOccupancy", true,
            "also carve the 4 leg volumes for OCCUPANCY only - thin legs, and the hollow space between them must never vote absence", rc::cfg::experiment);
    out.existence_local_clutter       = getb("TableModel.ExistenceLocalClutter", true,
            "score occupancy as a CONTRAST against a shell around the box, not in isolation", rc::cfg::experiment);
    out.existence_lidar_confirms      = getb("TableModel.ExistenceLidarConfirms", false,
            "let LiDAR confirm as well as hold up; removal stays with the one sensor that reads a label", rc::cfg::experiment);
    out.existence_absence_envelope    = getb("TableModel.ExistenceAbsenceEnvelope", true,
            "weight absence by the detector envelope, so point-blank blindness cannot delete a real table", rc::cfg::experiment);

    std::print("table_concept: configuration loaded.\n");
    // Peripheral (ricoh) existence confirmation — see table_config.h. OFF by default.
    out.ricoh_confirm_enabled        = getb("TableConcept.RicohConfirmEnabled", out.ricoh_confirm_enabled,
            "peripheral (ricoh) existence confirmation channel", rc::cfg::experiment);
    out.ricoh_confirm_detection_prob = getf("TableConcept.RicohConfirmDetectionProb", out.ricoh_confirm_detection_prob,
            "P(ricoh detects the table when it is there)");
    out.ricoh_confirm_clutter_prob   = getf("TableConcept.RicohConfirmClutterProb", out.ricoh_confirm_clutter_prob,
            "P(ricoh reports a table where there is none)");

    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★For a table, Tracker.BirthHeightM (0.75) IS the class surface height, so it is the right span top —
    // the opposite of refrigerator, where birth height is deliberately far below the object. The asymmetry is
    // real; do not copy either choice without checking which one this object's birth height means.
    {
        const auto span = rc::manifest::adopt_span(kManifestPath, "table",
                                                  rc::manifest::Support::floor_anchored,
                                                  out.tracker_birth_height_m, out.tracker_birth_height_m);
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = out.tracker_birth_height_m;
        decl.extent_m = out.tracker_birth_height_m;
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body("table ", "lidar_select",
                        span.z0 - out.lidar_select_margin_m, span.z1 + out.lidar_select_margin_m, decl);
        ok_bands &= rc::manifest::band_contains_body("table ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] table ✓ every derived z-band contains the declared body\n");
    }

    return out;
}

}  // namespace rc
