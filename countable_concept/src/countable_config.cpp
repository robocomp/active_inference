/*
 * countable_config.cpp  —  fill CountableConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in countable_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "countable_config.h"
#include "countable_belief.h"   // CountableEngineModel::kProportions (K2a-L)
#include "countable_sections.h"   // rc::refuse_pre_countable_config
#include "countable_config_error.h"   // rc::CountableConfigError — every refusal below throws it (main exits non-zero)

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <print>

#include "../../common/concept_manifest/concept_manifest.h"
#include "../../common/concept_manifest/class_prior.h"   // rc::manifest (SHARED)
#include "../../common/concept_manifest/wall_standoff.h"   // rc::manifest::wall_standoff (F1.4)
#include "../../common/concept_manifest/permanence.h"      // rc::manifest::permanence (F1 round 6)
#include <string>

#include <genericworker.h>   // ConfigLoader

namespace rc {

CountableConfig load_countable_config(const ConfigLoader& cfg)
{
    CountableConfig out;

    // ── A config written for a pre-countable clone ([BedModel], [NightstandConcept], …) is refused before anything is
    // read: every key here is optional, so it would otherwise start on code defaults with every tuned value ignored.
    rc::refuse_pre_countable_config(cfg);
    rc::refuse_bad_beam_config(cfg);   // eps_px in (0,1/2) and p0 c^2 > 1 when LidarMeasuresOnGated is on

    // ── Identity first: the object class is a run-time string (Concept.Class), and everything below that
    // names the class — the manifest path, the registry/log label — derives from it. Agent.name is read raw
    // (genericworker reads it too) because the registering Reader needs it as its label.
    try { out.agent_name = cfg.get<std::string>("Agent.name"); }
    catch (...) {}
    if (out.agent_name.empty())
    {
        throw CountableConfigError("countable_concept: ★ config has no [Agent] name — cannot start\n");
    }
    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, out.agent_name);
    out.concept_class = reader.s("Concept.Class", "",
            "REQUIRED object class (YOLO label, node-name prefix, object_subtype, manifest name), e.g. \"bed\"");
    if (out.concept_class.empty())
    {
        throw CountableConfigError(std::format("{}: ★ [Concept] Class is missing or empty — countable_concept needs the object class "
                           "(e.g. Class = \"bed\") and has no default. Refusing to start.\n", out.agent_name));
    }

    // The one place this path is written. It is relative to the agent's CWD (<agent>/), not to src/.
    const std::string manifest_path = countable_manifest_path(out.concept_class);
    // ★★AN INHERITED WORLD FACT IS FATAL. The manifest has carried `from = measured|fitted|nominal|inherited`
    // since 2026-08-11, with a note saying "inherited is worse than absent — an absent value gets asked
    // about, an inherited one gets trusted". A note stopped nothing: hood_concept shipped TEN cloned defects
    // that week, several of them DECLARED as inherited, in writing, in this very file. The declaration is
    // now the enforcement — see rc::manifest::provenance_ok. Refusing to start is the whole point: a comment
    // cannot fail a build, and everything that only warned was fixed around rather than fixed.
    if (not rc::manifest::provenance_ok(manifest_path, out.concept_class))   // it has printed why
        throw CountableConfigError(std::format("{}: ★ the class manifest {} fails its provenance check — refusing to start\n",
                                               out.agent_name, manifest_path));

    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("Concept.StateEps", 0.04f,
            "Σ|Δstate| convergence threshold between cycles (m+rad)");
    out.K_stable                 = geti("Concept.KStable", 30,
            "consecutive converged cycles before model_stable");
    out.detection_alive_max_frames = geti("Concept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh mask before detection_alive=false");
    out.matched_frames_before_aging = geti("Concept.MatchedFramesBeforeAging", 5,
            "a barely-born belief (fewer matched frames than this) is NOT");
    out.central_region_frac      = getf("Concept.CentralRegionFrac", 0.25f,
            "central image box is [frac, 1-frac]×[frac, 1-frac]; a detectable");
    out.epistemic_cooldown_cycles= geti("Concept.EpistemicCooldownCycles", 200,
            "min cycles withdrawn after satisfaction");
    out.countable_log_period_frames  = geti("Concept.LogPeriodFrames", 30,
            "per-cycle log throttle");
    out.support_bank_max_points    = geti("Concept.SupportBankMaxPoints", 4000,
            "cap on the bed-owned support-point memory bank");
    out.support_bank_quantization_m= getf("Concept.SupportBankQuantizationM", 0.02f,
            "support-bank dedup grid (m)");
    out.support_select_radius_margin_m = getf("Concept.SupportSelectRadiusMarginM", 0.50f,
            "XY margin (m) around the model for support-bank selection");
    out.support_select_height_margin_m = getf("Concept.SupportSelectHeightMarginM", 0.25f,
            "Z margin (m) around the model for support-bank selection");

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "Demote Operating→Degraded→Waiting when the retina's `masks` node stops advancing its mask_frame_id for this long (producer dead/stalled) — don't…");
    out.show_dashboard           = getb("Concept.ShowDashboard", true,
            "Show the combined GUI window (belief timeseries dashboard + evidence monitor in one splitter). false ⇒ no GUI windows are built at all (headless);…");
    out.shape_eval_period        = geti("Concept.ShapeEvalPeriod", 30,
            "Every shape_eval_period cycles (once the support bank has ≥ shape_eval_min_points) fit a ROUND model to the accumulated cloud and accumulate a…");
    out.shape_eval_min_points    = geti("Concept.ShapeEvalMinPoints", 300,
            "");
    out.shape_evidence_clamp     = getf("Concept.ShapeEvidenceClamp", 8.0f,
            "");
    out.dump_cloud_path          = gets("Concept.DumpCloudPath", "",
            "DIAGNOSTIC one-shot: if non-empty, dump a fitted bed's accumulated support-bank point cloud (room frame, XYZ per line) to this path ONCE…");

    // ─── CountableModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("Model.SigmaObs", 0.05f,
            "Forwarded to CountableModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top face is attributed to the slab…");
    out.sdf_threshold_for_storage = getf("Model.SdfThresholdForStorage", 0.08f,
            "On-surface membership for the candidate/residual split in CountableFitter::observe");

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("Model.AI2SigmaBaseM", 0.03f,
            "base on-surface obs noise std (m); R = σ² (+ motion_var + …)");
    out.ai2_clutter_frac     = getf("Model.AI2ClutterFrac", 0.10f,
            "ε: prior weight of the uniform clutter mixture component");
    out.ai2_clutter_scale_m  = getf("Model.AI2ClutterScaleM", 0.12f,
            "a point further than ~this from every surface is likely clutter");
    out.ai2_prior_size_std   = getf("Model.AI2PriorSizeStd", 0.30f,
            "(legacy; the size prior is now split into footprint vs height below)");
    out.ai2_prior_footprint_m   = getf("Model.AI2PriorFootprintM", 0.60f,
            "mean of the w & h(depth) prior (m)");
    out.ai2_prior_footprint_std = getf("Model.AI2PriorFootprintStd", 0.08f,
            "TIGHT → strong footprint prior (m)");
    out.ai2_prior_depth_std     = getf("Model.AI2PriorDepthStd", -1.0f,
            "depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std");
    // The back panel (a bed's headboard) is a CLASS fact: it comes from the manifest's [prior.headboard], absent
    // for classes without one (nightstand, toilet) = a plain box. No config key.
    bool manifest_opened = false;
    const auto bp = rc::manifest::load_back_panel(manifest_path, &manifest_opened);
    if (not manifest_opened)
    {
        throw CountableConfigError(std::format("{}: ★ the class manifest {} cannot be opened — refusing to start (a missing manifest "
                           "would silently turn the bed into a plain box)\n", out.agent_name, manifest_path));
    }
    if (bp)
    {
        out.ai2_headboard_h_m = bp->height_m;
        out.ai2_headboard_t_m = bp->thickness_m;
        std::print("[manifest] {} back panel: height {:.2f} m, thickness {:.2f} m\n", out.concept_class, bp->height_m, bp->thickness_m);
    }
    try
    {
        out.shape_parts = rc::manifest::load_shape_parts(manifest_path);
        out.solid_parts = rc::manifest::load_solid_is_parts(manifest_path);
        out.shape_proportions = rc::manifest::load_shape_proportions(manifest_path);
        out.shape_symmetric = rc::manifest::load_shape_symmetry(manifest_path) == rc::manifest::Symmetry::c2v;
        if (out.shape_proportions.size() > std::size_t(rc::CountableEngineModel::kProportions))
            throw std::runtime_error(std::format("{}: {} [[shape.proportion]] declared; the countable engine carries {}", manifest_path,
                                                 out.shape_proportions.size(), rc::CountableEngineModel::kProportions));
    }
    catch (const std::exception& e)
    {
        throw CountableConfigError(std::format("{}: ★ {} — refusing to start\n", out.agent_name, e.what()));
    }
    std::print("[manifest] {} shape: {} part(s); the model's solid is {}{}\n", out.concept_class, out.shape_parts.size(),
               out.solid_parts ? "the parts" : "the whole box (the parts are the carve's)",
               out.shape_symmetric ? "; symmetry c2v (no front: the four yaw modes are one body)" : "");
    try
    {   // The permanence prior (F1 round 6): how often a real object of the class is moved — the volatility's hyperprior.
        if (const auto pm = rc::manifest::permanence(manifest_path))
        {
            out.move_rate_per_day = pm->move_rate_per_day; out.move_scale_m = pm->move_scale_m; out.move_scale_rad = pm->move_scale_rad;
            std::print("[manifest] {} permanence: {:.5f} moves/day, {:.2f} m, {:.2f} rad — {}\n", out.concept_class, pm->move_rate_per_day,
                       pm->move_scale_m, pm->move_scale_rad, pm->source);
        }
        else
            std::print("[manifest] {} declares no [prior.permanence] — q0 = 0 on the persistent path\n", out.concept_class);
    }
    catch (const std::exception& e)
    {
        throw CountableConfigError(std::format("{}: ★ {} — refusing to start\n", out.agent_name, e.what()));
    }
    try
    {   // The wall stand-off (F1.4): a CLASS fact from typical real placement, no config key.
        if (const auto so = rc::manifest::wall_standoff(manifest_path))
        {
            out.wall_standoff_m = so->mean_m;
            out.wall_standoff_std_m = so->std_m;
            std::print("[manifest] {} wall stand-off {:.3f} ± {:.3f} m — {}\n", out.concept_class, so->mean_m, so->std_m, so->source);
        }
        else
            std::print("[manifest] {} declares no [prior.attachment] stand-off — the flush prior keeps its old form\n", out.concept_class);
    }
    catch (const std::exception& e)
    {
        throw CountableConfigError(std::format("{}: ★ {} — refusing to start\n", out.agent_name, e.what()));
    }
    try
    {   // Δ, the class's layer spread (shared-engine G4): a CLASS fact, no config key.
        const auto ls = rc::manifest::layer_spread(manifest_path, 6);
        if (ls.delta_m.empty())
            std::print("[manifest] {} declares no [shape].layer_spread_m — the shape latent is ZERO (no layering)\n", out.concept_class);
        else
        {
            std::ranges::copy(ls.delta_m, out.layer_spread_m.begin());
            out.layer_spread_source = ls.source;
            std::print("[manifest] {} layer spread Δ (cx, cy, H, w, h, yaw) = ({:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}) — {}\n",
                       out.concept_class, ls.delta_m[0], ls.delta_m[1], ls.delta_m[2], ls.delta_m[3], ls.delta_m[4],
                       ls.delta_m[5], ls.source);
        }
    }
    catch (const std::exception& e)
    {
        throw CountableConfigError(std::format("{}: ★ {} — refusing to start\n", out.agent_name, e.what()));
    }
    {   // π_cls: a CLASS fact too. Absent ⇒ 0.8 (said so); present but outside the open (0,1) ⇒ refuse (logit is ±inf).
        const auto pl = rc::manifest::load_class_prior(manifest_path);
        if (pl)
        {
            if (not (pl->p_label_true > 0.0f and pl->p_label_true < 1.0f))
                throw CountableConfigError(std::format("{}: ★ {} [prior.class] p_label_true = {} must lie in the open interval (0,1) "
                                   "(logit is infinite at 0 and 1) — refusing to start\n", out.agent_name, manifest_path, pl->p_label_true));
            for (const float f : {pl->fit_H_m, pl->fit_w_m, pl->fit_h_m})
                if (not (std::isfinite(f) and f >= 0.0f))
                    throw CountableConfigError(std::format("{}: ★ {} [prior.class] fit_error_*_m must be finite and >= 0 (got {}) — refusing to start\n",
                                       out.agent_name, manifest_path, f));
            out.class_p_label_true = pl->p_label_true;
            out.class_fit_H_m = pl->fit_H_m; out.class_fit_w_m = pl->fit_w_m; out.class_fit_h_m = pl->fit_h_m;
            std::print("[manifest] {} class prior: p_label_true {:.3f}, box-fit error (H,w,h) = ({:.3f}, {:.3f}, {:.3f}) m\n",
                       out.concept_class, pl->p_label_true, pl->fit_H_m, pl->fit_w_m, pl->fit_h_m);
        }
        else std::print("[manifest] {} has no [prior.class] p_label_true — using the default {:.2f}\n", out.concept_class, out.class_p_label_true);
    }
    out.class_alt_std_m           = getf("Model.ClassAltStdM", 1.0f,
            "size std (m) of the 'some other furniture' alternative, per size variable, Gaussian about the class mean (class evidence)");
    out.birth_height_noise_m      = getf("Model.BirthHeightNoiseM", 0.05f,
            "birth height noise σ_z (m): the cloud's robust z-extent is a measurement of the object height with this std");
    out.birth_height_outlier_frac = getf("Model.BirthHeightOutlierFrac", 0.05f,
            "birth mask-bleed outlier fraction ε: share of masks whose z-extent is unrelated to the class height (person, wall above)");
    out.ai2_wall_flush_prior    = getf("Model.WallFlushPrior", 0.0f,
            "class probability that the object stands against a wall (furniture prior); 0 = never assumed");
    out.ai2_room_axis_std_deg   = getf("Model.RoomAxisStdDeg", 0.0f,
            "room-axis alignment std (deg), 4-fold; <=0 = off");
    out.ai2_prior_depth_m       = getf("Model.AI2PriorDepthM", -1.0f,
            "depth (local Y, front-to-back) mean (m); <=0 = same as the footprint (square)");
    // ★FALLBACK ALIGNED TO THE MANIFEST (2026-08-17). This read 1.70 while the manifest and config.toml
    // both say 1.90 — invisible while config is present, and a silently shorter bed the moment it is
    // not. A code default that contradicts the declaration is the same defect class as an inherited one.
    out.ai2_prior_height_m      = getf("Model.AI2PriorHeightM", 1.90f,
            "mean of the H (vertical) static anchor (m) — a standard tall bed");
    out.ai2_prior_height_std    = getf("Model.AI2PriorHeightStd", 0.50f,
            "anchor std (m); stops the box top floating above the cloud");
    out.ai2_depth_unobs_precision = getf("Model.AI2DepthUnobsPrecision", 1500.0f,
            "extra 1/m² when depth extent is unobserved (single face)");
    out.ai2_depth_obs_band_m      = getf("Model.AI2DepthObsBandM", 0.10f,
            "ly-spread (m) below which the cloud is a single depth face");
    out.ai2_top_no_float_precision = getf("Model.AI2TopNoFloatPrecision", 10000.0f,
            "firm two-sided anchor pinning H → observed robust cloud top (1/m²)");
    out.ai2_top_no_float_margin_m  = getf("Model.AI2TopNoFloatMarginM", 0.02f,
            "upward allowance (m) added to the observed top as the anchor target");
    out.ai2_top_overseg_sigma_per_m = getf("Model.AI2TopOversegSigmaPerM", 2.0f,
            "R inflation (σ per m) for points ABOVE the robust top — fades the");
    out.ai2_wall_precision          = getf("Model.AI2WallPrecision", 400.0f,
            "1/m² at zero gap (back face → wall)");
    out.ai2_wall_parallel_precision = getf("Model.AI2WallParallelPrecision", 200.0f,
            "back face parallel to the wall (width-axis · wall-normal → 0)");
    out.ai2_wall_reach_m            = getf("Model.AI2WallReachM", 0.15f,
            "gap scale over which the flush hypothesis loses its weight");
    out.ai2_door_clearance_gain     = getf("Model.AI2DoorClearanceGain", 3.0f,
            "WALL as a competing per-point explanation (explaining away — see CountableBeliefParams). π_wall and the extra wall-plane std. 0 = OFF (mixture…");
    out.detect_min_fill             = getf("Model.DetectMinFill", 0.10f,
            "INFERRED VOLATILITY (MODEL_HISTORY.md §3) — the HISTORY stage");
    out.detect_max_fill             = getf("Model.DetectMaxFill", 0.60f,
            "");
    out.detect_soft                 = getf("Model.DetectSoft", 0.06f,
            "");
    out.ai2_volatility_infer        = getb("Model.AI2VolatilityInfer", false,
            "ON 08-03 for the A/B tour. Watch omega_w/tau_w in ai2_log.csv");
    out.ai2_volatility_lr           = getf("Model.AI2VolatilityLr", 0.02f,
            "gradient step on ω per MEASURED cycle (predict-only cycles never touch it)");
    out.ai2_volatility_sigma        = getf("Model.AI2VolatilitySigma", 2.0f,
            "hyperprior width σ_ω: how far evidence may pull ω from the configured Q");
    out.ai2_wall_explain_frac       = getf("Model.AI2WallExplainFrac", 0.25f,
            "π_wall: prior weight of 'this point belongs to a known room wall'");
    out.ai2_wall_explain_sigma_m    = getf("Model.AI2WallExplainSigmaM", 0.05f,
            "extra wall-plane std (m): room-polygon + localization slack (adds to WallRef σ)");
    out.ai2_wall_no_cross_precision = getf("Model.WallNoCrossPrecision", 2000.0f,
            "1/m² per m of penetration past the wall (>> the flush precision); persistent path: on/off only, λ = 1/σ_wall²");
    out.ai2_wall_no_cross_margin_m  = getf("Model.WallNoCrossMarginM", 0.0f,
            "interior offset (m) inside which the constraint activates");
    out.ai2_process_std_m    = getf("Model.AI2ProcessStdM", 0.005f,
            "predict process-noise std, length DOFs (m/frame)");
    out.ai2_process_std_yaw  = getf("Model.AI2ProcessStdYaw", 0.01f,
            "predict process-noise std, yaw (rad/frame)");
    out.ai2_age_nominal_dt_s = getf("Model.AI2AgeNominalDtS", 0.0f,
            "Stale-belief aging (measurement-age → covariance)");
    out.ai2_common_mode_pos_std  = getf("Model.AI2CommonModePosStd", 0.03f,
            "shared position error (m); pose-chain cov adds to it");
    out.ai2_common_mode_size_std = getf("Model.AI2CommonModeSizeStd", 0.02f,
            "shared size error w,h,H (m)");
    out.ai2_common_mode_yaw_std  = getf("Model.AI2CommonModeYawStd", 0.03f,
            "shared yaw error (rad)");
    out.motion_cm_pos_gain       = getf("Model.MotionCmPosGain", 0.10f,
            "position (cx,cy) shared-error std per m/s of motion_dotd");
    out.motion_cm_size_gain      = getf("Model.MotionCmSizeGain", 0.20f,
            "extent (w,h,H) shared-error std per m/s — the anti-RESHAPE lever");
    out.motion_cm_yaw_gain       = getf("Model.MotionCmYawGain", 0.12f,
            "yaw shared-error std (rad) per m/s — the anti-ROTATE lever");
    out.ai2_ang_lever_m           = getf("Model.AI2AngLeverM", 2.0f,
            "rad/s → m/s lever (tangential speed of a bed ~this far away) for ego-motion");
    out.ai2_periph_ref            = getf("Model.AI2PeriphRef", 0.50f,
            "centroid radius (focal-norm) at which the off-axis periphery penalty → 1");
    out.ai2_motion_ref_mps        = getf("Model.AI2MotionRefMps", 0.60f,
            "motion magnitude (m/s) at which a fully-peripheral frame is fully unreliable");
    out.ai2_motion_confirm_only   = getb("Model.AI2MotionConfirmOnly", true,
            "Discrete confirm-only gate ('only update if the robot is still'): true ⇒ the fitter CONFIRMS but does NOT move the geometry mean (predict-only…");
    out.ai2_still_lin_mps         = getf("Model.AI2StillLinMps", 0.05f,
            "camera linear speed (m/s) below which the robot counts as 'still'");
    out.ai2_still_ang_radps       = getf("Model.AI2StillAngRadps", 0.10f,
            "camera angular speed (rad/s) below which the robot counts as 'still'");
    out.ai2_still_dotd            = getf("Model.AI2StillDotd", 0.05f,
            "per-mask ego-motion corruption speed (m/s) still-level");
    out.ai2_moving_update_center_radius = getf("Model.AI2MovingUpdateCenterRadius", 0.35f,
            "mask centroid radius below which a moving update is still allowed");
    out.ai2_range_noise_lat_per_m = getf("Model.AI2RangeNoiseLatPerM", 0.02f,
            "lateral deprojection std growth (m per m of range)");
    out.ai2_range_noise_yaw_per_m = getf("Model.AI2RangeNoiseYawPerM", 0.03f,
            "yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)");
    out.ai2_range_noise_size_per_m = getf("Model.AI2RangeNoiseSizePerM", 0.08f,
            "SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)");
    out.ai2_trunc_gate_frac    = getf("Model.AI2TruncGateFrac", 0.10f,
            "mask can't reshape/inflate a converged bed (freezes geometry afar) THRESHOLD (truncation gate): skip the geometric update (predict only)…");
    out.ai2_gn_iters         = geti("Model.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_csv_path         = gets("Model.AI2CsvPath", "",
            "if non-empty, append per-cycle belief (state + Σ diag + mask R) to CSV");
    out.detect_probe_csv_path = gets("Concept.DetectProbeCsvPath", out.detect_probe_csv_path,
            "The DETECTOR's truth table (rc::probe) — one row per cycle per instance: viewpoint, framing, and what YOLO did about it");
    out.birth_surprise_probe = getb("Model.BirthSurpriseProbe", false,
            "EXPERIMENTAL birth-surprise probe (read-only): read residual_concept's `grid` node as an unexplained- occupancy (surprise) field, cluster it, and LOG…");
    out.pixel_sigma_over_f     = getf("Model.PixelSigmaOverF", 0.0015f,
            "(footprint-residual path — OFF)");
    out.depth_sigma0_m         = getf("Model.DepthSigma0M", 0.006f,
            "depth std floor (m)");
    out.depth_sigma_range_coef = getf("Model.DepthSigmaRangeCoef", 0.004f,
            "depth std growth (m per m² of range)");
    out.model_sigma_m          = getf("Model.ModelSigmaM", 0.010f,
            "residual model/surface std (m) added in quadrature (R floor)");
    out.footprint_residual     = getb("Model.FootprintResidual", false,
            "a1′+a2′: weighted 2-D footprint residual + shared depth-affine nuisance (replaces the moment channel; the real grazing-yaw fix). depth_bias/scale_std…");
    out.quotient_chart         = getb("Model.QuotientChart", false,
            "C2v symmetry-quotient chart (PRECISION_AS_INFORMATION.md Stage 3): optimise the footprint in [s,a₁,a₂] so the 4 box representatives collapse to one…");
    out.depth_tilt_std         = getf("Model.DepthTiltStd", 0.020f,
            "shared per-frame depth tilt prior std (m/rad) — the yaw nuisance");
    out.depth_bias_std         = getf("Model.DepthBiasStd", 0.015f,
            "shared per-frame depth bias prior std (m)");
    out.depth_scale_std        = getf("Model.DepthScaleStd", 0.010f,
            "shared per-frame depth scale prior std (fraction)");

    // ─── "Is this really a bed?" plausibility filter + soft singleton ───────
    out.countable_filter_enabled   = getb("Concept.FilterEnabled", true,
            "master switch for the whole plausibility filter");
    out.plaus_aspect_scale      = getf("Model.AspectScale", 0.15f,
            "aspect = |w−h|/(w+h); aspect_ok = exp(−(aspect/scale)²) → 1 square");
    out.plaus_size_scale        = getf("Model.SizeScale", 0.15f,
            "size_ok = exp(−((w−fp)²+(h−fp)²)/(2·scale²)), fp = prior_footprint_m");
    out.plaus_height_min        = getf("Model.HeightPlausibleMin", 1.20f,
            "logistic centre (m): height_ok ≈ 0.5 here, →1 tall, →0 below ~1 m");
    out.plaus_height_soft       = getf("Model.HeightSoft", 0.15f,
            "logistic softness (m) of the height_ok falloff");
    out.plaus_fe_ref            = getf("Model.FeRef", 2.0f,
            "'healthy bed fit' free-energy reference (NEEDS LIVE TUNING)");
    out.plaus_fe_scale          = getf("Model.FeScale", 1.0f,
            "fit_ok = exp(−max(0, FE−fe_ref)/fe_scale)");
    out.plaus_height_prior_gain = getf("Model.PlausHeightPriorGain", 2000.0f,
            "precision (1/m²) per unit (deficit/soft) below plaus_height_min");
    out.singleton_inhibition    = getf("Model.SingletonInhibition", 1.0f,
            "mutual-inhibition weight: a STRONGER bed subtracts this·P(exists) from a weaker one's existence L");
    out.countable_filter_log       = getb("Concept.FilterLog", false,
            "log per-instance plausibility / evidence / existence-delta each cycle");

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("Concept.RtCovScale", 1.0f,
            "Upload the bed pose covariance onto the room→bed RT edge (rt_covariance_att, 6×6 SE3), mapped from the belief's full Σ over…");
    out.publish_object_obs            = getb("Concept.PublishObjectObs", false,
            "Concept.PublishObjectObs");
    out.object_obs_frame              = gets("Concept.ObjectObsFrame", "body",
            "Concept.ObjectObsFrame (localizer base node)");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "χ²₂ gate (~3σ) for a mask↔instance match once it has a cov");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.50f,
            "metric XY gate (m) before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f,
            "Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I)");
    out.recall_pi_move = getf("Tracker.RecallPiMove", 0.05f,
            "Mixture weight of 'the remembered bed was moved' in the recall location likelihood (class prior; GUESS, beds almost never move)");
    out.recall_log_prior_new = getf("Tracker.RecallLogPriorNew", 0.0f,
            "Log prior of 'new bed' against one recalled candidate (0 = equal)");
    // K2b: the consumers' channels the legacy chair agent published (defaults = the pre-K2b behaviour of every other class)
    out.mesh_path = gets("Concept.MeshPath", "",
            "display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)");
    out.look_attrs = gets("Concept.LookAttrs", "table",
            "typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair "
            "(chair_roi_*, chair_detection_*, chair_front_seen*)");
    if (out.look_attrs != "table" and out.look_attrs != "chair")
        throw CountableConfigError(std::format("{}: Concept.LookAttrs = \"{}\": not table or chair — refusing to start\n",
                                               out.agent_name, out.look_attrs));
    out.appearance_enabled = getb("Concept.AppearanceEnabled", false,
            "colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)");
    out.publish_exist_prob = getb("Concept.PublishExistProb", false, "publish obj_exist_prob = P(exists) = occupancy x class");
    out.thumbnails_enabled = getb("Concept.ThumbnailsEnabled", false,
            "keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)");
    out.approach_enabled = getb("Concept.ApproachEnabled", false, "offer the pragmatic approach affordance aff_<name>_approach");
    out.epistemic_offer_enabled = getb("Concept.EpistemicOfferEnabled", true,
            "false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen");
    out.changed_object_enabled = getb("Concept.ChangedObjectEnabled", false,
            "changed object: ghost existence at remembered spots + the same_as identity link on objects born new");
    out.link_pi_move = getf("Tracker.LinkPiMove", 0.10f, "identity link: per-visit P(a remembered object was moved) (a GUESS)");
    out.link_pi_gone = getf("Tracker.LinkPiGone", 0.10f, "identity link: per-visit P(a remembered object left the room) (a GUESS)");
    out.link_lambda_new = getf("Tracker.LinkLambdaNew", 0.10f, "identity link: per-visit rate of a NEW object arriving (a GUESS)");
    out.ghost_evidence_gain = getf("Tracker.GhostEvidenceGain", 0.15f,
            "changed object: |dL| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)");
    out.tracker_birth_frames     = geti("Tracker.BirthFrames", 8,
            "frames a mask must stay unexplained before spawning a bed");
    out.birth_fusion             = getb("Tracker.BirthFusion", false,
            "FUSED BIRTH (EXPERIMENTAL, off by default): let residual-grid SURPRISE MASS under a detection accelerate its birth");
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain", 6.0f,
            "max extra evidence/frame at full corroboration (0 ⇒ baseline)");
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef", 8.0f,
            "residual mass at which corroboration is half-saturated (m/(m+ref))");
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM", 0.50f,
            "window (m) for the residual-mass sample under the detection");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.60f,
            "a birth must be ≥ this (m) from every existing bed");
    out.birth_self_body          = getb("Tracker.BirthSelfBody", true,
            "SELF-BODY: scale a detection's birth evidence by the fraction of its points OUTSIDE the robot's own body (common/self_body/self_body.h)");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.05f,
            "Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of the SMALLER footprint (two beds cannot…");
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM", 1.0f,
            "Default geometry for a bed BORN from a mask, used only when no birth burst is available (see birth_frag_* below); the belief refines from…");
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM", 0.6f,
            "");
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM", 0.75f,
            "");
    // Birth fragment: keep the probation burst and admit the birth on it (see countable_config.h).
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
            "extra angular tolerance on the bed's angular half-size");
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM", 1.0f,
            "extra range band (ricoh range is rough/indicative)");

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("Model.LidarPrecision", 0.0f,
            "per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF");
    out.lidar_bpearl_precision = getf("Model.LidarBpearlPrecision", 0.0f,
            "Low 'bpearl' LiDAR as a SEPARATE per-device ray-set (own origin, occlusion-aware first-hit; sees the legs the high helios grazes over)");
    out.lidar_robust_c_m     = getf("Model.LidarRobustCM", 0.05f,
            "Cauchy scale (m): returns this far off the surface fade out");
    out.lidar_select_margin_m = getf("Model.LidarSelectMarginM", 0.10f,
            "pre-select returns within (birth half-extent + margin), all z up to top");
    out.lidar_coverage_n0     = getf("Model.LidarCoverageN0", 60.0f,
            "LiDAR ray count for FULL weight; fewer → proportionally down-weighted");
    out.lidar_coverage_ang_power = getf("Model.LidarCoverageAngPower", 1.0f,
            "Angular-coverage weighting: precision ×= (1−R)^p, R = mean-resultant length of return bearings about the centre. p=0 disables (flat), p=1 = pure…");
    out.max_step_m            = getf("Model.MaxStepM", 1.0f,
            "THRESHOLD (outlier step guard): a static bed cannot physically move this far in one frame, so a GN step whose centre jump exceeds it is an…");
    out.coverage_precision    = getf("Model.CoveragePrecision", 0.0f,
            "Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a large mask grows to explain it (fixes…");
    out.coverage_robust_c_m   = getf("Model.CoverageRobustCM", 0.15f,
            "Cauchy scale (m): on-plane points this far outside fade out (stray-point guard)");
    out.free_space_precision  = getf("Model.FreeSpacePrecision", 0.0f,
            "The counter-force that BOUNDS coverage");
    out.footprint_moment_precision = getf("Model.FootprintMomentPrecision", 0.0f,
            "Measures (w,h,yaw) from the top-band cloud's 2D inertia tensor and folds it as a linear Gaussian factor — the escape from the clutter-trap that…");
    out.footprint_moment_range_per_m = getf("Model.FootprintMomentRangePerM", 0.03f,
            "GENTLE range term (m per m of range): shared per-frame moment variance grows as (this·range)²");
    out.fe_baseline_adapt_down       = getf("Model.FeBaselineAdaptDown", 0.05f,
            "Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the bed moved — stays surprising long enough to…");
    out.fe_baseline_adapt_up         = getf("Model.FeBaselineAdaptUp", 0.005f,
            "");
    out.fe_surprise_smooth           = getf("Model.FeSurpriseSmooth", 0.10f,
            "");
    out.footprint_moment_motion_gain = getf("Model.FootprintMomentMotionGain", 0.30f,
            "Ego-motion (motion_dotd) coupling: moment variance grows by (this·motion_dotd)² so a 'going-away/rotation' frame (degraded/split mask) can't reshape…");
    out.orientation_motion_ref       = getf("Model.OrientationMotionRef", 0.50f,
            "");

    // ─── Appearance-based FRONT (door) detection + yaw resolver ────────────────
    out.front_detect_enabled   = getb("Concept.FrontDetectEnabled", true,
            "master switch: subscribe to ZED RGB + run detect_front/resolve_front");
    out.contour_check_enabled  = getb("Concept.ContourCheckEnabled", true,
            "The classifier-free existence channel: score the believed silhouette against the RGB's own gradient and against the DEPTH plane's agreement with the…");
    out.front_min_face_area_px = getf("Concept.FrontMinFaceAreaPx", 900.0f,
            "min projected face area (px²) to score for door-ness");
    out.front_min_confidence   = getf("Concept.FrontMinConfidence", 0.10f,
            "min door-ness margin (max−second)/(max+eps) for detect_front to emit");
    out.front_log              = getb("Concept.FrontLog", false,
            "log each adopted door-mode flip / accepted cue");
    out.obliquity_moment_gain        = getf("Model.ObliquityMomentGain", 0.0f,
            "Grazing-view yaw stability (CSV rogue-rotation fix)");
    out.footprint_moment_completeness_gain = getf("Model.FootprintMomentCompletenessGain", 0.0f,
            "footprint_moment_completeness_gain / _min_completeness: forwarded to CountableBeliefParams — inflate the moment measurement variance as the…");
    out.footprint_moment_min_completeness  = getf("Model.FootprintMomentMinCompleteness", 0.02f,
            "clamp floor for 1/completeness in the term above (caps the inflation of");

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("Model.ExistenceRemovalEnabled", false,
            "Each cycle carve the LiDAR sweep against the bed footprint → occupancy/free-space log-odds; remove when P(occupied) < removal_prob");
    out.existence_carve_parts     = getb("Model.ExistenceCarveParts", false,
            "LiDAR existence carve against the class's composed parts (shape.part + back panel) instead of the solid fitted box");
    out.lidar_measures_on_gated   = getb("Model.LidarMeasuresOnGated", false,
            "camera attributes, LiDAR measures (spec 2026-11-15 lidar-measures §7c): persistent shared errors + class prior once, mask-attributed LiDAR-only updates on gated cycles");
    out.gn_trace                  = geti("Model.GnTrace", 0,
            "DIAGNOSTIC only, no model effect: 1 = write every persistent GN step to <AI2CsvPath stem>_gn_steps.csv and, on updates that end unconverged, a finite-difference check of each channel's gradient/curvature to <stem>_gn_fd.csv; 2 = the check on every update (slow)");
    out.replay_dump               = getb("Model.ReplayDump", false,
            "DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persistent belief's seed and every call that changes it (each update's complete frame, ~0.6 MB for a bed) to etc/replay/<node>_<run>_<serial>.bin; replay it offline with countable_concept/tests/build/replay");
    out.label_dump_enabled        = getb("Model.LabelDumpEnabled", false,
            "measurement only (spec 7c.8 a0): with LidarMeasuresOnGated write label_dump.csv, the camera label inputs of every Helios return in the select box");
    out.label_dump_margin_m       = getf("Model.LabelDumpMarginM", 0.0f,
            "measurement only: metres the label dump's select box (XY radius and z-band) is grown by; the fit's selection is unchanged");
    out.mask_pixel_error_rate     = reader.d("Model.MaskPixelErrorRate", 0.0,
            "eps_px (spec 7c.8 a0): the camera mask's measured pixel-error rate, per class; the persistent label is m = eps_px + (1-2 eps_px) q. No default: required, in (0, 1/2), when LidarMeasuresOnGated");
    out.floor_sigma_m             = reader.d("Model.FloorSigmaM", 0.0,
            "the floor's known-surface sigma (spec 7c.8 rev 3.3.8): the floor in the beam model's foreign density p_f. No default: required, > 0, when LidarMeasuresOnGated");
    out.existence_removal_prob    = getf("Model.ExistenceRemovalProb", 0.12f,
            "decision boundary: remove when L < log(p/(1−p))");
    out.existence_out_of_room_gain = getf("Model.ExistenceOutOfRoomGain", 1.5f,
            "|dL| per evidence cycle while the bed centre is outside the room outline by more than half its short side");
    out.existence_frame_correlation = getf("Model.ExistenceFrameCorrelation", 0.0f,
            "rho: frame-to-frame correlation of MISSES. 0 = independent trials (the historic behaviour and the default); opt in only with a value MEASURED from…");
    out.existence_logodds_max     = getf("Model.ExistenceLogoddsMax", 4.0f,
            "clamp |L| so evidence stays finite AND recoverable");
    out.existence_detection_prob  = getf("Model.ExistenceDetectionProb", 0.85f,
            "P(beam through OCCUPIED footprint returns from it)");
    out.existence_clutter_prob    = getf("Model.ExistenceClutterProb", 0.05f,
            "P(beam through EMPTY footprint returns anyway) — spurious rate");
    out.existence_sensor_sigma_m  = getf("Model.ExistenceSensorSigmaM", 0.03f,
            "LiDAR range σ (m) for the soft occ/free surface split");
    out.existence_remove_frames   = geti("Model.ExistenceRemoveFrames", 15,
            "debounce: require the removal decision this many consecutive");
    out.existence_absence_range_ref_m = getf("Model.ExistenceAbsenceRangeRefM", 2.5f,
            "range (m) below which absence is trusted at full weight");
    out.existence_absence_range_power = getf("Model.ExistenceAbsenceRangePower", 2.0f,
            "decay exponent (2 ≈ angular-area ∝ 1/range²); 0 disables");
    out.existence_verify_surprise     = getf("Model.ExistenceVerifySurprise", 20.0f,
            "decayed go-verify surprise (un-resolvable absence) above which");
    out.verify_surprise_smooth        = getf("Model.VerifySurpriseSmooth", 0.10f,
            "EMA weight of the go-verify surprise accumulator (new sample");
    out.existence_verify_gain         = getf("Model.ExistenceVerifyGain", 5.0f,
            "epistemic gain (nats) a wants_verification bed gets, so the");

    if (out.shape_symmetric)
        if (const std::string v = symmetric_config_violation(out); not v.empty())
            throw CountableConfigError(std::format("{}: ★ {} declares [shape] symmetry = \"c2v\" but {} — a term that is not invariant "
                                                   "under the class's symmetry would tell its four yaw modes apart — refusing to start\n",
                                                   out.agent_name, manifest_path, v));
    std::print("{}: configuration loaded (class '{}').\n", out.agent_name, out.concept_class);
    // Peripheral (ricoh) existence confirmation — see countable_config.h. OFF by default.
    out.ricoh_confirm_enabled        = getb("Concept.RicohConfirmEnabled", out.ricoh_confirm_enabled,
            "Concept.RicohConfirmEnabled");
    out.ricoh_confirm_detection_prob = getf("Concept.RicohConfirmDetectionProb", out.ricoh_confirm_detection_prob,
            "Concept.RicohConfirmDetectionProb");
    out.ricoh_confirm_clutter_prob   = getf("Concept.RicohConfirmClutterProb", out.ricoh_confirm_clutter_prob,
            "Concept.RicohConfirmClutterProb");

    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★z_top is the AI2 HEIGHT PRIOR (1.90), NOT Tracker.BirthHeightM. Birth height is deliberately 0.75 —
    // below plaus_height_min so every real bed grows INTO its size — so it is the wrong number for a
    // class-level span. Getting that backwards would declare a body half the height of the object.
    // The support is the manifest's (K1): floor_anchored keeps the span [0, H]; hangs takes the declared extent below the top
    // (the hood), and every z-band the agent derives asks CountableBeliefParams::span() for it. Any other support is refused:
    // this binary has no model for it, and running it as floor-anchored would be the clone defect the manifest exists to stop.
    {
        const auto g = rc::manifest::load_geometry(manifest_path, out.concept_class);
        const auto support = g.valid ? g.support : rc::manifest::Support::floor_anchored;
        if (support == rc::manifest::Support::hangs)
        {
            if (not (g.extent_m > 0.0f and g.extent_m < out.ai2_prior_height_m))
                throw CountableConfigError(std::format("{}: ★ {} declares support = hangs with extent_m = {} — it must lie in (0, the "
                                                       "height prior {}) — refusing to start\n", out.agent_name, manifest_path,
                                                       g.extent_m, out.ai2_prior_height_m));
            out.vertical_extent_m = g.extent_m;
        }
        else if (support != rc::manifest::Support::floor_anchored)
            throw CountableConfigError(std::format("{}: ★ {} declares support = {} — countable_concept models floor_anchored and "
                                                   "hangs only — refusing to start\n", out.agent_name, manifest_path,
                                                   rc::manifest::support_name(support)));
        const float extent = out.vertical_extent_m > 0.0f ? out.vertical_extent_m : out.ai2_prior_height_m;
        const auto span = rc::manifest::adopt_span(manifest_path, out.concept_class, support,
                                                  out.ai2_prior_height_m, extent);
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = out.ai2_prior_height_m;
        decl.extent_m = extent;
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body(out.concept_class + " ", "lidar_select",
                        span.z0 - out.lidar_select_margin_m, span.z1 + out.lidar_select_margin_m, decl);
        ok_bands &= rc::manifest::band_contains_body(out.concept_class + " ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] {} ✓ every derived z-band contains the declared body\n", out.concept_class);
    }

    return out;
}

std::string symmetric_config_violation(const CountableConfig& c)
{
    if (c.ai2_prior_depth_m > 0.0f and c.ai2_prior_depth_m != c.ai2_prior_footprint_m)
        return std::format("Model.AI2PriorDepthM = {} differs from AI2PriorFootprintM = {}", c.ai2_prior_depth_m, c.ai2_prior_footprint_m);
    if (c.ai2_prior_depth_std > 0.0f and c.ai2_prior_depth_std != c.ai2_prior_footprint_std)
        return std::format("Model.AI2PriorDepthStd = {} differs from AI2PriorFootprintStd = {}", c.ai2_prior_depth_std,
                           c.ai2_prior_footprint_std);
    if (c.ai2_wall_flush_prior > 0.0f) return std::format("Model.WallFlushPrior = {} (the flush puts the +y face on a wall)", c.ai2_wall_flush_prior);
    if (c.ai2_wall_precision > 0.0f)
        return std::format("Model.AI2WallPrecision = {} (the wall flush / stand-off acts on the +y face)", c.ai2_wall_precision);
    if (c.ai2_wall_parallel_precision > 0.0f)
        return std::format("Model.AI2WallParallelPrecision = {} (the +y face parallel to the wall)", c.ai2_wall_parallel_precision);
    if (c.ai2_wall_no_cross_precision > 0.0f)
        return std::format("Model.WallNoCrossPrecision = {} (the +y face may not cross the wall)", c.ai2_wall_no_cross_precision);
    if (c.wall_standoff_std_m > 0.0f)
        return std::format("[prior.attachment] standoff_std_m = {} (a stand-off of the back face)", c.wall_standoff_std_m);
    if (c.class_fit_w_m != c.class_fit_h_m)
        return std::format("[prior.class] fit_error_w_m = {} differs from fit_error_h_m = {}", c.class_fit_w_m, c.class_fit_h_m);
    if (c.front_detect_enabled) return "Concept.FrontDetectEnabled = true (a front cue)";
    if (c.approach_enabled) return "Concept.ApproachEnabled = true (a fronted approach)";
    if (c.thumbnails_enabled) return "Concept.ThumbnailsEnabled = true (thumbnails binned on the yaw)";
    return {};
}

// ─── Concept-manifest cross-check (declarative-priors experiment, step 1½) ────────────────────────
//
// common/concept_manifest/<concept>.concept.toml declares WHAT a bed IS — its priors as world facts,
// separate from the lifecycle knobs. It is not authoritative yet. This compares it against the priors the live
// etc/config.toml actually produced, so we learn whether a manifest can reproduce the running agent BEFORE
// anything is generated from it. Startup-only, read-only: it never changes a value, it only reports.
//
// A DIFFERS line is a finding, not a failure — it means the manifest and the running config disagree about a
// world fact, and one of them is wrong. A MISSING line means the manifest does not yet describe that prior.
bool verify_countable_manifest(const CountableConfig& out, const std::string& path)
{
    toml::table man;   // toml++, not ConfigLoader: the manifest carries arrays of tables (rc::manifest::parse_manifest)
    try { man = rc::manifest::parse_manifest(path); }
    catch (const std::exception& e) { std::print("[manifest] not loaded ({}) — cross-check skipped: {}\n", path, e.what()); return false; }

    int agree = 0, differ = 0, missing = 0;
    const auto chk = [&](const char* key, float live, const char* what) {
        const auto v = man.at_path(key).value<double>();
        if (not v) { ++missing;
            std::print("[manifest] MISSING  {:<42} live={:<10.4g} ({})\n", key, live, what); return; }
        const float m = static_cast<float>(*v);
        const float tol = 1e-4f * std::max(1.0f, std::abs(live));
        if (std::abs(m - live) <= tol) { ++agree; }
        else { ++differ;
            std::print("[manifest] DIFFERS  {:<42} manifest={:<10.4g} live={:<10.4g} ({})\n", key, m, live, what); }
    };

    chk("prior.footprint.mean_m",              out.ai2_prior_footprint_m,      "footprint mean");
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
