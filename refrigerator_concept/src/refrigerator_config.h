/*
 * refrigerator_config.h  —  plain-data configuration for the refrigerator_concept agent + its ConfigLoader loader.
 *
 * Kept separate from SpecificWorker so a new concept agent can copy this file and edit only the keys it
 * needs (mirrors bottle_concept/bottle_config.h). Every field carries its default here; the loader
 * (refrigerator_config.cpp) overrides from the RoboComp TOML when the key is present.
 */

#pragma once

#include <cstdint>
#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct RefrigeratorConfig
{
    // ── Agent convergence & cadence ───────────────────────────────────────────────────────────────
    float state_eps                    = 0.04f;   // Σ|Δstate| convergence threshold between cycles (m+rad)
    int   K_stable                     = 30;      // consecutive converged cycles before model_stable
    int   detection_alive_max_frames   = 40;      // cycles without a fresh mask before detection_alive=false
    int   matched_frames_before_aging  = 5;       // a barely-born belief (fewer matched frames than this) is NOT
                                                  // aged by a stale no-mask cycle — protects it from decaying before
                                                  // it has ever been fit. Small integer; not a belief threshold.
    float central_region_frac          = 0.25f;   // central image box is [frac, 1-frac]×[frac, 1-frac]; a detectable
                                                  // sample inside it counts toward central_frac → p_detect → removal
    int   epistemic_cooldown_cycles    = 200;     // min cycles withdrawn after satisfaction
    int   refrigerator_log_period_frames      = 30;      // per-cycle log throttle
    int   voxel_bank_max_points        = 4000;    // cap on the refrigerator-owned voxel memory bank
    float voxel_bank_quantization_m    = 0.02f;   // voxel-bank dedup grid (m)
    float voxel_select_radius_margin_m = 0.50f;   // XY margin (m) around the model for voxel-bank selection
    float voxel_select_height_margin_m = 0.25f;   // Z margin (m) around the model for voxel-bank selection

    // ── Primary-input stream gate (readiness + staleness) — LIFECYCLE, not a belief knob ──────────────
    // Demote Operating→Degraded→Waiting when the voxelizer's `masks` node stops advancing its mask_frame_id
    // for this long (producer dead/stalled) — don't integrate stale evidence; re-admit when it returns.
    // Orthogonal to ai2_age_nominal_dt_s (belief-axis Σ-aging). MUST exceed the voxelizer HOLD_ENTER_S so a
    // legitimately empty scene (still-advancing counter) never trips it. 0 = disable the gate.
    int   masks_stall_timeout_ms       = 3000;

    // Show the combined GUI window (belief timeseries dashboard + evidence monitor in one splitter).
    // false ⇒ no GUI windows are built at all (headless); the compute feed no-ops on the null widgets.
    bool  show_dashboard         = true;

    // ── Shape model-selection (round vs square) — free-energy evidence, no threshold ──────────────────
    // Every shape_eval_period cycles (once the voxel bank has ≥ shape_eval_min_points) fit a ROUND model to
    // the accumulated cloud and accumulate a bounded log-Bayes-factor (round − square) → inst.subtype. The
    // accumulator is clamped to ±shape_evidence_clamp so a converged run can still RECANT if evidence turns.
    int   shape_eval_period      = 30;
    int   shape_eval_min_points  = 300;
    float shape_evidence_clamp   = 8.0f;

    // DIAGNOSTIC one-shot: if non-empty, dump a fitted refrigerator's accumulated voxel-bank point cloud (room
    // frame, XYZ per line) to this path ONCE (when the bank exceeds ~200 pts), for the offline
    // square-vs-round model-comparison harness (tests/compare_models). "" = off. Not a runtime knob.
    std::string dump_cloud_path        = "";

    // ── Top/leg SDF split band ────────────────────────────────────────────────────────────────────
    // Forwarded to RefrigeratorModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top
    // face is attributed to the slab (candidate) vs a leg.
    float sigma_obs                 = 0.05f;
    // On-surface membership for the candidate/residual split in RefrigeratorFitter::observe.
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (REFRIGERATOR.md) — full-covariance recursive filter ──────────────────────────
    float ai2_sigma_base_m    = 0.03f;   // base on-surface obs noise std (m); R = σ² (+ motion_var + …)
    float ai2_clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m = 0.12f;   // a point further than ~this from every surface is likely clutter
    float ai2_prior_size_std  = 0.30f;   // (legacy; the size prior is now split into footprint vs height below)
    // FOOTPRINT + HEIGHT prior (a standard fridge footprint ≈ 0.60×0.60 is TIGHT; HEIGHT varies a lot). The
    // footprint std is small (strong pull on w,h≡depth so a partial front-only view can't float the depth);
    // the height std is broad (weak/data-driven). Split RefrigeratorBeliefParams' prior_footprint_* / prior_height_*.
    float ai2_prior_footprint_m   = 0.60f;   // mean of the w & h(depth) prior (m)
    float ai2_prior_footprint_std = 0.08f;   // TIGHT → strong footprint prior (m)
    float ai2_prior_height_m      = 1.90f;   // mean of the H (vertical) static anchor (m) — a standard tall fridge
    float ai2_prior_height_std    = 0.30f;   // anchor std (m); stops the box top floating above the cloud
    // DEPTH-OBSERVABILITY prior: depth (h) is only identifiable when the cloud spans the depth extent (front AND
    // back face seen). A front-only view is a thin ly-slab; its many points still spuriously drag depth to the
    // clamp (the size common-mode caps σ, not the mean). This EXTRA depth-prior precision is applied ∝ (1 − observed
    // depth-extent / footprint), so a single-face view holds depth at the footprint prior and it relaxes to
    // data-driven once an orbit reveals the back face. Covariance keyed on the covariate, not a gate. 0 = OFF.
    float ai2_depth_unobs_precision = 1500.0f;  // extra 1/m² when depth extent is unobserved (single face)
    float ai2_depth_obs_band_m      = 0.10f;    // ly-spread (m) below which the cloud is a single depth face
    // TOP anchor: resist the floor-anchored box top ratcheting ABOVE the observed cloud top (H→2.37 for a 1.9 m
    // cloud). A firm TWO-SIDED anchor pins H to z_top_obs (p97); the junk tail above it is faded via per-point R
    // (ai2_top_overseg_sigma_per_m). Data still sets the lower bound (front points force H ≥ real top).
    float ai2_top_no_float_precision = 10000.0f; // firm two-sided anchor pinning H → observed robust cloud top (1/m²)
    float ai2_top_no_float_margin_m  = 0.02f;   // upward allowance (m) added to the observed top as the anchor target
    float ai2_top_overseg_sigma_per_m = 2.0f;   // R inflation (σ per m) for points ABOVE the robust top — fades the
                                                // over-segmentation junk tail so it can't ratchet H up. 0 = OFF
    // WALL-FLUSH + WALL-PARALLEL factor (a fridge's back face rests against a room wall — ported from
    // cabinet_concept). Precision is the marginal of a {flush, free-standing} mixture keyed on the back-to-wall
    // gap, so it decays CONTINUOUSLY to 0 for a genuine mid-room fridge (no proximity gate). 0 = OFF.
    float ai2_wall_precision          = 400.0f;   // 1/m² at zero gap (back face → wall)
    float ai2_wall_parallel_precision = 200.0f;   // back face parallel to the wall (width-axis · wall-normal → 0)
    float ai2_wall_reach_m            = 0.15f;    // gap scale over which the flush hypothesis loses its weight
    // WALL as a competing per-point explanation (explaining away — see RefrigeratorBeliefParams). π_wall and the
    // extra wall-plane std. 0 = OFF (mixture stays box+clutter, i.e. the pre-existing behaviour).
    // Door CLEARANCE prior: nats of preference, at full flush, for the door mode facing INTO the room over
    // one facing into the wall. Scaled by the flush mixture weight, so it vanishes for a mid-room fridge.
    float ai2_door_clearance_gain     = 3.0f;
    // INFERRED VOLATILITY (MODEL_HISTORY.md §3) — the HISTORY stage. Q becomes exp(ω) per DOF, inferred from
    // how much the belief actually moves, so retention grows with consistent experience and collapses when the
    // object genuinely moves. false ⇒ Q is the constant process_std_* exactly as before.
    // DETECTOR ENVELOPE (common/detectability): P(detect) is unimodal in projected fill — too far = too few
    // pixels to segment, too close = the object overflows the frame and the mask truncates. Drives BOTH the
    // viewpoint stand-off and the weight given to absence. Properties of YOLO+ZED, measurable from a tour
    // (log fill vs mask-present and fit the shoulders); the defaults are a conservative prior, not a fit.
    float detect_min_fill             = 0.10f;
    float detect_max_fill             = 0.60f;
    float detect_soft                 = 0.06f;
    bool  ai2_volatility_infer        = false;
    float ai2_volatility_lr           = 0.02f;
    float ai2_volatility_sigma        = 2.0f;
    float ai2_wall_explain_frac       = 0.25f;
    float ai2_wall_explain_sigma_m    = 0.05f;
    // WALL NO-CROSS (ONE-SIDED): the flush factor above is two-sided (drives the back-face gap → 0) and its weight
    // decays away from the wall, so a fit can end up CROSSING the wall (back extending past it into the wall/
    // exterior) with the flush factor too weak there to recover it. This one-sided term resists penetration ONLY:
    // active when gap = (back_centre − wall.p)·(inward normal) goes negative (crossed) or within WallNoCrossMarginM,
    // with a precision that GROWS with how far past — leaving flush (gap≈0) untouched. STRONG (a wall is a hard
    // physical boundary), continuous/soft (no clamp). wall.ok==false ⇒ inert. 0 = OFF.
    float ai2_wall_no_cross_precision = 2000.0f;  // 1/m² per m of penetration past the wall (>> the flush precision)
    float ai2_wall_no_cross_margin_m  = 0.0f;     // interior offset (m) inside which the constraint activates
    float ai2_process_std_m   = 0.005f;  // predict process-noise std, length DOFs (m/frame)
    float ai2_process_std_yaw = 0.01f;   // predict process-noise std, yaw (rad/frame)
    // Stale-belief aging (measurement-age → covariance). Nominal inter-frame period (s) of the mask stream,
    // used to convert the per-frame process noise Q into a RATE so Σ keeps inflating on the AGENT's own clock
    // while the sensor is silent (rc::ai::inflate_for_age). <=0 DISABLES it → the belief simply freezes on
    // stale (historic information-filter behaviour). A dead ZED/mask feed then reads as a growing Σ downstream.
    float ai2_age_nominal_dt_s = 0.0f;
    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out). The frame's
    // information saturates here, so N≈10⁴ correlated points can't collapse σ → calibrated posterior.
    float ai2_common_mode_pos_std  = 0.03f;  // shared position error (m); pose-chain cov adds to it
    float ai2_common_mode_size_std = 0.02f;  // shared size error w,h,H (m)
    float ai2_common_mode_yaw_std  = 0.03f;  // shared yaw error (rad)
    // Ego-motion → COMMON-MODE ("be still to UPDATE, else CONFIRM" — the VOR/fixation term). Routes the per-mask
    // ego-motion smear (a SHARED error) into the per-frame common-mode so Woodbury caps the frame's authority to
    // move the GEOMETRY MEAN: a moving frame confirms but can't reshape/reposition/rotate the refrigerator; geometry
    // updates concentrate at stillness. Std growth per unit motion_dotd (m/s), CONTINUOUS (0 at stillness, no
    // gate). Reshape (w,h≡size) is the worst offender while rotating → size gain is largest. 0 disables a channel.
    float motion_cm_pos_gain  = 0.10f;   // position (cx,cy) shared-error std per m/s of motion_dotd
    float motion_cm_size_gain = 0.20f;   // extent (w,h,H) shared-error std per m/s — the anti-RESHAPE lever
    float motion_cm_yaw_gain  = 0.12f;   // yaw shared-error std (rad) per m/s      — the anti-ROTATE lever
    // ── Robust ego-motion magnitude + discrete "be-still-to-update" confirm-only gate (ported 1:1 from
    // chair_concept). motion_magnitude(inst) = max(|motion_dotd|, ego_lin_mps + ai2_ang_lever_m·ego_ang_radps):
    // a producer-INDEPENDENT motion signal (works even if the mask packet never populated motion_dotd), built from
    // the robot's OWN camera linear/angular speed via the transform chain. It feeds BOTH the continuous common-mode
    // above (motion_cm_* · motion_magnitude · periphery) AND the discrete confirm-only gate below.
    float ai2_ang_lever_m   = 2.0f;    // rad/s → m/s lever (tangential speed of a fridge ~this far away) for ego-motion
    float ai2_periph_ref    = 0.50f;   // centroid radius (focal-norm) at which the off-axis periphery penalty → 1
    float ai2_motion_ref_mps = 0.60f;  // motion magnitude (m/s) at which a fully-peripheral frame is fully unreliable
    // Discrete confirm-only gate ("only update if the robot is still"): true ⇒ the fitter CONFIRMS but does NOT move
    // the geometry mean (predict-only branch, ages Σ) when the robot is MOVING (ego lin/ang OR motion_dotd above its
    // still-level) AND the mask is off-centre. EXCEPTION: a well-centred mask (centroid radius ≤ the center radius)
    // is trusted even while moving → still updates. Geometry updates thus concentrate at stillness, with the
    // continuous common-mode as the graceful backstop. Default true (matches chair's requested policy).
    bool  ai2_motion_confirm_only = true;
    float ai2_still_lin_mps       = 0.05f;  // camera linear speed (m/s) below which the robot counts as "still"
    float ai2_still_ang_radps     = 0.10f;  // camera angular speed (rad/s) below which the robot counts as "still"
    float ai2_still_dotd          = 0.05f;  // per-mask ego-motion corruption speed (m/s) still-level
    float ai2_moving_update_center_radius = 0.35f;  // mask centroid radius below which a moving update is still allowed
    // STATIC range weighting (motion-free): the common-mode error grows with view distance, so a far, vague
    // mask cannot resolve pose — orientation least of all (a far view confirms existence, can't rotate the
    // refrigerator). Continuous, no gate. lat feeds R + position common-mode (m per m of range); yaw feeds the yaw
    // common-mode (rad per m), the binding term. Set 0 to disable. Range Z comes from the voxelizer (mask_range).
    float ai2_range_noise_lat_per_m  = 0.02f;  // lateral deprojection std growth (m per m of range)
    float ai2_range_noise_yaw_per_m  = 0.03f;  // yaw common-mode std growth (rad per m of range)
    float ai2_range_noise_size_per_m = 0.08f;  // SIZE (w,h,H) common-mode std growth (m per m of range): a distant
                                               // mask can't reshape/inflate a converged refrigerator (freezes geometry afar)
    // THRESHOLD (truncation gate): skip the geometric update (predict only) when this fraction of the mask
    // silhouette is on the image border — a truncated mask is a biased extent, not measurement noise.
    float ai2_trunc_gate_frac = 0.10f;
    int   ai2_gn_iters        = 4;       // Gauss-Newton iterations per frame
    std::string ai2_csv_path  = "";      // if non-empty, append per-cycle belief (state + Σ diag + mask R) to CSV
    // Anisotropic per-point R (PRECISION_AS_INFORMATION.md Stage 1). Replaces the scalar per-point variance with
    // the deprojection noise projected on the SDF normal → a grazing view carries ~0 yaw information by
    // construction (no obliquity/range yaw gains needed). The 4 constants are PHYSICAL (ZED sensor), not tuning.
    float pixel_sigma_over_f     = 0.0015f;  // σ_px/f → transverse std per m of range
    float depth_sigma0_m         = 0.006f;   // depth std floor (m)
    float depth_sigma_range_coef = 0.004f;   // depth std growth (m per m² of range)
    float model_sigma_m          = 0.010f;   // residual model std floor (m)
    // a1′+a2′: weighted 2-D footprint residual + shared depth-affine nuisance (replaces the moment channel; the
    // real grazing-yaw fix). depth_bias/scale_std are the ZED depth-affine priors (physical). See belief params.
    bool  footprint_residual     = false;
    // C2v symmetry-quotient chart (PRECISION_AS_INFORMATION.md Stage 3): optimise the footprint in [s,a₁,a₂]
    // so the 4 box representatives collapse to one point — no fold/flip/mode-accumulator, no 90°/180° yaw snaps,
    // and σ_yaw diverges honestly near-square. Global (set once at startup). Pairs with footprint_residual.
    bool  quotient_chart         = false;
    float depth_tilt_std         = 0.020f;   // shared per-frame depth tilt prior std (m/rad) — the yaw nuisance
    float depth_bias_std         = 0.015f;   // shared per-frame depth bias prior std (m)
    float depth_scale_std        = 0.010f;   // shared per-frame depth scale prior std (fraction)
    // EXPERIMENTAL birth-surprise probe (read-only): read residual_concept's `grid` node as an unexplained-
    // occupancy (surprise) field, cluster it, and LOG uncovered high-surprise regions next to the tracker's
    // birth decision — to check whether surprise flags births cleanly before it drives the lifecycle. OFF = no
    // read, no log, zero cost. See birth_surprise_probe.h. Writes etc/birth_surprise.csv when on.
    bool birth_surprise_probe = false;

    // ── YOLO-INDEPENDENT LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h) ───────
    // Second, sensor-independent evidence channel: lidar3D returns landing on the refrigerator (legs + rim) pin the
    // extent and centre in METRIC range, ALONG the viewing ray — an error mechanism uncorrelated with the
    // YOLO segmentation, so it attacks the mask-erosion under-size the mask cannot self-correct. Consumes
    // the lidar3D media plane via RefrigeratorLidarIngestor; dormant (no DDS participant) while precision == 0.
    float lidar_precision       = 0.0f;   // per-ray range precision (1/m², ≈1/σ_range²); 0 = OFF
    // Low "bpearl" LiDAR as a SEPARATE per-device ray-set (own origin, occlusion-aware first-hit; sees the legs
    // the high helios grazes over). Summed into the SAME GN as helios — NOT merged (merging loses the origin the
    // ray geometry needs). Also feeds the existence leg-occupancy carve from its own origin. 0 = OFF (helios only).
    float lidar_bpearl_precision = 0.0f;
    float lidar_robust_c_m      = 0.05f;  // Cauchy scale (m): returns this far off the surface fade out
    float lidar_select_margin_m = 0.10f;  // pre-select returns within (birth half-extent + margin), all z up to top
    float lidar_coverage_n0     = 60.0f;  // LiDAR ray count for FULL weight; fewer → proportionally down-weighted
    // Angular-coverage weighting: precision ×= (1−R)^p, R = mean-resultant length of return bearings about the
    // centre. p=0 disables (flat), p=1 = pure circular variance. Down-weights ONE-SIDED sweeps (which over-
    // commit the near-square w↔h mode from a single view); the recursive belief still accumulates over an orbit.
    float lidar_coverage_ang_power = 1.0f;

    // ── Divergence safety net (mirrors bottle_concept) ────────────────────────────────────────────
    // THRESHOLD (outlier step guard): a static refrigerator cannot physically move this far in one frame, so a GN
    // step whose centre jump exceeds it is an OUTLIER frame (corrupted mask cloud / one-sided LiDAR runaway)
    // → reject the update, restore state+Σ, widen Σ. 0 disables. Prevents the cx=−200m runaway.
    float max_step_m = 1.0f;

    // ── Coverage / traction (EXISTENCE_BELIEF_PLAN.md) ────────────────────────────────────────────
    // Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a
    // large mask grows to explain it (fixes refrigerator_1.png). 0 = OFF.
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;

    // ── Free-space / VACATE (EXISTENCE_BELIEF_PLAN.md Step 4) ──────────────────────────────────────
    // The counter-force that BOUNDS coverage. A LiDAR beam that traverses the TOP-SLAB z-band and continues
    // beyond (endpoint past the far face) demonstrates that slab region is EMPTY → a shrink-only pull retreats
    // the refrigeratortop boundary past the empty crossing. Coverage occupies where masked; free-space vacates where
    // the beam passed through. Together they settle the extent where camera and LiDAR agree, so coverage can
    // no longer run away onto clutter. 0 = OFF. Needs the LiDAR sweep staged (auto-staged whenever this OR
    // LidarPrecision > 0).
    float free_space_precision = 0.0f;

    // ── Footprint SECOND-MOMENT factor (refrigerator_belief.h) ───────────────────────────────────────────
    // Measures (w,h,yaw) from the top-band cloud's 2D inertia tensor and folds it as a linear Gaussian factor —
    // the escape from the clutter-trap that leaves a dense, bigger/yawed mask unable to rotate/resize the box
    // (refrigerators_5.png). Also seeds (w,h,yaw) at birth. yaw pull scales with extent anisotropy; capped by the
    // range-driven common-mode. 0 = OFF (baseline unchanged).
    float footprint_moment_precision = 0.0f;
    // GENTLE range term (m per m of range): shared per-frame moment variance grows as (this·range)². Much
    // smaller than AI2RangeNoiseSizePerM because a global footprint fit is range-robust; it makes the moment
    // accumulate over frames (stable) rather than snap to each frame's footprint.
    float footprint_moment_range_per_m = 0.03f;

    // ── FE attention / surprise dynamics (active-perception trigger) ───────────────────────────────
    // Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the refrigerator
    // moved — stays surprising long enough to attend); surprise = smoothed positive gap FE−baseline.
    float fe_baseline_adapt_down = 0.05f;
    float fe_baseline_adapt_up   = 0.005f;
    float fe_surprise_smooth     = 0.10f;
    // Ego-motion (motion_dotd) coupling: moment variance grows by (this·motion_dotd)² so a "going-away/rotation"
    // frame (degraded/split mask) can't reshape the established fit; the mode accumulator is halved at
    // motion_dotd = OrientationMotionRef. The observed reshapes/flips all arrived on motion frames.
    float footprint_moment_motion_gain = 0.30f;
    float orientation_motion_ref       = 0.50f;
    // Grazing-view yaw stability (CSV rogue-rotation fix). The continuity fold (canonicalize, fixed yaw scale) and
    // the per-point obliquity yaw cap (kObliquityYawGain in refrigerator_fitter.cpp) were validated live 2026-07-11 and
    // are now always on — no flags. The MOMENT-channel backoff below is still A/B.
    // obliquity_moment_gain: same view-angle backoff, but on the MOMENT channel — moment_extra_var +=
    //   (gain·(1/|cosθ|−1))². The per-point cap only protects the GN; the CSV rogue rotations came in on the
    //   MOMENT channel (dyaw_moment), which the per-point cap never sees. A grazing/foreshortened frame biases the
    //   2D inertia → back the whole moment off, not just its per-point yaw. 0 = OFF.
    float obliquity_moment_gain        = 0.0f;
    // footprint_moment_completeness_gain / _min_completeness: forwarded to RefrigeratorBeliefParams — inflate the moment
    //   measurement variance as the observed footprint area falls below the believed area (partial view). This is
    //   the direct fix for the going-away / detour-return yaw snap (ai2_log.csv: completeness≈0.01). 0 = OFF.
    float footprint_moment_completeness_gain = 0.0f;
    float footprint_moment_min_completeness  = 0.02f;

    // ── Appearance-based FRONT (door) detection + yaw resolver ─────────────────────────────────────
    // A fridge's square-ish footprint + wall-flush leave the YAW discrete-ambiguous (geometry can't tell which
    // way the DOOR faces). The door is an APPEARANCE feature (handle + seams = strong vertical lines), so we
    // project the FITTED box into the live ZED RGB, score vertical-edge energy per visible face, and fold the
    // winning face's bearing into the belief's sequential-Bayes door-mode resolver (RefrigeratorBelief::resolve_front).
    // Because it uses the belief's own box (not YOLO's mask) it keeps working when the robot is too close for YOLO.
    bool  front_detect_enabled   = true;    // master switch: subscribe to ZED RGB + run detect_front/resolve_front
    float front_min_face_area_px = 900.0f;  // min projected face area (px²) to score for door-ness
    float front_min_confidence   = 0.10f;   // min door-ness margin (max−second)/(max+eps) for detect_front to emit
    bool  front_log              = false;    // log each adopted door-mode flip / accepted cue

    // ── Existence / removal (common/existence_belief.h) ───────────────────────────────────────────
    // Each cycle carve the LiDAR sweep against the refrigerator footprint → occupancy/free-space log-odds; remove
    // when P(occupied) < removal_prob. Evidence-based, not a miss counter. OFF by default (a mis-removal
    // deletes furniture); enable to replace merge-only removal.
    bool  existence_removal_enabled = false;
    float existence_removal_prob    = 0.12f;  // decision boundary: remove when L < log(p/(1−p))
    float existence_logodds_max     = 4.0f;   // clamp |L| so evidence stays finite AND recoverable
    float existence_detection_prob  = 0.85f;  // P(beam through OCCUPIED footprint returns from it)
    float existence_clutter_prob    = 0.05f;  // P(beam through EMPTY footprint returns anyway) — spurious rate
    float existence_sensor_sigma_m  = 0.03f;  // LiDAR range σ (m) for the soft occ/free surface split
    int   existence_remove_frames   = 15;     // debounce: require the removal decision this many consecutive
                                              // cycles before deleting (transient association hiccups don't remove)
    // ABSENCE evidence degradation (no gate). "Predicted-visible but no mask/return there" is weak evidence of
    // removal when the sensor likely could not resolve the object anyway — a distant view (small projection /
    // sparse beams) or a largely-occluded one. So the FREE/absence half of BOTH channels is scaled by a
    // confidence factor c = (range_ref/range)^power (capped at 1) × visible_fraction, degrading it continuously
    // with range and occlusion while OCCUPANCY stays fully informative (a distant detection still confirms).
    // This is P(detect|present) falling with range, the codebase's "range→precision, not a gate" pattern.
    float existence_absence_range_ref_m = 2.5f;   // range (m) below which absence is trusted at full weight
    float existence_absence_range_power = 2.0f;   // decay exponent (2 ≈ angular-area ∝ 1/range²); 0 disables
    float existence_verify_surprise     = 20.0f;  // decayed go-verify surprise (un-resolvable absence) above which
                                                  // a refrigerator is flagged wants_verification (epistemic pull, not removal)
    float verify_surprise_smooth        = 0.10f;  // EMA weight of the go-verify surprise accumulator (new sample
                                                  // weight; 1−this holds the old). Parity with fe_surprise_smooth.
    float existence_verify_gain         = 5.0f;   // epistemic gain (nats) a wants_verification refrigerator gets, so the
                                                  // controller drives to a resolving ZED view (confirm-or-remove)
    // LiDAR carve: the WHOLE solid box z∈[0,H] from each available ray-set (helios + bpearl), OCCUPANCY only.
    // Free-space is model-consistent for a solid fridge (unlike the thin-plate tabletop this was cloned from),
    // but removal deliberately stays the camera silhouette's job — it HOLDs out-of-FoV, so an unseen fridge is
    // never deleted. There is no leg carve: a refrigerator has no legs.

    // ── "Is this really a fridge?" plausibility filter + soft singleton (model-evidence mis-detection reject) ─
    // A YOLO "refrigerator" mis-detection (e.g. a ~70 cm elongated cabinet) contradicts the fridge shape priors
    // (square-ish footprint, tall) → poorly explained by the fridge model → high free energy. Rather than a hard
    // `if (H<1) reject`, a CONTINUOUS plausibility score (aspect·size·height·fit) + a bounded plaus_evidence
    // accumulator gates BIRTH (scales the tracker's birth evidence) and drives EXISTENCE decay (implausible →
    // negative existence log-odds → removed by the existence debounce). A SOFT singleton (a kitchen has ~one
    // fridge) lets a stronger-evidence fridge inhibit weaker duplicates. All continuous, all bounded, no cuts.
    bool  fridge_filter_enabled   = true;    // master switch for the whole plausibility filter
    float plaus_aspect_scale      = 0.15f;   // aspect_ok = exp(−(|w−h|/(w+h)/scale)²): ~1 square, →0 elongated
    float plaus_size_scale        = 0.15f;   // size_ok  = exp(−((w−fp)²+(h−fp)²)/(2·scale²)); fp = AI2PriorFootprintM
    float plaus_alt_size_scale    = 0.60f;   // the ALTERNATIVE ("other furniture") footprint std (m) the fridge
                                             // hypothesis is scored against in fridge_log_evidence_ratio
    float plaus_height_min        = 1.20f;   // height_ok logistic centre (m): →1 tall, →0 below ~1 m ("70 cm improbable")
    float plaus_height_soft       = 0.15f;   // height_ok logistic softness (m)
    float plaus_fe_ref            = 2.0f;    // "healthy fridge fit" FE reference — NEEDS LIVE TUNING (≈ a good fridge's mean_energy)
    float plaus_fe_scale          = 1.0f;    // fit_ok = exp(−max(0, FE−FeRef)/FeScale)
    float plaus_clamp             = 8.0f;    // ±bound on the plaus_evidence accumulator (bounded memory: one bad frame can't kill a real fridge)
    float plaus_height_prior_gain = 2000.0f; // short-height prior precision (1/m²) per (deficit/soft) below HeightPlausibleMin; 0 = OFF
    float plaus_to_existence_gain = 1.5f;    // how strongly tanh(plaus_evidence) pushes the existence log-odds per cycle
    float singleton_inhibition    = 1.0f;    // mutual-inhibition weight: a STRONGER fridge subtracts this·P(exists) from a weaker one's existence L
    bool  fridge_filter_log       = false;   // log per-instance plausibility / evidence / existence-delta each cycle

    // ── RT-edge covariance upload ─────────────────────────────────────────────────────────────────
    // Upload the refrigerator pose covariance onto the room→refrigerator RT edge (rt_covariance_att, 6×6 SE3), mapped from
    // the belief's full Σ over [cx,cy,H,w,h,yaw]: x←cx, y←cy, z←H/2, yaw←ψ; roll/pitch are unobservable (large).
    // rt_cov_scale calibrates the raw variance toward NEES≈1.
    float rt_cov_scale     = 1.0f;
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so room_concept can use the
    // refrigerator as an SE(2) pose landmark. OFF by default (consumer side is also flag-gated + defaults OFF).
    bool        publish_object_obs = false;      // RefrigeratorConcept.PublishObjectObs
    std::string object_obs_frame   = "body";     // RefrigeratorConcept.ObjectObsFrame (localizer base node)

    // ── Multi-instance birth/associate/death (shared rc::InstanceTracker) ─────────────────────────
    // "refrigerator" masks are associated to instances by a covariance-gated global 1-to-1; an unexplained mask
    // spawns a new refrigerator. There is NO death-by-occlusion: a refrigerator is large static furniture, so a long
    // occlusion is not absence. An instance leaves only via the MERGE operator (two refrigerators cannot share
    // space) or the existence-removal channel (refrigerator_existence.cpp). Keep birth_min_sep large so no two
    // refrigerator centres can sit that close.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.50f;   // metric XY gate (m) before an instance has a usable covariance
    // Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I). For a refrigerator
    // the YOLO mask CENTROID sits well off the fitted geometric CENTRE under a partial view (~0.2–0.5 m),
    // so R must cover that offset or the overconfident fit (σ ~mm) rejects its own real detection.
    float tracker_detection_noise_m = 0.35f;
    int   tracker_birth_frames      = 8;      // frames a mask must stay unexplained before spawning a refrigerator
    // FUSED BIRTH (EXPERIMENTAL, off by default): let residual-grid SURPRISE MASS under a detection accelerate its
    // birth. Per frame a detection contributes birth_evidence = 1 + gain·(m/(m+ref)) toward the birth_frames
    // boundary, where m = residual mass within radius of the detection (birth_surprise_probe.h). A CORROBORATED
    // detection (real object → high unexplained-occupancy) crosses in ~1-2 frames; an UN-corroborated / phantom
    // detection (m≈0 → evidence 1.0) still serves the full birth_frames debounce → phantoms are NOT made easier,
    // only real births faster. Birth = evidence-to-a-boundary (residual raises the prior), not a fixed counter.
    bool  birth_fusion          = false;
    float birth_fusion_gain     = 6.0f;   // max extra evidence/frame at full corroboration (0 ⇒ baseline)
    float birth_fusion_mass_ref = 8.0f;   // residual mass at which corroboration is half-saturated (m/(m+ref))
    float birth_fusion_radius_m = 0.50f;  // window (m) for the residual-mass sample under the detection
    float tracker_birth_min_sep_m   = 0.60f;  // a birth must be ≥ this (m) from every existing refrigerator
    // Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of
    // the SMALLER footprint (two refrigerators cannot share space). 0 disables the merge.
    float tracker_merge_overlap = 0.05f;
    // Default geometry for a refrigerator BORN from a mask, used only when no birth burst is available
    // (see birth_frag_* below); the belief refines from here.
    float tracker_birth_width_m  = 1.0f;
    float tracker_birth_depth_m  = 0.6f;
    float tracker_birth_height_m = 0.75f;

    // ── BIRTH FRAGMENT: keep the probation burst (common/birth_fragment/birth_fragment.h) ─────────────
    // A candidate matures over tracker_birth_frames observations; until now every mask cloud seen on the way
    // was discarded and the instance was seeded from ONE frame's centroid plus the constants above. Banking
    // the burst buys two things: geometry measured from the object (above all the HEIGHT — the constant
    // 0.75 m sits below plaus_height_min=1.20 and far from ai2_prior_height_m=1.90, so every real fridge
    // was born implausible and had to climb out), and the right to REFUSE a birth before it reaches DSR.
    bool  birth_frag_enabled  = true;
    float birth_frag_voxel_m  = 0.03f;    // dedup grid for the burst (finer than the bank: a burst is short)
    int   birth_frag_max_pts  = 20000;    // hard cap per candidate; a lingering blob cannot grow without bound
    // LOCAL CONSISTENCY δ (ms), the Khronos fragment criterion: observations may only be fused as one view of
    // one object if they are close enough in time that neither localization error nor scene change matters.
    // A candidate whose burst spans longer (a blob flickering in and out across a room traverse) still births
    // on its streak, but its cloud is discarded and the constants above are used. 0 disables the test.
    std::uint64_t birth_frag_delta_ms = 4000;
    // ADMISSION on the accumulated burst. RefrigeratorBelief::candidate_plausibility is already applied
    // per-frame as a birth-evidence multiplier, but there it sees ONE partial view whose z-range understates
    // the height — biasing it against genuine fridges. Re-run on the union it measures the real extent, and
    // becomes a decision rather than a weight: below this, the birth is refused outright instead of being
    // created and later retracted by the plausibility→existence channel. 0 disables the refusal.
    float birth_admit_plausibility = 0.35f;

    // ── Ricoh-360 attention gate ──────────────────────────────────────────────────────────────────
    // A ricoh (depth_var>0) 360-RGB YOLO detection is bearing-only: it never births or fits, it only raises
    // attention (process_ricoh_bearings) so the robot seeks a good ZED view.
    float ricoh_attention_conf = 0.60f;   // min YOLO confidence for a ricoh detection to raise attention
    // Ricoh attention association gates (process_ricoh_bearings): small margins on the physical angular size /
    // range roughness when deciding whether an unassigned ricoh bearing already matches a known refrigerator.
    float ricoh_attention_angle_margin_rad = 0.05f;  // extra angular tolerance on the refrigerator's angular half-size
    float ricoh_attention_range_band_m     = 1.0f;   // extra range band (ricoh range is rough/indicative)
};

// Fill a RefrigeratorConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
RefrigeratorConfig load_refrigerator_config(const ConfigLoader& cfg);

// Cross-check common/concept_manifest/refrigerator.concept.toml against the priors etc/config.toml actually
// produced. Read-only, startup-only: the manifest is NOT authoritative yet — this measures whether it could be.
bool verify_refrigerator_manifest(const RefrigeratorConfig& out, const std::string& path);

}  // namespace rc
