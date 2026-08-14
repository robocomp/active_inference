/*
 * table_config.h  —  plain-data configuration for the table_concept agent + its ConfigLoader loader.
 *
 * Kept separate from SpecificWorker so a new concept agent can copy this file and edit only the keys it
 * needs (mirrors bottle_concept/bottle_config.h). Every field carries its default here; the loader
 * (table_config.cpp) overrides from the RoboComp TOML when the key is present.
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct TableConfig
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
    float obs_distance                 = 1.8f;    // d_obs for the epistemic planner
    int   epistemic_cooldown_cycles    = 200;     // min cycles withdrawn after satisfaction
    int   table_log_period_frames      = 30;      // per-cycle log throttle
    // Publish the accumulated bank onto the DSR node. OFF: audited 2026-08-14, nothing in the whole
    // components tree reads *_voxel_bank_pts. The bank itself is still built — evaluate_shape and
    // DumpCloudPath read it locally — this is only the graph traffic.
    bool  publish_voxel_bank           = false;
    int   voxel_bank_max_points        = 4000;    // cap on the table-owned voxel memory bank
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
    // Low-pass rate for the shape log-Bayes-factor. 0 = the legacy SUM (kept for A/B).
    // >0 makes shape_evidence track the CURRENT evidence with a time constant instead of accumulating,
    // because the quantity being re-scored is the accumulated voxel-bank cloud — largely the SAME points
    // each evaluation, so summing measures dwell time, not information. Same lesson as
    // ring_config.h's evidence_ema_alpha ("static furniture re-observed is not new evidence").
    float shape_evidence_ema_alpha = 0.0f;

    // DIAGNOSTIC one-shot: if non-empty, dump a fitted table's accumulated voxel-bank point cloud (room
    // frame, XYZ per line) to this path ONCE (when the bank exceeds ~200 pts), for the offline
    // square-vs-round model-comparison harness (tests/compare_models). "" = off. Not a runtime knob.
    std::string dump_cloud_path        = "";

    // ── Top/leg SDF split band ────────────────────────────────────────────────────────────────────
    // Forwarded to TableModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top
    // face is attributed to the slab (candidate) vs a leg.
    float sigma_obs                 = 0.05f;
    // On-surface membership for the candidate/residual split in TableFitter::observe.
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (TABLE.md) — full-covariance recursive filter ──────────────────────────
    float ai2_sigma_base_m    = 0.03f;   // base on-surface obs noise std (m); R = σ² (+ motion_var + …)
    // ── DETECTOR ENVELOPE (common/detectability) — the YOLO inverse model ─────────────────────────
    // Defaults are the fleet PRIOR, not a measurement, so behaviour is unchanged until these are set in
    // etc/config.toml. See the block there for the fit measured from this agent's own ai2_log.
    float detect_min_fill     = 0.10f;
    float detect_max_fill     = 0.60f;
    float detect_soft         = 0.06f;
    float ai2_clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m = 0.12f;   // a point further than ~this from every surface is likely clutter
    float ai2_prior_size_std  = 0.30f;   // broad size prior std (m) on w,h,H
    float ai2_process_std_m   = 0.005f;  // predict process-noise std, POSE (cx,cy) (m/frame)
    float ai2_process_std_yaw = 0.01f;   // predict process-noise std, yaw (rad/frame)
    // EXTENT (w,h,H) process noise, separate from the pose walk. 0 = a table's dimensions are CONSTANT
    // (the physical truth); <0 = historic behaviour (share ai2_process_std_m). See TableBeliefParams.
    float ai2_process_std_extent_m = 0.0f;
    // Cap Σ at the prior on predict (OU decay toward the stationary prior, not a divergent random walk), so
    // ageing an unobserved table can never make it more plastic than it was before it was ever seen.
    bool  ai2_clamp_sigma_to_prior = true;
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
    // move the GEOMETRY MEAN: a moving frame confirms but can't reshape/reposition/rotate the table; geometry
    // updates concentrate at stillness. Std growth per unit motion_dotd (m/s), CONTINUOUS (0 at stillness, no
    // gate). Reshape (w,h≡size) is the worst offender while rotating → size gain is largest. 0 disables a channel.
    float motion_cm_pos_gain  = 0.10f;   // position (cx,cy) shared-error std per m/s of motion_dotd
    float motion_cm_size_gain = 0.20f;   // extent (w,h,H) shared-error std per m/s — the anti-RESHAPE lever
    float motion_cm_yaw_gain  = 0.12f;   // yaw shared-error std (rad) per m/s      — the anti-ROTATE lever
    // STATIC range weighting (motion-free): the common-mode error grows with view distance, so a far, vague
    // mask cannot resolve pose — orientation least of all (a far view confirms existence, can't rotate the
    // table). Continuous, no gate. lat feeds R + position common-mode (m per m of range); yaw feeds the yaw
    // common-mode (rad per m), the binding term. Set 0 to disable. Range Z comes from the voxelizer (mask_range).
    float ai2_range_noise_lat_per_m  = 0.02f;  // lateral deprojection std growth (m per m of range)
    float ai2_range_noise_yaw_per_m  = 0.03f;  // yaw common-mode std growth (rad per m of range)
    float ai2_range_noise_size_per_m = 0.08f;  // SIZE (w,h,H) common-mode std growth (m per m of range): a distant
                                               // mask can't reshape/inflate a converged table (freezes geometry afar)
    // ── COVERAGE (completeness) common-mode — the second half of "close enough to reshape" ──────────────────
    // Range alone does NOT capture whether a view licenses a reshape, and the live log proves it: the worst
    // observed distortion happened at range 0.75 m (well inside every range term) with completeness 0.28 —
    // h jumped +0.76 m in ONE frame and the table stayed 1.7 m long for 1200 cycles until a completeness-1.9
    // view snapped it back (ai2_log.csv, table_9 cycles 1127→1136→2537). A partial/foreshortened footprint is
    // a SLIVER: the per-point SDF fit can only explain it by stretching the extent, because the rest of the
    // table is simply not in the measurement. So the covariate that gates RESHAPE authority is coverage, not
    // distance: c = observed footprint area / believed w·h (inst.last_completeness, one cycle stale).
    //   σ_extra = gain · (1/c − 1)      → 0 at c ≥ 1 (full view: UNCHANGED), → ∞ as the view fragments.
    // Same continuous-covariance form the moment channel already uses (FootprintMomentCompletenessGain) and the
    // same discipline as the range/obliquity/ego-motion terms: no gate, no cutoff — a sliver frame still
    // CONFIRMS the table and still refines what it can see, it just cannot re-cut the geometry. 0 disables.
    float ai2_coverage_size_gain = 0.30f;  // SIZE (w,h,H) common-mode std (m) per unit of (1/c − 1) — anti-RESHAPE
    float ai2_coverage_yaw_gain  = 0.15f;  // yaw common-mode std (rad) per unit of (1/c − 1)        — anti-ROTATE
    float ai2_coverage_pos_gain  = 0.05f;  // position (cx,cy) common-mode std (m) per unit of (1/c − 1); small:
                                           // a sliver still localises the near edge well, it just can't resize
    float ai2_coverage_min       = 0.02f;  // clamp floor on c so 1/c stays finite (caps the inflation)
    // THRESHOLD (truncation gate): skip the geometric update (predict only) when this fraction of the mask
    // silhouette is on the image border — a truncated mask is a biased extent, not measurement noise.
    float ai2_trunc_gate_frac = 0.10f;

    // ── FIXATION gate: the ATTENTION mechanism (a DELIBERATE, flagged threshold) ──────────────────────────
    // Rationale, and why the continuous covariance terms were NOT enough. All the graded levers here (range,
    // obliquity, coverage, ego-motion common-mode) act through the per-frame common-mode Σc, and the engine
    // saturates the frame's information at Σc⁻¹ — a nonzero asymptote. So a bad frame's authority is
    // ATTENUATED but never ZERO, and the GN mean step (recursive_laplace.h:221-222) still moves the mean by
    // ~Σc⁻¹ each frame. Hundreds of frames at 6 m therefore still walk a converged table: attenuation loses
    // to accumulation. Measured twice on live data after adding the range and coverage terms.
    // The biological model the user asked for is EYE FIXATION: primates take in detail only during fixations
    // and actively SUPPRESS intake otherwise (saccadic suppression) — attention is not a soft weight, it
    // GATES. In active-inference terms attention is precision, and an unattended channel's precision goes to
    // zero, i.e. the observation is not integrated at all. This robot has no pan-tilt, so a "fixation" is the
    // whole-body condition: CLOSE + the object CENTRED in the frame + the robot STILL.
    // Outside a fixation the cycle takes the existing truncation-gate path — predict() only, mean HELD, Σ
    // inflates. Association and existence do NOT read this, so an unfixated table is still CONFIRMED and
    // still tracked; only its GEOMETRY (pose/shape) is frozen. This is CLAUDE.md's "if a threshold is truly
    // unavoidable, flag it and justify why no model-level term works" — justified above.
    // CAVEAT, by design: a table the robot never approaches, centres and holds still for keeps its BIRTH
    // geometry forever. That is the honest outcome (never claim to know a shape you never resolved), but it
    // means convergence now REQUIRES the robot to actually go and look. Set FixationEnabled=false to revert.
    bool  fixation_enabled     = true;
    // RESOLVABLE (replaced the hard RANGE cut on 2026-07-30, same reasoning as chair_concept): range is only
    // a proxy for whether the frame can resolve the geometry; what actually decides it is how much
    // un-cluttered surface lands on the object. The range cut rejected dense far stares and accepted starved
    // near glances, and on chair_concept it froze every instance for 400/400 cycles.
    int   fixation_min_pts     = 300;    // RESOLVABLE: min mask points before a frame may touch the geometry
    float fixation_max_trunc   = 0.10f;  // RESOLVABLE: max silhouette truncation (mirrors ai2_trunc_gate_frac)
    float fixation_range_m     = 0.0f;   // RETIRED as a gate (0 = off). Kept so an A/B revert is one edit.
    float fixation_centre_frac = 0.60f;  // CENTRED: max |roi offset| (normalised, 0=centred, 1=image edge) —
                                         // the "fovea". <=0 disables just this condition.
    float fixation_still_dotd  = 0.05f;  // STILL: max |motion_dotd| = Z·‖ṡ‖ (m/s), the ego-motion mask smear
                                         // from the voxelizer. <=0 disables just this condition.
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
    // Second, sensor-independent evidence channel: lidar3D returns landing on the table (legs + rim) pin the
    // extent and centre in METRIC range, ALONG the viewing ray — an error mechanism uncorrelated with the
    // YOLO segmentation, so it attacks the mask-erosion under-size the mask cannot self-correct. Consumes
    // the lidar3D media plane via TableLidarIngestor; dormant (no DDS participant) while precision == 0.
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
    // THRESHOLD (outlier step guard): a static table cannot physically move this far in one frame, so a GN
    // step whose centre jump exceeds it is an OUTLIER frame (corrupted mask cloud / one-sided LiDAR runaway)
    // → reject the update, restore state+Σ, widen Σ. 0 disables. Prevents the cx=−200m runaway.
    float max_step_m = 1.0f;

    // ── Coverage / traction (EXISTENCE_BELIEF_PLAN.md) ────────────────────────────────────────────
    // Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a
    // large mask grows to explain it (fixes table_1.png). 0 = OFF.
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;

    // ── Free-space / VACATE (EXISTENCE_BELIEF_PLAN.md Step 4) ──────────────────────────────────────
    // The counter-force that BOUNDS coverage. A LiDAR beam that traverses the TOP-SLAB z-band and continues
    // beyond (endpoint past the far face) demonstrates that slab region is EMPTY → a shrink-only pull retreats
    // the tabletop boundary past the empty crossing. Coverage occupies where masked; free-space vacates where
    // the beam passed through. Together they settle the extent where camera and LiDAR agree, so coverage can
    // no longer run away onto clutter. 0 = OFF. Needs the LiDAR sweep staged (auto-staged whenever this OR
    // LidarPrecision > 0).
    float free_space_precision = 0.0f;

    // ── Footprint SECOND-MOMENT factor (table_belief.h) ───────────────────────────────────────────
    // Measures (w,h,yaw) from the top-band cloud's 2D inertia tensor and folds it as a linear Gaussian factor —
    // the escape from the clutter-trap that leaves a dense, bigger/yawed mask unable to rotate/resize the box
    // (tables_5.png). Also seeds (w,h,yaw) at birth. yaw pull scales with extent anisotropy; capped by the
    // range-driven common-mode. 0 = OFF (baseline unchanged).
    float footprint_moment_precision = 0.0f;
    // GENTLE range term (m per m of range): shared per-frame moment variance grows as (this·range)². Much
    // smaller than AI2RangeNoiseSizePerM because a global footprint fit is range-robust; it makes the moment
    // accumulate over frames (stable) rather than snap to each frame's footprint.
    float footprint_moment_range_per_m = 0.03f;

    // ── FE attention / surprise dynamics (active-perception trigger) ───────────────────────────────
    // Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the table
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
    // the per-point obliquity yaw cap (kObliquityYawGain in table_fitter.cpp) were validated live 2026-07-11 and
    // are now always on — no flags. The MOMENT-channel backoff below is still A/B.
    // obliquity_moment_gain: same view-angle backoff, but on the MOMENT channel — moment_extra_var +=
    //   (gain·(1/|cosθ|−1))². The per-point cap only protects the GN; the CSV rogue rotations came in on the
    //   MOMENT channel (dyaw_moment), which the per-point cap never sees. A grazing/foreshortened frame biases the
    //   2D inertia → back the whole moment off, not just its per-point yaw. 0 = OFF.
    float obliquity_moment_gain        = 0.0f;
    // footprint_moment_completeness_gain / _min_completeness: forwarded to TableBeliefParams — inflate the moment
    //   measurement variance as the observed footprint area falls below the believed area (partial view). This is
    //   the direct fix for the going-away / detour-return yaw snap (ai2_log.csv: completeness≈0.01). 0 = OFF.
    float footprint_moment_completeness_gain = 0.0f;
    float footprint_moment_min_completeness  = 0.02f;

    // ── Existence / removal (common/existence_belief.h) ───────────────────────────────────────────
    // Each cycle carve the LiDAR sweep against the table footprint → occupancy/free-space log-odds; remove
    // when P(occupied) < removal_prob. Evidence-based, not a miss counter. OFF by default (a mis-removal
    // deletes furniture); enable to replace merge-only removal.
    bool  existence_removal_enabled = false;
    float existence_removal_prob    = 0.12f;  // decision boundary: remove when L < log(p/(1−p))
    float existence_logodds_max     = 4.0f;   // clamp |L| so evidence stays finite AND recoverable
    // Frame-to-frame correlation of same-sign existence observations. A detector does not fail by
    // independent coin flip — it fails at a particular framing — so N consecutive misses are worth far
    // fewer than N observations (rc::exist::ExistenceBelief::set_frame_correlation). MEASURED, not tuned.
    // 0 = the previous behaviour (every frame independent).
    float existence_frame_correlation = 0.0f;
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
    // OPTIONAL EXTRA line-of-sight oracle from the LiDAR sweep, for occluders a room polygon cannot know
    // about (furniture, people, an open door leaf). WALLS are handled by the primary, exact and
    // sweep-independent rc::occlusion::walls_block() test — see TableProjection::set_room_polygon. This one
    // needs a fresh sweep and quantises bearing into bins, so it is OFF by default; set bins > 0 to enable.
    float existence_los_margin_m  = 0.25f;  // sample must be this far BEYOND the first hit to count as occluded
                                            // (absorbs the table's OWN returns + registration error)
    int   existence_los_azim_bins = 0;      // azimuth bins over 2π (720 ≈ 0.5° each); 0 = oracle OFF (default)
    float existence_verify_surprise     = 20.0f;  // decayed go-verify surprise (un-resolvable absence) above which
                                                  // a table is flagged wants_verification (epistemic pull, not removal)
    float verify_surprise_smooth        = 0.10f;  // EMA weight of the go-verify surprise accumulator (new sample
                                                  // weight; 1−this holds the old). Parity with fe_surprise_smooth.
    float existence_verify_gain         = 5.0f;   // epistemic gain (nats) a wants_verification table gets, so the
                                                  // controller drives to a resolving ZED view (confirm-or-remove)
    // LiDAR removal reliability: the model's top slab is a SOLID band, but a real tabletop is a THIN plate, so
    // horizontal beams passing UNDER the top surface exit the band and read as false "free" — unreliable
    // ABSENCE. So by default the LiDAR carve contributes OCCUPANCY only (holds L up / confirms presence) and
    // NEVER drives removal; removal is the camera silhouette's job (predicted-visible-but-absent, which HOLDs
    // when out of FoV). Set true to ALSO trust LiDAR free-space (only where the slab is a faithful solid model).
    // Leg OCCUPANCY: also carve the 4 leg volumes [0, H−t] for occupancy (never free — thin legs, and the hollow
    // space between them must not vote absence). Robustifies confirmation when the thin top slab is at an awkward
    // height for the LiDAR rings (tall legs are far likelier to be struck). Occupancy-only ⇒ cannot false-remove.
    bool  existence_leg_occupancy = true;

    // MEASURED clutter prior for the LiDAR occupancy half (rc::exist::measure_clutter / contrast_delta). The
    // configured ExistenceClutterProb answers "P(a beam returns from an EMPTY footprint anyway)" — but the
    // alternative to a table in a furnished room is a WALL, not empty space, and against that alternative a
    // return carries no information at all. With this on, the prior is measured each cycle from an equal-volume
    // shell around the box and the occupancy evidence becomes a CONTRAST: 0 wherever the box is indistinguishable
    // from what surrounds it. false = the historic hollow_guarded_delta (any occupancy = a full confident hit,
    // which pinned L at the clamp forever and made a phantom over real structure immortal). See table_existence.cpp.
    bool  existence_local_clutter = true;

    // Does the LiDAR get a VOTE on whether a TABLE exists? Default NO. Range data cannot separate a table from
    // any other object of similar extent, so P(return | table) ≈ P(return | that other thing) and its likelihood
    // ratio for table-ness is ≈1. Measured: the real table returned 1.96% of its beams and a phantom sitting on
    // a mistaken object 6.56%, both saturating to the same +2.83/cycle — the channel ranked the phantom 3.3×
    // above the truth and pinned L at the clamp forever. Removal is decided by the ONE sensor that reads a
    // label: predicted-visible-but-absent in the camera silhouette, from a viewpoint where p_detect says YOLO
    // should have seen it, for ExistenceRemoveFrames cycles running. true = restore a voting LiDAR (A/B).
    bool  existence_lidar_confirms = false;

    // Weight silhouette ABSENCE by the shared DETECTOR ENVELOPE (rc::detect::p_detect: two logistics on the
    // projected fill fraction) instead of the one-sided range curve. P(detect) is unimodal in FRAMING — it
    // falls to 0 both when the table is too few pixels AND when it overflows the frame — whereas
    // ExistenceAbsenceRangeRefM/Power only ever discount a FAR view, leaving the agent maximally confident at
    // point-blank range where YOLO is blind. That deleted the real table from 0.46 m. Uses the SAME envelope
    // instance the epistemic planner scores viewpoints with, so the robot drives to the framing removal trusts.
    // false = the historic range-only weighting (A/B). The range keys stay in use for the LiDAR free half.
    bool  existence_absence_envelope = true;

    // ── RT-edge covariance upload ─────────────────────────────────────────────────────────────────
    // Upload the table pose covariance onto the room→table RT edge (rt_covariance_att, 6×6 SE3), mapped from
    // the belief's full Σ over [cx,cy,H,w,h,yaw]: x←cx, y←cy, z←H/2, yaw←ψ; roll/pitch are unobservable (large).
    // rt_cov_scale calibrates the raw variance toward NEES≈1.
    float rt_cov_scale     = 1.0f;
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so room_concept can use the
    // table as an SE(2) pose landmark. OFF by default (consumer side is also flag-gated + defaults OFF).
    bool        publish_object_obs = false;      // TableConcept.PublishObjectObs
    std::string object_obs_frame   = "body";     // TableConcept.ObjectObsFrame (localizer base node)

    // ── Multi-instance birth/associate/death (shared rc::InstanceTracker) ─────────────────────────
    // "table" masks are associated to instances by a covariance-gated global 1-to-1; an unexplained mask
    // spawns a new table. There is NO death-by-occlusion: a table is large static furniture, so a long
    // occlusion is not absence. An instance leaves only via the MERGE operator (two tables cannot share
    // space) or the existence-removal channel (table_existence.cpp). Keep birth_min_sep large so no two
    // table centres can sit that close.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.50f;   // metric XY gate (m) before an instance has a usable covariance
    // Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I). For a table
    // the YOLO mask CENTROID sits well off the fitted geometric CENTRE under a partial view (~0.2–0.5 m),
    // so R must cover that offset or the overconfident fit (σ ~mm) rejects its own real detection.
    float tracker_detection_noise_m = 0.35f;
    int   tracker_birth_frames      = 8;      // frames a mask must stay unexplained before spawning a table
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
    float tracker_birth_min_sep_m   = 0.60f;  // a birth must be ≥ this (m) from every existing table
    // Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of
    // the SMALLER footprint (two tables cannot share space). 0 disables the merge.
    float tracker_merge_overlap = 0.05f;
    // Default geometry for a table BORN from a mask (no prior to seed it); the belief refines from here.
    float tracker_birth_width_m  = 1.0f;
    float tracker_birth_depth_m  = 0.6f;
    float tracker_birth_height_m = 0.75f;

    // ── Ricoh-360 attention gate ──────────────────────────────────────────────────────────────────
    // A ricoh (depth_var>0) 360-RGB YOLO detection is bearing-only: it never births or fits, it only raises
    // attention (process_ricoh_bearings) so the robot seeks a good ZED view.
    float ricoh_attention_conf = 0.60f;   // min YOLO confidence for a ricoh detection to raise attention
    // Ricoh attention association gates (process_ricoh_bearings): small margins on the physical angular size /
    // range roughness when deciding whether an unassigned ricoh bearing already matches a known table.
    float ricoh_attention_angle_margin_rad = 0.05f;  // extra angular tolerance on the table's angular half-size
    float ricoh_attention_range_band_m     = 1.0f;   // extra range band (ricoh range is rough/indicative)
};

// Fill a TableConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
TableConfig load_table_config(const ConfigLoader& cfg);

}  // namespace rc
