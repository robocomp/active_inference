/*
 * cabinet_config.h  —  plain-data configuration for the cabinet_concept agent + its ConfigLoader loader.
 *
 * Kept separate from SpecificWorker so a new concept agent can copy this file and edit only the keys it
 * needs (mirrors bottle_concept/bottle_config.h). Every field carries its default here; the loader
 * (cabinet_config.cpp) overrides from the RoboComp TOML when the key is present.
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct CabinetConfig
{
    // ── Agent convergence & cadence ───────────────────────────────────────────────────────────────
    float state_eps                    = 0.04f;   // Σ|Δstate| convergence threshold between cycles (m+rad)
    int   K_stable                     = 30;      // consecutive converged cycles before model_stable
    int   detection_alive_max_frames   = 40;      // cycles without a fresh mask before detection_alive=false

    // ── Primary-input stream gate (readiness + staleness) — LIFECYCLE, not a belief knob ──────────────
    // Demote Operating→Degraded→Waiting when the voxelizer's `masks` node stops advancing its mask_frame_id
    // for this long (producer dead/stalled) — don't integrate stale evidence; re-admit when it returns.
    // Orthogonal to ai2_age_nominal_dt_s (belief-axis Σ-aging). MUST exceed the voxelizer HOLD_ENTER_S so a
    // legitimately empty scene (still-advancing counter) never trips it. 0 = disable the gate.
    int   masks_stall_timeout_ms       = 3000;
    float obs_distance                 = 1.8f;    // d_obs for the epistemic planner
    int   epistemic_cooldown_cycles    = 200;     // min cycles withdrawn after satisfaction
    int   cabinet_log_period_frames      = 30;      // per-cycle log throttle
    bool  verbose_log                    = false;   // false = quiet terminal (only births/merges/removals);
                                                    // true = per-cycle diagnostics (dashboard/tracker/split/…)
    int   voxel_bank_max_points        = 4000;    // cap on the cabinet-owned voxel memory bank
    float voxel_bank_quantization_m    = 0.02f;   // voxel-bank dedup grid (m)
    float voxel_select_radius_margin_m = 0.50f;   // XY margin (m) around the model for voxel-bank selection
    float voxel_select_height_margin_m = 0.25f;   // Z margin (m) around the model for voxel-bank selection

    // ── Top/leg SDF split band ────────────────────────────────────────────────────────────────────
    // Forwarded to CabinetModelParams.sigma_obs: a mask point within TOP_THICKNESS + sigma_obs below the top
    // face is attributed to the slab (candidate) vs a leg.
    float sigma_obs                 = 0.05f;
    // On-surface membership for the candidate/residual split in CabinetFitter::observe.
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (CABINET.md) — full-covariance recursive filter ──────────────────────────
    float ai2_sigma_base_m    = 0.03f;   // base on-surface obs noise std (m); R = σ² (+ motion_var + …)
    float ai2_clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter mixture component
    float ai2_clutter_scale_m = 0.12f;   // a point further than ~this from every surface is likely clutter
    float ai2_prior_size_std  = 0.30f;   // broad size prior std (m) on w,h,H
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
    // EGO-MOTION → common-mode ("still to update"). The shared frame error is the camera's ego-DISPLACEMENT
    // during capture: pos_var=(gain·|v|·dt)², yaw_var=(gain·|ω|·dt)². gain is dimensionless (exposure-vs-frame
    // + deprojection amplification). At ~0.2 m/s or ~0.2 rad/s it dominates the base common-mode ⇒ geometry
    // frozen, existence-only. 0 disables. Continuous — no motion gate.
    float ai2_common_mode_motion_gain = 8.0f;   // strong: even slow driving (motion_dotd~0.1) freezes geometry;
                                                 // only a near-still frame (motion_dotd→0) commits a full update
    // STATIC range weighting (motion-free): the common-mode error grows with view distance, so a far, vague
    // mask cannot resolve pose — orientation least of all (a far view confirms existence, can't rotate the
    // cabinet). Continuous, no gate. lat feeds R + position common-mode (m per m of range); yaw feeds the yaw
    // common-mode (rad per m), the binding term. Set 0 to disable. Range Z comes from the voxelizer (mask_range).
    float ai2_range_noise_lat_per_m  = 0.02f;  // lateral deprojection std growth (m per m of range)
    float ai2_range_noise_yaw_per_m  = 0.03f;  // yaw common-mode std growth (rad per m of range)
    float ai2_range_noise_size_per_m = 0.08f;  // SIZE (w,h,H) common-mode std growth (m per m of range): a distant
                                               // mask can't reshape/inflate a converged cabinet (freezes geometry afar)
    // THRESHOLD (truncation gate): skip the geometric update (predict only) when this fraction of the mask
    // silhouette is on the image border — a truncated mask is a biased extent, not measurement noise.
    float ai2_trunc_gate_frac = 0.10f;
    int   ai2_gn_iters        = 4;       // Gauss-Newton iterations per frame

    // ── RUN-specific structural terms (see cabinet_belief.h for the full rationale) ───────────────
    // Wall-flush factor: the back face is never observed, so (centre-along-normal, depth) is a null
    // direction; the wall breaks it. The precision is the marginal of a {flush, free-standing}
    // mixture, so it decays continuously with the gap — a free-standing island needs no special case.
    float wall_precision          = 400.0f;  // 1/m² at zero gap (0 = OFF). NB: bumping this to force WT1's
                                              // back onto the wall did NOT help — d and the flush are a
                                              // coupled null-direction, so a precision knob just moves the
                                              // equilibrium. The clean fix is the wall-chart reparametrization
                                              // (θ_wall with δ_lat≡0), see WALL_KEYED_REKEY_PLAN.md.
    float wall_reach_m            = 0.35f;   // gap scale over which the flush hypothesis loses weight
    float wall_sigma_m            = 0.02f;   // room-model wall position uncertainty (m)
    float wall_parallel_precision = 200.0f;  // on sin(angle between run axis and wall) (0 = OFF)
    // Room-axis (Manhattan) yaw prior: a strong, wall-independent pull toward the nearest π/2 room axis
    // so every run (incl. a peninsula/island touching a wall on only one short side) stays parallel to
    // the room. Silent once aligned. capture_rad ≤0 ⇒ always active (see cabinet_belief.h).
    float room_axis_precision     = 300.0f;  // 1/rad² on the yaw→nearest-axis residual (0 = OFF)
    float room_axis_capture_rad   = 0.0f;    // release beyond |Δyaw| (rad); 0 ⇒ always on
    // Birth on the dominant room axis instead of the raw PCA diagonal — stops an L-shaped corner mask
    // from birthing one oblique box spanning both walls (which grow-only extent can never retract).
    bool  seed_room_axis_snap     = true;
    // Wall-split: attribute each mask point to its nearest room wall; points flush to a DIFFERENT wall than
    // the run they feed are pulled out of the fit (→ residual-birth of the neighbour). Splits one L/U-corner
    // mask into two axis-aligned runs at the corner bisector. See CabinetFitter::observe_slice.
    // L-corner mask split (a cabinet run cannot be L-shaped ⇒ an L-mask is two runs). Split one 'cabinet'
    // mask into its two perpendicular room-axis arms upstream (cabinet_lshape_split.h), so each run fits a
    // clean single-arm cloud. Replaces the fragile per-instance observe-time split.
    // Stage 2 kitchen-of-runs model: the (wall,tier) cells own the WallRunBeliefs (identity IS the cell, so no
    // birth/associate/merge, and crossings / 10 cm / ceiling boxes are unrepresentable). When true it fully
    // REPLACES the tracker/fitter/residual-birth path. Off = classic pipeline. See cabinet_kitchen.h.
    bool  kitchen_model           = false;
    float ceiling_height_m        = 2.6f;    // z-domain upper bound for wall-run tops (H_room)
    // Stillness/VOR common-mode (ALIGNED with chair_concept's non-movement validation): ego-motion from the
    // transform chain (producer-independent) × off-axis periphery. ego_motion_pos_var = (gain·motion_mag)²·periph
    // with motion_mag = max(mean|motion_dotd|, ego_lin + ang_lever·ego_ang) and periph = (radius/periph_ref)².
    // A still OR well-centred frame ⇒ ~0 common-mode ⇒ full update authority; moving AND peripheral ⇒ confirm-only.
    // Centred-while-moving stays trusted (the "pure-translation" exception, emergent from the periphery factor).
    float kitchen_motion_cm_gain  = 0.30f;   // position shared-error std per (m/s) of motion  (chair MotionCmPosGain)
    float kitchen_ang_lever_m     = 2.0f;    // rad/s → m/s lever for camera rotation           (chair AI2AngLeverM)
    float kitchen_periph_ref      = 0.50f;   // centroid radius (focal-norm) at which periphery saturates (chair AI2PeriphRef)
    // Kitchen RETIREMENT channel: LiDAR evidence of absence on the BORN cells. The mask channel can only add
    // presence to an active cell (a look-away is not evidence of absence) and the free-space carve only reshapes,
    // so without this a cell that ever births is immortal — the "phantom cabinets are never removed" defect.
    // A beam that traverses a run and lands on the WALL BEHIND it refutes it; occupancy (front-face/top returns)
    // defends it; an occluded run is not probed and simply holds. Reuses the classic path's Existence* sensor
    // rates (detection/clutter/σ/absence-range/remove-frames) — no new tuning surface. See cabinet_kitchen.h.
    bool  kitchen_lidar_existence = false;
    // Per-cycle CSV of the kitchen cells (state + existence + the absence evidence that drives retirement). The
    // birth-surprise probe reads fitter_->instances(), which is EMPTY in kitchen mode, so this is the only
    // observability the kitchen model has. Empty path = off.
    std::string kitchen_cells_csv_path = "";
    bool  counter_evidence_enabled = true;   // ingest 'counter'/'countertop' masks as top-face run evidence
    bool  lshape_split_enabled    = true;
    int   lshape_min_arm_pts      = 500;     // an arm (and the peeling residue) must exceed this to split
    float lshape_bin_m            = 0.15f;   // histogram bin for locating each arm's line
    float lshape_arm_halfwidth_m  = 0.45f;   // half-width of an arm's footprint band (≈ half a carcass depth)
    // Censored along-axis containment: the ONLY channel that can lengthen a run (the per-point
    // mixture cedes points past the end cap to clutter). One-sided ⇒ inert once the box contains
    // the cloud; shrinking belongs to the free-space channel, never here.
    float extent_precision        = 800.0f;  // 1/m² on the end-containment residuals (0 = OFF)
    // Standard kitchen carcass priors per TIER, selected by model evidence (never a height cut).
    float base_depth_m = 0.60f, base_depth_std = 0.04f;   // base cabinets are ~60 cm deep almost
                                                          // universally → STRONG prior. The back face is
                                                          // never observed, so without this the visible
                                                          // near-face slab shrinks d (WT1 fit 0.28); the
                                                          // wall-flush pins the back, this pins the depth.
    float base_z0_m    = 0.00f, base_z0_std    = 0.02f;   // a base carcass sits ON the floor: strong prior
                                                          // so a free-standing run whose floor contact is
                                                          // occluded (only its top/counter seen) is not
                                                          // lifted off z0=0 by the visible points.
    float base_z1_m    = 0.90f, base_z1_std    = 0.06f;   // worktop height
    float wall_depth_m = 0.35f, wall_depth_std = 0.08f;
    float wall_z0_m    = 1.45f, wall_z0_std    = 0.15f;
    float wall_z1_m    = 2.10f, wall_z1_std    = 0.20f;
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
    float erosion_px_std         = 0.020f;   // shared per-frame depth tilt prior std (m/rad) — the yaw nuisance
    float depth_bias_std         = 0.015f;   // shared per-frame depth bias prior std (m)
    float depth_scale_std        = 0.010f;   // shared per-frame depth scale prior std (fraction)
    // EXPERIMENTAL birth-surprise probe (read-only): read residual_concept's `grid` node as an unexplained-
    // occupancy (surprise) field, cluster it, and LOG uncovered high-surprise regions next to the tracker's
    // birth decision — to check whether surprise flags births cleanly before it drives the lifecycle. OFF = no
    // read, no log, zero cost. See birth_surprise_probe.h. Writes etc/birth_surprise.csv when on.
    bool birth_surprise_probe = false;

    // ── YOLO-INDEPENDENT LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h) ───────
    // Second, sensor-independent evidence channel: lidar3D returns landing on the cabinet (legs + rim) pin the
    // extent and centre in METRIC range, ALONG the viewing ray — an error mechanism uncorrelated with the
    // YOLO segmentation, so it attacks the mask-erosion under-size the mask cannot self-correct. Consumes
    // the lidar3D media plane via CabinetLidarIngestor; dormant (no DDS participant) while precision == 0.
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
    // THRESHOLD (outlier step guard): a static cabinet cannot physically move this far in one frame, so a GN
    // step whose centre jump exceeds it is an OUTLIER frame (corrupted mask cloud / one-sided LiDAR runaway)
    // → reject the update, restore state+Σ, widen Σ. 0 disables. Prevents the cx=−200m runaway.
    float max_step_m = 1.0f;

    // ── Coverage / traction (EXISTENCE_BELIEF_PLAN.md) ────────────────────────────────────────────
    // Grow-only pull from on-plane mask points the mixture ceded to clutter, so a model under-covering a
    // large mask grows to explain it (fixes cabinet_1.png). 0 = OFF.
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

    // ── Scene-object non-penetration (cabinet_wall_run_belief.h + cabinet_kitchen.h) ────────────────
    // A run may not cross another agent's furniture (fridge/table/…) read from the shared graph. The factor
    // retracts a penetrating END onto the object; a run mostly ENGULFED by an object is retired (the on-fridge
    // false-cabinet the LiDAR carve can't reach). Precision should DOMINATE extent_precision (≥ 800). 0 = OFF.
    float object_exclusion_precision = 0.0f;   // 1/m² per penetrating end (retract-only)
    float object_exclusion_margin_m  = 0.03f;  // flush clearance off the object boundary
    float object_engulf_frac         = 0.60f;  // retire when along-wall AND depth AND z overlap all exceed this

    // ── Footprint SECOND-MOMENT factor (cabinet_belief.h) ───────────────────────────────────────────
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
    // Baseline EMA rates: DOWN fast (consolidate a better fit), UP slow (a sustained mismatch — the cabinet
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
    // the per-point obliquity yaw cap (kObliquityYawGain in cabinet_fitter.cpp) were validated live 2026-07-11 and
    // are now always on — no flags. The MOMENT-channel backoff below is still A/B.
    // obliquity_moment_gain: same view-angle backoff, but on the MOMENT channel — moment_extra_var +=
    //   (gain·(1/|cosθ|−1))². The per-point cap only protects the GN; the CSV rogue rotations came in on the
    //   MOMENT channel (dyaw_moment), which the per-point cap never sees. A grazing/foreshortened frame biases the
    //   2D inertia → back the whole moment off, not just its per-point yaw. 0 = OFF.
    float obliquity_moment_gain        = 0.0f;
    // footprint_moment_completeness_gain / _min_completeness: forwarded to CabinetBeliefParams — inflate the moment
    //   measurement variance as the observed footprint area falls below the believed area (partial view). This is
    //   the direct fix for the going-away / detour-return yaw snap (ai2_log.csv: completeness≈0.01). 0 = OFF.
    float footprint_moment_completeness_gain = 0.0f;
    float footprint_moment_min_completeness  = 0.02f;

    // ── Existence / removal (common/existence_belief.h) ───────────────────────────────────────────
    // Each cycle carve the LiDAR sweep against the cabinet footprint → occupancy/free-space log-odds; remove
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
                                                  // a cabinet is flagged wants_verification (epistemic pull, not removal)
    float existence_verify_gain         = 5.0f;   // epistemic gain (nats) a wants_verification cabinet gets, so the
                                                  // controller drives to a resolving ZED view (confirm-or-remove)
    // LiDAR removal reliability: the model's top slab is a SOLID band, but a real tabletop is a THIN plate, so
    // horizontal beams passing UNDER the top surface exit the band and read as false "free" — unreliable
    // ABSENCE. So by default the LiDAR carve contributes OCCUPANCY only (holds L up / confirms presence) and
    // NEVER drives removal; removal is the camera silhouette's job (predicted-visible-but-absent, which HOLDs
    // when out of FoV). Set true to ALSO trust LiDAR free-space (only where the slab is a faithful solid model).
    // Leg OCCUPANCY: also carve the 4 leg volumes [0, H−t] for occupancy (never free — thin legs, and the hollow
    // space between them must not vote absence). Robustifies confirmation when the thin top slab is at an awkward
    // height for the LiDAR rings (tall legs are far likelier to be struck). Occupancy-only ⇒ cannot false-remove.
    // A run is a SOLID box (unlike a table's thin slab on legs), so LiDAR free-space evidence is
    // model-consistent and MAY drive removal. Default OFF so the first live runs prove it before it
    // can delete anything. (There are no legs on a run — the table's leg-occupancy pass is gone.)
    bool  existence_lidar_absence = false;

    // ── RT-edge covariance upload ─────────────────────────────────────────────────────────────────
    // Upload the cabinet pose covariance onto the room→cabinet RT edge (rt_covariance_att, 6×6 SE3), mapped from
    // the belief's full Σ over [cx,cy,H,w,h,yaw]: x←cx, y←cy, z←H/2, yaw←ψ; roll/pitch are unobservable (large).
    // rt_cov_scale calibrates the raw variance toward NEES≈1.
    float rt_cov_scale     = 1.0f;
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so room_concept can use the
    // cabinet as an SE(2) pose landmark. OFF by default (consumer side is also flag-gated + defaults OFF).
    bool        publish_object_obs = false;      // CabinetConcept.PublishObjectObs
    std::string object_obs_frame   = "body";     // CabinetConcept.ObjectObsFrame (localizer base node)

    // ── Multi-instance birth/associate/death (shared rc::InstanceTracker) ─────────────────────────
    // "cabinet" masks are associated to instances by a covariance-gated global 1-to-1; an unexplained mask
    // spawns a new cabinet; an instance unobserved for tracker_death_frames is retired (death off by default).
    // Cabinets are large static furniture: keep birth_min_sep large (no two cabinet centres that close) and
    // death_frames LARGE so a long occlusion doesn't drop a cabinet.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.50f;   // metric XY gate (m) before an instance has a usable covariance
    // Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I). For a cabinet
    // the YOLO mask CENTROID sits well off the fitted geometric CENTRE under a partial view (~0.2–0.5 m),
    // so R must cover that offset or the overconfident fit (σ ~mm) rejects its own real detection.
    float tracker_detection_noise_m = 0.35f;
    int   tracker_birth_frames      = 8;      // frames a mask must stay unexplained before spawning a cabinet
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
    int   tracker_death_frames      = 300;    // frames an instance may go unobserved before retirement (large)
    float tracker_birth_min_sep_m   = 0.60f;  // a birth must be ≥ this (m) from every existing cabinet
    float tracker_z_gate_m          = 0.60f;  // association VERTICAL gate: a mask can't bind a track >this in z
                                              // apart → WALL units (z≈1.7) never fuse with BASE runs (z≈0.35)
    // Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of
    // the SMALLER footprint (two cabinets cannot share space). 0 disables the merge.
    float tracker_merge_overlap = 0.05f;
    // Collinear-run merge (the operator that fuses fragments of one run): acceptance width in joint-σ, and
    // a floor on the along-axis gap tolerance (m) since two fragments seen seconds apart legitimately abut
    // with a gap on the order of the coarse mask boundary. See rc::geom::collinear_merge.
    float merge_n_sigma     = 3.0f;
    float merge_gap_floor_m = 0.30f;
    // Default geometry for a cabinet BORN from a mask (no prior to seed it); the belief refines from here.
    float tracker_birth_width_m  = 1.0f;
    float tracker_birth_depth_m  = 0.6f;
    float tracker_birth_height_m = 0.75f;

    // ── Residual-driven birth (SpecificWorker::birth_from_residual + cabinet_residual_birth.h) ─────
    // Cluster the pooled model-unexplained (residual) points; a coherent, elongated, separated arm that no
    // believed run covers (e.g. the perpendicular arm of an L-shaped corner mask) matures over a few cycles
    // into its own axis-aligned "cabinet_N". A residual-born run then footprint-claims the shared mask slice
    // so it is actually fed (the tracker's single-assignment gives the slice to the parent).
    bool  residual_birth_enabled  = true;
    int   residual_birth_frames   = 4;      // consecutive cycles a candidate must persist before birth
    float residual_birth_match_m  = 0.40f;  // candidate↔candidate match radius across cycles (debounce)
    int   residual_birth_min_pts  = 600;    // min residual cluster mass to consider
    float residual_birth_sep_m    = 0.60f;  // min separation of the arm from every existing run
    float residual_claim_frac     = 0.15f;  // slice claimed by a residual-born run if ≥ this fraction of its
                                            // support points fall on the run's footprint
    float residual_claim_margin_m = 0.20f;  // footprint expansion for the claim test

    // ── Ricoh-360 attention gate ──────────────────────────────────────────────────────────────────
    // A ricoh (depth_var>0) 360-RGB YOLO detection is bearing-only: it never births or fits, it only raises
    // attention (process_ricoh_bearings) so the robot seeks a good ZED view.
    float ricoh_attention_conf = 0.60f;   // min YOLO confidence for a ricoh detection to raise attention
    // Ricoh attention association gates (process_ricoh_bearings): small margins on the physical angular size /
    // range roughness when deciding whether an unassigned ricoh bearing already matches a known cabinet.
    float ricoh_attention_angle_margin_rad = 0.05f;  // extra angular tolerance on the cabinet's angular half-size
    float ricoh_attention_range_band_m     = 1.0f;   // extra range band (ricoh range is rough/indicative)
};

// Fill a CabinetConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
CabinetConfig load_cabinet_config(const ConfigLoader& cfg);

}  // namespace rc
