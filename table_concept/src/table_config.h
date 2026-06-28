/*
 * table_config.h
 *
 * Plain-data configuration for the table_concept agent, plus a loader that fills
 * it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a new
 * concept agent can copy this file and edit only the keys it needs (mirrors
 * bottle_concept/bottle_config.h).
 */

#pragma once

#include <string>

#include "../../common/robust_metrics/robust_metrics.h"   // RobustLossType

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct TableConfig
{
    // Paths

    // Agent convergence
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   max_direct_fit_points = 400; // strided live points fed straight into the gradient each frame (0=off)
    int   detection_alive_max_frames = 40; // cycles without a fresh table mask before detection_alive=false
    int   M_diverge         = 20;
    float explanation_ratio_thresh = 0.3f;
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float delta_min         = 20.0f;   // min face coverage count
    // Epistemic info-gain selection: score faces by the expected entropy reduction ΔH from the Fisher
    // posterior (Σ) instead of the coverage-deficit proxy, and pick the most informative face.
    int   epistemic_cooldown_cycles = 200;    // min cycles withdrawn after satisfaction
    int   table_log_period_frames = 30;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;
    // Top-band point gate: before the fit/extent terms, keep only points within ±top_band_m of the
    // estimated table-top height, dropping the floor / under-table / clutter population that is
    // attributed to legs and inflates the w/h footprint (and corrupts the leg/height fit). Off by
    // default; the extent diagnostic (n_top/n_leg, observed span vs fitted) shows whether it helps.
    bool  top_band_gate_enabled = true;
    float top_band_m            = 0.08f;   // half-width of the top-surface membership band (m)

    // TableModel parameters (forwarded to TableModelParams)
    float sigma_obs         = 0.05f;
    float lambda_size       = 0.15f;
    float lambda_pos        = 0.05f;
    float lambda_state      = 0.02f;
    float lambda_angle      = 0.01f;
    float lambda_extent     = 2.0f;   // footprint-extent term: fit top rectangle to point span
    float extent_pct_lo     = 0.05f;  // lower percentile of the point span the extent term pins to
    float extent_pct_hi     = 0.95f;  // upper percentile (tighter pair trims the depth-noise edge margin)
    float prior_size_std    = 0.30f;
    // Model-evidence ρ inlier width (m): a fresh mask point counts as "explained" within ~this distance of
    // the box surface; ρ = mean exp(−½(SDF/σ_ρ)²) over the mask drives the evidence-scaled precision (a box
    // too small → mask far outside → low ρ → belief loosens → grows). Wider than sigma_obs so a clean fit
    // reads ρ≈1 (precision persists) while a clearly under-/mis-sized box reads low.
    float evidence_sigma_m  = 0.12f;
    int   optimization_iters = 10;
    float optimization_lr   = 0.05f;
    float grad_clip         = 2.0f;
    std::string optimizer_type = "adam";
    float sgd_momentum     = 0.9f;
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.10f;
    float robust_gnc_start_scale = 0.80f;   // GNC: wide initial robust scale annealed to robust_loss_scale
    float mask_precision = 0.30f;           // RGB-mask silhouette term weight (0 disables)
    int   sil_tangent_samples = 8;          // >0 = height-agnostic occluding-contour (samples/ray); 0 = top-plane only
    // Extent re-open: if the observed point span (5–95 pct) exceeds the model footprint by more than this,
    // the table is bigger than the current (frozen) fit — the "mask extends beyond the RFE" symptom. Re-open
    // so the extent term grows w/h to the points. Self-limiting: closes the gap, then re-settles.
    int   robust_gnc_decay_cycles = 20;     // GNC start scale ramps to target over this many cycles, then off

    // SampleQueue parameters (forwarded to SampleQueueParams)
    int   num_angle_bins               = 24;
    int   num_z_bins                   = 10;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.08f;
    int   min_frames_before_historical = 10;
    int   historical_warmup_frames     = 50;
    int   max_new_points_per_frame     = 5;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float rfe_weight_gain              = 0.25f;
    float min_anchor_weight            = 0.12f;
    float edge_bonus_weight            = 0.3f;
    float edge_proximity_threshold     = 0.05f;

    // Observability-aware warm-start acceptance
    float warm_pts_min                 = 12.0f;
    float warm_pts_max                 = 30.0f;
    // Belief stabiliser (always on; the A/B toggles were removed in Stage 1): per-DOF SDF-likelihood
    // Fisher information, accumulated across viewpoints, drives the acceptance lerp through the Kalman
    // gain λ_j = obs_j/(Y_pred_j + obs_j) — a well-observed DOF hardens (belief→knowledge) while an
    // unseen face stays plastic. The keys below are its numeric tunings.
    float fisher_info_decay            = 0.95f;   // stiffener (acc): scalar fading memory; <1 forgets, 1 = pure accumulation
    // Quality-aware maturity (always on): normalise the views accumulator by a (decaying) running quality
    // bar so a CLOSE approach after far glimpses re-opens the fit (rescales the stale far-view "views"
    // down) instead of staying locked at the misrepresented far fit.
    float fisher_peak_decay            = 1.0f;    // quality-bar slow forgetting (1 = non-decaying)
    // Only re-open the fit when a view beats the quality bar by ≥ this ratio (a GENUINE close approach,
    // ~10–100×), and fold the new high in as an EMA — so ordinary per-frame obs flutter (which swings
    // orders of magnitude) no longer re-ratchets the bar and keeps a DOF perpetually plastic (the width
    // wander). Higher ratchet = stiffer / harder to re-open; lower EMA = the bar moves up more gently.
    float fisher_peak_ratchet          = 2.5f;
    float fisher_peak_ema              = 0.30f;
    // Calibrated-covariance branch (fisher_info_raw): the information-filter PREDICT step inflates
    // each DOF's covariance by process noise Q per fresh frame — Y_pred = (Y_prev⁻¹ + Q)⁻¹ — so the
    // posterior precision reaches a finite STEADY state (belief converges to a real uncertainty floor)
    // instead of the scalar-decay collapse toward zero. Physically Q = how much the table pose/size
    // knowledge stales per re-observation (dominated by the robot's localization jitter). Per-frame
    // process-noise std in each DOF's native units; variance Q = std².
    float fisher_process_std_m         = 0.005f;  // length DOFs (cx,cy,w,h,H,leg,inset): 5 mm / fresh frame
    float fisher_process_std_yaw       = 0.01f;   // yaw: ~0.57° / fresh frame
    // Freeze-on-bad-fit (always on): a FRESH frame whose raw-fit free energy is ≫ the instance's running
    // FE level is a contaminated observation (clutter / partial-or-wrong mask → points the model cannot
    // explain; extent blows up, FE spikes ~20× vs ~1). Skip the acceptance, keep the belief, recompute FE
    // for reporting. The baseline is an EMA over ACCEPTED frames only (bad frames don't bias it). This and
    // freeze-on-stale (no fresh mask → no update) are the information-filter axiom: untrustworthy → ignore.
    float bad_fit_fe_ratio             = 4.0f;    // reject a fresh frame if raw_fe > ratio · fe_baseline
    float fe_baseline_ema              = 0.10f;   // EMA weight for the accepted-FE running baseline
    // Size RATCHET: a size DOF (w,h,H,leg) grows at its full acceptance gain but SHRINKS at gain·this — a
    // table is rigid, so a far/occluded partial view (small visible extent) must NOT collapse it ("shrank
    // to a third across the room"). Small = strong ratchet; 1.0 = symmetric (off). Growth is unaffected.
    float size_shrink_gain             = 0.05f;
    // (B) Fisher-gradient clamp: observation_information builds the per-DOF Fisher via a central finite
    // difference of the NON-smooth min() box-SDF. A point sitting on a face-switch/corner makes the
    // difference a discontinuity JUMP, not a derivative, spiking |∂SDF/∂θ| by ~1000× → a single point
    // forces the Kalman gain K→1 and bypasses all accumulated stabilisation. For a well-behaved SDF
    // |∂SDF/∂θ| is O(1); clamp the per-point slope to this bound before squaring. 0 = disabled (for A/B).
    float fisher_grad_clamp            = 2.0f;
    // Maturity stiffening (always on): scale the per-frame Kalman acceptance gain by vh/(vh + views), so a
    // well-observed DOF (many equivalent views) barely moves on any single frame — history dominates a lone
    // noisy measurement — while an unseen DOF (views≈0 → factor≈1) stays fully plastic. The views signal is
    // the normalised Fisher accumulator (fisher_info[j]).
    float fisher_views_half            = 4.0f;    // equivalent-views at which the maturity stiffener halves the gain (lower = hardens faster)
    // Counter-evidence gate (CUSUM / sequential change-detector; always on): asymmetric, ratcheting
    // trust on top of the Kalman gain. Per DOF, the surprise of each fresh frame — innovation |raw−belief|
    // in deadband units — accumulates a SIGNED counter-evidence S_j only while one-sided; a lone glitch
    // bumps S_j once and decays away (rejected, gain→0), while a CONTIGUOUS same-direction run grows S_j
    // until it overcomes a barrier scaling with the filter's accumulated confidence. On unlock the DOF
    // snaps to the new evidence (gain→1), S_j resets, and its accumulated precision is deflated.
    float counter_evidence_band_m      = 0.02f;   // length-DOF surprise deadband (m): innovations within this are "coherent"
    float counter_evidence_band_rad    = 0.05f;   // yaw surprise deadband (rad)
    // Barrier = base + lambda·fisher_info[j], SPLIT into position (cx,cy) vs size/shape (w,h,H,leg,yaw,
    // inset) groups so the table CENTRE can be locked harder than its extents (position drifts one-sided
    // with robot localisation and was unlocking ~3× more often than w/h). Higher base/lambda → a longer,
    // more confident coherent run is needed before that group re-adapts.
    float counter_evidence_base_pos    = 6.0f;    // position barrier floor (excess-bands): centre is heavily locked
    float counter_evidence_lambda_pos  = 1.0f;    // extra position barrier per accumulated equivalent-view
    float counter_evidence_base_size   = 1.5f;    // size/shape barrier floor (excess-bands)
    float counter_evidence_lambda_size = 0.2f;    // extra size/shape barrier per accumulated equivalent-view
    // Yaw is decoupled from the size group with a HIGH barrier: a static rectangle's orientation, once
    // observed, is its most rigid property and the easiest to confuse from a partial/ambiguous view —
    // it was unlocking (ceg_yaw=1) every few minutes and rotating the table. Make it hard to unlock.
    float counter_evidence_base_yaw    = 12.0f;   // yaw barrier floor (excess-bands) — yaw is heavily locked
    float counter_evidence_lambda_yaw  = 1.0f;    // extra yaw barrier per accumulated equivalent-view
    // Yaw observability gate: yaw is only meaningful for a clearly RECTANGULAR footprint. Scale the yaw
    // acceptance gain by clamp01(|w−h|/max(w,h) / aspect_ref) so a near-SQUARE table (yaw geometrically
    // unobservable) freezes its yaw at the seed instead of chasing noise. 1.0 ≈ fully observable above
    // a (aspect_ref) fractional w/h difference.
    float yaw_obs_aspect_ref           = 0.12f;
    float counter_evidence_decay       = 0.8f;    // per-coherent-frame pay-down of S_j (a coherent frame is evidence FOR the belief)
    float counter_evidence_unlock_deflate = 0.3f; // on unlock, multiply that DOF's accumulated info by this (concede the old estimate)
    float counter_evidence_step_cap    = 3.0f;    // cap the per-frame CUSUM increment (excess-bands) so ONE big glitch can't
                                                  // jump the barrier — unlock then requires a RUN of coherent surprise (proper SPRT)
    // ── YOLO mask-score weighting of the update ─────────────────────────────────────────────────
    // The detector's confidence is the measurement's reliability: weight each fresh frame's observation
    // Fisher information by w(conf) so a low-score mask contributes ~zero precision → its Kalman gain
    // ≈ 0 and it barely moves the belief (and barely hardens it). w = clamp01((conf−floor)/(1−floor))^power:
    // below `floor` the mask is effectively ignored; `power` sets how steeply trust ramps with score.
    // w = clamp01((conf−floor)/(ref−floor))^power : 0 at/below `floor`, 1 at/above `ref`. Set `floor`
    // just above the detector's NOISE score and `ref` at a CONFIDENT score — for a partially-viewed
    // table YOLO scores stay low (~0.3–0.45), so floor/ref must sit in that band or every mask is
    // nuked (w=0 → the Fisher accumulator dies → no maturity stiffening → unstable).
    float mask_conf_floor  = 0.2f;   // YOLO score at/below which the mask is ~ignored in the update
    float mask_conf_ref    = 0.5f;   // YOLO score at/above which the mask is fully trusted (w=1)
    float mask_conf_power  = 2.0f;   // steepness of the trust ramp between floor and ref
    std::string fisher_csv_path        = "";      // if non-empty, append per-cycle Fisher-filter evolution to this CSV (for plotting)
    // Upload the table pose covariance onto the room→table RT edge (rt_covariance_att, 6×6 SE3),
    // built from the Fisher filter's per-DOF posterior precision (var = rt_cov_scale / fisher_info_raw):
    // x←cx, y←cy, z←H/2, yaw←ψ are data-driven; roll/pitch are unobservable (large). rt_cov_scale
    // calibrates the raw curvature toward NEES≈1 (raw precision over-counts spatially-correlated
    // points), like bottle_concept's cov_eff_scale. Written only when the geometry is (re)published.
    float rt_cov_scale  = 1.0f;
    float warm_confidence_decay         = 0.70f;
    float warm_confidence_coverage_gain = 0.35f;
    float warm_confidence_residual_gain = 0.65f;

    // ── Multi-instance birth/associate/death (shared rc::InstanceTracker) ─────────────────────────
    // OFF → legacy prior-scaffold + greedy nearest-mask. ON → "table" masks are associated to instances
    // by a covariance-gated global 1-to-1; an unexplained mask spawns a new table; an instance unobserved
    // for tracker_death_frames is retired. Tables are large static furniture: keep birth_min_sep large
    // (no two table centres that close) and death_frames LARGE so a long occlusion doesn't drop a table.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.50f;   // metric XY gate (m) before an instance has a usable covariance
    // Detection-noise std R (m) added to the fit cov in the Mahalanobis gate (S = P + R²I). For a table
    // the YOLO mask CENTROID sits well off the fitted geometric CENTRE under a partial view (~0.2–0.5 m),
    // so R must cover that offset or the overconfident fit (σ ~mm) rejects its own real detection. Large
    // for tables; the 4 m table separation keeps neighbours from cross-matching even at this R.
    float tracker_detection_noise_m = 0.35f;
    int   tracker_birth_frames     = 8;       // frames a mask must stay unexplained before spawning a table
    int   tracker_death_frames     = 300;     // frames an instance may go unobserved before retirement (large)
    bool  tracker_death_enabled    = false;   // OFF: a table is rigid, persistent furniture — never retired by
                                              // a miss timer (a long occlusion is not absence). Removal is by
                                              // the MERGE operator only (two tables can't share space).
    float tracker_birth_min_sep_m  = 0.60f;   // a birth must be ≥ this (m) from every existing table
    // Physical exclusion: collapse two instances whose oriented footprints overlap by ≥ this fraction of
    // the SMALLER footprint (two tables cannot share space). 0 disables the merge.
    float tracker_merge_overlap    = 0.30f;
    // Default geometry for a table BORN from a mask (no prior to seed it); the fit refines from here.
    float tracker_birth_width_m    = 1.0f;
    float tracker_birth_depth_m    = 0.6f;
    float tracker_birth_height_m   = 0.75f;
    // Prior PRECISION (size std, m) for a born table: the Birth* dims are applied not just as the optimizer
    // seed but as the size-KL prior mean with THIS σ — so the size is pinned near standard furniture
    // dimensions under weak/partial early views and only yields to genuine evidence (smaller = stiffer).
    // Without it a born table falls back to the loose generic prior_size_std and the first far/partial
    // mask can reshape it freely. Applied in ensure_instance when no matching prior entry exists.
    float tracker_birth_size_std   = 0.15f;
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
};

// Fill a TableConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
TableConfig load_table_config(const ConfigLoader& cfg);

}  // namespace rc
