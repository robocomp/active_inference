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
    std::string priors_path = "etc/object_priors.toml";

    // Agent convergence
    float fe_eps            = 1e-3f;   // |ΔFE| threshold for convergence (legacy)
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   max_direct_fit_points = 400; // strided live points fed straight into the gradient each frame (0=off)
    int   detection_alive_max_frames = 40; // cycles without a fresh table mask before detection_alive=false
    int   M_diverge         = 20;
    float staleness_frames  = 90.0f;
    float explanation_ratio_thresh = 0.3f;
    float write_threshold   = 1e-3f;   // min ‖Δθ‖ before writing RT to DSR
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float delta_min         = 20.0f;   // min face coverage count
    float gain_threshold    = 0.1f;    // min gain for the legacy coverage-deficit epistemic proxy
    // Epistemic info-gain selection: score faces by the expected entropy reduction ΔH from the Fisher
    // posterior (Σ) instead of the coverage-deficit proxy, and pick the most informative face.
    bool  epistemic_use_info_gain = true;
    float epistemic_min_info_gain = 0.3f;   // WITHDRAW threshold: ΔH (nats) below which the affordance is dropped
    // Schmitt hysteresis (anti-oscillation): a satisfied table stays withdrawn for cooldown cycles AND
    // until ΔH climbs back above the re-arm threshold (> withdraw), so it can't re-offer the instant ΔH
    // crosses the low withdraw threshold again.
    float epistemic_rearm_info_gain = 5.0f;   // ΔH (nats) above which a satisfied table re-arms
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
    bool  top_band_gate_enabled = false;
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
    float sil_reopen_residual_m = 0.15f;    // a fresh mask with silhouette residual above this re-opens a converged fit
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
    float warm_coverage_min_side       = 2.0f;
    float warm_rho_freeze              = 0.25f;
    float warm_size_pts_release        = 0.60f;   // rho_pts above which w/h trust points & ignore the side-coverage freeze
    int   warm_settle_cycles           = 40;      // cycles for acceptance gain to decay full→floor after a new-evidence burst
    int   warm_reopen_admit            = 8;       // net new queue anchors in a cycle that count as a fresh viewpoint (reset maturity)
    float warm_settle_floor            = 0.05f;   // acceptance-gain multiplier once mature (lower = stiffer lock; recovers on new evidence)
    float warm_info_half               = 20.0f;   // accumulated per-face view-info at which the w/h gain halves (lower = hardens faster)
    // Fisher information filter: replaces the face-coverage view-info PROXY with the real per-DOF
    // observation Fisher information (curvature of the SDF data-likelihood). Each fresh mask
    // contributes calibrated, anisotropic evidence per DOF; the accumulated info stiffens the
    // acceptance gain so a well-observed extent hardens (belief→knowledge) while an unseen face
    // stays plastic — the stabiliser for successive viewpoints as the robot orbits the table.
    bool  fisher_filter_enabled        = true;
    // Kalman-gain stiffness: drive each DOF's acceptance lerp gain directly from the calibrated
    // information filter as λ_j = obs_info_j / (Y_pred_j + obs_info_j) — the per-DOF Kalman gain —
    // instead of the coverage/confidence/InfoHalf heuristic stack. Well-observed DOFs get a small
    // gain (stiff) yet still snap to a genuinely high-information view; uniform across all 8 DOFs.
    bool  fisher_kalman_stiffness      = false;
    float fisher_info_decay            = 1.0f;    // stiffener (acc): scalar fading memory; <1 forgets, 1 = pure accumulation
    // Quality-aware maturity: normalise the views accumulator by a (decaying) running quality bar so a
    // CLOSE approach after far glimpses re-opens the fit (rescales the stale far-view "views" down)
    // instead of staying locked at the misrepresented far fit. Off → every view counts ~1 (far hardens
    // as fast as close — the over-stiffening that froze the fit).
    bool  fisher_quality_rescale       = true;
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
    // (A) Freeze-on-stale: the information-filter axiom — no measurement, no update. When no fresh mask
    // arrives this cycle, do NOT gradient-step or re-accept the belief (which would re-optimise the same
    // frozen anchor set and amplify noise into the accepted state/FE); just recompute FE at the unchanged
    // state for reporting. Set false to restore the legacy every-cycle re-fit (for A/B).
    bool  freeze_belief_on_stale       = true;
    // (B) Fisher-gradient clamp: observation_information builds the per-DOF Fisher via a central finite
    // difference of the NON-smooth min() box-SDF. A point sitting on a face-switch/corner makes the
    // difference a discontinuity JUMP, not a derivative, spiking |∂SDF/∂θ| by ~1000× → a single point
    // forces the Kalman gain K→1 and bypasses all accumulated stabilisation. For a well-behaved SDF
    // |∂SDF/∂θ| is O(1); clamp the per-point slope to this bound before squaring. 0 = disabled (for A/B).
    float fisher_grad_clamp            = 2.0f;
    // (C) Maturity stiffening: scale the per-frame Kalman acceptance gain by vh/(vh + accumulated_views),
    // so a well-observed DOF (many equivalent views) barely moves on any single frame — history dominates
    // a lone noisy measurement — while an unseen DOF (views≈0 → factor≈1) stays fully plastic. The
    // accumulated-views signal is the normalised Fisher accumulator (fisher_info[j]). false → pure
    // per-frame Kalman gain (for A/B); requires KalmanGainStiffness to have any effect.
    bool  fisher_maturity_stiffness    = true;
    float fisher_views_half            = 4.0f;    // equivalent-views at which the maturity stiffener halves the gain (lower = hardens faster)
    // (D) Counter-evidence gate (CUSUM / sequential change-detector): asymmetric, ratcheting trust on
    // top of the Kalman gain. Per DOF, the surprise of each fresh frame — innovation |raw−belief| in
    // units of the deadband — accumulates a SIGNED counter-evidence S_j only while it stays one-sided;
    // a lone glitch bumps S_j once and decays away (rejected, gain→0), while a CONTIGUOUS same-direction
    // run of surprises grows S_j until it overcomes a barrier that scales with the filter's accumulated
    // confidence (the better the prediction, the more coherent counter-evidence is needed to unlock it).
    // On unlock the DOF snaps to the new evidence (gain→1), S_j resets, and its accumulated precision is
    // deflated (the old estimate is conceded). Requires KalmanGainStiffness. false = disabled (A/B).
    bool  counter_evidence_gate        = false;
    float counter_evidence_band_m      = 0.02f;   // length-DOF surprise deadband (m): innovations within this are "coherent"
    float counter_evidence_band_rad    = 0.05f;   // yaw surprise deadband (rad)
    // Barrier = base + lambda·fisher_info[j], SPLIT into position (cx,cy) vs size/shape (w,h,H,leg,yaw,
    // inset) groups so the table CENTRE can be locked harder than its extents (position drifts one-sided
    // with robot localisation and was unlocking ~3× more often than w/h). Higher base/lambda → a longer,
    // more confident coherent run is needed before that group re-adapts.
    float counter_evidence_base_pos    = 6.0f;    // position barrier floor (excess-bands): centre is heavily locked
    float counter_evidence_lambda_pos  = 1.0f;    // extra position barrier per accumulated equivalent-view
    float counter_evidence_base_size   = 3.0f;    // size/shape barrier floor (excess-bands)
    float counter_evidence_lambda_size = 0.5f;    // extra size/shape barrier per accumulated equivalent-view
    float counter_evidence_decay       = 0.8f;    // per-coherent-frame pay-down of S_j (a coherent frame is evidence FOR the belief)
    float counter_evidence_unlock_deflate = 0.3f; // on unlock, multiply that DOF's accumulated info by this (concede the old estimate)
    float counter_evidence_step_cap    = 1.0f;    // cap the per-frame CUSUM increment (excess-bands) so ONE big glitch can't
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
    bool  mask_conf_weight = true;
    float mask_conf_floor  = 0.2f;   // YOLO score at/below which the mask is ~ignored in the update
    float mask_conf_ref    = 0.5f;   // YOLO score at/above which the mask is fully trusted (w=1)
    float mask_conf_power  = 2.0f;   // steepness of the trust ramp between floor and ref
    std::string fisher_csv_path        = "";      // if non-empty, append per-cycle Fisher-filter evolution to this CSV (for plotting)
    // Upload the table pose covariance onto the room→table RT edge (rt_covariance_att, 6×6 SE3),
    // built from the Fisher filter's per-DOF posterior precision (var = rt_cov_scale / fisher_info_raw):
    // x←cx, y←cy, z←H/2, yaw←ψ are data-driven; roll/pitch are unobservable (large). rt_cov_scale
    // calibrates the raw curvature toward NEES≈1 (raw precision over-counts spatially-correlated
    // points), like bottle_concept's cov_eff_scale. Written only when the geometry is (re)published.
    bool  rt_cov_upload = true;
    float rt_cov_scale  = 1.0f;
    float warm_lambda_pos_base         = 0.15f;
    float warm_lambda_pos_gain         = 0.45f;
    float warm_lambda_size_base        = 0.02f;
    float warm_lambda_size_gain        = 0.18f;
    float warm_lambda_yaw_base         = 0.01f;
    float warm_lambda_yaw_gain         = 0.12f;
    float warm_confidence_decay         = 0.70f;
    float warm_confidence_coverage_gain = 0.35f;
    float warm_confidence_residual_gain = 0.65f;
};

// Fill a TableConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
TableConfig load_table_config(const ConfigLoader& cfg);

}  // namespace rc
