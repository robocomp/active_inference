# The window problem in the motion-noise learner — memo (Fable, 2026-10-05, read-only review)

## Summary in plain words

1. The learner assumes the solver first predicts the pose, then nudges it a little towards the scan, and that the size of the nudge tells how noisy the odometry is. The solver does not work that way: each solve is essentially a fresh scan-only fix (the past enters only through a deliberately capped prior), so the "nudge" is really the full jump from the odometry guess to the scan's answer, and the posterior does not shrink from one solve to the next.
2. "Posterior wider than prediction" is therefore not a bug in the solver. It happens on position in 37-45 % of solves because two consecutive scan-only fixes differ by a few percent in confidence while one cycle of odometry noise is far smaller than that; it almost never happens on heading because one cycle of heading noise is as big as the scan's heading confidence. One genuine defect does leak in: the published covariance floor (`max(P, v²)`) seeds the learner's prediction, which inflated it up to 10x on heading in the RGB run.
3. The fix is to stop inferring the solver's gain from two covariances and instead score the scan's own innovation: the scan factor's gradient step `-H_sdf⁻¹ b_sdf` taken at the PREDICTED pose, every cycle, including early-exit cycles. Its variance is odometry noise over the cycle plus two scans' white noise, a linear model with no gain and no window in it. Three numbers (H_sdf, b_sdf, which pose they were taken at) have to leave the solver; both code paths already compute them and throw them away.
4. The slow 1° heading / 3-5 cm position wander against ground truth is frozen while the robot is parked and grows with distance driven, so it is a function of where the robot is, not of time. It cancels out of the cycle-to-cycle innovation (which is why the learner never sees it) and must be carried as a shared nuisance across the window plus a covariance term in the published pose, not as odometry noise.
5. A proxy fit of the proposed model on tonight's clean run gives sane coefficients without any collapse (k_long 1.4e-4 vs prior 9.4e-5, k_th_turn 6e-5 vs 1.9e-4, scan white 0.7 mm / 0.01°), whereas the old identity would have driven k_th_turn to 6e-6 on the midday run. A selftest and an offline replay are specified at the end.

---

## 0. What was verified, and where

Notation: P_pred = `sur_P_pred_`, P_post = `P_solver` (the solver's marginal before the floor), c = est − pred. All line numbers are from the uncommitted working tree.

**The two covariances the learner compares are produced by two different estimators.**

- P_post on a solve cycle is the newest pose's marginal of the GN window system, `last_marg_prec_` (room_concept.cpp:5673-5681 → room_gn_solver.cpp:1273-1326, Schur complement of everything else in the window), inverted after an eigenvalue floor and a ridge in `compute_posterior_covariance` (room_concept.cpp:6882-6895). This is the value `current_covariance` holds when `score_surprise` reads it as `P_solver` (7405).
- P_pred is a *recursive* quantity the window never consumes: it is seeded from the previous published covariance and then has one `selected_prior.covariance_eigen` added per cycle (3817-3820), scored and re-seeded at the next correction (7449: `sur_P_pred_ = sur_P_prev_ = P_post`).
- The window's *own* prior on the newest pose is something else entirely. With `SdfCurrentSlotOnly = true` (etc/config.toml:765) only the newest slot carries an SDF factor (room_gn_solver.cpp:1024-1025). Every older pose is held by motion factors only (1038-1046) and by the boundary prior on the oldest slot (1020-1022). That boundary prior is the FEJ Schur marginal of the dropped slot (room_concept.cpp:8874-8980) whose precision eigenvalues are clamped to `[1e-3, 500]` (8950-8953; `EigenvalueClampBoundaryMax = 500`, config:625, header default room_concept.h:353) and then multiplied by `boundary_weight_now()` = exp(u_b) (5414-5417, `HierPrecBoundaryEnabled = true`, config:1153). Measured on tonight's run: boundary_weight median 4.4, p10-p90 2.4-6.4 (etc/hier_prec.csv), so the past can never claim better than σ ≈ 1/√(500·4.4) ≈ 21 mm and 1.2° on the oldest slot, before the chain of 4 motion factors loosens it further.
- The newest scan's claimed precision, on the other hand, is σ ≈ 10 mm / 0.156° (median √pq, both runs). So the window's prior on the newest pose carries ≲ 20 % of the posterior's precision on position and ≲ 2 % on heading. The solver's effective gain on the scan is ≈ 1.

**The learner's gain is computed from the wrong pair.** `MotionNoiseVC::observe` (motion_noise_vc.h:100-160) forms Λ = P_post⁻¹ − P_pred⁻¹ and K = P_post Λ. On the live data that gives, on 1-cycle corrections, median K = 0.003 (fwd), 0.006 (lat), 0.34 (heading) — against an actual scan weight of ≈ 1. The regressors x_j = uᵀ K U_j Kᵀ u are then ≈ 0 on position (nothing can be learnt) and ≈ 0.12·U on heading, while the observed c is the full innovation. This is the mechanism behind "observed c² far outside the model variance".

**What the live posterior actually is, measured.** etc/hessian_check.csv (843 solves, 838 with 5 slots): median marg/pred = 1.00 (p90 1.00) on x and y, 0.81 on θ; block/marg = 0.22 on x — the block is dominated by the motion factor's Ω (slot_mcov_xx median 4.3e-6, σ 2 mm), not by the scan. The window marginal is numerically the same number from solve to solve on position.

## 1. Is P_post > P_pred a bug? — No; it is what a scan-only window looks like, plus one real leak

Measured on scored corrections (all 812 / 1785 scored cycles in both runs were GN solves, `iters > 0`; no polish cycle was scored — `sdf_polish_enabled` defaults false, room_concept.h:501, and is not set in the config):

| run | axis | P_post > P_pred | median P_post/P_pred | on APPEND cycles | on REPLACE cycles |
|---|---|---|---|---|---|
| 19-58-02 (RGB off) | fwd | 36.8 % | 0.997 | 0.0 % (median 0.78) | 44.7 % (median 0.999) |
| | lat | 15.0 % | 0.994 | 3.9 % (0.94) | 17.9 % (0.996) |
| | th | 0.2 % | 0.651 | 0.0 % (0.50) | 0.3 % (0.68) |
| 19-43-01 (RGB on) | fwd / lat / th | 31 / 33 / 5 % | 0.88 / 0.91 / 0.43 | | |

Solve-to-solve fluctuation of the marginal on 1-cycle corrections: pq(n)/pq(n−1) p10-p90 = 0.92-1.11 (fwd), 0.96-1.03 (lat), 0.79-1.12 (th). One cycle's modelled Q is 1.3e-6 m² on fwd (measured median pp(n) − pq(n−1) = 1.34e-6 with no floor, i.e. (1.16 mm)², consistent with k_long·0.02 m) against a posterior of (10.4 mm)² = 1.08e-4 m². A ±5 % wobble of the marginal is ±5e-6 m², four times the Q increment, so the sign of P_pred − P_post is a coin toss on position. On heading one cycle's Q is 4.25e-6 rad² = (0.118°)² against a posterior of (0.156°)² = 7.4e-6: pq/pp = 7.4/(7.4+4.25) = 0.64 — exactly the measured 0.65, and the wobble cannot flip it.

The append/replace split confirms the structure: on an APPEND the dropped slot's scan is freshly folded into the boundary prior (quality-weighted autograd Hessian, 8906-8927), at the 500×4.4 cap: (1/10.2² + 1/21²)⁻¹ gives a variance ratio 0.81 — the measured 0.78. On a REPLACE the boundary prior is untouched (`stride_span_.replace`, 3499; the replaced slot's own scan is simply overwritten) and the posterior is the previous one ±wobble. The window does not accumulate scan information between admissions; the learner's identity E[c cᵀ] = P_pred − P_post presumes it does.

Checked and ruled out as the cause:
- FEJ / re-linearisation: the marginal is computed at the converged poses from the same factor list the solve used (5666-5681); the boundary mean is frozen but that affects the mean, not the precision ordering.
- Early-exit stale covariance: with the polish OFF, `current_covariance` neither grows nor shrinks on early-exit cycles (growth gated at 4725-4730), but the learner's P_pred is its own accumulator (3820), unaffected. (The NIS column's `hess_pred_cov_` IS stale: `predict_step` adds only one cycle's Q to `prev_cov`, 7574-7652, however long the open stretch. Instrument-only.)
- Frames/slots/times: c and both covariances refer to the newest slot at the same cycle; the body-axis projection uses est_th for both (7399-7409). Fine.
- Boundary re-weighting between solves: corr(Δlog boundary_weight, Δlog pq) = −0.02 on 1-cycle corrections; not a driver tonight.

**The one genuine defect: the adaptive floor seeds the prediction.** `apply_adaptive_covariance` (7468-7490) sets `res.covariance(i,i) = max(P, EMA v², v²)` with v the innovation on that axis; it runs at 4157, before `score_surprise(res, true)` at 4393; `score_surprise` correctly reads the pre-floor posterior for pq (7405) but re-seeds `sur_P_pred_` from the post-floor `P_post = res.covariance` (7383, 7449). So P_pred ≥ c_prev² by construction whenever the floor bound. Measured: 7 % of scored corrections were floor-bound in 19-58-02, 55 % in 19-43-01; conditioned on the previous correction being floor-bound, pp(n)/pq(n−1) on heading had median 3.07 and p90 9.7 (vs 1.65 unfloored), and on fwd p90 5.55 (vs 1.19). That inflates o_a = P_pred − P_post, makes c² − o negative, and pushes k down. Fix: seed from `P_solver` (one-line change at 7449; the comment at 7403 defending the published value is about `prev_cov`/`predict_step`, which the window never reads while `CovarianceFromSolver = true`).

Secondary, measured, not a bug: the early-exit gate selects which cycles are scored. In 19-58-02, 386 of 782 1-cycle corrections were PARKED (|Δ| < 2 mm) — bursts of solves while the gate stays failed — and only 6 corrections had stretches ≥ 6 cycles. The long stretches are the ones where the prediction stayed inside the gate, so their c is small by selection (sum c²/Σ(pp−pq)⁺ = 0.11-0.29 on heading for stretches 2-100 vs 0.34-0.57 at 1 cycle). The header's "biases ρ up" is one half; "biases k down on long stretches" is the other.

## 2. The right observation model: the scan's own innovation at the prediction, every cycle

### Recommendation: option (b), built from (a)'s quantities

Define, each cycle n, at the predicted pose x_pred (the odometry-propagated previous estimate, `pred_x/pred_y/pred_theta`):

- (H_s, b_s): the newest slot's SDF factor linearised at x_pred — exactly the loop at room_gn_solver.cpp:66-104 (`SdfFactor::linearize`) and, verbatim, the polish's loop at room_concept.cpp:4940-4972. The early-exit path already evaluates the SDF and its gradients at x_pred (`mean_sdf_pred`, 4752); the 3x3 accumulation is the missing ~400-point loop.
- Innovation ν_n = −H_s⁻¹ b_s: the scan-alone Gauss-Newton step from the prediction, i.e. z_n − x_pred with z_n the scan-only pose, valid while x_pred is inside the SDF's linear range (early-exit cycles by definition; on solve cycles use the solver's final (H_s, b_s) at x_n and ν = (x_n − x_pred) − H_s⁻¹ b_s(x_n)).

Observation model, per body axis a, with x̂_{n−1} ≈ z_{n−1} (gain ≈ 1, section 0):

    ν_n = e_n − e_{n−1} + η_n          e = scan white error, η = odometry error over the cycle
    E[ν_a²] = Σ_j k_j · u_aᵀ U_j,n u_a + r_a,n + r_a,n−1 + ρ_a + ρ'_a            (★)

with U_j,n the unit components of this cycle's interval (`selected_prior.preint.unit[j]`, already summed into `sur_U_` at 3839), r_a = uᵀ H_s⁻¹ u the claimed scan variance, and ρ the per-axis excess (may be negative — it will be: see below). (★) is linear in (k, ρ) with a known offset, has no gain, no P_post and no window in it, and reduces to the learner's existing IRLS/REML normal equations with x = (uᵀU_j u, 2) and y = ν_a² − (r_n + r_{n−1}).

Why unbiased on a clean (consistent) case: if odometry errors are white per cycle with covariance Σk_jU_j and scan white errors are independent with covariance R, then (★) is the exact second moment of ν — no approximation of the solver's gain, no assumption about how the window combines the past. Ordinary least squares on squared residuals with correct expectation is consistent; the IRLS weights 1/(2m²) only change efficiency. The one structural correlation is MA(1): ν_n and ν_{n+1} share e_n (correlation −r/(2r+Q) ≈ −½ when the scan white dominates). That biases nothing in the mean; it over-counts information by ≤ 2x. Either feed every second cycle, or whiten pairs — I would start by feeding every cycle and halving the prior-relative weight.

Why it also fixes the gate selection: ν exists on every cycle, early-exit or not, so the learner sees the whole trajectory, parked stretches (which pin ρ and the per-second terms: Q ≈ 0 under the ZUPT) and driving stretches (per-metre) and pivots (per-radian) alike. The early-exit decision becomes irrelevant to the learner.

**What the solver must expose and where**

1. `gn::Result` (room_gn_solver.h:121-131): add `Eigen::Matrix3f H_sdf; Eigen::Vector3f b_sdf; bool sdf_ok;` filled by linearising the newest `SdfFactor` alone into a scratch `LinearSystem` after the last accepted step (the pattern at room_gn_solver.cpp:866-868 / 880-882 with `sys_gate` is exactly this). Cost: one extra factor linearisation on solve cycles (~4 % of cycles).
2. `newest_pose_marginal` (1273-1326): return also `prior = marginal − H_sdf` (PSD by construction: it is the rest-of-window's marginal information on the newest pose). This is the window's real P_pred⁻¹ — the number the current learner was trying to infer from P_post. With it, K_win = P_post · H_sdf is exact, and the learner's existing `observe()` would be *correct* on solve cycles if fed (P_pred = prior⁻¹, P_post = marginal⁻¹). I would still not use the correction path: it inherits the gate selection and the window-prior cap.
3. `try_prediction_early_exit` (4669-5030): hoist the H/bb accumulation at 4940-4972 out of the `sdf_polish_enabled` branch into a measure-only step (the function already has a `measure_only` flag, 4674/4780), store (H, bb) and `last_pred_*` into the `UpdateResult`, and let `score_surprise` consume ν = −H⁻¹bb every cycle.
4. `score_surprise` (7379-7450): replace the (c, P_pred, P_post) call with (ν, U_n, r_n, r_{n−1}); keep `sur_U_`/`sur_QK_` accumulation but reset every cycle (stretch = 1); drop the Λ-projection block in `observe()` (motion_noise_vc.h:118-128) entirely.
5. Log per cycle: ν (body axes), r_n, the three regressor sums, so the estimator can be replayed offline from the CSV (today it cannot: U_j and H_s are not logged, which is what limited section 4's replay to a proxy).

**Two caveats to design in, not discover later**

- *The claimed scan variance is not the white variance.* `SdfFactor::linearize` uses a = ½·w·u/(n·σ_obs²) (room_gn_solver.cpp:89) — the mean loss, so H_s is the information of ONE point at σ_obs = 0.05 m times the geometry, regardless of n (that 1/n is the crude common-mode cap). The measured scan-to-scan repeatability is 0.7-1.1 mm / 0.01-0.05° per scan (from 1-cycle corrections: rms c_fwd 1.6 mm of which ~1.4 mm is one cycle's modelled odometry), against claimed 10 mm / 0.156°. So ρ in (★) will come out ≈ −r: let it, it is a variance of the model's making and may go negative; but then the honest published P_post needs the slow term of section 3 or it will be 50x overconfident against the world.
- *A scale (systematic) odometry error is not a random walk.* The 1-cycle heading innovation is 0.065° rms against a modelled per-cycle 0.118° (k_th_turn = 190e-6 was fitted on long GT windows, se2_preintegration.h:129-134), and `NoiseModel` carries `scale_omega = 0.155` as a perfectly correlated fraction. Over N cycles a scale error grows as N² in variance, a random walk as N; a per-cycle regression on (★) learns only the random-walk part and will read k_th_turn ≈ 1/3 of the long-window value. Add the regressor U_scale = ΔΔᵀ (the stretch's own increment, outer product) so the two are separable — they are, because they scale differently with stretch length — rather than letting the learner shrink k to fit the short scale.

### The alternatives, weighed

- (a) alone (expose H_sdf, keep scoring the correction c with an exact K): correct on solve cycles, but still gate-selected (4 % of cycles, half of them parked bursts) and still built on a window prior that is capped at 500·bw — the Q signal over a 0.5 m window (√(94e-6·0.5) = 7 mm) is 11 % of the cap's 21 mm in σ, 1 % in variance. Expose the quantities (they cost nothing) but do not learn from c.
- (c) REML on the motion-factor residuals inside the window (redundancy numbers r_i = 1 − tr(H_iΣ)): with scans only on the newest slot the chain is tied at its two ends only; interior motion residuals are ≈ 0 with redundancy ≈ 0, and the whole chain behaves as one factor between a capped prior and one scan — the information is the same as (a)'s, at the cost of the full H⁻¹. Worth doing only if `SdfCurrentSlotOnly` is turned off, and then the correlated-scan problem of section 3 bites inside the window.
- (d) Innovation against the previous *window marginal* instead of the previous scan: E[ν νᵀ] = P_post,n−1 + Q + R_n is the textbook adaptive-KF form, but P_post,n−1 is the claimed 10 mm, 10x the actual white error of x̂_{n−1} — it would absorb Q into a wrong offset. (★) avoids it by using only quantities with measured meaning.

## 3. The slow bias: a function of pose, not of time; a shared nuisance, not odometry noise

Measured against GT (fresh supervisor rows; nees_report.py plus a structure-function script):

- Parked, the estimate's GT error does not move: rms Δe over 11 s parked = 0.05° / 0.2 mm (19-43-01, n = 166 pairs), 0.012° / 0.2 mm over 1.1 s (19-58-02). The midday run's 750-cycle parked stretch moved the scan-only heading by 0.07° in 38 s.
- Moving, it grows with distance: rms Δe_x = 5.6 → 9.4 → 14.4 → 27 → 53 mm over 0.04 → 0.21 → 0.43 → 1.4 → 4.5 m (19-58-02), exponent ≈ 0.57 in distance, saturating near the run's sd (49 mm / 1.1°). Heading: 0.48 → 1.29° over the same spans (the 0.46° at the first lag is the supervisor's timing jitter during turns, white; it is absent when parked).
- It is smooth at the cycle scale: the 1-cycle innovation (1.6 mm / 0.065°) is fully accounted for by one cycle's odometry noise plus ~1 mm of scan white; the bias increment over 2 cm is below 1 mm / 0.02°.

So it is a nuisance FIELD over pose, b(x), with a correlation length of order metres, frozen in time. Physical source not identified here (candidates: SDF/polygon systematic by viewpoint, a sensor mount offset rotating with heading, map scale/rotation relative to GT); the source does not change the modelling conclusion:

- In the window: a shared 3-vector nuisance b per window (all slots are within ≪ ℓ of each other), prior N(0, σ_b²), marginalised — the Woodbury common-mode the project already prescribes for correlated mask points. Its effect: the newest pose's marginal becomes R_w + σ_b² and stops collapsing with the number of scans (the measured "pose cov collapses when parked" is this defect). Between windows b evolves per METRE (Gauss-Markov over path length), with σ_b² and ℓ learnable from parked-vs-moving innovations exactly as above — parked stretches say Δb = 0, so anything per-second in the innovation is odometry; moving stretches at long baselines say var(Δb) ≈ 2σ_b²(1 − e^{−d/ℓ}).
- In the published posterior: a covariance term, J_map Σ_map J_mapᵀ-like, that does not shrink with scans. With it the heading NEES (7.3x) and fwd (2.5x) are addressable; without it no amount of Q learning will fix them, because Q is not where the error is.
- In the learner (★): nothing, at 1-cycle baselines — the bias cancels in e_n − e_{n−1} to below 1 mm / 0.02°. If the learner is ever fed longer stretches, add the regressor 2σ_b²·(1 − e^{−d/ℓ}); note its per-metre small-d limit 2σ_b²d/ℓ is collinear with k_long/k_lat and separable only by saturation, which is one more reason to score every cycle.

Should it be a STATE (OU) or a COVARIANCE term? Both, in different places: a state within the window (it is shared and must not be averaged away), a covariance term for consumers; an OU *in time* is the wrong clock — the parked data refute it by 20x.

## 4. Tests

**Selftest (extend tools/motion_noise_vc_selftest.cpp, part 3):** simulate the window the way the solver actually builds it, not a Kalman update. Per cycle: true pose propagated by the Integrator with k_true; scan-only pose z_n = x_true + e_n, e_n ~ N(0, R_w) with R_w = (1 mm)²/(0.02°)², PLUS a smooth bias b(x) (e.g. a sum of a few sinusoids in x, y with amplitude 40 mm / 1°); claimed H_s from the SdfFactor formula with σ_obs = 0.05 (so r = 10x the white); prediction = z_{n−1} + odom; ν_n = z_n − pred. Feed (★) with U_j from the same Integrator. Pass criteria: k_long, k_th_turn, k_lat_turn within 15 % after 20k cycles with mixed parked/drive/pivot; ρ ≈ R_w − r (negative) within 30 %; the per-second terms stay at their true (zero) value within the prior's sd under the bias field — this is the test that the bias is not attributed to Q. Then a second pass with a scale term (scale_omega = 0.155) to show the U_scale regressor absorbs it and k_th_turn does not drop.

**Offline replay on existing CSVs — what can be done today, and what it showed.** ν is not logged, so the replay used c on 1-cycle corrections (K ≈ 1 there, section 0) and the odometry increment pred − est_prev as the regressor, i.e. a proxy of (★) without U_j and without H_s. IRLS with the learner's own weights, priors at the configured k with sd = k:

| run | k_long (prior 94e-6) | k_lat / k_lat_turn (6e-6 / 10e-6) | k_th_turn (190e-6) | scan white (fwd / th) |
|---|---|---|---|---|
| 19-58-02, 782 corr. | 1.4e-4 (1.4e-4 with 100x weaker priors) | 1.7e-5 / 1.1e-5 | 6.1e-5 (6.2e-5 weak priors) | 0.7 mm / 0.011° |
| 12-07-20, 340 corr. | 4.7e-4 | 4.7e-6 / 1.6e-5 | 4.8e-5 | 1.7 mm / 0.034° |
| 19-43-01, 1731 corr. (run has a 433 mm divergence episode) | 9.0e-4 (8e-3 weak) | 1.3e-5 / 1.2e-5 | 2.9e-5 | 6.2 mm / 0.10° |

No collapse to zero on any run; the clean run's coefficients are within a factor 1.5-3 of the GT-fitted priors, with k_th_turn low for the scale-vs-random-walk reason above. The old identity on the same 12-07-20 corrections gives Σc²/Σ(pp−pq)⁺ = 0.03, i.e. it would scale k_th_turn toward 6e-6 — the live collapse reproduced offline. The 19-43-01 numbers say the learner must not train through a divergence/relocalisation episode (`search_active` / `lost_logodds` are in the localizer CSV; gate the TRAINING on the belief state, which is a model quantity, not a threshold on the data).

**Live acceptance, once (H_s, b_s, ν) are logged:** (i) median ν_a²/m_a per axis in [0.3, 0.7] (χ²₁ median 0.455) on every cycle class (parked / driving / pivot / early-exit / solve) — the per-class split is the check that the gate no longer selects; (ii) ρ stable across runs; (iii) `nees_report.py` heading err/σ moving toward 1 only after the section-3 covariance term is published — it must NOT move from Q learning alone, and if it does, something is wrong.

## Verified vs conjectured

Verified in code and logs: the factor layout and clamps (section 0); the append/replace split and the 0.78/0.999 ratios; the heading 0.65 ratio arithmetic; the floor leak and its magnitude; the gate selection counts; the parked-vs-moving structure of the GT error; the learner's K medians; the proxy fits.
Conjecture: the physical source of b(x); that the heading 1-cycle shortfall is a scale term rather than an overstated k_th_turn (both fit; U_scale resolves it); the MA(1) efficiency loss size; that the 19-43-01 divergence was the RGB factor (not examined).
