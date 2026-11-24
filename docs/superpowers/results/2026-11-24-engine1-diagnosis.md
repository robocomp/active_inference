# Engine1 live regressions: root-cause diagnosis

Date: 2026-11-24. This is diagnosis only. No file under the worktree was edited.
Build under test: `feat/shared-engine` at bada10a3 (run `objects_engine1`). Baseline: main 81ab85d1, which is the countable code at b36db5d9 (run `objects_step0`).
Paths below are relative to `/home/pbustos/robocomp/components/active_inference.worktrees/shared-engine`, and the line numbers are at bada10a3.

Scratch files, all in `/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/rootcause/`:
- `omega_replay.py` replays the volatility recursion from the logged steps (item 1).
- `nees_rows.py` is `tools/step0/nees_persistent.py` with the truth point and estimate written per row. It writes into the copies under `runs/`, not into the run directories.
- `s1.py` summarises sigma over gated rows.

## Summary

| # | Symptom | Root cause | Introduced by an engine commit? |
|---|---|---|---|
| 1 | Bed sigma grows to the caps on gated rows; error 13.8 cm | Inferred volatility (Q = exp ω) learned from two 0.3–0.4 m seed-correction steps. The steps come from a birth that skipped "birth on priors" because the birth mask was truncated by 0.23 %. The final error comes from three LiDAR-only updates while sigma was inflated. | **No.** The mechanisms pre-date the engine. The offer path is ruled out. |
| 2 | Toilet NEES/DOF 87, sigma_cx 1.4 cm | The θ-only wall no-cross hinge is re-entered with fixed precision on every update and never saturates. The back sits 2–10 mm "past" the wall on 131 of 131 rows, so 128 parked updates pin the back to mm. The 19 cm bias is the flush prior: the toilet stands 18 cm off the wall in this world. | Mostly no (bad3c092, 11-16). bd0b07a8 (constant sig2_cam) halves sigma_h and so tightens sigma_cx further. |
| 3 | Bed publish stage 25 ms per tick | The offer runs **every compute tick for every instance**: 4 `predicted_gain` calls, 8 when the best standpoint is blocked. It also runs during the cooldown, which only zeroes the published gain afterwards. | Yes: b125d941 / bc64f0e5 / 853460a7 / bada10a3 (G5) put the engine's offer in the per-tick path. |
| 4 | `nbv_gain_raw` is 0 on every row | `raw_gain` is taken from `prop.epistemic_gain`, which is already multiplied by `useful(...) ≡ 0` ("no consumer, no bid"). | No. step0 is also all zeros (74837fbd / 80442a3e). |
| 5 | Phantom bed_1 in room_1 | YOLOE labelled the same unexplained region in room_1 as "bed" in both runs. In engine1 the detection fell 0.47 m from the residual surprise region, so fused birth admitted it. The tracker and birth code are untouched by the engine diff. | **No.** |

---

## 1. Bed sigma grows to the caps; bed error 13.8 cm

### Hypothesis A (offer path mutates P): rejected
- `PersistentEngine::predicted_gain` (`common/ai_belief/persistent_engine.h:857`) is `const`. It copies `P_, b_, sig_prev_, lever_, last_sigma_pose_` (lines 862–866) and predicts and updates only those copies (`predict_theta_on(P, …)` at 873, `predict_latents_on(m, fl, P, b, sp, lev, lsp)` at 921). The only member it writes is the `mutable offer_cache_` (parts, Ω_last, offer frame), and only the offer reads that cache.
- The logs leave no room for an extra term. On pure predict rows (gated, `lidar_only=0`, `npts=0`), the per-row increase of each `std_*²` equals the replayed `exp(ω_i)` on all six DOFs (`omega_replay.py`):

| cycle | Δvar cx | q cx | Δvar cy | q cy | Δvar w | q w | ω_w logged | ω_w replayed |
|---|---|---|---|---|---|---|---|---|
| 47 | 5.94e-6 | 6.29e-6 | 8.99e-4 | 9.53e-4 | 3.40e-6 | 3.61e-6 | −12.53 | −12.53 |
| 201 | 1.16e-5 | 1.17e-5 | 1.80e-3 | 1.81e-3 | 7.53e-4 | 7.57e-4 | −7.19 | −7.19 |
| 541 | 4.17e-6 | 4.15e-6 | 4.76e-4 | 4.73e-4 | 2.05e-4 | 2.03e-4 | −8.50 | −8.50 |

A Δvar of 0 means the DOF sits at its prior cap. The replayed ω_w matches the logged one exactly at every change point (−12.53, −7.19, −8.50), so the replay counts the same rows the agent counts.

### What does grow sigma: the inferred volatility, fed by huge seed-correction steps
- Gated cycle: `countable_fitter.cpp:1538` calls `predict()`, then `countable_belief.h:567-573` calls `predict_theta_joint(1)` (`countable_belief.cpp:857`), then `persistent_engine.h:1043 predict_theta_on`. That adds `process_noise_diag()` (`countable_belief.cpp:214`), which is `exp(ω)` once volatility is inferred (`AI2VolatilityInfer = true`), and clamps to `prior_cov_diag()`. The caps are σ_pos 0.30 and σ_footprint 0.20; the user saw exactly those values.
- ω is learned in `update_volatility` (`countable_belief.cpp:248-291`, called at 978) from the post-minus-previous-posterior step of camera updates: `q = (n0·q0 + S)/(n0 + n)` with n0 = 1/σ_ω² = 0.25 and q0 = 0.005² = 2.5e-5.
- The bed_2 camera-update steps that dominate S:

| cycle | Δcy | Δw | Δh |
|---|---|---|---|
| 150 | +0.210 | +0.145 | −0.384 |
| 9 | +0.074 | +0.002 | +0.317 |
| 3 | +0.035 | +0.004 | +0.066 |

Step0's bed has at most |Δcy| 0.008, |Δh| 0.016 and |Δw| 0.075 over 84 steps. Those two steps alone set q_h ≈ 1.5e-2 per cycle (about 600× q0) and q_cy ≈ 1e-3 to 2e-3 (40–70× q0). The 783 predict-only rows then reach the caps within a few dozen 100 ms cycles.

### Why the steps are huge: the birth skipped "birth on priors"
- `objects_engine1/logs/bed_concept.log:791`: `[bed_2] cold-start state: cx=-0.34 cy=2.27 H=0.91 w=1.60 h=1.90 yaw=0.00`. No `cold-start footprint` / `birth on priors` line follows. bed_1 and step0's bed_19 both print it: step0 line 725 places the seed on the wall at (−0.33, 3.11), and the truth is cy 3.073.
- The guard is `countable_fitter.cpp:1139`: `if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)`. The birth row's `trunc_frac` is **0.002307**. A 0.23 % truncated mask skipped the whole seed, so the box was born at the cloud centroid, 0.80 m short of the true centre along the bed's depth. This guard is pre-existing (74837fbd, 11-14). The engine1 tour reached the bed ~0.6 m closer than step0 (lever_y 1.01 against 1.92), which is why the mask clipped.
- First update from that seed: θ_cy 2.787. The pose-chain latent absorbs `e_y = −0.226` m, which is 4.4σ of its 5 cm prior (step0: −0.010). GN does not converge (`gn_conv=0`, 25 iterations, cycles 3–9). The belief then corrects in the two jumps above.

### Where the 13.8 cm error comes from
Per-row truth from `nees_rows.py` gives tx −0.307, ty 3.073:
- Cycles 2–149: cy 2.75–2.89, error 20–33 cm. This is the bad seed.
- Cycle 150 (camera): cy 3.08, error 7 cm.
- Cycles 497–499 are three **LiDAR-only** updates at 0.95 m range, with σ_cy inflated to 0.088 by the volatility. They move cy 3.087 → 2.944 and h 2.035 → 2.080. No camera update follows to the end of the run, so the final error is 13.8 cm (rows from 1795497543650 on).
- The wide sigma is what lets three LiDAR frames move the mean 14 cm.

### Commit
None of the engine commits introduces it:
- The volatility inference is the HISTORY stage (08-03; omega_* is logged in step0 too).
- The truncation guard is from 74837fbd.
- The run-to-run difference is the birth view.

Not settled: whether the pre-engine code, given the *same* seed and frames, would also have split the first update into a 4.4σ chain latent. I could not replay it, because the run has no frame or mask dump, and a live tour cannot be bisected by rebuilding old commits without replay input. `feat/mask-dump` is merged in main 81ab85d1: turn it on in the next tour, then replay the bed_2 birth frames through b36db5d9 and bada10a3.

### Proposed minimal fix
Two changes, both model-level rather than gates:
1. **Volatility evidence.** In `update_volatility`, the squared step a *static* object is expected to make is the Kalman correction itself, Σ_pred,ii − Σ_post,ii, plus Q. Feed `max(0, δθ_i² − (Σ_pred,ii − Σ_post,ii))` into S, the method-of-moments excess. A seed collapsing onto data then reads as a large correction of a large prior, not as object motion. The pre-update Σ is already available in `update_persistent_cycle` (`countable_belief.cpp:955`).
2. **Birth.** Replace the `last_trunc_frac <= 1e-3f` switch at `countable_fitter.cpp:1139` with continuous handling. Always birth on the priors when a wall is known, and let truncation widen the birth position covariance (the `birth_pos_cov` excess), instead of discarding the seed.

---

## 2. Toilet over-confident (NEES/DOF 87, error 19 cm, σ_cx 1.4 cm)

### Evidence
- The toilet **is** LiDAR-measured. All 128 ungated rows have `ray_n = beam_staged = 179` and `Itr_L` ≈ 1.4e5. All 128 are fresh frames (128 distinct `pkt_fid`). The robot is parked at range 2.40 m (`motion_dotd` ≈ 0.01), so nothing gates them. step0 had 9 updates in 2.5 s; 63 of its 72 rows were gated because the robot was moving.
- Latents are persistent and are not refreshed: ρ_cam, ρ_L and ρ_chain are ≥ 0.993 on every row. They are floors, as designed.
- The shape latent is not the cause. Δ_H = 0 (manifest), and w/h carry Δ 0.25 / 0.12 (σ 7.2 / 3.5 cm), which can only *widen*.
- The neighbour density is not the cause: no neighbour is in the bathroom.
- **The no-cross hinge is active on 131 of 131 rows.** The birth puts the back exactly on the wall (`toilet_concept.log:926`: back = −2.36 − 0.365 = −2.725). The gap (back − wall)·n is negative on every row:

| cycle | gap (mm) | σ_cx | σ_h | √(σ_cx² − σ_h²/4) = implied σ_back |
|---|---|---|---|---|
| 2 | −10.5 | 0.0214 | 0.0346 | 12.6 mm |
| 10 | −2.9 | 0.0154 | 0.0289 | 5.4 mm |
| 66 | −2.4 | 0.0142 | 0.0275 | 3.4 mm |
| 130 | −2.1 | 0.0138 | 0.0269 | 3.0 mm |

`accumulate_no_cross(..., fixed_precision=true)` (`countable_belief.cpp:650-683`) is called from the θ-only `accumulate_extra` on **every** update (`countable_belief.cpp:781`). It adds λ = 1/σ_wall² = 2500 m⁻² of unsaturated information each time, while the wall-flush prior itself enters once (lines 776, 970). The back is therefore pinned to the mm, and σ_cx² ≈ σ_back² + σ_h²/4 leaves σ_cx ≈ σ_h/2 = 1.4 cm.
- step0 had the same hinge (gap −1 to −3 mm on its updates) but only 9 updates. It also had σ_h 5.5 cm, because its camera size common-mode still grew with range: 0.02² + (0.08·3.9 m)², so `sbcam_H` was 0.159 against 0.0198 now. σ_cx therefore stayed at 2.6–4.4 cm.
- **Bias.** The truth is cx −2.193, so the true back is at −2.558, 17 cm in front of the wall at −2.725. The manifest says so ("0.18 m off it in the world, hence reach 0.25"). The flush prior and birth on priors put the back on the wall, which gives cx = wall + h/2. engine1's h = 0.697 (truth 0.73) gives cx −2.379, error 19 cm. step0's h = 0.86, too deep, happened to give cx −2.28, error 13 cm. The mean bias is the flush prior against this world, in both builds.

### Commit
- Over-confidence: bad3c092 (11-16, the fixed-precision θ-only one-sided constraints), exposed by 128 parked updates.
- bd0b07a8 (sig2_cam = constant common-mode stds) tightened σ_h from about 5.5 to 2.7 cm, which is half of the remaining σ_cx.
- 5d07b226 (shape latent), the neighbour density (f1197dcf / 898ff59e) and the manifest change (8e7cc58b) play no part.

### Proposed minimal fix
The wall's position error is one persistent quantity, the same wall with the same σ_wall every frame, so its information must saturate at 1/σ_wall² in total. Either:
- (a) give the no-cross hinge the same once-per-anchor bookkeeping as the flush prior, keeping its gradient (mean push) every update but its Hessian only on first entry; or
- (b) more principled: route it through a ρ = 1 wall latent with marginal σ_wall², as the sensor latents are.

Separately, the 17 cm off-wall toilet is a prior/world mismatch. The flush prior's reach should let the data move the back off the wall, and with depth unobserved from the front it cannot. That is a modelling question for the owner, not a bug.

---

## 3. Bed publish stage 25.5 ms per tick

### Call path, every compute tick, every instance
1. `specificworker.cpp:772` `publish_countable_cycle` (timer `t_pub` at 783).
2. `:790` `publish_countable_intentions`.
3. `:832` `step_epistemic` (unconditional; the comment at 828 says so).
4. `common/epistemic_step/epistemic_step.h:124` `hooks.compute()`.
5. `specificworker.cpp:932` `EpistemicPlanner::compute`.
6. `epistemic_planner.cpp:152-153` `raw_gain_from_standpoints`, which calls `belief.predicted_gain`.
7. `common/nbv/countable_planner.h:121` (`plan_faces`), which calls `viewpoint_score.h:806` once per face: **4 calls**. If the best face's p_stand is 0, `countable_planner.h:131-137` re-plans: **8 calls**.

The cooldown suppression is applied *after* compute (`epistemic_step.h:136`). bed_2 published gain 0 on 97 % of rows, and the full offer was still computed on every one of those ticks.

### Cost
- Each `predicted_gain` runs `visible_surface` over every part (8×8 grid per face), `synth_rays` over the select cylinder's angular span × every Helios ring (`offer_view.h:234-264`, with a part/room/obstacle intersection per ray), `accumulate_beam_rays`, a 45×45 `joint_cov`, and two LDLTs.
- `offer_cache_` (`persistent_engine.h:1018-1030`) caches only the parts, Ω_last and the context frame per state. The result is not cached.
- The bed's span is wider (larger `select_box().rxy` and closer standpoints), so it synthesises more rays. That is consistent with about 6 ms per call for the bed against about 0.8 ms per call for a nightstand (3.3 ms per instance-tick ÷ 4).
- In the timing log, publish is about the same in the bed_1 window (median 21.6 ms) and the bed_2 window (24.2 ms). The 4-call pass dominates; the 8-call re-plan is not the main cost.

### Not settled
The per-call split between rays, points and `joint_cov` is not measured (no profiler run; another agent was compiling in the worktree). The 66–78 ms publish spikes with infer = 0 are probably ticks with the double plan plus an Ω_last recompute; that is not confirmed.

### Proposed minimal fix
Compute the offer once per belief change, not once per tick:
- Cache the last `EpistemicProposal` keyed on the engine's `gn_update_serial()`, which bumps on every update.
- On predict-only ticks, re-publish the cached proposal. Its promise changes only through Q.
- Skip `compute()` when `inst.epistemic_cooldown > 0`, and hold the offer instead.

This turns the bed's 783 gated ticks into zero offer evaluations.

---

## 4. `nbv_gain_raw` is 0 on every row

- `epistemic_step.h:134`: `const float raw_gain = prop.epistemic_gain;`.
- That is the planner's *final* gain, `countable_planner.h:156`: `face_gains[best] * useful(…) * p_stand`.
- `epistemic_planner.cpp:185` sets `const auto useful = [](float) { return 0.0f; };` (the "no consumer, no epistemic bid" ruling). So the "raw" column is 0 by construction. The 5 in `nbv_gain_pub` is the verification floor (`existence_verify_gain`) times p_stand, which is also the controller's `gain=5`.
- The true raw value is `proposal.promised_nats = best_raw` (`epistemic_planner.cpp:250`, written to `offers.csv`).
- step0 is also 0 on every row for all classes (checked), so this is not an engine regression. It has been in place since the countable `useful ≡ 0` (74837fbd) under the shared step (80442a3e).

Proposed fix: in `step_epistemic`'s `record` hook (`specificworker.cpp:951-959`), log `published.promised_nats` as `dbg_nbv_gain_raw`. Alternatively, have `epistemic::step` take the raw value from a `promised_nats` field when the proposal has one (`if constexpr (requires { prop.promised_nats; })`).

---

## 5. Phantom bed_1 in room_1

- The birth came from a YOLOE "bed" detection at room (2.12, −0.67), fused with the residual surprise (`bed_concept.log:584-591`: `mass_r05=110.6`, `dist=0.47` to the unexplained 0.90×1.85 m region near (2.36, −0.22) that the probe had reported for 20 s).
- step0 saw the same unexplained region (`bed_concept.log` around 1795479640: top (2.35, −0.22), ext 0.90×1.85) and a "bed" detection in the same room at (2.59, −1.05) (line 573). It did not birth: that detection sat about 0.85 m from the mass.
- `git diff b36db5d9 bada10a3` touches no tracker, birth or fusion file. The specificworker diff only adds `read_occupied_cells()` and `refresh_neighbours()` *after* the tracker.
- The shape latent and neighbour density act inside the fit after birth, so they cannot widen birth acceptance.

Root cause: a detector false positive on an unmodelled object in room_1, admitted by the experimental fused birth (`Tracker.BirthFusion=true`, `BirthAdmitPlausibility=0`) because of where the false mask landed. This is view-dependent and unrelated to the engine.

Not checked: which world object it is. A fix is out of scope here. A nonzero `BirthAdmitPlausibility` (the class shape score already exists) is the model-level lever.

---

## What I could not settle
1. Item 1: whether the engine changed how the first update splits a bad seed between θ and the chain latent. This needs a mask dump and replay; no dump exists for this run.
2. Item 3: the per-call cost breakdown and the cause of the 70 ms spikes. Not profiled.
3. Item 2: the exact share of σ_cx due to bd0b07a8 against the hinge. The σ_back / σ_h decomposition above assumes they are uncorrelated.
4. No commit bisection was built. Items 1, 2, 4 and 5 trace to pre-engine code by reading the code and the logs. A rebuild bisection cannot reproduce a live tour without replay input.
