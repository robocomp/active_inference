# chair beliefs stuck at std 0.30 m / price 3.19 nats — diagnosis (logs + code, no runs)

Runs: `estimation_scripts/crossing/<run>/chair_ai2_log.csv` (header: cycle,node,npts,gated,...,fixated,centroid_r,ego_lin,ego_ang,...,nbv_gain_raw,nbv_gain_pub,...,t_ms) and `logs/chair_concept.log`.
Code: `chair_concept/src/chair_fitter.cpp` (`run_inference`, `fixated`), `common/ai_belief/recursive_laplace.h` (`predict`), `chair_concept/src/specificworker.cpp` (`step_epistemic`).
s1 runs use the worktree `active_inference.worktrees/existence-info`; its chair src differs from main only in chair_instance.h, epistemic_planner.{h,cpp} (existence term), specificworker.cpp. `chair_fitter.cpp` is identical, so the fixation gate is the same.

## Short answer

The chairs are associated and fed every cycle, but the geometry update is INHIBITED by the FIXATION gate, and the one condition that never passes for them is CENTRED (`centroid_r <= 0.60`). Σ is never touched, so std stays at the prior (0.30), and the planner prices Σ, so the offer stays at 3.19 nats.

## Q1. Associated? Used?

- Associated: every row of every stuck chair has `npts>0` (e.g. chair_147: 440/440 rows, npts 1.5k–23k; trunc_frac 0, clutter_frac 0). The log is written by `log_ai2_csv` once per processed cycle, so the instance is being fed.
- The `gated` column is useless as a discriminator: it is 1 on ~95% of ALL rows, shrinking chairs included (e.g. chair_148 gated1 205/244). Its definition is `trunc_gated or confirm_only or not fix_ok` (chair_fitter.cpp:512-513).
- The path: chair_fitter.cpp:512 `fix_ok = not fixation_enabled or fixated(inst,npts)`; line 513 `gated = ...`; line 531 `if (gated) inst.ai2_belief.predict(); else {... ai2_belief.update(frame)}`. So a not-fixated frame never reaches `rc::ai::update`. There is no other early return between association and update except the stale-data return at lines 406-415 (`not has_fresh_data`, age only).
- `fixated()` (chair_fitter.cpp:818-839) = `enough_pts(>=150) and clean(clutter<=0.35)` AND `centred(centroid_r<=0.60)` AND `still(ego_lin<=0.05, ego_ang<=0.10, |motion_dotd|<=0.05)`. Defaults from `chair_config.cpp:145-163`; `FixationCentrePrecision=false` so centring is a hard gate.
- The "[fixation] SUPPRESSED" reason line (chair_fitter.cpp:515) never appears: `grep -c SUPPRESSED logs/chair_concept.log` = 0 in c3b5_on_r2 (it is printed only when `should_log`, i.e. every `chair_log_period_frames`). The only gate state in the log is the `fixated` column plus `centroid_r, ego_lin, ego_ang`. `motion_dotd` is NOT logged (only `motion_var`); unknown from the CSV.
- Which condition fails (computed from the CSVs): rows with npts>=150, clutter<=.35, ego_lin<=.05, ego_ang<=.1 and motion_var<2e-6 (proxy for dotd small; chair_148 fixated at motion_var 1e-9 to 4e-9, not at 1.0e-5):

| run | stuck chair | rows | still+clean rows | of those centroid_r>0.6 | median centroid_r |
|---|---|---|---|---|---|
| c3b5_on_r2 | chair_147 | 440 | 44 | 44 | 1.29 |
| s1_rs035_3 | chair_10 | 178 | 19 | 19 | 1.13 |
| s1_rs035_4 | chair_27 | 191 | 25 | 24 | 1.25 |
| s1_rs035_5 | chair_43 | 391 | 88 | 87 | 0.96 |
| s1_rs100_2 | chair_20 | 182 | 24 | 23 | 1.07 |

  Where the robot IS still and the mask is clean, the chair sits at the edge of the image (focal-normalised radius ~1.0–1.3, the fovea limit is 0.6). Example, c3b5_on_r2 chair_147 first life: cycles 12–54 (t=1.1–5.3 s after birth) npts=6380, clutter 0, ego_lin ~0.001, ego_ang 0–0.005, motion_var 1e-7, range 1.87, centroid_r constant 1.2867, fixated=0 on every row (chair_ai2_log.csv). Six seconds parked, 6380 points, rejected for off-axis only. During that time the affordance went offered (648.005) → executing (652.854) → satisfied (652.954) → "hold 200 cycles" (653.006) in logs/chair_concept.log:401-405, i.e. the controller reported "completed" without the chair ever being centred (why the controller declared completion with the chair at 1.29 is unknown; not in chair logs).
- Other stuck chairs (chair_145 in r2, chair_144/146/150/151, r3 chair_152/154/155/156) are mostly short-lived or moving views (still+clean rows 0–8); they are a second, weaker cause (never a still look).

## Q2. Re-birth / reset

- BIRTH starts at the prior: `[chair_147] RT-cov σ x=30.6cm y=30.3cm` at birth (chair_concept.log:398) and again at the second birth `31.0cm` (line 556, with `RESURRECT chair_147 ... p=0.924`, line 555). `predict()` (recursive_laplace.h:63-84) sets Σ=prior when Σ is zero and, with `clamp_to_prior`, never lets it exceed the prior, so a never-updated chair holds exactly 0.300.
- Re-births are frequent: duplicate `BIRTH 'chair_N'` per run: c3b5_on_r1 3 names (chair_122/123/124), r2 3 (145/147/148), r3 4 (153/156/157/158), s1_rs035_3 2 (7/8), s1_rs100_2 4 (18/20/22/23). `RESURRECT` 2–4 per run, "left behind in another room, dropped" 5–9 per run (chair_147: line 475 at t=776.9, re-born at 919.6).
- Reset is not universal. Counter-examples (chair_ai2_log, lives split on cycle reset): c3b5_on_r1 chair_124 life 3 starts at std 0.039 (carried); r3 chair_158 life 2 starts at 0.030. Other lives restart at 0.300 (chair_124 lives 1, 4, 5; r2 chair_148 life 2: 0.300 at a different cx -1.74 vs -2.68, i.e. the name was reused for another place). I did not find which code decides carry vs reset (recall/LTSM restore path not read); unknown.
- So re-birth makes it worse (each visit restarts at 0.30 and needs a fresh fixation) but it is not the root cause: chair_147's FIRST life (55 rows) and its 207-row life already had zero fixation.

## Q3. Posterior or prior published?

- std_cx in the CSV and the RT-cov on the node come from the belief Σ (`ai2_belief`, log at chair_fitter.cpp:~645; "src=Σ"). It is the posterior only after an `update`; on gated cycles it is the predicted Σ, clamped to the prior, so effectively the prior.
- Price: `specificworker.cpp:1366-1400` `step_epistemic` calls `epistemic_planner_.compute(inst.ai2_belief, ...)` and records `raw_gain` into `dbg_nbv_gain_raw` (→ CSV `nbv_gain_raw`). It is a function of the same Σ, so Σ at prior ⇒ the constant 3.18508 (all stuck chairs show exactly 3.18507/3.18508; s1 shows 3.19–3.25 with the existence term added by the worktree planner). Shrunk chairs show 0.7–1.7 (chair_148 1.40, chair_124 1.17).
- The offer is therefore honest about the belief; the belief is what is wrong (stale at the prior).

## Q4. Chair that DID shrink

c3b5_on_r2 chair_148 (-2.68,-0.77 world frame): same run, 244 rows, 39 fixated. Rows 168–203 are "still+clean" with centroid_r 0.28–0.50 (> not-fixated only because motion_var 1e-5 > ~1e-9, i.e. dotd not yet below 0.05 after stopping); at cycle 204 motion_var 4e-9, centroid_r 0.23, fixated=1, std 0.30 → 0.0705 in ONE update, 0.0506 next (chair_ai2_log.csv). So a single admissible frame is enough; chair_148 was centred (0.23) because the robot approached and faced it, chair_147 was never centred (1.29) at any still moment. Similar median centroid_r for shrinking chairs: fixated rows have centroid_r 0.2–0.5. Range is not the discriminator (chair_147 range 1.9 m, chair_148 2.1 m).

## Conclusion (ranked)

1. Defect-ish, evidenced: no admissible (centred) frame ever occurs for these chairs. The robot is still and sees a clean 6000-point mask but the chair is at the image periphery, and the CENTRED condition is a hard gate (`FixationCentrePrecision=false`). The Orient affordance completes (or the run moves on) without bringing the chair to the fovea.
2. Re-births restart at 0.30 (partial; carry-over exists sometimes).
3. Short-lived/moving views (never still), minor.

Cause (1) is a view-selection/policy gap plus a hard gate that CLAUDE.md's no-thresholds rule disfavours; the code already contains the model-level alternative: `ChairModel.FixationCentrePrecision` (chair_config.cpp:155; chair_fitter.cpp:431-440 inflates R by the periphery penalty instead of gating; floor `FixationPeriphFloor=0.05`).

## Targeted fix

- Option A (config only, one line): set `ChairModel.FixationCentrePrecision=true` in chair_concept etc/config (both main and the existence-info worktree). An off-fovea, still, clean frame then updates with R inflated by `periphery_penalty` (chair_fitter.cpp:753) instead of being dropped, so σ shrinks (slowly) from the 6000-point stare. Risk: the earlier history says far/peripheral frames moved chairs (comment at chair_fitter.cpp:510-511); the periphery R scale plus Woodbury common-mode is meant to handle it; unverified for chairs. Not tested.
- Option B: make the orient affordance's completion require the target chair to be centred (controller/affordance side; I did not read it). Larger change.
- Do not edit the still thresholds; they are not the blocker here.

## Rerun cost

Chair rebuild -j8 (small, one component, a few minutes) plus config edit; verify on the same scenario: c3b5_on_r2 (one furnished tour ~13 min) first, then an s1 run (rs035_5 has chair_43, 391 rows stuck) ~13 min: watch for `fixated=1` and `std_cx<0.30` on chair_147/chair_43 and no pose drift of nearby already-good chairs (cx,cy vs truth). Two tours ~26 min.

## Unknowns

- `motion_dotd` is not in the CSV; the dotd leg of `still` is only inferred through `motion_var`.
- Why the controller completed the chair_147 affordance with the chair at centroid_r 1.29.
- Which code decides whether re-birth/resurrect restores Σ or resets to prior.
