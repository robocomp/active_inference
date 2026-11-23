# Gyro calibrator and the stops (limitation TODO, `sections/10-limitations.tex:70`), 2026-11-23

Scripts: `auro_1122/scripts/review2/stops.py` (output `stops.csv`), run with `python3 -I`, reading existing logs only. No run, no Webots, nothing started.

## Answer in one paragraph
**The per-session numbers for the "four live base sessions" are not derivable here.** The paper's four sessions are on the physical robot
(`sections/09-evidence.tex:521-527`: "four live sessions of ordinary driving (12-22 min each)", cold starts, learnt gyro bias checked against the IMU at the stops). Their logs
are not on this machine: all 117 `room_concept/tmp/heading/heading_*.csv` (2026-11-16 to 11-23) and every `etc/runs/arm*` file have ground truth in `gt_x`
(the header says `gt_* NaN on the real robot`, and none is NaN), so they are Webots sessions; `etc/runs/arm7_0903_1104` holds calibration files only; no other
`heading_*.csv` or live-session folder exists under `/home/pbustos` or the Drive tree (searched `find` on names, `grep` for the column `hc_t_gyro` over all csv). The sessions come from the real-robot work and were not found in the repo, the memory notes
(`auro-cortex-journal-paper.md`, `auro-1122-evidence.md`, `resume-11-13-auro-series.md`) or the paper's `PROGRESS.md`.
What IS derivable is the mechanism (below) and the same quantities on the simulated sessions that use the same calibrator. The mechanism says the stops reach the bias estimate
with almost no weight; the simulated sessions measure how much.

## Mechanism (file:line)
1. **A parked robot cannot close an episode.** `motion_calibration.h:171-172` (`episode_min_trans`, `episode_min_rot`) and `:473-476`: an episode closes on `|acc_fwd| >= min_trans or |acc_th| >= min_rot` (or the falling edge of a pose correction, `:475`). Comment at `:462-466`: "A parked robot must never close an episode".
   The live config is `etc/config.toml:1035-1036`: `MotionCalibEpisodeMinTrans = 1.00` m, `MotionCalibEpisodeMinRot = 0.80` rad (the source default is 0.25 m / 0.20 rad). So there is no "stop sample" row: every row is a stretch of motion, and no row is made only of a stop.
2. **Stopped time can sit inside a moving episode, but it is scaled by P(moving).** `room_concept_prediction.cpp:94-98`: the covariates are `hc.t_gyro *= g_ro` (and `th_gyro`, `th_wheel` by `g_ro`, `fwd_wheel` by `g_tr`), with the gains `zupt_pred_gain_{ro,tr}_` set at `room_concept_motion_prior.cpp:302,344`.
   With rest-on-prediction on (`etc/config.toml:529-530`, `PreintZuptOnPrediction = true`, `PreintZuptPredLearnRest = true`) the gain on a parked cycle is about 0.01, so a parked second adds about 1% of a moving second to the `-t_gyro` regressor that identifies `b_omega`.
   `EXPERIMENT.md:1157-1160` records why: before 2026-10-04 parked time was counted in full and diluted the fit (rest-on-prediction withheld rotation on 85% of cycles on arm 7's gyro-all leg).
3. **So the stops are not "a small part of the data the calibrator receives" by count. They are down-weighted by design.** The statement that stops "are the only part of the driving that separates the bias from the wheel scales" concerns the IMU check done outside the calibrator (`sections/09-evidence.tex:485`), not the calibrator's input.
   The calibrator's bias estimate is identified from moving cycles at different angular speeds (`calibration_viewer.cpp:109-116`).
   Whether the four physical sessions ran with the rest gain on is not derivable (their configs are not here); the config in force today has it on.

## Same quantities on the simulated sessions (same calibrator, gt available, calibrator ON, `calib_applied > 0`, >= 5 min)
Definitions (all from columns of the heading log; `stops.py`): a "stopped cycle" is one with `rest_gain_ro < 0.5` (P(moving) < 0.5). The gyro time a cycle carries is `hc_t_gyro * rest_gain_ro`
(the log stores `hc_t_gyro` before the gain, as `room_concept_prediction.cpp:97` multiplies a local copy). "Ratio" = mean rest gain on stopped cycles / mean on moving cycles = the weight of one stopped second relative to one moving second in the `b_omega` regressor.
The number of episodes (rows) split by stopped/moving is not derivable: the log carries only `calib_episodes`, the count of the window at that moment (max 2..87 per session), not a per-row flag. By item 1 the stop-only rows are 0 by construction.

| session (sim, calibrator ON) | min | cycles | cycles with P(moving)<0.5 | gyro-time carried by those cycles (after the rest gain) | mean rest gain on stopped / moving cycles | ratio |
|---|---|---|---|---|---|---|
| 2026-11-21_13-37-25 | 17.7 | 19807 | 58.4% | 0.43% | 0.006 / 0.999 | 0.006 |
| 2026-11-21_14-46-04 | 22.4 | 24050 | 38.4% | 0.32% | 0.009 / 0.998 | 0.009 |
| 2026-11-22_18-21-09 | 19.5 | 20264 | 40.8% | 0.30% | 0.008 / 0.998 | 0.008 |
| 2026-11-23_10-55-45 | 12.5 | 14468 | 33.1% | 0.87% | 0.021 / 0.997 | 0.021 |
| all 69 calibrator-ON sim sessions, median | 10.7 | | 32.1% | 0.56% | | 0.015 |

Ranges over the 69 sessions: cycles with P(moving) < 0.5 = 24-98% (median 32%; the 98% one is a 61 min session, `2026-11-21_21-31-32`, mostly parked); the gyro time they carry = 0.27-10.7% (median 0.56%); weight ratio 0.004-0.022 (median 0.015). Wheel-stationary time `zupt_dt` / `total_dt` = 0.3-3.9% (median 0.8%), smaller than the P(moving)<0.5 share because the learnt rest filter calls rest earlier than the wheel ZUPT does.
The four sessions in the table are the simulated ones nearest to 12-22 min with the calibrator on, picked by duration only, and are **not** the paper's four physical sessions.

## Suggested wording (only what is measured)
"In simulation, on 69 sessions with the calibrator on, the robot is judged at rest on about a third of the cycles (median 32%), but because the covariates are multiplied by the probability of moving, those cycles carry 0.6% of the gyro time the bias is regressed on (one parked second counts about 1.5% of a moving second). A parked robot also never closes an episode (minimum 1 m or 0.8 rad)." The same measurement on the four physical sessions was not possible from the data kept on this machine.
