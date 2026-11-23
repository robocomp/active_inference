# Chair fix: FixationCentrePrecision = true, step-1 tours rerun (2026-11-23)

**Build**
- Worktree feat/existence-info, i.e. step 1 (existence MI, no floor).
- Plus `chair_concept/etc/config.toml: FixationCentrePrecision = true`. The centre condition is now a precision, not a gate.
- Main is unchanged.

**Runs**
- s1g_rs035_1/2/3 at room scale 0.35, and s1g_rs100_1/2 at 1.0.
- Cabinet out, LTOM empty, start (-5.33, 3.66, 0.58).
- Each run has a load log at `crossing/<run>/load.log`, every 10 s.

**Baseline**
- Step 1: s1_rs035_3/4/5 and s1_rs100_2/3, same build without the chair fix.

## Legs
| run | legs | peak 1-min load (32 cores) |
|---|---|---|
| s1g_rs035_1 | 4/4 | 28.5 |
| s1g_rs100_1 | 4/4 | 39.2 |
| s1g_rs035_2 | 4/4 | 31.6 |
| s1g_rs100_2 | 4/4 | 30.2 |
| s1g_rs035_3 | **0/4**: leg 1 FAILED, cross_unreachable at door_1 | 28.4 |

**The leg-1 failure recurred, and CPU load does not explain it.** TLC was not running. The failed run's load matched the passing runs, and the highest load (39) was in a passing run.

What the failed run shows:
- From t ≈ 50 s to 340 s the robot sat at the door_1 approach, at GT (-1.0..-2.5, 0.8).
- The policy logged "prediction failed" again and again at route point 15/23. The realized motion was about ¼ of the predicted motion (for example, 0.04 m predicted vs 0.01 m realized per second).
- The route failed twice, giving cross_unreachable. This happened in 3 rounds.

This looks like a physical hold (contact or a wedge) that the contact model does not predict (P(contact) ≈ 0), not stale data. The cause is not investigated further.

Its object data is graded below but covers only the start room plus a stray into room_1. The nightstand and toilet logs were correctly refused at harvest.

## Chair σ: fixed
| | chair nodes | updates | of which off-centre (r > 0.60) | nodes never below the 0.30 prior |
|---|---|---|---|---|
| step 1, 5 runs | 36 | 432 | 0 | **14** |
| chair fix, 5 runs | 32 | 1070 | 367 | **1** |

- **Far-view drift: none seen.**
  - Across each off-centre update, the change in planar error has a median of −0.001 to +0.002 m per run.
  - 1–3 % of off-centre updates worsen the error by more than 5 cm.
  - Per node, Δerror (last − first) has median 0.000 m. Its Spearman correlation with the off-centre share is +0.08 (p 0.66, n 30).
- **Calibration got worse in the other direction.**
  - Chair σ_last: median 0.149 m before the fix, 0.066 m after it.
  - Final planar error: median 0.135 m before, 0.157 m after. Different node sets, but the per-node Δerror is 0.
  - So err/σ went from about 0.9 to about 2.4: the chair is now over-confident. Its σ shrinks with frames and its error does not, which is the step-2 covariance issue now reaching the chair.
- **Re-births per run**, as names born more than once, out of names:
  - fix: 4/6, 4/6, 0/4, 3/8, 4/8;
  - step 1: 2/7, 4/7, 3/8, 4/7, 3/7.
  - Unchanged.

## Arbitration
| scale | set | object beat room | room beat object | median margin | chair winner price, median (max) |
|---|---|---|---|---|---|
| 0.35 | step 1 | 27 | 14 | 0.39–0.67 | 3.19 (3.28) |
| 0.35 | chair fix | 27 | 9 | 0.64–1.24 | **1.90** (3.25) |
| 1.0 | step 1 | 2 | 8 | 1.62–2.08 | 3.19 (3.27) |
| 1.0 | chair fix | 15 | 22 | 1.12–1.31 | **1.69** (3.19) |

- Chair prices now fall after the first look, instead of sitting at the 3.19 prior price.
- Room and door prices are unchanged.
- Refrigerator winners are priced 5.6–7.7 and table winners 5.1–5.5.
- s1g_rs100_2 accounts for 33 of the 37 contests at scale 1.0, 62 selections in all. 16 of the scale-1.0 object claims were refused as "standpoint blocked (occupancy), never moved".

## Promised vs delivered
**All claims, claim → next selection** (t3s1.py, n = 59 object claims):
- Spearman +0.02 overall (p 0.91), chair +0.05 (n 35), refrigerator −0.15 (n 10), table +0.36 (n 13).
- On windows with updates (n 33): Spearman +0.18 (p 0.33). Delivered median 0.47 against promised 2.90.

**Per-visit window, arrival at the viewpoint to the end of the dwell** (review5):
| set | n | chair promised | chair delivered | Spearman ALL |
|---|---|---|---|---|
| step 1 0.35 | 9 (chair 7) | 3.19 | 1.94 | −0.48 (p 0.19) |
| step 1 1.0 | 1 | 2.20 | 3.55 | n/a |
| floor build lam_0p2 | 5 (chair 2) | 3.19 | 2.19 | −0.50 (p 0.39) |
| **fix 0.35** | 16 (chair 10) | **1.72** | **1.95** | −0.29 (p 0.28); chair +0.36 (p 0.30) |
| **fix 1.0** | 4 (all chair) | 1.01 | 1.14 | −0.40 (n 4) |

- With the fix, **chair magnitudes now match per visit**: about 1.7 promised against about 1.95 delivered at 0.35, and 1.0 against 1.1 at 1.0.
- The ranking across claims is still not informative: per-class Spearman is small and not significant.
- The refrigerator per visit is 5.98 promised vs 0.19 delivered (n 5), still a large over-promise.

## C3a, for the record
| run | held of seen | phantoms | median planar error |
|---|---|---|---|
| s1g_rs035_1 | 12 of 15 | 0 | 0.17 m |
| s1g_rs035_2 | 12 of 15 | 0 | 0.16 m |
| s1g_rs035_3 (0/4 legs) | 6 of 8 | 2 | 0.23 m |
| s1g_rs100_1 | 9 of 13 | 0 | 0.14 m |
| s1g_rs100_2 | 11 of 15 | 1 | 0.09 m |

- Tables are the weak class: held 1, 0, 2, 0, 0, with 2 never born in four runs.
- The hood is not held at scale 1.0.

## Reading
- **The chair fix does what it was for.** Chairs update, their price falls after a look, and the per-visit promise matches the delivery in size. Off-centre updates do not drift.
- **It exposes the chair's over-confidence** (err/σ about 2.4). That is the same within-view shrinkage step 2 targets for the other classes.
- **Ranking is still uninformative.** Promised vs delivered remains at Spearman ≈ 0 across claims, and the refrigerator still over-promises by about 30×.
- **The leg-1 door failure is not CPU load.** It is a physical hold at the door_1 approach that the policy's motion model does not explain. It is a separate controller issue, not investigated.
