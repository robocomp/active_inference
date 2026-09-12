# MatterportLayout — 594 rooms

## Provenance (state this in the paper, do not let a reviewer find it)

MatterportLayout carries 2295 Manhattan room annotations and **no camera poses**. The observations
here are therefore **synthesised from the annotated polygon itself** by the selftest harness
(`tools/wall_slam_selftest.cpp`, `WS_LAYOUTS=<file>` mode, `name;x,y x,y …`), with the explorer
choosing viewpoints and the estimator seeing only simulated range returns.

That makes this a test of **geometric generality** — does the estimator handle the shape diversity of
594 real floor plans — and NOT a sensor benchmark. Reconstructing a layout from a scan generated from
that same layout is close to circular, and a 0.98 IoU reads as suspicious unless the framing is
explicit. Real-sensor evidence has to come from the robot (and, if it lands, MP3D-FPE, which has
real depth and poses).

Corpus: `corpus594.txt`, run in 12 chunks; logs in the session scratchpad `c594/`.
Two explorer arms: `pre` = EIG explorer, `dwell` = EIG + dwell term.

⚠ Failed rooms write **IoU 0.000 and Hausdorff 1e9**. The sentinel poisons any mean taken over the
raw column — the first aggregate computed here reported a mean Hausdorff of 1.7e6 m. Filter, and
report failures as their own count.

## Results

| arm | n | fail | IoU mean | median | p10 | ≥0.95 | ≥0.90 | Hausdorff med | p90 |
|---|---|---|---|---|---|---|---|---|---|
| EIG explorer | 594 | 2 | 0.9827 | 0.9880 | 0.9670 | 95.3% | 98.3% | 0.038 m | 0.191 m |
| EIG + dwell  | 594 | 1 | 0.9823 | 0.9870 | 0.9670 | 94.6% | 98.3% | 0.069 m | 0.209 m |

The two arms are within noise on IoU. Plain EIG has the better Hausdorff median (0.038 vs 0.069 m);
dwell halves the hard failures (1 vs 2). Neither difference is worth a claim on n=594 — report one
arm and say the other is equivalent.

## EIG + dwell, by ground-truth corner count

| corners | rooms | IoU mean | median | p10 |
|---|---|---|---|---|
| 6  | 330 | 0.9867 | 0.9905 | 0.9760 |
| 8  | 156 | 0.9801 | 0.9850 | 0.9670 |
| 10 | 74  | 0.9773 | 0.9805 | 0.9580 |
| 12 | 20  | 0.9519 | 0.9755 | 0.8640 |
| 14 | 10  | 0.9718 | 0.9730 | 0.9560 |

Rooms with ≥10 GT corners: n=107, IoU mean 0.9718. Degradation with complexity is mild and the p10
column is where it shows — the 12-corner bucket's p10 of 0.864 against a median of 0.9755 says the
losses are a few bad rooms, not a drift.

Worst 8 (dwell, failures excluded): 0.821, 0.832, 0.834, 0.848, 0.862, 0.864, 0.879, 0.892.

## Assets

`mosaic_cells_120.html` + `sample120.tsv` — 120-room mosaic (truth vs estimate polygons per cell).

---

# Is the PUBLISHED uncertainty honest?  (594 rooms, ground truth)

Method: final-frame published polygon from the `WS_POLY_ROOM` trace, mapped to world through the
trace's `# origin`, matched to the truth vertices by **Hungarian assignment**. Surplus corners on
either side stay UNMATCHED and are counted as structure errors — never as position error. That
matters: nearest-vertex matching instead gives a mean error of 0.43 m and a p90 of 1.99 m against a
median of 0.023 m, because a spurious corner has no truth counterpart and its "error" is noise.

`Corner::sigma` is the **largest-axis** σ, so this tests whether a BOUND holds, not a χ² fit.

## The defect, and the fix

Before: the corner covariance propagated only the two walls' statistical uncertainty, which averages
down with the returns behind it. The true error does not follow it down — it is FLAT at ~2 cm across
every σ band, so a corner claiming 1 cm was no more accurate than one claiming 5 cm.

Measured floor: **0.019 m**, against a harness sensor of 0.020 m per return. Added as
`Params::corner_model_sigma` (σ_eff² = σ_stat² + σ_model²) — a variance term, not a threshold.

| | ≤1σ | ≤2σ | ≤3σ | publishable | its claim holds |
|---|---|---|---|---|---|
| no floor | 46.7% | 70.5% | 81.2% | 89.0% | 85.8% |
| floor 0.02 m (predicted) | 64.9% | 85.3% | 91.1% | 88.4% | 85.9% |
| **floor 0.02 m (measured live)** | **65.3%** | **85.0%** | **91.2%** | **89.0%** | **86.1%** |

The live row is a re-run of all 594 rooms with the floor active, not post-processing — it had to be,
because the floor also feeds the explorer's own σ prediction and could have changed the trajectories.
It did not: mean IoU 0.9708 → 0.9695, exact-structure rooms 319/587 → 311/586. Too small to call a
regression, too small to call noise without repeats.

## The actionable claim

**A corner this system declares publishable (σ ≤ 6 cm) has a median true error of 2.0 cm, and its
error is within the 6 cm it claims in 86% of cases.** Corners it withholds: median 3.6 cm, within
6 cm only 64% of the time. The gate discriminates, and now the number it gates on means something.

## Caveat

Scans are still SYNTHESISED from the annotated polygons, so this validates the PROPAGATION — real
geometry, our noise model. The 0.019 m floor is measured against a simulated 0.020 m sensor and
**must be re-measured on real data**, not carried over.
