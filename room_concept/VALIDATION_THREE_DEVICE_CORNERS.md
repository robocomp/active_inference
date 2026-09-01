# Validating the ZED–Ricoh–Helios corner correspondence

Companion to [DESIGN_THREE_DEVICE_EXTRINSICS.md](DESIGN_THREE_DEVICE_EXTRINSICS.md). That document
says what to estimate; this one says how to know the *correspondence* is real before estimating
anything from it. Written 2026-09-01 against live logs.

★★★ **THE ORDER MATTERS AND IT IS THE POINT.** The extrinsics solve asks "what device parameters
make three observations of one corner agree?" — a question that presupposes the three observations
ARE of one corner. A vertex mis-associated in one device produces a confident, geometrically
consistent, entirely wrong constraint (`DESIGN §7`). So correspondence is validated FIRST, and by an
instrument that does not depend on the extrinsics it is meant to license.

---

## 0. What the design already settles, and must not be re-litigated

- **The model corner supplies CORRESPONDENCE, never a MEASUREMENT.** It licenses "these three
  detections are of the same physical thing" and must not enter the cost. Score a device against the
  model vertex's POSITION and the robot pose re-enters, the localiser's error contaminates the mount,
  and the method degrades into hand-eye calibration. Residuals are device↔device only.
- **The gauge is fixed by parameterisation, not by data.** A rotation common to all three devices
  cancels in every residual and is unobservable. State = 8 parameters (ricoh ×4, zed ×4) relative to
  the LiDAR. ★ **No output is a physical bolt angle.**
- **Mode B before mode A.** The closure test is a test only while the cross term is OUT of the cost;
  fold it in and closure is enforced by construction. Improving the estimator would destroy the
  instrument.

---

## 1. ⚠ MEASURED BLOCKERS — the validation cannot run today

All three found 2026-09-01 by reading the live logs, before any driving was planned for it.

### 1.1 ★★★ The association gate cannot reject anything

`etc/image_edge_pair.csv`, 55 350 rows, camera↔LiDAR:

| \|residual\| | n | median sigma | median assoc_chi2 |
|---|---|---|---|
| 0–2 px | 12 346 | 5.03 | 0.254 |
| 2–5 px | 22 232 | 5.31 | 0.314 |
| 5–15 px | 23 246 | 7.56 | 0.470 |
| 15–50 px | 4 398 | 27.66 | 0.681 |
| 50+ px | 548 | **4 392** | 1.171 |

`corr(log|r|, log sigma) = +0.494`. **The uncertainty grows with the residual it is judging**, so
chi2 stays near 1 however bad the match: **0.0% of associations exceed the 95% two-DoF bound of
5.99**, where 5% should, while residuals reach **2 482 px**. `assoc_prob` is exactly 1.0 on **93.9%**
of rows.

★ **This is not automatically an estimation defect.** A corner at grazing incidence genuinely has a
huge image-plane covariance, and down-weighting it is what a robust estimator should do. It is a
VALIDATION defect: `DESIGN §7` instructs that `assoc_prob` be *required* as an association guard, and
a guard that never fires filters nothing. The one failure mode the design singles out is the one the
instrument cannot see.

### 1.2 ★★★ No log identifies which camera produced a row

`image_edge_pair.csv` carries one undifferentiated camera↔LiDAR stream. The closure test needs
`zed←lidar` and `ricoh←lidar` as SEPARATE relations. Until a camera column exists there are not two
pairs to close over, only a mixture — and a mixture of two mounts fitted as one is a bias with no
symptom.

### 1.3 ★★ `image_edge_triple.csv` is not a triple

Of 138 926 rows: `depth_raw` = −1 on **100%**, `range_sigma` = −1 on **100%**, `depth_dt_ms` = 0 on
**100%**. `range_m` is populated (99.8%) but reaches **260.89 m** indoors, so it carries unfiltered
outliers and no uncertainty with which to discount them. Two of the three device channels are
sentinels. ★ Per the standing rule, a column of sentinels is not a measurement of zero — it is the
absence of a measurement, and must not be read as agreement.

### 1.4 The design's own step 1 is undone

`ricoh` accumulates into `mp_pool_` via `mount_pair_update()`; `zed` is a calib channel with its own
`ch.calib`. `DESIGN §8.1` names normalising these a prerequisite. 1.2 is the visible symptom of it.

---

## 2. The plan, in dependency order

### Stage 0 — make the instrument able to fail (no driving)

1. **Add a camera identifier** to every row of the pair log. Without it nothing below is possible.
2. **Log the association's INPUTS beside its verdict**: the candidate set size, the runner-up's
   distance, and the accepted match's distance. ★ A gate is judged by what it REJECTED; a log that
   records only acceptances cannot be audited, and this one currently records only acceptances.
3. **Populate or remove the triple's dead columns.** A sentinel that looks like data is worse than a
   missing column, because it survives into an analysis as a number.

### Stage 1 — audit correspondence with the extrinsics FROZEN (existing data)

Run on logs already recorded. The extrinsics must not move, or the audit is scoring the thing it is
meant to license.

- **Margin, not fit.** For each accepted association report `d(runner-up) / d(accepted)`. A
  correspondence is trustworthy when the second-best candidate is far away, NOT when the best one is
  close. ★ This is the quantity the current log cannot produce, and it is the one that matters.
- **Rigidity.** Two detections of one corner by two devices sit at a FIXED relative offset in body
  frame, whatever the pose. Compute that offset per (vertex, device-pair) across the run: its
  SPREAD is a correspondence error measure requiring no extrinsics and no pose. A correct
  correspondence gives a tight cluster with a possibly-wrong mean; a mis-association gives a wide one.
  **The mean is the extrinsic error and the spread is the correspondence error — separating them is
  the whole audit.**
- **Chi2 recalibration.** Report the empirical distribution of `assoc_chi2` against its 2-DoF
  reference. It currently has 0% above 5.99; a correctly scaled gate has ~5%. Recalibrate the
  covariance until it does, then re-run everything above. ⚠ Recalibrating changes which matches are
  accepted, so the audit must be re-run, not patched.

### Stage 2 — the closure test, mode B (needs driving)

Three INDEPENDENT pairwise solves, cross term OUT:

```
(zed ← lidar)  ∘  (lidar ← ricoh)   ==   (zed ← ricoh)
```

Estimate the third independently; the mismatch is the validation. **No ground truth, no pose, no
external instrument** — which is why it generalises to the real robot where there is no supervisor.

**Pre-registered endpoints, in order:**

1. **PRIMARY — the closure residual**, per parameter (pitch, height, yaw, dt), against each solve's
   own posterior sigma. Predicted: within combined sigma. ★ A closure residual LARGER than the
   sigmas says the correspondence or the geometry is wrong; a residual much SMALLER says the sigmas
   are inflated — which §1.1 already suggests, so read it in that light rather than as success.
2. **SECONDARY — closure stability across the run.** Split into halves and close each. A cycle that
   closes on the whole run but not on its halves is closing by averaging, not by agreeing.
3. **THIRD — closure under a deliberately corrupted association.** Inject a known vertex swap on a
   small fraction of frames and confirm the residual grows. ★★★ **Without this the test is
   unfalsified: a closure that would close anyway proves nothing.** This is the analogue of arm 2's
   injection and it is what makes stage 2 an experiment rather than a demonstration.

⚠ **Do not fold `r_cross` in until stage 2 has reported.** Mode A destroys the only pose-free check
available, and `DESIGN §8` makes the decision point explicit: if the pitch/height correlation does
not improve, stop at mode B.

---

## 3. What this plan will NOT establish

- **Absolute mounts.** The common-mode rotation is unobservable by construction. Closure validates
  that three devices tell one story, never that the story is true. ★ Only an external instrument
  decides that, which is what ground truth is for — and per the lesson from the motion work, an
  internal consistency measure goes quiet exactly when two factors are biased into agreement.
- **Per-corner detection offsets.** They are per-corner, not global, and belong in a corner-as-
  landmark design (`DESIGN §7`). A ~1.7 px per-corner bias already converted into heading error once
  through the `corr(x,θ) = 0.98` ridge.
- **That `cam_dt` is a time offset.** It is time × `k_v`, so it must never be read without the
  motion block's `k_v` from the same run.
