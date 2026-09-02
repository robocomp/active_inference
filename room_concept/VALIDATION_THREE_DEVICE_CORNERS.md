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
  the LiDAR. ★ **No output is a physical bolt angle.** ⚠ Measured 2026-09-02: only **6** of the 8 are
  ever informed — `dt` carries identically zero information in both evidence files (`H[i,3] = b[3] =
  0`), because the pair residual is formed at one instant and nothing in it varies with a time offset.
  It is a declared parameter that no observation reaches.
- **Mode B before mode A.** The closure test is a test only while the cross term is OUT of the cost;
  fold it in and closure is enforced by construction. Improving the estimator would destroy the
  instrument.

---

## 1. ⚠ THE BLOCKER, MEASURED — and three that were misdiagnosed

⚠⚠ **Revised 2026-09-02 after reading the code that writes the logs.** The first version of this
section listed four blockers found by reading the logs alone. Three of them were artefacts of the
reading, not defects in the system, and are corrected below with the evidence. ★★★ The pattern is
the one this project keeps paying for: **ask what a number IS before asking what is wrong with it.**
Each of the three was a quantity behaving exactly as its own code specifies, judged against an
expectation formed without opening that code.

### 1.1 ★★★ THE REAL BLOCKER: the mount posterior ignores that the rows are ~20 clusters

`etc/image_edge_pair.csv`, 395 319 rows, one run, ricoh↔LiDAR. Grouping the horizontal residual `ru`
by the model vertex it belongs to:

| vertex | n | mean `ru` (px) | sd `ru` |
|---|---|---|---|
| 11 | 3 861 | **−17.17** | 17.48 |
| 13 | 2 407 | **−13.59** | 20.66 |
| 7 | 2 372 | −9.22 | 9.23 |
| 20 | 29 742 | **−8.42** | 4.14 |
| 2 | 7 572 | −7.02 | 10.00 |
| 25 | 31 031 | −3.12 | 4.43 |
| 27 | 19 859 | +2.81 | 14.72 |

Each vertex carries its own systematic offset, and they **spread over 5.3 px (sd)**. The mount
solve's own signal is far smaller: its fitted yaw of −0.21° is 1.1 px on the ricoh, and its pitch
0.33 px. ★ **The per-corner offsets are five times the quantity being estimated.**

The estimator adds one independent measurement per ROW. The rows are not independent — they are
**23 clusters sampled thousands of times each**, and the cluster means disagree. That inflates the
effective sample size by `n_rows / n_clusters` and shrinks the posterior sigma by its square root:

```
reported yaw posterior sigma (ricoh)  0.0017 deg
cluster-robust standard error         0.2161 deg      <- 127x larger
design effect sqrt(395319 / 23)                 131x  <- and this is why
```

**127 measured against 131 predicted.** The two agree, which identifies the mechanism rather than
merely flagging a discrepancy. The consequence is decisive:

★★★ **The fitted ricoh yaw (−0.2119°) is smaller than its own cluster-robust error (±0.2161°). It is
not distinguishable from zero.** The same holds for the zed (−0.1981°, design effect √(60003/23) =
51, cluster-robust ≈ ±0.21°). Both cameras' mount estimates are, at present, statements about which
vertices happened to be seen most often — vertex 20 and vertex 25 together supply 15% of all rows
and disagree with each other by 5 px.

This is not an argument for widening a prior. It is the same defect the codebase has met before in
another costume: a tight number produced by counting correlated evidence as independent. The cure is
in the model — a **per-vertex offset nuisance**, marginalised, so that a corner contributes
information about the mount only through how its offset CHANGES with viewing geometry, not through
how often it is seen. `DESIGN §7` already reserves per-corner offsets for a corner-as-landmark
design and records that "a ~1.7 px per-corner bias already converted into heading error once". **The
offsets measured here reach 17 px, an order of magnitude beyond the one that already caused harm.**

⚠ Until that term exists, the three-device closure test cannot be run: closure compares two mount
solves against their posterior sigmas, and those sigmas are wrong by two orders of magnitude. A
closure that "passes" against them would prove nothing, and one that "fails" would indict the
correspondence for a defect in the covariance.

### 1.2 ✗ RETRACTED — "the association gate cannot reject anything"

The original claim: `assoc_chi2` never exceeds the 95% two-DoF bound of 5.99, where 5% should, so
the gate filters nothing. **The gate is what makes that true.** `corner_detector.cpp` leaves every
(model corner, detection) pair whose squared Mahalanobis distance exceeds `Params::assoc_chi2`
(= 5.991) INFEASIBLE; `solve_hungarian` never assigns an infeasible pair; and the only writer of
`assoc_chi2_val` writes that same gated cost. **An emitted match cannot carry a value above the
bound, so the observed 0.0% is arithmetic, not evidence.** Measured max over 395 319 rows: 5.9909.

★★★ **A gate cannot be audited by the distribution of the quantity it truncates.** The distribution
is not even concentrated near 1 as reported — it fills the interval (median 0.38, p99 4.75, p99.9
5.84), which if anything says the gate is *binding*, the opposite of the original reading.

The correlation `corr(log|r|, log sigma) = +0.49` was real but described two different associations
as though they were one: `r` and `sigma` are the camera↔LiDAR pair's, while `assoc_chi2` is the
detector's model-vertex association, computed upstream in a different frame. Judged against its OWN
sigma the pair covariance is **conservative, not inflated**: median `|r|/sigma` = 0.61, and 2.65% of
rows exceed the 2.448 bound where 5% would be calibrated.

★ What the log genuinely could not produce was the **margin** — and `runnerup_chi2` was being
computed in `CornerMatch` all along and simply never written out. Fixed in Stage 0.

### 1.3 ✗ RETRACTED — "`image_edge_triple.csv` is not a triple"

`depth_raw` = −1 on 100%, `range_sigma` = −1 on 100%, `depth_dt_ms` = 0 on 100%. All three are one
cause, and it is not a defect: **`ImageEdge.camera = "ricoh"`, and the ricoh advertises no depth
stream.** `probe_depth` returns 0 with a one-shot warning, leaving the three fields at their declared
"not available" values — and `range_sigma` is explicitly documented in the source as a placeholder
that nothing may consume until a depth sigma is MEASURED. That is a correct report of an absent
channel, not a failed measurement.

★ The real gap was that a reader of the CSV could not tell the two apart. Stage 0 adds `has_depth`
and a `camera` column, so absence is stated rather than inferred from a column of sentinels.

`range_m` reaching 1 241 m indoors is also geometry, not an outlier: `place_triple_points_in_room`
intersects the pixel ray with the corner's own height plane, and a near-parallel ray meets it far
away. The guard rejects only |d_room.z()| < 1e-3.

### 1.4 ⚠ REFRAMED — the closure test's third relation cannot be measured at all

The original claim was that the pair log mixes two cameras with no column to separate them. It does
not mix them: `room_concept` holds a **single `camera_ingestor_`** and `ImageEdge.camera` selects one
device per run, so each file is one camera throughout. The defect was **provenance and survival**,
not mixture — a fixed filename means run 2 silently destroys run 1's record of a *different* device.
★ Note the asymmetry that made this easy to miss: the estimator's own evidence file was already
keyed correctly (`camera_calib_<robot>_<camera>.txt`); only the diagnostic beside it was not.

But reframing it exposes something neither document had stated:

★★★ **`(zed ← ricoh)` can never be measured on this agent as it stands.** Closure needs that third
relation estimated INDEPENDENTLY, which requires one corner seen by both cameras in one frame. With
a single ingestor there is no code path that ever holds a ricoh corner and a zed corner together. A
second ingestor is real work, and it is a precondition for stage 2 that appears in neither this plan
nor `DESIGN`.

### 1.5 ✓ What already exists, and is worth having

Both cameras have accumulated evidence on disk in information form, correctly keyed:

| | n pairs | pitch | height | yaw | chi2/dof | cond | rho(pitch,height) |
|---|---|---|---|---|---|---|---|
| ricoh | 395 171 | −0.0619° | −0.0154 m | −0.2119° | 1.78 | 9.2 | −0.12 |
| zed | 60 003 | −0.1538° | +0.0032 m | −0.1981° | 1.74 | 145.5 | **−0.98** |

(Solved offline by `tools/solve_mount.py`, replicating `rc::mount::Accum::solve`. Read the values
subject to §1.1 — the sigmas that would normally accompany them are omitted here deliberately,
because they are the ones shown to be wrong by 127x.)

Two things survive §1.1 because they do not depend on the sigma scale:

- **`chi2/dof` = 1.78 and 1.74** on two cameras with different optics, different fields of view and a
  6.6x difference in sample count. The residual model is mildly optimistic (~1.3x) and **equally so on
  both** — which is a property of the shared pipeline, not of either mount.
- **`rho(pitch, height) = −0.98` on the zed against −0.12 on the ricoh**, with condition numbers 145.5
  and 9.2. ★★★ **The pitch/height ridge that `DESIGN §8` builds mode A to break is a ZED problem, not
  a system problem.** The panorama sees corners in every direction and decorrelates the two by
  geometry alone. The decision point in `DESIGN §8` — "if the pitch/height correlation does not
  improve, stop at mode B" — now has a measured value to be evaluated per camera, and for the ricoh
  the motivation for mode A is largely absent.

## 2. The plan, in dependency order

⚠ Revised 2026-09-02. The original stage 0 was three tasks against §1.1–1.3; two of those turned out
to have nothing to fix, and the task that mattered was already half-done in the source. Stage 1 now
carries the work, because §1.1 must be closed before stage 2 can mean anything.

### Stage 0 — make the instrument able to fail (no driving) ✓ DONE 2026-09-02

1. ✓ **Camera provenance travels with the evidence.** `ImageEdgeObs::camera` is stamped where the
   observation is built and read by every consumer downstream. ★ It is carried rather than looked up
   because `ImageEdge.camera` is runtime-overridable: a consumer that asks the config gets the camera
   selected NOW, not the one this row came from.
2. ✓ **Both diagnostic CSVs are keyed by camera** (`image_edge_pair_<cam>.csv`,
   `image_edge_triple_<cam>.csv`) and carry a `camera` column, so two runs of two devices both
   survive — the prerequisite for having two relations to close over at all.
3. ✓ **The association's INPUTS are logged beside its verdict**: `n_rivals` (how many model corners
   were in gate for this detection — 0 means there was no choice to get wrong) and `runnerup_chi2`
   (how far the best loser sat, written as −1 when there is no rival, since an uncontested match has
   an INFINITE margin and writing the 1e9 sentinel would put "no rival" into an average).
   `runnerup_chi2` was already computed in `CornerMatch` and discarded; `n_rivals` is new.
4. ✓ **`has_depth`** distinguishes an absent depth channel from a failed probe, so the sentinel
   columns state their meaning instead of leaving it to be inferred.

Tools: `tools/pair_audit.py` (residual/margin/cluster audit, and it REFUSES to report a margin on a
pre-2026-09-02 file rather than printing NaNs), `tools/solve_mount.py` (offline replica of
`rc::mount::Accum::solve`).

### Stage 1 — the per-vertex offset nuisance ★ THE CRITICAL PATH

§1.1 is a modelling gap, not a logging gap, and nothing downstream is interpretable until it closes.

- **Add a per-vertex offset to the generative model** and marginalise it. Per the standing rule this
  is not a threshold or a rejection test: a corner's detection has a systematic component that is a
  property of THAT corner's geometry, so it belongs in the model as a nuisance with its own prior,
  and the mount then draws information from how a corner's offset varies with viewing geometry
  rather than from how many times it was seen.
- **Pre-registered prediction.** The mount posterior sigmas inflate by roughly the design effect
  (≈130x ricoh, ≈50x zed) and the fitted values move by less than the current sigma. ★ If the VALUES
  move a lot, the pooled estimate was being set by cluster composition and every number in §1.5 is
  withdrawn; if the SIGMAS do not inflate, the nuisance is not absorbing what it was built for.
- **Margin audit**, on a run recorded with the stage-0 binary: `runnerup_chi2 / assoc_chi2` over
  contested rows. A correspondence is trustworthy when the second-best candidate is FAR, not when the
  best one is close. ★ Report the fraction of rows with `n_rivals` = 0 first: on those the gate made
  no choice, and including them would flatter the margin with cases that had no way to be wrong.
- **Rigidity.** Two detections of one corner by two devices sit at a FIXED relative offset in body
  frame, whatever the pose. Its SPREAD is a correspondence error needing no extrinsics and no pose.
  **The mean is the extrinsic error and the spread is the correspondence error — separating them is
  the whole audit.** ⚠ This now has a known confound: §1.1's per-corner offsets enter the MEAN, so it
  must be computed per vertex and not pooled.
- ✗ **Chi2 recalibration — dropped.** It rested on §1.2, and the gate truncates the statistic it
  would recalibrate against. The pair covariance's own calibration is measured instead (median
  `|r|/sigma` = 0.61, 2.65% over the 2.448 bound), and it is conservative by ~1.3x, which `chi2/dof`
  = 1.75 independently confirms.

### Stage 2 — the closure test, mode B (needs a second ingestor, then driving)

⚠ **Blocked on §1.4 and §1.1, in that order.** Three INDEPENDENT pairwise solves, cross term OUT:

```
(zed ← lidar)  ∘  (lidar ← ricoh)   ==   (zed ← ricoh)
```

The first two exist today. **The third requires one corner detected by both cameras in one frame,
which requires a second `camera_ingestor_`** — that is the precondition, and it is real work.

**Pre-registered endpoints, in order:**

1. **PRIMARY — the closure residual**, per parameter, against each solve's own posterior sigma.
   ⚠ Only meaningful after stage 1: today those sigmas are understated ~127x, so a residual "within
   sigma" would be a statement about the bookkeeping. ★ A residual LARGER than the sigmas says the
   correspondence or the geometry is wrong; much SMALLER says the sigmas are inflated.
2. **SECONDARY — closure stability across the run.** Split into halves and close each. A cycle that
   closes on the whole run but not on its halves is closing by averaging, not by agreeing. ★ With
   §1.1 understood, split by VERTEX SET as well as by time: halves that share vertices share their
   offsets, and would agree for the wrong reason.
3. **THIRD — closure under a deliberately corrupted association.** Inject a known vertex swap on a
   small fraction of frames and confirm the residual grows. ★★★ **Without this the test is
   unfalsified: a closure that would close anyway proves nothing.** This is the analogue of arm 2's
   injection and it is what makes stage 2 an experiment rather than a demonstration.

⚠ **Do not fold `r_cross` in until stage 2 has reported.** Mode A destroys the only pose-free check
available. ★ And §1.5 narrows the question mode A was built to answer: the pitch/height ridge is
−0.98 on the zed and −0.12 on the ricoh, so the decision is per camera, and for the panorama the
motivation is largely absent.

## 3. What this plan will NOT establish

- **Absolute mounts.** The common-mode rotation is unobservable by construction. Closure validates
  that three devices tell one story, never that the story is true. ★ Only an external instrument
  decides that, which is what ground truth is for — and per the lesson from the motion work, an
  internal consistency measure goes quiet exactly when two factors are biased into agreement.
- ⚠ **~~Per-corner detection offsets~~ — NO LONGER OUT OF SCOPE.** This bullet said they belong in a
  corner-as-landmark design (`DESIGN §7`), citing a ~1.7 px bias that once converted into heading
  error through the `corr(x,θ) = 0.98` ridge. §1.1 measured them at up to **17 px, spread sd 5.3 px
  across 23 vertices** — five times the mount signal itself — and showed they understate the mount
  posterior by 127x. ★★★ **A nuisance large enough to swallow the quantity being estimated is not
  someone else's problem.** They are now stage 1, and stage 2 is blocked behind them.
- **That `cam_dt` is a time offset.** It is time × `k_v`, so it must never be read without the
  motion block's `k_v` from the same run.
