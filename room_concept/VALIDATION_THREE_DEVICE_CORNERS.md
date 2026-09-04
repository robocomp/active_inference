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

### 1.4 ✗✗ RETRACTED TWICE — the third relation exists, is running, and is already measured

**First version:** "the pair log mixes two cameras with no column to separate them." Wrong — one
`camera_ingestor_`, one device per file. **Second version (2026-09-02, committed): "`(zed ← ricoh)`
can never be measured on this agent; a second ingestor is a precondition for stage 2."** ★★★ **Also
wrong, and wrong for the worse reason: I inferred it from `camera_ingestor_` being a single
`unique_ptr` and never looked for another ingestor.** There is one.

```
etc/config.toml:1544   calibCameras = ["zed", "ricoh"]
```

Every camera beyond the driving one gets a `CalibChannel` with **its own ingestor, its own extraction
and its own evidence file**, and `loop_closure_observe()` is called from BOTH paths. The sensor
triangle is not future work — it has been running.

★ **The evidence was in front of me and I explained it away.** Both `camera_calib_Shadow_*.txt` were
stamped the same second (sep 1 22:31); I noticed, remarked on it, and moved on. Two evidence files
written at the same instant means two channels running at once, which is the whole question.

**And the third leg is already recorded**, in `etc/camera_loop.csv` — 98 808 shared-corner
observations, ricoh↔zed, over 22 shared vertices, differenced only when the two sightings are within
60 ms so the robot's own motion cannot leak in:

```
camera-vs-camera du   −0.0115°   naive SEM ±0.0035°   CLUSTER-ROBUST ±0.0573°
```

⚠ Same defect as §1.1, 16x here: per-vertex spread 0.2625° over 21 vertices. Read honestly, the
camera-to-camera yaw difference is **consistent with zero**.

★★★ **And it closes.** The two mount solves give ricoh yaw −0.2119° and zed −0.1981°, a difference of
**−0.0138°**. The loop closure measures that same difference by a completely independent route —
differencing two cameras on one corner, with no mount, no pose and no model vertex in it — and gets
**−0.0115°**. The two agree to **0.0023°**. That is the closure test's primary endpoint, already
satisfied on data in hand.

⚠ Do not over-read it: all three legs share the clustering defect, so the agreement is between two
quantities whose stated precisions are both wrong. What it establishes is that the three devices tell
one story about yaw; it does not establish that the story is true (§3).

⚠⚠ **SUSPENDED 2026-09-02 — the closure leg's px→rad scale was biased.** `px_per_rad()` returns the
scale at the **principal point**: exact for the ricoh panorama, wrong off-axis for the zed pinhole,
whose true local scale is `fx·sec²θ`. The closure *differences* the two cameras in radians, so the
error fell entirely on the comparison — measured at **82% of the leakage a LiDAR error puts into the
closure** (§2b). Both `−0.0115°` and the `0.0023°` agreement therefore came through a biased channel
and **must be re-measured** on the next run; how far they move cannot be computed from data in hand,
because it depends on where the zed's corners sat in its field and the zed channel logged no rows at
all until this was fixed. Cure: `rc::img::px_per_rad_at()`, used per row at both call sites.
★ The mount-solve difference `−0.0138°` is unaffected — it never converts pixels to angles.

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
- **What the marginalisation actually is.** `δ_v` gets a prior `N(0, S)` and is INTEGRATED OUT, not
  estimated — its value is not wanted, only its contamination removed. Per vertex accumulate
  `A_v = ΣJᵀWJ`, `c_v = ΣJᵀW`, `D_v = ΣW`, `b_v = ΣJᵀWr`, `e_v = ΣWr`, and contribute
  `A_v − c_v(S⁻¹+D_v)⁻¹c_vᵀ` and `b_v − c_v(S⁻¹+D_v)⁻¹e_v`. ★ This is the **Woodbury common-mode
  marginalisation `CLAUDE.md` already names** for correlated mask points, applied one level up: a
  covariance term, not a gate. `S → 0` recovers today's estimator exactly and `S → ∞` removes the
  level entirely, so it is a continuous knob with the current behaviour at one end.
  ⚠ The persisted evidence file changes shape: per-vertex partials, not one global `(H, b)`.
  ⚠ `S` measured from this same data (sd 5.3 px) is an empirical-Bayes hyperparameter. Label it as
  one; it is not a prior the data then confirms.

- **★★★ PRE-REGISTERED PREDICTION — ASYMMETRIC, not uniform.** (Corrected 2026-09-02; the first
  version predicted a uniform ≈130x and was wrong for a specific, checkable reason.) As `n_v` grows,
  `(S⁻¹+D_v)⁻¹ → D_v⁻¹` and the vertex contributes only `A_v − c_vD_v⁻¹c_vᵀ` — its **within-vertex**
  information. The level is gone; only variation survives. What survives differs per parameter:

  | | within-vertex signature | predicted |
  |---|---|---|
  | **yaw** | a constant pixel shift — no range or bearing dependence, so **exactly degenerate with that vertex's own u-offset** | inflates by ~the full design effect, landing at the between-vertex SEM: **0.216° (ricoh)** |
  | **pitch, height** | range-dependent (`Δd = θ_p·d²/h`), and the median within-vertex range sd is **1.16 m** on a median range of 4.8 m | inflates **strictly less** than the design effect |

  ★ The sharp version: **the marginalised ricoh yaw sigma should come out at 0.216°**, and pitch and
  height should inflate by visibly less than yaw. If all three inflate equally the nuisance is
  removing more than the level and `S` is too loose; if yaw does NOT inflate to ≈0.216° the
  per-vertex accumulation is wrong. ★ If the VALUES move a lot, the pooled estimate was being set by
  cluster composition and every number in §1.5 is withdrawn.

- **★ Marginalising does not RECOVER yaw precision — it reveals we never had it.** Nothing in this
  stage improves the mount; it makes the reported uncertainty true. The only routes to a better yaw
  are more DISTINCT vertices or smaller per-corner offsets, and no amount of driving past the same 23
  corners is either. ⚠ There is a floor beneath even that: a detector bias common to every corner IS
  yaw, and no data from this instrument separates them.

- **Where `δ_v` lives — the one real design choice.** In PIXELS it is cheap, fixes the sigma, and
  assumes nothing about cause. In the ROOM FRAME (metres) it is more physical, since both leading
  causes — map error and LiDAR corner-extraction bias — live there, and the projection makes it
  range-aware for free. ★★★ Note what the metric version IS: `δ_v` in the room frame is **landmark
  position refinement — `DESIGN §7`'s corner-as-landmark design, marginalised instead of estimated.**
  Same term; integrating it out or solving for it is the only difference between the two designs.
  Start in pixels, move to metres. ⚠ Measured: a 1/range fit does NOT distinguish the two on this
  data (both R² ≈ 0), so the pixel choice does not assume what could not be measured.
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

### Stage 1b — does TRACKING a matched vertex add information? ⚠ MEASURED, mostly NO

Asked 2026-09-02: if a vertex is tracked over time rather than treated as an independent draw each
frame, does that recover what §1.1 costs? Two tests, both pre-stated, both on existing data.

**H1 — is the offset a FUNCTION of the corner, not 23 free numbers?** If `δ_v` depended on the corner's
opening angle it would be a 2–3 parameter detector-bias model, not 23 nuisances, and almost all the
precision would survive. `angle_deg` is already logged. Over 22 vertices:
`corr(mean_ru, angle) = −0.28`, `corr(|mean_ru|, angle) = +0.34`; **neither reaches the ±0.41 needed
for p < 0.05 at n = 22. No evidence.** ⚠ Read as weak: absence of evidence over 22 points, one
covariate. Other covariates (convex/concave, wall-ceiling vs wall-floor, wall material) are untried
and cheap, and a hit on any of them would be worth more than everything else in this stage.

**H2 — does the offset move with viewing BEARING within a vertex?** On a panorama `u_img` IS bearing,
so this is free. Median `|corr(ru, u_img)|` within a vertex = **0.095**; only 1 of 22 vertices exceeds
0.3. **The offset is close to constant within a vertex** — which CONFIRMS the constant-`δ_v` model is
the right one, and simultaneously confirms that bearing diversity will not rescue yaw.

★★★ **The design effect is set by the number of DISTINCT vertices, not by how well each is tracked.**
Tracking adds rows to a cluster; it does not add clusters. It therefore cannot improve yaw, and any
scheme that reports otherwise has re-introduced the original error.

What tracking DOES buy, and it is worth having:

1. **It separates a mis-association from a detection bias, which is Stage 1's rigidity goal.** A true
   per-corner offset is CONSTANT along a track; a mis-association is a STEP. The static per-vertex
   spread cannot tell those apart — a temporal signature can, and it needs no extrinsics and no pose.
2. **It confirms the nuisance model rather than assuming it** (H2 above is exactly that test).
3. **It supplies the range diversity pitch and height feed on** — median within-vertex range sd 1.16 m.
   That variation is the reason those two survive marginalisation while yaw does not.

★ And it names the excitation, in the same form as the motion work: to separate a METRIC offset (map
error) from a PIXEL offset (detector bias), the informative motion is maximum BEARING sweep on one
corner at close range — driving PAST a corner, not toward it. Driving toward it varies range, which
is what pitch and height want. **The two nuisances have different optimal manoeuvres, and an
apartment tour supplies both.**

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

## 2b. ARM 7 — can a WRONG extrinsic be recovered? (pre-registered 2026-09-02)

The question the thesis needs answered: started from a deliberately wrong camera or LiDAR extrinsic —
the sim2real situation — does the online estimator recover the true value, and can it say WHICH device
was wrong? ★ Nothing below is run yet. It is written before the data exists, on purpose.

### The design: attribution, not fitting

★★★ **Injecting into a camera's own mount and then recovering it is nearly tautological** — that is the
same quantity in and out, and it would repeat arm 2's weakness (an injection test that mostly proves
the injection acted). The experiment that can FAIL uses the third device. Each injection site has a
DIFFERENT pre-registered signature across the three channels, and the LiDAR row is the one that tests
attribution:

| injected, +δ yaw | ricoh mount solve | zed mount solve | loop closure (ricoh−zed) |
|---|---|---|---|
| **ricoh mount** | −δ | 0 | −δ |
| **zed mount** | 0 | −δ | +δ |
| **helios / LiDAR** | −δ | −δ | **0 ± 0.032·δ — see below** |

The third row is the claim `specificworker.h` already states and has never tested: *large individual
residuals with a small difference put the fault in the LiDAR; a large difference puts it between the
cameras.* ★ If a LiDAR injection moves the loop closure, that attribution logic is wrong, and finding
that out is worth more than another confirmation.

### ★★★ "0 — unchanged" WAS A FALSIFIER WITH NO TOLERANCE (measured 2026-09-02, `mount_replay --selftest`)

Exact cancellation was never the prediction, and writing it as `0` would have fired the falsifier on
geometry. A LiDAR yaw error rotates every corner about the **LiDAR's own origin**, and the two cameras
sit at different offsets from it, so a parallax term survives the difference. On a synthetic drive
built with the real pair (ricoh panorama + zed pinhole, 23 vertices, ~1300 rows each) a **1.0° LiDAR
yaw injection moves the closure by 0.032°** — 3.2%, or **0.56σ** against the cluster-honest closure SE
of ±0.0573°. So the row stands, *with a band*: a LiDAR fault is consistent with the closure moving up
to ~3% of the injection.

⚠ **AND THE FIRST MEASUREMENT OF THAT BAND WAS 0.196° — SIX TIMES LARGER — BECAUSE OF A DEFECT IN THE
CHANNEL, NOT THE GEOMETRY.** `px_per_rad()` returns the scale at the **principal point**; for a pinhole
the true local scale is `fx·sec²θ`, so the ZED's residuals converted to angles too large off-axis while
the panorama's were exact. The closure differences the two, so the error landed entirely in the
comparison. Split measured: **parallax 0.036°, the `fx` constant 0.161° — 82% of the leakage.**
Fixed by `rc::img::px_per_rad_at()` (a local jacobian in closed form, not a tuned factor), now used by
both closure call sites; the selftest's leakage falls to 0.032°, i.e. parallax alone.
★★★ **Consequence for §1.4: the recorded closure `−0.0115°`, and its 0.0023° agreement with the two
mount solves, were computed through the biased scale and must be RE-MEASURED on the next run.** The
size of the shift cannot be computed from data in hand — it depends on where the ZED's corners sat in
its field, and the ZED channel logged no rows at all until 2026-09-02. That is now fixed too.

### Magnitude, set by the honest errors and not by taste

Cluster-honest standard errors, measured: mount yaw **±0.216°**, loop closure **±0.0573°**.
**δ = 1.0° yaw** is then 4.6σ on the mount solve and 17.5σ on the loop — unambiguous on both, and the
asymmetry is itself informative. Pitch: 1.0°. Height: 0.05 m.

### ★★★ THE PREDICTION THAT MAKES THIS WORTH RUNNING — recovery will be INCOMPLETE

With the per-vertex nuisance ON, the data's information about yaw drops to the between-corner scale,
and the prior stops being negligible. The two become comparable:

```
prior sigma_yaw   0.2005 deg      (ImageEdge.mountYawSigma = 0.0035 rad)
data, cluster-honest   0.2160 deg
combined posterior     0.1470 deg
recovery fraction = data share = 0.463   (ricoh)   0.477 (zed)
```

★★★ **A 1.0° yaw injection is predicted to come back as ≈0.46°, not 1.0°** — the estimator converging
to a weighted average of a wrong prior and a weak measurement. That is a sharp, falsifiable number,
and it answers the user's question with a qualification rather than a yes: **on 23 corners the camera
yaw cannot be fully calibrated away from a wrong start.** The remedy is more DISTINCT corners, not
more driving past these ones (§2 stage 1b).

Pre-registered outcomes, in order:
1. **PRIMARY — recovery fraction per parameter**, nuisance ON. Predicted 0.46 ± 0.10 for yaw;
   substantially higher for pitch and height, which keep their within-vertex information.
2. **SECONDARY — the attribution table above**, all three rows. The LiDAR row is the falsifier.
3. **THIRD — the same run with the nuisance OFF.** Predicted: recovery ≈ 1.0 with a tiny sigma, i.e.
   the old model APPEARS to succeed completely. ★ That contrast is the point — an estimator that
   recovers an injection perfectly while overstating its certainty 127x is the failure mode this whole
   document is about, and arm 7 is where it is shown rather than argued.

### ✓✓ RESULT 2026-09-03 — MEASURED ON DATA ALREADY ON DISK, NO DRIVE

A camera-mount injection acts entirely in camera coordinates, `pc' = R_inj·pc`, and `pc` is recoverable
from a pre-2026-09-02 row: `(u_lidar, v_lidar)` is its bearing and `range_m` its magnitude, both logged.
So the central question — *does this estimator recover a camera extrinsic misalignment* — was answered
on the 395 319 ricoh rows recorded 09-01, with no driving at all (`mount_replay --legacy-pair`).

The reconstruction is CHECKED, not asserted: recomputing `r` from the reconstructed point reproduces the
agent's own `ru, rv` to **0.0019 px mean / 0.0100 px worst** across all 395 319 rows.

| injected | recovered, nuisance ON (5.3 px) | recovered, nuisance OFF |
|---|---|---|
| pitch 1.0° | **0.999** ± 0.0006 | — |
| height 0.05 m | **0.972** ± 0.0002 | 0.999 |
| yaw 1.0° | **0.483** ± 0.164 (2.95σ) | **1.000** ± 0.0017 (573σ) |

★★★ **The pre-registered prediction was 0.463 for yaw. Measured 0.483.** The asymmetry predicted in
§2 stage 1 — yaw degenerate with a per-corner constant, pitch and height range-dependent and therefore
safe — is confirmed quantitatively and in the right direction, which validates the marginalisation as
well as the claim.

★★★ **The two nuisance columns are the whole story.** OFF, yaw comes back at 1.000 with ±0.0017° —
an estimator that *appears* to succeed completely while overstating its certainty 127×. ON, the same
data returns 0.483 ± 0.164: half the injection, honestly. Outcome 3 of the pre-registration, measured.

⚠ Legacy-mode caveats, none of which touch a recovery FRACTION (a ratio taken inside one
reconstruction): the covariance's off-diagonal was not logged; the panorama's azimuth handedness is not
in the file, so the pitch axis is convention-dependent and the ABSOLUTE baseline values are not
comparable with the live ones (the fitted baseline yaw arrives sign-flipped, which is itself evidence
the ricoh's `azimuth_sign` is −1); and the evidence file holds more rows than the CSV.

**What is now proven, and what is not:**
- ✓ the estimator recovers a camera misalignment: **pitch and height essentially completely**, on real data.
- ✓ yaw is recoverable only to ~48% on 23 corners, for the structural reason predicted — and the remedy
  is more DISTINCT corners, not more driving.
- ✗ **the zed is unmeasured**: the auxiliary channel wrote no per-row file before 09-02, so no injection
  can be replayed for it. Its predicted recovery from the honest sigmas is 0.477 — the same mechanism,
  the same 23 corners — but predicted is not measured.
- ✗ **attribution** (which device is misaligned) needs two cameras and `p_robot`, so it needs the drive.

⇒ The drive's purpose is now narrow and stateable: **the zed leg and the attribution row.** It is no
longer needed to establish that self-calibration recovers a camera misalignment.

### ✓✓✓ ARM 7 RUN 2026-09-03 — ALL THREE ROWS MEASURED, THE FALSIFIER PASSED

One tour (`etc/runs/arm7_0903_1104`): ricoh 91 152 rows / 21 corners, zed 11 821 / 20, **20 shared
corners**. Nuisance 5.3 px, injection 1.0° yaw.

| injected +1.0° | ricoh mount | zed mount | closure (ricoh−zed) |
|---|---|---|---|
| **ricoh** | **−0.454** | −0.001 (0.03σ) | **+1.000** |
| **zed** | +0.000 (0.00σ) | **−0.984** | **−1.002** |
| **helios / LiDAR** | **−0.444** | **−0.982** | **−0.018** |

★★★ **Every pre-registered row holds, including the one that had never been tested.** A LiDAR error
moves BOTH mounts and leaves the closure at **1.8% of the injection** — inside the 3.2% parallax band
predicted from geometry. A camera error moves ONE mount and puts the full injection into the closure,
with the two cameras' signs opposite. **The attribution logic is validated: the triangle says which
device is wrong.**

Per axis (nuisance ON), and the contrast with it OFF:

| axis | ricoh | zed |
|---|---|---|
| yaw 1.0° | 0.454 ± 0.175 | **0.984 ± 0.018** |
| pitch 1.0° | 1.006 ± 0.002 | 0.900 ± 0.028 |
| height 0.05 m | 0.944 ± 0.001 | 0.985 ± 0.001 |
| yaw, nuisance OFF | 0.998 ± 0.004 | 0.995 ± 0.006 |

### ★★★★ THE NUISANCE DOES NOT ONLY WIDEN THE INTERVAL — IT MOVES THE ESTIMATE (2026-09-04)

Same rows, nuisance off then on. **Four of the six mount parameters change sign**, and the two that
do not are the informative ones.

| | OFF | ON | shift |
|---|---|---|---|
| ricoh yaw | +0.1803 ± 0.0035 | +0.0362 ± 0.1754 | −0.144° |
| ricoh pitch | +0.1114 ± 0.0012 | −0.0109 ± 0.0019 | −0.122° **sign** |
| ricoh height | +0.0138 ± 0.0001 | −0.0023 ± 0.0008 | −0.016 m **sign** |
| zed yaw | +0.2894 ± 0.0062 | −0.2690 ± 0.0181 | −0.558° **sign** |
| zed pitch | +0.2790 ± 0.0079 | −0.2167 ± 0.0317 | −0.496° **sign** |
| zed height | −0.0141 ± 0.0008 | −0.0070 ± 0.0014 | +0.007 m |

★ **The ricoh yaw is the near-miss worth keeping.** It moves 0.144° — 41σ of its dishonest sigma but
only **0.8σ** of its honest one, because the interval widened by **50x**. On the panorama's worst axis
the shift disappears inside the new interval, which is exactly what the yaw/offset degeneracy predicts
and is a better illustration of it than another crossing would have been. Per-parameter shift in
honest sigmas: ricoh pitch 64.4, zed yaw 30.9, ricoh height 20.1, zed pitch 15.6, zed height 5.1,
ricoh yaw 0.8.

★★★ **This corrects how §1.1 and the Atlas had been framing it, mine included** — "marginalising does
not recover precision, it reveals we never had it" is true and incomplete. The per-corner offsets were
not inflating confidence in a roughly right answer; they were **dominating the point estimate**. The
zed's yaw moves 0.558°, which is 31σ of its own honest sigma and 90σ of the dishonest one.
⚠ Read carefully: this says nothing about being nearer the TRUE mount. There is no external reference
in this experiment (§3), so "moved toward truth" is unavailable in either direction. What is
established is that the answer depends on the modelling choice far more than on the data's noise —
so the pre-nuisance numbers are not usable by widening their error bars after the fact.
★ Seen from the residual side this is the same event as `chi2/dof` 6.99 → 1.41: the fit was
inconsistent, and the term that fixed the inconsistency also moved the answer.

### ★★★★ THE ZED PREDICTION WAS WRONG, AND THE REASON IS THE INTERESTING PART

§2b predicted **0.477** for the zed from the design-effect argument. Measured **0.984** — the zed
recovers its yaw almost completely, with a cluster-honest sigma **ten times tighter than the ricoh's**
on **eight times fewer rows**. The prediction was not a small miss; it was the wrong model.

The design effect assumes the WITHIN-cluster information about yaw is nil — that seeing one corner
again tells you nothing new. That is exactly true on a **panorama**, where `u = az·W/2π` makes a yaw
rotation a CONSTANT pixel shift, identical everywhere in the image and therefore indistinguishable
from that corner's own constant offset. It is false on a **pinhole**, where `u = fx·x/y + cx` gives
`du/dδ = fx·sec²θ`: the shift GROWS toward the edges of the field, so one corner seen at several image
positions separates a yaw from a fixed offset by itself.

★★★ **So the panorama's virtue is precisely what costs it yaw.** The ricoh was chosen because uniform
angular resolution spreads bearings and breaks the pitch/height ridge (ρ = −0.12 against the zed's
−0.98) — and that same uniformity makes its yaw degenerate with per-corner detection bias. The two
cameras are complementary in exactly the way the triangle needs: **the panorama carries pitch and
height, the pinhole carries yaw.** Measured here on the same tour: ricoh pitch ±0.002° against the
zed's ±0.032°, and zed yaw ±0.018° against the ricoh's ±0.175°.
★ The earlier ±0.21° "honest sigma" for the zed is therefore an upper bound that is loose for a
pinhole, not a measurement of it. §1.1's ricoh number stands (0.454 measured against 0.463 predicted).

⚠ **A logging defect found by the δ=0 check, and bounded.** The pair CSV was written at the default
6 significant figures. The solve weights by `cov⁻¹`, and a near-singular 2×2 amplifies input error by
its condition number: typical cond 17, but 259 of 91 152 rows above 1e6 and **9 rows not
positive-definite once rounded** — dropped by the replay though they counted live, leaving `H` 6% off
on the yaw–height cross term while `b` and `rTr` agreed to 1e-2. **Bounded by re-running with all 559
pathological rows removed: every recovery moved by ≤ 0.007** (ricoh yaw 0.454→0.448, zed 0.984→0.986,
closure leakage unchanged at 0.018), so nothing above depends on it. Fixed for future runs
(`setprecision(max_digits10)` on the pair CSV) — the same lesson `camera_calibration.h::save`
already carried two files away, and which this file needed MORE because of the `cov⁻¹` amplification.
★ `--verify` also learned that a snapshot from a RUNNING agent lags: the pool saves on its own cadence
while the CSV appends, so it now re-accumulates the CSV's matching prefix instead of refusing.

### ✓ FACTOR B, REPLAYED 2026-09-04 — AND THREE OF FIVE METRICS ARE BLIND TO IT

`--apply CAM:AXIS=VALUE` evaluates the same rows from an already-corrected mount, which is the
chapter's factor B (B0 = the graph extrinsic, B1 = the self-calibrated one). Same 91 152 ricoh /
11 821 zed rows in both columns, nuisance 5.3 px.

| | B0 nominal | B1 self-calibrated |
|---|---|---|
| ricoh yaw | +0.0362 ± 0.1754 | +0.0198 ± 0.1754 |
| zed yaw | −0.2690 ± 0.0181 | +0.0000 ± 0.0181 |
| zed pitch | −0.2167 ± 0.0317 | −0.0065 ± 0.0330 |
| chi2/dof | 1.40 / 0.75 | 1.40 / 0.75 |
| closure du | −0.0289 | **+0.2778** |

★★★ **M4, M5 and M3 cannot grade factor B, each for its own reason.**
- **M4** — σ comes from `H`, and applying a correction moves only `b`. Identical to four figures.
- **M5** — 1.40 → 1.40 and 0.75 → 0.75. Not a null: it is M5 doing what it was recorded to do, now
  against a bias of KNOWN size (the zed was 0.269° out in yaw, 0.217° in pitch, and the post-fit
  chi2 did not notice in the third significant figure).
- **M3** — moves **by arithmetic**: the closure shifted +0.3067° and the difference of the two
  applied corrections is +0.3052°, agreeing to 0.0015°. The closure is a difference of the two
  cameras' residuals and B shifts each by its own camera's correction, so the treatment displaces
  the endpoint by a quantity fixed in advance. It restates B rather than discriminating it.
⇒ **The B column needs M1/M2, and those need the localiser.** ⚠ Retracts my own earlier claim that
the paired comparison needed no new driving: B1 is *reachable* on recorded data, which is not the
same as *gradeable* on it.

★ **The recovery identity holds in a use it was not derived for.** Applying `p` should leave
`p·(1 − recovery)`. Ricoh yaw, recovery 0.454: predicted +0.0198, measured **+0.0198**. Zed at 0.984:
predicted +0.004, measured +0.000. So it also says how much of a correction survives being applied —
the more useful direction for a running robot than the injection it was derived from.

★★ **The stage-2 closure comparison, restated with honest sigmas.** The two mount solves differ by
**+0.3052 ± 0.2051**; the closure says **−0.0289 ± 0.0237** — ★ BOTH bootstrap, and they must be:
quoting the mount difference at its FORMAL sigma (±0.1763) beside a bootstrap closure gives 1.88σ,
which is the same estimator-mismatch one level up, in the arithmetic instead of in the estimator. The two
routes differ by **+0.3341**, and nothing like the 0.0023° agreement once claimed from the dishonest
sigmas. **98.9%** of that interval's variance is the ricoh's yaw alone.

⚠ Those two routes are NOT independent — both come from the same corner sightings, and the closure is
a difference of the same per-camera residuals that drive the mount solves — so combining their sigmas
in quadrature assumes a covariance of zero. **Measured by a cluster bootstrap over corners**
(`--bootstrap N`; resample the 21 corners, recompute BOTH routes on the same resample, 2000
replicates): **ρ = −0.079**. Bootstrap sd of the difference **0.2083** against quadrature's 0.2045, so
the disagreement is **1.60σ** rather than 1.63 — the independence assumption was harmless, and very
slightly optimistic rather than conservative.
★ Cheap because `H`, `b` and `rTr` are sums over the per-vertex blocks: a replicate adds up the
selected blocks instead of re-reading 100 k rows.
### ⚠ TWO CLUSTER STANDARD ERRORS FOR THE CLOSURE, DIFFERING 4.3x — AND BOTH ARE DEFENSIBLE

I first quoted the closure as **−0.0289 ± 0.1037**, from `sd(per sighting)/√k`. Measured directly on
the corner means instead of inferred:

| | value |
|---|---|
| sd of the CORNER MEANS, unweighted | 0.4562 → se `/√20` = **0.1020** |
| sd of the corner means, sightings-weighted | 0.1006 → se ≈ **0.0225** |
| per-sighting sd | 0.4638 → `/√20` = 0.1037 |
| bootstrap over corners | **0.0237** |

★★★ **The defect was a MISMATCH, not an inflation.** 0.1037 is within 2% of the honest cluster se of
the *mean of corner means* — it was very nearly right, for an estimator I was not reporting. The
reported `−0.0289` is the sightings-**weighted** grand mean, whose cluster se is **0.0237**.
The factor of four is real and is about unequal corner sizes: corners seen a few times have noisy
means (hence the 0.4562 unweighted spread), corners seen thousands of times agree closely (hence
0.1006 weighted), and the weighted mean discounts the noisy ones.
⇒ **the closure is 1.22σ from zero, and it is the TIGHT route** (0.024) against the mount difference's
0.205. A reader who infers the closure is the limitation has it backwards; **98.9% is the headline**.
⚠ A weighted mean is precise because it leans on the most-seen corners, which are not a random sample
of corners but the ones the tour dwelt on. The corner-democratic version is ±0.102. The two together
say the corners seen most agree closely about the two cameras and the corners seen rarely do not —
a statement about the tour as much as about the cameras.
★★★ **Both halves survive inspection alone; only dividing one by the other exposes it.** Surfaced by
a thesis session that recomputed and divided. Same rule as [[measurement-tools-that-lied]], arriving
in the statistics rather than in the estimator. The direct cluster figures now print beside the
bootstrap so it cannot recur silently.

★★★ **AND 1.63 WAS RIGHT BY ACCIDENT.** My original pair — formal mount sigma (0.1763, too small) with
the mismatched closure (0.1037, too large) and ρ assumed 0 — gives 0.2045 against the correct 0.2083,
so the σ-count moved 1.63 → 1.60 while BOTH its inputs were wrong and one by a factor of four. **A
final number that barely moves is not evidence its inputs were right**; two errors in opposite
directions is the ordinary case, not the surprising one.

⚠ Read 1.60 to one decimal only. A 21-of-21 resample holds ~13.5 DISTINCT corners, and the ricoh's
yaw shrinks toward the prior in proportion to how many inform it, so replicates shrink harder than
the full sample — visible as a bootstrap mean of +0.2231 against the sample's +0.3052. Hence centre
from the sample, spread from the bootstrap. **With 21 clusters and a shrinkage estimator no sharper
figure exists from this tour; more DISTINCT corners is the only thing that moves it.**

### Cost: one drive, not four

★ The injection can be applied OFFLINE. `r = uv_image − uv_lidar`, and an extrinsic perturbation
changes `uv_lidar` deterministically from `p_robot`; the association is exact-by-index so it does not
move. **Logging `p_robot` (3 floats) in the pair CSV makes all four legs replays of ONE recorded
drive** — which also removes route variation, the nuisance that
[[rgb-corner-calibration-experiment]] found swamps between-run comparisons.

✓ **BUILT 2026-09-02: `tools/mount_replay.cpp`** (`make -C build mount_replay`). It calls the agent's
own `make_pair_from` and `Accum` rather than a python replica, so it grades the estimator instead of a
copy of it. `--selftest` replays a drive whose truth is set in the tool and checks that the three
injection signatures come out distinguishable — run it before trusting any leg. ALL PASS at first
green, including the refusal path.

What the recording drive needs, and what was MISSING when this section was first written:
- ⚠ **the auxiliary camera wrote no pair CSV at all.** Only the driving camera did, so "one drive, not
  four" could never have covered both mount legs. Both now go through one `open_pair_log()`.
- ⚠ **`ceiling` was not logged**, and the closure keys on `vertex*2 + ceiling`: a replay without it
  differences floor corners against ceiling ones.
- ⚠ **only the covariance DIAGONAL was logged.** The solve weights by the full 2×2 inverse, so a
  replay handed `sigu`/`sigv` alone computes a different `H` from the same rows — and then a bug in
  the replay could not be told apart from that difference. `cuv` closes the δ=0 self-check.
- ✓ **the "replay holds `cov` fixed" approximation is GONE, not bounded.** `cov_lidar` is now logged
  on its own, so the replay recovers `cov_xy = Pxy⁻¹·cov_lidar·Pxy⁻ᵀ` at the nominal mount and rebuilds
  the weight under the injected one exactly. `--fixed-cov` keeps the old behaviour so the size of what
  would have been assumed away is printed rather than asserted (measured: 0.000% of a 1° injection —
  it *was* negligible, which is now a result instead of a hope).
- ⚠ **a run-constants sidecar** (`etc/image_edge_replay_<cam>.txt`) carries the camera model, the
  nominal mount, the prior sigmas and the LiDAR origin. Without it a replay would hard-code the mount.
  If the LiDAR origin does not resolve the flag stays false and **the LiDAR leg refuses**: a rotation
  about the wrong centre is a different experiment, not an approximation of this one.

So: **replay all four legs offline from one tour, then confirm the single most informative one (the
helios injection) live.** A replay establishes that the information is present; only a live leg
establishes that the running estimator uses it.

⚠ Prerequisite: restart on the 2026-09-02 binary, and DELETE `etc/camera_calib_Shadow_*.txt` — they
are format 1, carry no per-vertex partials, and the nuisance refuses to run on them.

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
