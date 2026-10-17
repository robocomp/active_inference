# View Atlas — per-object appearance memory for visual re-identification

Design spec, 2026-10-17. Status: **phase 1 BUILT and tested** (module, archive IO, crop/encoder, tools, fitted constants, offline replay); the four offline replay criteria **FAILED** (see §8); phase 2 (agent integration) not started.

A concept agent keeps, for every object it owns, a small map from **viewing direction in the object's
own frame** to **expected appearance**, filled in as the robot moves around the object. The map lets the
agent (a) keep two same-class objects from swapping identity and (b) recognise an object that died
(left view, was moved, was carried) when it is detected again, instead of giving birth to a new one.
Atlases outlive their object's node in a per-agent archive.

---

## 1. Evidence this design rests on

Three offline spikes on 2026-10-17 (scripts were throwaway; numbers are recorded in memory
`mask-dino-descriptor-spike`). Ground truth = Webots instance, labelled by casting each mask's bottom
pixel onto the floor into the world's furniture footprints, checked by colour.

| Question | Data | Result |
|---|---|---|
| Can one DINO vector per **mask** separate same-class instances? | apartment ZED dump, 2114 frames, 649 masks | **No.** In-frame pooled DINOv2, same-class AUC 0.667. Centering + GeM (the place-encoder recipe) made it worse (0.54–0.56). Colour histogram 0.937. |
| Does accumulating many views per **object** help? | same, query vs per-object prototype | **Yes.** DINOv3 crop 0.90–0.95, colour 0.96, both **0.984**; top-1 ≈100 % on chairs/heaters/worktops. |
| How does appearance decay with viewing angle? Does moving the object break it? | orbit dataset, 898 frames, `retina/etc/zed_orbits` (world `webots-shadow/worlds/piso/piso_orbits.wbt`) | DINOv3 rival AUC 0.96 (<10°) → 0.80 (30–60°) → 0.63 (90–135°) → 0.51 (>135°). **Colour ≥ 0.96 at every angle.** Moved chair: DINO 78 % (box crop) / 81 % (background blanked); colour 100 % either way; blanking raised colour's margin (own 0.77→0.87, rival 0.48→0.19). |

Consequences: shape (DINO) needs a viewing-direction atlas with bins of ~20–30°; colour needs one
prototype; crops must be background-blanked; evidence must be counted in views, not pixels.

Known limits of that evidence: the two test chairs are the same model in different colours (hardest
case for DINO, easiest for colour); sim lighting is fixed, so the colour channel is optimistic for the
real robot; identical duplicates (worktops) are separable by no appearance cue — and the design must
then say so (log-ratio ≈ 0), not guess.

---

## 2. Architecture

New shared module `common/view_atlas/`:

| Unit | Responsibility | Depends on |
|---|---|---|
| `FrameHistory` | Ring of the last ~8 ZED BGR frames keyed by capture stamp, fed from the agent's existing `RgbIngestor::pump()`. `at(stamp)` returns the frame or nothing. ~20 MB. | OpenCV |
| `ViewEncoder` | (frame, mask pixels, bbox) → background-blanked square crop (1.15× bbox, 224², ImageNet-mean fill) → DINOv3 ViT-S/16 mean patch token (384-d, L2) + 16×4 H–S histogram over the mask pixels (sqrt, L2). ONNX Runtime, **CPU EP**, session created lazily on first use. | ONNX Runtime, OpenCV |
| `ViewAtlas` | Per-object state and math: update, predict at a bearing, log-likelihood ratio, value of encoding, shift marginalisation. **Header-only, pure Eigen** — unit-testable without models. | Eigen |
| `AtlasArchive` | Atlases of dead objects + last pose/cov + class + timestamps + owning agent; in memory and in `etc/atlas_<agent>_<id>.bin`. Versioned, agent-agnostic record so a future `LTObjectMemory` can absorb it unchanged. | — |

Per cycle in an agent (geometry is unchanged and comes first):

1. The tracker associates detections to instances by geometry, as today.
2. Each eligible detection gets a **value of encoding** (§4); those worth their price are deep-copied and
   queued to the encoder worker.
3. Drained results:
   - matched to a live instance → update that instance's atlas; if two instances compete for the
     detection, add the appearance log-ratio to the association cost;
   - belonging to a birth candidate → score against the archived atlases of the class (§3.4); the
     result is ready before the birth fires.
4. On death the instance's atlas moves to the archive.

Unchanged: retina, the graph contract (deaths delete nodes), `common/appearance_belief` (display tint).
A live atlas is **not** published to the graph in v1 — nothing reads it yet
(most published attributes already have no consumer).

---

## 3. Generative model

### 3.1 State per object (< 10 KB)

- **Shape:** 12 azimuth bins of 30° in the object frame. Bin b: unit mean `m_b ∈ R³⁸⁴`, effective count
  `n_b`, the last few view poses (azimuth, log-range) for the repeat-look discount, timestamp. Empty
  bins are normal.
- **Colour:** one prototype: unit mean `c ∈ R⁶⁴`, effective count `n_c`.

### 3.2 Observation and likelihood

An encoded view (q_d, q_c) at object-frame bearing β is reduced to two similarities:
`s_d = q_d · m̂(β)` against the shape predicted at β, and `s_c = q_c · c`.

The prediction at β is the kernel-weighted mean of populated bins,
`m̂(β) = normalise(Σ_b n_b κ(Δ_b) m_b)`, `κ(Δ) = exp(−Δ²/2ℓ²)`, and its **effective angular gap**
Δ(β) is taken from the same weights.

Two hypotheses, each a **2-D Gaussian over (s_d, s_c)** — one joint Gaussian, not two independent
channels, because both similarities come from one crop under one illumination and are correlated:

- **H_k (it is object k):** mean `(μ_d(Δ), μ_c)` with
  `μ_d(Δ) = μ_bg,d + (μ_0,d − μ_bg,d)·exp(−Δ²/2ℓ²)`, covariance `Σ_own + Σ_n / N(β)` (shrinks with
  evidence).
- **H_bg (another object of the same class):** fixed `(μ_bg, Σ_bg)` per class.

Score `Λ_k = log N((s_d,s_c); H_k) − log N((s_d,s_c); H_bg)`. As Δ grows, H_k relaxes into H_bg, so
**an unseen side contributes Λ → 0 by construction — no gate.**

### 3.3 Yaw: marginalise the cyclic shift

The object frame needs the object's yaw. The atlas is evaluated at all 12 cyclic shifts and the shifts
are **marginalised** with weights from the yaw posterior (log-sum-exp), never maximised:
well-fitted yaw → one shift dominates; symmetric shape (round table, square cabinet) → flat prior and
the atlas itself resolves yaw (as the place encoder resolves panorama yaw).

### 3.4 Re-identification at birth

Hypotheses {archived object 1…n of this class, new}. Log-weight of each:

`log prior + Λ (appearance) + log p(location)`

- `p(location | k)` = mixture of "stayed" (Gaussian at the archived pose and cov) and "moved"
  (uniform over the room), mixing weight π_move.
- Prior of "new" from the class birth rate.
- Decision = MAP (birth is a decision anyway). Posterior and its entropy are logged per birth.
- A resurrected instance keeps its archived atlas, its `object_uid` and (if free) its name.

### 3.5 Constants (fitted, never hand-set)

| Constant | Meaning | Source |
|---|---|---|
| μ_0,d, μ_bg,d, ℓ | shape similarity at Δ=0, background, angular length scale (expect ℓ ≈ 45–60°) | fit on `zed_orbits` |
| μ_c, Σ_own, Σ_bg, ρ(s_d,s_c) | colour mean, covariances, cross-channel correlation | fit on `zed_orbits` + `zed_frames` |
| Σ_n | per-view noise, with a term falling with mask pixel count | fit, binned by pixel count |
| σ²_illum | colour variance growth with brightness change obs↔prototype | **real robot only** — sim lighting is fixed |
| π_move | prior probability an archived object was moved | model constant, FLAGGED |
| N₀ | prior pseudo-count: "the prior is worth one view" (= 1) | model constant, FLAGGED |
| ℓ_r (repeat) | azimuth / log-range step below which a view is a repeat look | DERIVED: the step at which expected shape similarity drops by one per-view sd |

The fitting script writes them to a constants file with dataset provenance and fit date. Re-fit before
trusting them on new data.

---

## 4. When to encode

One quantity drives the trigger and the repeat-look discount: the predicted effective count near β,
`N(β) = N₀ + Σ_b n_b κ(Δ_b)`.

**Value of encoding** a detection, in nats, is the sum of:

1. **Learning:** `½ log((N(β) + w) / N(β))` — large on an unseen side, ≈ 0 for the thirtieth look
   from the same spot. `w` = view quality: `(1 − trunc_frac)` × predicted visible fraction
   (`common/occlusion`) × a resolution factor rising continuously with mask pixel count.
2. **Identity** (only when identity is in question: two live instances compete in one gate, or a birth
   candidate has same-class archived objects): expected entropy drop over those hypotheses, from the
   2-D Gaussians each candidate predicts **at this bearing**, by a few dozen samples (µs). Identical
   predictions (identical worktops) ⇒ value ≈ 0 ⇒ nothing is encoded for it.

Encode iff value > λ, the **price of one encoding**, with a rate cap (e.g. 4/s/agent).
★FLAGGED CONSTANT: λ is the cost of CPU time, which lies outside the generative model; pricing it is the
expected-free-energy trade of epistemic gain against action cost, so no model-level term can replace
it. The cap is a machine guard, not part of the model. Both live in config with this justification.

**Update:** `m_b ← normalise(n_b m_b + w q_d)`, `n_b += w (1 − κ_max)`, where κ_max is the kernel
between this view and the closest stored view pose of the bin (azimuth and log-range): repeat looks add
almost nothing (fractional counts must not manufacture confidence). Colour likewise with `n_c`.
**No constant process noise** (lifecycle §4.1): furniture does not change appearance; illumination is
an observation-variance term, not a decay of the belief.

**Eligible detections:** front ZED slices only (`MaskSlice::may_fit_geometry()`). Ricoh crops are out of
v1 (equirectangular distortion needs its own calibration). A detection whose frame has left
`FrameHistory` is skipped and counted.

**Threading:** the compute thread selects; selected crops are **deep-copied** `cv::Mat`s
(CLAUDE.md cross-thread rule) handed to one encoder worker per agent that owns the ONNX session; results
return on a queue drained by the compute thread, which alone touches atlases and the archive.

**Latency at birth:** a birth needs `birth_frames` (~5) observations; the candidate is encoded on its
first appearance, so re-ID is ready long before the birth fires and no birth is delayed.

**Diagnostics:** one CSV row per decision (encoded/skipped, value, dominant term, re-ID posterior +
entropy), per-run files + a report script.

---

## 5. Lifecycle conformance

| Stage | How the atlas conforms |
|---|---|
| CREATE | Births stay tracker-only. A resurrection **is** a birth, carrying an archived identity. |
| UPDATE | The atlas never moves geometry. Appearance enters only the association cost and the birth decision. |
| REMOVE | Unchanged: death deletes the node; the atlas goes to the archive. |
| HISTORY | No constant process noise (§4). |
| OWNERSHIP | Archive file keyed by `[Agent].id`; only the owner reads/writes it. |
| Identity | New typed attribute `object_uid` (agent-local persistent id). Needs a `REGISTER_TYPE` line in cortex's `dsr_attr_name.h` and **a cortex reinstall by the user**. |

`object_uid` is also the key the episodic memory needs (`docs/superpowers/specs/2026-10-17-episodic-memory-design.md`
snapshots object beliefs keyed by object node): without it a resurrected object reads there as a new one.
That archive is the intended home of the future long-term object memory; the per-agent atlas archive must
not grow into a second long-term store.

**Widening appearance beyond display-only is deliberate.** `appearance_belief.h` warns that colour
evidence from thousands of correlated pixels would out-precision geometry. The atlas avoids that by
counting evidence in **views, not pixels** (one similarity pair per encoded view, repeat-discounted) and
consulting it only when identity is in question. This reasoning is to be cross-referenced in that
header when the atlas lands.

---

## 6. Testing and rollout

Testing, each layer before the next:

1. **Unit tests** (`common/run_tests.sh`): N(β), update + repeat discount, Λ, shift marginalisation,
   re-ID posterior, archive round-trip. The archive test calls `setlocale(LC_ALL, "")` first so it runs
   under `es_ES` like the agents; archive I/O uses `std::from_chars` / `imbue(std::locale::classic())`.
2. **Encoder parity:** C++ ONNX vs the Python reference on identical crops, cosine ≥ 0.999. Export with
   `retina/tools/export_dinov3_crop.py`, ViT-S/16 (not S+: ORT's optimiser corrupted the S+ export). If parity fails, suspect graph optimisation, then the resize
   filter.
3. **Offline replay through the module itself** (a C++ harness linking `common/view_atlas`, not a Python
   re-implementation: a bench that re-implements what it tests once scored rooms the agent got wrong) on `zed_frames` + `zed_orbits` with floor-ray
   labels: moved-chair re-ID, swap rate, encodings/object/minute, re-ID posterior calibration (reliability
   diagram). The constant-fitting script runs on the same data.
4. **Live sim, chair_concept.** Pass criteria, declared now:
   - the yellow chair moved to another room while out of view (supervisor script; `DEF CHAIR_Y` added to
     `piso.wbt`) ⇒ **no new chair node; the old `object_uid` returns**;
   - the two chairs never swap over a full tour;
   - chair_concept CPU increase within the λ/cap budget (measured, reported).

Rollout, `[ViewAtlas] enabled = false` by default:

1. Module + tests + export + constants + offline harness. No agent changes.
2. chair_concept **shadow mode**: compute and log every re-ID decision next to what the tracker did; do
   not act.
3. chair_concept acting (resurrection + association term); then table, refrigerator, cabinet, one at a
   time, each with its §6 audit row.
4. Later: `LTObjectMemory` absorbs the per-agent archives.

---

## 7. Out of scope for v1

Ricoh crops; publishing live atlases to the graph; cross-agent / cross-class re-ID; elevation bins (the
robot's camera height is fixed; the arm could add them later); per-surface-point feature clouds (the
alternative kept in reserve if per-view pooling turns out to be the bottleneck).
No decoder (latent → image): the model predicts in descriptor space, which is the observation the
likelihood scores; a pixel decoder would add a trained model and score lighting and background, which
the descriptors were chosen to ignore. A per-bin stored thumbnail covers the display/debug use.

## 8. Phase 1 results (offline replay, 2026-10-17)

Dataset: `retina/etc/view_atlas_dataset` (2715 views, 14 physical objects), constants `view_atlas_constants.txt`.
Criteria were declared before the run (§6). **Verdict: A, B, C, D all FAIL.** The criteria and constants were not
touched to obtain a result. The primary table below is the corrected (repeat-discounted, §8.1) run, which is what
`atlas_replay` prints now (exit code 1); the first, undiscounted run is kept further down as superseded.

| Check | Criterion | Result (current code) |
|---|---|---|
| A. chair swaps | < 1 % | 7 / 518 = 1.35 % (FAIL) |
| B. re-ID, yellow moved / magenta stayed (yaw known) | >= 95 % each, other chair <= 2 % | yellow (`Chair`, moved) 15 / 53 = 28.3 %, magenta (`chair(1)`, stayed) 20 / 20 = 100.0 %; other chair chosen 0.0 % (FAIL) |
| B (info, yaw unknown, not graded) | none | yellow 21 / 53 = 39.6 %, magenta 20 / 20 = 100.0 %, other chair 0.0 % |
| C. encodings per object-minute | <= 60 (expect <= 30) | worst 1000.0 /min (FAIL), but NOT MEASURABLE on this data: see below |
| D. calibration | +-0.1 per bin with >= 20 decisions | [0.5,0.7) n=20 mean p 0.598 acc 0.300 (gap 0.298); [0.7,0.9) n=21 mean p 0.779 acc 0.333 (gap 0.446); [0.9,1.0] n=32 mean p 0.983 acc 0.688 (gap 0.295) (FAIL, all three bins overconfident) |

Superseded: the first run, **as run, superseded by the repeat-discount fix** (§8.1; it summed the five per-frame
log-ratios of a window raw):

| Check | As run (undiscounted, superseded) |
|---|---|
| B, yaw known | yellow 37 / 53 = 69.8 %, magenta 20 / 20 = 100.0 %, other chair 0.0 % |
| B info, yaw unknown | yellow 43 / 53 = 81.1 %, magenta 20 / 20, other 0.0 % |
| D | [0.5,0.7) n=4, p 0.627, acc 0.250; [0.7,0.9) n=5, p 0.828, acc 0.400; [0.9,1.0] n=64, p 0.996, acc 0.844 (gap 0.152; only this bin has >= 20 decisions) |

A and C do not use windows and did not change.

Reading of the results (not a change of criteria):

- B: the yellow windows that failed were never assigned to the *other chair* (0.0 %); the MAP was the "new object"
  hypothesis, i.e. the moved chair was not recognised as the archived one.
- C: the `frames` recording is a sparse sampling of an apartment tour (267 distinct stamps, 647 object views, spread over
  ~11 days of wall clock with one 868 000 s gap). First-to-last stamp therefore gives ~16 000 min per object and
  ~0 encodings/min, and the single-view object `Worktop4` gives 1 encoding / 0 min (clamped to 1e-3 min = 1000 /min).
  Neither is a real encoding rate; the line fails by arithmetic of the sparse stamps. The informative number is that the
  trigger encoded nearly every sparse view (e.g. 87 of 93 `Chair`, 111 of 146 `chair(1)`, all of the other objects):
  consecutive sparse views are far apart in bearing, so each carries high learning value. A continuous recording is
  needed to grade C.

**Harness assumptions that shape B and D** (hand-set in `atlas_replay.cpp`; spec §3.4 says the "new" prior should come
from the class birth rate):

- priors: new = 0.5, each chair = 0.25; `pose_cov` = (5 cm)^2; room area 60 m^2. With zero appearance evidence these give
  p(new) ~ 0.83, so yellow windows on an unseen side become confident "new" answers.
- magenta's 20 / 20 is carried by LOCATION (its position is the exact ground-truth centroid, about a 7.6-nat margin) and
  says nothing about appearance.
- windows are disjoint blocks of 5 consecutive views, not every 5-view window as the criterion text says.
- the fitted repeat scale is ill-conditioned: `fit_constants.py` derives `repeat_ell_rad` from sd/drop = 0.786 (>= 0.5
  warns; the value is recorded in the constants header). It is a derived number, not a robust fit.

### 8.1 Fix round 1: window evidence counted in views (corrected B and D)

The first run (the superseded table in §8) summed the five per-frame log-ratios of a window raw. Evidence counts in views and
repeat looks add about nothing (§4, §5), and the five frames of a window are mostly repeats, so that run counted the same
look about five times. The module now has `pose_kernel` (the repeat kernel `update()` uses) and `log_ratio_window`, which
discounts each view by its redundancy with the window's earlier views; the harness B and D use it. Only B and D change
(A and C do not use windows). The two runs differ for that reason alone; criteria, constants and dataset are unchanged.

The task review's own recomputation gave 26/53 for the yellow chair; the harness prints 15/53. The log-range term of the
kernel is inert here (all orbit ranges are 1.0). The difference comes from the fitted `repeat_ell_rad` = 0.758 rad (43 deg)
against a 5 deg frame spacing, which makes a window of 5 views count as about 1 view.

OPEN FINDING (measured; leading candidate: coverage). With the yaw labels given, the moved chair is not recognised. The
argmax over the 12 shifts of its lap-1 log-ratio peaks at shifts 9-10 (54 and 62 of 268 views; mean log-ratio 1.78 / 1.83
against 1.21 at shift 0), i.e. 60-90 degrees off the labelled yaw, whereas `chair(1)` (stayed) peaks at shift 0 (59 of 103
views). The yaw labels were verified against `zed_orbit_recorder.py:84` and `piso_orbits.wbt:1566`.

Measured bearing histograms (views per 12 bins of 30 deg, from -180):

| | bins |
|---|---|
| Chair lap 0 (archive) | 1, 3, 5, 12, 18, 56, 116, 37, 2, 0, 0, 0 |
| Chair lap 1 (test) | 26, 9, 7, 7, 12, 25, 27, 23, 25, 36, 40, 31 |
| chair(1) lap 0 | 43, 47, 135, 26, 7, 0, 1, 4, 0, 0, 3, 4 |
| chair(1) lap 1 | 40, 22, 8, 12, 6, 0, 6, 3, 0, 0, 2, 4 |

About 62 % of the moved chair's lap-1 views fall on sides the archive barely or never saw, where the information weight Lambda goes
to 0 by design (§3.2). Shifts 9-10 map those views onto the populated bins, which is where the shift argmax peaked.
chair(1)'s lap-1 views fall on bins already seen in lap 0 (peak at shift 0).

Gap-band table (yaw known, informative, not graded): window gap = angular distance from the first view's object-frame
bearing to the nearest populated bin of the correct chair's archive atlas.

| gap band | Chair (moved): n, re-ID | chair(1) (stayed): n, re-ID |
|---|---|---|
| [0,15) deg | 26, 46.2 % | 15, 100.0 % |
| [15,45) deg | 14, 14.3 % | 5, 100.0 % |
| [45,90) deg | 13, 7.7 % | 0 |
| [90,180] deg | 0 | 0 |

Decisions per posterior band by window colour (info): [0.5,0.7) yellow 20 / magenta 0; [0.7,0.9) yellow 21 / 0;
[0.9,1.0] yellow 12 / magenta 20.

Re-ID accuracy of the moved chair falls monotonically with the gap (46 %, 14 %, 8 %), so the cause is most likely
coverage, consistent with the gap-band table. Even in the smallest band the moved chair reaches only 46 %, so coverage
alone does not explain every miss; that remainder is not explained.

## 9. Before phase 2 (open)

(a) **The live mask producer is decimated.** `retina/src/graph_publisher.cpp` pushes every second pixel
    (`VOXEL_DECIMATION_FACTOR = 2`, `retina_params.h`), so `make_crop` on live masks blanks about 3/4 of the object into a
    lattice and `mask_px` / `trunc_frac` come out about 4x wrong. The dataset used dense masks. This must be settled
    (stride-aware dilation/closing, or a stride-1 publication) and pinned by a golden crop test against the Python mirror
    before any live run.
(b) **H_k covariance.** The code blends cov_bg to cov_own by R with Sigma_n only as view-quality noise, the fitter sets
    cov_n = cov_own, and the effective gap is the nearest populated bin. Sections 3.2 and 3.5 say Sigma_own + Sigma_n / N(beta),
    with Sigma_n fitted per pixel-count bin and the gap from kernel weights. The USER must rule whether to change the spec
    or the code.
(c) **Repeat check scope.** It looks only at the current bin's poses; with repeat_ell 43 deg > bin width 30 deg this biases
    counts at bin edges.
(d) **The archive save is non-atomic.** Fix before any live writer.
