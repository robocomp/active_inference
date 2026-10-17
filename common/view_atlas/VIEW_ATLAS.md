# View Atlas — per-object appearance memory for visual re-identification

Design spec, 2026-10-17. Status: **APPROVED DESIGN, NOT BUILT.** Nothing in this directory compiles yet.

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
| N₀ | prior pseudo-count (class background) | from Σ_bg |

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
