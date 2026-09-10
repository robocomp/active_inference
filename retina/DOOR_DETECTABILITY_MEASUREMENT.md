# Door detectability: what the perception stack actually knows

**2026-09-04. Step 1 of the plan. Offline, no robot, no agent touched.**

---

## DECISION (2026-09-04)

**Build the graded ADE20K output. Do not wire a second detector channel yet.**

Why, in order of weight:

1. **The graded output is the better-matched instrument for the actual failure.** The failure is
   that silence is charged as confident absence. `P(door)` measures how decisively the classifier
   resolved the question — 0.979 where it found a door, 0.636 on silences that really do contain
   one, 0.315 on silences that do not. The specialist cannot supply that number at all; it supplies
   another vote.
2. **One channel keeps `p_vis` semantics unchanged.** ADE20K runs every frame, so nothing in
   Appendix C applies: no frame-level "which channels looked" attribute, no per-channel `p_detect`,
   no channel-keyed field, no intermittency.
3. **It fixes four agents, not one.** `conf ≡ 1.000` today for door, hood, cabinet and shelf.
4. **It is reversible.** `output0` is byte-identical; reverting is pointing at the old model file.
5. **It strands nothing.** The measured decisiveness covariate is the prior the learnt p_detect
   `ViewField` wants, and `models/doors/doors_yolo11s_end2end.onnx` is already built and verified
   if step C is ever needed.

**What would reverse this decision:** step B shows real doors still dying after the graded channel
and the `p_detect` fix. Then add the specialist — with the frame-level provenance of Appendix C,
which is non-negotiable for any channel running below 100 % duty.

**Not blocking:** the specialist still rescues 124 silences at `t=0.20` that the posterior misses.
That is a reason to keep step C available, not a reason to do it first.

---

Frames: `retina/etc/depth_frames/` — 1108 ZED JPEGs spanning **40.9 min** of one apartment run at
~4.8 Hz. Measured on **every 4th frame (n=277)**. ⚠They are a single run, so 277 frames are *not*
277 independent observations; treat percentages as describing this tour, not the apartment.

Tools (new): `tools/expose_semantic_logits.py`, `tools/measure_door_detectability.py`.

---

## The question

`door_concept`'s only removal channel is the ZED silhouette, and the only thing that keeps a door
alive is a mask labelled `door`. `door` is not a COCO class, so it comes solely from the ADE20K
semantic path — whose ONNX export collapses a 150-way softmax to a `uint8` argmax **inside the
graph**:

```
Conv classifier.1 → [1,150,80,80] logits → Resize → [1,150,640,640] → ArgMax → Cast → uint8 [1,640,640]
```

So a door that loses *narrowly* to `wall` (ADE20K class 0, the most frequent class in the set) is
emitted identically to a door the model never saw. Two candidate cures, opposite work:

- **(a) narrow loss** → restore the posterior, consume `P(door|pixel)` as a continuous weight.
- **(b) genuinely blind** → no belief change helps; use a different model. One already exists:
  `door_learner/`, YOLO26x fine-tuned on three merged door datasets, mAP50 0.913 **on its own
  held-out split — not this apartment**.

Both were measured on the same frames.

---

## Result 1 — the posterior is recoverable for 77 kB/frame

`tools/expose_semantic_logits.py` appends `Softmax(axis=1) → Gather(accepted ids) → Cast(fp16)` on
the **logit tensor**, and registers it as a second output. Pure graph surgery with the `onnx`
package on the file already in production: no `.pt` (not on disk), no ultralytics, no retraining.

- Tap is at the classifier's native **80×80, above the Resize** — a full `[1,150,640,640]` volume
  would be 245 MB/frame. 5 classes at 80×80 fp16 = **64 kB/frame**.
- `output0` verified **byte-identical**. Inference cost 0.25 s → 0.26 s (CPU).
- **The cheap tap is faithful**: against a reference model tapping the argmax's own 640×640 input,
  on 25 argmax-misses-door frames — **r = 0.972**, median paired difference **+0.0009**,
  max |Δ| 0.080. The softmax/upsample order does not commute, but the error is negligible.

## Result 2 — doors are losing coin flips, not being unseen

On the 98 frames where the current pipeline produces **no door mask at all**, at the pixel with the
highest `P(door)` among those the model labelled `wall` — **paired, same pixel**:

| statistic | value |
|---|---|
| median peak P(door) | 0.378 |
| median P(wall) *at that pixel* | 0.443 |
| **median margin P(wall) − P(door)** | **0.0175** |
| frames where door was runner-up-close (margin < 0.10) | **67.3 %** |
| frames where wall won by a landslide (margin > 0.50) | 8.2 % |

At the true 640 posterior the margin is tighter still: median **0.0093**, 76 % within 0.10.

**A coin flip is being reported as certainty.** The belief layer then reads the resulting silence as
confident absence, because `door_concept`'s `p_detect` is purely geometric
(`resolvability × in_fov_frac × central_frac`) and says ≈1 for a door squarely in view.

⚠ First cut of this table quoted `max P(door)` and `max P(wall)` over the same *region* — different
*pixels*, inventing a margin no pixel has (it read P(wall)=0.9911 and looked like a landslide).
Fixed to a paired read before drawing any conclusion.

## Result 3 — but the posterior alone is not a clean door channel

Replaying `SemanticMaskStage` exactly (morph open+close k=5, connected components, drop below
`mask_min_area_frac = 0.003` of the pixels *looked at*), on a `P(door) ≥ t` sweep:

| t | frames with a door mask | of which argmax had none |
|---|---|---|
| — (current, argmax) | 178 / 277 (64.3 %) | — |
| 0.10 | 248 (89.5 %) | 70 (25.3 %) |
| 0.20 | 219 (79.1 %) | 41 (14.8 %) |
| 0.30 | 194 (70.0 %) | 16 (5.8 %) |

Cross-checked against the fine-tuned detector (conf ≥ 0.25) as an independent referee:

```
both agree door present : 156        detector YES, argmax NO :  66   ← the blindness set (24 %)
both agree none         :  33        argmax YES, detector NO :  22
agreement 68.2 %
```

At `t = 0.20` the posterior recovers **51.5 %** of the blindness set — but also fires on **21.2 %**
of frames the detector calls empty. The separation is real but overlapping: peak `P(door)` median
**0.622** where the detector confirms a door vs **0.308** where it does not.

⇒ **As a mask threshold that is a poor trade. As a continuous weight in a likelihood it is fine** —
`P(door) = 0.3` should contribute weak evidence, not a mask. That is the design intent anyway, and
this is the measurement that says the hard-cutoff version must not be built.

## Result 4 — the door model we already trained is simply better here

| channel | frames with a door | note |
|---|---|---|
| ADE20K argmax (live today) | 178 / 277 — **64.3 %** | |
| `doors_yolo11` (36 MB) | 198 / 277 — 71.5 % | raw `[1,7,8400]`, needs NMS |
| **`doors_yolo26` (212 MB, end2end)** | **222 / 277 — 80.1 %** | `[1,300,6]`, the layout `parse_detections_end2end` already reads |

405 detections over 222 frames (1.8/frame), essentially all `open_door` (404) — consistent with an
apartment toured with its doors open. **Confidences are modest: median 0.403, mean 0.431, p90
0.648.** So even the purpose-built model is often only ~40 % sure, which is an argument for
publishing the score as a likelihood rather than passing it through a 0.25 gate.

The two channels are **complementary, not ordered**: 22 frames go the other way. Union ≈ 88 %.

---

## Conclusion

**(a) and (b) are both true, and they answer different halves.**

1. The narrow-loss hypothesis is **confirmed** — median margin 0.0175, 67 % of misses within 0.10.
   The information exists and costs 77 kB/frame to recover. But it is a **weak, overlapping**
   signal, so it belongs in the likelihood as a continuous weight, never as a mask threshold.
2. The **fine-tuned door detector is the stronger channel** (80.1 % vs 64.3 %) and it is already
   trained, already exported, and matches a parser retina already has. Its one gap is that it emits
   **boxes, not masks**, and `door_concept` consumes `mask_pixels_xy`.

Recommended: do **both**, in this order — wire the door detector (bigger, cheaper win; a `-seg`
fine-tune on the same merged dataset closes the mask gap), and ship the posterior tap as the
semantic path's real `SemanticMap::scores`, which also fixes `conf ≡ 1.000` for hood/cabinet/shelf
and makes the dead `Semantic.conf_thresh` knob meaningful.

Neither is a threshold change, and neither is worth anything until `door_concept` stops treating
geometric detectability as semantic detectability — see Steps 3–5 of the plan.

---

# Appendix — calling the specialist selectively (measured, full rate n=1108)

**The mechanism already exists.** `rc::Stage::set_enabled()` is atomic and documented "a disabled
stage is skipped entirely, so the model does NO work"; each Stage owns its own ONNX session;
`PerceptionWorker::stage(name)` returns the handle; `sync_ricoh_depth_stage()`
(`specificworker.cpp:2055`) is the precedent for toggling one per-cycle from the main thread.
⚠`set_enabled(false)` skips `run()` but does NOT free the session — this buys throughput, not VRAM.

## Which trigger

| trigger | duty cycle | blindness set caught |
|---|---|---|
| `P(door) ≥ 0.20` from the cheap posterior | 90.3 % | 74 % |
| `P(door) ≥ 0.50` | 82.3 % | 58 % |
| **cheap channel produced no `door` mask** | **36.8 %** | **100 %, by construction** |

The posterior is a useless *gate* — some pixel almost always clears a low bar. Anding it with the
silence condition gains nothing (35.4 % vs 35.7 % at stride 4), so it is one fewer knob.

**The cascade argument is structural, not empirical: when the cheap channel already yields a door
mask, the door lives anyway.** The specialist can only ever change an outcome by refuting a
silence. Spend inference exactly where a null result would otherwise be charged as absence.

## Most silences are harmless; a few carry all the risk

54 silent runs over the 40.9 min tour:

| | frames | seconds |
|---|---|---|
| median | 2 | **0.23** |
| p75 | 5 | 1.34 |
| p90 | 26 | 5.98 |
| max | 57 | 13.2 |

**12 of 54 runs exceed 1.6 s (door_concept's ~15-observation removal budget) and those cover 78 %
of all silent frames.** So the distribution is bimodal: momentary flicker that the belief barely
feels, plus a few long silences that are the whole danger. (The flicker half is also what Step 4's
viewpoint-novelty damping neutralises — two mechanisms, opposite ends of the same problem.)

## Therefore: decimate inside the silence

| 1-in-N silent frames | duty of all frames | worst latency to first call in a run |
|---|---|---|
| 1 | 36.8 % | 0.00 s |
| 2 | 20.1 % | 0.21 s |
| **4** | **11.9 %** | **0.62 s** |
| 6 | 9.4 % | 1.04 s |
| 12 | 6.7 % | 2.29 s |

Every silent run still gets at least one call at any N (a run shorter than N still gets its first
frame); what N costs is *latency*, and the budget is the ~1.6 s it takes a silence to become a
removal. ⚠N is expressed in frames of THIS 4.8 Hz recording — the transferable statement is
**"call it at roughly 1.5–2 Hz while the cheap channel is silent about doors"**, ≈**10–12 % duty**,
an 8–9× saving over always-on with the first call inside 0.6 s.

⚠**Per-call cost on the real box is NOT measured here.** 0.28 s/frame is CPU-only
(`CPUExecutionProvider`, and partly under self-contention); retina runs TRT/CUDA. The duty cycle is
the transferable number; the millisecond cost has to be measured with the real EP before deciding
between `yolo26x` (212 MB) and `yolo11` (36 MB, needs an end2end re-export).

## The honest weakness

This trigger's duty cycle is **highest exactly where it is least useful**. In this tour 33 of 99
calls (33 %) fired where no door existed at all; in a door-free room it would fire constantly and
find nothing. The guard is a relevance condition — a `door_*` node or a room-polygon aperture within
sensor range — which is also where `ViewField::confidence()` (plan Step 5) eventually feeds in.

## Method note

Two earlier "full-rate" background runs silently did not execute (`nohup … &` returned immediately,
the harness reported completion, the JSON still held stride-4 data) and a third was killed by
self-contention at load average 77 on 32 cores. The numbers above are from a single bounded-thread
foreground pass, verified by `n=1108` in the JSON. ★A completion notification is not evidence that
the work happened — check the artefact's own row count.

---

# Appendix B — the low-memory door model (full rate, n=1108)

## It did not need creating from YOLO26 — it was already trained

YOLO11 and YOLO26 are different architectures; you cannot derive one's weights from the other
(that would be distillation, i.e. a training job). But `door_learner` trained **both** on the same
merged dataset, and on their own held-out split they are effectively tied:

| model | params | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| **YOLO11s** | **9.4 M** | **0.9220** | 0.8876 | 0.9319 | 0.8876 |
| YOLO26x | 58.8 M | 0.9279 | 0.8937 | 0.9462 | 0.8915 |

0.6 pp apart at **6.2× fewer parameters**. What YOLO11s lacked was only the export format: its
stock ONNX is the raw `[1,7,8400]` head, which would need an NMS implementation on our side.

**Produced: `retina/models/doors/doors_yolo11s_end2end.onnx`** (36 MB) — re-exported with
`nms=True` → `[1,300,6]`. Verified against the yolo26x export on a live frame: identical
semantics (`x1,y1,x2,y2,conf,cls`, 640-letterbox pixels, confidence-sorted, 300 rows), so it is a
drop-in for the layout `YoloSegDetector::parse_detections_end2end` already reads.

## On this apartment, at full rate

| channel | frames with a door | cost/frame (CPU) |
|---|---|---|
| ADE20K argmax (live today) | 700 / 1108 — 63.2 % | — |
| **YOLO11s end2end (36 MB)** | 775 — **69.9 %** | **0.03 s** |
| YOLO26x end2end (212 MB) | 864 — **78.0 %** | 0.28 s |
| union of the two | 980 — 88.4 % | — |

★**The held-out tie does not survive the domain shift**: 0.6 pp apart on the public split, **8.1 pp
apart here**. The apartment is the test that matters, and the big model generalises to it better.

## But on the job the specialist is actually for, the gap nearly closes

The cascade only ever runs on the 408 frames (36.8 %) where the cheap channel is silent:

| | silences rescued |
|---|---|
| YOLO26x | 264 / 408 — 64.7 % |
| **YOLO11s** | 240 / 408 — **58.8 %** |
| union | 329 / 408 — 80.6 % |

**YOLO11s recovers 90.9 % of what YOLO26x recovers, at 1/9 the compute and 1/6 the size.**

Amortised over the ~11.9 % duty cycle of Appendix A: **YOLO11s ≈ 3.6 ms/frame vs YOLO26x
≈ 33 ms/frame** (CPU; the TRT/CUDA numbers still have to be measured on the real box).

⇒ **Recommend YOLO11s for the specialist slot.**

## ⚠ The two detectors agree on only 71 % of frames

205 frames only YOLO26x sees; 116 only YOLO11s sees. They are not nested — the small model is not
a degraded copy of the big one. Two models, same training data, near-identical held-out mAP,
disagreeing on 29 % of frames means **both are sitting near their decision boundary on this
domain**, which is also what their modest confidences say (median 0.40–0.41 for both).

Do **not** read the 88.4 % union as free accuracy: running both defeats the point of a low-memory
slot. Read the disagreement as the same "narrow loss" finding one level up — and as one more
argument for publishing the detector's score as a **likelihood** rather than passing it through a
0.25 gate.

## Wiring constraint found

`YoloSegDetector::detect()` requires **≥2 output tensors** (detections + mask protos) and errors out
otherwise (`yolo_seg_detector.cpp:354-357`). Both door models emit ONE. So hosting a detect-task
model is a small sibling class (a `YoloBoxDetector`, no mask decode) rather than a patch — which is
also the honest option, because `door_concept` consumes `mask_pixels_xy` and a box is not a mask.
Shipping empty `cv::Mat` masks through the existing path would hide that gap instead of naming it.

## Files touched outside retina

`door_learner/runs/train/doors_yolo11/weights/best.onnx` was **overwritten** by the end2end export,
then **restored** by re-exporting with the original settings — output shape back to `[1,7,8400]`.
⚠It is not byte-identical to the February file: onnxslim is 0.1.94 here vs 0.1.80 then. Retina's
end2end copy lives under `retina/models/doors/`, so the training repo owns only its own export.

---

# Appendix C — what wiring a `YoloBoxDetector` stage to the silence cascade implies

## 1. The cascade is INTRA-FRAME. `set_enabled()` is the wrong mechanism.

Correcting Appendix A: `zed_stages` is one vector run **in sequence on one worker thread for one
frame** — `SegStage → PoseStage → SemanticStage → SemanticMaskStage → Sam2Stage → DepthStage` —
and cross-stage dependence is already the established pattern: `SemanticMaskStage::run()` reads
`out.semantic` (a previous stage's slot, `semantic_mask_stage.cpp:32-35`) and appends into
`out.masks` (`:159-162`); `Sam2Stage` reads `out.masks` (`sam2_stage.cpp:120`).

So a `DoorStage` placed **after** `SemanticMaskStage` reads `out.masks`, asks "did any `door` mask
land this frame?", and fires or skips **inside its own `run()`** — zero latency, no cross-thread
toggle, no frame lag. `set_enabled()` stays what it is: the user/viewer toggle.

## 2. Trigger on "the cheap channel RAN and said nothing", not "no door mask"

`SemanticMaskStage` returns early unless `out.semantic_fresh` (`:32`) — on a decimated cycle it
produces no masks at all, which is indistinguishable from silence if you only test `out.masks`.
`Semantic.decimation = 1` today so the two coincide; **raise it and the specialist would fire on
every skipped frame.** The condition is `out.semantic_fresh and no door mask`.

## 3. Placement also solves box-vs-mask, for free

`Sam2Stage` iterates `*out.masks`, filters by `refine_labels_`, and prompts with `d.bbox`
(`sam2_stage.cpp:120-153`). A door box appended before it becomes a **real silhouette**, which is
what `door_concept` consumes (`mask_pixels_xy`). SAM2 is currently OFF because it was 52 % of the
frame (49.9 ms of 95.3 ms) — but that was every frame, every object. Here it is **1–2 boxes on
~12 % of frames**, i.e. ~7.1 ms occasionally. That is a different budget question, and a much
smaller one.

## 4. ★THE ONE THAT BITES: the cascade changes what a MISSING mask MEANS

Today a frame with no `door` mask is charged as absence, weighted by a **geometric** `p_detect`
that reads ≈1 face-on (`door_concept/src/specificworker.cpp:1661`). If the specialist runs on ~12 %
of frames, then on the other 88 % there is still no specialist mask — and `door_concept` **cannot
tell "it looked and saw nothing" from "it was never run."**

Charging both as absence makes blindness **worse than before**: you would have added a better
detector and then let its silence count as evidence eight times more often than it was actually
consulted. This is the same defect shape as [[mask-source-not-has-depth]] — a guard on a proxy.

**`mask_source` cannot fix it.** It is a per-mask float (0.0 = zed, 1.0 = ricoh,
`graph_publisher.cpp:437`), and a per-mask field cannot express "ran and found nothing" — there is
no mask to attach it to. What is needed is a **frame-level** attribute on the `masks` node saying
which specialists were consulted this frame.

Given that, no new threshold is required: `p_vis = 0` on frames the channel did not run, and
`ΔL = p_vis · log-ratio` yields HOLD for free — the existing rule, doing exactly what it was
written to do.

## 5. `p_detect` stops being one number

The door then has **two camera sub-channels with different reliabilities and different duty
cycles** — ADE20K (always on, 63.2 % here) and the specialist (rare, 69.9 %). They cannot be summed
as if one sensor. The specialist's `p_detect` = geometric detectability × *did it run this frame*.

That is also what makes the learnt p_detect `ViewField` (plan Step 3) well-posed: a Bernoulli trial
is only recorded on frames where the channel was actually consulted.

## 6. The learnt field must be keyed BY CHANNEL, never pooled under label "door"

The specialist's trials are conditioned on the cheap channel being silent — a biased sample **by
construction**. That is fine for estimating "P(specialist fires | it ran, this cell, this bearing)",
which is what `p_detect` means. It is invalid for estimating the cheap channel's rate. Two channels,
two keys.

## 7. Births are ENABLED, not gated — the opposite of the p_FA rule

A specialist firing during a silence is precisely a door nobody had. Note `rc::birth::evidence`
weights an observation by `confidence × range decay`, and the specialist's confidences are modest
(median 0.404), so births will be **slow**. That is correct behaviour, not a bug to tune away.

## 8. Intermittency is a new signal shape

The `door` label would appear at ~1–2 Hz instead of continuously, so `frames_since_detection == 0`
is true only on specialist frames. Any consumer treating mask presence as continuous sees chatter.
The 360 viewer hit exactly this and the cure generalises: **the staleness bound is the specialist's
own call period**, not a tuned constant.

## 9. What it does not save

The cheap channel still runs every frame — it *is* the trigger — plus the surgical probs output.
The cascade saves only the specialist's own inference. The saving is real (≈3.6 ms/frame amortised
for YOLO11s vs 30 ms always-on) but it is not a whole-pipeline saving.

---

# Appendix D — sequencing: the graded output first

## The decisive statistic is not mask recovery, it is the ORDERING

Peak `P(door)` on the frames, grouped by what is actually there (specialist as referee):

| frames | median peak P(door) | n |
|---|---|---|
| cheap channel FOUND a door | **0.979** | 700 |
| silent, but the specialist CONFIRMS a door | **0.636** | 264 |
| silent, and the specialist DENIES a door | **0.315** | 144 |

**A properly ordered three-level signal that the argmax destroys completely.** The argmax emits one
bit: it collapses 0.979 and 0.636 into *different* answers though both are doors, and collapses
0.636 and 0.315 into the *same* answer though one is a door and one is not.

⇒ The graded output's primary value is **not** recovering masks. It is a **precision signal**:
"how decisively did the classifier resolve this question, here, from here?" That is exactly the
covariate `door_concept`'s absence channel is missing, and it separates confirmed-door silences
from genuinely-empty ones by **2×**.

## Which is why it needs no mask, no threshold, and no second channel

On a silent frame the agent already projects the door silhouette. Reading `max P(door)` **over that
silhouette** gives a measured `p_vis`:

- high (classifier ambivalent, ~0.44 vs ~0.44) ⇒ the look did **not** resolve the question ⇒ p_vis
  low ⇒ HOLD;
- low (classifier decisive that nothing is there) ⇒ charge absence.

That is `ΔL = p_vis · log-ratio` doing precisely what it was written to do, with a **measured**
p_vis instead of the geometric proxy that reads ≈1 face-on. No cutoff, no `if`.

## As a mask threshold it is only comparable to the specialist, not better

Of the 408 silences (full rate):

| rescuer | rescued | covers what YOLO26x rescues |
|---|---|---|
| YOLO26x specialist | 264 (64.7 %) | — |
| YOLO11s specialist | 240 (58.8 %) | — |
| posterior ≥ 0.10 | 298 (73.0 %) | 81.1 % |
| posterior ≥ 0.20 | 180 (44.1 %) | 53.0 % |

⚠The `≥0.10` row beats the specialist on count, but 84 of those 298 are frames the specialist
denies, and the threshold was chosen after seeing the table. **Do not read this as "the posterior
wins"** — read the ordering statistic above instead. This table is why the mask-threshold use was
already ruled out in the main report.

## Recommended sequencing

**A. Ship the graded output alone.** Model surgery + the third branch in `yolo_semantic.cpp` +
`p_detect` damping from classifier decisiveness. **One channel.** Every problem in Appendix C
disappears: the ADE20K stage already runs every frame, so `p_vis` semantics are unchanged, and no
frame-level provenance attribute, no per-channel `p_detect`, no channel-keyed field, no
intermittency. It also fixes `conf ≡ 1.000` for **hood, cabinet and shelf**, not just door, and
makes the dead `Semantic.conf_thresh` meaningful. `output0` is byte-identical, so today's behaviour
is a strict subset — the change is reversible by pointing at the old model file.

**B. Measure whether the door stops dying** (`door_events.csv` BIRTH/REMOVE, both directions).

**C. Add the specialist only if B is insufficient** — with Appendix C's frame-level "which channels
looked" requirement, which is non-negotiable once a channel runs at less than 100 % duty.

Bonus: A produces a *measured* detectability covariate, which is exactly the prior the learnt
p_detect `ViewField` (plan Step 3) wants. They compose; doing A first does not strand any of it.

## ⚠ The honest limit of this evidence

1. The 0.979 / 0.636 / 0.315 split is a frame-level max over **all** argmax-`wall` pixels, not over
   the door's projected silhouette. The agent would read it at the door's location — which should be
   *cleaner*, but I cannot verify that offline without door poses. The real separation may differ.
2. The specialist is a **referee, not ground truth** — it is itself 69.9–78.0 %, with only 71 %
   inter-model agreement. The 0.315 group certainly contains real doors both models missed.
3. Deferring the specialist is not free: at `t=0.20` it still rescues 124 silences the posterior
   does not (50 at `t=0.10`). A defers it; it does not refute it.
