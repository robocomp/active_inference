# Forward posterior channel — the one step that needs your cortex reinstall

The graded semantic path is built and verified in retina (`yolo_semantic.{h,cpp}`,
`models/yolo26/yolo26l-sem-ade20k-probs.onnx`). Getting it to `door_concept` splits into two
halves. **Half 1 needs nothing from you but a config pointer. Half 2 needs four lines in cortex.**

---

## Half 1 — real mask confidence. UNBLOCKED, works the moment the config points at the graded model.

`SemanticMaskStage` already scores each component as the mean of `SemanticMap::scores` over its
pixels (`semantic_mask_stage.cpp:134-142`), and publishes it in the existing `mask_confidences`
attribute. No new attribute, no code change.

Measured over 277 apartment frames, 224 published door masks:

| | confidence |
|---|---|
| before (ungraded) | **1.000 for every single one** |
| after (graded) | min 0.274 · p10 0.527 · **median 0.845** · p90 0.950 · max 0.987 |

**14.7 % of published door masks score below 0.60, 4.9 % below 0.40** — masks every semantic-path
agent currently treats as certain. Same fix lands for hood, cabinet and shelf.

**Your action:** `etc/config.toml:130`
```toml
model_path = "models/yolo26/yolo26l-sem-ade20k-probs.onnx"
```
then restart retina when convenient. Reverting is the same line. ⚠Deliberately NOT changed here —
it is a running-agent config, and the lifecycle is yours.

⚠**Half 1 does not stop doors dying.** It corrects how much a DETECTION is worth. The failure mode
is what a SILENCE is worth, and that is Half 2.

---

## Half 2 — the posterior field. BLOCKED on a cortex reinstall.

To damp absence, `door_concept` needs `max P(door)` over its **own projected silhouette** on frames
where no mask exists. Retina cannot know where that silhouette is without a top-down channel, and
top-down is ruled out (`IMPLEMENTATION_PLAN_decouple_retina.md` — that loop was built, limit-cycled
and removed; "Top-down ROI is allowed only as a *gain on the budget*, never as an evidence-
suppression mask fed to an estimator").

So retina must publish the **field**, forward, consumer-agnostic, and each agent samples it where it
needs to. That is one producer, many consumers — the existing contract, unchanged.

### The four lines to add

In `cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`, beside the existing
`semantic_*` block (currently lines 873-877):

```cpp
REGISTER_TYPE(semantic_class_probs,    std::reference_wrapper<const std::vector<float>>,   false)  // K planes, row-major, K*h*w
REGISTER_TYPE(semantic_prob_class_ids, std::reference_wrapper<const std::vector<float>>,   false)  // ADE20K id per plane (float: cortex has no int-vector type; mask_label_ids sets the precedent, :814)
REGISTER_TYPE(semantic_prob_width,     int,                                                false)
REGISTER_TYPE(semantic_prob_height,    int,                                                false)
```

Then reinstall cortex and rebuild retina + door_concept. (Per CLAUDE.md the header is root-owned and
**you** reinstall it; I have not touched it.)

### Why this is cheap, unlike the blob that crashed

`[Semantic].publish_node` is off because the **dense label blob** crashed — 1920×960 = 1.8 MB/frame.
The posterior field is at the classifier's native resolution, cropped to the active region:
**80×40 × 5 classes × 4 B = 64 kB**, i.e. **28× smaller than the label map**, and rate-capped by the
existing `publish_min_interval_s`. It is not the thing that crashed.

### Then, in order

1. `graph_publisher.cpp` — extend `publish_semantic()` (or a sibling) to flatten `SemanticMap::probs`
   into `semantic_class_probs` + the id/size attrs. Type-attributed access only; never
   `runtime_checked_*`.
2. `door_concept` — sample the field over the projected silhouette and use it as the measured
   `p_vis`, replacing the purely geometric
   `p_detect = resolvability × in_fov_frac × central_frac` (`specificworker.cpp:1661`) that reads ≈1
   face-on. A classifier that was at 0.44 vs 0.44 did **not** resolve the question; p_vis→0 and
   `ΔL = p_vis · log-ratio` yields HOLD with no threshold added.
3. A/B on `door_events.csv`, both directions: a real door survives a face-on parked stare, **and** a
   genuine phantom is still removed.

### One thing to decide when we get there

Publishing a per-class field means declaring which classes. Today the model exposes
`wall, cabinet, door, shelf, hood` (declared in the ONNX metadata, read at load — no config key can
drift from it). `wall` is in there because it is the competitor that beats a door; without it a low
P(door) cannot be read as "narrowly lost" rather than "not seen".
