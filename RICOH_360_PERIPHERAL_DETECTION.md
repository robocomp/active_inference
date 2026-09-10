# Ricoh 360 peripheral detection — plan

Status: **Part A DONE+LIVE-VALIDATED** (2026-07-02, UNCOMMITTED). Threaded worker (`rc::RicohYoloWorker`)
confirmed over an 85-minute run: hit 19.97Hz (target 20Hz), and the main `compute()` thread's tail
latency dropped sharply now that ricoh is fully off it (compute_ms mean 17-33ms→12.3ms, >25ms frames
7.7%→0.3%, the old 400+ms mystery spikes gone — see the memory file for full numbers). Parts B/C/D
still NOT STARTED — nothing wired to DSR/masks/concept-agents/affordances yet.

Two nested plans: (A) how to run YOLO on the 1920×960 ricoh panorama without destroying detection
quality, and (B) how a no-depth detection from it is allowed to affect the system at all, given the
existing architecture (retina = perception only, concept agents own identity + affordances,
controller only evaluates). (B) was the point that got corrected mid-design — retina must NOT
author affordances or ray-cast targets itself; it only ever produces evidence (masks).

## Idea (one paragraph)

A 360 detection is a **glance, not a look** — narrow-FOV zed gives depth and precision; ricoh gives
wide coverage and nothing else. Treat it exactly like the existing ego-motion mask-corruption pattern
(see MASK_MOTION_CORRUPTION.md): the producer (retina) reports the *physics of the observation*
(here: "I saw `label` at this bearing, with no range"), and it's entirely up to the *consumer*
(the owning concept agent) to decide what that's worth — a weak confirmation of something it already
tracks, or a candidate for something new worth turning to look at (a saccade). Same "masks" DSR
contract, same `common/mask_ingestor` choke point, no new component types.

## Division of labour

- **Producer (retina)** — run YOLO on the panorama (3-way split, see Part A), publish detections as
  `masks` slices with `has_depth=false` + a bearing instead of a 3D point (Part B). Zero identity
  reasoning, zero affordance authorship — same as it does for zed today.
- **Consumer (concept agent: table/chair/bottle/...)** — owns confirm-vs-new (Part C) and is the ONLY
  place allowed to author an affordance from this (Part D).
- **Actor (controller)** — unchanged; a new `Orient` policy is just one more Contract it evaluates
  through the existing EFE machinery.

---

## Part A — process the ricoh panorama in 3 vertical strips

**Why**: 1920×960 squashed directly to YOLO's 640×640 input halves vertical resolution and squashes
horizontal ~3×, badly distorting every object's aspect ratio right before the detector sees it.
Splitting into strips keeps each detector call close to a normal camera aspect ratio.

1. **Geometry**: 1920 / 3 = **640** px per strip (not 620 — corrected arithmetic from the first pass),
   full height 960. Strips are columns `[0,640)`, `[640,1280)`, `[1280,1920)`.
2. **Wraparound**: this is a full 360° cylindrical panorama, so there are **3 seams**, not 2 — the two
   internal cuts *and* the wraparound at column 0/1920 (physically adjacent in the real world). Every
   seam needs the same treatment.
3. **Overlap margin**: extract each strip with extra columns on both sides (e.g. ±64–96 px, modular
   indexing across the 0/1920 wrap) so an object centred on a seam isn't cut in half and lost.
4. **Per-strip detection**: feed each ~640(+2×overlap)×960 strip through the *existing*
   `YoloProcessor::detect_segmentation()` unchanged — it already letterboxes (scale-to-fit + pad, see
   `yolo_processor.cpp:296-309`), so no manual resize/squash step is needed.
5. **Reproject to global panorama coordinates**: `col_global = (strip_index*640 + col_local - overlap)
   mod 1920`. Do this to the raw boxes/mask pixels before anything downstream sees them.
6. **Cross-strip merge**: after per-strip NMS already happens inside `detect_segmentation()`, run a
   *second* NMS/IoU-merge pass over the union of all 3 strips' final boxes in global coordinates — this
   is what dedupes a seam-straddling object detected (partially) in two adjacent strips. Two-stage
   (existing per-strip NMS + one cross-strip merge) is simpler than restructuring the pipeline to expose
   raw pre-NMS proposals, and is equivalent for the case that matters.
7. **Where it lives**: a new method on the *existing* `YoloProcessor` (e.g. `detect_segmentation_360()`)
   rather than a new class — same model/session, same `decode_mask`/postprocess helpers, just different
   pre/post geometry. Returns `SegDetection`s already in full-panorama pixel coordinates.
8. **Column → azimuth**: for a cylindrical projection, global column maps ~linearly to azimuth
   (`azimuth = (col_global / 1920) * 2π`, plus the camera's mounting-yaw offset). **Verify this against
   the actual ricoh descriptor/projection model before relying on it** — not yet confirmed in code, only
   inferred from the bridge setup.

### Open decisions (Part A)

- **Sequential vs batched inference**: v1 = 3 sequential `detect_segmentation()` calls (simple, correct,
  ~3× the single-frame yolo_ms cost). True batching (one ONNX forward pass, batch=3) needs the model's
  input shape to support a dynamic batch dim — currently hardcoded to batch=1
  (`yolo_processor.cpp` input_shape). Given how much of this session was spent chasing `compute_ms`
  latency, start with v1, profile with the same `viewer_perf.csv`/`viewer_perf_render.csv`
  instrumentation, and only batch if it's the actual bottleneck.
- **Decimation**: this must run much slower than the main zed YOLO pass — it's peripheral awareness,
  not tight tracking. Add a `Ricoh.detect_decimation`-style knob (mirrors `HUMAN_POSE_DECIMATION` /
  `SEMANTIC_SEG_DECIMATION` already in `RetinaParams`), default maybe every 5–10 compute cycles.
- **Decouple from popup visibility**: today `poll_ricoh()` only decodes when the popup window is
  visible (`ricoh_wanted_`). Detection must run regardless of whether anyone has the popup open — needs
  a second "wanted for detection" gate, independent of the display gate, decided by the decimation
  counter inside `compute()` rather than the render tick.

### Part A — what actually got built (2026-07-02)

All decisions above resolved in favour of the cheap option — measure first, don't pre-optimize:

- `YoloProcessor::detect_segmentation_360()` (`yolo_processor.h`/`.cpp`) — 3 sequential
  `detect_on_strip()` calls (v1, no batching), circular-column-padded strip extraction (one `hconcat`
  handles the internal seams AND the 0/width wraparound identically, see `circular_pad_columns`), a
  linear "extended" global-coordinate space so no modular arithmetic is needed until the very last
  step, and a greedy highest-confidence-first cross-strip merge (`rect_iou_with_wrap` — the one
  wraparound strip PAIR needs a ±width-shifted IoU check too, since the same physical column is
  numerically ~width apart between the first and last strip's local spaces; caught this in testing,
  not in the original plan). No tray-phantom suppression (`detect_on_strip` skips it — zed-only
  calibration). `Detection360Config` is a free struct, not nested in `YoloProcessor` — GCC 15/C++23
  rejects a default *argument* built from a nested class's own default *member* initializers before the
  enclosing class is complete; cost half an hour to isolate, worth remembering.
- `SceneProcessor::poll_ricoh(bool force = false)` — added the force parameter so the decimated
  detection cycle can request a decode independent of `ricoh_wanted_` (popup visibility), without
  touching the popup's own polling behaviour (default `force=false` preserves it exactly).
- Wired into `SpecificWorker::compute()`, decimated (`Ricoh.yolo_decimation`, default 20 → ~once/sec at
  20Hz), OFF by default in the struct, **ON in `etc/config.toml`** (mirrors how `HumanPose.enabled` is
  struct-default-off but config-on) so it's actually live to test. Cost is measured, not assumed: two
  new `viewer_perf.csv` columns (`ricoh_yolo_ms`, `ricoh_det_count`), zero cost on non-decimated cycles.
  The popup (when visible) overlays the last cached detections via the existing
  `compose_detection_canvas` — a cheap draw reusing the cached result, no extra inference.
- **Live-tested 2026-07-02**: user confirmed it works (screenshot showed real table/chair boxes on the
  panorama). Found two follow-up issues, both fixed this session: a panorama-seam bug traced to
  `webots-bridge`'s camera stitch order (NOT this repo — see the `ricoh-360-stitch-seam-recenter`
  memory), and the rate being too low (see below — moved off decimation onto a dedicated thread).

### Part A.2 — moved onto a dedicated thread (2026-07-02)

Decimation traded ricoh's rate directly against `compute()`'s budget — user asked to decouple them so
ricoh can run near 20Hz on its own. Replaced the inline decimated call with `rc::RicohYoloWorker`
(`ricoh_yolo_worker.h`/`.cpp`) — a `std::thread` paced to `Ricoh.yolo_thread_period_ms` (default 50ms,
~20Hz), mirroring the `read_ricoh_thread` idiom `robot_concept` already uses for its Ice-bridge readers.

- **Own `YoloProcessor`, own ONNX session** — deliberately NOT sharing the zed detector's session
  across threads. Concurrent `Run()` on one ORT session is generally thread-safe, but that guarantee
  gets murkier under TensorRT; a second session is cheap enough that it wasn't worth relying on it.
- **`SceneProcessor`'s ricoh cache had to become thread-safe** — it was previously only ever touched
  from the main thread (`on_render_tick`). Added `media_ricoh_mutex_` guarding the `MediaRgbCache`,
  replaced the raw-reference `ricoh_bgr()` getter with a mutex-protected `ricoh_bgr_copy()` (a cv::Mat
  copy is shallow/refcounted — cheap, not a pixel copy; the OLD reference-returning getter was a
  latent cross-thread use-after-free waiting to happen once ANY second thread touched the cache), and
  made `ricoh_last_stamp_ms_` a `std::atomic` (plain `uint64_t` read/write across threads is UB even
  when it happens to work on this platform).
- `on_render_tick()` now branches: worker running → read its thread-safe snapshot only, no polling on
  the GUI thread at all; worker absent (`Ricoh.yolo_enabled=false`) → falls back to the exact old
  direct-poll behavior. Stream-rate telemetry (`ricoh_last_stamp_ms()` → `stream_mon_.tick`) works
  either way since the stamp is now atomic regardless of which thread last polled.
- Worker writes its OWN perf CSV (`etc/viewer_perf_ricoh_yolo.csv`: `t_ms,cycle_ms,det_count`) instead
  of borrowing `viewer_perf.csv` columns — it's no longer part of `compute()`'s critical path, so
  folding its cost into that row would be misleading. The old `ricoh_yolo_ms`/`ricoh_det_count`
  columns were removed from `viewer_perf.csv` for the same reason.
- Shutdown ordering: `ricoh_yolo_worker_` holds a raw `SceneProcessor*` — `request_shutdown()` now
  resets it (stop + join) explicitly BEFORE `scene_processor.reset()`.
- **Not yet done**: no live-camera test of the threaded version this session (code change only, builds
  green). Next session: check `viewer_perf_ricoh_yolo.csv`'s `cycle_ms` — 3-strip inference is roughly
  a 20ms floor, so it may not hit the full 20Hz target on real hardware, but should still land well
  above the old ~1/sec; also confirm `compute_ms` in `viewer_perf.csv` shows ZERO ricoh-shaped spikes
  now (the whole point of moving it off that thread).

---

## Part B — extend the `masks` DSR contract (retina → mask_ingestor)

Add to `MaskSlice` (`common/mask_ingestor/mask_ingestor.h:45-66`):

| field | meaning |
|---|---|
| `has_depth` | `true` for all existing zed masks (unchanged); `false` for ricoh masks |
| `azimuth_room_rad` | room-frame bearing from the robot's current position; populated only when `has_depth=false` |

Ricoh masks: `support_begin==support_end` (no 3D points), `centroid`/`bbox_min`/`bbox_max` left at a
sentinel (NaN), `label`/`class_id`/`confidence` populated as normal. One shared node, one shared
ingestor — no parallel contract, no second `MaskIngestor`-like component.

### Part B — what actually got built (2026-07-02)

- **Contract** (`common/mask_ingestor/mask_ingestor.h`): `MaskSlice` gained `bool has_depth = true`
  (default = the zed contract, so an older producer reads back true) + `float azimuth_room_rad`.
- **Wire format**: two new flat per-slice arrays on the `masks` node — `mask_has_depth` (1.0/0.0) and
  `mask_azimuth` — via `runtime_checked_add_or_modify_attrib_local` (string-named, no schema registration).
- **Producer** (`retina/src/graph_publisher.{h,cpp}`): new `BearingDetection{label,class_id,confidence,
  azimuth_room_rad}` threaded through `publish()`/`upload_masks()`. Zed slices are tagged `has_depth=1,
  az=0`; the ricoh detections are appended as `has_depth=0` slices (`support_begin==support_end`, NaN
  centroid/bbox, motion channels 0). Emitted only alongside a valid zed depth frame (the empty-depth
  early-return drops them for that rare frame — acceptable for peripheral evidence). Gated
  `Ricoh.publish_masks` (default true).
- **Azimuth** (`retina/src/specificworker.cpp` compute): `robot_yaw` (from `room_T_zed`, forward = +y)
  `+ Ricoh.azimuth_offset_rad +` panorama-centre bearing `(2π·col_c/width − π)`, wrapped. ⚠ **PROVISIONAL**
  — the sign/zero/mounting convention is unverified vs the descriptor projection model (Part A step 8); the
  offset knob + flag exist so it can be calibrated live once a consumer (Part C) needs it.
- **Consumer** (`mask_ingestor.cpp`): parses both arrays (back-compat default `true`); `select_nearest`
  skips `!has_depth` slices. **Safety until Part C**: table/chair's tracker feed already filters
  `support_end > support_begin` (a no-point slice is excluded); bottle's feed (label-only) got an explicit
  `and has_depth` guard. So the no-depth slices are inert-but-available today.

---

## Part C — concept-agent fitter: confirm vs new (reuses existing gating, no new threshold pattern)

`ChairFitter::observe()` (`chair_fitter.cpp:245-294`) needs a sibling branch for `has_depth==false`
slices matching its label:

1. Compute each existing instance's *predicted* azimuth from robot pose + instance centroid (`atan2`).
2. Gate the angular innovation the same way `instance_tracker.h:118` already gates position
   (`S = P + R²`, `m² ≤ gate_mahalanobis`) — just 1-D and angle-only. Same soft-gate style already in
   the codebase, not a new hard cutoff.
3. **Gate passes** (matches an existing instance) → "confirmation of existence": refresh
   staleness/last-seen, maybe a small precision nudge. No SDF fit — there's nothing 3D to fit.
4. **Gate fails against every instance of that label** → candidate for something new. Feed it through
   the *existing* `Candidate` birth-staging in `instance_tracker.h:165-201` (already requires
   `birth_frames` consecutive consistent hits before promotion) generalized to angle-only innovations —
   reuse, don't reinvent.

### Part C — what actually got built (2026-07-03, CONFIRM half only)

- **Shared helper** `common/bearing_confirm/bearing_confirm.h` (header-only, Eigen-only, object-agnostic):
  `rc::confirm_tracks_by_bearing(tracks, bearings, robot_xy, angular_gate_rad)` → for each live
  `rc::TrackView`, predicts azimuth `atan2(track − robot)` and matches the nearest bearing within a 1-D
  angular gate (greedy 1-to-1). Mirrors the tracker's soft-gate style; no new hard cutoff.
- **Wired into bottle only** (`bottle_concept` — the DEATH-enabled agent, so the confirm has an observable
  effect; table/chair keep death OFF → `expected_visible=false` is a no-op there, so they're intentionally
  NOT wired yet). In `run_instance_tracker`, between building the `TrackView`s and calling the tracker: it
  collects the `!has_depth` "bottle" bearing slices, resolves the robot room-XY (`inner_eigen` zed origin →
  room), and for each confirmed track sets `TrackView.expected_visible=false` so the InstanceTracker HOLDS
  that instance's death-miss this cycle — a peripheral "glance" is evidence it wasn't removed, exactly like
  being out of the zed frustum. A `[bearing] confirm bottle id=… innov=…deg` line logs each match.
- **Flag-gated, default OFF** (`Bearing.ConfirmEnabled`, gate `Bearing.ConfirmGateRad`≈10°): the azimuth
  convention from Part B is still PROVISIONAL, so the intended first use is to enable it and watch the log
  to VERIFY the bearing hits the right instance before anything harder relies on it.
- **DEFERRED — the birth half (step 4 above).** A single bearing is a ray, not a position; birthing an
  instance needs a range assumption (arbitrary) or a hypothesis that triggers an Orient move to go get
  depth — which is Part D. So angle-only birth is folded into the Part D work, not built here.

---

## Part D — affordance authorship (the actual gap — nothing like this exists yet)

Confirmed nothing today creates an affordance for an unconfirmed hypothesis: affordance creation
(`chair_affordance.cpp:143`, `write_contract`) only ever fires for instances already in `instances_`
(`specificworker.cpp:767`, `step_epistemic` on confirmed instances only).

- Once a bearing-only `Candidate` (Part C.4) stabilizes but hasn't promoted to a full track yet, the
  owning concept agent publishes a **low-gain exploratory `Contract`**.
- New policy: **`Policy::Orient`** — rotate in place toward the bearing, no distance/translation solve.
  Minimal new plumbing (no target `(x,y)` needed, unlike every existing Contract which requires
  `epistemic_target_x_m/y_m`), and the more faithful version of the glance→saccade metaphor: a saccade
  reorients, it doesn't walk over. Once oriented, the zed camera gets a real look; the candidate either
  promotes (a real, depth-backed mask arrives) or decays.
- Reuse the existing `epistemic_cooldown` anti-churn (`specificworker.cpp:857`) for exploratory
  proposals too, so a flickering bearing doesn't spam orientation affordances.

### Open decision (Part D)

- Does `Policy::Orient` need its own goal-clause semantics (e.g. "done" = a fresh `has_depth=true` mask
  arrived for this candidate) or can it reuse the existing `GoalClause` shape unchanged?

---

## Verify first

- Confirm the ricoh column→azimuth mapping against the actual descriptor/projection before trusting
  Part A.8 — don't build the room-frame bearing math on an assumption.
- Confirm `has_depth=false` masks don't break the *existing* zed-only consumers (table/bottle) that
  don't yet know about the new field — default `has_depth=true` and additive-only fields should make
  this a non-issue, but verify no consumer assumes every mask slice has a valid centroid.

## Suggested implementation order

1. Part A (retina, self-contained, testable in isolation via the YOLO popup — no DSR contract
   changes yet).
2. Part B (masks contract extension — small, mechanical, unlocks everything downstream).
3. Part C on ONE concept agent first (chair_concept, per earlier discussion — simplest of the three).
4. Part D once C is confirmed working end-to-end for that one agent.
