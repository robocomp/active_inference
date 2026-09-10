# Implementation Plan — Decouple retina ↔ table_concept + bounded voxel attention

**Date:** 2026-06-10
**Author:** pbustos (with Claude)
**Goal:** Break the unstable cross-agent feedback loop between `retina` and `table_concept`, and
bound the voxel population so FPS stays flat — without reintroducing any coupling.

---

## Motivation / problem statement

Today the two agents form a closed loop that is hard to stabilize:

1. `table_concept` fits a parametric table model → publishes a `model_table` OBB into the graph.
2. `retina` reads that OBB, **suppresses the voxels it explains**, builds tracks from the
   remainder, and writes `candidate_pts` / `residual_pts` back onto the `table` node.
3. `table_concept` re-fits from those points → publishes a new OBB → goto 2.

Three independent failure modes stack up:

- **Inlier destruction.** Suppressing explained voxels removes exactly the points that *anchor* the
  pose estimate. The estimator is left fitting to disagreement (residuals) at the margins → it chases
  edges instead of holding the good fit.
- **Circular causality, no contraction.** OBB grows → suppresses more → fewer anchors → drifts;
  OBB shrinks → edge points resurface as residual → epistemic expansion → grows. Free to limit-cycle.
- **Async phase lag.** Two agents at different rates, gated by frame-id staleness → the OBB suppressed
  against is always stale → oscillation.

The suppression also served two *legitimate* purposes that must be preserved by other means:
- (a) keeping the retina's own track segmentation clean (objects-on-table vs table), and
- (b) bounding the voxel population so FPS does not collapse as the grid grows.

## Guiding principles

- **Retina becomes a pure feed-forward sensor.** It never reads concept output. It clusters
  generically and publishes class-agnostic tracks in room frame.
- **Instance commitment lives where the prior lives.** `table_concept` decides "which track is my
  table" by scoring tracks against its own predicted model — data-association-by-prediction.
- **Classify, don't destroy.** The candidate/residual split moves *inside* the concept; inliers are
  retained as anchors in the `SampleQueue`. The free-energy objective (likelihood vs prior vs
  bounded-precision residual) is the stabilizer, and it only works if anchors are not stripped upstream.
- **Attention = bounded budget allocated by expected information gain.** Voxel population is bounded
  intrinsically (salience eviction + decay + foveation). Top-down ROI is allowed only as a *gain on the
  budget*, never as an evidence-suppression mask fed to an estimator.

This is a fractal of the existing `SampleQueue` design (bounded bins + RFE eviction + forgetting),
lifted from points-per-table up to voxels-per-scene.

---

## Workstream 3 (land first — independent, de-risks FPS)

**Bounded voxel attention in `UnifiedVoxelGrid`.** All signals already exist on `VoxelState`
(`n_obs`, `n_frames_seen`, `last_frame`, `n_ray_traversals`, `best_confidence`, `track_id`). This is a
policy change, fully self-contained in the retina.

- [x] **3.0** Reduce hard static cap `UnifiedGridConfig::max_voxels` 250'000 → **25'000**
  (`unified_voxel_grid.h:84`). *Done.* (Now just the conservative startup value; the regulator below
  manages the live cap.)
- [x] **3.0b Homeostatic FPS→budget regulator (AIMD).** *Done & tested.* Instead of a fixed cap, treat
  FPS as the metabolic constraint and adapt `max_voxels` to hold a target frame-rate:
  - `voxel_budget_regulator.h` — pure, header-only AIMD law (multiplicative back-off when FPS drops
    below the band, additive probe-up when there's headroom, dead-band hold). Defaults: target 8 fps,
    range [8k, 80k], start 40k.
  - `UnifiedVoxelGrid::set_max_voxels()/max_voxels()/size()` — runtime cap control.
  - `SpecificWorker::regulate_voxel_budget()` — called each `compute()`, self-throttled to one control
    tick / 2 s, reads `fps_counter_.get_frequency()` (guarded against the startup `inf`), resizes the
    cap, and emits a **`[VoxelBudget]` gauge log** (fps / target / cap / live / range).
  - **Experiment:** `tests/test_voxel_budget_regulator.cpp` — synthetic machine model
    `fps = 1/(t0 + k·voxels)` with a mid-run load spike. Validates: holds 40k @ 8 fps nominal; on spike
    fps→3.85, backs off 40k→16.9k in 3 ticks and recovers to 8.25; on relax, probes 16.9k→36.9k and
    settles at 8.56 fps (in-band 15/15). Build/run:
    `g++ -std=c++23 -O2 -Isrc tests/test_voxel_budget_regulator.cpp -o /tmp/vbr && /tmp/vbr`. **PASS.**
  - Retina rebuilt with `cbuild` — compiles & links clean.
- [x] **3.1 Salience-weighted eviction.** *Done & builds clean.* Replaced age-only `_enforce_max_voxels`
  with `_voxel_salience(vs, frame)`:
  ```
  salience = w_recency·1/(1+k·age) + w_persist·min(1, n_frames_seen/ref)
           + w_conf·best_confidence + w_assigned·(track_id != 0)
  ```
  Evicts lowest-salience via `partial_sort`. Weights exposed on `UnifiedGridConfig`
  (`evict_w_*`, `evict_recency_age_k`, `evict_persist_ref`). Sweep now evicts the full `overflow + 5 %`
  in one pass so a sudden regulator cap-drop is honoured promptly (was trimming only 5 %/frame).
  Recently-seen, long-confirmed, high-confidence, track-assigned voxels are protected; transient
  clutter goes first.
- [ ] **3.2 Decay / forgetting.** Per frame, decay an effective salience (`*= α`); re-observation
  refreshes. Evict any voxel below a floor immediately, independent of the cap, so the steady-state
  population settles *below* `max_voxels` (inflow ≈ outflow) and dynamic-scene ghosts self-clear.
  Implement as a continuous variant of `_enforce_max_voxels` driven by `last_frame` staleness +
  `n_frames_seen`. Add `decay_alpha`, `salience_floor`, `stale_frames` to config.
- [ ] **3.3 (optional, biggest win) Foveation.** Resolution / keep-priority as a function of distance
  to robot/sensor: full-res in the workspace, coarse far away. Most of the budget is far-field clutter.
  Defer if 3.1+3.2 already hold FPS at the new cap.
- [ ] **3.4 Verify** FPS is flat over a long run; confirm confirmed structure (table legs/top) survives
  eviction while clutter is dropped. Log steady-state `_grid.size()`.

## Workstream 1 — Retina feed-forward `tracks` publisher

Remove the concept dependency; publish generic tracks. Use **runtime/dynamic attributes** (same as the
`masks` node) → no `REGISTER_TYPE`, no shared-header edit, no recompile of the cortex type registry.

- [x] **1.1 Create `tracks` node** — `ensure_tracks_node_in_dsr()`, mirrors masks node, RT edge under
  `zed`, idempotent. *Done.*
- [x] **1.2 `publish_tracks_to_dsr()`** — iterates ALL track candidates, `get_flat_pts_for_track(id,400)`,
  fills the SoA layout, runtime attributes. **No** `table`/`model_table` read. *Done.* Logs `[Tracks]`.
- [x] **1.3 Delete `update_table_nodes_from_tracks`** — replaced by the two functions above; `compute()`
  now calls `ensure_tracks_node_in_dsr(); publish_tracks_to_dsr();`. `tracks` added to
  `cleanup_semantic_grid_nodes` reset. *Done. Retina builds clean.* (NB: `graph_object_boxes` is
  still passed to `process_rgbd_frame` for the retina-internal `[ModelSuppression]` — that's the
  separate "objects-on-table" concern, deferred.)
- [x] **1.4** `masks` node kept (still published for the yolo viewer / raw 2D); `tracks` is now the
  concept's primary input. *Done — masks retained, non-breaking.*

### `tracks` node schema (room frame, runtime attributes)

| attribute | type | per | meaning |
|---|---|---|---|
| `track_frame_id` | `int` | — | publish/sensing frame |
| `track_count` | `int` | — | N tracks this frame |
| `track_ids` | `vector<float>` | track | **persistent temporal id** |
| `track_labels` | `string` | track | `\|`-joined dominant label (a *vote*, not grouping authority) |
| `track_label_ids` | `vector<float>` | track | class id |
| `track_confidences` | `vector<float>` | track | label-vote confidence |
| `track_last_seen` | `vector<float>` | track | frame last observed → staleness |
| `track_voxel_counts` | `vector<float>` | track | point mass |
| `track_centroids_xyz` | `vector<float>` | 3·track | room frame |
| `track_bbox_min_xyz` | `vector<float>` | 3·track | AABB min, room frame |
| `track_bbox_max_xyz` | `vector<float>` | 3·track | AABB max, room frame |
| `track_support_offsets` | `vector<float>` | track+1 | prefix offsets (in POINTS) into support points |
| `track_support_points` | `vector<float>` | 3·pts | flat XYZ, room frame, decimated + capped/track |

## Workstream 2 — `table_concept` owns instance assignment (classify-don't-destroy)

- [x] **2.1** `TrackSlice` / `TracksPacket` structs + `refresh_tracks_packet()` (mirrors
  `MasksPacket` / `refresh_masks_packet`); called from `compute()`. *Done.*
- [x] **2.2** `select_track_for_table(TableInstance&)` — scores each `"table"`-labelled track by
  support-point overlap with the concept's own predicted OBB (cx,cy,w,h,yaw + 0.10 m pad), tie-broken
  by centroid distance. Cold start (model at prior, hits=0) degrades to nearest-centroid. *Done.*
- [x] **2.3 Sticky id / hysteresis.** `committed_track_id` on `TableInstance`, +2.0 score bias toward
  the committed track. Per-instance, so it generalizes to N tables. *Done.*
- [x] **2.4** Wired into `observe_table_node` as the **primary** branch (before masks), reusing the
  existing SDF split → candidates stay as `SampleQueue` anchors (the stabilizing change). *Done.
  table_concept builds clean.*
- [x] **2.5** Masks path kept as fallback; the legacy `candidate_pts_att` node-attr path is now dead
  (retina no longer writes it) but harmless — `last_sensing_frame_att` never advances. *Done.*

## Sequencing & verification

1. **WS3 first** (independent): lands the FPS fix, easy to verify in isolation.
2. **WS1 + WS2 together** (they replace the broken loop as a unit; landing one without the other leaves
   the concept with no input). Branch, then verify:
   - retina publishes `tracks`, never touches `table` nodes (grep confirms no `model_table` read);
   - `table_concept` converges from `tracks` alone; no oscillation across the old async boundary;
   - multi-table: two `table` nodes hold distinct `committed_track_id`s, no swap.
   - **Healthy log = stable `track_id`:** `[<table>] tracks=<frame> track_id=<id> label='table' cand=… resid=…`
     should keep the **same `track_id`** across frames (sticky association holding) while FE converges; a
     flickering `track_id` means the OBB-overlap score is ambiguous → raise the sticky bias or OBB padding.

## Open decisions (defer, not blocking)

- **Epistemic "unexplained nearby geometry"** residual was richer than an SDF split (it used lidar in a
  neighbourhood). If the next-best-view signal needs it, either ship a small set of *near-but-unassigned*
  points per track from the retina, or have the concept query lidar directly. Start without it.
- **Retina track segmentation** (objects-on-table vs table): if suppression was also keeping clusters
  clean, replace concept-OBB suppression with generic geometry (plane/RANSAC or support-surface height
  removal) owned by the retina. Not required for the loop fix.
- **Top-down ROI**: if/when a task agent broadcasts a workspace ROI, it enters WS3 as a *budget gain*
  (raise keep-priority/resolution in-ROI), never as a residual mask.

## Known issue — graph-viewer crash on agent start/stop (WS4)

The retina SIGSEGVs intermittently when table_concept starts/stops. gdb shows the crash in the
**Qt widget repaint path** (`QPaintEvent::~QPaintEvent` / `QRegion::~QRegion` →
`QWidgetRepaintManager::flush/paintAndFlush`) with **no app frames** — the classic signature of the DSR
**graph viewer painting a `QGraphicsItem` that was deleted mid-frame** as nodes churn across agent
join/leave. Removing our `trigger_graph_layout_twopi()` from `ensure_tracks_node_in_dsr()` reduced the
frequency (immediate → ~3 cycles) but did not fix it; the root cause is in the DSR graph-viewer library.

- [x] **Mitigation A — owned-node cleanup bug (table_concept).** *Done.* `cleanup_owned_nodes()` deleted
  the affordance node only `if (is_active())` (= FSM ≠ idle), orphaning a `table_afford` node whose FSM
  had returned to idle while its parent table node was deleted → dangling node + edge feeding the
  viewer churn. Fixed to delete by `node_id() != 0` (node exists). Complements the startup sweep of
  stale `table_afford*` nodes in `initialize()`. Reduces churn; not the core fix.
- [ ] **Mitigation B — stopgap: disable the graph view.** Set `graph = false` in `retina/etc/config.toml`
  (line ~12). The DSRViewer window still hosts the Voxel3D/YOLO docks; only the node-graph
  `QGraphicsView` (the thing churning) stops drawing. Verify the docks still attach with all of
  tree/graph/2d/3d off — if the `graph_viewers` map ends up empty, the custom-widget block in
  `initialize()` is skipped and we lose the 3D/YOLO panels.
- [ ] **Core fix — audit DSR graph-viewer guards.** The real bug: the graph viewer paints a node that is
  deleted in the middle of a frame. Audit `dsr/gui/viewers/graph_viewer` for missing guards on
  node-delete vs paint (likely needs the delete to be marshalled to the GUI thread and/or the
  `QGraphicsItem` removed from the scene before the node is freed). Confirm with a `-g3 -O0` or ASan
  build so the corrupting free shows its origin (current optimized binary gives no app frames/line nums).

## Build / run notes

- Build with `cbuild` (not raw ninja) per project convention.
- Both agents are Ice/DSR components; restart the bridge after schema-affecting changes to drop stale
  logs.
