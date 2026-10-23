# Room recognition on entry through a never-crossed door — design

Date: 2026-10-22. Author: Claude with the user (pbustos). Status: draft for review.

## Intent

A room is "known" today only through **passage memory**: crossing a door that ltsm remembers makes it recall the room
behind it (`recall_room_*`, `RoomRecall`). A door that has never been crossed carries no such memory, so whatever lies
behind it is "entered FRESH" (door_concept `specificworker.cpp:3454`) and goes down the **estimation branch** — even
when the room behind it is one the robot already knows. With the 2026-10-22 layout (new **DOOR_2** joining room_3 and
room_2 in `threerooms-piso.wbt`, closing a loop) this makes a **duplicate** of a known room, with its own frame, layout
and doors. The user's words: "the robot has to enter as if it were new but be able to recognize it and download the
LTSM instance". Both failure modes are disastrous: a new room mistaken for a known one (wrong layout downloaded), and a
known room not recognised (duplicate).

## The decision flow (user's)

1. **The door is the first key.** Door crossed before and the room behind it known ⇒ exactly as now (passage recall).
2. **Never-crossed door** ⇒ two hypotheses: it leads into a known room, or into a new one.
3. **Enter as if new.** The normal estimation branch runs (proto-room, layout, localisation in a provisional frame);
   nothing waits for recognition.
4. **Recognise in parallel**, combining TWO independent modalities (user: "both evidences should be combined"):
   - **Metric** (user): ltsm composes the robot's pose through the chain of seams along the path actually taken into a
     common frame (room_1's), with covariance propagated, and computes P(entering / inside known room k) against each
     known room's polygon expressed in that frame.
   - **Appearance**: the DINO descriptor-to-pose maps stored with each known room give a likelihood that the current
     RICOH views belong to room k (and at which pose).
5. **On recognition, download** the known room's ltsm instance (layout, frame, doors, contents) in place of the
   provisional room, re-expressing the robot's pose in that room's frame.
6. **Reversible** (user): if later evidence contradicts the commit, back up to the state "just entering" — the
   provisional new room — and continue as new.
7. **Late recognition** (user): the robot may be well inside the room when evidence becomes decisive; it must then be
   **relocalised** in the known room's frame.

## Goals

- One posterior over the hypotheses {known room 1…n, new}, fed by the metric and the appearance evidence; neither
  modality decides alone. Commit when the posterior is decisive (expressed as evidence in nats / posterior mass, not a
  hand threshold — CLAUDE.md "no thresholds"; any unavoidable decision boundary is flagged and justified, e.g. the
  commit is the MAP with a margin derived from the cost of each error).
- Pose consistency is part of the evidence: the pose in room k implied by the metric chain and the pose implied by the
  DINO match must agree within their covariances; disagreement is evidence against k.
- Commit is staged and reversible to "just entering"; a wrong commit never writes back into the known room's ltsm copy.
- Late recognition relocalises using room_concept's existing relocalisation mixture search, seeded by the metric-chain
  pose prior and the DINO pose likelihood.
- What the provisional phase learned that the known room lacks is carried over: at least the newly crossed door
  (DOOR_2), which creates the passage so the next crossing of that door recalls the room directly.

## Non-goals

- No global map / no merging of all rooms into one frame: rooms keep their own frames; room_1's frame is only the
  reference for the metric query.
- No recognition through doors already remembered (step 1 stays as is).
- No change to the estimation branch itself (proto-room, layout) beyond what the swap needs.
- Not a general loop-closure / SLAM back-end; seams are not re-optimised in this pass.

## Anti-goals

- A recognition that fires from one modality alone.
- A commit that cannot be undone, or that corrupts the remembered room when undone.
- A recognition that stalls the robot at the door waiting for evidence.

## Current state (verified 2026-10-22)

- **Appearance memory exists but is OFF and unread.** retina `PlaceStage` (`retina/src/place_stage.h`) encodes the
  RICOH panorama with DINOv2 (`models/dinov2/dinov2_vits14_reg_448x224.onnx`) into per-sector descriptors, tags each
  with the robot's room pose + covariance, and accumulates an `rc::place::PlaceMap` (`common/place_memory/place_map.h`:
  `Keyframe`, `MatchResult`, `MixtureParams` measured by `retina/tools/place_eval.cpp`). `[PlaceMemory] enabled = false`
  in retina's config; "NOTHING READS IT" (`place_stage.h:11-15`); it was built as an IN-ROOM pose likelihood for
  room_concept's `grid_search_initial_pose`, and it is one map, not one per room.
- **Metric chain exists in ltsm.** Eviction measures `T(room_new <- room_old)` through the robot
  (`ltsm_agent/src/room_eviction.h:10`); passages relate rooms through their two door faces
  (`seam_through_passage`, `room_eviction.h:47`); archived rooms keep their polygon (`delimiting_polygon_x/y`).
- **Known-room recall** is passage-driven only (`recall_room_*` posted under a door face; promotion "crossed door … ↔ …").
- **Relocalisation** in room_concept is a mixture search with rivals (memory: relocalisation redesign 09-16).

## Calls made by Claude (not decided by the user — check these)

1. **Per-room DINO maps:** PlaceStage tags every keyframe with the current room id and the maps are archived with the
   room in ltsm (so a recalled room carries its appearance memory); PlaceMemory gets enabled.
2. **Where the posterior lives:** in ltsm (it owns the rooms, seams and archive); retina provides appearance matches
   against archived maps on request or continuously for the provisional room; room_concept provides the provisional
   pose and does the relocalisation.
3. **Metric chain** along the path actually taken (the sequence of crossed passages since the last known room), with
   covariance through each seam and the odometry since the last seam.
4. **Carry-over** on download: the crossed door face (new passage) is added to the known room; other provisional
   contents (objects, walls) are handled by the existing concept agents' restore paths, not copied.
5. **Undo** keeps the provisional room's state from entry until the commit has held against new evidence; the archive of
   the known room is read-only while a commit is staged.

## Decisions after the builder check (2026-10-23)

User:
- **No audit of known passages.** If the door has been traversed before, recognition does NOT run; passage recall is
  trusted as today (step 1 unchanged).
- **Cortex reinstall is OK.** New DSR attributes are allowed (REGISTER_TYPE in cortex's `dsr_attr_name.h`; the user
  reinstalls; then rebuild the agents).

Claude's calls (check these):
- **Metric chain = the path actually taken.** The robot's pose is composed through the seams of the passages crossed
  since the last known room (e.g. R1 → R2 → DOOR_2) into room_1's frame; each known room's polygon is placed in room_1's
  frame through ITS OWN stored seam chain (e.g. R3 via R1 → R3). The disagreement between the two chains IS the metric
  evidence. Seams are trustworthy only since Tasks S/M (aperture/mouth seam, inlier/outlier fusion, 2026-10-22/23).
- **One joint over pose**, not a product of per-modality likelihoods, so pose information is not counted twice: the
  hypotheses are (room k, pose in k), scored by the metric chain's pose density and the DINO match's pose likelihood.
- **Error costs symmetric** (the user: both errors are disastrous): the commit is the MAP over {known 1…n, new} once its
  posterior dominates; the "new" hypothesis needs a calibrated likelihood (see phase 0).
- **Phase 0 before building: measure DINO cross-room discrimination.** Record DINO descriptors on recorded tours of R1,
  R2, R3 and measure how well room k's map separates its own views from other rooms' (that gives the "none match" density
  and P(new)); `place_eval` today covers only in-room pose. If DINO cannot separate the rooms, the design leans on the
  metric modality and says so.
- **"Known" = archived rooms in ltsm** (ltsm keeps one room live).
- **Carry-over:** the newly crossed door (e.g. DOOR_2) is added as a passage with a face in BOTH rooms' archives, its seam
  measured on this crossing (Task S/M seam path).
- **Swap = re-parent into the recalled room** through the existing recall/promotion machinery (the same path a passage
  recall takes), not a new node lifecycle; the provisional room's own nodes are evicted as on any promotion.
- **Gold-tour pass criteria:** no duplicate room node in ltsm memory after R2 → R3 via DOOR_2; robot pose error vs Webots
  ground truth within the localiser's 2σ after relocalisation; DOOR_2 present as a passage of R3 and R2.

## Left to the builder

- The exact likelihood forms (metric: polygon containment under the propagated pose covariance; appearance: the
  measured `MixtureParams`), their combination, and the commit/undo criterion — derived, not tuned.
- How the swap is executed in the graph (node replacement vs re-parenting) consistent with CONCEPT_AGENT_LIFECYCLE.md
  and the promotion/eviction code.
- Message/attribute interfaces between ltsm, retina and room_concept (type-attributed DSR attributes only; new
  attributes need a cortex REGISTER_TYPE + user reinstall).

## Success (user's gold proof)

Tour in `threerooms-piso.wbt` (2026-10-22 layout): **R1 → R3 → R1 → R2 → R3**, the last leg through DOOR_2 (never
crossed, R3 known):
- R3 and R2 on first entry are NEW (not recognised as R1, nor as each other);
- R2 → R3 through DOOR_2 is RECOGNISED as R3, its ltsm instance downloaded, no duplicate room, robot relocalised in
  R3's frame, and DOOR_2 becomes a passage of R3;
- the tour continues normally afterwards.
Plus a reversal test (offline or staged): a forced wrong commit is undone back to "just entering" without corrupting
the known room.
