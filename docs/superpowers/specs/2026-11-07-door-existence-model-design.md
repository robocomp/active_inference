# door_concept: one existence model, settled

Status: DESIGN for approval (2026-11-07). Nothing coded.
Decisions marked **(PI)** are the user's. Everything else is the designer's call and can be changed.
Inputs: the user's requirements (2026-11-06/07 conversation); Fable's review of the current code (summarised in §2,
verified against the code and the explore_tour18 data); `2026-11-05-structure-from-evidence-design.md` §3.2 (door
birth by comparison, size prior); memory notes door-birth-by-comparison-11-05 and door-work-queue-11-05.

## 1. Intent

Settle door_concept's existence logic once and for all. **(PI)** Remove every accumulated patch and replace them
with one clear generative model, in which:
- a door the robot has just crossed is created at once in the new proto-room;
- a door is kept alive by evidence from segmentation masks AND from reprojecting its 3D model on the RGB image;
- not seeing a door is evidence against it only when it should have been seen. Unseen means no evidence, however
  long it lasts;
- a phantom is deleted when views that should show it keep arriving without it;
- doors are durable memories, as they were in the warm-start runs. There they were restored from door_concept's
  memory and never died, and tours worked.

Why now: exploration from scratch exposes the existence logic. Two of the last five tours failed on it or next to it
(tour18 STUCK: both the loop door and the return door were removed "for absence").

### Goals
- One belief per door place. Birth, survival, deletion and memory all read and write it.
- Every channel enters as a likelihood ratio with its own sensor model. No clamps, floors, streaks, timers or
  per-cycle constants.
- The same visibility model ("should this door be seen in this image?") for the ZED and the ricoh, by reprojection.
- Tours from scratch with no door lost while it is needed. Phantoms do not accumulate.

### Non-goals
- The door's geometry fit (aperture, leaf angle, hinge), the tracker's association, and door actuation are not
  redesigned. They are inputs here.
- The other agents that share `common/existence_belief` (table, chair, cabinet, …) keep their copy until door's
  result is in **(designer, per the 11-05 spec)**.
- The passage controller is not changed, apart from reading the door's role (§3.6).

### Anti-goals (would be failure even if it "works")
- A new patch layered over the old code, or a constant tuned until tour18 passes.
- A door that dies while unseen. A door whose belief moves twenty times a second while the robot is parked.
- The agent below nominal rate (memory: rates before features).

## 2. What is wrong today (the review, verified)

`door_concept/src/core/door_existence.cpp::update_existence` (≈ lines 701-1697) is a stratigraphy of fixes: seven
separate view-correlation implementations, a ±4 clamp standing in for a missing precision, entry-relative
bookkeeping, a removal streak, holds, ghost re-acquisition radii and identity forgetting. Each was added for a
past defect.

tour18's root cause, verified in `explore_tour18/door_existence_log.csv`: the ricoh absence term
(`door_existence.cpp:983-1007`) rated door_5 as detectable with p = 1.0 from 1.4 m. Its detectability is only
`min(1, (2.5/r)²)·LOS·closedness`, with no near-field, size or aspect term. The ZED's proper model rated the same view
p_detect = 0 (the door overflowed or left the frame). Neither detector fired. Each new view charged about −1.1 nats,
and L went 3.9 → −3.9, so the door was removed. The stdout line read "free=0 occl=0" because, with no YOLO mask,
the code `continue`s at `:1009` before refreshing those fields. The evidence was false, not missing.

## 3. The model

### 3.1 The belief
For a door place θ (aperture pose on a wall, width, height, leaf angle) there are two hypotheses: H1 "a door is
here" and H0 "wall or clutter here". One running log-odds:

  L(θ) = structural prior + crossing evidence + Σ over views ℓ_v

owned by one object that lives from the first candidate observation to the last memory row (§3.7).

### 3.1b A door-shaped thing that is not a door (2026-11-08)
**(PI, 2026-11-08)** A place has three hypotheses: D (a door), N (nothing door-like) and K (something door-shaped that
is not a door: a wardrobe, a painted panel). The detectors fire on D and on K alike; only a physical test separates them.
Let Sc be the perceptual evidence of the place (per-view zed + ricoh + geometry + support, over the ledger, plus the
perceptual part carried from memory), Sp the physical evidence (actuation, crossing and the carried physical part), and
lambda = P(K | not a door, the detectors would fire), a sensor-model constant, `Existence.LookalikeProb`, default 0.05,
to learn offline like the detector rates. Then

  L = priors + carried + Sc + Sp − ln((1 − lambda) + lambda · exp(Sc))

computed in one function (`PlaceBelief`), for candidates and born doors alike. For Sc ≫ 0, L tends to
priors + Sp − ln(lambda): vision saturates near +3.0 (ln 20), so 25 agreeing views of one object are one object. A saturated phantom
crosses the delete line in two failed opens. A real door not yet opened stays believed; one that opens or is crossed rises above the
saturation. For Sc < 0 the last term is ≈ −ln(1 − lambda): absence evidence is as before (within 0.06 nats at 0.05).

Memory keeps the split: a stored row carries L_evidence and its physical part. Restoring (revisit or room change) puts the physical
part outside the term and returns the perceptual part to the Sc scale (the formula's inverse), inside it, so new views add under the
same bound and a door restored from memory cannot climb above it on vision alone. A row written before this model holds the old
unbounded sum, which is on the Sc scale; it is saturated once on load.

⚑ Consequence to decide (2026-11-08): the per-wall base rate of a second door is ln(0.01/0.99) = −4.6, more than vision's +3.0 can
overcome: a second door on a wall can no longer be born from vision alone (its L saturates near −1.6); it needs a physical test.

### 3.2 Prior (entered once, replaced not accumulated)
- Doors are in walls: the existing wall-band prior `band_log_prior`, with σ carrying the wall's own uncertainty.
  A proto-room's provisional outline must be treated as a frontier, not a wall (to be defined in the plan; see the 11-05 spec).
- Per-wall base rate **(PI, confirmed)**: first door on a wall P = 0.4, each further door P = 0.01. Walls are groups of
  outline edges (`wall_hypotheses`), not polygon edges.
- Size **(PI, approved 11-05)**: width 0.85 ± 0.12 m, height 2.05 ± 0.06 m, as a KL term on the fitted size. It is
  never scored on raw mask height (a truncated view pays nothing).

### 3.3 Crossing evidence and the protected status
- **(PI)** A crossed door is created in the new proto-room in the same cycle as the crossing (`exit`) fact. The
  crossing is evidence: + ln(P(crossed | door) / P(crossed | wall)), the existing `crossing_logodds` (≈ +2.3). Added
  to the carried belief, it starts the door well above the creation line.
- **(PI)** The door being crossed keeps its protected status for the duration of the crossing. In the decision
  rule (§3.6) its deletion cost is unbounded while the passage is committed to it. Evidence still accumulates
  during the crossing; only deletion is barred. The lifecycle document's crossing exception and the TLA+ crossing
  model therefore stay valid.

### 3.4 Per-view evidence: reprojection for both cameras
Every look reprojects the door model into the camera:
- **ZED**: pinhole, as today.
- **Ricoh (PI: build it)**: an equirectangular projection model, from the sensor intrinsics already carried by the media
  descriptor (SensorModel). It gives the door's projected extent (horizontal from azimuth span, vertical from
  elevation span) and pixel coordinates for sampling.

From the projection, for camera c:
- vis_c: the visible fraction of leaf and frame samples. Counted are samples in the frustum or panorama strip, not
  blocked by walls (own wall excluded), not blocked by other doors' cones, not covered by non-door masks, and
  weighted for beyond-far-face. It is marginalised over the leaf-angle posterior. This reuses today's silhouette
  geometry.
- q_c = P(detector c reports this door) = envelope_c(projected fill, h and v) · aspect · vis_c · κ · h_c.
  - envelope_c: the detector's near/far shoulders, refit per camera from its own hit/miss record.
  - aspect: |cos| of the viewing angle to the aperture plane.
  - κ: pose-uncertainty overlap.
  - h_c: learned per-view hit rate.
- f_c: false-alarm rate of detector c at that place (clutter_prob, learnable from phantom records).

Likelihoods (exact Bernoulli, no branches):
- mask hit associated to the door (association weight a): ℓ = a · ln(q_c / (f_c · vis_c))
- mask miss: ℓ = ln((1 − q_c) / (1 − f_c · vis_c)). When q_c → 0 this is exactly 0, which is the "unseen is not
  evidence" requirement **(PI)**. No HOLD or "probed" switch is needed.
- reprojection agreement on the RGB (runs on every RGB frame, mask or not): the door-frame and leaf-contour statistic
  (`door_frame_evidence.h`, `contour_edge_*`) and the relative depth step, scored as a likelihood ratio against the
  same frame's control placements (slid along the wall). Their mean and spread are measured in that frame, which
  gives the precision today's ±4 clamp stands in for. Weighted by vis_c · κ. A blank wall gives ≈ 0, and an edge where
  the model predicts none is negative.
- actuation (kept): a commanded open or close seen satisfied adds its likelihood ratio once.

### 3.5 One record per viewpoint
All channels share one view ledger per door. A view is a kernel over (bearing from the door in units of its angular
half-size, range in units of its half-size, time), widened by pose spread. This is the kernel `RicohBirthBank::credit`
already uses. All frames of a view are one trial: the view's ℓ is recomputed from its aggregate and replaces its
previous contribution. A parked robot adds one view, and a new standpoint adds one more. This replaces all seven
pools.

### 3.6 Decision: creation and costed deletion
- Create the node when P(door) > P(no door) (L > 0). Birth and survival use the same likelihoods (§3.4), so the
  tracker's streak and the ricoh birth bar go.
- **(PI)** Deletion is costlier: delete when P(no door)/P(door) > C_delete / C_keep, i.e.
  L < −ln(C_delete / C_keep). A door between the two lines lives and is worth a look; the epistemic value of
  checking it is highest there.
- **(PI)** Both costs are derived by the robot, in seconds, from its own experience:
  - C_keep: the expected time spent acting on a phantom (an approach or look that ends with no door). It is
    measured from the controller's passage records.
  - C_delete: the expected time to recover the door, given its role. An ordinary door costs the time to find it
    again. A door that is a room's entry or return path in the exploration drive also costs the route (bounded by
    the room's MaxRoomS clock). The door being crossed costs unboundedly (§3.3).
  - Until experience exists, the initial values are estimated from the existing tour logs and recorded with their
    source.

### 3.7 Memory: the row is the belief
- The identity row stores L (evidence part, without the structural prior), a summary of the view ledger, and the
  last-seen time. Leaving a room writes it; re-entering continues the same L, aged by a change hazard learned from
  revisits (found / not found). This replaces the borrowed 1/600 s.
- Deletion never forgets the row. A row below the deletion line means "no door here" and is the prior for any new
  candidate at that place. Ghosts, the re-acquisition radius, `forget_identity` and the look-affordance's private
  belief merge into this table. The look affordance values its next-best view from the same L.

### 3.8 Observability
- Every existence row prints every channel's contribution for that cycle (ZED, ricoh, reprojection, crossing,
  prior) and nothing stale. A field not computed this cycle is printed as such.
- The CSV gains a column per channel and the view id.

## 4. What goes, what stays

Goes (door_concept side):
- in `door_existence.cpp`: the seed at birth, the existence half of the HOLD, the band as replaced-term bookkeeping,
  the min-height per-frame prior, the whole ricoh block, the `mask_pixels.empty()` short-circuit, the probed()/HOLD
  branch, observed_w/view_untrustworthy/q_explain and the linear p·LLR, the confirm-only/tanh/W-max image block, the
  absence_share/streak/decide machinery, and the since_entry/L_bound bookkeeping;
- in `door_fit_state.h`: the three pools, existence_seeded/entry_L/since_entry, the debounce, existence_held_crossing
  (the crossing protection moves into the decision rule);
- in `specificworker.cpp`: remember_ghost/forget_identity/match_ghost, the carried-door seed (becomes the crossing
  observation), the restore seed and its hazard constant;
- `RicohBirthBank`'s ripening bar and the separate ricoh view tracking;
- config keys: BirthLogodds, MaxLogodds, RemovalProb, RemoveFrames, OutOfRoomGain, ShortGain, MinHeightConf,
  ReacquireRadiusM, GhostMax, CentralRegionFrac.

Stays: the silhouette sampling geometry (as vis), `door_frame_evidence.h`/`contour_edge_*` (plus the control
spread), `aperture_envelope`, `common/perception_card/terms.h`, `common/detectability`, `occlusion.h`,
`pose_overlap_factor`, `RicohHitPool`'s per-view calibration, `crossing_logodds`, the identity-table format
(extended), and the decision log and CSV (extended).

## 5. Tests that prove it (written first)

1. Crossed door: on the `exit` fact the node exists in the new room in the same cycle, with L ≥ crossing evidence.
2. Unseen door (behind the robot, around a wall, overflowing the frame) for minutes, both cameras: L unchanged.
3. Phantom: three distinct face-on views at good fill, with no mask and no edge, drive it below the deletion line and
   it is removed. A 20 Hz parked stare of one view counts exactly one view.
4. Real door at 70° aspect at 1 m, and at 6 m: misses cost less than one good hit-view. After one confirming view it is
   never deleted by such misses.
5. Ricoh from 1.4 m with the door overflowing the panorama strip: miss ≈ 0. This is tour18's case.
6. Costed deletion: an entry/return door survives evidence that deletes an ordinary door. The door being crossed is
   never deleted.
7. Memory: a restored door continues its L. A deleted phantom's row is kept and acts as the prior. Rows are never forgotten.
8. Birth: 0.80 m and 1.00 m doors are born; a 3 m candidate needs more views; a candidate off any wall is not born; a
   second door on a wall needs more evidence, yet strong evidence still births it.
9. Replay: tour18 (`door_inputs.bin`, `tools/door_replay`): door_5 and door_4 are not removed. The existing tapes
   (`tests/test_door_tapes.sh`) keep their door counts.
10. Live: a tour series from scratch; and the agent stays at nominal rate. The geometry probes now run on every RGB
    frame.

Tests to rewrite or retire: `test_door_core_existence.cpp` (entry-relative, cap, restore-at-−4, HOLD cases),
`test_door_absence_detectability.cpp`, `test_door_existence_moving_phantom.cpp`, `test_door_ricoh_lifecycle.cpp`,
`test_door_view_pool.cpp` (becomes the ledger test), `test_door_identity_table.cpp`, `test_door_core_tracker.cpp`
(the "not before birth_frames ideal hits" case), `test_door_look.cpp`.

## 6. Constraints
- C++23, project rules in CLAUDE.md: no thresholds (any that remain are flagged with their justification),
  type-attributed DSR attributes, from_chars for parsing, queued signal connections, -j8 builds.
- CONCEPT_AGENT_LIFECYCLE.md: birth stays tracker-mediated, as the tracker still associates candidates. The
  existence rules there are already what this design implements (absence weighted by P(detect), occlusion holds).
  §1's door exceptions must be amended for birth-by-comparison and the costed deletion.
- Remaining constants, flagged: detector envelope shoulders (sensor-model constants, refit from data), the view
  kernel's widths (the door's own scale), and the per-wall base rates (PI-confirmed priors).

## 7. Left to the builder
- The order of the work. Suggested: ricoh projection and the shared visibility first (it fixes tour18 alone); then the
  ledger and likelihoods; then the decision and memory; then deletion of the old code.
- Exactly how wall groups are formed for the base rate, and how the provisional proto-room frontier is represented.
- How C_keep and C_delete are estimated from logs and updated online, with the estimator documented.
- Whether existence work moves to a worker thread to hold the rate.

## 8. Settled after the builder review (designer's calls — the PI may overturn any)

1. **Crossed door: what it is and how it is born.** The door entered by is the SAME physical door as the face just
   crossed in the room left, so it carries that face's evidence L_src (§3.7 row). It does not pay the structural
   prior again, because the prior was paid once, in the room where the door was first seen. Its L in the new room =
   L_src + crossing evidence. Its geometry is the crossed aperture as room_concept and door_concept measured it at the
   crossing, as today's `recreate_lost_door`/`install_carried_door` path does. Creation uses that existing carried-door
   path; it is already a lifecycle exception and stays one. It is not a tracker candidate.
   Consequence: a crossed door is above the creation line whenever its source face was alive, so test 1 holds.
2. **Initial costs and the door's role.** Starting values come from the tour logs (explore_tour10-22):
   - C_keep: the median duration of door-passage attempts that ended with no crossing (passage failed, "door reads
     shut" back-off, vanished door).
   - C_delete for an ordinary door: the median time from a door's removal to its re-birth at the same place
     (`RE-ACQUIRE` rows).
   - C_delete for a route door: MaxRoomS, the time until the drive gives up the room.
   - The estimator is a running median, persisted in door_concept's etc/ between runs, and logged with its sample count.
   - The role is read from the LTSM topology summary already in the graph (`ltsm_topology` node): a door that is a
     live face of a passage with crossings ≥ 1 is a route door. No new interface.
3. **Same place.** Matching a new candidate to a memory row uses the tracker's own association: the Mahalanobis gate
   on the door pose with both covariances. That covariance grows with re-localisation uncertainty, so it is not a
   fixed radius. A matched row's L is the candidate's starting evidence ("no door here" pushes it down by exactly its
   L). Rows are one per door place per room, so the table is bounded by places, not by time.
4. **Sensor parameters.**
   - They start from today's values: the ZED `aperture_envelope` shoulders, and `RicohHitPool`'s per-view hit rates.
   - The ricoh envelope starts with the same shoulders applied to its projected fill. That alone gives q ≈ 0 for a
     door overflowing the strip at 1.4 m (tour18), with no refit needed.
   - They are learned online from detections associated to doors above the creation line, persisted across runs,
     and logged.
   - The circularity is accepted and stated: the records come from doors the same model believes in. The
     birth/deletion tests guard against drift.
5. **Reprojection agreement likelihood.**
   - The controls are today's control placements (`door_frame_evidence.h`), and the null is Gaussian with their
     same-frame mean and spread.
   - H1's mean is learned from doors above the creation line, starting from the value the bench gives.
   - The ln-ratio is ln N(x; μ1, s²) − ln N(x; μ0, s²), weighted by vis·κ.
   - Mask and geometry are separate channels of the same view, so the ledger counts each once per view, not per
     frame.
   - Fable's risk 5 stands: this channel is benched on recorded ROI tapes before it is allowed to push L down. Until
     then it can only push L up.
6. **Ricoh projection.** Use the existing equirectangular `CameraAPI` model, as `common/depth_projection` already does,
   with the extrinsics from the graph. No producer (robot_concept) or descriptor change is in scope.
7. **Ledger and memory.** The row stores the aggregate L and the list of view kernels (centre, extent, that view's ℓ),
   which is small: a few per door. A revisit of a stored view replaces that view's ℓ, so there is no double count on
   re-entry. Views from different visits are different views if their time kernel separates them.
