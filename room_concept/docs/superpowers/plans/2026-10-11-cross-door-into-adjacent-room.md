# Plan: cross a door of the apartment into the adjacent room and estimate it (2026-10-11)

**Goal:** after the start room has been estimated and promoted (bdb6095 / 22d763b9), the robot crosses DOOR_1 into the adjacent room, estimates that room with the same pipeline (proto → plain SLAM → map layout → explorer → promotion → refiner), and links the two rooms through the door.

## Where we stand (review of 2026-10-11)

### World
- `piso.wbt` has nothing behind either door; both open onto void.
- `webots-shadow/worlds/piso/tworooms-piso.wbt` adds a 6 × 4 m room north of DOOR_1. DOOR_1 there is at (−4.77, 9.38). Both doors are closed. They can be actuated through the bridge (`setDoorAngle(DEF, rad)`, strip button), but door_concept's own autonomous `open` is not configured (`ActuationProviderId` empty).

### Built and verified (single run, 2026-09-16, Given mode, joystick)
- Door perception in door_concept, as object `door_N`. The leaf angle phi comes from LiDAR only, using the full-field likelihood. It reads 90° open and 0° shut correctly, but the posterior saturates to 0/1.
- Door apertures fed into room_concept; `p_cross` reached 0.99999.
- Proto-room birth `room_<k>` in the aperture frame, with the old room's RT retired and `begin_room_estimation`.
- door_concept's entry mirror; ltsm's live passage metaconcept (stage test).

### Built, never run
- The door pragmatic affordances: approach / open / cross.

### Broken or missing, ranked
1. **The controller cannot plan through a door.**
   - Its grid is the room polygon's bounding box, and everything outside the polygon is occupied.
   - The `cross` target lies outside the room, so it is rejected as OutsideRoom.
   - The door node is an obstacle box across the opening.
   - The promoted Manhattan layout closes straight across the doorway.
2. **door_concept has never run in the Estimate / promotion flow.**
   - It hangs doors on wall nodes, which exist only after promotion.
   - It suppresses births outside the room polygon.
   - `door_open_prob` has been weak or saturated.
   - Without a believed-open door there is no aperture, no `p_cross` and no crossing.
3. **A crossing after promotion does not restart estimation.**
   - `begin_room_estimation` leaves `map_layout_adopted_` / `map_layout_` set, so `map_ready` never returns, the old polygon stays, and the refiner keeps editing room 1 with room-2 scans.
   - Start-up-only paths: `set_map_layout`, the explorer, promotion and wall nodes.
4. **Graph lifecycle conflicts.**
   - ltsm `Promotion.enabled=true` deletes the old room.
   - room_concept's `forget_room` → `birth_startup_proto` → `cleanup_room_graph_nodes` deletes all rooms, including the new one.
5. **Room resolution stays on room 1 until ltsm moves `current`.**
   - The controller and residual keep room 1's polygon.
   - Residual explains everything in room 2 as "outside, wall", so it sees no obstacles there.

## Decisions already on record (do not re-open)
- **Division of responsibility:** ltsm decides room identity, room_concept decides geometry, door_concept owns the door belief. room_concept must not decide about doors.
- **Crossing evidence:** a model comparison, with the aperture as a prior, never a gate. No thresholds; anything unavoidable is flagged.
- **Proto frame** = the crossed door's frame, frozen; promotion happens in place.
- **First test (decided 2026-09-14):** joystick through DOOR_1 in `tworooms-piso.wbt`; topology first. `current` stays on the apartment while room 2 is a proto.
- **Failures are reported, not hidden** (user, 2026-10-11).

## Revision after reading the code (2026-10-11, evening)
- **Defaults taken for the three open questions:** joystick first; DOOR_1 pre-opened in a world copy (`webots-shadow/worlds/piso/tworooms-piso-door1open.wbt`, `position -1.57`); ltsm off, room_concept promotes room 2 itself.
- **Phase 1 cannot be separated from door perception.**
  - The only crossing detector (`step_proto_room_impl`) needs an aperture.
  - Apertures reach room_concept only from door nodes in the graph (`specificworker.cpp:159`, `doors_.refresh`), i.e. from door_concept.
  - A pre-opened door with door_concept off gives no aperture, so no crossing is ever detected.
  - **Consequence:** door_concept must run from Phase 0. Phase 0 therefore also checks that DOOR_1 is born, reads open, and is emitted as an aperture (`tmp/door_filter.csv` shows the stage at which it fails).
- **Phase 1 design: a second start-up, not the old door-proto path.**
  - The whole working estimation pipeline lives in the start-up-proto path: free-space accumulator, layout pool, explorer, promotion, refiner (`birth_startup_proto` → `step_startup_proto` → `promote_startup_proto`).
  - The old door path only calls `begin_room_estimation`, which feeds none of these, and leaves `map_layout_adopted_` set.
  - **On crossing:**
    - mark room 1's RT edge not current, and leave room 1's node and walls in place, unedited from here on;
    - birth `room_k` with the same body as `birth_startup_proto`, minus the sweep;
    - give it a fresh free-space accumulator, an empty layout pool and the explorer re-armed;
    - in RoomConcept: `map_layout_adopted_ = false`, `map_layout_` cleared, a new plain-SLAM map seeded at the current pose;
    - after that, the unchanged start-up machinery estimates and promotes room k.
  - Frame of room k = the crossed aperture frame (decision on record), stored as `T_pub_int_`.
  - **Single-room state** (`dsr_room_id_`, refiner, localiser) is handed to room k; room 1 is left frozen. Nothing updates room 1's polygon after the crossing, which is also Phase 1's "room 1 unchanged" check.

## Phased plan (each phase ends with a measured pass/fail and a commit)

### Phase 0: world and fleet baseline (½ day)
- Load `tworooms-piso.wbt`. **Open DOOR_1 in the world file** (`position −1.57`), so the first crossing does not depend on door perception or actuation.
- Rerun the room-1 estimate from 3 starts in this world. Check that the open door and the room behind it do not break room 1. This is a new case: open-door free space leaking into room 1's outline.
- **Pass:** room-1 shape IoU ≥ the apartment v4 level, and the outline closes across the doorway, or the aperture shows up as a gap.

### Phase 1: room 2 estimation after a joystick crossing (room_concept only; 2–3 days)
- **Entry into estimation:** generalise the start-up path so that a crossing proto re-enters the full Estimate pipeline.
  - `begin_room_estimation` resets `map_layout_adopted_`, `map_layout_`, the layout pool and the free-space accumulator. The new accumulator starts in the proto (aperture) frame.
  - Plain SLAM continues uninterrupted across the door: it is the pose source while estimating.
  - The SDF localiser against room 1 stops at the crossing. The refiner for room 1 freezes: no room-2 scans may edit room 1.
  - The explorer, MDL promotion, wall nodes and refiner run for `room_2`, exactly as for room 1.
- **Lifecycle:** `birth_startup_proto` only when there is no room at all, never deleting rooms another agent owns. Reset `proto_room_id_` / `room_rt_retired_` in `forget_room`. Keep ltsm off, or ltsm Promotion off, for this phase.
- **Test:** joystick from the promoted room 1 through the open DOOR_1, then release.
- **Grade:**
  - room 2's shape IoU against the `tworooms` ground truth;
  - seam consistency: DOOR_1's pose expressed through room 1 and through room 2 must agree, error in cm;
  - room 1 unchanged after the crossing.
- **Pass:** room 2 is promoted, IoU in the same band as the synthetic rooms, seam error a few cm, room 1 IoU unchanged.

### Phase 2: door perception in this flow (door_concept; 1–2 days)
- Run door_concept with the promoted room 1, with wall nodes present, and check:
  - DOOR_0 and DOOR_1 are born on the right walls;
  - phi reads open and shut correctly;
  - `door_open_prob` is sane near the 72° passability boundary;
  - the entry mirror appears in room 2 after the crossing.
- Feed the aperture prior into Phase 1's crossing evidence. The world door is now actuated, not pre-opened: open it with the strip button.
- **Pass:** for one open and one closed door: correct state, an aperture offered only for the open one, `p_cross` rising only when the robot is actually past the line.

### Phase 3: autonomous crossing (controller; 3–4 days, the largest piece)
- **Planner domain:** the grid covers the current room plus every door the robot believes passable, i.e. room ∪ aperture corridor ∪ a stand-off past the door. Model-level: an aperture is free in proportion to `door_open_prob`, not by a hard gate.
- **The door node is not an obstacle while open;** the open leaf's swept footprint is.
- **Execute `aff_door_N_cross`:** approach the stand-off, align with the door normal, pass, and release on the far side. Completion is perceptual (crossing evidence), not by arrival.
- **Selection:** after room 1 is promoted and "finished", the epistemic value of the unseen space behind an open door must rank `cross` above idling. It must come from the same visible-unknown-area gain the explorer already uses, not from the 0.3 placeholder.
- **Test:** the full autonomous sequence from 3 starts in room 1, with DOOR_1 open.

### Phase 4: identity and graph handover (ltsm + all consumers; 2 days)
- Enable ltsm live passage + promotion: `current` moves to room 2 once it is promoted. Room 1 is evicted to memory, or kept, per ltsm's rule.
- The controller, residual and door_concept follow `current`: residual re-binds on `current` change, not only on node deletion.
- **Test:**
  - autonomous crossing;
  - `current` moves;
  - obstacles in room 2 appear in residual;
  - the robot can come back through the door, with re-entry recognised as room 1, not a new room. That needs the place descriptor and is optional for the first pass.

### Phase 5: campaign and thesis (1–2 days)
- 10 starts in room 1 × autonomous crossing.
- **Report:**
  - crossing success (first attempt);
  - room-2 IoU;
  - seam error;
  - time and distance;
  - every failure, with its cause.
- Add a two-room synthetic generator: two rooms joined by a doorway, with random door positions, for generality.

## Risks to watch
- **Open-door leakage.**
  - The free-space ring can flow through an open door before the crossing: room 1's outline swallows part of room 2.
  - The aperture filter (`filter_through_door`) must run first, but it needs door_concept.
  - In Phase 0 with a pre-opened door and no door_concept, expect the leak. Measure it; that is part of the test.
- **False birth on opening a door.** Seen 09-16: the symmetry flip, then a birth while still inside. Phase 2 must check `p_cross` stays low until the robot is past the line.
- **The parked-robot residual ring** is fixed (22d763b9). But crossings involve slow manoeuvres in tight spaces, so watch for it.
- **The passability boundary:** a 1.0 m door at 70° gives 0.67 m clear vs 0.60 + 0.10 needed. Open DOOR_1 fully (90°+) in the tests.

## Open questions for the user
1. **Phase order:** joystick-first (Phases 0–1–2, then 3) as decided 09-14, or go straight for autonomous crossing?
2. **Phase 0:** pre-open DOOR_1 in the world file (decouples door perception), or start with door_concept + actuation from day one?
3. **Phase 1:** keep ltsm off and let room_concept promote room 2 itself, as with room 1? This stays consistent with "ltsm decides identity" only if ltsm later confirms the identity. Or wire ltsm now?
