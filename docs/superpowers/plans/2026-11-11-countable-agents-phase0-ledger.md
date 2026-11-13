# Countable concept agents Phase 0 — ledger

Branch `feat/countable-phase0` (base 8a589262). Plain running record; later tasks append.

## Baseline rates

Live baseline (parked + one door crossing, three-room world) is run later by the controller with the user.
"Provisional" is the last `Fps` / `[CTRL] fps=` line found in
`room_concept/estimation_scripts/crossing/chat_e2e9/logs/<file>` (main checkout); not a substitute for the baseline.

| agent | Fps (live baseline) | source | provisional (from chat_e2e9 logs) | provisional source |
|---|---|---|---|---|
| chair_concept | TBD | live run | none in log | chair_concept.log |
| table_concept | TBD | live run | no log | - |
| bottle_concept | TBD | live run | no log | - |
| hood_concept | TBD | live run | no log | - |
| refrigerator_concept | TBD | live run | no log | - |
| cabinet_concept | TBD | live run | no log | - |
| room_concept | TBD | live run | 20 (period 50 ms, cpu 60-64%) | room_concept.log |
| controller | TBD | live run | 19.1-19.5 (period 51-52 ms, cpu 33-34%) | controller.log |
| retina | TBD | live run | 40 (period 25 ms, cpu 137-148%) | retina.log |

## Builds

Unmodified agents in the worktree, `systemd-run --user --scope` (MemoryHigh 40G / Max 44G / Swap 8G), `cmake -B build && make -C build -j8`.

| agent | result |
|---|---|
| chair_concept | OK (exit 0) |
| table_concept | OK (exit 0) |
| bottle_concept | OK (exit 0) |
| hood_concept | OK (exit 0) |
| refrigerator_concept | OK (exit 0) |
| cabinet_concept | OK (exit 0) |

## Findings (Task 2)

- chair_39 was born in run rose_chair3 (2026-11-09) 2.8 s after the leg room_1 -> room_2, at room_2 (-2.49,-0.66) = world ~(-6.87,4.02): a genuine ROOM_1 chair seen back through the open door. Later lives are LTOM recalls of the same uid, not new births.
- No INACTIVE / no-polygon line preceded the birth (grep count 0 in both logs); the gate is never inactive after startup because the fitter keeps the previous room's polygon when the new room has none (refresh_room_geometry returns early).
- Two causes remain unsplit by the logs: (c1) stale room_1 polygon kept after the room switch (room_1 bbox x[-4.25,4.25] y[-4.63,4.63] numerically contains the point), or (b/c2) room_2's own recalled/proto outline contained it. Polygon values are not logged anywhere.
- Hypothesis (a) of spec G2 (prior INACTIVE, no outline) is not supported.
- To settle it live: log outline bbox + room id on change and the inside verdict at each birth.

Owner 11-11: continue; WAIT rule stands; cause to be pinned live (outline bbox + room id logged on change, inside verdict logged at birth — Task 4).

## Behaviour differences

### Task 4 — chair_concept

1. **Births are held while the working room has no usable outline** (was: admitted; worse, judged against the PREVIOUS
   room's polygon, which `refresh_room_geometry` kept on a room switch — the stale-polygon cause c1 of chair_39).
   Mechanism: `InstanceTracker::update(..., hold_births)` — candidates keep maturing and keep their evidence; the
   first detection after the outline arrives is born at once. Pinned by `chair_tracker_defer_test`. Only when
   `Existence.RoomPrior` is on (default), as before.
2. **The fitter's room polygon is cleared when the working room has none** (was: kept the previous room's). Side
   effects, all of which already guard on `has_room_polygon()`: no out-of-room existence term, no wall line-of-sight in
   P(detect), the NBV gets no reachable region, and recall uses the default room area — all for the interval until the
   outline arrives.
3. **New logs:** (room id, outline verts + bbox) on every change; every birth logs its inside/OUTSIDE verdict with the
   room id; `births HELD` / `births released` per WAIT episode. These pin the chair_39 cause live (owner 11-11).
4. Unchanged: Admit/Suppress wherever an outline exists (same rule — inside or within `Existence.RoomMarginM` of the
   boundary — now `rc::room::birth_verdict`); the left-behind drop (now `rc::room::drop_left_behind`, same teardown,
   same log text); the existence term value (`out_of_room_logodds(false, gain) = -gain`).
5. Shared tracker change: `hold_births` defaults to false, so the other five agents are byte-for-byte unchanged until
   their own tasks pass it.

### Task 5 — table_concept

1. **The room now follows the robot** (was: latched at the first room for the whole run; tables born after a crossing
   were parented to the room just left). Uses `rc::room::follow_current_room` at the top of `compute()`.
2. **Tables left behind in another room are dropped** (`rc::room::drop_left_behind`, teardown = affordance + fitter
   forget) and recalled by LTOM on return (was: never dropped — kept and fitted in the wrong frame).
3. **Out-of-room births are suppressed** (was: none). Tolerance = half the birth footprint's short side
   (`0.5·min(Tracker.BirthWidthM, Tracker.BirthDepthM)` = 0.30 m), a physical dimension.
4. **Births are held while the working room has no usable outline** (was: admitted). The polygon is dropped on a room
   change and re-read each cycle until the new room publishes one (was: kept the old room's until the id changed, then
   retried). While empty: no wall line-of-sight in the silhouette test, no NBV reachable region, default recall area.
5. **New existence term: out-of-room** (was: none). A belief centre outside the outline by more than half its short side
   draws `-ExistenceOutOfRoomGain` (1.5, chair's value) per EVIDENCE cycle at p_detect = 1, and skips the sensor
   channels that cycle — as chair. Key is `TableModel.ExistenceOutOfRoomGain`, not the plan's `Existence.OutOfRoomGain`:
   table keeps all existence keys under `[TableModel]`. It runs only on evidence cycles because
   `update_and_remove` returns early when no channel has a fresh frame.
6. Logs: (room, outline bbox) on change, each birth's inside/OUTSIDE verdict with the room id, HELD/released.

### Task 6 — bottle_concept (+ shared `room_of`)

0. **Shared helper change, found here:** a bottle's `parent` is the table it stands on, not the room, so the Task 3
   `drop_left_behind` would have dropped EVERY bottle on a table. `rc::room::room_of` (room_follow.h) now walks
   `parent` up to the `room` node before `left_behind` is asked. For chair/table (parent = room) nothing changes. A
   parent that no longer exists is returned as-is ⇒ left behind ⇒ dropped (review focus 2). A bottle whose table was
   dropped as left behind therefore drops too, the next cycle.
1. **The room now follows the robot** (was: latched at the first room).
2. **Bottles left behind in another room are dropped** (teardown = affordance + fitter forget) and recalled by LTOM.
3. **Out-of-room births suppressed** (was: none); tolerance = `prior_radius` (0.035 m).
4. **Births held while the working room has no usable outline** (was: admitted). The polygon is now the working
   room's own and is cleared when it has none (was: kept the previous room's).
5. **No out-of-room existence term — skipped as the plan says.** Why: a bottle's position is set by its support surface
   (a table that is itself containment-checked) and its birth is checked; an out-of-room bottle after birth would need its
   table to have left the room first, which the left-behind drop already handles. Open for the owner if wanted.
6. Pre-existing, not Phase 0: `bottle_concept/tests/test_bottle_fit.cpp` does not compile on main either (uses
   `BottleState`/`BottleModel` without `rc::`); untouched. `bottle_card_test` and `test_bottle_contour` pass.

## Rulings

(none yet)
