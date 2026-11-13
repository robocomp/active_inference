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

### Task 7 — hood_concept

1. **The room now follows the robot** (was: latched at the first room).
2. **Hoods left behind in another room are dropped** (affordance + fitter forget) and recalled by LTOM.
3. **Out-of-room births suppressed** (was: none). WALL-MOUNTED (review focus 1): tolerance = prior depth / 2
   (`AI2PriorDepthM`, else `AI2PriorFootprintM` = 0.60 ⇒ 0.30 m), so a hood whose centre sits just outside the wall line is
   ADMITTED. Checked after the burst take, so a refused candidate's banked cloud is released.
4. **Births held while the working room has no usable outline** (was: admitted).
5. **Outline needs ≥ 3 vertices** (was: hood accepted n ≥ 2 for the wall-flush factor). A 2-vertex outline is now
   treated as none (spec: degenerate ⇒ WAIT). Rooms publish ≥ 4, so this only matters for a malformed outline. The
   polygon is cleared when the working room has none (was: kept the previous room's).
6. **New existence term: out-of-room** (was: none), `HoodModel.ExistenceOutOfRoomGain = 1.5`, tolerance = belief depth / 2,
   per evidence cycle at p_detect = 1, skipping the sensor channels that cycle.

### Task 8 — refrigerator_concept

Same structure as hood (hood was cloned from it); same six differences as Task 7, with these tolerances:
birth = `0.5 · AI2PriorFootprintM` (0.30 m; the agent has no separate depth prior), existence = half the belief's short
side `0.5·min(w,h)`. Key `RefrigeratorModel.ExistenceOutOfRoomGain = 1.5`. Outline now needs ≥ 3 vertices (was ≥ 2).

### Task 9 — cabinet_concept

Same six differences as Task 7 (room follows; left-behind drop; containment; WAIT; outline ≥ 3 vertices, was ≥ 2;
out-of-room existence term `CabinetModel.ExistenceOutOfRoomGain = 1.5`), plus:
- **Both birth paths are checked.** Tracker births: tolerance = `0.5·min(Tracker.BirthWidthM, BirthDepthM)`.
  Residual-born runs (L-corner arms): tolerance = `0.5·min(seed.L, BirthDepthM)`; no outline ⇒ return with the
  debounce INTACT (WAIT keeps the evidence); outside ⇒ the debounce is reset.
- Existence tolerance = half the run's short side `0.5·min(L, d)`.
- Cabinet nodes are `object` type (CLAUDE.md's "`box` nodes" note is stale), so the shared drop helper finds them.
- `tests/kitchen_selftest.cpp` is not a CMake target anywhere (pre-existing); only `cabinet_card_test` runs, and passes.

### Task 10 — shared countable planner core (`common/nbv/countable_planner.h`)

No agent uses it yet (Tasks 11–15). One deviation from the plan's test list: `non_finite_gain_refused` as written
(a raw gain returning NaN) cannot produce a non-finite proposal, because `plan_faces` scores `p_detect·max(0, raw)`
and `std::max(0, NaN)` is 0. The core's non-finite guard is reached by a NaN from `useful()` (e.g. a NaN variance in
the adequacy gap); the test pins that, plus "a NaN raw gain never bids > 0". `plan_faces` is unchanged.

### Task 11 — chair planner on the shared core

`EpistemicProposal` is now `rc::nbv::CountableProposal`; `compute()` keeps its signature, its Target, its raw gain
(continuous D-optimal via `d_optimal_gain<3>` + discrete mode-entropy × view-novelty) and its log line, and calls
`plan_countable` with `useful_fraction(kChairDofs, …)`. **No behaviour difference:** `planner_ab_test` compares it
with a verbatim pre-task copy (`tests/legacy_epistemic_planner.*`) on 22 belief states (fresh priors at 4 yaws;
1/5/30 updates from 2 bearings; mode-ambiguous ×2; one bearing exhausted; 2 bearing-only seeds; 3 near a wall;
2 with a table off the +x face; wall + obstacle; boxed in) — 21 valid proposals, all identical (|Δ| < 1e-4,
gain within 1e-5 relative); the boxed-in state is refused by both. `kExpectedDifferences` is empty.
One cosmetic difference: the throttled `[epistemic-NBV]` log line now prints only for a finite proposal (it printed
before the finite check).

### Task 12 — table planner on the shared core

**No behaviour difference:** `table_concept/tests/planner_ab_test` vs a verbatim pre-task copy, 21 belief states
(fresh ×4; 1/5/30 updates ×2 bearings; near-square mode-ambiguous ×2; long table one bearing ×40; small square wide
orbit; near a wall ×3; chair off the +x face ×2; wall + obstacle; boxed in) — 20 valid, every controller-read field
identical (pose, gain, face_gains, standoff band, framing, chosen_*, vfov, σ*). Two deviations from the plan text:
1. **Table keeps its own `EpistemicProposal` struct** (not `using = CountableProposal`): it carries the object-relative
   constraint the controller reads (face_gains, standoff band, framing, chosen_*, σ*). The core's pose/gain fill it.
2. **⚠ OPEN FOR THE OWNER — two adequacy bounds exist.** Table caps the expected gain at the gap:
   `min(p·raw, gap)`. Chair's `useful_fraction` (now the core's) scales it: `p·raw · min(raw, gap)/raw`. They agree only
   when p = 1 or the gap does not bind. Table passes `useful = 1` and keeps its own `min` after the core, so nothing
   changed; unifying is a behaviour change and a modelling choice (cap the EXPECTED information at what is still
   wanted, vs. discount it by the useful fraction of the RAW information), so it is not made here.

### Task 13 — refrigerator planner on the shared core

**No behaviour difference:** `planner_ab_test` (the table harness adapted: tall 0.7×0.65 box, no discrete mode call),
21 states, 20 valid, every controller-read field identical. Refrigerator already used chair's useful-fraction form,
without the `any_sigma_star` guard; it publishes σ* on every DOF, so the core's guard is inert and the number is the
same. Keeps its own `EpistemicProposal` (object-relative fields), like table. Its early "incomplete sensor model" log is
kept (plan_faces refuses the same case silently). Harness note: the near-wall room is 9 m deep because a 1.7 m box's
stand-off does not fit in the 4 m room the table harness used (all three refused there, by both planners).

## Rulings

(none yet)
