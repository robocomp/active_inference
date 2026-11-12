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

## Rulings

(none yet)
