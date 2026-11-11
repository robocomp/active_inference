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

(none yet)

## Rulings

(none yet)
