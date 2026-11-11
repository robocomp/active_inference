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

(none yet)

## Behaviour differences

(none yet)

## Rulings

(none yet)
