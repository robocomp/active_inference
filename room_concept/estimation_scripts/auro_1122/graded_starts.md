# Graded batch: start set (threerooms-piso, fixture three_rooms_door0) — 2026-11-24

`graded_starts.tsv`: 12 starts, each a Webots pose (x, y, yaw) and a leg plan (`LEGS`), run with
`START="x y yaw" START_ROOM=room_1 LEGS="..." scripts/warm_start_tour.sh <run> crossing/fixtures/three_rooms_door0`.

**All starts are in room_1 (the kitchen).**
- room_concept refuses a warm start in any other room (`initialize_room_model_from_recall`: "only room_1 is supported for now", a non-goal of the warm-start spec: the room node it creates is room_1 / room_id 1).
- Test B1 (a start in room_2) aborted at start-up for that reason, so starts in room_2 and room_3 would need agent code.
- Instead, the starts are spread over room_1, which is large: kitchen, dining area, the corridor to DOOR_1 and the east wing. Each has a different yaw.
- Every route visits room_2 and room_3 and **ends with a crossing back into room_1**. The kitchen is re-entered after a crossing in every tour, but its first visit is at boot.

**How the poses were chosen.**
- From the ground-truth tracks of 169 earlier threerooms tours started at (-5.33, 3.66), each 0.25 m cell was labelled with the true room. The labels use the door-crossing logic of `true_room.py`.
- A candidate cell was visited by at least 4 runs, and so were all 8 neighbours, all in room_1. The robot stood there, so the floor is free.
- Starting from the historical start, the farthest remaining candidate was added each time until there were 12.

**Routes**, cycling through four variants. All 12 end in room_1 after a crossing.
- `room_2 room_3 room_2 room_1`: back through DOOR_1.
- `room_2 room_3 room_1`: back through DOOR_0.
- `room_3 room_2 room_1`: out through DOOR_0.
- `room_3 room_2 room_3 room_1`.

**Reachability test (not graded)**: fleet only, main checkout, no concept agents (`crossing/reach_*`, `crossing/reach_summary.tsv`).
| id | result |
|---|---|
| S01 | historical start and route, not retested |
| S02 S03 S06 S07 S10 | 3/3 legs |
| S04 S05 S08 S09 S12 | 4/4 legs (S12 leg 2: DOOR_2 left open behind, crossing OK) |
| S11 | try 1: leg 2 failed (`open_not_offered` at DOOR_2 from room_3); retry: 3/3 legs |
