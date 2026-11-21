# cabinet_concept exit release — report

Branch `fix/cabinet-exit-release` (worktree `active_inference.worktrees/cabinet-exit`), off main `6f8a850b`.
Not pushed, not merged, not run live.

## Status
Built and unit-tested. The live check is still to do (the command is below).

## Commits
- `3d88aaf3` cabinet_concept: release kitchen runs when the robot leaves their room
- this report: a separate docs commit

## Root cause
cabinet runs with `KitchenModel = true`, but the shared release `rc::room::drop_left_behind` was only in the classic
pipeline's `else`. The kitchen cell table was built once over room_1's walls and never re-bound, and
`publish_kitchen_boxes()` used the followed room as the parent. So `insert_or_assign_edge_RT` moved room_1's existing
runs under room_2 and room_3. Evidence (graph_trace.csv lines 1143–1163 and 1408, same node ids) is in
`docs/cabinet-exit-release-diagnosis.md`.

## Fix (mirrors the convention, no new mechanism)
- `compute()` calls `drop_left_behind` at the top, in both pipelines. Its `forget` also clears kitchen bookkeeping.
- `cabinet_kitchen_room.h` adds `kitchen_room_`, the room the cell table was built over. When the robot leaves that
  room (the same `rc::room::left_behind` rule), its run nodes are deleted, the table is reset, and it is rebuilt over the
  new room's walls once that room has a usable outline. Until then, births wait.
- `publish_kitchen_boxes()` writes runs only under `kitchen_room_`, and only while the agent works in that room.
- The graceful-exit cleanup and the startup stale-sweep are unchanged. Kitchen runs have no `aff_*` children; classic
  instances still drop theirs through `forget`. Kitchen runs have no uid, so on return they are learnt again from the
  masks; ltom does not recall them.
- The conformance row and a dated note are added to CONCEPT_AGENT_LIFECYCLE.md.

## Tests
- `cabinet_concept/tests/kitchen_room_test` (new; replays the c3a_1 room sequence): PASS
- `cabinet_card_test`: OK. `kitchen_selftest`: PASS. `common/room_resolve/room_resolve_test`: ALL PASS.
- `cabinet_concept` built at -j8 in the worktree. `countable_concept` was also built there, at -j4 because another
  build was running, so the worktree's nightstand/bed/toilet links resolve. Without it, `concept_agents.sh` with
  `CONCEPT_WT` set to this worktree would skip those three agents.

## Live check (controller session, under the Webots claim protocol)
```
cd /home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts && \
ON_FLEET_UP="LTOM=1 CONCEPT_WT=/home/pbustos/robocomp/components/active_inference.worktrees/cabinet-exit CONCEPT_AGENTS='chair_concept table_concept bottle_concept hood_concept refrigerator_concept cabinet_concept nightstand_concept bed_concept toilet_concept' bash /home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts/concept_agents.sh start /home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/c3a_cab1" \
bash scripts/warm_start_tour.sh c3a_cab1 crossing/fixtures/three_rooms_door0
```
Before the run, empty the ltom repertory the c3a way: write `{"entries":[],...}`. Deleting the json is not enough,
because `.prev` is reloaded. After the run, stop the agents with `concept_agents.sh stop`, which sends SIGTERM.
Only `cabinet_concept`, `nightstand_concept`, `bed_concept` and `toilet_concept` come from the worktree; the other
agents fall back to the main checkout, as in c3a_1.

What to check in the run:
- 4/4 legs.
- `logs/cabinet_concept.log` shows `[room] kitchen model left behind: N run(s) dropped` at each crossing out of room_1.
- `graph_trace.csv` has no `attr parent` or `edge_insert … RT` by agent 21 that names room_2 or room_3 for a
  `cabinet_*` node born under room_1.
- `room_concept.log` has no `foreign node holds it: cabinet_*`.
