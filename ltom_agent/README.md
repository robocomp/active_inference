# ltom_agent

Long-term object memory (spec: ltom-agent-design 2026-11-01). Step 2 = OBSERVE the live graph and keep a persisted
repertory file. It never writes an object node (it publishes only `ltom_recall_<uid>` metaconcept offers), never decides identity, and no concept agent requires it.

**State (step 2, write path):** joins the live graph (domain 0), publishes its agent node `ltom_agent 28` via the shared
presence coordinator (no required peers) and, while Operating, polls the graph once per `Period.Compute` (the single
source of the poll period): object instances (`chair_`, `table_`, `bottle_`, `cabinet_`, `refrigerator_` with an
`object_uid`; proto-room objects skipped) become sightings, which `ltom::Repertory` consolidates into entries (lives,
pose keyframes). No DSR signal is connected.

**Files written** (relative to the working directory, paths in `[LTOM]`):
- `etc/ltom_repertory.json` (+ `.prev`, the previous good copy): the store of record. Saved every `save_period_s` when
  changed, and on graceful exit (before the node cleanup). A bad file is moved aside to `*.bad.<epoch_ms>`.
- `etc/ltom_repertory.csv`: read-only view, one row per entry. Columns: `uid, owner, class, name, lives, state, born_ms,
  first_seen_ms, last_seen_ms, end_ms, room_key, x, y, yaw, sx, sy, w, d, h, n_sightings, n_keyframes` (last life, last keyframe).

**Log lines** (stdout): `[ltom] repertory <path>: <how> (N entries)` at start (our own downtime is closed as a gap);
`[ltom] born|reopened|died|owner_exit <name> uid=<uid> room <room_key>` per change (`owner_exit` = the instance vanished
because its owner agent was gone or Degraded, not a death; an instance still on the graph but unresolvable this poll is held, not ended); `[ltom] saved (period|exit): N entries`.

- Config: `etc/config.toml`. ONE graph, `[Agent.dsr]` (the single-graph generated worker also reads the flat
  `Agent.tree/graph/2d`, so both are defined). DDS domain 5 is reserved for the future memory graph (not joined).
- `generated/` is copied from `episodic_agent/` (hand-patched: no `grid` sweep in the destructor, `_Exit` in main).
  Do NOT re-run `robocompdsl` over it without re-applying those patches.
- Run from this directory: `bin/ltom_agent etc/config.toml`. Stop with SIGTERM / Ctrl-C, never `kill -9`.
- Smoke test without touching the live graph: copy the config with `domain = 64` (not committed), plus a seed peer on
  the same domain (the same binary with `configFile = "../robot_concept/shadow.json"`, another id/name).

## Memory window

`ltom_agent` shows what it retains in a graph viewer on its own DSR graph (`[Agent.ltom]`, DDS domain 5, seeded from `ltom_root.json`):
root, then `room_<key>`, then one `object` node per retained object at its newest keyframe (`room_unknown` if it has none).
Click a node to inspect it: `object_uid`, `object_subtype`, size, `state` (live or remembered), and `em_record` for its sightings,
keyframes and lives. "View data" opens the object's retained views, the best per viewing-angle bin (retina cuts the crops, the owning agent picks them). The repertory file stays the store of record; the graph is rebuilt from it at start and nothing reads it back.

## Recall nodes (step 3)

While Operating, each poll `ltom_agent` offers the remembered objects that are missing from the room the robot is in
(`rc::room::current_room`) as **recall nodes**: candidates for a concept agent to resurrect; ltom never decides identity.
- Shape: node type `metaconcept`, name `ltom_recall_<uid decimal>`, attributes `object_uid` and `object_subtype` (the
  class) plus `parent`/`level`; no size attributes (viewers would draw it as a body). RT edge from the room at the last
  keyframe's x, y, yaw (z = 0), `rt_covariance` from the keyframe variances (z 1e-4, roll/pitch the floor prior; unused by consumers).
- Appears when: class in `[LTOM] recall_classes` (default `["chair"]`), the entry's last life is not alive, its last
  keyframe is in the current room (room key = `room_id` attribute, else trailing digits of the name) and its uid is not on
  the graph. A changed keyframe moves the same node (RT pose rewritten, no re-insert).
- Disappears when any of that stops holding (uid alive again, robot in another room, no current room, class unlisted),
  on graceful exit (`[Owns]` has `ltom_recall_*`), and at startup / first Operating entry (stale sweep of `metaconcept`
  nodes named `ltom_recall_*` left by a crashed run). Other nodes are never touched.
- Test: `tests/test_recall_publisher` (DSR domain 63).
