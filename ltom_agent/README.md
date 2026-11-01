# ltom_agent

Long-term object memory (spec: ltom-agent-design 2026-11-01). Step 2 = OBSERVE the live graph and keep a persisted
repertory file. It never writes an object node, never decides identity, and no concept agent requires it.

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
because its owner agent left, not a death); `[ltom] saved (period|exit): N entries`.

- Config: `etc/config.toml`. ONE graph, `[Agent.dsr]` (the single-graph generated worker also reads the flat
  `Agent.tree/graph/2d`, so both are defined). DDS domain 5 is reserved for the future memory graph (not joined).
- `generated/` is copied from `episodic_agent/` (hand-patched: no `grid` sweep in the destructor, `_Exit` in main).
  Do NOT re-run `robocompdsl` over it without re-applying those patches.
- Run from this directory: `bin/ltom_agent etc/config.toml`. Stop with SIGTERM / Ctrl-C, never `kill -9`.
- Smoke test without touching the live graph: copy the config with `domain = 64` (not committed), plus a seed peer on
  the same domain (the same binary with `configFile = "../robot_concept/shadow.json"`, another id/name).
