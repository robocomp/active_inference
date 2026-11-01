# ltom_agent

Long-term object memory (spec: ltom-agent-design 2026-11-01). Step 2 = OBSERVE the live graph and keep a persisted
repertory file. It never writes an object node, never decides identity, and no concept agent requires it.

**State (Task 1, skeleton):** joins the live graph (domain 0), publishes its agent node `ltom_agent 28` via the shared
presence coordinator (no required peers), reads `[LTOM]`, computes nothing, deletes its node on SIGTERM/SIGINT.

- Config: `etc/config.toml`. ONE graph, `[Agent.dsr]` (the single-graph generated worker also reads the flat
  `Agent.tree/graph/2d`, so both are defined). DDS domain 5 is reserved for the future memory graph (not joined).
- `generated/` is copied from `episodic_agent/` (hand-patched: no `grid` sweep in the destructor, `_Exit` in main).
  Do NOT re-run `robocompdsl` over it without re-applying those patches.
- Run from this directory: `bin/ltom_agent etc/config.toml`. Stop with SIGTERM / Ctrl-C, never `kill -9`.
- Smoke test without touching the live graph: copy the config with `domain = 64` (not committed), plus a seed peer on
  the same domain (the same binary with `configFile = "../robot_concept/shadow.json"`, another id/name).
