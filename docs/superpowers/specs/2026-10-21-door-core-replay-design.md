# door_concept: a DSR-free core with record/replay

Date: 2026-10-21 · Component: `door_concept/` (branch `feat/interaction-anchor`) · Status: design, awaiting approval

## 1. Why

Ten live three-room tours (tourP1–P10, 2026-10-20/21) showed the new local motion policy working (0 contacts, legs
room1→L-room→room1 completed in P2/P4/P6), and every remaining failure inside door_concept: doors not seen in time,
remembered doors not restored / restored at (0,0), names drifting (door_1 → door_4), the leaf φ estimate collapsing
side-on, a real door ruled out by over-counted absence, and finally (tourP10) a real door REMOVED by the existence
channel four seconds after its birth (silhouette `free=302, occ=0`). Each defect was real and got a tested fix, but each
20-minute tour exposed the next one — the one-run-one-fix loop.

The logic that failed live is untestable today: the existence update (`specificworker.cpp:2144-2848`, ~700 lines), the
tracker (birth/association/merge, `:1197-1790`), identity matching (`:1911-2043`) and the room-switch carry/restore
(`:3747-4415`) all live in `SpecificWorker`, interleaved with DSR reads and writes. `DoorFitter` holds the graph, the
camera API and the mask ingestor. No test exercises any of it. Nothing records door_concept's inputs (the anchor-frame
recorder covers only the anchor update; cortex's graph history cannot replay and misses the media plane).

User decision (2026-10-21): build a door-side harness before more live tours — **core + record/replay** now, synthetic
scenarios later.

## 2. Goals, non-goals, anti-goals

**Goals**
- A `DoorCore`: door_concept's per-cycle decision logic with NO DSR / Qt / Ice / media dependency, driven by one plain
  `DoorCycleInput` struct and producing one plain `DoorCycleOutput` struct (births, removals, merges, door states incl.
  pose/φ/existence, identity-table changes, affordance offers/withdrawals, anchor state, room-switch actions).
- `SpecificWorker` becomes a thin adapter: graph/media → `DoorCycleInput`; `DoorCycleOutput` → graph writes / actuator.
- A recorder (config key, default OFF) that writes every cycle's `DoorCycleInput` to a binary file (WIN1-style: exactly
  what the step received), with a version header.
- An offline replay tool that runs `DoorCore` over a recording and prints/compares the decisions; deterministic.
- The tourP10 false removal and at least two earlier failures become regression tests on recorded data.

**Non-goals**
- Synthetic 3-D scenarios / closed loop with the motion policy (next round).
- Fixing door_concept's perception defects (they will be fixed USING the harness, after it exists — separate tasks).
- Recording the whole DSR graph or the raw media streams.

**Anti-goals**
- A replay that re-implements door_concept's logic in the test (the room_concept lesson: "the bench must run the
  agent's code"). The replay calls the SAME `DoorCore` the agent runs.
- Behaviour changes hidden in the refactor. The extraction is behaviour-preserving; a live run before/after must make the
  same decisions on the same inputs (checked by recording live and replaying through both).

## 3. Design

### 3.1 The boundary
`DoorCycleInput` holds, per compute() cycle, everything door_concept reads (from the recon, 2026-10-21):
- time (steady + wall), cycle index;
- current room id/name, room polygon, walls (pose + extent, parent), recall names announced (ltsm `recall_<room>`),
  the `current`-edge change if any;
- robot pose (room frame) + covariance, robot twist (+ sim stamps), robot speed variance;
- the mask packet as `MaskIngestor` delivers it (labels, confidences, support points camera frame, centroids, bboxes,
  pixels + offsets, quality fields, colour, cam twist, timestamps) AND the zed→room transform resolved at capture;
- the semantic posterior fields;
- the bpearl sweep (room frame + robot frame + origin + capture stamp);
- camera intrinsics (zed) and, for the cycles that use them, the RGB/depth evidence door_concept extracts (the builder
  decides the representation: the derived per-cycle edge/depth evidence if it can be computed outside the core, else
  downsampled crops) — the requirement is that replay reproduces the live decisions;
- foreign claims (other agents' objects), residual field (as read), controller claims (affordance protocol attrs and
  `executing` edges incl. epoch/target), door nodes' own attributes as read back (feedback loop), actuator acks;
- configuration snapshot (once per file).

`DoorCycleOutput` holds every decision the adapter turns into graph writes / actuator calls, plus the diagnostic rows
today's CSV logs carry.

### 3.2 State
All persistent state (instances, beliefs, identity table in memory, ghosts, held/lost doors, carry pending, looks, anchor)
lives in `DoorCore`. The identity table file I/O stays in the adapter (load at start → core; core emits table writes).

### 3.3 Recorder and replay
- `DoorConcept.RecordInputs = false` (+ path, size cap). Recording happens in the adapter just before `DoorCore::step`.
  Files land in the agent's working directory `etc/` (note: the fleet runs from the main checkout; the run scripts copy
  them into the run folder on `collect`).
- `door_concept/tools/door_replay` (standalone, g++ like the other tools): reads a recording, runs `DoorCore`, prints the
  decision stream; options to diff two runs' outputs, to stop at a cycle, and to dump a door's existence/φ trace.
- Determinism: no wall-clock reads inside the core (time comes in the input), no randomness without a seeded generator.

### 3.4 Migration (behaviour-preserving)
Incremental extraction in slices (tracker → existence → identity/room switch → affordances → anchor), each slice:
move code into the core, adapter feeds it, existing tests green, a live smoke (or a recorded replay once the recorder
exists) shows identical decisions. The recorder lands early (after the input struct exists) so later slices are verified
by record-on-old / replay-on-new.

## 4. Validation
- Unit: `DoorCycleInput`/`Output` round-trip through the recorder (version, sizes, locale-independent).
- Replay equivalence: a live recording replayed through `DoorCore` reproduces the live decision stream (births, removals,
  φ within float tolerance) — the acceptance test for the extraction.
- Regression: the tourP10 false removal (record a new three-room run that reaches the door_2 re-birth) as a test that
  fails today; plus a restored-door case and a name-drift case.
- Live: one three-room tour with recording ON after the extraction — same behaviour as before the extraction.

## 5. Constraints
CLAUDE.md: no `DirectConnection`; type-attributed DSR access in the adapter; `std::from_chars`; classic-locale writes;
`cbuild -j8` (check `pgrep -a cc1plus`); SIGTERM only. One session edits door_concept at a time.

## 6. Decisions made by the designer (Claude) — check these
1. **Closed-loop on door_concept's own state, open-loop on foreign agents.** The core keeps its own mirror of what it
   writes (door node attributes, its affordance states, the anchor); recorded inputs carry only what OTHER agents wrote
   (controller claims/executing edges, ltsm recall/current, room geometry, residual, masks, sweeps, poses). Once the core
   changes, foreign reactions to its outputs are not re-simulated — a documented limit of replay.
2. **Regression data come from new tours run by Claude** with the recorder ON (the recorder lands first). The tourP10
   false removal is committed as an EXPECTED-FAILURE test until its fix task. tourP1–P10 data are not usable (partial
   inputs only).
3. **"Identical decisions" = bitwise-equal decision streams** on the same recorded input for pure-refactor slices (the
   core is deterministic); where float evaluation order unavoidably changes, relative tolerance 1e-6 on φ/existence,
   stated per field. To compare against the CURRENT code, the first task adds a decision log to today's SpecificWorker
   (births/removals/merges/φ/existence/offers per cycle).
4. **Asynchronous inputs become events.** Anything that today reaches door_concept between cycles (node-deleted slots,
   actuator acks, any callback-updated state) is latched by the adapter and delivered as timestamped events in the next
   cycle's input; the core API stays a single `step(input)`. The builder enumerates these paths first.
5. **Recording budget ~2 GB per tour; at the cap recording STOPS** (the start of a run — room switches, restores — must
   be kept), with a log line. RGB/depth: prefer the derived per-cycle evidence over images.
6. **Identity-table load is an explicit recorded event** (the table content at load, and the first-Operating sweep), so
   restore / name-drift cases replay exactly.

## 7. Left to the builder
- The exact slicing order and how big each slice is.
- The binary layout of the recording (versioned), compression if needed, and the RGB/depth representation (subject to
  "replay reproduces the live decisions").
- Whether `DoorFitter` becomes part of `DoorCore` or a dependency injected with a camera/mask interface.
