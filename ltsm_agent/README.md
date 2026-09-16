# ltsm_agent — long-term spatial memory, on its own DDS domain

One process, **two DSR graphs, two DDS domains**:

| graph | domain | config | role |
|---|---|---|---|
| `Graphs.at("dsr")` (= `G`) | 0 | `configFile = ""` | the live shared working graph — **joined** |
| `Graphs.at("ltsm")` | 2 | `configFile = "ltsm_root.json"` | long-term spatial memory — **seeded and owned** |

## Why it exists

The "crossing door" problem needs somewhere to put a room the robot has left. When the robot
crosses a door it loses the corners and walls that were giving it its pose and falls back to dead
reckoning; once the LiDAR and the ricoh see the new room, layout estimation takes over, and that
point is kept as the last reference in the old room and the new zero in the new one. After the new
layout stabilises the robot briefly carries **two RT edges — one to room 0, one to room 1**, and
something has to evict the old room, move it into memory, and leave the door as the connecting
element with coordinates in *both* frames.

This component is that evictor. Step 1 — both graphs open on their own domains — is what the rest
of this file describes. **Step 2, the eviction transaction itself, is in
[EVICTION.md](EVICTION.md)**: how the seam between the two rooms is measured in the one window where
it is computable, what memory's RT tree looks like afterwards, and why the fleet lets go of its own
nodes instead of this agent deleting them (the LET-GO RULE). Read that before touching
`room_eviction.{h,cpp}`.

## Running

```bash
cd .../active_inference/ltsm_agent
bin/ltsm_agent etc/config.toml            # both domains; needs the fleet up (see below)
bin/ltsm_agent etc/config_selftest.toml   # + the isolation regression test
bin/ltsm_agent etc/config_solo.toml       # memory graph only, no fleet needed, cannot disturb one
bin/ltsm_agent etc/config_stage.toml      # + EVICTION, against a synthetic live graph on domain 3
bin/ltsm_agent etc/config_stage_proto.toml # + LIVE PASSAGE / proto-room, synthetic graph on domain 3
```

`config_stage.toml` is how eviction is developed and regression-tested **before `room_concept` can
hand over two rooms**: the `dsr` graph is seeded from `etc/stage_two_rooms.json`
(`tools/make_stage_fixture.py` authors it and prints the ground truth) on domain 3, which is
nobody's, so it cannot reach the fleet. Going live is a config change and nothing else.

Paths in the config go straight to `QFile`, so they resolve against the **CWD** — always run from
the component root. With `etc/config.toml` only the memory window comes up (`ltsm_agent-19|ltsm`);
the working graph has no viewer here because `robot_concept` already shows it. The test configs
still open `ltsm_agent-19|dsr` too, since on their private domains nothing else shows that graph.

Stop with SIGTERM / Ctrl-C, **never `kill -9`**.

## Things a reader will otherwise get wrong

- **`G` aliases the *live* graph, not memory.** `getSurNames()` sorts
  (`classes/ConfigLoader/ConfigLoader.cpp:277`) and the generated code does
  `G = Graphs.at(surNames.front())`; `"dsr" < "ltsm"`. `initialize()` asserts this rather than
  trusting it. If you rename the sub-tables, keep the live one alphabetically first.
- **Never add a config table whose flattened key contains `Agent` and has ≥ 2 dots** (e.g.
  `[ViewAgent.anything]`). `getSurNames` filters on `k.contains("Agent")` and would build a third,
  phantom graph out of it.
- **Every sub-table must define `tree`, `graph` and `2d`.** `setupViewer` reads all three
  unconditionally and a missing key *throws* (`ConfigLoader.tpp:11`). `3d` is read by nothing.
- **`generated/genericworker.cpp` is hand-patched — re-running `robocompdsl` reverts it.** The
  stock `~GenericWorker()` deletes every `"grid"`-typed node from *every* graph it holds. One of
  ours is the live shared graph, and `residual_concept` publishes a node named `residual` whose
  type is `grid_node_type` (`residual_concept/src/specificworker.cpp:1683`) — so the stock
  destructor would delete another agent's live node on every shutdown. The body is now empty.
- **The JSON attribute switch is written TWICE — nodes at `dsr_utils.cpp:105`, edges at `:180` — and
  a fix landed on one copy only.** uint64 (type 7) on the NODE path goes through
  `toString().toULongLong()` and round-trips correctly (commit `af03fe6`); the EDGE path still read
  `QVariant::toUInt()`, 32 bits, so any uint64 carried on an EDGE in a JSON was silently zeroed —
  and every uint64 in the registry is a timestamp or an id, i.e. exactly the values that cannot
  survive truncation. door_concept fixed the edge twin and it landed in the 2026-09-13 reinstall. The
  node path was never affected, so `timestamp_creation` in a seed file is fine.
- **`std::vector<float>` (type 3) has no comma repair in either copy**, unlike the scalar float and
  double cases which do `toString().replace(",", ".")`. Latent rather than live: DSR writes its JSON
  through `QJsonDocument`, which is locale-independent, so a file this fleet wrote cannot contain
  comma decimals. It is only a hazard for a JSON written by something that is not Qt's writer —
  worth knowing before hand-editing a graph file on a machine running `LANG=es_ES.UTF-8`.
- **`insert_node(std::move(n))` compiles and does not link.** The forwarding-reference overload
  deduces `No = DSR::Node` from an rvalue, and libdsr only instantiates `insert_node<Node&>` and
  `insert_node<Node&&>` (`cortex/api/dsr_api.cpp:597-598`) — so the rvalue form is an undefined
  reference at LINK time, with no compile error to point at the call. Pass an lvalue.
  `insert_or_assign_edge(std::move(e))` is fine: `Edge` has the `<DSR::Edge>` instantiation too.
- **On startup the working memory is SEEDED into long-term memory.** Every `room` node in the live
  graph that memory does not already hold is written into memory as `r<idx>_<name>` with its
  STRUCTURE — floor, walls and every door that exists at that moment, under the same RT chain, so each
  door keeps its pose in its room's frame. Furniture and affordances are not copied (the eviction
  copies what is really there at the fence). Read-only on the live graph and NOT gated on
  `[Eviction] enabled`. A second room present at startup hangs off the first with its pose read through
  the live tree. ★ "Already held" is decided BY NAME (`r<room_id>_<live name>`) — the only identity key
  until place recognition exists. With persistence off (default) memory starts empty and every room is
  written; with it on, a different room that `room_concept` happens to name the same (today every room
  is literally `room`) would be taken for the remembered one. The seed and a later eviction share one
  memoised room numbering, so the eviction FILLS the seeded nodes rather than duplicating them.
- **Window size, dock layout and graph ZOOM are persisted** in `~/.config/RoboComp/ltsm_agent.conf`
  under `windows/<agent id>/<graph name>`. Geometry/state come from the same `save/restore_window_settings`
  block the rest of the fleet's generated workers carry (ported into `generated/genericworker.cpp`,
  which lacked it). The zoom is ours (`SpecificWorker::save/restore_graph_zoom`) and is kept ONLY when
  the user chose it (wheel or drag); "Fit graph to view" hands the view back to auto-fit and the key is
  removed on exit. Restoring it needs two workarounds for cortex's `GraphViewer`, both explained at
  the call site: a synthetic wheel notch to set its private `user_framed_` latch, and a second apply
  after the refit already in flight, whose timer lambda (`graph_viewer.cpp:52-56`) ignores that latch.
- **Proto-rooms are not rooms to this agent until promoted.** A `room` with a `proto` self-edge
  (room_concept's provisional room after a door crossing) is ignored by eviction, by
  `ensure_current_edge` and by the startup seed — `current` stays on the room the robot came from. While
  it exists, `src/passage_live.{h,cpp}` (`[LivePassage] enabled`) keeps ONE live `metaconcept`
  `passage_<n>` matching door_concept's entry mirror under the proto-room to the current room's nearest
  door, and deletes it when either door goes. ltsm_agent runs no presence monitor, so ownership is
  enforced in code: stale sweep at startup, removal on graceful exit, LIVE graph only.
  ⚠ The stage configs use memory domain 2 like the live config: never run one while a live ltsm_agent
  is up (same domain, same agent id).
- **The memory view is re-laid out with graphviz `twopi`** whenever a memory node is added or deleted
  (coalesced 100 ms). The viewer-only `collapsed` flag room_concept puts on the floor is NOT copied into
  memory, so walls — and the doors hanging from them — stay visible.
- **Memory persistence exists but is OFF.** `[Memory] persist = false` — armed, the graph is written
  at the eviction fence and on a graceful exit (`src/memory_store.{h,cpp}`: temp file + atomic
  `rename`, previous generation kept as `.prev`). Left off because there is nothing yet whose value
  is accumulation; arm it the day the first passage statistic is written, not on a date. Disarmed,
  the LTSM domain is re-seeded from `ltsm_root.json` every launch and a restart wipes it.
- **The DSR side needs a graph server already up.** With `configFile = ""` the graph is requested
  over DDS and `qFatal`s after `TIMEOUT * 3` = 15 s if nobody answers (`dsr_api.h:48`,
  `dsr_api.cpp:1376`). `robot_concept` is that server. Use `etc/config_solo.toml` when the fleet is
  down. This abort happens inside the `DSRGraph` constructor called from `GenericWorker`'s, so
  there is no hook to catch it; a real pre-flight would mean changing cortex.
- **Do not copy `robocomp-shadow/agents/agent_dual_dsr`.** It looks like the dual-domain example
  and is not: it passes `1` / `2` into the `DSRGraph` ctor's **5th** parameter, which is
  `bool all_same_host`, not `domain_id` (`dsr_api.h:73`), so both its graphs land on domain 0.
- **Graphs are looked up BY NAME, and a single-graph config needs BOTH shapes.** `Graphs` is keyed
  by the `[Agent.<name>]` sub-table names — except in the flat single-instance form, where
  `genericworker.cpp` files the one graph under `""`. Separately, `setupViewer`'s prefix is
  `"Agent"` and only becomes `"Agent.<name>"` when `Graphs.size() > 1`
  (`generated/genericworker.cpp:228-230`), so a **single**-graph config must ALSO carry flat
  `tree`/`graph`/`2d` under `[Agent]`. Miss either and you get an exception thrown from inside a
  Qt event handler, which prints only `Qt has caught an exception thrown from an event handler`
  (plus `error: unordered_map::at` for the first case) and then terminates. `initialize()` now
  fails with a message naming the graphs it actually found instead of throwing from `.at()`.
  `etc/config_solo.toml` documents both quirks in situ.
- **`generated/main.cpp` is hand-patched too — same regeneration warning.** It ends with
  `std::_Exit(status)` rather than `return`, to leave the process *before* C++ static destruction.
  libIce 3.7 destroys its own globals in the wrong order: at exit `_dl_call_fini` runs libIce's
  finalizer, a static object in it calls `IceInternal::FactoryTable::removeExceptionFactory()`, and
  that locks a mutex an earlier finalizer already destroyed. `pthread_mutex_lock` returns `EINVAL`,
  `IceUtil::Mutex::lock()` (`Mutex.h:295`) throws out of a destructor, and every clean shutdown died
  with `terminate called after throwing an instance of 'IceUtil::ThreadSyscallException'`. The whole
  fault is inside libIce's own finalizer chain, so it cannot be fixed by reordering anything here.
  Nothing of ours is skipped: `Ice::Application::run()` ends with `delete worker`, so both DSRGraphs
  and both DomainParticipants are already gone (the `Removing DSRParticipant` lines print first) and
  Qt is torn down. Verified no shared-memory leak: 99 `fastdds` segments in `/dev/shm` before and
  after. **The mechanism is in libIce, so every RoboComp component here is likely affected — this
  fix is worth lifting into the generator template rather than per-agent.**

## Not a concept agent

`CONCEPT_AGENT_LIFECYCLE.md`'s conformance audit covers agents that CREATE / UPDATE / REMOVE
*perceived* object nodes. This one perceives nothing and creates no object nodes, so it takes no
row in that table.

## Verified

Against the live fleet on 2026-09-12, with `robot_concept` up:

- both graphs load — `dsr` from the network (10 nodes with only `robot_concept` up, 44 with the
  fleet up), `ltsm` 1 node (`root`, id 1000) from file;
- **one pid listening on two disjoint RTPS port blocks**, `7400/7412/7413` (domain 0) and
  `7900/7912/7913` (domain 2 = `7400 + 250*2`) — `ss -lunp | grep pid=<pid>`;
- the self-test wrote a node per cycle into memory (`ltsm.size` 2→9) while the live graph stayed
  pinned at 44, with `leak(ltsm->dsr)` and `leak(dsr->ltsm)` both false throughout;
- an independent `pydsr` probe of domain 0 after shutdown showed the shared graph unchanged — no
  `ltsm_probe_*` leaked in, nothing deleted;
- `etc/config_solo.toml` runs to its time limit with no exception, confirming the memory-only path;
- **clean shutdown on both SIGTERM and SIGINT** — exit code 0, no `terminate`, no new core dump, the
  log ending at `Removing DSRParticipant`;
- **the destructor patch exercised for real**: with `residual_concept` up, the shared graph holds a
  live `grid`-typed node named `residual`, and it survived two `ltsm_agent` shutdowns. The stock
  generated destructor would have deleted it.
