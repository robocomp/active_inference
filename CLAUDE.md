# active_inference — cortex agent guidelines

These are hard-won rules for the DSR/cortex agents here (robot_concept, room_concept,
controller, retina, bottle_concept, …). Violating them causes nondeterministic
startup crashes that are very hard to diagnose. Mirror bottle_concept (known-good).

**Robot frames & sensor mounts: see [ROBOT_GEOMETRY.md](ROBOT_GEOMETRY.md)** — the frame tree
(root→Shadow→body→sensors), which frame is the localization frame (`Shadow`, type `"robot"`) vs the
de-facto base frame everyone reads (`body`, +4.5 cm z), and each sensor's mount + purpose (incl.
`bpearl` = downward dome, not a floor sensor). Derived from `robot_concept/shadow.json` (the truth).

**Agent lifecycle: see [CONCEPT_AGENT_LIFECYCLE.md](CONCEPT_AGENT_LIFECYCLE.md)** — the NORMATIVE contract for
what a concept agent may do to the shared graph: CREATE · UPDATE · REMOVE · HISTORY · OWNERSHIP, which shared
`common/` module owns each stage, and the invariants (birth is tracker-only; a frame must be resolvable +
centred + still to move geometry; occupancy confirms / absence removes / occlusion HOLDS, with absence weighted
by P(detect) and line-of-sight including WALLS; no constant process noise; `[Owns]` must match `[Agent].id`).
It carries a per-agent conformance audit — add a row and justify every ❌ before shipping a new agent. Theory
for the HISTORY stage is in [MODEL_HISTORY.md](MODEL_HISTORY.md).

## Modeling philosophy: no thresholds; find the Active-Inference solution

DO NOT add thresholds (gates/clamps/`if (x > k)`/σ-floors/magic cutoffs) if not
absolutely necessary. Try to find an Active-Inference-aligned solution first: encode the
effect in the generative model and let it fall out of the probabilistic inference —
typically as a continuous covariance/precision that grows or shrinks with the right
physical covariate, not a hard switch. Examples from this codebase: a distant view loses
pose/orientation information via a common-mode covariance that grows with range (not a
range gate); correlated mask points stop collapsing σ via Woodbury common-mode
marginalisation (not a σ-floor); ego-motion downweights masks via the interaction-matrix
variance added to R (not a motion gate). If a threshold is truly unavoidable, flag it and
justify why no model-level term works. See `table_concept/TABLE.md`.

## DSR graph-update signals: connect QUEUED, never DirectConnection

DSR emits `update_node_signal` / `update_edge_signal` (with `std::string` payloads)
from the **raw FastDDS reader threads**. How you connect to them matters:

- **`Qt::DirectConnection` → CRASH.** The slot runs *on the FastDDS reader thread*.
  Under live-peer graph churn this corrupts the heap; the visible symptom is a smashed
  AgentInfo heartbeat — SIGSEGV in `DSR::DSRGraph::get_agent_id` on a garbage
  `DSRGraph this`, fired ~1s after start once required peers come online. The
  backtrace points at the heartbeat (the victim), NOT the real cause.
- **Plain / default / `Qt::QueuedConnection` → SAFE.** The slot body is marshalled to
  the main thread. Verified: the controller crashed 100% with DirectConnection and ran
  5/5 clean with QueuedConnection (2026-06-23). A generated agent's connects are
  commented out by default; when you uncomment one, the default is already Queued
  (cross-thread Auto resolves to Queued) — so just never add `Qt::DirectConnection`.

Note: even Queued copies the `std::string` args on the emitting (DDS) thread at emit
time, and the presence MONITOR is poll-based on purpose. If you don't need a signal,
don't connect it at all (bottle_concept connects none; its modify_*_slot are empty).

## DSR attribute access: type-attributed only, never runtime_checked

Do NOT use the `runtime_checked_*` graph attribute methods (e.g.
`runtime_checked_add_or_modify_attrib_local(node, "foo", val)`). They take the attribute
name as a *string* and validate the type at runtime — a typo or type mismatch is a runtime
throw, not a compile error. Use the **type-attributed** template forms instead:
`add_or_modify_attrib_local<foo_att>(node, val)`, `get_attrib_by_name<foo_att>(node)`, etc.
These are checked at compile time (`valid_type<name, Ta>()` requires-clause) and self-document.

To add a NEW attribute the process is:
1. Add a `REGISTER_TYPE(foo, <c++ type>, false)` line to cortex's
   `core/include/dsr/core/types/type_checking/dsr_attr_name.h` (this generates the `foo_att`
   struct). For a large read-mostly blob, register it as
   `std::reference_wrapper<const std::vector<T>>` (zero-copy on read); you still SET it by
   passing a plain `std::vector<T>` by value — `valid_type` unwraps the reference_wrapper.
2. **The user reinstalls cortex** (the header is root-owned under `/usr/local/include/dsr/…`);
   do not `sudo cp` it yourself. Then rebuild the agent so its TUs see the new `foo_att`.
   (Code using `<foo_att>` won't compile until that reinstall lands — expected.)

Existing `runtime_checked_*` call sites (e.g. the `masks` node in
`retina/src/graph_publisher.cpp`) predate this rule; migrate opportunistically, don't add
new ones.

## DSR graph thread-safety — the actual rules (verified against cortex source 2026-07-11)

The old blanket rule here ("never touch the graph off the main thread — it corrupts it")
was over-broad. Verified against `cortex/api/dsr_api.cpp` + `dsr_inner_eigen_api.cpp`:

- **`get_node`/`update_node`/`get_nodes_by_type`/attrib R/W are thread-safe.** `DSRGraph`
  serializes all of them with a `std::shared_mutex _mutex` (reads take `shared_lock`,
  writes `unique_lock`). Safe from ANY thread, including concurrently during the join —
  the mutex serialises it. Graph-API access itself does not corrupt the heap.
- **`inner_eigen->get_transformation_matrix(dst,src,ts,...)` returns `std::optional` and
  bails to `{}` at every missing node/edge/rtmat in the RT chain.** It never null-derefs
  itself; the crash mode is a CALLER doing `.value()`/`->` on the nullopt
  (`bad_optional_access`). ALWAYS check the optional. Asking `zed<-room` when `room` is
  gone safely returns `nullopt`.
- **The one real cliff: `InnerEigenAPI`'s ts==0 cache.** `get_transformation_matrix` with
  `timestamp==0` sets `use_cache=true` and reads/writes an UNLOCKED `std::map` cache +
  `node_map`; the invalidation slots (`remove_cache_entry`) erase the same maps. No mutex.
  So the ts==0 path is safe ONLY single-threaded per instance. It's safe today because all
  our ts==0 calls (static `robot->zed`/`robot->ricoh` extrinsics, latest-pose viewer read)
  run on the main thread and the invalidation slots are `Qt::QueuedConnection` onto that
  same thread. A real timestamp (`ts!=0`) → `use_cache=false` → touches NO cache → fully
  thread-safe.

**Rules for putting graph work on a worker thread** (e.g. a ZED/LiDAR YOLO worker):
1. Reads/writes/`update_node` are fine off-thread.
2. Pin transforms to the **capture timestamp** (`ts!=0`) — correct pinning AND no cache.
3. If you must use ts==0 off-thread, give that thread its OWN `get_inner_eigen_api()`
   instance (it returns a fresh `unique_ptr` per call) — never share one instance's ts==0
   path across threads. Caveat: a per-thread instance created on a raw `std::thread` has no
   Qt event loop, so its queued invalidation slots never fire ⇒ its ts==0 cache goes stale.
   One more reason workers should use real timestamps.
4. Always check the returned optional.

**The real cross-thread hazard is `cv::Mat`, not DSR.** A `cv::Mat` copy is a refcounted
shallow handle; sharing one buffer across threads where one side writes/draws
(`addWeighted`/`fillPoly`/`clone`-that-you-forgot) is a heap smash whose victim surfaces
later as an unrelated crash (OpenCV, next alloc, ONNX). DEEP-COPY every `cv::Mat` at a
thread boundary; never draw into a shared frame. (This — not graph access — is what the
2026-07 cores point at; see [[opencv-fillpoly-viewport-clip]] / ricoh-compose SIGSEGV.)

Legacy DSR blob uploads (laser_*/cam_*) from reader threads must stay gated OFF (media
plane only). The `Qt::DirectConnection` crash (slot on the DDS reader thread) is a
SEPARATE, still-valid hazard — see the section above.

## Media plane (zero-copy DDS) consumer pattern

- **Use the shared factories — every agent initializes subscribers the SAME way.** In
  `common/media_transport/media_transport.h`:
  `rc::media::make_lidar_subscriber_from_graph(G, node, key)` and
  `make_image_subscriber_from_graph(G, node, key)`. They read the descriptor, verify the
  node+stream exist, create+init the typed subscriber on the descriptor's domain/topic,
  log uniformly, and return the subscriber or `nullptr`. Do NOT hand-roll the
  descriptor→config→init dance in an agent; call the factory.
- Domain + topic come ONLY from the media descriptor JSON the producer authors on the
  sensor node (`zed`/`lidar3D`/`imu`). No config entry for domain/topic. The media plane
  uses a DEDICATED DDS domain (7), isolated from the DSR `Agent.domain` (0) — never
  create a media participant on domain 0.
- Producer (robot_concept) advertises a PER-NODE descriptor: rgb+depth→`zed`,
  lidar→`lidar3D`, imu→`imu`; publishers + advertise run on the main thread in
  `initialize()` before reader threads start.
- Consumer creates the subscriber AFTER the graph is loaded and the sensor node is
  verified to exist — on the main thread before starting worker threads (controller), or
  lazily from the already-Operating compute thread (room_concept LidarIngestor / the
  CameraVisualizer drain timer). Never lazily from a free-running thread at t=0.
- LiDAR is media-only now; there is no DSR `laser_*` fallback in consumers.

## Presence protocol (copy bottle_concept exactly)

- Degraded must DEBOUNCE: `on_degraded_enter` must NOT cleanup+exit immediately. A
  transient required-peer flap at startup (handshake / DSR churn / peer restart) fires
  `presenceLost` momentarily; tearing down deletes the agent's own node and exits
  cleanly (symptom: `[Graph] node '<agent> <id>' deleted` right after `monitor started`,
  then a clean exit — no segfault). Instead schedule a grace timer (~3000ms) and shut
  down only if `presence_coordinator_.all_required_ready()` is still false.
- Each agent's `[Agent] id` must be unique across the shared graph (id collision =
  CRDT actor clash → SIGSEGV).
- Log on exceptional state transitions (Degraded/Emergency) and on a consumed stream
  stalling (enter a local emergency hold + stop the robot rather than acting on stale data).

## Stopping an agent: SIGTERM/SIGINT, NEVER `kill -9`

These agents own DSR nodes in the SHARED, persistent graph (e.g. cabinet_concept's `box`
nodes named `cabinet_*` + their `aff_*` affordance children). Cleanup runs ONLY on a
GRACEFUL exit: `main.cpp` traps SIGINT/SIGTERM → `QCoreApplication::quit()` → the
`SpecificWorker` destructor → `request_shutdown()` → `cleanup_owned_nodes()`, which deletes
the agent's own nodes (and `G->reset()` tears down the DDS participant cleanly).

- **Stop with `kill` (SIGTERM), `kill -INT`, or Ctrl-C.** Verified: a graceful stop logs
  `node <id> removed from DSR, destroying instance` per owned node, then
  `[Graph] node '<agent> <id>' deleted`, and the graph is left clean.
- **`kill -9` (SIGKILL) CANNOT be caught** → the destructor never runs → every node the
  agent created LEAKS into the shared graph and lingers with NO process alive (symptom the
  user sees: "stale cabinet nodes still there but the agent isn't running"). The next
  startup's stale-sweep is the only thing that reaps them.
- **Startup stale-sweep is the safety net for a crashed/SIGKILLed previous run.** Sweep
  `get_nodes_by_type("box")` filtered by `name.starts_with("cabinet")` BOTH in `initialize()`
  AND once on the first `on_operating_enter` (guarded) — the `initialize()` pass can run
  BEFORE leaked nodes finish syncing from the persistent server, so the post-sync Operating
  pass is what actually catches them. Deleting the `box` node drops its room→cabinet RT edge
  too. (Affordance backstop: a cabinet parent is `type()=="box" && name.starts_with("cabinet")`,
  NOT `type()=="cabinet"` — cortex registers no `cabinet` type.)

## Parsing numbers from files: `std::from_chars` ONLY, never `strtof`/`atof`/`stod`/`>>`

**These machines run `LANG=es_ES.UTF-8`, where the decimal separator is a COMMA.** Every agent here
is a Qt program, and Qt calls `setlocale(LC_ALL, "")` at startup, which activates that locale for the
**C** library. `strtof`/`atof`/`strtod` read through `LC_NUMERIC`, so on a file written with decimal
POINTS they stop dead at the `.` and return the integer part — **silently, with no error flag**.
`"0.260417"` becomes `0`. `"-9.2e-05"` becomes `-9`.

The asymmetry is what makes it invisible: `std::ofstream <<` formats through the **C++ global locale**,
which stays `"C"` unless someone calls `std::locale::global`. So the file is WRITTEN with points and
READ with comma rules. Round-tripping your own data corrupts it.

- **Read with `std::from_chars`** (`<charconv>`) — locale-independent by definition, no allocation,
  and it reports failure instead of guessing. `std::istringstream` is NOT a fix: it also formats
  through a locale (and is slow); if you must use a stream, `imbue(std::locale::classic())` first.
- **Write through `f.imbue(std::locale::classic())`** so output can never acquire a comma separator
  if the global locale is ever changed. Cheap insurance; do it on every data file you emit.
- **Reproduce it before trusting a test.** A standalone harness has no Qt, so it stays in `"C"` and
  the bug vanishes — a test binary and the agent will disagree on the same file and the same code.
  Add `std::setlocale(LC_ALL, "")` to any harness that parses agent data files, or the test is
  answering a different question than the one you asked.

Measured 2026-08-03 (retina `depth_dataset.cpp`, LiDAR-anchored depth-correction dataset): poses
truncated to integers so de-duplication threw away 334 of 406 frames; the `t` column became all-zero
so its fitted coefficient came out exactly `0.000`; scientific-notation values parsed as `±9` giving a
reported data range of `0.00012 .. 8103 m` on a file whose true span was `0.39 .. 10.16 m`; and the
fitted exponent read `0.153` instead of `0.193`. The DATA on disk was always correct — only the reader
was broken. Cost several rounds of chasing stale files and duplicate processes, because the failure
looks like bad data rather than bad parsing. When a harness and an agent disagree about the same file,
**that contradiction is the primary evidence** — chase it before any external explanation.

**Open sweep + generator status: see [LOCALE_PARSING_SWEEP.md](LOCALE_PARSING_SWEEP.md)** — the
sites still unfixed, and why newly generated components are already safe (the `robocompdsl` template
pins `LC_NUMERIC`; existing agents are NOT retrofitted). Re-run before trusting that list:
`grep -rn "strtof\|strtod\|\batof\b\|std::stof\|std::stod" --include=*.cpp --include=*.h .`

## Build

`cbuild` = `cmake -B build && make -C build -j32`. After adding sources via an INCLUDEd
`src/CMakeLists.txt`, do a clean reconfigure (nuke `build/CMakeCache.txt`) — incremental
cmake can miss appended sources. Match the prebuilt libdsr's Eigen alignment: NO
`-march=native` and NO `-DEIGEN_MAX_ALIGN_BYTES=0` (both change
`sizeof(optional<Eigen::Transform>)` and crash with SIGBUS/stack-smash). Grep ALL build
dirs' `flags.make`, not just CMakeLists.

**`-j32` HERE CAN OOM THE WHOLE MACHINE — cap it at `-j8`.** These are very large translation
units (`room_concept.cpp`, `wall_map.cpp`, the selftest) and each `cc1plus` peaks in the GIGABYTES,
not the hundreds of megabytes. Measured at a real OOM kill (2026-09-12 14:04, `journalctl -k`):
**20 `cc1plus` holding 43.7 GB of the 60.1 GB in use, the largest one 4.88 GB.** This box has 62 GB
and **NO SWAP**, so there is no soft failure mode — the kernel fires the global OOM killer.

- **The build is not the victim; something else is.** The killer scores by `oom_score_adj`, and the
  desktop apps carry a positive one — that run killed the user's *browser* while every compiler
  survived. So the symptom is "an unrelated program vanished" (and a half-finished `build/` with
  stale `CMakeFiles/Progress/*`), NOT a failed build. Do not go looking for a compile error.
- **Two Claude sessions on one checkout double it** (see the concurrent-sessions rule) — two `-j32`
  builds of different agents are 64 compilers against one 62 GB pool. Before a big build, check
  `pgrep -a cc1plus`; if another build is already running, wait or drop to `-j4`.
- **So: `make -C build -j8`** (≈11 GB peak) for normal work; `-j32` only for a small component you
  know compiles cheaply. The wall-clock cost is small — the long TUs serialise on memory bandwidth
  anyway — and the alternative is losing whatever else the user had open.

## Coding
- use C++23 containers and algorithm when possible
- use and,or and not instead of &&,|| and !
