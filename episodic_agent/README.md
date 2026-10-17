# episodic_agent

An observer that records the robot's **commitments** — missions, IR steps and affordance claims — into
an episodic memory. It never acts, never claims and is no agent's required peer. All the logic lives in
`common/episodic/` (pure, tested); this component is the shell plus the DSR adapters.

## What it does
Every `[EM] poll_ms` (250 ms) it polls, on the main thread, with no DSR signals connected:
1. the controller's journal: the `em_journal` attribute of its dedicated node (type `intention`, name `em_journal`,
   `Shadow --has--> em_journal`; not the controller's agent node, which cortex's 1 Hz heartbeat rewrites). This goes
   through `journal_from_wire` → `Builder::on_journal`. If that node disappears after it has been seen (a graceful
   controller stop deletes it), or the presence monitor loses the `controller` peer (a crash / `kill -9` leaves the
   node behind) → `Builder::owner_lost`, and the open records close `lost`.
2. a structural diff of the live graph (born/died, offered/withdrawn, current_moved) → `Builder::on_events`.
3. every record returned → `Recorder::write`, which appends to the archive and upserts to the domain-4 graph.
4. every 10 s, one row of `<archive_dir>/metrics_<YYYYMMDD>_<HHMMSS>.csv` (one file per run, `src/em_metrics.h`):
   journal bytes and updates, open / closed records, the largest record and most events on one record, poll time.

Startup (`src/em_recorder.h`):
- records that a crashed run left **open** in the archive are closed `lost` and their closing line is appended (R21);
- the all-time stats are recomputed from scratch over every past day;
- today's records are reseeded into domain 4.

When the day changes, the previous day's closed records are folded into the stats nodes and their record nodes are
deleted. Each record is folded exactly once.

## Graphs
| table | domain | role |
|---|---|---|
| `[Agent.dsr]`  | 0 | live working graph. Read only, apart from the presence agent node `episodic_agent 26`. |
| `[Agent.em]`   | 4 | episodic memory, seeded from `em_root.json`, **owned**. One `metaconcept` per record of today: name = id, `em_record` JSON, `has` edge parent→child. One `stats_<level>_<kind>[_<author>]_<identity\|any>` node per stats key (author for missions only), carrying `em_stats`. Read back with `common/episodic/em_query_dsr.h` `load_store`. |
| `[Agent.ltsm]` | 2 | ltsm memory, joined read-only. Resolves a door's `passage_N` identity. |

`dsr` must stay alphabetically first, so that `G` is the live graph.

Archive: `[EM] archive_dir = "etc/episodes"`, relative to the CWD (git-ignored). It holds one `<YYYYMMDD>.jsonl`
per local day and is append-only: a record is written when it opens and again when it closes, and readers keep the
last line per id.

## Run / stop
```
cd episodic_agent && bin/episodic_agent etc/config.toml
```
- **Start order:** start it after the fleet (domain 0) **and after ltsm_agent** (domain 2). Joining a domain with
  `configFile ""` aborts the process after 15 s if no peer answers (R20).
- **Stop with SIGTERM or Ctrl-C (`kill <pid>`), never `kill -9`.** A graceful stop closes every open record as
  `interrupted_by_em_shutdown`, writes those records to the archive and to domain 4, and deletes the presence node.
  SIGKILL skips all of that. The next start still closes those records, as `lost`.
- **chat_agent re-joins by itself.** This agent re-seeds domain 4 from `em_root.json` on every launch. chat_agent
  joins domain 4 only while `episodic_agent` is present on domain 0, leaves it when this agent's node is gone or
  carries a new process stamp (a restart), and joins the fresh graph again (R23, R24) — no reader restart needed.

## Build and tests
- Build: `cmake -B build && make -C build -j8` (no `-march=native`; see `src/CMakeLists.txt`).
- Adapter tests (DSR on private domains 60–62, plus the pure Recorder test):
  `cmake -S tests -B tests/build && make -C tests/build -j8 && tests/build/dsr_adapters_test && tests/build/recorder_test && tests/build/metrics_test`
- Pure library tests: `common/episodic/tests/run_tests.sh`.

## ⚠ Hand-patched generated code
`generated/{genericworker.cpp,genericworker.h,main.cpp}` are copies of ltsm_agent's hand-patched files.
Do NOT re-run `robocompdsl` blindly: `generated/` is not in its avoid-overwrite list, and a re-run reverts these
patches:
- the QSettings window geometry/state save and restore (`RoboComp/episodic_agent`, `windows/26/em`);
- the empty destructor, which would otherwise delete every `grid` node from the shared graph;
- `std::_Exit` at the end of `main`, which skips libIce's broken static finalizers.
Re-apply them after any regeneration.
