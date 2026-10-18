# Episodic memory (EM) — design (2026-10-17)

Status: DESIGN, agreed in conversation with the user; nothing built. Context: `WM_PREDICTION_PLAN.md`
(why we want a learned predictor over the working memory; this document is its Phase 0 recorder).

## 1. Intent

The robot should remember **what it committed to and what happened as a consequence**, in a form that is
(a) queryable live by chat_agent ("what did you do this morning?", "how often does door_27 turn out shut?") and
(b) directly the training data for a later predictor of action consequences on the graph.

So EM does **not** log the graph over time. Its unit is a **commitment**: a mission, a mission step, or an
affordance claim, recorded together with the facts it names before and after, its outcome, its duration, and the
discrete side effects that happened while it ran. Commitments nest (mission → steps → claims), so each mission is a
small tree and the memory is a time-ordered sequence of trees.

Why this grain: the planned predictor's job is "the outcomes of the robot's own commitments on the graph"
(`WM_PREDICTION_PLAN.md §0`). Recording at that grain means the dataset needs no later segmentation, and it is also
the grain at which a person asks about the robot's day.

## 2. Goals, non-goals, anti-goals

**Goals (v1)**
- A new **episodic agent** that observes the live graph and writes episode records.
- Three record levels: **mission**, **IR step**, **affordance claim**, correctly nested.
- Both **human missions** (chat_agent → LLM → IR → controller) and **self missions** (the epistemic affordance
  planner driving with no human mission) are recorded, in the same format.
- Records live in a **dedicated DSR graph on DDS domain 4** (*my call*: domain 3 is taken by ltsm's offline stage fixtures, `ltsm_agent/etc/config_stage*.toml`): **today in full detail**, plus **consolidated
  statistics** for everything older.
- An **append-only file archive** of every record ever made (training dataset; reseeds the graph on restart).
- **chat_agent** reads domain 4 read-only through a shared **query library**, exposed to the LLM as tools, plus a
  one-line "today so far" in its world digest.
- One **controller addition**: publish what the observer cannot infer (see §6).

**Non-goals (v1)**
- No predictor of any kind (count models come next, from the consolidated nodes and the archive).
- No arm primitives (P3Bot). The format must admit them as a fourth level without change (§4).
- Nothing flows back from EM into the concept agents or the controller's decisions.
- No change to ltsm_agent's `passage_harvest` — it keeps feeding the door Beta beliefs; EM records the same
  crossings from the outside, and neither depends on the other.
- No readers besides chat_agent.

**Anti-goals (failure even if it "works")**
- EM slows the live graph, or becomes a **required peer**: the fleet must start and run identically with EM down;
  chat_agent must answer "my episodic memory is not available" rather than block.
- Records drift into dumping every attribute update. EM stores named facts + discrete events, nothing else.
- EM invents verdicts: an outcome is copied from its owner (interpreter, `aff_outcome`), never inferred by EM.

## 3. Constraints (from CLAUDE.md, binding)

- Graph signals: connect **queued only**, or poll and diff (repo-preferred, cf. `ltsm_agent/src/passage_harvest.h`).
- Attributes: **type-attributed** access only. New attributes are added to cortex's `dsr_attr_name.h`; **the user
  reinstalls cortex** — the builder lists the new attributes and stops there until it lands.
- Unique `[Agent] id` on every domain the agent joins (ltsm is 19). Presence: copy bottle_concept, with debounced
  Degraded. Graceful SIGTERM cleanup; no `kill -9`.
- Number parsing: `std::from_chars`; files written through `imbue(std::locale::classic())` (es_ES locale).
- Build with `cbuild` / `-j8`.

## 4. The record

One schema, `common/episodic/em_schema.h`, shared by the writer and every reader.

| Field | Content |
|---|---|
| `id` | unique, stable across restarts (e.g. `<date>-<seq>`) |
| `level` | `mission` · `step` · `claim` (reserved: `arm`) |
| `parent` | id of the enclosing record (none for a mission; a step's parent may be a composite step) |
| `author` | `human` · `self` (inherited by children) |
| `action` | mission: title + the IR (JSON) · step: IR step id, op, grounded args · claim: affordance kind, producer, target node, contract |
| `t0`, `t1` | commitment and terminal times |
| `pre` | snapshot of the **named facts** at `t0` (below) + the current room |
| `post` | the same facts at `t1` |
| `outcome` | mission/step: the interpreter's state and outcome word (+ `on_fail` path taken, attempt) · claim: the `aff_outcome` word |
| `events` | discrete side effects during [t0, t1], each `{dt, kind, node}` (below) |
| `promised` / `delivered` (claim only) | the producer's quoted `epistemic_gain` at claim time / the change in the target's belief entropy by `t1`, when computable — else absent, never 0 |
| `alternatives` (claim only) | the selector's candidate set with scores at claim time (from the controller addition) |

**Named facts** = the graph nodes the action refers to: a step's refs (door, room, target node, anchor's room), a
claim's target object node. For each: type, name, the beliefs it carries among a **whitelist** (pose in the current
room frame, σ diagonal, existence, P(open), passable, …). The whitelist lives in `em_schema.h`; the builder seeds it
from the attributes the IR predicates and the affordance contracts read.

**Events** are graph-**structure** changes only, so no thresholds are needed: node birth / deletion (whitelisted
types), `current` room edge moves, affordance offered / withdrawn. Continuous changes (a door's P(open) drifting)
are captured by `pre`/`post`, not as events.

Absent ≠ zero: a fact that could not be read is omitted, never written as a default value.

## 5. How records are made: a pure observer

The episodic agent joins domain 0 (read), domain 2 (ltsm memory, read — identities only, §7) and domain 4 (own). It never asks another agent to write episodes.

- **Human mission** (user decision 2026-10-17): **every operator command, from any source** — chat (IR),
  `remote_cmd.txt`, the panel — including single commands that bypass the interpreter (`cross door_N`, `move`,
  `turn`, `uncross`, `stop`), which become **one-step missions**. Read from the controller's mission identity (§6),
  not from the chat-only `has_intention` edge. Ends at the terminal state the controller reports. **Steps**: from
  the controller's step history (§6).
- **Claim**: read from the controller's claim history (§6), not from the `executing` edge / `aff_outcome` directly
  (both can come and go within one poll). Parent = the running step, or the running self mission.
- **Self mission** (*builder call, revisable*): a synthetic root spanning one uninterrupted stretch of
  affordance-driven control (drive mode `AffordancesOnly`, no human mission), **split at each `current` room change**
  so trees stay small.
- `pre` is read when the commitment is observed, `post` when the terminal is observed. With polling at 2–5 Hz this
  lags by ≤ one poll; acceptable, and recorded (`t0`/`t1` are the observation times; the owner's stamps are stored
  too when they exist).
- A commitment still open when EM stops is written with `outcome = interrupted_by_em_shutdown`; one open when the
  owner disappears gets `outcome = lost`. EM never guesses a real outcome.

## 6. Controller addition

What the observer cannot see today, published by the controller on nodes/edges it already owns, written only on change.
A pure poller at 2–5 Hz cannot see these reliably (builder check, 2026-10-17): producers clear `aff_outcome` on
re-offer and delete affordance nodes within a cycle, and a selector switch deletes the `executing` edge with no
outcome at all. So the controller publishes **append-only histories with serials**, and the observer reads what it
has not yet seen:
0. **Mission identity** (user decision): for every operator command — serial, source (`chat`/`remote`/`panel`),
   the IR or the command text, start time, terminal state + outcome — on a node the controller owns, independent of
   chat. Single commands are wrapped as one-step missions here.
1. **Step history** (*my call*): one entry per **attempt and per repeat iteration** (step id, op, attempt,
   iteration, state, outcome, t0, t1) plus the **resolved grounding** of the step (which door a `Passage` ref became,
   which pose an anchor became) — extends the `mission_progress` attribute planned in
   `chat_agent/docs/mission_language_2026-10-13.md §3.6`, since `Interpreter::steps()` keeps one entry per step id
   and does not expose iterations. Composite ops (seq, repeat) are records too, so `parent` may be a step.
2. **Claim history** (*my call*): one entry per claim (serial, affordance node + producer + target, t0, t1, outcome).
   The outcome is the `aff_outcome` word, or **`preempted`** when the selector switched away — stamped by the
   controller, which knows it, never inferred by EM.
3. **Selector candidates at each claim**: every affordance the selector scored at the moment it committed, with
   `G`, distance and quoted gain — needed later to correct for the robot's own choice bias.
4. **Drive mode** (`AffordancesOnly` / `MissionOnly` / `Command` / …), so self missions can be delimited.

Histories are bounded (e.g. the last N entries); EM keeps up because N covers many poll periods.

New attributes → cortex reinstall by the user (§3).

## 7. Storage

**Domain 4 graph** (owned by the episodic agent; config sub-table e.g. `[Agent.episodic]`, as ltsm does
`[Agent.ltsm]`):
- One node per record **for today**, linked to its parent. A few indexed attributes for querying without parsing
  (`level`, `author`, `op`/kind, `target`, `t0`, `t1`, `outcome`) + the full record as one JSON string attribute.
- **Identity for anything kept beyond today** (user decision 2026-10-17): live node names are NOT stable across
  restarts (`door_concept` hands out the first free `door_<n>`; rooms are generically named). Records keep the live
  name AND, when it exists, the **ltsm identity** (`passage_N` for a door, the memory room name for a room), so EM
  also joins ltsm's domain 2 read-only. Consolidated statistics are keyed by (action kind, ltsm identity) when there
  is one, otherwise by action kind alone. Per-door all-time answers are only as good as ltsm's identity (and
  `round_trips.sh` wipes ltsm memory each run — such runs contribute to per-kind statistics only).
- **Consolidated nodes**, one per (action kind, identity key as above): outcome counts per word, duration statistics, promised vs
  delivered gain statistics, first/last seen. These are permanent and are updated as records fold in.
- **Day rollover** (*builder call*): on the first commitment of a new day (not on a timer), fold yesterday's records
  into the consolidated nodes and delete their nodes from the graph.

**File archive**: append-only, one file per day, one JSON record per line, written at each terminal (and the open
record rewritten as closed — never edit past lines: append a closing line, readers take the last line per id).
On startup the agent rebuilds today's nodes and the consolidated nodes from the archive. The archive is the source of
truth; the graph is a view of it.

Why not everything in the graph: every DSR participant holds the whole graph and receives all of it on join, so an
unbounded history would slow every join; "today + statistics" keeps it bounded.

## 8. Query side (chat_agent only, v1)

- chat_agent joins domain 4 **read-only by contract** (second graph sub-table); EM owns every node there.
- **Shared header library** `common/episodic/em_query.h`: pure functions over a `DSRGraph`, no network service.
  v1 set:
  - `episodes(since, until, author, op, target)` → today's record summaries
  - `episode(id)` → one full tree with pre/post, outcome, events
  - `stats(action_kind, target)` → consolidated counts, durations, promised vs delivered (all time)
  - `last(target)` → the most recent record that touched a node
- chat_agent exposes these as LLM tools beside `find_objects` / `where_is` (through `answer_query`), and adds one
  "today so far" line to the world digest (e.g. *3 missions, 2 succeeded; last: crossed door_27 → room_2*).
- If domain 4 is absent or empty, the tools answer that episodic memory is unavailable; nothing else in chat_agent
  changes.

## 9. Success

A day of `round_trips.sh` plus free epistemic wandering, in sim, produces:
- complete, correctly nested trees for every human and self mission (no orphan claims, no open records after a clean
  shutdown);
- outcomes that match the interpreter's and `aff_outcome` exactly;
- a restart mid-day that loses nothing and duplicates nothing;
- chat_agent answering "what did you do today" and "how often did door_N turn out shut" correctly from EM;
- no measurable change in the live fleet's behaviour or timing with EM running vs stopped.

## 10. Left to the builder

Agent name and ids; poll rate (2–5 Hz); node/edge types on domain 4 (registered types only, e.g. `metaconcept` with
a subtype as ltsm does); exact attribute names; the JSON field spelling; the initial facts whitelist; how
`delivered` entropy is computed per belief kind (omit where unclear); test strategy. Calls marked *builder call* above
may be revised with a one-line justification.
