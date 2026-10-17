# Mission language for chat_agent → controller — 2026-10-13

Three parts: **(A)** my review. **(B)** Fable's full design, unchanged, cited by file:line. **(C)** §7, added
2026-10-17: extensibility, the limits of expressiveness, and concrete syntax (JSON vs a DSL).

## A. Review and recommendation

### Verified against the code
- `chat_agent/src/ollama_client.cpp` sends no `format` field, so structured outputs are not used yet.
- `set_required_name` (`common/affordance_manager/affordance_manager.h:199`) is the only goal-directed hook in the
  selector, and it is a hard restriction.
- `MissionWithAffordances` is declared but unimplemented (`controller/src/controller_mission.h:18,52`).
- `readiness_explore_` is an atomic read only by the viewer (`room_concept/src/room_scene_graph.h:140,239`), so
  "explored" is not observable in the graph.

### Endorsed: the layered pipeline
1. The LLM emits **schema-constrained JSON**.
2. A **C++ resolver** grounds it against this cycle's graph, producing a grounded IR, a clarification question or a
   refusal.
3. The result is published as **one chat-owned `intention` node**, with status on the controller-owned edge.
4. The **controller `MissionInterpreter`** compiles each primitive into what already exists:
   - `DoorPassage`, for door steps;
   - a point target, for poses;
   - the free selector, for exploring;
   - a new *soft* `set_goal_prior`, for looking at objects.

This design:
- keeps "LLM proposes, C++ disposes";
- keeps the controller as the single executor;
- keeps the producers' authority over their preconditions;
- treats missions as preferences in the EFE rather than scripts.

### Amendments
1. **`set_goal_prior` is the Active-Inference heart; make it principled from the start.**
   - The prior is a log-preference `ln C` over outcomes. It should be expressed in nats against the same
     `G = λ·dist − gain` the selector already minimises.
   - `set_required_name` becomes its limit (κ → ∞) and is documented as such.
   - Do not let it become a second, ad-hoc score.
2. **Predicate thresholds are user preferences rendered on the card.** That is right. But `explored(room)` must also
   publish its *uncertainty*: the explorer's remaining expected information gain in nats, not only a [0,1]
   readiness. Otherwise "explore until done" inherits the stopping-rule problem the memory notes describe:
   coverage was flat at 86% and termination fired on 1/50 runs.
3. **Door reliability is the floor.** A mission is only as reliable as its worst crossing (try44 5/6, try45 3/6).
   Phase 1 should ship with `try/retry` per leg and honest failure reporting, and with
   `round_trips.sh`-style mission scripts driven from `remote_cmd.txt`. No LLM in that loop, so the bench runs
   the agent's code.
4. **Anchors after re-entry:**
   - carry the anchor's σ;
   - inflate the seam error after a re-entry (~26 cm);
   - make the arrival tolerance follow from that covariance rather than a fixed radius.

### Recommended first slice (~1 week)
1. **Phase 0:**
   - `room_explore_readiness`, plus the remaining-gain attribute;
   - the `ltsm_topology` digest.

   Both need new cortex attributes, which means a reinstall on your side.
2. **Phase 1:**
   - the IR and resolver as pure headers with tests;
   - the controller interpreter for `seq / repeat / try`, `cross`, `open`, `go_to room / anchor`, `explore`, `wait`;
   - driven by the text form from `remote_cmd.txt`.

   This already covers "go to room 2, explore it and come back here".
3. Then add the planner request with `format = schema`, and the confirmation card.

---

## B. Fable design (verbatim)

# A mission language for chat_agent → controller

Design study, 2026-10-13. Read-only survey of `active_inference/` at `ca293787` (branch `wip/box-estimator-rework`).
All paths are relative to `/home/pbustos/robocomp/components/active_inference/`.

## 0. Recommendation in one paragraph

Keep the two things that already work — the LLM proposes and C++ disposes (`chat_agent/src/command_gate.h:1-40`),
and the controller is the single executor that only ever *claims affordances* (`controller/src/controller_door_passage.h:1-25`)
— and put a **mission** between them instead of a one-line command. Concretely, a four-layer pipeline:

1. **LLM emits a schema-constrained JSON mission** (ollama `format` = JSON schema; one extra request, no streaming,
   ~1-2 s on the 8B instruct model). The schema is a flat list of typed steps with one level of nesting
   (`repeat`/`if`/`on_fail`), a closed `op` enum, and symbolic references (`room:kitchen`, `door:between(room_1,room_2)`,
   `object:bottle on table in kitchen`, `anchor:start`).
2. **chat_agent resolves and validates it in C++** against the graph *this cycle* (the same `WorldDigest` pass the
   gate already uses, `chat_agent/src/world_digest.cpp:49-183`): every reference becomes a node id or a pose, every
   predicate becomes an attribute read, ambiguity becomes a clarification question, impossibility becomes a refusal.
3. **The resolved mission is published to DSR** as ONE chat-owned `intention` node carrying the compiled IR
   (`mission_<n>`, `has_intention` from `Shadow`, exactly like `chat_intention` today), and the controller reports
   progress on the edge it owns (`controller/src/chat_inbox.h:14-25`). Per-step child nodes come later, only when a
   second reader (ltsm route annotation, room_concept explore target) actually needs them.
4. **The controller runs a `MissionInterpreter`** — a small step machine that compiles each primitive to what the
   controller already does: restrict/bias the affordance selector (`AffordanceManager::set_required_name`,
   `common/affordance_manager/affordance_manager.h:199`; add a *soft* `set_goal_prior` beside it), start a
   `DoorPassage` (`controller_door_passage.h:60`), set a point target (`specificworker.cpp:1397 set_manual_target`),
   or a relative motion. Active Inference stays intact: a mission is a **preference over outcomes** that reshapes the
   expected free energy of the affordances on offer; the selector still chooses, the producers still own
   preconditions and completion, and `stop` still aborts everything (`specificworker.cpp:1896 abort_all`).

A text DSL exists only as the human-readable rendering (confirmation card, logs, `remote_cmd.txt`); behaviour trees
are not adopted as a representation but the interpreter's combinators (`seq`, `repeat`, `if`, `on_fail`, `monitor`,
`timeout`) are exactly the BT control nodes, so a BT library is not needed and nothing is lost.

---

## 1. What exists today (survey)

### 1.1 The chat side

| Piece | File | What it does |
|---|---|---|
| Robot self-description | `chat_agent/etc/system_prompt.md` | Tells the model what it can do; explicitly *"You cannot go to a named place or object, follow a person, fetch things"* and *"the only movements you can start are crossing a door of the current room, short relative moves and turns, and stopping"*. |
| Closed command set | `chat_agent/src/command_gate.h` | `ToolCall{name ∈ cross_door, stop, cancel_passage, move_relative}` → `validate()` (`:95-170`) against `GateFacts{room, doors, nearest_door, controller_present}` → `GateResult{command, needs_confirmation}`. `parse_direct()` (`:230-262`) accepts text already in the controller grammar. `is_stop/is_yes/is_no` are the model-free fast paths. |
| Tool schema the model sees | `chat_agent/src/chat_session.cpp:296-345 tools_json()` | 7 tools: `cross_door(door enum)`, `stop`, `cancel_passage`, `move_relative(distance_m, turn_deg)`, read-only `find_objects`, `where_is`, `look(camera)`. |
| Turn flow | `chat_session.cpp:187-233 on_user_text()` | stop fast path → typed yes/no to a card → direct grammar → model. `ask_model()` (`:365-430`) rebuilds ONE system message per turn (description + `kInterfaceRules` `:243-270` + STATE digest) and keeps ≤16 history messages. A tool round with no words triggers one follow-up request. |
| World digest (STATE) | `chat_agent/src/world_digest.cpp:49-183` | Text: time, controller presence, current room (`rc::room::current_room`), other rooms (+provisional), robot pose in the room frame, doors of the current room with distance/bearing/`P(open)`/affordance states/`behind it: remembered <room>`, remembered rooms (from `recall_*` metaconcepts), OBJECTS of the current room (`object_facts.h`), last chat command + its status. Also fills `GateFacts` so the gate validates against exactly what the model saw. |
| Command channel | `chat_agent/src/intention_node.h:1-20`, `intention_node.cpp:84-96 send()` | ONE chat-owned node `chat_intention` (type `intention`, edge `Shadow --has_intention--> chat_intention`). chat writes `current_intention` (string) + `timestamp_agent` (a new stamp IS a new command). The controller writes `state`/`bt_state`/`timestamp_agent` on the **edge** only (one writer per object, LWW engine — `controller/src/chat_inbox.h:14-21`). |
| Status back | `chat_session.cpp:580-640 tick_status()` | Polls the edge every 250 ms; prints step/event transitions as robot lines; on `completed/aborted/failed|reason` appends a `role:"tool" robot_status` message so the model can say how the command ended. |
| Ollama client | `chat_agent/src/ollama_client.cpp:96-108` | `POST /api/chat {model, messages, stream:true, think:false, keep_alive, options{temperature,num_ctx}, tools}`. **No `format` field yet** — structured outputs are not used. Model `qwen3-vl:8b-instruct`, `num_ctx 8192` (`chat_agent/etc/config.toml` `[LLM]`). |
| LLM test harness | `chat_agent/tests/check_llm_tools.py` | Builds the same system message + tools from the source and grades the first tool call against a synthetic STATE; reports `prompt_eval_count`. Reusable for mission planning tests. |

### 1.2 The controller side

| Piece | File | What it does |
|---|---|---|
| Inbox | `controller/src/chat_inbox.h`, `chat_inbox.cpp:51-69 status_of()` | `poll()` returns a new `{text, stamp}`; `publish()` mirrors `(stamp, state, bt_state)` on the edge. Stale first-sight commands (>3 s) are adopted, not run. `Tracked{kind ∈ Cross, Stop, Uncross, Rejected, Motion}` — exactly ONE tracked command at a time (`specificworker.cpp:1909-1937 service_chat_inbox()`, 10 Hz on the control thread). |
| Grammar | `controller/src/specificworker.cpp:1757-1893 run_mission_command()` | `stop` · `uncross` · `move <m>` / `turn <deg>` (chained, `controller_relative_motion.h:1-35`) · `cross door` (only unambiguous in a one-door room, `:1838-1856`) · `cross door_N` (node must exist, `:1861-1878`) → `DriveMode::Command` + `DoorPassage::start`. Same parser for the panel text box, `remote_cmd.txt` (`:548-576`, also `run`/`stop`/`mission <text>`) and the chat. Returns `MissionResult{accepted, kind, door, reason, motion_id}`. |
| Door passage | `controller/src/controller_door_passage.h` | Sequences `aff_<door>_approach → _open → _cross` by naming the ONE affordance the selector may take (`set_required_name`), jumping to a later offered step, falling back when a step is withdrawn, going back to `open` when the door turns out shut, and declaring DONE when the controller's room changes while crossing (`:205-213`). Exposes `View{door, step, step_state, attempts, last_event}`. While crossing it hands the planner a corridor through the doorway and exempts the door node (`:262-268`). |
| Where it runs | `controller/src/controller_session.cpp:159-178` | `door_passage_.update()` every planning step, before the grid is built; `:795 affordance_manager.set_required_name(door_required_)`. Order in `compute()` (`specificworker.cpp:742-745`): relative motion owns the base first, then `build_planning_step` (manual target > mission route > affordance selection). |
| Drive modes | `controller/src/controller_mission.h:48-56` | `AffordancesOnly`, `MissionOnly` (recorded waypoint tours, a measuring instrument), `MissionWithAffordances` (unimplemented), `Target` (clicked point; `set_manual_target` `specificworker.cpp:1397`), `Command` (the chat/door passage: affordances drive, the command restricts selection). |
| Executor policies | `common/affordance_protocol/affordance_protocol.h:48` `Policy{Reach, Servo, Orient}`; `controller_session.cpp:4451-4474` | Reach = navigate to the pose; Servo = navigate then lock-on micro-search until the contract's goal clauses hold; Orient = rotate in place toward the bearing (no navigation). Completion = AND-ed clauses on the feedback node, `stable_n`, `timeout_ms`, `on_fail`. |
| Room following | `controller/src/controller_world_model.cpp:36-56` | The current room is re-resolved every cycle (`rc::room::current_room_node`); a room change is logged `current room changed: a -> b`. |

### 1.3 The affordance protocol (what a mission can bind to)

- **Selection**: `AffordanceManager::select_target()` (`affordance_manager.h:246`) scores `G = λ_cost·dist − gain` with
  hysteresis (`:224-226`), a room-vs-object preference (`set_room_gain_scale`, `:228-240`, "a PREFERENCE, NOT A GATE"),
  no-two-in-a-row suppression (`:262-273`), and `set_required_name()` (`:199`) — the only *goal-directed* hook today,
  and it is a hard restriction.
- **Claim**: an `executing` edge from the consumer's agent node to the affordance node carrying `executing_epoch` and
  the pose actually driven to (`:103-160`); `mark_reached(outcome)` is the only writer that clears `epistemic_pending`
  (`:252`). Outcomes (`aff_outcome`, cortex `dsr_attr_name.h:1102`): satisfied / timeout / unreachable / infeasible /
  refused / abandoned / outside_room.
- **Epistemic producers** (one `aff_<object>` per object, `common/object_affordance/object_affordance.h:1-25`):
  bottle, cabinet, chair, hood, human, refrigerator, table. Each carries a `Contract` keyed by object type
  (`affordance_protocol.h:196+ default_contract_for`): table = Servo on `table_roi_offset/fill` until
  `table_detection_alive`; bottle = Servo until `bottle_detection_alive`; person = Servo with a still dwell.
- **Pragmatic producers** (`common/pragmatic_affordance/pragmatic_affordance.h`): named, *preconditioned* offers with a
  Schmitt band (`Policy{offer_prob .60, withdraw_prob .35, stable_cycles 3}`, `:145-150`) and the rule "could not
  compute ≠ false" (`Offer::known`). door_concept publishes three per door (`door_concept/src/door_pragmatics.h:1-20`):
  `approach` (always, for a located door; completes on `door_reach_prob`), `open` (while in reach AND believed shut;
  **Orient** until `door_open_prob ≥ 0.70`, `:157-165`; claiming it makes door_concept ask the actuation provider),
  `cross` (while believed passable; **Reach** until `door_crossing_progress ≥ 0.90`, `:170-176`).
- **Room**: `afford_room` from room_concept — the live layout explorer while the start-up proto is proto, the
  `EpistemicPlanner` after promotion (`room_concept/src/room_scene_graph.cpp:2783-2795`). The explorer's
  `Plan.finished` ("closed polygon AND nothing left to learn", `layout_explorer.h:115-131`) is **internal**: it feeds
  `readiness_explore_` which only the viewer reads (`room_scene_graph.h:139, 239`). Nothing in the graph says "this
  room is explored".

### 1.4 World queries and identity

- `common/world_query/door_query.h`: `list_doors` (`:32`, `object` nodes named `door*`), `room_of` (`:41`, RT chain up
  to a `room`), `doors_of_room` (`:57`), `nearest_door` (`:65`). Shared by chat and controller on purpose so they can
  never disagree about which door "the door" is.
- `common/room_resolve/room_resolve.h:12-30`: THE rule for "the current room" — the `robot --current--> room` edge
  owned by ltsm_agent, else the unique room with a live robot RT, else the unique non-proto room, else a unique
  start-up proto, else nothing.
- Node/edge/attribute vocabulary already registered in cortex (`/usr/local/include/dsr/core/types/type_checking/`):
  node types `intention` (`dsr_node_type.h:39`), `task` (`:134`), `metaconcept` (`:145`), `affordance` (`:53`);
  edge types `has_intention` (`dsr_edge_type.h:60`), `executing` (`:34`), `following_action` (`:58`), `match`
  (`:66`), `current` (`:64`), `in` (`:37`), `looking_at` (`:44`); attributes `current_intention` (`dsr_attr_name.h:428`),
  `state` (`:692`, documented as `{waiting, in_progress, aborted, failed, completed}`), `bt_state` (`:697`),
  `plan_target_node_id` (`:423`), a legacy `task_*` family (`:665-673`), `door_open_prob/door_reach_prob/
  door_crossing_progress` (`:837-839`), `passage_crossings` + Beta posteriors `passage_pass_alpha/beta`,
  `passage_open_alpha/beta` (`:916-920`). Adding an attribute = cortex header edit + **user reinstall** (CLAUDE.md).

### 1.5 Long-term memory (ltsm_agent, id 19)

- Two graphs: live DSR on domain 0, memory on domain 2 (`ltsm_agent/src/specificworker.h:103-110`). Memory holds every
  room as `r<idx>_<room_name>` under root (`room_eviction.h:153 memory_prefix`), its doors as faces
  `r<idx>_door_N`, and one `passage_N` metaconcept per doorway with `match` edges to its two faces
  (memory note `ltsm-abstract-passage-node`), carrying `passage_crossings` and the two Beta posteriors.
- Identity: ltsm moves the `current` edge (promotion), evicts the departed room into memory and deletes it live
  (`EVICTION.md`), so **only the current room (and a proto next door) is live**; everything else is memory.
- Recall: for every live door whose face memory holds, ltsm posts `recall_<room>` (metaconcept, subtype
  `room_recall`, parent = the door, RT door→recall = T(face←room), the remembered outline) so room_concept re-creates
  THAT room with the same name and frame when the robot crosses (`room_recall.h:1-16`, `room_recall.cpp:112-150`;
  consumed at `room_concept/src/room_scene_graph.cpp:1444`). Round trips room_1↔room_2 work live (memory note
  `resume-door-passage-two-problems`: try44 5/6, try45 3/6; test `room_concept/estimation_scripts/scripts/round_trips.sh`
  drives legs by writing `cross door_N` to `controller/remote_cmd.txt`).
- `PassageLive` keeps one live `passage_<n>` between an entry mirror and the current room's door while a proto-room
  exists (`passage_live.h:1-25`).

### 1.6 What is missing for complex missions (the gap list)

1. **One command, one string, one tracked kind** (`ChatInbox::Tracked`, `chat_tracked_`): no sequencing, no state
   across commands, no conditionals, nothing survives a room change except the door passage.
2. **No "go to a pose / near a node" primitive in the grammar.** A clicked target exists (`DriveMode::Target`) but
   only from the GUI; the relative move is capped at 3 m and is straight-line.
3. **No requestable object affordance.** `aff_<object>` nodes are selected by EFE; the only way to *ask* for one is
   `set_required_name`, used only by the door passage.
4. **No route through memory.** The digest shows one hop (`behind it: remembered room_1`); multi-hop needs the memory
   topology, which lives on domain 2 where only ltsm_agent can read it.
5. **"Explored" is not observable** in the graph (§1.3) — `explore(room) until done` has nothing to wait on.
6. **No anchors.** "Here" / "where you started" is a pose in a room frame that must survive eviction and re-entry.
7. **No structured status.** `state|bt_state` carries one step and one event; "what are you doing?" over a 7-step
   mission needs a step index, per-step outcomes, and the current affordance.
8. **No structured outputs.** The model's only typed channel is tool-call arguments.

---

## 2. Language design

### 2.1 Abstract syntax

```
Mission   := { id, title, steps: Step[], on_fail?: Policy, deadline_s? }
Step      := Primitive | Seq | Repeat | If | Try | Monitor | Timeout
Seq       := { op:"seq",     steps: Step[] }
Repeat    := { op:"repeat",  times: n | until: Pred, steps: Step[] }
If        := { op:"if",      cond: Pred, then: Step[], else?: Step[] }
Try       := { op:"try",     steps: Step[], on_fail: Step[] | "skip" | "abort" | "retry(n)" }
Monitor   := { op:"monitor", cond: Pred, steps: Step[] }        -- run steps; abort them (→ on_fail) when cond turns false
Timeout   := { op:"timeout", s: seconds, steps: Step[] }

Primitive :=
  go_to      { to: Ref<room|object|door|anchor|pose>, standoff_m?, face?: Ref }  -- reach a place
  cross      { door: Ref<door> }                                                -- approach→open→cross (DoorPassage)
  open       { door: Ref<door> }                                                -- approach→open, stop there
  explore    { room?: Ref<room>, until?: Pred, budget_s? }                      -- afford_room epistemic, bounded
  look_at    { target: Ref<object|door|room-feature>, dwell_s? }                -- orient + the object's own affordance
  find       { what: ObjectSpec, in?: Ref<room>, then?: Step[] }                -- exists? else explore until exists
  wait       { s?: seconds, until?: Pred }                                      -- hold still; "until I come" = Pred
  say        { text }                                                           -- a robot line in the chat
  anchor     { name }                                                           -- capture robot pose as a named anchor
  move       { distance_m?, turn_deg? }                                         -- today's relative motion
  stop       {}
```

Every composite is a step, so `seq` nests; the JSON schema caps nesting at depth 2 for the 8B model (§4) while the IR
and interpreter are unbounded.

### 2.2 References

```
Ref := { kind: room|door|object|anchor|pose|person, ... }
  room    { name?: "room_2" | "kitchen" | "the room behind door_19" | "where I started" }
  door    { name?: "door_19" | between: [roomA, roomB] | of_room + nearest | behind which: room }
  object  { type: "table"|"chair"|"bottle"|…, name?: "table_1", in?: Ref<room>,
            rel?: { on: Ref<object> | near: Ref<object> | nearest },  attrs?: {…} }
  anchor  { name: "start" | "here_<n>" | user-named }
  pose    { room: Ref<room>, x, y, yaw? }      -- never emitted by the LLM; produced by resolution
  person  { name?: "the person" }
```

**Resolution happens in chat_agent, in C++, against the graph of this cycle**, and produces a *grounded* IR in which
every `Ref` carries the node id (or pose) it resolved to plus the evidence used, or a `Clarify{ref, candidates}`.

- *Rooms by name*: live rooms (`get_nodes_by_type("room")`, skipping protos) and remembered rooms (recall nodes + the
  ltsm topology digest, §3.2). "the kitchen" is a **label**, not a node name: add a `room_label` attribute (ltsm-owned,
  persisted in `ltsm_memory.json`) settable from chat ("this is the kitchen" → `label_room` tool). Until labels exist
  the resolver matches the user's word against the object content of each room (a room with a fridge and a hood is
  "the kitchen" with high probability) and asks when two rooms tie.
- *Rooms by memory identity*: the name IS the identity (ltsm re-creates a room with its remembered name,
  `room_recall.h:9-12`), so `room:room_1` is stable across eviction. The `room_id` attribute (`dsr_attr_name.h:278`)
  is the fallback if names are ever recycled.
- *Doors*: a door name is only valid while that door is live; across rooms a door is identified by its **passage**
  (`between: [room_1, room_2]` → the memory `passage_N` whose faces hang from `r*_room_1` and `r*_room_2` → the live
  face is whichever `door_*` of the current room carries the `recall_<other>` child, `room_recall.cpp:112-150`).
  "the door" alone follows the controller's own rule: unambiguous only in a one-door room (`specificworker.cpp:1838-1856`).
- *Objects*: `type` matches `object_subtype` (`object_facts.h:5-17`); `in` filters by `rc::world::room_of`; `on`
  resolves by geometry (the candidate's footprint centre inside the support's footprint and its base at the support's
  height ± σ — the same relation `common/support_parent` already computes for births); `near` by the smallest 2-D
  distance; `nearest` relative to the robot. Objects in a *remembered* room are not in the live graph: resolution
  returns `Deferred{re-resolve after go_to(room)}` and the interpreter re-runs the resolver on arrival (the IR keeps the
  symbolic ref beside the grounded one for exactly this).
- *Anchors*: `anchor(start)` is captured **at mission start** as `(room node name, room_id, x, y, yaw in that room's
  frame, σ)`; `here` in the user's sentence means the robot's pose when the sentence was received. Because a re-entered
  room comes back in its remembered frame (seam error ≈ a few cm in yaw, ~26 cm xy per the 10-12 note), the pose is
  valid again after `go_to(room)`; the resolver re-checks that the room with that name is live and inside-polygon
  before issuing the point target. Anchors live on the mission node (chat-owned) so they survive a controller restart.
- *Person*: `human_concept`'s `person_N` nodes; "until I come" = `near(person, robot) < d` or a chat message.

Ambiguity policy: a reference with 2+ candidates whose scores are within a factor e of each other (log-odds < 1 nat)
is a `Clarify`, which the chat turns into a question with the candidates named exactly as the STATE names them. A
reference with no candidate is a refusal with the reason ("no table is known in room_2; I can look for one — say
*find a table in room 2*").

### 2.3 Predicates over beliefs

```
Pred := { p: door_open(door)       }  -- P from door_open_prob (marginalised, dsr_attr_name.h:837)
       | { p: door_passable(door)  }  -- the cross affordance is OFFERED (door_concept's decision, not ours)
       | { p: exists(object_spec)  }  -- a matching node exists (+ obj_exist_prob when a producer writes it)
       | { p: in_room(room)        }  -- rc::room::current_room == room
       | { p: near(a, b, d_m)      }  -- 2-D distance between two resolvable frames
       | { p: explored(room)       }  -- room_concept's readiness (new attribute, §3.3)
       | { p: person_present(d_m) }
       | { p: elapsed_s(s) } | { p: step_outcome(id) == "satisfied" }
       | { not: Pred } | { all: Pred[] } | { any: Pred[] }
       | { prob: Pred, ge: 0.7 }     -- a decision on a probability: explicit, named, and the ONLY place a number appears
```

A predicate evaluates to a **probability** where the belief gives one (`door_open`, `exists`) and to 0/1 where the
graph gives a fact (`in_room`). `if` and `until` need a decision; the decision boundary is written in the mission by
the user's own words ("if it is shut" → `prob(door_open) < 0.5`, the Bayes decision with symmetric loss) and shown on
the confirmation card, so it is a stated preference, not a hidden gate. The default is 0.5; the LLM may only choose
from `{0.5}` unless the user gives a number ("if you are quite sure" → 0.8).

### 2.4 Representation: the four candidates

| | (a) text DSL | (b) JSON, schema-constrained | (c) mission as a DSR subgraph | (d) behaviour tree |
|---|---|---|---|---|
| LLM reliability (8B) | poor: free text drifts, needs a parser with error recovery | **best**: grammar-constrained decoding guarantees well-formed output; enums close the op set | n/a (the LLM cannot write nodes) | n/a |
| Validation before execution | needs a parser + resolver | **resolver on a typed tree** | every agent could validate its part | BT libs validate structure, not references |
| Visibility to the fleet | none | none until published | **all agents see goals** (ltsm can annotate routes, room_concept sees "explore requested") | none |
| Churn / CRDT cost | none | none | per-step nodes = many writers, LWW races (`chat_inbox.h:14-21`) | none |
| Extensibility | grammar edit | add an enum value + a resolver case + an interpreter case | add a node type/attr → cortex reinstall | add a node class |
| Human readability | **best** | fair | poor (graph viewer) | fair |
| AI fit | neutral | neutral | **best** (goals are beliefs other agents can condition on) | poor: a BT is a script, it does not reshape EFE |

**Decision**: (b) as the LLM↔chat contract, compiled to (c)-lite — one node carrying the IR, status on the edge — and
rendered as (a) for people. (d)'s combinators live inside the interpreter; its *representation* is not adopted.
Why not full (c) from day one: the "who reads this?" rule (memory `published-attrs-without-consumers`: 161 of 292
published attributes have no reader) — step nodes earn their place when ltsm or room_concept starts reading them
(Phase 3), not before.

Why this satisfies the requirements:
- *Efficiency*: one extra LLM request per mission (planner prompt ≈ 700 tokens + JSON digest ≈ 300-800 + output
  ≈ 100-300 tokens; ~1-2 s at the measured ~0.4 s median tool-call latency of the instruct model). The controller
  cost is one small step machine on the control thread; graph traffic is one node write per mission and edge writes
  only on status change (`chat_inbox.cpp publish` already writes only on difference).
- *Expressiveness*: sequence, bounded loops, belief conditionals, recovery, monitors and deadlines cover all six
  driving examples (§5) and compose.
- *Extensibility*: a new primitive = one enum value in the schema, one `Resolver` case, one `Interpreter` case; a new
  reference kind = one resolver function; a new predicate = one evaluator. All three are tables keyed by `op`.
- *Safety*: closed `op` set; references resolved against the live graph before anything moves; the confirmation card
  shows the grounded plan; `stop` fast path unchanged; per-step the controller re-checks what it already re-checks
  (footprint, room polygon, offered-ness).
- *Active Inference fit*: §2.5.

### 2.5 Missions as goal priors, not scripts

The selector minimises `G = λ·dist − gain` over offered affordances (`affordance_manager.h:224-246`). A mission step
adds the **pragmatic** term of expected free energy — a prior preference over outcomes `C` — which today is absent
except as the all-or-nothing `set_required_name`. Proposed:

```
AffordanceManager::set_goal_prior(std::map<std::string /*affordance name or kind*/, float /*nats*/> prior);
// G_i = λ·dist_i − gain_i − prior_i      (prior_i = 0 for everything not named)
```

- `cross door_N`, `open door_N`: the door passage keeps `set_required_name` (= an infinite prior). Rationale: a
  detour mid-doorway is unsafe and the passage's own state machine already encodes door_concept's beliefs; this is the
  limiting case of the prior, stated as such.
- `look_at(table_1)`: prior `+κ` nats on `aff_table_1` with `afford_room` still eligible — a lost robot may still
  re-localise first if its gain is large enough (`set_room_gain_scale`'s own argument, `affordance_manager.h:235`).
- `explore(room)`: no prior at all (pure epistemic), or a *negative* prior on object affordances if the user said
  "just the room".
- `go_to(pose)`: not an affordance — it is a point target (`DriveMode::Target`), exactly as a click. Between legs of a
  route the epistemic planner stays suppressed (the `MissionOnly` argument, `controller_mission.h:12-18`): a mission
  that lets `afford_room` win half-way to the door is not what the person asked for. Later (`MissionWithAffordances`,
  the unimplemented mode) a bounded epistemic detour between legs is a prior choice, not a code change.

Producers keep their authority: `open` is still offered only when door_concept believes the door shut and in reach;
`cross` only when passable; `aff_table_1` only while table_concept wants a look. A mission can *ask* for an outcome;
it cannot make an impossible offer appear — which is the safety property the pragmatic protocol was written for
(`pragmatic_affordance.h:5-15`: "an impossible offer with a plausible price is worse than silence").

---

## 3. Semantics: what each primitive compiles to

### 3.1 Compilation table

| Primitive | Controller action | Completion | Failure → outcome |
|---|---|---|---|
| `cross door` | `DoorPassage::start(door)`; `DriveMode::Command` (`specificworker.cpp:1872-1873`) | `View.step == Done` (room changed while crossing, `controller_door_passage.h:205-213`) | `attempts` on a step exceed `retry(n)` → `door_wont_open` / `cross_unreachable` (from `aff_outcome`); door node vanished → `door_lost` |
| `open door` | `DoorPassage::start(door, stop_after = Open)` — **new argument**: the passage stops at `Open` satisfied and does not require `cross` | `opened_` (`:245`) | as above |
| `go_to room R` | route = chain of passages (§3.2) → expands at compile time into `seq(cross d1, cross d2, …)` with `monitor(in_room(expected))` | `in_room(R)` | no route → `unreachable_room`; a leg fails → `Try` policy |
| `go_to object/anchor/pose` | after `go_to(room_of(target))`: point target = standpoint (§3.4) via `session_.set_manual_target` (`specificworker.cpp:1397`), `DriveMode::Target`, route follower + arrival test | arrival (the existing goal threshold) with an optional final facing (`wants_final_facing`, `controller_session.cpp:3567`) | `unreachable` (no plan), `infeasible` (no free standpoint), stall watchdog |
| `explore room` | `DriveMode::Command` with `required_name = ""` and `goal_prior = {}`; selector free; optional negative prior on `aff_*` objects | `explored(room)` (§3.3) OR `until` OR `budget_s` | none (exploration cannot fail, it can only run out of budget → `budget_exhausted`, which `Try` may treat as success) |
| `look_at target` | if `aff_<target>` is offered: `goal_prior {aff_<target>: +κ}`; else: Orient to the bearing of the target node from the robot (reuse `step_orient`, `controller_session.cpp:5790`) and optionally `go_to(target, standoff)` first | the object affordance completes Satisfied, or the orient `aligned` predicate + `dwell_s` | `timeout`, `not_visible` (the contract's detection clause never held) |
| `find what in R` | `if exists(what in R) → go_to(it)` else `explore(R, until: exists(what))` then `go_to(it)` | the object resolved and reached | `not_found` after the explore budget |
| `wait` | `motion_commander_.stop_robot()` each cycle, base armed, no selection (the `paused_` branch without the GUI) | `s` elapsed or `until` true | none |
| `anchor name` | chat-side: capture `(room, x, y, yaw)` at the step; written on the mission node | immediate | pose unknown → `no_pose` |
| `move` | today's `RelativeMotion` | its `completed` | its `failed\|reason` |
| `say` | none (chat-side: a robot line + history entry) | immediate | — |
| `stop` | `abort_all` (`specificworker.cpp:1896`) | immediate | — |

Outcome words are a closed set shared with `aff_outcome` where they coincide, extended with mission-level ones:
`satisfied, timeout, unreachable, infeasible, refused, abandoned, outside_room, not_found, no_route, door_wont_open,
door_lost, budget_exhausted, stopped_by_user, superseded`. Added by the interpreter (2026-10-17): `wrong_room` (a
`cross expect` that ended elsewhere, or a step addressed to a room the robot is not in), `no_pose`, `no_anchor`,
`passage_unresolved`, `cancelled`; a satisfied step records why it ended: `until`, `budget` (an explore with no
`until`), `resumed`.

### 3.2 Routes through memory (go_to room)

Memory is on domain 2; chat_agent and the controller cannot read it. Two options:

- **(i) ltsm publishes a live topology digest** — one ltsm-owned `metaconcept` node `ltsm_topology` with a JSON
  string attribute: `{rooms:[{name, label?, live:bool}], passages:[{id, a:{room, face}, b:{room, face}, crossings,
  p_pass:{alpha,beta}, p_open:{alpha,beta}, live_door?: "door_27"}]}`. Rebuilt only when memory changes (eviction
  fence, label change). Readers: chat_agent's resolver (route = BFS/Dijkstra over passages, cost = expected crossing
  cost `1/E[p_pass]` + distance), the digest ("remembered rooms: room_1 (behind door_27, crossed 2x)"), the controller
  (nothing — it receives the expanded sequence).
- **(ii) per-step mission nodes that ltsm annotates** with a route attribute — needs (c)-full; deferred.

Recommend (i). The route expands to `seq(cross d_1, …, cross d_k)` where only `d_1` is a live door name; later legs
carry the *passage id* and the interpreter resolves "the live face of passage P in the current room" on arrival
(`recall_<room>` parent, `room_recall.cpp:135`; or the `ltsm_topology.live_door` field, which ltsm keeps current).
A `monitor(in_room(expected_room_k))` around each leg catches the symmetry-flip / wrong-room case: if after a
crossing the current room is not the expected one, the step fails `wrong_room` and `Try` re-plans from where the
robot is (the resolver is re-run; the route is recomputed from the new room).

### 3.3 "Explored" must become observable

room_concept already computes it (`layout_last_plan_.finished`, `readiness_explore_`,
`room_scene_graph.cpp:2201-2206`). Publish it on the room node as two floats (one cortex registration each, user
reinstall): `room_explore_readiness` ∈ [0,1] (the explorer's remaining-gain fraction, continuous — no threshold) and
`room_layout_readiness` (`readiness_stable_`). `explored(room)` is then `prob(room_explore_readiness) ≥ 0.9` with
the boundary stated in the mission (a preference: "explore it well" → 0.95, "have a quick look" → 0.6). Until the
attribute lands, the fallback is the protocol-level proxy "no `afford_room` has been OFFERED for N s", which is
observable today but is a timer, so flag it as the stopgap it is.

### 3.4 Standpoints for `go_to(object)` / "stop next to"

The controller already turns an object footprint + stand-off band into feasible viewpoints for a
`ViewpointConstraint` (`affordance_protocol.h:103-125` describes the generator; faces × stand-off band × footprint
test × reachability). Reuse the same generator with `faces = all`, `standoff = [0.5, 1.0] m` (user said "next to"),
pick the reachable candidate with the shortest route, and submit it as a manual target. If the object publishes its
own affordance, prefer *its* standpoint (that pose was designed to see the object), which is what `look_at` uses.

### 3.5 Failure propagation

Each step reports an `outcome`; composites fold them: `seq` stops at the first non-satisfied unless wrapped in `try`;
`repeat` stops on failure; `monitor` turns a false condition into `abandoned`; `timeout` into `timeout`. The mission's
`on_fail` default is `abort` (robot holds, mission state `failed|<step>|<outcome>`), selectable per step. The chat
sees every transition (§3.6) and the model can propose a follow-up ("the door would not open; shall I try again or
go round?") — the person decides, as today's card does.

### 3.6 Progress and status (what "what are you doing?" reads)

- Controller → edge `Shadow --has_intention--> mission_<n>`: `state` (`in_progress|completed|aborted|failed|…`),
  `timestamp_agent` (which mission), and `bt_state` reused as `"<step_path>|<step_state>|<last_event>|driving=on/off"`
  where `step_path` is e.g. `3/7:cross door_19:open(executing)`. One new edge attribute `mission_progress` (string,
  JSON: `{step:3, of:7, steps:[{id, op, state, outcome, t0, t1}], current_affordance, last_event}`) when `bt_state`
  gets too cramped — written only on change.
- chat_agent `tick_status` already prints transitions and appends `robot_status` tool messages on terminal states
  (`chat_session.cpp:598-635`); extend it to step transitions ("step 3/7 done: crossed door_19 → room_2").
- The STATE digest gets a `mission:` section: title, step k/n, current primitive and affordance state, elapsed,
  the last event, and the outcomes of finished steps. The model answers "what are you doing" from it, nothing else.
- A read-only tool `mission_status()` returns the same in more detail; `cancel_mission()` compiles to `stop` of the
  mission (not of the base) — `abort_all` remains the hard stop.

### 3.7 New things, by owner

| What | Owner | Kind |
|---|---|---|
| `mission_<n>` intention node, grounded IR in `current_intention` (JSON) + anchors | chat_agent | reuse existing node type and attribute; one new attr `mission_progress` on the edge (controller) |
| `MissionResolver` (refs, preds, route), `MissionSchema` (JSON schema), `MissionRender` (text form) | chat_agent, `src/mission_*.h` (pure, testable like `command_gate.h`) | new code |
| `MissionInterpreter` (step machine), `set_goal_prior`, `DoorPassage::start(door, stop_after)`, standpoint-for-node, `wait`, Orient-to-node | controller + `common/affordance_manager` | new code |
| `room_explore_readiness`, `room_layout_readiness` on the room node | room_concept | 2 cortex attrs |
| `ltsm_topology` digest node; `room_label` attribute, persisted | ltsm_agent | 1 node (metaconcept), 1 cortex attr |
| `format` (JSON schema) in the ollama request; a second non-streaming request type | chat_agent `ollama_client` | small |

No new agent is needed for the minimum version. If the interpreter later wants its own node and presence (e.g. a
`mission_agent` that survives a controller restart), id 26 is free (`CONCEPT_AGENT_RECIPE.md:490-496`).

---

## 4. The LLM side

### 4.1 Two requests, two jobs

1. **Conversation** (unchanged): streaming, tools. Add one tool `plan_mission(request: string)` whose description says
   *"use for anything that is more than one step, names a place or object to go to, or has a condition"*. The model
   passes the person's request (and any clarification) verbatim.
2. **Planning**: `chat_agent` issues a second `/api/chat` with `stream:false`, **no tools**, `format: <mission JSON
   schema>`, `temperature 0`, a compact planner system prompt, the **JSON world digest** (not the prose STATE), 5-6
   few-shot pairs, and the request. The output is parsed → resolved → either a card ("Plan: 1. go to room_2 through
   door_19 … Confirm?") or a clarification question returned to the conversation as the tool result.

Ollama's `format` accepts a JSON schema and constrains decoding with a grammar (llama.cpp json-schema→GBNF), so the
output is always parseable; `enum`, `oneOf`/`anyOf`, `$ref`, arrays and `required` are supported. Keep it shallow.

### 4.2 The schema (sketch, depth ≤ 2)

```json
{ "type":"object", "required":["steps"],
  "properties":{
    "title":{"type":"string"},
    "ask":{"type":"string"},                       // alternative to steps: a clarification question
    "steps":{"type":"array","maxItems":12,"items":{"$ref":"#/$defs/step"}}},
  "$defs":{
    "ref":{"type":"object","required":["kind"],"properties":{
        "kind":{"enum":["room","door","object","anchor","person"]},
        "name":{"type":"string"},                  // exact STATE name or a label ("kitchen")
        "between":{"type":"array","items":{"type":"string"},"maxItems":2},
        "type":{"type":"string"},                  // object class
        "in":{"type":"string"},                    // room name/label
        "on":{"type":"string"}, "near":{"type":"string"}, "nearest":{"type":"boolean"}}},
    "pred":{"type":"object","required":["p"],"properties":{
        "p":{"enum":["door_open","door_passable","exists","in_room","near","explored","person_present","elapsed_s"]},
        "ref":{"$ref":"#/$defs/ref"}, "ref2":{"$ref":"#/$defs/ref"},
        "ge":{"type":"number"}, "lt":{"type":"number"}, "d_m":{"type":"number"}, "s":{"type":"number"},
        "not":{"type":"boolean"}}},
    "prim":{"type":"object","required":["op"],"properties":{
        "op":{"enum":["go_to","cross","open","explore","look_at","find","wait","say","anchor","move","stop"]},
        "to":{"$ref":"#/$defs/ref"}, "door":{"$ref":"#/$defs/ref"}, "room":{"$ref":"#/$defs/ref"},
        "target":{"$ref":"#/$defs/ref"}, "what":{"$ref":"#/$defs/ref"},
        "until":{"$ref":"#/$defs/pred"}, "budget_s":{"type":"number"}, "s":{"type":"number"},
        "standoff_m":{"type":"number"}, "text":{"type":"string"}, "name":{"type":"string"},
        "distance_m":{"type":"number"}, "turn_deg":{"type":"number"}}},
    "step":{"oneOf":[
        {"$ref":"#/$defs/prim"},
        {"type":"object","required":["op","steps"],"properties":{
            "op":{"enum":["repeat","if","try","timeout","monitor"]},
            "times":{"type":"integer"}, "cond":{"$ref":"#/$defs/pred"}, "s":{"type":"number"},
            "steps":{"type":"array","items":{"$ref":"#/$defs/prim"}},
            "else":{"type":"array","items":{"$ref":"#/$defs/prim"}},
            "on_fail":{"enum":["skip","abort","retry"]}}}]}}}
```

Inner `steps` hold primitives only (depth 2). Anything deeper is the resolver's job to reject with "too complex —
split it", which the model relays.

### 4.3 The planner prompt (≈ 600-700 tokens)

- Who/what: "You translate a request into a mission for a wheeled robot. Output ONLY the JSON." Then the primitive
  list with one line each and the **two rules the gate cannot check**: *never invent names — use the WORLD block's
  names or a plain label; if the request is unclear, output `{"ask": "..."}`*.
- Conventions: "here/where I am now" → `anchor(start)` captured automatically (the resolver always captures `start`;
  the model only references it); "come back" → `go_to anchor:start`; "the door between A and B" → `door.between`;
  "if the door is shut" → `pred door_open lt 0.5`; "twice" → `repeat times 2`; "wait for me" → `wait until
  person_present 1.5`.
- WORLD (JSON, from `WorldDigest`, ≈ 200-600 tokens): `{robot:{room, pose}, rooms:[{name, label?, live, doors:[{name,
  p_open, to_room?}]}], remembered:[…from ltsm_topology…], objects:[{name, type, room}], anchors:[…], persons:[…]}`.
  Only names that resolve are listed; the resolver still re-checks.
- Few-shots: the six driving examples (§5), as request → JSON pairs. Keep them short; they are what teaches the
  8B model the shape. Measured by the existing harness pattern (`tests/check_llm_tools.py`): grade exact `op`
  sequences and refs over N repeats; target ≥ 95% on the six and ≥ 90% on 20 paraphrases before shipping.

### 4.4 Validation and clarification

Resolver outcomes map to three tool results: `OK: plan <text form>; a card is shown` (→ card), `ASK: <question>`
(→ the model asks the person, verbatim), `REJECTED: <reason>` (→ the model explains). The card shows the grounded
text form (§4.5) so the person confirms *what will actually happen* (`door_19`, not "the door"). On confirm, the
resolver re-runs (as `confirm_card` re-validates today, `chat_session.cpp:537-560`) before publishing.

### 4.5 Text form (for the card, logs, `remote_cmd.txt`)

```
mission "patrol then wait by the fridge"
  repeat 2
    go_to room room_1 via door_27        # passage_0, crossed 2x, P(pass)=.83
    go_to room room_2 via door_19
  go_to object refrigerator_1 standoff 0.8
  wait until person_present 1.5 m
```

The same text parses back (`parse_direct` extension) so a mission can be typed or dropped into `remote_cmd.txt` by a
test script exactly as `round_trips.sh` does today with `cross door_N`.

### 4.6 Keeping it cheap on an 8B model

- Planner request only when the conversation model calls `plan_mission`; simple commands keep the existing tools.
- `format` + `temperature 0` + `num_predict ≤ 400`; the planner prompt is a constant prefix (KV-cache hit across
  missions with `keep_alive`); the WORLD block is JSON with short keys; few-shots ≤ 60 tokens each.
- No `think`; the instruct model is already the chosen one for latency (`config.toml` `[LLM]` comment).
- The resolver, not the model, does geometry (`on`, `near`, routes) and all name checking.

---

## 5. Worked examples

Robot in `room_2`; doors of room_2: `door_19` (behind it: remembered `room_1`, P(open)=.12), `door_31`; objects in
room_2: `table_1`, `chair_2`, `bottle_1`; `ltsm_topology`: `passage_0 = {room_1:door_27 ↔ room_2:door_19,
crossings 2}`; `refrigerator_1` is remembered in `room_1`.

**1. "go to room 2, explore it and come back here"** (said in room_1)
```json
{"title":"explore room_2 and return","steps":[
  {"op":"go_to","to":{"kind":"room","name":"room_2"}},
  {"op":"explore","room":{"kind":"room","name":"room_2"},"until":{"p":"explored","ge":0.9},"budget_s":300},
  {"op":"go_to","to":{"kind":"anchor","name":"start"}}]}
```
Resolve: `start` ← (room_1, x=0.42, y=-1.10, yaw=88°); route room_1→room_2 = passage_0 → live face `door_27`
(it carries `recall_room_2`). Compiled: `seq( cross door_27 [monitor in_room(room_2)], explore(room_2, until
readiness≥.9, 300 s), cross <live face of passage_0 in room_2> [monitor in_room(room_1)], go_to pose(room_1,
0.42,-1.10) )`. Execution: `DoorPassage` → room_2 born/recalled → Command mode with the selector free → room_concept's
`afford_room` is taken repeatedly until `room_explore_readiness ≥ .9` (or 300 s) → the interpreter resolves the
return face on arrival (`door_19`, the one with `recall_room_1`) → crossing → room_1 is re-created in its
remembered frame → point target (0.42,-1.10) → arrival → `completed`. Status lines: "step 1/4 cross door_27:
approach executing", "… passage DONE → room_2", "step 2/4 explore room_2: readiness .62", …

**2. "go to the table in the kitchen and look at the bottle on it"**
```json
{"steps":[
  {"op":"go_to","to":{"kind":"object","type":"table","in":"kitchen"},"standoff_m":0.8},
  {"op":"look_at","target":{"kind":"object","type":"bottle","on":"table"}}]}
```
Resolve: "kitchen" → `room_label` if set; else the room whose objects include `refrigerator`/`hood` (room_1,
score 2 nats over room_2) → go_to(room_1) prefix inserted; `table in room_1` is a remembered room's object ⇒
`Deferred` (re-resolve after entry; the digest's `where_is` already knows the room of an object,
`object_facts.h`). On arrival: `table_1` → standpoint from the generator or `aff_table_1`'s pose → Target. Then
`bottle on table_1` → `bottle_1` (footprint inside the table's, base at table height) → `goal_prior{aff_bottle_1:+κ}`;
the bottle's Servo contract (`default_contract_for("bottle")`) completes when `bottle_detection_alive` holds →
`satisfied`. If no bottle resolves on arrival: `ASK` is not possible mid-mission, so the step fails `not_found` and
the chat reports "I am at the table but I have not perceived a bottle on it".

**3. "check whether the door between room 1 and room 2 is open; if it is shut, open it"**
```json
{"steps":[
  {"op":"say","text":"door status"},
  {"op":"if","cond":{"p":"door_open","ref":{"kind":"door","between":["room_1","room_2"]},"lt":0.5},
   "steps":[{"op":"open","door":{"kind":"door","between":["room_1","room_2"]}}],
   "else":[{"op":"say","text":"it is already open"}]}]}
```
Resolve: passage_0 → live face in the current room = `door_19`; `door_open_prob` = .12 ⇒ cond true. The `say`
step is rendered by chat from the live value ("door_19: P(open)=0.12 — probably shut"). `open` →
`DoorPassage::start("door_19", stop_after=Open)`: approach (Servo on `door_reach_prob`) → `open` offered (in reach ∧
shut) → claimed → door_concept requests actuation → Orient until `door_open_prob ≥ .70` → `satisfied`, passage stops
before `cross`. Failure: `open` times out (`open_timeout_s`) ⇒ `door_wont_open` ⇒ mission `failed`, chat: "I asked
the door to open but it still looks shut".

**4. "patrol rooms 1 and 2 twice, then wait by the fridge"**
```json
{"steps":[
  {"op":"repeat","times":2,"steps":[
     {"op":"go_to","to":{"kind":"room","name":"room_1"}},
     {"op":"go_to","to":{"kind":"room","name":"room_2"}}]},
  {"op":"go_to","to":{"kind":"object","type":"refrigerator"},"standoff_m":0.8},
  {"op":"wait","until":{"p":"person_present","d_m":1.5}}]}
```
Compiled: 4 crossings alternating through passage_0 (each leg re-resolves the live face), then go_to(room_1) is a
no-op (already there after the 4th leg? no — after leg 4 the robot is in room_2, so the resolver inserts cross →
room_1), standpoint near `refrigerator_1`, then `wait` (base stopped, armed) until a `person_N` within 1.5 m or the
person types "I'm here" (chat → `resume`). A wait with no `until` and no `s` is refused ("wait for what?").

**5. "find a chair in room 2 and stop next to it"**
```json
{"steps":[{"op":"find","what":{"kind":"object","type":"chair"},"in":{"kind":"room","name":"room_2"},
           "then":[{"op":"go_to","to":{"kind":"object","type":"chair","in":"room_2","nearest":true},"standoff_m":0.6}]}]}
```
`find` expands to `if exists(chair in room_2) → then` else `explore(room_2, until exists(chair), budget 180 s) →
then`. The `exists` predicate is evaluated by the interpreter each cycle from the graph (`get_nodes_by_type("object")`
filtered by subtype + `room_of`), so a chair born by chair_concept mid-exploration ends the explore step at once.
`go_to` with `nearest` picks the closest chair at that moment.

**6. "go back to where you started" / "wait here until I come"**
```json
{"steps":[{"op":"go_to","to":{"kind":"anchor","name":"start"}}]}
{"steps":[{"op":"anchor","name":"here"},{"op":"wait","until":{"p":"person_present","d_m":1.5}}]}
```
`start` is the pose captured when the *first* mission of the conversation began (kept on the mission node and
re-captured per mission as `start_<n>`; "where you started" defaults to the oldest). If the anchor's room is not the
current room, the resolver prefixes `go_to(room)`. "Wait here" captures `here` (so a later "go back to where I told
you to wait" resolves) and holds.

---

## 6. Architecture and phased plan

### 6.1 Components

```
 chat window ─► ChatSession ─► [conversation request + tools]
                    │  plan_mission(request)
                    ▼
            MissionPlanner (ollama, format=schema)  ──► JSON
                    ▼
            MissionResolver (C++, graph of this cycle; ltsm_topology; anchors) ──► grounded IR | Ask | Reject
                    ▼  card confirmed
            IntentionNode::send(mission JSON)   (node mission_<n>, has_intention from Shadow)
                    ▼ DSR
 controller: ChatInbox.poll ─► MissionInterpreter.start(IR)
                    ├─ cross/open  → DoorPassage(door, stop_after)
                    ├─ go_to pose  → set_manual_target (Target)         ┐ one owner of the base at a time,
                    ├─ explore     → Command mode, selector free        │ selected by the interpreter per step
                    ├─ look_at     → set_goal_prior / step_orient       │
                    ├─ wait/move   → stop_robot / RelativeMotion        ┘
                    └─ predicates  → graph reads each cycle
                    ▼
            ChatInbox.publish(state, bt_state, mission_progress) on the edge ─► chat tick_status ─► STATE + lines
```

### 6.2 Phases, effort, what each unlocks

| Phase | Build | Unlocks | Effort |
|---|---|---|---|
| **0. Make the facts observable** | room_concept publishes `room_explore_readiness`/`room_layout_readiness`; ltsm publishes `ltsm_topology` (+ `room_label`); cortex: 3 attrs (user reinstall) | predicates `explored`, routes, "the kitchen" | 1-2 days |
| **1. Minimum useful mission** | chat: schema + planner request (`format`), resolver for `room/door/anchor` refs, `go_to room/anchor`, `cross`, `open`, `explore(until/budget)`, `wait(s)`, `say`, `repeat`, `try`, text form, card; controller: `MissionInterpreter` with `seq/repeat/try`, `DoorPassage.stop_after`, point target from a pose, explore = free selector, `mission_progress`; chat STATE `mission:` section + step lines | examples 1, 3 (check+open), 4 without the fridge, 6 | 4-6 days |
| **2. Objects** | resolver `object` refs (`type/in/on/near/nearest`, `Deferred`), standpoint-for-node, `look_at` via `set_goal_prior` + orient fallback, `find`, `if`, `monitor`, `timeout`, `person_present` | examples 2, 4, 5 | 3-4 days |
| **3. Fleet-visible goals** | expand steps into `task` child nodes (`following_action` chain, controller `executing` edge to the active step) only if ltsm (route annotation) or room_concept (explore target) reads them; `MissionWithAffordances` as a bounded epistemic prior between legs | other agents condition on goals; opportunistic looks on the way | 3-5 days, when a reader exists |
| **4. Hardening** | planner eval harness (`check_llm_missions.py`, 6 + 20 paraphrases × N), Spanish few-shots, `round_trips.sh`-style mission scripts, resume after controller restart (re-read the mission node, `kFirstSightMaxAgeMs` relaxed for missions with `state=in_progress`) | reliability numbers before the project lead sees it | 2-3 days |

Order of first commits inside Phase 1: (a) `mission_schema.h` + `mission_ir.h` + `mission_resolver.h` as pure
headers with tests next to `test_gate.cpp`; (b) controller `MissionInterpreter` driven by the text form from
`remote_cmd.txt` (no LLM in the loop — the bench must run the agent's code); (c) the planner request; (d) the card.

**Status 2026-10-17.** (a) is `common/mission/` (IR + registry, JSON codec, DSL; the resolver is still to come).
(b) is built: `common/mission/mission_interpreter.h` (the step machine over a `Host` interface, tested on a scripted
host) and `controller/src/specificworker_mission.cpp` (the controller's Host). A `remote_cmd.txt` whose first line
is `mission ["title"]` with indented steps below is run as one mission; `cancel` and `resume` are words in both the
remote file and the chat grammar; a chat payload starting with `{` is decoded as a mission (§7.5). Limits of (b):
`go_to` takes a pose in the current room only (no final facing yet); a passage ref resolves only through ltsm's live
`passage_<n>` (while the room beyond is proto); `explored(room)` is unknown until its Phase-0 attribute exists;
progress travels in `bt_state` (no `mission_progress` attribute yet).

### 6.3 Risks and mitigations

- **Door crossing reliability is the floor** (3-5 of 6 legs today, memory note `resume-door-passage-two-problems`).
  A patrol is as reliable as its worst leg. Mitigation: `try/retry(n)` per leg, `monitor(in_room)` to catch
  wrong-room births, and report honestly; do not hide a failed leg inside a loop.
- **Frame validity of anchors after re-entry** (seam xy ≈ 26 cm in the 10-12 note). A point target with a 25 cm
  arrival threshold tolerates it; a `look_at` from an anchor may not. Mitigation: anchors carry σ; the resolver inflates
  the arrival tolerance by it (a covariance, not a gate).
- **One writer per node** (`chat_inbox.h:14-21`): the IR must be written ONCE per mission; per-step status goes on
  the controller's edge; Phase 3's child nodes need one owner each (chat creates, controller writes only edges).
- **8B planner drift**: grammar-constrained decoding guarantees shape, not sense. The resolver is the safety net (closed
  ops, names re-checked, card shows the grounded plan). Measure with the harness before trusting it; keep the
  conversation tools for single actions so a planner regression cannot break "stop" or "cross door_19".
- **Threshold creep**: every number in a mission (0.5 decision, budgets, standoffs, `d_m`) is the user's stated
  preference, rendered on the card. Keep it that way: no defaults hidden in the interpreter beyond the Bayes 0.5.
- **Cortex reinstall latency**: Phase 0 needs the user; everything in Phase 1 that does not depend on the new
  attributes (routes via `recall_*` one-hop, explore by budget) can land first.
- **Two sessions on one checkout** building controller + chat_agent concurrently: `-j8`, check `pgrep -a cc1plus`
  (CLAUDE.md build rules).

### 6.4 What this does NOT change

The stop fast path, the yes/no card, `run_mission_command`'s single-command grammar (missions are *added* as
`mission <text>` / the mission node, the old commands keep working), the affordance protocol's wire format, and the
producers. A mission that the interpreter is running is just another thing `abort_all` ends.

---

## C. §7 — Extensibility, expressiveness and syntax (added 2026-10-17)

Not part of Fable's text. Terminology: the **IR** (intermediate representation) is the *grounded* mission — the
step tree after the resolver has turned every reference into a node name, pose or passage id and every predicate into
an attribute read with its decision boundary. chat_agent writes it into `current_intention`; the controller's
`MissionInterpreter` executes it. The LLM never writes the IR directly and the controller never sees anything less
grounded.

### 7.1 Extensibility

**Where each kind of addition lands**

| Addition | Example | Change |
|---|---|---|
| New object type | "go to the sofa" | **none** — the type is a string matched against `object_subtype` |
| New predicate | `battery_low`, `person_waving` | one evaluator |
| New reference kind | "the corridor", "the left side of the table" | one resolver function |
| New primitive | `follow(person)`, `dock`, `pick` | one op entry + the controller mechanism behind it |
| New combinator | `parallel`, `for_each` | one interpreter node (the expensive case, §7.2) |
| New optional field | `speed:"slow"` | the field and its default |

**Rule: the IR states goals over outcomes and beliefs; object-specific actions belong to the producers.** Which
action is possible on which object, with what preconditions and expected information gain, is the producer's
knowledge (the affordance protocol); which offer to take is the selector's choice. An op is justified only by a
controller mechanism the selector cannot reach through a goal prior — point target, relative motion, wait. `look_at`
works on any `aff_<x>` through `set_goal_prior`; there is no `look_at_table`. A new concept agent is then reachable
from chat the day it publishes its affordance, with no IR change. *(Revised 2026-10-17: the first wording, "ops map to
controller mechanisms, not to object types", read as a fixed verb applied to a known noun, which breaks for
interactive perception — §7.6.)*

**Mechanisms that keep the agents in agreement**

1. **One registry, one source of truth.** Each op is one entry in `common/mission/` declaring its fields, the
   resources it claims (§7.4), validation, text form and JSON-schema fragment. The LLM schema, the resolver's validation, the
   interpreter's dispatch, and the DSL parser/printer (§7.3) are all generated from that table, so an op cannot exist
   in one place and not another. Same reasoning as `common/world_query/door_query.h`, applied to the language.
2. **Compatibility rules.**
   - An unknown `op`, ref `kind` or predicate rejects the **whole mission**, `failed|unsupported:<name>`, before
     anything moves.
   - An unknown *optional* field is ignored.
   - A field whose meaning matters (one that cannot be safely ignored) is listed in `"requires":[…]`; a controller
     that does not know it rejects the mission.
   - `"v"` is bumped only when the meaning of an existing field changes.
3. **Capability advertisement (deferred).** The controller publishes `mission_caps` (op/predicate/kind names + `v`)
   on its agent node and chat builds the LLM schema from it, so an older controller makes the LLM unable to *propose*
   what it cannot run. Needs one cortex attribute (user reinstall); worth it once controller and chat_agent can run
   at different versions.

### 7.2 Limits of expressiveness

The planned IR (`seq/if/repeat/try/monitor/timeout` + primitives) is **structured control with no variables and a
single thread**. What that rules out, and the extension that would close each gap:

| # | Limit | Requests it rules out | Extension |
|---|---|---|---|
| **L1** | No variables or bindings: a step's result cannot be referred to later | "go back the way you came"; "find a bottle and take me to *it*"; "the room where you saw the chair" | a step may declare `"bind":"b"` for its resolved result (node, pose, door); refs may be `{"var":"b"}`. `find`'s `then` already needs this |
| **L2** | No iteration over a set | "visit every room"; "check all the doors"; "look at each table" | `for_each {var, in: Query, steps}`, query evaluated when the loop starts; the set is finite, so the loop ends |
| **L3** | No concurrency; `monitor` can only abort | "go to the kitchen and tell me if you see a person"; "look for bottles on the way" | `parallel` with typed resources — at most one branch owns the base, the others are observers (perception, `say`, binding); `when {pred, do}` lets an event trigger an action without aborting |
| **L4** | Procedural only: the user must say *how*, not *what* | "make sure the door is open"; "get to room 3 somehow"; "avoid the kitchen"; "go quietly" | (a) mission-level `prefs`: costs in nats over places/affordances that reshape G across all steps (avoid = negative prior; fast vs careful = cost weights); (b) declarative `achieve {pred}`, compiled to a prior on the affordances whose completion clauses make the predicate true — needs producers to declare their effects, which contracts partly do already |
| L5 | No dialogue mid-mission | "if there are two tables, ask me which" | `ask {question, candidates, bind}` suspends the mission (`state=waiting_user`); the answer arrives as a `reply` request carrying the mission's stamp |
| L6 | No schedules or standing orders | "every 10 minutes"; "always close doors behind you" | out of scope for the IR: a chat-side scheduler issues ordinary missions; standing orders would be persistent `prefs` |
| L7 | Unclear *when* a predicate is evaluated | `if door_open` at P = 0.5 from no observation is a coin flip | a semantics rule rather than a feature: a decision step first takes the **epistemic** affordance on that belief until the posterior is clear of the boundary or the step's budget runs out — the active-inference reading of "check whether X" |
| L8 | Planner depth | the 8B planner reliably emits depth ≤ 2 | a limit of the **planner**, not the IR (which nests without bound); deeper missions are built over several turns |

Things the robot cannot do (manipulation, following a person) are **capability** limits, not expressiveness limits:
once the controller can do them, each is a new op.

**The ceiling is deliberate.** Even with L1–L5 the IR stays a fixed program run over world predicates, with finite
sets and a budget on every loop. That keeps the two properties that make it safe: every mission **terminates**, and
it can be **checked before it runs** (every reference resolved, every op supported). A general programming language
(recursion, arithmetic, unbounded state) would lose both, and no request we have considered needs it.

**L4 is the Phase-3 headline, not an extra.** It is where the design becomes active inference rather than a script
runner with priors added on. Its `prefs` slot is reserved in v1 (§7.4).

### 7.3 Concrete syntax: JSON, a DSL, or both

A DSL is not an alternative to the IR; it is another way of **writing** it. The IR is the abstract syntax (the tree
and its meaning); JSON and a DSL are concrete syntaxes. Switching between them changes neither what can be said nor
what cannot. The question is which syntax goes where:

| Where | JSON | DSL |
|---|---|---|
| **LLM output** | ollama's `format` constrains decoding to a JSON schema, so the 8B model always emits well-formed output | ollama's API takes a JSON schema, not a custom grammar (llama.cpp supports GBNF; ollama does not expose it), so a DSL is generated unconstrained and needs a parse → error → retry loop; roughly half the tokens and latency |
| **Wire** (`current_intention`) | unknown optional fields can be ignored, so old and new builds tolerate each other; parsers exist | fine with `key=value` arguments, but we own the parser |
| **People** (card, logs, `remote_cmd.txt`, test scripts) | poor to read and write | clearly better |

**Decision: one IR, two syntaxes, round-trip tested.**
- **JSON** for LLM output and the wire: schema-constrained decoding is the only well-formedness guarantee on an 8B
  model.
- **DSL** for everything people read or write, including `remote_cmd.txt` mission scripts that exercise the
  interpreter with no LLM in the loop (amendment 3, Part A).

**The DSL is generic, generated from the registry.** One shape for every op, so the grammar never changes when an op
is added:

```
mission    := 'mission' STRING? NEWLINE step+
step       := op positional* (key '=' value)* block?      -- one per line; indentation = nesting
block      := INDENT step+ DEDENT  ('else' INDENT step+ DEDENT)?
value      := NUMBER unit? | STRING | NAME | ref | pred
ref        := NAME                         -- a live node:            door_27, table_1, room_2
            | 'anchor:' NAME               -- a captured pose:        anchor:start
            | 'passage:' NAME '@' NAME     -- resolved on arrival:    passage:passage_0@room_2
            | '$' NAME                     -- a binding (L1):         $found
pred       := NAME '(' ref (',' ref)* ')' (('>=' | '<') NUMBER)?
unit       := 's' | 'm' | 'deg'
```

```
mission "explore room_2 and return"
  cross door_27 expect=room_2 retry=2
  explore room_2 until=explored(room_2)>=0.9 budget=300s
  cross passage:passage_0@room_2 expect=room_1 retry=2
  go_to anchor:start
```

Each op's registry entry already names its positional and keyword fields; the DSL just spells them out. A per-op
hand-written syntax would make every extension a grammar change, so it is ruled out.

**When to reverse the LLM-side choice.** If measured planner latency is too high (JSON roughly doubles the output
tokens, and output tokens are most of the cost on the 8B model), let the LLM write the DSL with a retry loop and
compare the two in the existing harness (`tests/check_llm_tools.py`): validity rate, accuracy on the six §5 examples
plus 20 paraphrases, and latency. The IR does not change either way, so the experiment is cheap; run it before
Phase 1(c).

**Rejected: an embedded DSL** — the LLM writes Python or Lua that calls robot functions (the code-as-policies
approach). It gets variables, loops and concurrency (L1–L3) for free but loses what makes the IR safe to run: no
check before execution that a mission terminates or that its references resolve; an 8B model's code runs on the
robot; and conditions stop being probabilities with a stated decision boundary and become arbitrary `if`s. Adding
L1–L3 to the IR as limited constructs gets the useful part of that power without losing those guarantees.

### 7.4 What v1 must reserve now

Implement only the Phase-1 ops, but put these in the format from day one — adding any later changes the wire format:

1. **A step id on every step** (`"1"`, `"2.3"`), used by `mission_progress`, `bind`, `ask` and replies.
2. **`bind` / `{"var":…}` / `$name`** accepted by the grammar, rejected by v1's validation.
3. **The resources each op claims, as a set** (`base` · `arm` · `gripper`, plus an `instant` flag) in the registry,
   so `parallel` can be added later and the interpreter enforces one owner per resource. A composite claims the union
   of its body. *(Revised 2026-10-17 from a single class with one owner of the base: manipulation claims the arm and
   gripper separately — §7.6.)*
4. **A mission-level `prefs` slot**, empty in v1.
5. **The compatibility rules** of §7.1: whole-mission reject on unknown op/kind/predicate, unknown optional fields
   ignored, `requires`, `v`.
6. **The registry as the single source** of the schema, validation, dispatch and DSL.

Deferred: `mission_caps` (cortex reinstall) and every extension of §7.2 beyond the reserved slots.

### 7.5 Protocol decisions

Agreed 2026-10-17:

1. **Keep the single `chat_intention` node** (each mission is a new stamp) instead of a `mission_<n>` node per
   mission (§1, §3.7 above). Per-mission nodes add create/delete churn, break the exact-name `[Owns]` entry, and make
   `ChatInbox` follow a moving node id. The controller tells the payloads apart by the first character: `{` = a
   mission, anything else = the existing single-command grammar.
2. **The IR, resolver and registry live in `common/mission/`**, not `chat_agent/src` (§3.7 above). The controller
   must re-resolve passage and deferred references on arrival; two resolvers could disagree.
3. **Which requests replace a running mission.** `stop` aborts everything; `cancel_mission` stops the mission and
   leaves the base armed; `resume` ends only the current `wait`; `reply` answers an `ask` (L5); anything else
   replaces the mission and reports `aborted|superseded`.
4. **The 3 s first-sight rule does not apply to a mission whose edge says `in_progress`**: a restarted controller
   resumes it rather than adopting its stamp. A protocol rule, so it belongs in v1 of the spec even if it is
   implemented in Phase 4.

### 7.6 Interactive perception: where the action/object line really is (added 2026-10-17)

**The edge.** Read literally, "ops map to mechanisms, not object types" assumes every object-directed act is a fixed
verb applied to a known noun. Exploring an unknown object by manipulating it, to learn its *affordance signature*,
breaks that three ways:

1. **The affordance does not exist before the action.** Poking, pushing or lifting is how the robot finds out what
   the object affords. A grounded ref needs a node, and `look_at` needs that node to offer an `aff_<x>` already; an
   unknown object has neither a type nor an offer.
2. **The right action depends on the object, and the mission author cannot choose it.** Where to push, which grasp,
   how hard — these come from the object's uncertain geometry and the predicted information gain. A fixed `poke` op
   with planning-time parameters puts that decision in the wrong place.
3. **Action and object are defined together.** An affordance is a relation between agent and thing, not a property
   of either. A grammar with verbs in one table and nouns in another has no place for "the action whose outcome tells
   me what this object is".

The Phase-1 registry already leaks the other way: `cross` and `open` are door-specific verbs.

**The line that survives manipulation.** Not action vs. object, but:

- the **mission** states *what it wants*: an outcome, or a belief made certain enough;
- the **producers** own the coupling of actions to objects: which actions are possible on which thing, with what
  preconditions, at what expected information gain;
- the **selector** chooses which offer to take, and the **controller** executes it.

That is the existing affordance protocol (producers offer, contracts define completion, the selector minimises G);
the IR only has to address it at the right level.

**Planned path** (all new ops and predicates — no wire change; the §7.1 compatibility rules cover them):

| Addition | Meaning | Compiles to |
|---|---|---|
| `learn {target: Ref, belief?, until?, budget_s}` | reduce uncertainty about one node's belief | a goal prior on the epistemic affordances that target that node, whatever the producer offers |
| `gain(node, belief) < nats` | remaining expected information gain of a belief (amendment 2, generalised) | a read of the producer's published gain |
| `affords(node, kind) >= p` | the learnt signature: probability the node affords `kind` | a read of the signature on the node |
| `achieve {pred}` (L4) | a declarative goal | a prior on the affordances whose completion clauses make `pred` true |

Example — explore an unknown object and act on what was learnt:

```
mission "what is that"
  learn obj_12 until=gain(obj_12, affordances)<0.1 budget_s=120s
  if affords(obj_12, grasp)>=0.7
    achieve near(obj_12, person_1)
```

A manipulation producer offers `aff_obj12_push(dir)`, `aff_obj12_lift`, … each with its expected gain about the
signature; the selector picks among them. No manipulation verb appears in the IR.

The same pattern unifies what exists: `explore room_2` is `learn room_2` (layout readiness = `gain(room_2, layout)`),
and `cross door_19` becomes `achieve in_room(room_2)` with door_concept owning approach → open → cross. So the change
that removes the door special case also removes the manipulation edge.

**What is already in place.**
- Refs are type-agnostic: `{"node":"obj_12"}` works for an object of unknown type. Missing, on the resolver side:
  deictic references ("that thing", from what the robot sees or where the person points).
- Resources are a set (§7.4 point 3): base, arm and gripper have one owner each.
- `cross` and `open` are marked LEGACY in the registry.

