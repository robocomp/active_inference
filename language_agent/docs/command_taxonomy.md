# language_agent — command taxonomy: coverage audit and gap proposal (2026-11-14)

Scope: what the chat can understand and do today, classified along linguistic axes, the gaps that matter for a mobile
robot (Shadow: base + cameras + LiDARs, no arm in the current simulation), and ranked proposals. **Nothing is changed
by this document.** Code references are to `main` db8d39e3.

Reading guide: §1 inventory · §2 classification · §3 coverage matrix · §4 gaps (with ES/EN utterances) · §5 proposals
(composition first, then shared mechanisms) · §6 top-5 · §7 reorganising the tools · §8 designing for a small local
model. Execution contracts (aspect → ROS-style pattern, mismatches): `execution_contracts.md`. Test set:
`utterance_testset.yaml`. Worked schemas for four existing tools
are in `docs/tool_schema_template.json`.

---

## 1. Inventory

### 1.1 Layers a request can go through
1. **Fast paths in code, before the LLM** (`chat_session.cpp` ~410–510, `command_gate.h`): `stop/para/detente/alto`,
   `pause/pausa/espera`, `continue/sigue/continúa`, `cancel mission/aborta/cancela la misión`, the yes to a mission
   card / a claim card / an agent-control confirmation, `go to <remembered object>` (code-built card,
   `remembered_goto.h`), and a small grammar (`move 1 m`, `turn 90`, `turn 90 move 1`, `cross door_3`, `cruza la
   puerta`, `cancel passage`).
2. **Chat model + tools** (31 tool definitions, `ChatSession::tools_json`, `chat_session.cpp:648–1007`): reads,
   immediate actions, canvas, memory.
3. **Planner** (`plan_mission` → a second LLM request with `etc/skills/{missions,doors,exploring}.md`, temperature 0)
   writing the **mission IR** (`common/mission/mission_ir.h`), checked (`check_planned`, `mission_plan.h`), shown as a
   card, started only on the person's yes, run by the controller's interpreter.
4. **Robot events → wake turns** (`mission_watch`): a mission's completion/failure/skip wakes the chat with a note;
   a wake turn may only read, stop or propose (`allowed_in_wake`, `mission_plan.h:650`).

### 1.2 Chat tools (the model's verbs)

| Tool | Does | Arguments (required **bold**) | Completion / failure | Interruptible |
|---|---|---|---|---|
| `cross_door` | approach, open if shut, go through a door of the current room | **door** (enum of live doors + nearest) | controller status lines (`intention_state`), "THROUGH … passage DONE" / refusal text | yes: `stop`, `cancel_passage` |
| `sim_door` (sim only) | simulator opens/closes a door | **action** open/close, **door** | immediate text | n/a (instant) |
| `stop` | halt, abandon current action | — | immediate | — |
| `cancel_passage` | abandon a door crossing | — | immediate | — |
| `move_relative` | turn then drive | **distance_m**, **turn_deg** | status lines; refused if blocked | yes: `stop` |
| `approach_object` | drive in front of an object, facing it | **object** (name/class/"my X"), side | status lines; failure reason | yes |
| `go_around_object` | full loop around an object, back to start | **object**, direction | status lines | yes |
| `plan_mission` | multi-step / conditional / return → mission card | **request** (the chat's paraphrase) | card shown; then mission wakes (summary / failure / skipped steps) | yes: stop, pause, continue, cancel |
| `human_claim` | person states name / owns / disowns → confirmation card | **kind**, object, name | card; stored only on yes | n/a |
| `find_objects` | objects of the current room (+ colour) | subtype (class, colour word) | text | n/a (read) |
| `where_is` | where an object is now / last seen (LTOM) | **name** | text, verbatim from memory | n/a |
| `known_rooms` | remembered rooms + passages | — | text | n/a |
| `route_to` | remembered route to a room | **room** | text | n/a |
| `can_see` | would it be seen from here/there; where to stand | **concept**, where, from, stand | text | n/a |
| `imagine` | subjunctive: fit / see / succeed on an imagined copy | **question**, door, opening, via, target, from, to, between, move, distance | text | n/a |
| `imagine_view` | draw imagined instances as sensors see them | **concept**, where, from, sensor, n, seed | canvas + text | n/a |
| `agents` | software agents' state, CPU, memory | — | text | n/a |
| `agent_control` | start/stop/restart an agent | **action**, **agent** | text; controller/retina need yes | n/a |
| `sensors` | stream rates, ages, bandwidth | name | text | n/a |
| `graph_summary` | node/edge counts | — | text | n/a |
| `inspect` | one node's attributes and edges | **node** | text | n/a |
| `use_skill` | read a skill text | **name** | text | n/a |
| `episodes` | today's episodes | since, until (HH:MM), author, kind, target | text | n/a |
| `episode` | one episode in detail | **id** | text | n/a |
| `outcome_stats` | success rates/durations, all time | **kind**, target | text | n/a |
| `last_episode` | most recent episode about a node | **target** | text | n/a |
| `where_was` | the robot's room/pose at a past time + way back | ago_min \| at | text | n/a |
| `show` | ready-made canvas: camera, layout, plan, graph, model, lidar, saved | **name**, args, live | canvas | `hide_canvas` |
| `draw` | Python-scripted picture | **script**, **title**, live | canvas; error returned for a retry | `hide_canvas` |
| `table` | rows on the canvas | **title**, **rows**, header | canvas | `hide_canvas` |
| `hide_canvas` | close canvas | — | — | — |
| `look` | a camera picture to the MODEL (not the person) | camera front/360 | image in the next message | n/a |

### 1.3 Mission IR (the planner's verbs; `mission_ir.h:249–313`)

| Op | Claims | Fields (required **bold**) | Class |
|---|---|---|---|
| `seq` | — | body | composition |
| `repeat` | — | times \| until=pred, budget_s; body | iteration |
| `cross` | base | **door** (door / `room_X` / passage ref), expect, bind | accomplishment |
| `open`, `close` | base | **door** | accomplishment (via door_concept affordance) |
| `go_to` | base | **to** (pose / anchor / node), standoff_m | accomplishment |
| `approach` | base | **object**, side | accomplishment |
| `explore` | base | **room**, until=pred, **budget_s** | activity (bound REQUIRED) |
| `wait` | base | s \| until=pred \| at=ISO (one required) | state, bounded |
| `say`, `report` | — | **text** | achievement (instant) |
| `anchor` | — | **name** | achievement (instant; marks a place) |
| `move` | base | distance_m, turn_deg | accomplishment (short) / semelfactive turn |
| `stop` | — | — | achievement |

Step modifiers: `on_fail = skip | retry=N`, `bind=NAME` (on cross). Predicates: `explored(room) >= p`,
`in_room(room)`, `door_open(door) >= p`. Unknown op/ref/predicate rejects the whole mission (compatibility rule §7.1).

### 1.4 Existing support for mood and time
- **Indirect / subjunctive requests** ("could you…", "quiero que vayas…", "I'd like you to…"): no code path; the chat
  model interprets them (system prompt: answer in the person's language). Ability questions vs requests ("can you open
  doors?" vs "can you open door_1?"): only one explicit rule, for `agent_control` ("a general question whether you can
  start agents is answered yes, in words"). **Counterfactual questions** ("would / if / what if", "¿cabría?") are a
  real tool: `imagine` (fit / see / succeed), plus `imagine_view`.
- **Time:** immediate (all action tools) · relative deferred (`wait s=600s` first) · absolute deferred (`wait at=18:00`,
  grounded to ISO by language_agent; not live-tested) · bounded duration (`explore budget_s`, `wait s`, `repeat
  budget_s`) · event-triggered **only inside a mission and only on three predicates** (`wait until=door_open(…)`,
  `in_room`, `explored`) · past reference (`episodes since/until`, `where_was ago_min|at`, LTOM "last seen X ago").
  **Not supported:** recurring ("every morning"), deadlines ("before 12"), triggers on perception ("when you see the
  cup"), triggers on the person ("cuando llegue Pablo"), persistence across restarts.

---

## 2. Classification of existing items

Aspect is for the **verb phrase with its arguments** (e.g. `move` with no goal is an activity; `move 1 m` is an
accomplishment; `turn 90` is semelfactive-like; `approach chair_1` is an accomplishment with goal predicate
"in front of chair_1, facing it").

| Item | Speech act | Process | Aspect (VP) | Frame roles (req.) | Manner/Result | Mood typically | Temporal |
|---|---|---|---|---|---|---|---|
| cross_door / `cross` | directive | material | accomplishment (goal: in next room; progress: approach→open→through) | Agent, **Path**(door), Goal(expect room) | result (be in R) | imperative, indirect | immediate; in mission: any IR anchoring |
| approach_object / `approach` | directive | material | accomplishment (goal: facing object at standoff) | **Goal**(object), Manner(side) | result | imperative, "go to my chair" | immediate |
| go_around_object | directive | material | accomplishment (goal: loop closed, back at start) | **Theme/Path**(object), Manner(direction) | manner+result | imperative | immediate |
| move_relative / `move` | directive | material | accomplishment (distance) / semelfactive (turn) | **Manner**(turn), **Extent**(distance) | manner | imperative | immediate |
| `go_to` | directive | material | accomplishment | **Goal**(pose/anchor) | result | via planner | mission |
| `explore` | directive | material+perceptual | **activity** (atelic; bound required: budget_s, optional until) | **Location**(room) | manner | "explore", "look around" | bounded duration |
| `wait` | directive | — | state (maintain stillness) bounded by s / until / at | Duration \| Event \| Time | — | "wait until…", "a las 5…" | relative/absolute/event |
| `open` / `close`, sim_door | directive | material (by proxy) | achievement-ish (door state flips) | **Theme**(door) | result | imperative | immediate / mission |
| stop / cancel_passage / pause | directive | material (halt) | achievement | — | — | imperative | immediate (code fast path) |
| plan_mission | directive (complex) | any | composite (aspect = the steps') | **request** | — | imperative, conditional, temporal | all IR anchorings |
| `repeat` | directive | — | iterative (times) / activity bounded (until+budget) | Count \| Event | — | "twice", "until…" | bounded |
| `say` / `report` | directive (verbal) | verbal | achievement | **Message**, (Recipient = the person, implicit) | — | "tell me…" | inside missions |
| `anchor` | — (internal) | relational (mark a place) | achievement | **Name** | — | "remember this spot" (not exposed) | mission |
| human_claim | **informative** | relational (person owns/is named) | achievement (belief update on yes) | **kind**, Theme(object), Name | — | indicative statement | immediate |
| find_objects / where_is | question | existential / relational | state (read) | Theme(class/name) | — | interrogative, "can you see a chair?" | present (+ past via LTOM) |
| known_rooms / route_to | question | relational | state | Goal(room) | — | interrogative | present/memory |
| can_see | question | mental/perceptual | state (read) | **Phenomenon**(concept), Source(from) | — | "can you see…?" (ability form is a real question here) | present / hypothetical place |
| imagine | question | relational/perceptual on a counterfactual | state | **question**, Theme, Path(door), Source/Goal | — | **subjunctive/conditional** | hypothetical |
| imagine_view | directive (show) | verbal + perceptual | achievement | **Phenomenon** | — | "show me how you imagine…" | hypothetical |
| episodes / episode / last_episode / outcome_stats | question | relational (memory) | state | Time window, Theme(target) | — | interrogative | **past** |
| where_was | question | relational (own past) | state | **Time** (ago \| at) | — | "where were you at…" | **past** |
| look | (internal) perceptual | mental/perceptual | achievement | Instrument(camera) | — | "what do you see?" | present |
| show / draw / table / hide_canvas | directive | **verbal/semiotic** | achievement (live=true → activity until hidden) | **Theme**(what), Recipient(person) | — | "show me", "draw", "enséñame" | immediate |
| agents / sensors / graph_summary / inspect | question | relational (self-state) | state | Theme(node/agent) | — | interrogative | present |
| agent_control | directive | material (on software) | achievement | **Action**, **Theme**(agent) | result | imperative | immediate |
| (none) | expressive | — | — | — | — | "gracias", "well done" | handled as plain chat |

---

## 3. Coverage matrix

Rows = speech act × process type; columns = aspect. ✔ exists · ◐ partial · ✗ gap that makes sense · — not sensible /
out of scope (no arm). Extra columns: temporal anchorings and moods available today.

| Speech act · process | State | Activity | Accomplishment | Achievement | Semelfactive/iter. | Temporal today | Mood today |
|---|---|---|---|---|---|---|---|
| Question · relational | ✔ where_is, known_rooms, route_to, inspect, agents | — | — | — | — | present, past (episodes, where_was, LTOM) | interrogative; indirect via LLM |
| Question · existential | ✔ find_objects ("is there a chair?") ◐ (only current room live; other rooms via memory) | — | — | — | — | present, past | interrogative |
| Question · perceptual | ✔ can_see, look | — | — | — | — | present, hypothetical | ability form handled |
| Question · counterfactual | ✔ imagine (fit/see/succeed) | — | — | — | — | hypothetical | **subjunctive ✔** |
| Question · prospective ("will you make it by 12?", "how long to room 3?") | ✗ | — | — | — | — | — | — |
| Directive · material | ✗ **maintain/avoid** ("stay in this room", "no entres en la cocina") | ✔ explore (bounded) ◐ patrol/wander/follow ✗ | ✔ cross, approach, go_to, go_around, move | ✔ stop, open/close (sim) | ◐ move turn; repeat times | immediate, rel./abs. deferred, bounded; **event only on 3 preds** | imperative ✔, indirect via LLM ◐, **conditional ◐ (no if/else)** |
| Directive · perceptual | ✗ **watch/monitor** ("avísame si entra alguien") | ◐ explore (sees as a side effect) ✗ search-for-X ("busca la taza") | ✗ "look at the door" (orient gaze/body to a target as its own verb; only via move turn) | ✔ look (to the model) | — | immediate | imperative |
| Directive · verbal/semiotic | — | ◐ show live=true | — | ✔ show, draw, table, say, report | — | immediate; say/report in missions | imperative |
| Directive · verbal to a third party ("dile a Pablo que…") | — | — | ✗ (needs a person model + delivery) | ✗ | — | — | — |
| Directive · quantified ("check all the rooms", "approach every chair") | — | — | ✗ (planner unrolls names by hand; no foreach) | — | ✔ repeat times | — | — |
| Informative · relational (person tells a fact) | — | — | — | ✔ human_claim (name, owns) ✗ room names ("this is the kitchen"), ✗ object names ("call it the reading chair"), ✗ world facts ("door_2 is locked") | — | — | indicative |
| Commissive (robot commits: reminder, standing order) | ✗ standing orders ("whenever you pass door_2, close it") | — | — | ✗ reminders ("recuérdame a las 5…") | ✗ recurring ("every morning") | ✗ recurring, ✗ persistence | — |
| Expressive | ◐ plain reply (no effect on belief) | — | — | — | — | — | — |
| Directive · material on objects (grasp, push, carry) | — | — | — (no arm in sim; keep the cell for the real Shadow) | — | — | — | — |

Temporal anchoring × availability: immediate ✔ · absolute deferred ✔ (`wait at`, unverified live) · relative deferred
✔ · event-triggered ◐ (3 predicates, mission-only, base held while waiting) · recurring ✗ · bounded duration ✔
(explore/wait/repeat) ◐ (not for go_around/approach, which are telic anyway) · deadline ✗.
Mood: imperative ✔ · interrogative ✔ · subjunctive/counterfactual question ✔ (imagine) · optative/indirect request ◐
(LLM only, no eval set) · conditional directive ◐ (`wait until` only; no branch) · permission/ability ◐ (one tool rule).
Polarity: negative directive ✗ · negative question ✔ (LLM) · quantification ✗ · iteration ✔ (repeat).

---

## 4. Gaps (with utterances)

G1 **Event-triggered and conditional directives** (trigger ≠ wait-and-hold). The IR can only `wait until=` one of 3
predicates, holding the base; there is no branch and no perceptual/person trigger.
- "Cuando llegues a la habitación 3, avísame." / "When you get to room 3, let me know."
- "Si la puerta está cerrada, ábrela; si no, cruza." / "If the door is shut, open it, otherwise go through."
- "En cuanto veas la taza, para." / "As soon as you see the cup, stop."

G2 **Negative commands as constraints** (polarity). Nothing turns "don't" into a constraint on routes or actions.
- "No entres en la cocina." / "Don't go into the kitchen."
- "No cruces la puerta 2, está rota." / "Don't use door 2, it's broken."
- "Quédate en esta habitación." / "Stay in this room." (positive form of the same invariant)

G3 **Commissives: reminders, standing orders, recurring missions** (and persistence).
`wait at=17:00; say …` would work as a reminder but `wait` claims the base, so the robot is frozen until 17:00, and
nothing survives a restart.
- "Recuérdame a las cinco que llame a mi madre." / "Remind me at five to call my mother."
- "Todas las mañanas a las 9, revisa el baño." / "Every morning at 9, check the bathroom."
- "Cada vez que pases por la puerta 2, ciérrala." / "Whenever you pass door 2, close it."

G4 **Perceptual activities: search and watch** (directive × perceptual × activity/state).
- "Busca la taza." / "Look for the cup." (activity; telic once found: goal predicate seen(cup))
- "Vigila la puerta y avísame si se abre." / "Watch the door and tell me if it opens."
- "Mira hacia la puerta." / "Look at the door." (orient to a target; today only via `move turn_deg`)

G5 **Quantified directives** (foreach over a set the world defines).
- "Revisa todas las habitaciones." / "Check all the rooms."
- "Acércate a cada silla y dime de qué color es." / "Go to every chair and tell me its colour."
- "Visita dos habitaciones que no hayas visto hoy." / "Visit two rooms you haven't seen today."

G6 **Informative updates beyond the person** (the person as a sensor / namer).
- "Esta habitación es la cocina." / "This room is the kitchen." (room naming; owner = ltsm, memory note room-naming-later)
- "La puerta 2 está cerrada con llave." / "Door 2 is locked." (a told fact with source=told, lowering P(passable))
- "Llámala la silla de lectura." / "Call it the reading chair." (instance names spec)

G7 **Prospective questions and deadlines** (future × question; deadline anchoring).
- "¿Cuánto tardarías en llegar a la habitación 3?" / "How long would it take you to get to room 3?"
- "Ve a la cocina antes de las 12." / "Be in the kitchen before 12."
- "¿Llegarás a tiempo?" / "Will you make it in time?"

G8 **Mood handling as policy, not prompt luck**: ability-form requests and optatives, both languages.
- "¿Puedes abrir la puerta 1?" (request) vs "¿Puedes abrir puertas?" (ability) / "Can you open door 1?" vs "Can you open doors?"
- "Ojalá estuviera abierta la puerta." (wish, no request) vs "Quiero que vayas a la cocina." (request)
- "Would you mind going to room 2?"

G9 **"Show me where X is"** (composition relational + semiotic): today `where_is` + `show layout` cannot mark the
target, and the target is often in another room.
- "Enséñame dónde está mi silla." / "Show me where my chair is."
- "Muéstrame en el mapa por dónde vas a ir." / "Show me on the map the way you'll go."

G10 **Bounded activities of the base**: patrol, wander, follow — atelic, need a bound.
- "Patrulla la casa durante diez minutos." / "Patrol the house for ten minutes."
- "Sígueme." / "Follow me." (needs person tracking; out of scope until a person agent drives it)

---

## 5. Proposals — composition first, then shared mechanisms (ranked)

**How the taxonomy is used** (addendum): it is a design-time audit and a source of tool METADATA, not a runtime
classifier. The model reads the request and picks tools from their names, descriptions and schemas in a
think → act → observe loop. So each proposal below first asks whether existing tools compose to fill the gap; a new
tool appears only when composition is impossible, unsafe, or too unreliable for a small model. Time, conditions,
negation and iteration become **shared mechanisms that wrap any action**, never arguments repeated per tool.

### 5.1 Gaps that composition already fills (no new tool)

| Gap / utterance | Composition | Missing piece (small) |
|---|---|---|
| G9 "Enséñame dónde está mi silla" / "show me where my chair is" | `where_is("my chair")` → `show layout` | `show layout` takes a target to mark (args=node); for another room: `draw` with the remembered pose, or a `layout` of that room from LTSM (P7) |
| G1 (part) "Cuando llegues a la cocina, avísame / haz una foto" | `plan_mission`: `cross … expect=kitchen` then `say "he llegado"` / `report "lo que veo"` — "when you arrive" is SEQUENCE, not a trigger | none: a planner skill example; `report` already describes the view |
| G4 (part) "Mira hacia la puerta" / "look at the door" | `move_relative(turn_deg = the door's "to face it")` | the same "to face it" field for objects (doors have it since 583d247f) |
| G4 (part) "Busca la taza" / "look for the cup" | `plan_mission`: `explore room until=seen(cup) budget_s=…` | ONE new predicate `seen(node|class)` (mechanism M3) |
| G7 (part) "¿Cuánto tardarías en llegar a la 3?" | `route_to(room_3)` + `outcome_stats(cross)` (mean duration per crossing) | a skill line telling the model to multiply; better as `imagine question=succeed` returning duration too (extension) |
| G5 (part) "Revisa todas las habitaciones" | `known_rooms` → `plan_mission` naming every room | reliable only if code expands the set (M4), a small model drops items |

### 5.2 Shared mechanisms (where composition is not enough)

**M1 — Trigger / scheduler** (G1 non-sequential triggers, G3 commissives, recurring, persistence). One tool, typed:
`schedule(when, do, repeat?)` with `when` one of `{at: "HH:MM" | ISO}`, `{in_s: N}`, `{every: "daily HH:MM" |
"Ns"}`, `{on: predicate}` (M3 vocabulary), and `do` = a request text that is planned when it fires (the same
plan_mission path, so a mission card or, if the person said "sin preguntar", a direct start) or `{say: text}`.
Held by language_agent, persisted on disk, listed in the STATE ("standing orders: …"), cancelled by
`cancel_scheduled(id)`. **The base is not claimed while waiting** (today `wait at` freezes the robot). `wait` stays as
the HOLD form inside a mission ("wait here until 6"). Refactor: the existing `wait at=` grounding (ground_clock) moves
behind M1's `at`; one time grounding for `schedule`, `wait`, `episodes since/until` and `where_was at` (today three
separate HH:MM parsers: ground_clock, the where_was minus-one-day rule, episodes' own parser).

**M2 — Constraint store** (G2 negative commands, "stay in this room"). `constrain(kind: avoid | stay_in, target:
room|door, scope: mission | until_cancelled | for_s)`, listed in the STATE ("constraints: avoid room_3 (kitchen) until
cancelled"), `release(id)`. Enforced in ONE place per layer: route_to/legs_to filter, check_planned rejects a plan that
violates one (with the reason, so the planner retries), cross_door refuses with the constraint's text. Not a per-tool
argument.

**M3 — One predicate vocabulary** shared by `wait until`, `repeat until`, `explore until`, `if`, and M1's `on`:
today `explored`, `in_room`, `door_open`; add `seen(node|class)` (fresh observation, owned by us per ltom-chat-e2e),
`object_in_room(class, room)`, `person_present` (when a person agent drives it), `clock_after(ISO)`. Each predicate
is a belief query with a stated probability boundary (the IR's existing `cmp` rule).

**M4 — Control ops `if` and `each`** in the IR. `if pred then … else …` (conditionals, G1). `each $x in
rooms|objects(class)|doors [where pred] do …` expanded by language_agent at plan time into a grounded seq (like
passages_to_rooms), shown on the card as the expanded list, so the controller needs no new op and a small planner
cannot drop an item.

**M5 — Mood policy** (G8): not a mechanism in code but a fixed table in the system prompt + eval cases: ability form
with a concrete argument ("¿puedes abrir la puerta 1?") = request; with a generic one ("¿puedes abrir puertas?") =
capability question answered in words; optative with a present referent ("quiero que vayas…") = request; wish about the
world ("ojalá estuviera abierta") = expressive, no action (offer one); counterfactual question = `imagine`.

### 5.3 New tools only where needed

| # | Proposal | Kind | Gap | Value | Cost |
|---|---|---|---|---|---|
| P1 | M3 `seen()` + M4 `if` (conditional directives, search as explore-until-seen) | IR mechanism | G1, G4 | high | M |
| P2 | M2 constraint store `constrain`/`release` | mechanism (new tool pair) | G2 | high | M |
| P3 | M1 scheduler `schedule`/`cancel_scheduled` + one time grounding | mechanism (new tool pair) + refactor | G3, G1 | high | M–L |
| P4 | M4 `each` expanded at plan time | IR mechanism, code expansion | G5 | med–high | S–M |
| P5 | M5 mood table + eval cases | prompt + tests | G8 | med | S |
| P6 | `show layout` target marking; "to face it" for objects | extensions | G9, G4 | med | S |
| P7 | `tell_fact` (room names, told door states, object names) as claim cards | new tool | G6 | med | M (cross-agent) |
| P8 | duration in `imagine succeed` → prospective questions; `before=` deadlines later | extension | G7 | med | M |
| P9 | patrol / follow | IR ops | G10 | low now | L |

Full schema fields for the mechanisms (speech_act, process_type, aspect, arguments, temporal, mood_handling,
clarification_policy, execution_contract) follow the template in `tool_schema_template.json`; M1 and M2 sketched:

- **schedule** — speech_act commissive; process: that of `do`; aspect {class: achievement (fires) / iterative
  (every), telic: per firing, durative: no (the waiting is not the robot's), goal_predicate: `when` reached,
  progress_signal: STATE line "standing orders: #3 at 17:00 say …", interrupt: cancel_scheduled}; arguments {Time|Event
  required, Content required}; temporal {absolute, relative, recurring, event}; clarification: no time → ask; "por
  la mañana" → ask the hour; "every" with no end → allowed (listed, cancellable); execution: persisted entry in
  language_agent; on firing → plan_mission(do) or say; never claims the base before firing.
- **constrain** — speech_act directive (negative) / commissive (standing); process material (on routes); aspect
  {class: state, goal_predicate: invariant holds, progress_signal: STATE line, interrupt: release}; arguments {Theme
  room|door required, scope optional (default: until cancelled, said back to the person)}; clarification: unknown room
  name → ask which; execution: store in language_agent + checks in route/plan/cross; a violation attempt is refused
  with the constraint named.

---

## 6. Top 5 gaps to decide on
1. **G1 event-triggered and conditional directives** — sequence-type triggers ("cuando llegues, avisa") already
   compose; real triggers and if/else need M3 `seen()` + M4 `if` (P1).
2. **G2 negative commands as constraints** — no representation at all; safety-relevant (P2).
3. **G3 reminders, standing orders, recurring** — `wait at` holds the base and dies with a restart; needs M1 (P3).
4. **G4 search / watch** — "busca la taza" composes as explore-until-seen once `seen()` exists; "vigila la puerta"
   is M1 with `on: door_open(door_2)` (P1 + P3).
5. **G5 quantified directives** — code expansion of `each` (P4).

---

## 7. What the taxonomy says about reorganising the tools

- **The IR already encodes aspect.** Instant ops (say, report, anchor, stop) are achievements; cross/approach/go_to
  are accomplishments with goal predicates and progress; `explore` is the one activity and its bound is REQUIRED by the
  validator (`budget_s`), which is exactly the "atelic needs a stop condition" rule. Make that rule explicit for every
  new atelic op (search, patrol, watch): the planner must supply a bound or the card asks for one.
- **Chat tools group cleanly by process type**, which is a better system-prompt structure than today's flat list:
  *relational/existential reads* (where_is, find_objects, known_rooms, route_to, inspect, agents, sensors,
  graph_summary, episodes…, where_was), *perceptual* (can_see, look), *counterfactual* (imagine, imagine_view),
  *material immediate* (cross_door, approach_object, go_around_object, move_relative, stop, cancel_passage),
  *semiotic* (show, draw, table, hide_canvas), *informative* (human_claim), *complex/temporal* (plan_mission).
  The four memory reads (episodes, episode, last_episode, outcome_stats) could become one `remember(kind, target,
  since, until, detail)` tool; `where_was` stays separate (it returns a place to go).
- **Speech act decides the confirmation policy**: questions run at once; directives with a body move run at once
  (stop is always available); complex directives get a card; informatives get a claim card (human_claim pattern);
  commissives will need a card AND a visible list of what is standing.
- **Temporal anchoring belongs to the mission layer, never to the chat tools**: every non-immediate anchoring should
  go through plan_mission (or the scheduler of P3), so there is one place that grounds times (ground_clock) and
  events (predicates).
- **Polarity is a missing dimension**, not a missing tool: negation should map to constraints carried by the mission
  (P2), not to new verbs.

---

## 8. Designing for a small local model (Step 6)

The production chat/planner model is gemma4:12b local (the addendum says 8B; the same rules apply harder there).

- **Tool count.** 31 tools today is too many for a 12B model to separate reliably. Consolidation that loses nothing:
  `episodes` + `episode` + `last_episode` + `outcome_stats` → `remember(kind?, target?, since?, until?, id?, stats?)`;
  `agents` + `sensors` + `graph_summary` → `self_status(part: agents|sensors|graph)` (keep `inspect`);
  `table` folds into `show` (name=table); `cancel_passage` folds into `stop` (it is a stop of the passage).
  31 → 24, then + `schedule`, `cancel_scheduled`, `constrain`, `release` = 28, with the new capabilities.
- **Descriptions.** Each carries: one sentence of what it does, one ES and one EN example utterance, a "not for …"
  clause naming its nearest neighbour (many already do: where_is/can_see, show/draw, cross_door/sim_door). Missing
  today: examples in Spanish for most tools; a "not for" on `move_relative` vs `approach_object` exists, on
  `look` vs `show camera` exists, on `find_objects` vs `where_is` does not.
- **Typed time and triggers.** `schedule.when` is a tagged object (`at` | `in_s` | `every` | `on`), never free text;
  `on` uses the M3 predicate names with typed refs; durations carry units as the IR already does (`budget_s=60s`).
- **Clarification policy, enforced in code where possible.** A required role missing → the tool returns `ASK: …`
  (the human_claim / agent_control pattern), the model relays it; an atelic step with no bound → the plan checker
  fills the skill default and the CARD shows it (the person sees and can change it), never a silent unbounded run.
- **Pre-routing.** Not recommended now. The tools eval on gemma4:12b is 128/134 (config.toml comment, the eval in
  `tests/check_llm_tools.py`); the live failures seen in 11-12…11-14 runs were claims without tool calls (fixed in
  code, 24356d88), a wrong facing answer (583d247f), and planner route choice — none is a tool-SELECTION error a
  speech-act/temporal router would fix. Re-evaluate after adding schedule/constrain: if the new eval cases
  (`utterance_testset.yaml`) show the model confusing `schedule` with `plan_mission`, a two-label pre-router
  (immediate vs anchored) is the cheapest fix.
- **Test set.** `utterance_testset.yaml`: 40 utterances, ES and EN, one or more per matrix row, with the expected tool
  call(s); entries needing a proposed tool are marked `needs: P#`, so the file is usable today as a regression set for
  the rest.

