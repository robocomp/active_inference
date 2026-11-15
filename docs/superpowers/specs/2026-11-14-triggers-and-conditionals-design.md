# Triggers and conditionals in missions — design (2026-11-14)

Owner's choice from the command taxonomy (`language_agent/docs/command_taxonomy.md`, gap G1, proposal P1): build
**group 1** (a watch on something the robot is doing: "explore, and stop as soon as you see a bottle"; "go to the
kitchen and tell me the colour of any chair you pass") and **group 4** (a decision at one point of a plan: "go to door
1; if it is shut open it, otherwise go through"). Groups 2 (stand still and watch — already `wait until`), 3 (standing
orders outside a mission — the scheduler) and 5 (on-failure alternatives) are not in this design.

## Why
People chain actions with conditions and triggers far more than with plain sequences. Today the mission language
has only `wait until=<pred>` (the robot holds still) on three predicates, no branch, and no way to react to something
seen while moving. The taxonomy found "as soon as / when you see / if" in every row of the directive matrix.

## What the owner decided
- The description below (playback, 2026-11-14) is correct.
- A watch can **stop** the activity or **react and carry on**; both are needed.
- A watch like "any chair you pass" fires **for every new chair**, not only the first.
- "You see X" means a sighting during this mission, not something already on the map when it started.

## Goals
1. A **watch** attached to a mission: a condition, a reaction body, and an ending — `stop` (end the activity, run the
   body, then the mission ends, or continues with the steps after the watched part if the request says so) or
   `continue` (run the body; the activity goes on).
2. **`if <pred> then … else …`**: evaluated once when the mission reaches it, against the beliefs at that moment.
3. **One predicate vocabulary** shared by `wait until`, `repeat until`, `explore until`, watches and `if`; it gains
   **`seen(<class> | <node>)`**.
4. The chat model can write these from natural requests (ES/EN), the card shows them readably, and the chat reports
   each firing ("chair_51: magenta, p 0.73").

## Non-goals
- Standing orders that outlive a mission, reminders, recurring missions (the scheduler, later).
- On-failure alternatives ("if you can't, go through door 1") — later; `on_fail` stays as is.
- Triggers on the person ("cuando llegue Pablo"), on time inside a mission beyond `wait at`, on sounds.
- New object classes for perception (a cup is not detectable yet; "cup" maps to nothing, the chat says so).

## Anti-goals (a failure even if it works)
- A watch that fires on objects already known when the mission started ("stop as soon as you see a chair" in a room
  with a known chair must not end at once).
- A watch that fires twice for the same object, or never terminates (every watch lives at most as long as its
  mission; the mission's steps keep their budgets).
- A reaction that drives the robot while another step also drives it (one owner of the base at a time).
- Prompt-only rules: the plan checker enforces structure (as for `approach` of remembered objects).

## The language (text form the planner writes; JSON mirrors it as for other ops)

```
mission "go to the kitchen, tell me the colour of any chair on the way"
  watch seen(chair) each=c continue
    report "the colour of $c"
  cross door_22 expect=room_3
```
```
mission "explore and stop when you see a bottle"
  watch seen(bottle) each=b stop
    report "where $b is"
  explore room_1 budget_s=300s
```
```
mission "go through door 1, opening it if it is shut"
  if door_open(door_1)<0.5
    open door_1
  cross door_1 expect=room_2 retry=2
```
```
mission "if door 22 is open go to room 3, otherwise to room 2"
  if door_open(door_22)>=0.5
    cross door_22 expect=room_3
  else
    cross door_1 expect=room_2
```

- **`watch <pred> [each=NAME] (stop|continue)` + body.** It applies to the steps that FOLLOW it in the same
  sequence (its scope), and ends when they end. `each=NAME` binds the object that made it fire, usable as `$NAME` in
  the body (`report`, `approach $c`, `say`). One watch may fire many times (every new matching object, once each).
- **`if <pred>` + body, optional `else` + body.** Probabilistic predicates carry their comparison as the IR already
  requires (`door_open(door_1)>=0.5`, the decision boundary the person implied; the planner writes 0.5 when they
  said nothing).
- **`seen(chair)`** (a class) is true for an object of that class that APPEARS in the working memory after the watch
  started — a new node, or a remembered one resurrected by its concept agent on re-sighting. **`seen(chair_34)`** (a
  node) is true when that node appears, or reappears, after the watch started. "Appears" is the builder's mechanism
  to choose (see below); the meaning is the owner's: seen during this mission, not known before it.

## Behaviour
- **continue + a body that does not move** (say, report): the body runs at once; the watched step is not touched.
- **continue + a body that moves** (approach $c, move): the watched step is paused (the interpreter's existing pause:
  the running leaf is cancelled and restarted from its beginning with its whole budget afterwards, enclosing budgets
  resume), the body runs, then the step resumes. Firings that arrive while a body runs are queued in arrival order,
  each object once.
- **stop**: the watched steps are cancelled (state `cancelled`, outcome `watch:<pred>`), the body runs, and the
  mission continues with whatever follows the watch's scope (normally nothing, so it completes). Cancelling by a
  watch is a success of the mission, not a failure.
- **if**: evaluated once on entry; a predicate that cannot be evaluated now (nullopt from the host) fails the step
  with outcome `cannot_evaluate:<pred>` (on_fail applies), never silently picks a branch.
- **Card and progress**: the card shows the watch and if lines as written; while running, each firing adds a progress
  line ("watch seen(chair): chair_51 → report") and the chat is woken for each `report` in a body, with the bound
  object named, as reports are today.

## Constraints
- The IR lives in `common/mission/` (ours); the controller runs the interpreter and implements the Host
  (`controller/src/specificworker.h` MissionHost, in the rt-build tree, the room session's area). Anything the
  controller must add is a small, separately reviewed change; the interpreter must keep compiling against the old
  controller (new Host methods have defaults).
- The interpreter's rule "at most one leaf started and not finished" stays true (the pause-then-resume above keeps it).
- Unknown op / predicate rejects the whole mission (compatibility rule §7.1), so an old controller refuses a mission
  with a watch instead of running it half.
- CLAUDE.md: no thresholds beyond the person's stated boundaries; from_chars; and/or/not; -j8; commit only on the
  owner's say-so.

## Who builds the controller part (owner, 2026-11-14)
The language_agent session writes the controller's Host change itself, in the rt-build tree, and the room session
reviews it before it is merged. The plan includes it and ends with one live run.

## Settled after the builder check (designer's calls, owner may veto)
- **What a report in a body says.** `report "the colour of $c"` works as reports do today: the chat is woken with the
  topic and the room's objects (each with its measured colour), and the CHAT MODEL composes the sentence. The wake
  names the bound object (`$c` = chair_51) so the model talks about that one. `$NAME` is substituted in Ref fields
  (as `bind` vars are) and in the text of `say` / `report`.
- **Shape in the IR.** `watch` is a composite whose `steps` are the REACTION; its scope is the rest of the enclosing
  sequence after it (a marker with a body, not a second child list). `if` is a composite whose `steps` hold two
  `seq` children, `then` and (optional) `else`; the text form writes `else` as a line at the `if`'s indent.
- **Scope, nesting, repeat.** Scope = the remaining siblings of the enclosing sequence. Several watches may be active
  (a stack, innermost first); a watch inside a `repeat` body is re-armed (new snapshot) on each iteration. A `stop`
  watch cancels its scope only; whatever encloses that scope carries on normally (a repeat goes to its next
  iteration; at the top level the mission completes). The planner skill tells the model to put "stop as soon as"
  watches at the top level.
- **Class vocabulary.** `seen(<class>)` accepts the object subtypes the concept agents publish (the same list
  `find_objects` takes: chair, table, bottle, cabinet, refrigerator, hood, nightstand, bed, toilet, person); anything
  else is rejected by the validator, and the chat says the robot cannot recognise it.
- **Firings on the wire.** The progress snapshot gains an optional per-watch record (fire count and the last bound
  object); an older reader ignores it, an older controller never writes it. The chat wakes on a new firing.
- **`if` on an unknown belief.** `door_open(door)` has a belief as soon as the door node exists; nullopt only when the
  node is missing, which fails the step with `cannot_evaluate:<pred>` (on_fail applies). The planner may put an
  `approach` before an `if` when the person says "look and then decide".

## Left to the builder
- How "appears after the watch started" is detected (a snapshot of matching node ids at watch start and set
  difference each tick is the obvious one; a node that disappears and comes back counts as new only for `seen(<node>)`
  and for a class only if its uid was not already fired on).
- The Host extension for set-valued predicates (e.g. `matches(const Pred&) -> vector<node name>` with a default that
  returns nothing) and its controller implementation.
- How firings are carried in the progress snapshot so the chat sees each one (the current report detection compares
  step states between snapshots, which cannot see a body that runs twice).
- Planner skill text and examples (`etc/skills/missions.md`), `check_planned` rules (a watch needs a scope; `$NAME`
  only inside its body; an `if` predicate must be one of the vocabulary), and new eval cases from
  `docs/utterance_testset.yaml` (d-cond-1, d-cond-2, d-perc-1, plus "any chair on the way").
- Tests: interpreter unit tests with a fake Host for every behaviour above; text/JSON round trip; one live run.
