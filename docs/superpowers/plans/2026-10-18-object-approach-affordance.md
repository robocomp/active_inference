# Every object offers `approach` — the controller as a mediator (2026-10-18)

## Goal

The controller only **activates affordances that objects propose**. Missions and chat say *which* affordance; the
object that owns it says *whether it is possible* and *where to stand*. Agreed with the user 2026-10-18:

- **Through affordances:** everything that names a thing in the world — doors (done: approach/open/cross/close),
  and from this plan **every object's `approach`**.
- **Left out, as the robot's own affordances (later, not this plan):** click targets (Target mode), relative motion
  (`move`/`turn`), recorded routes (MissionOnly).

Today the only way to "go to the table" is chat's `approach_object` (de009540): chat computes a standpoint and sends a
`go_to` **pose** mission. That bypasses the table. This plan replaces it with a claim on `aff_table_1_approach`.

## What exists (verified 2026-10-18)

| piece | state |
|---|---|
| `common/object_affordance` (`rc::ObjectAffordance`) | ONE epistemic node per object (`aff_<object>`): "stand here and look, ΔH nats". Used by chair, table, bottle, cabinet, hood (and door). |
| `common/pragmatic_affordance` (`rc::pragmatic::PragmaticAffordance`) | NAMED, preconditioned nodes (`aff_<object>_<kind>`), Schmitt-debounced offer on a probability, "unknown is not false", never removed while executing. **Only door_concept uses it.** |
| `rc::affordance::Contract::reach()` | "navigate to the pose, done on arrival" — no completion attribute needed (`affordance_protocol.h:349`). |
| selector (`affordance_manager.cpp:807`) | pragmatic affordances are **never taken by default**: only when a consumer requires them by name (`set_required_name`). So offering `approach` on every object does not change exploration. |
| `DoorPassage` | a step sequence over one object's pragmatic affordances, restricting the selector to the current step's name; Done → `AffordancesOnly` for a single command (4fbc91af). |
| chat | `approach_object` tool → pose mission (to be replaced); STATE shows door affordances per door. |

## Design

### 1. Producer side — one shared helper, five agents add ~6 lines each

New `common/pragmatic_affordance/object_approach.h`: `rc::pragmatic::ObjectApproach`, wrapping one
`PragmaticAffordance` named `approach` plus the offer computation, so no agent re-derives it.

```cpp
struct ApproachInputs {
    bool known;                  // a fitted pose this cycle AND a robot pose ("unknown is not false": hold)
    Eigen::Vector2f centre;      // object centre, room frame
    float footprint_radius_m;    // half the footprint diagonal (from the fitted size)
    Eigen::Vector2f robot;       // robot, room frame
    float pos_sigma_m;           // object position σ (from its Σ) — widens the standoff band, no threshold
};
class ObjectApproach {
    void init(G, object_node_id, object_name, Policy);
    void update(const ApproachInputs&, const ApproachCfg&);   // offer + pose + value + contract
    void remove(); void on_node_deleted(id);                   // at every retirement site, like the door's siblings
    bool consume_completion(); Outcome outcome();
};
```

- **Precondition** `p`: a located object can always be approached (p = 1, like the door's `approach`), so `known`
  carries the whole decision. Value falls as the robot arrives: `value · (1 − P(within standoff))`, the door's
  ping-pong lesson (`door_concept specificworker.cpp`, approach block).
- **Standpoint:** on the line object → robot, at `footprint_radius + robot_radius + clearance + k·σ`, facing the
  object — the geometry now in `chat_agent/src/approach.h` (`approach_point`), **moved here** so the object owns it.
  σ enters as a widening of the standoff (a far, uncertain object is not approached to within 5 cm of a guess).
- **Contract:** `Contract::reach()` — done on arrival. A perceptual predicate (`obj_reach_prob`, as doors have
  `door_reach_prob`) is optional, Phase 6, and would need a cortex attribute (the user reinstalls).
- **Config** `[Approach]` per agent with shared defaults: `RobotRadiusM 0.35`, `ClearanceM 0.30`, `Value 0.3`,
  `SigmaK 1.0`, debounce from the agent's existing pragmatic/epistemic policy.

Each agent: add `rc::pragmatic::ObjectApproach approach;` to its `*_instance.h`; `init` where the instance gets its
node; `update(...)` in the cycle that already publishes the epistemic affordance (it has the fitted pose, size and
Σ there); `remove()`/`on_node_deleted()` at **every** site its `affordance.remove()` is called — the TIER-B
stranded-node lesson; `CONCEPT_AGENT_LIFECYCLE.md` §OWNERSHIP audit row updated.

### 2. Consumer side — the door passage generalised to an *order*

`DoorPassage` is already "a sequence of one object's pragmatic affordances". Extract the generic part:

- `rc::AffordanceOrder` (controller): `start(object, steps, then_affordances)`; `required()` = `aff_<object>_<step>`;
  advance on Satisfied; the door-specific rules (skip to a later offered step, back to `open` when shut, the
  crossing corridor, room-change ⇒ cross done, close deadline) stay in `DoorPassage`, which becomes the door's
  specialisation. An approach order is ONE step: `[approach]`.
- Grammar: `approach <object>` (chat / panel / `remote_cmd.txt`) → `start(object, {approach}, then_affordances=true)`;
  refused with a reason when the object node or its `aff_<object>_approach` does not exist.
- Missions: a new op `approach` (`to`: Node ref) in `mission_ir.h`'s registry → MissionHost starts the order; the
  leaf succeeds on Satisfied. `go_to` keeps Pose/Anchor only (the robot's own affordance, left out above).
- `controller_mission` progress / chat status lines report the order like a passage (`ChatInbox::status_of`).

### 3. chat_agent

- `approach_object` sends `approach <object>` (not a pose mission); the gate checks the object is in the current room
  **and offers `approach`** (the STATE says which objects do).
- STATE: objects list `approach=offered|executing|…` the way doors list their affordances.
- `approach.h`'s `approach_point` / `approach_mission` are deleted once the producers carry the geometry
  (`pick_object` stays: it is name resolution).

## Phases (each ends green and committed; nothing live is assumed)

| # | what | test | risk |
|---|---|---|---|
| 0 | Agree the open questions below | — | — |
| 1 | `ObjectApproach` in `common/pragmatic_affordance` + offer math + lifecycle | pure offer/standpoint test (σ, footprint, already-there value) + DSR lifecycle test (offer, claim, hold while executing, remove) with mutants | low |
| 2 | **Pilot: table_concept** wired; live: `aff_table_1_approach` appears, survives fitting churn, is removed with the table | log/graph check; no exploration change (pragmatic is never default) | medium — table_concept has uncommitted work from another session: coordinate first |
| 3 | chair, bottle, cabinet, hood | per-agent audit row; each built and run once | medium — chair/bottle also have uncommitted work |
| 4 | `AffordanceOrder` + `approach <object>` grammar + mission op `approach` | order test on a real graph (like `test_door_passage.cpp`); `test_door_passage` stays green (refactor guard); mission IR/text tests | medium — DoorPassage refactor touches a live-critical path |
| 5 | chat: `approach_object` → `approach <object>`; STATE object affordances | eval cases (approach the table / chair 1 / acércate a la mesa / the fridge in another room) | low |
| 6 | (optional) perceptual completion `obj_reach_prob` | needs a cortex attribute → user reinstall | low |
| 7 | delete the pose path (`approach_mission`), docs, `CONCEPT_AGENT_LIFECYCLE.md` audit | — | low |

## Open questions (Phase 0)

1. **Standoff per object type or one rule?** One rule (footprint + body + clearance + σ) proposed; a chair might want
   "in front of the seat", a fridge "facing the door" — later, as a per-agent `face` hint, not now?
2. **Persons** (person_concept): approach a person too? Same helper, but the standoff is social, not geometric.
3. **Which side:** the side the robot is on (proposed), or the object's "front" where it has one (table has none)?
4. **After the approach:** a single command hands back to `AffordancesOnly` like a crossing (proposed) — or holds?
5. **Pilot agent:** table_concept (has the richest size/Σ) — but it has uncommitted work in progress right now.

## Not in this plan

Robot affordances for click targets, relative motion and routes; MissionWithAffordances; persons' social standoff.
