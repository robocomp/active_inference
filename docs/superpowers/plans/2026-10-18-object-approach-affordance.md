# Every object offers `approach` — the controller as a mediator (2026-10-18)

## Goal

The controller only **activates affordances that objects propose**. Missions and chat say *which* affordance; the
object that owns it says *whether it is possible* and *where to stand*. Agreed with the user 2026-10-18:

- **Through affordances:** everything that names a thing in the world — doors (done: approach/open/cross/close),
  and from this plan **every object's `approach`**, optionally **from a named side** ("approach the table from the
  right side").
- **Left out, as the robot's own affordances (later, not this plan):** click targets (Target mode), relative motion
  (`move`/`turn`), recorded routes (MissionOnly).

Today the only way to "go to the table" is chat's `approach_object` (de009540): chat computes a standpoint and sends a
`go_to` **pose** mission. That bypasses the table. This plan replaces it with a claim on `aff_table_1_approach` (or
`aff_table_1_approach_<side>`).

## Decisions (Phase 0, user 2026-10-18)

1. **Standoff:** one rule for every object, defined below (§1.1).
2. **Persons:** yes — same helper, social standoff. There is no person agent in the repo today, so the helper takes
   the override now and the wiring waits for that agent.
3. **Default side:** the object's **front** where it has one; otherwise the side facing the robot.
4. **After an approach:** **hold position** (Command mode with no order: the robot stays, it does not forage).
   Inside a mission the leaf succeeds and the mission goes on.
5. **Pilot:** **chair_concept** (it has a front: the seat side, from its estimated yaw).

## What exists (verified 2026-10-18)

| piece | state |
|---|---|
| `common/object_affordance` (`rc::ObjectAffordance`) | ONE epistemic node per object (`aff_<object>`): "stand here and look, ΔH nats". Used by chair, table, bottle, cabinet, hood (and door). |
| `common/pragmatic_affordance` (`rc::pragmatic::PragmaticAffordance`) | NAMED, preconditioned nodes (`aff_<object>_<kind>`), Schmitt-debounced offer on a probability, "unknown is not false", never removed while executing. **Only door_concept uses it.** |
| `rc::affordance::Contract::reach()` | "navigate to the pose, done on arrival" — no completion attribute needed (`affordance_protocol.h:349`). |
| selector (`affordance_manager.cpp:807`) | pragmatic affordances are **never taken by default**: only when a consumer requires them by name (`set_required_name`). So offering `approach` (and its sides) on every object does not change exploration. |
| controller target repair | `grid_planner.h:250` `nearest_rotatable`: an affordance goal the robot cannot turn in is moved to the nearest spot it can turn in (the terminal facing rotation runs with no obstacle check — config.toml §GoalFacingYawEnabled). |
| controller geometry | no robot radius anywhere, on purpose (`controller_types.h:47`: stacked disc radii once demanded 0.95 m of gap for a robot that passes 0.461 m). Body 0.46 × 0.65 m; turning sweeps its half diagonal, 0.398 m. `ComfortStandoff` 0.6 is MPPI comfort, not geometry. |
| `DoorPassage` | a step sequence over one object's pragmatic affordances, restricting the selector to the current step's name; Done → `AffordancesOnly` for a single command (4fbc91af). |
| chat | `approach_object` tool → pose mission (to be replaced); STATE shows door affordances per door. |
| chair_concept | belief `[cx, cy, yaw]`; yaw can be 4-fold ambiguous (square seat), broken by the backrest and a discrete yaw-mode evidence (`chair_belief.h:80`). **Uncommitted work from another session** in `specificworker.{h,cpp}`, `CMakeLists.txt`, `specificworker_dashboard.cpp`. |

## Design

### 1. Producer side — one shared helper

New `common/pragmatic_affordance/object_approach.h`: `rc::pragmatic::ObjectApproach`, owning the object's approach
nodes and the offer computation, so no agent re-derives it.

```cpp
struct ApproachInputs {
    bool known;                    // a fitted pose this cycle AND a robot pose ("unknown is not false": hold)
    Eigen::Vector2f centre;        // object centre, room frame
    float yaw;                     // object yaw, room frame (the footprint's orientation)
    Eigen::Vector2f half_extent;   // footprint half sizes (a, b) along the object's own x, y
    std::optional<float> front_yaw;// the object's front direction, room frame; nullopt = no front, or not yet known
    Eigen::Matrix2f pos_cov;       // object position Σ (room frame)
    Eigen::Vector2f robot;         // robot, room frame
    float robot_pos_sigma_m;       // robot position σ
    std::optional<std::vector<Eigen::Vector2f>> room_polygon;   // nullopt = unknown → sides hold
};
class ObjectApproach {
    void init(G, object_node_id, object_name, Policy, Sides);   // Sides: Fronted | Frontless | DefaultOnly (persons)
    void update(const ApproachInputs&, const ApproachCfg&);     // offer + pose + value + contract, every node
    void remove(); void on_node_deleted(id);                     // at every retirement site, like the door's siblings
    std::optional<Completion> consume_completion();              // which node, and its Outcome
};
```

#### 1.1 The standoff rule (one rule, every object)

The standpoint for approach direction `d` (unit vector from the object's centre outwards) is

    D(d) = h(d) + R_turn + k · σ(d)

- `h(d) = a·|cos φ| + b·|sin φ|`, φ = angle between `d` and the object's x axis: the footprint box's **support
  function** — the exact distance from the centre to the box edge along `d` (a face when `d` is a normal; the
  half diagonal only at a corner). The chat version used the half diagonal on every side, i.e. too far from a face.
- `R_turn = 0.398 m`: half the robot body diagonal, from the controller's footprint (0.46 × 0.65). The robot ends an
  affordance with a facing rotation in place, so the standpoint must leave room for the body's swept circle.
  Nothing else is added: `ComfortStandoff` is the MPPI's comfort cost, not geometry, and stacking it is exactly the
  0.95 m mistake `controller_types.h:47` records. The controller's `nearest_rotatable` repairs a standpoint a wall
  or another object makes un-turnable.
- `σ(d) = sqrt(dᵀ Σ_obj d + σ_robot²)`: the object's position spread along the approach line plus the robot's own;
  `k = 2` (~95 %), the same figure residual_concept (`PublishExtentSigmaK`) and the controller's σ-padding use.
  Risk, from the controller's 2026-08-09 regression: object σ grows while the object is not looked at. Here the
  standpoint is computed while offered and **frozen while executing** (the robot is then looking at the object), so
  a growing σ cannot push a goal away mid-approach. A test pins this.
- **Persons** (`Sides::DefaultOnly`): `h = 0` and the box term replaced by a social distance, 1.2 m, the outer edge
  of Hall's personal zone — a configured figure with that citation, set by the person agent when it exists.

The robot's yaw at the standpoint faces the object's centre.

#### 1.2 Sides — one node per side the object can be approached from

"Approach the table from the right side" is a choice between affordances **the object proposes**, not a parameter
the consumer imposes. Each side is its own pragmatic node, so the object decides, per side, whether it is possible:

| object | nodes | side directions |
|---|---|---|
| fronted (chair; later cabinet, hood, fridge) | `approach` (= front), `approach_front`, `approach_back`, `approach_left`, `approach_right` | from `front_yaw`; left/right are the **object's own** (a chair's left as one sits in it) |
| frontless (table, bottle) | `approach` (= the face toward the robot), `approach_xp`, `approach_xn`, `approach_yp`, `approach_yn` | the object's box faces, each named by the **room axis** its outward normal is nearest. A table's yaw has a 180° symmetry, so names taken from the object's own axes would swap on a refit; room-axis names do not. Debounced so a table near 45° does not rename its faces back and forth. |
| person | `approach` only | facing the person when person_concept knows where it faces, else from the robot's side |

- **Precondition per side** `p`: the side's standpoint lies inside the room polygon with `R_turn` to spare
  (1 or 0; the room polygon unknown → hold, "unknown is not false"). Obstruction by other objects is the
  controller's (`nearest_rotatable`, footprint test), not re-done here. A fronted object whose front is not yet
  known (a chair's yaw mode not settled) holds `approach_front/back/left/right`, and its plain `approach` falls back
  to the face toward the robot.
- **Value** falls as the robot arrives: `value · (1 − P(within 0.10 m of the standpoint))`, the door's ping-pong
  lesson (door_concept, approach block). 0.10 m is the controller's arrival tolerance — to be read from its config in
  Phase 1, not restated.
- **Contract:** `Contract::reach()` — done on arrival. A perceptual predicate (`obj_reach_prob`, as doors have
  `door_reach_prob`) is optional, Phase 6, and would need a cortex attribute (the user reinstalls).
- **Config** `[Approach]` per agent with shared defaults in the helper: `TurnRadiusM 0.398`, `SigmaK 2.0`,
  `Value 0.3`, `SideDebounce` from the agent's pragmatic policy.

Each agent: an `ObjectApproach approach;` in its `*_instance.h`; `init` where the instance gets its node; `update(...)`
in the cycle that already publishes the epistemic affordance (it has pose, size and Σ there); `remove()` /
`on_node_deleted()` at **every** site its `affordance.remove()` is called — the TIER-B stranded-node lesson;
`CONCEPT_AGENT_LIFECYCLE.md` §OWNERSHIP audit row updated. Five nodes per object: pragmatic nodes are never
selected by default, so they only cost graph size.

### 2. Consumer side — the door passage generalised to an *order*

`DoorPassage` is already "a sequence of one object's pragmatic affordances". Extract the generic part:

- `rc::AffordanceOrder` (controller): `start(object, steps, after)`, where `after` ∈ {`Affordances`, `Hold`, `Return`
  (to the mission)}; `required()` = `aff_<object>_<step>`; advance on Satisfied; the door-specific rules (skip to a
  later offered step, back to `open` when shut, the crossing corridor, room-change ⇒ cross done, close deadline)
  stay in `DoorPassage`, which becomes the door's specialisation. An approach order is ONE step:
  `[approach]` or `[approach_<side>]`.
- Grammar: `approach <object> [from <side>]` (chat / panel / `remote_cmd.txt`) → `start(object, {step}, Hold)`;
  refused with a reason when the object node or that affordance does not exist, or exists but is not offered
  ("table_1 cannot be approached from xn: that side is against the wall").
- On `Hold`, when the order ends the controller goes to Command mode with no order — it stays there, it does not
  forage — until the next command or mission.
- Missions: a new op `approach` (`to`: Node ref, optional `side`) in `mission_ir.h`'s registry → MissionHost starts
  the order with `Return`; the leaf succeeds on Satisfied. `go_to` keeps Pose/Anchor only.
- `controller_mission` progress / chat status lines report the order like a passage (`ChatInbox::status_of`).

### 3. chat_agent

- `approach_object(object, side?)`. `side` is the person's words ("right", "front", "the side by the window"); chat
  resolves it to one of the object's **offered** side nodes and sends `approach <object> from <side>`. The gate
  refuses a side the object does not offer, with the object's reason.
- **Which "right":** for a fronted object, the object's own right (a chair's right); for a frontless one, right
  **as the robot sees it from where it stands now**. The STATE gives each side node a viewer-relative label
  computed at STATE time, e.g. `table_1 approach sides: yn (near side, 1.1 m ahead), xp (its right as seen from
  here), xn (its left as seen from here), yp (far side) — xn not offered: against the wall`. The model picks by
  label, never by computing geometry. The reply says which side was taken ("going to the table's right side, as
  seen from here").
- STATE: objects list `approach=offered|executing|…` and their sides the way doors list their affordances;
  `affordance_words` learns `approach_<side>` ("approach table_1 from xp").
- `approach.h`'s `approach_point` / `approach_mission` are deleted once the producers carry the geometry
  (`pick_object` stays: it is name resolution).

## Phases (each ends green and committed; nothing live is assumed)

| # | what | test | risk |
|---|---|---|---|
| 1 | `ObjectApproach` in `common/pragmatic_affordance`: standoff rule, sides (fronted / frontless / default-only), preconditions, lifecycle | pure tests: support function on faces and corners, σ widening, frozen-while-executing, room-axis face naming + debounce at 45°, side outside the room polygon not offered, unknown polygon holds, fronted with unknown front falls back; DSR lifecycle test (offer, claim, hold while executing, remove all nodes); mutants | low |
| 2 | **Pilot: chair_concept** wired; front from its yaw once the yaw mode is settled | live: `aff_chair_1_approach*` appear, survive refits, are removed with the chair; no exploration change | medium — chair_concept has uncommitted work from another session: coordinate first |
| 3 | `AffordanceOrder` + `approach <object> [from <side>]` grammar + Hold + mission op `approach` | order test on a real graph (like `test_door_passage.cpp`); `test_door_passage` stays green (refactor guard); mission IR/text tests | medium — DoorPassage refactor touches a live-critical path |
| 4 | chat: `approach_object(object, side?)` → `approach <object> from <side>`; STATE sides with viewer labels | unit tests for the labels; eval cases: approach the chair / the table from the right / acércate a la mesa por la izquierda / a side that is not offered / the fridge in another room | low |
| 5 | table, bottle, cabinet, hood (frontless or fronted as each one knows) | per-agent audit row; each built and run once | medium — table/bottle have uncommitted work too |
| 6 | (optional) perceptual completion `obj_reach_prob` | needs a cortex attribute → user reinstall | low |
| 7 | delete the pose path (`approach_mission`), docs, `CONCEPT_AGENT_LIFECYCLE.md` audit; persons when a person agent exists | — | low |

The controller (Phase 3) comes before chat (Phase 4) so that chat never sends a command the controller cannot parse.

## Open (small, decided in the phase that needs them)

- Phase 2: what chair_belief exposes as "yaw mode settled" (a mode posterior? an evidence margin?) — read it, do not
  invent a threshold.
- Phase 1: the controller's arrival tolerance, to read rather than restate the 0.10 m above.

## Not in this plan

Robot affordances for click targets, relative motion and routes; MissionWithAffordances; the person agent itself.
