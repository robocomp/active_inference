# Perception cards: a queryable sensor model per concept agent — design

Date: 2026-10-31. Status: **APPROVED by the user (2026-10-31)**, with the LiDAR term (§4b) added on their request.
Nothing is built yet. It depends on the sensor-intrinsics work on branch `feat/sensor-intrinsics`
(`rc::media::sensor_model_from_graph`).

## 1. Intent

To answer "can you see X?", "where should you stand to see X?", and later "would you see X from there?" (the
`imagine` plan), the robot needs its **sensor model A**, p(o | s): what its sensors and perception agents would
register *if* a given object were at a given place, seen from a given pose.

The other ingredients already exist (the user, 2026-10-31):
- **C, the concept inventory,** is the list of `*_concept` agents. Each agent owns one concept.
- **B, action consequences,** is in episodic memory (EM), and can be extended.
- **A is the gap.**

Each concept agent already carries a version of A, because the lifecycle contract makes it weigh absence by
P(detect) and line of sight. But that model is private, partly duplicated, and inconsistent between agents. This
design **exposes A** without inventing a second model:

- each agent publishes a **perception card** describing the detection model it actually runs;
- one **shared evaluator** in `common/` computes P(detect) for any hypothetical case from a card plus world facts;
- chat_agent uses it through a `can_see` tool, and later through `imagine`.

The scope is **known, single concepts first** (the user's choice). Collectives (cabinet runs, shelving) are in
`2026-10-31-collective-objects-design.md` and plug in later with their own card fields.

## 2. The concept-kind hierarchy

The kinds form a **complexity hierarchy**, in which each level includes the ones below (the user's view; it
supersedes the "independent axes" wording of `CONCEPT_AGENT_LIFECYCLE.md` §6.1). Each level contributes **one
multiplicative term** to P(detect):

| level | adds | P(detect) term | examples |
|---|---|---|---|
| 1 mass | stuff: a region that is seen or not | **coverage**: the fraction of the region in the field of view and not occluded (walls included) | floor, residual occupancy |
| 2 countable | an individual a detector must single out | × **detector envelope**: the projected image fill must be neither too small nor overflowing | table, chair, bottle, fridge, hood |
| 3 articulated | an internal configuration (joints) | × **configuration term**: appearance given the joint (door leaf obliquity, phi) | door; cabinet once it has doors |
| 4 animate | self-motion | the result **averaged over the position** it may have reached, with σ growing with time since it was last seen | person |

- The level-1 coverage term is the visible fraction already computed by `rc::nbv::visible_fraction`. The code is
  already layered this way; the upper levels have so far been written ad hoc in each agent.
- A collective is not a level but a grouping over members (see the collectives design).

## 3. The perception card

Each concept agent writes **one card**, a JSON string attribute on its **own agent node**: the one the presence
protocol uses, which the agent owns and which always exists.

- **When it is written:** at startup, and again if its configuration changes.
- **Where it comes from:** the configuration the agent **actually runs with**, the same values its existence
  channel reads, so it cannot drift from the agent's behaviour.

Card contents:
- `concept`: the concept name and node type, plus the `object_subtype` values it produces.
- `level`: `mass | countable | articulated | animate`.
- `sensors`: which sensors and channels feed the existence decision (for example ZED mask, Helios, Bpearl,
  ricoh bearings), each marked **identity**, **occupancy** or both. v1 models the ZED and both LiDARs (§4b); the
  ricoh is recorded but not modelled (bearing-only).
- `geometry_prior`: the typical size and z span, from the agent's `<concept>.concept.toml`, used when the
  hypothetical object has no measured size.
- **Level 1:** wall-occlusion settings (own-wall skip distance, wall thickness) and the object-occlusion model in
  use.
- **Level 2:** the detector envelope (`min_fill`, `max_fill`, `soft`), plus any extra countable term the agent
  really uses, as a named, parameterised term from a closed list. Example: cabinet's
  `range_power(ref_m, power)`.
- **Level 3:** the configuration term's name and parameters, for example door's
  `leaf_obliquity × resolvability`.
- **Level 4:** the position-σ growth law.
- `blind_spots`: known failure modes as short text, for the language model to quote ("glass is invisible to
  LiDAR", "below the wall band").
- `calibration`: one of `prior` (uncalibrated envelope), `fitted` (from `detect_probe` data via `fit_envelope`)
  or `blind_spot_map` (with a learned correction map present), plus a date.

**Closed term list.** The evaluator implements a fixed, named set of terms per level. A card may use only terms in
that list. An agent whose real model needs a new term gets it added to the shared evaluator, with a test, before
its card claims it. This keeps the card truthful and keeps one implementation per term.

## 4. The shared evaluator

A new pure module in `common/perception_card/`, depending on Eigen and std only and with no DSR include. It
composes the existing pieces:
- `rc::nbv` (`viewpoint_score.h`): `predicted_fill_axes`, `predicted_fill_sigma`, `visible_fraction`,
  `wall_obstacles`, `expected_p_detect`, `standoff_band`;
- `rc::detect` (`detectability.h`): `DetectorEnvelope`, `p_detect`;
- `rc::occlusion` (`occlusion.h`): `walls_block`, `cone_occlusion`.

Its interface:
- **`evaluate(card, world, hypothesis, viewpoint) → Result`**
  - `world`: the room polygon, door gaps, obstacles, sensor (intrinsics and mount);
  - `hypothesis`: a concept, a pose, an optional size, an optional joint state;
  - `viewpoint`: the robot pose; the yaw may be left free, meaning "best heading".
  - `Result` holds P(detect), the **per-level terms** (coverage, envelope, configuration, motion), and the
    **limiting factor**: the term with the lowest value, plus a reason (out of view / behind a wall / occluded
    by `<obstacle>` / too small at this range / too close / leaf edge-on).
- **`where_to_stand(card, world, hypothesis) → candidates`**: the viewpoints that maximise P(detect), from
  `standoff_band` and a sampled ring scored with `evaluate`.

Two more properties:
- **The occluder is named** by leave-one-out on the obstacles. This is a model term, not a cutoff.
- **The same module feeds `imagine`**: the "see" question in the imagine plan calls `evaluate` instead of its own
  `[Imagine.Detect.*]` table once cards exist.

## 4b. The LiDAR term (added 2026-10-31)

The LiDAR intrinsics are now in the graph: each sensor node's media-descriptor `SensorModel` carries rings, ring
elevations, azimuth step, min/max range and range noise, read through `rc::media::sensor_model_from_graph`. The
mount comes from the RT edge. So a card can model its LiDAR channels as well as the ZED.

**A LiDAR answers a different question from a camera.**
- A LiDAR tells the robot that *something occupies* a place.
- A camera tells it *what* is there.
- For a countable concept, LiDAR rarely gives identity, because a table and a box look alike in returns. So the
  evaluator reports two things:
  - **P(notice):** would something register there at all? This comes from LiDAR, ZED or both.
  - **P(recognise):** would it be detected *as this concept*? This comes from the channels the card marks as
    identity channels, typically the ZED mask.
- The card states, per channel, whether it provides **identity**, **occupancy**, or both, matching what the
  agent's existence channel really uses. Example: cabinet's masks give identity and its LiDAR gives the metric
  fit and absence.

**The returns model, per LiDAR, with no threshold:**
- **Expected returns:**
  - λ = Σ over the rings whose elevation, from the mount through the RT chain, crosses the object's z span at
    range r, of the number of azimuth columns its angular width covers, w / (r · Δaz);
  - × the visible fraction along those rays, from the same wall-and-obstacle visibility the camera term uses;
  - × a soft range factor: zero inside `range_min`, falling smoothly past the `range_max` 10%-reflectivity
    figure.
- **P_lidar = 1 − exp(−λ · q),** the probability that at least one return lands on it, treated as a Poisson
  process. q is the per-beam return probability: a per-subtype card parameter, default 1, calibratable from
  the probe logs.
  - λ is the honest quantity.
  - The agent's own minimum-evidence rule, if it has one, enters as a named card term, not as a hidden count
    cutoff.
- **Ring gaps:** they are why low or thin objects vanish. An object thinner than the gap between adjacent rings
  at range r can fall between them. That falls out of λ automatically; no rule is needed.
- **Limiting factor names:** "between my LiDAR rings at this range", "below/above my LiDAR's fan", "inside my
  LiDAR's dead zone".

**Combining channels.**
- **P(notice)** = 1 − Π_c (1 − P_c) over the channels that give occupancy, assuming the channels are
  conditionally independent.
- **P(recognise)** uses only the identity channels.
- The answer to `can_see` reports both when they differ: "my LiDAR would notice something there, but my camera
  can't tell from here that it's a table".

**Sensors:**
- The **Helios** is the main channel for walls and furniture.
- The **Bpearl** dome covers the near field and low objects.
- The **Ricoh** stays out of v1. It is bearing-only (attention, association), not an existence channel, which
  the card records.

## 5. The truth test

A card is only useful if the evaluator **reproduces the agent's own P(detect)**. For each agent:

- **Offline:** a unit test feeds the agent's existence-channel P(detect) function and the evaluator the same
  inputs, from a recorded or synthetic case, and checks that they agree to numerical tolerance.
- **Live:** `common/detect_probe` already logs (viewpoint, predicted P, outcome). A report compares
  evaluator(card) against the agent's logged prediction on the same rows. Any systematic difference is a bug in
  the card or the evaluator.
- **Calibration** against reality (predicted P versus the observed detection rate) uses the same logs and
  `fit_envelope`. That is how a card moves from `prior` to `fitted`.

## 6. Migration of the agents

Agents at the same level must use **the same formula** with their own parameters.

| agent | level | today | change |
|---|---|---|---|
| table, refrigerator, hood | countable | shared envelope + `walls_block` | card only |
| bottle | countable | full `rc::nbv` path | card only |
| chair | countable | its own `zed_detectability` (centre × range ramp), **no wall LoS** | move onto the shared countable formula; this also fixes its lifecycle-audit gap |
| cabinet | countable today (collective later) | `absence_range_conf` × fov × central | its range-power becomes a named countable term; collective fields come later |
| door | articulated | envelope × obliquity × resolvability × fov × central × kappa | its extra factors become the articulated term |
| residual | mass | no P(detect) | first card: coverage only |
| human | animate | no existence channel | card deferred until it has one |

- Each migration must leave the agent's existence behaviour **unchanged**, except for chair, where adding wall
  line of sight is the intended fix. The truth test is the gate for that.
- **Ownership:** each agent writes only its own card on its own node, which satisfies the lifecycle ownership
  rule.

## 7. chat_agent: the `can_see` tool

- **`can_see(concept, where?, from?)`.**
  - `where`: a named object instance or a place. If omitted, the instance the graph already holds.
  - `from`: a named place, or the robot's current pose.
  - It returns the probability, the limiting factor and the card's calibration state, phrased for the person:
    "Probably not from here (about 30%): the chair hides most of it. From the doorway it would be about 85%."
- **The reverse question** ("where should you stand to see the fridge?") uses `where_to_stand`.
- **Unknown concepts:** a concept with no card (no agent) gets "I have no detector for that". Later, with the
  vision model, the answer becomes "I could only see it with my camera, and I wouldn't track it" (method 4 of the
  brainstorm, out of scope here).
- **Registration:** read-only, so it goes in `allowed_in_wake`. Eval cases are added to `check_llm_tools.py`.

### 7b. How one `can_see` question is handled (end to end)

Added on 2026-10-31. This describes the flow once the final-review fixes are in, including rulings R10–R12. The
numbers are illustrative. The flow has not yet run against the live robot.

**0. The person asks.** For example: "Can you see the fridge from the kitchen door?"

**1. gemma picks the tool.** In the normal chat turn, gemma4:26b on the Orin calls
`can_see(concept="fridge", from="door_2")`.
- `stand=true` turns it into the reverse question: "where should you stand to see it?".
- `where` names a specific instance.
- The tool is read-only, so it is also allowed in robot-event turns.

**2. chat_agent finds the card.** This runs on the main thread.
- It reads every `*_concept/etc/perception_card.json` under the active_inference root. Later it reads the
  `perception_card` graph attribute, once cortex has it.
- It picks the card for the concept. For the fridge that is: countable; ZED identity, Helios occupancy;
  envelope (axes=1) × central; own-wall skip; calibration state.
- If there is no card for the concept: "I have no detector for that".
- If there are no cards at all: "I couldn't find any perception cards (looked in …)".

**3. It resolves the target.**
- The concept is matched to its instance in the graph (`refrigerator_1`): pose, size, the real z span from the
  room→object transform, and `object_subtype`.
- If several instances match, it asks which one.

**4. It resolves the viewpoint.**
- **A named door:** the door centre, stepped one robot radius into the current room. The heading faces the target,
  because the robot would arrive facing it.
- **A named object:** a stand-off point beside it, again facing the target.
- **"Here":** the robot's actual pose and its real heading, which is body yaw + π/2 because Shadow's forward is
  body +y. Best heading is evaluated as well (R11).
- **Anything else:** "which place do you mean?".

**5. It takes a world snapshot.** This is read-only and comes from the live graph.
- The room polygon, as walls with the card's wall thickness.
- The other objects, as named obstacles at their real heights. The target is excluded.
- The ZED's intrinsics and mount height.
- The Helios and Bpearl, each as a `SensorModel` from its node's media descriptor plus its body→sensor mount. A
  missing model or transform leaves that channel out (NotModelled); it is never P = 0.
- Doors, as gaps in the walls (R10):
  - a measured φ gives its gap;
  - otherwise a known P(open) gives a two-point mixture of open and shut;
  - with neither, the gap is averaged over a uniform φ prior.
  - The mixture covers at most 2 doors that lie on the sight line. It is never an average slit.

**6. The evaluator runs per channel.** This is `common/perception_card/evaluator.h`.
- **ZED (identity).**
  - Visible fraction past walls and obstacles. A wall within the card's own-wall skip of the target does not
    occlude it.
  - Predicted image fill → detector envelope × central × instance-prior terms × blind-spot map → P_zed.
  - The limit is found by leave-one-out (the term whose removal raises P the most), and the occluder is named the
    same way.
- **Helios / Bpearl (occupancy).**
  - The rings whose elevation, taken through the mount, crosses the target's z span.
  - Expected returns: λ0 = Σ columns × visibility.
  - P = range_factor · (1 − e^(−λ0·q)). The range factor is common-mode across returns (R5).
  - The limit is one of: between rings, below/above the fan, dead zone, range, occluded.
- **Combined.**
  - P(recognise) is the noisy-OR over the identity channels.
  - P(notice) = 1 − (1 − P_occ)(1 − P_rec), so it is never below P(recognise).

**7. Where to stand (only if `stand=true`).**
- `where_to_stand` scores 36 bearings × 3 stand-offs inside the detector's stand-off band.
- It drops spots that are unreachable or inside obstacles.
- Uncertain doors are taken at their most probable state (R12).
- It describes the best spot relative to the nearest landmark. Measured cost is about 10 ms.

**8. The answer is worded.** This is a pure, tested function in `chat_agent/src/can_see.h`. The numbers are always
shown. The answer names the channel, the limit and the occluder, and the door assumptions made. It adds the
calibration caveat when the card is only a prior, and it never claims the ricoh. For example:
> "Probably (P 0.74) from the kitchen door: my camera would see it, though the chair hides part of it. My detector
> for fridges isn't calibrated yet."

When notice and recognise differ at the printed decimals:
> "I'd notice something there (P 0.95) but couldn't tell it's the fridge (P 0.30): too far for my camera."

"From here", when turning would help (R11):
> "Not from where I'm facing (P 0.05); if I turn to face it, P 0.81."

**9. gemma replies to the person,** working from that text.

Nothing in this flow writes to the graph.

## 8. Goals, non-goals, anti-goals

**Goals:**
- A card on every concept agent at levels 1–3.
- One evaluator, with the truth test passing for each card.
- `can_see` and `where_to_stand` answering in chat.
- chair moved onto the shared formula.

**Non-goals:**
- A ricoh sensor model (bearing-only; recorded on the card, not modelled).
- Collectives (separate design).
- Animate cards before human_concept has an existence channel.
- Objects without an agent.
- Physics.

**Anti-goals:**
- **A card that describes a model the agent does not run.** This is the reason for the closed term list and the
  truth test.
- **Two implementations of one term.**
- **A threshold-based "visible: yes/no".** Answers are probabilities with their limiting factor.
- **Writing to another agent's node.**

## 9. Constraints

- All CLAUDE.md rules:
  - type-attributed attributes;
  - a new `perception_card` attribute needs a cortex `REGISTER_TYPE`, which the user reinstalls;
  - queued signals;
  - `from_chars`;
  - `-j8`;
  - C++23 with and/or/not.
- `CONCEPT_AGENT_LIFECYCLE.md`: ownership; P(detect) with wall line of sight; no pd floor.
- The evaluator is header-only Eigen and std, so chat_agent and the tests can link it without DSR.

## 9b. Builder-check rulings (all my calls; please check)

1. **Instance terms versus hypothesis terms.** A card holds the terms that can be evaluated for a *hypothetical*
   object: geometry, envelope, configuration, motion. The agents also multiply in per-instance terms that exist
   only for an object already tracked, such as table's `p_clear` from its YOLO score history.
   - These are named in the card as `instance_terms` and evaluated only when `can_see` is asked about an existing
     instance, using that instance's current values read from the graph.
   - For a hypothesis they are marginalised at their prior.
   - A learned `blind_spot_map` is referenced by path in the card and applied by the evaluator through the
     existing `common/detectability/blind_spot_map.h` loader.
   - So "card only" for table means card plus its listed instance terms, with no change to the agent.
2. **The truth test has two layers.** A card *predicts* before observing, while the agent computes *given* an
   observation, so the two cannot agree exactly on a live frame.
   - **(a) Term identity, exact.** Each shared term function, fed the agent's own intermediate values (roi_fill,
     in_fov_frac, central_frac, …), must return the agent's factor to floating-point tolerance. This guarantees
     one implementation per term.
   - **(b) Prediction, statistical.** The geometric prediction of those intermediates (`predicted_fill_axes`,
     `visible_fraction`) is compared with the measured ones on `detect_probe` logs. The checks are bias and
     coverage of the predicted σ, not exact equality.
   - A card passes when (a) is exact and (b) shows no systematic bias.
3. **Fallback before the cortex reinstall.** Each agent also writes its card to `etc/perception_card.json` in its
   own folder. chat_agent reads the graph attribute if present, else the file from the agent folders, which are
   also the concept inventory. Work is never blocked on the reinstall. The card is small (a few hundred bytes)
   and changes rarely.
4. **World adapter.**
   - chat_agent builds the evaluator's `world` from the graph on the main thread: room polygon (a proto-room
     polygon is fine), door gaps, objects as obstacles, ZED intrinsics and mount height.
   - It is the same snapshot the `imagine` plan builds (`ImaginedWorld`), so there is one adapter.
   - A target in another live room goes through the RT chain, with that room's walls added. Otherwise the
     answer is "no data".
5. **Sensors in the answer (revised once the LiDAR term was added).** Answers name the channel: "my camera
   would…", "my LiDAR would notice…". They report P(notice) and P(recognise) separately when the two differ.
   Only the ricoh is unmodelled. If a card lists it, the answer never claims more than the modelled channels
   support.
6. **What the closed term list contains.** It holds every factor the agents actually use:
   - envelope `p_detect`, `in_fov`, `central`;
   - `range_conf(ref, power)` (cabinet);
   - the range ramp (chair, until it migrates);
   - `leaf_obliquity`, `resolvability`, `kappa` (door);
   - coverage.

   A level defines its *required* terms. An agent may add named optional terms from the list. "Same formula per
   level" means the same required terms and the same implementations, not identical products.

   Chair's migration replaces its centre-and-range ramps with the envelope plus `in_fov`, `central` and wall
   line of sight. That is an intended behaviour change, validated by layer (b) of the truth test and by a live
   run.
7. **Obstacle set.**
   - The walls of the relevant rooms, plus every graph object except the target.
   - Hypothetical objects added by an `imagine` intervention are added as obstacles.
   - Leave-one-out runs over this set. `graph_obstacles.h` is not used, because it needs DSR; the adapter builds
     plain `rc::nbv::Obstacle`s.
8. **Joint state (level 3).** An optional input of the hypothesis, for example "if the door were half open".
   Otherwise the graph's `door_phi_rad`. If that is unmeasured, the evaluator marginalises over the door's φ
   prior.
9. **Cost.**
   - Best heading means the yaw pointing at the target, the best case, because the camera is fixed to the body.
   - `where_to_stand` samples a ring of about 36 bearings × 3 standoff distances inside `standoff_band`, about
     100 `evaluate` calls of about 48 visibility samples each.
   - The budget is under 200 ms per tool call, measured in the test.
10. **Calibration refresh.** An agent republishes its card on any event that changes it: loading a fitted
    envelope, saving a blind-spot map, or a config change. Calibration in the card is never stale by design.
11. **One P(detect) source at a time in chat_agent.** The `imagine` plan's `[Imagine.Detect.*]` table stays
    until the evaluator and the first cards land. The same commit then switches `imagine`'s "see" to `evaluate`,
    and both suites' eval cases are re-run.
12. **Lookup key.**
    - A card is found by concept name, the agent's concept.
    - Per-subtype parameters (e.g. table `round` vs `rectangular`) are a map inside the card, keyed by
      `object_subtype`, with a default entry.
    - Cards attach to the agent's own agent node, the one carrying `agent_id`.

## 10. Decided vs my calls vs left to the builder

**Decided with the user:**
- Known single concepts first.
- A card per agent, plus a shared evaluator, rather than live queries to each agent.
- The kind hierarchy, with one term per level.
- Chair and cabinet are welcome to move onto shared terms ("different formulae" allowed by level, not by agent).

**My calls:**
- The card lives on the agent's own node as JSON.
- v1 models the ZED plus both LiDARs. The ricoh is recorded but not modelled.
- P(notice) and P(recognise) are reported separately; channels are combined by noisy-OR for occupancy.
- The LiDAR returns model is Poisson, with the per-beam return probability q as a card parameter.
- The closed term list.
- Leave-one-out to name occluders.
- Cabinet stays countable until the collectives work.

**Left to the builder:**
- The exact JSON schema.
- How viewpoints are sampled in `where_to_stand`.
- The tolerance in the truth test.
- The order in which agents are migrated (suggested: table first, as the reference; then fridge, hood, bottle;
  then chair; then door; then residual).
