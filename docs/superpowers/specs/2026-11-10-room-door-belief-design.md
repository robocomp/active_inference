# A room has doors: a belief over them, a priced stop, and doors published as probabilities

Date: 2026-11-10. Status: DRAFT for the owner's review. Part A is being built on the defaults of §6, which are still the owner's
to rule on.
- Done, standalone-tested: door-existence `door_concept/src/core/room_door_count.h` and `tests/test_room_door_count.cpp`.
- Done, compiled, not yet run live: the integration in specificworker.cpp step_wall_hypotheses / step_candidate_looks.
- Not yet built: ltsm's counts for the prior.
Worktrees: door-existence (door_concept), rt-build (controller, room_concept, ltsm_agent, language_agent).

## 1. Why

Ten three-room tours since clean26, one pass (clean29). Almost every failure had a different proximate cause, in a
different agent, and each was fixed where it broke:

| tour | what ended it | the decision that turned a belief into a fact |
|---|---|---|
| clean27, clean34 | no door found → DONE | DONE = "no door in the graph now" |
| clean31 | one sliver mask deleted the real door; walls not re-seeded | door node deleted on one trial |
| clean32/33 | outline annexed the next room through a slit; seam turned | outline adopted; seam turn chosen once |
| clean34 | approach hold vs arrival check on two different targets | (controller protocol) |
| clean35 | one-viewpoint hinge flip, one support row deleted the door, DONE in the gap | door node deleted; DONE read in a gap |

Two recurrent shapes account for most of it:

- **Absence read at one instant ends the tour.** The drive's Done is a predicate on the graph *now*: no found,
  uncrossed door and `door_questions == settled` (controller/src/exploration_drive.h `Drive::step`). Nothing in the model
  says that a room the robot stands in has a door. "Not found yet" is read as "there is none", so a phantom-only room
  (clean27, clean34), a deleted real door (clean31) or a one-cycle gap between a removal and a re-birth (clean35) all
  end in DONE.
- **A soft belief becomes an irreversible graph fact.** door_concept keeps a proper existence ledger (spec
  2026-11-07-door-existence-model-design.md) and publishes P(exists) as `obj_exist_prob`, but its output that every
  consumer acts on is binary: the door node exists or was deleted (L < delete line). Of the 15 source files that read
  door nodes, two read `obj_exist_prob` (controller specificworker_mission.cpp, as a cut at 1/2; room_concept
  object_anchor_source.cpp). One unlucky trial therefore deletes an object, its affordances and its place in every
  consumer's world.

This design removes both shapes at the model level. Part A gives the room a belief over its doors, including the
ones not found yet, and makes stopping a priced decision on that belief. Part B keeps a door in the graph for as long
as the robot is in its room and lets its probability, not its node, carry existence.

## 2. Goals, non-goals, anti-goals

Goals
- A1. A room's belief states P(the room has a door not yet found), from a door-count prior and the per-place evidence
  door_concept already keeps.
- A2. Looking on and stopping are compared in one currency, seconds, the currency of the existence spec's C_keep and
  C_delete. "Open questions" is the outcome of that comparison, not "some candidate is undecided".
- A3. Removing or losing a door raises P(undiscovered door). It never ends the tour.
- B1. Within a room visit a born door is never deleted for absence. Its `obj_exist_prob` falls instead, and every
  consumer acts on the probability.
- B2. Each door consumer is named, with how it reads the probability.

Non-goals
- No change to how per-place evidence is gathered (masks, ZED, ricoh, support, geometry: spec 2026-11-07, with today's
  coverage and overlap weighting).
- No change to the crossing, the passage contract or recognition, beyond reading `obj_exist_prob` where Part B says.
- The refiner's outline (Part B touches only how a doorway constrains it).

Anti-goals: what would make this a failure even if it works
- A new threshold, gate or timeout. MaxRoomS stays the one flagged room limit and is given a meaning (§3.4).
- A stop rule that never stops: a phantom that no look can resolve must not hold the robot forever (§3.5).
- Phantoms leaking into long-term memory because Part B keeps them on the graph (§4.3).

## 3. Part A: the room's door belief and the priced stop

### 3.1 Variables

For the current, promoted room:
- The **places** i = 1..K that door_concept already holds as door hypotheses: the wall seeds (one per door width along
  the outline, `step_wall_hypotheses`), the mask candidates (`[candidate] place_k`), the doorway hypotheses from
  room_concept's necks, and the born doors. Each has an existence belief with log-odds L_i
  (`existence_update.cpp`: band + base + size + carried + crossing + the view ledger, which already scores door
  against nothing and look-alike, and saturation and the physical term Sp).
- **The odds the DP uses: O_i = exp(L_i − base_i).** The count prior (3.2) *replaces* the per-wall base rate
  (`place_prior.h`: 0.4 for a wall's first door, 0.01 for each further one, flagged constants of the existence spec
  §6), which was a door-count prior stated wall by wall. Everything else in L_i stays with the place: band and size
  (where and how wide), carried, crossing, the ledger with its look-alike hypothesis, Sp and saturation. A crossed door
  is proof: its crossing term already drives O_i → ∞, so it is a door in every configuration and needs no special case.
- N = the number of doors in the room = Σ_i d_i, with d_i ∈ {0,1} for "place i is a door".
- **Found is belief-weighted.** f_i is the weight with which place i is a door the drive can already act on.
  - Part A: f_i = 1 for a born door (any non-bearing instance; the entry door is one), and 0 for any other place.
  - Part B: f_i = the door's published `obj_exist_prob`, so a born door at p ≈ 0 is not "found", and both parts give
    the same answer.
  - "Undiscovered" means the door part of a place not covered by f_i.

### 3.2 The door-count prior: a room has at least one door

P(N = n) is a count prior with **P(N = 0) = 0**: a room that can be entered has a door. For every room but the start
room this is already satisfied by the entry door, which is known. It bites in the start room, which is exactly where
clean27 and clean34 ended. It is the model's one structural assumption, valid for the homes this robot works in.
Flagged; the owner rules (§6).

For n ≥ 1, P(N = n) comes from the doors-per-room counts of the rooms in long-term memory (empirical Bayes; ltsm holds
every remembered room's doors). Until two rooms are remembered, a flagged initial prior is used: geometric on n ≥ 1
with mean 2 (an apartment room has one or two doors). With memory it is replaced, never tuned.

### 3.3 The posterior

The per-place evidence Λ_i is independent across places (separate views of separate wall stretches). The count prior
couples the places:

    P(d_1..d_K | evidence) ∝ [ P(N = n) / C(K, n) ] · Π_i O_i^{d_i},     n = Σ d_i

(O_i from 3.1). Given n doors, which n of the K places hold them is a priori uniform, hence the 1/C(K, n). Without it, the
number of combinations swamps the count prior: the first implementation put a room with 14 evidence-free places at the
prior's ceiling. With no evidence the posterior count is the prior and every place carries E[N]/K. Knowing that one place
is a door size-biases the count (P(N = n | it) ∝ n P(N = n)), which is correct and tested. The computation is a dynamic
programme over places, O(K² · n_max), with K ≈ 20 to 40 (`door_concept/src/core/room_door_count.h`,
`tests/test_room_door_count.cpp`).

**The places are slots: one fixed set covering the walls.** The exchangeable prior needs K to mean "the locations a door
could be", each counted once. The slots are the wall seeds, one per door width along the outline. So wall seeds are **no
longer retired** when a door stands; they are how a second door gets looked for. A mask candidate, an outline doorway or a
born door near the wall takes the slot it falls in (its belief replaces the evidence-free seed's: it is the same location,
seen). One that falls in no slot, such as a phantom off the wall line, adds its own slot. The posterior gives:

- **p_more = P(Σ_i d_i (1 − f_i) ≥ 1 | evidence)**: the room has a door the drive cannot act on yet. In the DP, each
  place's "door" mass is split into a found part (f_i) and an unfound part.
- **p'_i = P(d_i = 1 | evidence)**: each place's marginal *under the count prior*. When every other wall stretch has
  been ruled out, the last unseen stretch carries the start room's "at least one door". The look planner uses p'_i in
  place of the independent p_i, so that stretch becomes the most valuable look.

### 3.4 The priced stop

Each look a at place i (the existing `rc::door::look::plan`: standpoint, yaw and p_detect_a, where p_detect is
nbv::score's envelope × visibility of that standpoint) costs t_a seconds. If it finds a door, the room gains a door to
cross. Its value is

    VOI_a = C_miss · p'_i (1 − f_i) · h · p_detect_a        (seconds)

- C_miss is the cost, in seconds, of leaving a door undiscovered.
- h is the ZED hit rate (`door_detectors.csv`).
- p'_i (1 − f_i) · h · p_detect_a is the probability that this look finds a door the drive cannot act on yet. Visibility
  enters once, inside p_detect.
- **This replaces the planner's ranking.** Today it ranks by p_detect · existence_eig(p_exists), an entropy gain in nats
  on one place. VOI_a is the decision-theoretic value of that same information in the drive's currency, so the planner
  ranks by VOI_a − t_a. The nats are kept in the log.
- **t_a**: straight-line distance from the robot to the standpoint over the controller's nominal speed (`max_adv`),
  plus the controller's post-arrival dwell (the 3 s `DWELL`, an existing constant). The straight line is a lower bound,
  so the error leans towards looking, never towards stopping early. No route planning in door_concept.

The room's door questions are **open** iff some look pays: max_a (VOI_a − t_a) > 0. They are **settled** otherwise.
Stopping costs p_more · C_miss in expectation, and that is what the best look is weighed against. `door_questions`
keeps its open|settled `state` for the drive (§3.6). p_more and the best look's VOI and cost go to door_concept's log
and CSV only, so **no new DSR attribute** is needed (no `REGISTER_TYPE`, no cortex reinstall).

**C_miss: no new constant, and constant per visit.** MaxRoomS (300 s) is already the one flagged room limit, the time
the drive is willing to spend in a room. Its meaning here is "a door that might be in this room is worth up to MaxRoomS
of looking", so C_miss := MaxRoomS for the whole visit. It does not shrink as the clock runs. The two roles stay
separate:
- the **price** decides whether a look is worth its time;
- the drive's existing `timed_out` clock (also MaxRoomS) stays as the outer cap, so a degenerate case still ends.

In normal operation a room ends because no look pays; the clock is the backstop.

### 3.5 What this changes in each failure, and how it terminates

- clean27/34 (start room, only a phantom found). N ≥ 1 and no real door found. As looks at the phantom's place come
  back empty, the phantom's p'_i falls and the mass moves to the unseen wall stretches; their VOI exceeds their cost,
  so the robot looks there. DOOR_2 lies on such a stretch.
- clean31/35 (the only real door deleted, or briefly removed). With Part B it is not deleted at all. Even without Part B,
  its place keeps its own evidence and p_more jumps (N ≥ 1 needs a door somewhere), so the questions are open.
  DONE cannot fire in the gap: the stop is a belief, not the presence of a node.
- Termination: every look either detects (the place's Λ rises, the door is found, it joins F) or misses (Λ falls by
  log(1 − p_detect), which is finite). With K places and finite looks per place (the ViewLedger counts one trial per
  new view), max VOI − t falls below 0 once the views that add information are exhausted. A phantom no look can
  resolve stops being worth a trip once its views are spent. That is the existing "no standpoint adds a view" rule,
  now priced. The start room with N ≥ 1 and every stretch seen and refuted is a contradiction the model surfaces: the
  count prior's mass sits on the least-refuted place, its look cost exceeds its VOI, and the drive reports "settled
  with p_more = x". It does not loop.

### 3.6 Who does what

- **door_concept** (door-existence): computes the posterior (3.3) each cycle from its PlaceBeliefs, uses p'_i in the
  look planner, and publishes `door_questions` {state, p_more, voi_s, cost_s} for the current room. A new header,
  `src/core/room_door_count.h` (pure; DP + marginals), tested standalone.
- **controller** `exploration_drive.h`: unchanged in shape. `doors_settled` still reads `state`, but `state` now means
  "no look pays". Logged with p_more. The door_coming and remembered_uncrossed holds stay.
- **ltsm**: exposes doors-per-room counts for the prior as a field of the `ltsm_topology` JSON it already publishes
  (a string attribute, so no new DSR type), read by door_concept.
- **Termination** rests on two existing things: the ViewLedger counts one trial per *new* standpoint kernel, and the
  planner offers only looks that add a view ("stays not born: no standpoint adds a view"). A place's looks are
  therefore finite, and with them its VOI. Anything the model cannot settle is still caught by MaxRoomS.

## 4. Part B: doors published as probabilities

### 4.1 The rule

Within a room visit, a born door node lives until the robot leaves the room (eviction) or the tracker merges it into
another (identity). The existence decision no longer deletes it. Its `obj_exist_prob` is the belief and goes as low as
the evidence says. The existence spec's costed delete (L < −ln(C_delete / C_keep)) is not dropped. It moves to the one
place where deleting has a real cost and benefit: **what long-term memory keeps** at eviction (§4.3). Inside a visit,
keeping a node costs nothing once consumers weight by p (C_keep is incurred only by acting on a phantom as if it were
certain), and deleting it costs the re-find, C_delete. So the delete line is never crossed inside a visit.

### 4.2 Consumers, and how each reads the probability

| consumer (file) | today | with Part B |
|---|---|---|
| controller drive (`specificworker_mission.cpp`, `exploration_drive.h`) | `believed` = p ≥ 1/2; nearest found door | choose the door that maximises p · V_cross − t_travel (V_cross = C_miss, §3.4). The ½ cut goes away. |
| door affordances (door_concept approach/open/cross) | already multiply P(exists) into P(served) | unchanged: the offer's value falls with p. A door at p ≈ 0 still has a node, with a near-worthless offer. |
| controller passage / approach order (`specificworker.cpp`) | acts on any door node | a passage in flight whose door's p falls is the existing `in_flight_disbelieved` path, re-expressed as expected value |
| room_concept doorways (`door_apertures.cpp` → `rc::refine::Doorway`, refiner `grows_through_doorway`) | hard: candidates growing through a known doorway are erased before scoring | `Doorway` gains `p` (the door's `obj_exist_prob`, carried through `DoorAperture`). A candidate that grows through doorway d is no longer erased: its gain gains −log(1 − p_d) nats, **once per edit** (the event "the outline passes this door"), added before the adopt test. A certain door forbids; a doubtful one barely resists. |
| ltsm passage harvest / live (`passage_harvest.cpp`, `passage_live.cpp`) | any door node | a passage is born from a crossing, which is proof whatever p was. Unchanged. |
| ltsm eviction / recall / swap / topology (`room_eviction.*`, `room_recall.cpp`, `room_swap.cpp`, `topology_*.cpp`) | copies every door face into memory | §4.3: the costed keep/forget at eviction; the stored face carries p |
| ltsm recognizer (`room_recognizer.cpp`) | crossed door's faces | crossed door: proof, unchanged |
| language agent (`world_digest.cpp`) | lists door nodes | says "a possible door" below p = 1/2: a wording choice, not a decision |

### 4.3 Memory: the costed decision moves to eviction

At eviction ltsm stores the room. For each door face it applies the existence spec's costed rule once:
keep iff P(door)/P(no door) > C_keep_mem / C_forget, where
- C_keep_mem is the cost of remembering a phantom: the next visit's recall offers it and a look refutes it, about one
  look, which is C_keep;
- C_forget is the cost of forgetting a real door: re-finding it on the next visit, which is C_delete.

So the line is the existing −ln(C_delete / C_keep). It is applied once per visit, on everything the visit learned, not
on one frame. A kept face stores its p in the `obj_exist_prob` it already carries when copied (no new attribute), and
recall seeds the door's prior from it.

Inside the visit, a door at low p keeps its node. So `door_apertures` keeps its aperture, now weighted (the refiner row
above), and the place pipeline keeps its PlaceBelief, as it already does for a REMOVED place.

### 4.4 Migration

door_concept gains a config switch `Existence.DeleteWithinVisit` (default true, today's behaviour) so Part A can ship
first. Each consumer in §4.2 is moved to reading p, with a test, and then the switch flips to false. The switch is
removed once the series passes with it off.

## 5. Tests

Part A
- `room_door_count.h` (standalone):
  - with no evidence, the DP agrees with brute-force enumeration for K ≤ 12;
  - P(N = 0) = 0 is honoured;
  - start-room case: K = 20 wall places, a refuted phantom and 19 unseen; p_more → 1 and the unseen places' p'_i rise
    as the phantom is refuted;
  - non-start room: entry door in F, p_more follows the count prior;
  - all places refuted: the posterior concentrates on the least refuted place and the stop still terminates (VOI < t).
- door_concept look planner: it uses p'_i; clean34's geometry (phantom behind the north wall, DOOR_2 on the east wall)
  ranks an east-wall look first once the phantom's place has two empty views.
- The drive: with `door_questions` {open, p_more 0.9}, there is no Done at any instant of a door's removal and re-birth
  (clean35's sequence, replayed as states).
- Live: the series from all three starts; pass = `tour_pass.py`, outlines checked, plus `door_questions` p_more in the
  logs at every Done.

Part B
- door_concept: a door whose L falls below the delete line keeps its node with p < 0.25 within the visit, and is
  dropped from memory at eviction.
- The drive: two found doors, one at p 0.9 and 6 m away, one at p 0.2 and 2 m away; the choice follows p · V − t.
- refiner: an edit through a doorway with p = 0.99 is rejected; through one with p = 0.05, the strip's own evidence
  decides (synthetic, as in tools/refine_doorway_test.cpp).
- ltsm eviction: faces kept and dropped by the costed rule; recall seeds the prior from the stored p.

## 6. For the owner to rule on

**RULED 2026-11-11 (owner: "accept them"): all four defaults below accepted as written** — P(N=0)=0; C_miss := MaxRoomS;
geometric count prior with mean 2 until memory holds rooms; per-wall base rates retired.


1. **P(N = 0) = 0** as the model's structural assumption (homes; a robot placed in a closed room is out of scope).
2. **C_miss := MaxRoomS**, turning the flagged timeout into the price of a door. The alternative is a separate,
   flagged mission value.
3. **The initial count prior** before memory holds two rooms: geometric on n ≥ 1 with mean 2.
4. **Retire the per-wall base rates** (0.4 for a wall's first door, 0.01 for each further one; existence spec §6). The
   room-level count prior replaces them, so keeping both would count the door number twice. What is lost is "a second
   door on the same wall is rare", which the room-level prior does not express. The alternative is to keep the
   per-wall rates and drop the count prior, but that cannot say "at least one door".
5. **Order:** Part A first (it addresses clean27, 31, 34 and 35 directly and touches only door_concept and the drive's
   logs), then Part B consumer by consumer behind the switch.

## 7. Left to the builder

- The DP's numerics (log space; K up to ~40).
- The field name in the `ltsm_topology` JSON for doors-per-room counts.
- How the language agent words "possible door".

Settled in this draft after a builder's review (my calls, open to the owner's veto):
- O_i = exp(L_i − base_i), the place's own evidence and geometry;
- found is belief-weighted (f_i);
- VOI_a replaces the planner's nats ranking, with visibility counted once;
- t_a is the straight line over nominal speed, plus the dwell;
- C_miss is constant per visit, with the clock kept as the outer cap;
- no new DSR attributes;
- the refiner's doorway charge is one event per edit.
