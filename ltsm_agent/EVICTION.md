# Room eviction — moving a departed room into long-term spatial memory

Step 2 of the crossing-door plan. Step 1 (the agent, two graphs, two DDS domains) is done and
described in [README.md](README.md); this is what `compute()` is supposed to do.

## The plan, and which half is whose

> When the robot crosses the door it loses the corners and walls that provided the pose, and falls
> back to dead reckoning on odometry alone. Once the LiDAR and the ricoh start to see the new room,
> layout estimation takes over. That exact point and orientation is kept as the last reference in
> the former room and as the new zero in the new one. After the layout stabilises the room is
> created and hung from the robot, giving the robot **two RTs: one to room 0 and one to room 1**. At
> that point the LTSM agent removes the old room and moves it to memory, leaving the new one as the
> current room and the door as the connecting element, in LTSM, with coordinates in both rooms'
> frames.

Two components, one seam:

| | owner | state |
|---|---|---|
| dead reckoning across the door, second layout, **second room node hung from the robot** | `room_concept` | **deferred — later** |
| detect two rooms, measure the seam, copy the old one into memory, delete it from the live graph | `ltsm_agent` | **this document** |

`room_concept` today creates exactly one node, literally named `room`
(`room_scene_graph.cpp:712`), and on startup ADOPTS any existing `room`-typed node it finds
(`:685`). So a second room cannot be born yet, and the eviction trigger cannot fire live. That is
fine and is the reason for the stage fixture below: **the whole transaction is built and tested
offline first**, against a synthetic two-room graph on a private domain, and only then pointed at
the live fleet.

### What `ltsm_agent` needs from `room_concept` (the contract to satisfy later)

1. Rooms get **distinct node names** (`room_0`, `room_1`, …) and a `room_id` attribute
   (`dsr_attr_name.h:278`, registered, currently written by nobody). Keeping the name `room` for
   the current one and adopting-by-type is what forbids two.
2. The robot carries **both** `RT` edges while the hand-over settles; the old one is not deleted by
   `room_concept`.
3. `room_concept` does **not** touch the `current` edge. Which of the two rooms is the newer one
   is read from the birth stamp `timestamp_creation` that `rc::provenance` already writes at
   creation — an ordering, not a threshold. The **`current` edge is owned and moved by LTSM alone**
   (see "the let-go rule"), because it is also the fence that releases the fleet.
4. The second room node appears *only once its layout has taken over*. Existence of the second
   room IS the stability signal; LTSM does not evaluate a stability criterion of its own (no
   thresholds here — CLAUDE.md).

## Trigger

On the live graph: **more than one `room`-typed node reachable by `RT` from the robot node.**
The one with the later `timestamp_creation` is the new room; every older one is a candidate,
oldest first, one per cycle. Nothing else arms it; no timer, no distance, no sigma.

## The transaction, in order

The order is the whole design: **nothing is deleted before the copy is verified**, and every phase
is idempotent so a crash between two of them leaves a graph the next run can finish.

**1 — Measure the seam, first.** With both RT edges still in place, the two room frames are related
through the robot:

`T(room_old→room_new) = T(room_old→robot) · T(robot→room_new)`

read from `inner_eigen->get_transformation_matrix("room_new", "room_old", ts)` (main thread,
optional checked — CLAUDE.md). This transform is the single most valuable thing eviction produces:
it is what lets memory compose the two rooms afterwards, and it is **only computable in this
window**, while both edges exist. It is measured and stored before anything is copied.

**2 — Copy the old room's subtree into `G_ltsm`.** BFS from the old room node over outgoing `RT`
edges (never entering the robot), copying node type, attributes and the RT edges'
`rt_translation` / `rt_rotation_euler_xyz` / timestamps. Ids are reassigned by the memory graph, so
a `live id → ltsm id` map carries the recursion. Names are suffixed per room (`wall_2` → `r0_wall_2`)
because DSR names are unique per graph and the next room brings its own `wall_2`; the room number
also goes on every copied node as `room_id`.

**3 — Hang the new room's placeholder off the old one, with the seam as its RT.** Memory becomes a
topological map whose RT tree IS the room graph:

```
root
 └RT→ r0_room ──────────────── the evicted room, fully populated
        ├RT→ r0_floor
        │      └RT→ r0_wall_0 … r0_wall_3
        │             └RT→ r0_door_1   room 0's OWN measurement of the hole, in its own frame
        ├RT→ r0_table_1, r0_fridge_1, …
        └RT→ r1_room                   STUB, RT = the seam measured in phase 1

passage_0   type `metaconcept`, object_subtype "passage"   ← NO RT edge, NO pose
   ├match→ r0_door_1                   the side we have seen
   └exit→  r1_room                     the room reached through THIS hole
```

**Two structures, not one.** The RT tree carries metric composition; the passage nodes carry
topology. A passage belongs to BOTH rooms and the RT tree is a tree, so it can never hang correctly
from either — the answer is that it does not hang in the tree at all.

Using RT for room→room keeps every node in memory reachable from `root` by the ordinary RT chain,
so `inner_eigen` works across rooms with no special case. The `exit` edge (`dsr_edge_type.h:65`)
marks *which* RT link is a doorway, and `match` edges link the two faces of one door. The
placeholder is filled in — not duplicated — when room 1 is itself evicted later; that is also the
hook for recognising a room the robot has been in before.

**4 — The door as the connecting element: ONE abstract passage, not a pair of faces.**

The crossed door lives in the old room as a child of one of its walls
(`door_scene_graph.cpp:140-147`) and comes over with the subtree as `r0_door_1` — room 0's own
measurement of the hole, in room 0's frame. Above it sits one **abstract passage node**: type
`metaconcept`, `object_subtype = "passage"`, **frameless** — no RT edge, no pose, no geometry. It is
the hole itself, and it is what owns the passage statistic (below).

- `match` edge → the room-side face we have actually seen. When room 1 is evicted later, carrying
  door_concept's own door node for the same hole, that node takes the SECOND `match` edge.
- `exit` edge → the room reached through this hole. It sits here rather than between the two rooms
  because a room pair can have more than one door, and only this edge says which one was crossed.

Three things this shape buys, each of which the earlier cross-linked-pair draft got wrong:

- **The far side's pose is not stored — it is composed.** `r0_room --RT[seam]--> r1_room` means
  `inner_eigen` answers "where is this door in room 1's frame?" from the numbers already there.
  Storing it as well would be a second pose for one physical hole: a thing that can disagree with
  itself. The plan's "coordinates in both rooms' frames" is satisfied by composition.
- **The far face is not fabricated.** A face derived from the seam can never disagree with the seam.
  A face MEASURED from room 1 can — and that disagreement is the loop closure that grades, and later
  corrects, the seam. Waiting for door_concept's own node is what keeps the second measurement
  independent.
- **`other_side_door_name` is not used.** Two `match` edges into one node say it structurally, and a
  string naming a peer node is exactly what goes stale when door_concept recycles a freed `door_N`.

★ **The node type is `metaconcept`, and NOT the registered-but-vacant `door`.** Cortex's own comment
on `metaconcept` is the spec — "a concept OVER concepts (level-2): an arrangement/grouping that
BELIEVES in a relation among other nodes rather than in a solid body of its own". And `door` only
looks free: door_concept's owned-node sweep is {types `object`,`door`} × {name starts with "door"}
(`common/owned_nodes/owned_nodes.h:58`), so a `door`-typed node named `doorway_*` in the LIVE graph
would be deleted by name at its next startup, silently. Immaterial on domain 2; free to avoid.
★ The lookup filters on the SUBTYPE as well as the type: `metaconcept` is shared (ring_metaconcept's
dining sets), and eviction copies whatever hangs under a room, so memory can hold metaconcepts that
are not passages.

### The passage statistic — a belief, not a log

What a passage accumulates is *how easy this hole is to get through*, and this repo already has a
prescription for that shape: `MODEL_HISTORY.md` §4 turns a constant false-alarm rate into an
inferred `p_FA(place × bearing)` — counts, conjugate posterior, no event list. The same move here:

| where | what | why there |
|---|---|---|
| **passage node, in memory** | two Beta posteriors — `p(passable without intervention)` and `p(an open request succeeds)` — plus the crossing count, the last crossing's local time as a STRING, and the last harvested sequence number | small, bounded, single-writer (LTSM only), and it is the form the pragmatic value of a `cross` affordance actually needs |
| **`etc/passages.csv`, on disk** | one append-only row per crossing: sequence, time, passable, open_prob, clear span, body width, open requested/answer, `aff_outcome`, duration | append-only ⇒ immune to BOTH hazards — no read-modify-write to lose an increment, no whole-node replace to erase a sibling attribute; survives a restart by construction; and it is the artefact you read when diagnosing, which is how everything else in this fleet is debugged |

The CSV is the durable truth and the Beta is the belief derived from it — so if the memory graph is
ever lost, the statistic can be recomputed. Written with `imbue(std::locale::classic())` and read
with `std::from_chars`, per CLAUDE.md.

**The last-crossing time is a STRING** — ISO-8601 local time, the `rc::provenance::creation_datetime`
format — because nothing computes with it and a person reading the memory graph in the viewer is its
only consumer; an epoch count there tells nobody anything. (Not for round-trip safety: uint64 node
attributes survive the JSON round trip perfectly well. An earlier draft of this file said otherwise
by reading the EDGE copy of the loader's attribute switch and taking it for the node one.) If
something later needs to subtract two crossing times, add the uint64 beside it then, the way
`rc::provenance` carries both.

**The unit of record is a VISIT, and what folds into the Beta is EVIDENCE, not a probability.**
Two ways this channel would otherwise manufacture confidence, both caught in review:

- *Per-cycle rows.* A robot parked in front of a shut door at 10 Hz writes ~600 rows a minute, ~95%
  of them the same pixels (door_concept's own `DetectProbeCsvPath` note says this about its
  absence integration). The posterior would then be dominated by how long the robot happened to
  loiter. So one row per VISIT — enter the zone, judge, leave or cross, emit on close — which is
  also the unit the question is about: *when I want to use this doorway, is it usable?*
- *Fractional counts.* `α += p, β += (1−p)` adds one count of mass per visit however little the
  visit discriminated, so it shrinks the variance at the same rate whether or not anything was
  learnt. A hundred looks that told us nothing give Beta(51,51) — "50/50 to ±5%", which nobody
  measured. The update takes a LIKELIHOOD RATIO instead: the exact posterior is a mixture of
  `Beta(α+1,β)` and `Beta(α,β+1)` weighted by `mean·L1` and `(1−mean)·L0`, moment-matched back to a
  Beta. With `L1 == L0` the weight equals the current mean, the mixture IS the prior, and the update
  is an exact no-op; with `L1 ≫ L0` it is a full count. The posterior MEAN is `(α+w)/(α+β+1)` either
  way — the two estimators differ only in the variance, which is the half a pragmatic price needs.
  This is not a new convention: `common/existence_belief/existence_belief.h` already accumulates in
  log-odds from an explicit LLR scaled by `p_detect` (:187, :537), and `saturate()` there is the
  guard against one freak frame contributing an unbounded ratio.

**What folds, and what is only logged.** Not every real event is admissible evidence, and the rule
is the censoring rule applied consistently:

| row | into the Beta | why |
|---|---|---|
| claimed cross, completed | positive | intent existed and it worked |
| claimed cross, `aff_outcome = Timeout` | negative | the robot GOT THERE, tried, and did not get through — the ONLY genuine negative the channel gets |
| **which** Beta a claim credits | see below | `Satisfied`/`Timeout` split on whether the crossing was ASSISTED, and "assisted" is not "we asked" |
| claimed cross, the other five outcome words | **logged only** | `Unreachable`, `Infeasible`, `Refused`, `Abandoned`, `OutsideRoom` are facts about the APPROACH, not the aperture; folding them teaches "this doorway is blocked" when the truth is "we could not get to it" — or, for `OutsideRoom`, that the producer published a pose outside its own layout, i.e. the channel would be learning about a bug |
| **unclaimed** crossing | **logged only** | the robot walks through doorways that are already passable: folding these makes α climb while β never moves, and a door shut 90% of the time would read "reliably passable" |
| any visit's `passage_llr` | folds normally | the silhouette channel is unconditioned, which is what makes it worth having even though it is weak |
| `passage_episode_cut` rows | excluded | a partial judgement must not move a belief |

The Beta then means something a price can use — *if I ask to cross, do I get through?* — rather than
a number that drifts upward with traffic. Unclaimed crossings are not wasted: they are the honest
estimate of a different quantity (how often this door is found open) and get their own pair when
there are enough of them to be worth one.

★ **THE INVERSION AUDIT, AND IT IS FREE.** An opened leaf swings OUT OF THE CAMERA'S VIEW, so raw
silhouette support is not comparable across leaf angles — door_concept has measured the consequence
(occ 207 → 0, free_eff 0 → 33, the existence channel falling +4.00 → −4.00 in 13 cycles). An LLR
built from unnormalised support therefore learns "blocked" from the visits where the door stood
widest open: a confident WRONG answer, worse than censoring's over-confident right one. The
normalisation (mean support per VISIBLE sample, per hypothesis) is door_concept's to get right, and
this side audits it instead of trusting it: `passage_crossed = true` rows are a labelled positive
set — the robot physically went through — so **the mean `passage_llr` over crossed rows must be ≥ 0,
and its sign is the alarm.** Both columns are already in the schema, so the audit costs nothing.
★The converse is not a label: `crossed = false` usually means the robot did not want to, not that
the door was blocked. Only a claimed-and-failed cross is a negative, and a test that treated every
non-crossing as one would report an inversion whenever the robot was busy elsewhere.
★★ **And the audit must not pass VACUOUSLY.** Measured peak silhouette support on this door is
0.007–0.245 — a nearly flat curve — so most `passage_llr` are ≈ 0 and a mean over crossed rows reads
non-negative whichever way the normalisation points. An inverted channel would clear the test
exactly while it is weak, which is its expected state. So the verdict is taken from the SIGN OF THE
TOP-|llr| ROWS, reported with `n` and the mean beside it, and when no crossed row carries meaningful
evidence the answer is **INCONCLUSIVE, never PASS**. A vacuous pass is worse than no audit: it is
what gets cited later when the channel turns out to have been backwards all along.
★★ **Negatives never disappear into the Beta.** A claimed-and-failed cross is the only source of
negative evidence in the channel, so it is the most consequential row type and the one a forged
claim or an unrelated navigation failure would fabricate. The audit publishes folded negatives per
doorway with their `passage_duration_s` and `passage_outcome`, so a cluster of instant failures is
visible as a cluster instead of as a quietly shifted posterior. An instrument, not a gate.

**Which Beta a claim credits.** `passage_open_requested` is a bool about US: it says a request was
made, not that anybody could act on it. The provider may have refused (unknown door, not actuable),
or the request may never have left this fleet — in both cases nothing was actuated, so the crossing
is physically identical to one with no request at all. `passage_open_answer` carries the provider's
own words, so ASSISTED = requested AND the answer was `Delivered`/`InProgress`, and an unrecognised
answer is reported once and treated as unassisted rather than silently suppressing the evidence.

| episode | credits | why |
|---|---|---|
| `Satisfied`, unassisted | `pass_alpha` +1 | it walked through unaided — near-ground-truth about the aperture |
| `Satisfied`, assisted | `open_alpha` +1, **`pass` nothing** | see below |
| `Timeout`, assisted | `open_beta` +1, **`pass` nothing** | asked, accepted, still could not get through |
| `Timeout`, unassisted | `pass_beta` +1 | it tried to walk through and could not |

★ **The assisted branch must not charge `pass_beta`,** tempting though it is ("we asked, so it must
have been shut"). That inference rests on `door_open_prob`, which on the near-flat leaf-angle curves
this fleet measures is mostly the PRIOR. The failure is concrete: the door is already open, our
prior-dominated estimate says shut, we ask anyway, the request is a no-op, the robot walks through —
and `pass_beta` takes a full count against a doorway that was passable all along. Repeat it on every
visit to a door we habitually ask about and `pass` drifts down for a reason entirely internal to us.
Every full count stays anchored to something physical the robot did; every inference is left to the
weighted `passage_llr` channel, which is what it is for.

★ **WHAT THE `open` BETA ACTUALLY MEASURES — and it is not "does the opener work".** door_concept's
`open` can only complete on a REAL observation of the leaf, because otherwise it satisfies itself:
with no resolvable hypothesis the fitter follows the commanded swing, the passability fallback
marginalises over it, `door_open_prob` → 1.0, the completion predicate fires, and the affordance
reports `Satisfied` on a leaf nobody saw move — our own anticipation arriving as evidence through
the back door. That is fixed on their side (a prior-derived phi is marked and the fallback refuses
to run on it). The consequence lands here: at the range `open` is taken from, YOLO drops the door
(its ADE20K posterior collapses 0.995 → 0.048 as the robot closes), so a **working** provider now
produces `Timeout`, which credits `open_beta`.

So this Beta measures *"an open request produced an aperture WE COULD SEE"*, not *"the opener
works"*, and the audit counts the difference explicitly: rows with `outcome = Timeout` and an
accepted `passage_open_answer` are reported as ACCEPTED-BUT-UNCONFIRMED. `open_beta` climbing while
every answer reads `Delivered` is a statement about our perception at close range, NOT about the
provider — and without that count the two are indistinguishable.
★ **SENTINELS ARE NEGATIVE, BECAUSE 0 IS A REAL READING.** `open_prob == 0` is a door believed
SHUT — evidence — while "not measured" is an absence, and for a while both were the same number in a
channel whose whole job is telling them apart. door_concept now publishes `-1` for NOT MEASURED on
`door_phi_rad` / `door_open_prob` / `door_reach_prob`, and `-2` on `door_crossing_progress` (whose
range is [−1, +1], so −1 is a legitimate reading and could not be the sentinel). Every sentinel sits
outside its own quantity's range, so it can never be mistaken for a measurement, and every contract
predicate still fails on it.

This side had the same defect in its READER: `value_or(0.f)` made an absent attribute
indistinguishable from a real zero — a missing `clear_span_m` reading as "fully shut", a missing
`duration_s` reading as the instant-Timeout signature this agent publishes as suspicious. The
measurement columns now default to `-1`, and an episode missing `passage_datetime` or
`passage_crossed` is RECORDED (the episode happened) but never FOLDED (folding defaults would invent
evidence). `passage_llr` is the one honest exception and keeps 0: for a log-ratio 0 IS neutral and
folding it is an exact no-op, so absent and neutral coincide truthfully. `passage_seq` is 1-based,
so 0 there is unambiguously absent.
The stage run asserts the distinction survives the whole round trip — graph → CSV → parser — because
a reader that turned `-1` into `0` would silently convert every unobservable door into a confidently
shut one, months later, in a file nobody would think to doubt.

★★★ **THE LATCHING TRAP — the failure that appears after everything works.** Once the price is
honest it is derived from this Beta; the price decides whether the controller claims `cross`; and a
claim is the only thing that produces strong evidence. The belief therefore governs its own
sampling. A run of bad luck — two Timeouts at a door that happened to be shut — drops the price,
claims stop, and no evidence can arrive to correct it: the belief freezes pessimistic with every
instrument reading healthy, and only another human with another config override would free it. The
two structural answers are both already in this fleet:

- **The price must carry the EPISTEMIC term, not the mean alone.** A wide Beta is an uncertain
  doorway, and an uncertain doorway is worth trying *because* trying resolves the uncertainty. This
  is the concrete answer to why the variance mattered enough to argue about: it is not decoration
  beside the mean, it is what the expected-information-gain term consumes. A channel publishing a
  ratio could not have an escape at all.
- **Forgetting.** A pessimistic posterior that decays back toward the prior restores willingness to
  retry — which is also the truth about doors: one that was shut last month says little about today.

Neither is today's work, but a price written as a function of the mean alone is the version that
latches, and by the time it does nobody will remember why.

Because the counts are floats rather than integers, a forgetting factor (`MODEL_HISTORY.md` §3,
inferred volatility — a doorway's state genuinely changes between visits) can be added later with no
schema change. Not now; just not foreclosed.

**Who writes what.** door_concept measures a crossing and writes it on its OWN live door node
(single writer, no contention). LTSM harvests those attributes at eviction — BEFORE the `current`
flip, so the fence guarantees the read happens before the owner lets the node go — appends the CSV
row, folds the two Betas, and never writes on door_concept's node.

**Not yet designed: RECALL.** door_concept cannot read the memory graph — different DDS domain,
it joins only domain 0 — so a statistic living in memory cannot reach the price of a `cross`
affordance on its own. The return path is LTSM publishing its own `metaconcept`/"passage" node into
the LIVE graph, `match`-linked to door_concept's door node, carrying the prior for the current
room's known passages. LTSM owns that node, door_concept only reads it: one writer each. That is the
next piece of work after eviction, and until it exists the statistic accumulates without a consumer.

**5 — Verify, then flip `current`. The fleet lets go of its own nodes.**

Only after the memory side reports the expected node and edge counts does LTSM move the `current`
edge from the old room to the new one. That flip is the whole release mechanism:

> ★ **THE LET-GO RULE.** An agent anchored to a room that is not the current one lets that room go:
> it stops fitting, and it REMOVES ITS OWN NODES through its own cleanup path — the same one it
> runs at shutdown. Nobody deletes anybody else's nodes.

This replaces the first draft of this document, in which LTSM walked the old subtree and deleted
`wall_*`, `door_*`, `table_*`, `refrigerator_*` … itself. That would have made this the one agent
that deletes nodes it does not own, against `CONCEPT_AGENT_LIFECYCLE.md`'s OWNERSHIP invariant, and
it would have raced every owner: an agent that still had the object in view would re-create the
node behind the deletion. With the let-go rule each owner removes its own nodes, in its own cycle,
with its own history and existence bookkeeping intact — and `room_concept` removes the old `room`,
its `floor` and its `wall_*` the same way.

Three consequences, which are what make the rule safe:

- **Order.** The snapshot happens BEFORE the flip. If the fleet let go first, memory would archive
  an already-emptied room. The flip is the serialisation point and LTSM owns it, so this is a
  fence, not a race.
- **Absence of the edge is not a release.** An agent lets go only when a `current` edge exists AND
  points at a DIFFERENT room. A graph with no `current` edge anywhere — the whole fleet today, with
  no LTSM running — releases nothing. The rule is backward compatible and LTSM stays optional.
- **A backstop, not a policy.** An owner that is down (or was `kill -9`ed) cannot let go, so its
  nodes would linger under a room nobody works in. LTSM keeps a sweep for exactly that, armed
  separately from eviction and logged node by node, in the spirit of the startup stale-sweep every
  concept agent already runs. It is a leak reaper, never the normal path.

The fleet-wide half — each concept agent learning the rule — belongs in
`CONCEPT_AGENT_LIFECYCLE.md` as a stage of the contract next to REMOVE, with a row per agent in its
conformance audit. It is written here first because eviction is what gives it a meaning.

## Prior art

`robocomp-shadow/agents/long_term_spatial_memory_agent` (Python) already implements this state
machine — `idle → crossing → crossed → known_room | initializing_room → initializing_doors →
removing → idle`, with `removing_dsr_room()` at `specificworker.py:915`. Two things do not port:
its memory is an **igraph pickled to `~/igraph_LTSM/graph.pkl`**, not a DSR domain, and it assumes
**room→robot** RT while this fleet publishes **robot→room** (`room_scene_graph.cpp:430`). Its
delete-by-descending-level ordering and its `has`-edge sweep do port, and are used above.

### Bootstrapping the passage statistic takes TWO human acts, in this order

The strong channel folds only CLAIMED attempts; a claim needs the controller to select `cross`; that
needs a price; and the price is the placeholder this statistic exists to replace. So it cannot
bootstrap itself, and the way out is a person — but it is two different acts, and the order matters:

1. **Joystick crossings.** Driving through the doorway is an UNCLAIMED crossing: logged, not folded,
   so it moves no Beta. What it produces is `passage_crossed = true` rows carrying `passage_llr` —
   exactly the labelled positive set the inversion audit needs. This validates that the silhouette
   channel is not learning backwards, BEFORE anything is bootstrapped from it.
2. **One claimed attempt.** Exercising the affordance path end to end — the controller selects
   `cross`, door_concept reports an outcome, the row folds. Someone has to make a claim actually
   happen; driving through the door does not do it. The cheapest honest way is raising
   `DoorAffordance.ValueCross` in door_concept's config for one run: the EFE selection then picks
   `cross` itself and the protocol runs end to end, so the only arbitrary thing is a price that is
   an admitted placeholder anyway. **It must be put back**, and that run cannot be shared with the
   room-handover test — while the value is raised, `cross` outbids the whole fleet's exploration,
   including the room work that the handover needs.
   ★ What must NOT be done, in either agent: setting `active=true` on the affordance node from the
   producer side. It is one line and it would produce a row, which is exactly the danger — nothing
   would execute the manoeuvre, so the row carries `crossed=false` with a fabricated outcome and
   folds as NEGATIVE evidence. That seeds the belief with manufactured failures at the very doorway
   being learnt about, and the CSV looks populated and healthy throughout. It is also the boundary
   the affordance protocol exists to keep: a claim the consumer never made.

Doing 2 before 1 would bootstrap a belief from a channel nobody had checked the sign of. And the
most diagnostic joystick crossings are the ones through a doorway standing WIDE open: that is the
highest-|llr| case and where the visibility normalisation is under most strain.

## Testing it without the fleet and without room_concept

`etc/config_stage.toml`: the `dsr` sub-table is seeded from a **fixture file on its own private
domain (3)** instead of joining the live graph on domain 0. The agent then stands up a synthetic
two-room graph — `root → Shadow → {room_0 (+4 walls +1 door), room_1}`, both RT edges present, the
`current` edge on `room_1` — runs the real eviction code against it, and asserts the outcome in
both graphs:

- memory holds the old room's subtree, with the door present twice and the seam on the
  `r0_room → r1_room` RT;
- the door's pose in room-1 coordinates round-trips back to its room-0 pose through the seam;
- in the fake live graph the `current` edge has MOVED to room_1 and the old room is still there,
  intact and simply no longer current — which is exactly what the fleet sees at the instant the
  let-go rule fires, and lets the test check the fence without needing any other agent running;
- nothing crossed domains (the isolation check from step 1 keeps running).

This cannot touch the fleet: domain 3 is nobody's. When `room_concept` grows the second room, the
same code runs live by pointing the `dsr` sub-table back at `configFile = ""` / `domain = 0` and
setting `[Eviction] enabled = true`.
