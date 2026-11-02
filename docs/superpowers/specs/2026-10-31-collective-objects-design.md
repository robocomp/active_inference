# Collective objects: seeing the forest before the trees — design

Date: 2026-10-31. Status: DRAFT, from a brainstorm in the chat_agent session. Nothing is built yet.

## 1. Intent

Some household structures are **collectives**: kitchen cabinet runs and wall shelving. They have no predefined
form, only a pattern, and they are the hardest things the robot has to perceive.

- **Their extent is arbitrary.** A run is as long as its wall. A partial view says nothing about where it ends.
- **They are identified by repetition**, not by an outline: cells, door fronts, shelf boards.
- **Their units are often indistinguishable.** Kitchen fronts are usually white and almost textureless. The
  boundary between two modules is a thin seam, a handle or a shadow gap, and is frequently invisible.
- **They are fused with the room.** They are built into or against a wall and compete with it.

The robot should perceive **the collective first**, the forest, from accumulated evidence. YOLO and ADE20K masks
strongly indicate "kitchen cabinets" or "wall shelves" over a region, and depth geometry confirms it. The
**units**, the trees, are inferred *inside* the collective, and only as far as the evidence allows.

Why now:
- **cabinet_concept's current implementation is not good enough** (the user, 2026-10-31). Its kitchen-of-runs
  model is described in §4.
- **The perception-card work** (chat_agent's `can_see`) needs an honest answer for these objects, and their
  answer is different: "I can see a 2.4 m run of cabinets, but I can't tell how many doors it has from here."
- **The chat** needs a referent for "the cabinets" as a whole, and later for "the second cabinet from the left".

## 2. Where collectives sit in the concept taxonomy

The user views the object kinds as a **complexity hierarchy**, in which each level includes the ones below:

| level | adds | example |
|---|---|---|
| mass | stuff: a region that is seen or not | floor, residual occupancy |
| countable | an individual with a boundary | table, chair, bottle, fridge |
| articulated | internal configuration (joints) | door, cabinet with doors |
| animate | self-motion | person |

This **supersedes the "three independent axes" wording** of `CONCEPT_AGENT_LIFECYCLE.md` §6.1, which is a
DRAFT, and §6.1 must be reconciled with it.

**A collective is not a fifth level.** It is a **grouping over members of any level**: cabinets (countable or
articulated), books on a shelf (countable), chairs around a table, a row of doors, a group of people (animate).
Linguistically it is a collective or aggregate noun: "the cabinets" versus "a cabinet". §6.1's
"kitchen / ring metaconcepts" row already points this way, calling a kitchen unit a count wrapper over mass runs.

**Representation: a metaconcept.** The graph already uses `metaconcept` nodes for abstract things that group or
summarise concrete nodes (the LTSM passage node, the `recall_*` nodes). A collective is a metaconcept that owns
the beliefs belonging to the *whole* and to no single member.

## 3. The model

### 3.0 Structure: collective, then segments, then units (agreed with the user, 2026-10-31)

A collective can span supports. An **L-shaped** kitchen run turns a corner, and a **U-shaped** counter can have
an aisle. So it has three levels:

| level | what it is | RT connection |
|---|---|---|
| **collective** (e.g. "the kitchen counter") | `metaconcept`: the whole L or U, with its existence, kind and membership | **no RT edge**, like the LTSM passage node. It links to its segments by membership edges, and its room is recorded in `parent` |
| **segment** | one straight run on one support and one tier: a stretch along a wall, or a peninsula or island with the floor as support | **RT from the room.** The segment's frame *is* its support chart: origin at the run's start, x along the support, y out of it, z up |
| **unit** (one cabinet, once resolved) | a countable or articulated member | **RT from its segment**, translation only (position along the run, depth, height). Rotation is fixed at zero, because the structure is exact in the chart |

- **Today's occupied-volume boxes become the segments.** `cabinet_w<seg>_<base|up>` already exist one per
  (wall, tier) with an RT from the room, and they keep serving navigation.
- **Units ride on their segment.** A re-fit of the segment moves all of its units, so there is one writer for
  one state.
- **Corner units belong to one segment.** A corner cabinet of an L or U is a unit of one segment, with an
  "adjacent" link to the other, so it is never counted twice. This ties in with the existing corner-fill
  handling.
- **Order of creation:**
  1. The collective is born **together with its first segment**, once evidence has accumulated on one support.
  2. Further segments join as their evidence builds up. A join is an UPDATE of the collective, not a birth.
  3. Units are born inside a segment only when they are resolved (§3.2).
- **Linking segments into one collective.** Two segments belong to the same collective when they are
  adjacent: meeting at a room corner, or facing each other across an aisle that is narrow for their kind. This
  must be a posterior over "same collective", not a distance cutoff. The builder designs the likelihood.

### 3.1 The collective hypothesis (the forest)

The collective's beliefs are held per segment, as listed below. The collective itself holds existence, kind and
membership.

- **Support:** for each segment, the wall it is attached to (a polygon edge of the room) or the floor, and its
  tier (base or upper).
- **Region in the support's chart:**
  - along-wall extent `[t0, t1]`;
  - depth `d` off the wall;
  - height band `[z0, z1]`.
  - The structure is exact in this chart (yaw equals the wall direction, the back is on the wall). This is the
    kitchen-of-runs insight, and it is kept.
- **Ends:** each end is *observed* (a seen termination: a wall corner, the fridge, an open floor gap) or *open*
  (not yet seen). An open end is a censored lower bound, and a partial view never shrinks it.
- **Existence:** log-odds that a collective of this kind is really here, using `rc::exist` semantics: occupancy
  confirms, predicted-visible absence removes, occlusion holds.
- **Kind of collective:** cabinet run, wall shelving, and later others. Each kind has its own prior on depth,
  height bands and unit widths.

**Evidence that builds the forest.** All of it is accumulated over time; none of it is required from a single
frame:

1. **Semantic masks.** YOLO `cabinet`, `shelf`, `counter`, `countertop`, `kitchen island`, `chest of drawers`,
   and the ADE20K `[Semantic]` classes. They are projected into the support chart and accumulated per region.
   This is the dominant birth and identity cue, because LiDAR cannot tell a wall-flush run from a wall.
2. **Depth and LiDAR geometry:**
   - a front plane standing roughly 0.3–0.6 m off the wall;
   - a countertop edge at roughly 0.9 m;
   - a plinth at the bottom;
   - for upper tiers, a gap above the counter.
   This is the metric fit, plus a consistency check against the semantic evidence.
3. **Free space as evidence of absence.** A beam that traverses the hypothesised volume and reaches the wall
   behind it shrinks that part of the run. This is the only legitimate way to shrink an end.

### 3.2 The members (the trees): a latent partition

The number of units and the positions of their boundaries are **hidden variables** of the collective:

- **Prior.** Standard module widths: 30, 40, 45, 50, 60 and 80 cm for kitchens, and per kind for shelving. A run
  of a known length therefore already implies a small set of likely partitions before any seam is seen.
- **Weak cues as boundary evidence:**
  - seams and shadow lines in the image;
  - handles, whose position implies a door edge;
  - depth discontinuities;
  - the edge of the countertop overhang;
  - changes in reflectance.
  Each cue is a likelihood term on "a boundary near t", weighted by its detectability (resolution, angle,
  lighting).
- **Posterior.** A distribution over partitions of `[t0, t1]`. This is a 1-D segmentation problem, so it can be
  solved exactly.
- **When a tree becomes a node.** A unit becomes a concrete node, a member of the metaconcept, only when *both*
  of its boundaries are well resolved. Until then the collective holds the partition as a distribution. It
  **never** creates units just to satisfy the prior.
- **Articulated members**, meaning doors and drawers as joints, come later. A resolved unit is where they will
  attach.

### 3.3 Its perception card

This links to the perception-card design. A collective's card reports **three separate numbers** for a
hypothetical viewpoint:

- **P(detect the collective):** coverage of its region times the detectability of the semantic and geometric
  cues. Typically high.
- **P(resolve the members):** the probability that the boundary cues are visible from that viewpoint. Typically
  low on white fronts, and rising sharply as the robot gets closer and views more obliquely.
- **P(see each end):** whether the viewpoint would observe each open end.

So `can_see("cabinets", …)` answers both "would I see the run?" and "would I be able to count its doors?". The
viewpoint that maximises the second is an **epistemic action** the robot can take.

## 4. Existing work this builds on

cabinet_concept already has a kitchen-of-runs model (`cabinet_kitchen.h`, `cabinet_wall_run_belief.h`):

- `(wall, tier)` cells, each owning one `WallRunBelief` in the wall chart;
- censored, grow-only extent;
- a face-gated free-space carve;
- LiDAR evidence of absence for retirement.

It **deliberately does not individuate modules**. That was confirmed with the user on 2026-07-21: downstream needs
were only occupied volume and a localisation landmark.

This design changes three things on top of it:

- it adds the **semantic forest** as the leading cue;
- it adds the **latent partition**, which is now wanted;
- it makes the collective a **metaconcept**.

The occupied-volume boxes stay, because navigation needs them.

## 5. Goals, non-goals, anti-goals

**Goals:**
- A stable collective hypothesis per run, built from evidence accumulated over time (masks plus geometry), with
  honest end states.
- A partition posterior with standard-width priors, and units promoted to nodes only when resolved.
- A perception card with the three numbers in §3.3.
- Chat referents: "the cabinets" (the collective); "the leftmost cabinet" once resolved.
- Wall shelving supported by the same machinery, as a second kind.

**Non-goals (for this design):**
- Manipulation, and opening drawers or doors. Joints come later.
- Free-standing units, such as a single bookcase. That is a countable object, not a collective.
- The detailed model of free-standing islands. Under §3.0 a peninsula or island is a segment with the floor as its
  support, but its chart and priors are designed later. For now, do not break the existing island handling.
- Other collectives (chairs around a table, rows of doors, groups of people). The representation must not
  preclude them, but none is built.

**Anti-goals (a failure even if it technically works):**
- **Inventing units:** reporting four doors because 2.4 m / 60 cm = 4 when no boundary was seen.
- **Bringing back churn:** births, merges and re-keys of runs. That is the mass-as-count symptom the
  kitchen-of-runs fix removed.
- **Thresholds instead of model terms.** Detectability, end states and unit promotion must come from
  likelihoods and posteriors (CLAUDE.md: no thresholds).
- **Eating its neighbours:** a collective that absorbs the wall (room_concept), the fridge, or the hood.

## 6. Constraints

- All CLAUDE.md rules:
  - queued signals only;
  - type-attributed DSR attributes, with any new attribute added to cortex by the user;
  - `from_chars`;
  - SIGTERM cleanup;
  - builds with `-j8`;
  - C++23 with `and`/`or`/`not`.
- `CONCEPT_AGENT_LIFECYCLE.md`:
  - birth is tracker-only;
  - occupancy confirms, absence removes, occlusion holds;
  - `[Owns]` matches `[Agent].id`.
- **Type:** the metaconcept must not take a concrete type such as `cabinet` or `box`. Another agent's sweep would
  reap it, as with the LTSM passage node and the door sweep. Use `metaconcept`.
- **Ownership (my call):** cabinet_concept owns the collective metaconcept, its members and the occupied-volume
  boxes. It is the only writer.

## 7. Decided vs left to the builder

**Decided in the brainstorm:**
- Forest first: the collective is perceived from accumulated semantic and geometric evidence; the units are
  inferred inside it.
- Units are a latent partition with standard-width priors, promoted only when resolved.
- The collective is a metaconcept.
- The kinds form a hierarchy; a collective is a grouping, not a level.
- The perception card reports separate numbers for the collective and its members.

**My calls (flagged for review):**
- cabinet_concept owns the collective; there is no new agent.
- The region is held in the support's wall chart (from the kitchen-of-runs model).
- The partition posterior is solved exactly as a 1-D segmentation.
- The card's three numbers.
- The existing occupied-volume boxes stay as derived concrete nodes.

**Left to the builder:**
- The attribute set on the metaconcept, and the membership edge type. Any new attribute needs a cortex
  `REGISTER_TYPE` that the user installs.
- How masks are projected into the chart and accumulated.
- The form of the boundary-cue likelihoods.
- How to reconcile `CONCEPT_AGENT_LIFECYCLE.md` §6.1 with the hierarchy view.
- The test strategy: synthetic runs with planted seams, plus a replay of recorded kitchen runs.

## 7b. Builder-check rulings (my calls unless marked)

- **Birth of a collective.**
  - It follows the **mass** CREATE row of `CONCEPT_AGENT_LIFECYCLE.md` §6.1: "birth of an extent cell keyed to
    its support; growth/split/merge are UPDATES".
  - Mask evidence accumulated in the support chart *is* the tracker for a mass concept. No count-style
    instance tracker is involved.
  - This exception is written into §6 next to the existing door exceptions.
- **The mask input already exists.** cabinet_concept gathers run masks through `common/mask_ingestor` today. The
  forest accumulates them per chart bin across frames, and does not decide from single frames.
- **Width prior.**
  - A per-kind config prior: a mixture over the standard widths, each component with a σ of a few cm, plus a
    broad "non-standard" component. It is not a hard set.
  - Learning it from resolved units is a later step.
- **Cues: measure first.** Phase 0 of the plan measures which boundary cues are actually detectable on the target
  fronts (RGB seams and handles, ZED depth steps, LiDAR) before any likelihood is designed. If only the prior is
  informative, the partition stays a prior and the card reports a low P(resolve). That is the honest outcome,
  not a failure.
- **Unit promotion as a model term.**
  - A unit's existence log-odds is its partition posterior mass: both of its boundaries located within their
    σ, integrated from the partition posterior.
  - The unit is promoted and retired through the same `rc::exist` decision boundary and debounce as every
    other concept. There is no extra cutoff.
- **Member changes.**
  - If a new seam splits a promoted unit, that unit is retired and two are born: a mass split is an UPDATE of
    the collective, and a birth or death only of members.
  - Chat referents such as "the second from the left" are computed at query time from the ordering along the
    run, so they stay meaningful across splits.
- **One writer, one cycle.** cabinet_concept writes the collective, its members and the occupied-volume boxes in
  the same cycle, from the same state. The boxes are derived and never updated separately.
- **Graph attributes.**
  - The model and its tests are built and pass offline first.
  - Publishing waits on the cortex `REGISTER_TYPE` reinstall. The plan orders that request early, so it never
    blocks the offline work.
- **Perception card.**
  - cabinet_concept publishes the card's *parameters* on its agent node.
  - The shared evaluator in `common/`, linked by chat_agent's `can_see`, computes the three numbers. This is
    the perception-card design.
  - This design only specifies the collective's card fields.
- **Terminators for "end observed".**
  - a room polygon corner, from room_concept's published polygon;
  - an adjacent concrete object (the fridge or hood node);
  - a free-space gap, from LiDAR traversal.
  - The neighbours' own nodes bound the run. That is also the test for the "eating its neighbours" anti-goal.
- **Islands.** Whatever `update_island` does today must keep working. The plan's first task records its current
  behaviour as a regression test.
- **Test data.**
  - Synthetic runs with planted seams.
  - The apartamento kitchen in Webots as the replay source. Recording one tour is in scope.
- **§6.1.** Reconciling it to the hierarchy view is part of this work, its first documentation task.

## 8. Open questions for the user

0. **The user, 2026-10-31:** cabinet_concept is left as it is until this aggregate-objects work starts. The
   perception-cards work only gives it a card describing what it runs today; it does not change it.
1. What exactly is "not good enough" in the current cabinet_concept implementation? The answer decides whether
   this is a change on top of the kitchen-of-runs model or a replacement.
2. ~~Can one collective span walls?~~ **Answered (user, 2026-10-31): yes, including L shapes and U-shaped
   counters with an aisle.** This led to the three-level structure in §3.0, which the user agreed matches.
