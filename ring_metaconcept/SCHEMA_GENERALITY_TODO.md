# ring_metaconcept — schema generality: limitations and the path out

Status: **TODO / design note**, written 2026-07-27 after M0–M4 landed (commit `300181c`).
The agent works and is live-validated on a dining set; this records what it *cannot* yet do and
the cheapest correct route to fixing it. Companion to `../DINING_SET_RIG_PLAN.md` (original design)
and `CONCEPT_AGENT_RECIPE.md` §6 (the meta-concept variant of the generation contract).

---

## 0. The short version

The agent is named after the **arrangement schema** (`ring`), not the concrete concept, and that was
the right call — `dining_set` is only the instance type it happens to birth. Roughly 80 % of it is
schema-agnostic. But four things are hard-coded, and one of them (**a single global rig instance**) is
a correctness bug rather than a missing feature.

The single highest-value change is **cluster members before fitting**. It is not just one more
feature: it simultaneously fixes the multi-rig bug, removes the anchor/ring class hard-coding, and
turns "which schema is this?" into model selection instead of a config choice. Do that before
adding rectangular support or any new schema.

Separately from all of that, §2.5 records a gap of a different kind: the belief is **incomplete**, not
merely narrow. It predicts member yaw but never scores that prediction, so a chair that deliberately
faces outward produces no surprise at all.

---

## 1. What is ALREADY general (config only, no code)

A `coffee_cluster` (coffee table + armchairs) or a `meeting_set` is a config file today:

```toml
[RingMetaconcept]
AnchorClass = "table"        RingClass = "chair"
NodeSubtype = "dining_set"   NodePrefix = "dining_set_"
MemberYawOffsetDeg = -90.0   SigmaSlotM = 0.20   OccupancyQ = 0.70
FacingModelStdDeg = 12.0     EvidenceEmaAlpha = 0.05
```

Nothing in `RingBelief` mentions dining. `θ = [cx, cy, yaw, radius]`, the slot residual is a distance
to slot *k*, and the anchor enters as a centre prior via `accumulate_extra`. All schema, no furniture.

Also fully reusable across schemas (this is the *chassis*, and it is the expensive part):

- graph-reader front end, main-thread polling, **no** `update_node`/`update_edge` signals
- each peer's published `rt_covariance` used directly as the measurement noise
- cavity leave-one-out (`compute_cavity_priors`)
- the non-RT `group_member` edge + payload + 1 s liveness heartbeat
- evidence vs. an explicit `none` null; occupancy × visibility; seat exclusivity
- the EMA on `log_odds` (redundant re-observation must not manufacture confidence)

---

## 2. What is HARD-CODED

### 2.1 ★ One rig instance, globally — *this is a bug, not a gap*

`RingSceneGraph::rig_node_id_` is a single id (`ring_scene_graph.h:79`), and `SpecificWorker::step_ring_belief`
fits **one** ring to **all** members of `ring_class` in the room. Two seating groups in one room →
a single fit spanning both, which the evidence will happily accept because nothing proposes the
alternative partition. Any apartment with a dining set *and* a coffee table with armchairs triggers
this the moment both use `table`/`chair` classes.

### 2.2 One anchor class + one ring class

`RingConfig::anchor_class` / `ring_class` are single strings (`ring_config.h:31-32`), consumed at
`specificworker.cpp:371` and `:461`. A set with mixed members (chairs *and* a bench) cannot be
expressed, and `member_yaw_offset` is a single global value that would be wrong for a second class —
the offset is a property of the member's own model, not of the rig.

### 2.3 Facing-inward is baked in

`RingBelief::slot_facing_yaw` = `slot_inward_direction` + `member_yaw_offset` (`ring_belief.h:186-190`),
and `facing_yaw_for()` likewise points at the centre. Correct for seating; wrong for a display
arrangement (members facing outward) or a queue (tangential). One function, but not parameterised.

### 2.4 Circular only

`DINING_SET_RIG_PLAN.md` specified `type ∈ {circular, rectangular, none}`; only circular + null was
built. **A rectangular dining table with chairs along its sides will fit badly** — the ring model puts
them on a circle and absorbs the mismatch in the residual. For real dining rooms this is the most
visible gap after 2.1.

Minor, related: `kSlotHypotheses{3,4,6,8}` (`ring_belief.h:161`) omits 5 and 7, and deliberately
excludes 2 — a facing-pair is a different schema, and admitting N=2 would let the ring claim every
sofa-and-armchair in the room.

### 2.5 ★★ The belief predicts member yaw but never SCORES it — one-way inference

Different in kind from §2.1–2.4: those are things the model cannot *express*; this is a claim the
model *makes* and never checks.

**Verified in code.** `SpecificWorker::build_ring_frame` packs only `m.xy` and a scalar `R` derived
from `Sigma_xy`. `RingMember::yaw` / `var_yaw` are read in `read_member` (`specificworker.cpp:410`,
`:419`) and then used ONLY by the diagnostic CSVs (`:779`, `:782`). They never reach the belief. So
`mixture_nll`, `log_evidence`, `instant_log_odds` and hence `log_odds` / `p_ring` are functions of
member **positions alone**.

**Consequence.** Take a chair deliberately turned to face outward, correctly estimated as such by
`chair_concept`, sitting at a perfectly good slot position. The rig registers **zero surprise**:
`log_odds` stays at its clamp, `p_ring` ≈ 1, and it keeps publishing an inward-facing prior to that
chair every cycle. The chair resists — its backrest evidence saturates at 6 nats while κ is capped at
1.5 — so the *outcome* is correct, but for the wrong reason: the cap is doing the work and the model
never learns that an exception exists. It is a standing disagreement neither side can resolve.

It is plainly visible to a human (`facing_resid_deg ≈ 180` in `ring_members.csv`, `correction_deg ≈ 180`
in `ring_fit.csv`) and invisible to the belief.

**Why this matters beyond the one case.** The rig pushes yaw DOWN but never takes yaw UP. If `z_rig`
predicts member yaw, then observed yaw is evidence about `z_rig` — and about whether that member
belongs to the rig at all. The `not-a-member` clutter column is exactly the right mechanism for that
and currently competes on position only. With yaw in it, the model can say the thing it should:
*"a chair standing where a seat would be, but not part of the set."* A rig with one deliberately
outward chair should then converge to **"a 3-seat ring with 2 members and one non-member"**, not to
"a slightly bad 3-seat ring" — a strictly better description, and precisely what the `none` null and
the clutter column were put there to express.

**Sketch of the fix.** Add a von-Mises term on `(member_yaw − predicted_facing)` to
`RingBelief::mixture_unnorm`, weighted by the member's own published `var_yaw` (already read, already
thrown away). An uncertain member's yaw then correctly counts for little, with no gate.

★**The hazard, and why this is not a ten-line change.** Scoring yaw closes a FEEDBACK LOOP: the rig
tells a chair to face inward, the chair complies a little, the rig reads that as confirmation and
grows more confident. That is exactly the self-confirmation the cavity discipline exists to prevent,
and the cavity today excludes only member *i*'s **position** (`compute_cavity_priors` rebuilds the
frame without `points[i]`). Closing it properly needs either the member to publish a **prior-free**
yaw, or the rig to down-date its own outgoing message before scoring the reply — leave-one-out applied
to the *message*, not just to the member. Do not add the yaw term without one of those.

---

## 3. The deeper limitation

The rig must be **told** which classes to look for. A general version would **discover** the
grouping — which objects participate in any arrangement at all — rather than being handed
`anchor_class` / `ring_class`. Everything in §2 is a symptom of that one design choice.

---

## 4. Path to a solution, in priority order

### Step 1 — cluster members, then fit one rig per cluster  ★ do this first

Replace "all members of the ring class" with "each spatial cluster of candidate members".

- Gather every `object` node with a pose and published Σ — **no class filter**.
- Cluster in room xy (a mutual-nearest / single-link pass at furniture scale is enough; N is tiny).
- Run the existing `RingBelief` per cluster; keep the clusters whose `log_odds > 0`.
- `RingSceneGraph` grows from one `rig_node_id_` to a map keyed by cluster identity, with the
  existing `retain_members` logic generalised to per-rig membership.

This alone fixes §2.1 and §2.2, and makes §2.4 a natural extension rather than a rewrite. It is also
the honest AI framing: the partition is part of the latent, and today it is silently fixed to
"everything".

★Hazard to design for: cluster identity must be **stable across cycles**, or the rig node churns
(birth/death every time a member flickers). Reuse the multi-instance tracker discipline — associate
this cycle's clusters to existing rigs by overlap of member sets before birthing anything.

### Step 2 — arrangement type as model selection

With clusters in hand, `{ring, row, none}` becomes exactly the comparison `resolve_slot_count`
already performs over N, one level up. Add `rectangular` (or better, `row`, which subsumes the
"chairs along one table edge" case) and pick by the same clutter-inclusive evidence.

★Note the model-selection *initialiser* is schema-specific, not just the slot function: the closed-form
phase seed (circular mean of bearings folded into `[0, 2π/N)`) that fixed the N=3-vs-N=4 failure is
meaningless for a row. Each schema authors its own basin-finder.

### Step 3 — parameterise the facing law

Make the member-facing prediction a property of the schema + member class:
`inward` (seating) | `outward` (display) | `along` (queue/run) | `none` (no yaw prior).
Move `member_yaw_offset` from a single rig-level value to per-member-class, since it belongs to the
member's own model.

### Step 4 — a second schema, to validate the abstraction

Author only when a concrete second case exists (recipe §6 warns against building an any-schema engine
speculatively). Candidates, all reusing the chassis and replacing only the slot function:

- **row / linear** — `slot_k = origin + k·spacing·dir`, state `[x, y, θ, spacing]`. Books on a shelf, a
  run of cabinets. ★`cabinet_concept` already implements a wall-anchored RUN model independently —
  that *is* this schema, built separately. Unifying them is the real test of the abstraction, and
  probably the best second case.
- **facing-pair** — sofa ↔ TV, bed + nightstands. Needs N=2, currently excluded on purpose.
- **grid / workspace** — desk + monitor + keyboard. A 2-D lattice; different slot function entirely.

---

## 5. What NOT to do

- Do not add rectangular support before Step 1. It would be a second special case bolted onto a
  single-instance fitter, and Step 1 changes the shape of the code it would live in.
- Do not generalise speculatively to an "any schema" engine. Author `row` only when unifying with
  `cabinet_concept` is actually on the table.
- Do not move the class filter into the belief. Which objects are candidates is a *front-end*
  question (`poll_members`); the belief should only ever see poses + Σ.
- Do not add the §2.5 yaw likelihood without first solving the message down-date. A rig that scores
  the yaw it just asked for will manufacture confidence, and it will look like the model working.

---

## 6. Open questions

- Should the rig own the partition, or should a separate agent publish "these objects are grouped"
  and let per-schema agents compete over it? The latter is cleaner but adds a graph hop.
- `cabinet_concept`'s run model and this ring share the chassis idea but were written independently.
  Is the right end state one `arrangement_metaconcept` with pluggable slot functions, or several
  sibling agents? Unifying is elegant; it also couples two working agents.
- The occupancy term needs per-slot visibility, which needs a camera frustum. For a row or a grid the
  "slot" geometry differs — is `slot_visibility` still the right interface, or should the schema
  expose predicted-visible *area* per slot (which is also what the missing free-space term wants —
  see `[[chair-template-world-mismatch]]`)?
