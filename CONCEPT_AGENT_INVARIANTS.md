# Concept-agent invariants — the core every new agent must satisfy

Derived 2026-08-09 from a deep review of the six object-concept agents (refrigerator, table, chair,
cabinet, door, bottle; room and human excluded). It exists because a single day produced **five
independent instances of two mistakes**, in five different files, written by people who had each
already documented the lesson somewhere else in the tree.

`CONCEPT_AGENT_RECIPE.md` says how to BUILD an agent. This says what must be TRUE of one. Where they
disagree, this wins — every clause below is a bug that was actually shipped, not a preference.
`tools/concept_audit.sh` checks the mechanical subset; the rest needs reading.

---

## The two mistakes, stated once

Everything found today is one of these.

### I. Evidence must be counted ONCE

A statistic that accumulates a quantity **re-derived from unchanged data** measures dwell time, not
information. Confidence then grows by staring, saturates at its clamp, and can no longer be recanted
in bounded time.

| where | status |
|---|---|
| `ExistenceBelief` — per-cycle absence across correlated frames | **was** summed; now `ρ_eff = ρ·(1−p_detect)` |
| `TableFitter::evaluate_shape` — log-Bayes-factor over the *accumulated* voxel bank | **was** summed; now low-passed (`ShapeEvidenceEmaAlpha`) |
| `ChairBelief::flip_acc_` — 4-mode orientation evidence | ✅ correct: weighted by per-bearing `novelty` |
| `RefrigeratorBelief::front_acc_` — door-facing modes | ⚠ **corrected 08-11**: a per-observer-bearing budget (24×15° bins, `front_view_budget_` 3.0) WAS added — same discipline as chair. What remains is the FALLBACK: `view_bearing_rad` NaN ⇒ full credit (see below) |
| `RingBelief` — rig-vs-null log-odds | ✅ correct: `evidence_ema_alpha`, and its comment says why |
| chair's *planner* — proposed a bearing whose novelty budget was already spent | fixed: gain now carries `mode_entropy · view_novelty` |

**The rule.** Any `x += f(observation)` must answer: *what makes the k-th observation different from the
first?* Legitimate answers are a **novelty/decorrelation weight** (chair, ring) or a **low-pass** (table
shape). "It is a new frame" is not an answer — a static object re-observed from an unchanged pose is
the same observation.

**Corollary — the planner is part of this.** If a belief refuses evidence from an exhausted viewpoint,
the epistemic planner must know, or it will keep proposing that viewpoint: a completed visit that
resolves nothing, forever, and an object that is re-selected while everything else starves.

### II. Every channel must be a likelihood RATIO, and every gate must fail to HOLD

A channel that can only push one way is a ratchet, not evidence. A gate whose "cannot resolve" branch
leaves a non-zero default is the same thing wearing a gate's clothes.

- **The occupancy-only floor.** `delta = d_conf + p_detect·(d_full − d_conf)` was written in four
  places, each commented "p_detect → 0 ⇒ pure HOLD". It is not: it leaves `+d_conf`, an occupancy-only
  likelihood, which is always positive. Live cost: a phantom refrigerator standing inside the dining
  set, `L = +4.00` pinned, `lfree = 483` against `locc = 206`, unremovable while stared at. Correct form
  is `delta = p_detect · d_full` — a probe that could not resolve the object is not evidence either way.
- **The bounded-total trap.** Saturating an unbroken run at `1/ρ` observations makes the *cumulative*
  absence evidence bounded. At ρ = 0.327 that is 4.6 nats against the 8 needed to cross from the +4
  ceiling to the −4 floor, i.e. removal became arithmetically impossible. Decorrelate by **what changed
  in the world** (viewpoint, conditions), never by run length.
- **A gate must be reachable.** `ai2_trunc_gate_frac = 0.10` is tested against `mask_trunc_frac`, which
  is a *border-contact* ratio (`on_border_pixels / n_mask_pixels`) and cannot approach 0.10 for any blob
  wider than ~40 px. Measured over live logs: table max 0.024, refrigerator 0.044, door 0.124 (2 rows
  of 5621). **The truncation gate is effectively dead in all six agents** and has never protected
  anything. Either measure truncation as an AREA fraction, or delete the gate — do not leave a
  protection that has never fired.

---

## The invariants

An agent is coherent when all of these hold. Each is annotated with what breaks when it does not.

**BELIEF**

1. Σ must be able to grow *and* shrink. If it only ages, the object becomes permanently uncertain and
   permanently attractive; if it only shrinks, a moved object is never recovered.
2. The REPORTED covariance must include discrete-mode entropy where the model has modes, and the
   planner must then optimise that same quantity. Chair violated the second half: it scored the
   continuous Σ (within-mode σ_yaw ≈ 0.049) while the real uncertainty was the 4-way mode
   (`std_yaw_rep` ≈ 0.64), so no visit could ever reduce what was displayed.
3. Every accumulator obeys mistake I.

**EXISTENCE**

4. One policy: `ΔL = p_vis · log[P(outcome|exists)/P(outcome|¬exists)]`, symmetric, and `p_vis → 0 ⇒
   ΔL = 0`. No channel may be occupancy-only, at any confidence.
5. Removal is a decision on L, never a miss counter, and L must be able to reach the floor from the
   ceiling under sustained absence. Check the arithmetic: `max_absence_per_run ≥ 2·L_max`.
6. Absence is weighted by the detector's own model (`p_detect`), and that model is per-object and
   MEASURED, not the fleet prior.

**PERCEPTION / NBV**

7. The planner knows the reachable region (`room_polygon`) and REFUSES when no viewpoint is usable.
   Publishing an unroutable pose is worse than publishing none: the controller repairs rather than
   rejects, snapping the goal onto the object.
8. The stand-off is the argmax of the detector envelope, and the envelope is config-driven per object.
9. Expected gain is bounded by the **adequacy gap** so an object stops attracting once it meets the
   consumer's precision. ❌ chair, door and bottle have no such bound — they can never stop asking.
10. A viewpoint's value accounts for evidence already collected from that bearing (corollary of I).

**CONTRACT**

11. Every `default_contract_for(key)` has a real case. The fallback returns a *valid-looking*
    `Contract::reach()`, so a missing case is invisible and the robot silently arrives facing the wrong
    way. Bit door and cabinet.
12. σ\* is the CONSUMER's tolerance, restated — never invented, never a prior. If no consumer publishes
    one, declare the absence (`SIGMA-STAR: none — <reason>`) rather than leave it looking forgotten.

---

## Conformance, 2026-08-09

| | refrigerator | table | chair | cabinet | door | bottle |
|---|---|---|---|---|---|---|
| room_polygon + refusal (7) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| envelope from config (8) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| gain bounded by adequacy (9) | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| contract case (11) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| σ\* stated or declared (12) | ✅ | ✅ | ✅ | ✅ | decl | decl |
| accumulators counted once (I) | ⚠ fallback | ✅ | ✅ | ✅ | ✅ | ✅ |
| no occupancy-only floor (II) | ✅ fixed | ✅ fixed | n/a | ⚠ unchecked | ✅ fixed | ✅ |
| truncation gate reachable (II) | ❌ dead | ❌ dead | ❌ dead | ❌ dead | ❌ dead | n/a |
| existence channel exists (4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 08-10 |
| removal on L, ONE authority (5) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| debounce counts LOOKS (5) | ✅ 08-10 | ✅ | ❌ none | ✅ 08-10 | ✅ 08-10 | ✅ |

**✅ two authorities — FIXED 2026-08-10** (`2f4c4e0`). Each agent's death counter is now tied to its own
existence flag, so the two are mutually exclusive by construction and still A/B-able. ★cabinet was not the
case it looked like: its existence removal is OFF by default, so the counter was its ONLY authority and a
blind `INT_MAX` would have made every cabinet immortal. The structural grep could not see that; reading the
call site could.

**★ III. A DEBOUNCE MUST COUNT LOOKS, NOT CYCLES** (found 2026-08-10, third member of the family above).
`if (should_remove(L)) ++streak` advances on every cycle once L is below the boundary — including the ones
where `p_detect` was 0 and the channel had just, correctly, HELD. The object is then condemned by evidence
gathered once and executed `RemoveFrames` cycles later, while the robot looks elsewhere. Live proof:
**all 12 door deaths had `fixated = 0`**, at ranges 2.1–6.7 m, five of them at `p_detect = 0.000`.

Two traps inside it, both of which cost a wrong "fixed" claim if missed:
- **`if (integrated)` is NOT this guard.** refrigerator and cabinet both claim to count "EVIDENCE cycles,
  not wall-clock" — but a channel that ran and resolved nothing still sets `integrated`. It answers *did a
  sensor fire?*, not *could it have seen the object?*
- **Check for a SECOND, WEAKER streak on the same L.** refrigerator's `plaus_remove_streak` tested the same
  boundary but advanced unconditionally, so it always fired first; fixing the sensor streak alone would
  have changed nothing observable.

A PRIOR is exempt and should stay at full weight — door's out-of-room and minimum-height branches are
categorical facts about where a door can be and what a door IS, not sensor absence. Only the SENSOR channel
owes a look.

**❌ chair has NO debounce at all** — a single cycle below `exist_remove_logodds` removes. It is the only
agent without one. Left alone deliberately: it slows removal, and the open chair complaint is a phantom
that would NOT die.

chair also states its boundary in NATS (`exist_logodds < exist_remove_logodds`) where everyone else states
a PROBABILITY (`should_remove(existence_removal_prob)`). Same decision, two vocabularies — the cheap kind of
divergence to remove while touching that code.

**★UNKNOWN NOVELTY MUST NOT MEAN FULL CREDIT** (refined 2026-08-11). `front_acc_` does carry a novelty
budget now, but only when the cue knows where the look came FROM; `view_bearing_rad` defaults to NaN and
the NaN branch keeps the legacy full-weight sum. That is fail-OPEN, and it is the same shape as invariant
II's "a gate must fail to HOLD": if you cannot tell a new view from a repeated one, the honest charge is
the one you would make for a repeat, not the one for a discovery. Narrow in practice — the single cue
producer sets the bearing whenever `room_T_zed_matrix` resolves — but it fails exactly when the pose chain
is unavailable, i.e. when registration is least trustworthy. Second, related: that matrix is queried at
**ts = 0** (latest) while the cue carries its own `rgb_stamp_ms_`, so a look taken while moving is filed
under the wrong bin — the novelty bookkeeping itself is mis-registered. Pin it to the capture stamp.

## ⚑ TO REVIEW — raised 2026-08-11 while closing bottle_concept, deliberately not acted on

**1. Attention is allocated by who is most UNCERTAIN, not by who MATTERS.** `piso.wbt` has no arm
(`DEF shadow Shadow`; Shadow.proto contains no Kinova), so nothing in that scene consumes a bottle's pose —
yet bottle_concept still publishes an epistemic affordance whose gain competes against door, table and
refrigerator for the robot's time. That gain is computed purely from its own Σ and never from a demand. In
AI2 terms expected information gain only has value relative to a PREFERENCE: uncertainty about something
nobody needs is not surprise worth resolving. An object with no live consumer should bid ~0 and be passive —
tracked when seen, never asking to be visited.

This is the ROOT of the "never stops asking" symptom logged against chair, door and bottle (invariant 9).
Those were treated as missing adequacy bounds; the bound is only half of it, because a satisfied bound still
leaves an object bidding for looks that serve no one. Fleet-wide design change, not an agent edit — every
`epistemic_planner` would need to know whether any consumer is present, which the affordance protocol can
already express (`ViewpointConstraint::sigma_star` flows agent→controller; nothing flows the other way).

**2. bottle's σ\* = 0.0129 m is PROVISIONAL and carries a falsifiable test.** Set from the gripper clearance
at 2σ (`c/2`) rather than 3σ specifically so the gap can reach zero — σ_cx has plateaued at 0.0212 m over
2400 rows, and against `c/3` the demand would be unmeetable, pinning the epistemic gain at a positive floor
forever (i.e. manufacturing defect 1 above). The plateau was measured at ~1.8 m with no approach ever made,
while `RangeNearM` is 0.6. **Read σ_cx at the end of one completed epistemic approach:** reaching ~0.013
confirms the demand and the affordance gains a "done"; plateauing above proves the 1.8 m figure is a floor,
and σ\* must then be re-derived from what the sensor can actually deliver rather than from what the gripper
would like. Either outcome is informative — do not leave it unrun.

**3. `p_resolve = r/(r + σ_surf)` belongs in the shared carve, not in bottle.** A sweep cannot judge the
existence of an object it localises worse than that object's own size. It reads ≈1 for a fridge or a table,
which is why it never surfaced until a 5 cm object appeared — but the fleet is heading toward smaller things
(a mug, a can), and six copies of the omission is how the last four defects propagated.

**Open, in priority order:** the truncation gate is dead fleet-wide (measure area or delete);
the `front_acc_` NaN fallback above; chair/door/bottle never stop attracting; the three double authorities above;
cabinet's envelope keys and its occupancy floor are unchecked; cabinet publishes no affordances at all
under the kitchen model; bottle's detector envelope is the fleet prior and caps `p_detect` at 0.36
(≈3× slower removal — measured, see `bottle_existence.h`).

---

## For a NEW agent

The belief unit is **not** necessarily an instance — cabinet's is a `(wall, tier)` cell. Anything shared
that assumes "one row / one card / one affordance per instance" breaks there. Write the core against a
*belief unit*, and let the agent say what one is.

★**That is a FAMILY boundary, not an exception** (settled 2026-08-16). There are two families — INSTANCE
(bottle · chair · door · hood · refrigerator · table) and RUN (cabinet, and `shelf` next) — over one core.
So the question for anything you are tempted to share has a sharp form: **does it depend on the belief
UNIT?** A covariance, a footprint, a z-band, a point cloud or a sample does not — that is core, and both
families use it. Iterating the beliefs does — that is family-level. A `shelf` must be generated from
cabinet, not from a box agent. See `CONCEPT_AGENT_RECIPE.md` §"TWO FAMILIES, ONE CORE".

★**And the priors are only as real as the manifest that is READ.** `provenance_ok()` is armed in all 8
agents (an `inherited` block refuses to start), but `load_geometry`/`z_span`/`resolve`/`band_contains_body`
are called in **hood alone** — measured 2026-08-17 — and refrigerator and table declare no
`[model.geometry]` block at all. A validated declaration nobody reads guards nothing; the four traps it
exists to catch are guarded in one agent.

Then: state σ\* from a real consumer or declare its absence; make every channel a symmetric ratio; make
every accumulator answer "what makes this observation new?"; give the planner the room polygon, the
adequacy bound, and whatever novelty term the belief itself uses; and check that every gate you write
can actually fire.
