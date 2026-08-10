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

Then: state σ\* from a real consumer or declare its absence; make every channel a symmetric ratio; make
every accumulator answer "what makes this observation new?"; give the planner the room polygon, the
adequacy bound, and whatever novelty term the belief itself uses; and check that every gate you write
can actually fire.
