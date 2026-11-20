# The Vygotskian machine — language over the belief hierarchy (culture, sequencing and imagination)

*The Vygotskian machine*: a robot whose higher functions arrive from outside — from the person and from the VLM's
culture — and are internalised into an inner voice, then into priors, counts and habits that need no language.
This note is its engineering design; the paper sections are in `paper/` next to it: `vygotskian_machine.tex`
(framing), `vygotskian_internalisation.tex` and `vygotskian_cultural_plane.tex`.

Status: **DESIGN** (2026-11-19). Nothing built. Drafted on `pbustos/active_inference` (commit `7e10e30`) and
moved here, `vygotskian_machine/`, on 2026-11-20. Companions: `CONCEPT_AGENT_RECIPE.md` §6 (the metaconcept chassis),
`kitchen_metaconcept/etc/config.toml` (authored class priors), `controller/src/exploration_drive.h` (the automatic
explorer), `language_agent/` (the VLM, its mission planner and its `imagine*` tools), `language_agent/docs/
execution_contracts.md`, and the paper draft `vygotskian_internalisation.tex` (Eqs. 5–7).

Paths are relative to `robocomp/active_inference` unless stated.

---

## 0. The one idea

The robot already builds its belief state automatically, and the build is hierarchical: concept agents fit objects
(level 1), metaconcepts fit arrangements over those beliefs and push cavity-corrected priors back down (level 2), and
the exploration drive moves the robot so that the hierarchy fills in. That is the driving on a familiar road. It
needs no language.

The VLM extends it in **three roles**, each over the same hierarchy and each under the same rule — **the VLM
proposes, the models dispose**:

| Role | What the VLM supplies | What decides | Part |
|---|---|---|---|
| **Culture** | priors and structure the hierarchy cannot author for itself: what a room is *for*, which classes go together, which schema an arrangement follows | evidence against a null (chassis rules) | A |
| **Sequencing** | goals decomposed into missions over the hierarchy's beliefs, under contextual constraints, with the belief changes each step should produce | `check_planned`, `CommandGate`, the controller, then the belief dynamics | B |
| **Imagination** | *which* subjunctive questions to ask the generative models while the automatic level runs well, and what to do with the answers | the robot's own models, run on an imagined copy (`imagine*`) | C |

Which role is active at a given moment is set by the demand signal and the three modes of the paper draft (Eq. 7):
**focused** (something diverged: diagnose and replan), **monitoring** (task running, voice silent), **wandering**
(demand low for a while: imagine). Part D puts them together.

---

## 1. What already exists (do not rebuild)

| Mechanism | Where | Used by |
|---|---|---|
| Graph-reader front end, main-thread polling, no `update_node`/`update_edge` signals | recipe §6 Delta 1; both metaconcepts | A |
| Measurements = peers' posteriors with their own published Σ / existence | `ARCHITECTURE.md` §9 | A |
| Existence by log Bayes factor vs an explicit null; persistent-evidence birth | recipe §6 "Detection without a threshold" | A |
| Cavity / leave-one-out; EP message down-date | `ring_metaconcept` `compute_cavity_priors`; `kitchen_belief.h` `down_date` | A, B, C |
| Class-membership log-odds, flagged AUTHORED, to be replaced by corpus counts | `kitchen_metaconcept/etc/config.toml` lines 56–82 | A |
| `room_explored_p`: P(continuing to explore no longer pays) | `room_concept`; read by `controller/src/mission_explored.h` | A, B |
| Exploration drive: explore until promoted, cross nearest uncrossed door, depth-first return | `controller/src/exploration_drive.h` | B, C |
| Per-instance epistemic cycle / NBV | `common/epistemic_step`, each agent's `EpistemicPlanner` | A, C |
| Mission language, planner, checker, card, gate, interpreter, progress snapshots | `language_agent/etc/skills/missions.md`, `mission_plan.h`, `command_gate.h`, `common/mission/` | B |
| Robot events waking the model; tools restricted in those turns (`allowed_in_wake`) | `mission_plan.h` | B, D |
| An imagined copy of the room that interventions edit and that can never write the graph | `language_agent/src/imagined_world.h` | C |
| Subjunctive fit on the controller's own fit model | `imagine_fit.h` | C |
| Subjunctive visibility with the perception cards | `imagine_see.h` (over `can_see.h`) | C |
| Sampling a concept's generative prior, rendering through ZED + LiDAR models, Monte-Carlo P(detect) | `imagine_view.h` | C |
| Success odds of a route from today's crossings (Beta), LTSM prior only when today has none | `imagine_succeed.h` | C |
| Person-confirmed claims, posted only after a yes | `human_claim.h`, `ltom_agent` | A |

The imagination tools are, today, **answers to a person's questions**. Part C makes them the robot's own.

---

# Part A — Culture: priors and structure

## A.1 Five uses, simplest first

1. **Class priors for existing metaconcepts.** `kitchen_metaconcept`'s `MemberClassLogOdds` are hand-elicited
   (`cabinet 0.2, refrigerator 2.0, counter 2.5, hood 3.0, microwave 1.5, chest of drawers −0.5`); the config says
   they "should be replaced by counts from a labelled scene corpus when one exists". The VLM is that corpus.
2. **Schema proposals.** A new typed arrangement on an existing schema is already a config file (`coffee_cluster`,
   `meeting_set`). The VLM proposes `(schema, anchor class, member class, yaw offset, slot σ)`; the chassis accepts
   or ignores it by evidence.
3. **Level-3 function** — `function_metaconcept`, A.2.
4. **Model-level surprise → structural proposals** (A.3).
5. **Open-vocabulary evidence** for classes without a producer (`counter`, `hood`, `microwave` have no node, per
   the kitchen config): VLM `look()` as a low-precision measurement. Riskiest; last.

## A.2 `function_metaconcept`

**Latent.** For each room `r` and function `f ∈ F` (`kitchen, dining, living, bedroom, bathroom, office,
corridor`): an independent hypothesis "`r` serves `f`" against the null "`r` is a generic space". Not a categorical:
open-plan rooms serve several functions, and one hypothesis per function is how the chassis already treats instances.

**Evidence.** Members are level-1 objects (class `c`, existence `p_c = σ(L_c)`) and level-2 arrangements. Presence,
not count, is the observation (ten chairs must not argue ten times — the log-odds-EMA discipline). With
`π_cf = P(c present | f)`, `π_c0` for the null, and `d_c` the probability that `c` would have been detected in `r`
had it been there:

```
Λ_f(r) = Σ_c  p_c · log( π_cf / π_c0 )
            + (1 − p_c) · log( (1 − π_cf · d_c) / (1 − π_c0 · d_c) )
```

The second term is expected-but-absent evidence, only as strong as the search was (`d_c = 0` ⇒ no evidence) — the
negative-information rule of instance death and residual carving, lifted to the scene. `d_c` starts as
`room_explored_p(r) · δ_c` (per-class detector rate); the detectability model replaces it when it lands. The
soft-presence expectation is an approximation, exact under hard presence. `Λ_f(r)` is the existence log-odds; birth
and death follow the chassis. No threshold.

**Priors: VLM pseudo-counts, then experience.**

```
π_cf ~ Beta( κ·β_f · π_cf^VLM ,  κ·β_f · (1 − π_cf^VLM) )
π̂_cf = ( κ·β_f·π_cf^VLM + n_cf ) / ( κ·β_f + n_f )
```

`κ` is small (the VLM is worth a few rooms, not a hundred); `β_f ∈ [0,1]` from measured calibration (A.4);
`n_f`, `n_cf` from rooms confirmed as `f` (by a person, after their yes, or by a saturated posterior over repeated
visits). This is the conjugate, implementable form of Eq. 5 in the paper draft.

**Messages down, phased.** Nothing into member beliefs at first. Then exploration targets (expected unseen classes
with coarse placement) as **viewpoint value** for the drive and the epistemic planners — selection bias only, kept
honest because absence counts in exactly the places searched. Later, if earned, birth seeds to level 2 with the
`down_date` accounting, since a seeded member's existence would contain our message. **The cavity rule binds the VLM
too**: "kitchen" must not confirm itself through the places it sent the robot to look.

**Node and owner.** `function_<room>` (metaconcept type: `function_labels`, `function_log_bf`, coverage), owned by a
**separate pure-C++ agent** — the VLM is not in the runtime loop and the agent keeps running when it is down. The
prior table is an `etc/function_priors.toml` written by an offline elicitation tool and reviewed like any authored
prior, with provenance. New attributes need `REGISTER_TYPE` lines: ask Pablo.

**Norms attached to functions.** Each function also carries the contextual norms the VLM supplies for it
(bedroom at night: quiet, slow; kitchen in use: keep out of the work area; any doorway: do not block). They are not
used by this agent; Part B reads them as planning constraints.

## A.3 Model-level surprise

Levels 1–2 already compute the signals that the **model** is wrong, not merely uncertain — inputs to the demand
signal and to focused-mode diagnosis:

| Signal | Source | What the focused voice proposes |
|---|---|---|
| Arrangement `log_bf` collapsing | ring / kitchen | wrong schema; moved; ask |
| A member with a large cavity residual | `compute_cavity_priors` / `down_date` | not a member; moved; misclassified |
| A class persistently near zero membership | kitchen membership log | **missing structure** — the live fridge read 0.00 until a `Tall` tier existed; a person diagnosed it |
| Two schemas both weak | `ring_partition` model selection | a schema not yet in the library |
| High expected-but-absent mass in an explored room | `function_metaconcept` | wrong function; an occluded area |
| Member yaw contradicting the arrangement | **not scored yet** (`SCHEMA_GENERALITY_TODO.md` §2.5) | only culture explains it (someone sitting, cleaning) |

Structural proposals are **text for a person to review**; self-modification of configs is out of scope.

## A.4 Eliciting priors

- **Closed vocabulary**: the graph's class names (`object_class_of`) and `F`; the VLM is asked probabilities about
  named things, never to name.
- One question per (class, function) and per class for the null; temperature 0; three paraphrases; keep the median,
  record the spread.
- **Clamp** log-odds: a strong prior makes a badly fitting member argue hard against the whole hypothesis (the
  kitchen config: clutter floor from −1.77 nats at +0.2 to −3.91 at +2.5; the first live run pinned at −8).
- **Calibration** (`β_f`) from rooms of known function: the Webots apartment first, person-confirmed rooms later;
  Brier per function. Expect cultural bias; counts correct it.
- Provenance in the file: model tag, date, prompt hash, per-entry spread.

---

# Part B — Sequencing over the belief hierarchy

## B.1 Plans are written over beliefs, at the level of the goal

Mission predicates already refer to beliefs (`explored(room)>=0.9`, `door_open(d)<0.5`, `seen(chair)`). Levels 2–3
extend the vocabulary in which goals are stated and decomposed — the **deep temporal model**: a goal is planned at
its own level and refined downward.

| Goal stated at | Example | Decomposition |
|---|---|---|
| Level 3 — function | "go where people eat" | room with highest `Λ_dining` → its `dining_set` → a standpoint |
| Level 2 — arrangement | "check every chair at the table" | iterate the ring's members **and its empty slots** (predicted, not yet seen) |
| Level 1 — object | "approach the fridge" | `approach_object` |
| Level 0 — pose | "turn 90°" | `move` |

The hierarchy supplies grounded names; the gate already refuses names that do not exist.

## B.2 Epistemic sequencing: plans that build belief

The exploration drive is the robot's **compiled** epistemic plan (the habit, `E`): explore until promoted, cross the
nearest uncrossed door, return depth-first. It is the right default for "build the map". A specific goal makes it
inefficient, and the VLM writes the epistemic sequence instead. "Find a mug" in a half-known house:

1. go to the room with the highest kitchen evidence (level 3);
2. look along the worktop plane (level 2: the kitchen frame's `worktop` is where mugs rest);
3. `approach` once one is `seen`.

The hierarchy says **where** the information is; the VLM says in **what order** and **when to stop**.

## B.3 Every step predicts its belief outcome

A step today reports done or failed. It should also declare the **belief change it expects**, checked by the
interpreter against the graph:

| Step | Expected change | A divergence means |
|---|---|---|
| `cross door_3 expect=room_4` | current room flips; `room_4` recognised | wrong room; mislocalised |
| `explore $next budget_s=60s` | `room_explored_p` rises; some `Λ_f` leaves 0 | blocked or occluded; an unexpected kind of room |
| `approach chair_12` | chair_12's existence stays high as range closes | phantom; moved |
| `report "the dining set"` | a `dining_set` exists with its slots | the arrangement is not there |

A divergence is a surprise: it raises demand, focused mode replans **from the belief state the robot actually has**
(today it replans only on a failed step, through the wake path).

**Plans and inner questions are the same object.** A plan is a question about the future — *if I do this, will my
beliefs end up like that?* — whose predictions the belief dynamics confirm or refute. One evaluator scores both.

## B.4 Assumptions and context

- **Assumptions** a plan (or a compiled macro) relies on are predicates over the same hierarchy — "valid while
  `door_open(door_3)>=0.5` and `Λ_kitchen(room_2)` stays high". A drop in their probability raises demand **before**
  a step fails: the driver slowing at the school sign, not braking for the child.
- **Contextual constraints** come from level 3: the norms attached to the active functions (A.2) become mission-wide
  invariants — quiet, slow, keep distance, avoid a region. `execution_contracts.md` already defines the pattern
  ("State (maintain)": a monitored invariant with a violation report and a release).

## B.5 Compilation

Sequences that recur and succeed become macros — the habit prior `E` in `p(π) = σ(ln E − γG)`, or the library of
P7(a) in the paper draft. They are naturally indexed by the hierarchy ("check a dining set", "survey a worktop"): a
compiled habit is **a plan schema over a metaconcept schema**, and it carries its assumptions (B.4) so that it knows
when it no longer applies.

## B.6 What the code lacks

1. Mission predicates and references for levels 2–3: `function(r, f)>=p`, iteration over an arrangement's members
   and slots, region references (`ref:region` is listed as unsupported in `common/mission/mission_ir.h:157`, as is
   `speed`).
2. Per-step expected belief outcomes, checked by the interpreter.
3. Assumption predicates on missions and macros, feeding demand.
4. Replanning at decision points, not only after failures.
5. Norm-derived invariants.

---

# Part C — Imagination while the automatic level runs well

## C.1 Principle

Imagining is **running the robot's own generative models offline, on an imagined copy of the world**. The copy
exists (`imagined_world.h`: interventions take the world by value; the live graph cannot be written from there),
and so do the models: the controller's fit model (`imagine_fit`), the perception cards (`imagine_see`), concept
priors rendered through the sensor models with a Monte-Carlo P(detect) (`imagine_view`), and episodic success odds
(`imagine_succeed`). Today a person asks; in wandering mode **the robot asks itself**.

The division of labour is the same as everywhere else: the **VLM chooses what to imagine** — the seed, the
intervention, the question — and **the models compute the answer**. The VLM never answers a subjunctive question
by itself; that would be confabulation, not imagination.

## C.2 Five kinds of imagining, by what they prepare

| Kind | Seed | Question put to the models | Product |
|---|---|---|---|
| **1. Rehearsal** of the current plan | the next steps of the running mission | `imagine_succeed` per leg; `imagine_see`: "from the next standpoint, will I see X?"; `imagine_fit`: "do I fit if door_3 is half open?" | a **contingency cache**: alternatives keyed by the assumption that would trigger them |
| **2. Scene completion** (levels 2–3) | empty slots of an arrangement; expected-but-unseen classes of a function hypothesis | `imagine_view`: sample the class at its predicted place, P(detect) from candidate viewpoints | **viewpoints**, with their P(detect), to confirm or refute the expectation when the drive gets there |
| **3. Counterfactual replay** | a failed or slow episode from episodic memory | `imagine_succeed` with another door; `imagine_fit` with another opening | **lessons as testable hypotheses**, and assumptions for the compiled macros (B.4–B.5) |
| **4. Anomaly pre-analysis** | a residual cluster, an open question closed as "unknown" | `imagine_view` of candidate classes at that place: which explains the footprint and the detection record? | **ranked hypotheses** the next focused episode starts from |
| **5. Structural replay** | metaconcept logs (ring / kitchen CSVs) | refit the logged member posteriors under an alternative schema or tier | **structural proposals** for review — the `Tall` tier is the kind of fix this could have found: replay membership with one more tier and see the evidence rise |

Kinds 1–2 serve the near future, 3–4 the recent past, 5 the model itself. Kinds 1, 2 and 4 use only tools that exist
today; 3 needs episodic queries the tools already make; 5 needs an offline replay harness for the metaconcepts.

## C.3 What to imagine first

Imagination competes for a bounded budget (VLM turns, Monte-Carlo samples, CPU) and is pre-empted at any moment, so
seeds are ranked by the value of their product for **upcoming decisions** — prioritised replay's gain × need
(Mattar & Daw 2018):

- **need**: how soon the situation it prepares will occur — the next legs of the running mission first, then the
  rooms the drive will enter next, then open questions, then old failures;
- **gain**: how much the product could change that decision — legs with uncertain odds, expectations with
  P(detect) near 0.5, hypotheses that are currently tied.

Idle consolidation (kind 5) runs only when nothing ahead has need.

## C.4 Where the products go

| Product | Destination | Weight |
|---|---|---|
| Contingency cache | attached to the running mission, keyed by assumption predicates | used at snap-back: the focused voice **verifies** a prepared alternative instead of composing one |
| Viewpoints with P(detect) | the drive and the epistemic planners, as viewpoint value | bounded; the same channel as A.2's targets; never existence |
| Ranked hypotheses | the open question objects | priors for the next focused episode, tempered by imagination's measured calibration (C.6) |
| Lessons, assumptions | candidate macros (B.5) | tested by later executions before they are trusted |
| Structural proposals | text for a person | none until reviewed |
| The imagined episode itself | an **imagined** store, with provenance | never the live graph, never factual episodic memory (source monitoring) |

## C.5 Gating

Only in **wandering** mode (demand below `θ_lo` for the dwell time). The request in flight is cancelled the instant
demand crosses `θ_hi`; its seed and progress are snapshotted and resumed if still relevant. Extended VLM thinking is
allowed here and nowhere else — there is time, and nothing waits on it (the deployed agent disables thinking
because a thinking model spent 3 000–6 000 tokens, 40–80 s, per tool call).

## C.6 Imagination is itself calibrated

Every product that predicts something — a leg's success, a P(detect), a ranked hypothesis — is checked when reality
arrives, and scored. Imagination earns trust per kind exactly as the VLM's priors do (A.4): a kind whose predictions
keep failing gets less weight, and a model whose imagined outputs diverge from its observed ones has itself become a
model-level surprise (A.3).

---

# Part D — The three roles in the three modes

| Mode | Entered when | Culture (A) | Sequencing (B) | Imagination (C) |
|---|---|---|---|---|
| **Focused** | demand > `θ_hi`: model-level surprise (A.3), step divergence (B.3), assumption drop (B.4), the person speaks | diagnose the model; structural proposal | replan from the actual belief state, **using the contingency cache first** | — |
| **Monitoring** | between thresholds | — | the interpreter checks expected outcomes and assumptions; no VLM calls | — |
| **Wandering** | demand < `θ_lo` for `T` | offline elicitation, schema proposals | prospective drafts for goals ahead | the five kinds, ranked by C.3 |

Demand's inputs are therefore exactly the signals this note introduces: model-level surprise (A.3), expected-outcome
divergence (B.3), and assumption probability (B.4). And the loop closes: imagination (C) prepares what focus (B) will
need, focus produces the lessons imagination replays, and both feed the culture-derived priors (A) with the counts
that gradually replace them.

---

# Part E — Phases and falsifiable checks

Policy against policy in one closed loop wherever it applies.

| Phase | What | Check |
|---|---|---|
| **P0** | Elicit the kitchen's class priors (A.4) | agreement with the authored values; would swapping them change any live membership decision? |
| **P1** | `function_metaconcept` **read-only**, CSV only (the kitchen's milestone-2 pattern) | the functions above the null are the right ones per Webots room; each `Λ_f` leaves 0 as `room_explored_p` rises; **fridge removed in sim**: kitchen weakens but survives |
| **P2** | Mission predicates for levels 2–3 (B.6.1) and function-driven exploration targets | goal-directed search ("find a mug") time and distance vs the drive alone; objects missed in rooms counted as explored |
| **P3** | Per-step expected outcomes and assumptions feeding demand (B.3–B.4) | time from a seeded divergence (door shut mid-mission, object moved) to a validated new plan; fraction caught **before** a step fails |
| **P4** | Wandering imagination, kinds 1–2 (rehearsal, scene completion), tools as they exist | snap-back latency with vs without the contingency cache; views to confirm expected objects with vs without precomputed viewpoints; imagination's calibration per kind |
| **P5** | Kinds 3–5 and structural proposals (A.3, C.2 kind 5) | **seeded structural defects**: re-create the two-tier kitchen — is the `Tall` tier proposed? |
| **P6** | Online priors through a graph node; level-2 birth seeds with down-date; macro compilation | `β_f` improves with confirmed rooms; no self-confirmation (seeded vs unseeded posteriors agree on held-out evidence); VLM calls per hour fall while success holds |

A P1 failure ends Part A before any coupling. A P4 failure (no latency or view gain) means wandering is not worth its
compute and the robot should simply idle.

---

# Part F — Risks

- **Self-confirmation** through selection (A.2, B.2) or through imagined content leaking into belief (C.4) — cavity,
  symmetric `d_c`, down-date, and the imagined store.
- **Confabulated imagination**: the VLM answering a subjunctive question itself instead of asking the models — C.1;
  enforce by making the `imagine*` tools the only source of imagined numbers.
- **Overconfident culture**: Beta strength, clamps, calibration (A.4).
- **Cultural mismatch** with this home: counts correct it; log-odds down-weight, never forbid.
- **Compute contention**: one VLM for the person, focus and wandering; strict pre-emption, person first.
- **Thresholds creeping in**: none is introduced here; flag any that appears, at its site (CLAUDE.md).

# Part G — Questions for Pablo

1. One `function_metaconcept`, or "function" as an instance type on a generalised metaconcept chassis (the open
   question closing `SCHEMA_GENERALITY_TODO.md`)?
2. Agent id, DDS domain, `REGISTER_TYPE` lines for the new attributes.
3. Is a person-confirmed room function a `human_claim` kind (`ltom_agent`), or does it belong with room identity
   (`ltsm_agent`)?
4. Per-step expected outcomes (B.3): in the mission IR (the planner writes them) or derived by the interpreter from
   the op (each op knows its own postcondition)? The second is safer and needs no planner change.
5. Is P0 worth doing before the kitchen frame is allowed to steer cabinets (read-only at milestone 2), so that what
   steers them uses measured rather than authored priors?
