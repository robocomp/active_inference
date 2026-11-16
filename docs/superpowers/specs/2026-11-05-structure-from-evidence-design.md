# Structure from evidence: instantiating rooms and doors by one comparison

Status: DRAFT for Pablo's review. Nothing is built. Date: 2026-11-05.
Paper: *Instantiating Rooms and Doors by Bayesian Model Reduction*, ECMR 2027, due 2027-02-26, 6 pages.

Marks used below: **[user]** Pablo decided it. **[call]** the author's call, open to correction. **[build]** left to the builder.

## 1. Intent

The paper's goal **[user]**: show that a robot can build and maintain a long-term map of rooms connected by doors, starting from no map, by treating rooms and doors as prior concepts whose instances must be justified by evidence.

Today, deciding what exists takes several unrelated mechanisms. This work makes those decisions one comparison in one unit (nats), adds removal and merging, and measures whether that beats the current rules. Closest prior work grows maps by expansion only and lists merging as future work (AIMAPP, arXiv 2510.09574v2, Sec 5.3), or has rooms and doors with no probabilities (Bao et al., ICRA 2025).

Decisions already taken **[user]**:
- Rooms and doors are pre-existing concepts, instantiated from evidence. They do not emerge from geometry. Door evidence stays in.
- Position on concepts: how human concepts come to exist is not known, as far as Pablo knows. Developmental or hierarchical clustering may or may not end in a concept, and the gap between a hierarchy of blobs and the concept of a room, table or fridge is huge. This work does not claim to explain the origin either. It provides a predefined concept template with very specific claims, used for all types, acknowledges that, and studies how concepts are instantiated and grounded in real time with Active Inference. Any claim about the gap or the origin must be cited.
- The stated advantages of this position: concepts can readily be used by a local VLM, and they can be communicated to humans, who share a good part of their semantics.
- A label from an LLM is not grounded by itself, and this work defends that it is not enough. A label only gives the referent shared with humans. To ground it, the internal representation (geometric, aspect, functional) has to make predictions.
- These priors describe connected rooms. Limits belong to the prior set, never to the method. Never write that the method "does not work" in warehouses or fields.
- `field_concept` (rectangle with N rows of M trees) is a second prior set being built by another student. It assumes it is in a field at start, as the robot assumes it is in a room, then aligns by exploring, loads from LTSM, or localises.
- Mission tours (R1-R2-R3-R1 and back) are not exploration. Exploration claims rest on tours from an empty LTSM.
- Noé can run real-robot experiments with two rooms. Extension is still to be worked out.
- Hand-edited leaf sides and non-cumulative memory can be fixed.
- The free-energy claim and the epistemic planner are to be built, not assumed. Experiment time is not the limit.
- Passages: the comparison decides only which door faces pair in memory. The crossing protocol and `exit` chain are unchanged.
- Door birth: the comparison replaces the birth test for doors. The tracker still holds and associates candidates, and the lifecycle contract gets an amendment.
- The door prior comes from typical door dimensions, not from the stored maps, so it stays independent of the test data.

## 2. What exists today

Paths are on `origin/main` = `feat/room-recognition` = `aff02fde` (2026-11-05).

| Decision | Mechanism | Where | In nats? |
|---|---|---|---|
| Door exists (birth) | Two paths. Tracker: streak of weighted observations (new observations only, confidence x range), `streak >= birth_frames`, expires on a miss. Ricoh bank: Bernoulli log-likelihood ratio with absence, ripens at `birth_frames x ln(h/pc)`, then credits the tracker. The evidence is dropped at birth: existence is seeded at a fixed 1.0 nat | `common/instance_tracker/`, `common/instance_tracker/birth_evidence.h`, `door_concept/src/core/ricoh_door_birth.h`, `door_core.cpp` | Tracker no, bank partly |
| Door exists (removal) | Log-odds pooled per viewpoint, then `decide_removal` | `door_concept/src/door_look.h`, `common/existence_belief/existence_belief.h` | Yes, but see 3.2 |
| Room shape | Accept edit iff `dNLL + kappa d|V| + ln n < 0`; promotion takes the shortest of 60 outlines | `room_concept/src/layout_refiner.h` | Yes |
| Room shape (boxes) | `simplify()` scores delete, merge, absorb on `mdl_cost` | `room_concept/src/room_boxes.cpp` | Yes |
| Room new or known | Posterior over known rooms + new; commit is the MAP; uniform prior | `common/room_recognition/room_posterior.h`, `ltsm_agent/src/room_recognizer.cpp` | Yes |
| Appearance for identity | Consumer only: `room_recognizer.cpp` reads a `place_evidence` node. Nothing publishes it, and rooms store no appearance in LTSM. Every recorded SWAP logs `appearance loglr 0.00 (unavailable)`. To build, see 3.4 | `ltsm_agent/src/room_recognizer.cpp`, `retina/src/place_stage.*` | Not built |
| Two door faces are one passage | Crossing protocol rules, then ltsm pairs; `passage_llr` is a separate statistic | `ltsm_agent/src/passage_live.*`, `ltsm_agent/src/room_eviction.cpp`, `ltsm_agent/EVICTION.md` | No |
| Where to look | Live: `layout_explorer` (EIG per metre). In a known room: `EpistemicPlanner` (FIM D-optimality + distance bonus + inhibition of return) | `room_concept/src/` | Partly |

Theory to follow, not to restate: AuRo draft `sections/03-foundations.tex` section 3.4 (equations `deltaF`, `roomF`).

Evidence so far (`room_concept/estimation_scripts/crossing/`):
- warm46-61: 46/48 legs, hands-off, all doors shut at start. Reload only: each tour copies a fresh hand-edited fixture, and R1 re-entry is recall, so these say nothing about recognition or exploration.
- Recognition: 7 SWAP events in warm13/15/16/18/20/26, posteriors .62-.91. Only warm13 and warm15 are graded correct.

Known wrong structural calls, from the `room` session's logs:
- warm30: a real shut door 5.7 m away deleted by absence evidence, losing its recall.
- warm11: R1 re-entered as a new room_4, with recognition off.
- proto15: false crossing through a phantom door makes a duplicate room (TLA+ CE-1).

## 3. The comparison

### 3.1 One library

`common/structure_evidence/structure_evidence.h`: header-only pure math, no graph, no agent code **[call]**.

For a hypothesis set H about one structural question:

    F(h) = -log_evidence(h) + complexity(h) - log_prior(h)     (nats)

The decision is the minimum F, with no margin or level. Ties decide nothing, as in `commit_step`. Any constant the builder cannot derive must be flagged at its site with its justification (CLAUDE.md, no thresholds).

The header must not contain the words room, door, wall or passage. A unit test instantiates it with a synthetic N-rows-by-M-trees hypothesis set, including a missing tree, to prove that **[call]**. That test is not `field_concept`, and no field code is in scope.

Every decision writes one row to a per-run `structure_decisions.csv`: time, concept, hypotheses, the three terms per hypothesis, the winner, and what changed in the graph. Ground truth is joined offline. This file is the paper's main evidence.

### 3.2 The four levels

| Level | Hypotheses | Evidence | Complexity | Current bug or gap |
|---|---|---|---|---|
| Door existence | exists / not | detections, aperture, absence weighted by P(detect), line of sight including walls | KL of the door's parameters against the door prior | Tracker birth is in other units, and birth discards its evidence. Absence is over-charged |
| Room identity | previous / recalled k / new | metric, appearance | recalled: small, from the seam chain; new: `kappa |V|` | Check whether `metric_log_new` carries `kappa |V|`; appearance not fed; prior uniform |
| Passage **[user: pairing only]** | two faces are one door / separate | crossing fact, seam consistency, width and position agreement, appearance | one door against two | Pairing follows the protocol, no probability. The crossing itself is not touched |
| Room shape | outline edits | existing NLL | `kappa |V|` | Already this form. Only adapt units and logging |

**Absence over-charge.** The silhouette absence in `door_concept/src/core/door_existence.cpp` and `door_look.h` charge `p_detect * log((1-ph)/(1-pf))` for a miss. The exact Bernoulli form, already used by `RicohBirthBank` and the Ricoh channel (`door_existence.cpp:996`), is

    log[ (1 - p_detect * ph) / (1 - p_detect * pf) ]

which gives no evidence at `p_detect` = 0 without any special case. With `ph = .85`, `pf = .05` (illustrative), the linear form charges about 1.9x at `p_detect = .41` and 2.2x at `p_detect = .1`. A hit stays `weight * log(ph/pf)`, as in the bank. Keep the per-viewpoint pooling.

`rc::exist` is shared by the fleet. Change door_concept first behind a config flag. Other agents move only after the door result, on Pablo's decision **[call]**.

The exact form goes into `door_concept` first (silhouette and `door_look.h`). The `llr_free` sites in `existence_belief.h` have the same linear-against-exact question and wait for the door result **[call]**.

**Door birth [user: replace the test].** Today's tracker test is not a raw count: it already weighs new observations by confidence and range. Three things are wrong with it for this work: it is not in nats, a candidate expires on its first miss, and the evidence is thrown away at birth, because existence restarts at a fixed `Existence.BirthLogodds` = 1.0.

- Both sensors feed one signed log-likelihood ratio per place, generalising `RicohBirthBank`: hits and misses by the exact Bernoulli form above, with its per-viewpoint crediting so a parked stare cannot manufacture evidence **[call]**.
- A place is ready when F(exists) < F(not). This replaces `birth_frames x ln(h/pc)` and the tracker streak for doors.
- It then credits a full `birth_frames` through `Detection::birth_evidence`, as the bank already does **[call]**. So `InstanceTracker` is not edited, other agents are unaffected, and min-separation, association and exclusion still apply.
- The accumulated evidence seeds the existence log-odds at birth, so birth and removal are one running quantity. Restoring a remembered door keeps its existing prior.
- `CONCEPT_AGENT_LIFECYCLE.md` section 1 and its door_concept exceptions get the amendment: birth stays tracker-only, and for doors the credit is decided by this comparison.
- `Existence.BirthLogodds` and the "not before `birth_frames` ideal hits" check in `door_concept/tests/test_door_core_tracker.cpp` change with the flag on.

**Door prior [user: typical dimensions].** Complexity is the KL divergence from the door's parameter estimate to a prior over width and height, plus a uniform prior on position along the wall. A parameter the views have not constrained costs almost nothing, so a truncated or occluded view is not penalised.

- Values **[user]**: width 0.85 m, std 0.12 m; height 2.05 m, std 0.06 m.
- They are separate from `Door.PriorWidthM/Std` (1.00, 0.06) and `Door.PriorHeightM/Std` (2.10, 0.08), which are strong fit priors. Using those here would charge a real 0.80 m door about 5.6 nats for width alone, which matters for E8.
- **Do not score the raw mask height against this prior.** `door_core.cpp` (the `size_w` note above line 537) records that using the prior's sigma as mask noise blocked births: masks under-estimate height, and that noise model has not been measured. The prior scores the door parameter estimate with its uncertainty. If the builder needs a mask-height likelihood, it is measured from logged `mask_h` first.

**Where a door may be [user].** Doors are in walls, and this prior is extremely hard. It already exists as the wall band, `band_log_prior` in `door_concept/src/door_room_band.h`: 0 on the band `[0,T]` beyond the room-side polygon face, `-0.5 (d/sigma)^2` off it on both sides, with `T` = 0.20 m and `sigma` = 0.10 m. Use it as the location part of `log_prior(exists)`; do not add a second one.

- Today it acts only after birth, on a resolved aperture (`door_existence.cpp:868`). At birth a ZED candidate faces only inside-polygon plus 0.40 m (`door_core.cpp:765`), and a Ricoh one a wall-agreement weight. The comparison applies the band at birth and carries it into existence as one term, entered once.
- A Ricoh position is fused onto the wall, so the band on that position is trivially satisfied, and its wall agreement stays in the likelihood. **[build]** confirms the same fact is not counted twice.
- The prior is hard about the true wall, but the code only knows an estimated wall. The real DOOR_2 sat 0.2 m inside room_3's estimated polygon and scored 0.15-0.23 (`recog_proto5`). So `sigma` should carry the wall's own uncertainty, and a wrong wall must not kill a real door. A test covers that case.
- In a provisional room the polygon is a free-space frontier, not walls (`door_core.cpp:217`), which is the main case in an empty-LTSM tour. **[build]** defines the band there and tests a door seen before its wall is estimated.
**How many doors on a wall [user].** More than one door in a wall is extremely rare, unless the space is a corridor, which is a different kind of space. So the base rate is per wall, not per metre of wall:

- A wall's first door pays `log P(wall has a door)`, then a uniform position along that wall. Each further door on the same wall also pays `log P(another door | wall already has one)`.
- Values, for Pablo to correct **[call]**: P(wall has a door) = 0.4, from rooms having one or two doors in about four walls; P(another door) = 0.01, or 4.6 nats. Report sensitivity for both.
- This is a cost, not a ban, so strong evidence can still add a second door and no threshold is added. Like the band, it enters once and is replaced, not accumulated: if one door on a wall is removed, the other stops paying.
- A wall is not a polygon edge. R1's stored outline has 32 edges, with one wall split by a niche, and provisional outlines change. **[build]** groups edges into walls, starting from `wall_hypotheses` in `ricoh_door_birth.h`, which already recovers the line a niche interrupts. Doors whose wall is ambiguous must not pay in full.
- Corridors are a different concept and are out of scope. These priors describe rooms, so a corridor is a limit of the prior set, not of the method.
- Check on the stored `three_rooms_door0` map: all 6 door faces are on different walls. That fits the prior but is only one world, and no paper world tests a real second door. A unit test covers one.


**Passage [user: pairing only].** The comparison runs where memory pairs faces (`room_eviction.cpp`, the `mint_passage` and `match` block, and `room_swap.cpp` carry-over). The live `PassageLive` is not touched, because crossings and recognition read it. A completed crossing always stands: the robot's room change, `exit` and the protocol do not wait for it. If the comparison says "separate" after a crossing, such as a false crossing through a phantom door (proto15), memory holds no passage and the row in `structure_decisions.csv` records it. It may also pair two faces, or undo a pair, later as evidence arrives. **[build]** must list what reads the pairing during a tour and confirm that none can stall the crossing, and that the TLA+ model needs no change. If one does, stop and bring it to Pablo.

**Active, not shadow [call].** With the flag on, the comparison makes the decision. With it off, today's rules decide and the comparison still logs what it would have decided, so disagreements can be read from any run. The paper's claim is about decisions, so logging alone would not support it.

**Room identity [call].** `commit_step` stays as the staging machine and is fed by the posterior, which already has this form. Add `kappa |V|` to the new-room term if `metric_log_new` lacks it. Numbers may shift, but warm13 and warm15 must still reach the correct room. The prior stays uniform and flagged, with sensitivity reported, until there is evidence for another; a CRP-style prior is not in scope.

### 3.3 What "Bayesian model reduction" means here

Title and paper use this term **[user approved the abstract]**, so it must be accurate:
- True reduction scores the simpler model from the full model's accumulated evidence, without refitting.
- Door existence and passage merge: true reduction. The accumulated log-odds is the log Bayes factor for "with" against "without".
- Room identity merge: true reduction where the provisional room's evidence is rescored under the recalled room's prior. **[build]** to confirm this holds without refitting; otherwise it is reported as model comparison.
- Room shape stays MDL, described in the paper as an approximation, because refit discards the box posterior **[call]**.

If door and identity cannot be made true reduction, the title becomes *Instantiating Rooms and Doors from Evidence*.

### 3.4 Atlas as appearance evidence

**Design [user].** Retina builds the descriptor-pose atlas of the room the robot is in. Before `room_concept` removes the old room, ltsm stores that atlas in its copy of the room. Rooms' aspects are then compared in ltsm. This replaces the deferred design in `docs/superpowers/plans/2026-10-23-room-recognition.md` Tasks 2 and 4, where retina held every room's map, matched against candidates and published `place_evidence`.

**Rule [user].** Agents communicate only through the memories. Files do not cross agents, so the atlas reaches ltsm through the graph. Retina owns the live atlas, on its own node under the room **[user]**. The rest of this section is the author's proposal **[call]**.

**The object atlas is the precedent.** The room atlas is the same idea with the view reversed **[user]**: an object is seen from outside, a room from inside. `common/view_atlas/VIEW_ATLAS.md`, section 13 on `feat/view-atlas-2b` (agreed design, not built) and the ltom spec section 6.3 settle the pattern: alive, the owner keeps the atlas and publishes it at a low rate, when it has absorbed new views; ltom copies it by observing; dead, ltom is the store of record.

| | Object atlas | Room atlas |
|---|---|---|
| Camera | Outside, looking in | Inside, looking out |
| Indexed by | Azimuth in the object frame, 12 bins | Position in the room frame |
| Unknown heading | Cyclic shift marginalised over the bins | Cyclic shift of the panorama sectors |
| Unseen part | Hypothesis relaxes to background, ratio goes to 0, no gate | Should do the same with distance from stored places |
| Size | Under 10 KB | Must be made comparable, see below |
| While alive | Owner | Retina, on its own node under the room |
| Long-term store | `ltom_agent` | `ltsm_agent` |
| Different encoder id | Not compared, ratio 1 | Same rule |

How it fits the code:
- Retina's `PlaceStage` builds one atlas per visit and publishes it on a node under the current room, when it changes. It needs no recalled maps.
- Eviction step 2 already copies whatever hangs under a room (`ltsm_agent/EVICTION.md`), so the atlas reaches memory in the existing verified transaction, with the JSON generation as its store. Views after the last publish are lost, which is harmless. **[build]** checks that an atlas node owned by another agent is released cleanly with the room.
- `common/place_memory/place_map.h` needs no ONNX, Qt or DSR, so ltsm can compare descriptors without a GPU and fill `AppearanceEvidence` directly. The `place_evidence` node and its attributes are no longer needed.
- Comparison is incremental: each change in the provisional room's atlas against each known room's atlas. Measure its cost.
- When a swap commits, the visit's atlas joins K's through T(K <- P), so appearance memory is cumulative. On undo it stays with P.
- The object atlas does not prove this works: its phase 1 offline criteria failed (`VIEW_ATLAS.md` section 8), and it uses another encoder and camera.

The log-likelihood ratio must be discounted to independent looks (`discounted_loglr`), with `n_eff_per_s` estimated from the measured autocorrelation, not chosen. It joins the room identity level only. Using it for goal setting is out of scope.

Known problems to solve:
1. **Size.** A raw keyframe is 26 KB (17 x 384 floats), so a fully covered 40 m2 room is about 4 MB, too much for a graph attribute. Consolidate as the object atlas does: one mean per place, at about the descriptor's decay distance, not every keyframe. Identity may not need the heading, because `posterior()` accepts appearance as a ratio with no pose. A heading-free summary is 384 floats, 1.5 KB per place, or about 60 KB for that room at 1 m spacing (arithmetic, not measured). Whether it keeps enough discrimination is measured in the evaluation below, as an arm against full sectors. Publishing cost is measured too, since the rule is that rates stay nominal.
2. **Frame.** Keyframe poses are in the room frame as estimated at capture, and a re-fit shifts them. Provisional rooms and promotion are the main risk. The atlas must store or follow the transform.
3. **Camera against robot pose.** Keyframes are inserted by camera pose but proposed by robot pose, a position error up to 34 cm (memory `place-map-keyed-on-robot-not-camera-pose`, specified, not built, to be re-verified). Fix it before fitting any constant.
4. **Overlap on entry.** Descriptors decay within about a metre (fitted r = 0.58 m, inflated by problem 3). Entering through a never-crossed door, the robot may be far from any stored place. As in the object atlas, the hypothesis relaxes to background with distance, so far from stored places the ratio goes to 0: weak evidence, not a false "new". That is E4's exact case. Report evidence against distance from stored places.
5. **Open doors.** A panorama at a doorway sees the next room. The evaluation below includes doorway views.

**Gate.** The cross-room evaluation comes first, on recorded tours. The plan's FULL, WEAK and NONE verdicts apply as written (Task 0.2), and its tool `place_xroom_eval` still has to be written. The AUC of .988 on record compares near and far moments in time, not different rooms, so it does not answer this. NONE means the appearance arm is dropped and the paper says metric only, after asking Pablo. WEAK means appearance enters at its small measured weight. The evaluation must also include similar rooms, because the three Webots rooms may not represent real ones.

Retina's rate must be re-measured with this on; the 0.83 ms median per frame on record predates today's retina. If it falls below nominal the arm is invalid until fixed, and if it cannot be fixed the appearance arm is dropped **[call]**. Update `CONCEPT_AGENT_LIFECYCLE.md`, which correctly says this expert does not exist yet, once it does.

## 4. Where to look

Objective **[call]**: expected reduction in free energy per metre, over geometry and unresolved structural hypotheses, using the same P(detect) as 3.2. An unresolved door is worth a look; a settled one is not. Base it on `room_concept`'s `layout_explorer` selection, and take the structural term from `door_concept`'s planner, which already uses `existence_eig` weighted by P(detect) **[call]**. The bench must run the agent's code. **[build]** decides how this joins `room_concept`'s `EpistemicPlanner`, and should read all three before proposing it. Other agents' planners are out of scope.

Fair test, since the 594-layout null saturated at the frame cap: arms are current explorer, new objective, frontier. Report coverage, `seen_frac`, structural errors and shape IoU **as curves against metres driven**, not endpoints. IoU is never reported without `seen_frac`.

## 5. Prerequisites, before any paper run

1. **Door leaf stall.** warm59 and warm61 failed with P(passable) stuck at .54, and warm37 had a shut door at .96. Leaf state is a door's state, not its existence, so it is outside the comparison and fixed first because it blocks tours **[call]**. A separate spec and plan cover this, approved and not yet built: `docs/superpowers/specs/2026-11-05-door-passable-and-passage-deadlock-design.md` and `docs/superpowers/plans/2026-11-05-door-passable-and-passage-deadlock.md`, on `rt-build`. Follow them. If it is unresolved by 2026-12-15, stop and bring it to Pablo, because every paper run depends on it **[call]**.
2. **No hand edits.** Leaf sides and hinges learned at first opening and kept in LTSM (decided 11-02).
3. **Cumulative memory.** A tour script that starts from the memory saved by the previous run, not a fresh copy.
4. **Empty-LTSM tour.** Works hands-off over the ring **[user expects it to]**.
5. **Rates logged per tour.** Controller, room and retina, against nominal. A run below nominal is a failed run.

## 6. Experiments

All in Webots unless noted. Each compares single comparison against current rules behind one config flag, with N >= 10 per arm **[call]**. Run on the same binary, with rates logged.

| # | Question | Design | Carrying result |
|---|---|---|---|
| E0 | Does it fix known failures? | Offline replay of warm30, warm11, proto15 moments where logs allow | Table of decision per case |
| E1 | The full chain | Empty LTSM, then at least 3 cumulative sessions from the robot's own memory, no hand edits | Map quality and structural errors per session |
| E2 | Better than rules? | Same tours, arms flag on and off. Metrics: false doors, missed doors, duplicate rooms, false merges, layout error against GT | Main table |
| E3 | Stress | Inject false door detections; vary how much of a room has been seen | Where each arm breaks |
| E4 | Recognition, and what concepts buy | Re-enter through never-crossed DOOR_0; arms atlas only (appearance expert alone), concepts only (metric), both. Negative control: a similar new room must not merge | Posterior traces, undo count |
| E5 | Goals | After a door state or furniture changes: does a stored goal image still match (place-match score), and does a state goal ("be in room_2") still resolve? A validity test, no image-goal navigation is built | Validity per arm |
| E6 | Planner | Section 4 | Curves against metres |
| E7 | Baselines | Run the public code of AIMAPP (`decide-ugent/AIMAPP`) and Bao et al. (`FreeformRobotics/Active_room_segmentation`) in the same worlds | Comparison row |
| E8 | Real robot | Noé, two rooms: build, stop, restart another day, reload, carry on; repeat with changed furniture or door state | One real chain |
| E9 | Second prior set (optional) | If `field_concept` is ready: what changed against what stayed, one instantiate-and-remove result | Short table, or stated as ongoing work with no measured claim |
| E10 | Label and grounding (optional) | After an empty-LTSM tour, a person asks about the robot's own map through the local `language_agent`. Questions cover the shared referent ("how do I get to the bathroom?"), belief ("are you sure there is a door there?") and prediction (what should be seen or should happen at a place). Prediction answers are then checked by going there | Answers graded against ground truth and against the robot's belief at that time; predictions against outcomes |

Two rooms cannot show entry through a never-crossed door, so E4 stays in simulation, and the paper says so.

**E0** needs logs from those runs. If they are insufficient, reproduce the situation with a new run and say so; it is a check on known cases, not a regression test, since numbers may shift.

**E10** uses `language_agent` as it stands, with no new language features. It can already handle these questions **[user]**. If E10 is not run, the advantages stay in the paper's discussion as motivation, not as a result.

**Grounding across experiments [call].** The paper reports what each part of a concept predicted and whether that held. Geometric: walls and door pose in views not yet taken, in E1, E2 and E6. Aspect: appearance of a recalled room on re-entry, in E4. Functional: that a door can be crossed, against the crossing outcome, in E1 and E8. A scene-graph comparison must be fair: such systems also attach geometry to labels, so check what each one actually predicts before writing that it does not.

**E7** reimplementing a baseline is not acceptable, because a re-implementation is not their method. If their code cannot run in our worlds, make no head-to-head claim and position them in related work **[call]**.

**Ground truth** comes from the Webots world (`threerooms-piso.wbt`): door lines and room regions, as `warm_start_tour.sh` already uses for legs. A new `grade_structure` script joins it with `structure_decisions.csv`. Definitions **[call]**: a false door is an instance alive for longer than one room visit with no true door within its width; a missed door is a true door in a visited room with no instance by the time the robot leaves; a duplicate room is two live or remembered rooms whose regions overlap the same true room; a false merge is a commit to a known room whose true room differs.

Required for the paper: E1, E2, E4, E6, E8 **[call]**. The others strengthen it and are run too unless they would delay those.

Failure counts as a result: a tie in E2 is reported, and section 8 applies.

## 7. Goals, non-goals, anti-goals

Goals: all of section 3, section 4, section 5, and the runs above.

Non-goals: object birth and removal (stays in AuRo; object agents keep current behaviour). General `field_concept` work. The agent-rate self-state. Multi-floor. Language or missions beyond what tours and E10 need.

Anti-goals, which would make this a failure even if it runs:
- Thresholds added to make the comparison behave.
- A run presented below nominal rates, or a hand-edited map presented as robot-built.
- Numbers reported from a bench that re-implements the agent's code.
- Agents other than door_concept changed in existence behaviour without a measured door result.
- A controller or crossing regression: warm46-61's 46/48 must not fall.
- The crossing protocol or `exit` chain changed by this work.

## 8. Decision date

**2027-01-15** **[call]**. Continue as planned if E2 shows fewer structural errors than current rules with no more misses, and E1 completes.

If E2 ties, the fallback paper is the build, restart, recognise chain with existing rules, and the title changes. The third fallback is the model-checked protocol paper.

ICRA decides around 2027-01-31 (date not independently confirmed).

## 9. Constraints

CLAUDE.md applies, notably: queued connections only; SIGTERM, never `kill -9`; type-attributed DSR attributes only (new ones need Pablo's cortex reinstall); `std::from_chars` and classic-locale writes for data files; `cbuild` at -j8. Any harness parsing agent files needs `setlocale(LC_ALL, "")`. Measure the cost of every per-cycle change. Overlap: AuRo cites this paper and does not restate it while under review.

## 10. Left to the builder

Library API, flag names, CSV columns, how each agent adapts. How to construct the similar-room world for E4. Whether E0 needs scan tapes that older runs lack. Where `ph` and `pf` come from for a door, reading `door_look.h`'s callers and `common/media_transport/sensor_model.h`.

Base branch: a new branch from local `main` (`b2b74947` or later, which contains `aff02fde`), coordinated with the `room` session, whose door passable plan is committed at `a7ae1633` **[call]**.
