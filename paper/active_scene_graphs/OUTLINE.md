# Active Scene Graphs — paper outline (arXiv preprint)

*Drafted 2026-09-14. Companion to `PAPER_PLAN_2027.md`: this preprint is the umbrella the two planned
papers hang from. It must NOT spend Paper 1's claim (the detectability model, experiments E1–E3). It
cites `bustos2026active` (IWAI) as the seed and the Applied Sciences 15(20):11084 paper as the
architecture ancestor, and says in one sentence what is new relative to each.*

---

## 0. The thesis in one paragraph

A scene graph is normally a *record*: a data structure that perception fills and planning reads. An
**Active Scene Graph (ASG)** is a scene graph that *acts to keep itself true*. Every node is a concept
with a posterior and a precision; every concept publishes what the robot could do to reduce its
uncertainty; the robot's own body is one such concept and calibrates itself the same way; a single
controller ranks all offers in one currency, nats, and executes them in real time over a distributed
graph. We state this as **five requirements** over standard scene graphs, show that the requirements are
**coupled** (satisfying them one at a time does not compose), give a model and an architecture that meet
them jointly, and evaluate the parts that run in simulation and on two physical robots.

The claim is NOT "nobody has done any of these". Each requirement has prior art. The claim is that they
are mutually entangled, so a system that meets them must meet them together, and that we have one.

**Working titles**
1. *Active Scene Graphs: a scene graph that acts to keep itself true*
2. *Active Scene Graphs: five requirements, one objective*
3. *Active Scene Graphs: requirements, model and architecture for a self-constructing world model*

---

## 1. The five requirements — the table

Columns: **R** requirement · **falsifiable statement** (what the paper commits to) · **standard SG**
(what a Hydra/ConceptGraphs-class system does instead) · **mechanism here** · **status** D=designed,
I=implemented, E=evaluated · **evidence / experiment**.

| R | Falsifiable statement | Standard SG | Mechanism here | Status | Evidence / experiment |
|---|---|---|---|---|---|
| **R1 Actionable** | Every node carries executable offers (affordances) whose preconditions are *probabilities read from the graph*, and one generic executor runs any of them without knowing the concept. | Nodes are labels + geometry; actions are planned outside the graph from a task spec. | `affordance` nodes under `has_intention`; epistemic (`rc::ObjectAffordance`, one per object) and pragmatic (`rc::PragmaticAffordance`, named + preconditioned: door approach/open/cross). Claim = epoch + `executing` edge the consumer owns. | protocol **I+E** (TLA+ `AffordanceExecutionClaim.tla`); door offers **I**, unrun; task-level EFE **D** | Sim: one room, one door, user task "cross the door on the left"; timeline of chosen offers. Real: same on Shadow. |
| **R2 Epistemic self-construction under FE** | The graph's own posterior uncertainty is *consumed*: it gates publication and is priced into the offers the robot acts on. No hand thresholds in the belief pipeline. | Uncertainty, if tracked, is stored; exploration is a frontier heuristic. | Per-concept generative model with Σ → Σ\* adequacy; NBV from ΔH with P(detect); existence belief `rc::exist` (log-odds, absence weighted by P(detect), occlusion holds); room adoption judge; corner σ floor. | **I+E** (objects, room); existence **I**, bottle unrun | ICRA draft (calibrated corner σ, real robot); 50 synthetic rooms .947→.954; 594 real layouts = NULL, cap-saturated (report as limitation, §7). |
| **R3 Concept-oriented** | The graph is populated by *independent concept agents*, each owning one class of node under a normative lifecycle (CREATE/UPDATE/REMOVE/HISTORY/OWNERSHIP), and the fleet composes without a central planner. | One monolithic mapper; classes are labels on one pipeline. | `CONCEPT_AGENT_LIFECYCLE.md` contract + per-agent conformance audit; ~10 concept agents + metaconcepts (ring, kitchen) + LTSM; `[Owns]` ties node identity to agent identity. | **I+E** (audit 2026-07-31; phantom births 130→1) | The conformance table itself; ablation: violate one stage (e.g. birth from a single frame) and count phantoms. |
| **R4 Embedded self-calibrating robot model** | The robot is a node like any other: its kinematic/extrinsic parameters are beliefs, updated online from the localiser's own correction, and their precision propagates into every offer's price. | Robot is a fixed URDF; calibration is offline. | `robot_concept` frame tree; batch estimator over episodes (SELF_CALIBRATION_DESIGN.md); closed-loop camera extrinsic; wheel-odometry velCov; motion preintegration. | **I+E** (ARM 7 replay passed; boresight +0.814° applied) | Correction load |est−pred| before/after; extrinsic yaw recovery 0.984. **Do not publish mount height/pitch** (measures the ceiling constant). |
| **R5 Real-time, scalable architecture** | N agents on M machines share one graph with bounded latency; sensor streams bypass the graph; agents join/leave without corrupting it. | Single process, or ROS topics with one map owner. | CORTEX/DSR CRDT graph; media plane on a dedicated DDS domain (zero-copy); presence protocol (debounced, TLA+); ownership cleanup on graceful exit. | **I+E** (lag 35→5 ms lidar, 36→9 zed; 20 Hz) | Latency table per stream; join/leave test. **Known defect**: CRDT dot cloud unbounded — bound it or scope to session length (§7). |

### 1.1 The coupling argument (why the table is a theorem, not a checklist)

State these as explicit dependencies; each has a measured incident behind it.

- **R1 needs R2 needs R4.** An offer's price is a log-probability. A precondition probability (door
  open?) is only meaningful if the posterior behind it is calibrated; calibration was only achieved once
  the self-model's extrinsics were solved (boresight, corner σ floor). Chain: actionable ⇐ calibrated ⇐ self-model.
- **R2 needs R3.** Epistemic self-construction with several writers requires ownership, or two agents
  hold different values of "the target" with no term in either able to reveal it (the boolean return
  channel bug, 2026-08-23; the cure is an edge the consumer owns + an epoch).
- **R3 needs R5.** Ownership is only enforceable if the transport guarantees a graceful exit runs
  cleanup and a transient peer flap does not delete a live node (presence debounce).
- **R5 needs R2.** Bandwidth is bounded only because the graph carries *beliefs* (Σ, log-odds), not
  sensor data; the media plane exists so the graph does not have to.
- **R4 needs R1.** Acting IS the calibration experiment: there is deliberately no "calibrate" affordance;
  the self-model learns from the corrections that ordinary offers generate.

Figure 1: the five requirements as nodes with these six arrows. Existing systems plotted as which
nodes they cover (from the related-work table).

---

## 2. One currency: nats

**Principle.** Every quantity the controller compares is a log-probability. There is one functional,
Expected Free Energy, and the task layer differs from the representational layer only by the prior
preference `C`:

```
G(π) = − E_q[ log C(o_π) ]            pragmatic:   preference over outcomes (the user's task)
       − E_q[ H(s) − H(s | o_π) ]     epistemic:   expected information gain (the concepts' offers)
```

- **Flat C** ⇒ only the epistemic term survives ⇒ the fleet's default behaviour: *move to represent,
  stay aligned with the world*. This is the bottom, always-on functional layer.
- **Sharp C over a goal state** ("robot in room B") ⇒ pragmatic term dominates; an epistemic offer off
  the route is taken exactly when its ΔH exceeds the pragmatic loss of the detour, in the same units.
  "If it does not perturb the goal too much" is this inequality, not a rule.

**Price table** — what each term is, and which constant it retires.

| Term | Price (nats) | Producer | Retires |
|---|---|---|---|
| Epistemic value of an offer | `P(detect)·ΔH_detect`, ΔH = H(Σ) − H(Σ\*) | each concept, from its own Σ | per-class gain scales (`aff_room_gain_scale = 0.35` must go: if a scale is needed the units are not homogenised) |
| Precondition | `log p(precondition)` (door open p=0.3 ⇒ −1.2 nats; p=0 ⇒ not an offer) | pragmatic affordance producer | on/off gating |
| Navigation cost | expected self-entropy growth `½ log det(Σ_pose + Q(path)) − ½ log det Σ_pose` from the **self-model's calibrated motion noise** (R4), + `log(1 − p_collision)` along the path, + delay against `C` | controller, from `robot_concept` | `Controller.AffordanceLambdaCost = 0.2` nats/m — a constant that should be a *function of the calibrated Q* |
| Commitment / interruption | hysteresis margin in nats (`switch_margin = 0.5`) — flag honestly as the one constant left, or derive as expected loss of the half-executed policy | controller | `if executing: don't switch` |
| Existence | log-odds, absence weighted by `P(detect)`, occlusion adds 0 | `rc::exist` | miss counters, debounce timers |
| Publication gate | posterior σ vs Σ\* adequacy gap, in nats | each concept | tuned publish bounds |

Why this works across a fleet: each producer computes ΔH from *its own* Σ, so the units are homogeneous
by construction, and the executor never learns what a door is (`aff_kind` is a label, never a dispatch
key). The controller's job reduces to global feasibility + argmin G.

Figure: for one decision instant, a bar per candidate offer decomposed into the price-table terms.

---

## 3. Section-by-section outline

1. **Introduction** (1.5 pp). Scene graphs as records vs scene graphs that act. The five requirements in
   one sentence each. The coupling claim. Contributions: (i) the requirements + coupling argument,
   (ii) the model: concept nodes with posterior/precision, affordances as priced offers, one EFE, (iii)
   the architecture that meets R3/R5, (iv) evaluation of the running parts in sim and on two robots,
   (v) an honest designed/implemented/evaluated ledger.
2. **Related work, organised per requirement, not per system** (1.5 pp). Ends with the table:
   systems × requirements. Candidates to verify: 3D Scene Graph (Armeni 2019), Kimera/Hydra (Rosinol
   2020, Hughes 2022), **S-Graphs** (Bavle 2022/23 — closest on rooms/walls-in-the-loop), ConceptGraphs
   (Gu 2023), Clio (Maggio 2024, task-driven), Khronos (Schmid 2024), HOV-SG, SayPlan, MoMa-LLM;
   active SLAM (Placed 2023); active inference in robotics (Lanillos 2021 survey; Pezzato; Oliver;
   Taniguchi world models 2023); **federated inference / belief sharing** (Friston 2024) for the
   multi-agent frame; body schema / self-perception (Hoffmann; Lanillos); CORTEX (Bustos 2019),
   Applied Sciences 2025, IWAI 2026. ⚠ Check whether "active scene graph" is already a term in vision
   (active learning for image scene-graph generation) and disambiguate in a footnote.
3. **Definition: the Active Scene Graph** (1 p). Formal: nodes = concepts with (μ, Σ, precision, existence
   log-odds); typed edges (RT with ts + covariance, `has_intention`, `executing`, `proto`, `owns`);
   agents = owners; the generative model per concept (SDF-based likelihoods: room polygon, table
   log-sum-exp box, cylinder); the lifecycle contract as the operational semantics. Section 1.1 goes here.
4. **One objective, two regimes** (1 p). Section 2. The representational layer and the task layer as
   one EFE. Interruptibility as re-scoring with a commitment cost (from `EPISTEMIC_PLANNER_DESIGN.md`).
5. **The robot as a concept** (0.75 p). R4: what is learnable (kinematic only; velocity loops hide
   dynamics), the localiser's correction as the free teacher, separability by component × covariate,
   why there is no calibration affordance. Downward coupling G P Gᵀ: P may only loosen.
6. **Architecture** (1 p). R3/R5: CORTEX graph, agents, media plane, presence + affordance protocols
   (both TLA+-verified; one sentence on what the models proved and what timing they abstracted away).
   Ledger table: mechanism × {designed, implemented, evaluated}.
7. **Evaluation** (2 pp). Premise inherited from `PAPER_PLAN_2027.md`: open-loop dataset comparison does
   not apply; comparisons are policy-vs-policy in one closed loop; our own replaced heuristics are the
   comparators.
   - E-R2a: calibrated corner σ on the physical robot (from ICRA draft; do not duplicate its figures).
   - E-R2b: room shape, 50 synthetic rooms + 594 real layouts, **including the null** and why
     (cap saturation: 594/594 rooms hit the 900-frame budget ⇒ the endpoint cannot discriminate).
   - E-R1/R2c: **the minimal slice** — one sim room, one door, one live epistemic offer (fridge NBV),
     controller logging the G decomposition per candidate per decision; figure = timeline of chosen
     offers coloured pragmatic/epistemic with goal distance underneath. Then the same on Shadow.
   - E-R3: conformance audit + phantom-birth count before/after the lifecycle contract.
   - E-R4: correction load and extrinsic recovery (ARM 7).
   - E-R5: latency per stream, join/leave.
8. **Limitations and claim discipline** (0.5 p). Explicitly: epistemic term not shown to beat a heuristic
   at single-room scale; CRDT growth unbounded; task-level EFE preliminary; two constants remain
   (switch margin, and λ until replaced by calibrated Q); detectability envelope uncalibrated on real
   images (reserved for Paper 1).
9. **Conclusion.**

Target: 8–10 pp two-column + appendix with the ledger and the conformance table. arXiv v1 = design +
running-parts evaluation; v2 adds the task-layer experiment on the real robot.

---

## 4. What must be built for the paper to be honest (minimal slice)

1. Replace `λ·nav_dist` with the self-entropy price from the calibrated motion covariance
   (`affordance_manager.cpp:770`); keep λ as an A/B flag for the figure.
2. Add `log p(precondition)` to pragmatic offers instead of on/off presence (`pragmatic_affordance.h`).
3. A task prior `C` in the controller: one goal state, one preference sharpness; log the decomposition.
4. Run the door approach/open/cross offers once in sim (built 2026-09-12, unrun).
5. Bound the CRDT dot cloud, or state the session-length scope.
6. Literature table (systems × requirements), with a search for the term "active scene graph".

Everything else in the table already has evidence on disk.
