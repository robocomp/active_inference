# Paper plan 2027

*Rewritten 2026-08-16. Supersedes the 2026-08-13 assessment: the ICRA deliberation, the decision
gate and the "no external baseline" framing have been resolved and are not repeated. What follows is
the plan, not its history.*

---

## Verdict: two papers, in this order

1. **An 8-page paper on the detectability model** — RA-L, submitted Dec 2026 – Jan 2027, with
   IROS 2027 (deadline 1 Mar 2027) as the fixed backstop. Pick one, not both.
2. **A journal paper** afterwards, which subsumes it. Spine: the two-room transition, plus a
   term-by-term accounting of which parts of the formalisation pay for themselves.

**ICRA 2027 is not attempted.** It closes 15 Sep 2026; the experiments below cannot be done and
repeated in that window, and a comparison built in three weeks and never repeated is what reviewers
punish hardest.

---

## Settled premises — do not re-litigate these

- **A dataset comparison does not apply to this work, and its absence is not an oversight.** A dataset
  is an open-loop recording: under Active Inference the observations received depend on the actions
  chosen, so replaying a fixed sequence scores one policy on data a different policy generated. No
  recording can say what the robot would have seen had it turned the other way. The comparison that
  *is* meaningful is **policy versus policy inside one closed loop** — body, sensors, noise, timings
  and environment held fixed, only the decision rule changed. This is now argued in the thesis at
  §2.4, and the active-SLAM literature (Placed et al. 2023) frames evaluation the same way.
- **A system that must be handed a target pose cannot enter that comparison at all**, because the
  loop it would have to close is the thing being compared.
- **Our own prior art is a legitimate comparator, not weak self-benchmarking.** Each heuristic being
  replaced was written by people who wanted it to work, tuned until it did, and shipped. A
  reimplementation of someone else's method can be called weakened; the code we replaced cannot.
- **The *Applied Sciences* 15(20):11084 paper is ours, contains no external quantitative baseline, and
  is intentions rather than formalisation** — no exact joint, no stated approximations with costs, no
  common-mode ceiling, no life-cycle contract, no cavity. It removes the *architecture* from the table
  as a novelty claim, but it does not block a formal treatment of the same system.
- **The IWAI 2026 extended abstract (`bustos2026active`) helps Paper 1 rather than blocking it.**
  Read 2026-08-16. It states the AASG claim for an Active Inference audience — the working memory
  reframed so perception and action are one variational quantity — and sketches per-class beliefs over
  differentiable distance fields (room-polygon SDF; a table as a smooth log-sum-exp box top with four
  cylindrical legs), epistemic affordances, and EFE selection. Critically, its viewpoint selection is
  **"a Fisher information gain proxy ($\Delta H_f$)"** — information gain *without* detectability,
  which is precisely **arm (b) of E2**. So our own prior art now publishes the baseline the RA-L paper
  improves on, and the delta is cleanly citable instead of being an internal comparison. It does not
  stake the detectability claim anywhere.
  ⚠**One divergence to keep in mind when citing it**: the abstract presents MPPI as the controller,
  whereas Chapter~12 of the thesis reports that the PD tracker beat the sampler. Do not let a paper
  imply MPPI is what ships.
- **The remaining unpublished claim at 8-page scale is the detectability model.**

---

# PAPER 1 — the 8-page paper

**Working title.** *One detectability model, two consumers: detection-weighted viewpoint selection and
existence belief for object-level scene graphs.*

**The claim.** A single calibrated inverse perception model `P(detect | viewpoint, object belief)`,
fitted from the robot's own logs and marginalised over the object's own uncertainty, should price
**both** where the robot goes to look **and** how much a non-detection counts against existence —
retiring the hand-set stand-off margins, visibility gates and debounce counters it replaces, while
yielding more uncertainty reduction per metre and fewer phantoms than information-gain-only NBV.

**Why this claim.** It is falsifiable in one experiment, it is not covered by the Applied Sciences
paper, and `action.tex` §11.3.2 is already close to paper-ready prose for it. The *two consumers* part
is what makes it more than an NBV tweak: the same calibrated quantity is shown to do two jobs that are
normally engineered separately.

**What would make it a different paper.** If E3 below fails, the honest move is to re-title around
viewpoint selection alone and drop the second consumer, rather than to keep the title and weaken the
evidence.

## The experiments, specifically

### E0 — the closed-loop harness *(prerequisite, not a result)*

One fixed authored tour. The **decision rule is the only free variable**: same body, same sensor
models and noise, same timings and transport latencies, same segmenter, same apartment. N ≥ 10 runs
per arm. Logs every metric below per run, per object, per cycle.

This does not exist. Component-level benches do (`controller/tools/route_bench.cpp`, `mppi_bench.cpp`,
the per-agent `*_eval.csv` logs) and Webots already runs identical code to the real robot through the
component layer, but there is no swap point. **~1–2 weeks, and it is the single highest-leverage item
in this plan** — it is also exactly what Paper 2 needs, so it is built once.

⚠ Known hazard: the unbounded-CRDT growth that forces `kill -9` is hostile to an unattended 40-run
batch. Fix or bound it before the batch, not during.

### E1 — the envelope, re-fitted on real images *(the motivating figure)*

Fit `P(detect)` on the **real** table and fridge from the robot's own logs, and compare against the
simulation-fitted envelope and against the hand-set prior it replaced.

- Known already: on a 2296-row fridge tour the hand prior was wrong by **more than 2×**
  (`max_fill` 1.317, CI [1.21, 1.49], soft 0.258), pushing the robot about **1 m too far back**.
- Known already: miss-correlation by `p_detect` band is **ρ = 0.602 / 0.327 / 0.198**, which is direct
  evidence the envelope is predictive rather than decorative.

**Why it is not optional.** The thesis's own sim-to-real position is that segmentation failures do not
transfer, so an envelope fitted in simulation is arguably measuring the renderer. This experiment
converts the paper's weakest point into its strongest. The hardware needed is live.

**What each outcome means.** If the real envelope differs materially from the sim one, that *is* the
result and the calibration is the contribution. If it transfers unchanged, say so plainly — the paper
loses a motivating figure but gains a transferability claim, and either is publishable.

**Cost:** days. **Extend beyond the fridge**: at least table and one more class, or "calibrated
envelope" reads as a single anecdote.

### E2 — the viewpoint consumer

Policies swapped into E0's harness, ≥10 runs each:

| arm | rule | what it isolates |
|---|---|---|
| (a) | calibrated `P(detect)`, both consumers | the claim |
| (b) | `p_detect ≡ 1` | information gain **without** detectability — literally the fleet's previous code, so a fair ablation by construction |
| (c) | retired hand-tuned constants (stand-off margins, visibility gates) | does the model beat the tuning it replaced |
| (d) | frontier / coverage exploration | the row reviewers expect; omitting it reads as avoidance |
| (e) | map-level information gain (ray-cast expected IG over the grid) | **the real rival** — information-driven but not object-centric |

Arm (e) is the honest opposing position — *"you don't need concept agents, just do
information-theoretic exploration on the grid"* — and beating it is what makes the object-level claim
non-trivial. Costs: (b) a config flag, (c) a config flag, (d) 3–5 days, (e) 1–2 weeks. None of it
inside anyone else's codebase.

**Primary metric:** nats·m⁻¹ to reach a stated Σ*. **Secondary:** fraction of executed viewpoints
returning no mask; pose and extent error against ground truth; time to Σ*.

**Falsified if** (a) does not beat (b) by more than the repeatability floor.

### E3 — the existence consumer *(this is what makes it "two consumers")*

Same harness, measuring the existence channel rather than the viewpoint one. Arms: `p_detect`-weighted
absence · unweighted miss-counting · the retired debounce counter.

**Metrics:** phantom lifetime; false removals under occlusion; identity churn.

⚠ **Design this to generate events, not to mine existing logs.** The phantom evidence in the corpus is
thinner than the flagship claim implies — 181 events fleet-wide, mostly *births*, with the covariates
zeroed on the largest contributor. Script the occlusions and the removals so that each arm sees the
same controlled sequence of disappearances and occlusions.

**Falsified if** weighting absence by `p_detect` does not reduce false removals or phantom lifetime.
In that case, re-title the paper around E2 alone (see above).

### E4 — ground truth in the real apartment

Hand-measure the poses and extents of ~10 objects with a tape measure. **About one day.** Without it
the real-robot section can only report self-consistency, which is much weaker, and a reviewer will ask
what the real numbers are measured against. State the procedure in the paper.

## Statistics discipline

Quote the **repeatability floor**: nine laps, cv **2.3%** on three metrics. Any claimed difference
smaller than that is not a difference. ⚠ **Do not use clearance as a metric** — its cv is **132%**.
Fix the arm list before running, report every arm including the ones that came out null, and pair
runs where the design allows it (the solver A/B managed 115 paired samples; match that standard).

## Page budget (8 pages including references)

| pages | content |
|---|---|
| 0.75 | introduction and claim |
| 0.75 | related work — active perception / NBV, detectability, existence estimation |
| 1.5 | the model: inverse perception model, marginalisation over the object belief, the two consumers |
| 0.5 | E1 calibration, with the envelope figure — the motivating result |
| 2.5 | E2 + E3: one table of arms × metrics, two figures |
| 0.5 | sim-to-real: what transferred and what did not |
| 0.5 | limitations, including the one-step horizon |
| 1.0 | references |

## Timeline to submission

- **September** — E0 harness; E4 ground truth (1 day); bound the CRDT growth.
- **October** — E1 on real images, table + fridge + one more; pilot E2 at N = 3 to shake out the harness.
- **November** — E2 and E3 at N ≥ 10. This is the month with no slack; protect it.
- **December** — write. Submit Dec–Jan.
- **1 Mar 2027** — IROS backstop if RA-L slips.

---

# PAPER 2 — the journal paper

**Why second, and why not simply longer.** Length does not fix what blocks the argument; the harness
does, and it is built for Paper 1. Going journal-first would commit six-plus months of evaluation
before knowing whether the detectability claim survives its own experiment.

**Spine — the two-room transition.** Crossing a door into a new room: activate long-term spatial
memory to move the abandoned room out and the new one in; decide *when* to transition while
localisation in the old frame degrades; estimate the new layout; set the new room's origin; register
the door just used as an object of the new room. Doors persist as nodes connecting room nodes, with
coordinates in both frames.

★ **The novelty argument is written by the prior paper itself.** *Applied Sciences* solves this
procedurally and then states: the inter-room alignment errors "could be corrected by constrained
optimisation of the rooms' centres, sizes, and orientations, **with the doors serving as fixed, shared
points between them. It is left for future work.**" That deferred item is what the current formulation
can supply, because here every belief carries a covariance. No reviewer has to be persuaded the delta
is real.

**How the pieces map:** the transition becomes a free-energy comparison of two generative models of
the same returns (not a geometric gate); "still attached to the old room" becomes a mixture over room
identity; the door becomes the inter-frame constraint with a covariance — the anchors idea one level
up; the room origin is a gauge choice; eviction to long-term memory is marginalisation plus volatility
decay; loop closure is data association over room identity.

**Second half — the term-by-term accounting.** The swap-one-rule design generalises to every place a
heuristic was replaced by a formalised term: gate vs per-frame covariance · miss-counter vs
`p_detect` log-odds · σ-floor vs Woodbury · range gates vs precision · asserted vs preintegrated
covariance (already null) · no-coupling vs meta-concept ΔF · naive vs cavity · MPPI vs PD (already
done, PD won) · M-of-N vs ΔF birth (not implemented). Collected into one harness on one tour, these
become a table nobody else publishes: **which parts of an Active-Inference formulation pay for
themselves, NULLs included.** An 8-page paper cannot hold that table; that is the argument for the
journal.

**Venue.** *Autonomous Robots* or *Robotics and Autonomous Systems*; IEEE T-CDS if the
active-inference framing leads. Not T-RO or IJRR, which would demand benchmarking this corpus has
never produced — and *Applied Sciences* accepting this material with an avowedly qualitative
comparison is the best evidence of where it currently lands.

**Timing.** Write it from the defended thesis, not alongside it.

---

## What already exists (do not re-measure)

- **Localisation vs Webots GT** — 72 mm median, 145 mm p90, 0.8° heading, 4 runs / 5468 poses;
  19.1 Hz, 0.8 ms median cycle over 1273 s and 352 m.
- **Object GT with calibrated uncertainty (bottle only)** — 13.6 mm position RMSE, 2.6 mm radius,
  **NEES median 0.46** over 2173 rows. The NEES is the important one: evidence the covariances are
  honest, which is exactly what a belief-based paper must show.
- **Solver A/B** — Gauss-Newton vs Adam, 66.6 → 9.0 ms (7.4×) at identical median loss, 115 **paired**
  samples.
- **Route-optimiser ablation** — clearance +29%, lap −11%, −14.5 s against 2σ of 4.9 s.
- Instrumentation: `ai2_log`, a 134k-row NBV log, `mission_metrics` ×57, `fit_envelope`,
  `concept_audit.sh`.

## What is missing

E0 harness · envelopes on real images beyond the fridge · a live run of detection-weighted NBV (built,
never executed live) · object GT in the real apartment · repeated-run statistics for the *perception*
loop (they exist for navigation) · a results chapter — ch.12 is robot and apartment description, §12.2
is still `[TO BE WRITTEN]`, and the WAF numbers are not in the thesis at all.

## What not to attempt

**Do not deploy external scene-graph systems as the headline.** ConceptGraphs is the most tractable
(Python, consumes a posed RGBD sequence exportable from both Webots and the robot) but it is
**passive** — no exploration policy — so it can only be compared along a fixed trajectory, which
removes our contribution. Hydra/Kimera and S-Graphs cost 3–6 weeks each, need their input formats and
tuning for an apartment they were not built for, and produce output that is not commensurable with
parametric object beliefs. **Cite, do not deploy.** Do not try to beat Hydra on dense reconstruction:
that is their turf and not our claim.

---

## PARKED EXPERIMENT — is the factorisation overconfident? *(designed, not run)*

Parked at the author's request; kept here so it can be picked up cold. No new hardware, no new agent,
no robot time — a recorded tour and an offline solve.

**Question.** A product of independent factors cannot represent the true posterior's correlations, so
its variances are systematically too small. This thesis factorises aggressively and *spends* its
covariances. If they are too tight, the epistemic drive is suppressed — a belief that believes itself
well determined proposes no action to improve itself — and every published covariance is optimistic.

**Oracle.** §ca-correlation option (a): one joint Gaussian over `ξ=(X,𝓡,θ₁…θₙ)` with dense
cross-blocks. Rejected as an *architecture* because it dissolves class ownership, which is exactly
what frees it to serve as an offline *reference*.

**Procedure.** Record one authored tour with the fleet live (LiDAR, masks, odometry, and every
per-instance posterior with capture stamps). Replay offline into a batch factor graph with **the same
factors**. Marginalise the joint result down to each published quantity and compare at the same stamp.

**Metrics.** Per-DOF `σ_fact / σ_joint`; the log-determinant gap `½ln(|Σ_joint|/|Σ_fact|)`, in nats and
so comparable with everything else; **NEES** against the joint mean — too tight shows as NEES above the
state dimension. Report **per DOF, not pooled**: which directions are wrong is the interesting part,
and yaw and extent are the candidates.

**Five traps, each of which invalidates it silently.** (1) Same model both sides — identical
linearisation points, robust losses and priors; FEJ on both. (2) **Gauge** fixed identically, or the
covariances are not comparable at all. (3) Same tangent frame and convention. (4) Sample immediately
after an update; the live covariance decays between observations, so a mid-decay sample measures
process noise. (5) Compare **marginals** — that is what agents publish — and validate the oracle first:
in simulation it must cover the ground-truth pose at its own stated covariance, or it is not an oracle.

**Related fix, independent of the measurement.** Where coupled quantities are factors of one free
energy, coordinate descent's update of any factor already takes expectations over the others, so the
cavity is not a safeguard but what the correct update is. Needing to remove a contribution by hand
indicates message passing rather than descent on a shared objective.

---

## Residual risks

1. **Sim-to-real for segmentation.** Detector failure modes in simulation are not the real ones, so a
   quantity fitted against simulated masks risks measuring the renderer. E1 exists to answer this.
2. **Prior art is crowded.** Detection-, semantics- and occlusion-aware NBV is busy; an ICRA 2026
   workshop paper on active 3D scene-graph generation already reports 2× objects against a frontier
   baseline. The delta must be visible in the abstract, and the abstract must state the delta against
   our own *Applied Sciences* paper explicitly — reviewers will look for it.
3. **Stability under batch.** Open phantom and churn defects, plus the unbounded-CRDT hang, are hostile
   to unattended 40-run batches.
4. **Thin phantom evidence.** See E3: generate the events rather than mining them.
5. **Competitor fairness.** If we implement arms (d) and (e), a reviewer will ask whether we
   implemented them well. Choose rules with unambiguous definitions, report each at its best
   configuration rather than its default, and release the harness.

## Dates

| venue | deadline | note |
|---|---|---|
| ICRA 2027 | 15 Sep 2026 | not attempted |
| **RA-L** | rolling | **target**: submit Dec 2026 – Jan 2027 |
| IROS 2027 | 1 Mar 2027 | fixed backstop; ~6.5 months is enough for the E-series above |

The RA-L already in flight is the **manipulation** paper — different topic, no self-competition, and
conversion work rather than new experiments. Finish that one first.

⚠ Verify all deadlines against the official CfPs before committing.
