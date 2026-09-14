# Literature check — *Active Scene Graphs: a scene graph that acts to keep itself true*

Date: 2026-09-14. Target: `main.tex` (abstract, §II requirements + Table I, §III related work, Table II).
Method: web search + primary-source verification (arXiv abs/HTML, publisher pages, PMLR, workshop
programmes). Every citation below was opened; anything that could not be confirmed is marked
**UNVERIFIED**. New BibTeX is in `refs_candidates.bib` (64 entries, no key clashes with `refs.bib`).
`refs.bib` and `main.tex` were not touched.

Two claims in the brief were wrong and are corrected below:

* The "ICRA 2026 workshop paper on active 3D scene-graph generation reporting 2× objects vs a
  frontier baseline" is **two different papers conflated**. The 2× number is Modi et al.
  (arXiv:2605.18197) — whose arXiv metadata carries no venue, but which **is** listed as accepted
  paper #29 on the programme of the *ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy*
  (5 June 2026). The other ICRA-2026-workshop paper in this space is **SCOUT**
  (arXiv:2606.06721, *ICRA 2026 Workshop on Uncertainty in Open World Robotics*), which makes no 2×
  claim and is the more dangerous of the two for us.
* "Kimera-Multi ICRA 2022" is **ICRA 2021**; the T-RO 2022 paper is the journal version. There is no
  "Hydra 2.0" paper (the IJRR 2024 *Foundations* article is it) and no separate "Clio-batch" paper
  (it is an ablation inside the RA-L paper).

---

## 1. The term: prior uses of "active / actionable scene graph"

### 1.1 Headline result

**No prior paper, system or product is named "Active Scene Graph(s)".** The phrase is free as a
name. What exists is (a) a computer-vision *active-learning* sense, (b) a robotics
*active-construction* sense that emerged in 2026 and is one word away from ours, (c) a large and
growing *actionable / functional* family, and (d) a generic graphics usage ("the active scene graph"
= the currently bound render graph in Unity/OpenSceneGraph, not a research term).

### 1.2 Hit table

| # | Citation (verified) | Their sense of the term | Collides with "a graph that acts to keep itself true"? |
|---|---|---|---|
| T1 | Modi, Buoso, Averta, De Martini, **"RGB-only Active 3D Scene Graph Generation for Indoor Mobile Robots"**, arXiv:2605.18197, 18 May 2026; ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy, paper #29 | Robot selects viewpoints from the partially built scene graph instead of a fixed trajectory; RGB-only, feed-forward reconstruction + open-vocabulary semantics + LLM-driven exploration. Reports "more than twice as many objects as a geometric frontier-based baseline under the same exploration budget" (ReplicaCAD) | **PARTIAL COLLISION, and the worst naming collision.** Same phrase modulo "Generation", same direction (the graph decides where to look). But the graph is still a record: no per-node posterior gates anything, no offers, no self-model, one process. Must be cited and distinguished. |
| T2 | Modi, Buoso, Averta, De Martini, **"Fixed External Cameras as Common Prior Maps for Active 3D Scene Graph Generation"**, arXiv:2605.18184, 18 May 2026 | Same framework bootstrapped from fixed external RGB cameras; "graph-based active semantic exploration … toward regions of high semantic uncertainty"; +79 % initial object recall | Same partial collision as T1. Cite once, with T1. |
| T3 | Sun, Zhi, Heikkilä, Liu, **"Evidential Uncertainty and Diversity Guided Active Learning for Scene Graph Generation"** (EDAL), ICLR 2023 | *Active learning* for image scene-graph generators: which images to annotate, using evidential uncertainty. Reaches full-supervision performance at ~10 % of the annotation budget | **HOMONYM.** Different object (a dataset, not a robot), different actor (a human annotator), different goal (label cost). This is the CV sense the paper's TODO asks for; EDAL is the right representative. |
| T4 | Rosinol, Gupta, Abate, Shi, Carlone, **"3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans"**, RSS 2020 (already cited as `rosinol20203ddynamicscenegraphs`) | "Actionable" = the representation is at an abstraction level that *supports* planning (traversable places, hierarchy, agents) | **HOMONYM, but the canonical one.** "Actionable" here means *usable by* a planner; ours means *carries executable offers*. The disambiguating footnote should name this paper. |
| T5 | Jiang, Huang, Wu, Li, Garg, Nayyeri, Wang, Li, **RoboEXP**, CoRL (PMLR 270:3027–3052, 2025) | *Action-Conditioned* Scene Graph: **action nodes inside the graph**, object↔action edges encoding preconditions revealed by interaction ("open the cabinet door → tape") | **CLOSEST R1 COLLISION.** Actions really are nodes. But preconditions come from a heuristic geometric-cue algorithm (deterministic, not probabilities), there are seven hand-written primitives rather than one generic executor, ~8 s per LMM query, tabletop single arm. |
| T6 | Wang, Fermoselle, Kelestemur, Wang, Li, **CuriousBot**, RA-L 11(4):4993–5000, 2026 | "Actionable 3D relational object graph": relations that support *active interaction* (flip a box to uncover a toy) rather than only active perception | Homonym-plus: actionable = interactively explorable. No uncertainty anywhere. |
| T7 | Viswanathan, Patel, Saucedo, Satpute, Kanellakis, Nikolakopoulos, **SPADE**, IROS 2025 (arXiv:2505.19098); and Viswanathan et al., **xFLIE**, arXiv:2412.19571, 2024 | "Actionable multi-domain 3D scene graphs" / "actionable hierarchical scene representations" = layered graphs whose layers are consumed by a path/inspection planner in real time | Homonym. Actionable = plannable-over. |
| T8 | Nguyen, Pomarlan, Jongebloed, Leusmann, Vu, Beetz, **"Generating Actionable Robot Knowledge Bases by Combining 3D Scene Graphs with Robot Ontologies"**, IROS 2025 (arXiv:2507.11770) | "Actionable knowledge": USD-standardised scene descriptions translated into an ontology a cognitive controller can query | Homonym (knowledge-representation sense). Not Kinova/LLM-planning; this is the Bremen ontology line. |
| T9 | Saucedo, Patel, Saradagi, Kanellakis, Nikolakopoulos, **"Belief Scene Graphs"**, ICRA 2024, pp. 9441–9447 (DOI 10.1109/ICRA57147.2024.10611352); follow-up arXiv:2505.02405 | Nodes that are *beliefs about unseen objects* ("blind nodes"), from a GCN-learned correlation histogram (CECI) | **NAME COLLISION worth one sentence.** "Belief" there = a learned prior over what *should* exist; ours = a maintained posterior over what *does*, with existence log-odds. |
| T10 | Zapata, Pérez, Torrejón, Núñez, Bustos, **"Concept-Guided Exploration: Building Persistent, Actionable Scene Graphs"**, *Applied Sciences* 15(20):11084, 2025 (arXiv:2608.23650) | The authors' own prior work: asynchronous concept agents (room, door) cooperatively building an "actionable, human-interpretable" shared scene graph without a global metric map | **OUR OWN PRIOR USE.** Cite the *Applied Sciences* version; see §3.1 — the delta must be stated explicitly. |
| T11 | Zhang, Delitzas, Wang, Zhang, Ji, Pollefeys, Engelmann, **"Open-Vocabulary Functional 3D Scene Graphs"**, CVPR 2025, pp. 19401–19413; Rotondi, Scaparro, Blum, Arras, **FunGraph**, IROS 2025 | "Functional" scene graphs: interactive sub-parts (knobs, handles) as nodes with telic-affordance edges | Homonym ("functional", not "active"), but this is the family the survey groups under §3.3 "Modeling Functionality and Actionability" — cite the survey instead of enumerating. |
| T12 | **"ActiveStructure: Plane Scene Graph–Guided Active 3D Gaussian Splatting"**, Springer chapter, DOI 10.1007/978-3-032-37393-9_24, 2025 | A plane scene graph whose "topological health" scores viewpoints for active Gaussian-splatting reconstruction | Close variant of the phrase. **UNVERIFIED**: the Springer page is behind an IdP redirect; authors and book title could not be confirmed. Do not cite without checking. |
| T13 | Generic graphics usage (Unity glossary, Adobe Medium docs, OpenSceneGraph) | "The active scene graph" = the render graph currently bound | Not a research term. No citation needed. |

### 1.3 Recommended disambiguation footnote

Replace the `\todo` in the "The term" paragraph of §III with:

> \footnote{The phrase is used in two unrelated senses elsewhere. In computer vision, ``active''
> qualifies the \emph{learning} of an image scene-graph generator: which images to annotate next, as
> in the evidential active-learning framework of Sun et al.~\cite{sun2023evidentialactivelearning}.
> In robotics, ``actionable'' has since Rosinol et al.~\cite{rosinol20203ddynamicscenegraphs} meant
> that the representation is at an abstraction a planner can use, and a 2026 line on
> \emph{active 3D scene-graph generation}~\cite{modi2026rgbonlyactive,modi2026fixedexternalcameras}
> selects viewpoints from the partially built graph. We use ``active'' in the control sense and in a
> stronger one: the graph carries its own uncertainty, publishes what would reduce it, and is the
> thing that acts. Closest in name, ``Belief Scene Graphs''~\cite{saucedo2024beliefscenegraphs}
> attach a learned prior over objects \emph{not yet seen}; our nodes carry a maintained posterior
> over what is there, including its log-odds of existing at all.}

---

## 2. Table II — verification of existing rows, and the proposed revision

### 2.1 Existing rows, cell by cell

Grading rule applied: ● only if the paper's own described mechanism satisfies the Table I
falsifiable statement in full.

| Row | Current | Verdict | Why |
|---|---|---|---|
| 3D Scene Graph (Armeni) | all blank | **correct** | Offline batch construction over Gibson; a layered data structure, no robot, no uncertainty. |
| Kimera, Hydra | R2 ○, R5 ○ | **correct, but under-scoped** | R2 ○ is generous-but-defensible: uncertainty is real (covariance-weighted pose/deformation-graph optimisation, per-voxel Bayesian semantics) but *collapsed to the most likely label before a node exists*, and the pipeline is threshold-driven (8 m active window, 10 cm voxels, observation-count minima). R5 ○: real-time and parallel, single robot, single machine. **Missing:** Hydra-Multi and Kimera-Multi belong in the table as separate rows — Hydra-Multi is centralised ("the frontend, running at the control station, receives the latest scene graphs from the individual robots"), Kimera-Multi is genuinely peer-to-peer but shares a semantic **mesh** plus per-robot estimates, not one live graph. |
| S-Graphs | R2 ○, R3 ○ | **R2 ○ only if the follow-up is cited; R5 should be ○** | On the RA-L 2022/2023 papers alone R2 fails the "no hand thresholds" clause with published numbers: corridor/room plane lengths 4.0/15.0 m and 3.0/9.0 m, separations 1.5/3.0 m and 3.5/6.0 m, a tunable Mahalanobis matching threshold; Multi S-Graphs adds SC=0.35, ICP=0.07. R2 ○ is earned by **Millán-Romera et al. (RA-L 2025/26)**, which learns the concept factors' covariances *explicitly to remove hand-specified ones*. R3 ○ correct (per-concept layers and detectors, but modules in one SLAM process, no ownership or lifecycle). **R5 should become ○:** Multi S-Graphs is decentralised over ROS 2 with no central back-end, exchanging distilled graphs (84.9–93.3 % less data) — leaving it blank invites a correction from that group. |
| ConceptGraphs | all blank | **correct** | Greedy association against hand constants (voxel 2.5 cm, δ_nn 2.5 cm, δ_sim 1.1); node/edge precision is an evaluation metric, not a carried belief; the only affordance-like field is an LLM-annotated "can be pushed or traversed" consumed by a costmap. Guard the R1 blank with one sentence: a scalar property with no precondition and no executor fails R1's first clause. |
| Clio | R1 ○, R5 ○ | **correct; add an R2 footnote** | R1 ○ earned, not generous: a parallel process embeds a natural-language pick/drop command, selects the highest-CLIP-similarity node, then Dijkstra over place nodes and grasps (57 % grasp success). But two hard-coded verbs, chosen *outside* the graph. R5 ○: real-time onboard on a Spot-carried laptop, single robot. R2 blank is the **most contestable cell in the table** — a reviewer will say the Information Bottleneck gates what enters the map. Pre-empt: IB prices *task relevance of a description*, not the map's own posterior, and the compression level is a hand-set operating point δ̄ (plus α=0.23, θ_track=0.7, γ=0.4, θ_vol=8.0, θ_obs=3). |
| SayPlan | R1 ○ | **correct, and the best-justified ○ in the table** | Nodes literally carry `affordances: [turn_on, turn_off, release]` and `verify_plan()` forward-simulates against "predicates, states and affordances". Not ● because preconditions are discrete symbolic states, and each affordance word has its own hand-built skill ("a behaviour tree"), not one generic executor. Everything else blank is right: the paper's own limitation is that it "is constrained by the need for a pre-built 3D scene graph". |
| Concept-first rooms (ours) | R2 ○, R3 ●, R5 ● | **R1 should be ○, not blank** | The paper's own title claims *Actionable* Scene Graphs, and its abstract claims asynchronous concept agents on a distributed cognitive architecture. Leaving R1 blank on our own prior work while the title says "actionable" is the first thing a reviewer will notice. R3 ● and R5 ● are defensible (autonomous per-concept processes, CORTEX, no central planner). |
| This work | all ● | see §3 for what each ● must be defended against | R2's ● in particular must be stated as the *conjunction* (gates publication **and** prices action **with checked coverage**), because "uncertainty over a scene graph drives exploration on a real robot" is no longer novel as of October 2025 (Tang & Chaudhari) / June 2026 (SCOUT). |

### 2.2 Proposed revised Table II

Rows marked **[new]** are additions. Justifications in §2.3.

| System | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| 3D Scene Graph (Armeni) | | | | | |
| Kimera, Hydra | | ○ | | | ○ |
| **[new]** Hydra-Multi | | ○ | | | ○ |
| **[new]** Kimera-Multi | | ○ | | | ○ |
| **[new]** Khronos | | ○ | | | ○ |
| S-Graphs / S-Graphs 2.0 (+ Multi S-Graphs, + Millán-Romera) | | ○ | ○ | | ○ |
| ConceptGraphs | | | | | |
| Clio | ○ | | | | ○ |
| SayPlan | ○ | | | | |
| **[new]** RoboEXP | ○ | | | | |
| **[new]** MoMa-LLM | ○ | | | | |
| **[new]** SCOUT | | ● | | | ○ |
| **[new]** Active Semantic Perception (Tang & Chaudhari) | | ○ | | | |
| **[new]** Active semantic mapping (SSMI, Asgharivaskasi & Atanasov) | | ● | | | ○ |
| **[new]** SERKET / Neuro-SERKET | | ○ | ● | | ○ |
| **[new]** Pezzato et al. (active inference + behaviour trees) | ○ | ○ | | | |
| **[new]** Distributed SLAM + auto-calibration (Murai et al.) | | ○ | | ○ | ○ |
| **[new]** Federated inference (Friston et al.) | | ○ | | | ○ |
| Concept-first rooms (ours) | ○ | ○ | ● | | ● |
| **This work** | ● | ● | ● | ● | ● |

If space forbids 19 rows, the minimum honest set that survives review is: Hydra (+Hydra-Multi
merged), Kimera-Multi, S-Graphs, ConceptGraphs, Clio, SayPlan, **RoboEXP, SCOUT, SERKET,
Murai et al.**, concept-first rooms, this work. Dropping the four bold ones is what makes the
coupling claim contestable.

### 2.3 Justifications for the new rows (one sentence per non-blank cell)

**Hydra-Multi** (Chang, Hughes, Ray, Carlone, IROS 2023, pp. 10995–11002).
R2 ○: Hydra's SLAM-layer uncertainty plus GNC-based robust multi-robot pose-graph optimisation; no
node-level posterior. R5 ○: 3 simulated / 2 real Jackals with bounded frontend iteration (~100 ms),
but the frontend runs *at a control station*, "the backend optimization time increases as the size of
the graph grows", and a fixed team is assumed with no join/leave.

**Kimera-Multi** (Tian, Chang, Herrera Arias, Nieto-Granda, How, Carlone, T-RO 38(4):2022–2038).
R2 ○: measurement covariances throughout the pose graph with GNC/PCM outlier rejection, nothing at
map-entity level. R5 ○ (the strongest ○ in the table): "fully distributed and only relies on local
(peer-to-peer) communication", 3–5 robots online on CPU — but what is built is "a globally consistent
metric-semantic 3D **mesh** … faces annotated with semantic labels", each robot holding its own
estimate, so there is no single live graph and no join/leave.

**Khronos** (Schmid, Abate, Chang, Carlone, RSS 2024).
R2 ○: posed as MAP estimation, but decided by hard votes — hypotheses with fewer than τ_Z = 15
observations are rejected, absence needs c_ray = 60 % of rays over a 5 s window, and
appearance/disappearance times are taken as the middle of a window. R5 ○: 45.5 ± 9.2 ms per frame /
22.2 FPS on an i7-12700H laptop CPU; the two heterogeneous robots each ran Khronos independently with
no map sharing. Purely perceptual — nothing acts.

**RoboEXP** (Jiang et al., CoRL, PMLR 270:3027–3052).
R1 ○: action nodes live *inside* the graph and object↔action edges carry preconditions discovered by
interaction, executed in topological order — but the preconditions come from a heuristic geometric-cue
algorithm, and seven hand-written primitives do the executing.

**MoMa-LLM** (Honerkamp, Büchner, Despinoy, Welschehold, Valada, RA-L 9(10):8298–8305).
R1 ○: the action space is object-centric and graph-parameterised — `navigate(room,object)`,
`go_to_and_open(room,object)`, `close(...)`, `explore(room)` — dispatched to subpolicies, but each
subpolicy is concept-specific and preconditions are boolean symbols repaired by feeding the LLM
"The last action <<function-call>> failed"; hand constants throughout (voxel 0.075 m, 50-pixel
detection threshold, an "empirically tuned probability threshold" for doors) and no timing reported.

**SCOUT** (Mao, Ayoubi, Sharma, Hadžić, Andrews, ICRA 2026 Workshop on Uncertainty in Open World
Robotics, arXiv:2606.06721).
R2 ●: nodes "maintain fused geometry and posterior beliefs over open-vocabulary object labels" and the
planner selects viewpoints by balancing expected semantic-certainty gain against geometric coverage
gain and travel cost, explicitly revisiting ambiguous objects — the graph's own posterior is what
drives the action. R5 ○: online and real-time, one centralised pipeline.

**Active Semantic Perception** (Tang & Chaudhari, arXiv:2510.05430, v2 Jun 2026).
R2 ○: information gain of a candidate waypoint is computed directly on a multi-layer scene graph
(rooms, objects, walls, windows) against LLM-sampled plausible completions of unobserved regions, and
it runs on a real Unitree Go 2 — but the uncertainty is over *hypothetical* graphs rather than each
node's own calibrated posterior, and nothing is gated on it.

**SSMI** (Asgharivaskasi & Atanasov, T-RO 39(3), 2023; precursor arXiv:2101.01831).
R2 ●: every octree voxel carries a categorical posterior over semantic classes and candidate
trajectories are scored by a closed-form bound on the Shannon mutual information between that map and
future measurements — no threshold in the decision loop. R5 ○: real-time onboard, single robot. No
graph, no entities, no affordances: the counterexample is to R2 alone.

**SERKET / Neuro-SERKET** (Nakamura, Nagai, Taniguchi, *Front. Neurorobot.* 12:25, 2018;
Taniguchi et al., *New Generation Computing* 38:23–48, 2020).
R3 ●: a joint probabilistic model is *decomposed* into elemental modules, each inferring its own
latents, with global learning achieved purely by message passing between neighbours — no central
inference engine. R2 ○: what crosses a module boundary is a posterior (samples/hyperparameters), but
nothing is gated on it and no action is priced. R5 ○: "distributed" here means *development-time*
(modules written independently and composed), not N processes on M machines with bounded latency.

**Pezzato et al.** (T-RO 39(2):1050–1069, 2023).
R1 ○: a behaviour-tree leaf names a *desired state* rather than an action, and one active-inference
loop decides online which action achieves it under partial observability — preconditions resolved by
inference over beliefs, not boolean checks. R2 ○: precision-weighted free energy drives the choice,
but over a designed discrete state factor. Not ● because the offers attach to a hand-authored plan
tree, not to nodes of a perceptual representation any agent may have written.

**Murai, Alzugaray, Kelly, Davison** (RA-L 9(3):2136–2143, 2024).
R4 ○: sensor and marker **extrinsics are variables in the factor graph, inferred online jointly with
poses** — calibration parameters really are beliefs maintained by the same estimator that localises.
R5 ○: "scalable, fully distributed, and online", the factor graph itself being the shared structure,
robust to dropped messages and limited comms range. R2 ○: Gaussian belief propagation, so precision
is the currency — but nothing acts, so precision is never priced, and it is a pose/extrinsics graph,
not a semantic entity graph with a lifecycle.

**Friston et al., "Federated inference and belief sharing"** (*Neurosci. Biobehav. Rev.* 156:105500,
2024). R2 ○: what agents broadcast *is* a belief, so precision is the medium of exchange. R5 ○:
multi-agent joint inference — but by communication only, in simulation with three synthetic agents,
and **there is no shared data structure**: each agent keeps its own generative model.

---

## 3. The papers that most threaten the coupling claim

### 3.1 Zapata, Pérez, Torrejón, Núñez, Bustos — *Concept-Guided Exploration* (Appl. Sci. 15:11084, 2025)

Not a novelty threat so much as a framing one: it is the authors' own published work, and it already
claims "asynchronous concept agents that directly instantiate and manage semantic entities", two
concepts (room, door) as "autonomous processes within a cognitive distributed architecture" that
"cooperatively build a shared scene graph" through "active exploration and incremental validation",
producing an "actionable" representation with no pre-existing global metric map. That is R1○ + R3● +
R5● in one running robot system, under the same byline, with "Actionable Scene Graphs" in the title.
What it does **not** do: N concepts under a normative lifecycle (it has two, hand-built), offers
priced in a single currency, an execution-claim protocol, a self-calibrating body concept, or a
publication gate driven by calibrated uncertainty. §III already cites it, but the current wording
("a set of design intentions rather than a formalisation") understates it; the delta needs a sentence
naming what is new (pricing, existence log-odds, the lifecycle contract, the body concept, the
evaluation) or the preprint reads as a re-announcement.

### 3.2 Murai, Alzugaray, Kelly, Davison — *Distributed SLAM and Auto-Calibration with GBP* (RA-L 2024)

The single strongest counterexample found, because it takes **R4 and R5 together** — the two
requirements the paper's own related-work section implies nobody addresses. Sensor and marker
extrinsics are variables in a factor graph inferred online alongside poses, so calibration parameters
genuinely are beliefs updated by the estimator that localises; and the inference is fully distributed
over that graph by Gaussian belief propagation, robust to dropped messages and limited range, with the
graph itself as the shared object. What it does not do: select any action at all, so a parameter's
precision can never be priced into a decision and nothing is gated on it; carry semantic entities,
existence, or a lifecycle; or separate a media plane from the belief plane. The honest defence of R4
is therefore narrow and should be written narrowly: *nobody puts the robot's own kinematic and
extrinsic beliefs into the semantic scene graph as first-class nodes whose precision propagates into
the price of an offer.* A dedicated search for "self-calibration inside a scene graph" returned
nothing, so that sentence survives.

### 3.3 Mao, Ayoubi, Sharma, Hadžić, Andrews — *SCOUT* (ICRA 2026 workshop)

Concurrent, four pages, and a miniature of the paper's own thesis: an incrementally built scene graph
whose nodes hold posterior beliefs over open-vocabulary labels, and one planner utility that trades
expected semantic-certainty gain against coverage and travel cost, with explicit revisiting of
ambiguous objects. It is the only system found that earns ● on R2 *inside a scene graph*. It does not
carry executable offers or a generic executor (R1), is one centralised pipeline rather than
independent concept owners under a lifecycle (R3), has no self-model (R4), and runs on one machine
(R5). Cite it as concurrent work, say it is a workshop paper, and make R2's ● rest on the conjunction
— gates publication, prices action, and has *checked coverage* (the 2σ 70→85 % result) — because
"uncertainty over the graph chooses the viewpoint" is settled art.

### 3.4 Tang & Chaudhari — *Active Semantic Perception* (arXiv:2510.05430)

Threatens credibility rather than scope. A hierarchical scene graph, a principled information gain
computed on it, LLM-sampled hypotheses for unobserved regions, and a real legged robot across a
six-room apartment, from a strong group, and evidently the ancestor of the Modi et al. objective. It
establishes that "the representation's uncertainty drives exploration on real hardware" was done by
October 2025. Nothing in it is actionable in the R1 sense, concept-distributed, or self-calibrating,
and nothing is gated on the uncertainty — but after this paper, R2 cannot be claimed as a first.

### 3.5 Nakamura / Taniguchi — SERKET and Neuro-SERKET

Owns R3 as an idea, and it is the citation most likely to be raised by an active-inference reviewer:
a joint probabilistic model decomposed into modules that each infer their own latents and coordinate
purely by message passing, with no central inference engine — explicitly proposed so that different
researchers can develop modules independently and compose them. The distinction to write down is
precise: SERKET decomposes *one model's latent variables at design time*; an ASG decomposes
*ownership of entity instances in a shared persistent graph at runtime*, under a birth/update/death
lifecycle, with instances created and retired while the system runs. Taniguchi et al.'s 2023
*Advanced Robotics* review is worth citing in the same breath, but as the **gap statement** — it lists
uncertainty-driven learning, modular composition and body schema among the open frontiers.

### 3.6 Pezzato, Hernández Corbato, Bonhof, Wisse — *Active Inference and Behaviour Trees* (T-RO 2023)

Owns three-quarters of R1: a leaf node names a desired state instead of an action, and a single
active-inference loop decides online which action achieves it, handling partially observable initial
states — i.e. preconditions resolved by inference over beliefs rather than by boolean checks, with one
generic loop doing the choosing. What it lacks is exactly the ASG move: the offers hang off a
hand-authored tree (a central plan structure) over a designed discrete state factor, not off nodes of
a shared perceptual representation. The one-line separation worth using verbatim: **offers attach to
the representation, not to a plan.**

### 3.7 Assembly risk (the real threat)

No single paper scores more than two requirements, and **no verified paper scores R4 at all** except
Murai et al. at ○. But a hostile reviewer can assemble the set from four: Murai (R4+R5),
SERKET (R3), Pezzato (R1), SCOUT/SSMI (R2). The claim should therefore not be "nobody has done any of
this" but the conjunction-in-one-system claim the introduction already gestures at: *each requirement
has prior art in isolation; none of these systems meets more than two; and none makes the same
posterior serve as publication gate, action price and body-parameter belief inside one representation
that N agents on M machines read and write at sensor rate.* Two further supports are worth citing:

* **Rotondi et al., "3D Scene Graphs: Open Challenges and Future Directions"** (Annual Review of
  Control, Robotics and Autonomous Systems vol. 10; arXiv:2606.19383) — thirteen authors including
  Carlone, Hughes, Valada, Tombari, Arras. It states the gap for us in its own words: "most
  approaches maintain a single estimate of the world state, despite the inherent uncertainty arising
  from partial observability, ambiguous data association, and future scene evolution. How to represent
  and update beliefs over past, present, and future states … has received little attention." It also
  files distributed/collaborative scene graphs under §5.5 "Emerging Applications", i.e. not yet
  solved. This is the single most valuable new citation in this report.
* **Millán-Romera et al.** (RA-L, arXiv:2409.11972) — already claims *learned* covariances replacing
  hand-specified ones inside a 3D scene graph. This retires a bare "no hand thresholds" novelty claim;
  R2 must be argued on what consumes the posterior, not on the absence of constants alone.

---

## 4. Corrections and loose ends

| Item | Action |
|---|---|
| `bavleSGraphs20HierarchicalSemantic2025` in `refs.bib` | Now published: RA-L 10(12):12461–12468, 2025. Replace or add `bavle2025sgraphs2`. IEEE DOI unconfirmed (Xplore doc. 11197654 is paywalled to automated fetches) — **UNVERIFIED**. |
| `maggioClioRealtimeTaskDriven2024` | Now published: RA-L 9(10):8921–8928, DOI 10.1109/LRA.2024.3451395. Nine authors. |
| `pezzatoMobileManipulationActive2025` | Now published: IWAI 2025 revised selected papers, CCIS 2857, Springer 2026, ch. 21, DOI 10.1007/978-3-032-16955-6_21. |
| `gu2023conceptgraphsopenvocabulary3dscene` | Published ICRA 2024, pp. 5021–5028, DOI 10.1109/ICRA57147.2024.10610243. |
| `lanillosActiveInferenceRobotics2021` | arXiv id 2112.01871 in `refs.bib` is correct. **No journal version exists** — do not upgrade the venue. |
| `zapatacornejoConceptGuidedExplorationBuilding2025` | Entry is correct (Appl. Sci. 15(20):11084, DOI 10.3390/app152011084). Note the arXiv posting 2608.23650 is dated Aug 2026, after the journal version; cite the journal. |
| "Pezzato, *Active inference based control for robot manipulators*" | **UNVERIFIED — no such title.** The real items are the RA-L 2020 adaptive controller and the IWAI 2020 fault-tolerant-control paper. Do not cite the phantom. |
| NodeSLAM as uncertainty-driven next-best-view | **Wrong.** "next best view", "information gain" and "active perception" appear zero times in the paper; its uncertainty is a per-pixel rendered-depth weight on the render loss, and keyframes are admitted by a 13° hand threshold. Cite it, if at all, as the canonical *passive* object-level shape-uncertainty SLAM and as an example of the hand gate R2 forbids. |
| SSMI page range 39(3):1910–1928 | From a search snippet plus the IEEE DOI; IEEE blocks fetching. **Verify against a library copy.** |
| Belief Scene Graphs pages 9441–9447 | From a search result reporting the IEEE record; consistent with DOI 10.1109/ICRA57147.2024.10611352 but not read directly. Low risk. |
| Millán-Romera et al. volume/pages | arXiv v4 says accepted at RA-L; volume and pages not yet assigned. **UNVERIFIED** as a full journal reference. |
| `rosinol20203ddynamicscenegraphs` "actionable" | Confirmed: the RSS 2020 title is "Actionable Spatial Perception with Places, Objects, and Humans". This is the term's origin in robotics and belongs in the disambiguation footnote. |
| ActiveStructure (T12) | **UNVERIFIED** metadata. |
| MA3DSG (arXiv:2602.04152, multi-agent 3D scene-graph generation) | Seen in a reference list only; metadata **UNVERIFIED**. Check before citing. |
| xFLIE venue | arXiv-only as of this check (2412.19571); SPADE is the peer-reviewed one (IROS 2025). |
