# WM_PREDICTION_PLAN.md — learning to predict the working memory (2026-10-17)

Status: PLAN, nothing built. Sources: a design discussion in this session plus an independent
review by Fable (claude-fable-5-1) grounded in this repo. Where the two differ, the sharper
framing (Fable's) is adopted and noted.

## 0. The question and the answer in one paragraph

*Question (user):* generative world models over raw image/LiDAR streams look like a dead end;
LLMs work because language is structured; CORTEX's graph is structured too — so record sequences
of graphs + robot actions and train a transformer (masking / next-token) to predict the next
working-memory state, as a seed for planning.

*Answer:* **yes, but re-aimed.** CORTEX already *is* a world model: every concept agent has a
`predict()` and the graph stores sufficient statistics (mean, Σ, existence L, P(open)). Predicting
"the next tick" is already solved by those models and dominated by persistence. What the
hand-built factorisation lacks is (a) **cross-factor dynamics** (room/door/object/robot co-evolving
under an action), (b) **event-level, long-horizon prediction**, (c) **learned habits** (this door
is usually shut; that view lies). One-line reframing:

> The graph is not the thing to predict; the **outcomes of the robot's own commitments on the
> graph** are.

## 1. Corrections to the premise

- "Language is structured" is not the reason LLMs work — images are structured too. What matters:
  text is an already-compressed, low-nuisance code; trillions of tokens; discrete tokens make
  maximum likelihood well-posed *and multimodal*.
- Pixel/latent WMs work in their regime (DreamerV3, TD-MPC2, V-JEPA 2-AC, DINO-WM, Genie): large
  data or a simulator in the loop, short horizon. They are the wrong *level* here: one robot,
  hours of data, semantic long-horizon questions.
- The WAF pilot ("retrieval beat the trained head") is evidence about **data scale**, not about
  world models. The same trap awaits a graph transformer: at small data, counts and kNN win.
- The graph tokens are where the LLM analogy is strongest — but structure lowers the data need,
  it does not remove it, and the graph's interesting events are rare (tens per hour).

## 2. Design decisions

### 2.1 Representation — fence keyframes + edit stream, never fixed-rate stacked graphs
- **Keyframes at semantic fences** (the pattern `ltsm_agent/src/memory_store.h` already uses):
  affordance claim, affordance outcome, `current` room flip, mission step start/end, intention
  stamp. Dump via `G->write_to_json_file` (`dsr_api.h:558`) or a filtered `get_nodes()`.
- **Between fences: the edit stream.** Poll at 2–5 Hz and diff locally (repo-preferred, like
  `passage_harvest.h`), or connect the DSR signals (`update_node_attr_signal`,
  `deleted_node_signal`, … `dsr_api.h:849-858`, `SignalInfo` carries the writer agent) —
  **QUEUED only** (CLAUDE.md).
- **Training unit:** `(keyframe_k, action_k) → edits until keyframe_{k+1}`. Diff prediction is the
  graph analogue of next-token prediction and maps 1:1 onto the CREATE/UPDATE/REMOVE ops of
  `CONCEPT_AGENT_LIFECYCLE.md`. Fixed-rate snapshots would mainly encode fleet latencies (the CRDT
  has no global clock).

### 2.2 Tokenisation
- **Whitelisted schema (~30–40 attrs):** node type, class (`door`, not `door_24`), parent room, RT
  pose in the *current room* frame, Σ diagonal, existence L, P(open)/passable, `aff_*` state,
  outcome, age. Everything else is noise (161 of 292 published attrs have no reader anyway).
- **Identity:** TokenGT-style per-episode random orthogonal id embeddings — learn to *bind*, never
  memorise run-specific ids/names.
- **Continuous values:** quantised bins (Trajectory Transformer) or mixture heads, so symmetry
  flips and open/shut doors stay multimodal. σ is an *input feature* — uncertainty is data.
- **Time:** Δt tokens (TGN-style marked point process), never a fixed tick.
- **Actions as prefix tokens:** affordance kind + target + contract, mission IR op
  (`common/mission/mission_ir.h`), aggregated twist over the interval.

### 2.3 What to predict (priority order)
1. **`P(aff_outcome, duration | claim, state)`** — the 7 outcome words + time. Small, well posed,
   and exactly the missing pragmatic term in the selector's `G = λ·dist − gain`.
2. **Next event:** type + node + time-to-event (birth / death / `current` flip / door state change).
3. **Masked sub-graph reconstruction** → a **surprise score**: anomaly / phantom detector for the
   whole fleet. Earliest practical payoff, independent of planning.
4. **Mission-level queries:** room identity after `cross`, `explored(room)` after `explore` with
   budget b.

**Not predicted:** robot pose, Σ trajectories, anything inside one agent's factor — odometry and
the inferred volatility of `MODEL_HISTORY.md §3` will always beat a transformer there.

### 2.4 Uncertainty
Categorical heads, checked with ECE. Epistemic: a 3–5 model ensemble; its disagreement is the
WM's own information gain (Plan2Explore) — a second epistemic term beside the agents' ΔH.

### 2.5 Coexistence with the hand-built models — ONE-WAY
The learned model **reads the graph and writes only to the planner.** If it wrote back into the
estimators (e.g. via `Detection::birth_evidence`), the agents would re-emit its predictions and
the next training set would contain them: a self-sealing loop. The hand-built, falsifiable
version of that prior is `MODEL_HISTORY.md §4`'s `p_FA(place × bearing)`; build that first and let
the learned model earn the right to replace it (Phase 4).

### 2.6 It will learn the estimators' bugs — label them
Phantom births, symmetry flips, lag, Σ amnesia are regularities of the stream. Apply
`MODEL_HISTORY.md §4`'s **attribution rule** to labels: a death counts as a phantom only on a
confident disconfirmation. Predicting where the fleet fails is useful (verification pulls), but
only if it is labelled as such.

### 2.7 Active inference + mission language
Mission doc §2.5: a mission is a preference `ln C` added to the EFE. What's missing for a real EFE
over *policies* is a transition model at the affordance timescale. With target 1:
- `MissionInterpreter` scores a step sequence as Σ E[−ln C(outcome)] + expected duration;
- `on_fail: retry` budgets come from predicted P(satisfied);
- `MissionWithAffordances` (unimplemented, `controller/src/controller_mission.h:52,330`) gets its
  criterion: take an epistemic detour iff predicted gain > predicted delay cost.
Depth 2–4 beam/MCTS over affordance sequences — a simulator at affordance rate, not 10 Hz.

## 3. Data

- **Recorder:** a `recorder` role inside `ltsm_agent` (already owns a second `DSRGraph`,
  fence-aligned persistence, poll-based harvest, CSV episodes) — or a tiny new agent with a unique
  `[Agent] id`. Writes rotating edit CSV/parquet (`common/diag_log/rotating_csv.h`,
  `from_chars` + `imbue(classic)`), keyframe JSON at fences, the **selector's scores for every
  candidate** (propensities for off-policy weighting), the mission IR step, and sim GT
  (`robot_gt_*`).
- **Curation tags:** mission script (`round_trips.sh`), world variant, known-artefact runs,
  per-death "confident disconfirmation" flag.
- **Volume:** target 1 needs ~1 000 claims for a count model, ~10⁴ for anything learned. At
  30–60 claims/h that is 20–30 h of **automated, randomised Webots missions** (doors toggled,
  objects moved, varied start rooms). Event models: ~10⁵ events. Real data ≈10× scarcer →
  pretrain sim, fine-tune real; the domain gap is the *estimators'* behaviour on real sensors, not
  pixels.
- **Policy bias:** ε-random affordance selection in a fraction of runs; log propensities.

## 4. Evaluation — cheap falsifiers

- **Mandatory baselines:** persistence; marginal frequencies; per-(affordance kind, door) Beta
  count model; first-order Markov over events; kNN retrieval on a hand-built graph embedding (the
  WAF pilot's winner); agents' own `predict()` for anything continuous.
- **Metrics:** NLL / Brier + ECE (outcomes), CRPS (durations), ranked accuracy (next event).
  Split by **held-out runs**, never held-out frames.
- **Downstream:** `round_trips.sh` success/time with vs without the learned expectation (bar:
  try44 5/6, try45 3/6 — needs more trials than that for a CI).
- **Kill rule:** if count or kNN is within CI of the transformer on a target, the transformer is
  dead for that target.

## 5. Roadmap with go/no-go

| Phase | Duration | Deliverable | Go iff |
|---|---|---|---|
| **0 Record** | 1–2 wk | Recorder (keyframes + edits + actions + selector scores + GT); automated randomised sim mission loop; event-rate + vocabulary report | ≥1 000 claims reachable in automated sim time; event vocabulary < 50 types. Ships regardless (= `MODEL_HISTORY §5` step 1) |
| **1 Counts** | 2 wk | Beta / Markov / kNN models for outcome, duration, next room; count model wired into `set_goal_prior` / `on_fail` budgets | some baseline beats the marginal by a clear NLL margin — else the stream has no signal yet: fix estimators first |
| **2 Transformer** | 3–4 wk | TokenGT or GNN encoder + temporal transformer, keyframe+action → edits, masked pretraining, ensemble | beats Phase 1 on held-out runs AND is calibrated (ECE) |
| **3 Planning** | — | Rollouts in `MissionInterpreter`; ensemble disagreement as epistemic term; `MissionWithAffordances` criterion | round-trip mission success improves with CI |
| **4 Feedback** (speculative) | — | learned prior into agents via `birth_evidence` | `p_FA` exists and the attribution rule is enforced in labels |

Phase 1 is useful even if Phase 2 never wins; Phase 0's surprise-ready data also feeds the
anomaly detector (target 3) as a side product.

## 6. Literature anchors
Interaction Networks (Battaglia 2016), Graph Networks (2018), C-SWM (Kipf 2020), SlotFormer (Wu
2022); Trajectory / Decision Transformer (Janner, Chen 2021); TokenGT (Kim 2022), Graphormer;
TGN (Rossi 2020), DyRep, JODIE; spatio-temporal scene-graph forecasting (Cong 2021, Action
Genome); Hydra / Kimera-DSG / Khronos (hand-built dynamics); deep active inference with learned
transitions (Çatal 2020, Fountas 2020); Plan2Explore (Sekar 2020). Claimed gap: no published
system learns a transition model over a *multi-agent belief graph with each agent's uncertainty
and an affordance-outcome vocabulary*, conditioned on symbolic actions, feeding an EFE selector.
Defensible — **if** it beats trivial baselines.
