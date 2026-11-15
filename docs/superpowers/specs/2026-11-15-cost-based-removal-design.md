# Cost-based removal: keep or remove an object by what each mistake costs

Date: 2026-11-15. Status: rev 2 APPROVED by the owner 11-15 (rev 2.1: four corrections made with the plan, marked ★) (owner picked approach A and the cost unit, 11-15; then "Do 1": fix the absence
evidence first, and A for that fix too). Nothing built.
Branch: `feat/cost-removal` (worktree `active_inference.worktrees/cost-removal`, from main 91784311).

## 1. Intent (owner)

Removal must be a fair comparison between what it costs to keep a model the sensors argue against and what it costs to
remove it — not a probability threshold plus a debounce. Owner, 11-15, on what counts as evidence:

> "a model the sensors argue against. If the object should had received positive evidence and did not for a while under
> the right conditions, that is also evidence for removal. If it is out of view, occluded or far away, that is not
> evidence for removal."

Shared in `common/existence_belief`, opt-in per agent (owner order 11-14: this follows the existence = occupancy × class
work, before the tracker rework and the bedroom-rig metaconcept).

## 2. What the code does today (verified 11-15 on main 91784311)

- **Evidence** (`rc::exist::ExistenceBelief`, `common/existence_belief/existence_belief.h`): L += p_vis · log-ratio per
  channel per cycle, clamped ±`logodds_max` (4). p_vis = P(this look could have resolved the object | it exists), so an
  out-of-view / occluded / far look adds ≈ 0. The RULE is the owner's; the LiDAR channel's SCORING is wrong (next bullet).
- **The LiDAR carve scores a real object's own returns as absence** (verified offline 11-15,
  `docs/superpowers/results/2026-11-15-carve-replay.md`). `carve_box` tests the box in 2-D: a return is "surface" when it
  lands near the side face the beam entered in x–y, else "interior" (= against, for an opaque solid). The Helios looks
  down on furniture below its 1.11 m, so most returns land on the TOP face and are scored against: on the real bed's drop
  cycles 1640 of 1744 against-units were returns from the top of the mattress (ΔL −151/cycle; a 3-D test gives +405).
  Second, the through term leaks on parts thinner than the surface σ (a 10 cm headboard). Third, a fitted cuboid is
  mostly air for the toilet (−6/cycle even in 3-D; +169 with bowl + pedestal + tank). This, not a missing debounce, is
  why real objects sit at L = −4, so the costed line cannot be switched on before it is fixed.
- **Decision** (`decide_removal`, same file): condemned when P(exists) < `removal_prob` (0.12 ⇒ L < −1.99); removed when the
  debounce `streak` (Σ p_detect over condemned cycles) reaches `required_observations` (15 ideal looks, or a confidence-scaled
  count when `confidence_scaled`). Reports `stalled` / `unfalsifiable` lines. Its comments record that the debounce has
  repeatedly hidden bad likelihoods.
- **Correlated frames.** `ExistenceBelief::marginal_weight` discounts the k-th same-sign observation by N_eff with
  ρ_eff = ρ · max(1 − p_detect, stationarity), where `stationarity` comes from `note_viewpoint` (geometry-scaled, no tuned
  constant). **With ρ = 0 the discount is off entirely, stationarity included.** countable_concept calls `note_viewpoint`
  but ships `Model.ExistenceFrameCorrelation = 0` (all three classes); table 0.327 and hood 0.477 are measured.
  So for countable today a parked stare counts every frame as a new look, and only the 15-look debounce stands between
  that and a removal.
- **countable_concept** (nightstand 29, bed 30, toilet 31) removes through two debounces (spec 2026-11-14 existence =
  occupancy × class): `inst.existence` (occupancy: LiDAR + silhouette, `countable_existence.cpp:479`) and
  `inst.class_belief` (L_cls = logit π + Λ − singleton, set on measured cycles, `countable_class_evidence.h:157`), each
  through `decide_removal` with the same policy; the class factor advances only when it argued against
  (Λ − singleton < 0). The published existence is p = p_occ · p_cls.
- **Prior art: door** (`door_concept/src/core/door_decision.h`): remove when L < −ln(C_delete / C_keep), never for a door
  being crossed; no streak, timer or clamp. Costs in SECONDS of the robot's own experience, `CostEstimator` = running median
  per kind, persisted in `etc/door_costs.csv` (`kind,seconds,source`), written by `tools/door_costs_from_logs.py`; compiled
  default C_keep 30 s, C_delete 88.25 s (median of 7 logged remove→re-found intervals); a file with C_delete ≤ C_keep is
  refused as a whole. Door's view ledger makes one viewpoint one trial.

## 3. Goals

- **G0. Absence evidence comes from the object's real shape (done FIRST, own commits).** For an opted-in agent the LiDAR
  carve is a 3-D ray test against the class's COMPOSED shape (owner picked A, 11-15): a return on any face of any part
  (top included) is surface ⇒ FOR; a return deep inside a part is interior ⇒ AGAINST; a beam that crosses a part and
  returns beyond it is through ⇒ AGAINST; a beam that crosses only space the shape does not fill says nothing. The
  through term counts only the part of a return a surface hit does not already explain (thin panels).
- **G0b. The object's position uncertainty is marginalised as ONE common-mode shift** (owner picked (a), 11-15; the
  replay found a wall-straddling phantom kept alive because each face test was broadened independently by the footprint
  σ, which ages to 0.17–0.6 m while masks are gated — `carve_replay --explain`, d550a081). The footprint offset
  δ ~ N(0, Σ_xy) (the belief's own covariance) moves ALL parts together; each face test uses the sensor σ only; the
  increment is the marginal ΔL = log Σ_k w̃_k exp(ℓ_k) over quadrature nodes δ_k, with ℓ_k the carve's log-odds at the
  shifted box and w̃_k ∝ (quadrature weight) · π_place(δ_k). π_place is the class's own wall no-cross factor (the fit's
  `wall_no_cross_precision` P, log π = −½·P·d³, d = how far the shifted footprint crosses the room outline): a shift that
  would put the object inside a wall is ruled out by the prior the class already has. No new constant; the quadrature
  order is numerical, not a model parameter. ★rev (owner 11-15, after the xy-only replay turned the real bed and nightstands
  negative: their fitted H sits 1.4–3.2 cm above the true top, within the belief's own std_H): the common-mode variable is
  the WHOLE box state θ = (cx, cy, H, w, h, yaw) ~ N(μ, Σ), Σ the belief's full 6×6 covariance, integrated with the
  spherical-radial cubature rule (2n = 12 points μ ± √n·L·e_i, L = chol Σ, weights 1/2n — no free parameter). A point with
  a non-positive size (H, w or h ≤ 0) is outside the state's support: its weight is 0. Each point also carries π_place.
  Its validity depends on Σ being honest; the replay reports this, and (b) is what keeps Σ honest.
- **Later (owner, 11-15): (b)** the LiDAR returns should UPDATE the belief's position, not only test existence — then
  Σ reflects what all sensors know instead of ageing while the LiDAR has its best view. A fit-side design, not this change.
- **G1. The costed line.** For an opted-in agent: remove when L < −ln(C_delete / C_keep). No `removal_prob`, no debounce,
  no streak, no timer. With the default costs the line is L = −1.08 (p ≈ 0.25).
- **G2. Costs are seconds of robot time, per class, learned from logs** (owner 11-15: "keep robot time only").
  - C_delete = seconds from a removal of a REAL object until that object is born again at the same place.
  - C_keep = seconds of robot time spent on a phantom while it exists: looks the epistemic planner spends on it, approaches
    and detours it causes.
  - No cost for informing the human wrongly, no memory-pollution term (owner: robot time only).
- **G3. Repeated frames count once.** Without a debounce, correlated frames must not add up to many looks. An opted-in agent
  MUST carry a ρ measured from its own logs (the table/hood procedure), so the existing stationarity term acts; ρ = 0 with
  the costed decision is refused at config load.
- **G4. The owner's evidence rule holds**: only looks that could have resolved the object move L; p_detect and the
  out-of-view HOLD are unchanged. What changes (G0) is which beams count as "resolved and absent".
- **G5. Opt-in.** Agents that do not opt in behave exactly as today, byte-for-byte in their decisions.
- **G6. countable_concept opts in first**, all three classes.
- **G7. Same reports.** The `stalled` / `unfalsifiable` lines survive (they report evidence, not the debounce); one log line
  per removal states L, the line, both costs and their sources.

## 4. Non-goals

- Door is not migrated (it keeps its own `door_decision.h`; moving it onto the shared code is a later change).
- Table, chair, bottle, cabinet, hood, refrigerator, residual stay on today's rule (each opts in later, after its absence
  channels are checked).
- No "value of looking" term in the decision (owner picked A over B; the epistemic planners already bid for looks).
- No change to the p_detect model or the out-of-view HOLD. `carve_box` stays bit-identical for every caller that does not
  opt in (table, cabinet, residual, door, …).
- Part sizes are NOT estimated here: the parts are laid out from the fitted box (w, h, H) and manifest fractions.
  Estimating part sizes is Step 2/3 of the one-binary plan.
- The silhouette channel's shape is out of this change unless the live diagnostic (§9) shows it removes a real object.
- No change to birth.

## 5. Anti-goals

- A real nightstand, bed or toilet removed in a live run where today's rule kept it.
- A threshold or debounce re-entering by another name (a minimum dwell, a streak floor, a cooldown).
- Costs picked by hand and labelled as measured.
- The same evidence counted N times (a parked stare at 15 Hz charging 15 looks a second).
- A phantom kept alive by the new carve: a box over empty floor or over a wall must still collect absence (the reason
  the interior term exists, [[existence-policy-unification]]).

## 6. Approaches (owner picked A, 11-15)

- **A. Door's costed line, shared (picked).** Decision on L and two measured costs; debounce gone; correlation handled by a
  measured ρ.
- B. A plus the value of looking (keep / remove / look; removal waits while a look is worth more). Closer to full active
  inference but couples removal to the planner, which already collects most of that value.
- C. Costs set the threshold, debounce kept (p* = C_keep / (C_keep + C_delete)). Smallest, but keeps the double count.

### Approaches for the evidence fix (owner picked A, 11-15)

- **A. Composed shape (picked).** The manifest lists the class's parts; the carve tests each part in 3-D. Serves the
  bathtub and the later shape work as well.
- B. A solid z-band per class. Quick, fixes bed and nightstand, cannot describe the toilet.
- (A 3-D test against the plain cuboid is the common core of both; the replay shows it fixes bed and nightstand but
  not the toilet.)

## 7. Constraints

CLAUDE.md rules: no thresholds (the costed line is a decision boundary derived from costs, not a tuned gate); and/or/not;
`std::from_chars` for the cost file, written through `imbue(std::locale::classic())`; -j8 capped builds; SIGTERM only;
Queued signals; type-attributed attributes, **no new graph attribute**. CONCEPT_AGENT_LIFECYCLE.md: absence weighted by
P(detect), occlusion HOLDS; one audit row update (countable REMOVE column). Commits local on `feat/cost-removal`; merge or
push only when the owner asks. Webots is shared: claim it from the chat and room sessions, FREE after.

## 8. Decided (my calls, owner may change)

- **Where:** a new header `common/existence_belief/existence_cost.h` holding `Costs`, `CostEstimator` (generalised from
  door's: kinds `keep` / `delete`, running median, `load_or_default`, refuses C_delete ≤ C_keep) and
  `decide_costed(L, costs) -> {remove, line}`. `RemovalPolicy` gains `costed` (default false) and the costs; when `costed`
  is false `decide_removal` is unchanged.
- **Cost file per class:** `<class>_concept/etc/existence_costs.csv`, same format as door's (`kind,seconds,source`);
  missing / invalid ⇒ a loud warning and the compiled default, which is door's logged pair (30 s, 88.25 s) labelled
  `compiled_default:door_logged` — a borrowed measurement, said so in the source column.
- **Measuring:** a script `tools/existence_costs_from_logs.py` reads a run folder's `<cls>_ai2_log.csv` + agent logs.
  C_delete samples: a removal followed by a birth of the same class within the object's own footprint (½·max(w, h)) — the
  interval in seconds; a removal never followed by a re-birth at that place is NOT a sample (it may have been a phantom).
  C_keep samples: per removed instance that was never re-born (a phantom by this definition), the seconds the epistemic
  planner targeted it plus the seconds the controller spent approaching it, from birth to removal. The script prints n per
  kind; with n = 0 the class keeps the default.
- **Two factors, one line.** countable decides on the product: logit(p_occ · p_cls) < line. Kept from the 11-14 design: a
  removal executes only on a cycle where at least one factor was measured AND argued against (occupancy: its last increment
  < 0; class: Λ − singleton < 0). The class factor's separate debounce goes.
- **ρ for countable:** measured per class from objects_live3–6 `<cls>_ai2_log.csv` with the table/hood procedure
  (P(miss | previous miss) on cycles with p_detect > 0.3), written to each class config. If a class has too few miss pairs
  to measure, the builder reports it and that class does not opt in.

- **Carve, shared:** a new `rc::exist::carve_parts(origin, sweep, cx, cy, yaw, parts, surface_sigma, sm) -> Evidence` in
  `common/existence_belief/existence_belief.h`, next to `carve_box` (untouched). `parts` = boxes in the object frame
  (centre offset, size, z0..z1). Per beam, decided against the UNION: surface if the return is within σ of the first
  entry into any part (any face, top included); interior if the return lies inside a part deeper than σ from that part's
  entry; through if the beam exits every part it entered and returns beyond, weighted by (1 − surface weight).
  `n_reached` is the reached mass over parts, so `coverage` keeps its meaning.
- **Parts in the manifest:** a `[shape.parts]` table per class, each part as fractions of the fitted (w, h, H) plus
  its z band; default = one part = the whole box (so a class without the table behaves as a 3-D cuboid). First tables
  (★2.1: the fractions are MEASURED numbers, as the manifests already carry; no Webots asset is copied):
  nightstand body (z 0.66..1.0 of H) on four legs; bed mattress (z 0.38..1.0 of H) + footboard + the existing
  `[prior.headboard]` panel; toilet pedestal + bowl + tank; each `why` says where its numbers were measured.
- **Q2 (my call): costs.** Until the cost script has samples for a class, its line uses door's logged pair, labelled
  `compiled_default:door_logged` in every removal log line.
- **Q3 (my call): a rival's inhibition alone never removes.** The class factor counts as "argued against" only when its
  own shape evidence says so: its Λ < 0 (★2.1: the earlier "Λ − singleton < 0" contradicted this rule). A cycle on which the class factor fell only because a rival class
  claimed the object (the singleton term) does not execute a removal, even if the product is below the line.
- **Order:** G0 (carve) commits first and runs live alone (objects_live7a, today's debounce); the costed line follows
  (objects_live7b).

## 9. Pass

- **Unit:** `decide_costed` line = −ln(C_delete / C_keep); the cost file round-trips under `LANG=es_ES.UTF-8`
  (`setlocale(LC_ALL, "")` in the test); a C_delete ≤ C_keep file is refused; `costed = false` reproduces today's verdicts
  on a recorded trajectory; a parked stare (stationarity 1, measured ρ) cannot remove a confident object (L = +4) in under
  1/ρ looks' worth of evidence; an out-of-view instance (p_vis = 0) is never removed.
- **Carve unit:** a return on the TOP of a box seen from above is surface (FOR); a beam passing above the box adds
  nothing; a return on the front of a 10 cm panel with σ = 5 cm is not through; a box over empty floor and a box drawn
  over a wall both get net AGAINST (phantoms still die); a single-part manifest reproduces the 3-D cuboid.
- **Carve replay** (the 11-15 script, rebuilt as `tools/carve_replay.py`): on objects_live6's drop cycles the new carve's
  ΔL is ≥ 0 for bed_10, nightstand_6, nightstand_7, toilet_6, and < 0 for synthetic phantoms (each real box moved onto
  empty floor, and onto a wall) (★2.1: bed_8 lives in room_1, whose frame the logs cannot tie to the world).
- **Replay** (★2.1: on objects_live7a — the live4–6 traces carry the broken carve, under which every real object sits at
  −4): the countable traces through the new decision: every real object kept, and each removal the
  old rule made (bed_8 occupancy removal, live5) still made, with the time shown.
- **Silhouette diagnostic (live7a)** (★2.1: the share of silhouette vs LiDAR absence on each real object's drop cycles,
  from the existing columns — a split by projected parts needs a projection the code does not have); if a real object's L still falls on silhouette absence, that
  channel gets the parts too (a follow-up, owner decides).
- **Live 7a** (carve only, same tour as objects_live6): the real bed, both nightstands and the toilet never go below
  L = 0 on LiDAR evidence; ~10 Hz.
- **Live 7b** (costed line): bed, both nightstands, toilet never removed; a phantom removed no later than today; ~10 Hz.

## 10. Left to the builder

- How the planner's targeted seconds and the controller's approach seconds are read from the logs (which columns / lines).
- The test layout and the replay harness.
- Whether `RemovalPolicy` carries the costs or a reference to the estimator.
- The exact part fractions per class (from each class's typical mesh) and how `carve_parts` resolves overlapping parts.
