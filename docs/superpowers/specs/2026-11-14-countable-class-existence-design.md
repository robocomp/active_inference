# Existence of a countable object = something is there × it is this class

Date: 2026-11-14. Status: DRAFT after builder check, awaiting the owner. Nothing built.
Scope: `countable_concept` (nightstand, bed, toilet). Other agents untouched.

## 1. Intent (owner)

A blackboard on a stand in room_1 was taken for a bed (objects_live4, bed_6). The owner wants it removed by the bed
concept's own shape knowledge for now ("shape check is enough for now, whiteboard later"), with approach A:
**split existence into P(something occupies this space) × P(it is this class | its shape)**, and the birth
plausibility checked too. Room-kind priors (a bedroom-rig metaconcept with the cavity rule) and a whiteboard concept
are later, separate designs; the tracker rework is a separate design that follows this one.

## 2. What the code does today (verified 11-14, objects_live4 log + `countable_concept/src`)

- **One existence log-odds L per instance** (`rc::exist::ExistenceBelief`, clamped to ±`Model.ExistenceLogOddsMax` = 4).
  LiDAR occupancy and the camera silhouette integrate into it.
- **The shape evidence** is `CountableBelief::countable_log_evidence_ratio` (`countable_belief.cpp`), log p(θ̂ | class)
  − log p(θ̂ | other furniture) on the fitted point estimate: aspect + footprint size + height terms. It is summed
  per measured frame into `plaus_evidence` (clamped ±8), and reaches L only through
  `singleton_existence_deltas`: gain·tanh(plaus_evidence/8), gain 1.5, minus a singleton inhibition
  (`specificworker_lifecycle.cpp:534 apply_countable_filter`).
- **bed_6:** LLR −39…−45 from the first fit, plaus_evidence at −8 at once, ΔL −1.14 per cycle; LiDAR occupancy
  (occ 567 vs free 186) held L at +4 (p 0.98) for ~13 s; p then fell to 0.45 and 0.02 and the node stayed ~40 s
  more because no channel measured it (out-of-view HOLD, by design) until the robot left the room.
  Real bed bed_7: LLR median +5.06.
- **Both height terms are one-sided, inherited from the refrigerator** ("tall → plausible, short → not"):
  - fitted LLR: log(2·σ((H − 0.35)/0.10)), at most +0.69, never against a too-tall shape;
  - birth: `candidate_plausibility` is HEIGHT ONLY, σ((z-range − 0.35)/0.10). A 1.9 m cloud scores ≈ 1, the
    maximum, so the blackboard passed birth at full strength. The bed manifest's own `[prior.height]` (mean 0.65,
    std 0.10) is not used by either.
  - Footprint is deliberately not used at birth: a front-only view is a vertical face whose footprint is a line.
- The fitted LLR ignores the posterior covariance: a barely-seen shape gives the same evidence as a well-seen one,
  which is why it was summed per frame, then had to be clamped and squashed.

## 3. Goals

- **G1. Two factors.** Each instance carries
  - `occ` — the existing `ExistenceBelief`, fed unchanged by LiDAR and silhouette: "something occupies this box";
  - `cls` — the class log-odds: logit(π_cls) + Λ, where π_cls is the class prior (how often this class's label on
    a mask is really this class) and Λ the shape log-evidence ratio of the **current fitted posterior**.
  Published existence p = p_occ · p_cls (independent factors). The node's existence attribute and every consumer
  see p, as now.
- **G2. Λ from the posterior, not an accumulator.** The fitted belief already accumulates the data, so Λ is a
  function of its current mean and covariance: log N(θ̂; μ_cls, Σ_cls + Σ_post) − log p_other(θ̂ | Σ_post), over the
  shape variables (w, h, H). A poorly seen shape has a wide Σ_post, so Λ is weak by construction; no per-frame sum,
  no clamp, no tanh. The class side takes μ, σ from the manifest (`[prior.footprint]`, `[prior.height]`); the
  "other furniture" side is a broad density (procedure default in `common/`, the same for every class).
- **G3. Two-sided height** everywhere (fit and birth): a shape far taller than the class prior is evidence against,
  as much as a far shorter one.
- **G4. Birth plausibility on the same scale.** At birth there is no fit: score the raw cloud's z-extent against
  the **composed shape's** expected extent — the tallest declared part (bed: headboard 1.25 m; nightstand and
  toilet: H) — as a log-evidence ratio, with the cloud's extent uncertainty. Footprint stays out of the birth score
  (the partial-view reason in §2 still holds). It multiplies the birth evidence as today.
- **G5. Removal** runs on the product, through the same `rc::exist::RemovalPolicy` / `decide_removal`, and only on
  measured cycles (the HOLD-when-unseen rule stays). The "a channel may only execute a removal it argued for" rule
  stays: the class factor may remove only when its own Λ is below the removal side.
- **G6. One log line per instance per measured cycle** with occ L, cls L, Λ, p — replacing `[bed-filter]`.

## 4. Non-goals

- No room-kind prior, no rig, no whiteboard concept (later, owner 11-14).
- No change to `common/existence_belief` or to the other agents' existence.
- No tracker change (next design).
- Not the bed's close-range height rise (H 0.65 → 0.80, Step 2 / footboard).

## 5. Anti-goals

- A height gate or a "too tall ⇒ reject" threshold. Height enters only as a density.
- Λ summed per frame again (the same evidence counted N times — the fleet's recurring defect).
- A real bed, nightstand or toilet removed or born later than in objects_live4.
- Rate regression (agents stay ~10 Hz).

## 6. Approaches (owner picked A, 11-14)

- **A. Two factors (picked).** Keeps "something is there" honest — LiDAR truly sees the blackboard — while "it is a
  bed" can be near zero. Cost: one contract change for the countable classes (one lifecycle-audit row).
- B. One L, shape evidence added in full per view: smaller, but the two questions stay tangled and LiDAR and shape
  keep tugging the same number.
- C. Reject at birth only: one early look decides the class; close to a gate.

## 7. Constraints

CLAUDE.md rules (no thresholds; and/or/not; `std::from_chars`; -j8 capped builds; SIGTERM only; Queued signals;
type-attributed attributes — **no new graph attribute**: p goes on the existing existence attribute).
CONCEPT_AGENT_LIFECYCLE.md: birth stays tracker-only; occlusion HOLDS; absence weighted by P(detect).
Commits local on feat/countable-phase0. Webots is shared: claim from the chat and room sessions, FREE after.

## 8. Pass

- Unit: Λ for bed_6's logged state (1.98 × 0.28, H 1.86, its Σ) gives p_cls < 0.05; bed_7's (1.75 × 2.10,
  H 0.65) > 0.9; nightstand and toilet fitted states from objects_live4 > 0.9; Λ → 0 as Σ_post grows.
  Birth: a 1.9 m z-extent scores well below a 1.25 m one for the bed; nightstand/toilet extents near their H score
  at the top.
- Live (objects_live5, same tour as objects_live4): no bed node in room_1 survives its first measured seconds;
  bed, both nightstands, toilet as in objects_live4 (positions within its errors, born no later); ~10 Hz.

## 9. Settled after the builder check (11-14; "my call" = the designer's, not the owner's)

- **No double counting of the class prior (my call).** The fitter's own size prior IS the class prior (bed config
  `AI2PriorFootprintM` 1.75 ± 0.20, `AI2PriorDepthM` 2.10, `AI2PriorHeightM` 0.65 ± 0.10), so the posterior already
  contains it. Λ is therefore computed on the **likelihood-only summary** over (w, h, H): information form
  Σ_lik⁻¹ = Σ_post⁻¹ − P₀ and θ_lik = Σ_lik(Σ_post⁻¹θ̂ − P₀μ₀), with P₀, μ₀ the size-prior precision and mean.
  Λ = log N(θ_lik; μ_cls, Σ_cls + Σ_lik) − log N(θ_lik; μ_oth, Σ_oth + Σ_lik). When the data carry no size
  information Σ_lik is wide and Λ → 0 — that is what the "Λ → 0" unit test checks. The builder must check how the
  size prior enters the recursive-Laplace update (static precision vs. folded into the transition prior) before
  subtracting it, and report if it is not separable.
- **Removal: two debounces (my call).** `occ` keeps today's sensor debounce unchanged (absence argues against
  "something there"). `cls` gets its own `rc::exist::ExistenceBelief` (L_cls = logit π_cls + Λ, same ±L_max) and its
  own `RemovalPolicy` debounce, advanced only on measured cycles where Λ < 0 (the class channel argues its own case;
  a sign, not a threshold). Either debounce may remove. The published p is the product. `common/existence_belief`
  is used, not changed.
- **When p_cls changes.** Λ is recomputed only on measured cycles (the posterior changes only then) and held
  otherwise; nothing integrates p_cls over time. The debounce over consecutive measured cycles is the persistence
  check against one bad fit, not extra evidence.
- **Footprint labelling (my call).** The class side is a two-component mixture over the w↔h labelling (equal
  weights), so a quarter-turn relabel never changes Λ.
- **Birth height density (my call).** The cloud's top depends on the view: anywhere from the body top (bed
  mattress 0.65) to the tallest declared part (headboard 1.25). Class side = uniform on [H_body, H_tallest]
  convolved with a Gaussian of σ_z (procedure default 0.05 m, per-point depth noise at ~2 m); for a class with no
  appendage it reduces to N(H, σ_H² + σ_z²). Other side = the same broad density as the fit.
- **π_cls default (my call): 0.8**, near YOLOE's median confidence on the real bed (.85, detcheck3); a manifest
  key `[prior.class] p_label_true`, same default for the three classes. The owner may change it.
- **Consumers of existence (my call).** `inst.existence` stays the occupancy factor. A `p_exists()` of the product
  is added; the published existence attribute, the singleton inhibition, the epistemic planner and the dashboard read
  the product; out-of-view verification (`wants_verification`, silhouette HOLD) reads occupancy ("is something
  there? go look"). The builder audits every `existence.` site (16 in `countable_concept/src`) and lists each choice.
- **Singleton inhibition** compares the other instances' product p, and enters L_cls (a class statement).
- **`[bed-filter]` log line:** no script parses it (grep 11-14); it is replaced.

## 10. Left to the builder

- The "other furniture" density (broad Gaussian vs log-uniform per variable).
- Which state variables enter Λ (my call: w, h, H; aspect is then implied, the separate aspect term goes).
- How the tallest declared part is read from the manifest (the back-panel reader already has the headboard).
