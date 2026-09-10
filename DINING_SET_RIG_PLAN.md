# Dining-set / circular-rig meta-concept — implementation plan

Status: **DESIGN** (2026-07-26). Motivating bug: chairs around a table fit with **wrong yaw** even
from good, face-on views — the seat is square (`sw==sd`), so yaw is 4-fold ambiguous and only the
backrest breaks it; the backrest is a thin plate with too few points to resolve reliably, so the fit
lands on a wrong mode and stays (`chair_1` stuck ~180°, backrest pointing *at* the table). No single
view fixes this — the information isn't in one chair's mask. The **scene arrangement** is the missing
evidence, and a higher-level concept is the right place to hold it.

Related: [[table-round-vs-square-model]] (chair ring IS the round-vs-square prior), the compositional
m-skill work (paper-3), the room-containment prior (same "structural prior disambiguates a locally
unobservable DOF" pattern), and `EXISTENCE_BELIEF_PLAN.md` (same precision-weighted-fusion discipline).

---

## 0. One-paragraph thesis

A **rig** (dining / circular / rectangular) is a generative model one level above the part concepts.
Its latent `z_rig` predicts each part's pose; those predictions **are** priors handed to the part
agents. The rig never *dictates* — it publishes a prior **mean + precision** per DOF, and each part
does precision-weighted Bayesian fusion. Precision is the whole trick: the rig sends **high** precision
only on the DOFs the arrangement pins (chair yaw, table shape) and **~zero** on the DOFs the part sees
better itself (chair tangential position). Where a part's likelihood is sharp the prior barely moves it;
where it's flat (today's yaw) the prior decides. Two disciplines keep it honest: **cavity** (the message
to part *i* excludes part *i*'s own estimate) and **evidence-gated down-precision** (a weak/partial rig
pushes softly).

---

## 1. Design principles (the invariants that make it correct)

1. **Priors, not commands.** The rig writes `(mean, precision)` per DOF; the part fuses. A part whose
   own observation is precise overrides the prior (a genuinely turned-away chair keeps its yaw).
2. **Precision-weighted, per-DOF.** Never a single scalar confidence. Yaw, radial position, tangential
   position, table shape, table size each carry their own down-precision.
3. **Cavity / leave-one-out.** The prior sent to part *i* is computed from `{table} ∪ {chairs ≠ i}`,
   never including *i*. Otherwise the rig echoes each part back to itself → self-confirmation
   (same FEJ discipline as [[room-corner-spatial-exclusion]]).
4. **Evidence-gated.** The rig has its own posterior over "is this actually a rig?". Down-precision
   scales with that evidence (member count, arrangement residual, pattern sharpness). 2 chairs → soft;
   a clean ring of 4 → firm.
5. **Null hypothesis is real.** `arrangement ∈ {circular, rectangular, none}`. If the parts don't form
   a pattern, `none` wins and **no priors are pushed**. The rig must be able to say "these are just
   furniture", so it can't force structure onto a random scene.
6. **No thresholds where a covariance will do** (CLAUDE.md): "chair faces table" is a von-Mises prior on
   yaw with concentration κ, not an `if`. Model selection uses log-evidence, not hard gates.
7. **Bidirectional, convergent.** parts→rig (evidence for `z_rig`) and rig→parts (priors) iterate as
   empirical Bayes; one message each per cycle is fine (the graph is the persistent state).

---

## 2. Architecture

New DSR agent **`dining_set_concept`** (a concept-of-concepts; keep it separate so the arrangement fit
+ cavity math live in one place and the same agent serves circular *and* rectangular rigs).

- **Agent id**: pick one unique across the fleet (chair=20; verify cabinet/others — propose **22**).
  Id collision = CRDT clash → SIGSEGV ([[agent-id-collision-crash]]).
- **Reads**: `object` nodes with `object_subtype ∈ {table, chair}` (and legacy name prefixes
  `table_*`/`chair_*` — use the `node_is_object_class` helper pattern already in room_scene_graph),
  with their room→part RT pose + `rt_covariance`.
- **Owns**: one `rig` node per detected group (`type()=="object"`, `object_subtype=="dining_set"`,
  name `dining_set_*`), placed at the arrangement center; and a `group_member` edge rig→each part.
- **Writes DOWN**: prior attributes on each member node (§5). It does **not** create an RT edge to the
  parts — that would double-parent them in the transform tree (they are already room→part children).
  The structural link is the non-RT `group_member` edge; the message is the prior attributes.
- **Presence**: required peers = the graph + at least `table`/`chair` producers; copy bottle_concept's
  debounced Degraded exactly. Owns-node stale-sweep on startup (delete orphan `dining_set_*`).
- **Lifecycle**: null-agent style (reads parts, writes priors); connects graph signals **QUEUED**, never
  DirectConnection.

```
        room
         │ RT
   ┌─────┼───────────────┐
 table  chair_1 … chair_N            ← part concepts (own perception/fit)
   ▲      ▲                          parts→rig: pose+cov  (evidence)
   │      │ group_member (non-RT)
   └──── dining_set_k ───────────────┐
          z_rig = {center, type,     │ rig→parts: prior mean+precision
                   r/w×h, φ, N}       ▼ (attributes on each part node)
```

---

## 3. Generative model `z_rig`

```
z_rig = {
  c        : center (room xy)                 // ~ table center
  type     : {circular, rectangular, none}    // discrete, model-selected
  r        : ring radius            (circular)
  w, h, α  : table footprint + yaw  (rectangular)
  φ        : phase / seat spacing
  N        : seat count (observed)
}
```

Predictions (the priors):

- **Chair i pose** — slot on the ring/perimeter:
  - circular: position on circle radius `r` about `c`; **facing yaw** = bearing from chair→`c` so the
    backrest normal `n̂=(sinψ,−cosψ)` points **away** from `c`.
  - rectangular: nearest table edge; facing = inward edge normal.
- **Table** — `c` = arrangement center; **shape posterior** (round if chairs lie on a circle, rect if on
  rows) with its own info; size bounded by the ring radius / row extent.

The chair-facing yaw is the immediately valuable prediction; ship it first (§8, Phase 1).

---

## 4. Inference loop (per compute cycle)

```
1. GATHER    read table + chair_* nodes: pose (RT room→part) + rt_covariance (→ per-part precision).
             Drop parts with stale/low-precision pose (freshness-as-precision).
2. FIT       for each arrangement type, fit z_rig to member positions (weighted LS by part precision):
               circular: (c, r) minimising Σ w_i (‖p_i−c‖ − r)²;  seats N = count.
               rectangular: anchor on the table footprint if present; assign chairs to sides.
             residual_type = weighted arrangement residual.
3. SELECT    log-evidence per type (residual + Occam over {c,r,…}); include `none` (flat null).
             p(type) = softmax(logZ). If p(none) dominates → push NOTHING this cycle.
4. CONFIDENCE  rig_evidence ∈ [0,1] = f(N_members, 1−normalised residual, p(type≠none)).
5. CAVITY + PUSH   for each chair i:
               recompute c_i (and r_i) from {table} ∪ {chairs ≠ i}   // leave-one-out
               yaw_prior_i   = facing(c_i, p_i)
               kappa_i       = kappa_max · rig_evidence · slot_sharpness_i
               radial_prior_i, radial_info_i (optional, Phase 2)
               write rig_yaw_prior/​rig_yaw_kappa (+ radial) on chair_i node (update_node)
             for the table: write shape/center/size priors + infos (Phase 2).
6. SELF-STATE  persist z_rig on the dining_set node (viewer + rt center); update group_member edges
               (add new members, drop departed).
```

Cost is trivial (N small); the cavity is N+1 tiny LS fits.

---

## 5. The down-channel (DSR attributes)

Register in cortex `dsr_attr_name.h` (`REGISTER_TYPE(..., false)`; **user reinstalls cortex** — see
CLAUDE.md typed-attr rule). Type-attributed access only, never `runtime_checked_*`.

On each **chair** node:

| attr | type | meaning |
|---|---|---|
| `rig_yaw_prior`   | float | room-frame facing yaw (rad), cavity-excluded |
| `rig_yaw_kappa`   | float | von-Mises concentration = yaw prior precision (0 ⇒ inert) |
| `rig_radial_prior`| float | expected distance from center (m) — Phase 2 |
| `rig_radial_info` | float | radial precision (1/m²) — Phase 2 |
| `rig_id`          | uint64| which rig authored this (staleness / multi-rig) |

On the **table** node (Phase 2):

| attr | type | meaning |
|---|---|---|
| `rig_shape_round_logodds` | float | +→round, −→rect (chair ring evidence) |
| `rig_center_x/y`, `rig_center_info` | float | center prior + precision |
| `rig_size_prior`, `rig_size_info`   | float | footprint scale prior + precision |

Precision as **concentration/information** makes the consumer-side fusion a plain add. Absent/zero ⇒
the consumer ignores it (backward compatible; agents that predate this just don't read it).

---

## 6. Consumer integration — chair side

Chair reads `rig_yaw_prior`/`rig_yaw_kappa` each cycle and folds them into the yaw belief in **two**
places (both already exist):

1. **`resolve_orientation` mode pick** — add a prior log-odds to `flip_acc_[k]`:
   `flip_acc_[k] += −κ·cos(yaw_k − rig_yaw_prior)` (von-Mises log-density up to const). This biases the
   4-mode selection toward the table-facing mode **without** overriding a mode the backrest evidence
   strongly prefers (the data term still accumulates in `flip_acc_`).
2. **Continuous yaw update** — an extra information term `κ·(rig_yaw_prior − yaw)` in the GN step, i.e.
   the rig prior enters the belief like any other measurement with precision κ. Reuses the existing
   common-mode / information-form machinery.

Config (chair): `RigYawPriorEnabled` (default true), `RigYawKappaMax` cap (safety clamp so the rig can
never *fully* override a close, sharp observation), `RigPriorStaleMs` (ignore a prior whose `rig_id`
node is gone / not refreshed). This is the same `AI2TableFacingPrior` hook discussed — the only change
is the facing direction is supplied by the rig (cavity + precision-scheduled), not hand-derived.

## 6b. Consumer integration — table side (Phase 2)

Table reads `rig_shape_round_logodds` as a prior in its round-vs-square discrete model selection
([[table-round-vs-square-model]]), and `rig_center_*`/`rig_size_*` as weak priors on its footprint —
each precision-weighted, so a table that sees its own footprint well is barely moved, while a
poorly-seen far table snaps to the chair-ring-implied shape/center.

---

## 7. Config (dining_set agent)

```
[Rig]
Enabled            = true
MinMembers         = 3        # below this, arrangement is under-determined → stay soft/none
ArrangementTypes   = ["circular","rectangular"]   # + implicit "none"
KappaYawMax        = 8.0      # cap on yaw down-precision (κ; ~1/(20°)² scale)
EvidenceResidualRefM = 0.15   # arrangement residual (m) at which rig_evidence halves
CavityLeaveOneOut  = true     # NEVER false in production (self-confirmation)
CenterFromTable    = true     # anchor center on the table node when present (else chair centroid)
PublishTablePriors = false    # Phase 2 gate
```

All continuous, no hard gates except `MinMembers` (flagged: a genuine under-determination floor, not a
tuning knob) and the safety `KappaYawMax` clamp.

---

## 8. Phasing

- **Phase 0 — plumbing.** Register the `rig_*` attributes in cortex; new `dining_set_concept` agent
  skeleton (presence, stale-sweep, reads parts, writes a `dining_set` node + `group_member` edges).
  No priors yet. Verify the node/edges appear and clean up on exit.
- **Phase 1 — yaw-only to chairs (fixes today's bug).** circular fit → cavity center → per-chair
  `rig_yaw_prior/kappa`; chair-side `resolve_orientation` + yaw-update hook. **Validate against the
  live dining set**: `chair_1` should snap from ~180° to facing the table; `std_yaw_rep` reflects the
  now-informed (tighter) yaw only when `rig_evidence` is high. A/B by `RigYawPriorEnabled`/`KappaYawMax`.
- **Phase 2 — radial chair prior + table shape/center prior.** Adds the round-vs-square disambiguation
  and center/size priors to the table. Closes the bidirectional loop.
- **Phase 3 — model selection + rectangular rig.** Full `{circular, rectangular, none}` evidence
  selection; rectangular seat-to-side assignment. Multi-rig (two tables) via clustering members first.
- **Phase 4 (optional) — affordances.** The rig exposes "under-observed / ambiguous member" as an
  epistemic target (go look at the chair whose `rig_evidence`-corrected yaw is still uncertain), tying
  into the controller EFE selection.

---

## 9. DSR / cortex invariants (must-follow, from CLAUDE.md)

- Type-attributed attrs only; register new `rig_*` in cortex + user reinstalls; rebuild consumers.
- RT edges timestamped; but the rig↔part link is a **non-RT** `group_member` edge (no double-parenting).
- Connect graph update signals **QUEUED**, never DirectConnection.
- Unique `[Agent] id`; debounced Degraded; stale-sweep owned `dining_set_*` on start + first Operating.
- Deep-copy any `cv::Mat` at thread boundaries (N/A here — no images — but keep the discipline).
- Eigen alignment: no `-march=native`, no `EIGEN_MAX_ALIGN_BYTES=0`.

---

## 10. Risks & mitigations

- **Self-confirmation** → cavity leave-one-out (hard requirement).
- **Forcing a rig on non-arrangement** → `none` null hypothesis + evidence-gated κ + `MinMembers`.
- **Overriding a genuinely turned-away chair** → κ capped (`KappaYawMax`); the chair's sharp backrest
  likelihood out-weighs a soft prior. Only the *ambiguous* (flat-likelihood) yaw is moved.
- **Phantom member poisoning the fit** → weight members by pose precision + existence log-odds; a
  low-existence / out-of-room member (already removable via the room prior) is excluded.
- **Departed chair** → drop `group_member` + zero its `rig_*` (κ→0) so a stale prior can't linger.
- **Circular dependency chair↔table via rig** → the rig is the single mediator; parts never read each
  other directly, and the cavity keeps the mediation acyclic per message.

---

## 11. Open questions for the user

1. **Which arrangement first** — circular only (fastest for the current rig) or circular+rectangular
   from the start? (Recommend circular-first; rectangular in Phase 3.)
2. **Table as center anchor** vs. chair-centroid-only for v1. (Recommend anchor on table when present.)
3. **KappaYawMax** — how hard may the rig ever push yaw? A cap of ~1/(20°)² lets a close ZED view win
   but disambiguates the square-seat case. Tune live.
4. **Separate agent vs. fold into an existing one** — plan assumes separate `dining_set_concept`.
