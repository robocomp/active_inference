# MODEL_HISTORY.md — how experience accumulates, and why the model settles

The theory behind stage 4 of [`CONCEPT_AGENT_LIFECYCLE.md`](CONCEPT_AGENT_LIFECYCLE.md). It answers a question
the belief theory did not previously address: **after hours of touring the apartment, what has the robot
actually learned, and what makes a settled model resist a spurious mask?**

Status: **design**. Nothing here is implemented. §5 is what ships first (recording only).

---

## 1. The problem

Everything the belief pipeline does today reasons about the *present frame*. CREATE weighs this frame's
detection, UPDATE weighs this frame's points, REMOVE weighs this frame's absence. Nothing weighs *how long
this has been true*.

The consequences are measurable:

- **Nothing survives a restart.** The mean warm-starts from DSR attributes, but Σ, existence `L` and every
  accumulator re-seed — and it is moot anyway, because every agent wipes its object nodes at startup. Hours of
  touring currently consolidate into **nothing**.
- **Hysteresis was deliberately removed and never durably replaced.** `TABLE.md` lists maturity stiffening,
  size ratchet, CUSUM, position lock and yaw barrier as deleted because it "emerges from the math". The
  emergent replacement is Σ. The module that implemented the old machinery
  (`common/belief_stabilizer`) still exists but is used by `human_concept` alone.
- **Experience is scattered** across at least five private accumulators with unrelated scales and no shared
  concept: `flip_acc_` (±6), existence `L` (±4), `shape_evidence` (±8), `verify_surprise` (vs 20),
  `view_spent_` (3.0). None is persisted. None answers "how well do I know this object?".

### The root cause

For a static model (`F = I`) the predict step is `Σ ← Σ + Q`, applied every frame whether or not anything was
observed, with `Q` a **fixed per-frame constant**. Two consequences follow, neither of which depends on how
long the object has been tracked.

**(a) Unobserved, precision decays linearly in frames.** `Σ(n) = Σ₀ + Q·n`, so the variance-doubling time is
`τ = Σ₀/Q`. With `AI2ProcessStdM = 0.005` (`Q = 2.5e-5 m²/frame`), a converged `σ_w ≈ 0.03 m` doubles in
**36 frames (3.6 s)** and returns all the way to the `σ = 0.30 m` prior in **≈ 5.9 minutes**. That ceiling is
the same after three hours of tracking as after thirty seconds, because nothing shrinks `Q`.

*Measured, not asserted* (`ai2_log.csv`, table_9 cycles 970→2290 — 1320 predict-only cycles, `npts = 0`
throughout): predicted `σ_w = √(9.0e-4 + 2.5e-5·1320) = 0.184 m`, observed **0.170 m**; `std_yaw` went
0.101 → 0.348 rad over the same stretch. The next sparse mask — 178 points at 2.39 m — then landed on that
5×-loosened belief and re-cut the table. Extrapolated to 15 minutes the walk reaches ±0.47 m of extent and
±54° of yaw: total amnesia.

**(b) Observed, precision does not accumulate — it sits at a floor.** In information form the update is
`Σ⁻¹ ← Σ⁻¹ + i`, with `i` capped at `Σc⁻¹` by the engine's common-mode marginalisation. The fixed point of
one predict + one update,

```
1/y = 1/(y+i) + q      ⇒      q·y² + q·i·y − i = 0      ⇒ (for q·i ≪ 1)      y ≈ √(i/q)
```

```
σ_steady ≈ (q / i)^¼
```

**A ratio of two rates — elapsed time does not appear.** Evidence is discarded by `Q` at exactly the rate it
is acquired, so hours of observation leave the belief in the same state as seconds. (Order-of-magnitude check:
with `i` at the `Σc⁻¹` cap = 1/0.02² = 2500 m⁻², `σ_steady ≈ 0.01 m`; the observed converged `σ_w ≈ 0.03 m` is
looser because the real per-frame information sits well below that cap once range, coverage and gating apply.)

⚠️ Be precise about what is *not* claimed: `τ = Σ/Q` is **not** identical across instances — a converged
belief has the *shorter* τ (36 frames vs 3600), because a small variance doubles sooner. What is identical is
`Q`, the absolute forgetting rate, and hence the retention ceiling `Σ_prior/Q`. The load-bearing statement is:
**resistance to disturbance is a function of *current* Σ; current Σ is pinned to a floor by (b) and decays to
the prior by (a) at a rate independent of history.** Nothing an agent learns extends how long it stays
resistant — which is precisely the missing history.

`clamp_to_prior`, `process_std_extent_m = 0` and arguably the coverage common-mode are all workarounds for
this one missing latent.

---

## 2. The move

Both halves of the history stage are the **same operation applied to two different constants**:

> A constant becomes an inferred, covariate-dependent latent, regularised by a hyperprior whose pull-back
> keeps it recoverable.

| today (a constant) | becomes (an inferred latent) | the question it answers |
|---|---|---|
| process noise `Q`, fixed per frame | `ω = log Q`, per instance, per DOF | *how fast does **this object** change?* |
| clutter / false-alarm rate, `0.05` | `p_FA(place × bearing)`, per class | *how often does **this view** lie to me?* |

This is the discipline `CLAUDE.md` already states — *encode the effect in the generative model and let it fall
out of the probabilistic inference, typically as a continuous precision that grows or shrinks with the right
physical covariate* — applied to the two constants that govern accumulation rather than to a single frame.

It is also not new machinery. `room_concept/HIERARCHICAL_PRECISION.md` already specifies the conjugate
log-precision hyperprior and its config surface (`hier_prec_*`, flag-gated off) for *observation* precision.
§3 applies the identical update to *process* noise.

---

## 3. Inferred volatility — "how fast does this object change?"

Each instance carries a per-DOF log-volatility `ω`, with `Q = exp(ω)`. After each update:

```
ω ← ω + lr · ( ½·exp(−ω)·d  −  k/2  −  (ω − ω₀)/σ_ω² )
```

- `d` — normalised surprise of this cycle's belief motion, `δθᵀ Σ⁻¹ δθ`, available from `update()`.
- `k` — DOF count of the group.
- `(ω₀, σ_ω²)` — the hyperprior: prior mean and how far evidence may pull ω away from it.

This repairs **both** consequences in §1, which is the test any candidate fix has to pass:

- **(a) decay** — `τ ≈ Σ·exp(−ω)` grows as ω falls, so the retention ceiling is no longer pinned at ~6 minutes.
- **(b) floor** — `σ_steady ≈ (q/i)^¼ = (exp(ω)/i)^¼` falls as ω falls, so precision genuinely accumulates
  instead of resting at a fixed floor.

A Σ-clamp addresses only (a); an evidence-count stiffener addresses only (b). Making `Q` itself the inferred
quantity is what moves both, because both are functions of `q`. Memory length becomes **continuous,
per-object, per-DOF, and self-tuning.**

**Behaviour:**

- Consistent observation over hours → `d` stays small → ω driven down → τ grows toward hours → a spurious
  mask cannot move the belief. *This is the hysteresis, earned by evidence rather than imposed by a clamp.*
- The object genuinely moves → sustained large `d` → ω rises → τ collapses to seconds → the model adapts.
  **Stiffness is never a lock.**
- **Recoverability is structural, not a threshold.** The `(ω − ω₀)/σ_ω²` term pulls ω back whenever evidence
  stops supporting extreme stiffness. Nothing needs clamping to stay escapable.
- **Per-DOF ω** lets a table's extent freeze while its pose stays adaptive — the correct physics for
  furniture, currently hard-coded as `process_std_extent_m = 0`.

**What it subsumes.** `clamp_to_prior` degrades to a safety net (with ω inferred low, Σ growth self-limits);
`process_std_extent_m = 0` becomes an inferred outcome rather than an assertion; the coverage common-mode
keeps its per-frame role but stops carrying the long-horizon burden.

**Known failure mode — bias, not noise.** A persistently *biased* observation (mask under-segmentation from a
recurring viewpoint — exactly what the common-mode `Σc` exists to model) looks perfectly *consistent*. It
drives `d` down, drives ω down, and stiffens the model **around the bias**. Volatility and `Σc` must be
reasoned about together: `Σc` is what prevents a shared per-frame error from being read as agreement.

---

## 4. Phantom memory — "how often does this view lie to me?"

Classifiers make *confident* mistakes under partial occlusion — part of a radiator reads as a chair with a
high score, births a phantom, and it survives until a good run of masks from a proper pose kills it. Score
cannot filter this. But the **recurrence** is learnable: the same place, from the same bearing, lies the same
way every tour.

A per-class false-alarm field `p_FA`, indexed on **(world cell × view bearing bin)**, lowers
`Detection::birth_evidence` (stage 1 of the contract).

- **Indexed on the pair, not place alone.** The failure is viewpoint-dependent. A place-only index would
  suppress a *genuine* chair placed there from every direction; with bearing in the key, an unpoisoned
  viewpoint still births it promptly. **The memory is never a permanent blacklist.**
- **Scales evidence, never vetoes.** `birthable` stays `true`; a poisoned view raises the evidence bar, so a
  close centred fixation still births a real object — it just has to work harder. This is what keeps the
  memory falsifiable, and it reuses a hook that already ships: `Tracker.BirthFusion` moves the same scalar in
  the *raise* direction from residual_concept's spatial field.
- **Same hyperprior pull-back as ω**, so the field decays when unsupported. The world changes; a radiator can
  be replaced by a real chair.

### The attribution rule (the reason §5 ships first)

A birth→death event is evidence of **either** a detector false positive **or** a removal false positive.
Attributing it to the detector is only sound when removal is trustworthy.

Right now it demonstrably is not. Three independent removal defects found in July 2026 —
`ZedClearLosFloor` deleting chairs the ZED could not see, silhouettes voting removal *through walls*, and a
`p_detect` weight that was inert at the saturation ceiling — **all produce exactly the birth→death signature
this field would learn from.**

> **Only count a death as a phantom event when the disconfirmation was confident** — high `p_detect`,
> fixated, close. Learning from unconfident deaths teaches an agent its own removal bugs.

And the error would be self-sealing: suppressed births generate no contradicting evidence, so the prior
becomes unfalsifiable. This is the same structure as the chair's circular obliquity covariate — a wrong belief
protecting itself — and the lifecycle contract forbids that pattern in UPDATE for the same reason.

---

## 5. Sequencing

1. **Record only (ships first).** Every agent logs births and deaths with world cell, robot pose, view
   bearing, age at death, and the existence state at death. Zero effect on behaviour.
2. **Baseline mission run.** Answers: does Σ converge and stay converged over hours; at what rate do the
   accumulators saturate; how many distinct bearings does a real object accumulate on a tour; and do phantom
   deaths cluster in (place × bearing) as predicted?
3. **Validate the teacher.** Cross-check recorded events against the known removal defects. If deaths cluster
   where `p_detect` was *low*, the log is recording removal bugs and must not be learned from.
4. **Implement §3** in `table_concept` behind a flag; A/B against the baseline. The check is τ: an object seen
   across several tour passes must show ω falling and `std_w` no longer growing while unobserved, while a
   deliberately moved object still adapts within seconds.
5. **Implement §4**, then choose persistence. Both `ω` (a short per-DOF vector) and `p_FA` (a sparse cell map)
   are compact — *those*, not Σ, are what is worth writing to a node attribute.

---

## 6. Relation to the existing accumulators

Each of these already implements a fragment of this idea with its own private scale. `MODEL_HISTORY` is the
concept they should converge on, so a ninth agent inherits one mechanism instead of inventing a sixth.

| accumulator | agent | what it really is | bound |
|---|---|---|---|
| existence `L` | shared `rc::exist` | evidence for *being* | `±L_max` (4) |
| `flip_acc_` | chair | evidence for a discrete yaw mode | `±kFlipClamp` (6) |
| `shape_evidence` | table | evidence for round vs square | `±8` |
| `view_spent_` | chair | information already taken from one bearing | budget (3.0) |
| `verify_surprise` | table, cabinet | unresolvable absence → go-look pull | EMA vs 20 |

All five share the right instinct — bounded, evidence-driven accumulation that stays recoverable — and all
five were tuned in isolation. The `flip_acc_` comment states the principle best: the bound exists *"so a long
confident run can still recant a real rotation."* That is the property §3 generalises to the whole state, with
the bound replaced by a hyperprior.

`common/belief_stabilizer` is prior art: maturity-stiffened Kalman gain plus a CUSUM/SPRT change detector —
hysteresis, explicitly. It was abandoned in the AI2 rewrite because its stiffening was a *tuned ratchet* with
no generative justification and no way to recant except a hand-built change detector. §3 keeps the goal and
discards the mechanism: stiffening becomes an inferred latent, and the "change detector" is just ω rising.
