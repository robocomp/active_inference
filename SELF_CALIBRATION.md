# Opportunistic self-calibration of a mobile base

Written for transcription into the thesis. Everything below is either **measured** (with the number
and the sample), **unit-tested**, or **designed but unrun** — each is labelled, because the
distinction is the point of several of the results.

Companion records: the estimator experiment in full is in the memory
`thesis-online-motion-calibration-experiment` (protocol, A/B, injected-error validation). This
document covers the whole system and, in depth, the **deliberate manoeuvre** and the **closure**
built on 2026-08-24/25, which that record predates.

---

## 1. The claim

A robot can keep its own motion model calibrated while doing its ordinary work, without a calibration
procedure, without a survey, and without anyone deciding when to calibrate. Three ingredients:

1. **The free data is most of the data.** Every traversal a robot makes to serve some other purpose is
   calibration data. A passive estimator that watches ordinary motion identifies most parameters
   without asking for anything.
2. **When the free data is not enough, the robot asks — it does not take.** The deliberate manoeuvre
   is published as an *affordance* carrying its expected information gain in nats, and it competes in
   the same expected-free-energy arbitration as every other thing the robot could do. It
   extinguishes itself as the posterior sharpens. No schedule, no trigger threshold.
3. **The manoeuvre it asks for is self-certifying.** A closure pivot — turn through whole turns and
   stop on the heading you started from — yields ground truth with no map, no survey and no
   localiser in the number.

The contribution that is new here is (3), and specifically §7.4: **a closure at angular rate ω does
not measure the rotation scale. It measures the scale plus the gyro bias divided by the rate.** Two
closures at two rates separate them in closed form, on headings alone.

---

## 2. The problem

`room_concept` localises by predicting with odometry and correcting with an SDF fit against a room
model. The optimizer is skipped while the predicted-pose residual stays under a gate. Between
corrections the prediction error ramps; when it crosses the gate the optimizer fires and knocks it
back. The visible signature is a **sawtooth** in prediction error, steeper the faster the robot moves.

A sawtooth of that shape says the *predictor* is systematically wrong, not noisy. Attribution
(memory §4–5): accumulated error correlates with **rotation** (r = +0.61) and not with **time**
(r = +0.07) — a scale error, not a bias or drift. The largest single term turned out to be the
command-velocity prior; removing it took rotation steepness from 15.06 to 2.20 mm SDF per radian.

What remains after that is the motion model's own parameters, and those are what this system learns.

**The design brief, in the user's words:** *"We don't want an eternal calibration procedure with tens
of tests. We want a partially calibrated robot to improve by itself in time."*

---

## 3. Design principles

**No thresholds; put the effect in the generative model.** Every place where a gate would be the
obvious implementation, the system instead carries a covariance that grows with the right physical
covariate, and lets the inference do the gating. Three worked examples:

- An uninformative episode (parked robot) teaches nothing because its regressor `H → 0`, so the
  Kalman gain `K = PH/(HPH + R) → 0`. Nothing says "ignore parked robots".
- A correction taken during a turn contains a component the translation parameters cannot explain.
  Rather than gating turns out, `R_pos += (rot_model_sigma·|Δθ|)²`.
- A correction produced by a *bad fit* is worth less. Rather than a regime detector,
  `R += (fit_model_gain·max|SDF|)²`.

**Precision, not point estimates, is what is published.** A parameter sitting at its prior and a
parameter measured to be zero look identical as numbers and are opposite as knowledge. Every
parameter carries a σ, and the system reports `identifiable()` as *"the data has outweighed the
prior"* — `data_precision > prior_precision` — rather than *"σ is small"*, which needs an invented
cutoff and lets an untouched channel call itself measured.

**A stale certainty is worse than an honest ignorance.** The scale is treated as a slowly varying
quantity with a random-walk density (fraction/√s), not a constant. Evidence ages; a channel the robot
has not exercised widens on its own.

---

## 4. Tier 1 — the passive observer

`common/motion_calib/scale_estimator.h`.

Over an interval `T` the odometry's error against a reference is modelled as

    e(T) = s·Δ(T) + ε,        ε ~ N(0, σ²·T)

A **scale** error is fully correlated across the interval and grows like `T`; the random walk is
independent and grows like `√T`. One weighted least squares separates them: the **slope** is the
scale, the **residual** is the density. This is not a heuristic — `∂F/∂s = 0` and `∂F/∂σ = 0` on the
negative log evidence return exactly these in closed form, so *calibrating the model and maximising
its evidence are the same computation*.

Weight is `1/T`, which is the ML weighting given `Var(ε) ∝ T`, not a preference.

**Recursion and ageing.** Information form, so the variance can never go negative however long the
gap: `I ← I / (1 + q²·Δt·I)`, with `q` the scale-walk density. Evidence about the scale ages; the
prior does not.

**Two priors, both measured rather than chosen.** The scale prior σ = 0.15 comes from what the
preintegration's asserted constants already claim, so the estimator starts no more confident than the
system was without it. The *density* prior is 0.008181 rad/√s for rotation and 0.024494 m/√s for
translation — measured live 2026-08-20 over 209 424 frames / 5 234 windows, with the online and batch
estimators agreeing to 1e-9.

> **Why the density needs a prior at all** — a cold estimator has no residuals to estimate σ from. An
> early version fell back to a large placeholder, and a cold estimator therefore advertised the
> *smallest* expected gain of its life: 0.03 nats cold against 3.7 nats after 400 s of turning. Exactly
> backwards, and it would have lost every arbitration on its first day.

---

## 5. Tier 2 — the joint estimator that steers

`room_concept/src/calibration_estimator.h`. Six parameters, solved jointly (scalars cannot separate a
mount yaw from a lever arm):

| parameter | residual component | covariate | supplied by |
|---|---|---|---|
| `k_v` translation scale | along-track | `d_forward` | every traversal |
| `eps_yaw` mount yaw | cross-track | `−d_forward` | every traversal |
| `k_lat` lateral scale | cross-track | `d_lateral` | strafing only |
| `k_omega` gyro scale | heading | `d_theta` | turning |
| `b_omega` gyro bias | heading | `duration` | turning at *varying* rates |
| `dk_wheel` wheel mismatch | heading | `d_forward` | every traversal |

**The observability rule, stated once:** two parameters are separable **iff they load on a different
(residual component × covariate) pair.** Everything about which manoeuvre is worth making follows
from this table and nothing else.

Two consequences worth stating in the thesis:

- `eps_yaw` and `k_lat` share a residual component and differ only in covariate; `dk_wheel` and
  `k_omega` share a covariate and differ only in component. Both pairs are separable.
- **`k_omega` and `b_omega` are collinear at any fixed rotation rate.** At constant ω, `d_theta =
  ω·duration`, so the two columns are proportional. The joint solve reports this honestly — its
  normalised condition number goes 14.5 → 216.4 on exactly this case in the self-test — but no
  estimator can undo it. Only motion at *different* angular rates can. §7.4 is the consequence.

---

## 6. Deciding when to act — expected information gain

The manoeuvre must justify itself against everything else the robot could be doing, in a common
currency. That currency is nats of expected entropy reduction, the same one the exploration and
object affordances already publish.

    ΔH = ½·log( I_after / I_before )

with `I = Σ w·Δ²/σ² + 1/σ²_prior`. **It is the marginal gain that is advertised**: the excitation the
robot's ordinary tours are expected to deliver over the same horizon is passed in as `passive_delta`,
and what comes back is what the *deliberate* manoeuvre adds beyond the free data. The passive rate is
measured (an EMA over the same windows the estimator is fed), not assumed.

Measured behaviour — this is the self-extinguishing property, and it is the argument that the system
needs no schedule:

| robot's recent diet | marginal gain of a pivot |
|---|---|
| cold, nothing measured | **3.46 nats** |
| after 400 s of ordinary turning | **0.01 nats** |
| after 400 s of straight-line driving | **3.70 nats** |
| (2026-08-25) cold at boot | 5.204 nats |
| (2026-08-25) immediately after one pivot | 1.092 nats |

A robot whose day is full of turns is never asked to spin. A robot that has only driven straight is.

**One correction of principle, found live:** the offer must be priced against the posterior that
*actually steers the robot* — the joint estimator — not against the passive observer's own posterior.
The passive observer feeds nothing back by design, so pricing against it would let a robot conclude
"my tours already taught me, a pivot is worthless" about a quantity nothing consumes, while the
estimator that does steer still held an uninformed parameter.

---

## 7. The manoeuvre

### 7.1 It is an affordance, not a command

`room_concept` publishes `afford_calib` as an ordinary affordance node with an ordinary contract:

| field | value | meaning |
|---|---|---|
| `policy` | `Orient` | turn in place; nothing to navigate to |
| target x,y | **the robot's own pose** | an orient makes no claim about where to be |
| target yaw | the next bearing | the whole of the request |
| `max_yaw_rate` | 0.5 or 0.25 rad/s | the producer names the rate (see §7.4) |
| `stable_n` | 2 | hold inside the band two cycles |
| `timeout_ms` | 4× nominal step, floor 30 s | patience, not a prediction |
| goal predicate | **none** | arriving at the bearing *is* the completion |

Nothing about calibration appears in the controller. The manoeuvre is a *sequence of ordinary
affordances*, and the controller executes it with the same code path it uses for a glance.

### 7.2 Opportunism is enforced by the protocol, not by policy

The producer cannot know whether the body fits through every heading where the robot happens to
stand; the consumer can, and already asks (`can_turn_here`). An `Infeasible` reply is taken at face
value: **stop asking, and wait to be carried somewhere with room by the ordinary work.** The
calibration never drives anywhere. That single rule is what makes it opportunistic rather than a
procedure. The refusal is re-armed only when the robot has moved *one body width* — the only distance
that means anything for a question about whether the body fits.

### 7.3 The closure argument

Turn through `N` complete turns and stop on the heading you started from, and the robot turned
*exactly* `N·2π` radians. That is a fact about turning, independent of the map, the survey and the
localiser. Comparing the odometry's accumulated Δθ against it gives the scale with **nothing
estimated in the denominator**.

Three design points, each of which was got wrong first and is worth stating as such:

**(a) The bearings must be anchored to the start heading.** The executor completes an Orient inside
its aligned band (0.05 rad ≈ 2.9°), so every step lands a couple of degrees short. Asking for
"120° more from wherever you are" bakes each shortfall in permanently. *Measured 2026-08-24: thirteen
consecutive steps of **117.4°**, never 120.* Twelve of those total 1409°, leaving the robot 31° from
its start — the heading test can never pass and the pivot runs for ever. Anchored (step *k* asks for
`start + k·120°`), a short step is absorbed by the next one and step twelve asks for the starting
heading itself. *After the fix: twelve steps, bearings a clean repeating triple 120° apart,
`truth = 1440.000°`.*

**(b) The number of turns must be counted, not asserted.** An early version used the *configured*
turn count as the denominator. Under (a)'s drift the sequence eventually closes after **thirteen**
turns rather than four, and dividing thirteen turns of odometry by four asserted ones reports
`s_ω = +226%` — a confident, catastrophically wrong calibration, which is far worse than a pivot that
never finishes. The truth is `round(ref_turn/2π)·2π`, which is exact here because the localiser's
heading error is degrees and the spacing is 360.

**(c) A closure is a total, so it counts every radian between its two ends.** An intermediate version
credited only motion made while the consumer demonstrably held the claim, to stop a competing
traversal being charged to a step. *Measured 2026-08-25: fifteen anchored steps, five real turns
(1800°), only 1440 credited* — a whole turn lost to poll and DDS latency at the two claim edges. The
gate was also unnecessary once (b) was in place: a detour raises the accumulated turn, raises the turn
count, raises the truth, and appears in the odometry too, so the ratio is unharmed. **Counting instead
of asserting removed the need to gate.**

### 7.4 ★ A closure at rate ω measures `k + b/ω`, not `k`

This is the result. Over `N` turns at angular rate ω the odometry accumulates

    k·(N·2π) + b·T,        T = N·2π/ω

so what a closure reports is

    s_closure = k + b/ω

— the rotation **scale plus the gyro bias divided by the rate**. Every single-rate closure ever taken
reports that sum and calls it the scale. No amount of turning at one rate does better: the two
unknowns enter through one number. This is the same collinearity §5 identifies in the joint solve,
arriving through a different door.

Two closures at two rates separate them exactly. With `u = 1/ω`:

    b = (s₁ − s₂)/(u₁ − u₂)          k = (u₁·s₂ − u₂·s₁)/(u₁ − u₂)

Both terms then rest on **headings returning to where they left** — no map, no survey, no localiser
anywhere in the arithmetic. That is a *stronger* instrument than the joint estimator's separation,
which needs the localiser for its reference.

**The manoeuvre implements this as two blocks of four turns**, at 0.50 and 0.25 rad/s — 24 steps of
120°, roughly four minutes. Four turns *per block*, not four turns split in two: resolution goes as
`tolerance / total angle`, and `b` comes out of a *difference* of two closures, so halving each
block's angle would put the noise exactly where the answer is.

Uncertainties propagate from the closures' own resolutions, which is what makes it an instrument
rather than a number: `σ_b = √(r₁² + r₂²)/|u₁−u₂|`, `σ_k = √((u₁r₂)² + (u₂r₁)²)/|u₁−u₂|`. The bias is
always the coarser of the two.

**Unit-tested (2026-08-25)**, injecting a 3% scale *and* a 0.0040 rad/s bias:

| block | rate | measures | value |
|---|---|---|---|
| 1 | 0.50 rad/s | `k + b/0.5` | +3.77% |
| 2 | 0.25 rad/s | `k + b/0.25` | +4.62% |
| separated | | `k` | **+2.92% ± 0.06%** (true +3.00%) |
| | | `b` | **+0.00426 ± 0.00018 rad/s** (true +0.00400) |

Neither block alone is the scale. The bias could not be seen at all before — at any rate, by any
amount of turning.

**Status: unit-tested, not yet run on the robot.**

---

## 8. Results on the robot

### 8.1 The estimator (full record in the companion memory)

- On the uninjected robot: `eps_yaw` = **−0.536° ± 0.13**, converged from cold, reproducible across
  restarts, and independently matching a straight-episode least squares (−0.532°).
- Injected-error validation, gyro `+3%`, 73 m / 24 episodes: predicted `k_ω` 0.96833, **measured
  0.96901 — 0.07% apart**. The first 97° turn alone took it most of the way. No cross-talk into `k_v`.
- Injected-error validation, wheels `+3%`, 109 m / 39 episodes: **83% of the error recovered**,
  systematically (0.9759 ± 0.001 across seven windows), shortfall not yet explained.
- A/B, 262 m ON vs 105 m OFF: prediction error **0.0263 vs 0.0303 m, +14.9%, t = 2.67 significant**.
  *It lowers the error floor; it does not flatten the sawtooth* — a ramp of unchanged slope simply
  starts lower and takes longer to reach the gate. Pre-registered before the run.

### 8.2 The closure pivot (2026-08-25, single-rate, after the §7.3 fixes)

| steps | odometry | truth | s_ω | resolution | usable | estimator's answer |
|---|---|---|---|---|---|---|
| 12 | 1440.205° | 1440° | **+0.014%** | 0.18% | no | +0.342% ± 0.192% |
| 12 | 1438.104° | 1440° | **−0.132%** | 0.18% | no | −0.011% ± 0.094% |

Two independent instruments, agreeing to 1.7σ and 1.3σ. For contrast, the run *before* the §7.3(c)
fix reported **+4.34%** against the estimator's +0.34% — 13× apart, ~30σ. **That disagreement is what
caught the bug**; no single number looked wrong on its own. This is the argument for building the
comparison into the record rather than reading one instrument at a time.

**`usable = no` on both, which is the correct answer.** A four-turn closure that misses by its 0.05
rad tolerance resolves to 0.18%, and a measured 0.014% is finer than the instrument. The system
reports *"the pivot closed but the scale it measured is finer than the closure resolves — not a
measurement, and not quoted as one"*, rather than publishing a number it cannot support. This robot's
offline rotation scale is +0.00007 ± 0.00039, consistent with both.

**The honest conclusion for this robot: it does not need the pivot.** Resolving 0.007% would take of
order a hundred turns, not four. The instrument correctly says it has nothing left to add — and the
marginal-gain rule of §6 would have said so too, which is why the runs above were forced with a
testing flag (`CalibForcedGainNats`) that advertises 50 nats regardless. Every number obtained under
that flag is a test of the *machinery*, never a valuation.

---

## 9. Status ledger

| component | status |
|---|---|
| passive observer, joint estimator | **validated live**, incl. injected-error recovery |
| affordance protocol for the manoeuvre | **validated live** — offered, claimed, executed, refused, completed |
| anchored bearings, counted truth, total accumulation | **validated live** — 12 steps, truth 1440.000° |
| single-rate closure | **validated live**, agrees with the estimator to <2σ |
| two-rate separation of scale and bias | **unit-tested**, not yet run on the robot |
| parameter persistence across restarts | **not built** (§10) |

---

## 10. Limitations and open work

**Persistence, and "why does it calibrate at every boot?" are the same problem.** The pivot's gain is
5.2 nats at startup because the estimator starts cold — a wide prior is an honest statement of
ignorance the robot need not have. Persisting the parameters *and their precisions* across restarts
makes the robot boot informed and the manoeuvre lose every contest on its own merits. No trigger rule
is needed; the existing rule suffices once it is not lied to at startup. The care is in ageing the
persisted precision by the random-walk density over the time the robot was off — `predict(Δt)` does
exactly this — or a robot that sat for a month boots claiming month-old certainty.

**There is no closure for translation, and that is not an oversight.** A pivot is self-certifying
because returning to a heading is checkable without a map. To know you travelled exactly *X* metres
you must know where you started and stopped, which needs the very localiser you are trying to check.
A "drive out and back" manoeuvre would only measure the odometry against the localiser — which is
what the passive estimator already does, for free, all day. The deliberate manoeuvre is therefore
rotation-only *by construction*, not by omission.

**The translation channel of the passive observer is not instantiated** (its density prior is
measured and recorded, 0.024494 m/√s). Rotation is the channel ordinary work under-supplies.

**`k_lat` is unobservable on a differential-drive base.** It is kept because the parameter set is
shared with omni bases; on this robot it correctly reports "not asked" for ever.

**A varying-rate manoeuvre is a different contract, not a parameter of this one** — the two-block
design in §7.4 is the minimal version of that idea, and a continuously swept rate would condition the
solve better still.

---

## 11. Methodological lessons

These generalise beyond calibration and are worth a discussion section.

★★★ **A gain measured on the corrected output says nothing about the predictor.** A 4×360° closure
gave a published-pose gain of 1.0008 while the predictor underneath was 4% wrong — the optimizer mops
it up every 1–6 s. Any calibration graded on corrected output measures the optimizer.

★★★ **A parameter at its prior and a parameter measured to be zero are opposite states that look
identical.** The only cure is to publish precision alongside value and to define "identifiable" as
*data outweighs prior*, not *σ is small*. Demonstrated unprompted during validation: a turn-heavy run
correctly *widened* `eps_yaw`'s σ, because its covariate is forward travel.

★★★ **Two instruments that should agree are worth more than either alone.** The closure/estimator
disagreement (13×, 30σ) exposed a defect that no single reading looked wrong enough to reveal. Build
the comparison into the record, not into the analyst's memory.

★★★ **Assert nothing you can count.** Using the *configured* turn count as ground truth would have
published +226% with a resolution figure attached, looking entirely credible. Counting the turns made
the same code robust to a defect that had not been found yet.

★★★ **A confident wrong number is worse than a stuck instrument.** Two of the three defects in §7.3
made the pivot *fail to finish*, which is loud. The third made it finish and lie, which is silent.

★★★ **An estimator driven by another estimator's residuals inherits its failure modes** and must
carry a covariance term for "the thing feeding me is currently broken". A pathological optimizer
window dragged `k_v` from 1.0059 to 0.8907 and wiped twenty minutes of correct estimation; the filter
half-noticed (σ widened 0.0024 → 0.0120) but nothing stopped it.

★★★ **Pooling two regimes produces confident false findings.** Repeatedly: parked vs moving cycles;
a pathological burst smeared across a run half and read as a trend; rotating and straight regimes
averaged into one. Pre-register the criterion and show convergence.

★★ **`opt/m` is not comparable across sessions** — two runs with identical settings gave 2.33 and
0.44. Steepness (mm/m, mm/rad) is stable because it measures a rate against motion inside a ramp
rather than a count against a threshold.

★★ **A protocol event is not evidence.** The affordance conversation had to be recorded per
*transition*, per *speaker* and per *candidate*; four separate defects in this system came from N
things sharing one "last seen" slot, so N−1 of them always looked new, and the record filled with
noise and displaced the transitions it existed to show.
