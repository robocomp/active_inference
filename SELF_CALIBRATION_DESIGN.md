# Self-calibration as inference: design note for the thesis

**Status.** Design, not implementation. The measurements and the failure modes cited here are real and
referenced; the architecture is proposed. Written for chapter 8 (`room-concept.tex`) with a hook into
chapter 11 (affordances / EFE).

**The claim.** A robot that localises against a map it already trusts has, in that map, everything it
needs to calibrate its own proprioception — and the calibration is not a separate procedure bolted
beside the estimator but *the same free energy, minimised over parameters instead of states*. The
practical consequence is that most of `SIM2REAL_CALIBRATION.md` should not need a human at all.

---

## 1. Why the room is enough

Dead reckoning cannot calibrate itself: an odometry error and a true motion are indistinguishable from
inside the odometry. What breaks the symmetry is an **absolute** reference, and the localiser already
has one — a fixed room polygon, against which the SDF anchors the posterior. So the posterior is an
independent witness of true motion, in the one sense that matters: its error does not grow with
distance travelled.

Two consequences, and the second is the interesting one:

**Rotation needs no reference at all.** A pivot returning to its starting heading has turned exactly
2πN. That is a fact about turning, not about the room, so it survives *any* map error. The robot need
only detect closure, which it can do from the SDF-anchored posterior — a quantity independent of the
odometry under test. Measured this way, twice: the gyro reads 1.0114 and 1.0172 against truth, and the
wheel channel 1.1450, on a base whose kinematic constants are provably correct.

**Translation can use the room as a ruler.** The only manual step in the current procedure is a
tape-measured leg. It is unnecessary: drive from near one wall toward another and the SDF knows the
displacement. The map that makes localisation possible also makes the calibration self-contained.

> ⚠ **And that is exactly where the limit is.** Self-calibration against the map inherits the map's
> errors. A room polygon 2% too large hands 2% straight to the translational scale, confidently.
> Rotation is immune; translation is only ever as good as the survey. This asymmetry should be stated
> in the thesis rather than discovered by a reader — it is the honest boundary of the idea.

---

## 2. The architecture: parameters are states on a slower timescale

Active inference already distinguishes **state inference** (fast, per observation) from **parameter
learning** (slow, across observations) as two rates of descent on one functional. Localisation is the
first. Calibration is the second. Writing them as one object is not an analogy; it is the same
`F` with a different partition of what is held fixed.

The window objective already reads

```
F = L_boundary(X_0) + Σ L_obs(X_k) + Σ L_motion(X_{k-1}, X_k) + Σ L_corner(X_k)
```

and the motion term is a precision-weighted quadratic on the discrepancy between the pose increment and
the measured one. **Calibration is what happens when the motion term's own parameters join the state
vector** — the scale errors `s_ω`, `s_v` become variables with slow random-walk priors, estimated
jointly by the same Gauss-Newton solver.

### 2.1 The machinery already exists

This is not speculative. `se2_preintegration.h` computes, for every interval, both the covariance and
the **scale Jacobians** `∂Δ/∂s_ω` and `∂Δ/∂s_v`, propagated by the same recursion:

```
J_{s,i} = A_i J_{s,i-1} + b_i
```

They are currently used only to build a rank-structured contribution to the covariance,
`Σ += σ_s² · J_s J_sᵀ`, which is the *marginalisation* of an unknown constant scale. Promoting the
scale from a marginalised nuisance to an inferred variable is the same Jacobian used differently:
register `s` in the solver's `VarIndex`, give the motion factor a third variable slot with Jacobian
`J_s`, and delete the outer product from Σ. The solver addresses variables through an index rather than
a hard-coded stride precisely so this is an addition rather than a rewrite.

### 2.2 Why this succeeds where the previous attempt failed

An online motion-model learner existed before and was measured and **rejected**: switching it on
collapsed early exit while rotating from 91.6% to 68.5% and drove the fused weight on the encoder from
0.55 to 0.10. Its diagnosis is precise and it matters for the thesis, because it is the argument for
inference over heuristics.

That learner attributed a **post-optimisation window-pair residual** to the odometry. But that residual
already contains the optimiser's own correction, so the odometry was charged for the SDF's work and
concluded to be uninformative. It is a credit-assignment error, and no amount of gating fixes it.

**A variable cannot make that error.** Credit assignment is what the Hessian does, jointly, and the
optimiser cannot double-count a quantity it is solving for. The distinction is not a detail of
implementation — it is the reason to prefer inference to adaptation, and it is empirically supported by
a measured failure rather than asserted.

### 2.3 What is genuinely harder: the noise densities

The scales and the densities are **not the same kind of parameter**, and the thesis should say so.

- A **scale** biases the *mean* of the residual. It is a location parameter; it joins the state vector
  and the existing machinery estimates it.
- A **density** sets the *spread*. It is a variance component. Adding it to the state vector does not
  work: the gradient of `F` with respect to a variance is not the gradient of a quadratic, and the mode
  of a variance is not what a Gauss-Newton step finds.

The density wants **evidence maximisation**, which is available in closed form for the motion factors.
Writing the interval error as `e(T) = s·Δ(T) + ε`, `ε ~ N(0, σ²T)`:

```
F(s, σ) = Σ_k [ (e_k - s Δ_k)² / (2σ² T_k) + ½ log(σ² T_k) ] + const
```

`∂F/∂s = 0` gives a weighted slope; `∂F/∂σ = 0` gives `σ² = (1/n) Σ (e_k - s Δ_k)² / T_k`. Both closed
form. The `log` term is what makes it well posed — the quadratic punishes a covariance that is too
loose, the log-determinant one that is too tight, and the optimum is the honest width.

★ **So "calibrate the model" and "maximise the evidence for the model" are the same computation.** That
is the sentence worth putting in the chapter: the covariance is inferred under the functional the
localiser already minimises, not tuned beside it. `tools/motion_calib.cpp` implements exactly this and
is validated against injected truth at 99% / 102% / 84% / 93% recovery on `σ_v, σ_ω, s_v, s_ω`.

The remaining design question is only *when* the M-step runs: per window (an EM that never converges
because the data keeps arriving), or on an accumulating batch with a forgetting factor. The second is
closer to what "learning on a slower timescale" means.

---

## 3. Calibration as an epistemic affordance

This is the part that closes a loop the thesis already argues for elsewhere, and it is the strongest
version of the idea.

The expected-free-energy machinery already selects actions by expected information gain, and it already
does so over **object** uncertainty (next-best-view). Nothing in it requires the uncertain thing to be
external. A parameter of the robot's own generative model is an equally legitimate target.

Then:

- The posterior over `s_ω` is wide.
- Turning is the action whose expected information gain about `s_ω` is maximal — because
  `∂Δ/∂s_ω` is proportional to `Δθ`, so the parameter's Jacobian *is* the rotation. The sensitivity of
  the observation to the parameter is literally the amount of turning.
- Therefore the robot pivots. Not because a procedure told it to, but because that is the
  information-maximising action available.

The same holds for translation and the room-as-ruler: driving between two walls is the action that most
sharpens `s_v`, and the polygon supplies the likelihood.

**Two things make this more than a rhetorical flourish.** First, the epistemic value is computable from
objects the system already builds — `J_s` is the sensitivity, and `Σ_s` the prior width, so the expected
reduction in parameter entropy under a candidate manoeuvre is a closed-form quantity, not a simulation.
Second, it unifies the two halves of the thesis under one rule: the same expected-free-energy criterion
that decides *where to look* decides *what to measure about oneself*. Perception and proprioception stop
being separately-engineered pipelines.

The closing image is worth keeping: **a robot can discover that it turns 16% more than it believes, from
nothing but a wall and the decision to spin.**

---

## 4. What it would take, in order

| step | what | status |
|---|---|---|
| 1 | Scale as a state: register `s` in `VarIndex`, third variable slot on the motion factor | machinery exists (`J_s`), unimplemented |
| 2 | Keep `s` OUT of the marginalisation, with a random-walk prior | design decided, see §5 |
| 3 | Density by evidence maximisation on an accumulating batch | closed form, implemented offline in `motion_calib` |
| 4 | Observability as precision, not as a gate | see §5 |
| 5 | Epistemic manoeuvre selection over parameter entropy | EFE machinery exists; parameter target is new |

---

## 5. The two design hazards, both already met once

**Marginalisation.** A scale spans the whole window and beyond, so when the oldest pose leaves, the
Schur complement involves it, and freezing its Jacobian at a stale value is *precisely* the mechanism
that once made the boundary prior ratchet monotonically 0 → 559 and jail the estimate until the window
was flushed. The cure is known and standard in visual-inertial systems: keep the parameter **out** of
the marginalisation as a persistent variable with its own prior, never marginalised. This is a
first-estimates-Jacobians consistency problem and should be cited as one.

**Observability.** A scale is identifiable only while moving; parked, only the random walk acts and the
estimate drifts. The obvious fix is a motion gate, and the thesis should resist it for the same reason
it resists every other threshold: gates are the shape the model takes when the modelling has not been
done. The AI2-shaped answer is that the parameter's prior precision should *grow* as the manoeuvre
stops exciting it — which falls out of the same Jacobian, since `J_s → 0` when `Δθ → 0` means the
observation carries no information about `s` and the posterior should relax to the prior on its own.
Written that way there is no gate: the update is simply uninformative, which is the correct behaviour
rather than an imposed one.

---

## 6. Honest limits, for the discussion section

1. **The map bounds the translation calibration.** Rotation is immune (2πN is geometry-free);
   translation inherits every survey error. Self-calibration cannot discover that the room is 2% wrong.
2. **A parameter that is never excited is never learned**, and the robot may have no task-driven reason
   to pivot. This is exactly what §3 answers — but it means the epistemic term is not decoration, it is
   what makes the scheme work at all.
3. **Simulation cannot validate the noise densities.** Measured: the bridge's odometry is the
   supervisor's ground truth at σ_v = 3.9e-7 m·s^−½, five to six orders below what the motion model
   assumes. Densities calibrated there would drive the motion prior to zero variance and make the SDF
   inert. What simulation *can* do is score the procedure against a known injected error, which is how
   `motion_calib` was validated.
4. **Two timescales can fight.** If the parameter adapts as fast as the state, a slow sensor fault and a
   real motion become confusable. The random-walk prior's width is what separates them, and it is
   itself a modelling choice that no measurement here constrains.
