# ICRA Paper Review — Calibrated Uncertainty for Active Room-Layout Estimation

## Overall assessment

Yes — I think this has a real shot at ICRA, but I would **not submit it exactly as it stands**.

My current assessment is roughly:

> **As submitted now: borderline / weak reject.**  
> **After targeted revisions: potentially solid accept, perhaps 7–8/10 territory.**

The important thing is that **the core idea is stronger than the current experimental validation**.

---

## My overall reviewer-style score

| Criterion | Assessment |
|---|---:|
| Novelty | **8/10** |
| Robotics relevance | **8.5/10** |
| Technical idea | **8/10** |
| Experimental evidence | **6/10** |
| Clarity | **8/10** |
| Related work positioning | **6.5/10** |
| Reproducibility | **6/10** |
| Overall ICRA potential | **7–7.5/10** |

---

# 1. What is genuinely strong

The paper has a very good central idea:

> **Don't just estimate room geometry; estimate uncertainty that is calibrated enough for the robot to make decisions with it.**

That's a much more interesting robotics story than simply reporting 98.3% IoU.

The abstract communicates this well. You aren't presenting uncertainty as an auxiliary visualization; you're making it **actionable**: it determines whether corners are published and determines where the robot looks next. The paper reports 0.983 mean IoU and, more importantly, a median true error of 2.0 cm for publishable corners, with 86% lying within the declared 6 cm. [Source: manuscript, Abstract and Introduction.]

That gives you three layers:

**estimation → calibrated uncertainty → action**

That's exactly the kind of chain that makes a paper feel like robotics rather than merely geometric reconstruction.

### Particularly strong point

I really like the experiment where the robot sits still for 681 seconds and essentially says:

> "I have not learned anything new."

Then one rotation suddenly collapses the uncertainty.

The geometry is already almost correct while the uncertainty remains huge; after the rotation, the geometry barely changes but the uncertainty drops by about **32×**. The paper reports the worst-corner uncertainty changing from 0.989 m to 0.031 m, while the recovered room dimensions change only by millimetres.

That's an excellent demonstration because it makes the concept intuitive:

**repeated observations ≠ independent information.**

That's a good robotics contribution.

---

# 2. The calibration story is probably your strongest contribution

The distinction between:

- uncertainty that decreases with more observations, and
- irreducible/systematic uncertainty

is compelling.

You explicitly show that scatter about the fitted wall has a median of about 1.5 cm, while fitted-wall-to-annotated-wall discrepancy has a median of about 1.8 cm.

Then you have the much more dramatic stationary-robot result:

- repeating component: **9.52 cm**
- scattering component: **1.13 cm**
- ratio: **8.1×**

with the bias dominating in 97% of cases.

That's interesting.

More importantly, you don't stop at:

> "our uncertainty is calibrated."

You argue:

> **calibration can diagnose errors elsewhere in the estimation pipeline.**

The fact that you claim to find defects in three different modules is potentially a very nice secondary contribution.

I'd actually make this more prominent.

---

# 3. The biggest problem: where is the real robot?

If I were reviewing this for ICRA, this is the first thing I'd attack.

Your experiments are actually three different things.

### Experiment A — 594 Matterport room polygons

The observations are **simulated from the polygons**.

You explicitly state that the estimator never sees photographs and that a range sensor is driven through the annotated polygon.

That's useful for controlled evaluation, but it isn't strong evidence of a complete real-world system.

### Experiment B — Real Matterport depth

This validates the decomposition of sensor scatter versus systematic error.

Good.

But it does **not** validate the complete estimator.

### Experiment C — Closed-loop robot

Excellent conceptually.

But it is **Webots + a real LiDAR model**, rather than a physical robot with real LiDAR data.

You've actually put the limitation in the paper yourself: the real-depth experiment validates the noise model but not the full pipeline on recorded data.

That's honest, but it is also exactly what a skeptical reviewer will circle in red.

---

# 4. The experiment I would add if at all possible

You don't need a huge experiment.

You need something like:

## Real robot experiment

**3–10 real rooms**

For each:

1. Start from a partial observation.
2. Estimate room layout.
3. Report uncertainty.
4. Let the robot choose a viewpoint.
5. Move.
6. Update layout.
7. Repeat.
8. Compare predicted uncertainty against actual error.

For example:

| Room | Initial error | Initial σ | After motion | Final σ |
|---|---:|---:|---:|---:|
| 1 | 8.2 cm | 42 cm | 2.1 cm | 3.4 cm |
| 2 | ... | ... | ... | ... |

Even **five carefully selected physical environments** would substantially change how I perceive the paper.

Right now the story is:

> "Here is a theoretically motivated estimator that behaves beautifully in controlled/simulated experiments."

You want:

> **"Here is an estimator that behaves this way on a real robot."**

That difference could be the difference between borderline and accepted.

---

# 5. Second biggest weakness: baselines

This is the other thing I expect an ICRA reviewer to say.

You don't really have a strong baseline comparison.

You compare:

> propagated uncertainty only  
> vs  
> propagated uncertainty + σₘ

That's a very useful **ablation**, but it isn't really a competing method.

The paper has lots of related work, but the experimental section doesn't demonstrate:

> "Our approach is better than X."

I'd add at least **two baselines**, even if they're simple.

### Baseline 1 — empirical/static covariance

For example:

> fixed σ calibrated globally from the training/evaluation corpus.

This directly tests your claim that **one scalar precision isn't enough**.

### Baseline 2 — covariance without systematic term

You already have this one.

### Baseline 3 — heuristic viewpoint selection

For example:

- random rotation
- maximum geometric coverage
- conventional information gain

Then show that your uncertainty-driven action isn't merely producing the same result as a generic exploration heuristic.

---

# 6. Potentially dangerous mathematical issue: what exactly does σ mean?

This is something I'd fix **before submission**.

You write:

> "σ is the largest-axis standard deviation, so this tests a bound."

A reviewer with a probabilistic estimation background may immediately object.

A **standard deviation is not a bound**.

If σ is the largest eigenvalue's standard deviation, you need to be very precise about what probability statement you're making.

This is potentially quite important because **the entire paper is about calibrated uncertainty**.

You cannot afford ambiguity here.

I'd change the terminology from things like:

> "declared 6 cm"

and

> "bound"

to something mathematically explicit, e.g.:

> "largest-axis standard deviation"

and then report empirical coverage:

P(||e||₂ ≤ r)

for specified radii, or use a proper covariance ellipse / Mahalanobis-distance calibration.

That would make the calibration claim considerably harder to attack.

---

# 7. Statistical issue: independence

You have **4,344 matched corners from 594 rooms**.

That's impressive.

But a reviewer could say:

> "Those 4,344 samples aren't independent."

Four corners from the same room are correlated. Multiple corners from the same environment are correlated. The estimator and annotation process are also correlated at the room level.

So I would add **confidence intervals at the room level**, not merely report the aggregate 86.1%.

For example:

> 86.1% empirical coverage, 95% bootstrap CI [X, Y], bootstrapped by room.

That would make the calibration claim considerably more convincing.

---

# 8. The "5× variation" claim needs strengthening

This is an interesting claim:

> the irreducible component varies by 5× between environments and 2× between motion states.

But right now I'm not completely convinced that the paper demonstrates the generality implied by the language.

I'd want to know:

- How many environments?
- How many trajectories?
- How many rooms?
- How was the 5× calculated?
- Is this median-to-median?
- min/max?
- percentile ratio?
- confidence interval?

Because the conclusion:

> "a constant tuned in one environment is wrong in the next"

is quite strong.

The data may support it, but you need to **make the statistical argument airtight**.

---

# 9. The free-energy framing could either help you or hurt you

This is an interesting strategic point.

You currently make the free-energy formulation almost the philosophical backbone of the paper:

> geometry, model complexity and viewpoint selection are all expressed in the same currency.

That's elegant.

But an ICRA reviewer could ask:

> "Why do I need free energy for this?"

Several pieces sound like things that could be implemented using:

- Bayesian model selection,
- MAP estimation,
- information gain,
- covariance propagation.

And you even acknowledge that your viewpoint-selection rule coincides with information gain under the Gaussian model.

So the reviewer may conclude:

> "The free-energy formulation is mostly a unifying interpretation rather than a novel algorithm."

That's dangerous if you sell the paper as a **new free-energy approach**.

I'd instead sell it as:

### Main contribution

**Calibrated actionable uncertainty for online room-layout estimation.**

### Unifying formulation

**Variational/free-energy formulation provides a principled way to couple estimation, structure selection and active sensing.**

That's much safer.

---

# 10. The paper has a very good story

And I would preserve it.

The strongest narrative is:

### Problem

Room-layout systems report geometry but not trustworthy uncertainty.

↓

### Observation

Statistical covariance can become arbitrarily confident when observations are correlated.

↓

### Solution

Explicitly model the irreducible/systematic component.

↓

### Validation

594 rooms + real depth.

↓

### Robotics consequence

The uncertainty determines whether geometry is publishable.

↓

### Active consequence

The robot moves specifically to reduce the uncertainty blocking publication.

↓

### Diagnostic consequence

Calibration exposes defects elsewhere in the pipeline.

That's actually a **very coherent ICRA paper**.

---

# 11. Title

Current:

> **Calibrated Uncertainty for Active Room-Layout Estimation**

Good, but somewhat generic.

Possible alternatives:

### More ICRA-ish

**Actionable Uncertainty for Active Room-Layout Estimation**

### Stronger

**From Geometry to Action: Calibrated Uncertainty for Active Room-Layout Estimation**

### Emphasizing the real contribution

**Calibrated Irreducible Uncertainty for Active Room-Layout Estimation**

I actually like the current title, though. I wouldn't change it unless the final paper's emphasis changes.

---

# 12. The abstract is already quite good

I wouldn't radically rewrite it.

But I would reconsider the sentence:

> "no scalar precision can be correct for this problem"

That's a bold claim.

I'd change it to something more defensible:

> **"we show that a single global precision is insufficient: the irreducible error varies by 5× across environments and by 2× across motion states."**

Same punch, less philosophical overclaim.

---

# 13. Related work needs one more level of positioning

Your related work is broad, which is good, but there is a potential reviewer reaction:

> "This is basically active SLAM + Bayesian uncertainty + room layout."

You need a very explicit paragraph saying **what existing approaches cannot do**.

The central positioning argument should be approximately:

> Existing room-layout methods evaluate geometric accuracy but do not calibrate a per-corner uncertainty against realized error. Active SLAM methods optimize information gain but generally assume the covariance consumed by the acquisition policy is a meaningful representation of uncertainty. Our contribution is to connect these two: we calibrate the uncertainty of the layout representation and then use the calibrated quantity both as a publication criterion and as the objective for viewpoint selection.

That is, in my opinion, the **central positioning argument**.

---

# 14. References are currently a submission blocker

Your PDF literally contains things like:

> `[VERIFY]`

throughout the bibliography.

That absolutely cannot go into the submission.

Some references also lack venue/year information.

This makes the paper look unfinished even if the research is strong.

And, more importantly, it gives a reviewer an excuse to distrust the literature positioning.

**Fix every reference before submission.**

---

# 15. Remove some defensive language

There are places where the paper sounds like it's arguing with a reviewer before the reviewer has even objected.

For example:

> "should not be scored against these numbers in either direction."

and:

> "We report it rather than quoting the 0.14 IoU it yields..."

I understand why you wrote this — you're anticipating the obvious criticism about simulated data.

But it draws attention to the weakness.

I'd make it more neutral:

> "Because observations are simulated from annotated polygons, this experiment evaluates geometric inference and uncertainty calibration rather than appearance-based layout recognition."

Done.

No need to tell the reviewer what they shouldn't do.

---

# 16. The limitation section is unusually honest — keep that

You don't hide the MP3D failure. You explain why it happens and refuse to report the meaningless 0.14 IoU.

That's scientifically good.

But I would turn it from:

> "we couldn't do this"

into:

> **"This defines the boundary of the task evaluated by this work."**

The distinction matters.

---

# 17. Add an ablation table

Something like:

| Model | Systematic term | Active viewpoint | Structure selection | Coverage | Median error |
|---|---:|---:|---:|---:|---:|
| A | ✗ | ✗ | fixed | ... | ... |
| B | ✓ | ✗ | fixed | ... | ... |
| C | ✓ | ✓ | fixed | ... | ... |
| Ours | ✓ | ✓ | Bayesian | **...** | **...** |

This would make the contribution much easier to digest.

Right now the paper contains all the ingredients, but the reviewer has to reconstruct the ablation logic mentally.

Don't make them work.

---

# 18. Fig. 3 is especially strong

The figure on page 5 is doing real scientific work.

Without the systematic term, the median realized error is essentially flat as declared σ changes.

With the systematic term, the relationship becomes much more sensible.

That's probably your **best calibration figure**.

I would make it larger if possible.

If I were cutting anything to enlarge it, I'd sacrifice some prose before sacrificing this figure.

---

# 19. What an ICRA reviewer might actually write

Something like:

> **Strengths:**  
> The paper addresses an important but underexplored problem of calibrated uncertainty for online room-layout estimation. The distinction between reducible measurement noise and irreducible systematic error is interesting and well motivated. The active sensing experiment provides an intuitive demonstration that the estimator avoids falsely increasing confidence under repeated observations. The calibration analysis across 594 layouts is compelling.

Then:

> **Weaknesses:**  
> The main evaluation is based on simulated observations from annotated polygons rather than real sensor data. The closed-loop experiment is performed in simulation, and only the sensor noise decomposition is validated on real depth. Consequently, it is unclear whether the complete method provides calibrated uncertainty on a physical robot. The method is also not compared against meaningful uncertainty or active exploration baselines. Some claims about calibration and scalar precision appear stronger than the statistical evidence presented.

That reviewer could easily give you:

**6 / 10 — Weak Accept or Weak Reject.**

And that's exactly why I think you're **close**.

---

# 20. The three most important changes

If you can do only **three things** before submission, do these:

### 🔴 Priority 1 — real robot validation

Even a relatively small physical experiment.

### 🔴 Priority 2 — baseline comparison

Especially a globally calibrated scalar σ and a conventional information-gain policy.

### 🔴 Priority 3 — make the statistical calibration watertight

Clarify exactly what σ means, use proper coverage terminology, and preferably give confidence intervals.

Then:

### 🟠 Priority 4

Clean every `[VERIFY]` reference.

### 🟠 Priority 5

Tone down the "no scalar precision can be correct" generalization unless you strengthen its statistical evidence.

### 🟢 Priority 6

Improve the paper's positioning around:

**calibrated uncertainty → actionable decision → active sensing.**

---

# Bottom line

**Yes, I would absolutely consider submitting this to ICRA.**

I don't think the problem is that the idea is too weak for ICRA. **The idea is ICRA-level. The question is whether the experimental evidence is currently ICRA-level.**

Right now I'd characterize it as:

> **Strong idea + strong narrative + interesting calibration results + insufficient real-world validation.**

That's a fixable situation.

The paper is already close to having a strong story. The main task is to make the evidence as strong as the claim.
