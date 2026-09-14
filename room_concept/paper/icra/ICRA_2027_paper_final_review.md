# ICRA 2027 Paper Review — Calibrated Uncertainty for Active Room-Layout Estimation

## Overall assessment

The revised 8-page paper is substantially stronger than the earlier version. The central contribution is now clear and coherent:

> **Calibrated, actionable uncertainty for online room-layout estimation, where uncertainty is used both to decide which corners are publishable and to drive active viewpoint selection.**

The paper combines:

- calibrated per-corner uncertainty,
- a decomposition of reducible scatter and irreducible/systematic error,
- active viewpoint selection,
- evaluation on **594 annotated real room shapes**,
- validation of the depth-noise decomposition on real Matterport depth,
- a closed-loop Webots simulation with a LiDAR sensor model,
- a heuristic exploration ablation,
- and a diagnostic experiment showing that the uncertainty formulation can expose implementation defects.

### Current estimated ICRA level

**Around 7.5–8/10 overall**, roughly **Weak Accept / Accept borderline (B/B+)**.

The lack of a physical robot experiment is a limitation, but it should not by itself be fatal. The current experimental story is coherent because the paper explicitly separates geometric inference, real-depth noise validation, and closed-loop simulation.

The main remaining risk is **not lack of experiments**. It is **overclaiming or leaving a few technical details insufficiently justified**.

---

# Major strengths

## 1. Strong central contribution

The paper does more than report an uncertainty value.

The uncertainty is actually consumed by the system:

1. it determines whether corners are published;
2. it drives active viewpoint selection;
3. it reveals whether uncertainty is reducible or irreducible;
4. and it can diagnose problems in the underlying estimation pipeline.

That gives the uncertainty formulation practical relevance rather than making it merely an auxiliary confidence score.

---

## 2. Excellent closed-loop example

The closed-loop simulation is one of the strongest parts of the paper.

The robot can remain parked for a long period while uncertainty saturates. A single in-place rotation then produces a dramatic uncertainty reduction while the estimated geometry changes only by millimetres.

The current reported result is approximately:

- parked for **665 s**;
- worst-corner uncertainty around **0.989 m**;
- after a **319°** in-place turn;
- first publishable frame after **24.5 s**;
- worst-corner uncertainty reduced to **0.048 m**;
- approximately **21× reduction**;
- robot centre remains within a **4.4 cm** disc and returns within about **1 mm**.

This is a compelling qualitative demonstration that the uncertainty is actionable.

A particularly good aspect is that the paper does **not** pretend this is a physical-robot experiment.

### Recommended wording change

The current section title:

> **D. Closed loop on a real sensor: the claim, demonstrated**

is misleading because the experiment is performed in Webots with a simulated LiDAR model.

Use instead:

> **D. Closed-loop simulation with a LiDAR sensor model**

or:

> **D. Closed loop with a realistic LiDAR model**

This removes an avoidable reviewer objection.

---

# 3. Strong calibration story

The calibration section is considerably improved.

The paper distinguishes:

- reducible observation scatter;
- systematic/irreducible error;
- uncertainty propagation;
- empirical coverage;
- and a global scalar uncertainty baseline.

The calibration table is useful because it demonstrates that adding the systematic term substantially improves empirical coverage.

The paper reports:

| Metric | No σ_m | With σ_m = 2 cm |
|---|---:|---:|
| ≤1σ | 46.7% | 65.3% |
| ≤2σ | 70.5% | 85.0% |
| ≤3σ | 81.2% | 91.2% |
| Publish | 89.0% | 89.0% |
| Holds | 85.8% | 86.1% |

There are also room-level bootstrap confidence intervals.

This is much more convincing than simply stating that the covariance "looks reasonable."

---

# 4. The global scalar comparison is useful

The comparison with a single global scalar σ_c is a good idea because it directly tests the paper's claim that uncertainty should be spatially and contextually meaningful.

The fitted global value is:

> **σ_c = 0.0317 m**

and it is fitted so that its empirical 1σ coverage matches the proposed method.

The paper then shows that a constant scalar cannot distinguish corners by uncertainty.

### Important correction

The current manuscript says that the constant predictor has rank correlation zero with realised error "by construction."

That is technically incorrect.

A constant predictor has **undefined Spearman rank correlation**, because it has no rank variation.

Replace the current statement with:

> **Being constant, it provides no ranking of corners by realised error.**

This is cleaner and mathematically correct.

---

# 5. The heuristic exploration ablation is a very good addition

The new frontier-and-coverage heuristic baseline is valuable because it addresses an obvious reviewer question:

> "Maybe any reasonable exploration strategy would achieve the same result."

The paper reruns the 594-room corpus under the same estimator, sensor, seeds, and 900-frame budget.

For the 583 rooms completed by both methods:

- mean IoU difference: **−0.0007**
- 95% CI: **[−0.0019, +0.0004]**
- heuristic wins: **263 vs 251**
- failures: **8 vs 6**
- median frames to publish difference: **6 frames [−14,+28]**
- no difference in corner-count band
- all rooms hit the frame cap.

This is an honest null result.

It supports the narrower claim that, **under this particular endpoint and budget, the proposed uncertainty-driven exploration does not improve final geometric accuracy over the heuristic baseline**.

That is actually useful evidence because it prevents the paper from making a stronger claim than the data support.

---

# 6. The 594 annotated floor plans provide useful geometric diversity

The corpus is a strong part of the evaluation.

It contains **594 real room shapes**, giving the method a realistic distribution of building geometries rather than relying entirely on synthetic polygons designed by the authors.

The results are strong:

- mean IoU: **0.9827**
- median IoU: **0.9870**
- 10th percentile IoU: **0.9690**
- IoU ≥ 0.95: **95.2%**
- IoU ≥ 0.90: **98.8%**
- median Hausdorff distance: **0.0705 m**
- mean truth-interior coverage: **0.9996**
- failures: **8/594**

The limitation is that the observations are simulated from annotated polygons, so this primarily evaluates **geometric inference and uncertainty calibration**, not full appearance-based room-layout recognition.

The paper now states that limitation, which is good.

---

# 7. Real depth validation is valuable

The Matterport depth experiment is a useful intermediate validation layer.

The paper reports:

- **588 wall segments**
- **120 rooms**
- median modal offset: **7.5 mm**
- scatter about fitted wall median: **1.49 cm**
- p90 scatter: **3.82 cm**
- fitted-vs-annotated wall median: **1.81 cm**
- p90: **5.72 cm**

This supports the claim that the simulated noise model is reasonably realistic and even slightly conservative relative to the real depth data.

However, this experiment should be framed as **noise-model validation**, not full pipeline validation.

---

# 8. The diagnostic contribution is unusually interesting

The calibration formulation does not merely produce a confidence number; it can expose implementation problems.

The paper identifies three defects.

One particularly convincing result is that removing a duplicated systematic term and mapping the surviving term changes the NIS from approximately:

> **0.47 → 1.03**

That is a strong diagnostic demonstration.

This is arguably one of the more distinctive aspects of the paper because it shows a practical benefit of having calibrated uncertainty beyond confidence reporting.

---

# Main issues to fix before submission

## 1. Fix the Spearman statement — definite fix

### Current idea

The manuscript says the constant global uncertainty has zero rank correlation with realised error "by construction."

### Problem

Spearman correlation with a constant predictor is not zero; it is **undefined** because the predictor has no rank variation.

### Replacement

Use:

> **Being constant, it provides no ranking of corners by realised error.**

This should definitely be changed.

---

## 2. Clarify the 594 → 586 transition

The paper evaluates 594 rooms, but the corner-level calibration uses:

> **4,344 matched corners from 586 rooms**

A reviewer may immediately ask where the eight rooms went.

### Recommended sentence

Add:

> **The eight rooms without a closed polygon are excluded from corner-level calibration, leaving 586 rooms and 4,344 matched corners.**

This is a tiny change but removes an unnecessary ambiguity.

---

## 3. Fix the 18 vs 22 corner inconsistency

There is an internal inconsistency:

- the text in §V-A describes the corpus as containing rooms with **6–18 corners**;
- the Figure 2 caption says the shapes contain **6–22 corners**.

One of these must be corrected.

**Make the number identical everywhere.**

---

## 4. Clarify that the global scalar baseline is deliberately favourable

The global scalar is fitted **post hoc on the same corpus**.

That is fine as a diagnostic comparison, but it is not a held-out baseline.

A reviewer could object that the scalar is being given favourable information.

### Recommended wording

Add:

> **For a deliberately favourable comparison, the scalar is fitted post hoc on the same corpus so that its empirical 1σ coverage exactly matches ours.**

This makes the experimental choice transparent.

If you have time and the implementation is easy, a cross-validated scalar would be even stronger, but it is **not essential** for submission.

---

## 5. Justify the 6 cm publication gate

The paper currently uses:

> σ ≤ 6 cm

as the publishability threshold.

A reviewer may ask:

> "Why 6 cm?"

If it is an application-level tolerance rather than a parameter tuned to maximise the reported results, say so explicitly.

### Recommended wording

> **We use a fixed 6 cm uncertainty gate as an application-level tolerance; it is specified independently of the evaluation results rather than tuned on the test corpus.**

If you can easily compute it, a small sensitivity analysis such as:

- 2 cm
- 4 cm
- 6 cm
- 8 cm
- 10 cm

would make the choice more robust.

But I would **not delay submission just to add this** unless it is trivial.

---

# 6. Clarify the free-energy/action relationship

The abstract currently gives the impression that:

> parameters, structure, and viewpoint selection are all obtained by minimising one variational free energy.

That is conceptually a little too compressed.

Inference/model comparison and active viewpoint selection are related but not literally the same optimisation operation.

### Safer formulation

Use:

> **Inference and model comparison minimise the variational free energy in (1); viewpoint selection minimises the corresponding expected-free-energy epistemic term, which reduces to expected information gain under the Gaussian model.**

This is more technically precise and will make the theoretical story easier for reviewers to follow.

---

# 7. Rename "Why no scalar precision can be correct"

The current heading is:

> **Why no scalar precision can be correct**

That is stronger than what the experiment actually proves.

A better title is:

> **Why a single global precision is insufficient**

This matches the evidence and avoids inviting a reviewer to find a counterexample.

---

# 8. Soften the "5× between environments" claim

The paper currently says that the irreducible error varies by **5× between environments**.

If the 1.8 cm and 9.5 cm values actually come from different annotation/settings rather than a set of independently sampled environments, the word "environments" may overstate the evidence.

A safer formulation is:

> **The measured irreducible component differs by 5× between the annotated-building and hand-traced-layout settings, and by 2× between stationary and moving operation.**

If there are genuinely multiple independent environments behind the 5× statistic, then make that explicit by stating the number of environments and how the statistic was computed.

---

# 9. Soften the causal claim about the sensor

The manuscript currently makes a statement along the lines of:

> **the systematic component is a property of the map, not of the sensor**

That is too categorical.

The experiment supports the weaker statement that independent sensor noise does not explain the observed systematic component.

### Recommended wording

> **The systematic component is not explained by independent sensor noise alone.**

This preserves the important conclusion without claiming more causal certainty than the experiment establishes.

---

# 10. Soften the cross-environment generalisation claim

The current phrase:

> **a constant tuned in one environment is wrong in the next**

is rhetorically strong.

A safer scientific version is:

> **a constant calibrated in one setting need not remain calibrated in another.**

Same message, much harder for a reviewer to attack.

---

# 11. Add one sentence separating the three evaluation levels

Because there is no physical robot experiment, I strongly recommend making the evidence structure explicit.

Add near the beginning of §V:

> **We separate evaluation into three levels: controlled geometric inference on 594 real room shapes, validation of the noise decomposition on real depth, and closed-loop behaviour using a realistic LiDAR model in simulation.**

This is a very useful defensive sentence.

It tells the reviewer that the authors understand exactly what each experiment does and does not establish.

---

# Real-robot experiment: do not panic

The absence of a physical-robot experiment remains the biggest obvious limitation.

However, given that you **cannot add one**, I would not try to compensate by adding random extra experiments.

The paper already has a coherent evidence chain:

1. **594 real building geometries**
2. **calibration against realised corner errors**
3. **real Matterport depth validation**
4. **closed-loop LiDAR simulation**
5. **heuristic exploration ablation**
6. **diagnostic failure analysis**

The correct strategy is therefore to be extremely explicit about the scope of each experiment rather than pretending the simulation is a real-world experiment.

---

# Current closed-loop results

The closed-loop experiment gives:

| Condition | Long side | Short side | Worst corner σ | Worst edge σd |
|---|---:|---:|---:|---:|
| Parked, frame 0 | 5.994 m | 4.015 m | 0.988 m | 0.500 m |
| Parked, 665 s | 5.995 m | 4.012 m | 0.989 m | 0.500 m |
| After turn | 5.994 m | 4.019 m | 0.048 m | 0.019 m |

The 319° rotation takes 24.5 s.

The final geometry remains very close to the ground truth:

- long-side error: about **−4 mm**
- short-side error: about **+5 mm**

This is a nice example of uncertainty collapsing without a corresponding geometric jump.

One caveat is already correctly acknowledged: the map is gauge-free, so there is no ground-truth robot pose and therefore no absolute heading error reported.

---

# Calibration interpretation

The calibration section is one of the strongest technical parts, but be precise with terminology.

The paper correctly states that the first three coverage values are:

> P(||e||₂ ≤ kσ)

and are **not probabilistic containment bounds**.

This distinction is important.

Calling σ a standard deviation or empirical uncertainty scale is defensible.

Calling it a guaranteed bound would not be.

Keep the explicit clarification that it is **not a bound**.

---

# Published vs withheld corners

The separation between published and withheld corners is useful:

### Published

- median error: **2.02 cm**
- 95% CI: **1.85–2.20 cm**
- within 6 cm: **86.1%**

### Withheld

- median error: **3.64 cm**
- 95% CI: **3.12–4.46 cm**
- within 6 cm: **63.7%**

The confidence intervals for the medians do not overlap.

This supports the claim that the uncertainty gate is useful for separating more reliable from less reliable corners.

However, the Spearman correlation is only:

> **ρ = 0.125**, 95% CI **[0.088, 0.161]**

This is small.

The manuscript does the right thing by not claiming that σ provides a highly accurate ranking of all corners.

The appropriate conclusion is:

> **The uncertainty is useful for the publish/withhold decision, but only weakly informative for fine-grained ranking of corner error.**

That is a credible result.

---

# Why the scalar baseline does not replace the proposed method

The argument should be framed carefully.

The point is not:

> "A scalar is mathematically impossible."

The point is:

> **A single scalar cannot represent spatially varying uncertainty or rank corners according to their predicted reliability.**

This is exactly what the corner-level uncertainty enables.

So the section should focus on **insufficient expressiveness**, not impossibility.

---

# Reference quality

The previous `[VERIFY]` placeholders have now been removed.

That is good.

There are currently 36 references and they are generally appropriate to the paper.

Before final submission, I would still do one final pass over:

- author names;
- exact paper titles;
- venue names;
- publication years;
- page/article numbers;
- conference names;
- capitalization;
- DOI formatting if used.

This is a final polish step, not a major submission blocker.

---

# Page/layout assessment

The current paper is **8 pages**, including references.

The pages are dense but professionally formatted.

### Page 1

Dense abstract/introduction, but Figure 1 is readable and the main idea comes through.

### Page 2

Dense but clear. The related-work positioning paragraph is particularly strong.

### Page 3

Good balance. Table I and the exploration ablation are readable.

### Page 4

Table II is substantially improved and legible. The page is text-dense but still manageable.

### Page 5

Probably one of the strongest pages visually.

Figure 3 communicates the calibration result well, Table III is clear, and Figure 4 is informative.

### Page 6

Table IV and the limitations section are readable, although the page is dense.

### Page 7

Acknowledgments and references.

### Page 8

Only the final references occupy the page, leaving some whitespace.

That is completely acceptable given the 8-page limit.

I would **not** force additional content onto page 8 merely to eliminate whitespace.

---

# Recommended final priority list

If you are close to the deadline, do these in this order:

## Must fix

1. **Correct the Spearman statement.**
2. **Resolve the 18 vs 22 corner inconsistency.**
3. **Explain why 594 rooms become 586 for corner calibration.**
4. **Rename "Why no scalar precision can be correct" to "Why a single global precision is insufficient."**
5. **Rename "Closed loop on a real sensor" to make clear it is simulation with a LiDAR model.**
6. **Justify the 6 cm gate as a fixed application-level tolerance.**

## Strongly recommended

7. Clarify that the global scalar is fitted post hoc on the same corpus as a deliberately favourable comparison.
8. Add the three-level evaluation sentence.
9. Clarify the relationship between variational free energy and expected free energy/information gain.
10. Soften the 5× "between environments" claim if the underlying statistic does not truly come from multiple independent environments.
11. Soften "property of the map, not of the sensor."
12. Soften "wrong in the next environment."

## Optional if time permits

13. Run a small sensitivity analysis for the 6 cm gate.
14. Run a cross-validated global-scalar baseline.
15. Do a final exact bibliographic verification.

---

# What I would NOT change

I would **not**:

- add experiments just for the sake of adding experiments;
- pretend the Webots experiment is a physical-robot experiment;
- remove the honest null-result exploration ablation;
- remove the diagnostic defect experiment;
- overstate the Spearman result;
- claim that the uncertainty is a guaranteed probabilistic bound;
- add a large amount of new related work at the last minute;
- substantially restructure the paper now.

The current paper has a recognizable and coherent story.

---

# Bottom line

The revised manuscript is now in a much better position than the earlier version.

Its strongest story is:

> **Room-layout estimates are only useful when their uncertainty is calibrated and actionable. The proposed formulation separates reducible observation scatter from systematic error, calibrates corner-level uncertainty against realised geometry, uses that uncertainty to decide which geometry is publishable and where the robot should look next, and demonstrates that the resulting uncertainty can both drive active reduction and diagnose estimator defects.**

The main remaining work is **precision of claims and consistency**, not a wholesale redesign.

If the six "must fix" items above are addressed, I would be comfortable submitting this version rather than trying to invent another major experiment at the last minute.

## Final estimated score after those fixes

**~7.5–8/10**

Likely reviewer range:

- **B+ / Accept** from reviewers who value the uncertainty + active estimation contribution;
- **B / Weak Accept** from reviewers concerned about the lack of physical-robot validation;
- potentially **B- / borderline** from a reviewer who expects a stronger real-world end-to-end demonstration.

The strongest defence against the latter is not another paragraph of claims; it is the paper's current disciplined separation of what is demonstrated by each evaluation layer.
