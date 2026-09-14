# ICRA 2027 — Final Review of Latest Version

## Overall assessment

I reviewed the new 8-page version page by page, including figures and tables, and checked the bibliography externally.

This version is **noticeably stronger than the previous one**. Most of the corrections we discussed are now in place.

### Current estimated ICRA level

**~8/10 overall**, roughly **Weak Accept / Accept territory**.

Likely reviewer range:

- **B+ / Accept** — likely from reviewers who value the uncertainty + active estimation contribution.
- **B / Weak Accept** — very plausible.
- **B− / Borderline** — mainly from a reviewer demanding physical-robot validation.

I would **submit this rather than trying to add another major experiment**.

The manuscript now explicitly separates its evaluation into three levels: geometric inference and uncertainty calibration on 594 room shapes, real-depth validation of the noise decomposition, and closed-loop simulation with a LiDAR sensor model. It also explicitly acknowledges that no experiment runs on a physical robot.

---

# 1. Major improvements

You have now fixed essentially all major conceptual issues discussed previously:

- the **594 → 586** calibration transition is explicitly explained;
- the **6 cm gate** is justified and sensitivity-tested;
- the global scalar baseline is explicitly described as favourable;
- **5-fold cross-validation** has been added for the scalar baseline;
- the Spearman issue is correctly stated as **undefined for a constant predictor**;
- the closed-loop experiment is correctly called **simulation with a LiDAR sensor model**;
- the three evaluation levels are explicitly separated;
- the 5× claim is carefully qualified;
- the map/sensor causal claim has been softened;
- the scalar section has been renamed appropriately;
- the absence of a physical robot experiment is explicitly acknowledged.

The calibration section is now much harder for a reviewer to attack.

---

# 2. Score by category

| Aspect | Assessment |
|---|---:|
| Novelty | 8/10 |
| Technical coherence | 8/10 |
| Experimental evidence | 7.5/10 |
| Uncertainty/calibration | 8.5/10 |
| Active robotics relevance | 8/10 |
| Writing/presentation | 8/10 |
| Reproducibility/clarity | 7.5–8/10 |
| **Overall** | **~8/10** |

---

# 3. Strongest part: calibration

The paper now has a very complete uncertainty story:

1. proposes a per-corner uncertainty;
2. tests it against realised error;
3. adds the irreducible term;
4. compares against a global scalar;
5. cross-validates that scalar;
6. evaluates publish/withhold performance;
7. evaluates ranking;
8. sweeps the gate.

The calibration table is particularly useful:

| Metric | No σm | With σm = 2 cm |
|---|---:|---:|
| ≤1σ | 46.7% | 65.3% |
| ≤2σ | 70.5% | 85.0% |
| ≤3σ | 81.2% | 91.2% |
| Publish | 89.0% | 89.0% |
| Holds | 85.8% | 86.1% |

The paper also uses room-level bootstrap confidence intervals rather than treating all corners as independent.

---

# 4. Global scalar comparison

The global scalar baseline is now handled correctly and transparently.

The paper gives:

> σc = 0.0317 m

and explains that it is fitted post hoc on the same corpus to match the proposed method's empirical 1σ coverage.

You then add 5-fold cross-validation:

> σc = 0.0318 m ± 0.0002 m  
> held-out 1σ coverage = 65.4%

This removes the obvious objection that the scalar baseline only looks bad because it was fitted on the evaluation set.

The key conclusion is now appropriately framed:

> **A scalar can be calibrated on average and still be unable to distinguish one corner from another.**

---

# 5. Spearman issue is fixed

The new wording is mathematically correct:

> “Being constant, it provides no ranking of corners by realised error at all: a rank correlation is not merely zero but undefined, since the predictor has no rank variation.”

**Keep this. Do not change it back to “zero.”**

---

# 6. The 6 cm gate is well defended

The paper now states that:

- the gate represents the downstream consumer's requirement;
- it was fixed before evaluation;
- it was not tuned on the evaluation corpus;
- sweeping the gate leaves the published median almost unchanged;
- the withheld median rises monotonically;
- below 2 cm the gate becomes vacuous.

This is enough justification.

### Tiny wording change

You write:

> “The 6 cm bound is the downstream consumer's requirement...”

Change **“6 cm bound”** to:

> **“6 cm gate”**

or:

> **“6 cm tolerance”**

This avoids confusion with the explicit statement elsewhere that σ is not a probabilistic bound.

---

# 7. 594 → 586 is fixed

The table now explicitly says:

> “4344 matched corners, from the 586 of 594 rooms that closed a polygon — the other 8 have no corners to match.”

Excellent.

---

# 8. Important remaining inconsistency: Contribution 3

Later in §V-E you correctly write:

> “These are two settings rather than a sample of environments...”

However, **Contribution 3 still says**:

> “the irreducible component varies by 5× across environments and 2× between motion states”

and describes this as:

> “a general property of map-relative estimation rather than a quirk of our system.”

That is stronger than the evidence presented later.

### Recommended replacement

> **3) Evidence that a single global precision is insufficient for this problem: the irreducible component differs by 5× between the two measured settings and by 2× between motion states, suggesting that systematic error is context-dependent rather than captured by a fixed scalar.**

**Definitely change this.**

---

# 9. Figure 3 has a small inconsistency

The Figure 3 caption currently says:

> “Every matched corner of 594 annotated floor plans...”

But matched corners come from **586 rooms**, because eight rooms did not produce a closed polygon.

### Recommended replacement

> **“All 4,344 matched corners from the 586 annotated floor plans that produced a closed polygon...”**

---

# 10. Closed-loop experiment

The title is now correctly:

> **D. Closed-loop simulation with a LiDAR sensor model**

This is exactly right.

The paper also explicitly states that no experiment runs on a physical robot.

The experiment remains one of the strongest parts:

- 665 s parked;
- worst-corner σ ≈ 0.989 m;
- 319° in-place turn;
- first publishable frame after 24.5 s;
- worst-corner σ drops to 0.048 m;
- ≈21× reduction;
- geometry moves only millimetres.

The final geometry is 5.996 m × 4.005 m against 6.000 m × 4.000 m.

---

# 11. Statistical wording

In the exploration ablation you write:

> “The two are indistinguishable.”

I recommend:

> **“We find no statistically detectable difference under this evaluation.”**

This is more defensible than “indistinguishable.”

---

# 12. “Independent rotation bands”

In §V-E you write:

> “across three independent rotation bands.”

If these bands come from the same 200 m trajectory, they are not independent datasets.

Use:

> **“across three distinct rotation-rate bands.”**

---

# 13. Figure 3 claim

The caption currently says:

> “the declared σ predicts the realised error.”

Given the Spearman correlation of only 0.125, this is slightly stronger than ideal.

Use:

> **“the declared σ is substantially more informative about realised error and supports the publication decision.”**

This is consistent with the main text, where you correctly say σ is useful for publish/withhold but weak for fine-grained ranking.

---

# 14. Real-depth validation

The real Matterport depth experiment is a useful intermediate validation layer.

Reported values:

- 588 wall segments;
- 120 rooms;
- median modal offset = 7.5 mm;
- scatter about fitted wall = 1.49 cm;
- p90 = 3.82 cm;
- fitted vs annotated wall median = 1.81 cm;
- p90 = 5.72 cm.

The simulated sensor uses about 2.0 cm scatter, so the model is slightly conservative.

I would change:

> “the simulated sensor used in §V is honest”

to:

> **“the simulated sensor model is consistent with, and slightly more conservative than, the real-depth measurements.”**

---

# 15. The 5× argument is now safer

The paper correctly states that the 5× difference is between:

- carefully annotated building;
- hand-traced layout.

It explicitly says these are **two settings rather than a sample of environments**.

Keep the conclusion:

> “a constant calibrated in one setting need not remain calibrated in another”

This is appropriately cautious.

---

# 16. Limitations

The limitations section is excellent and should mostly remain unchanged.

The MP3D-FPE negative result is handled responsibly:

- median 57% of returns fall outside the scored room;
- 0/401 rooms fall below 10%;
- even the union of rooms leaves 50% outside;
- corridors and stairwells are not annotated.

Therefore the 0.14 IoU is interpreted as an observation/annotation mismatch rather than a meaningful estimator score.

This is exactly how a strong paper should discuss an unsuitable benchmark.

---

# 17. Diagnostic contribution

The paper clearly shows three precision defects.

The strongest quantitative example is:

> NIS 0.47 → 1.03

after removing the duplicated systematic term and mapping the surviving term correctly.

This gives the uncertainty formulation a practical role beyond simply producing a confidence number.

Keep this experiment.

---

# 18. Physical robot experiment

Given that you cannot do one, I would **not** try to compensate by adding random experiments.

The evidence chain is already coherent:

1. 594 real building geometries;
2. calibration against realised corner errors;
3. real Matterport depth validation;
4. closed-loop LiDAR simulation;
5. heuristic exploration ablation;
6. diagnostic failure analysis.

The correct defence is the explicit separation of what each evaluation level establishes. You now do that well.

---

# 19. Reference verification

I externally checked the bibliography.

### Overall result

**All 36 references correspond to real publications or real preprints.**

I did **not** find a fabricated/nonexistent reference.

There are nevertheless a few bibliographic corrections I recommend.

## Reference [24]

Current title:

> “Exploration of indoor environments predicting the layout of partially observed rooms”

Correct title:

> **“Exploration of Indoor Environments through Predicting the Layout of Partially Observed Rooms”**

The paper is real and was published at AAMAS 2021, pp. 836–843.

**Fix the title by adding “through”.**

## Reference [25]

The paper is real, but the citation is incomplete.

Add:

> **IEEE Robotics and Automation Letters, vol. 9, no. 8, pp. 6832–6839, 2024.**

## Reference [28]

The cited work is real, but it has subsequently appeared as a published **Autonomous Robots** article in 2026.

I recommend citing the published version rather than only arXiv:2406.13482.

## Reference [33]

This is the clearest bibliographic error.

Current:

> pp. 395–402

Correct:

> **pp. 395–400**

## Reference [36]

The current citation is acceptable. For maximum completeness you can add:

> **pp. 667–676, DOI: 10.1109/3DV.2017.00081**

## Reference [31]

The citation is real and correct. For completeness, it appeared in IFAC-PapersOnLine 55(14), 2022, pp. 126–132.

---

# 20. Recent references are real

I specifically checked the newer-looking references as well.

The following all correspond to real works:

- [5] **FRI-Net** — ECCV 2024.
- [6] **SLIBO-Net** — NeurIPS 2023.
- [7] **PolyRoom** — ECCV 2024.
- [8] **CAGE** — NeurIPS 2025.
- [9] **PolyDiffuse** — NeurIPS 2023.
- [10] **Raster2Seq** — SIGGRAPH 2026.
- [11] **FloorSAM** — arXiv 2025.
- [21] **HouseLayout3D** — NeurIPS 2025.
- [27] **CogniPlan** — CoRL 2025.
- [28] **Estimating map completeness in robot exploration** — real work, now published in Autonomous Robots.
- [23] **SceneScript** — ECCV 2024.

So there is **no hidden fake-reference problem** in this bibliography.

---

# 21. Style: slightly too rhetorical in places

The paper is readable partly because it has personality, but I would reduce the rhetoric by roughly 10%.

Examples:

> “the claim of the paper”

> “The estimate is right before it is confident, and says so.”

> “precision is load-bearing”

> “Is the reported uncertainty honest?”

> “The shape was right before the confidence was.”

These are memorable, but some ICRA reviewers may prefer more conventional scientific language.

I would not sterilise the paper completely. Just soften the most provocative phrases.

For example:

### Current

> “The shape was right before the confidence was.”

### More conventional

> **“The geometry was already accurate before the uncertainty became small.”**

---

# 22. Important technical terminology

The paper correctly states that the first three calibration columns represent:

> P(||e||₂ ≤ kσ)

and explicitly says this is **not a probabilistic containment bound**.

Keep this.

Calling σ a standard deviation or empirical uncertainty scale is defensible.

Calling it a guaranteed bound would not be.

---

# 23. Current closed-loop result

| Condition | Long edge | Short edge | Worst corner σ | Worst edge σd |
|---|---:|---:|---:|---:|
| Parked, frame 0 | 5.994 m | 4.015 m | 0.988 m | 0.500 m |
| Parked, 665 s | 5.995 m | 4.012 m | 0.989 m | 0.500 m |
| After turn | 5.994 m | 4.019 m | 0.048 m | 0.019 m |

This is a strong result.

The important limitation is correctly acknowledged: the map frame is gauge-free, and there is no ground-truth robot pose, so absolute heading error is not reported.

---

# 24. Final priority list

## 🔴 Definitely fix

1. **Contribution 3:** replace “across environments” with “between the two measured settings” and remove/soften the “general property” claim.
2. **Figure 3 caption:** change “594 annotated floor plans” to **“4,344 matched corners from the 586 annotated floor plans that produced a closed polygon.”**
3. **Reference [24]:** add “through” to the title.
4. **Reference [28]:** update to the published 2026 Autonomous Robots version.
5. **Reference [33]:** change pages **395–402 → 395–400**.
6. **“6 cm bound” → “6 cm gate/tolerance.”**
7. **“three independent rotation bands” → “three distinct rotation-rate bands.”**

## 🟠 Strongly recommended

8. “The two are indistinguishable” → **“We find no statistically detectable difference under this evaluation.”**
9. Soften “the declared σ predicts the realised error.”
10. “the simulated sensor is honest” → **“the simulated sensor model is consistent with, and slightly more conservative than, the real-depth measurements.”**
11. Add volume/pages to [25].
12. Add pages/DOI to [36] if space permits.
13. Reduce the most rhetorical wording by ~10%.

## 🟢 Optional

14. Add complete bibliographic details to references that currently only give venue/year.
15. Do not add another major experiment unless it is essentially free.

---

# 25. What I would NOT change

I would **not**:

- add experiments just for the sake of adding experiments;
- pretend the Webots experiment is a physical-robot experiment;
- remove the honest null-result exploration ablation;
- remove the diagnostic defect experiment;
- change the Spearman statement back to “zero”;
- claim that σ is a guaranteed probabilistic bound;
- add a large amount of new related work;
- substantially restructure the paper;
- remove the limitations discussion.

The paper now has a coherent story.

---

# Bottom line

**This is now a submission-quality paper.**

The biggest improvement is that the paper now anticipates reviewer objections rather than merely responding to them:

- **Why 594 → 586?** Answered.
- **Why 6 cm?** Answered.
- **Is a global scalar enough?** Baseline + cross-validation answered.
- **Is the uncertainty calibrated?** Coverage + confidence intervals + error separation answered.
- **What about real sensors?** Real-depth validation answered.
- **What about a real robot?** Explicitly acknowledged as a limitation.
- **Does active exploration help?** Null result honestly reported.
- **Can uncertainty diagnose anything?** Three defects demonstrated.

The remaining issues are primarily **precision, consistency, and bibliography**, not scientific substance.

## Final estimated score

**~8/10**

If you make the seven definite fixes above, I would be comfortable submitting this version rather than attempting a last-minute redesign.

The strongest defence against a reviewer who wants a physical robot is not another claim. It is the disciplined separation of exactly what each evaluation level demonstrates—and your current version does that very well.
