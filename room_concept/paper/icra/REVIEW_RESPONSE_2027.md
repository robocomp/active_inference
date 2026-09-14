# Response to `ICRA_2027_paper_final_review.md` — 2026-09-14

All 6 must-fix, all 6 strongly-recommended and all 3 optional items are done. Two of them turned out
to be symptoms of something worse than the review described; those are marked ★.

## Must fix

| # | item | what was done |
|---|---|---|
| 1 | Spearman "zero by construction" | Now: "Being constant, it provides no ranking of corners by realised error at all: a rank correlation is not merely zero but **undefined**, since the predictor has no rank variation." The review is right — a constant has no ranks to correlate. |
| 2 | 18 vs 22 corners | ★ Settled from the corpus, not by picking one: `corpus594.txt` has 6..18 corners (330 rooms with 6, 2 with 18). **§V-A was right, the Fig. 2 caption was wrong.** See ★ below — the caption had a second, worse error. |
| 3 | 594 → 586 | Table II note now says the 8 rooms that produced no closed polygon have no corners to match and are excluded, leaving 586 of 594. |
| 4 | "Why no scalar precision can be correct" | → **"Why a single global precision is insufficient"**. The evidence shows insufficiency, not impossibility. |
| 5 | "Closed loop on a real sensor" | → **"Closed-loop simulation with a LiDAR sensor model"**. |
| 6 | Justify the 6 cm gate | Stated as the downstream consumer's clearance requirement, fixed before the evaluation. **Sensitivity added** (optional 13): published median error is flat at 1.99–2.04 cm for gates of 4, 6, 8, 10 cm while the withheld median rises 3.09 → 5.33 cm, so the separation is a property of the ordering, not of the cut. Below 2 cm the gate is vacuous the other way: the smallest declared sigma in the corpus is 0.0203 m, so a 2 cm gate publishes nothing. |

## Strongly recommended

| # | item | what was done |
|---|---|---|
| 7 | scalar fitted post hoc | Said explicitly in the Table II note: "a deliberately favourable comparison ... information our own estimate never gets." |
| 8 | three evaluation levels | New opening paragraph of §V naming what each level establishes, and stating plainly that no experiment runs on a physical robot. |
| 9 | VFE vs EFE | Abstract now separates them: parameters and structure are inferred by minimising a variational free energy; viewpoint selection minimises the corresponding **expected** free energy, whose epistemic term reduces to expected information gain under the Gaussian model. |
| 10 | "5x between environments" | → "between the two settings we can measure it in", and the body now says these are two settings rather than a sample of environments, so the factor **bounds the variation we can demonstrate rather than estimating its spread**. |
| 11 | "property of the map, not of the sensor" | → "not explained by independent sensor noise alone", with the reason stated: it survives at 1.8 cm in data whose per-return scatter is 1.5 cm, and scatter that averages down cannot produce an offset that does not. |
| 12 | "wrong in the next" | → "a constant calibrated in one setting need not remain calibrated in another" (abstract and body). |

## Optional — all three done

- **13. Gate sensitivity**: computed, in the text (see item 6).
- **14. Cross-validated scalar**: 5-fold by *room*, sigma_c = 0.0318 +- 0.0002 m, held-out 1-sigma
  coverage **65.4%** against the 65.3% it was fitted to reach. This strengthens the argument rather
  than weakening it, and the paper now says so: the post-hoc fit is **not** what defeats the scalar.
  It is perfectly well calibrated on average and generalises as such; what it cannot do is vary.
- **15. Bibliography**: done in the previous pass — all 36 entries checked against the publisher
  record, nine corrected. See `REFS_VERIFIED.md`.

## ★ Two things the review did not catch, found while checking its items

**★ The Fig. 2 caption's central claim was false.** It said the panels were "sampled evenly across the
corpus, so the population's *worst* room is at the left of the top row and the sample is not a
selection of successes". The mosaic was in fact built from `mosaic_cells_120.html`, a **120-room**
render. The worst of those 120 scored IoU 0.821, while three rooms in the full corpus are worse:
**0.654, 0.754, 0.756**. A figure whose entire selling point is that it does not flatter the method
was quietly excluding the three rooms that flatter it least.

Fixed at the source rather than by softening the caption: `mosaic_cells_594.tsv` is extracted from the
594-room run and carries every room that produced a closed polygon, `make_mosaic.py` reads it, and the
first cell is now the true worst room at **IoU 0.654**. The claim is true by construction.

**★ The corner range was wrong in the caption, not the text**, and the review could not tell which.
The corpus decides it: 6..18.

## Not done, deliberately

The review's own "what I would NOT change" list is followed: no new experiments, no restructuring, no
last-minute related work, and the null exploration ablation and the diagnostic experiment both stay.

Paper is 8 pages with 325 pt spare on page 8 after the 37 pt camera-ready reserve, no overfull boxes,
no Type 3 fonts, clean BibTeX.
