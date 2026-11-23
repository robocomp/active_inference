# Plan: existence information instead of the 5-nat floor, plus recalibrated object beliefs (2026-11-23, DRAFT for owner approval)

> **Scope narrowed 2026-11-23 (owner, relayed by auro): STEP 1 ONLY = §1 + the reduced runs below.**
> - §2 (covariance) and §3 (P(detect)) are DEFERRED until the owner has seen step 1.
> - Step 1 runs, on one build:
>   - 3 furnished C3a tours with per-class logs;
>   - 2 tours with `AffordanceRoomGainScale = 1.0`.
> - Step 1 grades:
>   - arbitration (contested choices, object vs room, margins, floor-free prices);
>   - promised vs delivered per class with Spearman;
>   - C3a for the record.
> - **Step 1 time** ≈ 6 h:
>   - code + tests ≈ 3 h;
>   - 8 agent builds at -j8 ≈ 45 min;
>   - 5 tours ≈ 65 min Webots;
>   - grading ≈ 1 h.

Work happens on a new worktree branched from main (`feat/existence-info`). Agents are run from it with `CONCEPT_WT`. Nothing is merged without the owner's word.

## 1. Replace the floor with the existence information of one view (≈ 3 h with tests)

**Today**
- `common/epistemic_step/epistemic_step.h:135-137` sets `gain = max(gain, verify_gain_floor)`.
- The floor is 5 nats, passed only while `wants_verification`, a thresholded absence accumulator.
- It applies to table, hood, refrigerator, cabinet and countable (nightstand/bed/toilet).
- Chair and bottle use no floor.

**Change**
- Move door_concept's existing `existence_mutual_info(p, p_hit, p_false)` (`door_concept/src/door_existence_info.h:12-26`) into `common/existence_belief/`.
- Add `I = h(p·d·P_det(v) + (1−p)·f) − p·h(d·P_det(v)) − (1−p)·h(f)` to the geometric term `P_det(v)·½ ln det(I+ΣΔI)` (`common/nbv/viewpoint_score.h:243`).
  - It is added for every instance on every cycle: no `wants_verification` gate and no floor.
  - p, d and f come from the instance's own `ExistenceBelief` and `SensorModel`: detection 0.85 (hood 0.55), clutter 0.05.
- Chair and bottle get the same term in their own planners.
- The door already does this, so all object classes would share one rule.
- I is at most ln 2 = 0.69 nats per view. Object prices fall to the size of the information a view actually adds.
- **Removed**:
  - the `verify_gain_floor` field;
  - `ExistenceVerifyGain` in 7 configs;
  - the `wants_verification` threshold, `ExistenceVerifySurprise = 20`, which was used only to switch the floor on.
- **Decision to confirm**: today the floor overrides the post-visit cooldown. With the floor gone, the cooldown applies to everything.
- **Tests**:
  - an existence-info unit test: I = 0 at p ∈ {0, 1}, maximal near p = ½, never above ln 2;
  - a regression showing the floor no longer appears in a priced offer;
  - `affordance_protocol_test` still passes.
  - These go in `common/run_tests.sh`.

## 2. Uncertainty that shrinks with VIEWS, not frames (≈ 5–6 h; the riskiest part)

**Today**
- `rc::ai::update` (`common/ai_belief/recursive_laplace.h:155`) saturates information per frame (Woodbury common mode, `Σc`).
- Frames from one standpoint are treated as independent, so σ falls as 1/√frames while the error does not.
- This matches the first-visit NEES/DOF: nightstand 5.7, hood 30, toilet 44, table 2.3.

**Change (model level)**: a per-viewpoint persistent error.
- The common-mode error is shared by consecutive frames from nearby viewpoints, with correlation ρ = exp(−d²/2ℓ²). d is a viewpoint distance in bearing and range from the object, a continuous kernel, not clusters.
- The information a frame adds is discounted by what correlated earlier frames already credited. Equivalently, the Woodbury saturation runs over the accumulated information of the current viewpoint, not one frame.
- One shared implementation in `recursive_laplace.h`, used by every agent (table, hood, refrigerator, countable, chair, bottle, cabinet).
- ℓ (bearing and range) is a physical model constant, set once for all classes, not per class.

**Fallback**, only if the model-level term cannot reach NEES/DOF ≈ 0.7 on development data:
- a per-class σ inflation fitted on development tours, stated as such in the paper.

**Data split**
- Development: the existing logs c3a_1, c3a_2, c3a_3, c3a_cab1, c3a_w2_1..3, lam_*.
- Held-out test: the new tours in §4, never used for fitting.

**Also**: the refrigerator (under-confident after the reference-point fix) and the room position (NEES 0.08) are under-confident. The same NEES is reported for them, but changing the room estimator is out of scope.

## 3. P(detect) calibration against logged detections (≈ 2 h)

**Today**
- P(detect | view) is the analytic fill envelope (`common/detectability/detectability.h`; defaults "a prior, not a measurement").
- Its outcomes are logged by `common/detect_probe` (`etc/detect_probe.csv` exists for bed, refrigerator, hood, nightstand, toilet, table).
- A fitter exists: `common/detectability/tools/fit_envelope.cpp`. Its notes say the hood and fridge fits were degenerate.

**Change**
- A reliability check of predicted P(detect) against outcomes, in bins, per class.
- Refit the envelope on development logs where the fit is not degenerate.
- Where the fill covariate does not explain the outcomes (hood, fridge), say so and keep the prior.
- **Test**: the 9 false removals of unmoved objects in the c3a tours are replayed or re-counted on the new tours.

## 4. Re-runs and grading (≈ 2.5 h Webots + 1 h grading)
- **C3a furnished tours**: n = 3, one world (the owner's current three-room world) and one build (the worktree). Cabinet out, as asked. Per-class ai2 logs in every run (harvest fixed).
- **Room scale 1.0**: n = 2, the same tours with `AffordanceRoomGainScale = 1.0`.
- **Grades**, as before:
  - C3a: held / phantoms / precision / error;
  - NEES per class, first visit and final belief;
  - arbitration counts;
  - promised vs delivered, now with per-class logs;
  - existence removals with ground truth.
- **Builds**: 8 agent binaries include `epistemic_step.h`, and `recursive_laplace.h` reaches most agents. -j8, about 45 min in total.

## Time
- About 1.5 working days: code and tests ≈ 10–11 h, builds ≈ 1 h, runs and grading ≈ 3.5 h.
- §2 dominates and carries the risk. If the model-level term fails, the fallback adds ≈ 1 h.

## What this invalidates (once the new results exist)
- **C3a grades** (old and new world): the prices change, so births, removals and order may change.
- **The λ sweep**: prices are on a new scale, so λ = 0.1–0.4 nats/m means something different. It should be redone if the paper keeps it.
- **The arbitration table and the T1/T3 review numbers** (floor share, promised vs delivered).
- **The object NEES rows.**
- **Item 13** (moved chair): the ON arm used the chair's own offer. The chair has no floor, so its price only gains the existence term. The mechanism finding (a recall does not restore a belief) stands.
- **Unaffected**:
  - room and door NEES;
  - the explorer rerun;
  - the C4 gyro control;
  - item 14's mechanism.
