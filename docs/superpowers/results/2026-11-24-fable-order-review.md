# Review of the shared-engine work order (feat/shared-engine, 2026-11-24)

Read-only review. Nothing in the repo was edited. Paths are relative to
`/home/pbustos/robocomp/components/active_inference.worktrees/shared-engine` unless absolute.
What I read: spec, plan, the full ledger, the baseline summary, step 0, engine1 live and diagnosis,
`persistent_engine.h` (first 520 lines in full, rest by outline), `box_engine_model.h`, the relevant parts of
`countable_belief.cpp` and `countable_fitter.cpp`, `beam_ray.h` (part and accumulate interfaces),
`countable_planner.h`, `standpoint_free.h` (header), `epistemic_step.h`, the chair and table state structs, the
manifests. I did not build or run anything. Where a claim is an inference and not something I read, it says so.

## 0. The five points that matter most

1. **The ruled volatility fix does not cure the bed on the logged numbers.** The two steps that drove ω were
   |Δh| = 0.384 and 0.317 m. The footprint prior caps Σ_pred,hh at 0.20² = 0.04. So
   `max(0, δθ² − (Σ_pred − Σ_post))` is still at least 0.107 and 0.060 m², thousands of times q0 = 2.5e-5. The
   rectifier is also a clamp with a positive bias on a static object (E[(z² − 1)⁺] ≈ 0.48 of the correction
   variance). The principled statistic exists and needs no max (section 2.3).
2. **The wall latent fixes the toilet's σ collapse but not its 19 cm bias**, and NEES is driven by the bias. The
   bias needs the stand-off to be a class distribution, from class knowledge, not from this Webots world.
3. **Fridge and hood probably do not need a fitter port at all.** The countable binary is already a
   manifest-driven clone of the refrigerator (config key sets differ by about 20 keys; `hood.concept.toml`
   already declares `support = "hangs"`). Running them as countable classes gives the engine switch at launch
   level and leaves the legacy agents untouched. This removes most of 10b's urgency.
4. **Test (a) as written is close to a coin flip for a perfectly calibrated filter** with one visit per
   instance, and **test (b) has no data for bed and toilet** under the present offer publication. Both are
   harness problems to fix now, not estimator problems.
5. **The owner's "parked test first" decision was skipped.** The order still has T13 after every port. It is a
   minutes-long test and it would have shown the toilet's parked collapse directly. Run it right after F1.

## 1. The order, and the common architecture

### 1.1 What the code is today

- `PersistentEngine<M>` is generic in N. The parts that are not: `beam_ray.h` (`BeamPart::J` is 7×6,
  `accumulate_beam_rays` has `static_assert(N == 6)`), `viewpoint.h::chain_B` (6×3), `foreign_box.h` (6).
  Because the engine instantiates `beam_.accumulate<N>`, a model with N ≠ 6 does not compile even if it stages
  no rays.
- The class geometry lives in three places that do not share a source:
  - camera SDF: `CountableBelief::sdf_box` plus a hard-coded `sdf_headboard`;
  - beam parts: `CountableBelief::beam_part(s, k)`, box plus headboard;
  - existence carve: `[[shape.part]]` from the manifest (`common/concept_manifest/shape_parts.h`), fractions of
    the fitted (w, h, H).
- `run_inference` in `countable_fitter.cpp` is one 700-line function (lines 1098–1802) that mixes the birth
  seed, the legacy per-frame path, the gates, the persistent staging and the logs. Only the belief has a
  golden test. The fitter has none.
- The escape hatches through which patch terms survive a port are `accumulate_camera_extra` (top anchor,
  wall boost, footprint / depth-unobserved shares) and `accumulate_extra` (wall, room axis, short height,
  no-cross). `box_engine_model.h` forwards both for fridge and hood (`persistent_camera_extra`,
  `persistent_theta_only`).

### 1.2 Target architecture: one binary, one fitter, one engine, class = manifest

The owner's phrase is "all countable objects". The cleanest reading is that every one of the seven is a class
of the countable binary, as bed, nightstand and toilet already are.

**Engine (`common/ai_belief/`, pure, no graph):**
- `persistent_engine.h`: joint (θ, b), latent predict, GN, joint covariance, `predicted_gain`, delivered nats.
  Three additions, all class-independent:
  - process-noise inference (section 2.3), because the engine owns Σ_pred and Σ_post;
  - a context-latent block for the wall (section 2.2), as a third `pu::Channel` whose selector is the wall
    normal on (cx, cy);
  - a once-per-anchor ledger for prior factors, replacing `wall_prior_added_`, `axis_prior_added_` and the
    class-prior-once flag now held in `CountableBelief`.
- `parts_geometry.h` (new): boxes whose centre and half-extents are affine in (w, h, H), placed by
  (cx, cy, yaw). From one `[[shape.part]]` table it returns the camera SDF and its Jacobian, the `BeamPart`
  list, and the carve boxes. The beam already computes the 7-vector derivative per part, so the SDF Jacobian
  is the same chain rule. One part reproduces `sdf_box` bit for bit; the bed's headboard becomes part 2. The
  hood needs one extension: a bound may be `a_f·dim + a_m` (z0 = H − extent).
- `beam_ray.h`, `viewpoint.h`, `foreign_box.h` stay N = 6. With every class on the 6-DOF box frame, the
  `static_assert` states a property and is no longer a blocker.

**Fitter (`common/persistent_fit/`, graph and sensors):** the persistent path of `countable_fitter.cpp`, split
into stages that can be tested without a graph. Suggested files and entry points:
- `pose_source.h`: `read_odometry_and_pose`, `pose_at_stamp`, `integrate_odometry`, chain correction.
- `latent_staging.h`: `stage_persistent` (ρ from paths, marginals, fresh shares). It should take the grazed
  horizontal edge from the parts, not `kHeight`, which is wrong for a hanging hood (T10a minor 1).
- `ray_staging.h`: `build_membership`, the select box, `feed_beam`, the room and floor foreign density
  (today `countable_beam_staging.h`, `countable_mask_membership.cpp`, `countable_lidar_range_channel.*`).
- `neighbours.h`: `refresh_neighbours`, `neighbours_of`.
- `room_context.h`: `nearest_wall`, `room_axis`, `interior_support`, σ_wall from the room's published
  covariance.
- `birth_seed.h`: the class-prior seed and its position covariance (section 2.4).
- `offer.h`: `offer_lidar`, the offer context, the cache keyed on the update serial, the promise and
  delivery log.
- `persistent_fitter.h`: the cycle (observe, integrate, stage, update or predict, consume paths, log).
- **Only the persistent path moves.** The flag-off terms (footprint moment, obliquity, coverage, range noise)
  stay in the legacy file. Addendum 1 of the baseline counts a library once per class that links it, so a
  patch constant moved into common counts against all seven classes at T14.

**Per class:**
- `common/concept_manifest/<class>.concept.toml`: state names, `[[shape.part]]`, size prior mean and std,
  attachment with a stand-off mean and std, alignment, layer spread Δ, symmetry, detector labels.
- `<class>_concept/etc/config.toml`: sensor rows and switches only.
- A small class hook only where a manifest cannot say it: the chair's front mode, the table's round variant,
  the fridge's door cue.
- A closed list of allowed prior terms: class prior (once), attachment (once, through the wall latent), room
  axis (once), room containment (no-cross, world knowledge), floor support. Anything else passed through
  `accumulate_extra` or `accumulate_camera_extra` must be argued in the ledger, or the recount will not move.

### 1.3 Chair and table: N = 6 box-part models, engine left generic

- Keep the engine template generic in N. It costs nothing and the toy model uses it.
- Make the classes N = 6 box-part models. Reasons:
  - one state convention for publication, `ForeignBox`, `chain_B`, the beam and the planner's `Target`;
  - the table's seventh state is the depth tilt, read only when `FootprintResidual` is on, which the baseline
    says is off everywhere, so the table is already a 6-DOF top plus legs;
  - the chair is "pose only, size is the fixed template". As a 6-DOF model its size gets the template as the
    class mean with a class std. That is more honest than a zero-variance size, and the engine cannot hold a
    zero-variance θ component anyway (`sanitize` re-seeds when P_ii is not positive).
- This pulls a small piece of part 2 forward (the parts vocabulary for chair and table). The spec defers
  part 2 to after grading, so this is an owner decision. The alternative (keep N = 3 and N = 7, generalise
  `BeamPart::J` to 7×N, adapt two more fitters) is work that part 2 would then replace.
- I did not verify how much of `chair_concept` (support bank, contour, ghost, discrete mode) the C3 tours
  depend on. That is the risk of the N = 6 route for the chair.

### 1.4 Recommended order

1. **R0, replay tap (half a day).** Behind a config flag, write each update's `Frame` + `FrameLatents` + birth
   seed at `CountableBelief::update_persistent_cycle`. Add a small replay binary under
   `countable_concept/tests`. The diagnosis could not settle its item 1 for lack of exactly this. Record one
   tour on the current build. Use `std::from_chars` and `imbue(classic)`, and `setlocale(LC_ALL, "")` in the
   harness, per CLAUDE.md.
2. **F1 before 10b.** The volatility and wall fixes belong in the engine and the offer cache in
   `common/epistemic_step` and `common/nbv`, so they land in common either way. Only the birth seed is in the
   fitter. A behaviour-preserving move cannot be validated by a live tour against a baseline whose bed σ
   blows up depending on the birth view (engine1 against step0 differed mainly by that).
3. **T13 parked test now**, on a nightstand and the toilet, then three tours (not one).
4. **10c′ and 10d′: fridge and hood as countable classes** (manifest plus folder). Spend the first half day
   on a spike: does the published node (type, attributes, affordance name) match what kitchen, cabinet, LTOM
   and the language agent read? If not, fall back to the ruled 10c and 10d. The 10a adapter
   (`box_engine_model.h`) then stays as a tested legacy-side artefact; that cost is sunk.
5. **Go / no-go on 12-02.**
6. **P1: `parts_geometry.h`** driving SDF, beam and carve from `[[shape.part]]`.
7. **T11 chair, then T12 table**, as manifests over P1.
8. **T14 recount, grading tours.**
9. **10b as ruled** (the full carve into `common/persistent_fit`) becomes a refactor for testability. Do the
   pure-function extraction opportunistically; do the rest after grading. With one fitter consuming it, no
   library API is needed for the date.

## 2. The rulings and fixes, one by one

Verdicts: **sound**, **sound with a condition**, **patch in disguise**.

### 2.1 p_stand ∈ {0, 1} mirroring the controller: sound with conditions
- It is a feasibility fact about an action, not an estimation threshold, and it mirrors a deterministic
  planner. It is honest as long as it is not called a probability. The spec's G6 text ("continuous, with no
  threshold") is now false and should be amended.
- The weak point is duplication, not the switch. Margin 0 against the controller's
  `footprint_safety_margin_m`, no passage polygon, a different lattice. engine1 already shows two refusals of
  a standpoint 9 cm from one the agent had not blocked.
- Clean form: one oracle. Move `controller/src/residual_risk.h` plus `pose_free` / `can_turn_here` into
  `common/` (the include from `common/nbv/standpoint_free.h` into `controller/` is a dependency inversion),
  link it from both sides, and publish the margin as a robot property (there is a `common/robot_capability`).
- Log `p_stand` per offer row. The engine1 analysis could not split zero-gain causes without it.

### 2.2 Wall information once, through a ρ = 1 wall latent: sound with two conditions
- Principle: the wall's position error is one persistent source, which is what the engine's latents are for.
  Moving the wall by w along its normal n equals moving the object by −w·n, so it fits as a third channel
  with selector column (−n_x, −n_y, 0, 0, 0, 0) and marginal σ_wall² from the room's published covariance.
- Condition 1: **every** wall term must read the same latent: flush, parallel, no-cross, the wall component
  of the point mixture and the wall in the beam's foreign density. One latent per wall anchor, a new source
  when the nearest wall changes. If only no-cross is routed through it, it is a patch on one term.
- Condition 2: a quadratic hinge re-entered every update still asserts equality once active. With the latent
  it saturates at σ_wall instead of millimetres, which is acceptable, but a back that is "past the wall" by
  2 mm of noise is then pinned to σ_wall = 2 cm. The exact one-sided form is the probit factor
  −log Φ(gap / σ_wall), entered once and re-linearised. Use it if the hinge still over-tightens in the parked
  test.
- **It does not fix the toilet's bias.** The back is forced to the wall and the truth is 17 cm off it. With
  σ_back ≈ 2 cm the NEES stays in the tens. The model fix is `[prior.attachment]` with a stand-off mean and
  std (class knowledge: plumbing and cistern clearance for a toilet, ventilation gap for a fridge), entered
  once as gap − w ~ N(standoff, σ²).
- No fitted bound: the numbers must come from cited typical installations. `reach_m = 0.25` in
  `toilet.concept.toml` is already justified by "0.18 m off it in the world", which is a world fit. If a
  class-honest stand-off does not cover this world's 18 cm, test (a) fails for the toilet and is reported so.
- The spec registered `WallNoCrossPrecision` / `MarginM` for removal under G2. Keep the term, but as room
  containment (world knowledge, σ from the room), and say in T14 that it moved and did not vanish.
- If time is short: Hessian once per anchor, gradient every update (the diagnosis's option a), recorded as an
  approximation.

### 2.3 Volatility on the excess innovation: patch in disguise as worded
- It contains a `max(0, ·)`, and on the logged steps it does not remove the blow-up (point 1 above).
- Principled form: the EM statistic for the process step w of a random walk. With Σ̄ the predicted and Σ⁺ the
  posterior θ-marginal, per DOF:

      E[w_i² | data] = q_i + (q_i / Σ̄_ii)² · (δθ_i² − (Σ̄_ii − Σ⁺_ii))

  - It is signed and always ≥ 0 (since q ≤ Σ̄), so there is no rectifier.
  - Its fixed point is "innovations are consistent", which is the definition of the right Q.
  - The weight (q/Σ̄)² is the share of the prior variance that motion accounts for. A 0.38 m correction of a
    0.2 m prior with q = 2.5e-5 contributes about 6e-8: a seed collapsing onto data reads as the prior being
    corrected, with no special case. A shove of a converged object (Σ̄ a few q) is learned.
- Keep the conjugate running mean and its hyperprior (n0, q0). The ω clamps (`volatility_omega_min`, 4.0)
  become unnecessary. `kLambda = 0.995` remains a memory constant; report it.
- Put it in the engine (`update()` has P̄ and the posterior), not in `CountableBelief::update_volatility`, so
  every class gets it.
- Before coding, run `rootcause/omega_replay.py` with both rules on the engine1 bed log. That is a ten-minute
  check of this section.
- Related and pre-existing: `predict` clamps Σ to the class prior (the 0.30 / 0.20 caps the user saw). A
  mean-reverting predict toward the class prior would bound it with no clamp. Not for part 1; count it.

### 2.4 Birth truncation cutoff replaced by a wider birth covariance: sound if the width is derived
- On the engine the class prior enters at the class mean on the first update; the seed is only the GN start
  plus `birth_pos_excess_`. So the gate at `countable_fitter.cpp:1139` protects nothing in the prior. Its only
  effect was to skip `birth_on_priors` and start GN 0.8 m away.
- Fix: always seed at the class-prior MAP given the cloud (class size, room-axis yaw, attachment). Truncation
  enters only the position covariance.
- The width must come from quantities already in the model: a clipped mask bounds the seen extent from below,
  so the centre along the clipped direction is uncertain over (class extent − seen extent), uniform, variance
  (·)²/12. A coefficient times `trunc_frac` would be a new patch constant.
- The same block also holds `(ext_major − ext_minor)/sum > 0.10` (yaw commit) and `std::max(0.10f, …)`. They
  are thresholds on the seed. On the engine they affect only the GN start; list them at T14.

### 2.5 Per-channel shape latents (Δ/√24 floor): a modelling choice that errs the unsafe way
- Not a patch, and owner-approved. But independence between s_cam and s_L is extra knowledge the manifest
  does not hold. What the manifest says is "a single box cannot say which layer it means". That ambiguity is
  about θ's referent and is common to both sensors, which gives one latent and a floor of Δ/√12.
- The independence assumption halves the variance exactly where the two errors share a sign. The bed is that
  case: the graded mattress (1.57 m) is the narrowest layer, so both sensors err outward. engine1's bed w was
  1.64 m.
- A cross-correlation parameter would be a new constant. The two constant-free choices are r = 1 (shared) and
  r = 0 (independent). Register both predictions before T13 and let the parked A/B decide.
- Second point: the misfit is not zero-mean when the graded layer is an extreme of the range. Writing the
  layer interval relative to the graded layer in the manifest ([a, b], not a symmetric Δ) gives the latent
  its mean with no new constant.

### 2.6 Neighbours with σ_size = 0: a known-false input, not a safe side
- It under-explains leak at the neighbour's edges, which is where the leak is. The self object then stays
  over-sized, the symptom G3 was meant to cure.
- Interim with no cortex change: the neighbour's layer-spread floor Δ/√12 from its own manifest in
  `common/concept_manifest/`. It is class knowledge compiled into every agent, not a file crossing agents at
  run time, but flag it against the "agents talk only through the graph" rule.
- Ask the owner now for the size-std attribute (`REGISTER_TYPE` plus a reinstall), and batch it with the
  reinstall that feat/ltom-permanence is waiting for.
- From the T6 minors, the ones that are model defects:
  - `explained_by_other` hard exclusion is still on beside the mixture: two mechanisms for one effect;
  - unnormalised ε·Σg_k doubles the foreign mass where neighbour faces coincide;
  - a same-object twin can explain the instance's own returns (watch item).

### 2.7 Constant metric sig2_cam: defensible interim, but a relabelled row
- The values are the old tuned floors (`AI2CommonModePosStd` 0.03, `SizeStd` 0.02, `YawStd` 0.03) renamed
  "sensor". They happen to agree with the one clean measurement (fridge: 0.6–2.2 cm mean absolute boundary
  offset, 4.5 px). Replace them with that measured pooled value and its provenance so the row is measured.
- Dropping the range term halved the toilet's σ_h (5.5 → 2.7 cm) and exposed the wall re-count. The grading
  must report NEES by range bin, or the far-view question the ruling left to the data is never asked.
- `MaskPixelErrorRate` stays owner-decided (toilet 0.001 against 0.009 measured). Fine, report both.

### 2.8 MotionCm kept: honest, and measurable without a new tour
- The config's own comment says the gains are an effective lag in seconds. Step 0 measured stamp alignment,
  not that. The step-0 dump has moving frames with ground-truth poses: regress the per-frame boundary offset
  in metres on camera speed, and the slope is τ_eff. That turns three patch rows into one measured row. Cut
  candidate if time is short.
- The larger item is the gate, not the gain: `gated = trunc > AI2TruncGateFrac or confirm_only(inst)` is
  still in force on the persistent path. 783 of the bed's 874 rows were gated. The fresh latent u_cam and
  the gate now handle the same motion twice, and the spec registered `AI2MotionConfirmOnly`, `AI2Still*` and
  the fixation gates for removal under G1. Decide per gate before T14: off on the persistent path, or
  reported as kept.

### 2.9 The other rulings and deferred minors
- **Sound:** T2's 0/0 → 0 limit in `rho_camera`; chain share kept in e and not added to sig2_cam; toilet
  Δ_H = 0; hood Δ from a cited canopy; the 5-argument `accumulate_extra` with energy; live steps run by the
  controller.
- **Offer cache on the update serial, skip in cooldown: sound with one condition.** Key the cache also on the
  stand-world version (p_stand is cheap, re-evaluate it every tick) and refresh at the planning rate, since
  the promise drifts with Q on predict-only ticks. A rate is a scheduling constant, not a threshold.
- **Must fix for test (b):**
  - T7 minor 7/11: define delivered as the net entropy change over the window. Summing per-update deliveries
    gave 136 negative rows of 228 for the bed.
  - Log `promised_nats` at the claim, not only per tick.
  - `nbv_gain_raw` (diagnosis item 4).
- **T9 "dominance factor 2×":** a threshold in the ground-truth tool chosen below the two observed ratios
  (2.5×, 3.4×). It is not in the estimator, but it is the fitted-bound pattern. An explicit layer list per
  class in the scorer is cleaner.
- **T3 restart scheme** (`SolverConfig::restart`): a countable test path inside the shared GN loop. Remove
  before final, as already noted.
- **`update_guarded` / `MaxStepM = 1.0`:** a rejection gate, pre-existing. Keep it and count it.
- **Cosmetic or safe to defer:** `ConstView`, the fd-2 capture, the golden's missing off-diagonals, the
  duplicated `lidar_ready`, T8's double field read, the T9 warning that always fires.

## 3. Smallest set for tests (a) and (b) by about 10 Dec

### Test (a), first-visit NEES/DOF in [0.5, 2]
1. EM process-noise statistic in the engine (2.3).
2. Birth always on the class priors, derived covariance (2.4).
3. Wall latent, plus stand-off mean and std in `[prior.attachment]` (2.2).
4. sig2_cam from the measured value; NEES reported by range bin (2.7).
5. Fridge and hood on the engine by the cheapest route (1.4 step 4). The paper's worst rows were toilet 44
   and hood 30, so the hood matters more than the table (2.3 on the old engine).
6. **Fix the test's statistic before the run.**
   - NEES/2 of a calibrated filter is Exp(1). One visit lands in [0.5, 2] with probability
     e^−0.5 − e^−2 = 0.47. Rows inside a visit are one draw (that is G1), so their median does not help.
   - Score all graded DOFs with the full covariance. Six DOFs give about 0.75 per visit. The ai2 log holds
     only diagonals today, so log the θ block.
   - State the statistic in advance: the mean over n independent first visits per class, with n ≥ 10.
   - The lower bound matters too. Nightstands sat at 0.07–0.61 before the shape latent widened σ.

### Test (b), promise tracks delivery
1. Offer cache, promise logged at the claim, net delivered (2.9).
2. **A scripted A / B / A visit per object**, which is T13 generalised. The countable planner publishes
   `useful ≡ 0` plus the verification floor of 5, so the controller never chooses by promise. Bed and toilet
   were never claimed in engine1 (0 windows), the nightstand had one window of nonzero length. The test needs
   pairs with spread: a new view and a return view per object.
3. Fridge and chair are the paper's rows for (b) (6.0 against 0.2, 3.19 against 1.9). The fridge comes with
   step 5 of (a). The chair needs T11.

### Cut order if time runs short
1. T12 table: stays on the old engine, reported so (the spec allows it).
2. Chair's Helios beam channel: chair camera-only on the engine.
3. τ_eff measurement (2.8), neighbour size std (2.6), p_stand oracle sharing (2.1): reported as kept.
4. The full 10b carve: after grading.
5. Do not cut: R0, the F1 four, T13, the test-(a) statistic, the A/B/A script.

## 4. Risks the current order misses

1. **No replay harness.** Every validation so far is one live tour, and the engine1-against-step0 difference
   was mostly the birth view. 10b as "behaviour-preserving, checked by golden + 29 tests + live tour" has no
   test at the fitter level.
2. **Test (c) is on track to under-deliver.** Removed so far: the range, periphery, lever and obliquity terms
   of the persistent sig2_cam. T6 kept all six of its rows. The confirm-only and truncation gates, the top
   anchor, the depth-unobserved shares and `explained_by_other` are all still live on the persistent path,
   and the ports forward the same terms. Decide the closed list of allowed prior terms (1.2) before 10c.
3. **Patch constants moved into common count seven times** at the recount (1.2).
4. **Legacy paths must stay selectable** (arXiv v1 rebuild, C3 non-regression). A carve of the intertwined
   flag-on / flag-off `run_inference` risks both. Class-as-manifest keeps the legacy agents byte-identical.
5. **Offer cost scales with instances.** About 5 ms per standpoint and four standpoints per instance was
   25 ms per tick for one bed. Several chairs make the cache a precondition for T11, and "rates before
   features" applies.
6. **The toilet may fail (a) honestly.** A class-knowledge stand-off may not cover a toilet 18 cm off the
   wall. Decide now that this is reported, not tuned.
7. **Wide σ lets a biased channel move the mean.** The bed's final 13.8 cm came from three LiDAR-only updates
   at 0.95 m that pulled cy by 14 cm while σ was inflated. After the volatility fix, check in replay that
   LiDAR-only updates at close range do not carry a layer bias larger than s_L allows.
8. **Phantom bed** from `Tracker.BirthFusion = true` with `BirthAdmitPlausibility = 0`. Out of scope, but a
   phantom publishes a 9-nat offer and pollutes test (b). Exclude unmatched instances in
   `promise_delivery.py`.
9. **Cortex reinstall dependency** (size std). Ask now.
10. **Held-out world for test (d)** has no owner or date in the order.
11. **Shared machine.** Two sessions build in worktrees and share Webots: `-j8`, check `pgrep -a cc1plus`.
