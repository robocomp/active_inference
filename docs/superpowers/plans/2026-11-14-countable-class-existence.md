# Countable existence = occupancy × class — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A countable object's existence becomes P(something occupies this box) × P(it is this class | its shape), so a
blackboard fitted as a 1.98 × 0.28 × 1.86 m "bed" is removed while LiDAR truthfully keeps seeing something there.

**Architecture:** Pure class-evidence math in a new header (`countable_class_evidence.h`). `CountableBelief` keeps a
parallel likelihood-only summary of the size variables (H, w, h) so the class prior is not counted twice. The worker
replaces the plausibility accumulator + tanh nudge (`apply_countable_filter`) with a class factor that has its own
`rc::exist::ExistenceBelief` and removal debounce; the birth score becomes the same evidence on the raw cloud's
robust height. Occupancy (`inst.existence`) is unchanged.

**Tech Stack:** C++23, Eigen, the countable_concept tests CMake project (`countable_concept/tests`), Webots for the
live check.

**Spec:** `docs/superpowers/specs/2026-11-14-countable-class-existence-design.md` (approved 11-14). Read §2 (what the
code does today) and §9 (settled decisions) before any task.

## Global Constraints

- Worktree `/home/pbustos/robocomp/components/active_inference.worktrees/phase0`, branch `feat/countable-phase0`;
  commit locally, never merge or push.
- CLAUDE.md: no thresholds/gates/clamps added to make a case pass (height enters only as a density); `and`/`or`/`not`;
  `std::from_chars` for any number read from a file; no `runtime_checked_*`; no new graph attribute.
- Builds: check `pgrep -a cc1plus` first; agent build =
  `cd countable_concept && systemd-run --user --scope -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G bash -c 'cmake -B build && make -C build -j8'`.
  Tests = `cd countable_concept/tests && cmake -B build && make -C build -j8 && ctest --test-dir build --output-on-failure`.
- Values (spec §9 / this plan): π_cls default **0.8** from manifest `[prior.class] p_label_true`; "other furniture"
  density = Gaussian about the class mean with std **1.0 m** per size variable (`Model.ClassAltStdM`, default 1.0);
  birth height noise σ_z **0.05 m** (`Model.BirthHeightNoiseM`); birth mask-bleed outlier fraction **0.05**
  (`Model.BirthHeightOutlierFrac`); robust cloud extent = z quantile **0.97 − 0.03**.
- Size variable order everywhere in the new code: **(H, w, h)** = state indices (2, 3, 4).
- Occupancy (`inst.existence`, its sensor debounce, the out-of-view HOLD) is not changed. `common/existence_belief`
  and `common/ai_belief/recursive_laplace.h` are not changed.
- Stop agents with SIGTERM only. Webots is shared: claim it from the `chat` and `room` sessions (SendMessage), send FREE after.

## Review Focus

1. A shape seen from one face only (depth unobserved): Λ must not be dragged negative by the unobserved extent — its
   likelihood information is ~0, so it drops out. Test in Task 2 (`one_face_view_depth_drops_out`).
2. A quarter-turn relabel (w↔h swapped by `apply_mode_rotation`) must not change Λ. Test in Task 1 (`swap_invariant`).
3. A real bed whose mask bleeds above the headboard (person, wall) at birth must still be born, only slower. Test in
   Task 1 (`birth_bleed_is_slowed_not_stopped`: extent 1.9 m gives p > 0.1 for the bed).
4. A sparse first glimpse (< 12 points) at birth gets the prior π_cls, not a penalty. Test in Task 3.
5. An instance not measured this cycle keeps its class L unchanged and its class debounce does not advance. Test in
   Task 4 (`unmeasured_cycle_holds_class`).

---

### Task 1: Class evidence math (pure, Qt-free)

**Files:**
- Create: `countable_concept/src/countable_class_evidence.h`
- Create: `countable_concept/tests/class_evidence_test.cpp`; Modify: `countable_concept/tests/CMakeLists.txt` (add target + `add_test`)

**Interfaces — Produces:**
```cpp
namespace rc::countable {
struct SizeGaussian { Eigen::Vector3f mean = Eigen::Vector3f::Zero();       // (H, w, h)
                      Eigen::Matrix3f info = Eigen::Matrix3f::Zero(); };     // information (Σ⁻¹); zero = no data
struct ClassShapePrior { float H_mean, H_std, w_mean, w_std, h_mean, h_std;
                         float H_top;            // tallest declared part (bed: headboard 1.25; else H_mean)
                         float p_label_true; };  // π_cls
// Λ (nats) = log N(θ_lik; μ_cls, Σ_cls+Σ_lik) − log N(θ_lik; μ_cls, σ_alt²I+Σ_lik), class side a 50/50 mixture
// over the w↔h labelling. Computed in the subspace the data inform (info eigenvalues > 0); zero info ⇒ 0.
float class_log_evidence(const SizeGaussian& lik, const ClassShapePrior& c, float alt_std);
// Birth: class side = (1−ε)·[Uniform(H_mean, H_top) ⊛ N(0,σ_z²)] + ε·N(H_mean, alt_std²); other = N(H_mean, alt_std²).
// For H_top ≤ H_mean the uniform degenerates to N(H_mean, H_std²+σ_z²).
float birth_height_log_evidence(float z_extent, const ClassShapePrior& c, float sigma_z, float eps, float alt_std);
float robust_z_extent(const std::vector<Eigen::Vector3f>& pts);      // q0.97 − q0.03 of z
inline float class_probability(float L) { return 1.0f / (1.0f + std::exp(-L)); }
inline float logit(float p) { return std::log(p / (1.0f - p)); }
}
```
Handling zero/partial information: eigen-decompose `info`; with Σ_lik = pseudo-inverse on the informed eigenvectors,
project both Gaussians onto those directions (U = informed eigenvectors) and evaluate there. This is the one algorithm
the signature does not determine.

- [ ] **Step 1: Write the failing tests** in `class_evidence_test.cpp` (plain `main`, `check(name, cond)` style as
  `furniture_prior_test.cpp`). Bed prior: H 0.65±0.10, w 1.75±0.20, h 2.10±0.20, H_top 1.25, π 0.8. Data info for a
  well-seen shape: diag(1/0.02²). `L = logit(π) + Λ`:
  - `bed6_rejected`: mean (1.86, 1.98, 0.28) → `class_probability(L) < 0.05`.
  - `bed7_accepted`: mean (0.65, 1.75, 2.10) → `> 0.9`.
  - `nightstand_accepted`: prior H .62±.10, w .51±.06, h .51±.06, data (0.62, 0.51, 0.51) → `> 0.9`.
  - `toilet_accepted`: prior H .90±.15, w .58±.08, h .73±.08, data (0.88, 0.34, 0.86) — objects_live4's fit — record
    the value it gives in the report, assert `> 0.5` (the fit itself was off; this guards against a too-tight class).
  - `no_data_is_zero`: info = 0 → `Λ == 0` exactly.
  - `wide_data_tends_to_zero`: info = diag(1/5²) → `|Λ| < 0.5`.
  - `swap_invariant`: bed7 with w,h swapped gives the same Λ within 1e-4.
  - `birth_mattress_ok`: bed, extent 0.65 → `class_probability(logit(π)+Λb) > 0.9`.
  - `birth_headboard_ok`: bed, extent 1.25 → `> 0.9`.
  - `birth_blackboard_low`: bed, extent 1.90 → `< 0.3`.
  - `birth_bleed_is_slowed_not_stopped`: same 1.90 → `> 0.1`.
  - `birth_nightstand_tall_low`: nightstand, extent 1.90 → lower than at 0.62.
  - `robust_extent`: 1000 points z uniform in [0,1] plus 5 at z = 3 → `robust_z_extent` within 0.02 of 0.94.
  Run under `std::setlocale(LC_ALL, "")` (CLAUDE.md).
- [ ] **Step 2: Run** the tests command → build fails (header missing).
- [ ] **Step 3: Implement** the header (inline functions). Use σ_z = 0.05, ε = 0.05, alt_std = 1.0 in the tests.
- [ ] **Step 4: Run** → all checks PASS; paste the printed Λ for bed6/bed7/toilet into the report.
- [ ] **Step 5: Commit** `countable_concept: class evidence — shape log-evidence ratio and birth height density`.

### Task 2: Likelihood-only size summary in CountableBelief

**Files:** Modify `countable_concept/src/countable_belief.h`, `countable_belief.cpp`; Create
`countable_concept/tests/size_likelihood_test.cpp` (+ CMake target linking `../src/countable_belief.cpp`, like
`furniture_prior_test`).

**Interfaces — Consumes:** `rc::countable::SizeGaussian` (Task 1). **Produces:**
`const rc::countable::SizeGaussian& CountableBelief::size_likelihood() const;` updated inside `CountableBelief::update`
on every frame that has points; reset with the belief.

**Algorithm (pinned):** after `ai::update<N>` returns in `CountableBelief::update` (`countable_belief.h:~335`):
1. At the converged state θ̂, run one accumulation of the data terms **with the class size prior blocks switched off**:
   blocks (2) footprint and (2a) height in `accumulate_extra` (`countable_belief.cpp:476+`, the `prior_footprint_std`
   and `prior_height_std` branches) — via a private `mutable bool size_prior_off_ = false;` they test. All other
   extras (wall, top anchor, depth observability) stay: they are data or structure, not the class size prior.
   Accumulate Id, bd exactly as `ai::update`'s lambda does (responsibilities, sdf_prim, sdf_jacobian, then
   accumulate_extra), then apply the same common-mode saturation (`ai::saturate<6>` with this frame's Scinv) —
   expose nothing new from recursive_laplace.h; call its existing functions.
2. Schur-marginalise the 6×6 (Ieff, beff) onto (H, w, h) = indices (2, 3, 4): J = I_ss − I_sp I_pp⁻¹ I_ps,
   g = b_s − I_sp I_pp⁻¹ b_p (use `ldlt`; pose block regularised by the belief's own `Sigma_` inverse pose block).
3. Frame summary: info J, mean θ̂_s + J⁺g (pseudo-inverse on eigenvalues > 1e-6·max).
4. Parallel recursion across frames, same forgetting as the belief: before adding, predict
   `info ← (info⁺ + Q_s)⁺` with Q_s the belief's process-noise diagonal for (2, 3, 4); then
   `η = info·mean + J·mean_frame`, `info += J`, `mean = info⁺η`.
   The labelling of w/h follows `apply_mode_rotation`: swap rows/cols w↔h of `size_lik_` there too.

- [ ] **Step 1: Failing tests** (`size_likelihood_test.cpp`), synthetic clouds built as in `CountableBelief::self_test`
  (four sides + top), bed params from `bed_concept/etc/config.toml` values (H 0.65±0.10, footprint 1.75±0.20, depth 2.10):
  - `panel_seen_as_panel`: a 1.98 × 0.28 × 1.86 panel cloud, 10 frames → `size_likelihood().mean` within 0.08 m of
    (1.86, 1.98, 0.28) in some w/h labelling, while `state()` (posterior) differs from it by > 0.08 m in H (prior pull).
  - `no_frames_zero_info`: fresh belief → `size_likelihood().info.norm() == 0`.
  - `one_face_view_depth_drops_out`: only the front face (local −y) of a 1.75 × 2.10 × 0.65 box → the information
    eigenvalue along h is < 1 % of the one along w.
  - `fit_unchanged`: with the summary in place, `state()` after 10 frames equals a build-time copy of the old
    behaviour — assert `size_prior_off_` is false after `update` returns and the posterior matches the value printed
    before the change (record it in Step 2).
- [ ] **Step 2: Run** → fails to compile; record the posterior of `fit_unchanged` from the current code first (run the
  test with only that case against HEAD) and hard-code it.
- [ ] **Step 3: Implement** as pinned above.
- [ ] **Step 4: Run** the whole tests project → all PASS (existing `headboard_sdf_test`, `furniture_prior_test`,
  `back_panel_test` included).
- [ ] **Step 5: Commit** `countable_concept: likelihood-only size summary (the class prior counted once)`.

### Task 3: Class prior from the manifest + birth score on the same scale

**Files:** Create `common/concept_manifest/class_prior.h` (Qt-free, same TOML style and `std::from_chars` parser as
`back_panel.h`); Modify `common/concept_manifest/{bed,nightstand,toilet}.concept.toml` (add the section),
`countable_concept/src/countable_config.{h,cpp}` (read it + the three `Model.*` keys of Global Constraints),
`countable_concept/src/countable_belief.{h,cpp}` (birth score), `countable_concept/src/specificworker_lifecycle.cpp`
(the two `candidate_plausibility` call sites, lines ~262 and ~458); Test: extend `back_panel_test.cpp`; extend
`class_evidence_test.cpp`.

**Interfaces — Produces:**
- `std::optional<float> rc::manifest::load_class_prior(const std::string& path, bool* opened = nullptr);` reads
  `[prior.class] p_label_true`.
- Manifest section, identical in the three files:
  ```toml
  [prior.class]
  kind = "label_reliability"
  p_label_true = 0.8
  why  = "P(a mask with this label is really this class); near YOLOE's median confidence on the real bed (.85, detcheck3). Owner may change it (11-14)"
  ```
- `CountableConfig` gains `float class_p_label_true` (absent in manifest ⇒ 0.8 and a startup line saying so),
  `float class_alt_std_m = 1.0f`, `float birth_height_noise_m = 0.05f`, `float birth_height_outlier_frac = 0.05f`.
- `rc::countable::ClassShapePrior CountableBeliefParams::class_shape_prior() const` (H_top = headboard height when
  > H mean, else H mean) — the params struct gains the four config values.
- `static float CountableBelief::candidate_plausibility(pts, p)` keeps its name and signature; its body becomes
  `pts.size() < 12 ? p_label_true : class_probability(logit(p_label_true) + birth_height_log_evidence(robust_z_extent(pts), …))`.
  Rewrite its comment to say so (the old "a bed is TALL ~1.5 m+" text is the refrigerator's and is wrong here).

- [ ] **Step 1: Failing tests:** `back_panel_test` gains `class_prior_bed == 0.8` for each of the three manifests and
  `nullopt` for a manifest without the section; `class_evidence_test` gains `candidate_sparse_is_prior` (8 points →
  exactly 0.8) and `candidate_blackboard_cloud` (a 1.9 m-tall panel cloud with bed params → < 0.3, a 0.65 m mattress
  cloud → > 0.9).
- [ ] **Step 2: Run** → fails.
- [ ] **Step 3: Implement.** Config keys get descriptions in the `getf` calls; add the three keys with their defaults
  to the three `etc/config.toml` files under `[Model]`, and add them to `config_migration_test`'s list of keys new
  since `476000a8` (it fails on unknown keys otherwise).
- [ ] **Step 4: Run** tests project → PASS; build the agent (Global Constraints command) → links.
- [ ] **Step 5: Commit** `countable_concept: class prior from the manifest; birth score is the class evidence of the cloud's height`.

### Task 4: The class factor in the worker (replaces the plausibility accumulator)

**Files:** Modify `countable_concept/src/countable_instance.h` (fields), `specificworker_lifecycle.cpp`
(`apply_countable_filter`, ~534-660), `countable_belief.{h,cpp}` (remove `countable_log_evidence_ratio`,
`singleton_existence_deltas` and their self-test cases once unused), `specificworker_dashboard.cpp:69`,
`specificworker.cpp:403` (phantom log), `countable_fitter.cpp:1228` (csv), `countable_config.{h,cpp}` (drop
`plaus_clamp`, `plaus_to_existence_gain`, the `plaus_*` identity scales read only by the removed code), the three
`etc/config.toml` (drop those keys; list them in `config_migration_test` as removed),
`CONCEPT_AGENT_LIFECYCLE.md` (conformance row for countable_concept: existence = occupancy × class);
Test: new `countable_concept/tests/class_factor_test.cpp`.

**Interfaces — Consumes:** `size_likelihood()` (Task 2), `class_log_evidence`, `class_shape_prior()` (Tasks 1, 3).
**Produces** (on `CountableInstance`):
```cpp
rc::exist::ExistenceBelief class_belief;      // L_cls, same ±Model.ExistenceLogOddsMax
rc::exist::RemovalDebounce class_debounce;    // replaces plaus_debounce
float class_lambda = 0.0f;                    // last Λ (nats), for logs
float p_exists() const { return existence.p_exists() * class_belief.p_exists(); }   // the product
```
Pure step function, so it is testable without the worker:
```cpp
// countable_class_evidence.h
struct ClassStep { float L_cls; bool argued_against; };
// singleton: inhibition·Σ_{j≠i} p_j·[Λ_j > Λ_i], p_j = the other instances' PRODUCT p.
std::vector<ClassStep> class_factor_step(const std::vector<float>& lambda, const std::vector<float>& p_product,
                                         float p_label_true, float inhibition);
// L_cls,i = logit(p_label_true) + Λ_i − singleton_i ;  argued_against = (Λ_i − singleton_i) < 0
```
Worker rule (`apply_countable_filter`, renamed `apply_class_factor`): for each fitted instance, **only on a measured
cycle** (the existing `matched_frames != plaus_seen_frames and not dbg_gated` test, field renamed
`class_seen_frames`): Λ = `class_log_evidence(inst.ai2_belief.size_likelihood(), cfg class prior, alt_std)`;
then `class_belief.set(L_cls)` and `decide_removal(class_belief, class_debounce, same RemovalPolicy as today,
(warm and argued_against) ? 1 : 0)`. Unmeasured cycle: nothing changes. Either this or the sensor debounce removes.
Log line (replaces `[<cls>-filter]`), measured cycles only, when `CountableFilterLog`:
`[{cls}-class] {name} occL={:.2f} clsL={:.2f} Λ={:+.2f} p={:.2f} streak={:.1f}`.
Consumers: dashboard `r.p_exists`, phantom-log `exist_logodds` (write logit of the product), fitter csv
(`p_exists` column = product; add `p_occ` and `p_cls` columns at the end). Every other `existence.` site keeps
occupancy; the report lists each of the 16 sites (`grep -n "existence\." countable_concept/src/*.cpp`) with
"occupancy" or "product" and why.

- [ ] **Step 1: Failing tests** (`class_factor_test.cpp`, links `../src/countable_belief.cpp`):
  - `bad_shape_argues`: Λ = −40 → `L_cls < logit(0.05)` and `argued_against`.
  - `good_shape_holds`: Λ = +5 → p > 0.99, not argued.
  - `singleton_weaker_inhibited`: Λ = {+5, +2}, products {0.95, 0.9}, inhibition 1.0 → second L lower by 0.95 than
    alone; first unchanged.
  - `unmeasured_cycle_holds_class`: drive the worker-side helper `apply_class_step(inst, measured=false, …)` (a free
    function in `countable_class_evidence.h` taking the three fields by reference) → `class_belief.logodds()` and
    `class_debounce.streak` unchanged.
  - `product`: occupancy p 0.98 × class p 0.02 → `p_exists()` ≈ 0.0196.
- [ ] **Step 2: Run** → fails.
- [ ] **Step 3: Implement** as above; delete the dead accumulator code and its self-test cases.
- [ ] **Step 4: Run** tests project → PASS; agent build links; `./build/... --startup-check` style run is not
  available — instead run `bash -n tools/concept_agents.sh` and confirm `grep -rn "plaus_evidence\|singleton_existence_deltas" countable_concept/src` is empty.
- [ ] **Step 5: Commit** `countable_concept: existence = occupancy × class; the class factor removes what its shape argues against`.

### Task 5: Live check (objects_live5)

**Files:** none in code. Results to `room_concept/estimation_scripts/crossing/objects_live5/` (same layout as objects_live4)
and a short `RESULTS.md` there.

- [ ] **Step 1:** Claim Webots from `chat` and `room` (SendMessage CLAIM); wait for no objection.
- [ ] **Step 2:** Rebuild `countable_concept` (Global Constraints command) — never start from a stale binary.
- [ ] **Step 3:** Run the objects_live4 tour command (read `objects_live4/` for the exact command and fleet) as
  objects_live5, with `CountableFilterLog = true` in the three configs for the run only (revert after, not committed).
- [ ] **Step 4:** Report against the spec §8 pass list, with numbers: every bed node in room_1 (birth time, measured
  seconds to removal, its Λ); bed, both nightstands, toilet positions vs GT (bed (−5.18, 12.39), nightstands
  (−6.36, 13.22) and (−4.00, 13.22), toilet (−0.87, 12.87) world) and birth times vs objects_live4; cycle rate per agent;
  the real bed's p_cls over the run (min, median).
- [ ] **Step 5:** Stop agents with SIGTERM; send FREE to `chat` and `room`; revert the log flags; commit `RESULTS.md`
  (`objects_live5: class factor live`).
