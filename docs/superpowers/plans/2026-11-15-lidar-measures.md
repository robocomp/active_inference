# Camera attributes, LiDAR measures — Implementation Plan (rev 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every cycle with a fresh ZED mask assigned to a countable instance, the Helios returns update the belief — each return weighted by its posterior ownership (camera m as prior, room and published neighbour boxes as the alternative) — while the class prior is counted once and every channel's shared error persists across frames, so Σ is honest and H follows the LiDAR top.

**Architecture:** Four layers. (1) Shared, pure: per-ray weights + weighted selection (DONE, Task 1); the engine's LiDAR-only frame hook (DONE, Task 2); a new persistent-latent update `common/ai_belief/persistent_update.h` (joint θ + shared-error latents, inverse-free, double); an ownership-mixture mode of the ray factor and a first-hit "rest" ray-cast in `common/`. (2) countable, pure: mask membership m = π·Φ(s/σ_px); viewpoint ρ from odometry. (3) countable belief behind `Model.LidarMeasuresOnGated`: 20-state belief (θ, b_chain, b_cam, b_L), prior counted once, Σ bounded by the class P, accumulation split into camera / LiDAR / θ-only channels. (4) fitter wiring, config, log, contract; then the live A/B.

**Tech Stack:** C++23, Eigen (float model, double solve), OpenCV (membership unit only), DSR (typed attributes), CMake/ctest, Python 3 (table + scoring scripts).

**Spec:** `docs/superpowers/specs/2026-11-15-lidar-measures-design.md` — §7c rev 3.1 (approved by the owner 11-16) supersedes G3, the §7b bullet "m and the down-weights" and §7c revs 1–2. Reviews behind it: `.superpowers/sdd/2026-11-15-lidar-measures/fable-7c-review.md`, `fable-7c-rev2-review.md`, `fable-7c-rev3-review.md`.

**Revision 3 (11-16):** Tasks 0–2 are done (commits ff49e768, 6569e6b1, b58f697b; merged to main 8f4cc821). Tasks 3–10 replace rev 2's Tasks 3–5.

**Execution order (owner 11-16, "go with the shorter"):** 3 → 4 → 5 → 6 → 6b → 9 → 10 (live8: §8 not passed) → owner 11-17 "Go": wall-offset check → 6c → 7 → 8 → 10 again. Tasks 7–8 (the ownership mixture) are DEFERRED until live8 shows whether a wrong-attribution pull exists with physical returns; until then the rays carry the camera's m as a plain weight (Task 1). Task 9 is built WITHOUT the rest scene (no `RestScene` parameter, no `resp_sum` log column); they are added with Tasks 7–8 if live8 asks for them.

## Global Constraints

- Everything new is behind `Model.LidarMeasuresOnGated` (bool, default false); false ⇒ bit-identical to today (`lidar_golden_test`). True in the nightstand, bed and toilet configs.
- No thresholds: weights are continuous (m = π_mask·Φ(s/σ_px); r_i the E-step responsibility); σ_px² = (f/z)²(σ_range² + σ_chain² + mot_pos_var) + q². The ONE bound is the engine's existing `clamp_to_prior` congruence on Σθθ with the class P (`prior_cov_diag`) — never on data or the mean.
- Class prior counted once (spec 7c.1): Σ₀ = `prior_cov_diag` at birth; the relational anchors (wall flush/parallel (1), room axis) are added ONCE, at the first ADMISSIBLE camera update, in the θ-only channel (flag `prior_added`); footprint (2) and height (2a) anchors never under the flag. `depth_unobs_precision` (all its shares: w-axis, depth, flush share) stays per frame, in the CAMERA channel. One-sided constraints (2b), (2c) every frame, θ-only channel.
- Latents (spec 7c.2): b_chain ∈ ℝ² (cx, cy) shared by both channels, marginal `inst.chain_cov_xx/yy`; b_cam ∈ ℝ⁶, marginal = config floors + range lateral/size + obliquity yaw + moment variance; b_L ∈ ℝ⁶, marginal = config floors. Ego-motion (`mot_pos_var`, `mot_size_var`, `mot_yaw_var`) is FRESH per frame, outside every b. Dynamics b_t = ρ D b_{t−1} + w, D = σ_t/σ_{t−1} (1 when σ_{t−1} = 0), Var w = (1 − ρ²)σ_t².
- ρ from ODOMETRY (robot node `robot_current_advance_speed`, `_side_speed`, `_angular_speed`, `robot_current_speed_timestamp`), never from the estimated pose. Camera ρ = exp(−|Δφ|/φ_obj − |Δz|/z), φ_obj = atan(w_⊥/2z). LiDAR lateral/yaw ρ = exp(−|Δφ|/g), H ρ = exp(−|Δz||tan ε|/(g z)), g = `max_ring_gap_rad()`; SensorModel absent ⇒ ρ_L = 1 (logged). Chain ρ = exp(−½ cᵀΣ_chain⁻¹c), c = localiser correction. Stale speeds ⇒ Δ = 0 ⇒ ρ = 1.
- Update (spec 7c.2 steps 1–4): biases restart at b̄ every GN iterate; fresh covariance marginalised inverse-free (I = (𝟙 + Id Σf)⁻¹Id); joint step x_{k+1} = x̃_k + (𝟙 + P̄W)⁻¹[P̄G − (x̃_k − x̄)]; P = (𝟙 + P̄W)⁻¹P̄ symmetrised; 20×20 in double; no P̄⁻¹.
- Mixture (spec 7c.3): box component N(t_i, σ_r² + J_i S_L J_iᵀ), NO Cauchy on the mixture path; rest = room faces + z = 0 (σ_rest = `wall_explain_sigma_m`) and other published boxes (σ_rest = their σ along the ray, floor `common_mode_pos_std`); unexplained 1/L_i; w_u = 0.2; σ_r² = 1/`LidarPrecision` + σ_sensor². Coverage counts Σr at the predicted state, frozen per update. Every other caller of the ray factor bit-identical.
- The size summary `size_lik_` is NOT changed (spec 7c.1, R3-C2).
- Every test fixture with LiDAR returns ray-casts a TRUE scene, first hit only (spec 7c.0).
- `lidar_ray_cost` and the other 8 agents untouched; `ai::update` unchanged.
- CLAUDE.md: and/or/not; `std::from_chars`; `imbue(std::locale::classic())` on written files; `make -j8` after `pgrep -a cc1plus`; type-attributed graph access only; no new graph attribute; `cv::Mat` never crosses a thread; RT transforms at a real timestamp, optional checked.
- Commits on `feat/cost-removal` (worktree `/home/pbustos/robocomp/components/active_inference.worktrees/cost-removal`); the controller pushes the branch when the plan finishes (owner 11-16); never merge to main unless the owner asks.
- Do not patch symptoms: a failing pass criterion is reported with its numbers; no term, exclusion, floor or gate is added to make it pass.

## Review Focus

1. **Parked robot, zero ego-motion** (fresh covariance 0): the fresh marginalisation must be the identity, no 1/1e-9 inverse, no zero information (Task 4 test `fresh_zero_is_identity`).
2. **Chain covariance disabled or a transform missing** (`inst.chain_cov_xx = 0`): a zero-variance latent must run, no NaN, D = 1 (Task 4 `zero_variance_latent_runs`, Task 6 `zero_chain_cov_runs`).
3. **Flag off**: bit-identical (`lidar_golden_test` after Tasks 6, 8, 9).
4. **Odometry missing or stale** (no robot node, old timestamp): ρ = 1 for every channel, logged, no NaN (Task 5 `stale_odometry_is_parked`).
5. **Mode rotation / canonicalisation (w ↔ h swap, yaw + k·90°) with latents**: b_cam, b_L swap their w, h components and blocks, b_chain untouched, P stays symmetric PSD (Task 6 `mode_rotation_permutes_latents`).
6. **A published box node missing size or pose, or the instance's own node**: skipped / excluded, never a crash (Task 7 `rest_cast_skips_incomplete_box`, Task 9 reviewer check).

---

### Task 0 (controller, no subagent): park the uncommitted Task 4c rev 2 work — DONE (wip/task-4c-rev2 1ecc5ab2)

The worktree holds Task 4c rev 2 uncommitted (8 files). It resumes after this plan on the live8 logs.

- [ ] **Step 1:** Verify the saved patch is current:
  `cd <worktree> && git add -N . && git diff HEAD > /tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/now.patch && diff <(grep -v "^index " /tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/now.patch) <(grep -v '^index ' .superpowers/sdd/2026-11-15-cost-based-removal/task-4c-rev2-uncommitted.patch)` → empty.
  If not empty, re-save the patch first (`git diff HEAD > …/task-4c-rev2-uncommitted.patch`).
- [ ] **Step 2:** Commit it to a side branch so it is in git, not only in a patch file:
  `git switch -c wip/task-4c-rev2 && git add -A && git commit -m "WIP Task 4c rev 2 (full-state marginal) — parked until lidar-measures lands; BLOCKED on inflated Σ" && git switch feat/cost-removal`.
  Expected: `git status --short` on `feat/cost-removal` is empty; `git log wip/task-4c-rev2 -1` shows the WIP commit.

---

### Task 1: Golden trajectory first, then per-ray weights and the weighted selection (shared, pure) — DONE (4e18c9de, ff49e768)

**Files:**
- Create: `countable_concept/tests/lidar_golden_test.cpp` (links `../src/countable_belief.cpp`)
- Modify: `common/ai_belief/lidar_ray_factor.h` (struct `LidarRays`, `accumulate_lidar_rays` only)
- Modify: `common/lidar_select/lidar_select.h` (add `select_into_weighted`; `select_into` clears `out.weights`)
- Create: `countable_concept/tests/lidar_weights_test.cpp`
- Modify: `countable_concept/tests/CMakeLists.txt` (both tests)

**Interfaces:**
- Produces:
  - `LidarRays::weights` — `std::vector<float>`; empty ⇒ every ray weight 1; else `assert(weights.size() == endpoints.size())`, each in [0, 1], multiplying `precision` per ray.
  - `int rc::lidar_select::select_into_weighted(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin, const Eigen::Vector2f& centroid_xy, const Box& box, float precision_base, const Weights& w, const std::function<float(const Eigen::Vector3f&)>& membership, rc::ai::LidarRays& out, float* cov_ang = nullptr, float* m_sum = nullptr)` — returns the number of staged endpoints; `*m_sum` = Σm.
  - `lidar_golden_test` — re-run by Tasks 2 and 4; must stay passing.

- [ ] **Step 0: Record the golden trajectory BEFORE any edit** (at the plan's BASE commit, code untouched). In `lidar_golden_test.cpp` (plain `CHECK` macro as in `furniture_prior_test.cpp:13`), build a nightstand-like `CountableBelief` seeded off-truth by (+0.05, −0.04, H +0.03) and run 40 deterministic frames, no `<random>`: closed-form point grids on the truth box's visible faces (200 points, R = σ_base²), 120 Helios rays on the truth box from an origin 1.5 m away at z 1.1075 (`precision 2500`, `robust_c_m 0.05`), a `WallRef`, `room_axis`; the sequence mixes `update` with rays, `update` without rays, and `predict()` (e.g. frame k: k%3==0 rays, k%3==1 no rays, k%5==4 predict). After each frame print `state().vec()`, `covariance().diagonal()` and `log_volatility()` with `%a`. Paste the 40 rows as hex-float tables into the test and assert bit-identity. Comment: bit-exact for this compiler at -O2 only. Run → passes. Commit it alone: `git commit -m "countable tests: golden 40-frame fit trajectory recorded before the lidar-measures change"`.
- [ ] **Step 1: Write the failing tests** in `lidar_weights_test.cpp` (toy model or `CountableBelief`, implementer's choice):
  - `weights_empty_equals_ones`: same rays, `weights = {}` vs `{1,…,1}` ⇒ `Id`, `bd` bit-identical.
  - `weights_half_scales_information`: all 0.5 ⇒ `Id`, `bd` = 0.5 × unweighted (rel. tol 1e-6).
  - `weighted_select_ones_matches_select_into`: synthetic sweep (≥ 80 points, some outside the box); membership ≡ 1 ⇒ same `endpoints`, same `precision` (bit-identical), `weights` all 1, `m_sum == n`.
  - `weighted_select_coverage_counts_msum`: `coverage_n0 = 60`, 100 in-box returns, membership 1 for 30 and 0 for 70 ⇒ `precision == precision_base · (30/60) · ca`, `ca` from the membership-weighted bearings.
  - `weighted_select_all_zero`: membership ≡ 0 ⇒ `m_sum == 0`, `out.precision == 0`, no NaN.
- [ ] **Step 2: Run to verify they fail** (configure the tests build with `cmake -B build -S countable_concept/tests` from the worktree; `make -C build -j8 lidar_weights_test`) → compile error.
- [ ] **Step 3: Implement.**
  - `LidarRays`: `std::vector<float> weights;` ("empty ⇒ 1 per ray (every existing caller)").
  - `accumulate_lidar_rays`: iterate by index; `wi = weights.empty() ? 1 : weights[i]` (assert sizes match when non-empty); skip `wi <= 0`; multiply the Cauchy-weighted contribution by `wi`. `lidar_ray_cost` is not touched.
  - `select_into`: `out.weights.clear();` next to `out.endpoints.clear();`.
  - `select_into_weighted`: same loop as `select_into`; per in-box return `m = membership(p)`; skip `m <= 0`; stage endpoint + weight; Σm and m-weighted bearing sums; `coverage = min(1, Σm / max(1, n0))`; `R = |Σ m·e^{iφ}| / Σm`; `ca = max(0, 1−R)^power`; `precision = precision_base · coverage · ca` when Σm > 0, else 0.
- [ ] **Step 4: Run** `lidar_weights_test` and `lidar_golden_test` → pass; every existing countable test still passes.
- [ ] **Step 5: Commit** `git commit -m "lidar_select + ray factor: per-ray membership weights; coverage counts Σm (spec 2026-11-15 lidar-measures G2)"`

---

### Task 2: The LiDAR-only frame in the belief and the engine (shared + countable belief) — DONE (6569e6b1, b58f697b; its wall/vanishing tests are rebuilt in Task 3)

**Files:**
- Modify: `common/ai_belief/recursive_laplace.h` (`update<N>`)
- Modify: `countable_concept/src/countable_belief.h` (`CountableFrame::lidar_only`, `has_extra_data`, `update()`), `countable_concept/src/countable_belief.cpp` (`accumulate_extra`)
- Create: `countable_concept/tests/lidar_only_update_test.cpp` (links `../src/countable_belief.cpp`)
- Modify: `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `LidarRays::weights` (Task 1).
- Produces:
  - `bool CountableFrame::lidar_only = false;` — "a frame that carries LiDAR rays only (spec §7b, C1 = A)".
  - `bool CountableBelief::has_extra_data(const CountableFrame& f) const` → `not f.lidar.endpoints.empty() and f.lidar.precision > 0`, or the same for any `f.lidar_extra`. The engine detects it with `if constexpr (requires { m.has_extra_data(frame); })`; bottle / chair / table (no hook) unchanged.

- [ ] **Step 1: Write the failing tests** in `lidar_only_update_test.cpp`. Fixture: the Task 1 golden nightstand belief, truth box, 120 truth rays; a LiDAR-only frame is `points` empty, `lidar_only = true`, `z_top_obs = −1`, wall + room axis staged.
  - `empty_frame_is_predict_only`: no points, no rays ⇒ state unchanged, Σ equals `predict()` on a copy, return 0, no NaN.
  - `lidar_only_moves_toward_rays`: one update ⇒ |centre error| and |H error| strictly smaller; Σ(cx,cx), Σ(cy,cy), Σ(H,H) strictly smaller than after a bare `predict()`.
  - `lidar_only_vanishing_membership_is_predict` (C1): 300 LiDAR-only frames, all weights 1e-6 ⇒ state within 1e-4 of the start; Σ diagonal within 0.1 % of 300 `predict()`s on a copy; `log_volatility()` unchanged. Control (in the test, printed, not asserted): the same 300 frames with `lidar_only = false` move the state by > 1e-2 (the anchors re-added) — proves the test discriminates.
  - `lidar_only_saturates` (I9): rays ×1 and ×10 (the same sweep duplicated), `chain_cov_xx = chain_cov_yy = 0.01`. Assert non-vacuity: the raw `Id(0,0)` from `accumulate_extra` on the ×10 frame exceeds 10·Σc⁻¹(0). Assert the bound with full matrices: Σ_post − (Σ_pred⁻¹ + diag(Σc⁻¹))⁻¹ is positive semi-definite up to 1 % (Σc⁻¹ from `common_mode_inv_diag(frame)`). Assert Σ(×10) within 5 % of Σ(×1) per diagonal entry.
  - `lidar_only_repeated_sweeps` (C2): the same sweep as 300 consecutive LiDAR-only frames ⇒ print √Σ(cx,cx), √Σ(cy,cy), √Σ(H,H) and the analytic √(Q·Σc) per DOF (Q = `process_noise_diag()`); assert each settled σ within a factor 2 of its analytic value and finite, non-zero. Record the printed numbers in the task report (they are the C2 baseline for Task 5).
  - `lidar_only_one_face_keeps_depth` (I1): front-face rays only, 200 LiDAR-only frames ⇒ |Δh| and |Δcy| from the start both < 0.02 m (the ray σ).
  - `wall_behind_does_not_pull` (I5): box 5 cm off a wall plane; rays on the box face AND on the wall above and beside it; weights 1 on box rays, 0.001 on wall rays; 100 LiDAR-only frames ⇒ the centre moves toward the wall by < 0.02 m. Control (printed): weights ≡ 1 ⇒ record how far it moves.
- [ ] **Step 2: Run** — `lidar_golden_test` passes; the LiDAR-only tests FAIL (engine returns early / anchors present).
- [ ] **Step 3: Implement.**
  - `update<N>`: replace `if (frame.points.empty()) return 0.0f;` with a return only when there are no points AND (no `has_extra_data` hook, or it returns false). Return 0 for the mean energy when `frame.points.empty()`. No other engine change.
  - `accumulate_extra`: when `f.lidar_only`, skip `accumulate_wall`, the room-axis term, (2) footprint and (2a) height anchor; keep (2b) short-height, (2c) wall no-cross and the rays. (2d) is already inert (`z_top_obs = −1`).
  - `CountableBelief::update`: when `frame.lidar_only`, do not call `update_volatility` (ω stays inferred from camera-measured cycles, spec §7b). Add `has_extra_data`.
- [ ] **Step 4: Run** `lidar_only_update_test` → all pass; `lidar_golden_test` and every other countable test still pass. Shared headers changed: build one non-countable agent at -j8 (`table_concept`) and run the door test `test_door_leaf_freespace` → pass.
- [ ] **Step 5: Commit** `git commit -m "ai engine + countable belief: the LiDAR-only frame (rays + one-sided constraints, no re-added class anchors; spec lidar-measures §7b C1=A)"`

---

### Task 3: Physical test scenes, the responsibility table, and the Task 2 tests rebuilt on them

**Files:**
- Create: `countable_concept/tests/test_scene.h` (header-only test utility)
- Create: `countable_concept/tools/responsibility_table.py`
- Modify: `countable_concept/tests/lidar_only_update_test.cpp`; `countable_concept/tests/lidar_golden_test.cpp` untouched
- Modify: `docs/superpowers/specs/2026-11-15-lidar-measures-design.md` §7c.3 (insert the table)

**Interfaces:**
- Produces:
  - `namespace rc::test`: `struct SceneBox { Eigen::Vector2f c; float yaw, w, h, H; };` (floor-anchored, z ∈ [0, H]); `struct SceneWall { Eigen::Vector2f a, b; float height; };`
  - `std::vector<Eigen::Vector3f> raycast_scene(const Eigen::Vector3f& origin, std::span<const SceneBox> boxes, std::span<const SceneWall> walls, bool floor, float az_step_deg, std::span<const float> ring_elev_deg, float range_max_m)` — one return per (ring, azimuth) ray: the FIRST hit over boxes ∪ walls ∪ (z = 0 if `floor`), none beyond `range_max_m`. Helios defaults for tests: `az_step_deg` and `ring_elev_deg` copied from `webots-shadow/…/shadow-bridge-dds/etc/config.toml` `[Lidar.helios]` (state the source line in a comment).
  - `responsibility_table.py` prints the §7c.3 table (classic formatting, `.` decimals) for: true return m = 0.9 at belief offset 10 / 14 / 23 cm with box σ along the ray 5 cm (room hit in chord / none); wall 2 and 5 cm behind, box σ 0 / 2 / 4 / 6 cm, m = 0.1; neighbour (unpublished) at e = 2 / 5 cm, box σ 0 / 2 / 4 / 6 cm, m = 0.1; coincident foreign box, m = 0.9 / 0.5; over-segmented wall return π = 0.9 at s = +3σ_px. Constants: σ_r² = 1/2500 (+ σ_sensor² = 0.02²), σ_rest = 0.05, L = 0.6, w_u = 0.2.
- [ ] **Step 1:** Write `raycast_scene` and a self-test `scene_first_hit_only` in `lidar_only_update_test.cpp`: a box 5 cm in front of a wall, origin 1.5 m away at z 1.1075 ⇒ no returned point lies on the wall behind the box's silhouette (every wall return's segment from the origin misses the box), and the count of box returns is > 0.
- [ ] **Step 2:** Rebuild every LiDAR fixture in `lidar_only_update_test.cpp` with `raycast_scene` (truth box, the wall, floor). Rewrite `wall_behind_does_not_pull`: back face 2 cm from the wall, physical returns, wall returns at weight 0.1, 2000 LiDAR-only sweeps ⇒ back-face gap within 0.5 cm of a box-returns-only baseline run (both printed). Remove the dead vanishing control (it is predict-only by construction); keep `lidar_only_vanishing_membership_is_predict` asserting at w = 1e-12.
- [ ] **Step 3:** Run `lidar_only_update_test` and `lidar_golden_test` → pass on the CURRENT code (spec 7c.0: with physical returns there is no pull today). Print and record in the report σ_h after 2000 LiDAR-only sweeps (spec: 20.9 cm on the old fixture).
- [ ] **Step 4:** Write `responsibility_table.py` with the formula of spec §7c.3 verbatim; run it; paste its table into §7c.3 under "The responsibility table" replacing the sentence that promises it.
- [ ] **Step 5: Commit** `git commit -m "countable tests: physical first-hit scenes (spec lidar-measures 7c.0); §7c.3 responsibility table from the final formula"`

---

### Task 4: The persistent-latent update (shared, pure)

**Files:**
- Create: `common/ai_belief/persistent_update.h`
- Create: `countable_concept/tests/persistent_update_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `rc::ai::persistent`; N = θ dimension, K = latent dimension, D = N + K; all solves in double):
  - `template<int N> struct FreshInfo { Eigen::Matrix<double,N,N> I; Eigen::Matrix<double,N,1> g; };`
    `template<int N> FreshInfo<N> marginalise_fresh(const Eigen::Matrix<float,N,N>& Id, const Eigen::Matrix<float,N,1>& bd, const Eigen::Matrix<float,N,N>& Sigma_f)` → I = (𝟙 + Id Σf)⁻¹ Id (symmetrised), g = (𝟙 + Id Σf)⁻¹ bd. Identity at Σf = 0.
  - `template<int N, int K> struct Channel { FreshInfo<N> info; Eigen::Matrix<double,N,K> B; };` (the channel measures θ + B b)
  - `template<int N, int K> void predict_latents(Eigen::Matrix<double,N+K,N+K>& P, Eigen::Matrix<double,N+K,1>& x, const Eigen::Matrix<double,K,1>& rho, const Eigen::Matrix<double,K,1>& sigma_prev, const Eigen::Matrix<double,K,1>& sigma_now)` — per latent DOF j: F_j = ρ_j D_j (D_j = σ_now/σ_prev, 1 when σ_prev = 0); x_b ← F x_b; P ← F P Fᵀ on the latent rows/cols (θ–b and b–b cross blocks included); P_bb += diag((1 − ρ²) σ_now²).
  - `template<int N, int K> Eigen::Matrix<double,N+K,1> joint_step(const Eigen::Matrix<double,N+K,1>& x_tilde, const Eigen::Matrix<double,N+K,1>& x_bar, const Eigen::Matrix<double,N+K,N+K>& P_bar, std::span<const Channel<N,K>> channels, const FreshInfo<N>& theta_only)` → x̃ + (𝟙 + P̄W)⁻¹[P̄G − (x̃ − x̄)], W = Σ H_cᵀ I_c H_c + H_θᵀ I_θ H_θ, G = Σ H_cᵀ g_c + H_θᵀ g_θ, H_c = [I B_c], H_θ = [I 0]. Solve with `partialPivLu` on 𝟙 + P̄W.
  - `template<int N, int K> Eigen::Matrix<double,N+K,N+K> joint_cov(const Eigen::Matrix<double,N+K,N+K>& P_bar, std::span<const Channel<N,K>> channels, const FreshInfo<N>& theta_only)` → (𝟙 + P̄W)⁻¹P̄, symmetrised.
- [ ] **Step 1: Write the failing tests** in `persistent_update_test.cpp` (plain `CHECK`):
  - `restart_matches_saturated_1d`: N = 1, K = 1, B = 1, P̄ = diag(0.0036, 4e-4), linear datum z = 0.1 with I = 1e4, prior 0; 4 iterates restarting the bias at b̄ = 0 ⇒ θ = 0.0878 (±1e-4) at EVERY iterate (spec 7c.2, R3-C1); the non-restarting variant drifts to 0.0973 (printed, proves discrimination).
  - `rho0_equals_saturate`: N = 6, a fixed random-but-seeded SPD Id, bd, Σc, P0 (std::mt19937 seed 7 — deterministic); θ posterior and Σθθ from `joint_step`/`joint_cov` with K = 6, B = I, latent prior Σc, ρ = 0 equal `ai::saturate`-based `(P0 + Ieff)⁻¹(P0·0 + beff)` and `(P0 + Ieff)⁻¹` to rel 1e-5.
  - `fresh_zero_is_identity`: Σf = 0 ⇒ I == Id, g == bd exactly; Σf = diag(1e-6) with Id = 2.19e6·𝟙 ⇒ I finite, PD, ≤ Id.
  - `zero_variance_latent_runs`: one latent with σ_prev = σ_now = 0 ⇒ no NaN in x, P; that latent stays 0 with zero variance; D = 1.
  - `rho1_n_sweeps_equal_one_with_n_id`: ρ = 1, no θ noise, linear channel, n = 10 identical updates (predict_latents between) ⇒ θ, Σθθ equal ONE update with 10·Id (rel 1e-6).
  - `shared_latent_position_information`: θ ∈ ℝ¹, latents (b_chain, b_cam, b_L) with variances (s_ch, s_c, s_L), two channels with infinite data information (Id = 1e12) ⇒ θ's data information = 1/(s_ch + (1/s_c + 1/s_L)⁻¹) (rel 1e-4).
  - `cov_symmetric_psd`: joint_cov output symmetric to 1e-12 and min eigenvalue ≥ 0 on the rho0 fixture.
- [ ] **Step 2:** Run → compile failure.
- [ ] **Step 3:** Implement the header as specified (header-only, templates, Eigen fixed sizes, no allocation in the loop beyond Eigen temporaries).
- [ ] **Step 4:** Run → all pass; `lidar_golden_test` passes (nothing uses the header yet).
- [ ] **Step 5: Commit** `git commit -m "ai engine: persistent shared-error latents — inverse-free joint step with bias restart (spec lidar-measures 7c.2)"`

---

### Task 5: Mask membership and viewpoint ρ (countable, pure)

**Files:**
- Create: `countable_concept/src/countable_mask_membership.h/.cpp` (OpenCV allowed here), `countable_concept/src/countable_viewpoint.h/.cpp`
- Modify: `countable_concept/src/countable_projection.h/.cpp` (two helpers), `countable_concept/src/CMakeLists.txt`
- Create: `countable_concept/tests/mask_membership_test.cpp`, `countable_concept/tests/viewpoint_rho_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt` (link OpenCV for the first)

**Interfaces:**
- Produces (membership — rev 2's Task 3 interface, with π_mask added):
  - `class rc::MaskMembership`, `static MaskMembership build(std::span<const Eigen::Vector2f> pixels)` (pixels in the DELIVERED image, col,row); `float stride_px() const` (q = median nearest distinct column step within a row; 1 dense or empty); `float signed_distance_px(float col, float row) const` (positive inside; `cv::distanceTransform` `DIST_L2`, `DIST_MASK_PRECISE` on a raster anchored at the slice's minimum (col,row), step q, one-cell border; scaled by q, offset ±q/2 so a mask pixel reads +q/2; bilinear; outside the raster `s(nearest border cell) − distance to it`); `bool empty() const`. Every `cv::Mat` local to `build()`.
  - `inline float rc::projection_sigma_px(float f_px, float z_m, float sigma_range_m, float sigma_chain_m, float mot_pos_var_m2, float q_px)` → √((f/z)²(σr² + σc² + mot_pos_var) + q²).
  - `inline float rc::membership_weight(float s_px, float sigma_px)` → ½·erfc(−s/(σ√2)).
  - `using Projector = std::function<std::optional<Eigen::Vector3f>(const Eigen::Vector3f&)>;` `std::function<float(const Eigen::Vector3f&)> rc::make_membership(Projector proj, const MaskMembership& mm, float f_px, float sigma_range_m, float sigma_chain_m, float mot_pos_var_m2, float pi_mask)` → nullopt ⇒ 0; else `pi_mask · membership_weight(s, projection_sigma_px(…, mm.stride_px()))`; captures `mm` by value.
  - `std::optional<Eigen::Vector3f> CountableProjection::project_to_zed(const Eigen::Matrix4f& zed_T_room, const Eigen::Vector3f& p_room, int frame_cols, int frame_rows) const` → (col, row, depth) in the delivered image (scale as `countable_projection.cpp:449–452`); nullopt when depth < 0.20 m, outside the image, no CameraAPI, or sizes ≤ 0. `float CountableProjection::focal_px(int frame_cols, int frame_rows) const` → ½(fx·sx + fy·sy), 0 if unavailable.
- Produces (viewpoint):
  - `struct rc::ViewpointPath { float dphi_rad = 0, dz_m = 0; };` (accumulated since the channel's last update of the instance; reset by the caller after an update)
  - `void rc::accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room, const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy)` — adds |step·t̂|/z to dphi and |step·r̂| to dz, r̂ = unit(sensor − object), t̂ ⊥ r̂, z = |sensor − object| (z → 0 guarded by returning without change when z < 1e-3: a degenerate geometry, not a tuning threshold — state it in a comment).
  - `Eigen::Vector2f rc::odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s)` → room-frame displacement of the robot over dt.
  - `float rc::rho_camera(const ViewpointPath&, float z_m, float w_perp_m)` → exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z).
  - `float rc::rho_lidar_lateral(const ViewpointPath&, float ring_gap_rad)` → exp(−|Δφ|/g); `float rc::rho_lidar_height(const ViewpointPath&, float z_m, float eps_rad, float ring_gap_rad)` → exp(−|Δz||tan ε|/(g z)); g ≤ 0 (SensorModel absent) ⇒ 1.
  - `float rc::rho_chain(const Eigen::Vector2f& correction_xy, float var_xx, float var_yy)` → exp(−½(cx²/var_xx + cy²/var_yy)); a zero variance ⇒ that term 0 when its correction is 0, else ρ = 0.
- [ ] **Step 1: Write the failing tests.**
  - `mask_membership_test`: rev 2's list — `inside_one_boundary_half_outside_zero` (40×30 rectangle: centre m > 0.999 at σ 2; 10 px outside < 1e-3; half a pixel past the edge |m − 0.5| < 0.02; an edge pixel s = +q/2 ± 0.05), `symmetric_in_s` (1e-6, s ∈ {0.3, 1, 4}), `sigma_grows_as_z_shrinks`, `stride_of_subsampled_mask` (stride 3; the edge test passes on it), `thin_mask`, `outside_raster_continuous` (< 0.1 px across the border; 200 px away m < 1e-6), `make_membership_nullopt_is_zero`, `make_membership_behind_is_zero`, plus `pi_mask_scales` (π = 0.6 ⇒ m at the centre = 0.6 ± 1e-3).
  - `viewpoint_rho_test`: `parked_is_one` (zero steps ⇒ every ρ == 1); `stale_odometry_is_parked` (dt = 0 or a speed of 0 ⇒ ρ == 1); `camera_efold_is_half_width` (nightstand w⊥ = 0.51 at z = 1, 2, 3 m: a tangential path of w⊥/2 gives ρ_cam = e⁻¹ ± 2 %); `lidar_efold` (g = 0.0392: tangential path z·g gives e⁻¹ ± 1 %); `composes` (one 30 cm step vs 100 steps of 3 mm on the same arc ⇒ equal ρ within 1e-4); `rotation_in_place_is_one` (pure rotation ⇒ ρ == 1); `absent_ring_gap_is_one` (g = 0 ⇒ ρ_L == 1); `rho_chain_zero_correction_is_one`.
- [ ] **Step 2:** Run → compile failure. **Step 3:** Implement. **Step 4:** Run both tests → pass; `countable_concept` agent builds at -j8.
- [ ] **Step 5: Commit** `git commit -m "countable: mask membership m = π·Φ(s/σ_px) and viewpoint ρ from odometry (spec lidar-measures G2, 7c.2)"`

---

### Task 6: The countable belief with persistent latents and the class prior counted once (flag on)

**Files:**
- Modify: `countable_concept/src/countable_belief.h/.cpp`
- Create: `countable_concept/tests/persistent_belief_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 4 `rc::ai::persistent::*`.
- Produces:
  - `CountableBeliefParams::persistent = false;` (set from `Model.LidarMeasuresOnGated` by the fitter in Task 9).
  - `CountableFrame` additions (all default 0 / 1, read only when `persistent`): `float rho_cam[6]`, `rho_L[6]`, `rho_chain` (defaults 0 ⇒ today's per-frame behaviour); `float sig2_cam[6]`, `sig2_L[6]`, `sig2_chain[2]` (latent marginal variances of this frame); `float fresh_cam[6]`, `fresh_L[6]` (ego-motion variances).
  - Belief state under `persistent`: `Eigen::Matrix<double,20,20> P_` and `Eigen::Matrix<double,14,1> b_` (order: b_chain (cx, cy), b_cam (6), b_L (6)); `bool prior_added_ = false;`. `covariance()` returns the θθ block (float). `latent_mean()`, `latent_std()` getters for the log.
  - Accumulation split (a private helper used only when `persistent`): `void accumulate_channels(const CountableBeliefState& s_cam, const CountableBeliefState& s_L, const CountableFrame& f, Mat6& Id_cam, Vec6& bd_cam, Mat6& Id_L, Vec6& bd_L, Mat6& Id_th, Vec6& bd_th) const` — camera: points + z_top_obs/top-no-float + moments + azimuth + `depth_unobs_precision` shares; LiDAR: `accumulate_lidar_rays` (+ `lidar_extra`); θ-only: (2b), (2c), and — on the first admissible camera update only — wall flush/parallel (1) and the room axis. Footprint (2) and height (2a) anchors are never added under `persistent`.
  - `update(frame)` under `persistent`: predict (θθ: +Q, then `clamp_to_prior` congruence applied to θ's rows/cols of P_; latents: `predict_latents` with the frame's ρ and σ), then `gn_iters` iterates with x̃_k = (θ_k, b̄), camera traced at θ_k + E b̄_chain + b̄_cam, rays at θ_k + E b̄_chain + b̄_L, `marginalise_fresh` per channel, `joint_step`; then `joint_cov`; `prior_added_ = true` after an admissible camera update. ω as today (camera-measured cycles only, post minus previous posterior). Flag off: today's code path, untouched.
  - `sanitize_covariance`: re-seeds θθ to P, latents to their marginals, cross terms 0, `prior_added_ = false`. `canonicalize`/`apply_mode_rotation`: swap the w, h components of b_cam, b_L and their P rows/cols with θ's; b_chain untouched.
- [ ] **Step 1: Write the failing tests** (physical scenes from Task 3):
  - `flag_off_golden`: `lidar_golden_test` still passes (run, not copied).
  - `rho0_camera_only_equals_engine`: `persistent = true`, ρ = 0, `prior_added_` already true, `depth_unobs_precision = 0`, a camera-only frame ⇒ θ and Σθθ equal `ai::update` on the same frame content with Σc = the summed marginals (camera latents + chain) and the anchors off, rel 1e-4 at every GN iterate (expose a test hook if needed).
  - `rho0_lidar_only_equals_engine`: the same for a LiDAR-only frame.
  - `prior_added_once`: first update LiDAR-only ⇒ `prior_added_ == false`; first admissible camera update ⇒ true, and the relational anchors' information appears in that update only (a second identical camera frame adds no flush/axis term — check via the test hook's θ-only Id).
  - `aged_belief_returns_to_class_P`: after birth + one camera update, 5000 `predict()` ⇒ θ mean unchanged (1e-6), Σθθ diagonal = `prior_cov_diag()` within 2 %.
  - `wall_behind_does_not_pull` (moved from Task 3, ruling T3-b): Task 3's physical scene, back face 2 cm from the wall, wall returns at weight 0.1, 2000 LiDAR-only sweeps, `persistent = true` ⇒ back-face gap within 0.5 cm of the box-returns-only baseline (both printed). A FAIL is reported to the owner as evidence for Tasks 7–8, not tuned.
  - `physical_lidar_only_sigma_bounded`: Task 3 scene, 2000 LiDAR-only sweeps ⇒ √Σ(h,h) ≤ `prior_footprint_std` (+1 %) and the back face moves < 1 mm.
  - `parked_camera_is_one_look`: 100 identical camera frames at ρ = 1, zero fresh covariance ⇒ Σθθ equals the one-frame result with 100·Id (rel 1 %).
  - `zero_chain_cov_runs`: `sig2_chain = 0` ⇒ no NaN, update proceeds.
  - `mode_rotation_permutes_latents`: set distinct b_cam/b_L w and h values, apply a 90° mode rotation ⇒ they swap; b_chain unchanged; P symmetric, min eigenvalue ≥ −1e-12.
  - `sanitize_reseeds`: corrupt P_ ⇒ after `sanitize_covariance` θθ = P, latents = marginals, `prior_added_ == false`.
- [ ] **Step 2:** Run → fail. **Step 3:** Implement. **Step 4:** Run the new test + every countable test → pass (`lidar_golden_test` included); build `countable_concept` at -j8.
- [ ] **Step 5: Commit** `git commit -m "countable belief: persistent shared errors (b_chain, b_cam, b_L) and the class prior counted once, behind Model.LidarMeasuresOnGated (spec 7c.1–7c.2)"`

---

### Task 6b: First-crossing rays and fixed-precision one-sided constraints on the persistent path (spec 7c.6)

**Files:**
- Modify: `common/ai_belief/lidar_ray_factor.h` (`LidarRays`: `bool first_crossing = false;` `float surface_sigma_m = 0;`; `accumulate_lidar_rays` honours them), `countable_concept/src/countable_belief.h/.cpp` (persistent path sets them; (2c)/(2b) fixed precision on the θ-only channel)
- Modify: `countable_concept/tests/persistent_belief_test.cpp`; Create: `countable_concept/tests/creep_set_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 6 persistent path; Task 3 `raycast_scene`. Reference candidate (scratch, not to be copied blindly): `/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/creep/src/ray_v2.h`, `creep_hooks.h`; evidence in `.superpowers/sdd/2026-11-15-lidar-measures/creep-investigation.md`.
- Produces: `LidarRays::first_crossing` — when true a ray contributes only if the box SDF changes sign along it before the measured range (+ the factor's existing tolerance band), the crossing refined by bisection to `1e-4` m; the residual and Jacobian are taken at the crossing; variance σ_r² + σ_s²/(n̂·u)², σ_s = `surface_sigma_m`. Default false ⇒ bit-identical for every caller. Persistent countable path: `first_crossing = true`, `surface_sigma_m = model_sigma_m`; (2c) λ = 1/`f.wall.sigma_m`² and (2b) λ = its existing gain, both constant, ½λr² on violation only.
- [ ] **Step 1: Failing tests.** `creep_set_test`: the 16-case set (cx offsets {0, 0.5, 1, 2} mm × w offsets {0, 1} mm × wall returns {on, off}), persistent path, 4000 LiDAR-only sweeps each ⇒ |Δback face| < 1 mm, |Δw| < 2 mm, |Δfront face| < 2 mm, no NaN; prints every case. `persistent_belief_test`: `physical_lidar_only_sigma_bounded` also asserts |Δw| < 2 mm; `no_cross_pushes_back_persistent` LiDAR-only and camera cases assert convergence (last GN step < 1 mm) and final gap > −0.005 m. Unit: `first_crossing_ignores_near_miss` (a ray passing 1 mm outside an edge contributes nothing; the legacy loop does), `first_crossing_hit_matches_legacy_on_face` (a ray square on a face: same residual, Jacobian within 1e-4).
- [ ] **Step 2:** Run → fail (creep set fails 10/16 on HEAD per the investigation).
- [ ] **Step 3:** Implement as specified. Flag-off and every non-countable caller unchanged.
- [ ] **Step 4:** All countable tests pass (config_migration_test excepted, fixed in Task 9); `lidar_golden_test` bit-exact; build `table_concept` and run `test_door_leaf_freespace` (shared header) → pass; one 20 000-sweep run of the creep set printed in the report.
- [ ] **Step 5: Commit** `git commit -m "persistent path: first-crossing LiDAR hits and fixed-precision one-sided constraints (spec 7c.6)"`

---

### Task 6c: The class prior as a factor at the class mean, counted once (spec 7c.1 amendment 11-17)

**Files:** Modify `countable_concept/src/countable_belief.h/.cpp` (persistent path's first update), `countable_concept/tests/persistent_belief_test.cpp`.

**Interfaces:** Consumes Task 6's persistent path. Produces: under `persistent`, the instance's FIRST update uses prior mean x̄_θ = (seed cx, seed cy, class H = `prior_height_m`, class w = `prior_footprint_m`, class depth = `depth_mean()`, yaw = nearest multiple of 90° to the seed relative to `f.room_axis` when `f.room_axis_ok`, else the seed yaw) with P̄_θθ = `prior_cov_diag()`, and GN start θ₀ = the seed. A `first_update_done_` flag (cleared by `sanitize_covariance`, copied by the snapshot). Flag off unchanged.

- [ ] **Step 1: Failing tests.** `bad_seed_recovers`: seed h = 2.14 (class depth 0.51), one-face camera frames of a true 0.51-deep nightstand (Task 3 scene) ⇒ after the first update |h − 0.51| < 0.10 and after 10 frames < 0.05; `good_seed_stays`: seed at the truth ⇒ first update moves size < 1 cm; `prior_counted_once`: a second identical update with the class mean changed in params moves nothing (the prior is not re-read); `first_update_flag_life`: sanitize clears it, snapshot copies it. `lidar_golden_test` bit-exact.
- [ ] **Step 2:** Run → fail. **Step 3:** Implement. **Step 4:** All countable tests pass; agent builds.
- [ ] **Step 5: Commit** `git commit -m "persistent path: class prior at the class mean, counted once; the seed is only the GN start (spec 7c.1 amendment)"`

---

### Task 7 (UN-DEFERRED 11-17 by live8 evidence): The ownership mixture in the ray factor and the rest ray-cast (shared, pure)

**Files:**
- Modify: `common/ai_belief/lidar_ray_factor.h` (`LidarRays`, `accumulate_lidar_rays`)
- Create: `common/ray_cast/rest_raycast.h` (header-only)
- Create: `countable_concept/tests/ownership_mixture_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `LidarRays` additions (empty ⇒ today's behaviour, bit-identical for every other caller): `std::vector<float> rest_t, rest_sigma;` (per ray; `rest_t < 0` ⇒ no rest surface predicted in the chord); `std::vector<float> chord_L;` (per ray, m); `Eigen::Matrix<float,6,6> S_pred;` (predicted covariance of θ + E b_chain + b_L); `float sigma_r2 = 0;` `float w_u = 0.2f;` `float coverage_scale = 1;`. Mixture mode ⇔ `not chord_L.empty()`.
  - In mixture mode, per ray at the current trace: box density N(r; t, σ_r² + J S_pred Jᵀ); rest density (1 − w_u)N(r; t_rest, σ_r² + σ_rest²) + w_u/L or 1/L; r_i = m_i·box/(m_i·box + (1 − m_i)·rest) with m_i = `weights[i]`; contribution r_i·JJᵀ/σ_r² and −r_i·J·e/σ_r², NO Cauchy factor; multiplied by `coverage_scale`. Optional out-param `std::vector<float>* resp` for logging/tests.
  - `struct rc::raycast::RestBox { Eigen::Vector2f c; float yaw, w, h, H, sigma; };` `std::optional<std::pair<float,float>> rc::raycast::first_rest_hit(const Eigen::Vector3f& o, const Eigen::Vector3f& dir, float t_lo, float t_hi, std::span<const Eigen::Vector2f> room_polygon, float wall_height, float sigma_room, std::span<const RestBox> boxes)` → (t, σ) of the first hit in [t_lo, t_hi] on the polygon's vertical faces (z ≤ wall_height; `wall_height ≤ 0` ⇒ unbounded), z = 0, or a box; σ = sigma_room for room/floor, the box's `sigma` for a box. `float rc::raycast::chord_through_box(const Eigen::Vector3f& o, const Eigen::Vector3f& dir, const rc::lidar_select::Box& sel)` → the chord length L (0 if missed).
- [ ] **Step 1: Write the failing tests:**
  - `empty_inputs_bit_identical`: a door-like caller (no `chord_L`) ⇒ Id, bd bit-identical to BASE (record before editing, as Task 1 did).
  - `responsibility_table`: each row of the Task 3 table within 10 %.
  - `no_cauchy_on_mixture_path`: a ray at e = 5 cm with r = 1 forced (m = 1, no rest, 1/L negligible… use L = 1e6) ⇒ information = JJᵀ/σ_r² exactly (no 0.5 Cauchy factor).
  - `rest_cast_first_hit`: a box in front of a wall ⇒ the box hit; room polygon square ⇒ the wall at the right t; floor when pointing down.
  - `rest_cast_skips_incomplete_box`: a RestBox with w, h or H ≤ 0 ⇒ ignored (no hit, no NaN).
  - `chord_through_box`: axis-aligned ray through a 0.6 m box ⇒ 0.6 (1e-5); missing ray ⇒ 0.
- [ ] **Step 2:** Run → fail. **Step 3:** Implement. **Step 4:** Run → pass; build `table_concept` and run `test_door_leaf_freespace` (shared header changed) → pass; `lidar_golden_test` passes.
- [ ] **Step 5: Commit** `git commit -m "ray factor: ownership-mixture mode (PDA box variance, room + published boxes as rest, no Cauchy); first-hit rest ray-cast (spec 7c.3)"`

---

### Task 8 (UN-DEFERRED 11-17 by live8 evidence): The belief uses the mixture (countable, flag on)

**Files:**
- Modify: `countable_concept/src/countable_belief.h/.cpp`
- Create: `countable_concept/tests/ownership_belief_test.cpp`; Modify `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 6 belief, Task 7 `LidarRays` mixture fields and `first_rest_hit`.
- Produces: under `persistent`, `CountableBelief::update` fills `frame.lidar.S_pred` (from P̄: θθ + chain + b_L blocks through H_L) and `coverage_scale` (Σ r at the predicted state ÷ the selection's Σm, computed once before the GN loop and frozen) on a working copy of the rays. The fitter (Task 9) supplies `rest_t`, `rest_sigma`, `chord_L`, `sigma_r2`.
- [ ] **Step 1: Write the failing tests** (Task 3 scenes; LiDAR-only sweeps unless stated):
  - `recovers_from_14cm_parked`: true box 14 cm off the belief, m = 0.9, ρ_L = ρ_chain = 1 ⇒ residual error = Σb/(P + Σb)·14 cm (analytic from the test's own P and Σb; spec: 3.7 cm at σ 5 cm with the 3 cm floor) within 10 %.
  - `recovers_from_14cm_moving`: the same with the origin moving on an arc so ρ < 1 each sweep ⇒ error < 3 cm.
  - `over_deep_belief_is_not_held`: belief 10 cm deeper than the truth, back against a wall ⇒ wall returns crossing the belief get r < 0.01 (printed list), depth not corrected by these sweeps (documents the choice).
  - `neighbour_box_explains_its_returns`: a published neighbour RestBox 3 cm beside the face, its returns at m = 0.1, 2000 sweeps ⇒ pull below the bound computed from the table row (stated in the test).
  - `coincident_foreign_box`: a foreign box coincident with the object ⇒ true returns r = 0.90 / 0.49 at m = 0.9 / 0.5 (±10 %).
  - `unpublished_neighbour`: the same neighbour without its RestBox ⇒ the pull is printed and pinned (the stated limit).
  - `analytic_pull_7c2_alone`: weights = m, no mixture fields, foreign surface within σ, belief ≠ truth ⇒ pull = β·G/(G + 1/P) within 10 %.
- [ ] **Step 2:** Run → fail. **Step 3:** Implement. **Step 4:** Run → pass; all countable tests pass.
- [ ] **Step 5: Commit** `git commit -m "countable belief: LiDAR returns by posterior ownership (spec 7c.3)"`

---

### Task 9: Wire it into the fitter behind `Model.LidarMeasuresOnGated`, plus config, log and contract

**Files:**
- Modify: `countable_concept/src/countable_config.h/.cpp` (`bool lidar_measures_on_gated = false;` via `getb("Model.LidarMeasuresOnGated", false, "…")`; sets `CountableBeliefParams::persistent`)
- Modify: `countable_concept/src/countable_fitter.h/.cpp`, `countable_concept/src/countable_lidar_range_channel.h/.cpp`, `countable_concept/src/countable_instance.h`
- Modify: `nightstand_concept/etc/config.toml`, `bed_concept/etc/config.toml`, `toilet_concept/etc/config.toml` (`LidarMeasuresOnGated = true` in `[Model]`, one-line comment citing the spec)
- Modify: `countable_concept/KEYMAP.md`, `countable_concept/tests/config_migration_test.cpp` (add `"Model.ExistenceCarveParts"`, `"Model.LidarMeasuresOnGated"` to `produced`)
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` (§2 paragraph + countable audit row)

**Interfaces:**
- Consumes: Tasks 5–6 (Tasks 7–8 deferred: the `rest` parameter, `RestScene`, the rest-box staging and the `resp_sum` column are NOT built in this pass).
- Produces:
  - `CountableObservation::slice_index` (int, −1) set by `observe_slice`.
  - `void CountableLidarRangeChannel::feed(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud, const std::function<float(const Eigen::Vector3f&)>& membership = {}, const RestScene* rest = nullptr) const` — empty membership ⇒ today's `select_into`; else `select_into_weighted`, Σm into `inst.dbg_lidar_msum`; with `rest`, fills `rest_t`, `rest_sigma`, `chord_L` per ray and `sigma_r2`. `struct RestScene { std::vector<Eigen::Vector2f> polygon; float wall_height; float sigma_room; std::vector<rc::raycast::RestBox> boxes; };`
  - `CountableInstance` additions: `rc::ViewpointPath path_cam, path_L;` `Eigen::Vector2f last_pose_xy; float last_pose_yaw; uint64_t last_odo_ts = 0;` `int dbg_lidar_only = 0; float dbg_lidar_msum = 0;`
  - Fitter members: `std::vector<rc::raycast::RestBox> rest_boxes_; uint64_t rest_boxes_ts_ = 0;` restaged at ≤ 1 Hz from every `box`-type node except the instance's own (same agent's other instances included): size from `width_m_att`, `depth_m_att` and the node's height attribute, pose via its RT edge at a real timestamp (optional checked), σ = the node's published std if any else `common_mode_pos_std`; a node missing any ⇒ skipped. Deep copies; built on the compute thread.
- [ ] **Step 1: Failing check first.** `config_migration_test` → FAIL on `Model.ExistenceCarveParts` (known defect); add both keys and the toml lines → PASS.
- [ ] **Step 2: Flag off** ⇒ `feed(inst, frame, frame.points, {}, nullptr)`; the gated branch is exactly `predict()`; refactors are mechanical moves (reviewer checks against BASE). `lidar_golden_test` passes.
- [ ] **Step 3: Flag on, in `run_inference`:**
  - **Odometry and ρ.** Each cycle read the robot node's speeds + `robot_current_speed_timestamp` (typed attributes); `dt` = timestamp difference since `last_odo_ts` (0 if stale/absent ⇒ no step); step = `odometry_step_room(...)` at the current room yaw; `accumulate_path` into both paths with the sensor positions (ZED / Helios mounts per ROBOT_GEOMETRY.md) and the belief centre. On an update of a channel compute its ρ (camera: `rho_camera` with z and w_⊥ from the belief; LiDAR: `rho_lidar_lateral` for cx, cy, w, h, yaw and `rho_lidar_height` for H, ε of the ring hitting the top, g from the Helios SensorModel), then reset that path. ρ_chain from the localiser correction (pose increment ⊖ odometry increment over the same stamps) with `inst.chain_cov_xx/yy`; on a relocalisation event ρ_chain = 0.
  - **Latent covariances.** `sig2_chain = (inst.chain_cov_xx, inst.chain_cov_yy)`; `sig2_cam` = config floors + range lateral/size + obliquity yaw + moment variance (today's fill minus chain and motion); `sig2_L` = config floors; `fresh_cam` = `fresh_L` = (mot_pos_var, mot_pos_var, mot_size_var ×3, mot_yaw_var). Flag off: today's fill verbatim.
  - **Membership** (flag on, `slice_index ≥ 0`, camera, slice pixels, delivered size from `rgb_frame_`): `make_membership(project_to_zed at the mask stamp, mm, focal_px, cfg_.existence_sensor_sigma_m, √(½(chain_xx + chain_yy)), mot_pos_var, slice.confidence)`; any input missing ⇒ no membership.
  - **Admissible branch:** `feed(..., membership, &rest_scene)`. **Gated branch:** flag on and membership ⇒ the LiDAR-only frame (`lidar_only = true`, no points, `z_top_obs = −1`, wall/interior/axis staged, anchor = candidate ∪ residual points), else `predict()`. No rays or precision 0 ⇒ `predict()`. The same snapshot / update / step-bound reject as the admissible branch (shared lambda). Energy, FE baseline, surprise, `resolve_orientation`, `resolve_front_from_rgb` untouched on LiDAR-only cycles. `dbg_lidar_only` 1 there, 0 otherwise. `[AI2]` line prints `LIDAR-ONLY`.
  - **ai2_log:** append at the END of header and row: `lidar_only, lidar_msum, rho_cam, rho_L, rho_chain, dphi_cam, dz_cam, bchain_x, bchain_y, bcam_H, bL_H, sbcam_H, sbL_H, resp_sum, ray_n, Itr_cam, Itr_L` (ρ as the cx component; Itr = trace of the channel's I). Classic locale.
- [ ] **Step 4: Contract.** `CONCEPT_AGENT_LIFECYCLE.md` §2: rev 2's paragraph, plus: "The class prior is counted once (birth + the first admissible camera frame's relational terms); a shared sensor error persists across frames and decorrelates with the viewpoint, so a parked look is one look." Audit row UPDATE: `✅ trunc + confirm_only for camera points; mask-attributed LiDAR-only updates on gated cycles; persistent shared errors (2026-11-16)`.
- [ ] **Step 5: Build and test.** `countable_concept` at -j8; every countable test → pass. Start nothing live.
- [ ] **Step 6: Commit** `git commit -m "countable: camera attributes, LiDAR measures — wiring behind Model.LidarMeasuresOnGated (odometry ρ, latents, membership, rest scene; spec 2026-11-15 §7c rev 3.1)"`

---

### Task 10 (controller, live): objects_live8, objects_live8off, objects_live8shift, then the verdict

Live; the controller runs it. Webots is shared: claim it from the chat and room sessions first, FREE after. The cortex was reinstalled 11-16 09:21: rebuild every agent of the fleet from its worktree before the run.

- [ ] **Step 1:** Rebuild nightstand, bed and toilet from the worktree. Three runs of the objects_live6 tour with the room session's validated fleet set (`CONCEPT_WT=<worktree>`): `objects_live8` (flag on), `objects_live8off` (flag off via a run-local config copy), `objects_live8shift` (flag on, the nightstand's `AI2PriorFootprintM` and `AI2PriorHeightM` shifted by +1σ, i.e. +0.06 and +0.10, in a run-local copy).
- [ ] **Step 2:** Score with `countable_concept/tools/lidar_measures_score.py RUN_DIR [OFF_DIR]` (classic locale; `ai2_log.csv` per class + `gt_pose.tsv`, GT `scene_threerooms.csv`). A *look* = a maximal run of rows with a fresh mask and no gap > 2 s. Per real object (bed, both nightstands, toilet), against live6 and the off run: final centre error (live6: 5.5 cm; 2–6 cm; 23 cm); |H − true top| (smaller than live6); per DOF (cx, cy, H, h) after the first look within 2.5 m: rms true error, min/max std, PASS iff max std ≤ 0.15 m AND rms/3 ≤ std ≤ 3·rms, median NEES; the largest centre step toward a wall or neighbour between looks (≤ 10 cm); fit rate (≈ 10 Hz); share of `lidar_only`; medians of ρ_cam, ρ_L, ρ_chain and of the measured Δ while parked. For the shift run: final w, h, H against the TRUE size (the belief must not sit at the shifted class mean). Each line ends PASS or FAIL.
- [ ] **Step 3:** Write `docs/superpowers/results/2026-11-16-lidar-measures-live8.md` and commit. A depth FAIL of the two-sided check on one-face views is the expected boost outcome (spec 7c.1) — reported as such. Any other FAIL is reported with its numbers, not tuned; a std below rms/3 goes to the owner as a model question.
- [ ] **Step 4:** Push `feat/cost-removal`. If the run passes, the removal plan resumes: re-apply `wip/task-4c-rev2` and run its G0b replay on objects_live8 (real objects ≥ 0, phantoms < 0).

---

## Revision 4 (11-18): spec §7c.7 rev 3.2.1 — persistent errors never rescale; e = the robot's pose error

**Why:** live9 nightstand_9 ended 0.374 m off: the D = σ_t/σ_{t−1} transition made a change of our own σ an observation
(root cause `.superpowers/sdd/2026-11-15-lidar-measures/bchain-jump-report.md`; failing test on `wip/bchain-jump`
f5b5496f). Spec §7c.7 rev 3.2.1 (owner "Go" 11-18) folds Fable's review `fable-7c-rev3.2-review.md` (R4-*).

**Execution order:** 11 → 12 → 13 → 14 → 10 (as `objects_live10`, `…off`, `…shift`).

**Global Constraints, rev 4 (replace the "Latents" and "Dynamics" parts of the Latents line above; all others stand):**
- Latents: e = (δx, δy, δψ) ∈ ℝ³ (robot pose error, room frame, e = estimated − true) shared by both channels;
  b_cam ∈ ℝ⁶, b_L ∈ ℝ⁶ as before. K = 15; offsets kChain = 0 (3), kCam = 3, kL = 9; the joint state is 21.
- Dynamics: b_t = R F_t b_{t−1} + w, R = diag(ρ), Var w = S Σ_t S, S = diag(√(1 − ρ²)); NO D, NO Q. F = I for b_cam,
  b_L; F_t = [[1, 0, −Δr_y], [0, 1, Δr_x], [0, 0, 1]] for e. New source (never predicted, or previous Σ zero) ⇒ ρ = 0.
- B_chain (6×3): row cx [1, 0, −ℓ_y], row cy [0, 1, ℓ_x], row yaw [0, 0, 1]; ℓ is instance state (set at a new source
  to seed centre − robot position, then ℓ ← ρ_chain(ℓ − Δr) + (1 − ρ_chain)(c − r_t)), constant over GN iterates.
- Σ_pose,t = J Σ_edge Jᵀ, Σ_edge = slots {0, 1, 5} of the robot→room edge's `rt_covariance` at the capture stamp,
  J = `se2_inverse_jacobian` (common/place_memory/place_map.h) at the stored edge pose.
- GN: Levenberg on θ only, (𝟙 + P̄(W + λM))δ = P̄G − (x̃ − x̄), M = blockdiag(diag W_θθ, 0); accept iff J falls
  (λ ← λ/4) else λ ← 4λ, first λ = 10⁻³; stop at θ step < 1 mm (cx, cy, H, w, h) and 1 mrad (yaw, wrapped, after
  `apply_constraints`) or `gn_iters_max` = 12 steps; P at the returned θ plus δ_last δ_lastᵀ on θθ; `gn_conv` logged,
  no gate. Flag-off path keeps `gn_iters` = 4 and is bit-identical.
- Everything else in the Global Constraints above stands (flag, no thresholds, locale, -j8, commits on feat/cost-removal,
  no symptom patches).

**Review Focus, rev 4:**
1. **No robot edge covariance** (empty or not a multiple of 36, or no entry at the stamp): Σ_pose = 0 ⇒ a zero-variance
   e that runs, logged once (Task 12 `pose_cov_absent_is_zero`, Task 13 `zero_pose_cov_runs`).
2. **A relocalisation while an instance is parked** (c large ⇒ ρ_chain ≈ 0): ℓ re-takes c − r_t and e is a fresh draw at
   Σ_pose; no step in θ (Task 13 `relocalisation_redraws_e`).
3. **Robot yaw near ±π** in the SE(2) correction and the transport: the ψ difference is wrapped (Task 12
   `chain_correction_wraps_psi`).
4. **A frame where every GN step is rejected** (λ grows to the cap): θ returns at θ₀, P is still taken, `gn_conv` = 0, no
   NaN (Task 14 `all_steps_rejected_is_finite`).
5. **Flag off**: bit-identical (`lidar_golden_test` after Tasks 13 and 14).

### Task 11: Block predict and the damped joint step (shared, pure)

**Files:** Modify `common/ai_belief/persistent_update.h`; Test `countable_concept/tests/persistent_update_test.cpp`.

**Interfaces:**
- Produces, in `rc::ai::persistent`:
  - `template <int N, int K> void predict_block(Eigen::Matrix<double, N + K, N + K>& P, Eigen::Matrix<double, N + K, 1>& x, int off, int n, const Eigen::MatrixXd& F, const Eigen::VectorXd& rho, const Eigen::MatrixXd& Sigma_now, bool new_source)` — latent block at x offset N + off, size n: x_b ← R F x_b; P's rows and columns of the block (cross blocks included) by R F; P_bb += S Σ S. `new_source` ⇒ R = 0.
  - `template <int N, int K> struct DampedStep { Eigen::Matrix<double, N + K, 1> delta, y; };`
  - `template <int N, int K> DampedStep<N, K> joint_step_damped(const Vec& x_tilde, const Vec& x_bar, const Mat& P_bar, std::span<const Channel<N, K>> channels, const FreshInfo<N>& theta_only, double lambda)` — solves (𝟙 + P̄(W + λM))δ = P̄G − (x̃ − x̄), M = blockdiag(diag W_θθ, 0); y = G − (W + λM)δ.
  - `template <int N, int K> double prior_energy(const DampedStep<N, K>& s, const Vec& x_new, const Vec& x_bar)` = ½ yᵀ(x_new − x̄).
  - `predict_latents` is REMOVED; `joint_step` stays (= `joint_step_damped` at λ = 0 plus x̃).

- [ ] **Step 1: Write the failing tests** in `persistent_update_test.cpp`; delete the four D-contract checks (l.211–215) and add:
  - `block_predict_no_rescale`: n = 1, ρ = 1, Σ_now 2× Σ_prev, x_b = 0.3 ⇒ x_b stays 0.3 and P_bb unchanged (1e-12).
  - `block_predict_marginal_mix`: ρ = 0.6, F = I ⇒ P_bb = 0.36·P_bb,prev + 0.64·Σ_now (1e-12); cross block θ–b scaled by 0.6.
  - `block_predict_transport`: n = 3, F = [[1,0,−0.2],[0,1,0.5],[0,0,1]], ρ = 1 ⇒ x_b = F x_b, P_bb = F P_bb Fᵀ, P stays symmetric PSD (min eig ≥ −1e-12).
  - `block_predict_new_source`: new_source = true ⇒ x_b = 0, cross blocks 0, P_bb = Σ_now.
  - `damped_lambda_zero_is_joint_step`: x̃ + δ at λ = 0 equals `joint_step` (1e-12) on the existing 2-channel fixture.
  - `damped_zero_variance_latent`: a latent with P̄ row/column 0 has δ = 0 for λ ∈ {0, 1e-3, 10}.
  - `damped_large_lambda_freezes_theta`: λ = 1e8 ⇒ |δ_θ| < 1e-6 while δ_b equals the b-only solve at fixed θ (1e-6).
  - `prior_energy_identity`: on an invertible P̄, ½yᵀ(x_new − x̄) = ½(x_new − x̄)ᵀP̄⁻¹(x_new − x̄) (1e-9), λ ∈ {0, 0.1}.
- [ ] **Step 2:** `cmake -B build && make -C build -j8 persistent_update_test` in `countable_concept/tests`; run it. Expected: FAIL (undefined `predict_block`).
- [ ] **Step 3:** Implement the three functions; remove `predict_latents` (its only caller is `CountableBelief::update_persistent`, which Task 13 rewires — until then call `predict_block` there with F = I per block so the build stays green; that interim drops D, which is the intended behaviour).
- [ ] **Step 4:** Run `persistent_update_test`, `persistent_belief_test`, `lidar_golden_test`. Expected: PASS (persistent_belief_test checks that encoded D, if any, are rewritten to the no-rescale expectation and named in the report).
- [ ] **Step 5:** Commit `persistent update: block predict without rescale, damped joint step (spec 7c.7 a, d)`.

### Task 12: Pose covariance in the room, the SE(2) chain correction and the lever (countable, pure)

**Files:** Modify `countable_concept/src/countable_viewpoint.h/.cpp`; Test `countable_concept/tests/viewpoint_rho_test.cpp`.

**Interfaces:**
- Consumes: `rc::se2_inverse_jacobian` / `invert_se2_cov` (common/place_memory/place_map.h).
- Produces, in `rc`:
  - `Eigen::Matrix3f pose_cov_in_room(std::span<const float> rt_cov36, const Eigen::Vector3f& edge_pose)` — Σ_edge from slots {0, 1, 5} (rows/cols of the 6×6 row-major block), returns J Σ_edge Jᵀ; an empty span ⇒ zero matrix.
  - `ChainTracker` on SE(2): `ref`, `prev`, `odo` become `Eigen::Vector3f` (x, y, ψ); `void chain_observe(ChainTracker&, const Eigen::Vector3f& pose, std::optional<Eigen::Vector3f> odo_step)`; `Eigen::Vector3f chain_correction(const ChainTracker&)` (ψ wrapped to (−π, π]); `chain_rebase` unchanged in meaning.
  - `Eigen::Vector3f odometry_step_room_se2(float v_adv, float v_side, float omega, float yaw_room, float dt_s)` — (x, y) as `odometry_step_room`, ψ = ω·dt.
  - `float rho_chain(const Eigen::Vector3f& c, const Eigen::Matrix3f& Sigma_pose)` = exp(−½ cᵀ Σ⁻¹ c) on the non-zero subspace (pseudo-inverse via `Eigen::CompleteOrthogonalDecomposition`); a component with zero variance and non-zero c ⇒ 0; c = 0 ⇒ 1. The 2-D overload is removed.
  - `Eigen::Matrix3d chain_transport(const Eigen::Vector2f& dr)` = F_t.
  - `struct ChainLever { Eigen::Vector2f l = Eigen::Vector2f::Zero(); bool set = false; };` `void lever_start(ChainLever&, const Eigen::Vector2f& seed_c, const Eigen::Vector2f& robot_r)`; `void lever_step(ChainLever&, const Eigen::Vector2f& dr, float rho_chain, const Eigen::Vector2f& belief_c, const Eigen::Vector2f& robot_r)` = ρ(ℓ − Δr) + (1 − ρ)(c − r).
  - `Eigen::Matrix<double, 6, 3> chain_B(const Eigen::Vector2f& l)`.

- [ ] **Step 1: Write the failing tests** in `viewpoint_rho_test.cpp`:
  - `pose_cov_round_trip`: Σ_pose = diag(0.034², 0.0435², 0.024²) with a 0.3 cross term on (x, ψ), robot pose in room (0.83, 2.24, 0.42); the edge pose = its SE(2) inverse; Σ_edge = J(robot pose) Σ_pose Jᵀ written into a 36-slot block (slots 0, 1, 5); `pose_cov_in_room` returns Σ_pose (1e-6 relative).
  - `pose_cov_absent_is_zero`: empty span ⇒ zero.
  - `chain_correction_se2`: odometry (0.1, 0, 0.05) per sample ×10, poses that drift by (0.01, −0.02, 0.003) beyond it ⇒ c = (0.01, −0.02, 0.003) (1e-6); a nullopt stretch is absorbed on all three components.
  - `chain_correction_wraps_psi`: ref ψ = 3.10, pose ψ = −3.10, odometry 0 ⇒ c_ψ = 2π − 6.20 (1e-6).
  - `rho_chain_se2`: c = (0, 0, 0.024), Σ = diag(0.034², 0.0435², 0.024²) ⇒ exp(−0.5) (1e-6); zero Σ_ψψ with c_ψ ≠ 0 ⇒ 0.
  - `lever_transport_invariance`: ℓ₀ = (0.09, 1.65), Δr = (0.3, −0.2), ρ = 1: chain_B(ℓ₁)·F(Δr) restricted to the cx, cy, yaw rows equals chain_B(ℓ₀) (1e-9).
  - `lever_relaxes_with_rho`: ρ = 0 ⇒ ℓ = c − r exactly; ρ = 0.5 ⇒ the midpoint.
- [ ] **Step 2:** Build and run `viewpoint_rho_test`. Expected: FAIL (undefined symbols).
- [ ] **Step 3:** Implement; update the existing 2-D callers in `countable_fitter.cpp` (`integrate_odometry`, the ρ_chain line ~235) to the SE(2) forms using `robot_yaw_now_` and the robot node's `robot_current_angular_speed` already read by `read_odometry_and_pose`, and keep `inst.chain_cov_xx/yy` as the Σ until Task 13 replaces it.
- [ ] **Step 4:** Run `viewpoint_rho_test`, `persistent_belief_test`, `lidar_golden_test`. Expected: PASS.
- [ ] **Step 5:** Commit `countable viewpoint: pose covariance in the room, SE(2) chain correction, fixed lever (spec 7c.7 b)`.

### Task 13: e in the belief and the fitter (countable, flag on)

**Files:** Modify `countable_concept/src/countable_belief.h/.cpp`, `countable_fitter.cpp/.h`, `countable_instance.h`; Test `countable_concept/tests/persistent_belief_test.cpp`, new `countable_concept/tests/bchain_jump_test.cpp` (cherry-pick f5b5496f from `wip/bchain-jump`, then restate), `tests/CMakeLists.txt`.

**Interfaces:**
- Consumes: Task 11's `predict_block`; Task 12's `pose_cov_in_room`, `chain_transport`, `ChainLever`, `lever_start`, `lever_step`, `chain_B`, SE(2) `rho_chain`.
- Produces:
  - `CountableBelief`: K = 15, `kChain = 0` (3), `kCam = 3`, `kL = 9`; alias `MatJd` (21×21) replaces `Mat20d` (update every user, tests included); `VecKd` 15.
  - `CountableFrame` replaces `rho_chain` (array 2) and `sig2_chain` with `float rho_chain`, `Eigen::Matrix3f sigma_pose`, `Eigen::Vector2f chain_dr` (room displacement since this instance's last update), `Eigen::Vector2f robot_xy` (at the capture stamp).
  - The belief stores `ChainLever lever_` and `Eigen::Matrix3f last_sigma_pose_`; `const ChainLever& chain_lever() const`.
  - ai2_log columns, appended after `Itr_L`: `e_x,e_y,e_psi,se_x,se_y,se_psi,lever_x,lever_y,spose_x,spose_y,spose_psi,c_x,c_y,c_psi` (the `bchain_x,bchain_y` columns are removed; update `countable_concept/tools/lidar_measures_score.py` to read `e_x,e_y` where it read them, tolerating both).

- [ ] **Step 1: Write the failing tests.**
  - `bchain_jump_test.cpp` (restated per spec 7c.7 Pass): `linear_scale_coupling` — live inputs (belief centre (0.92, 3.75) on update 1, (0.89, 3.90) on update 2, robot (0.83, 2.24), Σ_pose diag(0.034², 0.0435², 0.024²)): result equals the run whose belief centre does not move (1e-9), cx = 0.9385 (1 mm), |e| ≤ 0.1σ per component; `lever_arm_sensitivity` (ray-cast): e and θ equal the control seeded at the truth within σ_ψ·|seed − truth|.
  - `persistent_belief_test`: `parked_range_jitter_equality` (1000 identical frames, ρ = 1, mask range ±10⁻⁴ relative vs no jitter: θ, b, P equal at float tolerance); `transport_invariance_arc` (ray-cast nightstand, robot dead-reckoned with true δψ = 0.02 rad, 90° arc at 1.5 m, ρ_chain = 1: B_t F_t = B_{t−1} to 1e-9 every update; √diag Σ_e never below its value after update 1 (0.1 %); NEES of e in [0.1, 10]; θ error within its own 2σ); `rho0_regression_zero_pose_cov` (Σ_pose = 0: the persistent path equals the flag-off path without chain covariance at every iterate on camera-only and LiDAR-only frames, fixtures with no rejected step); `chain_once_for_both_3x3` (parked camera + LiDAR: data information on (cx, cy, yaw) = (B Σ_pose Bᵀ + (Σ_cam⁻¹ + Σ_L⁻¹)⁻¹)⁻¹, 1e-6 relative); `zero_pose_cov_runs`; `relocalisation_redraws_e` (ρ_chain = 0 ⇒ e mean 0, Σ_e = Σ_pose, ℓ = c − r, θ unmoved by the predict); `sanitize_reseeds_e`; `mode_rotation_leaves_e`.
- [ ] **Step 2:** Build `bchain_jump_test persistent_belief_test` (-j8); run. Expected: FAIL.
- [ ] **Step 3:** Implement. In `update_persistent`: predict e with `predict_block(F = chain_transport(f.chain_dr), ρ = f.rho_chain, Σ = f.sigma_pose, new_source = not lever_.set)`; on a new source `lever_start(seed centre = state_ (cx, cy), f.robot_xy)`, else `lever_step(f.chain_dr, f.rho_chain, state_ centre, f.robot_xy)` before the predict; B_cam, B_L get `chain_B(lever_.l)` in their e columns. In the fitter: Σ_pose from the robot→room edge's `rt_covariance` entry at the mask capture stamp (the history index matching the edge's `rt_time_stamps`; edge pose = the RT translation/rotation at that index) via `pose_cov_in_room`; `chain_dr` from the odometry `integrate_odometry` walks; `robot_xy` at the capture stamp; ρ_chain from Task 12 with Σ_pose. Remove the D-era `sig2_chain` / `chain_cov_xx` use on the flag-on path (flag-off keeps `compute_chain_cov`). Spec §7c.7 b bookkeeping: sanitize re-seeds e at `last_sigma_pose_`, canonicalize / mode rotation leave e and ℓ, LTSM restore stores neither.
- [ ] **Step 4:** Run every countable test (`ctest` in the tests build) and `lidar_golden_test`. Expected: PASS; flag-off golden bit-exact.
- [ ] **Step 5:** Commit `countable belief: e = robot pose error with a fixed lever (spec 7c.7 b, c)`.

### Task 14: The update converges before P is taken (countable, flag on)

**Files:** Modify `countable_concept/src/countable_belief.h/.cpp`, `countable_fitter.cpp` (log); Test `countable_concept/tests/persistent_belief_test.cpp`.

**Interfaces:**
- Consumes: Task 11's `joint_step_damped`, `prior_energy`.
- Produces: `CountableBeliefParams::gn_iters_max = 12` (persistent path only; no config key); `double CountableBelief::persistent_data_energy(const CountableBeliefState& s_cam, const CountableBeliefState& s_L, const CountableBeliefState& s_th, const CountableFrame& f) const` — the sum of the energies whose gradients `accumulate_channels` builds (camera, LiDAR, θ-only), each term's own energy at its own linearisation point; fresh-covariance marginalisation is NOT in the energy (stated in the header: Σf is ego-motion, zero parked). Belief accessors `int last_gn_iters() const`, `bool last_gn_converged() const`, `double last_lambda() const`; ai2_log columns `gn_it,gn_conv,gn_lambda` appended after Task 13's.

- [ ] **Step 1: Write the failing tests** in `persistent_belief_test.cpp`:
  - `energy_gradient_consistent`: on the nightstand fixture, the central finite difference of `persistent_data_energy` in each θ DOF equals −(bd_cam + bd_L + bd_th) within 1 % (or 1e-6 absolute).
  - `first_update_converges`: seed = birth on priors with the class mean 6 cm off: `last_gn_converged()` = true within 12; J non-increasing over accepted steps; θ equals the same solver run for 200 steps (1 mm, 1 mrad) and √diag P within 5 % per component.
  - `parked_second_update_still`: the next identical frame moves θ less than 1 mm / 1 mrad.
  - `all_steps_rejected_is_finite`: an energy made non-decreasing (test hook that rejects every step) ⇒ θ = θ₀, P finite and PSD, `last_gn_converged()` = false.
  - `cycle_cost`: time 100 updates of the nightstand fixture; print the mean ms and implied Hz (no assertion; the number goes into the report).
- [ ] **Step 2:** Build and run. Expected: FAIL.
- [ ] **Step 3:** Replace the fixed `gn_iters` loop in `update_persistent` with the spec 7c.7 d loop: build at θ_k (b restarted at b̄), `joint_step_damped(λ)`, constraints, J = `persistent_data_energy` at the candidate + `prior_energy`; accept ⇒ λ/4, else 4λ and re-solve without re-building; stop at the tolerance or 12 steps; P from `joint_cov` at the returned θ plus δ_last δ_lastᵀ on θθ. If J stalls above the tolerance on the fixture, switch to the consistent variant (x̃_k = (θ_k, b_k) linearised at θ_k + B b_k) and say so in the report.
- [ ] **Step 4:** Run every countable test and `lidar_golden_test`. Expected: PASS; report the `cycle_cost` Hz.
- [ ] **Step 5:** Commit `countable belief: damped GN converges before P is taken (spec 7c.7 d)`.

### Task 10 again (controller, live): objects_live10, objects_live10off, objects_live10shift

As Task 10 Steps 1–4 (claim Webots from room and chat; rebuild nightstand/bed/toilet; the fleet from main), plus: no
instance's |e| exceeds 3σ of Σ_pose; √diag Σ_pose,t while parked against `proto_room.csv`'s pose σ at the same stamps
(the C3 live check, 1 %); median `gn_it` and the share of `gn_conv` = 0; fit rate ≥ 8 Hz. Results in
`docs/superpowers/results/2026-11-18-lidar-measures-live10.md`.

## Revision 5 (11-20): spec §7c.8 rev 3.3.3 — a miss is evidence: the persistent ray likelihood is a smooth beam model

**Spec:** §7c.8 rev 3.3.3 (owner "approved" 11-20, i.e. "Yes" to the rev 3.3.3 read). Reviews behind it:
`fable-7c8-review.md`, `fable-7c8-rereview.md`, `fable-7c8-rereview2.md`. Executors read §7c.8 in full: its Pass list
is the test list below, and every number a test asserts is copied from it.

**Goal:** On the persistent path, every return whose ray passes the select box and has a camera label enters a smooth
beam likelihood (surface offset marginalised, ownership as a mixture), so a miss is evidence, the energy is continuous,
GN converges with rays, and cx / w are observed by silhouette edges.

**Architecture:** Three layers. (1) Shared, pure, double: `common/ai_belief/beam_ray.h` — the 1-D likelihood
(Student-t, Φ / Φ⁻¹, conditional-quantile quadrature, panels) and the ray geometry on ORIENTED-BOX PARTS (closest
approach d and ∇d in closed form, level crossings by one monotone march per part), the energy, the exact gradient and
the two-part curvature; plus a new ray-set type and a new staging function in `common/lidar_select`. (2) countable:
the model exposes its convex parts as oriented boxes with analytic Jacobians; a persistent label returning an
optional; config `Model.MaskPixelErrorRate`. (3) The persistent belief and the fitter use them; flag off and every
other caller untouched.

**Decision (controller's call): parts are oriented boxes.** Every convex part of a countable model is an oriented box
(the main box; the bed's headboard panel). The shared geometry takes `struct BeamPart { Eigen::Vector3d c; double yaw;
Eigen::Vector3d half; Eigen::Matrix<double, 7, 6> J; }` (J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ) and never calls a
model SDF, so every Jacobian is analytic and no `fd_eps` / `jac_slope_clamp` acts on this path (spec (a) "Convex
parts"). The headboard's thickness min(t_hb, h) contributes ∂/∂h only where t_hb > h (one-sided, stated in the hook).

### Global Constraints (revision 5, in addition to the above)

- Persistent path only. `LidarRays`, `select_into`, `select_into_weighted`, `make_membership`, every function in
  `lidar_ray_factor.h` and `lidar_select.h` that exists today: NOT edited (new functions and types only). Flag off ⇒
  `lidar_golden_test` bit-exact; door and hood tests that include `lidar_ray_factor.h` pass unchanged.
- Constants (spec (c)): K from the accuracy test (start K = 8); ε of the one-sided tangents and the 1e-4 m bracket as
  7c.6; the 8σ_s numerical floor (π_i < 7e-16), flagged as machine-precision, not a model gate. NO other constant.
  New model constant, flagged: ε_px = `Model.MaskPixelErrorRate`, per class, measured (Task 15).
- Config validations refused at load with a message: p₀c² > 1 (p₀ from `Model.LidarPrecision` and the SensorModel
  σ_sensor: 1/p₀ = 1/LidarPrecision + σ_sensor²; c = `Model.LidarRobustCM`); 0 < ε_px < ½.
- ρ_max = `max_range_m` (6 m); p_pass = 1/ρ_max; σ_s = `Model.ModelSigmaM`; ν = p₀c² − 1; scale² = c²/ν.
- Label m_i = ε_px + (1 − 2ε_px)·π_mask·Φ(s_i/σ_px); κ from the UNFLOORED q_i = π_mask·Φ(s_i/σ_px) on the
  endpoint-in-box subset, exactly today's two down-weights, handed over as one number κ = coverage·ca ∈ [0, 1].
- "The LiDAR channel measured" ⇔ κ > 0 and the staged set is not empty (persistent path: `frame_has_lidar`,
  `rays_usable`, the b_L predict, `record_channels`, `lidar_only`).
- Everything numerical in double; Φ via erfc; Φ⁻¹ accurate to 1e-15 (AS241).
- Do not patch symptoms: a failing Pass item is reported with its numbers (the gap fixture, the flush neighbour and the
  5 cm background go to the owner, not tuned).

### Review Focus (revision 5)

1. **Sensor inside the dilated part** (t = 0 crossing; robot brushing the bed): finite E, gradient and curvature, the
   formula refutes the model by itself, no special case (Task 17 `sensor_inside_part_is_finite`).
2. **Ray parallel to a face / exactly through an edge** (non-unique minimiser, d_i = 0): no NaN, gradient matches the
   one-sided differences (Task 17 `parallel_ray_is_finite`).
3. **No return in range** (ρ_i > ρ_max or the sweep's max-range sentinel): staged as ρ_max, E finite (Task 18
   `no_return_staged_at_rho_max`).
4. **`sl.confidence` = 1.0f / π_mask = 1** and **κ = 0**: finite energy; κ = 0 leaves the belief at the prediction
   (Task 18 `label_bounds`, Task 19 `kappa_zero_is_prediction`).
5. **Bed, a level where the first-hit part switches or two parts tie**: continuous energy, C¹ off the stated kinks
   (Task 17 `headboard_continuity`, `headboard_gradient`).

### Task 15 (controller, before building): the three numbers the spec asks for first

- [ ] **Step 1:** In live10's run folders (`room_concept/estimation_scripts/crossing/objects_live10*`), count per
  instance the persistent-path cycles with rays and no membership (no RGB frame staged / camera unbound / no focal /
  empty slice) and the longest consecutive run; if the log cannot tell, say so and add the column in Task 20. A
  systematic cause (pose lookup failing while turning) is reported before Task 20 wires the skip.
- [ ] **Step 2:** Confirm whether live10 recorded ground-truth masks. If yes, measure ε_px per class as spec (a0) says
  (over LiDAR returns if Webots ground truth allows, pixels otherwise; say which); if no, a short recorded run with the
  Webots segmentation camera (claim Webots: CLAIM → ACK → FREE with room and chat). Compute b*(ε_px) from
  ε/(1 − ε)·St(b*)/p_pass + G(−b*) = 1 at the configured p₀, c.
- [ ] **Step 3:** Write both numbers to `.superpowers/sdd/2026-11-15-lidar-measures/eps-px-measurement.md` and the
  ledger; they go to the owner before live11 (not before building). If ε_px < 0.5 %, record that Camera coverage (ii)
  decides and the value is NOT raised.

### Task 16: The beam likelihood in one dimension (shared, pure)

**Files:** Create `common/ai_belief/beam_ray.h` (math part); Test `countable_concept/tests/beam_ray_test.cpp` (new
target, header-only, in `tests/CMakeLists.txt`).

**Interfaces — Produces (namespace `rc::ai::beam`):**
- `struct StudentT { double p0, c, nu, scale2; static std::optional<StudentT> make(double p0, double c); double pdf(double e) const; double cdf(double e) const; double dlogpdf(double e) const; }` — nullopt when p₀c² ≤ 1. `cdf` exact or tabulated at load; if tabulated, its derivative is the table's own and |cdf′ − pdf| < 1e-6.
- `double Phi(double x)`, `double Phi_upper(double x)` (erfc), `double Phi_inv(double p)` (AS241).
- `struct GaussLegendre01 { std::vector<double> u, w; static GaussLegendre01 make(int K); }`.
- `struct RayLik { double E; double r_member; double o_front; double S_bar, G_bar, t_bar, var_t; }` — the per-ray result (E per spec "Energy"; r_i; o_i; Ŝ, Ḡ, t̄, Var t under the quadrature weights).
- `RayLik ray_likelihood(const StudentT& st, double rho, double m, double pi, double rho_max, std::span<const double> t_nodes, std::span<const double> w_nodes, std::span<double> dE_dt, double* dE_dpi)` — weights sum to 1 (conditional mean given a hit), p(ρ | θ) = (1 − π)(1 − m)p_pass + π·Σ_k w_k[m·St(t_k − ρ) + (1 − m)p_pass·G(t_k − ρ)], E = −log[p/((1 − m)p_pass)]; outputs ∂E/∂t_k and ∂E/∂π for Task 17's chain rule.
- `void quantile_nodes(double d, double sigma_s, std::span<const double> b, const GaussLegendre01& gl, std::vector<double>& delta, std::vector<double>& w)` — panels b_0 = 0 < … < b_n = π (upper-tail probabilities), δ_jk = −σ_s Φ⁻¹(b_{j−1} + (b_j − b_{j−1})u_k), weight (b_j − b_{j−1})/π · ω_k.

- [ ] **Step 1: Write the failing tests** in `beam_ray_test.cpp`:
  - `student_t_constants`: p₀ = 2500, c = 0.05 ⇒ ν = 5.25 (1e-12), pdf(0) ≈ 17.5 m⁻¹ (1e-2); −log pdf(e) − (−log pdf(0)) = ½p₀c²·log(1 + e²/c²) (1e-12); `make(2500, 0.02)` (p₀c² = 1) = nullopt; ∫pdf over [−1, 1] m = cdf(1) − cdf(−1) (1e-9); |cdf′ − pdf| < 1e-6 on a 1e-4 m grid over [−0.5, 0.5].
  - `phi_inv_accuracy`: Phi(Phi_inv(p)) = p, relative 1e-14, for p ∈ {1e-300, 1e-16, 1e-6, 0.3, 0.5, 1 − 1e-12}.
  - `nodes_fixed_deep_inside`: d = −10σ_s ⇒ the δ nodes equal those at d = −20σ_s to 1e-12 (no tail cancellation).
  - `face_numbers` (spec "Numbers"): one face-on node set, t_k = ρ + δ_k (n̂·u = 1), π = 1 − 1e-12: E at e = 0 within 0.1 nats of −6.85 at m = 0.9 and of −4.66 at m = 0.5; π = 0 ⇒ E = 0 exactly; m = 0.05, return 0.40 m behind a fully blocking model ⇒ E within 1 nat of +11; return 0.30 m in front, m = 0.9 ⇒ E ≤ −log(1 − 0.9) + 1e-9.
  - `dE_matches_fd`: ∂E/∂t_k and ∂E/∂π against central differences (1e-6 relative or 1e-9 absolute) at m ∈ {0.05, 0.9}, e ∈ {−0.3, 0, 0.1}.
  - `panels_sum_to_one`: n = 3 panels ⇒ Σw = 1 (1e-14); n = 1 panel ⇒ the one-part node set exactly.
- [ ] **Step 2:** Build `beam_ray_test` (-j8 after `pgrep -a cc1plus`); run. Expected: FAIL (header missing).
- [ ] **Step 3:** Implement the math part of `beam_ray.h` with the interfaces above.
- [ ] **Step 4:** Run `beam_ray_test`. Expected: PASS.
- [ ] **Step 5:** Commit `beam_ray: the 1-D smooth beam likelihood and its quadrature (spec 7c.8 a)`.

### Task 17: Ray geometry on oriented-box parts, energy, gradient, curvature (shared, pure)

**Files:** Modify `common/ai_belief/beam_ray.h` (geometry + assembly part); Test `beam_ray_test.cpp`; header note in
`common/ai_belief/lidar_ray_factor.h` (comment only: "a ray that misses the model carries no term — except on the
countable persistent path, `beam_ray.h`", spec (d) R6-M8).

**Interfaces:**
- Consumes: Task 16.
- Produces:
  - `struct BeamPart` (as in the Decision above).
  - `struct BeamRays { Eigen::Vector3d origin; std::vector<Eigen::Vector3d> endpoints; std::vector<double> m; double kappa = 0; double rho_max = 6.0; double sigma_s = 0.01; }` — the persistent ray-set (ranges ≥ ρ_max already clamped by staging).
  - `struct BeamConfig { StudentT st; GaussLegendre01 gl; double eps_tangent = 1e-4, bracket_m = 1e-4; }`.
  - `double closest_approach(const BeamPart&, const Eigen::Vector3d& o, const Eigen::Vector3d& u, double rho_max, Eigen::Matrix<double, 6, 1>* grad)` — d_ip and ∇_θ d_ip: closed-form segment–box distance for a miss; for a crossing the two-tangent crease d = intersection of the tangents at t* ∓ ε, ∇d = (s₊J₋ + s₋J₊)/(s₋ + s₊); one-sided at t = 0 / ρ_max.
  - `template <int N> void accumulate_beam_rays(std::span<const BeamPart> parts, const BeamRays&, const BeamConfig&, Eigen::Matrix<double, N, N>& W, Eigen::Matrix<double, N, 1>& g, double* E)` with N = 6 — κ·Σ E_i, its exact gradient (through π_i, node levels, panel limits, crossings: ∂t/∂θ = −(∂s/∂θ)/(∂s/∂t), ∂t/∂δ = 1/(∂s/∂t)), and the PSD curvature κ·Σ[r_i J̄J̄ᵀ/V_i + (1 − o_i) I_d,i ∇d∇dᵀ] with V_i = (1 + ē²/c²)/p₀ + Var t and I_d,i = (φ(d_i/σ_s)/σ_s)²·[m_i/π_i + (1 − m_i)(1 − t̄_i/ρ_max)/(1 − π_i)] (Mills ratio in the tails); cross term dropped. Rays with d_i > 8σ_s stop after the closest approach and contribute exactly 0.
  - `double beam_rays_energy(std::span<const BeamPart>, const BeamRays&, const BeamConfig&)` — the same E, no derivatives.
  - Cost savings are part of the task (spec c): one monotone march per part collecting all K·panels levels in t order; cos/sin of yaw once per part per build.

- [ ] **Step 1: Write the failing tests** (ray-cast true scenes from `test_scene.h`; m by hand: 0.9 on, ε_px = 0.02 off):
  - `closest_approach_corner`: rays 0.05–1 mm outside a corner: d equals the exact distance (1e-9), ∇d against central differences (1 %); rays cutting a corner at n̂·u far from 45°: ∇d against one-sided differences on each side (1 %).
  - `continuity_box`: cx swept 2 cm across a ray edge in 0.05 mm steps: |E(x+h) − 2E(x) + E(x−h)| < 0.01 nats at every step.
  - `headboard_continuity`: bed with headboard, rings at elevations −1°…+1° relative to the mattress top, swept across a first-hit switch and across the appearance of the headboard's own minimum: the same 0.01 bound.
  - `gradient_box` and `headboard_gradient`: analytic vs central differences (h = 0.1 mm), 1 % or 1e-3 nats/m, every step; where some d_i changes sign, two parts tie, or a panel appears inside the stencil: vs the one-sided difference on each side.
  - `quadrature_K`: |E_K − E_64| < 0.01 nats per ray over n̂·u ∈ [0.1, 1], e ∈ [−0.3, 0.3] m, d ∈ [−4σ_s, 4σ_s], m ∈ {0.9, 0.05}, box and headboard; on a face E_K decreasing toward its minimum from each side (minimum at e > 0: 7 µm at m = 0.9, 1.3 mm at m = 0.05, printed). Print the smallest passing K; `BeamConfig` default K = that value.
  - `several_parts_equals_one`: the panel code on a one-part model equals the one-part code to 1e-12 per ray, energy and gradient.
  - `far_rays_exact_zero`: d_i > 8σ_s ⇒ contribution exactly 0.0.
  - `reduction_to_7c6`: rays deep in a face (π > 1 − 1e-6), m = 1 − 1e-6, κ = 1, n̂·u ∈ {1, 0.25}: curvature = a aᵀ/((n̂·u)²·R_range + σ_s²) within 1 %.
  - `sensor_inside_part_is_finite`, `parallel_ray_is_finite`: finite E, g, W; W PSD.
  - `largest_curvature_printed`: for a claimed ray 1.5–3σ_s outside a corner, print the largest per-ray curvature along ∇d (spec R6-M3; no bound).
  - `cost`: 256 near rays, box and 2-part bed: ms per build and per energy, printed (the median ≤ 5 ms check is Task 19's on the belief).
- [ ] **Step 2:** Build and run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `beam_ray_test`, `lidar_golden_test`, and the door and hood tests that include `lidar_ray_factor.h` (`grep -rl lidar_ray_factor.h --include=CMakeLists.txt` finds their targets). Expected: PASS, golden bit-exact.
- [ ] **Step 5:** Commit `beam_ray: oriented-box parts, closest approach, smooth energy, gradient and curvature (spec 7c.8 a, b)`.

### Task 18: Ray staging, the persistent label, config (shared + countable)

**Files:** Modify `common/lidar_select/lidar_select.h` (new function only); `countable_concept/src/countable_mask_membership.h/.cpp` (new function only); `countable_config.h/.cpp`; `etc/config.toml` of nightstand, bed, toilet (`Model.MaskPixelErrorRate` = Task 15's value); Test `countable_concept/tests/beam_staging_test.cpp` (new), `mask_membership_test.cpp`, `config_migration_test.cpp`.

**Interfaces:**
- Consumes: Task 17's `BeamRays`.
- Produces:
  - `struct PersistentLabel { double m; float q; };` and `std::function<std::optional<PersistentLabel>(const Eigen::Vector3f&)> make_persistent_label(Projector, const MaskMembership&, float f_px, float sigma_range_m, float sigma_chain_m, float mot_pos_var_m2, float pi_mask, double eps_px)` — nullopt when the projector returns nullopt (behind / outside the image); q = π_mask·Φ(s/σ_px) (today's value), m = ε_px + (1 − 2ε_px)q.
  - `int rc::lidar_select::select_rays_beam(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin, const Eigen::Vector2f& centroid_xy, const Box& box, const Weights& w, double rho_max, const std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>& label, rc::ai::beam::BeamRays& out, BeamStageStats* stats)` — stages every return whose ray (origin → return → ρ_max) intersects the box's cylinder ∩ z-band, with a label (nullopt ⇒ not staged, counted); ρ_i > ρ_max staged at ρ_max; κ = coverage·ca computed exactly as `select_into_weighted` does but over the endpoint-in-box returns with weight q; returns the staged count.
  - `struct BeamStageStats { int staged, no_label, in_box; float kappa, q_sum; };`
  - `CountableConfig::mask_pixel_error_rate` (`Model.MaskPixelErrorRate`, no default when `Model.LidarMeasuresOnGated` = true: missing ⇒ refused); load refuses 0 < ε_px < ½ violations and p₀c² ≤ 1 (with SensorModel σ_sensor when present) with a message.

- [ ] **Step 1: Write the failing tests:**
  - `beam_staging_test`: `ray_through_box_staged` (a return 1 m behind the box, ray through it ⇒ staged; a return whose ray misses the cylinder ⇒ not); `short_return_staged` (return in front of the box on a ray through it ⇒ staged); `no_return_staged_at_rho_max` (endpoint at 8 m ⇒ staged at ρ_max = 6, E finite through `beam_rays_energy`); `no_label_not_staged` (label nullopt ⇒ not staged, `no_label` counted); `kappa_unfloored` (all returns q = 0 with m = ε_px ⇒ κ = 0; the same returns with today's labels ⇒ κ equals `select_into_weighted`'s precision/base on the same sweep, 1e-6).
  - `mask_membership_test`: `persistent_label_bounds` (m ∈ [ε_px, 1 − ε_px]; π_mask = 1.0f ⇒ m ≤ 1 − ε_px, finite; outside image ⇒ nullopt; `make_membership` unchanged on the same inputs, bit-exact).
  - `config_migration_test`: ε_px = 0, 0.5, missing (flag on) ⇒ refused; LidarPrecision 300 with c = 0.05 (p₀c² = 0.75) ⇒ refused.
- [ ] **Step 2:** Build and run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the three tests and `lidar_weights_test`, `lidar_golden_test`. Expected: PASS.
- [ ] **Step 5:** Commit `beam staging: rays through the select box, the persistent label with ε_px, config checks (spec 7c.8 a0)`.

### Task 19: The persistent belief uses the beam model (countable, flag on)

**Files:** Modify `countable_concept/src/countable_belief.h/.cpp`; Test `persistent_belief_test.cpp`, new `countable_concept/tests/beam_belief_test.cpp`, `creep_set_test.cpp`, `lidar_only_update_test.cpp` (the `over_deep_belief_is_not_held` expectation).

**Interfaces:**
- Consumes: Task 17's `BeamPart`, `BeamRays`, `BeamConfig`, `accumulate_beam_rays<6>`, `beam_rays_energy`.
- Produces:
  - `int CountableBelief::n_beam_parts() const` (1, or 2 with a headboard); `rc::ai::beam::BeamPart CountableBelief::beam_part(const CountableBeliefState&, int k) const` (analytic J; main box z ∈ [0, H]; headboard as `sdf_headboard`).
  - `CountableFrame::beam` (`rc::ai::beam::BeamRays`), read only when `params.persistent`; `CountableBeliefParams::beam` (`BeamConfig`, set by the fitter).
  - On the persistent path `accumulate_channels` and `persistent_data_energy` use the beam terms on `f.beam` in place of the first-crossing ray-sets (the flag-off path and `f.lidar` untouched); `frame_has_lidar`, `rays_usable`, the b_L predict and `record_channels` read the measured rule (κ > 0 and staged non-empty).
  - Accessors for the log: `int last_beam_staged() const`, `int last_gap_rays() const` (rays with π_i > 0.99 and e < −5c).

- [ ] **Step 1: Write the failing tests** in `beam_belief_test.cpp` (ray-cast true scenes, hand labels 0.9 on / ε_px off unless stated; "tolerance" = 1 mm / 1 mrad):
  - `kappa_zero_is_prediction`: κ = 0 ⇒ state, Σ, ω at the prediction; the channel's odometry path not restarted. Σq = 5 (κ ≈ 0.08): update finite, per-ray hit/miss energy difference = κ × the κ = 1 value.
  - `reduction_update`: the Task 17 reduction rays through a belief update: θ equals 7c.7's update within 0.1σ per component, √diag P within 3 % (difference printed).
  - `occluder_in_front`: occluder 30 cm in front, m = 0.05: θ moves less than the tolerance AND every √diag P component within 1 % of the run with those rays unstaged (κ unchanged). One silhouette occluded: σ_w, σ_cx not more than 1 % below the run without the occluded rays.
  - `wall_2cm_printed`: over-segmented mask on a wall 2 cm behind the face, m = 0.9 and 0.05: depth printed.
  - `depth_basin`: camera + rays, seed 30 cm behind the surface, m = 0.9: depth within its own σ of truth in ≤ 10 frames; summed ray gradient along the view axis never points away from the returns; rays-only run printed.
  - `silhouette_hand_labels`: front-only sweep, seed 3 cm too narrow and 3 cm too wide, backgrounds 20 cm and 1 m: w within σ_w of truth and σ_w below the prior's in 10 frames; 5 cm background printed; largest per-ray curvature printed.
  - `parked_floors`: 1000 identical sweeps: σ_w, σ_cx not below b_L's floors; w moves < 1 mm after frame 10; θ step sizes non-increasing and below the tolerance at frame 10.
  - `first_update_converges_with_rays`: the camera + ray fixture's first update returns `gn_conv` = 1.
  - In `bchain_jump_test`: `lever_arm_sensitivity` as a CHECK: θ's position equals the truth-seeded control within σ_ψ·|seed − truth|.
  - `persistent_belief_test`: the ρ = 0 regression restricted to CAMERA-ONLY frames (frames with rays differ by design; say so in the test).
  - `lidar_only_update_test`: re-run `over_deep_belief_is_not_held`, print its result, rewrite its expectation from that number (the commit message quotes it).
  - `creep_set_test`: 0/16 unchanged.
  - `beam_cost`: one update's ray work at 256 near rays, first update and parked, box and bed: median ms printed and ≤ 5 ms.
- [ ] **Step 2:** Build and run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run every countable test (`/usr/bin/ctest` in the tests build). Expected: PASS; `lidar_golden_test` bit-exact.
- [ ] **Step 5:** Commit `countable belief: the smooth beam model on the persistent path (spec 7c.8 a, b)`.

### Task 20: Fitter wiring, logs, and the live-staging fixtures (countable, flag on)

**Files:** Modify `countable_concept/src/countable_lidar_range_channel.h/.cpp` (new `feed_beam`), `countable_fitter.cpp/.h`, `countable_instance.h` (log fields); `countable_concept/tools/lidar_measures_score.py` (new columns); Test new `countable_concept/tests/beam_live_staging_test.cpp`, `countable_concept/tests/test_scene.h` (a rendered-mask helper).

**Interfaces:**
- Consumes: Task 18's `select_rays_beam`, `make_persistent_label`, `BeamStageStats`; Task 19's `CountableFrame::beam`, accessors.
- Produces:
  - `void CountableLidarRangeChannel::feed_beam(CountableInstance&, CountableFrame&, std::span<const Eigen::Vector3f> anchor_cloud, const std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>& label) const` — helios only (bpearl not staged on this path); the probe / `dbg_lidar_rays` / coverage instruments keep reading the endpoint-in-box subset.
  - Fitter, flag on: a cycle without membership (no RGB frame staged, camera unbound, no focal, empty slice) stages no rays, runs the camera update as usual and is counted; otherwise `feed_beam` with `make_persistent_label`.
  - ai2_log columns appended after Task 14's: `beam_staged,beam_nolabel,beam_kappa,nomem_cycle,gap_rays`.
  - `test_scene.h`: `RenderedMask render_mask(const Camera&, const SceneBox& target, std::span<const SceneBox> others, float erode_m)` + a projector, so live-staging fixtures build m through `make_persistent_label` with σ_px and ε_px as live.

- [ ] **Step 1: Write the failing tests** in `beam_live_staging_test.cpp` (spec Pass "live staging" items; bounds copied):
  - `silhouette_live`: correctly sized seed, backgrounds 20 cm and 1 m: |Δw|, |Δcx| < 5 mm over 10 frames; ±3 cm seeds end within σ_w of truth. 5 cm background: Δw, Δcx printed against the predicted +1.9 / +3.7 / +5.3 / +6.9 cm at the fixture's σ_px, and the shift per edge < 2σ_px.
  - `flush_neighbour_printed`: surface 2 cm behind the face plane, 30 cm beside the box, 10 frames: Δw with rays vs camera-only, printed.
  - `camera_coverage`: (i) box half outside the image, correct seed: |Δw|, |Δcx|, |Δcy| < 5 mm over 10 frames and no outside-image ray staged; (ii) mask eroded 3 cm on one side: bias of w and cx not above the camera-only run's (both printed).
  - `gap_under_furniture`: box on a 15 cm gap, background 1 m, Helios height: vs the run without the gap rays staged, cy, h, H each move < 1 cm; gap-ray count and energy printed. A failure is reported, not tuned.
  - `calibration_nees`: silhouette fixture through live staging, 200 draws (range noise + per-ray σ_s offset): mean NEES(cx, w)/2 ∈ [0.5, 2] on the 20 cm and 1 m backgrounds, each printed separately, cx and w alone printed; 5 cm printed only.
  - `no_membership_cycle_skips_rays`: a cycle without membership leaves P untouched by rays and increments the log counter.
  - `staging_cost_printed`: `select_rays_beam` + labels over a full Helios sweep, ms printed.
- [ ] **Step 2:** Build and run. Expected: FAIL.
- [ ] **Step 3:** Implement; build the three agents (nightstand, bed, toilet: -j8) so their binaries are current.
- [ ] **Step 4:** Run every countable test. Expected: PASS except items a spec line says are reported to the owner (gap, calibration on 20 cm / 1 m are CHECKs: a failure is reported with numbers, not tuned).
- [ ] **Step 5:** Commit `countable fitter: beam rays staged by geometry on the persistent path, logs (spec 7c.8 a0)`.

### Task 21 (controller, live): objects_live11, objects_live11off, objects_live11shift

As Task 10 again (claim Webots from room and chat; ai2_log cleared before each run; 2 legs each), plus spec Pass
"Live11": fit rate ≥ 8 Hz; `gn_conv` = 0 share on updates with rays below live10's (5 / 12 / 3 / 20 %); final centre
errors not worse than live10's by more than 1 cm; no DOF over-confident (two-sided §8), w and cx reported separately;
per instance the staged ray count, rays without camera information, cycles without membership and their longest run,
and the gap-ray count. ε_px and b* from Task 15, and Task 20's printed flush / 5 cm / gap numbers, go to the owner in
`docs/superpowers/results/2026-11-20-lidar-measures-live11.md`. Push the branch when done; no merge.
