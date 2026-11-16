# Camera attributes, LiDAR measures — Implementation Plan (rev 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every cycle with a fresh ZED mask assigned to a countable instance, the Helios returns update the belief — each return weighted by its posterior ownership (camera m as prior, room and published neighbour boxes as the alternative) — while the class prior is counted once and every channel's shared error persists across frames, so Σ is honest and H follows the LiDAR top.

**Architecture:** Four layers. (1) Shared, pure: per-ray weights + weighted selection (DONE, Task 1); the engine's LiDAR-only frame hook (DONE, Task 2); a new persistent-latent update `common/ai_belief/persistent_update.h` (joint θ + shared-error latents, inverse-free, double); an ownership-mixture mode of the ray factor and a first-hit "rest" ray-cast in `common/`. (2) countable, pure: mask membership m = π·Φ(s/σ_px); viewpoint ρ from odometry. (3) countable belief behind `Model.LidarMeasuresOnGated`: 20-state belief (θ, b_chain, b_cam, b_L), prior counted once, Σ bounded by the class P, accumulation split into camera / LiDAR / θ-only channels. (4) fitter wiring, config, log, contract; then the live A/B.

**Tech Stack:** C++23, Eigen (float model, double solve), OpenCV (membership unit only), DSR (typed attributes), CMake/ctest, Python 3 (table + scoring scripts).

**Spec:** `docs/superpowers/specs/2026-11-15-lidar-measures-design.md` — §7c rev 3.1 (approved by the owner 11-16) supersedes G3, the §7b bullet "m and the down-weights" and §7c revs 1–2. Reviews behind it: `.superpowers/sdd/2026-11-15-lidar-measures/fable-7c-review.md`, `fable-7c-rev2-review.md`, `fable-7c-rev3-review.md`.

**Revision 3 (11-16):** Tasks 0–2 are done (commits ff49e768, 6569e6b1, b58f697b; merged to main 8f4cc821). Tasks 3–10 replace rev 2's Tasks 3–5.

**Execution order (owner 11-16, "go with the shorter"):** 3 → 4 → 5 → 6 → 9 → 10. Tasks 7–8 (the ownership mixture) are DEFERRED until live8 shows whether a wrong-attribution pull exists with physical returns; until then the rays carry the camera's m as a plain weight (Task 1). Task 9 is built WITHOUT the rest scene (no `RestScene` parameter, no `resp_sum` log column); they are added with Tasks 7–8 if live8 asks for them.

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
  - `physical_lidar_only_sigma_bounded`: Task 3 scene, 2000 LiDAR-only sweeps ⇒ √Σ(h,h) ≤ `prior_footprint_std` (+1 %) and the back face moves < 1 mm.
  - `parked_camera_is_one_look`: 100 identical camera frames at ρ = 1, zero fresh covariance ⇒ Σθθ equals the one-frame result with 100·Id (rel 1 %).
  - `zero_chain_cov_runs`: `sig2_chain = 0` ⇒ no NaN, update proceeds.
  - `mode_rotation_permutes_latents`: set distinct b_cam/b_L w and h values, apply a 90° mode rotation ⇒ they swap; b_chain unchanged; P symmetric, min eigenvalue ≥ −1e-12.
  - `sanitize_reseeds`: corrupt P_ ⇒ after `sanitize_covariance` θθ = P, latents = marginals, `prior_added_ == false`.
- [ ] **Step 2:** Run → fail. **Step 3:** Implement. **Step 4:** Run the new test + every countable test → pass (`lidar_golden_test` included); build `countable_concept` at -j8.
- [ ] **Step 5: Commit** `git commit -m "countable belief: persistent shared errors (b_chain, b_cam, b_L) and the class prior counted once, behind Model.LidarMeasuresOnGated (spec 7c.1–7c.2)"`

---

### Task 7 (DEFERRED, see Execution order): The ownership mixture in the ray factor and the rest ray-cast (shared, pure)

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

### Task 8 (DEFERRED, see Execution order): The belief uses the mixture (countable, flag on)

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
