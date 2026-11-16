# Camera attributes, LiDAR measures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every cycle with a fresh ZED mask assigned to a countable instance, the Helios returns update the belief — weighted by how surely the mask says each return belongs to the object — including on cycles where the camera frame itself is gated.

**Architecture:** Three layers. (1) Shared, pure: `rc::ai::LidarRays` gains optional per-ray weights, honoured by the ray factor (`accumulate_lidar_rays`); `rc::lidar_select` gains a weighted selection whose coverage counts Σm; the engine `rc::ai::update` runs a frame with no camera points when the model says it carries other data. (2) countable, pure: a mask-membership map (signed pixel distance to the slice boundary, OpenCV distance transform) and m = Φ(s/σ_px). (3) countable wiring behind `Model.LidarMeasuresOnGated`: the fitter builds a LiDAR-only frame on gated cycles and weights the rays on admissible ones.

**Tech Stack:** C++23, Eigen, OpenCV (countable projection/membership unit only), DSR CameraAPI, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-11-15-lidar-measures-design.md` (approved by the owner 11-15; §7b and §8 amended 11-15 after Fable's plan review — owner: C1 = A, C2 = two-sided).

**Revision 2 (11-15):** folds in Fable's review (`.superpowers/sdd/2026-11-15-lidar-measures/fable-plan-review.md`): C1, C2, I1–I9 and the Minors.

## Global Constraints

- No thresholds: membership is a continuous weight m = Φ(s / σ_px); σ_px² = (f / z)² (σ_range² + σ_chain² + mot_pos_var) + q². No new constant.
- A return behind the camera (camera depth < 0.20 m, the silhouette channel's near clip) or projecting outside the delivered image has m = 0.
- m multiplies each ray's precision BEFORE the ray-count and coverage down-weights; the ray count is Σm (so a ray's information is quadratic in Σm below n0 — spec §7b, read the live `lidar_msum` with that in mind).
- **The LiDAR-only frame (C1 = A):** only the mask-weighted rays plus the one-sided constraints (wall no-cross 2c, short-height 2b). NOT the Gaussian class anchors: wall flush/parallel (`accumulate_wall`), room axis, footprint (2), height anchor (2a). NOT the camera-data terms: points, moments, `z_top_obs` (2d), azimuth. ω (volatility) is not updated on it. All 6 DOFs free.
- **The LiDAR-only frame's common mode:** config floors + `inst.chain_cov_xx/yy` + `mot_pos_var` (cx, cy); `mot_size_var` (size); `mot_yaw_var` (yaw). No range, obliquity or moment terms. The admissible frame's fill is bit-identical to today.
- The selection box stays anchored on the assigned slice's MASK-CLOUD centroid — never the belief mean.
- `Model.LidarMeasuresOnGated` (bool, default false) gates both the LiDAR-only update and the membership weights; false ⇒ byte-identical to today. True in nightstand, bed and toilet configs.
- The divergence net (snapshot + step-bound reject) applies to LiDAR-only updates.
- `lidar_ray_cost` (the door's beam model, `door_belief.cpp:635`) is NOT changed. Other agents untouched (spec §4).
- CLAUDE.md: and/or/not; `std::from_chars` for parsing; `imbue(std::locale::classic())` on written files; builds `make -j8` (check `pgrep -a cc1plus` first); no new graph attribute; `cv::Mat` never crosses a thread (membership raster built and read on the compute thread, no `cv::Mat` kept).
- Commits local on `feat/cost-removal`, worktree `/home/pbustos/robocomp/components/active_inference.worktrees/cost-removal`. Never push or merge.
- Do not patch symptoms: if a pass criterion fails, report it with the numbers; do not add a term, exclusion, floor or gate to make it pass.

## Review Focus

1. **A LiDAR-only frame whose memberships all → 0⁺** (mask elsewhere): must tend continuously to `predict()` — state, Σ and ω — with no switch at Σm = 0. Test `lidar_only_vanishing_membership_is_predict` (Task 2), with a control run showing the anchors would have failed it.
2. **Repeated sweeps of the same scene**: Σ must settle at a known number, not collapse silently. Test `lidar_only_repeated_sweeps` (Task 2) pins √Σ against √(Q·Σc); the live check is two-sided (Task 5).
3. **Flag off must be byte-identical**: a recorded 40-frame trajectory, recorded BEFORE any edit, is re-run after Tasks 1, 2 and 4 (`lidar_golden_test`).
4. **A wall or neighbour inside the box** (the owner's named risk): rays on the wall with m ≈ 0 must not pull the box. Test `wall_behind_does_not_pull` (Task 2) with a membership-≡-1 control that does move.
5. **No extrinsic, no delivered image size, or a return behind / outside the image**: m = 0 per return (pure test `make_membership_*`, Task 3); no membership at all ⇒ predict on a gated cycle, today's unweighted rays on an admissible one (Task 4, reviewer checks the branch).

---

### Task 0 (controller, no subagent): park the uncommitted Task 4c rev 2 work

The worktree holds Task 4c rev 2 uncommitted (8 files). It resumes after this plan on the live8 logs.

- [ ] **Step 1:** Verify the saved patch is current:
  `cd <worktree> && git add -N . && git diff HEAD > /tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/now.patch && diff <(grep -v "^index " /tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chair-concept/4f658c0f-b2ab-436b-8611-7d4e8ef6597a/scratchpad/now.patch) <(grep -v '^index ' .superpowers/sdd/2026-11-15-cost-based-removal/task-4c-rev2-uncommitted.patch)` → empty.
  If not empty, re-save the patch first (`git diff HEAD > …/task-4c-rev2-uncommitted.patch`).
- [ ] **Step 2:** Commit it to a side branch so it is in git, not only in a patch file:
  `git switch -c wip/task-4c-rev2 && git add -A && git commit -m "WIP Task 4c rev 2 (full-state marginal) — parked until lidar-measures lands; BLOCKED on inflated Σ" && git switch feat/cost-removal`.
  Expected: `git status --short` on `feat/cost-removal` is empty; `git log wip/task-4c-rev2 -1` shows the WIP commit.

---

### Task 1: Golden trajectory first, then per-ray weights and the weighted selection (shared, pure)

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

### Task 2: The LiDAR-only frame in the belief and the engine (shared + countable belief)

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

### Task 3: Mask membership (countable, pure + projection helpers)

**Files:**
- Create: `countable_concept/src/countable_mask_membership.h`, `countable_concept/src/countable_mask_membership.cpp` (OpenCV allowed here)
- Modify: `countable_concept/src/countable_projection.h/.cpp` (two helpers below)
- Modify: `countable_concept/src/CMakeLists.txt` (add the new .cpp)
- Create: `countable_concept/tests/mask_membership_test.cpp`; Modify: `countable_concept/tests/CMakeLists.txt` (link OpenCV)

**Interfaces:**
- Produces:
  - `class rc::MaskMembership`, `static MaskMembership build(std::span<const Eigen::Vector2f> pixels)` (pixels in the DELIVERED image, col,row):
    - `float stride_px() const`: q = median over pixels of the nearest distinct column step within the same row (1 dense; 1 for an empty slice). The raster grid is anchored at the slice's minimum (col, row) with step q in both axes, plus a one-cell border.
    - `float signed_distance_px(float col, float row) const`: positive inside, negative outside, in pixels. Inside and outside distances from `cv::distanceTransform` with `cv::DIST_L2, cv::DIST_MASK_PRECISE` on the raster, scaled by q, offset by ±q/2 so a mask pixel reads exactly +q/2 and the cell edge 0; **bilinear** lookup. Outside the raster: `s(nearest border cell) − (distance to that cell)` — continuous at the border, no margin constant.
    - `bool empty() const`.
  - `inline float rc::projection_sigma_px(float f_px, float z_m, float sigma_range_m, float sigma_chain_m, float mot_pos_var_m2, float q_px)` → `sqrt((f/z)²(σr²+σc²+mot_pos_var) + q²)`.
  - `inline float rc::membership_weight(float s_px, float sigma_px)` → `0.5·erfc(−s/(σ√2))`.
  - `using Projector = std::function<std::optional<Eigen::Vector3f>(const Eigen::Vector3f&)>;` and `std::function<float(const Eigen::Vector3f&)> rc::make_membership(Projector proj, const MaskMembership& mm, float f_px, float sigma_range_m, float sigma_chain_m, float mot_pos_var_m2)` — nullopt ⇒ 0; else `membership_weight(mm.signed_distance_px(col,row), projection_sigma_px(f, depth, …, mm.stride_px()))`. Captures `mm` by value.
  - `std::optional<Eigen::Vector3f> CountableProjection::project_to_zed(const Eigen::Matrix4f& zed_T_room, const Eigen::Vector3f& p_room, int frame_cols, int frame_rows) const` → (col, row, depth) in the DELIVERED image: projects through `camera_api_->project` (intrinsic frame), scales col by `frame_cols / W` and row by `frame_rows / H` (as `countable_projection.cpp:449–452`); nullopt when depth < 0.20 m (the silhouette near clip, :333; ZED y-forward), outside [0, frame_cols) × [0, frame_rows), no CameraAPI, or `frame_cols/rows ≤ 0`.
  - `float CountableProjection::focal_px(int frame_cols, int frame_rows) const` → ½(fx·sx + fy·sy), 0 when no CameraAPI or sizes ≤ 0.
- [ ] **Step 1: Write the failing tests** in `mask_membership_test.cpp`:
  - `inside_one_boundary_half_outside_zero`: filled 40×30 rectangle ⇒ m(centre, σ 2) > 0.999; m(10 px outside, σ 2) < 1e-3; on the edge (half a pixel past the last mask pixel) |m − 0.5| < 0.02; on a mask edge pixel itself s = +q/2 (tol 0.05).
  - `symmetric_in_s`: `membership_weight(s,σ) + membership_weight(−s,σ) == 1` (1e-6), s ∈ {0.3, 1, 4}.
  - `sigma_grows_as_z_shrinks`: `projection_sigma_px(700, 0.8, .03, .02, 0, 1) > projection_sigma_px(700, 3.0, .03, .02, 0, 1)`; adding `mot_pos_var = 0.01` increases it; result ≥ q.
  - `stride_of_subsampled_mask`: every 3rd column and row ⇒ `stride_px() == 3`; dense ⇒ 1; the edge test above passes on the stride-3 mask too.
  - `thin_mask`: a single row of 20 px ⇒ finite and ≥ 0 at its pixels, m(pixel, σ 1) ≥ 0.5.
  - `outside_raster_continuous`: s just inside vs just outside the raster border differ by < 0.1 px; a query 200 px away ⇒ m < 1e-6.
  - `make_membership_nullopt_is_zero` and `make_membership_behind_is_zero` with a fake pinhole projector (f 700, 1280×720; returns nullopt for depth < 0.20 or out of image) ⇒ m == 0 exactly; a point projecting to the rectangle centre ⇒ m > 0.99.
- [ ] **Step 2: Run** → fails to compile.
- [ ] **Step 3: Implement** as specified. Every `cv::Mat` local to `build()` (store distances in a `std::vector<float>` plus raster origin and size).
- [ ] **Step 4: Run** `mask_membership_test` → all pass. Agent build (`pgrep -a cc1plus`, then `make -C build -j8` in `countable_concept`) → compiles.
- [ ] **Step 5: Commit** `git commit -m "countable: mask membership — signed pixel distance to the assigned slice, m = Φ(s/σ_px) (spec lidar-measures G2)"`

---

### Task 4: Wire it into the fitter behind `Model.LidarMeasuresOnGated`, plus config, log and contract

**Files:**
- Modify:
  - `countable_concept/src/countable_config.h/.cpp`: `bool lidar_measures_on_gated = false;` read via `getb("Model.LidarMeasuresOnGated", false, "…")`.
  - `countable_concept/src/countable_fitter.h`: `int slice_index = -1;` in `CountableObservation`.
  - `countable_concept/src/countable_fitter.cpp`: `observe_slice` sets `observation.slice_index`; `run_inference` as below.
  - `countable_concept/src/countable_lidar_range_channel.h/.cpp`: `feed` takes an anchor cloud and a membership.
  - `countable_concept/src/countable_instance.h`: `int dbg_lidar_only = 0; float dbg_lidar_msum = 0.0f;`.
  - The ai2_log writer in `countable_fitter.cpp`: columns `lidar_only,lidar_msum` appended at the END of header and row.
  - `nightstand_concept/etc/config.toml`, `bed_concept/etc/config.toml`, `toilet_concept/etc/config.toml`: `LidarMeasuresOnGated = true` in `[Model]` next to `LidarPrecision`, one-line comment citing the spec.
  - `countable_concept/KEYMAP.md`: `Model.LidarMeasuresOnGated` noted as born after the migration; the existing line 90 `BedModel.ExistenceCarveParts -> Model.ExistenceCarveParts` stays (it is a rename, and correct).
  - `countable_concept/tests/config_migration_test.cpp`: add `"Model.ExistenceCarveParts"` and `"Model.LidarMeasuresOnGated"` to `produced` (closes the removal plan's parked Task 3 defect).
  - `CONCEPT_AGENT_LIFECYCLE.md`: §2 paragraph + countable audit row.

**Interfaces:**
- Consumes: Task 1 `select_into_weighted`; Task 2 `CountableFrame::lidar_only` and the engine; Task 3 `MaskMembership`, `make_membership`, `project_to_zed`, `focal_px`.
- Produces: `void CountableLidarRangeChannel::feed(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud, const std::function<float(const Eigen::Vector3f&)>& membership = {}) const` — anchor centroid and the "nothing to anchor" early return read `anchor_cloud`; empty `membership` ⇒ today's `select_into`; non-empty ⇒ `select_into_weighted`, Σm into `inst.dbg_lidar_msum`. bpearl stays unweighted (precision 0 for these classes).

- [ ] **Step 1: Failing check first.** `config_migration_test` on the current tree → FAIL on `Model.ExistenceCarveParts` (known defect). Add both keys to config and `produced` → FAIL on `LidarMeasuresOnGated` until the toml lines exist. Add them → PASS.
- [ ] **Step 2: The flag-off path.** Flag false ⇒ `run_inference` calls `lidar_channel_.feed(inst, frame, frame.points, {})`; the gated branch is exactly `inst.ai2_belief.predict()`. The common-mode and divergence-net refactor below must be a mechanical move (the reviewer checks it line by line against BASE). `lidar_golden_test` passes.
- [ ] **Step 3: The flag-on path, in `run_inference`.**
  - **Common mode.** A local lambda `fill_common_mode(CountableFrame&, bool lidar_only)`. `lidar_only == false`: today's fill verbatim (range, obliquity, moment, motion). `lidar_only == true`: `chain_cov_xx/yy = inst.chain_cov_xx/yy + mot_pos_var`, `chain_cov_size = mot_size_var`, `chain_cov_yaw = mot_yaw_var`, `moment_extra_var = 0`.
  - **Membership** (flag on, `observation.slice_index ≥ 0`, `have_cam`, the slice has pixels, and `rgb_frame_` non-empty for the delivered size `rgb_frame_.cols/rows`): `mm = MaskMembership::build(slice pixels)`; `membership = make_membership([&](p){ return projection_->project_to_zed(zed_T_room, p, cols, rows); }, mm, projection_->focal_px(cols, rows), cfg_.existence_sensor_sigma_m, √(½(inst.chain_cov_xx + inst.chain_cov_yy)), mot_pos_var)`. `zed_T_room` is the one computed at `inst.last_mask_timestamp_ms` (the packet stamp; equal to the ZED slice's `MaskSlice::timestamp_ms` — say so in a comment). Any condition missing ⇒ no membership.
  - **Admissible branch:** `feed(inst, frame, frame.points, membership)` (empty function if none).
  - **Gated branch:** flag on AND membership built ⇒ LiDAR-only frame, else `predict()` as today.
    - `CountableFrame lf; lf.lidar_only = true;` points, R, point_azim empty; `z_top_obs = −1`; `has_rays = false`.
    - `fill_common_mode(lf, true)`; stage wall, interior and room axis as the admissible branch does (the no-cross term needs the wall); `feed(inst, lf, anchor = candidate ∪ residual points, membership)`.
    - `lf.lidar.endpoints` empty or precision 0 ⇒ `predict()`.
    - Else the same snapshot, `update(lf)` and step-bound reject as the admissible branch (one shared lambda).
    - Energy held (`inst.dbg_energy`); FE baseline and surprise untouched; `resolve_orientation` and `resolve_front_from_rgb` not run.
    - `inst.dbg_lidar_only = 1`; 0 on every other cycle.
  - `[AI2]` print line: `LIDAR-ONLY` in place of ` GATED` when it applies.
  - Comments to add (known behaviour, not changed here): one `run_inference` per assigned slice feeds the same sweep (`specificworker.cpp:647–655`); a slice whose points were all dropped (`has_fresh_data` false) gets no LiDAR update although it has pixels.
- [ ] **Step 4: Contract.**
  - `CONCEPT_AGENT_LIFECYCLE.md` §2, add: "A LiDAR ray set attributed by an assigned mask is a frame of its own. Its *resolvable* is the existing ray-count/coverage precision, its *centred* is attribution by the mask's pixels (m = Φ(s/σ_px)), and its *still* is the common-mode Σc of the pose chain and ego-motion. It carries only its rays and the one-sided constraints; the class anchors are already in the predicted belief and are not re-added. It may move geometry on a cycle whose camera frame is gated; the camera points still may not (countable, `Model.LidarMeasuresOnGated`, spec 2026-11-15 lidar-measures)."
  - Countable audit row, UPDATE column: `✅ trunc + confirm_only for camera points; mask-attributed LiDAR-only updates on gated cycles (2026-11-15)`.
- [ ] **Step 5: Build and test.** `countable_concept` at -j8; every countable test (`ctest` in the tests build) → all pass, incl. `config_migration_test` and `lidar_golden_test`. Start nothing live.
- [ ] **Step 6: Commit** `git commit -m "countable: camera attributes, LiDAR measures — mask-weighted rays, LiDAR-only update on gated cycles (Model.LidarMeasuresOnGated; spec 2026-11-15)"`

---

### Task 5 (controller, live): objects_live8, then the verdict

Live; the controller runs it. Webots is shared: claim it from the chat and room sessions first, FREE after.

- [ ] **Step 1:** Rebuild nightstand, bed and toilet from the worktree. Run the objects_live6 tour twice with the room session's validated fleet set and the same launch script, concept agents from this worktree (`CONCEPT_WT=<worktree>`): `objects_live8` (flag on) and `objects_live8off` (the same tour, `LidarMeasuresOnGated = false` via a run-local config copy), into `room_concept/estimation_scripts/crossing/`. The off run separates the change from tour-to-tour variance.
- [ ] **Step 2:** Score with `countable_concept/tools/lidar_measures_score.py RUN_DIR [OFF_DIR]` (classic locale; inputs `ai2_log.csv` per class + `gt_pose.tsv`, GT from `scene_threerooms.csv`). A *look* = a maximal run of rows with a fresh mask and no gap > 2 s. Per real object (bed, both nightstands, toilet), against live6 and the off run:
  - final centre error (live6: 5.5 cm; 2–6 cm; 23 cm); |H − true top|;
  - per DOF (cx, cy, H), after the first look from within 2.5 m: rms of the true error, min and max std; PASS iff max std ≤ 0.15 m AND rms/3 ≤ std ≤ 3·rms (two-sided, spec §8); also print the median NEES;
  - the largest centre step toward a wall or a neighbour between consecutive looks (≤ 10 cm);
  - fit rate in Hz (about 10); share of `lidar_only == 1`; median `lidar_msum`.

  Each line ends PASS or FAIL.
- [ ] **Step 3:** Write `docs/superpowers/results/2026-11-15-lidar-measures-live8.md` and commit. A FAIL is reported with its numbers, not tuned. A std below rms/3 goes to the owner as a model question (a shared error persisting across sweeps), with the Task 2 `lidar_only_repeated_sweeps` baseline beside it.
- [ ] **Step 4:** If it passes, the removal plan resumes: re-apply `wip/task-4c-rev2` (cherry-pick, uncommitted) and run its G0b replay on objects_live8. Pass criterion stands: real objects ≥ 0, phantoms < 0.
