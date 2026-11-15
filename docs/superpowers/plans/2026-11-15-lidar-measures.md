# Camera attributes, LiDAR measures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On every cycle with a fresh ZED mask assigned to a countable instance, the Helios returns update the belief — weighted by how surely the mask says each return belongs to the object — including on cycles where the camera frame itself is gated.

**Architecture:** Three layers. (1) Shared, pure: `rc::ai::LidarRays` gains optional per-ray weights, honoured by the ray factor and its cost; `rc::lidar_select` gains a weighted selection whose coverage counts Σm; the engine `rc::ai::update` runs a frame with no camera points when the model says it carries other data. (2) countable, pure: a mask-membership map (signed pixel distance to the slice boundary, OpenCV distance transform) and m = Φ(s/σ_px). (3) countable wiring behind `Model.LidarMeasuresOnGated`: the fitter builds a LiDAR-only frame on gated cycles and weights the rays on admissible ones.

**Tech Stack:** C++23, Eigen, OpenCV (countable projection/membership unit only), DSR CameraAPI, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-11-15-lidar-measures-design.md` (approved by the owner 11-15).

## Global Constraints

- No thresholds: membership is a continuous weight m = Φ(s / σ_px); σ_px² = (f / z)² (σ_range² + σ_chain²) + q². No new constant.
- A return behind the camera or projecting outside the image has m = 0.
- m multiplies each ray's precision BEFORE the ray-count and coverage down-weights; the ray count is Σm.
- The LiDAR-only frame keeps the structural priors (wall flush / no-cross, room axis, footprint / height priors) and drops the camera-data terms (mask points, moments, `z_top_obs`, per-point azimuth). All 6 DOFs free.
- The selection box stays anchored on the assigned slice's MASK-CLOUD centroid — never the belief mean.
- `Model.LidarMeasuresOnGated` (bool, default false) gates both the LiDAR-only update and the membership weights; false ⇒ byte-identical to today. True in nightstand, bed and toilet configs.
- The divergence net (snapshot + step-bound reject) applies to LiDAR-only updates.
- CLAUDE.md: and/or/not; `std::from_chars` for parsing; `imbue(std::locale::classic())` on written files; builds `make -j8` (check `pgrep -a cc1plus` first); no new graph attribute; `cv::Mat` never crosses a thread (membership raster built and read on the compute thread).
- Commits local on `feat/cost-removal`, worktree `/home/pbustos/robocomp/components/active_inference.worktrees/cost-removal`. Never push or merge.
- Do not patch symptoms: if a pass criterion fails, report it with the numbers; do not add a term, exclusion or gate to make it pass.

## Review Focus

1. **A frame where every selected return has m ≈ 0** (the mask is elsewhere — wall behind a nightstand): the update must leave the state essentially at the prediction, not divide by Σm = 0. Test pinned in Task 1 (`weighted_select_all_zero`).
2. **Engine with no points and no rays**: must stay exactly predict-only and return 0 (mean energy must not divide by zero). Test in Task 2 (`empty_frame_is_predict_only`).
3. **Flag off must be byte-identical**: the admissible path's numbers must not move by one ulp. Golden test in Task 2 (`golden_mixed_frame`), and Task 4 calls the old `feed` path unchanged when off.
4. **A mask slice with very few pixels or a single pixel row** (distance transform on a 1-px-high raster): signed distance must stay finite and m must still be 0.5 on the pixel itself. Test in Task 3 (`thin_mask`).
5. **A gated cycle with no camera extrinsic at the mask stamp** (`room_T_zed_matrix` nullopt): no attribution is possible, so it must stay predict-only (G4 spirit); an admissible cycle without it keeps today's unweighted rays. Pinned by code path in Task 4 Step 3 and checked by the reviewer.

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

### Task 1: Per-ray weights and the weighted selection (shared, pure)

**Files:**
- Modify: `common/ai_belief/lidar_ray_factor.h` (struct `LidarRays`, `lidar_ray_cost`, `accumulate_lidar_rays`)
- Modify: `common/lidar_select/lidar_select.h` (add `select_into_weighted`; `select_into` clears `out.weights`)
- Create: `countable_concept/tests/lidar_weights_test.cpp`
- Modify: `countable_concept/tests/CMakeLists.txt` (add `lidar_weights_test`, header-only: needs Eigen only)

**Interfaces:**
- Produces:
  - `LidarRays::weights` — `std::vector<float>`; empty ⇒ every ray weight 1; else same length as `endpoints`, each in [0, 1], multiplying `precision` per ray.
  - `int rc::lidar_select::select_into_weighted(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin, const Eigen::Vector2f& centroid_xy, const Box& box, float precision_base, const Weights& w, const std::function<float(const Eigen::Vector3f&)>& membership, rc::ai::LidarRays& out, float* cov_ang = nullptr, float* m_sum = nullptr)` — returns the number of staged endpoints; `*m_sum` = Σm.

- [ ] **Step 1: Write the failing tests** in `lidar_weights_test.cpp` (plain `CHECK` macro as in `placement_prior_test.cpp`; a toy model with one axis-aligned box primitive exposing `n_prims()`, `sdf_prim`, `sdf_jacobian` for N = 6, or reuse `CountableBelief` linked via `../src/countable_belief.cpp` — implementer's choice):
  - `weights_empty_equals_ones`: same rays, `weights = {}` vs `weights = {1,…,1}` ⇒ `Id`, `bd` and `lidar_ray_cost` bit-identical.
  - `weights_half_scales_information`: `weights` all 0.5 ⇒ `Id` and `bd` equal 0.5 × the unweighted ones (rel. tol 1e-6); cost likewise.
  - `weighted_select_ones_matches_select_into`: a synthetic sweep (≥ 80 points, some outside the box); `membership ≡ 1` ⇒ same `endpoints`, same `precision` (bit-identical), `weights` all 1, `m_sum == n`.
  - `weighted_select_coverage_counts_msum`: `coverage_n0 = 60`, 100 in-box returns, membership 1 for 30 of them and 0 for 70 ⇒ `precision == precision_base · (30/60) · ca`, where `ca` is the angular weight computed over the membership-weighted bearings.
  - `weighted_select_all_zero`: membership ≡ 0 ⇒ `m_sum == 0`, `out.precision == 0` (the factor then skips it), no NaN anywhere.
- [ ] **Step 2: Run to verify they fail**
  `cmake -B build -S countable_concept/tests … && make -C build -j8 lidar_weights_test` → compile error (`weights` / `select_into_weighted` undefined).
- [ ] **Step 3: Implement.**
  - `LidarRays`: add `std::vector<float> weights;` with a comment "empty ⇒ 1 per ray (every existing caller)".
  - `accumulate_lidar_rays` and `lidar_ray_cost`: iterate by index; per-ray `wi = weights.empty() ? 1 : weights[i]`; skip `wi <= 0`; multiply the Cauchy weight (and the cost term) by `wi`. Nothing else changes.
  - `select_into`: add `out.weights.clear();` next to `out.endpoints.clear();` (unweighted callers stay unweighted).
  - `select_into_weighted`: same loop as `select_into`; per in-box return `m = membership(p)`; skip `m <= 0` (not staged); stage endpoint + weight; accumulate Σm and m-weighted bearing sums; `coverage = min(1, Σm / max(1, n0))`; `R = |Σ m·e^{iφ}| / Σm`; `ca = max(0, 1−R)^power`; `precision = precision_base · coverage · ca` when Σm > 0, else `precision = 0`.
- [ ] **Step 4: Run** `./build/lidar_weights_test` → `all passed`, exit 0. Also rebuild and run the existing `countable_concept/tests` (all must still pass).
- [ ] **Step 5: Commit** `git commit -m "lidar_select + ray factor: per-ray membership weights; coverage counts Σm (spec 2026-11-15 lidar-measures G2)"`

---

### Task 2: The engine runs a frame with no camera points (shared)

**Files:**
- Modify: `common/ai_belief/recursive_laplace.h` (`update<N>`)
- Modify: `countable_concept/src/countable_belief.h` (`bool has_extra_data(const CountableFrame&) const`)
- Create: `countable_concept/tests/lidar_only_update_test.cpp` (links `../src/countable_belief.cpp`)
- Modify: `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `LidarRays::weights` (Task 1).
- Produces: `CountableBelief::has_extra_data(const CountableFrame& f) const` → `not f.lidar.endpoints.empty() and f.lidar.precision > 0`, or any `f.lidar_extra` set likewise. The engine detects it with `if constexpr (requires { m.has_extra_data(frame); })`; bottle / chair / table (no hook) unchanged.

- [ ] **Step 1: Record the golden numbers at BASE, then write the tests.**
  Build a fixed `CountableFrame` (a nightstand-like box: 200 synthetic surface points with R = σ_base², 120 synthetic Helios rays from an origin 1.5 m away at z 1.1075, `precision 2500`, `robust_c_m 0.05`, a `WallRef` and `room_axis`) and a `CountableBelief` seeded off-truth by (+0.05, −0.04, H +0.03). Before touching the engine, run one `update()` and print `state().vec()` and `covariance().diagonal()` with `%a`. Hard-code those hex floats into:
  - `golden_mixed_frame`: after Task 2's change, the same frame gives bit-identical state and Σ diagonal.
  - `empty_frame_is_predict_only`: no points, no rays ⇒ state unchanged, Σ equals that after `predict()` on a copy, return value 0, no NaN.
  - `lidar_only_moves_toward_rays`: no points, the 120 rays of the truth box ⇒ |centre error| and |H error| both strictly smaller after one update than before; Σ(cx,cx), Σ(cy,cy) and Σ(H,H) strictly smaller than after a bare `predict()`.
  - `lidar_only_saturates`: the same, rays duplicated ×10 (same sweep) with `chain_cov_xx = chain_cov_yy = 0.01` ⇒ √Σ(cx,cx) ≥ 0.5·√(1/(1/Σ_pred + 1/0.01)) (the common-mode floor holds; N correlated rays do not collapse Σ) — compute the bound from the predicted Σ in the test.
- [ ] **Step 2: Run** — `golden_mixed_frame` passes (unchanged code), the three LiDAR-only tests FAIL (engine returns early).
- [ ] **Step 3: Implement.** In `update<N>`: replace `if (frame.points.empty()) return 0.0f;` with a return only when there are no points AND (the model has no `has_extra_data` hook, or it returns false). Guard the final mean energy: return 0 when `frame.points.empty()`. Add `has_extra_data` to `CountableBelief`. No other engine change.
- [ ] **Step 4: Run** `lidar_only_update_test` → all four pass; every other countable test still passes.
- [ ] **Step 5: Commit** `git commit -m "ai engine: a frame with no camera points updates when the model has other data (LiDAR-only frame; spec lidar-measures G1/G3)"`

---

### Task 3: Mask membership (countable, pure + one projection helper)

**Files:**
- Create: `countable_concept/src/countable_mask_membership.h`, `countable_concept/src/countable_mask_membership.cpp` (OpenCV allowed here)
- Modify: `countable_concept/src/countable_projection.h/.cpp` (two helpers below)
- Modify: `countable_concept/src/CMakeLists.txt` (add the new .cpp)
- Create: `countable_concept/tests/mask_membership_test.cpp`; Modify: `countable_concept/tests/CMakeLists.txt` (link OpenCV)

**Interfaces:**
- Produces:
  - `class rc::MaskMembership` built by `static MaskMembership build(std::span<const Eigen::Vector2f> pixels, float margin_px)`:
    - `float stride_px() const`: q, the slice's measured pixel spacing = the median over pixels of the nearest distinct column step within the same row (1 for a dense mask). An empty slice has stride 1.
    - `float signed_distance_px(float col, float row) const`: positive inside, negative outside, in pixels. It is computed with `cv::distanceTransform` (DIST_L2, mask 5) on the bounding raster plus margin, at cell = q, scaled by q, and offset by ±q/2 so that a pixel of the mask itself reads exactly ≥ 0. A query outside the raster returns `−(distance to the raster) − margin`.
    - `bool empty() const`.
  - `inline float rc::projection_sigma_px(float f_px, float z_m, float sigma_range_m, float sigma_chain_m, float q_px)` → `sqrt((f/z)²(σr²+σc²) + q²)`.
  - `inline float rc::membership_weight(float s_px, float sigma_px)` → `Φ(s/σ)` = `0.5·erfc(−s/(σ√2))`.
  - `std::optional<Eigen::Vector3f> CountableProjection::project_to_zed(const Eigen::Matrix4f& zed_T_room, const Eigen::Vector3f& p_room) const` → (col, row, depth). It returns nullopt when the depth axis is ≤ 0 (behind the camera; the ZED frame here is y-forward, as in `compute_silhouette_existence`), when the point is outside [0, W) × [0, H), or when there is no CameraAPI. It projects through `camera_api_->project`, the same path as the silhouette channel.
  - `float CountableProjection::focal_px() const` → ½(fx + fy), or 0 when no CameraAPI.
- [ ] **Step 1: Write the failing tests** in `mask_membership_test.cpp`:
  - `inside_one_boundary_half_outside_zero`: a filled 40×30 px rectangle. Check m(centre, σ = 2) > 0.999 and m(a point 10 px outside, σ = 2) < 1e-3. For the boundary, check |m(on the rectangle edge, σ = 2) − 0.5| < 0.1, where the edge lies half a pixel outside the last mask pixel.
  - `symmetric_in_s`: `membership_weight(s, σ) + membership_weight(−s, σ) == 1` (tol 1e-6) for s ∈ {0.3, 1, 4}.
  - `sigma_grows_as_z_shrinks`: `projection_sigma_px(700, 0.8, .03, .02, 1) > projection_sigma_px(700, 3.0, .03, .02, 1)`, and the result at q only is ≥ q.
  - `stride_of_subsampled_mask`: the same rectangle with every 3rd column ⇒ `stride_px() == 3`; dense ⇒ 1.
  - `thin_mask`: a single row of 20 px ⇒ `signed_distance_px` finite at the row's pixels and ≥ 0, and m(pixel, σ = 1) ≥ 0.5.
  - `outside_raster_is_far_outside`: a query 200 px away ⇒ m < 1e-6.
  - The "behind the camera / outside the image ⇒ m = 0" rule is enforced by the caller (Task 4) treating `project_to_zed == nullopt` as m = 0. It is pinned there.
- [ ] **Step 2: Run** → fails to compile (header missing).
- [ ] **Step 3: Implement** as specified above. Keep every `cv::Mat` local to `build()` (store the distance values in a `std::vector<float>` plus the raster origin and size), so that no `cv::Mat` outlives the call (CLAUDE.md thread rule).
- [ ] **Step 4: Run** `mask_membership_test` → all pass. Then do the agent build (`pgrep -a cc1plus`, then `make -C build -j8` in `countable_concept`) → compiles.
- [ ] **Step 5: Commit** `git commit -m "countable: mask membership — signed pixel distance to the assigned slice, m = Φ(s/σ_px) (spec lidar-measures G2)"`

---

### Task 4: Wire it into the fitter behind `Model.LidarMeasuresOnGated`, plus config, log and contract

**Files:**
- Modify:
  - `countable_concept/src/countable_config.h/.cpp`: `bool lidar_measures_on_gated = false;` read via `getb("Model.LidarMeasuresOnGated", false, "…")`.
  - `countable_concept/src/countable_fitter.h`: add `int slice_index = -1;` to `CountableObservation`.
  - `countable_concept/src/countable_fitter.cpp`: in `observe_slice`, set `observation.slice_index`. In `run_inference`: build the common mode once for both branches, add the gated LiDAR-only branch, and pass the membership to `feed` on the admissible branch.
  - `countable_concept/src/countable_lidar_range_channel.h/.cpp`: give `feed` an anchor cloud and a membership.
  - `countable_concept/src/countable_instance.h`: `int dbg_lidar_only = 0; float dbg_lidar_msum = 0.0f;`.
  - The ai2_log writer in `countable_fitter.cpp`: append columns `lidar_only,lidar_msum` at the END of the header and row.
  - `nightstand_concept/etc/config.toml`, `bed_concept/etc/config.toml`, `toilet_concept/etc/config.toml`: `LidarMeasuresOnGated = true` in `[Model]`, next to `LidarPrecision`, with a one-line comment citing the spec.
  - `countable_concept/KEYMAP.md`: note both new keys as born after the migration.
  - `countable_concept/tests/config_migration_test.cpp`: add `"Model.ExistenceCarveParts"` and `"Model.LidarMeasuresOnGated"` to the `produced` set. This also closes the parked Task 3 defect of the removal plan.
  - `CONCEPT_AGENT_LIFECYCLE.md`: add one paragraph to §2, and update the countable audit row's UPDATE column.

**Interfaces:**
- Consumes:
  - from Task 1, `select_into_weighted`;
  - from Task 2, the engine LiDAR-only update;
  - from Task 3, `MaskMembership`, `projection_sigma_px`, `membership_weight`, `project_to_zed`, `focal_px`.
- Produces: `void CountableLidarRangeChannel::feed(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud, const std::function<float(const Eigen::Vector3f&)>& membership = {}) const`.
  - The anchor centroid and the "nothing to anchor" early return now read `anchor_cloud`, not `frame.points`.
  - An empty `membership` calls today's `select_into`. A non-empty one calls `select_into_weighted`, and stores Σm in `inst.dbg_lidar_msum`.
  - bpearl stays unweighted (precision 0 for these classes; spec §9).

- [ ] **Step 1: Failing check first.** Run `config_migration_test` on the current tree → FAIL on `Model.ExistenceCarveParts`, the known defect. Add both keys to the config and to `produced` → still FAIL on `LidarMeasuresOnGated` until the toml lines exist. Add the toml lines → PASS.
- [ ] **Step 2: The flag-off path.** With the flag false, `run_inference` must call `lidar_channel_.feed(inst, frame, frame.points, {})`. That is the same selection as today, so the golden test of Task 2 and the existing tests are unchanged. The gated branch is still exactly `inst.ai2_belief.predict()`.
- [ ] **Step 3: The flag-on path, in `run_inference`.**
  - **Common mode.** Move the common-mode fill (`chain_cov_xx/yy/yaw/size`, `moment_extra_var`) into a local lambda `fill_common_mode(CountableFrame&)`, used verbatim by both branches. This is a pure move; the admissible numbers stay bit-identical.
  - **Membership** (only when the flag is on, `observation.slice_index ≥ 0`, `have_cam`, and the slice has pixels):
    - Build `MaskMembership::build(slice pixels, margin)`, where margin = 3·`projection_sigma_px(f, max(range, 0.3), σ_range, σ_chain, q)`.
    - For a return p, `membership(p)` is: `proj = projection_->project_to_zed(zed_T_room, p)`; if nullopt then m = 0; else m = `membership_weight(mm.signed_distance_px(col, row), projection_sigma_px(focal_px(), depth, cfg_.existence_sensor_sigma_m, σ_chain, mm.stride_px()))`.
    - σ_chain = √(½(inst.chain_cov_xx + inst.chain_cov_yy)).
    - The projection uses the `zed_T_room` already computed at `inst.last_mask_timestamp_ms`.
  - **Admissible branch:** `feed(inst, frame, frame.points, membership)`. If no membership could be built, use the empty function (today's rays).
  - **Gated branch:**
    - The LiDAR-only frame is used only when the flag is on and a membership was built. Otherwise keep `predict()` as today (no attribution means no update).
    - Build a `CountableFrame lf`. Leave `points`, `R` and `point_azim` empty, keep `z_top_obs = −1`, and set `has_rays = false`.
    - Run `fill_common_mode(lf)`; stage the wall, interior and room axis exactly as the admissible branch does; then `feed(inst, lf, anchor = candidate ∪ residual points, membership)`.
    - If `lf.lidar.endpoints` is empty or its precision is 0, fall back to `predict()`.
    - Otherwise run the same snapshot, `update(lf)` and step-bound reject as the admissible branch (factor the divergence net into a lambda shared by both; don't copy it).
    - Energy is held (`inst.dbg_energy`); the FE baseline and surprise are not touched; `resolve_orientation` and `resolve_front_from_rgb` are not run (they are camera data).
    - Set `inst.dbg_lidar_only = 1`; it is 0 on every other cycle.
  - **The `[AI2]` print line:** show `LIDAR-ONLY` in place of ` GATED` when it applies.
- [ ] **Step 4: Contract.**
  - In `CONCEPT_AGENT_LIFECYCLE.md` §2, add this paragraph: "A LiDAR ray set attributed by an assigned mask is a frame of its own. Its *resolvable* is the existing ray-count/coverage precision, its *centred* is attribution by the mask's pixels (m = Φ(s/σ_px)), and its *still* is the common-mode Σc. It may move geometry on a cycle whose camera frame is gated; the camera points still may not (countable, `Model.LidarMeasuresOnGated`, spec 2026-11-15 lidar-measures)."
  - In the countable audit row, the UPDATE column becomes: `✅ trunc + confirm_only for camera points; mask-attributed LiDAR updates on gated cycles (2026-11-15)`.
- [ ] **Step 5: Build and test.** Build `countable_concept` at -j8. Run every countable test (`ctest` in the tests build) → all pass, including `config_migration_test`. Then start nothing live.
- [ ] **Step 6: Commit** `git commit -m "countable: camera attributes, LiDAR measures — mask-weighted rays, LiDAR-only update on gated cycles (Model.LidarMeasuresOnGated; spec 2026-11-15)"`

---

### Task 5 (controller, live): objects_live8, then the verdict

This is live; the controller runs it, not a subagent. Webots is shared: claim it from the chat and room sessions first, and post FREE after.

- [ ] **Step 1:** Rebuild the nightstand, bed and toilet binaries from the worktree. Run the objects_live6 tour as objects_live8 using the room session's validated fleet set and the same launch script, with concept agents taken from this worktree (`CONCEPT_WT=<worktree>`). Output goes to `room_concept/estimation_scripts/crossing/objects_live8`.
- [ ] **Step 2:** Score it with a script `countable_concept/tools/lidar_measures_score.py RUN_DIR`, read under the classic locale. Its inputs are `ai2_log.csv` per class plus `gt_pose.tsv`, scored against the GT in the spec of the removal plan's replay (`scene_threerooms.csv`). For each real object (bed, both nightstands, toilet) it prints:
  - the final centre error, and the live6 value it is compared with (5.5 cm; 2–6 cm; 23 cm);
  - |H − true top|, against live6;
  - the rms of the true error and the max of std_cx, std_cy and std_H after the first look from within 2.5 m. It checks max std ≤ 0.15 m, and std ≤ 3 × the rms;
  - the largest centre step toward a wall or a neighbour between consecutive looks (≤ 10 cm);
  - the fit rate in Hz (about 10);
  - the share of rows with `lidar_only == 1`, and the median `lidar_msum`.
  
  Each line ends PASS or FAIL.
- [ ] **Step 3:** Write the result to `docs/superpowers/results/2026-11-15-lidar-measures-live8.md` and commit it. A FAIL is reported with its numbers; it is not tuned.
- [ ] **Step 4:** If it passes, the removal plan resumes:
  - re-apply `wip/task-4c-rev2` (cherry-pick, uncommitted);
  - run its G0b replay on objects_live8.

  The removal plan's pass criterion stands: real objects ≥ 0, phantoms < 0.
