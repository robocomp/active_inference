# Shared Estimation Engine, Part 1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One persistent estimation engine in `common/ai_belief` replaces the per-frame update for chair, table, refrigerator and hood, and the countable binary runs on it too. On that engine:
- shared error persists within a view and decays as the view changes;
- free space comes from beam rays;
- mask error is a measured sensor model;
- shape misfit is a persistent latent;
- offers are computed under the same cap and checked against occupancy.

**Architecture:** The countable belief's persistent path is lifted into a model-generic engine with its pieces:
- `persistent_update.h`, already generic;
- the GN/LM loop, fix A, fix B and the beam staging, currently in `countable_belief.cpp`;
- `countable_viewpoint.*`.

A class supplies a model through a C++23 concept: its state, its SDF primitives, its beam parts, a θ-only channel, and the chain lever. The engine owns the latents, the solver, the offer gain and the logs. Classes switch between engines through config (`Model.Engine = "persistent" | "legacy"`) until grading.

**Tech Stack:** C++23, Eigen (fixed-size, double in the solver), CMake + ctest (`/usr/bin/ctest`), Python 3 tools (numpy, no pandas required), Webots for live tests.

**Spec:** `docs/superpowers/specs/2026-11-23-shared-engine-design.md` (approved by the owner on 11-23). Baseline: `docs/superpowers/results/2026-11-23-patch-constants-baseline.md` (5823d634 + addenda 535d57ec, 80738627).

## Global Constraints

These come from CLAUDE.md, the spec and the owner.

- **No thresholds, gates, clamps or σ-floors as fixes.** Use model terms. A guard against degenerate geometry (division by ~0) is allowed and must be commented as such.
- **No fitted bounds.** A test bound is derived (stated before running) or the test is reported as failing. Never set a bound just above a measured value.
- **No new per-class constant** to make a test pass. A new constant must be class knowledge (in the manifest, with its source) or a measured sensor property (with the measurement file).
- **No view-change threshold:** ρ stays continuous.
- **Keep arXiv v1 rebuildable** (main 8a133295). The legacy engine stays selectable by config until grading.
- **Agent rates** stay at nominal. Report each port's update rate (Hz) against legacy.
- **Code style:**
  - C++23 containers and algorithms; `and` / `or` / `not`, never `&&`, `||`, `!`.
  - `std::from_chars` for parsing; `f.imbue(std::locale::classic())` on written data files.
  - Type-attributed DSR attributes only. A new attribute needs a cortex `REGISTER_TYPE` line and an owner reinstall, so avoid new attributes where an existing one serves.
- **Graph connections:** QueuedConnection only. Stop agents with SIGTERM, never `kill -9`.
- **Builds:** `-j8` max; check `pgrep -c cc1plus` first. Use `/usr/bin/ctest`, not `/usr/local/bin/ctest`.
- **Commits:** commit by explicit paths, ending each message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019DrdF9mYWK8PSUvLWeSAW1
  ```
  Push the feature branch after each task. Never merge to main without the owner's word.
- **Webots** is shared. Send CLAIM to room and chat, wait for ACK, and send FREE after. Objects has first call until about 10 Dec.
- **Latent layout per class:**
  - chain (3), camera shared (N), camera fresh (N), camera shape (N);
  - when the class has the beam channel, also LiDAR shared (N), LiDAR fresh (N), LiDAR shape (N).
  - K = 3 + 6N with the beam, 3 + 3N without. All four ported classes and the countables use the beam, so K = 3 + 6N.
  - The current countable K = 27 has no shape latents. It becomes 39, by adding kSc = 27 and kSL = 33.

## Review Focus

These are the input classes most likely to bite, and the task that must test each:

1. **Engine set to "persistent" on a class whose beam config is empty (no LiDAR SensorModel).** Expect the camera-only path, logged once, no NaN. Tested in Task 4 (`engine_no_beam_test`).
2. **A frame with zero mask points and zero rays** (fully occluded). Expect predict only: σ grows by the latents' prediction only, and the θ mean is unchanged. Tested in Task 4.
3. **A first update with a degenerate geometry** (sensor at the object centre, z → 0, or w⊥ → 0). Expect ρ = 1 and a finite variance, via the existing degenerate-geometry guard. Tested in Task 2.
4. **A return to the same standpoint after a long absence.** Expect a small promised gain and a small delivered gain, both under the same cap. Tested in Task 7 (`offer_return_view_test`).
5. **A neighbour instance whose published std is missing or zero.** Expect the neighbour to be skipped in the foreign density, never a zero-width density. Tested in Task 6.

---

## Phase 0: facts before code

### Task 1: Step-0 measurements

These are measurements only. No agent code changes.

**Files:**
- Create: `docs/superpowers/results/2026-11-24-step0.md`
- Create: `tools/mask_error/measure_mask_error.py` (new dir at repo root `tools/mask_error/`)
- Read: `room_concept/estimation_scripts/scripts/nees_agents.py`, `room_concept/estimation_scripts/auro_1122/claim_windows.md`, the run dirs `room_concept/estimation_scripts/crossing/objects_live14fixA`, `objects_live15birth`, branch `feat/mask-dump` (`retina` `[ZedDump] dump_masks`, `retina/tools/mask_figure.py`)

**Interfaces:**
- Produces three things:
  - The measured sensor constants, each with its file: `sigma_px_boundary` (shared-per-frame mask boundary offset, px), `eps_px` (pixel error rate), and `tau_cam_s` / `tau_lidar_s` (capture-to-pose latency, s). They are written in the results doc as a table, `detector | class | value | n frames | source`.
  - The refusal cause for G6.
  - The countables' NEES on the persistent path.

- [ ] **Step 1: NEES on the persistent path.** Run `python3 room_concept/estimation_scripts/scripts/nees_agents.py room_concept/estimation_scripts/crossing/objects_live14fixA room_concept/estimation_scripts/crossing/objects_live15birth`. Record the first-visit NEES/DOF for bed and nightstand. If the script needs a `WBT_PATH`, use the world those runs used (11-22 19:26; read `agents.txt` and the run's `graph_trace.csv` header for the world).
- [ ] **Step 2: Toilet on the persistent path.**
  - CLAIM Webots, then run one tour that ends in the bathroom (room_3):
    ```
    ON_FLEET_UP="CONCEPT_WT=<worktree> CONCEPT_AGENTS='nightstand_concept bed_concept toilet_concept' bash <worktree>/room_concept/estimation_scripts/scripts/concept_agents.sh start <out>" N_LEGS=2 LOOK_S=60 bash room_concept/estimation_scripts/scripts/warm_start_tour.sh objects_step0 room_concept/estimation_scripts/crossing/fixtures/three_rooms_door0
    ```
  - Score it with nees_agents.py and record the toilet's first-visit NEES/DOF. FREE after.
- [ ] **Step 3: Claim-refusal cause.** From the AuRo c3a and c3b runs listed in `auro_1122/nees_summary.md`, grep controller.log for refused epistemic claims. For each, compare the offered standpoint with the robot footprint at that time (`slam_shadow.csv`) and with residual_concept's `grid` node, if the graph trace has it. Write the count by cause: inside the robot footprint, occupied cell, or other.
- [ ] **Step 4: Mask error from dumps.**
  - Write `measure_mask_error.py <dump_dir> <wbt> <slam_shadow.csv>`. For each dumped ZED mask of a graded class:
    - project the graded-layer box (spec "Graded quantities", mesh extents) through the ZED intrinsics at the frame's ground-truth pose;
    - compute the signed contour offset in px between mask and projected silhouette, per contour sample;
    - output per frame: the mean offset (the shared part) and the residual std (the independent part).
  - Then `sigma_px_boundary` = std over frames of the per-frame mean offset, and `eps_px` = the mean fraction of mask pixels outside the projected silhouette.
  - Use existing dumps if any exist. Otherwise enable `[ZedDump] dump_masks` in retina's config for one tour (CLAIM first) and revert the config after.
- [ ] **Step 5: Latency.** From the same tour's logs, `tau_cam_s` = the median of (pose timestamp used − image capture timestamp), and `tau_lidar_s` likewise for the sweep. Countable's fitter logs both stamps in `<class>_ai2_log.csv` if present; otherwise use retina/room logs. Name the columns used.
- [ ] **Step 6: Write `2026-11-24-step0.md`** with the five results and commit:
  ```bash
  git add docs/superpowers/results/2026-11-24-step0.md tools/mask_error/measure_mask_error.py
  git commit -m "results: step 0 — persistent-path NEES, claim refusals, mask error, latencies"
  ```

## Phase 1: the engine (countable as its first model, no behaviour change)

### Task 2: Viewpoint and sensor-variance headers in common

**Files:**
- Create: `common/ai_belief/viewpoint.h`, header-only, moved from `countable_concept/src/countable_viewpoint.{h,cpp}`. Namespace `rc::ai::view`, same function names and signatures.
- Create: `common/ai_belief/sensor_var.h`
- Modify: `countable_concept/src/countable_viewpoint.h` becomes a shim (`#include "../../common/ai_belief/viewpoint.h"` plus `namespace rc { using namespace rc::ai::view; }`). Delete `countable_viewpoint.cpp` and drop it from `countable_concept/src/CMakeLists.txt` and `tests/CMakeLists.txt`.
- Test: create `common/ai_belief/tests/CMakeLists.txt` and `common/ai_belief/tests/sensor_var_test.cpp`; move `countable_concept/tests/viewpoint_rho_test.cpp` to `common/ai_belief/tests/`.

**Interfaces:**
- Produces in `sensor_var.h`, namespace `rc::ai::sensor`:
  - `double camera_shared_var(double sigma_px, double z_m, double f_px, double cos_obliquity)`. It returns (σ_px · z / f / cos_obl)². The 1/cos factor uses the degenerate-geometry guard `max(cos_obliquity, 1e-3)`, commented as numerical, not tuning.
  - `double fresh_var(double speed_mps, double tau_s)`. It returns (v · τ)².
- Produces `viewpoint.h`: `rho_camera`, `rho_lidar_lateral`, `rho_lidar_height`, `accumulate_path`, `ChainTracker`, `chain_*`, `pose_cov_in_room`, `chain_transport`, `ChainLever`, `lever_*`, with signatures unchanged.

- [ ] **Step 1: Failing test `sensor_var_test.cpp`.**
  - `camera_shared_var(1.0, 2.0, 700.0, 1.0)` equals `(2.0/700.0)²` within 1e-12.
  - At cos 0.5 the value is exactly 4× the cos-1 value.
  - `fresh_var(0.5, 0.04)` = 4e-4.
  - `fresh_var(0, τ)` = 0.
  - `camera_shared_var(σ, z, f, 0.0)` is finite.
- [ ] **Step 2: Configure, build and run.**
  ```
  cmake -S common/ai_belief/tests -B common/ai_belief/tests/build
  make -C common/ai_belief/tests/build -j8
  cd common/ai_belief/tests/build && /usr/bin/ctest
  ```
  Expected: FAIL (header missing).
- [ ] **Step 3: Implement both headers.** Moving viewpoint code must not change a single expression.
- [ ] **Step 4: Run both test dirs.** Run common/ai_belief/tests and `countable_concept/tests` (`cmake -B build && make -j8 && /usr/bin/ctest`). Expected: all pass, with countable at 26 tests after the viewpoint test moves out, plus the new common tests.
- [ ] **Step 5: Commit** the new files, the shim, the CMake edits and the moved test.

### Task 3: Engine extraction with a golden equivalence test

**Files:**
- Create: `common/ai_belief/persistent_engine.h`, the model concept and the engine class template.
- Create: `common/ai_belief/beam_channel.h`, the beam staging and ray accumulation now in `countable_belief.cpp`: `beam_parts` consumers, `stage_rays`, the per-ray loop over `beam_ray.h`, and the gap rays. It is generic over `std::span<const beam::BeamPart>` and the θ-Jacobians the parts give.
- Modify: `countable_concept/src/countable_belief.{h,cpp}`. `CountableBelief` becomes a model satisfying `rc::ai::engine::PersistentModel` and holds a `rc::ai::engine::PersistentEngine<CountableModel>`. Delete the moved code. Keep the public API the fitter and tests use (`update`, `latent_std`, `dbg_gn_steps`, `dbg_gn_fd`, `set_birth_position_cov`, `predict_latents_for_test`, and the rest) as forwarding members.
- Test: create `countable_concept/tests/engine_golden_test.cpp` and `countable_concept/tests/golden/engine_golden.bin`.

**Interfaces:**
- **`PersistentModel` concept.** A model `M` provides:
  - `static constexpr int N`, `using State`, `using Frame`;
  - `State State::from_vec(const Eigen::Matrix<float,N,1>&)` and `Eigen::Matrix<float,N,1> State::vec() const`;
  - the camera-point hooks with the signatures in `recursive_laplace.h`: `sdf_prim`, `sdf_jacobian`, `responsibilities`, `n_prims`;
  - `void accumulate_extra(const State&, const Frame&, Eigen::Matrix<float,N,N>& I, Eigen::Matrix<float,N,1>& b) const`, the θ-only channel;
  - `std::vector<rc::ai::beam::BeamPart> beam_parts(const State&) const`;
  - `Eigen::Matrix<double,N,3> chain_jacobian(const State&, const Eigen::Vector2f& lever) const`, how a robot pose error (δx, δy, δψ) moves θ;
  - `void apply_constraints(State&) const`, `void canonicalize(State&) const`;
  - `Eigen::Matrix<float,N,1> shape_spread() const` (Task 5; returns zeros until then).
- **`PersistentEngine<M>`:** `static constexpr int K = 3 + 6*M::N`. Latent offsets as constexpr members: `kChain = 0`, `kCam = 3`, `kL = 3+N`, `kFc = 3+2N`, `kFL = 3+3N`, `kSc = 3+4N`, `kSL = 3+5N`. For N = 6 this gives 0, 3, 9, 15, 21, 27, 33, matching today's countable offsets.
  - Methods:
    - `float update(M& m, const typename M::Frame& f, const FrameLatents<M::N>& fl)`;
    - `void predict(float q_scale)`;
    - `const Eigen::Matrix<double,M::N+K,M::N+K>& P() const`;
    - `Eigen::Matrix<float,M::N,M::N> Sigma_theta() const`;
    - the GnTrace accessors, moved as they are.
- **`FrameLatents<N>`:** the per-frame inputs now on `CountableFrame`: `rho_cam`, `rho_L`, `sig2_cam`, `sig2_L`, `fresh_c`, `fresh_L` (`std::array<float,N>` each), `sigma_pose` (Matrix3d), `chain_dr`, `robot_xy` (Vector2f), `camera_measured`, `lidar_measured` (bool).

- [ ] **Step 1: Record the golden file before touching code.**
  - Write `engine_golden_test.cpp --record`. Using `beam_fixture.h` and `test_scene.h`, it drives the CURRENT `CountableBelief`:
    - a birth;
    - 40 updates: camera-only, LiDAR-only and both, with a parked stretch and a moving stretch;
    - one mode rotation.
  - After each update it dumps θ, the diagonal of P and the GN step count to `golden/engine_golden.bin`.
  - Run it with `--record` and commit the bin with the test. The default mode (no flag) compares against the file.
- [ ] **Step 2: Extract** the engine and beam channel. Make `CountableBelief` the first model.
- [ ] **Step 3: Golden comparison.** Build and run `engine_golden_test`. Expected: every θ and P-diagonal value equals the recorded one within 1e-9 relative (double path), and step counts are equal. A mismatch is a bug, not a bound to widen.
- [ ] **Step 4: Full countable suite** with `/usr/bin/ctest` in `countable_concept/tests/build`. Expected: all pass.
- [ ] **Step 5: Rate check.** Build the agent (`countable_concept`, `cmake -B build && make -C build -j8`) and run the bench target if one exists; otherwise note "rate checked live in Task 10". Commit.

### Task 4: Engine robustness tests (Review Focus 1, 2)

**Files:**
- Test: create `common/ai_belief/tests/engine_no_beam_test.cpp`, `common/ai_belief/tests/engine_predict_only_test.cpp`, and `common/ai_belief/tests/toy_model.h` (a 1-box N = 6 model used by every engine test).

**Interfaces:**
- Consumes: `PersistentEngine`, `PersistentModel`, `FrameLatents` (Task 3).
- Produces: `toy_model.h` with `struct ToyBox` satisfying `PersistentModel`, plus `ToyBox::Frame make_frame(points, rays)`.

- [ ] **Step 1: Write the tests.**
  - (a) An empty beam config with camera frames: θ converges toward the toy truth, no NaN, and exactly one stderr line containing "beam not ready".
  - (b) A frame with zero points and zero rays: θ is bit-identical before and after, and diag P grows only by the latents' predicted variance (computed independently in the test from ρ and σ²).
- [ ] **Step 2: Run, fix any engine defect found, run again.** Commit.

## Phase 2: the engine items (G1 inputs, G4, G3, G5, G6)

### Task 5: Shape latent (G4) and sensor-derived latent variances (G1 inputs)

**Files:**
- Modify: `common/ai_belief/persistent_engine.h`. The kSc and kSL blocks are predicted with ρ = 1 and marginal `shape_spread()²/12`; the camera selector adds identity on kSc, and the LiDAR selector on kSL.
- Modify: `common/concept_manifest/` (the loader that reads `*.concept.toml`). Add the per-DOF key `shape.layer_spread_m` (array, length N) with a mandatory `shape.layer_spread_source` string.
- Modify: `bed_concept`, `nightstand_concept` and `toilet_concept` manifests. Δ comes from typical real objects. Each value carries its source (a standard dimension table, e.g. mattress vs frame width for a double bed: the frame is 10–20 cm wider than the mattress; cite the table used). If a DOF has no layering, Δ = 0.
- Modify: `countable_concept/src/countable_fitter.cpp:1306-1316`. `sig2_cam` = `rc::ai::sensor::camera_shared_var(sigma_px_boundary, z, f, cos_obl)` plus the existing pose-chain share. `fresh_c` / `fresh_L` = `fresh_var(v, tau)`. The values `sigma_px_boundary` and `tau_*` come from Task 1, set in config under `[Sensor]` with a comment naming the step-0 file. The keys `AI2RangeNoise*`, `MotionCm*`, `AI2PeriphRef` and `AI2AngLeverM` are removed from the three class configs and from the loader.
- Test: create `common/ai_belief/tests/engine_shape_latent_test.cpp` and `common/ai_belief/tests/parked_synthetic_test.cpp`.

**Interfaces:**
- Consumes: `toy_model.h` (Task 4), `sensor_var.h` (Task 2).
- Produces: the manifest accessor `rc::manifest::layer_spread(const Manifest&) -> std::vector<float>`.

- [ ] **Step 1: `engine_shape_latent_test`.**
  - The toy box has Δ_w = 0.20 m. The camera sees a 1.57 m width and the LiDAR a 1.77 m width (two layers), with 200 updates from 4 standpoints.
  - Assert the posterior σ_w ≥ 0.20/√12 · 0.9: the shape share cannot be averaged away. The 0.9 is the solver tolerance argued in the test comment from the stop rule 0.1σ, not fitted.
  - Assert |w − 1.67| ≤ 3σ_w.
- [ ] **Step 2: `parked_synthetic_test` (the spec's parked test, offline).** Toy box, camera only, Δ = 0.
  - (i) 200 frames at standpoint A: σ_cx after frame 200 ≥ 0.95 × σ_cx after frame 20. The plateau value is predicted in the test from Σ_cam alone.
  - (ii) Then 200 frames at B, 45° round the object at the same range: the drop factor in σ_cx matches the analytic two-view value from ρ_AB = `rho_camera` within 5 %. 5 % is the stop-rule tolerance.
  - (iii) Back at A: the delivered entropy drop ≤ the B drop.
- [ ] **Step 3: Run (fail), implement, run (pass).** Then run the countable suite and update the golden file only if the countable shape spread is non-zero. Re-record with `--record`, and state in the commit message that the golden change is the shape latent.
- [ ] **Step 4: Commit.** Push.

### Task 6: Mask error model (G3) and neighbour foreign density

**Files:**
- Modify: `common/ai_belief/persistent_engine.h` and `beam_channel.h`. The per-point and per-ray foreign density adds the neighbours' boxes: a `std::span<const ForeignBox>` passed in `FrameLatents`. Each neighbour contributes a density spread by its own published std.
- Modify: `countable_concept/src/countable_fitter.cpp`. Fill `FrameLatents::foreign_boxes` from the graph each cycle, for every other object node in the same room. Use the existing pose and size attributes and the std attributes; skip any node whose std is missing or zero (Review Focus 5).
  - `MaskPixelErrorRate` = the Task 1 `eps_px`.
  - The patch rows `AI2TopNoFloatPrecision`, `AI2TopNoFloatMarginM`, `AI2TopOversegSigmaPerM`, `AI2ClutterFrac`, `AI2ClutterScaleM` and `SdfThresholdForStorage` are removed from the three countable configs and from the loader. Remove each only after the test below passes without it. A row that cannot go is kept and listed in the commit message.
- Test: create `common/ai_belief/tests/engine_neighbour_test.cpp`.

**Interfaces:**
- Produces: `struct ForeignBox { Eigen::Vector2f c; float H, w, h, yaw; Eigen::Matrix<float,6,1> std; };` and the field `std::vector<ForeignBox> foreign_boxes` in `FrameLatents`.

- [ ] **Step 1: Write the test.**
  - The toy nightstand (0.51 m) has a mask cloud merged with a bed slab (as in `birth_footprint_test` case 1), with the bed's belief given as a ForeignBox. Assert the nightstand's h stays within 3σ of 0.51 and its centre within 3σ of the truth after 50 updates.
  - The same scene with a ForeignBox whose std is all zero: no NaN, and the result is the same as without that neighbour.
- [ ] **Step 2: Run (fail), implement, run (pass).** Run the countable suite. Commit and push.

### Task 7: Offers under the same cap (G5), with a promise/delivery log

**Files:**
- Modify: `common/ai_belief/persistent_engine.h`. Add `double predicted_gain(const M& m, const Standpoint& s) const`: the θ-marginal entropy drop of a hypothetical update at standpoint `s`.
  - Predict the latents with ρ computed along the straight path from the last update pose to `s` (`accumulate_path` over 10 equal steps).
  - Synthesise the camera points and rays the standpoint would give from `beam_parts` and the camera model.
  - Apply `joint_cov`. Gain = ½ log det Σθ_before − ½ log det Σθ_after.
- Modify: `common/nbv/countable_planner.h`. The `raw_gain_of` callback is fed by `predicted_gain`, and `d_optimal_gain` stays for the legacy engine.
- Modify: `countable_concept/src/epistemic_planner.cpp`. Use the engine's gain when `Model.Engine == "persistent"`.
  - Write `<class>_offers.csv` with the columns `t_ms, node, standpoint_x, standpoint_y, standpoint_yaw, promised_nats` at publish.
  - Write `<class>_delivered.csv` with the columns `t_ms, node, delivered_nats` after each update: the θ-marginal entropy drop of that update.
  - Use the classic locale.
- Create: `room_concept/estimation_scripts/scripts/promise_delivery.py <run>`. It joins offers to the updates within the claim window (same rule as `auro_1122/claim_windows.md`) and prints the per-class Spearman and the spread of promises.
- Test: create `common/ai_belief/tests/offer_return_view_test.cpp`.

**Interfaces:**
- Produces: `struct Standpoint { Eigen::Vector2f xy; float yaw; };` and `PersistentEngine::predicted_gain`.

- [ ] **Step 1: Test (Review Focus 4).**
  - The toy box is updated 100 frames at A.
  - Assert `predicted_gain(A) ≤ r · predicted_gain(B) · 1.05` for B at 90° round the object. r is the analytic ratio, computed in the test from ρ_AA = 1 against ρ_AB = `rho_camera` for that geometry. 1.05 is the stop-rule tolerance.
  - Then do one update at A and assert the delivered gain ≤ the predicted gain at A, times 1.05.
- [ ] **Step 2: Self-test ranking.** The chair's and the table's `self_test` NBV face-ranking checks must still pass once those classes are ported (Tasks 11, 12). Here, add the same ranking check for the toy box: a box seen only from its long face ranks the short face first.
- [ ] **Step 3: Run (fail), implement, run (pass).** Commit and push.

### Task 8: Occupancy check on offered standpoints (G6)

**Files:**
- Create: `common/nbv/standpoint_free.h` with `double p_footprint_free(const GridView& g, const Eigen::Vector2f& xy, float yaw, std::span<const Eigen::Vector2f> footprint_poly)`. It is the product over grid cells under the footprint of (1 − p_occ). A cell not in the grid uses the grid's prior.
- Modify: `common/nbv/countable_planner.h`. An offer's value is multiplied by `p_footprint_free`. The footprint polygon comes from `common/robot_footprint`, a robot property.
- Modify: `countable_concept/src/specificworker.cpp`. Read residual_concept's `grid` node (the same reader the birth-surprise probe uses) into `GridView` each planning cycle.
- If Task 1 Step 3 found the main refusal cause to be "inside the robot's own footprint" and not occupancy, also score the robot's current footprint as occupied for this check. It is a fact about the robot, not a threshold. Write which case applied in the commit message.
- Test: create `common/nbv/tests/standpoint_free_test.cpp` (add to `common/nbv` tests CMake, or create one).

**Interfaces:**
- Produces: `struct GridView { float origin_x, origin_y, res_m; int nx, ny; std::span<const float> p_occ; float p_prior; };` and `p_footprint_free`.

- [ ] **Step 1: Test.**
  - An all-free grid gives 1 within 1e-6.
  - One occupied cell (p = 1) under the footprint gives 0.
  - A footprint half outside the grid with p_prior = 0.5 gives the product of the inside cells times 0.5^(outside cells).
- [ ] **Step 2: Run (fail), implement, run (pass).** Commit and push.

## Phase 3: ports

Each port adds `Model.Engine` to the class config, defaulting to `"persistent"` on this branch, with `"legacy"` keeping today's path bit for bit. Each port turns the class belief into a `PersistentModel`, fills `FrameLatents` in the class fitter (copy the countable fitter's ρ / σ / chain code; do not reinvent it), and moves its offers to `predicted_gain`. Each port removes the class's registered-prediction rows (spec table) that its engine items cover. It lists, in the commit message, every row removed and every row kept with the reason.

### Task 9: Graded-quantity ground truth and scorers

**Files:**
- Create: `tools/gt_mesh/mesh_extents.py <wbt> <DEF>`. It prints the graded layer's centre, extents and yaw, read from the proto's visual meshes and named per class in the spec table: bed mattress, nightstand body, toilet bowl + tank footprint, chair seat, table top, fridge body, hood canopy.
  - The hood's Webots origin is verified here: print the proto origin against the mesh centroid.
- Modify: `countable_concept/tools/lidar_measures_score.py` and `room_concept/estimation_scripts/scripts/nees_agents.py`. Use mesh-extent ground truth by default and add a `--gt boundingobject` column for the old numbers.
- Test: `tools/gt_mesh/test_mesh_extents.py`. For the bed in the current world, the mattress width must be 1.571 ± 0.001 (measured 11-22 from Bed.proto).

- [ ] **Steps:** failing test → implement → pass → rescore live15birth with both ground truths into the step-0 doc → commit and push.

### Task 10: Refrigerator port, then hood port

**Files:**
- Modify: `refrigerator_concept/src/refrigerator_belief.{h,cpp}`, `refrigerator_fitter.cpp`, `refrigerator_config.{h,cpp}`, `etc/config.toml`, the manifest `shape.layer_spread_m`.
- Then the same for `hood_concept/`.
- `beam_parts`: one box from θ (cx, cy, H, w, h, yaw), the same as the countable belief.
- The `lidar_ray_factor` use is replaced by the engine's beam channel.
- Test: `refrigerator_concept/tests/fridge_engine_test.cpp`, `hood_concept/tests/hood_engine_test.cpp`.

- [ ] **Step 1: Test per class.**
  - On a synthetic scene from `toy_model.h`'s scene builder with the class's own SDF: the persistent engine's mean after 60 updates is within 3σ of the truth.
  - The legacy engine with `Model.Engine = "legacy"` gives bit-identical output to before the port, checked by a golden file recorded first, as in Task 3.
- [ ] **Step 2: Port, run, then a live tour with fridge, hood and countables.** CLAIM, `concept_agents.sh` with `CONCEPT_AGENTS='refrigerator_concept hood_concept nightstand_concept bed_concept toilet_concept'`, then FREE. Report the rates (Hz) against legacy and first-visit NEES/DOF.
- [ ] **Step 3: Commit and push** per class.

### Task 11: Chair port, with a new Helios beam channel

**Files:**
- Modify: `chair_concept/src/chair_belief.{h,cpp}`, `chair_fitter.cpp`, `chair_config.{h,cpp}`, `etc/config.toml`, the manifest.
- `beam_parts`: seat, back and four legs as boxes from the chair template at θ.
- The LiDAR ray staging copies the countable fitter's (`lidar_select.h`) and needs the `lidar3D` media subscriber via `rc::media::make_lidar_subscriber_from_graph`. Create it on the main thread before the workers start, as the CLAUDE.md media-plane rules require.
- Test: `chair_concept/tests/chair_engine_test.cpp`. `ChairBelief::self_test` must still PASS, including its NBV face-ranking check, now computed with `predicted_gain`.

- [ ] **Steps:** golden for legacy → test → port → ctest → live tour with chair, report Hz against legacy → commit and push.

### Task 12: Table port

**Files:**
- Modify: `table_concept/src/table_belief.{h,cpp}`, `round_table_belief.h` (stays legacy unless trivially portable; say which), `table_fitter.cpp`, config, manifest.
- N = 7 in the quotient chart. `chain_jacobian` maps the pose error through the chart. `common_mode_inv` (the full 7 × 7) is no longer used by the engine path.
- `beam_parts`: top slab plus legs.
- Test: `table_concept/tests/table_engine_test.cpp`. `TableBelief::self_test` must still PASS.
- **If this task is not passing by 12-06,** keep the table on legacy, write that in the progress ledger, and tell the owner the same day.

- [ ] **Steps:** as in Task 11.

## Phase 4: parked test live, count and grading handoff

### Task 13: Live parked test

**Files:**
- Create: `room_concept/estimation_scripts/scripts/parked_test.sh <out_dir> <object_DEF>`. It brings the fleet up with the fixture, sends controller missions through `remote_cmd.txt` (`goto A; wait 60s; goto B; wait 60s; goto A; wait 60s`), and harvests logs. A and B are computed from the object's ground-truth pose: 1.5 m in front of it, and 45° round it at the same range.
- Create: `room_concept/estimation_scripts/scripts/parked_report.py <out_dir>`. It checks the parked-synthetic predictions on the live σ trace. The pass conditions are stated in the spec's "Tests" item 3, with bounds computed from the logged ρ and Σ, not fitted.

- [ ] **Steps:** CLAIM → run on one chair in room_1 → report → FREE → write `docs/superpowers/results/2026-12-XX-parked.md` → commit and push.

### Task 14: Constant count after part 1, and the grading handoff

**Files:**
- Create: an "after part 1" section appended to `docs/superpowers/results/2026-11-23-patch-constants-baseline.md`, using the same build-derived scope rule (addendum 2) on the same five targets, with the tree hash. Rows removed, moved (patch to sensor or class) and kept are listed against the registered-prediction table.
- Create: `docs/superpowers/results/2026-12-XX-part1-grading.md`, with the per-class first-visit NEES/DOF, planar error, promise/delivery Spearman and the count.

- [ ] **Steps:**
  - Recount.
  - Send room the build commit for the combined rerun batch, and the auro session the results paths.
  - Report a failing test as failing, never with a widened bound.
  - Commit and push.
