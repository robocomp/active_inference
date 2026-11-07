# door_concept existence model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace door_concept's patched existence logic with one belief per door place: priors entered once, per-view
Bernoulli likelihoods from reprojecting the door into the ZED and the ricoh, one view ledger, a costed decision, and a
memory row that *is* the belief.

**Architecture:** New DSR-free pieces under `door_concept/src/core/` (camera models, door visibility, detector model,
view ledger, place prior, decision and costs, place table). Each is header-first and unit-tested alone. Then
`DoorCore::update_existence` is rewritten on them, the worker's ghost/identity/seed paths are routed through the place
table, and the old code is deleted. The tracker (`common/instance_tracker`) is not modified. It still associates
candidates and enforces separation. The door model decides birth, and the tracker is told through `birth_evidence`.

**Tech Stack:** C++23, Eigen, OpenCV core/imgproc, the door_concept core (`DoorCycleInput` → `DoorCycleOutput`),
`tools/door_replay`, `tests/test_door_tapes.sh`, Python 3 for the two log-fitting tools.

**Spec:** `docs/superpowers/specs/2026-11-07-door-existence-model-design.md` (commit 5fca33e4; §8 holds the designer's
calls). Read it before any task. Section numbers below (§3.4 etc.) refer to it.

## Global Constraints

- C++23 containers/algorithms; `and`/`or`/`not`, never `&&`/`||`/`!`.
- No thresholds, clamps, σ-floors, streaks or timers. The only constants allowed are the ones §6 lists, each flagged
  in a comment where it is defined: detector envelope shoulders (fitted, persisted), view-kernel widths (the door's own
  scale), and the per-wall base rates **0.4** (first door on a wall) and **0.01** (each further door).
- Size prior: width **0.85 ± 0.12 m**, height **2.05 ± 0.06 m**. Score it only on the fitted size, never on raw mask height.
- Crossing evidence: `rc::door::crossing_logodds` (door_leaf_visibility.h:67), ≈ +2.3. It is not re-derived.
- Deletion line: delete iff `L < −ln(C_delete / C_keep)`. Creation line: `L > 0`. A door being crossed is never deleted.
- Parse numbers with `std::from_chars`. Write data files through `imbue(std::locale::classic())`. Any harness that
  parses agent files calls `std::setlocale(LC_ALL, "")` first.
- Builds: `systemd-run --user --scope -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G make -C build -j8`
  (`cbuild` is not available in agent shells). Check `pgrep -a cc1plus` first.
- Never `kill -9` an agent. Stop with SIGTERM.
- Commits stay local. Nothing is pushed until Pablo says so. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_019DrdFCnuwgeRc7peaKRUSv`.
- Before any Webots run, message peer session `shadow-bridge-dds-d6`. Tell it again when Webots is free.
- The agent must hold its nominal rate (memory: rates before features). Task 12 measures it.

## Review Focus

1. **Pose lost or relocalising.** κ → 0, so every channel contributes ≈ 0. No door is born or deleted on views taken
   while the robot does not know where it is. Test: Task 8, `relocalising_robot_moves_nothing`.
2. **An open leaf.** A real door swung open must not collect misses from a closed-panel model. Visibility and q are
   marginalised over the leaf-angle posterior. Test: Task 2, `open_leaf_vis_is_marginal`, and Task 8,
   `open_door_survives_face_on_views`.
3. **Ricoh `windows` empty (producer without the field).** Unknown is not "looked". The ricoh adds no miss evidence.
   Test: Task 2, `ricoh_unknown_windows_no_miss`.
4. **Old files.** v4 tapes, and identity CSV rows without the new columns, must load. A missing ricoh model means no
   ricoh misses. A missing ledger means an empty ledger with the row's L kept. Tests: Task 1, `v4_tape_reads_ricoh_absent`;
   Task 7, `old_row_loads_with_empty_ledger`.
5. **Room frames change under a door.** The proto-room is promoted, re-registered or swapped on recognition.
   Ledger keys are door-relative (bearing and range from the door), so they must survive a change of the room frame
   unchanged. Test: Task 4, `ledger_invariant_to_room_frame`.

---

## File Structure

| File | Responsibility |
|---|---|
| `door_concept/src/core/camera_model.h` (new) | Pinhole and equirectangular projection, line for line `DSR::CameraAPI::project` |
| `door_concept/src/core/door_visibility.{h,cpp}` (new) | Project a door model into a camera: vis, projected fill, aspect, pixel samples. Marginalised over φ |
| `door_concept/src/core/detector_model.h` (new) | q_c, f_c, exact Bernoulli hit/miss ℓ, persisted per-camera parameters |
| `door_concept/src/core/view_ledger.h` (new) | One record per viewpoint per door. Replaces ViewPool, RicohViewPool, RicohHitPool, the three DoorFitState pools |
| `door_concept/src/core/place_prior.h` (new) | Wall band (two-sided promoted, one-sided proto), per-wall base rate, size KL |
| `door_concept/src/core/door_decision.h` (new) | Role, costs, the create/keep/delete verdict, `CostEstimator` (running median, persisted) |
| `door_concept/src/core/place_table.h` (new) | One row per door place per room: the belief, the ledger, last seen, visit. Absorbs ghosts and identity existence |
| `door_concept/src/core/door_existence.cpp` (rewrite) | `update_existence` on the pieces above |
| `door_concept/src/core/door_cycle_io.{h,cpp}` | Tape v5: ricoh intrinsics, `body_T_ricoh`, route faces |
| `door_concept/src/input_snapshot.cpp` | Capture the v5 fields live |
| `door_concept/src/specificworker.cpp` | Carried door through the place table, ghost/identity code removed |
| `door_concept/tools/fit_detector_envelopes.py` (new) | Initial ZED/ricoh envelope shoulders from tour logs |
| `door_concept/tools/door_costs_from_logs.py` (new) | Initial C_keep / C_delete from tour logs |
| `door_concept/etc/door_detectors.csv`, `etc/door_costs.csv` (new) | Persisted, learned parameters with sample counts and source |

Build every unit test the way the existing ones are built. One `g++` line goes in the test's header comment, e.g.
`g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_view_ledger.cpp -o /tmp/test_view_ledger && /tmp/test_view_ledger`.
Use the CHECK macro pattern of `room_concept/tools/refine_doorway_test.cpp`. A test prints `N/M checks passed` and
exits nonzero on failure.

---

### Task 0: Baseline and branch

- [ ] **Step 1:** Create the worktree with superpowers:using-git-worktrees. Branch `feat/door-existence`, based on
  rt-build `feat/room-recognition` HEAD.
- [ ] **Step 2:** Run every `door_concept/tests/test_*.cpp` by its header g++ line, then `tests/test_door_tapes.sh`.
  Save the pass/fail list to `docs/superpowers/plans/2026-11-07-door-existence-baseline.txt`. These are the
  pre-existing results. Later tasks may change only the tests §5 lists as rewritten or retired.
- [ ] **Step 3:** Replay tour18 with the current code and save the REMOVE rows for door_4 and door_5:
  `door_replay crossing/explore_tour18/door_inputs.bin crossing/explore_tour18/door_decisions.bin --trace '*'`.
  This is the "before" for Task 11.
- [ ] **Step 4:** Commit the baseline file.

### Task 1: Camera models and tape v5

**Files:**
- Create: `src/core/camera_model.h`, `tests/test_camera_model.cpp`
- Modify: `src/core/door_cycle_io.{h,cpp}`, `src/input_snapshot.cpp`, `tests/test_door_cycle_io.cpp`

**Interfaces:**
- Produces:
  - `enum class Projection : std::uint8_t { Pinhole, Equirectangular, Cylindrical };`
  - `struct CameraSpec { Projection model; std::uint32_t width, height; float focal_x, focal_y; float az_sign = 1.f, az_offset = 0.f; };`
  - `[[nodiscard]] CameraSpec camera_spec(const CameraIntrinsics&);`
  - `[[nodiscard]] std::optional<Eigen::Vector2d> project(const CameraSpec&, const Eigen::Vector3d& p_cam);`
    Returns nullopt behind a pinhole (y ≤ 0) or at the 360 origin.
  - `CameraIntrinsics` gains `float equirect_az_sign = 1.f, equirect_az_offset = 0.f;`.
  - `DoorCycleInput` gains `CameraIntrinsics ricoh;` and `std::vector<std::string> route_faces;`.
  - `RobotState` gains `std::optional<Mat4d> body_T_ricoh;`.
  - Tape version 5.

- [ ] **Step 1: Write the failing tests.**
  - `pinhole_matches_camera_project`: `project` equals the existing `camera_project` (door_existence.h) on 100 random
    points with y > 0.
  - `equirect_matches_cameraapi`: for sign −1, offset 0, W=1920, H=960, check the formulas in
    `cortex/api/dsr_camera_api.cpp:112-125`. Points straight ahead (0,1,0) → (960, 480). Straight up (0,0,1) → v = 0.
  - `equirect_seam_wraps`: azimuth just past ±π lands in [0, W).
  - `v4_tape_reads_ricoh_absent` (in test_door_cycle_io): a v4 tape reads with `ricoh.present == false`, no
    `body_T_ricoh`, and empty `route_faces`.
  - `v5_roundtrip`: write then read a v5 input with every new field set. It compares equal.
- [ ] **Step 2:** Run them. Expected: compile failure (no `camera_model.h`).
- [ ] **Step 3: Implement.**
  - `project` is CameraAPI's arithmetic for all three models. The pinhole branch reuses `camera_project`'s centring
    exactly.
  - The v5 writer and reader append the new fields after v4's. The reader accepts versions 1–5.
  - `input_snapshot.cpp` fills the new fields:
    - `ricoh` from the `ricoh` node: `cam_width`, `cam_height`, `cam_fov`, `cam_projection`,
      `cam_equirect_azimuth_sign`, `cam_equirect_azimuth_offset`, read with type-attributed getters;
    - `body_T_ricoh` from the static RT `body → ricoh`;
    - `route_faces` from `rc::topo::crossed_doors` (common/ltsm_topology/topology_decode.h) on the `ltsm_topology`
      node. These are door names that are a live passage face with crossings ≥ 1.
- [ ] **Step 4:** Run the tests. Expected: all pass, and the existing test_door_cycle_io cases still pass.
- [ ] **Step 5:** Commit: `door_concept core: camera models (pinhole+equirect) and tape v5 (ricoh, route faces)`.

### Task 2: Door visibility for any camera

**Files:**
- Create: `src/core/door_visibility.{h,cpp}`, `tests/test_door_visibility.cpp`
- Modify: `src/core/door_existence.cpp`. Move the silhouette sampling out of `compute_silhouette_existence`
  (lines ≈249-520) into `door_view`, and make `compute_silhouette_existence` call it.

**Interfaces:**
- Consumes: `CameraSpec`, `project` (Task 1). `ExistenceScene` (door_existence.h). `los_weight`, `ricoh_looked`
  (ricoh_door_birth.h). `door_room_band.h`.
- Produces:
  ```cpp
  struct ViewGeometry {
      float vis = 0.f;        // E_phi[visible fraction of leaf+frame samples]
      float fill_h = 0.f;     // projected extent / image extent, horizontal
      float fill_v = 0.f;     // ... vertical; > 1 means the door overflows
      float aspect = 0.f;     // |cos| of the view angle to the aperture plane
      float range_m = 0.f;
      std::vector<cv::Point2f> samples;   // pixel samples of the modal rendering (for mask lookup)
      bool projectable = false;           // false => nothing is computed, every likelihood is 0
  };
  struct CameraView { const CameraSpec& cam; Mat4d room_T_cam; std::span<const Vec2f> looked_windows; bool windows_known; };
  [[nodiscard]] ViewGeometry door_view(const CameraView&, const ExistenceScene&, const rc::DoorFitState&);
  ```
- A sample counts toward `vis` when all of these hold:
  - it projects inside the image, or inside a looked window for the ricoh;
  - it is not behind a wall (own wall excluded);
  - it is not inside another door's cone;
  - it is not under a nearer non-door mask;
  - it is weighted by the band's beyond-far-face factor.

  This is today's silhouette rule, unchanged.
- **Ricoh with `windows_known == false`:** vis = 0. This is Review Focus 3, and a change from today, which counted
  every bearing as looked.

- [ ] **Step 1: Write the failing tests.** Use the scene of `test_door_core_existence.cpp`: a 10×10 room with a door
  in the top wall.
  - `zed_behind_vis_zero`: camera looking −y gives vis == 0.
  - `zed_around_wall_vis_zero`: door hidden by an inner wall gives vis == 0.
  - `zed_overflow_fill`: 0.6 m face-on gives fill_v > 1.
  - `ricoh_outside_window_vis_zero`: door bearing outside every looked window gives vis == 0.
  - `ricoh_unknown_windows_no_miss`: `windows_known=false` gives vis == 0.
  - `ricoh_face_on_1p4m`: equirect W=1920 H=960, door 0.85×2.05 m at 1.4 m face-on. vis ≈ 1, and fill_h, fill_v equal
    the angular spans over 2π and π. Print them, because Task 3 needs them.
  - `open_leaf_vis_is_marginal`: a φ posterior with mass at 0 and 80° gives vis between the two single-angle values,
    weighted by that mass.
  - `silhouette_unchanged`: `compute_silhouette_existence` on the existing test scene returns the same e_occ, e_free
    and n_detectable as before the move.
- [ ] **Step 2:** Run them. Expected: compile failure.
- [ ] **Step 3: Implement** `door_view` by moving the existing sweep (the `sweep` lambda and φ loop) behind the
  `CameraSpec` projection. No new rules.
- [ ] **Step 4:** Run the tests and `test_door_core_existence`. Expected: the new tests pass and existence outputs are
  unchanged.
- [ ] **Step 5:** Commit: `door_concept core: camera-generic door visibility (ZED + ricoh reprojection)`.

### Task 3: Detector model and initial envelopes

**Files:**
- Create: `src/core/detector_model.h`, `tests/test_detector_model.cpp`, `tools/fit_detector_envelopes.py`,
  `etc/door_detectors.csv`

**Interfaces:**
- Consumes: `ViewGeometry` (Task 2). `rc::detect::DetectorEnvelope` (common/detectability).
- Produces:
  ```cpp
  struct DetectorModel { rc::detect::DetectorEnvelope env; float hit_rate; float false_alarm; float n_hit_trials; };
  [[nodiscard]] float detect_prob(const DetectorModel&, const ViewGeometry&, float kappa);   // q_c
  [[nodiscard]] float hit_llr(float q, float f, float vis, float assoc_w);   // a·ln(q/(f·vis)); 0 if vis·q == 0
  [[nodiscard]] float miss_llr(float q, float f, float vis);                 // ln((1−q)/(1−f·vis))
  struct DetectorTable { DetectorModel zed, ricoh; };
  [[nodiscard]] DetectorTable load_detectors(const std::filesystem::path&);   // from_chars; missing file => compiled defaults
  void save_detectors(const std::filesystem::path&, const DetectorTable&);    // classic locale
  ```
- Definitions:
  - `q = env(fill_h, fill_v) · aspect · vis · kappa · hit_rate`. `env` uses both axes: the product of the envelope at
    each fill.
  - f is the false-alarm rate at that place, `false_alarm · vis`.
- CSV format: `camera,param,value,n,source`.

- [ ] **Step 1: Write the failing tests.**
  - `miss_zero_when_unseen`: `miss_llr(0, f, vis) == 0` exactly, and `miss_llr(q, f, 0) == ln(1−q)`. Then assert vis
    == 0 implies q == 0 through `detect_prob`, which makes the miss exactly 0.
  - `miss_monotone`: miss_llr falls as q rises.
  - `aspect_70deg_1m_and_6m`: ZED defaults, door face-on vs 70°, at 1 m and 6 m. The sum of 10 miss-frames' ℓ, taken
    as ONE view (Task 4 pools them; here call miss_llr once), is smaller in magnitude than one hit-view at the same
    geometry. This is spec test 4.
  - `ricoh_1p4m_overflow`: ricoh model loaded from the committed `etc/door_detectors.csv`, with the geometry
    `ricoh_face_on_1p4m` printed in Task 2. `|miss_llr| < 0.1`. This is spec test 5.
  - `csv_roundtrip_es_locale`: save then load under `setlocale(LC_ALL,"")` with es_ES. Values are equal.
- [ ] **Step 2:** Run them. Expected: compile failure.
- [ ] **Step 3: Write `tools/fit_detector_envelopes.py`.**
  - Read every `crossing/explore_tour*/door_existence_log.csv` and `ai2_log.csv`.
  - For each ZED and ricoh look of a door that was alive with L > 0 for its whole life, and later crossed (a door known
    to be real), recompute fill_h and fill_v from range, door size and the camera's projection. Take the ricoh mount
    height from `robot_concept/shadow.json`.
  - Fit the envelope's near and far shoulders by maximum likelihood of hit/miss.
  - Write `etc/door_detectors.csv` with n and `source=fit_detector_envelopes.py explore_tour10-22 <date>`.
  - Print the fitted shoulders and the q this gives at tour18's door_5 view (1.4 m).
  - If the fit gives q ≥ 0.3 at that view, STOP and report. The §8.4 claim that the near shoulder alone explains
    tour18 would then be false, and the PI must hear it before going on.
- [ ] **Step 4: Implement `detector_model.h`.** Compiled defaults are today's values: the ZED `aperture_envelope`
  shoulders, and the ricoh's `rc::birth::Detectability{ai2_periph_ref, 2.5, 2.0}` with hit_rate 0.5 and clutter 0.05.
  Each default is flagged as a sensor-model constant, refit from data.
- [ ] **Step 5:** Run the tests. Expected: pass.
- [ ] **Step 6:** Commit: `door_concept core: per-camera detector model, exact Bernoulli likelihoods, fitted envelopes`.

### Task 4: The view ledger

**Files:**
- Create: `src/core/view_ledger.h`, `tests/test_view_ledger.cpp` (replaces `tests/test_door_view_pool.cpp`, deleted in
  Task 10)

**Interfaces:**
- Produces:
  ```cpp
  enum class Channel : std::uint8_t { Zed, Ricoh, Geometry, Actuation };
  struct ViewKey { float bearing = 0.f; float range = 0.f; int visit = 0; };   // door-relative
  struct BernoulliAgg { double n = 0, sum_q = 0, sum_hit_w = 0, sum_f = 0, sum_vis = 0; };
  struct GaussAgg { double n = 0, sum_x = 0, sum_w = 0; float mu0 = 0, mu1 = 0, s2 = 1; };
  struct View { ViewKey key; float weight_n = 0;
                std::array<BernoulliAgg, 2> bern;   // Zed, Ricoh
                GaussAgg geo; float actuation_llr = 0; float llr = 0; };
  class ViewLedger {
  public:
      // locate: kernel k = exp(−½[(Δb/α)² + (Δr/R_eff)²]), α = atan(R_eff/d), R_eff = hypot(R, spread_m); same visit only.
      // Frames are absorbed into the best view by k; a new view is opened with weight 1−k (no constant).
      std::size_t observe(const Vec2f& cam_xy, const Vec2f& door_xy, float R, float spread_m, int visit);
      void add_bernoulli(std::size_t v, Channel c, float q, float f, float vis, float hit_w);
      void add_geometry(std::size_t v, float x, float w, float mu0, float mu1, float s2);
      void add_actuation(std::size_t v, float llr);
      float recompute(std::size_t v, bool geometry_may_refute);   // returns the CHANGE in the ledger's sum
      [[nodiscard]] float sum() const;
      [[nodiscard]] std::span<const View> views() const;
      [[nodiscard]] std::string serialise() const;               // for the place row
      static std::optional<ViewLedger> parse(std::string_view); // from_chars
  };
  ```
- A view's ℓ is recomputed from its aggregate:
  - each Bernoulli channel: `d̄·hit_llr(q̄, f̄, v̄, 1) + (1−d̄)·miss_llr(q̄, f̄, v̄)`, with `d̄ = sum_hit_w / n`;
  - geometry: `w̄·[ln N(x̄;μ1,s²) − ln N(x̄;μ0,s²)]`, passed through `max(0,·)` when `geometry_may_refute` is false
    (§8.5);
  - plus `actuation_llr`.

  The view's previous ℓ is replaced, never added to.

- [ ] **Step 1: Write the failing tests.**
  - `parked_stare_is_one_view`: 200 frames (20 Hz × 10 s) from one standpoint, all misses at q=0.6. Expect
    `views().size() == 1`, and `sum()` equals one miss_llr at the same q to 1e-5.
  - `new_standpoint_adds_view`: a second standpoint 60° around the door adds one view, and `sum()` doubles.
  - `pose_spread_merges`: with spread_m = 4 the two standpoints are one view.
  - `hit_then_miss_same_view_is_mean`: in one view, 5 hit frames then 5 miss frames give one trial at d̄ = 0.5.
  - `geometry_up_only`: a negative geometry statistic contributes 0 with `geometry_may_refute=false`, and its
    negative value with true.
  - `ledger_invariant_to_room_frame`: apply the same rigid transform to cam_xy and door_xy. Views and sum are
    unchanged.
  - `serialise_roundtrip_es_locale`.
  - `revisit_replaces_not_adds`: parse a ledger, then observe the same standpoint in the same visit. The sum is
    unchanged. With visit+1, a new view opens.
- [ ] **Step 2:** Run them. Expected: compile failure.
- [ ] **Step 3: Implement `view_ledger.h`.**
- [ ] **Step 4:** Run the tests. Expected: pass.
- [ ] **Step 5:** Commit: `door_concept core: one view ledger per door (replaces the seven pools)`.

### Task 5: The place prior

**Files:**
- Create: `src/core/place_prior.h`, `tests/test_place_prior.cpp`

**Interfaces:**
- Consumes: `band_log_prior`, `RoomBand` (door_room_band.h). `aperture_band_offset` (door_existence.h).
- Produces:
  ```cpp
  struct PlacePriorInput { const std::vector<Vec2f>& polygon; bool proto; rc::door::RoomBand band;
                           Vec2f aperture_xy; Vec2f wall_dir; float width, height, width_sd, height_sd; bool size_resolved;
                           int doors_on_wall_before; };
  struct PlacePrior { float band = 0, base_rate = 0, size = 0; float total() const { return band + base_rate + size; } };
  [[nodiscard]] PlacePrior place_prior(const PlacePriorInput&);
  [[nodiscard]] std::vector<int> wall_groups(const std::vector<Vec2f>& polygon, float sigma_m);   // edge → wall id
  ```
- The terms:
  - `base_rate`: `ln(0.4/0.6)` when `doors_on_wall_before == 0`, else `ln(0.01/0.99)`.
  - `band`, promoted room: `band_log_prior`, two-sided.
  - `band`, proto room (designer's call, §3.2 and §7: "frontier, not a wall"): one-sided. An aperture inside the
    measured floor pays the band term. One at or beyond the outline pays 0.
  - `size`: −KL(N(w, width_sd²) ‖ N(0.85, 0.12²)) − KL(height likewise, 2.05 ± 0.06), only when `size_resolved`.
- `wall_groups` (designer's call, §7): consecutive edges join one wall when their unit normals agree and their line
  offsets differ by less than 2σ of the band. That 2σ is a χ² gate, flagged in the code with the reason: two edges are
  one wall exactly when their offsets are indistinguishable under the face uncertainty the band already carries.

- [ ] **Step 1: Write the failing tests** (spec test 8, prior part).
  - `typical_doors`: 0.80 m and 1.00 m first doors on a promoted wall get total within 0.5 nats of ln(0.4/0.6).
  - `wide_candidate_pays`: 3 m pays at least 10 nats more than 0.85 m.
  - `off_wall_floor`: an aperture 1.5 m inside a promoted room gets total < −10.
  - `second_door`: the base rate equals ln(0.01/0.99).
  - `proto_frontier`: an aperture 0.5 m beyond a proto outline pays band 0. 0.5 m inside it pays the same as promoted.
  - `jogged_wall_is_one_group`: an L-jog of 5 cm on one Manhattan wall is one group, and a real corner is two.
  - `unresolved_size_pays_nothing`.
- [ ] **Step 2:** Run them. Expected: compile failure.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run. Expected: pass.
- [ ] **Step 5:** Commit: `door_concept core: place prior (band, per-wall base rate, size KL)`.

### Task 6: Decision, roles and costs

**Files:**
- Create: `src/core/door_decision.h`, `tests/test_door_decision.cpp`, `tools/door_costs_from_logs.py`,
  `etc/door_costs.csv`

**Interfaces:**
- Produces:
  ```cpp
  enum class Role : std::uint8_t { Ordinary, Route, BeingCrossed };
  struct Costs { float keep_s; float delete_ordinary_s; float delete_route_s; };
  struct Verdict { bool exists; bool remove; float delete_line; };
  [[nodiscard]] Verdict decide(float L, Role, const Costs&);
      // exists = L > 0; delete_line = −ln(C_delete(role)/keep_s); remove = role != BeingCrossed and L < delete_line
  [[nodiscard]] Role door_role(const std::string& name, std::span<const std::string> route_faces, bool being_crossed);
  class CostEstimator {   // running median per kind, every sample kept (they are few), persisted
  public:
      enum class Kind : std::uint8_t { Keep, DeleteOrdinary };
      void add(Kind, float seconds, std::string_view source);
      [[nodiscard]] Costs costs(float max_room_s) const;   // delete_route_s = delete_ordinary_s + max_room_s
      [[nodiscard]] std::size_t n(Kind) const;
      static CostEstimator load(const std::filesystem::path&);   // from_chars
      void save(const std::filesystem::path&) const;              // kind,seconds,source
  };
  ```
- Online samples, all observable in `DoorCycleInput`:
  - Keep: the duration of an `executing` episode on one of this agent's `aff_door_N_{approach,cross}` affordances that
    ended without that door in `backed_faces`, i.e. no crossing.
  - DeleteOrdinary: the time from a REMOVE to a creation at a row matched to the removed one (Task 7).
- `max_room_s`: the controller's `MaxRoomS`, read from `door_concept/etc/config.toml`, new key
  `Existence.MaxRoomS = 300`, with a comment naming the controller key it mirrors.

- [ ] **Step 1: Write the failing tests** (spec test 6).
  - `route_door_survives`: with Costs{30, 120, 420} and L = −2.0, Ordinary is removed (line −1.39). Route is not
    (line −2.64).
  - `being_crossed_never_removed`: L = −50 is not removed.
  - `exists_at_zero`: L = 1e-4 exists. L = −1e-4 does not exist and is not removed.
  - `median_and_persist`: samples {10, 30, 50} give a median of 30, and survive a save/load round trip under es_ES.
  - `role_from_route_faces`.
- [ ] **Step 2:** Run. Expected: compile failure.
- [ ] **Step 3: Write `tools/door_costs_from_logs.py`.** It reads `explore_tour10..22`.
  - Keep: the duration of door-passage attempts that did not end in a crossing, from the controller's crossing.log or
    events.jsonl passage records.
  - DeleteOrdinary: removal → `RE-ACQUIRE`/re-birth at the same place, from door_events.csv.
  - Write `etc/door_costs.csv` with one row per sample and source `<tour>:<event>`. Print the medians and n.
  - If either kind has fewer than 3 samples, print that and fall back for that kind to the spec's stated reading: Keep
    = 30 s, the order of one failed approach. Flag the fallback in the CSV source column.
- [ ] **Step 4: Implement `door_decision.h`.**
- [ ] **Step 5:** Run. Expected: pass.
- [ ] **Step 6:** Commit: `door_concept core: costed decision, door role, cost estimator`.

### Task 7: The place table (memory = belief)

**Files:**
- Create: `src/core/place_table.h`, `tests/test_place_table.cpp`
- Modify: `src/door_identity_table.h`. The row format is extended. Ghost and restore-seed code is deleted in Task 10.

**Interfaces:**
- Consumes: `ViewLedger` (Task 4). The tracker gate `cfg.tracker_gate_mahalanobis`.
- Produces:
  ```cpp
  struct PlaceRow {
      std::string room, name;
      Vec2f xy; Eigen::Matrix2f cov; float yaw, width, height;
      float L_evidence;                 // ledger sum + crossing evidence, WITHOUT the structural prior
      ViewLedger ledger; int visit; std::int64_t last_seen_wall_ms;
      std::optional<std::int64_t> removed_wall_ms;   // a deleted phantom: "no door here", kept for ever
  };
  class PlaceTable {
  public:
      PlaceRow* match(std::string_view room, const Vec2f& xy, const Eigen::Matrix2f& cov, float gate);   // min Mahalanobis in gate
      PlaceRow& upsert(PlaceRow);                                 // never deletes
      void note_revisit(PlaceRow&, bool found, double elapsed_s); // updates the change hazard
      [[nodiscard]] float aged_L(const PlaceRow&, std::int64_t now_wall_ms) const;   // L·exp(−λ·Δt) toward 0
      [[nodiscard]] double hazard_per_s() const;                  // λ = changes / Σ elapsed, Gamma(1, 600 s) prior
  };
  ```
- The hazard prior Gamma(1, 600 s) is today's 1/600 s used once, as the prior's mean. It is not a constant in use, and
  the comment says so.
- `door_identity_table.h` gains three columns: `ledger`, `visit`, `removed_ms`. The reader treats them as optional.

- [ ] **Step 1: Write the failing tests** (spec test 7).
  - `restored_continues_L`: write a row with L 2.1, then load and match it. The candidate starts at 2.1 aged by
    `hazard_per_s()`.
  - `phantom_row_is_prior`: a removed row with L −3 matched by a new candidate starts it at −3.
  - `rows_never_forgotten`: 1000 upserts at 3 places give 3 rows.
  - `gate_grows_with_cov`: a candidate 0.8 m off matches with σ 0.5 and does not with σ 0.1.
  - `old_row_loads_with_empty_ledger`: a 19-column row from today's format loads with an empty ledger and its
    exist_logodds as L.
  - `hazard_learns`: 10 revisits, all found after 100 s, give a smaller λ than the prior mean.
- [ ] **Step 2:** Run. Expected: compile failure.
- [ ] **Step 3: Implement**, including the identity-table column extension with from_chars.
- [ ] **Step 4:** Run, together with `test_door_identity_table.cpp`. Expected: the new tests pass. Note which
  identity-table cases fail because they assert ghosts or forgetting, for retirement in Task 10.
- [ ] **Step 5:** Commit: `door_concept core: place table — the memory row is the belief`.

### Task 8: Rewrite `DoorCore::update_existence`

**Files:**
- Modify:
  - `src/core/door_existence.cpp` (`update_existence`, ≈ lines 701-1697), `src/core/door_existence.h`;
  - `src/core/door_core.{h,cpp}`: birth evidence, remove `ricoh_bank_`'s ripening, add `PlaceTable`,
    `DetectorTable`, `CostEstimator` members;
  - `src/door_fit_state.h`: new `PlaceBelief`, old fields removed in Task 10.
- Test: rewrite `tests/test_door_core_existence.cpp`.

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces:
  - `struct PlaceBelief { PlacePrior prior; float carried = 0, crossing = 0; ViewLedger ledger; float L() const; };`
    where `L() = prior.total() + carried + crossing + ledger.sum()`. `carried` is a source row's L_evidence (Task 9).
    A row stores `L_evidence = carried + crossing + ledger.sum()`.
  - `DoorFitState` gains `PlaceBelief belief; Role role;`.
  - Every consumer of `inst.existence.logodds()`/`.prob()` reads `inst.belief.L()`. The consumers are
    specificworker.cpp, door_look.h, epistemic_planner, door_identity_table.h and door_replay. P = σ(L).
- Per cycle with a fresh mask or ricoh frame, for every door AND every pending tracker candidate (a candidate's θ is
  the typical door at its blob, on the band):
  1. Role = `door_role(name, in.route_faces, being_crossed or fact_backed)`.
  2. Prior: `place_prior(...)`, replaced, never accumulated.
  3. For each camera (ZED, and the ricoh when `in.ricoh.present`): `door_view` → `detect_prob` →
     `ledger.observe` → `add_bernoulli`, with hit_w = the tracker association weight of a slice of that source (0 if
     none).
  4. Geometry (RGB frames, mask or not): the existing contour/frame/depth statistic and its controls → `add_geometry`.
     μ0 and s² come from the same-frame controls. μ1 starts at the bench value and is learned. `geometry_may_refute`
     comes from config `Existence.GeometryMayRefute = false` (§8.5).
  5. Actuation: a commanded open/close seen satisfied (the existing actuator reply path) → `add_actuation` once per
     command.
  6. `ledger.recompute`, then `decide(belief.L(), role, costs)`.
  7. Candidates: `birth_evidence = verdict.exists ? 1 : 0` and the door tracker's `birth_frames = 1`. The comment
     says the tracker's streak is degenerate on purpose: the door model decides, and the tracker enforces separation
     and association.
  8. Removal: `remove`, then upsert the row with `removed_wall_ms`. A creation that matches a removed row feeds
     `CostEstimator::add(DeleteOrdinary, …)`. An `executing` episode on a door affordance that ends without a crossing
     feeds `CostEstimator::add(Keep, …)`.
  9. Online detector learning (§8.4), one trial per view, never per frame:
     - hit_rate: hits on doors with L > 0;
     - false_alarm: detections associated to places whose row was later removed.

     The DetectorTable and the CostEstimator are saved on graceful shutdown and loaded at start. The stdout line names
     n for each.
- §3.8: one existence row per door per fresh frame, with columns for every channel's ℓ this cycle (`zed_dl, ric_dl,
  geo_dl, act_dl`), the prior parts, crossing, `view_id`, L, role and delete_line. A field not computed this cycle is
  printed as `nan`. Update `kExistenceCsvHeader`. The stdout existence line prints the same per-channel values, and a
  field not computed this cycle is printed as `—`, never a stale value (tour18's "free=0 occl=0").

- [ ] **Step 1: Write the failing tests** in the rewritten test_door_core_existence.cpp, on its existing scene.
  - `unseen_door_unchanged` (spec test 2): 3 min of frames with the door behind the camera, then around a wall, then
    overflowing the ZED at 0.5 m, with both cameras. L is bit-identical before and after.
  - `phantom_removed_three_views` (spec test 3): a phantom (no mask, no edge) seen face-on at good fill from 3
    standpoints 40° apart. It is REMOVED after the third and not before.
  - `parked_stare_one_view`: 20 Hz for 30 s from one standpoint moves L by exactly one view's ℓ.
  - `real_door_aspect_misses` (spec test 4): one confirming hit-view, then 20 miss-views at 70° aspect at 1 m and 6 m.
    It is never removed.
  - `ricoh_near_miss_zero` (spec test 5): the ricoh at 1.4 m with the door face-on, no detection, fitted envelope.
    The ricoh ℓ is less than 0.1 in magnitude per view.
  - `route_door_survives` (spec test 6, core level): evidence that removes an Ordinary door leaves a door named in
    `route_faces` alive.
  - `being_crossed_holds_evidence`: during `executing` on its cross, evidence still accumulates (the ledger
    changes), and no REMOVE happens.
  - `birth_by_comparison` (spec test 8):
    - 0.80 m and 1.00 m candidates are born after one good hit-view;
    - a 3 m candidate is not born after one, and is born after enough;
    - an off-wall candidate is never born;
    - a second door on a wall needs more views, and is still born under strong evidence.
  - `relocalising_robot_moves_nothing` (Review Focus 1): `rt_cov` with σ_xy = 3 m. Every ℓ is < 0.05 in magnitude.
  - `open_door_survives_face_on_views` (Review Focus 2): a φ posterior at 80° with the leaf mask present on the open
    leaf. No REMOVE after 10 views.
  - `determinism`: two cores, bit-identical rows (kept from the old file).
  - `csv_classic_locale`: kept from the old file.
- [ ] **Step 2:** Run. Expected: the new cases fail.
- [ ] **Step 3: Implement the new `update_existence`.** Do not port any branch of the old function. The old body is
  deleted in this task, except its probe calls, which now feed the ledger.
- [ ] **Step 4:** Run test_door_core_existence, test_door_core_tracker and test_door_core_fit. Expected: the new
  existence cases pass. Tracker and fit pass, except the "not before birth_frames ideal hits" tracker case, which
  is rewritten to assert birth at L > 0.
- [ ] **Step 5:** Run `tests/test_door_tapes.sh`. Expected: decisions may differ from the recorded logs, and every
  difference is examined. The asserted defects (D1, D2, D4, D-C) must still hold. Update ALLOWED with one reason per
  changed decision.
- [ ] **Step 6:** Commit: `door_concept core: update_existence on one belief (prior + crossing + view ledger), costed decision`.

### Task 9: The crossed door

**Files:**
- Modify: `src/specificworker.cpp` (`install_carried_door` ≈4208, `recreate_lost_door`, the restore seed),
  `src/core/place_table.h`
- Test: `tests/test_door_room_swap.cpp` (extend), new `tests/test_carried_door.cpp`

**Interfaces:**
- Consumes: `PlaceTable`, `PlaceBelief`, `crossing_logodds`.
- Produces: `[[nodiscard]] PlaceBelief carried_belief(const PlaceRow* src, float crossing_llr);`
  - With a source row: prior = {} (zero), carried = src L_evidence, crossing = crossing_llr, and an empty ledger for
    the new room.
  - Without one (the source face was never known): prior = the new room's place prior, carried = 0, crossing =
    crossing_llr.

- [ ] **Step 1: Write the failing tests** (spec test 1).
  - `carried_door_above_creation`: a source row with L 1.5 gives L = 1.5 + crossing_llr > 0.
  - `carried_without_source`: L = prior + crossing_llr.
  - In test_door_room_swap: on the `exit` fact the new room's door instance exists in the same cycle with role
    BeingCrossed and L ≥ crossing_llr.
- [ ] **Step 2:** Run. Expected: fail.
- [ ] **Step 3: Implement.** `install_carried_door` sets `inst.belief = carried_belief(match, crossing_logodds(...))`
  and upserts the row in the new room. The restore seed and its hazard constant are removed. A restore now reads
  `aged_L`.
- [ ] **Step 4:** Run. Expected: pass.
- [ ] **Step 5:** Commit: `door_concept: a crossed door is born in the new room carrying its evidence plus the crossing`.

### Task 10: Delete the old code; amend the lifecycle

**Files:**
- Modify: door_existence.cpp, door_fit_state.h, specificworker.{h,cpp}, core/ricoh_door_birth.h, door_view_pool.h
  (delete), door_config_fields.h, door_config.{h,cpp}, etc/config.toml, `CONCEPT_AGENT_LIFECYCLE.md` §1
- Tests: delete or rewrite the ones §5 lists

- [ ] **Step 1: Delete everything in spec §4 "Goes" that still exists.**
  - `existence_seeded`, `existence_entry_L`, `existence_since_entry`, `existence_debounce`, `existence_held_crossing`,
    `room_prior_term`, the three pools, `ricoh_view_pool`, `ricoh_hit_pool`;
  - `remember_ghost`, `forget_identity`, `match_ghost`, DoorGhost, GhostMatch, `reacquire`, `forget_ghost` in the
    world interface;
  - `RicohBirthBank` ripening and `RicohViewPool`;
  - `kRememberedExistenceHazardPerS`, `restore_existence_prior`, `seed_restored_instance`'s existence half;
  - config keys `BirthLogodds, MaxLogodds, RemovalProb, RemoveFrames, OutOfRoomGain, ShortGain, MinHeightConf,
    ReacquireRadiusM, GhostMax, CentralRegionFrac`. The config loader must reject them if present, with a message
    naming this plan.
- [ ] **Step 2: Verify the patch markers are gone.** `grep -rn "since_entry\|L_bound\|probed()\|ghost\|existence_seeded\|ripe_at\|RemoveFrames" door_concept/src`
  returns nothing. Every remaining `std::clamp` in the existence path has a comment naming why it is not a threshold.
- [ ] **Step 3: Retire or rewrite the tests.**
  - Retire: test_door_absence_detectability, test_door_existence_moving_phantom, test_door_ricoh_lifecycle,
    test_door_view_pool.
  - Rewrite test_door_identity_table to the row format, and test_door_look on `belief.L()`.
  - Run every remaining test. Expected: the baseline list from Task 0, minus the retired tests, plus the new ones.
- [ ] **Step 4: Amend `CONCEPT_AGENT_LIFECYCLE.md` §1's door exceptions.** Birth is by comparison (L > 0 on the
  door model, the tracker enforcing separation and association). Deletion is costed. The crossed door is created
  through the carried path. Update door_concept's conformance row.
- [ ] **Step 5: Build the agent and confirm it starts with the new config.** `cmake -B build` (clean reconfigure)
  then the capped `make -C build -j8`.
- [ ] **Step 6:** Commit: `door_concept: remove the existence patches (ghosts, streaks, pools, seeds, clamps)`.

### Task 11: Replay tour18

**Files:**
- Modify: `tools/door_replay.cpp`. Add `--ricoh-from <shadow.json>` to inject `ricoh` and `body_T_ricoh` into v4
  tapes. This is a stated difference from the live run, and the replay prints it.
- Modify: `tests/test_door_tapes.sh`. Add tour18.

- [ ] **Step 1: Add the assertion T18 to test_door_tapes.sh.** Replaying `explore_tour18` with `--ricoh-from` must
  produce no REMOVE row for door_4 or door_5. Print each one's L trajectory and its per-channel contributions at the
  cycles where the old code removed them (from Task 0 Step 3).
- [ ] **Step 2:** Run. Expected: FAIL before the flag exists.
- [ ] **Step 3:** Implement the flag.
- [ ] **Step 4:** Run `tests/test_door_tapes.sh`. Expected: T18 passes. The existing tapes keep their door counts
  (spec test 9).
- [ ] **Step 5:** Commit: `door_replay: --ricoh-from for v4 tapes; tour18 regression`.

### Task 12: Live: rate and tour series

- [ ] **Step 1:** Message `shadow-bridge-dds-d6` that Webots will be in use.
- [ ] **Step 2: One tour from scratch.**
  `RUN_LIMIT_S=… $WT/room_concept/estimation_scripts/scripts/explore_tour.sh explore_tour23`, from the main
  checkout's estimation_scripts, watched with Monitor.
  - Check door_concept's compute rate against its nominal [Period] in `door_existence_log.csv` cycle stamps. If it is
    below nominal, profile the per-RGB-frame geometry probes and move them to a worker thread (spec §7) before going on.
- [ ] **Step 3: Check spec test 1 live.** In `door_existence_log.csv`, the crossed door's first row in the new room
  has the same cycle as the `exit` fact in events.jsonl.
- [ ] **Step 4: A series** of 5 tours with scratchpad/series.sh. Grade each with grade_explore_tour.py. Record:
  - doors lost while needed (target 0);
  - phantoms alive at the end;
  - removals by role;
  - the learned costs and detector parameters at the end (`etc/door_costs.csv`, `etc/door_detectors.csv`) and their n.
- [ ] **Step 5:** Tell `shadow-bridge-dds-d6` that Webots is free. Append the results to the memory note
  `exploration-tour-from-scratch-11-05.md`.
- [ ] **Step 6:** Commit the learned parameter files and a short results note under
  `docs/superpowers/plans/2026-11-07-door-existence-results.md`.
