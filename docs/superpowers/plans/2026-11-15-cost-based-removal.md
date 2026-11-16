# Cost-Based Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real countable objects (nightstand, bed, toilet) stop collecting LiDAR absence from their own returns, and then
countable removes on the costed line L < −ln(C_delete/C_keep) instead of a probability threshold plus a debounce.

**Architecture:** Two stages, each live-tested on its own. Stage 1 (G0) adds a shared 3-D carve against a list of part
boxes (`rc::exist::carve_parts`, next to the untouched `carve_box`), a manifest parser for each class's parts, and
switches countable's LiDAR carve to it behind a config flag. Stage 2 adds a shared costed decision (`existence_cost.h`),
measured ρ and costs per class, and countable's one costed decision on the product p_occ·p_cls.

**Tech Stack:** C++23 header-only code in `common/`, countable_concept (Qt/DSR agent), standalone test executables
(CMake `countable_concept/tests`, plain `g++` for `common/`), Python 3 + numpy for log tools.

**Spec:** `docs/superpowers/specs/2026-11-15-cost-based-removal-design.md` (rev 2). Evidence for stage 1:
`docs/superpowers/results/2026-11-15-carve-replay.md` and its script.

## Global Constraints

- Worktree `/home/pbustos/robocomp/components/active_inference.worktrees/cost-removal`, branch `feat/cost-removal`.
  Commit locally only; never merge or push.
- `carve_box`, `solid_delta`, `hollow_delta`, `decide_removal` with `costed == false` stay byte-identical in behaviour:
  every other agent (table, chair, bottle, cabinet, hood, refrigerator, residual, door) is untouched.
- Default costs: C_keep 30 s, C_delete 88.25 s, source `compiled_default:door_logged`. Line = −ln(C_delete/C_keep)
  = −1.08 with the defaults. A cost file with C_delete ≤ C_keep is refused as a whole.
- Cost file format: `kind,seconds,source` with header; kinds `keep`, `delete`. Path `<class>_concept/etc/existence_costs.csv`.
- Costed decision + ρ = 0 is refused at config load (throw `rc::CountableConfigError`).
- No new graph attribute. No thresholds or gates beyond the costed line (CLAUDE.md "no thresholds").
- Numbers from files: `std::from_chars` only; files written through `imbue(std::locale::classic())`; every test that
  parses a file calls `std::setlocale(LC_ALL, "")` and is run with `LANG=es_ES.UTF-8`.
- C++: `and` / `or` / `not`; C++23 containers and algorithms.
- Builds: check `pgrep -a cc1plus` first; build inside the memory-capped scope (`cbuild`, or
  `systemd-run --user --scope -p MemoryMax=44G make -C build -j8`), `-j8`.
- Agents stop with SIGTERM/SIGINT, never `kill -9`.
- Webots is shared: before a live run, message the chat and room sessions to claim it; send FREE after.
- No Webots asset (proto, mesh, texture) is copied into the repo; measured numbers are fine.
- Stage 2 tasks (6–9) do not start until the stage-1 live run (Task 5) passes.

## Review Focus

1. **Beam origin inside a part** (robot standing over a low part's footprint, or a box drawn around the robot): that part
   must contribute nothing for that beam — a sensor cannot be occluded by a solid it sits inside. → Task 1 test
   `origin_inside_part_is_ignored`.
2. **A manifest part with zero/negative size or z0 ≥ z1, or a malformed number** must refuse to start (like the missing
   manifest does today), never silently become a box. → Task 2 test `bad_part_is_refused`.
3. **Cost file written under the Spanish locale / one kind missing**: a comma-decimal or a missing kind falls back to the
   compiled default with one loud warning, never a half-read file. → Task 6 tests `locale_roundtrip`,
   `missing_kind_falls_back`.
4. **Belief at the −4 clamp still argued against**: the "argued against" flag must come from the pre-clamp delta, or an
   object pinned at the clamp could never be removed. → Task 8 test `argued_at_clamp`.
5. **Rotated boxes**: parts must follow the box yaw (a toilet seen at yaw π/2). → Task 1 test `parts_follow_yaw`.

---

### Task 1: `rc::exist::carve_parts` — the 3-D carve against a union of part boxes

**Files:**
- Modify: `common/existence_belief/existence_belief.h` (add after `carve_box`, before `solid_delta`; touch nothing else)
- Create: `common/existence_belief/carve_parts_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace rc::exist {
  struct BoxPart { float cx, cy, sx, sy, z0, z1; };   // object frame: centre offset (m), full sizes (m), z band (m, floor = 0)
  inline Evidence carve_parts(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& sweep,
                              float cx, float cy, float yaw, std::span<const BoxPart> parts,
                              float surface_sigma_m, const SensorModel& p);
  }
  ```
  Fills `e_occ`, `e_free`, `e_interior`, `n_reached` exactly as `carve_box` names them; `log_odds_delta` =
  `solid_delta(ev, p)` (callers today overwrite it anyway).

- [ ] **Step 1: Write the failing tests** in `carve_parts_test.cpp` (self-contained `main`, `CHECK` macro like
  `countable_concept/tests/class_factor_test.cpp`). Sensor model: `sensor_sigma_m = 0.03`, `detection_prob = 0.85`,
  `clutter_prob = 0.05`; surface σ 0.02. Box: one part `{0,0, 0.5,0.5, 0,0.62}` at (0,0), yaw 0. Origin (−2, 0, 1.1075).
  Build each sweep by intersecting rays analytically with the scene you state in the test (a helper `first_hit` against a
  list of axis-aligned boxes plus the floor z = 0).
  - `top_return_is_surface`: 200 rays aimed at points spread over the box top (z = 0.62). `e_occ > 150`,
    `e_interior + e_free < 5`. (Compare: `carve_box` on the same sweep gives `e_interior > 100` — assert it, it documents
    the bug.)
  - `beam_above_box_adds_nothing`: rays to points on the floor 4 m beyond, passing ≥ 0.1 m above the top:
    `n_reached == 0`.
  - `thin_panel_front_is_not_through`: part `{0,0, 1.75,0.10, 0,1.25}`, surface σ 0.05, 100 rays returning on its front
    face: `e_free < 5` and `e_occ > 80`.
  - `empty_floor_box_is_against`: the box over empty floor (no scene object; rays land on the wall at x = 4):
    `e_free + e_interior > e_occ` and `solid_delta(ev, sm) < 0`.
  - `box_over_wall_is_against`: box centred on a wall slab (wall `{2.0, 0, 0.2, 6, 0, 3}`, box at (2.1, 0) size 0.5):
    returns on the wall's near face lie inside the box deeper than σ from its entry ⇒ `solid_delta < 0`.
  - `single_part_equals_cuboid`: for a sweep with no top-face returns (rays hitting the box's near SIDE face at z 0.3),
    `carve_parts` with one whole-box part and `carve_box` agree on `e_occ` within 5 % — the 3-D test only changes
    what the 2-D one got wrong.
  - `origin_inside_part_is_ignored`: origin (0, 0, 0.3) inside the box; any sweep ⇒ that part contributes 0 to every term.
  - `parts_follow_yaw`: part `{0, 0.2, 0.5, 0.1, 0, 1.0}` (offset along local +Y); box yaw π/2 ⇒ the part sits at world
    (−0.2, 0); a ray hitting world (−0.2, 0, 0.9) on its face scores `e_occ > 0.5`, the same ray with yaw 0 scores 0.

- [ ] **Step 2: Run to see it fail**

  Run: `cd common/existence_belief && g++ -std=c++23 -O2 -I/usr/include/eigen3 carve_parts_test.cpp -o /tmp/carve_parts_test && /tmp/carve_parts_test`
  Expected: compile error, `carve_parts` not declared.

- [ ] **Step 3: Implement `carve_parts`.** Same Φ, σ_surf = √(sensor_σ² + surface_σ²), σ_t = σ_surf/len, and the same
  rotation into the object frame as `carve_box`. The union rule (not determined by the tests — implement exactly this):
  ```
  for each beam (O → P, len):
    for each part k: 3-D slab test on x, y AND z → (tn_k, tf_k) or miss; skip k if miss, if tf_k < 0,
                     or if tn_k < 0 (origin inside k: Review Focus 1)
    reached = max_k Φ((1 − tn_k)/σ_t);            skip beam if reached < 1e-3
    t_last  = max { tn_k : tn_k ≤ 1 } over kept parts, else min_k tn_k   // entry of the part the return lies in
    depth   = max(0, (1 − t_last)·len);  ws = exp(−½ depth²/σ_surf²)
    through = Φ((1 − max_k tf_k)/σ_t)              // beyond the LAST exit of every part it entered
    inside  = max(0, reached − through)
    e_occ += inside·ws;  e_interior += inside·(1 − ws);  e_free += through·(1 − ws);  reached_mass += reached
  n_reached = lround(reached_mass)
  ```
  Comment the function with the 11-15 finding (2-D surface test scored top-face returns as interior; thin-part leak) and
  a pointer to the results doc.

- [ ] **Step 4: Run the tests until all pass** (same command). Expected: no `FAIL` lines, exit 0.

- [ ] **Step 5: Confirm the other agents are untouched:** `git diff --stat` shows only the two files; `git diff` of
  `existence_belief.h` contains only additions.

- [ ] **Step 6: Commit**
  ```bash
  git add common/existence_belief/existence_belief.h common/existence_belief/carve_parts_test.cpp
  git commit -m "existence_belief: carve_parts — 3-D carve against a union of part boxes (top-face returns are surface)"
  ```

### Task 2: Shape parts in the manifest

**Files:**
- Create: `common/concept_manifest/shape_parts.h`
- Modify: `common/concept_manifest/nightstand.concept.toml`, `bed.concept.toml`, `toilet.concept.toml`
- Modify: `countable_concept/src/countable_config.h`, `countable_config.cpp` (load the parts next to the back panel, ~line 140)
- Create: `countable_concept/tests/shape_parts_test.cpp`; Modify: `countable_concept/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `rc::exist::BoxPart` (Task 1); `rc::manifest::BackPanel` (`back_panel.h`).
- Produces:
  ```cpp
  namespace rc::manifest {
  struct ShapePart { std::string name; float x_f, y_f, w_f, h_f, z0_f, z1_f; };   // fractions of the fitted (w, h, H)
  // absent table ⇒ {} ; malformed ⇒ throws std::runtime_error naming the file, the part and the key
  std::vector<ShapePart> load_shape_parts(const std::string& manifest_path);
  }
  namespace rc::countable {
  // the parts for the CURRENT fitted box; empty manifest list ⇒ one whole-box part {0,0,w,h,0,H};
  // a back panel is appended as {0, h/2 − t/2, w, t, 0, panel.height_m} with t = min(panel.thickness_m, h)
  std::vector<rc::exist::BoxPart> carve_parts_for(float w, float h, float H,
                                                  std::span<const rc::manifest::ShapePart> parts,
                                                  std::optional<rc::manifest::BackPanel> panel);
  }
  ```
  (`carve_parts_for` lives in a new header `countable_concept/src/countable_shape_parts.h`.) `CountableConfig` gains
  `std::vector<rc::manifest::ShapePart> shape_parts` and keeps the back panel it already reads.
- Manifest syntax (TOML array of tables, parsed textually like `back_panel.h`, numbers via `std::from_chars`):
  ```toml
  [[shape.part]]
  name = "body"
  x_f = 0.0
  y_f = 0.0
  w_f = 1.0
  h_f = 1.0
  z0_f = 0.66
  z1_f = 1.0
  why = "..."
  ```
  Local axes: x across the width w, +y toward the BACK face (the countable convention, `countable_belief.cpp:50`).
  A refused part: any of w_f, h_f ≤ 0, z1_f ≤ z0_f, a missing key, or a non-numeric value.

- [ ] **Step 1: Write the failing tests** in `shape_parts_test.cpp` (writes temp manifests; `std::setlocale(LC_ALL, "")`):
  - `absent_table_is_empty`: a manifest without `[[shape.part]]` ⇒ `load_shape_parts` returns `{}` and
    `carve_parts_for(0.5,0.5,0.62,{},nullopt)` is one part `{0,0,0.5,0.5,0,0.62}`.
  - `three_classes_load`: the real three manifests load with 5 (nightstand: body + 4 legs), 2 (bed: mattress, footboard;
    the headboard comes from the back panel) and 3 (toilet) parts.
  - `bed_layout`: bed manifest + its back panel at w 1.75, h 2.1, H 0.65 ⇒ mattress z [0.25, 0.65] ±0.01, footboard at
    y ≈ −1.0 with z1 ≈ 0.83, panel at y ≈ +1.0 with z1 = 1.25.
  - `bad_part_is_refused`: `h_f = 0`, `z1_f = z0_f`, a missing `z0_f`, and `w_f = 0,5` each throw.
- [ ] **Step 2: Add the test target** to `countable_concept/tests/CMakeLists.txt` (same pattern as `back_panel_test`) and
  run: `cmake -S countable_concept/tests -B /tmp/ct_build && make -C /tmp/ct_build -j8 shape_parts_test && LANG=es_ES.UTF-8 /tmp/ct_build/shape_parts_test`.
  Expected: compile error (`shape_parts.h` missing).
- [ ] **Step 3: Implement `load_shape_parts`, `carve_parts_for`, and the manifest tables.** Values (measured on the
  Webots models, as the rest of each manifest; say so in each `why`; no asset copied):
  - nightstand (H = 0.62, w ≈ h ≈ 0.505): body `x 0, y 0, w 1, h 1, z 0.66..1.0`; four legs
    `x ±0.44, y ±0.44, w 0.08, h 0.08, z 0..0.66`.
  - bed (H = 0.65, w 1.75, h 2.1): mattress `x 0, y 0, w 1, h 1, z 0.38..1.0`; footboard
    `x 0, y −0.476, w 1, h 0.048, z 0..1.28`. Headboard = the existing `[prior.headboard]`.
  - toilet (H = 1.02 true top, w 0.58, h 0.733): tank `x 0, y 0.349, w 1.0, h 0.30, z 0.233..1.0`; bowl
    `x 0, y −0.153, w 0.83, h 0.696, z 0.235..0.451`; pedestal `x 0, y −0.033, w 0.62, h 0.682, z 0..0.235`.
  In `countable_config.cpp`, load the parts right after the back panel; a throw becomes `CountableConfigError` naming the
  manifest; print `[manifest] <cls> shape: <n> part(s)` once.
- [ ] **Step 4: Run the test** (same command). Expected: all pass.
- [ ] **Step 5: Commit**
  ```bash
  git add common/concept_manifest/shape_parts.h common/concept_manifest/{nightstand,bed,toilet}.concept.toml \
          countable_concept/src/countable_shape_parts.h countable_concept/src/countable_config.{h,cpp} \
          countable_concept/tests/shape_parts_test.cpp countable_concept/tests/CMakeLists.txt
  git commit -m "manifest: [[shape.part]] for nightstand, bed, toilet; countable lays the parts out on the fitted box"
  ```

### Task 3: countable's LiDAR carve uses the parts (opt-in flag) and records the pre-clamp increment

**Files:**
- Modify: `countable_concept/src/countable_existence.cpp:347-358` (the two `carve_box` calls), `countable_config.{h,cpp}`
  (new key), `countable_concept/src/countable_instance.h` (one field), the three `<cls>_concept/etc/config.toml`,
  `countable_concept/KEYMAP.md`

**Interfaces:**
- Consumes: `rc::exist::carve_parts` (Task 1), `rc::countable::carve_parts_for`, `CountableConfig::shape_parts` (Task 2).
- Produces:
  - config key `Model.ExistenceCarveParts` (bool, default `false`) → `cfg_.existence_carve_parts`; `true` in the three
    class configs.
  - `CountableInstance::occ_delta_cycle` (float, default 0): the sum of every `log_odds_delta · weight` integrated into
    `inst.existence` this cycle, BEFORE the clamp (reset to 0 at the start of the instance's loop body). Task 8 reads it.

- [ ] **Step 1:** With the flag on, both the Helios and the bpearl carve call `carve_parts(origin, sweep, bs.cx, bs.cy,
  bs.yaw, carve_parts_for(bs.w, bs.h, bs.H, cfg_.shape_parts, panel), surf_sigma, sm)`; with it off, the current
  `carve_box` lines run unchanged. Everything after the carve (exclusion, p_detect, `observed` zeroing, `solid_delta`,
  integrate) is untouched. The back panel goes into the config struct if it is not already reachable here.
- [ ] **Step 2:** Accumulate `inst.occ_delta_cycle` at each `inst.existence.integrate(...)` call in this function (silhouette,
  contour, LiDAR).
- [ ] **Step 3: Build** countable_concept in the worktree (`pgrep -a cc1plus` first; `cbuild` in `countable_concept`).
  Expected: builds; the existing countable tests still pass
  (`make -C /tmp/ct_build -j8 && ctest --test-dir /tmp/ct_build`).
- [ ] **Step 4: Config check:** start nothing; run `LANG=es_ES.UTF-8 /tmp/ct_build/countable_card_test ../../bed_concept`
  (and the other two) — passes; add the key to `KEYMAP.md`.
- [ ] **Step 5: Commit**
  ```bash
  git commit -am "countable: LiDAR carve against the class's composed parts (Model.ExistenceCarveParts, on for the 3 classes)"
  ```

### Task 4: Carve replay that runs the agent's own carve

**Files:**
- Create: `countable_concept/tools/carve_replay_rays.py` (from the 11-15 results script: the ray-cast half only)
- Create: `countable_concept/tools/carve_replay.cpp` + a target in `countable_concept/tests/CMakeLists.txt`
- Create: `countable_concept/tools/scene_threerooms.csv` (the GT boxes: `label,xmin,xmax,ymin,ymax,zmin,zmax`, the
  numbers in the results script — measured numbers, no asset)

**Interfaces:**
- `carve_replay_rays.py RUN_DIR SCENE_CSV OUT.bin [--phantoms]`: for every drop cycle (and, with `--phantoms`, every cycle
  for synthetic phantoms: each real box displaced 1.0 m into free floor, and one displaced onto the nearest wall), writes
  records `[u32 n_name][name][f32 cx cy yaw w h H std_cx std_cy][f32 origin xyz][u32 n][n × f32 xyz]` in the WORLD frame.
  Room→world as in the results doc (bedroom from the three objects; bathroom pinned by the toilet).
- `carve_replay OUT.bin MANIFEST_DIR` prints per node: Σ ΔL under `carve_box` and under `carve_parts`
  (via `carve_parts_for` and the class manifest), both through `solid_delta` with the agent's sensor model (0.85 / 0.05 /
  0.03 m), and the per-cycle mean.

- [ ] **Step 1:** Write both tools; the C++ one includes `existence_belief.h`, `shape_parts.h`, `back_panel.h`,
  `countable_shape_parts.h` — the agent's code, not a re-implementation ([[bench-runs-the-agents-code]]).
- [ ] **Step 2: Run on objects_live6**
  ```bash
  R=/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/objects_live6
  python3 countable_concept/tools/carve_replay_rays.py $R countable_concept/tools/scene_threerooms.csv /tmp/rays6.bin --phantoms
  /tmp/ct_build/carve_replay /tmp/rays6.bin common/concept_manifest
  ```
  **Pass:** `carve_parts` mean ΔL ≥ 0 for bed_10, nightstand_6, nightstand_7, toilet_6; < 0 for every synthetic phantom;
  `carve_box` numbers reproduce the results doc's first column within ±20 %. If a real object fails, stop and report —
  do not tune part fractions to pass.
- [ ] **Step 3:** Append the output table to `docs/superpowers/results/2026-11-15-carve-replay.md` (section "Agent code
  replay") and commit tools + doc:
  ```bash
  git add countable_concept/tools countable_concept/tests/CMakeLists.txt docs/superpowers/results/2026-11-15-carve-replay.md
  git commit -m "countable tools: carve replay through the agent's carve_parts (objects_live6 real objects >= 0, phantoms < 0)"
  ```

### Task 4b: WITHDRAWN by the owner 11-15 ("Do not patch the code") — replaced by Task 4c

### Task 4c: marginalise the box position as one common-mode shift (spec G0b)

**Files:**
- Modify: `common/existence_belief/existence_belief.h` (additions only), `common/existence_belief/carve_parts_test.cpp`
- Create: `countable_concept/src/countable_placement_prior.h`; Modify: `countable_concept/src/countable_existence.cpp`
  (the flag-on LiDAR branch only), `countable_concept/tools/carve_replay.cpp`, `countable_concept/tests/CMakeLists.txt`
  (+ `countable_concept/tests/placement_prior_test.cpp`)

**Interfaces:**
- Produces, in `existence_belief.h`:
  ```cpp
  struct ShiftNode { float dx, dy, log_w; };            // log_w = log of the quadrature weight (sums to 1 before the prior)
  // Gauss–Hermite tensor grid on the principal axes of Σ_xy (eigen-decomposition); order 3 ⇒ 9 nodes
  // (1-D nodes 0, ±√3·σ_i, weights 2/3, 1/6, 1/6); order 5 ⇒ 25 nodes (standard probabilists' GH).
  std::vector<ShiftNode> common_mode_nodes(const Eigen::Matrix2f& sigma_xy, int order);
  // log Σ_k exp(log_w_k + l_k) / Σ_k exp(log_w_k), computed with log-sum-exp; empty ⇒ 0
  float log_marginal(std::span<const float> l, std::span<const float> log_w);
  ```
- Produces, in `countable_placement_prior.h`:
  ```cpp
  // −½·P·d³, d = max over the 4 footprint corners of the distance OUTSIDE the closed outline (0 inside); P ≤ 0 or an
  // outline with < 3 vertices ⇒ 0. Same functional form and constant as the fit's wall no-cross factor
  // (countable_belief.cpp, residual min(0, gap), λ = P·penetration).
  float no_cross_logprior(float cx, float cy, float yaw, float w, float h,
                          std::span<const Eigen::Vector2f> outline, float precision);
  ```
- countable, flag on: Σ_xy = the belief covariance's (cx, cy) block; nodes = `common_mode_nodes(Σ_xy, 3)`; for each node:
  Helios `carve_parts` at (cx + dx, cy + dy) with surface σ **0** (the footprint σ is now in the nodes; the sensor σ stays in
  the SensorModel), bpearl likewise (occupancy only, as today), exclusion weight as today, the `observed` zeroing as today,
  ℓ_k = `solid_delta(ev_k, sm)`; log_w_k += `no_cross_logprior(shifted box, fitter.room_polygon(), cfg_.ai2_wall_no_cross_precision)`;
  l_full = `log_marginal(ℓ, log_w)`. Everything after (`p_detect_lidar` scaling, integrate, `occ_delta_cycle`) unchanged.
  The debug columns (`dbg_ex_lidar_occ/free`, `dbg_ex_lidar_free_eff`) carry the posterior-weighted means over nodes.
  `n_reached` for `coverage` = the posterior-weighted mean. No room polygon ⇒ the prior term is 0. Flag off ⇒ untouched.
- carve_replay: same computation through the same functions (outline = the GT room rectangle of the node: bedroom
  x −8.39..−1.815, y 9.35..13.48; bathroom x −1.615..1.69, y 9.35..13.48 — measured numbers; Σ_xy from the logged
  std_cx, std_cy, zero correlation), printed next to the per-beam carve_parts column; `--order 3|5`.

- [ ] **Step 1: Tests.** In `carve_parts_test.cpp`: `nodes_reproduce_moments` (order 3 and 5: Σ w·δ = 0, Σ w·δδᵀ = Σ_xy
  within 1e-4 for a rotated anisotropic Σ); `log_marginal_basics` (equal ℓ ⇒ that ℓ; one node with log_w −∞ ignored;
  stable for ℓ = ±50). In `placement_prior_test.cpp`: inside ⇒ 0; a box whose corner is 0.1 m past an edge with P 2000
  ⇒ −1.0 within 1e-3; < 3 vertices ⇒ 0. A scenario test (in carve_parts_test, using the helpers and a square outline):
  the box-over-wall geometry with Σ_xy = diag(0.17², 0.17²) — marginal ΔL with the prior < 0, and WITHOUT the prior
  (all log_w equal) documented as > 0; the real-object case (returns on the top and front of a box placed 0.10 m off its
  believed centre, Σ = diag(0.17²)) — marginal ΔL > 0.
- [ ] **Step 2: Implement** the helpers and the countable branch; syntax-check countable_existence.cpp (the agent build
  waits for the live test to end); build `carve_replay` with `-j2`.
- [ ] **Step 3: Smoke replay** (`--max-cycles 20 --phantoms`, orders 3 and 5): report every node's marginal mean ΔL.
  Expected: real nodes ≥ 0, every probed phantom < 0, order 3 and 5 of the same sign everywhere. If a real node or a
  phantom fails, STOP and report with `--explain` output — do not change part fractions, σ, or the prior.
- [ ] **Step 4: Commit** `"existence: marginalise the box position as one common-mode shift, weighted by the class's no-cross prior"`.

#### Task 4c rev 2 (owner 11-15): marginalise the WHOLE box state

The xy-only marginal (Steps 1–3 above, uncommitted, `task-4c-uncommitted.patch`) fixed the wall phantom but turned the
real bed and nightstands negative (fitted H 1.4–3.2 cm above the true top, within std_H). Replace the xy nodes by:
- `struct StateNode { Eigen::Matrix<float,6,1> d; float log_w; };`
  `std::vector<StateNode> cubature_nodes(const Eigen::Matrix<float,6,6>& sigma);` — 12 nodes ±√6·L·e_i, L = Cholesky of Σ
  (if Σ is not positive definite, use the eigen-decomposition with negative eigenvalues set to 0 and say so in a one-time
  warning), log_w = log(1/12). Order of the state = the belief's [cx, cy, H, w, h, yaw].
- per node: the box (cx+d0, cy+d1, yaw+d5) with sizes (H+d2, w+d3, h+d4); any size ≤ 0 ⇒ log_w = −∞ (outside the
  support); else parts = `carve_parts_for` of the shifted sizes; + `no_cross_logprior` of the shifted footprint; ℓ_k as before;
  l_full = `log_marginal`. Keep the xy helpers only if still used (otherwise delete them).
- Tests: `cubature_reproduces_moments` (Σ w·d = 0, Σ w·ddᵀ = Σ within 1e-4 for a random SPD 6×6); `size_support`
  (a node with h ≤ 0 has weight 0); the scenario tests of Step 1 re-run with a 6×6 Σ (diag: xy 0.17², H 0.05², w, h 0.06²,
  yaw 0.05²): wall phantom with prior < 0; real box with H 0.03 m above its true top and 0.10 m off-centre > 0.
- Replay: Σ from the logged std_* (diagonal; say so) — report per node the marginal mean ΔL, the fraction of nodes
  outside the support, and us/call.
- Same STOP rule: if any real node < 0 or probed phantom ≥ 0, stop and report ℓ_k / log_w_k per node; change nothing.
- Commit message: `"existence: marginalise the whole box state (cubature over the belief covariance), weighted by the class's no-cross prior"`.

### Task 5: Live 7a — the new carve alone, today's debounce (operator task)

**Files:** Create `docs/superpowers/results/2026-11-15-objects_live7a.md`.

- [ ] **Step 1:** Build countable_concept in the worktree (Task 3 build). Confirm the binary is newer than every source
  (`find countable_concept/src common/existence_belief common/concept_manifest -newer countable_concept/bin/countable_concept`
  prints nothing).
- [ ] **Step 2:** Claim Webots: message the chat and room sessions; wait for both to answer.
- [ ] **Step 3:** Run the objects_live6 tour from the worktree's binaries (same command as objects_live6,
  `warm_start_tour.sh` fixture `three_rooms_door0`, `N_LEGS=2`, `LOOK_S=60`), run name `objects_live7a`. Stop with SIGTERM.
- [ ] **Step 4:** Report against spec §9 "Live 7a": per real object the minimum `ex_L` and whether it ever fell below 0 on
  a cycle whose LiDAR increment was negative; rates per class (~10 Hz). **Silhouette check** (ruling: the split by parts
  needs a projection the code does not have; this is the question it answers): on each real object's L-drop cycles, the
  share of `ex_sfree_eff` vs `ex_lfree_eff`. If silhouette absence drives a real object below 0, say so in the results and
  stop for the owner before Task 6.
- [ ] **Step 5:** Send FREE to chat and room. Commit the results doc.

### Task 6: `existence_cost.h` — costs, estimator, costed decision

**Files:**
- Create: `common/existence_belief/existence_cost.h`, `common/existence_belief/existence_cost_test.cpp`
- Modify: `common/existence_belief/existence_belief.h` (`RemovalPolicy`, `decide_removal`)

**Interfaces:**
- Produces:
  ```cpp
  namespace rc::exist {
  struct Costs { float keep_s; float delete_s; std::string keep_source, delete_source; };
  struct CostedVerdict { bool remove; float line; };
  // line = −ln(delete_s / keep_s); remove = argued_against and L < line
  CostedVerdict decide_costed(float L, const Costs& c, bool argued_against);
  class CostEstimator {   // generalised from door_decision.h: kinds keep / delete, running median
  public:
    enum class Kind : std::uint8_t { Keep, Delete };
    void add(Kind, float seconds, std::string_view source);
    std::size_t n(Kind) const;
    Costs costs() const;                       // medians + the source of the median sample (or "median_of_N")
    static CostEstimator logged_default();     // 30 s / 88.25 s, sources "compiled_default:door_logged"
    static CostEstimator load(const std::filesystem::path&);
    static CostEstimator load_or_default(const std::filesystem::path&, std::string_view who);   // loud warning on fallback
    void save(const std::filesystem::path&) const;   // header kind,seconds,source; imbue(classic)
  };
  }
  ```
  `RemovalPolicy` gains `bool costed = false; Costs costs{30.0f, 88.25f, "compiled_default:door_logged", "compiled_default:door_logged"};`
  (`existence_belief.h` includes `existence_cost.h`). In `decide_removal`, when `p.costed`: `condemned = L < line`;
  `remove = condemned and look > 0` (no streak; `required` = 0); `stalled` / `unfalsifiable` computed as today.
  When `costed == false` nothing changes.

- [ ] **Step 1: Write the failing tests** in `existence_cost_test.cpp` (`std::setlocale(LC_ALL, "")`, run under
  `LANG=es_ES.UTF-8`):
  - `line_value`: `decide_costed(L, {30, 88.25}, true).line` = −ln(88.25/30) within 1e-5 (≈ −1.0790); L −1.0 keeps,
    L −1.1 removes; `argued_against = false` never removes.
  - `locale_roundtrip`: add (keep 12.5, delete 99.75), `save`, `load` ⇒ medians 12.5 and 99.75 exactly.
  - `refuse_inverted`: a file with keep 100, delete 50 ⇒ `load_or_default` returns the logged default.
  - `missing_kind_falls_back`: a file with only `keep` rows ⇒ logged default; a row `delete,88,25,x` (comma decimal) is
    malformed and skipped.
  - `costed_off_is_identical`: for a recorded 200-cycle trajectory of (L, p_detect) built in the test (L walking from +4 to
    −4 in steps of −0.04, p_detect alternating 0 and 0.8), `decide_removal` with the default policy gives the same
    `remove` sequence before and after this change (store the expected sequence computed with the CURRENT code before
    you edit `existence_belief.h`: write the test first, capture its output, hard-code it).
  - `out_of_view_never_removed`: costed policy, L = −4, `cycle_p_detect = 0` for 1000 cycles ⇒ never `remove`.
  - `parked_stare`: `ExistenceBelief` with ρ = 0.327, stationarity 1 (via `note_viewpoint` with the same viewpoint every
    cycle), starting at L = +4, absence increments of −0.5 at p_detect 0.8 for 1000 cycles ⇒ L stays above the line.
- [ ] **Step 2: Run to see it fail:** `g++ -std=c++23 -O2 -I/usr/include/eigen3 existence_cost_test.cpp -o /tmp/ect && LANG=es_ES.UTF-8 /tmp/ect` → compile error.
- [ ] **Step 3: Implement.** Port door's `CostEstimator` parsing (from_chars over the whole field, positive finite,
  malformed rows counted with one warning) with kinds `keep` / `delete`.
- [ ] **Step 4: Run until all pass**; then rebuild `door_concept/tests` and `countable_concept/tests` and run them to
  show the header change broke no includer.
- [ ] **Step 5: Commit**
  ```bash
  git add common/existence_belief/existence_cost.h common/existence_belief/existence_cost_test.cpp common/existence_belief/existence_belief.h
  git commit -m "existence_belief: costed removal line (existence_cost.h), opt-in through RemovalPolicy.costed"
  ```

### Task 7: Measure ρ and the costs per class

**Files:**
- Create: `countable_concept/tools/frame_correlation_from_logs.py`, `countable_concept/tools/existence_costs_from_logs.py`
- Modify: the three `<cls>_concept/etc/config.toml` (`Model.ExistenceFrameCorrelation`)
- Create: the three `<cls>_concept/etc/existence_costs.csv` (only for a class with ≥ 1 sample of a kind)

**Interfaces:**
- `frame_correlation_from_logs.py CLASS RUN_DIR...`: the table_concept procedure — over the class's `<cls>_ai2_log.csv`
  rows, per node in cycle order, a cycle is a MISS when `frames_since_det > 0`; on cycles with `ex_pdetect > 0.5`
  (the band table uses; also print the > 0.3 band) compute P(miss), P(miss | previous miss) and
  ρ = (P(m|m) − P(m)) / (1 − P(m)); print n of miss pairs. Fewer than 20 miss pairs ⇒ print `INSUFFICIENT` and exit 2.
- `existence_costs_from_logs.py CLASS RUN_DIR... [--write PATH]`:
  - C_delete sample: a removal line (`[existence] removing`, `[<cls>-class] removing`) of an instance, followed by a
    `[tracker] BIRTH` of the same class within ½·max(w, h) of that instance's last fitted centre (same room); the seconds
    between them. A removal never followed so is not a sample.
  - C_keep sample: for each removed instance never re-born at its place, the seconds of its life during which it was the
    epistemic target (its ai2_log rows with `nbv_gain_pub > 0`, summed by the cycle interval). The controller's approach
    seconds are not logged per object: say so in the output.
  - prints n and the median per kind; `--write` writes the csv (`kind,seconds,source` with source = `<run>:<node>`)
    through the same rule as the C++ loader (refuse delete ≤ keep).
  - Run dirs: `objects_live3..6` under `room_concept/estimation_scripts/crossing/` in the main checkout.

- [ ] **Step 1:** Write both scripts with a `--selftest` flag that runs them on a 30-row synthetic log built inside the
  script (one known miss pair structure ⇒ a known ρ; one remove→re-birth at 50 s ⇒ C_delete 50).
- [ ] **Step 2:** `python3 …/frame_correlation_from_logs.py --selftest` and the same for costs: both print PASS.
- [ ] **Step 3:** Run both on objects_live3–6 for each class. Write ρ (the > 0.5 band, the conservative one as in table)
  into each class config with a comment carrying the counts, like `table_concept/etc/config.toml:280`. A class that
  prints `INSUFFICIENT` keeps ρ = 0 and is NOT opted in by Task 8 (record it in the ledger).
- [ ] **Step 4:** Write `existence_costs.csv` for classes with samples; otherwise none (the agent then uses the labelled
  default).
- [ ] **Step 5: Commit** scripts + configs + csvs, message stating each class's ρ and n per cost kind.

### Task 8: countable decides on the costed line

**Files:**
- Modify: `countable_concept/src/countable_config.{h,cpp}` (keys), `countable_existence.cpp` (end of the per-instance
  loop), `specificworker_lifecycle.cpp` (`apply_class_factor`), `countable_class_evidence.h` (pure step), three class
  configs, `KEYMAP.md`, `CONCEPT_AGENT_LIFECYCLE.md` (countable row, REMOVE column)
- Create: `countable_concept/tests/costed_step_test.cpp` + CMake target

**Interfaces:**
- Consumes: `decide_costed`, `CostEstimator::load_or_default`, `Costs` (Task 6); `inst.occ_delta_cycle` (Task 3).
- Produces:
  - keys `Model.ExistenceCosted` (bool, default false) and `Model.ExistenceCostsFile` (string, default
    `etc/existence_costs.csv`); load refuses `costed = true` with `existence_frame_correlation == 0` (throw
    `CountableConfigError`, message names the key and Task 7's script).
  - `CountableInstance::cls_shape_against` (bool): set in `apply_class_factor` on a MEASURED cycle of a WARM instance
    when its own Λ < 0 (spec Q3: the singleton term never counts). Not measured ⇒ false.
  - pure function in `countable_class_evidence.h`:
    ```cpp
    struct CostedStep { bool remove; float L_product; float line; bool occ_against; bool cls_against; };
    CostedStep costed_step(float p_occ, float p_cls, float occ_delta_cycle, bool cls_shape_against,
                           const rc::exist::Costs& c);
    // L_product = logit(clamp(p_occ·p_cls, 1e-6, 1 − 1e-6)); occ_against = occ_delta_cycle < 0;
    // remove = decide_costed(L_product, c, occ_against or cls_shape_against).remove
    ```
- Behaviour with `costed = true`: `update_and_remove` never pushes to `doomed` (it still prints stall / unfalsifiable);
  `apply_class_step` is not called (no class debounce); after the class step, every fitted instance gets one
  `costed_step`; a removal goes through `retire_instance` plus the same `log_phantom_event("DEATH", …)` the occupancy path
  sends, and prints exactly one line:
  `"{agent}: [existence] removing {node} — L={:.2f} < line {:.2f} (C_keep {:.1f} s [{src}], C_delete {:.1f} s [{src}]); against: {occ|class|occ+class}"`.
  `occ_delta_cycle` and `cls_shape_against` reset after the decision.

- [ ] **Step 1: Write the failing tests** `costed_step_test.cpp`:
  - `argued_at_clamp`: p_occ = logistic(−4), p_cls 0.9, occ_delta −0.3 ⇒ remove (the increment, not the clamped L, says
    "against").
  - `rival_alone_never_removes`: p_occ 0.9, p_cls 0.05 (inhibited), occ_delta +0.1, cls_shape_against false ⇒ keep.
  - `own_shape_against_removes`: same but cls_shape_against true ⇒ remove.
  - `no_evidence_no_removal`: L_product −5, occ_delta 0, cls false ⇒ keep.
  - `line_from_costs`: `line` = −ln(88.25/30).
  - config: a temp config with `ExistenceCosted = true` and `ExistenceFrameCorrelation = 0` makes the config loader throw
    (if the loader cannot run standalone, test the validation function you factor out for it).
- [ ] **Step 2: Run to fail**, then **Step 3: implement**, **Step 4: run to pass** (`make -C /tmp/ct_build -j8 && ctest --test-dir /tmp/ct_build`).
- [ ] **Step 5:** Set `ExistenceCosted = true` in each class config whose ρ was measured (Task 7); rebuild the agent;
  update KEYMAP and the lifecycle audit row (REMOVE: "costed line on p_occ·p_cls, argued-against required, ρ measured").
- [ ] **Step 6: Commit** `"countable: remove on the costed line over occupancy x class (opt-in, measured rho required)"`.

### Task 9: Costed replay on 7a traces, then live 7b (operator task)

**Files:** Create `countable_concept/tools/costed_replay.py`, `docs/superpowers/results/2026-11-15-objects_live7b.md`.

- [ ] **Step 1:** `costed_replay.py RUN_DIR CLASS COSTS_CSV`: per node, walks its ai2_log rows; L_product from
  `p_occ`, `p_cls`; occ_against from the sign of the row-to-row change of `ex_L`; cls_against from `lam < 0` on rows where
  `lam` changed; prints the first cycle and time at which the costed rule would remove, and whether the old rule removed
  (from the agent log). Ruling recorded: the spec's replay over objects_live4–6 runs on objects_live7a instead — the old
  traces carry the broken carve, under which every real object sits at −4.
  **Pass on objects_live7a:** no real object removed; any phantom of the run removed, with its time vs the old rule.
- [ ] **Step 2:** Claim Webots (chat + room), run the same tour as `objects_live7b`, SIGTERM, FREE.
- [ ] **Step 3:** Report against spec §9 "Live 7b": bed, both nightstands, toilet never removed; each removal line's L,
  line, costs and sources; a phantom removed no later than today; ~10 Hz. Commit.

---

## Spec corrections made with this plan (rev 2.1)

- Q3: "argued against" for the class factor is its own Λ < 0; the parenthesis `Λ − singleton < 0` contradicted the rule.
- Part fractions are measured numbers (as the manifests already carry); no Webots asset is copied.
- Replay of the costed decision runs on objects_live7a, and the carve replay's phantom check uses synthetic phantoms
  (bed_8 lives in room_1, whose frame cannot be tied to the world from the logs).
- The silhouette diagnostic in 7a is the share of silhouette vs LiDAR absence on drop cycles.
