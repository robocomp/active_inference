# door_concept one-model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** door_concept decides every door question (which cause explains a mask, is a door born, is it still there,
where to look) from one labelled-Bernoulli model with honest two-sided evidence, removing the ten hard switches S1–S10.

**Architecture:** A self-contained math header (`core/bernoulli_filter.h`) holds the filter (log-odds Bernoulli update,
responsibilities, k-best joint hypotheses, discrete EIG). Door knowledge stays in door files: priors (`place_prior.h`,
new `wall_base_rate.h`, memory reader `door_memory_rooms.h`), mask causes (new `door_mask_causes.h`), leaf-to-frame-wall
hypotheses (new `door_leaf_birth.h`), the LiDAR opening likelihood (`door_opening_edges.h`) and the hypothesis look
(new `door_hypothesis_look.h`). Everything new runs behind ONE flag, `Existence.OneModel` (default false); flag off
reproduces today's decisions bit for bit on replay tapes.

**Tech Stack:** C++23, Eigen, OpenCV, DSR/cortex graph, standalone g++ test programs with a per-file `CHECK` macro,
Python 3 (`-I`) for log tools, Webots for live runs.

**Spec:** `docs/superpowers/specs/2026-11-14-door-agent-one-model-design.md` (approved 2026-11-14; read §12 errata —
it overrides older wording in §3). Interfaces recon: memory file `door-recon-11-14-gates.md` and the scratchpad
`door_interfaces.md` (copied to the plan workspace as `interfaces.md` at setup).

**Worktree:** `/home/pbustos/robocomp/components/active_inference.worktrees/door-one-model`, branch
`feat/door-one-model`. All paths below are relative to `door_concept/` unless they start with `common/`, `ltsm_agent/`,
`room_concept/` or `docs/`.

## Global Constraints

- No thresholds, gates, clamps or σ-floors where a model term exists. A kept one carries a one-line `// kept:` comment
  saying why no model term does its job. Physical model parameters (sensor σ, wall/leaf thickness) are not thresholds.
- Type-attributed DSR attribute access only (`get_attrib_by_name<x_att>`, `add_or_modify_attrib_local<x_att>`), never
  `runtime_checked_*`. Queued DSR signals only. No new cortex attribute is expected; if one is needed, STOP that task and
  report (owner reinstalls cortex).
- Files: read numbers with `std::from_chars`, write streams after `imbue(std::locale::classic())`. Harnesses that parse
  agent files call `std::setlocale(LC_ALL, "")` first.
- `and` / `or` / `not`; C++23 containers and algorithms.
- Build with `make -C build -j8` (never more than -j8) after `pgrep -a cc1plus` shows no other build; standalone tests
  with the g++ line in their file header.
- `common/instance_tracker` is NOT edited. room_concept, controller, cortex are NOT edited.
- Flag `Existence.OneModel` (bool, default false): every behaviour change is under it. Flag off = today's decisions,
  verified by `tools/door_replay` exit 0 on the Task 1 tapes.
- Typical-size prior: width 0.85 ± 0.12 m, height 2.05 ± 0.06 m (keys `Existence.TypicalWidthM` 0.85,
  `Existence.TypicalWidthStd` 0.12, `Existence.TypicalHeightM` 2.05, `Existence.TypicalHeightStd` 0.06). `Door.Prior*`
  (the fit's prior) is not changed.
- Per-wall rates: Beta beliefs seeded Beta(2, 3) (first door, mean 0.4) and Beta(0.05, 4.95) (another door, mean 0.01),
  updated with counts from LTSM's remembered rooms in the graph, excluding the room the robot is in.
- Decision loss unchanged: `core/door_decision.h` `decide()` and etc/door_costs.csv.
- Stop agents with SIGTERM only. Webots is shared: send CLAIM before and FREE after every live run to the `chat` and
  `objects` sessions. Disk: `door_inputs.bin` is 2–8 MB/s; record only in Task 1; stop a series if `/` has < 15 GB free.
- Commit locally on `feat/door-one-model` with the trailer lines below; never push or merge.
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_019DrdF96H7KKejaoSX5bLys`
- Rates before features: door_concept's cycle stays at its nominal rate (measured in Task 13 against flag off).
- Pre-existing test failures (do not fix, do not count as regressions): test_door_core_existence (4), test_door_core_tracker
  (5). Record the baseline of every test you touch before editing.

## Review Focus

1. A door with p_D = 0 for a long time (behind the robot, occluded by a wall or another door's leaf): its L must not move.
   → Task 7 test `occluded_door_holds`.
2. A mask that two live doors both explain about equally (adjacent doors on one wall): neither is removed, nothing is
   born. → Task 7 test `shared_mask_splits`.
3. An empty memory (first run, no remembered rooms) and a memory whose rooms have no walls: rates equal the seeds, no NaN.
   → Task 4 test `rates_from_empty_memory`, `room_without_walls_ignored`.
4. A provisional (proto) room with no estimated wall near a leaf: the leaf hypothesis is kept as a candidate, not killed
   and not born in a wrong wall. → Task 9 test `leaf_in_proto_room_waits`.
5. The robot crossing a door while the new path is on: a door being crossed is never removed (Role::BeingCrossed) even
   when its masks go to clutter. → Task 7 test `being_crossed_never_removed`.

---

### Task 1: Recording tapes of the three cases (controller, live — not a subagent task)

**Files:** none in the repo; run folders under `room_concept/estimation_scripts/crossing/`.

- [ ] **Step 1:** In the MAIN checkout's `door_concept/etc/config.toml` (the one the fleet runs), set
  `RecordInputs = true` and `DecisionLog = true`. Rebuild nothing: this is today's code (main 87b6018a).
- [ ] **Step 2:** CLAIM Webots. Run the dm12-type tour: `RUN_LIMIT_S=1800 bash room_concept/estimation_scripts/scripts/explore_tour.sh tape_dm`
  with `START` unset (big room). After it, copy `door_concept/etc/door_inputs.bin` and `etc/door_decisions.bin` into
  `crossing/tape_dm/`.
- [ ] **Step 3:** Ask the chat session (message) to repeat its chat_e2e18 and chat_e2e19 scenarios once each with the
  recording on, as runs `tape_e2e18` and `tape_e2e19`, and copy the two .bin files into each run folder. If the chat
  session cannot, run them from its recorded mission (chat_e2e18/events.jsonl) with the language_agent fleet.
- [ ] **Step 4:** Check each tape replays equal on today's code: build `tools/door_replay` with the g++ line in its header
  into `/tmp/door_replay_base`; run `/tmp/door_replay_base crossing/<run>/door_inputs.bin crossing/<run>/door_decisions.bin`.
  Expected: exit 0 for all three. Check each run shows its bug in `door_existence_log.csv` / `door_events.csv` (e2e18:
  restored door removed then re-born on the leaf plane; e2e19: geo_dl alternating sign; dm: a door created within 0.3 m of
  a crossed door). A tape that does not show its bug is still kept, and the plan's replay tests use the original run's
  logs for that case.
- [ ] **Step 5:** Set `RecordInputs = false` and `DecisionLog = false` again in the main checkout. FREE Webots. Record the
  three run names in the ledger.

### Task 2: Diagnose the existence swing (chat_e2e19)

**Files:**
- Create: `tools/existence_swing.py`
- Create: `tests/test_existence_swing.py`
- Modify: `docs/superpowers/specs/2026-11-14-door-agent-one-model-design.md` §3.7 (write the cause)

**Interfaces:**
- Produces: `python3 -I tools/existence_swing.py <door_existence_log.csv> [--node NAME]` prints, per door, the count of
  sign changes of `geo_dl` between consecutive rows, the median |geo_dl|, and the rows where |Δgeo_dl| is largest with
  their `geo_x, geo_mu0, geo_mu1, geo_s, sup_x, cx, cy`. Exit 0.

- [ ] **Step 1: Write the failing test** `tests/test_existence_swing.py`: builds a 6-row CSV with the real 38-column header
  (`cycle,node,L,p_exists,cx,cy,role,delete_line,view_id,ric_view_id,prior_band,prior_base,prior_size,carried,crossing,ledger,n_views,zed_dl,ric_dl,geo_dl,act_dl,zed_q,ric_q,zed_hit,ric_hit,kappa,geo_x,geo_mu0,geo_mu1,geo_s,sup_x,sup_dl,Sc,Sp,sat_term,sup_c,sup_q,prior_extent`)
  and geo_dl = +3,−3,+3,−3,+3,−3 for node door_2; asserts the output line for door_2 reports `sign_changes=5`.
- [ ] **Step 2:** Run `python3 -I tests/test_existence_swing.py` → FAIL (no tool).
- [ ] **Step 3:** Implement the tool (csv module; no pandas).
- [ ] **Step 4:** Run the test → PASS. Run the tool on `room_concept/estimation_scripts/crossing/chat_e2e19/door_existence_log.csv --node door_2`
  and, if Task 1 produced `tape_e2e19`, replay it with `--csv /tmp/e2e19_csv --trace door_2` to see per-frame geometry.
- [ ] **Step 5:** Decide the cause from the numbers: is `geo_x` alternating between two values (two explanations picked in
  turn) or is `geo_mu0/mu1/s` moving (the model, not the measurement)? Write the finding, with row numbers and values,
  into spec §3.7, replacing "Not yet diagnosed", and state which later task carries the fix (Task 6 if the geometry
  likelihood must marginalise; otherwise a new task added to this plan by the controller).
- [ ] **Step 6: Commit** `tools/existence_swing.py tests/test_existence_swing.py docs/...design.md` with message
  `door_concept: diagnose the existence swing of chat_e2e19`.

### Task 3: The flag, the typical-size prior and the per-wall rate as a function of other doors

**Files:**
- Modify: `src/door_config.h` (members), `src/door_config_fields.h` (readers), `etc/config.toml` (shipped values)
- Create: `src/core/wall_base_rate.h`
- Modify: `src/core/place_prior.h:85-95` (base rate), `src/core/existence_update.cpp` `prior_of` (~302),
  `src/core/door_existence.cpp` (~372-402 candidates, ~546-555 doors)
- Test: `tests/test_wall_base_rate.cpp` (new), `tests/test_place_prior.cpp`, `tests/test_door_config_load.cpp`

**Interfaces:**
- Produces (config, `rc::DoorConfig`): `bool exist_one_model = false;` (`Existence.OneModel`), `float exist_typical_w_m = 0.85f, exist_typical_w_std = 0.12f, exist_typical_h_m = 2.05f, exist_typical_h_std = 0.06f;`
- Produces (`src/core/wall_base_rate.h`, namespace `rc::door::core`, header-only):
  ```cpp
  struct BetaBelief { double a; double b; [[nodiscard]] double mean() const; };
  struct WallRates { BetaBelief first{2.0, 3.0}; BetaBelief another{0.05, 4.95}; };
  /// log prior odds of "this wall has this door" when the other doors on its wall exist with probabilities r_others:
  /// m = 1 − Π(1 − r_j);  return (1 − m)·logit(first.mean()) + m·logit(another.mean()).
  [[nodiscard]] float wall_base_rate(const WallRates&, std::span<const float> r_others);
  ```
- Produces (`PlacePriorInput`): new fields `const WallRates* rates = nullptr; std::span<const float> r_others = {};`
  When `rates` is non-null, `place_prior` uses `wall_base_rate(*rates, r_others)`; else today's integer rule (flag off).

- [ ] **Step 1: Write failing tests** `tests/test_wall_base_rate.cpp` (header g++ line:
  `g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_wall_base_rate.cpp -o /tmp/test_wall_base_rate`):
  - `seed_means`: `WallRates{}.first.mean() == 0.4`, `.another.mean() == 0.01` (to 1e-9).
  - `no_others_is_first`: `wall_base_rate(WallRates{}, {})` == `log(0.4/0.6)` (1e-5).
  - `certain_other_is_another`: `r_others = {1.0f}` → `log(0.01/0.99)` (1e-5).
  - `uncertain_other_pays_part`: `r_others = {0.5f}` → strictly between the two, equal to the 0.5 blend (1e-5).
  - `order_independent`: `{0.2f, 0.7f}` and `{0.7f, 0.2f}` give equal values.
  Add to `tests/test_place_prior.cpp`: `rates_null_is_today` (same `PlacePrior.base_rate` as before for
  `doors_on_wall_before` 0 and 1) and `rates_given_uses_others`.
- [ ] **Step 2:** Build and run both → FAIL to compile.
- [ ] **Step 3:** Implement `wall_base_rate.h`; extend `PlacePriorInput` / `place_prior`.
- [ ] **Step 4:** Wire under `cfg.exist_one_model`: `DoorCore` holds a `WallRates rates_{}` (set by Task 4 later);
  candidates (~372-402) and doors (~546-555) pass `&rates_` and the PREVIOUS cycle's `p()` of the other live doors and
  candidates on the same `wall_of(...)` wall (store a `std::unordered_map<std::uint64_t, float> last_p_` updated at the
  end of `update_existence`), so nothing depends on birth order. Also under the flag, `prior_of` builds its
  `SizePrior` from the four `exist_typical_*` fields instead of `door_prior_*`.
- [ ] **Step 5:** Add the five keys to `door_config.h`, `door_config_fields.h` (getb/getf with one-line descriptions) and
  `etc/config.toml` under `[Existence]` (`OneModel = false` plus the four typical values).
- [ ] **Step 6:** Run `test_wall_base_rate`, `test_place_prior`, `test_door_config_load`, `test_door_core_existence`,
  `test_door_core_tracker` → new tests PASS, old failures unchanged from baseline. Build `tools/door_replay` from
  this branch into `/tmp/door_replay` and run it on each Task 1 tape → exit 0 (flag off is identical).
- [ ] **Step 7: Commit** `door_concept: per-wall rate from the other doors' beliefs; typical-size prior; OneModel flag`.

### Task 4: Learn the per-wall rates and the room door-count prior from memory

**Files:**
- Create: `src/door_memory_rooms.h`
- Modify: `src/specificworker.cpp` (once per room entry and every N s on the existing slow timer: read, set
  `core_->set_wall_rates(...)`; `step_candidate_looks` ~4514-4517 count prior), `src/core/door_core.h` (setter)
- Test: `tests/test_door_memory_rooms.cpp` (new)

**Interfaces:**
- Consumes: `WallRates`, `BetaBelief` (Task 3); `count_prior_from_rooms(std::span<const int> counts, double mean_n, int n_max)` (core/room_door_count.h:62).
- Produces (`src/door_memory_rooms.h`, namespace `rc::door::memrooms`):
  ```cpp
  struct RememberedWall { int doors = 0; };
  struct RememberedRoom { std::uint64_t room_id = 0; std::vector<RememberedWall> walls; };
  /// Pure: Beta counts. first: a += walls with ≥1 door, b += walls with 0; another: over walls with ≥1 door,
  /// a += walls with ≥2, b += walls with exactly 1.
  [[nodiscard]] core::WallRates rates_from_rooms(std::span<const RememberedRoom>, core::WallRates seed = {});
  [[nodiscard]] std::vector<int> door_counts(std::span<const RememberedRoom>);   // doors per room, for count_prior_from_rooms
  /// Graph reader: LTSM memory room twins (type "room", name prefix r<k>_), their wall twins (type "wall") and door
  /// twins (is a door per ltsm's is_door), EXCLUDING the twin of `current_room_id`'s memory room. A door is assigned to
  /// the wall twin it hangs under if its parent is a wall; otherwise to the nearest wall segment in the memory room frame.
  [[nodiscard]] std::vector<RememberedRoom> read_remembered_rooms(DSR::DSRGraph&, std::uint64_t current_memory_room_id);
  ```
- Produces (`DoorCore`): `void set_wall_rates(const WallRates&);`

- [ ] **Step 1: Verify the memory shape** before coding the reader: load an archived memory (`crossing/dm12/ltsm_history`
  or the ltsm_agent etc/ history file the fleet uses) with ltsm_agent's own loader in a scratch harness, or read
  `ltsm_agent/src/room_eviction.cpp` `copy_subtree` / `hang_walls_on_floor` and `twin()`, and write in the task report
  which node is each door twin's parent (wall twin, floor twin or room twin), with one example. The reader's door-to-wall
  rule follows what you found (parent wall if present, geometry otherwise).
- [ ] **Step 2: Write failing tests** (`tests/test_door_memory_rooms.cpp`, pure part only, g++ line with Eigen):
  - `rates_from_empty_memory`: `rates_from_rooms({})` equals the seed exactly.
  - `rates_from_rooms_counts`: two rooms, walls doors {1,0,0,1} and {2,0,0,0} → first = Beta(2+3, 3+5) and
    another = Beta(0.05+1, 4.95+2).
  - `room_without_walls_ignored`: a room with zero walls changes nothing and gives no NaN.
  - `door_counts_per_room`: returns {2, 2} for the rooms above.
- [ ] **Step 3:** Build/run → FAIL. **Step 4:** Implement the pure functions → PASS.
- [ ] **Step 5:** Implement `read_remembered_rooms` with type-attributed attribute access only (`name`, `parent_att`,
  RT edges via `rt_translation_att`/`rt_rotation_euler_xyz_att`), and a graph test in the style of
  `tests/test_door_memory_graph.cpp` (registered in tests/CMakeLists.txt like it): a fake memory with one room twin,
  four wall twins, two door twins under one wall, and the current room's twin (which must be excluded) →
  `rates_from_rooms(read_remembered_rooms(...))` = first Beta(2+1, 3+3), another Beta(0.05+1, 4.95+0).
- [ ] **Step 6:** Wire in specificworker under `cfg.exist_one_model`: on room entry and on the existing slow
  (≥ 1 s) timer, `core_->set_wall_rates(rates_from_rooms(read_remembered_rooms(*G, mem_room_id)))`; in
  `step_candidate_looks` replace `geometric_count_prior(2.0, max(1, n_max))` by
  `count_prior_from_rooms(door_counts(rooms), 2.0, max(1, n_max))` (the 2.0 stays as the seed mean of that function's
  own prior). Flag off: unchanged.
- [ ] **Step 7:** Run the new tests, `test_room_door_count`, the agent build `make -C build -j8`. Commit
  `door_concept: per-wall rates and the room door-count prior learned from remembered rooms`.

### Task 5: Fit the detector table and the LiDAR/geometry likelihoods from the tapes

**Files:**
- Modify: `tools/door_detect_rate.cpp` (if needed, to emit per-view rows with GT labels), `tools/fit_detector_envelopes.py`
  (fit `false_alarm` too, and the Ricoh rows)
- Create: `tools/fit_geometry_channel.py`
- Modify: `etc/door_detectors.csv` (fitted values, `source` = the tape names, `n` > 0)
- Test: `tests/test_fit_detector_envelopes.sh` (existing; extend), `tests/test_fit_geometry_channel.py` (new)

**Interfaces:**
- Produces: `etc/door_detectors.csv` rows for camera `zed` and `ricoh`: `hit_rate`, `false_alarm` (κ per view), plus
  new rows `lidar,through_open` (P(a bin reads "through" | opening, in view)) and `lidar,through_wall` (P(through | solid
  wall)), and `geometry,mu1_sd` (replaces the fixed `geo_mu1_sd = 2`). `load_detectors` (core/detector_llr.h:164)
  learns the new rows with today's values as defaults when absent (so old files load unchanged).
- Ground truth: the Webots world's true door poses (the file `tools/door_detect_rate.cpp` already uses; Webots doors are
  the labels; a mask is a hit if it lies within the true door's aperture footprint, a false alarm otherwise).

- [ ] **Step 1: Write failing tests**: `tests/test_fit_geometry_channel.py` on a synthetic CSV of (geo_x, label) where
  label 1 rows are N(2, 1) and label 0 rows N(0, 1) → fitted `mu1_sd` within 0.1 of 2.0. Extend
  `test_fit_detector_envelopes.sh` with a fixture where 10 % of unlabelled-door masks are false alarms → fitted
  `false_alarm` 0.10 ± 0.02.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** → PASS.
- [ ] **Step 5:** Replay each Task 1 tape with `--csv` to produce `door_views_log.csv`; run the fitters over all three;
  write `etc/door_detectors.csv`. Record n per row in the report. Extend `load_detectors` for the new rows (default =
  today's compiled values) and `tests/test_detector_model.cpp` with `old_file_loads_with_defaults`.
- [ ] **Step 6:** Flag-off replay exit 0 still (the table is read but the flag-off path must not use the new rows; if a
  fitted ZED/Ricoh row changes flag-off decisions, keep the fitted table in `etc/door_detectors_fitted.csv`, loaded only
  under the flag). Commit `door_concept: detector and geometry likelihoods fitted from tapes with Webots labels`.

### Task 6: Every channel two-sided; truncation as a loss of information

**Files:**
- Modify: `src/core/view_ledger.h:92, :135, :139-143` ; `src/core/existence_update.cpp:20-29, :54, :74-78, :119, :178-200` ;
  `src/core/door_core.cpp:185-198` (`frame_admissible`), `src/core/door_fit.cpp:883-885`
- Test: `tests/test_view_ledger.cpp`, `tests/test_door_core_existence.cpp`

**Interfaces:**
- Consumes: fitted rows from Task 5 (`DetectorTable` ricoh `hit_rate`/`false_alarm`, geometry `mu1_sd`).
- Produces: `ViewLedger` gains a member `bool ricoh_counts_misses = false` (set from `cfg.exist_one_model`; also passed
  through `ViewLedger::parse`). `FrameDetection` gains `float trunc_frac = 0.f`. Under the flag
  `FrameDetection::admissible` is always true, and truncation acts only as a weight `(1 − trunc_frac)` on the size and
  support terms; place and existence evidence keep the full weight.

- [ ] **Step 1: Write failing tests** in `tests/test_view_ledger.cpp`:
  - `ricoh_miss_counts_under_flag`: a Ricoh view with q 0.8 and no hit lowers the view llr by `miss_llr(q, f)`; with the
    flag off, unchanged.
  - `geometry_may_refute_under_flag`: a geometry sample at mu0 − 2s gives a negative llr under the flag; 0 flag off.
  In `tests/test_door_core_existence.cpp`:
  - `truncated_view_pays_no_size_cost`: a 40 % truncated mask of a 0.80 m door: under the flag the place's L gains
    existence evidence from that frame and the size term is not charged (prior_size equal to a non-truncated view's
    within its widened sd).
  - `occluded_door_holds` (Review Focus 1): door with p_D 0 for 300 frames → L unchanged (1e-6).
- [ ] **Step 2:** Build/run (g++ lines in the file headers) → new tests FAIL.
- [ ] **Step 3:** Implement under the flag: Ricoh misses in `add_bernoulli`/`recompute` (no `i == 0` restriction) and in
  `observe_camera` / the hits-only branch (use the fitted Ricoh model); geometry clamp removed and `geo_mu1_sd` from the
  table; truncation: `frame_admissible` returns true under the flag, the truncation fraction travels on the detection and
  widens the size likelihood (the candidate path already does `width_sd / sqrt(1 − trunc)` at door_existence.cpp:328 —
  use the same form for the fit's support sample at door_fit.cpp:883 instead of skipping it).
  If Task 2 found the swing is the geometry channel choosing one explanation per frame, also here: the geometry
  likelihood marginalises over the explanations Task 2 named (log-sum-exp of their likelihoods weighted by their
  posterior), with a test `geometry_swing_marginalised` built from the Task 2 rows (alternating inputs → |Δllr| per
  update below the two-explanation difference).
- [ ] **Step 4:** Run → PASS; old failures unchanged; flag-off replay exit 0.
- [ ] **Step 5: Commit** `door_concept: two-sided Ricoh and geometry; truncation widens size, never erases a view`.

### Task 7: The filter math and the four causes of a mask

**Files:**
- Create: `src/core/bernoulli_filter.h` (no door, DSR, Qt or OpenCV includes; Eigen and std only)
- Create: `src/core/door_mask_causes.h`
- Modify: `src/door_belief.h/.cpp:112-165` (leaf marginal over all four branches), `src/core/door_core.cpp:611-689`
  (assignment), `:728-798` (S1, S2), `src/core/door_existence.cpp:290-435` (S8, S9 gates), `src/core/existence_update.cpp:112-125`
  (`association_weight`)
- Test: `tests/test_bernoulli_filter.cpp` (new), `tests/test_door_mask_causes.cpp` (new), `tests/test_door_core_existence.cpp`,
  `tests/test_door_restored_refuted.cpp`

**Interfaces:**
- Produces (`src/core/bernoulli_filter.h`, namespace `rc::door::core::bf`):
  ```cpp
  /// Exact Bernoulli update in log-odds: L' = L + log(1 − p_d + p_d · S), S = Σ_m ρ_m · g_m / κ_m.
  [[nodiscard]] double update_logodds(double L, double p_d, double S);
  /// Normalised responsibilities from log weights (log-sum-exp, stable); sums to 1.
  [[nodiscard]] std::vector<double> responsibilities(std::span<const double> log_w);
  struct Joint { std::vector<int> cause_of_mask; double log_w; };
  /// k best joint assignments of masks to causes. log_w(m, c) = log weight of mask m under cause c (−inf = impossible).
  /// `exclusive[c]` = cause c explains at most one mask (a door per sensor); non-exclusive (new door, clutter) any number.
  [[nodiscard]] std::vector<Joint> k_best(const Eigen::MatrixXd& log_w, std::span<const bool> exclusive, int k);
  /// Discrete EIG: H(w) − Σ_z p(z) H(w | z), p_z_given_h[h][z].
  [[nodiscard]] double expected_information_gain(std::span<const double> w, const std::vector<std::vector<double>>& p_z_given_h);
  ```
- Produces (`src/core/door_mask_causes.h`, namespace `rc::door::core`):
  ```cpp
  enum class CauseKind : std::uint8_t { Aperture, Leaf, NewDoor, Clutter };
  struct Cause { CauseKind kind; std::uint64_t door = 0; double log_w = 0; };   // log g + log(r·p_D) or log b or log κ
  struct MaskCauses { int slice = -1; std::vector<Cause> causes; std::vector<double> rho; };
  /// One mask's causes: every live door/candidate within likelihood reach (log g not below log κ − 20 nats, a
  /// numerical floor, not a decision), the new-door birth density, clutter.
  [[nodiscard]] MaskCauses causes_of_mask(/* mask points + R, live doors' beliefs, r, p_D, birth log density, log κ */);
  ```
  Exact parameter list is the implementer's, built from existing pieces: aperture `association_nll(pts, R + depth_var)`
  (door_core.cpp:652), leaf `DoorBelief::leaf_loglik_marginal(pts, R, const LeafChannel&)` (new, replaces the single-swing
  use of `clutter_fraction_leaf_marginal`: marginalise φ per branch over the four `LeafChannel::phi` filters weighted by
  the hinge and swing beliefs), new door `place_prior(...).total()` + `typical_door_at` likelihood, clutter `log κ` from the
  fitted detector table.
- Integration rule (under the flag): ρ(mask → door i) is the `hit_w` passed to `ViewLedger::add_bernoulli` for door i;
  ρ(mask → new door) is the detection's `birth_evidence`; the fit's support sample is weighted by ρ(mask → door i).
  The swept-arc / aperture-quad veto (S1), claimed-slice veto and greedy pairing (S2), one-detection-per-candidate (S8)
  and the hard gate sites (S9: door_existence.cpp ~298-355, door_core.cpp ~647, ~684, existence_update.cpp ~26, ~122) are
  bypassed under the flag. The tracker's own gate params (door_core.cpp ~375-376) stay, as its bookkeeping only.

- [ ] **Step 1: Write failing tests** `tests/test_bernoulli_filter.cpp` (g++ with Eigen only):
  - `miss_with_pd_zero_is_noop`: `update_logodds(1.3, 0.0, 0.0) == 1.3`.
  - `miss_lowers`: `update_logodds(0.0, 0.9, 0.0) == log(0.1)`.
  - `matches_probability_form`: for r = 0.3, p_d 0.8, S 5: `logistic(update_logodds(logit(0.3),0.8,5))` equals
    `r(1−pd+pd S)/(1−r pd + r pd S)` (1e-9).
  - `responsibilities_sum_to_one` and `responsibilities_stable` (log_w = {1000, 1000} → {0.5, 0.5}).
  - `k_best_matches_brute_force`: random 4×5 `log_w` (seeded), exclusive {1,1,1,0,0}, k = 5 → same top-5 as exhaustive.
  - `eig_bounds`: identical rows → 0; a perfectly discriminating 2×2 → H(w).
  `tests/test_door_mask_causes.cpp`:
  - `leaf_either_swing_explains`: a side-on leaf mask of a door whose swing belief is 0.9 for the WRONG side: ρ(leaf) > ρ(clutter).
    (Today's `clutter_fraction_leaf_marginal` fails this: chat_e2e18.)
  - `known_aperture_takes_mask`: a mask on a born door's aperture → ρ(aperture) > 0.9, ρ(new) < 0.05.
  - `shared_mask_splits` (Review Focus 2): two adjacent doors equally close → ρ ≈ 0.5 each (± 0.05).
  In `tests/test_door_core_existence.cpp` (flag on):
  - `open_leaf_confirms_restored_door`: the chat_e2e18 geometry (restored open door, 90° leaf seen side-on, wrong swing
    belief) → the door's L rises over 30 frames; no birth.
  - `being_crossed_never_removed` (Review Focus 5): Role::BeingCrossed with all masks to clutter → not removed.
  - `parked_phantom_never_born`: 600 frames of κ-rate clutter at one place → no birth.
  - `good_views_fewer_frames`: a near, centred door is born in fewer views than a far, oblique one.
- [ ] **Step 2:** Build/run → FAIL. **Step 3:** Implement `bernoulli_filter.h` (Murty or exact enumeration with pruning —
  the implementer's choice — must pass `k_best_matches_brute_force`), `door_mask_causes.h`, `leaf_loglik_marginal`, and the
  integration under the flag. **Step 4:** → PASS; old failures unchanged; flag-off replay exit 0.
- [ ] **Step 5:** Flag-on replay of `tape_e2e18`: `/tmp/door_replay` with a config override `Existence.OneModel=true`
  (add a `--set KEY=VALUE` option to the tool if it has none) and `--csv`: the restored door is not removed and no door is
  born on the leaf plane. Put the before/after lines in the report.
- [ ] **Step 6: Commit** `door_concept: every mask is shared among aperture, leaf, new door and clutter (S1 S2 S8 S9)`.

### Task 8: Duplicates by competition, not separation or merge (S3)

**Files:**
- Modify: `src/core/door_core.cpp:341-367` (merge), `:390` (`tp.birth_min_sep_m`)
- Test: `tests/test_door_core_tracker.cpp`

**Interfaces:**
- Consumes: Task 7 responsibilities; Task 3 `wall_base_rate` with r_others.
- Under the flag: `merge()` returns at once; `tp.birth_min_sep_m` = `cfg.tracker_detection_noise_m` (the tracker's
  own noise scale, so it still keeps one track per detection; reason in a `// kept:` comment).

- [ ] **Step 1: Write failing tests** (flag on) in `tests/test_door_core_tracker.cpp`:
  - `mirror_of_crossed_door_falls`: dm12 geometry — a crossed door (Role from `route_faces`) and an entry-mirror candidate
    0.17 m away fed the same masks → the candidate is never born, or if born falls below the delete line within 60 frames;
    the crossed door stays.
  - `two_real_doors_close_both_live`: two true doors 1.0 m apart on one wall, each with its own masks and openings → both
    born (the second-door cost is paid but evidence wins).
  - `removing_one_frees_other`: after one of them is removed, the other's `prior.base_rate` returns to the first-door value
    within one cycle.
  Also fix the stale header comment of the file (no "outside room" / "top" suppressions exist).
- [ ] **Step 2:** → FAIL. **Step 3:** Implement. **Step 4:** → PASS; flag-off replay exit 0.
- [ ] **Step 5:** Flag-on replay of `tape_dm` → no door created within 0.3 m of a crossed door (report the lines).
- [ ] **Step 6: Commit** `door_concept: two labels at one hole compete; no separation, no merge (S3)`.

### Task 9: A leaf seen side-on points to an opening in its frame wall

**Files:**
- Create: `src/door_leaf_birth.h`
- Modify: `src/door_opening_edges.h` (add the likelihood ratio), `src/core/door_mask_causes.h` (new-door cause gains the
  leaf mode), `src/core/door_core.cpp` births (aperture of the winning hypothesis)
- Test: `tests/test_door_leaf_birth.cpp` (new), `tests/test_door_core_existence.cpp`

**Interfaces:**
- Produces (`src/door_opening_edges.h`):
  ```cpp
  /// log p(sweep | opening along ap's span) − log p(sweep | solid wall there), from measure_openings' bins:
  /// through bins count log(p_open/p_wall), blocked bins log((1−p_open)/(1−p_wall)), unknown bins 0.
  /// p_open = through_open · clear_fraction(phi_curve); p_wall = through_wall (Task 5 fitted rows).
  [[nodiscard]] float opening_llr(const Aperture& ap, const RegisteredSweep& sw, float through_open, float through_wall,
                                  std::span<const std::pair<float,float>> phi_curve);
  ```
- Produces (`src/door_leaf_birth.h`, namespace `rc::door`):
  ```cpp
  struct LeafApertureHyp { Aperture aperture; LeafState leaf; int wall = -1; double log_w = 0; };
  /// From a side-on leaf mask (its two vertical edges in the room frame and its height): for each end as the hinge,
  /// the frame wall is the wall (wall_groups) that end touches under the band prior; the aperture runs along that wall
  /// from the hinge on either side, width = the leaf length under the typical-size prior, φ = angle(leaf, wall),
  /// swing = the side of the wall the leaf is on. log_w = band + size + base rate (+ opening_llr, added by the caller).
  [[nodiscard]] std::vector<LeafApertureHyp> apertures_from_leaf(const Eigen::Vector2f& end_a, const Eigen::Vector2f& end_b,
      float height_m, std::span<const Eigen::Vector2f> polygon, std::span<const int> wall_groups,
      const core::SizePrior& typical, const RoomBand& band);
  ```

- [ ] **Step 1: Write failing tests** `tests/test_door_leaf_birth.cpp`:
  - `leaf_perpendicular_to_wall_gives_frame_wall`: room 5×5, leaf from (2.0, 5.0) to (2.0, 4.15) (hinge on the y = 5 wall)
    → best hypothesis' aperture lies on the y = 5 wall, centre x ∈ {1.575, 2.425} ± 0.05, never on the x walls, and φ ≈ 90°.
  - `opening_llr_sign`: a synthetic sweep with through returns along the aperture span → > 0; all blocked → < 0; no
    returns → 0.
  - `leaf_in_proto_room_waits` (Review Focus 4): proto room, no wall within band reach of either end → no hypothesis gets
    a finite birth weight above the clutter one; the candidate is kept, not removed.
  In `tests/test_door_core_existence.cpp` (flag on):
  - `side_on_leaf_bears_door_in_frame_wall`: leaf mask + opening seen at its hinge end, no known door → a door is born
    whose aperture is on the frame wall (not in the leaf's plane).
  - `corner_two_doors`: open door A in wall 1 whose leaf hides door B's aperture in wall 2 (corner): with A's opening in
    the sweep, A wins (ρ(A leaf) > 0.8); with neither opening visible, the cluster keeps ≥ 2 joint hypotheses with weights
    in (0.2, 0.8).
- [ ] **Step 2:** → FAIL. **Step 3:** Implement; the new-door cause's log weight is the log-sum-exp of its aperture mode
  and its leaf-mode hypotheses (each plus its `opening_llr` when a registered sweep exists this cycle). Births create the
  door at the winning hypothesis' aperture. **Step 4:** → PASS; flag-off replay exit 0.
- [ ] **Step 5: Commit** `door_concept: a side-on leaf and the LiDAR opening bear a door in its frame wall`.

### Task 10: Restore through responsibility (S10)

**Files:**
- Modify: `src/door_identity_table.h:318-352, :618, :667`, `src/core/place_table.h:~125`, `src/specificworker.cpp`
  (`step_restore_from_notes`)
- Test: `tests/test_door_restore_from_notes.cpp`, `tests/test_door_identity_table.cpp`

- Under the flag: a restored note is a labelled Bernoulli (its note's L and state) that takes part in Task 7's causes;
  `plan_restore`'s same-hole radius skip and `lift_refutations`' radius are replaced by "a live door already carries this
  note's responsibility" (the note's place is explained by a live door's aperture with ρ > 0.5 → adopt into it, do not
  restore a second); the dated refutation rule (04d256d7) stays, with a `// kept: memory, not evidence` comment;
  `kFitWithoutFreshDataFrames` is replaced by "no new view in the ledger since the last fit".
- [ ] **Step 1: Write failing tests** (flag on): `note_adopted_by_live_door` (live door at 0.3 m from the note's place,
  explaining its masks → no second door), `note_restored_far_from_live` (live door 1.5 m away on another wall → restored),
  `refuted_note_still_dated` (the existing dated cases pass unchanged).
- [ ] **Step 2–4:** FAIL → implement → PASS; flag-off replay exit 0; `ctest`-registered tests in tests/CMakeLists.txt run
  via their binaries (ctest is broken here).
- [ ] **Step 5: Commit** `door_concept: a remembered door is restored unless a live door already explains it (S10)`.

### Task 11: The hypothesis look

**Files:**
- Create: `src/door_hypothesis_look.h`
- Modify: `src/specificworker.cpp` (`step_candidate_looks` ~4413-4630 and the remembered-look block ~4380-4430: under the
  flag, replaced by one producer), `src/core/door_core.h` (expose the cluster hypothesis sets)
- Test: `tests/test_door_hypothesis_look.cpp` (new)

**Interfaces:**
- Consumes: `bf::expected_information_gain` (Task 7), cluster `Joint` sets (Task 7/9), `rc::nbv::visible_fraction`
  (common/nbv/viewpoint_score.h:489), `rc::nbv::wall_obstacles` (:171), `rc::nbv::is_reachable` (:361),
  `rc::nbv::stands_inside` (:379), `opening_llr` inputs (Task 9), `rc::ObjectAffordance` (common/object_affordance).
- Produces (`src/door_hypothesis_look.h`, namespace `rc::door::hlook`):
  ```cpp
  struct Hyp { std::vector<Aperture> apertures; std::vector<LeafState> leaves; double w; };   // one joint hypothesis
  struct Question { std::string key; std::vector<Hyp> hyps; };   // a cluster; a single undecided place = {exists, not}
  struct Pose { Eigen::Vector2f xy; float yaw; };
  /// Reachable poses over the room floor: grid step = the robot's diameter, 8 headings (a sampling resolution, not a
  /// decision threshold), kept if is_reachable and stands_inside.
  [[nodiscard]] std::vector<Pose> room_poses(std::span<const Eigen::Vector2f> polygon, std::span<const rc::nbv::Obstacle>, float robot_radius_m);
  /// p(z | h, pose) over the discrete outcomes {each aperture/leaf seen by the camera or not} × {each aperture's opening
  /// seen through by the LiDAR or not}; camera visibility from visible_fraction with walls, furniture AND the other
  /// hypotheses' leaves as obstacles; LiDAR from line of sight to the aperture span; rates from the fitted detector table.
  [[nodiscard]] std::vector<std::vector<double>> outcome_table(const Question&, const Pose&, /* sensors, obstacles, detectors */);
  struct Best { Pose pose; double eig_nats = 0; };
  [[nodiscard]] Best best_look(const Question&, std::span<const Pose>, /* same */);
  ```
- Delivery (under the flag): one `rc::ObjectAffordance` per open Question, parent = the room node, name
  `aff_door_look_q_<key>`, object_type "door", `AffordanceTarget{x_m, y_m, yaw_rad, gain = eig_nats}`; withdrawn
  (`remove()`) when the question is settled (all hypothesis weights but one below 1e-3 is a numerical floor; the gain
  falling to ~0 does the rest through AffordanceManager's ranking). Producers (2) `aff_door_look_<name>` and (3)
  `aff_door_look_place_<key>` are not created under the flag (their questions are single-member Questions). Producer (1),
  the live door's D-optimal place look in epistemic_planner, stays: it refines a born door's place, a question H does not
  hold (reason in a comment).

- [ ] **Step 1: Write failing tests** `tests/test_door_hypothesis_look.cpp`:
  - `eig_zero_when_blind`: a pose facing away from every aperture → eig < 1e-6.
  - `eig_full_when_decisive`: two equally weighted hypotheses, a pose that sees A's opening through the LiDAR and not B's →
    eig > 0.9 · ln 2.
  - `corner_look_sees_one_opening`: the Task 9 corner, neither opening visible from the robot → `best_look` returns a pose
    from which at least one of the two openings is in LiDAR line of sight, with eig > 0.
  - `single_place_question`: an undecided candidate {exists 0.5, not 0.5} → best pose faces the place, eig > 0.
  - `room_poses_inside`: every pose lies inside the polygon and off obstacles.
- [ ] **Step 2:** → FAIL. **Step 3:** Implement; then wire the producer under the flag at the rate of the existing look
  step (no faster). **Step 4:** → PASS; agent builds `make -C build -j8`; measure `best_look` time for a 20 m² room with
  4 hypotheses and write it in the report (must fit the look step's period).
- [ ] **Step 5: Commit** `door_concept: one look per open question, at the pose that settles it most`.

### Task 12: Docs

**Files:**
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` (repo root) §1 and the door_concept CREATE row; `MODEL_HISTORY.md` if it names
  the vetoes; `door_concept/etc/config.toml` comments for S1–S10 keys (mark "unused when Existence.OneModel = true");
  spec status line → "BUILT (behind Existence.OneModel), series pending".
- [ ] **Step 1:** Write: a door is born by comparison inside door_concept; the shared tracker only holds candidates; every
  mask is shared among four causes; the per-wall rates are learned from memory. List each of S1–S10 with its fate and,
  for any kept, the reason.
- [ ] **Step 2: Commit** `docs: door_concept one model (lifecycle amendment, switch fates)`.

### Task 13: Acceptance (controller, live — not a subagent task)

- [ ] **Step 1:** Replays, flag on: `tape_e2e18`, `tape_e2e19`, `tape_dm` show the three fixes (Task 7 Step 5, Task 6 swing
  test numbers, Task 8 Step 5). Flag off: all three exit 0.
- [ ] **Step 2:** Rates: run one tour flag off and one flag on; compare door_concept cycle period (its log) — flag on must
  be within the nominal rate.
- [ ] **Step 3:** Live cases, flag on (CLAIM/FREE): ask the chat session to rerun chat_e2e18 and chat_e2e19 missions; run a
  dm-type tour. Pass = spec §7.1.
- [ ] **Step 4:** Series, flag on: the 15-run three-room series (order bed/bath/big ×5, `explore_tour.sh`, names
  `onemodel01..15`, scratchpad series script as in memory resume-11-13-auro-series) → pass ≥ 13/15; from-scratch tours
  find every true door; phantom births (door_events.csv births not matching a GT door) ≤ the flag-off reference (run 3
  flag-off tours for the count if no reference exists). Note this is a post-freeze build (not for the AuRo paper).
- [ ] **Step 5:** Report to the owner. Removing the flag and the old path is a separate step after the owner accepts.
