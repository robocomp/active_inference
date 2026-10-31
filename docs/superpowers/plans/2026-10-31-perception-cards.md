# Perception Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every concept agent publishes a perception card describing the detection model it actually runs. One
shared evaluator turns a card plus world facts into P(notice), P(recognise) and a limiting factor for any
hypothetical object and viewpoint. chat_agent answers "can you see X / where should you stand" with it.

**Architecture:**
- A header-only module `common/perception_card/` (Eigen + std, no DSR) holds four pieces:
  - the `Card` type and its JSON;
  - a **closed term library**;
  - the camera and LiDAR evaluators;
  - `where_to_stand`.
- Agents build their card from their effective config and publish it, both to their own agent node and to
  `etc/perception_card.json`.
- Terms that agents compute privately today are **moved** into the term library and called from there, so a
  card's terms and the agent's factors are identical by construction.

**Tech Stack:** C++23, Eigen, the DSR graph (cortex), CMake via `cbuild`, plain `main()` + assert tests.

**Spec:** `docs/superpowers/specs/2026-10-31-perception-cards-design.md`. It was approved on 2026-10-31,
including §4b (the LiDAR term) and §9b (the rulings).

## Global Constraints

- **Branch:** work in a new worktree `active_inference.worktrees/perception-cards`, branch `feat/perception-cards`,
  created FROM `feat/sensor-intrinsics`. It needs `rc::media::sensor_model_from_graph`, `lowest_elev_deg`,
  `highest_elev_deg` and `max_ring_gap_rad` from `common/media_transport/media_transport.h`.
- **CLAUDE.md:**
  - C++23 with `and`/`or`/`not`;
  - `std::from_chars` only, with tests that parse run under `std::setlocale(LC_ALL, "")`;
  - type-attributed DSR attributes only (`<foo_att>`), never `runtime_checked_*`;
  - never `Qt::DirectConnection`;
  - builds via `bash -ic 'cd <agent> && cbuild -j8'`, never above -j8, checking `pgrep -a cc1plus` first;
  - stop agents with SIGTERM only.
- **No thresholds** where a model term works. Every probability stays continuous; nothing returns "visible: yes/no".
- **New attribute:** `REGISTER_TYPE(perception_card, std::string, false)` in cortex `dsr_attr_name.h`. **The user
  reinstalls cortex** (never `sudo cp`). Until then, cards go to the file only (Task 6); the attribute write is
  added in Task 6b after the reinstall.
- **Ownership:** an agent writes its card only on its OWN agent node (the one with `agent_id`).
- **Behaviour:** no agent's existence behaviour changes, EXCEPT chair (Task 9), where wall line of sight is the
  intended fix.
- **cabinet_concept:** gets a card for what it runs today. Its model is not otherwise touched (the user, 2026-10-31).
- **Sensors:** ricoh is recorded on cards as bearing-only and is not modelled.
- **Commits** end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_019DrdF8AcGc5RMg89NxnGDz`.

## Review Focus

1. **No room polygon yet, or only a proto-room:** the evaluator must still answer, with walls omitted and that
   stated, and must not crash. Test in Task 3.
2. **No sensor model:** when a LiDAR descriptor carries no model (`nullopt`), that channel is skipped and named as
   "not modelled", never treated as P = 0. Test in Task 4.
3. **Impossible viewpoint:** a hypothetical object inside a wall, or a viewpoint inside an obstacle. Return
   P = 0 with the reason "that spot is inside <obstacle>"; never NaN. Test in Task 5.
4. **Unmeasured door φ** (`kUnmeasured`) for the articulated term: marginalise over the φ prior. Test in Task 8.
5. **No card for the asked concept:** `can_see` says "I have no detector for that"; it never guesses. Test in Task 12.

---

### Task 0: Request the cortex attribute (no code)

- [ ] Ask the user to add `REGISTER_TYPE(perception_card, std::string, false)` to
  `~/robocomp/components/cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h` and reinstall cortex.
  Tasks 1–6 do not need it.

### Task 1: The `Card` type and its JSON

**Files:**
- Create: `common/perception_card/card.h`
- Test: `common/perception_card/tests/card_test.cpp`, registered in `common/perception_card/CMakeLists.txt`
  (`add_test`, as in `common/media_transport/CMakeLists.txt`)

**Interfaces (produces):**
- `enum class Level { Mass, Countable, Articulated, Animate };`
- `enum class Role { Identity, Occupancy, Both };`
- `struct Channel { std::string sensor; Role role; bool modelled; };`
  - Sensors are graph node names: `zed`, `helios`, `bpearl`, `ricoh`.
- `struct TermSpec { std::string name; std::map<std::string, float> params; };`
- `struct SubtypeParams { rc::detect::DetectorEnvelope env; float q_return = 1.0f; std::vector<TermSpec> terms; };`
  - `q_return` is the per-beam LiDAR return probability.
- `struct GeometryPrior { float w, d, h, z0, z1; };`
- The card itself:
  ```
  struct Card {
    int version = 1;
    std::string concept, graph_type;
    Level level;
    std::vector<Channel> channels;
    GeometryPrior geometry;
    std::map<std::string, SubtypeParams> subtypes;   // key "" = default
    std::vector<std::string> instance_terms;
    std::vector<std::string> blind_spots;
    std::string calibration;                          // "prior" | "fitted" | "blind_spot_map"
    std::string calibration_date;
    std::string blind_spot_map_path;
  };
  ```
- `std::string to_json(const Card&)`
- `inline constexpr std::array kTermNames{"envelope","in_fov","central","range_conf","leaf_obliquity","resolvability","kappa","coverage","motion"};`
  - This is the closed list, defined here. Task 2 implements one function per name.
- `std::expected<Card, std::string> card_from_json(std::string_view)`
  - Rejects: a term name not in `kTermNames`; a missing level; an unknown role.
- `const SubtypeParams& params_for(const Card&, std::string_view subtype)`, which falls back to the `""` entry.
- **JSON facility:** reuse the one `common/media_transport/media_transport.cpp` uses for descriptors (its
  `from_chars`-based number reads). Add no new JSON library.

Steps:
- [ ] **Step 1: write `card_test.cpp`.**
  - A full card round-trips through `to_json`/`card_from_json`, every field equal.
  - An unknown term `"magic"` → error containing `"magic"`.
  - A missing `level` → error.
  - `params_for(c, "round")` returns the round entry; `params_for(c, "oval")` returns the default.
  - The whole test runs after `CHECK(std::setlocale(LC_ALL, "") != nullptr)`.
- [ ] **Step 2: run it and see it fail** (it doesn't compile yet).
- [ ] **Step 3: implement `card.h`.**
- [ ] **Step 4: build and run the test.** Expected: `ALL OK` under `LANG=es_ES.UTF-8`.
- [ ] **Step 5: commit** `perception_card: Card type + JSON (closed term names)`.

### Task 2: The closed term library, by MOVING the agents' private factors

**Files:**
- Create: `common/perception_card/terms.h`
- Modify:
  - `door_concept/src/door_existence.cpp:1260-1279`, so its factors call the moved functions;
  - `cabinet_concept/src/cabinet_existence.cpp:62-67,128`, for `absence_range_conf`.
- Test: `common/perception_card/tests/terms_test.cpp`

**Interfaces (produces):**
- **Consumes:** `kTermNames` from `card.h` (Task 1).
- One `inline float term_<name>(...)` per name (`term_coverage` = the visible fraction passed through;
  `term_motion` is added in Task 3). Each body is MOVED VERBATIM from the agent that owns it today,
  and that agent then calls it:
  - `term_envelope` wraps `rc::detect::p_detect` (both overloads);
  - `term_range_conf(range, ref_m, power)` is cabinet's `absence_range_conf`, `min(1, (ref/range)^power)`;
  - `term_leaf_obliquity`, `term_resolvability` and `term_kappa` come from door, with the signatures door uses
    today;
  - `term_in_fov` and `term_central` take the silhouette fractions the agents pass today.
- `float eval_term(const TermSpec&, const TermInputs&)` dispatches by name. `TermInputs` holds the intermediate
  quantities: `fill_max`, `fill_min`, `visible`, `in_fov`, `central`, `range`, `phi`, `obliquity`.

Steps:
- [ ] **Step 1: write the golden test.** Before moving anything, record each agent factor's output on 20 fixed
  input tuples per term, printed by a scratch harness that calls the CURRENT agent code. Paste them as
  constexpr tables into `terms_test.cpp`. The test asserts each `term_*` reproduces its table to 1e-6.
- [ ] **Step 2: run it and see it fail** (`terms.h` doesn't exist yet).
- [ ] **Step 3: move the functions into `terms.h`** and make door and cabinet call them. Leave no copies behind.
- [ ] **Step 4: run the test; it must pass.**
  - Build door_concept and cabinet_concept with `cbuild -j8`.
  - Run door's existing tests (`door_concept/tests/`) and the `door_replay` tool on a recorded tape
    (`door_concept/etc/door_inputs*.bin`, if any exists). Decisions must be byte-identical to before the move.
- [ ] **Step 5: commit** `perception_card: closed term library; door + cabinet factors moved, identical by construction`.

### Task 3: World and hypothesis types; the camera evaluator

**Files:**
- Create: `common/perception_card/evaluator.h`
- Test: `common/perception_card/tests/evaluator_camera_test.cpp`

**Interfaces:**
- **Consumes:** `Card` (Task 1); `term_*`/`eval_term` (Task 2); from `common/nbv/viewpoint_score.h`:
  - `Sensor`, `Target`, `Obstacle`, `WallGap`;
  - `wall_obstacles`, `predicted_fill_axes`, `predicted_fill_sigma`, `visible_fraction`, `expected_p_detect`.
- **Produces:**
  - `struct LidarRig { rc::media::SensorModel model; Eigen::Isometry3f mount; };`, mount in the room frame.
  - The world:
    ```
    struct World {
      std::vector<Eigen::Vector2f> polygon;   // empty = no walls known
      std::vector<rc::nbv::WallGap> gaps;
      std::vector<std::pair<std::string, rc::nbv::Obstacle>> obstacles;
      std::optional<rc::nbv::Sensor> zed;
      float zed_mount_z;
      std::optional<LidarRig> helios, bpearl;
    };
    ```
  - `struct Hypothesis { std::string concept, subtype; float x, y, yaw; std::optional<GeometryPrior> size; std::optional<float> phi; float seconds_unseen = 0; };`
  - `struct Viewpoint { Eigen::Vector2f xy; std::optional<float> yaw; };`
    - `nullopt` yaw = "best heading": facing the target.
  - `enum class Limit { None, OutOfView, BehindWall, Occluded, TooSmall, TooClose, EdgeOn, BetweenRings, BelowFan, AboveFan, DeadZone, NotModelled, Inside };`
  - `struct ChannelResult { std::string sensor; float p; Limit limit; std::string occluder; };`
  - `struct Result { float p_notice, p_recognise; std::vector<ChannelResult> channels; Limit limit; std::string occluder; std::vector<std::string> notes; };`
  - `ChannelResult evaluate_camera(const Card&, const World&, const Hypothesis&, const Viewpoint&)`
- **Per-level composition** (spec §2):
  - coverage = `visible_fraction`;
  - countable × the envelope, via `expected_p_detect` with `predicted_fill_sigma`;
  - articulated × the card's configuration terms at `phi`;
  - animate: Gauss–Hermite over position with σ² = σ0² + k·`seconds_unseen`, implemented as `term_motion`
    in `terms.h`. k is the parameter `k` of the card's `"motion"` term.
- **Occluder:** leave-one-out over `obstacles`. The occluder is the obstacle whose removal raises
  `visible_fraction` the most. Report it only if removing it raises visibility.

Steps:
- [ ] **Step 1: write the tests.** All use a 6×4 m room, the ZED intrinsics 1280×720, focal 448, mount 0.945 m,
  and the fridge geometry 0.7×0.7×1.8.
  - (a) Clear view at 3 m → `p > 0.5`, `limit == None`.
  - (b) A 1×1×1.2 m box halfway → `limit == Occluded`, `occluder == "box_1"`.
  - (c) Viewpoint 0.3 m from the fridge → `limit == TooClose`.
  - (d) Target in the next room behind a wall with a door gap: `half_w = 0` (φ = 0) gives `BehindWall` and lower
    p; `half_w = 0.45` gives higher p.
  - (e) Empty polygon → it answers, with a note `"walls unknown"` and no crash (Review Focus 1).
  - (f) Animate: `seconds_unseen` 0 vs 30 → p decreases.
- [ ] **Step 2: run them and see them fail.**
- [ ] **Step 3: implement `evaluate_camera`.**
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: commit** `perception_card: camera evaluator (coverage, envelope, configuration, animate) + occluder by leave-one-out`.

### Task 4: The LiDAR returns term

**Files:**
- Modify: `common/perception_card/evaluator.h`
- Test: `common/perception_card/tests/evaluator_lidar_test.cpp`

**Interfaces:**
- **Produces:**
  - `float expected_returns(const LidarRig&, const rc::nbv::Target&, const Eigen::Vector2f& from, std::span<const rc::nbv::Obstacle>)`
  - `ChannelResult evaluate_lidar(const Card&, const World&, const Hypothesis&, const Viewpoint&, std::string_view sensor)`
- **The algorithm** (spec §4b):
  - For each ring elevation (`ring_elev_deg`, or rings spread uniformly over `fov_start..fov_end`), transform the
    ring's cone through `mount` and find whether it crosses the target's z span at the target's horizontal
    range.
  - For each crossing ring, the number of columns = target angular width / (azimuth step), × `visible_fraction`
    along that bearing.
  - λ = the sum × `range_factor(r)`.
  - P = 1 − exp(−λ · `q_return`).
  - `range_factor(r) = Φ((r − range_min)/σ_r) · Φ((range_max − r)/σ_r)`, with σ_r = the model's
    `noise_range_sigma_floor_m`, or 0.02 if absent, stated in a note. This is my call: it's soft at both ends,
    and the far end never matters indoors.
  - **Limit:**
    - `BetweenRings` if no ring crosses the z span while the span lies inside the fan;
    - `BelowFan` / `AboveFan` if the span is outside the fan;
    - `DeadZone` if r < range_min;
    - `Occluded` / `BehindWall` as for the camera.
- `World.helios`/`bpearl` absent → `limit == NotModelled`, and the channel is left out of the combination (Review
  Focus 2).

Steps:
- [ ] **Step 1: write the tests.** Helios model: 32 rings −55..+15, az 0.4°, range 0.1–80 m, mounted inverted at
  1.075 m. Bpearl: 32 rings 0..90 (2.857°), range 0.1–30 m, inverted at 0.67 m.
  - (a) Fridge at 3 m → λ > 50, p > 0.99.
  - (b) A 2 cm-thick horizontal slab at a height between two Helios rings at 6 m → `BetweenRings`, p < 0.1.
  - (c) A 5 cm-tall object on the floor at 0.5 m, seen by Helios → `BelowFan`; by Bpearl → p > 0.9.
  - (d) Target at 0.05 m from the Bpearl → `DeadZone`.
  - (e) `World.helios = nullopt` → `NotModelled`.
  - (f) `q_return = 0.1` lowers p versus 1.
- [ ] **Step 2: run them and see them fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: commit** `perception_card: LiDAR returns term (Poisson, ring geometry through the mount)`.

### Task 5: Combining channels, `evaluate`, `where_to_stand`

**Files:**
- Modify: `common/perception_card/evaluator.h`
- Test: `common/perception_card/tests/evaluator_combine_test.cpp`

**Interfaces (produces):**
- `Result evaluate(const Card&, const World&, const Hypothesis&, const Viewpoint&)`
  - P(notice) = 1 − Π(1 − p_c) over modelled channels whose role is Occupancy or Both.
  - P(recognise) = the same over Identity or Both.
  - `limit` = the limit of the best identity channel.
  - If the viewpoint is inside an obstacle or outside the polygon, or the target is inside a wall: P = 0,
    `limit = Inside`, `occluder` = the obstacle (Review Focus 3).
- `struct Stand { Eigen::Vector2f xy; float yaw; Result r; };`
- `std::vector<Stand> where_to_stand(const Card&, const World&, const Hypothesis&, int k = 3)`
  - 36 bearings × 3 standoffs inside `rc::nbv::standoff_band`.
  - Drop stands that are not `stands_inside`-clear or not `is_reachable`.
  - Return the top k by p_recognise.

Steps:
- [ ] **Step 1: write the tests.**
  - (a) A table card with channels zed=Identity, helios=Occupancy: a low view where the LiDAR sees it but the ZED
    is occluded gives p_notice > 0.9 and p_recognise < 0.3.
  - (b) Viewpoint inside the box → `Inside`, p = 0, no NaN.
  - (c) `where_to_stand` for the occluded fridge returns a stand with p_recognise > 0.8 that is not behind the box.
  - (d) Timing: `where_to_stand` takes under 200 ms (measured with `steady_clock`, printed, asserted).
- [ ] **Step 2: run them and see them fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: commit** `perception_card: channel combination (notice vs recognise), where_to_stand`.

### Task 6: Publishing and reading cards (file now, attribute after the reinstall)

**Files:**
- Create: `common/perception_card/card_io.h`
- Test: `common/perception_card/tests/card_io_test.cpp`

**Interfaces (produces):**
- `bool write_card_file(const std::filesystem::path& agent_dir, const Card&)`
  - Writes `<agent_dir>/etc/perception_card.json` atomically (temp file + rename), imbued with
    `std::locale::classic()`.
- `std::optional<Card> read_card_file(const std::filesystem::path& agent_dir)`
- `std::vector<Card> read_all_cards(const std::filesystem::path& active_inference_root)`
  - Scans `*_concept/etc/perception_card.json` (the concept inventory).

Steps:
- [ ] **Step 1: write the tests.** Write then read in a temp dir; equal. A corrupt file → `nullopt`, logged.
  `read_all_cards` over a temp tree with two agents → 2.
- [ ] **Steps 2–4: red, implement, green.**
- [ ] **Step 5: commit.**

**Task 6b (after the user's reinstall):**
- `template<class G> void publish_card(G& graph, std::uint64_t agent_node_id, const Card&)`, written with
  `add_or_modify_attrib_local<perception_card_att>` plus `update_node`, on change only.
- `template<class G> std::optional<Card> read_card(G&, std::string_view concept)`, which prefers the attribute and
  falls back to the file.
- Test with a stub graph type, as `sensor_model_from_graph` does.
- Commit.

### Task 7: Cards for table, refrigerator, hood, bottle (one batch, same shape)

**Files, per agent X:**
- Create: `X/src/X_card.h` with `rc::card::Card make_card(const XConfig&)`.
- Modify: `X/src/specificworker.cpp`.
  - Call `write_card_file` (and `publish_card` once 6b lands) in `initialize()` after the config is loaded.
  - Call it again where the agent loads a fitted envelope or saves a blind-spot map.
- Test: `X/tests/X_card_test.cpp`.

What each card contains:
- **Level:** Countable.
- **Channels:** as the agent really uses them. table: zed Identity, helios Occupancy, ricoh not modelled.
  refrigerator and hood: zed Identity, helios Occupancy. bottle: zed Identity, helios Occupancy.
- **Envelope:** from `XModel.DetectMinFill/MaxFill/soft`.
- **Geometry:** from `common/concept_manifest/X.concept.toml`.
- **Subtypes:** table's `round` and `square`.
- **Instance terms:** table `p_clear` and blind_spot_map (`table_existence.cpp:~200-230`).
- **Blind spots:** short sentences from each agent's known failure modes.
- **Calibration:** `"prior"` unless a fitted envelope is loaded.

Steps:
- [ ] **Step 1: write the tests.** `make_card(default config)` gives the expected envelope values (e.g. table 0.10
  / 0.60 / `detect_soft`) and channel roles, and the round-trip JSON is equal.
- [ ] **Steps 2–4: red, implement, green.** Build all four agents with `cbuild -j8`, one at a time.
- [ ] **Step 5: commit** `cards: table, refrigerator, hood, bottle`.

### Task 8: Cards for door (articulated) and cabinet (as-is)

Both agents call the moved terms since Task 2. Same files pattern as Task 7.

- **door:** level Articulated.
  - Terms: envelope, `leaf_obliquity`, `resolvability`, `in_fov`, `central`, `kappa`.
  - Channels: zed Identity, helios Occupancy.
  - When door φ is `kUnmeasured`, the evaluator marginalises over the φ prior: uniform on [0, `Openable.PhiMaxRad`]
    (Review Focus 4).
- **cabinet:** level Countable (collective fields come with the aggregate-objects work).
  - Terms: `range_conf(ref, power)` from `existence_absence_range_ref_m` / `_power`, `in_fov`, `central`.
  - Channels: zed Identity, helios Occupancy.
- **Tests:**
  - `make_card` values.
  - An evaluator test with door φ unmeasured vs φ = π/2: they give different p, and unmeasured lies between the
    shut and wide values.
- **Commit:** `cards: door (articulated), cabinet (as run today)`.

### Task 9: chair moves onto the shared countable formula (intended behaviour change)

**Files:**
- Modify:
  - `chair_concept/src/chair_fitter.cpp:738`: replace `zed_detectability` (centre × range ramp) with
    `term_envelope × term_in_fov × term_central`, and add `rc::occlusion::walls_block` line of sight against the
    room polygon, as table does at `table_projection.cpp:293`;
  - `chair_concept/src/chair_existence.cpp:182,327`: call the new function.
- Retire the config keys `Existence.ZedEdgeOffset`, `ZedRangeFull`, `ZedRangeRef` and `ZedClearLosFloor`. Keep
  `ChairModel.DetectMinFill/MaxFill`, now used by existence too.
- Create the card: Countable, zed Identity, ricoh not modelled.
- Update `CONCEPT_AGENT_LIFECYCLE.md` §6: chair's "wall LoS" cell goes to ✅.

Steps:
- [ ] **Step 1: write the tests.**
  - A chair behind a wall (target in the next room, no gap) → P(detect) from the new function is about 0, where
    the old one was > 0.5. Use the agent's own test harness pattern in `chair_concept/tests/`.
  - A clear chair at 2 m → P within 0.15 of the old value. This keeps behaviour close where nothing should
    change; the bound is a regression guard for the test, not a model parameter.
- [ ] **Steps 2–4: red, implement, green.** Build chair_concept.
- [ ] **Step 5: commit** `chair: shared countable P(detect) + wall line of sight (lifecycle fix); card`.
- [ ] **Step 6: flag for a live check** in the final report. This is the only behaviour change.

### Task 10: residual's mass card

- **Create:** `residual_concept/src/residual_card.h`.
  - Level Mass, term `coverage` only.
  - Channels: helios Occupancy, bpearl Occupancy, zed Occupancy (depth boost).
  - No P(detect) exists in the agent, and none is added.
- **Write it** in `initialize()` and again after `refresh_beam_spacing()` (added on feat/sensor-intrinsics).
- **Test:** `make_card` values; the evaluator on a 1×1 m floor region at 2 m gives coverage > 0.9 with no obstacle.
- **Commit:** `cards: residual (mass)`.

### Task 11: Truth test, layer (b): predicted vs measured on the probe logs

**Files:**
- Create: `common/perception_card/tools/card_vs_probe.cpp`, a standalone executable that reads a `detect_probe`
  CSV and the agent's card.
- Modify: `common/detect_probe/detect_probe.h`, ONLY if a row lacks the fields needed to rebuild the hypothesis
  and viewpoint. If so, add them as new columns at the end, so old readers keep working.

What the tool reports:
- the bias (mean of predicted minus logged p);
- the coverage of the predicted σ (the fraction of |fill_meas − fill_pred| ≤ 2σ);
- a reliability table of predicted p against the observed detection rate, in 10 bins.

Verification:
- `setlocale(LC_ALL, "")` before parsing.
- Test on a synthetic CSV whose rows are generated from the evaluator itself: bias ≈ 0 and coverage ≈ 0.95.
- Run it on any real probe CSV found under `*_concept/etc/` and include the output in the report.

Commit: `perception_card: card_vs_probe report`.

### Task 12: chat_agent `can_see` (in the chat worktree)

- **Where:** in `active_inference.worktrees/vlm-missions`, branch `feat/vlm-missions`. First merge
  `feat/perception-cards` into it (a local merge of our own branches, never main).
- **Files:**
  - Create: `chat_agent/src/perception_world.h` to build `rc::card::World` from the graph on the main thread:
    - the room polygon, a proto polygon included;
    - door gaps at each door's `door_phi_rad`;
    - objects → named `Obstacle`s;
    - ZED intrinsics and mount;
    - `sensor_model_from_graph(G, "helios"/"bpearl")` with the mounts via inner eigen.
  - Modify: `chat_agent/src/chat_session.cpp`.
    - Add `tool_def("can_see", …)` with arguments `concept` (required), `where`, `from`, and `stand` (bool: "where
      should I stand").
    - Dispatch near the other read-only tools, at about :1003.
    - Add it to `allowed_in_wake` (`mission_plan.h:458`).
  - Test: `chat_agent/tests/test_can_see.cpp` exercises the answer formatter on fixed `Result`s.
- **What the tool says:**
  - It names the channel and the limit, with the occluder's name.
  - It gives P(notice) and P(recognise) separately when they differ by more than their own rounding.
  - With no card it says "I have no detector for that" (Review Focus 5).
  - Ricoh-only channels are never claimed.
- **Eval:** add `tests/check_llm_tools.py` cases.
  - Positive: "can you see the fridge from here?", "where should you stand to see the table?", "¿ves la silla
    desde la puerta?".
  - Negatives: "go to the fridge" → not can_see; "where is the fridge" → where_is.
  - Run it on the Orin with gemma4:26b; the old cases must stay ≥ 307/315.
- **Commit:** `chat_agent: can_see / where_to_stand on perception cards`.

## Verification (end to end)

- **Unit tests:** every `common/perception_card/tests/*` passes under `LANG=es_ES.UTF-8`, and the agents' existing
  tests still pass.
- **Byte-identical replay:** door replay gives identical decisions after Task 2.
- **Live, in Webots** (coordinate with the room session first):
  1. Start the fleet.
  2. Check that each agent's `etc/perception_card.json` exists, and the attribute too once Task 6b lands.
  3. In the chat, ask "can you see the fridge from here?", then "where should you stand to see it?", then drive
     there and confirm the fridge is detected.
  4. Watch chair births and removals for one tour (the Task 9 behaviour change).
