# Imagined Observations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** chat_agent can show how the robot imagines a chair. It samples chairs from the concept prior, then
renders each sample through the real ZED and LiDAR models, and the canvas shows the distribution together with the
P(detect) it implies.

**Architecture:**
- chair_concept exports a **generative block** into its perception card, taken from its own model and from a
  manifest that becomes the single source of the chair prior. It has three parts: a shape program of closed
  primitives, a distribution per state variable, and a placement prior.
- Pure Eigen headers in `common/perception_card/` do the rest: they sample instances, rasterise a ZED mask and depth
  image, and ray-cast the LiDAR rings.
- chat_agent's `imagine_view` writes the results to the canvas run dir, and `rcanvas.py` draws them.

**Tech Stack:** C++23, Eigen, DSR graph, Qt (only in chat_agent, for PNG writing), Python/matplotlib canvas.

**Spec:** `docs/superpowers/specs/2026-10-31-imagined-observations-design.md` (out of v1 by design: §4.4 experience refinement, `at_table` placement), approved by the user on 2026-10-31
("yes, write the plan"), including the §3 single-source decision and the §11b rulings.

## Global Constraints

- **Branches.**
  - Tasks 1–6 run in worktree `active_inference.worktrees/perception-cards`, branch `feat/perception-cards`.
  - Tasks 7–8 run in `active_inference.worktrees/vlm-missions`, branch `feat/vlm-missions`, after a local
    `git merge --no-ff feat/perception-cards`, the same pattern used for `can_see`.
- **The model that fits is the model that imagines.** No second chair definition may exist. The shape, the
  distributions and the bounds are exported from chair_concept's own model, config and manifest.
- **The manifest is the single source of the chair prior.** No config key and no struct default may carry chair
  size facts. An unreadable manifest refuses to start.
- **CLAUDE.md rules:**
  - C++23 with `and`/`or`/`not`;
  - `std::from_chars` only, and tests that parse run after `CHECK(std::setlocale(LC_ALL, "") != nullptr)`;
  - type-attributed DSR attributes only;
  - never `Qt::DirectConnection`;
  - no thresholds where a model term works (rejecting interpenetration is physical, spec §4.3);
  - `cbuild -j8`, one agent at a time, after checking `pgrep -a cc1plus`.
- **Code layout:**
  - `common/perception_card/` stays header-only, Eigen + std, with no DSR and no Qt.
  - Its tests are built -O2 and registered with `add_test`. ctest is broken here, so run the binaries directly.
- **Git:**
  - NEVER `git add -f`; never commit `build/` or `.superpowers/`; check `git status` before every commit.
  - Commits end with
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_019DrdF8AcGc5RMg89NxnGDz`.
- **Never** kill -9 or pkill -f. Never start the fleet or Webots.

## Review Focus

1. **A crowded world:** a small room full of obstacles where most draws interpenetrate. The sampler must return
   fewer samples together with an acceptance-rate note. It must never loop forever or return a sample inside a
   wall. Test in Task 3.
2. **No room polygon, or a proto-room:** sampling `free` still works, using the obstacles only plus a note, and
   `against_wall` reports "no walls known" and is skipped. Test in Task 3.
3. **The viewpoint inside the sampled chair:** the render returns empty and the evaluator returns Inside. No NaN,
   no crash. Test in Task 4.
4. **A LiDAR model absent from the world:** that panel says "not modelled" and is not shown as zero returns. Test
   in Task 5.
5. **The same seed:** the same seed and the same snapshot produce byte-identical `imagined.json`. Test in Task 7.

---

### Task 1: The manifest becomes the single source of the chair prior

**Files:**
- Modify:
  - `common/concept_manifest/chair.concept.toml`
  - `chair_concept/src/chair_config.cpp`
  - `chair_concept/src/chair_config.h`
  - `chair_concept/etc/config.toml`
  - `chair_concept/src/chair_model.h`
  - `chair_concept/src/chair_fitter.cpp` (`make_belief_params`, about lines 35-60)
  - `chair_concept/src/chair_card.h`
- Test: `chair_concept/tests/chair_prior_test.cpp`

**Interfaces (produces):**
- `struct ChairPrior { float seat_w, seat_d, seat_h, back_h, seat_thick, leg_half, seat_w_std, seat_d_std, seat_h_std, back_h_std; };` in
  `chair_concept/src/chair_prior.h`
- `std::expected<ChairPrior, std::string> load_chair_prior(const std::string& manifest_path)`, which reads the
  `[prior.template]` keys `seat_w_m`, `seat_d_m`, `seat_h_m`, `back_h_m`, `seat_thick_m`, `leg_half_m`, `seat_w_std_m`, `seat_d_std_m`, `seat_h_std_m`, `back_h_std_m` via the
  manifest's ConfigLoader, using from_chars.

**What changes:**
- The manifest `[prior.template]` gets the values the agent runs today:
  - `seat_w_m = 0.60`, `seat_d_m = 0.52`, `seat_h_m = 0.595`, `back_h_m = 0.655`, `seat_thick_m = 0.075`,
    `leg_half_m = 0.0375`, and one sigma per variable: `seat_w_std_m = 0.05`, `seat_d_std_m = 0.04`,
    `seat_h_std_m = 0.03`, `back_h_std_m = 0.06` (the shared 0.15 was never read by the code);
  - `from = "config"`, with a `why` saying these are the values chair ran as `Tracker.BirthSeat*` /
    `AI2PriorSizeStd` (never actually read) until 2026-10-31.
- `[model.geometry] z_top_m` and `extent_m` become 1.25. Correct the stale text: "fixed template, pose-only" stays
  true (the belief is pose-only), but the `zed_detectability` sentence goes and the 0.45 numbers are replaced.
- `chair_config` loads the ChairPrior at startup. On error it logs and refuses to start, via the same exit path as
  `provenance_ok` (`chair_config.cpp:28`).
- Remove `Tracker.BirthSeatW/D/H/BackH/SeatThick` and `ChairModel.AI2PriorSizeStd` from `etc/config.toml` and from
  the cfg struct. The birth template, `make_belief_params`, `chair_card.h` and every former reader now read
  `cfg_.prior`.
- `ChairState{}` member defaults for seat_w/seat_d/seat_h/back_h become 0. Add a comment that the initial state
  always comes from `ChairPrior`. Grep every `ChairState{}` construction and make it seed from the prior.

Steps:
- [ ] **Step 1: Write `chair_prior_test.cpp`.** It checks:
  - `load_chair_prior` on the real manifest gives {0.60, 0.52, 0.595, 0.655, 0.075, 0.15};
  - a manifest missing `seat_w_m` gives an error;
  - `make_card(cfg)`'s geometry equals the prior (w 0.60, d 0.52, z1 1.25);
  - the birth `ChairBeliefParams` template equals the prior.
- [ ] **Step 2: Run it and see it fail.**
- [ ] **Step 3: Implement the changes above.** Grep `tracker_birth_seat\|ai2_prior_size_std\|BirthSeat\|AI2PriorSizeStd`
  across chair_concept; it must find no matches afterwards.
- [ ] **Step 4: Verify.**
  - The test passes, and chair_concept builds.
  - chair's existing tests (`chair_detect_test`, `chair_card_test`) still pass, with the same numbers as before.
  - The pre-existing behaviour is identical: same values, new source.
- [ ] **Step 5: Commit** `chair: the concept manifest is the single source of the chair prior`.

### Task 2: The generative block in the card, with chair's export

**Files:**
- Modify: `common/perception_card/card.h`, `chair_concept/src/chair_card.h`, `chair_concept/src/chair_model.h`
  (make `sdf_point_at` usable from a test: public or a free function, with no behaviour change).
- Test: `common/perception_card/tests/card_test.cpp` (round trip), `chair_concept/tests/chair_shape_test.cpp`.

**Interfaces (produces), in card.h (rc::card):**
- `enum class Prim { Slab, Panel, LegSet, Box, Cylinder };`
- `struct Bind { std::string var; float a = 0, b = 0; };`. The value is `a·state[var] + b`; with `var` empty it is
  `b`.
- `struct Primitive { Prim type; std::map<std::string, Bind> dims; };`. dims keys come from a closed list per type:
  - Slab and Box: `w`, `d`, `h`, `x`, `y`, `z`;
  - Panel: `w`, `t`, `h`, `x`, `y`, `z`;
  - LegSet: `n`, `half`, `fw`, `fd`, `z0`, `z1`;
  - Cylinder: `r`, `h`, `x`, `y`, `z`.
  - All of them are in the object's local frame, with +y forward (as in ChairState).
- `struct VarDist { std::string var; float mean, sigma, lo, hi; bool log_normal = false; };`
- `struct PlacementPrior { std::string support; std::map<std::string, float> contexts; float wall_gap_mean, wall_gap_sigma; };`
  - support: `"floor"`.
  - contexts: weights keyed by `"free"` or `"against_wall"`.
- `struct Generative { std::vector<Primitive> shape; std::vector<VarDist> vars; PlacementPrior placement; std::string bounds_source; };`
- `Card` gains `std::optional<Generative> generative;`, written to and read from the flat JSON with the same dotted
  keys style. Unknown `Prim` names, dims keys or context names are rejected (a closed vocabulary).

**chair export (chair_card.h):**
- **Shape:** a seat Slab, a backrest Panel at the −y edge from z = seat_h, and a LegSet with n = 4,
  half = `ChairPrior::leg_half` (manifest, 0.0375), from 0 to seat_h − thick. One source: the manifest.
- **Vars:**
  - seat_w, seat_d, seat_h and back_h: mean from `ChairPrior`, sigma = the variable's own `*_std`;
  - lo/hi taken FROM `ChairModel::apply_constraints`'s bounds. Expose them as `static constexpr` in chair_model.h
    and use them in both places.
- **Placement:** support floor, contexts {free: 0.5, against_wall: 0.5}, wall gap mean 0.05 m and sigma 0.05 m.
  These go in the manifest as `[prior.placement]` with provenance `from = "nominal"`, read through
  `load_chair_prior`'s loader.

Steps:
- [ ] **Step 1: Write the tests.**
  - The card round trip includes a generative block.
  - An unknown prim is rejected.
  - `chair_shape_test`, the shape identity against `ChairModel::sdf_point_at`. Use 2000 random points in the
    chair's bounding box + 0.2 m, at the prior mean and at the 16 ±2σ corners clamped to the bounds:
    - wherever |sdf| > 0.01 and |primitive distance| > 0.01, the inside/outside sign agrees;
    - wherever |sdf| < 0.03, |primitive distance − sdf| < 0.01.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** When the identity test fails, fix the primitives, never the SDF (spec §11b.4).
- [ ] **Step 4: Verify.** All perception_card tests and chair tests pass, and chair_concept builds.
- [ ] **Step 5: Commit** `perception_card: generative block (closed primitives, var dists, placement); chair export + shape identity vs its SDF`.

### Task 3: The sampler

**Files:**
- Create: `common/perception_card/generative.h`
- Test: `common/perception_card/tests/generative_test.cpp`

**Interfaces (produces), in rc::card:**
- `struct Placed { std::vector<rc::nbv::Obstacle> boxes; };`. Every primitive becomes room-frame oriented boxes;
  a LegSet becomes n boxes and a Cylinder its bounding box.
- `struct Sample { std::map<std::string, float> state; float x, y, yaw; Placed parts; float weight = 1; };`
- `struct Context { std::optional<Eigen::Vector2f> at; std::optional<std::string> force_context; };`
- `struct SampleSet { std::vector<Sample> samples; int drawn = 0, rejected = 0; std::uint64_t seed; std::vector<std::string> notes; };`
- `Placed instantiate(const Generative&, const std::map<std::string, float>& state, float x, float y, float yaw)`
- `SampleSet sample_concept(const Card&, const World&, const Context&, int n, std::uint64_t seed, int max_draws = 50 * n)`

**Algorithm:**
- **Sizes:** each var drawn from a normal truncated to [lo, hi] (log-normal if flagged) with `std::mt19937_64(seed)`.
- **Context:** drawn by the weights unless it is forced or `at` is given.
- **`free`:**
  - x, y uniform in the polygon's bounding box, kept only if inside the polygon;
  - with no polygon, uniform in a 6×6 m square around the robot, plus a note;
  - yaw uniform.
- **`against_wall`:**
  - pick a wall edge with probability proportional to its length;
  - draw a position along it;
  - offset inward by seat_d/2 plus a gap drawn from the wall-gap normal truncated at 0;
  - yaw so the backrest faces the wall;
  - with no polygon the context is skipped, with a note.
- **`at`:** x, y fixed; yaw from the context, or uniform.
- **Reject** a draw if any part box at its z span overlaps a wall piece (`wall_obstacles(polygon, thickness)`) or a
  world obstacle in 2.5-D (footprint SAT plus z-span overlap).
- Stop at n accepted or at max_draws, and report drawn/rejected in the notes.

Steps:
- [ ] **Step 1: Write the tests.**
  - A seeded run gives the same samples twice.
  - In a 6×4 room, 200 samples have no part overlapping a wall or a 1×1 obstacle.
  - The size mean over 2000 draws is within 0.01 of the prior mean, and all sizes lie in [lo, hi].
  - `against_wall`: every backrest is within (gap + 3σ) of a wall and faces it (dot product > 0.9).
  - Review Focus 1: a 2×2 room with 4 obstacles returns fewer than n samples, `rejected > 0`, and a note. It
    terminates.
  - Review Focus 2: an empty polygon gives `free` samples plus a note, and `against_wall` is skipped with a note.
- [ ] **Steps 2–4: red, implement, green.**
- [ ] **Step 5: Commit** `perception_card: sample_concept (truncated priors, free/against_wall placement, physical rejection)`.

### Task 4: The ZED render, and its agreement with the evaluator

**Files:**
- Create: `common/perception_card/render.h`
- Modify: `common/perception_card/evaluator.h`. Add `float room_height = 2.5f;` and `bool room_height_known = false;`
  to `World`.
- Test: `common/perception_card/tests/render_zed_test.cpp`

**Interfaces (produces):**
- `struct ImaginedImage { int w, h; std::vector<std::uint8_t> visible, occluded; std::vector<float> depth; float fill_max, visible_frac; };`
  - `visible` and `occluded` are 0/1 per pixel.
- `ImaginedImage render_zed(const World&, const Viewpoint&, const Placed&, int out_w = 320, int out_h = 180)`

**Algorithm:**
- Use the pinhole from `World.zed`, scaled to out_w×out_h.
- Pose: the camera at the viewpoint with mount height `zed_mount_z`. The heading is `vp.yaw`, or the bearing to the
  placed parts' centre.
- For each pixel, cast a ray; the nearest hit among the parts' boxes, the world obstacles' boxes and the walls as
  vertical slabs [0, room_height] decides it.
  - visible = the nearest hit is a part;
  - occluded = a part is hit, but behind something else.
- `fill_max` = the max-axis bounding-box fraction of visible ∪ occluded, the agents' definition (projected extent
  regardless of occlusion, as in the agents' ROI).
- `visible_frac` = |visible| / |visible ∪ occluded|.
- With no room height, add a note "room height 2.5 m assumed" on the caller side.

Steps:
- [ ] **Step 1: Write the tests.**
  - A chair at the prior mean, 3 m in front, no obstacles: |fill_max − evaluator fill| ≤ 1/out_h, using
    `evaluate_camera` on the sample's bounding-box Hypothesis, and visible_frac ≥ 0.99.
  - A 1×1×1.2 box halfway: visible_frac drops, and |visible_frac − evaluator visible fraction| ≤ 0.05.
  - A wall between: visible is empty.
  - Review Focus 3: a viewpoint inside a part gives an empty image, `evaluate()` returns Inside, and there is no
    NaN anywhere.
- [ ] **Steps 2–4: red, implement, green.** Time one 320×180 render and print it.
- [ ] **Step 5: Commit** `perception_card: render_zed (rasterised mask/depth through the real pinhole + mount), agrees with evaluate`.

### Task 5: The LiDAR render, and its agreement with the returns model

**Files:**
- Modify: `common/perception_card/render.h`
- Test: `common/perception_card/tests/render_lidar_test.cpp`

**Interfaces (produces):**
- `struct ImaginedScan { std::vector<Eigen::Vector3f> on_target, on_other; std::vector<float> ring_elev_room_deg; float lambda0; bool modelled; };`
- `ImaginedScan render_lidar(const World&, const Viewpoint&, const LidarRig* rig, const Placed&, std::uint64_t seed)`

**Algorithm:**
- If `rig` is null or its model has no rings, return `modelled = false` (Review Focus 4).
- The rig in the room comes from `rig_in_room` (evaluator.h). Rings come from `ring_elev_deg`, or uniformly over
  the fov.
- Azimuth steps cover only the sector subtended by the parts' bounding box ± 5°, for cost.
- Ray-cast against the parts, the obstacles and the wall slabs:
  - a ray within [range_min, range_max] hitting a part is kept with probability `q_return` (seeded);
  - range noise is N(0, range_sigma_floor_m) along the ray.
- `on_target` = kept hits on parts; `on_other` = hits on the rest within the sector.
- `lambda0` = the count of rays that hit a part before the q and range factors. That is the evaluator's λ0
  computed on the compound shape.

Steps:
- [ ] **Step 1: Write the tests.** Use the Helios and Bpearl rigs as in `evaluator_lidar_test.cpp`.
  - A chair at 3 m: lambda0 is within 20% of `expected_returns` on the bounding box. The compound shape is
    expected to be lower, legs being thin, so assert `render ≤ box·1.05` and `render ≥ box·0.3`, and print both.
  - A thin chair whose seat lies between two Helios rings at 6 m has no on_target points on the seat's z band.
  - The same seed gives identical points.
  - `rig = nullptr` gives `modelled = false`.
- [ ] **Steps 2–4: red, implement, green.**
- [ ] **Step 5: Commit** `perception_card: render_lidar (ring ray-cast through the mount, q and range noise)`.

### Task 6: Monte-Carlo P(detect) for hypothetical objects

**Files:**
- Modify: `common/perception_card/generative.h`
- Test: `common/perception_card/tests/generative_test.cpp`

**Interfaces (produces):**
- `struct McResult { float p_notice_mean, p_notice_sd, p_recognise_mean, p_recognise_sd; Limit modal_limit; int n; };`
- `McResult evaluate_mc(const Card&, const World&, const Hypothesis& base, const Viewpoint&, int n, std::uint64_t seed)`
  - It samples the sizes, NOT the pose, keeping base x, y and yaw.
  - It calls `evaluate()` per sample with the sample's bounding box as the Hypothesis size.
  - It aggregates: mean, sd, and the most frequent limit.
- If the card has no generative block, it returns one `evaluate()` result with n = 1 and sd = 0.

Steps:
- [ ] **Step 1: Write the tests.**
  - n = 1 with all sigmas 0 equals `evaluate()` exactly.
  - n = 32 with seed fixed is reproducible.
  - A chair near the envelope's upper shoulder (1.3 m) shows sd > 0.
  - Timing: n = 32 takes under 200 ms (printed and asserted).
- [ ] **Steps 2–4: red, implement, green.**
- [ ] **Step 5: Commit** `perception_card: evaluate_mc over the concept prior`.

### Task 7: The canvas data path (in vlm-missions, after the merge)

**First:** in `active_inference.worktrees/vlm-missions`, run
`git merge --no-ff feat/perception-cards`. If the merge is not clean, stop and report.

**Files:**
- Create: `chat_agent/src/imagined_io.h`
  - `bool write_imagined(const std::filesystem::path& run_dir, const SampleSet&, const std::vector<ImaginedImage>&, const std::array<std::vector<ImaginedScan>,2>&, const McResult&, std::string_view caption)`
  - It writes `imagined.json` (classic locale) and `imagined_zed.png` with QImage.
    - The PNG's R channel = the per-pixel visible frequency over samples, scaled to 0–255; G = occluded frequency.
- Modify: `chat_agent/canvas/rcanvas.py`
  - add `imagined()`, the reader;
  - add `draw_imagined_zed(ax)`: the frequency image, the occluded parts hatched, and the frame border;
  - add `draw_imagined_lidar(ax_top, ax_side)`: the points per sample in faint colours, and the ring elevation
    lines in the side view.
- Test:
  - `chat_agent/tests/test_imagined_io.cpp`: write a 3-sample set, read the JSON back and check it; check that
    the PNG loads with QImage and has the expected size and frequencies.
  - Review Focus 5: two writes with the same inputs give byte-identical JSON.
  - A Python smoke test that `rcanvas.imagined()` parses the file and both draw functions run headless (the Agg
    backend), in the same style as `tests/test_canvas_stock.py`.

Steps:
- [ ] **Steps 1–4: red, implement, green.** Run the tests through `bash chat_agent/tests/run_tests.sh`.
- [ ] **Step 5: Commit** `chat_agent: imagined.json/png writer + rcanvas imagined views`.

### Task 8: The `imagine_view` tool, and MC `can_see` for hypotheticals

**Files:**
- Modify:
  - `chat_agent/src/chat_session.cpp`:
    - add `tool_def("imagine_view", …)`, with arguments `concept` (required), `where`, `from`,
      `sensor` (`zed`|`lidar`|`both`, default `both`), `n` (default 8) and `seed`;
    - dispatch it next to `can_see`, and add it to `allowed_in_wake` (`mission_plan.h`).
  - `chat_agent/src/can_see.h`: a hypothetical target, i.e. a concept with no graph instance and a named place, now
    uses `evaluate_mc` with n = 32, and the answer gives the mean ± sd.
  - `chat_agent/src/perception_world.h`: read `room_height_att` from the current room node into
    `World.room_height`, and set `room_height_known`.
  - `chat_agent/tests/check_llm_tools.py`: add cases.
    - Positive: "show me how you imagine a chair", "how would a chair look to your lidar?", "imagina una silla
      junto a la pared" (against_wall forced from the words "against/junto a la pared"; that is the model's job, via
      the `where` argument).
    - Negative: "can you see the chair?" goes to can_see; "draw the graph" goes to draw.
- **Flow:**
  1. Build the World as can_see does.
  2. `sample_concept(card, world, ctx, n, seed)`.
  3. Per sample: `render_zed`, then `render_lidar` for the helios and the bpearl.
  4. `evaluate_mc` at the requested viewpoint.
  5. `write_imagined` into the canvas run dir.
  6. Trigger a drawing with a fixed script that calls `draw_imagined_zed`/`draw_imagined_lidar`. Use the existing
     draw/serve path; read `canvas_runner.cpp`.
  7. Return the text.
- **Answer text:** the mean P ± sd and the modal limit, in words; the context used; the acceptance rate when below
  0.8; "room height assumed" when it was assumed; the seed.
- **Test:** `chat_agent/tests/test_imagine_view.cpp` checks the answer formatter on a fixed McResult and SampleSet,
  plus the dispatch on a stub World.

Steps:
- [ ] **Steps 1–4: red, implement, green.** `run_tests.sh` passes and chat_agent builds. Do NOT run the Orin eval;
  the controller runs it.
- [ ] **Step 5: Commit** `chat_agent: imagine_view (sample → render → canvas) + MC can_see for hypotheticals`.

## Verification (end to end)

- All `common/perception_card/tests/*` pass under `LANG=es_ES.UTF-8`, as do the chair tests and chat_agent's
  `run_tests.sh`.
- chair_concept and chat_agent build.
- **Orin:** `OLLAMA=http://192.168.1.34:11434 python3 tests/check_llm_tools.py gemma4:26b 3`. The new cases pass,
  and the old cases stay at or above 322/330.
- **Live** (with the person, coordinated with the room session): ask "show me how you imagine a chair by the wall".
  The canvas must show 8 chairs plus the ZED and LiDAR panels, and the caption's numbers must match the text.
