# Structure from Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make door existence, passage pairing, room identity and room shape decisions one free-energy comparison in nats, add room appearance from a retina atlas, test a free-energy exploration objective, and produce the runs for the ECMR 2027 paper (due 2027-02-26).

**Architecture:** A pure header library scores hypotheses and every decision is logged to CSV. Each agent keeps its own mechanism and owns its own flag: `door_concept` first, then `ltsm_agent`, then `retina` and `room_concept`. Flag off means today's behaviour, with the comparison only logging.

**Tech Stack:** C++23, Eigen, Qt, DSR/cortex, standalone g++ tests, Python 3 reports, Webots.

**Spec:** `docs/superpowers/specs/2026-11-05-structure-from-evidence-design.md`. Read it first; it carries the reasons.

## Global Constraints

- Base: branch `feat/structure-evidence` from local `main` (`b2b74947` or later, and it must contain `1a8b4814` crossing-born promotion and `48177601` door mask association, which is on `feat/room-recognition` until main is updated), worktree `active_inference.worktrees/structure-evidence`. Local `main` contains `aff02fde` plus 5 unpushed commits, including the re-aim fixes and the door passable plan.
- Do not push. Commit only in the worktree.
- CLAUDE.md applies: no new thresholds (flag any at its site with its justification); queued connections only; SIGTERM, never `kill -9`; type-attributed DSR access only; `and`/`or`/`not`; `std::from_chars` for reading; `imbue(std::locale::classic())` for writing; `setlocale(LC_ALL, "")` in any test that parses a data file.
- New DSR attributes need a `REGISTER_TYPE` line and Pablo's cortex reinstall. Never `sudo cp` the header. Ask him, then wait.
- Agents communicate only through the memories. No file written by one agent is read by another. Log files read by offline scripts are fine.
- Builds: `cbuild`, `make -C build -j8`. Check `pgrep -a cc1plus` first.
- Webots is shared: message the session named in `room`'s hand-off memory (`resume-11-05-after-relocaliser`) before and after using it.
- A run below nominal rates is a failed run. warm46-61's 46/48 must not fall. The crossing protocol and `exit` chain are not changed by this work.
- Spec values: door width 0.85 m, std 0.12 m; height 2.05 m, std 0.06 m; P(wall has a door) = 0.4; P(another door | wall has one) = 0.01; band `T` 0.20 m, `sigma` 0.10 m. N >= 10 per arm.
- Dates: leaf stall unresolved on 2026-12-15 goes to Pablo. Paper decision 2027-01-15.

## Review Focus

1. A detector that fires falsely on one wall panel from several viewpoints. Today's birth bar is `6.6 x ln(h/pc)`, about 15 nats at h = .5, pc = .05. The comparison's bar is near 5 nats, so phantoms could rise. Task 3 test `recurring_panel_false_positive`; E2 reports it, and a rise goes to Pablo.
2. Door seen in a provisional room, whose polygon is a frontier and not a wall: it must not be penalised against that frontier. Task 4 test `provisional_room_no_band`.
3. Room change while candidates are pending: bank entries are in the old room's frame and must not birth a door in the new one. Task 3 test `room_change_drops_pending`.
4. Retina stops or restarts while its atlas node hangs under the room: no stale node may be copied to memory as the room's atlas with a wrong encoder or frame. Task 12 test `stale_atlas_swept_at_start` and Task 13 test `encoder_mismatch_is_no_evidence`.
5. New-room complexity overwhelms the metric term, so `new` can never win and similar rooms merge. `kappa |V|` may reach tens of nats against log masses near 1. Task 8 test `new_room_still_wins_when_chain_says_outside`; E4's negative control checks it live.

---

## Stage 0: Base and prerequisites

### Task 1: Worktree, door plan coordination, rate report

**Files:**
- Create: `room_concept/estimation_scripts/scripts/rate_report.py`, `room_concept/estimation_scripts/scripts/test_rate_report.py`

**Interfaces:**
- Produces: `python3 rate_report.py <run_dir>` prints one row per agent (controller, room_concept, retina: median Hz, 5th percentile Hz, nominal, PASS/FAIL) and exits 1 on FAIL. Later runs call it.

- [ ] **Step 1:** Create the worktree and message the `room` session: this plan takes door queue item 3 (exact miss) and touches `door_core.cpp`, `door_existence.cpp`. Its door passable plan is committed (`a7ae1633`); ask which of its tasks are still to land, and rebase on them when they do. Verify: `git -C <worktree> merge-base --is-ancestor b2b74947 HEAD` succeeds.
- [ ] **Step 2:** Find how each agent logs its rate (`grep -n "Hz" crossing/warm63/logs/*.log`). Nominal values come from memory `rt-walk-copies-nodes-11-01`: controller about 19 Hz, retina 40, room_concept 20; `Policy.MinControlHz` = 10 is the controller's floor. Ask Pablo to confirm these as nominal. If an agent logs no rate, add a periodic line in that agent.
- [ ] **Step 3:** Write `test_rate_report.py` with canned logs: a passing run exits 0; a run with controller at 7 Hz exits 1; a missing log prints NOT MEASURED and exits 1. Run it, see it fail.
- [ ] **Step 4:** Implement, run `python3 -I test_rate_report.py`, then run `rate_report.py crossing/warm63`. Expected: PASS or a real finding to report.
- [ ] **Step 5:** Commit.

### Task 2: Cumulative memory and empty-LTSM tours

**Files:**
- Create: `room_concept/estimation_scripts/scripts/cumulative_tour.sh`, `room_concept/estimation_scripts/scripts/explore_tour.sh`

**Interfaces:**
- Produces: `cumulative_tour.sh <run> <prev_run>`: starts from the memory and door table saved by `<prev_run>`, not a fixture. `explore_tour.sh <run>`: empty memory, room-name counter and door table; explores, crossing the next uncrossed door until none is left or the timeout. Both write `legs.tsv`, `gt_pose.tsv` and call `rate_report.py`.

- [ ] **Step 1:** Agree with the `room` session who writes the exploration driver; its queue lists the same tour next (`door-work-queue-11-05`). Reuse its version if it exists.
- [ ] **Step 2:** Write the scripts from `warm_start_tour.sh` and `three_rooms_tour.sh`. Doors shut at start. No file edited by hand between sessions: the script checks the previous run's checksum against what it installs.
- [ ] **Step 3:** Run `explore_tour.sh expl01`. Expected: at least R1 and R2 built with no help. A tour failing on the door leaf goes to the door passable plan. Crossing-born rooms promote since `1a8b4814` (`explore_spike3`: room_2 promoted 630 s); the base must include it. Odometry stalls (one of ~159 s in spike 3) must be reported by `rate_report.py` and fail the run.
- [ ] **Step 4:** Check leaf side and hinge come from the robot: in `ltsm_memory.json`, `door_leaf_into_room_prob` and `door_leaf_hinge_plus_prob` are written and match `three-room-door-swings` with no fixture. If they are never learned, add a task before going on and tell Pablo.
- [ ] **Step 5:** Run `cumulative_tour.sh expl02 expl01`. Expected: known rooms recalled, no duplicates. Commit.

---

## Stage 1: The comparison and door birth

### Task 3: Library, decision log, door evidence store and birth test

**Files:**
- Create: `common/structure_evidence/structure_evidence.h`, `common/structure_evidence/structure_evidence_test.cpp`, `common/structure_evidence/decision_log.h`, `door_concept/src/door_birth_evidence.h`, `door_concept/tests/test_door_birth_evidence.cpp`, `door_concept/tests/test_door_birth_comparison.cpp`
- Modify: `door_concept/src/core/ricoh_door_birth.h`, `door_concept/src/core/door_core.cpp:380-392,530-540,600-620`, `door_concept/src/core/door_existence.cpp:826-833`, `door_concept/src/door_fit_state.h`, `door_concept/src/door_config.h`, `door_concept/src/door_config_fields.h`, `door_concept/etc/config.toml`, `door_concept/tests/CMakeLists.txt`

**Interfaces:**
- Produces, `rc::structure`:
  - `struct Term { double log_evidence, complexity, log_prior; };`
  - `struct Hypothesis { std::string name; Term t; };`
  - `double free_energy(const Term&)` returns `-log_evidence + complexity - log_prior`.
  - `struct Decision { int winner; std::vector<double> F; };`, with `winner == -1` on empty or an exact tie.
  - `Decision decide(std::span<const Hypothesis>)`
  - `double conjugate_kl(double m, double R, double mu0, double var0)`: KL of the Gaussian posterior after measuring `m` with variance `R`, against the prior. It tends to 0 as `R` grows, so an unconstrained parameter costs nothing.
  - `class DecisionLog { DecisionLog(path, agent); void write(t_ms, concept_, std::span<const Hypothesis>, const Decision&, std::string_view changed); };` Classic locale, one row per decision.
- Produces, `rc::door`:
  - `float mixture_llr(float q, float d)` returns `log1p(q * expm1(d))`.
  - `struct DoorPriors { float w = .85f, w_std = .12f, h = 2.05f, h_std = .06f, p_door = .4f, p_another = .01f; };`
  - `rc::structure::Term door_exists_term(float L, float wall_len_m, float sigma_s_m, std::optional<std::pair<float,float>> width_m_R, float band_log, float p_other_door_same_wall, const DoorPriors&)`
  - `rc::structure::Term door_absent_term(const DoorPriors&)`
- Config: `Existence.BirthByComparison`, default false; `door_concept` writes `etc/structure_decisions_door_concept.csv`.

Scoring: `log_evidence = L`. `complexity = conjugate_kl(width) + log(wall_len / (sigma_s * sqrt(2*pi*e)))`. `log_prior = log(p_door) + band_log + p_other * log(p_another)`, and absent is `log(1 - p_door)`. Height is unconstrained at birth, because the `size_w` note in `door_core.cpp` records that mask height is biased and its noise unmeasured. The fit scores it after birth (Task 4).

- [ ] **Step 1: Write the failing library test** in `structure_evidence_test.cpp` (picked up by `common/run_tests.sh`). `decide` returns min F; exact tie gives `winner == -1`; `conjugate_kl(m, 1e6, mu, var) < 1e-3`; `trees_missing_one`: a synthetic 3 x 4 grid where one tree's evidence is negative, so the hypothesis "11 trees" beats "12" and beats "10"; a CSV round trip under `setlocale(LC_ALL, "es_ES.UTF-8")` reads back with points. The test also opens `structure_evidence.h` and asserts it contains none of `room`, `door`, `wall`, `passage`.
- [ ] **Step 2:** Run `common/run_tests.sh`. Expected: `structure_evidence_test` BUILD FAILED.
- [ ] **Step 3:** Implement `structure_evidence.h` and `decision_log.h`. Header only, no Qt, no DSR.
- [ ] **Step 4:** Run `common/run_tests.sh`. Expected: all pass.
- [ ] **Step 5: Write the failing door math test** `test_door_birth_evidence.cpp`: `mixture_llr(0,d) == 0`; `mixture_llr(1,d) == d`; at q = .41, d = ln(.15/.95), the result is near -0.42, against -0.757 for linear; 0.80 m and 1.00 m widths with R = .01 cost under 1 nat; 3 m with R = .01 costs over 50; no width estimate costs 0; `p_other = 1` adds 4.6 nats against `p_other = 0`, and `p_other = .5` adds 2.3.
- [ ] **Step 6:** Build with the g++ line in the header of `test_door_core_existence.cpp` and see it fail. Implement `door_birth_evidence.h`. See it pass.
- [ ] **Step 7: Write the failing behaviour test** `test_door_birth_comparison.cpp`, using the `FakeWorld` scene from `test_door_core_tracker.cpp`, all with the flag on unless stated:
  - `flag_off_births_identical`: the same scene and frames give the same birth cycle as the flag-off core, on the ZED scene and on the parked Ricoh scene.
  - `parked_door_born_once`: one birth over 1500 frames.
  - `parked_phantom_not_born`: a stare with intermittent false hits and absence gives no birth.
  - `recurring_panel_false_positive`: the same false detection from 4 viewpoints with misses between. Assert it is still not born at `h`, `pc` from config, or record the count it takes. If it is born sooner than today, the test stays red and you stop and tell Pablo.
  - `q_zero_changes_nothing`: L is unchanged by an occluded or not-looked frame, apart from forgetting.
  - `good_views_need_fewer`: close views birth in fewer new observations than views at 7 m.
  - `both_sensors_one_store`: ZED and Ricoh hits at one place sum in one entry.
  - `seed_is_the_evidence`: `existence.logodds()` at the first existence cycle equals the accumulated `-dF`, not 1.0. A restored door still gets `restore_existence_prior`.
  - `min_sep_still_blocks`: a candidate within `birth_min_sep_m` of a live door is not born.
  - `room_change_drops_pending`: after a room change no entry from the old room births a door.
- [ ] **Step 8:** Run; see it fail.
- [ ] **Step 9: Implement behind the flag.**
  - Rename `RicohBirthBank` to `DoorBirthBank`; keep the file.
  - Add ZED looks: `Look::weight` is `rc::birth::evidence` for that slice, and the miss `q` is the same detectability, using the ZED field of view.
  - Add `Entry::width_m` and `Entry::width_R`, fused like `xy`, from the slice's horizontal extent when it is untruncated.
  - Replace `ricoh_bar` with `ready_at(xy, var, gate, score_fn)`, where `score_fn(entry)` runs `decide` on exists against absent.
  - With the flag on, `dv.birth_evidence` is 0 unless ready, then `tracker_birth_frames` through the existing hook. `InstanceTracker` is not edited.
  - Clear the store on room change.
  - Seed through `DoorFitState::birth_logodds` (`std::optional<float>`) at `door_existence.cpp:829`.
  - Write one `DecisionLog` row on each birth.
  - For now `band_log = 0`, `p_other = 0`, and `wall_len` is the nearest polygon edge: Task 4 replaces them.
- [ ] **Step 10:** Run all door tests in `door_concept/tests` plus `common/run_tests.sh`. Expected: PASS, with the "not before `birth_frames`" check in `test_door_core_tracker.cpp` unedited because that test runs with the flag off.
- [ ] **Step 11:** Build the agent. Measure `door_concept` cycle time with the flag on against off. Commit.

### Task 4: Wall band at birth, one door per wall, fitted size

**Files:**
- Create: `door_concept/src/door_wall_lines.h`, `door_concept/tests/test_door_wall_prior.cpp`
- Modify: `door_concept/src/door_birth_evidence.h`, `door_concept/src/core/door_core.cpp:765-776`, `door_concept/src/core/door_existence.cpp:867-878`, `door_concept/src/core/door_fit.h`, `door_concept/src/door_fit_state.h`

**Interfaces:**
- Consumes: `door_exists_term`, `rc::door::band_log_prior`, `wall_hypotheses` in `ricoh_door_birth.h`.
- Produces:
  - `struct WallLine { Eigen::Vector2f p, u; float length_m; };`
  - `std::vector<WallLine> wall_lines(const std::vector<Eigen::Vector2f>& poly, float sigma_m)`: collinear edges and niche-interrupted edges form one line.
  - `float p_same_wall(xy_a, xy_b, lines, sigma_m)`: continuous from the point-to-line distances, not a cut.
  - `RoomBand::sigma_m` becomes `hypot(cfg sigma, wall sigma)`, with wall sigma from the room's published layout sigma when the graph has one, 0 otherwise (flag the site).

- [ ] **Step 1: Write the failing test** `test_door_wall_prior.cpp`:
  - `r1_niche_wall_is_one_line`: R1's 32-vertex polygon, read from the fixture and pasted into the test: edges e7 and e31 are one line; e13 is another.
  - `collinear_doors_same_wall`: `p_same_wall` > .9; on perpendicular walls < .01; on a wall with no clear grouping it is between.
  - `off_wall_not_born`: a ZED candidate 1.5 m inside the room is not born.
  - `door_near_estimated_wall_born`: delta = -0.2 m with L at 8 nats is born.
  - `second_door_needs_more`: the second door on a wall needs about 4.6 nats more than the first, and strong evidence still births it.
  - `removal_frees_the_other`: after one is removed, the other's `L` rises by the same term it had paid.
  - `provisional_room_no_band`: room `is_proto`, a door anywhere near the frontier pays no band; after promotion the band applies, once.
  - `clutter_frame_keeps_door_on_wall`: the `explore_tour8` case: a born door on its wall gets one mask frame at 5.68 m whose points are all clutter (`DoorBelief::last_mask_assoc()` near 0). The aperture moves under 5 cm, the band term is unchanged, and the door is not removed. Repeat with the frame at full association and check it does move: the weight must not freeze honest updates.
  - `clutter_hit_birth_credit`: a ZED hit whose mask association is near 0 credits the birth store at most `last_mask_assoc()` times a clean hit's LLR **[call: weighting the hit by association is my proposal; Pablo to confirm]**.
  - `size_after_fit_once`: once `ai2_initialized`, the fitted width and height KL replaces the birth width term in `L`, with `conjugate_kl` on the fit's own variance. Repeated cycles do not accumulate it.
- [ ] **Step 2:** Run; see it fail.
- [ ] **Step 3: Implement.**
  - Band and one-per-wall go in `L` as replaced terms, like `room_prior_term` (`door_existence.cpp:872`), not per frame.
  - Skip the band when `fit_.room_is_proto()`, under the flag.
  - With the flag on, drop the `point_in_polygon` suppression for birth because the band replaces it, and keep it with the flag off.
  - Ricoh positions fused onto the wall trivially satisfy the band; state at the site that their wall agreement stays in the hit weight, so it is counted once.
- [ ] **Step 4:** Run the door tests. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 5: Exact miss in door_concept

**Files:**
- Modify: `door_concept/src/core/door_existence.cpp:1409`, `door_concept/src/door_look.h:248-253`, `door_concept/tests/test_door_absence_detectability.cpp`, `door_concept/tests/test_door_look.cpp`

**Interfaces:**
- Consumes: `mixture_llr`. Config: `Existence.ExactMiss`, default false. `common/existence_belief/existence_belief.h` is not edited.

- [ ] **Step 1: Add failing tests:** `exact_miss_half_charge_at_low_pd`: at p_detect = .1 the silhouette miss moves L by under 0.6 of the linear charge; `pd_zero_holds`; `pd_one_matches_linear`; the same three for `Evidence::observe`.
- [ ] **Step 2:** Run; see it fail.
- [ ] **Step 3: Implement.** `ev.log_odds_delta = mixture_llr(p_detect, d_full)` at `:1409`. In `Evidence::observe`, `v.sum += mixture_llr(pd, llr)`. Keep the pooling. Tell the `room` session that queue item 3 is done for doors.
- [ ] **Step 4:** Run the door tests. Expected: PASS with the flag off unchanged.
- [ ] **Step 5:** Commit.

### Task 6: Amend the contract and check Stage 1 live

**Files:**
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` (section 1 and "door_concept CREATE exceptions"), `door_concept/etc/config.toml`

- [ ] **Step 1:** Write the amendment: birth stays tracker-only; for doors the credit through `birth_evidence` is decided by the comparison when `Existence.BirthByComparison` is on. Add the conformance row.
- [ ] **Step 2:** Replay: `gold4` DOOR_0 scene from `test_door_core_tracker.cpp` is still born (test). Live: one `warm_start_tour.sh` forward and one backward with both flags on. Expected: all legs pass; `rate_report.py` PASS; `structure_decisions_door_concept.csv` non-empty.
- [ ] **Step 3:** Compare phantom births, flags on against off, over 4 tours per arm. A rise is reported to Pablo before Stage 2. Commit.

---

## Stage 2: Memory decisions

### Task 7: Passage pairing in memory

**Files:**
- Create: `ltsm_agent/src/passage_evidence.h`, `ltsm_agent/tools/passage_evidence_test.cpp`
- Modify: `ltsm_agent/src/room_eviction.cpp:825-880`, `ltsm_agent/src/room_swap.cpp:559`, `ltsm_agent/src/specificworker.cpp`, `ltsm_agent/etc/config.toml`, `ltsm_agent/CMakeLists.txt`

**Interfaces:**
- Consumes: `rc::structure::{Term, Hypothesis, decide, DecisionLog}`.
- Produces:
  - `struct FacePair { bool crossed; std::optional<float> seam_residual_m, seam_sigma_m, width_a_m, width_b_m, width_sigma_m; float appearance_loglr; };`
  - `std::array<rc::structure::Hypothesis, 2> passage_hypotheses(const FacePair&, double crossed_llr)`
  - `struct PairVerdict { bool same; rc::structure::Decision d; };`
- Config: `Passage.ByComparison`, default false; `Passage.CrossedLlr` = ln(P(reported cross | one door) / P(reported cross | not)). Take the value from `door_concept`'s cross-through-wall prior and flag it at its site.

- [ ] **Step 1: Audit.** List every reader of memory's `match` edges and `passage` nodes on the base commit, including `controller_door_passage.h`, `reloc_runtime.h`, `room_recall.cpp`, `room_chain.cpp`. For each, write one line: what it does if a crossed door has no memory passage. If any can stall a crossing, stop and bring it to Pablo (spec 3.2). `PassageLive` is not edited.
- [ ] **Step 2: Write the failing test** `ltsm_agent/tools/passage_evidence_test.cpp`, wired like `room_recognizer_test.cpp`: `crossed_consistent_same`; `phantom_crossing_separate`: crossed but seam residual over 2 m and widths disagree, the proto15 case; `unmeasured_terms_are_neutral`: a missing width changes nothing; `not_crossed_far_apart_separate`; `tie_keeps_none`.
- [ ] **Step 3:** Run; see it fail. Implement `passage_evidence.h`.
- [ ] **Step 4: Wire.** At the `mint_passage` and `match` block and the carry-over, with the flag on, match only if `same`. With the flag off, log what it would have decided. A "separate" verdict still copies both rooms, writes no `match`, and writes one `DecisionLog` row. On each later eviction of either room, rescore the pair: a new `same` adds the match, and a new `separate` removes it, with the old value in the row.
- [ ] **Step 5:** Run the `ltsm_agent` tests and one tour with the flag on. Expected: every graded crossing pairs; `trace_check.py` clean; the TLA+ model needs no change because crossing actions are unchanged. Commit.

### Task 8: Room identity and room shape logging

**Files:**
- Modify: `ltsm_agent/src/room_chain.cpp:302-337`, `ltsm_agent/src/room_recognizer.cpp:620,709`, `room_concept/src/layout_refiner.h`, `room_concept/src/room_boxes.cpp`, `ltsm_agent/etc/config.toml`, `room_concept/etc/config.toml`
- Test: `common/room_recognition/room_posterior_test.cpp`, `ltsm_agent/tools/room_recognizer_test.cpp`

**Interfaces:**
- Consumes: `DecisionLog`. `room_posterior.h`'s `posterior` and `commit_step` keep their signatures.
- Config: `Recognition.NewRoomComplexity`, default false. New room's `metric_log_new -= kappa * |V_provisional|`, with `kappa = ln(L / sigma)` as in `layout_refiner.h:17`.

- [ ] **Step 1: Write failing tests:** `new_room_still_wins_when_chain_says_outside`: chain outside all known rooms by 3 sigma, with |V| = 8; `recalled_room_wins_with_chain_inside`; `prior_sensitivity` existing test passes unedited. The first test measures whether `kappa |V|` alone makes "new" unwinnable, which is the spec's check on `metric_log_new`. If it fails after Step 2, new-room evidence must also score the provisional room's fit under each hypothesis (spec 3.3). Do that, or report identity as model comparison, and tell Pablo. If door and identity cannot both be true reduction, the title changes to *Instantiating Rooms and Doors from Evidence* (spec 3.3).
- [ ] **Step 2:** Implement. Write a `DecisionLog` row at each `commit_step` state change in ltsm, and at each adopted edit and promotion in `room_concept`. Units stay nats; no behaviour change in shape.
- [ ] **Step 3:** Run tests. Live: an `R1_LOOP=1` tour on the three-room fixture with the flag on. Expected: re-entry through DOOR_0 reaches room_1 as in warm13 and warm15, by posterior trace. Commit.

---

## Stage 3: Grading

### Task 9: grade_structure and E0

**Files:**
- Create: `room_concept/estimation_scripts/scripts/grade_structure.py`, `room_concept/estimation_scripts/scripts/test_grade_structure.py`

**Interfaces:**
- Consumes: `structure_decisions_*.csv`, `ltsm_memory.json`, `door_identities.csv`, `gt_pose.tsv`, door lines and room regions from `threerooms-piso.wbt`.
- Produces: `grade_structure.py <run_dir>` writes `structure_grade.csv`, with columns `false_doors, missed_doors, duplicate_rooms, false_merges, layout_error_m, seen_frac, iou, undo_count`, and exits 1 on missing inputs.

- [ ] **Step 1: Write failing tests** with canned runs, one per spec definition: a door alive beyond one room visit with no true door within its width counts false; a true door in a visited room with no instance at leaving counts missed; two remembered rooms over one true room count duplicate; a commit to a known room whose true room differs counts a false merge; IoU never appears without `seen_frac`; and a comma-locale check on the parser.
- [ ] **Step 2:** Implement. Run `python3 -I test_grade_structure.py`. Grade warm13, warm15, warm16, warm18, warm20, warm26 for the 7 SWAP events and record the results in the plan ledger: only warm13 and warm15 were graded before.
- [ ] **Step 3: E0.** Replay warm30, warm11, proto15 decisions where logs allow, using the door math and `passage_hypotheses`; where they do not, reproduce with a new run and say so. Write `crossing/E0/table.csv`. Commit.

---

## Stage 4: Aspect

### Task 10: Cross-room evaluation (the gate)

**Files:**
- Create: `retina/tools/place_xroom_eval.cpp`
- Modify: `common/place_memory/place_map.h:503-504`, `retina/src/place_stage.cpp`, `retina/tools/place_eval.cpp`, `retina/etc/config.toml`

**Interfaces:**
- Produces: `place_xroom_eval --sessions ... --heldout ... --doorway ... --out verdict.txt [--summary sectors|heading_free]`. `verdict.txt` holds X1 AUC with CI, X2/X3 confidently-wrong rates, fitted same-room and background similarity distributions, and decay `r`.

- [ ] **Step 1:** Re-verify memory `place-map-keyed-on-robot-not-camera-pose` against the code. If still present, fix it as that memory specifies, with a test that a 180 degree turn in place proposes the true robot position.
- [ ] **Step 2:** Write the tool with `--selftest` (`docs/superpowers/plans/2026-10-23-room-recognition.md` Task 0.2: synthetic rooms with known separation, AUC within .02).
- [ ] **Step 3:** Record tours of R1, R2, R3 with `log_queries` on, and doorway views. Add a look-alike room in a copy of the world. Measure retina's rate with the stage on.
- [ ] **Step 4:** Run both summaries. Commit the tool and `crossing/phase0/verdict_*.txt`. **FULL or WEAK: go on to Task 11; appearance enters at its measured weight. NONE: stop, drop Tasks 11 to 13 and the appearance arm, and ask Pablo.**

### Task 11: Room atlas module

**Files:**
- Create: `common/place_memory/room_atlas.h`, `common/place_memory/room_atlas_test.cpp`

**Interfaces:**
- Consumes: `rc::place::wrap_angle`, the fitted constants from `verdict.txt`.
- Produces, `rc::place`:
  - `struct AtlasPlace { Eigen::Vector2f xy; float n; std::vector<float> desc; };`
  - `class RoomAtlas { void add(xy, desc, weight); void merge(const RoomAtlas&, const Eigen::Affine2f& T_this_other); std::vector<float> pack() const; static std::optional<RoomAtlas> unpack(std::span<const float>, std::string_view encoder_id); };`
  - `struct AspectModel { float mu_same, var_same, mu_bg, var_bg, r_m; };`
  - `float aspect_llr(const RoomAtlas& known, std::span<const float> query, std::optional<Eigen::Vector2f> query_xy_in_known, const AspectModel&)`

The hypothesis relaxes to background with distance from stored places, as in `VIEW_ATLAS.md` section 3.2. Use `--summary`'s winner to fix the descriptor form. Place spacing is the fitted `r`.

- [ ] **Step 1: Write the failing test:** `far_from_places_llr_zero`; `no_pose_marginalises_over_places`; `same_room_positive_and_other_negative` on synthetic clusters; `merge_transforms_and_keeps_count`; `pack_round_trip`; `size_under_100kb_for_40m2`; `empty_atlas_zero`.
- [ ] **Step 2:** Run `common/run_tests.sh`; see it fail. Implement, header only. Run; see it pass. Commit.

### Task 12: Retina publishes the atlas under the room

**Files:**
- Modify: `retina/src/place_stage.h`, `retina/src/place_stage.cpp`, `retina/src/graph_publisher.cpp`, `retina/src/specificworker.cpp`, `retina/etc/config.toml`
- Cortex: `core/include/dsr/core/types/type_checking/dsr_attr_name.h`, new `place_atlas` (`std::reference_wrapper<const std::vector<float>>`), `place_atlas_encoder` (string), both non-streamed. Pablo reinstalls.

**Interfaces:**
- Produces: node type `metaconcept`, subtype `place_atlas`, name `place_atlas_<room_id>`, RT child of the current room, written only by retina, published when the atlas has absorbed a new place, never on the clock.

- [ ] **Step 1:** Ask Pablo for the reinstall. Wait.
- [ ] **Step 2: Add tests** in `retina/tests/test_place_atlas.cpp` with a fake graph: `publish_only_on_change`; `stale_atlas_swept_at_start`: a node from a previous run is removed at start, on `initialize` and again on the first `on_operating_enter`; `room_change_starts_new_atlas`; `no_room_no_publish`.
- [ ] **Step 3:** Implement, following CLAUDE.md's graph-off-main-thread rules: pin transforms to the capture timestamp. Keep the frame at capture, and have a re-fitted or adopted room restart or transform the atlas (read `room_concept`'s frame handling and test it).
- [ ] **Step 4:** Live tour. Expected: `rate_report.py` PASS, retina at nominal, and DDS update size logged. If publishing costs rate, reduce place count, then report; do not move the data outside the graph. Commit.

### Task 13: ltsm stores and compares aspects

**Files:**
- Modify: `ltsm_agent/src/room_eviction.cpp:273-330`, `ltsm_agent/src/room_recognizer.cpp:667-704`, `ltsm_agent/src/room_swap.cpp:343-374`, `ltsm_agent/etc/config.toml`
- Modify: `CONCEPT_AGENT_LIFECYCLE.md:257`, `retina/etc/config.toml`
- Test: `ltsm_agent/tools/room_recognizer_test.cpp`, `ltsm_agent/tests/`

**Interfaces:**
- Consumes: `rc::place::{RoomAtlas, aspect_llr, AspectModel}`, the `place_atlas` node.
- Produces: `AppearanceEvidence.loglr` per known room, filled in ltsm and discounted through `discounted_loglr` with `n_eff_per_s` from the measured autocorrelation. Config: `Recognition.Appearance`, default false; `AspectModel` values with their source in a comment.

- [ ] **Step 1: Add failing tests:** `atlas_reaches_memory_before_delete`: after an eviction the memory twin holds the atlas, and the live room is deleted only afterwards; `retina_gone_room_still_evicts`: no atlas, the transaction still completes, and that room reports appearance unavailable; `encoder_mismatch_is_no_evidence`: ratio 0, unavailable; `swap_commit_merges_atlas`; `swap_undo_keeps_atlas_with_P`; `incremental_only_new_places`; `place_evidence_absent_is_not_needed`.
- [ ] **Step 2:** Run; see it fail.
- [ ] **Step 3: Implement.** Remove the `place_evidence` reader. Keep ownership clear: ltsm never writes the live atlas node. Measure the comparison's cost per new place, and log it.
- [ ] **Step 4:** Run tests. Live: `R1_LOOP=1`. Expected: `appearance loglr` non-zero in the trace, re-entry correct. Plot evidence against distance from stored places; far from them it must be near 0, not negative. Commit.

---

## Stage 5: Where to look

### Task 14: Exploration objective

**Files:**
- Create: `room_concept/src/structural_gain.h`, `room_concept/tests/test_structural_gain.cpp`
- Modify: `room_concept/src/layout_explorer.cpp`, its caller in `room_concept`, `room_concept/etc/config.toml`, plus the bench that runs the agent's code

**Interfaces:**
- Consumes: `rc::layout::GainFn`, `plan_coverage_reachable`. `door_concept/src/door_look.h:67` `existence_eig`: extract its math to `common/existence_belief/` so no agent includes another agent's header. Door P(exists) comes from the door node's existing attribute through the graph.
- Produces: `float structural_gain_nats(const Eigen::Vector2f& standpoint, std::span<const DoorBelief> doors, const SensorModel&)`. `Explorer.Objective`: `"current"` (default), `"efe"`, `"frontier"`.

- [ ] **Step 1:** Read `layout_explorer`, `EpistemicPlanner` and `door_look`'s planner, and write in the plan ledger how they join, because the spec leaves this to the builder. Check what happened to the planner since the 594-layout null, which the spec records as unverified.
- [ ] **Step 2: Write the failing test:** `unresolved_door_worth_a_look`; `settled_door_worth_nothing`; `occluded_view_worth_nothing`; `coverage_converted_to_nats`: reveal area in the same unit; `route_length_divides`.
- [ ] **Step 3:** Implement, run, then run the bench. Output curves of coverage, `seen_frac`, structural errors and IoU against metres driven, for three arms, with N >= 10. No frame cap that saturates. Commit.

---

## Stage 6: Paper runs

### Task 15: Campaign

**Files:**
- Create: `room_concept/estimation_scripts/scripts/run_campaign.sh`, `docs/superpowers/plans/2026-11-05-structure-from-evidence-ledger.md`

**Interfaces:**
- Consumes: every flag and script above. All arms on one binary. Record the commit in each run.
- Produces: `crossing/paper/<experiment>/<arm>/<run>/` with `structure_grade.csv`, and `crossing/paper/summary.csv`.

- [ ] **Step 1: E1:** empty LTSM, then 3 cumulative sessions, N >= 10 chains. Map quality and errors per session.
- [ ] **Step 2: E2:** flags on against off, same tours, N >= 10. **Decision on 2027-01-15:** continue if E2 shows fewer structural errors with no more misses and E1 completes; otherwise bring the spec's section 8 fallbacks to Pablo.
- [ ] **Step 3: E4:** re-entry through DOOR_0, arms atlas only, metric only, both. Negative control with the look-alike room. Traces and undo count. Note that E4 stays in simulation.
- [ ] **Step 4: E6:** Task 14's arms, live on the empty-LTSM tour.
- [ ] **Step 5: E8:** give Noé the build, a run sheet and `grade_structure.py` for two rooms: build, stop, restart another day, reload, carry on; repeat with changed furniture or door state. Ground truth comes from hand-measured door lines, recorded in the run.
- [ ] **Step 6: Optional, unless they delay the above:** E3 stress (injected false detections, seen fraction); E5 goal validity; E7 baselines with `decide-ugent/AIMAPP` and `FreeformRobotics/Active_room_segmentation` public code, no reimplementation; E9 `field_concept` if ready; E10 language questions on the robot's own map with `language_agent` as it stands, graded against ground truth, the belief at that time, and predicted outcomes.
- [ ] **Step 7:** Before writing grounding claims, check what ConceptGraphs, HOV-SG and developmental robotics work actually predict and claim; the spec requires citations. Commit summaries and the ledger.

---

## Timeline

| By | Done |
|---|---|
| 2026-11-20 | Stage 0, Task 3 |
| 2026-12-05 | Stage 1, Task 9 |
| 2026-12-15 | Leaf stall resolved or with Pablo; Task 10 verdict; Stage 2 |
| 2026-12-31 | Stage 4, Stage 5 |
| 2027-01-15 | E1 and E2 done; paper decision |
| 2027-02-10 | E4, E6, E8 done; draft |
| 2027-02-26 | Submission |
