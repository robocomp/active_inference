# Step 1 — existence information replaces the 5-nat verification floor

Branch `feat/existence-info` (worktree `active_inference.worktrees/existence-info`), base `142b632d`. Nothing has been committed.

## Formula

For each candidate view, `rc::nbv::score()` now computes

```
geometric_gain = p_detect · max(0, raw_gain)                         (unchanged)
existence_gain = I(E;Y) = h(p·p_hit·p_detect + (1−p)·p_false) − p·h(p_hit·p_detect) − (1−p)·h(p_false)
                 with p = P(exists), only if Target::p_exists >= 0, else 0
expected_gain  = geometric_gain + existence_gain
```

P(detect) is applied only inside the hit likelihood, so the MI is not multiplied by p_detect a second time. The existence term is at most ln 2 nats, and it falls to 0 once existence is settled either way. `plan_faces` keeps its feasibility rule: an unusable face (the robot would stand inside an obstacle, or the pose is outside the room) scores 0 for every part. A usable face keeps its existence term even when its raw fit gain is 0.

How each agent's scalar gain is built from these parts:

- **Countable core** (`plan_countable`, used by table, hood, refrigerator, countable and chair): `gain = face_geometric_gains[best]·useful(raw) + face_existence_gains[best]`. The adequacy factor scales only the fit part, because Σ* is a demand on the geometric DOFs.
- **Cabinet**: `gain = min(geometric[best], adequacy_gap) + existence[best]`.
- **Bottle**: `gain = sc.expected_gain`, scored on its single far-side candidate.
- **Door**: unchanged. `Target.p_exists` stays at −1, and the door planner keeps adding its own MI as before.

When `p_exists` is unset, every one of these is bit-identical to the old formula (tested).

## Where p_exists / p_hit / p_false come from

| agent | p_exists | p_hit | p_false |
|---|---|---|---|
| table | `inst.existence.p_exists()` | `cfg_.existence_detection_prob` (TableModel.ExistenceDetectionProb, 0.85) | `cfg_.existence_clutter_prob` (0.05) |
| hood | same | `cfg_.existence_detection_prob` (HoodModel / manifest, 0.55 in config) | `cfg_.existence_clutter_prob` (0.05, the configured prior) |
| refrigerator | same | RefrigeratorModel.ExistenceDetectionProb (0.85) | RefrigeratorModel.ExistenceClutterProb (0.05) |
| cabinet | same | CabinetModel.ExistenceDetectionProb (0.85) | CabinetModel.ExistenceClutterProb (0.05) |
| countable (bed/nightstand/toilet) | same | Model.ExistenceDetectionProb (0.85) | Model.ExistenceClutterProb (0.05) |
| bottle | same | BottleModel.ExistenceDetectionProb (0.85) | BottleModel.ExistenceClutterProb (0.05) |
| chair | same | `rc::exist::SensorModel{}.detection_prob` (0.85) | `rc::exist::SensorModel{}.clutter_prob` (0.05) |
| door | not set (−1) | — | — |

Notes on two of the rows:

- **Hood:** its existence update replaces the clutter prior with a learned per-place false-alarm rate (`p_fa_->estimate(...)`). That lookup needs a viewing bearing, and the plan does not have one yet, so the planner uses the configured prior.
- **Chair:** it configures no hit or false-alarm rate, because its existence channel works in its own log-ratio units. It therefore uses the shared `SensorModel` defaults.

## Files changed (57 modified + 3 new)

- `common/existence_belief/existence_info.h` (NEW): `rc::exist::binary_entropy_nats` and `existence_mutual_info`, moved verbatim from the door header.
- `common/existence_belief/existence_info_test.cpp` (NEW): MI unit tests.
- `common/nbv/existence_score_test.cpp` (NEW): tests the existence term in `score`, `plan_faces` and `plan_countable`.
- `door_concept/src/door_existence_info.h`: now a thin forwarder (`using rc::exist::…` inside `rc::door`).
- `common/nbv/viewpoint_score.h`: adds `ExistenceView`; adds `Target::p_exists/p_hit/p_false` plus `prices_existence()` and `set_existence()`; adds `Score::geometric_gain/existence_gain`; adds `Plan::face_geometric_gains/face_existence_gains`; documents the change.
- `common/nbv/countable_planner.h`: the useful-fraction now scales only the geometric part, and the existence term is added afterwards.
- `common/epistemic_step/epistemic_step.h`: `verify_gain_floor` and its use are removed, and the header, `record` and StepInputs comments are updated.
- `{table,hood,refrigerator,cabinet,countable,chair,bottle}_concept/src/epistemic_planner.{h,cpp}`: `compute()` takes a trailing `const rc::nbv::ExistenceView& existence = {}`, which is copied onto the Target. Cabinet's gain formula is split as described above, and hood gets a comment noting its bid is no longer exactly 0.
- `{table,hood,refrigerator,cabinet,countable,chair,bottle}_concept/src/specificworker.cpp`: pass the `ExistenceView` and drop the `.verify_gain_floor = …` arguments (5 sites).
- `{table,hood,refrigerator,cabinet,countable}_concept/src/*_config.{h,cpp}`: the `existence_verify_gain` field and its `getf(...)` read are deleted.
- `{table,hood,refrigerator,cabinet,bed,nightstand,toilet}_concept/etc/config.toml`: the `ExistenceVerifyGain = 5.0` line is deleted, and the comment above it now says wants_verification is logged only.
- `*_instance.h` (7), `*_existence.cpp` (5), `{hood,refrigerator,countable}/specificworker_lifecycle.cpp`: comment-only. wants_verification is described as logged only and no longer pricing the offer, and the `dbg_nbv_gain_pub` comment no longer mentions the floor. The `wants_verification` / `verify_surprise` / `existence_verify_surprise` logic and their log columns are UNTOUCHED.

## Tests

- **`common/run_tests.sh`:** exit 1 with 3 failures, all pre-existing and unrelated. `mission/mission_ir_test.cpp`, `mission_progress_test.cpp` and `mission_text_test.cpp` fail to build with `fatal error: QByteArray/QJsonArray: No existe el archivo`, because the script passes no Qt include flags. The same compile fails identically on the main checkout's copy, and I did not touch `common/mission`. All other binaries pass, including `countable_planner_test: PASS`, `z_occlusion: all checks passed`, and the two new tests.
- **`existence_info_test`:** all passed.
  - MI(0,·,·)=0 and MI(1,·,·)=0.
  - 0 ≤ MI ≤ min(H(p), ln 2) over a 21³ grid; the grid maximum is 0.693147.
  - MI(0.5, p_hit = p_false) = 0.
  - MI(0.5, p_hit, 0.05) is strictly increasing in p_hit.
  - **MI(0.5, 0.85, 0.05) = 0.377527 nats.**
- **`existence_score_test`:** all passed.
  - When unset, `expected_gain == p_detect·max(0,raw)` exactly.
  - At p_exists=0.5 the term added is exactly MI(0.5, p_hit·p_detect, p_false).
  - At p_exists=1 the term is 0.
  - In `plan_faces`, every face satisfies face_gains = geometric + existence, and an unreachable face carries 0.
  - In `plan_countable`, the unset case is unchanged; useful=0 gives gain = existence term (0.3168 nats at p_detect 0.919); useful=1 gives geometric + existence.
- **door_concept tests**, built per their header commands with only the `-o` path moved to the scratchpad:
  - `test_door_epistemic_belief`: 6/6.
  - `test_door_look`: ALL PASS.
  - `test_door_look_vs_approach`: 19/19.
  - `test_door_ricoh_look`: ALL PASS.
  - `test_door_hypothesis_look`: ALL PASS.
  - `test_door_core_existence`: **4 FAILURE(S), pre-existing.** They are at lines 1031 (`worst < 0.05f`), 1275, 1276 and 1313 (`candidate_is_open` / `candidate_question_p`). The same test built from `git archive HEAD` (the unmodified base) gives the same 4 failures, and the two full logs are byte-identical (`diff` empty).
- **`common/affordance_manager/tests` (affordance_protocol_test):** cmake + make -j8 into the scratchpad, then run: "affordance protocol tests passed", exit 0.

## Builds

Each agent was built fresh (`cmake -B build -S . && make -C build -j8`), one at a time, with a check for other cc1plus processes before each. All eight succeeded:

| agent | time |
|---|---|
| table | 75 s |
| hood | 70 s |
| refrigerator | 64 s |
| cabinet | 81 s |
| countable | 86 s |
| chair | 60 s |
| bottle | 78 s |
| door | 126 s |

No new warnings. The warnings that remain are pre-existing and none is on a line I changed: table `table_existence.cpp:265` unused `e_conf`, chair `specificworker.cpp:701` unused `dbg`, bottle `-Wreorder` in `bottle_evaluator`, and door `existence_update.cpp:39`, `specificworker_anchor.cpp:77/1026`.

The bed, nightstand and toilet links `bin/<cls>_concept -> ../../countable_concept/bin/countable_concept` resolve to the freshly built binary (checked with `readlink -f` and `-x`).

## Startup / config consistency

- All five config loaders read the removed key through `getf(key, default, …)`, which has a default and is not required. The config registry (`common/config_report`) only REPORTS unread file keys and never rejects them; the only throw is `missing required config key`, and that applies to required reads.
- The TOML line and the code read were removed together, so there is no unread or missing key either way.
- No other file in the tree references `ExistenceVerifyGain` / `existence_verify_gain` except `countable_concept/KEYMAP.md`, a historical rename map that I left as is.

## Not done / open concerns

- `controller`, `room_concept` and `human_concept` also include `viewpoint_score.h`. The change there is additive (new defaulted members, `expected_gain` unchanged when unset), but those components were NOT rebuilt.
- **Hood's bid used to be exactly 0** (useful = 0, no σ* consumer), apart from the floor. It is now the existence MI, up to about 0.38 nats at p=0.5 and smaller at hood's 0.55 hit rate, so hoods will bid small amounts continuously while their existence is open. This is intended by "every object class", but the behaviour is new.
- Settled objects (p ≈ 1) gain essentially nothing from the existence term, so their bids are unchanged. A "might be gone" object now bids at most ln 2 nats instead of 5, and it is also subject to the cooldown.
- **Not run live:** nothing was started (as instructed).
