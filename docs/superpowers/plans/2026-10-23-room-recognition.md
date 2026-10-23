# Room Recognition on Entry Through a Never-Crossed Door — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A robot that enters a known room through a door it has never crossed (DOOR_2, R2 -> R3) enters it as new, recognises it from metric (seam chain) AND appearance (per-room DINO maps) evidence in one joint posterior, downloads R3's ltsm instance in place of the provisional room, relocalises in R3's frame, and leaves no duplicate room.

**Architecture:** Phase 0 measures whether DINO separates the rooms BEFORE anything is built. Retina archives one DINO map per room and serves read-only appearance matches against candidate rooms. ltsm owns the posterior over (room k, pose in k) + new: metric evidence from the path-taken seam chain against each known room's own chain, combined with the DINO pose mixture by product-of-experts on the pose. A staged commit swaps through the existing `recall_<room>` / promotion path; undo is the same path pointed at the provisional room. room_concept relocalises with the posterior as the mixture-search prior.

**Tech Stack:** C++23, Eigen, DSR (cortex), DINOv2 ONNX (retina PlaceEncoder), standalone g++ tests under `common/*/ *_test.cpp` (run by `common/run_tests.sh`), ltsm `tools/*_test.cpp` CMake targets (EXCLUDE_FROM_ALL), Webots `threerooms-piso.wbt` gold tour. Builds `cbuild -j8` after `pgrep -a cc1plus`.

**Spec:** `/home/pbustos/robocomp/components/active_inference/docs/superpowers/specs/2026-10-22-room-recognition-design.md` (approved; "Decisions after the builder check 2026-10-23" are binding). Executors read it first.

## Global Constraints

- CLAUDE.md: NO thresholds (gates/clamps/floors/cutoffs); use continuous covariance/likelihood terms. Every unavoidable decision boundary is listed in "Flagged decision points" below and carries a comment `// FLAGGED:` at the code site.
- DSR attributes are type-attributed only (`add_or_modify_attrib_local<foo_att>`), never `runtime_checked_*`. New attributes: `REGISTER_TYPE` in cortex `core/include/dsr/core/types/type_checking/dsr_attr_name.h`; **the user reinstalls cortex** (Task 3, explicit checkpoint). Code using a new `_att` does not compile before that.
- No `Qt::DirectConnection`; this plan connects no DSR signals (ltsm polls in `compute()`).
- Numbers from files: `std::from_chars` only; files written with `imbue(std::locale::classic())`; any test that parses a file calls `std::setlocale(LC_ALL, "")` first.
- `[Owns]` matches `[Agent].id`; each node/attribute has ONE writer (CONCEPT_AGENT_LIFECYCLE.md). Agents stop with SIGTERM/SIGINT, never `kill -9`.
- Media-plane/zero-copy rules untouched; cv::Mat deep-copied at thread boundaries (retina matcher runs on the ricoh PerceptionWorker thread).
- Rooms keep their own frames; room_1's frame is only the reference of the metric query. No change to the estimation branch beyond the swap. No recognition through a door ltsm already holds in a passage (step 1 unchanged, no audit).
- One implementer per component at a time. Commit per task, by explicit pathspec. Never push.
- C++: `and/or/not`, C++23 algorithms.

## Flagged decision points (the only places a boundary exists; each needs a justification comment)

1. **Commit = Bayes MAP under symmetric 0-1 loss** (user: both errors disastrous). Swap executes when argmax over {k, new} is a known room k; undo when argmax returns to `new` under evidence gathered AFTER the commit (Task 6). No mass threshold; the "margin" is the posterior itself. Justified: symmetric loss makes MAP the Bayes rule.
2. **Prior over hypotheses**: implemented as uniform over `{archived rooms except the live one} ∪ {new}`. Flagged because it is a modelling choice; the evidence dominates (Task 1 test shows prior sensitivity).
3. **Undo window ends when the robot leaves the recognised room** (the eviction fence), not on a timer or a confidence level (Task 7). No finalisation threshold exists.

## Review Focus

1. **Provisional room has no evidence yet** (just crossed the door; zero keyframes, empty chain): posterior must equal the prior, never NaN, never commit (Task 1, Task 6).
2. **A known room has no DINO map** (visited before PlaceMemory was on, or map file missing/corrupt): its appearance likelihood is the "no information" term, the metric term still decides; no exception (Tasks 2, 4).
3. **Two near-identical rooms (symmetry)**: posterior splits, no commit, robot keeps going as new (Task 1 test, Task 6).
4. **Robot turns around in the doorway during the staged commit** (re-crosses DOOR_2 backwards): swap/undo state machine must not leave a duplicate room or an orphan `recall_*` (Task 7).
5. **ltsm restart with a staged twin / stale recall on disk**: startup sweep removes it; no phantom "staged" room is ever a candidate (Tasks 6, 7).
6. **Locale**: map and evidence files read under `es_ES.UTF-8` (Tasks 0.2, 2).

---

## File Structure

| File | Component | Responsibility |
|---|---|---|
| `retina/tools/place_xroom_eval.cpp` (new) | retina | Phase 0 cross-room discrimination, pre-registered verdict |
| `retina/src/place_stage.{h,cpp}` (mod) | retina | `room_name` column in query log; per-room map files `etc/place_maps/room_<room_id>.bin` |
| `common/place_memory/place_registry.h` + `place_registry_test.cpp` (new) | common | read-only registry of per-room maps, load/match against candidates |
| `common/room_recognition/room_posterior.h` + `room_posterior_test.cpp` (new) | common | pure math: joint (k, pose) + new posterior, appearance LLR model, staged-commit rule |
| `retina/src/recognition_matcher.{h,cpp}` (new) | retina | per-frame matching against candidate maps; publishes `place_evidence` node |
| `ltsm_agent/src/room_chain.{h,cpp}` (new) | ltsm | path-taken seam chain, each known room's chain, polygon in reference frame, metric evidence |
| `ltsm_agent/src/room_recognizer.{h,cpp}` (new) | ltsm | posterior step, staged commit state machine, `recognition` node |
| `ltsm_agent/src/room_swap.{h,cpp}` (new) | ltsm | swap / undo through `recall_*` and promotion; twin dedupe; passage carry-over |
| `ltsm_agent/src/{room_recall,room_eviction}.{h,cpp}`, `specificworker.cpp` (mod) | ltsm | wiring, `Candidate::discard`, staged-twin exclusion |
| `ltsm_agent/tools/{room_chain_test,room_swap_test}.cpp` (new) | ltsm | graph-level tests |
| `room_concept/src/room_scene_graph*.{h,cpp}`, `room_concept.{h,cpp}` (mod) | room_concept | late recall adoption, relocalisation prior |
| `door_concept/src/specificworker.cpp` (mod) | door_concept | provisional-room identity cleanup |
| `room_concept/estimation_scripts/scripts/recognition_tour.sh`, `recognition_report.py`, `phase0_capture.sh` (new) | harness | capture + gold tour + criteria report |

DSR attributes added (Task 3 registers them all, one reinstall):

| attr | C++ type | on node | writer |
|---|---|---|---|
| `place_map_file` | `std::reference_wrapper<const std::string>` | memory room twin | ltsm |
| `recog_room_ids` | `std::reference_wrapper<const std::vector<uint64_t>>` | ltsm `recognition` node (candidates) | ltsm |
| `recog_loglr` | `std::reference_wrapper<const std::vector<float>>` | retina `place_evidence` node: per-candidate cumulative log-LR vs "none", same order as `recog_room_ids` | retina |
| `recog_mix` | `std::reference_wrapper<const std::vector<float>>` | `place_evidence`: per candidate, components packed as 10 floats `[w,x,y,th,Sxx,Sxy,Sxth,Syy,Syth,Sthth]` in k's frame | retina |
| `recog_mix_n` | `std::reference_wrapper<const std::vector<uint64_t>>` | `place_evidence`: component count per candidate | retina |
| `recog_posterior` | `std::reference_wrapper<const std::vector<float>>` | ltsm `recognition` node: P over `recog_room_ids` then P(new) last | ltsm |
| `recall_pose_prior` | `std::reference_wrapper<const std::vector<float>>` | `recall_<room>` node: same 10-float packing, mixture of robot pose in the recalled room's frame | ltsm |
| `recall_late` | `bool` | `recall_<room>` node: this recall targets the CURRENT provisional room | ltsm |

---

## Phase 0 — measure before building

### Task 0.1: Capture support and recording protocol (retina)

**Files:**
- Modify: `retina/src/place_stage.cpp` (`open_logs`, `log_query`, header ~L164), `place_stage.h` (`PlaceStageConfig::log_room_name`).
- Create: `room_concept/estimation_scripts/scripts/phase0_capture.sh`.

**Interfaces:** Produces query-log column `room_name` (live room node name at the stamp, from `rc::room::current_room_node`), appended as the LAST column so `place_eval` keeps parsing; descriptors still in `place_query_desc.f32`.

**What to record (the protocol, written into the script header):** with `[PlaceMemory] enabled=true, log_queries=true, build_map=false`, three sessions per room, each in its own `log_dir` (`etc/place_log/<session>/`): (S1) a full explore of R1, R2, R3 via the existing `explore` mission (door closed, no crossing); (S2) a second, independent explore of each room, started from a different pose after a restart (cross-session); (S3) the doorway views: for each room, 60 s parked within 1.5 m inside every door, rotating (views that include the neighbour room). The fleet is the loop_tour fleet. ~2 Hz x 10 min x 3 rooms x 3 sessions.

- [ ] **Step 1: Write the failing test** — extend `common/place_memory/place_map_test.cpp`? No: this is a log format. Add `retina/tools/place_log_format_test.cpp` asserting a header line ending in `,room_name` parses with `from_chars` columns unchanged (run under `setlocale(LC_ALL,"")`). Run: `g++ -std=c++23 -I/usr/include/eigen3 retina/tools/place_log_format_test.cpp -o /tmp/plft && /tmp/plft` — Expected: FAIL (file/column absent).
- [ ] **Step 2: Implement** the column and `log_room_name` (default true); the script (`phase0_capture.sh <session> <room>`) starts the fleet with the flags above, runs the mission, SIGTERMs, copies `etc/place_log` to `runs/phase0/<session>/`.
- [ ] **Step 3:** test passes; `cbuild -j8` retina builds. Expected: PASS, `0 errors`.
- [ ] **Step 4: Commit** `git add retina/src/place_stage.h retina/src/place_stage.cpp retina/tools/place_log_format_test.cpp room_concept/estimation_scripts/scripts/phase0_capture.sh && git commit -m "retina: room_name column in the place query log; phase0 capture script"`.

### Task 0.2: Cross-room discrimination tool and pre-registered verdict (retina)

**Files:**
- Create: `retina/tools/place_xroom_eval.cpp` (standalone, Eigen only, includes `../../common/place_memory/place_map.h`; build line in the file header like `place_eval.cpp`).
- Test: `retina/tools/place_xroom_eval.cpp --selftest` (synthetic descriptors, below).

**Interfaces:** `place_xroom_eval --sessions runs/phase0/S1_R1,... --heldout runs/phase0/S2_* --doorway runs/phase0/S3_* --out runs/phase0/verdict.txt`. Per room k builds a `PlaceMap` from S1 (same insertion rule as the live stage), then queries every S2 and S3 frame against every map k via `PlaceMap::match`.

**Pre-registered pass/fail block (written in the file header BEFORE the first run, same style as `place_eval`):**
- X1 own-vs-other separation: AUC of best-match similarity, own room vs any other room, S2 queries: reported with a bootstrap CI over 30 s blocks (frames within a block are not independent).
- X2 top-1 room accuracy on S2, and the **confidently-wrong rate** (posterior-free: top-1 wrong AND `margin` above the median own-room margin) <= 2 %.
- X3 doorway views (S3): X1/X2 recomputed on frames within 1.5 m of a door only — the case recognition actually faces.
- X4 "none match" density: leave-one-ROOM-out — maps of two rooms, queries from the third (the genuinely new room): the empirical distribution of best similarity gives p(s | new); also p(s | own) from S2. Fitted as two parametric densities by NLL (form chosen by the tool: Beta on the similarity, parameters printed). Output is the per-frame log-LR function `llr(s, margin)` coefficients.
- X5 frame correlation: autocorrelation of per-frame LLR along each session; the effective independent frames per second `n_eff_per_s` (integrated autocorrelation time). Output feeds the temporal discount in Task 1.
- X6 pose: for S2 queries, own-room top-1 position error vs the localiser pose (relative, as `place_eval` does) — gives the usable pose mixture width (reuses `MixtureParams::rho`).

**Verdict (what the numbers decide; recorded in `verdict.txt` and in the plan ledger):**
- **FULL**: X1 AUC lower CI >= 0.95 and X2/X3 confidently-wrong <= 2 %: appearance enters with the fitted llr; Tasks 2 and 4 proceed as written.
- **WEAK**: AUC between 0.7 and 0.95 or doorway views fail: appearance still enters, with the fitted (small) llr — no switch, its weight IS the measured LLR; the design leans on metric and the report says so (spec). Tasks 2 and 4 proceed.
- **NONE**: AUC lower CI < 0.7: Tasks 2 and 4 are DROPPED from this plan; Task 1 runs with the appearance term absent (llr = 0, pose from metric alone) and the spec's two-modality claim is reported as not supported by DINO. The user is asked before continuing.

- [ ] **Step 1: Write the failing selftest** `--selftest`: generate 3 synthetic rooms as Gaussian descriptor clusters with a known separation d' ; assert AUC within 0.02 of the analytic value, the leave-one-room-out density recovers the planted Beta parameters within 10 %, and a deliberately duplicated room (identical clusters) yields AUC 0.5 +- 0.05 and verdict NONE. Run `./place_xroom_eval --selftest` — Expected: FAIL (not implemented).
- [ ] **Step 2: Implement** (reusing `rc::place::PlaceMap`, parsing with `from_chars`, `setlocale(LC_ALL,"")` at start; block bootstrap for CIs).
- [ ] **Step 3:** selftest passes. Expected: `selftest OK`.
- [ ] **Step 4: Commit** `git add retina/tools/place_xroom_eval.cpp && git commit -m "retina: place_xroom_eval, cross-room discrimination with pre-registered verdict"`.

### Task 0.3: Record, evaluate, decide (harness; controller session runs Webots)

**Files:** Create `runs/phase0/verdict.txt` (committed result only, not raw logs).

- [ ] **Step 1:** Run `phase0_capture.sh` for S1, S2, S3 in each of R1, R2, R3 (nine runs). Acceptance: each session folder has >= 300 query rows and `place_query_desc.f32` of matching size (`rows * stride * 4` bytes).
- [ ] **Step 2:** `place_xroom_eval ...` on the recordings. Acceptance: `verdict.txt` states FULL / WEAK / NONE with X1-X6 numbers, the fitted llr coefficients and `n_eff_per_s`.
- [ ] **Step 3: GATE G0** — the verdict and the numbers are shown to the user; the plan continues per the verdict above. Commit `verdict.txt` (`git add runs/phase0/verdict.txt`, if `runs/` is tracked; otherwise copy it to `docs/superpowers/plans/2026-10-23-room-recognition-phase0.md`).

---

## Phase 1 — the math core and the archived appearance memory

### Task 1: Joint posterior over (room k, pose) + new (common)

**Files:**
- Create: `common/room_recognition/room_posterior.h`
- Test: `common/room_recognition/room_posterior_test.cpp` (picked up by `common/run_tests.sh`)

**Interfaces — Produces** (namespace `rc::recog`):
```cpp
struct PoseGauss { Eigen::Vector3f mu; Eigen::Matrix3f S; float w; };   // SE(2), angle wrapped
using PoseMix = std::vector<PoseGauss>;
struct MetricEvidence { PoseMix pose_in_k; float log_inside = 0.f; bool available = false; };
        // pose_in_k: robot pose in room k's frame from the chain; log_inside = log P(robot inside polygon_k)
struct AppearanceEvidence { PoseMix pose_in_k; float loglr = 0.f; bool available = false; };
        // loglr: cumulative log p(views | room k)/p(views | none), already temporally discounted
struct Hypothesis { int room = -1;  /* -1 = new */ float log_post; PoseMix pose_in_k; };
std::vector<Hypothesis> posterior(std::span<const MetricEvidence> metric,
                                  std::span<const AppearanceEvidence> app,
                                  std::span<const float> polygon_area_m2,   // per known room
                                  float metric_log_new);                    // log P(pose outside all known polygons)
float discounted_loglr(std::span<const float> frame_llr, float dt_s, float n_eff_per_s);
struct Commit { enum class State { Entering, Committed } state; int room; };
Commit commit_step(Commit prev, std::span<const Hypothesis> post_since_entry,
                   std::span<const Hypothesis> post_since_commit);
```
Consumed by Tasks 4, 6. `Hypothesis` list order: known rooms in input order, then `new` last; `log_post` normalised (logsumexp = 0).

**Model (decides the algebra the tests pin):** for each known k the two experts give densities on the same pose x in k's frame; the joint is `pi_k * p_metric(x|chain) * p_app(x|views) / p(x)` with `p(x)` uniform over polygon k (`1/area_k`) so the shared prior is not counted twice. Evidence for k is the Gaussian-mixture product integral `sum_ij w_i w_j N(mu_i - mu_j; 0, S_i + S_j)` times `area_k`, times `exp(loglr_k)` for the appearance magnitude (the pose-mixture shape carries location, the llr carries room identity) times `P(inside)`. When one expert is `available == false` its factor is omitted (not zeroed). Pose disagreement lowers the integral — no extra term. `new`: `pi_new * exp(metric_log_new)` (appearance contributes llr 0 by construction: it is the reference). `discounted_loglr` multiplies the summed per-frame llr by `n_eff_per_s * dt_s / frames` (X5 of Phase 0), so correlated frames cannot manufacture confidence. `commit_step` is the staged rule: Entering -> Committed when argmax over `post_since_entry` is a known room; Committed -> Entering when argmax over `post_since_commit` (evidence gathered after the commit, prior reset) is `new` or a different room.

- [ ] **Step 1: Write the failing tests** (names and assertions):
  - `no_evidence_equals_prior`: both spans `available=false` -> posterior == uniform over {k..., new}, all finite. (Review Focus 1.)
  - `agreeing_experts_beat_one`: metric alone gives P(k)=0.7; adding agreeing appearance (mu within 1 sigma) raises it above 0.7.
  - `disagreeing_poses_lower_k`: same loglr, appearance mean 4 sigma from the metric mean: P(k) strictly lower than with agreement (pose consistency is evidence).
  - `no_double_counted_prior`: identical metric and appearance mixtures that are the polygon-wide prior (S huge) leave the posterior at the prior (+-1e-4).
  - `missing_map_uses_metric_only`: `app.available=false` for k reproduces the metric-only posterior exactly. (Review Focus 2.)
  - `symmetric_rooms_split`: two rooms with identical evidence -> equal log_post, `commit_step` stays Entering with a 0.5/0.5 split. (Review Focus 3.)
  - `correlated_frames_not_overconfident`: 200 frames of llr=+0.3 at 2 Hz with `n_eff_per_s=0.2` give `discounted_loglr` = 0.3 * 0.2 * 100 = 6.0 (+-1e-4), not 60.
  - `undo_uses_only_post_commit_evidence`: strong pre-commit evidence for k, then post-commit evidence favouring `new`: `commit_step` returns Entering.
  - `prior_sensitivity`: halving pi_new moves the posterior by < 0.05 when |loglr| >= 5 (documents flagged point 2).
  Run `common/run_tests.sh` — Expected: FAIL (header missing).
- [ ] **Step 2: Implement** the header per the model above; angles via the existing `wrap_angle`; `logsumexp` for stability.
- [ ] **Step 3:** `common/run_tests.sh` — Expected: all PASS.
- [ ] **Step 4: Commit** `git add common/room_recognition/room_posterior.h common/room_recognition/room_posterior_test.cpp && git commit -m "common: room_posterior, joint (room, pose)+new posterior with staged commit rule"`.

### Task 2: Per-room appearance maps archived with the room (retina + common)  *(dropped if Phase 0 verdict is NONE)*

**Files:**
- Create: `common/place_memory/place_registry.h`, `common/place_memory/place_registry_test.cpp`
- Modify: `retina/src/place_stage.{h,cpp}` (keyframes tagged with the CURRENT room; one `PlaceMap` per `room_id`; `map_path` template -> `etc/place_maps/room_<room_id>.bin`; saved when the room changes and on shutdown), `retina/etc/config.toml` (`[PlaceMemory] enabled = true`, `build_map = true`, constants from `verdict.txt` with provenance comment).

**Interfaces — Produces:**
```cpp
namespace rc::place {
class PlaceRegistry {
public:
    bool load(std::uint64_t room_id, const std::filesystem::path& dir);   // read-only; false if absent/corrupt/header mismatch
    bool has(std::uint64_t room_id) const;
    std::vector<MatchResult> match(std::uint64_t room_id, std::span<const float> q, int top_n,
                                   std::vector<float>* xcorr = nullptr) const;
    const PlaceMap* map(std::uint64_t room_id) const;
};
}
```
`room_id` is the `room_id_att` value (the same one `recall_*` carries and the memory twin holds). The header's `room_name`/`model_sha1`/`azimuth_tune` mismatch refuses the load (existing check), counted in a log line.

- [ ] **Step 1: Failing tests** in `place_registry_test.cpp` (sets `setlocale(LC_ALL,"")`): `load_roundtrip` (write a 20-keyframe map, load via registry, `match` returns the planted keyframe); `missing_file_is_absent_not_error` (`load` false, `has` false, `match` empty); `corrupt_header_refused`; `rooms_isolated` (queries of room A never match keyframes of room B's map). Run `common/run_tests.sh` — Expected: FAIL.
- [ ] **Step 2: Implement** registry (header-only, reuses `PlaceMap::load`). Then the PlaceStage change: on `room_id` change save the old map and open the new one; the provisional room (proto, no stable `room_id` yet) accumulates under its live name and is renamed to its `room_id` file when the room gets one; `place_stage` never reads other rooms' maps (that is Task 4).
- [ ] **Step 3:** `common/run_tests.sh` PASS; `cbuild -j8` retina; short live check: after an R1 explore `ls retina/etc/place_maps/` shows `room_<id>.bin` with > 0 keyframes. Expected: file present, size > header.
- [ ] **Step 4: Commit** `git add common/place_memory/place_registry.h common/place_memory/place_registry_test.cpp retina/src/place_stage.h retina/src/place_stage.cpp retina/etc/config.toml && git commit -m "retina: per-room DINO maps (room_id keyed) + read-only PlaceRegistry"`.

### Task 3: USER CHECKPOINT — register the new DSR attributes, user reinstalls cortex

**Files:** `cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h` (cortex checkout; the header under `/usr/local/include/dsr/...` is root-owned — do NOT `sudo cp`).

- [ ] **Step 1:** Add the eight `REGISTER_TYPE` lines from the attribute table above (reference-wrapper types exactly as listed; `recall_late` is `bool`).
- [ ] **Step 2:** STOP and ask the user to reinstall cortex. Acceptance: the user confirms; then `grep -c "recog_room_ids\|recog_loglr\|recog_mix\|recog_mix_n\|recog_posterior\|recall_pose_prior\|recall_late\|place_map_file" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h` prints `8`.
- [ ] **Step 3:** Rebuild retina, ltsm_agent, room_concept, door_concept with `cbuild -j8` (one at a time; `pgrep -a cc1plus` first). Acceptance: all link (nothing uses the attrs yet).
- [ ] **Step 4: Commit** in the cortex checkout (separate repository) by pathspec: `git add core/include/dsr/core/types/type_checking/dsr_attr_name.h && git commit -m "attrs: room recognition (recog_*, recall_pose_prior, recall_late, place_map_file)"`.

Tasks 4, 6, 7, 8, 9 depend on this checkpoint; Tasks 1, 2, 5 do not and may proceed before it.

---

## Phase 2 — metric evidence and appearance service

### Task 5: Path-taken seam chain vs each known room's own chain (ltsm)

**Files:**
- Create: `ltsm_agent/src/room_chain.h`, `room_chain.cpp`, `ltsm_agent/tools/room_chain_test.cpp`
- Modify: `ltsm_agent/CMakeLists.txt` (test target after `seam_fusion_test`, EXCLUDE_FROM_ALL, output `bin/`), `ltsm_agent/src/CMakeLists.txt` (source).

**Interfaces — Consumes:** `ltsm::seam_through_passage(mem, passage, to, from)` (`room_eviction.h:47`), `ltsm::seam::{Measured, compose_cov, inverse_cov}` (`seam_fusion.h`), archived polygons `delimiting_polygon_x/y_att`, the passage's crossing history. **Produces:**
```cpp
namespace ltsm {
struct ChainPose { Eigen::Vector3d pose; Eigen::Matrix3d cov; };            // SE(2) in the reference room
struct KnownRoomMetric { std::uint64_t room_id; std::string mem_room; bool chain_ok;
                         Eigen::Affine2d T_ref_k; Eigen::Matrix3d cov_T;      // room k in reference, from k's OWN chain
                         std::vector<Eigen::Vector2d> polygon_in_ref; double area; };
struct MetricQuery { std::string reference_room;       // path start (room_1)
                     ChainPose robot_in_ref;           // robot via the path ACTUALLY taken + odometry since the last seam
                     std::vector<KnownRoomMetric> rooms; };
MetricQuery build_metric_query(DSR::DSRGraph& mem, const std::vector<std::string>& path_passages,
                               const std::string& path_start_room, const ChainPose& robot_in_last_room);
// -> rc::recog::MetricEvidence for each room (log_inside by Gaussian mass in the polygon, pose_in_k, available=chain_ok)
std::vector<rc::recog::MetricEvidence> metric_evidence(const MetricQuery& q, float& log_new_out);
}
```
`rooms[k].T_ref_k` composes seams along the shortest passage path in memory from the reference room to k (BFS over `passages()`), NOT along the robot's path — the disagreement between the two chains is the evidence. `log_inside` = log of the Gaussian mass of `robot_in_ref` (cov = robot chain cov + `cov_T`) inside `polygon_in_ref`, by deterministic sigma-point quadrature over the polygon-clipped Gaussian (no sampling); `log_new_out` = log of the mass outside every known polygon.

- [ ] **Step 1: Failing tests** (`room_chain_test`, built on an in-process memory graph from `etc/stage_two_rooms.json` extended to three rooms in a loop R1-R3, R1-R2, R2-R3 with known seams, as `seam_evict_test` builds scenes):
  - `loop_consistency`: truth robot inside R3 reached via R1->R2->DOOR_2; chain via R1->R3 and chain via the path agree to < 1 cm when seams are exact; `log_inside(R3)` > log(0.99).
  - `seam_error_widens_not_flips`: 10 cm yaw error injected on the R1-R2 seam raises `cov` and keeps R3 the argmax; `log_new` stays below log(0.05).
  - `outside_everything_is_new`: truth pose in an unknown location 5 m from every polygon -> `log_new` ~ 0, every `log_inside` < log(1e-6).
  - `missing_face_marks_unavailable`: delete R3's face of the R1-R3 passage -> `chain_ok=false` for R3, `available=false`, no throw. (Review Focus 2.)
  - `staged_twin_is_not_a_candidate`: a memory room with `object_subtype "staged"` never appears in `rooms`. (Review Focus 5.)
  Run `make -C build room_chain_test && ./bin/room_chain_test` (from `ltsm_agent`) — Expected: FAIL.
- [ ] **Step 2: Implement** per the signatures; the crossing history source is the same one `passage_live`/`passage_harvest` already keep (read it first; `path_passages` is ordered oldest -> newest since the last known room).
- [ ] **Step 3:** run the test. Expected: `all ok`.
- [ ] **Step 4: Commit** `git add ltsm_agent/src/room_chain.h ltsm_agent/src/room_chain.cpp ltsm_agent/tools/room_chain_test.cpp ltsm_agent/CMakeLists.txt ltsm_agent/src/CMakeLists.txt && git commit -m "ltsm: room_chain, path-taken seam chain vs each known room's own chain"`.

### Task 4: Appearance matcher service and `place_evidence` publication (retina)  *(dropped if verdict NONE; needs Task 3)*

**Files:**
- Create: `retina/src/recognition_matcher.h`, `recognition_matcher.cpp`, `retina/tools/recognition_matcher_test.cpp`
- Modify: `retina/src/place_stage.cpp` (call matcher after encoding, when a `recognition` node with candidates exists), `retina/src/CMakeLists.txt`, `retina/etc/config.toml` (llr coefficients from `verdict.txt`, `n_eff_per_s`).

**Interfaces — Consumes:** `PlaceRegistry` (Task 2), `rc::recog::discounted_loglr`, `PoseMix` (Task 1), ltsm's `recognition` node attr `recog_room_ids`. **Produces:** retina-owned `metaconcept` node `place_evidence` (object_subtype "place_evidence"; removed on shutdown) carrying `recog_loglr`, `recog_mix`, `recog_mix_n` in `recog_room_ids` order.
```cpp
class RecognitionMatcher {
public:
    RecognitionMatcher(const rc::place::PlaceRegistry&, const MixtureParams&, const LlrModel&);
    struct Frame { std::vector<float> desc; Eigen::Vector3f robot_pose_room; Eigen::Matrix3f cov; std::uint64_t stamp_ms; };
    void set_candidates(std::vector<std::uint64_t> room_ids);   // clears accumulators of rooms that left the list
    void add(const Frame&);                                      // per-frame: match each candidate map, accumulate llr + mixture
    struct Result { std::vector<float> loglr; std::vector<rc::recog::PoseMix> mix; };
    Result result() const;                                       // cumulative, temporally discounted
};
```
The candidate mixture is expressed in room k's frame (keyframe `robot_pose`, `MixtureParams::rho` widths, yaw from the circular shift), the pose-to-reference conversion is NOT done here (ltsm owns frames). A candidate with no map yields `loglr=0` and an empty mixture (Review Focus 2).

- [ ] **Step 1: Failing tests** (`recognition_matcher_test`, synthetic maps from the planted clusters of Task 0.2's selftest): `own_room_wins` (queries from room A: `loglr[A] > 0 > loglr[B]` after 20 frames); `no_map_is_zero_not_error`; `candidate_change_resets_only_departed`; `mixture_peak_at_planted_pose` (top component within 0.3 m of the planted keyframe pose, width = `rho(sim)`); `parse_locale` (the llr coefficients read from a text file under `setlocale(LC_ALL,"")`). Run `g++ -std=c++23 ... && ./recognition_matcher_test` — Expected: FAIL.
- [ ] **Step 2: Implement.** Publication every `decimation` frames from the worker thread; graph writes are thread-safe, the node is owned by retina (`[Owns]` row added in Task 11).
- [ ] **Step 3:** test PASS; `cbuild -j8` retina.
- [ ] **Step 4: Commit** `git add retina/src/recognition_matcher.h retina/src/recognition_matcher.cpp retina/tools/recognition_matcher_test.cpp retina/src/place_stage.cpp retina/src/CMakeLists.txt retina/etc/config.toml && git commit -m "retina: RecognitionMatcher, per-candidate appearance llr + pose mixture on place_evidence"`.

---

## Phase 3 — ltsm: posterior, staged commit, swap

### Task 6: RoomRecognizer — evidence in, posterior out, staged commit state (ltsm; needs Tasks 1, 3, 5, optionally 4)

**Files:**
- Create: `ltsm_agent/src/room_recognizer.h`, `room_recognizer.cpp`, `ltsm_agent/tools/room_recognizer_test.cpp`
- Modify: `ltsm_agent/src/specificworker.{h,cpp}` (construct after `recall_`; `step()` in `compute()` after the promotion/before `recall_->step()`), `ltsm_agent/etc/config` (`[Recognition] enabled`), `ltsm_agent/CMakeLists.txt`, `src/CMakeLists.txt`.

**Interfaces — Consumes:** `build_metric_query`/`metric_evidence` (Task 5), `rc::recog::{posterior, commit_step}` (Task 1), `place_evidence` node (Task 4). **Produces:** ltsm-owned `recognition` metaconcept node (candidate ids `recog_room_ids`, `recog_posterior`; created when a provisional room exists behind a crossing whose door is NOT in any passage; removed when the room resolves or on exit), and
```cpp
class RoomRecognizer {
public:
    RoomRecognizer(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, const RoomEviction&);
    struct Step { enum class Act { None, Swap, Undo } act = Act::None; int room_index = -1;      // index into candidates
                  std::uint64_t room_id = 0; rc::recog::PoseMix pose_in_k; };
    Step step();            // one cycle; Swap/Undo is a REQUEST, executed by RoomSwap (Task 7)
    int sweep_stale();      // startup: remove stale recognition/place_evidence leftovers
};
```
Eligibility is structural, not thresholded: it runs exactly while a crossed door has no passage (the "entered FRESH" case) and stops when that room is no longer current. Candidates = archived rooms (`room` nodes in memory not `staged`) except the live current/provisional one. Metric: `robot_in_last_room` from the robot->room RT edge with covariance (as `promotion_evidence` reads it); `path_passages` from the crossing history. Evidence-since-entry and evidence-since-commit accumulators feed `commit_step`.

- [ ] **Step 1: Failing tests** (`room_recognizer_test`, in-process live+memory graphs from the three-room loop scene, evidence injected by writing a `place_evidence` node):
  - `entry_without_evidence_does_nothing`: provisional room, no `place_evidence`, chain exact but robot still at the door -> posterior published, `act == None`. (Review Focus 1.)
  - `metric_alone_commits_known_room`: robot truly in R3 per chain, no place_evidence -> after a few steps `act == Swap` with R3's `room_id`.
  - `appearance_alone_cannot_commit_against_metric`: appearance favours R3 strongly while the chain puts the robot outside every polygon -> no Swap (anti-goal: one modality alone must not decide when the other contradicts).
  - `new_room_stays_new`: chain outside all polygons, appearance llr < 0 -> never Swap over 200 steps; `recog_posterior` last entry (new) > 0.9.
  - `symmetric_candidates_no_commit` (Review Focus 3).
  - `stale_recognition_swept_at_start` (Review Focus 5).
  - `no_dsr_signal_connections`: `grep -n "DirectConnection" ltsm_agent/src/room_recognizer.*` returns nothing (test asserts via a source scan).
  Run `make -C build room_recognizer_test && ./bin/room_recognizer_test` — Expected: FAIL.
- [ ] **Step 2: Implement** per interface; the `recognition` node written with type-attributed attrs only.
- [ ] **Step 3:** test PASS; `cbuild -j8` ltsm_agent.
- [ ] **Step 4: Commit** `git add ltsm_agent/src/room_recognizer.h ltsm_agent/src/room_recognizer.cpp ltsm_agent/tools/room_recognizer_test.cpp ltsm_agent/src/specificworker.h ltsm_agent/src/specificworker.cpp ltsm_agent/etc/config ltsm_agent/CMakeLists.txt ltsm_agent/src/CMakeLists.txt && git commit -m "ltsm: RoomRecognizer, joint posterior + staged commit requests"`.

### Task 7: Swap and undo through the recall/promotion path (ltsm; needs Task 6, and room_concept Task 9 to complete live)

**Files:**
- Create: `ltsm_agent/src/room_swap.h`, `room_swap.cpp`, `ltsm_agent/tools/room_swap_test.cpp`
- Modify: `ltsm_agent/src/room_recall.{h,cpp}` (post a `late` recall under the CROSSED door for the recognised room: attrs `recall_late=true`, `recall_pose_prior`; existing nodes unchanged), `room_eviction.{h,cpp}` (`Candidate::discard` — delete the provisional live room subtree WITHOUT writing it to memory; archived rooms flagged `object_subtype "staged"` are skipped by `passages()` consumers, `RoomRecall::step`, `seed_from_working_memory`, and `build_metric_query`), `specificworker.cpp` (wire `Step` -> `RoomSwap`).

**Interfaces — Produces:**
```cpp
class RoomSwap {
public:
    RoomSwap(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, RoomEviction&, RoomRecall&);
    bool swap(std::uint64_t known_room_id, const rc::recog::PoseMix& pose_in_k);   // true when the request was accepted
    bool undo();                                                                    // back to "just entering"
    void on_room_left(const std::string& room);   // the undo window closes: drop the staged twin
    int  sweep_stale();                           // startup
    enum class State { Entering, Staged } state() const;
};
```
**Two cases, both specified:**
- **A. Provisional is still a proto (not promoted, no memory twin):** `swap` posts `recall_<known>` (late) under the crossed door and removes the proto via `Candidate::discard`; room_concept (Task 9) re-births the known room from the recall. `undo` removes that recall; room_concept re-runs the normal proto birth at the crossing (`rearm_after_crossing` without a recall). Nothing was written to memory in either direction.
- **B. Provisional already promoted (its memory twin `r<n>_room_<k'>` exists, the previous room is archived with a passage to it):** `swap` marks the provisional twin `staged` (kept until the robot leaves, this is the undo snapshot), posts the late recall for the known room, discards the live provisional subtree. The previous room's passage face that pointed into the provisional twin is NOT yet rewritten. `undo` re-posts a recall for the staged twin (the SAME recall mechanism with the provisional room as "remembered room"), clears `staged`, removes the known room's late recall. `on_room_left` (eviction of the recognised room): the staged twin is deleted from memory, and the carry-over of Task 8 runs.
The archive of the known room is READ-ONLY while `Staged`: any `evict()` write into the known room's twin is deferred until `on_room_left` (guard in `RoomEviction::evict`, test below).

- [ ] **Step 1: Failing tests** (`room_swap_test`, in-process graphs):
  - `swap_proto_leaves_no_trace_in_memory` (case A): memory room count unchanged after swap and after undo; live proto removed on swap.
  - `swap_promoted_stages_twin_then_drops_it` (case B): after swap the provisional twin has `object_subtype=="staged"` and is excluded from `RoomRecall::step`; after `on_room_left` it is gone. **No duplicate room node**: `count(type=="room" in memory) == known rooms`.
  - `wrong_commit_undone_without_corrupting_known_room` (spec reversal test): snapshot the known room's memory subtree (all node names, attrs, RT edges) -> swap -> feed evidence for `new` -> undo -> assert the known room's subtree is bit-identical to the snapshot and the provisional twin/live room is restored with its door faces.
  - `known_room_archive_read_only_while_staged`: `evict()` of a recalled room while `Staged` writes nothing into the known twin until `on_room_left`.
  - `turn_around_in_doorway`: swap, then undo, then swap again -> exactly one late recall node, no orphan `recall_*`. (Review Focus 4.)
  - `stale_staged_swept_at_start`: leftover `staged` twin and `recall_late` node from a "crashed run" are removed by `sweep_stale()`. (Review Focus 5.)
  Run `make -C build room_swap_test && ./bin/room_swap_test` — Expected: FAIL.
- [ ] **Step 2: Implement.** Reuse `remove_live_room`, the memory-twin copy (`copy_subtree`) and `twin` for the restore; new code is the state machine and the flags.
- [ ] **Step 3:** all PASS; `cbuild -j8` ltsm_agent.
- [ ] **Step 4: Commit** by pathspec of the files above with message `ltsm: RoomSwap, staged swap/undo through recall; read-only archive while staged`.

### Task 8: Carry-over of the new passage and twin dedupe (ltsm)

**Files:** Modify `ltsm_agent/src/room_swap.{h,cpp}`, `room_eviction.cpp` (reuse `fuse_seam_through_aperture`); test in `room_swap_test.cpp`.

**Interfaces — Consumes:** Task 7 `on_room_left`. **Produces:** `bool RoomSwap::carry_over(const std::string& crossed_door_prev_room_face, std::uint64_t known_room_id)`.

When the recognised room is finally left (or the swap completes in case A with the previous room already archived), the crossed door (DOOR_2): (1) the previous room's memory face already exists; (2) the known room gets a NEW face of that door, at the pose of the aperture in the known room's frame — from the relocalised robot pose through the aperture seam (the Task S/M path: `fuse_seam_through_aperture`, mouth pose when present); (3) both faces are `match`-linked into one `passage` node (existing vocabulary: `other_side_door_name`, `connected_room_name`); (4) in case B any previous-room passage face that pointed into the discarded provisional twin is re-hung to the known room (or removed, if (3) supersedes it). Result: DOOR_2 is a passage of R3 and R2, and the next crossing of it is an ordinary passage recall.

- [ ] **Step 1: Failing tests:** `door2_becomes_passage_of_both_rooms` (R3 has a face, R2 has a face, one `passage` node with two `match` edges, `seam_through_passage(R3,R2)` within 5 cm of truth); `second_crossing_recalls_directly` (after carry-over, `RoomRecall::step` posts `recall_room_3` under DOOR_2 with no late flag); `no_face_in_provisional_name` (no memory node name contains the provisional room's name); `carry_over_idempotent` (calling twice changes nothing).
- [ ] **Step 2: Implement.** **Step 3:** PASS. **Step 4: Commit** `ltsm: carry the newly crossed door into the recognised room (passage with both faces), dedupe provisional twin`.

---

## Phase 4 — room_concept and door_concept

### Task 9: Late adoption of the recalled room and relocalisation (room_concept; needs Tasks 3, 7)

**Files:**
- Modify: `room_concept/src/room_scene_graph.{h,cpp}` (`find_recall` also reads `recall_late` and `recall_pose_prior`; new `RoomSceneGraph::adopt_late_recall(const Recall&)`), `room_concept/src/room_concept.{h,cpp}` (`seed_reloc_prior(const std::vector<rc::reloc::PriorMode>&)` consumed by `relocalise_from_belief`), `room_concept/src/reloc_search.h` (`PriorMode { Eigen::Vector3f mu; Eigen::Matrix3f S; float w; }` entering as a continuity term in the mode weights, same role as the existing seed yaws/pose prior — a likelihood term, not a gate), `room_concept/src/CMakeLists.txt`, test `room_concept/tools/gn_selftest.cpp` (new section `late_recall`).

**Behaviour:** When a `recall_<room>` with `recall_late` appears while the current room is a provisional one: room_concept creates the room node with the REMEMBERED name via the same code path as `rearm_after_crossing(..., recall)`, outline from `poly_pub`, `recalled_visit_` = true; the provisional room is let go through the existing fence (Task 7 removes it). The robot is already inside, so `commit_relocalised_pose` is reached through the reloc mixture search with `recall_pose_prior` converted into `PriorMode`s (the doorway/`passage_prior` door anchor is NOT used: the robot is not in the doorway). Pose covariance after the jump comes from the Laplace modes as today. Undo: a late recall for the staged provisional room is handled by the same function.

- [ ] **Step 1: Failing selftest `late_recall`** (synthetic scan of a known polygon, robot 2.3 m inside, tracker initialised in a provisional frame with an arbitrary SE(2) offset): (a) the search with prior modes lands within 2 sigma of truth where the prior-free search from the same state lands on the symmetric twin (rectangle room, so a symmetric twin exists); (b) a prior placed 3 m off with a covariance admitting truth still recovers truth (the likelihood overrules the prior); (c) an empty prior reproduces today's result bit-for-bit. Run `make -C build gn_selftest && ./bin/gn_selftest late_recall` — Expected: FAIL.
- [ ] **Step 2: Implement** per above (`-j8`, `pgrep -a cc1plus` first; these TUs are large).
- [ ] **Step 3:** selftest PASS; full `./bin/gn_selftest` unchanged elsewhere.
- [ ] **Step 4: Commit** by pathspec `room_concept: adopt a late recall, relocalise with the recognition posterior as prior`.

### Task 10: door_concept — drop the provisional room's identities on a swap (door_concept; needs Task 7)

**Files:** Modify `door_concept/src/specificworker.cpp` (`enter_room_memory` ~L3430: a recall that targets the room being entered right after a provisional room of another name existed; remove the provisional room's identity rows and decision-log them), `door_concept/src/core/*` if the slice is already extracted; test in door_concept's existing test harness (find with `ls door_concept/tools door_concept/src | grep -i test`).

**Behaviour:** the late-recalled room goes through the `recalled` branch (restore of its remembered doors, `pending_restore_room_`), not "entered FRESH"; identity rows of the discarded provisional room name are deleted (not stripped to identity: that name is dead — room names are never reused). Review Focus 4 (turn-around): undo re-enters the provisional name as FRESH.

- [ ] **Step 1: Failing test:** replay a room-switch sequence (provisional `room_4` created, then `recall_room_3` late + room `room_3` created): assert log line `[restore] 'room_3' ... recalled`, no `entered FRESH` for `room_3`, no identity row of `room_4` remains. Run — Expected: FAIL.
- [ ] **Step 2: Implement.** **Step 3:** PASS; `cbuild -j8` door_concept. **Step 4: Commit** `door_concept: a late recall restores the recognised room's doors, drops the provisional room's identities`.

### Task 11: controller — missions survive a mid-leg room swap (controller; needs Task 7)

**Files:** Modify `controller/src/controller_mission.cpp`, `controller_session.cpp` as the investigation requires; test in the controller's existing mission test (the `selftest`/`*_test` built by `controller/CMakeLists.txt`).

**Behaviour:** `cross <door>` completes on "now in a different room from where it started" including when the new room's NAME changes a second time within the leg (provisional -> recognised). Read how the leg tracks the destination room (`current room changed` log) and make completion independent of the intermediate name. No new mission verb.

- [ ] **Step 1: Failing test:** a scripted sequence of room-change events `room_2 -> room_4 -> room_3` during one `cross` step: expected `completed`, not `failed/aborted`. Run — Expected: FAIL (or, if it already passes, record that and keep the test as a regression pin; commit only the test).
- [ ] **Step 2: Implement.** **Step 3:** PASS. **Step 4: Commit** `controller: a cross leg completes across a provisional->recognised room rename`.

---

## Phase 5 — conformance and validation

### Task 12: Lifecycle audit rows and docs (docs only)

**Files:** Modify `CONCEPT_AGENT_LIFECYCLE.md` (conformance rows for retina `place_evidence`, ltsm `recognition`/`recall_late`/`staged`; each ❌ justified; note the undo window is the eviction fence), `ltsm_agent/EVICTION.md` (staged twin, discard candidate, read-only archive), `retina` PlaceMemory comment block in `config.toml` (now consumed). Acceptance: `grep -n "recognition" CONCEPT_AGENT_LIFECYCLE.md | head` shows the rows; no broken links (`grep -o "\]([^)]*\.md)"` targets exist). Commit by pathspec `docs: lifecycle audit rows for room recognition`.

### Task 13: Gold tour harness and criteria report (harness)

**Files:**
- Create: `room_concept/estimation_scripts/scripts/recognition_tour.sh` (modelled on `gold_tour.sh`: same LOCK, fleet `retina room_concept residual_concept controller door_concept ltsm_agent table_concept`, same fresh-state removal of `ltsm_memory.json*`, `room_name_hwm.txt`, plus `retina/etc/place_maps/`; legs R1 -DOOR_1-> R3, R3 -> R1, R1 -DOOR_0-> R2, R2 -DOOR_2-> R3; DOOR_2 name discovered as in `gold_tour.sh` `new_door`; closes all doors then opens the crossed ones with `door.py`), `room_concept/estimation_scripts/scripts/recognition_report.py`, `room_concept/estimation_scripts/scripts/test_recognition_report.py`.

**Report criteria (the spec's, exact; `recognition_report.py <run_dir>` prints PASS/FAIL per line and exits non-zero on any FAIL):**
- C1 R3 (leg 1) and R2 (leg 3) on first entry: `recognition` posterior MAP is `new` at every logged step, and the ltsm log has no Swap for them; no `recall_late` posted.
- C2 leg 4 (R2 -> R3 via DOOR_2): exactly one Swap to R3's `room_id`, logged with the posterior and the metric/appearance contributions.
- C3 **no duplicate room**: `ltsm_memory.json` `type=="room"` nodes (non-staged) have unique names with count == 3 after the tour; no node named like the provisional room.
- C4 **pose**: robot pose error vs Webots ground truth (`gt_error.csv`, relative-metric method of `ab_report.py`) after the relocalisation settles is within 2 sigma of the localiser's own covariance (NEES <= chi2_2 0.95 quantile = 5.99 on the xy block) for the rest of the leg; reports the settle time.
- C5 **DOOR_2 passage**: memory has a `passage` with faces in both R2 and R3 (`seam_through_passage` within 5 cm of the wbt door position).
- C6 the tour continues: leg 5 (R3 -> R1, optional extra) completes via ordinary passage recall (`recall_room_1 posted`, no `Swap`).
- C7 no `wrong commit`: zero Undo events (reported; a nonzero count is a FAIL for the gold proof, not for the code).

- [ ] **Step 1: Failing test:** `test_recognition_report.py` with two canned run folders (a passing synthetic log set and one with a duplicate room node) asserting exit codes 0 and 1 and the C3 line text. Run `python3 -I room_concept/estimation_scripts/scripts/test_recognition_report.py` — Expected: FAIL.
- [ ] **Step 2: Implement** report and script; parse numbers in Python normally (locale-safe).
- [ ] **Step 3:** test PASS. **Step 4: Commit** by pathspec of the three files `harness: recognition_tour.sh + recognition_report.py (gold tour, spec criteria)`.

### Task 14: Live validation, three consecutive runs (controller session runs Webots)

- [ ] **Step 1:** `pgrep -a cc1plus`; `cbuild -j8` retina, ltsm_agent, room_concept, door_concept, controller (one at a time). Confirm `retina/etc/place_maps/` is wiped by the script.
- [ ] **Step 2:** `bash recognition_tour.sh recog_try1`, then `python3 recognition_report.py crossing/recog_try1`. Acceptance: C1-C7 all PASS.
- [ ] **Step 3:** repeat for `recog_try2`, `recog_try3` (fresh state each). Acceptance: 3/3 PASS. Failures are investigated from the per-run logs (`recognition` posterior trace, `room_identity.txt`), one hypothesis at a time; no tuning of the report.
- [ ] **Step 4: Negative control run** `recog_neg`: same tour with R3's `place_maps` file deleted before leg 4 (appearance absent). Acceptance: still C2-C5 PASS (metric carries it) or an honest FAIL report; either result recorded in the ledger. This pins Review Focus 2 live.
- [ ] **Step 5:** Record the results (try ids, C1-C7 table, posterior values at the Swap, settle time) in the plan's ledger (append a "Ledger" section to this file) and commit `docs: room recognition gold-tour results`. Stop agents with SIGTERM only.

---

## Self-review notes

- **Spec coverage:** Phase 0 measurement -> 0.1-0.3; per-room DINO maps archived -> 2 (+ `place_map_file` attr on the memory twin is set in Task 7's swap/eviction wiring when the twin is written); metric chain + pose density -> 5; joint posterior + new -> 1, 6; staged reversible commit -> 1 (`commit_step`), 6, 7; swap through recall/promotion -> 7; late-recognition relocalisation -> 9; carry-over -> 8; gold tour + reversal test -> 13, 14, 7 (`wrong_commit_undone_without_corrupting_known_room`); user checkpoint -> 3. Non-goals respected: no global map, no change to step 1 (no audit), seams not re-optimised.
- **`place_map_file` writer:** ltsm sets it on the memory twin at eviction (`copy_subtree` writes the attribute from `etc/place_maps/room_<room_id>.bin` if the file exists); add this line to Task 7 Step 2 when implementing — retina finds maps by the `room_id` convention, the attribute is for audit/the topology view.
- **Dependency order:** 0.1 -> 0.2 -> 0.3 (G0) ; 1 and 5 anytime ; 2 after G0 ; 3 (user) before 4/6/7/8/9 ; 4 after 2+3 ; 6 after 1+3+5 ; 7 after 6 ; 8 after 7 ; 9 after 3+7 ; 10, 11 after 7 ; 12 any time after 8 ; 13 any time ; 14 last.
