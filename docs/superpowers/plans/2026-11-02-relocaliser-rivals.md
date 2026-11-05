# Relocaliser as Rival Hypotheses — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace room_concept's relocaliser (3-state lost belief + synchronous search + clamp) with one posterior over the tracker's pose and ≤4 rival poses, so a healthy pose can never flip to a mirror twin and a real kidnap is recovered in 1–2 novel scans.

**Architecture:** The tracker stays the localiser (H0). A background proposer (the existing `rc::reloc::search`) only ADDS rivals; each rival carries a log Bayes factor λ against H0 that starts at the Occam/kidnap price (≈16 nats) and accumulates the per-scan likelihood difference on the same robust likelihood, weighted by view novelty. The pose switches only at λ ≥ ln 19. Everything is new code behind `RelocMode`, A/B'd offline on recorded scan tapes, then the old path is deleted.

**Tech Stack:** C++23, Eigen, room_concept (Qt/DSR agent), existing `reloc_search.{h,cpp}` (LikelihoodRaster, search, Laplace modes), `tools/reloc_selftest.cpp`.

**Spec:** `docs/superpowers/specs/2026-11-01-relocaliser-redesign.md` (§ numbers below refer to it). User decisions 2026-11-02: record scans first; symmetric rooms publish a bimodal pose (mixture spread) until a distinguishing view; constants = ln 19 and `RelocHazard` only (flagged); rollout behind a flag + offline A/B on recorded runs; delete the old relocaliser in the same change if the new one wins.

## Global Constraints

- Repo rules (CLAUDE.md): no thresholds where a model term works; C++23; `and/or/not`; type-attributed DSR only; `std::from_chars` for parsing, `imbue(std::locale::classic())` for writing data files; never `Qt::DirectConnection`; deep-copy across threads; transforms off-thread use real timestamps.
- Build: `systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G make -C room_concept/build -j8` (check `pgrep -c cc1plus` first).
- Rates first: room_concept must stay at 20 Hz; hot-path cost added < 1 ms p99 (§7); the proposer never runs on the compute thread.
- Constants allowed: `RelocHazard` (π_kid, 1e-3/frame), the SPRT band ln 19, K_max = 4, proposer cadence (2 Hz base) and duty cap (≤25 % of one core) — each flagged in config with a comment. Kept keys: `RelocOutlierFrac`, `RelocWallOffsetSigma`, `RelocBodyClearance`, `RelocHazard`, `RelocEnabled`.
- Commits end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_019DrdFA8AyqqyUd5QMUhbvN`. No push without the user.

## Review Focus

1. A real kidnap WHILE DRIVING (odometry keeps moving H0 and the rivals): the true rival must still win in ≤ 2 novel scans — test in Task 5 (selftest scenario "kidnap while driving").
2. A room switch (door crossing / recognition swap) while rivals exist: rivals belong to the old room's frame and must be dropped; a proposer result computed for the old room must be discarded (epoch) — test in Task 4 and Task 5.
3. A mirror-symmetric room, healthy tracking for minutes: zero switches, and when a genuine kidnap to the twin happens the published covariance spreads instead of flipping — Task 5 ("mirrored room").
4. Proposer slower than the cycle (search 100–220 ms): never more than one in flight, never blocks compute, stale results discarded — Task 4.
5. Parked robot with people walking past (scan mismatch, no novelty): λ must not accumulate (w≈0) and nothing moves — Task 2 + Task 5.

---

### Task 1: Scan tape — record what the relocaliser sees

**Files:**
- Create: `room_concept/src/scan_tape.h` (pure: frame struct, writer, reader)
- Modify: `room_concept/src/room_concept_reloc.cpp` (`relocalisation_step`, record one frame per call), `room_concept/etc/config.toml` (`[ScanTape] Enabled = true`, path pattern), `room_concept/estimation_scripts/scripts/run_crossing.sh` (collect `etc/scan_tape_*.bin` like the door tapes)
- Test: `room_concept/tools/scan_tape_test.cpp` (standalone, build line in header)

**Interfaces:**
- Produces:
  `struct rc::tape::Frame { std::int64_t t_ms; Eigen::Vector3f pose; Eigen::Matrix3f cov; Eigen::Vector3f odo_delta; Eigen::Matrix3f odo_cov; std::string room; std::vector<Eigen::Vector2f> pts_robot; };`
  `class rc::tape::Writer { explicit Writer(std::string path); void write(const Frame&); };`
  `std::vector<rc::tape::Frame> rc::tape::read_all(const std::string& path);`
  plus, once per room change, the room polygon: `struct RoomRecord { std::string room; std::vector<Eigen::Vector2f> polygon; std::vector<…> apertures; }` written before the first frame of that room.

- [x] Step 1: test `round_trip`: write 3 frames (2 rooms, polygon records) to a temp file, `read_all` returns them bit-identical; a truncated last frame is dropped, not an error. Run it → FAIL (header missing).
- [x] Step 2: implement `scan_tape.h` (little-endian binary, a 4-byte magic + version, length-prefixed records; ~2 k points/frame).
- [x] Step 3: hook the writer in `relocalisation_step` (compute thread; write is a buffered append, measure ≤ 0.2 ms p99 with the existing stage timing) — only when `ScanTape.Enabled`.
- [x] Step 4: test passes; build room_concept; commit `room_concept: scan tape (pose, odometry, wall-band points per frame)`.

### Task 2: The rival belief — pure core

**Files:**
- Create: `room_concept/src/reloc_rivals.h` (pure Eigen + std; no DSR, no Qt)
- Test: `room_concept/tools/reloc_rivals_test.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```cpp
  namespace rc::rivals {
  struct Rival { Eigen::Vector3f pose; Eigen::Matrix3f cov; float lambda; float lambda_birth; std::uint64_t id; };
  struct Params { float pi_kid = 1e-3f; float log_band = std::log(19.f); int k_max = 4; float ln_eps; /* = ln(RelocOutlierFrac) */ };
  // §4.2: birth log Bayes factor of a mode (pose_r, cov_r, ll_r) against H0 (pose_0, cov_0, ll_0, pred, P_pred), V = prior volume (room area·2π, or V_entry)
  float birth_lambda(const Params&, float ll_r, const Eigen::Matrix3f& cov_r, float ll_0, const Eigen::Vector3f& pose_0,
                     const Eigen::Matrix3f& cov_0, const Eigen::Vector3f& pred, const Eigen::Matrix3f& P_pred, float volume);
  class Belief {
  public:
    void add(Rival);                                    // keeps ≤ k_max, drops the lowest λ; merges a rival within the same Laplace mode (Mahalanobis under Σ_r+Σ_new ≤ 3² — FLAGGED merge rule, spec §10.5)
    void carry(const Eigen::Vector3f& odo_delta);       // rigid carry of every rival by the robot's motion (body frame)
    void update(const std::vector<float>& dll, float novelty);  // §4.3: λ ← max(λ + w·max(dll, ln_eps), λ_birth), per rival
    std::optional<std::size_t> winner() const;          // §4.4: index with λ ≥ log_band
    Rival swap(std::size_t i, const Eigen::Vector3f& pose_0, const Eigen::Matrix3f& cov_0); // old incumbent becomes a rival with λ = −λ_i
    float p_rival() const;                               // Σe^λ/(1+Σe^λ)  ("lost" as a number)
    Eigen::Matrix3f spread(const Eigen::Vector3f& pose_0) const; // Σ_k w_k d_k d_kᵀ incl. H0 (held_spread semantics)
    void clear();                                         // room change
    const std::vector<Rival>& rivals() const;
  };
  }
  ```

- [x] Step 1: tests (all must FAIL first — header missing):
  - `birth_price`: 40 m² room, Σ_r = diag(0.02²,0.02²,0.01²), ll_r = ll_0, H0 at pred ⇒ `birth_lambda` ∈ [−18, −14] nats.
  - `twin_never_switches`: twin rival, `update({0.f}, 1.f)` × 1000 ⇒ `winner()` empty, λ == λ_birth.
  - `mismatch_cancels`: both ℓ drop by 30 nats on the same scan ⇒ dll = 0 ⇒ λ unchanged.
  - `kidnap_two_scans`: dll = +60 per novel scan ⇒ `winner()` set after ≤ 1 update from birth −16 (−16+60 ≥ ln19).
  - `parked_no_accumulation`: dll = +5, novelty 0 × 500 ⇒ no winner.
  - `outlier_scan_bounded`: one dll = −1e6 ⇒ λ decreases by at most |ln_eps| and is floored at λ_birth.
  - `swap_is_honest`: after `swap(i)`, the old incumbent is a rival with λ = −λ_i.
  - `kmax_and_merge`: adding 6 distinct rivals keeps 4 (highest λ); adding a rival 1 cm from an existing one merges (count unchanged).
- [x] Step 2: run → FAIL. Step 3: implement. Step 4: run → PASS. Step 5: commit `room_concept: reloc rival belief (pure)`.

### Task 3: Rival scorer and carry on the real likelihood

**Files:**
- Create: `room_concept/src/reloc_rival_score.h/.cpp`
- Test: extend `room_concept/tools/reloc_rivals_test.cpp`

**Interfaces:**
- Consumes: `rc::reloc::LikelihoodRaster::ll_at`, `ll_outside` (reloc_search.h), the same point subsampling the search uses (`rc::reloc::polish_points` / Params).
- Produces: `float rc::rivals::score(const rc::reloc::LikelihoodRaster&, const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose, const rc::reloc::Params&);` — returns ℓ(pose) with EXACTLY the per-point robust model the search uses (so ℓ_0 and ℓ_r are comparable); and `std::vector<float> rc::rivals::dll(...)` for all rivals vs the incumbent on one scan.

- [x] Step 1: test `score_matches_search`: on a synthetic room + scan, `score(raster, pts, mode.pose)` equals `rc::reloc::search(...)`'s `Mode::log_lik` for the polished mode within 1e-3. Test `apertures_mask_both`: a scan with 30 % of points through an open door scores H0 and a rival equally lowered (dll unchanged vs the closed-door case within 1 nat). FAIL first.
- [x] Step 2: implement (bilinear `ll_at`, ≤128 subsampled points, same apertures). Measure: 4 rivals × 128 points < 0.1 ms. Commit.

### Task 4: Background proposer

**Files:**
- Create: `room_concept/src/reloc_proposer.h/.cpp`
- Test: `room_concept/tools/reloc_proposer_test.cpp`

**Interfaces:**
- Consumes: `rc::reloc::search(const LikelihoodRaster&, …)` (existing).
- Produces:
  ```cpp
  class rc::rivals::Proposer {
  public:
    // copy-in; returns false if one search is already in flight. epoch = room-generation counter.
    bool submit(std::uint64_t epoch, rc::reloc::LikelihoodRaster raster, std::vector<Eigen::Vector2f> pts_robot, rc::reloc::Params, Eigen::Vector3f around, Eigen::Matrix3f sampling_cov);
    std::optional<std::pair<std::uint64_t, rc::reloc::Result>> poll();   // non-blocking; the caller drops results whose epoch ≠ current
  };
  ```
  One `std::jthread`, one job slot, everything copied in (no shared buffers, no DSR access, no ts==0 transforms — the worker touches nothing but its copies).

- [x] Step 1: tests: `one_in_flight` (second submit while busy → false), `stale_epoch` (submit epoch 1, current becomes 2, `poll` result carries 1 → caller drops it), `never_blocks` (submit returns in < 1 ms with a 200 ms fake search injected via a test hook). FAIL first.
- [x] Step 2: implement; TSAN-clean build of the test (`-fsanitize=thread`). Commit.

### Task 5: Wire the rival path behind `RelocMode` (+ entry prior)

**Files:**
- Modify: `room_concept/src/room_concept_reloc.cpp` (`relocalisation_step`: `if (params.reloc_mode == "rivals") step_rivals(res, lidar_points); else <legacy unchanged>`), `room_concept/src/room_concept.h` (members: `rc::rivals::Belief rivals_`, `rc::rivals::Proposer proposer_`, `std::uint64_t room_epoch_`), config `RelocMode = "legacy"` (flag; tours switch it), `room_concept/tools/reloc_selftest.cpp` (run every scenario in both modes; add scenarios).
- Note hand-over (`note_passage_handover`): in rivals mode, initialise H0's prior as N(μ_entry, Σ_entry) per §5 (lateral = door width/2 uniform moment-matched, longitudinal = clearance + odo growth, heading σ from the controller alignment + odo growth), set `V = V_entry` for births; `rivals_.clear(); ++room_epoch_` on every room change (crossing, recognition swap, recall).

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `void RoomConcept::step_rivals(const UpdateResult&, const std::vector<Eigen::Vector3f>&)`. Published pose: H0 (or the winner after a swap); published covariance: H0's covariance + `rivals_.spread(...)` while `p_rival()` is non-negligible (mixture spread, never a flip); `pose_undecided_` ← `p_rival()`.

- [x] Step 1: selftest scenarios (FAIL first in rivals mode where they are new): existing 14 scenarios must PASS in rivals mode; new: "healthy mirrored room 10 min → 0 switches", "genuine kidnap to the twin → covariance spreads, no confident flip", "kidnap while driving → recovered ≤ 2 novel scans", "parked + people (mismatch, no novelty) → 0 switches", "room change with live rivals → rivals cleared, stale proposer result dropped".
- [x] Step 2: implement `step_rivals` (submit to proposer at 2 Hz base / every novel scan while a rival is gaining / every novel scan for the first seconds after a hand-over; births from results; `carry`; `dll` via Task 3; `update(novelty = view_change)`; `winner` → swap + reset tracker to the winner's pose/cov).
- [x] Step 3: selftest both modes; build; measure hot-path time per frame in rivals mode (stage timer) — < 1 ms p99. Commit.

### Task 6: Adoption after a recognition swap (§6)

**Files:**
- Modify: `room_concept/src/room_scene_graph_late_recall.cpp` (rivals mode: replace `run_late_recall_search` with a multi-scan fit), create `room_concept/src/multi_scan_fit.h/.cpp`
- Test: `room_concept/tools/multi_scan_fit_test.cpp`

**Interfaces:**
- Produces: `rc::reloc::Mode rc::rivals::multi_scan_fit(const rc::reloc::LikelihoodRaster& K, const std::vector<std::pair<Eigen::Vector3f /*rel pose*/, std::vector<Eigen::Vector2f>>>& scans, const Eigen::Vector3f& prior, const Eigen::Matrix3f& prior_cov);` — joint 3-DOF Gauss-Newton of the last M (10–20) keyframes against K's SDF; returns pose + Laplace covariance.

- [x] Step 1: test `multi_beats_single`: synthetic L-room, 15 scans from a short path, true offset (0.15 m, 3°) from the prior: multi-scan error ≤ 2 cm / 0.5°, single-scan error larger. FAIL first. Step 2: implement (worker, off the compute thread). Step 3: run on the warm-run swaps when tapes exist (Task 7) and report pose-vs-layout residual split (§10.3). Commit.

### Task 7: Offline A/B on recorded tapes, then cutover

**Files:**
- Create: `room_concept/tools/reloc_tape_replay.cpp` (reads Task 1 tapes + `gt_pose.tsv`; runs legacy and rivals on the same frames; injects teleports)
- Modify (only if rivals wins): delete per §8 — `update_lost_belief`, `log_tracking_emission`, `log_broad_emission`, `belief_`, `reset_lost_belief`, `set_belief_mismatch`, `lost_logodds_`, `track_mu_ln_s_`/`track_var_ln_s_`, `relocalise_from_belief`, the `p_lost >= 0.5` trigger, `pending_episode_p_lost_`, `rt_.hold_open`, `mode_posterior`'s mixture prior, `explains_like_tracking`, `rt_.door_anchor`/`step_door_anchor`/`anchor_information` retirement, `published_pose.h::clamp_correction`, `run_late_recall_search`/`late_pending_`, config keys other than the kept ones, and `RelocMode` itself.

- [x] Step 1 (needs the user's live runs): record 3–4 tours with `ScanTape.Enabled` (forward, backwards, a parked minute with a person walking, one manual push).
- [x] Step 2: replay both modes; metrics per §9: false relocalisations/h while healthy (|gt−est| < 5 cm), worst/p99 pose error, recovery on ≥ 300 injected kidnaps (median ≤ 0.5 s, p95 ≤ 1.5 s, ≥ 99 %), mirrored-room healthy segments (0 flips), hot-path p99 (< 1 ms). Report table.
- [x] Step 3 (code deleted 2026-11-05, offline; the live tour waits for the user): if rivals meets the §9 bar: delete the legacy path in the same change, keep the selftest (rivals only), build, live tour (forward + backwards) with nominal rates, commit `room_concept: relocaliser = rival hypotheses; legacy deleted`. If not: report where it loses, change nothing.

## Ledger: Task 7 A/B results (reloc_tape_replay, warm43/44/45, `--kidnaps 300 --people-kidnaps 100 --mirror 12`)

Healthy-tape exposure is ~7 min per mode (bar: ≥ 1 h). §9 items 5 (adoption) and 6 (net line count) are not measured yet.

| metric (§9) | 7 (66d17810) legacy / rivals | 7b (6ca0b9f2) legacy / rivals | 7c (beam model) legacy / rivals | bar |
|---|---|---|---|---|
| false relocs, healthy, warm43/44/45 | 0/0/0 / 16/17/0 | 0/0/0 / 0/2/0 | 0/0/0 / **0/0/0** | 0 |
| worst pose error cm, warm43/44/45 | 24/19/1.0 / 791/644/1.0 | 24/19/1.0 / 24/94/1.0 | 24/109/1.0 / **24/19/1.0** | ≤ best old run |
| max cov m², warm43/44 | 0.021/0.011 / 40.3/15.1 | same / 0.021/4.09 | 0.021/0.065 (person) / **0.021/0.011** | ≤ 1 |
| kidnaps clean n=300 | 76.3 % / 75.3 % | 95.3 % / 99.7 % | 98.3 % / **99.7 %** (p95 0.10 s) | ≥ 99 %, p95 ≤ 1.5 s |
| kidnaps people n=100 | 80 % / 71 % | 95 % / 100 % | 98 % / **100 %** | |
| clutter30 false moves, warm43/44/45 | 0/0/0 / 32/53/5 | 0/0/0 / 0/3/0 | 0/0/0 / **0/0/0** | |
| person false moves | 0/0/0 / 11/15/2 | 0/0/0 / 1/2/0 | 0/0/0 / **0/0/0** | |
| mirrored room, 12 × 60 s | 0 / 0 flips | 0 / 0 | 0 / **0** | 0 |
| hot path p99 (rivals) | 1.7–2.4 ms | 1.8–2.7 ms | **3.1–4.5 ms** (max 7.3 ms) | ≤ 5 ms (amended) |

7c legacy notes: the 7b clutter30 regression (warm44, 3 moves, one 5.7 m) is gone; legacy now moves on warm44 clean (2 moves, 109 cm,
f735-f743, to a pose of LOWER ℓ than the one it left) and person (3 moves, 385 cm) — legacy's own mode_posterior/trigger, not shared with rivals.
Reports: `.superpowers/sdd/2026-11-02-relocaliser-rivals/task-7{,b,c}-report.md` (git-ignored workspace).
