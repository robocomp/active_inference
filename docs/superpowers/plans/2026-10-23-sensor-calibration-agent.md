# Head-sensor calibration agent (`sensor_calibration`): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Phases 3–6 are written at task level; expand each into steps (same format as Phases 1–2) before executing it.

**Goal:** Calibrate the four head sensors (Helios, Bpearl, ZED, Ricoh) online, in a new agent outside the
localiser loop. Move the localiser-free calibration code out of room_concept.

**Architecture:** The split follows the localiser-loop boundary. The motion calibrator and the kinematic mount
factor stay in room_concept, because they read per-cycle localiser internals and must undo their own applied
feedback. Every factor that needs only sensor streams, odometry/IMU and the published pose/polygon moves to a new
agent, `sensor_calibration`. These are Noé's offline factors plus the r2 floor/vertical factors. That agent
writes *proposals*. `robot_concept`, which owns the body→sensor RT edges, *commits* them to the graph and to a
versioned per-robot mount file. Consumers already transform through the RT tree, so a committed edge reaches
them without new plumbing.

**Tech Stack:** C++23, CORTEX/DSR, RoboComp agent template (`robocompdsl`, mirror `bottle_concept`), media
plane (`common/media_transport`), Eigen.

**Spec / sources:**
- the doc "Opportunistic Self-Calibration of the Shadow Sensor Head" (https://claude.ai/artifact/TC7U92HFEpHQUhuDbb9nsX), including its "Comparison…" and "Improvements…" sections;
- `/home/pbustos/drive/Tesis/Noe Zapata/tesis/self_calibration.tex`;
- `SELF_CALIBRATION.md`, `SELF_CALIBRATION_DESIGN.md`, `room_concept/DESIGN_THREE_DEVICE_EXTRINSICS.md`;
- `docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md`.

## Answer to "can it be a separate agent?"

**Partly, and the part that can move is the part Noé's design needs.** These are the coupling findings
(2026-10-23):

| Subsystem | Lines | Verdict | Why |
|---|---|---|---|
| Motion calibrator (`motion_calibration.h`, `calibration_estimator.h`, `calibration_intake.h`) | ~1.3k | **Stays** | Reads per-cycle `pred_*`, `heading_cov`, zupt gains, `noise_innov_`; applies inside the odometry window (`room_concept_odometry_window.cpp:293-402`); needs an exact per-episode record of what was applied |
| KinematicMount (`room_concept_mount.cpp:33-92`) | — | **Stays** | Fed from `observe_innovation`, coupled to the motion calibrator's `last_information().H` and `noise_innov_` |
| Calib pivot (`calib_pivot.h`, in RoomSceneGraph) | ~0.3k | Stays (off on Shadow) | Could move, but gains nothing |
| Floor / vertical factors (`mount_factors.h` floor + vertical, `lidar_mount.h`) | ~0.7k | **Moves** | Need sweeps (media plane), the pose and polygon (already on the graph) |
| Camera mount block (`mount_calibrator.*`, `calib_channel*`, `camera_calibration.h`, `camera_jacobian.h`, `mount_lidar_pair.h`) | ~3.2k | **Retired after replacement** (Phase 5) | Needs in-process LiDAR corner matches; Noé's ZED floor plane + Ricoh edge factors replace it from raw streams |
| Joint monitor / joint calibration | ~0.4k | Moves (log) | Reads results only |
| Calibration viewer | ~0.6k | Splits | Motion part stays, mount part moves |

Expected reduction: **~5k of room_concept's 64k lines and ~45 of its ~85 calibration config keys**. The
motion calibrator (~1.5k with glue) remains.

## Global Constraints

- Follow `CLAUDE.md`:
  - No thresholds: weight by covariance, with no situation gates (no "speed > k", no info cut-offs).
  - Connect DSR signals Queued, never `Qt::DirectConnection`.
  - Use type-attributed attribute access only. A new attribute means a `REGISTER_TYPE` in cortex, and the
    **user reinstalls cortex**.
  - Parse numbers with `std::from_chars` only; write files through `imbue(std::locale::classic())`.
  - Build with `cbuild` / `-j8`; stop agents with SIGTERM, never `kill -9`.
- Agent pattern: mirror `bottle_concept` (presence debounce ~3000 ms, stale sweep, cleanup).
  - `[Agent] id = 27`, the first free id; ids in use are 5–26.
  - Add a conformance row to `CONCEPT_AGENT_LIFECYCLE.md`.
- Media subscribers only via `rc::media::make_lidar_subscriber_from_graph` / `make_image_subscriber_from_graph`.
  Create them after the sensor nodes are verified, on domain 7 only.
- Real vs sim: real-Shadow mounts live in a separate versioned file. `robot_concept/shadow.json` stays the
  Webots description and is never overwritten with real values.
- Ownership:
  - `robot_concept` is the **only** writer of body→sensor RT edges.
  - `sensor_calibration` writes only its own node's proposal attributes.
  - `room_concept`'s `mountPublish` path is removed in Phase 5.
- Ignore the existing `self_calibration/` agent (arm camera). Do not modify or reuse it.

## Review Focus

1. **A committed RT edge must reach the running LiDAR reader.** `LidarIngestor` polls with
   `LIDAR_ROBOT_FRAME` through the RT tree (`lidar_ingestor.cpp:167`). If its inner-eigen instance lives on a
   raw thread, its ts==0 cache never invalidates and the commit is silently ignored. Test: Task 1.3.
2. **Real vs sim confusion.** A real mount file loaded in a Webots run, or the reverse, must refuse to load
   (robot + world tag mismatch), not apply. Test: Task 1.2.
3. **Helios commit while localising.** Changing the Helios edge rotates every map/pose; commit must happen only
   when the robot is idle, and Helios before the cameras. Test: Task 6.3.
4. **Proposal flood.** The agent publishes proposals at ≤1 Hz, and only when the posterior changes. Write
   storms on the shared graph are a known crash source. Test: Task 2.3.
5. **Double estimation during migration.** While a factor exists in both agents, only one may write proposals.
   Test: Task 3.3 (room_concept copy disabled before the agent copy publishes).

---

## Phase 0: real-robot benchmark (prerequisite; data question open)

### Task 0.1: Noé's runs as a replayable benchmark
**Open question for the user:** where are the four real runs and two static takes recorded, and in what format
(rosbag / our DDS record / CSV)? The format decides whether they replay through the media plane or need a
converter.
- Deliverable: `sensor_calibration/bench/shadow_real_2026-10-22/README.md`, recording each run's path plus Noé's
  per-sensor offline values as `expected.csv`. Columns: `sensor,dof,value,sigma,unit,source`, with the rows
  from his doc. Examples: `helios,yaw_vs_chassis,1.84,,deg,straight-line`; `helios,roll,0.46,,deg,walls`;
  `helios,pitch,-0.30,,deg,walls`; `zed,roll,0.62,,deg,floor`.
- Every later phase's verification compares against this file on real data, and against planted errors in Webots.

## Phase 1: mounts as graph truth, per robot (robot_concept)

### Task 1.1: overlay all four sensor mounts from a versioned file
**Files:**
- Modify: `robot_concept/src/specificworker.cpp:2325-2440` (`apply_mount_overlay`, `persist_mount_overlay`,
  `mount_overlay_path`).
- Test: `robot_concept/tests/mount_overlay_test.cpp`.

**Interfaces (produces):**
- File `etc/mount_calib_<robot>_<world>.txt`, one line per sensor:
  `sensor x y z roll pitch yaw version iso_date source`. `<world>` is `real` | `webots`.
- `apply_mount_overlay()` covers `helios`, `bpearl`, `zed` and `ricoh`; today it covers zed and ricoh only.

- [ ] Step 1: Failing test. Write a file with helios yaw +1.84° and bpearl z −0.005. After the overlay, the
  graph body→helios RT edge yaw equals nominal +1.84° (1e-9), and bpearl z equals nominal −0.005.
- [ ] Step 2: Run it, expect FAIL (helios and bpearl are not overlaid).
- [ ] Step 3: Implement. Parse with `std::from_chars`; write with the classic locale. Compose
  the correction in BODY-aligned axes (`R = R_corr·R_nom`, `t = t_nom + t_corr`; ruling 10-23: helios/bpearl are rot y=π, so a sensor-frame yaw would flip sign), never overwrite with absolute values. Keep `version` monotonic.
- [ ] Step 4: Run it, expect PASS. Run the existing robot_concept tests.
- [ ] Step 5: Commit `robot_concept: mount overlay covers all four head sensors, per robot+world file`.

### Task 1.2: refuse a mismatched world
- Test `mount_overlay_rejects_world_mismatch`. A `_real` file is offered while the config says Webots (key
  `Transforms.world = "webots"|"real"`, new). Expected: not applied, one warning, nominal edges unchanged.
- Implement the check; commit.

### Task 1.3: a changed edge reaches room_concept's LiDAR points at runtime
**Files:**
- Test: `room_concept/tests/lidar_mount_edge_live_test.cpp`, or a scripted Webots check if the reader cannot
  be unit-tested.
- Possibly modify: the reader / `LidarIngestor` inner-eigen ownership.

- [ ] Step 1: Test. Modify the body→helios edge yaw by +1° while the ingestor runs. Within 2 sweeps, a known
  wall point's body-frame bearing changes by 1° ± 0.05°.
- [ ] Step 2: Run it. If it FAILS, the cause is the ts==0 cache on a thread without an event loop (see
  CLAUDE.md "InnerEigenAPI" rule). Fix by using the main-thread-owned instance or a real timestamp.
- [ ] Step 3: Repeat the check for `CameraIngestor` (zed, ricoh), which binds the RT chain
  (`camera_ingestor.h:84`).
- [ ] Step 4: Commit.

## Phase 2: `sensor_calibration` agent skeleton

### Task 2.1: generate the agent
**Files:**
- Create: `sensor_calibration/sensor_calibration.cdsl`, `etc/config.toml`, `src/specificworker.{h,cpp}`,
  `CMakeLists.txt`.

Steps:
- Generate with `robocompdsl` from a cdsl copied from `bottle_concept`.
- `[Agent] id = 27`, name `sensor_calibration`.
- Presence: debounced Degraded, as in bottle_concept. Required peers: `robot_concept`, `room_concept`.
- Connect no DSR signals; poll instead.
- On start, create its own node `sensor_calibration` (type `"agent"`-style, mirroring bottle's own node).
- Verify: the agent starts and stops cleanly with SIGTERM 5/5, and the node is deleted on stop.
- Commit.

### Task 2.2: inputs
- Media subscribers via the factories:
  - helios and bpearl through `make_lidar_subscriber_from_graph`;
  - zed rgb+depth and ricoh through `make_image_subscriber_from_graph`;
  - imu.
- Graph reads:
  - odometry from the robot node (`robot_current_*_speed`);
  - pose and covariance from the room→robot RT edge (+ `rt_covariance`), the residual_concept precedent
    (`residual_concept/src/specificworker.cpp:400, 796`);
  - polygon from the room node `delimiting_polygon_x/y`.
- Test: a smoke run in Webots logs the stream rates once per 10 s; stall → local hold (CLAUDE.md).
- Commit.

### Task 2.3: proposal attributes (needs a cortex reinstall)
- Register in cortex `dsr_attr_name.h`:
  - `mount_proposal` (`std::vector<float>`): per sensor, 6 deltas + a 6×6 covariance;
  - `mount_proposal_version` (`int`);
  - `mount_proposal_ready` (`bool`).
  The user reinstalls cortex.
- Write them on the agent's own node, at ≤1 Hz, only when the posterior mean moves by more than 0.1σ of
  itself. This is a rate limiter on graph writes, not on the estimate; flag it as such.
- Test: the 1 Hz cap holds under a synthetic fast-changing posterior.
- Commit.

### Task 2.4: opportunistic harvester (no situation detector)
Every sweep or frame feeds every factor. The Jacobian plus the noise model decide what counts; there is no
"straight" / "turning" / "stopped" classifier. The pieces are:
- **Per-frame factors** (floor, verticality, static alignment, edges). Ego-motion and blur variance are added to
  R, so stills dominate naturally. PlaceCells collapse repeat views.
- **Trajectory factors** (Helios yaw vs chassis, Helios x/y, clock τ):
  - A ring buffer of ~30 s of keyframes: sweep, odometry + IMU accumulated since the previous keyframe, stamps.
  - Each new keyframe forms the pair, among the last K, of maximum information per DoF. Lever information of a
    piece is ∝ 4 sin²(dθ/2), so turns are never CHOPPED into small pieces (the thesis lost a hairpin's
    9.9 → 0.64 units with 0.20 rad cuts). A π turn as one pair and as two π/2 pairs carry equal signal, so the
    segmentation is whatever the noise model makes optimal (ruling 10-23; "one pair" was the wrong target).
  - Keyframes used in a pair are consumed, so nothing is counted twice.
  - Keyframe spacing is a compute budget. Test that the posterior does not change when it does.
- **Storage.** Per-DoF accumulated evidence, plus a small per-episode summary log for Phase 6 change detection.

Test: on a synthetic trajectory (straight 5 m, a π hairpin, a 30 s stop), check the following:
- yaw information comes ≥90 % from the straight;
- lever information comes ≥90 % from the hairpin, is ≥0.9 of the best achievable segmentation and ≥5× the 0.20-rad chopping, at 0.1 s spacing and odo_sigma_per_rad ∈ {0.001, 0.005};
- floor and static-alignment information comes mostly from the stop;
- halving or doubling the keyframe spacing changes each posterior mean by <0.1σ.

## Decisions (user, 2026-10-23)
- Task 4.4 **replaces** room_concept's `KinematicMount` yaw row; there is one owner of Helios yaw.
- Execution: subagent-driven.
- Noé's real runs: the user is receiving them and will provide path + format (Phase 0 blocked until then).

## Phase 3: move the localiser-free LiDAR factors

### Task 3.1: shared library
- Move `room_concept/src/mount_factors.h` (floor + vertical parts) and `lidar_mount.h` to
  `common/sensor_calib/`. Both agents include them from there.
- No behaviour change. room_concept's tests still pass.

### Task 3.2: run FloorPlaneMount (bpearl z/roll/pitch) and VerticalMount (helios roll/pitch, bpearl x/y/yaw) in the agent
- Inputs: sweeps, plus the pose snapshot and polygon from the graph, replacing `RoomConcept::mount_snapshot()`.
- Equivalence test: replay one Webots run with both copies logging. Estimates agree within 0.1σ.

### Task 3.3: remove them from room_concept
- Delete `mount_factor_step` from `LidarIngestor`, plus the `MountFactors` floor/vertical config keys and the
  joint-monitor parts that read them.
- `KinematicMount` and `LidarMountApply` stay.
- Disable the room_concept copy before the agent publishes (Review Focus 5).

## Phase 4: Noé's new factors (in the agent; each = synthetic test + Webots planted error + real bench row)

| Task | Factor | DoFs | Observation | Planted-error test (Webots) |
|---|---|---|---|---|
| 4.1 | ZED floor plane | zed z, roll, pitch | depth points near the floor, place nuisance per PlaceCell | pitch +1°, roll +0.6° recovered ≥90 % |
| 4.2 | ZED depth scale | s_z relative to the LiDAR range (G7) | retina's LiDAR-anchored depth correction (`retina/src/depth_dataset.cpp`) as a prior | +2 % scale recovered |
| 4.3 | ZED / Ricoh vs Helios, dense | zed yaw, x, y; ricoh full rotation | point-to-plane (zed), Helios depth edges vs panorama edges (ricoh); per-place nuisance against confident-wrong 6-DoF | ricoh yaw 1° recovered ≥90 % (thesis: 45 %) |
| 4.4 | Helios yaw vs chassis | helios yaw | scan-to-scan heading of travel vs integrated odometry + IMU; weighted by the Jacobian (no "straight" gate) | 1.84° planted, recovered within 0.1° |
| 4.5 | Camera clock | zed τ | image-derived yaw rate vs gyro ω (rotational channel, missing in the thesis) | τ = 30 ms planted |

- Order: 4.1 → 4.2 → 4.3, because ZED alignment needs depth scale first (Noé's ordering). 4.4 and 4.5 are
  independent.
- Before shipping 4.4, decide with the user whether it **replaces** room_concept's `KinematicMount` yaw row or
  runs beside it. Two owners of Helios yaw would violate the single-owner rule (`mount_factors.h`). The
  recommendation is to replace it, since 4.4 is localiser-free.

## Phase 5: retire room_concept's camera mount block

- Precondition: on the real bench, Tasks 4.1–4.3 match Noé's offline values within the cited σ.
- Delete `mount_calibrator.*`, `calib_channel*`, `camera_calibration.h`, `camera_jacobian.h`,
  `mount_lidar_pair.h`, the `ImageEdge.mount*` keys, and the camera parts of `calibration_viewer`.
- Keep the evidence files under `room_concept/etc/` in git history only.
- Verify:
  - room_concept builds;
  - a Webots tour has the same localisation IoU and pose error as before, within the run-to-run spread;
  - `wc -l room_concept/src/*` drops by about 5k.

## Phase 6: drift, change, commit

- **6.1 Drift.** Implement the thesis's two-level OU prior per DoF in `common/sensor_calib/`. The variance is
  driven by covariates (IMU shock events, time since `version`), not a constant. Test: a step change at t is
  tracked with a lag consistent with the OU time constant.
- **6.2 Change detection.** Compare "moved at t" vs "not moved" by evidence over the stored evidence log, reusing
  the SPRT-rival pattern from room_concept relocalisation. On "moved", drop the evidence before t. Test: a 1°
  bump planted mid-run is detected, and the estimate does not average old and new.
- **6.3 Commit protocol, in robot_concept.** Triggered by `mount_proposal_ready` when the robot is idle (velocity
  state from odometry and IMU). Sequence:
  1. Helios first: write the edge and bump `version`.
  2. Wait for room_concept to re-localise (pose covariance back to pre-commit level).
  3. Then the cameras and the Bpearl.
  4. Persist the file.

  The commit criterion is evidence for the new mount over the old one, not a fixed σ. Test: a commit never
  happens while moving; the cameras never commit before the Helios.

## Verification (end-to-end)

1. **Webots.** Plant Noé-sized errors (helios yaw 1.84°, tilt 0.46° / −0.30°, zed roll 0.62°, ricoh yaw 1°) via
   the existing `LidarMountInject*` keys and new camera inject keys. A tour recovers each within its stated σ,
   with 2σ coverage ≥ 85 %. One commit happens at the first idle, Helios before the cameras.
2. **Real bench (Phase 0 data).** Every DoF agrees with `expected.csv` within the combined σ.
3. **room_concept.** Localisation metrics are unchanged on the standard tour after Phases 3 and 5. Size drops
   about 5k lines.

## Status after Phases 1–2 (2026-10-23, branch feat/sensor-calibration)

Phases 1–2 (Tasks 1.1–1.3, 2.1–2.4) are DONE and reviewed. Nothing has been run on the live fleet (domain 0); all
tests ran on private DSR domains. Live check for the user: `sensor_calibration/bin/sensor_calibration etc/config.toml`
→ "subscriber up" for helios, bpearl, zed rgb/depth, ricoh, imu, then `[inputs] rates: … odom=yes pose=yes+cov
poly=Nv` with no `STALLED`.

**Rulings taken during execution** (each with its cost if wrong):
- Correction convention = BODY-aligned axes (R = Rx Ry Rz·R_nom, t = t_nom + t_corr) — cost: convert proposals.
- Legacy absolute `etc/mount_calib_<robot>.txt` is ignored with a warning — cost: none found in the repo.
- Graph-level overlay tests go through an EdgeIO seam, not a live graph — cost: the real DSR glue is unexercised live.
- Hairpin requirement = optimal segmentation (not "one pair"); criterion ≥0.9× the exhaustive optimum.
- Heading = precision-weighted gyro+wheel fusion; the chord is integrated along the fused heading.
- Keyframe spacing budget upper bound = 0.5 s.
- The proposal publisher is not wired (no mount posterior until 4.4).
- Live camera refresh is disabled while room_concept `ImageEdge.mountPublish` is on (prevents double correction).

**Carry-forwards:**
- **4.4 Helios yaw vs chassis:**
  - Gyro scale, bias and tilt are common-mode but are treated as independent per pair, so the posterior
    over-tightens. Model them as nuisance states, or use Woodbury common-mode marginalisation.
  - Add `Pair::chord_sigma²` and `dth_sigma` into the row variance.
  - Pin the harvester's side/rot sign convention against KinematicMount.
  - Verify that the Helios stamp and odometry share one clock.
- **Phase 3:** pin the IMU up axis (`gyro.z()` is assumed) to the IMU mount.
- **Phase 4 RGB-D factors:** zed rgb and depth are decimated independently, so they need a pairing rule (nearest
  stamp or a shared decimator).
- **Phase 4 wiring of the proposal publisher:**
  - `offer()` must commit `last_` and `version` only after a successful graph write.
  - A covariance-only change never triggers a write; consider a log-det-change trigger.
  - Correlations are ignored by the per-DoF movement test.
- **Phase 6 commit:**
  - A committed edge that already contains a learned camera correction must rebase `mount_corr_`, which today is
    reconciled only at load (`calib_channels.cpp:70`). The restart-with-mountPublish path is not traced.
  - `LidarIngestor::resolve_sensor_origins` latches the sensor origins once, so committed translations do not
    reach the mount factors.
  - The `mountPublish→set_live_graph_refresh` wiring is only tested via the setter.
- **Minor:**
  - The input layer's `pose_` and `polygon_` are not cleared on room change.
  - The factory "no descriptor" log repeats every 5 s.
  - A stall hold extrapolates the last rate (gated only by the input layer).
  - The stale sweep in `sensor_calibration` uses a placeholder `metaconcept` type and a name-prefix match.
  - The selftests are EXCLUDE_FROM_ALL and not registered with ctest.

## Status after Phase 3 (2026-10-29, branch feat/sensor-calib-phase3)

Tasks 3.1–3.3 DONE (no live run yet).
- **3.1** `common/sensor_calib/mount_factors.h` (Info3/Info6, PlaceCells, place_key, FloorPlaneMount, VerticalMount) and
  `common/sensor_calib/lidar_mount.h` are shared. `KinematicMount` split out to `room_concept/src/kinematic_mount.h`.
- **3.2** `sensor_calibration/src/lidar_mount_stage.*` runs the floor + vertical factors, `[MountFactors]` config,
  logs `tmp/mount_factors/{mounts,bpearl_sweeps}_<ts>.csv` + a `[mounts]` line every 10 s. Inputs now carry the
  robot (Shadow) frame mount, the pose AT the sweep stamp (`TimeQuery::Extrapolated`; Stale/Clamped = no pose) and
  the room's `room_height`. Test: `mount_factors_test` (parts 2/3/P byte-identical to the pre-move oracle, + part S).
- **3.3** Removed from room_concept: `mount_factor_step`, `walls_at`, the mount snapshot, `BpearlMountInject*`,
  `BpearlMountRate`, `MountPoseSigma*`, the joint monitor (`joint_calibration.h`, `joint_calib_monitor.h`,
  `JointCalibMonitor`, `UpdateResult::{calib_information, mount_kin_info, mount_yaw_offset}`), their selftests and
  `tools/joint_calib_report.py`. `RoomConcept.MountFactors` STAYS: it gates the kinematic factor AND the single-owner
  rule (`lidar_side_yaw`, eps apply bit), so it is not log-only.

**Rulings:**
- Equivalence (Review Focus 5) is satisfied by construction: the room_concept copy is deleted, not disabled, and the
  estimators are byte-identical on the synthetic oracle. A replayed Webots A/B (plan 3.2 "within 0.1σ") was NOT run;
  the inputs legitimately differ (pose from the RT ring vs room's velocity carry; published polygon vs
  nominal_room_polygon; room_height only when measured).
- Planted errors for these factors go through robot_concept's overlay file, not per-agent inject keys.
- The joint fusion (helios tilt ↔ bpearl planar coupling, cameras) is not ported; the agent reports the three
  blocks separately.

**Carry-forwards:** `calibration_estimator.h`'s `last_information()` and the camera pools' `marginal_information()`
/ `lidar_yaw_ref()` lost their only consumer (the monitor); clean them up in Phase 5. Room_concept's `kin_csv_` is
now the only output of `KinematicMount` (until 4.4 replaces it).
