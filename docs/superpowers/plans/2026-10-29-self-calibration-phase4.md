# self_calibration Phase 4, slice 1: Helios mount from scan registration (4.4) + ZED floor (4.1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The `self_calibration` agent (id 27, ex `sensor_calibration`) estimates the Helios planar mount (x, y, yaw) from scan-to-scan registration over the harvester's keyframe pairs, and the ZED (roll, pitch, z) from the floor in its depth image. Estimate + log only.

**Architecture:** Two new stages beside `LidarMountStage`, fed from the existing input layer in `compute()`.
- **Helios stage:** keeps the wall band (2-D) of each harvester keyframe, registers the two sweeps of every committed `Pair` with a 2-D point-to-line ICP (new, `common/sensor_calib`), turns the registration into the pair's innovation `y = (y_fwd, y_lat)` and adds it to `MountEvidence` together with a Schur-marginalised gyro common mode.
- **ZED stage:** back-projects the depth image into the robot frame and feeds the existing `FloorPlaneMount`.

**Tech Stack:** C++23, Eigen, nanoflann (installed, header-only), CORTEX `CameraAPI`, media plane via `SensorInputs`.

**Spec:**
- `docs/superpowers/plans/2026-10-23-sensor-calibration-agent.md`: Phase 4 table rows 4.1 and 4.4, and the "Carry-forwards" under "Status after Phases 1–2".
- Claude Doc "Opportunistic Self-Calibration of the Shadow Sensor Head" (https://claude.ai/artifact/TC7U92HFEpHQUhuDbb9nsX): "Improvements" items 3 (ZED floor-plane factor = reuse FloorPlaneMount on ZED depth) and 10 (no situation table; the Jacobian decides).
- Noé's real values (ONLY for the real bench, never against Webots): helios yaw 1.84°.

**Deferred to the next slice (each needs its own design pass):** 4.2 ZED depth scale (retina `depth_dataset.cpp` prior), 4.3 dense ZED/Ricoh alignment, 4.5 camera clock, every "real bench row" (blocked on Phase 0 data).

## Global Constraints

- `CLAUDE.md` applies.
  - No thresholds or situation gates: no "straight", "parked" or "speed > k" tests. Weight by covariance; common modes by Schur/Woodbury.
  - Type-attributed DSR attribute access only. No `Qt::DirectConnection`.
  - `std::from_chars` for parsing; every CSV goes through `imbue(std::locale::classic())`.
- Build with the `cbuild` caps at `-j8`; check `pgrep cc1plus` first. Stop agents with SIGTERM.
- Pure modules (no Qt/DSR/DDS) live in `common/sensor_calib/` or `self_calibration/src/*_stage.*`, and are tested from `self_calibration/tests/`.
- The DSR and media plumbing stays in `sensor_inputs.*` and `specificworker.cpp`.
- Correction convention (plan ruling, Phases 1–2): every estimate is the correction that RESTORES truth. A plant made through the overlay therefore reads back as its negative (`mount_factors_test` part O).
- Planted errors go ONLY through `robot_concept/etc/mount_calib_Shadow_webots.txt`. No per-agent inject keys.
- Logs only. The proposal publisher is not wired, and nothing writes a mount.

## Review Focus

1. **The ICP must not invent motion in a corridor.** With one wall direction, the along-corridor covariance must be wide, and the row's information along it ~0, so no fake mount comes out of it. Test: Task 1, `degenerate_corridor`.
2. **Sign and axes between three frames:**
   - robot frame: +Y forward;
   - harvester frame: +x forward, +y left (`advance()`: `x += adv·cos`);
   - `Pair::J` rows: (fwd, lat).
   One wrong sign silently doubles the error instead of cancelling it. Test: Task 2, `planted_yaw_sign`.
3. **The gyro is common to every pair.** Its scale and bias must enter as a shared nuisance, not as per-pair noise, or ψ over-tightens (carry-forward). Test: Task 2, `gyro_bias_not_absorbed`.
4. **ZED depth convention:** depth = the FORWARD coordinate (Y in the zed node frame), not the range along the ray. The depth image may differ in resolution from the RGB intrinsics. Test: Task 3, `depth_backprojection`.
5. **Stalled or missing streams.** A keyframe without a stored sweep, or a pair whose registration did not converge, adds nothing and is logged, never a zero row. Test: Task 2, `missing_sweep_adds_nothing`.

---

### Task 1: 2-D point-to-line scan registration

**Files:**
- Create: `common/sensor_calib/scan_registration.h`, `common/sensor_calib/scan_registration.cpp`
- Modify: `common/CMakeLists.txt` (add the .cpp to `ai_common_sensor_calib`)
- Test: `self_calibration/tests/scan_registration_test.cpp` + target in `self_calibration/generated/CMakeLists.txt` (same pattern as `trajectory_harvester_test`)

**Interfaces:**
- Produces (namespace `sc::calib`):
```cpp
struct ScanRegConfig {
    int    max_iters   = 30;
    int    normal_k    = 8;        // neighbours for the PCA normal of each reference point
    double sigma_m     = 0.01;     // inlier sigma of a point-to-line residual [m]
    double outlier_window_m = 0.5; // the uniform the outlier component spreads over (search window, not a gate)
    double pi_inlier0  = 0.8;      // prior inlier share, re-estimated by EM
};
struct ScanRegResult {
    bool ok = false;
    Eigen::Vector3d xyt = Eigen::Vector3d::Zero();    // pose of CUR in REF: p_ref = R(t) p_cur + (x, y)
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();// sigma_hat^2 * H^-1 at the solution
    int n_used = 0; double inlier_share = 0.0, rms = 0.0; int iters = 0;
};
[[nodiscard]] ScanRegResult register_scans_2d(const std::vector<Eigen::Vector2f>& ref,
                                              const std::vector<Eigen::Vector2f>& cur,
                                              const Eigen::Vector3d& init_xyt,
                                              const ScanRegConfig& cfg = {});
```
- Approach:
  - Gauss-Newton on the point-to-line residual `n_i·(R p_j + t − q_i)`, nearest neighbour by nanoflann (2-D kd-tree on `ref`).
  - Weights are EM responsibilities of an inlier Gaussian (`sigma_m`) against a uniform over `outlier_window_m`, so there is no distance gate.
  - `cov` = `(Σ w r² / Σ w)` · (Σ w J Jᵀ)⁻¹ in the end.
  - `ok` = converged (|Δ| < 1e-6) and H positive definite.

- [ ] **Step 1: Write the failing tests** in `scan_registration_test.cpp` (synthetic: a polyline room 4×6 m with a notch, rays every 0.25° from an origin, 1 cm Gaussian noise, seed 7):
  - `known_motion`: cur taken after moving (0.30, 0.05, 2°), init = zero ⇒ |x err|, |y err| < 5 mm, |t err| < 0.05°, `ok`.
  - `init_from_odometry`: true (1.2, −0.1, 15°), init off by (5 cm, 5 cm, 3°) ⇒ same tolerances.
  - `outliers_do_not_bias`: 15 % of cur points replaced by uniform clutter inside the room ⇒ same tolerances, `inlier_share` in [0.75, 0.95].
  - `degenerate_corridor`: two parallel walls only (corridor along x), true motion (0.5, 0, 0) ⇒ `sqrt(cov(0,0)) > 10 * sqrt(cov(1,1))` (along-corridor uninformed), and |y err| < 5 mm.
  - `honest_sigma`: 200 noise seeds of `known_motion` ⇒ the empirical sd of the yaw error is within [0.7, 1.5] × the mean reported `sqrt(cov(2,2))`.
- [ ] **Step 2: Run, verify they fail** — `make -C build scan_registration_test` fails (no header).
- [ ] **Step 3: Implement `register_scans_2d`** as above.
- [ ] **Step 4: Run** — `../bin/scan_registration_test`, expected `ALL PASS`.
- [ ] **Step 5: Commit** — `self_calibration: 2-D point-to-line scan registration (EM-weighted, honest covariance)`.

### Task 2: Helios planar mount from registered keyframe pairs (4.4)

**Files:**
- Create: `self_calibration/src/helios_mount_stage.h`, `self_calibration/src/helios_mount_stage.cpp`
- Modify: `self_calibration/src/specificworker.{h,cpp}` (feed + log), `self_calibration/src/CMakeLists.txt`, `self_calibration/etc/config.toml` (`[HeliosMount]`)
- Test: `self_calibration/tests/helios_mount_test.cpp` + target

**Interfaces:**
- Consumes:
  - `register_scans_2d` (Task 1).
  - `sc::calib::Pair`, `make_pair_from_motion`, `MountEvidence`, `HarvesterConfig` (trajectory_harvester.h).
  - `sc::LidarIn::points_robot()`.
- Produces:
```cpp
namespace sc {
struct HeliosMountConfig {
    double band_min_m = 1.5, band_max_m = 2.0;   // room_concept's Media.lidar_high_{min,max}_height: the 2-D wall band
    double gyro_bias_sigma = 0.002;              // rad/s prior on the common gyro bias (nuisance)
    double gyro_scale_sigma = 0.01;              // fractional prior on the common gyro scale (nuisance)
    std::string log_dir = "tmp/helios_mount";    // pairs_<ts>.csv + mount_<ts>.csv; "" = none
    sc::calib::ScanRegConfig reg{};
};
// The pair innovation in Pair::J's rows from a registration (cur = b, ref = a, both robot frame):
// y = (registered chord on the pair's fused midpoint heading) - (d_fwd, d_lat), harvester axes (+x fwd, +y left).
[[nodiscard]] Eigen::Vector2d pair_innovation(const sc::calib::Pair& p, const sc::calib::ScanRegResult& r);
class HeliosMountStage {
public:
    explicit HeliosMountStage(HeliosMountConfig cfg, sc::calib::HarvesterConfig hcfg);
    void store_keyframe(std::int64_t stamp_ms, const std::vector<Eigen::Vector3f>& points_robot); // band -> 2-D
    bool add_pair(const sc::calib::Pair& p);     // false = a sweep missing or registration not ok (logged, nothing added)
    [[nodiscard]] Eigen::Vector3d mean() const;  // (dx, dy, psi) helios correction, nuisances marginalised
    [[nodiscard]] Eigen::Vector3d sigma() const;
    [[nodiscard]] long pairs_used() const;
    [[nodiscard]] std::string summary() const;   // "[helios] x ±  y ± mm  yaw ± deg (n pairs)"
    void log_state(std::int64_t stamp_ms);
};
}
```
- Model:
  - Per pair, the row variance = `p.var` + the registration covariance mapped through the projection (replaces `reg_sigma_m`) + `chord_sigma²` on the lat row.
  - The **common mode**: a heading error δθ = b·dt + k·dth rotates the odometry chord, adding `−½·d_fwd·δθ` to y_lat. Keep the extended normal equations over (dx, dy, psi, b, k) and report the Schur complement over (b, k) with priors `gyro_*_sigma`. This is the `KinematicMount` pattern; do not use `MountEvidence` for it.
  - Keyframe sweeps are dropped once no pair can use them, i.e. older than the harvester's `ring_span_s` + 1 s.

- [ ] **Step 1: Write the failing tests** (synthetic trajectories through the Task 1 room, LiDAR sweeps generated in the TRUE sensor frame and expressed in the robot frame through the NOMINAL mount, odometry from the true motion):
  - `planted_yaw_sign`: helios mount error ψ = +1.84° (lever 0), 40 m of mixed driving ⇒ `mean()[2] = −1.84° ± 2σ` (it restores truth), σ < 0.1°.
  - `lever_from_pivots`: lever (30, −20) mm, ψ = 0, pivots only ⇒ (dx, dy) recovered within 2σ, ψ σ ≥ 0.9 × prior (uninformed).
  - `yaw_from_straights`: straights only ⇒ ψ informed (σ < 0.2°), dx/dy σ ≥ 0.9 × prior.
  - `gyro_bias_not_absorbed`: ψ = 0, gyro bias 0.003 rad/s, one-direction arcs ⇒ |ψ| < 2σ (the control run with the nuisance removed must be confidently wrong, |ψ| > 3σ).
  - `missing_sweep_adds_nothing`: a pair whose keyframe was never stored ⇒ `add_pair` returns false, `pairs_used()` unchanged.
- [ ] **Step 2: Run, verify they fail.**
- [ ] **Step 3: Implement `HeliosMountStage` and `pair_innovation`.**
- [ ] **Step 4: Wire it in `SpecificWorker::compute()`:**
  - on `harvester_->add_keyframe(...) == true`, call `store_keyframe(l.stamp_ms, l.points_robot())` for that helios `LidarIn`;
  - after `take_pairs()`, call `add_pair` for each pair;
  - log `summary()` + `log_state` every `[HeliosMount] log_period_ms` (default 10000).
  - Config keys `[HeliosMount] enabled, band_min_m, band_max_m, gyro_bias_sigma, gyro_scale_sigma, log_period_ms`, read through `rc::cfg::Reader` like `[MountFactors]`.
- [ ] **Step 5: Clock check (carry-forward "Helios stamp and odometry share one clock").** Once per run, log the median and the spread of (keyframe `stamp_ms` − nearest odometry `source_ts_ms`) over the first 50 keyframes. The `[helios] clock` line prints both. A median beyond one odometry period is reported, never corrected here (4.5 owns clocks).
- [ ] **Step 6: Run** the 5 tests + `mount_factors_test` + `trajectory_harvester_test`, all PASS; build the agent at `-j8`.
- [ ] **Step 7: Commit** — `self_calibration 4.4: helios planar mount from registered keyframe pairs (gyro common mode marginalised)`.

### Task 3: ZED floor plane (4.1)

**Files:**
- Create: `self_calibration/src/zed_floor_stage.h`, `self_calibration/src/zed_floor_stage.cpp`
- Modify:
  - `self_calibration/src/sensor_inputs.{h,cpp}`: `ImageIn` gains `robot_from_sensor` (as `LidarIn`) and the pinhole intrinsics `fx, fy` read once per sensor through `G->get_camera_api(node)`; cx, cy = size/2, as CameraAPI sets them.
  - `specificworker.{h,cpp}`, `src/CMakeLists.txt`, `etc/config.toml` (`[ZedFloor]`).
- Test: `self_calibration/tests/zed_floor_test.cpp` + target

**Interfaces:**
- Consumes: `rc::mountf::FloorPlaneMount`, `place_key` (common/sensor_calib/mount_factors.h); `sc::ImageIn`.
- Produces:
```cpp
namespace sc {
struct PinholeIntrinsics { double fx = 0, fy = 0, width = 0, height = 0; };   // of the RGB model; depth may be scaled
// Depth image (FORMAT_DEPTH_F32 metres, or FORMAT_Z16 millimetres) -> points in the ZED node frame
// (X right, Y forward = the depth value, Z up), every `stride`-th pixel, non-finite / <= 0 skipped.
[[nodiscard]] std::vector<Eigen::Vector3f> depth_to_points(const ImageIn& depth, const PinholeIntrinsics& K, int stride);
class ZedFloorStage {
public:
    struct Config { double rate_hz = 1.0; int stride = 4; std::string log_dir = "tmp/zed_floor"; };
    explicit ZedFloorStage(Config cfg);
    bool observe(const ImageIn& depth, const PinholeIntrinsics& K,
                 const std::optional<Eigen::Affine3d>& room_T_robot);   // robot_from_sensor must be set
    [[nodiscard]] rc::mountf::Info3 info() const;    // (droll, dpitch, dz) zed correction, robot axes
    [[nodiscard]] std::string summary() const;       // "[zed-floor] roll ± pitch ± deg  z ± mm (n frames)"
    void log_state(std::int64_t stamp_ms);
};
}
```
- `dz` is NOT the mount height alone until 4.2: a depth-scale error s moves the floor to s·h, so `dz` reads (height error) + (s−1)·h. Say so in the header and in `summary()` ("z includes depth scale"). Roll and pitch are unaffected.

- [ ] **Step 1: Write the failing tests:**
  - `depth_backprojection`:
    - a 640×360 F32 image of a plane Y = 2 m at fx = fy = 300 ⇒ every point has Y = 2.000 and X = (u−320)·2/300;
    - the same image in Z16 (2000 mm) gives the same points;
    - a 1280×720 depth image with 640×360 intrinsics is scaled (fx·2).
  - `planted_floor`: render the floor of the Task 1 room into a ZED depth image (mount 1.08 m, pitched down 25°) with a TRUE mount error of roll +0.6°, pitch +1.0°, z +10 mm, frames from 30 poses ⇒ `info().mean()` = (−0.6°, −1.0°, −10 mm) ± 2σ.
  - `walls_do_not_bias`: the same with walls and a 0.7 m table in view ⇒ same tolerances (FloorPlaneMount's EM share does the rejection).
- [ ] **Step 2: Run, verify they fail.**
- [ ] **Step 3: Implement** `depth_to_points`, `ZedFloorStage` (thin: points → robot frame → `FloorPlaneMount::observe_sweep(pts, t_zed, place)`), the `ImageIn` additions and the wiring.
  - Only `sensor == "zed" and stream == "depth"` images are consumed.
  - Skip a frame whose `robot_from_sensor` is unresolved.
- [ ] **Step 4: Run** the three tests + all earlier ones; build the agent at `-j8`.
- [ ] **Step 5: Commit** — `self_calibration 4.1: ZED floor plane (roll, pitch, z incl. depth scale) from the depth image`.

### Task 4: Webots planted-error check (both factors), recorded

**Files:**
- Create: `self_calibration/tools/planted_check.sh` (runs one plant, collects the agent's CSVs)
- Modify: `docs/superpowers/plans/2026-10-23-sensor-calibration-agent.md` (status block with the numbers)

- [ ] **Step 1:** The plant file `robot_concept/etc/mount_calib_Shadow_webots.plant-phase4.txt` = the current `mount_calib_Shadow_webots.txt` (keep its ricoh row) plus:
  - `helios 0 0 0 0 0 0.0321141 …` (yaw +1.84°);
  - `zed 0 0 0.010 0.0104720 0.0174533 0 …` (z +10 mm, roll +0.6°, pitch +1.0°).
- [ ] **Step 2:** `planted_check.sh`:
  - saves the clean overlay, installs the plant, restarts robot_concept first, then the fleet + `self_calibration`;
  - drives a standard tour with `room_concept/estimation_scripts/scripts/run_crossing.sh`;
  - stops everything with SIGTERM, restores the clean overlay (also on Ctrl-C, via `trap`);
  - copies `tmp/helios_mount/` and `tmp/zed_floor/` into `self_calibration/bench/webots_planted_<date>/`.
- [ ] **Step 3: Run it.** Expected in the last `mount_<ts>.csv` row:
  - helios yaw = −1.84° ± 2σ with σ < 0.1°;
  - zed roll = −0.6°, pitch = −1.0° ± 2σ;
  - zed z = −10 mm + (depth-scale term).
  If a check fails, record the numbers anyway. That is the result.
- [ ] **Step 4:** A null run (clean overlay, same tour) ⇒ every estimate within 2σ of 0 (helios) / of its null value (zed z).
- [ ] **Step 5: Commit** — the bench folder + status block: `self_calibration Phase 4 slice 1: Webots planted check (helios yaw, zed floor)`.
