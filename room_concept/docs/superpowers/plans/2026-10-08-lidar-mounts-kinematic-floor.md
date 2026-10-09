# LiDAR + Camera Mounts, Decoupled: Kinematic Factor, Floor, Verticals (joint calibration r2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the helios and bpearl 6-DoF mounts and the camera mounts against the robot BODY with the odometry treated as already calibrated (fixed), using three exteroceptive/kinematic factors: (1) a per-cycle KINEMATIC factor between consecutive scan-only poses and the calibrated odometry (helios x, y, yaw), (2) the FLOOR plane seen head-on by the bpearl (bpearl z, roll, pitch — absolute), (3) VERTICAL structure — walls are vertical, the ceiling is horizontal — seen by both LiDARs (helios roll, pitch; bpearl x, y, yaw relative to helios); and fuse them with the existing camera blocks.

**Why (Task 6 of the r1 plan, 2026-10-08):** the joint solve separated camera from LiDAR error correctly (planted 1° helios yaw: both cameras' relative yaw moved −0.98/−1.18°, their body-frame yaws stayed at baseline), but the helios mount itself came only from the motion calibrator, which learns from CORRECTIONS — and an error below the early-exit gate (~7.5 cm) never produces one: a planted (+30, −20) mm lever read (+7 ± 10, +2 ± 10) mm after 92 episodes. The same session's per-cycle scan-to-scan innovations recovered it: (+24.1 ± 3.1, −25.1 ± 3.0) mm and yaw 0.89 ± 0.14° (plant 1.0°). Exteroceptive sensors alone fix only RELATIVE mounts (gauge G3: a rigid motion of all mounts together is invisible); one body-anchored factor is needed. The user's decision: anchor through the kinematic factor ALONE with the odometry calibrated and stable (decoupled — the motion calibrator's mount columns are no longer the source), and add the bpearl with the floor and vertical measures it implies.

**Architecture:** a new pure header `src/mount_factors.h` with three accumulators (each keeps normal equations in PHYSICAL units, prior separate, exponential forgetting, per-sweep common-mode nuisance marginalised by Schur complement — never a σ-floor), fed from room_concept's existing streams: the innovation stream (`observe_innovation`, `innov_z_`, `cyc_odom_`, the noise learner's per-axis variance) for the kinematic factor; a permanent, THROTTLED bpearl reader plus the helios sweep for the floor/vertical factors. `src/joint_calibration.h` grows from (motion 9 + cameras 4·N) to (helios 6 + bpearl 6 + cameras 4·N), with the camera coupling unchanged (through the helios yaw). The r1 motion-block mount columns (eps_yaw, lever) are no longer fed to the joint (the motion calibrator keeps running for the odometry parameters, which this plan treats as fixed). Monitor-only first; application stays behind `LidarMountApply` (OFF) and the existing planted-error injection extends to 6-DoF for both LiDARs.

**Tech Stack:** C++23, Eigen 3, header-only accumulators with standalone selftests (`g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/<name>.cpp -o /tmp/<name>`), Qt/DSR agent room_concept, Webots with GT.

**Spec:** problem-definition artifact "What can Shadow learn about its own body?" (§3 gauges G1–G7, §8 verdict: "helios z, roll, pitch — vertical walls give roll and pitch; z goes through bpearl", "bpearl z, roll, pitch — floor plane seen head-on; defines z = 0 (G4)"); r1 plan `docs/superpowers/plans/2026-10-05-joint-calibration.md`; ROBOT_GEOMETRY.md (frames, the inverted helios and bpearl).

## r2.2 CORRECTIONS (Fable review 2026-10-08, `docs/fable_r2_review_2026-10-08.md`) — THESE SUPERSEDE THE TASK TEXT BELOW WHERE THEY CONFLICT

1. **Task 1 rows (midpoint frame, as the code forms δ):** δ is projected at θ_mid = z_th − ½·odom_th, so R(θ_mid)ᵀ[R(θ_n) − R(θ_{n−1})]Δ = 2 sin(dθ/2)·JΔ. Rows: **δ_lat = −2 sin(dθ/2)·Δy + ψ·d_fwd, δ_fwd = +2 sin(dθ/2)·Δx**, no (1−cos) columns, no mount term on δ_θ. ψ = the helios yaw as the localiser ADDS it to the heading (θ_est = θ_true + ψ); sign PINNED by the logs (planted +1° → +0.81/+0.92°, null +0.03°). Axis order from `T`: index 0 = FORWARD, 1 = LATERAL (`odom_var(0)`/`last_m(0)` is forward). Row variance = `noise_innov_.last_m(a)`, weight (1 − `last_r(a)`), every row weight 1 (NOT the learner's ma1 0.5); document σ as conservative (MA(1) of the scan white telescopes). Feed only when `fed`.
2. **Camera coupling with ψ carried directly:** eps_yaw = −ψ (kEps = −1), so the camera yaw couples to the ψ column with A = kCamPerLidarYaw = −1 DIRECTLY — do not divide by kEps again. Name the column ψ_h and write both constants beside it.
3. **Task 1 tests:** midpoint generator; add case E — arcs of ONE turning direction (v/ω = 1 m) with a 1 % forward-scale error in the odometry: the lever must not come out confidently wrong (needs 4).
4. **Task 1b is a COMMON-MODE marginalisation, not a per-row variance.** An odometry-parameter error is the same on every row (per-row EIV adds (0.4 mm)² to (4–6 mm)² rows: inert). Store N = Σ w x xᵀ, C = Σ w x J_oᵀ, D = Σ w J_o J_oᵀ and report H_mount = N − C (D + Σ_o⁻¹)⁻¹ Cᵀ (b likewise) — Schur, still DECOUPLED (θ_o never re-estimated). J_o: k_v on δ_fwd (−d_fwd); k_lat on δ_lat (≡ 0 on a differential base, keep the column); heading parameters are second order — omit. Σ_o = calibrator posterior + Σ_drift (exponentially weighted outer product of the APPLIED θ_o change per re-solve). Re-reference (old item 4) from the same C: on an applied change Δθ_o, b += C·Δθ_o. Single owner (old item 3) is MANDATORY: set `lidar_side_yaw = true` when `MountFactors` is on so eps_yaw does not act on the odometry, keep eps out of θ_o, and LOG `yaw_offset()` per cycle (Task 6 must prove it is 0).
5. **Task 2 frames:** points are in the SHADOW frame (floor z = 0); `t_nom` must be in the same frame: bpearl (0, 0.14, 0.7025), helios (0, −0.155, 1.1075) (= body + 0.0325). Header note on write-back: the sensor-frame increment of a body-frame rotation about the INVERTED sensor (R = Ry(π)) is (−ω_x, ω_y, −ω_z) — roll and yaw flip — so a future write-back to shadow.json must COMPOSE, not add. Uniform non-floor component's support = the sweep's observed z range. Hardware caveat in the header: per-sweep (droll_s, dpitch_s) nuisances needed on a real base (pitches under acceleration); Webots body is rigid. Test pin: the per-sweep dz_s nuisance does NOT absorb roll/pitch.
6. **Task 3: NO helios planar rows** (the map was fitted to the helios scan: nᵀp − d ≈ 0 by construction, and with a tight per-sweep prior the Schur would leave a false "dx = 0" claim out-voting the kinematic factor 10⁴:1). `VerticalMount` gives helios (droll, dpitch) only (walls two-sided q_z −0.55…+1.5 m + ceiling with h_c nuisance); bpearl verticals give PLANAR (dx, dy, dyaw relative to the helios) only — its wall span is one-sided, the floor carries its tilt. Per-sweep pose nuisance prior is WIDE (pose-field σ: 0.05 m, 2°, or the published `res.covariance`), never the solver's marginal; the bpearl sweep uses the pose at ITS stamp. Add case E: helios with a planted 30 mm lever and a correct map → vertical factor's helios planar information = 0 (σ = prior).
7. **Task 5: helios-tilt → bpearl-planar coupling** in `solve_mounts`: the map walls sit where the helios tilt put them at band height, Δz ≈ z_band − z_helios ≈ 0.65 m, so bpearl_rel_dx −= ω_hy·Δz, bpearl_rel_dy += ω_hx·Δz (sign pinned by a selftest: plant helios tilt, bpearl planar stays 0). Pin also: feeding the r1 motion-block eps/lever into `solve_mounts` must not change the answer. Helios dz reported as "prior" by the report.
8. **Before Task 6 grades the lever: explain the Δx shortfall** (19–27 mm recovered for a 30 mm plant; Δy and ψ fine; not the k_v confound — corr ≈ 0). Offline on `tmp/noise_innov/innov_2026-10-08_12-18-41.csv` and `_10-56-32`: fit CW and CCW pivots separately (a lever gives the same Δx for both; a timing/deskew effect flips sign), check the `mv·p_move·dt` term's down-weighting of pivots, and whether a residual eps/lever acted (yaw_offset). Report the cause before any lever arm is graded.
9. **Task 6:** `LidarMountInject*` are reset to 0 (done 2026-10-08 in etc/config.toml); prove `yaw_offset()` = 0 in each arm; reset list per arm = `etc/motion_calib_state.csv`, `etc/camera_calib_Shadow_*.txt`, and the mount-factor evidence; grade signed with r1's three-part rule, and for the floor factor also report the ABSOLUTE error in mm/° (its σ is tiny, so 2σ tests systematics).

## Global Constraints

- Project rules in `active_inference/CLAUDE.md` (no thresholds where a model term works; `and/or/not`; `std::from_chars` + `imbue(classic)`; never `Qt::DirectConnection`; SIGTERM never `kill -9`; `cbuild -j8`, check `pgrep -a cc1plus` first; no `-march=native`).
- Gauges: **G3** body = wheel frame (axle midpoint, +Y forward, X lateral, z = 0 at floor contact); **G4** the floor is z = 0 in `root`, `body` sits at +0.0325 m (fixed, not estimated); **G2** LiDAR range scale ≡ 1.
- **Odometry is FIXED** in this plan: its parameters are whatever the motion calibrator currently applies; nothing here re-estimates them. The kinematic factor's per-cycle variance comes from the innovation noise learner (`MotionNoiseInnov::odom_var`, `s`, outlier responsibility), as the pose-field estimator already uses it.
- Mount errors are SMALL corrections to the `shadow.json` nominal: a 6-vector (dx, dy, dz, droll, dpitch, dyaw) in the BODY frame, rotation about the SENSOR origin. One convention everywhere (r1's mount-pose M: points are restored by M, injected by M⁻¹).
- Monitor-only default; `LidarMountApply` stays OFF. Every new estimator logs; nothing moves the published pose.
- No git commits by implementing agents (the tree has unrelated uncommitted work); the human/main session commits.

## Review Focus

1. **Common-mode leakage into tilt.** A per-sweep pose error (the localiser's x, y, yaw for that sweep) shifts every wall point the same way; tilt shifts them in proportion to HEIGHT. If the per-sweep planar nuisance is not marginalised, pose error is read as tilt. Test: Task 3 case C (a pose error per sweep, zero tilt → tilt ≈ 0).
2. **Floor-point association** is the one place a band is tempting (|z| < k). Model it instead as a two-component assignment (floor plane vs not-floor) with the plane's own residual scale — responsibility-weighted rows, like the innovation learner's mixture. Test: Task 2 case B (furniture/legs 20 % of near points → floor tilt still recovered).
3. **The kinematic factor during pivots** carries the lever; straight legs carry yaw. A run with one kind only must report the other UNINFORMED at its prior (Task 1 case B), not a confident zero.
4. **Helios z** is weakly observed (floor first seen at 3.88 m, grazing). It must stay near its prior with an honest σ, never be "found" from noise (Task 3 case D).
5. **Sign/frame of the 6-DoF injection**: a correct estimate must cancel a planted M exactly (Task 4 selftest), or the live arms grade the wrong number.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/mount_factors.h` | **new**, pure: `KinematicMount` (helios x,y,yaw from the innovation stream), `FloorPlaneMount` (bpearl z,roll,pitch), `VerticalMount` (roll,pitch from walls+ceiling for one LiDAR; x,y,yaw relative to the wall map) | create |
| `src/joint_calibration.h` | fuse helios 6 + bpearl 6 + cameras 4·N | extend (new `solve_mounts`), keep r1 `solve` for comparison |
| `src/lidar_mount.h` | 6-DoF mount transform + injection (`Pose6`) beside `Planar` | extend |
| `src/lidar_ingestor.{h,cpp}` | permanent throttled bpearl reader; hand floor/wall points of both LiDARs to the estimators; 6-DoF correction/injection per LiDAR | modify |
| `src/room_concept.{h,cpp}` | feed `KinematicMount` from `observe_innovation`; expose the wall map segments + current pose to the vertical factor | modify |
| `src/specificworker.cpp`, `src/joint_calib_monitor.h` | monitor row: helios 6, bpearl 6, cameras; per-factor information shares | modify |
| `src/room_config.*`, `etc/config.toml` | `MountFactors = true` (log), `BpearlMountRate = 2` Hz, priors, 6-DoF `LidarMountInject*` for both LiDARs | modify |
| `tools/mount_factors_selftest.cpp`, `tools/joint_mounts_selftest.cpp`, `tools/mount_factors_replay.cpp`, `tools/joint_calib_report.py` | tests, offline replay on logged sweeps, report | create/extend |

---

### Task 1: `KinematicMount` — helios x, y, yaw from consecutive scan-only poses and the calibrated odometry

The validated offline fit (2026-10-08, planted (30, −20) mm / 1°: recovered (24.1 ± 3.1, −25.1 ± 3.0) mm, 0.89 ± 0.14°) made a per-cycle model: in the body frame (X lateral, Y forward), with δ = z_n − z_{n−1} − odom_n rotated into the body frame at the cycle's heading, dθ the cycle's odometry rotation and d_fwd its forward travel,

  δ_lat ≈ (1 − cos dθ)·Δx − sin dθ·Δy − ψ·d_fwd,   δ_fwd ≈ sin dθ·Δx + (1 − cos dθ)·Δy,   δ_θ ≈ 0

(ψ = helios yaw relative to body, with r1's sign verified live: the localiser reports θ_true + ψ.) Weight each row by 1/m_a from the noise learner (its model variance for that axis that cycle) times (1 − outlier responsibility). Accumulate normal equations (3×3) with forgetting; prior N(0, (0.05 m, 0.05 m, 1°)²).

**Files:** create `src/mount_factors.h` (class `KinematicMount`), `tools/mount_factors_selftest.cpp` (part 1).

**Interfaces — Produces:**
```cpp
namespace rc::mountf {
struct Info3 { Eigen::Matrix3d H_data = Eigen::Matrix3d::Zero(), H_prior = Eigen::Matrix3d::Zero();
               Eigen::Vector3d b_data = Eigen::Vector3d::Zero(), b_prior = Eigen::Vector3d::Zero(); long n = 0; };
class KinematicMount {
public:
    struct Params { double sigma_xy = 0.05, sigma_yaw = 0.0175, memory = 50000; };
    void set_params(const Params&);
    /// one cycle: delta_body = (lat, fwd, th) innovation in BODY axes; dth, d_fwd = the cycle's odometry;
    /// var = the noise model's per-axis variance (lat, fwd); w_in = 1 - outlier responsibility (per axis)
    void observe(const Eigen::Vector3d& delta_body, double dth, double d_fwd,
                 const Eigen::Vector2d& var, const Eigen::Vector2d& w_in);
    [[nodiscard]] Info3 info() const;            // order: (dx, dy, dyaw)  [dx lateral, dy forward]
    [[nodiscard]] Eigen::Vector3d mean() const;  [[nodiscard]] Eigen::Vector3d sigma() const;
};
}
```

- [ ] **Step 1: failing test** (`tools/mount_factors_selftest.cpp`, part 1): simulate 40k cycles of mixed parked/straight/pivot motion with a planted (Δx, Δy, ψ) = (0.03, −0.02, 0.0175), white δ noise with variance m (given to the estimator), 4 % gross outliers flagged by w_in; assert recovery within 2σ and σ < 5 mm / 0.1°; case B: straights only → Δx, Δy uninformed (σ ≥ 0.9 × prior); case C: pivots only → ψ uninformed. Also case D (replay pin): build δ rows from `tmp/noise_innov/innov_2026-10-08_12-18-41.csv` if present and assert the fit lands within 2σ of (0.03, −0.02, 0.0175) — skip with a printed note if the file is absent.
- [ ] **Step 2: run, see it fail** (header missing).
- [ ] **Step 3: implement** as specified (rows exactly as the model above; information form; forgetting; prior separate).
- [ ] **Step 4: run, all pass.**
- [ ] **Step 5:** (no commit — leave uncommitted)

### Task 1b: Odometry-readiness policy — the extrinsic waits for a stable odometry calibration, by WEIGHT, not by a gate

The kinematic factor treats the odometry as calibrated. That is only true once the odometry calibration has converged and stopped moving; before that, an odometry error would be read as a mount error and then frozen into the extrinsic. The user's requirement: extrinsic calibration must wait for a stable odometry calibration. The policy is a model term, not a switch (CLAUDE.md):

1. **Propagate the odometry calibration's posterior into each kinematic row (errors-in-variables).** Let θ_o be the odometry parameters the motion calibrator APPLIES (k_v, k_omega, b_omega, dk_wheel, k_omega_w, k_lat — NOT eps_yaw or the lever, see 3) with posterior covariance Σ_o = its information H⁻¹ (`MotionCalibrator::information()`), and J_o the cycle's Jacobian of δ on θ_o (the same covariates the calibrator already uses: d_fwd on k_v, the gyro/wheel rotation shares on the heading parameters, etc.; translation rows pick up heading-parameter error through the cycle's lever arm d_fwd·dθ_err). The row variance becomes  var_a + u_aᵀ J_o Σ_o J_oᵀ u_a . While Σ_o is wide (a fresh robot, a reset, too little excitation) the kinematic rows carry little information and the mount stays near its prior; as Σ_o shrinks, the same rows count more. No `informed` flag, no episode count, no threshold.
2. **Stability = not drifting, also as a variance.** Add the odometry calibration's recent movement as process noise: Σ_drift = the covariance of the applied θ_o's change over the last window (exponentially weighted outer product of Δθ_o per re-solve, units of θ_o), so  Σ_o ← Σ_o + Σ_drift  in 1. A calibration that is still sliding (its applied value moving) is distrusted in proportion to how much it moves, even if its reported σ is small — the case a pure posterior misses.
3. **One owner per parameter.** In the decoupled design the helios planar mount belongs to the kinematic factor. The motion calibrator's eps_yaw and lever columns must not ALSO act on the odometry: set their apply bits off (eps_yaw is then reported, not applied; the lever was never applied) so the odometry the kinematic factor trusts does not already contain a competing estimate of the same mount. θ_o in 1 excludes them.
4. **Evidence stays valid when the odometry calibration moves.** Kinematic rows were computed with the θ_o applied at the time. When the applied θ_o changes by Δθ_o, re-reference the accumulated evidence (b_data −= H_rowsᵀ … via the stored Σ J_o-weighted cross term), or — simpler and sufficient given 2 — keep a per-row record of the θ_o that acted (like r1's p_applied) and add J_o·(θ_o,now − θ_o,then) to the row before accumulating... Implement the simple form: store in the accumulator the cross-information C = Σ_rows w·x_mount·J_oᵀ and on a change Δθ_o apply b_data += C·Δθ_o (exact for the linear model). Test: a planted odometry change mid-run must not move the mount estimate.

**Files:** `src/mount_factors.h` (`KinematicMount::observe` gains `J_o` (3×K), and `set_odometry(Σ_o, θ_o_applied)`), `src/room_concept.cpp` (fill J_o from the cycle's covariates; pass Σ_o + Σ_drift and the applied θ_o from `motion_calib_`), `src/motion_calibration.h` (Σ_drift bookkeeping; apply bits of eps_yaw/lever off when `MountFactors` is on).

- [ ] **Step 1: failing test** (selftest part 1b): (a) Σ_o wide (prior-level) → after 40k cycles the mount σ stays ≥ 0.8 × prior and the mean within 1σ of 0 even with a planted odometry scale error of 3 %; (b) the same run with Σ_o tight and correct → the planted mount is recovered (Task 1's test, unchanged); (c) the odometry's applied k_v jumps by 2 % mid-run → the mount estimate after the jump is within 1σ of the estimate before it (re-reference works); (d) Σ_drift: an applied θ_o that oscillates by ±2 % with a small reported σ → mount σ inflates, no false detection.
- [ ] **Step 2–4:** fail → implement → pass.

### Task 2: `FloorPlaneMount` — bpearl z, roll, pitch from the floor seen head-on

For each bpearl sweep (throttled, `BpearlMountRate` Hz), points p (body frame via the NOMINAL mount) and q = p − t_nom (relative to the sensor origin). A floor point satisfies z_true = 0; a small mount error (ω, dt) moves it by ω × q + dt, so the floor row is

  r_i = p_z + ω_x·q_y − ω_y·q_x + dz,   J_i = [q_y, −q_x, 1]  on (droll, dpitch, dz)

Association without a band: a two-component mixture per point — floor N(0, σ_f²) vs non-floor (uniform over the sweep's z range near the floor); σ_f and the floor share learnt per sweep by 3 EM iterations; rows weighted by the floor responsibility. Per-sweep common mode: the floor is not perfectly flat and the whole sweep shares one timing/pose — add a per-sweep nuisance dz_s with prior σ_flat (0.005 m) and marginalise it (Schur) before adding the sweep's 3×3 to the accumulator, so 10⁴ points cannot claim 10⁴ independent measurements. Prior N(0, (0.02 m, 1°, 1°)²).

**Files:** `src/mount_factors.h` (class `FloorPlaneMount`), selftest part 2.

**Interfaces — Produces:** `class FloorPlaneMount { void observe_sweep(const std::vector<Eigen::Vector3f>& p_body, const Eigen::Vector3f& t_nom); Info3 info() const; /* order (droll, dpitch, dz) */ double floor_share() const; double sigma_floor() const; };`

- [ ] **Step 1: failing test** part 2: synthetic dome sweeps (inverted bpearl at 0.67 m: rays to the floor out to ~3 m plus wall hits), planted (roll 0.5°, pitch −0.8°, dz 0.01 m); case A recover within 2σ; case B 20 % of near points on furniture at z ∈ (0.3, 0.8) → still recovered; case C a per-sweep floor offset of ±5 mm (common mode) → σ does not collapse below ~σ_flat/√sweeps.
- [ ] **Step 2–4:** fail → implement → pass.

### Task 3: `VerticalMount` — roll/pitch from vertical walls and a horizontal ceiling; x/y/yaw relative to the wall map

For one LiDAR (helios or bpearl), per throttled sweep: the room's wall segments (from room_concept's wall map, in the BODY frame at this sweep's pose) and the sweep's points. Wall points (responsibility-weighted against "not this wall", as in Task 2) give rows on a VERTICAL plane n·xy = d:

  r_i = nᵀ p_xy − d + nᵀ (ω × q)_xy + nᵀ dt_xy,   (ω × q)_xy = (ω_y q_z − ω_z q_y,  ω_z q_x − ω_x q_z)

Tilt (ω_x, ω_y) enters through q_z (height above the sensor): points at different heights on one wall separate it from a planar shift. The per-sweep planar POSE error (the localiser's x, y, yaw for that sweep, identical for every point) is a nuisance with the localiser's own covariance as prior and is MARGINALISED per sweep (Schur), so pose error is never read as mount. Ceiling points (helios) add rows on a horizontal plane z = h_c with h_c an unknown per-session nuisance: r = p_z − h_c + (ω × q)_z + dz → roll/pitch, with dz absorbed by h_c (so helios dz stays at its prior: Review Focus #4). Use the full wall height span available (not the 1.5–2.0 m localisation band).

What each LiDAR gets: helios → (droll, dpitch) strongly (upper walls + ceiling), (dx, dy, dyaw) relative to the map — which is the helios's own frame, so ≈ uninformative (the kinematic factor carries those); bpearl → (droll, dpitch) from its lower-wall points (redundant with the floor), and (dx, dy, dyaw) RELATIVE TO THE HELIOS-BUILT MAP = bpearl relative to helios.

**Files:** `src/mount_factors.h` (class `VerticalMount`, 6-DoF info), selftest part 3.

**Interfaces — Produces:** `struct Info6 { Eigen::Matrix<double,6,6> H_data, H_prior; Eigen::Matrix<double,6,1> b_data, b_prior; long n; };` `class VerticalMount { struct Wall { Eigen::Vector2f n; float d; }; void observe_sweep(const std::vector<Eigen::Vector3f>& p_body, const Eigen::Vector3f& t_nom, const std::vector<Wall>& walls, const Eigen::Matrix3f& pose_cov, bool use_ceiling); Info6 info() const; /* (dx,dy,dz,droll,dpitch,dyaw) */ };`

- [ ] **Step 1: failing test** part 3: synthetic 4×6 m room, walls 2.6 m, ceiling; case A planted helios tilt (roll 0.7°, pitch −0.4°) → recovered within 2σ; case B planted bpearl planar offset (dx 0.02, dy −0.03, dyaw 0.8°) relative to the map → recovered; case C a random per-sweep pose error (σ 3 cm, 1°) with ZERO tilt → tilt within 2σ of 0 (Review Focus #1); case D helios dz planted 0.02 → stays at prior σ (unobservable with the ceiling nuisance).
- [ ] **Step 2–4:** fail → implement → pass.

### Task 4: 6-DoF mount transform and injection for both LiDARs

Extend `src/lidar_mount.h` with `Pose6 { Eigen::Vector3f t; Eigen::Vector3f rpy; }`, `apply(Pose6, p)`, `compose`, `inverse` (rotation about the sensor origin: p' = R(rpy)(p − t_nom) + t_nom + t). `LidarIngestor` keeps one correction and one injection per LiDAR (`helios`, `bpearl`); injection = inverse(M_inj) (r1 convention). Config: `LidarMountInject{X,Y,Z,RollDeg,PitchDeg,YawDeg}` and `BpearlMountInject{...}` (all 0; WARN when nonzero).

- [ ] **Step 1: failing test** `tools/lidar_mount_selftest.cpp` part 2: `compose(M, inverse(M)) == identity` for 6-DoF; a correct estimate cancels a planted M on a point cloud; rotation is about the sensor origin (a point AT the sensor origin is moved only by t).
- [ ] **Step 2–4:** fail → implement → pass; `cbuild -j8` clean.

### Task 5: Wiring and the joint over helios 6 + bpearl 6 + cameras

- room_concept: in `observe_innovation`, when the cycle `fed` the noise learner, feed `KinematicMount` with δ in body axes, the cycle's dθ and d_fwd, `noise_innov_.odom_var(0..1) + s·R` as var and (1 − last_r) as w_in. Expose `Info3` via `UpdateResult` (copy, like `calib_information` — thread rule from r1).
- LidarIngestor: permanent bpearl reader (same factory as helios), throttled to `BpearlMountRate`; per throttled sweep call `FloorPlaneMount` (bpearl) and `VerticalMount` (both LiDARs) with the wall segments + pose covariance snapshot passed in from the localiser (copied under the existing motion-ingress lock — never read the wall map from the ingest thread).
- `joint_calibration.h`: new `solve_mounts(helios: Kinematic Info3 ⊕ Vertical Info6, bpearl: Floor Info3 ⊕ Vertical Info6 (relative to helios), cameras)` → 12 + 4·N parameters. bpearl's vertical (dx, dy, dyaw) rows are RELATIVE to helios: map them with A = [I_bpearl − I_helios] on those three columns. Camera yaw couples to helios yaw exactly as in r1 (kCam/kEps with the helios yaw now carried directly, sign kept consistent with r1's live-verified kEps = −1).
- Monitor CSV `tmp/joint_calib/mounts_<ts>.csv`: helios 6, bpearl 6, cameras 4·N, each with σ and the per-factor information share (how much of each parameter's precision each factor supplies).
- Config: `MountFactors = true`, `BpearlMountRate = 2.0`.
- [ ] **Step 1:** `tools/joint_mounts_selftest.cpp`: synthetic Info blocks → (a) bpearl absolute planar = bpearl-relative + helios kinematic; (b) cameras body-frame yaw = relative + helios yaw (r1 sign); (c) helios dz stays at prior. Fail → implement → pass.
- [ ] **Step 2:** `cbuild -j8` clean; all selftests (r1's and this plan's) pass.

### Task 6: Live validation in Webots (user drives, main session grades)

Restore the r1 Task 6 config first (r1 plants off). Arms, each ~10 min with turns at two rates + straights, evidence of the mount estimators reset per arm:
- **Null**: no plant → helios 6 and bpearl 6 within 2σ of 0.
- **Helios planar** (30, −20) mm / 1° → kinematic factor recovers it (the r1 failure case).
- **Helios tilt** roll 0.7°, pitch −0.4° → vertical factor recovers it; helios planar unaffected.
- **bpearl** dz 0.01 m, roll 0.5°, pitch −0.8°, dx 0.02, dyaw 0.8° → floor + vertical recover it; helios unaffected.
Grade: detected (|est| > 2σ) AND agrees (|est − plant| < 2σ), signed. Then `LidarMountApply` arms as in r1.

## Revision log
- r2 (2026-10-08): decoupled from the motion calibrator by the user's decision; bpearl + floor + verticals added.
- r2.1 (2026-10-08): Task 1b — the extrinsic WAITS for a stable odometry calibration (user requirement) by propagating the odometry calibration's posterior + recent drift into the kinematic rows; single owner for the helios planar mount.
