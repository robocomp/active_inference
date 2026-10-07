# Joint Sensor-Mount + Odometry Calibration (planar stage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the chained calibration (LiDAR trusted → odometry corrected against it → cameras corrected against it) with ONE joint estimate in which the helios LiDAR mount (lever arm x, y and yaw), the odometry parameters and the camera mounts are solved together, with the wheels defining the body frame.

**Architecture:** The two existing calibrators already keep their evidence as normal equations (H, b): `rc::calib::BatchEstimator` (motion, `src/calibration_estimator.h`) and `rc::mount::Accum` (one per camera, `src/mount_lidar_pair.h`, wrapped by `rc::camcal::Estimator`). This plan (1) adds the helios lever arm to the motion block, (2) exposes each block's information in physical units, (3) fuses the blocks in a new pure header `src/joint_calibration.h` through the one parameter they share — the helios yaw, which the motion block calls `eps_yaw` and which every camera yaw "relative to the LiDAR" silently contains — (4) runs the joint solve live as a MONITOR (logs only), and (5) adds a LiDAR mount correction + a sim-only mount-error injection at the LiDAR ingestor so the whole loop can be validated against a KNOWN planted error in Webots.

**Tech Stack:** C++23, Eigen 3, header-only estimators with standalone selftests (`g++` direct, no Qt), Qt/DSR agent `room_concept`, Webots simulation with supervisor ground truth.

**Spec:** the self-calibration problem definition artifact "What can Shadow learn about its own body?" (https://claude.ai/artifact/TKT15B54gNRyXLXYaEqXCe, v3) — §2 joint model, §3 gauges G1–G7, §5 motion+LiDAR block, §9.1 "the motion and camera blocks are not block-diagonal", §10 factor graph. Thesis chapter 9 (`~/drive/Tesis/Noe Zapata/tesis/self_calibration.tex`).

## Revision log

- **r1 (2026-10-05, Fable review, `room_concept/docs/fable_joint_plan_review_2026-10-05.md`):** lever covariates accumulated per CORRECTION (lever_s, lever_c), not cos/sin of the net episode turn; `kEpsPerLidarYaw = −1` by derivation; camera PRIOR on the body-frame mount, data on the coupled relative row (b_data/b_prior split); one mount-pose convention M for correction (M) and injection (M⁻¹); the applied yaw stays in `p_applied` (odometry-side consumer zeroed instead of clearing the apply bit); camera evidence re-anchored by the applied LiDAR yaw; `rot_model_sigma` off while the lever acts; Task 6 grades signed values on detection + agreement, yaw-only sign-arbiter leg, both evidence sets reset, the null recorded first.

## Global Constraints

- Project rules in `active_inference/CLAUDE.md` apply. In particular: **no thresholds/gates/clamps** where a model term can do it; `and`/`or`/`not` instead of `&&`/`||`/`!`; C++23 containers/algorithms; numbers from files parsed with `std::from_chars`, files written through `imbue(std::locale::classic())`; never `Qt::DirectConnection`; stop agents with SIGTERM, never `kill -9`.
- Build with `cbuild -j8` (never more than `-j8`; check `pgrep -a cc1plus` first). NO `-march=native`, NO `-DEIGEN_MAX_ALIGN_BYTES=0`.
- Gauges (spec §3), fixed by convention and never estimated: **G2** LiDAR range scale `s_L ≡ 1`; **G3** `body` = wheel frame (origin at the axle midpoint, +Y = straight-drive direction in room_concept's body convention, forward = +Y, lateral = +X); **G5** helios clock is the time reference.
- Parameters are appended at the END of every parameter enum (stored files stay comparable).
- Evidence files persist evidence, not fitted values. Prior means stay at ZERO (re-centring is a measured ratchet — `calibration_estimator.h` Prior comment).
- Every new behaviour that can move the published pose ships **default OFF** behind a config flag the user owns, with a monitor-only mode first.
- Selftests are standalone and built with: `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/<name>.cpp -o /tmp/<name> && /tmp/<name>` from `room_concept/`.

## Review Focus

1. **Straight-only or spin-only driving** — the lever arm is visible ONLY under rotation and `eps_yaw` ONLY under forward travel (spec §5 null-direction table). Expected: the unexcited parameter is reported `informed=false` at its prior σ, never a confident wrong value. Test pinned in Task 1 (Step 1, case B).
2. **A parameter that was estimated but is NOT applied** must record `p_applied = 0` for that parameter, or the feedback undo (`r + J·p_applied`) manufactures a 50 % recovery cap (measured 2026-08-30). Lever columns must stay out of `p_applied` until Task 5 actually applies them. Test pinned in Task 1 (Step 1, case C).
3. **Sign of the helios-yaw ↔ camera-yaw coupling.** Derived on paper in Task 3; a wrong sign doubles the error instead of cancelling it. Pinned by the synthetic test (Task 3) AND by the live injection run (Task 6, check 2), which is the arbiter.
4. **Old evidence files** (`etc/motion_calib_state*.csv` with 13+7 fields, camera files format 3) must still load: lever entries were never applied, so padding them with 0 is exact, not approximate. Test pinned in Task 1 (Step 1, case D).
5. **Camera evidence that refuses to marginalise** (unattributed rows) must not enter the joint solve as if it had been. Test pinned in Task 2 (Step 1, case B).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/calibration_estimator.h` | motion block: episodes → (H, b) → solve | add `P_LEVER_X`, `P_LEVER_Y`; lever Jacobian; `information()` accessor; loader accepts 13+7 rows |
| `src/motion_calibration.h` | online driver of the motion block | lever never `acting()` until `apply_lever`; pass-through `information()` |
| `src/calibration_intake.h` | owns the `BatchEstimator` | `estimator()` accessor |
| `src/mount_lidar_pair.h` | camera block `rc::mount::Accum` | `MarginalInfo marginal_information() const` in physical units |
| `src/camera_calibration.h` | `rc::camcal::Estimator` wrapper | pass-through `marginal_information()` + `prior_sigma()` |
| `src/joint_calibration.h` | **new**, pure: fuse motion + N camera blocks through the shared helios yaw | create |
| `src/lidar_mount.h` | **new**, pure: the extra planar transform applied to LiDAR points (correction ∘ injection) | create |
| `src/lidar_ingestor.cpp` | applies `lidar_mount.h` to every helios point after the device→body transform | modify `pump()` |
| `src/room_config.cpp`, `etc/config.toml` | flags `JointCalibMonitor`, `LidarMountApply`, `LidarMountInject*`, prior `MotionCalibLeverSigma` | modify |
| `src/specificworker.cpp` | gathers the blocks once per motion solve, runs the joint solve, writes the CSV | modify |
| `tools/calib_estimator_selftest.cpp` | motion-block tests | extend (lever cases) |
| `tools/joint_calib_selftest.cpp` | **new**: joint fusion tests | create |
| `tools/lidar_mount_selftest.cpp` | **new**: transform tests | create |
| `tools/joint_calib_report.py` | **new**: grade a live run (monitor CSV vs injected truth) | create |

---

### Task 1: Helios lever arm in the motion block

The localiser estimates the HELIOS pose and converts it to `body` through the nominal mount. If the true mount is offset by Δ = (Δx, Δy) in body coordinates, the "body" pose it publishes is the true body pose shifted by R(θ)·Δ. Over an episode that turns by Δθ, the odometry (which moves the true body) and the localiser disagree by

  r_k = (I − R(Δθ_k)ᵀ) · Δ    per CORRECTION k (Δθ_k = rotation since the previous correction; residual = R(θ_k)ᵀ[R(θ_k)Δ − R(θ_{k−1})Δ]).

An episode SUMS its per-cycle corrections (`motion_calibration.h:405`), and live corrections are one cycle apart (782/812 scored corrections had `sur_open = 1`, run 19-58-02), so the episode's lever term is Σ_k (I − R(Δθ_k)ᵀ)Δ — NOT (I − R(Δθ_net)ᵀ)Δ, which is 10 % low at 0.8 rad, 43 % low at 1.73 rad, and invents a cross column (Fable review 2026-10-05, 1a). The covariates are therefore two ACCUMULATED sums per episode: `lever_s = Σ sin Δθ_k`, `lever_c = Σ (1 − cos Δθ_k)` (rows: X lateral, Y forward):

  r_X ⊃ lever_c·Δx − lever_s·Δy,   r_Y ⊃ lever_s·Δx + lever_c·Δy.

So the lever arm is a column driven by ROTATION, beside `eps_yaw`'s column driven by FORWARD TRAVEL — the "same component, different covariate" separation the header already describes for these two (`calibration_estimator.h:18`). Today the lever effect is absorbed as noise by `rot_model_sigma` (0.030 m per rad, `motion_calibration.h:478`); that term stays untouched in this task (so arm A's estimate is conservative) and is removed in Task 5 while the lever acts. Expected precision from the current evidence file: σ_lever ≈ 16 mm (109 episodes, Σdθ²/pos_var = 3724 m⁻²) — a 3 cm plant is a ~2σ detection until the rotation term goes.

**Files:**
- Modify: `src/calibration_estimator.h` (enum ~line 58, `param_name`, `Prior` ~178, `solve()` ~349-420, `load()` row-width check)
- Modify: `src/motion_calibration.h:139` (`apply_mask`), `:246` (`acting()`), `:520` (`p_applied` fill)
- Modify: `src/room_config.cpp:418` area (new key), `etc/config.toml` (`MotionCalibLeverSigma`)
- Test: `tools/calib_estimator_selftest.cpp`

**Interfaces:**
- Produces: `rc::calib::P_LEVER_X = 7`, `rc::calib::P_LEVER_Y = 8`, `P_COUNT = 9`; `Episode::lever_s`, `Episode::lever_c` (float, accumulated per correction); `Prior::sigma_lever = 0.05f` (m); `BatchEstimator::Information information() const` where
  ```cpp
  struct Information {
      Eigen::Matrix<float, P_COUNT, P_COUNT> H;        // data + prior
      Eigen::Matrix<float, P_COUNT, P_COUNT> H_prior;  // prior only (diagonal)
      Eigen::Matrix<float, P_COUNT, 1>       b;        // p = H^-1 b reproduces solve().value
      int episodes = 0;
  };
  ```

- [ ] **Step 1: Write the failing tests** — append to `tools/calib_estimator_selftest.cpp` before the final summary print, inside `main()`:

```cpp
    // ── 7. LEVER ARM (helios offset from the axle midpoint) ───────────────────────────────────────
    // Forward model: r = (I - R(dth)^T) * lever, in the END robot frame, body X = lateral, Y = forward.
    // Episodes are SUMS of per-cycle corrections (motion_calibration.h:405). Live: 10-60 corrections of a
    // few hundredths of a radian each; rarely one correction spanning a large turn. Both are generated.
    auto lever_rows = [&](BatchEstimator &est, const float truth[P_COUNT], bool spins, bool straights,
                          const Eigen::Matrix<float, P_COUNT, 1> &applied)
    {
        std::uniform_real_distribution<float> TH(-1.5f, 1.5f), D(0.5f, 2.0f);
        std::uniform_int_distribution<int> M(10, 60);
        for (int i = 0; i < 200; ++i)
        {
            const bool spin = spins and (not straights or i % 2 == 0);
            const float dth = spin ? TH(rng) : 0.02f * TH(rng);
            const float dfw = spin ? 0.02f * D(rng) : D(rng);
            const int m = (i % 10 == 9) ? 1 : M(rng);            // 1 in 10: a single end-of-episode correction
            float ls = 0.f, lc = 0.f;
            for (int k = 0; k < m; ++k) { const float t = dth / m; ls += std::sin(t); lc += 1.f - std::cos(t); }
            const float lx = truth[P_LEVER_X] - applied[P_LEVER_X];
            const float ly = truth[P_LEVER_Y] - applied[P_LEVER_Y];
            Episode e;
            e.d_forward = dfw; e.d_theta = dth; e.duration = 1.f + std::abs(dth);
            e.th_gyro = dth; e.t_gyro = e.duration; e.fwd_wheel = dfw;
            e.lever_s = ls; e.lever_c = lc;
            e.r_lateral = lc * lx - ls * ly - truth[P_EPS_YAW] * dfw + noise(0.004f);
            e.r_forward = ls * lx + lc * ly + truth[P_K_V] * dfw     + noise(0.004f);
            e.r_theta   = noise(0.001f);
            e.pos_var = 0.004f * 0.004f; e.theta_var = 0.001f * 0.001f;
            e.p_applied = applied;
            est.add(e);
        }
    };
    {
        float truth[P_COUNT] = {};
        truth[P_LEVER_X] = 0.03f; truth[P_LEVER_Y] = -0.02f; truth[P_EPS_YAW] = 0.01f;
        const Eigen::Matrix<float, P_COUNT, 1> none = Eigen::Matrix<float, P_COUNT, 1>::Zero();

        // A: mixed driving recovers both lever components AND eps_yaw.
        BatchEstimator a; a.configure(Prior{}, 512); lever_rows(a, truth, true, true, none);
        const auto ra = a.solve();
        std::printf("\n7A. lever x %+.4f (truth %+.4f) +- %.4f | y %+.4f (truth %+.4f) +- %.4f | eps %+.4f\n",
                    ra.value[P_LEVER_X], truth[P_LEVER_X], ra.sigma[P_LEVER_X],
                    ra.value[P_LEVER_Y], truth[P_LEVER_Y], ra.sigma[P_LEVER_Y], ra.value[P_EPS_YAW]);
        check(std::abs(ra.value[P_LEVER_X] - truth[P_LEVER_X]) < 3.f * ra.sigma[P_LEVER_X], "7A lever x recovered (3 sigma)");
        check(std::abs(ra.value[P_LEVER_Y] - truth[P_LEVER_Y]) < 3.f * ra.sigma[P_LEVER_Y], "7A lever y recovered (3 sigma)");
        check(ra.informed[P_LEVER_X] and ra.informed[P_LEVER_Y], "7A lever informed under rotation");
        check(std::abs(ra.value[P_EPS_YAW] - truth[P_EPS_YAW]) < 3.f * ra.sigma[P_EPS_YAW], "7A eps_yaw still recovered beside the lever");

        // B: straight-only driving must leave the lever UNINFORMED at its prior, not confidently wrong.
        BatchEstimator b; b.configure(Prior{}, 512); lever_rows(b, truth, false, true, none);
        const auto rb = b.solve();
        check(not rb.informed[P_LEVER_X] and not rb.informed[P_LEVER_Y], "7B straight-only: lever uninformed");

        // C: closed loop -- the applied lever is undone through p_applied and the total is recovered.
        BatchEstimator c; c.configure(Prior{}, 512);
        Eigen::Matrix<float, P_COUNT, 1> half = none; half[P_LEVER_X] = 0.015f; half[P_LEVER_Y] = -0.01f;
        lever_rows(c, truth, true, true, half);
        const auto rc_ = c.solve();
        check(std::abs(rc_.value[P_LEVER_X] - truth[P_LEVER_X]) < 3.f * rc_.sigma[P_LEVER_X], "7C closed loop: TOTAL lever recovered");

        // D: information() reproduces solve().value exactly.
        const auto info = a.information();
        const Eigen::Matrix<float, P_COUNT, 1> p = info.H.ldlt().solve(info.b);
        check((p - ra.value).cwiseAbs().maxCoeff() < 1e-5f, "7D information() reproduces solve()");
    }
```

and add a legacy-row load test right after (it writes a 20-field row and checks it loads):

```cpp
    {
        const char *path = "/tmp/calib_legacy_rows.csv";
        { std::ofstream f(path); f.imbue(std::locale::classic());
          f << "E,1,0,0.5,2,0.01,0,0.001,0.0001,0.0001,0.5,2,0,1,0,0,0,0,0,0,0\n"; }   // 13 + 7 fields
        BatchEstimator l; l.configure(Prior{}, 64);
        check(l.load(path) == 1, "7E a pre-lever (13+7) row still loads: p_applied lever = 0 (exact), "
                                 "lever_s = d_theta, lever_c = 0 (first-order APPROXIMATION)");
    }
```

(add `#include <fstream>` and `#include <locale>` at the top of the file if absent).

- [ ] **Step 2: Run to verify it fails**

Run: `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/calib_estimator_selftest.cpp -o /tmp/calib_estimator_selftest`
Expected: compile error `'P_LEVER_X' was not declared in this scope`.

- [ ] **Step 3: Implement**

In `src/calibration_estimator.h`, append to the enum before `P_COUNT`:

```cpp
        P_LEVER_X,      ///< helios offset from the axle midpoint along body X (lateral), metres.
        P_LEVER_Y,      ///< ... along body Y (forward). Visible ONLY under rotation: a sensor off the
                        ///< spin axis traces a circle, so (I - R(dth)^T)*lever lands in the end-frame
                        ///< translation residual. Separated from eps_yaw (driven by forward travel)
                        ///< by covariate, as the header describes. Spec §5.
```

add to `param_name`: `case P_LEVER_X: return "lever_x"; case P_LEVER_Y: return "lever_y";`

add to `Prior`: `float sigma_lever = 0.05f;   ///< m — 5 cm, the spec's mount prior`

in `solve()`, extend `p0` with `prior_.sigma_lever, prior_.sigma_lever`, and replace the along/cross rows so each carries the lever columns:

```cpp
                // ALONG-track (forward, body Y): wheel scale by distance; lever by the ACCUMULATED rotation
                // covariates (per correction, see Task 1 intro — not cos/sin of the net d_theta).
                Eigen::Matrix<float, P_COUNT, 1> j_along = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_along[P_K_V]     = e.d_forward;
                j_along[P_LEVER_X] = e.lever_s;
                j_along[P_LEVER_Y] = e.lever_c;
                ...
                // CROSS-track (lateral, body X): mount yaw by distance; lateral scale; lever by rotation.
                j_cross[P_EPS_YAW] = -e.d_forward;
                j_cross[P_K_LAT]   =  e.d_lateral;
                j_cross[P_LEVER_X] =  e.lever_c;
                j_cross[P_LEVER_Y] = -e.lever_s;
```

(keep the existing `H +=` / `b +=` lines and their `p_applied` undo unchanged).

Add the accessor after `solve()`: build `H`, `H_prior`, `b` exactly as `solve()` does (factor the accumulation into a private `accumulate(H, H_prior, b)` used by both, so they cannot diverge) and return `Information{H, H_prior, b, int(eps_.size())}`.

Add to `Episode` (after `fwd_wheel`): `float lever_s = 0.f; ///< Σ sin Δθ_k over the episode's corrections` and `float lever_c = 0.f; ///< Σ (1 − cos Δθ_k)`. `save()` appends them at the END of each `E` row (after `p_applied`); update the header comment line.

In `load()`, accept both widths: `13 + P_COUNT + 2` (current) and `13 + 7` (pre-lever). For the legacy width fill `p_applied` from the 7 entries (lever entries 0 — EXACT, the lever was never applied) and set `lever_s = d_theta, lever_c = 0` (a first-order APPROXIMATION — say so in the comment).

In `src/motion_calibration.h`, accumulate the two covariates per CORRECTION: add members `float th_since_corr_ = 0.f, acc_lever_s_ = 0.f, acc_lever_c_ = 0.f;`; in `observe()` add the cycle's `d_theta` to `th_since_corr_` every cycle, and on a corrected cycle (beside `acc_r_fwd_ += r_forward`, line ~405) do
```cpp
                acc_lever_s_ += std::sin(th_since_corr_);
                acc_lever_c_ += 1.f - std::cos(th_since_corr_);
                th_since_corr_ = 0.f;
```
copy them into the `Episode` in `flush()` and zero all three in `reset_episode()`.

In `src/motion_calibration.h`: add `bool apply_lever = false;` beside `apply_mask` (line 139) and make `acting(p)` return false for `P_LEVER_X`/`P_LEVER_Y` unless `cfg_.apply_lever` — so `p_applied` (line 520) records 0 for an estimated-but-unapplied lever. Wire `RoomConcept.MotionCalibLeverSigma` (default 0.05) in `src/room_config.cpp` next to line 418 into the `Prior`, and add to `etc/config.toml` next to `MotionCalibRotModelSigma`:

```toml
MotionCalibLeverSigma = 0.05   # m. Prior on the helios lever arm (spec §5). Learnt only from ROTATION.
```

Extend every consumer that iterates `P_COUNT` for CSV columns (`grep -rn "P_COUNT\|v_k_omega_w" src/ tools/*.py`) so the heading CSV gains `v_lever_x,v_lever_y,s_lever_x,s_lever_y` AT THE END of its header (`src/ground_truth_log.cpp`), and `calib_informed_mask` gains bits 7-8.

- [ ] **Step 4: Run tests to verify they pass**

Run: `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/calib_estimator_selftest.cpp -o /tmp/calib_estimator_selftest && /tmp/calib_estimator_selftest`
Expected: all previous checks still PASS, new `7A`–`7E` PASS, last line `ALL PASS (0 failures)`.
Then `cbuild -j8` from `room_concept/` — expected `[100%] Built target room_concept`, no errors.

- [ ] **Step 5: Commit**

```bash
git add src/calibration_estimator.h src/motion_calibration.h src/room_config.cpp etc/config.toml src/ground_truth_log.cpp tools/calib_estimator_selftest.cpp
git commit -m "room_concept: helios lever arm in the motion calibrator (estimated, not applied)"
```

---

### Task 2: Camera block information in physical units

`rc::mount::Accum::solve()` (`src/mount_lidar_pair.h:545`) works in prior-sigma units with solve vector `x = -p`, marginalises the per-vertex offset internally, and returns only `p` and `sigma`. The joint solve needs the marginalised DATA information and the prior separately, in radians/metres, signed like `p`.

**Files:**
- Modify: `src/mount_lidar_pair.h` (add `MarginalInfo` + `marginal_information()` next to `solve()`, sharing its marginalisation code)
- Modify: `src/camera_calibration.h:255` (pass-through)
- Test: create `tools/mount_info_selftest.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```cpp
  namespace rc::mount {
  struct MarginalInfo {
      Eigen::Matrix4d H_data;      // data-only information on p (physical units: rad, m, rad, -)
      Eigen::Matrix4d H_prior;     // prior information on p (diagonal, physical units)
      Eigen::Vector4d b_data;      // data part of the right-hand side
      Eigen::Vector4d b_prior;     // prior pull toward the anchor (`applied`)
                                   // (H_data + H_prior)^-1 (b_data + b_prior) == Solution::p
      bool ok = false;             // false when solve() would refuse (n < min_n, or unmarginalisable)
  };
  MarginalInfo Accum::marginal_information(const Eigen::Vector4d &unit, long min_n = 30) const;
  }
  // unit = (pitch_sigma, height_sigma, yaw_sigma, dt_sigma): the prior-sigma unit of each column
  rc::mount::MarginalInfo rc::camcal::Estimator::marginal_information(const Eigen::Vector4d &unit) const;
  ```

- [ ] **Step 1: Write the failing test** — `tools/mount_info_selftest.cpp`. Read `mount_lidar_pair.h` first for the exact `PairObs` fields and fill them from a synthetic camera with a known mount error; then:

```cpp
// Offline check: marginal_information() must reproduce Accum::solve() exactly, in physical units.
#include "../src/mount_lidar_pair.h"
#include <cstdio>
using rc::mount::Accum;
static int failures = 0;
static void check(bool ok, const char *what) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what); failures += not ok; }

int main()
{
    const Eigen::Vector4d unit(0.0035, 0.010, 0.0035, 1.0);   // etc/config.toml mount*Sigma
    // A: attributed evidence (every row carries a vertex id) -> marginalisable.
    Accum acc; acc.offset_sigma_px = 5.3;
    fill_synthetic(acc, /*attributed=*/true);   // helper defined above main(): 2000 PairObs over 12 vertices,
                                                // planted error pitch +0.002 rad, height -0.006 m, yaw +0.003 rad
    const auto s  = acc.solve();
    const auto mi = acc.marginal_information(unit);
    check(s.ok and mi.ok, "A: both solve");
    const Eigen::Vector4d p = (mi.H_data + mi.H_prior).ldlt().solve(mi.b_data + mi.b_prior);
    const Eigen::Vector4d p_phys = s.p.cwiseProduct(unit);
    check((p - p_phys).cwiseAbs().maxCoeff() < 1e-9 * (1.0 + p_phys.cwiseAbs().maxCoeff()),
          "A: (H_data + H_prior)^-1 (b_data + b_prior) == Solution::p in physical units");
    const Eigen::Vector4d sig = (mi.H_data + mi.H_prior).inverse().diagonal().cwiseSqrt();
    check((sig - s.sigma.cwiseProduct(unit)).cwiseAbs().maxCoeff() < 1e-9, "A: sigmas agree");
    // B: unattributed evidence with the nuisance ON -> solve() refuses, and so must this.
    Accum bad; bad.offset_sigma_px = 5.3;
    fill_synthetic(bad, /*attributed=*/false);
    check(not bad.solve().ok and not bad.marginal_information(unit).ok, "B: refuses exactly when solve() refuses");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
```

Write `fill_synthetic()` in the same file from `PairObs`'s real fields (the predicted-vs-observed corner pixel pair, its Jacobian rows on the 4 mount parameters, and its vertex id) so that the residual equals `J · planted + N(0, 1 px)`, plus a per-vertex constant offset `N(0, 5.3 px)`.

- [ ] **Step 2: Run to verify it fails**

Run: `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/mount_info_selftest.cpp -o /tmp/mount_info_selftest`
Expected: compile error `'class rc::mount::Accum' has no member named 'marginal_information'`.

- [ ] **Step 3: Implement** — in `Accum`, move the body of `solve()` that builds the marginalised `Hm`, `bm` (currently starting `Eigen::Matrix4d Hm = H; Eigen::Vector4d bm = b;` at line 549, through the Schur/Woodbury step that sets `s.marginalised = true` near line 572) into a private `bool marginalised_normal_equations(Eigen::Matrix4d &Hm, Eigen::Vector4d &bm, long min_n) const` that returns false where `solve()` refuses. `solve()` calls it and continues exactly as before. `marginal_information(unit, min_n)` calls it, then converts from the solve vector `x` (prior-sigma units, `x = -p`) to physical `p`:

```cpp
        MarginalInfo marginal_information(const Eigen::Vector4d &unit, long min_n = 30) const
        {
            MarginalInfo mi;
            Eigen::Matrix4d Hm; Eigen::Vector4d bm;
            if (not marginalised_normal_equations(Hm, bm, min_n)) return mi;
            // Hm, bm are DATA-only in x = -p / unit. The prior in those units is the identity
            // (applied at its anchor `applied`, see Accum::applied). Change of variables p = -D x, D = diag(unit):
            const Eigen::Matrix4d Dinv = unit.cwiseInverse().asDiagonal();
            mi.H_data  = Dinv * Hm * Dinv;
            mi.H_prior = Dinv * Dinv;                       // identity in sigma units
            mi.b_data  = -(Dinv * bm);
            mi.b_prior = -(Dinv * prior_rhs());   // the prior pull toward the anchor, exactly as solve() adds it
            mi.ok      = true;
            return mi;
        }
```

where `prior_rhs()` is the prior's contribution to the right-hand side that `solve()` already adds (factor it out of `solve()` too; it is `-applied` in sigma units per the comment at line ~432 — read the code and use whatever `solve()` uses, so the A-test's equality holds). Add the pass-through to `rc::camcal::Estimator`:

```cpp
        [[nodiscard]] rc::mount::MarginalInfo marginal_information(const Eigen::Vector4d &unit) const
        { return acc_.marginal_information(unit); }
```

- [ ] **Step 4: Run** `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/mount_info_selftest.cpp -o /tmp/mount_info_selftest && /tmp/mount_info_selftest` — expected `ALL PASS (0 failures)`. Then `cbuild -j8` — expected no errors (solve() behaviour unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/mount_lidar_pair.h src/camera_calibration.h tools/mount_info_selftest.cpp
git commit -m "room_concept: camera mount evidence exposed as physical-unit information (no behaviour change)"
```

---

### Task 3: The joint solve (pure header)

**The coupling, derived.** Let the helios be yawed by ψ_h and a camera by ψ_c, both relative to `body` (G3). A wall at body bearing β reaches the helios at β − ψ_h, which the nominal mount reads as a body bearing; the room is explained by an estimated body heading θ + ψ_h. The camera sees a corner at camera bearing β − ψ_c; the prediction from heading θ + ψ_h through the nominal camera mount puts it at β − ψ_h. The camera calibrator explains the gap with a yaw y such that β − ψ_h − y = β − ψ_c, so

  **y_cam(measured, "relative to the LiDAR") = ψ_c − ψ_h.**

The motion block's `eps_yaw` is applied to the odometry heading (`room_concept.cpp:8171`, `theta_mid = … + yaw_offset()`); the localiser reports θ_true + ψ_h, so the closed loop cancels iff **eps = −ψ_h ⇒ `kEpsPerLidarYaw = −1`** (Fable review 1b). The camera pair's yaw column gives p = ψ_c − ψ_h ⇒ `kCamPerLidarYaw = −1`, so A(yaw, eps) = kCam/kEps = +1 (y_rel = ψ_c + eps). Task 6 check 2 (a YAW-ONLY injection) remains the live arbiter for both.

Joint parameter vector θ = [ motion 9 | cam₁ 4 | … | cam_N 4 ], camera entries in BODY frame. For camera c, the relative parameters it measured are `y_c = A_c θ` with `A_c` = identity on its own 4 columns, plus `−1/k_e` on the `eps_yaw` column of its yaw row. Joint information:

  H = H_motion + Σ_c [ A_cᵀ H_c,data A_c + I_cᵀ H_c,prior I_c ],  b = b_motion + Σ_c [ A_cᵀ b_c,data + I_cᵀ b_c,prior ]

I_c = identity on camera c's own columns. The DATA measure the relative mount (coupled through eps); the PRIOR is on the camera's own BODY-frame mount, because the two mounts are physically independent. (Keeping the prior on the relative mount would make ψ_c and ψ_h 0.94-correlated before any data — Fable review Q2.) Consequence, pinned by the reduction test: with eps at its prior only and a precise y_rel, the relative measurement is SHARED between the mounts in proportion to their prior variances; "joint == chained" holds only in the limit σ_h → 0.

**Files:**
- Create: `src/joint_calibration.h`
- Test: create `tools/joint_calib_selftest.cpp`

**Interfaces:**
- Consumes: `rc::calib::BatchEstimator::Information` (Task 1), `rc::mount::MarginalInfo` (Task 2).
- Produces:
  ```cpp
  namespace rc::joint {
  inline constexpr float kEpsPerLidarYaw = -1.f;   // eps_yaw = kEpsPerLidarYaw * psi_h   (derived; Task 6 check 2)
  inline constexpr float kCamPerLidarYaw = -1.f;   // y_cam  = psi_c + kCamPerLidarYaw * psi_h (derived above)
  struct CameraBlock { std::string name; rc::mount::MarginalInfo info; };
  struct Solution {
      Eigen::VectorXd value, sigma;           // size 9 + 4*N; camera entries in BODY frame
      Eigen::MatrixXd cov;
      std::vector<std::string> names;         // "k_v", ..., "lever_y", "<cam>.pitch", "<cam>.height", "<cam>.yaw", "<cam>.dt"
      bool ok = false;
  };
  Solution solve(const rc::calib::BatchEstimator::Information &motion, const std::vector<CameraBlock> &cams);
  }
  ```

- [ ] **Step 1: Write the failing test** — `tools/joint_calib_selftest.cpp` builds blocks DIRECTLY (no episodes, no pairs) so it tests only the fusion algebra:

```cpp
#include "../src/joint_calibration.h"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc;

int main()
{
    const double psi_h = 0.010, psi_c = -0.004;          // truth, rad, body frame
    const double eps   = joint::kEpsPerLidarYaw * psi_h; // what the motion block sees
    const double y_rel = psi_c + joint::kCamPerLidarYaw * psi_h;   // what the camera block sees

    // Motion block: eps_yaw measured to sigma 0.002, everything else at prior (H = prior only).
    calib::BatchEstimator::Information m;
    m.H.setZero(); m.H_prior.setZero(); m.b.setZero();
    const float sp[calib::P_COUNT] = {0.02f, 0.0175f, 0.02f, 5e-4f, 0.05f, 0.02f, 0.155f, 0.05f, 0.05f};
    for (int i = 0; i < calib::P_COUNT; ++i) m.H_prior(i, i) = m.H(i, i) = 1.f / (sp[i] * sp[i]);
    const float w_eps = 1.f / (0.002f * 0.002f);
    m.H(calib::P_EPS_YAW, calib::P_EPS_YAW) += w_eps;
    m.b[calib::P_EPS_YAW] = w_eps * float(eps);

    // Camera block: relative yaw measured to sigma 0.001, other 3 at prior.
    joint::CameraBlock cam{"ricoh", {}};
    const Eigen::Vector4d unit(0.0035, 0.010, 0.0035, 1.0);
    cam.info.H_prior = unit.cwiseInverse().cwiseAbs2().asDiagonal();
    cam.info.H_data.setZero(); cam.info.H_data(2, 2) = 1.0 / (0.001 * 0.001);
    cam.info.b_prior.setZero();
    cam.info.b_data.setZero(); cam.info.b_data[2] = cam.info.H_data(2, 2) * y_rel;
    cam.info.ok = true;

    const auto s = joint::solve(m, {cam});
    check(s.ok, "joint solve ok");
    const int iy = calib::P_COUNT + 2;   // ricoh.yaw (body frame)
    std::printf("  eps %+.5f (truth %+.5f) | ricoh.yaw body %+.5f (truth %+.5f) +- %.5f | corr %+.3f\n",
                s.value[calib::P_EPS_YAW], eps, s.value[iy], psi_c, s.sigma[iy],
                s.cov(calib::P_EPS_YAW, iy) / (s.sigma[calib::P_EPS_YAW] * s.sigma[iy]));
    check(std::abs(s.value[iy] - psi_c) < 1e-3, "camera yaw recovered in the BODY frame");
    // Chained (today): camera yaw taken as relative + LiDAR assumed perfect => error = psi_h.
    check(std::abs(y_rel - psi_c) > 5e-3, "chained estimate is off by the LiDAR yaw (the defect)");
    // The joint sigma on the body-frame camera yaw must include the eps uncertainty (it is a sum):
    check(s.sigma[iy] > 0.0019, "joint sigma >= motion sigma (LiDAR-yaw uncertainty propagated)");
    check(s.names[iy] == "ricoh.yaw", "names line up");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
```

- [ ] **Step 2: Run to verify it fails** — `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/joint_calib_selftest.cpp -o /tmp/joint_calib_selftest` → `fatal error: ../src/joint_calibration.h: No such file or directory`.

- [ ] **Step 3: Implement** `src/joint_calibration.h`:

```cpp
/*  joint_calibration.h — ONE posterior over the odometry parameters, the helios mount and every camera
 *  mount, from the evidence the separate calibrators already keep.
 *
 *  WHY. The chain "LiDAR trusted -> odometry against it -> cameras against it" never questions the
 *  LiDAR, and each link reports itself certain of what the previous link assumed. The helios yaw is
 *  the parameter they share (spec §9.1): the motion block measures it as eps_yaw, and every camera yaw
 *  "relative to the LiDAR" contains it. Fusing through that one column gives camera mounts in the
 *  BODY frame (gauge G3: the wheels define body) with the LiDAR-yaw uncertainty carried, not dropped.
 *  Pure: no DSR, no Qt. Monitor-only until Task 5/6 validate application.
 */
#pragma once
#include "calibration_estimator.h"
#include "mount_lidar_pair.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace rc::joint
{
    inline constexpr float kEpsPerLidarYaw = -1.f;   ///< eps_yaw = k * psi_h: eps acts on the odometry heading
                                                     ///< (room_concept.cpp theta_mid + yaw_offset()) and the
                                                     ///< localiser reads theta_true + psi_h ⇒ eps = −psi_h.
                                                     ///< ⚠ CONFIRMED LIVE BY TASK 6 CHECK 2 (yaw-only leg).
    inline constexpr float kCamPerLidarYaw = -1.f;   ///< y_cam = psi_c + k * psi_h. Derived: plan Task 3.
    inline constexpr int   kCamYawRow = 2;           ///< rc::camcal::P_YAW

    struct CameraBlock { std::string name; rc::mount::MarginalInfo info; };
    struct Solution
    {
        Eigen::VectorXd value, sigma;
        Eigen::MatrixXd cov;
        std::vector<std::string> names;
        bool ok = false;
    };

    inline Solution solve(const rc::calib::BatchEstimator::Information &m, const std::vector<CameraBlock> &cams)
    {
        constexpr int NM = rc::calib::P_COUNT;
        const int n = NM + 4 * static_cast<int>(cams.size());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        H.topLeftCorner(NM, NM) = m.H.cast<double>();
        b.head(NM)              = m.b.cast<double>();

        Solution s;
        for (int i = 0; i < NM; ++i) s.names.emplace_back(rc::calib::param_name(i));
        for (std::size_t c = 0; c < cams.size(); ++c)
        {
            const auto &ci = cams[c].info;
            for (const char *p : {"pitch", "height", "yaw", "dt"}) s.names.push_back(cams[c].name + "." + p);
            const int o = NM + 4 * static_cast<int>(c);
            // y_rel = A theta: identity on the camera's own columns, + coupling on eps_yaw for the yaw row.
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, n);
            A.block(0, o, 4, 4).setIdentity();
            A(kCamYawRow, rc::calib::P_EPS_YAW) = kCamPerLidarYaw / kEpsPerLidarYaw;
            if (not ci.ok)                                  // no evidence: unit prior, zero mean, never singular
            { H.block(o, o, 4, 4) += Eigen::Matrix4d::Identity(); continue; }
            // DATA on the relative mount (coupled through eps); PRIOR on the camera's own body-frame mount
            // (the two mounts are physically independent — a relative prior would correlate them 0.94).
            H += A.transpose() * ci.H_data * A;
            b += A.transpose() * ci.b_data;
            H.block(o, o, 4, 4) += ci.H_prior;
            b.segment(o, 4)     += ci.b_prior;
        }
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() != Eigen::Success) return s;
        s.value = ldlt.solve(b);
        s.cov   = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
        s.sigma = s.cov.diagonal().cwiseMax(0.0).cwiseSqrt();
        s.ok    = s.value.allFinite() and s.cov.allFinite();
        return s;
    }
}
```

Add this reduction case to `tools/joint_calib_selftest.cpp` before the summary print — it pins that the joint changes nothing when the LiDAR yaw is unknown, and only widens σ:

```cpp
    {   // REDUCTION: eps_yaw at its prior only — the precise relative measurement is SHARED between the
        // two mounts in proportion to their prior variances (sigma_c^2 : sigma_h^2).
        calib::BatchEstimator::Information m0 = m;
        m0.H(calib::P_EPS_YAW, calib::P_EPS_YAW) = m0.H_prior(calib::P_EPS_YAW, calib::P_EPS_YAW);
        m0.b.setZero();
        const auto s0 = joint::solve(m0, {cam});
        const double vc = 1.0 / cam.info.H_prior(2, 2), vh = 1.0 / m0.H_prior(calib::P_EPS_YAW, calib::P_EPS_YAW);
        const double expect_c = y_rel * vc / (vc + vh);
        std::printf("  reduction: cam yaw %+.5f (expect %+.5f), eps %+.5f\n", s0.value[iy], expect_c, s0.value[calib::P_EPS_YAW]);
        check(std::abs(s0.value[iy] - expect_c) < 0.1 * std::abs(y_rel), "reduction: split in proportion to the prior variances");
        // Limit sigma_h -> 0 (helios yaw known): the joint must equal the chained answer.
        calib::BatchEstimator::Information m1 = m0;
        m1.H(calib::P_EPS_YAW, calib::P_EPS_YAW) += 1e12f;
        const auto s1 = joint::solve(m1, {cam});
        const Eigen::Matrix4d Hc = cam.info.H_data + cam.info.H_prior;
        const double own = Hc.ldlt().solve(cam.info.b_data + cam.info.b_prior)[2];
        check(std::abs(s1.value[iy] - own) < 1e-6, "limit sigma_h -> 0: joint == chained");
    }
```

- [ ] **Step 4: Run** `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/joint_calib_selftest.cpp -o /tmp/joint_calib_selftest && /tmp/joint_calib_selftest` → `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/joint_calibration.h tools/joint_calib_selftest.cpp
git commit -m "room_concept: joint calibration solve -- motion + camera blocks fused through the helios yaw (pure, monitor-only)"
```

---

### Task 4: Live joint MONITOR (logs only, changes nothing)

**Files:**
- Modify: `src/calibration_intake.h` (add `[[nodiscard]] const BatchEstimator& estimator() const noexcept { return est_; }`)
- Modify: `src/motion_calibration.h` (add `[[nodiscard]] rc::calib::BatchEstimator::Information information() const { return intake_.estimator().information(); }`)
- Modify: `src/room_concept.h` (public `[[nodiscard]] const rc::calib::MotionCalibrator& motion_calibrator() const noexcept { return motion_calib_; }`)
- Modify: `src/calib_channels.h` (public `[[nodiscard]] const std::vector<std::unique_ptr<CalibChannel>>& channels() const noexcept { return calib_channels_; }`; make `CalibChannel` public if it is not)
- Modify: `src/specificworker.cpp` (call site: where the motion calibrator's solve result is consumed each cycle; once per NEW solve, compare `last_solve().episodes`)
- Modify: `src/room_config.cpp`, `etc/config.toml` (`JointCalibMonitor = true`)
- Create: `tools/joint_calib_report.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: CSV `tmp/joint_calib/joint_<YYYY-MM-DD_HH-MM-SS>.csv`, one row per new motion solve: `ts_ms,episodes,<name>,<name>_sd,...` for every `Solution::names` entry, then `corr_eps_<cam>yaw` per camera, then the CHAINED values for comparison: `chain_<cam>.yaw` (= the camera's own relative solution). Written through `imbue(std::locale::classic())`.

- [ ] **Step 1: Write the failing test** — `tools/joint_calib_report.py` is the test of the wiring: it fails if the CSV is missing or malformed.

```python
#!/usr/bin/env python3
"""Grade a joint-calibration monitor log. usage: joint_calib_report.py [csv] [--inject x y yaw_deg]
Prints the last solve's joint vs chained camera yaw, the eps<->camera-yaw correlation, and (with
--inject, sim only) whether the joint helios lever/yaw recovered the planted mount error within 2 sigma."""
import argparse, glob, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--inject", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
a = ap.parse_args()
path = a.csv or max(glob.glob("tmp/joint_calib/joint_*.csv"), key=os.path.getmtime, default=None)
if not path: sys.exit("no joint_calib csv (is JointCalibMonitor on?)")
d = pd.read_csv(path)
need = {"ts_ms", "episodes", "eps_yaw", "eps_yaw_sd", "lever_x", "lever_x_sd", "lever_y", "lever_y_sd"}
missing = need - set(d.columns)
if missing: sys.exit(f"malformed: missing {sorted(missing)}")
last = d.iloc[-1]
print(f"{os.path.basename(path)}: {len(d)} solves, last at {last.episodes:.0f} episodes")
for c in [c for c in d.columns if c.endswith(".yaw") and not c.startswith("chain_")]:
    cam = c[:-4]
    print(f"  {cam}: joint yaw {np.degrees(last[c]):+.3f} +- {np.degrees(last[c + '_sd']):.3f} deg | "
          f"chained {np.degrees(last['chain_' + c]):+.3f} deg | corr(eps, yaw) {last['corr_eps_' + cam + 'yaw']:+.2f}")
print(f"  helios: eps_yaw {np.degrees(last.eps_yaw):+.3f} +- {np.degrees(last.eps_yaw_sd):.3f} deg, "
      f"lever ({last.lever_x*1e3:+.1f} +- {last.lever_x_sd*1e3:.1f}, {last.lever_y*1e3:+.1f} +- {last.lever_y_sd*1e3:.1f}) mm")
if a.inject:
    # Signed, on the median of the last N solves; PASS needs detection AND agreement (the prior alone
    # would pass "within 2 sigma of 0.5 deg"). K_EPS mirrors rc::joint::kEpsPerLidarYaw.
    K_EPS = -1.0
    tail = d.tail(min(20, len(d))).median(numeric_only=True)
    x, y, yaw = a.inject[0], a.inject[1], np.radians(a.inject[2])
    est = {"lever_x": (tail.lever_x, tail.lever_x_sd, x), "lever_y": (tail.lever_y, tail.lever_y_sd, y),
           "yaw": (tail.eps_yaw / K_EPS, tail.eps_yaw_sd, yaw)}
    ok = True
    for k, (v, sd, t) in est.items():
        if t == 0.0:
            continue                                  # not planted in this leg
        detected, agrees = abs(v) > 2 * sd, abs(v - t) < 2 * sd
        print(f"  {k}: est {v:+.5f} +- {sd:.5f} vs planted {t:+.5f} | detected {detected} agrees {agrees}")
        ok = ok and detected and agrees
    print(f"  implied kEps from the yaw leg: {np.sign(tail.eps_yaw) * np.sign(yaw) if yaw else float('nan'):+.0f}")
    sys.exit(0 if ok else 1)
```

Run (before implementing): `python3 tools/joint_calib_report.py` → expected `no joint_calib csv (is JointCalibMonitor on?)`.

- [ ] **Step 2: Implement** the accessors listed in Files, then in `src/specificworker.cpp` add a member `std::ofstream joint_csv_; int joint_last_episodes_ = -1;` (in `specificworker.h`) and, at the call site that already reads the motion calibrator each cycle, add:

```cpp
    // ── JOINT CALIBRATION MONITOR (plan 2026-10-05, Task 4): logs only, applies nothing ──────────
    if (params.JOINT_CALIB_MONITOR)
    {
        const auto &mc = room_concept->motion_calibrator();
        if (const int ep = mc.last_solve().episodes; ep != joint_last_episodes_ and ep > 0)
        {
            joint_last_episodes_ = ep;
            const Eigen::Vector4d unit(params.IMAGE_EDGE_MOUNT_PITCH_SIGMA, params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                       params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0);
            std::vector<rc::joint::CameraBlock> cams;
            cams.push_back({params.IMAGE_EDGE_CAMERA, mount_->pool().marginal_information(unit)});
            for (const auto &ch : calib_->channels()) cams.push_back({ch->name, ch->calib.marginal_information(unit)});
            const auto s = rc::joint::solve(mc.information(), cams);
            if (s.ok) write_joint_row(s, cams, unit);   // opens tmp/joint_calib/joint_<ts>.csv on first call
        }
    }
```

`write_joint_row` writes the header once (from `s.names`, each followed by `_sd`, then `corr_eps_<cam>yaw`, then `chain_<cam>.yaw`), with `chain_<cam>.yaw = (ci.H_data + ci.H_prior)^-1 (ci.b_data + ci.b_prior)` row `kCamYawRow` — the camera's own (chained) answer. Add `JOINT_CALIB_MONITOR` to the params struct, read `RoomConcept.JointCalibMonitor` in `room_config.cpp`, and in `etc/config.toml`:

```toml
JointCalibMonitor = true   # LOG ONLY: one joint posterior over odometry + helios mount + camera mounts per
                           # motion solve -> tmp/joint_calib/joint_<ts>.csv. Changes nothing. Grade with
                           # tools/joint_calib_report.py. Plan: docs/superpowers/plans/2026-10-05-joint-calibration.md
```

- [ ] **Step 3: Build** `cbuild -j8` → no errors.

- [ ] **Step 4: Verify on a short live run** (user drives; Claude reads files — never ask for pasted logs). After ≥ 2 minutes of driving with turns: `python3 tools/joint_calib_report.py` → prints ≥ 1 solve, finite values, a `corr(eps, yaw)` per camera. Confirm the published pose is unaffected: `python3 tools/nees_report.py` numbers in the same range as the 2026-10-05 baseline (solve heading ~7x, published ~5x).

- [ ] **Step 5: Commit**

```bash
git add src/calibration_intake.h src/motion_calibration.h src/room_concept.h src/calib_channels.h src/specificworker.cpp src/specificworker.h src/room_config.cpp etc/config.toml tools/joint_calib_report.py
git commit -m "room_concept: joint calibration MONITOR -- one posterior over odometry, helios mount and camera mounts (logs only)"
```

---

### Task 5: LiDAR mount correction + sim-only injection at the ingestor

The helios points are transformed DEVICE→`body` through the DSR RT tree in `LidarIngestor::pump()` (`src/lidar_ingestor.cpp:129`, `reader_->poll(params_->LIDAR_ROBOT_FRAME, false)`). A planar extra transform is applied to every point right after that call:

  p' = T_corr ∘ T_inject (p),  T(x, y, ψ): p ↦ R(ψ)·p + (x, y, 0)

Everything is defined through ONE object, the **mount pose error** M = Planar{lever_x, lever_y, ψ_h} (the sensor's displacement and CCW yaw from its nominal mount, body frame). The nominal transform yields p_nom = M⁻¹(p_true), so (Fable review 1c):
- `T_corr` (`LidarMountApply`) = M_est = `Planar{+lever_x, +lever_y, +eps_yaw / kEpsPerLidarYaw}` — restores p_true.
- `T_inject` (sim only, `LidarMountInject*`) = `inverse(Planar{InjectX, InjectY, InjectYaw})` — plants M_inj = {InjectX, InjectY, InjectYaw}, the way the +3 % WheelScaleV arm planted a wheel error.
- A correct estimate gives `compose(T_corr, T_inject) = M_est ∘ M_inj⁻¹ = identity`, and the report compares lever_x, lever_y, eps/kEps to +InjectX, +InjectY, +InjectYaw, SIGNED.

**Files:**
- Create: `src/lidar_mount.h`
- Modify: `src/lidar_ingestor.cpp` (`pump()` after line 129), `src/lidar_ingestor.h` (setter)
- Modify: `src/motion_calibration.h` (`apply_lever` follows `LidarMountApply`, so `p_applied` records the lever exactly when it acts)
- Modify: `src/room_config.cpp`, `etc/config.toml`
- Test: create `tools/lidar_mount_selftest.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace rc::lidar_mount {
  struct Planar { float x = 0.f, y = 0.f, yaw = 0.f; };     // body frame, metres / rad
  Planar compose(const Planar &a, const Planar &b);          // a ∘ b
  Planar inverse(const Planar &a);
  Eigen::Vector3f apply(const Planar &t, const Eigen::Vector3f &p);   // z untouched
  }
  void rc::LidarIngestor::set_mount_extra(const rc::lidar_mount::Planar &t);   // thread-safe (atomic swap or mutex)
  ```

- [ ] **Step 1: Write the failing test** — `tools/lidar_mount_selftest.cpp`:

```cpp
#include "../src/lidar_mount.h"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc::lidar_mount;
int main()
{
    const Planar inj{0.03f, -0.02f, 0.0087f};               // 3 cm, -2 cm, 0.5 deg
    const Eigen::Vector3f p(2.f, 1.f, 1.7f);
    const Eigen::Vector3f q = apply(compose(inverse(inj), inj), p);
    check((q - p).norm() < 1e-6f, "inverse(inj) o inj == identity");
    const Eigen::Vector3f r = apply(inj, p);
    check(std::abs(r.z() - p.z()) < 1e-9f, "z untouched");
    check(std::abs((r.head<2>() - Eigen::Vector2f(0.03f, -0.02f)).norm() - p.head<2>().norm()) < 1e-5f,
          "rigid: distance from the translated origin preserved");
    // The plan's convention: T_inject = inverse(M_inj), T_corr = M_est; a correct estimate cancels exactly.
    const Planar T_inject = inverse(inj), T_corr = inj;
    check((apply(compose(T_corr, T_inject), p) - p).norm() < 1e-6f, "correct estimate: T_corr o T_inject == identity");
    const Planar ab = compose(Planar{1.f, 0.f, 1.5707963f}, Planar{1.f, 0.f, 0.f});
    check(std::abs(ab.x - 1.f) < 1e-5f and std::abs(ab.y - 1.f) < 1e-5f, "compose applies b first, then a");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
```

- [ ] **Step 2: Run to verify it fails** → `fatal error: ../src/lidar_mount.h: No such file or directory`.

- [ ] **Step 3: Implement** `src/lidar_mount.h`:

```cpp
/*  lidar_mount.h — the planar correction (and sim-only injection) applied to helios points after the
 *  DSR device->body transform. Pure. See plan 2026-10-05 Task 5. */
#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace rc::lidar_mount
{
    struct Planar { float x = 0.f, y = 0.f, yaw = 0.f; };

    [[nodiscard]] inline Eigen::Vector3f apply(const Planar &t, const Eigen::Vector3f &p)
    {
        const float c = std::cos(t.yaw), s = std::sin(t.yaw);
        return {c * p.x() - s * p.y() + t.x, s * p.x() + c * p.y() + t.y, p.z()};
    }
    /// a ∘ b : apply b, then a.
    [[nodiscard]] inline Planar compose(const Planar &a, const Planar &b)
    {
        const float c = std::cos(a.yaw), s = std::sin(a.yaw);
        return {c * b.x - s * b.y + a.x, s * b.x + c * b.y + a.y, a.yaw + b.yaw};
    }
    [[nodiscard]] inline Planar inverse(const Planar &a)
    {
        const float c = std::cos(a.yaw), s = std::sin(a.yaw);
        return {-(c * a.x + s * a.y), -(-s * a.x + c * a.y), -a.yaw};
    }
}
```

In `LidarIngestor`: add `std::mutex mount_mx_; rc::lidar_mount::Planar mount_extra_;` and `set_mount_extra()`; in `pump()`, immediately after the `poll(...)` call, if `mount_extra_` is not identity, transform every point of `sweep->points` in place through `apply()` (read the point type in `common/media_transport` first and convert its x, y, z fields; do not change its type). Inject at construction from config:

```toml
LidarMountApply   = false  # apply the joint posterior's helios (lever_x, lever_y, yaw) to the points.
                           # OFF until Task 6's injection run recovers a planted error. Owner: user.
LidarMountInjectX = 0.0    # m    ⚠ SIMULATION ONLY: plant a KNOWN helios mount error = the sensor's displacement
LidarMountInjectY = 0.0    # m      and CCW yaw from its nominal mount, body frame (points get inverse() of it).
LidarMountInjectYawDeg = 0.0 # deg  Leave all three at 0 on the real robot -- room_config WARNs when any is nonzero.
```

At construction set `mount_extra_ = T_inject = inverse(Planar{InjectX, InjectY, InjectYaw·π/180})`. When `LidarMountApply` is true, after each joint solve (Task 4 site) set `mount_extra_ = compose(T_corr, T_inject)` with `T_corr = Planar{+lever_x, +lever_y, +eps_yaw / rc::joint::kEpsPerLidarYaw}` (use `inverse()` wherever an inverse is meant, never hand-negated fields), and:
1. `motion_calib.apply_lever = true`, so `p_applied` records the lever while it acts.
2. **Keep `acting(P_EPS_YAW)` unchanged** — `p_applied` must record the yaw that acted WHICHEVER side applied it, or the `r + J·p_applied` undo stops undoing it and the loop converges to half the error (Review Focus #2). Instead add `bool lidar_side_yaw = false;` to the calibrator config and make the ODOMETRY-side consumer `yaw_offset()` (`motion_calibration.h:249`, used at `room_concept.cpp:8171`) return 0 when it is set. The same yaw never acts twice, and nothing is lost from `p_applied`.
3. **Re-anchor every camera evidence pool** when the applied LiDAR yaw changes by dψ: call the existing re-anchoring path (`rc::camcal::Estimator::apply_correction`, `mount_lidar_pair.h:476-517`) with `Eigen::Vector4d(0, 0, -dψ / params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 0)` in the solve vector's units, so camera evidence stays expressed against the ORIGINAL LiDAR frame and A with the TOTAL eps stays valid. Verify the sign in the Task 2 selftest by adding a case: re-anchoring by d then solving must shift `p` by exactly `d·unit`.
4. Set `rot_model_sigma` to 0 in the episode variance while the lever acts (`motion_calibration.h:478`): that term charged turning episodes for exactly the effect the lever now models, and it caps σ_lever near 15 mm however long the run.
5. Config comment: turning `LidarMountApply` on mid-run moves the published pose by R(θ)Δ and ψ_h in one step — expected, not a relocalisation.

- [ ] **Step 4: Run** `g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/lidar_mount_selftest.cpp -o /tmp/lidar_mount_selftest && /tmp/lidar_mount_selftest` → `ALL PASS`; then `cbuild -j8` → no errors; then all earlier selftests again (Task 1, 2, 3) → `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/lidar_mount.h src/lidar_ingestor.cpp src/lidar_ingestor.h src/motion_calibration.h src/room_config.cpp etc/config.toml tools/lidar_mount_selftest.cpp src/specificworker.cpp
git commit -m "room_concept: helios mount correction + sim-only mount-error injection at the LiDAR ingestor (both OFF)"
```

---

### Task 6: Validation in Webots against a planted error (user drives, Claude grades)

Three registered runs, each a normal tour with several in-place turns at TWO different rates plus straight legs (spec §5: rotation teaches the lever, straights teach the yaw). Claude reads `tmp/joint_calib/*.csv` and `tmp/heading/*.csv`; the user never pastes logs.

- [ ] **Step 0: Arm C FIRST — the null.** No injection, `LidarMountApply = false`. Record the joint lever/yaw as the baseline (expected |lever| ≲ 1 cm: tonight's GT fit of e = R(θ)L gave (−8.6, +5.7) mm over 911° of heading). A run with little heading span aliases the slow pose-field bias into L — drive at least two full turns.
- [ ] **Before every injected arm:** back up AND delete BOTH evidence sets — `etc/motion_calib_state.csv` and the camera files `etc/camera_calib_Shadow_*.txt` (as `*.pre-joint.*`). Pairs recorded against the uninjected LiDAR contradict the injected ones.
- [ ] **Step 1a: Arm A-yaw — YAW-ONLY injection, monitor only (the SIGN ARBITER).** `LidarMountInjectYawDeg = 0.5`, X = Y = 0, `LidarMountApply = false`. User drives ~5 min with turns at two rates and straights.
  Check 1: `python3 tools/joint_calib_report.py --inject 0 0 0.5` → exit 0 (yaw detected and agreeing, signed).
  Check 2: the printed implied kEps must equal `kEpsPerLidarYaw` (−1); `chain_<cam>.yaw` moved by ≈ −0.5° while the joint `<cam>.yaw` stays within 2σ of arm C. If the joint camera yaw moved by ≈1° instead, `kCamPerLidarYaw` has the wrong sign: flip it, rerun Task 3's selftest, rebuild, re-grade the SAME csv offline. (Yaw-only because a planted lever also shifts every LiDAR corner the camera pairs use — ~0.57° of bearing at 3 m for 3 cm — which the camera's 4 parameters cannot express.)
- [ ] **Step 1b: Arm A-full — lever + yaw, monitor only.** `InjectX = 0.03`, `InjectY = -0.02`, `InjectYawDeg = 0.5`. Check: `--inject 0.03 -0.02 0.5` → lever detected and agreeing (expect σ_lever ≈ 15 mm while `rot_model_sigma` is still on, i.e. a marginal detection — that is the conservative arm). Camera yaw is expected to move by more than ψ_h here; do not grade it in this leg.
- [ ] **Step 2: Arm B — injection, applied.** Same injection, `LidarMountApply = true`. Check: after convergence the residual `lever_*`, `eps_yaw` → 0 within 2σ; `python3 tools/nees_report.py` heading error no worse than arm A; GT position error (same tool) no worse than the 2026-10-05 baseline.
- [ ] **Step 3: Arm C-applied — no injection, applied.** Check: corrections stay within 2σ of the Step 0 null. A significant nonzero value is either a real proto/json mismatch (the helios RT offset (0, −0.155) vs the Webots proto) or a model defect — record which before trusting it.
- [ ] **Step 4: Record** the three arms' numbers in `room_concept/EXPERIMENT.md` (new section) and in the self-calibration chapter's experiments section; reset `LidarMountInject* = 0`.
- [ ] **Step 5: Commit**

```bash
git add EXPERIMENT.md etc/config.toml src/joint_calibration.h
git commit -m "room_concept: joint calibration validated against a planted helios mount error (arms A/B/C)"
```

---

## What this plan deliberately does NOT do (follow-on plans, in order)

1. **3-D LiDAR mounts** — helios roll/pitch (from vertical walls), bpearl z/roll/pitch (floor plane as a continuous factor, replacing the one-shot `[FloorCheck]` in `lidar_ingestor.cpp:563`), helios z through bpearl (spec §8). Needs a 3-D residual the 2-D wall SDF does not have.
2. **Clock offsets** τ_o, τ_g, τ_cam relative to the helios (G5) — needs rate changes; spec §9.4 says the camera τ is mainly a YAW effect (ωτ), which `P_DT` currently models as translation.
3. **Motion noise η** — pending Fable's memo on the window-posterior problem (`motion_noise_vc.h` header). The joint solve's per-episode weights (`pos_var`, `theta_var`) are exactly where an honest η would enter.
4. **Slow drift (OU)** per parameter class (spec §7) — replaces the sliding window's hard forgetting.
5. **Write-back** of converged mounts to `shadow.json` / the graph RT edges (today corrections live only inside room_concept).
