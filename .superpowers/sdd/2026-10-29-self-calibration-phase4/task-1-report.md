# Task 1 report: 2-D point-to-line scan registration

Status: DONE_WITH_CONCERNS. Commit: see `git log -1` on feat/self-cal-phase4 ("self_calibration: 2-D point-to-line scan registration (EM-weighted, honest covariance)").

## Implemented
- `common/sensor_calib/scan_registration.{h,cpp}` (in `ai_common_sensor_calib`): GN on n_i.(R p + t - q_i), nanoflann 2-D kd-tree (double), PCA normals (k neighbours), EM responsibilities (inlier Gaussian vs uniform over outlier_window_m, pi re-estimated), no gate.
- `self_calibration/tests/synthetic_room.h` (Room2D, notched_room, corridor, raycast with noise/seed/max_range; namespace sc::calib::testing) and `scan_registration_test.cpp` + CMake target.

## TDD
- RED: `make -C build scan_registration_test` failed at configure/link (scan_registration.cpp absent).
- GREEN: `./bin/scan_registration_test`:
  known_motion err (0.80 mm, 0.03 mm, -0.0044 deg); init_from_odometry (-0.24, -1.21 mm, -0.0165 deg); outliers (0.90, -0.68 mm, -0.019 deg, share 0.853); corridor sd_x=2.9 mm sd_y=0.20 mm; honest_sigma ratio 1.209; `ALL PASS`.

## Deviations from the brief (all measured)
1. Annealed inlier sigma (0.25 -> sigma_m, x0.7/iter). With a fixed 1 cm sigma, known_motion (0.30 m shift) converged to -257 mm error: empty basin. It is a schedule, not a gate.
2. Convergence tolerance 1e-4 instead of 1e-6: nearest-neighbour flips leave a 3-cycle of ~2e-5 steps forever (ok=false on init_from_odometry with 1e-6, error already 1 mm). ref kept in double for the kd query.
3. Covariance is NOT sigma_hat^2 H^-1: that gave honest_sigma ratio 1.59 (fails 1.5; k=3..60 all 1.53-2.2). Isolated: cur-noise only 0.85, ref-noise only 2.2 -> reference noise is common to ~k residuals. cov now = sigma^2 (H^-1 + H^-1 M_ref H^-1) with a_i = sum_j w_j J_j/k over the lines containing ref point i, sigma^2 = residual var/(1+1/k) (same sensor). Ratio 1.21. Conservative (0.65) if the ref were noise-free.
4. degenerate_corridor needs normal_k=24 (test passes it): with default 8, sd_x/sd_y = 4.7 (<10). 8 neighbours at 0.25 deg / 1 cm span ~7 cm so normals tilt ~0.1 rad and fake along-wall information. Default in ScanRegConfig left at 8 per brief; the header advises >= 24. Consider changing the default for Task 2.

## Self-review
- Cur noise radial (lidar-like). max_range added to raycast so the corridor's far end walls do not inform x. Clutter is sensor-frame uniform in [-2,2]x[-3,3].
- Concern: honest_sigma checks only yaw; x/y not checked. cov uses assumption ref and cur noise equal.

# Fix round 1

Command: `cd self_calibration && make -C build -j8 scan_registration_test && ./bin/scan_registration_test` (pgrep cc1plus = 0 before). Result: ALL PASS.

Output (key lines):
- degenerate_corridor (DEFAULT config) sd_x=14.355 sd_y=0.00536 ex=-0.588 m ey=0.02 mm
- corridor over 100 seeds: sd_x > 5 cm in 99, x within 3 sigma in 97 (test requires >=95 and >=97)
- honest_sigma (empirical/reported, 200 seeds): sc0 (known_motion, init 0) x 1.017, y 1.038, yaw 0.808; sc1 (notched room, (1.2,-0.3,10deg), init off by (5cm,5cm,3deg)) x 1.036, y 0.994, yaw 0.911.

## What changed
- normal_k default 8 -> 24. degenerate_corridor now runs at the default and asserts sd_x > 0.05 m, |ex| < 3 sd_x, sd_x > 10 sd_y, |ey| < 5 mm, plus a 100-seed loop.
- [I2] Re-derivation. Measured: cur-noise-only honest (0.77-1.03), ref-noise-only under-reported 2.3x with the old formula. Residual is now the distance to the fitted line through the k neighbours (centroid + normal), not to the single point q_i; so ref noise enters a residual as sigma^2/k, residual var = sigma^2(1+1/k), and the ref correlation term is a_i = sum_{j: i in N(nn_j)} w_j J_j / k (ref point sits in k lines). With the single-point residual and sigma^2 = resid/2 (my first attempt at the reviewer's derivation) ratios were 1.4-1.9 (x,y,yaw), so the single-point residual was abandoned: that is a change of the measurement model (flagged: deviates from the brief's "q_i").
- [I1] The remaining fake x information came from the noisy normals themselves (J carries a tilt error). cov now uses errors-in-variables: H - E[dJ dJ^T], with Var(tilt) = lambda0/((k-2) lambda1) read per ref line (heteroscedastic ray noise; a global sigma under-estimated the bias 1.6x, local one gives mean residual 0.07). Axes whose info does not exceed the manufactured amount by 2 sd (rel sd sqrt(2k/n)) are reported uninformed: variance = (max cur range)^2, added as a separate term so it does not leak into other axes.
- Minor: non-literal printf format replaced by an if/else.

## Concerns
- The "2 sd" is one stated constant (one-sided 97.5% bound on the realised fake information); it is a threshold-like item in the covariance (not in the estimate). No model-level alternative found: 1 sd gave sd_x>5 cm in only 29/100 seeds, 2 sd gives 99/100.
- 1 seed in 100 reports sd_x < 5 cm in the corridor and 3 in 100 have x outside 3 sigma.
- corridor sd_y is 5.4 mm (leak from the x-uninformed eigenvector through its tiny y component x scan extent).
- Residual model changed to line distance (see above); cov is dishonest only under the "same sensor for both scans" assumption.

# Fix round 2 (N1: remove the hard uninformed-axis switch)

Command: `cd self_calibration && pgrep -c cc1plus (0) && systemd-run --user --scope -q -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G make -C build -j8 scan_registration_test && ./bin/scan_registration_test`

## Changes (scan_registration.cpp)
Removed `observed[]`, `Vbig`, the 2-sd bound and the 1e-12 test. Now, all continuous:
1. H_corr = H - E[manufactured information], eigen-clamped at 0 (PSD housekeeping).
2. Each axis' remaining information is shrunk by the reliability lambda^2/(lambda^2+s^2), s = rel * manufactured information on that axis (rel = sqrt(2k/n)): ~lambda when it dominates its noise, ~0 inside the noise.
3. Ref-correlation term in information form: info = Hc (Hc+M+Pinv)^-1 Hc (equal to the sigma^2 (H^-1 + H^-1 M H^-1) covariance, but finite on a zero-information axis).
4. Gaussian pose prior: variance extent^2 per translation, 1 rad^2 for yaw (extent = largest cur range), added as sigma^2 P0^-1.
5. cov = sigma^2 Hhat^-1. `ok` = converged and finite.
(First attempt, delta-method inflation (1+s^2/lambda^2), blew up to sd 2e10 and 61/100; abandoned.)

## Output
- known_motion/init_from_odometry/outliers unchanged (all ok, mm-level).
- degenerate_corridor (seed 7, default cfg): sd_x=14.355 m, sd_y=5.4 mm, x err 0.588 m (within 3 sd), sd_x>10 sd_y.
- Honesty ratios (200 seeds): sc0 x 1.017 y 1.038 yaw 0.809; sc1 x 1.036 y 0.995 yaw 0.912 (all in [0.7,1.5]).
- 100-seed corridor loop: sd_x > 5 cm in 66/100, x within 3 sigma in 58/100. TARGET (>=95 / >=97) NOT MET. The loop is now printed, not asserted (asserting it would need the cut back). With the 2-sd cut it was 99/97. Cause: a realised fake information that is a +2 sigma draw of its own noise looks like real information to any smooth reliability weight; killing it needs a steeper-than-Gaussian-reliability rule, which is a tuned constant.

## Notes for Task 2
- The residual is point-to-fitted-line (distance to the line through the normal_k neighbours of the nearest ref point), so `n_used`/`rms` are line distances, not distances to single points.
- Consumers MUST use `cov`, not `ok` alone: `ok` means converged and finite only; an unobserved axis (corridor) is signalled by a large cov entry (bounded by the scan extent^2 prior), not by ok=false. In the corridor ~1/3 of noise draws report a few-cm sd along the axis although the true information is zero (empirical x error there ~12 cm): Task 2 should additionally require geometric excitation (non-degenerate scan pairs) rather than trust cov alone on near-degenerate pairs.

# Fix round 3 (model comparison for real vs manufactured information)

Command: `cd self_calibration && pgrep -c cc1plus (0) && systemd-run --user --scope -q -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G make -C build -j8 scan_registration_test && ./bin/scan_registration_test`

## Changes
- Dropped the lambda^2/(lambda^2+s^2) shrink. Per eigen-axis v of H: lambda = v'Hv, F = v'Hfake v, rel^2 = 2k/n (n_eff = n/k independent neighbourhoods).
  H0 (noise): lambda ~ Gamma(shape 1/rel^2, scale F rel^2). H1 (real): lambda = G + S, S ~ Exponential(mean mu), mu = wsum/4 (a wall facing the axis subtends ~a quarter of a 360 deg scan, ~1 information per point; from the scan, not tuned). Closed form f1 = beta e^{-beta lambda}(1-beta theta)^{-a} P(a, lambda(1-beta theta)/theta) (regularised incomplete gamma, in logs). Prior odds 1:1; axis information = P(real) max(lambda - F, 0).
- Kept: Gaussian pose prior, reference-correlation term (information form). Restored the asserts on the 100-seed corridor loop (>=95 wide, >=97 within 3 sigma). No k-sigma cut, no hand-shaped shrink.

## Output
- seed-7 corridor: sd_x=14.355 m, sd_y=5.3 mm, x err 0.588 m (within 3 sd): pass.
- 100-seed corridor loop: sd_x > 5 cm in 99/100, x within 3 sigma in 99/100: pass (>=95 / >=97).
- Honesty (200 seeds, all in [0.7,1.5]): sc0 x 1.017 y 1.038 yaw 0.809; sc1 x 1.036 y 0.995 yaw 0.912.
- known_motion / init_from_odometry / outliers unchanged. `scan_registration_test: ALL PASS`.
- Not investigated: the single seed in 100 with sd_x < 5 cm (a ~+4 sigma draw of the manufactured information, presumably).
