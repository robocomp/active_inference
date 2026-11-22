# Task 22 report — κ ≡ 1 on the beam path (spec 7c.8 rev 3.3.11)

Branch feat/cost-removal (worktree active_inference.worktrees/cost-removal), base 2dd604f9.

## Implementation

- `common/ai_belief/beam_ray.h`: `accumulate_beam_rays` / `beam_rays_energy` return Σ_i E_i, Σ_i ∇E_i, Σ_i W_i with no
  factor; the `kappa == 0` early return is gone (only `parts.empty()` returns early). `BeamRays::kappa` stays as an
  INSTRUMENT (documented: multiplies nothing). New `BeamRays::q_sum` = Σ q_i over the staged rays (unfloored label).
- `common/lidar_select/lidar_select.h` `select_rays_beam`: accumulates q (lb->second) over every STAGED ray →
  `out.q_sum` (double) and `BeamStageStats::q_sum_staged` (float). κ (coverage·ca over the endpoint-in-box returns via
  `select_into_weighted`) is still computed and stored in `out.kappa` / `st.kappa` as the instrument; `st.q_sum`
  (endpoint-in-box Σq) unchanged.
- `countable_concept/src/countable_belief.cpp` `lidar_measured` (persistent path): `endpoints` non-empty and
  `beam.q_sum > 0`; κ no longer read. Flag-off branch unchanged.
- `countable_beam_staging.h`: `BeamLog::qsum` (reset in `begin_cycle`, set from `st.q_sum_staged`); `log.kappa` still the
  old coverage·ca.
- `countable_fitter.cpp`: ai2_log header appends `beam_qsum` at the end; row appends `inst.beam_log.qsum`.
  `beam_kappa` keeps logging the old coverage·ca.
- `tools/lidar_measures_score.py`: reads `beam_qsum` when present, prints `qsum med X (staged rows with qsum 0: N)` in the
  `beam:` line, `n/a (column absent)` on older logs; kappa labelled "(instrument)". Verified on objects_live11 (prints n/a)
  and on a scratch copy with a synthetic column (prints the median and the zero count).
- Comments updated wherever they said κ·Σ E_i or "measured ⇔ κ > 0" (countable_belief.h/.cpp, fitter,
  countable_lidar_range_channel.h, beam_ray.h).
- Flag-off path, probe, dbg_lidar_rays, coverage inputs: untouched.
- Agent binary rebuilt: `countable_concept/bin/countable_concept` (-j8 inside the cbuild-style systemd scope).

## Tests

New:
- `beam_staging_test::measured_needs_camera_attribution` (test now links countable_belief.cpp + viewpoint, CMakeLists
  updated): (a) 9 staged rays all q = 0 ⇒ not measured; (b) one q = 0.3 ⇒ measured; (c) bed geometry (band
  z∈[0, 0.45], Helios at 1.1 m, 20 floor returns beyond the select cylinder): endpoint-in-box 0, old κ 0, staged 20,
  Σq 1.0 ⇒ measured; empty set with q_sum 1 ⇒ not measured.
- `beam_belief_test::kappa_not_in_energy`: κ 0.03, 0.0 vs 1.0 on the same front sweep: E, trial E, g, W, and the
  updated θ/P bit-equal. Kept the per-ray "claimed return rewarded" check (d1 < −1).
- `beam_belief_test::qsum_zero_is_prediction` (replaces kappa_zero_is_prediction): every ray labelled ε_px ⇒ q = 0 ⇒
  not measured, update == predict, camera + rays == camera.
- `beam_live_staging_test::one_sided_view`: box seen from its +x face only (robot 1.5 m from it, ZED heading π, Helios at
  0.40 m below the 0.62 m top, known wall 1 m behind the far face), live staging, live ε_px.

Existing tests de-κ'd (no bound changed):
- `beam_fixture.h`: the fixture's q is now m inverted through m = ε + (1 − 2ε)q at the hand ε (was q = m) so an all-ε
  label gives Σq = 0; `subset` recomputes q_sum (ε parameter; live gap test passes kEpsLive), `append` sums it; helper
  `q_sum_of`. The `kappa` argument now only sets the instrument.
- `beam_belief_test::log_counters`: the deep run used κ = 1e-4 to keep θ in place; now uses the existing hook
  `gn_reject_all_for_test`. Result identical to before (29 gap rays, cy +0.2000, h 0.9100); the "κ = 0" run is now Σq = 0
  (staged 0, gap 0). Bounds unchanged.
- `beam_belief_test::reduction_update`, `persistent_belief_test::beam_of`: hand-built sets get `q_sum` from their labels.
- `beam_ray_test`: dropped the `kappa = 1.0` assignments (no-op now).
- `beam_live_staging_test`: no-membership check reads `r0.q_sum == 0.0` instead of `r0.kappa == 0.0`.

### RED (before the implementation, fields added but unfilled)

- beam_staging_test: 4 FAILED — (b) q_sum_staged 0.000 → measured 0; (c) q_sum_staged 0.000 → measured 0.
- beam_belief_test: FAILED — qsum_zero: measured 1, update ≠ predict, gn_it 4, camera+rays ≠ camera;
  kappa_not_in_energy: κ 0.03 E −85.954458766 vs −2865.148625532 (|g| 5.276e2 vs 1.759e4, tr W 6.49e4 vs 2.16e6);
  κ 0 E 0; log counters q_sum-0 staged 978; reduction (κ field no longer set ⇒ unmeasured) worst_sd fails ×2.
- beam_live_staging_test: OK already (all its fixtures have κ ≈ 0.98, so κ ≡ 1 barely changes them; one_sided_view is a
  Pass addition, not a RED test). Its RED numbers are given below beside GREEN.

### GREEN

- beam_staging_test: PASS. (a) staged 9, Σq 0 → 0; (b) Σq 0.300 → 1; (c) in-box 0, old κ 0.0000, staged 20, Σq 1.000 → 1.
  (One fix in my own new test after the first GREEN run: comparing the double `out.q_sum` to the float stat with 1e-12
  failed on float rounding of 20×0.05f; now `|out.q_sum − 1| < 1e-5`. Not a spec bound.)
- beam_belief_test: OK. kappa_not_in_energy: κ 0.03 and 0 vs 1: E −2865.148625532 all, |g| 1.758712e4, tr W 2.163745e6,
  update bit-equal 1. Per-ray hit − miss −6.129623 nats. qsum_zero: measured 0, update == predict 1, gn_it 0.
  Reduction n̂·u 1.00: worst |Δθ|/σ 0.0012 (yaw 0.0003), √diag P rel 0.0044; n̂·u 0.25: 0.0181 (yaw 0.2772), 0.0029.
- Full countable ctest (in the worktree, config_migration_test included): **26/26 passed**, 181 s.

### Printed numbers (beam_live_staging_test GREEN; RED in brackets)

calibration_nees (rev 3.3.10 (2) latents, 200 draws, gn_conv 200/200 each):
- bg 0.20 m: NEES(cx, w)/2 **1.052** [1.037]; cx alone 0.764 [0.753]; w alone 1.336 [1.317]  — bound [0.5, 2] PASS
- bg 1.00 m: **0.907** [0.892]; cx 0.756 [0.742]; w 1.050 [1.033]  — PASS
- bg 0.05 m (printed): 1.066 [1.050]; cx 0.880 [0.867]; w 1.249 [1.230]

one_sided_view (i), 10 LiDAR-only frames from the truth seed: mask 17426 px, anchor 288 pts, staged 962, nolabel 271,
old κ 0.9808, Σq 445.42, far-face hits 0, gn 1/1.
- σ per DOF (cx, cy, H, w, h, yaw): 0.04435, 0.03153, 0.02251, **0.05999**, 0.01924, 0.03076;
  prior 0.300, 0.300, 0.100, 0.060, 0.060, 0.600.
- σ_w / prior σ_w = **1.000** (CHECK ≥ 0.5 PASS). Far-face coordinate cx − w/2: σ 0.06815 vs its prior 0.3015 (ratio 0.226,
  printed — see concern 1).
- θ − truth: cx −0.00045, cy 0, H +0.02629, w −0.00001, h +0.00733, yaw 0.
one_sided_view (ii), 200 draws, 958 rays/draw, gn_conv 200/200:
- NEES(cy, h)/2 **0.893** [0.880 on the pre-implementation binary] (CHECK [0.5, 2] PASS).
- Printed: NEES(cx, w)/2 0.235; near face cx + w/2 alone 0.470; alone cx 0.001, cy 0.812, H 0.968, w 0.000, h 0.967,
  yaw 0.454.

Unchanged bounds, all PASS: silhouette live bg 0.20 / 1.00 m correct seed max |dw| 0.00734 / 0.00733 ≤ s 0.01062,
|dcx| 0 ≤ s/2; seeds ±3 cm |ew| 0.00435 / 0.01033 ≤ σ_w 0.01924 (identical to RED). Gap under furniture: with − without
cy +0.00445, h −0.00012, H +0.00011 (bound 0.01; gap-ray energy 335.037 / 348.155 nats [328.750 / 341.622]). Creep set and
bchain_jump pass in the full ctest.

## Files

common/ai_belief/beam_ray.h, common/lidar_select/lidar_select.h, countable_concept/src/{countable_beam_staging.h,
countable_belief.cpp, countable_belief.h, countable_fitter.cpp, countable_lidar_range_channel.h},
countable_concept/tests/{CMakeLists.txt, beam_belief_test.cpp, beam_fixture.h, beam_live_staging_test.cpp,
beam_ray_test.cpp, beam_staging_test.cpp, persistent_belief_test.cpp}, countable_concept/tools/lidar_measures_score.py.

## Concerns

1. **Interpretation of one_sided_view (decided before seeing numbers).** "σ along the unseen depth direction" is checked
   on w (the box's x extent, the depth of the unseen face) against prior σ_w 0.06. With the near face pinned, the far face
   cx − w/2 inherits σ_w (0.068), not the 0.30 m cx prior, so ½ of the far-face coordinate's own prior (0.15) would fail by
   construction; it is printed. The NEES pair is (cy, h) — the pair this view observes across the line of sight, the
   rotation of calibration_nees's (cx, w); with a truth seed the unseen direction's error is ≈ 0 by construction
   (cx alone 0.001, w alone 0.000), so a depth-pair NEES would sit near 0.5 × (observed part) and is not a calibration
   test (printed 0.235). Per-draw protocol is calibration_nees's (one update per draw), not 10 frames per draw. If the
   owner meant different DOFs or a 10-frame-per-draw protocol, the test needs re-specifying.
2. **The fixtures cannot tell κ ≡ 1 from the old κ**: every live-staging fixture has κ ≈ 0.98 (front face fully in the
   select box), so calibration_nees moves only +0.015. The bed case (κ ≈ 0) is only exercised by the staging unit test
   (c); the real test is live12.
3. Printed, not bound: on the one-sided view the near-face coordinate's 1-D NEES is 0.470 and yaw 0.454 (under-confident
   ≈ 1.4× in σ), cx/w calibration alone 0.76 / 1.34 on the front view — no DOF over-confident in these fixtures.
4. The fixture q is now the inverse of m at the hand ε (was q = m); only the κ instrument and Σq read it, so no
   existing number moved except where a test was about κ.
