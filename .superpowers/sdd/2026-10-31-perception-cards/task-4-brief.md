### Task 4: The LiDAR returns term

**Files:**
- Modify: `common/perception_card/evaluator.h`
- Test: `common/perception_card/tests/evaluator_lidar_test.cpp`

**Interfaces:**
- **Produces:**
  - `float expected_returns(const LidarRig&, const rc::nbv::Target&, const Eigen::Vector2f& from, std::span<const rc::nbv::Obstacle>)`
  - `ChannelResult evaluate_lidar(const Card&, const World&, const Hypothesis&, const Viewpoint&, std::string_view sensor)`
- **The algorithm** (spec §4b):
  - For each ring elevation (`ring_elev_deg`, or rings spread uniformly over `fov_start..fov_end`), transform the
    ring's cone through `mount` and find whether it crosses the target's z span at the target's horizontal
    range.
  - For each crossing ring, the number of columns = target angular width / (azimuth step), × `visible_fraction`
    along that bearing.
  - λ = the sum × `range_factor(r)`.
  - P = 1 − exp(−λ · `q_return`).
  - `range_factor(r) = Φ((r − range_min)/σ_r) · Φ((range_max − r)/σ_r)`, with σ_r = the model's
    `noise_range_sigma_floor_m`, or 0.02 if absent, stated in a note. This is my call: it's soft at both ends,
    and the far end never matters indoors.
  - **Limit:**
    - `BetweenRings` if no ring crosses the z span while the span lies inside the fan;
    - `BelowFan` / `AboveFan` if the span is outside the fan;
    - `DeadZone` if r < range_min;
    - `Occluded` / `BehindWall` as for the camera.
- `World.helios`/`bpearl` absent → `limit == NotModelled`, and the channel is left out of the combination (Review
  Focus 2).

Steps:
- [ ] **Step 1: write the tests.** Helios model: 32 rings −55..+15, az 0.4°, range 0.1–80 m, mounted inverted at
  1.075 m. Bpearl: 32 rings 0..90 (2.857°), range 0.1–30 m, inverted at 0.67 m.
  - (a) Fridge at 3 m → λ > 50, p > 0.99.
  - (b) A 2 cm-thick horizontal slab at a height between two Helios rings at 6 m → `BetweenRings`, p < 0.1.
  - (c) A 5 cm-tall object on the floor at 0.5 m, seen by Helios → `BelowFan`; by Bpearl → p > 0.9.
  - (d) Target at 0.05 m from the Bpearl → `DeadZone`.
  - (e) `World.helios = nullopt` → `NotModelled`.
  - (f) `q_return = 0.1` lowers p versus 1.
- [ ] **Step 2: run them and see them fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: commit** `perception_card: LiDAR returns term (Poisson, ring geometry through the mount)`.

