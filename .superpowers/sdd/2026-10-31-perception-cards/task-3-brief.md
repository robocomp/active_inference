### Task 3: World and hypothesis types; the camera evaluator

**Files:**
- Create: `common/perception_card/evaluator.h`
- Test: `common/perception_card/tests/evaluator_camera_test.cpp`

**Interfaces:**
- **Consumes:** `Card` (Task 1); `term_*`/`eval_term` (Task 2); from `common/nbv/viewpoint_score.h`:
  - `Sensor`, `Target`, `Obstacle`, `WallGap`;
  - `wall_obstacles`, `predicted_fill_axes`, `predicted_fill_sigma`, `visible_fraction`, `expected_p_detect`.
- **Produces:**
  - `struct LidarRig { rc::media::SensorModel model; Eigen::Isometry3f mount; };`, mount in the room frame.
  - The world:
    ```
    struct World {
      std::vector<Eigen::Vector2f> polygon;   // empty = no walls known
      std::vector<rc::nbv::WallGap> gaps;
      std::vector<std::pair<std::string, rc::nbv::Obstacle>> obstacles;
      std::optional<rc::nbv::Sensor> zed;
      float zed_mount_z;
      std::optional<LidarRig> helios, bpearl;
    };
    ```
  - `struct Hypothesis { std::string concept, subtype; float x, y, yaw; std::optional<GeometryPrior> size; std::optional<float> phi; float seconds_unseen = 0; };`
  - `struct Viewpoint { Eigen::Vector2f xy; std::optional<float> yaw; };`
    - `nullopt` yaw = "best heading": facing the target.
  - `enum class Limit { None, OutOfView, BehindWall, Occluded, TooSmall, TooClose, EdgeOn, BetweenRings, BelowFan, AboveFan, DeadZone, NotModelled, Inside };`
  - `struct ChannelResult { std::string sensor; float p; Limit limit; std::string occluder; };`
  - `struct Result { float p_notice, p_recognise; std::vector<ChannelResult> channels; Limit limit; std::string occluder; std::vector<std::string> notes; };`
  - `ChannelResult evaluate_camera(const Card&, const World&, const Hypothesis&, const Viewpoint&)`
- **Per-level composition** (spec §2):
  - coverage = `visible_fraction`;
  - countable × the envelope, via `expected_p_detect` with `predicted_fill_sigma`;
  - articulated × the card's configuration terms at `phi`;
  - animate: Gauss–Hermite over position with σ² = σ0² + k·`seconds_unseen`, implemented as `term_motion`
    in `terms.h`. k is the parameter `k` of the card's `"motion"` term.
- **Occluder:** leave-one-out over `obstacles`. The occluder is the obstacle whose removal raises
  `visible_fraction` the most. Report it only if removing it raises visibility.

Steps:
- [ ] **Step 1: write the tests.** All use a 6×4 m room, the ZED intrinsics 1280×720, focal 448, mount 0.945 m,
  and the fridge geometry 0.7×0.7×1.8.
  - (a) Clear view at 3 m → `p > 0.5`, `limit == None`.
  - (b) A 1×1×1.2 m box halfway → `limit == Occluded`, `occluder == "box_1"`.
  - (c) Viewpoint 0.3 m from the fridge → `limit == TooClose`.
  - (d) Target in the next room behind a wall with a door gap: `half_w = 0` (φ = 0) gives `BehindWall` and lower
    p; `half_w = 0.45` gives higher p.
  - (e) Empty polygon → it answers, with a note `"walls unknown"` and no crash (Review Focus 1).
  - (f) Animate: `seconds_unseen` 0 vs 30 → p decreases.
- [ ] **Step 2: run them and see them fail.**
- [ ] **Step 3: implement `evaluate_camera`.**
- [ ] **Step 4: tests pass.**
- [ ] **Step 5: commit** `perception_card: camera evaluator (coverage, envelope, configuration, animate) + occluder by leave-one-out`.

