# Interaction Anchor — Robot-Object Pose for Any Affordance (door crossing first)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While an object affordance executes, the robot-object relation is a tracked pose of its own; it does not pass through room frames.
- The owning agent hangs a **shallow copy** of the object from the robot node.
- It keeps that copy's RT edge (pose + covariance) predicted by odometry and corrected by the robot's own sensors.
- The affordance names that copy as **the node defining the robot-object pose**, and the controller executes the affordance against that edge.
- The door crossing is the first user.

**Architecture:**
- **One shared module, `common/interaction_anchor/`.** It holds:
  - the relative-pose belief: robot-frame x, y, yaw and a 3×3 covariance;
  - its odometry prediction;
  - the DSR lifecycle: create the copy under the robot node (`Shadow`, type `robot`), write the stamped RT edge with covariance, link the copy to its original and to the affordance, remove it, and sweep stale copies.
- **The owning agent supplies the object-specific likelihood.**
  - For door_concept this is the rays through and on the wall plane around the opening, in the robot frame, plus the door mask through the static zed mount.
  - It minimises free energy: the prediction is the prior, the sensors are the likelihood, and the result carries a Laplace covariance.
- **The controller reads the anchor named by the affordance** and drives a robot-frame servo while that anchor exists. The cross step plans nothing in the room frame.
- **The room frame is used once, to initialise the anchor,** as robot←room ∘ room←object at the moment the affordance starts executing. After that, only odometry and sensors move it.

**Tech Stack:** C++23, Eigen, DSR/cortex (RTBlock edges, type-attributed attributes), standalone g++ harnesses as in `door_concept/tests/`.

**Spec:** this conversation, 2026-10-18 (user):
- "It is a robot-object relation, not mediated by the robot in room pose or the object in room pose. It is like the door hangs from the robot during the crossing."
- "this has to be a general procedure to interact with any object, move a shallow copy to hang from the robot and indicate in the affordance that that is the node defining the robot-object pose."
- "when the door is open and it is hard for the camera to detect, the door model can predict that the lidar will pass through, check it and use it as a local evidence"
- "for an open door the lidar has to go through, and viceversa"
- "verify … using both the LIDAR and the camera"

Evidence: try63 leg 2. room_2's door_1 was fitted 0.4–0.5 m left of the real opening (σ_s 0.04–0.12, free energy ≈ 9.9 against a normal 1.5), and the room-frame crossing drove into the wall twice.

## Global Constraints

- CLAUDE.md:
  - No thresholds: use precisions and covariances that grow with the physical covariate, and flag any decision level that remains.
  - No constant process noise (lifecycle contract). Prediction noise comes from `robot_current_speed_variance` × dt.
  - Use type-attributed attribute access only, never `runtime_checked_*`.
  - Connect signals Queued, never `Qt::DirectConnection`.
  - Always check the optional returned by `get_transformation_matrix`.
  - SIGTERM cleanup must delete the anchor (`cleanup_owned_nodes`), and a startup stale-sweep reaps leaked anchors.
  - Parse numbers with `std::from_chars`. Use `and`/`or`/`not` and C++23.
  - Build with `-j8` after checking `pgrep -a cc1plus`.
- One new attribute: `interaction_node_id` (`std::uint64_t`) on the affordance node, naming the anchor. It is registered in cortex `dsr_attr_name.h`, and **the user reinstalls cortex**. The copy's link back to its original reuses the existing `parent` attribute (`std::uint64_t`).
- An anchor is a shallow copy:
  - type `object`, with the original's `object_subtype` and its `width_m`, `height_m` and `depth_m` where present;
  - name `<original>@robot`, `parent` = the original's id, `agent_id` = the owner;
  - one RT edge, `Shadow` → anchor;
  - no beliefs, affordances or children of its own.
- These existing tests keep passing:
  - `door_concept/tests/test_door_phi_stability.cpp`
  - `door_concept/tests/test_door_standpoint.cpp`
  - `DoorBelief::self_test`
  - `controller` `bin/route_bench --self-test`
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Room switch mid-crossing.** room_concept changes the current room while the robot is in the doorway, and door_concept re-parents or re-creates the door for the new room.
   - The anchor hangs from `Shadow` and must be untouched by this. Its `parent` id may point at a node that is about to vanish.
   - The servo keeps driving. Only `relink` refreshes `parent` once the door is re-identified.
2. **Readers that enumerate doors by `object_subtype` would double-count the anchor.** Each of these must skip `rc::interaction::is_anchor(G, n)`:
   - ltsm `passage_harvest.cpp:21` and `room_eviction.cpp:28`
   - episodic `dsr_fact_source.cpp:47`
   - chat_agent `world_digest.cpp`
   - graph3d `graph3d_scene_builder.cpp:144`
3. **The opening leaves the sensors' view in or past the doorway.** The update becomes prediction-only, the covariance grows with the odometry variance, and the servo continues. It must not stop because "nothing is measured".
4. **An agent crashes or is SIGKILLed mid-crossing.**
   - A leaked anchor must not steer the next run. `sweep_stale` deletes the anchors that carry the agent's own `agent_id`.
   - The controller ignores an anchor whose edge timestamp is older than 1 s, falls back to the room-frame route, and logs it. This is a flagged decision level: a stale edge means a dead owner, not a weak measurement.
5. **A leaf held at 90° on the robot's side.** Rays ending on the leaf panel stop short of the wall plane. They count as neither through nor blocked, which is the rule `select_doorway_rays` already applies, so the leaf cannot pull the opening sideways.

---

### Task 1: Relative-pose belief and odometry prediction (pure)

**Files:**
- Create: `common/interaction_anchor/relative_pose.h`
- Test: `common/interaction_anchor/tests/test_relative_pose.cpp`

**Interfaces:**
- Consumes: `rc::rt::Twist` from `common/rt_twist/rt_twist.h:42`.
  - `linear = {side, adv, 0}` in the body frame, `angular.z = rot`, plus `stamp_ms`.
  - The twist variance is `[var_adv, var_side, var_rot]` from `robot_current_speed_variance`.
- Produces:
  ```cpp
  namespace rc::interaction {
  struct RelPose {                    // the object frame expressed in the robot frame
      Eigen::Vector3f x{0,0,0};       // x, y (m), yaw (rad)
      Eigen::Matrix3f P = Eigen::Matrix3f::Identity();
      std::int64_t stamp_ms = 0;
  };
  // Moves the object frame by the inverse of the robot's motion over dt; P <- J P Jᵀ + G Σ_twist Gᵀ dt.
  RelPose predict(const RelPose& r, const rc::rt::Twist& tw, const Eigen::Vector3f& twist_var, float dt_s);
  // Initialisation, the only use of room frames.
  RelPose from_room(const Eigen::Affine3d& room_T_robot, const Eigen::Affine3d& room_T_object, const Eigen::Matrix3f& P0);
  }
  ```
- [ ] **Step 1: Write the failing tests.**
  - `pure_advance`: object at (0, 2, 0); adv = 0.5 m/s for 1 s. Expect x ≈ (0, 1.5, 0).
  - `pure_turn`: object at (0, 2, 0); rot = +π/2 rad/s for 1 s. Expect x ≈ (2, 0, −π/2) to 1e-4. This uses the robot frame: forward is +y, side is +x (ROBOT_GEOMETRY.md).
  - `covariance_grows_only_with_motion_variance`: twist_var = 0 leaves P unchanged; twist_var > 0 makes its trace strictly larger.
  - `from_room_round_trip`: robot (1, 1, 0.3), object (2, 4, 1.0) in the room. Composing back recovers the object's room pose to 1e-5.
- [ ] **Step 2: Run** `g++ -std=c++23 -O1 -I/usr/include/eigen3 -Icommon common/interaction_anchor/tests/test_relative_pose.cpp -o /tmp/trp && /tmp/trp`. Expected: compile FAIL.
- [ ] **Step 3: Implement** both functions in the header. Wrap yaw to (−π, π].
- [ ] **Step 4: Run.** Expected: `ALL PASS`.
- [ ] **Step 5: Commit** `common: interaction_anchor relative pose + odometry prediction`.

### Task 2: Anchor lifecycle in the graph (+ the new attribute)

**Files:**
- Modify (cortex; the user reinstalls): `core/include/dsr/core/types/type_checking/dsr_attr_name.h`. Add `REGISTER_TYPE(interaction_node_id, std::uint64_t, false)  // affordance → the node defining the robot-object pose`.
- Create: `common/interaction_anchor/interaction_anchor.h`, `common/interaction_anchor/interaction_anchor.cpp`
- Modify: the 5 readers in Review Focus 2.

**Interfaces:**
- Consumes:
  - `RelPose` from Task 1.
  - The `RTBlock` overload of `insert_or_assign_edge_RT`, as used at `room_concept/src/room_scene_graph.cpp:811` (stamped, with covariance).
- Produces:
  ```cpp
  namespace rc::interaction {
  class Anchor {
  public:
      // Shallow-copies `object` under the robot node; sets affordance.interaction_node_id to the copy.
      static std::optional<Anchor> create(DSR::DSRGraph& G, std::uint64_t object_id, std::uint64_t affordance_id,
                                          int agent_id, const RelPose& init);
      void write(DSR::DSRGraph& G, const RelPose& r);              // the stamped RT edge Shadow → anchor
      void relink(DSR::DSRGraph& G, std::uint64_t new_object_id);  // `parent` after re-identification
      void remove(DSR::DSRGraph& G);                               // node, edge, and the affordance's link
      std::uint64_t id() const;
  };
  bool is_anchor(DSR::DSRGraph& G, const DSR::Node& n);            // RT parent of type "robot" and a `parent` attr
  void sweep_stale(DSR::DSRGraph& G, int agent_id);                // startup: delete own leaked anchors
  std::optional<RelPose> read_affordance_anchor(DSR::DSRGraph& G, const DSR::Node& affordance);  // controller side
  }
  ```
- [ ] **Step 1: Ask the user to add the attribute and reinstall cortex.** Code using `interaction_node_id_att` will not compile until that is done.
- [ ] **Step 2: Implement** `interaction_anchor.cpp`, then add it to door_concept's and the controller's `src/CMakeLists.txt`. Reconfigure clean (`rm build/CMakeCache.txt`).
- [ ] **Step 3: Patch the 5 readers** to `continue` or `return` on `rc::interaction::is_anchor(G, n)`.
- [ ] **Step 4: Build** door_concept, the controller, ltsm_agent, episodic_agent and chat_agent at `-j8`. Expected: all build.
- [ ] **Step 5: Commit** `common: interaction anchor — a shallow copy of the object hangs from the robot; the affordance names it`.

### Task 3: Door opening likelihood in the robot frame (pure)

**Files:**
- Create: `door_concept/src/door_anchor_update.h`
- Test: `door_concept/tests/test_door_anchor_update.cpp`

**Interfaces:**
- Consumes: `RelPose` from Task 1; the door's `width_m` and wall thickness.
- Produces:
  ```cpp
  namespace rc::door {
  struct AnchorEvidence {
      int n_through = 0, n_blocked = 0, n_short = 0;   // ray classes against the predicted opening
      float p_open = 0.5f;                              // (n_through+1)/(n_through+n_blocked+2), rays inside the span
      float camera_nis = -1.f;                          // mask centroid's normalised innovation; -1 = no mask
  };
  // One free-energy minimisation: prior = the predicted RelPose; likelihood = rays (+ mask). Gauss-Newton on
  // (x, y, yaw), Laplace covariance.
  std::pair<rc::interaction::RelPose, AnchorEvidence>
  update_door_anchor(const rc::interaction::RelPose& prior, float width, float thickness,
                     const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                     const std::vector<Eigen::Vector3f>& mask_pts_robot);
  }
  ```
- **Ray model.** The anchor's x-axis is the wall line, with an opening of width `w` centred at its origin. Each ray is judged by where it crosses that line:
  - Inside the opening, the ray must end beyond the plane: a soft logistic on the signed depth.
  - Outside the opening, it must end on the plane: a Gaussian with σ = 0.05 m, the beam range noise used in `door_phi_posterior.h`.
  - A ray ending short of the plane is uninformative (the leaf, a person).
  - The jambs pull the anchor along the wall; the wall pulls its depth and yaw.
- **Mask model.** The along-wall centroid of the mask points is a Student-t observation (ν = 3) of the opening centre. Its σ is the projected spread / √min(N, 30). A misplaced mask loses weight through the tail, without a gate.
- [ ] **Step 1: Write the failing tests.** Use synthetic ray casts, as in `test_door_phi_stability.cpp`, with a 1.0 m opening and the robot 2.5 m from the wall.
  - `open_corrects_lateral_offset`: prior 0.45 m to the side with σ 0.5. After one update, the posterior's lateral error is < 0.05 m and `p_open` > 0.9.
  - `no_camera_still_converges`: the same with an empty mask, same bounds. This is the user's case: an open door the camera misses.
  - `leaf_rays_ignored`: a leaf at 90° on the robot's side. Expect `n_short` > 0 and a posterior within 0.05 m of the previous case's.
  - `shut_door`: a flush leaf gives `p_open` < 0.1. Depth and yaw are still corrected, but the lateral σ stays ≥ 0.8 × the prior's, because a flat wall says nothing along itself.
  - `misplaced_mask_downweighted`: the mask centroid 0.6 m off the opening. The posterior is within 0.08 m of the truth and `camera_nis` > 9.
  - `no_data_is_prior`: no rays and no mask return the prior exactly.
- [ ] **Step 2: Run** `cd door_concept && g++ -std=c++23 -O1 -I/usr/include/eigen3 -Isrc -I../common tests/test_door_anchor_update.cpp -o build/test_door_anchor_update && ./build/test_door_anchor_update`. Expected: compile FAIL.
- [ ] **Step 3: Implement** at most 5 Gauss-Newton iterations; the covariance is the inverse Hessian at the optimum.
- [ ] **Step 4: Run** this test and the existing door tests. Expected: all PASS.
- [ ] **Step 5: Commit** `door_concept: door anchor update — rays through/on the wall plane and the mask, in the robot frame`.

### Task 4: door_concept owns the anchor during a passage

**Files:**
- Modify: `common/lidar_ingestor/concept_lidar_ingestor.{h,cpp}`. Add `const std::vector<Eigen::Vector3f>& sweep_bpearl_robot()` and `Eigen::Vector3f origin_bpearl_robot()`. These come from a second poll of the reader with target frame `"Shadow"`, since `lidar_plane_reader.cpp:161` already takes the target frame.
- Modify: `door_concept/src/specificworker.cpp`, the affordance block around lines 3260–3285.
  - **Create** the anchor when `aff_door_N_approach` or `aff_door_N_cross` starts executing (its `executing_epoch_att` changes), initialised with `from_room`.
  - **Each cycle:** `predict` from `rc::rt::robot_body_twist`, then `update_door_anchor`, then `write`.
  - **Mask points:** `MasksPacket::support_points_cam` (zed frame) through the static robot←zed mount.
  - **Remove** the anchor when the cross affordance completes or is withdrawn.
  - **Room changes:** room-change handling never touches the anchor. It calls `relink` when the passage door is re-identified in the new room.
  - **Logging:** one row per cycle to `etc/anchor_log.csv`, written with the classic locale: `t_ms,x,y,yaw,sx,sy,syaw,n_through,n_blocked,n_short,p_open,camera_nis`.
- Modify: `door_concept/src/specificworker_presence.cpp:94`.
  - `cleanup_owned_nodes` removes the anchor.
  - `initialize()` and the first `on_operating_enter` call `sweep_stale`.
- [ ] **Step 1: Build** at `-j8`. Expected: built.
- [ ] **Step 2: Live.** Start the fleet with the robot in room_1 facing door_1, then send `cross door_1`. Expected:
  - a `door_1@robot` node under `Shadow`;
  - `aff_door_1_approach.interaction_node_id` equal to that node's id;
  - `anchor_log.csv` rows with `n_through` > 0 once the door is open.
- [ ] **Step 3: SIGTERM door_concept mid-approach.** Expected: the anchor is gone. Restart door_concept; nothing stale remains.
- [ ] **Step 4: Commit** `door_concept: hang the passage door from the robot while its affordance executes`.

### Task 5: The controller executes the affordance against its anchor

**Files:**
- Create: `controller/src/controller_anchor_servo.h`
- Test: `controller/tests/test_anchor_servo.cpp`
- Modify: `controller/src/controller_session.cpp`.
  - In the Cross step (`drive_point_target`, around line 2780), when `read_affordance_anchor` returns a pose, command the base from the servo and skip the room route.
  - Delete the live-gap probe and the `live_clearance` parameter of `smooth_plan` added in `f3017c04`.
  - Keep the door-axis anchors for the approach.
- Modify: `controller/src/controller_session.h`. Drop `passage_door_width_`.

**Interfaces:**
- Consumes: `read_affordance_anchor` from Task 2.
- Produces:
  ```cpp
  namespace rc {
  struct AnchorCommand { float adv = 0, side = 0, rot = 0; bool through = false; };
  // side nulls the lateral offset e to the opening's normal line, rot aligns the heading with it, adv scales with
  // P(body fits) = Φ((w/2 − body_half_width − |e|)/σ_lat). `through` once the opening is `standoff` behind along it.
  AnchorCommand anchor_cross_command(const rc::interaction::RelPose& r, float width, float body_half_width,
                                     float standoff, float v_max, float w_max);
  }
  ```
- [ ] **Step 1: Write the failing tests.**
  - `centred_goes_straight`: opening 2 m ahead, centred. Expect `side` ≈ 0, `rot` ≈ 0, `adv` > 0.8 v_max.
  - `offset_slides_sideways`: lateral offset 0.3 m. Expect a `side` that reduces it, and a lower `adv` than when centred.
  - `uncertain_slows`: σ_lat 0.3 gives a lower `adv` than σ_lat 0.03.
  - `through_after_standoff`: the opening 2.1 m behind with standoff 2.0 gives `through == true`.
- [ ] **Step 2: Run** `cd controller && g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc tests/test_anchor_servo.cpp -o /tmp/tas && /tmp/tas`. Expected: FAIL.
- [ ] **Step 3: Implement** the servo and the Cross-step switch. The collision guard and the STUCK recovery stay in force on the servo's commands.
- [ ] **Step 4: Run** the tests and `bin/route_bench --self-test`, then build at `-j8`. Expected: all PASS, built.
- [ ] **Step 5: Commit** `controller: cross through the affordance's anchor in the robot frame; drop the live-gap probe`.

### Task 6: Live acceptance — three round trips

- [ ] **Step 1:** Restart the fleet, then run `bash room_concept/estimation_scripts/scripts/round_trips.sh try64`.
- [ ] **Step 2:** Expected:
  - 6/6 legs;
  - `anchor_log.csv` shows lateral σ < 0.05 m before the robot reaches the opening on every leg;
  - `controller.log` has no `[recovery] STUCK` between the end of the approach and `passage DONE`;
  - no `@robot` node remains after a leg.
- [ ] **Step 3:** If a leg fails, record the leg, the anchor log around the failure, and the stall position, then stop. Do not patch the room-frame route around it.

**Next users of the module (not in this plan):** chair sit, table approach, bottle grasp. Each brings only its own likelihood (Task 3's role) and calls the same create, predict, write and remove.
