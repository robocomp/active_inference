# Door crossing and N rooms joined by doors: brief for the thesis chapter

**For:** the tesis-16 session, which drafts the chapter.
**Written:** 2026-10-12, from the code at commit **`b97c05f9`** (branch `wip/box-estimator-rework`, pushed) and the run folders `room_concept/estimation_scripts/crossing/try*/`.

**How to read this brief**
- Every `path:line` is **at `b97c05f9`**. Paths are relative to `/home/pbustos/robocomp/components/active_inference/`.
- The working tree holds newer, uncommitted edits in `controller/`, `door_concept/` and `chat_agent/`. They are not cited, except where they are named as pending in §10.
- **PROVISIONAL** marks anything still moving.
- Numbers measured through a bug that was fixed later are **removed**, not caveated. §7.1 says which runs are clean, quantity by quantity.

**Commits in this line of work:**

| Commit | Time (10-12) | What it established |
|---|---|---|
| `000e7c66` | 00:12 | First live crossing. Controller door passage, room hand-over, per-room survey after a crossing. |
| `28b6f711` | 00:46 | ltsm passage node; identity = moving `current`; door_concept adopts the crossed door (eviction off). |
| `6a1262fe` | 10:27 | ltsm **evicts** the room left behind; per-beam doorway filter; controller re-resolves the room before every LiDAR poll. |
| `7a6802bb` | 14:45 | Return to room 1 **from memory**: ltsm recall, door leaf carried across, controller step pre-emption. |
| `63265729` | 14:45 | Window geometry only. Irrelevant to the chapter. |
| `b97c05f9` | 19:48 | Repeated round trips; a door recreated on re-entry; passage reuse; controller Mission mode; chat_agent phase 1. |

The commit messages (`git show --stat <c>`) summarise each mechanism and the live finding that forced it. They are the best one-paragraph sources.

---

## 1. Problem

### 1.1 What one room cannot do
The Ch.10 pipeline estimates one room and promotes it:

> start-up proto-room → plain SLAM → free-space map → layout explorer → MDL promotion → refiner

Everything in the fleet is single-room by construction.
- **The controller cannot leave the room.**
  - Its grid is the room polygon's bounding box, and everything outside the polygon is occupied.
  - A target beyond the door is rejected as `OutsideRoom`.
  - The door node is an obstacle box across the opening.
  - The promoted Manhattan layout closes straight across the doorway.
  - Source: the review in `room_concept/docs/superpowers/plans/2026-10-11-cross-door-into-adjacent-room.md`, "Broken or missing, ranked".
- **room_concept cannot start a second estimation.** `begin_room_estimation` left `map_layout_adopted_`/`map_layout_` set. So:
  - `map_ready` never returned;
  - the old polygon stayed;
  - the refiner went on editing room 1 with room-2 scans.

  Same plan, item 3. Fixed in `000e7c66`: `room_concept/src/room_concept.cpp:2358-2394`.
- **Identity was latched.** Controller, residual and door_concept resolved "the room" once and re-looked only when the node vanished. During a hand-over both rooms exist, so they stayed on the old one.
  - door_concept would have born new doors as children of the old room, metres off, and then killed them with the containment prior.
  - See memory note `door-pragmatic-affordances.md`, section "ROOM-FOLLOWING + LET-GO".
- **Nothing remembered a room once the robot left it,** and nothing linked two rooms.

### 1.2 The crossing failure as first seen (2026-10-11 evening, builds that became `000e7c66` and `28b6f711`)
World `webots-shadow/worlds/piso/tworooms-piso.wbt`. Room 1 was loaded as **given**, room 2 is estimated, and DOOR_1 lies between them (§7.2).

| Symptom (run) | Cause | Fixed in |
|---|---|---|
| Room 2 estimated as 53.28 m², 20 vertices, IoU .368 against a 24 m² room (`try4_here`) | Room 1's floor, seen through the open door, flooded room 2's free-space map | `000e7c66`: the left room's outline is excluded from the new room's map (`room_concept/src/free_space_polygon.h:301-310`) |
| Room 2 promoted at 0.22 m², and the same door re-crossed into a `room_3` (`try12`, layout `room_2-…_00-37-14.svg` + `room_3-…_00-39-08.svg`) | At birth the robot's LiDAR origin was still inside room 1's outline, so every beam stopped on "re-entry" into the excluded room | `28b6f711`: a beam that *starts* inside an excluded region crosses it unrecorded (`free_space_polygon.h:823-847`) |
| Corridor ran into the wall; the door re-added as an obstacle | The corridor was anchored on the door node (leaf pose). Exempting the only door object triggered the obstacle tracker's fall-back all-nodes scan, which re-added it | `000e7c66` (commit message; corridor now anchored on the cross standpoint, `controller/src/controller_door_passage.h:300-319`) |
| A promoted room 2 sent the robot straight back through the door | Pragmatic affordances competed in the open contest | `000e7c66`: `aff_kind` nodes are eligible only when a mission names them (`common/affordance_manager/affordance_manager.cpp:734-743`) |
| Stale `current` edges from previous runs blocked ltsm's bootstrap; layout promotion beat the identity decision | Lifecycle ordering | `28b6f711` (commit message) |
| Room 2 published ~8 cm off (`try14`) | Door filter effectively OFF for the whole of room 2: an unmeasured door was treated as shut | `6a1262fe`: per-beam mixture with an even prior (`room_concept/src/door_apertures.cpp:110-122`) |
| Controller deadlocked after eviction (`try15`) | Room deleted before the controller followed `current` → sweeps rejected → LidarStall → `compute()` never re-resolved the room | `6a1262fe`: `controller/src/specificworker.cpp:1469-1475` |
| Re-entered room 1 rotated 172° (`try22`, second `[eviction] DONE` yaw +3.008 rad) | The recall's face is the same doorway turned by π | `7a6802bb`: π side test (`room_concept/src/room_scene_graph.cpp:1463-1482`) |
| Re-entry immediately re-crossed through the same door, birthing `room_3` (`try19`) | Any door could start a crossing, including one hanging under a room not current | `7a6802bb`: only doors under the CURRENT room count (`room_scene_graph.cpp:916-936`) |

Earlier, recorded on 09-16 (memory `proto-room-for-estimation-time-navigation.md`): opening a door caused a symmetry flip and a **false birth** while the robot was still inside. That is the reason crossing evidence must be a model comparison and not a geometric test.

---

## 2. Representation

### 2.1 Rooms, walls, doors in the DSR (live) graph
- **Room node** (type `room`, named `room_<k>`).
  - Attributes: `room_id` = k, `delimiting_polygon_x/y` once promoted, `room_height`, `level`, `parent` = robot.
  - It hangs from the **robot** by an RT edge. The robot→room RT carries the robot pose in that room's frame.
  - Proto birth: `room_concept/src/room_scene_graph.cpp:974-1014`.
- **Proto flag.** A `proto` **self-edge** room→room, not an attribute.
  - It means room_concept's "layout provisional", which was a user decision on 10-11 (`28b6f711` message).
  - Inserted at `room_scene_graph.cpp:1023`; deleted on promotion in place at `:1864`.
  - Consumers test it in `common/room_resolve/room_resolve.h:58-61`.
- **Floor and walls.** room_1 keeps `floor` and `wall_<i>`. Other rooms are prefixed: `room_2_floor`, `room_2_wall_<i>`, mesh `meshes/<room>_floor.obj` (`room_scene_graph.cpp:3695-3714`). Walls are created on promotion (`:1879`).
- **Door node.** Type **`object`**, named `door_<N>`, `object_subtype="door"` (`door_concept/src/door_scene_graph.cpp:92-158`).
  - Its RT parent is the nearest `wall_*` node, falling back to the room (`:146-177`).
  - The RT carries the **aperture** (the static hole), not the leaf (`door_scene_graph.cpp:433-501`; `door_concept/src/door_geometry.h:4-11,64-82`).
  - Not cortex's `door` type: memory `ltsm-abstract-passage-node.md` explains why (the startup sweep reaps it).
- **Door attributes.**
  - Size: `width_m`/`depth_m`/`height_m`.
  - Interaction: `door_phi_rad`, `door_open_prob`, `door_reach_prob`, `door_crossing_progress`, with sentinels −1/−2 for not measured.
  - Epistemic: `epistemic_*`.
  - Health: `free_energy`, `door_detection_*`.
  - Sources: `door_scene_graph.cpp:284-330,503-540`.
  - The `transitable` self-edge is a Schmitt band 0.60/0.35 (`door_concept/src/specificworker.cpp:3738-3781`).
- **Affordances.** Each is a DSR `affordance` node under the door via `has_intention`.
  - Epistemic `aff_door_N` (`common/object_affordance/object_affordance.cpp:196`).
  - Pragmatic `aff_door_N_{approach,open,cross}`, labelled by `aff_kind` (`common/pragmatic_affordance/pragmatic_affordance.cpp:52`).
- **`current` edge.** robot→room, owned **only by ltsm_agent** (`ltsm_agent/src/room_eviction.cpp:552-554`, bootstrap `:873-904`).
  - Readers resolve the room through `common/room_resolve/room_resolve_rule.h:42-80`:
    - rule 1: one `current` edge;
    - rule 2a: no `current` edge, a retired robot RT and exactly one live robot RT;
    - rule 2: the unique non-proto room;
    - rule 3: the unique proto.
- **`has_intention` edges.** They move with the room at a crossing (`room_scene_graph.cpp:1505-1519`). `afford_room`/`afford_calib` are re-parented to the new room at once (b97c05f9).
- **Entry mirror.** door_concept's copy of the crossed door, hung under the new (proto) room (`door_concept/src/specificworker.cpp:3397-3536`). §4.6 has the detail.

### 2.2 Door geometry (door_concept)
- **State.** θ = [s, w, h] in a **wall frame**.
  - s = along-wall offset of the aperture's near edge from wall corner O; centre = O + (s + w/2)·u; yaw = atan2(u).
  - Sources: `door_concept/src/door_belief.h:9-12,40-50,176-178`.
- **Wall normal.** n = (−u.y, u.x) (`door_geometry.h:78`). It is forced **inward** whatever the polygon winding (`door_concept/src/door_fitter.cpp:171-184`, added in `7a6802bb`).
  - Room 1's given map is CW and room 2's estimate is CCW, so before the fix the swing sign meant opposite things in the two rooms.
- **Leaf.** `LeafState{phi, hinge ∈ {Near, Far}, swing = ±1}` (`door_geometry.h:50-52,87-92`). Leaf pose: `door_geometry.h:113-138`.
  - Hinge and swing come from config when `Openable.Enabled`, which is off by default. Otherwise they are Near/+1 (`door_fitter.cpp:65-68`).
  - **Swing is never fitted** (`door_config.h:53-55`).
  - A leaf inherited from a mirror sets hinge and swing (`door_fitter.cpp:536-551`).

### 2.3 N-room topology: frames and ownership
- **Every room has its own frame.**
  - Room 1: given (recentred on its bbox), or frozen at the robot's start pose in Estimate mode.
  - A room born by crossing: **the crossed aperture frame, frozen**. x = tangent a→b, y = outward normal (forced right-handed), origin = aperture centre (`room_scene_graph.cpp:962-972`).
- **Live graph.** All rooms hang under the robot, which hangs under root. The robot→room RT is the only metric link between a room and the robot.
  - At a crossing the old robot→room RT is stamped `valid=false` **once** and frozen (`room_scene_graph.cpp:1039,1173-1192`).
  - From then on room_concept publishes into the new room's RT (`:500-511`).
- **Internal frame.** room_concept's estimator keeps ONE internal frame, continuous across doors. Each published room frame is a fixed gauge `T_pub_int` applied to the internal pose (`room_scene_graph.cpp:500-511`):
  - for a new room, `T_pub_int = T_room_proto⁻¹`;
  - for a recalled room, `T_int_recall⁻¹` (`:1522`).
- **Long-term memory (ltsm_agent).** A second DSR graph on **domain 2**, holding only `root` at start (`ltsm_agent/etc/config.toml:39-49`).
  - **Rooms are siblings under root.** Each hangs from memory `root` with an identity RT that "carries NO metric meaning (there is no global frame), and the relation between two rooms is the passage" (`ltsm_agent/src/room_eviction.cpp:337-345`).
  - **Prefix.** Memory names are `r<idx>_<live name>`, e.g. `r1_room_1`, `r2_door_36` (`room_eviction.cpp:124-157,328-329`).
  - **Persistence.** Saved atomically to `ltsm_agent/etc/ltsm_memory.json` (with a `.prev` rotation) at each eviction fence, after the start-up seed, and on shutdown (`ltsm_agent/src/memory_store.cpp:57-98`; config `[Memory] persist=true`).
  - ⚠ Composition caveat: inner_eigen across two memory rooms composes through root's identities and "returns a WRONG transform rather than nullopt" (`room_eviction.cpp:342-343`).
- **Passage.** `passage_N` is type `metaconcept`, `object_subtype="passage"`, with no pose and no RT edge (`room_eviction.cpp:489-495`).
  - It has `match` edges to both door **faces**, one per room (`:505-516`), and counts `passage_crossings` (`ltsm_agent/src/specificworker.cpp:580-588`).
  - Design note: memory `ltsm-abstract-passage-node.md`. A passage belongs to both rooms and the RT tree is a tree, so connectivity and metric composition are separate structures.
- **Inter-room transform.** It is never stored as an edge.
  - The only cross-room composition function is `seam_through_passage`: T(to←from) = T(to←face_to)·T(face_from←from), with the π flip of §5 (`room_eviction.cpp:572-621`).
  - At `b97c05f9` it has **no production caller**. Its only caller is ltsm's stage check (`ltsm_agent/src/specificworker.cpp:1029`). The recall (§5) composes face transforms itself.

| Transform | Owner | Where |
|---|---|---|
| robot→room RT (pose) | room_concept | `room_scene_graph.cpp:500-511` |
| room frame of a crossed room (frozen gauge) | room_concept | `room_scene_graph.cpp:962-972,1522` |
| wall→door RT (aperture) | door_concept | `door_scene_graph.cpp:433-501` |
| `current` edge (identity) | ltsm_agent | `room_eviction.cpp:552-554` |
| memory rooms, faces, passages | ltsm_agent | `room_eviction.cpp:309-569` |
| door→recall RT, T(face←remembered room) | ltsm_agent | `ltsm_agent/src/room_recall.cpp:113-149` |

---

## 3. Door detection and estimation (door_concept)

### 3.1 Evidence
**(a) Camera masks → geometry.**
- Points deprojected from "door" masks enter a recursive Laplace (Gauss–Newton) update of θ with a two-component point likelihood (`door_concept/src/door_belief.cpp:35-47`):
  - panel: (1−ε)·exp(−d²/2R);
  - clutter: ε·exp(−c²/2R);
  - ε = `AI2ClutterFrac` 0.10, c = `AI2ClutterScaleM` 0.12.
- Per-point R = σ_base² + motion_var + depth_var + (0.02·range)², with `AI2SigmaBaseM` 0.03 (`door_fitter.cpp:597-623`).
- Truncated masks (border fraction > 0.10) or a motion confirm-only verdict make the cycle predict-only (`door_fitter.cpp:606-616`).
- **Mask source. Be careful here.**
  - door_concept consumes the shared "door" semantic masks. Its config comments describe graded ADE20K confidences (`door_concept/etc/config.toml:294-304`).
  - The word "YOLOE" does not appear in door_concept.
  - In the runs, retina's YOLOE-26x-seg household model had `door` added to `accepted_labels`. That is an **uncommitted test config** (`retina/etc/config.toml`, "+door 2026-10-11").
  - Which detector produced the door masks that fed each run is **not recorded**. Write "semantic door masks from the retina", not "YOLOE".

**(b) LiDAR leaf-angle likelihood (φ only).** `phi_field_likelihood`, `door_belief.cpp:435-590`.
- Scene model: a wall plane with the aperture cut out, plus the leaf as an oriented box.
- Each bpearl ray is scored as a mixture:
  - hit a modelled surface: N(ρ; t, σ=0.05);
  - unexpected short return: w_u/t;
  - beyond: uniform;
  - through the opening: uniform past the aperture.
- Log-likelihoods are summed over rays. Depth (9 values) and lateral (3 values) nuisances are marginalised by log-sum-exp (`:568-581`).
- Constants: w_unexp 0.10, eps_beyond 0.03, max_rays 400 (`door_belief.h:270-279`). Ray selection slab: `door_fitter.cpp:934-999`.
- The camera no longer scores φ (since 09-14, `door_fitter.cpp:909-918`).

**(c) RGB contour.** Existence evidence only; positive excess only (`door_concept/src/specificworker.cpp:2175-2184,2452-2456`).

**(d) Depth contour.** Implemented but **disabled**: `ContourDepthCheck=false` (`door_concept/etc/config.toml:142`), against a code default of true (`door_config.h:337`).

### 3.2 Estimator
- **Prior.** Seed Σ = diag(0.60², 0.06², 0.08²) (`door_belief.cpp:141-147`; code defaults `door_config.cpp:251-262`).
  - Template mean w = 1.00 m, h = 2.10 m (`door_config.h:441-446`).
  - A template anchor is added every frame (`door_belief.cpp:116-125`).
- **Constraints.** w, h ≥ 0.05; s ∈ [0, wall_len − w] (`door_belief.cpp:128-138`).
- **Common-mode (Woodbury) caps.** On s: 1/(cm_s² + chain_cov_s). On w, h: 1/cm_wh² (`door_belief.cpp:157-165`).
  - The along-wall term widens with motion × periphery (`MotionCmPosGain` 0.30, `door_fitter.cpp:630-635`).
  - This is the "common-mode covariance instead of a σ-floor" pattern in CLAUDE.md.
- **Ageing.** None: `AI2AgeNominalDtS = 0`, so the belief freezes without masks. φ is still refreshed (`door_fitter.cpp:567-585`).
- ⚠ **Config/code contradictions.** List them as limitations; do not describe the header's intent as fact.
  - The header argues `process_std_s = 0` ("a doorway does not migrate", `door_belief.h:96-114`), but config sets `AI2ProcessStdM = 0.005` m/step (`door_fitter.cpp:48`; `etc/config.toml:64`).
  - The header says `common_mode_wh_std = 0.35`, but config has 0.02 (`etc/config.toml:74`).
- **Published RT covariance** (`door_scene_graph.cpp:341-383`):
  - along-wall σ_s² + chain;
  - across-wall 1e-4;
  - yaw 3e-4 (fixed values);
  - z AI2FloorStd².

### 3.3 φ curve → `door_open_prob`, `door_reach_prob`
- **Grid.** φ ∈ [0, 120°], 5° steps (code defaults `door_config.h:295,299`).
- **Prior.** Gaussian about the commanded angle when an actuation is active, else about φ̂. Its σ compounds with the swing rate (`door_fitter.cpp:861-907`).
- **Posterior.** Normalised LiDAR likelihood × prior (`:1063-1105`).
  - ⚠ The LiDAR curve counts as "measured" only if its coefficient of variation exceeds **0.10** (`:1047-1059`). That is a threshold.
- **`door_open_prob`** = Σ_φ w(φ)·[clear_width(w, φ) ≥ 0.70] / Σ_φ w(φ) (`door_concept/src/door_pragmatics.h:115-130`).
  - 0.70 = robot passage width 0.60 + margin 0.10 (`etc/config.toml:214-215`).
  - clear_width = w·(1 − max(0, cos φ)) (`door_pragmatics.h:107-110`). Since `7a6802bb` it uses cos, not |cos|, so φ > 90° is fully open.
  - It is along-wall only, so it is an upper bound.
  - With no curve: a 17-point Gaussian, σ 0.25 rad, about φ̂ (`specificworker.cpp:2909-2944`). ⚠ It clamps to [0, π/2], which is inconsistent with the 120° grid.
- **`door_reach_prob`** is a band probability, Φ((d_max−d)/σ) − Φ((d_min−d)/σ) (`door_pragmatics.h:134-150`):
  - d_min = w + 0.30 + 0.20 (safe stand-off);
  - d_max = max(2.00, d_min + 0.30) (`specificworker.cpp:2978-2981`);
  - σ² = σ_door,radial² + max(var_pose, 0.10²) (`:2964-2971`).
  - The σ floor keeps it from degenerating into a distance test.

### 3.4 Existence (shared `rc::exist`)
- **Log-odds.** Born at 1.0, capped at ±4.0 (`specificworker.cpp:2087-2091`).
- **Sensor model.** P_D 0.85, clutter 0.05, σ 0.03 (`etc/config.toml:346-348`).
- **Silhouette categories.** Lit by a door mask = occupancy; non-door mask = occluded (HOLD); nothing = absence; out of image = undetectable (`etc/config.toml:325-337`).
- **Weighting.** The update is weighted by **P(detect) = p_env·obliquity·resolvability·in_fov·central** (`specificworker.cpp:2313-2387`). Absence is marginalised over the φ posterior (`door_fitter.cpp:1306-1336`).
- **Removal.** Decision `RemovalProb` 0.12 over a debounce of 15 *looks* (accumulated p_detect, not cycles) (`specificworker.cpp:2472-2485`).
- **Priors** (`specificworker.cpp:2098-2134`):
  - room containment: margin 0.40 m, ΔL −1.5;
  - minimum height: 1.80 m.
- **Birth.** Tracker only, ZED slices only (`etc/config.toml:291-323`), which matches CONCEPT_AGENT_LIFECYCLE.md.

---

## 4. The crossing, step by step

**Ownership.** The controller code is owned by the **controller-a8** session (`controller/`). This brief describes it but did not write it; the chapter should credit it the same way.

### 4.1 Command
- **Grammar.** `cross door` | `cross door_N` | `stop` | `uncross`, parsed by `SpecificWorker::run_mission_command` (`controller/src/specificworker.cpp:1743-1801`).
  - The remote file channel uses the same parser (`:561-566`).
  - **chat_agent** (new in `b97c05f9`, id 25) emits the same closed grammar through a `chat_intention` node, with human confirmation for motion. See `CONCEPT_AGENT_LIFECYCLE.md:222-228`.
- **`cross door`** = the nearest door whose RT chain ends in the **current** room (`common/world_query/door_query.h:45-88`). Doors of a room left behind still resolve through root, so they are filtered out.
- **Acceptance.** It sets Command mode and `DoorPassage::start(door)` (`specificworker.cpp:1792-1793`).

### 4.2 Controller door passage (approach → open → cross)
`controller/src/controller_door_passage.h`:
- **Steps.** `Idle, Approach, Open, Cross, Done` (`:54`). Each maps to `aff_<door>_<step>` (`:276-280`).
  - The affordance state is decoded from `epistemic_pending`/`active` as Offered / Executing / Completed / Missing (`:288-298`).
- **Selection.** The controller's AffordanceManager is restricted to that one name (`required_name`, `common/affordance_manager/affordance_manager.cpp:732-733`). A pragmatic affordance can win only this way (`:741-743`).
- **Pre-emption.** A later step that is offered pre-empts the executing one (`:114-144`), with two rules:
  - `cross` is never taken while `open` is offered;
  - **never Approach → Cross** (`:132`, `b97c05f9`). Comment at `:129-131`: "a shut door can read passable from across the room".
- **Fallback.** If the current step is no longer offered and was never executing, drop back to the latest earlier offered step (`:146-165`). From Cross, go back to Open when the door turns out shut (`:167-178`).
- **Done.** The passage ends when the controller's current room changes during Cross (`:180-192`). That is normally before door_concept's own `cross` completes (0.8 m past the plane).
- **Normal advance.** On `Satisfied`. Other outcomes increment `attempts_` and wait for a re-offer (`:196-218`).
- **Reset.** A stale lock-on is reset when the passage starts or a step is pre-empted (`controller/src/controller_session.cpp:762-794`). Live try23: a stale servo stamped `cross` Satisfied.
- **Corridor on the Cross step.** A door-wide rectangle from 2.2 m behind to 0.8 m beyond the **cross standpoint**, along its heading (`controller_door_passage.h:300-319`; `controller/src/door_passage_corridor.h:27-34`). It is free space for:
  - the grid (`grid_planner.cpp:110-129`);
  - the MPPI boundary (`trajectory_controller.cpp:247-264`);
  - the outside-room standpoint check (`controller_session.cpp:1133`).
  - The crossed door is exempt from the obstacle tracker (`controller_obstacle_tracker.cpp:840,856`).

### 4.3 door_concept's pragmatic offers (the preconditions)
- **Schmitt band.** Every offer goes through Offer 0.60 / Withdraw 0.35 / 3 stable cycles (`common/pragmatic_affordance/pragmatic_affordance.cpp:77-101`; `door_concept/etc/config.toml:228-230`).

| Affordance | Offer probability | Advertised value | Stand-point / contract |
|---|---|---|---|
| approach | 1 when located | 0.5·(1 − P_reach) | Middle of the reach band (1.75 m for w = 1 m); Servo until `door_reach_prob` ≥ 0.90 |
| open | P_reach·(1 − P_open) | 0.3 | Orient until `door_open_prob` ≥ 0.70 |
| cross | P_open | 0.3 | Through-pose 1.20 m beyond the plane; Reach until `door_crossing_progress` ≥ 0.90 |

  - Sources: `door_concept/src/specificworker.cpp:3042-3109`; `door_pragmatics.h:154-231`.
  - The cross side is **latched at the claim** (`specificworker.cpp:3099-3101`, `7a6802bb`). It used to flip in the doorway and turn the robot round.
  - The advertised **values are policy placeholders**, not measurements (`door-pragmatic-affordances.md`).
- **Actuation.** Claiming `open` makes door_concept call `request_door_actuation` → RoboCompDoorControl `requestDoor` on the Webots bridge (`specificworker.cpp:3128-3155`; `door_actuator.cpp:135-153`).
  - A claim on a stale epoch is refused.
  - **In every run** the uncommitted test config set `ActuationProviderId = "DOOR_1"`, confirmed in `try14/23/30` `logs/door_concept.log`. **Every** door was therefore actuated as DOOR_1 (§10).
  - The committed value at `b97c05f9` is `""` (by place).

### 4.4 room_concept: crossing evidence and the birth of the next room
- **P(cross).** Per aperture (`room_concept/src/door_crossing.h:9-19,134-163`):
  - logit p_cross = logit p_open + logit p_geom;
  - p_geom = P(past)·P(span);
  - P(past) = Φ((s − r_body)/σ_s);
  - P(span) = Φ((w/2−u)/σ_t)·Φ((w/2+u)/σ_t);
  - σ_s, σ_t = the pose covariance projected on the outward normal and the tangent;
  - r_body = 0.24 m (`room_config.h:36-37`, `room_scene_graph.cpp:888`);
  - probabilities clamped to [1e-6, 1−1e-6] (flagged in source, `door_crossing.h:47-59`).
  - Only doors under the current room count (`room_scene_graph.cpp:916-936`). There is no crossing evidence while the start-up proto is unpromoted (`:871-874`).
- **Birth.** When max p_cross ≥ **`ProtoRoom.BirthProb` = 0.95** (`room_concept/etc/config.toml:2184`; test at `room_scene_graph.cpp:958-960`, flagged in source as "the one threshold in this path"):
  - a `room_<k>` is born with frame = the aperture (§2.3) and a `proto` self-edge (`:962-1023`);
  - the old robot RT is retired once (`:1039`);
  - `rearm_after_crossing` runs (`:1489-1574`).
  - Measured p_cross at birth in the clean runs: 0.966–0.99999 (`try*/proto_room.csv`, event `born+crossing_born`).
- **`rearm_after_crossing`** does, in order:
  1. Push the left room to `left_rooms_` and its internal-frame outline to `left_polys_int_` (`:1495-1500`).
  2. Delete the old room's `has_intention` edges and re-hang `afford_room`/`afford_calib` from the new room, releasing the stale claim (`:1505-1520`).
  3. Set `T_pub_int` (`:1522`).
  4. `begin_room_estimation` (`room_concept.cpp:2358-2394`): release the adopted layout and **keep the plain-SLAM scan map**.
  5. `arm_startup_proto` (`room_scene_graph.cpp:1582-1633`): fresh `FreeSpacePolygon` (cell 0.05 m, 64 vertices max), exclusion of the left rooms (`:1609-1618`), explorer, refiner and learnt-layout path reset, internal frame frozen.

  The unchanged Ch.10 machinery then surveys and promotes the new room **in place** (`promote_startup_proto`, `:1847-1903`; precondition: layout ready ∧ explorer finished, `:1645-1650`). **This is the main architectural point: a door crossing is a second start-up.**
- **Left rooms are never swept** (`room_scene_graph.cpp:1313-1335`). Room 1's node and walls stay unedited until ltsm evicts them.

### 4.5 Localisation across the door
- **No re-localisation at the door.**
  - The internal frame and plain SLAM continue on the same scan map (`room_scene_graph.cpp:1545-1546`; `room_concept.cpp:2366-2371`).
  - The pose in the new room is the internal pose seen through the new gauge: `T_pub_int·x_int`, with the xy covariance block rotated (`room_scene_graph.cpp:500-511`).
- **SLAM roles** (`room_concept.cpp:3793-3817`).
  - While estimating, plain scan-to-map SLAM *drives* (it replaces the window's newest pose).
  - After a layout is adopted, it runs as a SHADOW.
  - `begin_room_estimation` makes it drive again without re-seeding.
- **Per-beam doorway filter** (`room_concept/src/door_apertures.cpp`). A beam whose ray crosses the aperture mouth is weighted by the posterior that it hit a surface in the aperture rather than the next room (`:159-191`):
  - surface = π·N(d; δ, √(σ_obs² + half_depth²)), with σ_obs = 0.05 (`etc/config.toml:619`);
  - beyond = (1−π)/20 m;
  - the beam is multiplied by overlap·surface/(surface + beyond).
  - **π = max(0, cos φ)** from `door_phi_rad`; otherwise 1 − p_open if measured, **½ if unmeasured** (`:110-122`).
  - The aperture is seated on the room polygon's edge (recess δ, `:133-157`, `7a6802bb`).
  - The weights enter both `evaluate` and `linearize` of the GN solver (`room_gn_solver.cpp:63-80`; `room_concept.cpp:118-154`).

### 4.6 door_concept at the hand-over
- **Follows `current`.** Two `current` edges count as ambiguous and are followed by none (`door_concept/src/specificworker.cpp:3329-3393`).
- **Entry mirror.** One per proto, never into the room the agent is in (`:3443`).
  - Its source is the door whose frame contains the proto origin within ±w/2 (`:3449-3469`).
  - The leaf is captured **at mirror creation**: hinge point and opening direction in door-local terms, φ̂, σ_φ, the φ curve, the match count (`:3489-3507`). ltsm deletes the source with its room, so this must happen first.
  - The mirror survives its source's death (`:3404-3428`).
- **`release_room`** (`:3538-3678`):
  - adopt the entry mirror as the new room's door;
  - **else recreate the lost door nearest the robot from a memo** of every fitted door (pose under root, size, leaf) (`:3570-3580,3882-3937`, `b97c05f9`);
  - doors left in a still-live old room are **HELD**: node kept, affordances dropped (`:3601-3611`);
  - ghosts and the polygon are cleared before the new polygon is loaded;
  - the inherited leaf seeds hinge and swing against this room's inward normal (`door_fitter.cpp:536-551`).
- **Door deleted by another agent.** If ltsm evicts it, its four affordances go too (`specificworker.cpp:3825-3857`, `6a1262fe`).

### 4.7 ltsm: identity promotion and eviction
- **P(inside).** For the proto or entered room (`ltsm_agent/src/room_eviction.cpp:689-755`):
  - P(inside) = Φ((s − r_body)/σ_s), where s = the robot's +y in the proto (aperture) frame and σ_s comes from the RT pose covariance;
  - r_body = 0.5·max(width, depth) of `body` (`ltsm_agent/src/specificworker.cpp:496-505`).
- **Decision.** Promote when P ≥ **`Promotion.decision_prob` = 0.95**. The config flags it as "the one flagged threshold on this path" (`ltsm_agent/etc/config.toml:87-90`). Cycles are not fused (`specificworker.cpp:535-536`).
- **`evict()` transaction** (`room_eviction.cpp:309-569`):
  1. read the seam T(new←old) from the **live** graph through the robot;
  2. copy the old room (all of it);
  3. copy the new room's subtree;
  4. passage: reuse or mint, `match` both faces;
  5. verify;
  6. mark archived;
  7. **move `current`**;
  8. delete the old room from working memory.
- **Copy semantics.** Breadth-first along RT, attributes copied wholesale (`:186-290`). It never removes `proto`, which is room_concept's (`:776-779`).
- **Start-up seed.** Room 1 is seeded into memory, retried until room_concept publishes it (`room_eviction.cpp:787-870`; `specificworker.cpp:340-343,394-417`).
  - ⚠ The seed still chains later rooms off the first one (`:837-848`), which contradicts "siblings under root". It is latent: it only matters with two non-proto rooms at seed time.

---

## 5. Revisits: going back through a remembered door

1. **Recall hypothesis** (`ltsm_agent/src/room_recall.cpp:57-163`).
   - For each live door, ltsm finds its memory face, first by name (`r<idx>_<door>`) and **else by place**: the nearest passage face of this room's memory copy within **0.5·width_m** (`:65-88`, `b97c05f9`). ⚠ Hard-coded threshold.
   - It follows the passage to the OTHER face and that face's memory room. If that room is not live, it posts `recall_<room>`:
     - type `metaconcept`, `object_subtype="room_recall"`, parent = the live door;
     - the remembered outline in `delimiting_polygon_x/y`;
     - an RT door→recall carrying **T(other face ← remembered room)** (`:113-149`).
   - It is retired when not wanted (`:153-163`) and never archived (`room_eviction.cpp:279-282`).
   - **The recall is a pointer, not an inference.** It is posted unconditionally whenever memory says the far side is known. There is no place recognition and no comparison of the remembered room against what the robot sees (§6).
2. **room_concept re-creates the room** (`room_scene_graph.cpp:1427-1487,1557-1573`). When the crossed door carries a recall:
   - **T_int_recall = T_int_door·T_door_recall**, with T_int_door built from the aperture (x = a→b, origin = centre).
   - **π side test:** if the recalled outline's centroid lands on the *current* room's side of the aperture, use T_int_door·Rz(π)·T_door_recall. Rooms do not overlap. Origin: try22, re-entered 172° rotated.
   - The born room takes the remembered **name and `room_id`** (`:992-998`).
   - The outline is mapped into the internal frame, adopted as given, and **promoted at once** (event `born+recalled`). There is no survey.
   - Exclusion is lifted for a left outline inside the recalled one (`:1527-1541`).
   - No fall-back on misfit (flagged v1, `:1560`).
3. **door_concept** gives the re-entered room its door, from the entry mirror or, failing that, the memo recreation (§4.6). A re-entered room is promoted at once, so no proto mirror is made for it. That is why the memo was needed (`b97c05f9`).
4. **ltsm** reuses the passage of the same hole (`room_eviction.cpp:429-467`, `b97c05f9`):
   - a passage already matching this face is reused;
   - else the nearest face of the same memory room within 0.5·width_m is **replaced** instead of minting `passage_1`.
   - `passage_crossings` increments. In try30 it reached 3 on a single `passage_0`.
5. **Inter-room transforms are not corrected.**
   - There is **no loop closure and no transform optimisation**. A search of `ltsm_agent/src` for loop-closure/optimiser terms returns nothing.
   - `EVICTION.md:134-136` describes the two-face disagreement as a loop closure that "grades, and later corrects" the seam. That is aspirational. Do not cite it as implemented.
   - Each crossing re-measures a seam from the live graph. The remembered room is re-placed through the *door*, with whatever error the door and its face transform carry, and it stays wrong until the robot leaves.
   - §7.4 measures the consequence against ground truth: 8.5–29.5 cm, median 14.7 cm, and 0.55–2.1° on re-entry.

---

## 6. Active-inference reading

### 6.1 Where the code does use free-energy or epistemic quantities
- **Epistemic gain in nats.** `aff_door_N` prices a viewpoint as 0.5·log det(I + Σ·ΔI), the expected information gain on θ (`door_concept/src/epistemic_planner.cpp:102-108`).
  - It is detection-weighted, and a proposal is refused when no usable face exists (`:121-146`).
  - This is the epistemic term of expected free energy (Friston et al. 2015, below).
- **Pragmatic offers are probabilities, not tests.** `door_open_prob` is a marginal over the leaf-angle posterior, never its argmax. `door_reach_prob` is a band probability with a pose-σ floor.
  - Offer and withdraw is a Schmitt-band decision on those probabilities (§4.3).
  - Preconditions are beliefs; the band is the decision rule.
- **Existence as accumulated log-odds evidence** (§3.4).
  - Absence is weighted by P(detect) and marginalised over φ.
  - Occlusion holds.
  - This is the CONCEPT_AGENT_LIFECYCLE invariant.
- **Crossing as model comparison.** P(cross) fuses the door's open belief with the geometric "past the line" belief in log-odds (§4.4). ltsm's identity decision is P(inside) with the pose covariance (§4.7).
- **Per-beam doorway filter.** It is a two-hypothesis posterior per beam (surface vs next room), with a prior from the leaf angle (§4.5). It is not a gate: an unmeasured door is an even prior, not "shut".
- **Covariance instead of gates.** Common-mode along-wall covariance grows with motion and periphery (door), and the pose covariance is propagated into P(past) and P(inside).

### 6.2 Where it does NOT fit (say so in the chapter)
- **The passage sequencer is hand-coded.** Approach→open→cross is a fixed ordering with hand-written pre-emption, fallback and "never skip" rules (§4.2). The policy is not chosen by minimising expected free energy over the three affordances.
- **The pragmatic values are policy placeholders** (0.5/0.3/0.3). There is no goal prior that would make "being in room 2" preferred.
- **Decision levels:**
  - `ProtoRoom.BirthProb` 0.95;
  - `Promotion.decision_prob` 0.95;
  - Schmitt 0.60/0.35;
  - contract predicates 0.90/0.70/0.90;
  - the φ "measured" CV 0.10;
  - the half-width 0.5·w matching radius (recall and passage reuse);
  - the π side test (sign);
  - `RemovalProb` 0.12.
  - All are flagged in source. The two 0.95 levels are decisions on a posterior and are defensible as such. The 0.5·w radius and the CV test are plain thresholds.
- **The recall is a pointer, not inference** (§5.1). A proper treatment would be a hypothesis "behind this door is r1_room_1" with a likelihood from the first scans after the crossing, competing with "a new room".
- **No loop closure.** Inter-room geometry is never treated as an uncertain variable. The seam has no covariance and nothing fuses repeated crossings.
- **Leaf swing is never inferred.** Hinge and swing come from config or are inherited, never from data (§2.2).

---

## 7. Experiments

### 7.1 Clean / unclean, per quantity
**Builds.** The runs were made on working trees that became the following commits. Assignment is by time; run start times are in `try*/crossing.log`, commit times in §0:
- try9–11 → `000e7c66`/`28b6f711`;
- try14–15 → `6a1262fe`;
- try16–23 → `7a6802bb` (try23 is the run cited in its message);
- try24–30 → `b97c05f9`.

**Run config for every run** (logged in `logs/*.log`; uncommitted test config, §10):
- `MapMode = given` for room 1 (the newest learnt start-room layout);
- `[StartupProto] Enabled = true`;
- door_concept `ActuationProviderId = DOOR_1`;
- retina `accepted_labels += door`;
- ltsm Eviction and Promotion on from try14.

| Quantity | Clean runs | Removed runs and why |
|---|---|---|
| Room-2 shape IoU, birth→promotion time | **try23–25, 27–30** (last fix on this path: door-filter recess and φ prior, `7a6802bb`) | try4_here (left room not excluded); try12 (beam-start bug); try13 (never collected, cause undiagnosed); try14 (door filter off); try9–11, 15–22 predate the `6a1262fe`/`7a6802bb` filter fixes. Their numbers are listed only as "earlier builds" in §7.3, and **they are not to be used as results** |
| Pose error in room 2 (first visit) | **try23–25, 27–30** | try14 (filter off: 7.1 cm); try15–22 (before the recess fix) |
| Crossing success per command | **try28–30** (never Approach→Cross, `b97c05f9`; observed from try28 onwards) | try20, 21, 26 wedged after jumping Approach→Cross; try22–27 contain that jump |
| Re-entry #2 (room 2→room 1 from memory): pose error | **try23–25, 27–30** | try19 (re-crossed → room_3); try22 (180° flip). Kept with a flagged judgement: try24/25/27/28 have *later-crossing* door/passage bugs, but those do not enter T_int_recall for the second crossing (it uses the crossed mirror and the memory face, §5.2). Drop them if the chapter wants the strict reading |
| Re-entry #3 (room 1→room 2 from memory) | **try30** only | try28 (door not recalled → fresh room); try29 (passage_1 minted: bookkeeping bug, frame plausibly fine but unproven) |
| Passage bookkeeping (one passage, crossings count) | **try30** only | try28/29 minted `passage_1`, fixed in `b97c05f9` |

### 7.2 World, start, ground truth
- **World.** `webots-shadow/worlds/piso/tworooms-piso.wbt`: the simulated apartment ("room 1") plus a 6.0 × 4.0 m room north of it.
  - Room 2's inner faces: x −7.77…−1.77, y 9.48…13.48 (`room_concept/estimation_scripts/crossing/rooms_tworooms.json`; walls `ROOM2_WALL_*`, 0.11 m thick, `.wbt:1745-1834`).
  - **DOOR_1** joins them: `Door` at (−4.769, 9.380), rotation −1.586 rad, size 0.1 × 1.0 × 2.1 m (`.wbt:825-839`). The doorway gap is x −5.30…−4.24.
  - **DOOR_0** is a second door in room 1's inner wall, at (−3.583, 7.166) (`.wbt:810-824`). It was never crossed.
- **Start.** World (−5.0, 6.8, θ = 0) in all of try8–30 (`try*/teleport.txt`).
- **Room 1 frame.** World − (−4.254, 4.628), translation only. The offset is constant in `gt_x_world − gt_x` of every `gt_error.csv`.
- **GT heading convention.** gt_theta − est_theta = **1.5611–1.5618 rad** (circular mean, first 15 s, every run). That is π/2 minus ~0.5°, consistent with the ~0.1–0.8° map rotation already noted in `thesis_results/2026-10-11_bdb6095/NOTE_FOR_THESIS.md`. Pose heading errors below are taken relative to this per-run offset.
- **Which pose the CSV logs.** `est_*` in `gt_error.csv` is the estimator's **internal** pose (`room_concept/src/ground_truth_log.cpp:150-153`). It is continuous through doors, so est − gt is a valid error in all rooms.
  - The evidence: in room 2, heading error stays ~1° while the published room-2 frame is rotated.
  - On a recall, the remembered outline is placed in the internal frame and localised against, so a misplaced recall shows up directly as est − gt.

### 7.3 Results: first crossing and room 2
**Shape IoU.** The estimated outline is scored against the 6 × 4 m GT rectangle after the best rigid fit (24 rotation starts + Nelder–Mead). This is the same definition as in the bdb6095 note.
- Refined outline = the run's `layouts/learnt/room_2-apartamento_layout_<ts>.svg` (mirror_x undone).
- Promoted outline = the chosen row of `try*/promotion_select.csv`.
- Birth→promotion = the `born+crossing_born` → `promoted` timestamps in `try*/proto_room.csv`.

| Run (clean) | p_cross at birth | Birth→promotion (s) | IoU promoted (vertices) | IoU refined (vertices) | Area refined (m²) |
|---|---|---|---|---|---|
| try23 | 0.988 | 10.9 | .999 (4) | .983 (8) | 24.41 |
| try24 | 0.980 | 9.7 | 1.000 (4) | .995 (4) | 24.11 |
| try25 | 0.999 | 44.2 | .998 (4) | .998 (4) | 24.03 |
| try27 | 0.992 | 9.4 | .987 (8) | .985 (8) | 24.34 |
| try28 | 0.986 | 11.3 | .997 (4) | .997 (4) | 24.02 |
| try29 | 0.983 | 8.7 | .994 (4) | .994 (4) | 24.15 |
| try30 | 0.967 | 11.1 | .999 (4) | .992 (4) | 24.20 |
| **median (n = 7)** | | **10.9** | **.998** | **.994**, min .983 | |

- Room 2 is an **empty rectangle**. The 8-vertex outlines add a ~1.2 × 0.1 m notch at the doorway, the recess between the leaf plane and the wall face.
- **Do not generalise these IoUs to furnished rooms.** They show that the per-room pipeline re-arms and runs after a crossing.
- **Earlier builds** (not results; for the history paragraph only), refined IoU:
  - try9 .993, try10 .988, try11 .983;
  - try15 .987, try16 .989, try17 .988, try18 .994, try19 .990, try22 .996.
  - These reproduce the commit messages' .989 (000e7c66) and .987 (28b6f711).

**Pose error vs GT** (`gt_error.csv`, rows more than 3 s after entering a room, room by GT y > 9.43 m).

| Run (clean) | Room 1 before the crossing: median position / heading | Room 2: median position / p90 / heading |
|---|---|---|
| try23 | 2.6 cm / 0.12° | 3.2 / 5.0 cm / 0.92° |
| try24 | 1.8 / 0.24 | 5.2 / 6.4 / 0.65 |
| try25 | 2.7 / 0.14 | 7.1 / 7.3 / 1.73 |
| try27 | 2.5 / 0.23 | 5.2 / 6.0 / 0.86 |
| try28 | 2.8 / 0.41 | 8.3 / 10.8 / 1.76 |
| try29 | 2.5 / 0.24 | 2.7 / 5.3 / 1.22 |
| try30 | 2.3 / 0.33 | 2.1 / 3.1 / 0.75 |
| **median** | **2.5 cm** | **5.2 cm, range 2.1–8.3; heading 0.92°** |

- The room-1 floor of ~2.5 cm is the given (learnt) map's own offset against Webots, not the localiser.
- Room-2 stays are short in try24–30, 50–70 s, because the robot was sent back.
- **About the commit message `6a1262fe`** ("post-crossing pose error 2.0 cm vs ~8.5 cm"). Recomputed with this metric from the archived files `room_concept/tmp/sdf_localizer/gt_error_2026-10-12_09-47-14.csv` (try14) and `…_10-57-10.csv` (try15), it gives **7.1 cm (try14) → 2.6 cm (try15)**. That confirms the direction of the effect.
  - try14 is a filter-off ablation, measured through the bug. Use it only as the motivation for the filter, and say so.
  - try15 is pre-`7a6802bb`. Neither number is a final result.

**Door estimate (PROVISIONAL).**
- door_19's aperture centre in room 1, before the door opens (median of `door_existence_log.csv` cx, try16–30): **−0.585 to −0.611 m**. The GT gap centre from the world file is −0.515 m in the room-1 frame. That puts the along-wall error at **≈ 7–10 cm**, every run on the same side.
- **Across-wall is not comparable.** The estimate sits on the leaf plane, y = 4.614, against a polygon face at y ≈ 4.84 (`door_apertures.h:126-134`).
- **Width is not logged** in the collected files, so there is no width error. Producing one needs a probe of `width_m`.

**Timings.**
- **Final drive leg** (stationary ≥ 3 s → GT crosses the wall line): 5.4–15.6 s over 1.7–3.1 m, all runs. The robot halts while the leaf swings (`ActuationSwingRate` 0.5 rad/s).
- **Command-to-crossing time cannot be recovered.** The controller's `[remote]`/`[door-passage]` lines carry no timestamps.
- The "try16 ~40 s vs try17 ~12 s door stretch" figure from the session is **not re-derivable from the collected files** and is excluded. To report a timing, timestamp the controller lines.

### 7.4 Results: round trips
**Crossing success** (clean: try28–30). Every command that named the real door produced a GT crossing: **9/9**.
- try28: 1→2, 2→1, 1→2.
- try29: 1→2, 2→1, 1→2.
- try30: 1→2, 2→1, 1→2.
- Sources: `logs/controller.log` "passage DONE" and the GT y sign changes.
- **Separate failure.** The generic `cross door` after re-entering room 1 picked a **phantom door_24** in 2 of 2 attempts (try28, try30). Both were stopped by the operator. §10 covers it.
- try26, an earlier build, failed 3/3 by jumping Approach→Cross and wedging. That is the motivation for the `b97c05f9` rule, not a result.

**Re-entry error vs GT.** After the robot re-enters a remembered room, it localises against the remembered outline placed by T_int_recall. So est − gt measures how wrongly the remembered room was put back.

| Run | Re-entry #2 into room 1: median position / p90 / heading | Re-entry #3 into room 2 |
|---|---|---|
| try23 | 15.6 / 16.1 cm / 0.70° | — |
| try24 | 12.5 / 12.7 / 1.00 | — |
| try25 | 21.8 / 22.9 / 0.75 | — |
| try27 | 14.7 / 14.8 / 0.64 | — |
| try28 | 29.5 / 30.0 / 2.11 | (unclean) |
| try29 | 11.6 / 23.2 / 1.03 | 6.0 / 8.2 / 1.79 (provisional) |
| try30 | 8.5 / 14.9 / 0.55 | **2.8 / 3.5 / 1.20** |
| **median #2 (n = 7)** | **14.7 cm, range 8.5–29.5; heading 0.75°, range 0.55–2.11** | |

These are **PROVISIONAL** while the open items in §10 are fixed. The direction is solid: without loop closure, a remembered room is put back 1–3 decimetres off, about 5× the in-room localisation error.

**Logged seams. Read this before quoting them.** ltsm prints, on each eviction, `seam T(new<-old) = (x, y, yaw)` (`ltsm_agent/logs` line format: `ltsm_agent/src/specificworker.cpp:459-468`). The numbers:
- First-crossing seam yaws: −4.95, −0.63, +8.93, +2.53, +4.51, +3.88, +4.34, +2.85, +3.21, +5.62, +2.57, +6.38, +2.96, +0.14° (try14–30).
- The out-and-back composition T(1←2)#2 ∘ T(2←1)#1, which should be identity:
  - try23 22.8 cm / +0.42°;
  - try24 28.6 / +7.19°;
  - try25 40.5 / +7.45°;
  - try27 17.6 / +1.66°;
  - try28 11.7 / +5.80°;
  - try29 34.6 / +8.53°;
  - try30 49.2 / −0.51°.
- This "1–8° re-entry rotation" is what the `b97c05f9` message quotes.

**However, the logged seam is very likely not a measurement of the inter-room transform:**
1. It is composed through the robot (`room_eviction.cpp:316-324`). The robot→old-room RT was frozen at birth, and the robot keeps moving until the eviction.
2. Room 2's frame axis is the aperture tangent of a door attached to room 1's given wall (§2.3, §3.2). Its true yaw relative to room 1 is ≈ 0.5° at most, yet the logged yaws reach +8.9°.
3. The GT-measured re-entry heading error never exceeds 2.1°, while the logged loop residual reaches 8.5°.
4. The recall does not use the logged seam (§5.2).

So report the GT-based re-entry error. Describe the logged seam as a diagnostic that, as written, mixes the robot's motion after RT retirement into the transform. This diagnosis is **PROVISIONAL**: it is structural, and it has not been checked by time-aligning the ltsm log, which carries no timestamps, with GT.

### 7.5 Data provenance
- **Run folders:** `room_concept/estimation_scripts/crossing/try*/`. Each holds `crossing.log`, `logs/{room_concept,door_concept,controller,ltsm_agent,residual_concept,retina[,chat_agent]}.log`, `gt_error.csv`, `proto_room.csv`, `door_filter.csv`, `promotion_select.csv`, `map_layout.csv`, `layout_refine.csv`, `door_events.csv`, `door_existence_log.csv`, `slam_shadow.csv`, `refine_input.bin` and `events.jsonl`. try13–15 hold only `crossing.log` + `logs/`.
  - The folders are **not committed**; only `rooms_tworooms.json` is.
- **Launcher:** `room_concept/estimation_scripts/scripts/run_crossing.sh` @b97c05f9: teleport, start the fleet, `collect`, `CHAT=1`.
- **The analysis behind §7.3–7.4 is not in the repo.** These three steps re-create it in a few lines:
  1. **Pose error:** est vs gt per GT room; heading offset = the circular mean of gt_θ − est_θ over the first 15 s; skip 3 s after each room entry.
  2. **Shape IoU:** best rigid fit of the GT rectangle to the outline, from 24 rotation seeds.
  3. **Seams:** a regex over `[eviction] DONE` lines, composing SE(2).

---

## 8. Figures
**Existing:**
- **Learnt room-2 outlines,** one per run: `layouts/learnt/room_2-apartamento_layout_2026-10-12_<hh-mm-ss>.svg`. Clean runs: 14-40-24, 14-57-02, 17-31-00, 17-59-44, 19-15-13, 19-25-14, 19-32-02. The second visits are 19-27-29 and 19-40-31.
  - Copies are in each `try*/`, which also contain older files: pick the one inside the run's time window.
- **Room-1 learnt layouts** `apartamento_layout_*.svg` in the same folder.
- **Camera snapshots:** `try7/snapshots/{ricoh,zed}_1791753485*.png`, only for try7, which was not a crossing run.
- **`try*/door_filter.csv`.** Per scan: objects, named door, believed open, emitted aperture, p_open, a/b endpoints, prior_closed. It gives a time series of the doorway filter's inputs.
- **`try*/gt_error.csv`** for trajectories and error traces. `try*/proto_room.csv` carries p_open/p_past/p_span/p_cross per cycle and the birth/promotion events.

**Cheap to produce, recommended:**
1. **Two-room panel, try30.**
   - Content: GT rooms (green) + the promoted/refined room-2 outline (black/red) aligned by its shape fit + room 1's given map + the GT trajectory coloured by current room, with crossing points marked.
   - Inputs: `try30/gt_error.csv` and the two SVGs.
2. **p_cross anatomy at one crossing.** p_open, P(past), P(span), p_cross against time from `try30/proto_room.csv`, around the `born+crossing_born` row, with the 0.95 level drawn.
3. **Re-entry error bar chart.** Room 1 before / room 2 / room 1 after re-entry / room 2 after re-entry, median position error per clean run (the §7.4 table). It shows the step at the recall that loop closure would remove.
4. **Graph diagram (TikZ).** Live graph: root→robot→{room_1 (retired RT), room_2 (live RT, proto self-edge)}→walls→door; `current`; `aff_door_N_*`. Memory graph: root→{r1_room_1, r2_room_2}, passage_0 with `match` edges to r1_door_19 and r2_door_40, plus recall_room_1 under the live door.
5. **Sequencer state diagram.** Approach/Open/Cross/Done with pre-emption, fallback and the "never Approach→Cross" edge.

---

## 9. Designs tried and abandoned
- **VRT seam edge, "option B"** (decided 10-11). room_concept would write a temporary VRT edge crossed-door → new room carrying the seam; door_concept would use it to match the door seen from room 2. **Superseded** by the ltsm passage node with `match` edges, plus door_concept's entry mirror (`28b6f711`). Memory `door-crossing-plan-2026-10-11.md`.
- **Cross-linked door pair** (`r0_door_1` ↔ `r1_door_1`, `other_side_door_name`). Replaced by the abstract passage node, which gives the history one owner (`ltsm-abstract-passage-node.md`).
- **`exit` edge in memory.** Removed in `6a1262fe` (`room_eviction.cpp:502-504`).
- **Chained memory rooms** (`root → r0_room --RT[seam]--> r1_room`). Replaced by siblings under root (user decision 10-12, `room_eviction.cpp:337-341`). `EVICTION.md:88-130` still describes the chain and is stale.
- **Identity-only promotion with eviction OFF** (`28b6f711`). It kept room 1 live and its doors HELD. Superseded by eviction ON (`6a1262fe`). The code path remains (`ltsm_agent/src/specificworker.cpp:604-635`).
- **Unmeasured door treated as shut** in the doorway filter. try14, ~8 cm bias. Replaced by the even prior `6a1262fe`, then by the φ-based prior `7a6802bb`.
- **π from passability** (1 − p_open). Wrong for a beam: a door ajar at 40° is impassable yet leaves part of the aperture open. Replaced by max(0, cos φ) (`door_apertures.cpp:110-119`).
- **Nearest door for "cross door".** Kept at `b97c05f9` but known wrong when a room has two doors. A pending change refuses and lists the doors (§10).
- **Jumping Approach→Cross** when cross was offered. Allowed in `7a6802bb`, forbidden in `b97c05f9`. A shut door read passable from afar and the robot wedged on the approach (try20, 21, 26).
- **Corridor anchored on the door node.** The node carried the leaf pose and the corridor ran into the wall. Re-anchored on the cross standpoint (`000e7c66`).
- **Old door-proto path.** The pre-10-11 crossing branch only called `begin_room_estimation` and fed none of the start-up machinery. Replaced by `rearm_after_crossing` = a second start-up. The legacy branch remains when `[StartupProto]` is disabled (`room_scene_graph.cpp:1055-1062`).
- **Promotion on `map_ready` alone** (start-up, 10-09). It promoted on the first parked view. Now it requires the explorer to be finished (`room_scene_graph.cpp:1645-1650`).

---

## 10. Not solved (state as limitations)

| # | Open item | Evidence | Status |
|---|---|---|---|
| 1 | **The room-1 face of the passage is unpaired after a re-entry.** `matches r2_door_N ↔ <none>` on every eviction #2. The recall path makes no entry mirror for room 1 | `try23–30/logs/ltsm_agent.log` | Open |
| 2 | **door_concept rates a shut door passable from afar** (`door_open_prob` high at range). This was the reason for "never Approach→Cross" | `b97c05f9` message; try20/21/26 | Open, worked around in the controller |
| 3 | **Re-entry error 8.5–29.5 cm / 0.55–2.1° (GT); logged seam loop 0.4–8.5°.** No loop closure, no seam covariance, nothing fused over crossings. The logged seam is composed through a frozen robot RT (§7.4) | §7.4 | Open |
| 4 | **The identity table is not room-keyed.** A door-like detection mid-room-1 after re-entry (phantom `door_24`) kept an old identity and won `cross door` as nearest | try28/try30 controller logs; `door_concept/src/specificworker.cpp:3623-3631` flags it | Open. Pending uncommitted changes recycle the lowest free `door_<N>` name and make "cross door" refuse when a room has more than one door |
| 5 | **Door actuation hard-wired to one Webots door.** `ActuationProviderId = "DOOR_1"` in the runs sent every door, including DOOR_0, to DOOR_1 | `logs/door_concept.log`; working-tree `door_concept/etc/config.toml` comment | **Being fixed, uncommitted:** per-door provider choice by nearest advertised door, `ActuationMatchMaxM = 1.0` (⚠ a gate) |
| 6 | **Approach standpoint outside the room** for a door on an inner wall (DOOR_0) | — | Being fixed (controller-a8 / door_concept), uncommitted |
| 7 | **Two-room world only.** No N ≥ 3 test, no corridor, and DOOR_0 was never crossed | — | Open |
| 8 | **The recall is not inference** (§6.2). A wrong recall has no fall-back (`room_scene_graph.cpp:1560`) | — | Open |
| 9 | **Leaf swing never inferred;** a φ fallback clamp at π/2 vs a 120° grid; config/header contradictions on process noise and common-mode width (§3.2) | — | Open |
| 10 | The memory seed chains later rooms off the first, contrary to siblings-under-root | `room_eviction.cpp:837-848` | Latent |
| 11 | **Run config is uncommitted.** MapMode = given, ActuationProviderId = DOOR_1, retina `+door`. Memory persists across runs (`ltsm_memory.json`), and whether each try started from a clean memory file is not recorded | `git diff` of the three configs | Note in the protocol |
| 12 | **No campaign.** The plan's Phase 5 (10 starts × autonomous crossing, two-room synthetic generator) was never run. Every number here comes from single development runs from one start pose | — | Open |

---

## References (only what can be named precisely)
- **Expected free energy / epistemic value.** K. Friston, F. Rigoli, D. Ognibene, C. Mathys, T. Fitzgerald, G. Pezzulo, "Active inference and epistemic value", *Cognitive Neuroscience* 6(4):187–214, 2015.
- **Active inference process theory.** K. Friston, T. FitzGerald, F. Rigoli, P. Schwartenbeck, G. Pezzulo, "Active inference: a process theory", *Neural Computation* 29(1):1–49, 2017.
- **Laplace approximation in variational free energy.** K. Friston, J. Mattout, N. Trujillo-Barreto, J. Ashburner, W. Penny, "Variational free energy and the Laplace approximation", *NeuroImage* 34(1):220–234, 2007.
- **Log-odds evidence accumulation and occupancy.** S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
- **What we do NOT do: incremental smoothing with loop closure.** M. Kaess, A. Ranganathan, F. Dellaert, "iSAM: Incremental Smoothing and Mapping", *IEEE Trans. Robotics* 24(6):1365–1378, 2008.
- **Per-room local frames joined by a topology (closest prior art for "rooms as submaps").**
  - M. Bosse, P. Newman, J. Leonard, S. Teller, "Simultaneous localization and map building in large-scale cyclic environments using the Atlas framework", *IJRR* 23(12):1113–1139, 2004.
  - B. Kuipers, "The Spatial Semantic Hierarchy", *Artificial Intelligence* 119(1–2):191–233, 2000.
- **Scene graphs with rooms.**
  - I. Armeni et al., "3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera", ICCV 2019.
  - A. Rosinol, A. Gupta, M. Abate, J. Shi, L. Carlone, "3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans", RSS 2020.
  - N. Hughes, Y. Chang, L. Carlone, "Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization", RSS 2022.
- **Citations needed (do not invent):**
  - DSR/CORTEX shared graph (RoboComp);
  - YOLOE (the detector family) — only if the chapter keeps the detector name;
  - ADE20K, if the mask source is attributed to it;
  - Woodbury common-mode marginalisation: standard, cite a linear-algebra text if wanted;
  - Schmitt trigger: standard, no citation needed.
