# ARCHITECTURE — the `active_inference` CORTEX stack

A map of the whole system: what each layer is, what each agent believes, and how they exchange
evidence. It is descriptive — it says what the code does today, not what it should do. The
*normative* documents are [`CLAUDE.md`](CLAUDE.md) (hard rules that prevent crashes) and
[`CONCEPT_AGENT_LIFECYCLE.md`](CONCEPT_AGENT_LIFECYCLE.md) (what an agent may do to the shared graph).

Reading order for someone new: §1 → §2 → §3, then the agent you are touching.

| | |
|---|---|
| **§1** | CORTEX — the shared graph and what "agent" means here |
| **§2** | The sub-cognitive layer — sensor drivers and the media plane |
| **§3** | Cross-cutting machinery — presence, lifecycle contract, shared belief modules |
| **§4** | `robot_concept` — the body, the frame tree, the sensor bridge |
| **§5** | `room_concept` — localization and the room itself |
| **§6** | `retina` — the perception hub |
| **§7** | `residual_concept` — the null hypothesis |
| **§8** | The object concept agents — table, chair, bottle, cabinet, refrigerator, door, human |
| **§9** | `ring_metaconcept` — a concept over concepts |
| **§10** | Action — `controller`, `kinova_controller` |
| **§11** | Introspection — `scene_graph_viewer`, `self_calibration` |
| **§12** | Conventions — frames, units, ids, build, shutdown |

---

## 1. CORTEX

CORTEX is a **shared, distributed, persistent scene graph** (DSR) that a set of independent
processes ("agents") read and write concurrently. There is no central server arbitrating writes and
no message-passing protocol between agents: the graph *is* the interface.

**The graph.** Nodes carry a `type` (`object`, `room`, `wall`, `robot`, `agent`, …), a name, and
typed attributes. Edges carry a type; the important one is `RT`, which holds a rigid transform
(translation + rotation) *plus* a timestamp ring buffer *and* a covariance. The RT edges form a
tree — `root → Shadow → body → sensors`, `room → wall_k`, `room → table_1` — and
`InnerEigenAPI::get_transformation_matrix(dst, src, ts)` resolves any chain through it. Every pose
in this system is expressed by naming two frames and asking that question.

**CRDT replication.** Each agent replicates the whole graph locally and publishes deltas over
FastDDS. Merges are conflict-free, resolved per attribute by the writer's `agent_id` and a
timestamp — which is why **every agent's `[Agent] id` must be globally unique**: two agents sharing
an id are the same CRDT actor, and the graph corrupts (SIGSEGV, not a merge error). The registry is
in §12.

**What "agent" means here.** An agent is a process that maintains a **probabilistic belief over one
kind of thing** and publishes that belief into the graph. It is not a module or a service. A
`table_concept` does not *detect* tables; it holds a posterior over a 6-DOF table model, updates it
against evidence, and writes the mean and covariance onto a graph node. Other agents consume that
posterior as *evidence*, not as ground truth — the covariance is part of the message.

**The modelling philosophy** (stated fully in `CLAUDE.md` §"no thresholds"). Effects that a
conventional system would express as a gate — "ignore detections beyond 5 m", "don't rotate when the
view is poor", "delete after 10 missed frames" — are instead encoded in the *generative model* as a
covariance that grows or shrinks with the right physical covariate, so the behaviour falls out of
the inference. A distant view loses orientation information because a common-mode covariance grows
with range; correlated mask points stop collapsing σ through Woodbury marginalisation; ego-motion
downweights masks through an interaction-matrix variance added to `R`. Where a genuine threshold
survives it must be flagged and justified in the code.

**Layers.** Beliefs are stratified, and the stratum is a real structural property, not a label:

```
meta-concepts     ring_metaconcept — a belief over an ARRANGEMENT of instances
      ↑ group_member
instances         table · chair · bottle · cabinet · refrigerator · door · human · residual
      ↑ RT (containment)
room              room_concept — walls, floor, polygon, and the robot's pose in it
      ↑ RT
robot             robot_concept — the body and its sensor mounts
```

`scene_graph_viewer` (§11) renders exactly this ladder: `z` is the stratum, `x/y` the true metric
pose.

---

## 2. The sub-cognitive layer and the media plane

Below the agents sit processes that do **no inference at all** — they move bytes from hardware into
the system. They live outside this folder (on the robot host; see [`DDS_SHADOW.md`](DDS_SHADOW.md)
for the deployment and Fast DDS tuning) and are deliberately kept dumb.

| Component | Produces |
|---|---|
| `zed_camera` | ZED stereo RGB + aligned depth |
| `lidar3d_dds` | two 3-D LiDARs — `helios` (upper, main) and `bpearl` (lower, downward dome) |
| `ricoh_omni_dds` | Ricoh Theta 360° equirectangular panorama |
| `webots-bridge` / `webots-shadow` | the simulator, standing in for all of the above |

### 2.1 Why a second plane exists

Raw sensor data must never travel through the DSR graph. A 30 fps RGBD stream inside a CRDT graph
starves the replication channel, and the first symptom is a *missed heartbeat* — an agent declared
dead because a camera was busy. So there are two independent DDS planes:

| | DSR plane | Media plane |
|---|---|---|
| **Domain** | `0` (`[Agent] domain`) | `7`, hardcoded in the descriptors |
| **Carries** | beliefs, geometry, small attributes | pixels, point clouds, IMU |
| **QoS** | reliable, CRDT-replicated | `BEST_EFFORT` + `KEEP_LAST` + `VOLATILE`, shared memory |
| **Code** | cortex `dsr_api` | `common/media_transport/` |

Never create a media participant on domain 0.

### 2.2 Self-describing streams

A producer does not publish a topic name into anyone's config. It writes a **`MediaDescriptor` JSON
string onto its own sensor node** (`media_descriptor` attribute) naming the domain, the per-stream
topic names, the IDL type, and a `ready` flag:

```
zed      → { domain 7, streams: { rgb: "rc/zed/rgb", depth: "rc/zed/depth" }, ... }
lidar3D  → { domain 7, streams: { lidar: "rc/lidar3d/points" }, ... }
imu      → { domain 7, streams: { imu: "rc/imu/data" }, ... }
```

Consumers call the shared factories — `rc::media::make_image_subscriber_from_graph(G, node, key)`,
`make_lidar_subscriber_from_graph(...)` — which read the descriptor, verify the node and stream
exist, and return a live subscriber or `nullptr`. **Do not hand-roll the descriptor → config → init
dance in an agent.** Topic and domain come only from the descriptor, never from a config file.

Two details that have caused real crashes:

- **`type_tag`** (`ImageFrame.v2`, `LidarFrame.v1`, …) is a schema guard. Zero-copy maps the shared
  memory segment byte-for-byte, so a subscriber must refuse a stream whose tag differs. Bump the tag
  whenever the `.idl` changes.
- **`data_sharing` defaults to OFF.** True zero-copy maps a fixed SHM segment that Fast DDS
  *reconfigures on participant discovery*. In CORTEX, agents join and leave constantly, so a
  consumer holding a loaned frame across a discovery event reads a remapped segment — heap
  corruption surfacing later as an unrelated SEGV in OpenCV or ONNX. The SHM *transport* still gives
  same-board speed; only the loan layer is disabled, at the cost of one memcpy per frame.

### 2.3 Where the boundary sits

`robot_concept` is the only place Ice-based sensor drivers are bridged onto the media plane. That is
a deliberate rule: never put DDS publishing inside a simulator bridge or a hardware driver.

---

## 3. Cross-cutting machinery

### 3.1 The presence protocol

`common/agent_presence_monitor/` + `common/agent_presence_coordinator/`
(guide: [`common/AGENT_PRESENCE_PROTOCOL.md`](common/AGENT_PRESENCE_PROTOCOL.md)).

Each agent publishes a heartbeat node under `mind` and polls its peers, classifying each as
`Missing` / `Booting` / `Ready` / `Stale`. A three-state GRAFCET lifecycle follows:

```
Compute ──entered──▶ Waiting ──presenceReady──▶ Operating ──presenceLost──▶ Degraded ──▶ Waiting
```

`Operating` is where an agent does its work; `[Presence] required_agent_names` decides when it gets
there. Two rules learned the hard way:

- **Degraded must debounce.** A transient peer flap at startup fires `presenceLost` momentarily;
  tearing down immediately deletes the agent's own nodes and exits cleanly, which looks like a
  silent, causeless shutdown. Schedule a ~3 s grace timer and only exit if the peer is *still*
  missing.
- **The monitor is poll-based on purpose,** and DSR update signals are connected `Queued` or not at
  all. Those signals fire on raw FastDDS reader threads; a `Qt::DirectConnection` slot there
  corrupts the heap, and the visible symptom is a smashed heartbeat far from the cause.

### 3.2 The lifecycle contract

[`CONCEPT_AGENT_LIFECYCLE.md`](CONCEPT_AGENT_LIFECYCLE.md) is normative for every `<obj>_concept`.
Five stages, each owned by a shared module so no agent re-implements it:

| Stage | Rule | Module |
|---|---|---|
| **CREATE** | Birth is tracker-only, evidence-accumulating (never a frame counter). An agent may not create a node the tracker did not birth. | `common/instance_tracker` |
| **UPDATE** | A frame must be *resolvable*, *centred* and *still* to move geometry; otherwise `predict()`-only — mean held, Σ inflates. Quality enters as covariance, never as a gate. | `common/ai_belief/recursive_laplace.h` |
| **REMOVE** | Removal is a Bayesian decision on existence log-odds, never a miss-counter. Absence is weighted by P(detect); occlusion **holds** rather than refutes. | `common/existence_belief`, `common/occlusion` |
| **HISTORY** | No constant may stand in for experience: process noise must reflect accumulated evidence. Every agent records phantom births/deaths. | `common/phantom_log` (theory: [`MODEL_HISTORY.md`](MODEL_HISTORY.md)) |
| **OWNERSHIP** | One mechanism: `[Owns]` must match `[Agent] id`; owned nodes are swept at startup and on graceful exit. | presence coordinator |

The file carries a per-agent conformance audit — add a row and justify every ❌ before shipping a
new agent.

### 3.3 Shared belief modules (`common/`)

| Module | What it provides |
|---|---|
| `ai_belief/` | The recursive variational-Laplace engine. One templated inference core; each agent authors only its generative model. Also `dof_spec.h` — per-DOF metadata (name, unit, consumer-relevant σ*). |
| `instance_tracker/` | Birth / associate / death across frames, with anti-duplicate separation and continuous `birth_evidence`. |
| `mask_ingestor/` | Reads the retina's `masks` node into per-instance mask slices + support points. The perception input layer for every mask-fitting agent. |
| `existence_belief/` | Per-instance existence log-odds fed by interchangeable channels (LiDAR carve, mask silhouette). |
| `occlusion/` | Line-of-sight primitives — "is my view of this object blocked, so absence is not evidence?" Includes walls. |
| `motion_corruption/` | Producer-side ego-motion mask corruption model (saccadic suppression / VOR): a prior on the *sensor channel*, not on the world. |
| `depth_projection/` | Reproject a cloud into any DSR camera via the shared `CameraAPI`; scores how much depth information a mask receives. |
| `affordance_protocol/`, `affordance_manager/` | The execution contract between a producing agent (which owns the semantics) and the controller (a generic executor). |
| `object_anchor/` | Producer/consumer contract for using fitted objects as SE(2) landmarks for localization. |
| `robot_footprint/` | The robot's real 2-D shape, in one place — replacing six independent size margins owned by three agents. |
| `appearance_belief/` | Per-instance albedo-chromaticity belief (display-only by design). |
| `phantom_log/` | Append-only CSV of birth/death events, identical schema across agents. |
| `self_tuner/` | Online adaptation of an agent's own behavioural constants from a `[Tune]` block. |
| `belief_stabilizer/`, `sample_queue/`, `motion_filter/`, `robust_metrics/` | Older/auxiliary belief machinery; `motion_filter` is the constant-velocity filter for *movable* objects. |
| `media_transport/` | The media plane (§2). |
| `graph3d/`, `viewers/`, `dashboard/` | Visualization: the stratified scene builder, GL widgets, and the per-instance belief inspector. |
| `agent_state_publisher/` | Each agent paints its own node with its health; an overlay greys out agents whose heartbeat stopped. |

---

## 4. `robot_concept` — the body (id 6)

The lowest stratum: everything else is positioned relative to what this agent publishes.

**Frame tree.** It bootstraps the graph from `shadow.json` — `root → Shadow → body → {zed, ricoh,
helios, bpearl, imu}`. Two frames matter and are easy to confuse: **`Shadow`** (type `robot`) is the
*localization* frame that `room_concept` writes the pose onto; **`body`** (+4.5 cm z) is the de-facto
base frame most consumers read. Full detail in [`ROBOT_GEOMETRY.md`](ROBOT_GEOMETRY.md), which is
derived from `shadow.json` — the truth.

**Sensor bridge.** `sensor_media_publisher.{h,cpp}` advertises a per-node media descriptor (rgb +
depth → `zed`, lidar → `lidar3D`, imu → `imu`) and relays Ice-delivered frames onto the media plane.
Publishers and the advertise call run on the main thread in `initialize()`, before any reader thread
starts.

**Introspection host.** It is the agent that runs the cortex node-link graph viewer
(`Agent.graph = true`), and therefore hosts two observers: the **agent-status overlay** (greys out
agents whose heartbeat went stale — the one state nobody can self-report) and the **stream viewers**
(right-click a sensor node → a live media-plane window: `media_stream_viewers.h`,
`graph_attr_viewers.h`).

---

## 5. `room_concept` — localization and the room (id 5)

Estimates the robot's SE(2) pose inside a known room polygon and owns the room's own geometry.
Math: [`room_concept/ROOM_CONCEPT.md`](room_concept/ROOM_CONCEPT.md).

- **State.** Historically 5-D `[w, h, x, y, θ]`; in the polygon mode used in practice the polygon is
  fixed and only `q = [x, y, θ]` is optimized.
- **Model.** A signed-distance field over the room polygon (loaded from SVG —
  `svg_room_loader.{h,cpp}`), scored against LiDAR returns. Localization runs on its own thread fed
  by thread-safe buffers.
- **Corners are the disambiguator.** A rectangular room is symmetric under 180° flips; only corners
  break the tie. `corner_detector.{h,cpp}` extracts them, and association uses Mahalanobis gating +
  PDA rather than nearest-neighbour — pose jumps in this system have almost always been
  *misassociation*, not a bad optimizer.
- **Publishes.** The `room` node (with `delimiting_polygon_x/y`, `room_height`), the `wall_k` and
  `floor` nodes with their display meshes, and the RT edge carrying the robot pose — `world→robot`
  before the estimate stabilises, `room→robot` after.
- **Epistemic side.** `epistemic_planner` / `epistemic_controller` propose motions that *reduce
  uncertainty* (occupancy entropy, corner visibility), which is where the "active" in active
  inference becomes an actual command.
- **Object anchors** (`object_anchor_*`): fitted objects from the concept agents can be consumed as
  SE(2) landmarks, closing a loop from instance beliefs back into localization.
- [`room_concept/HIERARCHICAL_PRECISION.md`](room_concept/HIERARCHICAL_PRECISION.md) covers the
  inferred log-precision hyperprior that replaces a hand-tuned clamp.

---

## 6. `retina` — the perception hub (id 9)

The single producer of semantic perception. Nearly every concept agent's input is a node this agent
writes, which is why it is a required peer almost everywhere.
Details: [`retina/RETINA.md`](retina/RETINA.md).

**Pipeline** (staged, in `perception_worker` + `*_stage.cpp`):

1. RGB + depth in from the media plane (`media_plane_source`), Ricoh panorama via `ricoh_source`.
2. **`seg_stage`** — YOLO segmentation (ONNX Runtime, CUDA/TensorRT) → per-instance masks.
3. **`semantic_stage` / `semantic_mask_stage`** — dense ADE20K-style semantic labels, used to
   synthesise instance masks for furniture that YOLO-seg misses.
4. **`sam2_stage`** — optional SAM2 refinement of mask boundaries.
5. **`pose_stage` / `bearing_stage`** — deproject masked depth pixels into room-frame support
   points; bearing-only for the 360 camera.
6. **`scene_processor`** — associate detections with persistent voxel tracks, maintain the semantic
   voxel grid, and compare sensed clouds against the model boxes already in the graph.
7. **`graph_publisher`** — write the results back.

**What it publishes.** The `masks` node is the contract: per-instance mask slices, 3-D support
points in *both* room frame (`mask_support_points`) and camera frame
(`mask_support_points_cam`), a per-mask colour summary, and the ego-motion corruption terms
(`MASK_MOTION_CORRUPTION.md`) that let a consumer inflate `R` while the camera moves. It also
publishes `semantic_grid` nodes and, per model object, `candidate_pts` / `residual_pts` /
`explanation_ratio` — how much of the sensed cloud the current model box explains.

`model_projection_overlay` / `ricoh_projection_overlay` draw each agent's published model back into
the image, which is the fastest way to see a belief that has drifted.

---

## 7. `residual_concept` — the null hypothesis (id 14)

The catch-all at the bottom of the concept hierarchy, and the safety layer for navigation.

The idea: subtract everything the specialists claim to explain — floor, ceiling, walls (from the
room polygon), the robot's own body, and every fitted object's surface — and whatever LiDAR returns
remain are **unmodelled obstacles**. They get birthed as `residual_*` nodes that the controller's
planner avoids. Nothing needs to recognise an object for the robot to avoid it.

Chain in `compute()`: `lidar_ingestor` (room-frame sweep) → `clusterer` (residual filter + 3-D
DBSCAN) → `tracker` (birth/associate/death) → `fitter` (per-instance box-footprint free-energy fit
and write-back).

Two design points worth knowing:

- **Explanation is a soft marginalisation, not a margin.** A point is explained with probability
  `Φ(−sdf/σ_tot)`, where `σ_tot` convolves the object's *own published position covariance* (which
  already includes the localization-chain term) with sensor noise. A distant, uncertain table
  therefore claims points more loosely — no `k`, no floor.
- **Publication is deliberately conservative.** LiDAR sees only near faces, so the raw fit
  under-sizes obstacles. The published footprint is grown by `k·σ` of the extent — and because
  occlusion inflates exactly that σ, it grows on the unobserved side — plus a robot-clearance
  margin. Internal fit and carve use the raw estimate.

It also maintains an occupancy log-odds grid; the direction of travel is to expose it as a
continuous belief field rather than a set of boxes ([`RESIDUAL_BELIEF_FIELD.md`](RESIDUAL_BELIEF_FIELD.md)).

---

## 8. The object concept agents

All of these share one structure ([`CONCEPT_AGENT_PATTERN.md`](CONCEPT_AGENT_PATTERN.md) for
rationale, [`CONCEPT_AGENT_RECIPE.md`](CONCEPT_AGENT_RECIPE.md) for generating a new one): read the
`masks` node → track instances → update a recursive-Laplace belief over a class-specific model →
publish pose + covariance on the RT edge → advertise affordances. They differ only in the
*generative model* and in what evidence is available.

| Agent | id | Belief state | Notes |
|---|---|---|---|
| `table_concept` | 7 | `[cx, cy, H, w, h, yaw]`, full 6×6 Σ | The reference implementation. Math in [`table_concept/TABLE.md`](table_concept/TABLE.md); a per-point SDF mixture over {top, legs, clutter}, plus round-vs-square model selection by free energy. Exposes an object-relative viewpoint affordance. |
| `chair_concept` | 20 | pose-only 3-DOF + fixed template | A square seat makes yaw 4-fold ambiguous; only the backrest breaks it, and it is thin. Yaw evidence is information-weighted. This ambiguity is what motivated the rig meta-concept (§9). |
| `bottle_concept` | 10 | pose (+ velocity) | The known-good reference for presence/DSR discipline. *Movable*, so it tracks rather than hardens — `common/motion_filter` instead of a stiffening prior. |
| `cabinet_concept` | 21 | (wall, tier) cells owning wall-run beliefs | Individual cabinets are not identifiable; the **run** is. Keying beliefs to (wall, tier) cells makes identity structural, so association churn is zero by construction. |
| `refrigerator_concept` | 22 | box + wall-flush constraint | Cloned from table; adds a Sobel-based yaw resolver. |
| `door_concept` | 15 | `[s, w, h]` in the containing **wall frame** | Spec: [`DOOR_CONCEPT_SPEC.md`](DOOR_CONCEPT_SPEC.md). Parametrised along its wall, so the wall's own belief constrains it. |
| `human_concept` | 13 | 11-DOF kinematic model per person | Consumes BODY_18 skeletons through a decoupled `SkeletonSource`; fits by Laplace free-energy; publishes pelvis pose. Advertises a *reduce-occlusion* affordance so the controller can clear occluded joints. |

Shared behaviour worth stating once: geometry is published in the **room frame** on the
`room → <object>` RT edge, with the covariance the consumer needs to weigh it; the class lives in
`object_subtype` on a generic `object` node (not in the DSR node type — that migration is done);
and each agent owns its nodes by name prefix so a crashed run is swept at the next startup.

---

## 9. `ring_metaconcept` — a concept over concepts (id 23)

A **level-2** agent: its latent is not an object but an *arrangement*. The `dining_set` — a table
anchor plus a ring of chairs — is its first instance.
Design: [`DINING_SET_RIG_PLAN.md`](DINING_SET_RIG_PLAN.md).

- **Its measurements are other agents' posteriors.** It polls member nodes from the graph and reads
  each one's published pose *together with the covariance that agent published*. A poorly-seen chair
  therefore contributes weakly to the arrangement fit by construction — no confidence gate anywhere.
- **Its front end is a graph reader, not a sensor.** It connects no `update_node` signals; it polls
  on the main thread.
- **It sends messages back down.** A fitted ring implies where each seat should be and which way it
  should face; that becomes an empirical prior pushed down the `group_member` edges — which is
  exactly the information a single chair's mask does not contain.
- **The cavity rule is the one hard requirement.** The message sent down to member *i* must come
  from an arrangement fitted **without** member *i*. Otherwise the rig folds *i*'s own contribution
  into the centre it then tells *i* to face, and manufactures agreement with itself.

Open issues, including the fact that the belief scores member *positions* but not member *yaw*, are
in [`ring_metaconcept/SCHEMA_GENERALITY_TODO.md`](ring_metaconcept/SCHEMA_GENERALITY_TODO.md).

---

## 10. Action

### `controller` (id 8)

The base's executor and planner. `grid_planner` + `route_spline` + `route_follower` +
`trajectory_controller` produce motion; `controller_obstacle_model` / `controller_obstacle_tracker`
maintain the local obstacle picture (fed largely by `residual_concept`);
`controller_world_model` / `controller_session` / `controller_mission` hold the task layer.

`epistemic_planner` is the part that makes this more than a navigation stack: affordances published
by concept agents are scored by **expected free energy**, so the robot chooses to move where it will
*learn* — closing the loop from an agent's Σ back into action.
See [`controller/EPISTEMIC_PLANNER_DESIGN.md`](controller/EPISTEMIC_PLANNER_DESIGN.md).

### `kinova_controller`

The 7-DOF arm. Grasp FSM (Tracking → Inserting → Closing → Lifting → Holding), a 6-DOF
damped-least-squares controller with a preferred-flow attractor, and an EFE-based sequencer.
Kinematics via Pinocchio (`common/kinematics/`). Docs: `kinova_controller/{FSM,EFE_CONTROLLER_MATH,
FRAMES}.md`.

---

## 11. Introspection

### `scene_graph_viewer` (id 16)

The stratified 3-D view of the whole graph, as a standalone agent: `z` is the abstraction stratum,
`x/y` the true metric pose. It exists because the planar node-link viewer draws `room → table →
aff_table_1` with the same weight as `mind → agent`, so neither **agent ownership** nor
**meta-concept grouping** is visible in it — and `group_member`, being a non-RT edge, has no
transform for a spatial layout to use at all.

Pure observer: no required peers, writes nothing but its own presence node, so starting or stopping
it cannot perturb what anything believes. It owns a top-level window rather than docking into the
cortex viewer (a heavy `QOpenGLWidget` sharing that backing store crashes on peer join).
Builder in `common/graph3d/` (DSR-aware, `dsr_api` only), renderer in
`common/viewers/gl_graph3d_viewer.h` (DSR-blind, so any agent can drive it).

### `self_calibration` (id 12)

Estimates the arm↔camera extrinsics online from an FK-vs-depth residual with a GLS information
filter, and is the testbed for *epistemic* calibration moves — choosing arm motions that maximise
information about the parameters.

---

## 12. Conventions

### Agent ids

Unique across the shared graph; a collision is a CRDT actor clash → SIGSEGV in both agents.

```
room=5  robot=6  table=7  controller=8  retina=9  bottle=10  kinova=11  self_calib=12
human=13  residual=14  door=15  scene_graph_viewer=16  hood=17  viewer3d=18  ltsm_agent=19
chair=20  cabinet=21  refrigerator=22  ring_metaconcept=23  kitchen_metaconcept=24
                                              next free: 25+
```

### DDS domains

Also unique-by-convention, and just as easy to clash on. FastDDS maps domain `d` to port
`7400 + 250*d`, which is the quickest way to tell from the outside which plane a process is on
(`ss -lunp | grep pid=<pid>`).

```
0 = the shared DSR working graph     2 = the long-term spatial memory plane (ltsm_agent)
1 = isolated standalone runs         7 = the zero-copy media plane
```

### Frames and units

- **Metres, radians, room frame** for everything published. `+Z` up; `z = 0` is the robot-base datum
  (≈ floor).
- The DSR `zed` frame is x-right, **y-depth**, z-up — not an optical frame.
- Ask for poses with `get_transformation_matrix(dst, src, ts)` and **always check the optional**: a
  missing node or edge returns `{}` rather than throwing, and the crash is the caller's `.value()`.
- Pin transforms to the **capture timestamp** off the main thread. `ts == 0` uses an unlocked cache
  and is main-thread-only.

### Graph access

- Use the **type-attributed** API — `get_attrib_by_name<foo_att>(n)`,
  `add_or_modify_attrib_local<foo_att>(n, v)` — never the `runtime_checked_*` string forms. Adding a
  new attribute means adding a `REGISTER_TYPE` line to cortex and reinstalling it.
- `update_node` **deletes attributes absent from the submitted copy**. A cross-agent message must
  therefore ride an *edge* owned by its single writer, not a node many agents write.
- Reads and writes are thread-safe (`DSRGraph` serialises them). The real cross-thread hazard in
  this codebase is `cv::Mat`, whose copies are refcounted shallow handles — deep-copy at every
  thread boundary.

### Build

`cmake -B build && make -C build -j32`. Never `-march=native` and never
`-DEIGEN_MAX_ALIGN_BYTES=0`: both change `sizeof(std::optional<Eigen::Transform>)` away from the
prebuilt libdsr's 144 and produce SIGBUS or a smashed stack in whatever agent shares the graph.
After adding sources to an INCLUDEd `src/CMakeLists.txt`, nuke `build/CMakeCache.txt` and
reconfigure.

### Running and stopping

Agents are plain binaries: `./bin/<agent> etc/config.toml`. `run_agent_supervised.sh` restarts one
in a loop; `new_deployment.sh` opens a terminal tab per component.

**Stop with SIGTERM / SIGINT / Ctrl-C, never `kill -9`.** These agents own nodes in the shared,
persistent graph, and cleanup runs only on a graceful exit. A SIGKILLed agent leaks every node it
created, with no process alive to explain them; the next startup's stale-sweep is the only thing
that reaps them. To test a *stall* rather than a death, use `kill -STOP`.

---

## Document index

| File | What it covers |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Hard rules — DSR threading, signals, attributes, build, shutdown |
| [`CONCEPT_AGENT_LIFECYCLE.md`](CONCEPT_AGENT_LIFECYCLE.md) | Normative behaviour contract + per-agent conformance audit |
| [`CONCEPT_AGENT_RECIPE.md`](CONCEPT_AGENT_RECIPE.md) | How to generate a new concept agent |
| [`CONCEPT_AGENT_PATTERN.md`](CONCEPT_AGENT_PATTERN.md) | Rationale and history behind the pattern |
| [`ROBOT_GEOMETRY.md`](ROBOT_GEOMETRY.md) | Frame tree and sensor mounts |
| [`FRAMES.md`](FRAMES.md) | Coordinate conventions |
| [`DDS_SHADOW.md`](DDS_SHADOW.md) | Media-plane deployment and Fast DDS tuning |
| [`BELIEF_PIPELINE.md`](BELIEF_PIPELINE.md) | How an object belief is formed, tracked, published |
| [`EXISTENCE_BELIEF_PLAN.md`](EXISTENCE_BELIEF_PLAN.md) | Existence log-odds — the shared removal rule |
| [`MODEL_HISTORY.md`](MODEL_HISTORY.md) | How experience accumulates and models settle |
| [`MASK_MOTION_CORRUPTION.md`](MASK_MOTION_CORRUPTION.md) | Ego-motion mask corruption |
| [`AI2_MIGRATION_PLAN.md`](AI2_MIGRATION_PLAN.md) | Consolidation onto the shared belief engine |
| [`table_concept/TABLE.md`](table_concept/TABLE.md) | The reference belief, in full |
| [`room_concept/ROOM_CONCEPT.md`](room_concept/ROOM_CONCEPT.md) | Localization math |
| [`retina/RETINA.md`](retina/RETINA.md) | Perception pipeline |
| [`RESIDUAL_BELIEF_FIELD.md`](RESIDUAL_BELIEF_FIELD.md) | Residual as a continuous field |
| [`DINING_SET_RIG_PLAN.md`](DINING_SET_RIG_PLAN.md) | The meta-concept |
| [`RICOH_360_PERIPHERAL_DETECTION.md`](RICOH_360_PERIPHERAL_DETECTION.md) | 360° peripheral attention |
