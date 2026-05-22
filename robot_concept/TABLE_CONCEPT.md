# table-concept Agent — Complete Design Proposal

## 1. Motivation

`robot_concept` accumulates voxel evidence for every object it observes. Without a bounding
mechanism this evidence grows without limit as the robot navigates new areas and discovers new
objects. The long-term goal is a hierarchical perception architecture where higher-level concept
agents own, refine and maintain object models, and lower-level sensing agents are freed from
storing evidence that has already been explained.

`table-concept` is the first such concept agent. It is responsible for all table instances in the
scene. It owns their generative models, fits them via Active Inference, proposes epistemic actions
to reduce model uncertainty, and signals `robot_concept` to discard locally-held evidence once a
table has been adopted.

---

## 2. Agent Responsibilities

| Agent | Responsibility |
|---|---|
| `robot_concept` | RGBD + YOLO → voxel accumulation → split explained/residual → upload candidate stream + residual stream → receive model bounds for suppression |
| `table-concept` | Own table priors → run Active Inference on incoming streams → manage circular sample queue → update DSR model params → compute epistemic action proposals |
| `mission-controller` | Arbitrate epistemic action proposals from all concept agents against task goals → emit navigation targets |
| `room-concept` | Create / destroy room node → initially bootstraps the table node; after `table-concept` is running, node lifecycle transfers to `table-concept` |

---

## 3. Architecture Overview

```
                      ┌──────────────────────────────┐
                      │        DSR Graph              │
                      │                               │
     robot_concept    │  [room]──RT──[table_1]        │    table-concept
     ─────────────    │                │              │    ─────────────
     sense RGBD+YOLO  │    width_m     depth_m        │    read streams
     split explained  │    height_m   RT-pose         │    circular queue
     /residual        │                               │    SDF / Free Energy
     write streams ──►│  candidate_pts_att            │◄── write model params
     read model box ◄─│  residual_pts_att             │──► write epistemic props
                      │  residual_mass_att            │
                      │  explanation_ratio_att        │    mission-controller
                      │  last_sensing_frame_att       │    ─────────────────
                      │                               │◄── read epistemic props
                      │  free_energy_att              │──► emit nav target
                      │  model_stable_att             │
                      │  model_generation_att         │
                      │  epistemic_target_x_m         │
                      │  epistemic_target_y_m         │
                      │  epistemic_target_yaw_rad     │
                      │  epistemic_gain               │
                      │  epistemic_pending            │
                      └──────────────────────────────┘
```

---

## 4. DSR Node and Attribute Schema

### 4.1 Node type

`table` (existing in CORTEX node registry). One node per table instance.

### 4.2 RT edge

The RT edge from `room` to `table` defines the 6-DOF pose of the table model origin in the room
frame. `table-concept` is the sole writer of this edge. All point cloud attributes in this document
are expressed in **room frame**.

### 4.3 Existing geometry attributes (authoritative, written by table-concept)

| Attribute | Type | Description |
|---|---|---|
| `width_m_att` | float | table width in metres (X axis in table frame) |
| `depth_m_att` | float | table depth in metres (Y axis in table frame) |
| `height_m_att` | float | table height in metres (Z axis in table frame) |

### 4.4 New sensing attributes (written by robot_concept, read by table-concept)

| Attribute | Type | Max size | Description |
|---|---|---|---|
| `candidate_pts_att` | `vector<float>` | 1 200 floats (400 pts × XYZ) | Near-surface voxel centroids from the local grid, room frame, interleaved XYZ |
| `residual_pts_att` | `vector<float>` | 450 floats (150 pts × XYZ) | Voxel centroids outside the model box but within the model neighbourhood, room frame |
| `residual_mass_att` | `int` | — | Total unexplained voxel count before downsampling |
| `explanation_ratio_att` | `float` | — | Fraction of incoming category points absorbed by the model, EMA over last N frames |
| `last_sensing_frame_att` | `int` | — | Frame counter of last write; used by table-concept to detect stale data |

### 4.5 New inference attributes (written by table-concept, read by robot_concept and others)

| Attribute | Type | Description |
|---|---|---|
| `free_energy_att` | `float` | Current value of F after the last update step |
| `model_stable_att` | `bool` | True when F has not decreased by more than ε for K consecutive frames and all faces have coverage above δ_min |
| `model_generation_att` | `int` | Incremented each time model parameters (pose or geometry) change |
| `model_uncertainty_att` | `float` | Scalar summary of coverage deficit across all faces; 0 when all faces well-observed |
| `request_full_sample_att` | `bool` | Set by table-concept when the model has diverged and a fresh full re-sample is needed from robot_concept |

### 4.6 Epistemic action attributes (written by table-concept, read by mission-controller)

| Attribute | Type | Description |
|---|---|---|
| `epistemic_target_x_m` | `float` | Proposed robot position in room frame |
| `epistemic_target_y_m` | `float` | |
| `epistemic_target_yaw_rad` | `float` | Heading that faces the low-coverage surface |
| `epistemic_gain` | `float` | Expected entropy reduction ΔH from this viewpoint |
| `epistemic_pending` | `bool` | Set true to signal mission-controller; cleared after approval or rejection |

### 4.7 CORTEX attribute registry additions

All new attributes must be registered in:
```
/home/pbustos/robocomp/components/cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h
```

New entries follow the existing pattern with appropriate C++ type mapping:

```cpp
// ── table-concept sensing interface ──────────────────────────────────────────
REGISTER_TYPE(candidate_pts_att,          std::vector<float>, false)
REGISTER_TYPE(residual_pts_att,           std::vector<float>, false)
REGISTER_TYPE(residual_mass_att,          int,                false)
REGISTER_TYPE(explanation_ratio_att,      float,              false)
REGISTER_TYPE(last_sensing_frame_att,     int,                false)

// ── table-concept inference outputs ──────────────────────────────────────────
REGISTER_TYPE(free_energy_att,            float,              false)
REGISTER_TYPE(model_stable_att,           bool,               false)
REGISTER_TYPE(model_generation_att,       int,                false)
REGISTER_TYPE(model_uncertainty_att,      float,              false)
REGISTER_TYPE(request_full_sample_att,    bool,               false)

// ── epistemic action proposal ─────────────────────────────────────────────────
REGISTER_TYPE(epistemic_target_x_m,       float,              false)
REGISTER_TYPE(epistemic_target_y_m,       float,              false)
REGISTER_TYPE(epistemic_target_yaw_rad,   float,              false)
REGISTER_TYPE(epistemic_gain,             float,              false)
REGISTER_TYPE(epistemic_pending,          bool,               false)
```

---

## 5. The Voxel Adoption Protocol

This is the mechanism by which `robot_concept`'s local voxel count is kept bounded as the robot
discovers more objects.

### 5.1 Phases per table instance

```
Phase 1 — Accumulation
  Object track builds up in the local voxel grid normally.
  No suppression active for this track.
  No uploads to DSR yet.
  Triggered by: track age ≥ min_frames AND voxel_count ≥ min_voxels

Phase 2 — Adoption upload (once per instance)
  robot_concept:
    takes a near-surface spatial subsample (~400 pts) of the full track
    takes the current residual cluster (~150 pts)
    writes candidate_pts_att, residual_pts_att, residual_mass_att to the DSR node
    calls voxel_grid_.remove(track_id)  ← the entire track is deleted locally
  table-concept:
    reads the full sample, seeds the circular queue
    begins model fitting from the prior

Phase 3 — Residual-only steady state
  Incoming table-labeled points → suppressed by model box
  Small residual cluster may accumulate → uploaded every N frames as residual_pts_att
  After each residual upload → voxel_grid_.remove(residual_track_id)
  table-concept uses residual for Free Energy gradient steps

Phase 4 — Refresh (exceptional)
  table-concept detects divergence (F rising, queue coverage dropping)
  sets request_full_sample_att = true
  robot_concept suspends suppression for one accumulation window (e.g. 60 frames)
  new full track accumulates → adoption upload repeated → track deleted
  request_full_sample_att cleared
```

### 5.2 Voxel budget consequence

```
Before adoption:  N_local = N_objects × N_voxels_per_track      (unbounded growth)
After adoption:   N_local = N_unmodeled × N_voxels_per_track
                          + N_adopted  × N_residual_per_track   (bounded)

N_residual_per_track ≪ N_voxels_per_track  (typically 50–100 vs 2000+)
```

Provided adoption fires faster than new objects are discovered, the local voxel count is bounded
regardless of scene size.

---

## 6. Generative Model and Free Energy

### 6.1 Generative model

The table model follows the validated Python prototype (`box_concept/src/objects/table/`).
The state vector has **7 parameters**:

```
θ = [cx, cy, w, h, table_height, leg_length, yaw]
```

| Index | Name | Description |
|---|---|---|
| 0 | `cx` | Room-frame X of table centre |
| 1 | `cy` | Room-frame Y of table centre |
| 2 | `w` | Width of table top (X axis in table frame) |
| 3 | `h` | Depth of table top (Y axis in table frame) |
| 4 | `table_height` | Height of table surface from floor |
| 5 | `leg_length` | Length of legs from floor to underside of top |
| 6 | `yaw` | Rotation around Z axis |

Hard constraint enforced by clamp after each gradient step:
`leg_length ≤ table_height − TOP_THICKNESS`

Fixed constants: `TOP_THICKNESS = 0.03 m`, `LEG_RADIUS = 0.025 m`.

### 6.2 Compound SDF

The SDF is the **smooth union** of a box-top slab and four cylindrical legs:

```
SDF(θ, p) = min( SDF_top(θ, p),  min_{k=1..4} SDF_leg_k(θ, p) )
```

**Table top** — box of size `(w, h, TOP_THICKNESS)` centred at `z = table_height − TOP_THICKNESS/2`:

```
SDF_top(θ, p) = standard_box_sdf(local_p, [w/2, h/2, TOP_THICKNESS/2])
                with smooth interior via log-sum-exp
```

**Legs** — four cylinders of radius `LEG_RADIUS`, each at corner offset
`(±(w/2 − LEG_RADIUS − 0.02), ±(h/2 − LEG_RADIUS − 0.02))` in table-local XY,
extending from `z = 0` to `z = leg_length`:

```
SDF_leg_k(θ, p) = cylinder_sdf(local_p − leg_centre_k, LEG_RADIUS, leg_length/2)
```

This matches the prototype implementation in `box_concept/src/objects/table/sdf.py` and
is directly portable to C++ with Eigen substituting for PyTorch tensor ops.

### 6.3 Likelihood

```
log p(yᵢ | θ) = − SDF(θ, yᵢ)² / σ²
```

Points on the model surface have SDF ≈ 0 → log p ≈ 0 → no surprise.
Residual points have SDF > 0 → log p < 0 → positive surprise → drives model update.

### 6.3 Free Energy

```
F(θ) = − Σᵢ log p(yᵢ | θ)   +   KL( q(θ) ‖ p(θ) )
      = Σᵢ SDF(θ, yᵢ)²/σ²   +   λ‖θ − θ_prior‖²_Σ
```

The KL term is a quadratic regularizer toward the prior geometry and pose, preventing the model
from collapsing onto noise or drifting with robot motion when observations are sparse.

### 6.4 Update step

Gradient descent on θ, with per-parameter learning rates:

```
θ ← θ − η ∇_θ F(θ)

∇_θ F = − Σᵢ [2 SDF(θ, yᵢ) ∇_θ SDF(θ, yᵢ)] / σ²   +   2λΣ⁻¹(θ − θ_prior)
```

Step size η is decayed as `model_stable` approaches true. Geometry updates (W, D, H) are clamped
to ±20% of the prior per update step to prevent runaway fitting.

---

## 7. The Historical Sample Queue

### 7.1 Motivation

A fixed snapshot taken at adoption time represents exactly one viewpoint history. As the robot
moves, previously occluded surfaces become visible. A fixed snapshot has no entries for those
surfaces and provides no gradient signal from new perspectives. The historical sample queue solves
this by continuously evicting low-quality points and admitting fresh near-surface observations.

The design is validated by the working Python prototype (`box_concept/src/objects/table/belief.py`,
methods `add_historical_points`, `_add_to_bins`, `_compute_edge_score`, `update_rfe`,
`get_historical_points_in_robot_frame`). The C++ port should reproduce this design faithfully.

### 7.2 Data structure — binned, not circular

Points are **not** stored in a flat circular buffer. They are partitioned into angular × height
bins centred on the table:

```
num_angle_bins = 24   (15° sectors around table centre in XY)
num_z_bins     = 10   (0.1 m height slices)
max_per_bin    = 2    (best 2 points per bin, by quality metric)
```

Bin index for a point at room position `p`:
```
ab = floor((atan2(p.y − cy, p.x − cx) + π) / (2π) × 24) % 24
zb = clamp(floor(p.z / 0.1), 0, 9)
bin_idx = ab × 10 + zb
```

Total capacity: `24 × 10 × 2 = 480` points. This guarantees **uniform angular and vertical
coverage** regardless of robot path — no single viewpoint can over-fill the queue.

C++ data layout:

```cpp
struct SamplePoint {
    Eigen::Vector3f position;      // real observed voxel centroid, room frame
    Eigen::Matrix2f capture_cov;   // robot XY covariance at capture time (2×2)
    float           rfe;           // Remembered Free Energy (see §7.3)
};

std::vector<SamplePoint> historical_points;  // rebuilt each insertion cycle
```

### 7.3 Remembered Free Energy (RFE) — replaces the simple weight function

**RFE** replaces the instantaneous weight function `exp(−SDF²/2σ_s²)` used in the original
design. Each stored point accumulates a time-smoothed measure of model-point disagreement:

$$
\text{RFE}_{t+1}(y_i) = \alpha \cdot \text{RFE}_t(y_i) + w_t \cdot \text{SDF}(\theta_t, y_i)^2
\qquad w_t = \frac{1}{1 + \mathrm{tr}(\Sigma_{\text{robot},t})}
$$

Parameters: `α = 0.98`, `rfe_max_threshold = 2.0`.

`w_t` downweights accumulation when robot localisation is uncertain.

| RFE value | Meaning | Action |
|---|---|---|
| Low (< 0.03) | Point consistently near model surface — trusted | High gradient weight, retained |
| Medium (0.03–1.0) | Occasional mismatch — good quality | Normal weight |
| High (> 2.0) | Persistent mismatch — model has drifted | Hard eviction |

**Gradient weight** used when combining historical points with current observations in the VFE
optimizer:

$$
w_i = \frac{1}{1 + \mathrm{tr}(\Sigma_{\text{capture},i} + \Sigma_{\text{robot},\text{propagated},i}) + \text{RFE}_i}
$$

This is fully covariance-aware: high capture uncertainty and high accumulated error both reduce the
point's contribution to the gradient.

### 7.4 Edge/corner priority

A geometric priority score is computed for each point by counting how many table-top faces it
is simultaneously close to (within `edge_proximity_threshold = 0.05 m`):

```
edge_score ∈ [0, 1]:  0.0 = flat face,  0.5 = edge (near 2 faces),  1.0 = corner (near 3 faces)
```

The bin **quality metric** — lower is better:

```
quality = trace(Σ_capture) + RFE − edge_bonus_weight × edge_score
```

`edge_bonus_weight = 0.3`. Edge and corner points constrain multiple model dimensions
simultaneously (higher Fisher information) and are therefore preferred over flat-face points at
equal SDF quality.

### 7.5 Admission control — gradual warmup

Points are not stored until the gradient optimizer has had time to converge to a stable pose:

```
min_frames_before_historical = 10   (no storage during first 10 matched frames)
historical_warmup_frames     = 50   (ramp max_new_points_per_frame from 1 → 5 over 50 frames)
max_new_points_per_frame     = 5    (at full speed: max 5 new points admitted per frame)
sdf_threshold_for_storage    = 0.08 m  (only near-surface points: |SDF| < 0.08 m)
```

Of eligible candidates each frame, only the `max_new_points_per_frame` with the smallest
|SDF| are passed to the bin-insertion cycle.

### 7.6 Insertion cycle (each frame with fresh observations)

```
1. For each new candidate c with |SDF(θ, c)| < sdf_threshold_for_storage:
     compute bin_idx from (angle, z)
     compute edge_score and quality = trace(robot_cov) + 0.0 (RFE=0 for new pts) − edge_bonus_weight × edge_score

2. Merge new candidates with existing points in each bin

3. For each bin:
     sort by quality (ascending)
     keep top max_per_bin entries  →  lower quality wins (better points survive)

4. Rebuild historical_points from all bins

5. Run update_rfe() for all retained points:
     recompute SDF under current θ
     RFE ← α × RFE + w_t × SDF²
     evict any point with RFE > rfe_max_threshold
```

### 7.7 Self-regulating convergence

| Situation | RFE distribution | Effect |
|---|---|---|
| Model poorly fit | Many high-RFE points, frequent eviction | Fresh points dominate gradient, strong update |
| Model converging | RFE declining across bins | Fewer evictions, gradient settles |
| Robot moves to new viewpoint | New angular bins fill; occluded-bin points accumulate RFE | Queue adapts to current visibility |
| Model drifts from true surface | RFE rises for previously well-fit points | Eviction restores learning |
| Convergence reached | All RFE low, eviction rate ≈ 0, ΔF/step < ε | `model_stable = true` |

The bin eviction rate is itself a convergence signal: high eviction → model actively learning;
near-zero eviction → model has stabilised.

### 7.8 Why real points, not synthetic

The weight function is only meaningful if the points are real measurements. If synthetic points
(generated from the model surface) are used:

- SDF(θ, yᵢ) = 0 by construction for all points
- All RFE values stay at 0
- All gradient weights are 1
- The gradient is identically zero
- F reduces to the KL term alone
- The loop confirms itself every iteration and learns nothing

Real observed voxel centroids already encode temporal integration: each centroid is the EMA of all
raw RGBD points that hit that spatial cell across many frames. The queue is a spatial subsample of
that integrated surface, not a frame-by-frame replay.

---

## 8. Epistemic Actions

### 8.1 Motivation from Active Inference theory

Active Inference separates the total expected free energy into two terms:

```
G(a) = E_q[ F(o, a) ]
     = − E_q[ log p(o|a,s) ]   (expected surprise / pragmatic value)
     +   H[ q(s|a) ]           (epistemic value / information gain)
```

The epistemic term drives the agent to select actions that maximally reduce uncertainty about the
model state. For `table-concept`, this means proposing the robot position that would make the
least-observed table surface visible to the camera.

### 8.2 Coverage map

The queue's weight distribution over the model surface defines the coverage map. For each of the
six box faces, compute:

```
coverage(fₖ) = Σᵢ∈queue  wᵢ · 𝟙[ n̂ₖ · (yᵢ − cₖ) ∈ [−ε, ε] ]
```

where n̂ₖ is the outward normal of face k and cₖ is its centre. A face with low coverage has high
uncertainty and high epistemic value.

For a floor-navigating robot, only the four vertical faces (north, south, east, west in table
frame) are reachable. The top face is observed from above by normal operation. The bottom face is
never observable.

### 8.3 Optimal viewpoint computation

For the lowest-coverage face fₖ:

```
v* = cₖ + n̂ₖ · d_obs
```

projected to the floor plane (z = floor height). d_obs ≈ 1.5–2.0 m (comfortable observation
distance for the depth camera). The target heading is the inverse of n̂ₖ projected to XY.

This computation uses only the current model parameters θ. No additional sensing is required.

### 8.4 Expected information gain

The expected entropy reduction from viewpoint v:

```
ΔH(v) = ½ log det( I + Σ · ℐ(v) )
```

where ℐ(v) is the Fisher information matrix from points that would become visible on the model
surface from v. For a box SDF this is approximated analytically by the projected face area within
the camera FOV, scaled by 1/σ².

A practical proxy: count unobserved model surface cells within the camera FOV at v, weighted by
the current coverage deficit (1 − mean_weight) per cell.

### 8.5 Proposal protocol

```
table-concept (each cycle):
  1. Compute coverage per face
  2. If any face has coverage < δ_min AND model_stable == false:
       compute v* for the lowest-coverage face
       compute ΔH(v*)
       if ΔH(v*) > gain_threshold:
         write epistemic_target_x_m, _y_m, _yaw_rad, epistemic_gain to DSR
         set epistemic_pending = true
         set model_stable = false

mission-controller:
  Reads all epistemic proposals across all concept agents
  Ranks by gain/navigation_cost ratio
  Approves highest-priority feasible action → navigation target on robot node
  Clears epistemic_pending on the approved node
  Sets epistemic_pending = false on rejected nodes (no re-proposal for K frames)

robot_concept (after navigation):
  New viewpoint → new near-surface observations → enter candidate stream
  table-concept: gap fills → coverage rises → epistemic proposals cease
  When all faces above δ_min AND F < F_min for K frames:
    model_stable = true
    no new epistemic proposals generated
```

### 8.6 Diminishing returns

As coverage fills and the model converges, ΔH(v*) decreases monotonically. Below `gain_threshold`
no new proposals are emitted. The system naturally stops generating epistemic actions without any
explicit termination rule.

---

## 9. Scaffolding and Reinstantiation

### 9.1 Object prior library

`table-concept` owns a prior config file:

```
table_concept/etc/object_priors.toml
```

Example:

```toml
[[tables]]
node_name  = "bootstrap_table_1"
width_m    = 2.0
depth_m    = 0.8
height_m   = 0.75
room_x_m   = 3.5
room_y_m   = 1.2
yaw_rad    = 0.0
sigma_pose = 0.1   # prior uncertainty on position (metres)
sigma_size = 0.15  # prior uncertainty on dimensions (metres)
```

On startup, `table-concept` checks whether each listed node exists in DSR. Missing nodes are
created with the prior geometry and RT edge. This replaces the hardcoded bootstrap constants
currently in `room_concept`.

### 9.2 Convergence checkpoint

Before planned shutdown, `table-concept` writes the current fitted parameters to a checkpoint
alongside the prior file:

```toml
[checkpoint.bootstrap_table_1]
width_m    = 2.03
depth_m    = 0.81
height_m   = 0.74
room_x_m   = 3.52
room_y_m   = 1.18
yaw_rad    = 0.02
free_energy = 0.014
model_stable = true
```

After recreation, the checkpoint is applied before any new sensing data arrives. If no checkpoint
exists, the node starts from the prior and rebuilds.

### 9.3 robot_concept redetection guard

If `SceneProcessor` finds the expected table node is absent from DSR:
- Suspend the suppression path for that table (do not block new evidence)
- Do not let new voxel evidence create a competing free track near the prior location
- Set an internal flag `awaiting_table_node` that disables track creation for table-labeled
  observations within the prior neighbourhood
- Once the node reappears (created by `table-concept` on restart), normal suppression resumes
  without resetting the voxel grid

---

## 10. Modifications to robot_concept

### 10.1 SceneProcessor changes

Add methods:

```cpp
// Returns true if a reachable table node exists for a given category + location
bool table_node_exists_near(const Eigen::Vector2f& room_xy, float radius_m) const;

// Reads request_full_sample_att; if true, instructs VoxelProcessor to bypass suppression
bool is_full_sample_requested(int table_node_id) const;

// Writes the sensing attributes to the table node (rate-limited)
void upload_sensing_streams(int table_node_id,
                            std::span<const Eigen::Vector3f> candidate_pts,
                            std::span<const Eigen::Vector3f> residual_pts,
                            int residual_mass,
                            float explanation_ratio,
                            int frame_id);
```

### 10.2 VoxelProcessor changes

**Adoption trigger** (new method):

```cpp
// Returns track IDs that meet the adoption threshold
std::vector<int> find_adoption_ready_tracks(int min_frames,
                                            int min_voxels,
                                            const std::string& category) const;
```

Threshold parameters (configurable):
- `adoption_min_frames`: 90 (3 seconds at 30 fps)
- `adoption_min_voxels`: 200

**Candidate stream extraction** (new method):

```cpp
// Near-surface subsample of the full track for the queue seeding upload
std::vector<Eigen::Vector3f> extract_candidate_stream(
    int track_id,
    std::span<const GraphObjectBox> model_boxes,
    std::size_t max_pts,
    float surface_band_m) const;
```

Selects voxel centroids whose distance to the nearest model box surface is within
`surface_band_m` (e.g. 0.08 m). Falls back to full `get_points_clustered` if the surface band
yields fewer than `min_pts`.

**Adoption handoff** (in compute loop):

```cpp
// After upload confirmed, delete local evidence
for (const int tid : adoption_ready)
{
    voxel_grid_.remove(tid);
    active_tracks_.erase(tid);
}
```

**Full-sample refresh** (bypass suppression for one window):

```cpp
if (scene_processor_.is_full_sample_requested(table_node_id))
    bypass_suppression_for_track(category, refresh_window_frames_);
```

### 10.3 compute() changes (SpecificWorker)

```cpp
// Existing: fetch graph object boxes for suppression
const auto graph_object_boxes = scene_processor_.get_graph_object_boxes(...);

// New: adoption check
const auto adoption_ready = voxel_processor_.find_adoption_ready_tracks(
    config_.adoption_min_frames, config_.adoption_min_voxels, "table");
for (const int tid : adoption_ready)
{
    const auto candidates = voxel_processor_.extract_candidate_stream(tid, graph_object_boxes, 400, 0.08f);
    const auto residuals  = voxel_processor_.get_current_residual_pts(400);
    scene_processor_.upload_sensing_streams(table_node_id, candidates, residuals,
                                            residual_mass, explanation_ratio, frame_id);
    voxel_processor_.remove_adopted_track(tid);
}

// New: delta residual upload (every 30 frames, after adoption)
if (frame_id % 30 == 0 && table_adopted_)
{
    const auto residuals = voxel_processor_.get_current_residual_pts(150);
    scene_processor_.upload_sensing_streams(table_node_id, {}, residuals,
                                            residual_mass, explanation_ratio, frame_id);
    voxel_processor_.remove_residual_tracks_near_models(graph_object_boxes);
}
```

### 10.4 Config additions (etc/config.toml)

```toml
[VoxelProcessor]
adoption_min_frames   = 90
adoption_min_voxels   = 200
candidate_surface_band_m = 0.08
refresh_window_frames = 60
upload_period_frames  = 30
```

---

## 11. table-concept Agent Structure

### 11.1 Component layout

```
table_concept/
  src/
    specificworker.h/.cpp       orchestration, DSR callbacks, compute loop
    table_model.h/.cpp          generative model, SDF, Free Energy, gradient step
    sample_queue.h/.cpp         circular queue, weight management, eviction/insertion
    epistemic_planner.h/.cpp    coverage map, viewpoint computation, gain estimation
    prior_store.h/.cpp          load/save object_priors.toml + convergence checkpoint
  etc/
    config.toml
    object_priors.toml
  generated/                    (robocomp generated ICE stubs)
  CMakeLists.txt
  robot_table_concept.cdsl
```

### 11.2 Compute loop (table-concept)

```
each compute() call:
  ① Read DSR
       for each table node:
         read candidate_pts_att, residual_pts_att, residual_mass_att
         read explanation_ratio_att, last_sensing_frame_att
         check if data is fresh (frame delta < staleness_threshold)

  ② Queue update
       for each fresh candidate point:
         compute |SDF(θ_current, candidate)|
         if < surface_band_m: eligible for insertion
       insert best eligible candidates, evicting lowest-weight queue entries

  ③ Model update (gradient step)
       compute F(θ) = Σᵢ SDF(θ,yᵢ)²/σ² + λ‖θ−θ_prior‖²_Σ
                      (sum over queue + residual points)
       θ ← θ − η ∇_θ F(θ)    (clamped per-parameter)
       re-evaluate queue weights with new θ

  ④ Write model params to DSR
       if ‖Δθ‖ > write_threshold:
         update RT edge, width_m, depth_m, height_m
         increment model_generation_att
       write free_energy_att

  ⑤ Convergence check
       if F < F_min for K frames AND all_faces_covered:
         write model_stable_att = true
         clear any pending epistemic proposals

  ⑥ Epistemic planning (if not stable)
       compute coverage per face from queue weights
       find lowest-coverage face fₖ
       compute v* = face centre + outward normal × d_obs
       compute ΔH(v*) from projected visible area
       if ΔH(v*) > gain_threshold AND NOT epistemic_pending:
         write epistemic proposal attributes
         set epistemic_pending = true

  ⑦ Refresh check
       if F rising for M frames AND explanation_ratio < 0.3:
         set request_full_sample_att = true  (triggers full re-sample in robot_concept)
```

### 11.3 Key parameters

**Generative model (state prior)**

| Parameter | Default | Description |
|---|---|---|
| `prior_table_width` | 1.0 m | Prior width W for size regulariser |
| `prior_table_depth` | 0.6 m | Prior depth D for size regulariser |
| `prior_table_height` | 0.75 m | Prior table surface height |
| `prior_size_std` | 0.15 m | 1-σ uncertainty on prior dimensions |

**VFE optimiser**

| Parameter | Default | Description |
|---|---|---|
| `σ_obs` | 0.05 m | Observation noise (SDF likelihood width) |
| `λ_size` | 0.5 | Size prior strength (deviate from typical dimensions) |
| `λ_pos` | 0.05 | Position transition prior (resist large per-frame moves) |
| `λ_state` | 0.02 | Size state transition prior |
| `λ_angle` | 0.01 | Angle transition prior |
| `optimization_iters` | 10 | Gradient-descent iterations per update cycle |
| `optimization_lr` | 0.05 | Learning rate |
| `grad_clip` | 2.0 | Gradient clipping threshold |

**Historical sample queue (binned, validated from prototype)**

| Parameter | Default | Description |
|---|---|---|
| `num_angle_bins` | 24 | Angular sectors around table centre in XY (15° each) |
| `num_z_bins` | 10 | Height slices (0.1 m each) |
| `max_per_bin` | 2 | Best points retained per bin (quality-sorted) |
| `sdf_threshold_for_storage` | 0.08 m | Only near-surface points admitted |
| `min_frames_before_historical` | 10 | Wait for pose convergence before admitting any points |
| `historical_warmup_frames` | 50 | Ramp to `max_new_points_per_frame` over this many frames |
| `max_new_points_per_frame` | 5 | Max new points admitted per frame at full speed |
| `rfe_alpha` | 0.98 | Exponential decay factor for Remembered Free Energy |
| `rfe_max_threshold` | 2.0 | Hard evict points exceeding this RFE value |
| `edge_bonus_weight` | 0.3 | Priority boost for edge/corner points in quality metric |
| `edge_proximity_threshold` | 0.05 m | Distance to face to count as "close" for edge score |

**Lifecycle and convergence**

| Parameter | Default | Description |
|---|---|---|
| `confidence_decay` | 0.90 | Per-frame confidence decay when not observed |
| `confidence_decay_confirmed` | 0.98 | Slower decay for confirmed (high-confidence) beliefs |
| `confidence_boost` | 0.15 | Confidence increase on each matched observation |
| `confirmed_threshold` | 0.70 | Confidence above which belief is "confirmed" |
| `confirmed_grace_frames` | 50 | Frames before confirmed beliefs start decaying |
| `F_min` | 0.02 | Free energy threshold for convergence |
| `K_stable` | 30 | Frames below F_min required before declaring stable |

**Epistemic actions**

| Parameter | Default | Description |
|---|---|---|
| `δ_min` | 20 | Minimum face coverage count (points per face) |
| `gain_threshold` | 0.1 | Minimum ΔH to emit an epistemic proposal |
| `d_obs` | 1.8 m | Target observation distance from low-coverage face |
| `staleness_threshold` | 90 frames | Max frame gap before sensing data considered stale |


---

## 12. End-to-End Scenario Walk-Through

### Scene: robot enters room with one table on the left, table not yet in DSR

```
t=0     table-concept starts, reads object_priors.toml
        bootstrap_table_1 missing from DSR → creates node with prior geometry and RT

t=1     robot_concept starts, room_T_zed available
        SceneProcessor sees table node → model box available
        VoxelProcessor: table track begins accumulating (suppression NOT active yet
        because no confident model; suppression activates after adoption)

t=90    track has 200+ voxels, 90 frames old → adoption trigger fires
        extract_candidate_stream → 400 near-surface pts
        upload to candidate_pts_att
        voxel_grid_.remove(track_id)   ← local voxels deleted
        active_tracks_.erase(track_id)

t=91    table-concept reads candidate_pts_att
        seeds circular queue with 400 pts
        first gradient step: θ moves slightly from prior toward real table position
        model_generation_att incremented

t=91–   robot_concept: new table pts arrive → suppressed by model box
t=180   residual pts (near far side) accumulate → delta upload every 30 frames
        table-concept: queue evicts low-weight adoption-time pts
        inserts fresh near-surface residual pts → gradient continues
        F decreasing

t=180   coverage map: south face of table has coverage 5 < δ_min = 20
        compute v* = south_face_centre + (-ŷ) × 1.8 m
        ΔH(v*) = 0.34 > gain_threshold
        write epistemic proposal to DSR
        set epistemic_pending = true

t=181   mission-controller reads proposal, robot is idle → approves
        navigation target written to robot node

t=210   robot arrives at v*, south face now visible to camera
        robot_concept: new near-surface pts from south face → candidate stream
        table-concept: inserts south-face pts → south coverage rises to 35
        model constraint now symmetric → F drops sharply

t=250   all faces above δ_min, F < F_min for 30 frames
        model_stable_att = true
        no new epistemic proposals
        robot_concept voxel budget: 50 residual voxels (vs 2000+ at adoption)
```

---

## 14. Python Prototype Reference

A working Python prototype implementing the full Active Inference loop for multiple object types,
including tables, exists at:

```
/home/pbustos/robocomp/components/robocomp-shadow/agents/ainf_agents/box_concept/
```

### 14.1 Key source files

| File | Role |
|---|---|
| `src/objects/table/belief.py` | `TableBelief` — 7-param state, SDF, `add_historical_points`, `_add_to_bins`, `_compute_edge_score`, `update_rfe`, `get_historical_points_in_robot_frame` |
| `src/objects/table/sdf.py` | `compute_table_sdf` — compound SDF (box top + 4 cylindrical legs) |
| `src/objects/table/manager.py` | `TableManager` — thin wrapper over `BeliefManager`; DSR `update_dsr()` stub |
| `src/belief_manager.py` | `BeliefManager` — VFE optimizer, DBSCAN clustering, Hungarian association |
| `src/belief_core.py` | `Belief`, `BeliefConfig` — abstract base with lifecycle, covariance, process noise |
| `src/transforms.py` | Room↔robot frame transforms and Jacobians |

### 14.2 Component-to-component porting map

| Python prototype | C++ table-concept module | Notes |
|---|---|---|
| `TableBeliefConfig` (dataclass) | Config struct in `specificworker.h` or `table_model.h` | Add DSR upload rate fields |
| `compute_table_sdf` (PyTorch) | `table_model.cpp` | Replace torch ops with Eigen; keep clamp logic |
| `TableBelief.compute_prior_term` | `table_model.cpp::prior_energy()` | Direct port |
| `TableBelief.apply_constraints` | Clamp after each grad step | One line: `leg_length = clamp(leg_length, 0.05f, table_height − TOP_THICKNESS)` |
| `TableBelief._add_to_bins` | `sample_queue.cpp::insert()` | Eigen; room frame throughout (drop robot-frame transform) |
| `TableBelief._compute_edge_score` | `sample_queue.cpp::edge_score()` | Direct port |
| `TableBelief.update_rfe` | `sample_queue.cpp::update_rfe()` | Direct port; robot_cov from DSR |
| `TableBelief.get_historical_points_in_robot_frame` | `sample_queue.cpp::weighted_points()` | Room frame: drop the robot↔room Jacobian; just return room-frame pts + weights |
| `BeliefManager._optimize_belief` (autograd) | `table_model.cpp::gradient_step()` | Replace autograd with 7-param finite differences (central, δ = 1 × 10⁻⁴) |
| `BeliefManager._associate_clusters` (Hungarian) | Already in robot_concept | robot_concept does association before uploading to DSR |
| `BeliefManager.update_lifecycle` | Confidence decay in compute loop | Direct port |

### 14.3 Frame convention change

The prototype transforms historical points to **robot frame** for optimisation (because it receives
LiDAR in robot frame) and stores them in **room frame**. In `table-concept`, all voxel centroids
from `robot_concept` arrive already in **room frame**. The Jacobian propagation step
(`compute_jacobian_room_to_robot`) in `get_historical_points_in_robot_frame` is therefore
unnecessary — the gradient weight reduces to:

$$
w_i = \frac{1}{1 + \mathrm{tr}(\Sigma_{\text{capture},i}) + \text{RFE}_i}
$$

where `Σ_capture` is the robot XY covariance at the frame the point was stored.

### 14.4 What does not exist in the prototype (genuinely new)

- DSR attribute read/write protocol (`candidate_pts_att`, `residual_pts_att`, etc.)
- Voxel adoption handoff (upload → `voxel_grid_.remove(track_id)`)
- Epistemic action proposal (`epistemic_planner.h`)
- Scaffolding with `object_priors.toml` and convergence checkpoint
- `request_full_sample_att` refresh protocol
- robot_concept redetection guard (`awaiting_table_node` flag)


1. **Pose update authority**: does `table-concept` write the RT edge directly, or propose a delta
   that `room_concept` applies? Direct write is simpler; proposal-based keeps a single RT
   authority.

2. **Multi-table discovery**: if `table-concept` sees a large persistent residual far from the
   current model box (detected via a separate cluster in `residual_pts_att`), should it create a
   new table node autonomously, or delegate discovery to a supervisor? The answer determines
   whether `table-concept` is a pure fitter or also a hypothesis generator.

3. **Adoption confirmation**: the current protocol is optimistic (upload then delete). Should
   deletion wait for an acknowledgement from `table-concept` (e.g. a flag on the node) to avoid
   data loss if `table-concept` crashes immediately after the upload?

4. **Upload frame**: all point cloud attributes are stored in room frame. Confirm that
   `room_concept`'s room RT definition is the same frame used by `robot_concept`'s
   `room_T_zed` transform chain.

5. **Multiple table instances**: the design handles one instance. With N tables, `SceneProcessor`
   must match each voxel track to the closest table node and manage N separate upload/adoption
   paths. The matching criterion is Euclidean distance between track centroid and node RT origin.
