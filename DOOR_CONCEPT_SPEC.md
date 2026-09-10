# door_concept — object spec (input to CONCEPT_AGENT_RECIPE §3 generation)

Date 2026-07-26. Static door v1 (no swing angle yet). Wall-anchored, parametrised in the CONTAINING
WALL FRAME. Template tier: `cabinet_concept` (wall-frame machinery), but the belief is a SINGLE PANEL,
not a run — so cabinet's run/kitchen model (`kitchen_cells`, `lshape_split`, `wall_run_belief`, N-arm,
tier) is **dropped**; we reuse cabinet's wall-frame (`cabinet_geometry.h`), scene-graph, presence,
projection, lidar-cue, existence scaffolding.

```yaml
object:
  name:            door
  dsr_node_type:   object          # generic node type; class in object_subtype
  object_subtype:  door
  node_prefix:     door_           # door_1, door_2, …
  detection_labels: [door]         # ZED YOLO / YOLO-sem "door" masks (semantic-mask stage already emits door)
  agent_id:        15              # next free (registry: 5..14 taken, 20..23 taken; 15–19,24+ free)

frame:                             # ★ the door is expressed in the CONTAINING WALL FRAME, not room-free
  anchor:          wall            # associate to the nearest room-polygon wall (like cabinet)
  wall_frame:      { origin: near_corner, u: along_wall (corner→corner), n: inward_normal, z: up }
  # room-frame door geometry is DERIVED from (wall, s, w, h); the fit runs in the wall frame.

belief:
  dofs:                            # N = 3 fitted DOFs, all in the WALL frame
    - {name: s,  kind: length}     # distance from the wall's near CORNER to the door's near edge (along u)
    - {name: w,  kind: length}     # door width  (along u)
    - {name: h,  kind: length}     # door height (along z, base pinned to floor)
  # cz (base) pinned to floor z=0; lateral offset off the wall plane pinned to 0 (door lies IN the wall);
  # yaw pinned to the wall direction u. So the free surface is exactly (s, w, h) — 3 DOF.
  sdf:                             # ONE thin box panel in the wall/local frame
    - {prim: box, name: panel,
       centre: [s + w/2, 0, h/2],  # u = s+w/2 from corner; n = 0 (in wall plane); z = h/2
       half:   [w/2, T/2, h/2]}    # thin through the wall (T), spans w along u, h up
  constants: {T: 0.05}             # door thickness (fixed, non-fitted)
  canonicalize:    none            # a door in a wall has no w↔h / ±90° symmetry (wall fixes orientation)
  accumulate_extra: none           # v1: no structural GN factor (priors + panel SDF suffice)

priors:                            # ★ TWO STRONG priors (user): standard door
  s:       {mean: from_detection, sigma: 0.60}    # broad: position along the wall is what we localise
  w:       {mean: 0.70,           sigma: 0.06}    # STRONG — 70 cm standard leaf
  h:       {mean: 2.00,           sigma: 0.08}    # STRONG — 200 cm standard
  clutter: 0.05
  # sigma_pos/sigma_size feed <Obj>Model.AI2* ; w/h priors dominate unless a very clean mask overrides.

support:
  rests_on_surface: true           # door base rests on the floor
  anchored_dof:     cz             # base z pinned to floor (0), height h grows up from it

dynamics:
  model: static                    # transition() = I. Swing angle θ is a documented FUTURE DOF (see below).

sensors:                           # ★ multi-sensor policy: ZED births + drives geometry; others CUE only
  primary:   { source: zed_masks,  role: birth + geometry }     # depth_var==0 ZED "door" masks
  cue:       { source: [lidar, ricoh_masks], role: confirm/attention only, never birth/reshape }
  # ZED-only birth gate (birthable = depth_var==0); ricoh depth_var → R downweight; lidar first-hit = cue.

epistemic:
  target:        hidden-face       # face the robot hasn't resolved (a grazing wall view sees the door edge-on)
  degenerate_dof: w                # a face-on view most resolves width (edge-on collapses it)

capabilities:                      # ALL except open/close cross-affordances (deferred)
  dashboard:      true
  affordance:     true             # "look/verify" epistemic affordance ONLY — NO openable/open/close yet
  existence:      true             # support-mass removal belief (occupancy confirms / free-space removes)
  room_prior:     true             # inherent: wall-anchored → needs the room delimiting_polygon
  be_still:       true             # continuous ego-motion common-mode + confirm_only gate (per recipe)
  projection:     true             # camera ROI + pixel silhouette (existence evidence)
  lidar:          true             # cue-only first-hit range channel
  evaluator:      false
  ai2_csv:        true
  epistemic_csv:  true
  open_close:     DEFERRED         # swing angle θ DOF + openable affordance — v2

future (v2):                       # not built now — reserved so the model extends cleanly
  add_dof:   {name: theta, kind: angle, hinge: {edge: s | s+w, prior: 0 (closed), sigma: wide}}
  # θ rotates the panel about the hinge edge in the wall plane's normal; adds an "openable" affordance.
```

## Generation plan (per §3), delta vs a plain cabinet copy

1. **id = 15 FIRST** (registry updated: door_concept=15).
2. Copy `cabinet_concept` module set → `door_concept`, token-rename cabinet→door / Cabinet→Door; keep
   `common/ai_belief/recursive_laplace.h`, `common/mask_ingestor`, `common/instance_tracker`,
   `common/existence_belief`, cabinet's **wall-frame geometry** (`*_geometry.h` → `door_geometry.h`).
3. **DROP** the run/kitchen machinery (`cabinet_kitchen*.h`, `cabinet_lshape_split.h`,
   `cabinet_wall_run_belief.h`, N-arm/tier logic). The door belief is a **single wall-anchored panel**.
4. **Author `DoorBelief`/`DoorModel`**: 3-DOF (s,w,h) in the wall frame; ONE box `sdf_prim`;
   `responsibilities` = [panel, clutter]; strong w/h priors; cz floor-pin; yaw = wall dir.
   `self_test()` recovers a synthetic (s,w,h).
5. Fill priors (w=0.70/σ0.06, h=2.00/σ0.08, s broad), label `"door"`, epistemic (hidden-face, deg=w),
   scene-graph attrs (width_m/height_m/depth_m + wall-frame pose), `object_subtype="door"`,
   `mesh_path="door_concept/meshes/door.obj"`.
6. Capabilities: dashboard, affordance(look-only), existence, room_prior, be_still, projection, lidar,
   ai2_csv, epistemic_csv. NO open/close.
7. `.cdsl` (`options dsr`, no Webots), CMake (list every src incl `door_belief.{cpp,h}`, nuke cache),
   build, `DoorBelief::self_test()` PASS.

★Key model decision to confirm: **3 free DOFs (s, w, h) in the wall frame** — position-along-wall +
width + height, with yaw/lateral/base all pinned by the wall+floor. Width & height are strongly-prior'd
so the fit mainly localises `s` and lightly corrects the standard leaf size. This is the "static door
relative to the containing wall frame" the spec calls for; θ (swing) slots in later as a 4th DOF.
