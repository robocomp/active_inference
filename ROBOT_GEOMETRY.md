# Shadow robot — frames & sensors (reference)

Ground truth is `robot_concept/shadow.json` (the DSR bootstrap). This file summarises it so agents
don't have to re-derive the tree each time. If the two disagree, `shadow.json` wins — re-check and
update this doc.

## RT frame tree (parent → child, translation m, rotation deg)

```
root  (world datum; z=0 at the FLOOR)
└── Shadow   type "robot"   T[ 0,     0,     0    ]  R[0,   0, 0]   ← robot base, AT floor level
    └── body type "body"    T[ 0,     0,    +0.0325] R[0,   0, 0]   ← = the Webots world placement
        ├── 210  (no type)  T[ 0,     0,     0    ]  R[0,   0, 0]
        ├── zed   rgbd       T[ 0,   -0.075, +0.945]  R[0,   0, 0]
        ├── imu   imu        T[ 0,     0,     0    ]  R[0,   0, 0]
        ├── ricoh rgbd       T[ 0,   -0.170, +1.275]  R[0,   0, 0]
        ├── helios laser     T[ 0,   -0.155, +1.075]  R[0,   0, 0]   ← UPRIGHT, high
        └── bpearl laser     T[ 0,   +0.140, +0.670]  R[0, 180, 0]   ← flipped 180°: points DOWN
```

### ★ THE MOUNT INVARIANT — read this before "correcting" any number above

This is a SIMULATOR: `webots-shadow/protos/Shadow.proto` is the generative model of every sensor
reading, so it is ground truth and any disagreement is a model error on our side. The rule that ties
the two together, and the reason each number is what it is:

> **`body` = the robot's placement height in the .wbt world (0.0325 m).**
> **Then every sensor's `body`-frame z EQUALS its `Shadow.proto` translation z, exactly.**

    sensor   Shadow.proto (robot frame)   shadow.json (body frame)
    ricoh    0 -0.170 1.275               1.275   ✓
    zed      0 -0.075 0.945               0.945   ✓
    helios   0 -0.155 1.075               1.075   ✓   ← was 1.030; corrected 2026-07-30
    bpearl   0  0.140 0.670               0.670   ✓

`helios` was the sole outlier at 1.030 while the other three already matched, i.e. someone copied the
proto values in and helios's went stale. **If you ever see a sensor whose `body` z differs from its
proto z, that sensor is wrong — not this rule.**

Two traps that produced hours of wrong conclusions, so do not repeat them:

1. **Do not compare proto z against shadow.json z directly.** They are in different frames — the proto
   is the ROBOT frame (`DEF shadow Shadow { translation … 0.0325 }` in every apartment world; the proto's
   own default field is 0.033), shadow.json is the `body` frame. Comparing them raw makes three correct
   sensors look 45 mm wrong and the one wrong sensor look correct. Apply the `body` offset first.
2. **Do not copy the proto ROTATIONS.** Both lidars differ from shadow.json by *exactly* 90.0° about Z
   (helios `0 0 1 -4.7116`, bpearl `-1 -1 0 3.14159`). Two sensors differing by precisely the same
   90° is a Webots-`Lidar`-local-axis vs RoboComp frame-convention offset that is already absorbed
   downstream — **not** two independent mounting errors. Transcribing them literally would introduce a
   90° error, not remove one. (A real 90° yaw error would rotate every obstacle a quarter turn around
   the robot: instantly and catastrophically visible, not a subtle bias.)

**How to verify the mounts in 30 s** (no code, no rebuild): restart `room_concept` and read its one-shot
startup line. `bpearl` measures the floor HEAD-ON and is the only trustworthy z reference:

    [FloorCheck] OK: … | bpearl floor at world z = -10 mm (within 60 mm, 83083 pts) | helios ref 130 mm

`bpearl` within ±20 mm of 0 ⇒ the mounts are right. The histogram bins at 20 mm, so ±10 mm readings on
either side of zero are the same measurement, not a drift. **`helios ref` proves nothing about mounts** —
its floor is all grazing, so it reads 130–170 mm depending only on where the robot happens to stand
(measured at both values, same session, same correct mounts). Do not use it as a mount check.

## Frame conventions (the load-bearing facts)

- **`Shadow` (type `"robot"`) is THE localization frame.** room_concept optimises `room ← Shadow`
  and publishes the `robot ↔ room` RT edge onto this node. It derives its lidar/optimisation frame
  from the type-`"robot"` node (`RoomConfig.LIDAR_ROBOT_FRAME` empty ⇒ auto-derive), so "the frame we
  fit in" == "the node we write the edge onto" — no base offset in the published pose.
- **`body` is the de-facto base frame every OTHER agent hardcodes** (`get_transformation_matrix(
  "room","body")` in table/bottle/chair/voxelizer/self_cal). `Shadow → body` is a **pure +3.25 cm z**
  translation (no x/y/yaw), so `Shadow` and `body` are **identical in the plane (x, y, yaw)** and
  differ only in z. Any SE(2)/planar quantity is the same in either; only true-3D height cares.
- **`root` is the world datum at the floor** (z=0). `Shadow` sits at the floor; `body` at +3.25 cm —
  which is where the .wbt world actually places the robot, NOT an arbitrary modelling offset. It was
  +4.5 cm until 2026-07-30, which put every sensor 12.5 mm high; see the mount invariant above.
- All sensors hang off **`body`** (not `Shadow`).
- **`zed` frame axes**: x-right, y-**DEPTH**, z-up (NOT optical z-forward). Deproject:
  `px=(col−cx)·d/fx, py=depth, pz=(cy−row)·d/fy`.

## Sensors

| node | type | mount (on `body`) | what it is / used for |
|------|------|-------------------|-----------------------|
| `zed` | rgbd | +0.945 m, upright | ZED stereo RGB-D (pinhole). Masks/depth for concept fits; the "measurement" camera. |
| `ricoh` | rgbd | +1.275 m, upright | Ricoh 360 camera, **equirectangular** proj (`cam_fov≈2π`). Peripheral attention only (biased centroid). |
| `helios` | laser | +1.075 m, upright | High 360 spinning LiDAR. **Walls** (upper band) + floor-at-range. Media plane **id 0**. Grazing far-floor returns bias its floor estimate high (13–17 cm, and it MOVES with the robot's position because the bias grows with range) — not a mount error; use for walls, never as a floor datum or a mount check. residual_concept fits its floor plane from bpearl ONLY for this reason, and drops helios returns below `Clusterer.HeliosFloorZ0` so its own floor cannot latch as obstacle. |
| `bpearl` | laser | +0.670 m, **pointing DOWN** (R=[0,180,0]) | Low 360° dome under the tray (spreads 360° around + ~90° solid down). Legs / low obstacles / floor. Media plane **id 1**. **Good floor-datum sensor** — but detect the floor as the EXTENDED flat plane (the low z that wraps all azimuths), NOT the densest z-bin (that's near clutter/structure), and skip the helios 0.8 m self-hit cut (it strips bpearl's floor disc). room's startup check does exactly this. |
| `imu` | imu | at `body` | IMU. |

## Media plane

Zero-copy DDS on **domain 7** (isolated from the DSR `Agent.domain` 0). LiDAR is split per device:
`helios` (plane 0) + `bpearl` (plane 1), each published in its DEVICE frame; consumers transform
device→target via the RT tree with `rc::media::LidarPlaneReader`. Domain/topic come from the media
descriptor JSON on the sensor node, never from config. See CLAUDE.md "Media plane" section.

**LiDAR floor/ceiling filtering moved to the consumers (2026-07-12).** `robocomp-robolab/.../hardware/
laser/lidar3d_dds` runs `MeshFilter`, which does TWO things: (1) **self-return removal** — `footprint`
disc + `skirt` + robot-mesh Embree raycast (STAYS at source; only the driver knows the robot mesh); and
(2) a **scene z-band cut** `Floor.z`/`Top.z` that dropped the floor (≤110 mm) and ceiling (≥2100 mm).
The floor/ceiling are informative (startup LiDAR calibration, ground plane, occupancy), so the z-band cut
is now **disabled** in the live webots configs (`config_helios_webots.toml` + `config_bpearl_webots.toml`:
`[Floor] z = -100000`, `[Top] z = 100000`) — self-filter unchanged. Restore `110`/`2100` to revert.
Consequences: the real floor + the 2.4 m ceiling now reach every consumer, so **each agent must ground-
filter for itself** (room bands to [1.5,2.0] for walls; residual/controller already z-band). room's
startup floor/ceiling calibration check is re-enabled and now meaningful. (jetson/real-robot lidar3d_dds
configs still carry the old cut — propagate there deliberately if wanted.)

## ★★ THE ROBOT'S SHAPE — one source, and why there were seven (2026-08-01)

Everything above is FRAMES. This section is SHAPE, and it is the part that caused a full day of
misdiagnosis: the robot collided with unmodelled objects and clipped wall corners, and every fix
attempted at the controller was irrelevant because the fault was in what "the robot's shape" meant
three layers below.

### ★ THE ROBOT IS A COMPOSITION, NOT A MESH

`webots-shadow/protos/Shadow.proto` builds the robot from a **body mesh + four wheel assemblies +
camera/lidar solids**, 58 nested `Pose`/`Solid`/`HingeJoint` nodes, geometry as 28 `IndexedFaceSet`
+ 2 `Mesh` + 1 `Box` + 1 `Cylinder`. **No single mesh file is the robot.** Any consumer handed one
mesh has been handed a fraction of the robot and told it was the whole thing.

★★CORRECTION 2026-08-01 (later the same evening). The paragraph that used to sit here claimed the
mesh gap and a population of ~550 surviving returns per cycle were the SAME phenomenon. **That was
wrong, and it was my own instrument that produced the illusion.** Recorded in full because the failure
mode is more useful than the claim was:

- The `body_clear_bearing_deg` column is the bearing of `argmin(d − support_radius(dir))`. For returns
  at d ≈ 0 that reduces to `argmax(support_radius)` — a property of the POLYGON, not of the returns.
  On the shipping footprint that maximum is at **−52° (0.3249)** and **+51° (0.3245)**, which are
  exactly the two histogram peaks (465 and 452 cycles) I read as a physical bearing cluster.
- The same returns' DIRECT bearing (`raw_nearest_bearing_deg`) is **near-uniform across all 12 bins**,
  52% in [−90,+90] — i.e. noise, which is what r ≈ 1 mm must give.
- Their radius is `raw_nearest_m` p50 = **0.00124 m**: essentially ON the rotation axis, not at 0.2 m.
- The z band IS solid: p05 0.494 / p50 0.5148 / p95 0.5351, on 984 of 984 cycles.

So the mesh gap and the return population were correlated by INFERENCE, never by joint measurement.
Three separate diagnoses were built on that inference and all three were wrong. ★A metric that mixes a
measurement with a property of the model it is measured against cannot be read as evidence about the
world — and `min_lidar_all_m` (2-D) and `n_in_body` (projected footprint) failed the same way earlier
the same day. Three flawed instruments, one lesson.

★THE r ≈ 0 POPULATION IS STILL UNEXPLAINED. It is not the mesh gap. A bpearl beam aimed at the axis at
z = 0.514 terminates on the body at 0.0358 m (robot r = 0.116, z = 0.643), so the proto assembly does
not generate it either. The mount chain was verified correct end to end, including the `Rz(+90°)` at
`specificworker.cpp:464-466`, which reproduces the proto's helios `0 0 1 -4.7116` to 0.045° and bpearl
`-1 -1 0 3.14159` exactly. Diagnosing it needs a per-return radius/bearing histogram, which nothing
currently logs.

### What the proto assembly actually contains (measured, `webots_proto_loader`, 2026-08-01)

| | triangles | x | y | z |
|---|---|---|---|---|
| `shadow.stl` alone | 18 809 | ±0.2386 | ±0.2300 | 0.0034 … 1.3089 |
| **assembly (visual)** | **28 745** | **±0.2464** | ±0.2300 | **−0.0348** … 1.3089 |
| assembly + boundingObject | 29 153 | ±0.2600 | ±0.2300 | −0.0348 … 1.3350 |

★The assembly adds the four wheels (9 936 triangles) and NOTHING between z = 0.065 and z = 1.3089 —
the wheels top out at 0.0648 m. So at waist height the assembly's silhouette is **bit-identical** to
the body mesh's. Only `boundingObject` fills the −90…+90 bins there, and that is the PHYSICS hull,
which a Webots LiDAR cannot see (Webots ray-casts the graphics) — switching it on would delete real
returns in a shell around the robot, which is the 0.55 m disc's failure mode wearing a new hat. It is
wired as `proto_include_bounding_objects`, default **false**.

★The body's collision geometry is NOT the STL: it is a `boundingObject Group` of a 0.44×0.46×0.76 box
at z = 0.38 plus a 0.1×0.1×0.61 mast at (0, −0.17, 1.03). Visual and collision differ; do not assume
either is "the robot" without saying which question you are asking.

### The mesh files — which is which (say STL/OBJ/DAE, never "the mesh")

| file | what it is | size | extent (m) |
|---|---|---|---|
| `webots-shadow/protos/meshes/shadow.stl` | the Webots **BODY** solid — what the simulator renders as the body. **Byte-identical (md5) to the driver's copy** `lidar3d_dds/robots/Shadow/shadow.stl` | 18 809 tri | x ±0.2386, y ±0.2300, z 0.0034…1.3089 |
| `robot_concept/meshes/shadow.obj` | the **DISPLAY** mesh (viewers, voxelizer, graph3d). Fuller, but still body-only | 47 530 faces / 141 364 v | x ±0.2420, y −0.2306…+0.2471, z **−0.0500**…1.3671 |
| `.../meshes/shadow.dae` | referenced by the proto alongside the STL | — | — |

★The OBJ is authored with **z-min at −0.050 m** (it hangs below the floor plane) and the project's own
viewers recentre it before display. The driver applies `mesh_off_z = +0.0534` to put its z-min at the
STL's 0.0034. Its x is exactly symmetric (±0.2420) so its ORIGIN is right in the ground plane; the y
asymmetry (−0.2306…+0.2471) is real front geometry, NOT an origin error — recentring y would shift the
mesh 8 mm and reintroduce a silent offset of the same class as the 45 mm helios error.

### ★ THE ROBOT IS NOT A COLUMN — measured height bands (proto assembly, exact z-slab clipping)

| band (m) | \|x\|max | \|y\|max | circumscribed | inscribed | area (m²) |
|---|---|---|---|---|---|
| 0.0–0.1 | **0.2464** | 0.2229 | 0.3322 | 0.2229 | 0.2139 |
| 0.1–0.2 | 0.1770 | 0.2229 | 0.2846 | 0.1770 | 0.1578 |
| 0.2–0.3 | 0.1810 | 0.2260 | 0.2895 | 0.1810 | 0.1636 |
| 0.3–0.4 | 0.1450 | 0.2187 | 0.2579 | 0.1364 | 0.1028 |
| 0.4–0.5 | 0.1082 | 0.2114 | 0.2329 | 0.0488 | 0.0561 |
| **0.5–0.6** | 0.0789 | 0.2058 | **0.2158** | 0.0030 | **0.0301** |
| 0.6–0.7 | 0.1433 | 0.2150 | 0.2199 | 0.1310 | 0.0864 |
| 0.7–0.8 | 0.2403 | 0.2300 | 0.2870 | 0.2081 | 0.1821 |
| 0.8–1.2 | ≤0.0883 | ≤0.2227 | ≤0.2214 | ≤0.0156 | ≤0.0215 |
| 1.2–1.4 | 0.0241 | 0.1796 | 0.1806 | 0.0187 | 0.0011 |

★Area varies **194×** across bands. The flat projected polygon is **7.2× the true area** at waist
height and 1.52× too wide in circumscribed radius there. That is real clearance thrown away beside
anything at table/counter height.

★**The wheels are the outermost geometry ONLY below z = 0.10 m**, where they add **69 mm per side**
over the body mesh (0.2464 vs 0.1770 — the STL's wheel wells are cut out at 0.1770). Above 0.10 m the
body is outermost. That is entirely below both LiDARs' plausible reach, which is why the self-filter
probably never sees the wheels while the PLANNER absolutely must account for them.

### ★ THE WHEEL AXIS IS SETTLED — 25 mm per side recovered

`robot_footprint.h` bounded the wheels ORIENTATION-INDEPENDENTLY at |x| = 0.2716 because "the axis
direction could not be settled by reading the proto alone" (candidates 0.2460 / 0.2600 / 0.2600).
Measured per-shape bounds from the assembly settle it:

    visual wheel     x half-extent 0.036 (= 0.072 width), y,z half-extent 0.050 (= radius)  -> axis LATERAL
    boundingObject   half-extents 0.050, 0.050, 0.036                                       -> axis VERTICAL

The VISUAL wheel is axis-lateral — the only orientation that puts it on the floor, and consistent with
`HingeJoint axis -1 0 0`. So the answer is the **0.2460 candidate, confirmed to 0.4 mm at 0.2464**, and
the shipping bound is **25.2 mm pessimistic per side** — against a measured p05 clearance of 58 mm,
that is 43% of the working margin, per side.

⚠**The proto's own collision shape disagrees with its own visual wheel by 90°** (a flat puck lying
where the wheel stands; upstream Cyberbotics protos carry the same `Pose{rotation 1 0 0 -1.5708}`).
The usual rule "use collision geometry for a navigation envelope" assumes the two agree. Here they do
not, so the visual figure is the defensible one; the conservative alternative is 0.2600, still 11.6 mm
tighter than shipping.

### Derived footprint — DO NOT hand-transcribe this

Emitted by `WebotsProtoLoader::xy_hull()`; 36-vertex raw hull simplified to 12 at **+0.18% area**, with
outward-only simplification so it is a strict superset and can never shrink clearance.

    area 0.2182 m²   inscribed 0.2300   circumscribed 0.3278   |x|max 0.2464   |y|max 0.2307
    vs shipping:  area 0.2343   inscribed 0.2299   circumscribed 0.3252   |x|max 0.2716

Same inscribed radius — the passable half-width is unchanged, set by |y| — with 6.9% less area and
25 mm narrower per side. `xy_hull_band(z0,z1)` gives the bands above. ★Adopting it is a BEHAVIOURAL
change to the planner's collision test: the robot will fit through gaps it currently refuses. It wants
its own run and its own comparison, not to be bundled with anything else.

### The seven definitions of the robot's shape (2026-08-01 inventory)

| consumer | shape | status |
|---|---|---|
| `lidar3d_dds` `[Footprint]` | 0.55 m disc | **DISABLED 08-01** (was a 22 cm blind shell; see below) |
| `lidar3d_dds` `[Skirt]` | 0.75 m disc, low z-band | STILL ON |
| `lidar3d_dds` Embree query | one mesh + `Dilate` 0.05 | being replaced by the proto assembly |
| controller ESDF self-filter | `max(esdf_self_filter_radius, body_extent_max()+0.08)` = **0.4052 m**, `+0.08` HARDCODED | still on — **do not remove before the driver is correct** |
| controller ESDF self-filter | box 0.32/0.42/0.24 | still on |
| `common/robot_footprint` `shadow()` | hand-transcribed 18-vertex polygon | **32.5 mm adrift** from the mesh it claims to come from (max radius 0.3252 vs mesh hull 0.3105; worst direction 5°: 0.2845 vs 0.2520) |
| MPPI collision | `esdf < support_radius(−∇esdf)` | a PROXY, exact only w.r.t. the single nearest cell; ≤9.5 cm unchecked in reachier directions |

Each disc in that list is a compensation for the layer beneath it being wrong, and each one hides the
next. That is why removing the 0.55 disc exposed the mesh's gaps, and fixing the mesh placement then
exposed that the mesh is body-only.

### The self-filter chain, and the 45 mm landmine

`lidar3d_dds/src/mesh_filter.cpp` drops a return if ANY of: inside `[Footprint]` disc · inside
`[Skirt]` disc · outside `[Floor]`/`[Top]` (disabled) · `point_hits_mesh` (6 axis rays of length
`Dilate` = 0.05 m — a PROXIMITY test, so it detects nearness to a SURFACE, not containment).

★The 0.55 m disc was never geometry. It was added because the mesh query was missing self-returns —
because the helios mount was 45 mm stale (`shadow.json` 1.075 vs the config fallback `tz = 1030`) and
**a 45 mm offset against a 50 mm `Dilate` leaves 5 mm of margin**. The mount is now correct and loaded
(`[Mount] loaded static mount 'helios' from shadow.json`, verified live), so the disc was pure leftover.
★**The config fallback `tz = 1030` is STILL STALE in both webots configs and the rollback is silent
(stderr only).** If `shadow.json` ever fails to load, the 45 mm error returns invisibly. Set it to 1075.

MEASURED, before the disc was disabled: `nearest_lidar_m` median **0.550**, p01 0.530 over 4067 cycles —
the distribution pinned at the config value. Circumscribed body radius 0.3252 m ⇒ a **0.22 m annulus
around the body in which anything without a map representation simply ceased to exist.** Walls and
residual hulls survived only because they are injected as GEOMETRY, not sensed.

### Standing instrument — the near-body census

`controller/proximity_obstacles.csv` columns `body_clear_m` (signed distance to the body surface,
negative = inside), `body_clear_z`, `body_clear_bearing_deg`, `n_in_footprint`, `n_in_body`,
`n_in_hull`, `n_under_floor`.

★TRAP, hit twice on 2026-08-01: `min_lidar_all_m` is a **2-D horizontal** distance from the robot's
ORIGIN — a downward sensor puts floor returns at plan-view radius ≈0 on every cycle (median 0.0012 m),
which is expected and says nothing about self-hits. And `n_in_body` uses the **projected** footprint,
so it counts returns beside the narrow waist as if they were inside the body. Only `n_in_hull` — which
compares each return against the radius AT ITS OWN HEIGHT — means what its name says. Two separate
wrong diagnoses were built on the first two before this was understood.

### ★ OPEN / PENDING

- **Proto assembly** (`webots_proto_loader`, `geometry_source = "proto"`) — the real fix: build the
  Embree scene from ALL of `Shadow.proto`. Acceptance test = the −90…+90 bins at z∈[0.49,0.54] are no
  longer empty. **UNRUN as of this writing.**
- ★**The proto is the SIMULATOR's robot.** The real robot may be described by a different file (URDF /
  other proto / CAD export). If the two diverge the self-filter is correct in simulation and wrong on
  hardware. Treat the geometry source as a per-deployment input, never a constant.
- Derive `RobotFootprint::shadow()` from the assembly instead of transcribing it (kills the 32.5 mm
  drift and the stated wheel-axis guess).
- Height-banded footprint for planner + MPPI, replacing the projection.
- Controller's 0.4052 m self-filter floor: shrink ONLY after the driver assembly is verified — it is
  currently the only thing keeping interior returns out of the ESDF.
