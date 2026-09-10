# OpenGL Voxel Viewer Alignment — Debug Summary

## Project
`/home/pbustos/robocomp/components/active_inference/robot_concept` — C++23, Qt6/OpenGL 3.3, Eigen, RoboComp DSR graph.

## Goal
Display a semantic voxel cloud (table + chair detected by a ZED RGBD camera) **correctly aligned** with the room polygon outline in an OpenGL 3D viewer.

---

## DSR Graph structure
```
root (dummy)
└── Shadow  (type: robot)   ← robot_node_name_ = "Shadow"
    ├── room  (type: room)   ← room_node_name_ = "room"
    │     attrs: delimiting_polygon_x_att, delimiting_polygon_y_att, room_height_att=2.4m
    ├── zed   (type: rgbd)   ← ZED stereo camera
    └── lidar3D (type: laser)
```

---

## OpenGL viewer frame
- Y=up, floor at Y=0, ceiling at Y=room_height
- Room-frame → OpenGL mapping: `{p.x(), p.z(), p.y()}` (Z_room→Y_gl, Y_room→Z_gl)
- Camera pitch = `+0.35f` (above scene looking down)
- File: `src/voxel_opengl_viewer.cpp`

---

## Voxel transform — `src/specificworker.cpp` ~line 440
RGBD points arrive in **zed frame**. Transformed with:
```cpp
inner_eigen_api->get_transformation_matrix(room_node_name_, "zed")
// chains: zed → Shadow → room
```
Result: voxels in **room-local frame**. This is believed correct — the table and chair shapes look right.

---

## Polygon transform — `src/specificworker.cpp` ~line 617
`delimiting_polygon_x/y` corners are defined in a **canonical room frame** (X+ right, Y+ forward, CCW, room centre = origin). This frame was defined at SLAM build time and is **not** necessarily the same as the live room-DSR frame. Currently applying:
```cpp
inner_eigen_api->get_transformation_matrix(room_node_name_, robot_node_name_)
// room_T_robot = full rotation + translation
p_room = T.linear() * p_robot + T.translation()
```

---

## Current visual result
- Room polygon **shape** is correct (L-shaped room with dent)
- Voxel **shapes** are correct (table rectangle + chair dot)
- The polygon and voxels are in the same ballpark but still **rotated ~90° relative to each other** and possibly offset
- In Webots reality: robot is near the long wall facing it; dent is at the bottom; table's long axis is along the bottom wall
- In viewer: table appears to face a different direction

---

## What has been tried (polygon grounding block in `specificworker.cpp`)
1. No transform — polygon ungrounded, rotation wrong
2. Rotation-only `R * p` via `room_T_robot` — angle improved but offset remained
3. Full `T * p` (rotation + translation) via `room_T_robot` — current state, still ~90° off

---

## Active diagnostic prints in binary
```
[RT room<-Shadow] t=(x,y,z) rpy_xyz=(r,p,y)      ← every 10 compute cycles
[VoxelDiag] raw[i] room=(x y z)  opengl=(x y z)   ← first 5 voxels every 60 calls
[PolygonDiag] floor[i] OpenGL=(x,y,z)             ← every polygon refresh
[RoomPolygon] Polygon grounded via full room_T_robot: t=(x,y)
```

---

## The core question
**What frame are `delimiting_polygon_x/y` actually defined in?**

The attribute values come from the SLAM (ainf_slamo component) which populates the room node. The polygon is defined with room centre = origin, X+ right, Y+ forward, CCW winding. The open question is whether this canonical frame aligns with the frame produced by `get_transformation_matrix(room, "zed")` — i.e. whether the polygon X/Y axes map to the same axes as the voxel X/Y.

**Suggested next step**: Print the raw (untransformed) polygon corners alongside the raw voxel positions and the full `room_T_robot` RT matrix (4×4), so the relationship can be derived analytically.

---

## Build
```bash
cd /home/pbustos/robocomp/components/active_inference/robot_concept
cmake --build build -j$(nproc)
# binary at: bin/robot_concept
```
