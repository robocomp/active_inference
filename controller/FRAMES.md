# Frames And Conventions

This component mixes several coordinate systems. The safest rule is: always state both the source frame and the target frame before applying a transform or drawing anything.

## 1. RoboComp robot/body frame used here

- This is the working convention for RGBD points and local robot motion in this component.
- `+X`: robot right
- `+Y`: robot forward
- `+Z`: up
- Positive yaw is mathematical CCW on the ground plane.
- The RGBD point cloud handled in `robot_concept` is already robot-aligned in this convention. Do not treat it as an optical-camera frame.

## 2. Room frame

- `room_T_robot = get_transformation_matrix(room, robot)` means `room <- robot`.
- Room polygons, robot pose, and voxelized detections must all end up in this same room-local frame before rendering.
- In the current pipeline, voxel points are transformed with robot rotation and camera translation so they land in room coordinates.

## 3. DSR / RT semantics

- Read `A_T_B` as: coordinates of `B`, expressed in frame `A`.
- `get_transformation_matrix(room, robot)` returns robot pose expressed in room frame.
- `room.json` defines the static chain. Relevant current edge:
  - `root -> Shadow`: identity
  - `Shadow -> zed`: translation `[0, -0.075, 0.945]`, yaw `+1.57 rad`

## 4. RoboComp OpenGL viewer in this component

- The custom viewer does not render room coordinates directly.
- Current mapping used by `voxel_opengl_viewer` is:
  - `X_gl = -X_room`
  - `Y_gl = Z_room`
  - `Z_gl = Y_room`
- This means the rendered scene is mirrored in X with respect to the room frame.
- The robot marker, room polygon, and voxels must all use the same mapping, or left/right will look wrong.

## 5. Qt3D mapping used elsewhere

- For Qt3D rendering, use:
  - `X_qt3d = -x`
  - `Y_qt3d = h`
  - `Z_qt3d = y`
- For a standard planar yaw `theta`, the Qt3D `Ry` angle used in this codebase is:
  - `alpha = pi + theta`

## 6. Webots convention

- Webots uses a right-handed world with:
  - `+X` forward
  - `+Y` left
  - `+Z` up
- This is not the same as the RoboComp body convention used here.
- When comparing Webots poses with RoboComp body/room poses, be explicit about the axis swap and sign change.

In `room_concept`, incoming odometry is explicitly normalized to the local RoboComp body convention before fusion:

- `odom.adv = -pose.adv`
- `odom.side = pose.side`
- `odom.rot = -pose.rot`

This keeps downstream math coherent with `+Y` forward and CCW-positive yaw.

## 7. room_concept pose/RT semantics

- Internal localization state is handled as `room <- robot` (robot pose expressed in room frame).
- In DSR writing, `room_concept` may publish the inverse (`robot <- room`) when the room node is parented under the robot.
- Keep this distinction explicit: internal estimate frame and graph edge direction are not always the same thing.

## 8. Legacy upstream note (ainf_slamo only)

- `SceneGraphModel` has a non-standard 2D rotation convention:
  - `ydir = (cos(theta), sin(theta))`
  - `xdir = (sin(theta), -cos(theta))`
- `object_footprints.h` uses the standard planar rotation matrix:
  - `R(yaw) = [[cos, -sin], [sin, cos]]`
- Do not mix these two conventions when computing object footprints or viewer geometry.
- This note is for upstream/legacy integrations and is not the active runtime path of `robot_concept` + `room_concept`.

## 9. Practical checklist

Before trusting a result, answer these four questions:

1. In which frame are the raw points or poses currently expressed?
2. Which transform is being applied: `target <- source` or the opposite?
3. Is the consumer expecting room coordinates, robot/body coordinates, OpenGL coordinates, or Qt3D coordinates?
4. Is there an extra mirror or axis remap in the renderer?

If one of these answers is implicit, make it explicit in code or in a comment next to the transform.