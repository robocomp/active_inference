# SHADOW.robot Conventions

This document summarizes axis directions, signs, and sensor conventions used in this component for Shadow.

## 1) Robot Frame (Shadow)

Frame used in control and localization:
- Y+ is forward
- X+ is right (lateral)
- Z+ is up

Velocity sign conventions:
- adv (or adv_y): forward speed in m/s. Positive means forward.
- side (or adv_x): lateral speed in m/s. Positive means right.
- rot: angular speed in rad/s.

Command mapping to OmniRobot:
- setSpeedBase(advx, advz, rot) is called as:
  - advx = side * 1000 (mm/s)
  - advz = adv * 1000 (mm/s)
  - rot = rot (rad/s)

Left-right rotation sign:
- Positive rot is left turn (counterclockwise in XY seen from +Z).
- Negative rot is right turn (clockwise in XY seen from +Z).

Notes:
- With Y+ as forward, heading/error angles for control use atan2(x, y)
  (equivalently atan2(dx, dy) for a displacement vector [dx, dy]).
- Standard geometric yaw in an X-forward convention uses atan2(y, x).
- Pose state is [x, y, theta] in room/map XY.

## 2) LiDAR Conventions (HELIOS and BPEARL)

Configured proxies:
- HELIOS: Proxies.Lidar3D on port 11990
- BPEARL: Proxies.Lidar3D1 on port 11989

Roles in this component:
- HELIOS (high lidar): localization + MPPI support
- BPEARL (low lidar): obstacle contribution for MPPI

Configured names and filters:
- HELIOS name: helios
- BPEARL name: bpearl
- HELIOS max range: 100 m
- BPEARL max range: 100 m
- HELIOS height filters:
  - keep points with z < 2.2 m for low/high aggregate
  - high-only subset uses z > 1.2 m

Read pipeline:
- Both sensors are requested asynchronously in parallel.
- Points are converted from mm to m.
- Points inside robot body footprint are removed using robot semi-width.
- Buffer slots:
  - slot 1: HELIOS high subset
  - slot 2: combined low/high obstacle set after adding BPEARL

## 3) Camera Conventions

### 3.1 Camera frame used in this component

Camera coordinates follow:
- x is lateral
- y is forward (depth)
- z is up

Projection equation:
- u = cx + fx * (x / y)
- v = cy - fy * (z / y)

### 3.2 Intrinsics

Source:
- Real-time image intrinsics come from ImageSegmentation TImage fields focalx and focaly.

Runtime use:
- fx = timg.focalx, fy = timg.focaly (validated)
- fallback if invalid: fx = 0.9 * width, fy = 0.9 * height
- principal point: cx = width / 2, cy = height / 2

Synthetic overlay defaults:
- width = 640, height = 640
- fx = 576, fy = 576
- cx = 320, cy = 320
- Later replaced by runtime values when available.

### 3.3 Extrinsics (camera in robot frame)

Config keys in etc/config:
- camera_tx = 0.0
- camera_ty = -0.11
- camera_tz = 0.92

Meaning:
- Camera origin offset expressed in robot frame, in meters:
  - tx: lateral offset
  - ty: forward offset
  - tz: height

World/robot to camera conversion used by overlay:
- p_cam.x = p_robot.x - tx
- p_cam.y = p_robot.y - ty
- p_cam.z = z_world - tz
