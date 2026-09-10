# RETINA

This document describes the main logic of the `retina` agent and the math used in the current implementation.

## 1. Purpose

`retina` builds a semantic 3D representation of the scene in room coordinates.

It combines:

- RGB + depth from the `zed` node
- semantic masks from YOLO segmentation
- optional `lidar3D` support points
- model boxes already present in DSR
- a semantic voxel grid and a local OpenGL viewer

The component does two jobs at the same time:

1. Build and maintain semantic voxel tracks for objects such as tables, chairs, and monitors.
2. Compare those sensed objects against graph/model objects, especially tables, and publish:
   - `candidate_pts_att`: the sensed table cloud
   - `residual_pts_att`: points not explained by the current model
   - `explanation_ratio_att`: how much of the sensed cloud is inside the current model box

## 2. Main Runtime Flow

The main loop lives in `SpecificWorker::compute()`.

Each cycle does the following:

1. Read RGBD and transforms from DSR.
2. Read `lidar3D` points and transform them into room coordinates.
3. Run YOLO segmentation on the RGB image.
4. Convert only masked depth pixels into 3D room-frame points.
5. Associate detections with persistent tracks.
6. Update the voxel grid with the non-explained part of each detection.
7. Optionally attribute nearby `lidar3D` points to table tracks and fuse them into the voxel grid.
8. Match sensed table tracks against graph `model_table` boxes.
9. Write candidate and residual point clouds back to DSR table nodes.
10. Update the 3D viewer with voxels, graph boxes, model meshes, residual points, `rfe_pts`, and candidate cloud overlays.

The main classes are:

- `SpecificWorker`: orchestration and DSR writes
- `SceneProcessor`: DSR reads, transforms, and viewer updates
- `YoloProcessor`: semantic segmentation
- `VoxelProcessor`: masked unprojection, track association, suppression, candidate extraction
- `UnifiedVoxelGrid`: semantic voxel storage and ownership model
- `LidarTrackAttributor`: optional assignment of `lidar3D` points to sensed table tracks
- `VoxelOpenGLViewer`: local 3D visualization

## 3. Coordinate Frames

The agent uses several coordinate systems. The active conventions are:

### Robot/body frame

- `+X`: right
- `+Y`: forward
- `+Z`: up

### Room frame

This is the common world frame used for:

- graph/model boxes
n- voxel points
- table candidate and residual point clouds
- viewer geometry before OpenGL remapping

### Transform notation

`room_T_robot` means `room <- robot`.

A point in robot coordinates is moved to room coordinates as:

`p_room = R_room_robot * p_robot + t_room_robot`

### OpenGL viewer remapping

The custom viewer does not render room coordinates directly. It uses:

- `X_gl = -X_room`
- `Y_gl = Z_room`
- `Z_gl = Y_room`

So the scene is mirrored in X when drawn.

See `FRAMES.md` for the full frame reference.

## 4. RGBD to 3D Math

The RGBD pipeline intentionally computes 3D points only for pixels owned by a YOLO mask.

### 4.1 Pixel ownership

YOLO provides masks and bounding boxes.

For each pixel `(u, v)`:

- if depth is invalid, skip it
- if the pixel is not inside any accepted YOLO mask, skip it
- if the point range is too far from the detection median range, skip it

This is a masked-only unprojection strategy.

### 4.2 Unprojection

Depth is interpreted in the camera API convention currently used by this component:

- forward axis is `+Y`
- vertical axis is `+Z`

For depth value `d`, focal lengths `fx`, `fy`, and image center `(cx, cy)`:

- `p_cam.x = (u - cx) * d / fx`
- `p_cam.y = d`
- `p_cam.z = (cy - v) * d / fy`

This gives a 3D point in the ZED/camera frame.

### 4.3 Room transform

The point is moved to room coordinates with:

- `p_room = R_room_zed * p_cam + t_room_zed`

An optional vertical correction is then applied:

- `p_room.z += z_lift_m`

### 4.4 Range filtering

The unprojected range is:

- `r = sqrt(x^2 + y^2 + z^2)`

The point is kept only if:

- `0.01 <= r^2 <= 100.0`
- `|r - median_range_for_detection| <= 0.35 m` when a valid median exists

This rejects many bad depth samples near mask borders.

## 5. Detection-to-Track Association

Each accepted semantic detection gets a room-frame centroid computed from the full sensed cloud for that detection.

### 5.1 Cost function

Tracks are associated with detections using a Hungarian assignment.

For detection `i` and track `j`:

- if labels differ, cost is a large impossible value
- otherwise cost is Euclidean centroid distance

So:

- `cost(i, j) = ||c_det_i - c_track_j||`

### 5.2 Acceptance gate

A matched pair is accepted only if:

- `cost <= track_association_max_distance_m`

Otherwise a new track is created.

### 5.3 Track centroid smoothing

When a match is accepted, the track centroid is updated with an exponential moving average:

- `c_new = 0.65 * c_old + 0.35 * c_det`

This stabilizes track identity across frames.

## 6. Unified Voxel Grid Model

`UnifiedVoxelGrid` stores a semantic voxel state per occupied cell.

### 6.1 Voxel indexing

Given resolution `res`, a point is quantized to a voxel key as:

- `k_x = floor(x / res)`
- `k_y = floor(y / res)`
- `k_z = floor(z / res)`

### 6.2 Per-voxel state

Each voxel stores:

- centroid
- owning `track_id`
- Dirichlet-like semantic counts `alpha`
- observation counts and frame counters
- best confidence / best category

### 6.3 Semantic update

Each observation increases the count of the chosen category in `alpha`.

The dominant semantic label of a track is computed by aggregating voxel beliefs over all voxels owned by that track.

### 6.4 Ownership

A voxel belongs to one track at a time.

If the same voxel is later observed for another track, ownership is transferred and the category belief is reset to the prior.

### 6.5 Visibility update

The grid also supports a ray-based visibility update.

The math uses an AABB slab test to find traversed voxels and delete stale ones. That part is infrastructure for the voxel grid model, even though the most visible logic in this component is the positive observation path.

## 7. Track Boxes and Duplicate Merging

Track boxes are built from the current point set of each voxel track.

For each track:

- compute `min`, `max`, and centroid from its points
- use clustered points when the track is small enough
- keep only tracks with enough voxels and enough points

### 7.1 Duplicate test

Two track boxes are considered duplicates when:

- categories match
- the intersection volume is significant relative to the smaller box
- centroids are sufficiently close

The overlap ratio is:

- `overlap_ratio = intersection_volume / min(volume_a, volume_b)`

The current duplicate gate is approximately:

- `overlap_ratio >= 0.30`
- centroid distance below a size-dependent threshold

If two tracks look duplicate, voxel ownership is reassigned from one track to the other.

## 8. Model Suppression

This component keeps a separation between:

- the full sensed table cloud used for table capture and tracking
- the residual voxel cloud that remains after removing points already explained by a graph model

### 8.1 Point-level suppression during voxelization

While building per-detection point clouds, a point is skipped if it is already explained by a graph model box of the same semantic category.

For tables, this means model-covered points are not inserted into the live residual voxel grid.

### 8.2 Residual-track suppression

After track boxes are built, any track box that overlaps a matching graph model box is removed from the live voxel grid.

Important detail:

- the voxel payload is removed
- the sensing-side track identity is kept

This is why the current table `best_track` id can remain persistent while the viewer still shows only residual voxels.

## 9. Optional lidar3D Fusion

`lidar3D` points are read from DSR, transformed into room coordinates using the robot transform at the lidar timestamp, and then optionally fused into table tracks.

### 9.1 Timestamp-aware transform

For a lidar point in robot coordinates:

- `p_room = R_room_robot_lidar * p_robot + t_room_robot_lidar`

The transform can be interpolated in time.

### 9.2 Attribution math

`LidarTrackAttributor` considers only sensed table tracks.

A lidar point is attributed to a track only if:

1. It lies inside an expanded AABB gate around the track.
2. Its distance to the track centroid is below a radius based on the track diagonal.

The centroid gate is:

- `max_centroid_dist = 0.5 * diag(track_box) + centroid_radius_extra_m`

If several table tracks pass the test, the nearest centroid wins.

Low-support assignments are discarded if the track gets fewer than `min_points_per_track` lidar points.

### 9.3 UI toggle

There is a UI button that enables or disables inclusion of `lidar3D` support points in voxel updates. This affects voxel fusion, not the raw lidar viewer overlay.

## 10. Table-to-Model Matching

One of the main responsibilities of the agent is to compare sensed table clouds against graph `model_table` objects.

### 10.1 Graph model box

For each graph table node, the code builds a room-frame oriented box from:

- width
- depth
- height
- `room_T_object`

The axis-aligned `min` and `max` are computed from transformed corners, while the true box orientation is kept as `yaw_rad` plus `half_extents`.

### 10.2 Best sensed table track

For each graph table node, the best sensed table track is chosen by:

1. largest AABB overlap with a padded model AABB
2. tie-breaker: largest sample count inside the oriented model box
3. tie-breaker: smallest XY centroid distance

The AABB overlap volume is:

- `overlap = volume( intersect(track_aabb, padded_model_aabb) )`

### 10.3 Oriented box test

To test whether a point belongs to the model box, the point is rotated into the local frame of the model using the inverse yaw:

- `rel = p - centroid_model`
- `lx = rel.x * cos(-yaw) - rel.y * sin(-yaw)`
- `ly = rel.x * sin(-yaw) + rel.y * cos(-yaw)`

The point is inside if:

- `|lx| <= half_extent_x`
- `|ly| <= half_extent_y`
- `|rel.z| <= half_extent_z`

### 10.4 Explanation ratio

The candidate cloud stored on the table node is checked against the model OBB.

If `inside_count` candidate points are inside the OBB and the total candidate count is `n_cands`, then:

- `explanation_ratio = inside_count / n_cands`

This quantity answers: how much of the sensed table cloud is already explained by the current graph model?

## 11. Residual Point Extraction

Residual points represent geometry near the table that is not explained by the current model.

### 11.1 Neighborhood gate

Residual extraction is restricted to a padded neighborhood of the model AABB:

- `nb_min = model_min - margin`
- `nb_max = model_max + margin`

with the current margin set to `0.3 m`.

### 11.2 Residual condition

A lidar point is a residual if:

1. it lies in the model neighborhood
2. it is outside the oriented model box

So the residual set is:

- `Residual = { p | p in neighborhood(model) and p not in OBB(model) }`

The number of stored residual points is capped.

### 11.3 Fallback

If no lidar frame is available, the code falls back to candidate cloud points outside the model OBB but still inside the neighborhood.

## 12. DSR Outputs Written By This Agent

### Table nodes

For matched graph tables, the agent writes:

- `candidate_pts_att`: sensed table cloud
- `residual_pts_att`: unexplained nearby points
- `residual_mass_att`: number of residual points
- `last_sensing_frame_att`: frame id of the last update
- `explanation_ratio_att`: candidate fraction explained by the model

### Voxels node

The agent also writes a semantic voxel export to a `semantic_grid` node named `voxels`.

The stored format is stride-5:

- `[x, y, z, prob, track_id]`

for each exported voxel.

## 13. Viewer Layers

The OpenGL viewer currently shows several layers.

### Geometry layers

- semantic voxels
- raw lidar points
- room polygon
- tracked object boxes
- graph/model boxes
- object meshes
- robot mesh / robot marker

### Table-memory overlays

The table overlays currently use different colors:

- residual points: magenta
- `rfe_pts`: green
- candidate cloud: cyan

The candidate cloud is drawn as a separate overlay so it can be compared visually against the model box and mesh.

## 14. Why The Pipeline Is Structured This Way

There are three different geometric products in this agent, and they should not be confused:

1. `sensed table cloud`
   - full YOLO-masked table observation in room frame
   - used for table matching and explanation scoring

2. `live residual voxel grid`
   - only the part not already explained by graph models
   - used for residual scene understanding and viewer voxels

3. `table residual cloud`
   - lidar or fallback points near the model but outside the model OBB
   - used to show unexplained support around a modeled table

This split is intentional. It prevents the live voxel grid from being dominated by geometry that the graph already knows about, while preserving enough sensed geometry to evaluate model fit and residual structure.

## 15. Practical Reading Guide

If you need to debug the agent, start here:

- `SpecificWorker::compute()` for the global control flow
- `SceneProcessor::get_rgbd_frame_from_dsr()` for RGBD access and intrinsics
- `VoxelProcessor::process_rgbd_frame()` for masked unprojection and voxel updates
- `VoxelProcessor::associate_detections_hungarian()` for track identity
- `SpecificWorker::update_table_nodes_from_tracks()` for candidate/residual table logic
- `LidarTrackAttributor::attribute_points()` for optional lidar support fusion
- `VoxelOpenGLViewer` for room-to-OpenGL remapping and overlay rendering

## 16. Summary

The core math of `retina` is:

- project masked depth into 3D
- move all geometry into room coordinates
- associate detections to persistent tracks by Hungarian centroid matching
- maintain a semantic voxel grid with track ownership and semantic belief
- suppress geometry already explained by graph models from the live voxel grid
- compare the full sensed table cloud against graph model boxes using AABB overlap and oriented-box tests
- extract and visualize residual geometry around modeled tables

That is the current operational definition of the agent.
