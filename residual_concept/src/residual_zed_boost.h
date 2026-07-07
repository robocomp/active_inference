/*
 * residual_zed_boost.h  —  dense ZED-depth boost for residual obstacles.
 *
 * When a residual box falls in the ZED camera's field of view, we PROJECT the box into the depth image,
 * BACKPROJECT the depth in that pixel region into 3D, transform to the room frame, and keep the points that
 * land inside the box (+margin) — rejecting the background behind it. Those dense, accurate points are then
 * appended to the belief's likelihood (frame.points), so the box is refined against ZED depth wherever the
 * camera sees it (a much denser signal than the sparse LiDAR), while LiDAR still covers the 360° the ZED
 * cannot. Pure Eigen/STL, DSR-free → unit-testable in isolation (self_test()).
 */

#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

// A ZED depth frame: per-pixel depth in METRES, row-major. NOTE the DSR "zed" frame in THIS codebase is
// x-right, y-DEPTH(optical axis), z-up (see voxelizer/graph_publisher.cpp) — NOT the standard optical
// z-forward. The projection below follows that convention.
struct DepthImage
{
    std::vector<float> depth;   // metres; ≤0 / non-finite = invalid
    int width  = 0;
    int height = 0;
};

// Pinhole intrinsics (rectified), y-depth zed frame: u = fx·x/y + cx, v = cy − fy·z/y (depth = y).
struct CamIntrinsics { float fx = 0, fy = 0, cx = 0, cy = 0; };

// The room-frame box to boost: oriented footprint (centre, yaw, extents) + vertical band.
struct BoostBox { float cx = 0, cy = 0, yaw = 0, w = 0, d = 0, z_min = 0, z_max = 0; };

struct ZedBoostParams
{
    int   subsample     = 3;      // take every Nth pixel in u and v (dense depth → keep the count sane)
    float xy_margin_m   = 0.10f;  // keep a backprojected point if within this of the footprint edge
    float z_margin_m    = 0.10f;  // ...and within this of the [z_min,z_max] band
    float min_depth_m   = 0.30f;  // ZED near limit
    float max_depth_m   = 5.0f;   // ZED depth degrades with range² — ignore beyond this
    int   max_points    = 1500;   // cap per instance (keeps the GN fit bounded)
    // ── whole-FoV scene backprojection (feeds DBSCAN, not box-gated) ──
    float voxel_m         = 0.03f;   // leaf size for voxel-downsampling the FoV cloud (equalises density vs LiDAR)
    int   fov_subsample   = 4;       // pixel stride for the FoV pass (coarser than the box pass — the whole image)
    int   fov_max_points  = 8000;    // hard cap on the downsampled FoV cloud fed to clustering
};

// Project `box` into the image, backproject the depth in that pixel region, and return the room-frame 3D
// points that fall inside the box (+margins). `room_T_cam` maps camera→room (4×4). Empty if the box is
// behind the camera / out of frame / no valid depth there.
std::vector<Eigen::Vector3f> backproject_box_region(const DepthImage& depth, const CamIntrinsics& K,
                                                    const Eigen::Matrix4f& room_T_cam, const BoostBox& box,
                                                    const ZedBoostParams& params);

// Backproject the WHOLE depth FoV into room-frame points, voxel-downsampled to `params.voxel_m` and capped
// at `params.fov_max_points`. NOT box-gated — this is the dense scene cloud that (after specialist-explain
// and instance-claim) is concatenated with the LiDAR residual and fed to DBSCAN, so a dense forward surface
// (a tabletop) clusters as ONE blob instead of fragmenting into LiDAR ring-arcs. Floor/walls are removed by
// the worker's explainers, not here.
std::vector<Eigen::Vector3f> backproject_fov(const DepthImage& depth, const CamIntrinsics& K,
                                             const Eigen::Matrix4f& room_T_cam, const ZedBoostParams& params);

// Voxel-downsample a point cloud: one representative (cell centroid) per `leaf_m`³ cell, capped at max_out.
// Deterministic (sorted cell traversal) — no RNG. Exposed for reuse/tests.
std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& pts, float leaf_m, int max_out);

bool zed_boost_self_test();

}  // namespace rc
