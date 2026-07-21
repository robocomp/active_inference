/*
 * residual_floor_plane.h  —  data-driven floor-plane estimation for the occupancy grid's floor explainer.
 *
 * WHY: the grid's "is this return an obstacle?" test uses a FIXED nav band referenced to room z=0:
 *   in_band ⇔ z > floor_z0 + floor_slope·range. That assumes the floor is exactly the room's z=0 plane. In a
 * NEW scenario the floor can sit OFFSET or TILTED relative to z=0 (localization datum, an uneven floor, a small
 * pitch/roll in the room fit). Then a large swath of floor reads ABOVE the fixed band → every floor return
 * becomes a HIT → latches → and the z-aware clearing makes those phantoms permanent. The robot paints floor
 * everywhere it drives and can't navigate.
 *
 * FIX (AI2-aligned — infer the floor, don't hard-code it): robustly estimate the dominant floor PLANE
 * z=a·x+b·y+c from the low LiDAR returns each cycle (trimmed least-squares that rejects legs/low obstacles),
 * temporally smoothed. The grid then references its band to floor_z(x,y)=a·x+b·y+c instead of 0, so the floor
 * explainer FOLLOWS the actual floor — offset or tilted — in any scenario. When the floor really is flat at z=0
 * the fit returns (0,0,0) and behaviour is identical to the fixed band (zero regression).
 *
 * Pure Eigen/STL, DSR-free → unit-testable in isolation (floor_plane_self_test()).
 */

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace rc
{

// A floor plane z = a·x + b·y + c (room frame). `valid` is false until a trustworthy fit exists.
struct FloorPlane
{
    float a = 0.0f, b = 0.0f, c = 0.0f;
    bool  valid = false;
    // Fit diagnostics (of the LAST accepted fit — carried over unchanged when the fit is held/rejected), so the
    // quality of the estimate is visible on disk and not only as a stdout line.
    int   n_candidates = 0;      // floor candidates that survived the robust trim
    float rms          = 0.0f;   // RMS residual of those candidates to the fitted plane (m) — floor roughness
    float z_at(float x, float y) const { return a * x + b * y + c; }
};

struct FloorPlaneParams
{
    bool  enabled         = false;   // master flag (config FloorPlane.Enabled). When off the estimator still RUNS
                                     //   and LOGS (so the offset is visible), but the grid keeps the fixed band.
    float candidate_band_m = 0.25f;  // a return is a floor CANDIDATE if within this of the current band top (above
                                     //   the nominal floor) — wide enough to catch an offset floor, tight enough to
                                     //   exclude tall obstacles before the robust trim runs.
    float trim_k          = 2.5f;    // after each fit, drop candidates whose residual > trim_k·MAD (legs/clutter)
    int   iters           = 2;       // robust reweighting rounds
    int   min_candidates  = 80;      // fewer than this ⇒ keep the previous plane (don't fit on noise)
    float ema             = 0.10f;   // temporal smoothing toward the new fit (slow → stable; 1 = no smoothing)
    float max_offset_m    = 0.30f;   // |c| sanity clamp — reject a wild fit (keep previous) beyond this
    float max_tilt        = 0.15f;   // |a|,|b| sanity clamp (≈8.5° over 1 m) — reject a wild tilt
};

// Estimate the floor plane from a room-frame point cloud (LiDAR — precise, best for the floor). `origin` is the
// sensor position (for horizontal range in the candidate band). `floor_z0`/`floor_slope` are the SAME nominal band
// the grid uses; candidates are points within `candidate_band_m` above floor_z(prev). `prev` is last cycle's plane
// (for candidate selection + EMA). Returns the smoothed plane; if the fit is untrustworthy it returns `prev`
// unchanged (so a bad cycle never moves the floor).
FloorPlane estimate_floor_plane(const std::vector<Eigen::Vector3f>& points_room, const Eigen::Vector3f& origin,
                                float floor_z0, float floor_slope, const FloorPlaneParams& p, const FloorPlane& prev);

bool floor_plane_self_test();

}  // namespace rc
