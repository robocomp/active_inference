/*
 * nvblox_bridge.h  —  residual_nvblox's only door to NVIDIA nvblox.
 *
 * Plain C++ (no nvblox / CUDA / Eigen types) on purpose: residual links libdsr, whose Eigen ABI is pinned in
 * src/CMakeLists.txt, and nvblox ships its own Eigen + CUDA flags. Everything nvblox stays inside libnvblox_bridge.so.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace nvb
{

struct LidarModel
{
    int   azimuth_divisions = 900;
    int   rows              = 32;
    float min_range_m       = 0.1f;
    float below_rad         = 0.0f;   // lowest beam below the sensor's horizontal plane (positive)
    float above_rad         = 0.0f;   // highest beam above it (positive)
    // The driver's own blind volume (robot frame, vertical cylinders around the robot axis): returns inside it
    // are DROPPED (bpearl [Footprint]/[Skirt]), so a ray crossing it proves nothing there. Voxels inside may rise
    // during this sensor's integration but never fall. r = 0 ⇒ off.
    float blind_r = 0.0f, blind_z0 = 0.0f, blind_z1 = 0.0f;
    float skirt_r = 0.0f, skirt_z0 = 0.0f, skirt_z1 = 0.0f;
    std::vector<float> ring_elev_deg;   // real per-ring elevations (needs nvblox_lidar_rings.patch); empty = uniform rows
    int sensor_id = 0;                  // key of this sensor's self-mask
};

// SELF-MASK: the robot's own body is fixed in each sensor's frame, so the range-image pixels that see it — the body
// itself, the mixed returns along its edge (a beam half on the body, half on the floor reports a range in between,
// i.e. a point floating along the ray) and the rays the driver's body filter emptied — are always the same ones.
// Calibration (robot with >= clear_r_m of free floor all around): over `sweeps` sweeps a pixel is SELF if it returns
// within clear_r_m horizontally above floor_z + floor_tol_m in >= min_frac of them, or stays empty in >= 90 % of them
// although its ray meets the floor within 3 m; it is dilated (dilate_az columns, dilate_ring rows): its points
// within clear_r_m horizontally are dropped (beyond, the beam got past the body) and FloorFill never fills it. A pixel always empty although its ray meets the floor further out (up to
// max_range_m) is NO-FILL only: FloorFill skips it, but a real return there still counts. Saved to a file.
struct SelfMaskParams
{
    int   sweeps      = 100;
    float clear_r_m   = 1.0f;
    float floor_tol_m = 0.05f;
    float min_frac    = 0.3f;
    int   dilate_az   = 2;
    int   dilate_ring = 1;
};

struct Params
{
    float voxel_m       = 0.05f;
    float p_free        = 0.30f;      // inverse sensor model
    float p_occ         = 0.70f;
    float half_width_m  = 0.05f;      // occupied band around a return
    float max_range_m   = 10.0f;
    bool  decay         = false;      // decay() moves every voxel toward P=0.5 by these per call
    float decay_occ_p   = 0.45f;      // [0, 0.5)
    float decay_free_p  = 0.55f;      // (0.5, 1]
    // The drivers crop the floor (z < 0.11 m), so every beam aimed at it arrives EMPTY and nvblox leaves all
    // the space it crossed unobserved. floor_fill gives each empty pixel whose ray meets the floor within
    // range a virtual return floor_fill_below_m UNDER the floor: the ray clears everything above it and its
    // occupied band lands underground. A real return in the same pixel always wins (nvblox keeps the min).
    bool  floor_fill         = false;
    float floor_z            = 0.0f;
    float floor_fill_below_m = 0.15f;
    // Real returns near the floor, by height over the floor z and horizontal range hr from the sensor:
    //   z < floor_clear_z0 + floor_clear_slope·hr  → certainly floor: moved floor_fill_below_m under the floor along
    //                                                its ray (clears the space above it, marks nothing above ground)
    //   z < floor_push_z0  + floor_push_slope·hr   → floor OR a low obstacle (pose/tilt error grows with range):
    //                                                dropped — neither marks nor clears, so a low obstacle seen up
    //                                                close is not erased by its own far, ambiguous returns
    bool  floor_push         = false;
    float floor_clear_z0     = 0.05f;
    float floor_clear_slope  = 0.01f;
    float floor_push_z0      = 0.08f;
    float floor_push_slope   = 0.04f;
    // nvblox updates a voxel by nearest pixel only within this distance of the ray (voxels); 0.5 leaves the
    // space between helios rings (~2 voxels apart at 3 m) never updated wherever linear interpolation fails.
    float nearest_ray_dist_vox = 0.5f;
    // ...and along ELEVATION up to this fraction of the ring spacing at the voxel's range (needs the nvblox patch
    // nvblox_lidar_rings.patch): the space between two rings is observed by the nearest ring, while the
    // lateral (azimuth) reach stays nearest_ray_dist_vox. 0 = upstream nvblox.
    float ring_fraction = 0.0f;
    // 2D ESDF: occupied voxels between esdf_z0 and esdf_z1 collapse into one obstacle slice; distances up to
    // esdf_max_m are stored at esdf_z_out.
    float esdf_z0 = 0.11f, esdf_z1 = 1.80f, esdf_z_out = 0.5f, esdf_max_m = 2.0f;
    // every ~2 s: each sensor's cloud before processing and as integrated, to /dev/shm/nvblox_cloud_{pre,post}_<id>.bin
    // (12-float T_room_sensor, int n, n·xyz sensor frame) — read by tools/cloud_deleted.py
    bool dump_clouds = false;
};

class __attribute__((visibility("default"))) Mapper
{
public:
    explicit Mapper(const Params& p);
    ~Mapper();

    // T_room_sensor row-major 3x4 [R|t]; xyz = n points in the SENSOR frame. Returns the virtual floor returns added.
    // (robot_x, robot_y) = the robot axis in the room, centre of LidarModel's blind cylinders.
    int integrate_lidar(const float* T34, const float* xyz, int n, const LidarModel& m,
                        float robot_x = 0.0f, float robot_y = 0.0f);
    void decay();

    // Self-mask of sensor `id`: calibrate over its next sweeps (nothing is integrated meanwhile) and save to `file`,
    // or load a saved one (false if missing / size mismatch). Masked pixel count, -1 while calibrating / none.
    void self_mask_calibrate(int id, const SelfMaskParams& p, const std::string& file);
    bool self_mask_load(int id, const SelfMaskParams& p, const std::string& file);
    int  self_mask_pixels(int id) const;

    // Sampling lattice: cell centres of a w×h grid, n_bins heights (b+0.5)·bin_h from z=0.
    void set_sampling(float xmin, float ymin, float cell, int w, int h, int n_bins, float bin_h);
    // out[(iy·w+ix)·n_bins+b] = voxel log-odds there, 0 where nothing was ever observed.
    void sample(float* out);
    // Updates the 2D ESDF and samples it at the same cell centres: out[iy·w+ix] = signed distance to the nearest
    // obstacle (m, < 0 inside one), NaN where unobserved.
    void esdf(float* out);
    std::size_t allocated_blocks() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace nvb
