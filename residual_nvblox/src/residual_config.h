/*
 * residual_config.h — plain-data configuration of residual_nvblox + its loader (RoboComp ConfigLoader).
 */

#pragma once

#include <string>
#include <vector>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc
{

struct ResidualConfig
{
    // ── [Publish] ──
    // Compute cycles per publish of the `residual` node (Period.Compute = 100 ms, so 1 = 10 Hz). The controller
    // plans against it. ⚠ It is the largest attribute in the graph and cortex keeps a CRDT dot per write: if an
    // agent starts burning CPU in compact() with no obvious cause, raise this first.
    int   grid_publish_every_n = 1;
    // Asymmetric low-pass on the PUBLISHED belief field only (never the latch): risk rises at ema_up (1 = instant,
    // never lag a new obstacle) and falls at ema_down (slow, so a flickering edge cell holds its risk).
    float grid_field_ema_up   = 1.0f;
    float grid_field_ema_down = 0.30f;

    // ── [Readout] what a known model already explains ──
    float floor_z0       = 0.06f;   // floor band (m, room z); nvblox already judged the floor per sweep (FloorPush)
    float ceil_z         = 1.80f;   // overhead structure above the navigation band
    float robot_radius_m = 0.32f;   // the robot's own body around its axis
    float wall_margin_m  = 0.30f;   // within this of the room polygon boundary (or outside it) = wall
    float object_sensor_sigma_m = 0.03f;   // LiDAR range noise: an object's explained boundary is fuzzy by ≥ this
    float object_fit_margin_m   = 0.10f;   // claim this far past each object box edge (under-fit slack)

    // ── [Grid] room-frame 2-D cells read out of nvblox's voxel columns ──
    float grid_bin_span_m        = 2.0f;    // height covered by the 64 bins sampled per cell (3.125 cm bins)
    float grid_lidar_clearance_m = 0.55f;   // no latched cell inside this radius of the robot is released
    // SPECKLE. ★A SIZE THRESHOLD, flagged: a lone residual cell (or a clump under min_component_cells) is not
    // shipped unless a return struck it within grace_cycles — so a real thin obstacle survives while it is seen.
    int   grid_speckle_min_neighbours      = 1;    // 0 ⇒ off
    int   grid_speckle_grace_cycles        = 10;
    int   grid_speckle_min_component_cells = 3;    // 0 ⇒ off

    // ── [Lidar] ──
    bool  use_helios = true;
    bool  use_bpearl = true;
    float helios_min_range_m = 0.40f;   // dead shells: no return closer than this
    float bpearl_min_range_m = 0.10f;

    // ── [Nvblox] (see nvb::Params) ──
    float nvblox_voxel_m       = 0.05f;
    float nvblox_p_free        = 0.30f;
    float nvblox_p_occ         = 0.70f;
    float nvblox_half_width_m  = 0.05f;
    float nvblox_max_range_m   = 10.0f;
    bool  nvblox_decay         = false;
    float nvblox_decay_occ_p   = 0.49f;
    float nvblox_decay_free_p  = 0.51f;
    float nvblox_k_obs         = 20.0f;
    bool  nvblox_floor_fill    = true;
    bool  nvblox_floor_push    = true;
    float nvblox_floor_push_z0 = 0.08f, nvblox_floor_push_slope = 0.04f;
    float nvblox_floor_clear_z0 = 0.05f, nvblox_floor_clear_slope = 0.01f;
    float nvblox_floor_fill_below_m = 0.15f;
    float nvblox_nearest_ray_dist_vox = 1.5f;
    float nvblox_ring_fraction = 0.6f;
    float nvblox_read_min_z    = 0.11f;
    float nvblox_occ_clear_lo  = -0.4f;
    int   nvblox_release_cycles = 10;
    bool  nvblox_display       = true;
    int   nvblox_display_every_n = 5;
    int   nvblox_display_scale = 2;
    bool  nvblox_dump_3d       = false;   // /dev/shm snapshot for tools/nvblox_rerun_view.py
    bool  nvblox_dump_clouds   = false;   // /dev/shm pre/post clouds for tools/cloud_deleted.py
    float nvblox_esdf_z0 = 0.11f, nvblox_esdf_z1 = 1.80f, nvblox_esdf_max_m = 2.0f;
    float nvblox_bpearl_blind_r = 0.55f, nvblox_bpearl_blind_z0 = 0.11f, nvblox_bpearl_blind_z1 = 1.35f;
    float nvblox_bpearl_skirt_r = 0.75f, nvblox_bpearl_skirt_z0 = 0.11f, nvblox_bpearl_skirt_z1 = 0.35f;
    int   nvblox_helios_az     = 900,  nvblox_helios_rows = 32;
    float nvblox_helios_below_deg = 55.0f, nvblox_helios_above_deg = 15.0f;
    std::vector<float> nvblox_helios_ring_elev_deg;   // real per-ring elevations; empty = uniform rows
    int   nvblox_bpearl_az     = 900,  nvblox_bpearl_rows = 32;
    float nvblox_bpearl_below_deg = 2.0f,  nvblox_bpearl_above_deg = 90.0f;

    // ── [SelfMask] per-sensor mask of the pixels that see the robot's own body (see nvb::SelfMaskParams) ──
    bool  self_mask_enabled = true, self_mask_calibrate = false;
    int   self_mask_sweeps = 100, self_mask_dilate_az = 2, self_mask_dilate_ring = 1;
    float self_mask_clear_r = 1.0f, self_mask_floor_tol = 0.05f, self_mask_min_frac = 0.3f;
    std::string self_mask_prefix = "etc/self_mask_";
    std::string self_mask_sensors = "bpearl";

    // ── [Diag] (empty path / 0 = off) ──
    bool        csv_logging = false;       // master switch for every CSV this agent writes under etc/
    int         cell_dump_every_n = 600;   // etc/residual_cells.csv + etc/residual_occ_debug.csv every N cycles
    std::string birth_csv_path;            // one row per cell the moment it latches
    std::string release_csv_path;          // one row per cell the moment it is released
};

ResidualConfig load_residual_config(const ConfigLoader& cfg);

}  // namespace rc
