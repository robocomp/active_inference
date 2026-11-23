/*
 * residual_config.cpp — fill ResidualConfig from a RoboComp ConfigLoader.
 */

#include "residual_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <charconv>
#include <print>
#include <ranges>

#include <genericworker.h>   // ConfigLoader

namespace rc
{

ResidualConfig load_residual_config(const ConfigLoader& cfg)
{
    ResidualConfig out;

    // Every read registers its key, its code default and a one-line description for the startup table.
    rc::cfg::Reader reader(cfg, "residual_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what) { return reader.f(k, def, what); };
    const auto geti = [&](std::string_view k, int def, std::string_view what) { return reader.i(k, def, what); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what) { return reader.s(k, std::move(def), what); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what) { return reader.b(k, def, what); };

    out.grid_publish_every_n = geti("Publish.EveryN", 1, "compute cycles per publish of the `residual` node");
    out.grid_field_ema_up    = getf("Publish.FieldEmaUp", 1.0f, "published risk rises at this rate (1 = instant)");
    out.grid_field_ema_down  = getf("Publish.FieldEmaDown", 0.30f, "published risk falls at this rate (slow = stable)");

    out.floor_z0       = getf("Readout.FloorZ0M", 0.06f, "floor band (m, room z): cells below it are floor");
    out.ceil_z         = getf("Readout.CeilZM", 1.80f, "cells above this are ceiling / overhead");
    out.robot_radius_m = getf("Readout.RobotRadiusM", 0.32f, "cells within this of the robot axis are the robot");
    out.wall_margin_m  = getf("Readout.WallMarginM", 0.30f, "cells within this of the room polygon (or outside) are wall");
    out.object_sensor_sigma_m = getf("Readout.ObjectSensorSigmaM", 0.03f, "LiDAR range noise added to each object's σ");
    out.object_fit_margin_m   = getf("Readout.ObjectFitMarginM", 0.10f, "objects claim this far past their box edges");

    out.grid_bin_span_m        = getf("Grid.BinSpanM", 2.0f, "height covered by the 64 bins sampled per cell");
    out.grid_lidar_clearance_m = getf("Grid.LidarClearanceM", 0.55f, "no latched cell inside this radius is released");
    out.grid_speckle_min_neighbours = geti("Grid.SpeckleMinNeighbours", 1, "lone-cell filter; 0 = off");
    out.grid_speckle_grace_cycles   = geti("Grid.SpeckleGraceCycles", 10, "cycles a lone cell ships after its last return");
    out.grid_speckle_min_component_cells = geti("Grid.SpeckleMinComponentCells", 3, "isolated-clump filter; 0 = off");

    out.use_helios         = getb("Lidar.UseHelios", true, "integrate the helios plane");
    out.use_bpearl         = getb("Lidar.UseBpearl", true, "integrate the bpearl plane");
    out.helios_min_range_m = getf("Lidar.HeliosMinRangeM", 0.40f, "RS-Helios datasheet minimum range");
    out.bpearl_min_range_m = getf("Lidar.BpearlMinRangeM", 0.10f, "RS-Bpearl minimum range");

    out.nvblox_voxel_m       = getf("Nvblox.VoxelM", 0.05f, "nvblox voxel edge (m)");
    out.nvblox_p_free        = getf("Nvblox.PFree", 0.30f, "inverse sensor model: free");
    out.nvblox_p_occ         = getf("Nvblox.POcc", 0.70f, "inverse sensor model: occupied");
    out.nvblox_half_width_m  = getf("Nvblox.OccHalfWidthM", 0.05f, "occupied band half-width around a return (m)");
    out.nvblox_max_range_m   = getf("Nvblox.MaxRangeM", 10.0f, "LiDAR integration distance (m)");
    out.nvblox_decay         = getb("Nvblox.Decay", false, "decay every voxel toward P=0.5 each cycle");
    out.nvblox_decay_occ_p   = getf("Nvblox.DecayOccP", 0.49f, "per-cycle decay of occupied voxels, [0,0.5)");
    out.nvblox_decay_free_p  = getf("Nvblox.DecayFreeP", 0.51f, "per-cycle decay of free voxels, (0.5,1]");
    out.nvblox_k_obs         = getf("Nvblox.KObs", 20.0f, "Beta evidence weight of an observed cell");
    out.nvblox_floor_fill    = getb("Nvblox.FloorFill", true, "virtual sub-floor return for empty beams that meet the floor");
    out.nvblox_floor_push    = getb("Nvblox.FloorPush", true, "push real floor-band returns under the floor (clear, never mark)");
    out.nvblox_floor_clear_z0 = getf("Nvblox.FloorClearZ0M", 0.05f, "certain-floor band at the sensor (m): clears");
    out.nvblox_floor_clear_slope = getf("Nvblox.FloorClearSlope", 0.01f, "certain-floor band growth per metre");
    out.nvblox_floor_push_z0 = getf("Nvblox.FloorPushZ0M", 0.08f, "ambiguous floor band at the sensor (m): ignored");
    out.nvblox_floor_push_slope = getf("Nvblox.FloorPushSlope", 0.04f, "ambiguous floor band growth per metre");
    out.nvblox_floor_fill_below_m = getf("Nvblox.FloorFillBelowM", 0.15f, "depth of that virtual return under the floor (m)");
    out.nvblox_nearest_ray_dist_vox = getf("Nvblox.NearestRayDistVox", 1.5f, "nearest-pixel update reach off the ray (voxels)");
    out.nvblox_ring_fraction = getf("Nvblox.RingFraction", 0.6f, "nearest-pixel reach along elevation, fraction of the ring gap");
    out.nvblox_read_min_z    = getf("Nvblox.ReadMinZM", 0.11f, "lowest height read into the grid column (m, room)");
    out.nvblox_occ_clear_lo  = getf("Nvblox.OccClearLogOdds", -0.4f, "release a latched cell when its column max falls below this");
    out.nvblox_release_cycles = geti("Nvblox.ReleaseCycles", 10, "cycles the column must stay below OccClearLogOdds before release");
    out.nvblox_display       = getb("Nvblox.Display", true, "live window: nvblox P / height / ESDF / residual");
    out.nvblox_display_every_n = geti("Nvblox.DisplayEveryN", 5, "refresh the window (and the 3D dump) every N cycles");
    out.nvblox_display_scale = geti("Nvblox.DisplayScale", 2, "screen pixels per grid cell");
    out.nvblox_dump_3d       = getb("Nvblox.Dump3D", false, "write /dev/shm/residual_nvblox_3d.bin (tools/nvblox_rerun_view.py)");
    out.nvblox_dump_clouds   = getb("Nvblox.DumpClouds", false, "write /dev/shm/nvblox_cloud_{pre,post}_<id>.bin (tools/cloud_deleted.py)");
    out.nvblox_esdf_z0       = getf("Nvblox.EsdfZMinM", 0.11f, "ESDF slice: lowest obstacle height (m)");
    out.nvblox_esdf_z1       = getf("Nvblox.EsdfZMaxM", 1.80f, "ESDF slice: highest obstacle height (m)");
    out.nvblox_esdf_max_m    = getf("Nvblox.EsdfMaxDistM", 2.0f, "ESDF saturation distance (m)");
    out.nvblox_bpearl_blind_r  = getf("Nvblox.BpearlBlindRadiusM", 0.55f, "bpearl driver [Footprint].radius (m); 0 = no guard");
    out.nvblox_bpearl_blind_z0 = getf("Nvblox.BpearlBlindZMinM", 0.11f, "bpearl driver [Footprint].z_min (m)");
    out.nvblox_bpearl_blind_z1 = getf("Nvblox.BpearlBlindZMaxM", 1.35f, "bpearl driver [Footprint].z_max (m)");
    out.nvblox_bpearl_skirt_r  = getf("Nvblox.BpearlSkirtRadiusM", 0.75f, "bpearl driver [Skirt].radius (m); 0 = none");
    out.nvblox_bpearl_skirt_z0 = getf("Nvblox.BpearlSkirtZMinM", 0.11f, "bpearl driver [Skirt].z_min (m)");
    out.nvblox_bpearl_skirt_z1 = getf("Nvblox.BpearlSkirtZMaxM", 0.35f, "bpearl driver [Skirt].z_max (m)");
    out.nvblox_helios_az     = geti("Nvblox.HeliosAzimuthDivisions", 900, "helios returns per ring");
    out.nvblox_helios_rows   = geti("Nvblox.HeliosRows", 32, "helios range-image rows (uniform model)");
    out.nvblox_helios_below_deg = getf("Nvblox.HeliosBelowDeg", 55.0f, "helios lowest beam below horizontal");
    out.nvblox_helios_above_deg = getf("Nvblox.HeliosAboveDeg", 15.0f, "helios highest beam above horizontal");
    // from_chars, not stof: these machines run LANG=es_ES, where strtod stops at the '.'
    for (const std::string s = gets("Nvblox.HeliosRingElevDeg", "", "helios per-ring elevations (deg, ';'-separated)");
         const auto tok : std::views::split(s, ';'))
    {
        std::string_view t(tok.begin(), tok.end());
        while (not t.empty() and t.front() == ' ') t.remove_prefix(1);
        float v;
        if (std::from_chars(t.data(), t.data() + t.size(), v).ec == std::errc{}) out.nvblox_helios_ring_elev_deg.push_back(v);
    }
    out.nvblox_bpearl_az     = geti("Nvblox.BpearlAzimuthDivisions", 900, "bpearl returns per ring");
    out.nvblox_bpearl_rows   = geti("Nvblox.BpearlRows", 32, "bpearl range-image rows");
    out.nvblox_bpearl_below_deg = getf("Nvblox.BpearlBelowDeg", 2.0f, "bpearl lowest beam below its plane");
    out.nvblox_bpearl_above_deg = getf("Nvblox.BpearlAboveDeg", 90.0f, "bpearl highest beam above its plane");

    out.self_mask_enabled    = getb("SelfMask.Enabled", true, "drop each sensor's saved self-mask pixels (body + its edge)");
    out.self_mask_calibrate  = getb("SelfMask.Calibrate", false, "learn the self-masks at startup (>= ClearRadiusM free all around)");
    out.self_mask_sweeps     = geti("SelfMask.Sweeps", 100, "sweeps per sensor to learn its mask (nothing integrated meanwhile)");
    out.self_mask_clear_r    = getf("SelfMask.ClearRadiusM", 1.0f, "free floor radius around the robot during calibration");
    out.self_mask_floor_tol  = getf("SelfMask.FloorTolM", 0.05f, "a return above floor_z + this inside the radius is the robot");
    out.self_mask_min_frac   = getf("SelfMask.MinFraction", 0.3f, "share of sweeps a pixel must return the robot to be masked");
    out.self_mask_dilate_az  = geti("SelfMask.DilateAz", 2, "mask dilation along azimuth (columns)");
    out.self_mask_dilate_ring = geti("SelfMask.DilateRing", 1, "mask dilation across rings (rows)");
    out.self_mask_prefix     = gets("SelfMask.FilePrefix", "etc/self_mask_", "mask file = prefix + sensor name + .bin");
    out.self_mask_sensors    = gets("SelfMask.Sensors", "bpearl", "sensors that get a self-mask (helios, bpearl)");

    out.csv_logging       = getb("Diag.CsvLogging", false, "write the diagnostic CSVs under etc/ (grid/floor/perf diag, cells, births, releases)");
    out.cell_dump_every_n = geti("Diag.CellDumpEveryN", 600, "residual_cells / residual_occ_debug CSV every N cycles; 0 = off");
    out.birth_csv_path    = gets("Diag.BirthCsvPath", "etc/residual_births.csv", "per-latched-cell trace; empty = off");
    out.release_csv_path  = gets("Diag.ReleaseCsvPath", "etc/residual_releases.csv", "per-released-cell trace; empty = off");

    std::print("residual_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
