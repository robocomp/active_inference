/*
 * residual_nvblox_map.h  —  nvblox as the occupancy source of the residual grid.
 *
 * Owns the nvblox mapper (behind libnvblox_bridge.so), integrates each LiDAR plane once per new sweep in its own
 * sensor frame, and samples the voxel columns at the grid's Z_BINS bin centres, which OccupancyGrid::ingest_columns
 * turns into the latch / belief the rest of the agent reads. Also drives the live window and the 3D dump.
 * DSR-free: the caller supplies room←sensor transforms.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../nvblox_bridge/nvblox_bridge.h"
#include "residual_config.h"
#include "residual_lidar_ingestor.h"
#include "residual_nvblox_view.h"
#include "residual_occupancy_grid.h"

namespace rc
{

class NvbloxMap
{
public:
    // room←sensor at a capture stamp (ms), nullopt if the graph cannot resolve it.
    using SensorPose = std::function<std::optional<Eigen::Matrix4d>(const std::string& sensor, std::int64_t stamp_ms)>;

    // Built once the grid exists: sampling follows the grid's cells and bins. Loads or calibrates the self-masks.
    NvbloxMap(const ResidualConfig& cfg, const OccupancyGrid& grid);

    // Integrate every plane whose sweep is newer than the last one integrated, decay, and resample the columns.
    void integrate(const ResidualLidarIngestor& lidar, const SensorPose& room_T_sensor, long occupied_cells);
    // Per-cell voxel log-odds (w·h·Z_BINS, 0 = never observed), the input of OccupancyGrid::ingest_columns.
    const std::vector<float>& columns() const { return cols_; }
    // [map-time] mean cost of the map update over 100 cycles; ingest_ms = the grid's ingest_columns time.
    void log_map_time(double total_ms, double ingest_ms, long occupied_cells);
    // Live window and/or /dev/shm 3D snapshot (Nvblox.Display / Nvblox.Dump3D), every Nvblox.DisplayEveryN calls.
    void show(const OccupancyGrid& grid, const OccupancyGrid::CellExplained& explained, const Eigen::Vector3f& robot);

private:
    void dump_3d(const NvbloxView::Frame& f) const;

    const ResidualConfig&       cfg_;
    std::unique_ptr<nvb::Mapper> mapper_;
    std::unique_ptr<NvbloxView> view_;
    nvb::LidarModel             helios_, bpearl_;
    std::vector<std::string>    names_;          // plane order of ResidualLidarIngestor (plane id = index)
    std::vector<float>          cols_, esdf_, xyz_;
    std::vector<std::int64_t>   last_stamp_;
    std::vector<long>           sweeps_;         // sweeps integrated per plane (log)
    int                         filled_ = 0;     // virtual floor returns in the last sweep (log)
    float                       ms_integrate_ = 0.0f, ms_sample_ = 0.0f;
    double                      sum_total_ = 0, sum_ingest_ = 0, sum_integrate_ = 0, sum_sample_ = 0;
    int                         n_time_ = 0, n_log_ = 0, n_show_ = 0;
};

}  // namespace rc
