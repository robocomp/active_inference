/*
 * lidar_mount_stage — the LOCALISER-FREE LiDAR mount factors, run in this agent (plan
 * docs/superpowers/plans/2026-10-23-sensor-calibration-agent.md, Phase 3; moved out of room_concept's
 * LidarIngestor::mount_factor_step on 2026-10-29, same estimators, same priors):
 *
 *   FloorPlaneMount  bpearl (droll, dpitch, dz)    the floor seen head-on (absolute)
 *   VerticalMount    helios (droll, dpitch)        walls vertical, ceiling horizontal
 *                    bpearl (dx, dy, dyaw)         relative to the localiser's wall map
 *
 * The estimators are common/sensor_calib/mount_factors.h. This file only feeds them, and differs from the
 * room_concept copy in WHERE its inputs come from, not in what it does with them:
 *   - points in the robot (Shadow) frame through the CURRENT graph mount, from the media plane (sensor_inputs),
 *     not room_concept's ingest thread;
 *   - the pose at each sweep's stamp from the robot<->room RT ring (TimeQuery::Extrapolated), not a snapshot
 *     carried forward with the last cycle's velocity;
 *   - the wall map is the PUBLISHED room polygon, the ceiling hint the room node's room_height.
 * The per-sweep pose nuisance keeps room_concept's WIDE fixed prior (pose_sigma_*), never the localiser's own
 * marginal (r2.2 item 6): the published covariance is the solver's, and a tight one would let 10^3 sweeps turn
 * a pose error into a confident mount.
 *
 * ESTIMATE + LOG ONLY: no proposal is written (the publisher is wired in Phase 4), nothing moves a mount.
 * Pure: no Qt, no DSR, no DDS — the agent hands it plain data, so tests/mount_factors_test.cpp can drive it.
 */
#pragma once
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "../../common/sensor_calib/mount_factors.h"

namespace sc
{
// One LiDAR sweep as the stage needs it: points and the sensor origin in the ROBOT frame, plus the robot pose
// in the room at the sweep's own stamp (nullopt = no pose evidence for that instant: floor rows only).
struct MountSweep
{
    std::string                    sensor;          // "helios" | "bpearl"
    std::int64_t                   stamp_ms = 0;
    std::vector<Eigen::Vector3f>   points_robot;
    Eigen::Vector3f                origin_robot = Eigen::Vector3f::Zero();
    std::optional<Eigen::Affine3d> room_T_robot;
};

struct LidarMountStageConfig
{
    double rate_hz = 2.0;               // sweeps per second PER LiDAR that are fed: a COMPUTE budget
    double pose_sigma_xy = 0.05;        // m, WIDE per-sweep pose nuisance prior of the vertical factor
    double pose_sigma_yaw_deg = 2.0;    // deg, idem
    std::size_t max_points = 8000;      // computational stride per sweep (the per-sweep nuisance, not the count, bounds a sweep)
    std::string log_dir = "tmp/mount_factors";   // per-sweep bpearl planar + periodic summary CSVs; "" = no files
};

// The current posteriors, physical units (m, rad). Info6 order (dx, dy, dz, droll, dpitch, dyaw).
struct MountFactorsState
{
    rc::mountf::Info3 floor;                    // bpearl (droll, dpitch, dz)
    rc::mountf::Info6 vert_helios, vert_bpearl; // helios uses (droll, dpitch); bpearl (dx, dy, dyaw)
    Eigen::Vector3f   s_helios = Eigen::Vector3f::Zero(), s_bpearl = Eigen::Vector3f::Zero();   // robot frame
    double floor_share = 0.0, floor_sigma = 0.0, wall_sigma_h = 0.0, wall_sigma_b = 0.0;
    long   sweeps_helios = 0, sweeps_bpearl_floor = 0, sweeps_bpearl_walls = 0;
};

class LidarMountStage
{
public:
    explicit LidarMountStage(LidarMountStageConfig cfg);

    // Feed one sweep. `polygon_room` is the published room polygon (room frame); `ceiling_hint_m` the room height
    // (<= 0 = no ceiling rows). Sweeps closer than 1/rate_hz to the last one fed for that sensor are skipped.
    // Returns true when the sweep was fed.
    bool observe(const MountSweep& sweep, const std::vector<Eigen::Vector2f>& polygon_room, float ceiling_hint_m);

    [[nodiscard]] MountFactorsState state() const;
    // One human-readable line of the current estimates (mm / deg with 1-sigma).
    [[nodiscard]] std::string summary() const;
    // Append the current estimates to <log_dir>/mounts_<ts>.csv (no-op without a log_dir).
    void log_state(std::int64_t stamp_ms);

private:
    LidarMountStageConfig cfg_;
    rc::mountf::FloorPlaneMount floor_bp_;
    rc::mountf::VerticalMount   vert_h_, vert_bp_;
    Eigen::Vector3f s_helios_ = Eigen::Vector3f::Zero(), s_bpearl_ = Eigen::Vector3f::Zero();
    std::int64_t last_helios_ms_ = 0, last_bpearl_ms_ = 0;
    long n_floor_ = 0, n_walls_b_ = 0;
    std::ofstream sweep_csv_, state_csv_;

    [[nodiscard]] std::vector<Eigen::Vector3f> strided(const std::vector<Eigen::Vector3f>& in) const;
    void open_csv(std::ofstream& f, const std::string& stem, const char* header);
};

// The room polygon in the ROBOT frame at pose room_T_robot, as the vertical factor's walls (one per edge), and
// the place key of that pose (0.5 m cell x heading octant, rc::mountf::place_key).
[[nodiscard]] std::vector<rc::mountf::VerticalMount::Wall> walls_in_robot(const std::vector<Eigen::Vector2f>& polygon_room,
                                                                          const Eigen::Affine3d& room_T_robot);
[[nodiscard]] std::int64_t place_of(const Eigen::Affine3d& room_T_robot);
}  // namespace sc
