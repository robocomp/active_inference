#pragma once
/*
 * lidar_height_band.h — which height band a LiDAR return belongs to (Task 7, local-motion policy, 2026-10-20).
 *
 * Two sets are cut from the same capture, in the robot frame (z = height above the floor: the source cloud is
 * floor-referenced and robot←lidar is identity, see handle_lidar_points):
 *
 *   POLICY  [Lidar.ContactMinHeightM, TemporaryObstacleMaxHeight]  everything the body can touch. Its floor is the
 *           MEASURED floor-return spread plus the range noise (controller/tools/floor_band.py: p99.9 of the floor
 *           returns of a parked robot on open floor + sigma_r), so it starts just above the floor — a 0.10 m sill,
 *           kerb or foot-ring is contact for the policy.
 *   LEGACY  [TemporaryObstacleMinHeight (0.20), TemporaryObstacleMaxHeight]  the old consumers' band, UNCHANGED
 *           until Task 10 deletes them (switching it here would change every old path at once, unreviewed).
 *
 * ⚑ DECISION LEVEL, flagged: a return either is or is not in a set. The level is not tuned; it is the floor the
 *   sensor actually reports (measured), plus one sigma_r — a return the floor could have produced is not contact.
 *
 * Pure: no Qt, no graph.
 */

#include <cmath>

#include "controller_runtime_types.h"

namespace rc
{

struct LidarHeightBands
{
    float contact_min_m = 0.f;   // the policy's floor cut
    float legacy_min_m = 0.f;    // the old consumers' floor cut (0.20)
    float max_m = 0.f;           // shared ceiling
};

struct LidarBandMembership
{
    bool policy = false;
    bool legacy = false;
};

inline LidarHeightBands lidar_height_bands(const ControllerParams &p)
{
    return {.contact_min_m = p.lidar_contact_min_height_m,
            .legacy_min_m = p.temporary_obstacle_min_height_m,
            .max_m = p.temporary_obstacle_max_height_m};
}

inline LidarBandMembership lidar_band_membership(float z, const LidarHeightBands &b)
{
    if (not std::isfinite(z) or z > b.max_m) return {};
    return {.policy = z >= b.contact_min_m, .legacy = z >= b.legacy_min_m};
}

}  // namespace rc
