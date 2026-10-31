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
 * ★ Task 8 — THE HELIOS'S FAR FLOOR IS CUT FROM THE POLICY SET (the Task-7 review's required item). The helios hangs
 *   INVERTED at z 1.075 m; its fan spans +54.5 .. -15.5 deg (ROBOT_GEOMETRY.md), so its LOWEST beam meets the floor
 *   1.075 / tan(15.5 deg) = 3.88 m out, and those grazing floor returns read 0.13-0.17 m high (the bias grows with range:
 *   residual_concept fits its floor from the bpearl only for this reason). They fall inside [ContactMinHeightM, 0.20).
 *   PHYSICAL RULE: the same beam reaches the 0.20 m band floor only (1.075 - 0.20) / tan(15.5 deg) = 3.16 m out, so ANY
 *   helios return below 0.20 m is at >= 3.16 m and at a <= 15.5 deg grazing angle, where the floor itself returns at
 *   0.13-0.17 m — the helios cannot tell a low obstacle from the floor there. The policy's low band is therefore the
 *   BPEARL's (the downward dome, the floor-datum sensor); the helios contributes from the legacy floor (0.20 m, the level
 *   that was set above its floor ring in the first place: TemporaryObstacleMinHeight) up. No range constant: the cut is
 *   the existing legacy level applied by SOURCE. A return whose plane is unknown (no plane ids) keeps the plain bands.
 *
 * Pure: no Qt, no graph.
 */

#include <algorithm>
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

// Which sensor a return came from (the media plane id the obstacle tracker receives per point: helios 0, bpearl 1 —
// ROBOT_GEOMETRY.md "Media plane id").
enum class LidarSource { Unknown, Helios, Bpearl };
inline LidarSource lidar_source_of_plane(int plane_id)
{
    return plane_id == 0 ? LidarSource::Helios : plane_id == 1 ? LidarSource::Bpearl : LidarSource::Unknown;
}
// The geometry behind the helios cut: DOCUMENTATION AND TEST ONLY, never read at runtime. The rule itself uses
// legacy_min_m, and the points it is applied to arrive in the robot frame through the GRAPH's body<-helios edge
// (LidarPlaneReader::poll), so a mount corrected or planted through robot_concept's overlay already reaches it.
// This is the NOMINAL height (shadow.json body<-helios z), used to explain why 0.20 m is the right cut.
inline constexpr float kHeliosMountZ = 1.075f;                       // nominal, shadow.json body<-helios z
inline constexpr float kHeliosLowestBeamRad = 15.5f * 3.14159265f / 180.f;   // the inverted fan's lowest beam below horizontal

inline LidarBandMembership lidar_band_membership(float z, const LidarHeightBands &b, LidarSource src = LidarSource::Unknown)
{
    if (not std::isfinite(z) or z > b.max_m) return {};
    const float policy_min = src == LidarSource::Helios ? std::max(b.contact_min_m, b.legacy_min_m) : b.contact_min_m;
    return {.policy = z >= policy_min, .legacy = z >= b.legacy_min_m};
}

}  // namespace rc
