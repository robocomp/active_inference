// band_cross.h — [Route] BandDrivesCross (Task 6 fix round 1, coordinator ruling (b), 2026-10-21): the route the OLD chain's
// spline/band drives THROUGH a measured doorway, instead of the anchor servo. Pure (Eigen + std).
//
// ★ WHY. The user's hypothesis is about the SPLINE in narrow passages. In the old chain a cross with an interaction anchor is
//   driven by the anchor servo (rc::anchor_cross_command) and the band never runs there, so the near-field clearance
//   (near_field_clearance.h) would never act IN a doorway. Behind BandDrivesCross the route continues through the opening
//   and the follower + band drive it; only the DRIVER changes — the crossing's completion stays the cross protocol's
//   (ControllerSession::cross_protocol: `through`, the near-side gate, the timeouts).
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include "local_policy/doorway_corridor.h"

namespace rc
{
// Half the wall's thickness at a doorway: an interior wall of 0.20 m. ⚑ STATED, not measured — neither the interaction anchor
// nor the measured gap carries the wall's depth. It places the jamb's FACE along the normal (near_field_clearance.h
// jamb_elements) and the route's lead-in point (the body's front at the near face).
inline constexpr float kBandCrossWallHalfDepthM = 0.10f;

// The polyline through the opening g (room frame) from the robot: the point where the body, travelling along the opening's
// normal, has its front at the near face of the wall (lead_in_m = the body's front support + the wall's half-depth, before
// the line); the opening's centre; and the cross's goal past_m beyond the line. A point the robot is already past (along
// the normal) is not a waypoint — a route never asks the robot to go back to it.
inline std::vector<Eigen::Vector2f> band_cross_polyline(const Eigen::Vector2f &robot, const policy::RoomGap &g, float lead_in_m,
                                                        float past_m)
{
    const Eigen::Vector2f n = g.normal.normalized();
    const float s_robot = (robot - g.centre).dot(n);
    std::vector<Eigen::Vector2f> poly{robot};
    if (s_robot < -lead_in_m) poly.push_back(g.centre - lead_in_m * n);
    if (s_robot < 0.f) poly.push_back(g.centre);
    poly.push_back(g.centre + past_m * n);
    return poly;
}

// Where a route (a polyline / sampled curve, in order) first crosses the opening's line (s = 0), its lateral offset e from the
// opening's centre; NaN when it never crosses.
inline float band_cross_offset(std::span<const Eigen::Vector2f> route, const policy::RoomGap &g)
{
    const Eigen::Vector2f n = g.normal.normalized(), lat(-n.y(), n.x());
    for (std::size_t i = 0; i + 1 < route.size(); ++i)
    {
        const float s0 = (route[i] - g.centre).dot(n), s1 = (route[i + 1] - g.centre).dot(n);
        if (s0 <= 0.f and s1 > 0.f)
        {
            const float u = s0 == s1 ? 0.f : -s0 / (s1 - s0);
            return (route[i] + u * (route[i + 1] - route[i]) - g.centre).dot(lat);
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}

// Does the installed route still take the body THROUGH the opening as now measured? Its crossing must lie where the body's
// centre can pass — |e| <= W/2 - half_width (the body's lateral half-width) — the corridor in_doorway_corridor admits. A
// physical feasibility test on the measurement, not a tuned tolerance: false ⇒ the route is rebuilt on the current opening.
inline bool band_cross_route_passes(std::span<const Eigen::Vector2f> route, const policy::RoomGap &g, float half_width_m)
{
    const float e = band_cross_offset(route, g);
    return std::isfinite(e) and std::abs(e) <= 0.5f * g.width - half_width_m;
}
}  // namespace rc
