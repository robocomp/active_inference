/*
 * beam_room.h — the known room surfaces in the beam model's foreign density (spec 2026-11-15 lidar-measures §7c.8,
 * rev 3.3.7 "Known room surfaces in the foreign density"). SHARED, header-only, pure.
 *
 * Per staged ray, t_r is its first crossing in (0, ρ_max] with the room's known surfaces: the floor plane z = floor_z of the
 * room frame and the walls of the room polygon (vertical, unbounded in z). The foreign density becomes
 *
 *     p_f(ρ) = p_pass·𝟙[ρ < t_r] + (1 − p_pass·t_r)·St(ρ − t_r)          (no crossing before ρ_max: p_f = p_pass)
 *
 * — unmodelled clutter stays uniform at 1/ρ_max in front of the room surface, and the mass that went beyond it lands on it,
 * smeared by the same Student-t. No new parameter. p_f,i does not depend on θ (the surfaces are fixed during an update), so it
 * is staged once per ray into BeamRays::p_f and the beam terms read it as a constant.
 */
#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "beam_ray.h"

namespace rc::ai::beam
{

struct RoomSurfaces
{
    bool   floor   = true;                                     // the floor plane z = floor_z is a known surface
    double floor_z = 0.0;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> walls;   // room-polygon edges (a, b), vertical
};

// t_r: the first crossing of o + t·u, t ∈ (0, ρ_max], with the room surfaces; +inf when there is none.
inline double first_room_crossing(const Eigen::Vector3d& o, const Eigen::Vector3d& u, double rho_max, const RoomSurfaces& room)
{
    double t_r = std::numeric_limits<double>::infinity();
    if (room.floor and u.z() < 0.0)
    {
        const double t = (room.floor_z - o.z()) / u.z();
        if (t > 0.0 and t <= rho_max) t_r = std::min(t_r, t);
    }
    for (const auto& [a, b] : room.walls)
    {
        const Eigen::Vector2d e = b - a, d = u.head<2>();
        const double den = d.x() * e.y() - d.y() * e.x();
        if (den == 0.0) continue;                              // parallel: never crossed
        const Eigen::Vector2d q = a - o.head<2>();
        const double t = (q.x() * e.y() - q.y() * e.x()) / den;   // along the ray
        const double sg = (q.x() * d.y() - q.y() * d.x()) / den;  // along the segment, [0, 1]
        if (t > 0.0 and t <= rho_max and sg >= 0.0 and sg <= 1.0) t_r = std::min(t_r, t);
    }
    return t_r;
}

// p_f(ρ) for a ray whose room crossing is t_r (spec rev 3.3.7).
inline double room_foreign_density(double rho, double t_r, double rho_max, const StudentT& st)
{
    const double p_pass = 1.0 / rho_max;
    if (not std::isfinite(t_r)) return p_pass;
    return (rho < t_r ? p_pass : 0.0) + (1.0 - p_pass * t_r) * st.pdf(rho - t_r);
}

// Stage BeamRays::p_f for every ray of `rays` against `room` (call after the endpoints are staged and ρ clamped to ρ_max).
inline void stage_room_foreign_density(BeamRays& rays, const RoomSurfaces& room, const StudentT& st)
{
    rays.p_f.resize(rays.endpoints.size());
    for (std::size_t i = 0; i < rays.endpoints.size(); ++i)
    {
        const Eigen::Vector3d du = rays.endpoints[i] - rays.origin;
        const double rho = du.norm();
        rays.p_f[i] = rho > 0.0 ? room_foreign_density(rho, first_room_crossing(rays.origin, du / rho, rays.rho_max, room),
                                                       rays.rho_max, st)
                                : 1.0 / rays.rho_max;
    }
}

}  // namespace rc::ai::beam
