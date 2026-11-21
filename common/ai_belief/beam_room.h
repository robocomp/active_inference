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
 * smeared by the same Student-t, its scale widened by the surface's own σ projected on the ray (rev 3.3.8). p_f,i does not depend on θ (the surfaces are fixed during an update), so it
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

struct RoomWall
{
    Eigen::Vector2d a = Eigen::Vector2d::Zero(), b = Eigen::Vector2d::Zero();   // a room-polygon edge, vertical
    double sigma = 0.0;                                                          // its published position σ (m)
};
struct RoomSurfaces
{
    bool   floor       = true;    // the floor plane z = floor_z is a known surface
    double floor_z     = 0.0;
    double floor_sigma = 0.0;     // Model.FloorSigmaM (spec 7c.8 rev 3.3.8)
    std::vector<RoomWall> walls;
};

// The first crossing of o + t·u, t ∈ (0, ρ_max], with the room surfaces (t = +inf: none), and that surface's σ projected
// along the ray, σ_r = σ_surface/|n̂·u| (spec 7c.8 rev 3.3.8).
struct RoomCrossing { double t = std::numeric_limits<double>::infinity(), sigma_r = 0.0; };
inline RoomCrossing first_room_crossing(const Eigen::Vector3d& o, const Eigen::Vector3d& u, double rho_max, const RoomSurfaces& room)
{
    RoomCrossing c;
    if (room.floor and u.z() < 0.0)
    {
        const double t = (room.floor_z - o.z()) / u.z();
        if (t > 0.0 and t <= rho_max and t < c.t) { c.t = t; c.sigma_r = room.floor_sigma / std::abs(u.z()); }
    }
    for (const auto& w : room.walls)
    {
        const Eigen::Vector2d e = w.b - w.a, d = u.head<2>();
        const double den = d.x() * e.y() - d.y() * e.x();
        if (den == 0.0) continue;                                  // parallel: never crossed
        const Eigen::Vector2d q = w.a - o.head<2>();
        const double t = (q.x() * e.y() - q.y() * e.x()) / den;     // along the ray
        const double sg = (q.x() * d.y() - q.y() * d.x()) / den;    // along the segment, [0, 1]
        if (t > 0.0 and t <= rho_max and sg >= 0.0 and sg <= 1.0 and t < c.t)
        {
            const double nu = std::abs(den) / e.norm();             // |n̂·u|, n̂ the wall's horizontal normal
            c.t = t; c.sigma_r = w.sigma / nu;
        }
    }
    return c;
}

// p_f(ρ) for a ray whose room crossing is (t_r, σ_r): p_pass·𝟙[ρ < t_r] + (1 − p_pass·t_r)·St_r(ρ − t_r), St_r the Student-t
// with scale² + σ_r² at the same ν (spec rev 3.3.7, σ by rev 3.3.8).
inline double room_foreign_density(double rho, const RoomCrossing& c, double rho_max, const StudentT& st)
{
    const double p_pass = 1.0 / rho_max;
    if (not std::isfinite(c.t)) return p_pass;
    return (rho < c.t ? p_pass : 0.0) + (1.0 - p_pass * c.t) * smeared(st, c.sigma_r).pdf(rho - c.t);
}

// Stage BeamRays::p_f, t_r and sigma_r for every ray of `rays` against `room` (after the endpoints are staged and ρ clamped).
inline void stage_room_foreign_density(BeamRays& rays, const RoomSurfaces& room, const StudentT& st)
{
    const std::size_t n = rays.endpoints.size();
    rays.p_f.resize(n); rays.t_r.resize(n); rays.sigma_r.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector3d du = rays.endpoints[i] - rays.origin;
        const double rho = du.norm();
        const RoomCrossing c = rho > 0.0 ? first_room_crossing(rays.origin, du / rho, rays.rho_max, room) : RoomCrossing{};
        rays.p_f[i] = rho > 0.0 ? room_foreign_density(rho, c, rays.rho_max, st) : 1.0 / rays.rho_max;
        rays.t_r[i] = c.t; rays.sigma_r[i] = c.sigma_r;
    }
}

}  // namespace rc::ai::beam
