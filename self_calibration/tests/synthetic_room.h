// synthetic_room.h -- header-only test fixture: a 2-D polyline room, a ray caster and seeded noise.
//   Room2D            closed polygon (vertices in the room frame, any winding)
//   notched_room()    4 x 6 m rectangle centred on the origin with a 1 x 0.8 m notch in the +y wall
//   corridor(L, W)    L x W rectangle centred on the origin (long axis x)
//   raycast(room, pose_xyt, step_rad, noise_sigma, rng, max_range)
//                     one ray every step_rad over 360 deg from pose (x, y, theta) in the room frame; returns the
//                     hits in the SENSOR frame of that pose, range perturbed by N(0, noise_sigma) (0 = exact).
//                     Rays whose hit lies beyond max_range are dropped.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

namespace sc::calib::testing
{
struct Room2D { std::vector<Eigen::Vector2d> poly; };

inline Room2D notched_room()
{ return { { {-2, -3}, {2, -3}, {2, 3}, {0.5, 3}, {0.5, 2.2}, {-0.5, 2.2}, {-0.5, 3}, {-2, 3} } }; }

inline Room2D corridor(double length, double width)
{
    const double a = 0.5 * length, b = 0.5 * width;
    return { { {-a, -b}, {a, -b}, {a, b}, {-a, b} } };
}

inline std::vector<Eigen::Vector2f> raycast(const Room2D& room, const Eigen::Vector3d& pose_xyt, double step_rad,
                                            double noise_sigma, std::mt19937& rng,
                                            double max_range = std::numeric_limits<double>::infinity())
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Eigen::Vector2f> out;
    const Eigen::Vector2d o = pose_xyt.head<2>();
    const int n = int(std::llround(2.0 * std::numbers::pi / step_rad));
    for (int k = 0; k < n; ++k)
    {
        const double a = k * step_rad;
        const double ang = pose_xyt.z() + a;
        const Eigen::Vector2d d(std::cos(ang), std::sin(ang));
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < room.poly.size(); ++i)
        {
            const Eigen::Vector2d p = room.poly[i], e = room.poly[(i + 1) % room.poly.size()] - p;
            const double den = d.x() * e.y() - d.y() * e.x();               // d x e
            if (std::abs(den) < 1e-12) continue;                            // parallel
            const Eigen::Vector2d w = p - o;
            const double t = (w.x() * e.y() - w.y() * e.x()) / den;         // along the ray
            const double s = (w.x() * d.y() - w.y() * d.x()) / den;         // along the edge
            if (t > 1e-9 and s >= 0.0 and s <= 1.0) best = std::min(best, t);
        }
        if (not std::isfinite(best) or best > max_range) continue;
        const double r = best + (noise_sigma > 0.0 ? noise_sigma * nd(rng) : 0.0);
        out.emplace_back(float(r * std::cos(a)), float(r * std::sin(a)));
    }
    return out;
}
}  // namespace sc::calib::testing
