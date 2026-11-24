/*
 * common/ai_belief/offer_view.h — what a camera standing at an offered standpoint would see of a model's beam parts
 * (spec 2026-11-23 shared-engine G5; header-only, pure geometry).
 *
 * The engine's offer gain (PersistentEngine::predicted_gain) needs the camera points a look from a standpoint would give.
 * They are synthesised here from the model's convex beam parts and the camera model the planner already uses
 * (common/nbv/viewpoint_score.h: Sensor for the field of view and the optical centre's height, Obstacle for occluders):
 *
 *   - each part's six faces; a face is seen when its outward normal points at the camera (the part is convex, so a face
 *     turned away is hidden by the part itself);
 *   - a kGrid × kGrid grid of samples at the cell centres of every seen face;
 *   - a sample counts when it is inside the frustum (viewpoint_score.h's visible_fraction tests, the same field of view and
 *     the same vertical test) and no obstacle blocks the line of sight. The occlusion test IS viewpoint_score.h's
 *     (detail::segment_blocked_3d when an obstacle declares a height, detail::segment_hits_box otherwise, decided once per
 *     view as visible_fraction decides it), so the offer gain sees exactly the occluders the planner sees.
 *
 * Each kept sample also carries its solid angle, dA·|n·d̂|/d², so the caller can scale the number of mask points with how
 * many pixels the visible surface covers. kGrid is the quadrature resolution, not a decision threshold.
 *
 * Not modelled: one beam part hiding another part of the same model (the countable's headboard behind its box). Parts are
 * tested against the obstacles only.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "beam_ray.h"                      // rc::ai::beam::BeamPart
#include "../nbv/viewpoint_score.h"        // rc::nbv::Sensor, Obstacle, the occlusion test

namespace rc::ai::offer
{

// A robot standpoint as the planner offers it: the camera's position on the floor plane (plan_faces' viewpoint, which
// viewpoint_score.h treats as the camera's position) and its optical axis' heading (rad, room frame).
struct Standpoint
{
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();
    float yaw = 0.0f;
};

// The camera an offer puts to work and the scene it looks through: the planner's own Sensor (both FoVs and the optical
// centre's height above the floor) and its obstacle list.
struct OfferCamera
{
    rc::nbv::Sensor sensor;
    std::span<const rc::nbv::Obstacle> obstacles;
};

// The camera's optical centre, room frame, at a standpoint.
inline Eigen::Vector3f camera_at(const Standpoint& s, const OfferCamera& c)
{
    return {s.xy.x(), s.xy.y(), c.sensor.height_m};
}

struct View
{
    std::vector<Eigen::Vector3f> points;   // the seen samples, room frame
    double solid_angle = 0.0;              // Σ dA·|n·d̂|/d² over them (sr)
};

inline constexpr int kGrid = 8;            // samples per face side (quadrature resolution)

// The surface of `parts` a camera at `cam` with optical heading `heading` sees.
inline View visible_surface(std::span<const rc::ai::beam::BeamPart> parts, const Eigen::Vector3f& cam, float heading,
                            const OfferCamera& c)
{
    View v;
    const rc::nbv::Sensor& sn = c.sensor;
    // visible_fraction's field-of-view half angles, clamped as it clamps them
    const float half_h = 0.5f * std::clamp(sn.hfov_rad, 0.1f, 3.0f);
    const float half_v = 0.5f * std::clamp(sn.vfov_rad, 0.1f, 3.0f);
    const bool do_vert = sn.has_vertical();
    const bool z_aware = do_vert and std::ranges::any_of(c.obstacles, [](const rc::nbv::Obstacle& o) { return o.has_height(); });
    const Eigen::Vector2f cam2 = cam.head<2>();

    for (const auto& P : parts)
    {
        const double cy = std::cos(P.yaw), sy = std::sin(P.yaw);
        const std::array<Eigen::Vector3d, 3> axis{Eigen::Vector3d(cy, sy, 0.0), Eigen::Vector3d(-sy, cy, 0.0), Eigen::Vector3d::UnitZ()};
        for (int a = 0; a < 3; ++a)
            for (const double sgn : {1.0, -1.0})
            {
                const Eigen::Vector3d n = sgn * axis[a];
                const Eigen::Vector3d fc = P.c + sgn * P.half(a) * axis[a];
                if (not (n.dot(cam.cast<double>() - fc) > 0.0)) continue;         // turned away: hidden by the part itself
                const int u = (a + 1) % 3, w = (a + 2) % 3;
                const double dA = (2.0 * P.half(u) / kGrid) * (2.0 * P.half(w) / kGrid);
                for (int i = 0; i < kGrid; ++i)
                    for (int j = 0; j < kGrid; ++j)
                    {
                        const double su = (2.0 * (i + 0.5) / kGrid - 1.0) * P.half(u);
                        const double sw = (2.0 * (j + 0.5) / kGrid - 1.0) * P.half(w);
                        const Eigen::Vector3d pd = fc + su * axis[u] + sw * axis[w];
                        const Eigen::Vector3f p = pd.cast<float>();
                        const Eigen::Vector3f rel = p - cam;
                        const float range = rel.head<2>().norm();
                        if (range < 1e-3f) continue;                                // degenerate geometry: the camera over the sample
                        float bearing = std::atan2(rel.y(), rel.x()) - heading;
                        bearing = std::remainder(bearing, 2.0f * std::numbers::pi_v<float>);
                        if (std::abs(bearing) > half_h) continue;
                        if (do_vert and std::abs(std::atan2(rel.z(), range)) > half_v) continue;
                        bool blocked = false;
                        for (const auto& o : c.obstacles)
                            if (z_aware ? rc::nbv::detail::segment_blocked_3d(cam, p, o)
                                        : rc::nbv::detail::segment_hits_box(cam2, p.head<2>(), o)) { blocked = true; break; }
                        if (blocked) continue;
                        const Eigen::Vector3d d = cam.cast<double>() - pd;
                        const double dn = d.norm();
                        v.points.push_back(p);
                        v.solid_angle += dA * std::abs(n.dot(d)) / (dn * dn * dn);
                    }
            }
    }
    return v;
}

// The parts' extent across the line of sight from `from` to `centre` (the w⊥ of rho_camera): the spread of every part's
// footprint corners along the horizontal perpendicular t̂. For one box this is w|ex·t̂| + h|ey·t̂|, the countable fitter's
// own w⊥.
inline float extent_across(std::span<const rc::ai::beam::BeamPart> parts, const Eigen::Vector2f& from, const Eigen::Vector2f& centre)
{
    const Eigen::Vector2f r = from - centre;
    const float z = r.norm();
    if (z < 1e-3f or parts.empty()) return 0.0f;                                // degenerate geometry
    const Eigen::Vector2d t(-r.y() / z, r.x() / z);
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const auto& P : parts)
    {
        const Eigen::Vector2d ex(std::cos(P.yaw), std::sin(P.yaw)), ey(-std::sin(P.yaw), std::cos(P.yaw));
        for (const double sx : {1.0, -1.0})
            for (const double sy : {1.0, -1.0})
            {
                const double s = (P.c.head<2>() + sx * P.half.x() * ex + sy * P.half.y() * ey).dot(t);
                lo = std::min(lo, s); hi = std::max(hi, s);
            }
    }
    return static_cast<float>(hi - lo);
}

}  // namespace rc::ai::offer
