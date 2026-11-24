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
 * Each kept sample also carries its solid angle dΩ = dA·|n·d̂|/d²: mask points are spread over pixels, so a sample stands
 * for a share dΩ/Ω of the look's points and a grazing face gets few. kGrid is the quadrature resolution, not a decision
 * threshold.
 *
 * The LiDAR half (synth_rays): the sweep the Helios would return from the standpoint, staged by the LIVE staging function.
 *   - the sweep: the sensor's origin at the standpoint (OfferLidar::offset, measured by the caller from the RT chain), its own
 *     ring elevations (room frame) and azimuth step (the advertised SensorModel); each ray returns at its first surface — a
 *     beam part, an obstacle (viewpoint_score.h's footprint/height test), or the room's known surfaces (floor, walls,
 *     neighbours) within the sensor's range; a ray that meets none returns nothing, as on the device;
 *   - the label: the camera's, as the live persistent label is formed — nullopt where the return does not project into the
 *     camera's field of view from the standpoint, else q = 1 on the object (the return is the part's) and 0 elsewhere,
 *     m = ε_px + (1 − 2ε_px)q;
 *   - the staging: rc::lidar_select::select_rays_beam itself, with the agent's select box (OfferLidar::box: the fitter's
 *     countable_select_box) about the look's mask centroid, then p_f from the room's known surfaces (beam_room.h) — so the
 *     promise stages the same free-space rays a live look stages.
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

#include <optional>

#include "beam_ray.h"                      // rc::ai::beam::BeamPart, BeamRays
#include "beam_room.h"                     // rc::ai::beam::RoomSurfaces, stage_room_foreign_density
#include "../lidar_select/lidar_select.h"  // rc::lidar_select::Box, select_rays_beam (the live staging)
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

// The LiDAR an offer puts to work, as the agent stages its rays (absent ⇒ the offer is camera-only).
struct OfferLidar
{
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();   // origin − camera at the standpoint: x right, y forward (heading
                                                         // frame), z = the origin's height above the floor
    std::vector<double> ring_elev_rad;                  // each ring's elevation, room frame
    double azimuth_step_rad = 0.0;                      // the sensor's azimuth step
    double rho_max = 6.0, sigma_s = 0.01;               // the staging's ρ_max and the model σ (BeamRays fields)
    double range_max = 6.0;                             // the sensor's own range: a surface beyond it returns nothing
    rc::lidar_select::Box box;                          // the agent's select box (radius about the mask centroid, z band)
    double eps_px = 0.0;                                // the camera mask's pixel-error rate (the persistent label's ε)
    rc::ai::beam::RoomSurfaces room;                    // the known surfaces of p_f (floor, walls, neighbours)
    rc::ai::beam::StudentT st;                          // the beam density (p_f's)
};

// The sensors an offer puts to work and the scene they look through: the planner's own camera Sensor (both FoVs and the
// optical centre's height above the floor), its obstacle list, and the LiDAR.
struct OfferSensors
{
    rc::nbv::Sensor sensor;
    std::span<const rc::nbv::Obstacle> obstacles;
    std::optional<OfferLidar> lidar;
};

// The camera's optical centre, room frame, at a standpoint.
inline Eigen::Vector3f camera_at(const Standpoint& s, const OfferSensors& c)
{
    return {s.xy.x(), s.xy.y(), c.sensor.height_m};
}

struct View
{
    std::vector<Eigen::Vector3f> points;   // the seen samples, room frame
    std::vector<double> d_omega;           // each one's solid angle dA·|n·d̂|/d² (sr)
    double solid_angle = 0.0;              // their sum (sr)
};

inline constexpr int kGrid = 8;            // samples per face side (quadrature resolution)

// The surface of `parts` a camera at `cam` with optical heading `heading` sees.
inline View visible_surface(std::span<const rc::ai::beam::BeamPart> parts, const Eigen::Vector3f& cam, float heading,
                            const OfferSensors& c)
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
                        const double dw = dA * std::abs(n.dot(d)) / (dn * dn * dn);
                        v.points.push_back(p);
                        v.d_omega.push_back(dw);
                        v.solid_angle += dw;
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

// The LiDAR origin at a standpoint: the camera's position plus the offset in the heading frame.
inline Eigen::Vector3d lidar_origin(const Standpoint& s, const OfferLidar& L)
{
    const double c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2d fwd(c, sn), right(sn, -c);
    const Eigen::Vector2d xy = s.xy.cast<double>() + double(L.offset.y()) * fwd + double(L.offset.x()) * right;
    return {xy.x(), xy.y(), double(L.offset.z())};
}

// The widest gap between adjacent ring elevations (rad): the g of rho_lidar_lateral / rho_lidar_height. 0 with < 2 rings.
inline float max_ring_gap(std::vector<double> elev)
{
    std::ranges::sort(elev);
    double g = 0.0;
    for (std::size_t i = 1; i < elev.size(); ++i) g = std::max(g, elev[i] - elev[i - 1]);
    return static_cast<float>(g);
}

// Entry t of o + t·u into one convex part (slab test, local frame); +inf when it misses or the entry is behind o.
inline double part_entry(const rc::ai::beam::BeamPart& P, const Eigen::Vector3d& o, const Eigen::Vector3d& u)
{
    const double cy = std::cos(P.yaw), sy = std::sin(P.yaw), dx = o.x() - P.c.x(), dy = o.y() - P.c.y();
    const Eigen::Vector3d ol(cy * dx + sy * dy, -sy * dx + cy * dy, o.z() - P.c.z());
    const Eigen::Vector3d ul(cy * u.x() + sy * u.y(), -sy * u.x() + cy * u.y(), u.z());
    double a = 0.0, b = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i)
    {
        if (ul(i) == 0.0) { if (std::abs(ol(i)) > P.half(i)) return std::numeric_limits<double>::infinity(); continue; }
        double t1 = (-P.half(i) - ol(i)) / ul(i), t2 = (P.half(i) - ol(i)) / ul(i);
        if (t1 > t2) std::swap(t1, t2);
        a = std::max(a, t1); b = std::min(b, t2);
        if (a > b) return std::numeric_limits<double>::infinity();
    }
    return a;
}

// Entry t ∈ (0, t_far] of o + t·u into an obstacle (viewpoint_score.h's test decides; its footprint entry gives t); +inf
// when it does not block.
inline double obstacle_entry(const rc::nbv::Obstacle& ob, const Eigen::Vector3d& o, const Eigen::Vector3d& u, double t_far)
{
    const Eigen::Vector3f a = o.cast<float>(), b = (o + t_far * u).cast<float>();
    if (not rc::nbv::detail::segment_blocked_3d(a, b, ob)) return std::numeric_limits<double>::infinity();
    float t0 = 0.0f, t1 = 0.0f;
    rc::nbv::detail::segment_box_interval(a.head<2>(), b.head<2>(), ob, t0, t1);
    return std::max(0.0, double(t0)) * t_far;
}

// The ray-set a live look from the standpoint would stage (see the file header). `cam`/`heading`: the camera at the
// standpoint (for the label); `centroid_xy`: the look's mask centroid (the select box's anchor, as live).
inline rc::ai::beam::BeamRays synth_rays(std::span<const rc::ai::beam::BeamPart> parts, const Eigen::Vector3d& origin,
                                         const OfferLidar& L, const OfferSensors& sn, const Eigen::Vector3f& cam,
                                         float heading, const Eigen::Vector2f& centroid_xy)
{
    rc::ai::beam::BeamRays out;
    out.origin = origin;
    out.rho_max = L.rho_max;
    out.sigma_s = L.sigma_s;
    if (L.ring_elev_rad.empty() or not (L.azimuth_step_rad > 0.0)) return out;
    // azimuths over the select cylinder's angular span from the origin (the whole circle from inside it)
    const Eigen::Vector2d to_c = centroid_xy.cast<double>() - origin.head<2>();
    const double dist = to_c.norm(), r = double(L.box.radius_xy);
    const double bearing = std::atan2(to_c.y(), to_c.x());
    const double half = dist > r ? std::asin(r / dist) : std::numbers::pi;
    std::vector<Eigen::Vector3f> sweep;
    std::vector<char> on_object;
    for (long k = long(std::ceil((bearing - half) / L.azimuth_step_rad)); double(k) * L.azimuth_step_rad <= bearing + half; ++k)
    {
        const double az = double(k) * L.azimuth_step_rad;
        for (const double el : L.ring_elev_rad)
        {
            const Eigen::Vector3d u(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            double t_part = std::numeric_limits<double>::infinity();
            for (const auto& P : parts) t_part = std::min(t_part, part_entry(P, origin, u));
            double t = std::min(t_part, rc::ai::beam::first_room_crossing(origin, u, L.range_max, L.room).t);
            for (const auto& ob : sn.obstacles) t = std::min(t, obstacle_entry(ob, origin, u, std::min(t, L.range_max)));
            if (not (t <= L.range_max)) continue;                         // nothing within the sensor's range: no return
            sweep.push_back((origin + t * u).cast<float>());
            on_object.push_back(t == t_part ? 1 : 0);
        }
    }
    // the camera's label: nullopt outside its field of view from the standpoint (visible_surface's frustum test)
    const float half_h = 0.5f * std::clamp(sn.sensor.hfov_rad, 0.1f, 3.0f), half_v = 0.5f * std::clamp(sn.sensor.vfov_rad, 0.1f, 3.0f);
    const auto label = [&](const Eigen::Vector3f& p) -> std::optional<std::pair<double, float>>
    {
        const Eigen::Vector3f rel = p - cam;
        const float range = rel.head<2>().norm();
        if (range < 1e-3f) return std::nullopt;                           // degenerate geometry: over the camera
        if (std::abs(std::remainder(std::atan2(rel.y(), rel.x()) - heading, 2.0f * std::numbers::pi_v<float>)) > half_h)
            return std::nullopt;
        if (sn.sensor.has_vertical() and std::abs(std::atan2(rel.z(), range)) > half_v) return std::nullopt;
        // select_rays_beam calls label(sweep[i]): the index is the element's position in `sweep`
        const float q = on_object[std::size_t(&p - sweep.data())] ? 1.0f : 0.0f;
        return std::pair<double, float>{L.eps_px + (1.0 - 2.0 * L.eps_px) * double(q), q};
    };
    rc::lidar_select::select_rays_beam(sweep, origin.cast<float>(), centroid_xy, L.box, rc::lidar_select::Weights{}, L.rho_max,
                                       label, out);
    out.sigma_s = L.sigma_s;                                              // as stage_beam_cycle: the model σ is the caller's
    rc::ai::beam::stage_room_foreign_density(out, L.room, L.st);
    return out;
}

}  // namespace rc::ai::offer
