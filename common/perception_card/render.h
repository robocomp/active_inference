// render.h -- render_zed: rasterises a sampled (imagined) object into the predicted ZED mask and depth image through the
// real pinhole (World.zed hfov/vfov, scaled to out_w x out_h) and mount (World.zed_mount_z), z-buffered against the world
// snapshot (named obstacles + walls as slabs [0, room_height]). Header-only; Eigen + std.
//
// Conventions are the evaluator's (viewpoint_score.h): heading yaw -> forward (cos, sin), right (sin, -cos); u = x_right /
// depth, v = (z - mount) / depth; column grows to the right, row grows downward. The walls are
// rc::nbv::wall_obstacles(polygon, the card's WallOcclusion thickness, gaps) with the target's own walls left out, as in
// detail::build_obstacles. One ray per output pixel, a slab test per oriented box in its local frame.
#pragma once

#include "perception_card/generative.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace rc::card
{
struct ImaginedImage
{
    int w = 0, h = 0;
    std::vector<std::uint8_t> visible, occluded;   // 0/1 per pixel: part is the nearest hit / part hit but behind something
    std::vector<float> depth;                      // z-depth (along the optical axis) of the nearest hit; 0 where nothing is hit
    float fill_max = 0;                            // max-axis bounding-box fraction of visible U occluded (the agents' ROI fill)
    float visible_frac = 0;                        // |visible| / |visible U occluded|
};

namespace detail
{
struct RayBox { float cx, cy, c, s, hw, hh, z0, z1; };   // c,s = cos/sin(-yaw): room -> local

inline RayBox ray_box(const rc::nbv::Obstacle& o)
{
    const float inf = std::numeric_limits<float>::infinity();
    // z1 <= z0 = height unknown (viewpoint_score.h): taller than anything; walls are capped at the room height by the caller
    return {o.cx, o.cy, std::cos(-o.yaw), std::sin(-o.yaw), 0.5f * o.w, 0.5f * o.h, o.z0, o.has_height() ? o.z1 : inf};
}

// Entry distance (in units of d, so z-depth when d has unit forward component) of the ray o + t d into the box, or +inf.
// *inside is set when the origin is within the box (then the box is skipped: a camera cannot see the box it sits in).
inline float slab_hit(const RayBox& b, float ox, float oy, float oz, float dx, float dy, float dz, bool* inside)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float px = ox - b.cx, py = oy - b.cy;
    const float lox = b.c * px - b.s * py, loy = b.s * px + b.c * py;
    const float ldx = b.c * dx - b.s * dy, ldy = b.s * dx + b.c * dy;
    float tmin = 0.0f, tmax = inf;
    bool in = true;
    const auto axis = [&](float o, float d, float lo, float hi) -> bool
    {
        if (not (o > lo and o < hi)) in = false;
        if (std::abs(d) < 1e-9f) return o >= lo and o <= hi;
        float t0 = (lo - o) / d, t1 = (hi - o) / d;
        if (t0 > t1) std::swap(t0, t1);
        tmin = std::max(tmin, t0); tmax = std::min(tmax, t1);
        return tmin <= tmax;
    };
    const bool hit = axis(lox, ldx, -b.hw, b.hw) and axis(loy, ldy, -b.hh, b.hh) and axis(oz, dz, b.z0, b.z1);
    if (inside) *inside = in;
    return (hit and not in and tmin > 0.0f) ? tmin : inf;
}

}  // namespace detail

// The imagined ZED image of `placed` seen from `vp`. `card` is REQUIRED: it supplies the wall thickness and the own-wall
// skip, so the numbers agree with evaluate_camera on the same card. Callers add the note "room height 2.5 m assumed" when
// !world.room_height_known. The target must NOT also be in World.obstacles (it would occlude itself).
inline ImaginedImage render_zed(const World& world, const Viewpoint& vp, const Placed& placed, const Card& card,
                                int out_w = 320, int out_h = 180)
{
    namespace nbv = rc::nbv;
    ImaginedImage im;
    im.w = std::max(out_w, 1); im.h = std::max(out_h, 1);
    const std::size_t n = static_cast<std::size_t>(im.w) * static_cast<std::size_t>(im.h);
    im.visible.assign(n, 0); im.occluded.assign(n, 0); im.depth.assign(n, 0.0f);
    if (not world.zed or placed.boxes.empty()) return im;
    const nbv::Sensor& sensor = *world.zed;
    const float mount = world.zed_mount_z > 1e-3f ? world.zed_mount_z : sensor.height_m;
    const float inf = std::numeric_limits<float>::infinity();

    const nbv::Target tgt = detail::target_of(placed);
    const Eigen::Vector2f to = tgt.centre() - vp.xy;
    const float yaw = vp.yaw ? *vp.yaw : (to.norm() > 1e-6f ? std::atan2(to.y(), to.x()) : 0.0f);

    // occluders: walls as slabs [0, room_height] (own walls left out when the card says so), then the named obstacles
    const float wall_t = detail::wall_thickness_for(card);
    auto walls = nbv::wall_obstacles(world.polygon, wall_t, world.gaps);
    if (card.walls)
        std::erase_if(walls, [&](const nbv::Obstacle& w) { return detail::own_wall(w, *card.walls, tgt); });
    std::vector<detail::RayBox> others, parts;
    for (const auto& w : walls) { auto b = detail::ray_box(w); b.z0 = 0.0f; b.z1 = world.room_height; others.push_back(b); }
    for (const auto& [name, o] : world.obstacles) others.push_back(detail::ray_box(o));
    for (const auto& p : placed.boxes) parts.push_back(detail::ray_box(p));

    // a camera inside a part sees nothing (and the evaluator calls the stand infeasible: Inside)
    for (const auto& b : parts)
    {
        bool in = false;
        detail::slab_hit(b, vp.xy.x(), vp.xy.y(), mount, 1.0f, 0.0f, 0.0f, &in);
        if (in) return im;
    }

    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float thh = sensor.tan_half_h(), thv = sensor.tan_half_v();
    int c0 = im.w, c1 = -1, r0 = im.h, r1 = -1;
    std::size_t n_vis = 0, n_occ = 0;
    for (int j = 0; j < im.h; ++j)
    {
        const float v = -(static_cast<float>(j) + 0.5f - 0.5f * static_cast<float>(im.h)) / (0.5f * static_cast<float>(im.h)) * thv;
        for (int i = 0; i < im.w; ++i)
        {
            const float u = (static_cast<float>(i) + 0.5f - 0.5f * static_cast<float>(im.w)) / (0.5f * static_cast<float>(im.w)) * thh;
            // direction with unit forward component: forward + u * right + v * up
            const float dx = cy + u * sy, dy = sy - u * cy, dz = v;
            float t_part = inf, t_other = inf;
            for (const auto& b : parts) t_part = std::min(t_part, detail::slab_hit(b, vp.xy.x(), vp.xy.y(), mount, dx, dy, dz, nullptr));
            for (const auto& b : others) t_other = std::min(t_other, detail::slab_hit(b, vp.xy.x(), vp.xy.y(), mount, dx, dy, dz, nullptr));
            const std::size_t k = static_cast<std::size_t>(j) * static_cast<std::size_t>(im.w) + static_cast<std::size_t>(i);
            const float t = std::min(t_part, t_other);
            im.depth[k] = std::isfinite(t) ? t : 0.0f;
            if (not std::isfinite(t_part)) continue;
            (t_part < t_other ? im.visible : im.occluded)[k] = 1;
            (t_part < t_other ? n_vis : n_occ)++;
            c0 = std::min(c0, i); c1 = std::max(c1, i); r0 = std::min(r0, j); r1 = std::max(r1, j);
        }
    }
    if (n_vis + n_occ > 0)
    {
        im.fill_max = std::max(static_cast<float>(c1 - c0 + 1) / static_cast<float>(im.w), static_cast<float>(r1 - r0 + 1) / static_cast<float>(im.h));
        im.visible_frac = static_cast<float>(n_vis) / static_cast<float>(n_vis + n_occ);
    }
    return im;
}

struct ImaginedScan
{
    std::vector<Eigen::Vector3f> on_target, on_other;   // room-frame points: kept returns on the parts / on walls+obstacles in the sector
    std::vector<float> ring_elev_room_deg;              // per ring, at the sector's centre bearing (room frame)
    float lambda0 = 0;                                  // rays whose nearest hit is a part, before q_return and the range factor
    bool modelled = false;
    std::vector<std::string> notes;                     // e.g. "sensor inside a part"
    Eigen::Vector3f mount_xyz = Eigen::Vector3f::Zero();   // the sensor origin in the room frame
    float bearing_rad = 0;                              // the sector's centre bearing (room frame), toward the target
};

// The imagined LiDAR scan of `placed` from `vp`. `rig` is the BODY-frame rig (as evaluate_lidar takes it); null or ringless
// = not modelled. Each ring is cast through the real mount (rig_in_room) over the azimuths subtended by the parts' bounding
// box +-5 deg (all round if the sensor is inside it), against the parts, the world's obstacles and the wall slabs [0,
// room_height] (card thickness, own-wall skip). A ray with range in [range_min, range_max] that hits a part is kept with
// probability q_return (the card's, for `subtype`) and displaced along the ray by N(0, range_sigma_floor_m); lambda0 (rays whose nearest hit is a
// part) is counted BEFORE the range gate and q, as in the evaluator (its range factor lives on P); the gate only decides
// which points are pushed. The target must NOT also be in World.obstacles. Same seed, same points.
inline ImaginedScan render_lidar(const World& world, const Viewpoint& vp, const LidarRig* rig, const Placed& placed,
                                 std::uint64_t seed, const Card& card, std::string_view subtype = "")
{
    namespace nbv = rc::nbv;
    ImaginedScan sc;
    if (not rig or placed.boxes.empty()) return sc;
    const auto elevs = detail::ring_elevations_deg(rig->model);
    const float az_step = rig->model.azimuth_step_deg.value_or(0.0f) * detail::kDeg2Rad;
    if (elevs.empty() or az_step <= 0.0f) return sc;
    sc.modelled = true;

    const nbv::Target tgt = detail::target_of(placed);
    Hypothesis h; h.x = tgt.cx; h.y = tgt.cy;
    const LidarRig r = rig_in_room(*rig, vp, h);
    const Eigen::Vector3f o = r.mount.translation();
    const Eigen::Matrix3f R = r.mount.linear();
    const float rmin = r.model.range_min_m.value_or(0.0f), rmax = r.model.range_max_m.value_or(1e9f);
    const float sigma = std::max(r.model.range_sigma_floor_m.value_or(0.02f), 1e-3f);
    const float q = params_for(card, subtype).q_return;
    constexpr float pi = std::numbers::pi_v<float>;

    // the sector: bearings subtended by the footprint, +-5 deg; all round when the sensor is inside it
    const Eigen::Vector2f from = o.head<2>(), rel = tgt.centre() - from;
    const float bearing = rel.norm() > 1e-6f ? std::atan2(rel.y(), rel.x()) : 0.0f;
    const Eigen::Vector2f ex = tgt.axis_x(), ey = tgt.axis_y();
    bool inside = std::abs((from - tgt.centre()).dot(ex)) <= 0.5f * tgt.w and std::abs((from - tgt.centre()).dot(ey)) <= 0.5f * tgt.h;
    float lo = 0.0f, hi = 0.0f;
    for (int i = 0; i < 4 and not inside; ++i)
    {
        const Eigen::Vector2f c = tgt.centre() + ((i & 1) ? 0.5f : -0.5f) * tgt.w * ex + ((i & 2) ? 0.5f : -0.5f) * tgt.h * ey;
        const float a = std::remainder(std::atan2(c.y() - from.y(), c.x() - from.x()) - bearing, 2.0f * pi);
        lo = std::min(lo, a); hi = std::max(hi, a);
    }
    const float pad = 5.0f * detail::kDeg2Rad;
    sc.mount_xyz = o;
    sc.bearing_rad = bearing;
    for (const float e : elevs) sc.ring_elev_room_deg.push_back(detail::room_elevation(r.mount, e, bearing) / detail::kDeg2Rad);

    // occluders, as in render_zed
    auto walls = nbv::wall_obstacles(world.polygon, detail::wall_thickness_for(card), world.gaps);
    if (card.walls)
        std::erase_if(walls, [&](const nbv::Obstacle& w) { return detail::own_wall(w, *card.walls, tgt); });
    std::vector<detail::RayBox> others, parts;
    for (const auto& w : walls) { auto b = detail::ray_box(w); b.z0 = 0.0f; b.z1 = world.room_height; others.push_back(b); }
    for (const auto& [name, ob] : world.obstacles) others.push_back(detail::ray_box(ob));
    for (const auto& p : placed.boxes) parts.push_back(detail::ray_box(p));

    // a sensor inside a part sees nothing of it (mirrors render_zed's Inside): modelled, no points
    for (const auto& b : parts)
    {
        bool in = false;
        detail::slab_hit(b, o.x(), o.y(), o.z(), 1.0f, 0.0f, 0.0f, &in);
        if (in) { sc.notes.push_back("sensor inside a part: no returns"); return sc; }
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::normal_distribution<float> N(0.0f, sigma);
    const float inf = std::numeric_limits<float>::infinity();
    const int n_az = std::max(1, static_cast<int>(std::lround(2.0f * pi / az_step)));
    for (const float e_deg : elevs)
    {
        const float e = e_deg * detail::kDeg2Rad, ce = std::cos(e), se = std::sin(e);
        for (int k = 0; k < n_az; ++k)
        {
            const float az = static_cast<float>(k) * az_step;
            const Eigen::Vector3f d = R * Eigen::Vector3f(ce * std::cos(az), ce * std::sin(az), se);
            if (not inside)
            {
                const float rb = std::remainder(std::atan2(d.y(), d.x()) - bearing, 2.0f * pi);
                if (rb < lo - pad or rb > hi + pad) continue;
            }
            float t_part = inf, t_other = inf;
            for (const auto& b : parts) t_part = std::min(t_part, detail::slab_hit(b, o.x(), o.y(), o.z(), d.x(), d.y(), d.z(), nullptr));
            for (const auto& b : others) t_other = std::min(t_other, detail::slab_hit(b, o.x(), o.y(), o.z(), d.x(), d.y(), d.z(), nullptr));
            const float t = std::min(t_part, t_other);
            if (not std::isfinite(t)) continue;
            if (t_part < t_other) sc.lambda0 += 1.0f;   // counted BEFORE the range gate: the evaluator's range factor lives on P
            if (t < rmin or t > rmax) continue;
            if (t_part < t_other)
            {
                if (U(rng) < q) sc.on_target.push_back(o + (t + N(rng)) * d);
            }
            else sc.on_other.push_back(o + t * d);
        }
    }
    return sc;
}
}  // namespace rc::card
