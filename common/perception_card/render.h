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

inline RayBox ray_box(const rc::nbv::Obstacle& o, float room_height)
{
    const float inf = std::numeric_limits<float>::infinity();
    // z1 <= z0 = height unknown (viewpoint_score.h): taller than anything; walls are capped at the room height by the caller
    (void)room_height;
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

// The imagined ZED image of `placed` seen from `vp`. `card` (optional) supplies the wall thickness and the own-wall skip, so
// the numbers agree with evaluate_camera on the same card; without it, the legacy 0.12 m walls and no skip. Callers add the
// note "room height 2.5 m assumed" when !world.room_height_known.
inline ImaginedImage render_zed(const World& world, const Viewpoint& vp, const Placed& placed, int out_w = 320, int out_h = 180,
                                const Card* card = nullptr)
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

    // the target: the parts' footprint bounding box in the first part's frame (the sample's yaw); also the aim point
    nbv::Target tgt;
    {
        tgt.yaw = placed.boxes.front().yaw;
        const float c = std::cos(tgt.yaw), s = std::sin(tgt.yaw);
        float x0 = inf, x1 = -inf, y0 = inf, y1 = -inf, z0 = inf, z1 = -inf;
        for (const auto& b : placed.boxes)
        {
            const float bc = std::cos(b.yaw), bs = std::sin(b.yaw);
            for (const float sx : {-0.5f, 0.5f}) for (const float sy : {-0.5f, 0.5f})
            {
                const float px = b.cx + sx * b.w * bc - sy * b.h * bs, py = b.cy + sx * b.w * bs + sy * b.h * bc;
                const float lx = px * c + py * s, ly = -px * s + py * c;
                x0 = std::min(x0, lx); x1 = std::max(x1, lx); y0 = std::min(y0, ly); y1 = std::max(y1, ly);
            }
            z0 = std::min(z0, b.z0); z1 = std::max(z1, b.has_height() ? b.z1 : b.z0);
        }
        const float mx = 0.5f * (x0 + x1), my = 0.5f * (y0 + y1);
        tgt.cx = mx * c - my * s; tgt.cy = mx * s + my * c; tgt.w = x1 - x0; tgt.h = y1 - y0; tgt.z0 = z0; tgt.z1 = z1;
    }
    const Eigen::Vector2f to = tgt.centre() - vp.xy;
    const float yaw = vp.yaw ? *vp.yaw : (to.norm() > 1e-6f ? std::atan2(to.y(), to.x()) : 0.0f);

    // occluders: walls as slabs [0, room_height] (own walls left out when the card says so), then the named obstacles
    const float wall_t = card ? detail::wall_thickness_for(*card) : detail::kLegacyWallThicknessM;
    auto walls = nbv::wall_obstacles(world.polygon, wall_t, world.gaps);
    if (card and card->walls)
        std::erase_if(walls, [&](const nbv::Obstacle& w) { return detail::own_wall(w, *card->walls, tgt); });
    std::vector<detail::RayBox> others, parts;
    for (const auto& w : walls) { auto b = detail::ray_box(w, world.room_height); b.z0 = 0.0f; b.z1 = world.room_height; others.push_back(b); }
    for (const auto& [name, o] : world.obstacles) others.push_back(detail::ray_box(o, world.room_height));
    for (const auto& p : placed.boxes) parts.push_back(detail::ray_box(p, world.room_height));

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
}  // namespace rc::card
