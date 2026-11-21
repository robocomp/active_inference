// test_scene.h — physical first-hit test scenes (spec 2026-11-15 lidar-measures §7c.0).
// Every LiDAR fixture is built by ray-casting a TRUE scene (boxes ∪ walls ∪ floor); one return per (ring, azimuth) ray, the
// FIRST hit only. A fixture that places returns by hand puts wall returns on rays the true box occludes — an artefact
// (rev 1/rev 2 reported a back-face pull that did not exist).
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace rc::test
{
// z in [z0, H]: floor-anchored by default (z0 = 0); z0 > 0 raises the body on a gap (legs not modelled, spec 7c.8 Pass "Gap under
// furniture"). w along the yaw axis, h across it.
struct SceneBox  { Eigen::Vector2f c; float yaw, w, h, H; float z0 = 0.0f; };
struct SceneWall { Eigen::Vector2f a, b; float height; };      // vertical rectangle on the segment a-b, z in [0, height]

// RS-Helios H32F70 as simulated (webots-shadow/components/shadow-bridge-dds/etc/config.toml):
//  - [LidarChannels.helios] lines 236-250: Webots generates 50 symmetric layers over 110 deg (+-55); the window
//    ElevMinDeg = -55, ElevMaxDeg = 15 keeps layers 18..49 = 32 channels at +14.6 .. -55.0 deg, i.e.
//    elev_k = 55 - k * 110/49 deg, k = 18..49 (the [Lidar.helios] section at line 397 itself carries no ring list: "rings,
//    ring elevations ... READ FROM THE WEBOTS DEVICE").
//  - azimuth step: the device is horizontalResolution 900 over fieldOfView 6.28 rad (webots-shadow/protos/Shadow.proto,
//    Lidar "helios") = 0.4 deg.
inline constexpr float kHeliosAzStepDeg = 360.0f / 900.0f;
inline std::array<float, 32> helios_ring_elev_deg()
{
    std::array<float, 32> e{};
    for (int k = 18; k < 50; ++k) e[k - 18] = 55.0f - static_cast<float>(k) * 110.0f / 49.0f;
    return e;
}

namespace detail
{
constexpr float kInf = std::numeric_limits<float>::infinity();
// slab test of origin + t d against the box [lo, hi]; returns the entry t (>= 0) or kInf
inline float slab(const Eigen::Vector3f& o, const Eigen::Vector3f& d, const Eigen::Vector3f& lo, const Eigen::Vector3f& hi)
{
    float t0 = 0.0f, t1 = kInf;
    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(d(i)) < 1e-9f) { if (o(i) < lo(i) or o(i) > hi(i)) return kInf; continue; }
        float a = (lo(i) - o(i)) / d(i), b = (hi(i) - o(i)) / d(i);
        if (a > b) std::swap(a, b);
        t0 = std::max(t0, a); t1 = std::min(t1, b);
        if (t0 > t1) return kInf;
    }
    return t0;
}
// entry t (>= 0) of origin + t d into the (yawed) box b, or kInf
inline float box_entry(const Eigen::Vector3f& origin, const Eigen::Vector3f& d, float yaw, const Eigen::Vector2f& c,
                       const Eigen::Vector3f& lo, const Eigen::Vector3f& hi)
{
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const Eigen::Vector2f p = origin.head<2>() - c, dd = d.head<2>();
    const Eigen::Vector3f ol(cy * p.x() + sy * p.y(), -sy * p.x() + cy * p.y(), origin.z());
    const Eigen::Vector3f dl(cy * dd.x() + sy * dd.y(), -sy * dd.x() + cy * dd.y(), d.z());
    return slab(ol, dl, lo, hi);
}
}

// First hit (t along the unit d; kInf if none) of the ray origin + t d with the scene. `dilate` grows every box by that much on
// each side and on top (its floor-anchored bottom stays): a per-ray surface offset δ (spec 7c.8 Pass "Calibration").
inline float first_hit(const Eigen::Vector3f& origin, const Eigen::Vector3f& d, std::span<const SceneBox> boxes,
                       std::span<const SceneWall> walls, bool floor, float dilate = 0.0f)
{
    using namespace detail;
    float best = kInf;
    for (const SceneBox& b : boxes)
    {
        const Eigen::Vector3f lo(-0.5f * b.w - dilate, -0.5f * b.h - dilate, b.z0), hi(0.5f * b.w + dilate, 0.5f * b.h + dilate, b.H + dilate);
        best = std::min(best, box_entry(origin, d, b.yaw, b.c, lo, hi));
    }
    for (const SceneWall& w : walls)
    {
        const Eigen::Vector2f e = w.b - w.a, dd = d.head<2>();
        const float den = dd.x() * e.y() - dd.y() * e.x();           // cross(dd, e)
        if (std::abs(den) < 1e-9f) continue;
        const Eigen::Vector2f q = w.a - origin.head<2>();
        const float t = (q.x() * e.y() - q.y() * e.x()) / den;       // along the ray (horizontal-unit-scaled)
        const float s = (q.x() * dd.y() - q.y() * dd.x()) / den;     // along the segment
        if (t <= 0.0f or s < 0.0f or s > 1.0f) continue;
        const float z = origin.z() + t * d.z();
        if (z < 0.0f or z > w.height) continue;
        best = std::min(best, t);
    }
    if (floor and d.z() < -1e-9f) best = std::min(best, -origin.z() / d.z());
    return best;
}

// The unit direction of the (ring elevation, azimuth) beam.
inline Eigen::Vector3f beam_dir(float el_deg, float az_deg)
{
    constexpr float d2r = 0.017453292519943295f;
    const float el = el_deg * d2r, az = az_deg * d2r;
    return {std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
}

inline std::vector<Eigen::Vector3f> raycast_scene(const Eigen::Vector3f& origin, std::span<const SceneBox> boxes,
                                                  std::span<const SceneWall> walls, bool floor, float az_step_deg,
                                                  std::span<const float> ring_elev_deg, float range_max_m)
{
    std::vector<Eigen::Vector3f> out;
    const int n_az = static_cast<int>(std::lround(360.0f / az_step_deg));
    for (const float el_deg : ring_elev_deg)
        for (int k = 0; k < n_az; ++k)
        {
            const Eigen::Vector3f d = beam_dir(el_deg, static_cast<float>(k) * az_step_deg);
            const float best = first_hit(origin, d, boxes, walls, floor);
            if (best <= range_max_m) out.emplace_back(origin + best * d);
        }
    return out;
}

// The returns the camera mask would select: endpoints inside the truth box inflated by `margin` (xy) and up to H + margin.
// Floor returns (z < 2 cm) are NOT selected: the mask covers the object, not the ground around it (the floor is still in the
// scene, so it occludes and is hit by every ray that misses the box).
inline std::vector<Eigen::Vector3f> select_near(const std::vector<Eigen::Vector3f>& pts, const SceneBox& t, float margin)
{
    std::vector<Eigen::Vector3f> out;
    for (const auto& p : pts)
        if (std::abs(p.x() - t.c.x()) <= 0.5f * t.w + margin and std::abs(p.y() - t.c.y()) <= 0.5f * t.h + margin and p.z() <= t.H + margin and p.z() > 0.02f)
            out.push_back(p);
    return out;
}

// ── The camera side of a live-staging fixture (spec 7c.8 Pass: "a membership built from a rendered mask, σ_px and ε_px as
//    live"). A pinhole ZED in the agent's convention (countable_projection.cpp: X right, Y forward = depth, Z up;
//    col = cols/2 + X/Y·f, row = rows/2 − Z/Y·f), upright (ROBOT_GEOMETRY.md: the zed mount has no rotation). ──
struct Camera
{
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    float heading = 1.5707963f;   // room yaw of the optical (depth) axis; π/2 looks along +y, image right = +x
    float f = 350.0f;             // focal length (px) in the delivered image
    int   cols = 640, rows = 480;
    Eigen::Vector3f depth_axis() const { return {std::cos(heading), std::sin(heading), 0.0f}; }
    Eigen::Vector3f right_axis() const { return {std::sin(heading), -std::cos(heading), 0.0f}; }
    // (col, row, depth) of a room point, or nothing behind the camera / outside the image (project_to_zed's contract).
    std::optional<Eigen::Vector3f> project(const Eigen::Vector3f& p) const
    {
        const Eigen::Vector3f v = p - pos;
        const float X = v.dot(right_axis()), Y = v.dot(depth_axis()), Z = v.z();
        if (not (Y > 1e-3f)) return std::nullopt;
        const float col = 0.5f * cols + X / Y * f, row = 0.5f * rows - Z / Y * f;
        if (col < 0.0f or row < 0.0f or col >= static_cast<float>(cols) or row >= static_cast<float>(rows)) return std::nullopt;
        return Eigen::Vector3f(col, row, Y);
    }
    // The unit ray through pixel centre (u + ½, v + ½).
    Eigen::Vector3f pixel_ray(int u, int v) const
    {
        const float x = (static_cast<float>(u) + 0.5f - 0.5f * cols) / f, z = -(static_cast<float>(v) + 0.5f - 0.5f * rows) / f;
        return (right_axis() * x + depth_axis() + Eigen::Vector3f(0.0f, 0.0f, z)).normalized();
    }
};

// The target's mask as the detector would deliver it: every pixel whose FIRST hit (target eroded by erode_m on every side and
// from the top, against the full `others`) is the target. Pixels in the delivered image (col, row) = pixel centres.
struct RenderedMask { std::vector<Eigen::Vector2f> pixels; };
inline RenderedMask render_mask(const Camera& cam, const SceneBox& target, std::span<const SceneBox> others, float erode_m)
{
    using namespace detail;
    RenderedMask m;
    const Eigen::Vector3f tlo(-0.5f * target.w + erode_m, -0.5f * target.h + erode_m, target.z0),
                          thi(0.5f * target.w - erode_m, 0.5f * target.h - erode_m, target.H - erode_m);
    for (int v = 0; v < cam.rows; ++v)
        for (int u = 0; u < cam.cols; ++u)
        {
            const Eigen::Vector3f d = cam.pixel_ray(u, v);
            const float tt = box_entry(cam.pos, d, target.yaw, target.c, tlo, thi);
            if (tt == kInf) continue;
            bool occluded = false;
            for (const SceneBox& b : others)
            {
                const float to = box_entry(cam.pos, d, b.yaw, b.c, Eigen::Vector3f(-0.5f * b.w, -0.5f * b.h, b.z0),
                                           Eigen::Vector3f(0.5f * b.w, 0.5f * b.h, b.H));
                if (to < tt) { occluded = true; break; }
            }
            if (not occluded) m.pixels.emplace_back(static_cast<float>(u) + 0.5f, static_cast<float>(v) + 0.5f);
        }
    return m;
}
}
