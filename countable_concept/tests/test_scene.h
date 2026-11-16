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
#include <span>
#include <vector>

namespace rc::test
{
struct SceneBox  { Eigen::Vector2f c; float yaw, w, h, H; };   // floor-anchored (z in [0,H]); w along the yaw axis, h across it
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
}

inline std::vector<Eigen::Vector3f> raycast_scene(const Eigen::Vector3f& origin, std::span<const SceneBox> boxes,
                                                  std::span<const SceneWall> walls, bool floor, float az_step_deg,
                                                  std::span<const float> ring_elev_deg, float range_max_m)
{
    using namespace detail;
    constexpr float d2r = 0.017453292519943295f;
    std::vector<Eigen::Vector3f> out;
    const int n_az = static_cast<int>(std::lround(360.0f / az_step_deg));
    for (const float el_deg : ring_elev_deg)
    {
        const float el = el_deg * d2r;
        for (int k = 0; k < n_az; ++k)
        {
            const float az = (static_cast<float>(k) * az_step_deg) * d2r;
            const Eigen::Vector3f d(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float best = kInf;
            for (const SceneBox& b : boxes)
            {
                const float cy = std::cos(b.yaw), sy = std::sin(b.yaw);
                const Eigen::Vector2f p = origin.head<2>() - b.c, dd = d.head<2>();
                const Eigen::Vector3f lo(-0.5f * b.w, -0.5f * b.h, 0.0f), hi(0.5f * b.w, 0.5f * b.h, b.H);
                const Eigen::Vector3f ol(cy * p.x() + sy * p.y(), -sy * p.x() + cy * p.y(), origin.z());
                const Eigen::Vector3f dl(cy * dd.x() + sy * dd.y(), -sy * dd.x() + cy * dd.y(), d.z());
                best = std::min(best, slab(ol, dl, lo, hi));
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
            if (best <= range_max_m) out.emplace_back(origin + best * d);
        }
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
}
