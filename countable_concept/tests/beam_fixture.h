// beam_fixture.h — the persistent path's ray-set (spec 2026-11-15 lidar-measures §7c.8 a0) for the belief tests: a
// ray-cast TRUE scene (test_scene.h, first hit only) staged by RAY GEOMETRY through the real staging function
// (rc::lidar_select::select_rays_beam: every return whose ray, continued to ρ_max, crosses the select cylinder ∩ z-band),
// with HAND labels by default (m_on on the true object, ε_px off it; spec Pass "Silhouette, hand labels") and κ set by the
// caller (κ = 1 unless a test is about κ). sigma_s is the model σ (the caller of select_rays_beam sets it; spec Task 18).
// The room's known surfaces (spec 7c.8 rev 3.3.7) — the floor z = 0 and the scene's walls — enter every ray's foreign density
// p_f (beam_room.h); stage_beam takes them explicitly, helios_beam passes the floor and the scene walls.
#pragma once
#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include "countable_beam_config.h"
#include "ai_belief/beam_room.h"
#include "lidar_select/lidar_select.h"
#include "test_scene.h"

namespace rc::test
{
inline constexpr double kEpsPx = 0.02;    // ε_px of the hand labels (spec 7c.8 a0 example value)
inline constexpr double kMOn   = 0.9;     // the hand label on the object

// The beam config every belief test uses: LidarPrecision 2500, c = 5 cm, σ_sensor = 0 (the fixtures' ranges are exact).
inline rc::ai::beam::BeamConfig test_beam_config() { return *rc::make_beam_config(kEpsPx, 2500.0, 0.05, 0.0); }

// Signed distance from p to the floor-anchored box b (the scene's own geometry; ≤ tol ⇒ the return is on b).
inline float scene_box_sdf(const SceneBox& b, const Eigen::Vector3f& p)
{
    const float c = std::cos(b.yaw), s = std::sin(b.yaw), dx = p.x() - b.c.x(), dy = p.y() - b.c.y();
    const Eigen::Vector3f a(std::abs(c * dx + s * dy) - 0.5f * b.w, std::abs(-s * dx + c * dy) - 0.5f * b.h,
                            std::abs(p.z() - 0.5f * b.H) - 0.5f * b.H);
    return a.cwiseMax(0.0f).norm() + std::min(a.maxCoeff(), 0.0f);
}
inline bool on_object(std::span<const SceneBox> obj, const Eigen::Vector3f& p, float tol = 2e-3f)
{
    for (const auto& b : obj) if (scene_box_sdf(b, p) <= tol) return true;
    return false;
}

// Hand labels: m_on on the object's boxes, m_off elsewhere (q = m: what κ's coverage counts is irrelevant when κ is set).
inline std::function<double(const Eigen::Vector3f&)> hand_labels(std::vector<SceneBox> obj, double m_on = kMOn, double m_off = kEpsPx)
{
    return [obj, m_on, m_off](const Eigen::Vector3f& p) { return on_object(obj, p) ? m_on : m_off; };
}

// The select cylinder about `sel` (its footprint radius + margin), z ∈ [0, H + margin] (floor-anchored body).
inline rc::lidar_select::Box select_box(const SceneBox& sel, float margin)
{
    return {0.5f * std::hypot(sel.w, sel.h) + margin, 0.0f, sel.H + margin};
}

// Stage `sweep` (first hits from `origin`) through select_rays_beam with the select box about `sel`. label(p) = m, or
// NaN ⇒ no camera information (not staged). κ: set to `kappa` (≥ 0) or, with kappa < 0, select_rays_beam's own κ.
// The fixtures' room: the floor and every scene wall (a vertical segment) are known room surfaces.
inline rc::ai::beam::RoomSurfaces room_of(std::span<const SceneWall> walls, bool floor = true)
{
    rc::ai::beam::RoomSurfaces r;
    r.floor = floor;
    for (const auto& w : walls) r.walls.emplace_back(w.a.cast<double>(), w.b.cast<double>());
    return r;
}

inline rc::ai::beam::BeamRays stage_beam(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& sweep,
                                         const SceneBox& sel, float margin, const std::function<double(const Eigen::Vector3f&)>& label,
                                         double kappa = 1.0, double sigma_s = 0.01, double rho_max = 6.0,
                                         rc::lidar_select::BeamStageStats* stats = nullptr,
                                         const rc::ai::beam::RoomSurfaces* room = nullptr)
{
    rc::ai::beam::BeamRays out;
    const auto lab = [&](const Eigen::Vector3f& p) -> std::optional<std::pair<double, float>>
    {
        const double m = label(p);
        if (std::isnan(m)) return std::nullopt;
        return std::pair<double, float>{m, static_cast<float>(m)};
    };
    rc::lidar_select::select_rays_beam(sweep, origin, sel.c, select_box(sel, margin), rc::lidar_select::Weights{}, rho_max, lab,
                                       out, stats);
    out.sigma_s = sigma_s;
    if (kappa >= 0.0) out.kappa = kappa;
    if (room) rc::ai::beam::stage_room_foreign_density(out, *room, *rc::ai::beam::StudentT::make(2500.0, 0.05));   // = test_beam_config().st
    return out;
}

// Ray-cast the scene from `origin` with the Helios pattern (or the given rings / azimuth step) and stage it.
inline rc::ai::beam::BeamRays helios_beam(const Eigen::Vector3f& origin, std::span<const SceneBox> boxes, std::span<const SceneWall> walls,
                                          const SceneBox& sel, float margin, const std::function<double(const Eigen::Vector3f&)>& label,
                                          double kappa = 1.0, double sigma_s = 0.01)
{
    const auto rings = helios_ring_elev_deg();
    const auto sweep = raycast_scene(origin, boxes, walls, true, kHeliosAzStepDeg, rings, 12.0f);
    const auto room = room_of(walls);
    return stage_beam(origin, sweep, sel, margin, label, kappa, sigma_s, 6.0, nullptr, &room);
}
}  // namespace rc::test
