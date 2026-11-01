#pragma once
/*
 * robot_body_check.h — the Shadow mesh-vs-compiled-hull cross-check, ONE copy (controller and chat_agent both call it).
 * The robot node's `path` is shadow_footprint.obj (shadow.stl + the two drive wheels). An OLD mesh (shadow.obj: no wheels,
 * |x|max 0.242) would shrink the body 4 cm per side, so the mesh is accepted only if it agrees with the compiled hull:
 * |x|max and |y|max within 1 mm, area within 1 %. Header-only, no DSR/Qt.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <string>

#include "../robot_footprint/robot_footprint.h"

namespace rc
{

/// nullopt = the mesh agrees with RobotFootprint::shadow(); else the reason it is rejected (the compiled hull stays).
[[nodiscard]] inline std::optional<std::string> shadow_mesh_disagreement(const RobotFootprint::MeshReport &m, const std::string &path)
{
    const RobotFootprint ref = RobotFootprint::shadow();
    float rx = 0.f, ry = 0.f;
    for (const auto &v : ref.polygon()) { rx = std::max(rx, std::abs(v.x())); ry = std::max(ry, std::abs(v.y())); }
    float ref_area = 0.f;
    {
        const auto &q = ref.polygon();
        for (std::size_t i = 0, n = q.size(); i < n; ++i)
            ref_area += 0.5f * (q[i].x() * q[(i + 1) % n].y() - q[(i + 1) % n].x() * q[i].y());
    }
    if (std::abs(m.x_max - rx) > 1e-3f or std::abs(m.y_max - ry) > 1e-3f or std::abs(m.area_m2 - ref_area) > 0.01f * ref_area)
        return std::format("the mesh '{}' disagrees with the compiled Shadow hull (|x|max {:.4f} vs {:.4f}, |y|max {:.4f} "
                           "vs {:.4f}, area {:.4f} vs {:.4f}) — an old mesh without the two drive wheels?",
                           path, m.x_max, rx, m.y_max, ry, m.area_m2, ref_area);
    return std::nullopt;
}

}   // namespace rc
