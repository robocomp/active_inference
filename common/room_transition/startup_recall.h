#pragma once
/*
 * startup_recall.h — the WARM START's start-up recall (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md).
 *
 * At a crossing ltsm posts `recall_<room>` (metaconcept, object_subtype "room_recall") as an RT child of the live door the
 * room lies behind (ltsm_agent/src/room_recall.h). With [WarmStart] Room set, it posts the same node for the room the robot
 * STARTS in, hung from the ROBOT node, with no door transform and no `recall_late`. One definition for the three agents that
 * meet it: ltsm posts it, room_concept (MapMode = "ltsm") adopts its outline as the given layout, door_concept lets it
 * announce the first room (its remembered doors restored).
 */
#include <cmath>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace rc::room_transition
{
inline constexpr std::string_view kRecallSubtype = "room_recall";

/// A recall node is a START-UP recall when it hangs from a robot node and is not a late (recognition) recall. A crossing
/// recall hangs from a door; a late one carries `recall_late`.
[[nodiscard]] inline bool is_startup_recall(std::string_view subtype, std::string_view parent_type, bool late)
{
    return subtype == kRecallSubtype and parent_type == "robot" and not late;
}

/// The recall's outline (`delimiting_polygon_x/y`, the room's own frame). nullopt unless x and y pair up, there are at least
/// three vertices and every value is finite.
[[nodiscard]] inline std::optional<std::vector<Eigen::Vector2f>> recall_outline(std::span<const float> xs, std::span<const float> ys)
{
    if (xs.size() != ys.size() or xs.size() < 3) return std::nullopt;
    std::vector<Eigen::Vector2f> out;
    out.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        if (not std::isfinite(xs[i]) or not std::isfinite(ys[i])) return std::nullopt;
        out.emplace_back(xs[i], ys[i]);
    }
    return out;
}
}  // namespace rc::room_transition
