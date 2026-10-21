#pragma once
// The mission predicate explored(room) — Task 5, 2026-10-21.
// room_concept writes `room_explored_p` on its room node: P(continuing to explore no longer pays), the Luce choice
// between stopping (worth the explorer's price of a trip) and the best look still left, combined with what memory
// restored for a recalled room (room_concept/src/room_explored.h). The mission states the boundary (">= 0.9" = stop
// when stopping is preferred 9:1); this only reads the belief.
// ABSENT (never assessed) or no such room ⇒ nullopt ⇒ the interpreter's `until` never holds — "unknown" is not
// "unexplored" and is not "explored" either.
#include <dsr/api/dsr_api.h>

#include <optional>
#include <string>

namespace rc::mission
{
[[nodiscard]] inline std::optional<double> explored_belief(DSR::DSRGraph &G, const std::string &room)
{
    const auto node = G.get_node(room);
    if (not node.has_value()) return std::nullopt;
    const auto p = G.get_attrib_by_name<room_explored_p_att>(node.value());
    return p.has_value() ? std::optional<double>(p.value()) : std::nullopt;
}
}   // namespace rc::mission
