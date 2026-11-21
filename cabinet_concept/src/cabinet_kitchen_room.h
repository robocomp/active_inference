/*
 * cabinet_kitchen_room.h — the kitchen model belongs to ONE room.
 *
 * The (wall,tier) cell table (rc::KitchenManager) is built over the walls of the room the agent works in; every
 * cell's wall chart and run belief is in THAT room's frame. The working room follows the robot
 * (rc::room::follow_current_room), so when the robot crosses a door the model is LEFT BEHIND exactly like a classic
 * instance whose `parent` names the old room — the same rule (rc::room::left_behind), the same consequence: its nodes
 * are deleted, ltom keeps what it remembers, and the model is rebuilt over the new room's walls.
 *
 * Live 2026-11-20 (crossing/c3a_1): the table was built once over room_1 and never re-bound, while the publisher used
 * the FOLLOWED room as parent, so `insert_or_assign_edge_RT` moved room_1's runs under room_2 (and one under room_3),
 * and room_concept could not reap the archived rooms. See docs/cabinet-exit-release-diagnosis.md.
 *
 * Pure rules, no DSR / Qt: the agent supplies the deleter and the reset. Tested in tests/kitchen_room_test.cpp.
 */
#pragma once

#include <cstdint>
#include <optional>

#include "../../common/room_resolve/room_resolve_rule.h"   // rc::room::left_behind

namespace rc::kitchen_room
{

/// 0 is "no room" throughout this agent; the shared rule speaks optional.
inline std::optional<std::uint64_t> as_room(std::uint64_t id)
{
    return id == 0 ? std::nullopt : std::optional<std::uint64_t>{id};
}

/// The room a kitchen run may be published under: the room the model was built over, and only while the agent works
/// there. nullopt ⇒ publish nothing (not built yet, or built for a room the robot has left).
inline std::optional<std::uint64_t> publish_room(std::uint64_t bound, std::uint64_t working)
{
    return (bound != 0 and bound == working) ? as_room(bound) : std::nullopt;
}

/// Release the model when the robot has left the room it was built over: `del(cell, node_id)` for every published run,
/// clear the cell → node map, `reset()` the model, unbind. Returns how many nodes were deleted, or nullopt when the
/// model is not left behind (nothing done). A working room of 0 (nobody has said) is never a release.
template <class Nodes, class Delete, class Reset>
std::optional<int> release_if_left_behind(std::uint64_t& bound, std::uint64_t working, Nodes& nodes,
                                          Delete&& del, Reset&& reset)
{
    if (not rc::room::left_behind(as_room(bound), working))
        return std::nullopt;
    int n = 0;
    for (const auto& [cell, id] : nodes)
    {
        del(cell, id);
        ++n;
    }
    nodes.clear();
    reset();
    bound = 0;
    return n;
}

}   // namespace rc::kitchen_room
