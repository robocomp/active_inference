/*
 * common/room_resolve/room_resolve_rule.h — THE RULE behind rc::room::current_room, as a pure function.
 *
 * Separated from room_resolve.h (which reads the DSR graph) so the rule itself can be pinned by a test
 * without a graph, a DDS participant or Qt: room_resolve_test.cpp. room_resolve.h gathers the facts from
 * the graph and calls this; nothing else decides which room is "the" room.
 *
 * See room_resolve.h for the rule's rationale. Rule 3 (2026-10-09, start-up proto-room plan): with no
 * `current` edge and no non-proto room at all, a UNIQUE proto-room is the room. That is room_concept's
 * start-up proto in an estimation run: the robot's first room, whose identity nobody disputes, published
 * from the first second so the controller and residual can act while its layout is still being learnt.
 * It changes nothing when a real room exists — in a door crossing the old real room keeps winning by
 * rule 2 until ltsm_agent moves `current` — and it refuses two protos exactly as rule 2 refuses two rooms.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace rc::room
{

/// One `room` node as the rule sees it.
struct RoomFact
{
    std::uint64_t id = 0;
    bool proto = false;   ///< carries the `proto` self-edge
};

/// The robot→room `current` edges: how many point at a room node, and (when exactly one) which.
struct CurrentEdges
{
    int count = 0;
    std::uint64_t room = 0;   ///< meaningful only when count == 1
};

/// "The" room. nullopt = unknown, NEVER a release (see room_resolve.h).
inline std::optional<std::uint64_t> resolve_current_room(const CurrentEdges& cur, std::span<const RoomFact> rooms)
{
    // Rule 1: ltsm_agent's `current` edge, when exactly one exists.
    if (cur.count == 1)
        return cur.room;
    // Rule 2: the unique room that is not a proto-room. (Two `current` edges fall through to here; that is
    // the pre-2026-10-09 behaviour and it is kept as is.)
    std::optional<std::uint64_t> real, proto;
    int n_real = 0, n_proto = 0;
    for (const auto& r : rooms)
    {
        if (r.proto) { ++n_proto; proto = r.id; }
        else         { ++n_real;  real  = r.id; }
    }
    if (n_real == 1)
        return real;
    // Rule 3: nobody has said anything (no `current` edge AT ALL) and there is no surveyed room — a unique
    // proto-room is the room. Any non-proto room, or any `current` edge, disables it.
    if (n_real == 0 and cur.count == 0 and n_proto == 1)
        return proto;
    return std::nullopt;
}

}   // namespace rc::room
