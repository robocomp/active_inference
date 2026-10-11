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
 * It changes nothing when a real room exists, and it refuses two protos exactly as rule 2 refuses two rooms.
 * Rule 2a (2026-10-11): after a door crossing with no ltsm_agent, the room the robot's RT is still LIVE to.
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
    /// The robot's RT edge to this room: none, LIVE (no `valid` attribute, or valid=true), or RETIRED (valid=false —
    /// room_concept stamps it once, at a door crossing, and never writes that edge again).
    enum class Rt : std::uint8_t { None, Live, Retired } rt = Rt::None;
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
    // Rule 2a (2026-10-11, door crossing without ltsm_agent): no `current` edge, and room_concept has RETIRED the
    // robot's RT to some room — the robot left it through a door. The room is then the unique room the robot still
    // has a LIVE RT edge to: where room_concept is publishing the pose now (the new room, proto or promoted). This
    // says WHERE the robot is, not WHICH room that is: ltsm_agent's `current` edge (rule 1) still overrides it, and
    // identity stays ltsm_agent's. Without any retired RT nothing here changes, so every earlier case is untouched.
    if (cur.count == 0)
    {
        int n_retired = 0, n_live = 0;
        std::uint64_t live = 0;
        for (const auto& r : rooms)
        {
            if (r.rt == RoomFact::Rt::Retired) ++n_retired;
            else if (r.rt == RoomFact::Rt::Live) { ++n_live; live = r.id; }
        }
        if (n_retired > 0 and n_live == 1)
            return live;
    }
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
