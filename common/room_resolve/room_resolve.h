/*
 * common/room_resolve/room_resolve.h — WHICH ROOM IS "THE" ROOM. SHARED, header-only, main-thread-agnostic
 * (graph reads only; DSRGraph serialises them).
 *
 * ★WHY THIS EXISTS — `get_nodes_by_type("room").front()` IS A COIN TOSS ONCE TWO ROOMS EXIST.
 *   cortex keeps nodes per type in an `unordered_set<uint64_t>` (dsr_api.h, `nodeType`), so the order of
 *   get_nodes_by_type is hash order, not birth order. With ONE room that never mattered, and 25 call sites
 *   across the fleet were written as `.front()`. The moment a second room exists — a PROTO-ROOM born on a
 *   door crossing, or both rooms of a handover — every one of them may silently anchor to the wrong room:
 *   births parented to it, RT read in its frame, its (empty) polygon used as the containment prior.
 *   Nothing crashes; the agent just believes it is somewhere else.
 *
 * THE RULE (verified against the agreed split 2026-09-14: ltsm_agent decides room IDENTITY, room_concept
 * decides GEOMETRY):
 *   1. the robot --[current]--> room edge, owned by ltsm_agent, when exactly ONE such edge exists;
 *   2. otherwise the unique room that is NOT a proto-room (a single-room fleet with no ltsm running —
 *      exactly the old behaviour, now deterministic);
 *   3. otherwise NOTHING. Two `current` edges, or two non-proto rooms and no `current`, is ambiguous, and
 *      guessing is worse than waiting.
 *   ★nullopt means "nobody has told me" — never "released". A caller that lets go of a room on nullopt
 *   would release every room on the first cycle of a fleet that runs no ltsm_agent.
 *
 * PROTO-ROOM: a `room` node carrying a `proto` SELF-edge (room→room). It is room_concept's provisional
 * room for space the current room does not explain; its identity is still unresolved, so it is never
 * "the" room for a consumer. Promotion removes the edge and keeps the node id.
 *
 * PROTO MIRROR: an object node (e.g. door_concept's entry mirror `door_N`) whose `parent` attribute is a
 * proto-room. It references a physical thing whose single belief lives in the current room; every loop
 * that fits, merges, discounts or collides such objects must skip it or it counts that thing twice.
 */

#pragma once

#include <cstdint>
#include <optional>

#include <dsr/api/dsr_api.h>

namespace rc::room
{

/// True when `room_id` is a room node carrying the `proto` self-edge.
inline bool is_proto(DSR::DSRGraph& G, std::uint64_t room_id)
{
    return G.get_edge(room_id, room_id, "proto").has_value();
}

/// The room ltsm_agent's `current` edge points at, if exactly one such edge exists. Read by EDGE TYPE,
/// not by walking from a robot node named per platform; the destination's TYPE is the check.
inline std::optional<std::uint64_t> current_edge_room(DSR::DSRGraph& G)
{
    std::optional<std::uint64_t> found;
    int n = 0;
    for (const auto& e : G.get_edges_by_type("current"))
    {
        const auto dst = G.get_node(e.to());
        if (not dst.has_value() or dst->type() != "room")
            continue;
        ++n;
        found = e.to();
    }
    if (n > 1)
        return std::nullopt;   // mid-write or two writers — ambiguous, report nothing
    return found;
}

/// "The" room, per the rule in the header comment. nullopt = unknown, NEVER a release.
inline std::optional<std::uint64_t> current_room(DSR::DSRGraph& G)
{
    if (const auto c = current_edge_room(G); c.has_value())
        return c;
    std::optional<std::uint64_t> found;
    int n = 0;
    for (const auto& r : G.get_nodes_by_type("room"))
    {
        if (is_proto(G, r.id()))
            continue;
        ++n;
        found = r.id();
    }
    if (n != 1)
        return std::nullopt;
    return found;
}

/// Same, returning the node (convenience for call sites that read attributes off it).
inline std::optional<DSR::Node> current_room_node(DSR::DSRGraph& G)
{
    const auto id = current_room(G);
    if (not id.has_value())
        return std::nullopt;
    return G.get_node(*id);
}

/// True when `node`'s `parent` attribute names a proto-room — a mirror of something whose belief lives in
/// another room (see header). Nodes with no `parent` attribute are never mirrors.
inline bool is_proto_mirror(DSR::DSRGraph& G, const DSR::Node& node)
{
    const auto p = G.get_attrib_by_name<parent_att>(node);
    if (not p.has_value())
        return false;
    const auto pn = G.get_node(p.value());
    return pn.has_value() and pn->type() == "room" and is_proto(G, pn->id());
}

}   // namespace rc::room
