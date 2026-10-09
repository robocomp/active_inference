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
 *   3. (2026-10-09) otherwise, ONLY when there is no `current` edge at all AND no non-proto room, a UNIQUE
 *      proto-room — room_concept's START-UP proto in an estimation run (the robot's first room: nobody
 *      disputes its identity, so it is the room from the first second). Never fires beside a real room,
 *      so a door crossing still resolves to the old room until ltsm_agent moves `current`;
 *   4. otherwise NOTHING. Two non-proto rooms and no `current`, or two protos, is ambiguous, and guessing
 *      is worse than waiting. (Two `current` edges abstain from rule 1 and fall through to rule 2 — the
 *      code has always done that; rule 3 is disabled by them.)
 *   The rule itself is the pure function rc::room::resolve_current_room in room_resolve_rule.h, pinned by
 *   room_resolve_test.cpp; this header only gathers the facts from the graph.
 *   ★nullopt means "nobody has told me" — never "released". A caller that lets go of a room on nullopt
 *   would release every room on the first cycle of a fleet that runs no ltsm_agent.
 *
 * PROTO-ROOM: a `room` node carrying a `proto` SELF-edge (room→room). It is room_concept's provisional
 * room for space the current room does not explain; its identity is still unresolved, so it is "the" room
 * for a consumer only through rule 3 (no other room, no `current` edge). Promotion removes the edge and
 * keeps the node id.
 *
 * PROTO MIRROR: an object node (e.g. door_concept's entry mirror `door_N`) whose `parent` attribute is a
 * proto-room. It references a physical thing whose single belief lives in the current room; every loop
 * that fits, merges, discounts or collides such objects must skip it or it counts that thing twice.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <dsr/api/dsr_api.h>

#include "room_resolve_rule.h"   // the rule, as a pure function (pinned by room_resolve_test.cpp)

namespace rc::room
{

/// True when `room_id` is a room node carrying the `proto` self-edge.
inline bool is_proto(DSR::DSRGraph& G, std::uint64_t room_id)
{
    return G.get_edge(room_id, room_id, "proto").has_value();
}

/// The robot→room `current` edges, counted by EDGE TYPE (not by walking from a robot node named per
/// platform); the destination's TYPE is the check.
inline CurrentEdges current_edges(DSR::DSRGraph& G)
{
    CurrentEdges c;
    for (const auto& e : G.get_edges_by_type("current"))
    {
        const auto dst = G.get_node(e.to());
        if (not dst.has_value() or dst->type() != "room")
            continue;
        ++c.count;
        c.room = e.to();
    }
    return c;
}

/// The room ltsm_agent's `current` edge points at, if exactly one such edge exists.
inline std::optional<std::uint64_t> current_edge_room(DSR::DSRGraph& G)
{
    const auto c = current_edges(G);
    if (c.count != 1)
        return std::nullopt;   // none, or mid-write / two writers — ambiguous, report nothing
    return c.room;
}

/// "The" room, per the rule in the header comment. nullopt = unknown, NEVER a release.
inline std::optional<std::uint64_t> current_room(DSR::DSRGraph& G)
{
    const auto cur = current_edges(G);
    if (cur.count == 1)
        return cur.room;   // rule 1 needs no room scan
    std::vector<RoomFact> rooms;
    for (const auto& r : G.get_nodes_by_type("room"))
        rooms.push_back({.id = r.id(), .proto = is_proto(G, r.id())});
    return resolve_current_room(cur, rooms);
}

/// Same, returning the node (convenience for call sites that read attributes off it).
inline std::optional<DSR::Node> current_room_node(DSR::DSRGraph& G)
{
    const auto id = current_room(G);
    if (not id.has_value())
        return std::nullopt;
    return G.get_node(*id);
}

/// The NAME of the node with this id, or "" when it is not in the graph. Cheaper than get_node: it
/// copies a string, not a node with its polygon attributes.
inline std::string name_of(DSR::DSRGraph& G, std::uint64_t id)
{
    return G.get_name_from_id(id).value_or(std::string{});
}

/// ★THE ROOM'S FRAME NAME — what every `inner_eigen->get_transformation_matrix(...)` /
/// `transform(...)` / `transform_point(...)` argument wants. Use this INSTEAD of the literal "room".
///
/// WHY IT IS NOT A CONSTANT ANY MORE. room_concept names its first room `room_1` and every room
/// discovered afterwards `room_<k>` (room_scene_graph.cpp: dsr_create_room_and_reparent for the first,
/// step_proto_room for the rest). A literal "room" in a frame argument therefore names a node that no
/// longer exists, and `get_transformation_matrix` answers that with `nullopt` — SILENTLY, because a
/// missing node and an unresolvable RT chain are the same answer. That is the failure this function
/// exists to make impossible: the frame is resolved from the graph, by TYPE and by the robot's
/// `current` edge, exactly like `current_room`.
///
/// "" when the room is unknown. That is deliberate and NOT a sentinel that needs its own branch: no
/// node is named "", so every transform built on it returns nullopt, which is the same branch the
/// caller already takes when the chain is not resolvable yet. A caller that wants to tell "no room"
/// from "no chain" apart should ask `current_room` and say so.
///
/// Backward compatible with a stale graph: a leftover node still named plain "room" is resolved by
/// this function too, because nothing here looks at the name to decide WHICH room.
inline std::string current_room_frame(DSR::DSRGraph& G)
{
    const auto id = current_room(G);
    if (not id.has_value())
        return {};
    return name_of(G, *id);
}

/// Resolve a CONFIGURED frame name. The token "room" is the fleet's word for "whichever room is
/// current" — it is what every config file, every struct default and every comment already says — and
/// it is resolved from the graph HERE, at use time. Anything else is a literal node name and passes
/// through untouched.
///
/// ★RESOLVE IT EVERY TIME, never latch it. Which room is current changes (a door crossing hands over to
/// a proto-room), and a name latched at construction is a dead frame that cortex answers with a silent
/// nullopt — the same write-once node-name memo that has bitten this fleet before.
inline std::string resolve_frame(DSR::DSRGraph& G, const std::string& configured)
{
    return configured == "room" ? current_room_frame(G) : configured;
}

/// True when `node`'s `parent` attribute names a proto-room — a mirror of something whose belief lives in
/// another room (see header). Nodes with no `parent` attribute are never mirrors.
/// ★A proto-room that IS the current room (rule 3: the start-up proto of an estimation run) has no other
/// room for its objects to mirror: everything born in it is its own, single belief. Without this exception
/// every object a concept agent births under the start-up proto would be skipped by its own fitting loop.
inline bool is_proto_mirror(DSR::DSRGraph& G, const DSR::Node& node)
{
    const auto p = G.get_attrib_by_name<parent_att>(node);
    if (not p.has_value())
        return false;
    const auto pn = G.get_node(p.value());
    if (not (pn.has_value() and pn->type() == "room" and is_proto(G, pn->id())))
        return false;
    return current_room(G) != std::optional<std::uint64_t>{pn->id()};
}

}   // namespace rc::room
