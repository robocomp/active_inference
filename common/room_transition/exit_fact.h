#pragma once
/*
 * exit_fact.h — THE CROSSING FACT of the room-transition protocol (spec 2026-10-29, Revision 3, encoding A).
 *
 * A crossing is an ACTION: the controller drives the body through a hole it perceives in its own frame and measures, there,
 * when the body is through. That measurement is written ONCE, as an `exit` edge, and every other agent acts on it in turn,
 * each on its predecessor's edge — a chain, so no step can run ahead of the one it depends on (Lucas Bonilla's TFM 2024, §5.1/§6.3.2: "the robot → the door
 * it crossed" `exit` edge, written when the cross completes past the door; his LTSM then stores and removes the room left):
 *
 *   link 1  controller    `exit` ROBOT    -> OLD_FACE   room_id = the remembered room predicted behind the face when the
 *                                                       cross STARTED (the recall it resolved the leg with); 0 = new
 *   link 2  room_concept  `exit` OLD_FACE -> NEW_ROOM   on link 1 naming a face of ITS room: the robot entered NEW_ROOM
 *   link 3  door_concept  `exit` NEW_FACE -> OLD_FACE   on link 2: the hole's face in NEW_ROOM, the twin of OLD_FACE
 *   link 4  ltsm          archives OLD_ROOM on link 3 (deleting OLD_FACE: links 1 and 2 die with it)
 *
 * Which link an edge is follows from its endpoints' types (robot→door, door→room, door→door), so the edge type `exit`
 * (cortex dsr_edge_type.h, NON-RT) needs nothing else. Every edge dies with its source node. No counters: a fact about
 * leaving room R cannot be acted on twice, the robot is no longer in R once it has been.
 *
 * Header-only, graph API only (thread-safe), no Qt. Model: docs/superpowers/specs/tla/RoomTransitionExit.tla.
 */
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <dsr/api/dsr_api.h>

namespace rc::exit_fact
{
inline bool is_type(DSR::DSRGraph& G, std::uint64_t id, const char* type)
{
    const auto n = G.get_node(id);
    return n.has_value() and n->type() == type;
}
/// A door face: door_concept's generic `object` node carrying object_subtype "door" (door_scene_graph.cpp).
inline bool is_door(DSR::DSRGraph& G, std::uint64_t id)
{
    const auto n = G.get_node(id);
    return n.has_value() and n->type() == "object"
       and G.get_attrib_by_name<object_subtype_att>(n.value()).value_or(std::string{}) == "door";
}

// ── link 1 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
/// The controller: the body is through `face`. Replaces the robot's previous link-1 edge (one crossing at a time).
inline bool write_crossed(DSR::DSRGraph& G, std::uint64_t robot, std::uint64_t face, std::uint64_t predicted_room_id)
{
    if (const auto out = G.get_edges(robot); out.has_value())
    {
        std::vector<std::uint64_t> old;
        for (const auto& [key, e] : out.value())
            if (key.second == "exit" and key.first != face) old.push_back(key.first);
        for (const auto to : old) G.delete_edge(robot, to, "exit");
    }
    auto e = DSR::Edge::create<exit_edge_type>(robot, face);
    G.add_or_modify_attrib_local<room_id_att>(e, predicted_room_id);
    return G.insert_or_assign_edge(e);
}

struct Crossed
{
    std::uint64_t face = 0;               ///< OLD_FACE: the door node the body went through
    std::uint64_t predicted_room_id = 0;  ///< the remembered room behind it (room_id); 0 = a new room
};
/// room_concept: a link-1 edge on one of `faces` (the doors of its room); nullopt = no crossing out of this room.
inline std::optional<Crossed> crossed_through(DSR::DSRGraph& G, const std::vector<std::uint64_t>& faces)
{
    for (const auto f : faces)
        for (const auto& e : G.get_edges_to_id(f))
            if (e.type() == "exit" and is_type(G, e.from(), "robot"))
                return Crossed{.face = f, .predicted_room_id = G.get_attrib_by_name<room_id_att>(e).value_or(0)};
    return std::nullopt;
}
/// Has `face` been crossed (any link-1 edge on it, or it is a link-3 twin)? `cross new` never takes such a door.
inline bool is_crossed_face(DSR::DSRGraph& G, std::uint64_t face)
{
    for (const auto& e : G.get_edges_to_id(face))
        if (e.type() == "exit" and (is_type(G, e.from(), "robot") or is_door(G, e.from()))) return true;
    if (const auto n = G.get_node(face); n.has_value())
        for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(n.value(), "exit"))
            if (is_door(G, e.to())) return true;
    return false;
}

// ── link 2 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
inline bool write_entered(DSR::DSRGraph& G, std::uint64_t old_face, std::uint64_t new_room)
{
    return G.insert_or_assign_edge(DSR::Edge::create<exit_edge_type>(old_face, new_room));
}
/// The face the robot crossed to enter `room` (link 2), while that face lives.
inline std::optional<std::uint64_t> entered_through(DSR::DSRGraph& G, std::uint64_t room)
{
    for (const auto& e : G.get_edges_to_id(room))
        if (e.type() == "exit" and is_door(G, e.from())) return e.from();
    return std::nullopt;
}

// ── link 3 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
inline bool write_twin(DSR::DSRGraph& G, std::uint64_t new_face, std::uint64_t old_face)
{
    return G.insert_or_assign_edge(DSR::Edge::create<exit_edge_type>(new_face, old_face));
}
/// The face `new_face` is the far side of (link 3), while it lives.
inline std::optional<std::uint64_t> twin_of(DSR::DSRGraph& G, std::uint64_t new_face)
{
    const auto n = G.get_node(new_face);
    if (not n.has_value()) return std::nullopt;
    for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(n.value(), "exit"))
        if (is_door(G, e.to())) return e.to();
    return std::nullopt;
}
/// Link 4's guard: is the crossing into `room` complete (its entry face's twin written)?
inline bool twin_written(DSR::DSRGraph& G, std::uint64_t room, const std::vector<std::uint64_t>& faces_of_room)
{
    const auto old = entered_through(G, room);
    if (not old.has_value()) return false;
    for (const auto f : faces_of_room)
        if (twin_of(G, f) == old) return true;
    return false;
}
}  // namespace rc::exit_fact
