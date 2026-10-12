#pragma once
/*
 * common/world_query/door_query.h — WHICH DOORS EXIST, WHICH ROOM EACH BELONGS TO, AND WHICH IS NEAREST.
 * SHARED, header-only, graph reads only (DSRGraph serialises them; safe from any thread).
 *
 * Moved out of controller_door_passage.h (2026-10-12) so the controller and chat_agent answer "which door is
 * 'the door'?" with ONE piece of code: the chat agent validates a `cross door` request with exactly the predicate
 * the controller will execute it with, so the two can never disagree about which door that is.
 *
 *   list_doors(G)                           door_concept's beliefs: type "object" named door_* (its own sweep's
 *                                           predicate, door_concept specificworker.cpp), sorted by name.
 *   room_of(G, id)                          the room a node's RT chain hangs from (≤ 8 hops up), 0 if none.
 *   doors_of_room(G, room_id)               list_doors filtered to that room.
 *   nearest_door(G, inner, room, robot)     the door of THIS room nearest the robot — what "cross door" means.
 *                                           Doors of a room left behind resolve in this frame too (through root),
 *                                           so the room_of filter is what keeps them from being picked.
 *
 * ★ Every get_transformation_matrix optional is checked (CLAUDE.md): a missing node/edge in the chain returns
 *   nullopt and the door is skipped, never dereferenced.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc::world
{

[[nodiscard]] inline std::vector<std::string> list_doors(DSR::DSRGraph& G)
{
    std::vector<std::string> names;
    for (const auto& n : G.get_nodes_by_type("object"))
        if (n.name().starts_with("door")) names.push_back(n.name());
    std::ranges::sort(names);
    return names;
}

/// The room node id a node's RT chain hangs from, 0 when the chain does not reach a room within 8 hops.
[[nodiscard]] inline std::uint64_t room_of(DSR::DSRGraph& G, std::uint64_t id)
{
    for (int hop = 0; hop < 8 and id != 0; ++hop)
    {
        const auto n = G.get_node(id);
        if (not n.has_value()) return 0;
        if (n->type() == "room") return id;
        std::uint64_t up = 0;
        for (const auto& e : G.get_edges_to_id(id))
            if (e.type() == "RT") { up = e.from(); break; }
        id = up;
    }
    return 0;
}

[[nodiscard]] inline std::vector<std::string> doors_of_room(DSR::DSRGraph& G, std::uint64_t room_id)
{
    std::vector<std::string> out;
    for (const auto& name : list_doors(G))
        if (const auto d = G.get_node(name); d.has_value() and room_of(G, d->id()) == room_id)
            out.push_back(name);
    return out;
}

[[nodiscard]] inline std::optional<std::string> nearest_door(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                                             const std::string& room, const std::string& robot)
{
    const auto room_node = G.get_node(room);
    if (not room_node.has_value()) return std::nullopt;
    const auto rb = inner.get_transformation_matrix(room, robot);
    if (not rb.has_value()) return std::nullopt;
    std::optional<std::string> best;
    double best_d = std::numeric_limits<double>::max();
    for (const auto& name : list_doors(G))
    {
        const auto d = G.get_node(name);
        if (not d.has_value() or room_of(G, d->id()) != room_node->id()) continue;
        const auto T = inner.get_transformation_matrix(room, name);
        if (not T.has_value()) continue;
        if (const double dist = (T->translation() - rb->translation()).head<2>().norm(); dist < best_d)
        { best_d = dist; best = name; }
    }
    return best;
}

}   // namespace rc::world
