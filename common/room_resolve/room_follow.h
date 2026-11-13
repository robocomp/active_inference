/*
 * common/room_resolve/room_follow.h — the GRAPH side of room-following, shared by every countable concept agent.
 *
 *   read_outline    the room's delimiting polygon as 2-D points ({} when absent or inconsistent);
 *   drop_left_behind  delete this agent's `object` nodes whose `parent` is a room other than the one the agent works in.
 *
 * The working frame is an INPUT: this header never resolves the room (spec §8). The pure rules live in
 * room_resolve_rule.h / room_containment.h. Typed attribute access only.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <print>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <dsr/api/dsr_api.h>

#include "room_resolve_rule.h"

namespace rc::room
{

/// delimiting_polygon_x/y of `room_id` as points; empty when the node or either attribute is absent, or lengths differ.
inline std::vector<Eigen::Vector2f> read_outline(DSR::DSRGraph& G, std::uint64_t room_id)
{
    const auto room = G.get_node(room_id);
    if (not room.has_value())
        return {};
    const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value())
        return {};
    const auto& xs = px->get();
    const auto& ys = py->get();
    if (xs.size() != ys.size())
        return {};
    std::vector<Eigen::Vector2f> out;
    out.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        out.emplace_back(xs[i], ys[i]);
    return out;
}

/// The ROOM an object's `parent` places it in. A parent may be a supporting object rather than the room (a bottle's
/// parent is the table it stands on), so walk `parent` up until a `room` node. A parent that no longer exists (a room
/// recreated with a new id, or a supporting table dropped as left behind) is returned as-is: it names no live room,
/// so `left_behind` holds and the object is dropped, never fitted in a dead frame. nullopt ⇔ no parent at all.
inline std::optional<std::uint64_t> room_of(DSR::DSRGraph& G, std::optional<std::uint64_t> parent)
{
    for (int depth = 0; parent.has_value() and depth < 8; ++depth)   // 8 = a cycle guard, not a tuning value
    {
        const auto node = G.get_node(*parent);
        if (not node.has_value() or node->type() == "room")
            return parent;
        const auto up = G.get_attrib_by_name<parent_att>(node.value());
        if (not up.has_value())
            return parent;
        parent = *up;
    }
    return parent;
}

/// Delete every `object` node named `name_prefix*` that was left behind in a room other than `working_frame`.
/// `forget(node_id)` lets the agent tear down its own per-node state first. Returns how many were dropped.
template <class Forget>
int drop_left_behind(DSR::DSRGraph& G, std::string_view name_prefix, std::uint64_t working_frame, Forget&& forget)
{
    int dropped = 0;
    for (const auto& node : G.get_nodes_by_type("object"))
    {
        if (not node.name().starts_with(name_prefix))
            continue;
        if (not left_behind(room_of(G, G.get_attrib_by_name<parent_att>(node)), working_frame))
            continue;
        const auto id = node.id();
        forget(id);
        std::print("{}: [room] '{}' left behind in another room, dropped\n", name_prefix, node.name());
        G.delete_node(id);
        ++dropped;
    }
    return dropped;
}

}   // namespace rc::room
