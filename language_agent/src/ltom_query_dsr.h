#pragma once
/*
 * ltom_query_dsr.h — what ltom_agent's memory graph (domain 5) says about objects that are no longer on the live graph.
 * root -> room_<key> -> one `object` node per retained object (ltom_agent/src/memory_graph.h). Read on demand, on the
 * main thread. The node's `state` is ltom's own comparison of the object's uid against the live graph, so an object
 * that is there now is left to the live tools and never reported twice.
 */

#include <cstdint>
#include <string>

#include <dsr/api/dsr_api.h>

#include "object_facts.h"

namespace rc::chat
{

inline Remembered load_remembered(DSR::DSRGraph& G, std::int64_t now_ms)
{
    Remembered mem{.now_ms = now_ms};
    const auto rt = G.get_rt_api();
    for (const auto& n : G.get_nodes_by_type("object"))
    {
        const auto state = G.get_attrib_by_name<state_att>(n);
        if (not state.has_value() or state.value() != "remembered") continue;
        RememberedObject o{.name = n.name()};
        if (const auto st = G.get_attrib_by_name<object_subtype_att>(n); st.has_value()) o.subtype = st.value();
        if (const auto ts = G.get_attrib_by_name<timestamp_alivetime_att>(n); ts.has_value())
            o.last_seen_ms = static_cast<std::int64_t>(ts.value());
        if (const auto parent = G.get_attrib_by_name<parent_att>(n); parent.has_value())
            if (const auto room = G.get_node(parent.value()); room.has_value() and room->name() != "room_unknown")
            {
                o.room = room->name();
                if (const auto t = rt->get_translation(*room, n.id()); t.has_value())
                {
                    o.x = t->x();
                    o.y = t->y();
                }
            }
        mem.objects.push_back(std::move(o));
    }
    return mem;
}

}   // namespace rc::chat
