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
        if (not state.has_value() or state.value() != "remembered" or is_human_node(n.name())) continue;
        RememberedObject o{.name = n.name()};
        o.uid = G.get_attrib_by_name<object_uid_att>(n).value_or(0);
        if (const auto st = G.get_attrib_by_name<object_subtype_att>(n); st.has_value()) o.subtype = st.value();
        if (const auto ts = G.get_attrib_by_name<timestamp_alivetime_att>(n); ts.has_value())
            o.last_seen_ms = static_cast<std::int64_t>(ts.value());
        if (const auto c = G.get_attrib_by_name<color_name_att>(n); c.has_value())   // ltom_agent keeps the latest measured
        {
            o.color_name = c.value();
            o.color_prob = G.get_attrib_by_name<color_prob_att>(n).value_or(0.f);
        }
        if (const auto pe = G.get_attrib_by_name<obj_exist_prob_att>(n); pe.has_value())   // absent = not stated
            o.exist_prob = pe.value();
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

/// The human at the chat as ltom_agent shows them (human_1: person_name, person_role, `owns` edges to object memory
/// nodes, the answer to the chat's last claim). Empty when there is no human_1 (ltom not running, or an older ltom).
inline HumanFacts load_human(DSR::DSRGraph& G)
{
    HumanFacts h;
    const auto n = G.get_node("human_1");
    if (not n) return h;
    if (const auto v = G.get_attrib_by_name<person_name_att>(*n); v.has_value()) h.name = v.value().get();
    if (const auto v = G.get_attrib_by_name<person_role_att>(*n); v.has_value()) h.role = v.value().get();
    for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(*n, "owns"))
        if (const auto o = G.get_node(e.to()); o.has_value())
            if (const auto uid = G.get_attrib_by_name<object_uid_att>(*o); uid.has_value()) h.owned_uids.insert(uid.value());
    if (const auto s = G.get_attrib_by_name<human_claim_answer_stamp_att>(*n); s.has_value()) h.answer_stamp = s.value();
    h.answer = G.get_attrib_by_name<human_claim_answer_att>(*n).value_or("");
    return h;
}

}   // namespace rc::chat
