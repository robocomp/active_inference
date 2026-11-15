#pragma once
/*
 * ltom_query_dsr.h — what ltom_agent's memory graph (domain 5) says about objects that are no longer on the live graph.
 * root -> room_<key> -> one `object` node per retained object (ltom_agent/src/memory_graph.h). Read on demand, on the
 * main thread. The node's `state` is ltom's own comparison of the object's uid against the live graph, so an object
 * that is there now is left to the live tools and never reported twice.
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include <string>

#include <dsr/api/dsr_api.h>

#include "object_facts.h"
#include "question_policy.h"

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

/// LTOM's open questions about moved objects (`ltom_question_<stamp>` nodes; spec 2026-11-15-changed-object), oldest
/// first. Colour, class and name come from the memory node of the object asked about.
inline std::vector<OpenQuestion> load_questions(DSR::DSRGraph& G)
{
    std::vector<OpenQuestion> out;
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (not n.name().starts_with("ltom_question_")) continue;
        if (G.get_attrib_by_name<question_kind_att>(n).value_or("") != "moved") continue;
        OpenQuestion q;
        q.stamp = G.get_attrib_by_name<question_stamp_att>(n).value_or(0);
        q.memory_uid = G.get_attrib_by_name<question_memory_uid_att>(n).value_or(0);
        q.live_uid = G.get_attrib_by_name<question_live_uid_att>(n).value_or(0);
        q.room = G.get_attrib_by_name<question_room_att>(n).value_or("");
        q.again = G.get_attrib_by_name<question_again_att>(n).value_or(false);
        if (const auto o = G.get_attrib_by_name<question_old_xy_att>(n); o.has_value() and o.value().get().size() >= 2)
            q.old_xy = {o.value().get()[0], o.value().get()[1]};
        if (const auto w = G.get_attrib_by_name<question_new_xy_att>(n); w.has_value() and w.value().get().size() >= 2)
            q.new_xy = {w.value().get()[0], w.value().get()[1]};
        if (q.stamp == 0 or q.memory_uid == 0) continue;
        for (const auto& m : G.get_nodes_by_type("object"))
            if (G.get_attrib_by_name<object_uid_att>(m) == std::optional<std::uint64_t>(q.memory_uid))
            {
                q.memory_name = m.name();
                q.cls = G.get_attrib_by_name<object_subtype_att>(m).value_or("");
                q.colour = G.get_attrib_by_name<color_name_att>(m).value_or("");
                break;
            }
        if (q.cls.empty()) q.cls = q.memory_name.substr(0, q.memory_name.rfind('_'));
        if (q.cls.empty()) q.cls = "object";
        out.push_back(std::move(q));
    }
    std::ranges::sort(out, {}, &OpenQuestion::stamp);
    return out;
}

/// What LTOM learned from the person's answers (`ltom_belief_<key>` nodes): counts and the dated evidence lines.
inline std::vector<BeliefFacts> load_beliefs(DSR::DSRGraph& G)
{
    std::vector<BeliefFacts> out;
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (not n.name().starts_with("ltom_belief_")) continue;
        BeliefFacts b;
        b.key = n.name().substr(std::string_view("ltom_belief_").size());
        b.yes = G.get_attrib_by_name<belief_yes_att>(n).value_or(0);
        b.no = G.get_attrib_by_name<belief_no_att>(n).value_or(0);
        const std::string ev = G.get_attrib_by_name<belief_evidence_att>(n).value_or("");
        for (std::size_t a = 0; a < ev.size();)
        {
            const std::size_t e = ev.find('\n', a);
            const std::string line = ev.substr(a, e == std::string::npos ? std::string::npos : e - a);
            if (not line.empty()) b.evidence.push_back(line);
            if (e == std::string::npos) break;
            a = e + 1;
        }
        out.push_back(std::move(b));
    }
    std::ranges::sort(out, {}, &BeliefFacts::key);
    return out;
}

}   // namespace rc::chat
