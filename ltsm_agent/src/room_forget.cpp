/*
 * room_forget.cpp — see room_forget.h.
 */
#include "room_forget.h"

#include <QDebug>

#include <algorithm>
#include <deque>
#include <format>
#include <unordered_map>
#include <unordered_set>

namespace ltsm
{

std::optional<ForgetPlan> plan_forget(DSR::DSRGraph &mem, const std::string &room)
{
    const auto r = mem.get_node(room);
    if (not r.has_value() or r->type() != "room") return std::nullopt;

    // parent -> children by the `parent` attribute: catches a node hung below the room whose RT edge was never
    // written (or was lost), which an RT-only walk would leave dangling.
    const auto nodes = mem.get_nodes();
    std::unordered_multimap<std::uint64_t, std::uint64_t> by_parent;
    for (const auto &n : nodes)
        if (const auto p = mem.get_attrib_by_name<parent_att>(n); p.has_value() and p.value() != n.id())
            by_parent.emplace(p.value(), n.id());

    std::unordered_set<std::uint64_t> in{r->id()};
    std::deque<std::uint64_t> frontier{r->id()};
    const auto take = [&](std::uint64_t id)
    {
        const auto c = mem.get_node(id);
        if (not c.has_value() or c->type() == "root" or c->type() == "room") return;   // never another room
        if (in.insert(id).second) frontier.push_back(id);
    };
    while (not frontier.empty())
    {
        const auto id = frontier.front();
        frontier.pop_front();
        if (const auto n = mem.get_node(id); n.has_value())
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n.value(), "RT")) take(e.to());
        for (auto [it, end] = by_parent.equal_range(id); it != end; ++it) take(it->second);
    }

    ForgetPlan plan{.room = room, .room_id = r->id()};
    for (const auto &n : nodes)
    {
        if (n.type() != "metaconcept" or mem.get_attrib_by_name<object_subtype_att>(n).value_or("") != "passage")
            continue;
        const auto faces = DSR::DSRGraph::get_node_edges_by_type(n, "match");
        if (std::ranges::any_of(faces, [&](const auto &e) { return in.contains(e.to()); }))
            plan.passages.push_back(n.id());
    }

    std::vector<std::pair<std::uint64_t, int>> levelled;
    for (const auto id : in)
        if (id != r->id())
            if (const auto n = mem.get_node(id); n.has_value())
                levelled.emplace_back(id, mem.get_node_level(n.value()).value_or(0));
    std::ranges::sort(levelled, [](const auto &a, const auto &b) { return a.second > b.second; });
    for (const auto &[id, _] : levelled) plan.subtree.push_back(id);
    return plan;
}

int forget(DSR::DSRGraph &mem, const ForgetPlan &plan)
{
    int removed = 0;
    const auto drop = [&](std::uint64_t id, std::string_view what)
    {
        const auto n = mem.get_node(id);
        if (not n.has_value()) return;
        const std::string name = n->name();
        if (mem.delete_node(id))
        {
            ++removed;
            qInfo().noquote() << QString::fromStdString(std::format("[forget] removed {} {} ({})", what, name, plan.room));
        }
        else
            qWarning().noquote() << QString::fromStdString(std::format("[forget] could NOT remove {} {}", what, name));
    };
    for (const auto id : plan.passages) drop(id, "passage");
    for (const auto id : plan.subtree)  drop(id, "node");
    drop(plan.room_id, "room");
    return removed;
}

}   // namespace ltsm
