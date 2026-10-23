#include "topology_publish.h"
#include "room_eviction.h"      // ltsm::is_door
#include "topology_digest.h"

#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <algorithm>
#include <optional>

namespace ltsm
{
namespace
{
/// Memory names a node r<idx>_<live name>; the live name is what comes after the first '_'.
std::string live_name(const std::string& mem)
{
    const auto us = mem.find('_');
    if (mem.size() > 1 and mem.starts_with('r') and us != std::string::npos
        and std::all_of(mem.begin() + 1, mem.begin() + static_cast<std::ptrdiff_t>(us), [](unsigned char c) { return std::isdigit(c); }))
        return mem.substr(us + 1);
    return mem;
}

std::optional<DSR::Node> owning_room(DSR::DSRGraph& g, const DSR::Node& n)
{
    auto cur = std::optional<DSR::Node>(n);
    for (int hops = 0; cur.has_value() and hops < 16; ++hops)
    {
        if (cur->type() == "room") return cur;
        const auto p = g.get_attrib_by_name<parent_att>(cur.value());
        if (not p.has_value() or p.value() == cur->id()) return std::nullopt;
        cur = g.get_node(p.value());
    }
    return std::nullopt;
}
}   // namespace

std::string build_topology_json(DSR::DSRGraph& memory, DSR::DSRGraph& live)
{
    std::vector<rc::ltsm::TopoRoom> rooms;
    for (const auto& r : memory.get_nodes_by_type("room"))
    {
        const auto name = live_name(r.name());
        if (std::ranges::any_of(rooms, [&](const auto& x) { return x.name == name; })) continue;
        rooms.push_back({.name = name, .live = live.get_node(name).has_value()});
    }
    std::ranges::sort(rooms, {}, &rc::ltsm::TopoRoom::name);

    std::vector<rc::ltsm::TopoPassage> passages;
    for (const auto& n : memory.get_nodes_by_type("metaconcept"))
    {
        if (memory.get_attrib_by_name<object_subtype_att>(n).value_or("") != "passage") continue;
        rc::ltsm::TopoPassage p;
        p.id = n.name();
        p.crossings = memory.get_attrib_by_name<passage_crossings_att>(n).value_or(0);
        p.pass_alpha = memory.get_attrib_by_name<passage_pass_alpha_att>(n).value_or(1.f);
        p.pass_beta = memory.get_attrib_by_name<passage_pass_beta_att>(n).value_or(1.f);
        p.open_alpha = memory.get_attrib_by_name<passage_open_alpha_att>(n).value_or(1.f);
        p.open_beta = memory.get_attrib_by_name<passage_open_beta_att>(n).value_or(1.f);
        int faces = 0;
        for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(n, "match"))
        {
            const auto face = memory.get_node(e.to());
            if (not face.has_value() or not is_door(memory, face.value())) continue;
            const auto room = owning_room(memory, face.value());
            if (not room.has_value()) continue;
            const auto rn = live_name(room->name());
            const auto fn = live_name(face->name());
            auto& slot_room = faces == 0 ? p.room_a : p.room_b;
            auto& slot_face = faces == 0 ? p.face_a : p.face_b;
            slot_room = rn;
            slot_face = fn;
            // The live door this face is NOW: only when its room is live and holds a door of that name.
            if (not p.live_door.has_value() and live.get_node(rn).has_value())
                if (const auto d = live.get_node(fn); d.has_value() and is_door(live, d.value())) p.live_door = fn;
            if (++faces == 2) break;
        }
        if (faces == 2) passages.push_back(std::move(p));   // a passage with one face joins nothing
    }
    std::ranges::sort(passages, {}, &rc::ltsm::TopoPassage::id);
    return rc::ltsm::encode_topology(rooms, passages);
}

bool TopologyPublisher::refresh()
{
    const auto json = build_topology_json(*mem_, *live_);
    auto node = live_->get_node(kNodeName);
    if (node.has_value() and json == last_) return false;   // memory unchanged since our own write
    if (not node.has_value())
    {
        DSR::Node n;
        n.type("metaconcept");
        n.name(kNodeName);
        live_->add_or_modify_attrib_local<ltsm_topology_att>(n, json);
        live_->add_or_modify_attrib_local<level_att>(n, 0);   // like passage_live: no parent, no RT edge
        live_->add_or_modify_attrib_local<pos_x_att>(n, -300.f);   // viewer placement only
        live_->add_or_modify_attrib_local<pos_y_att>(n, -300.f);
        if (not live_->insert_node(n).has_value()) return false;
    }
    else
    {
        live_->add_or_modify_attrib_local<ltsm_topology_att>(node.value(), json);
        live_->update_node(node.value());
    }
    last_ = json;
    return true;
}

void TopologyPublisher::remove()
{
    for (const auto& n : live_->get_nodes_by_type("metaconcept"))
        if (n.name() == kNodeName) live_->delete_node(n.id());
    last_.clear();
}
}   // namespace ltsm
