#include "topology_publish.h"
#include "room_eviction.h"      // ltsm::is_door
#include "topology_digest.h"

#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include "../../common/room_resolve/room_resolve.h"   // current_edge_room

#include <algorithm>
#include <optional>

namespace ltsm
{
namespace
{
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
        const auto name = rc::ltsm::live_name(r.name());
        if (std::ranges::any_of(rooms, [&](const auto& x) { return x.name == name; })) continue;
        rooms.push_back({.name = name, .live = live.get_node(name).has_value()});
    }
    std::ranges::sort(rooms, {}, &rc::ltsm::TopoRoom::name);
    std::vector<std::uint64_t> room_ids;   // every memory room's id (a repeated live name still has its own)
    for (const auto& r : memory.get_nodes_by_type("room"))
        if (const auto k = memory.get_attrib_by_name<room_id_att>(r); k.has_value()) room_ids.push_back(k.value());
    std::ranges::sort(room_ids);
    room_ids.erase(std::unique(room_ids.begin(), room_ids.end()), room_ids.end());

    // The robot's CURRENT room (live name): only a face in it may name the live door -- with a proto-room live, the
    // far end of a passage is also "live" but its door is not the one the robot stands at.
    std::string current;
    if (const auto cid = rc::room::current_edge_room(live); cid.has_value())
        if (const auto cn = live.get_node(cid.value()); cn.has_value()) current = cn->name();

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
            const auto rn = rc::ltsm::live_name(room->name());
            const auto fn = rc::ltsm::live_name(face->name());
            auto& slot_room = faces == 0 ? p.room_a : p.room_b;
            auto& slot_face = faces == 0 ? p.face_a : p.face_b;
            slot_room = rn;
            slot_face = fn;
            // The live door this face is NOW: only when its room is live and holds a door of that name.
            if (not p.live_door.has_value() and rn == current)
                if (const auto d = live.get_node(fn); d.has_value() and is_door(live, d.value())) p.live_door = fn;
            if (++faces == 2) break;
        }
        // A face is REMADE under another name when its room is re-entered (explore_tour13: memory's r1_door_2, live door_4;
        // 'passage_0 match door_3 <-> door_4'). The name lookup above then finds nothing, yet the live passage of the same
        // name already matches the door standing for this hole: the one of its matched doors in the CURRENT room.
        if (not p.live_door.has_value() and not current.empty())
            if (const auto lp = live.get_node(p.id); lp.has_value())
                for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(lp.value(), "match"))
                    if (const auto d = live.get_node(e.to()); d.has_value() and is_door(live, d.value()))
                        if (const auto r = owning_room(live, d.value()); r.has_value() and r->name() == current)
                        { p.live_door = d->name(); break; }
        if (faces == 2) passages.push_back(std::move(p));   // a passage with one face joins nothing
    }
    std::ranges::sort(passages, {}, &rc::ltsm::TopoPassage::id);
    return rc::ltsm::encode_topology(rooms, passages, room_ids, remembered_door_memory(memory));
}

namespace
{
constexpr const char* kDoorMemoryNode = "ltsm_door_memory";   // memory graph: the door names' counter + change hazard record

std::optional<rc::doormem::DoorMemory> door_memory_on(DSR::DSRGraph& g, const DSR::Node& n)
{
    const auto next = g.get_attrib_by_name<door_next_index_att>(n);
    const auto hz = g.get_attrib_by_name<door_change_hazard_att>(n);
    if (not next.has_value() and not hz.has_value()) return std::nullopt;
    rc::doormem::DoorMemory m;
    m.next_index = next.value_or(0);
    if (hz.has_value() and hz->get().size() == 2) { m.changes = hz->get()[0]; m.exposure_s = hz->get()[1]; }
    return m;
}
}   // namespace

rc::doormem::DoorMemory remembered_door_memory(DSR::DSRGraph& memory)
{
    rc::doormem::DoorMemory m;
    if (const auto rec = memory.get_node(kDoorMemoryNode); rec.has_value())
        m = door_memory_on(memory, rec.value()).value_or(rc::doormem::DoorMemory{});
    std::vector<std::string> names;
    for (const auto& n : memory.get_nodes_by_type("object"))
        if (is_door(memory, n)) names.push_back(rc::ltsm::live_name(n.name()));
    m.next_index = std::max(m.next_index, rc::doormem::next_index_floor(names));
    return m;
}

std::optional<rc::doormem::DoorMemory> agent_door_memory(DSR::DSRGraph& live)
{
    for (const auto& n : live.get_nodes_by_type("agent"))
        // cortex names it "<agent name> <agent id>" (dsr_agent_info_api.cpp): match the name, whatever id door_concept runs with
        if (const auto nm = live.get_attrib_by_name<agent_name_att>(n); nm.has_value() and nm->get().starts_with("door_concept "))
            return door_memory_on(live, n);
    return std::nullopt;
}

bool keep_door_memory(DSR::DSRGraph& memory, DSR::DSRGraph& live)
{
    const auto agent = agent_door_memory(live);
    if (not agent.has_value()) return false;
    auto rec = memory.get_node(kDoorMemoryNode);
    const auto held = rec.has_value() ? door_memory_on(memory, rec.value()).value_or(rc::doormem::DoorMemory{}) : rc::doormem::DoorMemory{};
    const auto m = rc::doormem::merged(held, *agent);
    if (rec.has_value() and m == held) return false;
    // ★Its own node, not root's attributes: the memory graph is seeded with a root before the persisted file is read, and the
    //  file's root is then skipped as an id already present (insert_node_with_id), so what root carried would not come back.
    DSR::Node n = rec.value_or(DSR::Node{});
    if (not rec.has_value())
    {
        n.type("metaconcept");
        n.name(kDoorMemoryNode);
        memory.add_or_modify_attrib_local<object_subtype_att>(n, std::string("door_memory"));   // never "passage": the passage loops skip it
        memory.add_or_modify_attrib_local<level_att>(n, 0);
    }
    memory.add_or_modify_attrib_local<door_next_index_att>(n, m.next_index);
    memory.add_or_modify_attrib_local<door_change_hazard_att>(n,
        std::vector<float>{static_cast<float>(m.changes), static_cast<float>(m.exposure_s)});
    return rec.has_value() ? memory.update_node(n) : memory.insert_node(n).has_value();
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
