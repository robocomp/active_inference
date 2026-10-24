#include "room_recall.h"
#include "room_eviction.h"

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <format>
#include <optional>
#include <unordered_set>

namespace ltsm
{
namespace
{
/// The `room` a node hangs from (walking parent_att), or nullopt.
std::optional<DSR::Node> room_of(DSR::DSRGraph &g, const DSR::Node &n)
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

bool is_recall(DSR::DSRGraph &g, const DSR::Node &n)
{
    return n.type() == "metaconcept" and n.name().starts_with("recall_")
           and g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "room_recall";
}
}   // namespace

std::string RoomRecall::live_name_of(const std::string &memory_room)
{
    std::string name = memory_room;
    if (const auto us = name.find('_'); name.starts_with('r') and us != std::string::npos) name = name.substr(us + 1);
    return name;
}

std::uint64_t RoomRecall::post_late(std::uint64_t anchor_id, const std::string &memory_room, const Mat::RTMat &anchor_from_room,
                                    const std::vector<float> &pose_prior)
{
    const auto anchor = live_->get_node(anchor_id);
    const auto mroom = mem_->get_node(memory_room);
    if (not anchor.has_value() or not mroom.has_value()) return 0;
    const auto px = mem_->get_attrib_by_name<delimiting_polygon_x_att>(mroom.value());
    const auto py = mem_->get_attrib_by_name<delimiting_polygon_y_att>(mroom.value());
    if (not px.has_value() or not py.has_value()) return 0;

    const std::string rname = "recall_" + live_name_of(memory_room);
    auto node = live_->get_node(rname);
    const bool fresh = not node.has_value();
    if (fresh)
    {
        DSR::Node n;
        n.type("metaconcept");
        n.name(rname);
        node = n;
    }
    live_->add_or_modify_attrib_local<object_subtype_att>(node.value(), std::string("room_recall"));
    live_->add_or_modify_attrib_local<recall_late_att>(node.value(), true);
    if (not pose_prior.empty())
        live_->add_or_modify_attrib_local<recall_pose_prior_att>(node.value(), std::vector<float>(pose_prior));
    if (const auto old_parent = live_->get_attrib_by_name<parent_att>(node.value());
        not fresh and old_parent.has_value() and old_parent.value() != anchor_id)
        live_->delete_edge(old_parent.value(), node->id(), "RT");
    live_->add_or_modify_attrib_local<parent_att>(node.value(), anchor_id);
    live_->add_or_modify_attrib_local<level_att>(node.value(), live_->get_node_level(anchor.value()).value_or(0) + 1);
    live_->add_or_modify_attrib_local<room_id_att>(node.value(), mem_->get_attrib_by_name<room_id_att>(mroom.value()).value_or(0));
    live_->add_or_modify_attrib_local<delimiting_polygon_x_att>(node.value(), std::vector<float>(px->get()));
    live_->add_or_modify_attrib_local<delimiting_polygon_y_att>(node.value(), std::vector<float>(py->get()));
    live_->add_or_modify_attrib_local<pos_x_att>(node.value(), live_->get_attrib_by_name<pos_x_att>(anchor.value()).value_or(0.f) + 60.f);
    live_->add_or_modify_attrib_local<pos_y_att>(node.value(), live_->get_attrib_by_name<pos_y_att>(anchor.value()).value_or(0.f) + 60.f);
    std::uint64_t id = node->id();
    if (fresh)
    {
        const auto ins = live_->insert_node(node.value());
        if (not ins.has_value()) return 0;
        id = ins.value();
        qInfo().noquote() << QString::fromStdString(std::format("[recall] LATE {} posted under {}: the robot is inside remembered {}",
                                                                rname, anchor->name(), memory_room));
    }
    else
        live_->update_node(node.value());
    live_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(anchor_id, id, RoomEviction::rt_attributes(anchor_from_room)));
    return id;
}

bool RoomRecall::remove_recall_of(const std::string &memory_room)
{
    const auto n = live_->get_node("recall_" + live_name_of(memory_room));
    return n.has_value() and is_recall(*live_, n.value()) and live_->delete_node(n->id());
}

int RoomRecall::late_count() const
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (is_recall(*live_, node) and live_->get_attrib_by_name<recall_late_att>(node).value_or(false)) ++n;
    return n;
}

RoomRecall::RoomRecall(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory,
                       const RoomEviction &evictor)
    : live_(std::move(live)), mem_(std::move(memory)), ev_(evictor) {}

int RoomRecall::remove_all()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (is_recall(*live_, node) and live_->delete_node(node.id())) ++n;
    owned_.clear();
    return n;
}

int RoomRecall::step()
{
    std::unordered_set<std::uint64_t> wanted;   // door ids that should carry a recall this cycle
    auto mie = mem_->get_inner_eigen_api();

    for (const auto &door : live_->get_nodes_by_type("object"))
    {
        if (not is_door(*live_, door)) continue;
        const auto room = room_of(*live_, door);
        if (not room.has_value()) continue;

        // This door's FACE in memory, by the same prefix the eviction wrote it under.
        auto face = mem_->get_node(ev_.memory_prefix(room.value()) + door.name());
        // ★ BY PLACE WHEN NOT BY NAME (2026-10-12). A re-entered room comes back with its crossed door RECREATED by
        //   door_concept under a new name (try28: door_37 in room_1, memory face r1_door_19), so the name lookup misses
        //   and no recall was posted for the room behind it — room_2 was then surveyed as a new room. A re-entered room
        //   is published in its own remembered frame, so the live door and the memory faces of that room are directly
        //   comparable: take the face of this room's memory copy whose aperture contains the live door's centre.
        if (not face.has_value())
            if (const auto mself = mem_->get_node(ev_.memory_prefix(room.value()) + room->name()); mself.has_value())
                if (const auto Tl = live_->get_inner_eigen_api()->get_transformation_matrix(room->name(), door.name()); Tl.has_value())
                {
                    const double half_w = 0.5 * live_->get_attrib_by_name<width_m_att>(door).value_or(1.f);
                    double best = half_w;
                    for (const auto &pass : mem_->get_nodes_by_type("metaconcept"))
                        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pass, "match"))
                        {
                            const auto f = mem_->get_node(e.to());
                            if (not f.has_value()) continue;
                            const auto fr = room_of(*mem_, f.value());
                            if (not fr.has_value() or fr->id() != mself->id()) continue;
                            const auto Tm = mie->get_transformation_matrix(mself->name(), f->name());
                            if (not Tm.has_value()) continue;
                            if (const double d = (Tm->translation() - Tl->translation()).head<2>().norm(); d < best)
                            { best = d; face = f; }
                        }
                }
        if (not face.has_value()) continue;

        // The passage holding that face, and its OTHER face: one in a DIFFERENT room. A passage written before
        // 2026-10-18 can carry two faces of the same room (try53: r1_door_1 and r1_door_1_entry), and taking the
        // sibling made room_1 the room behind room_1's own door, so no recall_room_2 was ever posted.
        std::optional<DSR::Node> other;
        const auto face_room = room_of(*mem_, face.value());
        if (face_room.has_value() and is_staged_room(*mem_, face_room.value())) continue;
        for (const auto &pass : mem_->get_nodes_by_type("metaconcept"))
        {
            const auto m = DSR::DSRGraph::get_node_edges_by_type(pass, "match");
            if (std::ranges::none_of(m, [&](const auto &e) { return e.to() == face->id(); })) continue;
            for (const auto &e : m)
                if (e.to() != face->id())
                    if (const auto f = mem_->get_node(e.to()); f.has_value())
                        if (const auto fr = room_of(*mem_, f.value());
                            fr.has_value() and not is_staged_room(*mem_, fr.value())
                            and (not face_room.has_value() or fr->id() != face_room->id()))
                            other = f;
        }
        if (not other.has_value()) continue;
        const auto mroom = room_of(*mem_, other.value());
        if (not mroom.has_value() or is_staged_room(*mem_, mroom.value())) continue;

        // The remembered room's LIVE name: memory's name without its r<idx>_ prefix.
        std::string name = mroom->name();
        if (const auto us = name.find('_'); name.starts_with('r') and us != std::string::npos) name = name.substr(us + 1);
        // Already live (we are in it, or it was never evicted): nothing to recall.
        if (live_->get_node(name).has_value()) continue;

        const auto px = mem_->get_attrib_by_name<delimiting_polygon_x_att>(mroom.value());
        const auto py = mem_->get_attrib_by_name<delimiting_polygon_y_att>(mroom.value());
        const auto T  = mie->get_transformation_matrix(other->name(), mroom->name());   // face <- room, in memory
        if (not px.has_value() or not py.has_value() or not T.has_value()) continue;

        wanted.insert(door.id());
        const std::string rname = "recall_" + name;
        auto node = live_->get_node(rname);
        // a LATE recall (RoomSwap) owns its name: step() never refreshes, re-parents or retires it
        if (node.has_value() and live_->get_attrib_by_name<recall_late_att>(node.value()).value_or(false)) continue;
        const bool fresh = not node.has_value();
        if (fresh)
        {
            DSR::Node n;
            n.type("metaconcept");
            n.name(rname);
            node = n;
        }
        live_->add_or_modify_attrib_local<object_subtype_att>(node.value(), std::string("room_recall"));
        live_->add_or_modify_attrib_local<parent_att>(node.value(), door.id());
        live_->add_or_modify_attrib_local<level_att>(node.value(), live_->get_node_level(door).value_or(0) + 1);
        live_->add_or_modify_attrib_local<room_id_att>(node.value(),
                mem_->get_attrib_by_name<room_id_att>(mroom.value()).value_or(0));
        live_->add_or_modify_attrib_local<delimiting_polygon_x_att>(node.value(), std::vector<float>(px->get()));
        live_->add_or_modify_attrib_local<delimiting_polygon_y_att>(node.value(), std::vector<float>(py->get()));
        live_->add_or_modify_attrib_local<pos_x_att>(node.value(), live_->get_attrib_by_name<pos_x_att>(door).value_or(0.f) + 60.f);
        live_->add_or_modify_attrib_local<pos_y_att>(node.value(), live_->get_attrib_by_name<pos_y_att>(door).value_or(0.f) + 60.f);
        std::uint64_t id = node->id();
        if (fresh)
        {
            const auto ins = live_->insert_node(node.value());
            if (not ins.has_value()) continue;
            id = ins.value();
            qInfo().noquote() << QString::fromStdString(std::format(
                "[recall] {} posted under {}: behind it lies remembered {} (passage face {} ↔ {})",
                rname, door.name(), mroom->name(), face->name(), other->name()));
        }
        else
            live_->update_node(node.value());
        auto e = DSR::Edge::create<RT_edge_type>(door.id(), id, RoomEviction::rt_attributes(T.value()));
        live_->insert_or_assign_edge(std::move(e));
        owned_[door.id()] = id;
    }

    // Retire recalls whose door is gone, whose room is live again, or that no longer resolve.
    for (auto it = owned_.begin(); it != owned_.end();)
    {
        if (wanted.contains(it->first)) { ++it; continue; }
        if (const auto n = live_->get_node(it->second); n.has_value())
        {
            qInfo().noquote() << QString::fromStdString(std::format("[recall] {} retired", n->name()));
            live_->delete_node(it->second);
        }
        it = owned_.erase(it);
    }
    return static_cast<int>(owned_.size());
}
}   // namespace ltsm
