#include "../../common/room_transition/exit_fact.h"   // link 4: is_archived
#include "../../common/room_transition/startup_recall.h"   // warm start
#include "../../common/room_resolve/room_resolve.h"       // rc::room::current_room
#include "room_recall.h"
#include "../../door_concept/src/door_leaf_memory.h"   // recall_leaf_probs: near face, else far complemented
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
    std::unordered_set<std::uint64_t> wanted;          // door ids that should carry a recall this cycle
    std::unordered_set<std::uint64_t> wanted_recalls;  // recall nodes some door still claims (see the retirement below)
    auto mie = mem_->get_inner_eigen_api();

    // ONE DOOR PER FACE: two live doors can resolve to the same memory face (cmp_br2: just after an adoption room_concept has the
    // room in K's frame while door_13, an entry mirror door_concept re-seats a beat later, still stands at its provisional
    // (0.00,0.03), 0.11 m from r2_door_5 at (0.10,0.09), so door_13 took recall_room_1). The face's recall goes to the door
    // nearest the face: an argmin, no door is excluded by distance.
    std::unordered_map<std::uint64_t, std::pair<double, std::uint64_t>> nearest;   // face id -> (distance, door id)
    const auto doors = live_->get_nodes_by_type("object");
    for (const auto &door : doors)
        if (is_door(*live_, door))
            if (const auto hole = ev_.remembered_hole(door, /*staged_twin_ok=*/true); hole.has_value())
                if (const auto [it, added] = nearest.try_emplace(hole->face.id(), hole->distance, door.id());
                    not added and hole->distance < it->second.first)
                    it->second = {hole->distance, door.id()};

    for (const auto &door : doors)
    {
        if (not is_door(*live_, door)) continue;
        const auto room = room_of(*live_, door);
        if (not room.has_value()) continue;
        if (rc::exit_fact::is_archived(*live_, room->id())) continue;   // link 4: the room left; its owners are deleting it

        // The hole memory already knows behind this door, BY PLACE (RoomEviction::remembered_hole): the door's NAME is a
        // label door_concept re-makes at every crossing (try28: door_37 in room_1 for the face r1_door_19), never the key.
        // A re-entered room is published in its own remembered frame, so the live door and its room's memory faces are
        // directly comparable; the name is consulted only when the door's pose in its room does not resolve.
        const auto hole = ev_.remembered_hole(door, /*staged_twin_ok=*/true);   // F3: see the face_room test below
        if (not hole.has_value()) continue;
        if (nearest.at(hole->face.id()).second != door.id()) continue;     // a nearer door holds this face
        const auto &face = hole->face;

        // The passage's OTHER face: one in a DIFFERENT room. A passage written before 2026-10-18 can carry two faces of
        // the same room (try53: r1_door_1 and r1_door_1_entry), and taking the sibling made room_1 the room behind
        // room_1's own door, so no recall_room_2 was ever posted.
        std::optional<DSR::Node> other;
        const auto face_room = room_of(*mem_, face);
        // A face in a STAGED room is skipped — except in the staged twin of the door's OWN live room (F3, TLA ping-pong): while
        // a recognition proposal for the provisional room is pending, RoomSwap stages its twin, and the twin's face still
        // identifies the hole the live door is; the room BEHIND it (the other face's, checked below) is not staged. Skipping it
        // retired the recall back to the room just left, and a crossing back in that window birthed a NEW room.
        const bool own_twin = face_room.has_value()
            and face_room->name() == std::format("r{}_{}", live_->get_attrib_by_name<room_id_att>(room.value()).value_or(0), room->name());
        if (face_room.has_value() and is_staged_room(*mem_, face_room.value()) and not own_twin) continue;
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(hole->passage, "match"))
            if (e.to() != face.id())
                if (const auto f = mem_->get_node(e.to()); f.has_value())
                    if (const auto fr = room_of(*mem_, f.value());
                        fr.has_value() and not is_staged_room(*mem_, fr.value())
                        and (not face_room.has_value() or fr->id() != face_room->id()))
                        other = f;
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
        if (node.has_value()) wanted_recalls.insert(node->id());
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
        // RE-PARENTED: the RT edge from the old door goes with the parent (as in post_late); a stale one would keep naming the
        // recall under a door that no longer owns it (the controller and door_concept read recalls off a door's RT children).
        if (const auto old_parent = live_->get_attrib_by_name<parent_att>(node.value());
            not fresh and old_parent.has_value() and old_parent.value() != door.id())
        {
            live_->delete_edge(old_parent.value(), node->id(), "RT");
            qInfo().noquote() << QString::fromStdString(std::format("[recall] {} moves to {} (the door nearest its face)", rname, door.name()));
        }
        live_->add_or_modify_attrib_local<parent_att>(node.value(), door.id());
        live_->add_or_modify_attrib_local<level_att>(node.value(), live_->get_node_level(door).value_or(0) + 1);
        live_->add_or_modify_attrib_local<room_id_att>(node.value(),
                mem_->get_attrib_by_name<room_id_att>(mroom.value()).value_or(0));
        live_->add_or_modify_attrib_local<delimiting_polygon_x_att>(node.value(), std::vector<float>(px->get()));
        live_->add_or_modify_attrib_local<delimiting_polygon_y_att>(node.value(), std::vector<float>(py->get()));
        live_->add_or_modify_attrib_local<pos_x_att>(node.value(), live_->get_attrib_by_name<pos_x_att>(door).value_or(0.f) + 60.f);
        live_->add_or_modify_attrib_local<pos_y_att>(node.value(), live_->get_attrib_by_name<pos_y_att>(door).value_or(0.f) + 60.f);
        // The remembered LEAF SIDE for the live door this recall hangs under: the near face's (the one the live door matches),
        // else the far face's complemented (door_leaf_memory.h recall_leaf_probs). Neither -> nothing written.
        {
            const auto attrs = [&](const DSR::Node &f) {
                return rc::door::leafmem::LeafAttrs{
                    mem_->get_attrib_by_name<door_leaf_into_room_prob_att>(f),
                    mem_->get_attrib_by_name<door_leaf_hinge_plus_prob_att>(f)};
            };
            if (const auto lp = rc::door::leafmem::recall_leaf_probs(attrs(face), attrs(other.value())); lp.has_value())
            {
                live_->add_or_modify_attrib_local<door_leaf_into_room_prob_att>(node.value(), lp->first);
                live_->add_or_modify_attrib_local<door_leaf_hinge_plus_prob_att>(node.value(), lp->second);
            }
        }
        std::uint64_t id = node->id();
        if (fresh)
        {
            const auto ins = live_->insert_node(node.value());
            if (not ins.has_value()) continue;
            id = ins.value();
            wanted_recalls.insert(id);
            const Eigen::Matrix4d M = T.value().matrix();
            qInfo().noquote() << QString::fromStdString(std::format(
                "[recall] {} posted under {}: behind it lies remembered {} (passage face {} ↔ {}); RT door<-room = "
                "face {} <- {} in memory: ({:+.3f},{:+.3f}) yaw {:+.1f} deg",
                rname, door.name(), mroom->name(), face.name(), other->name(), other->name(), mroom->name(),
                M(0, 3), M(1, 3), std::atan2(M(1, 0), M(0, 0)) * 57.29578));
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
        // The door no longer holds it, but another door does (the recall moved): only the stale ownership goes, not the node.
        if (wanted_recalls.contains(it->second)) { it = owned_.erase(it); continue; }
        if (const auto n = live_->get_node(it->second); n.has_value())
        {
            qInfo().noquote() << QString::fromStdString(std::format("[recall] {} retired", n->name()));
            live_->delete_node(it->second);
        }
        it = owned_.erase(it);
    }
    return static_cast<int>(owned_.size());
}
std::uint64_t RoomRecall::post_startup(const std::string &live_room)
{
    const auto missing = [&](std::string why) -> std::uint64_t
    {
        if (why != startup_missing_)
            qInfo().noquote() << QString::fromStdString(std::format("[warm] start-up recall of {} not posted yet: {}", live_room, why));
        startup_missing_ = std::move(why);
        return 0;
    };
    const std::string rname = "recall_" + live_room;
    if (const auto existing = live_->get_node(rname); existing.has_value())
        return startup_id_ = existing->id(), startup_room_ = live_room, startup_id_;

    // the remembered room: memory r<k>_<live_room> with room_id k (RoomEviction::remembered's rule)
    std::optional<DSR::Node> mroom;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (const auto k = mem_->get_attrib_by_name<room_id_att>(r); k.has_value() and r.name() == std::format("r{}_{}", k.value(), live_room))
            mroom = r;
    if (not mroom.has_value()) return missing(std::format("memory holds no r<k>_{}", live_room));
    const auto px = mem_->get_attrib_by_name<delimiting_polygon_x_att>(mroom.value());
    const auto py = mem_->get_attrib_by_name<delimiting_polygon_y_att>(mroom.value());
    if (not px.has_value() or not py.has_value() or not rc::room_transition::recall_outline(px->get(), py->get()).has_value())
        return missing(std::format("{} has no usable outline", mroom->name()));
    const auto robots = live_->get_nodes_by_type("robot");
    if (robots.empty()) return missing("no robot node in the working memory yet");
    const auto &robot = robots.front();

    DSR::Node n;
    n.type("metaconcept");
    n.name(rname);
    live_->add_or_modify_attrib_local<object_subtype_att>(n, std::string(rc::room_transition::kRecallSubtype));
    live_->add_or_modify_attrib_local<parent_att>(n, robot.id());
    live_->add_or_modify_attrib_local<level_att>(n, live_->get_node_level(robot).value_or(0) + 1);
    live_->add_or_modify_attrib_local<room_id_att>(n, mem_->get_attrib_by_name<room_id_att>(mroom.value()).value_or(0));
    live_->add_or_modify_attrib_local<delimiting_polygon_x_att>(n, std::vector<float>(px->get()));
    live_->add_or_modify_attrib_local<delimiting_polygon_y_att>(n, std::vector<float>(py->get()));
    live_->add_or_modify_attrib_local<pos_x_att>(n, live_->get_attrib_by_name<pos_x_att>(robot).value_or(0.f) + 60.f);
    live_->add_or_modify_attrib_local<pos_y_att>(n, live_->get_attrib_by_name<pos_y_att>(robot).value_or(0.f) - 60.f);
    const auto id = live_->insert_node(n);
    if (not id.has_value()) return missing("insert_node failed");
    startup_id_ = id.value();
    startup_room_ = live_room;
    qInfo().noquote() << QString::fromStdString(std::format(
        "[warm] start-up recall {} posted under {} (room_id {}, {} vertices, remembered {})", rname, robot.name(),
        mem_->get_attrib_by_name<room_id_att>(mroom.value()).value_or(0), px->get().size(), mroom->name()));
    return startup_id_;
}

void RoomRecall::step_startup()
{
    if (startup_id_ == 0) return;
    const auto cur = rc::room::current_room_node(*live_);
    if (not cur.has_value() or cur->name() == startup_room_) return;
    if (const auto n = live_->get_node(startup_id_); n.has_value())
    {
        qInfo().noquote() << QString::fromStdString(std::format("[warm] start-up recall {} retired: the robot is in {}", n->name(), cur->name()));
        live_->delete_node(startup_id_);
    }
    startup_id_ = 0;
}
}   // namespace ltsm
