/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// LATE ADOPTION OF A RECALLED ROOM (Task 9, docs/superpowers/specs/2026-10-22-room-recognition-design.md).
// ltsm's recogniser can decide that the room the robot is standing in is a remembered one AFTER room_concept has already
// begun surveying (or even promoted) a provisional room. It then posts `recall_<room>` (metaconcept, subtype room_recall,
// recall_late = true, recall_pose_prior = the robot's pose in that room's frame) and swaps the provisional room out.
// This file is room_concept's side: re-birth the room from the recall (rearm_after_crossing's recall path, with the pose
// prior in place of a doorway anchor), and report a REJECTED adoption on the recall node (`valid = false`), which is what
// lets ltsm undo the swap. The recall node is ltsm's: this agent reads it and writes `valid`, and never deletes it.
// (late-adoption contract: ltsm_agent/src/room_swap.h)
#include "room_scene_graph.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

#include <cmath>

#include <QDebug>

namespace rc
{

std::optional<RoomSceneGraph::Recall> RoomSceneGraph::find_late_recall() const
{
    for (const auto& n : G_->get_nodes_by_type("metaconcept"))
    {
        if (G_->get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) != "room_recall") continue;
        if (not G_->get_attrib_by_name<recall_late_att>(n).value_or(false)) continue;
        const auto px = G_->get_attrib_by_name<delimiting_polygon_x_att>(n);
        const auto py = G_->get_attrib_by_name<delimiting_polygon_y_att>(n);
        const auto pp = G_->get_attrib_by_name<recall_pose_prior_att>(n);
        if (not px.has_value() or not py.has_value() or not pp.has_value()) continue;
        // Either anchor: the crossed door of the room being left (case A) or the robot itself (case B, no door survives).
        // The recall's own RT edge is not needed: the prior is a pose in the recalled room's frame, not a transform.
        const auto parent = G_->get_attrib_by_name<parent_att>(n).value_or(0);
        if (parent == 0 or not G_->get_node(parent).has_value()) continue;
        const auto& xs = px->get(); const auto& ys = py->get(); const auto& v = pp->get();
        if (xs.size() < 3 or xs.size() != ys.size() or v.empty() or v.size() % 10 != 0) continue;
        Recall r;
        r.late = true;
        r.node_id = n.id();
        r.name = n.name().starts_with("recall_") ? n.name().substr(7) : n.name();
        r.room_index = static_cast<std::uint64_t>(G_->get_attrib_by_name<room_id_att>(n).value_or(0));
        if (const auto ep = G_->get_attrib_by_name<room_explored_p_att>(n); ep.has_value()) r.explored_p = ep.value();
        for (std::size_t i = 0; i < xs.size(); ++i) r.poly_pub.emplace_back(xs[i], ys[i]);
        r.T_int_recall = Eigen::Affine2f::Identity();   // the remembered room's frame BECOMES the internal one
        for (std::size_t c = 0; c < v.size(); c += 10)   // [w,x,y,th,Sxx,Sxy,Sxth,Syy,Syth,Sthth]
        {
            rc::reloc::PriorMode pm;
            pm.w  = v[c];
            pm.mu = Eigen::Vector3f(v[c + 1], v[c + 2], v[c + 3]);
            pm.S << v[c + 4], v[c + 5], v[c + 6],
                    v[c + 5], v[c + 7], v[c + 8],
                    v[c + 6], v[c + 8], v[c + 9];
            if (pm.mu.allFinite() and pm.S.allFinite() and pm.w > 0.f) r.pose_prior.push_back(pm);
        }
        if (r.pose_prior.empty()) continue;
        return r;
    }
    return std::nullopt;
}

bool RoomSceneGraph::adopt_late_recall(const Recall& r)
{
    if (dsr_room_id_ == 0 or proto_room_id_ != 0 or r.poly_pub.size() < 3 or r.room_index == 0) return false;
    for (const auto& rn : G_->get_nodes_by_type("room"))
        if (rn.name() == r.name) return false;   // the name is taken (the room is live): nothing to re-birth
    const auto old = G_->get_node(dsr_room_id_);
    if (not old.has_value()) return false;
    DSR::Node node = DSR::Node::create<room_node_type>(r.name);
    G_->add_or_modify_attrib_local<room_id_att>(node, r.room_index);
    if (const auto h = G_->get_attrib_by_name<room_height_att>(old.value()); h.has_value())
        G_->add_or_modify_attrib_local<room_height_att>(node, h.value());
    if (const auto l = G_->get_attrib_by_name<level_att>(old.value()); l.has_value())
        G_->add_or_modify_attrib_local<level_att>(node, l.value());
    G_->add_or_modify_attrib_local<pos_x_att>(node, G_->get_attrib_by_name<pos_x_att>(old.value()).value_or(0.f) + 250.f);
    G_->add_or_modify_attrib_local<pos_y_att>(node, G_->get_attrib_by_name<pos_y_att>(old.value()).value_or(0.f));
    G_->add_or_modify_attrib_local<parent_att>(node, dsr_robot_id_);
    rc::provenance::stamp_creation(*G_, node);
    const auto id = G_->insert_node(node);
    if (not id.has_value())
    {
        qWarning() << "[room][late-recall] failed to insert the room node" << QString::fromStdString(r.name);
        return false;
    }
    // The `proto` self-edge, so that the promotion inside rearm_after_crossing removes it exactly as on a door re-entry.
    G_->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(id.value(), id.value()));
    note_room_name_issued(r.room_index);
    mark_room_rt_not_current();   // the room being replaced keeps the last pose that was true of it
    rearm_after_crossing(id.value(), r.name, "late recall", 1.f, r);
    late_watch_node_   = r.node_id;
    late_adopted_name_ = r.name;
    qInfo().noquote() << QString("[room][late-recall] %1 adopted from memory (node id %2), replacing room %3; localising with "
                                 "the recognition posterior (%4 mode(s)) as the search's prior")
                             .arg(QString::fromStdString(r.name)).arg(id.value()).arg(old->id()).arg(r.pose_prior.size());
    return true;
}

void RoomSceneGraph::step_late_recall()
{
    if (G_ == nullptr or dsr_room_id_ == 0) return;
    // The verdict first: the search that judges the adoption has run on the localiser thread.
    if (late_watch_node_ != 0)
    {
        switch (room_concept_->late_recall_verdict())
        {
        case rc::RoomConcept::LateVerdict::Rejected:
            if (auto n = G_->get_node(late_watch_node_); n.has_value())
            {
                G_->add_or_modify_attrib_local<valid_att>(n.value(), false);   // ltsm's undo reads this; never delete the node
                G_->update_node(n.value());
                qWarning().noquote() << QString("[room][late-recall] %1 REJECTED by the late-adoption fit: valid=false on its recall")
                                            .arg(QString::fromStdString(late_adopted_name_));
            }
            late_watch_node_ = 0;
            break;
        case rc::RoomConcept::LateVerdict::Accepted:
            late_watch_node_ = 0;
            break;
        default: break;
        }
        return;   // one adoption at a time: the next recall waits for this verdict
    }
    const std::int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - late_last_poll_ms_ < 1000) return;
    late_last_poll_ms_ = now;
    const auto r = find_late_recall();
    // A recall for the room already live is ltsm's own (retired by it); an undo posts one for the OTHER room, and the same
    // function adopts that one.
    if (r.has_value() and r->name != room_node_name())
        adopt_late_recall(*r);
}

}   // namespace rc
