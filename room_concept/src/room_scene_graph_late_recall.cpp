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

#include <algorithm>
#include <cmath>
#include <limits>

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
        // The parent must be one of the two anchors the contract allows: a door node (type door, or object with
        // object_subtype door) for case A, or the robot (case B).
        const auto parent = G_->get_attrib_by_name<parent_att>(n).value_or(0);
        const auto pnode = parent != 0 ? G_->get_node(parent) : std::nullopt;
        const bool is_door = pnode.has_value() and
                             (pnode->type() == "door" or
                              G_->get_attrib_by_name<object_subtype_att>(pnode.value()).value_or(std::string{}) == "door");
        if (not pnode.has_value() or not (is_door or parent == dsr_robot_id_))
        {
            if (late_bad_parent_logged_ != n.id())
            {
                late_bad_parent_logged_ = n.id();
                qWarning().noquote() << QString("[room][late-recall] %1 ignored: parent %2 is neither a door nor the robot")
                                            .arg(QString::fromStdString(n.name())).arg(parent);
            }
            continue;
        }
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

double RoomSceneGraph::recognition_log_odds(std::uint64_t room_id) const
{
    // ltsm's recogniser posts one `recognition` metaconcept: recog_room_ids (the candidates) and recog_posterior (P over them,
    // then P(new) LAST; ltsm_agent/src/room_recognizer.cpp). Its log-odds of K against "a new room" is the prior of the
    // adoption test (RoomConcept::seed_reloc_prior). Absent node / attribute / entry: no recognition evidence, 0.
    const auto n = G_->get_node(std::string("recognition"));
    if (not n.has_value() or n->type() != "metaconcept"
        or G_->get_attrib_by_name<object_subtype_att>(n.value()).value_or(std::string{}) != "recognition")
        return 0.0;
    const auto ids  = G_->get_attrib_by_name<recog_room_ids_att>(n.value());
    const auto post = G_->get_attrib_by_name<recog_posterior_att>(n.value());
    if (not ids.has_value() or not post.has_value()) return 0.0;
    const auto& iv = ids->get(); const auto& pv = post->get();
    if (pv.size() != iv.size() + 1) return 0.0;
    const auto it = std::ranges::find(iv, room_id);
    if (it == iv.end()) return 0.0;
    // Numerical guard: a posterior that underflowed to 0 is floored at the smallest normal float, not taken as -inf.
    const auto ln = [](float p) { return std::log(std::max(static_cast<double>(p), static_cast<double>(std::numeric_limits<float>::min()))); };
    const float pk = pv[static_cast<std::size_t>(it - iv.begin())], pn = pv.back();
    if (not std::isfinite(pk) or not std::isfinite(pn)) return 0.0;
    return ln(pk) - ln(pn);
}

bool RoomSceneGraph::adopt_late_recall(const Recall& recall)
{
    // The recognition evidence is read ONCE, at adoption: it is the adoption test's prior, not a running input.
    Recall r = recall;
    r.recog_log_odds = recognition_log_odds(r.room_index);
    if (proto_room_id_ != 0 or r.poly_pub.size() < 3 or r.room_index == 0) return false;
    for (const auto& rn : G_->get_nodes_by_type("room"))
        if (rn.name() == r.name) return false;   // the name is taken (the room is live): nothing to re-birth
    // ltsm discards the provisional room in the SAME cycle that posts the late recall, so by the time this runs the room we
    // think we are in is usually GONE from the graph (recog_try14: the old code returned false here, the reset path then
    // re-birthed a start-up proto and its sweep deleted the room being LEFT). A vanished room is released and the recalled
    // one is born in its place; a still-live provisional room is replaced as before.
    auto old = dsr_room_id_ != 0 ? G_->get_node(dsr_room_id_) : std::nullopt;
    if (not old.has_value() and dsr_room_id_ != 0) forget_room("late recall: the provisional room is already gone");
    DSR::Node node = DSR::Node::create<room_node_type>(r.name);
    G_->add_or_modify_attrib_local<room_id_att>(node, r.room_index);
    const float own_height = published_room_height_ > 0.f ? published_room_height_ : params_->room_height;
    const float height = old.has_value() ? G_->get_attrib_by_name<room_height_att>(old.value()).value_or(own_height)
                                         : own_height;
    G_->add_or_modify_attrib_local<room_height_att>(node, height);
    if (old.has_value())
        if (const auto l = G_->get_attrib_by_name<level_att>(old.value()); l.has_value())
            G_->add_or_modify_attrib_local<level_att>(node, l.value());
    G_->add_or_modify_attrib_local<pos_x_att>(node, (old.has_value() ? G_->get_attrib_by_name<pos_x_att>(old.value()).value_or(0.f) : 0.f) + 250.f);
    G_->add_or_modify_attrib_local<pos_y_att>(node, old.has_value() ? G_->get_attrib_by_name<pos_y_att>(old.value()).value_or(0.f) : 0.f);
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
    // The remembered room's frame is the internal one now: frame-bound state of the old frame goes.
    left_polys_int_.clear();
    have_last_int_pose_ = false;
    late_watch_node_   = r.node_id;
    late_watch_room_   = id.value();
    late_adopted_name_ = r.name;
    qInfo().noquote() << QString("[room][late-recall] %1 adopted from memory (node id %2), replacing room %3; localising with "
                                 "the recognition posterior (%4 mode(s)) as the search's prior; adoption test starts at "
                                 "log-odds %5 (recogniser ln P(K)/P(new))")
                             .arg(QString::fromStdString(r.name)).arg(id.value()).arg(old.has_value() ? old->id() : 0).arg(r.pose_prior.size())
                             .arg(r.recog_log_odds, 0, 'f', 2);
    return true;
}

bool RoomSceneGraph::step_late_recall()
{
    if (G_ == nullptr) return false;
    const std::int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    // No room at all (the provisional one was discarded under us): look EVERY cycle, so no start-up birth can run first.
    const bool poll = not room_node_created_ or now - late_last_poll_ms_ >= 1000;
    if (poll) late_last_poll_ms_ = now;
    std::optional<Recall> standing;
    if (poll or late_watch_node_ != 0) standing = find_late_recall();
    // The watch ends by MODEL EVENTS, never by waiting: the adopted room or the recall node is gone, or another late recall
    // stands in its place (an undo posts the staged twin's). Whatever the search still holds is dropped with it.
    if (late_watch_node_ != 0)
    {
        const bool gone = not G_->get_node(late_watch_node_).has_value() or not G_->get_node(late_watch_room_).has_value()
                          or dsr_room_id_ != late_watch_room_
                          or (standing.has_value() and standing->node_id != late_watch_node_);
        if (gone)
        {
            room_concept_->cancel_late_recall();
            late_watch_node_ = late_watch_room_ = 0;
        }
        else
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
                late_watch_node_ = late_watch_room_ = 0;
                break;
            case rc::RoomConcept::LateVerdict::Accepted:
                late_watch_node_ = late_watch_room_ = 0;
                break;
            default: return false;   // still judging
            }
            return false;
        }
    }
    if (not poll) return false;
    // A recall for the room already live is ltsm's own (retired by it); an undo posts one for the OTHER room, and the same
    // function adopts that one. ltsm's undo removes K's recall and posts the twin's; the twin's name may still be taken by
    // the provisional room for a moment: adopt_late_recall then returns false without side effects and the next poll retries.
    if (standing.has_value() and standing->name != room_node_name())
        return adopt_late_recall(*standing);
    return false;
}

}   // namespace rc
