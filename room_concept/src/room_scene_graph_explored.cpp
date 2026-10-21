/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// P(room explored) on the room node — the belief a mission's `until explored(room)` reads (Task 5, 2026-10-21).
// The model is in room_explored.h (Luce stop-vs-look on the explorer's own price, noisy-OR with memory); this file is
// the agent's side of it: writing the attribute, restoring memory on a re-entry, and the coverage plan of a partly
// remembered room. tools/room_explored_selftest.cpp runs the same explorer calls on a synthetic room.
#include "room_scene_graph.h"

#include <cmath>

#include <QDebug>

namespace rc
{

void RoomSceneGraph::publish_explored(std::optional<float> p)
{
    if (not p.has_value() or G_ == nullptr or dsr_room_id_ == 0) return;   // not assessed: the node says nothing
    if (explored_room_id_ == dsr_room_id_ and *p == explored_p_) return;   // unchanged: no graph churn
    auto rn = G_->get_node(dsr_room_id_);
    if (not rn.has_value()) return;
    G_->add_or_modify_attrib_local<room_explored_p_att>(rn.value(), *p);
    if (not G_->update_node(rn.value())) return;
    qInfo().noquote() << QString("[room][explored] %1: P(explored) = %2 (memory %3)")
                             .arg(QString::fromStdString(rn->name())).arg(*p, 0, 'f', 3).arg(recall_p_mem_, 0, 'f', 2);
    explored_room_id_ = dsr_room_id_;
    explored_p_ = *p;
    explored_memory_.remember(rn->name(), *p);   // what a re-entry into this room restores, later in the run
}

void RoomSceneGraph::note_recall_memory(const Recall& recall)
{
    recall_p_mem_ = rc::explored::restored(recall.explored_p, explored_memory_.recall(recall.name));
    // The explorer's standpoints and its countable outline are the REMEMBERED outline (published frame): a re-entry
    // never ran the start-up ring that fills fs_published_ for a surveyed room.
    if (fs_published_.size() < 3) fs_published_ = published_polygon();
    qInfo().noquote() << QString("[room][explored] %1 recalled: memory restores P(explored) = %2 (%3)")
                             .arg(QString::fromStdString(recall.name)).arg(recall_p_mem_, 0, 'f', 3)
                             .arg(recall.explored_p ? "carried by the recall"
                                  : explored_memory_.recall(recall.name) ? "this agent's record of the room"
                                  : "outline only: the promoted layout is complete");
    // A complete memory is explored on entry; a partial one is published on the explorer's first plan.
    if (recall_p_mem_ >= 1.f) publish_explored(1.f);
}

float RoomSceneGraph::footprint_area() const
{
    return static_cast<float>(M_PI) * 0.25f
         * (params_->ROBOT_WIDTH * params_->ROBOT_WIDTH + params_->ROBOT_LENGTH * params_->ROBOT_LENGTH);
}

rc::layout::Plan RoomSceneGraph::plan_recalled_rest(const rc::layout::FeasibleGrid& grid, const Eigen::Vector2f& robot,
                                                    const rc::layout::ExplorerParams& ep,
                                                    const rc::layout::Admissible& standable,
                                                    const rc::layout::Admissible& deferred)
{
    // The survey's coverage gain (unseen cells inside or bordering the outline, the free-space record restarted at the
    // crossing), counted only for what memory may lack. Same price as the survey: the robot's footprint.
    const auto countable = free_space_->countable_mask(fs_published_, 0.40f);
    const float p_mem = recall_p_mem_;
    return rc::layout::plan_coverage(grid, robot, ep, standable,
        [this, &countable, p_mem](const Eigen::Vector2f& v)
        { return rc::explored::recalled_gain(p_mem, free_space_->visible_unknown_area(T_pub_int_ * v, 8.f, 180, &countable)); },
        footprint_area(), deferred);
}

}   // namespace rc
