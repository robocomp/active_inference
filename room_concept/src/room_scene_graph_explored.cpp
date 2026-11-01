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
                             .arg(QString::fromStdString(rn->name())).arg(*p, 0, 'f', 3).arg(explored_.p_mem(), 0, 'f', 2);
    explored_room_id_ = dsr_room_id_;
    explored_p_ = *p;
}

std::string RoomSceneGraph::room_node_name() const
{
    if (G_ == nullptr or dsr_room_id_ == 0) return {};
    const auto rn = G_->get_node(dsr_room_id_);
    return rn.has_value() ? rn->name() : std::string{};
}

void RoomSceneGraph::note_recall_memory(const Recall& recall)
{
    resume_camera_cover(recall.name);   // before the settled publish below: an unfinished camera phase is not "explored"
    const bool had_record = explored_.memory().recall(recall.name).has_value();
    explored_.enter_recalled(recall.name, recall.explored_p);
    // The explorer's standpoints and its countable outline are the REMEMBERED outline (published frame): a re-entry
    // never ran the start-up ring that fills fs_published_ for a surveyed room.
    if (fs_published_.size() < 3) fs_published_ = published_polygon();
    qInfo().noquote() << QString("[room][explored] %1 recalled: memory restores P(explored) = %2 (%3)")
                             .arg(QString::fromStdString(recall.name)).arg(explored_.p_mem(), 0, 'f', 3)
                             .arg(recall.explored_p ? "carried by the recall"
                                  : had_record ? "this agent's record of the room"
                                  : "outline only: the promoted layout is complete");
    // A complete memory is explored on entry; a partial one is published on the explorer's first plan.
    if (not explored_.recalled_partial() and not camera_cover_open()) publish_explored(explored_.settled(recall.name));
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
    const float p_mem = explored_.p_mem();
    return rc::layout::plan_coverage(grid, robot, ep, standable,
        [this, &countable, p_mem](const Eigen::Vector2f& v)
        { return rc::explored::recalled_gain(p_mem, free_space_->visible_unknown_area(T_pub_int_ * v, 8.f, 180, &countable)); },
        footprint_area(), deferred);
}

// ── CAMERA COVERAGE OF THE BOUNDARY (camera_coverage.h; Task 12, 2026-10-29) ─────────────────────────────────────────
void RoomSceneGraph::resolve_zed_mount()
{
    if (zed_mount_resolved_ or not G_ or not rt_api_ or dsr_robot_id_ == 0) return;
    // Same walk as resolve_lidar_origin: zed → … → robot node, read once (the mount does not move). ROBOT_GEOMETRY.md:
    // the zed frame is x-right, y-DEPTH, z-up, so its optical axis is the frame's +y column.
    auto node = G_->get_node("zed");
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    bool ok = node.has_value();
    for (int depth = 0; ok and depth < 8 and node->id() != dsr_robot_id_; ++depth)
    {
        const auto rt = rt_api_->get_RT_pose_from_parent(node.value());
        const auto parent = G_->get_attrib_by_name<parent_att>(node.value());
        if (not rt.has_value() or not parent.has_value()) { ok = false; break; }
        T = Eigen::Affine3d(rt->matrix()) * T;
        node = G_->get_node(parent.value());
        ok = node.has_value();
    }
    ok = ok and node->id() == dsr_robot_id_;
    if (ok)
    {
        zed_xy_ = T.translation().head<2>().cast<float>();
        const Eigen::Vector2f f = T.linear().col(1).head<2>().cast<float>();
        if (f.norm() > 1e-6f) zed_fwd_ = f.normalized();
        cam_sensor_.height_m = static_cast<float>(T.translation().z());
        zed_mount_resolved_ = true;
        qInfo() << "[room][camera-cover] zed mount: (" << zed_xy_.x() << "," << zed_xy_.y() << ") m, axis ("
                << zed_fwd_.x() << "," << zed_fwd_.y() << "), height" << cam_sensor_.height_m << "m in the robot frame";
    }
    else if (++zed_mount_tries_ == 200 and not camera_model_warned_)
    {
        camera_model_warned_ = true;
        qWarning() << "[room][camera-cover] could not walk zed -> robot in the RT tree: NO camera coverage of the room's"
                   << "boundary — rooms will be called explored on the LiDAR alone (doors on unseen walls may be missed)";
    }
}

void RoomSceneGraph::observe_camera_cover(const rc::RoomConcept::UpdateResult& res)
{
    if (not G_ or not startup_proto_ or camera_finished_) return;   // this agent's room, while its camera question is open
    resolve_zed_mount();
    // intrinsics per call until complete: robot_concept publishes them only once frames flow (graph_obstacles.h)
    if (not cam_sensor_.complete())
        if (const auto zed = G_->get_node("zed"); zed.has_value())
            if (auto cam = G_->get_camera_api(zed.value()); cam)
            {
                const float fx = cam->get_focal_x(), fy = cam->get_focal_y();
                const float W = static_cast<float>(cam->get_width()), H = static_cast<float>(cam->get_height());
                if (fx > 0.f and W > 0.f and fy > 0.f and H > 0.f and not cam_sensor_.has_vertical())   // once
                {
                    cam_sensor_.hfov_rad = 2.f * std::atan(0.5f * W / fx);
                    cam_sensor_.vfov_rad = 2.f * std::atan(0.5f * H / fy);
                    qInfo() << "[room][camera-cover] zed intrinsics: hfov" << cam_sensor_.hfov_rad * 57.29578f
                            << "deg, vfov" << cam_sensor_.vfov_rad * 57.29578f << "deg";
                }
            }
    if (startup_promoted_ and not cam_sensor_.complete() and not camera_model_warned_)
    {
        camera_model_warned_ = true;
        qWarning() << "[room][camera-cover] camera model INCOMPLETE at promotion (hfov" << cam_sensor_.hfov_rad * 57.29578f
                   << "vfov" << cam_sensor_.vfov_rad * 57.29578f << "deg, height" << cam_sensor_.height_m << "m) — until it"
                   << "completes the room's boundary camera coverage cannot be assessed and afford_room goes to the pose"
                   << "planner on the LiDAR survey alone";
    }
    if (not zed_mount_resolved_ or not cam_sensor_.complete()) return;
    // The boundary in the PUBLISHED frame: the promoted room's polygon, or the survey's ring before promotion (its views
    // are carried onto the promoted outline by Coverage::set_outline).
    auto outline = startup_promoted_ ? published_polygon() : fs_published_;
    if (outline.size() < 3) return;
    cam_cover_.set_outline(outline);
    const Eigen::Affine2f pose_pub = T_pub_int_ * res.robot_pose;
    const Eigen::Vector2f cam = pose_pub * zed_xy_;
    const Eigen::Vector2f fwd = pose_pub.linear() * zed_fwd_;
    cam_cover_.observe(cam, std::atan2(fwd.y(), fwd.x()), cam_sensor_);
}

rc::layout::Plan RoomSceneGraph::plan_camera_cover(const rc::layout::FeasibleGrid& grid, const Eigen::Vector2f& robot,
                                                   const rc::layout::ExplorerParams& ep,
                                                   const rc::layout::Admissible& standable,
                                                   const rc::layout::Admissible& deferred)
{
    // A standpoint is in the internal frame; the coverage is in the published one. The look is scored at its BEST heading
    // (camera_coverage.h best_look) — and that heading is the yaw published with it (plan_layout), which the controller
    // turns to on arrival. (It was the ARRIVAL heading: the L's short leg left the robot facing its end wall, user 2026-11-01.) The zed sits a few cm from the robot origin — ignored here, as the LiDAR
    // gain ignores its own mount; observe() uses the real zed pose. An unreachable best target is excluded and the plan
    // re-run (plan_coverage_reachable): it never ends the phase by itself.
    const Eigen::Vector2f r_pub = T_pub_int_ * robot;
    return rc::layout::plan_coverage_reachable(grid, robot, ep, standable,
        [this, r_pub](const Eigen::Vector2f& v)
        {
            const Eigen::Vector2f v_pub = T_pub_int_ * v;
            return (v_pub - r_pub).squaredNorm() > 1e-9f ? cam_cover_.best_look(v_pub, cam_sensor_).first : 0.f;
        },
        cam_cover_.price(cam_sensor_), deferred);
}

void RoomSceneGraph::latch_camera_phase()
{
    // The camera phase's OWN verdict: plan_coverage's `finished` — no REACHABLE look newly shows one door's worth of wall
    // (unreachable targets were excluded and re-planned; recently refused ones keep it "deferred"). layout_finished_ is
    // the LiDAR survey's and is left alone.
    if (camera_visits0_ < 0) camera_visits0_ = layout_visits_;   // the phase's own visits, not the survey's
    if (not layout_last_plan_.finished or camera_finished_) return;
    camera_finished_ = true;
    float total = 0.f;
    for (const auto& c : cam_cover_.cells()) total += c.len;
    const float price = cam_cover_.price(cam_sensor_), residual = cam_cover_.residual_length(cam_sensor_);
    qInfo().noquote() << QString("[room][camera-cover] boundary camera coverage FINISHED: %1 of %2 m of wall could still "
                                 "be shown to the detector, no reachable look worth one door (%3 m) — afford_room goes "
                                 "to the pose planner")
                             .arg(residual, 0, 'f', 2).arg(total, 0, 'f', 2).arg(price, 0, 'f', 2);
    // ★ LOUD while the detector envelope is uncalibrated: a phase that finishes without a single camera-phase visit while
    //   more than a door's worth of wall was never shown is almost always a room too narrow for the envelope's band
    //   (~2.4-3.1 m for a door with the default envelope), i.e. the model, not the room, ended it.
    if (layout_visits_ == camera_visits0_ and residual > price and not camera_narrow_warned_)
    {
        camera_narrow_warned_ = true;
        qWarning().noquote() << QString("[room][camera-cover] WARNING: camera phase FINISHED WITHOUT A VISIT with %1 m of "
                                        "wall never shown to the door detector (attainable P %2) — the room is called "
                                        "explored but a door there may be undiscovered; the detector envelope is "
                                        "UNCALIBRATED (rc::detect defaults)")
                                    .arg(residual, 0, 'f', 2).arg(cam_cover_.attainable_p(cam_sensor_), 0, 'f', 2);
    }
}

void RoomSceneGraph::remember_camera_cover()
{
    if (const std::string name = room_node_name(); not name.empty() and startup_proto_ and cam_cover_.ready())
        camera_memory_[name] = CameraMemo{cam_cover_, camera_finished_};
}

void RoomSceneGraph::resume_camera_cover(const std::string& room)
{
    // A re-entry: the cameras' record of this room (published frame = the room's own, the one the recall restores).
    // No record = a room this run never surveyed with the cameras (remembered from before): no camera question is
    // reopened — a remembered door there is door_concept's look (door_look.h), not this coverage.
    if (const auto it = camera_memory_.find(room); it != camera_memory_.end())
    {
        cam_cover_ = it->second.cover;
        camera_finished_ = it->second.finished;
        camera_visits0_ = -1;
        if (not camera_finished_)
            qInfo().noquote() << QString("[room][camera-cover] %1 re-entered with its boundary camera coverage UNFINISHED "
                                         "(%2 m still to show the detector): the camera phase resumes")
                                     .arg(QString::fromStdString(room)).arg(cam_cover_.residual_length(cam_sensor_), 0, 'f', 2);
    }
    else
        camera_finished_ = true;
}

}   // namespace rc
