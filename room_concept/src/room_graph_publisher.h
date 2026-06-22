/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

// RoomGraphPublisher — DSR scene-graph writer for room_concept.
//
// Owns everything that turns a RoomConcept::UpdateResult into graph mutations:
// the robot-pose RT edge (world→robot before the room is stable, robot→room
// after), stabilization gating + room/wall/floor node creation, the epistemic
// affordance target, obstacle-footprint feedback to the planner, and the robot
// body-dimension read-back. It is a pure consumer of UpdateResult and holds no
// threading state; SpecificWorker calls update() on fresh localization frames.

#include <cstdint>
#include <functional>
#include <memory>

#include <genericworker.h>                 // DSR API + generated node/attr type tags

#include "room_concept.h"                  // rc::RoomConcept (+ UpdateResult)
#include "epistemic_controller.h"
#include "../../common/affordance_manager/affordance_manager.h"

namespace rc
{

class RoomGraphPublisher
{
public:
    struct Config
    {
        int   stable_frames_required = 30;
        float stable_sdf_mse_max     = 0.06f;
        float stable_cov_tt_max      = 0.001f;
        float room_height            = 2.4f;   // m, room node attribute
        // Robot footprint; defaults overwritten by the graph "body" node when present.
        float robot_width  = 0.460f;
        float robot_length = 0.480f;
        float robot_height = 1.6f;
    };

    struct Deps
    {
        std::shared_ptr<DSR::DSRGraph> graph;
        rc::RoomConcept*               room_concept         = nullptr;
        rc::EpistemicController*       epistemic_controller = nullptr;
        std::function<void()>          trigger_layout;       // GenericWorker::trigger_graph_layout_twopi
    };

    RoomGraphPublisher() = default;

    void init(const Config& cfg, Deps deps);

    // Resolve root/robot ids and read body dimensions (updates the planner footprint).
    void check_init_graph_is_valid();

    // Stabilize → create room / reparent, then publish pose + affordance each frame.
    // adv/side/rot are the latest robot-frame velocities for the RT velocity attrs.
    void update(const rc::RoomConcept::UpdateResult& res, float adv, float side, float rot);

    // Tick the affordance manager (execution monitoring); call periodically.
    void monitor_affordance();

    // Delete room/wall/floor/affordance nodes owned by this agent (start + shutdown).
    void cleanup_room_graph_nodes();

    // Controller peer lost: drop a stale execution claim so the room stays selectable.
    void on_controller_lost();

    [[nodiscard]] bool  room_node_created() const noexcept { return room_node_created_; }
    [[nodiscard]] std::uint64_t robot_id() const noexcept { return dsr_robot_id_; }
    [[nodiscard]] float robot_width()  const noexcept { return cfg_.robot_width; }
    [[nodiscard]] float robot_length() const noexcept { return cfg_.robot_length; }
    [[nodiscard]] float robot_height() const noexcept { return cfg_.robot_height; }

private:
    void dsr_update_pose(const rc::RoomConcept::UpdateResult& res);
    void dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res);
    void dsr_update_affordance(const rc::RoomConcept::UpdateResult& res);
    void update_planner_obstacle_footprints();
    void load_robot_body_dimensions_from_graph();
    void dsr_create_wall_nodes();

    Config cfg_;
    std::shared_ptr<DSR::DSRGraph> G_;
    std::unique_ptr<DSR::RT_API>   rt_api_;
    rc::RoomConcept*               room_concept_ = nullptr;
    rc::EpistemicController*       epistemic_    = nullptr;
    std::function<void()>          trigger_layout_;

    rc::AffordanceManager affordance_manager_{"afford_room"};

    std::uint64_t dsr_robot_id_ = 0;
    std::uint64_t dsr_body_id_  = 0;
    std::uint64_t dsr_world_id_ = 0;
    std::uint64_t dsr_room_id_  = 0;
    bool          room_node_created_ = false;
    int           stable_frames_     = 0;

    // Latest robot-frame velocities, refreshed per update() for the RT velocity attrs.
    float last_adv_  = 0.f;
    float last_side_ = 0.f;
    float last_rot_  = 0.f;
};

}  // namespace rc
