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

// RoomSceneGraph — DSR scene-graph writer for room_concept.
//
// Owns everything that turns a RoomConcept::UpdateResult into graph mutations:
// the robot-pose RT edge (world→robot before the room is stable, robot→room
// after), stabilization gating + room/wall/floor node creation, the epistemic
// affordance target, obstacle-footprint feedback to the planner, and the robot
// body-dimension read-back. It is a pure consumer of UpdateResult and holds no
// threading state; SpecificWorker calls update() on fresh localization frames.

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <genericworker.h>                 // DSR API + generated node/attr type tags

#include "room_concept.h"                  // rc::RoomConcept (+ UpdateResult)
#include "epistemic_controller.h"
#include "room_config.h"             // rc::RoomConfig (shared config)
#include "object_anchor_source.h"    // rc::ObjectAnchorSource (validated objects → SE(2) landmarks)
#include "../../common/affordance_manager/affordance_manager.h"

namespace rc
{

class RoomSceneGraph
{
public:
    // Constructor injection: graph + the worker-owned rt_api + shared params + the
    // localizer/planner + the graph-layout trigger. Fully constructed == ready.
    RoomSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                       DSR::RT_API*             rt_api,
                       rc::RoomConfig&     params,
                       rc::RoomConcept&         room_concept,
                       rc::EpistemicController& epistemic,
                       std::function<void()>    trigger_layout)
        : G_(std::move(graph)), rt_api_(rt_api), params_(&params),
          room_concept_(&room_concept), epistemic_(&epistemic),
          trigger_layout_(trigger_layout ? std::move(trigger_layout) : [] {}) {}

    // Resolve root/robot ids and read body dimensions (updates the planner footprint).
    void check_init_graph_is_valid();

    // Stabilize → create room / reparent, then publish pose + affordance each frame.
    // adv/side/rot are the latest robot-frame velocities for the RT velocity attrs.
    // write_rt=false ⇒ do room creation/affordance but DON'T write the robot↔room RT edge (the
    // odometry-driven complementary-filter publisher owns it; avoids past-stamped block interleaving).
    void update(const rc::RoomConcept::UpdateResult& res, float adv, float side, float rot,
                bool write_rt = true);

    // Tick the affordance manager (execution monitoring); call periodically.
    void monitor_affordance();

    // Delete room/wall/floor/affordance nodes owned by this agent (start + shutdown).
    void cleanup_room_graph_nodes();

    // Controller peer lost: drop a stale execution claim so the room stays selectable.
    void on_controller_lost();

    // High-rate predicted pose between lidar corrections (dead-reckoned room←robot estimate + grown
    // cov + forward validity stamp). Writes ONLY the robot↔room RT edge — no room creation/affordance.
    void dsr_publish_predicted_pose(const Eigen::Affine2f& robot_pose,
                                    const Eigen::Matrix3f& covariance,
                                    std::uint64_t timestamp_ms);

    [[nodiscard]] bool  room_node_created() const noexcept { return room_node_created_; }
    // Consecutive "stable" localization frames accumulated so far (resets to 0 on any unstable frame,
    // on relocalization and on room deletion). Drives the UI stabilization readout; meaningless once
    // room_node_created() is true, since the counter stops being advanced then.
    [[nodiscard]] int   stable_frames() const noexcept { return stable_frames_; }
    [[nodiscard]] std::uint64_t robot_id() const noexcept { return dsr_robot_id_; }

    // World-frame (room) positions of the pinned-object landmarks, refreshed each frame by
    // refresh_object_anchors(). Consumed by the viewer to draw robot→landmark sight lines.
    [[nodiscard]] const std::vector<Eigen::Vector2f>& pinned_landmarks() const noexcept
    { return latest_pinned_landmarks_; }

    // Per-landmark "currently being measured" flag (1:1 with pinned_landmarks()): true when the
    // object's producer is actively detecting it this frame (e.g. table_detection_alive). The viewer
    // draws the robot→landmark sight line only while measured.
    [[nodiscard]] const std::vector<char>& pinned_measured() const noexcept
    { return latest_pinned_measured_; }

private:
    void dsr_update_pose(const rc::RoomConcept::UpdateResult& res);
    void write_robot_room_rt(const Eigen::Affine2f& robot_pose,
                             const Eigen::Matrix3f& covariance,
                             std::uint64_t timestamp_ms);
    void dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res);
    void dsr_update_affordance(const rc::RoomConcept::UpdateResult& res);
    /// Liveness watchdog for a claimed-but-unreachable afford_room target. While the controller
    /// holds the execution claim the planner is deliberately idle, so if the controller can never
    /// get there (boxed in by an obstacle it cannot clear) NOTHING re-offers and the robot works
    /// its stuck-recovery forever — the run just stops making progress. This tracks the closest
    /// approach to the claimed target and, if it stops improving for long enough, releases the
    /// claim and stamps the target as visited so a different one is picked.
    ///
    /// NOTE (CLAUDE.md "no thresholds"): the no-progress window IS a threshold, deliberately. It
    /// guards LIVENESS of the offer/claim/complete handshake, not a modelling decision — there is
    /// no generative quantity here to make it fall out of, because the room agent cannot observe
    /// why the controller is not moving. Kept out of the belief entirely: the only thing it feeds
    /// back into the model is `mark_target_finished`, i.e. ordinary IoR evidence.
    /// Returns true if the claim was broken this cycle.
    bool break_execution_stall(const Eigen::Vector2f& robot_pos);
    void update_planner_obstacle_footprints();
    // Gather validated modelled objects from the graph (MAIN thread) and hand them to the
    // localizer as SE(2) pose landmarks.  No-op unless params_->ObjectAnchorEnable.
    void refresh_object_anchors();
    // Observe-only: log each detected table's ROOM-frame pose + ROBOT-frame observation + the robot pose
    // to stdout (throttled) and a CSV. Pure graph reads (no pin/anchor side effects); runs regardless of
    // the anchor factor being enabled, so the table-landmark behaviour can be studied as the robot moves.
    void log_table_landmarks();
    void load_robot_body_dimensions_from_graph();
    void dsr_create_wall_nodes();

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*                   rt_api_       = nullptr;   // worker-owned, injected
    rc::RoomConfig*           params_       = nullptr;   // shared config (read + body-dim write-back)
    rc::RoomConcept*               room_concept_ = nullptr;
    rc::EpistemicController*       epistemic_    = nullptr;

    // Affordance-outcome instrumentation (mirrored into the localiser CSV each cycle).
    // ★ The COUNTER, not the code, is what locates an event: the code is a LEVEL that persists for
    // thousands of rows after its one completion, so segmenting on the code measures dwell time
    // rather than events. Segment on aff_completions changing.
    long aff_completions_   = 0;   // completed affordances this run
    int  last_outcome_code_ = 0;   // 0 none · 1 satisfied · 2 timeout · 3 refused · 4 abandoned
    std::function<void()>          trigger_layout_;

    rc::AffordanceManager affordance_manager_{"afford_room"};
    rc::ObjectAnchorSource object_anchor_source_;
    // Chain-propagated pose+covariance reader for object anchors. Created lazily on the MAIN
    // thread (owns an RT_API whose ts==0 path uses the InnerEigen cache — see CLAUDE.md).
    std::unique_ptr<DSR::InnerGaussianAPI> inner_gaussian_;

    // ---- Execution-stall watchdog state (see break_execution_stall) ----
    // Closest the robot has come to the currently-claimed target, and when that best was last
    // improved. Reset whenever no claim is held or the claimed target moves.
    float                                 stall_best_dist_    = std::numeric_limits<float>::infinity();
    Eigen::Vector2f                       stall_target_{0.f, 0.f};
    bool                                  stall_tracking_     = false;
    std::chrono::steady_clock::time_point stall_last_progress_{};
    int                                   exec_hold_cycles_   = 0;   // THIS episode only
    int                                   publish_dbg_        = 0;   // throttle for the publish trace
    int                                   no_target_dbg_      = 0;   // throttle for the no-target trace

    std::uint64_t dsr_robot_id_ = 0;
    std::uint64_t dsr_body_id_  = 0;
    std::uint64_t dsr_world_id_ = 0;
    std::uint64_t dsr_room_id_  = 0;
    bool          room_node_created_ = false;
    int           stable_frames_     = 0;
    // Sustained-stability counter for the object-anchor PIN guard. Distinct from stable_frames_, which
    // only runs BEFORE the room node exists (bootstrap) and is reset by every re-anchor: pinning happens
    // in the steady state, long after that counter has stopped being maintained.
    int           anchor_stable_frames_ = 0;

    // World-frame positions of pinned-object landmarks (refreshed by refresh_object_anchors()).
    std::vector<Eigen::Vector2f> latest_pinned_landmarks_;
    // Parallel to latest_pinned_landmarks_: is that object being actively measured this frame?
    std::vector<char>            latest_pinned_measured_;

    // Latest robot-frame velocities, refreshed per update() for the RT velocity attrs.
    float last_adv_  = 0.f;
    float last_side_ = 0.f;
    float last_rot_  = 0.f;

    // Table-landmark tracking CSV (tmp/sdf_localizer/table_landmark_<ts>.csv): one row per detected table
    // per refresh, logging its ROOM-frame coord (stable if localization is consistent) + ROBOT-frame
    // observation + the robot pose, so the evolution of the detected table coords vs robot motion is
    // analysable offline.
    std::ofstream table_landmark_csv_;
    bool          table_landmark_csv_open_attempted_ = false;
    std::uint64_t table_landmark_log_k_ = 0;   // throttles the stdout line (CSV gets every frame)
};

}  // namespace rc
