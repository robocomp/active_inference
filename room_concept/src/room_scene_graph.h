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
#include <print>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <genericworker.h>                 // DSR API + generated node/attr type tags

#include "room_concept.h"                  // rc::RoomConcept (+ UpdateResult)
#include "epistemic_controller.h"
#include "room_config.h"             // rc::RoomConfig (shared config)
#include "object_anchor_source.h"    // rc::ObjectAnchorSource (validated objects → SE(2) landmarks)
#include "../../common/affordance_manager/affordance_manager.h"
#include "calib_channel.h"           // rc::calib::CalibChannel (the afford_calib producer)

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
          trigger_layout_(trigger_layout ? std::move(trigger_layout) : [] {})
    {
        // ★TWO SWITCHES GUARDED THIS FEATURE AND ONLY ONE WAS WIRED. RoomConcept.CalibPivotEnabled
        // gates the active half here, but CalibChannel has its OWN `enabled` -- deliberately, as a
        // second safety catch on something that had never driven a robot -- and nothing ever set it
        // from the config. So flipping the TOML flag produced no offer, no afford_calib node, and no
        // error: calib_.offer() simply returned nullopt for ever, which is indistinguishable from
        // "the robot is well calibrated and the pivot lost every contest". Observed 2026-08-23: only
        // afford_room in the graph with the flag on.
        rc::calib::CalibChannelParams cp;
        cp.enabled = params.CALIB_PIVOT_ENABLED;
        cp.forced_gain_nats = params.CALIB_FORCED_GAIN_NATS;
        calib_ = rc::calib::CalibChannel(cp);
        // Say it once, loudly, at construction. A forced price is invisible from every other vantage
        // point -- the offer looks like any other offer -- and the one thing that must not happen is
        // a run being graded on it weeks later because nobody remembered the flag was on.
        if (cp.forced_gain_nats > 0.0)
            std::print("[calib] ★TESTING: the advertised gain is FORCED to {:.3f} nats. afford_calib "
                       "will win contests it has not earned; the true valuation is logged beside it "
                       "and is the only one that means anything.\n", cp.forced_gain_nats);
    }

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
    // ── THE SECOND AFFORDANCE THIS AGENT PUBLISHES ──────────────────────────────────────────────
    // afford_calib: an Orient contract asking the robot to turn 120 degrees on the spot, offered
    // twelve times, which is four complete turns back to the heading it started from. That closure is
    // the measurement — the robot turned exactly 8*pi radians, as a fact about turning, so comparing
    // the odometry's own accumulation against it needs no map, no survey and no localiser.
    // ★It never drives. The offer goes out at wherever the robot happens to be, and if the consumer
    // says the body cannot sweep its diagonal there it is believed and the offer stops until ordinary
    // work has carried the robot somewhere with room. That is the whole design: the calibration is a
    // passenger on the day's driving, not a trip of its own.
    void dsr_update_calibration(const rc::RoomConcept::UpdateResult& res);
    /// Create afford_calib with its contract already on it and NOT on offer. See the definition: the
    /// consumer latches a contract once per node id, and this node is reused for every step.
    bool ensure_calib_node();
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
    // Graph-signal entry point: an attribute changed on our affordance node. Runs on the MAIN thread
    // (the connection is Queued — never DirectConnection, see CLAUDE.md) and only latches; the
    // compute loop still consumes the event, so ordering is unchanged.
public:
    void on_affordance_attr_changed(std::uint64_t node_id);
private:
    rc::EpistemicController*       epistemic_    = nullptr;

    // Affordance-outcome instrumentation (mirrored into the localiser CSV each cycle).
    // ★ The COUNTER, not the code, is what locates an event: the code is a LEVEL that persists for
    // thousands of rows after its one completion, so segmenting on the code measures dwell time
    // rather than events. Segment on aff_completions changing.
    // ★ THE PUBLISHED target, not the planner's internal one. They are NOT the same and the
    // difference is the whole bug: publish_target silently DECLINES when the node is Completed and
    // the proposal is unchanged, so the planner can be selecting a new cell every cycle while the
    // node still carries the one the robot is already standing on. Logging current_target() showed
    // the planner busy and healthy and told us nothing about what the controller was handed.
    float pub_tx_ = std::numeric_limits<float>::quiet_NaN();
    float pub_ty_ = std::numeric_limits<float>::quiet_NaN();
    bool  pub_ok_ = false;   // did the last publish_target actually re-arm the node?
    // ★THE TARGET ACTUALLY ON THE NODE, as distinct from the last one PROPOSED: pub_tx_/pub_ty_ are
    // written on every publish ATTEMPT, declined ones included, so they track what the planner wanted
    // rather than what any consumer was offered. A completion is evidence about the ARMED cell only.
    float armed_tx_ = std::numeric_limits<float>::quiet_NaN();
    float armed_ty_ = std::numeric_limits<float>::quiet_NaN();
    // ★ONE RETIREMENT PER ARMING. The level-triggered check reads a LEVEL ("not pending"), which stays
    // true until something re-arms, so without this it fires every cycle at ~20 Hz. Cleared on arming.
    bool  armed_retired_ = false;
    // ★★★HAVE WE ACTUALLY SEEN THIS ARMING GO LIVE? The level-triggered check below reads a LEVEL
    // ("not active and not pending" = Completed), and immediately after publish_target arms the node
    // that read can still return the PRE-ARM value — which is indistinguishable from a completion.
    // Live 2026-08-19: room armed the node to Offered and retired its own fresh offer on the next
    // cycle, over and over, so the consumer never saw anything but JustCompleted and froze with
    // cmd_adv = cmd_rot = 0.000 while valid cells sat on the wire.
    // ★So require the node to have been OBSERVED live (Offered or Executing) since it was armed before
    // a Completed reading may be believed. That turns a level into a genuine edge and costs nothing:
    // a real completion is always preceded by the node having been live.
    bool  armed_seen_live_ = false;
    // ★★★AN OFFER NEEDS A CLOCK. Room had two ways to move on from a target: the consumer completes it,
    // or break_execution_stall fires — and the latter only runs while the node is EXECUTING. So an offer
    // the consumer never CLAIMS left room with no path to re-select at all, and it held one cell while
    // the robot stood still. Measured 2026-08-19 over 44.6 min: interval between new targets p50 3.1 s,
    // p90 22.3 s, MAX 100.6 s — and every gap over 30 s had outcome=Refused with the robot moving less
    // than a metre. (Gaps with real travel, 7.6 m / 6.7 m, are a long drive and are fine.)
    // ★This is the same defect as everything else today: a wait with no temporal term. An offer the
    // consumer has not taken within a few seconds is one it is declining — it runs at 20 Hz, so seconds
    // are hundreds of chances — and the right response is to offer somewhere else, not to wait.
    std::uint64_t armed_at_ms_ = 0;
    long aff_completions_   = 0;   // completed affordances this run
    int  last_outcome_code_ = 0;   // 0 none · 1 satisfied · 2 timeout · 3 refused · 4 abandoned
    std::function<void()>          trigger_layout_;

    rc::AffordanceManager affordance_manager_{"afford_room"};
    // ── afford_calib: same protocol, second register, its own state ─────────────────────────────
    // ★A SEPARATE NODE, NOT A SECOND USE OF afford_room. Every standpoint this agent has ever offered
    // rides one node, and the ABA hazard that comes with reusing a register is exactly what the epoch
    // model exists to bound — putting a manoeuvre with completely different semantics on the same
    // register would make "did the consumer answer THIS offer" undecidable from the wire.
    rc::AffordanceManager calib_manager_{"afford_calib"};
    rc::calib::CalibChannel calib_;
    bool          calib_contract_written_ = false;
    std::uint64_t calib_armed_at_ms_      = 0;
    float         calib_bearing_rad_      = 0.f;
    double        calib_last_t_s_         = 0.0;
    int           calib_dbg_              = 0;
    std::ofstream calib_log_;
    bool          calib_log_open_ = false;
    std::string   calib_last_reason_;
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
    // Throttle for the epoch-disagreement report. Counts cycles, and says so — the last counter on
    // this pair conflated a duration with a count and read as 2407 separate events.
    int exec_stale_epoch_reports_ = 0;
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
