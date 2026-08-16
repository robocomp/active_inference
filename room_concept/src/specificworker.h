/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include "buffer_types.h"
#include "room_concept.h"
#include "svg_room_loader.h"
#include "epistemic_controller.h"
#include "room_scene_graph.h"
#include "lidar_ingestor.h"
#include "room_viewer.h"
#include "room_config.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <limits>
#include <QPointer>
#include <fps/fps.h>
#include "custom_widget.h"
#include "ui_localUI.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 *
 * Runs the SDF-based room localizer (RoomConcept) and publishes the estimated
 * robot pose into the DSR graph as an RT edge on the world→robot (pre-stable)
 * or room→robot (post-stable) relationship.
 */
class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
        ~SpecificWorker();

        void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);
        bool is_shutting_down() const noexcept { return shutting_down_.load(); }

    public slots:
        void initialize();
        void compute();
        void emergency();
        void restore();
        int  startup_check();

        void modify_node_slot(std::uint64_t id, const std::string &type);
        void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
        void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
        void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
        void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
        void del_node_slot(std::uint64_t from){};

        // twopi relayout of the DSR graph viewer; MOVED out of the regenerated genericworker
        // (robocompdsl clobbers hand edits there) — mirror robot_concept. Uses the inherited
        // find_graph_viewer(). Injected into RoomSceneGraph as the relayout callback.
        void trigger_graph_layout_twopi();

    private:
        rc::RoomConfig params;   // worker config; loaded by rc::load_room_config()

        bool startup_check_flag;
        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;

        // ── Velocity / odometry buffers (thread-safe) ──────────────────────────
        rc::VelocityBuffer velocity_buffer_{20};
        rc::OdometryBuffer odometry_buffer_{20};
        // Deeper than the odometry buffer: the IMU runs ~125 Hz against odometry's 10 Hz, and it must
        // still span the gap between two lidar sweeps (50-100 ms, so 6-13 samples) with slack for a
        // late one. 20 entries would not cover two sweeps.
        rc::ImuBuffer imu_buffer_{256};
        // Fitted from the (wall, sim) stamp pairs arriving on every odometry sample; used to put the
        // lidar sweep bounds on the same clock as the rates integrated between them.
        rc::SimClockMap sim_clock_;
        std::int64_t last_imu_sim_ts_ = 0;    // dedup: the graph re-signals on unrelated attribute writes
        std::uint64_t last_robot_ref_speed_timestamp_ = 0;
        std::uint64_t last_robot_current_speed_timestamp_ = 0;
        float last_robot_adv_speed_  = 0.f;   // robot-frame forward velocity (m/s), updated from DSR
        float last_robot_side_speed_ = 0.f;   // robot-frame lateral velocity (m/s)
        float last_robot_rot_speed_  = 0.f;   // robot-frame angular velocity (rad/s)

        std::atomic<bool> pose_saved_{false};
        void save_robot_pose_once();

        // ── LiDAR acquisition (decoupled ingest thread + buffer + health) ──────
        std::unique_ptr<rc::LidarIngestor> lidar_ingestor_;

        // ── LiDAR stream gate on Waiting→Operating ─────────────────────────────
        // Without LiDAR the localizer can never stabilize, so Operating would be a lie. True when the
        // media plane is advertised in the graph; `why` receives the reason when it is not.
        [[nodiscard]] bool lidar_stream_ready(std::string* why = nullptr) const;
        // Operating: has the stream gone silent past LidarStallTimeoutMs? Grace-counted from Operating
        // entry so a normal warm-up (subscriber discovery + first sweep) is not mistaken for a stall.
        [[nodiscard]] bool lidar_stream_stalled(std::int64_t* age_ms_out = nullptr) const;
        std::int64_t operating_since_ms_    = 0;   // wall clock of the last Operating entry
        std::int64_t last_wait_log_ms_      = 0;   // throttles the "still waiting" line
        bool         lidar_stall_reported_  = false; // one presenceLost per stall episode
        bool         degraded_from_lidar_   = false; // makes the Degraded log name the real reason

        // ── Compute-loop pacing / timing telemetry (worker-owned) ──────────────
        std::atomic<bool> operating_compute_queued_{false};
        std::int64_t      last_affordance_monitor_ms_ = 0;
        std::int64_t      last_dsr_publish_try_ms_    = 0;
        std::int64_t      last_compute_timing_log_ms_ = 0;
        std::string pose_file_path() const;

        // Publish the latest corrected pose (robot↔room RT) to the DSR graph if it is fresh. Called
        // both from compute() and — the instant the localizer produces a result — from a
        // Qt::QueuedConnection posted by room_concept_'s on_result_ready callback. Both run on the MAIN
        // thread (the queued hop marshals the localizer-thread trigger), so the publish bookkeeping is
        // race-free and the timestamp dedup makes whichever path fires second a no-op. Returns true iff
        // it actually published this call.
        bool maybe_publish_corrected_pose();

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        // Room contour AS HANDED TO THE LOCALIZER (already recentred when RECENTER_ROOM_POLYGON).
        // The viewer/camera overlay must reuse THIS, not re-load the SVG, or it would draw the
        // outline in the un-shifted frame.
        std::vector<Eigen::Vector2f> room_polygon_;
        // Old-frame coordinates of the new origin; zero when no recentring was applied.
        Eigen::Vector2f room_polygon_offset_ = Eigen::Vector2f::Zero();
        void initialize_room_model_from_svg();
        void save_robot_pose_on_exit() const;

        // ── Epistemic controller ────────────────────────────────────────────────
        rc::EpistemicController epistemic_controller_;
        bool self_target_active_ = false;

        // ── GUI / visualization (2-D viewer, plots, camera projection window) ───
        std::unique_ptr<rc::RoomViewer> viewer_;
        FPSCounter fps_counter_;

        void request_shutdown();
        void cleanup_owned_nodes();
        void waiting_enter();
        void waiting_loop();
        void operating_enter();
        void operating_loop();
        void degraded_enter();
        void degraded_loop();
        void on_optional_peer_lost(const std::string &name, std::uint32_t id);
        void on_optional_peer_ready(const std::string &name, std::uint32_t id);
        void cleanup_self_agent_node();

        // ── DSR scene-graph writer (robot-pose RT, room/wall/affordance nodes) ──
        std::unique_ptr<DSR::RT_API>            rt_api_;          // worker-owned, injected
        std::unique_ptr<rc::RoomSceneGraph> scene_graph_;
        std::int64_t last_dsr_published_ts_ms_ = 0;

        // Kinematic clamp state (see the clamp block in maybe_publish_corrected_pose). The clamp is
        // relative to what was actually PUBLISHED last, not to the optimizer's previous output, because
        // the invariant it enforces is about the stream consumers see.
        std::optional<Eigen::Affine2f> last_published_pose_;
        std::int64_t                   last_published_ts_ms_ = 0;
        long                           pose_clamp_hits_      = 0;


        // Pose trace CSV (etc/pose_trace.csv): logs CORRECTED (20 Hz, compute) and PREDICTED (60 Hz,
        // tick) poses with timestamps so the intermediate dead-reckoned poses can be compared against
        // the optimizer corrections (diagnose the noise predict-publish injects). Both writers run on
        // this->thread() → single ofstream, no lock. type: 0=corrected, 1=predicted.
        std::ofstream pose_trace_;
        bool          pose_trace_open_attempted_ = false;
        void log_pose_trace(int type, std::int64_t valid_ts_ms,
                            const Eigen::Affine2f& pose, float innov_norm);

        // Per-tick compute-timing CSV (etc/compute_timing.csv): exposes WHERE compute() stalls (viewer
        // vs dsr vs loc_fetch) so we can see why the corrected publish drops below the optimizer rate.
        std::ofstream compute_csv_;
        bool          compute_csv_open_attempted_ = false;

        // RT publish-rate monitor (shown in the window title at ~1 Hz so it can be watched visually).
        int          rt_corr_count_           = 0;   // corrected RT publishes this window
        std::int64_t rt_rate_window_start_ms_ = 0;
        void update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread);

        std::atomic<bool> shutting_down_{false};

    signals:
        void presenceReady();
        void presenceLost();
};

#endif // SPECIFICWORKER_H
