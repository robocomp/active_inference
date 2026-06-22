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
#include "room_graph_publisher.h"
#include "lidar_ingestor.h"
#include "room_view_controller.h"
#include "room_agent_params.h"
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

    private:
        rc::RoomAgentParams params;   // worker config; loaded by rc::load_room_config()

        bool startup_check_flag;
        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;

        // ── Velocity / odometry buffers (thread-safe) ──────────────────────────
        rc::VelocityBuffer velocity_buffer_{20};
        rc::OdometryBuffer odometry_buffer_{20};
        std::uint64_t last_robot_ref_speed_timestamp_ = 0;
        std::uint64_t last_robot_current_speed_timestamp_ = 0;
        float last_robot_adv_speed_  = 0.f;   // robot-frame forward velocity (m/s), updated from DSR
        float last_robot_side_speed_ = 0.f;   // robot-frame lateral velocity (m/s)
        float last_robot_rot_speed_  = 0.f;   // robot-frame angular velocity (rad/s)

        std::atomic<bool> pose_saved_{false};
        void save_robot_pose_once();

        // ── LiDAR acquisition (decoupled ingest thread + buffer + health) ──────
        rc::LidarIngestor lidar_ingestor_;

        // ── Compute-loop pacing / timing telemetry (worker-owned) ──────────────
        std::atomic<bool> operating_compute_queued_{false};
        std::int64_t      last_affordance_monitor_ms_ = 0;
        std::int64_t      last_dsr_publish_try_ms_    = 0;
        std::int64_t      last_compute_timing_log_ms_ = 0;
        std::string pose_file_path() const;

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        void initialize_room_model_from_svg();
        void save_robot_pose_on_exit() const;

        // ── Epistemic controller ────────────────────────────────────────────────
        rc::EpistemicController epistemic_controller_;
        bool self_target_active_ = false;

        // ── GUI / visualization (2-D viewer, plots, camera projection window) ───
        rc::RoomViewController view_;
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
        rc::RoomGraphPublisher graph_publisher_;
        std::int64_t last_dsr_published_ts_ms_ = 0;
        std::atomic<bool> shutting_down_{false};

    signals:
        void presenceReady();
        void presenceLost();
};

#endif // SPECIFICWORKER_H
