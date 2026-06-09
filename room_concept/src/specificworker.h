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
#include "viewer_2d.h"
#include "timeseries_plot.h"
#include "camera_visualizer.h"
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
        struct Params
        {
            float ROBOT_WIDTH  = 0.460f;   // m
            float ROBOT_LENGTH = 0.480f;   // m
            float ROBOT_HEIGHT = 1.6f;     // m, obstacle cloud ceiling

            // Lidar
            std::string LIDAR_NAME = "lidar3D";
            float MAX_LIDAR_HIGH_RANGE      = 100.f;  // m
            int   LIDAR_LOW_DECIMATION_FACTOR = 1;
            float LIDAR_HIGH_MIN_HEIGHT     = 1.5f;   // m
            float LIDAR_HIGH_MAX_HEIGHT     = 2.0f;   // m
            float LIDAR_HIGH_FLOOR_HEIGHT   = 0.15f;  // m

            // View
            QRectF GRID_MAX_DIM{-5, -5, 10, 10};
            int   MAX_LIDAR_DRAW_POINTS  = 500;

            // Localizer
            bool  PREDICTION_EARLY_EXIT  = true;
            std::string OptimizerType    = "LBFGS";
            float ODOMETRY_NOISE_FACTOR  = 0.0f;

            // DSR stabilization: require these many consecutive "stable" frames before
            // creating the room node and re-parenting robot under it.
            int   STABLE_FRAMES_REQUIRED = 30;
            float STABLE_SDF_MSE_MAX     = 0.06f;   // sdf_mse must be below this
            float STABLE_COV_TT_MAX      = 0.001f;  // angular covariance must be below this

            // Room height for DSR node attribute
            float room_height = 2.4f;  // meters

            // Debug bootstrap table hanging from the room node
            bool  BOOTSTRAP_TABLE_ENABLED = true;
            float BOOTSTRAP_TABLE_X = 0.f;
            float BOOTSTRAP_TABLE_Y = 0.f;
            float BOOTSTRAP_TABLE_YAW = 0.f;
            float BOOTSTRAP_TABLE_WIDTH = 1.5f;
            float BOOTSTRAP_TABLE_DEPTH = 1.4f;
            float BOOTSTRAP_TABLE_HEIGHT = 0.74f;

            // Media plane (zero-copy DDS RGBD transport)
            int         MEDIA_DOMAIN_ID = 0;
            std::string MEDIA_RGB_TOPIC = "rc/zed/rgb";
        };
        Params params;

        bool startup_check_flag;
        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;

        // ── Velocity / odometry buffers (thread-safe) ──────────────────────────
        rc::HighLidarBuffer high_lidar_buffer_{3};
        rc::VelocityBuffer velocity_buffer_{20};
        rc::OdometryBuffer odometry_buffer_{20};
        std::uint64_t last_robot_ref_speed_timestamp_ = 0;
        std::uint64_t last_robot_current_speed_timestamp_ = 0;
        float last_robot_adv_speed_  = 0.f;   // robot-frame forward velocity (m/s), updated from DSR
        float last_robot_side_speed_ = 0.f;   // robot-frame lateral velocity (m/s)
        float last_robot_rot_speed_  = 0.f;   // robot-frame angular velocity (rad/s)

        std::atomic<bool>      pose_saved_{false};
        std::optional<rc::LidarData> read_lidar_from_graph() const;
        void save_robot_pose_once();

        // ── Lidar/state health telemetry ─────────────────────────────────────
        std::atomic<std::uint64_t> lidar_signal_events_{0};
        std::atomic<std::uint64_t> lidar_copied_frames_{0};
        std::atomic<std::uint64_t> lidar_ingested_frames_{0};
        std::atomic<std::uint64_t> lidar_watchdog_pulls_{0};
        std::atomic<std::uint64_t> lidar_ingest_wakeups_{0};
        std::atomic<std::uint64_t> lidar_copy_mutex_busy_{0};
        std::atomic<std::uint64_t> lidar_ingest_loop_beats_{0};
        std::atomic<bool>          lidar_copy_queued_{false};
        std::atomic<bool>          operating_compute_queued_{false};
        std::atomic<std::int64_t>  lidar_last_signal_ms_{0};
        std::atomic<std::int64_t>  lidar_last_ingest_ms_{0};
        std::atomic<std::int64_t>  lidar_last_ingest_loop_beat_ms_{0};
        std::atomic<bool>          lidar_notify_inflight_{false};
        std::atomic<std::int64_t>  lidar_notify_inflight_since_ms_{0};
        std::int64_t               last_lidar_health_log_ms_ = 0;
        std::int64_t               last_lidar_stall_dump_ms_ = 0;
        std::int64_t               last_lidar_watchdog_pull_ms_ = 0;
        std::int64_t               last_waiting_heartbeat_log_ms_ = 0;
        std::int64_t               last_degraded_heartbeat_log_ms_ = 0;
        std::int64_t               last_affordance_monitor_ms_ = 0;
        std::int64_t               last_dsr_publish_try_ms_ = 0;
        std::int64_t               last_compute_timing_log_ms_ = 0;
        std::deque<std::int64_t>   lidar_recent_source_ts_;

        // ── Lidar ingestion thread (decoupled from the Qt/GUI thread) ──────────
        // The heavy point-cloud decode used to run inside modify_node_slot, a queued
        // Qt slot on the GUI thread. Under rendering load that thread starved and the
        // localization inner loop stopped receiving fresh lidar. This dedicated thread
        // does the decode + buffer fill + notify independently, so the inner loop runs
        // at full rate regardless of GUI load. It is woken by a DirectConnection trigger
        // on the DDS thread, with a periodic timeout fallback.
        std::thread             lidar_ingest_thread_;
        std::atomic<bool>       lidar_ingest_running_{false};
        std::atomic<bool>       lidar_ingest_dirty_{false};
        std::mutex              lidar_ingest_mutex_;
        std::condition_variable lidar_ingest_cv_;
        std::int64_t            last_ingested_lidar_ts_ = std::numeric_limits<std::int64_t>::min();
        struct PendingLidarScan
        {
            std::vector<float> xs;
            std::vector<float> ys;
            std::vector<float> zs;
            std::int64_t source_ts = std::numeric_limits<std::int64_t>::min();
            bool has_data = false;
        };
        PendingLidarScan pending_lidar_scan_;
        void lidar_ingest_loop();
        void ingest_latest_lidar();
        void schedule_lidar_copy_to_pending(std::optional<std::uint64_t> node_id, const char *origin, bool count_signal_event);
        bool copy_latest_lidar_to_pending(std::optional<std::uint64_t> node_id, const char *origin, bool count_signal_event);
        std::string pose_file_path() const;

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        void initialize_room_model_from_svg();
        void save_robot_pose_on_exit() const;

        // ── Epistemic controller ────────────────────────────────────────────────
        rc::EpistemicController epistemic_controller_;
        bool self_target_active_ = false;

        // ── 2-D viewer ─────────────────────────────────────────────────────────
        QPointer<Custom_widget> custom_widget_;
        QPointer<rc::Viewer2D> viewer_2d_;
        QPointer<rc::TimeSeriesPlot> ts_plot_sdf_;
        QPointer<rc::TimeSeriesPlot> ts_plot_fe_;
        std::unique_ptr<rc::CameraVisualizer> camera_viz_;
        bool camera_media_plane_initialized_ = false;
        FPSCounter fps_counter_;
        Eigen::Affine2f best_available_pose(const std::optional<rc::RoomConcept::UpdateResult>&, bool) const;
        void update_epistemic_overlay();
        void update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res);

        // ── Mouse-driven pose reset (Shift+Left = translate, Ctrl+Left = rotate) ──
        void slot_mouse_translate(QPointF scene_pos);
        void slot_mouse_rotate(QPointF scene_pos);
        void slot_show_camera_visualization();
        void slot_toggle_lidar_points_display(bool checked);

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

        // ── DSR graph state ────────────────────────────────────────────────────
        uint64_t dsr_robot_id_ = 0;
        uint64_t dsr_body_id_  = 0;
        uint64_t dsr_world_id_ = 0;
        uint64_t dsr_room_id_  = 0;
        bool     room_node_created_ = false;
        rc::AffordanceManager affordance_manager_{"afford_room"};
        int      stable_frames_     = 0;
        std::int64_t last_dsr_published_ts_ms_ = 0;
        void check_init_graph_is_valid();
        void cleanup_room_graph_nodes();    // delete "affordance" nodes hanging from room via "has" edges
        void load_robot_body_dimensions_from_graph();
        void update_dsr(const rc::RoomConcept::UpdateResult& res);
        void dsr_update_pose(const rc::RoomConcept::UpdateResult& res);
        void dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res);
        void dsr_create_wall_nodes();
        void update_planner_obstacle_footprints();
        void dsr_update_affordance(const rc::RoomConcept::UpdateResult& res);
        std::unique_ptr<DSR::RT_API> rt_api;
        std::atomic<bool> shutting_down_{false};

    signals:
        void presenceReady();
        void presenceLost();
};

#endif // SPECIFICWORKER_H
