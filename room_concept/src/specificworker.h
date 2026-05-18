/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
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
#include <atomic>
#include <thread>
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

        void FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose);
        void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);

    public slots:
        void initialize();
        void compute();
        void emergency();
        void restore();
        int  startup_check();

        void modify_node_slot(std::uint64_t id, const std::string &type);
        void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
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
            float ODOMETRY_NOISE_FACTOR  = 1.0f;

            // DSR stabilization: require these many consecutive "stable" frames before
            // creating the room node and re-parenting robot under it.
            int   STABLE_FRAMES_REQUIRED = 30;
            float STABLE_SDF_MSE_MAX     = 0.06f;   // sdf_mse must be below this
            float STABLE_COV_TT_MAX      = 0.001f;  // angular covariance must be below this

            // Room height for DSR node attribute
            float room_height = 2.4f;  // meters
        };
        Params params;

        bool startup_check_flag;

        // ── Velocity / odometry buffers (thread-safe) ──────────────────────────
        rc::HighLidarBuffer high_lidar_buffer_{3};
        rc::VelocityBuffer velocity_buffer_{20};
        rc::OdometryBuffer odometry_buffer_{20};

        // ── Lidar reader thread ─────────────────────────────────────────────────
        std::thread            read_lidar_th;
        std::atomic<bool>      stop_lidar_thread{false};
        std::atomic<bool>      pose_saved_{false};
        void read_lidar();
        std::optional<rc::LidarData> read_lidar_from_graph() const;
        void save_robot_pose_once();
        std::string pose_file_path() const;

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        int  rfe_saved_window_size_ = 10;
        void initialize_room_model_from_svg();
        void save_robot_pose_on_exit() const;

        // ── Epistemic controller ────────────────────────────────────────────────
        rc::EpistemicController epistemic_controller_;
        bool self_target_active_ = false;
        rc::EpistemicController::ControlCommand prev_cmd_{};
        std::chrono::steady_clock::time_point prev_cmd_time_ = std::chrono::steady_clock::now();
        float max_lin_accel_ = 1.5f;
        float max_rot_accel_ = 3.0f;
        void navigate_to_target(const std::optional<rc::RoomConcept::UpdateResult>& loc_res,
                                const std::optional<rc::ObstacleData>& obstacles);

        // ── 2-D viewer ─────────────────────────────────────────────────────────
        Custom_widget custom_widget;
        std::unique_ptr<rc::Viewer2D> viewer_2d_;
        rc::TimeSeriesPlot* ts_plot_sdf_ = nullptr;
        rc::TimeSeriesPlot* ts_plot_fe_  = nullptr;
        std::unique_ptr<rc::CameraVisualizer> camera_viz_;
        FPSCounter fps_counter_;
        Eigen::Affine2f best_available_pose(const std::optional<rc::RoomConcept::UpdateResult>&, bool) const;
        void update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res,
                    const Eigen::Affine2f& pose_for_draw);

        // ── Mouse-driven pose reset (Shift+Left = translate, Ctrl+Left = rotate) ──
        void slot_mouse_translate(QPointF scene_pos);
        void slot_mouse_rotate(QPointF scene_pos);
        void slot_show_camera_visualization();

        // ── DSR graph state ────────────────────────────────────────────────────
        uint64_t dsr_robot_id_ = 0;
        uint64_t dsr_world_id_ = 0;
        uint64_t dsr_room_id_  = 0;
        bool     room_node_created_ = false;
        int      stable_frames_     = 0;
        void check_init_graph_is_valid();
        void update_dsr(const rc::RoomConcept::UpdateResult& res);
        void dsr_update_pose(const rc::RoomConcept::UpdateResult& res);
        void dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res);
        std::unique_ptr<DSR::RT_API> rt_api;

    signals:
        //void customSignal();
};

#endif // SPECIFICWORKER_H
