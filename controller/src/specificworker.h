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
#include <Eigen/Dense>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "room_path_planner.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "trajectory_controller.h"
#include "viewer_2d.h"
#include "../../common/agent_presence_monitor/agent_presence_monitor.h"

class Custom_widget;
class FPSCounter;

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	void modify_node_slot(std::uint64_t, const std::string &type);
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type);
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
private:
	using Polygon = std::vector<Eigen::Vector2f>;
	using Polygons = std::vector<Polygon>;
	using PathPlan = RoomPathPlanner::PathPlan;
	static constexpr const char *kRobotRefAdvSpeedAttr = "robot_ref_adv_speed";
	static constexpr const char *kRobotRefRotSpeedAttr = "robot_ref_rot_speed";
	static constexpr const char *kRobotRefSideSpeedAttr = "robot_ref_side_speed";
	static constexpr const char *kRobotRefSpeedTimestampAttr = "robot_ref_speed_timestamp";

	struct Params
	{
		float clearance_m = 0.4f;
		float grid_resolution_m = 0.35f;
		float connection_radius_m = 1.2f;
		float waypoint_tolerance_m = 0.25f;
		float max_adv_speed_mps = 0.7f;
		float max_rot_speed_rps = 0.8f;
		float pos_gain = 1.2f;
		float rot_gain = 1.5f;
		bool interpolate_rt = true;
		int max_lidar_draw_points = 600;
		std::string lidar_name = "lidar3D";
		std::string target_edge_type = "target";
		float pose_xy_std_slow_m = 0.03f;
		float pose_xy_std_stop_m = 0.12f;
		float pose_theta_std_slow_rad = 0.04f;
		float pose_theta_std_stop_rad = 0.20f;
		float min_adv_speed_scale = 0.15f;
		float min_rot_speed_scale = 0.05f;
		float uncertainty_prediction_horizon_s = 0.4f;
		float pose_xy_std_growth_per_mps = 0.5f;
		float pose_theta_std_growth_per_rps = 1.0f;
		float adv_rotation_coupling_exponent = 0.5f;
		float temporary_obstacle_front_distance_m = 1.8f;
		float temporary_obstacle_half_width_m = 0.9f;
		float temporary_obstacle_cluster_margin_m = 0.35f;
		float temporary_obstacle_padding_m = 0.18f;
		int temporary_obstacle_min_points = 12;
		int temporary_obstacle_history_scans = 5;
		std::uint64_t temporary_obstacle_ttl_ms = 3000;
		float goal_clearance_relax_dist_m = 0.6f;
		float goal_obstacle_margin_m = 0.08f;
		float goal_clearance_min_ratio = 0.85f;
		float straight_speed_heading_threshold_rad = 0.08f;
		float straight_speed_clearance_margin_m = 0.20f;
		float straight_speed_min_goal_dist_m = 1.5f;
	};

	struct GraphState
	{
		uint64_t room_id = 0;
		uint64_t robot_id = 0;
		std::string room_name;
		std::string robot_name;
		bool ready() const { return room_id != 0 && robot_id != 0 && !room_name.empty() && !robot_name.empty(); }
	};

	struct TargetInfo
	{
		uint64_t node_id = 0;
		std::string node_name;
		Eigen::Vector2f room_pos = Eigen::Vector2f::Zero();
		float yaw_rad = 0.f;
		float epistemic_gain = 0.f;
		bool epistemic_pending = false;
		bool from_affordance = false;
	};

	struct RobotPose
	{
		Eigen::Vector2f pos = Eigen::Vector2f::Zero();
		float theta = 0.f;
		Eigen::Affine2f as_transform() const
		{
			Eigen::Affine2f tf = Eigen::Affine2f::Identity();
			tf.translation() = pos;
			tf.linear() = Eigen::Rotation2Df(theta).toRotationMatrix();
			return tf;
		}
	};

	struct PlanningStep
	{
		RobotPose robot_pose;
		TargetInfo target;
		Eigen::Vector2f plan_origin = Eigen::Vector2f::Zero();
		bool target_changed = false;
	};

	struct PoseUncertainty
	{
		float xy_std_m = 0.f;
		float theta_std_rad = 0.f;
	};

	Params params;

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;
	GraphState graph_state_;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api_;
	std::vector<Eigen::Vector2f> room_polygon_;
	std::vector<Eigen::Vector2f> inner_polygon_;
	Polygons obstacle_polygons_;
	rc::LidarPointBuffer lidar_room_buffer_{5};
	std::optional<PathPlan> current_plan_;
	std::optional<Eigen::Vector2f> current_target_room_;
	std::optional<Eigen::Vector2f> manual_target_room_;
	std::optional<Eigen::Vector2f> manual_target_origin_room_;
	std::optional<TargetInfo> last_target_info_;
	rc::AffordanceManager affordance_manager_;
	std::vector<Polygon> last_mppi_trajectories_;
	Polygon last_mppi_average_trajectory_;
	int last_best_mppi_trajectory_idx_ = -1;
	int last_display_wp_index_ = 0;
	bool path_following_active_ = false;
	bool stop_sent_when_paused_ = false;
	bool stop_command_latched_ = false;
	bool has_last_speed_command_ = false;
	Eigen::Vector3f last_speed_command_ = Eigen::Vector3f::Zero();
	bool manual_target_dirty_ = false;
	bool room_view_fitted_ = false;
	uint64_t active_target_id_ = 0;
	bool room_wait_logged_ = false;
	bool target_wait_logged_ = false;
	bool compute_debug_logged_ = false;
	mutable std::optional<std::uint64_t> last_lidar_timestamp_ms_;
	mutable std::uint64_t lidar_period_ms_ = 100;
	mutable std::string obstacle_debug_report_;
	std::optional<Polygon> temporary_obstacle_polygon_;
	std::uint64_t temporary_obstacle_expires_at_ms_ = 0;
	RoomPathPlanner planner_;
	rc::TrajectoryController path_controller_;
	std::unique_ptr<Custom_widget> custom_widget_;
	std::unique_ptr<rc::Viewer2D> viewer_2d_;

	void load_params();
	void log_first_compute_once();
	std::uint64_t current_time_ms() const;
	bool sync_world_state(std::uint64_t timestamp_ms);
	std::optional<PlanningStep> build_planning_step(std::uint64_t timestamp_ms);
	bool ensure_current_plan(const PlanningStep &step);
	bool refresh_graph_state();
	void update_custom_widget(const std::optional<RobotPose> &robot_pose);
	static bool same_target_instance(const TargetInfo &lhs, const TargetInfo &rhs);
	static float cross2d(const Eigen::Vector2f &origin, const Eigen::Vector2f &a, const Eigen::Vector2f &b);
	static Polygon convex_hull(Polygon points);
	static Polygon inflate_convex_polygon(const Polygon &polygon, float padding_m);
	static void log_compute_perf(FPSCounter &counter);
	static float ramp_uncertainty_scale(float value, float slow_threshold, float stop_threshold, float min_scale);
	static float preserve_sign_clamp(float value, float max_abs);
	std::optional<std::vector<Eigen::Vector2f>> read_room_polygon() const;
	Polygons read_obstacle_polygons(std::uint64_t timestamp_ms) const;
	std::vector<Eigen::Vector3f> read_recent_lidar_points_in_room(std::uint64_t timestamp_ms,
	                                                           int max_scans);
	void update_active_obstacle_polygons(std::uint64_t timestamp_ms);
	bool create_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
	                                   const RobotPose &robot_pose,
	                                   const Eigen::Vector2f &blockage_center_room,
	                                   float blockage_radius_m);
	Polygon make_obstacle_polygon(const Eigen::Vector2f &center,
	                             float yaw,
	                             float width_m,
	                             float depth_m) const;
	std::optional<RobotPose> read_robot_pose_in_room(std::uint64_t timestamp_ms) const;
	std::optional<PoseUncertainty> read_pose_uncertainty() const;
	void apply_uncertainty_speed_limit(float &adv_mps, float &side_mps, float &rot_rps) const;
	std::optional<TargetInfo> read_target_in_room(std::uint64_t timestamp_ms);
	void set_manual_target(const QPointF &point);
	void clear_manual_target();
	void execute_plan(const RobotPose &robot_pose);
	void publish_robot_reference_speed(float adv_mps, float side_mps, float rot_rps, std::uint64_t timestamp_ms);
	void send_speed_command(float adv_mps, float side_mps, float rot_rps);
	void stop_robot();

        // State machine
        void waiting_enter();
        void waiting_loop();
        void operating_enter();
        void operating_loop();
        void degraded_enter();
        void degraded_loop();

        void cleanup_owned_nodes();
        void on_optional_peer_lost(const std::string &name, std::uint32_t id);
        void on_optional_peer_ready(const std::string &name, std::uint32_t id);

        std::unique_ptr<AgentPresenceMonitor> presence_monitor;

signals:
        void presenceReady();
        void presenceLost();
};

#endif
