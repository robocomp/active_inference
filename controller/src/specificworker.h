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
#include <string>
#include <vector>

#include "room_path_planner.h"
#include "trajectory_controller.h"
#include "viewer_2d.h"

class Custom_widget;

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
	using PathPlan = RoomPathPlanner::PathPlan;

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
		std::string target_edge_type = "target";
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

	Params params;

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;
	GraphState graph_state_;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api_;
	std::vector<Eigen::Vector2f> room_polygon_;
	std::vector<Eigen::Vector2f> inner_polygon_;
	std::optional<PathPlan> current_plan_;
	std::optional<Eigen::Vector2f> manual_target_room_;
	bool manual_target_dirty_ = false;
	bool room_view_fitted_ = false;
	uint64_t active_target_id_ = 0;
	bool room_wait_logged_ = false;
	bool target_wait_logged_ = false;
	RoomPathPlanner planner_;
	rc::TrajectoryController path_controller_;
	std::unique_ptr<Custom_widget> custom_widget_;
	std::unique_ptr<rc::Viewer2D> viewer_2d_;

	void load_params();
	bool refresh_graph_state();
	void update_custom_widget(const std::optional<RobotPose> &robot_pose);
	std::optional<std::vector<Eigen::Vector2f>> read_room_polygon() const;
	std::optional<RobotPose> read_robot_pose_in_room(std::uint64_t timestamp_ms) const;
	std::optional<TargetInfo> read_target_in_room(std::uint64_t timestamp_ms) const;
	void set_manual_target(const QPointF &point);
	void clear_manual_target();
	void execute_plan(const RobotPose &robot_pose);
	void stop_robot();

signals:
	//void customSignal();
};

#endif
