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

#include "controller_display.h"
#include "controller_motion_commander.h"
#include "controller_obstacle_tracker.h"
#include "controller_runtime_types.h"
#include "controller_session.h"
#include "controller_world_model.h"
#include "room_path_planner.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "trajectory_controller.h"

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
	using Polygon = ControllerPolygon;
	using Polygons = ControllerPolygons;
	using PathPlan = ControllerPathPlan;
	using Params = ControllerParams;
	using TargetInfo = ControllerTargetInfo;
	using RobotPose = ControllerRobotPose;
	using PlanningStep = ControllerPlanningStep;

	Params params;

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api_;
	ControllerWorldModel world_model_;
	ControllerObstacleTracker obstacle_tracker_;
	ControllerMotionCommander motion_commander_;
	ControllerDisplay display_;
	ControllerSession session_;
	rc::AffordanceManager affordance_manager_;
	bool path_following_active_ = false;
	bool stop_sent_when_paused_ = false;
	bool compute_debug_logged_ = false;
	bool owned_nodes_cleaned_ = false;
	RoomPathPlanner planner_;
	rc::TrajectoryController path_controller_;

	void load_params();
	void log_first_compute_once();
	std::uint64_t current_time_ms() const;
	bool sync_world_state(std::uint64_t timestamp_ms);
	std::optional<PlanningStep> build_planning_step(std::uint64_t timestamp_ms);
	bool ensure_current_plan(const PlanningStep &step);
	void update_custom_widget(const std::optional<RobotPose> &robot_pose);
	static void log_compute_perf(FPSCounter &counter);
	void set_manual_target(const QPointF &point);
	void clear_manual_target();
	void execute_plan(const RobotPose &robot_pose);
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

		AgentPresenceCoordinator presence_coordinator_;

signals:
        void presenceReady();
        void presenceLost();
};

#endif
