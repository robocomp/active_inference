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

#include "kinematics.h"
#include "efe_gradient.h"
#include "arm_belief_viewer_3d.h"

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include <Eigen/Dense>
#include <memory>
#include <optional>

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

	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;

	std::unique_ptr<Kinematics> kinematics_;
	std::unique_ptr<ArmBeliefViewer3D> arm_belief_viewer_;

	// Active EFE target — arm-base frame, metres. Updated each compute() cycle
	// from the bottle pose in DSR (which is in turn fed from Webots).
	Eigen::Vector3d reach_target_{0.4, 0.0, 0.1};
	bool proxy_unreachable_warned_         = false;
	bool webots_proxy_unreachable_warned_  = false;
	bool joint_dump_pending_               = true;  // dump first received TJoints, once

	// Controller lifecycle: home to a known rest pose, idle until the user
	// arms the viewer's checkable button, then run continuous bottle-approach.
	// Un-checking the button drops back to SendingRestPose → Homing → idle.
	enum class Phase { SendingRestPose, Homing, WaitingForStart, ActiveEFE };
	Phase phase_ = Phase::SendingRestPose;
	bool  run_requested_ = false;   // mirrors the viewer button's checked state
	std::array<double, Kinematics::N_ARM_JOINTS> rest_pose_angles_{
		0.0, 0.3, 0.0, 1.5, 0.0, 1.4, 0.0
	};
	int homing_settled_ticks_ = 0;
	static constexpr double HOMING_TOLERANCE_RAD = 0.05;  // ≈ 2.9°
	static constexpr int    HOMING_SETTLE_TICKS  = 5;     // consecutive cycles within tolerance

	// Bottle-approach behaviour (replaces the previous random-target cycle).
	// Side grasp: standoff next to the bottle, tool +Z pointing into the bottle
	// horizontally, tool +X aligned with the bottle's vertical axis so the
	// Robotiq fingers wrap around the body.
	static constexpr double APPROACH_STANDOFF_M = 0.12;  // standoff along approach axis
	static constexpr double REACH_TOLERANCE_M   = 0.02;  // 2 cm — "arrived"
	bool arrived_logged_ = false;

	struct SideGraspTarget
	{
		Eigen::Vector3d stand_off_pos;  // position target (robot frame, m)
		Eigen::Vector3d z_tool_des;     // unit vector, robot frame
		Eigen::Vector3d x_tool_des;     // unit vector, robot frame
	};

	// Fraction of bottle height (measured from the base origin along the
	// bottle's +Z axis) that we aim the EE at. 0.5 = mid-body.
	static constexpr double BOTTLE_GRASP_HEIGHT_FRAC = 0.5;

	// DSR sub-APIs used by the bottle-approach loop.
	std::unique_ptr<DSR::RT_API>         rt_api_;
	std::unique_ptr<DSR::InnerEigenAPI>  inner_eigen_api_;

	// Webots scene-object DEF names.
	static constexpr const char* WEBOTS_BOTTLE_DEF = "bottle";
	static constexpr const char* WEBOTS_TABLE_DEF  = "table";

	// Pull bottle+table world poses from Webots, project bottle into the table
	// frame, and write the result as the DSR table→bottle RT edge.
	void update_bottle_pose_in_dsr();

	// Build a side-grasp target: read bottle pose in the DSR "robot" frame via
	// InnerEigenAPI, derive a horizontal approach axis perpendicular to the
	// bottle's vertical axis, and back the EE off by APPROACH_STANDOFF_M.
	// Returns nullopt if the chain isn't resolvable or the bottle sits on the
	// robot's vertical axis (radial direction undefined).
	std::optional<SideGraspTarget> compute_side_grasp_target();

	// Project the 8 table corners and the bottle origin into the robot frame
	// via InnerEigenAPI, then push them to the viewer for drawing.
	void update_viewer_scene_objects();

signals:
	//void customSignal();
};

#endif
