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
#include <random>

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

	// Controller lifecycle: home → wait for Start → run continuous EFE.
	// Un-checking the button drops back to SendingRestPose → Homing → idle.
	enum class Phase { SendingRestPose, Homing, WaitingForStart, ActiveEFE };
	Phase phase_ = Phase::SendingRestPose;
	bool  run_requested_ = false;   // mirrors the viewer button's checked state
	// "Ready over the table" rest pose, in the arm-base frame (base_tf_=identity:
	// +X forward, +Y left, +Z up, z=0 at the desk). Base yaw j1=0.30 centres the
	// hand on +X; it hovers ~10 cm above the table in front (EE≈(0.47,0.04,0.10))
	// with manipulability μ≈0.106 — far from the near-singular contortions that
	// were tipping the bottle (old extended pose μ≈0.03). Chosen by the scored
	// FK search in initialize(); override at runtime via Controller.rest_pose.
	std::array<double, Kinematics::N_ARM_JOINTS> rest_pose_angles_{
		0.30, 1.20, 0.0, 1.20, -0.80, 1.70, 0.0
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

	// Tip-trajectory logging (diagnose approach speed/shape). When tip_log_ is
	// set the ActiveEFE block prints one "[tiplog] ..." CSV line per cycle and
	// tracks the previous tip pose to report the measured Cartesian speed.
	bool                           tip_log_ = false;
	long                           tip_log_cycle_ = 0;
	std::optional<Eigen::Vector3d> tip_log_prev_pos_;

	struct SideGraspTarget
	{
		Eigen::Vector3d stand_off_pos;  // approach target (robot frame, m)
		Eigen::Vector3d grasp_pos;      // bottle body-centre: the grasp point, m
		Eigen::Vector3d z_tool_des;     // unit vector, robot frame
		Eigen::Vector3d x_tool_des;     // unit vector, robot frame
	};

	// Fraction of bottle height (measured from the base origin along the
	// bottle's +Z axis) that we aim the EE at. 0.5 = mid-body.
	static constexpr double BOTTLE_GRASP_HEIGHT_FRAC = 0.5;

	// ── Grasp FSM (inner state machine of Phase::ActiveEFE) ─────────────────
	// Hierarchical active inference: each state installs a prior (preferred
	// pose + gripper state) that the continuous EFE controller realises; a
	// transition fires when the lower level reports the prior is satisfied
	// (error inside the deadband tolerance) or an observation (finger force)
	// confirms the outcome. See EFE_CONTROLLER_MATH.md §1, §6.
	//   Tracking      : approach the live standoff, gripper open, bottle tracked.
	//   Inserting     : commit — latch the grasp frame, ease into the bottle body.
	//   Closing       : hold pose, close gripper, watch finger force.
	//   Lifting       : raise +Δz holding the grasped bottle.
	//   PlaceMoving   : carry the bottle to a hover above a random table spot.
	//   PlaceLowering : lower onto the table spot.
	//   PlaceReleasing: open gripper, let the bottle settle.
	//   PlaceRetreating: rise off the placed bottle.
	//   (then) return to rest pose via the outer Homing path, and loop.
	enum class GraspPhase { Tracking, Inserting, Closing, Lifting,
	                        PlaceMoving, PlaceLowering, PlaceReleasing, PlaceRetreating };
	GraspPhase grasp_phase_ = GraspPhase::Tracking;
	int        grasp_settle_ticks_ = 0;   // consecutive converged cycles before committing
	int        closing_ticks_      = 0;   // cycles spent closing (miss timeout)
	int        place_ticks_        = 0;   // watchdog within a place sub-state
	bool       returning_for_cycle_ = false;  // auto-restart Tracking after the rest-return
	// Grasp frame latched at Tracking→Inserting so gripper/bottle contact can't
	// make the target chase its own disturbance.
	SideGraspTarget latched_grasp_{};
	Eigen::Vector3d lift_target_{};       // latched grasp_pos + LIFT_HEIGHT·ẑ_world
	Eigen::Vector3d place_pos_{};         // random place point on the table (robot frame)
	Eigen::Vector3d place_hover_{};       // place_pos_ + LIFT_HEIGHT·ẑ_world
	// Place orientation: tool +Z re-pointed radially toward the place spot with
	// tool +Y kept up (bottle upright). Yaw is free, so this avoids the arm
	// contorting to hold the original grasp yaw across the table.
	Eigen::Vector3d place_z_des_{};
	Eigen::Vector3d place_x_des_{};
	std::mt19937    rng_{std::random_device{}()};

	static constexpr int    GRASP_SETTLE_TICKS   = 8;     // converged cycles to commit
	static constexpr double GRASP_ALIGN_TOL_RAD  = 0.10;  // ≈5.7° orientation tolerance to commit
	static constexpr double INSERT_VEL_MS        = 0.05;  // gentle approach speed for soft contact
	static constexpr float  GRASP_FORCE_THRESH   = 1.0f;  // finger force → object held (TUNE)
	static constexpr int    CLOSING_TIMEOUT_TICKS = 100;  // ~2 s closing w/o force → miss
	static constexpr double LIFT_HEIGHT_M        = 0.12;  // how high to pick the bottle
	// Random place target box (robot/arm-base frame, m): a conservative
	// reachable patch of the table. z is taken from the grasped bottle so its
	// base lands back on the table surface.
	static constexpr double PLACE_X_MIN = 0.35, PLACE_X_MAX = 0.55;
	static constexpr double PLACE_Y_MIN = -0.20, PLACE_Y_MAX = 0.20;
	static constexpr double PLACE_MIN_MOVE_M    = 0.10;  // ≥ this far from the pick spot
	static constexpr int    PLACE_TIMEOUT_TICKS = 300;   // ~6 s safety per place move
	static constexpr int    RELEASE_TICKS       = 25;    // ~0.5 s to let the bottle go

	// Gripper command sent every compute() cycle through
	//   kinovaarm_proxy->setGripperPos(gripper_command_).
	// 1.0 = fully open, 0.0 = fully closed (matching the bridge's KinovaArm
	// implementation, which internally inverts to motor position). The
	// upcoming grasp FSM (Approach / Grasp / Lift) just writes to this.
	// Default open so the controller starts and idles in a safe, harmless
	// configuration.
	float gripper_command_ = 1.0f;
	bool  gripper_proxy_warned_ = false;

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
