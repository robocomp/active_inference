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
#include <array>

class PickandPlaceFSM;   // manipulation behaviour layer (src/pick_and_place_fsm.{h,cpp})

/**
 * \brief Robot-I/O + scene-perception + lifecycle host for the Kinova controller.
 *
 * SpecificWorker owns the hardware-facing layer: reading joint state through the
 * KinovaArm proxy, pulling the arm/table/bottle world poses from Webots into the DSR
 * graph, driving the viewer, and running the outer lifecycle state machine
 * (SendingRestPose → Homing → WaitingForStart → ActiveEFE).
 *
 * All manipulation behaviour — the EFE/QP controller, the grasp-target perception and
 * the pick-and-place state machine — lives in PickandPlaceFSM, which is a friend and
 * reaches the services here through a reference. compute() is a thin orchestrator that,
 * once ActiveEFE, just calls fsm_->step().
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
	~SpecificWorker();

	// Outer lifecycle: home → wait for Start → run continuous EFE.
	enum class Phase { SendingRestPose, Homing, WaitingForStart, ActiveEFE };

public slots:
	void initialize();
	void compute();
	void emergency();
	void restore();
	int  startup_check();

	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};

private:
	friend class PickandPlaceFSM;   // the behaviour layer reaches these host services

	bool startup_check_flag;

	std::unique_ptr<Kinematics>        kinematics_;
	std::unique_ptr<ArmBeliefViewer3D> arm_belief_viewer_;
	std::unique_ptr<PickandPlaceFSM>   fsm_;

	// Current EFE target (world frame, m) — set by the FSM controller, drawn by the viewer.
	Eigen::Vector3d reach_target_{0.4, 0.0, 0.1};
	bool proxy_unreachable_warned_        = false;
	bool webots_proxy_unreachable_warned_ = false;
	bool joint_dump_pending_              = true;   // dump the first received TJoints, once

	// ── Outer lifecycle state ────────────────────────────────────────────────
	Phase phase_ = Phase::SendingRestPose;
	bool  run_requested_ = false;            // mirrors the viewer Start button
	// Fail-fast for UNATTENDED runs: on a homing timeout (arm jammed) quit cleanly instead of
	// idling in WaitingForStart forever (which wasted 15 min/variant in the ablation sweep).
	// Default false ⇒ interactive halt-and-wait behaviour unchanged. Set in sweep configs.
	bool  exit_on_homing_timeout_ = false;
	// Rest / parking pose (rad). Override via Controller.rest_pose.
	std::array<double, Kinematics::N_ARM_JOINTS> rest_pose_angles_{
		1.189, 0.244, 0.300, -2.101, 0.651, -1.502, 3.141
	};
	int homing_settled_ticks_ = 0;
	int homing_elapsed_ticks_ = 0;
	static constexpr double HOMING_TOLERANCE_RAD = 0.09;   // ≈5.2° — drooping a hair under gravity still counts as home
	static constexpr int    HOMING_SETTLE_TICKS  = 5;
	static constexpr int    HOMING_TIMEOUT_TICKS = 600;    // ~12 s — only a real jam trips this

	// Gripper command sent every cycle (1 = open, 0 = closed); the FSM writes it.
	float gripper_command_ = 1.0f;
	bool  gripper_proxy_warned_ = false;

	// DSR sub-APIs for the perception loop.
	std::unique_ptr<DSR::RT_API>        rt_api_;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api_;

	// ── World-frame scene state (FK reports world coords via base_tf_) ────────
	Eigen::Isometry3d arm_base_world_   = Eigen::Isometry3d::Identity();
	Eigen::Vector3d   bottle_pos_world_ {0.0, 0.0, 0.0};
	Eigen::Vector3d   bottle_axis_world_{0.0, 0.0, 1.0};
	double            bottle_radius_m_  = 0.035;
	double            bottle_height_m_  = 0.22;
	Eigen::Isometry3d table_world_      = Eigen::Isometry3d::Identity();
	double            table_top_z_      = 0.74;
	bool              scene_world_valid_ = false;
	bool              base_tf_set_       = false;
	std::array<double, Kinematics::N_ARM_JOINTS> cur_q_{};   // latest measured config (IK seed)
	Eigen::Vector2d   last_spawn_xy_{0.0, -0.14};            // last commanded bottle respawn (world xy)

	// Webots scene-object DEF/name strings.
	static constexpr const char* WEBOTS_BOTTLE_DEF = "bottle";
	static constexpr const char* WEBOTS_TABLE_DEF  = "table";
	static constexpr const char* WEBOTS_ROBOT_DEF  = "P3Bot";
	static constexpr const char* WEBOTS_ARM_DEF    = "kinova_arm_r";

	// ── Perception / scene services (used by the FSM via the friend reference) ─
	void refresh_arm_base_world();        // install base_tf_ from the live P3Bot world pose
	void update_bottle_pose_in_dsr();     // pull robot/table/bottle world poses into DSR
	void update_viewer_scene_objects();   // draw table corners + bottle in the viewer
	void respawn_bottle(double x, double y);   // teleport an upright bottle (bridge setObjectPose)

	// ── Recovery / kinematic helpers ─────────────────────────────────────────
	RoboCompKinovaArm::TJointAngles nearest_equiv_target(
	    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
	    const std::array<double, Kinematics::N_ARM_JOINTS>& desired) const;
	void teleport_to_rest();

	// ── Outer lifecycle phase handlers ───────────────────────────────────────
	void run_sending_rest_pose(const std::array<double, Kinematics::N_ARM_JOINTS>& q);
	void run_homing(const std::array<double, Kinematics::N_ARM_JOINTS>& q);
	void update_scene();
	void run_waiting_for_start();
	void abort_to_rest();

	// ── Per-cycle I/O helpers ────────────────────────────────────────────────
	void send_gripper_command();
	void update_viewer(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
	                   const std::array<double, Kinematics::N_ARM_JOINTS>& tau,
	                   const Eigen::Vector3d& ee_position);
#ifndef Q_MOC_RUN
	void handle_joint_dump(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
	                       const Eigen::Vector3d& ee_position,
	                       const RoboCompKinovaArm::TJoints& js);
#endif

signals:
	//void customSignal();
};

#endif
