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
class SelfProjectionViewer;                      // self-projection RGB dock (src/self_projection_viewer.{h,cpp})
namespace rc::media { class ThreadedMediaSource; }   // threaded media-plane subscriber (common/media_transport)

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

	// ── Self-projection RGB viewer (P1) ───────────────────────────────────────
	// Off the 100 Hz loop: a ThreadedMediaSource owns an RX thread on the
	// zero-copy media plane (discovered from the DSR graph), the dock polls it at
	// ~30 Hz. Disabled by default (SelfProjection.enable). See media_source.h.
	std::unique_ptr<rc::media::ThreadedMediaSource> media_source_;
	std::unique_ptr<SelfProjectionViewer>           self_projection_viewer_;
	void init_self_projection(const ConfigLoader& configLoader);

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
	// Source of the bottle pose. false (default): query Webots ground truth (getObjectPose).
	// true: READ the bottle pose from the DSR graph (the table->bottle RT edge written by an
	// external perception agent, e.g. bottle_concept fusing ZED+YOLO), composed with the cached
	// table_world_ to stay in the controller's world frame. A missing/stale edge ⇒ keep the last
	// belief (a dropped detection the look-up belief predicts through).
	bool              bottle_from_graph_ = false;
	bool              bottle_valid_      = false;   // a bottle pose has been obtained at least once
	// Perception refresh tracking. Reading the graph is free and happens every cycle, but the bottle
	// pose is only genuinely NEW when the producer (bottle_concept) bumps model_generation on the node.
	// So model_generation, not the read, is the real perception clock: bottle_obs_fresh_ flags the
	// cycle a new generation arrived, and bottle_refresh_period_s_ is the measured inter-generation
	// interval — the true held-velocity exposure τ_perc for the v·τ≤βΔ lag-safety contract. This
	// replaces the synthetic perception_latency_ms sleep with the producer's actual bandwidth.
	int               bottle_model_generation_ = -1;
	bool              bottle_obs_fresh_        = false;
	double            bottle_refresh_period_s_ = 0.0;
	std::chrono::steady_clock::time_point bottle_last_gen_time_{};
	bool              bottle_gen_time_valid_   = false;
	// OPT-IN scene publishing (default off): write robot->table->bottle RT edges from Webots ground
	// truth so other agents/viewer see a live scene. With bottle_from_graph also on, this gives a
	// self-contained ROUND-TRIP test of the graph read path before bottle_concept exists. In
	// production bottle_concept is the writer and this stays OFF (else graph mode reads back ground
	// truth and bypasses real perception).
	bool              publish_scene_to_graph_ = false;
	bool              base_tf_set_       = false;
	std::array<double, Kinematics::N_ARM_JOINTS> cur_q_{};   // latest measured config (IK seed)
	Eigen::Vector2d   last_spawn_xy_{0.0, -0.14};            // last commanded bottle respawn (world xy)

	// Webots scene-object DEF/name strings.
	static constexpr const char* WEBOTS_BOTTLE_DEF = "bottle";
	static constexpr const char* WEBOTS_TABLE_DEF  = "table";
	static constexpr const char* WEBOTS_ROBOT_DEF  = "KINOVA";  // shadow_arm.wbt: the arm's own top-level Robot (DEF KINOVA); its world pose IS the arm base
	static constexpr const char* WEBOTS_ARM_DEF    = "kinova_arm_r";

	// ── Perception / scene services (used by the FSM via the friend reference) ─
	void refresh_arm_base_world();        // install base_tf_ from the live P3Bot world pose
	void update_bottle_pose_in_dsr();     // pull robot/table/bottle world poses into DSR
	void update_viewer_scene_objects();   // draw table corners + bottle in the viewer
	void respawn_bottle(double x, double y);   // teleport an upright bottle (bridge setObjectPose)

	// Continuous GROUND-TRUTH bottle monitor (runs every compute, all phases, both modes). Reads the
	// real bottle pose from Webots — NOT the perceived belief, which stays locked thinking a bottle is
	// on the table while the real one rolled off. If the bottle tips or rolls, abort the attempt
	// (open gripper + fsm reset) and re-stand it at its last upright spot. Returns true if it aborted.
	bool monitor_bottle_and_recover();
	bool              bottle_monitor_     = true;
	int               monitor_cooldown_   = 0;     // settle ticks after a re-stand before re-checking
	int               monitor_tick_       = 0;     // throttle counter for the GT read
	Eigen::Vector2d   last_upright_bottle_xy_{0.44, -0.55};   // last GT xy seen standing on the table
	static constexpr double BOTTLE_TIP_RAD = 0.6;  // ~34°: tilt beyond this ⇒ tipped

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
