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
#include <atomic>
#include <string_view>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "controller_display.h"
#include "controller_camera_masks.h"
#include "controller_motion_commander.h"
#include "controller_obstacle_tracker.h"
#include "controller_runtime_types.h"
#include "controller_session.h"
#include "controller_world_model.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "trajectory_controller.h"

#include <chrono>
#include <cstdint>
#include <memory>

class FPSCounter;
namespace rc::media { class LidarPlaneReader; }  // shared zero-copy media-plane LiDAR consumer (keeps fastdds out of MOC)

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
	bool is_shutting_down() const noexcept { return shutting_down_.load(); }


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
	// The camera picture + YOLO masks shown in the affordance panel. Owns its own rgb subscriber and a
	// MaskIngestor; pumped from the control loop and pushed into the display as a finished image.
	// See controller_camera_masks.h for why the controller reads masks at all.
	std::unique_ptr<rc::ControllerCameraMasks> camera_masks_;
	std::string last_selected_affordance_;   // most recent distinct selection (for the "prev" label)
	std::string prev_selected_affordance_;   // the one selected before it
	// Driving is ON by default. The old Start/Stop toggle duplicated the mission bar's Stop and could
	// disagree with it, so there is now ONE halt control: Stop halts whatever is driving (mission, click or
	// affordance) and the next Run / target click / drive-mode change resumes.
	bool driving_enabled_ = false;
	// PAUSE is a SEPARATE axis from Run/Stop, not a third value of one.
	//   driving_enabled_  = the user pressed Run and has not aborted   (owns the route + mission)
	//   paused_           = the user pressed Pause                     (owns nothing; purely a hold)
	// The robot drives iff driving_enabled_ and not paused_. Keeping them separate is what lets Pause
	// resume exactly where it stopped while Stop cannot: Stop clears the first flag AND the route.
	bool paused_ = false;
	bool stop_sent_when_halted_ = false;
	bool compute_debug_logged_ = false;
	bool owned_nodes_cleaned_ = false;
	rc::TrajectoryController path_controller_;

	void load_params();
	void log_first_compute_once();
	std::uint64_t current_time_ms() const;
	bool sync_world_state(std::uint64_t timestamp_ms);
	std::optional<PlanningStep> build_planning_step(std::uint64_t timestamp_ms);
	bool ensure_current_plan(const PlanningStep &step);
	void update_custom_widget(const std::optional<RobotPose> &robot_pose);
	// Push the mission library into the dropdown. Called after load and after a recording is saved.
	void refresh_mission_list();
	// Push the mission/drive state to the panel. Called from the control loop EVERY iteration, not from
	// compute(): compute() returns early on any cycle without a target — which is precisely the state Stop
	// creates — so a push at its tail cannot report the thing the user just did.
	void push_mission_view();
	// Where recorded missions live. Alongside the agent's other config, so a tour is versioned with the
	// settings it was recorded under.
	std::string missions_path_ = "etc/missions.toml";
	static void log_compute_perf(FPSCounter &counter, ControllerDisplay *display);
	// Worst compute() period inside the current 1 s reporting window. The MEAN hides the problem — it sits at
	// ~105 ms against a 100 ms target and looks healthy while the tail runs past a second.
	static float worst_compute_period_ms_;
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
		void request_shutdown();
        void on_optional_peer_lost(const std::string &name, std::uint32_t id);
        void on_optional_peer_ready(const std::string &name, std::uint32_t id);

		AgentPresenceCoordinator presence_coordinator_;

		// ─── Control thread: runs compute() off the GUI / presence-SM thread ───
		// The MPPI optimisation in compute() can take hundreds of ms to seconds.
		// Running it here keeps the presence heartbeat loop and the Qt event loop
		// responsive. The GUI thread only renders (display_.present()) and feeds
		// user commands through command_queue_.
		std::thread control_thread_;
		std::atomic<bool> control_running_{false};
		std::atomic<bool> control_operating_{false};
		// LATCHED "we are in Operating", as opposed to control_operating_ which is an edge consumed by
		// the legacy timer path. Data-driven mode needs the STATE, not each tick.
		std::atomic<bool> operating_latched_{false};
		std::mutex control_mutex_;
		std::condition_variable control_cv_;
		std::mutex command_mutex_;
		std::vector<std::function<void()>> command_queue_;
		std::atomic<bool> command_pending_{false};
		QTimer *render_timer_ = nullptr;
		std::atomic<bool> shutting_down_{false};

		void control_loop();
		void stop_control_thread();
		void enqueue_command(std::function<void()> command);

		// ★ WHY poll_lidar_media() returned nothing. It has FOUR distinct exits and they used to
		// collapse into one bool, so `fresh_lidar=false` could mean "no subscriber", "the graph has
		// not named the robot yet", "no sweep arrived" or "the tracker rejected this stamp" — which
		// need completely different fixes. Rejected is the NORMAL one: a 20 Hz loop against a ~9.4 Hz
		// LiDAR sees no new sweep on about half its cycles, so a bare false is not a fault at all.
		// Conflating that with a dead subscriber cost a long diagnosis on 2026-08-18.
		enum class LidarPoll { Fresh, NoReader, NoRobotName, NoSweep, Rejected };
		[[nodiscard]] static std::string_view to_string(LidarPoll r)
		{
			switch (r)
			{
				case LidarPoll::Fresh:       return "fresh";
				case LidarPoll::NoReader:    return "NO SUBSCRIBER (lidar_reader_ is null)";
				case LidarPoll::NoRobotName: return "graph has not named the robot yet";
				case LidarPoll::NoSweep:     return "no sweep from the media plane";
				case LidarPoll::Rejected:    return "sweep rejected by the tracker (stamp not advanced)";
			}
			return "?";
		}
		LidarPoll last_lidar_poll_ = LidarPoll::NoReader;

		// ─── LiDAR stream watchdog ────────────────────────────────────────────
		// If the LiDAR media stream stops producing while operating,
		// hold the robot in a local emergency state until it recovers — never plan on
		// stale perception. Transitions are logged so the user sees what happened.
		std::chrono::steady_clock::time_point last_lidar_rx_{};
		bool lidar_ever_received_ = false;
		bool lidar_stalled_ = false;
		std::chrono::steady_clock::time_point lidar_stall_since_{};
		std::int64_t last_stall_log_ms_ = 0;

		// Grace period before a Degraded (required-peer-lost) state actually tears down.
		// A transient flap during startup handshake / DSR churn must NOT delete our own
		// node and exit — debounce and only shut down if a peer is STILL missing.
		static constexpr int kRequiredLossGraceMs = 3000;

		// ─── Zero-copy media-plane LiDAR source ───────────────────────────────
		// Shared multi-plane reader (the same one every agent uses). It prefers the two
		// per-device planes — "helios" (high) + "bpearl" (low), published in the DEVICE
		// frame — transforms each to the robot frame via the DSR RT tree and MERGES them
		// into one scan; robot_concept publishes onto those same nodes while it is
		// bridging. Subscribers come up lazily inside poll() (throttled), so this is safe
		// from the Operating control thread. The merged robot-frame scan is fed once per
		// cycle to the obstacle tracker (which then applies the dynamic room<-robot pose).
		std::unique_ptr<rc::media::LidarPlaneReader> lidar_reader_;
		void init_lidar_media();
		LidarPoll poll_lidar_media();

signals:
        void presenceReady();
        void presenceLost();
};

#endif
