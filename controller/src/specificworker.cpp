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
#include "specificworker.h"

#include "../../common/robust_metrics/robust_metrics.h"

#include "custom_widget.h"
#include <fps/fps.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
		
		const int period = configLoader.get<int>("Period.Compute");

		// State machine: Compute → Waiting → Operating → Degraded → Waiting
		states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
		    std::bind(&SpecificWorker::waiting_loop, this),
		    std::bind(&SpecificWorker::waiting_enter, this));
		states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
		    std::bind(&SpecificWorker::operating_loop, this),
		    std::bind(&SpecificWorker::operating_enter, this));
		states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
		    std::bind(&SpecificWorker::degraded_loop, this),
		    std::bind(&SpecificWorker::degraded_enter, this));

		// Compute → Waiting on start
		states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
		// Waiting → Operating when all required peers are ready
		states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
		// Operating → Degraded when a required peer is lost
		states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
		// Degraded → Waiting immediately (self-kill scheduled inside degraded_enter)
		states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

		statemachine.addState(states["Waiting"].get());
		statemachine.addState(states["Operating"].get());
		statemachine.addState(states["Degraded"].get());

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	cleanup_owned_nodes();
	stop_robot();
	qInfo() << "Destroying SpecificWorker";
	/*
	for (auto const& [name, g] : Graphs) {
	    g->write_to_json_file("./"+agent_name+"_"+name+".json");
	}
	*/
}


void SpecificWorker::initialize()
{
    qInfo() << "initialize worker";
	GenericWorker::initialize();

	presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
	presence_coordinator_.set_transition_hooks({
	    .request_presence_ready = [this]() { emit presenceReady(); },
	    .request_presence_lost = [this]() { emit presenceLost(); },
	});
	presence_coordinator_.set_peer_hooks({
	    .on_peer_restarted = [this](std::uint32_t id)
	    {
	        qInfo() << "[Presence] peer" << id << "restarted";
	    },
	    .on_optional_peer_lost = [this](const std::string &name, std::uint32_t id)
	    {
	        on_optional_peer_lost(name, id);
	    },
	    .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
	    {
	        on_optional_peer_ready(name, id);
	    },
	});
	presence_coordinator_.set_lifecycle_hooks({
	    .on_waiting_enter = [this]()
	    {
	        const auto missing = presence_coordinator_.missing_required_names();
	        if (missing.empty())
	            qInfo("[SM] -> Waiting");
	        else
	        {
	            QString m;
	            for (const auto &label : missing) m += " " + QString::fromStdString(label);
	            qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
	        }
	    },
	    .on_operating_enter = []()
	    {
	        qInfo("[SM] -> Operating: all required peers present");
	    },
	    .on_operating_loop = [this]()
	    {
	        compute();
	    },
	    .on_degraded_enter = [this]()
	    {
	        qInfo("[SM] -> Degraded: required peer lost. Cleaning up and exiting.");
	        cleanup_owned_nodes();
	        QTimer::singleShot(500, QCoreApplication::instance(), SLOT(quit()));
	    },
	});
	presence_coordinator_.start();

	QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
	                 this, &SpecificWorker::cleanup_owned_nodes, Qt::UniqueConnection);

	load_params();
	inner_eigen_api_ = G ? G->get_inner_eigen_api() : nullptr;
	session_.set_graph(G);
	world_model_.set_affordance_manager(&affordance_manager_);
	world_model_.set_dependencies(G, inner_eigen_api_.get());
	obstacle_tracker_.set_dependencies(G, inner_eigen_api_.get(), &world_model_.graph_state());
	obstacle_tracker_.set_graph_layout_callback([this]() { trigger_graph_layout_twopi(); });
	motion_commander_.set_dependencies(G,
	                                 &world_model_,
	                                 omnirobot_proxy,
	                                 agent_id,
	                                 [this](const QString &text) { display_.set_command_text(text); });
	path_controller_.set_lidar_buffer(obstacle_tracker_.lidar_buffer());

	//Subscription to DSR graph update signals. 
	// If multiple graphs exist, it is necessary to specify the graph name 
	// using 'Graphs.at("name")' to connect its signals to the Worker's slots.
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/
	//If you have more than one graph, you need to connect to the specific graph with the name
	//graph_viewers.at("")->add_custom_widget_to_dock("CustomWidget", &custom_widget);
	display_.initialize(graph_viewers,
	                  obstacle_tracker_.lidar_buffer(),
	                  [this](const QPointF &point) { set_manual_target(point); },
	                  [this]() { clear_manual_target(); },
	                  [this](bool checked)
	                  {
	                      path_following_active_ = checked;
	                      if (checked)
	                      {
	                          stop_sent_when_paused_ = false;
	                      }
	                      else if (!stop_sent_when_paused_)
	                      {
	                          path_controller_.stop();
	                          stop_robot();
	                          stop_sent_when_paused_ = true;
	                      }
	                  });
	update_custom_widget(std::nullopt);

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}


void SpecificWorker::compute()
{
	static FPSCounter compute_perf_counter;
	struct ScopedComputePerfLog
	{
		FPSCounter &counter;
		~ScopedComputePerfLog() { SpecificWorker::log_compute_perf(counter); }
	};
	ScopedComputePerfLog perf_log{compute_perf_counter};

	log_first_compute_once();

	if (!G)
	{
		update_custom_widget(std::nullopt);
		return;
	}

	const auto now_ms = current_time_ms();
	if (!sync_world_state(now_ms))
		return;

	const auto step = build_planning_step(now_ms);
	if (!step.has_value())
		return;

	if (!ensure_current_plan(*step))
		return;

	if (path_following_active_)
	{
		stop_sent_when_paused_ = false;
		execute_plan(step->robot_pose);
	}
	else if (!stop_sent_when_paused_)
	{
		path_controller_.stop();
		stop_robot();
		stop_sent_when_paused_ = true;
	}
	update_custom_widget(step->robot_pose);
}

/////////////////////////////////////////////////////////////////
// ─── State machine ─────────────────────────────────────────────────────────

void SpecificWorker::log_compute_perf(FPSCounter &counter)
{
	counter.cont++;
	const auto now = std::chrono::high_resolution_clock::now();
	const auto elapsed_ms = std::chrono::duration<double, std::milli>(now - counter.begin).count();
	if (elapsed_ms < 1000.0)
		return;

	counter.last_period = static_cast<float>(elapsed_ms / std::max(1u, counter.cont));
	counter.period = 1000;
	const float fps = counter.get_frequency();
	const float cpu = std::max(0.f, counter.get_cpu_use());
	qInfo("[CTRL] fps=%.1f cpu=%.0f%% period=%.1fms", fps, cpu, counter.get_period());
	counter.begin = now;
	counter.cont = 0;
}

////////////////////////////////////////////////////////////////////
void SpecificWorker::log_first_compute_once()
{
	if (compute_debug_logged_)
		return;

	compute_debug_logged_ = true;
	qInfo() << "[CTRL] first compute() G=" << (G ? 1 : 0)
	        << "inner_api=" << (inner_eigen_api_ ? 1 : 0)
	        << "room='" << QString::fromStdString(world_model_.graph_state().room_name) << "'"
	        << "robot='" << QString::fromStdString(world_model_.graph_state().robot_name) << "'";
}

std::uint64_t SpecificWorker::current_time_ms() const
{
	return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count());
}

bool SpecificWorker::sync_world_state(std::uint64_t timestamp_ms)
{
	return session_.sync_world_state(timestamp_ms,
	                               world_model_,
	                               planner_,
	                               obstacle_tracker_,
	                               path_controller_,
	                               motion_commander_,
	                               display_);
}

std::optional<SpecificWorker::PlanningStep> SpecificWorker::build_planning_step(std::uint64_t timestamp_ms)
{
	return session_.build_planning_step(timestamp_ms,
	                                  world_model_,
	                                  obstacle_tracker_,
	                                  affordance_manager_,
	                                  path_controller_,
	                                  motion_commander_,
	                                  display_);
}

bool SpecificWorker::ensure_current_plan(const PlanningStep &step)
{
	return session_.ensure_current_plan(step,
	                                  planner_,
	                                  obstacle_tracker_,
	                                  path_controller_,
	                                  motion_commander_,
	                                  display_);
}

/////////////////////////////////////////////////////////////////
void SpecificWorker::load_params()
{
	load_optional_cast<double>("Planner.Clearance", params.clearance_m);
	load_optional_cast<double>("Planner.GridResolution", params.grid_resolution_m);
	load_optional_cast<double>("Planner.ConnectionRadius", params.connection_radius_m);
	load_optional_cast<double>("Controller.WaypointTolerance", params.waypoint_tolerance_m);
	load_optional_cast<double>("Controller.MaxAdvSpeed", params.max_adv_speed_mps);
	load_optional_cast<double>("Controller.MaxRotSpeed", params.max_rot_speed_rps);
	load_optional_cast<double>("Controller.PosGain", params.pos_gain);
	load_optional_cast<double>("Controller.RotGain", params.rot_gain);
	load_optional("Transforms.interpolate_rt", params.interpolate_rt);
	load_optional("Viewer2D.MaxLidarDrawPoints", params.max_lidar_draw_points);
	load_optional("Lidar.Name", params.lidar_name);
	load_optional("Target.EdgeType", params.target_edge_type);
	load_optional_cast<double>("Controller.PoseXYStdSlow", params.pose_xy_std_slow_m);
	load_optional_cast<double>("Controller.PoseXYStdStop", params.pose_xy_std_stop_m);
	load_optional_cast<double>("Controller.PoseThetaStdSlow", params.pose_theta_std_slow_rad);
	load_optional_cast<double>("Controller.PoseThetaStdStop", params.pose_theta_std_stop_rad);
	load_optional_cast<double>("Controller.MinAdvSpeedScale", params.min_adv_speed_scale);
	load_optional_cast<double>("Controller.MinRotSpeedScale", params.min_rot_speed_scale);
	load_optional_cast<double>("Controller.PosePredictionHorizon", params.uncertainty_prediction_horizon_s);
	load_optional_cast<double>("Controller.PoseXYStdGrowthPerMps", params.pose_xy_std_growth_per_mps);
	load_optional_cast<double>("Controller.PoseThetaStdGrowthPerRps", params.pose_theta_std_growth_per_rps);
	load_optional_cast<double>("Controller.AdvRotationCouplingExponent", params.adv_rotation_coupling_exponent);
	load_optional_cast<double>("Controller.TemporaryObstacleFrontDistance", params.temporary_obstacle_front_distance_m);
	load_optional_cast<double>("Controller.TemporaryObstacleHalfWidth", params.temporary_obstacle_half_width_m);
	load_optional_cast<double>("Controller.TemporaryObstacleClusterMargin", params.temporary_obstacle_cluster_margin_m);
	load_optional_cast<double>("Controller.TemporaryObstaclePadding", params.temporary_obstacle_padding_m);
	load_optional_cast<double>("Controller.TemporaryObstacleOcclusionDepth", params.temporary_obstacle_occlusion_depth_m);
	if (configLoader.exists("Controller.TemporaryObstacleRobustLoss"))
	{
		const auto loss_name = configLoader.get<std::string>("Controller.TemporaryObstacleRobustLoss");
		if (const auto loss_type = robust_loss_type_from_string(loss_name); loss_type.has_value())
			params.temporary_obstacle_robust_loss = loss_type.value();
		else
			qWarning() << "controller: unknown temporary obstacle robust loss" << loss_name.c_str() << "- using huber";
	}
	load_optional_cast<double>("Controller.TemporaryObstacleRobustLossScale", params.temporary_obstacle_robust_loss_scale_m);
	load_optional("Controller.TemporaryObstacleMinPoints", params.temporary_obstacle_min_points);
	load_optional("Controller.TemporaryObstacleHistoryScans", params.temporary_obstacle_history_scans);
	int temporary_obstacle_ttl_ms = static_cast<int>(params.temporary_obstacle_ttl_ms);
	load_optional("Controller.TemporaryObstacleTTLms", temporary_obstacle_ttl_ms);
	params.temporary_obstacle_ttl_ms = static_cast<std::uint64_t>(std::max(0, temporary_obstacle_ttl_ms));
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceInitLogOdds", params.temporary_obstacle_existence_init_log_odds);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceMinLogOdds", params.temporary_obstacle_existence_min_log_odds);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceMaxLogOdds", params.temporary_obstacle_existence_max_log_odds);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceRemoveThresholdLogOdds", params.temporary_obstacle_existence_remove_threshold_log_odds);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceObservationBias", params.temporary_obstacle_existence_observation_bias);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceSupportGain", params.temporary_obstacle_existence_support_gain);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceRememberedGain", params.temporary_obstacle_existence_remembered_gain);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceWeakMissPenalty", params.temporary_obstacle_existence_weak_miss_penalty);
	load_optional_cast<double>("Controller.TemporaryObstacleExistenceAbsencePenalty", params.temporary_obstacle_existence_absence_penalty);
	load_optional_cast<double>("Controller.GoalClearanceRelaxDist", params.goal_clearance_relax_dist_m);
	load_optional_cast<double>("Controller.GoalObstacleMargin", params.goal_obstacle_margin_m);
	load_optional_cast<double>("Controller.GoalClearanceMinRatio", params.goal_clearance_min_ratio);
	load_optional_cast<double>("Controller.StraightSpeedHeadingThreshold", params.straight_speed_heading_threshold_rad);
	load_optional_cast<double>("Controller.StraightSpeedClearanceMargin", params.straight_speed_clearance_margin_m);
	load_optional_cast<double>("Controller.StraightSpeedMinGoalDist", params.straight_speed_min_goal_dist_m);
	planner_.params.clearance_m = params.clearance_m;
	planner_.params.robot_width_m = 0.4f;
	planner_.params.grid_resolution_m = params.grid_resolution_m;
	planner_.params.connection_radius_m = params.connection_radius_m;
	planner_.params.path_sample_spacing_m = std::max(0.1f, params.grid_resolution_m * 1.5f);
	planner_.params.waypoint_tolerance_m = params.waypoint_tolerance_m;
	world_model_.set_params(&params);
	obstacle_tracker_.set_params(&params);
	motion_commander_.set_params(&params);
	session_.set_params(&params);

	path_controller_.params.max_adv = params.max_adv_speed_mps;
	path_controller_.params.max_rot = params.max_rot_speed_rps;
	path_controller_.params.robot_radius = std::max(0.15f, params.clearance_m * 0.5f);
	path_controller_.params.d_safe = params.clearance_m;
	path_controller_.params.min_adv_cmd = 0.f;
	path_controller_.params.goal_clearance_relax_dist = std::max(0.05f, params.goal_clearance_relax_dist_m);
	path_controller_.params.goal_obstacle_margin = std::max(0.f, params.goal_obstacle_margin_m);
	path_controller_.params.goal_clearance_min_ratio = std::clamp(params.goal_clearance_min_ratio, 0.5f, 1.f);
	path_controller_.params.straight_speed_heading_threshold = std::max(0.f, params.straight_speed_heading_threshold_rad);
	path_controller_.params.straight_speed_clearance_margin = std::max(0.f, params.straight_speed_clearance_margin_m);
	path_controller_.params.straight_speed_min_goal_dist = std::max(0.f, params.straight_speed_min_goal_dist_m);
	path_controller_.set_control_mode(rc::TrajectoryController::ControlMode::MPPI);
}

void SpecificWorker::update_custom_widget(const std::optional<RobotPose> &robot_pose)
{
	session_.update_display(robot_pose,
	                      display_,
	                      obstacle_tracker_.display_obstacle_polygons(),
	                      obstacle_tracker_.temporary_obstacle_rfe_points(),
	                      params.max_lidar_draw_points);
}

void SpecificWorker::execute_plan(const RobotPose &robot_pose)
{
	session_.execute_plan(robot_pose,
	                    path_controller_,
	                    obstacle_tracker_,
	                    motion_commander_,
	                    display_,
	                    affordance_manager_,
	                    [this]() { return current_time_ms(); });
}

void SpecificWorker::set_manual_target(const QPointF &point)
{
	session_.set_manual_target(point,
	                        world_model_,
	                        obstacle_tracker_,
	                        affordance_manager_,
	                        path_controller_,
	                        [this]() { return current_time_ms(); },
	                        [this]() { hibernationTick(); });
}

void SpecificWorker::clear_manual_target()
{
	session_.clear_manual_target(affordance_manager_,
	                          path_controller_,
	                          motion_commander_,
	                          [this]() { hibernationTick(); });
}
void SpecificWorker::stop_robot()
{
	session_.stop(path_controller_, motion_commander_);
}

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
	if (G)
	{
		if (auto node_opt = G->get_node(id); node_opt.has_value())
			obstacle_tracker_.handle_lidar_node(node_opt.value());
	}

	if (type == "room" or type == "robot")
		hibernationTick();
}

void SpecificWorker::modify_edge_slot(std::uint64_t, std::uint64_t, const std::string &type)
{
	if (type == params.target_edge_type || type == "RT")
		hibernationTick();
}
/**************************************/
// From the RoboCompOmniRobot you can call this methods:
// RoboCompOmniRobot::void this->omnirobot_proxy->correctOdometer(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBasePose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBaseState(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->resetOdometer()
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometer(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometerPose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->setSpeedBase(float advx, float advz, float rot)
// RoboCompOmniRobot::void this->omnirobot_proxy->stopBase()

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

