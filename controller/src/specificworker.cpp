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

#include "custom_widget.h"
#include <fps/fps.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <sstream>

namespace
{
	constexpr auto kRobotRefAdvSpeedAttr = "robot_ref_adv_speed";
	constexpr auto kRobotRefRotSpeedAttr = "robot_ref_rot_speed";
	constexpr auto kRobotRefSideSpeedAttr = "robot_ref_side_speed";
	constexpr auto kRobotRefSpeedTimestampAttr = "robot_ref_speed_timestamp";

	float cross2d(const Eigen::Vector2f &origin, const Eigen::Vector2f &a, const Eigen::Vector2f &b)
	{
		return (a.x() - origin.x()) * (b.y() - origin.y()) - (a.y() - origin.y()) * (b.x() - origin.x());
	}

	std::vector<Eigen::Vector2f> convex_hull(std::vector<Eigen::Vector2f> points)
	{
		if (points.size() < 3)
			return points;

		std::sort(points.begin(), points.end(), [](const auto &lhs, const auto &rhs)
		{
			if (lhs.x() != rhs.x())
				return lhs.x() < rhs.x();
			return lhs.y() < rhs.y();
		});
		points.erase(std::unique(points.begin(), points.end(), [](const auto &lhs, const auto &rhs)
		{
			return (lhs - rhs).cwiseAbs().maxCoeff() < 1e-4f;
		}), points.end());
		if (points.size() < 3)
			return points;

		std::vector<Eigen::Vector2f> lower;
		lower.reserve(points.size());
		for (const auto &point : points)
		{
			while (lower.size() >= 2 && cross2d(lower[lower.size() - 2], lower.back(), point) <= 0.f)
				lower.pop_back();
			lower.push_back(point);
		}

		std::vector<Eigen::Vector2f> upper;
		upper.reserve(points.size());
		for (auto it = points.rbegin(); it != points.rend(); ++it)
		{
			while (upper.size() >= 2 && cross2d(upper[upper.size() - 2], upper.back(), *it) <= 0.f)
				upper.pop_back();
			upper.push_back(*it);
		}

		lower.pop_back();
		upper.pop_back();
		lower.insert(lower.end(), upper.begin(), upper.end());
		return lower;
	}

	std::vector<Eigen::Vector2f> inflate_convex_polygon(const std::vector<Eigen::Vector2f> &polygon, float padding_m)
	{
		if (polygon.empty() || padding_m <= 0.f)
			return polygon;

		Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
		for (const auto &point : polygon)
			centroid += point;
		centroid /= static_cast<float>(polygon.size());

		std::vector<Eigen::Vector2f> inflated;
		inflated.reserve(polygon.size());
		for (const auto &point : polygon)
		{
			Eigen::Vector2f dir = point - centroid;
			const float norm = dir.norm();
			if (norm > 1e-4f)
				dir /= norm;
			else
				dir = Eigen::Vector2f::UnitX();
			inflated.push_back(point + dir * padding_m);
		}
		return inflated;
	}

	void log_compute_perf(FPSCounter &counter)
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
		std::cout << "[CTRL] fps=" << std::fixed << std::setprecision(1) << fps
		          << " cpu=" << std::setprecision(0) << cpu << "%"
		          << " period=" << std::setprecision(1) << counter.get_period() << "ms"
		          << std::endl;
		counter.begin = now;
		counter.cont = 0;
	}

	struct ScopedComputePerfLog
	{
		FPSCounter &counter;
		~ScopedComputePerfLog() { log_compute_perf(counter); }
	};

float ramp_uncertainty_scale(float value, float slow_threshold, float stop_threshold, float min_scale)
{
	if (stop_threshold <= slow_threshold)
		return value <= slow_threshold ? 1.f : std::clamp(min_scale, 0.f, 1.f);

	const float clamped_min_scale = std::clamp(min_scale, 0.f, 1.f);
	const float alpha = std::clamp((value - slow_threshold) / (stop_threshold - slow_threshold), 0.f, 1.f);
	return 1.f - alpha * (1.f - clamped_min_scale);
}

float preserve_sign_clamp(float value, float max_abs)
{
	return std::copysign(std::min(std::abs(value), std::max(0.f, max_abs)), value);
}
}

bool SpecificWorker::same_target_instance(const TargetInfo &lhs, const TargetInfo &rhs)
{
	constexpr float pos_eps_m = 0.05f;
	constexpr float yaw_eps_rad = 0.05f;
	constexpr float gain_eps = 1e-3f;

	return lhs.node_id == rhs.node_id
		&& lhs.from_affordance == rhs.from_affordance
		&& lhs.epistemic_pending == rhs.epistemic_pending
		&& (lhs.room_pos - rhs.room_pos).cwiseAbs().maxCoeff() < pos_eps_m
		&& std::abs(lhs.yaw_rad - rhs.yaw_rad) < yaw_eps_rad
		&& std::abs(lhs.epistemic_gain - rhs.epistemic_gain) < gain_eps;
}

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
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

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
	stop_robot();
	std::cout << "Destroying SpecificWorker" << std::endl;
	/*
	for (auto const& [name, g] : Graphs) {
	    g->write_to_json_file("./"+agent_name+"_"+name+".json");
	}
	*/
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
	std::print("controller debug: initialize() entered\n");
	std::fflush(stdout);
	GenericWorker::initialize();
	load_params();
	inner_eigen_api_ = G ? G->get_inner_eigen_api() : nullptr;
	custom_widget_ = std::make_unique<Custom_widget>();

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
	if (custom_widget_ != nullptr)
	{
		if (graph_viewers.contains(""))
			graph_viewers.at("")->add_custom_widget_to_dock("controller", custom_widget_.get());
		else if (!graph_viewers.empty())
			graph_viewers.begin()->second->add_custom_widget_to_dock("controller", custom_widget_.get());

		viewer_2d_ = std::make_unique<rc::Viewer2D>(custom_widget_->frame, QRectF(-5.0, -5.0, 10.0, 10.0), true);
		viewer_2d_->add_robot(0.5f, 0.6f, 0.f, 0.f, QColor("Tomato"));
		viewer_2d_->set_lidar_buffer(&lidar_room_buffer_);
		viewer_2d_->set_lidar_visible(custom_widget_->lidar_toggle_btn != nullptr
		                             ? custom_widget_->lidar_toggle_btn->isChecked()
		                             : false);
		path_controller_.set_lidar_buffer(&lidar_room_buffer_);
		viewer_2d_->show();
		connect(viewer_2d_.get(), &rc::Viewer2D::new_mouse_coordinates, this,
		        [this](const QPointF &point) { set_manual_target(point); });
		connect(viewer_2d_.get(), &rc::Viewer2D::right_click, this,
		        [this](const QPointF &) { clear_manual_target(); });
		connect(custom_widget_->lidar_toggle_btn, &QPushButton::toggled, this,
		        [this](bool checked) { if (viewer_2d_) viewer_2d_->set_lidar_visible(checked); });
		connect(custom_widget_->follow_toggle_btn, &QPushButton::toggled, this,
		        [this](bool checked)
		        {
		            path_following_active_ = checked;
		            custom_widget_->follow_toggle_btn->setText(checked ? "Stop" : "Start");
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
		connect(custom_widget_->mppi_paths_toggle_btn, &QPushButton::toggled, this,
		        [this](bool checked) { if (viewer_2d_) viewer_2d_->set_mppi_paths_visible(checked); });
	}
	update_custom_widget(std::nullopt);

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}


void SpecificWorker::compute()
{
	static FPSCounter compute_perf_counter;
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
void SpecificWorker::log_first_compute_once()
{
	if (compute_debug_logged_)
		return;

	compute_debug_logged_ = true;
	std::print("controller debug: first compute() entered G={} inner_api={} room_name='{}' robot_name='{}'\n",
	           G ? 1 : 0,
	           inner_eigen_api_ ? 1 : 0,
	           graph_state_.room_name,
	           graph_state_.robot_name);
	std::fflush(stdout);
}

std::uint64_t SpecificWorker::current_time_ms() const
{
	return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count());
}

bool SpecificWorker::sync_world_state(std::uint64_t timestamp_ms)
{
	if (!refresh_graph_state())
	{
		room_polygon_.clear();
		inner_polygon_.clear();
		current_plan_.reset();
		room_view_fitted_ = false;
		if (!room_wait_logged_)
		{
			qInfo() << "Controller waiting for room and robot nodes in DSR";
			room_wait_logged_ = true;
		}
		update_custom_widget(std::nullopt);
		stop_robot();
		return false;
	}
	room_wait_logged_ = false;

	const auto room_polygon = read_room_polygon();
	if (!room_polygon.has_value() || room_polygon->size() < 3)
	{
		room_polygon_.clear();
		inner_polygon_.clear();
		current_plan_.reset();
		room_view_fitted_ = false;
		qInfo() << "Controller waiting for delimiting polygon attributes on room node";
		update_custom_widget(std::nullopt);
		stop_robot();
		return false;
	}

	room_polygon_ = room_polygon.value();
	inner_polygon_ = planner_.compute_inner_polygon(room_polygon_);
	update_active_obstacle_polygons(timestamp_ms);
	return true;
}

std::optional<SpecificWorker::PlanningStep> SpecificWorker::build_planning_step(std::uint64_t timestamp_ms)
{
	const auto robot_pose = read_robot_pose_in_room(timestamp_ms);
	if (!robot_pose.has_value())
	{
		qInfo() << "Controller waiting for robot pose in room frame";
		update_custom_widget(std::nullopt);
		stop_robot();
		return std::nullopt;
	}

	PlanningStep step;
	step.robot_pose = *robot_pose;
	step.plan_origin = robot_pose->pos;

	if (manual_target_room_.has_value())
	{
		step.target.node_name = "mouse_target";
		step.target.room_pos = *manual_target_room_;
		current_target_room_ = step.target.room_pos;
		affordance_manager_.clear_current();
		const bool use_snapped_manual_origin = manual_target_dirty_ and manual_target_origin_room_.has_value();
		if (use_snapped_manual_origin)
			step.plan_origin = *manual_target_origin_room_;
		step.target_changed = manual_target_dirty_ || !current_plan_.has_value();
		manual_target_dirty_ = false;
		last_target_info_.reset();
		active_target_id_ = 0;
		target_wait_logged_ = false;
		return step;
	}

	const auto target = read_target_in_room(timestamp_ms);
	if (!target.has_value())
	{
		if (!target_wait_logged_)
		{
			qInfo() << "Controller waiting for an affordance target in DSR";
			target_wait_logged_ = true;
		}
		current_plan_.reset();
		last_target_info_.reset();
		active_target_id_ = 0;
		current_target_room_.reset();
		affordance_manager_.clear_current();
		update_custom_widget(robot_pose);
		stop_robot();
		return std::nullopt;
	}

	target_wait_logged_ = false;
	step.target = *target;
	current_target_room_ = step.target.room_pos;
	step.target_changed = !last_target_info_.has_value() || !same_target_instance(*last_target_info_, step.target);
	last_target_info_ = step.target;
	active_target_id_ = target->node_id;
	return step;
}

bool SpecificWorker::ensure_current_plan(const PlanningStep &step)
{
	if (step.target_changed || !current_plan_.has_value())
	{
		current_plan_ = planner_.plan_path(room_polygon_, inner_polygon_, obstacle_polygons_, step.plan_origin, step.target.room_pos);
		std::ostringstream out;
		out << "Path debug: target='" << step.target.node_name << "'"
		    << " target_pos=(" << step.target.room_pos.x() << "," << step.target.room_pos.y() << ")"
		    << " origin=(" << step.plan_origin.x() << "," << step.plan_origin.y() << ")";
		if (!current_plan_.has_value() || current_plan_->room_path.empty())
			out << " waypoints=0";
		else
		{
			out << " waypoints=" << current_plan_->room_path.size();
			for (std::size_t index = 0; index < current_plan_->room_path.size(); ++index)
			{
				const auto &point = current_plan_->room_path[index];
				out << " | p" << index << "=(" << point.x() << "," << point.y() << ")";
			}
		}
		std::print("{}\n", out.str());
		std::fflush(stdout);
	}

	if (!current_plan_.has_value() || current_plan_->room_path.empty())
	{
		qWarning() << "Controller could not produce a path to target" << step.target.node_name.c_str();
		update_custom_widget(step.robot_pose);
		stop_robot();
		return false;
	}

	if (step.target_changed || !path_controller_.is_active())
		path_controller_.set_path(current_plan_->room_path);

	return true;
}

/////////////////////////////////////////////////////////////////
void SpecificWorker::load_params()
{
	try { params.clearance_m = static_cast<float>(configLoader.get<double>("Planner.Clearance")); } catch (...) {}
	try { params.grid_resolution_m = static_cast<float>(configLoader.get<double>("Planner.GridResolution")); } catch (...) {}
	try { params.clearance_m = static_cast<float>(configLoader.get<double>("Planner.Clearance")); } catch (...) {}
	try { params.connection_radius_m = static_cast<float>(configLoader.get<double>("Planner.ConnectionRadius")); } catch (...) {}
	try { params.waypoint_tolerance_m = static_cast<float>(configLoader.get<double>("Controller.WaypointTolerance")); } catch (...) {}
	try { params.max_adv_speed_mps = static_cast<float>(configLoader.get<double>("Controller.MaxAdvSpeed")); } catch (...) {}
	try { params.max_rot_speed_rps = static_cast<float>(configLoader.get<double>("Controller.MaxRotSpeed")); } catch (...) {}
	try { params.pos_gain = static_cast<float>(configLoader.get<double>("Controller.PosGain")); } catch (...) {}
	try { params.rot_gain = static_cast<float>(configLoader.get<double>("Controller.RotGain")); } catch (...) {}
	try { params.interpolate_rt = configLoader.get<bool>("Transforms.interpolate_rt"); } catch (...) {}
	try { params.max_lidar_draw_points = configLoader.get<int>("Viewer2D.MaxLidarDrawPoints"); } catch (...) {}
	try { params.lidar_name = configLoader.get<std::string>("Lidar.Name"); } catch (...) {}
	try { params.target_edge_type = configLoader.get<std::string>("Target.EdgeType"); } catch (...) {}
	try { params.pose_xy_std_slow_m = static_cast<float>(configLoader.get<double>("Controller.PoseXYStdSlow")); } catch (...) {}
	try { params.pose_xy_std_stop_m = static_cast<float>(configLoader.get<double>("Controller.PoseXYStdStop")); } catch (...) {}
	try { params.pose_theta_std_slow_rad = static_cast<float>(configLoader.get<double>("Controller.PoseThetaStdSlow")); } catch (...) {}
	try { params.pose_theta_std_stop_rad = static_cast<float>(configLoader.get<double>("Controller.PoseThetaStdStop")); } catch (...) {}
	try { params.min_adv_speed_scale = static_cast<float>(configLoader.get<double>("Controller.MinAdvSpeedScale")); } catch (...) {}
	try { params.min_rot_speed_scale = static_cast<float>(configLoader.get<double>("Controller.MinRotSpeedScale")); } catch (...) {}
	try { params.uncertainty_prediction_horizon_s = static_cast<float>(configLoader.get<double>("Controller.PosePredictionHorizon")); } catch (...) {}
	try { params.pose_xy_std_growth_per_mps = static_cast<float>(configLoader.get<double>("Controller.PoseXYStdGrowthPerMps")); } catch (...) {}
	try { params.pose_theta_std_growth_per_rps = static_cast<float>(configLoader.get<double>("Controller.PoseThetaStdGrowthPerRps")); } catch (...) {}
	try { params.adv_rotation_coupling_exponent = static_cast<float>(configLoader.get<double>("Controller.AdvRotationCouplingExponent")); } catch (...) {}
	try { params.temporary_obstacle_front_distance_m = static_cast<float>(configLoader.get<double>("Controller.TemporaryObstacleFrontDistance")); } catch (...) {}
	try { params.temporary_obstacle_half_width_m = static_cast<float>(configLoader.get<double>("Controller.TemporaryObstacleHalfWidth")); } catch (...) {}
	try { params.temporary_obstacle_cluster_margin_m = static_cast<float>(configLoader.get<double>("Controller.TemporaryObstacleClusterMargin")); } catch (...) {}
	try { params.temporary_obstacle_padding_m = static_cast<float>(configLoader.get<double>("Controller.TemporaryObstaclePadding")); } catch (...) {}
	try { params.temporary_obstacle_min_points = configLoader.get<int>("Controller.TemporaryObstacleMinPoints"); } catch (...) {}
	try { params.temporary_obstacle_ttl_ms = static_cast<std::uint64_t>(std::max(0, configLoader.get<int>("Controller.TemporaryObstacleTTLms"))); } catch (...) {}
	try { params.goal_clearance_relax_dist_m = static_cast<float>(configLoader.get<double>("Controller.GoalClearanceRelaxDist")); } catch (...) {}
	try { params.goal_obstacle_margin_m = static_cast<float>(configLoader.get<double>("Controller.GoalObstacleMargin")); } catch (...) {}
	try { params.goal_clearance_min_ratio = static_cast<float>(configLoader.get<double>("Controller.GoalClearanceMinRatio")); } catch (...) {}
	try { params.straight_speed_heading_threshold_rad = static_cast<float>(configLoader.get<double>("Controller.StraightSpeedHeadingThreshold")); } catch (...) {}
	try { params.straight_speed_clearance_margin_m = static_cast<float>(configLoader.get<double>("Controller.StraightSpeedClearanceMargin")); } catch (...) {}
	try { params.straight_speed_min_goal_dist_m = static_cast<float>(configLoader.get<double>("Controller.StraightSpeedMinGoalDist")); } catch (...) {}
	planner_.params.clearance_m = params.clearance_m;
	planner_.params.robot_width_m = 0.4f;
	planner_.params.grid_resolution_m = params.grid_resolution_m;
	planner_.params.connection_radius_m = params.connection_radius_m;
	planner_.params.path_sample_spacing_m = std::max(0.1f, params.grid_resolution_m * 1.5f);
	planner_.params.waypoint_tolerance_m = params.waypoint_tolerance_m;

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

bool SpecificWorker::refresh_graph_state()
{
	if (!G)
		return false;

	if (graph_state_.room_name.empty())
	{
		if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
		{
			graph_state_.room_id = room_nodes.front().id();
			graph_state_.room_name = room_nodes.front().name();
		}
	}
	else if (!G->get_node(graph_state_.room_id).has_value())
	{
		graph_state_.room_id = 0;
		graph_state_.room_name.clear();
	}

	if (graph_state_.robot_name.empty())
	{
		if (const auto robot_nodes = G->get_nodes_by_type("robot"); !robot_nodes.empty())
		{
			graph_state_.robot_id = robot_nodes.front().id();
			graph_state_.robot_name = robot_nodes.front().name();
		}
	}
	else if (!G->get_node(graph_state_.robot_id).has_value())
	{
		graph_state_.robot_id = 0;
		graph_state_.robot_name.clear();
	}

	return graph_state_.ready();
}

void SpecificWorker::update_custom_widget(const std::optional<RobotPose> &robot_pose)
{
	if (custom_widget_ == nullptr)
		return;

	if (viewer_2d_ != nullptr)
	{
		Polygon display_path;
		if (current_plan_.has_value())
			display_path = current_plan_->room_path;
		if (robot_pose.has_value() and !display_path.empty())
			display_path.front() = robot_pose->pos;

		viewer_2d_->draw_room_polygon(room_polygon_);
		viewer_2d_->draw_lidar_points_from_buffer(params.max_lidar_draw_points);
		viewer_2d_->draw_path({
			.path = std::move(display_path),
			.inner_poly = inner_polygon_,
			.graph_nodes = current_plan_.has_value() ? current_plan_->graph_nodes : Polygon{},
			.obstacle_polys = obstacle_polygons_,
			.candidate_trajectories = last_mppi_trajectories_,
			.best_trajectory_idx = last_best_mppi_trajectory_idx_
		});
		if (!room_view_fitted_ && !room_polygon_.empty())
		{
			viewer_2d_->fit_view();
			room_view_fitted_ = true;
		}

		if (current_target_room_.has_value())
			viewer_2d_->update_target_marker(current_target_room_->x(), current_target_room_->y(), true);
		else
			viewer_2d_->update_target_marker(0.f, 0.f, false);

		if (robot_pose.has_value())
			viewer_2d_->update_robot(robot_pose->as_transform());
	}

	if (robot_pose.has_value())
	{
		const float theta_deg = robot_pose->theta * 180.f / static_cast<float>(M_PI);
		custom_widget_->set_pose_text(QStringLiteral("x %1 m   y %2 m   th %3 deg")
			.arg(robot_pose->pos.x(), 0, 'f', 2)
			.arg(robot_pose->pos.y(), 0, 'f', 2)
			.arg(theta_deg, 0, 'f', 1));
	}
	else
	{
		custom_widget_->set_pose_text(QStringLiteral("Waiting for robot pose"));
	}
}

std::optional<std::vector<Eigen::Vector2f>> SpecificWorker::read_room_polygon() const
{
	if (!G || graph_state_.room_id == 0)
		return std::nullopt;

	auto room_node = G->get_node(graph_state_.room_id);
	if (!room_node.has_value())
		return std::nullopt;

	auto polygon_x = G->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
	auto polygon_y = G->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());
	if (!polygon_x.has_value() || !polygon_y.has_value())
		return std::nullopt;

	const auto &xs = polygon_x.value().get();
	const auto &ys = polygon_y.value().get();
	const std::size_t count = std::min(xs.size(), ys.size());
	if (count < 3)
		return std::nullopt;

	std::vector<Eigen::Vector2f> polygon;
	polygon.reserve(count);
	for (std::size_t index = 0; index < count; ++index)
		polygon.emplace_back(xs[index], ys[index]);

	return polygon;
}

SpecificWorker::Polygons SpecificWorker::read_obstacle_polygons(std::uint64_t timestamp_ms) const
{
	Polygons obstacles;
	std::ostringstream report;
	if (!G || !inner_eigen_api_ || graph_state_.room_name.empty())
	{
		if (obstacle_debug_report_ != "Obstacle debug: graph or inner api not ready")
		{
			obstacle_debug_report_ = "Obstacle debug: graph or inner api not ready";
			std::print("{}\n", obstacle_debug_report_);
			std::fflush(stdout);
		}
		return obstacles;
	}

	const auto time_query = params.interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
	                                          : DSR::RT_API::TimeQuery::Nearest;

	auto obstacle_nodes = G->get_nodes_by_type("object");
	bool using_fallback_nodes = false;
	if (obstacle_nodes.empty())
	{
		using_fallback_nodes = true;
		for (const auto &node : G->get_nodes())
		{
			const auto width_attr = G->get_attrib_by_name<width_m_att>(node);
			const auto depth_attr = G->get_attrib_by_name<depth_m_att>(node);
			if (!width_attr.has_value() || !depth_attr.has_value())
				continue;

			const auto translation = inner_eigen_api_->transform(graph_state_.room_name,
			                                                    node.name(),
			                                                    timestamp_ms,
			                                                    "RT",
			                                                    time_query);
			const auto euler = inner_eigen_api_->get_euler_xyz_angles(graph_state_.room_name,
			                                                         node.name(),
			                                                         timestamp_ms,
			                                                         "RT",
			                                                         time_query);
			if (!translation.has_value() || !euler.has_value())
				continue;

			obstacle_nodes.push_back(node);
		}
	}
	report << "Obstacle debug: room='" << graph_state_.room_name << "' nodes=" << obstacle_nodes.size();
	if (using_fallback_nodes)
		report << " fallback=width_depth_rt";
	for (const auto &node : obstacle_nodes)
	{
		report << " | node='" << node.name() << "' type='" << node.type() << "'";
		const auto width_attr = G->get_attrib_by_name<width_m_att>(node);
		const auto depth_attr = G->get_attrib_by_name<depth_m_att>(node);
		if (!width_attr.has_value() || !depth_attr.has_value())
		{
			report << " missing_attrs";
			continue;
		}

		const float width_m = width_attr.value();
		const float depth_m = depth_attr.value();
		report << " size=(" << width_m << "," << depth_m << ")";
		if (width_m <= 0.f || depth_m <= 0.f)
		{
			report << " invalid_size";
			continue;
		}

		const auto translation = inner_eigen_api_->transform(graph_state_.room_name,
		                                                    node.name(),
		                                                    timestamp_ms,
		                                                    "RT",
		                                                    time_query);
		const auto euler = inner_eigen_api_->get_euler_xyz_angles(graph_state_.room_name,
		                                                         node.name(),
		                                                         timestamp_ms,
		                                                         "RT",
		                                                         time_query);
		if (!translation.has_value() || !euler.has_value())
		{
			report << " missing_rt";
			continue;
		}

		const Eigen::Vector2f center(static_cast<float>(translation->x()),
		                            static_cast<float>(translation->y()));
		const float yaw = static_cast<float>(euler->z());
		auto polygon = make_obstacle_polygon(center, yaw, width_m, depth_m);
		report << " center=(" << center.x() << "," << center.y() << ") yaw=" << yaw;
		if (!polygon.empty())
			report << " first_vertex=(" << polygon.front().x() << "," << polygon.front().y() << ")";
		obstacles.push_back(std::move(polygon));
	}
	report << " | drawn=" << obstacles.size();

	if (const auto report_str = report.str(); report_str != obstacle_debug_report_)
	{
		obstacle_debug_report_ = report_str;
		std::print("{}\n", obstacle_debug_report_);
		std::fflush(stdout);
	}

	return obstacles;
}

void SpecificWorker::update_active_obstacle_polygons(std::uint64_t timestamp_ms)
{
	obstacle_polygons_ = read_obstacle_polygons(timestamp_ms);
	if (temporary_obstacle_polygon_.has_value())
	{
		if (timestamp_ms < temporary_obstacle_expires_at_ms_)
			obstacle_polygons_.push_back(*temporary_obstacle_polygon_);
		else
		{
			temporary_obstacle_polygon_.reset();
			temporary_obstacle_expires_at_ms_ = 0;
		}
	}
	path_controller_.set_static_obstacles(obstacle_polygons_);
}

bool SpecificWorker::create_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
	                                                 const RobotPose &robot_pose,
	                                                 const Eigen::Vector2f &blockage_center_room,
	                                                 float blockage_radius_m)
{
	const auto [cloud_opt] = lidar_room_buffer_.read_last();
	if (!cloud_opt.has_value())
		return false;

	const auto &[xs_room, ys_room, zs_room] = cloud_opt.value();
	const std::size_t count = std::min({xs_room.size(), ys_room.size(), zs_room.size()});
	if (count == 0)
		return false;

	const Eigen::Affine2f robot_from_room = robot_pose.as_transform().inverse();
	std::vector<Eigen::Vector2f> candidate_points_room;
	candidate_points_room.reserve(count);

	const float max_front_distance = std::max(0.2f, params.temporary_obstacle_front_distance_m);
	const float half_width = std::max(0.1f, params.temporary_obstacle_half_width_m);
	const float cluster_margin = std::max(0.f, params.temporary_obstacle_cluster_margin_m);
	const float max_blockage_distance = std::max(blockage_radius_m + cluster_margin, cluster_margin);

	for (std::size_t index = 0; index < count; ++index)
	{
		const float x_room = xs_room[index];
		const float y_room = ys_room[index];
		const float z_room = zs_room[index];
		if (!std::isfinite(x_room) || !std::isfinite(y_room) || !std::isfinite(z_room))
			continue;
		if (z_room < 0.05f || z_room > 1.8f)
			continue;

		const Eigen::Vector2f point_room(x_room, y_room);
		const Eigen::Vector2f point_robot = robot_from_room * point_room;
		if (point_robot.y() <= params.clearance_m * 0.5f)
			continue;
		if (point_robot.y() > max_front_distance)
			continue;
		if (std::abs(point_robot.x()) > half_width)
			continue;
		if (max_blockage_distance > 0.f && (point_room - blockage_center_room).norm() > max_blockage_distance)
			continue;

		candidate_points_room.push_back(point_room);
	}

	if (static_cast<int>(candidate_points_room.size()) < std::max(3, params.temporary_obstacle_min_points))
		return false;

	auto hull = convex_hull(std::move(candidate_points_room));
	if (hull.size() < 3)
		return false;

	temporary_obstacle_polygon_ = inflate_convex_polygon(hull, std::max(0.f, params.temporary_obstacle_padding_m));
	temporary_obstacle_expires_at_ms_ = timestamp_ms + params.temporary_obstacle_ttl_ms;
	update_active_obstacle_polygons(timestamp_ms);
	return true;
}

SpecificWorker::Polygon SpecificWorker::make_obstacle_polygon(const Eigen::Vector2f &center,
	                                                          float yaw,
	                                                          float width_m,
	                                                          float depth_m) const
{
	const float half_width = width_m * 0.5f;
	const float half_depth = depth_m * 0.5f;
	const Eigen::Rotation2Df rotation(yaw);

	Polygon polygon;
	polygon.reserve(4);
	for (const Eigen::Vector2f &corner : {Eigen::Vector2f(-half_width, -half_depth),
	                                     Eigen::Vector2f(half_width, -half_depth),
	                                     Eigen::Vector2f(half_width, half_depth),
	                                     Eigen::Vector2f(-half_width, half_depth)})
		polygon.push_back(center + rotation * corner);

	return polygon;
}

std::optional<SpecificWorker::RobotPose> SpecificWorker::read_robot_pose_in_room(std::uint64_t timestamp_ms) const
{
	if (!inner_eigen_api_ || !graph_state_.ready())
		return std::nullopt;

	const auto time_query = params.interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
	                                          : DSR::RT_API::TimeQuery::Nearest;
	const std::uint64_t query_ts = last_lidar_timestamp_ms_.value_or(timestamp_ms);
	auto room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
	                                                               graph_state_.robot_name,
	                                                               query_ts,
	                                                               "RT",
	                                                               time_query);
	if (!room_T_robot.has_value())
		room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
	                                                          graph_state_.robot_name,
	                                                          timestamp_ms,
	                                                          "RT",
	                                                          time_query);
	if (!room_T_robot.has_value())
		room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
	                                                          graph_state_.robot_name,
	                                                          0,
	                                                          "RT",
	                                                          time_query);
	if (!room_T_robot.has_value())
		return std::nullopt;

	const auto &matrix = room_T_robot->matrix();

	RobotPose pose;
	pose.pos = Eigen::Vector2f(static_cast<float>(matrix(0, 3)), static_cast<float>(matrix(1, 3)));
	pose.theta = std::atan2(static_cast<float>(matrix(1, 0)), static_cast<float>(matrix(0, 0)));
	return pose;
}

std::optional<SpecificWorker::PoseUncertainty> SpecificWorker::read_pose_uncertainty() const
{
	if (!G || !graph_state_.ready())
		return std::nullopt;

	auto rt_edge = G->get_edge(graph_state_.robot_id, graph_state_.room_id, "RT");
	if (!rt_edge.has_value())
		rt_edge = G->get_edge(graph_state_.room_id, graph_state_.robot_id, "RT");
	if (!rt_edge.has_value())
		return std::nullopt;

	auto covariance_att = G->get_attrib_by_name<rt_covariance_att>(rt_edge.value());
	if (!covariance_att.has_value())
		return std::nullopt;

	const auto &flat_covariance = covariance_att.value().get();
	if (flat_covariance.size() < 15)
		return std::nullopt;

	PoseUncertainty uncertainty;
	uncertainty.xy_std_m = std::sqrt(std::max(0.f, std::max(flat_covariance[0], flat_covariance[7])));
	uncertainty.theta_std_rad = std::sqrt(std::max(0.f, flat_covariance[14]));
	return uncertainty;
}

void SpecificWorker::apply_uncertainty_speed_limit(float &adv_mps, float &side_mps, float &rot_rps) const
{
	const auto uncertainty = read_pose_uncertainty();
	if (!uncertainty.has_value())
		return;

	const float current_trans_speed_mps = std::hypot(adv_mps, side_mps);
	const float current_rot_speed_rps = std::abs(rot_rps);
	const float forward_ratio = (current_trans_speed_mps > 1e-3f)
		? std::clamp(std::abs(adv_mps) / current_trans_speed_mps, 0.f, 1.f)
		: 0.f;
	const float lateral_ratio = (current_trans_speed_mps > 1e-3f)
		? std::clamp(std::abs(side_mps) / current_trans_speed_mps, 0.f, 1.f)
		: 0.f;
	const float turning_ratio = std::clamp(current_rot_speed_rps / std::max(0.12f, 0.35f * params.max_rot_speed_rps), 0.f, 1.f);
	const float straight_motion_ratio = std::clamp(forward_ratio * (1.f - lateral_ratio) * (1.f - turning_ratio), 0.f, 1.f);

	const float effective_xy_slow_m = params.pose_xy_std_slow_m * (1.f + 1.0f * straight_motion_ratio);
	const float effective_xy_stop_m = params.pose_xy_std_stop_m * (1.f + 0.6f * straight_motion_ratio);

	float adv_scale = ramp_uncertainty_scale(uncertainty->xy_std_m,
		effective_xy_slow_m,
		effective_xy_stop_m,
		params.min_adv_speed_scale);
	float rot_scale = ramp_uncertainty_scale(uncertainty->theta_std_rad,
		params.pose_theta_std_slow_rad,
		params.pose_theta_std_stop_rad,
		params.min_rot_speed_scale);

	const float horizon_s = std::max(1e-3f, params.uncertainty_prediction_horizon_s);
	const float xy_growth = std::max(1e-3f,
		params.pose_xy_std_growth_per_mps / (1.f + 1.5f * straight_motion_ratio));
	const float theta_growth = std::max(1e-3f, params.pose_theta_std_growth_per_rps);

	if (current_trans_speed_mps > 1e-3f)
	{
		const float xy_margin = std::max(0.f, effective_xy_stop_m - uncertainty->xy_std_m);
		const float max_predicted_trans_speed_mps = xy_margin / (horizon_s * xy_growth);
		const float predictive_adv_scale = std::clamp(max_predicted_trans_speed_mps / current_trans_speed_mps,
			params.min_adv_speed_scale,
			1.f);
		adv_scale = std::min(adv_scale, predictive_adv_scale);
	}

	if (current_rot_speed_rps > 1e-3f)
	{
		const float theta_margin = std::max(0.f, params.pose_theta_std_stop_rad - uncertainty->theta_std_rad);
		const float max_predicted_rot_speed_rps = theta_margin / (horizon_s * theta_growth);
		const float predictive_rot_scale = std::clamp(max_predicted_rot_speed_rps / current_rot_speed_rps,
			params.min_rot_speed_scale,
			1.f);
		rot_scale = std::min(rot_scale, predictive_rot_scale);
	}

	const float coupled_adv_scale = std::pow(std::max(rot_scale, 0.f),
		std::max(0.f, params.adv_rotation_coupling_exponent) * turning_ratio);
	adv_scale = std::min(adv_scale, coupled_adv_scale);

	adv_mps *= adv_scale;
	side_mps *= adv_scale;
	rot_rps = preserve_sign_clamp(rot_rps, current_rot_speed_rps * rot_scale);
}

std::optional<SpecificWorker::TargetInfo> SpecificWorker::read_target_in_room(std::uint64_t timestamp_ms)
{
	if (!G || !inner_eigen_api_ || !graph_state_.ready())
		return std::nullopt;

	// 1) Main target source: affordance manager protocol.
	if (const auto affordance_target = affordance_manager_.select_target(G); affordance_target.has_value())
	{
		TargetInfo info;
		info.node_id = affordance_target->node_id;
		info.node_name = affordance_target->node_name;
		info.room_pos = affordance_target->room_pos;
		info.yaw_rad = affordance_target->yaw_rad;
		info.epistemic_gain = affordance_target->epistemic_gain;
		info.epistemic_pending = affordance_target->epistemic_pending;
		info.from_affordance = true;
		return info;
	}

	// 2) Fallback: legacy target edge workflow.
	const auto edges = G->get_edges_by_type(params.target_edge_type);
	if (edges.empty())
		return std::nullopt;

	const auto time_query = params.interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
		                                          : DSR::RT_API::TimeQuery::Nearest;

	for (const auto &edge : edges)
	{
		uint64_t target_id = edge.to();
		if (edge.to() == graph_state_.room_id || edge.to() == graph_state_.robot_id)
			target_id = edge.from();
		else if ((edge.from() == graph_state_.room_id || edge.from() == graph_state_.robot_id) && edge.to() != 0)
			target_id = edge.to();

		if (target_id == 0 || target_id == graph_state_.room_id || target_id == graph_state_.robot_id)
			continue;

		const auto target_name = G->get_name_from_id(target_id);
		if (!target_name.has_value())
			continue;

		const auto translation = inner_eigen_api_->transform(graph_state_.room_name, *target_name, timestamp_ms, "RT", time_query);
		if (!translation.has_value())
			continue;

		TargetInfo info;
		info.node_id = target_id;
		info.node_name = *target_name;
		info.room_pos = Eigen::Vector2f(static_cast<float>(translation->x()), static_cast<float>(translation->y()));
		info.from_affordance = false;
		return info;
	}

	return std::nullopt;
}

void SpecificWorker::execute_plan(const RobotPose &robot_pose)
{
	if (!current_plan_.has_value())
	{
		path_controller_.stop();
		stop_robot();
		return;
	}

	const auto &boundary_polygon = inner_polygon_.empty() ? room_polygon_ : inner_polygon_;
	if (boundary_polygon.size() >= 3)
		path_controller_.set_room_boundary(boundary_polygon);

	const auto control_output = path_controller_.compute(robot_pose.as_transform());
	last_mppi_trajectories_ = control_output.trajectories_room;
	last_best_mppi_trajectory_idx_ = control_output.best_trajectory_idx;
	if (control_output.path_blocked)
	{
		create_temporary_lidar_obstacle(current_time_ms(),
		                              robot_pose,
		                              control_output.blockage_center_room,
		                              control_output.blockage_radius);
		current_plan_.reset();
		path_controller_.stop();
		stop_robot();
		return;
	}

	if (control_output.goal_reached)
	{
		affordance_manager_.mark_reached(G);

		current_plan_.reset();
		last_target_info_.reset();
		active_target_id_ = 0;
		current_target_room_.reset();
		if (manual_target_room_.has_value())
			manual_target_room_.reset();
		manual_target_origin_room_.reset();
		manual_target_dirty_ = false;
		path_controller_.stop();
		stop_robot();
		return;
	}

	if (!path_controller_.is_active())
	{
		stop_robot();
		return;
	}

	float adv_mps = control_output.adv;
	float side_mps = control_output.side;
	float rot_rps = -control_output.rot;
	apply_uncertainty_speed_limit(adv_mps, side_mps, rot_rps);
	if (std::abs(adv_mps) < 5e-4f && std::abs(side_mps) < 5e-4f && std::abs(rot_rps) < 1e-3f)
	{
		stop_robot();
		return;
	}

	send_speed_command(adv_mps, side_mps, rot_rps);
}

void SpecificWorker::set_manual_target(const QPointF &point)
{
	manual_target_room_ = Eigen::Vector2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
	current_target_room_ = manual_target_room_;
	manual_target_origin_room_.reset();
	if (inner_eigen_api_ and (graph_state_.ready() || refresh_graph_state()))
	{
		const auto now_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count());
		if (const auto robot_pose = read_robot_pose_in_room(now_ms); robot_pose.has_value())
			manual_target_origin_room_ = robot_pose->pos;
	}
	manual_target_dirty_ = true;
	current_plan_.reset();
	last_target_info_.reset();
	active_target_id_ = 0;
	path_controller_.stop();
	hibernationTick();
}

void SpecificWorker::clear_manual_target()
{
	manual_target_room_.reset();
	current_target_room_.reset();
	affordance_manager_.clear_current();
	manual_target_origin_room_.reset();
	manual_target_dirty_ = false;
	current_plan_.reset();
	last_target_info_.reset();
	active_target_id_ = 0;
	stop_robot();
	hibernationTick();
}

void SpecificWorker::publish_robot_reference_speed(float adv_mps, float side_mps, float rot_rps, std::uint64_t timestamp_ms)
{
	if (!G || graph_state_.robot_id == 0)
		return;

	auto robot_node_opt = G->get_node(graph_state_.robot_id);
	if (!robot_node_opt.has_value())
		return;

	auto robot_node = robot_node_opt.value();
	auto &attrs = robot_node.attrs();
	attrs[kRobotRefAdvSpeedAttr] = DSR::Attribute{adv_mps, timestamp_ms, static_cast<std::uint32_t>(agent_id)};
	attrs[kRobotRefSideSpeedAttr] = DSR::Attribute{side_mps, timestamp_ms, static_cast<std::uint32_t>(agent_id)};
	attrs[kRobotRefRotSpeedAttr] = DSR::Attribute{rot_rps, timestamp_ms, static_cast<std::uint32_t>(agent_id)};
	attrs[kRobotRefSpeedTimestampAttr] = DSR::Attribute{timestamp_ms, timestamp_ms, static_cast<std::uint32_t>(agent_id)};
	if (!G->update_node(robot_node))
		qWarning() << "Controller failed to publish robot reference speed attrs to DSR";
}

void SpecificWorker::send_speed_command(float adv_mps, float side_mps, float rot_rps)
{
	// Convert to mm/s for the robot proxy and for display.
	const float adv_mm_s = adv_mps * 1000.f;
	const float side_mm_s = side_mps * 1000.f;

	if (custom_widget_)
		custom_widget_->set_cmd_vel_text(QStringLiteral("adv %1 mm/s   side %2 mm/s   rot %3 rad/s")
			.arg(adv_mm_s, 0, 'f', 0)
			.arg(side_mm_s, 0, 'f', 0)
			.arg(rot_rps, 0, 'f', 2));

	if (!omnirobot_proxy)
		return;

	const Eigen::Vector3f cmd(adv_mps, side_mps, rot_rps);
	if (has_last_speed_command_ && (cmd - last_speed_command_).cwiseAbs().maxCoeff() < 1e-4f)
		return;

	const auto timestamp_ms = current_time_ms();
	publish_robot_reference_speed(adv_mps, side_mps, rot_rps, timestamp_ms);

	try
	{
		omnirobot_proxy->setSpeedBase(side_mm_s, adv_mm_s, rot_rps);
		last_speed_command_ = cmd;
		has_last_speed_command_ = true;
		stop_command_latched_ = false;
	}
	catch(const Ice::Exception &e)
	{
		qWarning() << "Controller setSpeedBase failed:" << e.what();
	}
}

void SpecificWorker::stop_robot()
{
	path_controller_.stop();
	last_mppi_trajectories_.clear();
	last_best_mppi_trajectory_idx_ = -1;
	if (custom_widget_)
		custom_widget_->set_cmd_vel_text(QStringLiteral("adv 0 mm/s   side 0 mm/s   rot 0.00 rad/s"));

	if (stop_command_latched_)
		return;

	if (!omnirobot_proxy)
		return;

	publish_robot_reference_speed(0.f, 0.f, 0.f, current_time_ms());

	try
	{
		omnirobot_proxy->stopBase();
		stop_command_latched_ = true;
		has_last_speed_command_ = false;
	}
	catch(const Ice::Exception &) {}
}

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
	if (!G)
		return;

	auto node_opt = G->get_node(id);
	if (!node_opt.has_value())
		return;

	if (node_opt->name() == params.lidar_name)
	{
		if (!inner_eigen_api_)
			return;
		if (!graph_state_.ready() and !refresh_graph_state())
			return;

		auto node_copy = std::move(node_opt.value());
		auto &attrs = node_copy.attrs();
		auto xs_it = attrs.find(laser_X_att::attr_name.data());
		auto ys_it = attrs.find(laser_Y_att::attr_name.data());
		auto zs_it = attrs.find(laser_Z_att::attr_name.data());
		if (xs_it == attrs.end() or ys_it == attrs.end() or zs_it == attrs.end())
			return;

		auto ts_it = attrs.find(laser_timestamp_att::attr_name.data());
		const std::uint64_t timestamp_ms = ts_it != attrs.end()
			? static_cast<std::uint64_t>(std::max<std::int64_t>(0, static_cast<std::int64_t>(ts_it->second.uint64())))
			: 0ULL;
		last_lidar_timestamp_ms_ = timestamp_ms;
		const auto interp = params.interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
			                                     : DSR::RT_API::TimeQuery::Nearest;
		const auto room_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
		                                                                       node_copy.name(),
		                                                                       timestamp_ms,
		                                                                       "RT",
		                                                                       interp);
		if (!room_from_lidar.has_value())
			return;

		auto xs = std::move(xs_it->second.float_vec());
		auto ys = std::move(ys_it->second.float_vec());
		auto zs = std::move(zs_it->second.float_vec());
		const std::size_t raw_count = std::min({xs.size(), ys.size(), zs.size()});
		if (raw_count == 0)
			return;

		const auto room_from_lidar_matrix = room_from_lidar->matrix();
		lidar_room_buffer_.put<0>(
			rc::RawLidarPointVectors{
				.xs = std::move(xs),
				.ys = std::move(ys),
				.zs = std::move(zs)
			},
			timestamp_ms,
			[room_from_lidar_matrix, raw_count](rc::RawLidarPointVectors &&raw_points, rc::LidarPointVectors &room_points)
			{
				auto &[room_xs, room_ys, room_zs] = room_points;
				const auto &xs_in = raw_points.xs;
				const auto &ys_in = raw_points.ys;
				const auto &zs_in = raw_points.zs;
				const std::size_t count = std::min({raw_count, xs_in.size(), ys_in.size(), zs_in.size()});
				if (count == 0)
					return;

				const double m00 = room_from_lidar_matrix(0, 0);
				const double m01 = room_from_lidar_matrix(0, 1);
				const double m02 = room_from_lidar_matrix(0, 2);
				const double m03 = room_from_lidar_matrix(0, 3);
				const double m10 = room_from_lidar_matrix(1, 0);
				const double m11 = room_from_lidar_matrix(1, 1);
				const double m12 = room_from_lidar_matrix(1, 2);
				const double m13 = room_from_lidar_matrix(1, 3);
				const double m20 = room_from_lidar_matrix(2, 0);
				const double m21 = room_from_lidar_matrix(2, 1);
				const double m22 = room_from_lidar_matrix(2, 2);
				const double m23 = room_from_lidar_matrix(2, 3);

				room_xs.reserve(count);
				room_ys.reserve(count);
				room_zs.reserve(count);

				for (std::size_t index = 0; index < count; ++index)
				{
					const float x = xs_in[index];
					const float y = ys_in[index];
					const float z = zs_in[index];
					if (!std::isfinite(x) or !std::isfinite(y) or !std::isfinite(z))
						continue;
					if (z < 0.15f or z > 1.6f)
						continue;

					room_xs.push_back(static_cast<float>(m00 * x + m01 * y + m02 * z + m03));
					room_ys.push_back(static_cast<float>(m10 * x + m11 * y + m12 * z + m13));
					room_zs.push_back(static_cast<float>(m20 * x + m21 * y + m22 * z + m23));
				}
			});
		return;
	}

	if (type == "room" or type == "robot")
		hibernationTick();
}

void SpecificWorker::modify_edge_slot(std::uint64_t, std::uint64_t, const std::string &type)
{
	if (type == params.target_edge_type || type == "RT")
		hibernationTick();
}


void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
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

