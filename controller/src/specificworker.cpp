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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
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
		path_controller_.set_lidar_buffer(&lidar_room_buffer_);
		viewer_2d_->show();
		connect(viewer_2d_.get(), &rc::Viewer2D::new_mouse_coordinates, this,
		        [this](const QPointF &point) { set_manual_target(point); });
		connect(viewer_2d_.get(), &rc::Viewer2D::right_click, this,
		        [this](const QPointF &) { clear_manual_target(); });
		connect(custom_widget_->lidar_toggle_btn, &QPushButton::toggled, this,
		        [this](bool checked) { if (viewer_2d_) viewer_2d_->set_lidar_visible(checked); });
	}
	update_custom_widget(std::nullopt);

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}


void SpecificWorker::compute()
{
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

	update_custom_widget(step->robot_pose);
	execute_plan(step->robot_pose);
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
	obstacle_polygons_ = read_obstacle_polygons(timestamp_ms);
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
		const bool use_snapped_manual_origin = manual_target_dirty_ and manual_target_origin_room_.has_value();
		if (use_snapped_manual_origin)
			step.plan_origin = *manual_target_origin_room_;
		step.target_changed = manual_target_dirty_ || !current_plan_.has_value();
		manual_target_dirty_ = false;
		active_target_id_ = 0;
		target_wait_logged_ = false;
		return step;
	}

	const auto target = read_target_in_room(timestamp_ms);
	if (!target.has_value())
	{
		if (!target_wait_logged_)
		{
			qInfo() << "Controller waiting for a target edge in DSR";
			target_wait_logged_ = true;
		}
		current_plan_.reset();
		active_target_id_ = 0;
		update_custom_widget(robot_pose);
		stop_robot();
		return std::nullopt;
	}

	target_wait_logged_ = false;
	step.target = *target;
	step.target_changed = active_target_id_ != target->node_id;
	if (step.target_changed)
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
			.obstacle_polys = obstacle_polygons_
		});
		if (!room_view_fitted_ && !room_polygon_.empty())
		{
			viewer_2d_->fit_view();
			room_view_fitted_ = true;
		}

		if (!current_plan_.has_value() && manual_target_room_.has_value())
			viewer_2d_->update_target_marker(manual_target_room_->x(), manual_target_room_->y(), true);
		else if (!current_plan_.has_value())
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

	const auto translation = inner_eigen_api_->transform(graph_state_.room_name, graph_state_.robot_name, timestamp_ms, "RT");
	const auto euler = inner_eigen_api_->get_euler_xyz_angles(graph_state_.room_name, graph_state_.robot_name, timestamp_ms, "RT");
	if (!translation.has_value() || !euler.has_value())
		return std::nullopt;

	RobotPose pose;
	pose.pos = Eigen::Vector2f(static_cast<float>(translation->x()), static_cast<float>(translation->y()));
	pose.theta = static_cast<float>(euler->z());
	return pose;
}

std::optional<SpecificWorker::TargetInfo> SpecificWorker::read_target_in_room(std::uint64_t timestamp_ms) const
{
	if (!G || !inner_eigen_api_ || !graph_state_.ready())
		return std::nullopt;

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
	if (control_output.path_blocked)
	{
		current_plan_.reset();
		path_controller_.stop();
		stop_robot();
		return;
	}

	if (control_output.goal_reached)
	{
		current_plan_.reset();
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

	try
	{
		omnirobot_proxy->setSpeedBase(control_output.adv * 1000.f, control_output.side * 1000.f, control_output.rot);
	}
	catch(const Ice::Exception &e)
	{
		qWarning() << "Controller setSpeedBase failed:" << e.what();
	}
}

void SpecificWorker::set_manual_target(const QPointF &point)
{
	manual_target_room_ = Eigen::Vector2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
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
	active_target_id_ = 0;
	path_controller_.stop();
	hibernationTick();
}

void SpecificWorker::clear_manual_target()
{
	manual_target_room_.reset();
	manual_target_origin_room_.reset();
	manual_target_dirty_ = false;
	current_plan_.reset();
	active_target_id_ = 0;
	stop_robot();
	hibernationTick();
}

void SpecificWorker::stop_robot()
{
	path_controller_.stop();

	if (!omnirobot_proxy)
		return;

	try { omnirobot_proxy->stopBase(); }
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

