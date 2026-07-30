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
#include <print>
#include "specificworker.h"

#include "../../common/robust_metrics/robust_metrics.h"
#include "../../common/media_transport/lidar_plane_reader.h"

#include "custom_widget.h"
#include <fps/fps.h>

#include <QTimer>

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
	request_shutdown();
	qInfo() << "Destroying SpecificWorker";
	/*
	for (auto const& [name, g] : Graphs) {
	    g->write_to_json_file("./"+agent_name+"_"+name+".json");
	}
	*/
}

void SpecificWorker::request_shutdown()
{
	if (shutting_down_.exchange(true))
		return;

	display_.save_window_geometry();   // persist the standalone planner window (GUI thread)
	cleanup_owned_nodes();
	stop_robot();
}


void SpecificWorker::initialize()
{
    qInfo() << "initialize worker";
	GenericWorker::initialize();

	 // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


	presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
	// Colour this agent's node in the graph view by its live health: the coordinator already
	// publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
	// Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
	// cannot break it.
	presence_coordinator_.attach_state_machine(&statemachine);
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
	        // Do NOT run compute() here: this hook executes on the presence-SM /
	        // GUI thread and compute() (MPPI) can block for hundreds of ms. Just
	        // wake the control thread, which runs compute() asynchronously.
	        control_operating_.store(true, std::memory_order_release);
	        control_cv_.notify_one();
	    },
	    .on_degraded_enter = [this]()
	    {
	        if (shutting_down_.load())
	            return;
	        QString m;
	        for (const auto &label : presence_coordinator_.missing_required_names())
	            m += " " + QString::fromStdString(label);
	        // DEBOUNCE — do NOT cleanup/exit on entry. A transient required-peer flap
	        // (startup handshake, brief DSR node churn, a peer restarting) fires
	        // presenceLost momentarily and then recovers; tearing down here deletes our
	        // own node ('controller 8') and disconnects the graph, leaving a broken
	        // half-shutdown. Wait a grace period and only shut down if a required peer
	        // is STILL genuinely missing. (Matches bottle_concept's presence protocol.)
	        qWarning() << "[SM] -> DEGRADED: required peer lost (missing:" << m.trimmed()
	                   << ") —" << kRequiredLossGraceMs << "ms grace before shutdown.";
	        QTimer::singleShot(kRequiredLossGraceMs, this, [this]()
	        {
	            if (shutting_down_.load())
	                return;
	            if (presence_coordinator_.all_required_ready())
	            {
	                qInfo() << "[SM] required peers recovered during grace — staying alive.";
	                return;
	            }
	            qWarning() << "[SM] required peer still missing after grace — shutting down cleanly.";
	            cleanup_owned_nodes();
	            QTimer::singleShot(500, QCoreApplication::instance(), SLOT(quit()));
	        });
	    },
	});
	presence_coordinator_.start();

	QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
	                 this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

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

	// Media-plane LiDAR: the graph is already loaded here, so verify the producer's
	// sensor node + its media descriptor exist and bring up the subscriber NOW — on
	// the main thread, BEFORE the control thread starts. This keeps every graph access
	// on the main thread during startup (a worker thread touching the graph while it
	// is still joining corrupts the DSR / AgentInfo heartbeat). The DDS domain + topic
	// come from the descriptor JSON; if the node/descriptor is absent we log and fall
	// back to the DSR laser_* path.
	init_lidar_media();

	// DSR graph update signals — QUEUED connections only. NEVER Qt::DirectConnection:
	// DSR emits update_node_signal/update_edge_signal from the raw FastDDS reader threads,
	// and a DirectConnection runs the slot ON that reader thread, corrupting the heap
	// under peer graph churn (symptom: smashed AgentInfo heartbeat — get_agent_id on a
	// garbage DSRGraph — once required peers come online). A plain (default/Auto →
	// Queued) connection marshals the slot to the main thread and is safe: verified with
	// 5/5 clean startups under live peers on 2026-06-23, after the DirectConnection
	// version crashed every time. (See [[dsr-graphviewer-crash-fixes]].)
	connect(G.get(), &DSR::DSRGraph::update_node_signal, this,
	        &SpecificWorker::modify_node_slot, Qt::QueuedConnection);
	connect(G.get(), &DSR::DSRGraph::update_edge_signal, this,
	        &SpecificWorker::modify_edge_slot, Qt::QueuedConnection);

	// The planner GUI (ControllerDisplay) is its OWN top-level window, NOT docked into the DSR graph
	// viewer — so the agent runs with Agent.graph=false (no DSRViewer created). See ControllerDisplay.
	// User-input callbacks fire on the GUI thread but mutate session_/path_controller_
	// state owned by the control thread. Marshal them through the command queue so
	// they execute on the control thread, avoiding data races with compute().
	display_.initialize(obstacle_tracker_.lidar_buffer(),
	                  [this](const QPointF &point) { enqueue_command([this, point]() { set_manual_target(point); }); },
	                  [this]() { enqueue_command([this]() { clear_manual_target(); }); },
	                  [this](bool checked)
	                  {
	                      enqueue_command([this, checked]()
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
	                  });
	update_custom_widget(std::nullopt);

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 

	// GUI-thread render timer: draws the latest snapshot staged by the control
	// thread. Decoupled from compute(), so rendering and the event loop stay
	// responsive even while MPPI is busy.
	render_timer_ = new QTimer(this);
	connect(render_timer_, &QTimer::timeout, this, [this]() { display_.present(); });
	render_timer_->start(33);

	// Start the control thread last, once all dependencies are wired up.
	control_running_.store(true, std::memory_order_release);
	control_thread_ = std::thread(&SpecificWorker::control_loop, this);
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

	// Verify the output-rate guarantee rather than assume it. period_max is the number that matters: it is the
	// longest the base went without a command, and it must stay near VelocityOutputPeriodMs no matter how badly
	// this cycle overran. cmd_age_max / scale_min show how stale the planner got and how much the freshness
	// term attenuated it — a scale_min well under 1 means compute() stalled and the robot was coasting down.
	if (const auto stats = motion_commander_.take_output_rate_stats(); stats.ticks > 0)
	{
		static std::uint64_t last_rate_log_ms = 0;
		const auto now_ms = static_cast<std::uint64_t>(
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch()).count());
		if (now_ms - last_rate_log_ms >= 5000)
		{
			last_rate_log_ms = now_ms;
			std::println("[vel-out] {} ticks | period mean {:.1f} ms max {:.1f} ms | cmd age max {:.0f} ms | "
			             "freshness scale min {:.2f}",
			             stats.ticks, stats.period_mean_ms, stats.period_max_ms,
			             stats.cmd_age_max_ms, stats.scale_min);
		}
	}

	auto update_selected_affordance_label = [this]()
	{
		const auto current_affordance_name = affordance_manager_.current_name();
		// Track the previous DISTINCT non-empty selection so the label shows the last flip.
		if (!current_affordance_name.empty() && current_affordance_name != last_selected_affordance_)
		{
			prev_selected_affordance_ = last_selected_affordance_;
			last_selected_affordance_ = current_affordance_name;
		}
		const QString cur  = current_affordance_name.empty() ? QStringLiteral("none")
		                                                     : QString::fromStdString(current_affordance_name);
		const QString prev = prev_selected_affordance_.empty() ? QStringLiteral("—")
		                                                       : QString::fromStdString(prev_selected_affordance_);
		display_.set_selected_affordance_text(cur + QStringLiteral("   (prev: ") + prev + QStringLiteral(")"));
	};
	update_selected_affordance_label();

	// Feed the EFE time-series panel (below the 2D view): score (gain − λ·dist) + raw gain (ΔH) per
	// affordance evaluated last cycle. The gap between the two lines is λ·dist.
	{
		std::vector<ControllerDisplay::AffordanceEfeSample> efe;
		for (const auto &c : affordance_manager_.last_candidates())
			efe.push_back({c.node_name, c.gain, c.efe_score});
		display_.update_affordance_efe(efe);
	}

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
	update_selected_affordance_label();
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
	                                  planner_,
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
	                                  display_,
	                                  [this]() { return current_time_ms(); });
}

/////////////////////////////////////////////////////////////////
void SpecificWorker::load_params()
{
	load_optional_cast<double>("Planner.Clearance", params.clearance_m);
	load_optional_cast<double>("Planner.GridResolution", params.grid_resolution_m);
	load_optional_cast<double>("Planner.ConnectionRadius", params.connection_radius_m);
	// Grounded EFE affordance selection (common/affordance_manager): nav-cost weight (nats/m) +
	// commitment hysteresis (nats). G = λ_cost·dist − epistemic_gain; the room/table choice is now
	// in one information currency instead of a hard table>room priority.
	{
		double aff_lambda_cost = 0.2, aff_switch_margin = 0.5;
		load_optional_cast<double>("Controller.AffordanceLambdaCost", aff_lambda_cost);
		load_optional_cast<double>("Controller.AffordanceSwitchMargin", aff_switch_margin);
		affordance_manager_.set_selection_params(static_cast<float>(aff_lambda_cost),
		                                         static_cast<float>(aff_switch_margin));
	}
	load_optional_cast<double>("Controller.WaypointTolerance", params.waypoint_tolerance_m);
	load_optional_cast<double>("Controller.MaxAdvSpeed", params.max_adv_speed_mps);
	load_optional_cast<double>("Controller.MaxRotSpeed", params.max_rot_speed_rps);
	load_optional_cast<double>("Controller.PosGain", params.pos_gain);
	load_optional_cast<double>("Controller.RotGain", params.rot_gain);
	load_optional_cast<double>("Controller.VelocityOutputPeriodMs", params.velocity_output_period_ms);
	load_optional_cast<double>("Controller.CommandFreshnessTauMs", params.command_freshness_tau_ms);
	load_optional("Transforms.interpolate_rt", params.interpolate_rt);
	load_optional("Transforms.overlay_extrapolate_to_now", params.overlay_extrapolate_to_now);
	load_optional("Transforms.overlay_draw_one_frame_old", params.overlay_draw_one_frame_old);
	load_optional_cast<double>("Transforms.overlay_extrapolation_max_dt_s", params.overlay_extrapolation_max_dt_s);
	load_optional("Transforms.overlay_csv_path", params.overlay_csv_path);
	load_optional("Viewer2D.MaxLidarDrawPoints", params.max_lidar_draw_points);
	load_optional("Lidar.Name", params.lidar_name);
	load_optional("Lidar.HeliosName", params.lidar_helios_name);
	load_optional("Lidar.BpearlName", params.lidar_bpearl_name);
	load_optional("Lidar.StallTimeoutMs", params.lidar_stall_timeout_ms);
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
	load_optional_cast<double>("Controller.TemporaryObstacleMinHeight", params.temporary_obstacle_min_height_m);
	load_optional_cast<double>("Controller.TemporaryObstacleMaxHeight", params.temporary_obstacle_max_height_m);
	load_optional_cast<double>("Controller.UnmodelledScanMinZ", params.unmodelled_scan_min_z_m);
	load_optional_cast<double>("Controller.UnmodelledScanMaxZ", params.unmodelled_scan_max_z_m);
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

	// Controller-side LiDAR obstacle creation (false ⇒ residual_concept is the sole obstacle source).
	load_optional("Controller.ObstacleCreationEnabled", params.obstacle_creation_enabled);

	// Affordance servo ("lock-on") executor — HOW only; WHAT/WHEN is per-affordance (contract).
	load_optional("Controller.LockOnEnabled", params.lockon_enabled);
	load_optional_cast<double>("Controller.LockOnSweepSpeedMps", params.lockon_sweep_speed_mps);
	load_optional_cast<double>("Controller.LockOnSweepRangeM",   params.lockon_sweep_range_m);
	load_optional_cast<double>("Controller.LockOnOffsetTol",    params.lockon_offset_tol);
	load_optional_cast<double>("Controller.LockOnKYaw",         params.lockon_k_yaw);
	load_optional_cast<double>("Controller.LockOnMaxYawRps",    params.lockon_max_yaw_rps);
	load_optional_cast<double>("Controller.LockOnDitherYawRps", params.lockon_dither_yaw_rps);
	load_optional_cast<double>("Controller.LockOnSettleMs",     params.lockon_settle_ms);
	load_optional_cast<double>("Controller.LockOnStepMs",       params.lockon_step_ms);
	load_optional("Controller.LockOnMaxAttempts",              params.lockon_max_attempts);

	// Physical-stuck detection + reverse-and-turn escape (see ControllerParams).
	load_optional("Controller.StuckRecoveryEnabled",            params.stuck_recovery_enabled);
	load_optional_cast<double>("Controller.StuckCmdLinEps",     params.stuck_cmd_lin_eps);
	load_optional_cast<double>("Controller.StuckCmdRotEps",     params.stuck_cmd_rot_eps);
	load_optional_cast<double>("Controller.StuckMeasLinEps",    params.stuck_meas_lin_eps);
	load_optional_cast<double>("Controller.StuckMeasRotEps",    params.stuck_meas_rot_eps);
	load_optional_cast<double>("Controller.StuckSlipRatio",     params.stuck_slip_ratio);
	load_optional_cast<double>("Controller.StuckConfirmMs",     params.stuck_confirm_ms);
	load_optional_cast<double>("Controller.EscapeAdvSpeedMps",  params.escape_adv_speed_mps);
	load_optional_cast<double>("Controller.EscapeRotSpeedRps",  params.escape_rot_speed_rps);
	load_optional_cast<double>("Controller.EscapeDistanceM",    params.escape_distance_m);
	load_optional_cast<double>("Controller.EscapeMaxMs",        params.escape_max_ms);
	load_optional_cast<double>("Controller.EscapeSideProbeM",   params.escape_side_probe_m);
	load_optional_cast<double>("Controller.EscapeRearProbeM",   params.escape_rear_probe_m);
	load_optional_cast<double>("Controller.EscapeRearMinM",     params.escape_rear_min_m);
	load_optional_cast<double>("Controller.StuckVirtualObstacleRadiusM",  params.stuck_virtual_obstacle_radius_m);
	load_optional_cast<double>("Controller.StuckVirtualObstacleForwardM", params.stuck_virtual_obstacle_forward_m);
	int stuck_virtual_obstacle_ttl_ms = static_cast<int>(params.stuck_virtual_obstacle_ttl_ms);
	load_optional("Controller.StuckVirtualObstacleTTLms", stuck_virtual_obstacle_ttl_ms);
	params.stuck_virtual_obstacle_ttl_ms = static_cast<std::uint64_t>(std::max(0, stuck_virtual_obstacle_ttl_ms));
	load_optional("Controller.ProximityLogEnabled",             params.proximity_log_enabled);
	load_optional("Controller.ProximityCsvPath",                params.proximity_csv_path);
	load_optional_cast<double>("Controller.ProximityLogDistance", params.proximity_log_distance_m);

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
	// Runs on the MAIN thread (QueuedConnection — never DirectConnection; see the
	// connect note in initialize). LiDAR is NOT read from the graph (media plane only),
	// so there is no "laser" handling here.
	(void)id;
	if (type == "room" or type == "robot")
		hibernationTick();
}

void SpecificWorker::modify_edge_slot(std::uint64_t, std::uint64_t, const std::string &type)
{
	if (type == params.target_edge_type || type == "RT")
		hibernationTick();
}

void SpecificWorker::control_loop()
{
	using namespace std::chrono_literals;
	while (control_running_.load(std::memory_order_acquire))
	{
		{
			std::unique_lock<std::mutex> lock(control_mutex_);
			control_cv_.wait_for(lock, 20ms, [this]()
			{
				return !control_running_.load(std::memory_order_acquire)
				    || control_operating_.load(std::memory_order_acquire)
				    || command_pending_.load(std::memory_order_acquire);
			});
		}
		if (!control_running_.load(std::memory_order_acquire))
			break;

		// 1) Drain user commands (target clicks, follow toggle) on the owning thread.
		if (command_pending_.load(std::memory_order_acquire))
		{
			std::vector<std::function<void()>> commands;
			{
				std::lock_guard<std::mutex> lock(command_mutex_);
				commands.swap(command_queue_);
				command_pending_.store(false, std::memory_order_release);
			}
			for (auto &command : commands)
				if (command) command();
		}

		// 2) Lidar decode off the GUI thread. LiDAR comes ONLY from the zero-copy media
		//    plane (robot_concept no longer publishes the laser_* node to DSR). The shared
		//    reader brings up its subscribers lazily and dedups by timestamp.
		bool fresh_lidar = poll_lidar_media();
		if (fresh_lidar)
		{
			last_lidar_rx_ = std::chrono::steady_clock::now();
			lidar_ever_received_ = true;
		}

		// 3) Heavy pipeline (planning + MPPI + command emission), guarded by the LiDAR
		//    stream watchdog: if the stream stalls, hold the robot in a local emergency
		//    state and wait for recovery rather than planning on stale perception.
		if (control_operating_.exchange(false, std::memory_order_acq_rel))
		{
			if (lidar_ever_received_)
			{
				const auto age = std::chrono::steady_clock::now() - last_lidar_rx_;
				const bool stalled = age > std::chrono::milliseconds(params.lidar_stall_timeout_ms);
				if (stalled && !lidar_stalled_)
				{
					lidar_stalled_ = true;
					stop_robot();
					const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(age).count();
					qWarning() << "[EMERGENCY] LiDAR stream stalled — no data for" << ms
					           << "ms (>" << params.lidar_stall_timeout_ms
					           << "). Robot stopped; waiting for the stream to recover.";
					display_.set_command_text("EMERGENCY: LiDAR stream stalled — waiting for recovery");
				}
				else if (!stalled && lidar_stalled_)
				{
					lidar_stalled_ = false;
					qInfo() << "[EMERGENCY] LiDAR stream recovered — resuming control.";
					display_.set_command_text("LiDAR stream recovered — resuming");
				}
			}

			if (!lidar_stalled_)
				compute();
		}
	}
}

void SpecificWorker::init_lidar_media()
{
	// Build the shared multi-plane reader on the main thread (graph already loaded). Its
	// subscribers come up lazily inside poll() from the Operating control thread (throttled,
	// descriptor-driven) — the sanctioned pattern; nothing touches DDS here. Prefers the two
	// per-device planes (helios + bpearl, DEVICE frame) and falls back to the fused lidar3D
	// plane; inner_eigen_api_ backs the device->robot RT transform + merge.
	if (!params.lidar_use_media || lidar_reader_ || !G)
		return;
	lidar_reader_ = std::make_unique<rc::media::LidarPlaneReader>(
		G, inner_eigen_api_.get(),
		std::vector<std::string>{params.lidar_helios_name, params.lidar_bpearl_name},
		params.lidar_name, "lidar");
}

bool SpecificWorker::poll_lidar_media()
{
	if (!lidar_reader_)
		return false;

	// One shared call: newest helios + bpearl sweeps merged into the ROBOT frame (static mounts),
	// or the fused lidar3D sweep while bridging. interpolate=false — device->robot only crosses the
	// static mount edges; the dynamic room<-robot leg is applied downstream by the tracker at the
	// scan stamp. graph_state().robot_name is the frame the tracker treats as identity.
	const std::string &robot_name = world_model_.graph_state().robot_name;
	if (robot_name.empty())
		return false;

	const auto sweep = lidar_reader_->poll(robot_name, /*interpolate=*/false);
	if (!sweep.has_value() || sweep->points.empty())
		return false;

	std::vector<float> xs, ys, zs;
	xs.reserve(sweep->points.size()); ys.reserve(sweep->points.size()); zs.reserve(sweep->points.size());
	for (const auto &p : sweep->points) { xs.push_back(p.x()); ys.push_back(p.y()); zs.push_back(p.z()); }

	// Feed as a single robot-frame scan: the tracker sees robot<-robot = identity for the height
	// filter and applies the dynamic room<-robot pose at the merged stamp. Dedup is by that stamp.
	return obstacle_tracker_.handle_lidar_points(robot_name, std::move(xs), std::move(ys), std::move(zs),
	                                             static_cast<std::uint64_t>(sweep->stamp_ms));
}

void SpecificWorker::stop_control_thread()
{
	control_running_.store(false, std::memory_order_release);
	control_cv_.notify_all();
	if (control_thread_.joinable())
		control_thread_.join();
}

void SpecificWorker::enqueue_command(std::function<void()> command)
{
	{
		std::lock_guard<std::mutex> lock(command_mutex_);
		command_queue_.push_back(std::move(command));
		command_pending_.store(true, std::memory_order_release);
	}
	control_cv_.notify_one();
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

