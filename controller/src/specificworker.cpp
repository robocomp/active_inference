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
	                                 [this](float adv, float side, float rot)
	                                 { display_.set_command_values(adv, side, rot); });
	motion_commander_.set_profile_sink(
	    [this](std::uint64_t t_ms, float adv, float side, float rot, float freshness)
	    { session_.mission().add_profile_sample(t_ms, adv, side, rot, freshness); });
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
	ControllerDisplay::Callbacks gui;
	// A left click means "add a waypoint" while recording and "drive there" otherwise. Branching here, on
	// the control thread, keeps the viewer ignorant of missions — it just reports where you clicked.
	gui.on_manual_target = [this](const QPointF &point)
	{
		// GUI thread. While recording, a click is a waypoint and nothing else. Otherwise the click becomes a
		// TEMPORARY target that supersedes a running mission — and since that discards the rest of a
		// measured run, it asks first. The question has to be posed here, on the GUI thread, before the
		// intent is queued; asking from the control thread would mean a modal dialog off the GUI thread.
		if (not display_.mission_recording() and display_.mission_running()
		    and not display_.confirm_mission_supersede())
			return;
		enqueue_command([this, point]()
		{
			if (session_.mission().recording())
			{
				session_.mission().add_point({static_cast<float>(point.x()), static_cast<float>(point.y())});
				return;
			}
			// Supersede: end the run (its CSV rows are closed out as aborted) and drop its overlay, so the
			// canvas shows the target the robot is actually driving to and not the tour it abandoned.
			if (session_.mission().running())
				session_.mission().stop("superseded by click", current_time_ms());
			// The selector names what is driving, so clicking a point puts it on "Target". Mode first, then
			// the point: set_mode clears the click target on any transition AWAY from Target.
			session_.mission().set_mode(rc::DriveMode::Target);
			session_.mission().set_click_target(Eigen::Vector2f{static_cast<float>(point.x()),
			                                                    static_cast<float>(point.y())});
			driving_enabled_ = true;
			set_manual_target(point);
		});
	};
	gui.on_clear_target = [this]()
	{
		enqueue_command([this]()
		{
			// While recording, a right click takes back the last waypoint — the same gesture that clears a
			// target, applied to the thing you are actually building.
			if (session_.mission().recording()) { session_.mission().undo_point(); return; }
			session_.mission().set_click_target(std::nullopt);
			clear_manual_target();
		});
	};
	gui.mission.on_drive_mode = [this](int index)
	{
		enqueue_command([this, index]()
		{
			session_.mission().set_mode(rc::from_index(index));
		});
	};
	gui.mission.on_select = [this](std::string name)
	{ enqueue_command([this, name]() { session_.mission().select(name); }); };
	gui.on_waypoint_moved = [this](int index, float x, float y)
	{
		enqueue_command([this, index, x, y]()
		{
			// Dragging IS the editor — there is no Edit mode to enter and no Save to remember. So the edit
			// is persisted immediately; a route silently lost on exit would be worse than a stray write.
			if (session_.mission().move_waypoint(index, {x, y}))
				session_.mission().save(missions_path_);
		});
	};
	gui.mission.on_record_begin = [this]()
	{ enqueue_command([this]() { session_.mission().start_recording(); }); };
	gui.mission.on_record_finish = [this](std::string name)
	{
		enqueue_command([this, name]()
		{
			if (name.empty()) { session_.mission().cancel_recording(); return; }
			if (session_.mission().finish_recording(name))
			{
				session_.mission().save(missions_path_);
				refresh_mission_list();
			}
		});
	};
	gui.mission.on_delete = [this](std::string name)
	{
		enqueue_command([this, name]()
		{
			if (session_.mission().remove(name))
			{
				session_.mission().save(missions_path_);
				refresh_mission_list();
			}
		});
	};
	gui.mission.on_safety_bias = [this](float bias)
	{
		// Through the command queue like every other GUI intent: the slider is moved on the Qt thread and
		// the value is read by the control thread when it next builds or repairs a route.
		enqueue_command([this, bias]() { params.route_safety_bias = std::clamp(bias, 0.f, 1.f); });
	};
	gui.mission.on_smooth = [this]()
	{
		enqueue_command([this]()
		{
			// Smoothing rewrites the authored route, so it is persisted at once — the same reasoning
			// as a waypoint drag: there is no separate save step to remember.
			if (session_.smooth_selected_mission() > 0)
				session_.mission().save(missions_path_);
		});
	};
	gui.mission.on_run = [this](int laps)
	{
		enqueue_command([this, laps]()
		{
			// Stamp the run with the configuration it was produced under. Comparing two runs is only
			// meaningful if you can tell whether the ROBOT changed between them, and by the time the
			// numbers are read the config has usually moved on.
			session_.mission().set_run_context(rc::MissionRunContext{
			    .build = "",
			    .max_adv_mps = params.max_adv_speed_mps,
			    .max_rot_rps = params.max_rot_speed_rps,
			    .comfort_standoff_m = params.comfort_standoff_m,
			    .footprint_safety_margin_m = params.footprint_safety_margin_m,
			    .planner_cell_size_m = params.planner_cell_size_m,
			    .body_inscribed_m = path_controller_.footprint().inscribed_radius(),
			    .body_circumscribed_m = path_controller_.footprint().circumscribed_radius()});
			// Without a mission there is nothing to start — Run just lets whatever is driving, drive.
			if (not rc::uses_mission(session_.mission().mode()))
			{
				driving_enabled_ = true;
				return;
			}
			session_.mission().set_click_target(std::nullopt);   // a tour replaces a lingering click target
			clear_manual_target();
			const bool started = session_.mission().start(laps, current_time_ms());
			// A new run starts with a clean canvas: the driven trace belongs to the run that drew it, and
			// leaving the previous one on screen makes two different runs look like one confused path.
			if (started) display_.clear_robot_trajectory();
			if (started) driving_enabled_ = true;
			// Say what Run actually decided. Driving is only enabled when the mission starts, so a
			// refused start leaves the robot halted — and without this line that is silent from the
			// user's side, who sees a button that did nothing.
			std::println("[mission] RUN: mode={} mission='{}' laps={} -> {}  (driving {})",
			             rc::to_string(session_.mission().mode()),
			             session_.mission().selected_name(), laps,
			             started ? "STARTED" : "REFUSED", driving_enabled_ ? "on" : "OFF");
		});
	};
	gui.mission.on_stop = [this]()
	{
		enqueue_command([this]()
		{
			// The ONE halt. Ends the mission, drops any click target, and stops the base whatever was
			// driving it — including the affordance planner, which would otherwise re-target immediately.
			session_.mission().stop("user", current_time_ms());
			session_.mission().set_click_target(std::nullopt);
			clear_manual_target();
			driving_enabled_ = false;
		});
	};

	display_.initialize(obstacle_tracker_.lidar_buffer(), std::move(gui));
	session_.mission().set_csv_path(params.mission_csv_path);
	session_.mission().set_run_dir(params.mission_run_dir);
	// Keep the per-cycle MPPI diagnostics with the run they describe. Written live to a fixed path and
	// truncated by the next run, so without this a comparison destroys its own baseline.
	session_.mission().archive_on_stop("mppi_diag.csv");
	session_.mission().archive_on_stop("route_events.csv");
	session_.mission().archive_on_stop("route_geometry.csv");
	session_.mission().archive_on_stop("mppi_cycle.txt");
	session_.mission().archive_on_stop("mppi_reversal.txt");
	session_.mission().load(missions_path_);
	refresh_mission_list();
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
			std::println("[vel-out] {} ticks | period mean {:.1f} ms WORST {:.1f} ms | ice mean {:.1f} ms "
			             "max {:.1f} ms | cmd age max {:.0f} ms | freshness min {:.2f}",
			             stats.ticks, stats.period_mean_ms, stats.period_max_ms,
			             stats.ice_mean_ms, stats.ice_max_ms, stats.cmd_age_max_ms, stats.scale_min);
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
		display_.set_selected_affordance(cur, prev);
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

	// HALT IS CHECKED BEFORE PLANNING, not after. ensure_current_plan() steps an in-flight escape maneuver,
	// which commands the base directly — so a halt tested further down would leave Stop unable to stop a
	// robot that was reversing out of a wedge, i.e. exactly when stopping matters most. Nothing below this
	// point may run while halted; the display still updates so the view stays live.
	if (!driving_enabled_)
	{
		if (!stop_sent_when_halted_)
		{
			path_controller_.stop();
			stop_robot();
			stop_sent_when_halted_ = true;
		}
		update_custom_widget(step->robot_pose);
		return;
	}
	stop_sent_when_halted_ = false;

	if (!ensure_current_plan(*step))
		return;

	execute_plan(step->robot_pose);
	update_custom_widget(step->robot_pose);
}

/////////////////////////////////////////////////////////////////
// ─── State machine ─────────────────────────────────────────────────────────

float SpecificWorker::worst_compute_period_ms_ = 0.f;

void SpecificWorker::log_compute_perf(FPSCounter &counter)
{
	counter.cont++;
	const auto now = std::chrono::high_resolution_clock::now();
	// Track the worst gap between consecutive compute() completions inside this reporting window.
	{
		static std::chrono::high_resolution_clock::time_point last_call{};
		if (last_call.time_since_epoch().count() != 0)
			worst_compute_period_ms_ = std::max(
				worst_compute_period_ms_,
				static_cast<float>(std::chrono::duration<double, std::milli>(now - last_call).count()));
		last_call = now;
	}
	const auto elapsed_ms = std::chrono::duration<double, std::milli>(now - counter.begin).count();
	if (elapsed_ms < 1000.0)
		return;

	counter.last_period = static_cast<float>(elapsed_ms / std::max(1u, counter.cont));
	counter.period = 1000;
	const float fps = counter.get_frequency();
	const float cpu = std::max(0.f, counter.get_cpu_use());
	// std::println, NOT qInfo. genericworker installs a NO-OP Qt message handler whenever
	// Component.Debug.Verbose is false, which silently eats every qInfo/qDebug in the process — that is why
	// this line never appeared. It is not a miswired initialization: table_concept's generated handler block
	// is byte-identical and its config also has Verbose=false, so its qInfo is swallowed too; everything you
	// actually see from these agents (table_concept, residual_concept) is printed with std::println. Setting
	// Verbose=true would restore qInfo globally, at the cost of un-swallowing every Qt/library message too.
	//
	// MEAN period alone hid the problem: the loop's median is ~105 ms against a 100 ms target and looks
	// healthy, while the tail runs past a second. So report the WORST period in the window as well — that is
	// the number that corresponds to what you feel as a stall.
	std::println("[CTRL] fps={:.1f} cpu={:.0f}% period mean {:.1f} ms  WORST {:.1f} ms",
	             fps, cpu, counter.get_period(), worst_compute_period_ms_);
	worst_compute_period_ms_ = 0.f;
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
	                                  obstacle_tracker_,
	                                  path_controller_,
	                                  motion_commander_,
	                                  display_,
	                                  [this]() { return current_time_ms(); });
}

/////////////////////////////////////////////////////////////////
void SpecificWorker::load_params()
{
	load_optional_cast<double>("Planner.CellSize", params.planner_cell_size_m);
	load_optional("Mission.LibraryPath", missions_path_);
	load_optional("Mission.MetricsCsvPath", params.mission_csv_path);
	load_optional("Mission.RunDir", params.mission_run_dir);
	load_optional_cast<double>("Controller.ComfortStandoff", params.comfort_standoff_m);
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
	load_optional_cast<double>("Controller.MaxAdvSpeed", params.max_adv_speed_mps);
	load_optional_cast<double>("Controller.MaxLateralAccel", params.max_lateral_accel_mps2);
	load_optional_cast<double>("Controller.MaxRotSpeed", params.max_rot_speed_rps);
	load_optional_cast<double>("Controller.FootprintSafetyMarginM", params.footprint_safety_margin_m);
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
	// No robot_radius: the MPPI derives every body extent from the footprint itself. d_safe is the ONE
	// standoff knob, and it is comfort only — the hard constraint is the footprint test.
	path_controller_.params.d_safe = params.comfort_standoff_m;
	load_optional("Controller.PathHorizonWaypoints", params.path_horizon_waypoints);
	load_optional("Controller.RouteContinuous", params.route_continuous);
	load_optional("Controller.SmoothPlannedPath", params.smooth_planned_path);
	load_optional_cast<double>("Controller.RouteSpacing", params.route_spacing_m);
	load_optional_cast<double>("Controller.RouteSmoothing", params.route_smoothing_m);
	load_optional("Controller.RouteOptimize", params.route_optimize);
	load_optional_cast<double>("Controller.RouteSafetyBias", params.route_safety_bias);

	load_optional_cast<double>("Controller.LambdaContinuity", params.lambda_continuity);
	load_optional_cast<double>("Controller.ContinuityRotFactor", params.continuity_rot_factor);
	path_controller_.params.lambda_continuity = std::max(0.f, params.lambda_continuity);
	path_controller_.params.continuity_rot_factor = std::max(0.f, params.continuity_rot_factor);
	path_controller_.params.min_adv_cmd = 0.f;
	path_controller_.params.goal_clearance_relax_dist = std::max(0.05f, params.goal_clearance_relax_dist_m);
	path_controller_.params.goal_obstacle_margin = std::max(0.f, params.goal_obstacle_margin_m);
	path_controller_.params.goal_clearance_min_ratio = std::clamp(params.goal_clearance_min_ratio, 0.5f, 1.f);
	path_controller_.params.straight_speed_heading_threshold = std::max(0.f, params.straight_speed_heading_threshold_rad);
	path_controller_.params.straight_speed_clearance_margin = std::max(0.f, params.straight_speed_clearance_margin_m);
	path_controller_.params.straight_speed_min_goal_dist = std::max(0.f, params.straight_speed_min_goal_dist_m);
	path_controller_.set_control_mode(rc::TrajectoryController::ControlMode::MPPI);
	// Say which way the robot will be driven, at startup, unconditionally. Two attempts at diagnosing
	// "it still does the old thing" were spent reasoning about whether a flag had arrived; one printed
	// line settles it.
	std::println("[route] mode = {}   (RouteContinuous={}, spacing={:.2f} m, smoothing={:.2f} m, "
	             "PathHorizonWaypoints={}, LambdaContinuity={:.1f}, SmoothPlannedPath={})",
	             params.route_continuous ? "CONTINUOUS (one C2 curve, arc-length)" : "WAYPOINTS (per-leg)",
	             params.route_continuous, params.route_spacing_m, params.route_smoothing_m,
	             params.path_horizon_waypoints, params.lambda_continuity, params.smooth_planned_path);
}

void SpecificWorker::refresh_mission_list()
{
	display_.set_mission_list(session_.mission().names(), session_.mission().selected_name());
}

void SpecificWorker::push_mission_view()
{
	const auto &mission = session_.mission();
	display_.set_mission_state(rc::MissionPanel::View{
	                               .status = mission.status_text(),
	                               .controls_enabled = rc::uses_mission(mission.mode()),
	                               .running = mission.running(),
	                               .recording = mission.recording(),
	                               .driving = driving_enabled_,
	                               .mode_index = rc::to_index(mission.mode()),
	                               .recorded_points = static_cast<int>(mission.recorded().size()),
	                               .laps_remaining = mission.laps_remaining()},
	                           mission.display_waypoints(),
	                           -1);   // no waypoint index: the route is one curve
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

		// Reflect mission/drive state EVERY iteration (~20 ms), independent of compute(). Selecting a mode
		// or pressing Stop changes what the panel must show, and both leave the pipeline with nothing to
		// plan — so pushing from compute()'s tail meant the UI only caught up on the next cycle that
		// happened to have a target. That is why the mission list stayed shaded until Run, and why Stop
		// left the button reading "Stop".
		// A tour that finished on its own is not a reason to keep driving: drop back to halted so the
		// button reads "Run" again and the robot is not left armed with nothing to do.
		if (session_.mission().consume_completed())
			driving_enabled_ = false;
		push_mission_view();

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
					display_.set_command_text("");   // clear the alert badge
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

