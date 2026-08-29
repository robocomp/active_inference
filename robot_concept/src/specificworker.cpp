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
#include "../../common/robot_footprint/robot_footprint.h"

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/media_transport/media_transport.h"

#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include "media_stream_viewers.h"
#include "graph_attr_viewers.h"
#include "graph_safe.h"   // rc::safe_update_node — guard update_node against exceptions
#include <QTimer>
#include <QLabel>
#include <QVBoxLayout>

#include <ConfigLoader/ConfigLoader.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iterator>
#include <iostream>
#include <print>

namespace
{
// Read + normalize + validate a media source selector ("auto" | "ice" | "dds").
// Unknown values warn once and fall back to "auto".
std::string read_media_source(const ConfigLoader& cfg, const char* key)
{
	std::string s = "auto";
	rc::ConfigLoaderUtils::load_optional(cfg, key, s);
	std::ranges::transform(s, s.begin(), ::tolower);
	if (s != "auto" and s != "ice" and s != "dds")
	{
		qWarning() << "[Media] unknown" << key << "="
		           << QString::fromStdString(s) << "— falling back to 'auto'";
		s = "auto";
	}
	return s;
}
}  // namespace

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

		states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
			std::bind(&SpecificWorker::waiting_loop, this),
			std::bind(&SpecificWorker::waiting_enter, this));

		states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
			std::bind(&SpecificWorker::operating_loop, this),
			std::bind(&SpecificWorker::operating_enter, this));

		states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
			std::bind(&SpecificWorker::degraded_loop, this),
			std::bind(&SpecificWorker::degraded_enter, this));

		states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
		states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
		states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
		states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

		statemachine.addState(states["Waiting"].get());
		statemachine.addState(states["Operating"].get());
		statemachine.addState(states["Degraded"].get());

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0)
		{
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	qInfo() << "Destroying SpecificWorker";
	request_shutdown();
}

void SpecificWorker::request_shutdown()
{
	if (shutting_down_.exchange(true))
		return;

	stop_imu_thread = true;
	stop_lidar_thread = true;
	stop_rgbd_thread = true;
	stop_ricoh_thread = true;

	if (imu_thread.joinable())
		imu_thread.join();
	if (lidar_thread.joinable())
		lidar_thread.join();
	if (rgbd_thread.joinable())
		rgbd_thread.join();
	if (ricoh_thread.joinable())
		ricoh_thread.join();

	// Tear down DDS endpoints after producer threads are fully stopped.
	media_.shutdown();

	cleanup_owned_nodes();
}


void SpecificWorker::initialize()
{
	qInfo() << "initialize robot_concept worker";
	GenericWorker::initialize();

	// ── Robot identity (name + canonical node id) — config-driven, never hard-coded ──────────────
	// Every robot-node lookup in this agent (mount dump, identity check, pose writes from
	// FullPoseEstimationPub) keys off robot_name. It comes from Agent.robot_name in the live config
	// (etc/config_beta.toml), so pointing the agent at a different Agent.configFile / different robot
	// needs no code change. If the key is absent we fall back to the single node of type "robot"
	// already in the graph rather than guessing a name.
	rc::ConfigLoaderUtils::load_optional(configLoader, "Agent.robot_name", robot_name);
	// The MESH's frame, not the robot's shape — see the declaration. P3Bot needs 90.
	// ★Loaded as double: ConfigLoader's variant carries no float alternative, and asking it for one is a
	// static_assert, not a conversion.
	{
		double yaw_deg = mesh_yaw_deg;
		rc::ConfigLoaderUtils::load_optional(configLoader, "Agent.mesh_yaw_deg", yaw_deg);
		mesh_yaw_deg = static_cast<float>(yaw_deg);
	}
	rc::ConfigLoaderUtils::load_optional<std::uint64_t, int>(configLoader, "Agent.robot_node_id", canonical_robot_id_);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Agent.graph_layout", graph_layout_);
	if (robot_name.empty())
	{
		if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
		{
			robot_name = robots.front().name();
			qWarning() << "[Agent] robot_name absent from config — using the graph's robot node"
			           << QString::fromStdString(robot_name);
		}
		else
			qWarning() << "[Agent] robot_name absent from config and no node of type 'robot' in the graph"
			              " — robot pose updates and the identity check have no node to address.";
	}
	else
		qInfo() << "[Agent] robot node =" << QString::fromStdString(robot_name)
		        << "(canonical id" << canonical_robot_id_ << ")";

	// ── WHERE the robot is, published for everyone ───────────────────────────────────────────────
	// A robot is not a place: the same P3Bot runs in the WAF room and in the apartment, and the same
	// apartment hosts P3Bot and Shadow. This agent owns the robot's identity, and it is also the one
	// component that exists in simulation AND on the real robot — the bridge knows the world too,
	// but only in simulation — so the scenario belongs here beside robot_name.
	// room_concept consumes it to pick its layout, and REFUSES TO START without it: it authors the
	// room polygon every other agent reads, so a wrong floor plan puts the whole fleet in the wrong
	// building while every number downstream stays self-consistent.
	rc::ConfigLoaderUtils::load_optional(configLoader, "Agent.scenario", scenario_name_);
	if (scenario_name_.empty())
		qWarning() << "[Agent] Agent.scenario is not set, so `scenario_name` will not be published."
		              " room_concept will refuse to start if its config defines scenario overlays.";

	rc::ConfigLoaderUtils::load_optional(configLoader, "Camera.dsr_rgb_fps", params.DSR_RGB_FPS);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Camera.dsr_depth_fps", params.DSR_DEPTH_FPS);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Lidar.dsr_lidar_fps", params.DSR_LIDAR_FPS);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Lidar.decimation_factor", params.LIDAR_DECIMATION_FACTOR);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Transforms.interpolate_rt", params.TRANSFORMS_INTERPOLATE_RT);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Debug.verbose", verbose_debug_);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Component.Debug.Verbose", verbose_debug_);

	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.domain_id", params.MEDIA_DOMAIN_ID);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.rgb_topic", params.MEDIA_RGB_TOPIC);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.depth_topic", params.MEDIA_DEPTH_TOPIC);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.ricoh_topic", params.MEDIA_RICOH_TOPIC);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.lidar_topic", params.MEDIA_LIDAR_TOPIC);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.imu_topic", params.MEDIA_IMU_TOPIC);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.enable_zed",   params.ENABLE_ZED);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.enable_ricoh", params.ENABLE_RICOH);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.enable_lidar", params.ENABLE_LIDAR);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.enable_imu",   params.ENABLE_IMU);
	rc::ConfigLoaderUtils::load_optional(configLoader, "Media.data_sharing", params.MEDIA_DATA_SHARING);
	// Per-sensor source selector ("auto" | "ice" | "dds"), normalized+validated by read_media_source().
	// Forced modes seed the runtime gate up front: "dds" starts already bypassing (monitor-only),
	// "ice"/"auto" start bridging. negotiate() then honours or re-checks it each tick.
	params.ZED_SOURCE   = read_media_source(configLoader, "Media.zed_source");
	params.RICOH_SOURCE = read_media_source(configLoader, "Media.ricoh_source");
	params.LIDAR_SOURCE = read_media_source(configLoader, "Media.lidar_source");
	params.IMU_SOURCE   = read_media_source(configLoader, "Media.imu_source");
	bridge_zed_.store(  params.ZED_SOURCE   != "dds", std::memory_order_relaxed);
	bridge_ricoh_.store(params.RICOH_SOURCE != "dds", std::memory_order_relaxed);
	bridge_lidar_.store(params.LIDAR_SOURCE != "dds", std::memory_order_relaxed);
	bridge_imu_.store(  params.IMU_SOURCE   != "dds", std::memory_order_relaxed);
	qInfo() << "[Media] sources: zed =" << QString::fromStdString(params.ZED_SOURCE)
	        << "| ricoh =" << QString::fromStdString(params.RICOH_SOURCE)
	        << "| lidar =" << QString::fromStdString(params.LIDAR_SOURCE)
	        << "| imu =" << QString::fromStdString(params.IMU_SOURCE)
	        << "(auto=negotiate, ice=always bridge, dds=always monitor external)";

	qInfo() << "[DSR Upload Rates] rgb=" << params.DSR_RGB_FPS
	        << "depth=" << params.DSR_DEPTH_FPS
	        << "lidar=" << params.DSR_LIDAR_FPS
	        << "(Hz; 0=every frame, <0=disabled)";

	// Seed body node dimensions (read from the graph as loaded from Agent.configFile; write defaults if absent).
	if (auto body_node = G->get_node("body"); body_node.has_value())
	{
		// Re-write pos_x / pos_y with their current values to force a full node update.
		const float px = G->get_attrib_by_name<pos_x_att>(body_node.value()).value_or(0.f);
		const float py = G->get_attrib_by_name<pos_y_att>(body_node.value()).value_or(0.f);
		G->add_or_modify_attrib_local<pos_x_att>(body_node.value(), px);
		G->add_or_modify_attrib_local<pos_y_att>(body_node.value(), py);

		bool seeded = false;
		if (!G->get_attrib_by_name<width_m_att>(body_node.value()).has_value())
		{ G->add_or_modify_attrib_local<width_m_att>(body_node.value(), 0.47f); seeded = true; }
		if (!G->get_attrib_by_name<depth_m_att>(body_node.value()).has_value())
		{ G->add_or_modify_attrib_local<depth_m_att>(body_node.value(), 0.47f); seeded = true; }
		if (!G->get_attrib_by_name<height_m_att>(body_node.value()).has_value())
		{ G->add_or_modify_attrib_local<height_m_att>(body_node.value(), 1.6f); seeded = true; }

		// ── ★MEASURE THE ROBOT INSTEAD OF DECLARING IT ───────────────────────────────────────────
		// The 0.47 / 0.47 above are a PLACEHOLDER, and provably so: they are byte-identical in
		// shadow.json and p3bot.json, i.e. the same numbers for two robots of different shapes. Whoever
		// reads them plans against a robot that does not exist — room_concept's epistemic planner takes
		// 0.5*hypot(w, d) = 0.332 m as its collision radius, against P3Bot's true 0.371 m.
		// This agent is the ONE that knows which robot this is (Agent.robot_name / configFile), and the
		// mesh named by the robot node's own `path` is the only per-robot statement of its shape. So
		// measure it here, once, and let every existing consumer of width_m/depth_m read the truth
		// without changing a line — one producer, rather than a mesh loader in every agent.
		// ★HEIGHT IS DELIBERATELY NOT TOUCHED. Despite its name, `height_m` is consumed as room_concept's
		// OBSTACLE CLOUD CEILING (RoomConfig::ROBOT_HEIGHT, "m, obstacle cloud ceiling"), not as the
		// robot's stature. Setting it to the mesh's true 1.31 m would silently change which LiDAR returns
		// that agent treats as obstacles — a perception change wearing a geometry change's clothes.
		const auto robot_node = G->get_node(robot_name);
		if (const auto path_attr = robot_node.has_value()
		        ? G->get_attrib_by_name<path_att>(robot_node.value())
		        : std::optional<std::reference_wrapper<const std::string>>{};
		    path_attr.has_value())
		{
			rc::RobotFootprint::MeshReport rep;
			const float yaw = mesh_yaw_deg * static_cast<float>(M_PI) / 180.f;
			if (const auto body = rc::RobotFootprint::from_obj(path_attr.value().get(), yaw, rep);
			    body.has_value())
			{
				const float w = 2.f * rep.x_max, d = 2.f * rep.y_max;
				const float w_old = G->get_attrib_by_name<width_m_att>(body_node.value()).value_or(0.f);
				const float d_old = G->get_attrib_by_name<depth_m_att>(body_node.value()).value_or(0.f);
				G->add_or_modify_attrib_local<width_m_att>(body_node.value(), w);
				G->add_or_modify_attrib_local<depth_m_att>(body_node.value(), d);
				qInfo().nospace() << "[body] dimensions MEASURED from '"
				                  << QString::fromStdString(path_attr.value().get()) << "' (yaw "
				                  << mesh_yaw_deg << " deg): width " << w_old << " -> " << w
				                  << " m, depth " << d_old << " -> " << d
				                  << " m. height_m left at its declared value (it is an obstacle-cloud "
				                     "ceiling, not the robot's stature).";
			}
			else
				qWarning().nospace() << "[body] could NOT measure the robot from its mesh ("
				                     << QString::fromStdString(rep.reason)
				                     << ") — the declared dimensions stand, and they may be a placeholder.";
		}

		rc::safe_update_node(*G, body_node.value());
		if (seeded)
			qInfo() << __FUNCTION__ << "Seeded body node with default dimensions (width=0.47, depth=0.47, height=1.6)";
		else
			qInfo() << __FUNCTION__ << "Body node dimensions already present in graph";
	}
	else
		qWarning() << __FUNCTION__ << "Body node not found in DSR graph";

	// ── One-shot dump of the static mount edges (both directions) ─────────────
	// Prints <robot>->room and body->kinova_arm_r once at startup so they can be
	// eyeballed against the Webots scene. get_RT_pose_from_parent(child) is the
	// parent->child transform; the inverse (child->parent) is what the DSR viewer
	// tends to display, so we print both. Planar mounts → translation + yaw is enough.
	if (auto rt = G->get_rt_api())
	{
		auto yaw_deg = [](const Eigen::Affine3d &M)
		{ const auto R = M.linear(); return std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI; };
		auto dump = [&](const char *child, const char *parent)
		{
			const auto cn = G->get_node(child);
			if (!cn.has_value()) { qInfo() << "[mount]" << parent << "->" << child << ": node missing"; return; }
			const auto pose = rt->get_RT_pose_from_parent(cn.value());
			if (!pose.has_value()) { qInfo() << "[mount]" << parent << "->" << child << ": no RT edge"; return; }
			const Eigen::Affine3d T = pose.value();
			const Eigen::Affine3d Ti = T.inverse();
			const Eigen::Vector3d t = T.translation();
			const Eigen::Vector3d ti = Ti.translation();
			qInfo().noquote() << QString::asprintf(
				"[mount] %s->%s t=(%.4f, %.4f, %.4f) yaw=%.2f deg | inv %s->%s t=(%.4f, %.4f, %.4f) yaw=%.2f deg",
				parent, child, t.x(), t.y(), t.z(), yaw_deg(T),
				child, parent, ti.x(), ti.y(), ti.z(), yaw_deg(Ti));
		};
		dump("room", robot_name.c_str());
		dump("kinova_arm_r", "body");
	}

	// Media plane: zero-copy DDS publishers carrying the raw sensor streams OUT of
	// the DSR/CRDT graph — RGB + depth images, the LiDAR scan and the IMU sample,
	// each stamped per-frame so consumers can realign them upstream. The DSR
	// cam_rgb/cam_depth/laser_* blob writes below remain governed by the DSR_*_FPS
	// knobs (<0 disables them once consumers migrate to the plane). Registering
	// another image-like stream is one extra StreamSpec — see SensorMediaPublisher.
	{
		SensorMediaPublisher::Config mcfg;
		mcfg.domain_id     = static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID);
		mcfg.history_depth = 8;
		mcfg.data_sharing  = params.MEDIA_DATA_SHARING;   // zero-copy SHM loans (static topology only)
		// Each stream is registered only when its [Media].enable_* gate is on.
		if (params.ENABLE_ZED)
			mcfg.image_streams = {{"rgb",   params.MEDIA_RGB_TOPIC,   rc::media::STREAM_ZED_RGB},
			                      {"depth", params.MEDIA_DEPTH_TOPIC, rc::media::STREAM_ZED_DEPTH}};
		// The RGBD_360 panorama rides its OWN wide Image360Frame plane (5.5 MB buffer),
		// kept separate from the ZED ImageFrame so its small buffer is not enlarged.
		if (params.ENABLE_RICOH)
			mcfg.image360_stream = {{"rgb360", params.MEDIA_RICOH_TOPIC, rc::media::STREAM_RICOH_RGB}};
		if (params.ENABLE_LIDAR)
			mcfg.lidar_stream  = {{"lidar", params.MEDIA_LIDAR_TOPIC, rc::media::STREAM_LIDAR}};
		if (params.ENABLE_IMU)
			mcfg.imu_stream    = {{"imu",   params.MEDIA_IMU_TOPIC,   rc::media::STREAM_IMU}};
		media_.init(mcfg);
		qInfo() << "[Media] stream gates — zed:" << params.ENABLE_ZED << "ricoh:" << params.ENABLE_RICOH
		        << "lidar:" << params.ENABLE_LIDAR << "imu:" << params.ENABLE_IMU
		        << "| data_sharing:" << params.MEDIA_DATA_SHARING;
	}

	// Advertise the plane PER SENSOR NODE so ANY agent can discover and subscribe
	// without hardcoding topic/domain/QoS — consumers read it generically via
	// rc::media::descriptor_from_graph(G, "<node>"). Each node carries only its own
	// stream(s): rgb+depth on "zed", lidar on "helios", imu on "imu". Only these
	// small JSON strings live in the graph; the heavy frames stay out-of-band on the
	// zero-copy plane. (Registering a new stream = one more (node, keys) line here.)
	struct { const char* node; std::vector<std::string> keys; bool enabled; } media_ads[] = {
		{"zed",     {"rgb", "depth"}, params.ENABLE_ZED},
		{"ricoh",   {"rgb360"},       params.ENABLE_RICOH},
		{"helios",  {"lidar"},        params.ENABLE_LIDAR},
		{"imu",     {"imu"},          params.ENABLE_IMU},
	};
	for (const auto& ad : media_ads)
	{
		if (not ad.enabled)
			continue;                                    // gated OFF: no publisher, no descriptor
		// Cold start: no driver has spoken yet, so there is no model to carry and the sensor
		// fields stay ABSENT — which the schema defines as "unknown, keep your own constant",
		// not as zero. They fill in as soon as a driver advertises one.
		if (G->get_node(ad.node).has_value())
		{
			write_media_descriptor(ad.node, media_.descriptor_json(ad.keys));
			qInfo() << "[Media] descriptor advertised on" << ad.node << ":"
			        << QString::fromStdString(media_.descriptor_json(ad.keys));
		}
		else
			qWarning() << "[Media]" << ad.node << "node not found at init — media descriptor NOT advertised";
	}

	// Build the media-negotiation table now that the generated proxies are wired.
	build_media_groups();

	// One bounded negotiation round before starting the bridge threads: if a producer is
	// already publishing its plane on DDS, adopt its descriptor now and start with bridging
	// OFF (compute() then re-checks — non-blocking — at low rate to follow producers up/down).
	prime_media_groups();

	// Start only the reader threads whose stream is enabled — a gated-off sensor
	// spins up no Ice proxy traffic at all.
	if (params.ENABLE_LIDAR) lidar_thread = std::thread(&SpecificWorker::read_lidar_thread, this);
	if (params.ENABLE_ZED)   rgbd_thread  = std::thread(&SpecificWorker::read_rgbd_thread,  this);
	if (params.ENABLE_IMU)   imu_thread   = std::thread(&SpecificWorker::read_imu_thread,   this);
	if (params.ENABLE_RICOH) ricoh_thread = std::thread(&SpecificWorker::read_ricoh_thread, this);
	qInfo() << __FUNCTION__ << "Started reader threads (enabled sensors only)";

	presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
	// Colour this agent's node in the graph view by its live health: the coordinator already
	// publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
	// Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
	// cannot break it.
	presence_coordinator_.attach_state_machine(&statemachine);
	presence_coordinator_.set_transition_hooks({
		.request_presence_ready = [this]() { emit presenceReady(); },
		.request_presence_lost  = [this]() { emit presenceLost(); },
	});
	presence_coordinator_.set_peer_hooks({
		.on_peer_restarted = [](std::uint32_t id)
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
			qInfo() << "[SM] -> Waiting";
			const auto missing = presence_coordinator_.missing_required_names();
			if (!missing.empty())
			{
				QString m;
				for (const auto &label : missing)
					m += " " + QString::fromStdString(label);
				qInfo() << "  missing:" << m;
			}
		},
		.on_operating_enter = [this]()
		{
			qInfo() << "[SM] -> Operating: all required constraints satisfied";
			// Post-sync check (once): a stale robot node from an unclean previous exit may have
			// synced in AFTER initialize() ran; flag an unexpected id so a lingering illegal
			// robot node is visible rather than silently colliding on the next update_node.
			check_robot_identity();
		},
		.on_operating_loop = [this]()
		{
			compute();
			if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
				it->second->set_external_fps(states.at("Operating")->getActualFps());
		},
		.on_degraded_enter = []()
		{
			qInfo() << "[SM] -> Degraded: a required peer or node is no longer available";
		},
	});
	presence_coordinator_.start();

	QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
	                 this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

	// Debounced relayout timer: a burst of node arrivals (e.g. residual_concept spawning several obstacles,
	// or a peer joining) collapses into ONE twopi instead of one-per-node. Single-shot, restarted on each
	// structural change; fires on the main thread.
	relayout_timer_ = new QTimer(this);
	relayout_timer_->setSingleShot(true);
	QObject::connect(relayout_timer_, &QTimer::timeout, this, [this]()
	{
		if (not shutting_down_.load())
			trigger_graph_layout();
	});

	// Seed the known-node set with the graph as loaded, then relayout the DSR viewer whenever a NODE is
	// added or removed — so late arrivals (the residual_concept agent node id=14, its residual_* obstacles,
	// any joining peer) re-tidy the tree, not just the startup snapshot. QUEUED, never DirectConnection:
	// these signals fire on the FastDDS reader threads and the slot must run on the main thread (CLAUDE.md).
	for (const auto& n : G->get_nodes())
		known_node_ids_.insert(n.id());
	QObject::connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot,
	                 Qt::QueuedConnection);
	QObject::connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot,
	                 Qt::QueuedConnection);

	// One-shot radial (twopi) relayout once the bootstrap graph has been ingested by the viewer, so the DSR
	// tree is well organized in the UI at startup. Delayed on the main-thread event loop so it runs after the
	// initial node/edge update signals have been processed (graph access stays on the main thread).
	QTimer::singleShot(1500, this, [this]()
	{
		if (shutting_down_.load())
			return;
		trigger_graph_layout();
		wire_view_data_signal();   // enable right-click "View data" → live FPS viewer on media-plane nodes
		wire_agent_status_overlay();   // colour every agent node by its live health (green/orange/red/grey)
		check_robot_identity();   // flag a stale/mismatched robot node ingested during bootstrap
	});
}

// A node was inserted OR an attribute was updated (update_node_signal covers both). Relayout only when the id
// is genuinely NEW — attribute churn (residual writes its box + Σ every cycle) must NOT trigger a relayout.
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string& /*type*/)
{
	if (shutting_down_.load())
		return;
	if (known_node_ids_.insert(id).second)   // .second == true ⇒ this id was not present ⇒ a NEW node
		schedule_graph_relayout();
}

// A node was removed → drop it from the set and re-tidy.
void SpecificWorker::del_node_slot(std::uint64_t id)
{
	if (shutting_down_.load())
		return;
	if (known_node_ids_.erase(id) > 0)
		schedule_graph_relayout();
}

// Coalesce a burst of structural changes into a single relayout: (re)start the single-shot debounce timer.
void SpecificWorker::schedule_graph_relayout()
{
	if (relayout_timer_)
		relayout_timer_->start(400);
}

// One-shot diagnostic: verify the robot node (Agent.robot_name) carries the canonical id declared in
// Agent.configFile. A stale robot node left by an unclean previous exit (which no peer reaps while
// etc/robot_concept.toml is absent) surfaces here as an unexpected id — the same mismatch that makes
// update_node(robot_name) throw. We only WARN: DSR keeps names unique, so this is the sole robot node and
// deleting it would be worse than the leak.
void SpecificWorker::check_robot_identity()
{
	if (robot_identity_checked_ or shutting_down_.load() or robot_name.empty())
		return;
	robot_identity_checked_ = true;
	if (auto s = G->get_node(robot_name); s.has_value() and s->id() != canonical_robot_id_)
		qWarning() << "[graph] robot node" << QString::fromStdString(robot_name) << "has unexpected id"
		           << s->id() << "(expected" << canonical_robot_id_
		           << ") — likely a stale node from an unclean previous exit; stop agents with SIGTERM, "
		              "not kill -9, and ensure etc/robot_concept.toml exists so peers reap it.";

	// Publish WHERE this robot is, once, alongside the identity check that already runs here — same
	// node, same one-shot timing, and it needs the graph to be up for exactly the same reason.
	// ⚠ update_node is a WHOLE-NODE write: an attribute absent from the copy is ERASED (see
	//   dsr-node-attr-erasure-hazard). Re-fetch, add, write back — never write a stale copy.
	if (not scenario_name_.empty())
	{
		if (auto n = G->get_node(robot_name); n.has_value())
		{
			G->add_or_modify_attrib_local<scenario_name_att>(n.value(), scenario_name_);
			if (rc::safe_update_node(*G, n.value()))
				qInfo() << "[graph] scenario_name =" << QString::fromStdString(scenario_name_)
				        << "published on" << QString::fromStdString(robot_name);
			else
				qWarning() << "[graph] could NOT publish scenario_name — room_concept will refuse to"
				              " start if its config defines scenario overlays.";
		}
	}
}

void SpecificWorker::compute()
{
	// robot_concept's compute() is intentionally minimal: sensor reading runs in
	// background threads. We only turn their atomic frame counters into a 3 Hz
	// rate readout here (unconditional qInfo, so no Verbose/DSR/DDS log flood).
	static FPSCounter compute_fps;
	compute_fps.print("Compute (void)", 5000);

	static auto   last      = std::chrono::steady_clock::now();
	static std::uint64_t last_rgbd = 0, last_imu = 0, last_ricoh = 0;
	static std::uint64_t last_helios = 0, last_bpearl = 0, last_fullpose = 0;
	const auto    now = std::chrono::steady_clock::now();
	const double  dt  = std::chrono::duration<double>(now - last).count();
	// The raw estimate is (integer frame delta / dt): over a short window this quantises
	// hard — at a ~0.4s window one missed/extra frame is ±2.5Hz, and the ext-DDS streams
	// count in wait_and_poll() batches whose boundaries alias against the window, so the
	// readout jitters even though the source rate is stable. Fix = a longer window (finer
	// quantum: at 2s, ±1 frame is only ±0.5Hz) + a light EWMA to absorb the residual
	// batch aliasing. This measures reception precisely; it is NOT DDS propagation noise.
	constexpr double kRateWindowS = 2.0;
	if (dt >= kRateWindowS)   // long window + EWMA below; only PRINT when the smoothed rate moves
	{
		const std::uint64_t r = rgbd_frames_.load(std::memory_order_relaxed);
		const std::uint64_t i = imu_frames_.load(std::memory_order_relaxed);
		const std::uint64_t c = ricoh_frames_.load(std::memory_order_relaxed);
		const std::uint64_t hl = helios_frames_.load(std::memory_order_relaxed);
		const std::uint64_t bp = bpearl_frames_.load(std::memory_order_relaxed);
		const std::uint64_t fp = fullpose_frames_.load(std::memory_order_relaxed);
		// Instantaneous window rate → EWMA. Seed on first sample; alpha 0.5 settles in a few
		// windows after a real rate change while erasing per-window quantisation wobble.
		constexpr double kAlpha = 0.5;
		static double ema_rgbd = -1.0, ema_helios = -1.0, ema_bpearl = -1.0, ema_imu = -1.0, ema_ricoh = -1.0,
		              ema_fullpose = -1.0;
		const auto ewma = [](double& s, double inst)
		{ s = (s < 0.0) ? inst : kAlpha * inst + (1.0 - kAlpha) * s; return s; };
		const double f_rgbd   = ewma(ema_rgbd,   static_cast<double>(r  - last_rgbd)   / dt);
		const double f_imu    = ewma(ema_imu,    static_cast<double>(i  - last_imu)    / dt);
		const double f_ricoh  = ewma(ema_ricoh,  static_cast<double>(c  - last_ricoh)  / dt);
		const double f_helios = ewma(ema_helios, static_cast<double>(hl - last_helios) / dt);
		const double f_bpearl = ewma(ema_bpearl, static_cast<double>(bp - last_bpearl) / dt);
		const double f_fullpose = ewma(ema_fullpose, static_cast<double>(fp - last_fullpose) / dt);
		// Per-thread heartbeat, one table row each. src label: "off" when the whole path is
		// gated out (no reader thread), "local" while robot_concept bridges Ice→media itself,
		// "ext-DDS" once negotiation hands the plane to the external producer (bridge off) — which
		// now includes the IMU, whose producer is imu_dds. Adding a sensor is one row here. Print
		// only when a rate moves >= kHzPrintDelta or a src label flips, so steady-state jitter
		// doesn't reprint every sample.
		const auto src_label = [](bool enabled, const std::atomic<bool>* bridge) -> const char*
		{
			if (not enabled)         return "off";
			if (bridge == nullptr)   return "local";   // no external producer (IMU)
			return bridge->load(std::memory_order_relaxed) ? "local" : "ext-DDS";
		};
		struct FpsField { const char* label; double hz; const char* src; };
		const FpsField fields[] = {
			{ "ZEDThread",      f_rgbd,   src_label(params.ENABLE_ZED,   &bridge_zed_)   },
			{ "HeliosThread",   f_helios, src_label(params.ENABLE_LIDAR, &bridge_lidar_) },
			{ "BpearlThread",   f_bpearl, src_label(params.ENABLE_LIDAR, &bridge_lidar_) },
			{ "IMUThread",      f_imu,    src_label(params.ENABLE_IMU,   &bridge_imu_)   },
			{ "Ricoh360Thread", f_ricoh,  src_label(params.ENABLE_RICOH, &bridge_ricoh_) },
			// Odometry (FullPoseEstimationPub). Not a thread of ours and not on the media plane: it is
			// pushed by the producer's Ice callback, so the src is always "ice" — 0.0 Hz here means the
			// publisher (webots-bridge / base) is silent, not that a reader of ours stalled.
			{ "FullPoseSub",    f_fullpose, "ice" },
		};
		constexpr int NFIELDS = static_cast<int>(std::size(fields));
		// Print every rate window (~2 s) so the per-stream Hz table is always visible in the terminal,
		// not only when a rate happens to move.
		{
			QString line;
			for (int k = 0; k < NFIELDS; ++k)
				line += QString::asprintf("%s[%s:%s] %5.1f Hz",
				                          k ? " | " : "", fields[k].label, fields[k].src, fields[k].hz);
			qInfo().noquote() << line;
		}
		last = now; last_rgbd = r; last_imu = i; last_ricoh = c;
		last_helios = hl; last_bpearl = bp; last_fullpose = fp;

			// Publish live media-plane throughput onto each sensor's descriptor node (main thread, so
			// the graph write is safe). The frames travel over zero-copy DDS SHM and never hit the wire,
			// so this self-reported media_bps is the only way the mind network view can show real
			// per-stream media bandwidth.
			//
			// media_bps = LOCAL publish rate (SensorMediaPublisher, while robot_concept bridges) +
			// EXTERNAL received rate (bytes summed by the DDS monitor, while a hardware producer owns
			// the plane). Only one side is active per stream at a time, so the sum is the true rate
			// either way. External-only nodes (bpearl) carry no local publisher (keys empty). "helios" carries
			// BOTH: our own publisher while bridging, the external producer otherwise.
			const struct { const char* node; std::vector<std::string> keys;
			               std::atomic<std::uint64_t>* ext; bool enabled; } media_nodes[] = {
				{"zed",     {"rgb", "depth"}, &zed_bytes_,    params.ENABLE_ZED},
				{"ricoh",   {"rgb360"},       &ricoh_bytes_,  params.ENABLE_RICOH},
				{"helios",  {"lidar"},        &helios_bytes_, params.ENABLE_LIDAR},
				{"bpearl",  {},               &bpearl_bytes_, params.ENABLE_LIDAR},
				{"imu",     {"imu"},          &imu_bytes_,    params.ENABLE_IMU},
			};
			static std::map<std::string, std::uint64_t> last_ext_bytes;
			for (const auto& mn : media_nodes)
			{
				if (not mn.enabled)
					continue;
				double bps = mn.keys.empty() ? 0.0 : media_.current_bps(mn.keys);
				if (mn.ext != nullptr)
				{
					const std::uint64_t cur = mn.ext->load(std::memory_order_relaxed);
					auto [it, inserted] = last_ext_bytes.try_emplace(mn.node, cur);   // seed → no first-tick spike
					if (not inserted)
					{
						bps += static_cast<double>(cur - it->second) / dt;
						it->second = cur;
					}
				}
				if (auto n = G->get_node(mn.node); n.has_value())
				{
					G->add_or_modify_attrib_local<media_bps_att>(n.value(), static_cast<float>(bps));
					rc::safe_update_node(*G, n.value());
				}
			}

			// Publish the node→ICE-port map (constant; written only when missing/changed to avoid CRDT
			// churn) so the mind view can merge each SHM producer with its mediaplanedds endpoint.
			for (const auto& [pnode, pport] : media_ice_ports_)
				if (auto n = G->get_node(pnode); n.has_value())
				{
					const std::string val = std::to_string(pport);
					const auto it = n->attrs().find("media_ice_port");
					if (it == n->attrs().end() or it->second.str() != val)
					{
						G->add_or_modify_attrib_local<media_ice_port_att>(n.value(), val);
						rc::safe_update_node(*G, n.value());
					}
				}
	}

	// Low-rate media-source negotiation (~every 3 s): are zed_camera / ricoh_omni_dds
	// publishing their planes on DDS? Relay each descriptor + stop bridging if so; fall
	// back to bridging if not. Both run on the same timer, each gated by its own selector.
	static auto last_neg = std::chrono::steady_clock::now();
	if (std::chrono::duration<double>(now - last_neg).count() >= 3.0)
	{
		for (auto& g : media_groups_)
			negotiate(g);
		last_neg = now;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::emergency()
{
	qInfo() << "Emergency worker";
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}

void SpecificWorker::restore()
{
	qInfo() << "Restore worker";
    //restoreCODE
    //Restore emergency component
}

int SpecificWorker::startup_check()
{
	qInfo() << "Startup check";
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}


void SpecificWorker::read_lidar_thread()
{
	bool empty_lidar_logged = false;
	// Normalize a source timestamp (ns or ms) to epoch ms; see read_rgbd_thread.
	constexpr auto to_epoch_ms = [](long long t) -> std::uint64_t
	{
		return t > 1'000'000'000'000'000LL
		           ? static_cast<std::uint64_t>(t / 1'000'000)   // ns -> ms
		           : static_cast<std::uint64_t>(t);              // already ms
	};
	std::vector<float> lidar_xyz;   // reused interleaved xyz scratch for the media plane
	// Start fast so the loop oversamples the source and the EMA below can measure the TRUE source
	// period (starting slow would lock the loop to its own rate); it then converges up to ~1× period.
	auto wait_period = std::chrono::milliseconds(5);
	std::uint64_t last_lidar_stamp_ms = 0;   // self-sync: dedup by source stamp
	double lidar_src_period_ms = -1.0;       // self-sync: source period from stamp deltas (NOT wall clock)
	const bool lidar_upload_enabled = params.DSR_LIDAR_FPS >= 0;
	const auto lidar_interval_ms = params.DSR_LIDAR_FPS > 0
		? std::chrono::milliseconds(1000 / params.DSR_LIDAR_FPS)
		: std::chrono::milliseconds(0);
	auto last_lidar_upload = std::chrono::steady_clock::time_point{};
	// DDS monitors for the two external lidar planes (created on demand once their
	// descriptors are relayed onto the helios/bpearl nodes); reset when we bridge again.
	std::unique_ptr<rc::media::LidarSubscriber> helios_sub, bpearl_sub;
	// Retry cadence for bringing those monitors up. Once ONE of them is live this loop paces on its
	// 100 ms wait_and_poll (~10 Hz), and the sleep below only fires when NEITHER exists — so an
	// un-throttled retry re-asks the factory ten times a second for a plane whose producer simply is
	// not up yet, and each miss speaks. Throttle to ~1 Hz, matching LidarPlaneReader's own discovery.
	auto next_sub_attempt = std::chrono::steady_clock::time_point{};
	while (!stop_lidar_thread && !shutting_down_.load())
	{
		if (not bridge_lidar_.load(std::memory_order_relaxed))
		{
			// lidar3d_dds owns the planes. Don't bridge — SUBSCRIBE to helios+bpearl only to
			// report their FPS ([HeliosThread]/[BpearlThread] in compute()).
			if (const auto now = std::chrono::steady_clock::now();
			    (not helios_sub or not bpearl_sub) and now >= next_sub_attempt)
			{
				next_sub_attempt = now + std::chrono::seconds(1);
				if (not helios_sub) helios_sub = rc::media::make_lidar_subscriber_from_graph(*G, "helios", "lidar");
				if (not bpearl_sub) bpearl_sub = rc::media::make_lidar_subscriber_from_graph(*G, "bpearl", "lidar");
			}
			// Sum received bytes (count × stride × 4) so compute() can report each lidar's real SHM
			// throughput as media_bps, not only its frame rate.
			if (helios_sub)   // wait_and_poll paces the loop (blocks up to 100 ms)
				helios_frames_.fetch_add(
					helios_sub->wait_and_poll([this](const rc::media::LidarFrame& f, std::int64_t)
						{ helios_bytes_.fetch_add(static_cast<std::uint64_t>(f.count()) * f.stride() * sizeof(float),
						                          std::memory_order_relaxed); }, 100),
					std::memory_order_relaxed);
			if (bpearl_sub)
				bpearl_frames_.fetch_add(
					bpearl_sub->poll([this](const rc::media::LidarFrame& f, std::int64_t)
						{ bpearl_bytes_.fetch_add(static_cast<std::uint64_t>(f.count()) * f.stride() * sizeof(float),
						                          std::memory_order_relaxed); }),
					std::memory_order_relaxed);
			if (not helios_sub and not bpearl_sub)
				std::this_thread::sleep_for(std::chrono::milliseconds(200));   // descriptors not ready yet
			continue;
		}
		if (helios_sub) helios_sub.reset();   // producing/bridging again -> stop monitoring
		if (bpearl_sub) bpearl_sub.reset();
		RoboCompLidar3D::TData data;
		try
		{
			data = lidar3d_proxy->getLidarData(
				"", 0.f, static_cast<float>(M_PI) * 2.f, params.LIDAR_DECIMATION_FACTOR);
		}
		catch (const Ice::Exception& e)
		{
			qWarning() << "[read_lidar] getLidarData failed:" << e.what() << "retrying...";
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		if (data.points.empty())
		{
			if (!empty_lidar_logged)
			{
				std::print(stderr, "[read_lidar] Empty LiDAR stream received. Waiting for points...\n");
				empty_lidar_logged = true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		else if (empty_lidar_logged)
		{
			std::print("[read_lidar] LiDAR stream recovered.\n");
			empty_lidar_logged = false;
		}

		// Self-synchronize with the source: advance only on a genuinely new scan (dedup by timestamp)
		// and poll at ~2× the source period — derived from the SOURCE stamp deltas, not our own loop
		// timing (using wall-clock between our ingests is circular and death-spirals the rate down).
		const std::uint64_t stamp_ms = to_epoch_ms(data.timestamp);
		if (stamp_ms != 0 && stamp_ms == last_lidar_stamp_ms)
		{
			std::this_thread::sleep_for(wait_period);   // duplicate: wait one poll period, re-check
			continue;
		}
		if (stamp_ms != 0)
		{
			if (last_lidar_stamp_ms != 0)
			{
				const double src_dt = static_cast<double>(stamp_ms - last_lidar_stamp_ms);   // source period (ms)
				if (src_dt > 0.5 && src_dt < 2000.0)
				{
					lidar_src_period_ms = (lidar_src_period_ms < 0.0) ? src_dt : 0.8 * lidar_src_period_ms + 0.2 * src_dt;
					wait_period = std::chrono::milliseconds(std::max<long>(1, static_cast<long>(0.5 * lidar_src_period_ms + 0.5)));
				}
			}
			last_lidar_stamp_ms = stamp_ms;
		}

		// --- Media plane: publish the full-rate LiDAR scan (interleaved xyz, metres) ---
		// Carried OUT of the DSR graph, stamped per-frame. Independent of the throttled
		// DSR laser_* upload below (which may be decimated/disabled via DSR_LIDAR_FPS).
		if (media_.lidar_ready())
		{
			const std::size_t n = data.points.size();
			lidar_xyz.resize(n * 3);
			for (std::size_t i = 0; i < n; ++i)
			{
				lidar_xyz[3 * i + 0] = data.points[i].x * 0.001f;
				lidar_xyz[3 * i + 1] = data.points[i].y * 0.001f;
				lidar_xyz[3 * i + 2] = data.points[i].z * 0.001f;
			}
			SensorMediaPublisher::LidarFrameView v;
			v.stamp_ms = to_epoch_ms(data.timestamp);
			v.points   = lidar_xyz.data();
			v.count    = static_cast<std::uint32_t>(n);
			v.stride   = 3;
			v.format   = rc::media::LIDAR_FORMAT_XYZ_F32;
			media_.publish_lidar(v);
		}
		media_.maybe_report_stats(SensorMediaPublisher::StatsGroup::Lidar, std::chrono::seconds(5));

		const auto  now_steady_lidar  = std::chrono::steady_clock::now();
		const bool do_lidar_upload = lidar_upload_enabled && ((lidar_interval_ms.count() == 0)
			|| (now_steady_lidar - last_lidar_upload >= lidar_interval_ms));
		if (do_lidar_upload)
		{
			last_lidar_upload = now_steady_lidar;
			const auto n = data.points.size();
			std::vector<float> xs(n), ys(n), zs(n);
			for (std::size_t i = 0; i < n; ++i)
			{
				xs[i] = data.points[i].x * 0.001f;
				ys[i] = data.points[i].y * 0.001f;
				zs[i] = data.points[i].z * 0.001f;
			}
			if (auto laser_node = G->get_node("helios"); laser_node.has_value())
			{
				G->add_or_modify_attrib_local<laser_X_att>(laser_node.value(), std::move(xs));
				G->add_or_modify_attrib_local<laser_Y_att>(laser_node.value(), std::move(ys));
				G->add_or_modify_attrib_local<laser_Z_att>(laser_node.value(), std::move(zs));
				G->add_or_modify_attrib_local<laser_timestamp_att>(laser_node.value(), static_cast<uint64_t>(data.timestamp));
				// Last use of laser_node: move it in so the (multi-MB) laser blobs are
				// moved into the engine instead of deep-copied under the graph write lock.
				rc::safe_update_node(*G, std::move(laser_node.value()));
			}
			else if (!shutting_down_.load())
				qWarning() << "Laser node not found in DSR graph";
		}

		lidar_frames_.fetch_add(1, std::memory_order_relaxed);
		std::this_thread::sleep_for(wait_period);
	}
}

template <class Frame, class Sub>
void SpecificWorker::monitor_external_image_plane(std::unique_ptr<Sub>& sub,
                                                  const std::type_identity_t<std::function<std::unique_ptr<Sub>()>>& make_sub,
                                                  std::atomic<std::uint64_t>& frame_counter,
                                                  std::atomic<std::uint64_t>& byte_counter,
                                                  std::uint64_t& mon_frames,
                                                  std::chrono::steady_clock::time_point& report_at,
                                                  const char* producer, const char* stream_label)
{
	if (not sub)
	{
		sub = make_sub();          // descriptor relayed onto its DSR node -> bind the subscriber
		mon_frames = 0;
		report_at = std::chrono::steady_clock::now();
	}
	if (not sub)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(200));   // descriptor not ready yet
		return;
	}
	// Accumulate the received payload bytes (size() = valid image bytes) so compute() can report the
	// external stream's real SHM throughput as media_bps, not just its frame rate.
	const int n = sub->wait_and_poll([&byte_counter](const Frame& f, std::int64_t)
		{ byte_counter.fetch_add(static_cast<std::uint64_t>(f.size()), std::memory_order_relaxed); }, 200);
	mon_frames += static_cast<std::uint64_t>(n);
	// Feed the compute() heartbeat with the external rate: while bypassing, the producer path
	// never runs, so this counter only ever counts DDS frames — the heartbeat reads the live
	// topic instead of a misleading 0.0 Hz.
	frame_counter.fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);
	const auto now = std::chrono::steady_clock::now();
	const double dt = std::chrono::duration<double>(now - report_at).count();
	if (dt >= 5.0)
	{
		// ALIVE rate meter dropped (redundant with the compute() heartbeat); keep only the
		// exceptional SILENT stall warning per the stream-stall logging rule.
		if (mon_frames == 0)
			std::print("[Media] {} external DDS publisher SILENT — no {} frames on domain 7 in {:.0f}s\n",
			           producer, stream_label, dt);
		mon_frames = 0;
		report_at = now;
	}
}

void SpecificWorker::read_rgbd_thread()
{
	bool empty_rgbd_logged = false;
	// Start fast so the loop oversamples the camera and the EMA below measures the TRUE frame period
	// (starting slow would lock it to its own rate); it then converges up to ~1.1× the camera rate.
	auto wait_period = std::chrono::milliseconds(5);
	const bool rgb_upload_enabled = params.DSR_RGB_FPS >= 0;
	const bool depth_upload_enabled = params.DSR_DEPTH_FPS >= 0;
	const auto rgb_interval_ms = params.DSR_RGB_FPS > 0
		? std::chrono::milliseconds(1000 / params.DSR_RGB_FPS)
		: std::chrono::milliseconds(0);
	const auto depth_interval_ms = params.DSR_DEPTH_FPS > 0
		? std::chrono::milliseconds(1000 / params.DSR_DEPTH_FPS)
		: std::chrono::milliseconds(0);
	auto last_rgb_upload   = std::chrono::steady_clock::time_point{};
	auto last_depth_upload = std::chrono::steady_clock::time_point{};
	std::uint64_t last_rgbd_stamp_ms = 0;   // self-sync: dedup by source stamp
	double rgbd_src_period_ms = -1.0;       // self-sync: source period from stamp deltas (NOT wall clock)
	// External-publisher monitor (only used while zed_camera owns the plane).
	std::unique_ptr<rc::media::MediaSubscriber> ext_rgb_sub;
	std::uint64_t ext_frames = 0;
	auto ext_report_at = std::chrono::steady_clock::now();
	while (!stop_rgbd_thread && !shutting_down_.load())
	{
		if (not bridge_zed_.load(std::memory_order_relaxed))
		{
			// zed_camera is the external DDS producer (descriptor relayed into DSR by
			// negotiate()). Don't pull/publish; SUBSCRIBE to its rgb stream just to report
			// "external publish" + the observed frame rate (fed to the [ZEDThread] heartbeat).
			monitor_external_image_plane<rc::media::ImageFrame>(
				ext_rgb_sub, [this]{ return rc::media::make_image_subscriber_from_graph(*G, "zed", "rgb"); },
				rgbd_frames_, zed_bytes_, ext_frames, ext_report_at, "zed_camera", "rgb");
			continue;
		}
		if (ext_rgb_sub) ext_rgb_sub.reset();   // we are producing again -> stop monitoring
		const auto loop_start = std::chrono::steady_clock::now();
		try
		{
			RoboCompCameraRGBDSimple::TRGBD frame;
			try
			{
				frame = camerargbdsimple_proxy->getAll("camera");
			}
			catch (const Ice::Exception& e)
			{
				qWarning() << "[read_rgbd] getAll failed:" << e.what() << "retrying...";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			const bool empty_rgb = frame.image.width <= 0 || frame.image.height <= 0 || frame.image.image.empty();
			const bool empty_depth = frame.depth.width <= 0 || frame.depth.height <= 0 || frame.depth.depth.empty();
			// Point cloud is disabled in the webots_bridge (unused now); gate on rgb+depth only.
			if (empty_rgb || empty_depth)
			{
				if (!empty_rgbd_logged)
				{
					std::print("[read_rgbd] Empty RGBD stream received. Waiting for RGB and depth data...\n");
					empty_rgbd_logged = true;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
			else if (empty_rgbd_logged)
			{
				std::print("[read_rgbd] RGBD stream recovered.\n");
				empty_rgbd_logged = false;
			}

			// Self-synchronize with the source: advance only on a genuinely new frame (dedup by the RGB
			// capture stamp) and poll at just above the source rate (~1.1×) — derived from the SOURCE stamp
			// deltas, not our own loop timing (wall-clock between ingests is circular and death-spirals the
			// rate down). getAll() ships a full image even for a DUPLICATE, so every extra poll is a full
			// wasted RGBD transfer over Ice; keep the poll factor just above 1× to not miss frames while
			// minimizing double-pulls. The dedup below only stops RE-PUBLISHING — the Ice payload already
			// crossed the wire — so the poll factor, not the dedup, is what bounds zed->robot_concept flow.
			{
				const long long rgb_t = frame.image.alivetime;
				const std::uint64_t stamp_ms = rgb_t > 1'000'000'000'000'000LL
					? static_cast<std::uint64_t>(rgb_t / 1'000'000) : static_cast<std::uint64_t>(rgb_t);
				if (stamp_ms != 0 && stamp_ms == last_rgbd_stamp_ms)
				{
					std::this_thread::sleep_for(wait_period);   // duplicate: wait one poll period, re-check
					continue;
				}
				if (stamp_ms != 0)
				{
					if (last_rgbd_stamp_ms != 0)
					{
						const double src_dt = static_cast<double>(stamp_ms - last_rgbd_stamp_ms);   // source period (ms)
						if (src_dt > 0.5 && src_dt < 2000.0)
						{
							rgbd_src_period_ms = (rgbd_src_period_ms < 0.0) ? src_dt : 0.8 * rgbd_src_period_ms + 0.2 * src_dt;
							// Poll at ~1.1× the source rate (wait 0.9× the source period): fast enough to
							// never miss a frame, but avoids the ~2× full-image over-pull that 0.5× caused.
							wait_period = std::chrono::milliseconds(std::max<long>(1, static_cast<long>(0.9 * rgbd_src_period_ms + 0.5)));
						}
					}
					last_rgbd_stamp_ms = stamp_ms;
				}
			}

			// --- Media plane: publish full-rate RGB + depth pixels via zero-copy DDS ---
			// This carries the heavy payload OUT of the DSR graph. SensorMediaPublisher
			// copies into the loaned SHM slot, leaving the source buffers intact for the
			// (optional) DSR writes below, which can still std::move them.
			if (media_.ready())
			{
				// Normalize the camera alivetime to epoch MILLISECONDS so the media
				// stamp matches the rest of the system (joints, lidar timestamp_ms).
				// The ZED SDK reports epoch ns (~1.7e18); a Webots bridge may already
				// report ms (~1.7e12). Scale down only ns-magnitude values, so this is
				// correct whichever source is live (and a no-op if already ms).
				constexpr auto to_epoch_ms = [](long long t) -> std::uint64_t
				{
					return t > 1'000'000'000'000'000LL
					           ? static_cast<std::uint64_t>(t / 1'000'000)   // ns -> ms
					           : static_cast<std::uint64_t>(t);              // already ms
				};
				const auto &img = frame.image;
				// Tag the TRUE channel order: the webots-bridge component (our live CameraRGBDSimple source)
				// delivers RGB-ordered bytes, so publish them as FORMAT_RGB8. Consumers with an RGB8 path
				// (retina RGB2BGR, the room/robot viewers no-swap) then get correct colours without the
				// per-consumer BGR8 workaround. If a real ZED emitting true BGR is ever bridged here, tag
				// FORMAT_BGR8 for that source instead (drive it from config rather than hard-coding).
				media_.publish_image("rgb", {
					.stamp_ms = to_epoch_ms(img.alivetime),
					.width    = static_cast<std::uint32_t>(img.width),
					.height   = static_cast<std::uint32_t>(img.height),
					.step     = static_cast<std::uint32_t>(img.width) * 3u,
					.format   = rc::media::FORMAT_RGB8,
					.data     = reinterpret_cast<const std::uint8_t*>(img.image.data()),
					.nbytes   = img.image.size()});

				const auto &dep = frame.depth;
				const std::size_t dep_pix = static_cast<std::size_t>(dep.width) * static_cast<std::size_t>(dep.height);
				if (dep_pix > 0)
				{
					const std::size_t bpp = dep.depth.size() / dep_pix;
					const bool f32 = (bpp == 4);
					media_.publish_image("depth", {
						.stamp_ms = to_epoch_ms(dep.alivetime),
						.width    = static_cast<std::uint32_t>(dep.width),
						.height   = static_cast<std::uint32_t>(dep.height),
						.step     = static_cast<std::uint32_t>(dep.width) * (f32 ? 4u : 2u),
						.format   = f32 ? rc::media::FORMAT_DEPTH_F32 : rc::media::FORMAT_Z16,
						.data     = reinterpret_cast<const std::uint8_t*>(dep.depth.data()),
						.nbytes   = dep.depth.size()});
				}
			}
			media_.maybe_report_stats(SensorMediaPublisher::StatsGroup::Image, std::chrono::seconds(5));

			const auto now_steady = std::chrono::steady_clock::now();
			const bool do_rgb   = rgb_upload_enabled && ((rgb_interval_ms.count()   == 0) || (now_steady - last_rgb_upload   >= rgb_interval_ms));
			const bool do_depth = depth_upload_enabled && ((depth_interval_ms.count() == 0) || (now_steady - last_depth_upload >= depth_interval_ms));
			if (auto cam_node = G->get_node("zed"); cam_node.has_value())
			{
				if (do_rgb)
				{
					last_rgb_upload = now_steady;
					G->add_or_modify_attrib_local<cam_rgb_width_att>(cam_node.value(), frame.image.width);
					G->add_or_modify_attrib_local<cam_rgb_height_att>(cam_node.value(), frame.image.height);
					G->add_or_modify_attrib_local<cam_rgb_focalx_att>(cam_node.value(), frame.image.focalx);
					G->add_or_modify_attrib_local<cam_rgb_focaly_att>(cam_node.value(), frame.image.focaly);
					G->add_or_modify_attrib_local<cam_rgb_depth_att>(cam_node.value(), 3);
					G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(cam_node.value(), 0);
					G->add_or_modify_attrib_local<cam_rgb_att>(cam_node.value(), std::move(frame.image.image));
					G->add_or_modify_attrib_local<cam_rgb_alivetime_att>(cam_node.value(), static_cast<std::uint64_t>(frame.image.alivetime));
					rc::safe_update_node(*G, cam_node.value());
				}

				if (do_depth)
				{
					last_depth_upload = now_steady;
					G->add_or_modify_attrib_local<cam_depth_width_att>(cam_node.value(), frame.depth.width);
					G->add_or_modify_attrib_local<cam_depth_height_att>(cam_node.value(), frame.depth.height);
					G->add_or_modify_attrib_local<cam_depth_focalx_att>(cam_node.value(), frame.depth.focalx);
					G->add_or_modify_attrib_local<cam_depth_focaly_att>(cam_node.value(), frame.depth.focaly);
					G->add_or_modify_attrib_local<cam_depthFactor_att>(cam_node.value(), frame.depth.depthFactor);
					G->add_or_modify_attrib_local<cam_depth_att>(cam_node.value(), std::move(frame.depth.depth));
					// Last use of cam_node in this block: move it in so the camera blobs are
					// moved into the engine instead of deep-copied under the graph write lock.
					rc::safe_update_node(*G, std::move(cam_node.value()));
				}
			}
			else if (!shutting_down_.load())
				qWarning() << "Camera node not found in DSR graph";

			rgbd_frames_.fetch_add(1, std::memory_order_relaxed);
			const auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
			if (loop_elapsed < wait_period)
				std::this_thread::sleep_for(wait_period - loop_elapsed);
		}
		catch (const Ice::Exception& e)
		{
			qWarning() << "[read_rgbd] Ice exception:" << e.what();
		}
	}
}

void SpecificWorker::build_media_groups()
{
	// One entry per sensor plane group. The proxy pointers bind to the generated members:
	//   mediaplanedds_proxy   Proxies.MediaPlaneDDS   zed_camera      port 12002
	//   mediaplanedds1_proxy  Proxies.MediaPlaneDDS1  ricoh_omni_dds  port 10099
	//   mediaplanedds2_proxy  Proxies.MediaPlaneDDS2  lidar3d_dds helios  port 11890
	//   mediaplanedds3_proxy  Proxies.MediaPlaneDDS3  lidar3d_dds bpearl  port 11889 (Shadow only)
	// The lidar group holds the physical lidars under a single shared bridge_lidar_: it only stops
	// bridging once EVERY plane in the group relays a descriptor (see negotiate()). advertise_node/
	// -streams is what robot_concept re-advertises as its own plane while it is the producer (bridging).
	//
	// A plane whose DSR node is ABSENT is not added at all. The graph loaded from Agent.configFile is
	// what says which sensors this robot actually carries: shadow.json has helios + bpearl, p3bot.json
	// has helios only. Registering a plane the robot does not have would leave `p.present` false for
	// ever, so negotiate()'s all_up could never become true and "auto" would bridge over Ice for ever
	// while the real helios DDS producer was ignored — a silent, permanent half-failure. Skipping it
	// keeps the group honest: it waits for exactly the producers this robot has.
	//
	// Built imperatively (not a braced initializer_list): MediaPlane holds a std::future and is
	// therefore move-only, which an initializer_list can't hold.
	const auto add_plane = [this](MediaGroup& g, std::string node,
	                              const RoboCompMediaPlaneDDS::MediaPlaneDDSPrxPtr* prx)
	{
		if (not G->get_node(node).has_value())
		{
			qInfo() << "[MediaNeg]" << QString::fromStdString(g.tag) << ": no"
			        << QString::fromStdString(node)
			        << "node in the graph — this robot has no such sensor; plane not registered";
			return;
		}
		auto& p = g.planes.emplace_back();
		p.node = std::move(node);
		p.proxy = prx;
	};

	media_groups_.clear();
	media_groups_.reserve(4);

	auto& zed = media_groups_.emplace_back();
	zed.tag = "ZED"; zed.bridge = &bridge_zed_; zed.source = &params.ZED_SOURCE;
	zed.enabled = &params.ENABLE_ZED; zed.advertise_node = "zed"; zed.advertise_streams = {"rgb", "depth"};
	add_plane(zed, "zed", &mediaplanedds_proxy);

	auto& ricoh = media_groups_.emplace_back();
	ricoh.tag = "360"; ricoh.bridge = &bridge_ricoh_; ricoh.source = &params.RICOH_SOURCE;
	ricoh.enabled = &params.ENABLE_RICOH; ricoh.advertise_node = "ricoh"; ricoh.advertise_streams = {"rgb360"};
	add_plane(ricoh, "ricoh", &mediaplanedds1_proxy);

	auto& lidar = media_groups_.emplace_back();
	lidar.tag = "LiDAR"; lidar.bridge = &bridge_lidar_; lidar.source = &params.LIDAR_SOURCE;
	lidar.enabled = &params.ENABLE_LIDAR; lidar.advertise_node = "helios"; lidar.advertise_streams = {"lidar"};
	add_plane(lidar, "helios", &mediaplanedds2_proxy);
	add_plane(lidar, "bpearl", &mediaplanedds3_proxy);

	// IMU. That day came: imu_dds (Proxies.MediaPlaneDDS4, port 11891) is the external producer, so
	// this plane negotiates exactly like the others — "auto" adopts it as soon as it relays a
	// descriptor, and read_imu_thread then stops pulling over Ice and only subscribes to report the
	// observed rate. Point MediaPlaneDDS4 at a dead port (or force "ice") to go back to bridging.
	auto& imu = media_groups_.emplace_back();
	imu.tag = "IMU"; imu.bridge = &bridge_imu_; imu.source = &params.IMU_SOURCE;
	imu.enabled = &params.ENABLE_IMU; imu.advertise_node = "imu"; imu.advertise_streams = {"imu"};
	add_plane(imu, "imu", &mediaplanedds4_proxy);

	// Resolve each plane's MediaPlaneDDS ICE port from its proxy string ("… -p 12002 …") so the mind
	// view can fuse the SHM producer node with its mediaplanedds:<port> endpoint. Parsed once here.
	media_ice_ports_.clear();
	for (auto& g : media_groups_)
		for (auto& p : g.planes)
			if (p.proxy and *p.proxy)
			{
				const std::string s = (*p.proxy)->ice_toString();
				if (const auto pos = s.find("-p "); pos != std::string::npos)
					try { media_ice_ports_[p.node] = std::stoi(s.substr(pos + 3)); } catch (...) {}
			}
}

void SpecificWorker::prime_media_groups()
{
	// One bounded round at startup so adoption is near-immediate instead of waiting for the
	// first ~3 s compute() tick: fire every plane's async query concurrently, wait a short
	// bounded time for the in-flight results (down peers fail fast on localhost), then run a
	// normal negotiate() pass to harvest + decide. Bounded blocking here only, once.
	for (auto& g : media_groups_)
		if (*g.enabled and *g.source != "ice")
			for (auto& p : g.planes)
				if (p.proxy and *p.proxy)
					try { p.pending = (*p.proxy)->getMediaDescriptorAsync(); }
					catch (const Ice::Exception&) { p.present = false; }
	for (auto& g : media_groups_)
		for (auto& p : g.planes)
			if (p.pending.valid())
				p.pending.wait_for(std::chrono::milliseconds(300));   // concurrent -> total ~= max, not sum
	for (auto& g : media_groups_)
		negotiate(g);
}

void SpecificWorker::negotiate(MediaGroup& g)
{
	if (not *g.enabled)
		return;   // whole path disabled — nothing to negotiate

	// Forced "ice": never defer to an external DDS producer — stay bridging via Ice RPC.
	if (*g.source == "ice")
	{
		g.bridge->store(true, std::memory_order_relaxed);
		return;
	}

	// Non-blocking per plane: harvest a completed async query (if ready), relay its descriptor,
	// then relaunch. p.present carries the last completed result across ticks so the compute()
	// thread never blocks on getMediaDescriptor(). all_up gates the group's bridge: EVERY plane the
	// group registered must have its producer up. build_media_groups() only registers planes whose
	// DSR sensor node exists, so on a robot without a bpearl the lidar group is helios alone and
	// all_up means exactly that. An EMPTY group registered nothing (the robot has none of these
	// sensors) — treat that as "no external producer found" and keep bridging, rather than letting a
	// vacuous all-of-nothing hand the plane to a DDS producer that does not exist.
	bool all_up = not g.planes.empty();
	for (auto& p : g.planes)
	{
		if (p.pending.valid() and p.pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			std::string desc;
			try { desc = p.pending.get(); }
			catch (const Ice::Exception&) { desc.clear(); }   // peer down/hung -> absent
			// A descriptor only counts as an adoption signal when the producer says its stream is
			// actually LIVE. ready=false means "plane advertised, no data flowing" — adopting that
			// stops our bridge and hands the sensor to a silent producer, taking it dark. Seen for
			// real: an imu_dds pointed at the wrong IMU port had its DDS writer up (so it answered
			// with a descriptor) while every getDataImu() was refused. Unparseable JSON is treated
			// the same as absent, which keeps us bridging — the safe side.
			p.present = false;
			if (not desc.empty())
				if (const auto d = rc::media::MediaDescriptor::from_json(desc); d.has_value())
				{
					p.present = d->ready;
					// Latch the physics whenever the driver states it, INDEPENDENTLY of `ready`.
					// A producer that is up but not yet streaming still knows its own geometry and
					// noise, and that fact does not expire when the stream does.
					if (not d->model.empty())
						p.model = d->model;
				}
			if (p.present and desc != p.last_relayed)
			{
				write_media_descriptor(p.node, desc);
				p.last_relayed = desc;
			}
		}
		if (not p.pending.valid() and p.proxy and *p.proxy)
		{
			try { p.pending = (*p.proxy)->getMediaDescriptorAsync(); }
			catch (const Ice::Exception&) { p.present = false; }   // couldn't even dispatch
		}
		if (not p.present)
			all_up = false;   // peer absent -> keep/resume bridging
	}

	// Forced "dds": always treat the external producer(s) as authoritative (never bridge).
	// Relays above still ran so the monitor's subscriber can bind to the plane.
	if (*g.source == "dds")
	{
		g.bridge->store(false, std::memory_order_relaxed);
		return;
	}

	// auto: hand the plane(s) to DDS once every producer is up; resume bridging if any drops.
	if (all_up)
	{
		if (g.bridge->exchange(false))
			qInfo().noquote() << QString::asprintf(
				"[MediaNeg] %s: all producers publishing on DDS — relaying descriptors; robot_concept stops bridging",
				g.tag.c_str());
	}
	else
	{
		if (not g.bridge->exchange(true))
			qInfo().noquote() << QString::asprintf(
				"[MediaNeg] %s: a producer is absent — robot_concept resumes bridging", g.tag.c_str());
		// coverage dropped: re-advertise our own plane and forget the relays.
		bool had_relay = false;
		for (auto& p : g.planes)
			if (not p.last_relayed.empty()) { p.last_relayed.clear(); had_relay = true; }
		if (had_relay)
		{
			// Carry the physics across the transport switch: our own bridged descriptor
			// re-states whatever model this node's driver last advertised, so geometry and
			// noise do not blink out of the graph at the moment the sensor path degrades.
			const rc::media::SensorModel* m = nullptr;
			for (const auto& p : g.planes)
				if (p.node == g.advertise_node and not p.model.empty()) { m = &p.model; break; }
			write_media_descriptor(g.advertise_node,
			                       media_.descriptor_json(g.advertise_streams, m));
		}
	}
}

void SpecificWorker::write_media_descriptor(const std::string& node_name, const std::string& descriptor_json)
{
	auto node = G->get_node(node_name);
	if (not node.has_value())
	{
		qWarning() << "[MediaNeg]" << QString::fromStdString(node_name)
		           << "node not found — cannot relay media descriptor";
		return;
	}
	G->add_or_modify_attrib_local<media_descriptor_att>(node.value(), descriptor_json);
	rc::safe_update_node(*G, node.value());
}

void SpecificWorker::trigger_graph_layout()
{
	// Re-run the DSR graph viewer's automatic layout with the Graphviz engine from Agent.graph_layout
	// so the node/edge graph is legible at startup. No-op when the graph view is disabled
	// (Agent.graph=false) or not yet created. Main-thread only (graph/GUI access).
	const auto it = graph_viewers.find("");
	if (it == graph_viewers.end() || !it->second)
		return;

	QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
	auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
	if (!graph_viewer)
		return;

	// Run now and once more queued, so the layout also happens after any pending node/edge
	// update signals have been processed by the viewer.
	const std::string alg = graph_layout_;
	graph_viewer->compute_layout(alg.c_str());
	QMetaObject::invokeMethod(graph_viewer,
	                          [graph_viewer, alg]() { graph_viewer->compute_layout(alg.c_str()); },
	                          Qt::QueuedConnection);
}

void SpecificWorker::wire_view_data_signal()
{
	// Right-clicking a node whose raw stream lives on the media plane (no inline blob in the graph)
	// makes dsr_gui emit GraphViewer::view_data_signal(id, type). robot_concept is the one agent with
	// the graph up AND subscribed to every stream, so it answers with a live per-node FPS viewer.
	// Main-thread only (graph/GUI access); no-op when the graph view is disabled (Agent.graph=false).
	const auto it = graph_viewers.find("");
	if (it == graph_viewers.end() || !it->second)
		return;
	auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(it->second->get_widget(DSR::DSRViewer::view::graph));
	if (!graph_viewer)
		return;
	// QueuedConnection + UniqueConnection: the source signal originates on the GUI thread, but keep the
	// CLAUDE.md discipline uniform, and make a second call (re-init) idempotent rather than double-wire.
	QObject::connect(graph_viewer, &DSR::GraphViewer::view_data_signal, this,
	                 &SpecificWorker::open_stream_viewer,
	                 static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
}

void SpecificWorker::wire_agent_status_overlay()
{
	// Every agent already paints its own node under "mind" (rc::AgentStatePublisher writes
	// agent_fsm_state/agent_presence_state/color on it). The one state nobody can self-report is
	// death: a crashed agent's node just freezes on its last colour. Since robot_concept is the agent
	// that displays the complete graph, it is the natural observer — it ages each agent's heartbeat
	// and greys out the ones that stopped beating, so "everything green" actually means everything is
	// alive AND working. Main-thread only; no-op when the graph view is disabled (Agent.graph=false).
	const auto it = graph_viewers.find("");
	if (it == graph_viewers.end() || !it->second)
		return;
	auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(it->second->get_widget(DSR::DSRViewer::view::graph));
	if (!graph_viewer)
		return;
	// Display-only horizon, INTENTIONALLY decoupled from Presence.heartbeat_timeout_ms. That one is
	// deliberately long (15-25 s across the fleet) because acting on a short silence — declaring a
	// required peer lost and shutting down — proved hypersensitive to graph-publish stalls and to the
	// retina's restart window. Greying a node early costs nothing and is undone the moment the
	// heartbeat resumes, so the display gets its own, much shorter knob.
	const int stale_after_ms = configLoader.exists("Presence.stale_display_ms")
	                         ? configLoader.get<int>("Presence.stale_display_ms")
	                         : 3000;
	agent_status_overlay_.start(G, graph_viewer, stale_after_ms);
}

void SpecificWorker::open_stream_viewer(std::uint64_t node_id, const std::string &type)
{
	// Map the clicked DSR node to its media-plane stream and open the matching viewer. Names are the
	// media descriptor node names authored in initialize() (zed/ricoh/helios/imu) plus the per-device
	// lidar nodes (helios/bpearl). Each viewer builds its OWN subscriber via the shared factory here on
	// the main thread (the required consumer pattern) and polls it itself.
	std::string node_name;
	if (auto n = G->get_node(node_id); n.has_value())
		node_name = n->name();
	qInfo() << "[view-data] request for node" << QString::fromStdString(node_name)
	        << "id" << node_id << "type" << QString::fromStdString(type);

	// One live viewer per node id: re-clicking raises the existing one instead of stacking windows.
	if (auto existing = stream_viewers_.find(node_id); existing != stream_viewers_.end() and existing->second)
	{
		existing->second->show();
		existing->second->raise();
		existing->second->activateWindow();
		return;
	}

	QWidget *viewer = nullptr;
	if (node_name == "zed")
	{
		if (auto sub = rc::media::make_image_subscriber_from_graph(*G, "zed", "rgb"))
			viewer = new rc::viewers::ImageStreamViewer(std::move(sub), "zed — RGB (media plane)");
	}
	else if (node_name == "ricoh")
	{
		if (auto sub = rc::media::make_image360_subscriber_from_graph(*G, "ricoh", "rgb360"))
			viewer = new rc::viewers::Image360Viewer(std::move(sub), "ricoh — 360 panorama (media plane)");
	}
	else if (node_name == "helios" or node_name == "bpearl")
	{
		// each node shows its own ring.
		std::unique_ptr<rc::media::LidarSubscriber> helios, bpearl;
		if (node_name != "bpearl") helios = rc::media::make_lidar_subscriber_from_graph(*G, "helios", "lidar");
		if (node_name != "helios") bpearl = rc::media::make_lidar_subscriber_from_graph(*G, "bpearl", "lidar");
		if (helios or bpearl)
			viewer = new rc::viewers::LidarStreamViewer(std::move(helios), std::move(bpearl),
			                                            QString::fromStdString(node_name + " — points (media plane)"));
	}
	else if (node_name == "imu")
	{
		if (auto sub = rc::media::make_imu_subscriber_from_graph(*G, "imu", "imu"))
			viewer = new rc::viewers::ImuStreamViewer(std::move(sub), "imu — data (media plane)");
	}
	else if (type == "room")
	{
		// Data lives in the graph node (delimiting_polygon_x/y), not the media plane → read from G.
		viewer = new rc::viewers::RoomPolygonViewer(G, node_id, "room — delimiting polygon (graph)");
	}
	else if (type == "robot")
	{
		// 3D OpenGL view of the .obj mesh named in the node's `path` attribute.
		viewer = new rc::viewers::RobotMeshViewer(G, node_id, "robot — mesh (path attr)");
	}
	else if (node_name == "semantic")
	{
		// Dense ADE20K-150 label map lives as attributes on the node (semantic_labels/width/height),
		// published low-freq by the retina → colourise + hover-readout from G, no media plane.
		// NB: the retina's semantic/skeleton/masks nodes all share type "semantic_grid", so these
		// branch on node NAME, not type.
		viewer = new rc::viewers::SemanticGridViewer(G, node_id, "semantic — ADE20K label map (graph)");
	}
	else if (node_name == "skeleton")
	{
		// BODY_18 human poses live as attributes on the node (skeleton_count + skeleton_kp_xyz,
		// count*18*3 floats, ZED camera frame) → 3D OpenGL skeleton view read from G.
		viewer = new rc::viewers::SkeletonNodeViewer(G, node_id, "skeleton — BODY_18 poses (graph)");
	}
	else if (node_name == "residual")   // node renamed "grid"→"residual" (type stays "grid")
	{
		// residual_concept's Beta occupancy belief field (grid_occupancy_prob/var + meta) + occupied/
		// border cell layers → 3D risk-column field, mirroring the retina residual display.
		viewer = new rc::viewers::GridNodeViewer(G, node_id, "residual — belief field (graph)");
	}

	if (viewer == nullptr)
	{
		qInfo() << "[view-data] node" << QString::fromStdString(node_name)
		        << "(type" << QString::fromStdString(type)
		        << ") has no media-plane stream to view (producer not up yet, or unsupported type)";
		return;
	}

	// Drop the map entry when the window is closed (WA_DeleteOnClose) so a later click re-creates it.
	QObject::connect(viewer, &QObject::destroyed, this, [this, node_id](QObject *)
	{
		stream_viewers_.erase(node_id);
	});
	stream_viewers_[node_id] = viewer;
	viewer->show();
}

void SpecificWorker::read_ricoh_thread()
{
	if (camera360rgb_proxy == nullptr)
	{
		qWarning() << "[read_ricoh] no Camera360RGB proxy configured — ricoh media stream disabled";
		return;
	}
	// Bridge Ice → media plane: pull the RAW stitched 360 panorama straight from the
	// Camera360RGB source (the webots-bridge component) and republish it on the zero-copy
	// DDS plane (rc/ricoh/rgb). This bypasses RGBD_360's lidar-fusion hop, which caps the
	// rate at ~12 Hz; the raw panorama runs at the bridge's full rate. RGB only (no depth).
	bool empty_logged = false;
	// Start fast so the loop oversamples the source; the EMA converges the poll
	// period up to ~1× the source period (see read_rgbd_thread for the rationale).
	auto wait_period = std::chrono::milliseconds(5);
	std::uint64_t last_stamp_ms = 0;        // self-sync: dedup by source stamp
	double src_period_ms = -1.0;            // self-sync: source period from stamp deltas
	// getROI can throw / return empty while the Camera360RGB source is not yet up. Treat
	// that as peer-not-ready: back off and log only the first failure + a periodic
	// heartbeat instead of flooding at 10 Hz.
	std::uint64_t getroi_fail = 0;
	constexpr std::uint64_t GETROI_LOG_EVERY = 50;   // ~ every 50 backoff cycles
	constexpr auto to_epoch_ms = [](long long t) -> std::uint64_t
	{
		return t > 1'000'000'000'000'000LL
		           ? static_cast<std::uint64_t>(t / 1'000'000)   // ns -> ms
		           : static_cast<std::uint64_t>(t);              // already ms
	};
	// External-publisher monitor (only used while ricoh_omni_dds owns the 360 plane).
	std::unique_ptr<rc::media::Image360Subscriber> ext_ricoh_sub;
	std::uint64_t ext_ricoh_frames = 0;
	auto ext_ricoh_report_at = std::chrono::steady_clock::now();
	while (!stop_ricoh_thread && !shutting_down_.load())
	{
		if (not bridge_ricoh_.load(std::memory_order_relaxed))
		{
			// ricoh_omni_dds is the external DDS producer (descriptor relayed into DSR by
			// negotiate()). Don't pull/publish; SUBSCRIBE to its 360 stream just to report
			// "external publish" + the observed frame rate. Mirror of the ZED monitor branch.
			monitor_external_image_plane<rc::media::Image360Frame>(
				ext_ricoh_sub, [this]{ return rc::media::make_image360_subscriber_from_graph(*G, "ricoh", "rgb360"); },
				ricoh_frames_, ricoh_bytes_, ext_ricoh_frames, ext_ricoh_report_at, "ricoh_omni_dds", "360");
			continue;
		}
		if (ext_ricoh_sub) ext_ricoh_sub.reset();   // we are producing again -> stop monitoring
		const auto loop_start = std::chrono::steady_clock::now();
		try
		{
			RoboCompCamera360RGB::TImage frame;
			try
			{
				// Full panorama: getROI(-1,…) returns the whole stitched image.
				frame = camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
			}
			catch (const Ice::Exception& e)
			{
				if (getroi_fail++ % GETROI_LOG_EVERY == 0)
					qWarning() << "[read_ricoh] Camera360RGB getROI unavailable (source not up"
					           << "yet?) — backing off. err:" << e.what();
				std::this_thread::sleep_for(std::chrono::milliseconds(500));   // peer-not-ready backoff
				continue;
			}
			if (getroi_fail != 0)
			{
				qInfo() << "[read_ricoh] getROI recovered after" << getroi_fail << "failed attempts";
				getroi_fail = 0;
			}

			const bool empty_rgb = frame.width <= 0 || frame.height <= 0 || frame.image.empty();
			if (empty_rgb)
			{
				if (!empty_logged)
				{
					std::print("[read_ricoh] Empty 360 stream received. Waiting for Camera360RGB data...\n");
					empty_logged = true;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
			else if (empty_logged)
			{
				std::print("[read_ricoh] 360 stream recovered.\n");
				empty_logged = false;
			}

			// Self-synchronize with the source: advance only on a genuinely new frame
			// (dedup by capture stamp) and pace at ~2× the source period, derived from
			// SOURCE stamp deltas (not wall clock — that death-spirals the rate).
			{
				const std::uint64_t stamp_ms = to_epoch_ms(frame.timestamp);
				if (stamp_ms != 0 && stamp_ms == last_stamp_ms)
				{
					std::this_thread::sleep_for(wait_period);
					continue;
				}
				if (stamp_ms != 0)
				{
					if (last_stamp_ms != 0)
					{
						const double src_dt = static_cast<double>(stamp_ms - last_stamp_ms);
						if (src_dt > 0.5 && src_dt < 2000.0)
						{
							src_period_ms = (src_period_ms < 0.0) ? src_dt : 0.8 * src_period_ms + 0.2 * src_dt;
							wait_period = std::chrono::milliseconds(std::max<long>(1, static_cast<long>(0.5 * src_period_ms + 0.5)));
						}
					}
					last_stamp_ms = stamp_ms;
				}
			}

			// Publish the RGB panorama on the wide Image360Frame plane. The Camera360RGB source is
			// webots-derived and delivers RGB-ordered bytes (same as the ZED path above), so tag the
			// TRUE order IMG360_FORMAT_RGB8. Consumers' RGB8 path (retina RGB2BGR) then yields correct
			// colours without the BGR8 workaround. Tag BGR8 only for a source that emits genuine BGR.
			if (media_.image360_ready())
			{
				media_.publish_image360({
					.stamp_ms = to_epoch_ms(frame.timestamp),
					.width    = static_cast<std::uint32_t>(frame.width),
					.height   = static_cast<std::uint32_t>(frame.height),
					.step     = static_cast<std::uint32_t>(frame.width) * 3u,
					.format   = rc::media::IMG360_FORMAT_RGB8,
					.data     = reinterpret_cast<const std::uint8_t*>(frame.image.data()),
					.nbytes   = frame.image.size()});
			}
			media_.maybe_report_stats(SensorMediaPublisher::StatsGroup::Image360, std::chrono::seconds(5));

			ricoh_frames_.fetch_add(1, std::memory_order_relaxed);
			const auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
			if (loop_elapsed < wait_period)
				std::this_thread::sleep_for(wait_period - loop_elapsed);
		}
		catch (const Ice::Exception& e)
		{
			qWarning() << "[read_ricoh] Ice exception:" << e.what();
		}
	}
}

void SpecificWorker::read_imu_thread()
{
	// A null IMU proxy no longer kills this thread — it only rules out BRIDGING. When imu_dds owns
	// the plane the monitor branch below needs no proxy at all, and reporting that external rate is
	// precisely what this thread is for then.
	if (imu_proxy == nullptr)
		qWarning() << "[read_imu] no IMU proxy configured — Ice bridging disabled (external DDS producer only)";
	bool error_logged = false;
	// Pace by the measured IMU interval (NOT the Compute period, which hard-capped it). Start small
	// so it ramps up to the source rate rather than down from a slow 10 Hz.
	auto wait_period = std::chrono::milliseconds(5);
	std::uint64_t last_imu_stamp_ms = 0;   // self-sync: dedup by source stamp
	double imu_src_period_ms = -1.0;       // self-sync: source period from stamp deltas (NOT wall clock)
	// Normalize a source timestamp (ns or ms) to epoch ms; see read_rgbd_thread.
	constexpr auto to_epoch_ms = [](long long t) -> std::uint64_t
	{
		return t > 1'000'000'000'000'000LL
		           ? static_cast<std::uint64_t>(t / 1'000'000)   // ns -> ms
		           : static_cast<std::uint64_t>(t);              // already ms
	};
	// External-publisher monitor (only used while imu_dds owns the plane).
	std::unique_ptr<rc::media::ImuSubscriber> ext_imu_sub;
	// Throttle the discovery retry to ~1 Hz. Once the subscriber is live the loop paces on its
	// wait_and_poll; the sleep below only fires while it does NOT exist, so an un-throttled retry
	// would re-ask the factory five times a second and each miss speaks. Mirrors the lidar monitor.
	auto next_imu_sub_attempt = std::chrono::steady_clock::time_point{};
	while (!stop_imu_thread && !shutting_down_.load())
	{
		if (not bridge_imu_.load(std::memory_order_relaxed))
		{
			// imu_dds is the external DDS producer (its descriptor was relayed onto the "imu" node by
			// negotiate()). Don't pull over Ice and don't publish — the samples pass straight through
			// from imu_dds to the consumers. SUBSCRIBE only to note the observed frequency, so the
			// [IMUThread] heartbeat reads the live topic instead of a misleading 0.0 Hz.
			if (const auto now = std::chrono::steady_clock::now();
			    not ext_imu_sub and now >= next_imu_sub_attempt)
			{
				next_imu_sub_attempt = now + std::chrono::seconds(1);
				ext_imu_sub = rc::media::make_imu_subscriber_from_graph(*G, "imu", "imu");
			}
			if (ext_imu_sub)   // wait_and_poll paces the loop (blocks up to 100 ms)
				imu_frames_.fetch_add(
					ext_imu_sub->wait_and_poll([this](const rc::media::ImuFrame&, std::int64_t)
						{
							// ImuFrame is bounded+plain (is_plain()==true), so its sizeof IS the
							// per-sample wire cost — no length field to read like the image/lidar planes.
							imu_bytes_.fetch_add(sizeof(rc::media::ImuFrame), std::memory_order_relaxed);
						}, 100),
					std::memory_order_relaxed);
			else
				std::this_thread::sleep_for(std::chrono::milliseconds(200));   // descriptor not ready yet
			continue;
		}
		if (ext_imu_sub)
		{
			ext_imu_sub.reset();       // we are bridging again -> stop monitoring
			last_imu_stamp_ms = 0;     // the monitoring gap is not a source period; don't feed the EMA
			imu_src_period_ms = -1.0;
		}
		if (imu_proxy == nullptr)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(500));   // nothing to bridge from
			continue;
		}
		RoboCompIMU::DataImu data;
		try
		{
			data = imu_proxy->getDataImu();
		}
		catch (const Ice::Exception& e)
		{
			if (!error_logged)
			{
				qWarning() << "[read_imu] getDataImu failed:" << e.what() << "retrying...";
				error_logged = true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}
		if (error_logged)
		{
			std::print("[read_imu] IMU stream recovered.\n");
			error_logged = false;
		}

		// Self-synchronize with the source: advance only on a genuinely new sample (dedup by stamp) and
		// poll at ~2× the source period — derived from the SOURCE stamp deltas, not our own loop timing
		// (wall-clock between ingests is circular and death-spirals the rate down; the old fixed
		// Compute-period sleep also hard-capped it at ~10 Hz).
		const std::uint64_t stamp_ms = to_epoch_ms(data.acc.timestamp);
		if (stamp_ms != 0 && stamp_ms == last_imu_stamp_ms)
		{
			std::this_thread::sleep_for(wait_period);   // duplicate: wait one poll period, re-check
			continue;
		}
		if (stamp_ms != 0)
		{
			if (last_imu_stamp_ms != 0)
			{
				const double src_dt = static_cast<double>(stamp_ms - last_imu_stamp_ms);   // source period (ms)
				if (src_dt > 0.5 && src_dt < 2000.0)
				{
					imu_src_period_ms = (imu_src_period_ms < 0.0) ? src_dt : 0.8 * imu_src_period_ms + 0.2 * src_dt;
					wait_period = std::chrono::milliseconds(std::max<long>(1, static_cast<long>(0.5 * imu_src_period_ms + 0.5)));
				}
			}
			last_imu_stamp_ms = stamp_ms;
		}

		// The IMU is NOT mirrored into the graph. It used to be: six attribute writes per sample on one
		// shared node at ~125 Hz, CRDT-replicated to every agent and waking every peer's attribute slot
		// whether it wanted the IMU or not. It now rides the media plane only (rc/imu/data), which is
		// what that plane exists for — a dedicated DDS domain isolated from the DSR domain precisely so
		// high-rate churn cannot perturb cortex resync.
		// ★ And it is what the real robot does: the IMU component publishes to that topic directly, so a
		// consumer's code is now identical in simulation and on hardware.
		// The two fields the graph carried that a naive frame would drop — the SIM clock and the gyro
		// variance — are in ImuFrame, so nothing was lost with the attributes.

		// --- Media plane: publish the IMU sample (tiny fixed payload), stamped per-frame ---
		// bridge_imu_ false ⇒ an external component owns rc/imu/data (Media.imu_source = "dds").
		// Publishing anyway would put two producers on one topic, and a consumer cannot tell which
		// sample it got — the failure would look like an intermittently wrong IMU, not a duplicate.
		if (media_.imu_ready() and bridge_imu_.load(std::memory_order_relaxed))
		{
			SensorMediaPublisher::ImuFrameView v;
			// The acc substruct carries the freshest capture stamp; all substructs
			// share a clock, so use it as the frame timestamp.
			v.stamp_ms = to_epoch_ms(data.acc.timestamp);
			v.acc[0]  = data.acc.XAcc;  v.acc[1]  = data.acc.YAcc;  v.acc[2]  = data.acc.ZAcc;
			v.gyro[0] = data.gyro.XGyr; v.gyro[1] = data.gyro.YGyr; v.gyro[2] = data.gyro.ZGyr;
			v.mag[0]  = data.mag.XMag;  v.mag[1]  = data.mag.YMag;  v.mag[2]  = data.mag.ZMag;
			v.rpy[0]  = data.rot.Roll;  v.rpy[1]  = data.rot.Pitch; v.rpy[2]  = data.rot.Yaw;
			v.temperature = data.temperature;
			// The two fields a media-plane consumer cannot reconstruct and must not have to guess.
			// simTimestamp is 0 on a real IMU, which is exactly the "not simulated" signal the
			// consumer needs; the gyro covariance is diagonal and isotropic, so m22 is its variance.
			v.sim_stamp_ms = static_cast<std::uint64_t>(std::max<long>(0, data.gyro.simTimestamp));
			v.gyro_var     = data.gyro.cov.m22;
			// The accelerometer covariance is diagonal and isotropic like the gyro's; m00 is the
			// horizontal variance, which is the pair a consumer integrates for a velocity change.
			v.acc_var      = data.acc.cov.m00;
			media_.publish_imu(v);
		}
		media_.maybe_report_stats(SensorMediaPublisher::StatsGroup::Imu, std::chrono::seconds(5));

		imu_frames_.fetch_add(1, std::memory_order_relaxed);
		std::this_thread::sleep_for(wait_period);
	}
}


/////////////////////////////////////////////////////////////////////////
//SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
/////////////////////////////////////////////////////////////////////////
void SpecificWorker::FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)
{
	if (shutting_down_.load())
		return;

	// Count the ARRIVAL, before any graph work: this measures the producer's publication rate, so a
	// sample that finds no robot node still counts as received.
	fullpose_frames_.fetch_add(1, std::memory_order_relaxed);

	// we do not add any noise here. It is up to the users.
	if (auto pose_node = G->get_node(robot_name); pose_node.has_value())
	{
		G->add_or_modify_attrib_local<robot_current_advance_speed_att>(pose_node.value(), pose.adv);
		G->add_or_modify_attrib_local<robot_current_side_speed_att>(pose_node.value(), pose.side);
		G->add_or_modify_attrib_local<robot_current_angular_speed_att>(pose_node.value(), pose.rot);
		G->add_or_modify_attrib_local<robot_current_speed_timestamp_att>(pose_node.value(), static_cast<unsigned long>(pose.timestamp));
		// Carry the producer's simulation clock through as well. The velocities above are per
		// SIMULATION second when they come from a simulator, so a consumer integrating them -- a
		// high-rate propagation between optimized poses, say -- has to integrate over this clock, not
		// over the wall stamp, or it over-counts by however far the sim is running behind real time.
		// The wall stamp stays authoritative for latency and staleness. Passing the flag rather than a
		// config setting means each consumer reads the answer off the sample itself.
		G->add_or_modify_attrib_local<robot_current_speed_sim_timestamp_att>(pose_node.value(), static_cast<unsigned long>(pose.simTimestamp));
		G->add_or_modify_attrib_local<robot_current_speed_simulated_att>(pose_node.value(), pose.simulated);

		// ── Per-sample velocity VARIANCE, as the producer states it ───────────────────────────────
		// FullPoseEuler has always carried this in velCov and this function used to DROP it, which is
		// why every consumer downstream falls back on an asserted constant. Forwarding it is what lets
		// a preintegrator weight a segment by the noise the sensor actually has rather than by a config
		// value that may describe a different robot — and it is the only channel through which a FASTER
		// odometry stream can tighten anything at all, because a noise DENSITY is invariant to sample
		// rate: only a smaller per-sample variance, or a shorter dt_sample, moves it.
		//
		// ★ FRAME, AND IT IS THE TRAP HERE. velCov is a BODY-frame quantity, and it could not be
		// anything else: it is written by the base driver, a component that has no notion of a room or
		// a world at all. It cannot express a covariance in a frame it does not know exists. (The same
		// conclusion from the physics: a world-frame velocity covariance would rotate with the robot and
		// fold the heading's own uncertainty into what is meant to be a property of the SENSOR.)
		// So its linear block is indexed on the BODY axes (m00 = x, m11 = y, m55 = yaw rate).
		// ⚠ The `// vx` / `// vy` labels beside the writes in webots-bridge are cosmetic and prove
		// nothing either way — it stores one nominal value into both slots, so they never disagree.
		//
		// ★★ AND THE INDICES CROSS OVER, because this robot's forward axis is +Y:
		//        adv  = forward = body +Y  ⇒  var_adv  = m11
		//        side = lateral = body +X  ⇒  var_side = m00
		// Verified against the producer, which builds the same pair the same way (webots-bridge:
		// `adv = velocity_local(1)` i.e. y, `side = -velocity_local(0)` i.e. -x; the sign is irrelevant
		// to a variance). Writing m00 into adv would silently hand the forward channel the lateral
		// channel's noise — which on a mecanum, where roller slip makes lateral much the worse of the
		// two, is precisely the swap that would go unnoticed while quietly mis-weighting every segment.
		//
		// A negative entry means "this producer does not know" for that channel, matching the .idsl's
		// m00 = -1 convention and ImuFrame's gyro_var/acc_var. Each channel is independent: a producer
		// may know its yaw rate noise and not its linear noise. When NOTHING is known the attribute is
		// not written at all, so "never published" stays distinguishable from "published as unknown" —
		// a zero would read downstream as INFINITE CONFIDENCE, which is the trap the .idsl warns about.
		{
			const auto stated = [](float v) { return v >= 0.f ? v : -1.f; };   // NaN also reads as unknown
			const float var_adv  = stated(pose.velCov.m11);   // forward, body +Y
			const float var_side = stated(pose.velCov.m00);   // lateral, body +X
			const float var_rot  = stated(pose.velCov.m55);   // yaw rate, frame-independent
			if (var_adv >= 0.f or var_side >= 0.f or var_rot >= 0.f)
			{
				const std::vector<float> vel_var{ var_adv, var_side, var_rot };
				G->add_or_modify_attrib_local<robot_current_speed_variance_att>(pose_node.value(), vel_var);
			}
		}

		// ── GROUND TRUTH, SIMULATION ONLY ────────────────────────────────────────────────────────
		// pose.x/y/rz come straight off the Webots supervisor node (robotNode->getPosition() /
		// getOrientation() in the bridge), so in simulation they ARE the truth. Everything above is
		// odometry; this is the only thing on this message that no estimator produced. Published so
		// a localiser can be graded against a witness OUTSIDE itself: a wrong pose fitted well
		// scores exactly like a right one on the residual, which is how a 0.35 rad yaw error sat
		// behind an SDF of 0.009 for a whole session (2026-08-22).
		//
		// The GATE IS THE DATA, not a config key: written only while the producer says simulated.
		// On the real robot these attributes are never written at all, so a consumer reads nothing
		// rather than something plausible and stale — and there is no flag anyone can forget to
		// flip. ⚠ For validation only. Feeding this back into an estimator would make every
		// accuracy number self-fulfilling.
		if (pose.simulated)
		{
			G->add_or_modify_attrib_local<robot_gt_x_att>(pose_node.value(), pose.x);
			G->add_or_modify_attrib_local<robot_gt_y_att>(pose_node.value(), pose.y);
			G->add_or_modify_attrib_local<robot_gt_angle_att>(pose_node.value(), pose.rz);
			G->add_or_modify_attrib_local<robot_gt_timestamp_att>(pose_node.value(),
			                                                     static_cast<std::uint64_t>(pose.timestamp));
		}
		// pose_node is not read after this; move it so update_node forwards
		// the rvalue into the engine (no deep blob copy under the graph mutex).
		rc::safe_update_node(*G, std::move(pose_node.value()));
	}
	else if (!shutting_down_.load())
		qWarning() << "FullPose node not found in DSR graph";
}
