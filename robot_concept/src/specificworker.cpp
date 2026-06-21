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

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/media_transport/media_transport.h"

#include <ConfigLoader/ConfigLoader.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <print>

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

	if (imu_thread.joinable())
		imu_thread.join();
	if (lidar_thread.joinable())
		lidar_thread.join();
	if (rgbd_thread.joinable())
		rgbd_thread.join();

	// Tear down DDS endpoints after producer threads are fully stopped.
	media_publishers_ready_ = false;
	media_rgb_pub_.reset();
	media_depth_pub_.reset();

	cleanup_owned_nodes();
}


void SpecificWorker::initialize()
{
	qInfo() << "initialize robot_concept worker";
	GenericWorker::initialize();

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

	qInfo() << "[DSR Upload Rates] rgb=" << params.DSR_RGB_FPS
	        << "depth=" << params.DSR_DEPTH_FPS
	        << "lidar=" << params.DSR_LIDAR_FPS
	        << "(Hz; 0=every frame, <0=disabled)";

	// Seed body node dimensions (read from graph/shadow.json; write defaults if absent).
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

		G->update_node(body_node.value());
		if (seeded)
			qInfo() << __FUNCTION__ << "Seeded body node with default dimensions (width=0.47, depth=0.47, height=1.6)";
		else
			qInfo() << __FUNCTION__ << "Body node dimensions already present in graph";
	}
	else
		qWarning() << __FUNCTION__ << "Body node not found in DSR graph";

	// ── One-shot dump of the static mount edges (both directions) ─────────────
	// Prints Shadow->room and body->kinova_arm_r once at startup so they can be
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
		dump("room", "Shadow");
		dump("kinova_arm_r", "body");
	}

	// Media plane: initialize zero-copy DDS publishers for RGB and depth pixels.
	// These carry the heavy RGBD payload OUT of the DSR/CRDT graph (shared-memory,
	// RELIABLE, data-sharing). The DSR cam_rgb/cam_depth blob writes below remain
	// governed by DSR_RGB_FPS/DSR_DEPTH_FPS (<0 disables them once consumers migrate).
	{
		media_rgb_pub_   = std::make_unique<rc::media::MediaPublisher>();
		media_depth_pub_ = std::make_unique<rc::media::MediaPublisher>();
		rc::media::PublisherConfig rgb_cfg;
		rgb_cfg.domain_id     = static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID);
		rgb_cfg.topic_name    = params.MEDIA_RGB_TOPIC;
		rgb_cfg.history_depth = 8;
		rc::media::PublisherConfig depth_cfg = rgb_cfg;
		depth_cfg.topic_name = params.MEDIA_DEPTH_TOPIC;

		const bool rgb_ok   = media_rgb_pub_->init(rgb_cfg);
		const bool depth_ok = media_depth_pub_->init(depth_cfg);
		media_publishers_ready_ = rgb_ok && depth_ok;
		if (media_publishers_ready_)
			qInfo() << "[Media] publishers ready domain=" << params.MEDIA_DOMAIN_ID
			        << "rgb=" << QString::fromStdString(params.MEDIA_RGB_TOPIC)
			        << "depth=" << QString::fromStdString(params.MEDIA_DEPTH_TOPIC)
			        << "data_sharing=" << (media_rgb_pub_->data_sharing_active() && media_depth_pub_->data_sharing_active());
		else
			qWarning() << "[Media] publisher init FAILED (rgb=" << rgb_ok << "depth=" << depth_ok
			           << ") - RGBD will only flow through the DSR graph";

		// Advertise the media plane in the graph so ANY agent can discover and subscribe
		// without hardcoding topic/domain/QoS (rc::media::MediaDescriptor). Written on the
		// camera node ("zed") that already carries cam_rgb_*; consumers read it generically
		// via rc::media::descriptor_from_graph(G, "zed"). The heavy frames stay out-of-band
		// on the zero-copy plane — only this small JSON string lives in the graph.
		if (auto cam_node = G->get_node("zed"); cam_node.has_value())
		{
			rc::media::MediaDescriptor desc;
			desc.domain_id          = static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID);
			desc.history_depth      = rgb_cfg.history_depth;       // same QoS the publishers use
			desc.shared_memory_only = rgb_cfg.shared_memory_only;
			desc.data_sharing       = rgb_cfg.data_sharing;
			desc.ready              = media_publishers_ready_;     // false ⇒ consumers should wait
			desc.streams["rgb"]     = params.MEDIA_RGB_TOPIC;
			desc.streams["depth"]   = params.MEDIA_DEPTH_TOPIC;
			G->runtime_checked_add_or_modify_attrib_local(cam_node.value(),
			                                              rc::media::MEDIA_DESCRIPTOR_ATTR, desc.to_json());
			G->update_node(cam_node.value());
			qInfo() << "[Media] descriptor advertised on 'zed':" << QString::fromStdString(desc.to_json());
		}
		else
			qWarning() << "[Media] 'zed' node not found at init — media descriptor NOT advertised";
	}

	lidar_thread = std::thread(&SpecificWorker::read_lidar_thread,  this);
	rgbd_thread  = std::thread(&SpecificWorker::read_rgbd_thread,   this);
	qInfo() << __FUNCTION__ << "Started lidar and RGBD reader threads";

	presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
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
		.on_operating_enter = []()
		{
			qInfo() << "[SM] -> Operating: all required constraints satisfied";
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
}

void SpecificWorker::compute()
{
	// robot_concept's compute() is intentionally minimal:
	// sensor reading is done in background threads;
	static FPSCounter compute_fps;
	compute_fps.print("Compute (void)", 5000);
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
	static FPSCounter lidar_fps;
	bool empty_lidar_logged = false;
	auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
	const bool lidar_upload_enabled = params.DSR_LIDAR_FPS >= 0;
	const auto lidar_interval_ms = params.DSR_LIDAR_FPS > 0
		? std::chrono::milliseconds(1000 / params.DSR_LIDAR_FPS)
		: std::chrono::milliseconds(0);
	auto last_lidar_upload = std::chrono::steady_clock::time_point{};
	while (!stop_lidar_thread && !shutting_down_.load())
	{
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
			if (auto laser_node = G->get_node("lidar3D"); laser_node.has_value())
			{
				G->add_or_modify_attrib_local<laser_X_att>(laser_node.value(), std::move(xs));
				G->add_or_modify_attrib_local<laser_Y_att>(laser_node.value(), std::move(ys));
				G->add_or_modify_attrib_local<laser_Z_att>(laser_node.value(), std::move(zs));
				G->add_or_modify_attrib_local<laser_timestamp_att>(laser_node.value(), static_cast<uint64_t>(data.timestamp));
				// Last use of laser_node: move it in so the (multi-MB) laser blobs are
				// moved into the engine instead of deep-copied under the graph write lock.
				G->update_node(std::move(laser_node.value()));
			}
			else if (!shutting_down_.load())
				qWarning() << "Laser node not found in DSR graph";
		}

		const long p_ms = static_cast<long>(data.period);
		if (wait_period > std::chrono::milliseconds(p_ms + 2)) --wait_period;
		else if (wait_period < std::chrono::milliseconds(p_ms - 2)) ++wait_period;

		if (verbose_debug_)
			lidar_fps.print("[LidarThread]", 2000);
		std::this_thread::sleep_for(wait_period);
	}
}

void SpecificWorker::read_rgbd_thread()
{
	static FPSCounter rgbd_fps;
	struct MediaStats
	{
		std::uint64_t rgb_ok = 0;
		std::uint64_t depth_ok = 0;
		std::uint64_t rgb_bytes = 0;
		std::uint64_t depth_bytes = 0;
		std::uint64_t rgb_loan_fail = 0;
		std::uint64_t depth_loan_fail = 0;
		std::uint64_t rgb_publish_fail = 0;
		std::uint64_t depth_publish_fail = 0;
		auto reset_window() -> void
		{
			rgb_ok = depth_ok = rgb_bytes = depth_bytes = 0;
			rgb_loan_fail = depth_loan_fail = 0;
			rgb_publish_fail = depth_publish_fail = 0;
		}
	};
	MediaStats media_stats;
	auto last_media_report = std::chrono::steady_clock::now();
	bool empty_rgbd_logged = false;
	auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
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
	while (!stop_rgbd_thread && !shutting_down_.load())
	{
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

			// --- Media plane: publish full-rate RGB + depth pixels via zero-copy DDS ---
			// This carries the heavy payload OUT of the DSR graph. We copy into the loaned
			// SHM slot here, leaving the source buffers intact for the (optional) DSR writes
			// below, which can still std::move them.
			if (media_publishers_ready_)
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
				if (const std::size_t nbytes = img.image.size();
					nbytes > 0 && nbytes <= rc::media::MAX_IMAGE_BYTES && media_rgb_pub_)
				{
					if (auto *s = media_rgb_pub_->loan())
					{
						s->stream_id(rc::media::STREAM_ZED_RGB);
						s->frame_id(media_rgb_frame_id_++);
						s->stamp_ms(to_epoch_ms(img.alivetime));
						s->width(static_cast<std::uint32_t>(img.width));
						s->height(static_cast<std::uint32_t>(img.height));
						s->step(static_cast<std::uint32_t>(img.width) * 3u);
						s->format(rc::media::FORMAT_BGR8);
						s->size(static_cast<std::uint32_t>(nbytes));
						std::memcpy(s->data().data(), img.image.data(), nbytes);
						if (media_rgb_pub_->publish(s))
						{
							++media_stats.rgb_ok;
							media_stats.rgb_bytes += nbytes;
						}
						else
							++media_stats.rgb_publish_fail;
					}
					else
						++media_stats.rgb_loan_fail;
				}

				const auto &dep = frame.depth;
				const std::size_t dep_bytes = dep.depth.size();
				const std::size_t dep_pix = static_cast<std::size_t>(dep.width) * static_cast<std::size_t>(dep.height);
				if (dep_bytes > 0 && dep_pix > 0 && dep_bytes <= rc::media::MAX_IMAGE_BYTES && media_depth_pub_)
				{
					const std::size_t bpp = dep_bytes / dep_pix;
					std::uint32_t fmt  = rc::media::FORMAT_Z16;
					std::uint32_t step = static_cast<std::uint32_t>(dep.width) * 2u;
					if (bpp == 4) { fmt = rc::media::FORMAT_DEPTH_F32; step = static_cast<std::uint32_t>(dep.width) * 4u; }
					if (auto *s = media_depth_pub_->loan())
					{
						s->stream_id(rc::media::STREAM_ZED_DEPTH);
						s->frame_id(media_depth_frame_id_++);
						s->stamp_ms(to_epoch_ms(dep.alivetime));
						s->width(static_cast<std::uint32_t>(dep.width));
						s->height(static_cast<std::uint32_t>(dep.height));
						s->step(step);
						s->format(fmt);
						s->size(static_cast<std::uint32_t>(dep_bytes));
						std::memcpy(s->data().data(), dep.depth.data(), dep_bytes);
						if (media_depth_pub_->publish(s))
						{
							++media_stats.depth_ok;
							media_stats.depth_bytes += dep_bytes;
						}
						else
							++media_stats.depth_publish_fail;
					}
					else
						++media_stats.depth_loan_fail;
				}
			}

			const auto now_report = std::chrono::steady_clock::now();
			if (now_report - last_media_report >= std::chrono::seconds(5))
			{
				// Always report ok-counts so we can positively confirm the producer is
				// publishing (not just see failures). Empty points no longer gate this.
				qInfo() << "[Media] 5s stats"
				        << "rgb_ok=" << media_stats.rgb_ok
				        << "depth_ok=" << media_stats.depth_ok
				        << "rgb_MB=" << (static_cast<double>(media_stats.rgb_bytes) / 1e6)
				        << "depth_MB=" << (static_cast<double>(media_stats.depth_bytes) / 1e6)
				        << "rgb_loan_fail=" << media_stats.rgb_loan_fail
				        << "depth_loan_fail=" << media_stats.depth_loan_fail
				        << "rgb_publish_fail=" << media_stats.rgb_publish_fail
				        << "depth_publish_fail=" << media_stats.depth_publish_fail;
				media_stats.reset_window();
				last_media_report = now_report;
			}

			const long p_ms = static_cast<long>(frame.image.period);

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
					G->update_node(cam_node.value());
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
					G->update_node(std::move(cam_node.value()));
				}
			}
			else if (!shutting_down_.load())
				qWarning() << "Camera node not found in DSR graph";

			if (p_ms > 0)
			{
				if (wait_period > std::chrono::milliseconds(p_ms + 2)) --wait_period;
				else if (wait_period < std::chrono::milliseconds(p_ms - 2)) ++wait_period;
			}

			if (verbose_debug_)
				rgbd_fps.print("[RGBDThread]", 2000);
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


/////////////////////////////////////////////////////////////////////////
//SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
/////////////////////////////////////////////////////////////////////////
void SpecificWorker::FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)
{
	if (shutting_down_.load())
		return;

	// we do not add any noise here. It is up to the users.
	if (auto pose_node = G->get_node(robot_name); pose_node.has_value())
	{
		G->add_or_modify_attrib_local<robot_current_advance_speed_att>(pose_node.value(), pose.adv);
		G->add_or_modify_attrib_local<robot_current_side_speed_att>(pose_node.value(), pose.side);
		G->add_or_modify_attrib_local<robot_current_angular_speed_att>(pose_node.value(), pose.rot);
		G->add_or_modify_attrib_local<robot_current_speed_timestamp_att>(pose_node.value(), static_cast<unsigned long>(pose.timestamp));
		G->update_node(pose_node.value());
	}
	else if (!shutting_down_.load())
		qWarning() << "FullPose node not found in DSR graph";
}
