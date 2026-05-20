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
#ifdef emit
#undef emit
#endif
#include "unified_voxel_grid.h"
#include "voxel_opengl_viewer.h"
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <dsr/api/dsr_camera_api.h>
#include <print>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <execution>
#include <numeric>
#include <iterator>
#include <unordered_map>
#include <Eigen/Geometry>
#include "custom_widget.h"
#include "ui_localUI.h"

namespace
{
	struct RoomToCameraBasis
	{
		Mat::Vector3d origin{0.0, 0.0, 0.0};
		Mat::Vector3d axis_x{0.0, 0.0, 0.0};
		Mat::Vector3d axis_y{0.0, 0.0, 0.0};
		Mat::Vector3d axis_z{0.0, 0.0, 0.0};
	};

	float axis_overlap(float amin, float amax, float bmin, float bmax)
	{
		return std::max(0.0f, std::min(amax, bmax) - std::max(amin, bmin));
	}

	float box_volume(const TrackBoxCandidate& box)
	{
		const Eigen::Vector3f ext = (box.max - box.min).cwiseMax(Eigen::Vector3f::Zero());
		return ext.x() * ext.y() * ext.z();
	}

	float intersection_volume(const TrackBoxCandidate& a, const TrackBoxCandidate& b)
	{
		const float ox = axis_overlap(a.min.x(), a.max.x(), b.min.x(), b.max.x());
		const float oy = axis_overlap(a.min.y(), a.max.y(), b.min.y(), b.max.y());
		const float oz = axis_overlap(a.min.z(), a.max.z(), b.min.z(), b.max.z());
		return ox * oy * oz;
	}

	bool boxes_look_duplicate(const TrackBoxCandidate& a, const TrackBoxCandidate& b)
	{
		if (a.category != b.category)
			return false;

		const float inter = intersection_volume(a, b);
		if (inter <= 1e-5f)
			return false;

		const float a_vol = box_volume(a);
		const float b_vol = box_volume(b);
		const float smaller_vol = std::max(1e-5f, std::min(a_vol, b_vol));
		const float overlap_ratio = inter / smaller_vol;
		const float centroid_distance = (a.centroid - b.centroid).norm();
		const float smaller_diag = std::min((a.max - a.min).norm(), (b.max - b.min).norm());

		return overlap_ratio >= 0.30f
		    && centroid_distance <= std::max(0.55f, 0.85f * std::max(smaller_diag, 1e-3f));
	}

	int max_instances_for_category(const std::string& category)
	{
		if (category == "table")
			return 1;
		if (category == "monitor" || category == "blackboard")
			return 2;
		return std::numeric_limits<int>::max();
	}

	bool prefer_candidate(const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs, int frame_id)
	{
		const bool lhs_seen_now = lhs.last_seen_frame == frame_id;
		const bool rhs_seen_now = rhs.last_seen_frame == frame_id;
		if (lhs_seen_now != rhs_seen_now)
			return lhs_seen_now;

		if (lhs.voxel_count != rhs.voxel_count)
			return lhs.voxel_count > rhs.voxel_count;

		return lhs.track_id < rhs.track_id;
	}

	std::uint64_t get_imu_timestamp_ms(const RoboCompIMU::DataImu& data)
	{
		const long latest = std::max({data.acc.timestamp,
		                             data.gyro.timestamp,
		                             data.mag.timestamp,
		                             data.rot.timestamp});
		return latest > 0 ? static_cast<std::uint64_t>(latest) : 0ULL;
	}

	bool compute_room_to_camera_basis(DSR::InnerEigenAPI* inner_eigen_api,
	                                 const std::string& camera_node_name,
	                                 const std::string& room_frame_name,
	                                 std::uint64_t rt_timestamp,
	                                 bool interpolate_rt,
	                                 RoomToCameraBasis& basis)
	{
		if (inner_eigen_api == nullptr)
			return false;

		const auto time_query = interpolate_rt
			? DSR::RT_API::TimeQuery::Interpolated
			: DSR::RT_API::TimeQuery::Nearest;

		const auto origin_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
		const auto x_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(1.0, 0.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
		const auto y_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 1.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
		const auto z_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 1.0), room_frame_name, rt_timestamp, "RT", time_query);
		if (!origin_opt.has_value() || !x_opt.has_value() || !y_opt.has_value() || !z_opt.has_value())
			return false;

		basis.origin = origin_opt.value();
		basis.axis_x = x_opt.value() - basis.origin;
		basis.axis_y = y_opt.value() - basis.origin;
		basis.axis_z = z_opt.value() - basis.origin;
		return true;
	}
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
	qInfo() << "Destroying SpecificWorker";
	stop_imu_thread = true;
	stop_lidar_thread = true;
	stop_rgbd_thread = true;
	if (imu_thread.joinable())
		imu_thread.join();
	if (lidar_thread.joinable())
		lidar_thread.join();
	if (rgbd_thread.joinable())
		rgbd_thread.join();
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

    // ── Initialise YOLO-seg detector ─────────────────────────
    try
    {
        params.YOLO_MODEL_PATH  = configLoader.get<std::string>("Yolo.model_path");
    }
    catch (...) { /* key absent — use default */ }
	try { params.YOLO_ACCEPTED_LABELS = configLoader.get<std::vector<std::string>>("Yolo.accepted_labels"); } catch (...) {}
	try { params.YOLO_CONF_THRESH = static_cast<float>(configLoader.get<double>("Yolo.conf_thresh")); } catch (...) {}
	try { params.YOLO_IOU_THRESH  = static_cast<float>(configLoader.get<double>("Yolo.iou_thresh"));  } catch (...) {}
	try { params.YOLO_USE_GPU     = configLoader.get<bool>("Yolo.use_gpu");  } catch (...) {}
	try { params.YOLO_USE_TRT     = configLoader.get<bool>("Yolo.use_trt");  } catch (...) {}
	try { params.YOLO_MASK_ERODE_KERNEL = configLoader.get<int>("Yolo.mask_erode_kernel"); } catch (...) {}
	try { params.TRACK_ASSOCIATION_MAX_DISTANCE_M = static_cast<float>(configLoader.get<double>("Yolo.track_association_max_distance_m")); } catch (...) {}
	try { params.TRACK_MAX_MISSED_FRAMES = configLoader.get<int>("Yolo.track_max_missed_frames"); } catch (...) {}
	try { params.DSR_RGB_FPS   = configLoader.get<int>("Camera.dsr_rgb_fps");   } catch (...) {}
	try { params.DSR_DEPTH_FPS = configLoader.get<int>("Camera.dsr_depth_fps"); } catch (...) {}
	try { params.DSR_LIDAR_FPS = configLoader.get<int>("Lidar.dsr_lidar_fps"); } catch (...) {}
	try { params.TRANSFORMS_INTERPOLATE_RT = configLoader.get<bool>("Transforms.interpolate_rt"); } catch (...) {}
	for (auto& label : params.YOLO_ACCEPTED_LABELS)
		label = normalize_yolo_label(label);
	std::sort(params.YOLO_ACCEPTED_LABELS.begin(), params.YOLO_ACCEPTED_LABELS.end());
	params.YOLO_ACCEPTED_LABELS.erase(std::unique(params.YOLO_ACCEPTED_LABELS.begin(), params.YOLO_ACCEPTED_LABELS.end()),
	                                  params.YOLO_ACCEPTED_LABELS.end());
	try { verbose_debug_ = configLoader.get<bool>("Debug.verbose"); } catch (...) { verbose_debug_ = false; }
	if (verbose_debug_)
		std::println("[YOLO] effective flags: use_gpu={} use_trt={}", params.YOLO_USE_GPU, params.YOLO_USE_TRT);

    yolo_detector.emplace(params.YOLO_MODEL_PATH,
                          std::vector<std::string>{},   // default COCO names
                          params.YOLO_CONF_THRESH,
                          params.YOLO_IOU_THRESH,
                          params.YOLO_INPUT_SIZE,
                          params.YOLO_USE_GPU,
                          params.YOLO_USE_TRT);
	if (verbose_debug_)
		std::println("YOLO-seg detector ready: {}", params.YOLO_MODEL_PATH);

	//Subscription to DSR graph update signals. 
	// If multiple graphs exist, it is necessary to specify the graph name 
	// using 'Graphs.at("name")' to connect its signals to the Worker's slots.
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
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
	
	if (!graph_viewers.empty())
	{
		const std::string viewer_key = graph_viewers.contains("")
			? std::string("")
			: graph_viewers.begin()->first;
		graph_viewers.at(viewer_key)->add_custom_widget_to_dock("Voxel3D", &custom_widget);
		graph_viewers.at(viewer_key)->add_custom_widget_to_dock("ZED+YOLO", &custom_widget_yolo);

		if (custom_widget.frame->layout() == nullptr)
		{
			auto* layout = new QVBoxLayout(custom_widget.frame);
			layout->setContentsMargins(0, 0, 0, 0);
			custom_widget.frame->setLayout(layout);
		}

		voxel_lidar_toggle_button_ = new QPushButton("Hide LiDAR", custom_widget.frame);
		voxel_lidar_toggle_button_->setCheckable(true);
		voxel_lidar_toggle_button_->setChecked(false);
		voxel_lidar_toggle_button_->setText("Show LiDAR");
		custom_widget.frame->layout()->addWidget(voxel_lidar_toggle_button_);

		voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(custom_widget.frame);
		voxel_viewer_gl->set_show_lidar(false);
		std::string robot_mesh_path = "meshes/shadow.obj";
		if (auto robot_node = G->get_node("Shadow"); robot_node.has_value())
			if (auto mesh_path = G->get_attrib_by_name<path_att>(robot_node.value()); mesh_path.has_value() && !mesh_path.value().get().empty())
				robot_mesh_path = mesh_path.value().get();
		voxel_viewer_gl->load_robot_mesh(robot_mesh_path);
		custom_widget.frame->layout()->addWidget(voxel_viewer_gl.get());
		QObject::connect(voxel_lidar_toggle_button_, &QPushButton::toggled, custom_widget.frame,
		                 [this](bool checked)
		                 {
			                 if (voxel_viewer_gl)
				                 voxel_viewer_gl->set_show_lidar(checked);
			                 if (voxel_lidar_toggle_button_)
				                 voxel_lidar_toggle_button_->setText(checked ? "Hide LiDAR" : "Show LiDAR");
		                 });
		qInfo() << __FUNCTION__ << "Voxel OpenGL custom widget attached to graph viewer";

		if (custom_widget_yolo.frame->layout() == nullptr)
		{
			auto* layout = new QVBoxLayout(custom_widget_yolo.frame);
			layout->setContentsMargins(0, 0, 0, 0);
			layout->setSpacing(4);
			custom_widget_yolo.frame->setLayout(layout);
		}

		yolo_fps_label_ = new QLabel(custom_widget_yolo.frame);
		yolo_fps_label_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
		yolo_fps_label_->setText("YOLO display FPS: --");
		custom_widget_yolo.frame->layout()->addWidget(yolo_fps_label_);

		yolo_image_label_ = new QLabel(custom_widget_yolo.frame);
		yolo_image_label_->setMinimumSize(320, 240);
		yolo_image_label_->setAlignment(Qt::AlignCenter);
		yolo_image_label_->setScaledContents(false);
		yolo_image_label_->setText("ZED RGB + YOLO overlay");
		custom_widget_yolo.frame->layout()->addWidget(yolo_image_label_);
		qInfo() << __FUNCTION__ << "ZED+YOLO custom widget attached to graph viewer";
	}
	else
		qWarning() << __FUNCTION__ << "No graph viewer available; Voxel3D widget not attached";

	// Allocate here so the heavy header remains out of specificworker.h and MOC units.
	voxel_grid = std::make_unique<UnifiedVoxelGrid>();
	inner_eigen_api = G->get_inner_eigen_api();
	if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
	{
		std::scoped_lock lk(node_names_mutex_);
		room_node_name_ = room_nodes.front().name();
		if (verbose_debug_)
			std::println("[VoxelInit] Found room node: '{}'", room_node_name_);
	}
	else
	{
		if (verbose_debug_)
			std::println("[VoxelInit] No room nodes found in graph");
	}
	
	if (const auto robot_nodes = G->get_nodes_by_type("robot"); !robot_nodes.empty())
	{
		std::scoped_lock lk(node_names_mutex_);
		robot_node_name_ = robot_nodes.front().name();
	}
	std::string room_name_snapshot, robot_name_snapshot;
	{
		std::scoped_lock lk(node_names_mutex_);
		room_name_snapshot = room_node_name_;
		robot_name_snapshot = robot_node_name_;
	}
	if (!room_name_snapshot.empty() && !robot_name_snapshot.empty())
	{
		if (verbose_debug_)
			std::println("[VoxelInit] Cached room='{}' robot='{}' from initial graph", room_name_snapshot, robot_name_snapshot);
	}

	// ── Start IMU reader thread ──────────────────────────────
	imu_thread = std::thread(&SpecificWorker::read_imu_thread, this);
	qInfo() << __FUNCTION__ << "Started IMU reader";

	 // ── Start lidar reader thread ────────────────────────────
    lidar_thread = std::thread(&SpecificWorker::read_lidar_thread, this);
    qInfo() << __FUNCTION__ << "Started lidar reader";

    // ── Start RGBD reader thread ─────────────────────────────
    rgbd_thread = std::thread(&SpecificWorker::read_rgbd_thread, this);
    qInfo() << __FUNCTION__ << "Started RGBD reader";

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}



void SpecificWorker::compute()
{
	static FPSCounter compute_fps;
	check_input_stream_startup_status();

	// ── 1. Per-cycle node name snapshot ──────────────────────────────────────
	const auto [room_name, robot_name] = get_room_robot_names_for_compute();

	// ── 2. Acquire latest RGBD frame ─────────────────────────────────────────
	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	const auto &[rgbd_opt] = rgbd_buffer.read(now_ms);
	if (!rgbd_opt.has_value())
		return;
	const auto& rgbd = rgbd_opt.value();
	const std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(rgbd);

	// ── 3. YOLO segmentation ──────────────────────────────────────────────────
	const auto detections = detect_segmentation(rgbd);

	// ── 4. Update ZED+YOLO overlay tab (only when tab is visible) ────────────
	update_yolo_tab_display(rgbd, detections);

	// ── 5. Guard: room and robot nodes must exist in the graph ────────────────
	if (!ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
		return;

	// ── 6. Fetch room-frame transforms at frame timestamp ────────────────────
	const auto room_T_robot = get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
	if (!room_T_robot.has_value())
		return;
	const auto room_T_zed = get_room_zed_transform(compute_fps, room_name, frame_ts_ms);
	if (!room_T_zed.has_value())
		return;
	log_room_robot_pose_periodic(room_T_robot.value());
	if (!room_rt_ready_logged_) { room_rt_ready_logged_ = true; room_rt_wait_logged_ = false; }

	// ── 7. Push robot pose to 3-D viewer ─────────────────────────────────────
	update_viewer_robot_pose(room_T_robot.value());
	update_viewer_lidar_points(room_name, robot_name, room_T_robot.value());

	// ── 8. Refresh room polygon overlay ──────────────────────────────────────
	update_room_polygon_periodic();

	// ── 9. Update semantic voxel map ─────────────────────────────────────────
	track_association_max_distance_m = params.TRACK_ASSOCIATION_MAX_DISTANCE_M;
	track_max_missed_frames = params.TRACK_MAX_MISSED_FRAMES;
	update_voxel_grid_from_rgbd(rgbd, detections, room_T_robot.value(), room_T_zed.value());

	if (verbose_debug_)
		compute_fps.print("[Compute]", 2000);
}

//////////////////////////////////////////////////////////////////////////////

std::pair<std::string, std::string> SpecificWorker::get_room_robot_names_for_compute()
{
	std::string room_name_snapshot;
	std::string robot_name_snapshot;
	{
		std::scoped_lock lk(node_names_mutex_);
		room_name_snapshot = room_node_name_;
		robot_name_snapshot = robot_node_name_;
	}

	if (room_name_snapshot.empty())
	{
		if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
			room_name_snapshot = room_nodes.front().name();
	}

	if (robot_name_snapshot.empty())
	{
		if (const auto robot_nodes = G->get_nodes_by_type("robot"); !robot_nodes.empty())
			robot_name_snapshot = robot_nodes.front().name();
	}

	{
		std::scoped_lock lk(node_names_mutex_);
		if (room_node_name_.empty() && !room_name_snapshot.empty())
			room_node_name_ = room_name_snapshot;
		if (robot_node_name_.empty() && !robot_name_snapshot.empty())
			robot_node_name_ = robot_name_snapshot;
	}

	return {room_name_snapshot, robot_name_snapshot};
}

bool SpecificWorker::ensure_room_and_robot_ready(FPSCounter& compute_fps,
	                                              const std::string& room_name,
	                                              const std::string& robot_name)
{
	if (room_name.empty())
	{
		if (!room_wait_logged_)
		{
			qWarning() << "Room node not found in DSR graph. Voxelization paused until a room exists.";
			room_wait_logged_ = true;
			room_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return false;
	}

	if (robot_name.empty())
	{
		if (!room_wait_logged_)
		{
			qWarning() << "Robot node not found in DSR graph. Voxelization paused until a robot exists.";
			room_wait_logged_ = true;
			room_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return false;
	}

	if (!room_ready_logged_)
	{
		qInfo() << "Room node found in DSR graph. Voxelization enabled.";
		room_ready_logged_ = true;
	}
	return true;
}

std::optional<Mat::RTMat> SpecificWorker::get_room_robot_transform(FPSCounter& compute_fps,
	                                                               const std::string& room_name,
	                                                               const std::string& robot_name,
	                                                               std::uint64_t timestamp_ms)
{
	if (!inner_eigen_api)
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "InnerEigen API is not available. Voxelization paused.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	auto room_T_robot = inner_eigen_api->get_transformation_matrix(room_name, robot_name, timestamp_ms);
	if (!room_T_robot.has_value())
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "robot->room RTMat not available in InnerEigen API. Voxelization paused until transform is available."
			           << "room=" << QString::fromStdString(room_name)
			           << "robot=" << QString::fromStdString(robot_name)
			           << "ts_ms=" << timestamp_ms;
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	return room_T_robot;
}

std::optional<Mat::RTMat> SpecificWorker::get_room_zed_transform(FPSCounter& compute_fps,
	                                                             const std::string& room_name,
	                                                             std::uint64_t timestamp_ms)
{
	if (!inner_eigen_api)
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "InnerEigen API is not available. Voxelization paused.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	auto room_T_zed = inner_eigen_api->get_transformation_matrix(room_name, "zed", timestamp_ms);
	if (!room_T_zed.has_value())
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "zed->room RTMat not available in InnerEigen API. Voxelization paused until transform is available."
			           << "room=" << QString::fromStdString(room_name)
			           << "zed=zed"
			           << "ts_ms=" << timestamp_ms;
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		if (verbose_debug_)
			compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	return room_T_zed;
}

std::uint64_t SpecificWorker::get_rgbd_frame_timestamp_ms(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
{
	if (rgbd.image.alivetime > 0)
		return static_cast<std::uint64_t>(rgbd.image.alivetime);
	if (rgbd.depth.alivetime > 0)
		return static_cast<std::uint64_t>(rgbd.depth.alivetime);
	if (rgbd.points.alivetime > 0)
		return static_cast<std::uint64_t>(rgbd.points.alivetime);
	return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count());
}

void SpecificWorker::check_input_stream_startup_status()
{
	constexpr auto startup_grace = std::chrono::seconds(3);
	const auto now = std::chrono::steady_clock::now();
	if (now - input_stream_watchdog_start_ < startup_grace)
		return;

	if (!rgbd_stream_seen_.load(std::memory_order_relaxed)
		&& !rgbd_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
	{
		std::print(stderr, "[read_rgbd] No RGBD frames received since startup. Waiting for input stream...\n");
	}

	if (!lidar_stream_seen_.load(std::memory_order_relaxed)
		&& !lidar_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
	{
		std::print(stderr, "[read_lidar] No LiDAR frames received since startup. Waiting for input stream...\n");
	}
}

void SpecificWorker::log_room_robot_pose_periodic(const Mat::RTMat& room_T_robot) const
{
	(void)room_T_robot;
}

void SpecificWorker::update_room_polygon_periodic()
{
	static int polygon_check_count = 0;
	if (++polygon_check_count % 50 == 0)
		update_room_polygon_in_viewers();
}

std::vector<SegDetection> SpecificWorker::detect_segmentation(const RoboCompCameraRGBDSimple::TRGBD& rgbd)
{
	if (!yolo_detector.has_value() || rgbd.image.width <= 0 || rgbd.image.height <= 0)
		return {};

	const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
		const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
	const cv::Mat masked_rgb_frame = apply_tray_mask(rgb_frame);
	auto detections = yolo_detector->detect(masked_rgb_frame, /*is_rgb=*/true);
	postprocess_yolo_detections(detections);
	return detections;
}

std::vector<cv::Point> SpecificWorker::get_tray_mask_polygon(const cv::Size& image_size) const
{
	if (!params.YOLO_MASK_TRAY || params.YOLO_TRAY_MASK_POLYGON_PX.size() < 3
		|| image_size.width <= 0 || image_size.height <= 0
		|| params.YOLO_TRAY_MASK_REF_WIDTH <= 0 || params.YOLO_TRAY_MASK_REF_HEIGHT <= 0)
		return {};

	const float scale_x = static_cast<float>(image_size.width) / static_cast<float>(params.YOLO_TRAY_MASK_REF_WIDTH);
	const float scale_y = static_cast<float>(image_size.height) / static_cast<float>(params.YOLO_TRAY_MASK_REF_HEIGHT);

	std::vector<cv::Point> polygon;
	polygon.reserve(params.YOLO_TRAY_MASK_POLYGON_PX.size());
	for (const auto& p : params.YOLO_TRAY_MASK_POLYGON_PX)
	{
		polygon.emplace_back(
			std::clamp(static_cast<int>(std::lround(static_cast<float>(p.x) * scale_x)), 0, image_size.width - 1),
			std::clamp(static_cast<int>(std::lround(static_cast<float>(p.y) * scale_y)), 0, image_size.height - 1));
	}

	return polygon;
}

cv::Mat SpecificWorker::apply_tray_mask(const cv::Mat& rgb_frame) const
{
	if (rgb_frame.empty())
		return {};

	const auto polygon = get_tray_mask_polygon(rgb_frame.size());
	if (polygon.size() < 3)
		return rgb_frame.clone();

	cv::Mat masked = rgb_frame.clone();
	const std::vector<std::vector<cv::Point>> polygons{polygon};
	cv::fillPoly(masked, polygons, cv::Scalar(0, 0, 0));
	return masked;
}


std::string SpecificWorker::normalize_yolo_label(const std::string& label) const
{
	std::string out = label;
	std::transform(out.begin(), out.end(), out.begin(),
	               [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
	if (out == "dining table") return "table";
	if (out == "tv") return "monitor";
	return out;
}

bool SpecificWorker::is_accepted_yolo_label(const std::string& label) const
{
	if (params.YOLO_ACCEPTED_LABELS.empty())
		return true;

	return std::find(params.YOLO_ACCEPTED_LABELS.begin(), params.YOLO_ACCEPTED_LABELS.end(), label)
	       != params.YOLO_ACCEPTED_LABELS.end();
}


// YOLO postprocessing logic 
void SpecificWorker::postprocess_yolo_detections(std::vector<SegDetection>& detections) const
{
	for (auto& det : detections)
	{
		det.label = normalize_yolo_label(det.label);
	}

	std::erase_if(detections, [&](const SegDetection& det)
	{
		return !is_accepted_yolo_label(det.label);
	});

	for (auto& det : detections)
	{
		const bool is_target = (det.label == "table" || det.label == "chair" || det.label == "monitor");
		if (is_target && !det.mask.empty())
		{
			cv::Mat eroded;
			int k = std::max(1, params.YOLO_MASK_ERODE_KERNEL);
			cv::erode(det.mask, eroded, cv::Mat(), cv::Point(-1, -1), k);
			det.mask = eroded;
		}
	}
}

// voxel grid update logic 
void SpecificWorker::update_voxel_grid_from_rgbd(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                              const std::vector<SegDetection>& detections,
	                                              const Mat::RTMat& room_T_robot,
	                                              const Mat::RTMat& room_T_zed)
{
	// Update the unified voxel grid: project each RGBD point into 3D,
	// label it with the YOLO class if its pixel falls inside a mask.
	const auto& raw_pts = rgbd.points.points;
	const int   img_w   = rgbd.image.width;
	const int   img_h   = rgbd.image.height;

	// Sanity check: the points array should have one entry per pixel. 
	//If not, something is wrong with the RGBD data and we skip processing this frame.
	if (!raw_pts.empty() && img_w > 0 && img_h > 0
	    && static_cast<int>(raw_pts.size()) == img_w * img_h)
	{
		std::size_t                  valid_points = 0;
		std::size_t                  masked_points = 0;
		std::size_t                  selected_points = 0;
		std::size_t                  decimated_points = 0;
		std::unordered_map<std::string, std::size_t> selected_by_class;
		std::vector<float> det_median_range_m(detections.size(), std::numeric_limits<float>::quiet_NaN());
		std::vector<int32_t> pixel_owner(static_cast<std::size_t>(img_w * img_h), -1);
		const float point_scale = detect_point_scale_once(rgbd);
		build_owner_map_and_medians(rgbd, point_scale, detections, pixel_owner, det_median_range_m);

		const std::size_t n_dets = detections.size();
		std::vector<std::vector<Eigen::Vector3f>> points_by_det(n_dets);
		std::vector<std::vector<std::string>> labels_by_det(n_dets);
		std::vector<std::vector<float>> confs_by_det(n_dets);
		std::vector<Eigen::Vector3f> selected_points_robot;
		std::vector<std::size_t> selected_det_indices;
		selected_points_robot.reserve(static_cast<std::size_t>(img_w * img_h / 6));
		selected_det_indices.reserve(selected_points_robot.capacity());

		for (int row = 0; row < img_h; ++row)
		{
			for (int col = 0; col < img_w; ++col)
			{
				const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
				const auto& p = raw_pts[idx];

				if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
					continue;

				const float px = p.x * point_scale;
				const float py = p.y * point_scale;
				const float pz = p.z * point_scale;
				const float rng_sq = px * px + py * py + pz * pz;
				if (rng_sq < 0.01f || rng_sq > 100.0f)
					continue;
				const float rng = std::sqrt(rng_sq);

				++valid_points;

				const int32_t owner = pixel_owner[idx];
				if (owner < 0)
					continue;

				const std::size_t det_idx = static_cast<std::size_t>(owner);
				if (det_idx >= n_dets)
					continue;

				const auto& det = detections[det_idx];
				const float ref_rng = det_median_range_m[det_idx];
				if (std::isfinite(ref_rng) && std::abs(rng - ref_rng) > 0.35f)
					continue;

				++masked_points;
				++selected_points;
				selected_points_robot.emplace_back(px, py, pz);
				selected_det_indices.push_back(det_idx);

				if (det.label == "table") ++selected_by_class["table"];
				else if (det.label == "chair") ++selected_by_class["chair"];
				else if (det.label == "monitor") ++selected_by_class["monitor"];
			}
		}

		if (!selected_points_robot.empty())
		{
			// Points are expressed in robot-oriented axes but originate at the camera.
			// So use room->robot rotation and room->zed translation, both sampled at
			// the same RGBD timestamp, to preserve heading and eliminate temporal blur.

			const std::size_t n_sel = selected_points_robot.size();
			Eigen::Matrix<double, 3, Eigen::Dynamic> pts_robot(3, static_cast<Eigen::Index>(n_sel));
			for (std::size_t i = 0; i < n_sel; ++i)
			{
				pts_robot(0, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].x());
				pts_robot(1, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].y());
				pts_robot(2, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].z());
			}

			Eigen::Matrix<double, 3, Eigen::Dynamic> pts_room =
				(room_T_robot.linear() * pts_robot).colwise() + room_T_zed.translation();

			for (std::size_t i = 0; i < n_sel; ++i)
			{
				const std::size_t det_idx = selected_det_indices[i];
				points_by_det[det_idx].emplace_back(
					static_cast<float>(pts_room(0, static_cast<Eigen::Index>(i))),
					static_cast<float>(pts_room(1, static_cast<Eigen::Index>(i))),
					static_cast<float>(pts_room(2, static_cast<Eigen::Index>(i))));
				labels_by_det[det_idx].push_back(detections[det_idx].label);
				confs_by_det[det_idx].push_back(detections[det_idx].confidence);
			}
		}

		const int frame_id = ++compute_frame_;
		std::vector<DetectionObservation> observations;
		observations.reserve(n_dets);

		for (std::size_t d = 0; d < n_dets; ++d)
		{
			if (points_by_det[d].empty())
				continue;
			if (!is_target_label(detections[d].label))
				continue;

			Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
			for (const auto& p : points_by_det[d])
				centroid += p;
			centroid /= static_cast<float>(points_by_det[d].size());

			observations.push_back(DetectionObservation{
				.det_index = d,
				.centroid = centroid,
				.label = detections[d].label,
				.confidence = detections[d].confidence
			});
		}

		std::vector<int> det_to_track(n_dets, -1);
		if (!observations.empty())
		{
			auto track_ids = associate_detections_hungarian(observations, frame_id);
			for (std::size_t i = 0; i < observations.size(); ++i)
				det_to_track[observations[i].det_index] = track_ids[i];
		}
		else
		{
			prune_stale_tracks(frame_id);
		}

		std::vector<TrackBoxCandidate> box_candidates;

		if (voxel_grid)
		{
			const std::size_t voxel_decimation_step = std::max<std::size_t>(1, params.VOXEL_DECIMATION_FACTOR);
			for (std::size_t d = 0; d < n_dets; ++d)
			{
				const int track_id = det_to_track[d];
				if (track_id < 0 || points_by_det[d].empty())
					continue;

				std::vector<Eigen::Vector3f> pts_decimated;
				std::vector<std::string>     labels_decimated;
				std::vector<float>           confs_decimated;

				pts_decimated.reserve((points_by_det[d].size() + voxel_decimation_step - 1) / voxel_decimation_step);
				labels_decimated.reserve(pts_decimated.capacity());
				confs_decimated.reserve(pts_decimated.capacity());

				for (std::size_t i = 0; i < points_by_det[d].size(); i += voxel_decimation_step)
				{
					pts_decimated.push_back(points_by_det[d][i]);
					labels_decimated.push_back(labels_by_det[d][i]);
					confs_decimated.push_back(confs_by_det[d][i]);
				}

				decimated_points += pts_decimated.size();
				voxel_grid->observe(track_id,
				                    pts_decimated,
				                    detections[d].label,
				                    frame_id,
				                    labels_decimated,
				                    confs_decimated,
				                    detections[d].confidence);
			}

			box_candidates = build_track_box_candidates();
			merge_duplicate_tracks(box_candidates, frame_id);
		}

		if (voxel_grid)
		{
			const auto sem = voxel_grid->export_semantic_voxels();
			std::vector<QVector3D> qpts;
			qpts.reserve(sem.points.size());
			for (const auto& p : sem.points)
				qpts.emplace_back(p.x(), p.y(), p.z());
			if (voxel_viewer_gl)
			{
				voxel_viewer_gl->update_voxels(qpts, sem.categories, sem.probs);

				const auto filtered_boxes = filter_track_boxes_for_viewer(box_candidates);

				std::vector<QVector3D> box_mins;
				std::vector<QVector3D> box_maxs;
				std::vector<std::string> box_categories;
				box_mins.reserve(filtered_boxes.size());
				box_maxs.reserve(filtered_boxes.size());
				box_categories.reserve(filtered_boxes.size());
				for (const auto& box : filtered_boxes)
				{
					box_mins.emplace_back(box.min.x(), box.min.y(), box.min.z());
					box_maxs.emplace_back(box.max.x(), box.max.y(), box.max.z());
					box_categories.push_back(box.category);
				}

				voxel_viewer_gl->update_track_boxes(box_mins, box_maxs, box_categories);
			}
		}

		if (verbose_debug_ && compute_frame_ % 30 == 0)
		{
			const float ratio = valid_points > 0
				? (100.0f * static_cast<float>(masked_points) / static_cast<float>(valid_points))
				: 0.0f;
			std::println("[VoxelDebug] valid_pts={} masked_pts={} ({:.1f}%) selected_pts={} decimated_pts={} table={} chair={} monitor={} detections={} active_tracks={}",
			             valid_points,
			             masked_points,
			             ratio,
			             selected_points,
			             decimated_points,
			             selected_by_class["table"],
			             selected_by_class["chair"],
			             selected_by_class["monitor"],
			             detections.size(),
			             active_tracks.size());
		}
	}
}

std::optional<SpecificWorker::RoomPolygonData> SpecificWorker::get_room_polygon_from_graph() const
{
	if (!G)
		return std::nullopt;

	std::string room_name_snapshot;
	{
		std::scoped_lock lk(node_names_mutex_);
		room_name_snapshot = room_node_name_;
	}

	if (room_name_snapshot.empty())
	{
		if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
			room_name_snapshot = room_nodes.front().name();
	}

	if (room_name_snapshot.empty())
		return std::nullopt;

	auto room_node = G->get_node(room_name_snapshot);
	if (!room_node.has_value())
		return std::nullopt;

	auto polygon_x_opt = G->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
	auto polygon_y_opt = G->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());
	if (!polygon_x_opt.has_value() || !polygon_y_opt.has_value())
		return std::nullopt;

	const auto &polygon_x_src = polygon_x_opt.value().get();
	const auto &polygon_y_src = polygon_y_opt.value().get();
	if (polygon_x_src.empty() || polygon_y_src.empty())
		return std::nullopt;

	RoomPolygonData data;
	data.room_name = std::move(room_name_snapshot);
	data.polygon_x.assign(polygon_x_src.begin(), polygon_x_src.end());
	data.polygon_y.assign(polygon_y_src.begin(), polygon_y_src.end());
	if (auto height_opt = G->get_attrib_by_name<room_height_att>(room_node.value()); height_opt.has_value())
		data.room_height = height_opt.value();

	return data;
}

void SpecificWorker::overlay_room_polygon_on_canvas(cv::Mat& canvas,
	                                                const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
{
	if (canvas.empty() || !G || !inner_eigen_api)
		return;

	auto room_data = get_room_polygon_from_graph();
	if (!room_data.has_value())
		return;

	auto zed_node = G->get_node("zed");
	if (!zed_node.has_value())
		return;

	auto camera_api = G->get_camera_api(zed_node.value());
	if (!camera_api)
		return;

	const std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(rgbd);
	const std::size_t n = std::min(room_data->polygon_x.size(), room_data->polygon_y.size());
	if (n < 2)
		return;

	RoomToCameraBasis basis;
	if (!compute_room_to_camera_basis(inner_eigen_api.get(), "zed", room_data->room_name, frame_ts_ms, params.TRANSFORMS_INTERPOLATE_RT, basis))
		return;

	Eigen::Matrix<double, 3, Eigen::Dynamic> room_points(3, static_cast<Eigen::Index>(n));
	Eigen::Matrix<double, 3, Eigen::Dynamic> room_points_top(3, static_cast<Eigen::Index>(n));
	for (std::size_t i = 0; i < n; ++i)
	{
		room_points(0, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_x[i]);
		room_points(1, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_y[i]);
		room_points(2, static_cast<Eigen::Index>(i)) = 0.0;
		room_points_top(0, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_x[i]);
		room_points_top(1, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_y[i]);
		room_points_top(2, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->room_height);
	}

	Eigen::Matrix3d basis_matrix;
	basis_matrix.col(0) = basis.axis_x;
	basis_matrix.col(1) = basis.axis_y;
	basis_matrix.col(2) = basis.axis_z;
	const Eigen::Vector3d basis_origin = basis.origin;

	Eigen::Matrix<double, 3, Eigen::Dynamic> zed_points =
		(basis_matrix * room_points).colwise() + basis_origin;
	Eigen::Matrix<double, 3, Eigen::Dynamic> zed_points_top =
		(basis_matrix * room_points_top).colwise() + basis_origin;

	auto project_clipped_segment = [&](Eigen::Vector3d a, Eigen::Vector3d b,
	                                  cv::Point& out_a, cv::Point& out_b) -> bool
	{
		constexpr double near_y = 1e-4;

		if (a.y() <= near_y && b.y() <= near_y)
			return false;

		if (a.y() <= near_y)
		{
			const double t = (near_y - a.y()) / (b.y() - a.y());
			a = a + t * (b - a);
		}
		else if (b.y() <= near_y)
		{
			const double t = (near_y - b.y()) / (a.y() - b.y());
			b = b + t * (a - b);
		}

		const Eigen::Vector2d uv0 = camera_api->project(a);
		const Eigen::Vector2d uv1 = camera_api->project(b);
		if (!std::isfinite(uv0.x()) || !std::isfinite(uv0.y()) || !std::isfinite(uv1.x()) || !std::isfinite(uv1.y()))
			return false;

		out_a = cv::Point(static_cast<int>(std::lround(uv0.x())), static_cast<int>(std::lround(uv0.y())));
		out_b = cv::Point(static_cast<int>(std::lround(uv1.x())), static_cast<int>(std::lround(uv1.y())));
		return true;
	};

	std::vector<std::optional<cv::Point>> projected_floor(n);
	std::vector<std::optional<cv::Point>> projected_top(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		const Eigen::Vector3d floor_point_zed = zed_points.col(static_cast<Eigen::Index>(i));
		if (floor_point_zed.y() > 1e-6)
		{
			const Eigen::Vector2d uv = camera_api->project(floor_point_zed);
			if (std::isfinite(uv.x()) && std::isfinite(uv.y()))
				projected_floor[i] = cv::Point(static_cast<int>(std::lround(uv.x())),
				                              static_cast<int>(std::lround(uv.y())));
		}

		const Eigen::Vector3d top_point_zed = zed_points_top.col(static_cast<Eigen::Index>(i));
		if (top_point_zed.y() > 1e-6)
		{
			const Eigen::Vector2d uv = camera_api->project(top_point_zed);
			if (std::isfinite(uv.x()) && std::isfinite(uv.y()))
				projected_top[i] = cv::Point(static_cast<int>(std::lround(uv.x())),
				                            static_cast<int>(std::lround(uv.y())));
		}
	}

	const cv::Scalar floor_colour(255, 0, 255);
	const cv::Scalar top_colour(0, 0, 255);
	const cv::Scalar vertical_colour(0, 200, 255);
	for (std::size_t i = 0; i < n; ++i)
	{
		const std::size_t j = (i + 1) % n;

		cv::Point p0;
		cv::Point p1;
		if (project_clipped_segment(zed_points.col(static_cast<Eigen::Index>(i)),
		                           zed_points.col(static_cast<Eigen::Index>(j)),
		                           p0, p1))
		{
			if (cv::clipLine(canvas.size(), p0, p1))
				cv::line(canvas, p0, p1, floor_colour, 2, cv::LINE_AA);
		}

		if (project_clipped_segment(zed_points_top.col(static_cast<Eigen::Index>(i)),
		                           zed_points_top.col(static_cast<Eigen::Index>(j)),
		                           p0, p1))
		{
			if (cv::clipLine(canvas.size(), p0, p1))
				cv::line(canvas, p0, p1, top_colour, 2, cv::LINE_AA);
		}

		if (project_clipped_segment(zed_points.col(static_cast<Eigen::Index>(i)),
		                           zed_points_top.col(static_cast<Eigen::Index>(i)),
		                           p0, p1))
		{
			if (cv::clipLine(canvas.size(), p0, p1))
				cv::line(canvas, p0, p1, vertical_colour, 2, cv::LINE_AA);
		}
	}

	for (const auto& p : projected_floor)
	{
		if (!p.has_value())
			continue;
		if (p->x >= 0 && p->x < canvas.cols && p->y >= 0 && p->y < canvas.rows)
			cv::circle(canvas, p.value(), 4, floor_colour, cv::FILLED, cv::LINE_AA);
	}

	for (const auto& p : projected_top)
	{
		if (!p.has_value())
			continue;
		if (p->x >= 0 && p->x < canvas.cols && p->y >= 0 && p->y < canvas.rows)
			cv::circle(canvas, p.value(), 4, top_colour, cv::FILLED, cv::LINE_AA);
	}
}

void SpecificWorker::update_room_polygon_in_viewers()
{
	auto room_data = get_room_polygon_from_graph();
	if (!room_data.has_value())
		return;

	try
	{
		// Polygon corners are already in room-local frame (centered near origin).
		// Voxels from get_transformation_matrix(room, "zed") are also in room-local frame.
		// No transform needed; the robot pose lives inside this same frame.
		if (!room_data->polygon_x.empty() && !room_data->polygon_y.empty() && voxel_viewer_gl)
			voxel_viewer_gl->update_room_polygon_dual(room_data->polygon_x, room_data->polygon_y, room_data->room_height);
	}
	catch (const std::exception& e)
	{
		qWarning() << "update_room_polygon_in_viewers failed:" << e.what();
	}
}

void SpecificWorker::draw_detections(const cv::Mat& rgb_frame,
                                     const std::vector<SegDetection>& detections) const
{
	const cv::Mat canvas = compose_detection_canvas(rgb_frame, detections);

	cv::imshow("YOLO detections", canvas);
	cv::waitKey(1);
}

cv::Mat SpecificWorker::compose_detection_canvas(const cv::Mat& rgb_frame,
	                                             const std::vector<SegDetection>& detections) const
{
	// Work in BGR (OpenCV native)
	cv::Mat canvas;
	cv::cvtColor(rgb_frame, canvas, cv::COLOR_RGB2BGR);

	// Colour palette — one colour per class (mod 20)
	static const std::array<cv::Scalar, 20> palette = {{
		{220,  20,  60}, {119,  11,  32}, {  0,   0, 142}, {  0,   0, 230}, { 106,   0, 228},
		{  0,  60, 100}, {  0,  80, 100}, {  0,   0, 192}, {250, 170,  30}, {100, 170,  30},
		{220, 220,   0}, {175, 116, 175}, {250,   0,  30}, {165,  42,  42}, {255,  77, 255},
		{  0, 226, 252}, {182, 182, 255}, {  0,  82,   0}, {120, 166, 157}, {110,  76,   0},
	}};

	for (const auto& det : detections)
	{
		const cv::Scalar& colour = palette[static_cast<std::size_t>(det.class_id) % palette.size()];

        // Coloured mask overlay (alpha blend directly onto canvas)
        if (!det.mask.empty())
        {
            // Build a solid-colour BGR image the same size as canvas
            cv::Mat colour_layer(canvas.size(), CV_8UC3, colour);
            // Binary mask (CV_8UC1, 0 or 255)
            cv::Mat mask_bin;
            cv::threshold(det.mask, mask_bin, 127, 255, cv::THRESH_BINARY);
            // Blend only the masked pixels: canvas = canvas*0.55 + colour*0.45
            cv::Mat blended;
            cv::addWeighted(canvas, 0.55, colour_layer, 0.45, 0.0, blended);
            blended.copyTo(canvas, mask_bin);   // copy blended pixels where mask==255

            // Contour outline for crisp mask boundary
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(canvas, contours, -1, colour, 1, cv::LINE_AA);
        }

        // Bounding box
        cv::rectangle(canvas, det.bbox, colour, 2);

        // Label background + text
		const std::string text = std::format("{} {:.2f}", det.label, det.confidence);
		const int font        = cv::FONT_HERSHEY_SIMPLEX;
		const double scale    = 0.65;
		const int thickness   = 2;
		int baseline          = 0;
		const cv::Size ts     = cv::getTextSize(text, font, scale, thickness, &baseline);

		const int label_pad = 4;
		const int box_x0 = std::clamp(det.bbox.x, 0, std::max(0, canvas.cols - 1));
		const int box_y0 = std::clamp(det.bbox.y, 0, std::max(0, canvas.rows - 1));
		const int label_w = std::min(canvas.cols, ts.width + 2 * label_pad);
		const int label_h = ts.height + baseline + 2 * label_pad;
		const bool place_above = box_y0 >= label_h;
		const int label_x0 = std::clamp(box_x0, 0, std::max(0, canvas.cols - label_w));
		const int label_y0 = place_above ? (box_y0 - label_h) : std::min(box_y0 + 2, std::max(0, canvas.rows - label_h));
		const cv::Rect label_rect(label_x0, label_y0, label_w, std::min(label_h, canvas.rows - label_y0));

		cv::rectangle(canvas, label_rect, colour, cv::FILLED);
		cv::putText(canvas, text,
			    cv::Point(label_rect.x + label_pad, label_rect.y + label_rect.height - baseline - label_pad),
			    font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }

	return canvas;
}

void SpecificWorker::update_yolo_tab_display(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
                                              const std::vector<SegDetection>& detections)
{
	static auto last_display_update = std::chrono::steady_clock::time_point{};
	static float display_fps = 0.f;

	if (yolo_image_label_ == nullptr || rgbd.image.width == 0 || rgbd.image.height == 0
		|| !custom_widget_yolo.isVisible())
		return;

	const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
		const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
	const cv::Mat masked_rgb_frame = apply_tray_mask(rgb_frame);
	cv::Mat yolo_canvas = compose_detection_canvas(masked_rgb_frame, detections);
	overlay_room_polygon_on_canvas(yolo_canvas, rgbd);
	cv::Mat yolo_canvas_rgb;
	cv::cvtColor(yolo_canvas, yolo_canvas_rgb, cv::COLOR_BGR2RGB);
	QImage yolo_qimg(yolo_canvas_rgb.data,
		yolo_canvas_rgb.cols,
		yolo_canvas_rgb.rows,
		static_cast<int>(yolo_canvas_rgb.step),
		QImage::Format_RGB888);
	QPixmap yolo_pix = QPixmap::fromImage(yolo_qimg, Qt::NoFormatConversion);
	yolo_image_label_->setPixmap(yolo_pix.scaled(yolo_image_label_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

	if (yolo_fps_label_ != nullptr)
	{
		const auto now = std::chrono::steady_clock::now();
		if (last_display_update != std::chrono::steady_clock::time_point{})
		{
			const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_display_update).count();
			if (dt_ms > 0)
			{
				const float inst_fps = 1000.0f / static_cast<float>(dt_ms);
				display_fps = (display_fps > 0.f) ? (0.85f * display_fps + 0.15f * inst_fps) : inst_fps;
			}
		}
		last_display_update = now;
		yolo_fps_label_->setText(QString("YOLO display FPS: %1").arg(display_fps, 0, 'f', 1));
	}
}

void SpecificWorker::update_viewer_robot_pose(const Mat::RTMat& room_T_robot)
{
	if (!voxel_viewer_gl)
		return;
	const auto& t = room_T_robot.translation();
	const Eigen::Matrix3d R = room_T_robot.rotation();
	const float theta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
	voxel_viewer_gl->set_robot_pose(static_cast<float>(t.x()), static_cast<float>(t.y()), theta);
}

void SpecificWorker::update_viewer_lidar_points(const std::string& room_name,
	                                            const std::string& robot_name,
	                                            const Mat::RTMat& room_T_robot_fallback)
{
	if (!voxel_viewer_gl)
		return;

	std::vector<float> xs, ys, zs;
	std::uint64_t lidar_timestamp_ms = 0;
	{
		std::scoped_lock lk(lidar_points_mutex_);
		xs = latest_lidar_xs_;
		ys = latest_lidar_ys_;
		zs = latest_lidar_zs_;
		lidar_timestamp_ms = latest_lidar_timestamp_ms_;
	}

	if (xs.empty() || ys.size() != xs.size() || zs.size() != xs.size())
	{
		voxel_viewer_gl->update_lidar_points({});
		return;
	}

	Mat::RTMat room_T_robot = room_T_robot_fallback;
	if (inner_eigen_api && lidar_timestamp_ms > 0)
	{
		const auto time_query = params.TRANSFORMS_INTERPOLATE_RT
			? DSR::RT_API::TimeQuery::Interpolated
			: DSR::RT_API::TimeQuery::Nearest;
		if (auto interpolated = inner_eigen_api->get_transformation_matrix(
				room_name,
				robot_name,
				lidar_timestamp_ms,
				"RT",
				time_query); interpolated.has_value())
		{
			room_T_robot = interpolated.value();
		}
	}

	std::vector<QVector3D> lidar_points_room;
	lidar_points_room.reserve(xs.size());
	for (std::size_t i = 0; i < xs.size(); ++i)
	{
		const Eigen::Vector3d point_robot(static_cast<double>(xs[i]),
		                                 static_cast<double>(ys[i]),
		                                 static_cast<double>(zs[i]));
		const Eigen::Vector3d point_room = room_T_robot.linear() * point_robot + room_T_robot.translation();
		lidar_points_room.emplace_back(static_cast<float>(point_room.x()),
		                              static_cast<float>(point_room.y()),
		                              static_cast<float>(point_room.z()));
	}

	voxel_viewer_gl->update_lidar_points(lidar_points_room);
}

void SpecificWorker::read_lidar_thread()
{
	static FPSCounter lidar_fps;
	bool empty_lidar_logged = false;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_lidar_thread)
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
				if (!lidar_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
					std::print(stderr, "[read_lidar] Empty LiDAR stream received. Waiting for points...\n");
				empty_lidar_logged = true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		else if (empty_lidar_logged)
		{
			lidar_stream_seen_.store(true, std::memory_order_relaxed);
			lidar_stream_wait_logged_.store(false, std::memory_order_relaxed);
			std::print("[read_lidar] LiDAR stream recovered.\n");
			empty_lidar_logged = false;
		}
		else
		{
			lidar_stream_seen_.store(true, std::memory_order_relaxed);
		}

		const auto n = data.points.size();
		std::vector<float> xs(n), ys(n), zs(n);
		for (std::size_t i = 0; i < n; ++i)
		{
			xs[i] = data.points[i].x/1000.f;  // mm to m
			ys[i] = data.points[i].y/1000.f;  // mm to m
			zs[i] = data.points[i].z/1000.f;  // mm to m
		}
		// Upload to DSR graph (rate-limited by dsr_lidar_fps; 0 = every scan).
		static auto last_lidar_upload = std::chrono::steady_clock::time_point{};
		const auto  now_steady_lidar  = std::chrono::steady_clock::now();
		const auto  lidar_interval_ms = params.DSR_LIDAR_FPS > 0
			? std::chrono::milliseconds(1000 / params.DSR_LIDAR_FPS)
			: std::chrono::milliseconds(0);
		const bool do_lidar_upload = (lidar_interval_ms.count() == 0)
			|| (now_steady_lidar - last_lidar_upload >= lidar_interval_ms);
		if (do_lidar_upload)
		{
			last_lidar_upload = now_steady_lidar;
			if (auto laser_node = G->get_node("lidar3D"); laser_node.has_value())
			{
				G->add_or_modify_attrib_local<laser_X_att>(laser_node.value(), xs);
				G->add_or_modify_attrib_local<laser_Y_att>(laser_node.value(), ys);
				G->add_or_modify_attrib_local<laser_Z_att>(laser_node.value(), zs);
				G->add_or_modify_attrib_local<laser_timestamp_att>(laser_node.value(), static_cast<uint64_t>(data.timestamp));
				G->update_node(laser_node.value());
			}
			else
				qWarning() << "Laser node not found in DSR graph";
		}

		{
			std::scoped_lock lk(lidar_points_mutex_);
			latest_lidar_xs_ = xs;
			latest_lidar_ys_ = ys;
			latest_lidar_zs_ = zs;
			latest_lidar_timestamp_ms_ = static_cast<std::uint64_t>(data.timestamp);
		}

		const long p_ms = static_cast<long>(data.period);
		if (wait_period > std::chrono::milliseconds(p_ms + 2)) --wait_period;
		else if (wait_period < std::chrono::milliseconds(p_ms - 2)) ++wait_period;

		if (verbose_debug_)
			lidar_fps.print("[LidarThread]", 2000);
		std::this_thread::sleep_for(wait_period);
    }
}

void SpecificWorker::read_imu_thread()
{
	static FPSCounter imu_fps;
	auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
	std::uint64_t prev_sensor_timestamp_ms = 0;
	bool missing_imu_node_logged = false;

	while (!stop_imu_thread)
	{
		RoboCompIMU::DataImu data;
		try
		{
			data = imu_proxy->getDataImu();
		}
		catch (const Ice::Exception& e)
		{
			qWarning() << "[read_imu] getDataImu failed:" << e.what() << "retrying...";
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}

		const std::uint64_t sensor_timestamp_ms = get_imu_timestamp_ms(data);
		const std::vector<float> acceleration{data.acc.XAcc, data.acc.YAcc, data.acc.ZAcc};
		const std::vector<float> gyroscope{data.gyro.XGyr, data.gyro.YGyr, data.gyro.ZGyr};
		const std::vector<float> euler_xyz{data.rot.Roll, data.rot.Pitch, data.rot.Yaw};

		if (auto imu_node = G->get_node("imu"); imu_node.has_value())
		{
			G->add_or_modify_attrib_local<imu_accelerometer_att>(imu_node.value(), acceleration);
			G->add_or_modify_attrib_local<imu_linear_acceleration_att>(imu_node.value(), acceleration);
			G->add_or_modify_attrib_local<imu_gyroscope_att>(imu_node.value(), gyroscope);
			G->add_or_modify_attrib_local<imu_angular_velocity_att>(imu_node.value(), gyroscope);
			G->add_or_modify_attrib_local<imu_angular_euler_xyz_pose_att>(imu_node.value(), euler_xyz);
			G->add_or_modify_attrib_local<imu_compass_att>(imu_node.value(), data.rot.Yaw);
			G->add_or_modify_attrib_local<imu_time_stamp_att>(imu_node.value(), sensor_timestamp_ms);
			G->add_or_modify_attrib_local<imu_sensor_tick_att>(imu_node.value(), sensor_timestamp_ms);
			G->update_node(imu_node.value());

			if (missing_imu_node_logged)
			{
				qInfo() << "[read_imu] IMU node recovered in DSR graph.";
				missing_imu_node_logged = false;
			}
		}
		else if (!missing_imu_node_logged)
		{
			qWarning() << "[read_imu] IMU node not found in DSR graph.";
			missing_imu_node_logged = true;
		}

		if (sensor_timestamp_ms > prev_sensor_timestamp_ms)
		{
			const auto sensor_period = std::chrono::milliseconds(sensor_timestamp_ms - prev_sensor_timestamp_ms);
			if (sensor_period.count() > 0 && sensor_period <= std::chrono::seconds(1))
			{
				if (wait_period > sensor_period + std::chrono::milliseconds(2)) --wait_period;
				else if (wait_period < sensor_period - std::chrono::milliseconds(2)) ++wait_period;
			}
		}
		if (sensor_timestamp_ms > 0)
			prev_sensor_timestamp_ms = sensor_timestamp_ms;

		if (verbose_debug_)
			imu_fps.print("[ImuThread]", 2000);
		std::this_thread::sleep_for(wait_period);
	}
}

bool SpecificWorker::is_target_label(const std::string& label) const
{
	return label == "table" || label == "chair" || label == "monitor";
}

float SpecificWorker::detect_point_scale_once(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
{
	for (const auto& p : rgbd.points.points)
	{
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
			continue;
		const float max_abs = std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)});
		return (max_abs > 50.0f) ? 0.001f : 1.0f;
	}
	return 1.0f;
}

void SpecificWorker::build_owner_map_and_medians(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                             float point_scale,
	                                             const std::vector<SegDetection>& detections,
	                                             std::vector<int32_t>& pixel_owner,
	                                             std::vector<float>& det_median_range_m) const
{
	const auto& raw_pts = rgbd.points.points;
	const int img_w = rgbd.image.width;
	const int img_h = rgbd.image.height;

	for (std::size_t d = 0; d < detections.size(); ++d)
	{
		const auto& det = detections[d];
		if (!is_target_label(det.label) || det.mask.empty())
			continue;

		const int x0 = std::max(0, det.bbox.x);
		const int y0 = std::max(0, det.bbox.y);
		const int x1 = std::min({img_w, det.bbox.x + det.bbox.width, det.mask.cols});
		const int y1 = std::min({img_h, det.bbox.y + det.bbox.height, det.mask.rows});
		if (x0 >= x1 || y0 >= y1)
			continue;

		for (int row = y0; row < y1; ++row)
		{
			for (int col = x0; col < x1; ++col)
			{
				if (det.mask.at<uint8_t>(row, col) == 0)
					continue;
				const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
				if (pixel_owner[idx] == -1)
					pixel_owner[idx] = static_cast<int32_t>(d);
			}
		}

		std::vector<float> ranges;
		ranges.reserve(static_cast<std::size_t>((x1 - x0) * (y1 - y0) / 4));
		for (int row = y0; row < y1; ++row)
		{
			for (int col = x0; col < x1; ++col)
			{
				if (det.mask.at<uint8_t>(row, col) == 0)
					continue;

				const auto& p = raw_pts[static_cast<std::size_t>(row * img_w + col)];
				if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
					continue;

				const float px = p.x * point_scale;
				const float py = p.y * point_scale;
				const float pz = p.z * point_scale;
				const float rng_sq = px * px + py * py + pz * pz;
				if (rng_sq >= 0.01f && rng_sq <= 100.0f)
					ranges.push_back(std::sqrt(rng_sq));
			}
		}

		if (!ranges.empty())
		{
			auto mid = ranges.begin() + static_cast<std::ptrdiff_t>(ranges.size() / 2);
			std::nth_element(ranges.begin(), mid, ranges.end());
			det_median_range_m[d] = *mid;
		}
	}
}

SpecificWorker::VoxelSelectionResult SpecificWorker::collect_points_parallel(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                                                        float point_scale,
	                                                                        const std::vector<int32_t>& pixel_owner,
	                                                                        const std::vector<SegDetection>& detections,
	                                                                        const std::vector<float>& det_median_range_m) const
{
	struct BlockAccum
	{
		std::vector<Eigen::Vector3f> pts;
		std::vector<std::string> labels;
		std::vector<float> confs;
		std::size_t valid = 0;
		std::size_t masked = 0;
		std::size_t selected = 0;
		std::size_t table = 0;
		std::size_t chair = 0;
		std::size_t monitor = 0;
	};

	const auto& raw_pts = rgbd.points.points;
	const int img_w = rgbd.image.width;
	const int img_h = rgbd.image.height;
	constexpr int row_block = 24;
	const int n_blocks = (img_h + row_block - 1) / row_block;
	std::vector<int> block_ids(static_cast<std::size_t>(n_blocks));
	std::iota(block_ids.begin(), block_ids.end(), 0);
	std::vector<BlockAccum> block_accums(static_cast<std::size_t>(n_blocks));

	std::for_each(std::execution::par, block_ids.begin(), block_ids.end(), [&](int block_id)
	{
		BlockAccum& acc = block_accums[static_cast<std::size_t>(block_id)];
		const int row0 = block_id * row_block;
		const int row1 = std::min(img_h, row0 + row_block);
		acc.pts.reserve(static_cast<std::size_t>(std::max(1, (row1 - row0) * img_w / 8)));
		acc.labels.reserve(acc.pts.capacity());
		acc.confs.reserve(acc.pts.capacity());

		for (int row = row0; row < row1; ++row)
		{
			for (int col = 0; col < img_w; ++col)
			{
				const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
				const auto& p = raw_pts[idx];

				if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
					continue;

				const float px = p.x * point_scale;
				const float py = p.y * point_scale;
				const float pz = p.z * point_scale;
				const float rng_sq = px * px + py * py + pz * pz;
				if (rng_sq < 0.01f || rng_sq > 100.0f)
					continue;
				const float rng = std::sqrt(rng_sq);

				++acc.valid;

				const int32_t owner = pixel_owner[idx];
				if (owner < 0)
					continue;

				const auto& det = detections[static_cast<std::size_t>(owner)];
				const float ref_rng = det_median_range_m[static_cast<std::size_t>(owner)];
				if (std::isfinite(ref_rng) && std::abs(rng - ref_rng) > 0.35f)
					continue;

				++acc.masked;
				acc.pts.emplace_back(px, py, pz);
				acc.labels.push_back(det.label);
				acc.confs.push_back(det.confidence);
				++acc.selected;

				if (det.label == "table") ++acc.table;
				else if (det.label == "chair") ++acc.chair;
				else if (det.label == "monitor") ++acc.monitor;
			}
		}
	});

	VoxelSelectionResult out;
	for (auto& acc : block_accums)
	{
		out.valid_points += acc.valid;
		out.masked_points += acc.masked;
		out.selected_points += acc.selected;
		out.table_points += acc.table;
		out.chair_points += acc.chair;
		out.monitor_points += acc.monitor;

		out.points.insert(out.points.end(),
		                 std::make_move_iterator(acc.pts.begin()),
		                 std::make_move_iterator(acc.pts.end()));
		out.labels.insert(out.labels.end(),
		                 std::make_move_iterator(acc.labels.begin()),
		                 std::make_move_iterator(acc.labels.end()));
		out.confidences.insert(out.confidences.end(),
		                      std::make_move_iterator(acc.confs.begin()),
		                      std::make_move_iterator(acc.confs.end()));
	}

	return out;
}

std::vector<int> SpecificWorker::hungarian_min_cost(const std::vector<std::vector<float>>& cost) const
{
	const std::size_t n = cost.size();
	if (n == 0)
		return {};

	std::size_t m = 0;
	for (const auto& row : cost)
		m = std::max(m, row.size());

	if (m == 0)
		return std::vector<int>(n, -1);

	const std::size_t dim = std::max(n, m);
	constexpr double big_cost = 1e9;
	std::vector<std::vector<double>> a(n, std::vector<double>(dim, big_cost));
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = 0; j < cost[i].size(); ++j)
			a[i][j] = static_cast<double>(cost[i][j]);

	std::vector<double> u(n + 1, 0.0), v(dim + 1, 0.0);
	std::vector<std::size_t> p(dim + 1, 0), way(dim + 1, 0);

	for (std::size_t i = 1; i <= n; ++i)
	{
		p[0] = i;
		std::size_t j0 = 0;
		std::vector<double> minv(dim + 1, std::numeric_limits<double>::infinity());
		std::vector<bool> used(dim + 1, false);

		do
		{
			used[j0] = true;
			const std::size_t i0 = p[j0];
			double delta = std::numeric_limits<double>::infinity();
			std::size_t j1 = 0;
			for (std::size_t j = 1; j <= dim; ++j)
			{
				if (used[j])
					continue;
				const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
				if (cur < minv[j])
				{
					minv[j] = cur;
					way[j] = j0;
				}
				if (minv[j] < delta)
				{
					delta = minv[j];
					j1 = j;
				}
			}

			for (std::size_t j = 0; j <= dim; ++j)
			{
				if (used[j])
				{
					u[p[j]] += delta;
					v[j] -= delta;
				}
				else
				{
					minv[j] -= delta;
				}
			}
			j0 = j1;
		}
		while (p[j0] != 0);

		do
		{
			const std::size_t j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		}
		while (j0 != 0);
	}

	std::vector<int> assignment(n, -1);
	for (std::size_t j = 1; j <= dim; ++j)
	{
		if (p[j] == 0)
			continue;
		const std::size_t row = p[j] - 1;
		const std::size_t col = j - 1;
		if (row < n && col < m && col < cost[row].size())
			assignment[row] = static_cast<int>(col);
	}

	return assignment;
}

std::vector<int> SpecificWorker::associate_detections_hungarian(const std::vector<DetectionObservation>& observations,
	                                                            int frame_id)
{
	std::vector<int> out(observations.size(), -1);
	if (observations.empty())
		return out;

	if (active_tracks.empty())
	{
		for (std::size_t i = 0; i < observations.size(); ++i)
		{
			const int new_id = next_track_id_++;
			active_tracks[new_id] = InstanceTrack{
				.id = new_id,
				.centroid = observations[i].centroid,
				.label = observations[i].label,
				.last_seen_frame = frame_id
			};
			out[i] = new_id;
		}
		return out;
	}

	std::vector<int> track_ids;
	track_ids.reserve(active_tracks.size());
	for (const auto& [track_id, _] : active_tracks)
		track_ids.push_back(track_id);

	constexpr float impossible_cost = 1e6f;
	std::vector<std::vector<float>> cost(observations.size(), std::vector<float>(track_ids.size(), impossible_cost));
	for (std::size_t i = 0; i < observations.size(); ++i)
	{
		for (std::size_t j = 0; j < track_ids.size(); ++j)
		{
			const auto it = active_tracks.find(track_ids[j]);
			if (it == active_tracks.end())
				continue;
			const auto& tr = it->second;
			if (tr.label != observations[i].label)
				continue;
			cost[i][j] = (observations[i].centroid - tr.centroid).norm();
		}
	}

	const auto assignment = hungarian_min_cost(cost);
	for (std::size_t i = 0; i < observations.size(); ++i)
	{
		const int col = assignment[i];
		if (col < 0 || static_cast<std::size_t>(col) >= track_ids.size())
			continue;

		const float c = cost[i][static_cast<std::size_t>(col)];
		if (c > track_association_max_distance_m || c >= impossible_cost * 0.5f)
			continue;

		const int track_id = track_ids[static_cast<std::size_t>(col)];
		out[i] = track_id;
		auto& tr = active_tracks[track_id];
		tr.centroid = 0.65f * tr.centroid + 0.35f * observations[i].centroid;
		tr.label = observations[i].label;
		tr.last_seen_frame = frame_id;
	}

	for (std::size_t i = 0; i < observations.size(); ++i)
	{
		if (out[i] != -1)
			continue;

		const int new_id = next_track_id_++;
		active_tracks[new_id] = InstanceTrack{
			.id = new_id,
			.centroid = observations[i].centroid,
			.label = observations[i].label,
			.last_seen_frame = frame_id
		};
		out[i] = new_id;
	}

	prune_stale_tracks(frame_id);
	return out;
}

void SpecificWorker::prune_stale_tracks(int frame_id)
{
	std::erase_if(active_tracks, [&](const auto& kv)
	{
		return (frame_id - kv.second.last_seen_frame) > track_max_missed_frames;
	});
}

std::vector<TrackBoxCandidate> SpecificWorker::build_track_box_candidates() const
{
	std::vector<TrackBoxCandidate> candidates;
	if (!voxel_grid)
		return candidates;

	const auto track_ids = voxel_grid->get_all_track_ids();
	candidates.reserve(track_ids.size());

	for (const int tid : track_ids)
	{
		if (tid <= 0)
			continue;

		const auto [dom_cat, _] = voxel_grid->object_dominant_category(tid);
		auto pts = voxel_grid->get_points_clustered(tid, dom_cat);
		if (pts.size() < 10)
			pts = voxel_grid->get_points(tid);
		if (pts.size() < 10)
			continue;

		Eigen::Vector3f mn = pts.front();
		Eigen::Vector3f mx = pts.front();
		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		for (const auto& p : pts)
		{
			mn = mn.cwiseMin(p);
			mx = mx.cwiseMax(p);
			centroid += p;
		}
		centroid /= static_cast<float>(pts.size());

		int last_seen_frame = -1;
		if (const auto it = active_tracks.find(tid); it != active_tracks.end())
			last_seen_frame = it->second.last_seen_frame;

		candidates.push_back(TrackBoxCandidate{
			.track_id = tid,
			.category = dom_cat,
			.min = mn,
			.max = mx,
			.centroid = centroid,
			.voxel_count = voxel_grid->get_n_voxels(tid),
			.last_seen_frame = last_seen_frame
		});
	}

	return candidates;
}

void SpecificWorker::merge_duplicate_tracks(std::vector<TrackBoxCandidate>& candidates, int frame_id)
{
	if (!voxel_grid || candidates.empty())
		return;

	std::sort(candidates.begin(), candidates.end(),
	          [frame_id](const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs)
	          {
		          return prefer_candidate(lhs, rhs, frame_id);
	          });

	std::unordered_set<int> merged_tracks;
	for (std::size_t i = 0; i < candidates.size(); ++i)
	{
		auto& keep = candidates[i];
		if (merged_tracks.contains(keep.track_id))
			continue;

		for (std::size_t j = i + 1; j < candidates.size(); ++j)
		{
			auto& drop = candidates[j];
			if (merged_tracks.contains(drop.track_id))
				continue;
			if (!boxes_look_duplicate(drop, keep))
				continue;

			const int moved = voxel_grid->reassign_ownership(drop.track_id, keep.track_id);
			if (moved <= 0)
				continue;

			const float keep_weight = static_cast<float>(std::max(1, keep.voxel_count));
			const float drop_weight = static_cast<float>(std::max(1, drop.voxel_count));
			keep.centroid = (keep_weight * keep.centroid + drop_weight * drop.centroid) / (keep_weight + drop_weight);
			keep.min = keep.min.cwiseMin(drop.min);
			keep.max = keep.max.cwiseMax(drop.max);
			keep.voxel_count += drop.voxel_count;
			keep.last_seen_frame = std::max(keep.last_seen_frame, drop.last_seen_frame);

			auto keep_it = active_tracks.find(keep.track_id);
			const auto drop_it = active_tracks.find(drop.track_id);
			if (keep_it == active_tracks.end())
			{
				active_tracks[keep.track_id] = InstanceTrack{
					.id = keep.track_id,
					.centroid = keep.centroid,
					.label = keep.category,
					.last_seen_frame = keep.last_seen_frame
				};
				keep_it = active_tracks.find(keep.track_id);
			}
			else
			{
				keep_it->second.centroid = keep.centroid;
				keep_it->second.label = keep.category;
				keep_it->second.last_seen_frame = keep.last_seen_frame;
			}

			if (drop_it != active_tracks.end())
				active_tracks.erase(drop_it);

			merged_tracks.insert(drop.track_id);
			if (verbose_debug_)
				std::println("[TrackMerge] merged track={} into track={} category={} moved_voxels={}",
				             drop.track_id,
				             keep.track_id,
				             keep.category,
				             moved);
		}
	}

	if (!merged_tracks.empty())
		std::erase_if(candidates, [&](const TrackBoxCandidate& candidate)
		{
			return merged_tracks.contains(candidate.track_id);
		});
}

std::vector<TrackBoxCandidate> SpecificWorker::filter_track_boxes_for_viewer(const std::vector<TrackBoxCandidate>& candidates) const
{
	std::vector<TrackBoxCandidate> filtered_boxes;
	filtered_boxes.reserve(candidates.size());
	std::unordered_map<std::string, int> kept_per_category;

	for (const auto& candidate : candidates)
	{
		bool is_duplicate = false;
		for (const auto& kept : filtered_boxes)
		{
			if (!boxes_look_duplicate(candidate, kept))
				continue;

			is_duplicate = true;
			if (verbose_debug_)
				std::println("[TrackBox] suppressing duplicate box track={} category={} kept_track={} overlap_vol={:.3f}",
				             candidate.track_id,
				             candidate.category,
				             kept.track_id,
				             intersection_volume(candidate, kept));
			break;
		}
		if (is_duplicate)
			continue;

		const int max_instances = max_instances_for_category(candidate.category);
		const int kept_instances = kept_per_category[candidate.category];
		if (kept_instances >= max_instances)
		{
			if (verbose_debug_)
				std::println("[TrackBox] suppressing by category cap track={} category={} cap={}",
				             candidate.track_id,
				             candidate.category,
				             max_instances);
			continue;
		}

		filtered_boxes.push_back(candidate);
		++kept_per_category[candidate.category];
	}

	return filtered_boxes;
}

////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_rgbd_thread()
{
    static FPSCounter rgbd_fps;
	bool empty_rgbd_logged = false;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_rgbd_thread)
    {
		const auto loop_start = std::chrono::steady_clock::now();
		const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
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
			const bool empty_points = frame.points.points.empty();
			if (empty_rgb || empty_depth || empty_points)
			{
				if (!empty_rgbd_logged)
				{
					if (!rgbd_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
						std::print(stderr, "[read_rgbd] Empty RGBD stream received. Waiting for RGB, depth and point cloud data...\n");
					empty_rgbd_logged = true;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
			else if (empty_rgbd_logged)
			{
				rgbd_stream_seen_.store(true, std::memory_order_relaxed);
				rgbd_stream_wait_logged_.store(false, std::memory_order_relaxed);
				std::print("[read_rgbd] RGBD stream recovered.\n");
				empty_rgbd_logged = false;
			}
			else
			{
				rgbd_stream_seen_.store(true, std::memory_order_relaxed);
			}

            const long p_ms = static_cast<long>(frame.image.period);
			std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(frame);

			// Upload RGB to DSR at the configured rate (dsr_rgb_fps=0 means every frame).
			// Depth is large; upload at dsr_depth_fps.
			static auto last_rgb_upload   = std::chrono::steady_clock::time_point{};
			static auto last_depth_upload = std::chrono::steady_clock::time_point{};
			const auto now_steady = std::chrono::steady_clock::now();
			const auto rgb_interval_ms   = params.DSR_RGB_FPS   > 0
				? std::chrono::milliseconds(1000 / params.DSR_RGB_FPS) : std::chrono::milliseconds(0);
			const auto depth_interval_ms = params.DSR_DEPTH_FPS > 0
				? std::chrono::milliseconds(1000 / params.DSR_DEPTH_FPS) : std::chrono::milliseconds(0);
			const bool do_rgb   = (rgb_interval_ms.count()   == 0) || (now_steady - last_rgb_upload   >= rgb_interval_ms);
			const bool do_depth = (depth_interval_ms.count() == 0) || (now_steady - last_depth_upload >= depth_interval_ms);
			if (auto cam_node = G->get_node("zed"); cam_node.has_value())
			{
				G->add_or_modify_attrib_local<cam_rgb_width_att>(cam_node.value(), frame.image.width);
				G->add_or_modify_attrib_local<cam_rgb_height_att>(cam_node.value(), frame.image.height);
				G->add_or_modify_attrib_local<cam_rgb_focalx_att>(cam_node.value(), frame.image.focalx);
				G->add_or_modify_attrib_local<cam_rgb_focaly_att>(cam_node.value(), frame.image.focaly);
				G->add_or_modify_attrib_local<cam_rgb_depth_att>(cam_node.value(), 3);
				G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(cam_node.value(), 0);
				if (do_rgb)
				{
					last_rgb_upload = now_steady;
					G->add_or_modify_attrib_local<cam_rgb_att>(cam_node.value(),
						std::vector<uint8_t>(frame.image.image.begin(), frame.image.image.end()));
					G->add_or_modify_attrib_local<cam_rgb_alivetime_att>(cam_node.value(), static_cast<std::uint64_t>(frame.image.alivetime));
					G->update_node(cam_node.value());
				}

				if (do_depth)
				{
					last_depth_upload = now_steady;
					if (auto depth_node = G->get_node("zed"); depth_node.has_value())
					{
						G->add_or_modify_attrib_local<cam_depth_width_att>(depth_node.value(), frame.depth.width);
						G->add_or_modify_attrib_local<cam_depth_height_att>(depth_node.value(), frame.depth.height);
						G->add_or_modify_attrib_local<cam_depth_focalx_att>(depth_node.value(), frame.depth.focalx);
						G->add_or_modify_attrib_local<cam_depth_focaly_att>(depth_node.value(), frame.depth.focaly);
						G->add_or_modify_attrib_local<cam_depthFactor_att>(depth_node.value(), frame.depth.depthFactor);
						G->add_or_modify_attrib_local<cam_depth_att>(depth_node.value(),
							std::vector<uint8_t>(frame.depth.depth.begin(), frame.depth.depth.end()));
						G->update_node(depth_node.value());
					}
				}
			}
			else
				qWarning() << "Camera node not found in DSR graph";
			if (frame_ts_ms == 0)
				frame_ts_ms = static_cast<std::uint64_t>(now_ms);

			rgbd_buffer.put<0>(
                std::move(frame),
				frame_ts_ms,
                [](auto &&input, auto &output) { output = std::forward<decltype(input)>(input); });

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
        { qWarning() << "[read_rgbd] Ice exception:" << e.what(); }
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


//Execute one when exiting to emergencyState
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



