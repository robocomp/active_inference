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
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <print>
#include <limits>
#include <algorithm>
#include <cctype>
#include <execution>
#include <numeric>
#include <iterator>
#include <unordered_map>
#include <Eigen/Geometry>
#include "custom_widget.h"
#include "ui_localUI.h"


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
	GenericWorker::initialize();

    // ── Initialise YOLO-seg detector ─────────────────────────
    try
    {
        params.YOLO_MODEL_PATH  = configLoader.get<std::string>("Yolo.model_path");
    }
    catch (...) { /* key absent — use default */ }
	try { params.YOLO_CONF_THRESH = static_cast<float>(configLoader.get<double>("Yolo.conf_thresh")); } catch (...) {}
	try { params.YOLO_IOU_THRESH  = static_cast<float>(configLoader.get<double>("Yolo.iou_thresh"));  } catch (...) {}
	try { params.YOLO_USE_GPU     = configLoader.get<bool>("Yolo.use_gpu");  } catch (...) {}
	try { params.YOLO_USE_TRT     = configLoader.get<bool>("Yolo.use_trt");  } catch (...) {}
	try { params.YOLO_MASK_ERODE_KERNEL = configLoader.get<int>("Yolo.mask_erode_kernel"); } catch (...) {}
	try { params.TRACK_ASSOCIATION_MAX_DISTANCE_M = static_cast<float>(configLoader.get<double>("Yolo.track_association_max_distance_m")); } catch (...) {}
	try { params.TRACK_MAX_MISSED_FRAMES = configLoader.get<int>("Yolo.track_max_missed_frames"); } catch (...) {}
	std::println("[YOLO] effective flags: use_gpu={} use_trt={}", params.YOLO_USE_GPU, params.YOLO_USE_TRT);

    yolo_detector.emplace(params.YOLO_MODEL_PATH,
                          std::vector<std::string>{},   // default COCO names
                          params.YOLO_CONF_THRESH,
                          params.YOLO_IOU_THRESH,
                          params.YOLO_INPUT_SIZE,
                          params.YOLO_USE_GPU,
                          params.YOLO_USE_TRT);
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

		voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(custom_widget.frame);
		custom_widget.frame->layout()->addWidget(voxel_viewer_gl.get());
		qInfo() << __FUNCTION__ << "Voxel OpenGL custom widget attached to graph viewer";

		if (custom_widget_yolo.frame->layout() == nullptr)
		{
			auto* layout = new QHBoxLayout(custom_widget_yolo.frame);
			layout->setContentsMargins(0, 0, 0, 0);
			custom_widget_yolo.frame->setLayout(layout);
		}

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
		room_node_name_ = room_nodes.front().name();
		std::println("[VoxelInit] Found room node: '{}'", room_node_name_);
	}
	else
	{
		std::println("[VoxelInit] No room nodes found in graph");
	}
	
	if (const auto robot_nodes = G->get_nodes_by_type("robot"); !robot_nodes.empty())
		robot_node_name_ = robot_nodes.front().name();
	if (!room_node_name_.empty() && !robot_node_name_.empty())
	{
		std::println("[VoxelInit] Cached room='{}' robot='{}' from initial graph", room_node_name_, robot_node_name_);
	}

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
	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	// Read latest RGDB frame
	const auto &[rgbd_opt] = rgbd_buffer.read(now_ms);
	if (!rgbd_opt.has_value())
	{
		qWarning() << "No RGBD data available at timestamp" << now_ms;
		return;
	}
	const auto& rgbd = rgbd_opt.value();
	const std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(rgbd);

	// Always compute detections for voxel grid update
	auto detections = detect_segmentation(rgbd);

	// Only update ZED RGB and YOLO overlay if the widget is visible (tab selected)
	if (yolo_image_label_ != nullptr && rgbd.image.width > 0 && rgbd.image.height > 0 && custom_widget_yolo.isVisible())
	{
		const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
			const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
		const cv::Mat yolo_canvas = compose_detection_canvas(rgb_frame, detections);
		cv::Mat yolo_canvas_rgb;
		cv::cvtColor(yolo_canvas, yolo_canvas_rgb, cv::COLOR_BGR2RGB);
		QImage yolo_qimg(yolo_canvas_rgb.data,
			yolo_canvas_rgb.cols,
			yolo_canvas_rgb.rows,
			static_cast<int>(yolo_canvas_rgb.step),
			QImage::Format_RGB888);
		QPixmap yolo_pix = QPixmap::fromImage(yolo_qimg.copy());
		yolo_image_label_->setPixmap(yolo_pix.scaled(yolo_image_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}

	// Only update voxel/room/robot if ready
	if (!ensure_room_and_robot_ready(compute_fps))
		return;

	// Get robot pose in room frame
	auto room_T_robot = get_room_robot_transform(compute_fps, frame_ts_ms);
	if (!room_T_robot.has_value())
		return;
	auto room_T_zed = get_room_zed_transform(compute_fps, frame_ts_ms);
	if (!room_T_zed.has_value())
		return;

	// Log robot pose periodically (even if not ready) so the user gets feedback when it becomes available.
	log_room_robot_pose_periodic(room_T_robot.value());
	if (!room_rt_ready_logged_)
	{
		room_rt_ready_logged_ = true;
		room_rt_wait_logged_ = false;
	}

	// Push the robot pose (in room frame) to the GL viewer so the user can
	// visually verify polygon/voxels/robot are all coherent in the same frame.
	if (voxel_viewer_gl)
	{
		const auto& T = room_T_robot.value();
		const auto& t = T.translation();
		const Eigen::Matrix3d R = T.rotation();
		const float theta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
		voxel_viewer_gl->set_robot_pose(static_cast<float>(t.x()), static_cast<float>(t.y()), theta);
	}

	// Update the room polygon in the viewers periodically (in case it changes in the graph or simply to trigger a redraw).
	update_room_polygon_periodic();

	// Update the voxel grid with the new RGBD frame and YOLO detections, using the robot pose to keep everything aligned in the room frame.
	track_association_max_distance_m = params.TRACK_ASSOCIATION_MAX_DISTANCE_M;
	track_max_missed_frames = params.TRACK_MAX_MISSED_FRAMES;
	update_voxel_grid_from_rgbd(rgbd, detections, room_T_robot.value(), room_T_zed.value());

	compute_fps.print("[Compute]", 2000);
}

//////////////////////////////////////////////////////////////////////////////

bool SpecificWorker::ensure_room_and_robot_ready(FPSCounter& compute_fps)
{
	if (room_node_name_.empty())
		if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
			room_node_name_ = room_nodes.front().name();

	if (room_node_name_.empty())
	{
		if (!room_wait_logged_)
		{
			qWarning() << "Room node not found in DSR graph. Voxelization paused until a room exists.";
			room_wait_logged_ = true;
			room_ready_logged_ = false;
		}
		compute_fps.print("[Compute]", 2000);
		return false;
	}

	if (robot_node_name_.empty())
		if (const auto robot_nodes = G->get_nodes_by_type("robot"); !robot_nodes.empty())
			robot_node_name_ = robot_nodes.front().name();

	if (robot_node_name_.empty())
	{
		if (!room_wait_logged_)
		{
			qWarning() << "Robot node not found in DSR graph. Voxelization paused until a robot exists.";
			room_wait_logged_ = true;
			room_ready_logged_ = false;
		}
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

std::optional<Mat::RTMat> SpecificWorker::get_room_robot_transform(FPSCounter& compute_fps, std::uint64_t timestamp_ms)
{
	if (!inner_eigen_api)
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "InnerEigen API is not available. Voxelization paused.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	auto room_T_robot = inner_eigen_api->get_transformation_matrix(room_node_name_, robot_node_name_, timestamp_ms);
	if (!room_T_robot.has_value())
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "robot->room RTMat not available in InnerEigen API. Voxelization paused until transform is available.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	return room_T_robot;
}

std::optional<Mat::RTMat> SpecificWorker::get_room_zed_transform(FPSCounter& compute_fps, std::uint64_t timestamp_ms)
{
	if (!inner_eigen_api)
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "InnerEigen API is not available. Voxelization paused.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
		compute_fps.print("[Compute]", 2000);
		return std::nullopt;
	}

	auto room_T_zed = inner_eigen_api->get_transformation_matrix(room_node_name_, "zed", timestamp_ms);
	if (!room_T_zed.has_value())
	{
		if (!room_rt_wait_logged_)
		{
			qWarning() << "zed->room RTMat not available in InnerEigen API. Voxelization paused until transform is available.";
			room_rt_wait_logged_ = true;
			room_rt_ready_logged_ = false;
		}
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
	auto detections = yolo_detector->detect(rgb_frame, /*is_rgb=*/true);
	postprocess_yolo_detections(detections);
	return detections;
}


// YOLO postprocessing logic 
void SpecificWorker::postprocess_yolo_detections(std::vector<SegDetection>& detections) const
{
    auto normalize_label = [](const std::string& label) -> std::string
    {
        std::string out = label;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (out == "dining table") return "table";
        if (out == "tv") return "monitor";
        return out;
    };

	for (auto& det : detections)
	{
		det.label = normalize_label(det.label);
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
		}

		if (voxel_grid && (compute_frame_ % 8 == 0))
		{
			const auto sem = voxel_grid->export_semantic_voxels();
			std::vector<QVector3D> qpts;
			qpts.reserve(sem.points.size());
			for (const auto& p : sem.points)
				qpts.emplace_back(p.x(), p.y(), p.z());
			if (voxel_viewer_gl)
			{
				voxel_viewer_gl->update_voxels(qpts, sem.categories, sem.probs);

				std::vector<QVector3D> box_mins;
				std::vector<QVector3D> box_maxs;
				std::vector<std::string> box_categories;

				const auto track_ids = voxel_grid->get_all_track_ids();
				box_mins.reserve(track_ids.size());
				box_maxs.reserve(track_ids.size());
				box_categories.reserve(track_ids.size());

				for (const int tid : track_ids)
				{
					auto pts = voxel_grid->get_points(tid);
					if (pts.size() < 10)
						continue;

					Eigen::Vector3f mn = pts.front();
					Eigen::Vector3f mx = pts.front();
					for (const auto& p : pts)
					{
						mn = mn.cwiseMin(p);
						mx = mx.cwiseMax(p);
					}

					const auto [dom_cat, _] = voxel_grid->object_dominant_category(tid);
					box_mins.emplace_back(mn.x(), mn.y(), mn.z());
					box_maxs.emplace_back(mx.x(), mx.y(), mx.z());
					box_categories.push_back(dom_cat);
				}

				voxel_viewer_gl->update_track_boxes(box_mins, box_maxs, box_categories);
			}
		}

		if (compute_frame_ % 30 == 0)
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

void SpecificWorker::update_room_polygon_in_viewers()
{
	if (room_node_name_.empty())
		return;

	try
	{
		auto room_node = G->get_node(room_node_name_);
		if (!room_node)
			return;

		// Use typed DSR API accessors for registered attribute names.
		auto polygon_x_opt = G->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
		auto polygon_y_opt = G->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());

		if (!polygon_x_opt.has_value())
			return;
		if (!polygon_y_opt.has_value())
			return;

		const auto &polygon_x_src = polygon_x_opt.value().get();
		const auto &polygon_y_src = polygon_y_opt.value().get();

		std::vector<float> polygon_x(polygon_x_src.begin(), polygon_x_src.end());
		std::vector<float> polygon_y(polygon_y_src.begin(), polygon_y_src.end());

		// Polygon corners are already in room-local frame (centered near origin).
		// Voxels from get_transformation_matrix(room, "zed") are also in room-local frame.
		// No transform needed; the robot pose lives inside this same frame.
		float room_height = 0.f;
		if (auto height_opt = G->get_attrib_by_name<room_height_att>(room_node.value()); height_opt.has_value())
			room_height = height_opt.value();

		if (!polygon_x.empty() && !polygon_y.empty() && voxel_viewer_gl)
			voxel_viewer_gl->update_room_polygon_dual(polygon_x, polygon_y, room_height);
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
        const double scale    = 0.55;
        const int thickness   = 1;
        int baseline          = 0;
        const cv::Size ts     = cv::getTextSize(text, font, scale, thickness, &baseline);
        const cv::Point tl    = det.bbox.tl();
        cv::rectangle(canvas,
                      cv::Point(tl.x, tl.y - ts.height - 4),
                      cv::Point(tl.x + ts.width + 2, tl.y),
                      colour, cv::FILLED);
        cv::putText(canvas, text,
                    cv::Point(tl.x + 1, tl.y - 3),
                    font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }

	return canvas;
}

// update_yolo_tab_views removed: now handled inline in compute()

void SpecificWorker::read_lidar_thread()
{
	static FPSCounter lidar_fps;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_lidar_thread)
    {
        //const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        //    std::chrono::system_clock::now().time_since_epoch()).count();
        try
        {
            RoboCompLidar3D::TData data;
            try
            {
                data = lidar3d_proxy->getLidarData(
                    "", 0.f, static_cast<float>(M_PI) * 2.f, params.LIDAR_DECIMATION_FACTOR);
            }
            catch (const Ice::Exception& e)
            { qWarning() << "[read_lidar] getLidarData failed:" << e.what(); std::terminate(); }

			const auto n = data.points.size();
			std::vector<float> xs(n), ys(n), zs(n);
			for (std::size_t i = 0; i < n; ++i)
			{
				xs[i] = data.points[i].x;
				ys[i] = data.points[i].y;
				zs[i] = data.points[i].z;
			}
			// Upload to DSR graph
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

			// pointcloud_buffer.put<0>(
			// 	std::make_pair(std::move(data.points), static_cast<std::uint64_t>(data.timestamp)),
			// 	timestamp,
			// 	[](auto &&input, auto &output)
			// 	{
			// 		auto &&[points, lidar_ts] = input;
			// 		auto &[ts, xs, ys, zs] = output;
			// 		ts = lidar_ts;
			// 		const auto n = points.size();
			// 		xs.resize(n);
			// 		ys.resize(n);
			// 		zs.resize(n);
			// 		for (std::size_t i = 0; i < n; ++i)
			// 		{
			// 			xs[i] = points[i].x;
			// 			ys[i] = points[i].y;
			// 			zs[i] = points[i].z;
			// 		}
			// 	});
        
            const long p_ms = static_cast<long>(data.period);
            if (wait_period > std::chrono::milliseconds(p_ms + 2)) --wait_period;
            else if (wait_period < std::chrono::milliseconds(p_ms - 2)) ++wait_period;

            lidar_fps.print("[LidarThread]", 2000);
            std::this_thread::sleep_for(wait_period);
        }
        catch (const Ice::Exception& e)
        { qWarning() << "[read_lidar] Ice exception:" << e.what(); }
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

////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_rgbd_thread()
{
    static FPSCounter rgbd_fps;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_rgbd_thread)
    {
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
            { qWarning() << "[read_rgbd] getAll failed:" << e.what(); std::terminate(); }

            const long p_ms = static_cast<long>(frame.image.period);

			// Upload to DSR graph (throttled to ~2 Hz to keep FastDDS bandwidth sane;
			// the voxel pipeline consumes the RGBD via rgbd_buffer, not the DSR attribute).
			static auto last_dsr_upload = std::chrono::steady_clock::time_point{};
			const auto now_steady = std::chrono::steady_clock::now();
			const bool do_dsr_upload = (now_steady - last_dsr_upload) >= std::chrono::milliseconds(500);
			if (do_dsr_upload)
			{
				last_dsr_upload = now_steady;
				if (auto cam_node = G->get_node("zed"); cam_node.has_value())
				{
					// First push the small metadata attributes (incl. the ones the GUI's CameraAPI needs).
					G->add_or_modify_attrib_local<cam_rgb_width_att>(cam_node.value(), frame.image.width);
					G->add_or_modify_attrib_local<cam_rgb_height_att>(cam_node.value(), frame.image.height);
					G->add_or_modify_attrib_local<cam_rgb_focalx_att>(cam_node.value(), frame.image.focalx);
					G->add_or_modify_attrib_local<cam_rgb_focaly_att>(cam_node.value(), frame.image.focaly);
					G->add_or_modify_attrib_local<cam_rgb_depth_att>(cam_node.value(), 3);
					G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(cam_node.value(), 0);
					G->add_or_modify_attrib_local<cam_depth_width_att>(cam_node.value(), frame.depth.width);
					G->add_or_modify_attrib_local<cam_depth_height_att>(cam_node.value(), frame.depth.height);
					G->add_or_modify_attrib_local<cam_depth_focalx_att>(cam_node.value(), frame.depth.focalx);
					G->add_or_modify_attrib_local<cam_depth_focaly_att>(cam_node.value(), frame.depth.focaly);
					G->add_or_modify_attrib_local<cam_depthFactor_att>(cam_node.value(), frame.depth.depthFactor);
					G->update_node(cam_node.value());

					// Send one big blob per tick, alternating RGB and depth.
					// Sending both back-to-back causes FastDDS to drop the first
					// (only the most recent large sample reaches the GUI subscriber).
					static bool send_rgb_this_tick = true;
					if (send_rgb_this_tick)
					{
						if (auto cam_node2 = G->get_node("zed"); cam_node2.has_value())
						{
							G->add_or_modify_attrib_local<cam_rgb_att>(cam_node2.value(),
								std::vector<uint8_t>(frame.image.image.begin(), frame.image.image.end()));
							G->update_node(cam_node2.value());
						}
					}
					else
					{
						if (auto cam_node3 = G->get_node("zed"); cam_node3.has_value())
						{
							G->add_or_modify_attrib_local<cam_depth_att>(cam_node3.value(),
								std::vector<uint8_t>(frame.depth.depth.begin(), frame.depth.depth.end()));
							G->update_node(cam_node3.value());
						}
					}
					send_rgb_this_tick = !send_rgb_this_tick;

				}
				else
					qWarning() << "Camera node not found in DSR graph";
			}

			std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(frame);
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

            rgbd_fps.print("[RGBDThread]", 2000);
            std::this_thread::sleep_for(wait_period);
        }
        catch (const Ice::Exception& e)
        { qWarning() << "[read_rgbd] Ice exception:" << e.what(); }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////
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



