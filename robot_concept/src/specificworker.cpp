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
#include "scene_processor.h"
#include "voxel_processor.h"
#include "yolo_processor.h"
#ifdef emit
#undef emit
#endif

#include "unified_voxel_grid.h"
#include "voxel_opengl_viewer.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <algorithm>
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

	try
	{
		params.YOLO_MODEL_PATH = configLoader.get<std::string>("Yolo.model_path");
	}
	catch (...) { }
	try { params.YOLO_ACCEPTED_LABELS = configLoader.get<std::vector<std::string>>("Yolo.accepted_labels"); } catch (...) {}
	try { params.YOLO_CONF_THRESH = static_cast<float>(configLoader.get<double>("Yolo.conf_thresh")); } catch (...) {}
	try { params.YOLO_IOU_THRESH = static_cast<float>(configLoader.get<double>("Yolo.iou_thresh")); } catch (...) {}
	try { params.YOLO_USE_GPU = configLoader.get<bool>("Yolo.use_gpu"); } catch (...) {}
	try { params.YOLO_USE_TRT = configLoader.get<bool>("Yolo.use_trt"); } catch (...) {}
	try { params.YOLO_MASK_ERODE_KERNEL = configLoader.get<int>("Yolo.mask_erode_kernel"); } catch (...) {}
	try { params.TRACK_ASSOCIATION_MAX_DISTANCE_M = static_cast<float>(configLoader.get<double>("Yolo.track_association_max_distance_m")); } catch (...) {}
	try { params.TRACK_MAX_MISSED_FRAMES = configLoader.get<int>("Yolo.track_max_missed_frames"); } catch (...) {}
	try { params.DSR_RGB_FPS = configLoader.get<int>("Camera.dsr_rgb_fps"); } catch (...) {}
	try { params.DSR_DEPTH_FPS = configLoader.get<int>("Camera.dsr_depth_fps"); } catch (...) {}
	try { params.DSR_LIDAR_FPS = configLoader.get<int>("Lidar.dsr_lidar_fps"); } catch (...) {}
	try { params.TRANSFORMS_INTERPOLATE_RT = configLoader.get<bool>("Transforms.interpolate_rt"); } catch (...) {}
	try { verbose_debug_ = configLoader.get<bool>("Debug.verbose"); } catch (...) { verbose_debug_ = false; }
	yolo_processor = std::make_unique<YoloProcessor>();
	YoloProcessor::Config yolo_config;
	yolo_config.model_path = params.YOLO_MODEL_PATH;
	yolo_config.conf_thresh = params.YOLO_CONF_THRESH;
	yolo_config.iou_thresh = params.YOLO_IOU_THRESH;
	yolo_config.input_size = params.YOLO_INPUT_SIZE;
	yolo_config.use_gpu = params.YOLO_USE_GPU;
	yolo_config.use_trt = params.YOLO_USE_TRT;
	yolo_config.mask_erode_kernel = params.YOLO_MASK_ERODE_KERNEL;
	yolo_config.mask_tray = params.YOLO_MASK_TRAY;
	yolo_config.tray_mask_ref_width = params.YOLO_TRAY_MASK_REF_WIDTH;
	yolo_config.tray_mask_ref_height = params.YOLO_TRAY_MASK_REF_HEIGHT;
	yolo_config.tray_mask_polygon_px = params.YOLO_TRAY_MASK_POLYGON_PX;
	yolo_config.accepted_labels = params.YOLO_ACCEPTED_LABELS;
	yolo_config.verbose_debug = verbose_debug_;
	yolo_processor->configure(yolo_config);

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

	voxel_grid = std::make_unique<UnifiedVoxelGrid>();
	voxel_processor = std::make_unique<VoxelProcessor>(*voxel_grid);
	VoxelProcessor::Config voxel_processor_config;
	voxel_processor_config.voxel_decimation_factor = params.VOXEL_DECIMATION_FACTOR;
	voxel_processor_config.track_association_max_distance_m = params.TRACK_ASSOCIATION_MAX_DISTANCE_M;
	voxel_processor_config.track_max_missed_frames = params.TRACK_MAX_MISSED_FRAMES;
	voxel_processor_config.verbose_debug = verbose_debug_;
	voxel_processor->configure(voxel_processor_config);
	inner_eigen_api = G->get_inner_eigen_api();
	scene_processor = std::make_unique<SceneProcessor>(G,
	                                                  lidar_points_mutex_,
	                                                  latest_lidar_xs_,
	                                                  latest_lidar_ys_,
	                                                  latest_lidar_zs_,
	                                                  latest_lidar_timestamp_ms_,
	                                                  lidar_stream_seen_,
	                                                  rgbd_stream_seen_,
	                                                  lidar_stream_wait_logged_,
	                                                  rgbd_stream_wait_logged_);
	scene_processor->configure(inner_eigen_api.get(), voxel_viewer_gl.get(), params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_);

	imu_thread = std::thread(&SpecificWorker::read_imu_thread, this);
	qInfo() << __FUNCTION__ << "Started IMU reader";

	lidar_thread = std::thread(&SpecificWorker::read_lidar_thread, this);
	qInfo() << __FUNCTION__ << "Started lidar reader";

	rgbd_thread = std::thread(&SpecificWorker::read_rgbd_thread, this);
	qInfo() << __FUNCTION__ << "Started RGBD reader";
}

void SpecificWorker::compute()
{
	static FPSCounter compute_fps;
	if (scene_processor)
		scene_processor->check_input_stream_startup_status();

	const auto [room_name, robot_name] = scene_processor
		? scene_processor->get_room_robot_names_for_compute()
		: std::pair<std::string, std::string>{};

	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	const auto &[rgbd_opt] = rgbd_buffer.read(now_ms);
	if (!rgbd_opt.has_value())
		return;
	const auto& rgbd = rgbd_opt.value();
	const std::uint64_t frame_ts_ms = scene_processor
		? scene_processor->get_rgbd_frame_timestamp_ms(rgbd)
		: 0;

	const auto detections = yolo_processor ? yolo_processor->detect_segmentation(rgbd) : std::vector<SegDetection>{};
	update_yolo_tab_display(rgbd, detections);

	if (!scene_processor || !scene_processor->ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
		return;

	const auto room_T_robot = scene_processor->get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
	if (!room_T_robot.has_value())
		return;
	const auto room_T_zed = scene_processor->get_room_zed_transform(compute_fps, room_name, frame_ts_ms);
	if (!room_T_zed.has_value())
		return;
	scene_processor->log_room_robot_pose_periodic(room_T_robot.value());
	scene_processor->mark_room_rt_ready();

	scene_processor->update_viewer_robot_pose(room_T_robot.value());
	scene_processor->update_viewer_lidar_points(room_name, robot_name, room_T_robot.value());
	scene_processor->update_room_polygon_periodic();

	if (voxel_processor)
		voxel_processor->process_rgbd_frame(rgbd, detections, room_T_robot.value(), room_T_zed.value(), voxel_viewer_gl.get());

	if (verbose_debug_)
		compute_fps.print("[Compute]", 2000);
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