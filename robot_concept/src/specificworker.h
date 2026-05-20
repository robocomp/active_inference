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

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <doublebuffer_sync/doublebuffer_sync.h>
#include <fps/fps.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "yolo_seg_detector.h"
#include "custom_widget.h"

class UnifiedVoxelGrid;
class VoxelProcessor;
class YoloProcessor;
class SceneProcessor;
namespace rc { class VoxelOpenGLViewer; }
class QLabel;
class QPushButton;

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();

	 /// Lidar buffer: input LidarData and output a tuple of three vectors of floats (x, y, z coordinates) for G uploading to the DSR graph.
    using PointCloud_Buffer = BufferSync<InOut<std::pair<RoboCompLidar3D::TPoints, std::uint64_t>,
		std::tuple<std::uint64_t, std::vector<float>, std::vector<float>, std::vector<float>>>>;

	/// RGBD buffer: input TRGBD frame, output same TRGBD frame (pass-through).
	using RGBD_Buffer = BufferSync<InOut<RoboCompCameraRGBDSimple::TRGBD,
		RoboCompCameraRGBDSimple::TRGBD>>;
                                    

public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	// DSR graph update slots. These slots are called when the corresponding signals are emitted by the DSR graph.
	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     

private:
	bool startup_check_flag;

	  struct Params
        {
            float ROBOT_WIDTH  = 0.460f;   // m
            float ROBOT_LENGTH = 0.480f;   // m
            float ROBOT_HEIGHT = 1.6f;     // m, obstacle cloud ceiling

			  // Timestamped RT queries
			  bool  TRANSFORMS_INTERPOLATE_RT = true;

			  // Lidar
			  int   LIDAR_DECIMATION_FACTOR = 1;

			  // YOLO segmentation
			  std::string YOLO_MODEL_PATH   = "yolo11-seg.onnx";
			  float       YOLO_CONF_THRESH  = 0.25f;
			  float       YOLO_IOU_THRESH   = 0.45f;
			  int         YOLO_INPUT_SIZE   = 640;
			  bool        YOLO_USE_GPU      = true;
			  bool        YOLO_USE_TRT      = true;
			  std::vector<std::string> YOLO_ACCEPTED_LABELS = {
				  "backpack", "handbag", "suitcase",
				  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
				  "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
				  "hot dog", "pizza", "donut", "cake",
				  "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
				  "laptop", "mouse", "remote", "keyboard", "cell phone",
				  "microwave", "oven", "toaster", "sink", "refrigerator",
				  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
			  };
			  int         YOLO_MASK_ERODE_KERNEL = 2; // pixels
			  bool        YOLO_MASK_TRAY = true;
			  int         YOLO_TRAY_MASK_REF_WIDTH = 1280;
			  int         YOLO_TRAY_MASK_REF_HEIGHT = 720;
			  std::vector<cv::Point> YOLO_TRAY_MASK_POLYGON_PX = {
				  {210, 719}, {252, 690}, {320, 662}, {410, 636}, {520, 620},
				  {640, 617}, {760, 620}, {870, 636}, {960, 662}, {1026, 690}, {1068, 719}
			  }; // Estimated tray silhouette in 1280x720 ZED RGB image.

			  // Voxel grid
			  std::size_t VOXEL_DECIMATION_FACTOR = 2;
			  std::size_t VOXEL_VIEWER_MAX_RENDERED_VOXELS = 30'000;
			  int         VOXEL_VIEWER_FPS = 10;

			  // Hungarian association
			  float TRACK_ASSOCIATION_MAX_DISTANCE_M = 0.7f;
			  int   TRACK_MAX_MISSED_FRAMES = 10;

			  // DSR upload rates (Hz)
			  int DSR_RGB_FPS   = 0;   // 0 = every frame (no throttle)
			  int DSR_DEPTH_FPS = 5;
			  int DSR_LIDAR_FPS = 0;   // 0 = every captured scan (no throttle)
        };
        Params params;

	// YOLO-seg detector (constructed in initialize() once model path is known)
	std::optional<YoloSegDetector> yolo_detector;

	// lidar
	PointCloud_Buffer pointcloud_buffer;
	void read_lidar_thread();
	std::thread lidar_thread;	
	std::atomic<bool> stop_lidar_thread{false};

	// IMU
	void read_imu_thread();
	std::thread imu_thread;
	std::atomic<bool> stop_imu_thread{false};

	// RGBD camera
	RGBD_Buffer rgbd_buffer;
	void read_rgbd_thread();
	std::thread rgbd_thread;
	std::atomic<bool> stop_rgbd_thread{false};

	// Visualisation
	void update_yolo_tab_display(const RoboCompCameraRGBDSimple::TRGBD& rgbd, const std::vector<SegDetection>& detections);
	
	// Custom widget for docking in the graph viewer
	Custom_widget custom_widget;
	Custom_widget custom_widget_yolo;
	QPushButton* voxel_lidar_toggle_button_ = nullptr;
	QPushButton* voxel_clear_button_ = nullptr;
	QLabel* yolo_image_label_ = nullptr;
	QLabel* yolo_fps_label_ = nullptr;
	std::unique_ptr<rc::VoxelOpenGLViewer> voxel_viewer_gl;
	std::mutex lidar_points_mutex_;
	std::vector<float> latest_lidar_xs_;
	std::vector<float> latest_lidar_ys_;
	std::vector<float> latest_lidar_zs_;
	std::uint64_t latest_lidar_timestamp_ms_ = 0;

	// Unified voxel grid — scene-level semantic map
	std::unique_ptr<YoloProcessor> yolo_processor;
	std::unique_ptr<UnifiedVoxelGrid> voxel_grid;
	std::unique_ptr<VoxelProcessor> voxel_processor;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api;
	std::unique_ptr<SceneProcessor> scene_processor;
	bool verbose_debug_ = false;
	std::atomic<bool> lidar_stream_seen_{false};
	std::atomic<bool> rgbd_stream_seen_{false};
	std::atomic<bool> lidar_stream_wait_logged_{false};
	std::atomic<bool> rgbd_stream_wait_logged_{false};
	
signals:
	//void customSignal();
};

#endif
