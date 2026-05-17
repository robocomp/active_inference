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
#include <memory>
#include "yolo_seg_detector.h"
#include "custom_widget.h"

class UnifiedVoxelGrid;
namespace rc { class VoxelOpenGLViewer; }
class QLabel;

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

			  // Lidar
			  int   LIDAR_DECIMATION_FACTOR = 1;

			  // YOLO segmentation
			  std::string YOLO_MODEL_PATH   = "yolo11-seg.onnx";
			  float       YOLO_CONF_THRESH  = 0.25f;
			  float       YOLO_IOU_THRESH   = 0.45f;
			  int         YOLO_INPUT_SIZE   = 640;
			  bool        YOLO_USE_GPU      = true;
			  bool        YOLO_USE_TRT      = true;
			  int         YOLO_MASK_ERODE_KERNEL = 2; // pixels

			  // Voxel grid
			  std::size_t VOXEL_DECIMATION_FACTOR = 2;

			  // Hungarian association
			  float TRACK_ASSOCIATION_MAX_DISTANCE_M = 0.7f;
			  int   TRACK_MAX_MISSED_FRAMES = 10;
        };
        Params params;

	// YOLO-seg detector (constructed in initialize() once model path is known)
	std::optional<YoloSegDetector> yolo_detector;

	// lidar
	PointCloud_Buffer pointcloud_buffer;
	void read_lidar_thread();
	std::thread lidar_thread;	
	std::atomic<bool> stop_lidar_thread{false};

	// RGBD camera
	RGBD_Buffer rgbd_buffer;
	void read_rgbd_thread();
	std::thread rgbd_thread;
	std::atomic<bool> stop_rgbd_thread{false};

	// Visualisation
	struct VoxelSelectionResult
	{
		std::vector<Eigen::Vector3f> points;
		std::vector<std::string> labels;
		std::vector<float> confidences;
		std::size_t valid_points = 0;
		std::size_t masked_points = 0;
		std::size_t selected_points = 0;
		std::size_t table_points = 0;
		std::size_t chair_points = 0;
		std::size_t monitor_points = 0;
	};

	struct DetectionObservation
	{
		std::size_t det_index = 0;
		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		std::string label;
		float confidence = 0.0f;
	};

	struct InstanceTrack
	{
		int id = -1;
		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		std::string label;
		int last_seen_frame = -1;
	};

	void draw_detections(const cv::Mat& rgb_frame, const std::vector<SegDetection>& detections) const;
	cv::Mat compose_detection_canvas(const cv::Mat& rgb_frame, const std::vector<SegDetection>& detections) const;
	void update_yolo_tab_views(const cv::Mat& rgb_frame_rgb, const cv::Mat& yolo_canvas_bgr);
	bool ensure_room_and_robot_ready(FPSCounter& compute_fps);
	std::optional<Mat::RTMat> get_room_robot_transform(FPSCounter& compute_fps);
	void log_room_robot_pose_periodic(const Mat::RTMat& room_T_robot) const;
	void update_room_polygon_periodic();
	std::vector<SegDetection> detect_segmentation(const RoboCompCameraRGBDSimple::TRGBD& rgbd);
	void postprocess_yolo_detections(std::vector<SegDetection>& detections) const;
	bool is_target_label(const std::string& label) const;
	float detect_point_scale_once(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const;
	void build_owner_map_and_medians(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                float point_scale,
	                                const std::vector<SegDetection>& detections,
	                                std::vector<int32_t>& pixel_owner,
	                                std::vector<float>& det_median_range_m) const;
	VoxelSelectionResult collect_points_parallel(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                            float point_scale,
	                                            const std::vector<int32_t>& pixel_owner,
	                                            const std::vector<SegDetection>& detections,
	                                            const std::vector<float>& det_median_range_m) const;
	std::vector<int> hungarian_min_cost(const std::vector<std::vector<float>>& cost) const;
	std::vector<int> associate_detections_hungarian(const std::vector<DetectionObservation>& observations,
	                                                int frame_id);
	void prune_stale_tracks(int frame_id);
	void update_voxel_grid_from_rgbd(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                const std::vector<SegDetection>& detections,
	                                const Mat::RTMat& room_T_robot);

	void update_room_polygon_in_viewers();
	
	// Custom widget for docking in the graph viewer
	Custom_widget custom_widget;
	Custom_widget custom_widget_yolo;
	QLabel* yolo_image_label_ = nullptr;
	std::unique_ptr<rc::VoxelOpenGLViewer> voxel_viewer_gl;

	// Unified voxel grid — scene-level semantic map
	std::unique_ptr<UnifiedVoxelGrid> voxel_grid;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api;
	int compute_frame_ = 0;
	int next_track_id_ = 1;
	std::unordered_map<int, InstanceTrack> active_tracks;
	// Hungarian association parameters (now set from params)
	float track_association_max_distance_m = 0.7f;
	int track_max_missed_frames = 10;
	bool room_ready_logged_ = false;
	bool room_wait_logged_ = false;
	bool room_rt_ready_logged_ = false;
	bool room_rt_wait_logged_ = false;
	std::string room_node_name_;
	std::string robot_node_name_;
	
signals:
	//void customSignal();
};

#endif
