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
#include "yolo_seg_detector.h"

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
            bool        YOLO_USE_TRT       = false;
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
	void draw_detections(const cv::Mat& rgb_frame, const std::vector<SegDetection>& detections) const;

	
signals:
	//void customSignal();
};

#endif
