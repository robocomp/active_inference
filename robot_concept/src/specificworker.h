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

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <fps/fps.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "sensor_media_publisher.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
	~SpecificWorker();

	void FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose);
	bool is_shutting_down() const noexcept { return shutting_down_.load(); }

public slots:
	void initialize();
	void compute();
	void emergency();
	void restore();
	int  startup_check();

	void modify_node_slot(std::uint64_t, const std::string&){}
	void modify_node_attrs_slot(std::uint64_t, const std::vector<std::string>&){}
	void modify_edge_slot(std::uint64_t, std::uint64_t, const std::string&){}
	void modify_edge_attrs_slot(std::uint64_t, std::uint64_t, const std::string&, const std::vector<std::string>&){}
	void del_edge_slot(std::uint64_t, std::uint64_t, const std::string&){}
	void del_node_slot(std::uint64_t){}

private:
	bool startup_check_flag;
	AgentPresenceCoordinator presence_coordinator_;
	bool owned_nodes_cleaned_ = false;

	struct Params
	{
		float ROBOT_WIDTH  = 0.460f;
		float ROBOT_LENGTH = 0.480f;
		float ROBOT_HEIGHT = 1.6f;

		bool  TRANSFORMS_INTERPOLATE_RT = true;
		int   LIDAR_DECIMATION_FACTOR   = 1;

		// DSR upload rates (Hz):
		//   >0: throttle to that rate
		//    0: upload every captured frame
		//   <0: disable upload
		int DSR_RGB_FPS   = 0;
		int DSR_DEPTH_FPS = 5;
		// Legacy DSR-graph lidar upload rate. <0 DISABLES it (default): lidar now flows over the media
		// plane (SensorMediaPublisher LidarFrame). Set >=0 to re-enable the graph write at that fps
		// (0 = unthrottled) for consumers still reading laser_* (e.g. controller_obstacle_tracker).
		int DSR_LIDAR_FPS = -1;

		// Media plane (zero-copy DDS) for raw sensor streams carried OUT of the graph.
		// Per-sensor gates: when false, the stream's media publisher is not created,
		// its descriptor is not advertised, and its Ice reader thread is not started.
		bool        ENABLE_ZED    = true;   // ZED RGB + depth  (rc/zed/rgb, rc/zed/depth)
		bool        ENABLE_RICOH  = true;   // RGBD_360 panorama (rc/ricoh/rgb)
		bool        ENABLE_LIDAR  = true;   // 3D LiDAR cloud    (rc/lidar3d/points)
		bool        ENABLE_IMU    = true;   // IMU sample        (rc/imu/data)
		int         MEDIA_DOMAIN_ID   = 0;
		std::string MEDIA_RGB_TOPIC   = "rc/zed/rgb";
		std::string MEDIA_DEPTH_TOPIC = "rc/zed/depth";
		std::string MEDIA_RICOH_TOPIC = "rc/ricoh/rgb";
		std::string MEDIA_LIDAR_TOPIC = "rc/lidar3d/points";
		std::string MEDIA_IMU_TOPIC   = "rc/imu/data";
	};
	Params params;

	bool verbose_debug_ = false;
	std::string robot_name = "Shadow";

	// Lidar reader thread
	void read_lidar_thread();
	std::thread lidar_thread;
	std::atomic<bool> stop_lidar_thread{false};

	// IMU reader thread
	void read_imu_thread();
	std::thread imu_thread;
	std::atomic<bool> stop_imu_thread{false};

	// Per-thread frame counters (relaxed); compute() turns deltas into a 3 Hz
	// [RGBDThread]/[LidarThread]/[IMUThread] Hz readout without needing Verbose.
	std::atomic<std::uint64_t> rgbd_frames_{0};
	std::atomic<std::uint64_t> lidar_frames_{0};
	std::atomic<std::uint64_t> imu_frames_{0};
	std::atomic<std::uint64_t> ricoh_frames_{0};

	// RGBD camera reader thread
	void read_rgbd_thread();
	std::thread rgbd_thread;
	std::atomic<bool> stop_rgbd_thread{false};

	// Ricoh 360 camera reader thread: pulls the RGBD_360 panorama over Ice
	// (Camera360RGBD) and republishes the RGB image on the media plane. Bridges
	// Ice → DDS so the hardware-facing RGBD_360 stays DDS-free.
	void read_ricoh_thread();
	std::thread ricoh_thread;
	std::atomic<bool> stop_ricoh_thread{false};

	// Media plane (zero-copy DDS); RGBD pixels leave the DSR graph here.
	SensorMediaPublisher media_;

	void waiting_enter();
	void waiting_loop();
	void operating_enter();
	void operating_loop();
	void degraded_enter();
	void degraded_loop();
	void request_shutdown();

	void cleanup_owned_nodes();
	void on_optional_peer_lost(const std::string &name, std::uint32_t id);
	void on_optional_peer_ready(const std::string &name, std::uint32_t id);
	std::atomic<bool> shutting_down_{false};

signals:
	void presenceReady();
	void presenceLost();
};

#endif

