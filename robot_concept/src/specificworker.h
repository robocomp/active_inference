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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
	~SpecificWorker();

public slots:
	void initialize();
	void compute();
	void emergency();
	void restore();
	int  startup_check();

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
		float ROBOT_WIDTH  = 0.460f;
		float ROBOT_LENGTH = 0.480f;
		float ROBOT_HEIGHT = 1.6f;

		bool  TRANSFORMS_INTERPOLATE_RT = true;
		int   LIDAR_DECIMATION_FACTOR   = 1;

		// DSR upload rates (Hz); 0 = every frame
		int DSR_RGB_FPS   = 0;
		int DSR_DEPTH_FPS = 5;
		int DSR_LIDAR_FPS = 0;
	};
	Params params;

	bool verbose_debug_ = false;

	// Lidar reader thread
	void read_lidar_thread();
	std::thread lidar_thread;
	std::atomic<bool> stop_lidar_thread{false};

	// IMU reader thread
	void read_imu_thread();
	std::thread imu_thread;
	std::atomic<bool> stop_imu_thread{false};

	// RGBD camera reader thread
	void read_rgbd_thread();
	std::thread rgbd_thread;
	std::atomic<bool> stop_rgbd_thread{false};

signals:
	//void customSignal();
};

#endif

