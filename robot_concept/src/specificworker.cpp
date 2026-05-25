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

#include <algorithm>
#include <chrono>
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
	qInfo() << "initialize robot_concept worker";
	GenericWorker::initialize();

	try { params.DSR_RGB_FPS   = configLoader.get<int>("Camera.dsr_rgb_fps"); }   catch (...) {}
	try { params.DSR_DEPTH_FPS = configLoader.get<int>("Camera.dsr_depth_fps"); } catch (...) {}
	try { params.DSR_LIDAR_FPS = configLoader.get<int>("Lidar.dsr_lidar_fps"); }  catch (...) {}
	try { params.LIDAR_DECIMATION_FACTOR = configLoader.get<int>("Lidar.decimation_factor"); } catch (...) {}
	try { params.TRANSFORMS_INTERPOLATE_RT = configLoader.get<bool>("Transforms.interpolate_rt"); } catch (...) {}
	try { verbose_debug_ = configLoader.get<bool>("Debug.verbose"); }
	catch (...) { verbose_debug_ = false; }

	imu_thread   = std::thread(&SpecificWorker::read_imu_thread,   this);
	lidar_thread = std::thread(&SpecificWorker::read_lidar_thread,  this);
	rgbd_thread  = std::thread(&SpecificWorker::read_rgbd_thread,   this);
	qInfo() << __FUNCTION__ << "Started IMU, lidar and RGBD reader threads";
}

void SpecificWorker::compute()
{
	// robot_concept's compute() is intentionally minimal:
	// sensor reading is done in background threads; voxelizer agent handles perception.
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

////////////////////////////////////////////////////////////////////////////////////////////////
namespace
{
	std::uint64_t get_imu_timestamp_ms(const RoboCompIMU::DataImu& data)
	{
		const long latest = std::max({data.acc.timestamp,
		                             data.gyro.timestamp,
		                             data.mag.timestamp,
		                             data.rot.timestamp});
		return latest > 0 ? static_cast<std::uint64_t>(latest) : 0ULL;
	}
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

		const auto n = data.points.size();
		std::vector<float> xs(n), ys(n), zs(n);
		for (std::size_t i = 0; i < n; ++i)
		{
			xs[i] = data.points[i].x/1000.f;
			ys[i] = data.points[i].y/1000.f;
			zs[i] = data.points[i].z/1000.f;
		}

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

void SpecificWorker::read_rgbd_thread()
{
	static FPSCounter rgbd_fps;
	bool empty_rgbd_logged = false;
	auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
	while (!stop_rgbd_thread)
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
			const bool empty_points = frame.points.points.empty();
			if (empty_rgb || empty_depth || empty_points)
			{
				if (!empty_rgbd_logged)
				{
					std::print("[read_rgbd] Empty RGBD stream received. Waiting for RGB, depth and point cloud data...\n");
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

			const long p_ms = static_cast<long>(frame.image.period);

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