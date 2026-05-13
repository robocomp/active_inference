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

	 // ── Start lidar reader thread ────────────────────────────
    lidar_thread = std::thread(&SpecificWorker::read_lidar_thread, this);
    qInfo() << __FUNCTION__ << "Started lidar reader";


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

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}


void SpecificWorker::compute()
{
	static FPSCounter compute_fps;
    
	const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
	// read pointcloud data from the buffer and upload it to the DSR graph as attributes of the lidar3d node
	const auto &[data_opt] = pointcloud_buffer.read(timestamp);
	if(not data_opt.has_value())
	{ qWarning() << "No pointcloud data available at timestamp" << timestamp; return;}
	const auto &[ts, xs, ys, zs] = data_opt.value();
	std::cout << " " <<xs.size()*3*4 << "  uploaded to DSR graph at timestamp " << timestamp << std::endl;
	// Upload to DSR graph
	 if (auto laser_node = G->get_node("lidar3D"); laser_node.has_value())
	 {
	 	G->add_or_modify_attrib_local<laser_X_att>(laser_node.value(), xs);
	 	G->add_or_modify_attrib_local<laser_Y_att>(laser_node.value(), ys);
	 	G->add_or_modify_attrib_local<laser_Z_att>(laser_node.value(), zs);
	 	G->add_or_modify_attrib_local<laser_timestamp_att>(laser_node.value(), static_cast<uint64_t>(timestamp));
	 	G->update_node(laser_node.value());
	 }
	 else
	 	qWarning() << "Laser node not found in DSR graph";
	compute_fps.print("[Compute]", 2000);
}

////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_lidar_thread()
{
	static FPSCounter lidar_fps;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_lidar_thread)
    {
        const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
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

			pointcloud_buffer.put<0>(
				std::make_pair(std::move(data.points), static_cast<std::uint64_t>(data.timestamp)),
				timestamp,
				[](auto &&input, auto &output)
				{
					auto &&[points, lidar_ts] = input;
					auto &[ts, xs, ys, zs] = output;
					ts = lidar_ts;
					const auto n = points.size();
					xs.resize(n);
					ys.resize(n);
					zs.resize(n);
					for (std::size_t i = 0; i < n; ++i)
					{
						xs[i] = points[i].x;
						ys[i] = points[i].y;
						zs[i] = points[i].z;
					}
				});
        
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



