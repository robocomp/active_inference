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

    // ── Start RGBD reader thread ─────────────────────────────
    rgbd_thread = std::thread(&SpecificWorker::read_rgbd_thread, this);
    qInfo() << __FUNCTION__ << "Started RGBD reader";

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

	// Allocate here so the heavy header remains out of specificworker.h and MOC units.
	voxel_grid = std::make_unique<UnifiedVoxelGrid>();

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

	// read RGBD data from the buffer
	const auto &[rgbd_opt] = rgbd_buffer.read(timestamp);
	if(not rgbd_opt.has_value())
	{ qWarning() << "No RGBD data available at timestamp" << timestamp; return;}
	const auto &rgbd = rgbd_opt.value();

	// Upload to DSR graph
	if (auto cam_node = G->get_node("zed"); cam_node.has_value())
	{
		G->add_or_modify_attrib_local<cam_rgb_att>(cam_node.value(),
			std::vector<uint8_t>(rgbd.image.image.begin(), rgbd.image.image.end()));
		G->add_or_modify_attrib_local<cam_depth_att>(cam_node.value(),
			std::vector<uint8_t>(rgbd.depth.depth.begin(), rgbd.depth.depth.end()));
		G->add_or_modify_attrib_local<cam_rgb_width_att>(cam_node.value(), rgbd.image.width);
		G->add_or_modify_attrib_local<cam_rgb_height_att>(cam_node.value(), rgbd.image.height);
		G->add_or_modify_attrib_local<cam_depth_width_att>(cam_node.value(), rgbd.depth.width);
		G->add_or_modify_attrib_local<cam_depth_height_att>(cam_node.value(), rgbd.depth.height);
		G->add_or_modify_attrib_local<cam_depthFactor_att>(cam_node.value(), rgbd.depth.depthFactor);
		G->update_node(cam_node.value());
	}
	else
		qWarning() << "Camera node not found in DSR graph";

	// Process RGB image through YOLO-seg
	std::vector<SegDetection> detections;
	if (yolo_detector.has_value() && rgbd.image.width > 0 && rgbd.image.height > 0)
	{
		// Reconstruct cv::Mat from the raw byte vector (RGB, 3 channels)
		const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
			const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
		detections = yolo_detector->detect(rgb_frame, /*is_rgb=*/true);
		//draw_detections(rgb_frame, detections);
		// for (const auto& det : detections)
		// 	std::println("Detected {} with confidence {:.3f}", det.label, det.confidence);
	}

	// Update the unified voxel grid: project each RGBD point into 3D,
	// label it with the YOLO class if its pixel falls inside a mask.
	const auto& raw_pts = rgbd.points.points;
	const int   img_w   = rgbd.image.width;
	const int   img_h   = rgbd.image.height;

	if (!raw_pts.empty() && img_w > 0 && img_h > 0
	    && static_cast<int>(raw_pts.size()) == img_w * img_h)
	{
		std::vector<Eigen::Vector3f> pts_eigen;
		std::vector<std::string>     pt_labels;
		std::vector<float>           pt_confs;
		std::size_t                  valid_points = 0;
		std::size_t                  masked_points = 0;
		pts_eigen.reserve(raw_pts.size() / 4);
		pt_labels.reserve(raw_pts.size() / 4);
		pt_confs .reserve(raw_pts.size() / 4);

		for (int row = 0; row < img_h; ++row)
		{
			for (int col = 0; col < img_w; ++col)
			{
				const auto& p = raw_pts[static_cast<std::size_t>(row * img_w + col)];

				// Skip invalid / out-of-range points
				if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
					continue;

				// RoboComp RGBD points may come in mm depending on camera driver.
				// If magnitudes are large, convert mm -> m before filtering/voxelizing.
				const float max_abs = std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)});
				const float unit_scale = (max_abs > 50.0f) ? 0.001f : 1.0f;
				const float px = p.x * unit_scale;
				const float py = p.y * unit_scale;
				const float pz = p.z * unit_scale;

				const float rng_sq = px*px + py*py + pz*pz;
				if (rng_sq < 0.01f || rng_sq > 100.0f) // 0.1 m … 10 m
					continue;

				++valid_points;
				pts_eigen.emplace_back(px, py, pz);

				// Check if this pixel falls inside any YOLO detection mask.
				// Detections are returned highest-confidence first by the model.
				std::string label   = "background";
				float       conf    = 0.0f;
				for (const auto& det : detections)
				{
					if (det.mask.empty()
					    || col >= det.mask.cols || row >= det.mask.rows)
						continue;
					if (det.mask.at<uint8_t>(row, col) > 127)
					{
						label = det.label;
						conf  = det.confidence;
						++masked_points;
						break; // first (highest-confidence) match wins
					}
				}
				pt_labels.push_back(std::move(label));
				pt_confs .push_back(conf);
			}
		}

		if (!pts_eigen.empty() && voxel_grid)
			voxel_grid->observe(/*track_id=*/0,
			                    pts_eigen,
			                    /*category=*/"",
			                    ++compute_frame_,
			                    pt_labels,
			                    pt_confs,
			                    /*detection_confidence=*/1.0f);

		if (compute_frame_ % 30 == 0)
		{
			const float ratio = valid_points > 0
				? (100.0f * static_cast<float>(masked_points) / static_cast<float>(valid_points))
				: 0.0f;
			std::println("[VoxelDebug] valid_pts={} masked_pts={} ({:.1f}%) detections={}",
			             valid_points,
			             masked_points,
			             ratio,
			             detections.size());
		}
	}


	compute_fps.print("[Compute]", 2000);
}

////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::draw_detections(const cv::Mat& rgb_frame,
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

    cv::imshow("YOLO detections", canvas);
    cv::waitKey(1);
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
void SpecificWorker::read_rgbd_thread()
{
    static FPSCounter rgbd_fps;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_rgbd_thread)
    {
        const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
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

            rgbd_buffer.put<0>(
                std::move(frame),
                timestamp,
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



