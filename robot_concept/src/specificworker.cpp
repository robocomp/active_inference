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
#include "voxel_viewer_3d.h"
#include "voxel_opengl_viewer.h"
#include <QVBoxLayout>
#include <print>
#include <limits>
#include <algorithm>
#include <cctype>
#include <execution>
#include <numeric>
#include <iterator>
#include <unordered_map>
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
	
	if (!graph_viewers.empty())
	{
		const std::string viewer_key = graph_viewers.contains("")
			? std::string("")
			: graph_viewers.begin()->first;
		graph_viewers.at(viewer_key)->add_custom_widget_to_dock("Voxel3D", &custom_widget);

		if (custom_widget.frame->layout() == nullptr)
		{
			auto* layout = new QVBoxLayout(custom_widget.frame);
			layout->setContentsMargins(0, 0, 0, 0);
			custom_widget.frame->setLayout(layout);
		}

		voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(custom_widget.frame);
		custom_widget.frame->layout()->addWidget(voxel_viewer_gl.get());
		qInfo() << __FUNCTION__ << "Voxel OpenGL custom widget attached to graph viewer";
	}
	else
		qWarning() << __FUNCTION__ << "No graph viewer available; Voxel3D widget not attached";

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

	// read RGBD data from the buffer
	const auto &[rgbd_opt] = rgbd_buffer.read(timestamp);
	if(not rgbd_opt.has_value())
	{ qWarning() << "No RGBD data available at timestamp" << timestamp; return;}
	const auto &rgbd = rgbd_opt.value();

	// Process RGB image through YOLO-seg
	std::vector<SegDetection> detections;
	if (yolo_detector.has_value() && rgbd.image.width > 0 && rgbd.image.height > 0)
	{
		// Reconstruct cv::Mat from the raw byte vector (RGB, 3 channels)
		const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
			const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
		detections = yolo_detector->detect(rgb_frame, /*is_rgb=*/true);
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
				cv::erode(det.mask, eroded, cv::Mat(), cv::Point(-1, -1), 1);
				det.mask = eroded;
			}
		}
		//draw_detections(rgb_frame, detections);
		// for (const auto& det : detections)
		// 	std::println("Detected {} with confidence {:.3f}", det.label, det.confidence);
	}

	update_voxel_grid_from_rgbd(rgbd, detections);


	compute_fps.print("[Compute]", 2000);
}

void SpecificWorker::update_voxel_grid_from_rgbd(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                              const std::vector<SegDetection>& detections)
{
	// Update the unified voxel grid: project each RGBD point into 3D,
	// label it with the YOLO class if its pixel falls inside a mask.
	const auto& raw_pts = rgbd.points.points;
	const int   img_w   = rgbd.image.width;
	const int   img_h   = rgbd.image.height;

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
		auto selected = collect_points_parallel(rgbd, point_scale, pixel_owner, detections, det_median_range_m);

		auto& pts_eigen = selected.points;
		auto& pt_labels = selected.labels;
		auto& pt_confs = selected.confidences;
		valid_points = selected.valid_points;
		masked_points = selected.masked_points;
		selected_points = selected.selected_points;
		selected_by_class["table"] = selected.table_points;
		selected_by_class["chair"] = selected.chair_points;
		selected_by_class["monitor"] = selected.monitor_points;

		if (!pts_eigen.empty() && voxel_grid)
		{
			const std::size_t voxel_decimation_step = std::max<std::size_t>(1, params.VOXEL_DECIMATION_FACTOR);
			std::vector<Eigen::Vector3f> pts_decimated;
			std::vector<std::string>     labels_decimated;
			std::vector<float>           confs_decimated;

			pts_decimated.reserve((pts_eigen.size() + voxel_decimation_step - 1) / voxel_decimation_step);
			labels_decimated.reserve(pts_decimated.capacity());
			confs_decimated.reserve(pts_decimated.capacity());

			for (std::size_t i = 0; i < pts_eigen.size(); i += voxel_decimation_step)
			{
				pts_decimated.push_back(pts_eigen[i]);
				labels_decimated.push_back(pt_labels[i]);
				confs_decimated.push_back(pt_confs[i]);
			}

			decimated_points = pts_decimated.size();
			voxel_grid->observe(/*track_id=*/0,
			                    pts_decimated,
			                    /*category=*/"",
			                    ++compute_frame_,
			                    labels_decimated,
			                    confs_decimated,
			                    /*detection_confidence=*/1.0f);
		}

		if (voxel_grid && (compute_frame_ % 8 == 0))
		{
			const auto sem = voxel_grid->export_semantic_voxels();
			std::vector<QVector3D> qpts;
			qpts.reserve(sem.points.size());
			for (const auto& p : sem.points)
				qpts.emplace_back(p.x(), p.y(), p.z());

			std::println("[VoxelViewer] Exported {} points to viewer", qpts.size());
			if (voxel_viewer_gl)
				voxel_viewer_gl->update_voxels(qpts, sem.categories, sem.probs);
			else if (voxel_viewer_3d)
				voxel_viewer_3d->update_voxels(qpts, sem.categories, sem.probs);
		}

		if (compute_frame_ % 30 == 0)
		{
			const float ratio = valid_points > 0
				? (100.0f * static_cast<float>(masked_points) / static_cast<float>(valid_points))
				: 0.0f;
			std::println("[VoxelDebug] valid_pts={} masked_pts={} ({:.1f}%) selected_pts={} decimated_pts={} table={} chair={} monitor={} detections={}",
			             valid_points,
			             masked_points,
			             ratio,
			             selected_points,
			             decimated_points,
			             selected_by_class["table"],
			             selected_by_class["chair"],
			             selected_by_class["monitor"],
			             detections.size());
		}
	}
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
 
			// Upload to DSR graph
			if (auto cam_node = G->get_node("zed"); cam_node.has_value())
			{
				G->add_or_modify_attrib_local<cam_rgb_att>(cam_node.value(),
					std::vector<uint8_t>(frame.image.image.begin(), frame.image.image.end()));
				G->add_or_modify_attrib_local<cam_depth_att>(cam_node.value(),
					std::vector<uint8_t>(frame.depth.depth.begin(), frame.depth.depth.end()));
				G->add_or_modify_attrib_local<cam_rgb_width_att>(cam_node.value(), frame.image.width);
				G->add_or_modify_attrib_local<cam_rgb_height_att>(cam_node.value(), frame.image.height);
				G->add_or_modify_attrib_local<cam_depth_width_att>(cam_node.value(), frame.depth.width);
				G->add_or_modify_attrib_local<cam_depth_height_att>(cam_node.value(), frame.depth.height);
				G->add_or_modify_attrib_local<cam_depthFactor_att>(cam_node.value(), frame.depth.depthFactor);
				G->update_node(cam_node.value());
			}
			else
				qWarning() << "Camera node not found in DSR graph";

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



