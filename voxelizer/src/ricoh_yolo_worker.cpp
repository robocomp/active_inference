#include "ricoh_yolo_worker.h"
#include "scene_processor.h"

#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <print>

namespace rc
{

RicohYoloWorker::~RicohYoloWorker()
{
    stop();
}

bool RicohYoloWorker::start(SceneProcessor* scene_processor, const Config& config)
{
    if (!scene_processor || !scene_processor->ricoh_available())
        return false;

    try
    {
        yolo_.configure(config.yolo_config);
    }
    catch (const std::exception& e)
    {
        std::println("[RicohYoloWorker] disabled — failed to load model {}: {}",
                     config.yolo_config.model_path, e.what());
        return false;
    }

    scene_processor_ = scene_processor;
    config_ = config;

    // Own InnerEigenAPI instance for this worker (created here, on the MAIN thread). We only ever call
    // get_transformation_matrix with a real capture timestamp (ts!=0 ⇒ use_cache=false), so the unlocked
    // ts==0 cache is never touched — safe to use from the worker thread. See CLAUDE.md graph thread-safety.
    if (config_.publish_bearings)
    {
        graph_ = scene_processor->graph();
        if (graph_)
            inner_eigen_ = graph_->get_inner_eigen_api();
    }

    stop_requested_.store(false, std::memory_order_relaxed);
    thread_ = std::thread(&RicohYoloWorker::run, this);
    return true;
}

void RicohYoloWorker::stop()
{
    stop_requested_.store(true, std::memory_order_relaxed);
    if (thread_.joinable())
        thread_.join();
}

void RicohYoloWorker::run()
{
    const auto perf_log_start = std::chrono::steady_clock::now();
    std::ofstream perf_csv;
    if (config_.perf_log)
    {
        perf_csv.open("etc/viewer_perf_ricoh_yolo.csv", std::ios::trunc);
        perf_csv << "t_ms,cycle_ms,det_count\n";
    }

    std::uint64_t last_stamp = 0;      // source stamp of the last panorama we actually ran YOLO on
    std::size_t   last_det_count = 0;  // held for the perf CSV on skipped (unchanged-frame) cycles

    while (!stop_requested_.load(std::memory_order_relaxed))
    {
        const auto t0 = std::chrono::steady_clock::now();

        scene_processor_->poll_ricoh(/*force=*/true);
        // Skip inference when the panorama has NOT advanced since we last ran (idle scene / producer
        // slower than our poll). The source stamp is updated on every dequeued frame; unchanged stamp ⇒
        // same image ⇒ re-running the 3-strip YOLO would just reproduce the cached detections. A
        // stamp of 0 means the producer doesn't timestamp — fall back to always-process (old behaviour).
        const std::uint64_t stamp = scene_processor_->ricoh_last_stamp_ms();
        const bool fresh = (stamp == 0) or (stamp != last_stamp);
        std::size_t det_count = last_det_count;
        if (fresh)
        {
            cv::Mat pano_bgr = scene_processor_->ricoh_bgr_copy();
            if (!pano_bgr.empty())
            {
                last_stamp = stamp;
                cv::Mat pano_rgb;
                cv::cvtColor(pano_bgr, pano_rgb, cv::COLOR_BGR2RGB);   // detect() wants RGB, cache is BGR
                auto dets = yolo_.detect_segmentation_360(pano_rgb, config_.detect_config);
                det_count = last_det_count = dets.size();

                // DSR access ON THIS THREAD (step-1 experiment): panorama pixels → room-frame bearings.
                auto bearings = compute_bearings(dets, pano_rgb, stamp);

                std::scoped_lock lk(result_mutex_);
                latest_bgr_ = std::move(pano_bgr);
                latest_detections_ = std::move(dets);
                latest_bearings_ = std::move(bearings);
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const double cycle_ms = std::chrono::duration<double, std::milli>(now - t0).count();

        if (perf_csv.is_open())
        {
            const long long t_ms = static_cast<long long>(
                std::chrono::duration<double, std::milli>(now - perf_log_start).count());
            perf_csv << t_ms << ',' << cycle_ms << ',' << det_count << '\n';
            perf_csv.flush();
        }

        const auto target = std::chrono::milliseconds(std::max(1, config_.target_period_ms));
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        if (elapsed < target)
            std::this_thread::sleep_for(target - elapsed);
    }
}

cv::Mat RicohYoloWorker::latest_bgr() const
{
    std::scoped_lock lk(result_mutex_);
    return latest_bgr_;
}

std::vector<SegDetection> RicohYoloWorker::latest_detections() const
{
    std::scoped_lock lk(result_mutex_);
    return latest_detections_;
}

std::vector<BearingDetection> RicohYoloWorker::latest_bearings() const
{
    std::scoped_lock lk(result_mutex_);
    return latest_bearings_;
}

void RicohYoloWorker::latest_detections_and_bearings(std::vector<SegDetection>& dets,
                                                     std::vector<BearingDetection>& bearings) const
{
    std::scoped_lock lk(result_mutex_);
    dets = latest_detections_;
    bearings = latest_bearings_;
}

// Runs on the worker thread. Every DSR call here is either mutex-protected (get_node / get_nodes_by_type
// / get_camera_api) or a real-timestamp transform (ts!=0 ⇒ InnerEigenAPI cache untouched) — so this is
// the concrete test of "graph reads from a worker thread are safe" (CLAUDE.md graph thread-safety).
std::vector<BearingDetection> RicohYoloWorker::compute_bearings(const std::vector<SegDetection>& dets,
                                                               const cv::Mat& pano_rgb, std::uint64_t stamp)
{
    std::vector<BearingDetection> bearings;
    if (not inner_eigen_ or not graph_ or dets.empty() or pano_rgb.empty() or stamp == 0)
        return bearings;

    // Cache the ricoh CameraAPI once (equirectangular; reads cam_equirect_* intrinsics off the node).
    if (not ricoh_camera_api_)
        if (const auto rn = graph_->get_node("ricoh"); rn.has_value())
            ricoh_camera_api_ = graph_->get_camera_api(rn.value());
    if (not ricoh_camera_api_)
        return bearings;

    std::string room_name;
    if (const auto rooms = graph_->get_nodes_by_type("room"); not rooms.empty())
        room_name = rooms.front().name();
    if (room_name.empty())
        return bearings;

    // room<-ricoh at the PANORAMA capture stamp. ts!=0 ⇒ use_cache=false ⇒ no shared-cache mutation.
    auto room_T_ricoh = inner_eigen_->get_transformation_matrix(room_name, "ricoh", stamp);
    if (not room_T_ricoh.has_value())
        return bearings;
    Mat::RTMat T = room_T_ricoh.value();
    if (config_.azimuth_tune_deg != 0.0f)   // extra yaw on top of the graph intrinsics (same as the main path)
        T.rotate(Eigen::AngleAxisd(config_.azimuth_tune_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()));

    const double row_horizon = pano_rgb.rows * 0.5;   // horizon row → horizontal bearing
    bearings.reserve(dets.size());
    for (const auto& d : dets)
    {
        const double col_c = static_cast<double>(d.bbox.x) + 0.5 * static_cast<double>(d.bbox.width);
        const Eigen::Vector3d ray_room = T.linear() * ricoh_camera_api_->ray_from_pixel(col_c, row_horizon);
        const float az = std::atan2(static_cast<float>(ray_room.y()), static_cast<float>(ray_room.x()));
        bearings.push_back(BearingDetection{d.label, static_cast<float>(d.class_id), d.confidence, az});
    }
    return bearings;
}

} // namespace rc
