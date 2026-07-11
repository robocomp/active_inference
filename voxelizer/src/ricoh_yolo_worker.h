#pragma once

/*
 * ricoh_yolo_worker.h
 *
 * Runs the ricoh-360 3-strip YOLO pipeline (YoloProcessor::detect_segmentation_360) on its OWN
 * thread, paced to a target period independent of SpecificWorker::compute()'s budget — see
 * RICOH_360_PERIPHERAL_DETECTION.md "Can it be increased?" follow-up. Previously this ran inline,
 * decimated, inside compute(); that traded ricoh's rate directly against the main perception
 * budget. A dedicated thread removes that trade-off: ricoh can run near its own achievable rate
 * (~50ms/cycle for 3 strips) without stealing time from zed/pose/publish.
 *
 * Owns its OWN YoloProcessor (own ONNX session), never the zed one — avoids relying on the
 * execution provider's concurrent-Run()-on-one-session thread-safety (a real guarantee for CUDA/CPU,
 * but a needless risk to take with TensorRT when a second session is this cheap to just create).
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo_processor.h"
#include "graph_publisher.h"   // BearingDetection

class SceneProcessor;
namespace DSR { class DSRGraph; class InnerEigenAPI; class CameraAPI; }

namespace rc
{

class RicohYoloWorker
{
public:
    struct Config
    {
        YoloProcessor::Config yolo_config;
        Detection360Config    detect_config;
        int                   target_period_ms = 50;   // ~20 Hz
        bool                  perf_log = false;         // -> etc/viewer_perf_ricoh_yolo.csv
        // [thread-safety test, step 1] compute the room-frame bearing for each detection ON THIS THREAD
        // (own InnerEigenAPI, room<-ricoh at the panorama capture stamp ⇒ ts!=0 ⇒ no InnerEigenAPI cache
        // touched ⇒ thread-safe). Bearing-only for now; the LiDAR depth-fill stays on the main thread.
        bool                  publish_bearings = false;   // Ricoh.publish_masks
        float                 azimuth_tune_deg = 0.0f;    // Ricoh.azimuth_tune_deg (applied on top of graph intrinsics)
    };

    ~RicohYoloWorker();

    // scene_processor must outlive the worker (both owned by the same SpecificWorker; stopped
    // explicitly in request_shutdown() before scene_processor is torn down — see there). Returns
    // false (no thread started) if the ricoh media plane isn't up yet, or the model fails to load.
    bool start(SceneProcessor* scene_processor, const Config& config);
    void stop();

    // Thread-safe snapshots for the GUI thread (on_render_tick). A cv::Mat copy is a cheap shallow
    // (refcounted) copy, not a pixel copy.
    cv::Mat latest_bgr() const;
    std::vector<SegDetection> latest_detections() const;
    // Room-frame bearings for the latest detections (computed on the worker thread; empty if
    // publish_bearings is off or the room<-ricoh transform couldn't be resolved this cycle).
    std::vector<BearingDetection> latest_bearings() const;
    // Detections + their bearings under ONE lock (consistent 1:1 snapshot — sizes match unless the
    // transform failed, in which case bearings is empty). Used by the main-thread LiDAR depth-fill.
    void latest_detections_and_bearings(std::vector<SegDetection>& dets,
                                        std::vector<BearingDetection>& bearings) const;

private:
    void run();
    // Convert panorama-pixel detections to room-frame bearings via the ricoh CameraAPI + room<-ricoh at
    // `stamp`. Runs ON the worker thread — this is the DSR-access-from-a-worker experiment (step 1).
    std::vector<BearingDetection> compute_bearings(const std::vector<SegDetection>& dets,
                                                   const cv::Mat& pano_rgb, std::uint64_t stamp);

    SceneProcessor* scene_processor_ = nullptr;
    Config config_;
    YoloProcessor yolo_;

    // DSR access owned by this worker (see compute_bearings). inner_eigen_ is this thread's OWN instance
    // (created on the main thread in start()); used only with real timestamps ⇒ no ts==0 cache races.
    std::shared_ptr<DSR::DSRGraph>       graph_;
    std::unique_ptr<DSR::InnerEigenAPI>  inner_eigen_;
    std::unique_ptr<DSR::CameraAPI>      ricoh_camera_api_;   // cached lazily on the worker thread

    std::thread thread_;
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex result_mutex_;
    cv::Mat latest_bgr_;
    std::vector<SegDetection> latest_detections_;
    std::vector<BearingDetection> latest_bearings_;
};

} // namespace rc
