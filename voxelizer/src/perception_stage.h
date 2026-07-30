#pragma once

// Perception-stage abstraction. A camera source produces a PerceptionFrame; a PerceptionWorker (its own
// thread) runs a configurable list of Stages on it, each filling its typed slot in a PerceptionResult.
// The result carries the frame it was computed on, so every published map pairs with its own depth /
// transform / stamp (no skew). Adding a new map = add a Stage + a PerceptionResult slot + a Publisher
// route — all localized. See RGBD_360_PERIPHERAL_DETECTION.md / the worker-thread plan.
//
// Step 1: the Stage interface + the three concrete stages (seg/pose/semantic) wrapping the existing
// processors. Not yet wired into a worker — that's step 2 (ZedYoloWorker → PerceptionWorker[stages]).

#include <genericworker.h>          // Mat::RTMat
#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

#include "rgbd_data.h"              // RGBDData
#include "yolo_processor.h"         // SegDetection
#include "yolo_human.h"            // rc::human_pose::PoseDetection
#include "yolo_semantic.h"        // rc::semantic::SemanticMap
#include "graph_publisher.h"      // BearingDetection (ricoh 360 bearing channel)

namespace rc
{

// One camera frame ready for inference AND self-consistent publishing. Stages run on rgbd.bgr (the ZED
// rgb or the ricoh panorama); publish uses rgbd.depth + intrinsics.
struct PerceptionFrame
{
    RGBDData       rgbd;                                  // bgr + depth + intrinsics
    Mat::RTMat     room_T_sensor = Mat::RTMat::Identity();// transform pinned to `stamp`
    std::uint64_t  stamp = 0;                             // acquisition stamp
    bool           is_360 = false;                        // equirect source → 3-strip seg / bearing publish
};

// Everything a worker produced for one frame. One optional slot per map type; *_fresh marks whether the
// model actually ran this frame (vs a held-last cache) so decimated publishes can gate on it.
struct PerceptionResult
{
    PerceptionFrame frame;                               // the frame these results belong to

    std::optional<std::vector<SegDetection>>                  masks;
    std::optional<std::vector<rc::human_pose::PoseDetection>> poses;
    bool                                                      poses_fresh = false;
    std::optional<rc::semantic::SemanticMap>                  semantic;
    bool                                                      semantic_fresh = false;
    std::optional<std::vector<BearingDetection>>              bearings;   // ricoh: room-frame bearing per mask
    std::optional<std::vector<SegDetection>>                  refined_masks;   // SAM2-sharpened masks (subset of `masks`)
    bool                                                      refined_fresh = false;
};

// One inference capability. Owns its own ONNX session. run() is called every frame on the worker thread
// and fills this stage's slot in `out` (fresh inference, or a held-last cache on decimated cycles).
class Stage
{
public:
    virtual ~Stage() = default;
    virtual const char* name() const = 0;
    virtual bool ready() const { return true; }
    virtual void run(const PerceptionFrame& in, PerceptionResult& out) = 0;

    // Runtime gate (e.g. the semantic viewer toggle): a disabled stage is skipped entirely, so the model
    // does NO work. Toggled from the main/UI thread, read on the worker thread — atomic.
    void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> enabled_{true};
};

} // namespace rc
