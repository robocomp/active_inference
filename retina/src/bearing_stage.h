#pragma once

// Ricoh 360 bearing stage → PerceptionResult::bearings. Runs AFTER the seg stage (reads out.masks) and
// converts each panorama detection to a room-frame azimuth via the ricoh CameraAPI unproject + the
// frame's room_T_sensor (which the RicohSource already resolved at the panorama stamp, incl. az-tune).
// The lidar depth-fill that upgrades these to 3D stays on the main thread (it needs the lidar cloud).

#include "perception_stage.h"

#include <memory>

namespace DSR { class DSRGraph; class CameraAPI; }

namespace rc
{

class BearingStage : public Stage
{
public:
    explicit BearingStage(std::shared_ptr<DSR::DSRGraph> graph);
    ~BearingStage();   // out-of-line: holds a unique_ptr to an incomplete DSR::CameraAPI

    const char* name() const override { return "bearing"; }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

private:
    std::shared_ptr<DSR::DSRGraph>  graph_;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ricoh equirect unproject; cached lazily on the worker thread
};

} // namespace rc
