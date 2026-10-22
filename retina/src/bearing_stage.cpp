#include "bearing_stage.h"

#include <genericworker.h>                 // DSR graph API
#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>
#include <cmath>
#include <utility>

namespace rc
{

BearingStage::BearingStage(std::shared_ptr<DSR::DSRGraph> graph)
    : graph_(std::move(graph))
{
}

BearingStage::~BearingStage() = default;

void BearingStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    // Depends on the seg stage having filled masks; always emit a (possibly empty) bearings vector so a
    // consumer can tell "ran, none" from "didn't run".
    out.bearings = std::vector<BearingDetection>{};
    if (!out.masks || out.masks->empty() || in.rgbd.bgr.empty() || !graph_)
        return;

    if (!camera_api_)
        if (const auto rn = graph_->get_node("ricoh"); rn.has_value())
            camera_api_ = graph_->get_camera_api(rn.value());
    if (!camera_api_)
        return;

    const double row_horizon = in.rgbd.bgr.rows * 0.5;   // horizon row → horizontal bearing
    auto& bearings = *out.bearings;
    bearings.reserve(out.masks->size());
    for (const auto& d : *out.masks)
    {
        const double col_c = static_cast<double>(d.bbox.x) + 0.5 * static_cast<double>(d.bbox.width);
        // panorama column -> ray in the RICOH camera frame -> azimuth. NO room frame: the producer publishes
        // camera-frame bearings and the mask ingestor converts them to the room at ingest. The runtime
        // az-tune is a yaw about the ricoh up axis on top of the graph intrinsics, i.e. it adds to the
        // camera-frame azimuth directly (Rz(t)*ray has azimuth atan2(ray)+t).
        const Eigen::Vector3d ray_cam = camera_api_->ray_from_pixel(col_c, row_horizon);
        float az = std::atan2(static_cast<float>(ray_cam.y()), static_cast<float>(ray_cam.x()))
                 + static_cast<float>(in.azimuth_tune_rad);
        az = std::atan2(std::sin(az), std::cos(az));   // wrap to (-pi, pi]
        bearings.push_back(BearingDetection{d.label, static_cast<float>(d.class_id), d.confidence, az});
    }
}

} // namespace rc
