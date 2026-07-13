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
    if (!out.masks || out.masks->empty() || in.rgbd.rgb.empty() || !graph_)
        return;

    if (!camera_api_)
        if (const auto rn = graph_->get_node("ricoh"); rn.has_value())
            camera_api_ = graph_->get_camera_api(rn.value());
    if (!camera_api_)
        return;

    const double row_horizon = in.rgbd.rgb.rows * 0.5;   // horizon row → horizontal bearing
    auto& bearings = *out.bearings;
    bearings.reserve(out.masks->size());
    for (const auto& d : *out.masks)
    {
        const double col_c = static_cast<double>(d.bbox.x) + 0.5 * static_cast<double>(d.bbox.width);
        // panorama column → ray in the ricoh frame → room frame → azimuth. room_T_sensor was resolved at
        // the panorama stamp (with az-tune) by the RicohSource, so no knobs here.
        const Eigen::Vector3d ray_room = in.room_T_sensor.linear() * camera_api_->ray_from_pixel(col_c, row_horizon);
        const float az = std::atan2(static_cast<float>(ray_room.y()), static_cast<float>(ray_room.x()));
        bearings.push_back(BearingDetection{d.label, static_cast<float>(d.class_id), d.confidence, az});
    }
}

} // namespace rc
