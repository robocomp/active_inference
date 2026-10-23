#include "bearing_stage.h"

#include <genericworker.h>                 // DSR graph API
#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>
#include <algorithm>
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
    out.looked_azimuth_windows.clear();
    if (!out.masks || in.rgbd.bgr.empty() || !graph_)
        return;

    if (!camera_api_)
        if (const auto rn = graph_->get_node("ricoh"); rn.has_value())
            camera_api_ = graph_->get_camera_api(rn.value());
    if (!camera_api_)
        return;

    // Which azimuths did the detector run on THIS frame? Same column->ray->azimuth chain as the bearings below (graph
    // intrinsics + live az-tune), at the strip edges. Empty strips_looked = every strip. Never a single 2*pi window
    // (the consumer reads from==to as width 0), so a full frame is sent as >= 3 adjacent windows.
    {
        const int cols = in.rgbd.bgr.cols;
        const int n    = std::max(out.n_strips, 1);
        std::vector<std::pair<int, int>> spans;   // [c0,c1) columns
        if (out.strips_looked.empty() or n < 3)
        {
            for (int k = 0; k < 3; ++k)
                spans.emplace_back(k * cols / 3, (k + 1) * cols / 3);
        }
        else
            for (const int st : out.strips_looked)
                if (st >= 0 and st < n)
                    spans.emplace_back(st * cols / n, (st + 1) * cols / n);
        const double row_h = in.rgbd.bgr.rows * 0.5;
        const auto az_at = [&](int col) -> float
        {
            const Eigen::Vector3d r = camera_api_->ray_from_pixel(static_cast<double>(col), row_h);
            const float a = std::atan2(static_cast<float>(r.y()), static_cast<float>(r.x()))
                          + static_cast<float>(in.azimuth_tune_rad);
            return std::atan2(std::sin(a), std::cos(a));
        };
        for (const auto& [c0, c1] : spans)
        {
            float a0 = az_at(c0), a1 = az_at(c1);
            const float d = std::atan2(std::sin(a1 - a0), std::cos(a1 - a0));   // signed width, strip < pi wide
            if (d < 0.0f) std::swap(a0, a1);                                    // column order may run CW: keep from->to CCW
            out.looked_azimuth_windows.insert(out.looked_azimuth_windows.end(), {a0, a1});
        }
    }
    if (out.masks->empty())
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
