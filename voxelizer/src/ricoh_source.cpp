#include "ricoh_source.h"
#include "scene_processor.h"

#include <genericworker.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <cmath>
#include <string>
#include <utility>

namespace rc
{

RicohSource::RicohSource(SceneProcessor* scene, std::shared_ptr<DSR::DSRGraph> graph, float azimuth_tune_deg)
    : scene_(scene), graph_(std::move(graph)), azimuth_tune_deg_(azimuth_tune_deg)
{
    if (graph_)
        inner_eigen_ = graph_->get_inner_eigen_api();
}

RicohSource::~RicohSource() = default;

std::optional<PerceptionFrame> RicohSource::operator()()
{
    if (!scene_)
        return std::nullopt;

    scene_->poll_ricoh(/*force=*/true);
    const std::uint64_t stamp = scene_->ricoh_last_stamp_ms();
    if (stamp == 0 || stamp == last_stamp_)
        return std::nullopt;   // no new panorama since last time
    last_stamp_ = stamp;

    cv::Mat pano = scene_->ricoh_bgr_copy();
    if (pano.empty())
        return std::nullopt;

    PerceptionFrame pf;
    pf.rgbd.rgb = pano.clone();   // owned BGR panorama
    pf.stamp    = stamp;
    pf.is_360   = true;

    // room<-ricoh at the panorama stamp (real ts → no InnerEigenAPI cache), plus the runtime az-tune.
    if (inner_eigen_ && graph_)
    {
        std::string room_name;
        if (const auto rooms = graph_->get_nodes_by_type("room"); !rooms.empty())
            room_name = rooms.front().name();
        if (!room_name.empty())
            if (auto T = inner_eigen_->get_transformation_matrix(room_name, "ricoh", stamp); T.has_value())
            {
                pf.room_T_sensor = T.value();
                if (azimuth_tune_deg_ != 0.0f)
                    pf.room_T_sensor.rotate(Eigen::AngleAxisd(azimuth_tune_deg_ * M_PI / 180.0,
                                                             Eigen::Vector3d::UnitZ()));
            }
    }
    return pf;
}

} // namespace rc
