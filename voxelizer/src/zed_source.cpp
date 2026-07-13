#include "zed_source.h"
#include "scene_processor.h"

#include <genericworker.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <string>
#include <utility>

namespace rc
{

ZedSource::ZedSource(SceneProcessor* scene, std::shared_ptr<DSR::DSRGraph> graph)
    : scene_(scene), graph_(std::move(graph))
{
    if (graph_)
        inner_eigen_ = graph_->get_inner_eigen_api();
}

ZedSource::~ZedSource() = default;

std::optional<PerceptionFrame> ZedSource::operator()()
{
    if (!scene_)
        return std::nullopt;

    auto rgbd = scene_->get_rgbd_frame_from_dsr();   // drains the aligned RGBD (worker thread now)
    if (!rgbd.has_value())
        return std::nullopt;

    const std::uint64_t stamp = scene_->get_frame_timestamp_ms();
    if (stamp == 0 || stamp == last_stamp_)
        return std::nullopt;   // no new grab since last time
    last_stamp_ = stamp;

    PerceptionFrame pf;
    pf.rgbd   = std::move(*rgbd);   // rgb already an owned clone (get_rgbd_frame_from_dsr)
    pf.stamp  = stamp;
    pf.is_360 = false;

    // room<-zed at the capture stamp (real ts → no InnerEigenAPI cache), own instance.
    if (inner_eigen_ && graph_)
    {
        std::string room_name;
        if (const auto rooms = graph_->get_nodes_by_type("room"); !rooms.empty())
            room_name = rooms.front().name();
        if (!room_name.empty())
            if (auto T = inner_eigen_->get_transformation_matrix(room_name, "zed", stamp); T.has_value())
                pf.room_T_sensor = T.value();
    }
    return pf;
}

} // namespace rc
