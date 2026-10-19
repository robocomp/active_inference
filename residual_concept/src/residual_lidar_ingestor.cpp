/*
 *    residual_lidar_ingestor.cpp  —  see residual_lidar_ingestor.h
 */

#include "residual_lidar_ingestor.h"

#include <utility>
#include <vector>

#include "../../common/media_transport/lidar_plane_reader.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

namespace rc
{

ResidualLidarIngestor::ResidualLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                                             const ResidualConfig& cfg)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(&cfg)
{
    // Subscribers come up lazily inside reader_->poll() once each node + descriptor exists; nothing
    // touches DDS here.
    // helios (high 360, sees tops/far) + optional bpearl (low, sees legs/low obstacles). The shared reader
    // caches each plane and always merges the latest of both (flicker-free after the merge-cache fix), so the
    // clouds are jointly clustered — fuller vertical coverage. bpearl toggleable (ResidualModel.UseBpearl).
    std::vector<std::string> planes{"helios"};
    if (cfg.use_bpearl)
        planes.emplace_back("bpearl");
    plane_nodes_ = planes;                                         // plane id = index in this list
    plane_origin_room_.assign(plane_nodes_.size(), Eigen::Vector3f::Zero());
    plane_origin_ok_.assign(plane_nodes_.size(), false);
    reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_, std::move(planes), "lidar");
}

ResidualLidarIngestor::~ResidualLidarIngestor()
{
    reader_.reset();
}

bool ResidualLidarIngestor::pump()
{
    if (not reader_)
        return false;

    // Newest "helios" sweep, transformed into the ROOM frame at its capture
    // stamp (interpolate=true — a rotating robot's room<-robot pose differs from the latest pose).
    const std::string room_frame = rc::room::current_room_frame(*G_);
    const auto sweep = reader_->poll(room_frame, /*interpolate=*/true);
    if (not sweep.has_value() or sweep->points.empty())
        return false;

    sweep_room_  = std::move(sweep->points);
    plane_id_    = std::move(sweep->plane_id);   // per-point device tag (helios=0, bpearl=1) for per-device filtering
    origin_room_ = sweep->origin;
    last_ts_     = sweep->stamp_ms;
    // Each plane's OWN sensor centre, at the same stamp + interpolation the reader registered its points with.
    // Only the translation is needed: it is the ray origin the occupancy grid traces that device's beams from.
    for (std::size_t k = 0; k < plane_nodes_.size(); ++k)
    {
        plane_origin_ok_[k] = false;
        if (k >= sweep->plane_stamp_ms.size() or sweep->plane_stamp_ms[k] <= 0 or inner_eigen_ == nullptr) continue;
        const auto T = inner_eigen_->get_transformation_matrix(room_frame, plane_nodes_[k],
                                                               static_cast<std::uint64_t>(sweep->plane_stamp_ms[k]),
                                                               "RT", DSR::RT_API::TimeQuery::Interpolated);
        if (not T.has_value()) continue;                         // ALWAYS check the optional (CLAUDE.md)
        const auto& M = T.value().matrix();
        plane_origin_room_[k] = Eigen::Vector3f(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)),
                                                static_cast<float>(M(2, 3)));
        plane_origin_ok_[k] = true;
    }
    return true;
}

}  // namespace rc
