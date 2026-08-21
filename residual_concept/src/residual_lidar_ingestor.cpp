/*
 *    residual_lidar_ingestor.cpp  —  see residual_lidar_ingestor.h
 */

#include "residual_lidar_ingestor.h"

#include <utility>
#include <vector>

#include "../../common/media_transport/lidar_plane_reader.h"

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
    const auto sweep = reader_->poll("room", /*interpolate=*/true);
    if (not sweep.has_value() or sweep->points.empty())
        return false;

    sweep_room_  = std::move(sweep->points);
    plane_id_    = std::move(sweep->plane_id);   // per-point device tag (helios=0, bpearl=1) for per-device filtering
    origin_room_ = sweep->origin;
    last_ts_     = sweep->stamp_ms;
    return true;
}

}  // namespace rc
