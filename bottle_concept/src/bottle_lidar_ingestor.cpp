/*
 * bottle_lidar_ingestor.cpp  —  see bottle_lidar_ingestor.h
 */

#include "bottle_lidar_ingestor.h"

#include <utility>
#include <vector>

#include "../../common/media_transport/lidar_plane_reader.h"

namespace rc
{

BottleLidarIngestor::BottleLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                                         const BottleConfig& cfg)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(&cfg)
{
    // Subscribers come up lazily inside reader_->poll() once each node + descriptor exists AND the
    // feature is enabled (cfg.lidar_precision > 0); nothing touches DDS here.
    reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_, std::vector<std::string>{"helios"}, "lidar3D", "lidar");
}

BottleLidarIngestor::~BottleLidarIngestor()
{
    reader_.reset();
}

bool BottleLidarIngestor::pump()
{
    if (not reader_)
        return false;

    // Newest "helios" (or fallback "lidar3D") sweep, transformed into the ROOM frame at its capture
    // stamp (interpolate=true — a rotating robot's room<-robot pose differs from the latest pose).
    // enabled follows the feature switch so the reader stays dormant while lidar_precision == 0.
    const auto sweep = reader_->poll("room", /*interpolate=*/true, /*enabled=*/cfg_->lidar_precision > 0.0f);
    if (not sweep.has_value() or sweep->points.empty())
        return false;

    sweep_room_  = std::move(sweep->points);
    origin_room_ = sweep->origin;
    return true;
}

}  // namespace rc
