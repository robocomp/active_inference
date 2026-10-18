/*
 *    common/lidar_ingestor/concept_lidar_ingestor.cpp  —  see concept_lidar_ingestor.h
 */

#include "concept_lidar_ingestor.h"

#include <utility>
#include <vector>

#include "../media_transport/lidar_plane_reader.h"
#include "../room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

namespace rc
{

ConceptLidarIngestor::ConceptLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                                           std::function<LidarGates()> gates)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), gates_(std::move(gates))
{
    // Subscribers come up lazily inside reader_->poll() once each node + descriptor exists AND the feature is
    // enabled (the gate > 0); nothing touches DDS here.
    reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_, std::vector<std::string>{"helios"}, "lidar");
    // Low bpearl plane as a SEPARATE reader: if that plane is absent it simply stages nothing.
    // Consumed as its own ray-set with its own origin (occlusion-aware first-hit).
    reader_bpearl_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_, std::vector<std::string>{"bpearl"}, "lidar");
}

ConceptLidarIngestor::~ConceptLidarIngestor()
{
    reader_.reset();
    reader_bpearl_.reset();
}

bool ConceptLidarIngestor::pump()
{
    helios_fresh_ = false;
    bpearl_fresh_ = false;
    bpearl_robot_fresh_ = false;

    const LidarGates g = gates_ ? gates_() : LidarGates{};

    // Newest "helios" sweep, transformed into the ROOM frame at its capture stamp
    // (interpolate=true — a rotating robot's room<-robot pose differs from the latest pose). enabled follows the
    // feature switches so the reader stays dormant while BOTH helios-side features are off.
    if (reader_)
    {
        const bool en = g.helios_precision > 0.0f or g.free_space_precision > 0.0f;
        if (const auto sweep = reader_->poll(rc::room::current_room_frame(*G_), /*interpolate=*/true, /*enabled=*/en);
            sweep.has_value() and not sweep->points.empty())
        {
            sweep_room_   = std::move(sweep->points);
            origin_room_  = sweep->origin;
            helios_fresh_ = true;
        }
    }
    // Low bpearl plane — its OWN capture stamp, its OWN origin. Only pumped while its feature is on.
    if (reader_bpearl_)
    {
        if (const auto bp = reader_bpearl_->poll(rc::room::current_room_frame(*G_), /*interpolate=*/true,
                                                 /*enabled=*/g.bpearl_precision > 0.0f);
            bp.has_value() and not bp->points.empty())
        {
            sweep_bpearl_room_  = std::move(bp->points);
            origin_bpearl_room_ = bp->origin;
            bpearl_fresh_       = true;
        }
        // The same capture in the ROBOT frame, from the reader's cache (no second drain). Keyed on the cache's
        // stamp, not on bpearl_fresh_, so a sweep whose ROOM transform failed is still staged here.
        if (g.bpearl_precision > 0.0f)
            if (const auto ts = reader_bpearl_->newest_cache_stamp(); ts > bpearl_robot_stamp_ms_)
                if (const auto robots = G_->get_nodes_by_type("robot"); not robots.empty())
                    if (auto rb = reader_bpearl_->restage(robots.front().name(), /*interpolate=*/false);
                        rb.has_value() and not rb->points.empty())
                    {
                        sweep_bpearl_robot_    = std::move(rb->points);
                        origin_bpearl_robot_   = rb->origin;
                        bpearl_robot_stamp_ms_ = ts;
                        bpearl_robot_fresh_    = true;
                    }
    }
    return helios_fresh_ or bpearl_fresh_;
}

}  // namespace rc
