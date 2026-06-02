#include "lidar_track_attributor.h"

#include <algorithm>
#include <cmath>
#include <limits>

LidarTrackAttributor::LidarTrackAttributor()
    : config_{}
{
}

LidarTrackAttributor::LidarTrackAttributor(const Config& config)
    : config_(config)
{
}

LidarTrackAttributor::AttributionMap
LidarTrackAttributor::attribute_points(std::span<const Eigen::Vector3f> lidar_points_room,
                                       std::span<const TrackCandidate> track_candidates) const
{
    AttributionMap attributed;
    if (lidar_points_room.empty() || track_candidates.empty())
        return attributed;

    std::vector<const TrackCandidate*> table_tracks;
    table_tracks.reserve(track_candidates.size());
    for (const auto& track : track_candidates)
        if (track.track_id >= 0 && track.category == "table")
            table_tracks.push_back(&track);

    if (table_tracks.empty())
        return attributed;

    for (const auto& point : lidar_points_room)
    {
        const TrackCandidate* best_track = nullptr;
        float best_score = std::numeric_limits<float>::max();

        for (const auto* track : table_tracks)
        {
            const Eigen::Vector3f min_gate(track->min.x() - config_.gate_expand_xy_m,
                                           track->min.y() - config_.gate_expand_xy_m,
                                           track->min.z() - config_.gate_expand_z_m);
            const Eigen::Vector3f max_gate(track->max.x() + config_.gate_expand_xy_m,
                                           track->max.y() + config_.gate_expand_xy_m,
                                           track->max.z() + config_.gate_expand_z_m);

            if ((point.array() < min_gate.array()).any() || (point.array() > max_gate.array()).any())
                continue;

            const float diag = (track->max - track->min).norm();
            const float max_centroid_dist = 0.5f * diag + config_.centroid_radius_extra_m;
            const float centroid_dist = (point - track->centroid).norm();
            if (centroid_dist > max_centroid_dist)
                continue;

            if (centroid_dist < best_score)
            {
                best_score = centroid_dist;
                best_track = track;
            }
        }

        if (best_track != nullptr)
            attributed[best_track->track_id].push_back(point);
    }

    for (auto it = attributed.begin(); it != attributed.end(); )
    {
        if (it->second.size() < config_.min_points_per_track)
            it = attributed.erase(it);
        else
            ++it;
    }

    return attributed;
}
