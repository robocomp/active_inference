#pragma once

#include <Eigen/Core>

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

class LidarTrackAttributor
{
    public:
        struct Config
        {
            float gate_expand_xy_m = 0.12f;
            float gate_expand_z_m = 0.08f;
            float centroid_radius_extra_m = 0.20f;
            std::size_t min_points_per_track = 8;
            bool verbose_debug = false;
        };

        struct TrackCandidate
        {
            int track_id = -1;
            std::string category;
            Eigen::Vector3f min = Eigen::Vector3f::Zero();
            Eigen::Vector3f max = Eigen::Vector3f::Zero();
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        };

        using AttributionMap = std::unordered_map<int, std::vector<Eigen::Vector3f>>;

        LidarTrackAttributor();
        explicit LidarTrackAttributor(const Config& config);

        AttributionMap attribute_points(std::span<const Eigen::Vector3f> lidar_points_room,
                                        std::span<const TrackCandidate> track_candidates) const;

    private:
        Config config_;
};
