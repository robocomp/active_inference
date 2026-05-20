#pragma once

#include <genericworker.h>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "yolo_seg_detector.h"

class UnifiedVoxelGrid;

namespace rc
{
    class VoxelOpenGLViewer;
}

class VoxelProcessor
{
public:
    struct Config
    {
        std::size_t voxel_decimation_factor = 2;
        float track_association_max_distance_m = 0.7f;
        int track_max_missed_frames = 10;
        bool verbose_debug = false;
    };

    explicit VoxelProcessor(UnifiedVoxelGrid& voxel_grid);

    void configure(const Config& config);

    void process_rgbd_frame(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
                            const std::vector<SegDetection>& detections,
                            const Mat::RTMat& room_T_robot,
                            const Mat::RTMat& room_T_zed,
                            rc::VoxelOpenGLViewer* voxel_viewer);

private:
    struct TrackBoxCandidate
    {
        int track_id = -1;
        std::string category;
        Eigen::Vector3f min = Eigen::Vector3f::Zero();
        Eigen::Vector3f max = Eigen::Vector3f::Zero();
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        int voxel_count = 0;
        int last_seen_frame = -1;
    };

    struct DetectionObservation
    {
        std::size_t det_index = 0;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        std::string label;
        float confidence = 0.0f;
    };

    struct InstanceTrack
    {
        int id = -1;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        std::string label;
        int last_seen_frame = -1;
    };

    bool is_target_label(const std::string& label) const;
    float detect_point_scale_once(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const;
    void build_owner_map_and_medians(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
                                     float point_scale,
                                     const std::vector<SegDetection>& detections,
                                     std::vector<int32_t>& pixel_owner,
                                     std::vector<float>& det_median_range_m) const;
    std::vector<int> hungarian_min_cost(const std::vector<std::vector<float>>& cost) const;
    std::vector<int> associate_detections_hungarian(const std::vector<DetectionObservation>& observations,
                                                    int frame_id);
    void prune_stale_tracks(int frame_id);
    std::vector<TrackBoxCandidate> build_track_box_candidates() const;
    void merge_duplicate_tracks(std::vector<TrackBoxCandidate>& candidates, int frame_id);
    std::vector<TrackBoxCandidate> filter_track_boxes_for_viewer(const std::vector<TrackBoxCandidate>& candidates) const;

    float axis_overlap(float amin, float amax, float bmin, float bmax) const;
    float box_volume(const TrackBoxCandidate& box) const;
    float intersection_volume(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const;
    bool boxes_look_duplicate(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const;
    int max_instances_for_category(const std::string& category) const;
    bool prefer_candidate(const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs, int frame_id) const;

    UnifiedVoxelGrid& voxel_grid_;
    Config config_;
    int compute_frame_ = 0;
    int next_track_id_ = 1;
    std::unordered_map<int, InstanceTrack> active_tracks_;
};