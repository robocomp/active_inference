#pragma once

#include <genericworker.h>

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_object_box.h"
#include "rgbd_data.h"
#include "yolo_processor.h"

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
        std::size_t viewer_max_rendered_voxels = 30'000;
        float track_association_max_distance_m = 0.7f;
        float z_lift_m = 0.0f;
        float mask_surface_band_m = 0.35f;   // |range - median| gate for opaque masked points
        int track_max_missed_frames = 10;
        int viewer_voxel_fps = 10;
        bool verbose_debug = false;
    };

    explicit VoxelProcessor(UnifiedVoxelGrid& voxel_grid);

    void configure(const Config& config);
    void clear_state(rc::VoxelOpenGLViewer* voxel_viewer);
    void fuse_lidar_support_points(const std::unordered_map<int, std::vector<Eigen::Vector3f>>& lidar_points_by_track);

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

    // Sensing data accessors — call after process_rgbd_frame each cycle
    const std::vector<TrackBoxCandidate>& last_track_candidates() const { return last_box_candidates_; }
    int                                   last_frame_id()         const { return last_frame_id_; }
    std::vector<float> get_flat_pts_for_track(int track_id, int max_pts) const;

    void process_rgbd_frame(const RGBDData& rgbd,
                            const std::vector<SegDetection>& detections,
                            const Mat::RTMat& room_T_robot,
                            const Mat::RTMat& room_T_zed,
                            std::span<const GraphObjectBox> explained_boxes,
                            rc::VoxelOpenGLViewer* voxel_viewer);

private:
    struct PointCloudStats
    {
        Eigen::Vector3f min = Eigen::Vector3f::Zero();
        Eigen::Vector3f max = Eigen::Vector3f::Zero();
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        std::size_t count = 0;
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
    std::optional<std::string> observable_category_for_model(const std::string& model_category) const;
    bool model_matches_label(const GraphObjectBox& box, const std::string& label) const;
    bool point_inside_box(const Eigen::Vector3f& point,
                          const GraphObjectBox& box,
                          float padding_m = 0.05f) const;
    bool boxes_overlap(const Eigen::Vector3f& min_a,
                       const Eigen::Vector3f& max_a,
                       const Eigen::Vector3f& min_b,
                       const Eigen::Vector3f& max_b,
                       float padding_m = 0.0f) const;
    bool point_explained_by_model(const Eigen::Vector3f& point,
                                  const std::string& label,
                                  std::span<const GraphObjectBox> explained_boxes) const;
    void build_owner_map_and_medians(const RGBDData& rgbd,
                                     const std::vector<SegDetection>& detections,
                                     std::vector<int32_t>& pixel_owner,
                                     std::vector<float>& det_median_range_m) const;
    std::vector<int> hungarian_min_cost(const std::vector<std::vector<float>>& cost) const;
    std::vector<int> associate_detections_hungarian(const std::vector<DetectionObservation>& observations,
                                                    int frame_id);
    void prune_stale_tracks(int frame_id);
    PointCloudStats compute_point_cloud_stats(std::span<const Eigen::Vector3f> points) const;
    std::vector<TrackBoxCandidate> build_track_box_candidates() const;
    void merge_duplicate_tracks(std::vector<TrackBoxCandidate>& candidates, int frame_id);
    std::size_t suppress_residual_tracks_near_models(std::vector<TrackBoxCandidate>& candidates,
                                                     std::span<const GraphObjectBox> explained_boxes);
    std::vector<TrackBoxCandidate> filter_track_boxes_for_viewer(const std::vector<TrackBoxCandidate>& candidates) const;

    float axis_overlap(float amin, float amax, float bmin, float bmax) const;
    float box_volume(const TrackBoxCandidate& box) const;
    float intersection_volume(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const;
    bool boxes_look_duplicate(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const;
    int max_instances_for_category(const std::string& category) const;
    bool prefer_candidate(const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs, int frame_id) const;

    UnifiedVoxelGrid& voxel_grid_;
    Config config_;
    static constexpr std::size_t max_clustered_box_points_ = 512;
    static constexpr float model_point_padding_m_ = 0.08f;
    static constexpr float model_track_padding_m_ = 0.12f;
    std::chrono::steady_clock::time_point last_voxel_viewer_update_{};
    int compute_frame_ = 0;
    int next_track_id_ = 1;
    std::unordered_map<int, InstanceTrack> active_tracks_;
    std::vector<TrackBoxCandidate> last_box_candidates_;
    std::unordered_map<int, std::vector<Eigen::Vector3f>> last_track_points_;
    int last_frame_id_ = 0;
};