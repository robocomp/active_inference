/*
 * sample_queue.h
 *
 * Binned historical sample queue for the bottle-concept agent.
 *
 * Points are partitioned into azimuth × height bins centred on the bottle:
 *   num_angle_bins = 16  (22.5° sectors around the vertical axis)
 *   num_z_bins     = 6   (height slices spanning the cylinder)
 *   max_per_bin    = 2   (best 2 points per bin by quality metric)
 *
 * Each point carries a 2×2 robot-covariance at capture time and an
 * exponentially-smoothed Remembered Free Energy (RFE) value that drives
 * eviction and gradient weighting. This is the historical memory that anchors
 * the self-occluded back face of the bottle and supplies the prior-prediction
 * term when a YOLO detection drops out.
 *
 * Ported from table_concept/src/sample_queue.{h,cpp}; geometry retargeted from a
 * box (6 faces, yaw) to a vertical cylinder (radial side + caps, no yaw). The
 * bin/RFE/eviction machinery is unchanged.
 */

#pragma once

#include "bottle_model.h"

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace rc {

// ─── Parameters ──────────────────────────────────────────────────────────────

struct SampleQueueParams
{
    int   num_angle_bins               = 16;
    int   num_z_bins                   = 6;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.03f;   // Only |SDF| < this admitted
    int   min_frames_before_historical = 10;      // No storage until pose converges
    int   historical_warmup_frames     = 5;       // Ramp to max_new_points_per_frame
    int   max_new_points_per_frame     = 20;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float edge_bonus_weight            = 0.3f;    // Priority boost for rim/cap pts
    float edge_proximity_threshold     = 0.01f;   // Distance to surface = "close"
    float z_bin_size                   = 0.04f;   // Height slice width (m)
    float current_sdf_weight           = 1.0f;    // Utility penalty for current-model mismatch
    float edge_anchor_score_threshold  = 0.3f;    // Diagnostic threshold for rim-like anchors
    float rfe_weight_gain              = 0.25f;   // Scale RFE impact in anchor weights/utility
    float min_anchor_weight            = 0.12f;   // Keep historical anchors from vanishing
};

// ─── SamplePoint ─────────────────────────────────────────────────────────────

struct SamplePoint
{
    Eigen::Vector3f position;       // Observed voxel centroid, room frame
    Eigen::Matrix2f capture_cov;    // Robot XY covariance at capture time
    float           rfe = 0.0f;     // Remembered Free Energy
    Eigen::Vector3f local_position = Eigen::Vector3f::Zero();
    float           last_abs_sdf   = 0.0f;
    float           edge_score_cache = 0.0f;
    float           utility_score  = 0.0f;
    std::uint64_t   insertion_id   = 0;
};

struct SampleQueueCounters
{
    int accepted_new        = 0;
    int rejected_warmup     = 0;
    int rejected_sdf        = 0;
    int rejected_frame_cap  = 0;
    int rejected_bin_rank   = 0;
    int evicted_bin_rank    = 0;
    int evicted_rfe         = 0;
};

struct SampleQueueMetrics
{
    int   anchor_count           = 0;
    int   capacity               = 0;
    float effective_weight_mass  = 0.0f;
    float rfe_mean               = 0.0f;
    float rfe_p50                = 0.0f;
    float rfe_p90                = 0.0f;
    float bin_occupancy_ratio    = 0.0f;
    float edge_anchor_ratio      = 0.0f;
    float mean_abs_sdf           = 0.0f;
    SampleQueueCounters counters;
};

// ─── SampleQueue ─────────────────────────────────────────────────────────────

class SampleQueue
{
public:
    explicit SampleQueue(const SampleQueueParams& params = {});

    void begin_cycle();

    /**
     * Admit new near-surface candidates.
     *
     * @param new_pts      Room-frame points to consider.
     * @param sdf_values   SDF values for each point under the current model state.
     * @param robot_cov    2×2 robot XY covariance at this frame.
     * @param model        Current bottle model (needed for edge-score computation).
     * @param frame_count  Number of matched frames so far (drives admission warmup).
     */
    void insert(const std::vector<Eigen::Vector3f>& new_pts,
                const std::vector<float>&           sdf_values,
                const Eigen::Matrix2f&              robot_cov,
                const BottleModel&                  model,
                int                                 frame_count);

    /**
     * Update Remembered Free Energy for all stored points.
     * Evicts points whose RFE exceeds rfe_max_threshold.
     */
    void update_rfe(const BottleModel& model, const Eigen::Matrix2f& robot_cov);

    void refresh_scores(const BottleModel& model);

    /** All stored positions (room frame). */
    std::vector<Eigen::Vector3f> points()  const;

    /**
     * Per-point gradient weights:
     *   w_i = 1 / (1 + tr(Σ_capture_i) + RFE_i)
     */
    std::vector<float> weights() const;

    /**
     * Azimuthal/cap coverage (6 buckets: +x, -x, +y, -y, top cap, bottom cap in
     * bottle frame), weight-summed. Diagnostic only.
     */
    std::array<float, 6> face_coverage(const BottleModel& model) const;

    int  size()  const { return static_cast<int>(pts_.size()); }
    bool empty() const { return pts_.empty(); }
    void clear()       { pts_.clear(); }
    int  capacity() const { return params_.num_angle_bins * params_.num_z_bins * params_.max_per_bin; }

    const SampleQueueMetrics& metrics() const { return metrics_; }

private:
    struct BinEntry
    {
        SamplePoint point;
        float       utility = 0.0f;
        bool        from_existing = false;
    };

    // ── bin helpers ──────────────────────────────────────────────────────────
    int  bin_index(const Eigen::Vector3f& p, const BottleState& s) const;
    float edge_score(const Eigen::Vector3f& p, const BottleModel& model) const;
    Eigen::Vector3f to_local_frame(const Eigen::Vector3f& p, const BottleState& s) const;
    void refresh_sample_score(SamplePoint& sp, const BottleModel& model,
                              const std::optional<float>& abs_sdf_override = std::nullopt);
    void update_metrics(const BottleModel* model = nullptr);
    static float percentile(std::vector<float> values, float q);

    /** Rebuild pts_ from a bin map (sorts each bin by quality, keeps max_per_bin). */
    void flush_bins(std::unordered_map<int, std::vector<BinEntry>>& bins, const BottleModel& model);

    std::vector<SamplePoint>  pts_;
    SampleQueueParams         params_;
    SampleQueueCounters       counters_;
    SampleQueueMetrics        metrics_;
    std::uint64_t             next_insertion_id_ = 1;
};

}  // namespace rc
