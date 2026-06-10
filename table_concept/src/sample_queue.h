/*
 * sample_queue.h
 *
 * Binned historical sample queue for the table-concept agent.
 *
 * Points are partitioned into angular × height bins centred on the table:
 *   num_angle_bins = 24  (15° sectors)
 *   num_z_bins     = 10  (0.1 m height slices)
 *   max_per_bin    = 2   (best 2 points per bin by quality metric)
 *
 * Total capacity: 24 × 10 × 2 = 480 points.
 *
 * Each point carries a 2×2 robot-covariance at capture time and an
 * exponentially-smoothed Remembered Free Energy (RFE) value that drives
 * eviction and gradient weighting.
 *
 * Ported from box_concept/src/objects/table/belief.py (add_historical_points,
 * _add_to_bins, _compute_edge_score, update_rfe).
 */

#pragma once

#include "table_model.h"

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

// ─── Parameters ──────────────────────────────────────────────────────────────

struct SampleQueueParams
{
    int   num_angle_bins               = 24;
    int   num_z_bins                   = 10;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.08f;   // Only |SDF| < this admitted
    int   min_frames_before_historical = 10;      // No storage until pose converges
    int   historical_warmup_frames     = 50;      // Ramp to max_new_points_per_frame
    int   max_new_points_per_frame     = 5;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float edge_bonus_weight            = 0.3f;    // Priority boost for edge/corner pts
    float edge_proximity_threshold     = 0.05f;   // Distance to face = "close"
    float z_bin_size                   = 0.1f;    // Height slice width
    float current_sdf_weight           = 1.0f;    // Utility penalty for current-model mismatch
    float edge_anchor_score_threshold  = 0.3f;    // Diagnostic threshold for edge-like anchors
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
     * @param model        Current table model (needed for edge-score computation).
     * @param frame_count  Number of matched frames so far (drives admission warmup).
     */
    void insert(const std::vector<Eigen::Vector3f>& new_pts,
                const std::vector<float>&           sdf_values,
                const Eigen::Matrix2f&              robot_cov,
                const TableModel&                   model,
                int                                 frame_count);

    /**
     * Update Remembered Free Energy for all stored points.
     * Evicts points whose RFE exceeds rfe_max_threshold.
     *
     * @param model      Current model (used to recompute SDF).
     * @param robot_cov  Current robot XY covariance (w_t = 1/(1+tr(Σ))).
     */
    void update_rfe(const TableModel& model, const Eigen::Matrix2f& robot_cov);

    void refresh_scores(const TableModel& model);

    /** All stored positions (room frame). */
    std::vector<Eigen::Vector3f> points()  const;

    /**
     * Per-point gradient weights:
     *   w_i = 1 / (1 + tr(Σ_capture_i) + RFE_i)
     */
    std::vector<float> weights() const;

    /**
     * Coverage count per face (6 faces: +x, -x, +y, -y, +z_top, -z_bot in table frame).
     * Only the 4 vertical faces are reachable by a floor-navigating robot.
     *
     * Returns an array of 6 counts (float for continuity; each weight-summed).
     */
    std::array<float, 6> face_coverage(const TableModel& model) const;

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
    int  bin_index(const Eigen::Vector3f& p, float cx, float cy) const;
    float edge_score(const Eigen::Vector3f& p, const TableModel& model) const;
    Eigen::Vector3f to_local_frame(const Eigen::Vector3f& p, const TableState& s) const;
    void refresh_sample_score(SamplePoint& sp, const TableModel& model,
                              const std::optional<float>& abs_sdf_override = std::nullopt);
    void update_metrics(const TableModel* model = nullptr);
    static float percentile(std::vector<float> values, float q);

    /** Rebuild pts_ from a bin map (sorts each bin by quality, keeps max_per_bin). */
    void flush_bins(std::unordered_map<int, std::vector<BinEntry>>& bins, const TableModel& model);

    std::vector<SamplePoint>  pts_;
    SampleQueueParams         params_;
    SampleQueueCounters       counters_;
    SampleQueueMetrics        metrics_;
    std::uint64_t             next_insertion_id_ = 1;
};
