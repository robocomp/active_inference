/*
 * common/sample_queue/sample_queue.h
 *
 * Shared binned historical sample queue for the concept agents (bottle/table/chair/…).
 *
 * Header-only `template<class Model> class SampleQueue`: ALL the queue machinery (binning, admission,
 * eviction, RFE, weights, metrics) is shared verbatim; the 4 object-specific GEOMETRY primitives
 * (bin_index / edge_score / to_local_frame / face_coverage) are delegated to a per-object policy
 * `SampleQueueGeometry<Model>` that each agent specialises with its exact existing bodies. The one
 * genuine bulk divergence between the reference agents — the per-frame admission strategy — is gated
 * by `SampleQueueParams::diversity_admission`, so this template is BEHAVIOUR-PRESERVING for both
 * (bottle: smallest-|SDF|; table: spatial-diversity).
 *
 * Requirements on Model: `using State = <Obj>State;`, `const State& state() const`,
 * `float sdf_point(const Eigen::Vector3f&) const`, and a `SampleQueueGeometry<Model>` specialisation.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace rc {

// ─── Parameters / point / metrics (object-agnostic) ──────────────────────────

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
    // Per-frame admission strategy. true = prefer UNDER-FILLED bins then smallest |SDF| (spatial
    // diversity; table's behaviour). false = globally-smallest |SDF| (bottle's behaviour). Set per
    // agent in make_queue_params so the shared template preserves each agent's existing belief.
    bool  diversity_admission          = true;
};

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

inline float anchor_weight(const SamplePoint& sp, const SampleQueueParams& params)
{
    const float denom = 1.0f + sp.capture_cov.trace() + params.rfe_weight_gain * sp.rfe;
    return std::max(params.min_anchor_weight, 1.0f / std::max(1e-6f, denom));
}

// ─── Per-object geometry policy (each agent specialises) ─────────────────────
//   static int             bin_index(p, State, params)        spatial bin
//   static float           edge_score(p, Model, params)       rim/edge proximity in [0,1]
//   static Eigen::Vector3f to_local_frame(p, State)           room → object-local
//   static array<float,6>  face_coverage(Model, pts, params)  weighted per-face coverage
template <class Model>
struct SampleQueueGeometry;

// ─── SampleQueue ─────────────────────────────────────────────────────────────

template <class Model>
class SampleQueue
{
public:
    using State = typename Model::State;
    using Geom  = SampleQueueGeometry<Model>;

    explicit SampleQueue(const SampleQueueParams& params = {}) : params_(params) {}

    void begin_cycle()
    {
        counters_ = {};
        metrics_.counters = counters_;
    }

    void insert(const std::vector<Eigen::Vector3f>& new_pts,
                const std::vector<float>&           sdf_values,
                const Eigen::Matrix2f&              robot_cov,
                const Model&                        model,
                int                                 frame_count)
    {
        if (frame_count < params_.min_frames_before_historical)
        {
            counters_.rejected_warmup += static_cast<int>(new_pts.size());
            metrics_.counters = counters_;
            return;
        }

        refresh_scores(model);

        const float progress = std::min(1.0f,
            static_cast<float>(frame_count - params_.min_frames_before_historical)
            / static_cast<float>(std::max(1, params_.historical_warmup_frames)));
        const int max_this_frame = std::max(1, static_cast<int>(
            progress * static_cast<float>(params_.max_new_points_per_frame)));

        std::vector<std::pair<float, int>> eligible;   // (|sdf|, index)
        for (std::size_t i = 0; i < new_pts.size() and i < sdf_values.size(); ++i)
        {
            const float abs_sdf = std::abs(sdf_values[i]);
            if (abs_sdf < params_.sdf_threshold_for_storage)
                eligible.emplace_back(abs_sdf, static_cast<int>(i));
            else
                ++counters_.rejected_sdf;
        }

        if (params_.diversity_admission)
        {
            // Prefer points falling into UNDER-FILLED bins (unseen faces/legs), then smallest |SDF|,
            // so the queue accumulates edge/leg/new-face points instead of re-admitting the well-fit
            // top face (sdf≈0) every frame.
            const State& st0 = model.state();
            std::unordered_map<int, int> occ;
            for (const auto& sp : pts_) ++occ[Geom::bin_index(sp.position, st0, params_)];
            std::sort(eligible.begin(), eligible.end(),
                      [&](const std::pair<float, int>& a, const std::pair<float, int>& b)
                      {
                          const bool fa = occ[Geom::bin_index(new_pts[a.second], st0, params_)] >= params_.max_per_bin;
                          const bool fb = occ[Geom::bin_index(new_pts[b.second], st0, params_)] >= params_.max_per_bin;
                          if (fa != fb) return !fa;          // under-filled bins first
                          return a.first < b.first;          // then smallest |SDF|
                      });
            if (static_cast<int>(eligible.size()) > max_this_frame)
            {
                counters_.rejected_frame_cap += static_cast<int>(eligible.size()) - max_this_frame;
                eligible.resize(max_this_frame);
            }
        }
        else
        {
            // Globally smallest |SDF|.
            if (static_cast<int>(eligible.size()) > max_this_frame)
            {
                counters_.rejected_frame_cap += static_cast<int>(eligible.size()) - max_this_frame;
                std::partial_sort(eligible.begin(), eligible.begin() + max_this_frame, eligible.end());
                eligible.resize(max_this_frame);
            }
        }

        if (eligible.empty())
        {
            update_metrics(&model);
            return;
        }

        const State& st = model.state();

        std::unordered_map<int, std::vector<BinEntry>> bins;
        for (const auto& sp : pts_)
        {
            const int idx = Geom::bin_index(sp.position, st, params_);
            bins[idx].push_back(BinEntry{sp, sp.utility_score, true});
        }

        for (const auto& [abs_sdf, i] : eligible)
        {
            SamplePoint sp;
            sp.position     = new_pts[i];
            sp.capture_cov  = robot_cov;
            sp.rfe          = 0.0f;
            sp.insertion_id = next_insertion_id_++;
            refresh_sample_score(sp, model, abs_sdf);

            const int idx = Geom::bin_index(sp.position, st, params_);
            bins[idx].push_back(BinEntry{sp, sp.utility_score, false});
        }

        flush_bins(bins, model);
    }

    void update_rfe(const Model& model, const Eigen::Matrix2f& robot_cov)
    {
        if (pts_.empty())
        {
            update_metrics(&model);
            return;
        }

        const float alpha    = params_.rfe_alpha;
        const float max_rfe  = params_.rfe_max_threshold;
        const float robot_tr = robot_cov.trace();
        const float w_t      = 1.0f / (1.0f + robot_tr);

        std::vector<SamplePoint> survivors;
        survivors.reserve(pts_.size());

        for (auto& sp : pts_)
        {
            const float sdf  = model.sdf_point(sp.position);
            const float sdf2 = sdf * sdf;
            sp.rfe = alpha * sp.rfe + w_t * sdf2;

            if (sp.rfe < max_rfe)
                survivors.push_back(sp);
            else
                ++counters_.evicted_rfe;
        }

        pts_ = std::move(survivors);
        refresh_scores(model);
    }

    void refresh_scores(const Model& model)
    {
        for (auto& sp : pts_)
            refresh_sample_score(sp, model);
        update_metrics(&model);
    }

    std::vector<Eigen::Vector3f> points() const
    {
        std::vector<Eigen::Vector3f> out;
        out.reserve(pts_.size());
        for (const auto& sp : pts_)
            out.push_back(sp.position);
        return out;
    }

    std::vector<float> weights() const
    {
        std::vector<float> out;
        out.reserve(pts_.size());
        for (const auto& sp : pts_)
            out.push_back(anchor_weight(sp, params_));
        return out;
    }

    std::array<float, 6> face_coverage(const Model& model) const
    {
        return Geom::face_coverage(model, pts_, params_);
    }

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

    void refresh_sample_score(SamplePoint& sp, const Model& model,
                              const std::optional<float>& abs_sdf_override = std::nullopt)
    {
        sp.local_position = Geom::to_local_frame(sp.position, model.state());
        sp.last_abs_sdf = abs_sdf_override.has_value() ? abs_sdf_override.value()
                                                       : std::abs(model.sdf_point(sp.position));
        sp.edge_score_cache = Geom::edge_score(sp.position, model, params_);
        sp.utility_score = sp.capture_cov.trace()
                         + params_.rfe_weight_gain * sp.rfe
                         + params_.current_sdf_weight * sp.last_abs_sdf
                         - params_.edge_bonus_weight * sp.edge_score_cache;
    }

    static float percentile(std::vector<float> values, float q)
    {
        if (values.empty())
            return 0.0f;
        std::sort(values.begin(), values.end());
        const float clamped_q = std::clamp(q, 0.0f, 1.0f);
        const float idx = clamped_q * static_cast<float>(values.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
        if (lo == hi)
            return values[lo];
        const float frac = idx - static_cast<float>(lo);
        return values[lo] + frac * (values[hi] - values[lo]);
    }

    void update_metrics(const Model* model = nullptr)
    {
        metrics_ = {};
        metrics_.capacity = capacity();
        metrics_.anchor_count = static_cast<int>(pts_.size());
        metrics_.counters = counters_;

        if (pts_.empty())
            return;

        std::vector<float> rfe_values;
        rfe_values.reserve(pts_.size());
        std::set<int> occupied_bins;
        float weight_sum = 0.0f;
        float edge_like_count = 0.0f;
        float abs_sdf_sum = 0.0f;

        const State* state = model ? &model->state() : nullptr;
        for (const auto& sp : pts_)
        {
            const float weight = anchor_weight(sp, params_);
            weight_sum += weight;
            rfe_values.push_back(sp.rfe);
            abs_sdf_sum += sp.last_abs_sdf;
            if (sp.edge_score_cache >= params_.edge_anchor_score_threshold)
                edge_like_count += 1.0f;

            if (state != nullptr)
                occupied_bins.insert(Geom::bin_index(sp.position, *state, params_));
            else
            {
                // No model this cycle: a generic angle×z bin from the cached local position (the
                // primitive bin reference is unavailable). Diagnostic-only path.
                const float angle = std::atan2(sp.local_position.y(), sp.local_position.x());
                int ab = static_cast<int>((angle + std::numbers::pi_v<float>) / (2.0f * std::numbers::pi_v<float>) * params_.num_angle_bins) % params_.num_angle_bins;
                if (ab < 0) ab += params_.num_angle_bins;
                int zb = static_cast<int>(sp.local_position.z() / params_.z_bin_size);
                zb = std::clamp(zb, 0, params_.num_z_bins - 1);
                occupied_bins.insert(ab * params_.num_z_bins + zb);
            }
        }

        metrics_.effective_weight_mass = weight_sum;
        metrics_.rfe_mean = std::accumulate(rfe_values.begin(), rfe_values.end(), 0.0f) /
                            static_cast<float>(rfe_values.size());
        metrics_.rfe_p50 = percentile(rfe_values, 0.5f);
        metrics_.rfe_p90 = percentile(rfe_values, 0.9f);
        metrics_.bin_occupancy_ratio = static_cast<float>(occupied_bins.size()) /
                                       static_cast<float>(std::max(1, params_.num_angle_bins * params_.num_z_bins));
        metrics_.edge_anchor_ratio = edge_like_count / static_cast<float>(pts_.size());
        metrics_.mean_abs_sdf = abs_sdf_sum / static_cast<float>(pts_.size());
    }

    void flush_bins(std::unordered_map<int, std::vector<BinEntry>>& bins, const Model& model)
    {
        std::vector<SamplePoint> next_pts;
        next_pts.reserve(capacity());
        for (auto& [idx, entries] : bins)
        {
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) {
                          if (a.utility == b.utility)
                              return a.point.insertion_id < b.point.insertion_id;
                          return a.utility < b.utility;
                      });
            const int keep = std::min(static_cast<int>(entries.size()), params_.max_per_bin);
            for (int i = 0; i < keep; ++i)
            {
                next_pts.push_back(entries[i].point);
                if (!entries[i].from_existing)
                    ++counters_.accepted_new;
            }
            for (std::size_t i = keep; i < entries.size(); ++i)
            {
                if (entries[i].from_existing)
                    ++counters_.evicted_bin_rank;
                else
                    ++counters_.rejected_bin_rank;
            }
        }

        pts_ = std::move(next_pts);
        refresh_scores(model);
    }

    std::vector<SamplePoint>  pts_;
    SampleQueueParams         params_;
    SampleQueueCounters       counters_;
    SampleQueueMetrics        metrics_;
    std::uint64_t             next_insertion_id_ = 1;
};

}  // namespace rc
