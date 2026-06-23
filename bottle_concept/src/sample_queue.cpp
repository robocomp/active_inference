/*
 * sample_queue.cpp
 *
 * Ported from table_concept/src/sample_queue.cpp. Frame change is the same: all
 * voxel centroids arrive already in the room frame, so the robot↔room Jacobian
 * propagation is omitted and the gradient weight reduces to
 *   w_i = 1 / (1 + tr(Σ_capture_i) + RFE_i).
 *
 * Geometry retargeted to a vertical cylinder: bins are azimuth × height relative
 * to the bottle centre, edge_score measures proximity to the lateral surface and
 * the end caps (the cylinder's "rim"), and to_local_frame is a pure translation
 * (a cylinder is yaw-symmetric).
 */

#include "sample_queue.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <set>

namespace rc {

namespace
{
float anchor_weight(const SamplePoint& sp, const SampleQueueParams& params)
{
    const float denom = 1.0f + sp.capture_cov.trace() + params.rfe_weight_gain * sp.rfe;
    return std::max(params.min_anchor_weight, 1.0f / std::max(1e-6f, denom));
}
}

// ─── Constructor ─────────────────────────────────────────────────────────────

SampleQueue::SampleQueue(const SampleQueueParams& params)
    : params_(params)
{}

void SampleQueue::begin_cycle()
{
    counters_ = {};
    metrics_.counters = counters_;
}

// ─── Bin index (azimuth × height, relative to bottle centre) ───────────────────

int SampleQueue::bin_index(const Eigen::Vector3f& p, const BottleState& s) const
{
    const int na = params_.num_angle_bins;
    const int nz = params_.num_z_bins;

    const float angle = std::atan2(p.y() - s.cy, p.x() - s.cx);
    int ab = static_cast<int>((angle + std::numbers::pi_v<float>) / (2.0f * std::numbers::pi_v<float>) * na) % na;
    if (ab < 0) ab += na;

    // Height measured from the cylinder base so bins span the bottle, not the floor.
    const float z_rel = p.z() - (s.cz - s.height * 0.5f);
    int zb = static_cast<int>(z_rel / params_.z_bin_size);
    zb = std::clamp(zb, 0, nz - 1);

    return ab * nz + zb;
}

// ─── Edge score (lateral surface + end caps = the rim) ─────────────────────────

float SampleQueue::edge_score(const Eigen::Vector3f& p, const BottleModel& model) const
{
    const auto& s = model.state();
    const float lx = p.x() - s.cx;
    const float ly = p.y() - s.cy;
    const float lz = p.z() - s.cz;

    const float r_xy   = std::sqrt(lx * lx + ly * ly);
    const float d_side = std::abs(r_xy - s.radius);            // to lateral surface
    const float d_cap  = std::abs(std::abs(lz) - s.height * 0.5f);  // to nearer cap plane

    const float th = params_.edge_proximity_threshold;
    const bool close_side = d_side < th;
    const bool close_cap  = d_cap  < th;

    // 1.0 on the rim (near both), 0.4 on a single surface, 0 elsewhere.
    float score = (close_side and close_cap) ? 1.0f
                : (close_side or close_cap)   ? 0.4f
                                              : 0.0f;
    // Proximity bonus toward the rim circle.
    score += std::exp(-std::min(d_side, d_cap) / 0.005f) * 0.2f;

    return std::min(score, 1.0f);
}

Eigen::Vector3f SampleQueue::to_local_frame(const Eigen::Vector3f& p, const BottleState& s) const
{
    // A cylinder is rotationally symmetric: no yaw, just translate to its centre.
    return {p.x() - s.cx, p.y() - s.cy, p.z() - s.cz};
}

void SampleQueue::refresh_sample_score(SamplePoint& sp, const BottleModel& model,
                                       const std::optional<float>& abs_sdf_override)
{
    sp.local_position = to_local_frame(sp.position, model.state());
    sp.last_abs_sdf = abs_sdf_override.has_value() ? abs_sdf_override.value() : std::abs(model.sdf_point(sp.position));
    sp.edge_score_cache = edge_score(sp.position, model);
    sp.utility_score = sp.capture_cov.trace()
                     + params_.rfe_weight_gain * sp.rfe
                     + params_.current_sdf_weight * sp.last_abs_sdf
                     - params_.edge_bonus_weight * sp.edge_score_cache;
}

float SampleQueue::percentile(std::vector<float> values, float q)
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

void SampleQueue::update_metrics(const BottleModel* model)
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

    const BottleState* state = model ? &model->state() : nullptr;
    for (const auto& sp : pts_)
    {
        const float weight = anchor_weight(sp, params_);
        weight_sum += weight;
        rfe_values.push_back(sp.rfe);
        abs_sdf_sum += sp.last_abs_sdf;
        if (sp.edge_score_cache >= params_.edge_anchor_score_threshold)
            edge_like_count += 1.0f;

        if (state != nullptr)
            occupied_bins.insert(bin_index(sp.position, *state));
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

// ─── Flush bins ──────────────────────────────────────────────────────────────

void SampleQueue::flush_bins(std::unordered_map<int, std::vector<BinEntry>>& bins,
                             const BottleModel& model)
{
    std::vector<SamplePoint> next_pts;
    next_pts.reserve(capacity());
    for (auto& [idx, entries] : bins)
    {
        // Sort by quality ascending (lower utility = better point)
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

// ─── Insert ──────────────────────────────────────────────────────────────────

void SampleQueue::insert(const std::vector<Eigen::Vector3f>& new_pts,
                         const std::vector<float>&           sdf_values,
                         const Eigen::Matrix2f&              robot_cov,
                         const BottleModel&                  model,
                         int                                 frame_count)
{
    // Admission warmup: wait for pose to converge first
    if (frame_count < params_.min_frames_before_historical)
    {
        counters_.rejected_warmup += static_cast<int>(new_pts.size());
        metrics_.counters = counters_;
        return;
    }

    refresh_scores(model);

    // Maximum points this frame (gradual ramp)
    const float progress = std::min(1.0f,
        static_cast<float>(frame_count - params_.min_frames_before_historical)
        / static_cast<float>(std::max(1, params_.historical_warmup_frames)));
    const int max_this_frame = std::max(1, static_cast<int>(
        progress * static_cast<float>(params_.max_new_points_per_frame)));

    // Filter candidates: only near-surface points
    std::vector<std::pair<float, int>> eligible;   // (|sdf|, index)
    for (std::size_t i = 0; i < new_pts.size() and i < sdf_values.size(); ++i)
    {
        const float abs_sdf = std::abs(sdf_values[i]);
        if (abs_sdf < params_.sdf_threshold_for_storage)
            eligible.emplace_back(abs_sdf, static_cast<int>(i));
        else
            ++counters_.rejected_sdf;
    }

    // Keep the max_this_frame candidates with smallest |SDF|
    if (static_cast<int>(eligible.size()) > max_this_frame)
    {
        counters_.rejected_frame_cap += static_cast<int>(eligible.size()) - max_this_frame;
        std::partial_sort(eligible.begin(),
                          eligible.begin() + max_this_frame,
                          eligible.end());
        eligible.resize(max_this_frame);
    }

    if (eligible.empty())
    {
        update_metrics(&model);
        return;
    }

    const auto& s = model.state();

    // Build bins from existing stored points
    std::unordered_map<int, std::vector<BinEntry>> bins;
    for (const auto& sp : pts_)
    {
        const int idx = bin_index(sp.position, s);
        bins[idx].push_back(BinEntry{sp, sp.utility_score, true});
    }

    // Insert new candidates into bins
    for (const auto& [abs_sdf, i] : eligible)
    {
        SamplePoint sp;
        sp.position    = new_pts[i];
        sp.capture_cov = robot_cov;
        sp.rfe         = 0.0f;
        sp.insertion_id = next_insertion_id_++;
        refresh_sample_score(sp, model, abs_sdf);

        const int idx = bin_index(sp.position, s);
        bins[idx].push_back(BinEntry{sp, sp.utility_score, false});
    }

    flush_bins(bins, model);
}

// ─── Update RFE ──────────────────────────────────────────────────────────────

void SampleQueue::update_rfe(const BottleModel& model, const Eigen::Matrix2f& robot_cov)
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

void SampleQueue::refresh_scores(const BottleModel& model)
{
    for (auto& sp : pts_)
        refresh_sample_score(sp, model);
    update_metrics(&model);
}

// ─── Points / weights ────────────────────────────────────────────────────────

std::vector<Eigen::Vector3f> SampleQueue::points() const
{
    std::vector<Eigen::Vector3f> out;
    out.reserve(pts_.size());
    for (const auto& sp : pts_)
        out.push_back(sp.position);
    return out;
}

std::vector<float> SampleQueue::weights() const
{
    std::vector<float> out;
    out.reserve(pts_.size());
    for (const auto& sp : pts_)
        out.push_back(anchor_weight(sp, params_));
    return out;
}

// ─── Coverage (diagnostic) ─────────────────────────────────────────────────────

std::array<float, 6> SampleQueue::face_coverage(const BottleModel& model) const
{
    // Buckets: 0=+x, 1=-x, 2=+y, 3=-y, 4=top cap, 5=bottom cap (bottle frame).
    std::array<float, 6> coverage{};
    coverage.fill(0.0f);

    if (pts_.empty())
        return coverage;

    const auto& s   = model.state();
    const float hh  = s.height * 0.5f;
    const float eps = 0.02f;

    for (const auto& sp : pts_)
    {
        const float lx = sp.position.x() - s.cx;
        const float ly = sp.position.y() - s.cy;
        const float lz = sp.position.z() - s.cz;
        const float w_i = anchor_weight(sp, params_);

        if (std::abs(lx) >= std::abs(ly))
            coverage[lx > 0.0f ? 0 : 1] += w_i;
        else
            coverage[ly > 0.0f ? 2 : 3] += w_i;

        if (lz >  hh - eps) coverage[4] += w_i;
        if (lz < -hh + eps) coverage[5] += w_i;
    }

    return coverage;
}

}  // namespace rc
