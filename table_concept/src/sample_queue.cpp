/*
 * sample_queue.cpp
 *
 * Ported from box_concept/src/objects/table/belief.py:
 *   add_historical_points, _add_to_bins, _compute_edge_score, update_rfe,
 *   get_historical_points_in_robot_frame
 *
 * Frame change from prototype: all voxel centroids arrive already in room
 * frame from robot_concept, so the robot↔room Jacobian propagation is omitted.
 * Gradient weight reduces to: w_i = 1 / (1 + tr(Σ_capture_i) + RFE_i)
 */

#include "sample_queue.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <tuple>

// ─── Constructor ─────────────────────────────────────────────────────────────

SampleQueue::SampleQueue(const SampleQueueParams& params)
    : params_(params)
{}

// ─── Bin index ───────────────────────────────────────────────────────────────

int SampleQueue::bin_index(const Eigen::Vector3f& p, float cx, float cy) const
{
    const int na = params_.num_angle_bins;
    const int nz = params_.num_z_bins;

    const float angle = std::atan2(p.y() - cy, p.x() - cx);
    int ab = static_cast<int>((angle + std::numbers::pi_v<float>) / (2.0f * std::numbers::pi_v<float>) * na) % na;
    if (ab < 0) ab += na;

    int zb = static_cast<int>(p.z() / params_.z_bin_size);
    zb = std::clamp(zb, 0, nz - 1);

    return ab * nz + zb;
}

// ─── Edge score ──────────────────────────────────────────────────────────────

float SampleQueue::edge_score(const Eigen::Vector3f& p, const TableModel& model) const
{
    const auto& s = model.state();
    const float half_w = s.w * 0.5f;
    const float half_h = s.h * 0.5f;
    const float half_t = TableModel::TOP_THICKNESS * 0.5f;
    const float top_cz = s.table_height - half_t;

    // Transform to table-local frame
    const float cos_t = std::cos(-s.yaw);
    const float sin_t = std::sin(-s.yaw);
    const float px = p.x() - s.cx;
    const float py = p.y() - s.cy;
    const float lx = px * cos_t - py * sin_t;
    const float ly = px * sin_t + py * cos_t;
    const float lz = p.z();

    const float th = params_.edge_proximity_threshold;

    // Distance to each face boundary
    const float dist_x = std::abs(std::abs(lx) - half_w);
    const float dist_y = std::abs(std::abs(ly) - half_h);
    const float dist_z = std::abs(lz - top_cz);

    const float close_x = (dist_x < th) ? 1.0f : 0.0f;
    const float close_y = (dist_y < th) ? 1.0f : 0.0f;
    const float close_z = (dist_z < half_t + th) ? 1.0f : 0.0f;

    const float faces_close = close_x + close_y + close_z;

    // Score: 0.0 = flat face, 0.5 = edge (near 2 faces), 1.0 = corner (near 3)
    float score = std::max(0.0f, faces_close - 1.0f) / 2.0f;

    // Proximity bonus for being very close to edges
    const float min_dist = std::min(dist_x, dist_y);
    score += std::exp(-min_dist / 0.02f) * 0.2f;

    return std::min(score, 1.0f);
}

// ─── Flush bins ──────────────────────────────────────────────────────────────

void SampleQueue::flush_bins(
    std::unordered_map<int, std::vector<std::tuple<SamplePoint, float>>>& bins)
{
    pts_.clear();
    for (auto& [idx, entries] : bins)
    {
        // Sort by quality ascending (lower quality = better point)
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) {
                      return std::get<1>(a) < std::get<1>(b);
                  });
        const int keep = std::min(static_cast<int>(entries.size()), params_.max_per_bin);
        for (int i = 0; i < keep; ++i)
            pts_.push_back(std::get<0>(entries[i]));
    }
}

// ─── Insert ──────────────────────────────────────────────────────────────────

void SampleQueue::insert(const std::vector<Eigen::Vector3f>& new_pts,
                          const std::vector<float>&           sdf_values,
                          const Eigen::Matrix2f&              robot_cov,
                          const TableModel&                   model,
                          int                                 frame_count)
{
    // Admission warmup: wait for pose to converge first
    if (frame_count < params_.min_frames_before_historical)
        return;

    // Maximum points this frame (gradual ramp)
    const float progress = std::min(1.0f,
        static_cast<float>(frame_count - params_.min_frames_before_historical)
        / static_cast<float>(params_.historical_warmup_frames));
    const int max_this_frame = std::max(1, static_cast<int>(
        progress * static_cast<float>(params_.max_new_points_per_frame)));

    // Filter candidates: only near-surface points
    std::vector<std::pair<float, int>> eligible;   // (|sdf|, index)
    for (std::size_t i = 0; i < new_pts.size() and i < sdf_values.size(); ++i)
    {
        const float abs_sdf = std::abs(sdf_values[i]);
        if (abs_sdf < params_.sdf_threshold_for_storage)
            eligible.emplace_back(abs_sdf, static_cast<int>(i));
    }

    // Keep the max_this_frame candidates with smallest |SDF|
    if (static_cast<int>(eligible.size()) > max_this_frame)
    {
        std::partial_sort(eligible.begin(),
                          eligible.begin() + max_this_frame,
                          eligible.end());
        eligible.resize(max_this_frame);
    }

    if (eligible.empty())
        return;

    const float cx = model.state().cx;
    const float cy = model.state().cy;

    // Build bins from existing stored points
    std::unordered_map<int, std::vector<std::tuple<SamplePoint, float>>> bins;
    for (const auto& sp : pts_)
    {
        const int idx = bin_index(sp.position, cx, cy);
        const float es = edge_score(sp.position, model);
        const float quality = sp.capture_cov.trace() + sp.rfe - params_.edge_bonus_weight * es;
        bins[idx].emplace_back(sp, quality);
    }

    // Insert new candidates into bins
    for (const auto& [abs_sdf, i] : eligible)
    {
        SamplePoint sp;
        sp.position    = new_pts[i];
        sp.capture_cov = robot_cov;
        sp.rfe         = 0.0f;

        const int idx = bin_index(sp.position, cx, cy);
        const float es = edge_score(sp.position, model);
        const float quality = robot_cov.trace() + 0.0f - params_.edge_bonus_weight * es;
        bins[idx].emplace_back(sp, quality);
    }

    flush_bins(bins);
}

// ─── Update RFE ──────────────────────────────────────────────────────────────

void SampleQueue::update_rfe(const TableModel& model, const Eigen::Matrix2f& robot_cov)
{
    if (pts_.empty())
        return;

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
    }

    pts_ = std::move(survivors);
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
    {
        const float cov_trace = sp.capture_cov.trace();
        out.push_back(1.0f / (1.0f + cov_trace + sp.rfe));
    }
    return out;
}

// ─── Face coverage ───────────────────────────────────────────────────────────

std::array<float, 6> SampleQueue::face_coverage(const TableModel& model) const
{
    // Faces indexed as: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z(top), 5=-z(bottom)
    std::array<float, 6> coverage{};
    coverage.fill(0.0f);

    if (pts_.empty())
        return coverage;

    const auto& s    = model.state();
    const float hw   = s.w * 0.5f;
    const float hh   = s.h * 0.5f;
    const float eps  = 0.05f;
    const float cos_t = std::cos(-s.yaw);
    const float sin_t = std::sin(-s.yaw);

    for (const auto& sp : pts_)
    {
        const float px = sp.position.x() - s.cx;
        const float py = sp.position.y() - s.cy;
        const float lx = px * cos_t - py * sin_t;
        const float ly = px * sin_t + py * cos_t;

        const float w_i = 1.0f / (1.0f + sp.capture_cov.trace() + sp.rfe);

        if (std::abs(lx - hw)  < eps) coverage[0] += w_i;  // +x face
        if (std::abs(lx + hw)  < eps) coverage[1] += w_i;  // -x face
        if (std::abs(ly - hh)  < eps) coverage[2] += w_i;  // +y face
        if (std::abs(ly + hh)  < eps) coverage[3] += w_i;  // -y face
        // +z top face
        if (std::abs(sp.position.z() - s.table_height) < eps) coverage[4] += w_i;
        // -z bottom (not reachable by robot, skip)
    }

    return coverage;
}
