/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see occupancy_belief.h.
 */

#include "occupancy_belief.h"

#include <algorithm>
#include <cmath>

#include "corner_visibility.h"   // point_in_polygon, ray_segment_t

namespace rc
{

namespace
{
/// Bernoulli entropy (bits) of an occupancy log-odds value. l=0 ⇒ p=0.5 ⇒ 1 bit.
inline float entropy_from_logodds(float l)
{
    const float p = 1.f / (1.f + std::exp(-l));
    if (p <= 1e-6f || p >= 1.f - 1e-6f) return 0.f;
    return -(p * std::log2(p) + (1.f - p) * std::log2(1.f - p));
}
}  // namespace

void OccupancyBelief::configure(const Eigen::Vector2f& room_min, const Eigen::Vector2f& room_max,
                                const std::vector<Eigen::Vector2f>& polygon, const Params& p)
{
    params_  = p;
    polygon_ = polygon;
    origin_  = room_min;
    cols_ = std::max(1, static_cast<int>(std::ceil((room_max.x() - room_min.x()) / params_.cell_size)));
    rows_ = std::max(1, static_cast<int>(std::ceil((room_max.y() - room_min.y()) / params_.cell_size)));

    logodds_.assign(static_cast<std::size_t>(cols_) * rows_, 0.f);     // unknown everywhere
    in_room_.assign(static_cast<std::size_t>(cols_) * rows_, 0);
    visit_stamp_.assign(static_cast<std::size_t>(cols_) * rows_, 0);
    epoch_ = 0;

    for (int i = 0; i < cols_ * rows_; ++i)
        in_room_[i] = (polygon_.size() < 3 ||
                       corner_visibility::point_in_polygon(cell_center(i), polygon_)) ? 1 : 0;
    configured_ = true;
}

int OccupancyBelief::index_of(const Eigen::Vector2f& pos) const
{
    const int c = std::clamp(static_cast<int>((pos.x() - origin_.x()) / params_.cell_size), 0, cols_ - 1);
    const int r = std::clamp(static_cast<int>((pos.y() - origin_.y()) / params_.cell_size), 0, rows_ - 1);
    return r * cols_ + c;
}

Eigen::Vector2f OccupancyBelief::cell_center(int idx) const
{
    const int c = idx % cols_;
    const int r = idx / cols_;
    return {origin_.x() + (c + 0.5f) * params_.cell_size,
            origin_.y() + (r + 0.5f) * params_.cell_size};
}

float OccupancyBelief::ray_block_distance(const Eigen::Vector2f& o, const Eigen::Vector2f& d) const
{
    float best = params_.sensor_range;

    // Walls (polygon edges).
    const int N = static_cast<int>(polygon_.size());
    for (int i = 0; i < N; ++i)
        if (auto t = corner_visibility::ray_segment_t(o, d, polygon_[i], polygon_[(i + 1) % N]);
            t && *t < best)
            best = *t;

    // Obstacles (circle approximation): nearest positive root of |o + t d − c|² = r².
    for (const auto& ob : obstacles_)
    {
        if (ob.radius <= 0.f) continue;
        const Eigen::Vector2f m = o - ob.center;
        const float b = m.dot(d);
        const float c = m.squaredNorm() - ob.radius * ob.radius;
        const float disc = b * b - c;
        if (disc < 0.f) continue;
        const float t = -b - std::sqrt(disc);     // nearest intersection
        if (t > 1e-3f && t < best) best = t;
    }
    return best;
}

template <class F>
void OccupancyBelief::march_visible(const Eigen::Vector2f& p, F&& f) const
{
    if (!configured_) return;
    ++epoch_;
    const float step = params_.cell_size;
    for (int k = 0; k < params_.n_rays; ++k)
    {
        const float ang = (2.f * static_cast<float>(M_PI) * k) / static_cast<float>(params_.n_rays);
        const Eigen::Vector2f d{std::cos(ang), std::sin(ang)};
        const float dmax = ray_block_distance(p, d);
        for (float s = step * 0.5f; s < dmax; s += step)
        {
            const int idx = index_of(p + s * d);
            if (!in_room_[idx]) continue;
            if (visit_stamp_[idx] == epoch_) continue;   // dedup across rays
            visit_stamp_[idx] = epoch_;
            f(idx);
        }
    }
}

void OccupancyBelief::observe_from(const Eigen::Vector2f& robot_pos)
{
    march_visible(robot_pos, [this](int idx)
    {
        logodds_[idx] = std::clamp(logodds_[idx] - params_.free_logodds,
                                   -params_.logodds_clamp, params_.logodds_clamp);
    });
}

float OccupancyBelief::expected_info_gain(const Eigen::Vector2f& cand) const
{
    float gain = 0.f;
    march_visible(cand, [this, &gain](int idx) { gain += entropy_from_logodds(logodds_[idx]); });
    return gain;
}

float OccupancyBelief::total_entropy() const
{
    float h = 0.f;
    for (int i = 0; i < cols_ * rows_; ++i)
        if (in_room_[i]) h += entropy_from_logodds(logodds_[i]);
    return h;
}

float OccupancyBelief::entropy_at(const Eigen::Vector2f& pos) const
{
    if (!configured_) return 0.f;
    const int idx = index_of(pos);
    return in_room_[idx] ? entropy_from_logodds(logodds_[idx]) : 0.f;
}

std::vector<OccupancyBelief::EntropyCell> OccupancyBelief::entropy_field() const
{
    std::vector<EntropyCell> out;
    out.reserve(logodds_.size());
    for (int i = 0; i < cols_ * rows_; ++i)
        if (in_room_[i])
            out.push_back({cell_center(i), entropy_from_logodds(logodds_[i])});
    return out;
}

}  // namespace rc
