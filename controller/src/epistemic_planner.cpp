#include "epistemic_planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace rc
{

EpistemicPlanner::EpistemicPlanner() : params{} {}
EpistemicPlanner::EpistemicPlanner(Params p) : params(std::move(p)) {}

void EpistemicPlanner::VisitGrid::init(const Eigen::Vector2f& room_min, const Eigen::Vector2f& room_max, float cs)
{
    cell_size = cs;
    origin = room_min;
    cols = static_cast<int>(std::ceil((room_max.x() - room_min.x()) / cell_size));
    rows = static_cast<int>(std::ceil((room_max.y() - room_min.y()) / cell_size));
    cols = std::max(1, cols);
    rows = std::max(1, rows);
    cells.assign(cols * rows, Cell{});
    initialized = true;
}

int EpistemicPlanner::VisitGrid::to_index(const Eigen::Vector2f& pos) const
{
    int c = static_cast<int>((pos.x() - origin.x()) / cell_size);
    int r = static_cast<int>((pos.y() - origin.y()) / cell_size);
    c = std::clamp(c, 0, cols - 1);
    r = std::clamp(r, 0, rows - 1);
    return r * cols + c;
}

Eigen::Vector2f EpistemicPlanner::VisitGrid::cell_center(int idx) const
{
    const int c = idx % cols;
    const int r = idx / cols;
    return {origin.x() + (c + 0.5f) * cell_size,
            origin.y() + (r + 0.5f) * cell_size};
}

void EpistemicPlanner::VisitGrid::mark_visited(const Eigen::Vector2f& pos)
{
    if (!initialized) return;
    cells[to_index(pos)].last_visit = std::chrono::steady_clock::now();
}

void EpistemicPlanner::VisitGrid::update_fim(const Eigen::Vector2f& pos, float gain, float alpha)
{
    if (!initialized) return;
    auto& cell = cells[to_index(pos)];
    cell.fim_gain = (1.f - alpha) * cell.fim_gain + alpha * gain;
}

float EpistemicPlanner::VisitGrid::staleness(const Eigen::Vector2f& pos,
                                             float decay_s,
                                             std::chrono::steady_clock::time_point now) const
{
    if (!initialized) return 1.f;
    const auto& tp = cells[to_index(pos)].last_visit;
    if (tp == std::chrono::steady_clock::time_point{}) return 1.f;
    const float elapsed = std::chrono::duration<float>(now - tp).count();
    return std::min(1.f, elapsed / std::max(0.1f, decay_s));
}

void EpistemicPlanner::set_room_bounds(const Eigen::Vector2f& min_corner,
                                       const Eigen::Vector2f& max_corner)
{
    room_min_ = min_corner;
    room_max_ = max_corner;
    room_bounds_set_ = true;
    grid_dirty_ = true;
    if (!visit_grid_.initialized)
        visit_grid_.init(room_min_, room_max_, params.ior_cell_size);
}

void EpistemicPlanner::set_room_polygon(const std::vector<Eigen::Vector2f>& vertices)
{
    room_corners_ = vertices;
    grid_dirty_ = true;
}

void EpistemicPlanner::set_robot_state(const Eigen::Affine2f& pose,
                                       const Eigen::Matrix3f& covariance)
{
    robot_pose_ = pose;
    robot_cov_ = covariance;
    robot_state_set_ = true;
}

std::vector<Eigen::Vector2f> EpistemicPlanner::generate_candidates() const
{
    if (!room_bounds_set_)
        return {};

    if (grid_dirty_)
    {
        cached_grid_.clear();
        const float res = params.grid_resolution;
        const float wall_margin_sq = params.target_wall_margin * params.target_wall_margin;

        for (float x = room_min_.x() + res * 0.5f; x < room_max_.x(); x += res)
            for (float y = room_min_.y() + res * 0.5f; y < room_max_.y(); y += res)
            {
                const Eigen::Vector2f p{x, y};
                if (!room_corners_.empty() && !corner_visibility::point_in_polygon(p, room_corners_))
                    continue;
                if (!room_corners_.empty())
                {
                    bool too_close = false;
                    for (std::size_t index = 0; index < room_corners_.size(); ++index)
                    {
                        const auto& a = room_corners_[index];
                        const auto& b = room_corners_[(index + 1) % room_corners_.size()];
                        const Eigen::Vector2f ab = b - a;
                        const float t = std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f);
                        if ((p - (a + t * ab)).squaredNorm() < wall_margin_sq)
                        {
                            too_close = true;
                            break;
                        }
                    }
                    if (too_close) continue;
                }
                cached_grid_.emplace_back(p);
            }
        grid_dirty_ = false;
    }

    const float min_distance_sq = params.min_distance * params.min_distance;
    std::vector<Eigen::Vector2f> candidates;
    candidates.reserve(std::min(static_cast<int>(cached_grid_.size()), params.max_candidates));

    for (const auto& point : cached_grid_)
    {
        if ((point - robot_pos()).squaredNorm() < min_distance_sq)
            continue;
        candidates.emplace_back(point);
        if (static_cast<int>(candidates.size()) >= params.max_candidates)
            break;
    }
    return candidates;
}

bool EpistemicPlanner::is_angular_dominated() const
{
    const float sigma2_theta = robot_cov_(2, 2);
    const float max_pos = std::max(robot_cov_(0, 0), robot_cov_(1, 1));
    if (max_pos < 1e-9f) return false;
    return (sigma2_theta / max_pos) > params.angular_dominance_ratio;
}

float EpistemicPlanner::score_fim_gain(const Eigen::Vector2f& candidate,
                                       const Eigen::Matrix3f& prior_precision) const
{
    if (room_corners_.empty()) return 0.f;

    auto visible = corner_visibility::visible_corners(candidate, room_corners_, params.fim_max_range);
    if (visible.empty()) return 0.f;

    const Eigen::Vector2f dir = candidate - robot_pos();
    const float heading = (dir.squaredNorm() > 1e-4f)
        ? std::atan2(-dir.x(), dir.y())
        : robot_theta();

    const auto fim = corner_visibility::corner_fim(candidate, heading, visible,
                                                   room_corners_, params.fim_corner_sigma);
    return corner_visibility::d_optimality_gain(prior_precision, fim);
}

std::vector<EpistemicPlanner::Target> EpistemicPlanner::evaluate_targets() const
{
    auto& self = const_cast<EpistemicPlanner&>(*this);

    if (!room_bounds_set_ || !robot_state_set_)
    {
        self.cell_scores_.clear();
        return {};
    }

    if (is_angular_dominated())
    {
        Target rot;
        rot.position = robot_pos();
        rot.distance = 0.f;
        rot.rotate_in_place = true;
        rot.score = robot_cov_(2, 2);
        self.cell_scores_.clear();
        return {rot};
    }

    const auto candidates = generate_candidates();
    if (candidates.empty())
    {
        self.cell_scores_.clear();
        return {};
    }

    const Eigen::Matrix3f reg_cov = robot_cov_ + 1e-6f * Eigen::Matrix3f::Identity();
    const Eigen::Matrix3f prior_precision = reg_cov.inverse();
    const auto now = std::chrono::steady_clock::now();

    std::vector<Target> targets;
    targets.reserve(candidates.size());

    for (const auto& pos : candidates)
    {
        Target target;
        target.position = pos;
        target.distance = (pos - robot_pos()).norm();
        const float fim_gain = score_fim_gain(pos, prior_precision);

        self.visit_grid_.update_fim(pos, fim_gain);

        const float dist_bonus = 1.f + params.w_exploration * target.distance;
        const float ior_bonus = 1.f + params.w_ior * visit_grid_.staleness(pos, params.ior_decay_time, now);
        target.score = fim_gain * dist_bonus * ior_bonus;
        target.eigenvector_score = fim_gain;
        targets.push_back(target);
    }

    std::sort(targets.begin(), targets.end(), [](const Target& a, const Target& b) { return a.score > b.score; });

    if (visit_grid_.initialized)
    {
        self.cell_scores_.clear();
        self.cell_scores_.reserve(visit_grid_.cells.size());
        for (int index = 0; index < static_cast<int>(visit_grid_.cells.size()); ++index)
        {
            const auto center = visit_grid_.cell_center(index);
            if (!room_corners_.empty() && !corner_visibility::point_in_polygon(center, room_corners_))
                continue;
            const auto& cell = visit_grid_.cells[index];
            const float stale = visit_grid_.staleness(center, params.ior_decay_time, now);
            const float combined = cell.fim_gain * (1.f + params.w_ior * stale);
            self.cell_scores_.push_back({center, combined});
        }
    }

    return targets;
}

std::optional<EpistemicPlanner::Target> EpistemicPlanner::select_target()
{
    auto targets = evaluate_targets();
    if (targets.empty())
        return std::nullopt;

    if (targets.front().rotate_in_place)
        return targets.front();

    std::vector<float> weights(targets.size());
    for (std::size_t index = 0; index < targets.size(); ++index)
        weights[index] = std::max(0.f, targets[index].score);

    const float total = std::accumulate(weights.begin(), weights.end(), 0.f);
    if (total < 1e-12f)
        return targets.front();

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return targets[distribution(rng_)];
}

std::optional<EpistemicPlanner::Target> EpistemicPlanner::update_target()
{
    if (dwelling_)
    {
        if (std::chrono::steady_clock::now() < dwell_until_)
            return current_target_;
        dwelling_ = false;
        current_target_.reset();
    }

    visit_grid_.mark_visited(robot_pos());

    if (current_target_.has_value() && !current_target_->rotate_in_place)
    {
        const float dist = (current_target_->position - robot_pos()).norm();
        if (dist < params.arrival_distance)
        {
            dwelling_ = true;
            dwell_until_ = std::chrono::steady_clock::now()
                         + std::chrono::milliseconds(static_cast<int>(params.dwell_time * 1000.f));
            return current_target_;
        }
    }

    if (!current_target_.has_value())
    {
        auto target_opt = select_target();
        if (!target_opt)
            return std::nullopt;
        current_target_ = *target_opt;
    }

    return current_target_;
}

} // namespace rc