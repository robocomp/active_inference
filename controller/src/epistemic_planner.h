#pragma once

#include <vector>
#include <optional>
#include <chrono>
#include <random>
#include <Eigen/Dense>

#include "corner_visibility.h"

namespace rc
{

class EpistemicPlanner
{
public:
    struct Params
    {
        float grid_resolution = 0.5f;
        float min_distance = 1.0f;
        int   max_candidates = 2000;
        float target_wall_margin = 1.0f;
        float angular_dominance_ratio = 50.0f;
        float w_exploration  = 0.5f;
        float ior_cell_size   = 0.5f;
        float ior_decay_time  = 120.0f;
        float w_ior           = 2.0f;
        float fim_corner_sigma  = 0.04f;
        float fim_max_range     = 10.0f;
        float arrival_distance = 0.15f;
        float dwell_time       = 2.0f;
    };

    struct Target
    {
        Eigen::Vector2f position{0.f, 0.f};
        float score = 0.0f;
        float distance = 0.0f;
        float eigenvector_score = 0.0f;
        bool  rotate_in_place = false;
    };

    EpistemicPlanner();
    explicit EpistemicPlanner(Params params);

    void set_room_bounds(const Eigen::Vector2f& min_corner, const Eigen::Vector2f& max_corner);
    void set_room_polygon(const std::vector<Eigen::Vector2f>& vertices);
    void set_robot_state(const Eigen::Affine2f& pose, const Eigen::Matrix3f& covariance);

    std::vector<Target> evaluate_targets() const;
    std::optional<Target> select_target();
    std::optional<Target> update_target();

    void clear_target() { current_target_.reset(); }
    const std::optional<Target>& current_target() const { return current_target_; }

    struct CellScore
    {
        Eigen::Vector2f center;
        float score;
    };
    const std::vector<CellScore>& cell_scores() const { return cell_scores_; }
    float cell_size() const { return params.ior_cell_size; }

    Eigen::Vector2f robot_pos() const { return robot_pose_.translation(); }
    float robot_theta() const { return std::atan2(robot_pose_.linear()(1,0), robot_pose_.linear()(0,0)); }
    const Eigen::Affine2f& robot_pose() const { return robot_pose_; }
    const Eigen::Matrix3f& robot_cov()  const { return robot_cov_; }
    const std::vector<Eigen::Vector2f>& room_corners() const { return room_corners_; }
    const Eigen::Vector2f& room_min() const { return room_min_; }
    const Eigen::Vector2f& room_max() const { return room_max_; }
    bool room_bounds_set() const { return room_bounds_set_; }

    Params params;

private:
    std::vector<Eigen::Vector2f> generate_candidates() const;
    bool is_angular_dominated() const;
    float score_fim_gain(const Eigen::Vector2f& candidate,
                         const Eigen::Matrix3f& prior_precision) const;

    Eigen::Vector2f room_min_{0, 0};
    Eigen::Vector2f room_max_{0, 0};
    bool room_bounds_set_ = false;

    std::vector<Eigen::Vector2f> room_corners_;
    mutable std::vector<Eigen::Vector2f> cached_grid_;
    mutable bool grid_dirty_ = true;

    Eigen::Affine2f robot_pose_ = Eigen::Affine2f::Identity();
    Eigen::Matrix3f robot_cov_ = Eigen::Matrix3f::Identity();
    bool robot_state_set_ = false;

    std::optional<Target> current_target_;
    std::chrono::steady_clock::time_point dwell_until_{};
    bool dwelling_ = false;

    struct VisitGrid
    {
        struct Cell
        {
            std::chrono::steady_clock::time_point last_visit{};
            float fim_gain = 0.f;
        };
        std::vector<Cell> cells;
        int cols = 0, rows = 0;
        float cell_size = 0.5f;
        Eigen::Vector2f origin{0.f, 0.f};
        bool initialized = false;

        void init(const Eigen::Vector2f& room_min, const Eigen::Vector2f& room_max, float cs);
        int to_index(const Eigen::Vector2f& pos) const;
        Eigen::Vector2f cell_center(int idx) const;
        void mark_visited(const Eigen::Vector2f& pos);
        void update_fim(const Eigen::Vector2f& pos, float gain, float alpha = 0.3f);
        float staleness(const Eigen::Vector2f& pos, float decay_s,
                        std::chrono::steady_clock::time_point now) const;
    };
    VisitGrid visit_grid_;

    std::vector<CellScore> cell_scores_;
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace rc