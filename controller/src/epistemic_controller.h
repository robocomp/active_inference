#pragma once

#include <vector>
#include <optional>
#include <Eigen/Dense>

#include "epistemic_planner.h"

namespace rc
{

class EpistemicController
{
public:
    struct Params
    {
        int   num_arc_curvatures = 9;
        int   horizon_steps  = 20;
        float dt             = 0.2f;
        float max_adv_speed  = 0.9f;
        float max_rot_speed  = 0.75f;
        float w_epistemic    = 1.0f;
        float w_pragmatic    = 1.0f;
        float w_heading      = 2.0f;
        float w_boundary     = 10.0f;
        float k_rot  = 2.0f;
        float gaussian_sigma = 0.5f;
        float speed_horizon_s = 1.0f;
        float obstacle_radius  = 0.35f;
        float obstacle_k       = 0.5f;
        float obstacle_step_cap = 10.0f;
        float w_obstacle       = 2.0f;
        float wall_filter_margin = 0.30f;
        float bandwidth_coupling = 0.7f;
        float sdf_safe    = 0.06f;
        float sdf_danger  = 0.10f;
        float governor_alpha_min = 0.4f;
        float fim_corner_sigma  = 0.04f;
        float fim_max_range     = 10.0f;
    };

    struct ControlCommand
    {
        float adv_x = 0.f;
        float adv_y = 0.f;
        float rot   = 0.f;
    };

    struct Policy
    {
        std::vector<ControlCommand> commands;
        std::vector<Eigen::Vector3f> predicted_states;
        float efe             = 0.f;
        float epistemic_value = 0.f;
        float pragmatic_value = 0.f;
        float obstacle_value  = 0.f;
        float boundary_value  = 0.f;
    };

    struct PlanResult
    {
        ControlCommand command;
        EpistemicPlanner::Target target;
        Policy         best_policy;
        std::vector<Policy> all_policies;
        bool           valid = false;
    };

    EpistemicController();
    explicit EpistemicController(Params planner_params, EpistemicPlanner::Params selector_params = {});

    void set_room_bounds(const Eigen::Vector2f& min_corner, const Eigen::Vector2f& max_corner);
    void set_room_polygon(const std::vector<Eigen::Vector2f>& vertices);
    void set_robot_state(const Eigen::Affine2f& pose, const Eigen::Matrix3f& covariance);
    void set_lidar_obstacles(std::vector<Eigen::Vector2f> points);
    void set_localization_quality(float sdf_mse);

    std::optional<PlanResult> plan();
    std::optional<PlanResult> plan_to_target(const Eigen::Vector2f& target_position, bool rotate_in_place = false);

    void clear_target() { epistemic_planner_.clear_target(); }
    float governor_alpha() const { return governor_alpha_; }

    EpistemicPlanner&       epistemic_planner()       { return epistemic_planner_; }
    const EpistemicPlanner& epistemic_planner() const { return epistemic_planner_; }

    Params params;

private:
    std::optional<PlanResult> plan_for_target(const EpistemicPlanner::Target& target) const;
    std::vector<Policy> generate_arc_policies(const EpistemicPlanner::Target& target) const;
    void rollout_policy(Policy& policy) const;
    void evaluate_policy_efe(Policy& policy, const EpistemicPlanner::Target& target,
                             const Eigen::Matrix3f& prior_precision) const;
    ControlCommand apply_speed_limit(ControlCommand cmd) const;

    struct EdgeSegment { Eigen::Vector2f a, ab; float ab_sq_norm; };
    std::vector<EdgeSegment> edge_segments_;

    EpistemicPlanner epistemic_planner_;
    std::vector<Eigen::Vector2f> lidar_obstacles_;
    float governor_alpha_ = 1.f;
};

} // namespace rc