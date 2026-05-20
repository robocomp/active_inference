#include "epistemic_controller.h"

#include <algorithm>
#include <cmath>

namespace rc
{

EpistemicController::EpistemicController() : params{} {}
EpistemicController::EpistemicController(Params planner_params, EpistemicPlanner::Params selector_params)
    : params(std::move(planner_params)), epistemic_planner_(std::move(selector_params)) {}

void EpistemicController::set_room_bounds(const Eigen::Vector2f& min_corner,
                                          const Eigen::Vector2f& max_corner)
{
    epistemic_planner_.set_room_bounds(min_corner, max_corner);
}

void EpistemicController::set_room_polygon(const std::vector<Eigen::Vector2f>& vertices)
{
    epistemic_planner_.set_room_polygon(vertices);

    edge_segments_.clear();
    edge_segments_.reserve(vertices.size());
    for (std::size_t index = 0; index < vertices.size(); ++index)
    {
        const auto& a = vertices[index];
        const Eigen::Vector2f ab = vertices[(index + 1) % vertices.size()] - a;
        edge_segments_.push_back({a, ab, ab.squaredNorm()});
    }
}

void EpistemicController::set_robot_state(const Eigen::Affine2f& pose,
                                          const Eigen::Matrix3f& covariance)
{
    epistemic_planner_.set_robot_state(pose, covariance);
}

void EpistemicController::set_lidar_obstacles(std::vector<Eigen::Vector2f> points)
{
    if (!edge_segments_.empty())
    {
        const float margin2 = params.wall_filter_margin * params.wall_filter_margin;
        std::vector<Eigen::Vector2f> filtered;
        filtered.reserve(points.size());

        for (const auto& point : points)
        {
            bool near_wall = false;
            for (const auto& edge : edge_segments_)
            {
                const float t = std::clamp((point - edge.a).dot(edge.ab) / edge.ab_sq_norm, 0.f, 1.f);
                const float d2 = (point - (edge.a + t * edge.ab)).squaredNorm();
                if (d2 < margin2) { near_wall = true; break; }
            }
            if (!near_wall)
                filtered.push_back(point);
        }
        lidar_obstacles_ = std::move(filtered);
    }
    else
        lidar_obstacles_ = std::move(points);
}

void EpistemicController::set_localization_quality(float sdf_mse)
{
    const float range = params.sdf_danger - params.sdf_safe;
    if (range < 1e-9f)
        governor_alpha_ = 1.f;
    else
        governor_alpha_ = std::clamp(1.f - (sdf_mse - params.sdf_safe) / range,
                                     params.governor_alpha_min, 1.f);
}

EpistemicController::ControlCommand EpistemicController::apply_speed_limit(ControlCommand cmd) const
{
    const float eff_rot_max = params.max_rot_speed * governor_alpha_;
    const float eff_adv_max = params.max_adv_speed * governor_alpha_;

    cmd.rot = std::clamp(cmd.rot, -eff_rot_max, eff_rot_max);
    const float rot_frac = std::abs(cmd.rot) / std::max(1e-6f, eff_rot_max);
    const float eff_max  = eff_adv_max * std::max(0.f, 1.f - params.bandwidth_coupling * rot_frac);
    const float adv_norm = std::sqrt(cmd.adv_x * cmd.adv_x + cmd.adv_y * cmd.adv_y);
    if (adv_norm > eff_max && adv_norm > 1e-6f)
    {
        const float scale = eff_max / adv_norm;
        cmd.adv_x *= scale;
        cmd.adv_y *= scale;
    }
    return cmd;
}

std::vector<EpistemicController::Policy>
EpistemicController::generate_arc_policies(const EpistemicPlanner::Target& target) const
{
    std::vector<Policy> policies;
    const int count = params.num_arc_curvatures;

    if (target.rotate_in_place)
    {
        for (int index = 0; index < std::max(count, 2); ++index)
        {
            const float frac = static_cast<float>(index) / std::max(1.f, static_cast<float>(count - 1));
            const float rot  = params.max_rot_speed * (2.f * frac - 1.f);
            Policy policy;
            policy.commands.resize(params.horizon_steps, {0.f, 0.f, rot});
            policies.push_back(std::move(policy));
        }
        return policies;
    }

    const Eigen::Vector2f tb = epistemic_planner_.robot_pose().inverse() * target.position;
    const float angle_err  = std::atan2(-tb.x(), tb.y());
    const float dist       = tb.norm();
    const float horizon_time = static_cast<float>(params.horizon_steps) * params.dt;
    const float speed_horizon = std::min(horizon_time, params.speed_horizon_s);
    const float nominal_speed = std::min(dist / speed_horizon, params.max_adv_speed);
    const float nominal_rot   = std::clamp(params.k_rot * angle_err,
                                           -params.max_rot_speed, params.max_rot_speed);
    const Eigen::Vector2f tb_dir = (dist > 1e-3f) ? tb / dist : Eigen::Vector2f{0.f, 1.f};

    for (int index = 0; index < count; ++index)
    {
        const float frac = (count > 1)
            ? 2.f * static_cast<float>(index) / static_cast<float>(count - 1) - 1.f
            : 0.f;

        const float rot = std::clamp(nominal_rot + frac * params.max_rot_speed,
                                     -params.max_rot_speed, params.max_rot_speed);
        const float speed = std::max(0.f, nominal_speed * (1.f - 0.4f * std::abs(frac)));

        const auto arc_cmd = apply_speed_limit({tb_dir.x() * speed, tb_dir.y() * speed, rot});
        Policy policy;
        policy.commands.resize(params.horizon_steps, arc_cmd);
        policies.push_back(std::move(policy));
    }

    for (float sign : {-1.f, -0.5f, 0.5f, 1.f})
    {
        Policy policy;
        policy.commands.resize(params.horizon_steps, {0.f, 0.f, sign * params.max_rot_speed});
        policies.push_back(std::move(policy));
    }

    return policies;
}

void EpistemicController::rollout_policy(Policy& policy) const
{
    policy.predicted_states.clear();
    policy.predicted_states.reserve(params.horizon_steps + 1);

    Eigen::Vector3f state{epistemic_planner_.robot_pos().x(),
                          epistemic_planner_.robot_pos().y(),
                          epistemic_planner_.robot_theta()};
    policy.predicted_states.push_back(state);

    for (int step = 0; step < params.horizon_steps; ++step)
    {
        const auto& cmd = policy.commands[step];
        const float ct = std::cos(state[2]);
        const float st = std::sin(state[2]);

        state[0] += (cmd.adv_x * ct - cmd.adv_y * st) * params.dt;
        state[1] += (cmd.adv_x * st + cmd.adv_y * ct) * params.dt;
        state[2] += cmd.rot * params.dt;
        state[2] = std::atan2(std::sin(state[2]), std::cos(state[2]));

        policy.predicted_states.push_back(state);
    }
}

void EpistemicController::evaluate_policy_efe(Policy& policy,
                                              const EpistemicPlanner::Target& target,
                                              const Eigen::Matrix3f& prior_precision) const
{
    float pragmatic = 0.f;
    float boundary  = 0.f;
    float obstacle  = 0.f;
    float epistemic = 0.f;

    const float obs_r2 = params.obstacle_radius * params.obstacle_radius;
    const auto& room_corners = epistemic_planner_.room_corners();
    const bool have_polygon = !room_corners.empty();

    bool boundary_fired = false;
    for (std::size_t step = 1; step < policy.predicted_states.size(); ++step)
    {
        const auto& state = policy.predicted_states[step];
        const Eigen::Vector2f pos{state[0], state[1]};

        bool out_of_bounds = false;
        if (have_polygon)
        {
            if (!corner_visibility::point_in_polygon(pos, room_corners))
                out_of_bounds = true;
        }
        else if (epistemic_planner_.room_bounds_set() &&
                 (pos.x() < epistemic_planner_.room_min().x() || pos.x() > epistemic_planner_.room_max().x() ||
                  pos.y() < epistemic_planner_.room_min().y() || pos.y() > epistemic_planner_.room_max().y()))
        {
            out_of_bounds = true;
        }

        if (out_of_bounds)
        {
            if (!boundary_fired)
            {
                boundary += params.w_boundary;
                boundary_fired = true;
            }
            continue;
        }

        pragmatic += (pos - target.position).squaredNorm();

        const Eigen::Vector2f diff = target.position - pos;
        if (diff.squaredNorm() > 0.01f)
        {
            const float ct = std::cos(state[2]);
            const float st = std::sin(state[2]);
            const float bx =  ct * diff.x() + st * diff.y();
            const float by = -st * diff.x() + ct * diff.y();
            const float heading_err = std::atan2(-bx, by);
            pragmatic += params.w_heading * heading_err * heading_err;
        }

        float step_obs = 0.f;
        for (const auto& obstacle_point : lidar_obstacles_)
        {
            const float d2 = (pos - obstacle_point).squaredNorm();
            if (d2 < obs_r2 && d2 > 1e-6f)
                step_obs += params.obstacle_k / d2;
        }
        obstacle += std::min(step_obs, params.obstacle_step_cap);
    }

    if (have_polygon && policy.predicted_states.size() > 1)
    {
        const auto& final_state = policy.predicted_states.back();
        const Eigen::Vector2f final_pos{final_state[0], final_state[1]};
        const float final_theta = final_state[2];

        auto visible = corner_visibility::visible_corners(final_pos, room_corners, params.fim_max_range);
        if (!visible.empty())
        {
            const auto fim = corner_visibility::corner_fim(final_pos, final_theta, visible,
                                                           room_corners, params.fim_corner_sigma);
            epistemic = corner_visibility::d_optimality_gain(prior_precision, fim);
        }
    }

    policy.epistemic_value = epistemic;
    policy.pragmatic_value = pragmatic;
    policy.obstacle_value = obstacle;
    policy.boundary_value = boundary;
    policy.efe = -params.w_epistemic * epistemic
               + params.w_pragmatic * pragmatic
               + params.w_obstacle  * obstacle
               + boundary;
}

std::optional<EpistemicController::PlanResult> EpistemicController::plan_for_target(
    const EpistemicPlanner::Target& target) const
{
    const Eigen::Matrix3f reg_cov = epistemic_planner_.robot_cov() + 1e-6f * Eigen::Matrix3f::Identity();
    const Eigen::Matrix3f prior_precision = reg_cov.inverse();

    auto policies = generate_arc_policies(target);
    for (auto& policy : policies)
    {
        rollout_policy(policy);
        evaluate_policy_efe(policy, target, prior_precision);
    }

    auto best = std::min_element(policies.begin(), policies.end(),
                                 [](const Policy& a, const Policy& b) { return a.efe < b.efe; });
    if (best != policies.end() && !best->commands.empty())
    {
        auto best_policy = *best;
        const auto final_cmd = apply_speed_limit(best_policy.commands.front());
        return PlanResult{
            .command       = final_cmd,
            .target        = target,
            .best_policy   = std::move(best_policy),
            .all_policies  = std::move(policies),
            .valid         = true
        };
    }

    ControlCommand cmd{};
    if (target.rotate_in_place)
    {
        cmd.rot = params.max_rot_speed * 0.5f;
    }
    else
    {
        const Eigen::Vector2f tb = epistemic_planner_.robot_pose().inverse() * target.position;
        const float ae = std::atan2(-tb.x(), tb.y());
        cmd.rot   = std::clamp(params.k_rot * ae, -params.max_rot_speed, params.max_rot_speed);
        cmd.adv_y = params.max_adv_speed
                  * std::exp(-0.5f * ae * ae / (params.gaussian_sigma * params.gaussian_sigma));
    }
    return PlanResult{.command = apply_speed_limit(cmd), .target = target, .best_policy = {}, .valid = true};
}

std::optional<EpistemicController::PlanResult> EpistemicController::plan()
{
    auto target_opt = epistemic_planner_.update_target();
    if (!target_opt)
        return std::nullopt;
    return plan_for_target(*target_opt);
}

std::optional<EpistemicController::PlanResult> EpistemicController::plan_to_target(
    const Eigen::Vector2f& target_position,
    bool rotate_in_place)
{
    EpistemicPlanner::Target target;
    target.position = target_position;
    target.rotate_in_place = rotate_in_place;
    target.distance = (target.position - epistemic_planner_.robot_pos()).norm();
    target.score = 1.f;
    return plan_for_target(target);
}

} // namespace rc