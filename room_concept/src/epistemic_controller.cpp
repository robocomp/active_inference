#include "epistemic_controller.h"
#include <algorithm>
#include <cmath>

namespace rc
{

EpistemicController::EpistemicController() : params{} {}
EpistemicController::EpistemicController(Params planner_params, EpistemicPlanner::Params selector_params)
    : params(std::move(planner_params)), epistemic_planner_(std::move(selector_params)) {}

// ---------------------------------------------------------------------------
// State setters — forwarded to EpistemicPlanner + local obstacle state
// ---------------------------------------------------------------------------
void EpistemicController::set_room_bounds(const Eigen::Vector2f& min_corner,
                                       const Eigen::Vector2f& max_corner)
{
    epistemic_planner_.set_room_bounds(min_corner, max_corner);
}

void EpistemicController::set_room_polygon(const std::vector<Eigen::Vector2f>& vertices)
{
    epistemic_planner_.set_room_polygon(vertices);

    // Pre-compute edge segment data for wall filtering
    edge_segments_.clear();
    edge_segments_.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        const auto& a = vertices[i];
        const Eigen::Vector2f ab = vertices[(i + 1) % vertices.size()] - a;
        edge_segments_.push_back({a, ab, ab.squaredNorm()});
    }
}

void EpistemicController::set_robot_state(const Eigen::Affine2f& pose,
                                       const Eigen::Matrix3f& covariance)
{
    epistemic_planner_.set_robot_state(pose, covariance);
}

void EpistemicController::set_robot_footprint(float width_m, float length_m)
{
    robot_footprint_radius_ = 0.5f * std::hypot(width_m, length_m);
    epistemic_planner_.set_robot_footprint(width_m, length_m);
}

void EpistemicController::set_lidar_obstacles(std::vector<Eigen::Vector2f> points)
{
    if (!edge_segments_.empty())
    {
        const float margin2 = params.wall_filter_margin * params.wall_filter_margin;
        std::vector<Eigen::Vector2f> filtered;
        filtered.reserve(points.size());

        for (const auto& p : points)
        {
            bool near_wall = false;
            for (const auto& e : edge_segments_)
            {
                const float t = std::clamp((p - e.a).dot(e.ab) / e.ab_sq_norm, 0.f, 1.f);
                const float d2 = (p - (e.a + t * e.ab)).squaredNorm();
                if (d2 < margin2) { near_wall = true; break; }
            }
            if (!near_wall)
                filtered.push_back(p);
        }
        lidar_obstacles_ = std::move(filtered);
    }
    else
    {
        lidar_obstacles_ = std::move(points);
    }
}

float EpistemicController::nearest_wall_distance_sq(const Eigen::Vector2f& pos) const
{
    float min_distance_sq = std::numeric_limits<float>::infinity();
    for (const auto& e : edge_segments_)
    {
        if (e.ab_sq_norm <= 1e-9f)
            continue;

        const float t = std::clamp((pos - e.a).dot(e.ab) / e.ab_sq_norm, 0.f, 1.f);
        min_distance_sq = std::min(min_distance_sq, (pos - (e.a + t * e.ab)).squaredNorm());
    }
    return min_distance_sq;
}

// ===========================================================================
// Reactive speed governor — localization quality
// ===========================================================================
void EpistemicController::set_localization_quality(float sdf_mse)
{
    const float range = params.sdf_danger - params.sdf_safe;
    if (range < 1e-9f)
        governor_alpha_ = 1.f;
    else
        governor_alpha_ = std::clamp(1.f - (sdf_mse - params.sdf_safe) / range,
                                     params.governor_alpha_min, 1.f);
}

// ===========================================================================
// Perceptual-bandwidth speed limit
// ===========================================================================
EpistemicController::ControlCommand
EpistemicController::apply_speed_limit(ControlCommand cmd) const
{
    // Governor: scale both limits by localization quality
    const float eff_rot_max = params.max_rot_speed * governor_alpha_;
    const float eff_adv_max = params.max_adv_speed * governor_alpha_;

    cmd.rot = std::clamp(cmd.rot, -eff_rot_max, eff_rot_max);
    const float rot_frac = std::abs(cmd.rot) / std::max(1e-6f, eff_rot_max);
    const float eff_max  = eff_adv_max
                         * std::max(0.f, 1.f - params.bandwidth_coupling * rot_frac);
    const float adv_norm = std::sqrt(cmd.adv_x * cmd.adv_x + cmd.adv_y * cmd.adv_y);
    if (adv_norm > eff_max && adv_norm > 1e-6f)
    {
        const float scale = eff_max / adv_norm;
        cmd.adv_x *= scale;
        cmd.adv_y *= scale;
    }
    return cmd;
}

// ===========================================================================
// Level 2 — Arc trajectory generation
// ===========================================================================
std::vector<EpistemicController::Policy>
EpistemicController::generate_arc_policies(const EpistemicPlanner::Target& target) const
{
    std::vector<Policy> policies;
    const int K = params.num_arc_curvatures;

    // ---- Rotate-in-place target ----
    if (target.rotate_in_place)
    {
        for (int i = 0; i < std::max(K, 2); ++i)
        {
            const float frac = static_cast<float>(i) / std::max(1.f, static_cast<float>(K - 1));
            const float rot  = params.max_rot_speed * (2.f * frac - 1.f);
            Policy p;
            p.commands.resize(params.horizon_steps, {0.f, 0.f, rot});
            policies.push_back(std::move(p));
        }
        return policies;
    }

    // ---- Target-seeking nominal command ----
    // For an omni robot, translate directly toward the target in robot-local frame
    // regardless of heading. This decouples navigation from orientation.
    const Eigen::Vector2f tb = epistemic_planner_.robot_pose().inverse() * target.position;
    const float angle_err  = std::atan2(-tb.x(), tb.y());   // heading error (for rotation control)
    const float dist       = tb.norm();
    const float hor_time   = static_cast<float>(params.horizon_steps) * params.dt;
    const float speed_hor  = std::min(hor_time, params.speed_horizon_s);
    const float nom_speed  = std::min(dist / speed_hor, params.max_adv_speed);
    const float nom_rot    = std::clamp(params.k_rot * angle_err,
                                         -params.max_rot_speed, params.max_rot_speed);

    // Body-frame unit vector pointing toward target (omni: use both adv_x and adv_y)
    const Eigen::Vector2f tb_dir = (dist > 1e-3f) ? tb / dist : Eigen::Vector2f{0.f, 1.f};

    // ---- K arcs: target-directed translation + spread rotation ----
    for (int i = 0; i < K; ++i)
    {
        const float frac = (K > 1)
            ? 2.f * static_cast<float>(i) / static_cast<float>(K - 1) - 1.f
            : 0.f;

        const float rot   = std::clamp(nom_rot + frac * params.max_rot_speed,
                                        -params.max_rot_speed, params.max_rot_speed);
        const float speed = std::max(0.f, nom_speed * (1.f - 0.4f * std::abs(frac)));

        // Translate toward target in body frame (omni motion)
        const auto arc_cmd = apply_speed_limit({tb_dir.x() * speed, tb_dir.y() * speed, rot});
        Policy p;
        p.commands.resize(params.horizon_steps, arc_cmd);
        policies.push_back(std::move(p));
    }

    // ---- Pure rotations (only kept for epistemic gain scoring, low weight now) ----
    for (float s : {-1.f, -0.5f, 0.5f, 1.f})
    {
        Policy p;
        p.commands.resize(params.horizon_steps, {0.f, 0.f, s * params.max_rot_speed});
        policies.push_back(std::move(p));
    }

    return policies;
}

// ===========================================================================
// Kinematic rollout
// ===========================================================================
void EpistemicController::rollout_policy(Policy& policy) const
{
    policy.predicted_states.clear();
    policy.predicted_states.reserve(params.horizon_steps + 1);

    Eigen::Vector3f s{epistemic_planner_.robot_pos().x(),
                      epistemic_planner_.robot_pos().y(),
                      epistemic_planner_.robot_theta()};
    policy.predicted_states.push_back(s);

    for (int t = 0; t < params.horizon_steps; ++t)
    {
        const auto& c = policy.commands[t];
        const float c_t = std::cos(s[2]);
        const float s_t = std::sin(s[2]);

        s[0] += (c.adv_x * c_t - c.adv_y * s_t) * params.dt;
        s[1] += (c.adv_x * s_t + c.adv_y * c_t) * params.dt;
        s[2] += c.rot * params.dt;
        s[2]  = std::atan2(std::sin(s[2]), std::cos(s[2]));

        policy.predicted_states.push_back(s);
    }
}

// ===========================================================================
// EFE evaluation
// ===========================================================================
void EpistemicController::evaluate_policy_efe(Policy& policy,
                                            const EpistemicPlanner::Target& target,
                                            const Eigen::Matrix3f& prior_precision) const
{
    float pragmatic = 0.f;
    float boundary  = 0.f;
    float obstacle  = 0.f;
    float epistemic = 0.f;

    const float collision_radius = std::max(params.obstacle_radius, robot_footprint_radius_);
    const float obs_r2 = collision_radius * collision_radius;
    const float footprint_r2 = robot_footprint_radius_ * robot_footprint_radius_;
    const auto& room_corners = epistemic_planner_.room_corners();
    const bool have_polygon = !room_corners.empty();

    bool boundary_fired = false;  // fire boundary penalty only once per arc
    for (std::size_t t = 1; t < policy.predicted_states.size(); ++t)
    {
        const auto& s = policy.predicted_states[t];
        const Eigen::Vector2f pos{s[0], s[1]};

        // Check room boundary first — once crossed, skip this and all further steps
        // for pragmatic/heading so near-target arcs that overshoot aren't unfairly penalised.
        bool out_of_bounds = false;
        if (have_polygon)
        {
            if (!corner_visibility::point_in_polygon(pos, room_corners))
                out_of_bounds = true;
            else if (robot_footprint_radius_ > 0.f && !edge_segments_.empty() &&
                     nearest_wall_distance_sq(pos) < footprint_r2)
                out_of_bounds = true;
        }
        else if (epistemic_planner_.room_bounds_set() &&
                 (pos.x() < epistemic_planner_.room_min().x() + robot_footprint_radius_ ||
                  pos.x() > epistemic_planner_.room_max().x() - robot_footprint_radius_ ||
                  pos.y() < epistemic_planner_.room_min().y() + robot_footprint_radius_ ||
                  pos.y() > epistemic_planner_.room_max().y() - robot_footprint_radius_))
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
            continue;  // skip pragmatic and obstacle accumulation beyond the wall
        }

        pragmatic += (pos - target.position).squaredNorm();

        const Eigen::Vector2f diff = target.position - pos;
        if (diff.squaredNorm() > 0.01f)
        {
            const float ct = std::cos(s[2]);
            const float st = std::sin(s[2]);
            const float bx =  ct * diff.x() + st * diff.y();
            const float by = -st * diff.x() + ct * diff.y();
            const float heading_err = std::atan2(-bx, by);
            pragmatic += params.w_heading * heading_err * heading_err;
        }

        float step_obs = 0.f;
        for (const auto& lp : lidar_obstacles_)
        {
            const float d2 = (pos - lp).squaredNorm();
            if (d2 < obs_r2 && d2 > 1e-6f)
                step_obs += params.obstacle_k / d2;
        }
        obstacle += std::min(step_obs, params.obstacle_step_cap);
    }

    // Epistemic: FIM-based information gain at the final predicted state
    if (have_polygon && policy.predicted_states.size() > 1)
    {
        const auto& sf = policy.predicted_states.back();
        const Eigen::Vector2f final_pos{sf[0], sf[1]};
        const float final_theta = sf[2];

        auto vis = corner_visibility::visible_corners(final_pos, room_corners,
                                                       params.fim_max_range);
        if (!vis.empty())
        {
            const auto fim = corner_visibility::corner_fim(final_pos, final_theta, vis,
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

// ===========================================================================
// plan — orchestrates Level 1 + Level 2
// ===========================================================================
std::optional<EpistemicController::PlanResult> EpistemicController::plan()
{
    auto target_opt = epistemic_planner_.update_target();
    if (!target_opt)
        return std::nullopt;

    const auto& target = *target_opt;

    // ---- Fallback: simple proportional controller ----
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

} // namespace rc
