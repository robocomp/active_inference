#include "mppi_tracker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace rc
{

namespace
{
// Same two helpers TrajectoryController keeps for its own parameter shaping. Duplicated rather than
// shared because this file deliberately does not include the controller: the only thing that needs
// them here is the goal-clearance ramp below.
float clamp01(float x)
{
    return std::clamp(x, 0.f, 1.f);
}

float smoothstep01(float x)
{
    const float t = clamp01(x);
    return t * t * (3.f - 2.f * t);
}
} // namespace

float MppiTracker::effective_d_safe_for_goal_dist(float goal_dist, const TrackerParams& p) const
{
    // No bearing is available here (this is the scalar the ramp relaxes TO), so the worst-case extent is the
    // honest floor: the goal must at least admit the body at whatever heading it arrives with.
    const float body_r = path_.body_extent_max();
    const float near_safe = std::max({body_r + p.goal_obstacle_margin,
                                      p.d_safe * p.goal_clearance_min_ratio,
                                      body_r + 0.005f});
    const float far_safe = std::max(p.d_safe, near_safe);
    const float tau = std::max(p.goal_clearance_relax_dist, 1e-3f);
    const float blend = smoothstep01(std::max(goal_dist, 0.f) / tau); // 0 near goal, 1 far
    return std::clamp(near_safe + blend * (far_safe - near_safe), near_safe, far_safe);
}

float MppiTracker::obstacle_step_cost(float esdf_val, float d_safe_eff, float body_r,
                                      const TrackerParams& p) const
{
    // body_r is the robot's REAL extent toward the nearest obstacle (footprint support function), not a disc
    // radius. Everything below is unchanged in shape — the only difference is that the body term is now
    // direction- and heading-dependent, so a rectangle is no longer forced to claim its diagonal in every
    // direction. d_safe stays what it always was: a PREFERRED standoff beyond the body, i.e. a comfort term.
    const float d_safe = std::max(d_safe_eff, body_r + 1e-3f);
    const float soft_span = std::max(d_safe - body_r, 1e-3f);
    const float hard_margin = std::max(p.close_obstacle_margin, 1e-3f);
    const float hard_threshold = body_r + hard_margin;

    float soft_penalty = 0.f;
    if (esdf_val < d_safe)
    {
        const float normalized = (d_safe - esdf_val) / soft_span;
        soft_penalty = normalized * normalized;
    }

    float hard_penalty = 0.f;
    if (esdf_val < hard_threshold)
    {
        const float normalized = (hard_threshold - esdf_val) / hard_margin;
        hard_penalty = p.close_obstacle_gain * normalized * normalized;
    }

    const float g_obs = p.lambda_obstacle * (soft_penalty + hard_penalty);
    return std::min(g_obs, p.obstacle_cost_cap);
}

float MppiTracker::obstacle_repulsion_strength(float esdf_val, float d_safe_eff, float body_r,
                                               const TrackerParams& p) const
{
    const float d_safe = std::max(d_safe_eff, body_r + 1e-3f);
    const float soft_span = std::max(d_safe - body_r, 1e-3f);
    const float hard_margin = std::max(p.close_obstacle_margin, 1e-3f);
    const float hard_threshold = body_r + hard_margin;

    float strength = 0.f;
    if (esdf_val < d_safe)
    {
        const float normalized = (d_safe - esdf_val) / soft_span;
        strength += p.lambda_obstacle * normalized;
    }

    if (esdf_val < hard_threshold)
    {
        const float normalized = (hard_threshold - esdf_val) / hard_margin;
        strength += p.lambda_obstacle * p.close_obstacle_gain * normalized;
    }

    return strength;
}

// ============================================================================
// Per-path and stop resets — the MPPI half of what TrajectoryController::reset_mppi_state and
// TrajectoryController::stop used to do inline.
// ============================================================================

void MppiTracker::reset_state(const TrackerParams& p)
{
    has_prev_vel_ = false;
    smoothed_vel_ = Eigen::Vector3f::Zero();

    // Initialize MPPI state
    adaptive_K_ = p.num_samples;
    adaptive_T_ = p.trajectory_steps;
    adaptive_lambda_ = p.mppi_lambda;
    ess_smooth_ = static_cast<float>(adaptive_K_) * p.ess_initial_ratio;
    dominance_smooth_ = 0.5f;
    explore_ = 0.f;
    last_mppi_ms_ = 0.f;
    safety_guard_mood_cooldown_ = 0;

    prev_optimal_.assign(adaptive_T_, {0.f, 0.f});
    last_cmd_valid_ = false;   // a new path has no command to be continuous with
    adaptive_sigma_adv_ = p.sigma_adv;
    adaptive_sigma_rot_ = p.sigma_rot;
}

void MppiTracker::clear_state()
{
    has_prev_vel_ = false;
    smoothed_vel_ = Eigen::Vector3f::Zero();
    prev_optimal_.clear();
    last_cmd_valid_ = false;
    safety_guard_mood_cooldown_ = 0;
}

// ============================================================================
// The sampler — everything that used to be steps 5..14 of TrajectoryController::compute.
// ============================================================================

ControlOutput& MppiTracker::compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p)
{
    // Names kept from the controller body this moved out of, so the arithmetic below reads unchanged.
    static const std::vector<Eigen::Vector3f> no_cloud;
    const std::vector<Eigen::Vector3f>& lidar_points = lidar_points_ ? *lidar_points_ : no_cloud;
    const Eigen::Affine2f& robot_pose = in.robot_pose;
    const Eigen::Vector2f& carrot_robot = in.carrot_robot;
    const Eigen::Vector2f& goal_robot = in.goal_robot;

    const auto t_mppi_start = std::chrono::steady_clock::now();

    const int T = adaptive_T_;

    // Safety-Guard proximity proxy from frontal LiDAR distance (continuous in [0,1]).
    // Exploration should ramp up only near SG activation distance.
    float sg_gate = 0.f;
    float frontal_min_dist = std::numeric_limits<float>::infinity();
    {
        constexpr float sg_front_cone_rad = 0.45f;
        const float sg_on_dist = p.d_safe + 0.08f;
        const float sg_center = sg_on_dist + p.sg_explore_pre_distance;
        const float sg_width = std::max(1e-3f, p.sg_explore_sigmoid_width);

        for (const auto &p : lidar_points)
        {
            const float px = p.x();
            const float py = p.y();
            if (py <= 0.f) continue;

            const float d = std::hypot(px, py);
            // Self-return reject. The bearing is right there in the point, so ask what the body actually
            // reaches at that bearing — a disc throws away the one piece of information we have.
            if (d < path_.body_extent({px, py}, 0.f) + 0.05f) continue;

            const float ang = std::abs(std::atan2(px, py));
            if (ang <= sg_front_cone_rad)
                frontal_min_dist = std::min(frontal_min_dist, d);
        }

        if (std::isfinite(frontal_min_dist))
        {
            const float x = (frontal_min_dist - sg_center) / sg_width;
            const float raw = 1.f / (1.f + std::exp(x));
            sg_explore_gate_smooth_ = 0.8f * sg_explore_gate_smooth_ + 0.2f * std::clamp(raw, 0.f, 1.f);
            sg_gate = std::clamp(sg_explore_gate_smooth_, 0.f, 1.f);
        }
        else
        {
            sg_explore_gate_smooth_ *= 0.8f;
            sg_gate = std::clamp(sg_explore_gate_smooth_, 0.f, 1.f);
        }
    }

    // 5-6. Tracking MPPI: compute nominal toward carrot and blend it with
    //       the shifted previous optimum to create the actual sampling center.
    //       This preserves carrot coverage while adding stronger temporal momentum.
    Seed nominal_seed = compute_nominal(carrot_robot, T, p);
    Seed sampling_center = nominal_seed;
    float ws_adv_eff = 0.f;
    float ws_rot_eff = 0.f;
    {
        const float ws_boost = 1.f + 0.75f * sg_gate + 0.50f * explore_;
        ws_adv_eff = std::clamp(p.warm_start_adv_weight * ws_boost, 0.f, 0.97f);
        ws_rot_eff = std::clamp(p.warm_start_rot_weight * ws_boost, 0.f, 0.97f);

        if (!prev_optimal_.empty())
        {
            const int prev_T = static_cast<int>(prev_optimal_.size());
            for (int t = 0; t < T; ++t)
            {
                // (was `p`; renamed because `p` is now the parameter block this tracker is called with)
                const int pi = std::min(t + 1, prev_T - 1);  // shifted warm start
                const float prev_adv = prev_optimal_[pi].adv;
                const float prev_rot = prev_optimal_[pi].rot;

                sampling_center.adv[t] = std::clamp(
                    (1.f - ws_adv_eff) * nominal_seed.adv[t] + ws_adv_eff * prev_adv,
                    p.min_adv_cmd, p.max_adv);
                sampling_center.rot[t] = std::clamp(
                    (1.f - ws_rot_eff) * nominal_seed.rot[t] + ws_rot_eff * prev_rot,
                    -p.max_rot, p.max_rot);
            }
        }
    }

    // 7. Sample K trajectories around the blended center
    auto seeds = sample_trajectories(carrot_robot, sampling_center, p);
    const int actual_K = static_cast<int>(seeds.size());

    // 8. Simulate, optimize, and score each sample
    std::vector<SimResult> results(actual_K);
    int best_idx = -1;
    float best_G = std::numeric_limits<float>::max();

    for (int k = 0; k < actual_K; ++k)
    {
        optimize_seed(seeds[k], carrot_robot, p);
        results[k] = simulate_and_score(seeds[k], carrot_robot, goal_robot, p);
        if (results[k].G_total < best_G)
        {
            best_G = results[k].G_total;
            best_idx = k;
        }
    }

    // 9. MPPI weighted average with fixed λ
    std::vector<ControlStep> optimal(T, {0.f, 0.f});
    std::vector<ControlStep> weighted_optimal(T, {0.f, 0.f});
    float ess_current = 0.f;
    float lambda_used = adaptive_lambda_;
    float cost_range_diag = 0.f;
    float dominance_current = 0.f;
    float p_free_current = 0.f;
    float steering_concentration_current = 0.f;
    float clearance_quality_current = 0.f;
    int num_collisions = 0;
    {
        const float G_min = best_G;

        // Count collisions, compute cost range over non-colliding seeds
        float g_max_nc = G_min;
        for (int k = 0; k < actual_K; ++k)
        {
            if (results[k].collides)
                num_collisions++;
            else
                g_max_nc = std::max(g_max_nc, results[k].G_total);
        }
        // Adaptive floor: λ must cover the cost range so weight ratios stay ~exp(-5)
        const float g_range = g_max_nc - G_min;
        cost_range_diag = g_range;
        const float range_lambda = std::max(1.f, g_range / 5.f);
        lambda_used = p.lambda_fixed ? p.mppi_lambda
                                                  : std::max(adaptive_lambda_, range_lambda);

        std::vector<float> weights(actual_K);
        float w_sum = 0.f;

        for (int k = 0; k < actual_K; ++k)
        {
            if (results[k].collides)
            {
                weights[k] = 0.f;
            }
            else
            {
                const float exponent = std::max(-60.f, -(results[k].G_total - G_min) / std::max(lambda_used, 1e-6f));
                weights[k] = std::exp(exponent);
            }
            w_sum += weights[k];
        }

        // Compute ESS (diagnostics)
        ess_current = compute_ess(weights, actual_K, p);
        ess_smooth_ = 0.75f * ess_smooth_ + 0.25f * ess_current;

        // Dominance D = free-survival mass * steering concentration
        // D≈0 -> no viable dominant bypass (explore more)
        // D≈1 -> clear viable direction dominates (exploit more)
        const int num_survivors = actual_K - num_collisions;
        const float p_free = static_cast<float>(std::max(0, num_survivors))
                           / static_cast<float>(std::max(actual_K, 1));
        float steering_concentration = 0.f;
        float clearance_quality = 0.f;
        if (num_survivors > 0)
        {
            float cos_acc = 0.f;
            float sin_acc = 0.f;
            float clear_acc = 0.f;
            float w_acc = 0.f;
            const bool use_soft_weights = (w_sum > p.weights_epsilon);
            // Normalisation only (a 0..1 clearance QUALITY score, not a safety test), and min_esdf carries no
            // bearing — so the worst-case extent is the right scale here.
            const float body_r_norm = path_.body_extent_max();
            const float clear_denom = std::max(p.d_safe - body_r_norm, 1e-3f);
            for (int k = 0; k < actual_K; ++k)
            {
                if (results[k].collides)
                    continue;

                const float wk = use_soft_weights
                    ? (weights[k] / w_sum)
                    : (1.f / static_cast<float>(num_survivors));

                const int steps_k = static_cast<int>(seeds[k].rot.size());
                const int heading_window = std::min(8, steps_k);
                float steering_angle = 0.f;
                for (int s = 0; s < heading_window; ++s)
                    steering_angle += seeds[k].rot[s] * p.trajectory_dt;

                const float clearance_norm = std::clamp(
                    (results[k].min_esdf - body_r_norm) / clear_denom,
                    0.f, 1.f);

                cos_acc += wk * std::cos(steering_angle);
                sin_acc += wk * std::sin(steering_angle);
                clear_acc += wk * clearance_norm;
                w_acc += wk;
            }
            if (w_acc > p.weights_epsilon)
            {
                steering_concentration = std::sqrt(cos_acc * cos_acc + sin_acc * sin_acc) / w_acc;
                clearance_quality = clear_acc / w_acc;
            }
        }
        p_free_current = p_free;
        steering_concentration_current = steering_concentration;
        clearance_quality_current = clearance_quality;

        // Dominance: feasibility × directional concentration (clearance excluded
        // to avoid over-exploration in narrow passages).
        dominance_current = std::clamp(p_free * steering_concentration, 0.f, 1.f);

        // Textbook MPPI weighted average: U_new = Σ w_k * V_k
        // Since all V_k = U + ε_k and Σ w_k = 1, this is equivalent to
        // U_new = U + Σ w_k * ε_k (perturbation-weighted update).
        if (w_sum > p.weights_epsilon)
        {
            for (int k = 0; k < actual_K; ++k)
            {
                const float w = weights[k] / w_sum;
                if (w > p.weights_epsilon)
                {
                    const int steps_k = static_cast<int>(seeds[k].adv.size());
                    for (int t = 0; t < std::min(T, steps_k); ++t)
                    {
                        weighted_optimal[t].adv += w * seeds[k].adv[t];
                        weighted_optimal[t].rot += w * seeds[k].rot[t];
                    }
                }
            }

            optimal = weighted_optimal;
        }
        else if (best_idx >= 0)
        {
            // ── EVERY ROLLOUT IS INFEASIBLE ────────────────────────────────────────────────────────
            // Rare (measured: ~0.1% of cycles, once or twice a lap) but it is precisely the moment the
            // robot is in trouble, so what gets chosen here matters more than in any other branch.
            //
            // It used to take `best_idx`, the lowest-cost seed overall — and that is BIASED TOWARD THE
            // EARLIEST COLLISION. A rollout stops accumulating obstacle cost at the step it collides
            // (the simulation `break`s) and every collider then receives the same flat collision_penalty,
            // so a rollout that hits at step 2 carries almost nothing while one that hits at step 20
            // carries nineteen steps of soft cost. The cheapest collider is therefore usually the one
            // that crashes soonest, and cost ranking cannot express what actually matters here.
            //
            // What matters when nothing is feasible is TIME. The latest collision is the best available
            // outcome: it maximises the distance the robot travels before contact, and — far more
            // importantly — it buys the most cycles for the next solve to find a way out, for the
            // obstacle to move, or for the speed limiter to bleed off speed. `positions.size()` is the
            // number of steps simulated before the break, i.e. exactly that time.
            //
            // ★This is a tie-break among BAD options, not a safety mechanism. It does not prevent
            // contact; it stops the controller from actively selecting the soonest one. The barrier that
            // should have kept us out of this state is a separate question (the hard test is only 2 cm
            // wide, and the terms covering everything outside it are capped) and is deliberately NOT
            // touched here.
            int chosen = best_idx;
            std::size_t latest = 0;
            float latest_cost = std::numeric_limits<float>::max();
            for (int k = 0; k < actual_K; ++k)
            {
                const std::size_t steps_before_impact = results[k].positions.size();
                if (steps_before_impact > latest
                    or (steps_before_impact == latest and results[k].G_total < latest_cost))
                {
                    latest = steps_before_impact;
                    latest_cost = results[k].G_total;
                    chosen = k;
                }
            }
            static int all_infeasible_logs = 0;
            if ((all_infeasible_logs++ % 20) == 0)
                std::printf("[mppi] ALL %d rollouts infeasible — driving the LATEST-impact one "
                            "(%zu of %d steps clear, cost %.1f).\n",
                            actual_K, latest, T, latest_cost);
            const int steps_b = static_cast<int>(seeds[chosen].adv.size());
            for (int t = 0; t < std::min(T, steps_b); ++t)
            {
                optimal[t].adv = seeds[chosen].adv[t];
                optimal[t].rot = seeds[chosen].rot[t];
            }
        }
    }

    // 10. Single-metric adaptation from rollout dominance gated by SG proximity
    adapt_from_dominance(dominance_current, actual_K, sg_gate, p);

    // Top-K decisive blending: when exploration is high, blend weighted mean
    // toward the weighted average of the top-3 feasible seeds rather than the
    // single best.  This eliminates frame-to-frame sign flips when two
    // competing seeds alternate as "best" near doorways.
    {
        constexpr int top_k_count = 3;
        const float decisive = std::clamp(explore_ * explore_, 0.f, 0.85f);
        if (decisive > 1e-4f && actual_K > 0)
        {
            // Collect indices of non-colliding seeds sorted by cost
            std::vector<int> free_indices;
            free_indices.reserve(actual_K);
            for (int k = 0; k < actual_K; ++k)
                if (!results[k].collides) free_indices.push_back(k);
            // Fall back to all seeds if none are free
            if (free_indices.empty())
                for (int k = 0; k < actual_K; ++k) free_indices.push_back(k);

            std::sort(free_indices.begin(), free_indices.end(),
                      [&](int a, int b) { return results[a].G_total < results[b].G_total; });
            const int n_top = std::min(top_k_count, static_cast<int>(free_indices.size()));

            // Softmax weights over top-K costs (temperature = lambda_used)
            float top_w_sum = 0.f;
            std::vector<float> top_w(n_top);
            const float G_top_min = results[free_indices[0]].G_total;
            for (int i = 0; i < n_top; ++i)
            {
                const float exp_arg = std::max(-30.f, -(results[free_indices[i]].G_total - G_top_min)
                                                       / std::max(lambda_used, 1e-6f));
                top_w[i] = std::exp(exp_arg);
                top_w_sum += top_w[i];
            }
            if (top_w_sum > 1e-10f)
            {
                for (int t = 0; t < T; ++t)
                {
                    float blend_adv = 0.f, blend_rot = 0.f;
                    for (int i = 0; i < n_top; ++i)
                    {
                        const int ki = free_indices[i];
                        const float w = top_w[i] / top_w_sum;
                        const int steps_k = static_cast<int>(seeds[ki].adv.size());
                        if (t < steps_k)
                        {
                            blend_adv += w * seeds[ki].adv[t];
                            blend_rot += w * seeds[ki].rot[t];
                        }
                    }
                    optimal[t].adv = (1.f - decisive) * optimal[t].adv + decisive * blend_adv;
                    optimal[t].rot = (1.f - decisive) * optimal[t].rot + decisive * blend_rot;
                }
            }
        }
    }

    // Export ESS diagnostics
    out.ess = ess_smooth_;
    out.lambda_used = lambda_used;
    out.lambda_adaptive = adaptive_lambda_;
    out.cost_range = cost_range_diag;
    if (best_idx >= 0 and best_idx < static_cast<int>(results.size()))
    {
        const SimResult &b = results[best_idx];
        out.cost_best = b.G_total;
        out.g_goal = b.G_goal;
        out.g_obs = b.G_obs;
        out.g_vel = b.G_vel;
        out.g_smooth = b.G_smooth;
        out.g_lat = b.G_lat;
        out.g_cbf = b.G_cbf;
    }
    out.n_collisions = num_collisions;
    out.best_seed_idx = best_idx;
    if (best_idx >= 0 and best_idx < static_cast<int>(seeds.size()) and not seeds[best_idx].rot.empty())
        out.best_seed_rot = seeds[best_idx].rot[0];
    out.p_free = p_free_current;
    out.steering_concentration = steering_concentration_current;
    {
        // Same probe geometry and the same normalisation as the balance sub-term inside the rollout cost,
        // evaluated at the robot itself (theta = 0 in its own frame) so this is the servo's input now.
        const float off = std::max(0.f, p.lateral_probe_offset);
        const float body_lat = std::max(path_.body_extent({+1.f, 0.f}, 0.f), path_.body_extent({-1.f, 0.f}, 0.f));
        const float target = body_lat + std::max(0.f, p.lateral_clearance_margin);
        const float span = std::max(p.lateral_clearance_margin, 1e-3f);
        float dl = std::numeric_limits<float>::infinity(), dr = dl;
        for (const float lon : {-p.lateral_probe_rear_offset, 0.f,
                                 p.lateral_probe_front_offset})
        {
            dr = std::min(dr, world_.esdf_at(+off, lon));
            dl = std::min(dl, world_.esdf_at(-off, lon));
        }
        const float ul = std::clamp((target - dl) / span, 0.f, 1.f);
        const float ur = std::clamp((target - dr) / span, 0.f, 1.f);
        out.side_asymmetry = ul - ur;
    }

    out.ess_K = adaptive_K_;
    out.explore = explore_;

    // Store optimal for smoothing reference (G_smooth term) — NOT used as
    // sampling center (that role belongs to the freshly-computed nominal).
    prev_optimal_ = optimal;

    // Wall-clock time for CPU budget tracking
    last_mppi_ms_ = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_mppi_start).count();

    float cmd_adv = 0.f, cmd_rot = 0.f;
    // 11. Extract command: average first 3 steps of the optimal sequence.
    // Step 0 has the highest perturbation variance; averaging [0..2] is a
    // standard MPPI technique that smooths the output without adding lag.
    {
        constexpr int cmd_window = 3;
        const int n_avg = std::min(cmd_window, T);
        float sum_adv = 0.f, sum_rot = 0.f;
        for (int t = 0; t < n_avg; ++t)
        {
            sum_adv += optimal[t].adv;
            sum_rot += optimal[t].rot;
        }
        cmd_adv = sum_adv / static_cast<float>(n_avg);
        cmd_rot = sum_rot / static_cast<float>(n_avg);
    }

    const float carrot_heading = std::atan2(carrot_robot.x(), carrot_robot.y());
    const float straight_heading_limit = std::max(p.straight_speed_heading_threshold * 2.5f, 0.12f);
    const float straight_rot_limit = std::max(0.30f * p.max_rot, 0.08f);
    const float straight_clearance_gate = p.d_safe
                                        + std::min(p.straight_speed_clearance_margin, 0.05f);
    const bool straight_clear_segment = std::abs(carrot_heading) < straight_heading_limit
                                     && out.dist_to_goal > p.straight_speed_min_goal_dist
                                     && out.min_esdf > straight_clearance_gate
                                     && num_collisions == 0
                                     && clearance_quality_current > 0.75f
                                     && std::abs(cmd_rot) < straight_rot_limit;
    if (straight_clear_segment)
        cmd_adv = std::max(cmd_adv, p.max_adv);

    // 12. Viz: export top 3 trajectories and the weighted-average homotopy.
    {
        const Eigen::Matrix2f R = robot_pose.linear();
        const Eigen::Vector2f t_pos = robot_pose.translation();

        std::vector<int> ranked_indices;
        ranked_indices.reserve(actual_K);
        for (int k = 0; k < actual_K; ++k)
            if (!results[k].collides)
                ranked_indices.push_back(k);
        if (ranked_indices.empty())
            for (int k = 0; k < actual_K; ++k)
                ranked_indices.push_back(k);

        std::sort(ranked_indices.begin(), ranked_indices.end(),
                  [&](int lhs, int rhs) { return results[lhs].G_total < results[rhs].G_total; });

        const int n_draw = std::min(3, static_cast<int>(ranked_indices.size()));
        out.trajectories_room.resize(n_draw);
        for (int i = 0; i < n_draw; ++i)
        {
            const int k = ranked_indices[i];
            auto& tr = out.trajectories_room[i];
            tr.reserve(results[k].positions.size() + 1);
            tr.push_back(t_pos);
            for (const auto& p : results[k].positions)
                tr.push_back(R * p + t_pos);
        }
        out.best_trajectory_idx = (n_draw > 0) ? 0 : -1;

        out.average_trajectory_room.clear();
        out.average_trajectory_room.reserve(optimal.size() + 1);
        out.average_trajectory_room.push_back(t_pos);
        float avg_x = 0.f;
        float avg_y = 0.f;
        float avg_theta = 0.f;
        const float dt = p.trajectory_dt;
        for (const auto &step : optimal)
        {
            avg_x += step.adv * std::sin(avg_theta) * dt;
            avg_y += step.adv * std::cos(avg_theta) * dt;
            avg_theta += step.rot * dt;
            out.average_trajectory_room.push_back(R * Eigen::Vector2f(avg_x, avg_y) + t_pos);
        }
    }

    // 13. Smooth + Gaussian brake
    float eff_smoothing = p.velocity_smoothing;
    if (straight_clear_segment)
        eff_smoothing = std::min(eff_smoothing, 0.15f);

    Eigen::Vector3f raw(cmd_adv, 0.f, cmd_rot);
    if (has_prev_vel_)
        smoothed_vel_ = eff_smoothing * smoothed_vel_ + (1.f - eff_smoothing) * raw;
    else { smoothed_vel_ = raw; has_prev_vel_ = true; }

    const float rot_ratio = smoothed_vel_[2] / p.max_rot;
    const float brake = std::exp(-p.gauss_k * rot_ratio * rot_ratio);

    out.adv  = smoothed_vel_[0] * brake;
    out.side = smoothed_vel_[1];
    out.rot  = smoothed_vel_[2];
    // 14. Safety gate on top of MPPI output (short forward prediction on inflated ESDF)
    {
        constexpr float gate_horizon_s = 0.30f;     // shorter look-ahead to reduce false positives
        constexpr float gate_inflate_m = 0.03f;       // soft margin around d_safe
        constexpr float gate_hard_margin_m = 0.01f;
        constexpr int gate_soft_consecutive_needed = 5; // consecutive close steps to trigger
        const float gate_dt = std::max(0.03f, p.trajectory_dt);

        if (safety_guard_mood_cooldown_ > 0)
            safety_guard_mood_cooldown_--;

        struct RiskEval
        {
            bool trigger = false;
            bool hard_collision = false;
            float min_esdf = std::numeric_limits<float>::infinity();
        };

        auto eval_risk = [&](float adv, float rot, float horizon_s)
        {
            RiskEval r;
            float x = 0.f, y = 0.f, theta = 0.f;
            const int steps = std::max(1, static_cast<int>(std::ceil(horizon_s / gate_dt)));
            // Thresholds are per-STEP, below: the body's reach depends on the heading the rollout has turned
            // to and on where the obstacle is, and this gate exists precisely to catch tight passages —
            // exactly where a heading-blind disc is most wrong.
            int soft_consecutive = 0;

            for (int i = 0; i < steps; ++i)
            {
                x += adv * std::sin(theta) * gate_dt;
                y += adv * std::cos(theta) * gate_dt;
                theta += rot * gate_dt;

                const float d = world_.esdf_at(x, y);
                r.min_esdf = std::min(r.min_esdf, d);

                const float body_r = world_.body_extent_toward_obstacle(x, y, theta);
                if (d < body_r + gate_hard_margin_m)
                {
                    r.trigger = true;
                    r.hard_collision = true;
                    return r;
                }

                if (d < body_r + gate_inflate_m)
                {
                    soft_consecutive++;
                    if (soft_consecutive >= gate_soft_consecutive_needed)
                    {
                        r.trigger = true;
                        return r;
                    }
                }
                else
                {
                    soft_consecutive = 0;
                }
            }
            return r;
        };

        // Simpler arming from direct LiDAR measurements (robot frame):
        // trigger only when an obstacle is close in a frontal cone.
        constexpr float lidar_front_cone_rad = 0.40f; // ~23 deg frontal cone
        const float lidar_trigger_dist = p.d_safe + 0.08f;
        constexpr float lidar_self_margin_m = 0.08f;   // beyond the body, a return is the world, not us
        constexpr int lidar_min_points_to_arm = 5;
        bool gate_armed = false;
        int frontal_close_count = 0;
        for (const auto &p : lidar_points)
        {
            const float px = p.x();
            const float py = p.y(); // forward axis
            if (py <= 0.f) continue; // front only

            const float d = std::hypot(px, py);
            // Near-body noise reject, at the bearing of the return itself.
            if (d < path_.body_extent({px, py}, 0.f) + lidar_self_margin_m) continue;

            const float ang = std::abs(std::atan2(px, py));
            if (ang <= lidar_front_cone_rad && d < lidar_trigger_dist)
            {
                frontal_close_count++;
                if (frontal_close_count >= lidar_min_points_to_arm)
                {
                    gate_armed = true;
                    break;
                }
            }
        }
        const auto nominal_risk = gate_armed ? eval_risk(out.adv, out.rot, gate_horizon_s) : RiskEval{};
        // Record what this gate SAW, so the gate_* columns are not silently PD-only. Only the two
        // observations are recorded: this gate's response is a ladder plus backup manoeuvres, not a
        // single speed scale, so gate_speed_scale/gate_hard_stop have no honest value here and keep
        // their defaults. ★The two gates have DIVERGED — this one still uses a fixed 0.30 s horizon and
        // a {0.6,0.35,0.15,0} ladder while the PD one derives its horizon from stopping distance and
        // bisects. Unifying them is a behaviour change and needs its own measurement, so it is left as
        // a follow-up rather than smuggled into a cleanup.
        if (gate_armed)
        {
            out.gate_horizon_s = gate_horizon_s;
            out.gate_min_esdf = nominal_risk.min_esdf;
            out.gate_hard_collision = nominal_risk.hard_collision;
        }

        if (nominal_risk.trigger)
        {
            out.safety_guard_triggered = true;
            const float base_rot = out.rot;
            bool accepted = false;

            // Back up while turning toward free space
            float preferred_sign = 1.f;
            const Eigen::Vector2f grad0 = world_.esdf_gradient_at(0.f, 0.f);
            if (grad0.norm() > p.esdf_grad_min_norm)
                preferred_sign = (grad0.x() >= 0.f) ? 1.f : -1.f;
            else if (std::abs(carrot_robot.x()) > 1e-3f)
                preferred_sign = (carrot_robot.x() >= 0.f) ? 1.f : -1.f;
            else if (std::abs(base_rot) > 1e-3f)
                preferred_sign = (base_rot >= 0.f) ? 1.f : -1.f;

            const float backup_adv = -std::max(0.05f, std::min(p.max_back_adv, 0.35f * p.max_adv));
            const float backup_rot = preferred_sign * std::max(0.12f, 0.35f * p.max_rot);

            if (!eval_risk(backup_adv, backup_rot, gate_horizon_s).trigger)
            {
                out.adv = backup_adv;
                out.rot = backup_rot;
                accepted = true;
            }
            else if (!eval_risk(backup_adv, 0.f, gate_horizon_s).trigger)
            {
                out.adv = backup_adv;
                out.rot = 0.f;
                accepted = true;
            }

            // Try reducing forward speed instead of full backup
            for (float scale : {0.6f, 0.35f, 0.15f, 0.f})
            {
                if (accepted) break;
                const float test_adv = out.adv * scale;
                if (!eval_risk(test_adv, base_rot, gate_horizon_s).trigger)
                {
                    out.adv = test_adv;
                    out.rot = base_rot;
                    accepted = true;
                    break;
                }
            }

            if (!accepted)
            {
                out.adv = 0.f;
                out.rot = preferred_sign * std::max(0.15f, 0.7f * p.max_rot);

                // Already touching something at the current pose (heading 0 in the robot frame) → back off.
                if (world_.esdf_at(0.f, 0.f) < world_.body_extent_toward_obstacle(0.f, 0.f, 0.f) + 0.01f)
                    out.adv = -std::min(p.max_back_adv, 0.05f);
            }
        }
    }

    // Remember what we are about to command: the continuity cost is referenced to this next cycle.
    last_cmd_adv_ = out.adv;
    last_cmd_rot_ = out.rot;
    last_cmd_valid_ = true;

    return out;
}

MppiTracker::Seed MppiTracker::compute_nominal(
    const Eigen::Vector2f& carrot_robot, int steps, const TrackerParams& p) const
{
    Seed seed;
    seed.adv.resize(steps);
    seed.rot.resize(steps);

    const float dt = p.trajectory_dt;
    // Y+ forward: atan2(x, y) gives angle from forward axis
    const float carrot_angle = std::atan2(carrot_robot.x(), carrot_robot.y());
    const float carrot_dist = carrot_robot.norm();
    float theta = 0.f;

    for (int s = 0; s < steps; ++s)
    {
        // Remaining angle to carrot from current simulated heading
        float angle_err = carrot_angle - theta;
        while (angle_err > static_cast<float>(M_PI)) angle_err -= 2.f * static_cast<float>(M_PI);
        while (angle_err < -static_cast<float>(M_PI)) angle_err += 2.f * static_cast<float>(M_PI);

        // PD-like proportional gain instead of bang-bang (angle_err/dt).
        // Kp=1.8 saturates at ~22° heading error (vs 16° at 2.5), reducing
        // unnecessary max-rot commands during small course corrections while
        // still reaching max_rot at doors/corners (>22° error).
        const float Kp_nom = 1.8f;
        float rot_cmd = std::clamp(Kp_nom * angle_err,
                                   -p.max_rot, p.max_rot);

        // Forward speed: proportional to alignment with carrot
        float alignment = std::cos(angle_err);
        float adv_cmd = p.max_adv * std::max(p.nominal_alignment_floor, alignment);
        // Reduce speed near goal
        float dist_factor = std::min(1.f, carrot_dist / std::max(p.nominal_goal_dist_scale, 1e-6f));
        adv_cmd *= dist_factor;
        adv_cmd = std::clamp(adv_cmd, p.min_adv_cmd, p.max_adv);

        seed.adv[s] = adv_cmd;
        seed.rot[s] = rot_cmd;

        theta += rot_cmd * dt;
    }
    return seed;
}

// ============================================================================
// Sample K trajectories — Tracking MPPI
//
// The sampling center is the freshly-computed nominal (straight-to-carrot),
// NOT the shifted previous solution.  This guarantees that:
//   1. The exploration region always contains the carrot direction
//   2. No warm-start death spiral (prev_optimal_ starting at zero)
//   3. The nominal adapts instantly to carrot/path changes (corners)
//
// Structure:
//   Seed 0         = nominal (zero perturbation)
//   Seeds 1..N_inj = structured exploration at wide angles (±30°, ±60°, ±90°)
//   Seeds N_inj+1..K-1 = nominal + i.i.d. Gaussian perturbations
//
// The exploration seeds use the same PD-like gain as the nominal (Kp=2.5)
// to avoid the bang-bang oscillation that angle_err/dt caused.
// In open space, rot_cost_factor=40 ensures they get near-zero weight.
// Near obstacles, they provide the wide-angle coverage needed to find detours.
// ============================================================================

std::vector<MppiTracker::Seed> MppiTracker::sample_trajectories(
    const Eigen::Vector2f& carrot_robot,
    const Seed& nominal,
    const TrackerParams& p)
{
    const int K = adaptive_K_;
    const int T = adaptive_T_;
    const int nom_T = static_cast<int>(nominal.adv.size());
    const float dt = p.trajectory_dt;
    const float carrot_angle = std::atan2(carrot_robot.x(), carrot_robot.y());
    const float carrot_dist = carrot_robot.norm();

    std::vector<Seed> seeds;
    seeds.reserve(K);

    // --- Seed 0: the nominal itself (zero perturbation) --------------------
    Seed nominal_copy = nominal;
    seeds.push_back(std::move(nominal_copy));

    // --- Structured exploration seeds at wide angles -----------------------
    // Always inject 6 seeds: ±30°, ±60°, ±90° offsets from carrot direction.
    // These provide coarse coverage for detour discovery around obstacles.
    // With rot_cost_factor=40 they are heavily penalized in open space
    // (near-zero weight), so they don't cause oscillation.
    if (p.enable_injection_seeds)
    {
    const std::array<float, 6> offsets = {
        p.inject_offset_30, -p.inject_offset_30,
        p.inject_offset_60, -p.inject_offset_60,
        p.inject_offset_90, -p.inject_offset_90
    };

    const float Kp_nom = 1.8f;  // same PD-like gain as compute_nominal

    for (float offset : offsets)
    {
        Seed s;
        s.adv.resize(T);
        s.rot.resize(T);
        float theta = 0.f;

        for (int t = 0; t < T; ++t)
        {
            // Target heading: gradually reduce offset over time (converge to carrot)
            float phase = static_cast<float>(t) / static_cast<float>(std::max(T - 1, 1));
            float target_heading = carrot_angle + offset * (1.f - phase);

            float angle_err = target_heading - theta;
            while (angle_err > static_cast<float>(M_PI)) angle_err -= 2.f * static_cast<float>(M_PI);
            while (angle_err < -static_cast<float>(M_PI)) angle_err += 2.f * static_cast<float>(M_PI);

            // PD-like proportional gain — NOT bang-bang (angle_err/dt)
            float rot_cmd = std::clamp(Kp_nom * angle_err,
                                       -p.max_rot, p.max_rot);

            // Reduced forward speed for exploration
            float alignment = std::cos(angle_err);
            float adv_cmd = p.max_adv * p.injection_adv_scale
                          * std::max(p.nominal_alignment_floor, alignment);
            float dist_factor = std::min(1.f, carrot_dist / std::max(p.nominal_goal_dist_scale, 1e-6f));
            adv_cmd *= dist_factor;
            adv_cmd = std::clamp(adv_cmd, p.min_adv_cmd, p.max_adv);

            s.adv[t] = adv_cmd;
            s.rot[t] = rot_cmd;
            theta += rot_cmd * dt;
        }
        seeds.push_back(std::move(s));
    }
    }

    // --- What a differential drive can do that forward-biased noise cannot propose ----------------
    // STOP, and PIVOT IN PLACE either way. Three deterministic seeds out of K.
    // These are not exploration, they are the actions that are available exactly when nothing else is.
    // Measured: at the tightest cycle of a lap the robot sat 3.7 mm from a wall, and ALL 100 rollouts were
    // infeasible — even after the feasibility predicate was corrected so that standing still is admissible
    // (esdf 0.3207 vs support 0.3170). The predicate was no longer the obstacle; the PROPOSAL was. Every
    // sample carries adv >= 0 drawn around a forward nominal, so at a 3.7 mm gap every one of them
    // overlaps within a step or two, and `cbf_max_decel` had no effect on the outcome at all — the
    // signature of a feasible action that is never offered rather than one that is rejected.
    // A stopped robot is feasible under the ICS predicate by construction, so with these seeds present the
    // feasible set cannot be empty unless the robot is ALREADY overlapping, which is a different fault.
    if (p.enable_stop_pivot_seeds)
    {
        const std::array<float, 3> pivot_rates = {0.f, p.max_rot, -p.max_rot};
        for (const float w : pivot_rates)
        {
            Seed s_stop;
            s_stop.adv.assign(T, 0.f);
            s_stop.rot.assign(T, w);
            seeds.push_back(std::move(s_stop));
        }
    }

    // --- Remaining seeds: nominal + i.i.d. Gaussian perturbations ----------
    const int n_random = std::max(0, K - static_cast<int>(seeds.size()));
    for (int k = 0; k < n_random; ++k)
    {
        Seed s;
        s.adv.resize(T);
        s.rot.resize(T);
        for (int t = 0; t < T; ++t)
        {
            const float base_adv = (t < nom_T) ? nominal.adv[t] : 0.f;
            const float base_rot = (t < nom_T) ? nominal.rot[t] : 0.f;

            const float eps_adv = normal_(rng_) * adaptive_sigma_adv_;
            const float eps_rot = normal_(rng_) * adaptive_sigma_rot_;

            s.adv[t] = std::clamp(base_adv + eps_adv,
                                  p.min_adv_cmd, p.max_adv);
            s.rot[t] = std::clamp(base_rot + eps_rot,
                                  -p.max_rot, p.max_rot);
        }
        seeds.push_back(std::move(s));
    }

    return seeds;
}

// ============================================================================
// Forward simulate a seed and compute EFE score
// ============================================================================

MppiTracker::SimResult MppiTracker::simulate_and_score(
    const Seed& seed,
    const Eigen::Vector2f& carrot_robot,
    const Eigen::Vector2f& goal_robot,
    const TrackerParams& p)
{
    SimResult res;
    const int steps = static_cast<int>(seed.adv.size());
    const float dt = p.trajectory_dt;
    res.positions.reserve(steps);

    float x = 0.f, y = 0.f, theta = 0.f;
    float G_obs_total = 0.f;
    float G_lat_total = 0.f;
    float G_cbf_total = 0.f;
    float G_progress = 0.f;
    const float discount = p.cost_discount;
    float discount_acc = 1.f;
    float prev_dist_to_carrot = carrot_robot.norm();
    [[maybe_unused]] float prev_side_min = std::numeric_limits<float>::infinity();
    // CBF state: h(x,v) = d_ESDF - r - v²/(2*a_max)
    float prev_h_cbf = std::numeric_limits<float>::infinity();
    int actual_steps = 0;

    // Fix 1: Compute the scoring horizon — stop accumulating goal/progress cost
    // once the trajectory would overshoot the carrot.  We still simulate all T
    // steps for obstacle scoring and positions, but the goal-related costs use
    // the *closest-approach* point as the effective endpoint.
    float best_dist_to_carrot = carrot_robot.norm();
    int best_step = 0;  // step index of closest approach to carrot

    for (int s = 0; s < steps; ++s)
    {
        // Differential-drive kinematics (Y+ = forward, X+ = right)
        x += seed.adv[s] * std::sin(theta) * dt;
        y += seed.adv[s] * std::cos(theta) * dt;
        theta += seed.rot[s] * dt;

        res.positions.emplace_back(x, y);
        actual_steps = s + 1;

        // Track closest approach to carrot
        float cur_dist = (Eigen::Vector2f(x, y) - carrot_robot).norm();
        if (cur_dist < best_dist_to_carrot)
        {
            best_dist_to_carrot = cur_dist;
            best_step = s;
        }

        // Per-step progress: only penalize moving away BEFORE closest approach
        if (s <= best_step)
        {
            float step_progress = prev_dist_to_carrot - cur_dist;
            G_progress += discount_acc * std::max(0.f, -step_progress);
        }
        prev_dist_to_carrot = cur_dist;

        float esdf_val = world_.esdf_at(x, y);
        res.min_esdf = std::min(res.min_esdf, esdf_val);

        const float dist_goal_step = (Eigen::Vector2f(x, y) - goal_robot).norm();
        const float d_safe_eff = effective_d_safe_for_goal_dist(dist_goal_step, p);
        // The body's actual reach toward the nearest obstacle at THIS pose. Computed once and reused by the
        // obstacle cost, the CBF barrier and the collision test, so all three agree by construction — they
        // used to share a constant disc, which is the only reason they agreed before.
        const float body_r = world_.body_extent_toward_obstacle(x, y, theta);
        const float G_obs = obstacle_step_cost(esdf_val, d_safe_eff, body_r, p);

        // Lateral-clearance shaping (continuous, pre-SG):
        // sample ESDF on both sides of the predicted body at front/center/rear
        // stations. This lets the rollout feel obstacles while surpassing them,
        // not only when they align with the body center.
        {
            const float probe_offset = std::max(0.f, p.lateral_probe_offset);
            const float ct = std::cos(theta);
            const float st = std::sin(theta);

            const Eigen::Vector2f forward(st, ct);
            const Eigen::Vector2f right(ct, -st);
            const std::array<float, 3> longitudinal_offsets = {
                -std::max(0.f, p.lateral_probe_rear_offset),
                0.f,
                std::max(0.f, p.lateral_probe_front_offset)
            };

            float d_right_min = std::numeric_limits<float>::infinity();
            float d_left_min = std::numeric_limits<float>::infinity();
            float front_right_min = std::numeric_limits<float>::infinity();
            float front_left_min = std::numeric_limits<float>::infinity();
            for (const float longitudinal_offset : longitudinal_offsets)
            {
                const Eigen::Vector2f station = Eigen::Vector2f(x, y) + longitudinal_offset * forward;
                const Eigen::Vector2f right_probe = station + probe_offset * right;
                const Eigen::Vector2f left_probe  = station - probe_offset * right;

                const float d_right = world_.esdf_at(right_probe.x(), right_probe.y());
                const float d_left = world_.esdf_at(left_probe.x(), left_probe.y());
                d_right_min = std::min(d_right_min, d_right);
                d_left_min = std::min(d_left_min, d_left);

                if (longitudinal_offset > 0.f)
                {
                    front_right_min = std::min(front_right_min, d_right);
                    front_left_min = std::min(front_left_min, d_left);
                }
            }

            const float side_min = std::min(d_left_min, d_right_min);

            // side_min is a LATERAL distance, so the body's lateral reach is the exact number to beat — not
            // its diagonal. The Shadow hull is 0.272 across the wheels against 0.325 corner-to-corner, so a
            // disc was charging this term 5 cm of clearance the robot does not occupy sideways.
            const float body_lat = std::max(path_.body_extent({+1.f, 0.f}, 0.f), path_.body_extent({-1.f, 0.f}, 0.f));
            const float side_target = body_lat
                                    + std::max(0.f, p.lateral_clearance_margin);
            const float side_span = std::max(p.lateral_clearance_margin, 1e-3f);

            float G_lat_step = 0.f;
            if (side_min < side_target)
            {
                const float deficit = (side_target - side_min) / side_span;
                G_lat_step += p.lambda_lateral_clearance * deficit * deficit;
            }

            const float bumper_target = body_lat
                                      + std::max(0.f, p.lateral_bumper_margin);
            const float bumper_span = std::max(p.lateral_bumper_margin, 1e-3f);
            if (side_min < bumper_target)
            {
                const float bumper_deficit = (bumper_target - side_min) / bumper_span;
                G_lat_step += p.lambda_lateral_bumper
                            * bumper_deficit * bumper_deficit * bumper_deficit;
            }

            const float front_corner_min = std::min(front_left_min, front_right_min);
            if (std::isfinite(front_corner_min) && front_corner_min < bumper_target)
            {
                const float front_deficit = (bumper_target - front_corner_min) / bumper_span;
                G_lat_step += p.lambda_lateral_bumper
                            * std::max(0.f, p.lateral_corner_bias_gain)
                            * front_deficit * front_deficit;
            }

            // LATERAL BALANCE, on the one-sided DEFICITS rather than on the raw distance difference.
            //
            // The previous form was |d_left - d_right| / side_span, clamped at 1.5. With one side open,
            // query_esdf returns the unknown-distance sentinel (~100 m), so the ratio was ~400 and the
            // clamp SATURATED: a constant, with zero derivative, identical across every rollout — so it
            // cancelled in the softmax and contributed nothing but cost. That is the single-wall case,
            // i.e. every corner, which is exactly where a recentring signal was wanted.
            //
            // Measuring each side's own deficit instead makes the sentinel handle itself: an open side
            // is simply not deficient, so it contributes 0 with no special case and no clamp to saturate.
            // Both quantities are bounded in [0,1] by construction, so the term stays differentiable
            // everywhere it is nonzero. In a corridor with both walls in range it is a genuine centring
            // term (zero when the deficits match); with a single wall it degenerates to that wall's own
            // deficit, reinforcing G_obs's push away from it rather than going silent. (It used to say
            // "the clearance term" — that term lived here and was deleted as a duplicate of G_obs.)
            const float u_left  = std::clamp((side_target - d_left_min) / side_span, 0.f, 1.f);
            const float u_right = std::clamp((side_target - d_right_min) / side_span, 0.f, 1.f);
            const float asymmetry = u_left - u_right;
            if (std::abs(asymmetry) > 1e-4f)
                G_lat_step += p.lambda_lateral_clearance
                            * std::max(0.f, p.lateral_balance_gain)
                            * asymmetry * asymmetry;

            // Lateral closing-gain term — disabled; replaced by CBF below.
            // if (std::isfinite(prev_side_min) && side_min < prev_side_min)
            // {
            //     const float closing = (prev_side_min - side_min) / side_span;
            //     G_lat_step += active_params_.lambda_lateral_clearance
            //                 * active_params_.lateral_closing_gain
            //                 * std::max(0.f, closing);
            // }

            G_lat_total += discount_acc * (p.bounded_costs
                                           ? std::min(G_lat_step, p.obstacle_cost_cap)
                                           : G_lat_step);
            prev_side_min = side_min;
        }

        // Control Barrier Function cost:
        // h(x,v) = d_ESDF - r_robot - v²/(2*a_max)
        // Penalise ḣ + α·h < 0  (barrier decaying faster than class-K allows)
        if (p.enable_cbf)
        {
            const float v        = seed.adv[s];
            const float a_max    = std::max(p.cbf_max_decel, 1e-3f);
            const float h_curr   = esdf_val - body_r
                                 - (v * v) / (2.f * a_max);
            if (std::isfinite(prev_h_cbf))
            {
                const float h_dot    = (h_curr - prev_h_cbf) / dt;
                const float cbf_cond = h_dot + p.cbf_alpha * h_curr;
                if (cbf_cond < 0.f)
                {
                    const float viol = cbf_cond * cbf_cond;
                    G_cbf_total += discount_acc
                                 * std::min(p.lambda_cbf * viol,
                                            p.cbf_cost_cap);
                }
            }
            prev_h_cbf = h_curr;
        }

        G_obs_total += discount_acc * G_obs;
        discount_acc *= discount;

        // Collision semantics with finite hard horizon:
        // - near-term collision => hard infeasible (weight zero)
        // - far collision      => soft penalty only (keeps rollout useful)
        //
        // ★AN ICS PREDICATE WAS TRIED HERE AND REVERTED (2026-08-02) — read this before trying again.
        // `esdf < support + v*reaction + v^2/(2*decel)` is the right IDEA and was verified offline to fix
        // what it was aimed at: at a reversal cycle p_free went 0.24 -> 0.93. Live, the robot barely moved.
        // The defect is that the ESDF is OMNIDIRECTIONAL — the distance to the nearest obstacle in ANY
        // direction — so inflating it by the stopping distance demands braking room from a wall the robot
        // is driving PARALLEL to. At 0.7 m/s it requires 0.63 m of clearance in every direction, and this
        // apartment's route runs at 0.4-0.55 m. Nothing at speed is feasible, so the robot crawls.
        // The braking requirement is DIRECTIONAL and must be applied along the heading — the ESDF along
        // the stopping segment ahead — not as an isotropic inflation of the current clearance.
        // ★And the offline verification was too weak to catch it: two snapshots (the tightest cycle and
        // one reversal), and I checked p_free without checking that commanded SPEED survived. A predicate
        // that empties the feasible set and one that empties only the fast half look identical in p_free
        // at a cycle that was already slow.
        if (esdf_val < body_r + p.close_obstacle_margin)
        {
            const float ttc_s = static_cast<float>(s + 1) * dt;
            if (ttc_s <= p.hard_collision_horizon_s)
            {
                res.collides = true;
                break;
            }
            else
            {
                G_obs_total += discount_acc * p.far_collision_penalty_scale * p.collision_penalty;
                break;
            }
        }
    }

    // G_goal: use the closest-approach point, not the far-future endpoint.
    // This prevents overshoot from dominating the cost when horizon >> carrot distance.
    const float initial_dist = carrot_robot.norm();
    const float progress = initial_dist - best_dist_to_carrot;
    float G_goal = p.lambda_goal
                 * (best_dist_to_carrot + p.lambda_progress * std::max(0.f, -progress));

    // Heading cost: penalize initial angular divergence from carrot direction.
    const float carrot_angle = std::atan2(carrot_robot.x(), carrot_robot.y()); // Y+ forward
    {
        // Weighted average of angular error over the first few steps (not just step 0)
        // to capture the seed's rotational tendency, not just a single-step noise sample.
        const int heading_window = std::min(5, actual_steps);
        float heading_err_acc = 0.f;
        float hw_theta = 0.f;
        for (int s = 0; s < heading_window; ++s)
        {
            hw_theta += seed.rot[s] * dt;
            float err = std::abs(carrot_angle - hw_theta);
            if (err > static_cast<float>(M_PI)) err = 2.f * static_cast<float>(M_PI) - err;
            heading_err_acc += err;
        }
        heading_err_acc /= std::max(1, heading_window);
        G_goal += p.lambda_heading * p.lambda_goal * heading_err_acc;
    }

    // G_smooth: continuity with warm-started baseline
    float dv = seed.adv[0] - (prev_optimal_.empty() ? 0.f : prev_optimal_[0].adv);
    float dr = seed.rot[0] - (prev_optimal_.empty() ? 0.f : prev_optimal_[0].rot);
    float G_smooth = p.lambda_smooth * (dv * dv + dr * dr);

    // G_continuity: distance from the command ACTUALLY EXECUTED last cycle (see Params).
    // Skipped on the first cycle after a (re)plan: there is no executed command to be continuous with,
    // and anchoring to a stale one would fight the new path exactly when it must be followed.
    float G_cont = 0.f;
    if (last_cmd_valid_ and p.lambda_continuity > 0.f)
    {
        const float inv_adv = 1.f / std::max(p.max_adv, 1e-3f);
        const float inv_rot = 1.f / std::max(p.max_rot, 1e-3f);
        const float cv = (seed.adv[0] - last_cmd_adv_) * inv_adv;
        const float cr = (seed.rot[0] - last_cmd_rot_) * inv_rot;
        G_cont = p.lambda_continuity
               * (cv * cv + std::max(0.f, p.continuity_rot_factor) * cr * cr);
    }

    // G_velocity: magnitude regularization + action change penalty
    float G_vel_mag = 0.f;
    float G_vel_delta = 0.f;
    for (int s = 0; s < actual_steps; ++s)
    {
        G_vel_mag += seed.adv[s] * seed.adv[s]
                   + p.rot_cost_factor * seed.rot[s] * seed.rot[s];
        if (s > 0)
        {
            float da = seed.adv[s] - seed.adv[s - 1];
            float dro = seed.rot[s] - seed.rot[s - 1];
            G_vel_delta += da * da + p.rot_cost_factor * dro * dro;
        }
    }
    G_vel_mag *= p.lambda_velocity;
    G_vel_delta *= p.lambda_delta_vel;

    // ── NO INFORMATION-THEORETIC CORRECTION (G_info) HERE — DELETED ON PURPOSE ───────────────
    // There used to be a Williams-et-al. importance-sampling term
    //     S_k += λ · Σ_t  u_tᵀ Σ⁻¹ ε_t        (applied to the Gaussian seeds only)
    // added into G_total. It is gone, and it must not come back. Three reasons, in order of
    // how badly they broke the controller:
    //
    // 1. SCALE — it drowned every real cost. 1/σ_adv² = 1/0.12² ≈ 69; on a straight the
    //    nominal pins adv at max_adv = 0.7 (compute_nominal), and × λ = 8 gives ≈ 389·ε per
    //    step. Over 50 i.i.d. steps the across-sample spread is ≈ ±1500 — against a TASK-cost
    //    spread of order 8. So `g_range` in the weighting block was essentially G_info alone,
    //    and the adaptive floor `lambda_used = max(adaptive_lambda_, g_range/5)` landed near
    //    300 instead of the configured 8. Goal, obstacle, CBF and lateral terms were all
    //    diluted ~40× inside the softmax: the robot was being steered by sampling noise.
    //
    // 2. IT WAS A MULTIPLICATIVE BRAKE ON THE COMMAND. Under the resulting exponential tilt,
    //    E[ε_t] = −(adaptive_lambda_ / lambda_used) · u_t, so the weighted mean this function
    //    feeds back is u · (1 − adaptive_lambda_/lambda_used) — a gain strictly below 1 on
    //    every commanded velocity. That gain lands in prev_optimal_ (the warm start) and is
    //    re-applied next cycle, so it COMPOUNDS.
    //
    // 3. IT EXCLUDED EXACTLY THE SEEDS THAT FIND BYPASSES. Only the ~93 Gaussian seeds carried
    //    it; the nominal and the 6 structured wide-angle detour seeds did not. That is a large
    //    constant cost offset between the two families, so the detour seeds — the entire
    //    mechanism for discovering a way around an obstacle — were effectively unweighted.
    //
    // ⚠ THE TRAP: do NOT "restore it properly" by making the two temperatures agree (the
    // weights used lambda_used, the term used adaptive_lambda_). That drives the gain in (2)
    // to exactly zero — i.e. it drives the commanded speed to zero. Williams' correction is
    // only valid when the base measure is ZERO control, so the STATE cost has to actively pay
    // for motion; here the state-cost spread is ~8 against a correction spread of ~1500 and it
    // simply cannot. Dropping the term makes the base measure the nominal / warm start, which
    // is what most practical MPPI implementations actually do.

    res.G_goal = G_goal + p.lambda_goal * G_progress;
    if (p.bounded_costs)
    {
        // Per-step averages, so a total is horizon-independent and comparable across terms — the
        // property that lets one small fixed temperature serve all of them.
        const float inv = 1.f / static_cast<float>(std::max(1, actual_steps));
        G_obs_total *= inv; G_lat_total *= inv; G_cbf_total *= inv;
    }
    res.G_obs = G_obs_total + (res.collides ? p.collision_penalty : 0.f);
    res.G_lat = G_lat_total;
    res.G_cbf = G_cbf_total;
    res.G_smooth = G_smooth + G_cont;
    res.G_vel = G_vel_mag + G_vel_delta;
    res.G_total = res.G_goal + res.G_obs + res.G_lat + res.G_cbf + res.G_smooth + res.G_vel;
    return res;
}

// ============================================================================
// Gradient optimization of a seed using ESDF gradient
// ============================================================================

void MppiTracker::optimize_seed(Seed& seed, const Eigen::Vector2f& carrot_robot,
                                const TrackerParams& p)
{
    const int steps = static_cast<int>(seed.adv.size());
    const float dt = p.trajectory_dt;
    const float lr = p.optim_lr;

    for (int iter = 0; iter < p.optim_iterations; ++iter)
    {
        std::vector<float> px(steps), py(steps), ptheta(steps);
        float x = 0.f, y = 0.f, theta = 0.f;

        for (int s = 0; s < steps; ++s)
        {
            x += seed.adv[s] * std::sin(theta) * dt;
            y += seed.adv[s] * std::cos(theta) * dt;
            theta += seed.rot[s] * dt;
            px[s] = x; py[s] = y; ptheta[s] = theta;
        }

        for (int s = 0; s < steps; ++s)
        {
            Eigen::Vector2f correction = Eigen::Vector2f::Zero();

            // Goal: pull towards carrot
            Eigen::Vector2f to_carrot = carrot_robot - Eigen::Vector2f(px[s], py[s]);
            float dist_c = to_carrot.norm();
            if (dist_c > p.heading_norm_epsilon)
                correction += p.lambda_goal * to_carrot.normalized()
                           * std::min(dist_c, p.optimize_goal_pull_dist_cap);

            // ESDF: push away from obstacles
            // Cap the obstacle correction magnitude so it cannot overpower the goal pull.
            // This prevents trajectories from being pushed backward at narrow passages.
            float esdf_val = world_.esdf_at(px[s], py[s]);
            const float d_safe_eff = effective_d_safe_for_goal_dist(dist_c, p);
            if (esdf_val < d_safe_eff)
            {
                Eigen::Vector2f grad = world_.esdf_gradient_at(px[s], py[s]);
                Eigen::Vector2f obs_correction =
                    obstacle_repulsion_strength(esdf_val, d_safe_eff,
                                                world_.body_extent_toward_obstacle(px[s], py[s], ptheta[s]),
                                                p) * grad;
                // Cap obstacle correction to at most 2x the goal correction magnitude
                float goal_mag = correction.norm();
                float obs_mag = obs_correction.norm();
                if (obs_mag > goal_mag * p.optimize_obstacle_cap_ratio
                    && goal_mag > p.optimize_goal_min_norm)
                    obs_correction *= (goal_mag * p.optimize_obstacle_cap_ratio) / obs_mag;
                correction += obs_correction;
            }

            // Jacobians (Y+ forward, X+ right)
            float ct = std::cos(ptheta[s]);
            float st = std::sin(ptheta[s]);
            Eigen::Vector2f d_pos_d_adv(st * dt, ct * dt);
            // Limit the remaining-time factor to avoid wild corrections on early steps
            float remaining = std::min(static_cast<float>(steps - s) * dt,
                                       p.optimize_remaining_cap_steps * dt);
            Eigen::Vector2f d_pos_d_rot(seed.adv[s] * ct * remaining * dt,
                                        -seed.adv[s] * st * remaining * dt);

            seed.adv[s] += lr * correction.dot(d_pos_d_adv);
            seed.rot[s] += lr * correction.dot(d_pos_d_rot);

            // Keep advance strictly positive (no backward motion)
            seed.adv[s] = std::clamp(seed.adv[s], p.min_adv_cmd, p.max_adv);
            seed.rot[s] = std::clamp(seed.rot[s], -p.max_rot, p.max_rot);
        }
    }
}

// ============================================================================
// ESS computation
// ============================================================================

float MppiTracker::compute_ess(const std::vector<float>& weights, int K,
                               const TrackerParams& p) const
{
    float w_sum = 0.f, w_sq_sum = 0.f;
    for (int k = 0; k < K; ++k)
    {
        w_sum += weights[k];
        w_sq_sum += weights[k] * weights[k];
    }
    if (w_sq_sum < p.ess_den_epsilon) return 1.f;  // degenerate: all zero
    return (w_sum * w_sum) / w_sq_sum;
}

// ============================================================================
// Dominance-based adaptation (single metric)
//
// Dominance D in [0,1] captures both:
//  - free-survival mass of rollouts
//  - directional concentration of surviving steering
// Explore signal is E = 1 - D.
// ============================================================================

void MppiTracker::adapt_from_dominance(float dominance, int /*K*/, float sg_gate,
                                       const TrackerParams& p)
{
    const float alpha = p.ess_smoothing;
    const float D = std::clamp(dominance, 0.f, 1.f);
    dominance_smooth_ = (1.f - alpha) * dominance_smooth_ + alpha * D;

    // Control adaptation should react quickly to feasibility collapse,
    // so it is driven mostly by instantaneous dominance, with a small
    // contribution of the smoothed state to avoid jitter.
    const float dominance_for_control = std::clamp(0.8f * D + 0.2f * dominance_smooth_, 0.f, 1.f);
    const float base_explore = std::clamp(1.f - dominance_for_control, 0.f, 1.f);
    const float sg = std::clamp(sg_gate, 0.f, 1.f);
    explore_ = std::clamp(base_explore * sg, 0.f, 1.f);

    // Single-law modulation: low dominance -> soften weighting (higher λ)
    const float lambda_target = p.mppi_lambda * (1.f + 2.f * explore_);
    adaptive_lambda_ = 0.8f * adaptive_lambda_ + 0.2f * lambda_target;
    adaptive_lambda_ = std::clamp(adaptive_lambda_, p.lambda_min, p.lambda_max);

    // Wider angular search and slightly reduced speed perturbation when exploring
    const float sigma_rot_target = p.sigma_rot
        + explore_ * (p.sigma_max_rot - p.sigma_rot);
    adaptive_sigma_rot_ = 0.8f * adaptive_sigma_rot_ + 0.2f * sigma_rot_target;
    adaptive_sigma_rot_ = std::clamp(adaptive_sigma_rot_, p.sigma_min_rot, p.sigma_max_rot);

    const float sigma_adv_target = p.sigma_adv * (1.f - 0.3f * explore_);
    adaptive_sigma_adv_ = 0.8f * adaptive_sigma_adv_ + 0.2f * sigma_adv_target;
    adaptive_sigma_adv_ = std::clamp(adaptive_sigma_adv_, p.sigma_min_adv, p.sigma_max_adv);

    // Expand horizon under low dominance to search around unexpected obstacles
    const float target_T = static_cast<float>(p.trajectory_steps)
                         + explore_ * static_cast<float>(p.T_max - p.trajectory_steps);
    adaptive_T_ = std::clamp(static_cast<int>(std::lround(0.8f * static_cast<float>(adaptive_T_) + 0.2f * target_T)),
                             p.T_min, p.T_max);

    // Keep sample count at configured value (clamped), avoiding extra heuristics
    adaptive_K_ = std::clamp(p.num_samples, p.K_min, p.K_max);
}

} // namespace rc
