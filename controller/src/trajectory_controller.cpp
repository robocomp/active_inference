#include "trajectory_controller.h"
#include "route_spline.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <chrono>
#include <print>

namespace rc
{

namespace
{
// Wrap an angle to (-pi, pi].
float wrap_pi(float a)
{
    while (a > static_cast<float>(M_PI)) a -= 2.f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.f * static_cast<float>(M_PI);
    return a;
}
} // namespace

float TrajectoryController::clamp01(float x)
{
    return std::clamp(x, 0.f, 1.f);
}

float TrajectoryController::smoothstep01(float x)
{
    const float t = clamp01(x);
    return t * t * (3.f - 2.f * t);
}

void TrajectoryController::refresh_active_params()
{
    active_params_ = params;
    const float safety_priority = std::max(1.f, params.safety_priority_scale);

    const float base_d_safe_priority = std::max(0.01f, params.d_safe * safety_priority);
    const float base_lambda_obstacle = std::max(1e-5f, params.lambda_obstacle);
    const float base_close_obstacle_gain = std::max(1e-5f, params.close_obstacle_gain);
    const float base_collision_penalty_priority = std::max(1.f, params.collision_penalty * safety_priority);

    active_params_.d_safe = base_d_safe_priority;
    active_params_.lambda_obstacle = base_lambda_obstacle;
    active_params_.close_obstacle_gain = base_close_obstacle_gain;
    active_params_.collision_penalty = base_collision_penalty_priority;

    if (params.enable_mood)
    {
        const float mood = clamp01(params.mood);
        const float s = smoothstep01(mood);
        const float n = 2.f * s - 1.f;  // -1 calm, +1 excited

        auto scale_with_gain = [n](float base, float gain, float min_value)
        {
            return std::max(min_value, base * (1.f + gain * n));
        };

        const float speed_gain = std::max(0.f, params.mood_speed_gain);
        const float reactivity_gain = std::max(0.f, params.mood_reactivity_gain);
        const float caution_gain = std::max(0.f, params.mood_caution_gain);

        // Speed family: kinematic limits and lookahead
        active_params_.max_adv = scale_with_gain(params.max_adv, speed_gain, 0.01f);
        active_params_.max_rot = scale_with_gain(params.max_rot, speed_gain, 0.01f);
        active_params_.carrot_lookahead = scale_with_gain(params.carrot_lookahead, speed_gain, 0.05f);
        if (params.mood_scales_goal_threshold)   // off by design — see the flag's comment
            active_params_.goal_threshold = scale_with_gain(params.goal_threshold, speed_gain * 0.5f, 0.01f);

        // Reactivity family: output smoothing, warm-start inertia, brake
        active_params_.velocity_smoothing = std::clamp(
            params.velocity_smoothing * (1.f - reactivity_gain * n), 0.f, 0.98f);
        active_params_.warm_start_adv_weight = std::clamp(
            params.warm_start_adv_weight * (1.f - reactivity_gain * n), 0.f, 0.98f);
        active_params_.warm_start_rot_weight = std::clamp(
            params.warm_start_rot_weight * (1.f - reactivity_gain * n), 0.f, 0.98f);
        active_params_.gauss_k = scale_with_gain(params.gauss_k, -reactivity_gain, 1e-5f);

        // Caution family: safety margins (calm side only)
        const float calm_factor = std::max(0.f, -n);  // 0 at neutral/excited, 1 at calm
        active_params_.d_safe = std::max(base_d_safe_priority,
                                         base_d_safe_priority * (1.f + caution_gain * calm_factor));
        active_params_.collision_penalty = std::max(base_collision_penalty_priority,
                                                    base_collision_penalty_priority * (1.f + 0.5f * caution_gain * calm_factor));
    }

    active_params_.K_min = std::max(1, std::min(active_params_.K_min, active_params_.K_max));
    active_params_.T_min = std::max(1, std::min(active_params_.T_min, active_params_.T_max));
    active_params_.num_samples = std::clamp(active_params_.num_samples, active_params_.K_min, active_params_.K_max);
    active_params_.trajectory_steps = std::clamp(active_params_.trajectory_steps, active_params_.T_min, active_params_.T_max);
    // d_safe is a standoff BEYOND the body, so it is meaningless below the body's own worst-case reach.
    active_params_.d_safe = std::max(active_params_.d_safe, body_extent_max() + 0.01f);
    // The curvature ceiling is applied AFTER mood and the priority scaling, so it bounds whatever those
    // produced rather than being overwritten by them. Everything downstream — the nominal seed, the
    // sampling clamp, the rollouts — reads max_adv, so the whole plan respects it, not just the output.
    if (speed_limit_.has_value())
        active_params_.max_adv = std::clamp(*speed_limit_, 0.05f, active_params_.max_adv);
    active_params_.max_back_adv = std::clamp(active_params_.max_back_adv, 0.f, active_params_.max_adv);
    active_params_.min_adv_cmd = std::clamp(active_params_.min_adv_cmd, 0.f, active_params_.max_adv);
    active_params_.mppi_lambda = std::clamp(active_params_.mppi_lambda, active_params_.lambda_min, active_params_.lambda_max);
}

float TrajectoryController::body_extent_here() const { return body_extent_toward_obstacle(0.f, 0.f, 0.f); }

float TrajectoryController::body_extent_toward_obstacle(float rx, float ry, float theta) const
{
    // The ESDF gradient points AWAY from the nearest obstacle, so -grad is the direction the body would have
    // to reach to touch it. For a convex footprint the reach in that direction is exactly the support
    // function, so `esdf < support` is an EXACT collision test rather than a disc approximation. Where the
    // gradient is degenerate (flat field, far from everything) support_radius falls back to the
    // circumscribed radius, i.e. the old worst-case disc — safe, and only in the region where it cannot matter.
    const Eigen::Vector2f g = query_esdf_gradient(rx, ry);
    return body_extent(-g, theta);
}

// ============================================================================
// Public API
// ============================================================================

// Everything a new path resets EXCEPT the geometry conditioning (relax_path / smooth_path_spline).
// Split out so set_path_presmoothed can get the resets WITHOUT paying for a relax+spline pass whose
// result it immediately overwrites — for a ~700-sample route against a 5 cm-sampled room boundary
// that discarded work was tens of millions of distance evaluations, on the main Qt thread, at the
// instant a route is installed.
// `wp_index_` is deliberately NOT set here: the two callers disagree about it (set_path starts at 1,
// set_path_presmoothed at 0) and each owns its own answer.
void TrajectoryController::reset_mppi_state(const std::vector<Eigen::Vector2f>& path_room)
{
    refresh_active_params();

    path_room_ = path_room;
    active_ = path_room.size() >= 2;

    carrot_seg_hint_ = 0;   // a fresh path starts at the robot; the caller re-seeds it if it does not
    has_last_carrot_ = false;   // a NEW route may legitimately put the carrot somewhere else entirely

    // Initialize MPPI state (warm start, adaptive K/T/λ/σ, ESS, output smoothing — all of it now
    // lives in the tracker that uses it).
    mppi_tracker_.reset_state(active_params_);
    carrot_curve_active_ = false;

    // Reset blockage detection
    blockage_streak_ = 0;
    blockage_cooldown_ = 0;

    // A fresh path means we are navigating again, not doing the final rotation.
    aligning_ = false;
}

void TrajectoryController::set_path(const std::vector<Eigen::Vector2f>& path_room)
{
    reset_arrival_watch();
    reset_mppi_state(path_room);
    wp_index_ = (path_room.size() > 1) ? 1 : 0;

    // Push waypoints away from walls/furniture toward center of free space
    relax_path(40);

    // Smooth path with Catmull-Rom spline interpolation for continuous curvature
    smooth_path_spline();
}

void TrajectoryController::set_path_presmoothed(const std::vector<Eigen::Vector2f>& path_room)
{
    reset_arrival_watch();
    // Resets only — the geometry is already smooth and already footprint-checked, so relax_path
    // and smooth_path_spline are not merely unnecessary here, they are harmful (see the header).
    // This used to call set_path() and then overwrite path_room_, which RAN both passes in full
    // and threw the result away.
    reset_mppi_state(path_room);
    wp_index_ = 0;
}

void TrajectoryController::update_path_geometry(const std::vector<Eigen::Vector2f>& path_room)
{
    // Deliberately NOT reset_mppi_state — see the header. Everything the sampler carries between cycles
    // survives; only the geometry it is measured against changes.
    if (path_room.size() < 2) return;
    path_room_ = path_room;
    active_ = true;
    // The carrot hint indexes path_room_. A deformation can change the sample COUNT slightly (the curve
    // is resampled at fixed arc length, so a shorter curve has fewer samples), so the index is clamped
    // rather than trusted. It is never reset: a forward-only carrot that restarts at 0 would search the
    // whole route and can bind to a distant branch on this self-crossing tour.
    carrot_seg_hint_ = std::min(carrot_seg_hint_, static_cast<int>(path_room_.size()) - 1);
    wp_index_ = std::min(wp_index_, static_cast<int>(path_room_.size()) - 1);
}

void TrajectoryController::set_goal_facing_yaw(std::optional<float> yaw_rad)
{
    goal_facing_yaw_ = yaw_rad;
}

void TrajectoryController::stop()
{
    reset_arrival_watch();
    active_ = false;
    path_room_.clear();
    wp_index_ = 0;
    mppi_tracker_.clear_state();
    // ★2026-08-05 — STOP MUST INVALIDATE THE ARC-LENGTH PROJECTION. PlainTracker carries s_hint_, a
    // monotone-forward projection, and nothing was resetting it. After a mission stop it still held the
    // arc length where the robot left off — near the END of the route — so on START the 2 m search
    // window looked at [75, 76] on a fresh 76 m route and the tracker steered at the route's END POINT
    // while the robot stood at the beginning: "it does not go to the mission, it just moves down to the
    // wall", with route_s pinned at 0.20 m.
    // reset() asks for a RE-ACQUIRE, not a rewind to zero — this same stop() is called mid-mission to
    // install a repaired curve, where the correct answer is the robot's current position, not s = 0.
    plain_tracker_.reset();
    pd_tracker_.reset();
    carrot_curve_active_ = false;
    blockage_streak_ = 0;
    blockage_cooldown_ = 0;
}

void TrajectoryController::set_static_obstacles(
    const std::vector<std::vector<Eigen::Vector2f>>& obstacles_room, float sample_spacing)
{
    static_obstacle_points_room_.clear();
    for (const auto& obs : obstacles_room)
    {
        if (obs.size() < 2) continue;
        const int n = static_cast<int>(obs.size());
        for (int i = 0; i < n; ++i)
        {
            const Eigen::Vector2f& a = obs[i];
            const Eigen::Vector2f& b = obs[(i + 1) % n];
            const float len = (b - a).norm();
            const int num_samples = std::max(1, static_cast<int>(len / sample_spacing));
            for (int s = 0; s <= num_samples; ++s)
            {
                const float t = static_cast<float>(s) / static_cast<float>(num_samples);
                static_obstacle_points_room_.push_back(a + t * (b - a));
            }
        }
    }
}

void TrajectoryController::set_room_boundary(const std::vector<Eigen::Vector2f>& polygon,
                                              float sample_spacing)
{
    room_boundary_points_room_.clear();
    const int n = static_cast<int>(polygon.size());
    if (n < 3) return;
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = polygon[i];
        const Eigen::Vector2f& b = polygon[(i + 1) % n];
        const float len = (b - a).norm();
        const int num_samples = std::max(1, static_cast<int>(len / sample_spacing));
        for (int s = 0; s <= num_samples; ++s)
        {
            const float t = static_cast<float>(s) / static_cast<float>(num_samples);
            room_boundary_points_room_.push_back(a + t * (b - a));
        }
    }
}

// ============================================================================
// Elastic-band path relaxation: push waypoints toward center of free space
// ============================================================================
void TrajectoryController::relax_path(int iterations)
{
    if (path_room_.size() < 2) return;

    // --- Interpolate: insert points so no segment exceeds ~1 m ---
    constexpr float max_seg = 1.0f;
    std::vector<Eigen::Vector2f> dense;
    dense.push_back(path_room_.front());
    for (size_t i = 1; i < path_room_.size(); ++i)
    {
        const Eigen::Vector2f& a = path_room_[i - 1];
        const Eigen::Vector2f& b = path_room_[i];
        const float len = (b - a).norm();
        const int n_seg = std::max(1, static_cast<int>(std::ceil(len / max_seg)));
        for (int s = 1; s <= n_seg; ++s)
        {
            const float t = static_cast<float>(s) / static_cast<float>(n_seg);
            dense.push_back(a + t * (b - a));
        }
    }
    path_room_ = std::move(dense);

    const int n = static_cast<int>(path_room_.size());
    if (n < 3) return;

    // Combine room boundary + furniture points for nearest-obstacle queries
    std::vector<Eigen::Vector2f> all_obs;
    all_obs.reserve(room_boundary_points_room_.size() + static_obstacle_points_room_.size());
    all_obs.insert(all_obs.end(), room_boundary_points_room_.begin(), room_boundary_points_room_.end());
    all_obs.insert(all_obs.end(), static_obstacle_points_room_.begin(), static_obstacle_points_room_.end());

    if (all_obs.empty()) return;

    constexpr float alpha_obs    = 0.35f;   // obstacle repulsion gain per iteration
    constexpr float alpha_smooth = 0.15f;   // smoothing gain per iteration
    const float d_thresh = 1.5f;            // meters — influence radius for push
    constexpr float smoothing_standoff = 0.05f;   // comfort beyond the body when nudging a waypoint

    // Nearest obstacle point, and the distance to it. The POINT matters, not just the distance: it gives the
    // bearing at which the body has to fit, and the body's reach is direction-dependent. Returning only a
    // scalar is what forced the old code to compare against a worst-case disc.
    auto nearest_obs = [&](const Eigen::Vector2f& p) -> std::pair<Eigen::Vector2f, float>
    {
        float best = std::numeric_limits<float>::max();
        Eigen::Vector2f at = p;
        for (const auto& obs_pt : all_obs)
        {
            const float dsq = (p - obs_pt).squaredNorm();
            if (dsq < best) { best = dsq; at = obs_pt; }
        }
        return {at, std::sqrt(best)};
    };
    // The robot arrives at a waypoint travelling along the path, and that is the heading the footprint will
    // present there — the same model the grid planner searches under (heading = direction of travel). So the
    // smoother can ask the exact question the planner already answered, instead of a disc approximation.
    auto heading_at = [&](int i) -> float
    {
        const Eigen::Vector2f t = path_room_[std::min(i + 1, n - 1)] - path_room_[std::max(i - 1, 0)];
        return t.squaredNorm() > 1e-12f ? std::atan2(t.x(), t.y()) : 0.f;
    };

    for (int iter = 0; iter < iterations; ++iter)
    {
        // Skip first and last waypoint (start & goal are fixed)
        for (int i = 1; i < n - 1; ++i)
        {
            const Eigen::Vector2f& p = path_room_[i];
            const float heading = heading_at(i);

            const auto [nearest, min_dist] = nearest_obs(p);

            // Obstacle repulsion: push away from nearest wall, proportional to proximity
            Eigen::Vector2f obs_force = Eigen::Vector2f::Zero();
            if (min_dist < d_thresh && min_dist > 0.01f)
            {
                const Eigen::Vector2f away = (p - nearest) / min_dist;  // unit vector away
                obs_force = away * (d_thresh - min_dist);
            }

            // Smoothing: pull toward midpoint of neighbors
            const Eigen::Vector2f midpoint = 0.5f * (path_room_[i - 1] + path_room_[i + 1]);
            const Eigen::Vector2f smooth_force = midpoint - p;

            // Candidate new position
            const Eigen::Vector2f candidate = p + alpha_obs * obs_force + alpha_smooth * smooth_force;

            // Safety check: reject the move if the BODY would not fit there, at the heading it will arrive
            // with, toward the obstacle that is actually nearest to the candidate.
            const auto [cand_nearest, cand_dist] = nearest_obs(candidate);
            const float need = body_extent(candidate - cand_nearest, heading) + smoothing_standoff;
            if (cand_dist >= need)
                path_room_[i] = candidate;
        }
    }

}

std::optional<Eigen::Vector2f> TrajectoryController::current_waypoint_room() const
{
    if (!active_ || wp_index_ >= static_cast<int>(path_room_.size()))
        return std::nullopt;
    return path_room_[wp_index_];
}

// ============================================================================
// Catmull-Rom spline path smoothing
// Replaces the piecewise-linear (relaxed) path with a smooth curve that
// preserves start/end points and passes through all original waypoints.
// The output spacing is uniform (~spline_spacing meters between points).
// ============================================================================

void TrajectoryController::smooth_path_spline()
{
    const int n = static_cast<int>(path_room_.size());
    if (n < 3) return;  // need at least 3 points for meaningful spline

    constexpr float spline_spacing = 0.15f;  // output resolution in meters
    constexpr float alpha = 0.5f;             // centripetal Catmull-Rom (0.5)

    // Evaluate one Catmull-Rom segment between P1 and P2
    // (P0, P1, P2, P3 are the four control points)
    auto catmull_rom = [alpha](const Eigen::Vector2f& P0, const Eigen::Vector2f& P1,
                               const Eigen::Vector2f& P2, const Eigen::Vector2f& P3,
                               float t) -> Eigen::Vector2f
    {
        // Knot parameterization
        auto knot = [alpha](float ti, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
        {
            float d = (b - a).squaredNorm();
            return ti + std::pow(std::max(d, 1e-8f), alpha * 0.5f);
        };
        float t0 = 0.f;
        float t1 = knot(t0, P0, P1);
        float t2 = knot(t1, P1, P2);
        float t3 = knot(t2, P2, P3);

        // Map t in [0,1] to [t1, t2]
        float u = t1 + t * (t2 - t1);

        auto lerp = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b, float ta, float tb, float tu)
        {
            float f = (tu - ta) / std::max(tb - ta, 1e-8f);
            return (1.f - f) * a + f * b;
        };

        Eigen::Vector2f A1 = lerp(P0, P1, t0, t1, u);
        Eigen::Vector2f A2 = lerp(P1, P2, t1, t2, u);
        Eigen::Vector2f A3 = lerp(P2, P3, t2, t3, u);
        Eigen::Vector2f B1 = lerp(A1, A2, t0, t2, u);
        Eigen::Vector2f B2 = lerp(A2, A3, t1, t3, u);
        return lerp(B1, B2, t1, t2, u);
    };

    std::vector<Eigen::Vector2f> smooth;
    smooth.push_back(path_room_.front());

    for (int i = 0; i < n - 1; ++i)
    {
        // Clamp control points at boundaries
        const Eigen::Vector2f& P0 = path_room_[std::max(0, i - 1)];
        const Eigen::Vector2f& P1 = path_room_[i];
        const Eigen::Vector2f& P2 = path_room_[std::min(n - 1, i + 1)];
        const Eigen::Vector2f& P3 = path_room_[std::min(n - 1, i + 2)];

        float seg_len = (P2 - P1).norm();
        int n_sub = std::max(1, static_cast<int>(std::ceil(seg_len / spline_spacing)));

        for (int s = 1; s <= n_sub; ++s)
        {
            float t = static_cast<float>(s) / static_cast<float>(n_sub);
            smooth.push_back(catmull_rom(P0, P1, P2, P3, t));
        }
    }

    // Safety: verify spline points stay inside free space (min clearance from obstacles)
    if (!room_boundary_points_room_.empty() || !static_obstacle_points_room_.empty())
    {
        constexpr float spline_standoff = 0.03f;   // comfort beyond the body
        std::vector<Eigen::Vector2f> all_obs;
        all_obs.insert(all_obs.end(), room_boundary_points_room_.begin(), room_boundary_points_room_.end());
        all_obs.insert(all_obs.end(), static_obstacle_points_room_.begin(), static_obstacle_points_room_.end());

        const int m = static_cast<int>(smooth.size());
        for (int i = 0; i < m; ++i)
        {
            auto& pt = smooth[i];
            float min_dist_sq = std::numeric_limits<float>::max();
            Eigen::Vector2f nearest = pt;
            for (const auto& obs_pt : all_obs)
            {
                float dsq = (pt - obs_pt).squaredNorm();
                if (dsq < min_dist_sq) { min_dist_sq = dsq; nearest = obs_pt; }
            }
            const float dist = std::sqrt(min_dist_sq);
            if (dist < 1e-4f) continue;

            // Heading = spline tangent: the robot follows this curve, so that is the pose the body presents
            // here, and the reach toward THIS obstacle at THAT heading is the exact clearance required.
            const Eigen::Vector2f tangent = smooth[std::min(i + 1, m - 1)] - smooth[std::max(i - 1, 0)];
            const float heading = tangent.squaredNorm() > 1e-12f ? std::atan2(tangent.x(), tangent.y()) : 0.f;
            const float min_clearance = body_extent(pt - nearest, heading) + spline_standoff;
            if (dist < min_clearance)
                pt = nearest + (pt - nearest) / dist * min_clearance;   // push out along the same bearing
        }
    }

    path_room_ = std::move(smooth);
}

// ============================================================================
// Main compute — Proper MPPI with warm-start + Gaussian perturbations
// ============================================================================

// The live entry point: read the cloud, then solve. Split from the solver below so that a cycle is a
// FUNCTION OF ITS INPUTS rather than of a buffer the solver reaches into — which is what lets
// tools/mppi_bench replay a recorded cycle through exactly this code instead of an imitation of it.
TrajectoryController::ControlOutput TrajectoryController::compute(const Eigen::Affine2f& robot_pose)
{
    return compute(robot_pose, read_lidar_points_robot(robot_pose));
}

TrajectoryController::ControlOutput TrajectoryController::compute(
    const Eigen::Affine2f& robot_pose, const std::vector<Eigen::Vector3f>& lidar_points)
{
    refresh_active_params();

    ControlOutput out;
    if (!active_ || path_room_.empty()) { active_ = false; out.goal_reached = true; return out; }

    // ESDF-input diagnostics: count the raw points that survived the self-filter and the nearest one.
    // If something is in the cloud yet nearest_esdf_point_m stays large, it was dropped before the ESDF.
    out.n_esdf_points = static_cast<int>(lidar_points.size());
    out.esdf_model_dropped = esdf_model_points_dropped_;
    {
        float nearest = std::numeric_limits<float>::infinity();
        for (const auto &p : lidar_points)
            nearest = std::min(nearest, std::hypot(p.x(), p.y()));
        out.nearest_esdf_point_m = std::isfinite(nearest) ? nearest : -1.f;
    }

    // 1. ESDF
    build_esdf(lidar_points, robot_pose);

    // 2. Advance waypoints
    advance_waypoints(robot_pose);

    // 3. Goal check.
    // goal_robot is the END OF THE PATH and drives the SPEED shaping (see set_arrival_point).
    // arrival_robot is where arrival is judged. They are the same thing unless the caller has extended
    // the path past the point it actually wants to reach.
    const Eigen::Vector2f goal_robot = room_to_robot(path_room_.back(), robot_pose);
    const Eigen::Vector2f arrival_robot = arrival_point_room_.has_value()
                                        ? room_to_robot(*arrival_point_room_, robot_pose)
                                        : goal_robot;
    out.dist_to_goal = arrival_robot.norm();

    // PASSED the arrival point? Once the path continues past it the robot cuts the corner by design and
    // may never enter the arrival radius, so proximity alone would leave the waypoint forever unreached.
    bool passed_arrival = false;
    if (arrival_point_room_.has_value() and arrival_outgoing_room_.has_value())
    {
        const Eigen::Vector2f dir = *arrival_outgoing_room_;
        if (dir.squaredNorm() > 1e-9f)
            passed_arrival = (robot_pose.translation() - *arrival_point_room_).dot(dir.normalized()) > 0.f;
    }
    // ── PASSED IT WITHOUT EVER BEING INSIDE THE BAND ────────────────────────────────────────────
    // The proximity test samples dist_to_goal ONCE PER CYCLE, so it can only fire if some cycle lands
    // inside the band. It usually does — 0.7 m/s over a 50 ms cycle is 35 mm against a 250 mm band — but
    // the loop's tail is long (the align note below records cycles reaching ~1.5 s), and at 0.7 m/s that
    // is over a metre of travel in ONE cycle. The robot then steps clean over the band, never samples
    // inside it, and nothing fires: it drives on with the goal behind it. Rare, and rare in the worst
    // way, because it depends on a timing hiccup coinciding with the last metre.
    // Being AT the goal can be missed; PASSING it cannot — the distance starts growing and never stops.
    // So: remember the closest approach, and if the robot is receding from a goal it genuinely got near,
    // treat that as arrival. `passed_arrival` above does not cover this: it needs an arrival_point AND an
    // outgoing direction, which only a waypoint on a longer path has.
    // ★THE CAPTURE RADIUS IS MEASURED, NOT GUESSED. What could have been jumped is exactly the largest
    // per-cycle change in distance seen on this approach, so the reach is the band plus that. A tighter
    // radius would miss precisely the slow-cycle case this exists for; a fixed larger one would fire on
    // approaches that never came close.
    // Three consecutive receding cycles, not one: pose jitter alone can grow the distance for a cycle,
    // and a false arrival stops the robot short of its goal.
    if (endpoint_arrival_)
    {
        const float step = std::abs(out.dist_to_goal - last_dist_to_goal_);
        if (std::isfinite(last_dist_to_goal_)) max_goal_step_ = std::max(max_goal_step_, step);
        if (out.dist_to_goal < closest_to_goal_) { closest_to_goal_ = out.dist_to_goal; receding_cycles_ = 0; }
        else if (out.dist_to_goal > closest_to_goal_ + 0.01f) ++receding_cycles_;
        last_dist_to_goal_ = out.dist_to_goal;
    }
    const bool passed_by_recession =
        endpoint_arrival_ and receding_cycles_ >= 3
        and closest_to_goal_ <= active_params_.goal_threshold + max_goal_step_;

    // The whole arrival test is skipped when the caller owns arrival itself (a continuous route ends by
    // ARC LENGTH — see set_endpoint_arrival). Skipping it, rather than ignoring its result, matters: this
    // branch also clears active_, so a caller that merely discarded goal_reached would find the follower
    // switched off and the base stopped for good.
    if (endpoint_arrival_
        and (out.dist_to_goal < active_params_.goal_threshold or passed_arrival or aligning_
             or passed_by_recession))
    {
        // Position reached. If the target carries a commanded facing yaw, rotate in
        // place to face it before finishing (so e.g. the robot looks AT the table at
        // an epistemic affordance). active_ stays true while aligning, so the session
        // keeps re-entering here and does not replan.
        if (goal_facing_yaw_.has_value())
        {
            // Body frame is +X right, +Y forward, so the forward axis points along
            // (theta + pi/2) in the room; to face goal_facing_yaw_ the body angle
            // theta must equal facing - pi/2.
            const float robot_theta = std::atan2(robot_pose.linear()(1, 0), robot_pose.linear()(0, 0));
            const float theta_des = wrap_pi(*goal_facing_yaw_ - static_cast<float>(M_PI_2));
            const float yaw_err = wrap_pi(theta_des - robot_theta);
            out.goal_yaw_err_rad = yaw_err;
            if (std::abs(yaw_err) <= active_params_.align_yaw_tol_rad)
            {
                aligning_ = false;
                active_ = false;
                out.goal_reached = true;
                reset_arrival_watch();
                mppi_tracker_.forget_executed_command();   // arrived: nothing to be continuous with
                return out;
            }
            aligning_ = true;
            out.aligning = true;
            out.adv = 0.f;
            out.side = 0.f;
            // The session sends rot_rps = -out.rot; in the OmniRobot convention a
            // positive sent rot turns the base CCW (theta increases). Drive theta
            // toward theta_des, with a floor to overcome stiction.
            float sent_rot = std::clamp(active_params_.align_kp * yaw_err,
                                        -active_params_.max_rot, active_params_.max_rot);
            // STICTION FLOOR — but only while it cannot overshoot. Applying it unconditionally is what made
            // the final rotation HUNT instead of settling: the floor is 0.10 rad/s and the tolerance is
            // 0.06 rad, so one cycle at the floor sweeps 0.10·dt. At the healthy 105 ms cycle that is 0.010 rad
            // (fine), but the loop's measured tail runs to ~1.5 s, and 0.10·1.5 = 0.15 rad — more than TWICE
            // the tolerance band. The robot therefore drives straight through the band, flips sign, and does it
            // again: a limit cycle that never terminates, which is exactly "turns near the target and can't
            // find the final position".
            // So the floor is only applied while the remaining error is big enough to absorb a worst-case
            // cycle at that rate. Inside that, the proportional term is used as-is and simply decays. This
            // removes the overshoot without weakening the stiction help where it actually matters.
            const float overshoot_guard_rad = active_params_.align_min_rot * align_worst_cycle_s_;
            if (std::abs(sent_rot) < active_params_.align_min_rot
                and std::abs(yaw_err) > overshoot_guard_rad)
                sent_rot = std::copysign(active_params_.align_min_rot, sent_rot);
            // Never command more rotation than the error itself, over a worst-case cycle: a deadbeat bound
            // that makes overshoot impossible regardless of how long this cycle ends up taking.
            const float no_overshoot = std::abs(yaw_err) / std::max(0.05f, align_worst_cycle_s_);
            sent_rot = std::clamp(sent_rot, -no_overshoot, no_overshoot);
            out.rot = -sent_rot;
            out.goal_reached = false;
            // The alignment path bypasses the MPPI but still commands the base, so it must leave the
            // continuity reference pointing at what was actually sent. Otherwise the first MPPI cycle
            // after alignment would be scored for continuity against a command from before the turn.
            mppi_tracker_.note_executed_command(out.adv, out.rot);
            return out;
        }
        active_ = false;
        out.goal_reached = true;
        return out;
    }

    out.min_esdf = query_esdf(0.f, 0.f);

    // 4. Carrot (fixed lookahead), CLIPPED TO WHAT THE ROBOT CAN CUT TO.
    // The carrot is a straight-line attractor: the nominal seed steers at it and every rollout is a
    // perturbation of that. So a carrot placed around a corner does not ask the robot to follow the
    // route — it asks it to drive INTO the inside wall of the corner and rely on the obstacle cost to
    // claw it back. The MPPI is not flexible enough to recover that, and the harder the corner the more
    // the attractor and the obstacle term fight, which is exactly where the stutter lives.
    // The rule: advance the carrot only as far as the ROUTE FRAGMENT LEFT BEHIND still admits the body.
    // Tested against the LIVE ESDF (so it sees dynamic obstacles the static grid does not) with the
    // footprint's real reach at the chord heading — the same predicate the obstacle cost uses.
    const Eigen::Vector2f carrot_room_raw = compute_carrot(robot_pose);
    Eigen::Vector2f carrot_robot = clip_carrot_to_reachable(room_to_robot(carrot_room_raw, robot_pose));

    // ── RATE-LIMIT THE CARROT, AFTER THE CLIP ────────────────────────────────────────────────────
    // ★It must be AFTER. A first version limited carrot_room BEFORE clip_carrot_to_reachable and did
    // nothing measurable: the per-cycle step stayed at mean 0.147 m / max 1.794 m against a 0.21 m cap,
    // because the jump is INTRODUCED by the clip, not by the projection. The clip re-tests the route
    // fragment against the LIVE ESDF every cycle and yanks the carrot in — measured, it fires on 75% of
    // cycles and collapses the carrot under 0.5 m on 15% — so when that test flickers the target moves
    // 1.7 m in one cycle. Limiting upstream of the thing that jumps limits nothing.
    //
    // ASYMMETRIC ON PURPOSE. Pulling the carrot IN is the safety direction (the clip has just decided
    // the route fragment beyond here is not admissible), so that is allowed to happen immediately and
    // in full. Only the RELEASE is rate-limited. Chatter needs both directions — pull in, let out, pull
    // in — so damping the release alone kills the oscillation without ever delaying a safety pull-in.
    {
        const Eigen::Vector2f carrot_room_now = robot_pose * carrot_robot;
        if (active_params_.carrot_rate_limit_factor > 0.f and has_last_carrot_)
        {
            const float step_max = std::max(0.02f, active_params_.carrot_rate_limit_factor
                                                 * active_params_.max_adv * active_params_.trajectory_dt);
            const Eigen::Vector2f d = carrot_room_now - last_carrot_room_;
            const float n = d.norm();
            // Closer to the robot than last cycle's carrot ⇒ a pull-in ⇒ never delayed.
            const bool pulling_in = carrot_robot.norm()
                                  < (room_to_robot(last_carrot_room_, robot_pose)).norm();
            if (n > step_max and not pulling_in)
            {
                const Eigen::Vector2f limited = last_carrot_room_ + d * (step_max / n);
                carrot_robot = room_to_robot(limited, robot_pose);
            }
        }
        last_carrot_room_ = robot_pose * carrot_robot;
        has_last_carrot_ = true;
    }
    out.carrot_room = robot_pose * carrot_robot;   // report the CLIPPED carrot: the viewer must show
                                                   // where the robot is actually being pulled
    out.current_wp_index = wp_index_;
    // Recorded for BOTH modes, before either control law runs, so the steering target can be compared
    // against what each mode did with it.
    out.carrot_bearing_rad = std::atan2(carrot_robot.x(), carrot_robot.y());
    out.carrot_dist_m = carrot_robot.norm();

    // ---- PLAIN: the route-following tracker. It does no avoidance, so blockage detection (which is
    // what triggers a replan) MUST run first — it is the only thing that notices the route has become
    // undrivable, and in this mode nothing else is looking. ----
    // PLAIN always drives in PLAIN mode — including click targets, which now carry a fitted
    // plan_spline_. PlainTracker returns a zero command when handed no curve at all, so the "is there a
    // curve" test belongs in ONE place (the session, per cycle) rather than being duplicated here.
    // ★PLAIN WITH NO CURVE FALLS BACK TO PD, LOUDLY. PlainTracker steers at a curve and returns zero
    // without one, so making the curve a precondition turned a failed spline fit into a robot that sits
    // still with a valid plan and says nothing — the worst possible failure mode, and one this dispatch
    // introduced. smooth_plan's own contract is that smoothing is "an improvement, never a
    // precondition"; PD honours that by following the polyline directly.
    if (control_mode_ == ControlMode::PLAIN and route_spline_ != nullptr and route_spline_->valid())
    {
        // Say what it COST, once the curve is back. Without this the fallback is unmeasurable: the
        // notice below was a one-shot latch for the whole process, so "PD drove one cycle" and "PD drove
        // the entire session" printed exactly the same single line, and nothing downstream recorded
        // which tracker was actually steering.
        if (plain_no_curve_cycles_ > 0)
        {
            std::println("[controller] PLAIN curve restored after {} cycles ({:.1f} s) on the PD "
                         "fallback.", plain_no_curve_cycles_, plain_no_curve_cycles_ * 0.05f);
            plain_no_curve_cycles_ = 0;
            plain_no_curve_logged_ = false;
        }
        detect_path_blockage(out, robot_pose);
        return plain_tracker_.compute(out, make_tracker_input(robot_pose, carrot_robot, goal_robot),
                                     active_params_);
    }
    // ---- PD carrot-follower mode: simple proportional-derivative controller ----
    if (control_mode_ == ControlMode::PLAIN)
    {
        // COUNTED, not latched. Rate-limited to one line per 100 cycles (5 s at 20 Hz) so a persistent
        // fallback keeps saying so — a fault that announces itself once and then goes quiet reads as a
        // fault that stopped.
        if (not plain_no_curve_logged_ or plain_no_curve_cycles_ % 100 == 0)
        {
            plain_no_curve_logged_ = true;
            std::println("[controller] PLAIN has no fitted curve for this path — following it with the PD "
                         "tracker instead ({} cycles so far). The robot still drives; the spline fit is "
                         "what failed.", plain_no_curve_cycles_ + 1);
        }
        ++plain_no_curve_cycles_;
    }
    if (control_mode_ == ControlMode::PD or control_mode_ == ControlMode::PLAIN)
    {
        // Blockage detection BEFORE the early return, or the planner can never be triggered in this
        // mode — it is the ONLY thing that notices the route has become undrivable once the sampler is
        // not running. ★It is stateful (blockage_streak_/cooldown_) and must run exactly ONCE per
        // cycle: that holds only because this branch returns, so the step-15 call below is unreachable
        // from here. A third control mode, or removing this early return, breaks that silently.
        detect_path_blockage(out, robot_pose);
        // The tracker writes through its ControlOutput& and returns the same object; no copy needed.
        // last_cmd_* is deliberately NOT set here: it feeds the MPPI continuity cost only, which never
        // runs in this mode, and writing it would imply a coupling that does not exist.
        return pd_tracker_.compute(out, make_tracker_input(robot_pose, carrot_robot, goal_robot),
                                  active_params_);
    }

    // ---- MPPI: the sampler in trackers/mppi_tracker.h. It queries the field itself, so unlike the
    // two branches above it does NOT need blockage detection to run before it — step 15 below still
    // runs on this path. ----
    // The cloud is lent for the duration of the call: the Safety-Guard ramp and the gate arming count
    // frontal returns rather than reading the ESDF, and that input has no place in TrackerInput (see
    // the header). Cleared straight afterwards so it can never be read stale.
    mppi_tracker_.set_cycle_cloud(&lidar_points);
    mppi_tracker_.compute(out, make_tracker_input(robot_pose, carrot_robot, goal_robot), active_params_);
    mppi_tracker_.set_cycle_cloud(nullptr);

    if (not snapshot_path_.empty())
    {
        write_snapshot(robot_pose, lidar_points);
        snapshot_path_.clear();      // one request, one snapshot
    }

    // ── 15. Path blockage detection ──────────────────────────────────
    // Runs for BOTH control modes — see detect_path_blockage, called before the PD branch returns.
    detect_path_blockage(out, robot_pose);

    return out;
}

// ============================================================================
// PD carrot-follower: simple proportional-derivative controller
//
// rot = Kp * angle_error + Kd * d(angle_error)/dt
// adv = max_adv * cos(angle_error) * dist_factor
//
// Uses the same ESDF, carrot, safety gate, gaussian brake and smoothing
// as the MPPI mode but replaces all the sampling/weighting logic.
// ============================================================================

// Look ahead along the planned path in room frame and query the ESDF at each sample. If several
// consecutive samples are blocked for several consecutive cycles, tell the caller to replan.
//
// ★MUST RUN IN BOTH CONTROL MODES. This used to be step 15 of compute(), which the PD branch returns
// long before reaching — so under ControlMode="pd" `path_blocked` was permanently false and NOTHING
// could trigger the planner. That is the one thing the elastic band structurally cannot do for itself:
// a gradient band cannot escape an obstacle sitting on the route, because the distance field's gradient
// there is axial (Zhou 2020 Fig. 2, and RouteSpline::self_test reproduces it). The band owns geometry,
// A* owns homotopy, and this is the handover between them — with the tracker driving there is no
// sampler left to notice the route is unusable.
//
// Depends only on path_room_, the pose and the ESDF, so it is valid anywhere after the field is built.
void TrajectoryController::detect_path_blockage(ControlOutput &out, const Eigen::Affine2f &robot_pose)
{
    out.path_blocked = false;
    if (blockage_cooldown_ > 0)
    {
        --blockage_cooldown_;
    }
    else
    {
        const float look_m = active_params_.blockage_lookahead_m;
        const float thr = active_params_.blockage_esdf_threshold;
        const int   min_wp = active_params_.blockage_min_waypoints;

        int   consec_blocked = 0;
        Eigen::Vector2f block_sum = Eigen::Vector2f::Zero();
        int   block_count = 0;

        float accum_dist = 0.f;
        for (int i = wp_index_; i < static_cast<int>(path_room_.size()); ++i)
        {
            if (i > wp_index_)
                accum_dist += (path_room_[i] - path_room_[i - 1]).norm();
            if (accum_dist > look_m) break;

            const Eigen::Vector2f p_rob = room_to_robot(path_room_[i], robot_pose);
            const float esdf_val = query_esdf(p_rob.x(), p_rob.y());

            if (esdf_val < thr)
            {
                ++consec_blocked;
                block_sum += path_room_[i];
                ++block_count;
            }
            else
            {
                // reset streak only if we had fewer than required
                if (consec_blocked < min_wp)
                {
                    consec_blocked = 0;
                    block_sum = Eigen::Vector2f::Zero();
                    block_count = 0;
                }
            }
        }

        auto fill_blockage_region = [&]()
        {
            out.blockage_detected_ahead = true;
            out.blockage_center_room = block_sum / static_cast<float>(block_count);

            float max_r = 0.f;
            for (int i = wp_index_; i < static_cast<int>(path_room_.size()); ++i)
            {
                const Eigen::Vector2f p_rob = room_to_robot(path_room_[i], robot_pose);
                if (query_esdf(p_rob.x(), p_rob.y()) < thr)
                    max_r = std::max(max_r, (path_room_[i] - out.blockage_center_room).norm());
            }
            // A region radius reported to the caller, not a safety test: worst-case extent so the region
            // covers the body at any heading.
            out.blockage_radius = max_r + body_extent_max();
        };

        if (consec_blocked >= min_wp)
        {
            fill_blockage_region();
            ++blockage_streak_;
            if (blockage_streak_ >= active_params_.blockage_confirm_cycles)
            {
                out.path_blocked = true;
                blockage_streak_ = 0;
                blockage_cooldown_ = active_params_.blockage_cooldown_cycles;
            }
        }
        else
        {
            blockage_streak_ = std::max(0, blockage_streak_ - 1); // decay slowly
        }
    }


}




// ============================================================================
// Carrot computation
// ============================================================================

bool TrajectoryController::chord_admits_body(const Eigen::Vector2f& carrot_robot) const
{
    const float len = carrot_robot.norm();
    if (len < 1e-3f) return true;
    // Heading of travel along the chord, in this controller's convention (x right, y forward, measured
    // from +y toward +x). The body presents THAT heading while cutting to the carrot.
    const float heading = std::atan2(carrot_robot.x(), carrot_robot.y());
    const int steps = std::max(2, static_cast<int>(std::ceil(len / 0.10f)));
    for (int i = 1; i <= steps; ++i)
    {
        const Eigen::Vector2f p = carrot_robot * (static_cast<float>(i) / steps);
        // The body's reach toward whatever is nearest AT that point, not a disc — the same predicate the
        // obstacle cost and the safety gate use, so the three cannot disagree about what fits.
        if (query_esdf(p.x(), p.y()) < body_extent(-query_esdf_gradient(p.x(), p.y()), heading))
            return false;
    }
    return true;
}

Eigen::Vector2f TrajectoryController::clip_carrot_to_reachable(const Eigen::Vector2f& carrot_robot) const
{
    if (chord_admits_body(carrot_robot)) return carrot_robot;
    // Walk back along the chord to the furthest admissible point. Never below a short minimum: a carrot
    // at the robot's own position is not a direction to go, and the follower needs SOMETHING to aim at
    // even when it is already tight — the MPPI's obstacle terms are what keep it safe there, not this.
    constexpr float kMinCarrot = 0.30f;
    const float len = carrot_robot.norm();
    for (float f = 0.9f; f > 0.05f; f -= 0.1f)
    {
        if (len * f < kMinCarrot) break;
        if (const Eigen::Vector2f c = carrot_robot * f; chord_admits_body(c)) return c;
    }
    return carrot_robot * std::min(1.f, kMinCarrot / std::max(len, 1e-3f));
}

Eigen::Vector2f TrajectoryController::compute_carrot(const Eigen::Affine2f& robot_pose)
{
    const Eigen::Vector2f robot_pos = robot_pose.translation();
    float min_dist_sq = std::numeric_limits<float>::max();
    int closest_seg = carrot_seg_hint_;
    float closest_t = 0.f;

    // ── WINDOWED, NOT GLOBAL ──────────────────────────────────────────────────────────────────────
    // This used to scan EVERY segment of path_room_ and take the spatial nearest. In continuous mode
    // path_room_ is the whole multi-lap route, and a tour crosses itself: the apartment tour's outbound
    // and inbound legs run 0.6 m apart. A global search binds the carrot to whichever branch is nearer
    // IN SPACE, which at the middle of a spur is the leg going the OTHER WAY — so the robot turns
    // around and drives back down the way it came. Reported live as "it turns around where the way out
    // and the way in cross or are very close", and measured as cross_track 1.10 m rms / 4.03 m max on a
    // run that then declared itself complete at waypoint 6.
    // RouteSpline::project and RouteFollower::advance both already refuse to do this, and say why.
    //
    // The window must be SHORT relative to the arc-length separation between nearby branches, not merely
    // "small": on that spur the two legs are 0.6 m apart in space but only ~3 m apart in arc length, so a
    // 3 m window would still reach the wrong one. It only has to cover how far the robot can travel in a
    // cycle plus tracking drift — 0.7 m/s over a 100 ms cycle is 7 cm — so 1.5 m forward is already
    // generous by a factor of twenty, and a little backward slack absorbs the robot being pushed back.
    constexpr float kWindowFwdM = 1.5f;
    constexpr float kWindowBackM = 0.75f;
    const int n_seg = static_cast<int>(path_room_.size()) - 1;
    if (n_seg <= 0) return path_room_.empty() ? Eigen::Vector2f::Zero()
                                              : room_to_robot(path_room_.back(), robot_pose);
    carrot_seg_hint_ = std::clamp(carrot_seg_hint_, 0, n_seg - 1);

    int lo = carrot_seg_hint_;
    for (float back = 0.f; lo > 0 and back < kWindowBackM; --lo)
        back += (path_room_[lo] - path_room_[lo - 1]).norm();
    int hi = carrot_seg_hint_;
    for (float fwd = 0.f; hi + 1 < n_seg and fwd < kWindowFwdM; ++hi)
        fwd += (path_room_[hi + 1] - path_room_[hi]).norm();

    for (int i = lo; i <= hi; ++i)
    {
        const Eigen::Vector2f a = path_room_[i];
        const Eigen::Vector2f ab = path_room_[i + 1] - a;
        const float ab_sq = ab.squaredNorm();
        float t = 0.f;
        if (ab_sq > 1e-6f)
            t = std::clamp((robot_pos - a).dot(ab) / ab_sq, 0.f, 1.f);
        const Eigen::Vector2f proj = a + t * ab;
        const float d_sq = (robot_pos - proj).squaredNorm();
        if (d_sq < min_dist_sq) { min_dist_sq = d_sq; closest_seg = i; closest_t = t; }
    }
    // Carried to the next cycle. A robot that has fallen far off the route keeps the windowed anchor and
    // is steered back toward the stretch it was on — which is what we want. Snapping to whatever branch
    // is nearest in space is what produced the turn-around.
    carrot_seg_hint_ = closest_seg;
    // The PROJECTION itself, not just the anchor index. compute_carrot already knows exactly where the
    // robot is on the route and threw it away; the PD tracker needs it, because steering at the carrot
    // alone converges to the carrot's DIRECTION rather than to the path, and cuts every corner.
    {
        const Eigen::Vector2f a = path_room_[closest_seg];
        last_path_proj_room_ = a + closest_t * (path_room_[closest_seg + 1] - a);
        last_path_proj_valid_ = true;
    }

    float lookahead_eff = active_params_.carrot_lookahead;
    if (active_params_.carrot_curve_adaptation_enabled)
    {
        const int n = static_cast<int>(path_room_.size());
        if (n >= 3)
        {
            const int v = std::clamp(closest_seg + 1, 1, n - 2);
            const Eigen::Vector2f seg_prev = path_room_[v] - path_room_[v - 1];
            const Eigen::Vector2f seg_next = path_room_[v + 1] - path_room_[v];
            const float len_prev = seg_prev.norm();
            const float len_next = seg_next.norm();

            if (len_prev > active_params_.segment_length_epsilon && len_next > active_params_.segment_length_epsilon)
            {
                const Eigen::Vector2f t_prev = seg_prev / len_prev;
                const Eigen::Vector2f t_next = seg_next / len_next;
                float dpsi = std::acos(std::clamp(t_prev.dot(t_next), -1.f, 1.f));

                const float enter_th = std::max(active_params_.carrot_curve_min_heading_change, 1e-3f);
                const float release_th = std::clamp(active_params_.carrot_curve_release_heading_change,
                                                    0.f,
                                                    enter_th);

                if (!carrot_curve_active_)
                {
                    if (dpsi >= enter_th)
                        carrot_curve_active_ = true;
                }
                else
                {
                    if (dpsi <= release_th)
                        carrot_curve_active_ = false;
                }

                if (carrot_curve_active_)
                    lookahead_eff = std::clamp(active_params_.carrot_curve_lookahead_min,
                                               0.05f,
                                               active_params_.carrot_lookahead);
            }
        }
        else
        {
            carrot_curve_active_ = false;
        }
    }
    else
    {
        carrot_curve_active_ = false;
    }

    // Walk forward from the projection, and stop as soon as the chord would cut more of the route away
    // than carrot_max_route_cut_m. The deviation is measured against the fragment ACTUALLY SPANNED, so a
    // straight run is unaffected (deviation stays ~0 and the full lookahead is used) while a tight curve
    // pulls the carrot in until the chord tracks it.
    const Eigen::Vector2f ab0 = path_room_[closest_seg + 1] - path_room_[closest_seg];
    const Eigen::Vector2f proj = path_room_[closest_seg] + closest_t * ab0;
    const float max_cut = std::max(0.01f, active_params_.carrot_max_route_cut_m);

    // Perpendicular distance of `p` from the chord proj->carrot.
    const auto deviation = [&proj](const Eigen::Vector2f &carrot, const Eigen::Vector2f &p)
    {
        const Eigen::Vector2f ch = carrot - proj;
        const float len2 = ch.squaredNorm();
        if (len2 < 1e-9f) return (p - proj).norm();
        const float t = std::clamp((p - proj).dot(ch) / len2, 0.f, 1.f);
        return (p - (proj + t * ch)).norm();
    };

    std::vector<Eigen::Vector2f> spanned;      // route points between the projection and the carrot
    spanned.reserve(64);
    Eigen::Vector2f best = proj;
    float travelled = 0.f;
    Eigen::Vector2f prev = proj;

    for (int i = closest_seg + 1; i < static_cast<int>(path_room_.size()); ++i)
    {
        const Eigen::Vector2f p = path_room_[i];
        travelled += (p - prev).norm();
        prev = p;

        float worst = 0.f;
        for (const auto &q : spanned) worst = std::max(worst, deviation(p, q));
        if (worst > max_cut)
            break;                              // extending further would skip part of the route

        best = p;
        spanned.push_back(p);
        if (travelled >= lookahead_eff) break;  // full lookahead reached with the route still tracked
    }

    // A carrot at the projection is not a direction to go. Fall back to the next sample ahead: the
    // clip-to-reachable pass and the obstacle terms are what keep it safe in tight places, not this.
    if ((best - proj).norm() < 1e-3f and closest_seg + 1 < static_cast<int>(path_room_.size()))
        best = path_room_[closest_seg + 1];
    return best;
}

// ============================================================================
// Waypoint advancement
// ============================================================================

void TrajectoryController::advance_waypoints(const Eigen::Affine2f& robot_pose)
{
    const Eigen::Vector2f robot_pos = robot_pose.translation();
    const int last = static_cast<int>(path_room_.size()) - 1;
    while (wp_index_ < last)
    {
        const float dist = (robot_pos - path_room_[wp_index_]).norm();
        if (dist < active_params_.carrot_lookahead * active_params_.waypoint_advance_lookahead_factor) { wp_index_++; continue; }
        if (wp_index_ > 0)
        {
            const Eigen::Vector2f seg = path_room_[wp_index_] - path_room_[wp_index_ - 1];
            const float seg_len = seg.norm();
            if (seg_len > active_params_.segment_length_epsilon)
            {
                const float proj = (robot_pos - path_room_[wp_index_ - 1]).dot(seg / seg_len);
                if (proj > seg_len) { wp_index_++; continue; }
            }
        }
        break;
    }
}

// ============================================================================
// ESDF
// ============================================================================

// ── CYCLE SNAPSHOT ───────────────────────────────────────────────────────────────────────────────
// Text, not binary: a snapshot is something a person greps while working out why a cycle went the way it
// did, and a few thousand points cost nothing. Only what compute() actually consumes is written — the
// ESDF is NOT among it, because the replay rebuilds it with this same build_esdf from these same inputs.
void TrajectoryController::write_snapshot(const Eigen::Affine2f &robot_pose,
                                          const std::vector<Eigen::Vector3f> &lidar_points) const
{
    std::ofstream f(snapshot_path_, std::ios::out | std::ios::trunc);
    if (not f.is_open()) return;
    f << std::setprecision(9);
    f << "# MPPI cycle snapshot — replay with tools/mppi_bench\n";
    f << "version 1\n";
    const Eigen::Rotation2Df rot(robot_pose.linear());
    f << "pose " << robot_pose.translation().x() << ' ' << robot_pose.translation().y() << ' '
      << rot.angle() << '\n';
    // The knobs a replay must reproduce to be the same cycle. Anything absent falls back to the
    // constructor default, which is the same thing the live agent would have used.
    f << "params " << active_params_.max_adv << ' ' << active_params_.max_rot << ' '
      << active_params_.d_safe << ' ' << active_params_.mppi_lambda << ' '
      << active_params_.num_samples << ' ' << active_params_.trajectory_steps << ' '
      << active_params_.trajectory_dt << ' ' << active_params_.lambda_obstacle << ' '
      << active_params_.lambda_goal << ' ' << active_params_.lambda_lateral_clearance << ' '
      << active_params_.lambda_lateral_bumper << ' ' << active_params_.lambda_cbf << '\n';
    for (const auto &p : path_room_) f << "path " << p.x() << ' ' << p.y() << '\n';
    for (const auto &p : lidar_points) f << "lidar " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (const auto &p : static_obstacle_points_room_) f << "obs " << p.x() << ' ' << p.y() << '\n';
    for (const auto &p : room_boundary_points_room_) f << "wall " << p.x() << ' ' << p.y() << '\n';
    std::printf("[mppi] cycle snapshot written to %s (%zu lidar, %zu obstacle, %zu wall points)\n",
                snapshot_path_.c_str(), lidar_points.size(),
                static_obstacle_points_room_.size(), room_boundary_points_room_.size());
    std::fflush(stdout);
}

bool TrajectoryController::load_snapshot(std::istream &is, Eigen::Affine2f &pose_out,
                                         std::vector<Eigen::Vector3f> &lidar_out)
{
    std::vector<Eigen::Vector2f> path;
    lidar_out.clear();
    static_obstacle_points_room_.clear();
    room_boundary_points_room_.clear();
    bool have_pose = false;
    std::string line;
    while (std::getline(is, line))
    {
        if (line.empty() or line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key; ls >> key;
        if (key == "pose")
        {
            float x, y, th; ls >> x >> y >> th;
            pose_out = Eigen::Translation2f(x, y) * Eigen::Rotation2Df(th);
            have_pose = true;
        }
        else if (key == "params")
            ls >> params.max_adv >> params.max_rot >> params.d_safe >> params.mppi_lambda
               >> params.num_samples >> params.trajectory_steps >> params.trajectory_dt
               >> params.lambda_obstacle >> params.lambda_goal >> params.lambda_lateral_clearance
               >> params.lambda_lateral_bumper >> params.lambda_cbf;
        else if (key == "path")  { Eigen::Vector2f p; ls >> p.x() >> p.y(); path.push_back(p); }
        else if (key == "lidar") { Eigen::Vector3f p; ls >> p.x() >> p.y() >> p.z(); lidar_out.push_back(p); }
        else if (key == "obs")   { Eigen::Vector2f p; ls >> p.x() >> p.y(); static_obstacle_points_room_.push_back(p); }
        else if (key == "wall")  { Eigen::Vector2f p; ls >> p.x() >> p.y(); room_boundary_points_room_.push_back(p); }
    }
    if (not have_pose or path.size() < 2) return false;
    set_path_presmoothed(path);      // the route is already smoothed and feasibility-checked upstream
    return true;
}

void TrajectoryController::build_esdf(const std::vector<Eigen::Vector3f>& lidar_points,
                                      const Eigen::Affine2f& robot_pose)
{
    const float res = active_params_.grid_resolution;
    const float half = active_params_.grid_half_size;
    const int N = static_cast<int>(2.f * half / res);
    esdf_N_ = N;

    std::vector<int> occ(N * N, 0);

    // Mark lidar points (already in robot frame)
    for (const auto& p : lidar_points)
    {
        const int ci = static_cast<int>((p.x() + half) / res);
        const int cj = static_cast<int>((p.y() + half) / res);
        if (ci >= 0 && ci < N && cj >= 0 && cj < N)
            occ[cj * N + ci] = 1;
    }

    // ── WHAT THE LIDAR ACTUALLY MEASURED, kept separate from what the room MODEL asserts ──────────
    // This grid mixes two sources with different frames of truth. Lidar points are EGOCENTRIC: they are
    // already in the robot frame and a pose error cannot move them. The furniture and the room polygon
    // are room-frame MODEL, transformed in below with robot_pose.inverse() — so a pose error moves them,
    // and a pose JUMP slides them sideways relative to the lidar returns of the very same walls.
    // Measured 2026-08-02: the localiser snaps laterally by 75 mm on ~7% of updates. Taking the union of
    // two disagreeing wall sets then paints a PHANTOM wall alongside the real one, narrowing the apparent
    // corridor by up to that offset and moving it every time the pose jumps. Everything downstream reads
    // it: clip_carrot_to_reachable, the safety gate, the PD bumper's side probes, the band's live field.
    //
    // The model exists to fill what the lidar CANNOT see (walls vanish at grazing incidence and close
    // range) — not to restate what it already reports. So injection now defers to the measurement: a
    // model point is dropped where the lidar has already put an obstacle nearby. Where they agree it was
    // adding nothing; where they disagree the lidar is the one that cannot be wrong about the robot frame.
    const std::vector<int> lidar_occ = occ;
    const int merge_cells = std::max(0, static_cast<int>(std::lround(
                                active_params_.model_merge_radius_m / std::max(res, 1e-3f))));
    const auto lidar_says_obstacle_near = [&](int ci, int cj)
    {
        for (int dj = -merge_cells; dj <= merge_cells; ++dj)
            for (int di = -merge_cells; di <= merge_cells; ++di)
            {
                const int i = ci + di, j = cj + dj;
                if (i >= 0 && i < N && j >= 0 && j < N && lidar_occ[j * N + i]) return true;
            }
        return false;
    };
    esdf_model_points_dropped_ = 0;

    // Inject static obstacle points (furniture) — transform from room frame to robot frame
    if (!static_obstacle_points_room_.empty())
    {
        const Eigen::Affine2f robot_inv = robot_pose.inverse();
        for (const auto& p_room : static_obstacle_points_room_)
        {
            const Eigen::Vector2f p_robot = robot_inv * p_room;
            const int ci = static_cast<int>((p_robot.x() + half) / res);
            const int cj = static_cast<int>((p_robot.y() + half) / res);
            if (ci >= 0 && ci < N && cj >= 0 && cj < N)
            {
                if (lidar_says_obstacle_near(ci, cj)) { ++esdf_model_points_dropped_; continue; }
                occ[cj * N + ci] = 1;
            }
        }
    }

    // ── THE WALLS ────────────────────────────────────────────────────────────────────────────────
    // The room boundary was sampled by set_room_boundary and, until now, was consumed ONLY by
    // relax_path and smooth_path_spline — path SHAPING. It never reached this grid, so every safety
    // term downstream (obstacle cost, CBF barrier, collision test, and all six lateral probes) knew a
    // wall existed only while the LiDAR was currently returning from it. Three things make that fail
    // exactly at a thin salient corner, which is where the robot was observed clipping walls:
    //   • the self-filter drops every return within max(0.40, body+0.08) = 0.405 m of the robot centre,
    //     against a circumscribed radius of 0.325 — so a corner closing on the body is DELETED from the
    //     field in the last 8 cm, precisely when it matters;
    //   • a salient corner seen at a grazing angle yields few returns, and its far face is occluded
    //     until the robot is already around it;
    //   • probing harder cannot help: the probes read this field, and the wall was not in it.
    // Walls are geometry we already know from the room model, so there is no reason to rediscover them
    // from a sensor every cycle. Same room->robot transform as the furniture above.
    esdf_boundary_cells_ = 0;
    esdf_boundary_rejected_ = false;
    if (!room_boundary_points_room_.empty())
    {
        const Eigen::Affine2f robot_inv = robot_pose.inverse();
        // Remember only the cells THIS pass turned on, so a rollback cannot erase a wall the LiDAR or the
        // furniture set had already reported. Clearing every boundary cell would delete real obstacles
        // wherever the two happen to coincide — which is exactly along the walls, i.e. everywhere it matters.
        boundary_cells_.clear();
        for (const auto& p_room : room_boundary_points_room_)
        {
            const Eigen::Vector2f p_robot = robot_inv * p_room;
            const int ci = static_cast<int>((p_robot.x() + half) / res);
            const int cj = static_cast<int>((p_robot.y() + half) / res);
            if (ci >= 0 && ci < N && cj >= 0 && cj < N)
            {
                // Same deference as the furniture above: where the lidar already reports a wall, the
                // model must not paint a second one offset by the pose error. See build_esdf's header.
                if (lidar_says_obstacle_near(ci, cj)) { ++esdf_model_points_dropped_; continue; }
                if (!occ[cj * N + ci]) { occ[cj * N + ci] = 1; boundary_cells_.push_back(cj * N + ci); }
            }
        }
        // A room polygon is ONE input, and this stack has already been bitten once by a bad one (the
        // grid planner saw 27018 of 27018 cells "outside" and refused to plan anywhere). A boundary is
        // an outline, so it should mark a thin trace — if it claims a quarter of the window, it is not
        // an outline and every rollout would collide. Drop it and say so rather than freeze the robot.
        esdf_boundary_cells_ = static_cast<int>(boundary_cells_.size());
        if (esdf_boundary_cells_ > N * N / 4)
        {
            for (const int c : boundary_cells_) occ[c] = 0;
            esdf_boundary_rejected_ = true;
            esdf_boundary_cells_ = 0;
            boundary_cells_.clear();
        }
    }

    esdf_data_.assign(N * N, active_params_.esdf_init_distance);

    // Forward pass
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
        {
            const int idx = j * N + i;
            if (occ[idx]) { esdf_data_[idx] = 0.f; continue; }
            float d = active_params_.esdf_init_distance;
            if (i > 0)            d = std::min(d, esdf_data_[idx - 1] + 1.f);
            if (j > 0)            d = std::min(d, esdf_data_[(j-1)*N + i] + 1.f);
            if (i > 0 && j > 0)   d = std::min(d, esdf_data_[(j-1)*N + (i-1)] + active_params_.esdf_diag_step);
            if (i < N-1 && j > 0) d = std::min(d, esdf_data_[(j-1)*N + (i+1)] + active_params_.esdf_diag_step);
            esdf_data_[idx] = d;
        }

    // Backward pass
    for (int j = N-1; j >= 0; --j)
        for (int i = N-1; i >= 0; --i)
        {
            const int idx = j * N + i;
            float d = esdf_data_[idx];
            if (i < N-1)              d = std::min(d, esdf_data_[idx + 1] + 1.f);
            if (j < N-1)              d = std::min(d, esdf_data_[(j+1)*N + i] + 1.f);
            if (i < N-1 && j < N-1)   d = std::min(d, esdf_data_[(j+1)*N + (i+1)] + active_params_.esdf_diag_step);
            if (i > 0   && j < N-1)   d = std::min(d, esdf_data_[(j+1)*N + (i-1)] + active_params_.esdf_diag_step);
            esdf_data_[idx] = d;
        }

    for (auto& d : esdf_data_) d *= res;
}

std::vector<Eigen::Vector3f> TrajectoryController::read_lidar_points_robot(const Eigen::Affine2f& robot_pose) const
{
    if (lidar_buffer_ == nullptr)
        return {};

    const auto [cloud_opt] = lidar_buffer_->read_last();
    if (!cloud_opt.has_value())
        return {};

    const auto &[xs_room, ys_room, zs_room] = cloud_opt.value();
    const std::size_t count = std::min({xs_room.size(), ys_room.size(), zs_room.size()});
    if (count == 0)
        return {};

    const Eigen::Affine2f robot_from_room = robot_pose.inverse();
    std::vector<Eigen::Vector3f> lidar_points;
    lidar_points.reserve(count);

    // Floor on the configured self-filter disc. Worst-case extent: this radius is applied before any bearing
    // is known, so it has to cover the body whichever way a return comes in.
    const float self_filter_radius = std::max(active_params_.esdf_self_filter_radius,
                                              body_extent_max() + 0.08f);
    const float self_filter_half_width = std::max(0.05f, active_params_.esdf_self_filter_half_width);
    const float self_filter_front = std::max(0.05f, active_params_.esdf_self_filter_front);
    const float self_filter_rear = std::max(0.05f, active_params_.esdf_self_filter_rear);

    for (std::size_t index = 0; index < count; ++index)
    {
        const float x_room = xs_room[index];
        const float y_room = ys_room[index];
        const float z_room = zs_room[index];
        if (!std::isfinite(x_room) || !std::isfinite(y_room) || !std::isfinite(z_room))
            continue;

        const Eigen::Vector2f point_robot = robot_from_room * Eigen::Vector2f(x_room, y_room);

        // Remove returns that fall inside the robot body envelope. These points
        // are typically self-reflections or very near-body clutter and should
        // not collapse the local ESDF around the robot center.
        if (point_robot.norm() < self_filter_radius)
            continue;
        if (std::abs(point_robot.x()) < self_filter_half_width
            && point_robot.y() > -self_filter_rear
            && point_robot.y() < self_filter_front)
            continue;

        lidar_points.emplace_back(point_robot.x(), point_robot.y(), z_room);
    }

    return lidar_points;
}

float TrajectoryController::query_esdf(float rx, float ry) const
{
    if (esdf_data_.empty()) return active_params_.esdf_unknown_distance;
    const float res = active_params_.grid_resolution;
    const float half = active_params_.grid_half_size;
    const int N = esdf_N_;

    const float gx = (rx + half) / res;
    const float gy = (ry + half) / res;
    const int ix = static_cast<int>(std::floor(gx));
    const int iy = static_cast<int>(std::floor(gy));
    if (ix < 0 || ix >= N-1 || iy < 0 || iy >= N-1) return active_params_.esdf_unknown_distance;

    const float fx = gx - static_cast<float>(ix);
    const float fy = gy - static_cast<float>(iy);
    const float d00 = esdf_data_[iy * N + ix];
    const float d10 = esdf_data_[iy * N + ix + 1];
    const float d01 = esdf_data_[(iy+1) * N + ix];
    const float d11 = esdf_data_[(iy+1) * N + ix + 1];
    return (1.f-fx)*(1.f-fy)*d00 + fx*(1.f-fy)*d10 + (1.f-fx)*fy*d01 + fx*fy*d11;
}

Eigen::Vector2f TrajectoryController::query_esdf_gradient(float rx, float ry) const
{
    const float h = active_params_.grid_resolution;
    const float dx = query_esdf(rx + h, ry) - query_esdf(rx - h, ry);
    const float dy = query_esdf(rx, ry + h) - query_esdf(rx, ry - h);
    Eigen::Vector2f grad(dx / (2.f * h), dy / (2.f * h));
    const float mag = grad.norm();
    if (mag > active_params_.esdf_grad_min_norm) grad /= mag;
    else grad = Eigen::Vector2f::Zero();
    return grad;
}

// ============================================================================
// Coordinate transform
// ============================================================================

TrackerInput TrajectoryController::make_tracker_input(const Eigen::Affine2f& robot_pose,
                                                      const Eigen::Vector2f& carrot_robot,
                                                      const Eigen::Vector2f& goal_robot) const
{
    TrackerInput in;
    in.robot_pose      = robot_pose;
    in.carrot_robot    = carrot_robot;
    in.goal_robot      = goal_robot;
    in.path_proj_room  = last_path_proj_room_;
    in.path_proj_valid = last_path_proj_valid_;
    return in;
}

Eigen::Vector2f TrajectoryController::room_to_robot(const Eigen::Vector2f& p_room,
                                                     const Eigen::Affine2f& robot_pose)
{
    return robot_pose.linear().transpose() * (p_room - robot_pose.translation());
}



} // namespace rc

