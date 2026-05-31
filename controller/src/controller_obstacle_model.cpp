#include "controller_obstacle_model.h"

#include "../../common/robust_metrics/robust_metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
constexpr float kPi = 3.14159265358979323846f;
}

ControllerObstacleModel::ControllerObstacleModel(const ControllerObstacleState &initial_state,
                                                 ControllerObstacleModelParams params)
    : state_(initial_state), prior_(initial_state), anchor_state_(initial_state), params_(std::move(params))
{
    apply_constraints(state_);
    prior_ = state_;
    anchor_state_ = state_;
    pose_precision_ = params_.pose_precision_min;
    size_precision_ = params_.size_precision_min;
}

std::optional<ControllerObstacleObservation> ControllerObstacleModel::make_observation(const std::vector<Eigen::Vector2f> &points,
                                                                                       const Eigen::Vector2f &viewpoint,
                                                                                       float padding_m,
                                                                                       float occlusion_depth_m)
{
    if (points.size() < 3)
        return std::nullopt;

    ControllerObstacleObservation observation;
    observation.points = points;
    observation.weights.assign(points.size(), 1.f);
    observation.viewpoint = viewpoint;

    for (const auto &point : points)
        observation.centroid += point;
    observation.centroid /= static_cast<float>(points.size());

    Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
    for (const auto &point : points)
    {
        const Eigen::Vector2f delta = point - observation.centroid;
        covariance += delta * delta.transpose();
    }
    covariance /= static_cast<float>(points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
    Eigen::Vector2f principal = Eigen::Vector2f::UnitX();
    if (solver.info() == Eigen::Success)
        principal = solver.eigenvectors().col(1).normalized();

    observation.yaw_rad = std::atan2(principal.y(), principal.x());
    const Eigen::Rotation2Df room_to_local(-observation.yaw_rad);
    Eigen::Vector2f min_local = Eigen::Vector2f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector2f max_local = Eigen::Vector2f::Constant(-std::numeric_limits<float>::max());
    for (const auto &point : points)
    {
        const Eigen::Vector2f local = room_to_local * (point - observation.centroid);
        min_local = min_local.cwiseMin(local);
        max_local = max_local.cwiseMax(local);
    }

    observation.width_m = std::max(0.12f, max_local.x() - min_local.x() + 2.f * padding_m);
    observation.depth_m = std::max(0.12f, max_local.y() - min_local.y() + 2.f * padding_m);

    const Eigen::Vector2f view_local = room_to_local * (viewpoint - observation.centroid);
    const float view_norm = view_local.norm();
    if (view_norm > 1e-4f && occlusion_depth_m > 0.f)
    {
        const Eigen::Vector2f dominant = view_local.cwiseAbs();
        if (dominant.x() >= dominant.y())
            observation.width_m += std::min(occlusion_depth_m, occlusion_depth_m * dominant.x() / view_norm);
        else
            observation.depth_m += std::min(occlusion_depth_m, occlusion_depth_m * dominant.y() / view_norm);
    }

    return observation;
}

float ControllerObstacleModel::wrap_angle(float angle)
{
    while (angle <= -kPi)
        angle += 2.f * kPi;
    while (angle > kPi)
        angle -= 2.f * kPi;
    return angle;
}

float ControllerObstacleModel::clamp_angle_delta(float delta, float limit)
{
    return std::clamp(wrap_angle(delta), -limit, limit);
}

float ControllerObstacleModel::box_sdf(const Eigen::Vector2f &point, const ControllerObstacleState &state)
{
    const Eigen::Vector2f local = Eigen::Rotation2Df(-state.yaw_rad) * (point - state.center);
    const Eigen::Vector2f half_extents(state.width_m * 0.5f, state.depth_m * 0.5f);
    const Eigen::Vector2f d = local.cwiseAbs() - half_extents;
    const Eigen::Vector2f outside = d.cwiseMax(0.f);
    const float outside_norm = outside.norm();
    const float inside = std::min(std::max(d.x(), d.y()), 0.f);
    return outside_norm + 0.25f * inside;
}

float ControllerObstacleModel::support_along_polygon(const ControllerPolygon &polygon, const Eigen::Vector2f &dir)
{
    float support = -std::numeric_limits<float>::max();
    for (const auto &point : polygon)
        support = std::max(support, point.dot(dir));
    return support;
}

ControllerObstacleState ControllerObstacleModel::interpolate(const ControllerObstacleState &from,
                                                             const ControllerObstacleState &to,
                                                             float alpha)
{
    ControllerObstacleState state;
    state.center = from.center + alpha * (to.center - from.center);
    state.width_m = from.width_m + alpha * (to.width_m - from.width_m);
    state.depth_m = from.depth_m + alpha * (to.depth_m - from.depth_m);
    state.yaw_rad = wrap_angle(from.yaw_rad + alpha * wrap_angle(to.yaw_rad - from.yaw_rad));
    return state;
}

ControllerObstacleState ControllerObstacleModel::align_observation_to_prior(const ControllerObstacleState &prior,
                                                                            const ControllerObstacleObservation &observation)
{
    ControllerObstacleState direct{.center = observation.centroid,
                                   .yaw_rad = observation.yaw_rad,
                                   .width_m = observation.width_m,
                                   .depth_m = observation.depth_m};
    ControllerObstacleState swapped{.center = observation.centroid,
                                    .yaw_rad = wrap_angle(observation.yaw_rad + kPi * 0.5f),
                                    .width_m = observation.depth_m,
                                    .depth_m = observation.width_m};

    const float direct_delta = std::abs(wrap_angle(direct.yaw_rad - prior.yaw_rad));
    const float swapped_delta = std::abs(wrap_angle(swapped.yaw_rad - prior.yaw_rad));
    return swapped_delta < direct_delta ? swapped : direct;
}

void ControllerObstacleModel::apply_constraints(ControllerObstacleState &state) const
{
    state.width_m = std::clamp(state.width_m, params_.min_size_m, params_.max_size_m);
    state.depth_m = std::clamp(state.depth_m, params_.min_size_m, params_.max_size_m);
    state.yaw_rad = wrap_angle(state.yaw_rad);
}

float ControllerObstacleModel::compute_free_energy(const ControllerObstacleObservation &observation,
                                                   const ControllerObstacleState &candidate) const
{
    const float inv_sigma2 = 1.f / std::max(1e-4f, params_.sigma_obs_m * params_.sigma_obs_m);
    float likelihood = 0.f;
    float weight_sum = 0.f;
    for (std::size_t index = 0; index < observation.points.size(); ++index)
    {
        const float weight = index < observation.weights.size() ? observation.weights[index] : 1.f;
        likelihood += weight * robust_loss_value(box_sdf(observation.points[index], candidate),
                                                 params_.robust_loss,
                                                 params_.robust_loss_scale_m) * inv_sigma2;
        weight_sum += weight;
    }
    likelihood /= std::max(1e-4f, weight_sum);

    const float pose_precision = std::clamp(pose_precision_,
                                            params_.pose_precision_min,
                                            params_.pose_precision_max);
    const float size_precision = std::clamp(size_precision_,
                                            params_.size_precision_min,
                                            params_.size_precision_max);
    const Eigen::Vector2f center_delta = candidate.center - anchor_state_.center;
    const float angle_delta = wrap_angle(candidate.yaw_rad - anchor_state_.yaw_rad);
    const float dynamic_size_delta = std::pow(candidate.width_m - prior_.width_m, 2.f)
                                   + std::pow(candidate.depth_m - prior_.depth_m, 2.f);
    const float anchor_size_delta = std::pow(candidate.width_m - anchor_state_.width_m, 2.f)
                                  + std::pow(candidate.depth_m - anchor_state_.depth_m, 2.f);

    const float prior_energy = params_.lambda_pos * pose_precision * center_delta.squaredNorm()
                             + params_.lambda_size * size_precision * dynamic_size_delta
                             + params_.lambda_anchor_size * std::sqrt(size_precision) * anchor_size_delta
                             + params_.lambda_angle * pose_precision * angle_delta * angle_delta;

    const float compactness = params_.lambda_compact * candidate.width_m * candidate.depth_m;

    float support_penalty = 0.f;
    Eigen::Vector2f occlusion_dir = candidate.center - observation.viewpoint;
    const float occlusion_norm = occlusion_dir.norm();
    if (occlusion_norm > 1e-4f)
    {
        occlusion_dir /= occlusion_norm;
        float observed_support = -std::numeric_limits<float>::max();
        for (const auto &point : observation.points)
            observed_support = std::max(observed_support, point.dot(occlusion_dir));
        const float predicted_support = support_along_polygon(polygon_from_state(candidate), occlusion_dir);
        const float excess = std::max(0.f, predicted_support - observed_support - params_.occlusion_growth_cap_m);
        support_penalty = params_.lambda_support * excess * excess;
    }

    return likelihood + prior_energy + compactness + support_penalty;
}

float ControllerObstacleModel::update(const ControllerObstacleObservation &observation)
{
    const ControllerObstacleState aligned_observation = align_observation_to_prior(state_, observation);
    ControllerObstacleState clamped = state_;
    const float pose_precision = std::clamp(pose_precision_,
                                            params_.pose_precision_min,
                                            params_.pose_precision_max);
    const float size_precision = std::clamp(size_precision_,
                                            params_.size_precision_min,
                                            params_.size_precision_max);
    const float pose_hardening = std::sqrt(pose_precision);
    const float size_hardening = std::sqrt(size_precision);

    Eigen::Vector2f center_delta = aligned_observation.center - state_.center;
    const float center_norm = center_delta.norm();
    const float max_center_step = params_.max_center_step_m / std::max(1.f, pose_hardening);
    if (center_norm > max_center_step)
        center_delta *= max_center_step / center_norm;
    clamped.center += center_delta;

    const float max_size_step = params_.max_size_step_m / std::max(1.f, size_hardening);
    clamped.width_m += std::clamp(aligned_observation.width_m - state_.width_m,
                                  -max_size_step,
                                  max_size_step);
    clamped.depth_m += std::clamp(aligned_observation.depth_m - state_.depth_m,
                                  -max_size_step,
                                  max_size_step);
    const float max_yaw_step = params_.max_yaw_step_rad / std::max(1.f, pose_hardening);
    clamped.yaw_rad = wrap_angle(state_.yaw_rad + clamp_angle_delta(aligned_observation.yaw_rad - state_.yaw_rad,
                                                                    max_yaw_step));
    apply_constraints(clamped);

    std::array<ControllerObstacleState, 5> candidates{
        state_,
        interpolate(state_, clamped, 0.25f),
        interpolate(state_, clamped, 0.50f),
        interpolate(state_, clamped, 0.75f),
        clamped};

    const float current_energy = compute_free_energy(observation, state_);
    const ControllerObstacleState *best_candidate = &candidates.front();
    float best_energy = current_energy;
    for (std::size_t index = 1; index < candidates.size(); ++index)
    {
        apply_constraints(candidates[index]);
        const float candidate_energy = compute_free_energy(observation, candidates[index]);
        if (candidate_energy < best_energy)
        {
            best_energy = candidate_energy;
            best_candidate = &candidates[index];
        }
    }

    state_ = *best_candidate;
    apply_constraints(state_);

        const float sigma2 = std::max(1e-4f, params_.sigma_obs_m * params_.sigma_obs_m);
        const Eigen::Vector2f pose_residual = aligned_observation.center - state_.center;
        const float angle_residual = wrap_angle(aligned_observation.yaw_rad - state_.yaw_rad);
        const float pose_prediction_error = pose_residual.squaredNorm() / sigma2 + angle_residual * angle_residual;
        const float width_residual = aligned_observation.width_m - state_.width_m;
        const float depth_residual = aligned_observation.depth_m - state_.depth_m;
        const float size_prediction_error = (width_residual * width_residual + depth_residual * depth_residual) / sigma2;

        const float target_pose_precision = params_.pose_precision_min
                              + (params_.pose_precision_max - params_.pose_precision_min)
                                  / (1.f + pose_prediction_error);
        const float target_size_precision = params_.size_precision_min
                              + (params_.size_precision_max - params_.size_precision_min)
                                  / (1.f + size_prediction_error);
        pose_precision_ = params_.pose_precision_alpha * pose_precision_
                  + (1.f - params_.pose_precision_alpha) * target_pose_precision;
        size_precision_ = params_.size_precision_alpha * size_precision_
                  + (1.f - params_.size_precision_alpha) * target_size_precision;
        pose_precision_ = std::clamp(pose_precision_, params_.pose_precision_min, params_.pose_precision_max);
        size_precision_ = std::clamp(size_precision_, params_.size_precision_min, params_.size_precision_max);

        const float anchor_pose_learning_rate = 1.f / pose_precision_;
        const float anchor_size_learning_rate = params_.anchor_size_learning_rate / size_precision_;
    anchor_state_.center += anchor_pose_learning_rate * (state_.center - anchor_state_.center);
    anchor_state_.yaw_rad = wrap_angle(anchor_state_.yaw_rad
                                     + anchor_pose_learning_rate * wrap_angle(state_.yaw_rad - anchor_state_.yaw_rad));
        anchor_state_.width_m += anchor_size_learning_rate * (state_.width_m - anchor_state_.width_m);
        anchor_state_.depth_m += anchor_size_learning_rate * (state_.depth_m - anchor_state_.depth_m);
    apply_constraints(anchor_state_);

    prior_ = state_;
    free_energy_ = best_energy;
    return free_energy_;
}

void ControllerObstacleModel::set_state(const ControllerObstacleState &state)
{
    state_ = state;
    apply_constraints(state_);
    anchor_state_ = state_;
    pose_precision_ = params_.pose_precision_min;
    size_precision_ = params_.size_precision_min;
}

void ControllerObstacleModel::set_prior(const ControllerObstacleState &prior)
{
    prior_ = prior;
    apply_constraints(prior_);
}

float ControllerObstacleModel::association_radius() const
{
    return 0.5f * std::sqrt(state_.width_m * state_.width_m + state_.depth_m * state_.depth_m)
         + params_.association_margin_m;
}

ControllerPolygon ControllerObstacleModel::polygon() const
{
    return polygon_from_state(state_);
}

ControllerPolygon ControllerObstacleModel::polygon_from_state(const ControllerObstacleState &state)
{
    const float half_width = state.width_m * 0.5f;
    const float half_depth = state.depth_m * 0.5f;
    const Eigen::Rotation2Df rotation(state.yaw_rad);

    ControllerPolygon polygon;
    polygon.reserve(4);
    for (const Eigen::Vector2f &corner : {Eigen::Vector2f(-half_width, -half_depth),
                                          Eigen::Vector2f(half_width, -half_depth),
                                          Eigen::Vector2f(half_width, half_depth),
                                          Eigen::Vector2f(-half_width, half_depth)})
        polygon.push_back(state.center + rotation * corner);

    return polygon;
}