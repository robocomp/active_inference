#pragma once

#include <Eigen/Dense>

#include <optional>
#include <vector>

#include "../../common/robust_metrics/robust_metrics.h"
#include "controller_runtime_types.h"

struct ControllerObstacleState
{
    Eigen::Vector2f center = Eigen::Vector2f::Zero();
    float yaw_rad = 0.f;
    float width_m = 0.35f;
    float depth_m = 0.35f;
};

struct ControllerObstacleObservation
{
    std::vector<Eigen::Vector2f> points;
    std::vector<float> weights;
    Eigen::Vector2f viewpoint = Eigen::Vector2f::Zero();
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    float yaw_rad = 0.f;
    float width_m = 0.35f;
    float depth_m = 0.35f;
};

struct ControllerObstacleModelParams
{
    float sigma_obs_m = 0.06f;
    float lambda_pos = 18.f;
    float lambda_size = 2.5f;
    float lambda_angle = 3.0f;
    float lambda_anchor_size = 0.6f;
    float lambda_compact = 0.4f;
    float lambda_support = 10.f;
    RobustLossType robust_loss = RobustLossType::Huber;
    float robust_loss_scale_m = 0.08f;
    float min_size_m = 0.20f;
    float max_size_m = 1.80f;
    float max_center_step_m = 0.05f;
    float max_size_step_m = 0.18f;
    float max_yaw_step_rad = 0.08f;
    float occlusion_growth_cap_m = 0.12f;
    float association_margin_m = 0.25f;
    float pose_precision_min = 1.0f;
    float pose_precision_max = 12.0f;
    float pose_precision_alpha = 0.90f;
    float size_precision_min = 1.0f;
    float size_precision_max = 12.0f;
    float size_precision_alpha = 0.90f;
    float anchor_size_learning_rate = 0.10f;
};

class ControllerObstacleModel
{
public:
    ControllerObstacleModel() = default;
    ControllerObstacleModel(const ControllerObstacleState &initial_state,
                            ControllerObstacleModelParams params = {});

    static std::optional<ControllerObstacleObservation> make_observation(const std::vector<Eigen::Vector2f> &points,
                                                                         const Eigen::Vector2f &viewpoint,
                                                                         float padding_m,
                                                                         float occlusion_depth_m);

    float update(const ControllerObstacleObservation &observation);
    float compute_free_energy(const ControllerObstacleObservation &observation,
                              const ControllerObstacleState &candidate) const;

    const ControllerObstacleState &state() const { return state_; }
    const ControllerObstacleState &prior() const { return prior_; }
    const ControllerObstacleState &anchor() const { return anchor_state_; }
    const ControllerObstacleModelParams &params() const { return params_; }
    float free_energy() const { return free_energy_; }
    float posterior_precision() const { return 0.5f * (pose_precision_ + size_precision_); }
    float pose_precision() const { return pose_precision_; }
    float size_precision() const { return size_precision_; }

    void set_state(const ControllerObstacleState &state);
    void set_prior(const ControllerObstacleState &prior);

    float association_radius() const;
    ControllerPolygon polygon() const;

    static ControllerPolygon polygon_from_state(const ControllerObstacleState &state);

private:
    static float wrap_angle(float angle);
    static float clamp_angle_delta(float delta, float limit);
    static float box_sdf(const Eigen::Vector2f &point, const ControllerObstacleState &state);
    static float support_along_polygon(const ControllerPolygon &polygon, const Eigen::Vector2f &dir);
    static ControllerObstacleState interpolate(const ControllerObstacleState &from,
                                               const ControllerObstacleState &to,
                                               float alpha);
    static ControllerObstacleState align_observation_to_prior(const ControllerObstacleState &prior,
                                                              const ControllerObstacleObservation &observation);

    void apply_constraints(ControllerObstacleState &state) const;

    ControllerObstacleState state_;
    ControllerObstacleState prior_;
    ControllerObstacleState anchor_state_;
    ControllerObstacleModelParams params_;
    float free_energy_ = 0.f;
    float pose_precision_ = 1.f;
    float size_precision_ = 1.f;
};