#include "controller_obstacle_tracker.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <print>
#include <sstream>
#include <unordered_set>

std::array<Eigen::Vector2f, ControllerObstacleTracker::kRememberedEdgeCount>
ControllerObstacleTracker::make_local_obstacle_corners(const ControllerObstacleState &state)
{
    const float half_width = state.width_m * 0.5f;
    const float half_depth = state.depth_m * 0.5f;
    return {Eigen::Vector2f(-half_width, -half_depth),
            Eigen::Vector2f(half_width, -half_depth),
            Eigen::Vector2f(half_width, half_depth),
            Eigen::Vector2f(-half_width, half_depth)};
}

std::optional<ControllerObstacleTracker::RememberedEdgeSlot>
ControllerObstacleTracker::classify_remembered_edge_slot(const ControllerObstacleState &state,
                                                         const Eigen::Vector2f &local_point)
{
    const auto corners = make_local_obstacle_corners(state);
    float best_distance = std::numeric_limits<float>::max();
    float best_t = 0.f;
    int best_edge = 0;

    for (int edge_index = 0; edge_index < kRememberedEdgeCount; ++edge_index)
    {
        const Eigen::Vector2f &from = corners[edge_index];
        const Eigen::Vector2f &to = corners[(edge_index + 1) % kRememberedEdgeCount];
        const Eigen::Vector2f edge = to - from;
        const float edge_length_sq = edge.squaredNorm();
        if (edge_length_sq < 1e-6f)
            continue;

        const float projected_t = std::clamp((local_point - from).dot(edge) / edge_length_sq, 0.f, 1.f);
        const Eigen::Vector2f projected = from + projected_t * edge;
        const float distance = (local_point - projected).norm();
        if (distance < best_distance)
        {
            best_distance = distance;
            best_t = projected_t;
            best_edge = edge_index;
        }
    }

    if (best_distance > kRememberedEdgeDistanceThreshold)
        return std::nullopt;

    const int slot_index = std::min(kRememberedSlotsPerEdge - 1,
                                    static_cast<int>(best_t * static_cast<float>(kRememberedSlotsPerEdge)));
    return RememberedEdgeSlot{.edge_index = best_edge,
                              .slot_index = slot_index,
                              .distance_to_edge = best_distance};
}

int ControllerObstacleTracker::remembered_slot_index(const RememberedEdgeSlot &slot)
{
    return slot.edge_index * kRememberedSlotsPerEdge + slot.slot_index;
}

Eigen::Vector2f ControllerObstacleTracker::estimate_obstacle_center(const ControllerObstacleObservation &observation,
                                                                    float width_m,
                                                                    float depth_m)
{
    const Eigen::Vector2f view_dir = observation.centroid - observation.viewpoint;
    const float view_norm = view_dir.norm();
    if (view_norm < 1e-4f || observation.points.empty())
        return observation.centroid;

    const Eigen::Vector2f away_from_robot = view_dir / view_norm;
    float observed_min = std::numeric_limits<float>::max();
    float observed_max = -std::numeric_limits<float>::max();
    for (const auto &point : observation.points)
    {
        const float projection = point.dot(away_from_robot);
        observed_min = std::min(observed_min, projection);
        observed_max = std::max(observed_max, projection);
    }

    const Eigen::Vector2f local_dir = Eigen::Rotation2Df(-observation.yaw_rad) * away_from_robot;
    const float estimated_full_span = std::abs(local_dir.x()) * width_m
                                    + std::abs(local_dir.y()) * depth_m;
    const float observed_span = std::max(0.f, observed_max - observed_min);
    const float hidden_span = std::max(0.f, estimated_full_span - observed_span);
    const float observed_mid = 0.5f * (observed_min + observed_max);
    const float target_projection = std::max(observed_max + 0.02f,
                                             observed_mid + 0.5f * hidden_span);
    const float centroid_projection = observation.centroid.dot(away_from_robot);
    return observation.centroid + (target_projection - centroid_projection) * away_from_robot;
}

Eigen::Vector2f ControllerObstacleTracker::estimate_initial_obstacle_center(const ControllerObstacleObservation &observation)
{
    return estimate_obstacle_center(observation, observation.width_m, observation.depth_m);
}

std::string ControllerObstacleTracker::published_obstacle_name(std::uint64_t obstacle_id)
{
    return "obs" + std::to_string(obstacle_id);
}

float ControllerObstacleTracker::distance_visibility_scale(const ControllerParams *params,
                                                           const ControllerRobotPose &robot_pose,
                                                           const ControllerObstacleState &state)
{
    if (params == nullptr)
        return 1.f;

    const float distance_to_obstacle = (state.center - robot_pose.pos).norm();
    const float reference_distance = std::max(0.5f, params->temporary_obstacle_front_distance_m);
    return std::clamp(reference_distance / std::max(reference_distance, distance_to_obstacle),
                      0.2f,
                      1.f);
}

void ControllerObstacleTracker::recompute_observation_summary(ControllerObstacleObservation &observation,
                                                              float padding_m,
                                                              float occlusion_depth_m)
{
    if (observation.points.size() < 3)
        return;

    float weight_sum = 0.f;
    observation.centroid.setZero();
    for (std::size_t index = 0; index < observation.points.size(); ++index)
    {
        const float weight = index < observation.weights.size() ? observation.weights[index] : 1.f;
        observation.centroid += weight * observation.points[index];
        weight_sum += weight;
    }
    observation.centroid /= std::max(1e-4f, weight_sum);

    Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
    for (std::size_t index = 0; index < observation.points.size(); ++index)
    {
        const float weight = index < observation.weights.size() ? observation.weights[index] : 1.f;
        const Eigen::Vector2f delta = observation.points[index] - observation.centroid;
        covariance += weight * delta * delta.transpose();
    }
    covariance /= std::max(1e-4f, weight_sum);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
    Eigen::Vector2f principal = Eigen::Vector2f::UnitX();
    if (solver.info() == Eigen::Success)
        principal = solver.eigenvectors().col(1).normalized();

    observation.yaw_rad = std::atan2(principal.y(), principal.x());
    const Eigen::Rotation2Df room_to_local(-observation.yaw_rad);
    Eigen::Vector2f min_local = Eigen::Vector2f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector2f max_local = Eigen::Vector2f::Constant(-std::numeric_limits<float>::max());
    for (const auto &point : observation.points)
    {
        const Eigen::Vector2f local = room_to_local * (point - observation.centroid);
        min_local = min_local.cwiseMin(local);
        max_local = max_local.cwiseMax(local);
    }

    observation.width_m = std::max(0.12f, max_local.x() - min_local.x() + 2.f * padding_m);
    observation.depth_m = std::max(0.12f, max_local.y() - min_local.y() + 2.f * padding_m);

    const Eigen::Vector2f view_local = room_to_local * (observation.viewpoint - observation.centroid);
    const float view_norm = view_local.norm();
    if (view_norm > 1e-4f && occlusion_depth_m > 0.f)
    {
        const Eigen::Vector2f dominant = view_local.cwiseAbs();
        if (dominant.x() >= dominant.y())
            observation.width_m += std::min(occlusion_depth_m, occlusion_depth_m * dominant.x() / view_norm);
        else
            observation.depth_m += std::min(occlusion_depth_m, occlusion_depth_m * dominant.y() / view_norm);
    }
}

void ControllerObstacleTracker::set_params(const ControllerParams *params)
{
    params_ = params;
}

void ControllerObstacleTracker::set_dependencies(std::shared_ptr<DSR::DSRGraph> graph,
                                                 DSR::InnerEigenAPI *inner_eigen_api,
                                                 const ControllerGraphState *graph_state)
{
    graph_ = std::move(graph);
    inner_eigen_api_ = inner_eigen_api;
    rt_api_ = graph_ ? graph_->get_rt_api() : nullptr;
    graph_state_ = graph_state;
}

void ControllerObstacleTracker::set_graph_layout_callback(std::function<void()> callback)
{
    graph_layout_callback_ = std::move(callback);
}

void ControllerObstacleTracker::clear_published_obstacles()
{
    for (auto &instance : temporary_obstacles_)
    {
        delete_published_obstacle_node(instance);
        instance.published_node_id = 0;
    }
}

ControllerObstacleModelParams ControllerObstacleTracker::make_model_params() const
{
    ControllerObstacleModelParams model_params;
    if (!params_)
        return model_params;

    model_params.robust_loss = params_->temporary_obstacle_robust_loss;
    model_params.robust_loss_scale_m = params_->temporary_obstacle_robust_loss_scale_m;
    return model_params;
}

ControllerPolygons ControllerObstacleTracker::temporary_obstacle_rfe_points() const
{
    ControllerPolygons rfe_points;
    rfe_points.reserve(temporary_obstacles_.size());

    for (const auto &instance : temporary_obstacles_)
    {
        ControllerPolygon points;
        points.reserve(instance.remembered_points.size());

        const auto &state = instance.model.state();
        for (const auto &remembered_point : instance.remembered_points)
            points.push_back(to_room_point(state, remembered_point.local_point));

        rfe_points.push_back(std::move(points));
    }

    return rfe_points;
}

void ControllerObstacleTracker::delete_published_obstacle_node(const TemporaryObstacleInstance &instance)
{
    if (!graph_)
        return;

    if (instance.published_node_id != 0 && graph_->delete_node(instance.published_node_id))
    {
        if (graph_layout_callback_)
            graph_layout_callback_();
        return;
    }

    if (graph_->delete_node(published_obstacle_name(instance.id)) && graph_layout_callback_)
        graph_layout_callback_();
}

void ControllerObstacleTracker::sync_temporary_obstacles_to_dsr(std::uint64_t timestamp_ms)
{
    if (!graph_ || !rt_api_ || !graph_state_ || graph_state_->room_id == 0)
        return;

    auto room_node = graph_->get_node(graph_state_->room_id);
    if (!room_node.has_value())
        return;

    const float room_pos_x = graph_->get_attrib_by_name<pos_x_att>(room_node.value()).value_or(200.f);
    const float room_pos_y = graph_->get_attrib_by_name<pos_y_att>(room_node.value()).value_or(200.f);
    bool inserted_obstacle_node = false;

    for (auto &instance : temporary_obstacles_)
    {
        const auto &state = instance.model.state();
        const std::string node_name = published_obstacle_name(instance.id);

        std::optional<DSR::Node> node_opt;
        if (instance.published_node_id != 0)
            node_opt = graph_->get_node(instance.published_node_id);
        if (!node_opt.has_value())
        {
            instance.published_node_id = 0;
            node_opt = graph_->get_node(node_name);
            if (node_opt.has_value())
                instance.published_node_id = node_opt->id();
        }

        if (!node_opt.has_value())
        {
            DSR::Node obstacle_node = DSR::Node::create<obstacle_node_type>(node_name);
            graph_->add_or_modify_attrib_local<level_att>(obstacle_node, 3);
            graph_->add_or_modify_attrib_local<parent_att>(obstacle_node, graph_state_->room_id);
            graph_->add_or_modify_attrib_local<pos_x_att>(obstacle_node, room_pos_x + 120.f + 18.f * static_cast<float>(instance.id % 6));
            graph_->add_or_modify_attrib_local<pos_y_att>(obstacle_node, room_pos_y + 40.f + 18.f * static_cast<float>(instance.id / 6));
            const auto node_id = graph_->insert_node(obstacle_node);
            if (!node_id.has_value())
                continue;
            instance.published_node_id = node_id.value();
            inserted_obstacle_node = true;
            node_opt = graph_->get_node(node_id.value());
            if (!node_opt.has_value())
                continue;
        }

        auto node = node_opt.value();
        graph_->add_or_modify_attrib_local<width_m_att>(node, state.width_m);
        graph_->add_or_modify_attrib_local<depth_m_att>(node, state.depth_m);
        graph_->add_or_modify_attrib_local<height_m_att>(node, kPublishedObstacleHeightM);
        graph_->add_or_modify_attrib_local<level_att>(node, 3);
        graph_->add_or_modify_attrib_local<parent_att>(node, graph_state_->room_id);
        if (!graph_->update_node(node))
            continue;

        rt_api_->insert_or_assign_edge_RT(room_node.value(),
                                          instance.published_node_id,
                                          {state.center.x(), state.center.y(), 0.5f * kPublishedObstacleHeightM},
                                          {0.f, 0.f, state.yaw_rad},
                                          timestamp_ms);
    }

    if (inserted_obstacle_node && graph_layout_callback_)
        graph_layout_callback_();
}

std::vector<Eigen::Vector2f> ControllerObstacleTracker::read_temporary_obstacle_points(std::uint64_t timestamp_ms,
                                                                                        const ControllerRobotPose &robot_pose,
                                                                                        const Eigen::Vector2f &region_center_room,
                                                                                        float region_radius_m,
                                                                                        bool forward_only)
{
    std::vector<Eigen::Vector2f> candidate_points_room;
    if (!params_)
        return candidate_points_room;

    const auto fused_points_room = read_recent_lidar_points_in_room(timestamp_ms,
                                                                    params_->temporary_obstacle_history_scans);
    if (fused_points_room.empty())
        return candidate_points_room;

    const Eigen::Affine2f robot_from_room = robot_pose.as_transform().inverse();
    candidate_points_room.reserve(fused_points_room.size());

    const float max_front_distance = std::max(0.2f, params_->temporary_obstacle_front_distance_m);
    const float half_width = std::max(0.1f, params_->temporary_obstacle_half_width_m);
    const float cluster_margin = std::max(0.f, params_->temporary_obstacle_cluster_margin_m);
    const float known_obstacle_explanation_margin_m = std::clamp(0.5f * std::max(0.f, params_->temporary_obstacle_padding_m),
                                                                 0.05f,
                                                                 0.14f);
    const float search_radius = std::max(region_radius_m + cluster_margin, cluster_margin + 0.1f);
    const float max_robot_distance = max_front_distance + search_radius + 0.25f;

    for (const auto &point3d_room : fused_points_room)
    {
        const Eigen::Vector2f point_room(point3d_room.x(), point3d_room.y());
        if ((point_room - region_center_room).norm() > search_radius)
            continue;

        const Eigen::Vector2f point_robot = robot_from_room * point_room;
        if (forward_only)
        {
            if (point_robot.y() <= params_->clearance_m * 0.5f)
                continue;
            if (point_robot.y() > max_front_distance)
                continue;
            if (std::abs(point_robot.x()) > half_width)
                continue;
        }
        else if (point_robot.norm() > max_robot_distance)
        {
            continue;
        }

        const bool explained_by_known_obstacle = std::any_of(known_graph_obstacles_.cbegin(),
                                                             known_graph_obstacles_.cend(),
                                                             [&point_room, known_obstacle_explanation_margin_m](const auto &state)
                                                             {
                                                                 return point_explained_by_obstacle(state.state,
                                                                                                    point_room,
                                                                                                    known_obstacle_explanation_margin_m);
                                                             });
        if (explained_by_known_obstacle)
            continue;

        candidate_points_room.push_back(point_room);
    }

    return candidate_points_room;
}

std::optional<ControllerObstacleObservation> ControllerObstacleTracker::build_temporary_obstacle_observation(std::uint64_t timestamp_ms,
                                                                                                              const ControllerRobotPose &robot_pose,
                                                                                                              const Eigen::Vector2f &region_center_room,
                                                                                                              float region_radius_m,
                                                                                                              bool forward_only)
{
    if (!params_)
        return std::nullopt;

    const auto candidate_points_room = read_temporary_obstacle_points(timestamp_ms,
                                                                      robot_pose,
                                                                      region_center_room,
                                                                      region_radius_m,
                                                                      forward_only);
    if (static_cast<int>(candidate_points_room.size()) < std::max(3, params_->temporary_obstacle_min_points))
        return std::nullopt;

    return ControllerObstacleModel::make_observation(candidate_points_room,
                                                     robot_pose.pos,
                                                     std::max(0.f, params_->temporary_obstacle_padding_m),
                                                     std::max(0.f, params_->temporary_obstacle_occlusion_depth_m));
}

Eigen::Vector2f ControllerObstacleTracker::to_local_point(const ControllerObstacleState &state, const Eigen::Vector2f &point)
{
    return Eigen::Rotation2Df(-state.yaw_rad) * (point - state.center);
}

float ControllerObstacleTracker::obstacle_sdf(const ControllerObstacleState &state, const Eigen::Vector2f &point)
{
    const Eigen::Vector2f local_point = to_local_point(state, point);
    const Eigen::Vector2f half_extents(state.width_m * 0.5f, state.depth_m * 0.5f);
    const Eigen::Vector2f d = local_point.cwiseAbs() - half_extents;
    const Eigen::Vector2f outside = d.cwiseMax(0.f);
    return outside.norm() + std::min(std::max(d.x(), d.y()), 0.f);
}

bool ControllerObstacleTracker::point_explained_by_obstacle(const ControllerObstacleState &state,
                                                            const Eigen::Vector2f &point,
                                                            float margin_m)
{
    return obstacle_sdf(state, point) <= std::max(0.f, margin_m);
}

bool ControllerObstacleTracker::obstacles_overlap(const ControllerObstacleState &lhs,
                                                  const ControllerObstacleState &rhs,
                                                  float margin_m)
{
    const float margin = std::max(0.f, margin_m);
    if (point_explained_by_obstacle(lhs, rhs.center, margin)
        || point_explained_by_obstacle(rhs, lhs.center, margin))
        return true;

    const auto lhs_polygon = ControllerObstacleModel::polygon_from_state(lhs);
    for (const auto &vertex : lhs_polygon)
    {
        if (point_explained_by_obstacle(rhs, vertex, margin))
            return true;
    }

    const auto rhs_polygon = ControllerObstacleModel::polygon_from_state(rhs);
    for (const auto &vertex : rhs_polygon)
    {
        if (point_explained_by_obstacle(lhs, vertex, margin))
            return true;
    }

    return false;
}

float ControllerObstacleTracker::obstacle_distance_shape_metric(const ControllerObstacleState &reference,
                                                                const ControllerObstacleState &candidate,
                                                                bool candidate_has_shape)
{
    const float ref_diag = std::hypot(reference.width_m, reference.depth_m);
    const float cand_diag = std::hypot(std::max(candidate.width_m, 0.f), std::max(candidate.depth_m, 0.f));
    const float center_scale = std::max(0.25f, 0.5f * (ref_diag + std::max(cand_diag, 0.25f)));
    const float center_term = (reference.center - candidate.center).norm() / center_scale;
    const float yaw_delta = std::abs(std::atan2(std::sin(reference.yaw_rad - candidate.yaw_rad),
                                                std::cos(reference.yaw_rad - candidate.yaw_rad)));

    float metric = center_term + 0.35f * (yaw_delta / static_cast<float>(M_PI));
    if (candidate_has_shape)
    {
        metric += 0.5f * std::abs(reference.width_m - candidate.width_m) / std::max(reference.width_m, 0.1f);
        metric += 0.5f * std::abs(reference.depth_m - candidate.depth_m) / std::max(reference.depth_m, 0.1f);
    }
    return metric;
}

Eigen::Vector2f ControllerObstacleTracker::to_room_point(const ControllerObstacleState &state, const Eigen::Vector2f &point)
{
    return state.center + Eigen::Rotation2Df(state.yaw_rad) * point;
}

void ControllerObstacleTracker::retire_temporary_obstacles_explained_by_graph()
{
    if (temporary_obstacles_.empty() || known_graph_obstacles_.empty())
        return;

    const float overlap_margin_m = std::max(0.08f,
                                            params_ ? 0.5f * std::max(params_->temporary_obstacle_padding_m, 0.f)
                                                    : 0.08f);

    auto first_retired = std::remove_if(temporary_obstacles_.begin(),
                                        temporary_obstacles_.end(),
                                        [this, overlap_margin_m](const auto &instance)
                                        {
                                            const auto &state = instance.model.state();
                                            const bool explained = std::any_of(known_graph_obstacles_.cbegin(),
                                                                               known_graph_obstacles_.cend(),
                                                                               [&state, overlap_margin_m](const auto &known_state)
                                                                               {
                                                                                   return obstacles_overlap(known_state.state,
                                                                                                            state,
                                                                                                            overlap_margin_m);
                                                                               });
                                            if (explained)
                                                delete_published_obstacle_node(instance);
                                            return explained;
                                        });
    temporary_obstacles_.erase(first_retired, temporary_obstacles_.end());
}

ControllerObstacleObservation ControllerObstacleTracker::augment_with_remembered_points(const ControllerObstacleObservation &observation,
                                                                                         const TemporaryObstacleInstance &instance) const
{
    ControllerObstacleObservation augmented = observation;
    augmented.points.reserve(observation.points.size() + instance.remembered_points.size());
    augmented.weights.reserve(augmented.points.capacity());

    if (augmented.weights.size() < observation.points.size())
        augmented.weights.resize(observation.points.size(), 1.f);

    const auto &state = instance.model.state();
    for (const auto &remembered_point : instance.remembered_points)
    {
        augmented.points.push_back(to_room_point(state, remembered_point.local_point));
        augmented.weights.push_back(1.f / (1.f + 4.f * remembered_point.rfe));
    }

    recompute_observation_summary(augmented,
                                  params_ ? std::max(0.f, params_->temporary_obstacle_padding_m) : 0.f,
                                  params_ ? std::max(0.f, params_->temporary_obstacle_occlusion_depth_m) : 0.f);
    return augmented;
}

void ControllerObstacleTracker::update_remembered_points(TemporaryObstacleInstance &instance,
                                                         const ControllerObstacleObservation &observation)
{
    auto &remembered_points = instance.remembered_points;
    const auto &state = instance.model.state();

    for (auto &remembered_point : remembered_points)
    {
        const auto slot = classify_remembered_edge_slot(state, remembered_point.local_point);
        if (!slot.has_value())
        {
            remembered_point.rfe = kRememberedRfeMax + 1.f;
            continue;
        }

        const Eigen::Vector2f room_point = to_room_point(state, remembered_point.local_point);
        const float sdf = obstacle_sdf(state, room_point);
        remembered_point.rfe = kRememberedRfeAlpha * remembered_point.rfe + sdf * sdf;
    }

    std::array<std::vector<std::pair<RememberedPoint, float>>, kRememberedEdgeCount * kRememberedSlotsPerEdge> bins;
    for (const auto &remembered_point : remembered_points)
    {
        if (remembered_point.rfe > kRememberedRfeMax)
            continue;

        const auto slot = classify_remembered_edge_slot(state, remembered_point.local_point);
        if (!slot.has_value())
            continue;

        bins[remembered_slot_index(*slot)].emplace_back(remembered_point, remembered_point.rfe);
    }

    for (const auto &point : observation.points)
    {
        const Eigen::Vector2f local_point = to_local_point(state, point);
        const auto slot = classify_remembered_edge_slot(state, local_point);
        if (!slot.has_value())
            continue;

        const RememberedPoint candidate{.local_point = local_point, .rfe = 0.f};
        bins[remembered_slot_index(*slot)].emplace_back(candidate, slot->distance_to_edge);
    }

    remembered_points.clear();
    for (auto &bin : bins)
    {
        std::sort(bin.begin(), bin.end(), [](const auto &lhs, const auto &rhs)
        {
            return lhs.second < rhs.second;
        });

        const int keep = std::min<int>(kRememberedPointsPerSlot, bin.size());
        for (int index = 0; index < keep; ++index)
            remembered_points.push_back(bin[index].first);
    }
}

float ControllerObstacleTracker::size_hardening_evidence(const TemporaryObstacleInstance &instance) const
{
    std::array<float, kRememberedEdgeCount * kRememberedSlotsPerEdge> slot_scores{};
    for (const auto &remembered_point : instance.remembered_points)
    {
        const auto slot = classify_remembered_edge_slot(instance.model.state(), remembered_point.local_point);
        if (!slot.has_value())
            continue;

        const float score = 1.f - std::clamp(remembered_point.rfe / std::max(1e-4f, kRememberedRfeMax), 0.f, 1.f);
        slot_scores[remembered_slot_index(*slot)] = std::max(slot_scores[remembered_slot_index(*slot)], score);
    }

    float slot_score_sum = 0.f;
    for (float score : slot_scores)
        slot_score_sum += score;

    return slot_score_sum / static_cast<float>(slot_scores.size());
}

bool ControllerObstacleTracker::has_compelling_absence_evidence(std::uint64_t timestamp_ms,
                                                                const ControllerRobotPose &robot_pose,
                                                                const TemporaryObstacleInstance &instance)
{
    if (!params_ || instance.missed_updates < kRemovalEvidenceMinMissedUpdates)
        return false;

    const auto &state = instance.model.state();
    const Eigen::Vector2f center_robot = robot_pose.as_transform().inverse() * state.center;
    if (center_robot.y() <= params_->clearance_m * 0.25f)
        return false;

    const auto points_room = read_temporary_obstacle_points(timestamp_ms,
                                                            robot_pose,
                                                            state.center,
                                                            instance.model.association_radius(),
                                                            false);
    if (points_room.size() < static_cast<std::size_t>(std::max(3, params_->temporary_obstacle_min_points / 2)))
        return false;

    const Eigen::Vector2f view_dir_room = state.center - robot_pose.pos;
    const float view_norm = view_dir_room.norm();
    if (view_norm < 1e-4f)
        return false;

    const float visibility_scale = distance_visibility_scale(params_, robot_pose, state);
    const int required_penetrating_points = std::max(kRemovalEvidenceMinPenetratingPoints,
                                                     static_cast<int>(std::ceil(static_cast<float>(kRemovalEvidenceMinPenetratingPoints)
                                                                                 / visibility_scale)));

    const Eigen::Vector2f away_local = Eigen::Rotation2Df(-state.yaw_rad) * (view_dir_room / view_norm);
    const bool dominant_x = std::abs(away_local.x()) >= std::abs(away_local.y());
    const float away_sign = dominant_x ? (away_local.x() >= 0.f ? 1.f : -1.f)
                                      : (away_local.y() >= 0.f ? 1.f : -1.f);
    const float half_width = state.width_m * 0.5f;
    const float half_depth = state.depth_m * 0.5f;

    int support_points = 0;
    int penetrating_points = 0;
    for (const auto &point_room : points_room)
    {
        const Eigen::Vector2f local = to_local_point(state, point_room);
        const float sdf = obstacle_sdf(state, point_room);
        if (std::abs(sdf) < kRemovalSupportDistanceThreshold)
            ++support_points;

        if (dominant_x)
        {
            if (std::abs(local.y()) > half_depth + kRemovalCrossAxisMargin)
                continue;
            const float near_face = -away_sign * half_width;
            const float penetration = away_sign * (local.x() - near_face);
            if (penetration > kRemovalPenetrationMargin)
                ++penetrating_points;
        }
        else
        {
            if (std::abs(local.x()) > half_width + kRemovalCrossAxisMargin)
                continue;
            const float near_face = -away_sign * half_depth;
            const float penetration = away_sign * (local.y() - near_face);
            if (penetration > kRemovalPenetrationMargin)
                ++penetrating_points;
        }
    }

    return support_points <= 1 && penetrating_points >= required_penetrating_points;
}

void ControllerObstacleTracker::prune_expired_temporary_obstacles(std::uint64_t timestamp_ms)
{
    Q_UNUSED(timestamp_ms)
    if (!params_)
        return;

    auto first_removed = std::remove_if(temporary_obstacles_.begin(),
                                        temporary_obstacles_.end(),
                                        [this](const auto &instance)
                                        {
                                            return instance.existence_log_odds
                                                < params_->temporary_obstacle_existence_remove_threshold_log_odds;
                                        });
    for (auto it = first_removed; it != temporary_obstacles_.end(); ++it)
        delete_published_obstacle_node(*it);
    temporary_obstacles_.erase(first_removed, temporary_obstacles_.end());
}

std::optional<std::size_t> ControllerObstacleTracker::match_temporary_obstacle(const ControllerObstacleObservation &observation) const
{
    std::optional<std::size_t> best_index;
    float best_distance = std::numeric_limits<float>::max();
    const float observation_match_radius = 0.5f * std::sqrt(observation.width_m * observation.width_m
                                                           + observation.depth_m * observation.depth_m);

    for (std::size_t index = 0; index < temporary_obstacles_.size(); ++index)
    {
        const auto &instance = temporary_obstacles_[index];
        const float distance = (instance.model.state().center - observation.centroid).norm();
        const float max_distance = std::max(instance.model.association_radius(), observation_match_radius + 0.2f);
        if (distance <= max_distance && distance < best_distance)
        {
            best_distance = distance;
            best_index = index;
        }
    }

    return best_index;
}

ControllerPolygons ControllerObstacleTracker::read_obstacle_polygons(std::uint64_t timestamp_ms)
{
    ControllerPolygons obstacles;
    known_graph_obstacles_.clear();
    display_obstacle_polygons_.clear();
    std::ostringstream report;
    std::ostringstream object_report;
    if (!graph_ || !inner_eigen_api_ || !graph_state_ || graph_state_->room_name.empty())
    {
        if (obstacle_debug_report_ != "Obstacle debug: graph or inner api not ready")
        {
            obstacle_debug_report_ = "Obstacle debug: graph or inner api not ready";
            std::print("{}\n", obstacle_debug_report_);
            std::fflush(stdout);
        }
        return obstacles;
    }

    auto room_node = graph_state_->room_id != 0 ? graph_->get_node(graph_state_->room_id) : std::optional<DSR::Node>{};

    const auto time_query = params_ && params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                               : DSR::RT_API::TimeQuery::Nearest;
    auto is_robot_body_node = [this](const auto &node)
    {
        return graph_state_ && (node.id() == graph_state_->robot_id
            || node.name() == graph_state_->robot_name
            || node.type() == "robot"
            || node.type() == "body"
            || node.name() == "body");
    };

    auto obstacle_nodes = graph_->get_nodes_by_type("object");
    std::unordered_set<std::uint64_t> obstacle_node_ids;
    obstacle_node_ids.reserve(obstacle_nodes.size() + temporary_obstacles_.size());
    for (const auto &node : obstacle_nodes)
        obstacle_node_ids.insert(node.id());

    for (const auto &node : graph_->get_nodes_by_type("obstacle"))
    {
        if (obstacle_node_ids.insert(node.id()).second)
            obstacle_nodes.push_back(node);
    }

    bool using_fallback_nodes = false;
    if (obstacle_nodes.empty())
    {
        using_fallback_nodes = true;
        for (const auto &node : graph_->get_nodes())
        {
            if (is_robot_body_node(node))
                continue;

            const auto width_attr = graph_->get_attrib_by_name<width_m_att>(node);
            const auto depth_attr = graph_->get_attrib_by_name<depth_m_att>(node);
            if (!width_attr.has_value() || !depth_attr.has_value())
                continue;

            const auto translation = inner_eigen_api_->transform(graph_state_->room_name,
                                                                 node.name(),
                                                                 timestamp_ms,
                                                                 "RT",
                                                                 time_query);
            const auto euler = inner_eigen_api_->get_euler_xyz_angles(graph_state_->room_name,
                                                                      node.name(),
                                                                      timestamp_ms,
                                                                      "RT",
                                                                      time_query);
            if (!translation.has_value() || !euler.has_value())
                continue;

            obstacle_nodes.push_back(node);
        }
    }

    report << "Obstacle debug: room='" << graph_state_->room_name << "' nodes=" << obstacle_nodes.size();
    object_report << "Graph objects debug: room='" << graph_state_->room_name << "'";
    if (using_fallback_nodes)
        report << " fallback=width_depth_rt";

    std::vector<bool> adopted_temporary_obstacles(temporary_obstacles_.size(), false);
    auto adopt_temporary_obstacle_into_object = [this, &room_node, timestamp_ms](DSR::Node &node, const ControllerObstacleState &state)
    {
        if (!graph_ || !rt_api_ || !room_node.has_value())
            return false;

        const auto parent_id = room_node->id();

        graph_->add_or_modify_attrib_local<width_m_att>(node, state.width_m);
        graph_->add_or_modify_attrib_local<depth_m_att>(node, state.depth_m);
        graph_->add_or_modify_attrib_local<height_m_att>(node, kPublishedObstacleHeightM);
        graph_->add_or_modify_attrib_local<level_att>(node, 3);
        graph_->add_or_modify_attrib_local<parent_att>(node, parent_id);
        if (!graph_->update_node(node))
            return false;

        rt_api_->insert_or_assign_edge_RT(room_node.value(),
                                          node.id(),
                                          {state.center.x(), state.center.y(), 0.5f * kPublishedObstacleHeightM},
                                          {0.f, 0.f, state.yaw_rad},
                                          timestamp_ms);
        return true;
    };

    for (const auto &node : obstacle_nodes)
    {
        if (is_robot_body_node(node))
        {
            report << " | node='" << node.name() << "' type='" << node.type() << "' skipped=self";
            continue;
        }

        const bool is_tracker_published_obstacle = node.type() == "obstacle"
            && std::any_of(temporary_obstacles_.begin(), temporary_obstacles_.end(), [&node](const auto &instance)
               {
                   return instance.published_node_id != 0 && instance.published_node_id == node.id();
               });
        if (is_tracker_published_obstacle)
        {
            report << " | node='" << node.name() << "' type='" << node.type() << "' skipped=local_tmp";
            continue;
        }

        report << " | node='" << node.name() << "' type='" << node.type() << "'";
        const auto translation = inner_eigen_api_->transform(graph_state_->room_name,
                                                             node.name(),
                                                             timestamp_ms,
                                                             "RT",
                                                             time_query);
        const auto euler = inner_eigen_api_->get_euler_xyz_angles(graph_state_->room_name,
                                                                  node.name(),
                                                                  timestamp_ms,
                                                                  "RT",
                                                                  time_query);
        if (!translation.has_value() || !euler.has_value())
        {
            report << " missing_rt";
            continue;
        }

        const Eigen::Vector2f center(static_cast<float>(translation->x()), static_cast<float>(translation->y()));
        const float yaw = static_cast<float>(euler->z());
        ControllerObstacleState state{.center = center,
                                      .yaw_rad = yaw,
                                      .width_m = 0.f,
                                      .depth_m = 0.f};
        const auto width_attr = graph_->get_attrib_by_name<width_m_att>(node);
        const auto depth_attr = graph_->get_attrib_by_name<depth_m_att>(node);
        const bool has_shape = width_attr.has_value() && depth_attr.has_value()
            && std::isfinite(width_attr.value()) && std::isfinite(depth_attr.value())
            && width_attr.value() > 0.f && depth_attr.value() > 0.f;
        if (has_shape)
        {
            state.width_m = width_attr.value();
            state.depth_m = depth_attr.value();
            report << " size=(" << state.width_m << "," << state.depth_m << ")";
        }

        if (node.type() == "object")
        {
            std::optional<std::size_t> best_match;
            float best_metric = std::numeric_limits<float>::max();
            for (std::size_t index = 0; index < temporary_obstacles_.size(); ++index)
            {
                if (adopted_temporary_obstacles[index])
                    continue;

                const auto &candidate_state = temporary_obstacles_[index].model.state();
                const bool close_enough = point_explained_by_obstacle(candidate_state, state.center, 0.2f)
                    || (candidate_state.center - state.center).norm() <= std::max(0.6f, std::hypot(candidate_state.width_m, candidate_state.depth_m));
                if (!close_enough)
                    continue;

                const float metric = obstacle_distance_shape_metric(candidate_state, state, has_shape);
                if (metric < best_metric)
                {
                    best_metric = metric;
                    best_match = index;
                }
            }

            if (best_match.has_value() && best_metric <= 1.6f)
            {
                const auto &matched_state = temporary_obstacles_[best_match.value()].model.state();
                auto updated_node = node;
                if (adopt_temporary_obstacle_into_object(updated_node, matched_state))
                {
                    state = matched_state;
                    adopted_temporary_obstacles[best_match.value()] = true;
                    std::print("Obstacle adoption trace: object='{}' id={} <- temp='{}' temp_id={} metric={:.3f} center=({:.3f},{:.3f}) size=({:.3f},{:.3f}) yaw={:.3f}\n",
                               node.name(),
                               node.id(),
                               published_obstacle_name(temporary_obstacles_[best_match.value()].id),
                               temporary_obstacles_[best_match.value()].id,
                               best_metric,
                               state.center.x(),
                               state.center.y(),
                               state.width_m,
                               state.depth_m,
                               state.yaw_rad);
                    std::fflush(stdout);
                    report << " adopted_tmp='" << published_obstacle_name(temporary_obstacles_[best_match.value()].id)
                           << "' metric=" << best_metric;
                }
            }

            if (state.width_m <= 0.f || state.depth_m <= 0.f)
            {
                report << (has_shape ? " invalid_size" : " missing_attrs");
                continue;
            }
        }
        else if (!has_shape)
        {
            report << " missing_attrs";
            continue;
        }

        if (state.width_m <= 0.f || state.depth_m <= 0.f)
        {
            report << " invalid_size";
            continue;
        }

        auto polygon = ControllerObstacleModel::polygon_from_state(state);
        report << " center=(" << state.center.x() << "," << state.center.y() << ") yaw=" << state.yaw_rad;
        if (!polygon.empty())
            report << " first_vertex=(" << polygon.front().x() << "," << polygon.front().y() << ")";
        const ControllerObstacleKind kind = node.type() == "object"
            ? ControllerObstacleKind::Object
            : ControllerObstacleKind::Obstacle;
        if (kind == ControllerObstacleKind::Object)
        {
            object_report << " | object='" << node.name() << "' id=" << node.id()
                          << " pos=(" << state.center.x() << "," << state.center.y() << ")"
                          << " size=(" << state.width_m << "," << state.depth_m << ")"
                          << " yaw=" << state.yaw_rad;
        }
        known_graph_obstacles_.push_back(GraphObstacleRecord{.node_id = node.id(),
                                                             .node_name = node.name(),
                                                             .state = state,
                                                             .kind = kind});
        display_obstacle_polygons_.push_back(ControllerObstacleVisual{.polygon = polygon,
                                                                      .kind = kind});
        obstacles.push_back(std::move(polygon));
    }

    if (!adopted_temporary_obstacles.empty())
    {
        std::vector<TemporaryObstacleInstance> surviving_obstacles;
        surviving_obstacles.reserve(temporary_obstacles_.size());
        for (std::size_t index = 0; index < temporary_obstacles_.size(); ++index)
        {
            if (adopted_temporary_obstacles[index])
            {
                delete_published_obstacle_node(temporary_obstacles_[index]);
                continue;
            }
            surviving_obstacles.push_back(std::move(temporary_obstacles_[index]));
        }
        temporary_obstacles_ = std::move(surviving_obstacles);
    }

    report << " | drawn=" << obstacles.size();
    if (const auto report_str = report.str(); report_str != obstacle_debug_report_)
    {
        obstacle_debug_report_ = report_str;
        std::print("{}\n", obstacle_debug_report_);
        std::fflush(stdout);
    }
    if (const auto object_report_str = object_report.str(); object_report_str != graph_object_debug_report_)
    {
        graph_object_debug_report_ = object_report_str;
        std::print("{}\n", graph_object_debug_report_);
        std::fflush(stdout);
    }

    return obstacles;
}

void ControllerObstacleTracker::update_active_obstacle_polygons(std::uint64_t timestamp_ms,
                                                                rc::TrajectoryController &path_controller)
{
    prune_expired_temporary_obstacles(timestamp_ms);
    obstacle_polygons_ = read_obstacle_polygons(timestamp_ms);
    retire_temporary_obstacles_explained_by_graph();
    std::ostringstream current_obstacles_report;
    current_obstacles_report << "Current obstacles debug:";
    for (const auto &record : known_graph_obstacles_)
    {
        current_obstacles_report << " | kind="
                                 << (record.kind == ControllerObstacleKind::Object ? "object" : "obstacle")
                                 << " name='" << record.node_name << "' id=" << record.node_id
                                 << " pos=(" << record.state.center.x() << "," << record.state.center.y() << ")"
                                 << " size=(" << record.state.width_m << "," << record.state.depth_m << ")"
                                 << " yaw=" << record.state.yaw_rad;
    }
    for (const auto &instance : temporary_obstacles_)
    {
        const auto &state = instance.model.state();
        current_obstacles_report << " | kind=temporary"
                                 << " name='" << published_obstacle_name(instance.id) << "' id=" << instance.id
                                 << " pos=(" << state.center.x() << "," << state.center.y() << ")"
                                 << " size=(" << state.width_m << "," << state.depth_m << ")"
                                 << " yaw=" << state.yaw_rad;
        const auto polygon = instance.model.polygon();
        obstacle_polygons_.push_back(polygon);
        display_obstacle_polygons_.push_back(ControllerObstacleVisual{.polygon = polygon,
                                                                      .kind = ControllerObstacleKind::Temporary});
    }
    if (const auto report = current_obstacles_report.str(); report != current_obstacles_debug_report_)
    {
        current_obstacles_debug_report_ = report;
        std::print("{}\n", current_obstacles_debug_report_);
        std::fflush(stdout);
    }
    sync_temporary_obstacles_to_dsr(timestamp_ms);
    path_controller.set_static_obstacles(obstacle_polygons_);
}

void ControllerObstacleTracker::refresh_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                                                 const ControllerRobotPose &robot_pose,
                                                                 rc::TrajectoryController &path_controller)
{
    if (!params_ || temporary_obstacles_.empty())
        return;

    prune_expired_temporary_obstacles(timestamp_ms);
    for (auto &instance : temporary_obstacles_)
    {
        const auto observation = build_temporary_obstacle_observation(timestamp_ms,
                                                                      robot_pose,
                                                                      instance.model.state().center,
                                                                      instance.model.association_radius(),
                                                                      false);
        if (!observation.has_value()
            || (observation->centroid - instance.model.state().center).norm() > instance.model.association_radius())
        {
            ++instance.missed_updates;
            const float visibility_scale = distance_visibility_scale(params_, robot_pose, instance.model.state());
            if (has_compelling_absence_evidence(timestamp_ms, robot_pose, instance))
                instance.existence_log_odds = std::clamp(instance.existence_log_odds
                                                             - params_->temporary_obstacle_existence_absence_penalty * visibility_scale,
                                                         params_->temporary_obstacle_existence_min_log_odds,
                                                         params_->temporary_obstacle_existence_max_log_odds);
            else
                instance.existence_log_odds = std::clamp(instance.existence_log_odds
                                                             - params_->temporary_obstacle_existence_weak_miss_penalty * visibility_scale,
                                                         params_->temporary_obstacle_existence_min_log_odds,
                                                         params_->temporary_obstacle_existence_max_log_odds);
            continue;
        }

        const auto augmented_observation = augment_with_remembered_points(*observation, instance);
        instance.free_energy = instance.model.update(augmented_observation, size_hardening_evidence(instance));
        update_remembered_points(instance, *observation);
        int support_points = 0;
        for (const auto &point : observation->points)
        {
            if (std::abs(obstacle_sdf(instance.model.state(), point)) < kRemovalSupportDistanceThreshold)
                ++support_points;
        }
        const float positive_delta = params_->temporary_obstacle_existence_observation_bias
                       + params_->temporary_obstacle_existence_support_gain * static_cast<float>(support_points)
                       + params_->temporary_obstacle_existence_remembered_gain * static_cast<float>(instance.remembered_points.size());
        instance.existence_log_odds = std::clamp(instance.existence_log_odds + positive_delta,
                             params_->temporary_obstacle_existence_min_log_odds,
                             params_->temporary_obstacle_existence_max_log_odds);
        instance.last_seen_ms = timestamp_ms;
        instance.expires_at_ms = 0;
        instance.missed_updates = 0;
    }

    update_active_obstacle_polygons(timestamp_ms, path_controller);
}

std::vector<Eigen::Vector3f> ControllerObstacleTracker::read_recent_lidar_points_in_room(std::uint64_t timestamp_ms,
                                                                                          int max_scans)
{
    std::vector<Eigen::Vector3f> points;
    const int scan_count = std::clamp(max_scans, 1, 5);
    const std::uint64_t anchor_timestamp_ms = last_lidar_timestamp_ms_.value_or(timestamp_ms);
    const std::uint64_t step_ms = std::clamp(lidar_period_ms_, std::uint64_t{20}, std::uint64_t{250});
    const std::uint64_t max_diff_ms = std::max<std::uint64_t>(20ULL, step_ms / 2);

    for (int scan_index = 0; scan_index < scan_count; ++scan_index)
    {
        const std::uint64_t offset_ms = static_cast<std::uint64_t>(scan_index) * step_ms;
        const std::uint64_t query_timestamp_ms = anchor_timestamp_ms > offset_ms
            ? anchor_timestamp_ms - offset_ms
            : 0ULL;

        const auto [cloud_opt] = lidar_room_buffer_.read(query_timestamp_ms, max_diff_ms);
        if (!cloud_opt.has_value())
            continue;

        const auto &[xs_room, ys_room, zs_room] = cloud_opt.value();
        const std::size_t count = std::min({xs_room.size(), ys_room.size(), zs_room.size()});
        for (std::size_t index = 0; index < count; ++index)
        {
            const float x_room = xs_room[index];
            const float y_room = ys_room[index];
            const float z_room = zs_room[index];
            if (!std::isfinite(x_room) || !std::isfinite(y_room) || !std::isfinite(z_room))
                continue;
            points.emplace_back(x_room, y_room, z_room);
        }
    }

    return points;
}

bool ControllerObstacleTracker::create_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                                                const ControllerRobotPose &robot_pose,
                                                                const Eigen::Vector2f &blockage_center_room,
                                                                float blockage_radius_m,
                                                                rc::TrajectoryController &path_controller)
{
    if (!params_)
        return false;

    prune_expired_temporary_obstacles(timestamp_ms);

    const float cluster_margin = std::max(0.f, params_->temporary_obstacle_cluster_margin_m);
    const auto observation = build_temporary_obstacle_observation(timestamp_ms,
                                                                  robot_pose,
                                                                  blockage_center_room,
                                                                  std::max(blockage_radius_m + cluster_margin, cluster_margin),
                                                                  true);
    if (!observation.has_value())
        return false;

    if (const auto matched_index = match_temporary_obstacle(*observation); matched_index.has_value())
    {
        auto &instance = temporary_obstacles_[matched_index.value()];
        const auto augmented_observation = augment_with_remembered_points(*observation, instance);
        instance.free_energy = instance.model.update(augmented_observation, size_hardening_evidence(instance));
        update_remembered_points(instance, *observation);
        int support_points = 0;
        for (const auto &point : observation->points)
        {
            if (std::abs(obstacle_sdf(instance.model.state(), point)) < kRemovalSupportDistanceThreshold)
                ++support_points;
        }
        const float positive_delta = params_->temporary_obstacle_existence_observation_bias
                       + params_->temporary_obstacle_existence_support_gain * static_cast<float>(support_points)
                       + params_->temporary_obstacle_existence_remembered_gain * static_cast<float>(instance.remembered_points.size());
        instance.existence_log_odds = std::clamp(instance.existence_log_odds + positive_delta,
                             params_->temporary_obstacle_existence_min_log_odds,
                             params_->temporary_obstacle_existence_max_log_odds);
        instance.last_seen_ms = timestamp_ms;
        instance.expires_at_ms = 0;
        instance.missed_updates = 0;
    }
    else
    {
        ControllerObstacleState initial_state{.center = estimate_initial_obstacle_center(*observation),
                                              .yaw_rad = observation->yaw_rad,
                                              .width_m = observation->width_m,
                                              .depth_m = observation->depth_m};
        TemporaryObstacleInstance instance;
        instance.id = next_temporary_obstacle_id_++;
        instance.model = ControllerObstacleModel(initial_state, make_model_params());
        instance.free_energy = instance.model.update(*observation, 0.f);
        update_remembered_points(instance, *observation);
        instance.existence_log_odds = params_->temporary_obstacle_existence_init_log_odds;
        instance.last_seen_ms = timestamp_ms;
        instance.expires_at_ms = 0;
        temporary_obstacles_.push_back(std::move(instance));
    }

    update_active_obstacle_polygons(timestamp_ms, path_controller);
    return true;
}

ControllerPolygon ControllerObstacleTracker::make_obstacle_polygon(const Eigen::Vector2f &center,
                                                                   float yaw,
                                                                   float width_m,
                                                                   float depth_m) const
{
    const float half_width = width_m * 0.5f;
    const float half_depth = depth_m * 0.5f;
    const Eigen::Rotation2Df rotation(yaw);

    ControllerPolygon polygon;
    polygon.reserve(4);
    for (const Eigen::Vector2f &corner : {Eigen::Vector2f(-half_width, -half_depth),
                                          Eigen::Vector2f(half_width, -half_depth),
                                          Eigen::Vector2f(half_width, half_depth),
                                          Eigen::Vector2f(-half_width, half_depth)})
        polygon.push_back(center + rotation * corner);

    return polygon;
}

bool ControllerObstacleTracker::handle_lidar_points(const std::string &lidar_node_name,
                                                    std::vector<float> xs,
                                                    std::vector<float> ys,
                                                    std::vector<float> zs,
                                                    std::uint64_t timestamp_ms)
{
    if (!params_ || !inner_eigen_api_ || !graph_state_ || !graph_state_->ready())
        return false;

    // Dedup by source timestamp: when the laser node is polled (DSR fallback), the same
    // scan would otherwise be reprocessed every cycle (and read as "fresh", defeating the
    // stall watchdog). A 0 timestamp can't be deduped, so it always passes.
    if (timestamp_ms != 0 && last_lidar_timestamp_ms_.has_value()
        && timestamp_ms <= last_lidar_timestamp_ms_.value())
        return false;

    if (timestamp_ms > 0 && last_lidar_timestamp_ms_.has_value() && timestamp_ms > last_lidar_timestamp_ms_.value())
    {
        const std::uint64_t diff_ms = timestamp_ms - last_lidar_timestamp_ms_.value();
        if (diff_ms >= 20 && diff_ms <= 500)
        {
            const double blended_period_ms = 0.7 * static_cast<double>(lidar_period_ms_)
                                           + 0.3 * static_cast<double>(diff_ms);
            lidar_period_ms_ = static_cast<std::uint64_t>(std::clamp(blended_period_ms, 20.0, 500.0));
        }
    }
    last_lidar_timestamp_ms_ = timestamp_ms;

    const auto interp = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                : DSR::RT_API::TimeQuery::Nearest;
    const auto room_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                             lidar_node_name,
                                                                             timestamp_ms,
                                                                             "RT",
                                                                             interp);
    const auto robot_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_->robot_name,
                                                                               lidar_node_name,
                                                                               timestamp_ms,
                                                                               "RT",
                                                                               interp);
    if (!room_from_lidar.has_value() || !robot_from_lidar.has_value())
        return false;

    const std::size_t raw_count = std::min({xs.size(), ys.size(), zs.size()});
    if (raw_count == 0)
        return false;

    const auto room_from_lidar_matrix = room_from_lidar->matrix();
    const auto robot_from_lidar_matrix = robot_from_lidar->matrix();
    const bool robot_from_lidar_is_identity = robot_from_lidar_matrix.isApprox(Eigen::Matrix4d::Identity(), 1e-5);
    lidar_room_buffer_.put<0>(
        rc::RawLidarPointVectors{
            .xs = std::move(xs),
            .ys = std::move(ys),
            .zs = std::move(zs)},
        timestamp_ms,
        [room_from_lidar_matrix, robot_from_lidar_matrix, robot_from_lidar_is_identity, raw_count](rc::RawLidarPointVectors &&raw_points, rc::LidarPointVectors &room_points)
        {
            auto &[room_xs, room_ys, room_zs] = room_points;
            const auto &xs_in = raw_points.xs;
            const auto &ys_in = raw_points.ys;
            const auto &zs_in = raw_points.zs;
            const std::size_t count = std::min({raw_count, xs_in.size(), ys_in.size(), zs_in.size()});
            if (count == 0)
                return;

            const double m00 = room_from_lidar_matrix(0, 0);
            const double m01 = room_from_lidar_matrix(0, 1);
            const double m02 = room_from_lidar_matrix(0, 2);
            const double m03 = room_from_lidar_matrix(0, 3);
            const double m10 = room_from_lidar_matrix(1, 0);
            const double m11 = room_from_lidar_matrix(1, 1);
            const double m12 = room_from_lidar_matrix(1, 2);
            const double m13 = room_from_lidar_matrix(1, 3);
            const double m20 = room_from_lidar_matrix(2, 0);
            const double m21 = room_from_lidar_matrix(2, 1);
            const double m22 = room_from_lidar_matrix(2, 2);
            const double m23 = room_from_lidar_matrix(2, 3);
            const double r20 = robot_from_lidar_matrix(2, 0);
            const double r21 = robot_from_lidar_matrix(2, 1);
            const double r22 = robot_from_lidar_matrix(2, 2);
            const double r23 = robot_from_lidar_matrix(2, 3);

            room_xs.reserve(count);
            room_ys.reserve(count);
            room_zs.reserve(count);

            for (std::size_t index = 0; index < count; ++index)
            {
                const float x = xs_in[index];
                const float y = ys_in[index];
                const float z = zs_in[index];
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                    continue;

                const float robot_z = robot_from_lidar_is_identity
                    ? z
                    : static_cast<float>(r20 * x + r21 * y + r22 * z + r23);
                if (robot_z < kTemporaryObstacleMinHeightAboveFloorM || robot_z > kTemporaryObstacleMaxHeightAboveFloorM)
                    continue;

                const float room_x = static_cast<float>(m00 * x + m01 * y + m02 * z + m03);
                const float room_y = static_cast<float>(m10 * x + m11 * y + m12 * z + m13);
                const float room_z = static_cast<float>(m20 * x + m21 * y + m22 * z + m23);

                room_xs.push_back(room_x);
                room_ys.push_back(room_y);
                room_zs.push_back(room_z);
            }
        });
    return true;
}