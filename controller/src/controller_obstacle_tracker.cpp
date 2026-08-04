#include "controller_obstacle_tracker.h"

#include "../../common/robot_footprint/robot_footprint.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <print>
#include <sstream>
#include <unordered_map>
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

std::string ControllerObstacleTracker::object_label_prefix(const std::string &object_type)
{
    if (object_type == "table") return "t";
    if (object_type == "chair") return "c";
    if (object_type == "cylinder" || object_type == "bottle") return "b";
    if (!object_type.empty())
        return std::string(1, static_cast<char>(std::tolower(object_type.front())));
    return "x";
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
    virtual_obstacles_.clear();   // local-only, nothing to delete from DSR
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
    // Returns inside the robot's own footprint are self/body/floor-ring hits, not obstacles — and any
    // real contact is already inside the footprint anyway. Drop them so we never fit an obstacle glued
    // to the body. This is the FOOTPRINT, tested at each return's own bearing: the old code used a 0.50 m
    // disc (clearance_m), which deleted every return in 0.79 m² around the robot — 3.4× the robot's actual
    // area — including the obstacles closest to it, the ones that matter most.
    static const rc::RobotFootprint footprint = rc::RobotFootprint::shadow();

    for (const auto &point3d_room : fused_points_room)
    {
        const Eigen::Vector2f point_room(point3d_room.x(), point3d_room.y());
        if ((point_room - region_center_room).norm() > search_radius)
            continue;
        const Eigen::Vector2f from_robot = point_room - robot_pose.pos;
        if (from_robot.norm() < footprint.support_radius_yaw(robot_pose.theta, from_robot))
            continue;

        const Eigen::Vector2f point_robot = robot_from_room * point_room;
        if (forward_only)
        {
            // Must be genuinely AHEAD of the body, not merely ahead of the origin. point_robot is already in
            // the footprint's own frame (x right, y forward), so this is the front extent directly.
            if (point_robot.y() <= footprint.support_radius(0.f, {0.f, 1.f}))
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
    // The obstacle must be in front of the body for its absence to mean anything — behind it, "we see no
    // points there" is a statement about the sensor, not about the world.
    if (center_robot.y() <= rc::RobotFootprint::shadow().support_radius(0.f, {0.f, 1.f}))
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
    object_label_counts_.clear();   // per-type object tags (t_1/c_1/b_1) restart each rebuild
    obstacle_label_count_ = 0;      // shared o_N counter spans graph "obstacle" nodes + temporaries
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

    // Gather every graph node that represents a real shaped object the robot must avoid AND that can
    // EXPLAIN/retire a controller-published temporary obstacle. Post graph-schema migration every
    // concept agent's node is the generic type "object" (class in the object_subtype attribute), so
    // "object" catches all of them; "obstacle" catches the controller's own published temp obstacles.
    // (The old per-class entries "table"/"cylinder"/"chair" are dead now — no node carries those types.)
    // ★`metaconcept` is DELIBERATELY absent: a level-2 arrangement (ring_metaconcept's dining_set) is a
    // belief about a RELATION among nodes, not a shaped thing. Its published footprint is the ring's
    // extent — the robot drives through the middle of it — and its members are already here as their own
    // `object` nodes, so adopting the rig would double-count them behind one huge phantom box.
    static constexpr std::array<const char *, 2> kObjectTypes = {"object", "obstacle"};
    std::vector<DSR::Node> obstacle_nodes;
    std::unordered_set<std::uint64_t> obstacle_node_ids;
    for (const char *type : kObjectTypes)
        for (const auto &node : graph_->get_nodes_by_type(type))
            if (obstacle_node_ids.insert(node.id()).second)
                obstacle_nodes.push_back(node);

    bool using_fallback_nodes = false;
    if (obstacle_nodes.empty())
    {
        using_fallback_nodes = true;
        for (const auto &node : graph_->get_nodes())
        {
            if (is_robot_body_node(node))
                continue;
            // Level-2 arrangements (ring_metaconcept's dining_set) publish width_m/depth_m as the
            // ring's EXTENT so a viewer can outline it — but nothing occupies that footprint, and
            // the robot must be free to drive through the middle of a dining set. The typed sweep
            // above already excludes them by never asking for "metaconcept"; this width/depth
            // fallback is the one path that could still turn an abstract grouping into an obstacle.
            if (node.type() == "metaconcept")
                continue;
            // ★ THE ROOM IS NOT AN OBSTACLE. The room node carries width_m/depth_m (its extent, for viewers),
            // so this width/depth sweep happily turns the entire room into one room-sized obstacle box — the
            // robot is then inside a solid object covering every cell, no target is ever footprint-feasible,
            // and the 2-D canvas paints the whole floor in the green Object colour instead of leaving it the
            // near-white room fill. Observed live: 27018/27018 cells occupied with a green floor.
            // This fallback only runs when NO typed object/obstacle node exists, which is a perfectly normal
            // state (no concept agent running) — so the failure appears exactly when the world is EMPTIEST,
            // which is the least likely case to be tested and the worst one to fail in.
            // Excluded by identity, not by type, because room_concept's node type has changed before.
            if (graph_state_->room_id != 0 and node.id() == graph_state_->room_id) continue;
            if (node.name() == graph_state_->room_name or node.type() == "room") continue;
            // ...and the same for every other node that carries an extent WITHOUT being a thing to avoid.
            // ★ THE FLOOR IS THE ONE THAT BIT US. room_concept publishes a node named "floor" whose
            // width_m/depth_m are the WHOLE ROOM's extent (room_scene_graph.cpp:887) purely so a viewer can
            // scale its display mesh. This sweep turned it into a room-sized obstacle box: every cell
            // occupied, no target ever footprint-feasible, and the 2-D canvas painted entirely in the green
            // Object colour — the literal "green floor". Walls are the same story (width_m = wall length).
            // These carry an extent because something has to DRAW them, which is not the same as something
            // to drive around: the room boundary is already enforced by the room polygon, and the residual
            // hulls arrive separately, already shaped.
            // Matched by type AND name, because these node types have been renamed before in this codebase
            // and a silent miss here disables navigation completely rather than degrading it.
            static constexpr std::array<const char *, 5> kNotObstacles = {"floor", "wall", "plane", "grid", "room"};
            const auto is_infrastructure = [&](const auto &n)
            {
                for (const char *t : kNotObstacles)
                    if (n.type() == t or n.name() == t or n.name().starts_with(std::string(t) + "_"))
                        return true;
                return n.name() == "residual";
            };
            if (is_infrastructure(node)) continue;

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
        // Anything a concept agent modelled (object/table/cylinder/chair) is an "object"; only
        // an explicit graph "obstacle" node counts as an unmodelled obstacle here.
        const ControllerObstacleKind kind = node.type() == "obstacle"
            ? ControllerObstacleKind::Obstacle
            : ControllerObstacleKind::Object;
        std::string visual_label;
        if (kind == ControllerObstacleKind::Object)
        {
            // Post-migration the node type is generic "object"; the class label lives in the
            // object_subtype attribute (cosmetic log label). Fall back to type() for legacy nodes.
            const auto subtype_attr = graph_->get_attrib_by_name<object_subtype_att>(node);
            const std::string prefix = object_label_prefix(subtype_attr.value_or(node.type()));
            visual_label = prefix + "_" + std::to_string(++object_label_counts_[prefix]);
        }
        else
        {
            visual_label = "o_" + std::to_string(++obstacle_label_count_);
        }
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
                                                                      .kind = kind,
                                                                      .label = visual_label});
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

    // ── residual_concept OCCUPANCY GRID obstacles ──
    // residual_concept no longer publishes `obstacle` nodes; it publishes a `grid` node under room whose
    // `vec_float` attribute encodes the inflated occupancy component HULLS (already half-robot-width inflated):
    //   [ P, (V, x0,y0, …, x_{V-1},y_{V-1}) × P ].
    // Decode each hull and add it to the planner obstacles — so the robot plans around the residual/unmodelled
    // world (from the grid) AND the known object boxes (read above), one unified obstacle list.
    if (const auto g = graph_->get_node("residual"); g.has_value())   // node renamed "grid"→"residual" (type kept)
    {
        if (const auto opt = graph_->get_attrib_by_name<grid_obstacle_hulls_att>(g.value()); opt.has_value())
        {
            const auto &f = opt.value().get();
            std::size_t i = 0;
            const int P = (i < f.size()) ? static_cast<int>(f[i++]) : 0;
            int added = 0;
            for (int p = 0; p < P and i < f.size(); ++p)
            {
                const int V = static_cast<int>(f[i++]);
                ControllerPolygon poly;
                poly.reserve(static_cast<std::size_t>(std::max(0, V)));
                for (int v = 0; v < V and i + 1 < f.size(); ++v) { poly.emplace_back(f[i], f[i + 1]); i += 2; }
                if (poly.size() >= 3)
                {
                    // residual now publishes CONCAVE outlines here (not convex hulls), so these few
                    // polygons are BOTH what the planner avoids AND a faithful, cheap occupied/free
                    // display — no need for the hundreds-of-squares per-cell overlay (UI-heavy).
                    display_obstacle_polygons_.push_back(ControllerObstacleVisual{
                        .polygon = poly, .kind = ControllerObstacleKind::GridOccupancy, .label = {}});
                    obstacles.push_back(std::move(poly));
                    ++added;
                }
            }
            report << " | grid_poly=" << added;
        }
    }

    report << " | drawn=" << obstacles.size();
    // Per-cycle obstacle/object debug prints removed (they embedded the table's wobbling size/center,
    // so the dedup reprinted every frame and clogged the console).

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
                                                                      .kind = ControllerObstacleKind::Temporary,
                                                                      .label = "o_" + std::to_string(++obstacle_label_count_)});
    }
    // Local-only virtual obstacles (stuck recovery): geometric discs the planner/MPPI must avoid,
    // aged out on their TTL. Appended to obstacle_polygons_ here but deliberately left OUT of
    // sync_temporary_obstacles_to_dsr below (that loop only walks temporary_obstacles_), so they
    // never reach the DSR graph.
    std::erase_if(virtual_obstacles_, [&](const VirtualObstacle &vo)
                  { return vo.expires_at_ms != 0 and timestamp_ms >= vo.expires_at_ms; });
    for (const auto &vo : virtual_obstacles_)
    {
        const float side = 2.f * vo.radius_m;
        const auto polygon = make_obstacle_polygon(vo.center, 0.f, side, side);
        obstacle_polygons_.push_back(polygon);
        display_obstacle_polygons_.push_back(ControllerObstacleVisual{.polygon = polygon,
                                                                      .kind = ControllerObstacleKind::Temporary,
                                                                      .label = "v_" + std::to_string(++obstacle_label_count_)});
    }
    // Per-cycle "Current obstacles debug" print removed (wobbling geometry → reprinted every frame).
    sync_temporary_obstacles_to_dsr(timestamp_ms);
    path_controller.set_static_obstacles(obstacle_polygons_);
}

void ControllerObstacleTracker::add_virtual_obstacle(std::uint64_t now_ms,
                                                     const Eigen::Vector2f &center,
                                                     float radius_m)
{
    const float radius = std::max(0.05f, radius_m);
    const std::uint64_t ttl = params_ ? params_->stuck_virtual_obstacle_ttl_ms : 5000;
    const std::uint64_t expires_at = ttl != 0 ? now_ms + ttl : 0;
    // Refresh-or-create: if a virtual disc already covers this spot (repeated escapes at the same
    // wedge), grow it to the larger radius and reset its TTL instead of stacking duplicates.
    for (auto &vo : virtual_obstacles_)
        if ((vo.center - center).norm() <= std::max(vo.radius_m, radius))
        {
            vo.center = center;
            vo.radius_m = std::max(vo.radius_m, radius);
            vo.expires_at_ms = expires_at;
            return;
        }
    virtual_obstacles_.push_back(VirtualObstacle{.center = center, .radius_m = radius, .expires_at_ms = expires_at});
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

    // Hard invariant: never create an obstacle whose footprint intersects the robot body. Such a box is
    // either a self/floor phantom or something we are already in contact with; injected into the planner
    // it just traps every MPPI rollout ("already in collision") and can't be routed around — backing out
    // is the stuck-recovery escape's job, not a planner obstacle's. Require the fitted box surface to sit
    // outside the body — worst-case extent, since obstacle_sdf returns a bare distance with no bearing.
    const ControllerObstacleState candidate{.center = observation->centroid,
                                            .yaw_rad = observation->yaw_rad,
                                            .width_m = observation->width_m,
                                            .depth_m = observation->depth_m};
    if (obstacle_sdf(candidate, robot_pose.pos) < rc::RobotFootprint::shadow().circumscribed_radius())
        return false;

    ingest_obstacle_observation(*observation, timestamp_ms);

    update_active_obstacle_polygons(timestamp_ms, path_controller);
    return true;
}

// Match-or-create a temporary obstacle from an observation and bump its Bayesian existence filter.
// Extracted so the reactive blockage path and the proactive scan share one code path.
void ControllerObstacleTracker::ingest_obstacle_observation(const ControllerObstacleObservation &observation,
                                                            std::uint64_t timestamp_ms)
{
    if (!params_)
        return;

    if (const auto matched_index = match_temporary_obstacle(observation); matched_index.has_value())
    {
        auto &instance = temporary_obstacles_[matched_index.value()];
        const auto augmented_observation = augment_with_remembered_points(observation, instance);
        instance.free_energy = instance.model.update(augmented_observation, size_hardening_evidence(instance));
        update_remembered_points(instance, observation);
        int support_points = 0;
        for (const auto &point : observation.points)
            if (std::abs(obstacle_sdf(instance.model.state(), point)) < kRemovalSupportDistanceThreshold)
                ++support_points;
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
        ControllerObstacleState initial_state{.center = estimate_initial_obstacle_center(observation),
                                              .yaw_rad = observation.yaw_rad,
                                              .width_m = observation.width_m,
                                              .depth_m = observation.depth_m};
        TemporaryObstacleInstance instance;
        instance.id = next_temporary_obstacle_id_++;
        instance.model = ControllerObstacleModel(initial_state, make_model_params());
        instance.free_energy = instance.model.update(observation, 0.f);
        update_remembered_points(instance, observation);
        instance.existence_log_odds = params_->temporary_obstacle_existence_init_log_odds;
        instance.last_seen_ms = timestamp_ms;
        instance.expires_at_ms = 0;
        temporary_obstacles_.push_back(std::move(instance));
    }
}

namespace
{
bool point_in_polygon_2d(const std::vector<Eigen::Vector2f> &poly, const Eigen::Vector2f &p)
{
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        if (((poly[i].y() > p.y()) != (poly[j].y() > p.y())) &&
            (p.x() < (poly[j].x() - poly[i].x()) * (p.y() - poly[i].y()) /
                         (poly[j].y() - poly[i].y() + 1e-12f) + poly[i].x()))
            inside = !inside;
    return inside;
}

float distance_to_polygon_edges(const std::vector<Eigen::Vector2f> &poly, const Eigen::Vector2f &p)
{
    float best = std::numeric_limits<float>::max();
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f a = poly[j], b = poly[i];
        const Eigen::Vector2f ab = b - a;
        const float t = std::clamp((p - a).dot(ab) / std::max(ab.squaredNorm(), 1e-9f), 0.0f, 1.0f);
        best = std::min(best, (p - (a + t * ab)).norm());
    }
    return best;
}
}  // namespace

ControllerObstacleTracker::ObstacleProximityDiag
ControllerObstacleTracker::obstacle_proximity_diag(const Eigen::Vector2f &query_room,
                                                   std::uint64_t now_ms) const
{
    ObstacleProximityDiag diag;
    diag.n_temp = static_cast<int>(temporary_obstacles_.size());
    diag.n_virtual = static_cast<int>(virtual_obstacles_.size());

    float best_temp = std::numeric_limits<float>::infinity();
    for (const auto &instance : temporary_obstacles_)
    {
        const auto polygon = instance.model.polygon();
        if (polygon.size() < 2)
            continue;
        float d = distance_to_polygon_edges(polygon, query_room);
        if (point_in_polygon_2d(polygon, query_room))
            d = -d;   // robot centre INSIDE the obstacle ⇒ hard self-collision trap (negative distance)
        if (d < best_temp)
        {
            best_temp = d;
            diag.near_temp_m = d;
            diag.near_temp_log_odds = instance.existence_log_odds;
            diag.near_temp_missed = instance.missed_updates;
            diag.near_temp_age_ms = now_ms >= instance.last_seen_ms ? now_ms - instance.last_seen_ms : 0;
        }
    }

    float best_virt = std::numeric_limits<float>::infinity();
    for (const auto &vo : virtual_obstacles_)
        best_virt = std::min(best_virt, (query_room - vo.center).norm() - vo.radius_m);  // <0 ⇒ inside disc
    if (std::isfinite(best_virt))
        diag.near_virtual_m = best_virt;

    return diag;
}

ControllerObstacleTracker::NearestObstacleInfo
ControllerObstacleTracker::nearest_obstacle_info(const Eigen::Vector2f &query_room, float robot_theta) const
{
    NearestObstacleInfo info;
    float best = std::numeric_limits<float>::infinity();
    Eigen::Vector2f best_point = Eigen::Vector2f::Zero();

    for (const auto &visual : display_obstacle_polygons_)
    {
        const auto &poly = visual.polygon;
        const std::size_t n = poly.size();
        if (n < 2)
            continue;
        for (std::size_t i = 0; i < n; ++i)   // min point-to-edge distance, same metric as nearest_obst_m
        {
            const Eigen::Vector2f &a = poly[i];
            const Eigen::Vector2f &b = poly[(i + 1) % n];
            const Eigen::Vector2f ab = b - a;
            const float len2 = ab.squaredNorm();
            const float t = len2 > 1e-9f ? std::clamp((query_room - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
            const Eigen::Vector2f closest = a + t * ab;
            const float d = (query_room - closest).norm();
            if (d < best)
            {
                best = d;
                best_point = closest;
                info.kind = visual.kind;
                info.label = visual.label;
                info.inside = point_in_polygon_2d(poly, query_room);
            }
        }
    }

    if (!std::isfinite(best))
        return {};
    info.distance_m = best;
    // Bearing of the offending edge in the ROBOT frame (x right, y forward), so the CSV says which way
    // to look in the viewer rather than just how close it is.
    const Eigen::Vector2f d_room = best_point - query_room;
    const Eigen::Vector2f d_robot = Eigen::Rotation2Df(-robot_theta) * d_room;
    info.bearing_deg = std::atan2(d_robot.x(), d_robot.y()) * 180.f / static_cast<float>(M_PI);
    return info;
}

std::vector<Eigen::Vector2f> ControllerObstacleTracker::read_room_polygon_room_frame() const
{
    std::vector<Eigen::Vector2f> poly;
    if (!graph_ || !graph_state_ || graph_state_->room_id == 0)
        return poly;
    const auto room = graph_->get_node(graph_state_->room_id);
    if (!room.has_value())
        return poly;
    const auto px = graph_->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = graph_->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (!px.has_value() || !py.has_value())
        return poly;
    const auto &xs = px->get();
    const auto &ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    poly.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        poly.emplace_back(xs[i], ys[i]);
    return poly;
}

void ControllerObstacleTracker::scan_for_unmodelled_obstacles(std::uint64_t timestamp_ms,
                                                              const ControllerRobotPose &robot_pose,
                                                              rc::TrajectoryController &path_controller)
{
    if (!params_)
        return;

    // Throttle: the full-scene cluster pass is heavier than the per-instance refresh, so run it at a
    // few Hz, not every control cycle.
    constexpr std::uint64_t kScanPeriodMs = 700;
    if (last_scan_ms_ != 0 && timestamp_ms >= last_scan_ms_ && timestamp_ms - last_scan_ms_ < kScanPeriodMs)
        return;
    last_scan_ms_ = timestamp_ms;

    // Cold start: ensure the known-object set is populated before filtering (otherwise a real table
    // would be re-detected as an obstacle); in steady state it is refreshed every cycle elsewhere.
    if (known_graph_obstacles_.empty())
        (void)read_obstacle_polygons(timestamp_ms);
    prune_expired_temporary_obstacles(timestamp_ms);

    const auto points_room = read_recent_lidar_points_in_room(timestamp_ms, 2);
    if (points_room.empty())
        return;

    // Tunables (local for now; promote to ControllerParams if they need field tuning).
    const float kZMin = params_->unmodelled_scan_min_z_m;   // ignore floor returns
    const float kZMax = params_->unmodelled_scan_max_z_m;   // ignore high ceiling/overhead returns
    constexpr float kWallMarginM = 0.30f;   // drop returns near/beyond the room boundary (walls)
    constexpr float kRobotRadiusM = 0.40f;  // drop the robot's own returns
    constexpr float kCellM = 0.20f;         // grid-cluster cell size
    constexpr std::size_t kMaxObstacles = 24;
    const float explain_margin = std::max(0.08f, 0.5f * std::max(0.0f, params_->temporary_obstacle_padding_m));
    const int   min_cluster_pts = std::max(4, params_->temporary_obstacle_min_points);

    const auto room_poly = read_room_polygon_room_frame();
    const bool have_room = room_poly.size() >= 3;

    // ── Keep only LiDAR returns that NOTHING explains: not floor/ceiling, not the robot, inside the
    //    room and clear of the walls, and not inside any modelled object (table/chair/cylinder/…). ──
    std::vector<Eigen::Vector2f> unexplained;
    unexplained.reserve(points_room.size());
    for (const auto &p : points_room)
    {
        if (p.z() < kZMin || p.z() > kZMax)
            continue;
        const Eigen::Vector2f q(p.x(), p.y());
        if ((q - robot_pose.pos).norm() < kRobotRadiusM)
            continue;
        if (have_room && (!point_in_polygon_2d(room_poly, q) || distance_to_polygon_edges(room_poly, q) < kWallMarginM))
            continue;
        bool explained = false;
        for (const auto &known : known_graph_obstacles_)
            if (point_explained_by_obstacle(known.state, q, explain_margin)) { explained = true; break; }
        if (explained)
            continue;
        unexplained.push_back(q);
    }
    if (static_cast<int>(unexplained.size()) < min_cluster_pts)
        return;

    // ── Grid clustering: bin to cells, connect 8-neighbour occupied cells, gather per cluster. ──
    auto pack = [](std::int32_t ix, std::int32_t iy) -> std::int64_t
    { return (static_cast<std::int64_t>(ix) << 32) | static_cast<std::int64_t>(static_cast<std::uint32_t>(iy)); };
    auto cell_of = [&](const Eigen::Vector2f &q) -> std::int64_t
    { return pack(static_cast<std::int32_t>(std::floor(q.x() / kCellM)), static_cast<std::int32_t>(std::floor(q.y() / kCellM))); };

    std::unordered_map<std::int64_t, std::vector<std::size_t>> cells;
    for (std::size_t i = 0; i < unexplained.size(); ++i)
        cells[cell_of(unexplained[i])].push_back(i);

    std::unordered_set<std::int64_t> visited;
    std::vector<std::vector<Eigen::Vector2f>> clusters;
    for (const auto &[key, idxs] : cells)
    {
        (void)idxs;
        if (visited.count(key))
            continue;
        std::vector<std::int64_t> stack{key};
        visited.insert(key);
        std::vector<Eigen::Vector2f> cluster;
        while (!stack.empty())
        {
            const std::int64_t k = stack.back();
            stack.pop_back();
            const std::int32_t ix = static_cast<std::int32_t>(k >> 32);
            const std::int32_t iy = static_cast<std::int32_t>(static_cast<std::uint32_t>(k & 0xffffffffLL));
            for (const std::size_t pi : cells[k])
                cluster.push_back(unexplained[pi]);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                {
                    if (dx == 0 && dy == 0) continue;
                    const std::int64_t nk = pack(ix + dx, iy + dy);
                    if (cells.count(nk) && !visited.count(nk)) { visited.insert(nk); stack.push_back(nk); }
                }
        }
        if (static_cast<int>(cluster.size()) >= min_cluster_pts)
            clusters.push_back(std::move(cluster));
    }

    // ── Each cluster → an obstacle (match-or-create), capped. ──
    for (auto &cluster : clusters)
    {
        const auto observation = ControllerObstacleModel::make_observation(
            cluster, robot_pose.pos,
            std::max(0.0f, params_->temporary_obstacle_padding_m),
            std::max(0.0f, params_->temporary_obstacle_occlusion_depth_m));
        if (!observation.has_value())
            continue;
        const bool matched = match_temporary_obstacle(*observation).has_value();
        if (!matched && temporary_obstacles_.size() >= kMaxObstacles)
            continue;   // cap NEW obstacles; existing ones still refresh
        ingest_obstacle_observation(*observation, timestamp_ms);
    }

    update_active_obstacle_polygons(timestamp_ms, path_controller);
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

// Is there a room←robot pose NEWER than the scan we are about to register?
//
// This is the question `RTdelta_m` in the overlay CSV could never answer. That probe compares the
// pose the tree returns at "now" with the one at the scan stamp, and reports 0 both when the ring
// is bracketing perfectly (the newest block IS the scan's block) and when it has nothing newer to
// offer and clamps — opposite situations with the same reading. The RT ring is only
// RT_API::HISTORY_SIZE (5) blocks deep and never extrapolates, so the sign of this lead is what
// decides which case you are in:
//   lead > 0  the pose feed is ahead of the scan; InterpolatedRT can bracket it → registration exact.
//   lead ≈ 0  the newest block IS this scan's pose; correct, but nothing to interpolate against.
//   lead < 0  the pose feed is BEHIND the scan; every query clamps to a stale block and the whole
//             cloud is registered with a pose |lead| ms too old → a bulk ω·|lead| rotation error
//             that grows with turn rate and vanishes at rest.
// Reads the ring directly rather than inferring it from returned poses. Diagnostic only.
void ControllerObstacleTracker::update_rt_block_lead(std::uint64_t scan_ts)
{
    rt_block_lead_ms_.reset();
    if (!graph_ || !graph_state_ || !graph_state_->ready() || scan_ts == 0)
        return;

    // room_concept writes the edge parent→child, and which of the two is the parent depends on
    // whether the room is anchored to the robot or the other way round — so try both directions.
    auto edge = graph_->get_edge(graph_state_->room_id, graph_state_->robot_id, "RT");
    if (!edge.has_value())
        edge = graph_->get_edge(graph_state_->robot_id, graph_state_->room_id, "RT");
    if (!edge.has_value())
        return;

    const auto stamps = graph_->get_attrib_by_name<rt_timestamps_att>(edge.value());
    if (!stamps.has_value())
        return;
    const std::vector<std::uint64_t> &ring = stamps->get();
    std::uint64_t newest = 0, oldest = std::numeric_limits<std::uint64_t>::max();
    for (const auto ts : ring)          // min/max over the ring, so head-index semantics don't matter
    {
        if (ts == 0) continue;          // unused slot
        newest = std::max(newest, ts);
        oldest = std::min(oldest, ts);
    }
    if (newest == 0)
        return;                          // untimestamped edge: InterpolatedRT is a no-op entirely

    rt_block_newest_ts_ = newest;
    rt_block_oldest_ts_ = oldest;
    rt_block_lead_ms_ = static_cast<std::int64_t>(newest) - static_cast<std::int64_t>(scan_ts);

    // The body twist room_concept publishes on this same edge, written there expressly so consumers
    // reconstruct motion between blocks instead of differentiating pose. [adv(+x fwd), side(+y), _]
    // and [_, _, rot] in the ROBOT frame — the writer does not invert them when the edge runs
    // robot→room, so they are the robot's own twist in either direction.
    rt_twist_valid_ = false;
    const auto lin = graph_->get_attrib_by_name<rt_translation_velocity_att>(edge.value());
    const auto ang = graph_->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(edge.value());
    if (lin.has_value() and ang.has_value() and lin->get().size() >= 2 and ang->get().size() >= 3)
    {
        rt_twist_adv_  = lin->get()[0];
        rt_twist_side_ = lin->get()[1];
        rt_twist_rot_  = ang->get()[2];
        rt_twist_valid_ = std::isfinite(rt_twist_adv_) and std::isfinite(rt_twist_side_)
                          and std::isfinite(rt_twist_rot_);
    }
}

// Bring a room←robot pose that the RT tree could only CLAMP onto the instant the scan was actually
// taken, by walking the robot along its own published twist.
//
// The identity is exact and direction-free:
//     room←robot(t_scan) = room←robot(t_block) · robot(t_block)←robot(t_scan)
//                                                └─ Exp(ξ·Δt), Δt = t_scan − t_block ─┘
// so "extrapolate the pose forward" and "move the cloud back onto the pose" are the same single
// matrix compose — this does it on the pose side, once per plane, instead of per point.
//
// Applied ONLY when the query fell outside the ring and the returned pose is therefore a clamped
// end block: that is the case the RT API gets wrong and the one we can detect (inside the ring its
// own interpolation is already exact and is left alone). Δt of either sign works — beyond the
// leading edge it walks forward, before the trailing edge it walks back.
//
// NO horizon cap, deliberately. Skipping the correction is not the neutral choice it looks like: it
// is the same extrapolation with the velocity assumed to be ZERO, which is strictly worse than
// integrating a twist that was actually measured. A stale pose feed is already handled where it
// belongs — as growing covariance in the speed throttle, and by the LiDAR stall hold.
Eigen::Matrix4d ControllerObstacleTracker::twist_corrected(const Eigen::Matrix4d &room_T_robot,
                                                           std::uint64_t target_ts,
                                                           std::int64_t *applied_dt_ms) const
{
    if (applied_dt_ms) *applied_dt_ms = 0;
    if (!params_ || !params_->rt_twist_compensation || !rt_twist_valid_
        || !rt_block_newest_ts_.has_value() || !rt_block_oldest_ts_.has_value() || target_ts == 0)
        return room_T_robot;

    // Inside the ring the RT API interpolated properly — nothing to repair.
    std::int64_t dt_ms = 0;
    if (target_ts > *rt_block_newest_ts_)
        dt_ms = static_cast<std::int64_t>(target_ts) - static_cast<std::int64_t>(*rt_block_newest_ts_);
    else if (target_ts < *rt_block_oldest_ts_)
        dt_ms = static_cast<std::int64_t>(target_ts) - static_cast<std::int64_t>(*rt_block_oldest_ts_);
    if (dt_ms == 0)
        return room_T_robot;

    if (applied_dt_ms) *applied_dt_ms = dt_ms;
    return room_T_robot * twist_delta(dt_ms);   // right-multiply: delta is in the robot frame at t_block
}

// Exp(ξ·Δt) for the cached body twist — the robot's own motion over Δt, expressed in the robot frame
// it started in. Δt may be negative (walk back). Split out of twist_corrected() so the accuracy probe
// exercises the SAME arithmetic the correction relies on, rather than a re-derivation of it.
Eigen::Matrix4d ControllerObstacleTracker::twist_delta(std::int64_t dt_ms) const
{
    const double dt = static_cast<double>(dt_ms) * 1e-3;
    const double phi = static_cast<double>(rt_twist_rot_) * dt;       // yaw swept in the interval
    // ★ AXIS ASSIGNMENT — this robot's frame is +Y FORWARD, +X lateral. The twist ARRAY is ordered
    // [adv, side, _], so index 0 is the FORWARD rate and belongs on the frame's y axis, not its x.
    // Getting this backwards is not a small error: it rotates the predicted displacement by 90°, so
    // the prediction lands √2·|motion| away from the truth — WORSE than assuming the robot never
    // moved. Measured 2026-08-04 against 421 logged forward-driving cycles, predicting the next pose
    // from the true body twist: adv→x gives p50 25.86 mm, adv→y gives p50 0.07 mm.
    // (The frame is stated by the safety gate's own integrator, `x += adv*sin(th); y += adv*cos(th)`,
    // and by the carrot bearing being atan2(x, y) rather than atan2(y, x).)
    const double vx = static_cast<double>(rt_twist_side_) * dt;
    const double vy = static_cast<double>(rt_twist_adv_) * dt;

    // SE(2) exponential. The left Jacobian V turns the body velocity into the CHORD of the arc the
    // robot actually drove; using v·dt straight would cut the corner, and at ω·dt ≈ 0.1 rad that
    // error is already ~0.2% of the displacement — small, but free to do right.
    double dx = vx, dy = vy;
    if (std::abs(phi) > 1e-6)
    {
        const double s = std::sin(phi), c = std::cos(phi);
        dx = ( s * vx - (1.0 - c) * vy) / phi;
        dy = ((1.0 - c) * vx +  s   * vy) / phi;
    }

    Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
    delta(0, 0) =  std::cos(phi); delta(0, 1) = -std::sin(phi);
    delta(1, 0) =  std::sin(phi); delta(1, 1) =  std::cos(phi);
    delta(0, 3) = dx;             delta(1, 3) = dy;
    return delta;
}

// How much can the twist actually be TRUSTED over one lidar period?
//
// This is the number the one-frame buffer decision turns on. Keeping the buffer means registering an
// exact pose 78 ms late; dropping it means registering a fresh scan whose pose was walked forward by
// twist_corrected() over ~one period. Which is better is not a matter of opinion — it is whether the
// twist predicts that interval to better than the ω·Δt error the buffer exists to avoid. Nobody had
// measured it, in either simulation or the robot, so the flag was being argued from assumption.
//
// Method: take two instants one lidar period apart that are BOTH strictly inside the RT ring, so both
// poses are genuine interpolations rather than clamped end blocks. Predict the later from the earlier
// with the same Exp(ξ·Δt) the correction uses, and compare against what the tree reports. The residual
// is exactly the error dropping the buffer would introduce.
//
// Standing still it is trivially ~0, so read it filtered by motion — that is an analysis choice, not
// a gate here: this logs what it measures and never suppresses a sample.
void ControllerObstacleTracker::update_twist_prediction_error(std::uint64_t scan_ts)
{
    twist_pred_err_m_.reset();
    twist_pred_err_deg_.reset();
    twist_pred_dt_ms_ = 0;
    if (!params_ || !inner_eigen_api_ || !graph_state_ || !rt_twist_valid_
        || !rt_block_newest_ts_.has_value() || !rt_block_oldest_ts_.has_value())
        return;

    const std::int64_t dt_ms = static_cast<std::int64_t>(lidar_period_ms_);
    if (dt_ms <= 0)
        return;
    // Anchor at the scan when the ring reaches it, else at the newest block; both ends must sit
    // inside the ring or we would be comparing against a clamp and measuring nothing.
    const std::uint64_t t_late = std::min<std::uint64_t>(scan_ts, *rt_block_newest_ts_);
    if (t_late < *rt_block_oldest_ts_ + static_cast<std::uint64_t>(dt_ms))
        return;
    const std::uint64_t t_early = t_late - static_cast<std::uint64_t>(dt_ms);

    const auto interp = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                : DSR::RT_API::TimeQuery::Nearest;
    const auto early = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                   graph_state_->robot_name,
                                                                   t_early, "RT", interp);
    const auto late  = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                   graph_state_->robot_name,
                                                                   t_late, "RT", interp);
    if (!early.has_value() || !late.has_value())
        return;

    const Eigen::Matrix4d predicted = early->matrix() * twist_delta(dt_ms);
    const Eigen::Matrix4d actual    = late->matrix();

    const double dx = predicted(0, 3) - actual(0, 3);
    const double dy = predicted(1, 3) - actual(1, 3);
    const double yaw_pred = std::atan2(predicted(1, 0), predicted(0, 0));
    const double yaw_act  = std::atan2(actual(1, 0), actual(0, 0));
    double dyaw = yaw_pred - yaw_act;
    while (dyaw >  M_PI) dyaw -= 2.0 * M_PI;          // shortest signed angle
    while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

    twist_pred_dt_ms_ = dt_ms;
    twist_pred_err_m_ = static_cast<float>(std::hypot(dx, dy));
    twist_pred_err_deg_ = static_cast<float>(std::abs(dyaw) * 180.0 / M_PI);
}

bool ControllerObstacleTracker::handle_lidar_points(const std::string &lidar_node_name,
                                                    std::vector<float> xs,
                                                    std::vector<float> ys,
                                                    std::vector<float> zs,
                                                    std::uint64_t timestamp_ms,
                                                    std::vector<std::uint8_t> plane_ids,
                                                    std::vector<std::int64_t> plane_stamps)
{
    if (!params_ || !inner_eigen_api_ || !graph_state_ || !graph_state_->ready())
        return false;

    // Dedup + period tracking against the newest RAW source stamp: when the laser node is polled
    // (DSR fallback), the same scan would otherwise be reprocessed every cycle. A 0 stamp can't be
    // deduped, so it always passes.
    if (timestamp_ms != 0 && newest_raw_lidar_ts_.has_value()
        && timestamp_ms <= newest_raw_lidar_ts_.value())
        return false;

    if (timestamp_ms > 0 && newest_raw_lidar_ts_.has_value() && timestamp_ms > newest_raw_lidar_ts_.value())
    {
        const std::uint64_t diff_ms = timestamp_ms - newest_raw_lidar_ts_.value();
        if (diff_ms >= 20 && diff_ms <= 500)
        {
            const double blended_period_ms = 0.7 * static_cast<double>(lidar_period_ms_)
                                           + 0.3 * static_cast<double>(diff_ms);
            lidar_period_ms_ = static_cast<std::uint64_t>(std::clamp(blended_period_ms, 20.0, 500.0));
        }
    }
    newest_raw_lidar_ts_ = timestamp_ms;

    // ── ONE-FRAME HOLD (not optional; was Transforms.overlay_draw_one_frame_old until 2026-08-04) ──
    // Register the PREVIOUS scan, not the newest one. Two things come of the delay, and only the first
    // is obvious:
    //  1. REGISTRATION. room_concept has had a full frame to publish that scan's room←robot pose, so
    //     InterpolatedRT brackets it exactly instead of clamping at the leading edge.
    //  2. SMOOTHING. A one-frame-old cloud is also a temporally smoothed one, and the safety gate
    //     downstream is exquisitely sensitive to transient returns.
    // ★A/B'd and SETTLED, do not re-litigate without reading this. twist_corrected() made (1)
    // unnecessary — it reproduces the pose at the scan instant to 0.22 cm p50, against the 2.8 cm of
    // obstacle lag the wait costs — so the hold was removed to reclaim ~48 ms of reaction latency.
    // Measured with it off (gap_ms 78 → 30 ms, so the change demonstrably took): cross-track and
    // clearance UNCHANGED exactly as predicted, but safety_guard_cycles 124 vs a 16-61 baseline (2x the
    // worst lap), lin_jerk_effort 92.6 vs 17.5-55.0, rot_reversals 54 vs 17-28, and the slowest lap of
    // the nine. The freshest scan puts more transient returns in front of a gate that already fires in
    // 1-cycle pulses, and it fired three times as often — so the 12x-favourable registration trade was
    // real and measured the axis that did not matter.
    // ★To reclaim the latency, fix the GATE (TRACKER_REVIEW.md R2: move the safety cut inside the speed
    // profile so it stops emitting 1-cycle pulses). Once the gate is continuous, (2) is redundant and
    // this hold can go. n=1, Webots — but 2x clear of the baseline spread on three consistent metrics.
    if (!pending_lidar_scan_.has_value())
    {
        pending_lidar_scan_ = PendingLidarScan{std::move(xs), std::move(ys), std::move(zs), timestamp_ms,
                                              std::move(plane_ids), std::move(plane_stamps)};
        return true;   // first frame: nothing held back yet
    }
    std::vector<float> proc_xs = std::move(pending_lidar_scan_->xs);
    std::vector<float> proc_ys = std::move(pending_lidar_scan_->ys);
    std::vector<float> proc_zs = std::move(pending_lidar_scan_->zs);
    const std::uint64_t proc_ts = pending_lidar_scan_->ts;
    std::vector<std::uint8_t> proc_plane_ids = std::move(pending_lidar_scan_->plane_ids);
    std::vector<std::int64_t> proc_plane_stamps = std::move(pending_lidar_scan_->plane_stamps);
    pending_lidar_scan_ = PendingLidarScan{std::move(xs), std::move(ys), std::move(zs), timestamp_ms,
                                          std::move(plane_ids), std::move(plane_stamps)};
    last_lidar_timestamp_ms_ = proc_ts;
    update_rt_block_lead(proc_ts);            // caches the ring bounds + twist the two calls below use
    update_twist_prediction_error(proc_ts);

    const auto interp = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                : DSR::RT_API::TimeQuery::Nearest;
    const auto room_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                             lidar_node_name,
                                                                             proc_ts,
                                                                             "RT",
                                                                             interp);
    const auto robot_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_->robot_name,
                                                                               lidar_node_name,
                                                                               proc_ts,
                                                                               "RT",
                                                                               interp);
    if (!room_from_lidar.has_value() || !robot_from_lidar.has_value())
        return false;

    const std::size_t raw_count = std::min({proc_xs.size(), proc_ys.size(), proc_zs.size()});
    if (raw_count == 0)
        return false;

    // Repair the leading-edge clamp before anything is registered: the freshest scan is normally
    // NEWER than the freshest pose block (the pose is computed FROM a scan, so it can only arrive
    // after one), which is precisely when InterpolatedRT silently returns the newest block instead.
    rt_twist_fix_dt_ms_ = 0;
    const Eigen::Matrix4d room_from_lidar_matrix =
        twist_corrected(room_from_lidar->matrix(), proc_ts, &rt_twist_fix_dt_ms_);
    const auto robot_from_lidar_matrix = robot_from_lidar->matrix();

    // ── PER-PLANE REGISTRATION ────────────────────────────────────────────────────────────────────
    // `proc_ts` is the MAX of the merged planes' capture stamps, so registering the whole cloud with
    // the pose at that one instant misplaces every plane that fired earlier by ω·(proc_ts − its own
    // stamp) — up to a full LiDAR period between helios and bpearl. Invisible standing still, and
    // exactly the term that grows with turn rate. So resolve room←lidar once PER PLANE, at that
    // plane's own stamp. A plane whose query fails, or whose stamp matches proc_ts, falls back to the
    // single matrix above, which is also what a one-plane sweep and any caller without per-plane
    // information get — i.e. the previous behaviour, unchanged.
    std::vector<Eigen::Matrix4d> room_from_plane;
    if (not proc_plane_stamps.empty() and not proc_plane_ids.empty())
    {
        room_from_plane.assign(proc_plane_stamps.size(), room_from_lidar_matrix);
        for (std::size_t k = 0; k < proc_plane_stamps.size(); ++k)
        {
            const std::int64_t stamp = proc_plane_stamps[k];
            if (stamp <= 0 or static_cast<std::uint64_t>(stamp) == proc_ts)
                continue;
            if (const auto plane_T = inner_eigen_api_->get_transformation_matrix(
                    graph_state_->room_name, lidar_node_name,
                    static_cast<std::uint64_t>(stamp), "RT", interp);
                plane_T.has_value())
                room_from_plane[k] = twist_corrected(plane_T->matrix(),
                                                     static_cast<std::uint64_t>(stamp));
        }
    }
    // NOTE: the RoboComp Lidar3D source already returns points in the robot/base frame (floor ≈ z=0),
    // so the robot→lidar RT edge is deliberately identity; robot_z below is therefore height above the
    // floor. If someone ever moves the lidar node to its true mount height in shadow.json this shortcut
    // double-counts — keep the edge identity for this data convention.
    const bool robot_from_lidar_is_identity = robot_from_lidar_matrix.isApprox(Eigen::Matrix4d::Identity(), 1e-5);

    const float min_h = params_->temporary_obstacle_min_height_m;
    const float max_h = params_->temporary_obstacle_max_height_m;

    // ── RAW-cloud proximity, measured BEFORE the z-band cut below ──
    // Everything else here reads the already-filtered buffer, so every controller diagnostic has been
    // blind to returns under min_h — yet those returns DO feed residual_concept's grid. A low obstacle
    // residual legitimately sees would therefore look like an unsupported phantom from this side. Take
    // the measurement now, while the unfiltered sweep is still in hand.
    {
        raw_cloud_proximity_ = RawCloudProximity{};
        float best = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < raw_count; ++i)
        {
            const float x = proc_xs[i], y = proc_ys[i], z = proc_zs[i];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                continue;
            const float rz = robot_from_lidar_is_identity
                ? z
                : static_cast<float>(robot_from_lidar_matrix(2, 0) * x + robot_from_lidar_matrix(2, 1) * y
                                     + robot_from_lidar_matrix(2, 2) * z + robot_from_lidar_matrix(2, 3));
            const float rx = robot_from_lidar_is_identity
                ? x
                : static_cast<float>(robot_from_lidar_matrix(0, 0) * x + robot_from_lidar_matrix(0, 1) * y
                                     + robot_from_lidar_matrix(0, 2) * z + robot_from_lidar_matrix(0, 3));
            const float ry = robot_from_lidar_is_identity
                ? y
                : static_cast<float>(robot_from_lidar_matrix(1, 0) * x + robot_from_lidar_matrix(1, 1) * y
                                     + robot_from_lidar_matrix(1, 2) * z + robot_from_lidar_matrix(1, 3));
            const float d = std::hypot(rx, ry);
            if (rz < min_h && d <= 1.0f)
                ++raw_cloud_proximity_.below_band_within_1m;   // what the band is discarding near the robot
            if (d < best)
            {
                best = d;
                raw_cloud_proximity_.z_m = rz;
                raw_cloud_proximity_.bearing_deg = std::atan2(rx, ry) * 180.f / static_cast<float>(M_PI);
            }
        }
        if (std::isfinite(best))
            raw_cloud_proximity_.distance_m = best;
    }

    // Flatten each plane's room←lidar into the 12 coefficients the hot loop actually uses, indexed by
    // plane id. One entry ⇒ every point takes it, which is the single-matrix path this replaced.
    std::vector<std::array<double, 12>> plane_coeffs;
    {
        const auto flatten = [](const Eigen::Matrix4d &m)
        {
            return std::array<double, 12>{m(0,0), m(0,1), m(0,2), m(0,3),
                                          m(1,0), m(1,1), m(1,2), m(1,3),
                                          m(2,0), m(2,1), m(2,2), m(2,3)};
        };
        if (room_from_plane.empty())
            plane_coeffs.push_back(flatten(room_from_lidar_matrix));
        else
            for (const auto &m : room_from_plane)
                plane_coeffs.push_back(flatten(m));
    }

    lidar_room_buffer_.put<0>(
        rc::RawLidarPointVectors{
            .xs = std::move(proc_xs),
            .ys = std::move(proc_ys),
            .zs = std::move(proc_zs)},
        proc_ts,
        [plane_coeffs = std::move(plane_coeffs), plane_sel = std::move(proc_plane_ids),
         robot_from_lidar_matrix, robot_from_lidar_is_identity, raw_count, min_h, max_h](rc::RawLidarPointVectors &&raw_points, rc::LidarPointVectors &room_points)
        {
            auto &[room_xs, room_ys, room_zs] = room_points;
            const auto &xs_in = raw_points.xs;
            const auto &ys_in = raw_points.ys;
            const auto &zs_in = raw_points.zs;
            const std::size_t count = std::min({raw_count, xs_in.size(), ys_in.size(), zs_in.size()});
            if (count == 0)
                return;

            const bool per_plane = plane_coeffs.size() > 1 and plane_sel.size() >= count;
            const auto &m0 = plane_coeffs.front();
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
                if (robot_z < min_h || robot_z > max_h)
                    continue;

                // Each point is placed with ITS OWN plane's pose (see PER-PLANE REGISTRATION above).
                const auto &m = per_plane
                    ? plane_coeffs[std::min<std::size_t>(plane_sel[index], plane_coeffs.size() - 1)]
                    : m0;
                const float room_x = static_cast<float>(m[0] * x + m[1] * y + m[2]  * z + m[3]);
                const float room_y = static_cast<float>(m[4] * x + m[5] * y + m[6]  * z + m[7]);
                const float room_z = static_cast<float>(m[8] * x + m[9] * y + m[10] * z + m[11]);

                room_xs.push_back(room_x);
                room_ys.push_back(room_y);
                room_zs.push_back(room_z);
            }
        });
    return true;
}