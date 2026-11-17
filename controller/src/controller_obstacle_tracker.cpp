#include "controller_obstacle_tracker.h"

#include "../../common/robot_footprint/robot_footprint.h"
#include "lidar_body_guard.h"
#include "lidar_height_band.h"
#include "residual_risk.h"

#include <array>
#include <format>
#include <fstream>
#include <chrono>
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
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

std::string ControllerObstacleTracker::published_obstacle_name(std::uint64_t obstacle_id)
{
    return "obs" + std::to_string(obstacle_id);
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
            rc::provenance::stamp_creation(*graph_, obstacle_node);   // birth stamp: epoch ms + local ISO-8601
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

namespace
{
// Worst-axis position sigma (m) from an object's RT edge covariance, or 0 when unavailable.
//
// LAYOUT: 6x6 row-major SE3 [x,y,z,rx,ry,rz] (see RT_COVARIANCE_BLOCK_SIZE in cortex's dsr_rt_api.cpp
// and the writers in each concept agent), so var_x = [0] and var_y = [1*6+1] = [7]. ★Yaw would be
// [5*6+5] = [35], NOT [14] — [14] is var_z.
//
// EXACT size 36 only. Concept agents write a single block straight onto the edge, but cortex's RT api
// keeps a RING of blocks on edges it manages; block 0 of a ring is not the newest, and quietly planning
// against a stale covariance is worse than not using one at all. A ring needs the head index, which is
// a separate change.
struct NullReport { template <class T> NullReport &operator<<(const T &) { return *this; } };

float object_position_sigma_from_edge(const std::shared_ptr<DSR::DSRGraph> &graph, const DSR::Node &node)
{
    if (graph == nullptr) return 0.f;
    const auto parent = graph->get_attrib_by_name<parent_att>(node);
    if (not parent.has_value()) return 0.f;
    const auto edge = graph->get_edge(parent.value(), node.id(), "RT");
    if (not edge.has_value()) return 0.f;
    const auto cov = graph->get_attrib_by_name<rt_covariance_att>(edge.value());
    if (not cov.has_value()) return 0.f;
    const auto &c = cov.value().get();
    if (c.size() != 36) return 0.f;
    return std::sqrt(std::max(0.f, std::max(c[0], c[7])));
}
}   // namespace

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

ControllerPolygons ControllerObstacleTracker::read_obstacle_polygons(std::uint64_t timestamp_ms)
{
    ControllerPolygons obstacles;
    known_graph_obstacles_.clear();
    display_obstacle_polygons_.clear();
    object_label_counts_.clear();   // per-type object tags (t_1/c_1/b_1) restart each rebuild
    obstacle_label_count_ = 0;      // shared o_N counter spans graph "obstacle" nodes + temporaries
    // The two debug reports were built every cycle (float formatting per object) and never printed or stored.
    // The sink keeps the call sites as documentation of what each branch decided, at no cost.
    NullReport report, object_report;
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
    // room<-node: ONE RT lookup per object. Objects hang directly off the room, so read that single room->node
    // RT edge (an Edge copy, no parent-node copy) instead of two InnerEigen tree walks (transform + euler),
    // each of which copies the whole room node (an RT edge per object) at every hop. Objects not parented to the
    // room keep the full-chain lookup. Yaw is atan2 of the rotation (objects are yaw-only).
    struct RoomPose { Eigen::Vector2f xy; float yaw; };
    const auto room_pose_of = [&](const DSR::Node &node) -> std::optional<RoomPose>
    {
        std::optional<Mat::RTMat> m;
        if (rt_api_ and graph_state_->room_id != 0)
            if (const auto edge = graph_->get_edge(graph_state_->room_id, node.id(), "RT"); edge.has_value())
                m = rt_api_->get_edge_RT_as_rtmat(edge.value(), timestamp_ms, time_query);
        if (not m.has_value())
            m = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name, node.name(),
                                                            timestamp_ms, "RT", time_query);
        if (not m.has_value()) return std::nullopt;
        const auto &M = m->matrix();
        return RoomPose{Eigen::Vector2f(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3))),
                        static_cast<float>(std::atan2(M(1, 0), M(0, 0)))};
    };

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
        {
            // A door being CROSSED is not an obstacle (set_exempt_names, controller_door_passage.h): its node is a
            // box across the opening the robot has been offered to pass through. The open leaf is a real thing and
            // still reaches the planner through the residual LiDAR cells.
            if (exempt_names_.contains(node.name())) continue;
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
            // The door being crossed is not an obstacle here either (2026-10-11): when it is the ONLY object node,
            // exempting it above empties the typed list and lands HERE, which put its box straight back across
            // the doorway (live: "no route" through the passage corridor, the box at the wall line).
            if (exempt_names_.contains(node.name()))
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

            if (not room_pose_of(node).has_value())
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
        const auto room_pose = room_pose_of(node);
        if (not room_pose.has_value())
        {
            report << " missing_rt";
            continue;
        }

        const Eigen::Vector2f center = room_pose->xy;
        const float yaw = room_pose->yaw;
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

        // ── Clearance from the object's OWN position uncertainty — PLANNING COPY ONLY ───────────
        // ★This must NOT touch `state`. state flows into known_graph_obstacles_ (the recorded geometry),
        // into the display polygon and into round_radius_m, so inflating it in place makes the padded
        // footprint the object's SIZE everywhere downstream — which is what blocked the robot when this
        // was first written: the world model started believing tables were metres across.
        //
        // rt_covariance is a 6x6 row-major SE3 covariance [x,y,z,rx,ry,rz]; position variances are [0]
        // and [1*6+1]=[7]. Concept agents write ONE 36-element block directly on the edge, unlike edges
        // cortex's RT api manages, where it is a RING and block 0 is not the newest — hence the exact
        // size test rather than a >=.
        //
        // ★AND IT MUST BE BOUNDED. A concept agent grows Sigma while it is not measuring the object, by
        // design, so sigma rises without limit whenever the robot looks away — and an unbounded margin
        // walls off the room. Past the robot's own radius, more clearance does not make a path safer, it
        // makes it nonexistent; "I do not know where it is" is an argument to go and LOOK, which is what
        // the epistemic affordances are for, not an argument to refuse to move.
        ControllerObstacleState plan_state = state;   // planning-only; `state` stays the fitted truth
        // Read sigma ALWAYS, even with the inflation disabled: the canvas draws it as a halo, and the
        // whole point of that display is to watch sigma grow and shrink while the inflation is off.
        const float sigma_pos = (node.type() != "obstacle")
                              ? object_position_sigma_from_edge(graph_, node) : 0.f;
        if (params_ != nullptr and params_->object_sigma_inflation_k > 0.f
            and node.type() != "obstacle")
        {
            if (const float sigma = sigma_pos; sigma > 0.f)
            {
                const float want = 2.0f * params_->object_sigma_inflation_k * sigma;   // both sides
                const float grow = std::min(want, std::max(0.f, params_->object_sigma_inflation_max_m));
                plan_state.width_m += grow;
                plan_state.depth_m += grow;
                report << " sigma_pos=" << sigma << " inflated_by=" << grow
                       << (grow < want ? " (CAPPED)" : "");
            }
        }

        auto polygon = ControllerObstacleModel::polygon_from_state(plan_state);
        report << " center=(" << state.center.x() << "," << state.center.y() << ") yaw=" << state.yaw_rad;
        if (!polygon.empty())
            report << " first_vertex=(" << polygon.front().x() << "," << polygon.front().y() << ")";
        // Anything a concept agent modelled (object/table/cylinder/chair) is an "object"; only
        // an explicit graph "obstacle" node counts as an unmodelled obstacle here.
        const ControllerObstacleKind kind = node.type() == "obstacle"
            ? ControllerObstacleKind::Obstacle
            : ControllerObstacleKind::Object;
        std::string visual_label;
        bool round_footprint = false;
        if (kind == ControllerObstacleKind::Object)
        {
            // Post-migration the node type is generic "object"; the class label lives in the
            // object_subtype attribute (cosmetic log label). Fall back to type() for legacy nodes.
            const auto subtype_attr = graph_->get_attrib_by_name<object_subtype_att>(node);
            const std::string subtype = subtype_attr.value_or(node.type());
            // ★ THE NODE'S OWN NAME (2026-10-18, user). The canvas used to print a per-rebuild counter ("d_1"),
            //   so the door every other agent and window calls door_5 read d_1 here, and which object was which had
            //   to be guessed. Every concept agent names its nodes (door_5, table_1, …); that IS the identity.
            visual_label = node.name();
            // ROUND-vs-SQUARE comes off mesh_path, not off a shape attribute, because that IS the
            // contract: object_subtype carries the CLASS ("table") for every table, and table_concept
            // publishes its inferred SHAPE by choosing round_table.obj over table.obj (free-energy
            // model evidence — see TableFitter::evaluate_shape). Sniffing the path is what the
            // retina's 3D viewer already does; if that contract ever grows a real attribute this is
            // the one line that has to follow it.
            if (subtype == "table")
                if (const auto mesh = graph_->get_attrib_by_name<mesh_path_att>(node); mesh.has_value())
                    round_footprint = mesh.value().find("round") != std::string::npos;
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
        display_obstacle_polygons_.push_back(ControllerObstacleVisual{
            .polygon = polygon,
            .kind = kind,
            .label = visual_label,
            // The identity the canvas colours by — see ControllerObstacleVisual::color_key. The node
            // name, not visual_label: that one is a per-rebuild counter and would re-hue an object
            // whenever a sibling of the same type came or went.
            .color_key = node.name(),
            .round = round_footprint,
            // The disc spans the same extent as the box it replaces: a round table whose belief says
            // 0.90 m across is drawn 0.90 m across. Taking the LARGER of the two is what makes it a
            // bound rather than a guess — the two are equal for a true disc and differ only while the
            // fit is still settling, and under-drawing a footprint is the failure that matters.
            .round_radius_m = round_footprint ? 0.5f * std::max(state.width_m, state.depth_m) : 0.f,
            .sigma_pos_m = sigma_pos});
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

    // ── residual_concept OCCUPANCY GRID: TAKEN AS CELLS, NOT AS POLYGONS ─────────────────────────
    // ★★★THE ROUND TRIP IS GONE (2026-08-20). residual used to publish `grid_obstacle_hulls`: each
    // connected component of its occupancy, C-space inflated by 0.25 m, decomposed into an exact
    // rectangular cover, which this function decoded back into polygons for the planner to rasterise
    // into a grid again. Two faults lived in that trip. The inflation was counted TWICE — 0.25 m here
    // plus the robot's own half-extent applied by GridPlanner at query time — so one 5 cm cell of
    // residual became a no-go zone close to a metre across; and a cover of tiles is a lossy,
    // unstable re-encoding of a grid, which churned the planner's world at 20 Hz under a stationary
    // robot (measured: one (cell,pose) verdict re-decided 33 times in five minutes).
    // ★So take what residual actually knows: `grid_occupied_cells`, its own resolution, marked
    // directly. The robot's shape is then applied exactly once, where it belongs.
    if (const auto g = graph_->get_node("residual"); g.has_value())
    {
        // residual republishes its grid at its own rate, not ours: key on the publish stamps of the cells and the
        // belief field (exact — an attribute's stamp changes on every write). Unchanged => the cells, risks and
        // display squares from the last read are still the truth; a vanished node never reaches here, so a stale
        // grid cannot outlive its source any longer than it did before.
        const auto stamp_cells = graph_->get_attrib_timestamp<grid_occupied_cells_att>(g.value()).value_or(0);
        const auto stamp_prob  = graph_->get_attrib_timestamp<grid_occupancy_prob_att>(g.value()).value_or(0);
        if (residual_stamp_cells_ == stamp_cells and residual_stamp_prob_ == stamp_prob and stamp_cells != 0)
        {
            display_obstacle_polygons_.insert(display_obstacle_polygons_.end(),
                                              residual_display_.begin(), residual_display_.end());
            return obstacles;
        }
        residual_stamp_cells_ = stamp_cells;
        residual_stamp_prob_  = stamp_prob;
        residual_display_.clear();
        residual_cells_.clear();
        residual_cell_risk_nats_.clear();
        residual_cell_size_m_ = 0.f;
        if (const auto cs = graph_->get_attrib_by_name<grid_cell_size_att>(g.value()); cs.has_value())
            residual_cell_size_m_ = cs.value();
        if (const auto opt = graph_->get_attrib_by_name<grid_occupied_cells_att>(g.value()); opt.has_value())
        {
            const auto &f = opt.value().get();
            residual_cells_.reserve(f.size() / 3);
            for (std::size_t i = 0; i + 2 < f.size(); i += 3)      // [x,y,z] triplets; z is the cell's top
                residual_cells_.emplace_back(f[i], f[i + 1]);
            // ── ...AND HOW MUCH RESIDUAL BELIEVES EACH ONE (2026-10-19, try96 leg 3) ─────────────
            // The cell list is a LATCH (one near hit sets it); the Beta field beside it says how much
            // evidence stands behind the latch. The planner turns that into a cost (residual_risk.h)
            // instead of a wall, so a weak curtain cannot seal the map. No field ⇒ every cell HARD.
            rc::residual_risk::BeliefField bf;
            const auto pa = graph_->get_attrib_by_name<grid_occupancy_prob_att>(g.value());
            const auto va = graph_->get_attrib_by_name<grid_occupancy_var_att>(g.value());
            const auto ma = graph_->get_attrib_by_name<grid_field_meta_att>(g.value());
            if (pa.has_value() and va.has_value() and ma.has_value() and ma.value().get().size() >= 5)
            {
                const auto &M = ma.value().get();
                bf.prob = pa.value().get();
                bf.var  = va.value().get();
                bf.xmin = M[0]; bf.ymin = M[1]; bf.cell = M[2];
                bf.width = static_cast<int>(M[3]); bf.height = static_cast<int>(M[4]);
            }
            residual_cell_risk_nats_ = rc::residual_risk::risks_from_field(residual_cells_, bf);
            // Drawn as what they are — one square per occupied cell at residual's own resolution. The
            // display used to show the inflated hulls, which over-stated the obstacle by a body width
            // and hid exactly the discrepancy this change exists to remove.
            if (residual_cell_size_m_ > 0.f)
            {
                const float h = 0.5f * residual_cell_size_m_;
                for (const auto &c : residual_cells_)
                    residual_display_.push_back(ControllerObstacleVisual{
                        .polygon = ControllerPolygon{{c.x() - h, c.y() - h}, {c.x() + h, c.y() - h},
                                                     {c.x() + h, c.y() + h}, {c.x() - h, c.y() + h}},
                        .kind = ControllerObstacleKind::GridOccupancy, .label = {}});
            }
        }
        display_obstacle_polygons_.insert(display_obstacle_polygons_.end(), residual_display_.begin(), residual_display_.end());
        report << " | grid_cells=" << residual_cells_.size() << " @" << residual_cell_size_m_ << "m";
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
    for (const auto &instance : temporary_obstacles_)
    {
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
// RT_API::HISTORY_SIZE blocks deep and never extrapolates, so the sign of this lead is what
// ★NOT 5 ON THIS ROBOT: room_concept sets rt_api_->HISTORY_SIZE = 25 (specificworker.cpp:169), and
// HISTORY_SIZE is a plain data member, not a constant — so the ring is as deep as whoever WROTE the
// edge made it, and reading the header's default to reason about it gives the wrong answer.
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

// (twist_corrected() lived here. It composed a clamped room<-robot pose with the published body twist
// to land it on the scan instant, and it was ONE OF FOUR copies of that SE(2) step in this fleet.
// Removed 2026-09-09: RT_API::TimeQuery::Extrapolated does it, once, inside the transform query — see
// the note at the query site in handle_lidar_points.)
//
// Exp(xi*dt) for the cached body twist — the robot's own motion over dt, in the robot frame it started
// in. dt may be negative (walk back).
// ★★★THIS IS NOW A DELIBERATE SECOND IMPLEMENTATION, AND THAT IS THE POINT. It used to be shared with
// twist_corrected() so the probe below "exercised the same arithmetic the correction relies on". That
// reasoning inverted the moment the correction moved into RT_API: a probe that calls the code under
// test cannot detect that code being wrong. So this stays here, reading the LEGACY [adv, side] array
// pair, while production reads the AXIS-order ring through cortex — two implementations, two attribute
// layouts, one expected answer. If they diverge, twist_pred_err_m says so per cycle.
// ⚠It follows that this must NOT be "simplified" by calling the API. Its value is that it does not.
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
// the RT query's own extrapolation over ~one period. Which is better is not a matter of opinion — it is whether the
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

// ── SETTLE: ANSWER AN OLD QUESTION WITH MEASUREMENT INSTEAD OF PREDICTION ───────────────────────
// For each parked pose, re-ask for the SAME instant with plain interpolation. Once the ring holds real
// blocks either side of that instant, the answer is measured rather than predicted, and the difference
// from what we guessed at the time is the error of the guess. Reported per cycle; at most one settles
// per call, so this costs one extra transform query and never loops on a backlog.
//
// ★A "NO" FROM THE RE-QUERY IS AMBIGUOUS AND IS TREATED AS SUCH. It can mean the ring has not reached
// the instant yet (wait) or that it has already scrolled past it (give up). The outcome of the
// re-query distinguishes them, which is why this checks it instead of trusting elapsed time: an entry
// is settled ONLY if the re-query itself reports Exact or Interpolated. Anything else -- a walk, a
// clamp, a stale verdict -- means the answer would be another prediction, and grading a prediction
// against a prediction measures nothing.
void ControllerObstacleTracker::settle_extrapolation_checks(const std::string &lidar_node_name,
                                                            std::uint64_t now_ts)
{
    extrap_check_err_m_.reset();
    extrap_check_err_deg_.reset();
    extrap_check_dt_ms_.reset();
    if (inner_eigen_api_ == nullptr or graph_state_ == nullptr)
        return;

    while (not pending_extrap_checks_.empty())
    {
        const auto pending = pending_extrap_checks_.front();
        // Too old to ever be bracketed again: the ring is finite and has scrolled past it.
        if (now_ts > pending.ask_ts + 3000)
        {
            pending_extrap_checks_.pop_front();
            continue;
        }

        DSR::RT_API::TimeQueryInfo info;
        const auto measured = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                          lidar_node_name,
                                                                          pending.ask_ts, "RT",
                                                                          DSR::RT_API::TimeQuery::Interpolated,
                                                                          &info);
        const bool answered_by_measurement =
            measured.has_value()
            and (info.outcome == DSR::RT_API::TimeQueryInfo::Outcome::Interpolated
                 or info.outcome == DSR::RT_API::TimeQueryInfo::Outcome::Exact);
        if (not answered_by_measurement)
            return;   // the ring has not reached this instant yet; entries are in time order

        const auto &truth = measured->matrix();
        const double dx = pending.x - truth(0, 3);
        const double dy = pending.y - truth(1, 3);
        double dyaw = pending.yaw - std::atan2(truth(1, 0), truth(0, 0));
        while (dyaw >  M_PI) dyaw -= 2.0 * M_PI;
        while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

        extrap_check_err_m_   = static_cast<float>(std::hypot(dx, dy));
        extrap_check_err_deg_ = static_cast<float>(std::abs(dyaw) * 180.0 / M_PI);
        extrap_check_dt_ms_   = pending.applied_dt_ms;
        pending_extrap_checks_.pop_front();
        return;
    }
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

    // ── THE RATE THE UI SHOWS, COUNTED WHERE A SCAN IS ACTUALLY ACCEPTED ─────────────────────────
    // ★★★MEASURED IN WALL TIME, NOT FROM THE STAMPS. `lidar_period_ms_` above is an EMA of the
    // difference between successive SOURCE STAMPS, which answers "how fast does the sensor say it is
    // sampling" — it stays at the nominal period even when we drop every other frame, because the
    // frames we do get are still stamped one period apart. The question a human asks looking at a
    // lagging overlay is the other one: how many scans per second is this process actually turning
    // into a cloud. Those two differ by exactly the drop rate, which is the interesting quantity.
    // ★This codebase has been bitten by rate displays before — four of them once disagreed, each
    // measuring a different stage and none saying which. So: counted HERE, after the dedup gate and
    // before any work, one place, and the label says what it counts.
    // A sliding window of arrival instants: rate = (n-1) / (newest - oldest). Robust to a burst and
    // self-clearing after a stall, with no EMA constant to explain.
    {
        const auto now_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        lidar_accept_wall_ms_.push_back(now_ms);
        while (lidar_accept_wall_ms_.size() > 32) lidar_accept_wall_ms_.pop_front();
        // Drop anything older than 2 s so a stall decays the reading toward 0 instead of freezing it
        // at the last healthy value — a rate display that keeps showing 20 Hz through a stall is worse
        // than none, because it is the display you would consult to notice the stall.
        while (lidar_accept_wall_ms_.size() > 2 and now_ms - lidar_accept_wall_ms_.front() > 2000)
            lidar_accept_wall_ms_.pop_front();
        if (lidar_accept_wall_ms_.size() >= 2)
        {
            const auto span = lidar_accept_wall_ms_.back() - lidar_accept_wall_ms_.front();
            lidar_processed_hz_ = span > 0
                ? 1000.f * static_cast<float>(lidar_accept_wall_ms_.size() - 1) / static_cast<float>(span)
                : 0.f;
        }
    }

    // ── THE NEWEST SCAN IS REGISTERED DIRECTLY (the one-frame hold was removed 2026-09-09) ────────
    // What the hold was for, and why neither reason survives:
    //  1. REGISTRATION. Waiting a frame let room_concept publish that scan's room←robot pose, so
    //     InterpolatedRT bracketed it instead of clamping at the leading edge. The extrapolating RT
    //     query below does that job directly, reproducing the pose at the scan instant to 0.22 cm p50
    //     against the 2.8 cm of obstacle lag the wait cost. Superseded, and measured.
    //  2. SMOOTHING. The stated beneficiary was the MPPI safety gate, which fired in 1-cycle pulses
    //     and had no temporal state of its own: dropping the hold took safety_guard_cycles to 124
    //     against a 16-61 baseline, lin_jerk_effort to 92.6 against 17.5-55.0, rot_reversals to 54
    //     against 17-28 (2026-08-04, n=1, Webots). ★THAT GATE NO LONGER EXISTS. ControlMode became
    //     "plain" one day after that A/B, and the MPPI sampler is now deleted outright. The live
    //     consumers of this buffer are detect_path_blockage → replan, which already debounces with
    //     blockage_streak_/blockage_confirm_cycles + a cooldown; the temporary-obstacle tracker,
    //     which fuses temporary_obstacle_history_scans (5) frames; and the footprint predicate along
    //     the carrot chord. None of them turns a single transient scan into a command pulse.
    // ★So the hold was paying ~48 ms of reaction latency to protect a consumer that had been retired.
    // ★WHAT TO WATCH on the first runs, and what to score it on: the August numbers CANNOT be the
    // baseline — safety_guard_cycles is structurally 0 in plain mode, so a run comparing it against
    // 16-61 is comparing against a tracker that no longer runs. Score this on `replans`,
    // blockage_streak behaviour and clearance instead, and expect gap_ms to fall ~78 → ~30.
    std::vector<float> proc_xs = std::move(xs);
    std::vector<float> proc_ys = std::move(ys);
    std::vector<float> proc_zs = std::move(zs);
    const std::uint64_t proc_ts = timestamp_ms;
    std::vector<std::uint8_t> proc_plane_ids = std::move(plane_ids);
    std::vector<std::int64_t> proc_plane_stamps = std::move(plane_stamps);
    last_lidar_timestamp_ms_ = proc_ts;
    update_rt_block_lead(proc_ts);            // caches the ring bounds + twist the probes below use
    // ★THE TWO PROBES (twist-vs-RT residual, extrapolation settle) ARE DIAGNOSTICS: their only reader is the
    // overlay CSV, which is itself written at <=10 Hz. Together they cost 3 timestamped RT-tree walks per scan, so
    // they run at 1 Hz and only when that CSV is on; between probes their outputs are cleared (empty CSV cells).
    const bool run_probes = not params_->overlay_csv_path.empty() and proc_ts >= last_probe_ts_ + 1000;
    if (run_probes) last_probe_ts_ = proc_ts;
    else { twist_pred_err_m_.reset(); twist_pred_err_deg_.reset(); twist_pred_dt_ms_ = 0;
           extrap_check_err_m_.reset(); extrap_check_err_deg_.reset(); extrap_check_dt_ms_.reset(); }
    if (run_probes) update_twist_prediction_error(proc_ts);

    // ── THE RT QUERY DOES THE EXTRAPOLATION NOW (2026-09-09) ─────────────────────────────────
    // TimeQuery::Extrapolated is Interpolated inside the ring and, past its ends, walks the pose along
    // the twist room_concept publishes in that same ring slot. This used to be twist_corrected() here,
    // one of FOUR copies of the same SE(2) step across this fleet — retina, viewer3d and
    // media_transport carried the others, and the axis-assignment bug had to be fixed in three of them
    // on 2026-09-04. Now there is one implementation, in RT_API, and it reads the twist in AXIS order
    // so no consumer re-encodes a body-frame convention.
    // ★rt_twist_compensation still selects it, so the A/B against plain interpolation is unchanged.
    const auto interp = params_->rt_twist_compensation ? DSR::RT_API::TimeQuery::Extrapolated
                      : params_->interpolate_rt        ? DSR::RT_API::TimeQuery::Interpolated
                                                       : DSR::RT_API::TimeQuery::Nearest;
    // ★AND IT NOW SAYS WHETHER IT CLAMPED. A timestamped RT query that falls outside the ring used
    // to return the end block and look exactly like a success, so a whole cloud could be registered
    // against a pose from another instant with nothing to show for it. rt_query_clamped_ is that
    // admission, logged per cycle beside the fix dt.
    DSR::RT_API::TimeQueryInfo rt_info;
    const auto room_from_lidar = inner_eigen_api_->get_transformation_matrix(graph_state_->room_name,
                                                                             lidar_node_name,
                                                                             proc_ts,
                                                                             "RT",
                                                                             interp,
                                                                             &rt_info);
    rt_twist_fix_dt_ms_ = rt_info.applied_dt_ms;
    rt_query_outcome_ = static_cast<int>(rt_info.outcome);
    rt_query_gap_ms_ = rt_info.gap_ms;
    rt_query_stale_edges_ = rt_info.stale_edges;

    // Settle any earlier forward-nudged pose whose instant the ring has now caught up with, THEN park
    // this one. In that order, so a pose is never graded against itself.
    if (run_probes) settle_extrapolation_checks(lidar_node_name, proc_ts);
    if (run_probes and rt_info.outcome == DSR::RT_API::TimeQueryInfo::Outcome::Extrapolated
        and rt_info.applied_dt_ms != 0 and room_from_lidar.has_value())
    {
        const auto &nudged = room_from_lidar->matrix();
        pending_extrap_checks_.push_back(PendingExtrapCheck{
            .ask_ts        = proc_ts,
            .x             = nudged(0, 3),
            .y             = nudged(1, 3),
            .yaw           = std::atan2(nudged(1, 0), nudged(0, 0)),
            .applied_dt_ms = rt_info.applied_dt_ms});
        // Bounded: the settle step drops anything the ring has already run past, but a producer that
        // stops publishing would otherwise let this grow for the life of the run.
        while (pending_extrap_checks_.size() > 128)
            pending_extrap_checks_.pop_front();
    }
    // The reader already merged into the robot frame (lidar_node_name == robot_name): robot<-robot is the
    // identity by definition, so skip the timestamped tree walk for it.
    const auto robot_from_lidar =
        lidar_node_name == graph_state_->robot_name
            ? std::optional<Mat::RTMat>{Mat::RTMat::Identity()}
            : inner_eigen_api_->get_transformation_matrix(graph_state_->robot_name, lidar_node_name,
                                                          proc_ts, "RT", interp);
    // ── THE POSE THIS CLOUD IS ACTUALLY REGISTERED WITH ─────────────────────────────────────────
    // Recorded so it can be compared against the pose the ICON is drawn at. The two are supposed to be
    // the same instant — both are pinned to proc_ts — but "supposed to" is what an instrument is for.
    // room←robot = room←lidar · (robot←lidar)^-1, i.e. the mount divided out, so this is directly
    // comparable with the robot pose the display uses. ★A CONSTANT offset between them is expected
    // (this is a room YAW, the display carries the footprint's theta convention); the VARIATION with
    // rotation is the quantity that would explain a cloud swinging off and snapping back.
    if (room_from_lidar.has_value() and robot_from_lidar.has_value())
    {
        const Eigen::Matrix4d room_T_robot =
            room_from_lidar->matrix() * robot_from_lidar->matrix().inverse();
        last_cloud_yaw_ = static_cast<float>(std::atan2(room_T_robot(1, 0), room_T_robot(0, 0)));
        last_cloud_ts_ = proc_ts;
    }

    if (!room_from_lidar.has_value() || !robot_from_lidar.has_value())
        return false;

    const std::size_t raw_count = std::min({proc_xs.size(), proc_ys.size(), proc_zs.size()});
    if (raw_count == 0)
        return false;

    // The leading-edge clamp is already repaired: the freshest scan is normally NEWER than the
    // freshest pose block (the pose is computed FROM a scan, so it can only arrive after one), which
    // is precisely when a plain interpolated query silently returns the newest block instead — and
    // rt_twist_fix_dt_ms_, set by the query above, is how far it had to walk to fix that.
    const Eigen::Matrix4d room_from_lidar_matrix = room_from_lidar->matrix();
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
                room_from_plane[k] = plane_T->matrix();
        }
    }
    // NOTE: the RoboComp Lidar3D source already returns points in the robot/base frame (floor ≈ z=0),
    // so the robot→lidar RT edge is deliberately identity; robot_z below is therefore height above the
    // floor. If someone ever moves the lidar node to its true mount height in shadow.json this shortcut
    // double-counts — keep the edge identity for this data convention.
    const bool robot_from_lidar_is_identity = robot_from_lidar_matrix.isApprox(Eigen::Matrix4d::Identity(), 1e-5);

    const float min_h = params_->temporary_obstacle_min_height_m;   // legacy floor: only the raw-proximity count below uses it

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

    // ── RAW z DUMP for controller/tools/floor_band.py (Task 7): the next N sweeps, robot frame, pre band, pre
    // self-filter (the self flag is a column, so the script can drop them itself). Armed by the remote `zdump <n>`;
    // costs nothing when not armed. Written on this thread only; the arm is an atomic.
    if (z_dump_scans_left_.load(std::memory_order_relaxed) > 0)
    {
        if (not z_dump_.is_open())
        {
            z_dump_scan_ = 0;
            z_dump_.open("lidar_z_dump.csv", std::ios::out | std::ios::trunc);
            z_dump_.imbue(std::locale::classic());   // decimal POINT under LANG=es_ES (CLAUDE.md)
            z_dump_ << "# controller zdump: robot-frame returns (z = height above floor), pre z-band, pre self-filter\n"
                       "scan,ts_ms,plane,x,y,z,self\n";
        }
        const auto &dump_body = self_body_.empty() ? body_.polygon() : self_body_;
        float dump_r = 0.f;
        for (const auto &v : dump_body) dump_r = std::max(dump_r, v.norm());
        const auto &M = robot_from_lidar_matrix;
        for (std::size_t i = 0; i < raw_count; ++i)
        {
            const double x = proc_xs[i], y = proc_ys[i], z = proc_zs[i];
            if (not (std::isfinite(x) and std::isfinite(y) and std::isfinite(z))) continue;
            const Eigen::Vector3d q = robot_from_lidar_is_identity
                ? Eigen::Vector3d(x, y, z)
                : Eigen::Vector3d(M(0, 0) * x + M(0, 1) * y + M(0, 2) * z + M(0, 3),
                                  M(1, 0) * x + M(1, 1) * y + M(1, 2) * z + M(1, 3),
                                  M(2, 0) * x + M(2, 1) * y + M(2, 2) * z + M(2, 3));
            const bool self = rc::lidar_capture_is_self(dump_body, dump_r, Eigen::Vector2f(float(q.x()), float(q.y())));
            z_dump_ << std::format("{},{},{},{:.4f},{:.4f},{:.4f},{}\n", z_dump_scan_, proc_ts,
                                   i < proc_plane_ids.size() ? int(proc_plane_ids[i]) : -1, q.x(), q.y(), q.z(), self ? 1 : 0);
        }
        ++z_dump_scan_;
        if (z_dump_scans_left_.fetch_sub(1, std::memory_order_relaxed) == 1)
        {
            z_dump_.close();
            std::println("[zdump] {} sweeps written to lidar_z_dump.csv", z_dump_scan_);
        }
        else z_dump_.flush();
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

    // ── TWO SETS FROM ONE CAPTURE (Task 7, lidar_height_band.h) ─────────────────────────────────────
    // The converter is built once per band: the LEGACY buffer keeps [TemporaryObstacleMinHeight, max] for every old
    // consumer, unchanged; the CONTACT buffer holds [Lidar.ContactMinHeightM, max] for the local policy. Same
    // registration, same self-filter — only the floor cut differs. The raw sweep is copied once for the second.
    const auto bands = rc::lidar_height_bands(*params_);
    const auto body_poly_now = self_body_.empty() ? body_.polygon() : self_body_;
    const float body_r_now = [&] { float r = 0.f; for (const auto &v : body_poly_now) r = std::max(r, v.norm()); return r; }();
    const auto make_converter = [&](bool policy_set, std::atomic<int> *self_dropped)
    {
        return
        [plane_coeffs, plane_sel = proc_plane_ids,
         robot_from_lidar_matrix, robot_from_lidar_is_identity, raw_count, bands, policy_set,
         body_poly = body_poly_now, body_r = body_r_now,
         self_dropped](rc::RawLidarPointVectors &&raw_points, rc::LidarPointVectors &room_points)
        {
            int n_self = 0;
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
                // Task 8: the helios's grazing far floor is cut from the POLICY set by source (lidar_height_band.h).
                const auto src = plane_sel.size() >= count ? rc::lidar_source_of_plane(int(plane_sel[index])) : rc::LidarSource::Unknown;
                if (const auto in = rc::lidar_band_membership(robot_z, bands, src); not (policy_set ? in.policy : in.legacy))
                    continue;
                // ★ Task 14 fix round 3 (try77): the robot's own returns are dropped HERE, in the rigid robot frame at
                //   capture (rc::lidar_capture_is_self), not by the reader after the cloud is room-frame — re-expressed on a
                //   later pose, a self return is displaced by the motion since capture and can land in the contact band.
                {
                    const Eigen::Vector2f q_robot = robot_from_lidar_is_identity
                        ? Eigen::Vector2f(x, y)
                        : Eigen::Vector2f(static_cast<float>(robot_from_lidar_matrix(0, 0) * x + robot_from_lidar_matrix(0, 1) * y
                                                             + robot_from_lidar_matrix(0, 2) * z + robot_from_lidar_matrix(0, 3)),
                                          static_cast<float>(robot_from_lidar_matrix(1, 0) * x + robot_from_lidar_matrix(1, 1) * y
                                                             + robot_from_lidar_matrix(1, 2) * z + robot_from_lidar_matrix(1, 3)));
                    if (rc::lidar_capture_is_self(body_poly, body_r, q_robot)) { ++n_self; continue; }
                }

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
            if (self_dropped != nullptr) self_dropped->store(n_self, std::memory_order_relaxed);
        };
    };
    contact_room_buffer_.put<0>(
        rc::RawLidarPointVectors{.xs = proc_xs, .ys = proc_ys, .zs = proc_zs},
        proc_ts, make_converter(/*policy_set=*/true, nullptr));
    lidar_room_buffer_.put<0>(
        rc::RawLidarPointVectors{
            .xs = std::move(proc_xs),
            .ys = std::move(proc_ys),
            .zs = std::move(proc_zs)},
        proc_ts, make_converter(/*policy_set=*/false, &capture_self_dropped_));

    // ── ONE BUFFER, BOTH CONSUMERS (the display buffer was removed 2026-09-09) ───────────────────
    // This used to fill a SECOND buffer with the newest scan for the viewer, because the control path
    // was holding its own scan back a frame and the overlay should not pay that frame of visible lag
    // (4-6 deg of cloud swing at 0.9 rad/s). With the hold gone the two buffers would carry the SAME
    // scan, and the display copy was the worse of the two: it registered the whole sweep with one
    // transform where the room buffer resolves room←lidar PER PLANE, which is the term that grows with
    // turn rate. So the viewer now reads lidar_buffer() directly and a full cloud transform + copy per
    // cycle goes away with it.

    return true;
}