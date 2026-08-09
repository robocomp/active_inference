#include "controller_world_model.h"

#include <algorithm>
#include <cmath>

namespace
{
bool finite_target_info(const ControllerTargetInfo &info)
{
    return std::isfinite(info.room_pos.x())
        && std::isfinite(info.room_pos.y())
        && std::isfinite(info.yaw_rad)
        && std::isfinite(info.epistemic_gain);
}
}

void ControllerWorldModel::set_params(const ControllerParams *params)
{
    params_ = params;
}

void ControllerWorldModel::set_affordance_manager(rc::AffordanceManager *affordance_manager)
{
    affordance_manager_ = affordance_manager;
}

void ControllerWorldModel::set_dependencies(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI *inner_eigen_api)
{
    graph_ = std::move(graph);
    inner_eigen_api_ = inner_eigen_api;
    rt_api_ = graph_ ? graph_->get_rt_api() : nullptr;   // for get_edge_RT_covariance()
}

bool ControllerWorldModel::refresh_graph_state()
{
    if (!graph_)
        return false;

    if (graph_state_.room_name.empty())
    {
        if (const auto room_nodes = graph_->get_nodes_by_type("room"); !room_nodes.empty())
        {
            graph_state_.room_id = room_nodes.front().id();
            graph_state_.room_name = room_nodes.front().name();
        }
    }
    else if (!graph_->get_node(graph_state_.room_id).has_value())
    {
        graph_state_.room_id = 0;
        graph_state_.room_name.clear();
    }

    if (graph_state_.robot_name.empty())
    {
        if (const auto robot_nodes = graph_->get_nodes_by_type("robot"); !robot_nodes.empty())
        {
            graph_state_.robot_id = robot_nodes.front().id();
            graph_state_.robot_name = robot_nodes.front().name();
        }
    }
    else if (!graph_->get_node(graph_state_.robot_id).has_value())
    {
        graph_state_.robot_id = 0;
        graph_state_.robot_name.clear();
    }

    return graph_state_.ready();
}

std::optional<std::vector<Eigen::Vector2f>> ControllerWorldModel::read_room_polygon() const
{
    if (!graph_ || graph_state_.room_id == 0)
        return std::nullopt;

    auto room_node = graph_->get_node(graph_state_.room_id);
    if (!room_node.has_value())
        return std::nullopt;

    auto polygon_x = graph_->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
    auto polygon_y = graph_->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());
    if (!polygon_x.has_value() || !polygon_y.has_value())
        return std::nullopt;

    const auto &xs = polygon_x.value().get();
    const auto &ys = polygon_y.value().get();
    const std::size_t count = std::min(xs.size(), ys.size());
    if (count < 3)
        return std::nullopt;

    std::vector<Eigen::Vector2f> polygon;
    polygon.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        polygon.emplace_back(xs[index], ys[index]);

    return polygon;
}

std::optional<ControllerRobotPose> ControllerWorldModel::read_robot_pose_in_room(
    std::uint64_t timestamp_ms,
    const std::optional<std::uint64_t> &last_lidar_timestamp_ms) const
{
    if (!inner_eigen_api_ || !graph_state_.ready() || !params_)
        return std::nullopt;

    const auto time_query = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                    : DSR::RT_API::TimeQuery::Nearest;
    const std::uint64_t query_ts = last_lidar_timestamp_ms.value_or(timestamp_ms);
    auto room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
                                                                    graph_state_.robot_name,
                                                                    query_ts,
                                                                    "RT",
                                                                    time_query);
    if (!room_T_robot.has_value())
        room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
                                                                   graph_state_.robot_name,
                                                                   timestamp_ms,
                                                                   "RT",
                                                                   time_query);
    if (!room_T_robot.has_value())
        room_T_robot = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name,
                                                                   graph_state_.robot_name,
                                                                   0,
                                                                   "RT",
                                                                   time_query);
    if (!room_T_robot.has_value())
        return std::nullopt;

    const auto &matrix = room_T_robot->matrix();

    ControllerRobotPose pose;
    pose.pos = Eigen::Vector2f(static_cast<float>(matrix(0, 3)), static_cast<float>(matrix(1, 3)));
    pose.theta = std::atan2(static_cast<float>(matrix(1, 0)), static_cast<float>(matrix(0, 0)));
    return pose;
}

std::optional<Eigen::Vector2f> ControllerWorldModel::read_node_room_xy(std::uint64_t node_id,
                                                                      std::uint64_t timestamp_ms) const
{
    if (!graph_ || !inner_eigen_api_ || !graph_state_.ready() || !params_)
        return std::nullopt;
    const auto node = graph_->get_node(node_id);
    if (!node.has_value())
        return std::nullopt;

    const auto time_query = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                    : DSR::RT_API::TimeQuery::Nearest;
    auto room_T_node = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name, node->name(),
                                                                   timestamp_ms, "RT", time_query);
    if (!room_T_node.has_value())
        room_T_node = inner_eigen_api_->get_transformation_matrix(graph_state_.room_name, node->name(),
                                                                  0, "RT", time_query);
    if (!room_T_node.has_value())
        return std::nullopt;

    const auto &m = room_T_node->matrix();
    return Eigen::Vector2f(static_cast<float>(m(0, 3)), static_cast<float>(m(1, 3)));
}

std::optional<std::uint64_t> ControllerWorldModel::pose_stamp_age_ms(std::uint64_t now_ms) const
{
    if (!graph_ || !graph_state_.ready())
        return std::nullopt;
    auto rt_edge = graph_->get_edge(graph_state_.robot_id, graph_state_.room_id, "RT");
    if (!rt_edge.has_value())
        rt_edge = graph_->get_edge(graph_state_.room_id, graph_state_.robot_id, "RT");
    if (!rt_edge.has_value())
        return std::nullopt;
    // The edge keeps a RING BUFFER of stamps (rt_timestamps) with rt_head_index pointing at the newest,
    // so the freshest stamp is NOT simply the last element. Fall back to the maximum when the index is
    // missing — a wrong pick here would silently report a whole buffer's worth of extra latency.
    const auto stamps = graph_->get_attrib_by_name<rt_timestamps_att>(rt_edge.value());
    if (!stamps.has_value()) return std::nullopt;
    const auto &v = stamps.value().get();
    if (v.empty()) return std::nullopt;
    std::uint64_t newest = 0;
    if (const auto head = graph_->get_attrib_by_name<rt_head_index_att>(rt_edge.value());
        head.has_value() and head.value() >= 0 and static_cast<std::size_t>(head.value()) < v.size())
        newest = v[static_cast<std::size_t>(head.value())];
    else
        newest = *std::max_element(v.begin(), v.end());
    if (newest == 0 or now_ms < newest) return std::nullopt;   // unstamped, or clocks disagree
    return now_ms - newest;
}

std::optional<ControllerPoseUncertainty> ControllerWorldModel::read_pose_uncertainty() const
{
    if (!graph_ || !graph_state_.ready())
        return std::nullopt;

    auto rt_edge = graph_->get_edge(graph_state_.robot_id, graph_state_.room_id, "RT");
    if (!rt_edge.has_value())
        rt_edge = graph_->get_edge(graph_state_.room_id, graph_state_.robot_id, "RT");
    if (!rt_edge.has_value())
        return std::nullopt;

    // ★READ IT THROUGH cortex's OWN ACCESSOR, not by hand-indexing the flat attribute.
    //
    // What this replaced was wrong twice over. rt_covariance is a 6x6 ROW-MAJOR SE3 covariance
    // [x,y,z,rx,ry,rz] (RT_COVARIANCE_BLOCK_SIZE = 36 in cortex's dsr_rt_api.cpp; the concept agents'
    // writers spell the same order out), so:
    //   · flat[0] and flat[7] ARE var_x and var_y — those two were right;
    //   · flat[14] = [2*6+2] is var_Z. Yaw is [5*6+5] = [35]. The rotation throttle was therefore being
    //     driven by the robot's HEIGHT uncertainty;
    //   · and the guard `size() < 15` waved through a 36-element block, and a whole RING of them, without
    //     noticing either. On edges cortex's RT api manages the attribute is a ring of blocks, so block 0
    //     is not necessarily the newest — the flat read could also be an arbitrarily stale covariance.
    //
    // RT_API::get_edge_RT_covariance() already handles all of that: ring vs single block, the head index,
    // and timestamp selection. Asking it is both correct and the only version that stays correct if the
    // packing changes.
    if (rt_api_ == nullptr)
        return std::nullopt;
    const auto cov = rt_api_->get_edge_RT_covariance(rt_edge.value());   // defaults: newest block, Nearest
    if (not cov.has_value())
        return std::nullopt;

    ControllerPoseUncertainty uncertainty;
    uncertainty.xy_std_m = static_cast<float>(std::sqrt(std::max(0.0, std::max((*cov)(0, 0), (*cov)(1, 1)))));
    uncertainty.theta_std_rad = static_cast<float>(std::sqrt(std::max(0.0, (*cov)(5, 5))));   // yaw, was (2,2)
    return uncertainty;
}

std::optional<ControllerTargetInfo> ControllerWorldModel::read_target_in_room(std::uint64_t timestamp_ms) const
{
    if (!graph_ || !inner_eigen_api_ || !graph_state_.ready() || !params_)
        return std::nullopt;

    if (affordance_manager_)
    {
        // Robot room-frame position feeds the grounded EFE nav-cost term in select_target.
        // Leave it unset (nullopt) until a live pose is available so the nav-cost is ignored,
        // rather than scoring affordances by their distance from the room origin.
        std::optional<Eigen::Vector2f> robot_pos;
        if (const auto pose = read_robot_pose_in_room(timestamp_ms, std::nullopt); pose.has_value())
            robot_pos = pose->pos;
        if (const auto affordance_target = affordance_manager_->select_target(graph_, robot_pos); affordance_target.has_value())
        {
            ControllerTargetInfo info;
            info.node_id = affordance_target->node_id;
            info.node_name = affordance_target->node_name;
            info.room_pos = affordance_target->room_pos;
            info.yaw_rad = affordance_target->yaw_rad;
            info.epistemic_gain = affordance_target->epistemic_gain;
            info.epistemic_pending = affordance_target->epistemic_pending;
            info.from_affordance = true;
            info.parent_node_id = affordance_target->parent_node_id;
            if (!finite_target_info(info))
                return std::nullopt;
            return info;
        }
    }

    const auto edges = graph_->get_edges_by_type(params_->target_edge_type);
    if (edges.empty())
        return std::nullopt;

    const auto time_query = params_->interpolate_rt ? DSR::RT_API::TimeQuery::Interpolated
                                                    : DSR::RT_API::TimeQuery::Nearest;
    for (const auto &edge : edges)
    {
        std::uint64_t target_id = edge.to();
        if (edge.to() == graph_state_.room_id || edge.to() == graph_state_.robot_id)
            target_id = edge.from();
        else if ((edge.from() == graph_state_.room_id || edge.from() == graph_state_.robot_id) && edge.to() != 0)
            target_id = edge.to();

        if (target_id == 0 || target_id == graph_state_.room_id || target_id == graph_state_.robot_id)
            continue;

        const auto target_name = graph_->get_name_from_id(target_id);
        if (!target_name.has_value())
            continue;

        const auto translation = inner_eigen_api_->transform(graph_state_.room_name,
                                                             *target_name,
                                                             timestamp_ms,
                                                             "RT",
                                                             time_query);
        if (!translation.has_value())
            continue;

        ControllerTargetInfo info;
        info.node_id = target_id;
        info.node_name = *target_name;
        info.room_pos = Eigen::Vector2f(static_cast<float>(translation->x()), static_cast<float>(translation->y()));
        info.from_affordance = false;
        if (!finite_target_info(info))
            continue;
        return info;
    }

    return std::nullopt;
}

bool ControllerWorldModel::same_target_instance(const ControllerTargetInfo &lhs, const ControllerTargetInfo &rhs)
{
    constexpr float pos_eps_m = 0.05f;
    constexpr float yaw_eps_rad = 0.05f;
    constexpr float gain_eps = 1e-3f;

    return lhs.node_id == rhs.node_id
        && lhs.from_affordance == rhs.from_affordance
        && lhs.epistemic_pending == rhs.epistemic_pending
        && (lhs.room_pos - rhs.room_pos).cwiseAbs().maxCoeff() < pos_eps_m
        && std::abs(lhs.yaw_rad - rhs.yaw_rad) < yaw_eps_rad
        && std::abs(lhs.epistemic_gain - rhs.epistemic_gain) < gain_eps;
}