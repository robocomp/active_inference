/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "room_path_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

float RoomPathPlanner::signed_area(const Polygon &polygon)
{
    if (polygon.size() < 3)
        return 0.f;

    float area = 0.f;
    for (std::size_t index = 0; index < polygon.size(); ++index)
    {
        const auto &current = polygon[index];
        const auto &next = polygon[(index + 1) % polygon.size()];
        area += current.x() * next.y() - next.x() * current.y();
    }
    return 0.5f * area;
}

bool RoomPathPlanner::point_in_polygon(const Polygon &polygon, const Eigen::Vector2f &point)
{
    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
    {
        const auto &a = polygon[i];
        const auto &b = polygon[j];
        const bool intersects = ((a.y() > point.y()) != (b.y() > point.y()))
            and (point.x() < (b.x() - a.x()) * (point.y() - a.y()) /
                std::max(k_epsilon, b.y() - a.y()) + a.x());
        if (intersects)
            inside = !inside;
    }
    return inside;
}

float RoomPathPlanner::distance_to_segment(const Eigen::Vector2f &point,
                                           const Eigen::Vector2f &seg_a,
                                           const Eigen::Vector2f &seg_b)
{
    const Eigen::Vector2f ab = seg_b - seg_a;
    const float denom = ab.squaredNorm();
    if (denom < k_epsilon)
        return (point - seg_a).norm();

    const float t = std::clamp((point - seg_a).dot(ab) / denom, 0.f, 1.f);
    const Eigen::Vector2f projection = seg_a + t * ab;
    return (point - projection).norm();
}

float RoomPathPlanner::distance_to_polygon_edges(const Polygon &polygon, const Eigen::Vector2f &point)
{
    float min_distance = std::numeric_limits<float>::max();
    for (std::size_t index = 0; index < polygon.size(); ++index)
    {
        const auto &a = polygon[index];
        const auto &b = polygon[(index + 1) % polygon.size()];
        min_distance = std::min(min_distance, distance_to_segment(point, a, b));
    }
    return min_distance;
}

bool RoomPathPlanner::is_clear_point(const Polygon &polygon, const Eigen::Vector2f &point, float clearance)
{
    return point_in_polygon(polygon, point) and distance_to_polygon_edges(polygon, point) >= clearance;
}

bool RoomPathPlanner::line_intersection(const Eigen::Vector2f &point_a,
                                        const Eigen::Vector2f &dir_a,
                                        const Eigen::Vector2f &point_b,
                                        const Eigen::Vector2f &dir_b,
                                        Eigen::Vector2f &intersection)
{
    const float cross = dir_a.x() * dir_b.y() - dir_a.y() * dir_b.x();
    if (std::abs(cross) < k_epsilon)
        return false;

    const Eigen::Vector2f delta = point_b - point_a;
    const float t = (delta.x() * dir_b.y() - delta.y() * dir_b.x()) / cross;
    intersection = point_a + t * dir_a;
    return true;
}

bool RoomPathPlanner::segment_is_navigable(const Polygon &polygon,
                                           const Eigen::Vector2f &from,
                                           const Eigen::Vector2f &to,
                                           float clearance,
                                           float sample_step)
{
    const float length = (to - from).norm();
    if (length < k_epsilon)
        return true;

    const int samples = std::max(2, static_cast<int>(std::ceil(length / std::max(sample_step, 0.05f))));
    for (int step = 1; step < samples; ++step)
    {
        const float ratio = static_cast<float>(step) / static_cast<float>(samples);
        const Eigen::Vector2f point = from + ratio * (to - from);
        if (!is_clear_point(polygon, point, clearance))
            return false;
    }
    return true;
}

RoomPathPlanner::Polygon RoomPathPlanner::deduplicate_points(Polygon points, float threshold)
{
    Polygon deduplicated;
    for (const auto &point : points)
    {
        const bool exists = std::ranges::any_of(deduplicated, [&](const auto &other)
        {
            return (point - other).norm() < threshold;
        });
        if (!exists)
            deduplicated.push_back(point);
    }
    return deduplicated;
}

std::vector<Eigen::Vector2f> RoomPathPlanner::compute_inner_polygon(const Polygon &polygon) const
{
    if (polygon.size() < 3)
        return {};

    const float orientation = signed_area(polygon);
    if (std::abs(orientation) < k_epsilon)
        return {};

    std::vector<Eigen::Vector2f> shifted_vertices;
    shifted_vertices.reserve(polygon.size());

    for (std::size_t index = 0; index < polygon.size(); ++index)
    {
        const Eigen::Vector2f prev = polygon[(index + polygon.size() - 1) % polygon.size()];
        const Eigen::Vector2f curr = polygon[index];
        const Eigen::Vector2f next = polygon[(index + 1) % polygon.size()];

        const Eigen::Vector2f edge_prev = (curr - prev).normalized();
        const Eigen::Vector2f edge_next = (next - curr).normalized();
        const Eigen::Vector2f normal_prev = orientation > 0.f
            ? Eigen::Vector2f(-edge_prev.y(), edge_prev.x())
            : Eigen::Vector2f(edge_prev.y(), -edge_prev.x());
        const Eigen::Vector2f normal_next = orientation > 0.f
            ? Eigen::Vector2f(-edge_next.y(), edge_next.x())
            : Eigen::Vector2f(edge_next.y(), -edge_next.x());

        Eigen::Vector2f intersection;
        if (!line_intersection(prev + normal_prev * params.clearance_m,
                               edge_prev,
                               curr + normal_next * params.clearance_m,
                               edge_next,
                               intersection))
            return {};

        shifted_vertices.push_back(intersection);
    }

    for (const auto &vertex : shifted_vertices)
    {
        if (!is_clear_point(polygon, vertex, params.clearance_m * 0.5f))
            return {};
    }

    return shifted_vertices;
}

std::optional<RoomPathPlanner::PathPlan> RoomPathPlanner::plan_path(const Polygon &room_polygon,
                                                                    const Polygon &inner_polygon,
                                                                    const Eigen::Vector2f &robot_pos,
                                                                    const Eigen::Vector2f &target_room_pos) const
{
    if (room_polygon.size() < 3)
        return std::nullopt;

    const Polygon &collision_polygon = room_polygon;
    Polygon nodes;
    nodes.push_back(robot_pos);
    nodes.push_back(target_room_pos);
    nodes.insert(nodes.end(), inner_polygon.begin(), inner_polygon.end());

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto &point : collision_polygon)
    {
        min_x = std::min(min_x, point.x());
        min_y = std::min(min_y, point.y());
        max_x = std::max(max_x, point.x());
        max_y = std::max(max_y, point.y());
    }

    for (float x = min_x; x <= max_x; x += params.grid_resolution_m)
        for (float y = min_y; y <= max_y; y += params.grid_resolution_m)
        {
            Eigen::Vector2f point{x, y};
            if (is_clear_point(collision_polygon, point, params.clearance_m))
                nodes.push_back(point);
        }

    nodes = deduplicate_points(std::move(nodes));
    if (nodes.size() < 2)
        return std::nullopt;

    auto nearest_navigable = [&](const Eigen::Vector2f &goal)
    {
        std::size_t best_index = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t index = 0; index < nodes.size(); ++index)
        {
            if (!is_clear_point(collision_polygon, nodes[index], params.clearance_m))
                continue;
            const float distance = (nodes[index] - goal).squaredNorm();
            if (distance < best_distance)
            {
                best_distance = distance;
                best_index = index;
            }
        }
        return best_index;
    };

    constexpr std::size_t start_index = 0;
    std::size_t goal_index = 1;
    if (!is_clear_point(collision_polygon, target_room_pos, params.clearance_m))
        goal_index = nearest_navigable(target_room_pos);

    std::vector<std::vector<std::pair<int, float>>> adjacency(nodes.size());
    for (std::size_t from = 0; from < nodes.size(); ++from)
        for (std::size_t to = from + 1; to < nodes.size(); ++to)
        {
            const float distance = (nodes[to] - nodes[from]).norm();
            if (distance > params.connection_radius_m and !(from < 2 or to < 2))
                continue;
            if (!segment_is_navigable(collision_polygon, nodes[from], nodes[to], params.clearance_m, params.grid_resolution_m * 0.5f))
                continue;
            adjacency[from].emplace_back(static_cast<int>(to), distance);
            adjacency[to].emplace_back(static_cast<int>(from), distance);
        }

    const auto heuristic = [&](int index)
    {
        return (nodes[static_cast<std::size_t>(index)] - nodes[goal_index]).norm();
    };

    std::vector<float> g_score(nodes.size(), std::numeric_limits<float>::max());
    std::vector<int> parent(nodes.size(), -1);
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> open_set;
    g_score[start_index] = 0.f;
    open_set.emplace(heuristic(static_cast<int>(start_index)), static_cast<int>(start_index));

    while (!open_set.empty())
    {
        const auto [_, current] = open_set.top();
        open_set.pop();
        if (current == static_cast<int>(goal_index))
            break;

        for (const auto &[neighbor, weight] : adjacency[static_cast<std::size_t>(current)])
        {
            const float tentative = g_score[static_cast<std::size_t>(current)] + weight;
            if (tentative >= g_score[static_cast<std::size_t>(neighbor)])
                continue;
            g_score[static_cast<std::size_t>(neighbor)] = tentative;
            parent[static_cast<std::size_t>(neighbor)] = current;
            open_set.emplace(tentative + heuristic(neighbor), neighbor);
        }
    }

    if (!std::isfinite(g_score[goal_index]))
        return std::nullopt;

    std::vector<Eigen::Vector2f> path;
    for (int current = static_cast<int>(goal_index); current >= 0; current = parent[static_cast<std::size_t>(current)])
    {
        path.push_back(nodes[static_cast<std::size_t>(current)]);
        if (current == static_cast<int>(start_index))
            break;
    }
    std::reverse(path.begin(), path.end());
    if (path.empty())
        return std::nullopt;

    if ((path.back() - target_room_pos).norm() > params.waypoint_tolerance_m)
        path.push_back(target_room_pos);

    return PathPlan{.room_path = std::move(path), .graph_nodes = std::move(nodes)};
}