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

#ifndef ROOM_PATH_PLANNER_H
#define ROOM_PATH_PLANNER_H

#include <Eigen/Dense>

#include <optional>
#include <vector>

class RoomPathPlanner
{
public:
    using Polygon = std::vector<Eigen::Vector2f>;

    struct Params
    {
        float clearance_m = 0.4f;
        float grid_resolution_m = 0.35f;
        float connection_radius_m = 1.2f;
        float waypoint_tolerance_m = 0.25f;
    };

    struct PathPlan
    {
        std::vector<Eigen::Vector2f> room_path;
        std::vector<Eigen::Vector2f> graph_nodes;
    };

    Params params;

    std::vector<Eigen::Vector2f> compute_inner_polygon(const Polygon &polygon) const;
    std::optional<PathPlan> plan_path(const Polygon &room_polygon,
                                      const Polygon &inner_polygon,
                                      const Eigen::Vector2f &robot_pos,
                                      const Eigen::Vector2f &target_room_pos) const;

private:
    static constexpr float k_epsilon = 1e-4f;

    static float signed_area(const Polygon &polygon);
    static bool point_in_polygon(const Polygon &polygon, const Eigen::Vector2f &point);
    static float distance_to_segment(const Eigen::Vector2f &point,
                                     const Eigen::Vector2f &seg_a,
                                     const Eigen::Vector2f &seg_b);
    static float distance_to_polygon_edges(const Polygon &polygon, const Eigen::Vector2f &point);
    static bool is_clear_point(const Polygon &polygon, const Eigen::Vector2f &point, float clearance);
    static bool line_intersection(const Eigen::Vector2f &point_a,
                                  const Eigen::Vector2f &dir_a,
                                  const Eigen::Vector2f &point_b,
                                  const Eigen::Vector2f &dir_b,
                                  Eigen::Vector2f &intersection);
    static bool segment_is_navigable(const Polygon &polygon,
                                     const Eigen::Vector2f &from,
                                     const Eigen::Vector2f &to,
                                     float clearance,
                                     float sample_step);
    static Polygon deduplicate_points(Polygon points, float threshold = 0.05f);
};

#endif