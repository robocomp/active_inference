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
    using Polygons = std::vector<Polygon>;

    struct Params
    {
        float clearance_m = 0.4f;
        float robot_width_m = 0.4f;
        float grid_resolution_m = 0.35f;
        float connection_radius_m = 1.2f;
        float path_sample_spacing_m = 0.15f;
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
                                      const Polygons &obstacle_polygons,
                                      const Eigen::Vector2f &robot_pos,
                                      const Eigen::Vector2f &target_room_pos) const;

private:
    static constexpr float k_epsilon = 1e-5f;

    // Point-in-polygon (ray casting)
    static bool point_in_polygon(const Polygon &poly, const Eigen::Vector2f &p);

    // Proper (non-touching) segment-segment intersection
    static bool segments_intersect_proper(const Eigen::Vector2f &a1, const Eigen::Vector2f &a2,
                                          const Eigen::Vector2f &b1, const Eigen::Vector2f &b2);

    // Insert extra vertices on long edges
    static Polygon subdivide_polygon(const Polygon &poly, float max_edge_len);

    // Sampling-based inward offset (robust at any polygon shape, including concave)
    static Polygon offset_polygon_inward(const Polygon &poly, float offset);

    // Sampling-based outward offset (Minkowski expansion for obstacles)
    static Polygon offset_polygon_outward(const Polygon &poly, float offset);

    // Densify a path to a given spacing
    static Polygon densify_path(const Polygon &path, float spacing);
};

#endif