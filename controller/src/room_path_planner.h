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
#include <string>
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

    // Why the last plan_path() returned nullopt. plan_path has several distinct ways to fail and they want
    // completely different responses — a robot standing inside an obstacle polygon, a target inside one, and a
    // genuinely disconnected free space are not the same fault. Without this the caller can only report "no
    // path", which is exactly what sent one round of debugging chasing a performance red herring while the
    // real cause was a precondition failing in the first few lines.
    const std::string &last_failure() const { return last_failure_; }

    std::vector<Eigen::Vector2f> compute_inner_polygon(const Polygon &polygon) const;
    std::optional<PathPlan> plan_path(const Polygon &room_polygon,
                                      const Polygon &inner_polygon,
                                      const Polygons &obstacle_polygons,
                                      const Eigen::Vector2f &robot_pos,
                                      const Eigen::Vector2f &target_room_pos) const;

    // If `target` is blocked (outside the navigable region or inside an inflated obstacle), return the
    // nearest free point via a small outward ring search; returns `target` unchanged when it is already
    // free, or nullopt if nothing free is found within the search radius. Uses the SAME free-space
    // definition as plan_path, so a repaired target is guaranteed plannable. Lets the controller rescue
    // an affordance viewpoint that fell inside an obstacle footprint (which would otherwise stall).
    // `min_clearance_m` is the clearance the LOCAL controller will still enforce at the goal — pass
    // TrajectoryController::goal_clearance_requirement(). Repairing to anything tighter produces a target the
    // robot can never settle on. <=0 falls back to the planner's own half-robot-width.
    std::optional<Eigen::Vector2f> repair_target(const Polygon &room_polygon,
                                                 const Polygon &inner_polygon,
                                                 const Polygons &obstacle_polygons,
                                                 const Eigen::Vector2f &target,
                                                 float min_clearance_m = 0.f) const;

private:
    mutable std::string last_failure_;   // set by plan_path on every nullopt return; see last_failure()

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