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
#include <format>
#include <cmath>
#include <limits>
#include <queue>

// =====================================================================
// File-local helper: distance from point to nearest edge of a polygon
// =====================================================================
static float dist_to_boundary(const Eigen::Vector2f &p,
                               const std::vector<Eigen::Vector2f> &poly)
{
    float best = std::numeric_limits<float>::max();
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i)
    {
        const auto &a = poly[i];
        const auto &b = poly[(i + 1) % n];
        const Eigen::Vector2f ab = b - a;
        const float len_sq = ab.squaredNorm();
        const float t = len_sq > 1e-10f
            ? std::clamp((p - a).dot(ab) / len_sq, 0.f, 1.f)
            : 0.f;
        best = std::min(best, (p - (a + t * ab)).norm());
    }
    return best;
}

// =====================================================================
// Point-in-polygon — ray casting (reference implementation)
// =====================================================================
bool RoomPathPlanner::point_in_polygon(const Polygon &poly, const Eigen::Vector2f &p)
{
    bool inside = false;
    const int n = static_cast<int>(poly.size());
    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        const auto &vi = poly[i];
        const auto &vj = poly[j];
        if (((vi.y() > p.y()) != (vj.y() > p.y())) &&
            (p.x() < (vj.x() - vi.x()) * (p.y() - vi.y()) / (vj.y() - vi.y()) + vi.x()))
            inside = !inside;
    }
    return inside;
}

// =====================================================================
// Proper segment-segment intersection (endpoints excluded)
// =====================================================================
bool RoomPathPlanner::segments_intersect_proper(const Eigen::Vector2f &a1,
                                                const Eigen::Vector2f &a2,
                                                const Eigen::Vector2f &b1,
                                                const Eigen::Vector2f &b2)
{
    auto cross2 = [](const Eigen::Vector2f &u, const Eigen::Vector2f &v)
    { return u.x() * v.y() - u.y() * v.x(); };

    const Eigen::Vector2f d1 = a2 - a1, d2 = b2 - b1, d3 = b1 - a1;
    const float denom = cross2(d1, d2);
    if (std::abs(denom) < 1e-10f) return false;
    const float t = cross2(d3, d2) / denom;
    const float u = cross2(d3, d1) / denom;
    return (t > k_epsilon && t < 1.f - k_epsilon &&
            u > k_epsilon && u < 1.f - k_epsilon);
}

// =====================================================================
// Subdivide polygon: insert extra vertices on edges longer than max_edge_len
// =====================================================================
RoomPathPlanner::Polygon RoomPathPlanner::subdivide_polygon(const Polygon &poly, float max_edge_len)
{
    const int n = static_cast<int>(poly.size());
    if (n < 3) return poly;

    Polygon result;
    result.reserve(n * 3);
    for (int i = 0; i < n; ++i)
    {
        const auto &a = poly[i];
        const auto &b = poly[(i + 1) % n];
        result.push_back(a);
        const float len = (b - a).norm();
        if (len > max_edge_len)
        {
            const int segs = static_cast<int>(std::ceil(len / max_edge_len));
            for (int s = 1; s < segs; ++s)
                result.push_back(a + (static_cast<float>(s) / segs) * (b - a));
        }
    }
    return result;
}

// =====================================================================
// Sampling-based inward polygon offset (robust at concave corners).
// For each vertex, sweeps 36 candidate directions at `offset` radius and
// picks the interior candidate with maximum wall clearance.
// Falls back to shorter radii for tight corners.
// =====================================================================
RoomPathPlanner::Polygon RoomPathPlanner::offset_polygon_inward(const Polygon &poly, float offset)
{
    const int n = static_cast<int>(poly.size());
    if (n < 3) return {};

    Polygon result(n);
    constexpr int num_angles = 36;

    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector2f best = poly[i];
        float best_clearance = 0.f;

        for (int a = 0; a < num_angles; ++a)
        {
            const float angle = 2.f * static_cast<float>(M_PI) * a / num_angles;
            const Eigen::Vector2f cand = poly[i] + Eigen::Vector2f(std::cos(angle), std::sin(angle)) * offset;
            if (!point_in_polygon(poly, cand)) continue;
            const float c = dist_to_boundary(cand, poly);
            if (c > best_clearance) { best = cand; best_clearance = c; }
        }

        // Fallback: shorter radii when offset doesn't fit
        if (best_clearance < 0.01f)
        {
            for (int a = 0; a < num_angles; ++a)
            {
                const float angle = 2.f * static_cast<float>(M_PI) * a / num_angles;
                const Eigen::Vector2f dir(std::cos(angle), std::sin(angle));
                for (int r = 9; r >= 1; --r)
                {
                    const Eigen::Vector2f cand = poly[i] + dir * (offset * r / 10.f);
                    if (!point_in_polygon(poly, cand)) continue;
                    const float c = dist_to_boundary(cand, poly);
                    if (c > best_clearance) { best = cand; best_clearance = c; }
                }
            }
        }
        result[i] = best;
    }
    return result;
}

// =====================================================================
// Sampling-based outward polygon offset (Minkowski expansion for obstacles).
// For each vertex, sweeps 36 candidate directions and picks the exterior
// candidate with maximum clearance from the obstacle boundary.
// =====================================================================
RoomPathPlanner::Polygon RoomPathPlanner::offset_polygon_outward(const Polygon &poly, float offset)
{
    const int n = static_cast<int>(poly.size());
    if (n < 3) return poly;

    Polygon result(n);
    constexpr int num_angles = 36;

    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector2f best = poly[i];
        float best_clearance = -1.f;

        for (int a = 0; a < num_angles; ++a)
        {
            const float angle = 2.f * static_cast<float>(M_PI) * a / num_angles;
            const Eigen::Vector2f cand = poly[i] + Eigen::Vector2f(std::cos(angle), std::sin(angle)) * offset;
            if (point_in_polygon(poly, cand)) continue; // must be OUTSIDE obstacle
            const float c = dist_to_boundary(cand, poly);
            if (c > best_clearance) { best = cand; best_clearance = c; }
        }

        if (best_clearance < 0.01f)
        {
            for (int a = 0; a < num_angles; ++a)
            {
                const float angle = 2.f * static_cast<float>(M_PI) * a / num_angles;
                const Eigen::Vector2f dir(std::cos(angle), std::sin(angle));
                for (int r = 9; r >= 1; --r)
                {
                    const Eigen::Vector2f cand = poly[i] + dir * (offset * r / 10.f);
                    if (point_in_polygon(poly, cand)) continue;
                    const float c = dist_to_boundary(cand, poly);
                    if (c > best_clearance) { best = cand; best_clearance = c; }
                }
            }
        }
        result[i] = best;
    }
    return result;
}

// =====================================================================
// Densify a path to a given point spacing
// =====================================================================
RoomPathPlanner::Polygon RoomPathPlanner::densify_path(const Polygon &path, float spacing)
{
    if (path.size() < 2) return path;
    const float sp = std::max(spacing, 0.05f);
    Polygon dense;
    dense.push_back(path.front());
    for (std::size_t i = 0; i + 1 < path.size(); ++i)
    {
        const Eigen::Vector2f seg = path[i + 1] - path[i];
        const float len = seg.norm();
        const int nsamp = std::max(1, static_cast<int>(std::ceil(len / sp)));
        for (int s = 1; s <= nsamp; ++s)
            dense.push_back(path[i] + (static_cast<float>(s) / nsamp) * seg);
    }
    return dense;
}

// =====================================================================
// compute_inner_polygon — sampling-based, robust for any room shape.
// Subdivides long edges first to get more nav nodes along straight walls.
// =====================================================================
std::vector<Eigen::Vector2f> RoomPathPlanner::compute_inner_polygon(const Polygon &polygon)  const
{
    if (polygon.size() < 3) return {};
    // Subdivide to add intermediate nodes on long edges
    const float max_edge = std::max(params.clearance_m * 2.f, 0.5f);
    const Polygon sub = subdivide_polygon(polygon, max_edge);
    return offset_polygon_inward(sub, params.clearance_m);
}

// =====================================================================
// plan_path — visibility-graph A* (reference approach):
//   1. Expand obstacles (Minkowski)
//   2. Build nav nodes from subdivided inner polygon + obstacle bypass nodes
//   3. Build visibility graph with proper segment-intersection checks
//   4. Add robot_pos and target temporarily as nodes n and n+1
//   5. Dijkstra, reconstruct, densify
// =====================================================================
std::optional<RoomPathPlanner::PathPlan> RoomPathPlanner::plan_path(
    const Polygon &room_polygon,
    const Polygon &inner_polygon,
    const Polygons &obstacle_polygons,
    const Eigen::Vector2f &robot_pos,
    const Eigen::Vector2f &target_room_pos) const
{
    last_failure_.clear();
    if (room_polygon.size() < 3)
    { last_failure_ = "room polygon has <3 vertices (room node not read yet?)"; return std::nullopt; }

    // ----- Expand obstacles outward by half the robot width -----
    const float obs_clearance = std::max(0.01f, params.robot_width_m * 0.5f);
    Polygons expanded_obs;
    expanded_obs.reserve(obstacle_polygons.size());
    for (const auto &obs : obstacle_polygons)
        if (obs.size() >= 3)
            expanded_obs.push_back(offset_polygon_outward(obs, obs_clearance));

    // ----- Helpers -----
    // START-IN-COLLISION RECOVERY. The robot's own position can legitimately end up inside an obstacle
    // polygon — margins from different owners stack (residual excludes a 0.45 m disc around the robot, then
    // grows the set back 0.25 m by C-space dilation, then plan_path expands it another 0.20 m here, then the
    // publish lattice rounds outward), and the sum can reach the robot centre. Treating that as unplannable
    // BRICKS the robot: no edge can leave the start, so there is never a route, from anywhere, until the
    // obstacle set happens to change. A planner should instead route OUT.
    // So obstacles that contain the start are ignored for edges incident to the start ONLY. They keep their
    // full effect everywhere else, so the path may leave the enclosing region but may not cut through any
    // other obstacle, and nothing else in the graph is weakened.
    std::vector<const Polygon *> start_enclosing;
    for (const auto &obs : expanded_obs)
        if (point_in_polygon(obs, robot_pos)) start_enclosing.push_back(&obs);
    for (const auto &obs : obstacle_polygons)
        if (point_in_polygon(obs, robot_pos)) start_enclosing.push_back(&obs);
    const bool start_in_collision = not start_enclosing.empty();
    auto ignored_for_start = [&](const Polygon *poly) -> bool
    {
        for (const auto *p : start_enclosing) if (p == poly) return true;
        return false;
    };

    auto inside_any_obs = [&](const Eigen::Vector2f &p, bool relax_start = false) -> bool
    {
        for (const auto &obs : expanded_obs)
            if ((!relax_start or !ignored_for_start(&obs)) and point_in_polygon(obs, p)) return true;
        for (const auto &obs : obstacle_polygons)
            if ((!relax_start or !ignored_for_start(&obs)) and point_in_polygon(obs, p)) return true;
        return false;
    };

    auto seg_crosses_poly = [&](const Eigen::Vector2f &a,
                                const Eigen::Vector2f &b,
                                const Polygon &poly) -> bool
    {
        const int n = static_cast<int>(poly.size());
        for (int i = 0; i < n; ++i)
            if (segments_intersect_proper(a, b, poly[i], poly[(i + 1) % n]))
                return true;
        return false;
    };

    // inner_boundary used only as a soft wall constraint (keeps path off walls)
    const bool have_inner = inner_polygon.size() >= 3;

    // Visibility: reference approach — proper intersection, no sampling false-negatives
    auto is_visible = [&](const Eigen::Vector2f &a, const Eigen::Vector2f &b, bool relax_start = false) -> bool
    {
        if (!point_in_polygon(room_polygon, a) || !point_in_polygon(room_polygon, b))
            return false;
        if (inside_any_obs(a, relax_start) || inside_any_obs(b, relax_start)) return false;
        // Midpoint check (catches pass-through)
        if (inside_any_obs(0.5f * (a + b), relax_start)) return false;
        // Must not cross room boundary
        if (seg_crosses_poly(a, b, room_polygon)) return false;
        // Must not cross inner polygon boundary (wall clearance constraint)
        // Relaxed for start edges too: an enclosed start is often enclosed precisely because it sits in the
        // wall-clearance ring, and the escape leg has to be allowed to cross it.
        if (have_inner && !relax_start && seg_crosses_poly(a, b, inner_polygon)) return false;
        // Must not cross any obstacle boundary
        for (const auto &obs : expanded_obs)
            if ((!relax_start or !ignored_for_start(&obs)) and seg_crosses_poly(a, b, obs)) return false;
        for (const auto &obs : obstacle_polygons)
            if ((!relax_start or !ignored_for_start(&obs)) and seg_crosses_poly(a, b, obs)) return false;
        return true;
    };

    // ----- Sanity: robot must be inside the room, and not standing in an obstacle -----
    // These two preconditions abort BEFORE any search, so they look identical to "no route" from outside
    // while having nothing to do with connectivity. Distinguishing them matters: a robot reported inside an
    // obstacle means the obstacle set has swallowed the robot's own position, which is a perception/inflation
    // bug, not a planning one — and no amount of replanning will ever clear it.
    if (!point_in_polygon(room_polygon, robot_pos))
    {
        last_failure_ = std::format("robot ({:.2f},{:.2f}) is OUTSIDE the room polygon", robot_pos.x(), robot_pos.y());
        return std::nullopt;
    }
    // NOTE: a start inside an obstacle is NOT an abort any more — see start-in-collision recovery above.
    // It is still worth surfacing, because it means some layer's geometry has covered the robot itself.
    if (!point_in_polygon(room_polygon, target_room_pos))
    {
        last_failure_ = std::format("target ({:.2f},{:.2f}) is OUTSIDE the room polygon",
                                    target_room_pos.x(), target_room_pos.y());
        return std::nullopt;
    }
    if (inside_any_obs(target_room_pos))
    {
        last_failure_ = std::format("target ({:.2f},{:.2f}) is INSIDE an obstacle polygon (repair_target did not "
                                    "move it clear)", target_room_pos.x(), target_room_pos.y());
        return std::nullopt;
    }

    // ----- Direct line-of-sight shortcut -----
    if (is_visible(robot_pos, target_room_pos, start_in_collision))
    {
        Polygon direct{robot_pos, target_room_pos};
        return PathPlan{
            .room_path = densify_path(direct, params.path_sample_spacing_m),
            .graph_nodes = {robot_pos, target_room_pos}
        };
    }

    // ----- Build nav nodes -----
    // Use subdivided inner polygon (or room polygon offset-inward on the fly)
    const Polygon &nav_boundary = have_inner ? inner_polygon : room_polygon;
    const float max_edge = std::max(params.clearance_m * 2.f, 0.8f);
    Polygon nodes = subdivide_polygon(nav_boundary, max_edge);

    // Add bypass nodes around each obstacle vertex
    constexpr int num_angles = 24;
    for (std::size_t oi = 0; oi < obstacle_polygons.size(); ++oi)
    {
        const auto &obs = obstacle_polygons[oi];
        if (obs.size() < 3) continue;
        for (const auto &vertex : obs)
        {
            Eigen::Vector2f best = vertex;
            float best_c = -1.f;
            for (int a = 0; a < num_angles; ++a)
            {
                const float angle = 2.f * static_cast<float>(M_PI) * a / num_angles;
                const Eigen::Vector2f cand =
                    vertex + Eigen::Vector2f(std::cos(angle), std::sin(angle)) * obs_clearance * 1.3f;
                if (!point_in_polygon(room_polygon, cand)) continue;
                if (inside_any_obs(cand)) continue;
                const float c = dist_to_boundary(cand, obs);
                if (c > best_c) { best = cand; best_c = c; }
            }
            if (best_c > 0.f) nodes.push_back(best);
        }
    }

    if (nodes.empty())
    { last_failure_ = "no navigation nodes (inner polygon empty and no obstacle bypass nodes)"; return std::nullopt; }

    // ----- Visibility graph over nav nodes (static part) -----
    const int n = static_cast<int>(nodes.size());
    struct Edge { int to; float cost; };
    std::vector<std::vector<Edge>> adj(n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (is_visible(nodes[i], nodes[j]))
            {
                const float c = (nodes[i] - nodes[j]).norm();
                adj[i].push_back({j, c});
                adj[j].push_back({i, c});
            }

    // ----- Temporarily add robot_pos (si) and target_room_pos (gi) -----
    const int si = n, gi = n + 1;
    adj.resize(n + 2);
    for (int i = 0; i < n; ++i)
    {
        if (is_visible(robot_pos, nodes[i], start_in_collision))
        {
            const float c = (robot_pos - nodes[i]).norm();
            adj[si].push_back({i, c});
            adj[i].push_back({si, c});
        }
        if (is_visible(target_room_pos, nodes[i]))
        {
            const float c = (target_room_pos - nodes[i]).norm();
            adj[gi].push_back({i, c});
            adj[i].push_back({gi, c});
        }
    }

    // ----- Dijkstra from si to gi -----
    const int total = n + 2;
    std::vector<float> dist_v(total, std::numeric_limits<float>::infinity());
    std::vector<int> parent(total, -1);
    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<>> pq;
    dist_v[si] = 0.f;
    pq.push({0.f, si});

    while (!pq.empty())
    {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist_v[u]) continue;
        if (u == gi) break;
        for (const auto &[to, cost] : adj[u])
        {
            const float nd = d + cost;
            if (nd < dist_v[to])
            {
                dist_v[to] = nd;
                parent[to] = u;
                pq.push({nd, to});
            }
        }
    }

    if (parent[gi] == -1)
    {
        // Reached the search and lost. The degrees say WHICH end is walled off — if the robot or the target
        // has no visible nav node at all, free space is not disconnected, that endpoint is simply enclosed.
        last_failure_ = std::format("no route: {} nav nodes, robot sees {} of them, target sees {} "
                                    "({} obstacle polygons){}",
                                    n, adj[si].size(), adj[gi].size(), obstacle_polygons.size(),
                                    start_in_collision
                                        ? " [robot started INSIDE an obstacle; escape edges were allowed and "
                                          "still found nothing]" : "");
        return std::nullopt;
    }

    // ----- Reconstruct path -----
    std::vector<int> idx_path;
    for (int v = gi; v != -1; v = parent[v])
        idx_path.push_back(v);
    std::reverse(idx_path.begin(), idx_path.end());

    auto idx_to_pt = [&](int idx) -> Eigen::Vector2f
    {
        if (idx == si) return robot_pos;
        if (idx == gi) return target_room_pos;
        return nodes[idx];
    };

    Polygon path;
    path.reserve(idx_path.size());
    for (int idx : idx_path)
        path.push_back(idx_to_pt(idx));

    if (path.size() < 2) { last_failure_ = "reconstructed path has <2 points"; return std::nullopt; }

    // ----- Catmull-Rom spline smoothing -----
    // Smooth the raw waypoints; fall back to straight segment on visibility failure.
    if (path.size() >= 3)
    {
        const float sp = std::max(params.path_sample_spacing_m, 0.05f);
        Polygon smoothed;
        smoothed.push_back(path.front());

        for (std::size_t i = 0; i + 1 < path.size(); ++i)
        {
            const auto &p0 = path[i > 0 ? i - 1 : 0];
            const auto &p1 = path[i];
            const auto &p2 = path[i + 1];
            const auto &p3 = path[i + 2 < path.size() ? i + 2 : path.size() - 1];

            const int samples = std::max(1, static_cast<int>(
                std::ceil((p2 - p1).norm() / sp)));

            Polygon curve;
            curve.reserve(samples);
            bool valid = true;
            Eigen::Vector2f prev = p1;

            for (int s = 1; s <= samples; ++s)
            {
                const float t  = static_cast<float>(s) / samples;
                const float t2 = t * t, t3 = t2 * t;
                const Eigen::Vector2f pt = 0.5f * (
                    (2.f * p1) +
                    (-p0 + p2) * t +
                    (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t2 +
                    (-p0 + 3.f * p1 - 3.f * p2 + p3) * t3);
                if (!is_visible(prev, pt)) { valid = false; break; }
                curve.push_back(pt);
                prev = pt;
            }

            if (valid)
                smoothed.insert(smoothed.end(), curve.begin(), curve.end());
            else
                smoothed.push_back(p2); // fall back: keep original waypoint
        }

        if (smoothed.size() >= 2)
            path = std::move(smoothed);
    }

    // Collect all nav nodes for visualization
    Polygon all_nodes = nodes;
    all_nodes.push_back(robot_pos);
    all_nodes.push_back(target_room_pos);

    return PathPlan{
        .room_path = densify_path(path, params.path_sample_spacing_m),
        .graph_nodes = std::move(all_nodes)
    };
}

std::optional<Eigen::Vector2f> RoomPathPlanner::repair_target(
    const Polygon &room_polygon,
    const Polygon &inner_polygon,
    const Polygons &obstacle_polygons,
    const Eigen::Vector2f &target,
    float min_clearance_m) const
{
    if (room_polygon.size() < 3)
        return std::nullopt;

    // Free-space definition for a TARGET is NOT the same as for a path waypoint. A waypoint only has to be
    // traversable (half the robot width); a target has to be somewhere the local controller will let the robot
    // COME TO REST. That second number is larger — the MPPI keeps enforcing its near-goal d_safe right up to
    // the goal — and when the two disagree, repair "succeeds" onto a point the robot then hunts at forever,
    // never reaching it. So take the caller's min_clearance_m (the controller's own requirement) whenever it
    // is the stricter of the two.
    const float obs_clearance = std::max({0.01f, params.robot_width_m * 0.5f, min_clearance_m});
    Polygons expanded_obs;
    expanded_obs.reserve(obstacle_polygons.size());
    for (const auto &obs : obstacle_polygons)
        if (obs.size() >= 3)
            expanded_obs.push_back(offset_polygon_outward(obs, obs_clearance));

    const bool have_inner = inner_polygon.size() >= 3;
    auto is_free = [&](const Eigen::Vector2f &p) -> bool
    {
        if (!point_in_polygon(room_polygon, p)) return false;
        if (have_inner && !point_in_polygon(inner_polygon, p)) return false;
        for (const auto &obs : expanded_obs)      if (point_in_polygon(obs, p)) return false;
        for (const auto &obs : obstacle_polygons) if (point_in_polygon(obs, p)) return false;
        return true;
    };

    if (is_free(target))
        return target;   // already navigable — no change

    // Ring search outward; the first radius yielding any free candidate is the (angularly quantised)
    // nearest exit from the blocked region. Bounded so a hopeless target fails fast.
    const float step  = std::max(0.10f, params.grid_resolution_m * 0.5f);
    const float max_r  = 3.0f;
    constexpr int kAng = 24;
    for (float r = step; r <= max_r; r += step)
    {
        std::optional<Eigen::Vector2f> best;
        float best_d2 = std::numeric_limits<float>::max();
        for (int i = 0; i < kAng; ++i)
        {
            const float a = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / kAng;
            const Eigen::Vector2f cand = target + r * Eigen::Vector2f(std::cos(a), std::sin(a));
            if (!is_free(cand)) continue;
            const float d2 = (cand - target).squaredNorm();
            if (d2 < best_d2) { best_d2 = d2; best = cand; }
        }
        if (best.has_value())
            return best;
    }
    return std::nullopt;   // no free point within max_r → caller keeps original (planner then fails)
}

