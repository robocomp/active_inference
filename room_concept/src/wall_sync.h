#pragma once
// ── WALLS MOVE IN PLACE (room-transition protocol, rule 1, 2026-10-29) ─────────────────────────────────────────────
// room_concept never deletes a wall that carries another agent's node (door_concept hangs its doors from walls, the
// RT wall→door holding the door's offset along the wall). When the published layout changes, every existing wall is
// assigned to an edge of the new polygon and gets that edge's pose — SAME node, new RT — so whatever hangs from it
// moves with it. Live try32: room_3 was stored with walls 0–6 from an earlier small estimate standing mid-room while
// walls 7–15 traced the real outline; the doors hung from the stale ones came out 1.14 m off any wall and 90° wrong.
//
// The plan is pure (no graph) so it is pinned by tools/wall_update_test.cpp:
//   * one wall per edge, chosen greedily by a symmetric segment distance (position, orientation and extent all
//     enter, no gate: an edge with a free wall takes it however far it is — "far" is what moving fixes);
//   * walls carrying another agent's nodes choose FIRST, so a door's wall is never the one left without an edge;
//   * a door-carrying wall with no free edge left (more of them than edges) shares its nearest edge — never deleted;
//   * a childless wall with no edge is deleted; an edge with no wall gets a new one.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace rc::walls
{

// ⚠ PRE-EXISTING CONSTANT, not a new threshold: dsr_create_wall_nodes has always skipped polygon edges shorter than
//   this (a 0-length "wall" has no direction, so no yaw and no RT). Kept identical so the two paths agree.
inline constexpr float kMinWallLength = 0.1f;

struct WallPose { Eigen::Vector2f mid = Eigen::Vector2f::Zero(); float yaw = 0.f; float length = 0.f; bool has_children = false; };
struct EdgePose { Eigen::Vector2f mid = Eigen::Vector2f::Zero(); float yaw = 0.f; float length = 0.f; };

struct WallSyncPlan
{
    std::vector<int> target_edge;   ///< per existing wall: the polygon edge it is moved onto, or -1 = delete it
    std::vector<int> new_edges;     ///< edges no existing wall took: each gets a new wall
};

inline EdgePose edge_pose(const std::vector<Eigen::Vector2f>& poly, int i)
{
    const int n = static_cast<int>(poly.size());
    const Eigen::Vector2f& p0 = poly[i];
    const Eigen::Vector2f& p1 = poly[(i + 1) % n];
    const Eigen::Vector2f d = p1 - p0;
    return {(p0 + p1) * 0.5f, std::atan2(d.y(), d.x()), d.norm()};
}

namespace detail
{
inline float point_segment_d2(const Eigen::Vector2f& q, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f ab = b - a;
    const float L2 = ab.squaredNorm();
    const float t = L2 > 0.f ? std::clamp((q - a).dot(ab) / L2, 0.f, 1.f) : 0.f;
    return (q - (a + t * ab)).squaredNorm();
}
// Symmetric: each segment's endpoints against the other segment. Zero iff the segments coincide (either direction).
inline float segment_cost(const Eigen::Vector2f& a0, const Eigen::Vector2f& a1,
                          const Eigen::Vector2f& b0, const Eigen::Vector2f& b1)
{
    return point_segment_d2(a0, b0, b1) + point_segment_d2(a1, b0, b1)
         + point_segment_d2(b0, a0, a1) + point_segment_d2(b1, a0, a1);
}
}  // namespace detail

inline WallSyncPlan plan_wall_sync(const std::vector<WallPose>& walls, const std::vector<Eigen::Vector2f>& poly)
{
    WallSyncPlan P;
    P.target_edge.assign(walls.size(), -1);
    const int n = static_cast<int>(poly.size());
    std::vector<int> edges;   // the edges that can carry a wall
    for (int i = 0; i < n and n >= 3; ++i)
        if (edge_pose(poly, i).length >= kMinWallLength) edges.push_back(i);

    // cost[w][k] for wall w vs candidate edge edges[k]
    std::vector<std::vector<float>> cost(walls.size(), std::vector<float>(edges.size()));
    for (std::size_t w = 0; w < walls.size(); ++w)
    {
        const Eigen::Vector2f dir(std::cos(walls[w].yaw), std::sin(walls[w].yaw));
        const Eigen::Vector2f a0 = walls[w].mid - 0.5f * walls[w].length * dir;
        const Eigen::Vector2f a1 = walls[w].mid + 0.5f * walls[w].length * dir;
        for (std::size_t k = 0; k < edges.size(); ++k)
            cost[w][k] = detail::segment_cost(a0, a1, poly[edges[k]], poly[(edges[k] + 1) % n]);
    }

    // Greedy one-to-one: walls with children first, then by cost (index order breaks ties deterministically).
    std::vector<std::tuple<bool, float, std::size_t, std::size_t>> pairs;   // (childless, cost, wall, k)
    for (std::size_t w = 0; w < walls.size(); ++w)
        for (std::size_t k = 0; k < edges.size(); ++k)
            pairs.emplace_back(not walls[w].has_children, cost[w][k], w, k);
    std::ranges::sort(pairs);
    std::vector<bool> edge_taken(edges.size(), false);
    for (const auto& [childless, c, w, k] : pairs)
        if (P.target_edge[w] < 0 and not edge_taken[k])
        { P.target_edge[w] = edges[k]; edge_taken[k] = true; }

    // A wall carrying another agent's nodes is never deleted: with no free edge left it shares its nearest one.
    for (std::size_t w = 0; w < walls.size(); ++w)
        if (P.target_edge[w] < 0 and walls[w].has_children and not edges.empty())
            P.target_edge[w] = edges[std::ranges::distance(cost[w].begin(), std::ranges::min_element(cost[w]))];

    for (std::size_t k = 0; k < edges.size(); ++k)
        if (not edge_taken[k]) P.new_edges.push_back(edges[k]);
    return P;
}

}  // namespace rc::walls
