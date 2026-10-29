// Task 7 (room-transition protocol, rule 1): walls MOVE IN PLACE, never deleted under a door.
// Pure plan: existing walls (pose + "carries another agent's node") vs the new polygon ⇒ which wall goes to which
// edge, which childless walls are deleted, which edges need a new wall. No graph.
//   make -C build wall_update_test && ../bin/wall_update_test
#include "wall_sync.h"
#include <cmath>
#include <cstdio>
#include <numbers>

using Eigen::Vector2f;
static int failures = 0;
static void check(bool ok, const char* what)
{ if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) <= tol; }
static bool near_angle(float a, float b, float tol = 1e-4f)
{ return std::abs(std::remainder(a - b, 2.f * std::numbers::pi_v<float>)) <= tol; }

static rc::walls::WallPose wall_on(const Vector2f& p0, const Vector2f& p1, bool children)
{
    const Vector2f d = p1 - p0;
    return {(p0 + p1) * 0.5f, std::atan2(d.y(), d.x()), d.norm(), children, {}};
}

int main()
{
    using rc::walls::plan_wall_sync;
    using rc::walls::edge_pose;
    const std::vector<Vector2f> rect = {{0, 0}, {4, 0}, {4, 3}, {0, 3}};

    std::printf("walls_follow_polygon\n");
    {
        // walls of a slightly different (shifted, smaller) rectangle, deliberately listed out of order
        const std::vector<Vector2f> old = {{0.2f, 0.1f}, {3.6f, 0.1f}, {3.6f, 2.7f}, {0.2f, 2.7f}};
        std::vector<rc::walls::WallPose> W = {wall_on(old[2], old[3], false), wall_on(old[0], old[1], false),
                                              wall_on(old[3], old[0], false), wall_on(old[1], old[2], false)};
        const auto P = plan_wall_sync(W, rect);
        check(P.target_edge == std::vector<int>({2, 0, 3, 1}), "each wall goes to its own edge (nearest)");
        check(P.new_edges.empty(), "no new walls");
        for (std::size_t w = 0; w < W.size(); ++w)
        {
            if (P.target_edge[w] < 0) { check(false, "wall kept"); continue; }
            const auto e = edge_pose(rect, P.target_edge[w]);
            const Vector2f p0 = rect[P.target_edge[w]], p1 = rect[(P.target_edge[w] + 1) % rect.size()];
            check(near(e.mid.x(), 0.5f * (p0.x() + p1.x())) and near(e.mid.y(), 0.5f * (p0.y() + p1.y()))
                      and near_angle(e.yaw, std::atan2(p1.y() - p0.y(), p1.x() - p0.x()))
                      and near(e.length, (p1 - p0).norm()),
                  "wall RT = its edge midpoint / yaw / length");
        }
    }

    std::printf("wall_with_child_survives_shrink (RF5)\n");
    {
        // An early, larger estimate: 6 walls (an L), wall 4 carries a door. The layout shrinks to the rectangle:
        // two walls have no edge left. The door's wall must SURVIVE (moved to its nearest edge); the childless
        // extra is deleted.
        const std::vector<Vector2f> L = {{0, 0}, {4, 0}, {4, 3}, {2, 3}, {2, 5}, {0, 5}};
        std::vector<rc::walls::WallPose> W;
        for (std::size_t i = 0; i < L.size(); ++i) W.push_back(wall_on(L[i], L[(i + 1) % L.size()], i == 4));
        const auto P = plan_wall_sync(W, rect);
        check(P.target_edge[4] >= 0, "the wall carrying a door is kept");
        check(P.target_edge[4] == 2, "...and moved onto its nearest edge (y = 3)");
        int deleted = 0;
        for (const int t : P.target_edge) deleted += (t < 0);
        check(deleted == 2, "two childless extras deleted");
        check(P.new_edges.empty(), "every edge has a wall");
        std::vector<int> per_edge(rect.size(), 0);
        for (const int t : P.target_edge) if (t >= 0) ++per_edge[t];
        check(std::ranges::all_of(per_edge, [](int c) { return c == 1; }), "one wall per edge");
    }

    std::printf("wall_with_child_survives_even_without_a_free_edge\n");
    {
        // More door-carrying walls than edges: none may be deleted; the surplus shares its nearest edge.
        const std::vector<Vector2f> tri = {{0, 0}, {4, 0}, {0, 3}};
        std::vector<rc::walls::WallPose> W;
        for (std::size_t i = 0; i < rect.size(); ++i) W.push_back(wall_on(rect[i], rect[(i + 1) % rect.size()], true));
        const auto P = plan_wall_sync(W, tri);
        check(std::ranges::all_of(P.target_edge, [](int t) { return t >= 0; }), "no wall with children deleted");
    }

    std::printf("door_wall_matched_by_its_door_not_its_stale_segment\n");
    {
        // try32: a wall left from an early small estimate stands mid-room (y = 0.5, nearest the y = 0 edge), but the
        // door hanging from it was seen in the real y = 3 wall. The wall must go where its door is.
        std::vector<rc::walls::WallPose> W;
        for (std::size_t i = 0; i < rect.size(); ++i) W.push_back(wall_on(rect[i], rect[(i + 1) % rect.size()], false));
        auto stale = wall_on({0, 0.5f}, {4, 0.5f}, true);
        stale.children_xy = {{2.f, 3.f}};
        W.insert(W.begin(), stale);
        const auto P = plan_wall_sync(W, rect);
        check(P.target_edge[0] == 2, "the door's wall goes to the edge its door is on (y = 3)");
        check(P.target_edge[1] == 0, "the y = 0 edge keeps its own wall");
        int deleted = 0;
        for (const int t : P.target_edge) deleted += (t < 0);
        check(deleted == 1 and P.target_edge[3] < 0, "the childless wall that lost y = 3 is deleted");
    }

    std::printf("childless_extra_wall_deleted\n");
    {
        const std::vector<Vector2f> L = {{0, 0}, {4, 0}, {4, 3}, {2, 3}, {2, 5}, {0, 5}};
        std::vector<rc::walls::WallPose> W;
        for (std::size_t i = 0; i < L.size(); ++i) W.push_back(wall_on(L[i], L[(i + 1) % L.size()], false));
        const auto P = plan_wall_sync(W, rect);
        int deleted = 0;
        for (const int t : P.target_edge) deleted += (t < 0);
        check(deleted == 2, "6 childless walls onto 4 edges: 2 deleted");
        check(P.target_edge[0] == 0 and P.target_edge[1] == 1, "the coincident walls keep their edges");
    }

    std::printf("new_edges_get_new_walls\n");
    {
        const std::vector<Vector2f> L = {{0, 0}, {4, 0}, {4, 3}, {2, 3}, {2, 5}, {0, 5}};
        std::vector<rc::walls::WallPose> W;
        for (std::size_t i = 0; i < rect.size(); ++i) W.push_back(wall_on(rect[i], rect[(i + 1) % rect.size()], false));
        const auto P = plan_wall_sync(W, L);
        check(std::ranges::all_of(P.target_edge, [](int t) { return t >= 0; }), "every old wall kept");
        check(P.new_edges.size() == 2, "two new edges get new walls");
    }

    std::printf("degenerate_edge_gets_no_wall\n");
    {
        const std::vector<Vector2f> d = {{0, 0}, {4, 0}, {4, 0.05f}, {4, 3}, {0, 3}};
        const auto P = plan_wall_sync({}, d);
        check(P.new_edges.size() == 4, "a 5 cm edge is not a wall");
    }

    std::printf(failures == 0 ? "ALL PASS\n" : "%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
