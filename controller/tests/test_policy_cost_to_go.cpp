// Task 3 (local-motion policy): geodesic cost-to-go with trap memory.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_policy_cost_to_go.cpp src/local_policy/cost_to_go.cpp -o /tmp/tctg && /tmp/tctg
#include "../src/local_policy/cost_to_go.h"

#include <chrono>
#include <cmath>
#include <cstdio>

using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static const auto no_extra = [](const V&) { return 0.f; };

int main()
{
    const CostToGo::Grid g{{0.f, 0.f}, 0.1f, 100, 100};   // 10 x 10 m

    { // L: horizontal arm y in [0,1], x in [0,9]; vertical arm x in [8,9], y in [0,9]. Goal at top of vertical arm.
        auto free = [](const V& p) { return (p.y() <= 1.0f and p.x() <= 9.0f) or (p.x() >= 8.0f and p.x() <= 9.0f and p.y() <= 9.0f); };
        CostToGo c; V goal{8.5f, 9.0f}; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        const float v = c.at({0.5f, 0.5f});
        const float path = 8.0f + 8.5f - 0.5f;   // along x then up y ~ 16, vs Euclid ~ 12.5
        CHECK("l_corridor_geodesic", std::fabs(v - path) < 1.5f and v > 14.f);
        CHECK("l_corridor_goal_zero", c.at(goal) < 0.15f);
    }
    { // wall across the middle
        auto free = [](const V& p) { return std::fabs(p.x() - 5.f) > 0.3f; };
        CostToGo c; V goal{8.f, 5.f}; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        CHECK("walled_off_unreachable", not c.reachable({2.f, 5.f}) and std::isinf(c.at({2.f, 5.f})) and c.reachable({7.f, 5.f}));
    }
    { // two corridors y in [1,2] (short, direct) and y in [7,8] joined by vertical links at x<=1 and x>=9
        auto free = [](const V& p) { return (p.y() >= 1.f and p.y() <= 2.f) or (p.y() >= 7.f and p.y() <= 8.f)
                                          or (p.x() <= 1.f and p.y() >= 1.f and p.y() <= 8.f) or (p.x() >= 9.f and p.y() >= 1.f and p.y() <= 8.f); };
        // Start at (0.5,4.5) on the left link, goal at (9.5,4.5) on the right link. Lower route ~ 3.5+9+3.5 = 16, upper ~ 16 too:
        // make them unequal by starting nearer the lower corridor.
        V start{0.5f, 2.5f}, goal{9.5f, 2.5f};
        CostToGo c; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        const float before = c.at(start);
        c.add_trap({5.f, 1.5f}, 0.6f, 12.f); c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        const float after = c.at(start);
        const float longer = 17.5f;   // via the upper corridor: 5 up + ~8 across + 5 down, minus 8-connected corner cuts
        std::printf("  before %.2f after %.2f\n", before, after);
        CHECK("trap_reroutes", before < 10.f and after > before + 5.f and std::fabs(after - longer) < 1.5f);
        c.decay_traps(1e-4f); c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        CHECK("trap_decays_away", std::fabs(c.at(start) - before) < 1e-3f);
    }
    { // goal = a segment
        auto free = [](const V&) { return true; };
        CostToGo c; std::vector<V> line; for (float x = 2.f; x <= 6.f; x += 0.05f) line.push_back({x, 5.f});
        c.build(g, free, no_extra, line);
        CHECK("goal_set_line", c.at({4.f, 5.f}) < 0.05f and std::fabs(c.at({4.f, 6.f}) - 1.f) < 0.15f and std::fabs(c.at({2.f, 5.f})) < 0.05f);
    }
    { // goal in a non-free cell: snaps within one cell, or all inf
        auto free = [](const V& p) { return p.x() < 5.f; };
        CostToGo c; V near{5.02f, 3.f}, far{8.f, 3.f};
        c.build(g, free, no_extra, std::span<const V>(&near, 1));
        CHECK("goal_snaps_one_cell", c.reachable({3.f, 3.f}));
        c.build(g, free, no_extra, std::span<const V>(&far, 1));
        CHECK("goal_unsnappable_all_inf", not c.reachable({3.f, 3.f}) and std::isinf(c.at({3.f, 3.f})));
        V off{-5.f, -5.f}; c.build(g, free, no_extra, std::span<const V>(&off, 1));
        CHECK("goal_offgrid_no_throw", not c.reachable({3.f, 3.f}));
    }
    { // bilinear + inf fallback
        auto free = [](const V& p) { return p.x() < 5.f; };
        CostToGo c; V goal{1.f, 1.f}; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        const float a = c.at({2.0f, 1.0f}), b = c.at({2.1f, 1.0f}), m = c.at({2.05f, 1.0f});
        CHECK("bilinear_midpoint", std::fabs(m - 0.5f * (a + b)) < 1e-4f);
        const float e = c.at({4.97f, 1.f});   // between last free column (4.9) and wall (5.0): inf neighbour
        CHECK("inf_neighbour_falls_back_finite", std::isfinite(e));
    }
    { // uniform density: 45 deg line vs axis line of equal length
        auto free = [](const V&) { return true; };
        auto ex = [](const V&) { return 0.5f; };
        CostToGo c1, c2; V g1{1.f, 1.f}; c1.build(g, free, ex, std::span<const V>(&g1, 1));
        const float diag = c1.at({6.f, 6.f}), axis = c1.at({1.f + 5.f * 1.41421356f, 1.f});
        CHECK("diagonal_vs_straight_consistent", std::fabs(diag - axis) < 0.1f and std::fabs(axis - 5.f * 1.41421356f * 1.5f) < 0.2f);
    }
    { // trap resolution independence
        auto free = [](const V&) { return true; };
        float v[2]; int k = 0;
        for (float cell : {0.05f, 0.10f})
        {
            const CostToGo::Grid G{{0.f, 0.f}, cell, int(10.f / cell), int(10.f / cell)};
            CostToGo c; c.add_trap({5.f, 5.f}, 1.f, 2.f); V goal{9.f, 5.f};
            c.build(G, free, no_extra, std::span<const V>(&goal, 1)); v[k++] = c.at({1.f, 5.f});
        }
        std::printf("  trap start values %.3f %.3f\n", v[0], v[1]);
        CHECK("trap_cost_resolution_independent", std::fabs(v[0] - v[1]) < 0.1f * v[0] and v[0] > 8.5f);
    }
    { // pose inside blocked, last row, seeded goals, corner cut
        auto free = [](const V& p) { return p.x() < 5.f or p.x() > 5.35f; };
        CostToGo c; V goal{2.f, 2.f}; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        const V q{5.15f, 2.f};   // between non-free columns 5.0..5.3: all 4 surrounding cells blocked
        const float a = c.at(q);
        CHECK("pose_inside_blocked_reads_nearest_plus_distance", std::isfinite(a) and a > c.at({4.9f, 2.f}) - 0.1f and a < c.at({4.9f, 2.f}) + 0.5f);
        CHECK("last_row_finite", std::isfinite(c.at({3.f, 9.9f})) and std::isfinite(c.at({0.f, 3.f})) and std::isinf(c.at({10.5f, 3.f})));
        { auto all = [](const V&) { return true; }; CostToGo f; V gl{2.f, 2.f}; f.build(g, all, no_extra, std::span<const V>(&gl, 1));
          CHECK("last_col_finite", std::isfinite(f.at({9.9f, 3.f})) and std::isfinite(f.at({3.f, 9.9f}))); }
        auto blocked = [](const V& p) { return p.x() < 5.f; };
        CostToGo d; std::vector<V> seg; for (float x = 6.f; x <= 7.f; x += 0.1f) seg.push_back({x, 5.f});
        d.build(g, blocked, no_extra, seg);
        CHECK("seeded_goal_cells_zero_when_blocked", d.seeded_goal_cells() == 0);
        d.build(g, blocked, no_extra, std::span<const V>(&goal, 1));
        CHECK("seeded_goal_cells_positive", d.seeded_goal_cells() == 1);
        // two blocked cells touching at a corner: (3.0,3.0) and (3.1,3.1) cells... wall along diagonal
        auto diagfree = [](const V& p) { return std::fabs(p.x() - p.y()) > 0.05f; };
        CostToGo e; V gl{6.f, 2.f}; e.build(g, diagfree, no_extra, std::span<const V>(&gl, 1));
        CHECK("no_corner_cut", not e.reachable({2.f, 6.f}) or e.at({2.f, 6.f}) > 100.f);
    }
    { // timing 200x200
        const CostToGo::Grid G{{0.f, 0.f}, 0.05f, 200, 200};
        auto free = [](const V& p) { return std::fmod(p.x(), 2.f) > 0.2f or p.y() < 1.f or p.y() > 9.f; };
        auto ex = [](const V& p) { return 0.1f * std::fabs(std::sin(p.x() * 3.f)); };
        CostToGo c; V goal{9.5f, 9.5f};
        const auto t0 = std::chrono::steady_clock::now();
        c.build(G, free, ex, std::span<const V>(&goal, 1));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("build 200x200: %.2f ms\n", ms);
        CHECK("build_reachable", c.reachable({0.5f, 0.5f}));
    }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
