// Task 3 (local-motion policy): geodesic cost-to-go with trap memory.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_policy_cost_to_go.cpp src/local_policy/cost_to_go.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tctg && /tmp/tctg
#include "../src/local_policy/cost_to_go.h"
#include "../src/local_policy/doorway_corridor.h"
#include "../../common/robot_footprint/robot_footprint.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <optional>

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
    { // zero-radius trap ignored
        auto free = [](const V&) { return true; };
        CostToGo c; c.add_trap({5.f, 5.f}, 0.f, 3.f); c.add_trap({5.f, 5.f}, -1.f, 3.f); c.add_trap({5.f, 5.f}, 1.f, 0.f);
        V goal{9.f, 5.f}; c.build(g, free, no_extra, std::span<const V>(&goal, 1));
        CHECK("trap_zero_radius_ignored", std::fabs(c.at({1.f, 5.f}) - 8.f) < 0.05f);
    }
    { // ring order != Euclid order: query at the corner of a cell; finite cell at ring 1 diagonal (dist ~0.19)
      // vs a cell at ring 2 straight (dist ~0.15 from the query).
        const CostToGo::Grid G{{0.f, 0.f}, 0.1f, 40, 40};
        // finite cells: only (30,30)=goal region reachable; free set = {(10,12)} union nothing else near.
        // Query p=(0.95,0.95)ish between cells (9,9),(10,9),(9,10),(10,10), all non-free. Cell (12,10)... build explicit:
        auto free = [](const V& p) { const int i = int(std::lround(p.x() / 0.1f)), j = int(std::lround(p.y() / 0.1f));
                                     return (i == 11 and j == 11) or (i == 12 and j == 9); };
        CostToGo c; V a{1.1f, 1.1f}, b{1.2f, 0.9f}; std::vector<V> gl{a, b};
        c.build(G, free, no_extra, gl);   // both seeded with value 0
        const V q{1.0f, 0.97f};   // cell (10,10)/(10,9) region
        // (11,11): dx .1 dy .13 -> .164 ; (12,9): dx .2 dy .07 -> .212; make (12,9) the ring-2 one but closer? swap below
        const float got = c.at(q);
        const float expect = std::min((V(1.1f, 1.1f) - q).norm(), (V(1.2f, 0.9f) - q).norm());
        CHECK("ring_search_nearest_is_euclidean", std::fabs(got - expect) < 1e-4f);
        // randomised brute-force: random sparse finite cells, random blocked queries, compare to exhaustive nearest
        unsigned s = 12345u; auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (s >> 8) / float(1 << 24); };
        bool all_ok = true;
        for (int trial = 0; trial < 200; ++trial)
        {
            std::vector<V> pts; for (int k = 0; k < 3; ++k) pts.push_back({0.3f + 3.4f * rnd(), 0.3f + 3.4f * rnd()});
            auto fr = [&](const V& p) { for (const auto& q : pts) if ((p - q).norm() < 0.04f) return true; return false; };
            CostToGo d; d.build(G, fr, no_extra, pts);
            const V qq{0.3f + 3.4f * rnd(), 0.3f + 3.4f * rnd()};
            float bf = 1e9f; bool seeded = false;
            for (int j = 0; j < G.h; ++j) for (int i = 0; i < G.w; ++i)
            { const V cc{i * 0.1f, j * 0.1f}; if (fr(cc)) { bf = std::min(bf, (cc - qq).norm()); seeded = true; } }
            const float got2 = d.at(qq);
            if (seeded and std::isfinite(got2) and std::fabs(got2 - bf) > 1e-3f)
            {   // got2 may be a bilinear/neighbour read when a surrounding cell is free; accept only if q is near a free cell
                if (bf > 0.15f) all_ok = false;
            }
        }
        CHECK("ring_search_random_vs_bruteforce", all_ok);
    }
    { // worst-case at() on all-blocked 200x200
        const CostToGo::Grid G{{0.f, 0.f}, 0.05f, 200, 200};
        auto none = [](const V&) { return false; };
        CostToGo c; V goal{5.f, 5.f}; c.build(G, none, no_extra, std::span<const V>(&goal, 1));
        const auto t0 = std::chrono::steady_clock::now();
        const float r = c.at({5.f, 5.f});
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("worst-case at() all-blocked 200x200: %.3f ms\n", ms);
        CHECK("at_worst_case_fast_and_inf", std::isinf(r) and ms < 1.0);
        CostToGo bad; bad.build({{0.f, 0.f}, 0.f, 10, 10}, none, no_extra, std::span<const V>(&goal, 1));
        CHECK("invalid_build_consistent", std::isinf(bad.at({1.f, 1.f})));
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
    { // Live fix 3 ruling: PRICE CHOOSES THE ROUTE, METRES MEASURE PROGRESS. A 6 x 6 m open area, goal at (5, 3), start (1, 3).
      // (i) an UNAVOIDABLE band (x 2.9 .. 3.1, full height) of density 5: the route crosses it; priced = 4 + 0.2 x 5 = 5 m,
      //     its length 4 m. (ii) a high band (density 200) with a gap at y > 5: the cheapest route DETOURS through the gap;
      //     the length is the detour's metres, the priced value equals it (no risk on the detour).
        const CostToGo::Grid g{{0.f, 0.f}, 0.05f, 121, 121};
        const std::vector<Eigen::Vector2f> goal{{5.f, 3.f}};
        CostToGo band;
        band.build(g, [](const Eigen::Vector2f&) { return true; },
                   [](const Eigen::Vector2f& p) { return (p.x() >= 2.9f and p.x() <= 3.1f) ? 5.f : 0.f; }, goal);
        std::printf("  unavoidable band: priced %.3f m, route length %.3f m\n", band.at({1.f, 3.f}), band.at_length({1.f, 3.f}));
        CHECK("route length: the route's METRES (4.0), not its price (5.0)", std::abs(band.at_length({1.f, 3.f}) - 4.f) < 0.05f
              and std::abs(band.at({1.f, 3.f}) - 5.f) < 0.1f);
        CostToGo det;
        det.build(g, [](const Eigen::Vector2f&) { return true; },
                  [](const Eigen::Vector2f& p) { return (p.x() >= 2.9f and p.x() <= 3.1f and p.y() < 5.f) ? 200.f : 0.f; }, goal);
        const float L = det.at_length({1.f, 3.f}), P = det.at({1.f, 3.f});
        const float detour = 2.f * std::hypot(2.f, 2.05f);   // via the gap's lower edge (3, 5.05)
        std::printf("  detour: priced %.3f m, route length %.3f m (straight 4.0, detour ~%.2f)\n", P, L, detour);
        CHECK("route length: the cheapest route detours — length ~ the detour (> 4 m), equal to its price", L > 5.f and std::abs(L - P) < 0.05f
              and std::abs(L - detour) < 0.3f);
        CHECK("route length: 0 at the goal", det.at_length({5.f, 3.f}) < 0.05f);
        CHECK("route length: +inf off-grid, like the price", std::isinf(det.at_length({-1.f, 3.f})));
    }

    { // ★ F4 — THE MEASURED JAMBS IN THE ROUTE'S RISK. A wall along y = 0 (0.15 m thick) with a 0.90 m opening centred at x = 0,
      // normal +y; Shadow's bare hull; the cross's goal line 2 m past the gap over the span the centre can occupy (doorway_goal);
      // the corridor joins the two sides as the controller's also_free does. Sigmas as measured live (rtD1b: gap sigma_lat
      // 0.017-0.021 m) and the contact model's shared offset sqrt(SigmaRelXyM 0.011^2 + body 0.01^2).
        rc::RobotFootprint B = rc::RobotFootprint::shadow(); B.set_safety_margin(0.f);
        const float r_ins = B.inscribed_radius();
        const RoomGap gap{{0.f, 0.f}, {0.f, 1.f}, 0.90f};
        // sigma_lat 0.02, sigma_n 0.02, sigma_s = sqrt(0.011^2 + 0.01^2), LiDAR range sigma 0.02 (LidarRangeSigmaM); the
        // corridor the controller joins the rooms with
        const DoorwayJambs J = doorway_jambs(gap, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.02f, B, 1.0f, 2.05f);
        std::printf("  shadow hull crossing along the normal: half-widths %.3f / %.3f, front %.3f back %.3f, placement %.3f\n",
                    J.half_left_m, J.half_right_m, J.front_m, J.back_m, J.placement_m);
        const CostToGo::Grid G{{-3.f, -3.f}, 0.05f, 121, 121};
        auto wall_dist = [](const V& p) {   // distance to the solid wall: |y| <= 0.075, |x| >= 0.45
            const float dy = std::max(0.f, std::abs(p.y()) - 0.075f), dx = std::max(0.f, 0.45f - std::abs(p.x()));
            return std::hypot(dx, dy); };
        auto free = [&](const V& p) { return wall_dist(p) > r_ins or in_doorway_corridor(gap, p, 1.0f, 2.05f, r_ins); };
        std::vector<V> line; for (float x = -0.22f; x <= 0.2201f; x += 0.05f) line.push_back({x, 2.f});
        const float w = 1.f;   // Planner.ResidualRiskWeight (config default)
        auto with_jambs = [&](const V& p) { return w * doorway_jamb_density(J, p); };
        CostToGo plain, jam;
        plain.build(G, free, no_extra, line);
        jam.build(G, free, with_jambs, line);
        // the descent line: from p, repeatedly the 0.01 m step (32 directions) that lowers the field most; the lateral offset
        // where it crosses the jamb line y = 0
        auto crossing = [](const CostToGo& c, V p) {
            for (int k = 0; k < 2000 and p.y() < 0.5f; ++k)
            {
                V best = p; float bf = c.at(p);
                for (int a = 0; a < 32; ++a)
                {
                    const float th = a * 2.f * 3.14159265f / 32.f; const V q = p + 0.01f * V(std::cos(th), std::sin(th));
                    if (const float f = c.at(q); f < bf) { bf = f; best = q; }
                }
                if (best == p) break;
                if (p.y() < 0.f and best.y() >= 0.f) return p.x() + (best.x() - p.x()) * (0.f - p.y()) / (best.y() - p.y());
                p = best;
            }
            return std::numeric_limits<float>::quiet_NaN(); };
        const float th20 = 20.f * 3.14159265f / 180.f;
        const V start{0.15f, -0.15f / std::tan(th20)};   // 0.15 m off-centre, the gap centre 20 deg off the normal from here
        const float e_plain = crossing(plain, start), e_jam = crossing(jam, start), e_mirror = crossing(jam, V(-start.x(), start.y()));
        const float hl = J.half_left_m;
        std::printf("  descent crosses the jamb line at e %+.3f m without the jambs (clearance %.3f m), %+.3f m with them "
                    "(clearance %.3f m; mirrored start %+.3f)\n", e_plain, 0.45f - hl - std::abs(e_plain), e_jam,
                    0.45f - hl - std::abs(e_jam), e_mirror);
        CHECK("F4: without the jambs the descent cuts the corner (the baseline the test discriminates)", std::abs(e_plain) > 0.10f);
        // ★ ORACLE (deviation from the brief's "centre line +- one cell", stated in task-3-report.md): the field descends until the
        //   marginal risk of a further lateral metre no longer pays for the metres it costs — with the measured sigma (~2.5 cm)
        //   R(e) is ~0 within ~6 cm of the centre and the route settles there, 6.8 cm from the jamb (2.7 sigma) instead of 1.6
        //   cm. The model-level statement is: the crossing the field leads to is one the policy's OWN eligibility bar admits
        //   (P(contact) < PolicyParams::risk_eps = 0.01), and the one it led to without the jambs is not.
        const float risk_eps = 0.01f;   // PolicyParams{}.risk_eps (local_policy.h)
        auto p_contact = [&](float e) { return 1.f - std::exp(-doorway_jamb_crossing_nats(J, e)); };
        std::printf("  P(contact) at the crossing: %.4f without the jambs, %.4f with them\n", p_contact(e_plain), p_contact(e_jam));
        CHECK("F4: without the jambs the field's crossing is NOT one the policy can drive (P(contact) > risk_eps)", p_contact(e_plain) > risk_eps);
        CHECK("F4: with the measured jambs it is (P(contact) < risk_eps), on the centre's side of the start", p_contact(e_jam) < risk_eps
              and std::abs(e_jam) < std::abs(e_plain));
        CHECK("F4: and symmetrically from the mirrored start (to the field's cell)", std::abs(e_mirror + e_jam) <= G.cell);
        { // the live configuration's LidarRangeSigmaM 0.03 (the unit test above uses the harness's 0.02)
            const DoorwayJambs L = doorway_jambs(gap, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.03f, B, 1.0f, 2.05f);
            CostToGo live; live.build(G, free, [&](const V& p) { return w * doorway_jamb_density(L, p); }, line);
            const float e_live = crossing(live, start);
            std::printf("  live sigmas (range 0.03): crossing at e %+.3f m, clearance %.3f m\n", e_live, 0.45f - hl - std::abs(e_live));
            CHECK("F4: live sigmas — the crossing is drivable too, nearer the centre than without", 1.f - std::exp(-doorway_jamb_crossing_nats(L, e_live)) < risk_eps
                  and std::abs(e_live) < std::abs(e_plain));
        }
        { // ★ rtD1c leg 5: the [gap] detector locked onto the open LEAF (0.40-0.52 m at score 0.03-0.3 against an 0.87 m door).
          // The anchor's opening (door_concept's door: 0.87 m, here centred at x = 0) and a LOW-SCORE gap measurement of 0.52 m (wide enough that its corridor exists: 0.45 m would have none)
          // whose centre sits 0.20 m east: the gap enters only as a hypothesis weighted by its score — the route's risk and its
          // centre line stay the anchor's. A reliable (score 0.99) measurement of the same door does move it.
            const RoomGap door{{0.f, 0.f}, {0.f, 1.f}, 0.87f}, leaf{{0.20f, 0.f}, {0.f, 1.f}, 0.52f};
            const DoorwayJambs A = doorway_jambs(door, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.03f, B, 1.0f, 2.05f);
            const DoorwayJambs Gl = doorway_jambs(leaf, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.03f, B, 1.0f, 2.05f);
            auto free87 = [&](const V& p) { const float dy = std::max(0.f, std::abs(p.y()) - 0.075f), dx = std::max(0.f, 0.435f - std::abs(p.x()));
                                            return std::hypot(dx, dy) > r_ins or in_doorway_corridor(door, p, 1.0f, 2.05f, r_ins)
                                                   or in_doorway_corridor(leaf, p, 1.0f, 2.05f, r_ins); };
            std::vector<V> l87; for (float x = -0.205f; x <= 0.2051f; x += 0.05f) l87.push_back({x, 2.f});
            const DoorwayRisk anchor_only{A, std::nullopt, 0.f}, low{A, Gl, 0.03f};
            CostToGo ca, cl;
            ca.build(G, free87, [&](const V& p) { return w * doorway_risk_density(anchor_only, p); }, l87);
            cl.build(G, free87, [&](const V& p) { return w * doorway_risk_density(low, p); }, l87);
            const V st{-0.15f, -0.15f / std::tan(th20)};   // from the WEST, so the phantom's centre (east) would pull it across
            const float ea = crossing(ca, st), el = crossing(cl, st);
            std::printf("  anchor 0.87 alone: crossing e %+.3f, field at start %.3f; + a 0.52 m gap at score 0.03: e %+.3f, field %.3f\n",
                        ea, ca.at(st), el, cl.at(st));
            // the discriminating baseline: had the phantom REPLACED the anchor (weight 1), the anchor's jambs would not price the route
            CostToGo cr; cr.build(G, free87, [&](const V& p) { return w * doorway_risk_density(DoorwayRisk{A, Gl, 1.f}, p); }, l87);
            const float er = crossing(cr, st);
            std::printf("  (the phantom as THE geometry, weight 1: crossing e %+.3f, P(contact) on the real door %.3f)\n", er,
                        1.f - std::exp(-doorway_jamb_crossing_nats(A, er)));
            CHECK("F4 mixture: a score-0.03 phantom gap leaves the centre line the anchor's (+- one cell)", std::abs(el - ea) <= G.cell
                  and std::abs(er - ea) > G.cell);
            CHECK("F4 mixture: ...and the field at the robot within -log(1 - 0.03) x placement of the anchor's",
                  std::abs(cl.at(st) - ca.at(st)) <= -std::log(1.f - 0.03f) * A.placement_m + 0.02f);
            CHECK("F4 mixture: no gap = the anchor's jambs exactly", doorway_risk_density(anchor_only, V(0.1f, 0.f)) == doorway_jamb_density(A, V(0.1f, 0.f)));
            { // ★ fix round 1, I-1: a NON-parallel phantom (normal 30 deg off the door's), 0.62 m wide (a 14 cm strip: on a 5 cm grid a 6 cm
              // slanted strip has no 8-connected path, so 0.62 makes the leak testable), score
              // 0.03, centred in the WALL 2 m east of the door, its strip a 0.8 m SHORTCUT to the goal line from the east start. also_free admits its
              // strip, which runs through solid; under the anchor that strip is wall (contact-free probability 0), so it costs
              // >= -log 0.03 nats there and the field must still descend through the anchor's opening.
                const RoomGap slant{{2.0f, 0.f}, {-0.5f, 0.8660254f}, 0.62f};   // normal 30 deg off the door's, pointing back toward it
                const DoorwayJambs Gx = doorway_jambs(slant, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.03f, B, 1.0f, 2.05f);
                const DoorwayRisk phantom{A, Gx, 0.03f};
                auto map87 = [&](const V& p) { const float dy = std::max(0.f, std::abs(p.y()) - 0.075f), dx = std::max(0.f, 0.435f - std::abs(p.x()));
                                               return std::hypot(dx, dy) > r_ins; };
                auto freex = [&](const V& p) { return map87(p) or in_doorway_risk_corridor(phantom, p); };
                const std::function<bool(const V&)> mf = map87;
                CostToGo fixed, leaky;
                fixed.build(G, freex, [&](const V& p) { return w * doorway_risk_density(phantom, p, mf); }, l87);
                leaky.build(G, freex, [&](const V& p) { return w * doorway_risk_density(phantom, p); }, l87);   // no map: pre-fix
                const V east{2.6f, -0.8f}, in_wall{2.0f, 0.f};
                const float d_wall = doorway_risk_density(phantom, in_wall, mf);
                // the Dijkstra route itself: from the start's cell, always to the cheapest of the 8 neighbouring cells (the field
                // is a priced distance, so this ends at the goal); the x where it first reaches y >= 0
                auto route_cross = [&](const CostToGo& c, V p) {
                    p = {std::round(p.x() / G.cell) * G.cell, std::round(p.y() / G.cell) * G.cell};
                    for (int k = 0; k < 400; ++k)
                    {
                        V best = p; float bf = c.at(p);
                        for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di)
                            if (const V q = p + G.cell * V(float(di), float(dj)); c.at(q) < bf) { bf = c.at(q); best = q; }
                        if (best == p) break;
                        if (p.y() < 0.f and best.y() >= -1e-4f) return best.x();
                        p = best;
                    }
                    return std::numeric_limits<float>::quiet_NaN(); };
                const float xf = route_cross(fixed, east), xl = route_cross(leaky, east);
                std::printf("  slanted phantom (score 0.03): density in the wall %.2f nats/m (pre-fix %.3f); route from (2.6,-0.8) "
                            "crosses y=0 at x %+.3f (pre-fix %+.3f)\n", d_wall, doorway_risk_density(phantom, in_wall), xf, xl);
                CHECK("F4 I-1: the phantom's strip through solid costs >= -log(score) nats (no near-free strip)",
                      d_wall >= -std::log(0.03f) * A.placement_m / (A.front_m + A.back_m) - 1e-3f);
                CHECK("F4 I-1: the field descends through the ANCHOR's opening, not the phantom's strip",
                      std::abs(xf) < 0.435f - r_ins + G.cell);
                // the anchor alone on the map + its corridor: the field the phantom must not undercut
                CostToGo base; base.build(G, [&](const V& p) { return map87(p) or in_doorway_corridor(door, p, 1.0f, 2.05f, r_ins); },
                                          [&](const V& p) { return w * doorway_jamb_density(A, p); }, l87);
                std::printf("  field at the start: anchor alone %.3f, with the phantom %.3f, pre-fix %.3f (route x %+.3f)\n",
                            base.at(east), fixed.at(east), leaky.at(east), xl);
                CHECK("F4 I-1: the phantom does not undercut the anchor's field at the start (within -log(1-0.03) x placement)",
                      fixed.at(east) >= base.at(east) - 0.05f);
                CHECK("F4 I-1: (discriminating) without the map test the strip was a shortcut (> 0.3 m cheaper)",
                      leaky.at(east) < base.at(east) - 0.3f);
            }
            const DoorwayJambs Gs = doorway_jambs(RoomGap{{0.10f, 0.f}, {0.f, 1.f}, 0.87f}, 0.02f, 0.02f, std::hypot(0.011f, 0.01f), 0.03f, B, 1.0f, 2.05f);
            const DoorwayRisk sure{A, Gs, 0.99f};
            CostToGo cs; cs.build(G, free87, [&](const V& p) { return w * doorway_risk_density(sure, p); }, l87);
            const float es = crossing(cs, V(0.25f, -0.15f / std::tan(th20)));
            const float es_anchor = crossing(ca, V(0.25f, -0.15f / std::tan(th20)));
            std::printf("  a score-0.99 gap measured 0.10 m east: crossing e %+.3f (anchor alone %+.3f)\n", es, es_anchor);
            CHECK("F4 mixture: a reliable gap measurement moves the route toward ITS centre", es > es_anchor);
        }
        // no gap active: the controller adds nothing (std::optional empty) — the density helper is the only F4 term
        const std::optional<DoorwayJambs> none;
        auto extra_none = [&](const V& p) { return none ? w * doorway_jamb_density(*none, p) : 0.f; };
        CostToGo nog; nog.build(G, free, extra_none, line);
        bool same = true;
        for (float y = -2.9f; y < 2.9f; y += 0.13f) for (float x = -2.9f; x < 2.9f; x += 0.13f)
            if (const V q{x, y}; nog.at(q) != plain.at(q) and not (std::isinf(nog.at(q)) and std::isinf(plain.at(q)))) same = false;
        CHECK("F4: no gap active -> the field is the plain one, bit for bit", same);
        // away from the doorway: no density outside the corridor or beyond the body's reach of the line; and past the
        // doorway (every route from there to the goal stays clear of it) the field is unchanged
        bool zero_away = true, unchanged_past = true;
        for (float y = -2.9f; y < 2.9f; y += 0.05f) for (float x = -2.9f; x < 2.9f; x += 0.05f)
        {
            const V q{x, y};
            const bool near = in_doorway_corridor(gap, q, 1.0f, 2.05f, r_ins) and y > -J.front_m - 0.2f and y < J.back_m + 0.2f;
            if (not near and doorway_jamb_density(J, q) != 0.f and doorway_jamb_density(J, q) > 1e-6f) zero_away = false;
            if (y > J.back_m + 0.2f and std::isfinite(plain.at(q)) and std::abs(jam.at(q) - plain.at(q)) > 1e-4f) unchanged_past = false;
        }
        CHECK("F4: no risk density away from the doorway", zero_away);
        CHECK("F4: field values past the doorway unchanged", unchanged_past);
        // the centre-line crossing costs (almost) nothing: F1 reads R(0) nats there
        std::printf("  R(e) nats: e 0 %.2e, 0.05 %.2e, 0.10 %.2e, 0.15 %.3f\n", doorway_jamb_crossing_nats(J, 0.f),
                    doorway_jamb_crossing_nats(J, 0.05f), doorway_jamb_crossing_nats(J, 0.10f), doorway_jamb_crossing_nats(J, 0.15f));
        CHECK("F4: the centre crossing of a 0.90 m gap is ~free (R(0) < 1e-6 nats)", doorway_jamb_crossing_nats(J, 0.f) < 1e-6f);
        { // the WIDTH is measured: an opening 1.8 cm narrower than the body (0.55 m) is improbable, not impossible; a 0.40 m one
          // costs more nats than any route can repay (P(no contact) < 1e-9)
            DoorwayJambs n = J; n.gap.width = 0.55f; DoorwayJambs m = J; m.gap.width = 0.40f;
            const float rn = doorway_jamb_crossing_nats(n, 0.f), rm = doorway_jamb_crossing_nats(m, 0.f);
            std::printf("  R(0): 0.55 m opening %.2f nats, 0.40 m opening %g\n", rn, rm);
            CHECK("F4: a gap measured just narrower than the body costs finite nats (> 1)", std::isfinite(rn) and rn > 1.f);
            CHECK("F4: a gap far narrower than the body is all but impassable (R > 20 nats)", rm > 20.f);
        }
        CHECK("F4: R grows monotonically toward a jamb", doorway_jamb_crossing_nats(J, 0.05f) < doorway_jamb_crossing_nats(J, 0.10f)
              and doorway_jamb_crossing_nats(J, 0.10f) < doorway_jamb_crossing_nats(J, 0.15f));
    }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
