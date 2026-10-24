// D4 (gold13): the epistemic look aff_door_1 was offered INSIDE a bed. The policy built a field with the goal in a non-free
// cell — infinite from the robot — and spun in NO ROUTE; it must hand the place back instead (standpoint_fit.h place_unreachable).
//   1. a place inside furniture (goal cell not free): the field at the robot is infinite  => handed back;
//   2. a place in free space: finite => driven;
//   3. a free place the robot is merely walled off from is ALSO infinite — it too is handed back (the producer re-sites);
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_look_handback.cpp src/grid_planner.cpp src/local_policy/cost_to_go.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tlhb && /tmp/tlhb
#include "../src/local_policy/cost_to_go.h"
#include "../src/standpoint_fit.h"

#include <cmath>
#include <cstdio>
#include <span>

using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static const auto no_extra = [](const V&) { return 0.f; };

int main()
{
    const CostToGo::Grid g{{-3.f, 0.f}, 0.05f, 120, 100};   // x -3..3, y 0..5
    const V robot{-0.94f, 1.83f};
    const auto bed = [](const V& p) { return p.x() > -1.13f and p.x() < 0.47f and p.y() > 1.95f and p.y() < 4.05f; };
    const auto free_with_bed = [&](const V& p) { return not bed(p); };
    { // the look at (-0.07, 2.82) — inside the bed
        CostToGo c; V goal{-0.07f, 2.82f};
        c.build(g, free_with_bed, no_extra, std::span<const V>(&goal, 1));
        CHECK("place_in_furniture_field_infinite", not std::isfinite(c.at(robot)));
        CHECK("place_in_furniture_handed_back", rc::standpoint::place_unreachable(c.at(robot)));
    }
    { // a viewpoint that clears the bed
        CostToGo c; V goal{1.2f, 2.8f};
        c.build(g, free_with_bed, no_extra, std::span<const V>(&goal, 1));
        CHECK("free_place_finite", std::isfinite(c.at(robot)) and c.at(robot) > 0.f);
        CHECK("free_place_driven", not rc::standpoint::place_unreachable(c.at(robot)));
    }
    { // a free place the robot is walled off from
        const auto walled = [&](const V& p) { return std::fabs(p.x() - 0.8f) > 0.2f; };
        CostToGo c; V goal{1.8f, 2.8f};
        c.build(g, walled, no_extra, std::span<const V>(&goal, 1));
        CHECK("walled_off_place_handed_back", rc::standpoint::place_unreachable(c.at(robot)));
    }
    CHECK("not_a_number_is_unreachable", rc::standpoint::place_unreachable(std::nanf("")));
    std::printf(fails == 0 ? "ALL PASS\n" : "FAILED %d\n", fails);
    return fails == 0 ? 0 : 1;
}
