// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/boxed_in_handoff_test.cpp src/layout_explorer.cpp \
//       src/wall_map.cpp src/wall_segmenter.cpp -o /tmp/bih && /tmp/bih
//
// Task X: a proto room right after birth, gainful standpoints all OUTSIDE the robot's grid component ("boxed in" by the
// furniture layer). The handoff must publish the best candidate's OWN gain (in nats), never 0 — the controller's
// filter drops gain <= 0, and its body-aware planner is the judge of reachability, not the coarse grid.
#include "layout_explorer.h"

#include <clocale>
#include <cmath>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
}

int main()
{
    std::setlocale(LC_ALL, "");
    using V = Eigen::Vector2f;
    const std::vector<V> room{{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    rc::layout::ExplorerParams ep;
    rc::layout::FeasibleGrid grid(room, ep.cell, ep.clearance);
    const V robot(0.8f, 2.f);
    grid.block_if([](const V& c) { return c.x() > 1.5f and c.x() < 2.4f; });   // a wall of "furniture" boxes the robot in
    // every standpoint beyond the wall holds 3 m2 of unknown floor; the robot's side holds none
    auto plan = rc::layout::plan_coverage(grid, robot, ep, {}, [](const V& v) { return v.x() > 2.4f ? 3.f : 0.f; },
                                          0.5f, {});
    CHECK(plan.why == "unreachable");
    CHECK(not plan.ok);
    CHECK(plan.best_unreached.has_value());
    CHECK(plan.best_unreached_gain == 3.f);            // the candidate's own gain survives
    CHECK(plan.gain == 0.f);                           // ...but there is no route, so no routed gain
    CHECK(rc::layout::boxed_in_handoff(plan, 0));
    CHECK(plan.ok and plan.why == "boxed_in_handoff");
    CHECK(plan.gain == 3.f);                           // the handoff carries it: the controller keeps the offer
    CHECK(plan.target.x() > 2.4f);

    // after a visit the 10-09 rule holds: "unreachable" is a verdict, not a handoff
    auto again = rc::layout::plan_coverage(grid, robot, ep, {}, [](const V& v) { return v.x() > 2.4f ? 3.f : 0.f; }, 0.5f, {});
    CHECK(not rc::layout::boxed_in_handoff(again, 1));
    CHECK(not again.ok);

    std::println("boxed_in_handoff_test: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
