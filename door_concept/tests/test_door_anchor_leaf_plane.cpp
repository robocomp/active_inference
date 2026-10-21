// test_door_anchor_leaf_plane — Task 5b item 2: the passage anchor's leaf hypotheses hang on the leaf plane door_concept
// believes in (DoorBeliefState::d), not on the wall's mid-plane. The controller treats the anchor's leaf as a known obstacle
// (its Task 3b); with the leaf drawn at y = ±thickness/2 while door_concept's own model puts it d off the wall line, the two
// agents disagreed about where the obstacle is by up to 0.12 m (DOOR_0). One generative model.
//   1. the anchor frame's y is sign·across_u (the robot on −y): a leaf plane d beyond the line (out of the room, −across_u)
//      sits at y = −sign·(d + reg_depth) — reg_depth (> 0 into the room) because the anchor registers the WALL it sees;
//   2. every hypothesis (all four hinge × swing) moves rigidly by that offset — the leaf's shape, angle and probability kept;
//   3. d = 0 with no registration ⇒ the published leaves are bit-identical to before.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_anchor_leaf_plane.cpp -o /tmp/test_door_anchor_leaf_plane && /tmp/test_door_anchor_leaf_plane
#include <cmath>
#include <print>
#include "../src/door_anchor_leaf_plane.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) { ++g_checks; if (not ok) { ++g_fail; std::println("  FAIL: {}", what); } }
}

int main()
{
    std::println("1. the plane's y in the anchor frame");
    check(std::abs(rc::door::anchor_leaf_plane_y(0.12f, 0.0f, false, +1.f) + 0.12f) < 1e-6f, "d beyond the line, sign +1");
    check(std::abs(rc::door::anchor_leaf_plane_y(0.12f, 0.0f, false, -1.f) - 0.12f) < 1e-6f, "d beyond the line, sign -1");
    check(std::abs(rc::door::anchor_leaf_plane_y(-0.115f, 0.06f, true, +1.f) - 0.055f) < 1e-6f,
          "DOOR_0: leaf 0.115 room side, wall registered 0.06 room side ⇒ the leaf 0.055 in front of the WALL");
    check(rc::door::anchor_leaf_plane_y(-0.115f, 0.06f, false, +1.f) == 0.115f, "no registration ⇒ the line");

    std::println("2. every hypothesis moves rigidly");
    auto base = rc::interaction::anchor_leaf_hypotheses(-0.45f, 0.02f, 1.f, 1.f, 0.9f, 0.6f, 0.1f, 1.f, 0.5f);
    auto moved = base;
    rc::door::place_leaves_on_plane(moved, 0.07f);
    bool rigid = moved.size() == base.size() and not base.empty();
    for (std::size_t i = 0; rigid and i < base.size(); ++i)
        rigid = (moved[i].hinge - base[i].hinge - Eigen::Vector2f(0.f, 0.07f)).norm() < 1e-6f
            and (moved[i].tip - base[i].tip - Eigen::Vector2f(0.f, 0.07f)).norm() < 1e-6f
            and moved[i].phi == base[i].phi and moved[i].p == base[i].p and moved[i].sigma_phi == base[i].sigma_phi;
    check(rigid, "a hypothesis was not translated rigidly by the plane offset");

    std::println("3. d = 0, no registration ⇒ unchanged");
    auto same = base;
    rc::door::place_leaves_on_plane(same, rc::door::anchor_leaf_plane_y(0.f, 0.f, false, +1.f));
    bool eq = true;
    for (std::size_t i = 0; i < base.size(); ++i) eq = eq and same[i].hinge == base[i].hinge and same[i].tip == base[i].tip;
    check(eq, "the leaves changed with the plane on the line");
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
