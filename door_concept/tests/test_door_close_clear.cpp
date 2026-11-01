// door_close_clear.h: P(the robot is clear of the leaf's swing) for `close`, using the leaf's SWING SIDE (2026-11-01,
// warm10: four closes never offered — the robot stood 1.0-1.3 m from a door whose leaf opens into the OTHER room, inside the
// symmetric 1.5 m arc the old rule assumed).
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_close_clear.cpp -o /tmp/tdcc && /tmp/tdcc
#include "../src/door_close_clear.h"

#include <cmath>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what) { ++checks; if (not ok) { ++failures; std::println("FAIL: {}", what); } }
}

int main()
{
    using rc::door::prob_clear_of_swing;
    // arc 1.5 m (1.0 m leaf + body), range sigma 0.1, body radius 0.33
    constexpr float R = 1.5f, s = 0.1f, rb = 0.33f;
    // leaf opens toward -n (swing -1, confident: llr +6 for the MAP), robot 1.1 m out on +n: CLEAR although inside the arc
    check(prob_clear_of_swing(1.1f, +1.1f, R, s, -1.f, 6.f, rb) > 0.95f, "leaf opens away from the robot: clear");
    // same robot, leaf opening toward it: inside the arc, NOT clear
    check(prob_clear_of_swing(1.1f, +1.1f, R, s, +1.f, 6.f, rb) < 0.05f, "leaf opens toward the robot inside the arc: not clear");
    // leaf toward the robot but the robot beyond the arc: clear
    check(prob_clear_of_swing(2.2f, +2.2f, R, s, +1.f, 6.f, rb) > 0.95f, "beyond the arc: clear");
    // swing unknown (llr 0): the arc counts at half weight — inside it, about half
    const float half = prob_clear_of_swing(1.1f, +1.1f, R, s, -1.f, 0.f, rb);
    check(half > 0.4f and half < 0.6f, "swing unknown: about one half inside the arc");
    // standing IN the doorway plane (robot_side 0): the body reaches both sides — the leaf's side is the robot's side
    check(prob_clear_of_swing(0.3f, 0.f, R, s, -1.f, 6.f, rb) < 0.05f, "in the doorway: never clear");
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
