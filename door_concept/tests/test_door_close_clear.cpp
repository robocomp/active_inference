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
    // warm18 leg 3 (2026-11-01): DOOR_0's leaf believed to swing OUT of the robot's room at llr +4.2 (it swings INTO it); the
    // robot 0.48 m past the plane. P(clear) 0.94 passed the old bar (kApproachCompleteProb 0.90) and the door was closed
    // onto the robot. Closing is an action that can hit the robot: it takes the motion policy's chance constraint.
    {
        const float p = prob_clear_of_swing(0.48f, 0.48f, 1.5f, 0.10f, -1.f, 4.2f, 0.3f);
        std::println("    warm18 leg 3: P(clear) {:.3f}; beyond the arc {:.3f}", p, prob_clear_of_swing(1.8f, 1.75f, 1.5f, 0.10f, -1.f, 4.2f, 0.3f));
        check(p < 1.f - rc::door::kCloseContactRiskMax, "warm18 leg 3: not clear enough to close");
        const float far = prob_clear_of_swing(1.8f, 1.75f, 1.5f, 0.10f, -1.f, 4.2f, 0.3f);   // beyond the arc on either side
        check(far >= 1.f - rc::door::kCloseContactRiskMax, "beyond the arc: clear whatever the swing");
    }
    // warm24 leg 1: standpoint at the band's middle 1.75 m, arc 1.5 m, stop up to 0.25 m short ⇒ P(clear) < 0.99, close timed out
    {
        const float sp = rc::door::close_standoff(1.0f, 1.5f, 0.10f);
        const float worst = sp - rc::door::kPlaceArrivalTolM;   // where the robot may come to rest
        const float p = prob_clear_of_swing(worst, worst, 1.5f, 0.10f, +1.f, 6.f, 0.3f);
        std::println("    close standpoint {:.2f} m; stopping {:.2f} m short leaves P(clear) {:.4f}", sp, rc::door::kPlaceArrivalTolM, p);
        check(p >= 1.f - rc::door::kCloseContactRiskMax, "the close request can go out wherever the robot stops");
        check(rc::door::close_standoff(2.6f, 1.5f, 0.10f) == 2.6f, "already clear: stay where it is");
    }
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
