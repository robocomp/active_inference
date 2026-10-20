// Offline harness for two tourP8 rulings (door_passability.h):
//  1. IN REACH INCLUDES THE VIEWING ANGLE. tourP8 leg 1 opened and crossed door_2 from door_4's standpoint: 1.95 m
//     from the aperture centre (inside the range band) but 58 deg off the door's normal, 0.69 m from the wall plane —
//     `open` and `cross` were offered there and the passage skipped approach. P(facing the opening) is the apparent
//     opening seen from the robot (w·|cos θ|) against the body it must admit, under the range σ: same range, on the
//     normal ⇒ high, 58 deg off ⇒ low.
//  2. NO PASSABLE ON PREDICTION ALONE. `transitable` was ASSERTED at phi = 89.5 deg from the actuator-command
//     anticipation with measured = 0 on every row. The streak toward assertion advances only on MEASURED cycles.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_reach_view.cpp -o /tmp/test_door_reach_view && /tmp/test_door_reach_view
#include <cmath>
#include <numbers>
#include <print>
#include "../src/door_passability.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

int main()
{
    // door_2: aperture 0.84 m (tourP8), body 0.60 + margin 0.10, range σ 0.10.
    const float w = 0.84f, needed = 0.70f, sigma = 0.10f;
    const float deg = std::numbers::pi_v<float> / 180.f;
    const float on    = prob_facing_opening(std::cos(0.f * deg), w, needed, sigma);
    const float off58 = prob_facing_opening(std::cos(58.f * deg), w, needed, sigma);
    const float off30 = prob_facing_opening(std::cos(30.f * deg), w, needed, sigma);
    std::print("P(facing) on-normal {:.3f}, 30 deg {:.3f}, 58 deg {:.3f}\n", on, off30, off58);
    CHECK(on > 0.85f);
    CHECK(off58 < 0.10f);
    CHECK(on > off30 and off30 > off58);                         // continuous, monotone in the angle
    CHECK(near_equal(prob_facing_opening(-std::cos(10.f * deg), w, needed, sigma),
                     prob_facing_opening(std::cos(10.f * deg), w, needed, sigma)));   // either face of the doorway
    // The same RANGE (1.95 m, in the band [1.32, 2.00]) gives the same band probability; the angle is what differs.
    const float band = prob_within(1.95f, 2.0f, sigma) - prob_within(1.95f, 1.32f, sigma);
    CHECK(band * on > 0.4f and band * off58 < 0.05f);
    // From the geometry: robot at (-0.64,1.89) room, door_2 centre (0.54,2.55), normal -y (the room side).
    {
        const float rx = -0.64f - 0.54f, ry = 1.89f - 2.55f;
        const float cos_t = facing_cos(rx, ry, 0.f, -1.f);
        std::print("tourP8 pose: |cos| {:.2f} ⇒ {:.1f} deg off the normal, P(facing) {:.3f}\n", std::abs(cos_t),
                   std::acos(std::abs(cos_t)) / deg, prob_facing_opening(cos_t, w, needed, sigma));
        CHECK(prob_facing_opening(cos_t, w, needed, sigma) < 0.10f);
        CHECK(prob_facing_opening(facing_cos(0.f, -1.6f, 0.f, -1.f), w, needed, sigma) > 0.85f);   // straight in front
    }

    // ── 2. transitable: unmeasured (predicted) cycles never assert ──
    {
        TransitableStreaks s;
        for (int k = 0; k < 50; ++k)   // phi anticipated to 90 deg: P(passable) 0.99 but measured = false
        {
            const auto d = transitable_step(s, false, 0.99f, false, 0.90f, 0.20f, 3);
            CHECK(not d.assert_now);
        }
        CHECK(s.yes == 0);
        int asserted_at = -1;
        for (int k = 0; k < 10 and asserted_at < 0; ++k)   // now measured
            if (transitable_step(s, false, 0.99f, true, 0.90f, 0.20f, 3).assert_now) asserted_at = k;
        CHECK(asserted_at == 2);                            // after 3 MEASURED cycles
        // An unmeasured cycle in the middle holds the streak (neither advances nor resets).
        TransitableStreaks h;
        transitable_step(h, false, 0.99f, true, 0.90f, 0.20f, 3);
        transitable_step(h, false, 0.99f, false, 0.90f, 0.20f, 3);
        CHECK(h.yes == 1);
        CHECK(transitable_step(h, false, 0.99f, true, 0.90f, 0.20f, 3).assert_now == false and h.yes == 2);
        // Retraction is the safe direction and may follow a measured shut leaf as before.
        TransitableStreaks r;
        bool retracted = false;
        for (int k = 0; k < 3; ++k) retracted = transitable_step(r, true, 0.05f, true, 0.90f, 0.20f, 3).retract_now or retracted;
        CHECK(retracted);
    }

    if (fails == 0) std::print("test_door_reach_view: ALL PASS\n");
    return fails == 0 ? 0 : 1;
}
