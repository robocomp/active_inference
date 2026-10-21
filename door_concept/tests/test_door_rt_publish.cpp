// Task D-C (R2 review M-6): door_rt_publish.h — the RT edge rewrite dead-band is on the aperture's JAMBS, so a pure yaw
// re-fit about a fixed centre (the restored DOOR_0 corrected from the partition back to its wall) is published.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_rt_publish.cpp -o /tmp/t && /tmp/t
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include "../src/door_rt_publish.h"

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

int main()
{
    using namespace rc::door;
    const float pi = std::numbers::pi_v<float>;
    // never written ⇒ write
    CHECK(rt_write_due(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::quiet_NaN(),
                       0.9f, 2.66f, -3.20f, 0.5f));
    // the restore's 90° error corrected about a fixed centre: the old centre-only rule wrote nothing (Δc = 0)
    CHECK(rt_write_due(0.9f, 2.66f, 1.547f, 0.9f, 2.66f, 1.547f - 0.5f * pi, 0.5f));
    std::print("90° about the centre moves a jamb {:.3f} m\n", aperture_jamb_move(0.9f, 2.66f, 1.547f, 0.9f, 2.66f, 1.547f - 0.5f * pi, 0.5f));
    // a yaw wobble whose jambs move less than a translation would have to (1° on a 1 m door: 8.7 mm) ⇒ not rewritten
    CHECK(not rt_write_due(0.9f, 2.66f, 0.0f, 0.9f, 2.66f, pi / 180.f, 0.5f));
    // 6° on a 1 m door moves a jamb 5.2 cm ⇒ rewritten, exactly as a 5.2 cm translation is
    CHECK(rt_write_due(0.9f, 2.66f, 0.0f, 0.9f, 2.66f, 6.f * pi / 180.f, 0.5f));
    // translation alone: the old 5 cm rule, unchanged
    CHECK(rt_write_due(0.0f, 0.0f, 0.0f, 0.06f, 0.0f, 0.0f, 0.5f));
    CHECK(not rt_write_due(0.0f, 0.0f, 0.0f, 0.04f, 0.0f, 0.0f, 0.5f));
    // the same aperture with its direction reversed (yaw + π) is the same segment but its jambs swap: a 1 m move — written
    // (the edge's frame flipped; consumers that read the x axis must see it)
    CHECK(rt_write_due(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, pi, 0.5f));
    std::print("{}\n", fails == 0 ? "test_door_rt_publish: ALL PASS" : "test_door_rt_publish: FAILURES");
    return fails == 0 ? 0 : 1;
}
