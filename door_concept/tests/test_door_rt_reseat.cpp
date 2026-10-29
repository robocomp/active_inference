// Room-transition rule 1 (2026-10-29): room_concept moves a wall IN PLACE when the layout changes; the door's RT (offset
// along the wall) goes with it, so the PUBLISHED door can leave the belief while the belief stands still. The RT dead-band
// must compare with the published pose, so a slid door is re-seated on the next cycle.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_rt_reseat.cpp -o /tmp/t && /tmp/t
#include <cmath>
#include <numbers>
#include <print>
#include "../src/door_rt_publish.h"

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(float a, float b) { return std::abs(a - b) < 1e-4f; }

int main()
{
    using namespace rc::door;
    const float pi = std::numbers::pi_v<float>;
    // Door seen at (2, 3), facing the wall's tangent; the wall y = 3 from x 0..4 (mid (2, 3), yaw π). The door's RT
    // under that wall: along = 0, across = 0, lyaw = 0.
    const float cx = 2.f, cy = 3.f, yaw = pi, half_w = 0.45f;
    // composition round-trip
    const auto p0 = door_room_pose({2.f, 3.f}, pi, 0.f, 0.f, 0.f);
    CHECK(near(p0.cx, 2.f) and near(p0.cy, 3.f) and near(std::remainder(p0.yaw - pi, 2.f * pi), 0.f));
    const auto p1 = door_room_pose({1.f, 1.f}, 0.5f * pi, 0.3f, -0.2f, 0.1f);   // along +y, across -x
    CHECK(near(p1.cx, 1.2f) and near(p1.cy, 1.3f) and near(p1.yaw, 0.5f * pi + 0.1f));

    // Belief == last written: the old rule writes nothing.
    CHECK(not rt_write_due(cx, cy, yaw, cx, cy, yaw, half_w));
    // room_concept moved the wall: the layout grew to x 0..6, mid (3, 3). The door rides the midpoint to (3, 3).
    const std::optional<PublishedPose> slid = door_room_pose({3.f, 3.f}, pi, 0.f, 0.f, 0.f);
    CHECK(near(slid->cx, 3.f));
    CHECK(rt_write_due_published(slid, cx, cy, yaw, cx, cy, yaw, half_w));   // re-seated although belief == last written
    // The wall moved by less than the dead-band: no churn.
    const std::optional<PublishedPose> nudged = door_room_pose({2.02f, 3.f}, pi, 0.f, 0.f, 0.f);
    CHECK(not rt_write_due_published(nudged, cx, cy, yaw, cx, cy, yaw, half_w));
    // Published == belief but last written differs (another frame wrote it): nothing to do.
    CHECK(not rt_write_due_published(PublishedPose{cx, cy, yaw}, 0.f, 0.f, 0.f, cx, cy, yaw, half_w));
    // No published pose readable (no edge yet): falls back to last written.
    CHECK(rt_write_due_published(std::nullopt, 0.f, 0.f, 0.f, cx, cy, yaw, half_w));
    CHECK(not rt_write_due_published(std::nullopt, cx, cy, yaw, cx, cy, yaw, half_w));

    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}
