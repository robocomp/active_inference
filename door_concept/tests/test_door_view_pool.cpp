// test_door_view_pool — Task 5b, D4: the image-geometry existence channel (leaf contour + depth + doorway frame) pooled
// by VIEW NOVELTY. Consecutive frames from (nearly) the same viewpoint are the same observation repeated; they refine
// its verdict, they do not multiply it. The live channel instead added one full observation's worth (±2.83 nats,
// log(pd/pc)) on EVERY frame: on the tapes it saturated on 95 % of cycles and, its sign flipping with the frame noise,
// walked a correct, shut DOOR_0 between the +4 clamp and the −4 floor (tourR1 door_4: 45 cycles at the floor).
//   1. one static view, the verdict flipping ±3 frame to frame: the total never exceeds ONE observation's worth, and ends
//      where the pooled (mean) verdict puts it;
//   2. one static view, a steady +3: one observation (llr·tanh 3), not 100;
//   3. ten genuinely new views (the bearing stepping by many angular half-sizes): ten observations — independence kept;
//   4. a door that is NOT there (a steady −3) seen from new views is still refuted at full weight;
//   5. moving within the door's own angular half-size is the same view; leaving it by several is a new one — the scale
//      is the object's geometry (as ExistenceBelief::note_viewpoint), no tuned constant;
//   6. with L at its clamp, a view that revises its verdict moves L by one observation, not by the revision's full swing.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_view_pool.cpp -o /tmp/test_door_view_pool && /tmp/test_door_view_pool
#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>

#include "../src/door_view_pool.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
const float kLlr = std::log(0.85f / 0.05f);   // one confident observation (SensorModel defaults): 2.83 nats
// a camera on a circle of radius r around a door at the origin, at bearing b (rad)
Eigen::Vector2f cam_at(float b, float r) { return {r * std::cos(b), r * std::sin(b)}; }
}  // namespace

int main()
{
    const Eigen::Vector2f door{0.f, 0.f};
    const float R = 0.5f;   // half the door's width

    std::println("1. static view, the verdict flipping ±3");
    {
        rc::door::ViewPool pool;
        float L = 0.f, worst = 0.f;
        for (int k = 0; k < 100; ++k)
        {
            L += pool.add(cam_at(0.3f, 4.f), door, R, (k % 2 == 0) ? 3.f : -3.f, kLlr);
            worst = std::max(worst, std::abs(L));
        }
        std::println("    final L {:+.3f}, worst |L| {:.3f} (one observation = {:.2f})", L, worst, kLlr);
        check(worst <= kLlr + 1e-4f, "a static view added more than one observation");
        check(std::abs(L) < 0.1f * kLlr, "a static view's pooled verdict (mean 0) did not end near 0");
    }
    std::println("2. static view, a steady +3");
    {
        rc::door::ViewPool pool;
        float L = 0.f;
        for (int k = 0; k < 100; ++k) L += pool.add(cam_at(0.3f, 4.f), door, R, 3.f, kLlr);
        std::println("    L {:+.3f} (one observation at verdict 3 = {:.3f})", L, kLlr * std::tanh(3.f));
        check(std::abs(L - kLlr * std::tanh(3.f)) < 1e-3f, "a static view did not count as ONE observation");
    }
    std::println("3. ten new views, +1 each");
    {
        rc::door::ViewPool pool;
        float L = 0.f;
        for (int k = 0; k < 10; ++k) L += pool.add(cam_at(0.3f + 0.6f * static_cast<float>(k), 4.f), door, R, 1.f, kLlr);
        std::println("    L {:+.3f} (ten observations = {:.3f})", L, 10.f * kLlr * std::tanh(1.f));
        check(L > 0.95f * 10.f * kLlr * std::tanh(1.f), "new views were not counted as new observations");
    }
    std::println("4. a door that is not there, new views");
    {
        rc::door::ViewPool pool;
        float L = 0.f;
        for (int k = 0; k < 4; ++k) L += pool.add(cam_at(0.3f + 0.6f * static_cast<float>(k), 4.f), door, R, -3.f, kLlr);
        std::println("    L {:+.3f}", L);
        check(L < -0.95f * 4.f * kLlr * std::tanh(3.f), "refutation from new views was damped");
    }
    std::println("5. the scale is the door's angular half-size");
    {
        const float half = std::atan2(R, 4.f);
        rc::door::ViewPool a, b;
        (void)a.add(cam_at(0.3f, 4.f), door, R, 2.f, kLlr);
        const float near = a.add(cam_at(0.3f + 0.1f * half, 4.f), door, R, 2.f, kLlr);   // same view, same verdict
        (void)b.add(cam_at(0.3f, 4.f), door, R, 2.f, kLlr);
        const float far = b.add(cam_at(0.3f + 4.0f * half, 4.f), door, R, 2.f, kLlr);    // a new aspect
        std::println("    second look within 0.1 half-size adds {:+.4f}; 4 half-sizes away adds {:+.4f}", near, far);
        check(std::abs(near) < 0.05f * kLlr, "a look inside the door's own angular size counted as new");
        check(far > 0.95f * kLlr * std::tanh(2.f), "a look several angular sizes away did not count as new");
    }
    std::println("6. L at the +4 clamp: a view that says +3 then -3 moves L by one observation, not two");
    {
        rc::door::ViewPool pool;
        float L = 4.f;
        const auto step = [&](float v)
        {
            const float dL = pool.add(cam_at(0.3f, 4.f), door, R, v, kLlr);
            const float L1 = std::clamp(L + dL, -4.f, 4.f);
            pool.commit(L1 - L);
            L = L1;
        };
        for (int k = 0; k < 20; ++k) step(3.f);
        for (int k = 0; k < 200; ++k) step(-3.f);   // the view's mean verdict turns towards -3
        std::println("    L {:+.3f} (4 less one observation at the pooled verdict ≥ {:+.3f})", L, 4.f - kLlr);
        check(L >= 4.f - kLlr - 1e-3f, "a view revising its own verdict at the clamp moved L by more than one observation");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
