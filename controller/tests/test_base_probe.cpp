// BaseProbe (controller_base_probe.h, Task 7): the open-loop `probe <v> <w> <s>` used by tools/measure_deadband.py —
// grammar (locale-proof), limits, the predicted sweep it is checked against, and the hold-then-stop timing.
// Build+run (pure Eigen + std, from controller/):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 tests/test_base_probe.cpp -o /tmp/tbp && /tmp/tbp
#include "../src/controller_base_probe.h"

#include <clocale>
#include <cmath>
#include <numbers>
#include <print>
#include <string_view>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // the agent runs under es_ES: parsing must not depend on it (CLAUDE.md)
    using rc::BaseProbe;
    // 1. Grammar.
    {
        const auto p = BaseProbe::parse("probe 0.03 -0.1 2.5");
        CHECK(p.has_value());
        if (p) { CHECK(std::abs(p->v - 0.03f) < 1e-6f and std::abs(p->w + 0.1f) < 1e-6f and std::abs(p->hold_s - 2.5f) < 1e-6f); }
        CHECK(not BaseProbe::parse("probe 0.03 0.1").has_value());          // three numbers
        CHECK(not BaseProbe::parse("probe 0,03 0 2").has_value());          // a comma is not a decimal point
        CHECK(not BaseProbe::parse("probe 0.5 0 2").has_value());           // above the probe's |v| limit
        CHECK(not BaseProbe::parse("probe 0 1.0 2").has_value());           // above its |w| limit
        CHECK(not BaseProbe::parse("probe 0.02 0 0").has_value());          // a hold must be positive
        CHECK(not BaseProbe::parse("probe 0.02 0 9").has_value());          // and short
        CHECK(not BaseProbe::parse("move 0.3").has_value());
    }
    // 2. Predicted sweep: forward at theta = 0 is +y; + w turns CCW (theta grows), like RelativeMotion.
    {
        const auto poses = BaseProbe::predicted_poses({0.1f, 0.f, 2.f}, Eigen::Vector2f(1.f, 2.f), 0.f, 0.05f);
        CHECK(not poses.empty());
        CHECK((poses.back().first - Eigen::Vector2f(1.f, 2.2f)).norm() < 1e-3f);
        const auto turn = BaseProbe::predicted_poses({0.f, 0.2f, 2.f}, Eigen::Vector2f(0.f, 0.f), 0.f, 0.05f);
        CHECK(std::abs(turn.back().second - 0.4f) < 1e-3f and turn.back().first.norm() < 1e-6f);
        const auto back = BaseProbe::predicted_poses({-0.1f, 0.f, 1.f}, Eigen::Vector2f(0.f, 0.f), std::numbers::pi_v<float> / 2, 0.05f);
        CHECK((back.back().first - Eigen::Vector2f(0.1f, 0.f)).norm() < 1e-3f);   // facing -x, reversing goes +x
    }
    // 3. Hold, then stop and end.
    {
        BaseProbe b;
        CHECK(not b.active());
        b.start({0.02f, -0.05f, 1.5f}, 1000);
        CHECK(b.active());
        auto c = b.step(1000);
        CHECK(c.adv == 0.02f and c.rot == -0.05f);
        c = b.step(2499);
        CHECK(b.active() and c.adv == 0.02f);
        c = b.step(2500);
        CHECK(not b.active() and c.adv == 0.f and c.rot == 0.f);
        CHECK(b.view().t_start_ms == 1000 and b.view().t_end_ms == 2500);
        b.start({0.02f, 0.f, 1.f}, 3000);
        b.cancel();
        CHECK(not b.active());
    }
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
