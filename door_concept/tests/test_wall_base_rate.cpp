// Offline harness for src/core/wall_base_rate.h (the per-wall base rate as a function of the OTHER doors' beliefs).
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_wall_base_rate.cpp -o /tmp/test_wall_base_rate && /tmp/test_wall_base_rate
#include "../src/core/wall_base_rate.h"
#include <cmath>
#include <cstdio>

using namespace rc::door::core;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static float logit(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }

int main()
{
    const WallRates seed{};
    // seed_means
    CHECK(std::abs(seed.first.mean() - 0.4) < 1e-9);
    CHECK(std::abs(seed.another.mean() - 0.01) < 1e-9);
    // no_others_is_first
    CHECK(std::abs(wall_base_rate(seed, {}) - std::log(0.4f / 0.6f)) < 1e-5f);
    // certain_other_is_another
    {
        const float r[] = {1.0f};
        CHECK(std::abs(wall_base_rate(seed, r) - std::log(0.01f / 0.99f)) < 1e-5f);
    }
    // uncertain_other_pays_part
    {
        const float r[] = {0.5f};
        const float v = wall_base_rate(seed, r), f = logit(0.4), a = logit(0.01);
        CHECK(v > a and v < f);
        CHECK(std::abs(v - (0.5f * f + 0.5f * a)) < 1e-5f);
    }
    // order_independent
    {
        const float r1[] = {0.2f, 0.7f}, r2[] = {0.7f, 0.2f};
        CHECK(wall_base_rate(seed, r1) == wall_base_rate(seed, r2));
    }
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail != 0;
}
