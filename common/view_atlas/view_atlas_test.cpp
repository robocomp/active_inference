/*
 * view_atlas_test.cpp — unit tests for view_atlas.h. Standalone (common/run_tests.sh):
 *     g++ -std=c++23 -O1 $(pkg-config --cflags eigen3) view_atlas_test.cpp -o t && ./t
 */
#include "view_atlas.h"

#include <cmath>
#include <cstdio>
#include <numbers>

using namespace rc::atlas;   // brings in kPi (= std::numbers::pi_v<float>) from the header

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (not ok) ++g_fail;
}
static Eigen::VectorXf unit(int dim, int k)
{
    Eigen::VectorXf v = Eigen::VectorXf::Zero(dim);
    v[k] = 1.f;
    return v;
}
static ViewObs obs(int k_shape, int k_col, float bearing, float range = 1.5f, float q = 1.f)
{
    ViewObs o;
    o.shape = unit(4, k_shape); o.colour = unit(3, k_col);
    o.bearing_rad = bearing; o.range_m = range; o.quality = q; o.stamp_ms = 1;
    return o;
}

static void test_bins_and_wrap()
{
    std::printf("bins and wrap\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    check(a.bin_of(0.f) == 6, "bearing 0 is bin 6 of 12");
    check(a.bin_of(kPi - 1e-4f) == 11, "just below +pi is bin 11");
    check(a.bin_of(-kPi + 1e-4f) == 0, "just above -pi is bin 0");
    check(a.bin_of(3.f * kPi) == a.bin_of(kPi), "bearings are wrapped first");
    a.update(obs(0, 0, kPi - 0.01f));
    check(std::fabs(a.gap(-kPi + 0.01f) - 0.02f) < 1e-3f, "gap across +-pi is 0.02 rad, not 2pi-0.02");
}

static void test_empty_atlas()
{
    std::printf("empty atlas\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    check(a.effective_count(0.3f) == a.params().n0, "empty atlas: N(beta) == n0");
    check(std::isinf(a.gap(0.f)), "empty atlas: gap is +inf");
    check(not a.predict(0.f).has_shape, "empty atlas: no shape prediction");
    check(a.kernel(a.gap(0.f)) == 0.f, "kernel(+inf) == 0");
}

static void test_update_and_repeat_discount()
{
    std::printf("update and repeat-look discount\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(0, 0, 0.f));
    const float n1 = a.state().bins[6].n;
    check(std::fabs(n1 - 1.f) < 1e-6f, "first view in an empty bin adds its full quality");
    a.update(obs(0, 0, 0.f));
    check(a.state().bins[6].n - n1 < 0.01f, "an identical repeat look adds ~nothing");
    a.update(obs(1, 0, 0.4f));          // same bin (bin 6 = [0, 30 deg)), 23 deg away
    check(a.state().bins[6].n > 1.9f, "a view 23 deg away counts as a new view");
    check(std::fabs(a.state().bins[6].mean.norm() - 1.f) < 1e-5f, "bin mean stays unit length");
    const float before = a.state().bins[6].n;
    a.update(obs(0, 0, 0.f, 1.5f, 0.f));
    check(a.state().bins[6].n == before, "quality 0 adds nothing");
    ViewObs bad = obs(0, 0, 0.f); bad.shape = Eigen::VectorXf::Ones(7);
    a.update(bad);
    check(a.state().bins[6].n == before, "wrong shape dimension is ignored");
    a.update(obs(0, 0, 0.f, 4.0f));     // same bearing, much farther: log-range differs by ~1
    check(a.state().bins[6].n > before + 0.5f, "same bearing at a very different range is a new view");
}

static void test_count_and_learning_value()
{
    std::printf("effective count and learning value\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(0, 0, 0.f));
    const float n0 = a.params().n0;
    check(std::fabs(a.effective_count(0.f) - (n0 + 1.f)) < 1e-3f, "N(0) == n0 + 1 after one view at 0");
    check(a.effective_count(kPi) < n0 + 0.01f, "N(pi) ~ n0: the far side is unseen");
    check(a.learning_value(0.f, 1.f) < a.learning_value(kPi, 1.f), "seen side is worth less than the unseen side");
    check(a.learning_value(kPi, 0.f) == 0.f, "quality 0 is worth nothing");
    check(a.learning_value(0.f, 1.f) > 0.f, "learning value is positive");
}

static void test_prediction()
{
    std::printf("prediction\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(2, 0, 0.f));
    auto p = a.predict(kPi / 2);
    check(p.has_shape, "a populated atlas predicts a shape everywhere");
    check(std::fabs(p.gap_rad - kPi / 2) < 1e-4f, "gap to the only view is 90 deg");
    check(std::fabs(p.shape[2] - 1.f) < 1e-5f, "prediction is the stored mean");
}

static void test_quality()
{
    std::printf("view quality\n");
    check(view_quality(0.f, 1.f, 100000, 2000.f) > 0.99f, "large, untruncated, visible mask ~ 1");
    check(view_quality(0.5f, 1.f, 100000, 2000.f) < 0.51f, "half truncated ~ halves quality");
    check(view_quality(0.f, 1.f, 100, 2000.f) < 0.06f, "tiny mask has low quality");
    check(view_quality(0.f, 1.f, 0, 2000.f) == 0.f, "empty mask has zero quality");
}

int main()
{
    test_bins_and_wrap();
    test_empty_atlas();
    test_update_and_repeat_discount();
    test_count_and_learning_value();
    test_prediction();
    test_quality();
    std::printf("\nview_atlas_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
