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

static void test_repeats_do_not_flush_history()
{
    std::printf("repeats do not flush a distinct pose\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    a.update(obs(0, 0, 0.f));                          // distinct pose at bearing 0
    for (int i = 0; i < 6; ++i)
        a.update(obs(0, 0, 0.4f));                     // one new view, then five repeats from 23 deg
    bool kept = false;
    for (const auto& ps : a.state().bins[6].poses) kept = kept or std::fabs(ps.x()) < 1e-3f;
    check(kept, "the bin still holds the pose at bearing 0 after the repeats");
    const float before = a.state().bins[6].n;
    a.update(obs(0, 0, 0.f));                          // back to the first place
    check(a.state().bins[6].n - before < 0.01f, "a return to the first pose is still a repeat");
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

static ViewAtlas atlas_with(int k_shape, int k_col, float bearing, int n_views = 3)
{
    ViewAtlas a(AtlasParams{}, 4, 3);
    for (int i = 0; i < n_views; ++i) a.update(obs(k_shape, k_col, bearing + 0.3f * float(i) - 0.3f));
    return a;
}

static void test_log_ratio()
{
    std::printf("log ratio\n");
    ViewAtlas empty(AtlasParams{}, 4, 3);
    const float l0 = empty.log_ratio(obs(0, 0, 0.f), 0.f);
    check(l0 == 0.f, "empty atlas: log ratio is exactly 0");
    check(std::isfinite(l0), "empty atlas: log ratio is finite");

    ViewAtlas a = atlas_with(0, 0, 0.f);
    check(a.log_ratio(obs(0, 0, 0.f), 0.f) > 2.f, "matching shape+colour at a seen bearing: strong evidence for");
    check(a.log_ratio(obs(1, 1, 0.f), 0.f) < -2.f, "different shape+colour at a seen bearing: evidence against");

    // shape evidence must vanish on the unseen side while colour evidence stays
    const float match_far = a.log_ratio(obs(0, 0, kPi), kPi);
    const float other_far = a.log_ratio(obs(1, 0, kPi), kPi);
    const float match_near = a.log_ratio(obs(0, 0, 0.f), 0.f);
    const float other_near = a.log_ratio(obs(1, 0, 0.f), 0.f);
    check(std::fabs(match_far - other_far) < 0.05f, "unseen side: shape no longer discriminates");
    check(match_near - other_near > 2.f, "seen side: shape discriminates");
    check(a.log_ratio(obs(0, 0, 0.f, 1.5f, 1e-6f), 0.f) < 1e-2f, "quality ~0: no evidence either way");

    // exact zero must hold for ANY background covariance, not just the default diagonal one
    AtlasParams Pn;
    Pn.cov_bg << 0.0031f, 0.0007f,
                 0.0007f, 0.0517f;
    ViewAtlas blank(Pn, 4, 3);
    check(blank.log_ratio(obs(0, 0, 0.f, 1.5f, 1.f), 0.f) == 0.f, "empty atlas, off-diagonal cov_bg, q=1: exactly 0");
    check(blank.log_ratio(obs(0, 0, 0.f, 1.5f, 0.37f), 0.f) == 0.f, "empty atlas, off-diagonal cov_bg, q=0.37: exactly 0");
}

static void test_shift_marginal()
{
    std::printf("yaw shift marginalisation\n");
    const auto sharp = shift_log_weights(12, 0.f);
    check(sharp[0] == 0.f and std::isinf(sharp[1]), "sigma 0: all weight on shift 0");
    const auto flat = shift_log_weights(12, 1e3f);
    check(std::fabs(flat[0] - flat[6]) < 1e-3f, "huge sigma: flat over shifts");
    float lse = -std::numeric_limits<float>::infinity();
    for (float w : flat) lse = std::max(lse, w) + std::log1p(std::exp(-std::fabs(lse - w)));
    check(std::fabs(lse) < 1e-3f, "shift weights are normalised");

    ViewAtlas a = atlas_with(0, 0, 0.f);
    const float off = 3.f * 2.f * kPi / 12.f;                 // observed 3 bins away (wrong yaw)
    const float l_fixed = a.log_ratio(obs(0, 0, off), off);
    const float l_flat  = a.log_ratio_marginal(obs(0, 0, off), off, flat);
    const float l_sharp = a.log_ratio_marginal(obs(0, 0, off), off, sharp);
    check(std::fabs(l_sharp - l_fixed) < 1e-4f, "sigma 0 reproduces the fixed-yaw ratio");
    check(l_flat > l_fixed, "unknown yaw: marginalising recovers evidence a wrong yaw hid");
}

static void test_log_ratio_window()
{
    std::printf("window evidence discounts repeat looks\n");
    const auto sharp = shift_log_weights(12, 0.f);
    ViewAtlas a = atlas_with(0, 0, 0.f);
    const ViewObs o = obs(0, 0, 0.f);
    const float single = a.log_ratio_marginal(o, 0.f, sharp);
    check(std::fabs(single) > 1e-3f, "single-view ratio is non-trivial (test is not vacuous)");
    check(std::fabs(log_ratio_window(a, {o, o, o, o, o}, {0.f, 0.f, 0.f, 0.f, 0.f}, sharp) - single) < 1e-4f,
          "5 identical views count as one view");
    const std::vector<float> br = {0.f, 1.5f, 3.0f, -1.5f, -3.0f};
    std::vector<ViewObs> vs;
    float plain = 0.f;
    for (float b : br) { vs.push_back(obs(0, 0, b)); plain += a.log_ratio_marginal(vs.back(), b, sharp); }
    const float w = log_ratio_window(a, vs, br, sharp);
    std::printf("    plain sum %.5f window %.5f\n", plain, w);
    check(std::fabs(w - plain) <= 0.01f * std::fabs(plain), "5 far-apart views are within 1 % of the plain sum");
    check(log_ratio_window(a, {}, {}, sharp) == 0.f, "empty window is 0");
}

static void test_reid()
{
    std::printf("re-identification posterior\n");
    std::vector<ReidCandidate> c = {{8.f, 0.f, 0.f}, {-3.f, 0.f, 0.f}};
    auto p = reid_posterior(c, 0.f, 0.f);
    float s = 0.f; for (float v : p) s += v;
    check(p.size() == 3 and std::fabs(s - 1.f) < 1e-5f, "posterior has candidates + new and sums to 1");
    check(p[0] > 0.99f, "strong appearance evidence picks candidate 0");
    auto pe = reid_posterior({}, 0.f, 0.f);
    check(pe.size() == 1 and pe[0] == 1.f, "no candidates: certainly new");
    check(std::fabs(entropy({0.5f, 0.5f}) - std::log(2.f)) < 1e-5f, "entropy of a fair coin is log 2");

    const Eigen::Vector2f mu(0.f, 0.f);
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.01f;
    const float near_stay = location_log_lik({0.f, 0.f}, mu, cov, 0.f, 50.f);
    check(std::fabs(near_stay - (-std::log(2.f * kPi * 0.01f))) < 1e-3f, "pi_move 0: plain Gaussian at the mean");
    const float far = location_log_lik({20.f, 0.f}, mu, cov, 0.2f, 50.f);
    check(std::fabs(far - std::log(0.2f / 50.f)) < 1e-3f, "far away: only the moved-uniform term remains");
}

static void test_identity_value()
{
    std::printf("identity value of encoding\n");
    const AtlasParams P;
    ViewAtlas a = atlas_with(0, 0, 0.f), b = atlas_with(0, 0, 0.f);
    std::vector<IdHyp> same = {{a.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f},
                               {b.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    std::vector<IdHyp> none = {};
    const float v_none = identity_value(none, 0.f, 0.f, 64, 1);
    check(v_none == 0.f, "no candidates: nothing to learn about identity");
    ViewAtlas empty(P, 4, 3);
    std::vector<IdHyp> blind = {{empty.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    check(identity_value(blind, 0.f, 0.f, 256, 1) < 0.02f, "an atlas that predicts nothing: ~0 value");
    std::vector<IdHyp> one = {{a.hypothesis(0.f, 1.f), background(P, 1.f), 0.f, 0.f}};
    check(identity_value(one, 0.f, 0.f, 256, 1) > 0.1f, "a seen candidate vs new: encoding is worth something");
    check(std::isfinite(identity_value(same, 0.f, 0.f, 256, 1)), "two identical candidates: finite value");
}

static void test_prototype_uncertainty()
{
    std::printf("prototype uncertainty\n");
    // cov_bg == cov_own isolates the evidence term: the blend then cannot change the covariance, so only
    // D·Σn·D can make one view less certain than twenty.
    AtlasParams P; P.cov_bg = P.cov_own;
    ViewAtlas one(P, 4, 3), many(P, 4, 3);
    one.update(obs(0, 0, 0.f));
    for (int i = 0; i < 20; ++i) many.update(obs(0, 0, -0.5f + 0.05f * float(i), 1.0f + 0.3f * float(i)));
    const Gauss2 h1 = one.hypothesis(0.f, 1.f), h20 = many.hypothesis(0.f, 1.f);
    check(h1.cov(1, 1) > h20.cov(1, 1), "colour: one view is less certain than twenty");
    check(h1.cov(0, 0) > h20.cov(0, 0), "shape: one view is less certain than twenty");
    ViewAtlas empty(AtlasParams{}, 4, 3);
    check(empty.log_ratio(obs(0, 0, 0.f), 0.f) == 0.f, "empty atlas: still exactly 0 with the evidence term");
}

static void test_repeat_across_bin_edge()
{
    std::printf("repeat check across a bin edge\n");
    ViewAtlas a(AtlasParams{}, 4, 3);
    const float edge = 0.f;                                   // boundary between bins 5 and 6
    a.update(obs(0, 0, edge - 0.005f));
    float before = 0.f; for (const auto& b : a.state().bins) before += b.n;
    a.update(obs(0, 0, edge + 0.005f));                       // 0.6 deg away, other side of the edge
    float after = 0.f; for (const auto& b : a.state().bins) after += b.n;
    check(after - before < 0.01f, "a look 0.6 deg away across a bin edge is a repeat");
}

static void test_repeat_scale_beyond_bin()
{
    std::printf("repeat check spans bins when the repeat scale exceeds a bin\n");
    auto added = [](float ell) {
        AtlasParams P; P.repeat_ell_rad = ell;
        ViewAtlas a(P, 4, 3);
        a.update(obs(0, 0, 0.f));
        float before = 0.f; for (const auto& b : a.state().bins) before += b.n;
        a.update(obs(0, 0, 0.7f));                            // two bins away
        float after = 0.f; for (const auto& b : a.state().bins) after += b.n;
        return after - before;
    };
    const float wide = added(0.776f), narrow = added(0.10f);
    std::printf("    added: wide %.4f narrow %.4f\n", wide, narrow);
    check(wide < 0.6f, "0.7 rad repeat under repeat_ell_rad 0.776 is discounted");
    check(narrow > 0.95f, "same look under the default 0.10 adds ~1 view");
}

int main()
{
    test_bins_and_wrap();
    test_empty_atlas();
    test_update_and_repeat_discount();
    test_repeats_do_not_flush_history();
    test_count_and_learning_value();
    test_prediction();
    test_quality();
    test_log_ratio();
    test_shift_marginal();
    test_log_ratio_window();
    test_reid();
    test_identity_value();
    test_prototype_uncertainty();
    test_repeat_across_bin_edge();
    test_repeat_scale_beyond_bin();
    std::printf("\nview_atlas_test: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
