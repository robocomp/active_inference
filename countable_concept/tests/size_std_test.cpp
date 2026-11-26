// size_std_test — S1: the published size std (size_std_m = {width_m, depth_m, height_m} 1-σ) and the neighbour reader's mapping.
// Every expected value is exact arithmetic on the belief's own Σ (state order cx, cy, H, w, h, yaw):
//   floor class : {√Σ33, √Σ44, √Σ22}      hanging : {√Σ33, √Σ44, 0} (height_m is the class constant e, not H)
//   c2v class   : the same entries (the state's own w, h are what is published)
//   reader      : {σ_w, σ_h} = the first two entries; absent / wrong size / negative / non-finite ⇒ nullopt (the manifest fallback).
// Run: ./size_std_test
#include <cmath>
#include <cstdio>
#include <limits>
#include <tuple>
#include "countable_belief.h"
#include "countable_size_std.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static rc::CountableBelief make(float extent, bool c2v)
{
    rc::CountableBeliefParams p;
    p.vertical_extent_m = extent;
    p.symmetric = c2v;
    p.prior_footprint_std = 0.08f; p.prior_depth_std = 0.05f;   // the class prior makes w, h and H distinguishable in the engine's own Σ
    rc::CountableBelief b(rc::CountableBeliefState{1.0f, 2.0f, 0.75f, 0.90f, 0.50f, 0.3f}, p);
    return b;
}

int main()
{
    using rc::countable::VerticalSpan;
    for (const auto& [name, extent, c2v] : {std::tuple{"floor", -1.0f, false}, std::tuple{"hanging", 0.5f, false}, std::tuple{"c2v", -1.0f, true}})
    {
        const auto b = make(extent, c2v);
        const auto sd = rc::published_size_std(b.covariance(), VerticalSpan{extent});
        std::printf("  %-8s {%.4f %.4f %.4f}\n", name, sd[0], sd[1], sd[2]);
        CHECK(sd.size() == 3);
        const auto& S = b.covariance();                                         // the engine's own θθ Σ
        std::printf("  Sigma diag %.5g %.5g %.5g %.5g %.5g %.5g\n", S(0,0), S(1,1), S(2,2), S(3,3), S(4,4), S(5,5));
        CHECK(S(3, 3) > 0.0f and S(4, 4) > 0.0f and S(2, 2) > 0.0f and S(3, 3) != S(4, 4) and S(2, 2) != S(3, 3));   // axes distinguishable
        CHECK(sd[0] == std::sqrt(S(3, 3)));                                     // width_m  = state w
        CHECK(sd[1] == std::sqrt(S(4, 4)));                                     // depth_m  = state h
        CHECK(sd[2] == (extent > 0.0f ? 0.0f : std::sqrt(S(2, 2))));            // height_m = H floor-anchored; hanging: the constant e
    }
    // a rounding-negative variance is a zero std, not NaN
    { Eigen::Matrix<float, 6, 6> S = Eigen::Matrix<float, 6, 6>::Zero(); S(3, 3) = -1e-12f;
      CHECK(rc::published_size_std(S, VerticalSpan{})[0] == 0.0f); }
    // publish policy (fix round 1): once after seeding even when the covariance gate did not fire; then only with the gate; σ follows Σ
    {
        const auto b = make(-1.0f, false);
        Eigen::Matrix<float, 6, 6> S = b.covariance();
        bool ever = false;
        const auto first = rc::size_std_to_publish(false, ever, S, VerticalSpan{});    // geometry settled, gate silent
        CHECK(first.has_value() and ever);
        CHECK(not rc::size_std_to_publish(false, ever, S, VerticalSpan{}).has_value()); // nothing moved ⇒ nothing written
        S(3, 3) *= 0.25f; S(4, 4) *= 0.25f; S(2, 2) *= 0.25f;                          // Σ shrinks, extents static; the gate fires
        const auto again = rc::size_std_to_publish(true, ever, S, VerticalSpan{});
        CHECK(again.has_value());
        CHECK(again and std::abs((*again)[0] - 0.5f * (*first)[0]) < 1e-6f and std::abs((*again)[2] - 0.5f * (*first)[2]) < 1e-6f);
    }
    // reader mapping and fallback
    { const auto f = rc::neighbour_footprint_std({0.07f, 0.03f, 0.04f}); CHECK(f and (*f)[0] == 0.07f and (*f)[1] == 0.03f); }
    CHECK(not rc::neighbour_footprint_std({}));                                  // legacy agent: nothing published
    CHECK(not rc::neighbour_footprint_std({0.07f, 0.03f}));                      // wrong size
    CHECK(not rc::neighbour_footprint_std({0.07f, -0.03f, 0.04f}));              // negative
    CHECK(not rc::neighbour_footprint_std({0.07f, std::numeric_limits<float>::quiet_NaN(), 0.04f}));
    std::printf(bad ? "size_std_test: %d FAILED\n" : "size_std_test: OK\n", bad);
    return bad != 0;
}
