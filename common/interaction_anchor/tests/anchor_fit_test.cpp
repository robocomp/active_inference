// Standalone harness for interaction_anchor/anchor_fit.h.
// g++ -std=c++23 -O1 -Icommon common/interaction_anchor/tests/anchor_fit_test.cpp -o /tmp/afit && /tmp/afit
#include "interaction_anchor/anchor_fit.h"
#include <cassert>
#include <cmath>
#include <cstdio>

int main()
{
    using namespace rc;
    assert(std::abs(anchor_std_normal_cdf(0.f) - 0.5f) < 1e-6f);

    const float half = 0.31f;
    assert(anchor_p_fit(0.9f, half, 0.f, 0.02f) > 0.99f);      // wide opening, small sigma
    assert(anchor_p_fit(0.5f, half, 0.f, 0.02f) < 0.01f);      // narrower than the body
    const float p_small = anchor_p_fit(0.9f, half, 0.f, 0.02f);
    const float p_big   = anchor_p_fit(0.9f, half, 0.f, 0.5f);
    assert(p_big < p_small and p_big > 0.5f);                  // more sigma moves P toward 0.5
    const float q_small = anchor_p_fit(0.5f, half, 0.f, 0.02f);
    const float q_big   = anchor_p_fit(0.5f, half, 0.f, 0.5f);
    assert(q_big > q_small and q_big < 0.5f);

    const auto g = anchor_clear_gap(0.9f, nullptr);            // no leaf: the opening itself
    assert(g.centre == 0.f and std::abs(g.width - 0.9f) < 1e-6f);
    assert(std::abs(anchor_p_fit_leaves(0.9f, {}, half, 0.f, 0.02f) - p_small) < 1e-6f);
    std::puts("anchor_fit_test OK");
    return 0;
}
