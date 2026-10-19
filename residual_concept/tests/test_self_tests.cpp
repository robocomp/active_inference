/*
 * test_self_tests.cpp — runs the pure (DSR-free) self_test()s that live inside residual_concept's sources.
 * They were written as in-source properties but nothing called them; this is the runner. Build + run:
 * tests/run_tests.sh
 */
#include "../src/residual_occupancy_grid.h"
#include "../src/residual_floor_plane.h"
#include "../src/residual_clusterer.h"
#include "../src/residual_belief.h"
#include "../src/residual_zed_boost.h"
#include "../src/residual_semantic.h"

#include <cstdio>

int main()
{
    int fail = 0;
    if (not rc::OccupancyGrid::self_test()) ++fail;
    if (not rc::floor_plane_self_test())    ++fail;
    if (not rc::ResidualClusterer::self_test()) ++fail;
    if (not rc::ResidualBelief::self_test())    ++fail;
    if (not rc::zed_boost_self_test())          ++fail;
    if (not rc::semantic_self_test())           ++fail;
    std::printf("\ntest_self_tests %s (%d failing suite(s))\n", fail == 0 ? "PASS" : "FAIL", fail);
    return fail == 0 ? 0 : 1;
}
