// standpoint_free_test.cpp — P(footprint free) over an occupancy grid (common/nbv/standpoint_free.h). Standalone.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../standpoint_free.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-64s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

int main()
{
    using rc::nbv::GridView;
    // A 1.0 x 1.0 m square body (robot frame), cell 0.1 m, grid 40 x 40 cells from (-2,-2).
    const std::vector<Eigen::Vector2f> body{{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
    std::vector<float> occ(40 * 40, 0.0f);
    GridView g{-2.0f, -2.0f, 0.1f, 40, 40, occ, 0.5f};

    std::printf("standpoint_free_test\n");
    CHECK(std::abs(rc::nbv::p_footprint_free(g, {0.0f, 0.0f}, 0.0f, body) - 1.0) < 1e-6, "all-free grid gives 1");

    occ[20 * 40 + 20] = 1.0f;   // cell centred (0.05, 0.05): under the body at the origin
    CHECK(rc::nbv::p_footprint_free(g, {0.0f, 0.0f}, 0.0f, body) == 0.0, "one certainly-occupied cell under the body gives 0");
    CHECK(rc::nbv::p_footprint_free(g, {1.0f, 1.0f}, 0.0f, body) > 0.999999, "the same cell outside the body does not count");
    occ[20 * 40 + 20] = 0.2f;
    CHECK(std::abs(rc::nbv::p_footprint_free(g, {0.0f, 0.0f}, 0.0f, body) - 0.8) < 1e-6, "p_occ 0.2 on one cell gives 0.8");

    // Half outside the grid: body centred on the grid's left edge x = -2, lattice cells with x centre < -2 are outside.
    std::fill(occ.begin(), occ.end(), 0.1f);
    const double p = rc::nbv::p_footprint_free(g, {-2.0f, 0.0f}, 0.0f, body);
    const double want = std::pow(0.9, 50) * std::pow(0.5, 50);   // 5 x 10 cells inside (0.9 each), 5 x 10 outside (prior 0.5)
    CHECK(std::abs(p - want) < 1e-12 + 1e-6 * want, "half outside: inside cells x 0.5^(outside cells)");

    // The quarter turn: a 1.0 x 0.2 bar is long along the robot's forward (+y). At yaw 0 forward is room +x.
    const std::vector<Eigen::Vector2f> bar{{-0.1f, -0.5f}, {0.1f, -0.5f}, {0.1f, 0.5f}, {-0.1f, 0.5f}};
    std::fill(occ.begin(), occ.end(), 0.0f);
    occ[20 * 40 + 27] = 1.0f;   // the cell centred (0.75, 0.05)
    CHECK(rc::nbv::p_footprint_free(g, {0.0f, 0.0f}, 0.0f, bar) == 1.0, "yaw 0 at the origin: the bar spans x +-0.5, the cell is outside");
    CHECK(rc::nbv::p_footprint_free(g, {0.4f, 0.0f}, 0.0f, bar) == 0.0, "yaw 0 at x=0.4: the bar spans x -0.1..0.9, it covers the cell");
    CHECK(rc::nbv::p_footprint_free(g, {0.4f, 0.0f}, 1.57079633f, bar) == 1.0, "yaw pi/2 at x=0.4: the bar spans y, x 0.3..0.5, the cell is outside");
    std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
