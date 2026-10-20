// Task 1 (local-motion policy): the TRUE Shadow body — webots-shadow shadow.stl ∪ the TWO drive wheels of Shadow.proto
// (Cylinder r 0.1, h 0.05 along the lateral axle, centred x = ±0.259 ⇒ each wheel x ∈ [0.234, 0.284], y ∈ [−0.10, 0.10]).
// The old polygon modelled four mecanum wheels (|x|max 0.2716) and is wrong for the simulated robot.
// Build+run (from controller/):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_shadow_footprint.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tsf && /tmp/tsf
#include "../../common/robot_footprint/robot_footprint.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c, m) do { if (not (c)) { ++fails; std::printf("  FAIL: %s\n", m); } } while (0)

int main()
{
    const auto f = rc::RobotFootprint::shadow();
    const auto &p = f.polygon();
    float xmax = -1e9f, area = 0.f;
    for (const auto &v : p) xmax = std::max(xmax, v.x());
    for (std::size_t i = 0, n = p.size(); i < n; ++i)
    {
        const auto &a = p[i], &b = p[(i + 1) % n];
        area += 0.5f * (a.x() * b.y() - b.x() * a.y());
    }
    std::printf("  verts %zu  x_max %.4f  area %.4f  inscribed %.4f  circumscribed %.4f\n",
                p.size(), xmax, area, f.inscribed_radius(), f.circumscribed_radius());
    CHECK(xmax >= 0.2835f and xmax <= 0.2845f, "x_max is the wheel's outer face 0.284");
    CHECK(f.contains({+0.280f, 0.f}) and f.contains({-0.280f, 0.f}), "wheel region is inside");
    CHECK(not f.contains({+0.280f, +0.15f}) and not f.contains({+0.280f, -0.15f}) and
          not f.contains({-0.280f, +0.15f}) and not f.contains({-0.280f, -0.15f}), "beside the wheel, fore/aft, is outside");
    CHECK(f.contains({0.f, +0.225f}) and f.contains({0.f, -0.225f}), "body front/back");
    CHECK(area > 0.f, "CCW");
    bool convex = true;
    for (std::size_t i = 0, n = p.size(); i < n; ++i)
    {
        const auto &a = p[i], &b = p[(i + 1) % n], &c = p[(i + 2) % n];
        if ((b.x() - a.x()) * (c.y() - b.y()) - (b.y() - a.y()) * (c.x() - b.x()) < -1e-6f) convex = false;
    }
    CHECK(convex, "convex");
    CHECK(f.contains({0.f, 0.f}), "origin inside");
    CHECK(rc::RobotFootprint::self_test(), "self_test");
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
