// Task 1 (local-motion policy): the TRUE Shadow body — webots-shadow shadow.stl ∪ the TWO drive wheels of Shadow.proto
// (Cylinder r 0.1, h 0.05 along the lateral axle, centred x = ±0.259 ⇒ each wheel x ∈ [0.234, 0.284], y ∈ [−0.10, 0.10]).
// The old polygon modelled four mecanum wheels (|x|max 0.2716) and is wrong for the simulated robot.
// Build+run (from controller/):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_shadow_footprint.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tsf && /tmp/tsf
#include "../../common/robot_footprint/robot_footprint.h"

#include <cstdio>
#include <vector>

// Raw hull of shadow.stl (z<=1.45) ∪ the two wheels, printed by tools/shadow_footprint.py. shadow() must CONTAIN it.
static const std::vector<Eigen::Vector2f> kRaw{
    {-0.284000f, -0.100000f}, {-0.178020f, -0.223121f}, {-0.175411f, -0.224974f}, {-0.173577f, -0.225663f},
    {-0.173251f, -0.225743f}, {-0.171000f, -0.226000f}, {-0.040506f, -0.229904f}, {-0.035000f, -0.230000f},
    {+0.035000f, -0.230000f}, {+0.039490f, -0.229936f}, {+0.168805f, -0.226000f}, {+0.174315f, -0.224684f},
    {+0.176680f, -0.223116f}, {+0.176965f, -0.222908f}, {+0.178241f, -0.221530f}, {+0.284000f, -0.100000f},
    {+0.284000f, +0.100000f}, {+0.216794f, +0.186696f}, {+0.213404f, +0.190972f}, {+0.212851f, +0.191630f},
    {+0.203286f, +0.201556f}, {+0.177411f, +0.223674f}, {+0.176398f, +0.224418f}, {+0.175110f, +0.225116f},
    {+0.173543f, +0.225671f}, {+0.171000f, +0.226000f}, {+0.139993f, +0.229539f}, {+0.130000f, +0.230000f},
    {-0.130000f, +0.230000f}, {-0.143711f, +0.229131f}, {-0.171000f, +0.226000f}, {-0.175218f, +0.225067f},
    {-0.177527f, +0.223576f}, {-0.199002f, +0.205271f}, {-0.208042f, +0.196935f}, {-0.213269f, +0.191135f},
    {-0.214106f, +0.190122f}, {-0.284000f, +0.100000f},
};

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
    // ★ the simplification may only GROW the body: every raw-hull vertex is inside or on the polygon (margin 0, 1e-5 slack).
    int outside = 0;
    for (const auto &q : kRaw)
        for (std::size_t i = 0, n = p.size(); i < n; ++i)
        {
            const auto &a = p[i], &b = p[(i + 1) % n];
            if ((b.x() - a.x()) * (q.y() - a.y()) - (b.y() - a.y()) * (q.x() - a.x()) < -1e-5f) { ++outside; break; }
        }
    CHECK(kRaw.size() == 38 and outside == 0, "every raw-hull vertex is contained by shadow()");
    CHECK(rc::RobotFootprint::self_test(), "self_test");
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
