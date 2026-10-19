// A short route must not inherit the previous route's control polygon (try92/try95: the elastic band rebuilt the
// old route over a <0.8 m approach and the robot parked at the old route's end).
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_route_spline_short.cpp \
//       src/route_spline.cpp src/route_optimizer.cpp -o /tmp/test_route_spline_short && /tmp/test_route_spline_short
#include "route_spline.h"

#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool ok, const char *what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++fails;
}

int main()
{
    const auto free_all = [](const Eigen::Vector2f &, float) { return true; };
    rc::RouteSpline s;

    const std::vector<Eigen::Vector2f> long_path{{1.256f, 1.882f}, {0.8f, 2.3f}, {0.4f, 2.7f}, {0.139f, 3.029f}};
    check(s.build(long_path, 0.05f, free_all, 0.40f), "long route builds");
    check(s.control_points().size() >= 4, "long route has a deformable polygon (>= 4 control points)");

    const std::vector<Eigen::Vector2f> short_path{{0.13f, 3.04f}, {0.20f, 2.42f}};   // 0.62 m (try95) → < 4 points at 0.40 m
    check(s.build(short_path, 0.05f, free_all, 0.40f), "short route builds");
    const auto &c = s.control_points();
    check(c.size() < 4, "short route's polygon is NOT the old one (< 4 points ⇒ the band skips it)");
    check(not c.empty() and (c.front() - short_path.front()).norm() < 1e-4f, "polygon starts at the new path");
    check((s.samples().back() - short_path.back()).norm() < 0.41f, "samples end near the new goal");

    const std::vector<Eigen::Vector2f> one_point{{0.f, 0.f}};
    check(not s.build(one_point, 0.05f, free_all, 0.40f), "degenerate route fails");
    check(s.control_points().empty(), "a failed build leaves no stale polygon");

    std::printf("%s (%d failed)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
