// Build+run (from controller/): g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_door_leaf_draw.cpp -o /tmp/tdld && /tmp/tdld
#include "../src/door_leaf_geometry.h"
#include <cstdio>
#include <numbers>
using namespace rc::doordraw;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static bool near(V2 a, V2 b, float e = 1e-4f) { return (a - b).norm() < e; }
int main()
{
    const V2 c(2.f, 1.f), u(1.f, 0.f);
    const float w = 0.9f, hp = std::numbers::pi_v<float> / 2;
    auto l0 = leaf_from_aperture(c, u, w, -1, +1, 0.f);
    CHECK("shut leaf spans the aperture", near(l0.hinge, V2(1.55f, 1.f)) and near(l0.tip, V2(2.45f, 1.f)));
    auto l1 = leaf_from_aperture(c, u, w, -1, +1, hp);
    CHECK("90 deg west hinge swings to +n", near(l1.tip, V2(1.55f, 1.9f)));
    auto l2 = leaf_from_aperture(c, u, w, +1, -1, hp);
    CHECK("90 deg east hinge swings to -n", near(l2.hinge, V2(2.45f, 1.f)) and near(l2.tip, V2(2.45f, 0.1f)));
    auto l3 = leaf_from_aperture(c, u, w, +1, +1, 0.f);
    CHECK("shut leaf from the other jamb also spans it", near(l3.tip, V2(1.55f, 1.f)));
    CHECK("leaf length is the width", std::abs((l1.tip - l1.hinge).norm() - w) < 1e-4f);
    Eigen::Affine2f T = Eigen::Affine2f::Identity();
    T.rotate(hp); T.translate(V2(1.f, 0.f));
    CHECK("anchor frame origin maps through T", near(anchor_to_room({0, 0}, V2(0, 1), 0.f, T), T * V2(0, 1)));
    CHECK("anchor x axis follows yaw", near(anchor_to_room({1, 0}, V2(0, 0), hp, T), T * V2(0, 1)));
    auto wd = wedge(l1.hinge, l1.tip, 0.2f);
    CHECK("wedge endpoints at leaf length", wd.size() == 10 and std::abs((wd[1] - wd[0]).norm() - w) < 1e-4f
          and std::abs((wd.back() - wd[0]).norm() - w) < 1e-4f);
    CHECK("no sigma, no wedge", wedge(l1.hinge, l1.tip, 0.f).empty());
    const std::vector<V2> poly{{0, 0}, {4, 0}, {4, 3}, {0, 3}};
    auto o = aperture_overhang(V2(3.8f, 0.f), u, 0.9f, poly);
    CHECK("aperture past the east end overhangs hi", std::abs(o.hi - 0.25f) < 1e-4f and o.lo == 0.f);
    auto o2 = aperture_overhang(V2(2.f, 0.f), u, 0.9f, poly);
    CHECK("inside: no overhang", o2.lo == 0.f and o2.hi == 0.f);
    auto o3 = aperture_overhang(V2(0.2f, 3.f), -u, 0.9f, poly);   // u reversed on the top wall (t runs against u)
    CHECK("reversed u: overhang on the hi end", std::abs(o3.hi - 0.25f) < 1e-4f and o3.lo == 0.f);
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
