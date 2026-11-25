// depth_order_test — the depth ordering of a LiDAR return against the object along the camera's line of sight (F1 round 7 B,
// src/countable_depth_order.h). Toilet geometry of P1 p1_toilet2: a 0.43 × 0.64 × 0.92 m box whose back is 0.02 m off a wall,
// the camera 2.6 m in front at 0.98 m. Before this change a return's label was the mask's q whatever its depth (a wall return
// beside the narrow bowl, inside the blurred mask edge, carried m ≈ 0.6). Checked:
//   (1) a wall return BESIDE the box (the line passes it laterally, the return 0.4 m beyond): P(behind) ≈ 1, label ≈ ε;
//   (2) a wall return straight behind the box, 0.18 m beyond its back face: ≈ 1;
//   (3) a return on the visible front face, and (4) one in front of the box: ≈ 0 (the label keeps q);
//   (5) a return exactly at the far face reads ½, and P(behind) is continuous as the return slides across the box's side edge
//       (the line going from crossing the box to missing it).
// Run: ./depth_order_test
#include <cmath>
#include <cstdio>

#include "countable_depth_order.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main()
{
    std::printf("depth_order_test\n");
    // box centred at the origin, w (x) 0.43, h (y, depth) 0.64, H 0.92; back face at y = +0.32, wall at y = +0.34
    const rc::LosBox box{0.0, 0.0, 0.92, 0.43, 0.64, 0.0, 0.017 * 0.017, 0.022 * 0.022, 0.015 * 0.015, 0.044 * 0.044, 0.024 * 0.024};
    const Eigen::Vector3d C(0.0, -2.6, 0.98);
    const double sm = std::sqrt(0.03 * 0.03 + 0.01 * 0.01), eps = 0.001, q = 0.6;
    const auto pb = [&](const Eigen::Vector3d& p) { return rc::behind_object_probability(C, p, box, sm); };
    const double p1 = pb({-0.30, 0.34, 0.45});          // (1) beside the 0.43 m box (|x| 0.30 > 0.215), on the wall
    const double p2 = pb({0.0, 0.50, 0.45});            // (2) straight behind, 0.18 m beyond the back face
    const double p3 = pb({0.0, -0.32, 0.50});           // (3) on the front face
    const double p4 = pb({0.0, -0.80, 0.50});           // (4) 0.48 m in front of the box
    // (5) on the back face, where the camera's line (through the front face at z 0.57) leaves the box
    const Eigen::Vector3d far(0.0, 0.32, 0.45);
    const double p5 = pb(far);
    std::printf("  P(behind): beside %.4f, straight behind %.4f, front face %.4f, in front %.4f, at the far face %.4f\n", p1, p2, p3, p4, p5);
    std::printf("  labels (q %.1f): beside %.4f (before: %.4f), front face %.4f\n", q, rc::depth_ordered_label(q, p1, eps),
                eps + (1.0 - 2.0 * eps) * q, rc::depth_ordered_label(q, p3, eps));
    CHECK(p1 > 0.99 and p2 > 0.99);
    CHECK(p3 < 0.01 and p4 < 0.01);
    CHECK(std::abs(p5 - 0.5) < 0.05);
    CHECK(rc::depth_ordered_label(q, p1, eps) < eps + 0.01);
    CHECK(std::abs(rc::depth_ordered_label(q, p3, eps) - (eps + (1.0 - 2.0 * eps) * q)) < 0.01);
    // (5) continuity across the side edge: a return 0.05 m behind the back-face plane, sliding in x from inside the box's width to
    // beside it (the line changes from crossing to missing at |x| ≈ 0.215)
    double worst = 0.0, prev = pb({0.0, 0.37, 0.45});
    for (int i = 1; i <= 400; ++i)
    {
        const double x = 0.4 * i / 400.0, v = pb({x, 0.37, 0.45});
        worst = std::max(worst, std::abs(v - prev)); prev = v;
    }
    std::printf("  sliding across the side edge (1 mm steps): largest step in P(behind) %.4f\n", worst);
    CHECK(worst < 0.05);
    std::printf(bad ? "depth_order_test: %d FAILED\n" : "depth_order_test: OK\n", bad);
    return bad != 0;
}
