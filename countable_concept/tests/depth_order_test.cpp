// depth_order_test — the depth ordering of a LiDAR return against the object along the camera's line of sight (F1 round 7 B,
// src/countable_depth_order.h). Toilet geometry of P1 p1_toilet2: a 0.43 × 0.64 × 0.92 m box (w along x, h along y, centred at
// the origin, back face y = +0.32) whose back is 0.02 m off a wall, the camera 2.6 m in front at 0.98 m. Before this change a
// return's label was the mask's q whatever its depth (a wall return beside the narrow bowl, inside the blurred mask edge,
// carried m ≈ 0.6). Every bound below is DERIVED from the geometry (fix round 1, I2), with σ_min = σ_m ≤ σ ≤ σ_max =
// √(σ_m² + max face variance) (the blended face variance never exceeds its largest term):
//   (1) a wall return BESIDE the box: for s ≥ s(y = −0.32) the line is beside the box (|x| > ½w) and the SDF only grows, so the
//       exit/closest approach t_e ≤ r·s(y = −0.32) ⇒ P ≥ Φ(r(1 − s)/σ_max); its label ≤ ε + (1 − 2ε)q(1 − that);
//   (2) straight behind: the line leaves by the back face, t_e = r·s_back ⇒ P ≥ Φ(r(1 − s_back)/σ_max);
//   (3) on the visible front face and (4) in front of the box: the line, extended, leaves by the back face beyond p, t_e =
//       r·s_back with s_back > 1 ⇒ P ≤ Φ(r(1 − s_back)/σ_max); the front label ≥ ε + (1 − 2ε)q(1 − that);
//   (5) ON the back face P = Φ(0) = ½ up to t_e's numerical error (bisection to 2⁻⁸⁰ of a step, and double rounding of r):
//       |P − ½| ≤ φ(0)·1e-12/σ_min;
//   (6) continuity at the vertical edge where the exit switches from the back face to the side face (x₀ = ½w/s_back for a
//       return at y = 0.37): by the mean value theorem |P(x₀ + δ) − P(x₀ − δ)| ≤ 2δ[φ(0)L_d + φ(1)L_σ]/σ_min, with L_d =
//       |∂(r − t_e)/∂x| ≤ 2 + r·(½w)/x² (side exit t_e = r·½w/x; back exit t_e = r·s_back) and L_σ ≤ 3·Δv·e^{−½}/σ_m · (L_d + 1)/
//       (2σ_min) (σ² = σ_m² + V, V the face blend: |∂V/∂e_k| ≤ Δv·e^{−½}/σ_m when a face holds the exit point, |∂e/∂x| ≤ the exit
//       point's speed ≤ L_d + 1). The previous hard switch of the face normal stepped σ there.
// Run: ./depth_order_test
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "countable_depth_order.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
static double phi(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); }

int main()
{
    std::printf("depth_order_test\n");
    const double w = 0.43, h = 0.64, H = 0.92;
    const rc::LosBox box{0.0, 0.0, H, w, h, 0.0, 0.017 * 0.017, 0.022 * 0.022, 0.015 * 0.015, 0.044 * 0.044, 0.024 * 0.024};
    const Eigen::Vector3d C(0.0, -2.6, 0.98);
    const double sm = std::sqrt(0.03 * 0.03 + 0.01 * 0.01), eps = 0.001, q = 0.6;
    const double vx = box.var_cx + 0.25 * box.var_w, vy = box.var_cy + 0.25 * box.var_h, vz = box.var_H;
    const double vmax = std::max({vx, vy, vz}), vmin = 0.0;                 // the bottom face's variance is 0
    const double s_max = std::sqrt(sm * sm + vmax), s_min = sm;
    const auto P = [&](const Eigen::Vector3d& p) { return rc::behind_object_probability(C, p, box, sm); };
    const auto s_at_y = [&](const Eigen::Vector3d& p, double y) { return (y - C.y()) / (p.y() - C.y()); };   // line parameter
    const auto label = [&](double pb) { return rc::depth_ordered_label(q, pb, eps); };

    const Eigen::Vector3d p1(-0.30, 0.34, 0.45), p2(0.0, 0.50, 0.45), p3(0.0, -0.32, 0.50), p4(0.0, -0.80, 0.50), p5(0.0, 0.32, 0.45);
    const double r1 = (p1 - C).norm(), r2 = (p2 - C).norm(), r3 = (p3 - C).norm(), r4 = (p4 - C).norm();
    const double b1 = Phi(r1 * (1.0 - s_at_y(p1, -0.5 * h)) / s_max);    // lower bound
    const double b2 = Phi(r2 * (1.0 - s_at_y(p2, 0.5 * h)) / s_max);     // lower bound
    const double b3 = Phi(r3 * (1.0 - s_at_y(p3, 0.5 * h)) / s_max);     // upper bound
    const double b4 = Phi(r4 * (1.0 - s_at_y(p4, 0.5 * h)) / s_max);     // upper bound
    const double P1 = P(p1), P2 = P(p2), P3 = P(p3), P4 = P(p4), P5 = P(p5);
    std::printf("  (1) beside      P %.6f >= %.6f   label %.5f (before: q-label %.4f) <= %.5f\n", P1, b1, label(P1),
                eps + (1.0 - 2.0 * eps) * q, eps + (1.0 - 2.0 * eps) * q * (1.0 - b1));
    std::printf("  (2) behind      P %.6f >= %.6f\n", P2, b2);
    std::printf("  (3) front face  P %.3e <= %.3e   label %.5f >= %.5f\n", P3, b3, label(P3), eps + (1.0 - 2.0 * eps) * q * (1.0 - b3));
    std::printf("  (4) in front    P %.3e <= %.3e\n", P4, b4);
    std::printf("  (5) back face   |P - 1/2| %.3e <= %.3e\n", std::abs(P5 - 0.5), phi(0.0) * 1e-12 / s_min);
    CHECK(P1 >= b1 and P2 >= b2);
    CHECK(P3 <= b3 and P4 <= b4);
    CHECK(label(P1) <= eps + (1.0 - 2.0 * eps) * q * (1.0 - b1));
    CHECK(label(P3) >= eps + (1.0 - 2.0 * eps) * q * (1.0 - b3));
    CHECK(std::abs(P5 - 0.5) <= phi(0.0) * 1e-12 / s_min);

    // (6) continuity at the back/side edge
    const double y = 0.37, z = 0.45, s_back = (0.5 * h - C.y()) / (y - C.y()), x0 = 0.5 * w / s_back, delta = 1e-7;
    const Eigen::Vector3d pa(x0 - delta, y, z), pb(x0 + delta, y, z);
    const double r0 = (pa - C).norm(), Ld = 2.0 + r0 * 0.5 * w / ((x0 - delta) * (x0 - delta));
    const double Ls = 3.0 * (vmax - vmin) * std::exp(-0.5) / sm * (Ld + 1.0) / (2.0 * s_min);
    const double step = std::abs(P(pb) - P(pa)), bound = 2.0 * delta * (phi(0.0) * Ld + phi(1.0) * Ls) / s_min;
    std::printf("  (6) at the back/side edge x0 %.5f: |P(x0+d) - P(x0-d)| %.3e <= %.3e (d %.0e m, L_d %.1f, L_sigma %.4f)\n",
                x0, step, bound, delta, Ld, Ls);
    CHECK(step <= bound);
    std::printf(bad ? "depth_order_test: %d FAILED\n" : "depth_order_test: OK\n", bad);
    return bad != 0;
}
