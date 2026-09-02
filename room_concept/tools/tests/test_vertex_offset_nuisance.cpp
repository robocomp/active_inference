// Numerical check of the per-vertex offset marginalisation, on SYNTHETIC data whose truth is known.
// ★ Reproduces the live defect first: without the nuisance a per-corner offset must produce a tight,
//   wrong yaw; with it, the sigma must open to the between-vertex scale.
#include <Eigen/Dense>
#include <cstdio>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>
#define IMAGE_EDGE_NUISANCES 5
namespace rc { struct TriplePoint; struct CameraModel; }
#include "vb.h"     // the VertexBlock/Accum pair lifted verbatim
int main()
{
    std::mt19937 g(7);
    std::normal_distribution<double> eps(0.0, 1.0), off(0.0, 5.3);   // 5.3 px measured spread
    const int NV = 23, NPER = 17000;
    const double TRUE_YAW = 0.0;          // truth: NO yaw error at all
    rc::mount::Accum a_off, a_on;
    a_on.offset_sigma_px = 5.3;
    for (int v = 0; v < NV; ++v)
    {
        const double dv = off(g);          // this corner's own u-offset, constant for the corner
        for (int k = 0; k < NPER; ++k)
        {
            rc::mount::PairObs o; o.ok = true; o.vertex = v; o.assoc_prob = 1.f;
            o.cov = Eigen::Matrix2f::Identity() * 4.0f;              // 2 px measurement noise
            const double d = 3.0 + 3.0 * (k % 100) / 100.0;         // range 3..6 m
            o.J.setZero();
            o.J(0, 2) = 1.0f;                                        // yaw  -> constant u shift
            o.J(1, 0) = static_cast<float>(d * d / 1.4);             // pitch-> range dependent v
            o.J(1, 1) = static_cast<float>(1.0 / d);                 // height
            o.r = Eigen::Vector2f(static_cast<float>(-TRUE_YAW + dv + 2.0*eps(g)),
                                  static_cast<float>(2.0*eps(g)));
            a_off.add(o); a_on.add(o);
        }
    }
    const auto s0 = a_off.solve(), s1 = a_on.solve();
    printf("truth: yaw = 0.000 px, per-corner offsets ~ N(0, 5.3 px), %d corners x %d sightings\n\n",
           NV, NPER);
    printf("               yaw p      yaw sigma   pitch sigma  height sigma  chi2/dof  clusters  eff_par\n");
    printf("nuisance OFF %9.4f  %10.4f  %11.4f  %12.4f  %8.2f  %8d  %7.1f\n",
           s0.p(2), s0.sigma(2), s0.sigma(0), s0.sigma(1), s0.chi2_dof, s0.clusters, s0.eff_params);
    printf("nuisance ON  %9.4f  %10.4f  %11.4f  %12.4f  %8.2f  %8d  %7.1f\n",
           s1.p(2), s1.sigma(2), s1.sigma(0), s1.sigma(1), s1.chi2_dof, s1.clusters, s1.eff_params);
    printf("\nyaw sigma inflation   %.0fx      (design effect sqrt(%d/%d) = %.0fx)\n",
           s1.sigma(2)/s0.sigma(2), NV*NPER, NV, std::sqrt(double(NPER)));
    printf("pitch sigma inflation %.1fx     height %.1fx   <- must be SMALLER than yaw's\n",
           s1.sigma(0)/s0.sigma(0), s1.sigma(1)/s0.sigma(1));
    // S -> 0 must reproduce the old estimator bit for bit
    rc::mount::Accum a_zero = a_on; a_zero.offset_sigma_px = 0.0;
    const auto sz = a_zero.solve();
    printf("\nS -> 0 reproduces OFF exactly: yaw %s, sigma %s\n",
           sz.p(2) == s0.p(2) ? "YES" : "NO", sz.sigma(2) == s0.sigma(2) ? "YES" : "NO");
    return 0;
}
