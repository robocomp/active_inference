// g++ -std=c++23 -O2 -Isrc tools/motion_cov_scale_selftest.cpp -o /tmp/x
// A 1-D localiser: truth random-walks with variance kappa_true*Q per stretch, the prior the localiser USES
// is kappa_hat*Q (closed loop, kappa fed back), a measurement of variance R corrects it. The learner sees the
// UNSCALED Q, the previous posterior, the correction and the new posterior, and must find kappa_true.
#include "motion_cov_scale.h"
#include <cstdio>
#include <random>
int main()
{
    int fails = 0;
    for (double kt : {0.2, 1.0, 3.0})
        for (double R : {1e-6, 1e-4})
        {
            rc::preint::MotionCovScale L;
            std::mt19937 rng(11); std::normal_distribution<double> n01;
            double P = 1e-4, err = 0.0;          // posterior variance and the estimate's true error
            const double Q = 4e-4;
            for (int k = 0; k < 6000; ++k)
            {
                const double kh = L.kappa(0);
                err += std::sqrt(kt * Q) * n01(rng);                       // truth moves; prediction does not
                const double Pp = P + kh * Q;                              // what the localiser believes
                const double z  = err + std::sqrt(R) * n01(rng);           // measurement of the error
                const double K  = Pp / (Pp + R);
                const double c  = K * z;                                   // the correction applied
                const double Pq = (1 - K) * Pp;
                L.observe(0, c, P, Q, Pq);
                err -= c; P = Pq;
            }
            const double kh = L.kappa(0);
            const bool ok = std::abs(kh / kt - 1.0) < 0.15;
            std::printf("kappa_true %.2f  R %.0e  ->  learnt %.3f  %s\n", kt, R, kh, ok ? "ok" : "FAIL");
            fails += not ok;
        }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
