// g++ -std=c++23 -O2 -I/usr/include/eigen3 -I../src tools/surprise_selftest.cpp -o /tmp/x
// Checks: KL = 0 for identical Gaussians; the calibrated-model identity E[mismatch] = expected, by
// simulation (a prior, a measurement, the Kalman posterior; c = posterior mean - prior mean); and that a
// biased prediction drives sum(mismatch)/sum(expected) above 1.
#include "surprise.h"
#include <cstdio>
#include <random>
int main()
{
    using namespace rc::surprise;
    int fails = 0;
    const Eigen::Matrix3f P = Eigen::Vector3f(1e-4f, 4e-4f, 1e-5f).asDiagonal();
    const auto z = kl_gauss(Eigen::Vector3f::Zero(), P, P);
    std::printf("identical: kl=%.2e scored=%d\n", z.kl, z.scored);
    fails += not (z.scored and std::abs(z.kl) < 1e-5f);
    std::mt19937 rng(7);
    std::normal_distribution<double> n01;
    for (double bias : {0.0, 0.05})
    {
        double sm = 0, se = 0;
        for (int k = 0; k < 20000; ++k)
        {
            Eigen::Matrix3d Pp; Pp << 4e-4, 1e-4, 0, 1e-4, 9e-4, 0, 0, 0, 2e-4;
            const Eigen::Matrix3d R = Eigen::Vector3d(1e-4, 1e-4, 1e-4).asDiagonal();
            const Eigen::Matrix3d Lp = Pp.llt().matrixL(), Lr = R.llt().matrixL();
            Eigen::Vector3d x, w; for (int i = 0; i < 3; ++i) { x[i] = n01(rng); w[i] = n01(rng); }
            const Eigen::Vector3d truth = Lp * x + Eigen::Vector3d(bias, 0, 0);   // pred = 0
            const Eigen::Vector3d meas  = truth + Lr * w;
            const Eigen::Matrix3d K = Pp * (Pp + R).inverse();
            const Eigen::Vector3d est = K * meas;
            const Eigen::Matrix3d Pq = (Eigen::Matrix3d::Identity() - K) * Pp;
            const auto s = kl_gauss(est.cast<float>(), Pp.cast<float>(), Pq.cast<float>());
            sm += s.mismatch; se += s.expected;
        }
        std::printf("bias %.2f m: sum(mismatch)/sum(expected) = %.3f\n", bias, sm / se);
        fails += bias == 0.0 ? not (std::abs(sm / se - 1.0) < 0.05) : not (sm / se > 1.5);
    }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
