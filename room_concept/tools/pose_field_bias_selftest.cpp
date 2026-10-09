// Offline validation of rc::PoseFieldBias (src/pose_field_bias.h) against a KNOWN bias field.
//
// The robot drives laps; the scan-only pose is z = x + b(x) + e, with b a smooth field over POSITION of
// standard deviation sigma_b and correlation length l, e white. The odometry is x's increments plus white
// noise whose variance the estimator is TOLD (the noise learner's job). The estimator sees only z, the odometry
// and that variance — never x — and must recover sigma_b; with no field it must report ~0.
//
// Build:  g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/pose_field_bias_selftest.cpp -o /tmp/pfb && /tmp/pfb
#include "../src/pose_field_bias.h"
#include <cmath>
#include <cstdio>
#include <random>

namespace
{
int failures = 0;
void check(bool ok, const char *what) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what); failures += not ok; }
std::mt19937 rng(7);
double N(double s) { return std::normal_distribution<double>(0.0, s)(rng); }

// A smooth random field with unit variance per axis: a sum of random-phase cosines (spectral method),
// wavenumbers ~ N(0, 1/l^2) => squared-exponential-like correlation with length ~ l.
struct Field
{
    std::vector<Eigen::Vector2d> k; std::vector<double> ph_x, ph_y;
    explicit Field(double l, int n = 200)
    {
        for (int i = 0; i < n; ++i)
        {
            k.emplace_back(N(1.0 / l), N(1.0 / l));
            ph_x.push_back(std::uniform_real_distribution<double>(0, 2 * M_PI)(rng));
            ph_y.push_back(std::uniform_real_distribution<double>(0, 2 * M_PI)(rng));
        }
    }
    Eigen::Vector2d at(const Eigen::Vector2d &p) const
    {
        Eigen::Vector2d v = Eigen::Vector2d::Zero();
        for (std::size_t i = 0; i < k.size(); ++i)
        { v.x() += std::cos(k[i].dot(p) + ph_x[i]); v.y() += std::cos(k[i].dot(p) + ph_y[i]); }
        return v * std::sqrt(2.0 / k.size());
    }
};

// Drive laps of a 6 x 4 m rectangle with rounded corners; return the estimator after `cycles`.
rc::PoseFieldBias drive(double sigma_b, double l, int cycles, double odo_k = 4e-4)
{
    rc::PoseFieldBias est;
    Field f(l);
    double s = 0.0;                                   // arclength along the loop
    const double per = 2 * (6.0 + 4.0);
    auto pos = [&](double u) -> Eigen::Vector2d {     // point on the rectangle at arclength u
        u = std::fmod(u, per);
        if (u < 6) return {-3 + u, -2};
        if (u < 10) return {3, -2 + (u - 6)};
        if (u < 16) return {3 - (u - 10), 2};
        return {-3, 2 - (u - 16)};
    };
    Eigen::Vector2d x = pos(0), z_prev;
    bool first = true;
    // claimed scan covariance: 10 mm, 0.156 deg (the live claim); true white 1 mm
    Eigen::Matrix3f R = Eigen::Vector3f(1e-4f, 1e-4f, 7.4e-6f).asDiagonal();
    for (int n = 0; n < cycles; ++n)
    {
        const double v = (n / 400) % 5 == 4 ? 0.0 : 0.02;   // park one stretch in five
        s += v;
        const Eigen::Vector2d xn = pos(s);
        const Eigen::Vector2d step = xn - x;
        const double var = odo_k * step.norm();       // the odometry's per-cycle variance per axis, as told
        const Eigen::Vector2d odom = step + Eigen::Vector2d(N(std::sqrt(var)), N(std::sqrt(var)));
        x = xn;
        const Eigen::Vector2d z = x + sigma_b * f.at(x) + Eigen::Vector2d(N(0.001), N(0.001));
        est.observe(Eigen::Vector3f(float(z.x()), float(z.y()), 0.f), R,
                    Eigen::Vector3f(float(odom.x()), float(odom.y()), 0.f),
                    float(2.0 * var), 0.f, /*chain_ok=*/not first);
        first = false;
        z_prev = z;
    }
    return est;
}
}   // namespace

int main()
{
    std::printf("1. bias field sigma_b = 40 mm, l = 2 m, odometry 4e-4 m^2/m (the live forward level)\n");
    const auto A = drive(0.040, 2.0, 120000);
    std::printf("    sigma_b %.1f mm  l %.2f m  pairs %ld\n", A.sigma_b() * 1e3, A.length(), A.pairs());
    check(std::abs(A.sigma_b() - 0.040) < 0.010, "sigma_b recovered within 10 mm (25 %)");
    check(A.length() > 0.5 and A.length() < 8.0, "correlation length in the right octave band");

    std::printf("\n2. no field: the estimator must not invent one\n");
    const auto B = drive(0.0, 2.0, 120000);
    std::printf("    sigma_b %.1f mm\n", B.sigma_b() * 1e3);
    check(B.sigma_b() < 0.010, "sigma_b < 10 mm with no field");

    std::printf("\n3. the published term has the SCAN'S geometry: Sigma_b = beta * R, beta = sigma_b^2 / (tr R_xy / 2)\n");
    const Eigen::Matrix3f R = Eigen::Vector3f(1e-4f, 1e-4f, 7.4e-6f).asDiagonal();
    const Eigen::Matrix3f S = A.covariance(R);
    std::printf("    Sigma_b: xy %.1f mm, heading %.3f deg\n", std::sqrt(S(0, 0)) * 1e3, std::sqrt(S(2, 2)) * 180 / M_PI);
    check(std::abs(std::sqrt(S(0, 0)) - A.sigma_b()) < 1e-6, "xy diagonal == sigma_b^2");
    check(std::abs(S(2, 2) / S(0, 0) - 7.4e-6 / 1e-4) < 1e-6, "heading follows the scan's own lever (R_tt / R_xx)");

    std::printf("\n4. parked only: no path, no pairs, the prior stands\n");
    {
        rc::PoseFieldBias P;
        const Eigen::Matrix3f R0 = Eigen::Vector3f(1e-4f, 1e-4f, 7.4e-6f).asDiagonal();
        for (int n = 0; n < 5000; ++n)
            P.observe(Eigen::Vector3f(float(N(0.001)), float(N(0.001)), 0.f), R0, Eigen::Vector3f::Zero(), 0.f, 0.f, n > 0);
        std::printf("    sigma_b %.1f mm (prior %.1f mm), pairs %ld\n", P.sigma_b() * 1e3, P.params().sigma0 * 1e3, P.pairs());
        check(P.pairs() == 0 and std::abs(P.sigma_b() - P.params().sigma0) < 1e-6, "parked: the prior, untouched");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
