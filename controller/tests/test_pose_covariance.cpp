// Task 7 (local-motion policy): the controller's pose uncertainty carries the FULL SE(2) covariance (x, y, yaw) read out
// of cortex's 6x6 SE(3) block [x, y, z, rx, ry, rz], with xy_std_m / theta_std_rad derived from it as before.
// Build+run (from controller/):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 tests/test_pose_covariance.cpp -o /tmp/tpc && /tmp/tpc
#include "../src/controller_runtime_types.h"

#include <cmath>
#include <print>
#include <string_view>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
bool near(double a, double b) { return std::abs(a - b) < 1e-6; }
}  // namespace

int main()
{
    // A 6x6 with distinct entries everywhere: the SE(2) block must pick rows/cols {0, 1, 5} (x, y, yaw) — NOT z (2).
    Eigen::Matrix<double, 6, 6> c6;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) c6(r, c) = 0.001 * (r + 1) * (c + 1) + (r == c ? 0.01 * (r + 1) : 0.0);
    const auto u = pose_uncertainty_from_se3(c6);
    const int idx[3] = {0, 1, 5};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) CHECK(near(u.cov(r, c), c6(idx[r], idx[c])));
    CHECK(near(u.cov(2, 2), c6(5, 5)));                                        // yaw, not z
    CHECK(near(u.xy_std_m, std::sqrt(std::max(c6(0, 0), c6(1, 1)))));          // derived exactly as before
    CHECK(near(u.theta_std_rad, std::sqrt(c6(5, 5))));
    // A negative diagonal (a malformed block) never yields NaN stds.
    Eigen::Matrix<double, 6, 6> bad = Eigen::Matrix<double, 6, 6>::Zero();
    bad(0, 0) = -1.0; bad(5, 5) = -1.0;
    const auto ub = pose_uncertainty_from_se3(bad);
    CHECK(ub.xy_std_m == 0.f and ub.theta_std_rad == 0.f);
    // Default-constructed: zero covariance (no claim), stds zero.
    const ControllerPoseUncertainty z;
    CHECK(z.cov.isZero());
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
