// Offline check of rc::lidar_mount (plan 2026-10-05 Task 5): the planar correction / injection applied
// to helios points after the DSR device->body transform.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/lidar_mount_selftest.cpp -o /tmp/lidar_mount_selftest && /tmp/lidar_mount_selftest
#include "../src/lidar_mount.h"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc::lidar_mount;
int main()
{
    const Planar inj{0.03f, -0.02f, 0.0087f};               // 3 cm, -2 cm, 0.5 deg
    const Eigen::Vector3f p(2.f, 1.f, 1.7f);
    const Eigen::Vector3f q = apply(compose(inverse(inj), inj), p);
    check((q - p).norm() < 1e-6f, "inverse(inj) o inj == identity");
    const Eigen::Vector3f r = apply(inj, p);
    check(std::abs(r.z() - p.z()) < 1e-9f, "z untouched");
    check(std::abs((r.head<2>() - Eigen::Vector2f(0.03f, -0.02f)).norm() - p.head<2>().norm()) < 1e-5f,
          "rigid: distance from the translated origin preserved");
    // The plan's convention: T_inject = inverse(M_inj), T_corr = M_est; a correct estimate cancels exactly.
    const Planar T_inject = inverse(inj), T_corr = inj;
    check((apply(compose(T_corr, T_inject), p) - p).norm() < 1e-6f, "correct estimate: T_corr o T_inject == identity");
    const Planar ab = compose(Planar{1.f, 0.f, 1.5707963f}, Planar{1.f, 0.f, 0.f});
    check(std::abs(ab.x - 1.f) < 1e-5f and std::abs(ab.y - 1.f) < 1e-5f, "compose applies b first, then a");
    check(is_identity(Planar{}) and not is_identity(inj), "identity is detected exactly");
    // A sensor displaced by M reads a fixed world point p_b at M^-1(p_b): the nominal transform. A
    // YAW-ONLY mount error psi rotates the readings by -psi (review 1b: bearings read -psi_h).
    {
        const Planar M{0.f, 0.f, 0.01f};
        const Eigen::Vector3f pb(3.f, 0.f, 1.f);
        const Eigen::Vector3f nom = apply(inverse(M), pb);
        check(std::abs(std::atan2(nom.y(), nom.x()) + 0.01f) < 1e-6f, "a +psi mount yaw reads bearings at -psi");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
