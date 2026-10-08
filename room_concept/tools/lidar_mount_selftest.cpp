// Offline check of rc::lidar_mount (plan 2026-10-05 Task 5; 6-DoF part: plan 2026-10-08 Task 4): the planar correction / injection applied
// to helios points after the DSR device->body transform.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/lidar_mount_selftest.cpp -o /tmp/lidar_mount_selftest && /tmp/lidar_mount_selftest
#include "../src/lidar_mount.h"
#include <cmath>
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
    // ── part 2: the 6-DoF mount (plan 2026-10-08 Task 4): rotation about the SENSOR origin, body frame ──
    {
        const Eigen::Vector3f s(0.f, -0.155f, 1.1075f);                       // helios origin, Shadow frame
        const Pose6 M{Eigen::Vector3f(0.03f, -0.02f, 0.01f), Eigen::Vector3f(0.0122f, -0.0070f, 0.0175f)};
        const Pose6 I = compose(M, inverse(M)), I2 = compose(inverse(M), M);
        check(I.t.norm() < 1e-6f and I.rpy.norm() < 1e-6f and I2.t.norm() < 1e-6f and I2.rpy.norm() < 1e-6f,
              "6-DoF: compose(M, inverse(M)) == identity (both orders)");
        // a cloud through the NOMINAL mount reads M^-1(p_true); the injection plants exactly that and a correct
        // estimate restores it
        bool ok = true;
        for (const Eigen::Vector3f &pt : {Eigen::Vector3f(2.f, 1.f, 1.7f), Eigen::Vector3f(-3.f, 0.5f, 0.f),
                                         Eigen::Vector3f(0.5f, -2.f, 2.6f), Eigen::Vector3f(0.f, 0.f, 0.f)})
        {
            const Eigen::Vector3f nom = apply(inverse(M), s, pt);             // T_inject = inverse(M_inj)
            ok = ok and (apply(M, s, nom) - pt).norm() < 1e-5f;               // T_corr = M_est
            ok = ok and (apply(compose(M, inverse(M)), s, pt) - pt).norm() < 1e-5f;
        }
        check(ok, "6-DoF: a correct estimate cancels a planted M on a point cloud");
        check((apply(M, s, s) - (s + M.t)).norm() < 1e-6f, "6-DoF: rotation is about the SENSOR origin (a point there moves only by t)");
        const Pose6 Rz{Eigen::Vector3f::Zero(), Eigen::Vector3f(0.f, 0.f, 0.5f)};
        const Eigen::Vector3f q = apply(Rz, s, s + Eigen::Vector3f(1.f, 0.f, 0.f));
        check((q - (s + Eigen::Vector3f(std::cos(0.5f), std::sin(0.5f), 0.f))).norm() < 1e-6f, "6-DoF: +yaw is CCW about +z");
        const Pose6 Rx{Eigen::Vector3f::Zero(), Eigen::Vector3f(0.5f, 0.f, 0.f)};
        const Eigen::Vector3f qx = apply(Rx, s, s + Eigen::Vector3f(0.f, 1.f, 0.f));
        check((qx - (s + Eigen::Vector3f(0.f, std::cos(0.5f), std::sin(0.5f)))).norm() < 1e-6f, "6-DoF: +roll is CCW about +x");
        // small angles: rpy == the rotation vector the estimators use (p' = p + omega x q + t)
        const Pose6 small{Eigen::Vector3f(0.01f, 0.02f, -0.01f), Eigen::Vector3f(0.004f, -0.006f, 0.008f)};
        const Eigen::Vector3f p0(2.f, -1.f, 1.5f), qq = p0 - s;
        const Eigen::Vector3f lin = p0 + small.rpy.cross(qq) + small.t;
        check((apply(small, s, p0) - lin).norm() < 2e-4f, "6-DoF: first order == omega x (p - s) + t (the estimators' rows)");
        check(is_identity(Pose6{}) and not is_identity(M), "6-DoF: identity is detected exactly");
        // the INVERTED-sensor write-back flip (header note): R_nom = Ry(pi) -> sensor-frame increment (-wx, wy, -wz)
        const Eigen::Vector3f w(0.01f, 0.02f, 0.03f);
        const Eigen::Vector3f ws = sensor_frame_increment_inverted(w);
        check((ws - Eigen::Vector3f(-0.01f, 0.02f, -0.03f)).norm() < 1e-7f, "write-back: inverted sensor flips roll and yaw, not pitch");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
