// Task S: the ltsm seam is fused from the robot chain and the crossed aperture, by covariance.
//   g++ -std=c++23 -I/usr/include/eigen3 -I../src seam_fusion_test.cpp -o seam_fusion_test && ./seam_fusion_test
#include "seam_fusion.h"
#include <cstdio>
using namespace ltsm::seam;
static int failures = 0;
static void check(bool ok, const char *what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

// A door node whose proto frame composes to the seam S: door pose in old = inverse(S), robot a metre into the new room.
static Measured through_door(const Vec3 &S, double sigma_xy, double sigma_yaw)
{
    const Vec3 P = inverse(S);
    const Eigen::Vector2d robot = P.head<2>() + rot(P.z()) * Eigen::Vector2d(0.3, 1.0);
    Mat3 C = Mat3::Zero();
    C.diagonal() << sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_yaw * sigma_yaw;
    return aperture_seam(P, C, robot);
}

int main()
{
    std::printf("aperture seam\n");
    {
        const Vec3 S(0.03, -4.32, -0.084);
        const auto m = through_door(S, 0.03, 0.01);
        check((m.pose - S).norm() < 1e-9, "T(new<-old) = inverse of the aperture pose");
        // door frame turned by pi (y looks away from the robot) must give the same proto
        const Vec3 P = inverse(S);
        Vec3 Pflip(P.x(), P.y(), wrap(P.z() + M_PI));
        const Eigen::Vector2d robot = P.head<2>() + rot(P.z()) * Eigen::Vector2d(0.3, 1.0);
        const auto m2 = aperture_seam(Pflip, Mat3::Identity() * 1e-4, robot);
        check((m2.pose - S).norm() < 1e-9, "door x/y order irrelevant: y is chosen toward the robot");
        check(m.cov(0, 0) > 0 and m.cov(2, 2) > 0, "covariance propagated");
    }
    std::printf("gold1 garbage chain\n");
    {
        const Vec3 S(0.03, -4.32, -0.084);
        const auto ap = through_door(S, 0.03, 0.01);
        Measured chain; chain.pose = Vec3(-14262.68, 1667.96, -1.1285);
        chain.cov = Mat3::Zero(); chain.cov.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;   // overconfident
        const auto f = fuse(chain, ap);
        check((f.pose - S).head<2>().norm() < 0.05 and std::abs(wrap(f.pose.z() - S.z())) < 0.02, "stored seam = the aperture seam");
        check(f.chain_weight < 1e-6, "contradicting chain gets ~0 weight");
    }
    std::printf("sane chains (gold1 seams 1 and 2)\n");
    for (const Vec3 &S : {Vec3(0.03, -4.32, -0.084), Vec3(-0.45, 4.36, 0.0008)})
    {
        const auto ap = through_door(S, 0.03, 0.01);
        Measured chain; chain.pose = S + Vec3(0.02, -0.02, 0.005);
        chain.cov = Mat3::Zero(); chain.cov.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;
        const auto f = fuse(chain, ap);
        const Vec3 d = f.pose - S;
        check(d.head<2>().norm() < 0.05 and std::abs(d.z()) < 0.02, "unchanged within sigma");
        check(f.chain_weight > 0.2, "consistent chain keeps its information");
        check(f.cov(0, 0) < ap.cov(0, 0), "fused is tighter than the door alone");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
