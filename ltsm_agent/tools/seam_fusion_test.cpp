// Task S: the ltsm seam is fused from the robot chain and the crossed aperture (inlier/outlier mixture + information).
//   g++ -std=c++23 -I/usr/include/eigen3 -I../src seam_fusion_test.cpp -o seam_fusion_test && ./seam_fusion_test
#include "seam_fusion.h"
#include "../../room_concept/src/door_crossing.h"   // the REAL proto frame room_concept builds
#include <cstdio>
#include <random>
using namespace ltsm::seam;
static int failures = 0;
static void check(bool ok, const char *what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

static Mat3 diag3(double sxy, double syaw) { Mat3 C = Mat3::Zero(); C.diagonal() << sxy*sxy, sxy*sxy, syaw*syaw; return C; }

// A room rotated by `rot_room` in the old frame, with a door in its wall, built the way door_concept does: the DOOR NODE's
// x axis is the wall tangent (door_apertures.cpp:137), the aperture is c -/+ x*w/2. room_concept's crossing::evaluate +
// proto_frame turn that into the proto frame the new room is born with; the seam must be the inverse of THAT.
// (a/b swapped = door_concept's yaw pointing the other way along the wall.)
static void door_case(double room_yaw, bool swap_ab, bool door_x_is_ccw, const char *what)
{
    const Eigen::Vector2d c0(1.7, -2.2);                   // door centre in the old room
    // wall direction: the room's +x rotated; room interior is on the wall's left or right depending on door_x_is_ccw
    const double wall_yaw = room_yaw + (door_x_is_ccw ? 0.0 : M_PI);
    const Eigen::Vector2d tx(std::cos(wall_yaw), std::sin(wall_yaw)), ty(-std::sin(wall_yaw), std::cos(wall_yaw));
    const double w = 0.9;
    // old room: a big rectangle whose wall passes through c0 along tx; interior = -ty side when door_x_is_ccw (y of door looks OUT)
    const Eigen::Vector2d inward = -ty;                    // door's +y looks out of the room, toward the new room
    std::vector<Eigen::Vector2f> poly;
    for (auto [u, v] : std::vector<std::pair<double, double>>{{-3, 0}, {3, 0}, {3, -4}, {-3, -4}})
    { const Eigen::Vector2d q = c0 + u * tx + v * ty; poly.emplace_back(float(q.x()), float(q.y())); }
    if (std::vector<Eigen::Vector2f>(poly).size() == 4)
    { float a2 = 0; for (int i = 0; i < 4; ++i) a2 += poly[i].x()*poly[(i+1)%4].y() - poly[(i+1)%4].x()*poly[i].y(); if (a2 < 0) std::ranges::reverse(poly); }
    rc::DoorAperture ap;
    ap.a = (c0 - tx * w / 2).cast<float>(); ap.b = (c0 + tx * w / 2).cast<float>();
    if (swap_ab) std::swap(ap.a, ap.b);
    ap.p_open = 1.f;
    const Eigen::Vector2d robot = c0 + 0.6 * ty + 0.2 * tx;   // 0.6 m past the door in the new room
    const auto cr = rc::crossing::evaluate(ap, poly, robot.cast<float>(), Eigen::Matrix2f::Identity() * 1e-4f, 0.25f);
    std::printf("%s\n", what);
    check(cr.valid, "crossing valid");
    const auto T = rc::crossing::proto_frame(cr);
    const Vec3 proto_truth(T.translation().x(), T.translation().y(), std::atan2(T.linear()(1, 0), T.linear()(0, 0)));
    const Vec3 truth_seam = inverse(proto_truth);
    // the DOOR NODE pose in the old room: x = wall tangent, centred
    const Vec3 door_pose(c0.x(), c0.y(), wall_yaw);
    const auto m = aperture_seam(door_pose, diag3(0.03, 0.01), robot);
    check((m.pose - truth_seam).head<2>().norm() < 1e-4 and std::abs(wrap(m.pose.z() - truth_seam.z())) < 1e-4,
          "seam = inverse of room_concept's proto frame");
    check((m.pose.head<2>() - Eigen::Vector2d(0, 0)).norm() > 0, "non-trivial");
    // robot lands where it truly is relative to the aperture: seam applied to the robot's old-frame position
    const Eigen::Vector2d in_new = rot(m.pose.z()) * robot + m.pose.head<2>();
    check(std::abs(in_new.y() - 0.6) < 1e-4 and std::abs(std::abs(in_new.x()) - 0.2) < 1e-4,
          "robot in the new frame = its pose relative to the aperture");
}

int main()
{
    for (double yaw : {0.0, 0.7, -2.4})
        for (bool sw : {false, true})
            for (bool ccw : {true, false})
            {
                char buf[96]; std::snprintf(buf, sizeof buf, "door: room yaw %+.1f, a/b %s, door x %s", yaw, sw ? "swapped" : "normal", ccw ? "ccw" : "cw");
                door_case(yaw, sw, ccw, buf);
            }

    const double R = 4.0;
    auto through_door = [](const Vec3 &S, double sxy, double syaw) {
        const Vec3 P = inverse(S);
        const Eigen::Vector2d robot = P.head<2>() + rot(P.z()) * Eigen::Vector2d(0.3, 1.0);
        return aperture_seam(P, diag3(sxy, syaw), robot);
    };
    std::printf("gold1 garbage chain\n");
    {
        const Vec3 S(0.03, -4.32, -0.084);
        const auto ap = through_door(S, 0.03, 0.01);
        Measured chain{Vec3(-14262.68, 1667.96, -1.1285), diag3(0.05, 0.02)};
        const auto f = fuse(chain, ap, R);
        check((f.pose - S).head<2>().norm() < 0.05 and std::abs(wrap(f.pose.z() - S.z())) < 0.02, "stored seam = the aperture seam");
        check(f.p_inlier < 1e-9 and f.log_p_wrong > -1e-6, "chain judged garbage (P(inlier) ~ 0, log P(wrong) ~ 0)");
    }
    std::printf("sane chains (gold1 seams 1 and 2)\n");
    for (const Vec3 &S : {Vec3(0.03, -4.32, -0.084), Vec3(-0.45, 4.36, 0.0008)})
    {
        const auto ap = through_door(S, 0.03, 0.01);
        Measured chain{S + Vec3(0.02, -0.02, 0.005), diag3(0.05, 0.02)};
        const auto f = fuse(chain, ap, R);
        const Vec3 d = f.pose - S;
        check(d.head<2>().norm() < 0.05 and std::abs(d.z()) < 0.02, "unchanged within sigma");
        check(f.p_inlier > 0.99, "consistent chain keeps ~all its information");
        check(f.cov(0, 0) < ap.cov(0, 0), "fused is tighter than the door alone");
    }
    std::printf("gold2 live numbers (door node vs mouth: 0.495 m along the normal)\n");
    {
        const Vec3 ap_seam(0.7051, -4.6052, 0.0138);               // door-node aperture seam as stored first
        const Vec3 chain_pose(0.638, -4.110, 0.011);               // the robot chain that was thrown away
        const Vec3 P = inverse(ap_seam);
        const Eigen::Vector2d robot = P.head<2>() + rot(P.z()) * Eigen::Vector2d(0.3, 1.0);
        const auto ap = aperture_seam(P, diag3(0.03, 0.01), robot, 0.25, 0.25);
        Measured chain{chain_pose, diag3(0.05, 0.02)};
        const auto f = fuse(chain, ap, 6.0);
        check(f.p_inlier > 0.5, "the chain is an inlier once the recess nuisance is modelled");
        check((f.pose - chain_pose).head<2>().norm() < 0.12, "fused seam stays with the chain along the normal (|diff| < 0.12 m)");
        Measured garbage{Vec3(-14262.68, 1667.96, -1.1285), diag3(0.05, 0.02)};
        check(fuse(garbage, ap, 6.0).p_inlier < 1e-9, "gold1 garbage still rejected");
    }
    std::printf("chain covariance (Monte Carlo vs Jacobian)\n");
    {
        const Measured en{Vec3(-1.0, 0.3, 0.6), diag3(0.04, 0.02)}, eo{Vec3(-2.0, -0.4, 0.0), diag3(0.03, 0.015)};
        const auto lin = chain_seam(en, eo);
        std::mt19937 g(1); std::normal_distribution<double> N(0, 1);
        Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, 40000);
        for (int i = 0; i < 40000; ++i)
        {
            Vec3 a = en.pose + Vec3(0.04 * N(g), 0.04 * N(g), 0.02 * N(g)), b = eo.pose + Vec3(0.03 * N(g), 0.03 * N(g), 0.015 * N(g));
            const Measured s = [&] { Measured A{a, Mat3::Zero()}, B{b, Mat3::Zero()}; return chain_seam(A, B); }();
            X.col(i) = s.pose - lin.pose; X(2, i) = wrap(X(2, i));
        }
        const Mat3 emp = X * X.transpose() / 40000.0;
        check(((emp - lin.cov).cwiseAbs().array() <= 0.1 * lin.cov.diagonal().maxCoeff()).all(), "J C J^T matches the sampled covariance");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
