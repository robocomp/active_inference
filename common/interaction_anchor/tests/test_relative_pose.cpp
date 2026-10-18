// Standalone harness (no framework) for interaction_anchor/relative_pose.h.
#include "interaction_anchor/relative_pose.h"
#include <cstdio>

using namespace rc::interaction;
static int fails = 0;
static void check(const char* name, bool ok) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); if (not ok) ++fails; }

static rc::rt::Twist twist(float side, float adv, float rot)
{
    rc::rt::Twist t; t.linear = {side, adv, 0.f}; t.angular = {0.f, 0.f, rot}; return t;
}

int main()
{
    const Eigen::Vector3f zero = Eigen::Vector3f::Zero();
    {   RelPose r; r.x = {0, 2, 0};
        const auto o = predict(r, twist(0, 0.5f, 0), zero, 1.f);
        check("pure_advance", (o.x - Eigen::Vector3f(0, 1.5f, 0)).norm() < 1e-4f); }
    {   RelPose r; r.x = {0, 2, 0};
        const auto o = predict(r, twist(0, 0, float(M_PI / 2)), zero, 1.f);
        check("pure_turn", (o.x - Eigen::Vector3f(2, 0, -float(M_PI / 2))).norm() < 1e-4f); }
    {   RelPose r; r.x = {0.3f, 2, 0.4f}; r.P = Eigen::Vector3f(0.01f, 0.02f, 0.03f).asDiagonal();
        const auto still = predict(r, twist(0, 0, 0), zero, 1.f);
        const auto mv    = twist(0.1f, 0.5f, 0.3f);
        const auto a     = predict(r, mv, zero, 1.f);
        const auto b     = predict(r, mv, Eigen::Vector3f(0.01f, 0.01f, 0.01f), 1.f);
        // J for rot*dt = 0.3: rotation block R(-0.3), yaw passes through
        const float c = std::cos(0.3f), s = std::sin(0.3f);
        Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
        J.topLeftCorner<2,2>() << c, s, -s, c;
        const Eigen::Matrix3f JPJt = J * r.P * J.transpose();
        check("covariance_grows_only_with_motion_variance",
              (still.P - r.P).norm() < 1e-6f and (a.P - JPJt).norm() < 1e-6f and b.P.trace() > JPJt.trace()); }
    {   Eigen::Affine3d rr = Eigen::Affine3d::Identity(), ro = Eigen::Affine3d::Identity();
        rr.translate(Eigen::Vector3d(1, 1, 0)); rr.rotate(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
        ro.translate(Eigen::Vector3d(2, 4, 0)); ro.rotate(Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()));
        const auto r = from_room(rr, ro, Eigen::Matrix3f::Identity());
        Eigen::Affine3d rel = Eigen::Affine3d::Identity();
        rel.translate(Eigen::Vector3d(r.x.x(), r.x.y(), 0)); rel.rotate(Eigen::AngleAxisd(r.x.z(), Eigen::Vector3d::UnitZ()));
        const Eigen::Affine3d back = rr * rel;
        check("from_room_round_trip", (back.matrix() - ro.matrix()).norm() < 1e-5); }
    std::puts(fails == 0 ? "ALL PASS" : "FAILED");
    return fails;
}
