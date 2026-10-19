// Standalone harness (no framework) for interaction_anchor/relative_pose.h.
#include "interaction_anchor/relative_pose.h"
#include <cmath>
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
        Eigen::Matrix3f Jc, Gc;
        predict_jacobians(r.x, 0.1f, 0.5f, 0.3f, Jc, Gc);
        const float sr = kRotScaleSigma * 0.3f;   // Task 12: the turn's scale variance, through the rot column of G
        const Eigen::Matrix3f JPJt = J * r.P * J.transpose() + (sr * sr) * (Gc.col(2) * Gc.col(2).transpose());
        check("covariance_grows_only_with_motion_variance",
              (still.P - r.P).norm() < 1e-6f and (a.P - JPJt).norm() < 1e-6f and b.P.trace() > JPJt.trace()); }
    // Task 12 (try73): the rotation-scale term. One 5-rad turn in one step: σ_yaw ≥ k·5·0.8 (0.36 rad) from 0.02; the
    // anchor 2 m ahead gains lateral/depth σ through the lever arm; a pure translation is unchanged.
    {   RelPose r; r.x = {0, 2, 0}; r.P = Eigen::Vector3f(4e-4f, 4e-4f, 4e-4f).asDiagonal();
        const auto spin = predict(r, twist(0, 0, 5.f), zero, 1.f);
        const float sy = std::sqrt(spin.P(2, 2)), sxy = std::sqrt(spin.P(0, 0) + spin.P(1, 1));
        std::printf("  one 5-rad step: sigma_yaw %.3f rad, sigma_xy %.3f m\n", sy, sxy);
        check("rot_scale_one_step_spin", sy >= kRotScaleSigma * 5.f * 0.8f and sxy > 0.5f);
        // The same spin in 100 steps of 0.05 rad (door_concept's ~10 Hz): a per-step (white) term accumulates as sqrt(N).
        RelPose s = r;
        for (int k = 0; k < 100; ++k) s = predict(s, twist(0, 0, 0.5f), zero, 0.1f);
        const float sy100 = std::sqrt(s.P(2, 2));
        std::printf("  the same spin in 100 steps: sigma_yaw %.3f rad (k*sqrt(sum dth^2) = %.3f)\n", sy100,
                    std::sqrt(4e-4f + kRotScaleSigma * kRotScaleSigma * 100.f * 0.05f * 0.05f));
        check("rot_scale_per_step_accumulates", sy100 > 2.f * 0.02f);
        const auto tr = predict(r, twist(0.1f, 0.5f, 0.f), zero, 1.f);
        check("rot_scale_pure_translation_unchanged", (tr.P - r.P).norm() < 1e-7f); }
    {   Eigen::Affine3d rr = Eigen::Affine3d::Identity(), ro = Eigen::Affine3d::Identity();
        rr.translate(Eigen::Vector3d(1, 1, 0)); rr.rotate(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
        ro.translate(Eigen::Vector3d(2, 4, 0)); ro.rotate(Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()));
        const auto r = from_room(rr, ro, Eigen::Matrix3f::Identity());
        Eigen::Affine3d rel = Eigen::Affine3d::Identity();
        rel.translate(Eigen::Vector3d(r.x.x(), r.x.y(), 0)); rel.rotate(Eigen::AngleAxisd(r.x.z(), Eigen::Vector3d::UnitZ()));
        const Eigen::Affine3d back = rr * rel;
        check("from_room_round_trip", (back.matrix() - ro.matrix()).norm() < 1e-5); }
    {   RelPose r; r.x = {0, 2, 0}; r.P.setZero();
        const Eigen::Vector3f v(0.04f, 0.05f, 0.06f);
        const float t1 = predict(r, twist(0, 0, 0), v, 1.f).P.trace();
        const float t5 = predict(r, twist(0, 0, 0), v, 0.5f).P.trace();
        check("process_noise_scales_with_dt_squared", t1 > 0 and std::fabs(t5 - 0.25f * t1) < 1e-6f); }
    {   const Eigen::Vector3f x(0.7f, 1.9f, 0.4f);
        const float adv = 0.3f * 0.4f, side = -0.1f * 0.4f, dth = 0.6f * 0.4f, e = 1e-4f;
        Eigen::Matrix3f J, G, Jn, Gn;
        predict_jacobians(x, side, adv, dth, J, G);
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3f d = Eigen::Vector3f::Zero(); d[i] = e;
            Jn.col(i) = (move_mean(x + d, side, adv, dth) - move_mean(x - d, side, adv, dth)) / (2 * e);
        }
        Gn.col(0) = (move_mean(x, side, adv + e, dth) - move_mean(x, side, adv - e, dth)) / (2 * e);
        Gn.col(1) = (move_mean(x, side + e, adv, dth) - move_mean(x, side - e, adv, dth)) / (2 * e);
        Gn.col(2) = (move_mean(x, side, adv, dth + e) - move_mean(x, side, adv, dth - e)) / (2 * e);
        check("jacobians_match_finite_differences", (J - Jn).cwiseAbs().maxCoeff() < 1e-3f and (G - Gn).cwiseAbs().maxCoeff() < 1e-3f); }
    std::puts(fails == 0 ? "ALL PASS" : "FAILED");
    return fails;
}
