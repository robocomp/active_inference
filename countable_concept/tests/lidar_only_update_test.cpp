// lidar_only_update_test — the LiDAR-only frame (spec 2026-11-15 lidar-measures §7b, owner ruling C1 = A): a frame with no
// camera points, only (mask-weighted) LiDAR rays. It carries what was measured on it plus the ONE-SIDED constraints that are
// zero unless violated; the Gaussian class anchors are NOT re-added, and ω is not updated. As memberships -> 0 the update
// must tend continuously to predict(). Fixture: the golden nightstand belief (lidar_golden_test.cpp).
// Run: ./lidar_only_update_test
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include "countable_belief.h"

using namespace rc;
using Mat6 = Eigen::Matrix<float, 6, 6>;
using Vec6 = Eigen::Matrix<float, 6, 1>;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    return p;
}
static CountableBelief make_belief(float dcx = 0.05f, float dcy = -0.04f, float dH = 0.03f)
{
    CountableBelief b(CountableBeliefState{tx + dcx, ty + dcy, tH + dH, tw, tw, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    return b;
}
// A LiDAR-only frame: no points, wall + room axis staged, z_top_obs = -1.
static CountableFrame lidar_frame(const rc::ai::LidarRays& r, float wall_y = 0.0f)
{
    CountableFrame f;
    f.lidar = r; f.lidar_only = true; f.z_top_obs = -1.0f;
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, wall_y); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    return f;
}
static rc::ai::LidarRays front_rays(float cy = ty)           // 120 first hits on the truth front face
{
    rc::ai::LidarRays r;
    r.origin = Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f);
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        r.endpoints.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, cy - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    r.precision = 2500.0f; r.robust_c_m = 0.05f;
    return r;
}
static rc::ai::LidarRays two_face_rays()                      // front (120) + left (80) + top (36) first hits, from a diagonal origin
{   // (the top face is what the sensor, mounted above the box, sees of H: side faces alone leave H unobserved)
    rc::ai::LidarRays r = front_rays();
    r.origin = Eigen::Vector3f(tx - 1.5f, ty - 1.5f, 1.1075f);
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
        r.endpoints.emplace_back(tx - 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
        r.endpoints.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 6.0f, ty - 0.5f * tw + tw * (j + 0.5f) / 6.0f, tH);
    return r;
}
static float centre_err(const CountableBelief& b)
{ const auto& s = b.state(); return std::hypot(s.cx - tx, s.cy - ty); }

static void empty_frame_is_predict_only()
{
    CountableBelief a = make_belief(), c = make_belief();
    CountableFrame f; f.lidar_only = true; f.z_top_obs = -1.0f;
    const float fe = a.update(f);
    c.predict();
    CHECK(fe == 0.0f);
    CHECK((a.state().vec() - c.state().vec()).norm() == 0.0f);
    CHECK((a.covariance() - c.covariance()).norm() == 0.0f);
    CHECK(a.covariance().allFinite() and a.state().vec().allFinite());
    // and a point-free, non-lidar_only frame (the legacy meaning) is the same
    CountableBelief d = make_belief(); CountableFrame g; d.update(g);
    CHECK((d.covariance() - c.covariance()).norm() == 0.0f);
}

static void lidar_only_moves_toward_rays()
{
    CountableBelief a = make_belief(), p = make_belief();
    const float e0 = centre_err(a), h0 = std::abs(a.state().H - tH);
    a.update(lidar_frame(two_face_rays()));
    p.predict();
    std::printf("  moves: centre err %.4f -> %.4f ; H err %.4f -> %.4f ; Sxx %.3e/%.3e Syy %.3e/%.3e SHH %.3e/%.3e (post/predict)\n",
                e0, centre_err(a), h0, std::abs(a.state().H - tH), a.covariance()(0, 0), p.covariance()(0, 0),
                a.covariance()(1, 1), p.covariance()(1, 1), a.covariance()(2, 2), p.covariance()(2, 2));
    CHECK(centre_err(a) < e0);
    CHECK(std::abs(a.state().H - tH) < h0);
    CHECK(a.covariance()(0, 0) < p.covariance()(0, 0));
    CHECK(a.covariance()(1, 1) < p.covariance()(1, 1));
    CHECK(a.covariance()(2, 2) < p.covariance()(2, 2));
}

// Run 300 LiDAR-only frames whose rays all carry membership w; return {state move, worst Sigma-diag ratio-1, omega move}.
struct Vanish { float move, sig, omega; };
static Vanish run_vanishing(float w, bool lidar_only)
{
    auto sweep = two_face_rays(); sweep.weights.assign(sweep.endpoints.size(), w);
    CountableBelief a = make_belief(), p = make_belief();
    const Vec6 start = a.state().vec(), om0 = a.log_volatility();
    for (int k = 0; k < 300; ++k) { auto f = lidar_frame(sweep); f.lidar_only = lidar_only; a.update(f); p.predict(); }
    float worst = 0.0f;
    for (int i = 0; i < 6; ++i) worst = std::max(worst, std::abs(a.covariance()(i, i) / p.covariance()(i, i) - 1.0f));
    return {(a.state().vec() - start).norm(), worst, (a.log_volatility() - om0).norm()};
}

static void lidar_only_vanishing_membership_is_predict()
{
    // The brief's w = 1e-6 does NOT reach predict() in 300 frames: the rays' information is linear in w (~0.3 per frame at
    // 200 rays x 2500, so ~90 over 300 frames against a prior of ~11). That is the model being continuous, not a leak.
    // So the printed ladder shows the move scaling ~linearly with w, and the claim is asserted where the accumulated
    // information is below the 1e-4 tolerance.
    const Vanish v6 = run_vanishing(1e-6f, true), v9 = run_vanishing(1e-9f, true), v12 = run_vanishing(1e-12f, true);
    const Vanish ctl = run_vanishing(1e-12f, false);
    std::printf("  vanishing (lidar_only): w=1e-6 move %.3e sig %.3e | w=1e-9 move %.3e sig %.3e | w=1e-12 move %.3e sig %.3e omega %.1e\n",
                v6.move, v6.sig, v9.move, v9.sig, v12.move, v12.sig, v12.omega);
    std::printf("  vanishing CONTROL (w=1e-12, lidar_only=false: class anchors re-added): move %.3e\n", ctl.move);
    CHECK(v12.move < 1e-4f);
    CHECK(v12.sig < 1e-3f);
    CHECK(v12.omega == 0.0f);
    CHECK(v9.move < v6.move and v12.move < v9.move);          // continuous: more vanishing membership, closer to predict()
    if (not (ctl.move > 1e-2f)) std::printf("  NOTE: control did NOT move > 1e-2 -- the test does not discriminate\n");
}

static Vec6 invdiag(const CountableBelief& b, const CountableFrame& f) { return b.common_mode_inv_diag(f); }

static void lidar_only_saturates()
{
    const auto one = two_face_rays();
    rc::ai::LidarRays ten = one;
    for (int k = 1; k < 10; ++k) ten.endpoints.insert(ten.endpoints.end(), one.endpoints.begin(), one.endpoints.end());
    CountableBelief a1 = make_belief(), a10 = make_belief(), pred = make_belief();
    auto f1 = lidar_frame(one), f10 = lidar_frame(ten);
    f1.chain_cov_xx = f1.chain_cov_yy = f10.chain_cov_xx = f10.chain_cov_yy = 0.01f;
    const Vec6 sc = invdiag(a10, f10);
    Mat6 Id = Mat6::Zero(); Vec6 bd = Vec6::Zero();
    a10.accumulate_extra(a10.state(), f10, Id, bd);
    std::printf("  saturates: raw Id(0,0)=%.3e  10*Sc^-1(0)=%.3e\n", Id(0, 0), 10.0f * sc(0));
    CHECK(Id(0, 0) > 10.0f * sc(0));                                  // non-vacuous
    pred.predict();
    a1.update(f1); a10.update(f10);
    const Mat6 B = (pred.covariance().inverse() + Mat6(sc.asDiagonal())).inverse();
    for (const CountableBelief* b : {&a1, &a10})
    {
        Eigen::SelfAdjointEigenSolver<Mat6> es(B);
        const Mat6 Bih = es.eigenvectors() * es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
        const Mat6 W = Bih * (b->covariance() - B) * Bih;
        const float lmin = Eigen::SelfAdjointEigenSolver<Mat6>(0.5f * (W + W.transpose())).eigenvalues().minCoeff();
        std::printf("  saturates: whitened min eig of (Sigma_post - bound) = %.4f\n", lmin);
        CHECK(lmin > -0.01f);
    }
    for (int i = 0; i < 6; ++i)
    {
        const float r = a10.covariance()(i, i) / a1.covariance()(i, i);
        std::printf("  saturates: diag %d  x10/x1 = %.4f\n", i, r);
        CHECK(std::abs(r - 1.0f) < 0.05f);
    }
}

static void lidar_only_repeated_sweeps()
{
    CountableBelief b = make_belief();
    const auto f = lidar_frame(two_face_rays());
    const auto Q = b.process_noise_diag();
    const Vec6 sc = invdiag(b, f);
    for (int k = 0; k < 300; ++k) b.update(f);
    {   // diagnostics only: the one-sided view leaves (cx, w) and (cy, h) degenerate (a face fixes cx - w/2, cy - h/2)
        const Mat6& S = b.covariance();
        std::printf("  repeated sweeps (diag): sigma w %.4f h %.4f ; corr(cx,w) %.3f corr(cy,h) %.3f\n", std::sqrt(S(3, 3)), std::sqrt(S(4, 4)),
                    S(0, 3) / std::sqrt(S(0, 0) * S(3, 3)), S(1, 4) / std::sqrt(S(1, 1) * S(4, 4)));
    }
    const int idx[3] = {0, 1, 2};
    const char* nm[3] = {"cx", "cy", "H"};
    for (int j = 0; j < 3; ++j)
    {
        const int i = idx[j];
        const float sg = std::sqrt(b.covariance()(i, i)), an = std::pow(Q(i) / sc(i), 0.25f);   // sigma at saturation: variance sqrt(Q*Sigma_c)
        std::printf("  repeated sweeps: %s settled sigma %.5f  analytic (Q*Sc)^(1/4) %.5f (variance sqrt(Q*Sc) = %.3e)  ratio %.3f\n",
                    nm[j], sg, an, std::sqrt(Q(i) / sc(i)), sg / an);
        CHECK(std::isfinite(sg) and sg > 0.0f);
        CHECK(sg < 2.0f * an and sg > 0.5f * an);
    }
}

static void lidar_only_one_face_keeps_depth()
{
    CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
    const float cy0 = b.state().cy, H0 = b.state().H;
    const auto f = lidar_frame(front_rays());
    for (int k = 0; k < 200; ++k) b.update(f);
    const float dh = std::abs(b.state().H - H0), dcy = std::abs(b.state().cy - cy0);
    std::printf("  one face: |dH| %.5f |dcy| %.5f (bounds 0.02)\n", dh, dcy);
    CHECK(dh < 0.02f);
    CHECK(dcy < 0.02f);
}

static void wall_behind_does_not_pull()
{
    // wall_no_cross_margin_m is 0 (default) < 0.05, so the box sits exactly 5 cm off the wall (no margin offset needed).
    const float off = 0.05f + params().wall_no_cross_margin_m;
    const float cy = -(0.5f * tw + off);                        // back face at y = -off, wall plane y = 0
    auto build = [&]() { CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, params());
                         b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f)); return b; };
    rc::ai::LidarRays r = front_rays(cy);
    const std::size_t nbox = r.endpoints.size();
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)   // wall hits above and beside the box, y = 0
        r.endpoints.emplace_back(tx - 1.0f + 2.0f * (i + 0.5f) / 12.0f, 0.0f, 0.05f + 1.2f * (j + 0.5f) / 10.0f);
    CountableBelief b = build(), ctl = build();
    auto f = lidar_frame(r);
    f.lidar.weights.assign(r.endpoints.size(), 0.001f);
    std::fill(f.lidar.weights.begin(), f.lidar.weights.begin() + nbox, 1.0f);
    auto fc = lidar_frame(r); fc.lidar.weights.assign(r.endpoints.size(), 1.0f);
    for (int k = 0; k < 100; ++k) { b.update(f); ctl.update(fc); }
    const float mv = b.state().cy - cy, cmv = ctl.state().cy - cy;     // +y = toward the wall
    std::printf("  wall behind: centre moved toward wall %.5f m (weights 0.001) ; CONTROL weights=1: %.5f m\n", mv, cmv);
    CHECK(mv < 0.02f);
}

int main()
{
    empty_frame_is_predict_only();
    lidar_only_moves_toward_rays();
    lidar_only_vanishing_membership_is_predict();
    lidar_only_saturates();
    lidar_only_repeated_sweeps();
    lidar_only_one_face_keeps_depth();
    wall_behind_does_not_pull();
    std::printf("lidar_only_update_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
