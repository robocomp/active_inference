// viewpoint_rho_test — rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2).
// Run: ./viewpoint_rho_test  (moved from countable_concept/tests; now tests common/ai_belief/viewpoint.h)
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <algorithm>
#include <span>
#include <cstdint>
#include <vector>
#include "ai_belief/viewpoint.h"
#include "place_memory/place_map.h"

using namespace rc;
using namespace rc::ai::view;
using V2 = Eigen::Vector2f;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// Move the sensor along the circle of radius z about the object by arc length L, in n steps (tangent each step).
static ViewpointPath arc(float z, float L, int n)
{
    ViewpointPath p;
    const V2 obj(1.f, 2.f);
    float th = 0.f;
    for (int i = 0; i < n; ++i)
    {
        const V2 s0 = obj + z * V2(std::cos(th), std::sin(th));
        const float dth = (L / n) / z;
        const V2 s1 = obj + z * V2(std::cos(th + dth), std::sin(th + dth));
        accumulate_path(p, s0, s1 - s0, obj);
        th += dth;
    }
    return p;
}

static void parked_is_one()
{
    const ViewpointPath p;
    CHECK(rho_camera(p, 2.f, 0.5f) == 1.f);
    CHECK(rho_lidar_lateral(p, 0.0392f) == 1.f);
    CHECK(rho_lidar_height(p, 2.f, 0.3f, 0.0392f) == 1.f);
}

static void stale_odometry_is_parked()
{
    ViewpointPath p;
    const V2 obj(0, 0), sensor(2, 0);
    accumulate_path(p, sensor, odometry_step_room(1.f, 0.5f, 0.7f, 0.f), obj);   // dt = 0
    accumulate_path(p, sensor, odometry_step_room(0.f, 0.f, 0.7f, 0.1f), obj);   // speed 0
    CHECK(rho_camera(p, 2.f, 0.5f) == 1.f);
    CHECK(rho_lidar_lateral(p, 0.0392f) == 1.f);
}

static void odometry_axes()
{
    const V2 a = odometry_step_room(1.f, 0.f, 0.f, 1.f);                    // forward at yaw 0 = +y
    CHECK(std::abs(a.x()) < 1e-6f and std::abs(a.y() - 1.f) < 1e-6f);
    const V2 b = odometry_step_room(1.f, 0.f, static_cast<float>(M_PI_2), 1.f);   // yaw +90 deg (CCW): forward = -x
    CHECK(std::abs(b.x() + 1.f) < 1e-5f and std::abs(b.y()) < 1e-5f);
    const V2 c = odometry_step_room(0.f, 1.f, 0.f, 1.f);                    // side = +x (robot right)
    CHECK(std::abs(c.x() - 1.f) < 1e-6f and std::abs(c.y()) < 1e-6f);
}

static void camera_efold_is_half_width()
{
    const float w = 0.51f;
    for (float z : {1.f, 2.f, 3.f})
    {
        const auto p = arc(z, 0.5f * w, 50);
        const float r = rho_camera(p, z, w);
        std::printf("  camera z=%.0f rho=%.5f (e^-1=%.5f) rel=%.4f\n", z, r, std::exp(-1.f), r / std::exp(-1.f) - 1.f);
        // BRIEF CONCERN: the brief's e^-1 +-2% cannot hold at z = 1 m for rho = exp(-dphi/atan(w/2z)): a path of w/2
        // gives exp(-(w/2z)/atan(w/2z)), which is -2.09% off at z = 1 (-0.55% at 2, -0.25% at 3). z = 1 is checked
        // against that analytic value instead of loosening the 2% for the others.
        const float analytic = std::exp(-(0.5f * w / z) / std::atan(0.5f * w / z));
        CHECK(std::abs(r / analytic - 1.f) < 1e-3f);
        if (z > 1.5f) CHECK(std::abs(r / std::exp(-1.f) - 1.f) < 0.02f);
    }
}

static void lidar_efold()
{
    const float g = 0.0392f;
    for (float z : {1.f, 2.f, 3.f})
    {
        const auto p = arc(z, z * g, 50);
        const float r = rho_lidar_lateral(p, g);
        CHECK(std::abs(r / std::exp(-1.f) - 1.f) < 0.01f);
    }
}

static void composes()
{
    const V2 obj(0, 0), sensor(2, 0.5f);
    ViewpointPath one, many;
    const V2 dir = V2(0.6f, 0.8f);
    accumulate_path(one, sensor, 0.30f * dir, obj);
    for (int i = 0; i < 100; ++i) accumulate_path(many, sensor, 0.003f * dir, obj);
    CHECK(std::abs(rho_camera(one, 2.f, 0.5f) - rho_camera(many, 2.f, 0.5f)) < 1e-4f);
    CHECK(std::abs(rho_lidar_lateral(one, 0.0392f) - rho_lidar_lateral(many, 0.0392f)) < 1e-4f);
    CHECK(std::abs(rho_lidar_height(one, 2.f, 0.3f, 0.0392f) - rho_lidar_height(many, 2.f, 0.3f, 0.0392f)) < 1e-4f);
}

static void rotation_in_place_is_one()
{
    ViewpointPath p;
    const V2 obj(0, 0), sensor(2, 1);
    for (int i = 0; i < 100; ++i)
        accumulate_path(p, sensor, odometry_step_room(0.f, 0.f, 0.01f * i, 0.1f), obj);   // yaw changes, no translation
    CHECK(rho_camera(p, 2.f, 0.5f) == 1.f);
    CHECK(rho_lidar_lateral(p, 0.0392f) == 1.f);
    CHECK(rho_lidar_height(p, 2.f, 0.3f, 0.0392f) == 1.f);
}

static void absent_ring_gap_is_one()
{
    ViewpointPath p; p.dphi_rad = 1.f; p.dz_m = 1.f;
    CHECK(rho_lidar_lateral(p, 0.f) == 1.f);
    CHECK(rho_lidar_height(p, 2.f, 0.3f, 0.f) == 1.f);
}

static void degenerate_geometry_unchanged()
{
    ViewpointPath p;
    accumulate_path(p, V2(1, 1), V2(1, 0), V2(1, 1));   // sensor at the object centre
    CHECK(p.dphi_rad == 0.f and p.dz_m == 0.f);
}

using V3 = Eigen::Vector3f;
using M3 = Eigen::Matrix3f;
using V3d = Eigen::Vector3d;
using M3d = Eigen::Matrix3d;

static void rho_chain_zero_correction_is_one_se2()
{
    CHECK(rho_chain(V3d::Zero(), M3d::Zero()) == 1.0);
}

// ρ_chain fails CLOSED when odometry cannot be integrated (final review I1), now on SE(2).
static void chain_fails_closed()
{
    const V3 p0(1.f, 2.f, 0.3f), moved = p0 + V3(0.2f, 0.f, 0.f);
    const M3 S = V3(0.02f * 0.02f, 0.02f * 0.02f, 0.02f * 0.02f).asDiagonal();
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, std::nullopt);
        CHECK(rho_chain(chain_correction(t).cast<double>(), S.cast<double>()) == 1.0); }
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V3(0.2f, 0.f, 0.f));
        CHECK(std::abs(rho_chain(chain_correction(t).cast<double>(), S.cast<double>()) - 1.0) < 1e-9); }
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V3::Zero());
        CHECK(rho_chain(chain_correction(t).cast<double>(), S.cast<double>()) < 1.0);
        CHECK((chain_correction(t) - V3(0.2f, 0.f, 0.f)).norm() < 1e-6f); }
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V3::Zero());
        chain_observe(t, moved + V3(0.f, 0.5f, 0.f), std::nullopt);
        CHECK((chain_correction(t) - V3(0.2f, 0.f, 0.f)).norm() < 1e-6f);
        chain_rebase(t);
        CHECK(chain_correction(t).norm() == 0.f); }
    CHECK(chain_correction(ChainTracker{}).norm() == 0.f);
}

static void pose_cov_round_trip()
{
    M3 S = V3(0.034f * 0.034f, 0.0435f * 0.0435f, 0.024f * 0.024f).asDiagonal();
    S(0, 2) = S(2, 0) = 0.3f * std::sqrt(S(0, 0) * S(2, 2));
    const V3 robot(0.83f, 2.24f, 0.42f);
    const float c = std::cos(robot.z()), s = std::sin(robot.z());
    const V3 edge(-(c * robot.x() + s * robot.y()), -(-s * robot.x() + c * robot.y()), -robot.z());
    const M3 Se = place::invert_se2_cov(S, robot);
    float blk[36] = {};
    const int idx[3] = {0, 1, 5};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) blk[idx[i] * 6 + idx[j]] = Se(i, j);
    const M3d R = pose_cov_in_room(std::span<const float>(blk, 36), edge);
    CHECK((R.cast<float>() - S).norm() <= 1e-6f * S.norm());   // matrix-relative: float round-off leaves ~3e-7 on zero entries
}

static void pose_cov_absent_is_zero()
{
    CHECK(pose_cov_in_room({}, V3(1.f, 2.f, 0.3f)).norm() == 0.f);
}

static void chain_correction_se2()
{
    ChainTracker t;
    V3 pose(1.f, 2.f, 0.1f);
    chain_observe(t, pose, std::nullopt);
    const V3 od(0.1f, 0.f, 0.05f), drift(0.01f, -0.02f, 0.003f);
    for (int i = 0; i < 10; ++i) { pose += od + drift; chain_observe(t, pose, od); }
    CHECK((chain_correction(t) - 10.f * drift).norm() < 1e-4f);   // 10 samples x drift
    // brief: poses drift by (0.01,-0.02,0.003) beyond odometry in total
    ChainTracker u; V3 q(1.f, 2.f, 0.1f); chain_observe(u, q, std::nullopt);
    for (int i = 0; i < 10; ++i) { q += od; if (i == 9) q += drift; chain_observe(u, q, od); }
    CHECK((chain_correction(u) - drift).norm() < 1e-6f);
    // a nullopt stretch is absorbed on all three components
    chain_observe(u, q + V3(0.3f, -0.4f, 0.2f), std::nullopt);
    CHECK((chain_correction(u) - drift).norm() < 1e-6f);
}

static void chain_correction_wraps_psi()
{
    ChainTracker t;
    chain_observe(t, V3(0.f, 0.f, 3.10f), std::nullopt);
    chain_observe(t, V3(0.f, 0.f, -3.10f), V3::Zero());
    CHECK(std::abs(chain_correction(t).z() - (2.f * static_cast<float>(M_PI) - 6.20f)) < 1e-6f);
}

static void rho_chain_se2()
{
    const M3 S = V3(0.034f * 0.034f, 0.0435f * 0.0435f, 0.024f * 0.024f).asDiagonal();
    CHECK(std::abs(rho_chain(V3d(0, 0, 0.024), S.cast<double>()) - std::exp(-0.5)) < 1e-6);
    M3 Z = S; Z(2, 2) = 0.f;
    CHECK(rho_chain(V3d(0, 0, 0.01), Z.cast<double>()) == 0.0);
    CHECK(rho_chain(V3d(0.01, 0, 0), Z.cast<double>()) > 0.0);
}

static void lever_transport_invariance()
{
    const Eigen::Vector2f l0(0.09f, 1.65f), dr(0.3f, -0.2f);
    Eigen::Vector2f l1 = l0;
    ChainLever L; L.l = l0; L.set = true;
    lever_step(L, dr, 1.f, Eigen::Vector2f::Zero(), Eigen::Vector2f::Zero());
    l1 = L.l;
    const Eigen::Matrix<double, 6, 3> lhs = chain_B(l1) * chain_transport(dr), rhs = chain_B(l0);
    CHECK((lhs - rhs).cwiseAbs().maxCoeff() < 1e-6);   // float lever
}

static void lever_relaxes_with_rho()
{
    const Eigen::Vector2f c(2.f, 3.f), r(0.5f, 1.f), dr(0.1f, 0.1f);
    ChainLever L; lever_start(L, Eigen::Vector2f(9.f, 9.f), Eigen::Vector2f(0.f, 0.f));
    CHECK(L.set);
    lever_step(L, dr, 0.f, c, r);
    CHECK((L.l - (c - r)).norm() == 0.f);
    ChainLever M; M.l = Eigen::Vector2f(1.f, 1.f); M.set = true;
    lever_step(M, dr, 0.5f, c, r);
    CHECK((M.l - 0.5f * (Eigen::Vector2f(1.f, 1.f) - dr + (c - r))).norm() < 1e-6f);
}

static void rho_chain_rank_deficient_float()
{
    M3 S = M3::Zero();
    S(0, 0) = 0.034f * 0.034f; S(1, 1) = 0.0435f * 0.0435f;
    S(0, 1) = S(1, 0) = 0.5f * 0.034f * 0.0435f;   // x-y cross term, psi variance exactly 0
    const V3 robot(0.83f, 2.24f, 0.42f);
    const float cy = std::cos(robot.z()), sy = std::sin(robot.z());
    const V3 edge(-(cy * robot.x() + sy * robot.y()), -(-sy * robot.x() + cy * robot.y()), -robot.z());
    const M3 Se = place::invert_se2_cov(S, robot);   // float, as an rt_covariance would be
    float blk[36] = {};
    const int idx[3] = {0, 1, 5};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) blk[idx[i] * 6 + idx[j]] = Se(i, j);
    const M3d R = pose_cov_in_room(std::span<const float>(blk, 36), edge);
    const V3d c(0.02, -0.01, 0.0);
    const Eigen::Matrix2f S2 = S.topLeftCorner<2, 2>();
    CHECK(R(2, 2) == 0.0 and R(0, 2) == 0.0 and R(1, 2) == 0.0);   // psi null survives J and back EXACTLY
    const double want = std::exp(-0.5 * c.head<2>().dot(S2.cast<double>().inverse() * c.head<2>()));
    CHECK(std::abs(rho_chain(c, R) - want) < 1e-5);
    CHECK(want > 0.0 and std::abs(want - 1.0) > 1e-3);                 // positive control is non-trivial
    CHECK(rho_chain(V3d(c.x(), c.y(), 0.01), R) == 0.0);                // in-range xy part + psi in the null space
    CHECK(rho_chain(V3d(0, 0, 0.01), R) == 0.0);
}

static void rho_chain_near_singular_is_continuous()
{
    double prev = 2.0;
    for (const double s : {1e-6, 1e-10, 1e-14})
    {
        const M3d S = V3d(0.034 * 0.034, 0.0435 * 0.0435, s).asDiagonal();
        const double r = rho_chain(V3d(0, 0, 1e-3), S);
        CHECK(std::isfinite(r) and r >= 0.0 and r <= prev);
        prev = r;
    }
    CHECK(prev < 1e-3);   // exp(-0.5e-6/s) underflows to exactly 0 for s <= 1e-10, hence non-strict
}

static void rho_chain_never_above_one()
{
    M3 S = V3(1e-4f, 1e-4f, 1e-4f).asDiagonal();
    S(2, 2) = -1e-12f;
    const double r = rho_chain(V3d(0.01, 0, 0), S.cast<double>());
    CHECK(r >= 0.0 and r <= 1.0);
}


// ── F1 (task-F1): ρ from the SEPARATION of viewpoints (anchor + paid frontier), not from the odometry arc length ────────────
// One update per sensor position s_k (object at the origin): dr = s_k − s_{k−1}. F1 = view_stage / anchor_separation /
// view_commit, exactly as the fitter calls them; OLD = the path reset after every update, accumulate_path over the same step
// (what the fitter did before F1; integrated in 1000 chords so it is the step's arc length, not a one-chord approximation).
struct F1Run { std::vector<float> rho_new, rho_old; };
static F1Run run_f1(const std::vector<V2>& s, float g, float w)
{
    F1Run out;
    const V2 obj(0.f, 0.f);
    ViewState v;
    for (std::size_t k = 0; k < s.size(); ++k)
    {
        const V2 dr = k == 0 ? V2::Zero() : V2(s[k] - s[k - 1]);
        v = view_stage(v, s[k], s[k], dr, obj);
        const float z = s[k].norm();
        const float rl = rho_lidar_lateral(anchor_separation(v.a_lat, v.L), g);
        const float rc = rho_camera(anchor_separation(v.a_cam, v.cam), z, w);
        const float rh = rho_lidar_height(anchor_separation(v.a_H, v.L), z, 0.3f, g);
        v = view_commit(v, true, true, rc, rl, rh);
        ViewpointPath p;   // the old arc length of the step, integrated finely (1000 chords): the path's own length
        if (k > 0) for (int j = 0; j < 1000; ++j) accumulate_path(p, s[k - 1] + (float(j + 1) / 1000.f) * dr, dr / 1000.f, obj);
        if (k > 0) { out.rho_new.push_back(rl); out.rho_old.push_back(rho_lidar_lateral(p, g)); }
        if (k > 0) { out.rho_new.push_back(rc); out.rho_old.push_back(rho_camera(p, z, w)); }
        if (k > 0) { out.rho_new.push_back(rh); out.rho_old.push_back(rho_lidar_height(p, z, 0.3f, g)); }
    }
    return out;
}
// Σ(1 − ρ²) of one latent (stride 3: 0 = lateral, 1 = camera, 2 = height), the number of fresh views it was re-drawn by.
static double redraws(const std::vector<float>& r, int which)
{
    double a = 0.0;
    for (std::size_t i = std::size_t(which); i < r.size(); i += 3) a += 1.0 - double(r[i]) * double(r[i]);
    return a;
}

// (1) Zero-mean jitter of radius r around one view: Σ(1 − ρ²) stays O(r/ℓ) and does not grow with N; the old arc length grows
//     linearly. Jitter uniform in a disc of r = 2 mm at z = 2 m (r/z = 1 mrad against g = 39 mrad).
static void f1_jitter_does_not_accumulate()
{
    const float g = 0.0392f, w = 0.5f, r = 0.002f, z = 2.f;
    std::uint32_t st = 12345u;
    const auto u01 = [&] { st = st * 1664525u + 1013904223u; return float(st >> 8) / float(1u << 24); };
    std::vector<V2> s;
    for (int k = 0; k <= 10000; ++k)
    {
        float x, y;
        do { x = 2.f * u01() - 1.f; y = 2.f * u01() - 1.f; } while (x * x + y * y > 1.f);
        s.emplace_back(z + r * x, r * y);
    }
    double new_n[3] = {}, old_n[3] = {};
    const int Ns[3] = {100, 1000, 10000};
    for (int i = 0; i < 3; ++i)
    {
        const F1Run f = run_f1(std::vector<V2>(s.begin(), s.begin() + Ns[i] + 1), g, w);
        new_n[i] = redraws(f.rho_new, 0); old_n[i] = redraws(f.rho_old, 0);
        std::printf("  jitter N=%5d  Σ(1-ρ²) lateral: F1 %.4f  old %.3f   camera: F1 %.5f old %.4f   height: F1 %.5f old %.4f\n",
                    Ns[i], new_n[i], old_n[i], redraws(f.rho_new, 1), redraws(f.rho_old, 1), redraws(f.rho_new, 2),
                    redraws(f.rho_old, 2));
        // O(r/ℓ): the separations stay inside the jitter's hull (2r across), so Σ −ln ρ ≤ (2r/z)/g to first order and
        // Σ(1 − ρ²) ≤ 2·Σ −ln ρ. 4.5 = 2 · 2 + the anchor's own small drift.
        CHECK(new_n[i] <= 4.5 * double(r / z) / double(g));
    }
    CHECK(new_n[2] < 2.0 * new_n[0]);          // does not grow with N (×100 samples)
    CHECK(old_n[2] > 50.0 * old_n[0]);         // the old arc length does (linearly)
    CHECK(old_n[2] > 100.0 * new_n[2]);
}

// (2) Monotone straight drive: the per-update ρ (and so the product) equals the old arc-length ρ to float precision, on every
//     latent — radially outward, and on a line whose bearing and range both grow.
static void f1_monotone_drive_matches_arc_length()
{
    const float g = 0.0392f, w = 0.5f;
    for (const V2 dir : {V2(1.f, 0.f), V2(0.6f, 0.8f)})
    {
        std::vector<V2> s;
        for (int k = 0; k <= 200; ++k) s.push_back(V2(2.f, 0.3f) + 0.005f * float(k) * dir);
        const F1Run f = run_f1(s, g, w);
        double worst = 0.0, lp_new = 0.0, lp_old = 0.0;
        for (std::size_t i = 0; i < f.rho_new.size(); ++i)
        {
            worst = std::max(worst, std::abs(double(f.rho_new[i]) - double(f.rho_old[i])));
            lp_new += std::log(double(f.rho_new[i])); lp_old += std::log(double(f.rho_old[i]));
        }
        std::printf("  monotone dir (%.1f,%.1f): max |ρ_F1 − ρ_old| %.2e, Σ ln ρ F1 %.6f old %.6f\n", dir.x(), dir.y(), worst,
                    lp_new, lp_old);
        CHECK(worst < 2e-5);                                       // float: φ is a running sum, the difference is ~1e-7 rad
        CHECK(std::abs(lp_new - lp_old) < 1e-4 * std::abs(lp_old));
        CHECK(lp_old < -1.0);                                      // positive control: the drive does decorrelate
    }
}

// (3) Out and back along the circle (pure bearing): on the way back ρ = 1 while the view is inside what the latent was already
//     decorrelated over (★ the feasibility bound active), so a short excursion is not re-drawn twice; a long one is re-drawn
//     only beyond the frontier (≈ 2ℓ behind the turn), never the full way back as the arc length does.
static void f1_out_and_back()
{
    const float g = 0.0392f, w = 0.5f, z = 2.f;
    for (const float D : {0.5f, 3.f})   // out-leg length in units of ℓ = g (angle)
    {
        const int n = 200;
        std::vector<V2> s;
        for (int k = 0; k <= n; ++k) { const float th = D * g * float(k) / n; s.emplace_back(z * std::cos(th), z * std::sin(th)); }
        for (int k = n - 1; k >= 0; --k) s.push_back(s[std::size_t(k)]);
        const F1Run f = run_f1(s, g, w);
        double out_new = 0.0, back_new = 0.0, out_old = 0.0, back_old = 0.0;
        int back_one = 0;
        for (std::size_t i = 0; i < f.rho_new.size(); i += 3)
        {
            const bool back = i / 3 >= std::size_t(n);
            (back ? back_new : out_new) += -std::log(double(f.rho_new[i]));
            (back ? back_old : out_old) += -std::log(double(f.rho_old[i]));
            if (back and f.rho_new[i] == 1.f) ++back_one;
        }
        std::printf("  out-and-back D=%.1fℓ: Σ −ln ρ_lateral out F1 %.4f old %.4f | back F1 %.4f old %.4f | back ρ==1 on %d of %d\n",
                    D, out_new, out_old, back_new, back_old, back_one, n);
        CHECK(std::abs(out_new - out_old) < 1e-3 * out_old);       // the out leg is a monotone drive
        if (D < 1.f)
        {
            CHECK(back_one == n);                                  // every return update at the bound: no second draw
            CHECK(back_new == 0.0);
        }
        else
        {   // the frontier's trailing edge sits x − 2L(x) behind the turn, L(x) = ℓ(1 − e^{−x/ℓ}) (the anchor's lag)
            const double L = 1.0 - std::exp(-double(D)), want = double(D) - 2.0 * L;
            CHECK(std::abs(back_new - want) < 0.03 * want);
            CHECK(back_one > 0 and back_new < 0.5 * back_old);
        }
    }
}

int main()
{
    chain_fails_closed(); rho_chain_zero_correction_is_one_se2(); pose_cov_round_trip(); pose_cov_absent_is_zero();
    chain_correction_se2(); chain_correction_wraps_psi(); rho_chain_se2(); lever_transport_invariance(); lever_relaxes_with_rho();
    rho_chain_rank_deficient_float(); rho_chain_never_above_one(); rho_chain_near_singular_is_continuous();
    parked_is_one(); stale_odometry_is_parked(); odometry_axes(); camera_efold_is_half_width(); lidar_efold();
    composes(); rotation_in_place_is_one(); absent_ring_gap_is_one();
    degenerate_geometry_unchanged();
    f1_jitter_does_not_accumulate(); f1_monotone_drive_matches_arc_length(); f1_out_and_back();
    std::printf(bad == 0 ? "viewpoint_rho_test: PASS\n" : "viewpoint_rho_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
