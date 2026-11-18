// viewpoint_rho_test — rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2).
// Run: ./viewpoint_rho_test
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <algorithm>
#include <span>
#include "countable_viewpoint.h"
#include "place_memory/place_map.h"

using namespace rc;
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

static void rho_chain_zero_correction_is_one_se2()
{
    CHECK(rho_chain(V3::Zero(), M3::Zero()) == 1.f);
}

// ρ_chain fails CLOSED when odometry cannot be integrated (final review I1), now on SE(2).
static void chain_fails_closed()
{
    const V3 p0(1.f, 2.f, 0.3f), moved = p0 + V3(0.2f, 0.f, 0.f);
    const M3 S = V3(0.02f * 0.02f, 0.02f * 0.02f, 0.02f * 0.02f).asDiagonal();
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, std::nullopt);
        CHECK(rho_chain(chain_correction(t), S) == 1.f); }
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V3(0.2f, 0.f, 0.f));
        CHECK(rho_chain(chain_correction(t), S) == 1.f); }
    {   ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V3::Zero());
        CHECK(rho_chain(chain_correction(t), S) < 1.f);
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
    const M3 R = pose_cov_in_room(std::span<const float>(blk, 36), edge);
    CHECK((R - S).norm() <= 1e-6f * S.norm());   // matrix-relative: float round-off leaves ~3e-7 on zero entries
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
    CHECK(std::abs(rho_chain(V3(0.f, 0.f, 0.024f), S) - std::exp(-0.5f)) < 1e-6f);
    M3 Z = S; Z(2, 2) = 0.f;
    CHECK(rho_chain(V3(0.f, 0.f, 0.01f), Z) == 0.f);
    CHECK(rho_chain(V3(0.01f, 0.f, 0.f), Z) > 0.f);
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

int main()
{
    chain_fails_closed(); rho_chain_zero_correction_is_one_se2(); pose_cov_round_trip(); pose_cov_absent_is_zero();
    chain_correction_se2(); chain_correction_wraps_psi(); rho_chain_se2(); lever_transport_invariance(); lever_relaxes_with_rho();
    parked_is_one(); stale_odometry_is_parked(); odometry_axes(); camera_efold_is_half_width(); lidar_efold();
    composes(); rotation_in_place_is_one(); absent_ring_gap_is_one();
    degenerate_geometry_unchanged();
    std::printf(bad == 0 ? "viewpoint_rho_test: PASS\n" : "viewpoint_rho_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
