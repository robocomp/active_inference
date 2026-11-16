// viewpoint_rho_test — rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2).
// Run: ./viewpoint_rho_test
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include "countable_viewpoint.h"

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
    CHECK(rho_chain(V2::Zero(), 1e-4f, 1e-4f) == 1.f);
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

static void rho_chain_zero_correction_is_one()
{
    CHECK(rho_chain(V2::Zero(), 0.f, 0.f) == 1.f);
    CHECK(rho_chain(V2(0.f, 0.1f), 0.f, 1e-2f) > 0.f);   // zero variance, zero correction: that term contributes 0
    CHECK(rho_chain(V2(0.1f, 0.f), 0.f, 1e-2f) == 0.f);  // zero variance, non-zero correction: rho = 0
    CHECK(std::abs(rho_chain(V2(0.1f, 0.f), 1e-2f, 1e-2f) - std::exp(-0.5f)) < 1e-6f);
}

static void degenerate_geometry_unchanged()
{
    ViewpointPath p;
    accumulate_path(p, V2(1, 1), V2(1, 0), V2(1, 1));   // sensor at the object centre
    CHECK(p.dphi_rad == 0.f and p.dz_m == 0.f);
}

// ρ_chain fails CLOSED when odometry cannot be integrated (final review I1).
static void chain_fails_closed()
{
    const V2 p0(1.f, 2.f), moved = p0 + V2(0.2f, 0.f);
    const float var = 0.02f * 0.02f;
    {   // pose moves 0.2 m, no odometry for the stretch ⇒ not a correction
        ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, std::nullopt);
        CHECK(rho_chain(chain_correction(t), var, var) == 1.f);
    }
    {   // pose moves 0.2 m with matching odometry
        ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V2(0.2f, 0.f));
        CHECK(rho_chain(chain_correction(t), var, var) == 1.f);
    }
    {   // pose moves 0.2 m, odometry VALID and zero ⇒ a real localiser correction
        ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V2::Zero());
        CHECK(rho_chain(chain_correction(t), var, var) < 1.f);
        CHECK((chain_correction(t) - V2(0.2f, 0.f)).norm() < 1e-6f);
    }
    {   // an unknown stretch after a known correction keeps the earlier correction; rebase restarts it
        ChainTracker t; chain_observe(t, p0, std::nullopt); chain_observe(t, moved, V2::Zero());
        chain_observe(t, moved + V2(0.f, 0.5f), std::nullopt);
        CHECK((chain_correction(t) - V2(0.2f, 0.f)).norm() < 1e-6f);
        chain_rebase(t);
        CHECK(chain_correction(t).norm() == 0.f);
    }
    CHECK(chain_correction(ChainTracker{}).norm() == 0.f);   // before the first sample
}

int main()
{
    chain_fails_closed();
    parked_is_one(); stale_odometry_is_parked(); odometry_axes(); camera_efold_is_half_width(); lidar_efold();
    composes(); rotation_in_place_is_one(); absent_ring_gap_is_one(); rho_chain_zero_correction_is_one();
    degenerate_geometry_unchanged();
    std::printf(bad == 0 ? "viewpoint_rho_test: PASS\n" : "viewpoint_rho_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
