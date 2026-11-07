// Offline harness for src/core/camera_model.h. Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_camera_model.cpp src/core/door_cycle_io.cpp -lzstd -o /tmp/test_camera_model && /tmp/test_camera_model
#include "../src/core/camera_model.h"
#include "../src/core/door_existence.h"
#include <cmath>
#include <cstdio>
#include <random>

using namespace rc::door::core;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main()
{
    std::mt19937_64 g(7);
    std::uniform_real_distribution<double> U(-5.0, 5.0), Uy(0.1, 8.0);

    // pinhole_matches_camera_project
    {
        CameraIntrinsics c; c.present = true; c.width = 1280; c.height = 720; c.focal_x = 530.5f; c.focal_y = 531.25f;
        const CameraSpec s = camera_spec(c);
        CHECK(s.model == Projection::Pinhole);
        for (int i = 0; i < 100; ++i)
        {
            const Eigen::Vector3d p(U(g), Uy(g), U(g));
            const auto q = project(s, p);
            CHECK(q.has_value());
            if (q) CHECK(*q == camera_project(c, p));
        }
        CHECK(not project(s, {0.3, 0.0, 0.2}).has_value());
        CHECK(not project(s, {0.3, -1.0, 0.2}).has_value());
    }
    // equirect_matches_cameraapi
    {
        CameraIntrinsics c; c.present = true; c.width = 1920; c.height = 960; c.projection_model = 1;
        c.equirect_az_sign = -1.f; c.equirect_az_offset = 0.f;
        const CameraSpec s = camera_spec(c);
        CHECK(s.model == Projection::Equirectangular);
        const auto a = project(s, {0, 1, 0});
        CHECK(a.has_value() and std::abs((*a)(0) - 960.0) < 1e-9 and std::abs((*a)(1) - 480.0) < 1e-9);
        const auto up = project(s, {0, 0, 1});
        CHECK(up.has_value() and std::abs((*up)(1)) < 1e-9);
        const auto dn = project(s, {0, 0, -2});
        CHECK(dn.has_value() and std::abs((*dn)(1) - 960.0) < 1e-9);
        for (int i = 0; i < 100; ++i)
        {
            const Eigen::Vector3d p(U(g), U(g), U(g));
            double u = ((-1.0 * std::atan2(p.x(), p.y()) + 0.0) / (2.0 * M_PI) + 0.5) * 1920;
            u = std::fmod(u, 1920.0); if (u < 0.0) u += 1920;
            const double v = (std::asin(-p.z() / p.norm()) / M_PI + 0.5) * 960;
            const auto q = project(s, p);
            CHECK(q.has_value() and std::abs((*q)(0) - u) < 1e-9 and std::abs((*q)(1) - v) < 1e-9);
        }
        CHECK(not project(s, {0, 0, 0}).has_value());
        // the sign mirrors: a point to the +X side lands left of centre
        const auto right = project(s, {1, 1, 0});
        CHECK(right.has_value() and (*right)(0) < 960.0);
    }
    // cylindrical: linear azimuth, tan elevation
    {
        CameraIntrinsics c; c.width = 1920; c.height = 960; c.projection_model = 2;
        const auto q = project(camera_spec(c), {0, 2, 1});
        const double fv = 1920.0 / (2.0 * M_PI);
        CHECK(q.has_value() and std::abs((*q)(0) - 960.0) < 1e-9 and std::abs((*q)(1) - (480.0 - fv * 0.5)) < 1e-9);
    }
    // equirect_seam_wraps
    {
        CameraIntrinsics c; c.width = 1920; c.height = 960; c.projection_model = 1;   // sign +1, offset 0
        for (const double az : {M_PI - 1e-6, M_PI + 1e-6, -M_PI + 1e-6, -M_PI - 1e-6, 3 * M_PI + 1e-3})
        {
            const Eigen::Vector3d p(std::sin(az), std::cos(az), 0.1);
            const auto q = project(camera_spec(c), p);
            CHECK(q.has_value() and (*q)(0) >= 0.0 and (*q)(0) < 1920.0);
        }
        c.equirect_az_offset = 2.0f;   // offset pushing past the seam wraps too
        const auto q = project(camera_spec(c), {std::sin(2.5), std::cos(2.5), 0});
        CHECK(q.has_value() and (*q)(0) >= 0.0 and (*q)(0) < 1920.0);
    }
    std::printf(g_fail == 0 ? "ALL PASS\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
