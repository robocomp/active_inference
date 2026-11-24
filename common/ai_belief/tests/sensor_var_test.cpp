// sensor_var_test — sensor shared-error variance helpers (common/ai_belief/sensor_var.h) and the degenerate-geometry
// behaviour of the viewpoint decorrelation (common/ai_belief/viewpoint.h). Run: ./sensor_var_test
#include <cmath>
#include <cstdio>
#include "ai_belief/sensor_var.h"
#include "ai_belief/viewpoint.h"

using namespace rc::ai;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main()
{
    const double v1 = sensor::camera_shared_var(1.0, 2.0, 700.0, 1.0);
    CHECK(std::abs(v1 - (2.0 / 700.0) * (2.0 / 700.0)) < 1e-12);
    CHECK(std::abs(sensor::camera_shared_var(1.0, 2.0, 700.0, 0.5) - 4.0 * v1) < 1e-15);
    CHECK(std::abs(sensor::fresh_var(0.5, 0.04) - 4e-4) < 1e-15);
    CHECK(sensor::fresh_var(0.0, 0.04) == 0.0);
    CHECK(std::isfinite(sensor::camera_shared_var(1.0, 2.0, 700.0, 0.0)));

    // Degenerate geometry of the viewpoint decorrelation.
    view::ViewpointPath p;
    p.dphi_rad = 0.3f; p.dz_m = 0.2f;
    for (const float r : {view::rho_camera(p, 1e-6f, 0.5f), view::rho_camera(p, 2.f, 0.f)})
    {
        std::printf("rho_camera degenerate = %g\n", static_cast<double>(r));
        CHECK(std::isfinite(r));
        CHECK(r >= 0.f and r <= 1.f);
    }
    // w_perp = 0: zero path = no decorrelation exactly; dz-only path = pure radial term.
    view::ViewpointPath z0;
    CHECK(view::rho_camera(z0, 2.f, 0.f) == 1.f);
    view::ViewpointPath dz;
    dz.dz_m = 0.2f;
    CHECK(std::abs(view::rho_camera(dz, 2.f, 0.f) - std::exp(-0.2f / 2.f)) < 1e-6f);

    view::ViewpointPath q;
    view::accumulate_path(q, Eigen::Vector2f(1.0005f, 2.f), Eigen::Vector2f(0.1f, 0.1f), Eigen::Vector2f(1.f, 2.f));
    CHECK(q.dphi_rad == 0.f and q.dz_m == 0.f);

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
