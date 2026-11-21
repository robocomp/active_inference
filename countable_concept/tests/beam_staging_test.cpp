// beam_staging_test — ray staging for the smooth beam model (spec 2026-11-15 lidar-measures 7c.8, ray staging).
// Run: ./beam_staging_test
#include <cmath>
#include <cstdio>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "lidar_select/lidar_select.h"

using namespace rc::lidar_select;
using Lab = std::optional<std::pair<double, float>>;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static const Box kBox{0.5f, 0.0f, 1.0f};                        // cylinder r = 0.5 m at (4, 0), z in [0, 1]
static const Eigen::Vector2f kC{4.f, 0.f};
static const Eigen::Vector3f kO{0.f, 0.f, 0.5f};
static const Weights kW{};
static const auto yes = [](const Eigen::Vector3f&) -> Lab { return Lab{{0.6, 0.5f}}; };

static void ray_through_box_staged()
{
    rc::ai::beam::BeamRays out;
    BeamStageStats st;
    // 1 m behind the box (x = 5.5), on the +x ray through it; and a return on a ray that misses (y = 3)
    const std::vector<Eigen::Vector3f> sweep{{5.5f, 0.f, 0.5f}, {4.f, 3.f, 0.5f}};
    const int n = select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, yes, out, &st);
    CHECK(n == 1 and out.endpoints.size() == 1 and out.m.size() == 1);
    CHECK(std::abs(out.endpoints[0].x() - 5.5) < 1e-6 and std::abs(out.m[0] - 0.6) < 1e-15);
    CHECK(st.staged == 1 and st.no_label == 0 and st.in_box == 0);
    CHECK(out.rho_max == 6.0 and out.origin == Eigen::Vector3d(0, 0, 0.5));
}

static void short_return_staged()
{
    rc::ai::beam::BeamRays out;
    const std::vector<Eigen::Vector3f> sweep{{2.f, 0.f, 0.5f}};   // stops 2 m before the box on a ray through it
    CHECK(select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, yes, out, nullptr) == 1);
    CHECK(std::abs(out.endpoints[0].x() - 2.0) < 1e-9);          // the endpoint is the return, not the box
    // a ray above the z-band does not cross the cylinder inside the band
    rc::ai::beam::BeamRays out2;
    const std::vector<Eigen::Vector3f> high{{2.f, 0.f, 2.5f}};
    const Eigen::Vector3f oh{0.f, 0.f, 2.5f};
    CHECK(select_rays_beam(high, oh, kC, kBox, kW, 6.0, yes, out2, nullptr) == 0);
}

static void no_return_staged_at_rho_max()
{
    rc::ai::beam::BeamRays out;
    const std::vector<Eigen::Vector3f> sweep{{8.f, 0.f, 0.5f}};  // the return is 8 m away, rho_max = 6
    CHECK(select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, yes, out, nullptr) == 1);
    CHECK(std::abs((out.endpoints[0] - out.origin).norm() - 6.0) < 1e-9);
    CHECK(std::abs(out.endpoints[0].x() - 6.0) < 1e-9);
    // E finite through the beam model
    namespace bm = rc::ai::beam;
    bm::BeamPart part;
    part.c = Eigen::Vector3d(4.0, 0.0, 0.5);
    part.half = Eigen::Vector3d(0.3, 0.3, 0.5);
    part.J.setZero();
    out.sigma_s = 0.01;
    const auto cfg = bm::BeamConfig::make(*bm::StudentT::make(2500.0, 0.05));
    const double E = bm::beam_rays_energy(std::span<const bm::BeamPart>(&part, 1), out, cfg);
    CHECK(std::isfinite(E));
}

static void no_label_not_staged()
{
    rc::ai::beam::BeamRays out;
    BeamStageStats st;
    const std::vector<Eigen::Vector3f> sweep{{5.5f, 0.f, 0.5f}, {4.f, 0.1f, 0.5f}};
    const auto none = [](const Eigen::Vector3f&) -> Lab { return std::nullopt; };
    CHECK(select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, none, out, &st) == 0);
    CHECK(st.staged == 0 and st.no_label == 2 and st.in_box == 1 and st.kappa == 0.f);
}

static void kappa_unfloored()
{
    // a ring of returns around the box, so the angular weight is nonzero
    std::vector<Eigen::Vector3f> sweep;
    for (int i = 0; i < 80; ++i)
    {
        const float a = 0.4f + 2.2f * static_cast<float>(i) / 80.f;   // partial ring: R < 1
        sweep.emplace_back(4.f + 0.4f * std::cos(a), 0.4f * std::sin(a), 0.5f);
    }
    const double eps = 0.094;
    rc::ai::beam::BeamRays out;
    BeamStageStats st;
    const auto zero = [&](const Eigen::Vector3f&) -> Lab { return Lab{{eps, 0.0f}}; };
    select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, zero, out, &st);
    CHECK(st.kappa == 0.f and out.kappa == 0.0 and st.q_sum == 0.f and st.in_box == 80);
    // today's labels: kappa equals select_into_weighted's precision / base on the same sweep
    const auto q_of = [](const Eigen::Vector3f& p) -> float { return 0.2f + 0.7f * std::abs(std::sin(7.f * p.y())); };
    const auto lab = [&](const Eigen::Vector3f& p) -> Lab { return Lab{{eps + (1 - 2 * eps) * q_of(p), q_of(p)}}; };
    select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, lab, out, &st);
    rc::ai::LidarRays ref;
    const float base = 2500.f;
    select_into_weighted(sweep, kO, kC, kBox, base, kW, q_of, ref);
    CHECK(ref.precision > 0.f);
    CHECK(std::abs(static_cast<double>(st.kappa) - static_cast<double>(ref.precision / base)) < 1e-6);
    CHECK(std::abs(out.kappa - st.kappa) < 1e-12 and st.kappa > 0.f and st.kappa <= 1.f);
}

int main()
{
    ray_through_box_staged();
    short_return_staged();
    no_return_staged_at_rho_max();
    no_label_not_staged();
    kappa_unfloored();
    std::printf(bad == 0 ? "beam_staging_test: PASS\n" : "beam_staging_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
