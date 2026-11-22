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
#include "countable_belief.h"
#include "beam_fixture.h"

using namespace rc::lidar_select;
using rc::CountableBelief; using rc::CountableBeliefParams; using rc::CountableBeliefState; using rc::CountableFrame;
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

// ─── "The LiDAR channel measured" (spec 7c.8 rev 3.3.11): ⇔ the staged set is non-empty and Σ q_i > 0 over it, q_i the
//     UNFLOORED label (m_i ≥ ε_px always). (a) staged rays all with q = 0 ⇒ not measured; (b) one q > 0 ⇒ measured; (c) the
//     live11 bed geometry — a LOW box, the sensor above it, every ray crossing the box's top band and landing on the floor
//     beyond the select cylinder: the endpoint-in-box set (the old κ's) is EMPTY, the staged set is not ⇒ measured.
static void measured_needs_camera_attribution()
{
    CountableBeliefParams p;
    p.persistent = true;
    p.beam = rc::test::test_beam_config();
    const CountableBelief b(CountableBeliefState{4.f, 0.f, 1.f, 0.6f, 0.6f, 0.f}, p);
    const double eps = 0.094;
    const auto frame = [](const rc::ai::beam::BeamRays& r) { CountableFrame f; f.beam = r; f.lidar_only = true; return f; };
    // (a), (b): returns 1 m behind the box on rays through it (the same geometry as ray_through_box_staged)
    std::vector<Eigen::Vector3f> sweep;
    for (int i = 0; i < 9; ++i) sweep.emplace_back(5.5f, -0.2f + 0.05f * static_cast<float>(i), 0.5f);
    {
        rc::ai::beam::BeamRays out;
        BeamStageStats st;
        const auto zero = [&](const Eigen::Vector3f&) -> Lab { return Lab{{eps, 0.0f}}; };
        select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, zero, out, &st);
        out.sigma_s = 0.01;
        const bool meas = b.lidar_measured(frame(out));
        std::printf("  measured (a) all q = 0: staged %d, q_sum_staged %.3f, BeamRays q_sum %.3f -> lidar_measured %d\n", st.staged,
                    st.q_sum_staged, out.q_sum, int(meas));
        CHECK(st.staged == 9 and st.q_sum_staged == 0.f and out.q_sum == 0.0);
        CHECK(not meas);
    }
    {
        rc::ai::beam::BeamRays out;
        BeamStageStats st;
        const auto one = [&](const Eigen::Vector3f& p) -> Lab
        { return std::abs(p.y()) < 1e-4f ? Lab{{eps + (1 - 2 * eps) * 0.3, 0.3f}} : Lab{{eps, 0.0f}}; };
        select_rays_beam(sweep, kO, kC, kBox, kW, 6.0, one, out, &st);
        out.sigma_s = 0.01;
        const bool meas = b.lidar_measured(frame(out));
        std::printf("  measured (b) one q = 0.3: staged %d, q_sum_staged %.3f, BeamRays q_sum %.3f -> lidar_measured %d\n", st.staged,
                    st.q_sum_staged, out.q_sum, int(meas));
        CHECK(st.staged == 9 and std::abs(st.q_sum_staged - 0.3f) < 1e-6f and std::abs(out.q_sum - 0.3) < 1e-6);
        CHECK(meas);
    }
    {   // (c) a 0.45 m bed band, the Helios 1.1 m above the floor 2.5 m from the box centre; returns on the floor 0.6–0.9 m
        // beyond the centre (outside the 0.5 m cylinder): each ray passes through the band inside the cylinder.
        const Box low{0.5f, 0.0f, 0.45f};
        const Eigen::Vector3f oh{1.5f, 0.f, 1.1f};
        std::vector<Eigen::Vector3f> floor;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 5; ++j)
                floor.emplace_back(4.6f + 0.1f * static_cast<float>(i), -0.2f + 0.1f * static_cast<float>(j), 0.0f);
        rc::ai::beam::BeamRays out;
        BeamStageStats st;
        const auto lab = [&](const Eigen::Vector3f&) -> Lab { return Lab{{eps + (1 - 2 * eps) * 0.05, 0.05f}}; };
        select_rays_beam(floor, oh, kC, low, kW, 6.0, lab, out, &st);
        out.sigma_s = 0.01;
        const bool meas = b.lidar_measured(frame(out));
        std::printf("  measured (c) bed geometry: endpoint-in-box %d (old kappa %.4f), staged %d, q_sum_staged %.3f -> lidar_measured %d\n",
                    st.in_box, st.kappa, st.staged, st.q_sum_staged, int(meas));
        CHECK(st.in_box == 0 and st.kappa == 0.f);
        CHECK(st.staged == 20 and std::abs(st.q_sum_staged - 1.0f) < 1e-5f and std::abs(out.q_sum - 1.0) < 1e-5);   // out.q_sum is the double sum, the stats the float
        CHECK(meas);
    }
    {   // an empty staged set is never measured, whatever q_sum says
        rc::ai::beam::BeamRays e;
        e.q_sum = 1.0;
        CHECK(not b.lidar_measured(frame(e)));
    }
}

int main()
{
    ray_through_box_staged();
    short_return_staged();
    no_return_staged_at_rho_max();
    no_label_not_staged();
    kappa_unfloored();
    measured_needs_camera_attribution();
    std::printf(bad == 0 ? "beam_staging_test: PASS\n" : "beam_staging_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
