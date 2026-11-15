// lidar_weights_test — per-ray membership weights on the shared LiDAR ray factor, and the weighted LiDAR selection
// (spec 2026-11-15 lidar-measures G2). Uses CountableBelief as the model for the factor, a synthetic sweep for the selection.
#include <cmath>
#include <cstdio>
#include <cstring>
#include "countable_belief.h"
#include "lidar_select/lidar_select.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

using M6 = Eigen::Matrix<float, 6, 6>;
using V6 = Eigen::Matrix<float, 6, 1>;

static rc::ai::LidarRays make_rays()
{
    rc::ai::LidarRays r;
    r.origin = Eigen::Vector3f(0.30f, -2.255f, 1.1075f);
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        r.endpoints.emplace_back(0.045f + 0.51f * (i + 0.5f) / 12.0f, -0.51f, 0.62f * (j + 0.5f) / 10.0f);
    r.precision = 2500.0f; r.robust_c_m = 0.05f;
    return r;
}
static void fold(const rc::ai::LidarRays& r, M6& Id, V6& bd)
{
    CountableBeliefParams p;
    CountableBeliefState s{0.35f, -0.295f, 0.65f, 0.51f, 0.51f, 0.0f};   // off-truth ⇒ non-zero residuals
    CountableBelief b(s, p);
    Id.setZero(); bd.setZero();
    rc::ai::accumulate_lidar_rays<6>(b, s, r, Id, bd);
}

static void weights_empty_equals_ones()
{
    auto a = make_rays(); auto b = make_rays();
    b.weights.assign(b.endpoints.size(), 1.0f);
    M6 Ia, Ib; V6 da, db;
    fold(a, Ia, da); fold(b, Ib, db);
    CHECK(Ia.norm() > 0.0f);
    CHECK(std::memcmp(Ia.data(), Ib.data(), sizeof(float) * 36) == 0);
    CHECK(std::memcmp(da.data(), db.data(), sizeof(float) * 6) == 0);
}

static void weights_half_scales_information()
{
    auto a = make_rays(); auto b = make_rays();
    b.weights.assign(b.endpoints.size(), 0.5f);
    M6 Ia, Ib; V6 da, db;
    fold(a, Ia, da); fold(b, Ib, db);
    CHECK((Ib - 0.5f * Ia).norm() <= 1e-6f * Ia.norm());
    CHECK((db - 0.5f * da).norm() <= 1e-6f * da.norm());
    CHECK(da.norm() > 0.0f);
}

static std::vector<Eigen::Vector3f> sweep_inside(int n, float z, float a0, float a1)   // bearings spread over [a0, a1]
{
    std::vector<Eigen::Vector3f> v;
    for (int i = 0; i < n; ++i)
    {
        const float a = a0 + (a1 - a0) * static_cast<float>(i) / static_cast<float>(n);
        v.emplace_back(0.5f * std::cos(a), 0.5f * std::sin(a), z);
    }
    return v;
}

static void weighted_select_ones_matches_select_into()
{
    auto sweep = sweep_inside(90, 0.5f, 0.0f, 3.0f);
    for (int i = 0; i < 20; ++i) sweep.emplace_back(3.0f + 0.1f * i, 2.0f, 0.5f);   // outside the radius
    sweep.emplace_back(0.2f, 0.1f, 5.0f);                                           // above the band
    const lidar_select::Box box{1.0f, 0.0f, 2.0f};
    const lidar_select::Weights w;
    const Eigen::Vector3f o(0, -3, 1);
    rc::ai::LidarRays a, b;
    float ca_a = 0, ca_b = 0, msum = 0;
    const int na = lidar_select::select_into(sweep, o, Eigen::Vector2f(0, 0), box, 2500.0f, w, a, &ca_a);
    const int nb = lidar_select::select_into_weighted(sweep, o, Eigen::Vector2f(0, 0), box, 2500.0f, w,
                       [](const Eigen::Vector3f&) { return 1.0f; }, b, &ca_b, &msum);
    CHECK(na == 90 and nb == 90);
    CHECK(a.endpoints.size() == b.endpoints.size());
    CHECK(a.endpoints == b.endpoints);
    CHECK(std::memcmp(&a.precision, &b.precision, sizeof(float)) == 0);
    CHECK(a.weights.empty());
    CHECK(b.weights.size() == 90);
    for (float x : b.weights) CHECK(x == 1.0f);
    CHECK(msum == 90.0f);
    CHECK(ca_a == ca_b);
}

static void weighted_select_coverage_counts_msum()
{
    auto sweep = sweep_inside(30, 0.5f, 0.0f, 1.0f);              // m = 1 (z < 1)
    const auto rest = sweep_inside(70, 1.5f, 2.0f, 6.0f);         // m = 0 (z > 1)
    sweep.insert(sweep.end(), rest.begin(), rest.end());
    const lidar_select::Box box{1.0f, 0.0f, 2.0f};
    lidar_select::Weights w; w.coverage_n0 = 60.0f;
    rc::ai::LidarRays out; float ca = 0, msum = 0;
    const int n = lidar_select::select_into_weighted(sweep, Eigen::Vector3f(0, -3, 1), Eigen::Vector2f(0, 0), box, 2500.0f, w,
                      [](const Eigen::Vector3f& p) { return p.z() < 1.0f ? 1.0f : 0.0f; }, out, &ca, &msum);
    double c = 0, s = 0;
    for (int i = 0; i < 30; ++i) { const double a = 1.0 * i / 30.0; c += std::cos(a); s += std::sin(a); }
    const double R = std::hypot(c, s) / 30.0, ca_ref = 1.0 - R;
    CHECK(n == 30 and out.endpoints.size() == 30 and out.weights.size() == 30);
    CHECK(msum == 30.0f);
    CHECK(std::abs(ca - ca_ref) < 1e-5);
    CHECK(std::abs(out.precision - 2500.0 * (30.0 / 60.0) * ca_ref) < 1e-3 * out.precision);
}

static void weighted_select_all_zero()
{
    const auto sweep = sweep_inside(100, 0.5f, 0.0f, 3.0f);
    const lidar_select::Box box{1.0f, 0.0f, 2.0f};
    rc::ai::LidarRays out; float ca = -1, msum = -1;
    out.precision = 99.0f;
    const int n = lidar_select::select_into_weighted(sweep, Eigen::Vector3f(0, -3, 1), Eigen::Vector2f(0, 0), box, 2500.0f,
                      lidar_select::Weights{}, [](const Eigen::Vector3f&) { return 0.0f; }, out, &ca, &msum);
    CHECK(n == 0 and msum == 0.0f and out.precision == 0.0f and out.endpoints.empty() and out.weights.empty());
    CHECK(std::isfinite(ca));
}

int main()
{
    weights_empty_equals_ones();
    weights_half_scales_information();
    weighted_select_ones_matches_select_into();
    weighted_select_coverage_counts_msum();
    weighted_select_all_zero();
    std::printf("lidar_weights_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
