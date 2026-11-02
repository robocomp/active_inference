// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/multi_scan_fit_test.cpp src/multi_scan_fit.cpp src/reloc_search.cpp src/wall_segmenter.cpp -pthread -o /tmp/multi_scan_fit_test && /tmp/multi_scan_fit_test
#include "multi_scan_fit.h"
#include <chrono>
#include <cmath>
#include <print>
#include <random>
using namespace rc::rivals;
using V2 = Eigen::Vector2f; using V3 = Eigen::Vector3f;
static int failures = 0;
#define CHECK(c) do { if (not (c)) { ++failures; std::println("FAIL line {}: {}", __LINE__, #c); } } while (0)

static std::vector<V2> L_room() { return {{0, 0}, {6, 0}, {6, 3}, {3, 3}, {3, 6}, {0, 6}}; }
static float ray(const std::vector<V2>& poly, const V2& o, const V2& d)
{
    float best = 1e9f;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const V2 a = poly[j], e = poly[i] - a;
        const float den = d.x() * e.y() - d.y() * e.x();
        if (std::abs(den) < 1e-9f) continue;
        const V2 ao = a - o;
        const float t = (ao.x() * e.y() - ao.y() * e.x()) / den, u = (ao.x() * d.y() - ao.y() * d.x()) / den;
        if (t > 0 and u >= 0 and u <= 1) best = std::min(best, t);
    }
    return best;
}
// A 100 degree forward-looking scan from `pose` (room frame), points in the robot frame.
static std::vector<V2> scan(const std::vector<V2>& poly, const V3& pose, std::mt19937& rng, float fov_deg)
{
    std::normal_distribution<float> n(0.f, 0.02f);
    std::vector<V2> out;
    const int N = 120;
    for (int k = 0; k < N; ++k)
    {
        const float a = (-0.5f + k / float(N - 1)) * fov_deg * float(M_PI) / 180.f;
        const V2 d(std::cos(pose.z() + a), std::sin(pose.z() + a));
        const float r = ray(poly, pose.head<2>(), d) + n(rng);
        out.emplace_back(r * std::cos(a), r * std::sin(a));
    }
    return out;
}
static V3 inv_compose(const V3& a, const V3& b)   // pose of b in a's frame
{
    const float c = std::cos(a.z()), s = std::sin(a.z());
    const V2 d = b.head<2>() - a.head<2>();
    return {c * d.x() + s * d.y(), -s * d.x() + c * d.y(), std::remainder(b.z() - a.z(), 2.f * float(M_PI))};
}

int main()
{
    const auto poly = L_room();
    rc::reloc::Params p;
    rc::reloc::LikelihoodRaster K; K.build(poly, p);
    std::mt19937 rng(7);
    // multi_beats_single: 15 scans along a short path (0.6 m, turning 20 deg); truth is (0.15 m, 3 deg) off the prior.
    std::vector<V3> path;
    for (int i = 0; i < 15; ++i) path.push_back({1.2f + 0.04f * i, 1.0f + 0.03f * i, 0.5f + 0.025f * i});
    const V3 truth = path.back();
    const V3 prior(truth.x() + 0.15f, truth.y(), truth.z() + 3.f * float(M_PI) / 180.f);
    Eigen::Matrix3f Sp = Eigen::Matrix3f::Zero();
    Sp.diagonal() << 0.3f * 0.3f, 0.3f * 0.3f, std::pow(6.f * float(M_PI) / 180.f, 2.f);
    std::vector<std::pair<V3, std::vector<V2>>> scans;
    for (const auto& q : path) scans.push_back({inv_compose(truth, q), scan(poly, q, rng, 90.f)});
    auto err = [&](const rc::reloc::Mode& m)
    { return std::pair{(m.pose.head<2>() - truth.head<2>()).norm(), std::abs(std::remainder(m.pose.z() - truth.z(), 2.f * float(M_PI))) * 180.f / float(M_PI)}; };
    const auto t0 = std::chrono::steady_clock::now();
    const auto multi = multi_scan_fit(K, poly, scans, prior, Sp, p);
    std::println("multi_scan_fit: {} ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    const auto single = multi_scan_fit(K, poly, {scans.back()}, prior, Sp, p);
    const auto [em, am] = err(multi); const auto [es, as_] = err(single);
    std::println("multi  err {:.4f} m {:.3f} deg  sd {:.4f} m", em, am, std::sqrt(multi.cov(0, 0)));
    std::println("single err {:.4f} m {:.3f} deg  sd {:.4f} m", es, as_, std::sqrt(single.cov(0, 0)));
    CHECK(em <= 0.02f and am <= 0.5f);
    CHECK(es > em);
    // parked ring: 15 fresh-noise scans of ONE view must not be tighter than a single scan; the moving ring is tighter.
    {
        std::vector<std::pair<V3, std::vector<V2>>> parked;
        for (int i = 0; i < 15; ++i) parked.push_back({V3::Zero(), scan(poly, truth, rng, 90.f)});
        const auto mp = multi_scan_fit(K, poly, parked, prior, Sp, p);
        const auto sp = multi_scan_fit(K, poly, {parked.back()}, prior, Sp, p);
        std::println("parked sd {:.5f} m, single sd {:.5f} m, moving sd {:.5f} m", std::sqrt(mp.cov(0, 0)), std::sqrt(sp.cov(0, 0)), std::sqrt(multi.cov(0, 0)));
        CHECK(std::sqrt(mp.cov(0, 0)) >= 0.99f * std::sqrt(sp.cov(0, 0)));
        CHECK(std::sqrt(multi.cov(0, 0)) < std::sqrt(single.cov(0, 0)));
    }
    // degenerate inputs return the prior
    const auto none = multi_scan_fit(K, poly, {}, prior, Sp, p);
    CHECK((none.pose - prior).norm() < 1e-6f);
    std::println("{}", failures ? "FAILED" : "multi_scan_fit_test: PASS");
    return failures ? 1 : 0;
}
