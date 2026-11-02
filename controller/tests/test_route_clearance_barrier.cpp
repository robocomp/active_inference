// The route optimiser must keep routes in the MIDDLE of free space, using the LIVE field (2026-11-02).
// Legacy clearance = one-sided deficit (zero beyond body+margin): curvature + anchors cut the 90 deg corner and nothing
// pushes off the walls. The barrier (sigma/gap)^2 is active everywhere. World: an L corridor 1.2 m wide (arm A along x,
// arm B along y), the STATIC field knows the walls only; the LIVE field also has a box (r = 0.15 m) beside the medial line of
// arm A that the planner grid has not seen. Route: start -> corner -> goal anchored on the medial line.
//   (1) legacy: min live clearance after < before / passes near the box; (2) barrier: min clearance after >= before,
//   mean |offset from the medial line| smaller, and the live-only box is cleared.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_route_clearance_barrier.cpp src/route_optimizer.cpp -o /tmp/test_route_clearance_barrier && /tmp/test_route_clearance_barrier
#include "route_optimizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

using V = Eigen::Vector2f;
static constexpr float W = 1.2f, LA = 5.f, BODY = 0.25f;
static const V kBox{2.2f, 1.0f};
static constexpr float kBoxR = 0.15f;

static bool free_cell(float x, float y)
{
    const bool a = x >= 0.f and x <= LA and y >= 0.f and y <= W;
    const bool b = x >= LA - W and x <= LA and y >= 0.f and y <= LA;
    return a or b;
}
// exact EDT of the static walls by brute force on a 2 cm lattice (bilinear), built once
struct Edt
{
    static constexpr float res = 0.02f, x0 = -1.f, y0 = -1.f;
    int n = 0;
    std::vector<float> d;
    Edt()
    {
        n = static_cast<int>(7.f / res);
        std::vector<V> occ;
        d.assign(static_cast<std::size_t>(n) * n, 0.f);
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
            if (not free_cell(x0 + i * res, y0 + j * res)) occ.push_back({x0 + i * res, y0 + j * res});
        // only boundary occupied cells matter, but keep simple: search a window
        std::vector<char> o(static_cast<std::size_t>(n) * n, 1);
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) o[static_cast<std::size_t>(j) * n + i] = not free_cell(x0 + i * res, y0 + j * res);
        const int R = 40;
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        {
            if (o[static_cast<std::size_t>(j) * n + i]) continue;
            float best = 1e6f;
            for (int dj = -R; dj <= R; ++dj) for (int di = -R; di <= R; ++di)
            {
                const int ii = i + di, jj = j + dj;
                if (ii < 0 or jj < 0 or ii >= n or jj >= n or not o[static_cast<std::size_t>(jj) * n + ii]) continue;
                best = std::min(best, std::hypot(static_cast<float>(di), static_cast<float>(dj)) * res);
            }
            d[static_cast<std::size_t>(j) * n + i] = best;   // >= R*res (0.8 m) reads as 0.8+: far enough for this test
        }
        for (std::size_t k = 0; k < d.size(); ++k) if (d[k] == 1e6f) d[k] = R * res;
    }
    float at(const V &p) const
    {
        const float gx = (p.x() - x0) / res, gy = (p.y() - y0) / res;
        const int i = std::clamp(static_cast<int>(gx), 0, n - 2), j = std::clamp(static_cast<int>(gy), 0, n - 2);
        const float fx = gx - i, fy = gy - j;
        const auto v = [&](int a, int b) { return d[static_cast<std::size_t>(b) * n + a]; };
        return (1 - fx) * (1 - fy) * v(i, j) + fx * (1 - fy) * v(i + 1, j) + (1 - fx) * fy * v(i, j + 1) + fx * fy * v(i + 1, j + 1);
    }
};
static const Edt kEdt;
static float live_d(const V &p) { return std::min(kEdt.at(p), std::max(0.f, (p - kBox).norm() - kBoxR)); }
static V grad_of(float (*f)(const V &), const V &p)
{
    const float e = 0.03f;
    V g(f({p.x() + e, p.y()}) - f({p.x() - e, p.y()}), f({p.x(), p.y() + e}) - f({p.x(), p.y() - e}));
    return g.norm() > 1e-6f ? V(g / g.norm()) : V(V::Zero());
}
static float static_d(const V &p) { return kEdt.at(p); }

// uniform cubic B-spline sampled densely (clamped by tripled ends, as the optimiser pads)
static std::vector<V> curve(const std::vector<V> &c)
{
    std::vector<V> pad{c.front(), c.front()};
    pad.insert(pad.end(), c.begin(), c.end());
    pad.push_back(c.back()); pad.push_back(c.back());
    std::vector<V> out;
    for (std::size_t s = 0; s + 3 < pad.size(); ++s)
        for (int q = 0; q < 10; ++q)
        {
            const float t = q / 10.f, t2 = t * t, t3 = t2 * t;
            const float b0 = (1 - 3 * t + 3 * t2 - t3) / 6, b1 = (4 - 6 * t2 + 3 * t3) / 6, b2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6, b3 = t3 / 6;
            out.push_back(b0 * pad[s] + b1 * pad[s + 1] + b2 * pad[s + 2] + b3 * pad[s + 3]);
        }
    return out;
}
// distance to the medial polyline (0.6,0.6) -> (4.4,0.6) -> (4.4,4.4)
static float medial_offset(const V &p)
{
    const auto seg = [&](V a, V b) { const V ab = b - a; const float t = std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f); return (p - (a + t * ab)).norm(); };
    return std::min(seg({0.6f, 0.6f}, {4.4f, 0.6f}), seg({4.4f, 0.6f}, {4.4f, 4.4f}));
}

struct Result { float min_before, min_after, mean_off, box_min, wall_before, wall_after; };
static Result run(bool barrier, bool use_live)
{
    const std::vector<V> way{{0.6f, 0.6f}, {4.4f, 0.6f}, {4.4f, 4.4f}};
    std::vector<V> ctrl;
    const float h = 0.40f;
    std::vector<float> aS;
    float s = 0.f;
    for (std::size_t k = 0; k + 1 < way.size(); ++k)
    {
        const V d = way[k + 1] - way[k];
        const int n = static_cast<int>(std::round(d.norm() / h));
        for (int i = 0; i < n; ++i) ctrl.push_back(way[k] + d * (static_cast<float>(i) / n));
        s += d.norm();
        if (k == 0) aS.push_back(0.f);
    }
    ctrl.push_back(way.back());
    rc::RouteOptimizerConfig cfg;
    float (*f)(const V &) = use_live ? live_d : static_d;
    cfg.distance = [f](const V &p) { return f(p); };
    cfg.distance_gradient = [f](const V &p) { return grad_of(f, p); };
    cfg.anchors = {way[1]};
    cfg.anchor_s = {3.8f};
    cfg.support_radius = [](float, const V &) { return BODY; };
    cfg.safety_margin = 0.10f;
    cfg.clearance_barrier = barrier;
    cfg.verbose = false;
    cfg.iterations = 60;
    Result r{};
    const auto mc = [&](const std::vector<V> &c) { float m = 1e6f; for (const auto &p : curve(c)) m = std::min(m, live_d(p)); return m; };
    const auto mw = [&](const std::vector<V> &c) { float m = 1e6f; for (const auto &p : curve(c)) m = std::min(m, static_d(p)); return m; };
    r.min_before = mc(ctrl); r.wall_before = mw(ctrl);
    rc::optimize_route(ctrl, cfg);
    r.min_after = mc(ctrl); r.wall_after = mw(ctrl);
    float so = 0.f; int n = 0; float bm = 1e6f;
    for (const auto &p : curve(ctrl)) { so += medial_offset(p); ++n; bm = std::min(bm, (p - kBox).norm() - kBoxR); }
    r.mean_off = so / n; r.box_min = bm;
    return r;
}

int main()
{
    int fails = 0;
    const auto chk = [&](bool ok, const char *m) { std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", m); fails += not ok; };
    const Result leg = run(false, false), leg_live = run(false, true), bar = run(true, true), bar_static = run(true, false);
    std::printf("legacy  static-only: min %.3f -> %.3f  wall %.3f -> %.3f  mean|off| %.3f  box gap %.3f\n", leg.min_before, leg.min_after, leg.wall_before, leg.wall_after, leg.mean_off, leg.box_min);
    std::printf("legacy  live       : min %.3f -> %.3f  wall %.3f -> %.3f  mean|off| %.3f  box gap %.3f\n", leg_live.min_before, leg_live.min_after, leg_live.wall_before, leg_live.wall_after, leg_live.mean_off, leg_live.box_min);
    std::printf("barrier static-only: min %.3f -> %.3f  wall %.3f -> %.3f  mean|off| %.3f  box gap %.3f\n", bar_static.min_before, bar_static.min_after, bar_static.wall_before, bar_static.wall_after, bar_static.mean_off, bar_static.box_min);
    std::printf("barrier live       : min %.3f -> %.3f  wall %.3f -> %.3f  mean|off| %.3f  box gap %.3f\n", bar.min_before, bar.min_after, bar.wall_before, bar.wall_after, bar.mean_off, bar.box_min);
    chk(bar.min_after >= bar.min_before - 1e-3f, "barrier+live: min clearance after >= before");
    chk(bar.box_min > BODY, "barrier+live: the live-only box is cleared by more than the body radius");
    chk(bar_static.wall_after >= bar_static.wall_before - 1e-3f, "barrier: the 90 deg corner is not cut (wall clearance after >= before)");
    chk(bar_static.wall_after >= leg.wall_after, "barrier keeps at least the legacy wall clearance");
    chk(bar_static.mean_off <= leg.mean_off, "barrier stays closer to the medial line than legacy");
    chk(bar.min_after > leg_live.min_after + 0.02f, "barrier keeps a larger minimum clearance than legacy on the live field");
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
