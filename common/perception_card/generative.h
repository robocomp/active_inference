// generative.h -- the SAMPLER of the card's generative block: draws concrete instances of a concept from its prior
// (sizes, context, pose) and rejects the physically impossible ones (a part inside a wall or an obstacle).
// Header-only; Eigen + std. The later tasks render each Sample through the real sensor models.
//
// Conventions (see card.h): local frame rotated by `yaw` -- local +x = (cos yaw, sin yaw), local +y = (-sin yaw, cos yaw)
// (rc::nbv::Obstacle's axis_x/axis_y, so an Obstacle's yaw is the sample's yaw, w along local x, h along local y).
#pragma once

#include "perception_card/evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace rc::card
{
struct Placed { std::vector<rc::nbv::Obstacle> boxes; };   // room-frame oriented boxes, z0..z1 set

struct Sample
{
    std::map<std::string, float> state;
    float x = 0, y = 0, yaw = 0;
    Placed parts;
    float weight = 1;
};

struct Context
{
    std::optional<Eigen::Vector2f> at;            // fixed position (yaw drawn uniform)
    std::optional<std::string>     force_context; // "free" | "against_wall"
};

struct SampleSet
{
    std::vector<Sample> samples;
    int drawn = 0, rejected = 0;
    std::uint64_t seed = 0;
    std::vector<std::string> notes;
};

namespace detail
{
inline float eval_bind(const Bind& b, const std::map<std::string, float>& st)
{
    if (b.var.empty()) return b.b;
    const auto it = st.find(b.var);
    return b.a * (it == st.end() ? 0.0f : it->second) + b.b;
}

// ---- truncated normal by inverse CDF (bounded time, no rejection loop) ----
inline double phi_cdf(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }
inline double phi_inv(double p)   // bisection on the CDF; p in (0,1)
{
    double lo = -12.0, hi = 12.0;
    for (int i = 0; i < 80; ++i)
    {
        const double mid = 0.5 * (lo + hi);
        (phi_cdf(mid) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}
// N(mean, sigma) truncated to [lo, hi] (either may be +-inf).
inline double trunc_normal(double mean, double sigma, double lo, double hi, std::mt19937_64& rng)
{
    if (not (sigma > 0.0)) return std::clamp(mean, lo, hi);
    double a = (lo - mean) / sigma, b = (hi - mean) / sigma;
    const bool flip = a > 0.0;          // work in the lower tail, where erfc keeps its precision
    if (flip) { const double na = -b, nb = -a; a = na; b = nb; }
    const double pa = phi_cdf(a), pb = phi_cdf(b);
    double z;
    if (not (pb - pa > 1e-14))          // the whole interval is >~7 sigma out in a tail: uniform over it
    {
        const double ca = std::max(a, -40.0), cb = std::min(b, 40.0);
        z = ca + std::uniform_real_distribution<double>(0.0, 1.0)(rng) * (cb - ca);
    }
    else
    {
        const double u = pa + std::uniform_real_distribution<double>(0.0, 1.0)(rng) * (pb - pa);
        z = std::clamp(phi_inv(std::clamp(u, 1e-300, 1.0 - 1e-16)), a, b);
    }
    if (flip) z = -z;
    return std::clamp(mean + sigma * z, lo, hi);
}

// ---- oriented-box overlap: 2-D SAT on the footprints and z-span overlap ----
inline bool footprints_overlap(const rc::nbv::Obstacle& A, const rc::nbv::Obstacle& B)
{
    const auto ax = [](const rc::nbv::Obstacle& o) { return Eigen::Vector2f(std::cos(o.yaw), std::sin(o.yaw)); };
    const auto ay = [](const rc::nbv::Obstacle& o) { return Eigen::Vector2f(-std::sin(o.yaw), std::cos(o.yaw)); };
    const auto half = [&](const rc::nbv::Obstacle& o, const Eigen::Vector2f& u)   // support function of the rectangle
    { return 0.5f * o.w * std::abs(u.dot(ax(o))) + 0.5f * o.h * std::abs(u.dot(ay(o))); };
    const Eigen::Vector2f axes[4] = {ax(A), ay(A), ax(B), ay(B)};
    const Eigen::Vector2f d(B.cx - A.cx, B.cy - A.cy);
    for (const auto& u : axes)
        if (std::abs(d.dot(u)) >= half(A, u) + half(B, u) - 1e-6f) return false;   // separated (touching is not overlap)
    return true;
}
}  // namespace detail

// Overlap of two boxes in 2.5-D. z1 <= z0 means "height unknown" = infinitely tall (viewpoint_score.h), and then the box
// spans [z0, inf).
inline bool boxes_overlap(const rc::nbv::Obstacle& A, const rc::nbv::Obstacle& B)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float a1 = A.has_height() ? A.z1 : inf, b1 = B.has_height() ? B.z1 : inf;
    if (not (A.z0 < b1 and B.z0 < a1)) return false;
    return detail::footprints_overlap(A, B);
}

// The shape program placed in the room: x, y, yaw is the object's local origin (the footprint centre of the template).
inline Placed instantiate(const Generative& g, const std::map<std::string, float>& state, float x, float y, float yaw)
{
    Placed out;
    const float c = std::cos(yaw), s = std::sin(yaw);
    const auto put = [&](float lx, float ly, float w, float d, float z0, float z1)
    {
        out.boxes.push_back(rc::nbv::Obstacle{x + lx * c - ly * s, y + lx * s + ly * c, w, d, yaw, z0, z1});
    };
    for (const auto& pr : g.shape)
    {
        const auto D = [&](const char* k) { const auto it = pr.dims.find(k); return it == pr.dims.end() ? 0.0f : detail::eval_bind(it->second, state); };
        switch (pr.type)
        {
            case Prim::Slab: case Prim::Box: put(D("x"), D("y"), D("w"), D("d"), D("z"), D("z") + D("h")); break;
            case Prim::Panel:                put(D("x"), D("y"), D("w"), D("t"), D("z"), D("z") + D("h")); break;
            case Prim::Cylinder:             put(D("x"), D("y"), 2 * D("r"), 2 * D("r"), D("z"), D("z") + D("h")); break;
            case Prim::LegSet:
            {
                const float half = D("half"), cx = 0.5f * D("fw") - half, cy = 0.5f * D("fd") - half;
                const int n = std::clamp(static_cast<int>(std::lround(D("n"))), 0, 4);
                constexpr float sx[4] = {-1, 1, -1, 1}, sy[4] = {-1, -1, 1, 1};
                for (int i = 0; i < n; ++i) put(sx[i] * cx, sy[i] * cy, 2 * half, 2 * half, D("z0"), D("z1"));
                break;
            }
        }
    }
    return out;
}

inline SampleSet sample_concept(const Card& card, const World& world, const Context& ctx, int n, std::uint64_t seed,
                                int max_draws = -1)   // -1 = 50 * n (a default argument cannot name n)
{
    SampleSet out;
    out.seed = seed;
    if (max_draws < 0) max_draws = 50 * n;
    const auto note = [&](std::string s)
    { if (std::ranges::find(out.notes, s) == out.notes.end()) out.notes.push_back(std::move(s)); };
    if (not card.generative) { note("card has no generative block"); return out; }
    const Generative& g = *card.generative;
    std::mt19937_64 rng(seed);
    const auto U = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    constexpr double inf = std::numeric_limits<double>::infinity();

    const auto& P = world.polygon;
    const bool has_poly = P.size() >= 3;
    // contexts: usable ones with their weights
    std::vector<std::pair<std::string, float>> ctxs;
    if (ctx.at) ctxs = {{"at", 1.0f}};
    else if (ctx.force_context) ctxs = {{*ctx.force_context, 1.0f}};
    else for (const auto& [k, w] : g.placement.contexts) if (w > 0.0f) ctxs.emplace_back(k, w);
    std::erase_if(ctxs, [&](const auto& kv)
    {
        if (kv.first == "at" or kv.first == "free") return false;
        if (kv.first != "against_wall") { note("unknown context '" + kv.first + "' skipped"); return true; }
        if (not has_poly) { note("against_wall skipped: no room polygon"); return true; }
        return false;
    });
    if (ctxs.empty()) { note("no usable placement context"); return out; }
    if (not has_poly) note("no room polygon: free placement in a 6x6 m square around the origin (robot pose unknown to the sampler)");
    float wsum = 0; for (const auto& kv : ctxs) wsum += kv.second;

    // polygon geometry for the wall placement and the free-space bounding box
    float minx = -3, maxx = 3, miny = -3, maxy = 3;
    std::vector<float> elen; float ltot = 0, area2 = 0;
    if (has_poly)
    {
        minx = miny = inf; maxx = maxy = -inf;
        for (const auto& p : P) { minx = std::min(minx, p.x()); maxx = std::max(maxx, p.x()); miny = std::min(miny, p.y()); maxy = std::max(maxy, p.y()); }
        for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
        {
            elen.push_back((P[i] - P[j]).norm()); ltot += elen.back();
            area2 += P[j].x() * P[i].y() - P[i].x() * P[j].y();
        }
    }
    const auto inside = [&](float px, float py)
    {
        bool in = false;
        for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
            if ((P[i].y() > py) != (P[j].y() > py) and px < (P[j].x() - P[i].x()) * (py - P[i].y()) / (P[j].y() - P[i].y()) + P[i].x())
                in = not in;
        return in;
    };

    auto obstacles = rc::nbv::wall_obstacles(P, detail::wall_thickness_for(card), world.gaps);
    for (auto& o : obstacles) { o.z0 = 0.0f; o.z1 = 0.0f; }   // a wall is floor-to-ceiling: z1 <= z0 = unbounded
    for (const auto& [name, o] : world.obstacles) obstacles.push_back(o);

    // the offset from the local origin to the backrest's OUTER face, measured on the template itself
    const auto back_offset = [&](const std::map<std::string, float>& st)
    {
        const auto p = instantiate(g, st, 0, 0, 0);
        float m = 0; for (const auto& b : p.boxes) m = std::min(m, b.cy - 0.5f * b.h);
        return -m;
    };

    while (static_cast<int>(out.samples.size()) < n and out.drawn < max_draws)
    {
        ++out.drawn;
        Sample sm;
        for (const auto& v : g.vars)
        {
            double x;
            if (v.log_normal and v.mean > 0.0f and v.sigma > 0.0f)   // log x ~ N(log mean, (sigma/mean)^2), truncated in log space
                x = std::exp(detail::trunc_normal(std::log(v.mean), v.sigma / v.mean, std::log(std::max(v.lo, 1e-9f)), std::log(std::max(v.hi, 1e-9f)), rng));
            else
                x = detail::trunc_normal(v.mean, v.sigma, v.lo, v.hi, rng);
            sm.state[v.var] = static_cast<float>(x);
        }
        double r = U(0, wsum); std::string chosen = ctxs.back().first;
        for (const auto& kv : ctxs) { if (r < kv.second) { chosen = kv.first; break; } r -= kv.second; }

        if (chosen == "against_wall")
        {
            double r2 = U(0, ltot); std::size_t e = elen.size() - 1;
            for (std::size_t k = 0; k < elen.size(); ++k) { if (r2 < elen[k]) { e = k; break; } r2 -= elen[k]; }
            const Eigen::Vector2f A = P[(e + P.size() - 1) % P.size()], B = P[e];   // elen[e] is the edge (e-1 -> e)
            const Eigen::Vector2f u = (B - A) / std::max(1e-6f, (B - A).norm());
            const Eigen::Vector2f nin = (area2 > 0 ? 1.0f : -1.0f) * Eigen::Vector2f(-u.y(), u.x());   // inward normal
            const float gap = static_cast<float>(detail::trunc_normal(g.placement.wall_gap_mean, g.placement.wall_gap_sigma, 0.0, inf, rng));
            const Eigen::Vector2f on = A + u * static_cast<float>(U(0, elen[e]));
            const Eigen::Vector2f c = on + nin * (back_offset(sm.state) + gap);
            sm.x = c.x(); sm.y = c.y();
            sm.yaw = std::atan2(-nin.x(), nin.y());       // local +y = inward normal, so the backrest (-y) faces the wall
        }
        else if (chosen == "at")
        {
            sm.x = ctx.at->x(); sm.y = ctx.at->y(); sm.yaw = static_cast<float>(U(-std::numbers::pi, std::numbers::pi));
        }
        else   // free
        {
            sm.x = static_cast<float>(U(minx, maxx)); sm.y = static_cast<float>(U(miny, maxy));
            sm.yaw = static_cast<float>(U(-std::numbers::pi, std::numbers::pi));
            if (has_poly and not inside(sm.x, sm.y)) { ++out.rejected; continue; }
        }
        sm.parts = instantiate(g, sm.state, sm.x, sm.y, sm.yaw);
        const bool hit = std::ranges::any_of(sm.parts.boxes, [&](const auto& b)
        { return std::ranges::any_of(obstacles, [&](const auto& o) { return boxes_overlap(b, o); }); });
        if (hit) { ++out.rejected; continue; }
        out.samples.push_back(std::move(sm));
    }
    if (static_cast<int>(out.samples.size()) < n)
        note("drew " + std::to_string(out.drawn) + ", rejected " + std::to_string(out.rejected) + ": only " +
             std::to_string(out.samples.size()) + " of " + std::to_string(n) + " samples fit (stopped at max_draws)");
    return out;
}
}  // namespace rc::card
