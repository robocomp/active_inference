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
#include <ranges>
#include <utility>
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
    int drawn = 0;
    int rejected = 0;          // physical: a part inside a wall or an obstacle (or no room beside the wall)
    int outside_polygon = 0;   // the draw's centre (or, for a free draw, any part-box corner) fell outside the room polygon
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
// N(mean, sigma) truncated to [lo, hi] (either may be +-inf). lo >= hi returns lo (never calls clamp with lo > hi).
inline double trunc_normal(double mean, double sigma, double lo, double hi, std::mt19937_64& rng)
{
    if (not (lo < hi)) return lo;
    if (not (sigma > 0.0)) return std::clamp(mean, lo, hi);
    double a = (lo - mean) / sigma, b = (hi - mean) / sigma;
    const bool flip = a > 0.0;          // work in the lower tail, where erfc keeps its precision
    if (flip) { const double na = -b, nb = -a; a = na; b = nb; }
    const double pa = phi_cdf(a), pb = phi_cdf(b);
    const double u01 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double z;
    if (not (pb - pa > 1e-14))
    {
        // The whole interval lies > ~7 sigma out in the lower tail (b < 0 here). There the density is
        // exp(-z^2/2) = exp(-b^2/2) exp(|b| e - e^2/2)... with z = b - e, e >= 0: ~ exp(-|b| e), an exponential in the
        // distance e below the upper end b, truncated to e <= b - a. Inverse CDF of that.
        const double lam = std::max(-b, 1e-9), L = b - a;
        const double e = -std::log1p(-u01 * (1.0 - std::exp(-lam * L))) / lam;
        z = std::clamp(b - e, a, b);
    }
    else
    {
        const double u = pa + u01 * (pb - pa);
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

// The concept prior over the card's size variables: one draw of every var (truncated normal, or log-normal in log space).
// Shared by sample_concept and evaluate_mc so both draw the same way from the same rng stream.
inline std::map<std::string, float> sample_sizes(const Generative& g, std::mt19937_64& rng)
{
    std::map<std::string, float> state;
    for (const auto& v : g.vars)
    {
        double x;
        if (v.log_normal and v.mean > 0.0f and v.sigma > 0.0f)   // log x ~ N(log mean, (sigma/mean)^2), truncated in log space
        {
            const double llo = std::log(std::max(v.lo, 1e-9f)), lhi = std::log(std::max(v.hi, 1e-9f));
            x = std::exp(detail::trunc_normal(std::log(v.mean), v.sigma / v.mean, llo, lhi, rng));
            x = v.lo < v.hi ? std::clamp(x, static_cast<double>(v.lo), static_cast<double>(v.hi)) : v.lo;
        }
        else
            x = detail::trunc_normal(v.mean, v.sigma, v.lo, v.hi, rng);
        state[v.var] = static_cast<float>(x);
    }
    return state;
}

namespace detail
{
// The parts' footprint bounding box in the first part's frame (the sample's yaw), as the evaluator's Target.
inline rc::nbv::Target target_of(const Placed& placed)
{
    rc::nbv::Target tgt;
    const float inf = std::numeric_limits<float>::infinity();
    tgt.yaw = placed.boxes.front().yaw;
    const float c = std::cos(tgt.yaw), s = std::sin(tgt.yaw);
    float x0 = inf, x1 = -inf, y0 = inf, y1 = -inf, z0 = inf, z1 = -inf;
    for (const auto& b : placed.boxes)
    {
        const float bc = std::cos(b.yaw), bs = std::sin(b.yaw);
        for (const float sx : {-0.5f, 0.5f}) for (const float sy : {-0.5f, 0.5f})
        {
            const float px = b.cx + sx * b.w * bc - sy * b.h * bs, py = b.cy + sx * b.w * bs + sy * b.h * bc;
            const float lx = px * c + py * s, ly = -px * s + py * c;
            x0 = std::min(x0, lx); x1 = std::max(x1, lx); y0 = std::min(y0, ly); y1 = std::max(y1, ly);
        }
        z0 = std::min(z0, b.z0); z1 = std::max(z1, b.has_height() ? b.z1 : b.z0);
    }
    const float mx = 0.5f * (x0 + x1), my = 0.5f * (y0 + y1);
    tgt.cx = mx * c - my * s; tgt.cy = mx * s + my * c; tgt.w = x1 - x0; tgt.h = y1 - y0; tgt.z0 = z0; tgt.z1 = z1;
    return tgt;
}
}  // namespace detail

// Placement against a wall (ruling P4): the backrest's OUTER face stands `gap` from the wall's INNER face, i.e. at
// (wall thickness / 2 + gap) from the polygon edge line, inward. The thickness is the one the evaluator's wall pieces
// are built with (detail::wall_thickness_for). Edges are picked over their SOLID runs only (door gaps removed), by the
// run length the chair can actually use: a run loses the chair's half-width at each end (plus half a wall thickness
// at a polygon corner), so a chair never lands in a doorway or straddles a corner.
// max_draws < 0 means 50 * n (a default argument cannot name n).
inline SampleSet sample_concept(const Card& card, const World& world, const Context& ctx, int n, std::uint64_t seed,
                                int max_draws = -1)
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

    for (const auto& v : g.vars)
        if (v.lo > v.hi) note("var '" + v.var + "': lo > hi, drawn at lo");

    const auto& P = world.polygon;
    const bool has_poly = P.size() >= 3;
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

    const float thick = detail::wall_thickness_for(card);
    float minx = -3, maxx = 3, miny = -3, maxy = 3, area2 = 0;
    struct Run { Eigen::Vector2f A, u; float s0, s1, len; };   // a solid span [s0,s1] of the edge starting at A (direction u)
    std::vector<Run> runs;
    if (has_poly)
    {
        minx = miny = inf; maxx = maxy = -inf;
        for (const auto& p : P) { minx = std::min(minx, p.x()); maxx = std::max(maxx, p.x()); miny = std::min(miny, p.y()); maxy = std::max(maxy, p.y()); }
        for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
        {
            area2 += P[j].x() * P[i].y() - P[i].x() * P[j].y();
            const Eigen::Vector2f A = P[j], d = P[i] - P[j];
            const float len = d.norm();
            if (len < 1e-3f) continue;
            const Eigen::Vector2f u = d / len, nrm(-u.y(), u.x());
            std::vector<std::pair<float, float>> open;     // the same ownership test as wall_obstacles
            for (const auto& gp : world.gaps)
            {
                const Eigen::Vector2f r = gp.centre - A;
                if (std::abs(r.dot(nrm)) > 0.5f * thick + gp.half_w) continue;
                const float s = r.dot(u), s0 = std::max(0.0f, s - gp.half_w), s1 = std::min(len, s + gp.half_w);
                if (s1 > s0) open.emplace_back(s0, s1);
            }
            std::ranges::sort(open);
            float cur = 0;
            for (const auto& [o0, o1] : open) { if (o0 > cur) runs.push_back({A, u, cur, o0, len}); cur = std::max(cur, o1); }
            if (len > cur) runs.push_back({A, u, cur, len, len});
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

    auto obstacles = rc::nbv::wall_obstacles(P, thick, world.gaps);
    for (auto& o : obstacles) { o.z0 = 0.0f; o.z1 = 0.0f; }   // a wall is floor-to-ceiling: z1 <= z0 = unbounded
    for (const auto& [name, o] : world.obstacles) obstacles.push_back(o);

    while (static_cast<int>(out.samples.size()) < n and out.drawn < max_draws)
    {
        ++out.drawn;
        Sample sm;
        sm.state = sample_sizes(g, rng);
        double r = U(0, wsum); std::string chosen = ctxs.back().first;
        for (const auto& kv : ctxs) { if (r < kv.second) { chosen = kv.first; break; } r -= kv.second; }

        if (chosen == "against_wall")
        {
            // The template's extent in its local frame: back = distance from the origin to the backrest's outer face
            // (-y), hx = half-width along x.
            const auto tp = instantiate(g, sm.state, 0, 0, 0);
            float back = 0, hx = 0;
            for (const auto& b : tp.boxes) { back = std::max(back, -(b.cy - 0.5f * b.h)); hx = std::max(hx, std::abs(b.cx) + 0.5f * b.w); }
            std::vector<std::pair<float, float>> usable(runs.size());   // [lo, hi] along the edge, empty when hi <= lo
            double tot = 0;
            for (std::size_t k = 0; k < runs.size(); ++k)
            {
                const auto& rn = runs[k];
                const float lo = rn.s0 + hx + (rn.s0 <= 0.0f ? 0.5f * thick : 0.0f);
                const float hi = rn.s1 - hx - (rn.s1 >= rn.len ? 0.5f * thick : 0.0f);
                usable[k] = {lo, hi};
                tot += std::max(0.0f, hi - lo);
            }
            if (not (tot > 0.0)) { ++out.rejected; continue; }          // no wall long enough for this chair
            double r2 = U(0, tot); std::size_t e = 0;
            for (std::size_t k = 0; k < runs.size(); ++k)
            {
                const double l = std::max(0.0f, usable[k].second - usable[k].first);
                if (l > 0.0) e = k;
                if (r2 < l) break;
                r2 -= l;
            }
            const auto& rn = runs[e];
            const Eigen::Vector2f nin = (area2 > 0 ? 1.0f : -1.0f) * Eigen::Vector2f(-rn.u.y(), rn.u.x());   // inward normal
            const float gap = static_cast<float>(detail::trunc_normal(g.placement.wall_gap_mean, g.placement.wall_gap_sigma, 0.0, inf, rng));
            const float s = static_cast<float>(U(usable[e].first, std::max(usable[e].first, usable[e].second)));
            const Eigen::Vector2f c = rn.A + rn.u * s + nin * (0.5f * thick + gap + back);
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
        }
        if (has_poly and chosen != "at" and not inside(sm.x, sm.y)) { ++out.outside_polygon; continue; }
        sm.parts = instantiate(g, sm.state, sm.x, sm.y, sm.yaw);
        // A free draw must lie wholly in the room: ANY part-box corner outside the polygon rejects it (a centre inside
        // is not enough: a chair centred beside a door gap would straddle the wall line through the gap).
        if (has_poly and chosen == "free")
        {
            const bool all_in = std::ranges::all_of(sm.parts.boxes, [&](const auto& b)
            {
                const float bc = std::cos(b.yaw), bs = std::sin(b.yaw);
                for (const float sx : {-0.5f, 0.5f}) for (const float sy : {-0.5f, 0.5f})
                    if (not inside(b.cx + sx * b.w * bc - sy * b.h * bs, b.cy + sx * b.w * bs + sy * b.h * bc)) return false;
                return true;
            });
            if (not all_in) { ++out.outside_polygon; continue; }
        }
        const bool hit = std::ranges::any_of(sm.parts.boxes, [&](const auto& b)
        { return std::ranges::any_of(obstacles, [&](const auto& o) { return boxes_overlap(b, o); }); });
        if (hit) { ++out.rejected; continue; }
        out.samples.push_back(std::move(sm));
    }
    if (out.rejected > 0 or out.outside_polygon > 0 or static_cast<int>(out.samples.size()) < n)
        note("drew " + std::to_string(out.drawn) + ": accepted " + std::to_string(out.samples.size()) + " of " + std::to_string(n) +
             ", physical rejections " + std::to_string(out.rejected) + ", centre/corner outside polygon " + std::to_string(out.outside_polygon) +
             (static_cast<int>(out.samples.size()) < n ? " (stopped at max_draws)" : ""));
    return out;
}

// ---- Monte-Carlo P(detect) of a HYPOTHETICAL object: the evaluator averaged over the concept prior on its sizes ----
struct McResult { float p_notice_mean = 0, p_notice_sd = 0, p_recognise_mean = 0, p_recognise_sd = 0; Limit modal_limit = Limit::None; int n = 0; };

// Samples the SIZES from the card's generative prior; the POSE IS FIXED at base x, y, yaw (a caller that wants the pose drawn
// too, with physical rejection, uses sample_concept and aggregates with mc_over_samples instead), instantiates each, and evaluates with
// the sample's overall bounding box as the Hypothesis size. sd = population SD over the n samples. modal_limit = most
// frequent limit (ties to the lower enum value). n < 1 is clamped to 1. A card without a generative block yields one plain evaluate() (n = 1).
inline McResult evaluate_mc(const Card& card, const World& world, const Hypothesis& base, const Viewpoint& vp, int n, std::uint64_t seed)
{
    McResult out;
    std::vector<Result> rs;
    if (not card.generative) rs.push_back(evaluate(card, world, base, vp));
    else
    {
        std::mt19937_64 rng(seed);
        for (int i = 0; i < std::max(n, 1); ++i)
        {
            const Placed pl = instantiate(*card.generative, sample_sizes(*card.generative, rng), base.x, base.y, base.yaw);
            const auto t = detail::target_of(pl);
            Hypothesis h = base;   // the evaluator puts the target at h.x/h.y: centre it on the parts' bbox, not on the origin
            h.x = t.cx; h.y = t.cy; h.yaw = t.yaw;
            h.size = GeometryPrior{t.w, t.h, t.z1 - t.z0, t.z0, t.z1};
            rs.push_back(evaluate(card, world, h, vp));
        }
    }
    out.n = static_cast<int>(rs.size());
    const double m = out.n;
    double sn = 0, sr = 0;
    for (const auto& r : rs) { sn += r.p_notice; sr += r.p_recognise; }
    sn /= m; sr /= m;
    double vn = 0, vr = 0;
    for (const auto& r : rs) { vn += (r.p_notice - sn) * (r.p_notice - sn); vr += (r.p_recognise - sr) * (r.p_recognise - sr); }
    out.p_notice_mean = static_cast<float>(sn); out.p_recognise_mean = static_cast<float>(sr);
    out.p_notice_sd = static_cast<float>(std::sqrt(vn / m)); out.p_recognise_sd = static_cast<float>(std::sqrt(vr / m));
    std::map<Limit, int> cnt;
    for (const auto& r : rs) ++cnt[r.limit];
    int best = -1;
    for (const auto& [l, c] : cnt) if (c > best) { best = c; out.modal_limit = l; }   // std::map iterates ascending: ties keep the lower
    return out;
}
}  // namespace rc::card
