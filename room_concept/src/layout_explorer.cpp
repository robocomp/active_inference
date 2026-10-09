/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see layout_explorer.h. A PORT: every block below names the offline
 *    code it was moved from (tools/wall_slam_selftest.cpp, run_explore and its helpers) and keeps its
 *    arithmetic. The offline comments that explain WHY each term exists stay there, with the measurements.
 */
#include "layout_explorer.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <queue>
#include <unordered_map>

#include "corner_visibility.h"
#include "line_fit.h"

namespace rc::layout
{
namespace
{
    constexpr float kPi = static_cast<float>(M_PI);

    float point_to_segment(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float t = std::clamp((p - a).dot(ab) / std::max(1e-9f, ab.squaredNorm()), 0.f, 1.f);
        return (p - (a + t * ab)).norm();
    }
    float point_to_poly(const Eigen::Vector2f& p, const std::vector<Eigen::Vector2f>& poly)
    {
        float best = 1e9f;
        for (size_t i = 0; i < poly.size(); ++i)
            best = std::min(best, point_to_segment(p, poly[i], poly[(i + 1) % poly.size()]));
        return best;
    }
}   // namespace

// ── offline `struct Explorer` (constructor, astar) ────────────────────────────────────────────────────
FeasibleGrid::FeasibleGrid(const std::vector<Eigen::Vector2f>& poly, float cell_, float clearance_)
    : cell(cell_), clearance(clearance_)
{
    if (poly.size() < 3)
        return;
    float xa = 1e9f, ya = 1e9f, xb = -1e9f, yb = -1e9f;
    for (const auto& v : poly) { xa = std::min(xa, v.x()); ya = std::min(ya, v.y()); xb = std::max(xb, v.x()); yb = std::max(yb, v.y()); }
    x0 = xa; y0 = ya;
    nx = static_cast<int>((xb - xa) / cell) + 1;
    ny = static_cast<int>((yb - ya) / cell) + 1;
    free_.assign(static_cast<size_t>(nx * ny), 0);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            const Eigen::Vector2f p = at(i, j);
            if (rc::corner_visibility::point_in_polygon(p, poly) and point_to_poly(p, poly) > clearance)
                free_[static_cast<size_t>(j * nx + i)] = 1;
        }
}

int FeasibleGrid::free_count() const
{
    return static_cast<int>(std::ranges::count(free_, static_cast<char>(1)));
}

std::vector<Eigen::Vector2f> FeasibleGrid::astar(const Eigen::Vector2f& from, const Eigen::Vector2f& to) const
{
    auto [si, sj] = cell_of(from);
    auto [ti, tj] = cell_of(to);
    if (not is_free(ti, tj) or not is_free(si, sj)) return {};
    const int n = nx * ny;
    std::vector<float> g(static_cast<size_t>(n), 1e18f);
    std::vector<int> par(static_cast<size_t>(n), -1);
    auto idx = [&](int i, int j) { return j * nx + i; };
    auto h = [&](int i, int j) { return std::hypot(static_cast<float>(i - ti), static_cast<float>(j - tj)); };
    using QN = std::pair<float, int>;
    std::priority_queue<QN, std::vector<QN>, std::greater<>> q;
    g[static_cast<size_t>(idx(si, sj))] = 0.f;
    q.push({h(si, sj), idx(si, sj)});
    const int di[8] = {1,-1,0,0,1,1,-1,-1}, dj[8] = {0,0,1,-1,1,-1,1,-1};
    while (not q.empty())
    {
        const auto [f, u] = q.top(); q.pop();
        const int ui = u % nx, uj = u / nx;
        if (ui == ti and uj == tj) break;
        if (f > g[static_cast<size_t>(u)] + h(ui, uj) + 1e-4f) continue;
        for (int k = 0; k < 8; ++k)
        {
            const int vi = ui + di[k], vj = uj + dj[k];
            if (not is_free(vi, vj)) continue;
            if (k >= 4 and (not is_free(ui, vj) or not is_free(vi, uj))) continue;  // no corner cutting
            const float w = (k < 4) ? 1.f : 1.41421f;
            if (g[static_cast<size_t>(u)] + w < g[static_cast<size_t>(idx(vi, vj))])
            {
                g[static_cast<size_t>(idx(vi, vj))] = g[static_cast<size_t>(u)] + w;
                par[static_cast<size_t>(idx(vi, vj))] = u;
                q.push({g[static_cast<size_t>(idx(vi, vj))] + h(vi, vj), idx(vi, vj)});
            }
        }
    }
    if (par[static_cast<size_t>(idx(ti, tj))] < 0 and not (si == ti and sj == tj)) return {};
    std::vector<Eigen::Vector2f> path;
    for (int u = idx(ti, tj); u >= 0; u = par[static_cast<size_t>(u)])
    { path.push_back(at(u % nx, u / nx)); if (u == idx(si, sj)) break; }
    std::reverse(path.begin(), path.end());
    return path;
}

Eigen::Vector2f FeasibleGrid::nearest_free(const Eigen::Vector2f& p) const
{
    if (nx <= 0 or ny <= 0) return p;
    auto [si, sj] = cell_of(p);
    si = std::clamp(si, 0, nx - 1); sj = std::clamp(sj, 0, ny - 1);
    std::vector<char> seen(static_cast<size_t>(nx * ny), 0);
    std::deque<std::pair<int, int>> q{{si, sj}};
    seen[static_cast<size_t>(sj * nx + si)] = 1;
    while (not q.empty())
    {
        const auto [i, j] = q.front(); q.pop_front();
        if (is_free(i, j)) return at(i, j);
        for (const auto& [a, b] : {std::pair{i + 1, j}, {i - 1, j}, {i, j + 1}, {i, j - 1}})
        {
            if (a < 0 or b < 0 or a >= nx or b >= ny) continue;
            auto& s = seen[static_cast<size_t>(b * nx + a)];
            if (s) continue;
            s = 1;
            q.emplace_back(a, b);
        }
    }
    return p;
}

// ── offline `entropy_nats` ────────────────────────────────────────────────────────────────────────────
float entropy_nats(float lodds)
{
    const float pp = std::clamp(1.f / (1.f + std::exp(-lodds)), 1e-6f, 1.f - 1e-6f);
    return -pp * std::log(pp) - (1.f - pp) * std::log(1.f - pp);
}

// ── offline `collect_entropy_targets` (verbatim) ──────────────────────────────────────────────────────
std::vector<Unknown> collect_entropy_targets(const wallmap::WallMap& map)
{
    std::vector<Unknown> out;
    const auto poly_now = map.build_polygon();
    const auto interior_far_from_boundary = [&](const Eigen::Vector2f& q)
    {
        if (not poly_now.closed or poly_now.verts.size() < 3) return false;
        if (not rc::corner_visibility::point_in_polygon(q, poly_now.verts)) return false;
        const float clear = 2.f * map.fgrid.cell;
        for (size_t e = 0; e < poly_now.verts.size(); ++e)
        {
            const Eigen::Vector2f a = poly_now.verts[e], ab = poly_now.verts[(e + 1) % poly_now.verts.size()] - a;
            const float l2 = ab.squaredNorm();
            const float tt = l2 > 1e-9f ? std::clamp((q - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
            if ((q - (a + tt * ab)).norm() < clear) return false;
        }
        return true;
    };
    if (map.fgrid.ready())
    {
        const int stride = 3;
        for (int i = 0; i < map.fgrid.nx; i += stride)
            for (int j = 0; j < map.fgrid.ny; j += stride)
            {
                const size_t id = static_cast<size_t>(map.fgrid.idx(i, j));
                float h = entropy_nats(map.fgrid.lodds[id]);
                const unsigned short hits = map.fgrid.hits[id];
                if (hits >= 1 and hits < 3 and map.fgrid.lodds[id] < 0.f
                    and interior_far_from_boundary(map.fgrid.at(i, j)))
                    h = std::max(h, std::log(2.f));
                if (h > 0.15f) out.push_back({map.fgrid.at(i, j), h * static_cast<float>(stride * stride)});
            }
    }
    for (const auto& w : map.walls)
    {
        const Eigen::Vector2f n = w.normal(), t = w.tangent();
        for (size_t b = 0; b < w.exist_bins.size(); ++b)
        {
            const float h = entropy_nats(w.exist_bins[b] - map.params.birth_nats);
            if (h > 0.15f)
                out.push_back({n * w.d + t * (w.bins_s0 + (static_cast<float>(b) + 0.5f) * map.params.exist_bin_m), h});
        }
        if (w.exist_bins.empty() and w.has_extent)
            out.push_back({n * w.d + t * (0.5f * (w.s_min + w.s_max)), std::log(2.f)});
    }
    for (const auto& c : map.candidates)
        if (c.npts >= 3)
            out.push_back({rc::linefit::normal_of(c.phi) * c.d
                           + rc::linefit::tangent_of(c.phi) * (0.5f * (c.s_min + c.s_max)),
                           entropy_nats(c.evidence() - map.params.birth_nats)});
    for (const auto& c : poly_now.corners)
    {
        const float sig = std::isfinite(c.sigma) ? c.sigma : 1e3f;
        if (sig > map.params.publish_corner_sigma)
            out.push_back({c.p, std::log(sig / map.params.publish_corner_sigma)});
    }
    return out;
}

// ── offline `eig_of` (verbatim, minus `to_map`: live candidates are already in the map frame, and the
//    polygon is passed in instead of rebuilt per call) ────────────────────────────────────────────────
float expected_information_gain(const wallmap::WallMap& map, const wallmap::Polygon& poly,
                                const Eigen::Vector2f& v, const ExplorerParams& p)
{
    if (not poly.closed or poly.verts.size() < 3) return 0.f;
    const auto& fg = map.fgrid;
    if (not fg.ready()) return 0.f;
    const int NB = p.beams;
    const float sig2 = map.params.obs_sigma * map.params.obs_sigma;
    std::unordered_map<std::uint64_t, Eigen::Matrix2f> J;   // wall id → added information
    float gain = 0.f;
    for (int b = 0; b < NB; ++b)
    {
        const float a = 2.f * kPi * static_cast<float>(b) / static_cast<float>(NB);
        const Eigen::Vector2f dir(std::cos(a), std::sin(a));
        float r_wall = map.params.sensor_range; int hit = -1;
        for (size_t e = 0; e < poly.verts.size(); ++e)
            if (const auto t = rc::corner_visibility::ray_segment_t(
                    v, dir, poly.verts[e], poly.verts[(e + 1) % poly.verts.size()]);
                t and *t > 1e-3f and *t < r_wall) { r_wall = *t; hit = static_cast<int>(e); }
        const float r_pred = map.params.sensor_range;
        bool blocked = false;
        for (float rr = fg.cell; rr < r_pred; rr += fg.cell)
        {
            const Eigen::Vector2f q = v + dir * rr;
            const int i = static_cast<int>((q.x() - fg.x0) / fg.cell);
            const int j = static_cast<int>((q.y() - fg.y0) / fg.cell);
            if (not fg.in(i, j)) continue;
            const float l = fg.lodds[static_cast<size_t>(fg.idx(i, j))];
            const float pr = 1.f / (1.f + std::exp(-l));
            const float h_now = entropy_nats(l);
            const float h_after = pr * entropy_nats(std::min(4.f, l + 1.0f))
                                + (1.f - pr) * entropy_nats(std::max(-4.f, l - 0.4f));
            gain += std::max(0.f, h_now - h_after);
            if (p.occl_pred and fg.is_occupied(i, j)) { blocked = true; break; }
        }
        if (not blocked and hit >= 0 and r_wall < r_pred
            and static_cast<size_t>(hit) < poly.wall_of_edge.size())
        {
            const auto* w = map.find(poly.wall_of_edge[static_cast<size_t>(hit)]);
            if (w != nullptr)
            {
                const Eigen::Vector2f q = v + dir * r_wall;
                const float sc_ = w->tangent().dot(q);
                Eigen::Vector2f h(sc_, -1.f);
                // ★DEVIATION FROM THE BENCH, ON PURPOSE: the bench writes `J[w->id] += ...`, and
                // unordered_map::operator[] value-initialises an Eigen::Matrix2f, which Eigen does NOT
                // zero — the first look at a wall started from whatever the allocator handed back (in
                // practice the previous call's J, so repeated evaluations of one standpoint DRIFTED
                // upwards: 368.6 → 373.2 nats on two identical calls, measured in
                // tools/layout_explorer_selftest.cpp). Zero-initialised here; the bench still has it.
                J.try_emplace(w->id, Eigen::Matrix2f::Zero()).first->second += (h * h.transpose()) / sig2;
            }
        }
    }
    if (p.dwell)
        for (const auto& [wid, Jw] : J)
        {
            const auto* w = map.find(wid);
            if (w == nullptr) continue;
            const Eigen::Matrix2f L0 = w->information;
            const float d0 = L0.determinant();
            if (not std::isfinite(d0) or d0 <= 1e-9f) continue;
            const float d1 = (L0 + Jw).determinant();
            if (std::isfinite(d1) and d1 > d0) gain += 0.5f * std::log(d1 / d0);
        }
    for (const auto& c : poly.corners)
    {
        if (not std::isfinite(c.sigma) or c.sigma <= map.params.publish_corner_sigma) continue;
        const auto* wa = map.find(c.wall_a); const auto* wb = map.find(c.wall_b);
        if (wa == nullptr or wb == nullptr) continue;
        const auto ja = J.find(wa->id), jb = J.find(wb->id);
        if (ja == J.end() and jb == J.end()) continue;
        wallmap::WallLandmark a2 = *wa, b2 = *wb;
        if (ja != J.end()) a2.information += ja->second;
        if (jb != J.end()) b2.information += jb->second;
        const auto c2 = wallmap::WallMap::intersect_walls(a2, b2, c.inferred);
        if (std::isfinite(c2.sigma) and c2.sigma > 0.f and c2.sigma < c.sigma)
            gain += std::log(c.sigma / c2.sigma);
    }
    return gain;
}

// ── offline run_explore replan body, EIG branch (cfg.info_gain = true, WS_NO_EIG unset) ────────────────
Plan plan(const wallmap::WallMap& map, const wallmap::Polygon& poly, const FeasibleGrid& grid,
          const Eigen::Vector2f& robot, const ExplorerParams& p, const Admissible& admissible)
{
    Plan out;
    const auto unknowns = collect_entropy_targets(map);
    out.unknowns = static_cast<int>(unknowns.size());
    // offline: `if (unknowns.empty() and closed_now) break;` — the run is over; live: nothing to offer.
    if (unknowns.empty() and map.build_polygon().closed)
    {
        out.finished = true;
        out.why = "finished";
        return out;
    }
    // LIVE ONLY (see header): a robot standing off the feasible grid walks from the nearest feasible cell.
    const auto [ri, rj] = grid.cell_of(robot);
    out.start = grid.is_free(ri, rj) ? robot : grid.nearest_free(robot);
    if (grid.free_count() == 0)
    {
        out.why = "no_feasible";
        return out;
    }
    std::vector<std::pair<float, Eigen::Vector2f>> cands;
    for (int i = 0; i < grid.nx; i += 2)
        for (int j = 0; j < grid.ny; j += 2)
        {
            if (not grid.is_free(i, j)) continue;
            const Eigen::Vector2f v = grid.at(i, j);
            if (admissible and not admissible(v)) continue;   // LIVE ONLY (see Admissible)
            ++out.candidates;
            float sc = expected_information_gain(map, poly, v, p);
            if (sc <= 0.f) continue;
            ++out.with_gain;
            sc /= (1.0f + (v - robot).norm());   // nats per look (offline: tru.head<2>(), the robot itself)
            cands.emplace_back(sc, v);
        }
    std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (static_cast<int>(cands.size()) > p.keep_best) cands.resize(static_cast<size_t>(p.keep_best));
    for (const auto& [sc, v] : cands)
    {
        auto path = grid.astar(out.start, v);
        if (path.size() > 1)
        {
            path.erase(path.begin());
            out.ok = true;
            out.target = v;
            out.score = sc;
            out.gain = sc * (1.0f + (v - robot).norm());
            out.path = std::move(path);
            out.why = "ok";
            return out;
        }
    }
    out.why = cands.empty() ? "no_gain" : "unreachable";
    return out;
}

}   // namespace rc::layout
