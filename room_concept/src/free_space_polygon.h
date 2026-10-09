/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
#pragma once

// ── CONFIRMED FREE SPACE AS A POLYGON — the start-up proto-room's extent ─────────────────────────────────
//
// WHAT IT IS FOR. In an estimation run the room exists in the graph from the first second as a PROTO-ROOM
// (room_scene_graph.cpp, start-up proto). The controller plans inside whatever `delimiting_polygon` it is
// handed and CANNOT tell a proto from a surveyed room, so the polygon must be safe BY CONSTRUCTION: it may
// only contain space that has actually been SEEN to be free. Never sensor range plus a margin — an outward
// margin claims unobserved space as room and puts targets inside walls (memory: nbv-room-polygon-missing-
// wall-face, proto-room-for-estimation-time-navigation decision 2).
//
// THE MODEL. Every LiDAR beam is an observation that the space it crossed is empty and that its endpoint
// is matter. A raster in the proto frame (frozen, so it never moves with the estimator's gauge) keeps:
//   · FREE   — some beam crossed this cell before reaching its return;
//   · MATTER — some return landed in it. STICKY: a later beam crossing it never clears it. That is the
//              conservative direction: a cell that was ever a surface is never claimed as floor.
//   · UNKNOWN — everything else.
// The polygon is the FREE region's 4-connected component around the robot, ERODED by the body radius
// (exact Euclidean distance transform to the nearest non-free cell, so unknown and matter both repel it),
// its outer boundary traced on cell edges and simplified to <= max_verts vertices.
//
// WHY THE SIMPLIFICATION CANNOT PUSH IT INTO UNSEEN SPACE. Douglas–Peucker with tolerance eps moves the
// boundary by at most eps. So the region is eroded by (radius + eps + one cell) BEFORE simplifying: any
// point the simplified ring gains is within eps of a cell whose centre is radius + eps + cell from the
// nearest non-free cell, hence still at least `radius` inside seen free space. If the ring still has too
// many vertices, eps doubles and the erosion grows with it — the vertex budget is paid in CONSERVATISM,
// never in safety.
//
// ⚠ KNOWN, DOCUMENTED LIMITS (not thresholds; consequences of a single-ring polygon):
//   · HOLES ARE FILLED. delimiting_polygon is one ring, so an island inside the free region (a pillar, a
//     free-standing tall cabinet, its shadow) ends up inside the ring. That is exactly the status of
//     furniture inside a surveyed room polygon today: the controller's own LiDAR obstacle layer owns it.
//   · The beams are the agent's WALL BAND (above the furniture), so free space is free AT THAT HEIGHT. Low
//     furniture under it is, again, the controller's obstacle layer's — as in any surveyed room.
//   · A person walking through leaves sticky MATTER behind. Conservative: it can only shrink the region.
//
// Pure: no graph, no Qt, no threads. One owner (RoomSceneGraph on the LOCALISER thread).
// Selftest: tools/free_space_polygon_selftest.cpp (synthetic L room).

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace rc::freespace
{

struct Params
{
    float cell       = 0.05f;   ///< m — raster resolution
    float half_span  = 20.f;    ///< m — the raster covers [origin ± half_span]; beams beyond it are clipped
    float erode_m    = 0.24f;   ///< m — body radius the region is eroded by (the controller's footprint fits)
    int   max_verts  = 64;      ///< vertex budget of the published ring
};

/// Signed area (CCW positive).
inline float signed_area(const std::vector<Eigen::Vector2f>& p)
{
    float a = 0.f;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto& u = p[i];
        const auto& v = p[(i + 1) % p.size()];
        a += u.x() * v.y() - v.x() * u.y();
    }
    return 0.5f * a;
}

inline bool point_in_polygon(const Eigen::Vector2f& q, const std::vector<Eigen::Vector2f>& poly)
{
    bool in = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const auto& a = poly[i];
        const auto& b = poly[j];
        if (((a.y() > q.y()) != (b.y() > q.y()))
            and (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x()))
            in = not in;
    }
    return in;
}

/// True when no two non-adjacent edges of the closed ring touch or cross.
inline bool is_simple(const std::vector<Eigen::Vector2f>& p)
{
    const std::size_t n = p.size();
    if (n < 3)
        return false;
    const auto orient = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& c)
    {
        const float v = (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
        return (v > 1e-9f) - (v < -1e-9f);
    };
    const auto on_seg = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& q)
    {
        return std::min(a.x(), b.x()) - 1e-6f <= q.x() and q.x() <= std::max(a.x(), b.x()) + 1e-6f
               and std::min(a.y(), b.y()) - 1e-6f <= q.y() and q.y() <= std::max(a.y(), b.y()) + 1e-6f;
    };
    const auto touch = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b,
                           const Eigen::Vector2f& c, const Eigen::Vector2f& d)
    {
        const int o1 = orient(a, b, c), o2 = orient(a, b, d), o3 = orient(c, d, a), o4 = orient(c, d, b);
        if (o1 != o2 and o3 != o4) return true;
        if (o1 == 0 and on_seg(a, b, c)) return true;
        if (o2 == 0 and on_seg(a, b, d)) return true;
        if (o3 == 0 and on_seg(c, d, a)) return true;
        if (o4 == 0 and on_seg(c, d, b)) return true;
        return false;
    };
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
        {
            if (j == i + 1 or (i == 0 and j == n - 1))
                continue;   // adjacent edges share a vertex by construction
            if (touch(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]))
                return false;
        }
    return true;
}

class FreeSpacePolygon
{
public:
    explicit FreeSpacePolygon(Params p = {}) : p_(p) {}

    [[nodiscard]] const Params& params() const noexcept { return p_; }
    [[nodiscard]] bool ready() const noexcept { return n_ > 0; }
    /// Cells ever marked free (monotone non-decreasing except where a later return claims one as matter).
    [[nodiscard]] long free_cells() const noexcept { return n_free_; }

    /// One scan: beams from `origin` to each endpoint, both in the proto frame.
    /// The robot's OWN footprint is confirmed free: the robot is standing on it (2026-10-09). The LiDAR's beams
    /// never sweep the sector its own body shadows, so without this the erosion cut the robot's position OUT of
    /// its own room — live, the robot stood 5 cm outside the published ring, the controller's start cell was
    /// occupied and it never found a path. Overrides matter: nothing else can be where the robot is (a self-
    /// return or a person who walked through). Called every update, so the driven path is free too.
    void add_footprint(const Eigen::Vector2f& centre, float radius)
    {
        if (not ready())
            init(centre);
        const int r = static_cast<int>(std::ceil(radius / p_.cell));
        const auto [ci, cj] = cell_of(centre);
        for (int dj = -r; dj <= r; ++dj)
            for (int di = -r; di <= r; ++di)
            {
                const int i = ci + di, j = cj + dj;
                if (not in(i, j)) continue;
                const float dx = di * p_.cell, dy = dj * p_.cell;
                if (dx * dx + dy * dy > radius * radius) continue;
                // matter under the robot is not matter: the passes outvote whatever ended here
                frees_[idx(i, j)] = static_cast<std::uint16_t>(std::max<int>(frees_[idx(i, j)], hits_[idx(i, j)]));
                if (low_.size() == st_.size()) low_[idx(i, j)] = 0;
                set_free(i, j);
            }
    }

    /// Obstacles the scan beams pass OVER (counters, tables, chairs), in a SEPARATE layer: they do NOT shape the
    /// published ring (live 2026-10-09, stamped into the free/matter state they cut the ring 35 -> 3 m² and the
    /// explorer had no standpoint left); only the explorer reads them, through matter_within.
    void add_matter(const std::vector<Eigen::Vector2f>& pts)
    {
        if (not ready()) return;
        if (low_.size() != st_.size()) low_.assign(st_.size(), 0);
        for (const auto& q : pts)
            if (const auto [i, j] = cell_of(q); in(i, j)) low_[idx(i, j)] = 1;
    }
    /// True if any low-obstacle cell lies within `radius` of `c` — a standpoint there would put the body on furniture.
    [[nodiscard]] bool matter_within(const Eigen::Vector2f& c, float radius) const
    {
        if (not ready()) return false;
        const int r = static_cast<int>(std::ceil(radius / p_.cell));
        const auto [ci, cj] = cell_of(c);
        for (int dj = -r; dj <= r; ++dj)
            for (int di = -r; di <= r; ++di)
            {
                const int i = ci + di, j = cj + dj;
                if (not in(i, j)) continue;
                const float dx = di * p_.cell, dy = dj * p_.cell;
                if (dx * dx + dy * dy <= radius * radius and low_.size() == st_.size() and low_[idx(i, j)] != 0)
                    return true;
            }
        return false;
    }

    void add_scan(const Eigen::Vector2f& origin, const std::vector<Eigen::Vector2f>& endpoints)
    {
        if (not ready())
            init(origin);
        for (const auto& e : endpoints)
            trace(origin, e);
    }

    /// Raw state of the cell holding `q` (for tests / diagnostics): 0 unknown, 1 free, 2 matter, -1 outside.
    [[nodiscard]] int state_at(const Eigen::Vector2f& q) const
    {
        const auto [i, j] = cell_of(q);
        if (not in(i, j)) return -1;
        return st_[idx(i, j)];
    }

    /// The eroded, simplified free region around `seed` as a CCW ring of <= max_verts vertices, or empty
    /// when there is none yet (no scan, or no cell anywhere near the seed is far enough from the unknown).
    [[nodiscard]] std::vector<Eigen::Vector2f> polygon(const Eigen::Vector2f& seed) const
    {
        if (not ready() or n_free_ == 0)
            return {};
        // Work on the bounding box of the free cells, padded so every boundary has a blocked cell beside it.
        const int i0 = std::max(0, fi0_ - 2), j0 = std::max(0, fj0_ - 2);
        const int i1 = std::min(n_ - 1, fi1_ + 2), j1 = std::min(n_ - 1, fj1_ + 2);
        const int W = i1 - i0 + 1, H = j1 - j0 + 1;
        // Squared distance (in cells) from each cell centre to the nearest NON-free cell centre.
        std::vector<float> d2(static_cast<std::size_t>(W) * H);
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i)
            {
                const bool border = i == 0 or j == 0 or i == W - 1 or j == H - 1;
                const bool free = not border and st_[idx(i0 + i, j0 + j)] == 1;
                d2[static_cast<std::size_t>(j) * W + i] = free ? kInf : 0.f;
            }
        edt_2d(d2, W, H);

        auto [si, sj] = cell_of(seed);
        si -= i0; sj -= j0;
        for (float eps = p_.cell; eps <= 16.f * p_.cell + 1e-6f; eps *= 2.f)
        {
            const float thr = (p_.erode_m + eps + p_.cell) / p_.cell;   // in cells
            const float thr2 = thr * thr;
            std::vector<std::uint8_t> keep(d2.size(), 0);
            for (std::size_t k = 0; k < d2.size(); ++k)
                keep[k] = d2[k] > thr2 ? 1 : 0;
            // Component around the seed, holes filled, pinches cut — and again, because cutting a pinch can
            // split the region; the ring must be the piece the ROBOT is in.
            std::vector<std::uint8_t> comp;
            bool settled = false;
            for (int pass = 0; pass < 16 and not settled; ++pass)
            {
                const auto start = nearest_kept(keep, W, H, si, sj);
                if (not start.has_value())
                    return {};
                comp = component(keep, W, H, start->first, start->second);
                fill_holes(comp, W, H);
                settled = remove_pinches(comp, W, H) == 0;
                keep = comp;
            }
            if (not settled)
                continue;
            auto ring = trace_outer(comp, W, H);
            if (ring.size() < 3)
                continue;
            for (auto& v : ring)
                v = Eigen::Vector2f(x0_ + (static_cast<float>(i0) + v.x()) * p_.cell,
                                    y0_ + (static_cast<float>(j0) + v.y()) * p_.cell);
            ring = merge_collinear(ring);
            ring = douglas_peucker_closed(ring, eps);
            if (ring.size() >= 3 and static_cast<int>(ring.size()) <= p_.max_verts and is_simple(ring))
            {
                if (signed_area(ring) < 0.f)
                    std::ranges::reverse(ring);
                return ring;
            }
        }
        return {};
    }

    /// ── MANHATTAN LAYOUT FROM THE MAP (2026-10-09) ─────────────────────────────────────────────────────────
    /// The room as the SLAM map shows it, as an axis-aligned polygon in the raster's (published) frame:
    ///  1. region  = 4-connected component of cells beams PASSED more than they ENDED in, not surface (>=2 hits),
    ///               containing `start`; closed once, holes filled (furniture islands are inside the room);
    ///  2. axes    = dominant wall direction: 4θ circular mean of the gradient of the (smoothed) surface field;
    ///  3. outline = the region resampled into those axes, its pixel boundary, Douglas–Peucker at `tol`, every edge
    ///               classed H/V at its length-weighted line, same-class neighbours merged, corners = H∩V;
    ///  4. snap    = each edge moved onto the MEASURED wall: the hit-weighted median of surface cells within `reach`
    ///               of it, inside its extent (the free region stops at the inner edge of the hit band; without the
    ///               snap a uniform 0.2–0.3 m band of the room was missing).
    /// Offline vs Webots ground truth (scratchpad/slam/rectify.py, same algorithm): IoU 0.957 / 0.963 / 0.973 on three
    /// live recordings, 30–34 vertices for a 32-vertex apartment. Empty if there is no usable region.
    [[nodiscard]] std::vector<Eigen::Vector2f> manhattan_layout(const Eigen::Vector2f& start, float tol = 0.12f,
                                                                float reach = 0.40f, float* theta_out = nullptr) const
    {
        if (not ready() or hits_.size() != st_.size()) return {};
        const int N = n_; const float c = p_.cell;
        const auto surf = [&](std::size_t k) { return hits_[k] >= 2; };
        // 1. region
        std::vector<std::uint8_t> keep(st_.size(), 0);
        for (std::size_t k = 0; k < st_.size(); ++k) keep[k] = (frees_[k] > hits_[k] and not surf(k)) ? 1 : 0;
        auto [si, sj] = cell_of(start);
        if (not in(si, sj)) return {};
        if (not keep[idx(si, sj)])
        {
            const auto nk = nearest_kept(keep, N, N, si, sj);
            if (not nk) return {};
            si = nk->first; sj = nk->second;
        }
        auto comp = component(keep, N, N, si, sj);
        {   // one closing (3x3)
            std::vector<std::uint8_t> d(comp.size(), 0), e(comp.size(), 0);
            for (int j = 1; j + 1 < N; ++j) for (int i = 1; i + 1 < N; ++i)
            { bool any = false; for (int b = -1; b <= 1 and not any; ++b) for (int a = -1; a <= 1; ++a) if (comp[idx(i + a, j + b)]) { any = true; break; } d[idx(i, j)] = any; }
            for (int j = 1; j + 1 < N; ++j) for (int i = 1; i + 1 < N; ++i)
            { bool all = true; for (int b = -1; b <= 1 and all; ++b) for (int a = -1; a <= 1; ++a) if (not d[idx(i + a, j + b)]) { all = false; break; } e[idx(i, j)] = all; }
            for (std::size_t k = 0; k < comp.size(); ++k) comp[k] = comp[k] or e[k];
        }
        fill_holes(comp, N, N);
        // 2. dominant axis: smoothed surface field (5-tap σ≈1 cell), Sobel, Σ w·e^{4iθ}
        std::vector<float> S(st_.size(), 0.f), T(st_.size(), 0.f);
        constexpr float g5[5] = {0.054f, 0.244f, 0.403f, 0.244f, 0.054f};
        for (int j = 0; j < N; ++j) for (int i = 2; i + 2 < N; ++i)
        { float a = 0.f; for (int t = -2; t <= 2; ++t) a += g5[t + 2] * (surf(idx(i + t, j)) ? 1.f : 0.f); T[idx(i, j)] = a; }
        for (int j = 2; j + 2 < N; ++j) for (int i = 0; i < N; ++i)
        { float a = 0.f; for (int t = -2; t <= 2; ++t) a += g5[t + 2] * T[idx(i, j + t)]; S[idx(i, j)] = a; }
        double zr = 0.0, zi = 0.0;
        for (int j = 1; j + 1 < N; ++j) for (int i = 1; i + 1 < N; ++i)
        {
            const float gx = (S[idx(i + 1, j - 1)] + 2 * S[idx(i + 1, j)] + S[idx(i + 1, j + 1)]) - (S[idx(i - 1, j - 1)] + 2 * S[idx(i - 1, j)] + S[idx(i - 1, j + 1)]);
            const float gy = (S[idx(i - 1, j + 1)] + 2 * S[idx(i, j + 1)] + S[idx(i + 1, j + 1)]) - (S[idx(i - 1, j - 1)] + 2 * S[idx(i, j - 1)] + S[idx(i + 1, j - 1)]);
            const double w = std::hypot(gx, gy);
            if (w <= 0.0) continue;
            const double a = 4.0 * std::atan2(gy, gx);
            zr += w * std::cos(a); zi += w * std::sin(a);
        }
        const float th = static_cast<float>(0.25 * std::atan2(zi, zr));
        if (theta_out) *theta_out = th;
        const float ct = std::cos(th), st = std::sin(th);
        // 3. resample the region into the room's axes (u = R(-θ) x)
        float u0 = 1e9f, v0 = 1e9f, u1 = -1e9f, v1 = -1e9f;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
            if (comp[idx(i, j)])
            {
                const float x = x0_ + (i + 0.5f) * c, y = y0_ + (j + 0.5f) * c;
                const float u = ct * x + st * y, v = -st * x + ct * y;
                u0 = std::min(u0, u); v0 = std::min(v0, v); u1 = std::max(u1, u); v1 = std::max(v1, v);
            }
        if (u1 < u0) return {};
        u0 -= 2 * c; v0 -= 2 * c;
        const int W = static_cast<int>((u1 - u0) / c) + 4, H = static_cast<int>((v1 - v0) / c) + 4;
        std::vector<std::uint8_t> M(static_cast<std::size_t>(W) * H, 0);
        for (int b = 0; b < H; ++b) for (int a = 0; a < W; ++a)
        {
            const float u = u0 + (a + 0.5f) * c, v = v0 + (b + 0.5f) * c;
            const float x = ct * u - st * v, y = st * u + ct * v;
            const auto [i, j] = cell_of(Eigen::Vector2f(x, y));
            if (in(i, j) and comp[idx(i, j)]) M[static_cast<std::size_t>(b) * W + a] = 1;
        }
        fill_holes(M, W, H);
        {   // largest 4-connected component only
            std::vector<int> lab(M.size(), 0); int best = 0, bestn = 0, cur = 0;
            for (std::size_t k0 = 0; k0 < M.size(); ++k0)
            {
                if (not M[k0] or lab[k0]) continue;
                ++cur; int cnt = 0; std::vector<std::size_t> q{k0}; lab[k0] = cur;
                while (not q.empty())
                {
                    const auto k = q.back(); q.pop_back(); ++cnt;
                    const int a = static_cast<int>(k % W), b = static_cast<int>(k / W);
                    const int na[4] = {a + 1, a - 1, a, a}, nb[4] = {b, b, b + 1, b - 1};
                    for (int t = 0; t < 4; ++t)
                        if (na[t] >= 0 and nb[t] >= 0 and na[t] < W and nb[t] < H)
                            if (const auto kk = static_cast<std::size_t>(nb[t]) * W + na[t]; M[kk] and not lab[kk]) { lab[kk] = cur; q.push_back(kk); }
                }
                if (cnt > bestn) { bestn = cnt; best = cur; }
            }
            for (std::size_t k = 0; k < M.size(); ++k) M[k] = (lab[k] == best) ? 1 : 0;
        }
        remove_pinches(M, W, H);
        auto ring = trace_outer(M, W, H);
        if (ring.size() < 4) return {};
        for (auto& p : ring) p = Eigen::Vector2f(u0 + p.x() * c, v0 + p.y() * c);   // aligned metres
        ring = douglas_peucker_closed(merge_collinear(ring), tol);
        // H/V lines, merged
        struct E { bool h; float v; float L; };
        std::vector<E> es;
        for (std::size_t k = 0; k < ring.size(); ++k)
        {
            const auto& a = ring[k]; const auto& b = ring[(k + 1) % ring.size()];
            const Eigen::Vector2f d = b - a; const float L = d.norm();
            if (L < 1e-6f) continue;
            if (std::abs(d.x()) >= std::abs(d.y())) es.push_back({true, 0.5f * (a.y() + b.y()), L});
            else                                     es.push_back({false, 0.5f * (a.x() + b.x()), L});
        }
        for (bool changed = true; changed and es.size() > 4;)
        {
            changed = false;
            for (std::size_t k = 0; k < es.size(); ++k)
            {
                const std::size_t n2 = (k + 1) % es.size();
                if (es[k].h != es[n2].h) continue;
                const float L = es[k].L + es[n2].L;
                es[k].v = (es[k].v * es[k].L + es[n2].v * es[n2].L) / L; es[k].L = L;
                es.erase(es.begin() + static_cast<long>(n2)); changed = true; break;
            }
        }
        if (es.size() < 4 or es.size() % 2) return {};
        const auto corners = [&](const std::vector<E>& e)
        {
            std::vector<Eigen::Vector2f> V;
            for (std::size_t k = 0; k < e.size(); ++k)
            {
                const auto& a = e[k]; const auto& b = e[(k + 1) % e.size()];
                V.emplace_back(a.h ? b.v : a.v, a.h ? a.v : b.v);
            }
            return V;
        };
        // 4. snap each edge onto the measured wall (hit-weighted median of surface cells near it)
        std::vector<std::array<float, 3>> hu;   // (u, v, weight)
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
            if (const auto k = idx(i, j); surf(k))
            {
                const float x = x0_ + (i + 0.5f) * c, y = y0_ + (j + 0.5f) * c;
                hu.push_back({ct * x + st * y, -st * x + ct * y, static_cast<float>(hits_[k])});
            }
        auto V = corners(es);
        for (std::size_t k = 0; k < es.size(); ++k)
        {
            const auto& a = V[(k + es.size() - 1) % es.size()]; const auto& b = V[k];   // edge k runs from corner k-1 to k
            std::vector<std::pair<float, float>> cand;
            for (const auto& h : hu)
            {
                const float along = es[k].h ? h[0] : h[1], across = es[k].h ? h[1] : h[0];
                const float lo = std::min(es[k].h ? a.x() : a.y(), es[k].h ? b.x() : b.y());
                const float hi = std::max(es[k].h ? a.x() : a.y(), es[k].h ? b.x() : b.y());
                if (along >= lo and along <= hi and std::abs(across - es[k].v) < reach) cand.emplace_back(across, h[2]);
            }
            if (cand.size() <= 3) continue;
            std::ranges::sort(cand);
            float tot = 0.f; for (const auto& q : cand) tot += q.second;
            float acc = 0.f;
            for (const auto& q : cand) { acc += q.second; if (acc >= 0.5f * tot) { es[k].v = q.first; break; } }
        }
        V = corners(es);
        std::vector<Eigen::Vector2f> out; out.reserve(V.size());
        for (const auto& q : V) out.emplace_back(ct * q.x() - st * q.y(), st * q.x() + ct * q.y());
        if (signed_area(out) < 0.f) std::ranges::reverse(out);
        return out;
    }

private:
    static constexpr float kInf = 1e20f;
    Params p_;
    int   n_ = 0;                     // raster is n_ x n_
    float x0_ = 0.f, y0_ = 0.f;       // corner of cell (0,0)
    std::vector<std::uint8_t> st_;    // 0 unknown, 1 free, 2 matter
    std::vector<std::uint8_t> low_;   // low-obstacle layer (add_matter); explorer-only, never shapes the ring
    std::vector<std::uint16_t> hits_, frees_;   // per-cell evidence: beams ENDING here vs beams PASSING through
    long  n_free_ = 0;
    int   fi0_ = std::numeric_limits<int>::max(), fj0_ = std::numeric_limits<int>::max();
    int   fi1_ = -1, fj1_ = -1;       // bbox of cells ever marked free

    [[nodiscard]] std::size_t idx(int i, int j) const { return static_cast<std::size_t>(j) * n_ + i; }
    [[nodiscard]] bool in(int i, int j) const { return i >= 0 and j >= 0 and i < n_ and j < n_; }
    [[nodiscard]] std::pair<int, int> cell_of(const Eigen::Vector2f& q) const
    {
        return {static_cast<int>(std::floor((q.x() - x0_) / p_.cell)),
                static_cast<int>(std::floor((q.y() - y0_) / p_.cell))};
    }

    void init(const Eigen::Vector2f& origin)
    {
        n_  = std::max(8, static_cast<int>(std::ceil(2.f * p_.half_span / p_.cell)));
        x0_ = origin.x() - 0.5f * n_ * p_.cell;
        y0_ = origin.y() - 0.5f * n_ * p_.cell;
        st_.assign(static_cast<std::size_t>(n_) * n_, 0);
        hits_.assign(st_.size(), 0);
        frees_.assign(st_.size(), 0);
    }

    // ── STATE = THE VOTE OF THE EVIDENCE (2026-10-09) ────────────────────────────────────────────────────
    // A cell is MATTER while more beams have ENDED in it than PASSED through it, FREE once passes lead. It used to
    // be one-shot sticky: a single endpoint (a grazing return, a pose wobble, a person) turned a cell matter for
    // ever, even one crossed by hundreds of beams — live the ring collapsed 44 -> 6.4 m² round the robot in 24 s
    // and the explorer had nothing left (no_gain). This is the occupancy log-odds with equal hit/miss weights.
    void restate(int i, int j)
    {
        const auto k = idx(i, j);
        const std::uint8_t now = hits_[k] > frees_[k] ? 2 : (frees_[k] > 0 ? 1 : 0);
        auto& s = st_[k];
        if (s == now) return;
        if (s == 1) --n_free_;
        if (now == 1)
        {
            ++n_free_;
            fi0_ = std::min(fi0_, i); fj0_ = std::min(fj0_, j);
            fi1_ = std::max(fi1_, i); fj1_ = std::max(fj1_, j);
        }
        s = now;
    }
    void set_free(int i, int j)
    {
        auto& f = frees_[idx(i, j)];
        if (f < std::numeric_limits<std::uint16_t>::max()) ++f;
        restate(i, j);
    }
    void set_matter(int i, int j)
    {
        auto& h = hits_[idx(i, j)];
        if (h < std::numeric_limits<std::uint16_t>::max()) ++h;
        restate(i, j);
    }

    /// Amanatides–Woo walk from `a` to `b`: every cell strictly before b's cell is free, b's cell is matter.
    void trace(const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        auto [i, j] = cell_of(a);
        const auto [ie, je] = cell_of(b);
        const Eigen::Vector2f d = b - a;
        const int sx = d.x() > 0.f ? 1 : (d.x() < 0.f ? -1 : 0);
        const int sy = d.y() > 0.f ? 1 : (d.y() < 0.f ? -1 : 0);
        const float gx = x0_ + static_cast<float>(i + (sx > 0 ? 1 : 0)) * p_.cell;
        const float gy = y0_ + static_cast<float>(j + (sy > 0 ? 1 : 0)) * p_.cell;
        float tmx = sx != 0 ? (gx - a.x()) / d.x() : kInf;
        float tmy = sy != 0 ? (gy - a.y()) / d.y() : kInf;
        const float tdx = sx != 0 ? p_.cell / std::abs(d.x()) : kInf;
        const float tdy = sy != 0 ? p_.cell / std::abs(d.y()) : kInf;
        const int max_steps = 4 * n_;
        for (int s = 0; s < max_steps; ++s)
        {
            if (i == ie and j == je)
            {
                if (in(i, j)) set_matter(i, j);
                return;
            }
            if (not in(i, j))
                return;                         // left the raster: nothing beyond is recorded, no matter mark
            set_free(i, j);
            if (tmx < tmy) { tmx += tdx; i += sx; }
            else           { tmy += tdy; j += sy; }
        }
    }

    // Felzenszwalb–Huttenlocher 1-D squared distance transform (in place over a strided line).
    static void edt_1d(std::vector<float>& f, std::size_t off, std::size_t stride, int n,
                       std::vector<float>& tmp, std::vector<int>& v, std::vector<float>& z)
    {
        tmp.resize(n); v.resize(n); z.resize(n + 1);
        for (int q = 0; q < n; ++q) tmp[q] = f[off + q * stride];
        int k = 0;
        v[0] = 0; z[0] = -kInf; z[1] = kInf;
        const auto inter = [&](int q, int r)
        { return ((tmp[q] + static_cast<float>(q) * q) - (tmp[r] + static_cast<float>(r) * r)) / (2.f * static_cast<float>(q - r)); };
        for (int q = 1; q < n; ++q)
        {
            float s = inter(q, v[k]);
            while (s <= z[k]) { --k; s = inter(q, v[k]); }
            ++k; v[k] = q; z[k] = s; z[k + 1] = kInf;
        }
        k = 0;
        for (int q = 0; q < n; ++q)
        {
            while (z[k + 1] < static_cast<float>(q)) ++k;
            const float dq = static_cast<float>(q - v[k]);
            f[off + q * stride] = dq * dq + tmp[v[k]];
        }
    }
    static void edt_2d(std::vector<float>& f, int W, int H)
    {
        std::vector<float> tmp, z; std::vector<int> v;
        for (int i = 0; i < W; ++i) edt_1d(f, static_cast<std::size_t>(i), static_cast<std::size_t>(W), H, tmp, v, z);
        for (int j = 0; j < H; ++j) edt_1d(f, static_cast<std::size_t>(j) * W, 1, W, tmp, v, z);
    }

    /// The kept cell nearest the seed, reached through kept-or-not cells (plain BFS over the window).
    static std::optional<std::pair<int, int>> nearest_kept(const std::vector<std::uint8_t>& keep, int W, int H,
                                                           int si, int sj)
    {
        si = std::clamp(si, 0, W - 1); sj = std::clamp(sj, 0, H - 1);
        std::vector<std::uint8_t> seen(keep.size(), 0);
        std::deque<std::pair<int, int>> q{{si, sj}};
        seen[static_cast<std::size_t>(sj) * W + si] = 1;
        while (not q.empty())
        {
            const auto [i, j] = q.front(); q.pop_front();
            if (keep[static_cast<std::size_t>(j) * W + i]) return std::pair{i, j};
            for (const auto& [di, dj] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}})
            {
                const int a = i + di, b = j + dj;
                if (a < 0 or b < 0 or a >= W or b >= H) continue;
                auto& s = seen[static_cast<std::size_t>(b) * W + a];
                if (s) continue;
                s = 1;
                q.emplace_back(a, b);
            }
        }
        return std::nullopt;
    }

    static std::vector<std::uint8_t> component(const std::vector<std::uint8_t>& keep, int W, int H, int si, int sj)
    {
        std::vector<std::uint8_t> c(keep.size(), 0);
        std::deque<std::pair<int, int>> q{{si, sj}};
        c[static_cast<std::size_t>(sj) * W + si] = 1;
        while (not q.empty())
        {
            const auto [i, j] = q.front(); q.pop_front();
            for (const auto& [di, dj] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}})
            {
                const int a = i + di, b = j + dj;
                if (a < 0 or b < 0 or a >= W or b >= H) continue;
                const std::size_t k = static_cast<std::size_t>(b) * W + a;
                if (not keep[k] or c[k]) continue;
                c[k] = 1;
                q.emplace_back(a, b);
            }
        }
        return c;
    }

    /// Everything the 8-connected outside cannot reach becomes part of the region (see "HOLES ARE FILLED").
    static void fill_holes(std::vector<std::uint8_t>& c, int W, int H)
    {
        std::vector<std::uint8_t> out(c.size(), 0);
        std::deque<std::pair<int, int>> q;
        const auto push = [&](int i, int j)
        {
            const std::size_t k = static_cast<std::size_t>(j) * W + i;
            if (c[k] or out[k]) return;
            out[k] = 1;
            q.emplace_back(i, j);
        };
        for (int i = 0; i < W; ++i) { push(i, 0); push(i, H - 1); }
        for (int j = 0; j < H; ++j) { push(0, j); push(W - 1, j); }
        while (not q.empty())
        {
            const auto [i, j] = q.front(); q.pop_front();
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di)
                {
                    const int a = i + di, b = j + dj;
                    if ((di == 0 and dj == 0) or a < 0 or b < 0 or a >= W or b >= H) continue;
                    push(a, b);
                }
        }
        for (std::size_t k = 0; k < c.size(); ++k)
            if (not out[k]) c[k] = 1;
    }

    /// A 2x2 block holding only a diagonal pair is a pinch: the traced boundary would touch itself there.
    /// Remove one of the two cells (the conservative direction) until none is left.
    static int remove_pinches(std::vector<std::uint8_t>& c, int W, int H)
    {
        int removed = 0;
        for (bool changed = true; changed;)
        {
            changed = false;
            for (int j = 0; j + 1 < H; ++j)
                for (int i = 0; i + 1 < W; ++i)
                {
                    auto& a = c[static_cast<std::size_t>(j) * W + i];
                    auto& b = c[static_cast<std::size_t>(j) * W + i + 1];
                    auto& d = c[static_cast<std::size_t>(j + 1) * W + i];
                    auto& e = c[static_cast<std::size_t>(j + 1) * W + i + 1];
                    if (a and e and not b and not d) { e = 0; changed = true; ++removed; }
                    else if (b and d and not a and not e) { d = 0; changed = true; ++removed; }
                }
        }
        return removed;
    }

    /// The outer boundary of the region as a CCW ring of cell-corner points (window cell units). The region
    /// is 4-connected, hole-free and pinch-free by now, so each corner on the boundary has exactly one way on.
    static std::vector<Eigen::Vector2f> trace_outer(const std::vector<std::uint8_t>& c, int W, int H)
    {
        const auto at = [&](int i, int j) -> bool
        { return i >= 0 and j >= 0 and i < W and j < H and c[static_cast<std::size_t>(j) * W + i] != 0; };
        // Directed boundary edges, interior on the LEFT (CCW), keyed by their start corner.
        std::map<std::pair<int, int>, std::pair<int, int>> next;
        std::pair<int, int> first{-1, -1};
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i)
            {
                if (not at(i, j)) continue;
                if (not at(i, j - 1)) next[{i, j}]         = {i + 1, j};       // bottom, left→right
                if (not at(i + 1, j)) next[{i + 1, j}]     = {i + 1, j + 1};   // right, bottom→top
                if (not at(i, j + 1)) next[{i + 1, j + 1}] = {i, j + 1};       // top, right→left
                if (not at(i - 1, j)) next[{i, j + 1}]     = {i, j};           // left, top→bottom
                if (first.first < 0 and not at(i, j - 1)) first = {i, j};      // lowest row's first bottom edge
            }
        std::vector<Eigen::Vector2f> ring;
        if (first.first < 0) return ring;
        auto cur = first;
        const std::size_t cap = next.size() + 1;
        do
        {
            ring.emplace_back(static_cast<float>(cur.first), static_cast<float>(cur.second));
            const auto it = next.find(cur);
            if (it == next.end()) return {};
            cur = it->second;
        } while (cur != first and ring.size() <= cap);
        if (cur != first) return {};
        return ring;
    }

    static std::vector<Eigen::Vector2f> merge_collinear(const std::vector<Eigen::Vector2f>& r)
    {
        std::vector<Eigen::Vector2f> out;
        const std::size_t n = r.size();
        for (std::size_t k = 0; k < n; ++k)
        {
            const auto& a = r[(k + n - 1) % n];
            const auto& b = r[k];
            const auto& c = r[(k + 1) % n];
            const float cr = (b.x() - a.x()) * (c.y() - b.y()) - (b.y() - a.y()) * (c.x() - b.x());
            if (std::abs(cr) > 1e-12f) out.push_back(b);
        }
        return out;
    }

    static void dp(const std::vector<Eigen::Vector2f>& p, std::size_t a, std::size_t b, float eps,
                   std::vector<char>& keep)
    {
        if (b <= a + 1) return;
        const Eigen::Vector2f ab = p[b] - p[a];
        const float L = ab.norm();
        float best = -1.f; std::size_t bi = a;
        for (std::size_t k = a + 1; k < b; ++k)
        {
            const Eigen::Vector2f ap = p[k] - p[a];
            const float dist = L > 1e-9f ? std::abs(ab.x() * ap.y() - ab.y() * ap.x()) / L : ap.norm();
            if (dist > best) { best = dist; bi = k; }
        }
        if (best > eps)
        {
            keep[bi] = 1;
            dp(p, a, bi, eps, keep);
            dp(p, bi, b, eps, keep);
        }
    }
    /// Closed-ring Douglas–Peucker: split at vertex 0 and the vertex farthest from it.
    static std::vector<Eigen::Vector2f> douglas_peucker_closed(const std::vector<Eigen::Vector2f>& r, float eps)
    {
        const std::size_t n = r.size();
        if (n < 4) return r;
        std::size_t far = 0; float fd = -1.f;
        for (std::size_t k = 1; k < n; ++k)
            if (const float d = (r[k] - r[0]).squaredNorm(); d > fd) { fd = d; far = k; }
        std::vector<Eigen::Vector2f> p(r.begin(), r.end());
        p.push_back(r[0]);                       // close the ring: index n is vertex 0 again
        std::vector<char> keep(p.size(), 0);
        keep[0] = keep[far] = keep[n] = 1;
        dp(p, 0, far, eps, keep);
        dp(p, far, n, eps, keep);
        std::vector<Eigen::Vector2f> out;
        for (std::size_t k = 0; k < n; ++k)
            if (keep[k]) out.push_back(p[k]);
        return out;
    }
};

}   // namespace rc::freespace
