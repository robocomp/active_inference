// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/layout_explorer_selftest.cpp src/layout_explorer.cpp \
//       src/wall_map.cpp src/wall_segmenter.cpp ../common/status_stream/status_stream.cpp -o /tmp/les && /tmp/les
//
// THE PORT MUST CHOOSE WHAT THE BENCH CHOSE. src/layout_explorer.{h,cpp} moved the offline explorer of
// tools/wall_slam_selftest.cpp (run_explore, EIG branch) into a live unit. This test builds a real WallMap
// state — the agent's segmenter and wall map fed by ray-cast scans of a synthetic L room along a drive —
// and at several moments asks BOTH for the next target:
//   · `offline::` below is a FROZEN COPY of the bench code (struct Explorer, eig_of, the candidate loop and
//     the top-16 reachability fallback), with the truth polygon as feasibility and origin = start pose so
//     the bench's to_map/from_map are the identity — exactly the bench's situation;
//   · rc::layout::plan with the same feasibility polygon, clearance and polygon.
// PASS = identical target and identical per-candidate gain, bit for bit, at every state; and the live-only
// behaviour (a robot standing off the feasible grid) still yields a target where the bench would park.
#include "layout_explorer.h"
#include "corner_visibility.h"
#include "line_fit.h"
#include "wall_map.h"
#include "wall_segmenter.h"

#include <cmath>
#include <cstdio>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

namespace
{
    using Poly = std::vector<Eigen::Vector2f>;
    constexpr float kPi = static_cast<float>(M_PI);
    int fails = 0;

    // ═══ FROZEN COPY of tools/wall_slam_selftest.cpp (2026-10-09, HEAD a3105d6..df95b97) ═══════════════
    namespace offline
    {
        float point_to_segment(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
        {
            const Eigen::Vector2f ab = b - a;
            const float t = std::clamp((p - a).dot(ab) / std::max(1e-9f, ab.squaredNorm()), 0.f, 1.f);
            return (p - (a + t * ab)).norm();
        }
        float point_to_poly(const Eigen::Vector2f& p, const Poly& poly)
        {
            float best = 1e9f;
            for (size_t i = 0; i < poly.size(); ++i)
                best = std::min(best, point_to_segment(p, poly[i], poly[(i + 1) % poly.size()]));
            return best;
        }
        struct Explorer
        {
            const Poly& room;
            float cell = 0.30f, clearance = 0.35f;
            float x0, y0; int nx, ny;
            std::vector<char> free_;
            explicit Explorer(const Poly& r) : room(r)
            {
                float xa = 1e9f, ya = 1e9f, xb = -1e9f, yb = -1e9f;
                for (const auto& v : room) { xa = std::min(xa, v.x()); ya = std::min(ya, v.y()); xb = std::max(xb, v.x()); yb = std::max(yb, v.y()); }
                x0 = xa; y0 = ya;
                nx = static_cast<int>((xb - xa) / cell) + 1;
                ny = static_cast<int>((yb - ya) / cell) + 1;
                free_.assign(static_cast<size_t>(nx * ny), 0);
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                    {
                        const Eigen::Vector2f p = at(i, j);
                        if (rc::corner_visibility::point_in_polygon(p, room) and point_to_poly(p, room) > clearance)
                            free_[static_cast<size_t>(j * nx + i)] = 1;
                    }
            }
            Eigen::Vector2f at(int i, int j) const { return {x0 + (i + 0.5f) * cell, y0 + (j + 0.5f) * cell}; }
            bool is_free(int i, int j) const
            { return i >= 0 and i < nx and j >= 0 and j < ny and free_[static_cast<size_t>(j * nx + i)] != 0; }
            std::pair<int,int> cell_of(const Eigen::Vector2f& p) const
            { return {static_cast<int>((p.x() - x0) / cell), static_cast<int>((p.y() - y0) / cell)}; }
            std::vector<Eigen::Vector2f> astar(const Eigen::Vector2f& from, const Eigen::Vector2f& to) const
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
                        if (k >= 4 and (not is_free(ui, vj) or not is_free(vi, uj))) continue;
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
        };
        float entropy_nats(float lodds)
        {
            const float pp = std::clamp(1.f / (1.f + std::exp(-lodds)), 1e-6f, 1.f - 1e-6f);
            return -pp * std::log(pp) - (1.f - pp) * std::log(1.f - pp);
        }
        /// eig_of, with to_map = identity (origin = start pose).
        float eig_of(const rc::wallmap::WallMap& Rmap, const Eigen::Vector2f& v)
        {
            const auto poly = Rmap.build_polygon();
            if (not poly.closed or poly.verts.size() < 3) return 0.f;
            const auto& fg = Rmap.fgrid;
            if (not fg.ready()) return 0.f;
            const int NB = 36;
            const bool occl_pred = true;
            const float sig2 = Rmap.params.obs_sigma * Rmap.params.obs_sigma;
            std::unordered_map<std::uint64_t, Eigen::Matrix2f> J;
            float gain = 0.f;
            for (int b = 0; b < NB; ++b)
            {
                const float a = 2.f * kPi * static_cast<float>(b) / static_cast<float>(NB);
                const Eigen::Vector2f dir(std::cos(a), std::sin(a));
                float r_wall = Rmap.params.sensor_range; int hit = -1;
                for (size_t e = 0; e < poly.verts.size(); ++e)
                    if (const auto t = rc::corner_visibility::ray_segment_t(
                            v, dir, poly.verts[e], poly.verts[(e + 1) % poly.verts.size()]);
                        t and *t > 1e-3f and *t < r_wall) { r_wall = *t; hit = static_cast<int>(e); }
                const float r_pred = Rmap.params.sensor_range;
                bool blocked = false;
                for (float rr = fg.cell; rr < r_pred; rr += fg.cell)
                {
                    const Eigen::Vector2f q = v + dir * rr;
                    const int i = static_cast<int>((q.x() - fg.x0) / fg.cell);
                    const int j = static_cast<int>((q.y() - fg.y0) / fg.cell);
                    if (not fg.in(i, j)) continue;
                    const float l = fg.lodds[static_cast<size_t>(fg.idx(i, j))];
                    const float p = 1.f / (1.f + std::exp(-l));
                    const float h_now = entropy_nats(l);
                    const float h_after = p * entropy_nats(std::min(4.f, l + 1.0f))
                                        + (1.f - p) * entropy_nats(std::max(-4.f, l - 0.4f));
                    gain += std::max(0.f, h_now - h_after);
                    if (occl_pred and fg.is_occupied(i, j)) { blocked = true; break; }
                }
                if (not blocked and hit >= 0 and r_wall < r_pred
                    and static_cast<size_t>(hit) < poly.wall_of_edge.size())
                {
                    const auto* w = Rmap.find(poly.wall_of_edge[static_cast<size_t>(hit)]);
                    if (w != nullptr)
                    {
                        const Eigen::Vector2f q = v + dir * r_wall;
                        const float sc_ = w->tangent().dot(q);
                        Eigen::Vector2f h(sc_, -1.f);
                        // ★THE ONE EDIT TO THE FROZEN COPY: the bench's `J[w->id] += ...` reads an
                        // uninitialised Eigen::Matrix2f (operator[] value-initialises; Eigen does not zero),
                        // so its value depends on the allocator's leftovers and two identical calls disagree
                        // (UB — no deterministic reference exists to compare against). Zero-initialised
                        // here exactly as in the port; the bench itself still carries the bug (reported).
                        J.try_emplace(w->id, Eigen::Matrix2f::Zero()).first->second += (h * h.transpose()) / sig2;
                    }
                }
            }
            for (const auto& [wid, Jw] : J)
            {
                const auto* w = Rmap.find(wid);
                if (w == nullptr) continue;
                const Eigen::Matrix2f L0 = w->information;
                const float d0 = L0.determinant();
                if (not std::isfinite(d0) or d0 <= 1e-9f) continue;
                const float d1 = (L0 + Jw).determinant();
                if (std::isfinite(d1) and d1 > d0) gain += 0.5f * std::log(d1 / d0);
            }
            for (const auto& c : poly.corners)
            {
                if (not std::isfinite(c.sigma) or c.sigma <= Rmap.params.publish_corner_sigma) continue;
                const auto* wa = Rmap.find(c.wall_a); const auto* wb = Rmap.find(c.wall_b);
                if (wa == nullptr or wb == nullptr) continue;
                const auto ja = J.find(wa->id), jb = J.find(wb->id);
                if (ja == J.end() and jb == J.end()) continue;
                rc::wallmap::WallLandmark a2 = *wa, b2 = *wb;
                if (ja != J.end()) a2.information += ja->second;
                if (jb != J.end()) b2.information += jb->second;
                const auto c2 = rc::wallmap::WallMap::intersect_walls(a2, b2, c.inferred);
                if (std::isfinite(c2.sigma) and c2.sigma > 0.f and c2.sigma < c.sigma)
                    gain += std::log(c.sigma / c2.sigma);
            }
            return gain;
        }
        /// The replan body, EIG branch, as run_explore has it. Returns best_v, or nullopt for "path empty".
        std::optional<Eigen::Vector2f> replan(const rc::wallmap::WallMap& Rmap, const Explorer& ex,
                                              const Eigen::Vector2f& tru)
        {
            std::vector<std::pair<float, Eigen::Vector2f>> cands;
            for (int i = 0; i < ex.nx; i += 2)
                for (int j = 0; j < ex.ny; j += 2)
                {
                    if (not ex.is_free(i, j)) continue;
                    const Eigen::Vector2f v = ex.at(i, j);
                    float sc = eig_of(Rmap, v);
                    if (sc <= 0.f) continue;
                    sc /= (1.0f + (v - tru).norm());
                    cands.emplace_back(sc, v);
                }
            std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            if (cands.size() > 16) cands.resize(16);
            for (const auto& [sc, v] : cands)
            {
                auto path = ex.astar(tru, v);
                if (path.size() > 1) return v;
            }
            return std::nullopt;
        }
    }   // namespace offline
    // ═══ end of the frozen copy ═══════════════════════════════════════════════════════════════════════

    std::optional<float> ray_t(const Eigen::Vector2f& o, const Eigen::Vector2f& d, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    { return rc::corner_visibility::ray_segment_t(o, d, a, b); }

    /// One scan in the ROBOT frame (theta = 0 throughout, so robot frame = map frame shifted).
    std::vector<Eigen::Vector2f> scan(const Poly& room, const Eigen::Vector2f& o, std::mt19937& rng)
    {
        std::normal_distribution<float> nz(0.f, 0.02f);
        std::vector<Eigen::Vector2f> pts;
        for (int b = 0; b < 360; ++b)
        {
            const float a = 2.f * kPi * b / 360.f;
            const Eigen::Vector2f u(std::cos(a), std::sin(a));
            float t = 1e9f;
            for (size_t e = 0; e < room.size(); ++e)
                if (const auto tt = ray_t(o, u, room[e], room[(e + 1) % room.size()]); tt and *tt < t) t = *tt;
            if (t < 12.f) pts.push_back(u * (t + nz(rng)));
        }
        return pts;
    }
}   // namespace

int main()
{
    // The L room, expressed so the START pose is the origin (the bench's map frame = start-relative).
    const Eigen::Vector2f start(4.0f, 1.0f);
    Poly room = {{0.f, 0.f}, {6.f, 0.f}, {6.f, 3.f}, {3.f, 3.f}, {3.f, 6.f}, {0.f, 6.f}};
    for (auto& v : room) v -= start;

    rc::wallmap::WallMap map;
    map.params.obs_sigma = 0.05f;
    map.params.huber_delta = 0.15f;
    rc::wallseg::Params sp;
    sp.sensor_sigma = 0.02f;
    std::mt19937 rng(3);

    Poly drive;
    for (float x = 0.f; x >= -2.5f; x -= 0.25f) drive.emplace_back(x, 0.f);
    for (float y = 0.25f; y <= 3.0f; y += 0.25f) drive.emplace_back(-2.5f, y);

    rc::layout::ExplorerParams prm;            // the offline constants: cell 0.30, clearance 0.35, 36 beams
    const rc::layout::FeasibleGrid grid(room, prm.cell, prm.clearance);
    const offline::Explorer ex(room);
    int compared = 0;
    for (size_t f = 0; f < drive.size(); ++f)
    {
        const Eigen::Vector2f tru = drive[f];
        const auto pts = scan(room, tru, rng);
        if (map.walls.empty())
        {   // the bench's model-first init: the first scan's OBB (theta = 0 here, so axis-aligned is the OBB)
            float lo0 = 1e9f, hi0 = -1e9f, lo1 = 1e9f, hi1 = -1e9f;
            for (const auto& p : pts) { lo0 = std::min(lo0, p.x()); hi0 = std::max(hi0, p.x()); lo1 = std::min(lo1, p.y()); hi1 = std::max(hi1, p.y()); }
            map.initialize_rect({{lo0, lo1}, {hi0, lo1}, {hi0, hi1}, {lo0, hi1}});
        }
        const auto seg = rc::wallseg::segment(pts, sp, rng);
        const Eigen::Matrix3f pcov = Eigen::Vector3f(0.05f * 0.05f, 0.05f * 0.05f, 0.03f * 0.03f).asDiagonal();
        map.observe(seg, pts, Eigen::VectorXf(), Eigen::Vector3f(tru.x(), tru.y(), 0.f), pcov,
                    static_cast<std::int64_t>(f) * 50);
        map.merge_indistinguishable();

        if (f % 3 != 0 and f + 1 != drive.size()) continue;
        ++compared;
        const auto poly = map.build_polygon();
        // per-candidate gain, bit for bit
        int gain_mismatch = 0, n = 0;
        for (int i = 0; i < grid.nx; i += 2)
            for (int j = 0; j < grid.ny; j += 2)
            {
                if (not grid.is_free(i, j)) continue;
                const Eigen::Vector2f v = grid.at(i, j);
                ++n;
                if (offline::eig_of(map, v) != rc::layout::expected_information_gain(map, poly, v, prm)) ++gain_mismatch;
            }
        const auto ref  = offline::replan(map, ex, tru);
        const auto live = rc::layout::plan(map, poly, grid, tru, prm);
        const bool same = ref.has_value() == live.ok and (not ref.has_value() or (*ref - live.target).norm() == 0.f);
        const bool grid_same = grid.nx == ex.nx and grid.ny == ex.ny and grid.free_count() == static_cast<int>(std::ranges::count(ex.free_, 1));
        std::printf("frame %2zu robot (%5.2f,%5.2f) walls=%zu closed=%d | offline %s | port %s (%s, %.3f nats/m, %d/%d with gain) | "
                    "gain mismatches %d/%d %s\n",
                    f, tru.x(), tru.y(), map.walls.size(), poly.closed,
                    ref ? std::format("({:.2f},{:.2f})", ref->x(), ref->y()).c_str() : "none",
                    live.ok ? std::format("({:.2f},{:.2f})", live.target.x(), live.target.y()).c_str() : "none",
                    live.why.c_str(), live.score, live.with_gain, live.candidates, gain_mismatch, n,
                    (same and gain_mismatch == 0 and grid_same) ? "PASS" : "FAIL");
        if (not same or gain_mismatch != 0 or not grid_same) ++fails;
        if (not live.ok and poly.closed) { ++fails; std::printf("  FAIL: closed polygon but no target\n"); }
    }
    // LIVE-ONLY: a robot standing where no feasible standpoint is (hugging a wall). The bench returns nothing
    // there (its A* refuses a non-free start); the port walks from the nearest feasible cell instead.
    {
        const Eigen::Vector2f hug = room[0] + Eigen::Vector2f(0.15f, 0.15f);
        const auto poly = map.build_polygon();
        const auto ref  = offline::replan(map, ex, hug);
        const auto live = rc::layout::plan(map, poly, grid, hug, prm);
        const bool ok = live.ok and grid.is_free(grid.cell_of(live.start).first, grid.cell_of(live.start).second);
        std::printf("wall-hugging start (%.2f,%.2f): offline %s, port %s from (%.2f,%.2f) %s\n", hug.x(), hug.y(),
                    ref ? "a target" : "NOTHING (parks)", live.ok ? "a target" : "nothing",
                    live.start.x(), live.start.y(), ok ? "PASS" : "FAIL");
        if (not ok) ++fails;
    }
    std::printf("%d states compared. %s: %d failure(s)\n", compared, fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
