/*
 * grid_planner.cpp — see grid_planner.h
 */

#include "grid_planner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <queue>

namespace rc
{

namespace
{
bool point_in_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p)
{
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        if (((poly[i].y() > p.y()) != (poly[j].y() > p.y())) and
            (p.x() < (poly[j].x() - poly[i].x()) * (p.y() - poly[i].y()) / (poly[j].y() - poly[i].y()) + poly[i].x()))
            inside = not inside;
    return inside;
}
}  // namespace

void GridPlanner::rebuild_offsets()
{
    // Only recompute when the geometry that determines coverage actually changed — this is per-heading
    // rasterisation of the footprint and it would otherwise dominate a per-cycle rebuild.
    if (offsets_.size() == kHeadings and std::abs(offsets_cell_ - cell_) < 1e-6f
        and std::abs(offsets_margin_ - params.safety_margin_m) < 1e-6f)
        return;
    footprint.set_safety_margin(params.safety_margin_m);
    offsets_.assign(kHeadings, {});
    for (int h = 0; h < kHeadings; ++h)
        offsets_[h] = footprint.cell_offsets(cell_, 2.0f * static_cast<float>(M_PI) * h / kHeadings);
    offsets_cell_ = cell_;
    offsets_margin_ = params.safety_margin_m;
}

void GridPlanner::set_world(const std::vector<Eigen::Vector2f>& room_polygon,
                            const std::vector<std::vector<Eigen::Vector2f>>& obstacles)
{
    cell_ = std::max(0.02f, params.cell_size_m);
    dist_valid_ = false;          // the world changed; any cached distance field describes the old one
    if (room_polygon.size() < 3) { w_ = h_ = 0; occ_.clear(); dist_.clear(); return; }

    float xmn = room_polygon[0].x(), xmx = xmn, ymn = room_polygon[0].y(), ymx = ymn;
    for (const auto& p : room_polygon)
    { xmn = std::min(xmn, p.x()); xmx = std::max(xmx, p.x()); ymn = std::min(ymn, p.y()); ymx = std::max(ymx, p.y()); }
    // A margin of one circumscribed radius so the footprint of a robot near the boundary is representable.
    const float pad = footprint.circumscribed_radius() + params.safety_margin_m + cell_;
    xmin_ = xmn - pad; ymin_ = ymn - pad;
    w_ = std::max(1, static_cast<int>(std::ceil((xmx + pad - xmin_) / cell_)));
    h_ = std::max(1, static_cast<int>(std::ceil((ymx + pad - ymin_) / cell_)));
    occ_.assign(static_cast<std::size_t>(w_) * h_, 0);
    last_obstacle_count_ = static_cast<int>(obstacles.size());

    // OUTSIDE the room is occupied. Walls therefore need no separate representation and cannot be
    // accidentally omitted — a cell is free only if it is positively inside the room polygon.
    //
    // ...UNLESS the polygon turns out to be unusable, in which case this single input would brick the whole
    // planner: every cell reads "outside", every goal is infeasible, and the robot never moves again. That is
    // exactly what was observed live — 27018 of 27018 cells occupied — while the same code on a known-good
    // polygon marks ~39%. A planner must not be one bad input away from total failure, so the room mask is
    // applied only if it leaves a plausible amount of free space; otherwise it is DISCARDED and planning
    // continues on the obstacle set alone. That is strictly better than refusing to move: obstacles still
    // block, the robot is merely no longer confined by a boundary we cannot trust. It is loud about it,
    // because navigating without a room boundary is a degraded mode, not a normal one.
    long outside = 0;
    for (int iy = 0; iy < h_; ++iy)
        for (int ix = 0; ix < w_; ++ix)
            if (not point_in_polygon(room_polygon, cell_to_world(ix, iy)))
                { occ_[idx(ix, iy)] = 1; ++outside; }
    room_mask_usable_ = outside < static_cast<long>(0.95 * w_ * h_);
    if (not room_mask_usable_)
    {
        std::fill(occ_.begin(), occ_.end(), 0);          // discard it; obstacles are re-applied below
        static int rc = 0;
        if ((rc++ % 50) == 0)
            std::printf("[grid-world] ROOM POLYGON UNUSABLE — %zu verts, bbox x[%.2f,%.2f] y[%.2f,%.2f] marks "
                        "%ld/%d cells (%.0f%%) as outside. Discarding it and planning on obstacles only "
                        "(DEGRADED: no room boundary). Check delimiting_polygon_x/y on the room node.\n",
                        room_polygon.size(), xmn, xmx, ymn, ymx, outside, w_ * h_,
                        100.0 * outside / std::max(1, w_ * h_));
        std::fflush(stdout);
    }

    // Obstacles at TRUE extent — no inflation anywhere. Rasterised over each polygon's bbox; a cell counts as
    // occupied if its centre is inside. Sub-cell slivers are missed by design at this resolution, which is why
    // the safety margin exists and why it is ONE number rather than a per-stage guess.
    for (const auto& poly : obstacles)
    {
        if (poly.size() < 3) continue;
        float pxn = poly[0].x(), pxx = pxn, pyn = poly[0].y(), pyx = pyn;
        for (const auto& p : poly)
        { pxn = std::min(pxn, p.x()); pxx = std::max(pxx, p.x()); pyn = std::min(pyn, p.y()); pyx = std::max(pyx, p.y()); }
        int ix0, iy0, ix1, iy1;
        world_to_cell({pxn, pyn}, ix0, iy0);
        world_to_cell({pxx, pyx}, ix1, iy1);
        for (int iy = std::max(0, iy0); iy <= std::min(h_ - 1, iy1); ++iy)
            for (int ix = std::max(0, ix0); ix <= std::min(w_ - 1, ix1); ++ix)
                if (not occ_[idx(ix, iy)] and point_in_polygon(poly, cell_to_world(ix, iy)))
                    occ_[idx(ix, iy)] = 1;
    }
    rebuild_offsets();
}

bool GridPlanner::world_to_cell(const Eigen::Vector2f& p, int& ix, int& iy) const
{
    ix = static_cast<int>(std::floor((p.x() - xmin_) / cell_));
    iy = static_cast<int>(std::floor((p.y() - ymin_) / cell_));
    return in_bounds(ix, iy);
}

Eigen::Vector2f GridPlanner::cell_to_world(int ix, int iy) const
{
    return {xmin_ + (static_cast<float>(ix) + 0.5f) * cell_, ymin_ + (static_cast<float>(iy) + 0.5f) * cell_};
}

long GridPlanner::occupied_cells() const
{
    long n = 0;
    for (const auto v : occ_) n += v;
    return n;
}

// ── Exact Euclidean distance transform (Felzenszwalb & Huttenlocher) ──────────────────────────────
// Two O(n) passes of a 1-D squared-distance transform — down the columns, then across the rows —
// give the EXACT squared distance to the nearest occupied cell. The 1-D transform works by finding
// the lower envelope of the parabolas rooted at each sample; `v` holds the parabolas currently on
// the envelope and `z` the boundaries between them.
namespace
{
void dt_1d(const std::vector<float>& f, std::vector<float>& d, int n,
           std::vector<int>& v, std::vector<float>& z)
{
    constexpr float kInf = std::numeric_limits<float>::max();
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q)
    {
        const auto sq = [](float a) { return a * a; };
        float s = ((f[q] + sq(static_cast<float>(q))) - (f[v[k]] + sq(static_cast<float>(v[k]))))
                / (2.f * static_cast<float>(q) - 2.f * static_cast<float>(v[k]));
        while (k > 0 and s <= z[k])
        {
            --k;
            s = ((f[q] + sq(static_cast<float>(q))) - (f[v[k]] + sq(static_cast<float>(v[k]))))
              / (2.f * static_cast<float>(q) - 2.f * static_cast<float>(v[k]));
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }
    k = 0;
    for (int q = 0; q < n; ++q)
    {
        while (z[k + 1] < static_cast<float>(q)) ++k;
        const float dq = static_cast<float>(q - v[k]);
        d[q] = dq * dq + f[v[k]];
    }
}
}  // namespace

void GridPlanner::build_distance_field() const
{
    if (dist_valid_) return;
    if (w_ <= 0 or h_ <= 0) { dist_.clear(); dist_valid_ = true; return; }

    constexpr float kInf = std::numeric_limits<float>::max();
    const std::size_t n = static_cast<std::size_t>(w_) * static_cast<std::size_t>(h_);
    std::vector<float> g(n);
    for (std::size_t i = 0; i < n; ++i) g[i] = occ_[i] ? 0.f : kInf;

    const int maxdim = std::max(w_, h_);
    std::vector<float> f(maxdim), d(maxdim), z(maxdim + 1);
    std::vector<int> v(maxdim);

    for (int ix = 0; ix < w_; ++ix)                       // columns
    {
        for (int iy = 0; iy < h_; ++iy) f[iy] = g[idx(ix, iy)];
        dt_1d(f, d, h_, v, z);
        for (int iy = 0; iy < h_; ++iy) g[idx(ix, iy)] = d[iy];
    }
    for (int iy = 0; iy < h_; ++iy)                       // rows
    {
        for (int ix = 0; ix < w_; ++ix) f[ix] = g[idx(ix, iy)];
        dt_1d(f, d, w_, v, z);
        for (int ix = 0; ix < w_; ++ix) g[idx(ix, iy)] = d[ix];
    }

    dist_.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        dist_[i] = (g[i] >= kInf) ? 1e6f : std::sqrt(g[i]) * cell_;   // cells -> metres
    dist_valid_ = true;
}

float GridPlanner::distance_at(const Eigen::Vector2f& pos_room) const
{
    if (w_ <= 0 or h_ <= 0) return 1e6f;      // no world: wide open, never a spurious obstacle
    build_distance_field();
    if (dist_.empty()) return 1e6f;

    // Bilinear interpolation on cell CENTRES, so the field is continuous rather than blocky.
    const float fx = (pos_room.x() - xmin_) / cell_ - 0.5f;
    const float fy = (pos_room.y() - ymin_) / cell_ - 0.5f;
    const int ix = static_cast<int>(std::floor(fx));
    const int iy = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(ix);
    const float ty = fy - static_cast<float>(iy);
    // Outside the grid is outside the room, which set_world already marked occupied — so clamping to the
    // border reports the border cell's distance rather than inventing free space beyond the world.
    const auto at = [this](int cx, int cy)
    {
        return dist_[idx(std::clamp(cx, 0, w_ - 1), std::clamp(cy, 0, h_ - 1))];
    };
    const float d00 = at(ix, iy),     d10 = at(ix + 1, iy);
    const float d01 = at(ix, iy + 1), d11 = at(ix + 1, iy + 1);
    return (1.f - tx) * (1.f - ty) * d00 + tx * (1.f - ty) * d10
         + (1.f - tx) * ty * d01 + tx * ty * d11;
}

Eigen::Vector2f GridPlanner::distance_gradient_at(const Eigen::Vector2f& pos_room, float fd_cells) const
{
    const float e = std::max(0.25f, fd_cells) * cell_;
    const float dx = distance_at({pos_room.x() + e, pos_room.y()}) - distance_at({pos_room.x() - e, pos_room.y()});
    const float dy = distance_at({pos_room.x(), pos_room.y() + e}) - distance_at({pos_room.x(), pos_room.y() - e});
    return {dx / (2.f * e), dy / (2.f * e)};
}

bool GridPlanner::cell_free(int ix, int iy, int h) const
{
    if (offsets_.size() != kHeadings) return false;
    for (const auto& o : offsets_[h])
    {
        const int nx = ix + o.x(), ny = iy + o.y();
        if (not in_bounds(nx, ny)) return false;      // footprint off the map == unsafe
        if (occ_[idx(nx, ny)]) return false;
    }
    return true;
}

bool GridPlanner::pose_free(const Eigen::Vector2f& pos_room, float theta) const
{
    int ix, iy;
    if (not world_to_cell(pos_room, ix, iy)) return false;
    float t = std::fmod(theta, 2.0f * static_cast<float>(M_PI));
    if (t < 0) t += 2.0f * static_cast<float>(M_PI);
    const int h = static_cast<int>(std::lround(t / (2.0f * static_cast<float>(M_PI)) * kHeadings)) % kHeadings;
    return cell_free(ix, iy, h);
}

std::optional<Eigen::Vector2f> GridPlanner::nearest_free(const Eigen::Vector2f& pos_room, float theta,
                                                         float max_radius_m) const
{
    if (pose_free(pos_room, theta)) return pos_room;
    const int max_r = static_cast<int>(std::ceil(max_radius_m / cell_));
    int cx, cy;
    world_to_cell(pos_room, cx, cy);
    // Expanding ring search on the same predicate the planner uses, so a repaired pose is feasible by
    // construction — the old repair used its own clearance number and could return a pose the controller
    // then refused, leaving the robot hunting at a goal it was never allowed to reach.
    for (int r = 1; r <= max_r; ++r)
    {
        std::optional<Eigen::Vector2f> best;
        float best_d2 = std::numeric_limits<float>::max();
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
            {
                if (std::max(std::abs(dx), std::abs(dy)) != r) continue;   // ring only
                const int nx = cx + dx, ny = cy + dy;
                if (not in_bounds(nx, ny)) continue;
                const auto w = cell_to_world(nx, ny);
                if (not pose_free(w, theta)) continue;
                const float d2 = (w - pos_room).squaredNorm();
                if (d2 < best_d2) { best_d2 = d2; best = w; }
            }
        if (best.has_value()) return best;
    }
    return std::nullopt;
}

std::optional<std::vector<Eigen::Vector2f>> GridPlanner::plan(const Eigen::Vector2f& start_room,
                                                              const Eigen::Vector2f& goal_room)
{
    last_failure_.clear();
    if (w_ <= 0 or h_ <= 0) { last_failure_ = "grid not built (no room polygon)"; return std::nullopt; }
    rebuild_offsets();

    int sx, sy, gx, gy;
    if (not world_to_cell(start_room, sx, sy))
    { last_failure_ = std::format("start ({:.2f},{:.2f}) outside the grid", start_room.x(), start_room.y()); return std::nullopt; }
    if (not world_to_cell(goal_room, gx, gy))
    { last_failure_ = std::format("goal ({:.2f},{:.2f}) outside the grid", goal_room.x(), goal_room.y()); return std::nullopt; }

    // The goal must be feasible at SOME heading, else no amount of searching helps and the caller should be
    // told to move the target rather than left to retry forever.
    bool goal_ok = false;
    for (int h = 0; h < kHeadings and not goal_ok; ++h) goal_ok = cell_free(gx, gy, h);
    if (not goal_ok)
    {
        // Report the MAP state with it. "goal infeasible" alone cannot distinguish a genuinely tight spot
        // from a map that is occupied nearly everywhere, and those need opposite fixes — move the target vs
        // fix the obstacle source. The room-boundary-only baseline for this apartment is ~39% occupied.
        last_failure_ = std::format("goal ({:.2f},{:.2f}) is not footprint-feasible at any heading "
                                    "(robot needs {:.2f} m of width) | grid {}x{} cells, {} occupied ({:.0f}%), "
                                    "{} obstacle polygons",
                                    goal_room.x(), goal_room.y(), 2.f * footprint.inscribed_radius(),
                                    w_, h_, occupied_cells(),
                                    100.0 * static_cast<double>(occupied_cells()) / std::max(1, w_ * h_),
                                    last_obstacle_count_);
        return std::nullopt;
    }

    const int n_cells = w_ * h_;
    const int n_states = n_cells * kHeadings;
    auto sid = [&](int ix, int iy, int h) { return (iy * w_ + ix) * kHeadings + h; };

    std::vector<float> g(n_states, std::numeric_limits<float>::infinity());
    std::vector<int> parent(n_states, -1);
    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<>> open;

    const auto goal_w = cell_to_world(gx, gy);
    auto heur = [&](int ix, int iy) { return (cell_to_world(ix, iy) - goal_w).norm(); };

    // Seed every heading the start is feasible at.
    bool seeded = false;
    for (int h = 0; h < kHeadings; ++h)
        if (cell_free(sx, sy, h))
        { g[sid(sx, sy, h)] = 0.f; open.push({heur(sx, sy), sid(sx, sy, h)}); seeded = true; }

    // START IN COLLISION. Seeding the start's headings is not enough on its own: if the robot is properly
    // inside an obstacle then its NEIGHBOURS are infeasible too, so the search cannot take a single step and
    // dies after 8 expansions — which is precisely how the robot stayed bricked. So walk out first: a BFS that
    // ignores collision entirely finds the nearest footprint-feasible cell, and the plan starts from there
    // with the original position prepended. "Get out, then navigate" — the same thing a person would do.
    const bool start_in_collision = not seeded;
    Eigen::Vector2f escape_from = start_room;
    if (start_in_collision)
    {
        if (not params.allow_start_in_collision)
        { last_failure_ = "start is not footprint-feasible"; return std::nullopt; }
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(n_cells), 0);
        std::queue<std::pair<int, int>> q;
        q.push({sx, sy});
        seen[idx(sx, sy)] = 1;
        int ex = -1, ey = -1;
        while (not q.empty() and ex < 0)
        {
            const auto [cx, cy] = q.front();
            q.pop();
            constexpr int NX[4] = {1, -1, 0, 0}, NY[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4; ++d)
            {
                const int nx = cx + NX[d], ny = cy + NY[d];
                if (not in_bounds(nx, ny) or seen[idx(nx, ny)]) continue;
                seen[idx(nx, ny)] = 1;
                for (int h = 0; h < kHeadings; ++h)
                    if (cell_free(nx, ny, h)) { ex = nx; ey = ny; break; }
                if (ex >= 0) break;
                q.push({nx, ny});
            }
        }
        if (ex < 0)
        { last_failure_ = "start is inside an obstacle and NO footprint-feasible cell is reachable from it"; return std::nullopt; }
        escape_from = cell_to_world(ex, ey);
        sx = ex; sy = ey;
        for (int h = 0; h < kHeadings; ++h)
            if (cell_free(sx, sy, h))
            { g[sid(sx, sy, h)] = 0.f; open.push({heur(sx, sy), sid(sx, sy, h)}); }
    }

    constexpr int DX[kHeadings] = {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr int DY[kHeadings] = {0, 1, 1, 1, 0, -1, -1, -1};
    int goal_state = -1, expansions = 0;
    while (not open.empty())
    {
        const auto [f, s] = open.top();
        open.pop();
        const int h = s % kHeadings, cell = s / kHeadings;
        const int ix = cell % w_, iy = cell / w_;
        if (f > g[s] + heur(ix, iy) + 1e-6f) continue;
        if (ix == gx and iy == gy) { goal_state = s; break; }
        if (++expansions > params.max_expansions)
        { last_failure_ = std::format("search exceeded {} expansions", params.max_expansions); return std::nullopt; }

        for (int nh = 0; nh < kHeadings; ++nh)
        {
            const int nx = ix + DX[nh], ny = iy + DY[nh];
            if (not in_bounds(nx, ny)) continue;
            // The move sets the heading to its direction of travel: the robot turns toward the next waypoint
            // and then translates, which is what the trajectory controller actually does.
            if (not cell_free(nx, ny, nh)) continue;
            const float step = ((DX[nh] != 0 and DY[nh] != 0) ? 1.41421356f : 1.0f) * cell_;
            // A small turning penalty keeps the path from zig-zagging between diagonal and axial moves of
            // equal length, which the MPPI would otherwise chase.
            const float turn = (nh == h) ? 0.f : 0.25f * cell_;
            const int ns = sid(nx, ny, nh);
            if (const float ng = g[s] + step + turn; ng < g[ns])
            { g[ns] = ng; parent[ns] = s; open.push({ng + heur(nx, ny), ns}); }
        }
    }

    if (goal_state < 0)
    {
        last_failure_ = std::format("no route: {} expansions over {} free cells{}",
                                    expansions, n_cells - occupied_cells(),
                                    start_in_collision ? " [start was NOT footprint-feasible; escape was "
                                                         "allowed and still found nothing]" : "");
        return std::nullopt;
    }

    std::vector<Eigen::Vector2f> path;
    for (int s = goal_state; s != -1; s = parent[s])
    {
        const int cell = s / kHeadings;
        path.push_back(cell_to_world(cell % w_, cell / w_));
    }
    std::reverse(path.begin(), path.end());
    path.front() = start_room;
    path.back() = goal_room;

    // Keep only turning points. A dense cell chain gives the trajectory controller nothing but jitter for its
    // carrot to chase; the straight runs between corners are exactly the information it needs.
    //
    // Douglas-Peucker on PERPENDICULAR DEVIATION, not on consecutive-segment direction. Direction comparison
    // looks equivalent and is not: the endpoints are snapped to cell centres and then overwritten with the true
    // start/goal, so the first and last segments can be a small fraction of a cell long, and an arbitrarily
    // small positional offset over a very short segment is a large ANGLE. That made a dead-straight 8 m run
    // retain four waypoints. Deviation from the chord is the quantity actually being approximated, and it does
    // not care how the points are spaced.
    const float tol = 0.5f * cell_;
    std::vector<Eigen::Vector2f> simplified;
    const auto rdp = [&](auto&& self, std::size_t lo, std::size_t hi) -> void
    {
        if (hi <= lo + 1) return;
        const Eigen::Vector2f a = path[lo], b = path[hi], ab = b - a;
        const float len2 = ab.squaredNorm();
        std::size_t worst = lo; float worst_d = 0.f;
        for (std::size_t i = lo + 1; i < hi; ++i)
        {
            const Eigen::Vector2f ap = path[i] - a;
            const float t = len2 > 1e-12f ? std::clamp(ap.dot(ab) / len2, 0.f, 1.f) : 0.f;
            if (const float d = (ap - t * ab).norm(); d > worst_d) { worst_d = d; worst = i; }
        }
        if (worst_d <= tol) return;
        self(self, lo, worst);
        simplified.push_back(path[worst]);
        self(self, worst, hi);
    };
    simplified.push_back(path.front());
    rdp(rdp, 0, path.size() - 1);
    simplified.push_back(path.back());
    return simplified;
}

bool GridPlanner::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    const std::vector<Eigen::Vector2f> room = {{-5, -5}, {5, -5}, {5, 5}, {-5, 5}};

    // (1) Empty room: a straight path exists and stays clear.
    {
        GridPlanner p; p.params.safety_margin_m = 0.f;
        p.set_world(room, {});
        const auto path = p.plan({-4, 0}, {4, 0});
        std::printf("  empty room: %s (%zu waypoints)\n", path ? "planned" : p.last_failure().c_str(),
                    path ? path->size() : 0);
        check(path.has_value(), "an empty room must be trivially plannable");
        check(path and path->size() == 2, "a straight run must simplify to its two endpoints");
    }

    // (1b) DISTANCE FIELD vs BRUTE FORCE. The transform is exact, so it must agree with an O(n^2) scan
    // over every occupied cell to within the interpolation error — not "closely", exactly. Checked at
    // cell centres (where interpolation is the identity) so any discrepancy is the transform's, and on a
    // world with obstacles in the interior AND the room border, since outside-the-room is occupied too.
    {
        GridPlanner p; p.params.safety_margin_m = 0.f; p.params.cell_size_m = 0.10f;
        const std::vector<std::vector<Eigen::Vector2f>> obs = {
            {{-1.f, -1.f}, {0.f, -1.f}, {0.f, 0.f}, {-1.f, 0.f}},
            {{2.f, 1.5f}, {3.f, 1.5f}, {3.f, 2.5f}, {2.f, 2.5f}}};
        p.set_world(room, obs);

        // Brute force: nearest occupied cell centre, in metres.
        std::vector<Eigen::Vector2f> occupied;
        for (int iy = 0; iy < p.height(); ++iy)
            for (int ix = 0; ix < p.width(); ++ix)
                if (p.occ_[p.idx(ix, iy)]) occupied.push_back(p.cell_to_world(ix, iy));
        check(not occupied.empty(), "the test world must contain occupied cells");

        float worst = 0.f;
        int checked = 0;
        for (int iy = 0; iy < p.height(); iy += 3)
            for (int ix = 0; ix < p.width(); ix += 3)
            {
                const Eigen::Vector2f c = p.cell_to_world(ix, iy);
                float best = std::numeric_limits<float>::max();
                for (const auto& o : occupied) best = std::min(best, (o - c).norm());
                worst = std::max(worst, std::abs(p.distance_at(c) - best));
                ++checked;
            }
        std::printf("  distance field: %d cells vs brute force, worst error %.6f m (cell %.2f m)\n",
                    checked, worst, 0.10f);
        check(worst < 1e-4f, "the distance transform must be EXACT at cell centres, not approximate");

        // The gradient must point AWAY from an obstacle and be a unit vector in open space (|grad d| = 1
        // is the eikonal property of a true distance field — a chamfer violates it by up to 8%).
        const Eigen::Vector2f probe{1.0f, -0.5f};             // right of the first box, clear of both
        const Eigen::Vector2f g = p.distance_gradient_at(probe);
        check(g.x() > 0.f, "the gradient must point away from the obstacle to the left");
        check(std::abs(g.norm() - 1.f) < 0.12f, "|grad d| must be ~1 in open space (eikonal)");
        std::printf("  gradient at (1.0,-0.5): (%.3f,%.3f), |g| = %.3f\n", g.x(), g.y(), g.norm());

        // Inside an obstacle the distance is zero, and an empty world is wide open rather than blocked.
        check(p.distance_at({-0.5f, -0.5f}) < 1e-6f, "distance inside an obstacle must be zero");
        GridPlanner empty;
        check(empty.distance_at({0.f, 0.f}) > 1e5f, "with no world set, distance must read wide open");
    }

    // (2) THE POINT OF ALL OF THIS. A gap wider than the robot must be usable, and a gap narrower must not.
    // The old stacked margins demanded ~0.95 m for a robot that physically passes 0.461 m; both of these
    // assertions would have failed under that pipeline.
    {
        auto wall_with_gap = [&](float gap) {
            const float hw = gap * 0.5f;
            return std::vector<std::vector<Eigen::Vector2f>>{
                {{-0.3f, -5.f}, {0.3f, -5.f}, {0.3f, -hw}, {-0.3f, -hw}},
                {{-0.3f,  hw}, {0.3f,  hw}, {0.3f,  5.f}, {-0.3f,  5.f}}};
        };
        GridPlanner p; p.params.safety_margin_m = 0.f; p.params.cell_size_m = 0.05f;
        const float need = 2.f * p.footprint.inscribed_radius();

        p.set_world(room, wall_with_gap(need + 0.20f));
        const auto wide = p.plan({-3, 0}, {3, 0});
        p.set_world(room, wall_with_gap(need - 0.15f));
        const auto narrow = p.plan({-3, 0}, {3, 0});
        std::printf("  robot needs %.3f m: gap %.2f -> %s | gap %.2f -> %s\n",
                    need, need + 0.20f, wide ? "PASSES" : "blocked",
                    need - 0.15f, narrow ? "passes" : "BLOCKED (correct)");
        check(wide.has_value(), "a gap WIDER than the footprint must be navigable");
        check(not narrow.has_value(), "a gap NARROWER than the footprint must NOT be navigable");
    }

    // (3) A sealed goal is reported as infeasible, not as a mysterious "no path".
    {
        GridPlanner p; p.params.safety_margin_m = 0.f;
        p.set_world(room, {{{1.f, 1.f}, {3.f, 1.f}, {3.f, 3.f}, {1.f, 3.f}}});
        const auto path = p.plan({-3, -3}, {2, 2});
        std::printf("  goal inside an obstacle: %s\n", path ? "planned (WRONG)" : p.last_failure().c_str());
        check(not path.has_value(), "a goal inside an obstacle must fail");
        check(p.last_failure().find("not footprint-feasible") != std::string::npos,
              "...and must say so, rather than reporting a generic no-route");
    }

    // (4) A start in collision must still plan OUT — the failure that bricked the robot for a whole session.
    {
        GridPlanner p; p.params.safety_margin_m = 0.f;
        p.set_world(room, {{{-3.4f, -0.4f}, {-2.6f, -0.4f}, {-2.6f, 0.4f}, {-3.4f, 0.4f}}});
        const auto path = p.plan({-3.0f, 0.0f}, {3.0f, 0.0f});
        std::printf("  start INSIDE an obstacle: %s\n", path ? "planned out (correct)" : p.last_failure().c_str());
        check(path.has_value(), "a start in collision must still be able to plan OUT");
    }

    // (5) nearest_free must agree with the planner's own feasibility predicate — the two disagreeing is what
    // produced targets the controller would never accept.
    {
        GridPlanner p; p.params.safety_margin_m = 0.f;
        p.set_world(room, {{{1.f, 1.f}, {3.f, 1.f}, {3.f, 3.f}, {1.f, 3.f}}});
        const auto fixed = p.nearest_free({2.f, 2.f}, 0.f);
        std::printf("  nearest_free from inside an obstacle: %s\n",
                    fixed ? std::format("({:.2f},{:.2f})", fixed->x(), fixed->y()).c_str() : "none");
        check(fixed.has_value(), "a blocked pose must be repairable nearby");
        check(fixed and p.pose_free(*fixed, 0.f), "the repaired pose must satisfy the planner's OWN predicate");
    }

    std::printf("GridPlanner::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
