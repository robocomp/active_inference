/*
 * residual_occupancy_grid.cpp  —  see residual_occupancy_grid.h
 */

#include "residual_occupancy_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace rc
{

namespace
{
// Andrew monotone-chain convex hull (self-contained so the grid stays dependency-free).
std::vector<Eigen::Vector2f> hull2d(std::vector<Eigen::Vector2f> pts)
{
    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b)
              { return a.x() < b.x() or (a.x() == b.x() and a.y() < b.y()); });
    const auto cross = [](const Eigen::Vector2f& o, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
                       { return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x()); };
    std::vector<Eigen::Vector2f> h(2 * n);
    int k = 0;
    for (int i = 0; i < n; ++i) { while (k >= 2 and cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--; h[k++] = pts[i]; }
    for (int i = n - 2, t = k + 1; i >= 0; --i) { while (k >= t and cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--; h[k++] = pts[i]; }
    h.resize(k - 1);
    return h;
}
}  // namespace

void OccupancyGrid::reset(float xmin, float ymin, float xmax, float ymax, const OccGridParams& p)
{
    p_ = p;
    const float cs = std::max(0.01f, p.cell_size_m);
    xmin_ = xmin; ymin_ = ymin; inv_cell_ = 1.0f / cs;
    w_ = std::max(1, static_cast<int>(std::ceil((xmax - xmin) / cs)));
    h_ = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / cs)));
    lo_.assign(static_cast<std::size_t>(w_) * h_, 0.0f);
    a_.assign(lo_.size(), p_.beta_prior_a);        // Beta prior: unobserved ⇒ P=0.5 at max variance (unknown)
    b_.assign(lo_.size(), p_.beta_prior_b);
    zmn_.assign(lo_.size(), 0.0f);
    zmx_.assign(lo_.size(), 0.0f);
    dispz_.assign(lo_.size(), 0.0f);
    hit_.assign(lo_.size(), 0);
    occ_.assign(lo_.size(), 0);
    shit_.assign(lo_.size(), 0);
    smiss_.assign(lo_.size(), 0);
    shz_lo_.assign(lo_.size(), 0.0f);
    shz_hi_.assign(lo_.size(), 0.0f);
    shit_w_.assign(lo_.size(), 0.0f);
    smiss_w_.assign(lo_.size(), 0.0f);
}

bool OccupancyGrid::world_to_cell(float x, float y, int& ix, int& iy) const
{
    ix = static_cast<int>(std::floor((x - xmin_) * inv_cell_));
    iy = static_cast<int>(std::floor((y - ymin_) * inv_cell_));
    return in_bounds(ix, iy);
}

void OccupancyGrid::cell_to_world(int ix, int iy, float& x, float& y) const
{
    const float cs = 1.0f / inv_cell_;
    x = xmin_ + (ix + 0.5f) * cs;
    y = ymin_ + (iy + 0.5f) * cs;
}

bool OccupancyGrid::occupied(int ix, int iy) const
{
    // HYSTERESIS: report the LATCHED state, not a raw threshold on lo_. Latching happens in commit_cycle().
    return in_bounds(ix, iy) and occ_[idx(ix, iy)];
}

// ── per-scan ACCUMULATION (no log-odds change yet) — one flag per cell, OR-ed over all this cycle's beams ──

void OccupancyGrid::mark_hit_flag(int ix, int iy, float zlo, float zhi, float w)
{
    if (not in_bounds(ix, iy)) return;
    const int i = idx(ix, iy);
    if (not shit_[i]) { shz_lo_[i] = zlo; shz_hi_[i] = zhi; shit_[i] = 1; shit_w_[i] = w; }
    else { shz_lo_[i] = std::min(shz_lo_[i], zlo); shz_hi_[i] = std::max(shz_hi_[i], zhi);
           shit_w_[i] = std::max(shit_w_[i], w); }   // the most RELIABLE (nearest/stillest) hit sets the weight
}

void OccupancyGrid::mark_miss_flag(int ix, int iy, float beam_z, float w)
{
    if (not in_bounds(ix, iy)) return;
    const int i = idx(ix, iy);
    // Z-AWARE: a beam only carries FREE evidence for a cell if it passes THROUGH the cell's occupied z-band
    // (± margin), or the cell has never been hit (empty). A beam over/under a known obstacle carries none → skip.
    if (hit_[i] and (beam_z < zmn_[i] - p_.z_band_margin_m or beam_z > zmx_[i] + p_.z_band_margin_m))
        { ++sd_.miss_blocked_zaware; return; }
    if (not smiss_[i]) { smiss_[i] = 1; smiss_w_[i] = w; }
    else smiss_w_[i] = std::max(smiss_w_[i], w);        // the most RELIABLE (nearest) see-through sets the weight
}

void OccupancyGrid::commit_cycle()
{
    if (not valid()) return;
    // ONE log-odds update per cell: hit precedence (a cell hit by any beam this scan is a HIT, never cleared
    // this scan), else a single see-through miss. This is the costmap_2d/OctoMap marking-vs-clearing rule and
    // is what makes the field stable (no more dozens of misses per cell per cycle).
    const std::size_t n = lo_.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (shit_[i])
        {
            const float w = shit_w_[i];                         // precision weight (range × ego-motion) of this hit
            lo_[i] = std::clamp(lo_[i] + w * p_.l_hit, -p_.l_clamp, p_.l_clamp);
            if (not hit_[i]) { zmn_[i] = shz_lo_[i]; zmx_[i] = shz_hi_[i]; dispz_[i] = shz_hi_[i]; hit_[i] = 1; }
            else { zmn_[i] = std::min(zmn_[i], shz_lo_[i]); zmx_[i] = std::max(zmx_[i], shz_hi_[i]);
                   // Display height tracks the CURRENT top (EMA rise/fall), so a transient tall return doesn't
                   // permanently inflate it the way the running-max zmx_ does. 0.4 = quick but spike-rejecting.
                   dispz_[i] += 0.4f * (shz_hi_[i] - dispz_[i]); }
            a_[i] += w * p_.beta_hit_w;                          // Beta: occupied evidence, weighted by precision
            ++sd_.hits;
            if (lo_[i] > p_.occ_set and not occ_[i]) { occ_[i] = 1; ++sd_.cells_latched; }
        }
        else if (smiss_[i])
        {
            const float w = smiss_w_[i];                        // precision of this see-through (range × ego-motion)
            lo_[i] = std::clamp(lo_[i] - w * p_.l_miss, -p_.l_clamp, p_.l_clamp);
            b_[i] += w * p_.beta_miss_w;                         // Beta: free evidence, weighted → far/moving clears weakly
            ++sd_.misses;
            if (lo_[i] < p_.occ_clear and occ_[i]) { occ_[i] = 0; ++sd_.cells_released; }
        }
        // Concentration cap: bound α+β ≤ κ_max (preserving the mean) → variance floor + bounded memory so old
        // evidence is discounted as new arrives (dynamic obstacles clear; the field is never over-confident).
        if (const float k = a_[i] + b_[i]; k > p_.beta_kappa_max)
        {
            const float s = p_.beta_kappa_max / k;
            a_[i] *= s; b_[i] *= s;
        }
    }
}

float OccupancyGrid::prob(int ix, int iy) const
{
    if (not in_bounds(ix, iy)) return 0.5f;                      // out of bounds ⇒ unobserved prior
    const int i = idx(ix, iy);
    return a_[i] / std::max(1e-9f, a_[i] + b_[i]);
}

float OccupancyGrid::prob_variance(int ix, int iy) const
{
    if (not in_bounds(ix, iy)) { const float a = p_.beta_prior_a, b = p_.beta_prior_b, k = a + b;
                                 return a * b / (k * k * (k + 1.0f)); }
    const int i = idx(ix, iy);
    const float a = a_[i], b = b_[i], k = a + b;
    return a * b / std::max(1e-9f, k * k * (k + 1.0f));          // Var of Beta(α,β) = αβ/((α+β)²(α+β+1))
}

float OccupancyGrid::prob_std(int ix, int iy) const
{
    return std::sqrt(std::max(0.0f, prob_variance(ix, iy)));
}

void OccupancyGrid::occupancy_fields(std::vector<float>& prob_out, std::vector<float>& var_out,
                                     const CellExplained& explained) const
{
    const std::size_t n = a_.size();
    prob_out.assign(n, 0.0f); var_out.assign(n, 0.0f);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
        {
            const int i = idx(x, y);
            float keep = 1.0f;                                 // 1−p_explained: how much residual survives here
            if (explained)                                     // SOFT collapse: attenuate by the explained prob
            {
                float wx, wy; cell_to_world(x, y, wx, wy);
                const float zlo = hit_[i] ? zmn_[i] : 0.0f, zhi = hit_[i] ? zmx_[i] : 0.05f;
                keep = 1.0f - std::clamp(explained(wx, wy, zlo, zhi), 0.0f, 1.0f);
            }
            const float a = a_[i], b = b_[i], k = a + b;
            prob_out[i] = keep * (a / std::max(1e-9f, k));      // an object owns this cell (keep→0) ⇒ no residual risk
            var_out[i]  = keep * (a * b / std::max(1e-9f, k * k * (k + 1.0f)));   // ...and no epistemic pull
        }
}

void OccupancyGrid::integrate_sweep(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& points_room,
                                    bool begin_cycle, float reliability,
                                    const std::vector<float>* hit_weight_scale)
{
    if (not valid()) return;
    const bool use_scale = hit_weight_scale != nullptr and hit_weight_scale->size() == points_room.size();
    if (begin_cycle)                                       // first sensor of the cycle: reset diagnostics + scratch
    {
        sd_ = SweepDiag{};
        std::fill(shit_.begin(),  shit_.end(),  0);
        std::fill(smiss_.begin(), smiss_.end(), 0);
        std::fill(shit_w_.begin(), shit_w_.end(), 0.0f);
        std::fill(smiss_w_.begin(), smiss_w_.end(), 0.0f);
    }
    const float cs = 1.0f / inv_cell_;
    const float ox = origin.x(), oy = origin.y(), oz = origin.z();
    const float r0 = p_.hit_reliable_range_m;              // precision falloff scale (0 ⇒ uniform full weight)
    const float rel = std::clamp(reliability, 0.0f, 1.0f); // global ego-motion trust for this sweep
    // precision weight of a hit at horizontal range r: ego-motion × range falloff (0<w≤rel; near/still → rel).
    const auto hit_w = [&](float r) { return r0 > 0.0f ? rel * (r0 * r0) / (r0 * r0 + r * r) : rel; };

    for (std::size_t pi = 0; pi < points_room.size(); ++pi)
    {
        const auto& p = points_room[pi];
        const float px = p.x(), py = p.y(), pz = p.z();
        const float dx = px - ox, dy = py - oy;
        const float L = std::hypot(dx, dy);
        const float range = L;                                  // horizontal range
        // Obstacle band referenced to the DATA-DRIVEN floor plane (0,0,0 ⇒ the original fixed z=0 band). A return
        // is an obstacle only if it rises above the local floor by more than the grazing band — so an offset/tilted
        // floor (new scenario) is explained away and never latches.
        const float floor_z = fp_a_ * px + fp_b_ * py + fp_c_;
        const bool in_band = (pz > floor_z + p_.floor_z0 + p_.floor_slope * range) and (pz < p_.ceil_z);
        // this return's HIT precision weight: range×ego (hit_w) × optional per-point cue (e.g. semantic floor).
        // The MISS/clearing weight is deliberately left un-scaled — clearing free space is always safe.
        const float w = hit_w(range) * (use_scale ? (*hit_weight_scale)[pi] : 1.0f);

        int ex, ey; const bool endp_in = world_to_cell(px, py, ex, ey);
        if (L < 1e-4f)                                          // degenerate: just the endpoint
        { if (endp_in and in_band) mark_hit_flag(ex, ey, pz, pz, w); continue; }

        const float ux = dx / L, uy = dy / L;
        int cx, cy;
        if (not world_to_cell(ox, oy, cx, cy))                 // sensor outside the grid → only place the hit
        { if (endp_in and in_band) mark_hit_flag(ex, ey, pz, pz, w); continue; }

        mark_miss_flag(cx, cy, oz, hit_w(0.0f));               // sensor cell is free (range 0 → full trust)

        const int stepx = ux > 0 ? 1 : -1, stepy = uy > 0 ? 1 : -1;
        const float BIG = std::numeric_limits<float>::max();
        // distance along the ray to the first x/y cell boundary, then per-cell strides (in distance units)
        const float nbx = xmin_ + (cx + (stepx > 0 ? 1 : 0)) * cs;
        const float nby = ymin_ + (cy + (stepy > 0 ? 1 : 0)) * cs;
        float tMaxX = std::abs(ux) > 1e-9f ? (nbx - ox) / ux : BIG;
        float tMaxY = std::abs(uy) > 1e-9f ? (nby - oy) / uy : BIG;
        const float tDx = std::abs(ux) > 1e-9f ? cs / std::abs(ux) : BIG;
        const float tDy = std::abs(uy) > 1e-9f ? cs / std::abs(uy) : BIG;

        int guard = w_ + h_ + 4;
        while (guard-- > 0)
        {
            float t_here;
            if (tMaxX < tMaxY) { cx += stepx; t_here = tMaxX; tMaxX += tDx; }
            else               { cy += stepy; t_here = tMaxY; tMaxY += tDy; }
            if (cx == ex and cy == ey) break;                  // reached the endpoint cell (handled below)
            if (t_here >= L)           break;                  // overshot the return
            if (not in_bounds(cx, cy)) break;                  // left the grid
            const float bz = oz + (t_here / L) * (pz - oz);    // beam height at this cell
            mark_miss_flag(cx, cy, bz, hit_w(t_here));         // weight the clearing by the CLEARED CELL's range
        }
        if (endp_in and in_band) mark_hit_flag(ex, ey, pz, pz, w);    // the return itself: occupied (if a nav-band obstacle)
    }
}


std::vector<std::uint8_t> OccupancyGrid::residual_mask(const CellExplained& explained) const
{
    std::vector<std::uint8_t> m(lo_.size(), 0);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
        {
            if (not occupied(x, y)) continue;
            if (explained)
            {
                const int i = idx(x, y);
                float wx, wy; cell_to_world(x, y, wx, wy);
                const float zlo = hit_[i] ? zmn_[i] : 0.0f, zhi = hit_[i] ? zmx_[i] : 0.05f;
                if (explained(wx, wy, zlo, zhi) > 0.5f) continue;   // MAP decision: more likely explained than not
            }
            m[idx(x, y)] = 1;
        }
    return m;
}

std::vector<std::uint8_t> OccupancyGrid::dilate_mask(const std::vector<std::uint8_t>& m, int R) const
{
    if (R <= 0) return m;
    std::vector<std::uint8_t> out(m.size(), 0);
    const int R2 = R * R;
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
        {
            if (not m[idx(x, y)]) continue;
            for (int dy = -R; dy <= R; ++dy) for (int dx = -R; dx <= R; ++dx)
            {
                if (dx * dx + dy * dy > R2) continue;                 // Euclidean disk
                const int nx = x + dx, ny = y + dy;
                if (in_bounds(nx, ny)) out[idx(nx, ny)] = 1;
            }
        }
    return out;
}

std::vector<float> OccupancyGrid::residual_cell_centres(const CellExplained& explained) const
{
    std::vector<float> out;
    if (not valid()) return out;
    const auto m = residual_mask(explained);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
            if (m[idx(x, y)]) { float wx, wy; cell_to_world(x, y, wx, wy); out.push_back(wx); out.push_back(wy); }
    return out;
}

std::vector<float> OccupancyGrid::residual_cell_centres_xyz(const CellExplained& explained) const
{
    std::vector<float> out;
    if (not valid()) return out;
    const auto m = residual_mask(explained);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
            if (m[idx(x, y)])
            {
                float wx, wy; cell_to_world(x, y, wx, wy);
                out.push_back(wx);
                out.push_back(wy);
                out.push_back(dispz_[idx(x, y)]);   // CURRENT top height (EMA), not the running-max z-band
            }
    return out;
}

std::vector<float> OccupancyGrid::inflated_border_centres(const CellExplained& explained, float inflate_radius_m) const
{
    std::vector<float> out;
    if (not valid() or inflate_radius_m <= 0.0f) return out;
    const auto m = residual_mask(explained);
    const int R = static_cast<int>(std::round(inflate_radius_m * inv_cell_));
    const auto infl = dilate_mask(m, R);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
        {
            const int i = idx(x, y);
            if (not infl[i] or m[i]) continue;                        // border = inflated but not itself occupied
            float wx, wy; cell_to_world(x, y, wx, wy);
            if (explained)                                            // a clearance ring INSIDE a known object is
            {                                                        // redundant (the planner avoids its box) → drop it
                const float zlo = hit_[i] ? zmn_[i] : 0.0f, zhi = hit_[i] ? zmx_[i] : 0.05f;
                if (explained(wx, wy, zlo, zhi) > 0.5f) continue;
            }
            out.push_back(wx); out.push_back(wy);
        }
    return out;
}

std::vector<OccComponent> OccupancyGrid::occupied_components(int min_cells, const CellExplained& explained,
                                                             float inflate_radius_m) const
{
    std::vector<OccComponent> out;
    if (not valid()) return out;
    const float cs = 1.0f / inv_cell_;
    const auto occ = residual_mask(explained);                        // z-band source (only true-occupied cells)
    const int R = static_cast<int>(std::round(std::max(0.0f, inflate_radius_m) * inv_cell_));
    const auto mask = (R > 0) ? dilate_mask(occ, R) : occ;            // components on the inflated set (bridges + clearance)

    // Corner-space origin (bottom-left corner of cell (0,0)): cell centres are origin+((x+0.5)cs,(y+0.5)cs).
    float ox0, oy0; cell_to_world(0, 0, ox0, oy0); ox0 -= 0.5f * cs; oy0 -= 0.5f * cs;
    const int cstride = w_ + 1;                                       // corner lattice is (w_+1)×(h_+1)
    // Trace a component's cell set into CONCAVE CCW boundary loops (axis-aligned staircase). Uses the
    // interior-on-left edge convention → outer boundary is CCW (kept); holes are CW (dropped = conservative).
    // Returns empty on a diagonal-pinch ambiguity (a corner shared by two boundary edges) so the caller
    // can fall back to the convex hull for that odd shape.
    const auto trace_outline = [&](const std::vector<std::pair<int, int>>& cells)
        -> std::vector<std::vector<Eigen::Vector2f>>
    {
        std::unordered_set<long> S; S.reserve(cells.size() * 2);
        for (const auto& [cx, cy] : cells) S.insert(static_cast<long>(cy) * w_ + cx);
        const auto occ_in = [&](int x, int y) { return S.count(static_cast<long>(y) * w_ + x) > 0; };
        std::unordered_map<long, long> nxt; nxt.reserve(cells.size() * 2);
        bool ambiguous = false;
        const auto add_edge = [&](int ai, int aj, int bi, int bj) {
            const long a = static_cast<long>(aj) * cstride + ai, b = static_cast<long>(bj) * cstride + bi;
            if (not nxt.emplace(a, b).second) ambiguous = true;
        };
        for (const auto& [cx, cy] : cells)                            // interior-on-left directed edges
        {
            if (not occ_in(cx + 1, cy)) add_edge(cx + 1, cy,     cx + 1, cy + 1);
            if (not occ_in(cx, cy + 1)) add_edge(cx + 1, cy + 1, cx,     cy + 1);
            if (not occ_in(cx - 1, cy)) add_edge(cx,     cy + 1, cx,     cy);
            if (not occ_in(cx, cy - 1)) add_edge(cx,     cy,     cx + 1, cy);
        }
        if (ambiguous or nxt.empty()) return {};
        const auto to_world = [&](long key) {
            return Eigen::Vector2f(ox0 + static_cast<float>(key % cstride) * cs,
                                   oy0 + static_cast<float>(key / cstride) * cs);
        };
        std::vector<std::vector<Eigen::Vector2f>> loops;
        std::unordered_set<long> used;
        for (const auto& [start, _] : nxt)
        {
            if (used.count(start)) continue;
            std::vector<Eigen::Vector2f> pts; long cur = start;
            for (std::size_t guard = 0; guard <= nxt.size(); ++guard)
            {
                if (used.count(cur)) break;
                used.insert(cur); pts.push_back(to_world(cur));
                const auto it = nxt.find(cur); if (it == nxt.end()) { pts.clear(); break; }
                cur = it->second; if (cur == start) break;
            }
            if (pts.size() < 4) continue;
            float area2 = 0.f;                                        // signed area: +CCW outer, −CW hole
            for (std::size_t i = 0; i < pts.size(); ++i)
            { const auto& a = pts[i]; const auto& b = pts[(i + 1) % pts.size()]; area2 += a.x() * b.y() - b.x() * a.y(); }
            if (area2 <= 0.f) continue;                               // hole → drop (conservative)
            std::vector<Eigen::Vector2f> simp;                        // drop collinear staircase vertices
            const std::size_t n = pts.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto& p0 = pts[(i + n - 1) % n]; const auto& p1 = pts[i]; const auto& p2 = pts[(i + 1) % n];
                if (std::abs((p1.x() - p0.x()) * (p2.y() - p0.y()) - (p1.y() - p0.y()) * (p2.x() - p0.x())) > 1e-9f)
                    simp.push_back(p1);
            }
            if (simp.size() >= 3) loops.push_back(std::move(simp));
        }
        return loops;
    };

    std::vector<std::uint8_t> seen(mask.size(), 0);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
        {
            if (seen[idx(x, y)] or not mask[idx(x, y)]) continue;
            std::vector<Eigen::Vector2f> centres;
            std::vector<std::pair<int, int>> cellsxy;
            float zlo = 1e9f, zhi = -1e9f;
            std::queue<std::pair<int, int>> q; q.push({x, y}); seen[idx(x, y)] = 1;
            while (not q.empty())
            {
                const auto [ax, ay] = q.front(); q.pop();
                float wx, wy; cell_to_world(ax, ay, wx, wy);
                centres.emplace_back(wx, wy);
                cellsxy.emplace_back(ax, ay);
                const int i = idx(ax, ay);
                if (occ[i] and hit_[i]) { zlo = std::min(zlo, zmn_[i]); zhi = std::max(zhi, zmx_[i]); }   // z from real cells
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 and dy == 0) continue;
                    const int nx = ax + dx, ny = ay + dy;
                    if (in_bounds(nx, ny) and not seen[idx(nx, ny)] and mask[idx(nx, ny)])
                    { seen[idx(nx, ny)] = 1; q.push({nx, ny}); }
                }
            }
            if (static_cast<int>(centres.size()) < min_cells) continue;
            OccComponent c;
            c.n_cells = static_cast<int>(centres.size());
            float xmn = 1e9f, ymn = 1e9f, xmx = -1e9f, ymx = -1e9f;
            for (const auto& p : centres)
            { xmn = std::min(xmn, p.x()); xmx = std::max(xmx, p.x()); ymn = std::min(ymn, p.y()); ymx = std::max(ymx, p.y()); }
            c.cx = 0.5f * (xmn + xmx); c.cy = 0.5f * (ymn + ymx);
            c.w  = (xmx - xmn) + cs;   c.d  = (ymx - ymn) + cs;   c.yaw = 0.0f;
            c.z_min = (zhi >= zlo) ? zlo : 0.0f;
            c.z_max = (zhi >= zlo) ? zhi : 0.05f;
            c.hull    = hull2d(centres);
            c.outline = trace_outline(cellsxy);   // concave loops; empty ⇒ publisher falls back to hull
            out.push_back(std::move(c));
        }
    return out;
}

// ─── Self-test: the three SAFETY properties a costmap must hold ──────────────────────────────────────────
bool OccupancyGrid::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    OccGridParams P;                       // defaults
    const Eigen::Vector3f sensor(0.0f, 0.0f, 0.5f);

    // A synthetic obstacle: a vertical near-face at room x≈2, y∈[-0.2,0.2], z∈[0.2,0.6] (in the nav band).
    auto face_returns = [&](std::vector<Eigen::Vector3f>& v) {
        for (int i = 0; i < 40; ++i)
        {
            const float y = -0.2f + 0.4f * (i / 39.0f);
            for (float z : {0.25f, 0.40f, 0.55f}) v.push_back({2.0f, y, z});
        }
    };

    // ── (1) COMPLETENESS: one sweep of in-band returns → those cells occupied (≤2 frames) ──
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> sweep; face_returns(sweep);
        g.integrate_sweep(sensor, sweep); g.commit_cycle();
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        check(g.occupied(ix, iy), "an in-band return must make its cell OCCUPIED in one frame (completeness)");
        const auto comps = g.occupied_components();
        std::printf("  completeness: %zu component(s); face cell occupied=%d\n", comps.size(), g.occupied(ix, iy));
        check(not comps.empty(), "occupied cells must form a publishable component");
    }

    // ── (2) OCCLUDED ≠ FREE: a beam stopping SHORT leaves the space behind it UNKNOWN (log-odds 0) ──
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> sweep;
        for (int i = 0; i < 40; ++i) sweep.push_back({1.5f, -0.2f + 0.4f * (i / 39.0f), 0.4f});  // occluder at x=1.5
        g.integrate_sweep(sensor, sweep); g.commit_cycle();
        int ix, iy; g.world_to_cell(3.0f, 0.0f, ix, iy);                 // behind the occluder
        std::printf("  occluded: cell behind occluder logodds=%.2f (want 0/unknown)\n", g.logodds(ix, iy));
        check(std::abs(g.logodds(ix, iy)) < 1e-6f, "space behind an occluder must stay UNKNOWN, never marked free");
    }

    // ── (3) CARVE CLEARS: an obstacle seen, then REMOVED → see-through beams clear its cells ──
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> obst; face_returns(obst);
        for (int k = 0; k < 3; ++k) { g.integrate_sweep(sensor, obst); g.commit_cycle(); }     // build the obstacle up
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        const bool was_occ = g.occupied(ix, iy);
        // now it's GONE: beams pass through x≈2 at the obstacle's height and return from the back wall x=4.5.
        std::vector<Eigen::Vector3f> clear;
        for (int i = 0; i < 40; ++i) for (float z : {0.25f, 0.40f, 0.55f})
            clear.push_back({4.5f, -0.2f + 0.4f * (i / 39.0f), z});
        for (int k = 0; k < 12; ++k) { g.integrate_sweep(sensor, clear); g.commit_cycle(); }
        std::printf("  carve: was_occ=%d after see-through occupied=%d logodds=%.2f\n",
                    was_occ, g.occupied(ix, iy), g.logodds(ix, iy));
        check(was_occ, "the obstacle must have been occupied first");
        check(not g.occupied(ix, iy), "a removed obstacle must be CLEARED by see-through beams (carve)");
    }

    // ── (4) Z-AWARE: a LOW obstacle is NOT erased by a HIGH beam passing over it ──
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> low;                                // low box at x=2, z∈[0.10,0.20]
        for (int i = 0; i < 40; ++i) for (float z : {0.12f, 0.18f}) low.push_back({2.0f, -0.2f + 0.4f * (i / 39.0f), z});
        for (int k = 0; k < 3; ++k) { g.integrate_sweep(sensor, low); g.commit_cycle(); }
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        const bool occ_before = g.occupied(ix, iy);
        // HIGH beams passing OVER it (return from the back wall at z≈1.2, well above the 0.10-0.20 obstacle).
        std::vector<Eigen::Vector3f> high;
        for (int i = 0; i < 40; ++i) high.push_back({4.5f, -0.2f + 0.4f * (i / 39.0f), 1.2f});
        for (int k = 0; k < 12; ++k) { g.integrate_sweep(sensor, high); g.commit_cycle(); }
        std::printf("  z-aware: low occ_before=%d after high-beams occupied=%d\n", occ_before, g.occupied(ix, iy));
        check(occ_before, "the low obstacle must be occupied first");
        check(g.occupied(ix, iy), "a HIGH beam over a LOW obstacle must NOT erase it (z-aware clearing)");
    }

    // ── (5) BELIEF FIELD: Var[P] must separate UNKNOWN from CONFLICTED, and shrink with evidence ──
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        int ux, uy; g.world_to_cell(3.5f, 3.5f, ux, uy);        // a corner cell no beam ever touches → UNKNOWN
        const float var_unknown = g.prob_variance(ux, uy);

        // A CONFLICTED cell: alternate a sweep that HITS x≈2 with a see-through sweep that clears it, many times.
        std::vector<Eigen::Vector3f> hit; face_returns(hit);
        std::vector<Eigen::Vector3f> thru;
        for (int i = 0; i < 40; ++i) for (float z : {0.25f, 0.40f, 0.55f})
            thru.push_back({4.5f, -0.2f + 0.4f * (i / 39.0f), z});  // see-through past x=2 at the obstacle's height
        for (int k = 0; k < 30; ++k)
        { g.integrate_sweep(sensor, (k % 2) ? thru : hit); g.commit_cycle(); }
        int cx, cy; g.world_to_cell(2.0f, 0.0f, cx, cy);
        const float p_conflict   = g.prob(cx, cy);
        const float var_conflict = g.prob_variance(cx, cy);

        // A CONFIDENT cell: hit x≈2 every sweep.
        OccupancyGrid g2; g2.reset(-1, -1, 5, 5, P);
        for (int k = 0; k < 30; ++k) { g2.integrate_sweep(sensor, hit); g2.commit_cycle(); }
        const float p_conf   = g2.prob(cx, cy);
        const float var_conf = g2.prob_variance(cx, cy);

        std::printf("  belief: unknown[P≈0.5 var=%.4f] conflict[P=%.2f var=%.4f] confident[P=%.2f var=%.4f]\n",
                    var_unknown, p_conflict, var_conflict, p_conf, var_conf);
        check(var_unknown > var_conflict, "UNKNOWN cell must have HIGHER variance than a CONFLICTED one");
        check(var_conflict > var_conf,    "a CONFLICTED cell must still be more uncertain than a CONFIDENT one");
        check(p_conf > 0.9f,              "a continuously-hit cell must have HIGH occupancy probability (risk)");
        check(var_conf < var_unknown,     "evidence must SHRINK the variance below the unobserved prior");
    }

    std::printf("OccupancyGrid::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
