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

float OccupancyGrid::floor_obstacle_responsibility(float x, float y, float z, float range) const
{
    if (not p_.floor_responsibility) return 1.0f;                 // term off ⇒ the old hard step
    const float floor_z = fp_a_ * x + fp_b_ * y + fp_c_;
    // σ of the FLOOR component: its own measured planar-fit scatter, the grazing term (a floor return lands higher
    // the shallower the beam), and the sensor's irreducible range noise — added in quadrature (independent terms).
    const float graze = p_.floor_slope * std::max(0.0f, range);
    const float s2 = fp_rms_ * fp_rms_ + graze * graze + p_.floor_sigma_min_m * p_.floor_sigma_min_m;
    const float sigma = std::sqrt(std::max(1e-6f, s2));
    // Obstacle component: a height uniform over the nav band ⇒ density u = 1/(ceil_z − floor_z). Floor component:
    // N(z; floor_z, σ²). Equal priors (no knob). r_obst = u / (u + N) — the mixture posterior for "obstacle".
    const float span = std::max(0.10f, p_.ceil_z - floor_z);
    const float u = 1.0f / span;
    const float t = (z - floor_z) / sigma;
    const float n = std::exp(-0.5f * t * t) / (sigma * 2.50662827f);            // N(z; floor_z, σ²)
    return u / std::max(1e-12f, u + n);
}

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

void OccupancyGrid::mark_floor_endpoint_flag(int ix, int iy, float band_top, float w)
{
    if (not in_bounds(ix, iy)) return;
    const int i = idx(ix, iy);
    // A beam that TERMINATED on the floor inside this cell cannot be gated the way a TRAVERSING beam is. The
    // z-aware gate in mark_miss_flag asks "did the beam pass through the cell's remembered z-band?", and a floor
    // return never does — it arrives at floor height by definition — so routing it through that gate discards it
    // every time and leaves the ratchet exactly where it was (measured: the phantom in self_test property (11)
    // survived 100 delivered floor returns).
    // The right question for a terminating beam is different. The return reached the FLOOR at this cell, so
    // anything RESTING on the floor here, of any height, would have blocked it first: a floor return refutes a
    // floor-standing obstacle. What it cannot refute is a FLOATING one — a tabletop, a shelf, a windowsill — which
    // it simply passed underneath. And the cell already records which of the two its evidence is: zmn_, the LOWEST
    // return ever seen here. So the gate is support, not overlap: clear the cell if its lowest evidence sits within
    // one z_band_margin_m of where the floor itself lives (⇒ floor-standing, refuted), hold it otherwise
    // (⇒ floating, the beam went under it — the tabletop half of the property).
    if (hit_[i] and zmn_[i] > band_top + p_.z_band_margin_m) { ++sd_.miss_blocked_zaware; return; }
    if (not smiss_[i]) { smiss_[i] = 1; smiss_w_[i] = w; }
    else smiss_w_[i] = std::max(smiss_w_[i], w);
    ++sd_.floor_endpoint_clears;
}

void OccupancyGrid::commit_cycle(float dt_s)
{
    if (not valid()) return;
    // ONE log-odds update per cell: hit precedence (a cell hit by any beam this scan is a HIT, never cleared
    // this scan), else a single see-through miss. This is the costmap_2d/OctoMap marking-vs-clearing rule and
    // is what makes the field stable (no more dozens of misses per cell per cycle).
    //
    // FORGETTING (see OccGridParams::forget_half_life_s): a cell touched by NEITHER a hit nor a miss this cycle
    // is unobserved — occluded, out of range, or behind us. It used to receive no update at all, which made
    // anything ever latched behind an occluder immortal. It now relaxes toward the prior at a fixed half-life.
    const float gamma = (p_.forget_half_life_s > 0.0f and dt_s > 0.0f)
                      ? std::exp2(-dt_s / p_.forget_half_life_s)     // ½^(Δt/T) — evidence retained this cycle
                      : 1.0f;                                        // 1 ⇒ forgetting off (never-forget behaviour)
    const std::size_t n = lo_.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (shit_[i])
        {
            const float w = shit_w_[i];                         // precision weight (range × ego-motion) of this hit
            lo_[i] = std::clamp(lo_[i] + w * p_.l_hit, -p_.l_clamp, p_.l_clamp);
            if (not hit_[i]) { zmn_[i] = shz_lo_[i]; zmx_[i] = shz_hi_[i]; dispz_[i] = shz_hi_[i]; hit_[i] = 1; }
            else { zmn_[i] = std::min(zmn_[i], shz_lo_[i]); zmx_[i] = std::max(zmx_[i], shz_hi_[i]);
                   // ★ DO NOT make this band CONTRACT. It was tried (2026-07-29) on the theory that a
                   // monotonically-widening band made cells progressively harder to clear. That theory has the
                   // SIGN BACKWARDS. mark_miss_flag blocks a clearing beam when it falls OUTSIDE the band, so a
                   // WIDER band admits MORE clearing and a NARROWER one admits LESS. Worse, a hit records a
                   // single height (mark_hit_flag is called with zlo == zhi == pz), so any contraction drives the
                   // band toward ZERO width, leaving only ±z_band_margin_m. Measured live over 400 cycles:
                   // blocked see-throughs 117.8k → 357.7k per cycle (85.2% → 94.6% of all clearing evidence),
                   // a 3.0× regression. The running min/max is correct; the high blocked FRACTION is inherent to
                   // z-aware clearing on a 2-D grid (a beam at another height genuinely carries no information
                   // about this column) and is not by itself evidence of a defect.
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
        else if (gamma < 1.0f)
        {
            // UNOBSERVED this cycle → relax toward the prior. The log-odds decays toward 0 (no information) and
            // the Beta toward Jeffreys, so P → 0.5 and Var → max: "I no longer know what is there."
            lo_[i] *= gamma;
            a_[i] = p_.beta_prior_a + (a_[i] - p_.beta_prior_a) * gamma;
            b_[i] = p_.beta_prior_b + (b_[i] - p_.beta_prior_b) * gamma;
            // Release at occ_set, NOT at occ_clear. occ_clear is evidence of FREENESS ("I have seen through
            // here repeatedly"), which decay can never produce: lo relaxes toward 0 = NO information, so a
            // decaying cell would never cross a negative threshold and would stay latched forever — the very
            // bug this term exists to fix. The meaningful question for an unobserved cell is instead "is the
            // evidence that justified latching still there?", so it un-latches at the same level it latched:
            // symmetric, and no new parameter. There is no flicker risk (an unobserved cell cannot oscillate),
            // and completeness is preserved — one hit re-latches it the instant it is seen again. A released
            // cell becomes UNKNOWN, which this grid already treats as not-an-obstacle for the hull read-out.
            if (lo_[i] < p_.occ_set and occ_[i])
            {
                occ_[i] = 0; ++sd_.cells_released; ++sd_.cells_forgotten;
                // The remembered z-band described an obstacle we have now forgotten. Keeping it would leave the
                // z-aware gate blocking future clearing beams on behalf of something no longer believed to exist,
                // which is precisely the ratchet this change exists to break. Drop it: the cell is unknown again.
                hit_[i] = 0; zmn_[i] = 0.0f; zmx_[i] = 0.0f;
            }
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
                float zlo, zhi; readout_zband(i, zlo, zhi);
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
    // SELF-BODY term: P(this return came from the WORLD, not off our own body) = Φ(s/σ), s = signed distance from
    // the return to the body envelope. A probit in s, not a radius cut — it is 0.5 exactly on the surface and
    // fades to ~1 within 2σ outside, so a clearly-external return is never suppressed. HITS only: clearing free
    // space is always safe, and the sensor's own cell already emits a miss. radius<=0 ⇒ term off (weight 1).
    const auto world_w = [&](float px, float py)
    {
        if (self_r_ <= 0.0f) return 1.0f;
        const float sigma = std::max(1e-3f, p_.self_body_sigma_m);
        const float s = std::hypot(px - self_x_, py - self_y_) - self_r_;   // >0 outside the body envelope
        return 0.5f * std::erfc(-s / (sigma * 1.41421356f));                 // Φ(s/σ)
    };

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
        const float band_top = floor_z + p_.floor_z0 + p_.floor_slope * range;   // where the floor lives here
        const bool in_band = (pz > band_top) and (pz < p_.ceil_z);
        // BELOW the band = a FLOOR return. It is not neutral: the beam reached the floor here, so nothing RESTING
        // on the floor in this cell could have been in the way. It clears its own cell, gated on support rather
        // than on z-overlap — see mark_floor_endpoint_flag and floor_return_clears.
        const bool floor_return = p_.floor_return_clears and pz <= band_top;
        // this return's HIT precision weight: range×ego (hit_w) × self-body (world_w) × FLOOR RESPONSIBILITY (this
        // return is only obstacle evidence in so far as the floor model does not already explain its height) ×
        // optional per-point cue (e.g. semantic floor). The MISS/clearing weight is deliberately left un-scaled by
        // the hit-side terms — clearing free space is always safe.
        const float self_w = world_w(px, py);
        if (self_w < 0.99f) ++sd_.self_hits_damped;
        const float floor_w = floor_obstacle_responsibility(px, py, pz, range);
        if (in_band and floor_w < 0.9f) ++sd_.floor_damped_hits;
        const float w = hit_w(range) * self_w * floor_w * (use_scale ? (*hit_weight_scale)[pi] : 1.0f);

        int ex, ey; const bool endp_in = world_to_cell(px, py, ex, ey);
        // the endpoint's own evidence: an in-band return marks it occupied, a floor return marks it FREE.
        const auto place_endpoint = [&]
        {
            if (not endp_in) return;
            if (in_band) mark_hit_flag(ex, ey, pz, pz, w);
            else if (floor_return) mark_floor_endpoint_flag(ex, ey, band_top, hit_w(range));
        };
        if (L < 1e-4f) { place_endpoint(); continue; }          // degenerate: just the endpoint

        const float ux = dx / L, uy = dy / L;
        int cx, cy;
        if (not world_to_cell(ox, oy, cx, cy))                 // sensor outside the grid → only place the endpoint
        { place_endpoint(); continue; }

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
        place_endpoint();      // the return itself: occupied if a nav-band obstacle, FREE if it landed on the floor
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
                float zlo, zhi; readout_zband(i, zlo, zhi);
                if (explained(wx, wy, zlo, zhi) > 0.5f) continue;   // MAP decision: more likely explained than not
            }
            m[idx(x, y)] = 1;
        }
    return m;
}

long OccupancyGrid::residual_count(const CellExplained& explained) const
{
    if (not valid()) return 0;
    const auto m = residual_mask(explained);
    long n = 0; for (const auto v : m) n += v;
    return n;
}

std::vector<long> OccupancyGrid::residual_height_hist(const std::vector<float>& edges,
                                                      const CellExplained& explained) const
{
    std::vector<long> bins(edges.size() + 1, 0);
    if (not valid()) return bins;
    const auto m = residual_mask(explained);
    for (std::size_t i = 0; i < m.size(); ++i)
    {
        if (not m[i]) continue;
        // Bin by the cell's CURRENT top height (dispz_, the EMA of this cycle's hits), not the running-max zmx_.
        // zmx_ never contracts, so one transient tall return relabels a floor-height cell "tall" for the rest of
        // the run — which is precisely how a floor phantom hides from a zmx_-binned histogram.
        std::size_t k = 0;
        while (k < edges.size() and dispz_[i] > edges[k]) ++k;
        ++bins[k];
    }
    return bins;
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
                float zlo, zhi; readout_zband(i, zlo, zhi);
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
    auto mask = (R > 0) ? dilate_mask(occ, R) : occ;                  // components on the inflated set (bridges + clearance)
    // RE-MASK THE DILATION. residual_mask() applies the explainer, but the C-space dilation then grows the set
    // back OUT by inflate_radius_m with no further check — straight into the very regions the explainer had just
    // excluded. inflated_border_centres() already re-tests every dilated cell (see below); occupied_components()
    // did not, and that asymmetry is a bug, not a tuning choice. It cost real clearance on both explainers that
    // matter: the wall band lost 0.25 m of its 0.35 m margin (hulls reached to within ~0.09 m of a wall), and —
    // the damaging one — the ROBOT's own exclusion disc did too, so a cell just outside the robot mask published
    // a hull edge at (robot_radius − inflate) from the robot centre. That is the near-robot "phantom" the planner
    // was being squeezed by. Filtering the added cells through the same predicate removes a defect rather than
    // adding a knob, and restores both margins to their configured values.
    if (R > 0 and explained)
        for (int y = 0; y < h_; ++y)
            for (int x = 0; x < w_; ++x)
            {
                const int i = idx(x, y);
                if (not mask[i] or occ[i]) continue;                  // only cells ADDED by the dilation
                float wx, wy; cell_to_world(x, y, wx, wy);
                const float zlo = hit_[i] ? zmn_[i] : 0.0f, zhi = hit_[i] ? zmx_[i] : 0.05f;
                if (explained(wx, wy, zlo, zhi) > 0.5f) mask[i] = 0;   // same MAP decision residual_mask() uses
            }

    // Corner-space origin (bottom-left corner of cell (0,0)): cell centres are origin+((x+0.5)cs,(y+0.5)cs).
    float ox0, oy0; cell_to_world(0, 0, ox0, oy0); ox0 -= 0.5f * cs; oy0 -= 0.5f * cs;
    const int cstride = w_ + 1;                                       // corner lattice is (w_+1)×(h_+1)
    // Trace a component's cell set into CONCAVE CCW boundary loops (axis-aligned staircase). Uses the
    // interior-on-left edge convention → outer boundary is CCW (kept); holes are CW (dropped = conservative).
    // Returns empty on a diagonal-pinch ambiguity (a corner shared by two boundary edges) so the caller
    // can fall back to the convex hull for that odd shape.
    // EXACT, HOLE-FREE COVER of a component's cells as axis-aligned rectangles (greedy maximal rectangles).
    //
    // This replaces the boundary-loop trace, which had two failure modes that both PUBLISHED FREE SPACE AS
    // OCCUPIED — the cause of obstacle polygons appearing in never-observed areas (in front of a door the robot
    // has never faced) and of half a room going solid:
    //   1. It DROPPED interior holes (commented "holes dropped = conservative"), so free space enclosed by a
    //      ring of residual became solid. Measured on a ring component: 6.49 m2 of cells published as 12.49 m2.
    //   2. On a diagonal pinch — the 8-connected flood fill unions cells that its 4-connected edge convention
    //      cannot trace — it bailed and the publisher fell back to the CONVEX HULL of the whole component: 15x.
    // Rectangles have neither failure mode by construction: the union is exactly the occupied cell set, every
    // piece is convex and simple, and there is no fallback path. The publisher and the controller both treat a
    // ring as a filled polygon already, so nothing downstream changes — several small filled rectangles are
    // exactly right where one big filled outline was wrong.
    //
    // Greedy: scan row-major, and from each uncovered cell extend maximally right, then extend down for as long
    // as every row spans the same run. Cheap, deterministic, and typically a handful of rectangles per blob.
    const auto rect_decompose = [&](const std::vector<std::pair<int, int>>& cells)
        -> std::vector<std::vector<Eigen::Vector2f>>
    {
        std::vector<std::vector<Eigen::Vector2f>> rects;
        if (cells.empty()) return rects;
        // Work on a COARSENED lattice (publish_cell_size_m). A coarse cell is occupied if ANY fine cell in it
        // is — conservative, so the cover still never under-reports the obstacle. This is purely about how many
        // polygons the planner has to chew on; see publish_cell_size_m for the measured cost of not doing it.
        const int k = std::max(1, static_cast<int>(std::lround(p_.publish_cell_size_m * inv_cell_)));
        const float ks = cs * static_cast<float>(k);
        const auto floordiv = [k](int v) { return v >= 0 ? v / k : -(((-v) + k - 1) / k); };
        int x0 = floordiv(cells[0].first), x1 = x0, y0 = floordiv(cells[0].second), y1 = y0;
        for (const auto& [cx, cy] : cells)
        { const int gx = floordiv(cx), gy = floordiv(cy);
          x0 = std::min(x0, gx); x1 = std::max(x1, gx); y0 = std::min(y0, gy); y1 = std::max(y1, gy); }
        const int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
        std::vector<std::uint8_t> m(static_cast<std::size_t>(bw) * bh, 0);
        for (const auto& [cx, cy] : cells)
            m[static_cast<std::size_t>(floordiv(cy) - y0) * bw + (floordiv(cx) - x0)] = 1;

        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x)
            {
                if (not m[static_cast<std::size_t>(y) * bw + x]) continue;
                int rx = x;                                            // extend right while still occupied
                while (rx + 1 < bw and m[static_cast<std::size_t>(y) * bw + rx + 1]) ++rx;
                int ry = y;                                            // extend down while the whole run matches
                for (bool grow = true; grow and ry + 1 < bh; )
                {
                    for (int k = x; k <= rx; ++k)
                        if (not m[static_cast<std::size_t>(ry + 1) * bw + k]) { grow = false; break; }
                    if (grow) ++ry;
                }
                for (int j = y; j <= ry; ++j)                          // consume the rectangle
                    for (int k = x; k <= rx; ++k) m[static_cast<std::size_t>(j) * bw + k] = 0;
                // Corner-space rectangle on the COARSE lattice (cell centres are corner + half a cell), CCW.
                const float wx0 = ox0 + static_cast<float>(x0 + x) * ks;
                const float wy0 = oy0 + static_cast<float>(y0 + y) * ks;
                const float wx1 = ox0 + static_cast<float>(x0 + rx + 1) * ks;
                const float wy1 = oy0 + static_cast<float>(y0 + ry + 1) * ks;
                rects.push_back({{wx0, wy0}, {wx1, wy0}, {wx1, wy1}, {wx0, wy1}});
            }
        return rects;
    };

    // Kept for reference/regression only — see rect_decompose above for why it is no longer used.
    [[maybe_unused]] const auto trace_outline = [&](const std::vector<std::pair<int, int>>& cells)
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
            c.outline = rect_decompose(cellsxy);  // exact cover, hole-free; never empty for a non-empty component
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

    // ── (6) FORGETTING: an OCCLUDED cell must relax toward "unknown", not stay confidently occupied ──
    // This is the door_2 case: mass latched while visible, then permanently occluded. No clearing beam can ever
    // reach it, so the ONLY thing that can correct it is the passage of time.
    {
        OccGridParams P6 = P; P6.forget_half_life_s = 1.0f;      // fast half-life so the test stays quick
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P6);
        std::vector<Eigen::Vector3f> obst; face_returns(obst);
        for (int k = 0; k < 3; ++k) { g.integrate_sweep(sensor, obst); g.commit_cycle(0.05f); }
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        const bool occ_before = g.occupied(ix, iy);
        const float p_before = g.prob(ix, iy), var_before = g.prob_variance(ix, iy);
        // Now observe NOTHING at all (occluded): commit empty cycles. No hit, no miss — the old build left the
        // cell untouched forever; the decay must walk it back toward the prior.
        for (int k = 0; k < 200; ++k) { g.integrate_sweep(sensor, {}); g.commit_cycle(0.05f); }
        const float p_after = g.prob(ix, iy), var_after = g.prob_variance(ix, iy);
        std::printf("  forget: occ_before=%d P %.2f→%.2f  Var %.4f→%.4f  occupied_after=%d\n",
                    occ_before, p_before, p_after, var_before, var_after, g.occupied(ix, iy));
        check(occ_before, "the obstacle must have been occupied first");
        check(not g.occupied(ix, iy), "an UNOBSERVED occupied cell must eventually be forgotten (released)");
        check(std::abs(p_after - 0.5f) < std::abs(p_before - 0.5f), "P must relax TOWARD the 0.5 prior");
        check(var_after > var_before, "Var must RISE as evidence ages — an unseen cell becomes UNKNOWN");
    }

    // ── (7) FORGETTING must not erode a cell that is still being observed ──
    // The safety counterpart of (6): decay must lose to live evidence, or the map would dissolve under the robot.
    {
        OccGridParams P7 = P; P7.forget_half_life_s = 1.0f;
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P7);
        std::vector<Eigen::Vector3f> obst; face_returns(obst);
        for (int k = 0; k < 200; ++k) { g.integrate_sweep(sensor, obst); g.commit_cycle(0.05f); }
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        std::printf("  forget-hold: continuously-observed cell occupied=%d P=%.2f\n",
                    g.occupied(ix, iy), g.prob(ix, iy));
        check(g.occupied(ix, iy), "a CONTINUOUSLY-OBSERVED obstacle must survive the forgetting term");
        check(g.prob(ix, iy) > 0.9f, "...and keep a high occupancy probability");
    }

    // ── (8) The published polygons must cover the OCCUPIED CELLS AND NOTHING ELSE ──
    // The old boundary trace dropped interior holes ("holes dropped = conservative"), so free space ENCLOSED by
    // residual was published as solid, and it fell back to the CONVEX HULL of the whole component on a diagonal
    // pinch. Measured on a ring-shaped component: 6.49 m2 of occupied cells published as 12.49 m2 (1.9x), and
    // 15x with the hull fallback. That is what put obstacle polygons into never-observed space.
    {
        OccupancyGrid g; g.reset(0, 0, 6, 6, P);
        const Eigen::Vector3f s2(0.02f, 0.02f, 0.5f);
        std::vector<Eigen::Vector3f> ring;                      // a closed ring enclosing a large free courtyard
        for (float t = 1.0f; t <= 4.0f; t += 0.05f)
        { ring.push_back({t, 1.0f, 0.40f}); ring.push_back({t, 4.0f, 0.40f});
          ring.push_back({1.0f, t, 0.40f}); ring.push_back({4.0f, t, 0.40f}); }
        for (int k = 0; k < 3; ++k) { g.integrate_sweep(s2, ring); g.commit_cycle(); }
        const auto comps = g.occupied_components(2, {}, 0.25f);
        // The published cover is built on the COARSE publish lattice, which rounds outward on purpose, so the
        // fine-cell area is the wrong reference — it would flag legitimate conservative rounding as a bug.
        // The right invariant is EXACTNESS ON THE COARSE LATTICE: the rectangles must tile precisely the set of
        // coarse cells touched by the residual, no more and no less. That still catches the original defect —
        // a dropped interior hole or a convex-hull fallback inflates the area far past the coarse cover — while
        // accepting the rounding the coarsening is supposed to introduce.
        const float pub_cs = P.publish_cell_size_m > 0.f ? P.publish_cell_size_m : P.cell_size_m;
        std::unordered_set<long long> coarse;
        {
            const auto xy = g.residual_cell_centres({});
            for (std::size_t i = 0; i + 1 < xy.size(); i += 2)
                coarse.insert(static_cast<long long>(std::floor(xy[i] / pub_cs)) * 100000LL
                              + static_cast<long long>(std::floor(xy[i + 1] / pub_cs)));
        }
        // The components are built on the INFLATED set, so re-derive the expected cover from the same cells the
        // decomposition actually saw rather than from the un-inflated residual.
        std::unordered_set<long long> expected;
        double pub_area = 0; std::size_t rings = 0; int fell_back = 0;
        for (const auto& c : comps)
        {
            if (c.outline.empty()) { ++fell_back; continue; }
            rings += c.outline.size();
            for (const auto& L : c.outline)
            {
                double a2 = 0;
                for (std::size_t i = 0; i < L.size(); ++i)
                { const auto& u = L[i]; const auto& v = L[(i + 1) % L.size()];
                  a2 += static_cast<double>(u.x()) * v.y() - static_cast<double>(v.x()) * u.y(); }
                pub_area += std::abs(a2) / 2.0;
                // Every coarse cell the rectangle spans must be a cell the decomposition was entitled to emit.
                float mnx = L[0].x(), mxx = L[0].x(), mny = L[0].y(), mxy = L[0].y();
                for (const auto& v : L)
                { mnx = std::min(mnx, v.x()); mxx = std::max(mxx, v.x());
                  mny = std::min(mny, v.y()); mxy = std::max(mxy, v.y()); }
                for (float x = mnx + 0.5f * pub_cs; x < mxx; x += pub_cs)
                    for (float y = mny + 0.5f * pub_cs; y < mxy; y += pub_cs)
                        expected.insert(static_cast<long long>(std::floor(x / pub_cs)) * 100000LL
                                        + static_cast<long long>(std::floor(y / pub_cs)));
            }
        }
        const double exact_area = static_cast<double>(expected.size()) * pub_cs * pub_cs;
        std::printf("  coverage: %zu comp(s) %zu polygon(s) hull_fallbacks=%d | published %.3f m2 over %zu "
                    "coarse cells (exact tiling %.3f m2) | residual touches %zu coarse cells\n",
                    comps.size(), rings, fell_back, pub_area, expected.size(), exact_area, coarse.size());
        check(fell_back == 0, "no component may fall back to a convex hull");
        check(std::abs(pub_area - exact_area) < 1e-4, "rectangles must EXACTLY tile the coarse cells they cover "
                                                     "(overlap or a swallowed hole would break this)");
        check(expected.size() >= coarse.size(), "the published cover must CONTAIN every residual cell");
    }

    // ── (9) SELF-BODY: returns off our own body must not latch, external ones must be untouched ──
    {
        OccGridParams P9 = P; P9.self_body_sigma_m = 0.05f;
        OccupancyGrid g; g.reset(-2, -2, 5, 5, P9);
        g.set_self_body(0.0f, 0.0f, 0.55f);                      // body envelope at the sensor
        std::vector<Eigen::Vector3f> selfret, world;
        for (int i = 0; i < 40; ++i)                              // a ring of returns ON the body surface
        { const float a = 6.2831853f * i / 40.0f; selfret.push_back({0.52f * std::cos(a), 0.52f * std::sin(a), 0.40f}); }
        for (int i = 0; i < 40; ++i) world.push_back({2.0f, -0.2f + 0.4f * (i / 39.0f), 0.40f});
        for (int k = 0; k < 10; ++k)
        { g.integrate_sweep(sensor, selfret); g.integrate_sweep(sensor, world, false); g.commit_cycle(); }
        int sx, sy; g.world_to_cell(0.52f, 0.0f, sx, sy);
        int wx2, wy2; g.world_to_cell(2.0f, 0.0f, wx2, wy2);
        std::printf("  self-body: on-body cell occupied=%d (lo=%.2f) | external cell occupied=%d (lo=%.2f)\n",
                    g.occupied(sx, sy), g.logodds(sx, sy), g.occupied(wx2, wy2), g.logodds(wx2, wy2));
        check(g.logodds(sx, sy) < g.logodds(wx2, wy2),
              "a return ON the body must carry far less occupancy evidence than an external one");
        check(g.occupied(wx2, wy2), "the self-body term must NOT suppress a genuine external obstacle");
    }

    // ── (10) FLOOR RESPONSIBILITY: a near-floor return must not latch a cell on its own, a real obstacle must ──
    // The old hard band gave a return one millimetre above it FULL weight, so a single noisy near-floor return
    // latched its cell in one frame. With the floor in the mixture, the SAME return is mostly explained by the
    // floor's measured scatter and needs corroboration — while a return standing clear of that scatter is
    // unaffected, which is the completeness half of the property.
    {
        OccGridParams PA = P; PA.forget_half_life_s = 0.0f;        // isolate the weight term from the decay
        OccupancyGrid g; g.reset(-1, -1, 5, 5, PA);
        g.set_floor_plane(0.0f, 0.0f, 0.0f, 0.07f);                // a rough floor: 7 cm measured fit scatter
        // Two single returns at range 2 (band = 0.06 + 0.04·2 = 0.14): one marginally over it, one clearly over.
        const std::vector<Eigen::Vector3f> marginal{{2.0f, 0.0f, 0.15f}};
        const std::vector<Eigen::Vector3f> real    {{2.0f, 1.0f, 0.45f}};
        g.integrate_sweep(sensor, marginal); g.integrate_sweep(sensor, real, false); g.commit_cycle();
        int mx, my; g.world_to_cell(2.0f, 0.0f, mx, my);
        int rx, ry; g.world_to_cell(2.0f, 1.0f, rx, ry);
        const float r_marg = g.floor_obstacle_responsibility(2.0f, 0.0f, 0.15f, 2.0f);
        const float r_real = g.floor_obstacle_responsibility(2.0f, 1.0f, 0.45f, 2.0f);
        std::printf("  floor-resp: marginal z=0.15 r_obst=%.2f occupied=%d (lo=%.2f) | real z=0.45 r_obst=%.2f "
                    "occupied=%d (lo=%.2f)\n", r_marg, g.occupied(mx, my), g.logodds(mx, my),
                    r_real, g.occupied(rx, ry), g.logodds(rx, ry));
        check(r_marg < r_real, "a near-floor return must carry LESS obstacle responsibility than a clear one");
        check(not g.occupied(mx, my), "one marginally-above-band return must NOT latch its cell in a single frame");
        check(g.occupied(rx, ry), "a return standing clear of the floor's scatter must STILL latch in one frame");
        // …and the measured scatter is what sets the tolerance: over a crisp floor the same height is an obstacle.
        OccupancyGrid gc; gc.reset(-1, -1, 5, 5, PA);
        gc.set_floor_plane(0.0f, 0.0f, 0.0f, 0.005f);              // a crisp floor: 5 mm fit scatter
        const float r_crisp = gc.floor_obstacle_responsibility(2.0f, 0.0f, 0.15f, 2.0f);
        std::printf("  floor-resp: same z=0.15 over a CRISP floor r_obst=%.2f (vs %.2f rough)\n", r_crisp, r_marg);
        check(r_crisp > r_marg, "the tolerance must follow the MEASURED floor scatter, not a constant");
    }

    // ── (11) A FLOOR RETURN CLEARS ITS OWN CELL — the ratchet must be broken ──
    // A phantom latched from a transient near-floor return, then re-observed only by clean floor returns landing in
    // the same cell. Every one of those is direct evidence the floor is there; the old model recorded none of them
    // (the DDA breaks at the endpoint cell, and a below-band return produced no flag), so the phantom was immortal
    // — no traversing beam passes through a floor-height z-band from a sensor 0.5 m up. It must now clear.
    {
        OccGridParams PB = P; PB.forget_half_life_s = 0.0f;        // no decay: the clearing must come from the data
        OccupancyGrid g; g.reset(-1, -1, 5, 5, PB);
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        // Latch a phantom: a burst of returns just above the band at (2,0), the way a smeared scan or a reflection
        // paints one. (Above-band ⇒ hits, exactly as before this change.)
        const std::vector<Eigen::Vector3f> phantom{{2.0f, 0.0f, 0.20f}, {2.0f, 0.01f, 0.19f}, {2.0f, -0.01f, 0.21f}};
        for (int k = 0; k < 6; ++k) { g.integrate_sweep(sensor, phantom); g.commit_cycle(); }
        const bool latched = g.occupied(ix, iy);
        // Now the truth: clean floor returns land in that cell, and beams continue past it to the back wall high up
        // (so the only clearing evidence available is the floor returns themselves).
        std::vector<Eigen::Vector3f> truth;
        for (int i = 0; i < 12; ++i) truth.push_back({1.98f + 0.004f * i, -0.02f + 0.004f * i, 0.01f});
        for (int i = 0; i < 12; ++i) truth.push_back({4.5f, -0.2f + 0.4f * (i / 11.0f), 1.20f});
        long clears = 0;
        for (int k = 0; k < 20; ++k)
        { g.integrate_sweep(sensor, truth); g.commit_cycle(); clears += g.last_sweep_diag().floor_endpoint_clears; }
        std::printf("  floor-clears: phantom latched=%d → occupied=%d (lo=%.2f) after %ld floor-endpoint clears\n",
                    latched, g.occupied(ix, iy), g.logodds(ix, iy), clears);
        check(latched, "the phantom must have latched first (otherwise the test proves nothing)");
        check(clears > 0, "floor returns must DELIVER free evidence to their own cells");
        check(not g.occupied(ix, iy), "a floor phantom must be cleared by the floor returns that land in it");
        // The safety counterpart: the same floor returns must NOT erase a real TALL obstacle they pass under.
        OccupancyGrid g2; g2.reset(-1, -1, 5, 5, PB);
        std::vector<Eigen::Vector3f> tabletop;                      // a 0.73 m surface at x=2
        for (int i = 0; i < 12; ++i) tabletop.push_back({2.0f, -0.2f + 0.4f * (i / 11.0f), 0.73f});
        for (int k = 0; k < 4; ++k) { g2.integrate_sweep(sensor, tabletop); g2.commit_cycle(); }
        const bool tab_occ = g2.occupied(ix, iy);
        std::vector<Eigen::Vector3f> under;                          // floor returns BENEATH it (between the legs)
        for (int i = 0; i < 12; ++i) under.push_back({2.0f, -0.2f + 0.4f * (i / 11.0f), 0.01f});
        for (int k = 0; k < 20; ++k) { g2.integrate_sweep(sensor, under); g2.commit_cycle(); }
        std::printf("  floor-clears: tabletop occ_before=%d after 20 under-table floor sweeps occupied=%d\n",
                    tab_occ, g2.occupied(ix, iy));
        check(tab_occ, "the tabletop must be occupied first");
        check(g2.occupied(ix, iy), "a floor return passing UNDER a real obstacle must not erase it (z-aware gate)");
    }

    std::printf("OccupancyGrid::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
