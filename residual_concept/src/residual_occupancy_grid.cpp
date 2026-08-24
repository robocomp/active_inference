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
    clear_r2_ = p.lidar_clearance_m * p.lidar_clearance_m;
    const float cs = std::max(0.01f, p.cell_size_m);
    xmin_ = xmin; ymin_ = ymin; inv_cell_ = 1.0f / cs;
    w_ = std::max(1, static_cast<int>(std::ceil((xmax - xmin) / cs)));
    h_ = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / cs)));
    lo_.assign(static_cast<std::size_t>(w_) * h_, 0.0f);
    kobs_.assign(lo_.size(), 0.0f);                // accumulated observation weight (0 ⇒ unobserved prior)
    zmn_.assign(lo_.size(), 0.0f);
    zsup_.assign(lo_.size() * OccGridParams::Z_BINS, 0.0f);
    zstab_.assign(lo_.size() * OccGridParams::Z_BINS, 0.0f);
    shbits_.assign(lo_.size(), 0ull);
    smbits_.assign(lo_.size(), 0ull);
    zmx_.assign(lo_.size(), 0.0f);
    dispz_.assign(lo_.size(), 0.0f);
    slook_.assign(lo_.size(), 0);
    szblock_.assign(lo_.size(), 0);
    hit_.assign(lo_.size(), 0);
    occ_.assign(lo_.size(), 0);
    shit_.assign(lo_.size(), 0);
    smiss_.assign(lo_.size(), 0);
    shz_lo_.assign(lo_.size(), 0.0f);
    shz_hi_.assign(lo_.size(), 0.0f);
    shit_w_.assign(lo_.size(), 0.0f);
    smiss_w_.assign(lo_.size(), 0.0f);
    smiss_z_.assign(lo_.size(), 0.0f);
    smiss_src_.assign(lo_.size(), 0);
    seenf_.assign(lo_.size(), 0.0f);
    occ_since_.assign(lo_.size(), 0u);
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

// ── VOXEL SENSOR MODEL ────────────────────────────────────────────────────────────────────────────────────────
// The update rule is the textbook one (robocomp/classes/grid2d: a hit raises the log-odds, a traversal lowers
// it, clamped) applied PER VOXEL instead of per column, and projected to 2-D afterwards. That is
// costmap_2d::VoxelLayer / STVL / OctoMap, and it is what this file should have been from the start.
//
// Everything the 2-D version needed a special term for disappears here, by construction and not by patch:
//   · a beam grazing 2 cm over a tabletop passes through the voxels ABOVE the plate and lowers those. The
//     plate's own voxel is not on the ray at all. (Was: z_band_margin, p_block, the continuous-surface term.)
//   · a beam under a table clears the voxels under the table. (Was: mark_floor_endpoint_flag's support gate.)
//   · a stray return a metre above a table marks its own voxel and nothing else, and later beams through that
//     voxel erase it without ever touching the plate. (Was: the running zmn_/zmx_ hull, then per-bin log-odds.)
//   · the near cone the lidar cannot look into is simply never on any ray, so it is never cleared and stays
//     UNKNOWN. (Was: the sensor-envelope model and its config block.)
//   · a table that is carried away has its voxels crossed by the beams that used to stop on it, so they fall
//     and the column empties. Removal needs no separate channel.
// A hit and a traversal on the SAME voxel in one cycle is hit precedence, as before — but now that is a
// statement about one voxel, not about a whole column.
void OccupancyGrid::mark_hit_voxel(int ix, int iy, int iz, float z, float w)
{
    if (not in_bounds(ix, iy) or iz < 0 or iz >= OccGridParams::Z_BINS) return;
    const int i = idx(ix, iy);
    slook_[i] = 1;
    shbits_[i] |= (1ull << iz);
    if (not shit_[i]) { shz_lo_[i] = z; shz_hi_[i] = z; shit_[i] = 1; shit_w_[i] = w; }
    else { shz_lo_[i] = std::min(shz_lo_[i], z); shz_hi_[i] = std::max(shz_hi_[i], z);
           shit_w_[i] = std::max(shit_w_[i], w); }
}

void OccupancyGrid::mark_free_voxel(int ix, int iy, int iz, float z, float w)
{
    if (not in_bounds(ix, iy) or iz < 0 or iz >= OccGridParams::Z_BINS) return;
    const int i = idx(ix, iy);
    slook_[i] = 1;
    smbits_[i] |= (1ull << iz);
    if (not smiss_[i] or w > smiss_w_[i])
    { smiss_[i] = 1; smiss_w_[i] = w; smiss_z_[i] = z; smiss_src_[i] = sensor_id_; }
}

// Per-voxel evidence: a return in the voxel confirms it, a beam that passed through it without returning
// refutes it, both weighted by the precision w(r) they were collected at, and both clamped — grid2d's rule,
// per voxel. Hit precedence is per voxel, which is the whole point: a beam grazing over a plate refutes the
// voxel it actually crossed and cannot touch the plate's.
// How many height bins of this column hold material — the shape of what is there. One or two ⇒ a plate; dozens
// ⇒ a wall or a person. This is what tells a beam that stopped on a tabletop from one that stopped on a wall.
int OccupancyGrid::column_thickness_bins(int ix, int iy) const
{
    if (not in_bounds(ix, iy)) return 0;
    const float* sup = &zsup_[static_cast<std::size_t>(idx(ix, iy)) * OccGridParams::Z_BINS];
    int n = 0;
    for (int b = 0; b < OccGridParams::Z_BINS; ++b) if (sup[b] > 0.0f) ++n;
    return n;
}

bool OccupancyGrid::voxel_has_material(int ix, int iy, int iz) const
{
    if (not in_bounds(ix, iy) or iz < 0 or iz >= OccGridParams::Z_BINS) return false;
    return zsup_[static_cast<std::size_t>(idx(ix, iy)) * OccGridParams::Z_BINS + iz] > 0.0f;
}

void OccupancyGrid::update_bins(std::size_t i, float w_hit, float w_miss)
{
    const std::uint64_t hb = shbits_[i], mb = smbits_[i];
    if (hb == 0ull and mb == 0ull) return;
    float* sup = &zsup_[i * OccGridParams::Z_BINS];
    float* stb = &zstab_[i * OccGridParams::Z_BINS];
    for (int b = 0; b < OccGridParams::Z_BINS; ++b)
    {
        const std::uint64_t bit = 1ull << b;
        const bool confirm = (hb & bit) != 0ull, refute = not confirm and (mb & bit) != 0ull;
        if (not confirm and not refute) continue;
        // CONSISTENCY: does this observation agree with what the voxel already believes? Agreement earns
        // stiffness slowly, contradiction spends it fast. A voxel with no belief yet (sup == 0) is not yet
        // consistent about anything, so its first observation only starts the count.
        const bool agrees = (confirm and sup[b] >= 0.0f) or (refute and sup[b] <= 0.0f);
        stb[b] = std::clamp(agrees ? stb[b] + p_.stable_rise : stb[b] - p_.stable_fall, 0.0f, 1.0f);
        const float cap = p_.l_clamp * (1.0f + p_.stable_gain * stb[b]);   // earned capacity
        if (confirm) { sup[b] = std::clamp(sup[b] + w_hit  * p_.l_hit,  -cap, cap); ++sd_.bins_confirmed; }
        else         { sup[b] = std::clamp(sup[b] - w_miss * p_.l_miss, -cap, cap); ++sd_.bins_refuted; }
    }
}

void OccupancyGrid::clear_bins(std::size_t i)
{
    float* sup = &zsup_[i * OccGridParams::Z_BINS];
    float* stb = &zstab_[i * OccGridParams::Z_BINS];
    for (int b = 0; b < OccGridParams::Z_BINS; ++b) { sup[b] = 0.0f; stb[b] = 0.0f; }
}

bool OccupancyGrid::has_support(std::size_t i) const
{
    const float* sup = &zsup_[i * OccGridParams::Z_BINS];
    for (int b = 0; b < OccGridParams::Z_BINS; ++b) if (sup[b] > 0.0f) return true;
    return false;
}

// One released cell, with everything needed to judge whether the removal was justified.
void OccupancyGrid::trace_release(std::size_t i, float lo_before, float clear_z, float w, std::uint8_t cause,
                                  float prev_lo, float prev_hi, float prev_top)
{
    ReleaseEvent e;
    cell_to_world(static_cast<int>(i % w_), static_cast<int>(i / w_), e.x, e.y);
    e.lo_before = lo_before; e.lo_after = lo_[i];
    e.zmn = prev_lo; e.zmx = prev_hi; e.last_z = prev_top;   // the band as it stood BEFORE this cycle
    e.src = smiss_src_[i];
    e.clear_z = clear_z; e.clear_w = w;
    e.range_m = observer_valid_ ? std::hypot(e.x - self_x_, e.y - self_y_) : -1.0f;
    e.age_cycles = static_cast<long>(cycle_) - static_cast<long>(occ_since_[i]);
    e.cause = cause;
    releases_.push_back(e);
}

void OccupancyGrid::commit_cycle(float dt_s)
{
    if (not valid()) return;
    ++cycle_;
    releases_.clear();
    // ONE update per VOXEL: hit precedence within the voxel, then the 2-D cell is a PROJECTION of its column.
    // This is costmap_2d::VoxelLayer's contract, with grid2d's log-odds rule as the per-voxel update.
    //
    // FORGETTING still applies to the BELIEF FIELD only (see forget_can_unlatch): the occupancy decision is made
    // from evidence, and absence of observation is not evidence of absence.
    const float gamma = (p_.forget_half_life_s > 0.0f and dt_s > 0.0f)
                      ? std::exp2(-dt_s / p_.forget_half_life_s) : 1.0f;
    const float bw = p_.bin_span_m / OccGridParams::Z_BINS;
    const std::size_t n = lo_.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        // Self-heal: a cell whose ledger has gone non-finite is invisible to every threshold below.
        if (not std::isfinite(lo_[i]))
        { lo_[i] = 0.0f; kobs_[i] = 0.0f; occ_[i] = 0;
          zmn_[i] = zmx_[i] = dispz_[i] = 0.0f; clear_bins(i); ++sd_.cells_repaired; }

        const bool touched = (shbits_[i] != 0ull) or (smbits_[i] != 0ull);
        if (touched)
        {
            update_bins(i, shit_w_[i], smiss_w_[i]);        // the per-voxel log-odds update
            if (shit_[i]) ++sd_.hits; else if (smiss_[i]) ++sd_.misses;
        }
        else if (not occ_[i] and lo_[i] == 0.0f) continue;  // never observed and not latched: nothing to project

        // ── PROJECTION: the column's most-occupied voxel inside the navigable band decides the 2-D cell ──
        // Voxels below the floor band are ground and are never obstacles; voxels above ceil_z are overhead.
        float wx, wy; cell_to_world(static_cast<int>(i % w_), static_cast<int>(i / w_), wx, wy);
        const float floor_z = fp_a_ * wx + fp_b_ * wy + fp_c_ + p_.floor_z0;
        const int b_lo = std::clamp(static_cast<int>(std::floor(floor_z / std::max(1e-6f, bw))),
                                    0, OccGridParams::Z_BINS - 1);
        const int b_hi = std::clamp(static_cast<int>(std::floor(p_.ceil_z / std::max(1e-6f, bw))),
                                    0, OccGridParams::Z_BINS - 1);
        const float* sup = &zsup_[i * OccGridParams::Z_BINS];
        // Only voxels that carry EVIDENCE vote. An unobserved voxel sits at log-odds 0, so including it in the
        // max would peg every column at "unknown" and nothing could ever be released — costmap_2d's VoxelLayer
        // makes the same distinction (a cell is lethal by its MARKED voxel count; unknown voxels do not mark).
        float col = 0.0f; int b_top = -1, b_bot = -1, seen = 0; bool any = false;
        for (int b = b_lo; b <= b_hi; ++b)
        {
            if (sup[b] == 0.0f) continue;                   // never observed: no opinion either way
            if (not any or sup[b] > col) { col = sup[b]; any = true; }
            if (sup[b] > 0.0f) { if (b_bot < 0) b_bot = b; b_top = b; }
            ++seen;                                          // ...and how much of the column we have resolved
        }
        const float lo_before = lo_[i];
        lo_[i] = col;
        // OBSERVABILITY: the fraction of the navigable column any beam has ever settled. A near cell the lidar
        // cannot look into stays UNKNOWN instead of being painted free — no sensor envelope needed, because a
        // voxel no ray reaches simply keeps its prior. See prob().
        // OBSERVABILITY over the COLLISION BAND (see collision_band_top_m), not the whole column.
        const int b_col = p_.collision_band_top_m > 0.0f
                        ? std::clamp(static_cast<int>(std::floor(p_.collision_band_top_m / std::max(1e-6f, bw))),
                                     b_lo, b_hi)
                        : b_lo;
        int groups_seen = 0;
        const int span = std::max(1, b_col - b_lo + 1);
        for (int gI = 0; gI < OccGridParams::COLLISION_GROUPS; ++gI)
        {
            const int gb0 = b_lo + (span * gI) / OccGridParams::COLLISION_GROUPS;
            const int gb1 = b_lo + (span * (gI + 1)) / OccGridParams::COLLISION_GROUPS - 1;
            for (int b = gb0; b <= std::max(gb0, gb1); ++b)
                if (sup[b] != 0.0f) { ++groups_seen; break; }
        }
        seenf_[i] = p_.collision_band_top_m > 0.0f
                  ? static_cast<float>(groups_seen) / OccGridParams::COLLISION_GROUPS : 1.0f;
        // The READ-OUT band, derived from the occupied voxels. hit_ must be maintained here: it is what
        // readout_zband() uses to decide the cell has a height at all, and the explainers score the cell against
        // that band. Dropping it in the rewrite made every occupied cell report a 0-0.05 m band, so the FLOOR
        // explainer claimed the whole map and the agent published nothing — `residual components=0` with
        // occ=2547, and an empty grid in the viewer.
        const float prev_top = dispz_[i];                  // ...remembered BEFORE the wipe below, because a
        const float prev_lo  = zmn_[i];                    // release fires exactly when the column has emptied,
        const float prev_hi  = zmx_[i];                    // and a trace of band[0,0] says nothing about what went
        if (b_top >= 0) { zmn_[i] = b_bot * bw; zmx_[i] = (b_top + 1) * bw; hit_[i] = 1;
                          dispz_[i] += 0.4f * (zmx_[i] - dispz_[i]); }
        else            { hit_[i] = 0; zmn_[i] = 0.0f; zmx_[i] = 0.0f; dispz_[i] = 0.0f; }

        // Hysteresis on the projected column, unchanged: latch at occ_set, release at occ_clear.
        if (col > p_.occ_set and not occ_[i]) { occ_[i] = 1; occ_since_[i] = cycle_; ++sd_.cells_latched; }
        else if (col < p_.occ_clear and occ_[i])
        {
            trace_release(i, lo_before, smiss_z_[i], smiss_w_[i], 0, prev_lo, prev_hi, prev_top);
            occ_[i] = 0; ++sd_.cells_released;
        }

        // ── the BELIEF FIELD — risk + epistemic, for the five agents that consume it ──
        // It now RIDES the projection instead of a second, parallel accumulator. The two had come apart: the old
        // Beta was driven by cell-level hit/miss flags, which after the rewrite mean "some voxel in this column
        // was hit / crossed" — so a beam flying OVER a tabletop set the miss flag and the field reported the cell
        // FREE while the occupancy layer, correctly, still called it blocked. One evidence source, two answers.
        // Now: RISK is sigmoid(column), and this accumulator carries only the CONFIDENCE in that answer —
        // how much observation the cell has banked. See prob() / prob_variance().
        if (touched) kobs_[i] = std::min(p_.beta_kappa_max, kobs_[i] + std::max(shit_w_[i], smiss_w_[i]));
        else if (gamma < 1.0f)
        {
            // Unobserved: the FIELD ages toward "I no longer know" so Var[P] points the epistemic drive at it.
            // The occupancy ledger above is untouched — absence of observation is not evidence of absence.
            if (p_.forget_visible_only and not slook_[i]) ++sd_.cells_unseen;
            else if (p_.forget_occupied_only and lo_[i] <= 0.0f) ++sd_.cells_held;
            else
            {
                ++sd_.cells_decayed;
                float g = gamma;
                if (p_.forget_range_weighted and observer_valid_ and p_.hit_reliable_range_m > 0.0f)
                {
                    const float r0 = p_.hit_reliable_range_m;
                    const float ddx = wx - self_x_, ddy = wy - self_y_;
                    const float wr = (r0 * r0) / (r0 * r0 + ddx * ddx + ddy * ddy);
                    g = 1.0f - (1.0f - gamma) * wr;
                    sd_.decay_weight_sum += wr;
                }
                kobs_[i] *= g;                          // ...confidence ages; the projection itself does not
                if (p_.forget_can_unlatch and lo_[i] < p_.occ_set and occ_[i])
                { occ_[i] = 0; ++sd_.cells_released; ++sd_.cells_forgotten;
                  trace_release(i, lo_before, 0.0f, 0.0f, 1, zmn_[i], zmx_[i], dispz_[i]); clear_bins(i); }
            }
        }
        // (the concentration cap is applied at accumulation, above: bounded memory ⇒ old evidence is discounted)
    }
}

// RISK and CONFIDENCE, from ONE source. p = sigmoid(column) is what the occupancy layer decided; kobs_ is how
// much observation stands behind it. The exported Beta is reconstructed from the pair, with a Jeffreys prior of
// unit weight, so an unobserved cell reads exactly P=0.5 / Var=0.125 as it always did:
//     alpha = 0.5 + kobs*p,  beta = 0.5 + kobs*(1-p)
// Aged (kobs -> 0) it returns to the prior in BOTH channels, which is what property (6) means by relaxing
// toward "I no longer know". Conflicted (much observation, p near 0.5) sits between unknown and confident, as
// property (5) requires — that ordering is what makes Var[P] usable as the epistemic drive.
void OccupancyGrid::cell_belief(int ix, int iy, float& alpha, float& beta) const
{
    if (not in_bounds(ix, iy)) { alpha = beta = 0.5f; return; }
    const int i = idx(ix, iy);
    const float p = column_prob(i);
    const float k = kobs_.empty() ? 0.0f : kobs_[i];
    alpha = 0.5f + k * p;
    beta  = 0.5f + k * (1.0f - p);
}
// P(occupied) from the column: the most-occupied navigable voxel that carries evidence. Unobserved voxels have
// no opinion (see commit_cycle). OCCUPIED is conclusive — one voxel with material settles the cell — but FREE
// is a claim about the WHOLE column, so it is mixed back toward the prior by the fraction actually resolved.
// That is what keeps the near cone the lidar cannot look into reading UNKNOWN rather than free.
float OccupancyGrid::column_prob(int i) const
{
    const float bw = p_.bin_span_m / OccGridParams::Z_BINS;
    float wx, wy; cell_to_world(i % w_, i / w_, wx, wy);
    const float floor_z = fp_a_ * wx + fp_b_ * wy + fp_c_ + p_.floor_z0;
    const int b_lo = std::clamp(static_cast<int>(std::floor(floor_z / std::max(1e-6f, bw))),
                                0, OccGridParams::Z_BINS - 1);
    const int b_hi = std::clamp(static_cast<int>(std::floor(p_.ceil_z / std::max(1e-6f, bw))),
                                0, OccGridParams::Z_BINS - 1);
    const float* sup = &zsup_[static_cast<std::size_t>(i) * OccGridParams::Z_BINS];
    float col = 0.0f; bool any = false;
    for (int b = b_lo; b <= b_hi; ++b)
        if (sup[b] != 0.0f and (not any or sup[b] > col)) { col = sup[b]; any = true; }
    const float p = 1.0f / (1.0f + std::exp(-col));
    if (col >= 0.0f) return p;
    const float f = seenf_.empty() ? 1.0f : seenf_[i];
    return f * p + (1.0f - f) * 0.5f;
}
float OccupancyGrid::prob(int ix, int iy) const
{
    float a, b; cell_belief(ix, iy, a, b);
    return a / std::max(1e-9f, a + b);
}
float OccupancyGrid::prob_variance(int ix, int iy) const
{
    float a, b; cell_belief(ix, iy, a, b);
    const float k = a + b;
    return a * b / std::max(1e-9f, k * k * (k + 1.0f));          // Var of Beta(α,β) = αβ/((α+β)²(α+β+1))
}
float OccupancyGrid::prob_std(int ix, int iy) const
{
    return std::sqrt(std::max(0.0f, prob_variance(ix, iy)));
}

void OccupancyGrid::occupancy_fields(std::vector<float>& prob_out, std::vector<float>& var_out,
                                     const CellExplained& explained) const
{
    const std::size_t n = kobs_.size();
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
            float a, b; cell_belief(x, y, a, b);
            const float k = a + b;
            prob_out[i] = keep * (a / std::max(1e-9f, k));      // an object owns this cell (keep→0) ⇒ no residual risk
            var_out[i]  = keep * (a * b / std::max(1e-9f, k * k * (k + 1.0f)));   // ...and no epistemic pull
        }
}

void OccupancyGrid::integrate_sweep(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& points_room,
                                    bool begin_cycle, float reliability,
                                    const std::vector<float>* hit_weight_scale,
                                    const std::vector<std::uint8_t>* mark_mask)
{
    if (not valid()) return;
    const bool use_scale = hit_weight_scale != nullptr and hit_weight_scale->size() == points_room.size();
    const bool use_mask  = mark_mask != nullptr and mark_mask->size() == points_room.size();
    if (begin_cycle)                                       // first sensor of the cycle: reset diagnostics + scratch
    {
        sd_ = SweepDiag{};
        std::fill(shit_.begin(),  shit_.end(),  0);
        std::fill(smiss_.begin(), smiss_.end(), 0);
        std::fill(slook_.begin(), slook_.end(), 0);
        std::fill(shbits_.begin(), shbits_.end(), 0ull);
        std::fill(smbits_.begin(), smbits_.end(), 0ull);
        std::fill(shit_w_.begin(), shit_w_.end(), 0.0f);
        std::fill(smiss_w_.begin(), smiss_w_.end(), 0.0f);
        std::fill(smiss_z_.begin(), smiss_z_.end(), 0.0f);
        std::fill(smiss_src_.begin(), smiss_src_.end(), 0);
    }
    const float cs = 1.0f / inv_cell_;
    const float bw = p_.bin_span_m / OccGridParams::Z_BINS;                  // voxel height
    const float ox = origin.x(), oy = origin.y(), oz = origin.z();
    const float r0 = p_.hit_reliable_range_m;              // precision falloff scale (0 ⇒ uniform full weight)
    const float rel = std::clamp(reliability, 0.0f, 1.0f); // global ego-motion trust for this sweep
    const auto hit_w = [&](float r) { return r0 > 0.0f ? rel * (r0 * r0) / (r0 * r0 + r * r) : rel; };
    // SELF-BODY term: P(this return came from the WORLD, not off our own body) = Φ(s/σ). HITS only — freeing
    // space is always safe, and the sensor's own voxel already emits a traversal.
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
        // A non-finite return (the ZED emits them where depth is invalid) must never reach the sensor model.
        if (not (std::isfinite(px) and std::isfinite(py) and std::isfinite(pz))) { ++sd_.bad_points; continue; }

        const float dx = px - ox, dy = py - oy, dz = pz - oz;
        const float range = std::hypot(dx, dy);                 // horizontal range (the precision covariate)
        // Obstacle band referenced to the DATA-DRIVEN floor plane. This is now purely a MARKING gate — the ray is
        // traced either way, so a floor return still clears everything it passed through (costmap_2d's
        // marking-vs-clearing rule, and what mark_floor_endpoint_flag used to have to reconstruct by hand).
        const float floor_z = fp_a_ * px + fp_b_ * py + fp_c_;
        const float band_top = floor_z + device_floor_z0() + p_.floor_slope * range;
        const bool in_band = (pz > band_top) and (pz < p_.ceil_z);
        if (not in_band) ++sd_.floor_endpoint_returns;

        const float self_w = world_w(px, py);
        if (self_w < 0.99f) ++sd_.self_hits_damped;
        const float floor_w = floor_obstacle_responsibility(px, py, pz, range);
        if (in_band and floor_w < 0.9f) ++sd_.floor_damped_hits;
        const float w_hit = hit_w(range) * self_w * floor_w * (use_scale ? (*hit_weight_scale)[pi] : 1.0f);

        const bool may_mark = not use_mask or (*mark_mask)[pi] != 0;
        if (not may_mark) ++sd_.marks_suppressed;

        int ex, ey; const bool endp_in = world_to_cell(px, py, ex, ey);
        const int ez = static_cast<int>(std::floor(pz / std::max(1e-6f, bw)));
        const auto place_endpoint = [&]
        {
            if (not endp_in) return;
            if (in_band) { if (may_mark) mark_hit_voxel(ex, ey, ez, pz, w_hit); }
            // A BELOW-BAND return is not neutral: the beam reached the floor inside this cell, so the voxel it
            // landed in was empty. It marks nothing and clears its own voxel — which is all that
            // mark_floor_endpoint_flag and its support gate used to have to reconstruct by hand.
            else { mark_free_voxel(ex, ey, ez, pz, hit_w(range)); ++sd_.floor_endpoint_clears; }
        };

        const float L = std::sqrt(dx * dx + dy * dy + dz * dz);             // 3-D ray length
        int cx, cy;
        if (L < 1e-4f or not world_to_cell(ox, oy, cx, cy)) { place_endpoint(); continue; }
        int cz = static_cast<int>(std::floor(oz / std::max(1e-6f, bw)));

        // ── 3-D DDA (Amanatides & Woo) over voxels of cs x cs x bw ──────────────────────────────────────────
        const float ux = dx / L, uy = dy / L, uz = dz / L;
        const int stepx = ux > 0 ? 1 : -1, stepy = uy > 0 ? 1 : -1, stepz = uz > 0 ? 1 : -1;
        const float BIG = std::numeric_limits<float>::max();
        const float nbx = xmin_ + (cx + (stepx > 0 ? 1 : 0)) * cs;
        const float nby = ymin_ + (cy + (stepy > 0 ? 1 : 0)) * cs;
        const float nbz =          (cz + (stepz > 0 ? 1 : 0)) * bw;
        float tMaxX = std::abs(ux) > 1e-9f ? (nbx - ox) / ux : BIG;
        float tMaxY = std::abs(uy) > 1e-9f ? (nby - oy) / uy : BIG;
        float tMaxZ = std::abs(uz) > 1e-9f ? (nbz - oz) / uz : BIG;
        const float tDx = std::abs(ux) > 1e-9f ? cs / std::abs(ux) : BIG;
        const float tDy = std::abs(uy) > 1e-9f ? cs / std::abs(uy) : BIG;
        const float tDz = std::abs(uz) > 1e-9f ? bw / std::abs(uz) : BIG;

        // DO NOT CLEAR CLOSE TO THE HIT (see OccGridParams::clear_stop_max_m): skip the run over which the beam
        // has not yet changed voxel height, because over that stretch the sensor cannot say which voxel first
        // stopped it. One voxel for a steep beam; the whole grazing run for a shallow one over a flat surface.
        // This sweep's endpoint uncertainty at this range, and the clearing authority it earns.
        const float sig_e   = sens_s0_ > 0.0f ? sens_s0_ + sens_quad_ * range * range : p_.reference_sigma_m;
        const float sref    = p_.reference_sigma_m;
        const float prec_w  = (sref * sref) / std::max(sref * sref, sig_e * sig_e);   // 1 for a lidar-grade sweep
        const float d_stop  = std::max(std::min(p_.clear_stop_max_m, bw / std::max(std::abs(uz), 1e-4f)),
                                       p_.clear_stop_sigma_k * sig_e);
        // The sensor's own voxel and everything inside the dead shell carry NO free evidence: below its minimum
        // range the device returns nothing whatever is there. (This used to be marked free at hit_w(0) — the
        // largest weight in the grid.)
        if (sensor_min_r_ <= 0.0f and clear_r2_ <= 0.0f)
            mark_free_voxel(cx, cy, cz, oz, hit_w(0.0f) * prec_w);   // the sensor's own cell is inside the disc
        else ++sd_.clear_blind_shell;

        // What did this beam stop ON? A thin endpoint column is a horizontal surface it may have skimmed; a tall
        // one is a wall, which it cannot have. See OccGridParams::plate_bins_max.
        const bool endpoint_is_plate = endp_in and column_thickness_bins(ex, ey) <= p_.plate_bins_max;

        int guard = 4 * (w_ + h_ + OccGridParams::Z_BINS);
        while (guard-- > 0)
        {
            float t_here;
            if (tMaxX < tMaxY and tMaxX < tMaxZ)      { cx += stepx; t_here = tMaxX; tMaxX += tDx; }
            else if (tMaxY < tMaxZ)                   { cy += stepy; t_here = tMaxY; tMaxY += tDy; }
            else                                      { cz += stepz; t_here = tMaxZ; tMaxZ += tDz; }
            if (cx == ex and cy == ey and cz == ez) break;   // reached the endpoint voxel (handled below)
            if (t_here >= L)                        break;   // overshot the return
            if (not in_bounds(cx, cy))              break;   // left the grid
            // INSIDE THE DEAD SHELL the device returns nothing whatever is there, so this crossing is not an
            // observation and carries no free evidence. See set_sensor_min_range: this is what stopped the map
            // deleting a table the moment the robot closed on it.
            if (t_here < sensor_min_r_) { ++sd_.clear_blind_shell; continue; }
            // ...and NOTHING inside the robot's lidar clearance radius may be cleared, whichever sensor the ray
            // came from. See OccGridParams::lidar_clearance_m.
            if (clear_r2_ > 0.0f and observer_valid_)
            {
                float wx2, wy2; cell_to_world(cx, cy, wx2, wy2);
                const float ddx = wx2 - self_x_, ddy = wy2 - self_y_;
                if (ddx * ddx + ddy * ddy < clear_r2_) { ++sd_.clear_blind_shell; continue; }
            }
            // weight by the range of the CLEARED voxel, not of the return: a far see-through is weak evidence.
            // ONE SURFACE EXPLAINS BOTH: a voxel that already holds material and sits at the SAME HEIGHT as the
            // voxel the beam stopped in is plausibly the very surface that stopped it — the beam skimmed along
            // it and struck it further on. That crossing refutes nothing. A distance rule cannot express this:
            // the grazing run across a 1.2 m tabletop is far longer than any endpoint margin worth having
            // (measured: capping at 0.35 m still left 1246 releases). Stated in voxels it needs no parameter.
            // A beam ending on a WALL is untouched — the voxels it crosses at that height hold no material, so
            // they clear exactly as before. A table that has GONE is untouched too: those beams now end on the
            // floor, whose bin is nowhere near the plate's, so the plate's voxels clear.
            // ★ I tried ALSO counting the cell being LATCHED as material here, to stop the rule collapsing once a
            // voxel is first eroded. It measured better on a standing table (835 -> 381 releases) and it is a
            // LEAK: latched ⇒ protected ⇒ stays latched, so a table carried away kept 202 of its 384 cells 220
            // cycles later, against 20 without it. An obstacle that cannot be removed shreds the free space the
            // planner needs just as surely as one that vanishes. Voxel material only.
            if (p_.clear_stop_max_m > 0.0f and endpoint_is_plate
                and std::abs(cz - ez) <= 1 and voxel_has_material(cx, cy, cz))
            { ++sd_.clear_stopped; continue; }
            if (L - t_here < d_stop) { ++sd_.clear_stopped; continue; }      // ...and costmap_2d's endpoint rule
            const float rr = t_here * (range / std::max(1e-6f, L));          // its horizontal range
            if (prec_w < 0.99f) ++sd_.clear_imprecise;
            mark_free_voxel(cx, cy, cz, oz + t_here * uz, hit_w(rr) * prec_w);
        }
        place_endpoint();
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
        // The endpoints span z 0..0.9 so that, INTERPOLATED at x = 2, the beams cross every height the obstacle
        // ever claimed (0.25-0.55). Firing only at the obstacle's own endpoint heights does not do that — from a
        // 0.5 m sensor those beams cross x = 2 at 0.39-0.52 and never look at 0.25, so the support model
        // correctly refuses to declare that height empty. A carve test must actually LOOK where it claims.
        std::vector<Eigen::Vector3f> clear;
        // The z step must be fine enough that the beams cross EVERY voxel the obstacle claimed. At 0.1 the fan
        // crossed bins 8,10,11,13,14,16,17,... and skipped bin 12 — where one of the three returns lives — so the
        // column's max stayed positive and the cell correctly refused to clear. Refutation is per voxel now: a
        // carve test has to sweep the column, not just its endpoints.
        for (int i = 0; i < 40; ++i) for (float z = -0.2f; z <= 1.0f; z += 0.02f)
            clear.push_back({4.5f, -0.2f + 0.4f * (i / 39.0f), z});
        for (int k = 0; k < 30; ++k) { g.integrate_sweep(sensor, clear); g.commit_cycle(); }
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

    // ── (5) THE BELIEF FIELD SEPARATES "NEVER LOOKED" FROM "BARELY LOOKED" FROM "CONFIRMED" ──
    // Var[P] is the epistemic term the planner and five concept agents read, so it has to order those three.
    // NOTE the middle case changed with the voxel rewrite: alternating hit/miss no longer produces an ambiguous
    // cell, because hit precedence is per VOXEL and the sensor model is deliberately biased toward occupied
    // (l_hit > l_miss). The middle case that matters now is a cell seen BRIEFLY — real evidence, little of it —
    // which is exactly what a distant or glancing observation gives you.
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> obst; face_returns(obst);
        int ux, uy; g.world_to_cell(0.5f, 1.5f, ux, uy);        // never observed
        const float p_unknown = g.prob(ux, uy), var_unknown = g.prob_variance(ux, uy);
        for (int k = 0; k < 2; ++k) { g.integrate_sweep(sensor, obst); g.commit_cycle(0.1f); }
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        const float p_brief = g.prob(ix, iy), var_brief = g.prob_variance(ix, iy);
        for (int k = 0; k < 200; ++k) { g.integrate_sweep(sensor, obst); g.commit_cycle(0.1f); }
        const float p_conf = g.prob(ix, iy), var_conf = g.prob_variance(ix, iy);
        std::printf("  belief: unknown[P=%.2f var=%.4f] brief[P=%.2f var=%.4f] confirmed[P=%.2f var=%.4f]\n",
                    p_unknown, var_unknown, p_brief, var_brief, p_conf, var_conf);
        check(std::abs(p_unknown - 0.5f) < 1e-3f, "a never-observed cell must read exactly the 0.5 prior");
        check(var_unknown > var_brief, "UNKNOWN must be more uncertain than a BRIEFLY observed cell");
        check(var_brief   > var_conf,  "...and a briefly observed cell more uncertain than a CONFIRMED one");
        check(p_conf > 0.9f,           "a continuously-hit cell must have HIGH occupancy probability (risk)");
    }

    // ── (6) FORGETTING: an OCCLUDED cell must relax toward "unknown" — WITHOUT being released ──
    // This is the door_2 case: mass latched while visible, then permanently occluded. Its BELIEF must age (P→0.5,
    // Var→max) so the epistemic channel marks it worth re-observing. Its LATCH must not move: no beam contradicted
    // it, and a safety layer under maximum uncertainty calls a cell occupied. The control at the end restores the
    // old coupling (forget_can_unlatch) and shows the same cell being released by time alone.
    {
        // Isolates the time-decay DYNAMICS. It simulates "unobserved" with an EMPTY sweep, which under the
        // visibility gate means "sensor off" — no beam reaches anything, so the gate would (correctly) hold the
        // whole map and this property could never fire. Property (15) owns the visibility axis; turn the gate
        // off here so this one keeps testing exactly what it always tested.
        OccGridParams P6 = P; P6.forget_half_life_s = 1.0f;      // fast half-life so the test stays quick
        P6.forget_visible_only = false;                          // see above — (15) owns this axis
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
        check(g.occupied(ix, iy), "an UNOBSERVED occupied cell must NOT be released — no beam contradicted it");
        check(std::abs(p_after - 0.5f) < std::abs(p_before - 0.5f), "P must relax TOWARD the 0.5 prior");
        check(var_after > var_before, "Var must RISE as evidence ages — an unseen cell becomes UNKNOWN");
        // (the old forget_can_unlatch control is gone with the term: lo_ is recomputed from the voxel
        //  column every cycle, so a time decay has nothing to erode. Removal is evidence-only.)
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

    // ── (11) A BEAM THAT REACHES THE FLOOR CLEARS WHAT IT PASSED THROUGH — AND ONLY THAT ──
    // In the 2-D build this needed a dedicated path (mark_floor_endpoint_flag) with a support gate, because a
    // terminating floor return never overlapped a cell's remembered z-band and was discarded — the ratchet that
    // left `floor_clears` at 0 on 9381 of 9381 cycles. In 3-D it is free: the ray clears the voxels it actually
    // traverses, its endpoint is below the nav band so it marks nothing, and a beam that passed UNDER a tabletop
    // never touched the tabletop's voxel. Both halves are asserted, because the whole point is that clearing the
    // floor must not cost us the furniture standing on it.
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        const Eigen::Vector3f eye(0.02f, 0.02f, 0.60f);
        // (a) a phantom at floor height, then honest floor returns landing in the same cell.
        std::vector<Eigen::Vector3f> phantom;
        for (int i = 0; i < 12; ++i) phantom.push_back({2.0f, -0.1f + 0.2f * (i / 11.0f), 0.30f});
        for (int k = 0; k < 6; ++k) { g.integrate_sweep(eye, phantom); g.commit_cycle(0.1f); }
        int px, py; g.world_to_cell(2.0f, 0.0f, px, py);
        const bool phantom_occ0 = g.occupied(px, py);
        std::vector<Eigen::Vector3f> floorret;                   // returns ON the floor, beyond and at the cell
        // Reaching well BEYOND the cell: a ray only crosses 0.30 m at x = 2 if its floor endpoint is far enough
        // out. Stopping the sweep at x = 3 crosses that column at 0.20 m and refutes the wrong voxel — the
        // recurring lesson, in yet another form.
        for (float x = 1.0f; x <= 4.5f; x += 0.02f)
            for (int i = 0; i < 12; ++i) floorret.push_back({x, -0.1f + 0.2f * (i / 11.0f), 0.005f});
        for (int k = 0; k < 60; ++k) { g.integrate_sweep(eye, floorret); g.commit_cycle(0.1f); }

        // (b) a real TABLETOP at 0.75, then the same floor sweep passing UNDER it. It must survive.
        OccupancyGrid g2; g2.reset(-1, -1, 5, 5, P);
        const Eigen::Vector3f high(0.02f, 0.02f, 1.075f);
        std::vector<Eigen::Vector3f> top;
        for (float x = 1.6f; x <= 2.4f; x += 0.02f)
            for (int i = 0; i < 12; ++i) top.push_back({x, -0.1f + 0.2f * (i / 11.0f), 0.75f});
        for (int k = 0; k < 10; ++k) { g2.integrate_sweep(high, top); g2.commit_cycle(0.1f); }
        int tx, ty; g2.world_to_cell(2.0f, 0.0f, tx, ty);
        const bool top_occ0 = g2.occupied(tx, ty);
        const Eigen::Vector3f low(0.02f, 0.02f, 0.30f);          // a LOW sensor: its beams pass under the table
        for (int k = 0; k < 60; ++k) { g2.integrate_sweep(low, floorret); g2.commit_cycle(0.1f); }

        std::printf("  floor-clears: phantom occ %d->%d (lo=%.2f) | tabletop occ %d->%d after 60 under-table "
                    "floor sweeps\n", phantom_occ0, g.occupied(px, py), g.logodds(px, py),
                    top_occ0, g2.occupied(tx, ty));
        check(phantom_occ0 and top_occ0, "both must be occupied before the floor sweeps");
        check(not g.occupied(px, py), "a floor-height phantom must be cleared by floor returns landing in it");
        check(g2.occupied(tx, ty), "...while a TABLETOP must survive beams that merely passed underneath it");
    }

    // ── (12) PER-DEVICE NAV BAND — marking must be UNCHANGED, clearing must be RECOVERED ──
    // The worker used to delete each device's near-floor returns before integration, which also deleted the
    // free-space evidence mark_floor_endpoint_flag exists to bank (measured: floor_clears == 0 on 9381 of 9381
    // live cycles, while this very self_test passed — because the test feeds raw floor returns straight in).
    // set_device_floor_z0 moves that band into the sensor model. The property that makes the change safe to ship
    // is EQUIVALENCE of marking: a return the old filter would have deleted must still not mark, and a return it
    // would have kept must still mark, at the SAME z0. Only the clearing is new.
    {
        OccGridParams PC = P; PC.forget_half_life_s = 0.0f;
        const float hz0 = 0.20f;                                   // helios's band — well above PC.floor_z0 (0.06)
        // (a) a return INSIDE the device band (0.15 m, i.e. above the default 0.06 band but below helios's 0.20)
        //     must NOT mark, and must instead clear its own cell. Under the default band it WOULD have marked —
        //     that is exactly the helios grazing bias the per-device band exists to absorb.
        OccupancyGrid gd; gd.reset(-1, -1, 5, 5, PC);
        int ix, iy; gd.world_to_cell(2.0f, 0.0f, ix, iy);
        std::vector<Eigen::Vector3f> graze;
        for (int i = 0; i < 12; ++i) graze.push_back({2.0f, -0.02f + 0.004f * i, 0.15f});
        gd.set_device_floor_z0(hz0);
        for (int k = 0; k < 6; ++k) { gd.integrate_sweep(sensor, graze); gd.commit_cycle(); }
        const bool dev_marked = gd.occupied(ix, iy);
        const long dev_clears = gd.last_sweep_diag().floor_endpoint_clears;
        const long dev_rets   = gd.last_sweep_diag().floor_endpoint_returns;
        // (b) the SAME returns with the band unset must mark — proving the band is what changed, not the data.
        OccupancyGrid gu; gu.reset(-1, -1, 5, 5, PC);
        gu.set_device_floor_z0(-1.0f);
        for (int k = 0; k < 6; ++k) { gu.integrate_sweep(sensor, graze); gu.commit_cycle(); }
        // (c) a return ABOVE the device band must still mark in one frame — completeness is not weakened.
        // NB the band is z0 + floor_slope·range, so at range 2 m with z0=0.20 it sits at 0.28, not 0.20; and the
        // floor RESPONSIBILITY still discounts anything close to it. 0.50 m is unambiguously an obstacle.
        OccupancyGrid ga; ga.reset(-1, -1, 5, 5, PC);
        std::vector<Eigen::Vector3f> real;
        for (int i = 0; i < 12; ++i) real.push_back({2.0f, -0.02f + 0.004f * i, 0.50f});
        ga.set_device_floor_z0(hz0);
        ga.integrate_sweep(sensor, real); ga.commit_cycle();
        std::printf("  device-band: graze@0.15 band=0.20 occupied=%d (rets=%ld clears=%ld) | same graze band=off "
                    "occupied=%d | obstacle@0.50 band=0.20 occupied=%d\n",
                    dev_marked, dev_rets, dev_clears, gu.occupied(ix, iy), ga.occupied(ix, iy));
        check(dev_rets > 0, "a below-device-band return must REACH the grid as a floor return (not be deleted)");
        check(dev_clears > 0, "...and must deliver its free evidence to its own cell");
        check(not dev_marked, "a return inside the device's floor band must not latch it as an obstacle");
        check(gu.occupied(ix, iy), "the same return under the DEFAULT band must latch — the band is the variable");
        check(ga.occupied(ix, iy), "a return above the device band must still latch in one frame (completeness)");
    }

    // ── (13) MARK MASK — a masked return clears its ray but must leave NO trace at its endpoint ──
    // The ZED path used to delete its floor/ceiling/wall points, losing the longest rays in the frame. Masking
    // instead keeps the ray. The subtle requirement is that a masked endpoint must not install a z-band either:
    // routing this through a zero HIT WEIGHT would still call mark_hit_flag, which sets hit_/zmn_/zmx_ and would
    // manufacture a fresh clearing gate out of no evidence at all — the exact ratchet we are removing.
    {
        OccGridParams PD = P; PD.forget_half_life_s = 0.0f;
        OccupancyGrid g; g.reset(-1, -1, 5, 5, PD);
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        int mx, my; g.world_to_cell(1.0f, 0.0f, mx, my);            // a cell the ray crosses on its way there
        std::vector<Eigen::Vector3f> wall;
        for (int i = 0; i < 12; ++i) wall.push_back({2.0f, -0.02f + 0.004f * i, 0.60f});
        const std::vector<std::uint8_t> mask(wall.size(), 0);       // every return: trace, do not mark
        for (int k = 0; k < 8; ++k)
        { g.integrate_sweep(sensor, wall, true, 1.0f, nullptr, &mask); g.commit_cycle(); }
        const long suppressed = g.last_sweep_diag().marks_suppressed;
        std::printf("  mark-mask: endpoint occupied=%d (lo=%.2f) | traversed cell lo=%.2f | suppressed=%ld/sweep\n",
                    g.occupied(ix, iy), g.logodds(ix, iy), g.logodds(mx, my), suppressed);
        check(suppressed == static_cast<long>(wall.size()), "every masked return must be counted as suppressed");
        check(not g.occupied(ix, iy), "a masked return must not mark its endpoint occupied");
        check(g.logodds(ix, iy) <= 0.0f, "...and must not push its endpoint's log-odds up at all");
        check(g.logodds(mx, my) < 0.0f, "...while its RAY must still clear the cells it traverses");
        // And the unmasked control: the same returns must mark when the mask allows it.
        OccupancyGrid gm; gm.reset(-1, -1, 5, 5, PD);
        gm.integrate_sweep(sensor, wall); gm.commit_cycle();
        check(gm.occupied(ix, iy), "the same returns must latch when marking is NOT masked");
    }

    // ── (14) FORGETTING IS ASYMMETRIC — free space must NOT be un-learned ──
    // Property (6) already proves an unobserved OCCUPIED cell relaxes toward unknown. Its counterpart is that an
    // unobserved FREE cell must NOT: the robot paid beams to establish that free space, and a cell drifting back
    // to P=0.5 because we looked away is the map un-learning its own work. Invisible to the polygon channel (a
    // free cell cannot re-latch) but published in the FIELD, which five concept agents consume for birth gating.
    // Both halves are asserted here so the asymmetry cannot be silently reverted from either side.
    {
        // Same reasoning as (6): this property isolates the OCCUPIED-vs-FREE axis and drives it with empty
        // sweeps, so the visibility gate must be off or nothing decays at all and both halves pass vacuously.
        OccGridParams PE = P; PE.forget_half_life_s = 1.0f;        // fast half-life so the test stays quick
        PE.forget_visible_only = false;                            // (15) owns the visibility axis
        OccupancyGrid g; g.reset(-1, -1, 5, 5, PE);
        int fx, fy; g.world_to_cell(1.0f, 0.0f, fx, fy);           // a cell the beams pass THROUGH → free
        int ox, oy; g.world_to_cell(3.0f, 0.0f, ox, oy);           // ...and the wall they end on → occupied
        // A FAN reaching BELOW the sensor, not one height: "this cell is free" is a claim about the collision
        // band, and the read-out now discounts a band the beams never resolved. One horizontal sweep at 0.6 m
        // says nothing about whether something is standing at 0.2 m, and must not be allowed to claim it does.
        std::vector<Eigen::Vector3f> wall;
        for (float z = -1.0f; z <= 1.8f; z += 0.03f)
            for (int i = 0; i < 12; ++i) wall.push_back({3.0f, -0.2f + 0.4f * (i / 11.0f), z});
        for (int k = 0; k < 12; ++k) { g.integrate_sweep(sensor, wall); g.commit_cycle(0.05f); }
        const float free_p0 = g.prob(fx, fy), free_v0 = g.prob_variance(fx, fy);
        const bool occ0 = g.occupied(ox, oy);
        // 200 cycles observing NOTHING at all.
        const std::vector<Eigen::Vector3f> nothing;
        long decayed = 0, held = 0;
        for (int k = 0; k < 200; ++k)
        { g.integrate_sweep(sensor, nothing); g.commit_cycle(0.05f);
          decayed += g.last_sweep_diag().cells_decayed; held += g.last_sweep_diag().cells_held; }
        const float free_p1 = g.prob(fx, fy), free_v1 = g.prob_variance(fx, fy);
        std::printf("  forget-asym: FREE cell P %.4f→%.4f Var %.4f→%.4f | wall occ_before=%d after=%d | "
                    "decayed=%ld held=%ld\n",
                    free_p0, free_p1, free_v0, free_v1, occ0, g.occupied(ox, oy), decayed, held);
        check(free_p0 < 0.10f, "the traversed cell must read as FREE before the idle period");
        check(std::abs(free_p1 - free_p0) < 1e-4f, "an unobserved FREE cell must NOT drift back toward 0.5");
        check(std::abs(free_v1 - free_v0) < 1e-4f, "...and must not lose confidence either");
        check(occ0, "the wall must be occupied before the idle period");
        check(g.occupied(ox, oy), "an unobserved OCCUPIED cell must still be HELD (property 6 holds)");
        check(held > 0, "free/unknown cells must be COUNTED as held, not silently skipped");
        // The control: with the asymmetry off, the same free cell DOES drift — proving the flag is the variable
        // and this test is not passing for some unrelated reason.
        OccGridParams PF = PE; PF.forget_occupied_only = false;
        OccupancyGrid g2; g2.reset(-1, -1, 5, 5, PF);
        for (int k = 0; k < 12; ++k) { g2.integrate_sweep(sensor, wall); g2.commit_cycle(0.05f); }
        const float sym_p0 = g2.prob(fx, fy);
        for (int k = 0; k < 200; ++k) { g2.integrate_sweep(sensor, nothing); g2.commit_cycle(0.05f); }
        std::printf("  forget-asym: control (symmetric) FREE cell P %.4f→%.4f\n", sym_p0, g2.prob(fx, fy));
        check(g2.prob(fx, fy) - sym_p0 > 0.05f, "the OLD symmetric rule must measurably un-learn free space");
    }

    // ── (15) A BEAM THAT FLEW OVER AN OBSTACLE IS NOT A REFUTATION OF IT ──
    // The 2-D build needed a z-gate plus a visibility counter to express this, and got it wrong twice. In 3-D it
    // is arithmetic: the beam lowers the voxels it passed through, which are not the obstacle's. Three arms —
    // flown over (held), never reached (held), and genuinely seen through (cleared) — so the model cannot pass
    // by simply refusing to clear anything.
    {
        OccGridParams PG = P; PG.forget_half_life_s = 1.0f;   // fast half-life so the test stays quick
        const Eigen::Vector3f eye(0.02f, 0.02f, 0.50f);
        std::vector<Eigen::Vector3f> low;                     // an obstacle at 1 m, top at 0.30
        for (int i = 0; i < 12; ++i) low.push_back({1.0f, -0.1f + 0.2f * (i / 11.0f), 0.30f});
        const auto build = [&](OccupancyGrid& g)
        { g.reset(-1, -1, 6, 6, PG);
          for (int k = 0; k < 8; ++k) { g.integrate_sweep(eye, low); g.commit_cycle(0.05f); } };

        OccupancyGrid g;  build(g);
        int lx, ly; g.world_to_cell(1.0f, 0.0f, lx, ly);
        const bool occ0 = g.occupied(lx, ly);
        std::vector<Eigen::Vector3f> over;                    // same bearing, passing well ABOVE the obstacle
        for (int i = 0; i < 12; ++i) over.push_back({5.0f, -0.5f + 1.0f * (i / 11.0f), 1.50f});
        for (int k = 0; k < 200; ++k) { g.integrate_sweep(eye, over); g.commit_cycle(0.05f); }

        OccupancyGrid g2; build(g2);                          // nothing points at it at all
        std::vector<Eigen::Vector3f> away;
        for (int i = 0; i < 12; ++i) away.push_back({-0.8f, -0.5f + 1.0f * (i / 11.0f), 0.60f});
        long unseen = 0;
        for (int k = 0; k < 200; ++k)
        { g2.integrate_sweep(eye, away); g2.commit_cycle(0.05f); unseen += g2.last_sweep_diag().cells_unseen; }

        OccupancyGrid g3; build(g3);                          // ...and a genuine see-through THROUGH its voxels
        std::vector<Eigen::Vector3f> through;
        for (float z = -1.4f; z <= 0.8f; z += 0.01f)          // low enough that the beams cross the obstacle's
            for (int i = 0; i < 12; ++i)                     // own voxel at x = 1, not just the ones above it
                through.push_back({5.0f, -0.2f + 0.4f * (i / 11.0f), z});
        long cleared_at = -1;
        for (int k = 0; k < 300 and cleared_at < 0; ++k)
        { g3.integrate_sweep(eye, through); g3.commit_cycle(0.05f); if (not g3.occupied(lx, ly)) cleared_at = k; }

        std::printf("  flown-over: occ %d | OVER->%d (lo=%.2f) | NEVER-SEEN->%d (unseen=%ld) | THROUGH->%d @%ld\n",
                    occ0, g.occupied(lx, ly), g.logodds(lx, ly), g2.occupied(lx, ly), unseen,
                    g3.occupied(lx, ly), cleared_at);
        check(occ0, "the obstacle must be occupied before all three arms");
        check(g.occupied(lx, ly),  "beams passing OVER an obstacle must not release it - different voxels");
        check(g2.occupied(lx, ly), "a cell NO beam reached must be HELD - no information is not evidence of absence");
        check(unseen > 0, "never-reached cells must be COUNTED as unseen");
        check(cleared_at >= 0 and not g3.occupied(lx, ly),
              "...and a beam THROUGH its voxels must still clear it - the model is not merely refusing");
    }

    // ── (16) CLEARING IS RANGE-WEIGHTED — a distant see-through must erode as weakly as it was learned ──
    // Every hit and every miss carries w(r) = r0²/(r0²+r²), deliberately: a far see-through barely clears and a
    // distant obstacle PERSISTS until the robot closes on it. The term used to be tested on the time decay; with
    // the voxel rewrite the decay no longer touches occupancy at all, so it is tested where it now lives — on
    // the evidence itself. Two identical obstacles, identical clearing beams, differing ONLY in how far the
    // observer stands: the near one must give way and the far one must hold.
    {
        const auto run = [&](float obs_x, bool weighted)
        {
            OccGridParams Q = P; Q.hit_reliable_range_m = weighted ? P.hit_reliable_range_m : 0.0f;
            OccupancyGrid g; g.reset(-1, -1, 14, 6, Q);
            const Eigen::Vector3f near_eye(9.0f, 0.02f, 0.50f); // BUILD it from close in both arms, so the only
            const Eigen::Vector3f eye(obs_x, 0.02f, 0.50f);     // difference is where the CLEARING is observed from
            std::vector<Eigen::Vector3f> obst;                  // the obstacle always sits at x = 10
            for (int i = 0; i < 12; ++i) obst.push_back({10.0f, -0.1f + 0.2f * (i / 11.0f), 0.30f});
            for (int k = 0; k < 12; ++k) { g.integrate_sweep(near_eye, obst); g.commit_cycle(0.05f); }
            int cx, cy; g.world_to_cell(10.0f, 0.0f, cx, cy);
            const float lo0 = g.logodds(cx, cy);
            std::vector<Eigen::Vector3f> through;               // ...then it is gone: beams run past it
            for (float z = -1.4f; z <= 0.8f; z += 0.01f)
                for (int i = 0; i < 12; ++i) through.push_back({13.5f, -0.2f + 0.4f * (i / 11.0f), z});
            for (int k = 0; k < 25; ++k) { g.integrate_sweep(eye, through); g.commit_cycle(0.05f); }
            return std::pair{lo0, g.logodds(cx, cy)};
        };
        const auto [n0, n1] = run(9.0f,  true);    // observer 1 m from the obstacle  → w ≈ 0.86
        const auto [f0, f1] = run(2.0f,  true);    // observer 8 m from it            → w ≈ 0.09
        const auto [u0, u1] = run(2.0f,  false);   // control: same 8 m, weighting OFF
        std::printf("  clear-range: NEAR lo %.2f→%.2f | FAR lo %.2f→%.2f | FAR unweighted lo %.2f→%.2f\n",
                    n0, n1, f0, f1, u0, u1);
        check(n0 > 0.0f and f0 > 0.0f and u0 > 0.0f, "all three must start with occupancy evidence");
        check(n1 < n0 - 1.0f, "a NEAR see-through must measurably erode the obstacle");
        check(f1 > n1, "the FAR one must retain strictly more evidence - same beams, same cycles, only range differs");
        check(u1 < f1, "with the weighting OFF the far see-through bites harder - the weight is the variable");
    }

    // ── (17) A FLAT SURFACE MUST NOT ERASE ITSELF AT GRAZING INCIDENCE ──
    // The user's constraint, as a property: a table standing in the middle of the room cannot lose its residual
    // cells, because the table does not fly and no table_concept is running to explain it away. The only thing
    // that may remove it is evidence that the space is free.
    // The failure this pins was MEASURED on the real geometry (helios upright at 1.075 m, plate at 0.75 m, robot
    // circling at 1.6 m): the beam descends at ~18 deg, crosses four or five plate cells within 2 cm of the
    // surface, and terminates on that same plate. Every crossing was recorded as free space. Over 900 cycles the
    // footprint was destroyed 2799 times and held only 284 of its 384 cells; cells that had stood 402 cycles
    // were deleted in a handful. With the continuous-surface term: 36 releases, 382 of 384 cells held, and the
    // open-floor phantom count and wall coverage are bit-identical between the two arms.
    // Both halves are asserted, because a term that merely refuses to clear is not a fix but a leak.
    {
        const Eigen::Vector3f eye(0.0f, 0.0f, 1.075f);          // helios mount height
        // A 1 m plate at z=0.75. First it is seen whole, so every cell banks its evidence...
        std::vector<Eigen::Vector3f> whole;
        for (float x = 1.2f; x <= 2.2f; x += 0.02f) for (float y = -0.5f; y <= 0.5f; y += 0.02f)
            whole.push_back({x, y, 0.75f});
        // ...then only its FAR rim returns, which is what a moving robot actually delivers: the near cells are no
        // longer struck, they are only CROSSED, by beams descending onto the far rim of the same plate. Those
        // crossings are 1-3 cm above a surface the beam is about to hit.
        std::vector<Eigen::Vector3f> far_rim;
        for (float x = 2.10f; x <= 2.2f; x += 0.02f) for (float y = -0.5f; y <= 0.5f; y += 0.02f)
            far_rim.push_back({x, y, 0.75f});
        // ...and the control for (b): the plate is CARRIED AWAY, so the same bearings now return from the wall
        // beyond it, crossing the cells well BELOW the remembered surface.
        // A fan of endpoint heights, so the beams cross the plate's cells AT the plate's own height and not
        // 4 cm under it. Refutation requires looking where the material is — the same correction (3) needed.
        std::vector<Eigen::Vector3f> gone;
        for (float z = 0.1f; z <= 1.4f; z += 0.05f)
            for (float y = -0.5f; y <= 0.5f; y += 0.04f) gone.push_back({4.5f, y, z});

        // The observer ORBITS, as the real robot does. A fixed eye grazes the plate from one bearing only, and
        // the guard's effect is then invisible — the erosion that motivated this property comes from the plate
        // being skimmed from a new angle every cycle, each pass crossing a different set of its cells.
        const auto run = [&](bool guard_on, const std::vector<Eigen::Vector3f>& after, int n)
        {
            OccGridParams Q = P; Q.clear_stop_max_m = guard_on ? P.clear_stop_max_m : 0.0f;
            OccupancyGrid g; g.reset(-1, -1, 5, 5, Q);
            const auto orbit = [&](int k)
            {
                const float a2 = 0.35f * std::sin(0.11f * k);          // sweep the bearing across the plate
                // A SHALLOW graze (eye only 10 cm above the plate). From helios's 1.075 m onto a 0.75 m plate at
                // 1.6 m the beam descends 20 cm per metre and, at 3.125 cm voxels, crosses the plate's own voxel
                // for barely a cell — the resolution already handles that case, which is why the guard shows no
                // effect there. It is the shallow approach (a lower sensor, or a taller surface) where a beam
                // stays inside the surface's voxel for half a metre, and where the guard earns its place.
                return Eigen::Vector3f(1.7f - 1.6f * std::cos(a2), 1.6f * std::sin(a2), 0.85f);
            };
            for (int k = 0; k < 20; ++k) { g.integrate_sweep(orbit(k), whole); g.commit_cycle(0.1f); }
            int cx, cy; g.world_to_cell(1.6f, 0.0f, cx, cy);     // a cell in the MIDDLE of the plate
            const bool occ0 = g.occupied(cx, cy);
            long rel = 0, first_clear = -1;
            for (int k = 0; k < n; ++k)
            {
                g.integrate_sweep(orbit(20 + k), after); g.commit_cycle(0.1f);
                rel += g.last_sweep_diag().cells_released;
                if (first_clear < 0 and not g.occupied(cx, cy)) first_clear = k;
            }
            return std::tuple{occ0, g.occupied(cx, cy), rel, first_clear};
        };
        const auto [s_occ0, s_occ1, s_rel, s_when] = run(true,  far_rim, 300);   // standing, term ON
        const auto [r_occ0, r_occ1, r_rel, r_when] = run(true,  gone,    300);   // carried away
        // The guard must actually FIRE here, or the property passes for the wrong reason.
        OccupancyGrid gg; gg.reset(-1, -1, 5, 5, P);
        for (int k = 0; k < 20; ++k) { gg.integrate_sweep({0.15f, 0.0f, 0.85f}, whole); gg.commit_cycle(0.1f); }
        gg.integrate_sweep({0.15f, 0.0f, 0.85f}, far_rim); gg.commit_cycle(0.1f);
        const long stopped = gg.last_sweep_diag().clear_stopped;
        std::printf("  grazing-surface: STANDING occ %d->%d rel=%ld | REMOVED occ %d->%d cleared@%ld | "
                    "guard fired on %ld voxel crossings\n",
                    s_occ0, s_occ1, s_rel, r_occ0, r_occ1, r_when, stopped);
        check(s_occ0 and r_occ0, "the plate must be occupied before both periods");
        check(s_occ1, "a STANDING flat surface must NOT erase itself with its own grazing beams");
        check(s_rel == 0, "...and must release NO cell at all while it is still there");
        check(stopped > 0, "the grazing guard must actually FIRE, or this passes for the wrong reason");
        // NOTE: a with/without control could NOT be made to discriminate here. At 3.125 cm voxels a synthetic
        // fixed-plate geometry no longer grazes hard enough for the guard to matter — the RESOLUTION already
        // handles it. The evidence that the guard earns its place is the full probe, on the real mounts with a
        // circling robot: releases inside the table footprint 2530 (no guard) vs 835 (guard) at 6.25 cm voxels,
        // and 1337 vs 442 at 3.125 cm. Recorded here so nobody mistakes this property for that measurement.
    }
    // ── (18) THE BLIND CONE IS NOT FREE SPACE ──
    // helios stands at 1.075 m with a vertical field of about [-55, +15] deg, so at 0.3 m range it can only ever
    // return from 0.65-1.15 m: a chair seat at 0.45 m there is not unobserved, it is UNOBSERVABLE, and sweeping
    // past says nothing about it. The 2-D build needed a declared sensor envelope for this. In 3-D the voxels
    // the beams cannot reach are simply never cleared, and the read-out declines to call a cell free on the
    // strength of a collision band it never resolved (see collision_band_top_m). Same beams, same cycles: the
    // near cell must stay near UNKNOWN while the far cell, whose whole band the fan does resolve, reads free.
    {
        OccGridParams PB = P; PB.forget_half_life_s = 0.0f;      // isolate this from any ageing
        OccupancyGrid g; g.reset(-1, -1, 6, 6, PB);
        const Eigen::Vector3f eye(0.0f, 0.0f, 1.075f);
        // A real helios fan: 32 layers over [-55, +15] deg, out to a far wall.
        std::vector<Eigen::Vector3f> fan;
        for (int L = 0; L < 32; ++L)
        {
            const float el = (-55.0f + 70.0f * L / 31.0f) * 3.14159265f / 180.0f;
            for (int A = -40; A <= 40; ++A)
            {
                const float az = A * 0.006f;
                const float t = std::abs(std::tan(el)) > 1e-3f and el < 0.0f
                              ? std::min(5.0f, 1.075f / std::abs(std::tan(el))) : 5.0f;   // floor or far wall
                fan.push_back({t * std::cos(az), t * std::sin(az), std::max(0.0f, 1.075f + t * std::tan(el))});
            }
        }
        for (int k = 0; k < 80; ++k) { g.integrate_sweep(eye, fan); g.commit_cycle(0.1f); }
        int nx, ny; g.world_to_cell(0.30f, 0.0f, nx, ny);        // 0.3 m: only 0.65-1.15 m is reachable here
        int fx, fy; g.world_to_cell(2.50f, 0.0f, fx, fy);        // 2.5 m: the whole collision band is swept
        std::printf("  blind-cone: P(near 0.3 m)=%.3f P(far 2.5 m)=%.3f\n", g.prob(nx, ny), g.prob(fx, fy));
        check(g.prob(fx, fy) < 0.25f, "a cell the sensor CAN resolve must be cleared to free");
        check(g.prob(nx, ny) > g.prob(fx, fy) + 0.15f,
              "...and one inside the blind cone must stay measurably closer to UNKNOWN - silence is not absence");
    }

    // ── (19) A NON-FINITE RETURN MUST NOT FREEZE THE MAP ──
    // Found live 2026-08-22, and it was introduced by the clearing-probability work itself. A NaN return used to
    // be inert: every comparison against NaN is false, so it slipped through the z-gate and carried a finite
    // weight. Once the weight became a computed probability, NaN ran into p_block, into w, into lo_ — and a NaN
    // log-odds satisfies NEITHER `lo > occ_set` NOR `lo < occ_clear`, so the cell drops out of the latch
    // permanently. clear_p read NaN from cycle 7 and by cycle 1500 the whole map was latched=0 released=0 with
    // 1614 cells stuck. The visible symptom was a residual cloud that would not clear; the cause was arithmetic.
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        std::vector<Eigen::Vector3f> obst; face_returns(obst);
        const float nan_f = std::numeric_limits<float>::quiet_NaN();
        std::vector<Eigen::Vector3f> poisoned = obst;
        poisoned.push_back({nan_f, 0.0f, 0.5f});               // invalid depth, as the ZED emits it
        poisoned.push_back({2.0f, 0.0f, nan_f});
        poisoned.push_back({std::numeric_limits<float>::infinity(), 0.0f, 0.5f});
        for (int k = 0; k < 6; ++k) { g.integrate_sweep(sensor, poisoned); g.commit_cycle(0.1f); }
        int ix, iy; g.world_to_cell(2.0f, 0.0f, ix, iy);
        const bool occ_after_poison = g.occupied(ix, iy);
        const long bad = g.last_sweep_diag().bad_points;
        // ...and it must still be CLEARABLE afterwards: the real test is that the latch still works.
        std::vector<Eigen::Vector3f> beyond;                    // a FAN, so every voxel the obstacle claimed is
        for (float z = -0.4f; z <= 1.2f; z += 0.02f)            // actually crossed — refutation is per voxel now
            for (int i = 0; i < 12; ++i) beyond.push_back({4.5f, -0.2f + 0.4f * (i / 11.0f), z});
        long clear_at = -1;
        for (int k = 0; k < 200 and clear_at < 0; ++k)
        { g.integrate_sweep(sensor, beyond); g.commit_cycle(0.1f); if (not g.occupied(ix, iy)) clear_at = k; }
        std::printf("  nan-guard: bad_points=%ld/sweep occupied_after_poison=%d cleared@%ld lo=%.2f "
                    "repaired=%ld\n", bad, occ_after_poison, clear_at, g.logodds(ix, iy),
                    g.last_sweep_diag().cells_repaired);
        check(bad == 3, "every non-finite return must be dropped and COUNTED, not silently absorbed");
        check(occ_after_poison, "a real obstacle must still latch in a sweep that contained NaNs");
        check(std::isfinite(g.logodds(ix, iy)), "the log-odds must stay finite through a poisoned sweep");
        check(clear_at >= 0, "...and the cell must still be CLEARABLE - a NaN must not freeze it out of the latch");
    }
    // ── (20) A STRAY RETURN MUST NOT MAKE THE AIR ABOVE A TABLE INTO SOMETHING TO SEE THROUGH ──
    // This is the live failure of 2026-08-22, in an empty room with one 1.40x0.70 m table. zmn_/zmx_ are a
    // running MIN and MAX, so ONE stray return above the plate — ZED noise, someone walking past, a reflection —
    // stretched the believed material to [0.73, 1.41] permanently, and every beam crossing the empty air between
    // them counted as a full-weight see-through. The published residual fell to 26 of ~392 cells in components of
    // 2, the planner had nothing to avoid, and the robot drove into the table.
    // Evidence of material at 0.73 and a stray at 1.41 is not evidence of material at 0.92.
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        const Eigen::Vector3f eye(0.0f, 0.0f, 1.075f);
        std::vector<Eigen::Vector3f> plate;                     // the tabletop: returns ONLY at 0.73
        for (float x = 1.6f; x <= 2.4f; x += 0.02f) for (float y = -0.4f; y <= 0.4f; y += 0.02f)
            plate.push_back({x, y, 0.73f});
        for (int k = 0; k < 10; ++k) { g.integrate_sweep(eye, plate); g.commit_cycle(0.1f); }
        int cx, cy; g.world_to_cell(2.0f, 0.0f, cx, cy);
        std::vector<Eigen::Vector3f> stray = plate;             // ...and ONE stray return 68 cm above it
        stray.push_back({2.0f, 0.0f, 1.41f});
        for (int k = 0; k < 3; ++k) { g.integrate_sweep(eye, stray); g.commit_cycle(0.1f); }
        const bool occ0 = g.occupied(cx, cy);
        // Beams through the HOLLOW AIR between the plate and the stray, ending on a wall well beyond.
        std::vector<Eigen::Vector3f> hollow;
        for (float z = 0.90f; z <= 1.25f; z += 0.05f)
            for (int i = 0; i < 20; ++i) hollow.push_back({4.5f, -0.4f + 0.8f * (i / 19.0f), z});
        long rel = 0;
        for (int k = 0; k < 300; ++k)
        { g.integrate_sweep(eye, hollow); g.commit_cycle(0.1f); rel += g.last_sweep_diag().cells_released; }
        const bool held = g.occupied(cx, cy);
        // ...and the control that stops this being a leak: beams at the PLATE'S OWN height still clear it.
        // ...sweeping the WHOLE column, the stray's own height included: a cell is removed when every height it
        // claimed has been looked at and found empty, so a control that only looks at the plate leaves the stray
        // bin unrefuted and the cell correctly alive.
        // Beams that CROSS the plate's voxels on their way down to the floor beyond — which is what a robot
        // actually collects once the table has gone. (A beam that both crosses the plate's voxel AND terminates
        // at the plate's own height is the grazing case property (17) protects, by design: from a 2-D column you
        // cannot tell it from the surface stopping the beam. Removal comes from the other beams, and does.)
        std::vector<Eigen::Vector3f> at_plate;
        // ...spanning high enough to also cross the STRAY's voxel: while any voxel in the column still holds
        // material the cell stays occupied, which is the whole point of the projection.
        for (float z = 0.0f; z <= 2.0f; z += 0.01f)
            for (int i = 0; i < 20; ++i) at_plate.push_back({4.5f, -0.4f + 0.8f * (i / 19.0f), z});
        long clear_at = -1;
        for (int k = 0; k < 300 and clear_at < 0; ++k)
        { g.integrate_sweep(eye, at_plate); g.commit_cycle(0.1f);
          if (not g.occupied(cx, cy)) clear_at = k; }
        // ...and the flag is the variable: with the support off, the stretched hull erases the plate.
        OccGridParams PN = P; PN.bin_span_m = 0.0f;             // fall back to the zmn/zmx hull
        OccupancyGrid g2; g2.reset(-1, -1, 5, 5, PN);
        for (int k = 0; k < 10; ++k) { g2.integrate_sweep(eye, plate); g2.commit_cycle(0.1f); }
        for (int k = 0; k < 3; ++k)  { g2.integrate_sweep(eye, stray); g2.commit_cycle(0.1f); }
        for (int k = 0; k < 300; ++k){ g2.integrate_sweep(eye, hollow); g2.commit_cycle(0.1f); }
        std::printf("  height-support: plate+stray occ %d -> held=%d (rel=%ld) | cleared at its OWN height @%ld "
                    "| control (hull) occ->%d\n", occ0, held, rel, clear_at, g2.occupied(cx, cy));
        check(occ0, "the tabletop must be occupied before the hollow beams");
        check(held, "beams through the AIR above a table must not clear it - no return ever landed there");
        check(clear_at >= 0, "...but beams at the PLATE'S OWN height must still clear it - not a leak");
        check(not g2.occupied(cx, cy), "with the support OFF the stretched hull does erase it - the flag is it");
    }
    // ── (21) THE READ-OUT BAND MUST CARRY THE COLUMN'S REAL HEIGHT ──
    // The explainers score each cell against the band readout_zband() reports, so if that band is wrong every
    // published obstacle is judged as something it is not. Found live 2026-08-22 immediately after the voxel
    // rewrite: hit_ was no longer maintained, so readout_zband returned 0-0.05 m for EVERY occupied cell, the
    // FLOOR explainer claimed the whole map, and the agent published nothing — `residual components=0` beside
    // `occ=2547`, and an empty grid in the viewer. Nothing else caught it: occupancy, clearing and the field
    // were all correct, and the publish path just went quiet.
    // So: build an obstacle at 0.75 m, hand occupied_components a FLOOR explainer (one that claims anything
    // lying at floor level), and require the obstacle to survive it.
    {
        OccupancyGrid g; g.reset(-1, -1, 5, 5, P);
        const Eigen::Vector3f eye(0.02f, 0.02f, 1.075f);
        std::vector<Eigen::Vector3f> top;
        for (float x = 1.6f; x <= 2.4f; x += 0.02f)
            for (int i = 0; i < 12; ++i) top.push_back({x, -0.15f + 0.3f * (i / 11.0f), 0.75f});
        for (int k = 0; k < 10; ++k) { g.integrate_sweep(eye, top); g.commit_cycle(0.1f); }
        // A stand-in for the read-out floor explainer: claims a cell whose band sits at floor level.
        const auto floor_explainer = [](float, float, float zlo, float zhi)
        { return (zhi < 0.10f) ? 1.0f : 0.0f; };
        const auto bare  = g.occupied_components(2);
        const auto after = g.occupied_components(2, floor_explainer, 0.0f);
        long bare_cells = 0, after_cells = 0;
        for (const auto& c : bare)  bare_cells  += c.n_cells;
        for (const auto& c : after) after_cells += c.n_cells;
        std::printf("  readout-band: %zu comp / %ld cells unexplained → %zu comp / %ld cells past a FLOOR "
                    "explainer\n", bare.size(), bare_cells, after.size(), after_cells);
        check(bare_cells > 50, "the obstacle must be occupied before the explainer runs");
        check(not after.empty() and after_cells > bare_cells / 2,
              "an obstacle at 0.75 m must SURVIVE a floor explainer - its read-out band must carry its height");
    }
    // ── (22) A RING LIDAR MUST NOT MAKE MID-ROOM PHANTOMS IMMORTAL ──
    // helios is a ring lidar at 1.075 m, so a beam at ring height that crosses a phantom in open floor
    // terminates on a WALL at that same height. Under a height-only grazing guard that is |cz - ez| <= 1, the
    // guard fires, and the phantom can never be cleared. Measured live 2026-08-23: 1402 residual cells above
    // 0.9 m, 1052 of them more than 1.5 m from any wall, accumulated in two minutes then frozen at
    // latched = released = 0 — a map the planner could not cross, which is how this was noticed at all.
    // The guard's premise is that the beam SKIMMED a horizontal surface and struck it further on. A beam cannot
    // skim a wall. So both halves are asserted together: the phantom must go, the tabletop must stay.
    {
        const Eigen::Vector3f ring(0.02f, 0.02f, 1.00f);
        // (a) a phantom in open floor at ring height, then level beams passing through it to a far WALL.
        OccupancyGrid g; g.reset(-1, -1, 6, 6, P);
        std::vector<Eigen::Vector3f> phantom;
        for (int i = 0; i < 12; ++i) phantom.push_back({2.0f, -0.1f + 0.2f * (i / 11.0f), 1.00f});
        for (int k = 0; k < 8; ++k) { g.integrate_sweep(ring, phantom); g.commit_cycle(0.1f); }
        int px, py; g.world_to_cell(2.0f, 0.0f, px, py);
        const bool ph0 = g.occupied(px, py);
        std::vector<Eigen::Vector3f> wall;                   // a WALL: material floor to ceiling at x = 5
        for (float z = 0.10f; z <= 1.75f; z += 0.02f)
            for (int i = 0; i < 12; ++i) wall.push_back({5.0f, -0.2f + 0.4f * (i / 11.0f), z});
        long ph_clear = -1;
        for (int k = 0; k < 300 and ph_clear < 0; ++k)
        { g.integrate_sweep(ring, wall); g.commit_cycle(0.1f); if (not g.occupied(px, py)) ph_clear = k; }

        // (b) the counterpart: a TABLETOP, skimmed by beams that end on the tabletop itself. It must survive.
        OccupancyGrid g2; g2.reset(-1, -1, 6, 6, P);
        const Eigen::Vector3f eye(0.02f, 0.02f, 1.075f);
        std::vector<Eigen::Vector3f> plate;
        for (float x = 1.2f; x <= 2.2f; x += 0.02f)
            for (int i = 0; i < 12; ++i) plate.push_back({x, -0.2f + 0.4f * (i / 11.0f), 0.75f});
        for (int k = 0; k < 20; ++k) { g2.integrate_sweep(eye, plate); g2.commit_cycle(0.1f); }
        int tx, ty; g2.world_to_cell(1.5f, 0.0f, tx, ty);
        const bool tb0 = g2.occupied(tx, ty);
        std::vector<Eigen::Vector3f> far_rim;                 // only the far rim returns: the near cells are
        for (float x = 2.10f; x <= 2.2f; x += 0.02f)          // crossed, not struck
            for (int i = 0; i < 12; ++i) far_rim.push_back({x, -0.2f + 0.4f * (i / 11.0f), 0.75f});
        for (int k = 0; k < 300; ++k) { g2.integrate_sweep(eye, far_rim); g2.commit_cycle(0.1f); }

        std::printf("  ring-phantom: mid-room phantom occ %d -> cleared@%ld (endpoint = WALL) | tabletop occ "
                    "%d -> %d (endpoint = PLATE)\n", ph0, ph_clear, tb0, g2.occupied(tx, ty));
        check(ph0 and tb0, "both must be occupied before their clearing periods");
        check(ph_clear >= 0, "a mid-room phantom must be cleared by level beams that end on a WALL - a beam "
                             "cannot have skimmed a wall, so the grazing guard must not protect it");
        check(g2.occupied(tx, ty), "...while a TABLETOP, whose beams end on the plate itself, must still survive");
    }
    // ── (23) NOTHING INSIDE THE LIDAR CLEARANCE RADIUS MAY BE CLEARED ──
    // Every ray starts inside the disc the lidar driver self-filters and marks its way out, weighted by w(r),
    // which is LARGEST at r = 0. So the ring of cells the robot is standing in was being cleared at the highest
    // weight in the grid, in the one place no sensor can see — which is what deleted a table the moment the
    // robot closed on it. The rule is about the ROBOT, not one sensor's minimum range: it must hold whichever
    // sweep the ray came from, and be measured from the robot to the cell rather than along the ray.
    // Two cells, same obstacle, same beams, differing only in whether the robot is standing on top of them.
    {
        const auto run = [&](float robot_x, float clearance)
        {
            OccGridParams Q = P; Q.lidar_clearance_m = clearance; Q.forget_half_life_s = 0.0f;
            OccupancyGrid g; g.reset(-1, -1, 6, 6, Q);
            std::vector<Eigen::Vector3f> obst;                   // an obstacle at x = 2
            for (int i = 0; i < 12; ++i) obst.push_back({2.0f, -0.1f + 0.2f * (i / 11.0f), 0.60f});
            g.set_self_body(0.0f, 0.0f, 0.55f);
            for (int k = 0; k < 12; ++k)
            { g.integrate_sweep({0.0f, 0.0f, 0.60f}, obst); g.commit_cycle(0.1f); }
            int cx, cy; g.world_to_cell(2.0f, 0.0f, cx, cy);
            const bool occ0 = g.occupied(cx, cy);
            // ...now the robot drives up to it and the beams run PAST, from a sensor beside the obstacle.
            std::vector<Eigen::Vector3f> past;
            for (float z = -0.6f; z <= 1.2f; z += 0.01f)
                for (int i = 0; i < 12; ++i) past.push_back({5.0f, -0.2f + 0.4f * (i / 11.0f), z});
            g.set_self_body(robot_x, 0.0f, 0.55f);
            long cleared_at = -1;
            for (int k = 0; k < 200 and cleared_at < 0; ++k)
            { g.integrate_sweep({robot_x, 0.0f, 0.60f}, past); g.commit_cycle(0.1f);
              if (not g.occupied(cx, cy)) cleared_at = k; }
            return std::pair{occ0, cleared_at};
        };
        const auto [n0, n_when] = run(1.75f, 0.55f);   // robot 0.25 m from the cell: INSIDE the clearance disc
        const auto [f0, f_when] = run(0.50f, 0.55f);   // robot 1.50 m away: outside it, ordinary clearing
        const auto [c0, c_when] = run(1.75f, 0.0f);    // control: same close pass, rule OFF
        std::printf("  clearance-radius: robot 0.25 m away -> cleared@%ld (must be NEVER) | 1.50 m away -> "
                    "cleared@%ld | control (rule off, 0.25 m) -> cleared@%ld\n", n_when, f_when, c_when);
        check(n0 and f0 and c0, "the obstacle must be occupied before all three passes");
        check(n_when < 0, "a cell INSIDE the lidar clearance radius must never be cleared - nothing can see it");
        check(f_when >= 0, "...while the same cell observed from outside the radius must still clear");
        check(c_when >= 0, "with the rule OFF the close pass DOES delete it - the radius is the variable");
    }
    std::printf("OccupancyGrid::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
