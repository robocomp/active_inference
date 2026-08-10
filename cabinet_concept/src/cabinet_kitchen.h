/*
 * cabinet_kitchen.h — Stage 2 of the kitchen-of-runs model (Fable design): the KITCHEN MANAGER. Owns the
 * finite (wall_id, tier) cells; each cell holds a WallRunBelief that ACTIVATES on routed evidence and
 * RETIRES on absence, and is fit by soft-routed points. This REPLACES the free-box tracker/fitter for
 * wall-anchored runs: route → per-cell weighted update → existence log-odds → derived room-frame box.
 *
 * Identity IS the cell, so there is no birth/associate/merge, no duplicates, no crossings, no 10 cm / ceiling
 * boxes — all made unrepresentable by the cell + WallRunBelief chart. Islands (free-standing) are NOT handled
 * here (Stage 4, generic tracker). Pure logic, no DSR (the worker publishes from active_boxes()).
 *
 * Validated by kitchen_manager_self_test().
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <ranges>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "cabinet_kitchen_cells.h"     // KitchenWall, KitchenTier, KitchenRouting, signature
#include "cabinet_wall_run_belief.h"   // WallRunBelief, WallChart, WallTierPrior
#include "cabinet_belief.h"            // CabinetBeliefParams, CabinetFrame

namespace rc {

struct KitchenManagerParams
{
    float sigma_lat   = 0.15f;   // routing lateral/along softness (m)
    float sigma_z     = 0.20f;   // routing z softness (m)
    float clutter     = 0.05f;   // routing clutter weight
    float min_route_r = 0.10f;   // a point contributes to a cell's fit only above this responsibility
    // Existence is split into PRESENCE (mask mass present/absent) and COVERAGE QUALITY (is the mass an extended
    // run or a point cluster). Coverage = along-wall span / coverage_ref_m, low-passed into cov_ema so a run's
    // corner leakage — plenty of presence, a few lucky wide frames — cannot birth a sliver, and a collapsed run
    // is retired. All scales are physical coverage FRACTIONS, not magic gates.
    float coverage_ref_m   = 0.30f;   // along-wall span that counts as full coverage (cov=1); ~half a carcass
    float coverage_ema_rate = 0.10f;  // low-pass rate of the coverage EMA (slow ⇒ robust to per-frame mask noise)
    float coverage_birth   = 0.60f;   // birth needs the coverage EMA this high (a genuinely extended run)
    float coverage_kill    = 0.35f;   // retire an active cell whose coverage EMA falls this low (sliver/collapsed)
    float presence_gain    = 0.70f;   // presence log-odds added per frame with mask mass
    float absence_decay    = 0.50f;   // presence log-odds subtracted per frame with ~no mass
    float activate_logodds = 2.0f;    // birth needs presence above this (P≈0.88)
    float retire_logodds   = -2.0f;   // retire (on absence) below this
    float logodds_cap      = 6.0f;
    // Corner-fill: two active perpendicular runs sharing a room vertex extend to MEET there (no hole in the L).
    float corner_join_tol_m = 0.10f;  // walls "share a vertex" if endpoints are within this
    float corner_capture_m  = 1.30f;  // ...and both runs stop within this of it (the unobserved corner void — a
                                      // corner cabinet + the occluded near-corner run portions — reaches ~1 m)
    // Scene-object engulfment retirement: a run mostly INSIDE another agent's object (e.g. a false run on the
    // fridge that the LiDAR carve cannot reach) is retired. The exclusion FACTOR retracts a crossing end; this
    // removes a run whose CENTRE is engulfed. One-directional (cabinet yields; the other agent is unaware of us).
    bool  object_exclusion_enabled = false;   // gate the engulfment-retirement pass
    float engulf_frac = 0.60f;                // retire when along-wall AND depth AND z overlap all exceed this
    // LiDAR EVIDENCE OF ABSENCE — the retirement channel for a BORN run. The mask channel above can only ever
    // ADD presence to an active cell (a look-away is not evidence of absence), and the free-space carve only
    // RESHAPES; without this, a cell that ever births is immortal. Here a beam that traverses the run and lands
    // on the WALL BEHIND it is a direct refutation, integrated into the SAME existence log-odds so retirement
    // stays the one decision boundary (retire_logodds), debounced over retire_frames evidence cycles.
    // Rates are the physical sensor rates shared with the classic path (rc::exist::SensorModel).
    bool  lidar_existence_enabled = false;    // gate the whole channel
    float sensor_sigma_m      = 0.03f;        // LiDAR range σ (m)
    float detection_prob      = 0.85f;        // P(return | occupied & observable)
    float clutter_prob        = 0.05f;        // P(return | empty)
    float absence_range_ref_m = 2.5f;         // range below which absence is trusted at full weight
    float absence_range_power = 2.0f;         // decay exponent (2 ≈ beam density ∝ 1/range²); 0 = no decay
    int   retire_frames       = 15;           // consecutive EVIDENCE cycles below retire_logodds before retiring
};

// Per-cell, per-cycle diagnostics (the kitchen_cells.csv row + the dashboard). Purely observational.
struct KitchenCellDiag
{
    int   n_route = 0;                         // mask points routed to this cell this cycle
    float ex_occ = 0.0f, ex_free = 0.0f;       // LiDAR occupancy / free-space evidence mass
    float ex_dL  = 0.0f;                       // the log-odds delta it integrated (0 ⇒ HOLD)
    int   ex_n   = 0;                          // beams that reached the box (0 ⇒ not probed ⇒ HOLD)
    int   retire_streak = 0;                   // consecutive evidence cycles the removal decision has held
};

struct KitchenCellRun
{
    KitchenCell                     geom;
    WallChart                       chart;
    WallTierPrior                   tier_prior;
    std::unique_ptr<WallRunBelief>  belief;   // null until activated
    float                           fe        = 0.0f;   // free energy of the last fit (dashboard gauge)
    float                           existence = 0.0f;   // existence log-odds (mask presence + LiDAR occupancy/absence)
    float                           cov_ema   = 0.0f;   // slow EMA of along-wall coverage (run vs point cluster)
    std::uint64_t                   node_id   = 0;       // DSR node (worker-managed)
    KitchenCellDiag                 diag;                // per-cycle observability (CSV/dashboard)
    bool active() const { return belief != nullptr; }
};

struct KitchenBox   // an active cell's derived room-frame box (for DSR publishing)
{
    std::string id;
    int   wall_seg_id = -1;
    int   tier = 0;
    float cx, cy, yaw, L, d, z0, z1;
    float existence;
    // ── Room-frame pose covariance, mapped from the belief's wall-chart Σ (fill_box_covariance) ──
    // Published on the room→run RT edge so a CONSUMER — the controller, and above all a level-2
    // metaconcept whose measurement noise IS each peer's published Σ — can weight a well-observed
    // run above a barely-glimpsed one without any confidence gate. Before this, the kitchen path
    // wrote pose with NO covariance while the classic path wrote one, so every kitchen run looked
    // equally (un)certain.
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_xy = 0.0f;   // footprint-centre covariance (m²)
    float var_z   = 0.0f;                                // variance of z0 = the RT translation z (m²)
    float var_yaw = 0.0f;                                // chart-axis direction variance (rad²)
};

class KitchenManager
{
public:
    // Build the cell table from the room walls. tier_priors: [base, upper]. H_room = ceiling.
    void build(const std::vector<KitchenWall>& walls, const KitchenTier tiers[2],
               const WallTierPrior tier_priors[2], const CabinetBeliefParams& bp, float H_room)
    {
        cells_.clear();
        bp_ = bp;
        base_tier_prior_ = tier_priors[0];   // island uses the base tier prior
        h_room_ = H_room;
        walls_ = walls;                      // kept for island axis-snapping
        for (const auto& wl : walls)
            for (int t = 0; t < 2; ++t)
            {
                KitchenCellRun c;
                c.geom.id = KitchenRouting::signature(wl, t);
                c.geom.wall_seg_id = wl.seg_id; c.geom.tier = t;
                c.geom.a = wl.a; c.geom.u = wl.u; c.geom.n = wl.n; c.geom.W = wl.W;
                c.geom.z_lo = tiers[t].z_lo; c.geom.z_hi = tiers[t].z_hi; c.geom.depth = tiers[t].depth;
                c.chart.A = wl.a; c.chart.u = wl.u; c.chart.n = wl.n; c.chart.W = wl.W; c.chart.H_room = H_room;
                c.tier_prior = tier_priors[t];
                cells_.push_back(std::move(c));
            }
    }
    bool built() const { return not cells_.empty(); }

    // One cycle: soft-route the points, update existence, activate/retire cells, and fit each active cell.
    // `frame_template` carries the shared per-frame terms (chain cov, ego-motion common-mode, lidar_freespace);
    // its points are IGNORED here (we inject per-cell routed points).
    void update(const std::vector<Eigen::Vector3f>& pts, const KitchenManagerParams& mp, const CabinetFrame& frame_template)
    {
        if (cells_.empty()) return;
        const std::size_t K = cells_.size();
        std::vector<double> cell_mass(K, 0.0);
        // Per-cell fit buffers: points ASSIGNED to the cell (hard argmax), with R weighted by responsibility.
        std::vector<std::vector<Eigen::Vector3f>> cpts(K);
        std::vector<std::vector<float>>           cR(K);
        const float il2 = 1.0f / std::max(1e-6f, mp.sigma_lat * mp.sigma_lat);
        const float iz2 = 1.0f / std::max(1e-6f, mp.sigma_z   * mp.sigma_z);
        const float baseR = bp_.sigma_base_m * bp_.sigma_base_m;
        std::vector<double> w(K);
        for (const auto& p : pts)
        {
            double sum = mp.clutter, wbest = mp.clutter;
            int kbest = -1;
            for (std::size_t k = 0; k < K; ++k)
            {
                const auto& c = cells_[k].geom;
                const float t   = (Eigen::Vector2f(p.x(), p.y()) - c.a).dot(c.u);
                const float lat = (Eigen::Vector2f(p.x(), p.y()) - c.a).dot(c.n);
                const float da  = std::max({0.0f, -t, t - c.W});
                const float dl  = std::max({0.0f, -lat, lat - c.depth});
                const float dz  = std::max({0.0f, c.z_lo - p.z(), p.z() - c.z_hi});
                w[k] = std::exp(-0.5 * ((da * da + dl * dl) * il2 + (dz * dz) * iz2));
                sum += w[k];
                if (w[k] > wbest) { wbest = w[k]; kbest = static_cast<int>(k); }
            }
            // HARD assignment to the single best cell. At an L/U corner the region touches two walls, but the
            // physical corner cabinet belongs to ONE run; soft-sharing let BOTH cells accrue ~0.2 m of the same
            // corner and double-count it as two runs. Argmax gives the corner to whichever wall owns it and
            // starves the neighbour, so exactly one run extends to the corner. R still carries the soft
            // responsibility so a marginal winner is downweighted in the fit.
            if (kbest < 0) continue;                                  // clutter (nearer nothing than the floor)
            const double r = w[kbest] / std::max(1e-12, sum);
            if (r <= mp.min_route_r) continue;
            cell_mass[kbest] += r;
            cpts[kbest].push_back(p);
            cR[kbest].push_back(baseR / static_cast<float>(std::max(0.05, r)));
        }

        for (std::size_t k = 0; k < K; ++k)
        {
            auto& c = cells_[k];
            c.diag.n_route = static_cast<int>(cpts[k].size());
            // Along-wall SPAN of this cell's routed points = coverage evidence. A run is an EXTENDED object; a
            // near-zero span means the mass is a point cluster (the tail of the adjacent wall's run leaking
            // across a corner), not a run — so it must NOT activate a degenerate sliver.
            float tmn = 1e9f, tmx = -1e9f;
            for (const auto& p : cpts[k])
            { const float t = (Eigen::Vector2f(p.x(), p.y()) - c.geom.a).dot(c.geom.u);
              tmn = std::min(tmn, t); tmx = std::max(tmx, t); }
            const float span = (cpts[k].size() >= 2) ? (tmx - tmn) : 0.0f;
            const float cov  = std::clamp(span / mp.coverage_ref_m, 0.0f, 1.0f);

            // TWO decoupled signals. (1) PRESENCE (existence log-odds): is there mask mass on this cell at all —
            // a simple present/absent debounce. (2) COVERAGE QUALITY (cov_ema): is that mass an EXTENDED run or a
            // point cluster — a slow EMA of the along-wall span. Corner leakage from an adjacent run has plenty of
            // presence but its coverage EMA stays low (a few lucky wide frames can spike existence but cannot move
            // a slow EMA), so it never births — and if one ever slips through, the kill-on-low-coverage retires it.
            const float m = static_cast<float>(cell_mass[k]);
            if (m > 0.5f)
            {
                c.existence = std::min(mp.logodds_cap, c.existence + mp.presence_gain);
                c.cov_ema  += mp.coverage_ema_rate * (cov - c.cov_ema);
            }
            else if (not c.active())      // fade only CANDIDATES; a BORN wall run is a static structure and
                c.existence = std::max(-mp.logodds_cap, c.existence - mp.absence_decay);   // PERSISTS through
            // look-away — absence of evidence (no mask in view) is NOT evidence of absence. A born run retires
            // only on OBSERVED collapse (cov_ema below), or later a free-space carve seeing through it.

            // Birth: present (existence) AND a persistently well-covered run (cov_ema) AND a real current seed (cov).
            if (not c.active() and c.existence >= mp.activate_logodds
                and c.cov_ema >= mp.coverage_birth and cov >= 0.5f and cpts[k].size() >= 8)
            {
                // Seed the interval from the routed span (already computed above); depth/z from the tier prior.
                WallRunState s0; s0.t0 = std::clamp(tmn, 0.0f, c.geom.W); s0.t1 = std::clamp(tmx, 0.0f, c.geom.W);
                s0.d = c.tier_prior.d_mean; s0.z0 = c.tier_prior.z0_mean; s0.z1 = c.tier_prior.z1_mean;
                // Suppress birth INTO another object (else engulfment retirement below would just re-birth it next
                // cycle — the run stays well-covered, so cov_ema never falls). No-op unless the feature is on.
                if (not (mp.object_exclusion_enabled and run_engulfed(s0, c.chart, frame_template.scene_objects, mp)))
                    c.belief = std::make_unique<WallRunBelief>(s0, bp_, c.chart, c.tier_prior);
            }
            // Retire ONLY on sustained ABSENCE (existence). NOT on low coverage: a partially-OCCLUDED run is seen
            // with reduced span (cov_ema drifts down), but that is absence of evidence, NOT evidence it is small —
            // killing on it removes occluded cabinets. What DOES refute an active run is the LiDAR absence channel
            // below (a beam reaching the wall BEHIND it), which drives this same log-odds negative — that is the
            // ONLY thing that can make this test fire for an active cell (mask absence never fades a born run).
            if (c.active() and c.existence <= mp.retire_logodds and c.diag.retire_streak >= mp.retire_frames)
            {
                std::printf("[kitchen] RETIRE cell %-12s — refuted: L=%.2f after %d agreeing evidence cycles "
                            "(last: occ=%.1f free=%.1f n=%d)\n", c.geom.id.c_str(), c.existence,
                            c.diag.retire_streak, c.diag.ex_occ, c.diag.ex_free, c.diag.ex_n);
                c.belief.reset(); c.existence = 0.0f; c.cov_ema = 0.0f; c.diag.retire_streak = 0;
            }
            // Clear the absence readouts AFTER the retire decision (which reports the evidence that caused it);
            // the pass at the end of this cycle refills them for every cell the sweep actually probes.
            c.diag.ex_occ = c.diag.ex_free = c.diag.ex_dL = 0.0f; c.diag.ex_n = 0;
        }

        // Engulfment retirement: a born run whose CENTRE is inside another agent's object (a false run on the
        // fridge the LiDAR carve cannot reach — no through-beams above it) is spurious → retire. The exclusion
        // FACTOR only retracts a crossing END; this removes a fully-engulfed run. One-directional (cabinet yields).
        if (mp.object_exclusion_enabled and not frame_template.scene_objects.empty())
            for (std::size_t k = 0; k < K; ++k)
                if (cells_[k].active()
                    and run_engulfed(cells_[k].belief->state(), cells_[k].belief->chart(), frame_template.scene_objects, mp))
                    cells_[k].belief.reset();

        // Corner-fill pass: an L of two perpendicular runs must MEET at their shared room corner, not leave a
        // hole (hard-argmax gives the corner points to one run, starving the other). Now the active set is known,
        // flag each run's corner-side end to extend to the shared vertex when the perpendicular neighbour is also
        // active and reaching that same corner — then the belief's corner-fill factor pushes the end to it.
        route_corner_fill(mp);

        // Second pass: update each active cell's belief with its assigned points (+ the corner-fill flags set above).
        for (std::size_t k = 0; k < K; ++k)
        {
            auto& c = cells_[k];
            if (not c.active() or cpts[k].empty()) continue;
            CabinetFrame f = frame_template;            // carries chain/ego/lidar_freespace
            f.points = std::move(cpts[k]); f.R = std::move(cR[k]);
            c.fe = c.belief->update(f);   // the returned free energy was being discarded
        }

        // Third pass — LiDAR EVIDENCE OF ABSENCE on the now-current boxes. Integrates into the SAME existence
        // log-odds the mask presence feeds, so there is exactly ONE decision boundary (retire_logodds, checked at
        // the top of the next cycle once the streak holds). Without this pass an active cell is immortal: mask
        // absence deliberately does not fade a born run, and the free-space carve only reshapes it.
        accumulate_absence(mp, frame_template);
    }

    // A run is "engulfed" when its box overlaps some scene object by ≥ engulf_frac along ALL THREE run axes
    // (along-wall, depth, z). Uses the exact OBB→wall shadow (same as the exclusion factor). Fraction is taken
    // relative to the RUN's own extents, so a small false run inside a big fridge reads ~1 on every axis.
    static bool run_engulfed(const WallRunState& s, const WallChart& ch,
                             const std::vector<SceneObjectBox>& objs, const KitchenManagerParams& mp)
    {
        const float run_span = std::max(1e-3f, s.t1 - s.t0);
        const float run_dep  = std::max(1e-3f, s.d);
        const float run_z    = std::max(1e-3f, s.z1 - s.z0);
        for (const auto& o : objs)
        {
            const Eigen::Vector2f ex(std::cos(o.yaw), std::sin(o.yaw)), ey(-std::sin(o.yaw), std::cos(o.yaw));
            const Eigen::Vector2f c(o.cx, o.cy);
            float o_t0 = 1e30f, o_t1 = -1e30f, o_s0 = 1e30f, o_s1 = -1e30f;
            for (float sx : {-0.5f, 0.5f}) for (float sy : {-0.5f, 0.5f})
            {
                const Eigen::Vector2f q = c + sx * o.w * ex + sy * o.d * ey - ch.A;
                const float t = q.dot(ch.u), sl = q.dot(ch.n);
                o_t0 = std::min(o_t0, t); o_t1 = std::max(o_t1, t);
                o_s0 = std::min(o_s0, sl); o_s1 = std::max(o_s1, sl);
            }
            const float f_t = std::max(0.0f, std::min(s.t1, o_t1) - std::max(s.t0, o_t0)) / run_span;
            const float f_s = std::max(0.0f, std::min(s.d,  o_s1) - std::max(0.0f, o_s0)) / run_dep;
            const float f_z = std::max(0.0f, std::min(s.z1, o.z1) - std::max(s.z0, o.z0)) / run_z;
            if (f_t >= mp.engulf_frac and f_s >= mp.engulf_frac and f_z >= mp.engulf_frac) return true;
        }
        return false;
    }

    // Set per-cell corner-fill flags. Two active same-tier cells on DIFFERENT walls that share a polygon vertex V
    // (walls meet there) form an L; if BOTH runs already reach near V, extend each run's corner-side end to V.
    void route_corner_fill(const KitchenManagerParams& mp)
    {
        const std::size_t K = cells_.size();
        std::vector<char> fl(K, 0), fh(K, 0);   // fill-low / fill-high, accumulated (a run can corner on both ends)
        const auto endpt = [](const KitchenCell& g, bool high)
        { return high ? Eigen::Vector2f(g.a + g.W * g.u) : g.a; };
        for (std::size_t i = 0; i < K; ++i)
        {
            if (not cells_[i].active()) continue;
            for (std::size_t j = i + 1; j < K; ++j)
            {
                if (not cells_[j].active()) continue;
                const auto& gi = cells_[i].geom; const auto& gj = cells_[j].geom;
                if (gi.tier != gj.tier or gi.wall_seg_id == gj.wall_seg_id) continue;
                // Try all four endpoint pairings for a shared vertex V (walls meet at a polygon corner).
                for (int hi = 0; hi < 2; ++hi) for (int hj = 0; hj < 2; ++hj)
                {
                    if ((endpt(gi, hi) - endpt(gj, hj)).norm() > mp.corner_join_tol_m) continue;  // no shared vertex
                    const auto& si = cells_[i].belief->state(); const auto& sj = cells_[j].belief->state();
                    const float gap_i = hi ? (gi.W - si.t1) : si.t0;    // how far each run stops short of the vertex
                    const float gap_j = hj ? (gj.W - sj.t1) : sj.t0;
                    if (gap_i > mp.corner_capture_m or gap_j > mp.corner_capture_m) continue;      // not an L here
                    (hi ? fh[i] : fl[i]) = 1;
                    (hj ? fh[j] : fl[j]) = 1;
                }
            }
        }
        for (std::size_t k = 0; k < K; ++k)
            if (cells_[k].active()) cells_[k].belief->set_corner_fill(fl[k] != 0, fh[k] != 0);

        if (corner_fill_log_)   // opt-in diagnostic (agent only): per active cell, wall/interval/fill flags
        {
            for (std::size_t k = 0; k < K; ++k)
                if (cells_[k].active())
                {
                    const auto& g = cells_[k].geom; const auto& s = cells_[k].belief->state();
                    const Eigen::Vector2f bb = g.a + g.W * g.u;
                    std::printf("[kitchen] cell %-12s wall=%d a=(%.2f,%.2f) b=(%.2f,%.2f) t=[%.2f,%.2f] fill=t0%d/t1%d\n",
                                g.id.c_str(), g.wall_seg_id, g.a.x(), g.a.y(), bb.x(), bb.y(),
                                s.t0, s.t1, fl[k] ? 1 : 0, fh[k] ? 1 : 0);
                }
        }
    }
    void set_corner_fill_log(bool on) { corner_fill_log_ = on; }

    // The free-standing island (peninsula / 4th cabinet): a single run, NOT wall-anchored. Its chart is derived
    // from its own points (horizontal PCA, snapped to the nearest wall axis for room-grid alignment) and frozen at
    // birth. Same coverage-gated existence + persistence as the wall cells. `pts` = only the kitchen-island masks.
    void update_island(const std::vector<Eigen::Vector3f>& pts, const KitchenManagerParams& mp,
                       const CabinetFrame& frame_template)
    {
        // EVIDENCE OF ABSENCE FIRST — before any early return on "no island mask this frame". A phantom island
        // is exactly the case that gets no masks, so deferring this behind the mask check would make it immortal.
        island_diag_.n_route = static_cast<int>(pts.size());
        island_diag_.ex_occ = island_diag_.ex_free = island_diag_.ex_dL = 0.0f; island_diag_.ex_n = 0;
        if (island_belief_ and mp.lidar_existence_enabled and not frame_template.lidar_freespace.endpoints.empty())
        {
            integrate_absence(*island_belief_, island_exist_, island_diag_, mp, frame_template);
            if (island_exist_ <= mp.retire_logodds and island_diag_.retire_streak >= mp.retire_frames)
            { island_belief_.reset(); island_exist_ = 0.0f; island_cov_ema_ = 0.0f; island_diag_ = {};
              island_yaw_var_ = kWallChartYawVar * 100.0f; return; }
        }
        if (pts.size() < 2)                        // no island evidence this frame
        {
            if (not island_belief_)                // fade a CANDIDATE; a born island PERSISTS through look-away
                island_exist_ = std::max(-mp.logodds_cap, island_exist_ - mp.absence_decay);
            return;
        }
        WallChart ch; bool anchored = island_anchored_;
        if (island_belief_) ch = island_chart_;    // frozen chart once born
        else if (not derive_island_chart(pts, ch, anchored)) return;

        float tmn = 1e9f, tmx = -1e9f, smn = 1e9f, smx = -1e9f, t, s;
        for (const auto& p : pts) { ch.to_wall(p, t, s);
            tmn = std::min(tmn, t); tmx = std::max(tmx, t); smn = std::min(smn, s); smx = std::max(smx, s); }
        const float cov = std::clamp((tmx - tmn) / mp.coverage_ref_m, 0.0f, 1.0f);
        island_exist_   = std::min(mp.logodds_cap, island_exist_ + mp.presence_gain);
        island_cov_ema_ += mp.coverage_ema_rate * (cov - island_cov_ema_);

        if (not island_belief_ and island_exist_ >= mp.activate_logodds
            and island_cov_ema_ >= mp.coverage_birth and cov >= 0.5f and pts.size() >= 8)
        {
            WallRunState s0;
            s0.t0 = anchored ? 0.0f : std::clamp(tmn, 0.0f, ch.W);   // peninsula near end sits ON the wall
            s0.t1 = std::clamp(tmx, 0.0f, ch.W);
            s0.d  = std::clamp(smx - smn, 0.35f, 1.2f);              // seed depth = observed cabinet width
            s0.z0 = base_tier_prior_.z0_mean; s0.z1 = base_tier_prior_.z1_mean;
            if (not (mp.object_exclusion_enabled and run_engulfed(s0, ch, frame_template.scene_objects, mp)))
            { island_chart_ = ch; island_anchored_ = anchored;
              island_belief_ = std::make_unique<WallRunBelief>(s0, bp_, island_chart_, base_tier_prior_); }
        }
        // (No coverage-kill: a partially-occluded island is absence of evidence, not evidence it shrank — the
        //  free-space carve retracts a genuinely-empty end. Killing on low coverage removed occluded units.)
        if (island_belief_ and mp.object_exclusion_enabled and not frame_template.scene_objects.empty()
            and run_engulfed(island_belief_->state(), island_belief_->chart(), frame_template.scene_objects, mp))
        { island_belief_.reset(); island_exist_ = 0.0f; island_cov_ema_ = 0.0f;
          island_yaw_var_ = kWallChartYawVar * 100.0f; return; }
        if (island_belief_)
        {
            island_belief_->set_corner_fill(island_anchored_, false);   // anchor the near end to the wall (t0→0)
            CabinetFrame f = frame_template;
            f.points = pts; f.R.assign(pts.size(), bp_.sigma_base_m * bp_.sigma_base_m);
            island_belief_->update(f);
            // Re-measure how well THIS cycle's points pin the frozen chart's axis. Held across look-away
            // (it is a property of the chart + the evidence that produced it, not of the current frame).
            island_yaw_var_ = derived_chart_yaw_var(pts, island_belief_->chart(), bp_.sigma_base_m);
        }
    }

    std::vector<KitchenBox> active_boxes() const
    {
        std::vector<KitchenBox> out;
        for (const auto& c : cells_)
            if (c.active())
            {
                KitchenBox b; b.id = c.geom.id; b.wall_seg_id = c.geom.wall_seg_id;
                b.tier = c.geom.tier; b.existence = c.existence;
                c.belief->room_box(b.cx, b.cy, b.yaw, b.L, b.d, b.z0, b.z1);
                fill_box_covariance(*c.belief, kWallChartYawVar, b);   // chart axis = the room polygon's wall
                out.push_back(b);
            }
        if (island_belief_)                        // the free-standing 4th cabinet (wall_seg_id = -1 ⇒ named _island)
        {
            KitchenBox b; b.id = "island"; b.wall_seg_id = -1; b.tier = 0; b.existence = island_exist_;
            island_belief_->room_box(b.cx, b.cy, b.yaw, b.L, b.d, b.z0, b.z1);
            // ★The island's axis was derived from its OWN points, so it reports the MEASURED variance —
            // typically far looser than a wall cell's. A consumer that fits a shared kitchen axis must be
            // able to see that difference, or it would weight a self-derived axis like a polygon-pinned one.
            fill_box_covariance(*island_belief_, island_yaw_var_, b);
            out.push_back(b);
        }
        return out;
    }
    const std::vector<KitchenCellRun>& cells() const { return cells_; }
    const KitchenCellDiag& island_diag() const { return island_diag_; }
    // The free-standing island/peninsula run — it has no cell (its chart is derived from its own points),
    // so the dashboard needs a direct handle to give it a belief card. Null until activated.
    const WallRunBelief* island()          const { return island_belief_.get(); }
    float                island_existence() const { return island_exist_; }

    static bool self_test();

private:
    // ── Chart-axis (yaw) direction variance ──────────────────────────────────────────────────────
    // A run's yaw is NOT a belief DOF — the wall chart fixes it (see cabinet_wall_run_belief.h). So the
    // yaw variance we publish is the uncertainty of whoever AUTHORED the chart, and it differs sharply
    // between the two cases. That difference is the point: it is what lets a consumer tell a
    // polygon-pinned run from a self-derived one.
    //
    // WALL-ANCHORED cells inherit the collinear-merged room-polygon wall direction. `kColinCos = cos(3°)`
    // in CabinetFitter::rebuild_wall_ids is this codebase's own statement of how much direction spread one
    // merged "wall" may contain, so 3° is read off the existing model rather than invented.
    // ⚠FLAGGED (CLAUDE.md "no thresholds"): this is a STAND-IN for a wall-direction covariance that
    // room_concept does not currently publish. When it does, read it and delete this constant. It is not a
    // tuning knob — do not adjust it to change downstream behaviour.
    static constexpr float kWallChartYawStd = 3.0f * std::numbers::pi_v<float> / 180.0f;
    static constexpr float kWallChartYawVar = kWallChartYawStd * kWallChartYawStd;

    // The ISLAND's chart is derived from its OWN points and then FROZEN at birth, so its axis error is
    // dominated by BIAS, not scatter: derive_island_chart makes a one-shot discrete choice (snap to a wall
    // axis / take a wall normal) that no later evidence can revise, because yaw is not a DOF. Reporting only
    // the precision of a slope estimate would therefore be badly over-confident — it answers "how repeatably
    // could I measure a tilt?" when the question is "how far is the frozen axis from what the points say?".
    //
    // So measure the MISALIGNMENT itself. In chart coords (t along the axis, s across it) the point cloud's
    // own principal angle relative to the chart axis is the standard
    //       δ = ½·atan2(2·S_ts, S_tt − S_ss)
    // which is exactly 0 when the cloud's axes line up with the chart. Add the slope-estimate variance
    // σ_s²/Σ(t−t̄)² so a short or sparse cloud cannot claim a sharp axis merely by having δ≈0:
    //       var(axis) = δ² + σ_s²/Σ(tᵢ − t̄)²
    // Both terms are measured, neither is a gate, and the result is self-diagnosing — a chart that snapped to
    // the wrong segment advertises its own error instead of hiding it. Units: rad².
    static float derived_chart_yaw_var(const std::vector<Eigen::Vector3f>& pts, const WallChart& ch, float sigma_s)
    {
        constexpr float kNoAxisVar = kWallChartYawVar * 100.0f;   // (30°)²: an axis derived from ~nothing
        if (pts.size() < 3) return kNoAxisVar;
        float t, s, tsum = 0.0f, ssum = 0.0f;
        std::vector<Eigen::Vector2f> ts; ts.reserve(pts.size());
        for (const auto& p : pts) { ch.to_wall(p, t, s); ts.emplace_back(t, s); tsum += t; ssum += s; }
        const float inv_n = 1.0f / static_cast<float>(ts.size());
        const float tbar = tsum * inv_n, sbar = ssum * inv_n;
        float Stt = 0.0f, Sss = 0.0f, Sts = 0.0f;
        for (const auto& q : ts)
        { const float dt = q.x() - tbar, ds = q.y() - sbar;
          Stt += dt * dt; Sss += ds * ds; Sts += dt * ds; }
        if (not (Stt > 1e-6f)) return kNoAxisVar;                 // no extent along the axis ⇒ it says nothing
        // Principal angle of the cloud w.r.t. the chart axis. atan2 handles Stt≈Sss (a round blob) without a
        // division; such a cloud has no meaningful axis, and the scatter term below is what dominates there.
        const float delta = 0.5f * std::atan2(2.0f * Sts, Stt - Sss);
        return std::min(kNoAxisVar, delta * delta + sigma_s * sigma_s / Stt);
    }

    // Map a run's wall-chart Σ over θ=[t0,t1,d,z0,z1] onto the ROOM-frame pose covariance the RT edge
    // publishes. The footprint centre is c = A + tc·u + (d/2)·n with tc = (t0+t1)/2, so
    //     ∂c/∂t0 = ∂c/∂t1 = u/2 ,  ∂c/∂d = n/2
    // and Σ_xy = M Σ Mᵀ with M = [u/2, u/2, n/2, 0, 0]. Anisotropic BY CONSTRUCTION and correctly rotated
    // into the room: uncertain ENDS spread the centre ALONG the wall, an uncertain DEPTH spreads it
    // PERPENDICULAR to it. (The classic path publishes a diagonal-only cov; the chart gives us the real
    // orientation for free, so a grazing run reports its true error ellipse.)
    static void fill_box_covariance(const WallRunBelief& b, float yaw_var, KitchenBox& box)
    {
        const auto& S  = b.covariance();
        const auto& ch = b.chart();
        const float var_tc = 0.25f * (S(0, 0) + 2.0f * S(0, 1) + S(1, 1));   // along-wall centre
        const float var_hd = 0.25f *  S(2, 2);                               // half-depth
        const float cov_th = 0.25f * (S(0, 2) + S(1, 2));                    // their cross-term
        const Eigen::Vector2f& u = ch.u;
        const Eigen::Vector2f& n = ch.n;
        box.cov_xx = var_tc * u.x() * u.x() + var_hd * n.x() * n.x() + 2.0f * cov_th * u.x() * n.x();
        box.cov_yy = var_tc * u.y() * u.y() + var_hd * n.y() * n.y() + 2.0f * cov_th * u.y() * n.y();
        box.cov_xy = var_tc * u.x() * u.y() + var_hd * n.x() * n.y() + cov_th * (u.x() * n.y() + n.x() * u.y());
        box.var_z   = std::max(0.0f, S(3, 3));   // the RT edge's translation z IS z0 (publish_kitchen_boxes)
        box.var_yaw = yaw_var;
    }

    // Absence confidence vs range: beams get sparse with distance, so a far pass-through is weak evidence of
    // removal. Scales the FREE half only (occupancy always counts at full weight). power 0 ⇒ constant 1.
    static float absence_conf(const KitchenManagerParams& mp, float range_m)
    {
        if (mp.absence_range_ref_m <= 0.0f or mp.absence_range_power <= 0.0f) return 1.0f;
        return std::min(1.0f, std::pow(mp.absence_range_ref_m / std::max(range_m, 1e-3f), mp.absence_range_power));
    }

    // One run's occupancy/absence update from this cycle's sweep. HOLDs (returns false, streak frozen) when the
    // run was not probed at all — out of range, behind the sensor, or fully occluded. See
    // WallRunBelief::lidar_existence_evidence for the per-beam physics.
    static bool integrate_absence(WallRunBelief& b, float& L, KitchenCellDiag& dg,
                                  const KitchenManagerParams& mp, const CabinetFrame& f)
    {
        exist::SensorModel sm;
        sm.sensor_sigma_m = mp.sensor_sigma_m;
        sm.detection_prob = mp.detection_prob;
        sm.clutter_prob   = mp.clutter_prob;
        float cx, cy, yaw, run_L, d, z0, z1;
        b.room_box(cx, cy, yaw, run_L, d, z0, z1);
        const Eigen::Vector3f centre(cx, cy, 0.5f * (z0 + z1));
        const float range = (f.lidar_freespace.origin - centre).norm();
        const exist::Evidence ev = b.lidar_existence_evidence(f.lidar_freespace, sm, absence_conf(mp, range));
        dg.ex_occ = ev.e_occ; dg.ex_free = ev.e_free; dg.ex_n = ev.n_reached; dg.ex_dL = ev.log_odds_delta;
        if (ev.n_reached == 0) return false;                  // not probed this cycle ⇒ HOLD (streak frozen)
        L = std::clamp(L + ev.log_odds_delta, -mp.logodds_cap, mp.logodds_cap);
        // Debounce on EVIDENCE cycles (not wall-clock), so a mixed/irregular sensor cadence can't bias it.
        if (L <= mp.retire_logodds) ++dg.retire_streak; else dg.retire_streak = 0;
        return true;
    }

    void accumulate_absence(const KitchenManagerParams& mp, const CabinetFrame& f)
    {
        if (not mp.lidar_existence_enabled or f.lidar_freespace.endpoints.empty()) return;
        for (auto& c : cells_)
            if (c.active())
                integrate_absence(*c.belief, c.existence, c.diag, mp, f);
    }

    // Derive the island/peninsula chart. The kitchen "island" here is a PENINSULA: an elongated cabinet attached
    // to a wall by one SHORT side, its long axis PERPENDICULAR to that wall. So the chart's run axis u = the
    // attached wall's inward normal (into the room), n = the wall direction, origin ON the wall (t=0 at the wall),
    // and the near end is ANCHORED to the wall (anchored=true ⇒ t0→0). Falls back to free-standing PCA if the
    // points are not adjacent to any wall. False if too few / degenerate.
    bool derive_island_chart(const std::vector<Eigen::Vector3f>& pts, WallChart& ch, bool& anchored) const
    {
        if (pts.size() < 8) return false;
        // 1) Attached-peninsula: the wall some point comes within ~25 cm of (inward side, within the wall span).
        int best = -1; float bestd = 0.25f;
        for (std::size_t wi = 0; wi < walls_.size(); ++wi)
        {
            const auto& w = walls_[wi];
            float mind = 1e9f;
            for (const auto& p : pts)
            {
                const Eigen::Vector2f d = p.head<2>() - w.a;
                const float tw = d.dot(w.u);                         // along the wall
                if (tw < -0.2f or tw > w.W + 0.2f) continue;
                const float s = d.dot(w.n);                          // inward distance (into room)
                if (s < -0.1f) continue;                             // on the outward side ⇒ not this wall
                mind = std::min(mind, std::abs(s));
            }
            if (mind < bestd) { bestd = mind; best = static_cast<int>(wi); }
        }
        if (best >= 0)                                               // PENINSULA: axis = wall normal, anchored to wall
        {
            const auto& w = walls_[static_cast<std::size_t>(best)];
            const Eigen::Vector2f u = w.n, n = w.u;                  // u into room (run axis), n along wall (depth axis)
            float tmx = -1e9f, smn = 1e9f;
            for (const auto& p : pts) { const Eigen::Vector2f d = p.head<2>() - w.a;
                tmx = std::max(tmx, d.dot(u)); smn = std::min(smn, d.dot(n)); }
            ch.A = w.a + smn * n;      // ON the wall (t=0 = wall), shifted along the wall to the near depth-edge
            ch.u = u; ch.n = n;
            ch.W = std::max(0.4f, tmx) + 0.30f;                      // how far it may reach into the room
            ch.H_room = h_room_;
            anchored = true;
            return true;
        }
        // 2) Free-standing fallback: horizontal PCA major axis, snapped to the nearest wall axis.
        Eigen::Vector2f mean(0.0f, 0.0f);
        for (const auto& p : pts) mean += p.head<2>();
        mean /= static_cast<float>(pts.size());
        Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
        for (const auto& p : pts) { const Eigen::Vector2f d = p.head<2>() - mean; C += d * d.transpose(); }
        C /= static_cast<float>(pts.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
        if (es.info() != Eigen::Success) return false;
        Eigen::Vector2f u = es.eigenvectors().col(1).normalized();
        float bd = 0.0f; Eigen::Vector2f snapped = u;
        for (const auto& w : walls_)
        {
            const Eigen::Vector2f cands[2] = { w.u, Eigen::Vector2f(-w.u.y(), w.u.x()) };
            for (const auto& cand : cands)
            { const float dp = u.dot(cand);
              if (std::abs(dp) > bd) { bd = std::abs(dp); snapped = (dp < 0.0f) ? Eigen::Vector2f(-cand) : cand; } }
        }
        if (bd > 0.9f) u = snapped.normalized();
        const Eigen::Vector2f n(-u.y(), u.x());
        float tmn = 1e9f, tmx = -1e9f, smn = 1e9f, smx = -1e9f;
        for (const auto& p : pts) { const Eigen::Vector2f d = p.head<2>() - mean;
            const float t = d.dot(u), s = d.dot(n);
            tmn = std::min(tmn, t); tmx = std::max(tmx, t); smn = std::min(smn, s); smx = std::max(smx, s); }
        ch.A = mean + tmn * u + smn * n;
        ch.u = u; ch.n = n;
        ch.W = (tmx - tmn) + 0.20f;
        ch.H_room = h_room_;
        anchored = false;
        return true;
    }

    std::vector<KitchenCellRun> cells_;
    CabinetBeliefParams         bp_;
    bool                        corner_fill_log_ = false;
    // Free-standing island (peninsula / 4th cabinet) — NOT wall-anchored: a single run whose chart is derived
    // from its OWN points (PCA), snapped to the nearest wall axis, then fit with the same WallRunBelief.
    std::vector<KitchenWall>       walls_;              // kept for island axis-snap
    WallTierPrior                  base_tier_prior_;
    float                          h_room_ = 2.6f;
    std::unique_ptr<WallRunBelief> island_belief_;      // null until activated
    WallChart                      island_chart_;       // frozen at birth
    bool                           island_anchored_ = false;   // peninsula: near end pinned to the wall (t0→0)
    // MEASURED variance of the frozen chart's axis (derived_chart_yaw_var). Seeded pessimistic so a
    // just-born island never advertises a tight axis before any point has scored it.
    float                          island_yaw_var_ = kWallChartYawVar * 100.0f;
    float                          island_exist_   = 0.0f;
    float                          island_cov_ema_ = 0.0f;
    KitchenCellDiag                island_diag_;               // per-cycle observability (CSV/dashboard)
};

inline bool KitchenManager::self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* w) { if (not c) { std::printf("[kitchen_mgr::self_test] FAIL: %s\n", w); ok = false; } };

    const Eigen::Vector2f I(1.0f, 1.5f);
    const std::vector<KitchenWall> walls = {
        make_kitchen_wall({0, 0}, {0, 3}, I, 0),   // west
        make_kitchen_wall({0, 3}, {2, 3}, I, 1),   // north
        make_kitchen_wall({2, 3}, {2, 0}, I, 2),   // east
    };
    const KitchenTier tiers[2] = { {0.0f, 0.87f, 0.55f, "base"}, {1.40f, 2.10f, 0.35f, "upper"} };
    const WallTierPrior tp[2] = {
        { 0.55f, 0.10f, 0.0f,  0.02f, 0.85f, 0.06f },   // base
        { 0.35f, 0.05f, 1.45f, 0.10f, 2.10f, 0.10f },   // upper
    };
    CabinetBeliefParams bp;
    KitchenManager mgr; mgr.build(walls, tiers, tp, bp, 2.6f);
    check(mgr.cells().size() == 6, "6 cells built");

    // Base front-face clouds for the 3 walls, but each run only spans PART of its wall (west y∈[0.4,2.6],
    // north x∈[0.3,1.7], east y∈[0.4,2.6]) — so we can check the fitted intervals, not just activation.
    const KitchenManagerParams mp;
    const CabinetFrame tmpl;
    const auto push_front = [&](std::vector<Eigen::Vector3f>& v, const KitchenWall& w, float t0, float t1) {
        for (int i = 0; i <= 20; ++i) { const float t = t0 + (t1 - t0) * i / 20.0f;
            const Eigen::Vector2f p = w.a + t * w.u + 0.55f * w.n;
            for (int j = 0; j <= 4; ++j) v.emplace_back(p.x(), p.y(), 0.80f * j / 4.0f); } };
    std::vector<Eigen::Vector3f> pts;
    push_front(pts, walls[0], 0.4f, 2.6f);
    push_front(pts, walls[1], 0.3f, 1.7f);
    push_front(pts, walls[2], 0.4f, 2.6f);

    for (int it = 0; it < 30; ++it) mgr.update(pts, mp, tmpl);
    const auto boxes = mgr.active_boxes();
    int base = 0, upper = 0;
    for (const auto& b : boxes) (b.tier == 0 ? base : upper)++;
    check(base == 3, "exactly 3 BASE cells activate (the U's three runs)");
    check(upper == 0, "NO upper cells activate (no upper evidence)");

    // CORNER-FILL: the U's back (north) run is observed only over [0.3, 1.7], but BOTH side runs (west, east)
    // reach the shared corners (0,3) and (2,3), so the north run must extend to MEET them — filling to the full
    // wall [0, 2] (L ≈ 2.0), leaving no hole in either inside corner. Depth/height untouched by the fill.
    for (const auto& b : boxes)
        if (b.tier == 0 and std::abs(b.cy - 2.45f) < 0.2f)   // north run (front face at y = 3 - 0.55)
        {
            check(b.L > 1.9f, "north run fills corner-to-corner to both shared vertices (no hole in the L)");
            check(b.d > 0.35f and b.d < 0.75f, "north run depth stays physical (near standard)");
            check(b.z1 < 1.0f, "north run top stays at the base worktop (no ceiling-touching)");
        }
    // A run with NO reaching neighbour at a corner must NOT fill to it: the west run's LOW end (0,0) has no
    // perpendicular run, so it stays at its observed start (~0.4), only its NW-corner (high) end fills.
    for (const auto& b : boxes)
        if (b.tier == 0 and std::abs(b.cx - 0.55f) < 0.2f)   // west run (front face at x = 0.55)
            check(b.L > 2.4f and b.L < 2.75f, "west run fills only its shared (NW) corner, not the free (0,0) end");

    // Corner-leakage rejection: a TIGHT cluster (many points, ~4 cm along-wall span) must NOT activate a run.
    // It is the tail of an adjacent run leaking across a corner, not a run of its own — the raw-count existence
    // used to spawn 5 cm slivers exactly here (live: cabinet_w16_base/up, L=0.05). Coverage < 0.5 ⇒ existence
    // is driven negative, so it can never activate no matter how many frames the cluster persists.
    {
        KitchenManager m2; m2.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> tight;
        for (int i = 0; i < 12; ++i)                        // 4 cm span at t≈0.30 on the west base wall
        {
            const float t = 0.30f + 0.04f * static_cast<float>(i) / 11.0f;
            const Eigen::Vector2f p = walls[0].a + t * walls[0].u + 0.55f * walls[0].n;
            for (int j = 0; j <= 3; ++j) tight.emplace_back(p.x(), p.y(), 0.75f * static_cast<float>(j) / 3.0f);
        }
        for (int it = 0; it < 60; ++it) m2.update(tight, mp, tmpl);
        check(m2.active_boxes().empty(), "a 4 cm cluster does NOT activate a degenerate sliver run");
    }

    // A GENUINE narrow run (~0.35 m, one carcass) still activates — coverage discriminates slivers from real
    // narrow units, it does not just penalise short runs.
    {
        KitchenManager m3; m3.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> narrow;
        for (int i = 0; i <= 18; ++i)                       // 0.35 m span on the west base wall
        {
            const float t = 0.30f + 0.35f * static_cast<float>(i) / 18.0f;
            const Eigen::Vector2f p = walls[0].a + t * walls[0].u + 0.55f * walls[0].n;
            for (int j = 0; j <= 3; ++j) narrow.emplace_back(p.x(), p.y(), 0.75f * static_cast<float>(j) / 3.0f);
        }
        for (int it = 0; it < 40; ++it) m3.update(narrow, mp, tmpl);
        check(m3.active_boxes().size() == 1, "a genuine 0.35 m narrow run still activates");
    }

    // PERSISTENCE: a BORN wall run is a static structure — it must SURVIVE sustained look-away (no masks). Absence
    // of evidence is not evidence of absence. (The old behaviour retired here; that removed the kitchen on look-away.)
    for (int it = 0; it < 100; ++it) mgr.update({}, mp, tmpl);
    check(mgr.active_boxes().size() == static_cast<std::size_t>(base), "born runs PERSIST through sustained look-away");

    // OCCLUSION: a born run observed with REDUCED coverage (only a tight part visible — partial occlusion / a
    // grazing view) must PERSIST, not retire. Low coverage is absence of evidence, not evidence the run shrank;
    // the old coverage-kill removed occluded cabinets. Genuine removal is the free-space carve's job.
    {
        KitchenManager m4; m4.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> wide;
        push_front(wide, walls[0], 0.4f, 2.6f);                // born wide on the west wall
        for (int it = 0; it < 30; ++it) m4.update(wide, mp, tmpl);
        const bool born = m4.active_boxes().size() == 1;
        std::vector<Eigen::Vector3f> tight;
        for (int i = 0; i < 12; ++i) { const float t = 1.40f + 0.04f * static_cast<float>(i) / 11.0f;
            const Eigen::Vector2f p = walls[0].a + t * walls[0].u + 0.55f * walls[0].n;
            for (int j = 0; j <= 3; ++j) tight.emplace_back(p.x(), p.y(), 0.75f * static_cast<float>(j) / 3.0f); }
        for (int it = 0; it < 40; ++it) m4.update(tight, mp, tmpl);
        check(born and m4.active_boxes().size() == 1, "a run observed with reduced coverage (occlusion) PERSISTS");
    }

    // ISLAND: a free-standing rectangular cloud NOT on any wall activates a single island run (the 4th cabinet),
    // room-axis-aligned, roughly matching the cloud's extent — and does NOT spawn any wall cell.
    {
        KitchenManager m5; m5.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> isl;                      // 1.2 m (x) × 0.6 m (y) box centred at (1.0, 1.4)
        for (int i = 0; i <= 24; ++i) for (int j = 0; j <= 12; ++j)
        {
            const float x = 1.0f - 0.6f + 1.2f * static_cast<float>(i) / 24.0f;
            const float y = 1.4f - 0.3f + 0.6f * static_cast<float>(j) / 12.0f;
            for (int k = 0; k <= 3; ++k) isl.emplace_back(x, y, 0.80f * static_cast<float>(k) / 3.0f);
        }
        for (int it = 0; it < 40; ++it) { m5.update({}, mp, tmpl); m5.update_island(isl, mp, tmpl); }
        const auto ib = m5.active_boxes();
        int islands = 0, walls_on = 0;
        for (const auto& b : ib) (b.wall_seg_id < 0 ? islands : walls_on)++;
        check(islands == 1 and walls_on == 0, "a free-standing cloud activates ONE island (4th cabinet), no wall cell");
        for (const auto& b : ib) if (b.wall_seg_id < 0)
        {
            check(b.L > 0.9f and b.L < 1.5f, "island length matches the cloud extent (~1.2 m)");
            check(std::abs(b.cx - 1.0f) < 0.3f and std::abs(b.cy - 1.4f) < 0.35f, "island sits at the cloud centre");
        }
        // island persists through look-away too
        for (int it = 0; it < 80; ++it) m5.update_island({}, mp, tmpl);
        int isl_after = 0; for (const auto& b : m5.active_boxes()) if (b.wall_seg_id < 0) ++isl_after;
        check(isl_after == 1, "the island PERSISTS through look-away");
    }

    // PENINSULA: an elongated cabinet ATTACHED to the north wall (y=3) by a short side, long axis PERPENDICULAR
    // to it (runs in −y into the room). The chart axis = the wall normal and the near end is ANCHORED to the wall
    // (reaches y≈3), even though the observed cloud starts ~2 cm short — NOT a PCA free box.
    {
        KitchenManager m6; m6.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> pen;
        for (int i = 0; i <= 24; ++i) for (int j = 0; j <= 8; ++j)   // 1.2 m (y) × 0.5 m (x), touching y≈3 at x≈1.0
        {
            const float y = 2.98f - 1.2f * static_cast<float>(i) / 24.0f;
            const float x = 1.0f - 0.25f + 0.5f * static_cast<float>(j) / 8.0f;
            for (int k = 0; k <= 3; ++k) pen.emplace_back(x, y, 0.80f * static_cast<float>(k) / 3.0f);
        }
        for (int it = 0; it < 40; ++it) { m6.update({}, mp, tmpl); m6.update_island(pen, mp, tmpl); }
        int isl = 0; float cy = 0.0f, L = 0.0f;
        for (const auto& b : m6.active_boxes()) if (b.wall_seg_id < 0) { ++isl; cy = b.cy; L = b.L; }
        check(isl == 1, "a wall-adjacent peninsula activates ONE island run");
        check(L > 1.0f, "peninsula length spans into the room (long axis ⊥ wall)");
        check(cy > 2.2f and cy < 2.75f, "peninsula near end anchored to the north wall (box reaches y≈3)");
    }

    // ENGULFMENT retirement: a run mostly INSIDE another agent's object (the on-fridge false-cabinet the LiDAR
    // carve can't reach) is retired — and does NOT re-birth while the object is present. Partial overlap survives.
    {
        KitchenManager m7; m7.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> wide; push_front(wide, walls[0], 0.4f, 2.6f);
        for (int it = 0; it < 30; ++it) m7.update(wide, mp, tmpl);          // born on the west wall (exclusion off)
        const bool born = m7.active_boxes().size() == 1;
        KitchenManagerParams mpx = mp; mpx.object_exclusion_enabled = true; mpx.engulf_frac = 0.60f;
        CabinetFrame tf; tf.scene_objects = { {0.275f, 1.5f, 0.0f, 0.70f, 2.5f, 0.0f, 1.8f} };   // engulfs the run
        for (int it = 0; it < 5; ++it) m7.update(wide, mpx, tf);
        check(born and m7.active_boxes().empty(), "a run engulfed by another object is retired (and not re-born)");

        KitchenManager m8; m8.build(walls, tiers, tp, bp, 2.6f);
        for (int it = 0; it < 30; ++it) m8.update(wide, mp, tmpl);
        CabinetFrame tf2; tf2.scene_objects = { {0.275f, 0.7f, 0.0f, 0.70f, 0.5f, 0.0f, 1.8f} };  // only y∈[0.45,0.95]
        for (int it = 0; it < 5; ++it) m8.update(wide, mpx, tf2);
        check(m8.active_boxes().size() == 1, "a partially-overlapping object does NOT retire the run");
    }

    // LiDAR EVIDENCE OF ABSENCE — the retirement channel. Without it a born cell is immortal (mask absence
    // deliberately does not fade a born run), which is exactly the "phantom cabinets are never removed" defect.
    // West wall: a=(0,0) b=(0,3), inward normal +x ⇒ the base run's front face is at x=0.55, back on the wall x=0.
    {
        KitchenManagerParams mpl = mp;
        mpl.lidar_existence_enabled = true; mpl.retire_frames = 15;
        const auto born_west = [&](KitchenManager& m) {
            std::vector<Eigen::Vector3f> wide; push_front(wide, walls[0], 0.4f, 2.6f);
            for (int it = 0; it < 30; ++it) m.update(wide, mp, tmpl);
            return m.active_boxes().size() == 1;
        };
        const auto sweep_to = [&](float x_end) {                      // 21 beams from the room toward the west wall
            CabinetFrame f; f.lidar_freespace.origin = Eigen::Vector3f(1.5f, 1.5f, 0.60f);
            for (int i = 0; i <= 20; ++i)
                f.lidar_freespace.endpoints.emplace_back(x_end, 0.4f + 2.2f * static_cast<float>(i) / 20.0f, 0.60f);
            return f;
        };
        // PHANTOM: every beam reaches the WALL BEHIND the run (x=0) ⇒ the run is refuted and retired (debounced).
        KitchenManager m9; m9.build(walls, tiers, tp, bp, 2.6f);
        const bool born9 = born_west(m9);
        const CabinetFrame refute = sweep_to(0.0f);
        for (int it = 0; it < 40; ++it) m9.update({}, mpl, refute);   // no masks: the phantom gets no mask support
        check(born9 and m9.active_boxes().empty(), "a run the LiDAR sees THROUGH (wall behind) is retired");
        // REAL: the same beams stop on the run's FRONT face ⇒ occupancy ⇒ it survives indefinitely.
        KitchenManager m10; m10.build(walls, tiers, tp, bp, 2.6f);
        const bool born10 = born_west(m10);
        const CabinetFrame confirm = sweep_to(0.55f);
        for (int it = 0; it < 40; ++it) m10.update({}, mpl, confirm);
        check(born10 and m10.active_boxes().size() == 1, "a run whose FRONT FACE returns are seen is never retired");
        // OCCLUDED / NOT PROBED: no sweep at all ⇒ no evidence ⇒ HOLD (this is the look-away case, must persist).
        KitchenManager m11; m11.build(walls, tiers, tp, bp, 2.6f);
        const bool born11 = born_west(m11);
        for (int it = 0; it < 60; ++it) m11.update({}, mpl, tmpl);    // mpl on, but tmpl carries NO rays
        check(born11 and m11.active_boxes().size() == 1, "no sweep ⇒ no absence evidence ⇒ the run persists (HOLD)");
        // DEBOUNCE: the refutation must not remove the run before retire_frames evidence cycles have agreed.
        KitchenManager m12; m12.build(walls, tiers, tp, bp, 2.6f);
        born_west(m12);
        for (int it = 0; it < 8; ++it) m12.update({}, mpl, refute);
        check(m12.active_boxes().size() == 1, "removal is debounced (a few refuting cycles cannot delete a run)");
    }

    // ── (i) PUBLISHED ROOM-FRAME COVARIANCE: the Σ every consumer weights us by ──────────────────
    // The kitchen path used to publish pose with no covariance at all, so a consumer could not tell a long,
    // well-observed run from a sliver. These checks pin the three properties that make the published number
    // meaningful: it is a real PSD covariance, it is ORIENTED by the chart (not diagonal-by-assumption), and
    // a self-derived island axis reports honestly looser than a polygon-pinned wall run.
    {
        KitchenManager m13; m13.build(walls, tiers, tp, bp, 2.6f);
        std::vector<Eigen::Vector3f> west; push_front(west, walls[0], 0.4f, 2.6f);
        for (int it = 0; it < 30; ++it) m13.update(west, mp, tmpl);
        const auto bx = m13.active_boxes();
        check(bx.size() == 1, "(i) one west run for the covariance checks");
        if (bx.size() == 1)
        {
            const auto& b = bx.front();
            // PSD: positive variances and a cross-term that respects Cauchy-Schwarz. A consumer inverts this.
            check(b.cov_xx > 0.0f and b.cov_yy > 0.0f and b.var_z >= 0.0f and b.var_yaw > 0.0f,
                  "(i) published covariance is positive");
            check(b.cov_xy * b.cov_xy <= b.cov_xx * b.cov_yy * 1.0001f,
                  "(i) XY block is PSD (|cov_xy| <= sqrt(vxx*vyy))");
            // ORIENTATION: the west wall runs along +y, so the ALONG-wall (t0,t1) uncertainty must land on Y
            // and the DEPTH uncertainty on X. A diagonal-by-assumption write would get this backwards half
            // the time; the chart mapping cannot.
            check(b.cov_yy > b.cov_xx,
                  "(i) a wall run along +y is more uncertain ALONG the wall than across it");
            // A wall run's yaw comes from the room polygon ⇒ the merge-tolerance variance, exactly.
            check(std::abs(b.var_yaw - kWallChartYawVar) < 1e-9f,
                  "(i) a wall-anchored run publishes the polygon's chart-axis variance");
        }
        // ISLAND: the axis is derived from its own points and FROZEN, so what must be published is the
        // chart's MISALIGNMENT. This is the property the whole channel exists for — a chart that snapped to
        // the wrong segment (live: island births 2°–11° off the 0.3°/89.6° kitchen grid, permanently, because
        // yaw is not a DOF) has to advertise that error rather than look as certain as a polygon-pinned run.
        // Build the same run twice: once aligned to the chart the snap will pick, once genuinely rotated.
        const auto island_axis_var = [&](float tilt_rad) -> float {
            std::vector<Eigen::Vector3f> isl;
            const Eigen::Vector2f c(1.20f, 1.10f);                       // clear of every wall ⇒ PCA branch
            const Eigen::Vector2f dir(std::sin(tilt_rad), std::cos(tilt_rad));   // tilt off +y
            const Eigen::Vector2f nrm(-dir.y(), dir.x());
            for (int i = 0; i <= 24; ++i)
            {
                const float t = -0.5f + 1.0f * static_cast<float>(i) / 24.0f;    // 1 m long
                for (int j = 0; j <= 4; ++j)                                     // front face, 0.6 m deep
                {
                    const Eigen::Vector2f p = c + t * dir + 0.30f * nrm;
                    isl.emplace_back(p.x(), p.y(), 0.80f * static_cast<float>(j) / 4.0f);
                }
            }
            KitchenManager m; m.build(walls, tiers, tp, bp, 2.6f);
            for (int it = 0; it < 30; ++it) m.update_island(isl, mp, tmpl);
            const auto b = m.active_boxes();
            const auto it = std::ranges::find_if(b, [](const KitchenBox& x) { return x.wall_seg_id < 0; });
            return it == b.end() ? -1.0f : it->var_yaw;
        };
        const float v_aligned  = island_axis_var(0.0f);
        const float v_tilted   = island_axis_var(8.0f * std::numbers::pi_v<float> / 180.0f);
        check(v_aligned >= 0.0f and v_tilted >= 0.0f, "(i) both islands activated for the axis check");
        // An axis that genuinely matches its points may report tightly — that is honest.
        check(v_aligned < kWallChartYawVar,
              "(i) an island whose frozen axis MATCHES its points reports a tight axis");
        // ★The one that matters: 8° of real misalignment must surface as a much looser published axis, so a
        // consumer fitting a shared kitchen frame down-weights it instead of trusting a wrong angle.
        check(v_tilted > 10.0f * v_aligned,
              "(i) a MISALIGNED frozen island axis publishes a much looser variance (self-diagnosing)");
        check(v_tilted > kWallChartYawVar,
              "(i) ...and looser than a polygon-pinned wall run's");
    }

    if (ok) std::printf("[kitchen_mgr::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
