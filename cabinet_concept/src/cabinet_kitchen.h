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

#include "../../common/footprint/footprint.h"   // rc::geom::Footprint / overlap_ratio (self_test invariant)
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
    // ★A CABINET IS ALWAYS ATTACHED — there is no free-standing island (2026-08-16, the user's rule). Every
    // run in this kitchen abuts a WALL or another CABINET; a cluster that abuts neither is not a cabinet we
    // can interpret, and is refused rather than fitted at an arbitrary place and angle. This is the distance
    // at which "abuts" is read, and it is not a new knob: it is the 0.25 m that was already buried as a
    // literal inside the chart derivation, lifted out so it is visible and can be argued with.
    float attach_gap_m = 0.25f;
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

    // ── WHAT THE ROUTED POINTS ACTUALLY SAY (added 2026-08-10) ──────────────────────────────────
    // The count alone cannot tell us whether a wrong depth came from a wrong FIT or from points that
    // genuinely sit there. These are the distribution of the routed points along the DEPTH axis (the
    // same coordinate the belief calls `d`), so a row can be read directly:
    //   d ≈ lat_max and lat_max ≈ 0.9  ⇒ the fit is right and the MASK/ROUTING is wrong
    //   d ≈ 0.9 but lat_max ≈ 0.6      ⇒ the points are fine and the FIT ran away
    // Without this the two are indistinguishable in the log, which is why the 90 cm run could not be
    // diagnosed from the previous one.
    float lat_min = 0.0f, lat_mean = 0.0f, lat_max = 0.0f;   // off-wall distance of routed points (m)
    float span    = 0.0f;                                    // along-wall extent of routed points (m)

    // ── WHICH MASK LABEL put points where (added 2026-08-10) ────────────────────────────────────
    // A run is fed from one pool that mixes `cabinet`/`chest of drawers` with `counter`/`countertop`.
    // Live, one base run fitted d=0.89 with points reaching 0.92 m off the wall — the fit is faithful,
    // so the question is purely which label those far points carry. A countertop mask can easily take
    // in the overhang plus whatever is standing on it; a carcass mask cannot. Splitting the count AND
    // the reach by label answers that directly instead of by elimination.
    int   n_carcass = 0, n_counter = 0;          // routed points by label group
    float latmax_carcass = 0.0f, latmax_counter = 0.0f;   // how far each group reaches (m)

    // ── WHERE ALONG THE RUN the too-deep points sit (added 2026-08-10) ──────────────────────────
    // A run's depth is ONE number, so a normal 0.6 m run that abuts something bigger at one end has
    // to swallow the intruder and comes out deep everywhere. `lat_max` cannot tell those apart. This
    // does: "far" = beyond the tier's own nominal depth (the routing box), and if those points are
    // CLUSTERED at one along-wall position, a distinct object is being absorbed at that spot; if they
    // are SPREAD across the run, the wall really does carry something that deep along its length.
    // ★Live motive: one run's carcass reach sat at 0.903-0.914 across 6744 cycles — pinned at the
    // router's measured 0.91 m acceptance limit, i.e. a truncation, not an object edge.
    int   n_far = 0;                              // routed points beyond the tier's nominal depth
    float far_t_min = 0.0f, far_t_max = 0.0f;     // their along-wall span, same frame as t0/t1

    // ── PENINSULA chart provenance (peninsula row only) ─────────────────────────────────────────
    // ★The "peninsula" in this apartment is NOT free-standing: it is a cabinet whose SHORT side is
    // against a wall. derive_peninsula_chart has a peninsula branch for exactly that (axis = the wall's
    // inward normal, near end anchored to the wall), and a PCA fallback for a genuinely free unit.
    // Which one fired decides whether the run lands on the kitchen grid or at an arbitrary angle —
    // and it was never logged, so we could only infer it from the resulting yaw.
    int   attach_seg = -1;      // wall segment the peninsula anchored to; -1 = PCA fallback fired
    float wall_gap   = -1.0f;   // closest approach of its points to that wall (m); -1 = not measured
    bool  anchored   = false;   // true = peninsula (wall-derived axis), false = free-standing PCA
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
    // True when this run's chart axis came from a WALL (the peninsula branch), false for a genuinely
    // free-standing unit fitted by PCA. ★The apartment's "peninsula" is the former: a cabinet with its
    // SHORT side against a wall. Only the mask LABEL ("kitchen island", an ADE20K class) sends it
    // down the island path; its geometry is wall-attached, and the distinction changes both how much
    // its yaw should be trusted and what the node should honestly be called.
    bool  anchored = false;
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_xy = 0.0f;   // footprint-centre covariance (m²)
    float var_z   = 0.0f;                                // variance of z0 = the RT translation z (m²)
    float var_yaw = 0.0f;                                // chart-axis direction variance (rad²)
    // Marginal variances of the three SIZE attributes this run publishes (width_m, depth_m, height_m),
    // m². The other half of what a consumer needs: rt_covariance says how well we know WHERE the run
    // is, this says how well we know HOW BIG it is. A level-2 agent fitting a shared worktop plane has
    // to weight members by exactly this, or it falls back to a proxy like run length — and a long run
    // can still be badly fitted.
    float var_width = 0.0f, var_depth = 0.0f, var_height = 0.0f;
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
        base_tier_prior_ = tier_priors[0];   // the peninsula uses the base tier prior
        h_room_ = H_room;
        walls_ = walls;                      // kept for peninsula attachment + axis-snapping
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
    // `labels` (optional, parallel to pts): 0 = carcass mask (cabinet / chest of drawers),
    // 1 = worktop mask (counter / countertop). Diagnostics only — routing and fitting are unchanged.
    void update(const std::vector<Eigen::Vector3f>& pts, const KitchenManagerParams& mp,
                const CabinetFrame& frame_template, const std::vector<std::uint8_t>* labels = nullptr)
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
        std::vector<int>   lab_n[2];   // [0] carcass, [1] counter — per-cell routed counts
        std::vector<float> lab_far[2];
        for (int g = 0; g < 2; ++g) { lab_n[g].assign(K, 0); lab_far[g].assign(K, 0.0f); }
        for (std::size_t pi = 0; pi < pts.size(); ++pi)
        {
            const auto& p = pts[pi];
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
            // Attribute this point to its mask label, and record how far out that label reached.
            {
                const int g = (labels and pi < labels->size() and (*labels)[pi] == 1) ? 1 : 0;
                const auto& cg = cells_[static_cast<std::size_t>(kbest)].geom;
                const float lat = (Eigen::Vector2f(p.x(), p.y()) - cg.a).dot(cg.n);
                ++lab_n[g][static_cast<std::size_t>(kbest)];
                float& far = lab_far[g][static_cast<std::size_t>(kbest)];
                if (lab_n[g][static_cast<std::size_t>(kbest)] == 1 or lat > far) far = lat;
            }
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
            // Depth-axis distribution of exactly the points this cell will be fitted to.
            c.diag.span = span;
            c.diag.n_carcass = lab_n[0][k];  c.diag.latmax_carcass = lab_far[0][k];
            c.diag.n_counter = lab_n[1][k];  c.diag.latmax_counter = lab_far[1][k];
            // Along-wall position of the points that lie beyond a nominal carcass for this tier.
            c.diag.n_far = 0; c.diag.far_t_min = 0.0f; c.diag.far_t_max = 0.0f;
            for (const auto& p : cpts[k])
            {
                const Eigen::Vector2f q(p.x(), p.y());
                if ((q - c.geom.a).dot(c.geom.n) <= c.geom.depth) continue;   // within a normal carcass
                const float tf = (q - c.geom.a).dot(c.geom.u);
                if (c.diag.n_far == 0) { c.diag.far_t_min = c.diag.far_t_max = tf; }
                else { c.diag.far_t_min = std::min(c.diag.far_t_min, tf);
                       c.diag.far_t_max = std::max(c.diag.far_t_max, tf); }
                ++c.diag.n_far;
            }
            c.diag.lat_min = c.diag.lat_mean = c.diag.lat_max = 0.0f;
            if (not cpts[k].empty())
            {
                float lmn = 1e9f, lmx = -1e9f; double lsum = 0.0;
                for (const auto& p : cpts[k])
                { const float lat = (Eigen::Vector2f(p.x(), p.y()) - c.geom.a).dot(c.geom.n);
                  lmn = std::min(lmn, lat); lmx = std::max(lmx, lat); lsum += lat; }
                c.diag.lat_min = lmn; c.diag.lat_max = lmx;
                c.diag.lat_mean = static_cast<float>(lsum / static_cast<double>(cpts[k].size()));
            }

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

        // ★A SIBLING CELL IS AN OBJECT TOO (2026-08-16). The runs of ONE kitchen could grow through each
        // other anywhere, and nothing said they could not: `accumulate_object_exclusion` — the non-penetration
        // term that already retracts a run out of another agent's furniture, with an exact OBB shadow and a
        // full depth/z/along-wall conflict gate — only ever saw `scene_objects`, i.e. OTHER agents' boxes.
        // Its own siblings were invisible to it, and so was `rc::exclusion` (foreign_claims() drops
        // same-prefix nodes by design). route_corner_fill covers ONE case of this, the shared vertex, and
        // only when both runs are already within corner_capture_m of it; away from a corner, or before that
        // pairing engages, there was no term at all.
        //
        // So feed it the siblings. No new mechanism, no new constant, and no special-casing of tiers: the
        // existing gate `o.z1 <= s.z0 or o.z0 >= s.z1` already lets a wall unit sit above a base unit
        // untouched, which is the same z reasoning common/exclusion needed.
        //
        // ★SNAPSHOT BEFORE THE LOOP, not inside it. Reading each sibling's box as we go would let cell 0's
        // freshly updated state be what cell 1 avoids while cell 1's stale state is what cell 0 avoided —
        // making the result depend on cell ORDER, which is the kind of asymmetry that shows up months later
        // as one particular run always losing.
        sib_boxes_.clear(); sib_ids_.clear();
        const auto add_box = [&](const WallRunBelief& b, const std::string& owner)
        {
            SceneObjectBox o; float L = 0.0f;
            b.room_box(o.cx, o.cy, o.yaw, L, o.d, o.z0, o.z1);
            o.w = L;                              // SceneObjectBox.w is the extent along its own +x = the run's L
            sib_boxes_.push_back(o); sib_ids_.push_back(owner);
        };
        for (std::size_t k = 0; k < K; ++k)
            if (cells_[k].active()) add_box(*cells_[k].belief, cells_[k].geom.id);
        if (peninsula_belief_) add_box(*peninsula_belief_, "peninsula");

        // Second pass: update each active cell's belief with its assigned points (+ the corner-fill flags set above).
        for (std::size_t k = 0; k < K; ++k)
        {
            auto& c = cells_[k];
            if (not c.active() or cpts[k].empty()) continue;
            CabinetFrame f = frame_template;            // carries chain/ego/lidar_freespace
            f.points = std::move(cpts[k]); f.R = std::move(cR[k]);
            for (std::size_t b = 0; b < sib_boxes_.size(); ++b)
                if (sib_ids_[b] != c.geom.id)           // everything except THIS run
                    f.scene_objects.push_back(sib_boxes_[b]);
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

    // Set per-cell corner-fill flags AND targets. Two active same-tier cells on DIFFERENT walls that share a
    // polygon vertex V (walls meet there) form an L; if BOTH runs already reach near V, close the inside corner.
    //
    // ★ONLY ONE OF THEM MAY REACH V (fixed 2026-08-16). Filling both to the vertex is not "closing the
    // corner", it is claiming the corner TWICE: run i extended to V already occupies the first d_i metres of
    // wall j, so run j extended to that same V interpenetrates it by exactly d_i x d_j. Measured live:
    // cabinet_w13_base (d=0.511) and cabinet_w14_base (d=0.741) overlapped 0.370 m2 against 0.379 m2
    // predicted by that product — the 2% is the two depths drifting between logs. A real L kitchen has ONE
    // corner unit and the other run butts into its side.
    //
    // WHO OWNS IT IS READ OFF THE EVIDENCE rather than chosen: the run whose end already sits nearer V (the
    // smaller gap) is the one with support at the corner, so it takes the vertex and its neighbour stops at
    // its front face — V offset by the owner's depth, along the neighbour's own axis. Ties go to the DEEPER
    // run (a deeper carcass is the one whose side face the shallower can butt into), then to the lower cell
    // index, so the answer is deterministic and cannot oscillate from cycle to cycle.
    //
    // ⚠THIS CANNOT BE DELEGATED TO common/exclusion. foreign_claims() drops every node sharing the agent's
    // own prefix by design — cabinet-vs-cabinet is the cell partition's business, not another concept's — so
    // the shared no-two-objects rule is structurally blind to exactly this pair, however it is wired.
    void route_corner_fill(const KitchenManagerParams& mp)
    {
        const std::size_t K = cells_.size();
        std::vector<char> fl(K, 0), fh(K, 0);   // fill-low / fill-high, accumulated (a run can corner on both ends)
        // Where each flagged end should reach. Starts at the vertex (0 / W); a run that LOSES a corner has its
        // target pulled back to the winner's front face. A run cornering at both ends keeps the tighter target.
        // `sl`/`sh` mark an end whose target is a NEIGHBOUR rather than the room, which makes the prior
        // two-sided — that is what retracts a run already sitting inside its neighbour, as today's pair is.
        std::vector<float> tl(K, 0.0f), th(K, 0.0f);
        std::vector<char>  sl(K, 0), sh(K, 0);
        for (std::size_t k = 0; k < K; ++k) th[k] = cells_[k].geom.W;
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
                    // The corner is ONE volume. It goes to the run with support nearest V; the other stops a
                    // depth short of the vertex, which is where the winner's carcass side actually stands.
                    const bool i_owns = (gap_i != gap_j) ? (gap_i < gap_j)
                                      : (si.d  != sj.d)  ? (si.d  > sj.d)
                                                         : true;          // i < j ⇒ deterministic
                    const std::size_t lose    = i_owns ? j  : i;
                    const int         lose_hi = i_owns ? hj : hi;
                    const float       setback = i_owns ? si.d : sj.d;     // the winner's depth = its side face
                    if (lose_hi)
                    { th[lose] = std::min(th[lose], cells_[lose].geom.W - setback); sh[lose] = 1; }
                    else
                    { tl[lose] = std::max(tl[lose], setback);                       sl[lose] = 1; }
                }
            }
        }
        for (std::size_t k = 0; k < K; ++k)
            if (cells_[k].active())
                cells_[k].belief->set_corner_fill({fl[k] != 0, tl[k], sl[k] != 0},
                                                  {fh[k] != 0, th[k], sh[k] != 0});

        if (corner_fill_log_)   // opt-in diagnostic (agent only): per active cell, wall/interval/fill flags
        {
            for (std::size_t k = 0; k < K; ++k)
                if (cells_[k].active())
                {
                    const auto& g = cells_[k].geom; const auto& s = cells_[k].belief->state();
                    const Eigen::Vector2f bb = g.a + g.W * g.u;
                    // The TARGETS are printed beside the flags: "fill=t0 1" with target 0.00 is a run that
                    // owns its corner, and one with a non-zero target is a run stopping at its neighbour's
                    // face. Without them the log cannot distinguish the two, which is the whole question here.
                    std::printf("[kitchen] cell %-12s wall=%d a=(%.2f,%.2f) b=(%.2f,%.2f) t=[%.2f,%.2f] "
                                "fill=t0%d/t1%d target=[%.2f,%.2f] W=%.2f\n",
                                g.id.c_str(), g.wall_seg_id, g.a.x(), g.a.y(), bb.x(), bb.y(),
                                s.t0, s.t1, fl[k] ? 1 : 0, fh[k] ? 1 : 0, tl[k], th[k], g.W);
                }
        }
    }
    void set_corner_fill_log(bool on) { corner_fill_log_ = on; }

    // The PENINSULA: the run that is not one of the (wall, tier) cells. Its chart is derived
    // from its own points (horizontal PCA, snapped to the nearest wall axis for room-grid alignment) and frozen at
    // birth. Same coverage-gated existence + persistence as the wall cells. `pts` = only the island-LABELLED masks
    // (the ADE20K class is still called "kitchen island"; what we build from it is a peninsula).
    void update_peninsula(const std::vector<Eigen::Vector3f>& pts, const KitchenManagerParams& mp,
                       const CabinetFrame& frame_template)
    {
        // EVIDENCE OF ABSENCE FIRST — before any early return on "no island mask this frame". A phantom island
        // is exactly the case that gets no masks, so deferring this behind the mask check would make it immortal.
        peninsula_diag_.n_route = static_cast<int>(pts.size());
        peninsula_diag_.ex_occ = peninsula_diag_.ex_free = peninsula_diag_.ex_dL = 0.0f; peninsula_diag_.ex_n = 0;
        if (peninsula_belief_ and mp.lidar_existence_enabled and not frame_template.lidar_freespace.endpoints.empty())
        {
            integrate_absence(*peninsula_belief_, peninsula_exist_, peninsula_diag_, mp, frame_template);
            if (peninsula_exist_ <= mp.retire_logodds and peninsula_diag_.retire_streak >= mp.retire_frames)
            { peninsula_belief_.reset(); peninsula_exist_ = 0.0f; peninsula_cov_ema_ = 0.0f; peninsula_diag_ = {};
              peninsula_yaw_var_ = kWallChartYawVar * 100.0f; return; }
        }
        if (pts.size() < 2)                        // no island evidence this frame
        {
            if (not peninsula_belief_)                // fade a CANDIDATE; a born island PERSISTS through look-away
                peninsula_exist_ = std::max(-mp.logodds_cap, peninsula_exist_ - mp.absence_decay);
            return;
        }
        WallChart ch; bool anchored = peninsula_anchored_, axis_from_wall = peninsula_axis_from_wall_;
        if (peninsula_belief_) ch = peninsula_chart_;    // frozen chart once born
        else if (not derive_peninsula_chart(pts, mp, ch, anchored, axis_from_wall,
                                         &peninsula_diag_.attach_seg, &peninsula_diag_.wall_gap))
        {
            // ★SAY SO. A cluster of cabinet-labelled points that abuts neither a wall nor a cabinet is now
            // REFUSED, where before it became a free-standing island at an arbitrary angle. That is the
            // intended behaviour, but a refusal that happens in silence is indistinguishable from a mask
            // that never arrived — and this agent has paid for that confusion before.
            static int refuse_log = 0;
            if ((refuse_log++ % 30) == 0)
                std::printf("[kitchen] peninsula candidate REFUSED (%zu pts): abuts no wall and no cabinet — "
                            "there is no free-standing island in this model\n", pts.size());
            return;
        }

        float tmn = 1e9f, tmx = -1e9f, smn = 1e9f, smx = -1e9f, t, s;
        for (const auto& p : pts) { ch.to_wall(p, t, s);
            tmn = std::min(tmn, t); tmx = std::max(tmx, t); smn = std::min(smn, s); smx = std::max(smx, s); }
        const float cov = std::clamp((tmx - tmn) / mp.coverage_ref_m, 0.0f, 1.0f);
        peninsula_exist_   = std::min(mp.logodds_cap, peninsula_exist_ + mp.presence_gain);
        peninsula_cov_ema_ += mp.coverage_ema_rate * (cov - peninsula_cov_ema_);

        if (not peninsula_belief_ and peninsula_exist_ >= mp.activate_logodds
            and peninsula_cov_ema_ >= mp.coverage_birth and cov >= 0.5f and pts.size() >= 8)
        {
            WallRunState s0;
            s0.t0 = anchored ? 0.0f : std::clamp(tmn, 0.0f, ch.W);   // peninsula near end sits ON the wall
            s0.t1 = std::clamp(tmx, 0.0f, ch.W);
            s0.d  = std::clamp(smx - smn, 0.35f, 1.2f);              // seed depth = observed cabinet width
            s0.z0 = base_tier_prior_.z0_mean; s0.z1 = base_tier_prior_.z1_mean;
            if (not (mp.object_exclusion_enabled and run_engulfed(s0, ch, frame_template.scene_objects, mp)))
            { peninsula_chart_ = ch; peninsula_anchored_ = anchored; peninsula_axis_from_wall_ = axis_from_wall;
              peninsula_belief_ = std::make_unique<WallRunBelief>(s0, bp_, peninsula_chart_, base_tier_prior_); }
        }
        // (No coverage-kill: a partially-occluded island is absence of evidence, not evidence it shrank — the
        //  free-space carve retracts a genuinely-empty end. Killing on low coverage removed occluded units.)
        if (peninsula_belief_ and mp.object_exclusion_enabled and not frame_template.scene_objects.empty()
            and run_engulfed(peninsula_belief_->state(), peninsula_belief_->chart(), frame_template.scene_objects, mp))
        { peninsula_belief_.reset(); peninsula_exist_ = 0.0f; peninsula_cov_ema_ = 0.0f;
          peninsula_yaw_var_ = kWallChartYawVar * 100.0f; return; }
        if (peninsula_belief_)
        {
            // Anchor the near end to the wall (t0→0). A peninsula touches ONE wall and no second run, so it
            // has no corner to share and its target is the wall itself.
            peninsula_belief_->set_corner_fill({peninsula_anchored_, 0.0f, false}, {});
            CabinetFrame f = frame_template;
            f.points = pts; f.R.assign(pts.size(), bp_.sigma_base_m * bp_.sigma_base_m);
            // The wall runs are objects to the island exactly as it is one to them — the same snapshot taken
            // in update(), minus the island's own box. A peninsula grows along a wall it shares with a run,
            // so this is the pair most likely to interpenetrate, not a corner case.
            for (std::size_t b = 0; b < sib_boxes_.size(); ++b)
                if (sib_ids_[b] != "peninsula")
                    f.scene_objects.push_back(sib_boxes_[b]);
            peninsula_belief_->update(f);
            peninsula_diag_.anchored = peninsula_anchored_;
            // Depth-axis spread of the island's own points, same reading as a wall cell's.
            {
                float t, sv, lmn = 1e9f, lmx = -1e9f, tmn2 = 1e9f, tmx2 = -1e9f; double lsum = 0.0;
                for (const auto& p : pts) { peninsula_belief_->chart().to_wall(p, t, sv);
                    lmn = std::min(lmn, sv); lmx = std::max(lmx, sv); lsum += sv;
                    tmn2 = std::min(tmn2, t); tmx2 = std::max(tmx2, t); }
                peninsula_diag_.lat_min = lmn; peninsula_diag_.lat_max = lmx;
                peninsula_diag_.lat_mean = static_cast<float>(lsum / static_cast<double>(pts.size()));
                peninsula_diag_.span = tmx2 - tmn2;
                // Same reading for the free-standing run, against the base tier's nominal depth.
                peninsula_diag_.n_far = 0; peninsula_diag_.far_t_min = 0.0f; peninsula_diag_.far_t_max = 0.0f;
                for (const auto& p : pts)
                {
                    peninsula_belief_->chart().to_wall(p, t, sv);
                    if (sv <= base_tier_prior_.d_mean) continue;
                    if (peninsula_diag_.n_far == 0) { peninsula_diag_.far_t_min = peninsula_diag_.far_t_max = t; }
                    else { peninsula_diag_.far_t_min = std::min(peninsula_diag_.far_t_min, t);
                           peninsula_diag_.far_t_max = std::max(peninsula_diag_.far_t_max, t); }
                    ++peninsula_diag_.n_far;
                }
            }
            // Re-measure how well THIS cycle's points pin the frozen chart's axis. Held across look-away
            // (it is a property of the chart + the evidence that produced it, not of the current frame).
            peninsula_yaw_var_ = derived_chart_yaw_var(pts, peninsula_belief_->chart(), bp_.sigma_base_m);
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
        if (peninsula_belief_)                        // the peninsula (wall_seg_id = -1 ⇒ named cabinet_peninsula)
        {
            KitchenBox b; b.id = "peninsula"; b.wall_seg_id = -1; b.tier = 0; b.existence = peninsula_exist_;
            b.anchored = peninsula_anchored_;
            peninsula_belief_->room_box(b.cx, b.cy, b.yaw, b.L, b.d, b.z0, b.z1);
            // ★If the PENINSULA branch fired, this run's axis is a WALL's inward normal — polygon-derived
            // exactly like a wall cell's, so it deserves the same trust, and reporting the PCA misalignment
            // instead would make the metaconcept distrust an axis that is in fact pinned. Take the WORSE of
            // the two: a peninsula anchored to the WRONG wall still shows its measured disagreement.
            // ★KEYED ON THE AXIS, NOT ON BEING ATTACHED. Every run is attached now, so `anchored` here would
            // hand the polygon's variance to a run whose direction came from PCA — claiming a certainty it
            // has not got, in the one channel built to stop exactly that. A cabinet-attached run keeps its
            // MEASURED value, which is what lets an 8° misalignment surface instead of hiding.
            fill_box_covariance(*peninsula_belief_,
                                peninsula_axis_from_wall_ ? std::max(kWallChartYawVar, peninsula_yaw_var_)
                                                       : peninsula_yaw_var_, b);
            out.push_back(b);
        }
        return out;
    }
    // Hand a level-2 END PRIOR to one cell's belief. `t0/t1` are already in that cell's chart — the
    // worker projects the room-frame targets, since only it reads the graph. info 0 clears that end.
    //
    // ★THE TARGET IS CLAMPED OUT OF A SIBLING'S BODY FIRST. The arrangement prior is the one mechanism that
    // drives an end to where NO point supports it — that is its whole job, making the kitchen read as one
    // continuous surface — and it knows nothing about which run already stands there. Left alone it becomes
    // a tug-of-war against non-penetration decided by whichever precision is larger, which is comparing a
    // metaconcept's confidence about STYLE with a physical fact. Measured on the rig: with both at 800 they
    // split the difference and left 0.145 m² of interpenetration standing.
    //
    // Clamping the TARGET resolves it where the contradiction actually is, and it costs the metaconcept
    // nothing it was entitled to: continuity is still requested, just to the neighbour's FACE, which is
    // where a continuous kitchen surface has its seam anyway. Exactly the setback route_corner_fill applies
    // at a vertex, applied to the other driver of growth. No precision comparison, no new constant.
    void set_cell_end_prior(const std::string& cell_id, float t0, float t0_info, float t1, float t1_info)
    {
        for (auto& c : cells_)
            if (c.geom.id == cell_id and c.belief)
            {
                clamp_end_targets_out_of_siblings(*c.belief, cell_id, t0, t1);
                c.belief->set_end_prior(t0, t0_info, t1, t1_info); return;
            }
        if (cell_id == "peninsula" and peninsula_belief_)
        {
            clamp_end_targets_out_of_siblings(*peninsula_belief_, cell_id, t0, t1);
            peninsula_belief_->set_end_prior(t0, t0_info, t1, t1_info);
        }
    }
    void clear_end_priors()
    {
        for (auto& c : cells_) if (c.belief) c.belief->clear_end_prior();
        if (peninsula_belief_) peninsula_belief_->clear_end_prior();
    }

    // Move an end TARGET off any sibling run it lands inside, onto that run's near face. Uses the same OBB
    // shadow the fit's non-penetration term uses (obb_shadow_on_chart), so the target and the fit cannot
    // disagree about where the neighbour is. Reads the snapshot taken in update(); on the very first cycle
    // it is empty and nothing is clamped, which is correct — no run has a body yet.
    void clamp_end_targets_out_of_siblings(const WallRunBelief& belief, const std::string& self_id,
                                           float& t0, float& t1) const
    {
        const auto& s = belief.state();
        for (std::size_t b = 0; b < sib_boxes_.size(); ++b)
        {
            if (sib_ids_[b] == self_id) continue;                   // never clamp against ourselves
            const auto& o = sib_boxes_[b];
            float o_t0, o_t1, o_s0, o_s1;
            obb_shadow_on_chart(o, belief.chart(), o_t0, o_t1, o_s0, o_s1);
            if (o_s1 <= 0.0f or o_s0 >= s.d)  continue;             // sibling is in front of / behind us
            if (o.z1 <= s.z0 or o.z0 >= s.z1) continue;             // a different storey — see common/exclusion
            // ★THE TEST IS "WOULD REACHING IT CROSS THE SIBLING", NOT "IS IT INSIDE THE SIBLING". A target
            // BEYOND a neighbour is the worse request of the two — it asks the run to swallow it whole — and
            // an inside-the-body test misses it entirely, which is exactly what the rig caught. So this is
            // the same case split accumulate_object_exclusion makes, on the same margin, so the target the
            // fit is given and the boundary the fit enforces are the identical number.
            const float tc = 0.5f * (s.t0 + s.t1);
            const float m  = bp_.object_exclusion_margin_m;
            if (tc >= o_t1)      t0 = std::max(t0, o_t1 + m);       // sibling below our centre ⇒ t0 stops above it
            else if (tc <= o_t0) t1 = std::min(t1, o_t0 - m);       // sibling above our centre ⇒ t1 stops below it
            // else our centre is inside the sibling: engulfed, which is the retirement pass's call, not ours.
        }
    }
    // The chart a cell is fitted in — the worker needs it to project a room-frame end target onto t.
    const WallChart* cell_chart(const std::string& cell_id) const
    {
        for (const auto& c : cells_)
            if (c.geom.id == cell_id and c.belief) return &c.belief->chart();
        if (cell_id == "peninsula" and peninsula_belief_) return &peninsula_belief_->chart();
        return nullptr;
    }

    const std::vector<KitchenCellRun>& cells() const { return cells_; }
    const KitchenCellDiag& peninsula_diag() const { return peninsula_diag_; }
    // The peninsula run — it has no cell (its chart is derived from its own points or its host wall),
    // so the dashboard needs a direct handle to give it a belief card. Null until activated.
    const WallRunBelief* peninsula()          const { return peninsula_belief_.get(); }
    float                peninsula_existence() const { return peninsula_exist_; }
    // ★Was hard-coded to 0 in the CSV writer, hiding one of the island's own birth gates — awkward
    // precisely when birth is what is being debugged.
    float                peninsula_cov_ema()   const { return peninsula_cov_ema_; }

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
    // dominated by BIAS, not scatter: derive_peninsula_chart makes a one-shot discrete choice (snap to a wall
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
        // SIZE marginals, from the SAME Σ over θ=[t0,t1,d,z0,z1]:
        //   width  = t1 − t0  ⇒ var = S00 + S11 − 2·S01
        //   depth  = d        ⇒ var = S22
        //   height = z1 − z0  ⇒ var = S33 + S44 − 2·S34
        // The two differences carry their cross-terms explicitly: the ends of a run are strongly
        // correlated (the whole interval slides together), so dropping S01/S34 would over-state the
        // extent uncertainty by roughly a factor of two.
        box.var_width  = std::max(0.0f, S(0, 0) + S(1, 1) - 2.0f * S(0, 1));
        box.var_depth  = std::max(0.0f, S(2, 2));
        box.var_height = std::max(0.0f, S(3, 3) + S(4, 4) - 2.0f * S(3, 4));
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

    // Derive the chart for a run that is NOT one of the (wall, tier) cells — the elongated cabinet this
    // apartment has projecting into the room. It is a PENINSULA: attached by one SHORT side, long axis
    // PERPENDICULAR to what it attaches to. Chart run axis u points into the room, n runs across, origin sits
    // ON the attachment (t=0), and the near end is ANCHORED there (anchored=true ⇒ t0→0).
    //
    // ★THERE IS NO FREE-STANDING BRANCH ANY MORE (2026-08-16, the user's rule). "Island" is not an object this
    // agent may interpret: every cabinet abuts a WALL or another CABINET, and a cluster that abuts neither is
    // refused rather than fitted at an arbitrary place and angle. That old PCA fallback is exactly where an
    // arbitrary yaw came from, and the kitchen metaconcept then had to carry a whole unpinned-member case to
    // stop it dragging the grid (its own header says so). Removing the branch removes the case.
    //
    // Attachment is looked for in two places, in order of how much it tells us:
    //   (a) a WALL — gives a polygon-derived axis, the strongest thing available;
    //   (b) another active CABINET RUN — gives an attachment POINT; the axis then comes from the cluster's own
    //       spread, snapped to a wall direction, so the run still lands on the kitchen grid.
    // False if too few points, degenerate, or attached to nothing.
    bool derive_peninsula_chart(const std::vector<Eigen::Vector3f>& pts, const KitchenManagerParams& mp,
                             WallChart& ch, bool& anchored, bool& axis_from_wall,
                             int* out_seg = nullptr, float* out_gap = nullptr) const
    {
        if (out_seg) *out_seg = -1;
        if (out_gap) *out_gap = -1.0f;
        if (pts.size() < 8) return false;
        // (a) Attached to a WALL: the wall some point comes within attach_gap_m of (inward side, within span).
        int best = -1; float bestd = mp.attach_gap_m;
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
            if (out_gap and (*out_gap < 0.0f or mind < *out_gap)) *out_gap = mind;   // closest wall, pass or fail
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
            axis_from_wall = true;      // the room polygon gave us this direction
            if (out_seg) *out_seg = w.seg_id;
            return true;
        }
        // (b) Attached to another CABINET RUN. Nothing else may anchor it: no wall, no sibling ⇒ no birth.
        // The boxes are this cycle's snapshot, taken in update() before any cell was refitted.
        const SceneObjectBox* host = nullptr;
        {
            float bestg = mp.attach_gap_m;
            for (std::size_t b = 0; b < sib_boxes_.size(); ++b)
            {
                if (sib_ids_[b] == "peninsula") continue;                   // that is us
                const auto& o = sib_boxes_[b];
                const float c = std::cos(o.yaw), s = std::sin(o.yaw);
                float mind = 1e9f;
                for (const auto& p : pts)
                {
                    const float dx = p.x() - o.cx, dy = p.y() - o.cy;
                    const float lx =  c * dx + s * dy, ly = -s * dx + c * dy;   // room → box-local
                    const float ex = std::max(0.0f, std::abs(lx) - 0.5f * o.w); // outside-distance per axis;
                    const float ey = std::max(0.0f, std::abs(ly) - 0.5f * o.d); // both 0 ⇒ the point is inside
                    mind = std::min(mind, std::hypot(ex, ey));
                }
                if (mind < bestg) { bestg = mind; host = &o; }
            }
            if (host == nullptr)
            {
                if (out_gap and *out_gap < 0.0f) *out_gap = -1.0f;
                return false;         // ★abuts neither a wall nor a cabinet ⇒ not a cabinet we can interpret
            }
        }
        // Axis from the cluster's own spread, snapped to a wall direction so it still lands on the kitchen grid.
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
        // ★POINT u AWAY FROM THE HOST, so t grows into the room and t=0 is the attachment — the same sense the
        // wall branch above gives it. PCA hands back an axis with an arbitrary sign; leaving it would put the
        // anchor on whichever end the eigenvector happened to point at, which is the run's FAR end half the
        // time, and the t0→0 prior would then drag the whole run through its host.
        if ((mean - Eigen::Vector2f(host->cx, host->cy)).dot(u) < 0.0f) u = -u;
        const Eigen::Vector2f n(-u.y(), u.x());
        float tmn = 1e9f, tmx = -1e9f, smn = 1e9f, smx = -1e9f;
        for (const auto& p : pts) { const Eigen::Vector2f d = p.head<2>() - mean;
            const float t = d.dot(u), s = d.dot(n);
            tmn = std::min(tmn, t); tmx = std::max(tmx, t); smn = std::min(smn, s); smx = std::max(smx, s); }
        ch.A = mean + tmn * u + smn * n;
        ch.u = u; ch.n = n;
        ch.W = (tmx - tmn) + 0.20f;
        ch.H_room = h_room_;
        // ★t = 0 MUST BE THE ATTACHMENT, not the first point we happened to see. The chart above starts at the
        // cluster's near edge, which is wherever the mask stopped — so pull the origin back onto the host's
        // face and lengthen the chart by the same amount. Now the anchor prior (t0→0) says exactly what the
        // rule says: this cabinet touches the one it is attached to. The host's shadow is taken with the SAME
        // projection the fit's non-penetration uses, so the two cannot disagree about where its face is.
        {
            float h_t0, h_t1, h_s0, h_s1;
            obb_shadow_on_chart(*host, ch, h_t0, h_t1, h_s0, h_s1);
            if (h_t1 < 0.0f) { ch.A += h_t1 * ch.u; ch.W -= h_t1; }   // host behind t=0 ⇒ move the origin onto it
        }
        anchored = true;          // attached to a cabinet rather than a wall, but attached — no other kind exists
        axis_from_wall = false;   // ...and the DIRECTION is still our own, so it must publish its own variance
        if (out_gap) *out_gap = 0.0f;
        return true;
    }

    std::vector<KitchenCellRun> cells_;
    CabinetBeliefParams         bp_;
    bool                        corner_fill_log_ = false;
    // The PENINSULA — a run that is not a (wall, tier) cell but is still ATTACHED, to a wall or to another
    // cabinet (there is no free-standing kind), then fit with the same WallRunBelief.
    std::vector<KitchenWall>       walls_;              // kept for peninsula attachment + axis-snap
    WallTierPrior                  base_tier_prior_;
    float                          h_room_ = 2.6f;
    std::unique_ptr<WallRunBelief> peninsula_belief_;      // null until activated
    WallChart                      peninsula_chart_;       // frozen at birth
    bool                           peninsula_anchored_ = false;   // peninsula: near end pinned to the wall (t0→0)
    // ★`anchored` STOPPED BEING ONE FACT when the free-standing branch went. It used to mean both "the near
    // end is pinned" and "the axis came from the room polygon", because only the wall branch produced either.
    // Now every run is pinned — that is the rule — while a run attached to a CABINET still takes its axis
    // from its own points. Conflating them made a PCA-snapped axis report the polygon's variance, i.e. claim
    // a certainty it has not got, which is the exact failure the published-covariance channel exists to
    // prevent. So the axis provenance is its own bit and only it selects the yaw variance.
    bool                           peninsula_axis_from_wall_ = false;
    // MEASURED variance of the frozen chart's axis (derived_chart_yaw_var). Seeded pessimistic so a
    // just-born island never advertises a tight axis before any point has scored it.
    float                          peninsula_yaw_var_ = kWallChartYawVar * 100.0f;
    float                          peninsula_exist_   = 0.0f;
    float                          peninsula_cov_ema_ = 0.0f;
    KitchenCellDiag                peninsula_diag_;               // per-cycle observability (CSV/dashboard)
    // This cycle's snapshot of every run's box, taken in update() BEFORE any of them is refitted, so which
    // run avoids which does not depend on the order cells happen to be stored in. update_peninsula() runs after
    // update() and reuses it rather than re-reading half-updated states.
    std::vector<SceneObjectBox>    sib_boxes_;
    std::vector<std::string>       sib_ids_;                   // parallel to sib_boxes_ ("peninsula" for the island)
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

    // ★THESE CHECKS MATCH ON wall_seg_id, NOT ON A COORDINATE (fixed 2026-08-16). They used to select a run
    // by `std::abs(b.cy - 2.45f) < 0.2f`, i.e. by its FRONT-FACE coordinate — but KitchenBox.cx/cy is the box
    // CENTRE, half a depth further back (the north run's centre is 2.726, not 2.45). Every corner-fill check
    // below therefore matched NOTHING and the test reported "all checks passed" while asserting nothing at
    // all about the corner. That is how an expectation encoding the double-claimed corner survived in here.
    const auto box_on_wall = [&](int seg) -> const KitchenBox*
    { for (const auto& b : boxes) if (b.tier == 0 and b.wall_seg_id == seg) return &b; return nullptr; };
    const KitchenBox* west  = box_on_wall(0);
    const KitchenBox* north = box_on_wall(1);
    const KitchenBox* east  = box_on_wall(2);
    check(west and north and east, "one base run on each of the three walls");

    // CORNER OWNERSHIP, read off the evidence. At the NW vertex (0,3) the west run stops 0.4 m short and the
    // north run 0.55 m short, so WEST owns it and north is set back by west's depth; at the NE vertex (2,3)
    // north is the nearer and owns it, so EAST is set back by north's depth. Measured, steady state:
    // west t=[0.40, 3.00] (L 2.60, reaching its vertex), north t=[0.55, 2.00] (L 1.45), east t=[0.55, 2.60]
    // (L 2.05). Each vertex is reached by exactly one run and each corner volume is filled exactly once.
    if (north)
    {
        check(north->L > 1.35f and north->L < 1.65f,
              "north run keeps the NE vertex but stops at the west run's face (it does not take both corners)");
        check(north->d > 0.35f and north->d < 0.75f, "north run depth stays physical (near standard)");
        check(north->z1 < 1.0f, "north run top stays at the base worktop (no ceiling-touching)");
    }
    // A run with NO reaching neighbour at a corner must not fill to it: the west run's LOW end (0,0) has no
    // perpendicular run, so it stays at its observed start (~0.4) while its NW end fills TO the vertex.
    if (west)
        check(west->L > 2.5f and west->L < 2.7f, "west run fills its owned (NW) corner, not its free (0,0) end");
    if (east)
        check(east->L > 1.9f and east->L < 2.2f, "east run stops at the north run's face, not at the NE vertex");

    // ★NO OVERLAP AND NO HOLE ARE TWO HALVES OF ONE STATEMENT, so both are checked, at the two places where
    // they compete: just inside each shared vertex. Covered TWICE was the reported defect (both runs filling
    // to V); covered ZERO times is what a naive fix trades it for, and is what the corner-fill precision
    // sat at for 400 cycles before it was weighed properly. Exactly once is the whole requirement.
    const auto covered_by = [&](float px, float py)
    {
        int n = 0;
        for (const auto& b : boxes)
        {
            if (b.tier != 0) continue;
            const float c = std::cos(b.yaw), sn = std::sin(b.yaw);
            const float dx = px - b.cx, dy = py - b.cy;
            const float lx =  c * dx + sn * dy, ly = -sn * dx + c * dy;
            if (std::abs(lx) <= 0.5f * b.L and std::abs(ly) <= 0.5f * b.d) ++n;
        }
        return n;
    };
    check(covered_by(0.20f, 2.90f) == 1, "the NW inside corner is filled by exactly ONE run (not two, not none)");
    check(covered_by(1.80f, 2.90f) == 1, "the NE inside corner is filled by exactly ONE run (not two, not none)");

    // ★AND THE PROPERTY THE WHOLE CHANGE EXISTS FOR: NO TWO ACTIVE RUNS SHARE SPACE. Every check above is
    // about one run's length; none of them could have caught two runs occupying the same corner, which is why
    // the defect survived a self-test that passed. State the invariant directly.
    for (std::size_t bi = 0; bi < boxes.size(); ++bi)
        for (std::size_t bj = bi + 1; bj < boxes.size(); ++bj)
        {
            if (boxes[bi].tier != boxes[bj].tier) continue;      // different tiers stack by design
            const rc::geom::Footprint fi{boxes[bi].cx, boxes[bi].cy, boxes[bi].L, boxes[bi].d, boxes[bi].yaw};
            const rc::geom::Footprint fj{boxes[bj].cx, boxes[bj].cy, boxes[bj].L, boxes[bj].d, boxes[bj].yaw};
            check(rc::geom::overlap_ratio(fi, fj) < 0.02f, "no two same-tier runs overlap (the L corner is claimed once)");
        }

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

    // ★CELL-vs-CELL PENETRATION AWAY FROM A SHARED VERTEX. route_corner_fill only arbitrates a corner two
    // runs BOTH reach, and rc::exclusion cannot see this pair at all (foreign_claims drops same-prefix nodes).
    // In between sat a real gap, and the live room is the reason it is not hypothetical: a wall SEGMENT can
    // extend past the room corner, so a run's own chart legitimately covers its neighbour's body and nothing
    // stopped it growing there. Reproduced here by starting the north wall 0.8 m PAST the corner — the two
    // endpoints are then 0.8 m apart, far beyond corner_join_tol_m (0.10), so the corner machinery never
    // engages and only sibling non-penetration can answer.
    //
    // ⚠The term rides on ObjectExclusionPrecision, which DEFAULTS TO 0 (off) and is 800 in cabinet's
    // config.toml. Set here to match the live agent — a default-constructed params would test nothing and
    // pass, which is the same shape of trap as the front-face predicates above.
    {
        CabinetBeliefParams bp2 = bp;
        bp2.object_exclusion_precision = 800.0f;
        const Eigen::Vector2f I2(1.0f, 1.5f);
        const std::vector<KitchenWall> w2 = {
            make_kitchen_wall({0.0f, 0.0f}, {0.0f, 3.0f}, I2, 0),   // west
            make_kitchen_wall({-0.8f, 3.0f}, {2.0f, 3.0f}, I2, 1),  // north, starting 0.8 m PAST the corner
        };
        KitchenManager m7; m7.build(w2, tiers, tp, bp2, 2.6f);
        std::vector<Eigen::Vector3f> p2;
        push_front(p2, w2[0], 0.4f, 2.9f);   // west run driven almost INTO the corner
        push_front(p2, w2[1], 1.1f, 2.5f);   // north run over x ∈ [0.3, 1.7] — its chart reaches x = −0.8
        for (int it = 0; it < 20; ++it) m7.update(p2, mp, tmpl);
        // ★NOW DRIVE THE END WHERE NO POINT SUPPORTS IT — with the LEVEL-2 ARRANGEMENT END PRIOR, which is
        // the live mechanism that does exactly that. kitchen_metaconcept tells a run where to end so the
        // kitchen reads as one continuous surface; it knows nothing about which sibling already stands
        // there, and its prior enters at whatever information it declares. Here it aims the north run's low
        // end at t = 0.0 (x = −0.8), straight through the west run's body, with information matching the
        // model's own end precision. Without sibling non-penetration nothing at all opposes it.
        std::string north_id;   // looked up, never spelled out — a hardcoded cell id silently aims the
        for (const auto& c : m7.cells())                 // prior at nothing and the test then proves nothing
            if (c.active() and c.geom.wall_seg_id == 1 and c.geom.tier == 0) north_id = c.geom.id;
        check(not north_id.empty(), "the north run is found for the end-prior probe");
        for (int it = 0; it < 60; ++it)
        {
            m7.set_cell_end_prior(north_id, 0.0f, 800.0f, 0.0f, 0.0f);
            m7.update(p2, mp, tmpl);
        }
        const auto b7 = m7.active_boxes();
        float worst = 0.0f;
        for (std::size_t i = 0; i < b7.size(); ++i)
            for (std::size_t j = i + 1; j < b7.size(); ++j)
            {
                if (b7[i].tier != b7[j].tier) continue;
                worst = std::max(worst, rc::geom::overlap_ratio({b7[i].cx, b7[i].cy, b7[i].L, b7[i].d, b7[i].yaw},
                                                                {b7[j].cx, b7[j].cy, b7[j].L, b7[j].d, b7[j].yaw}));
            }
        check(b7.size() == 2, "both runs activate on overlapping wall segments");
        check(worst < 0.05f, "two runs whose CHARTS overlap still do not occupy the same space");

        // ★THE SECOND DRIVER: EVIDENCE, not a prior. Above, the end prior aimed a run through its neighbour;
        // here the run's OWN mask points do, which is the case the target clamp cannot touch and only the
        // fit's non-penetration term answers. Routing breaks a tie between two charts that both contain a
        // point by taking the FIRST cell, so listing the north wall first hands it the contested points and
        // its censored extent bound then grows it straight along the west run's body.
        const std::vector<KitchenWall> w3 = {
            make_kitchen_wall({-0.8f, 3.0f}, {2.0f, 3.0f}, I2, 0),  // north FIRST ⇒ it wins the contested points
            make_kitchen_wall({0.0f, 0.0f}, {0.0f, 3.0f}, I2, 1),   // west
        };
        KitchenManager m8; m8.build(w3, tiers, tp, bp2, 2.6f);
        std::vector<Eigen::Vector3f> p3;
        push_front(p3, w3[0], 0.4f, 2.5f);   // north evidence reaching x = −0.4, THROUGH the west run's body
        push_front(p3, w3[1], 0.4f, 2.9f);
        for (int it = 0; it < 80; ++it) m8.update(p3, mp, tmpl);
        const auto b8 = m8.active_boxes();
        float worst8 = 0.0f;
        for (std::size_t i = 0; i < b8.size(); ++i)
            for (std::size_t j = i + 1; j < b8.size(); ++j)
            {
                if (b8[i].tier != b8[j].tier) continue;
                worst8 = std::max(worst8, rc::geom::overlap_ratio({b8[i].cx, b8[i].cy, b8[i].L, b8[i].d, b8[i].yaw},
                                                                  {b8[j].cx, b8[j].cy, b8[j].L, b8[j].d, b8[j].yaw}));
            }
        check(worst8 < 0.05f, "a run's own mask evidence does not grow it through a sibling either");

        // ★THE ISLAND IS THE CASE ROUTING CANNOT SOLVE. Wall cells partition their evidence by hard argmax
        // over charts, so a point belongs to exactly one of them and neither can grow on the other's returns
        // — which is why the two cases above are answered by the target clamp alone. The island is fitted
        // from a DIFFERENT mask label through a SEPARATE call (update_peninsula), so no argmax ever compares it
        // with a wall run, and its chart is PCA-derived and free to lie anywhere. Nothing but the fit's
        // non-penetration term stands between a peninsula and the run it grows along.
        {
            const float ev_x0 = 0.30f, ev_x1 = 1.60f;   // the island's raw mask evidence, along +x
            KitchenManager m9; m9.build(walls, tiers, tp, bp2, 2.6f);
            std::vector<Eigen::Vector3f> island;   // a run along +x at y = 1.2, starting INSIDE the west run
            for (int i = 0; i <= 20; ++i)
            {
                const float x = ev_x0 + (ev_x1 - ev_x0) * static_cast<float>(i) / 20.0f;
                for (int j = 0; j <= 4; ++j) island.emplace_back(x, 1.20f, 0.80f * static_cast<float>(j) / 4.0f);
            }
            for (int it = 0; it < 80; ++it) { m9.update(pts, mp, tmpl); m9.update_peninsula(island, mp, tmpl); }
            const KitchenBox* pen = nullptr;
            for (const auto& b : m9.active_boxes()) if (b.wall_seg_id < 0) pen = &b;
            check(pen != nullptr, "the island activates (else the check below proves nothing)");
            // ★WHAT THIS ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. The island's near end is driven off its
            // own raw evidence, away from the west run — measured 0.300 → 0.367 — which is only possible if
            // update_peninsula is being handed the wall runs. Blind, it sits at exactly ev_x0 and its length is
            // exactly the evidence span.
            //
            // It does NOT assert the overlap reaches zero, because here it does not: ~0.18 m of
            // interpenetration survives. Both beliefs hold GENUINE mask evidence inside the contested volume
            // (the island's points really do start 0.25 m inside the west run's believed body), so the model
            // does what it says it does — no arbitration, the cost falls on whoever is wrong, and with
            // object_exclusion_precision and extent_precision both at 800 that cost is split. Making one
            // yield outright would mean weighting by support strength: a real change to the generative model,
            // and the user's call, not something to smuggle in behind a test tolerance.
            if (pen)
            {
                const float low = pen->cx - 0.5f * pen->L;
                check(low > ev_x0 + 0.01f,               // 1 cm ⇒ clear of float noise, vs the 6.7 cm measured
                      "the island is pushed off its own evidence, out of the wall run it started inside");
                check(pen->L < (ev_x1 - ev_x0),
                      "the island is SHORTER than its raw evidence span (it yielded, rather than growing)");
            }
        }
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

    // ★NO FREE-STANDING ISLAND (2026-08-16, the user's rule). This block used to assert the opposite — that a
    // cloud floating in the middle of the room activates "the 4th cabinet" — and that is precisely the object
    // the model no longer admits. Every cabinet abuts a WALL or another CABINET; a cloud that abuts neither
    // gets no run at all, at any angle. The same cloud, moved against a wall, is born exactly as before, so
    // what changed is the interpretation, not the machinery.
    const auto cloud_at = [](float cx, float cy)          // 1.2 m (x) × 0.6 m (y) box
    {
        std::vector<Eigen::Vector3f> v;
        for (int i = 0; i <= 24; ++i) for (int j = 0; j <= 12; ++j)
        {
            const float x = cx - 0.6f + 1.2f * static_cast<float>(i) / 24.0f;
            const float y = cy - 0.3f + 0.6f * static_cast<float>(j) / 12.0f;
            for (int k = 0; k <= 3; ++k) v.emplace_back(x, y, 0.80f * static_cast<float>(k) / 3.0f);
        }
        return v;
    };
    {
        KitchenManager m5; m5.build(walls, tiers, tp, bp, 2.6f);
        const auto floating = cloud_at(1.0f, 1.4f);       // nearest wall 0.4 m away — abuts nothing
        for (int it = 0; it < 40; ++it) { m5.update({}, mp, tmpl); m5.update_peninsula(floating, mp, tmpl); }
        check(m5.active_boxes().empty(), "a cloud abutting NOTHING activates no cabinet at all");
    }
    {
        KitchenManager m6b; m6b.build(walls, tiers, tp, bp, 2.6f);
        const auto against_wall = cloud_at(0.6f, 1.4f);   // same cloud, left edge ON the west wall (x = 0)
        for (int it = 0; it < 40; ++it) { m6b.update({}, mp, tmpl); m6b.update_peninsula(against_wall, mp, tmpl); }
        const auto ib = m6b.active_boxes();
        int peninsulas = 0, walls_on = 0;
        for (const auto& b : ib) (b.wall_seg_id < 0 ? peninsulas : walls_on)++;
        check(peninsulas == 1 and walls_on == 0, "the SAME cloud, against a wall, activates ONE peninsula run");
        for (const auto& b : ib) if (b.wall_seg_id < 0)
        {
            check(b.L > 0.9f and b.L < 1.6f, "peninsula length matches the cloud extent (~1.2 m)");
            check(b.anchored, "a run that reaches the graph is ALWAYS attached — there is no other kind now");
        }
        // and it persists through look-away, exactly as before
        for (int it = 0; it < 80; ++it) m6b.update_peninsula({}, mp, tmpl);
        int pen_after = 0; for (const auto& b : m6b.active_boxes()) if (b.wall_seg_id < 0) ++pen_after;
        check(pen_after == 1, "the peninsula PERSISTS through look-away");
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
        for (int it = 0; it < 40; ++it) { m6.update({}, mp, tmpl); m6.update_peninsula(pen, mp, tmpl); }
        int n_pen = 0; float cy = 0.0f, L = 0.0f;
        for (const auto& b : m6.active_boxes()) if (b.wall_seg_id < 0) { ++n_pen; cy = b.cy; L = b.L; }
        check(n_pen == 1, "a wall-adjacent peninsula activates ONE peninsula run");
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
        // ★THE SELF-DERIVED AXIS NOW ONLY ARISES WHEN ATTACHING TO A CABINET. This probe used to float its
        // cloud clear of every wall, which was the free-standing PCA branch — the branch the user's rule
        // deleted. A run whose axis comes from a WALL cannot exercise this at all: it publishes the polygon's
        // variance by construction, checked above. So the cloud hangs off the north RUN instead (0.55 m from
        // the north wall, too far for the wall branch, touching the run's face), which is a real peninsula
        // and still takes its axis from its own points.
        const auto peninsula_axis_var = [&](float tilt_rad) -> float {
            std::vector<Eigen::Vector3f> pen;
            const Eigen::Vector2f c(1.00f, 1.95f);                       // hangs off the north run, clear of walls
            const Eigen::Vector2f dir(std::sin(tilt_rad), std::cos(tilt_rad));   // tilt off +y
            const Eigen::Vector2f nrm(-dir.y(), dir.x());
            for (int i = 0; i <= 24; ++i)
            {
                const float t = -0.5f + 1.0f * static_cast<float>(i) / 24.0f;    // 1 m long
                for (int j = 0; j <= 4; ++j)                                     // front face, 0.6 m deep
                {
                    const Eigen::Vector2f p = c + t * dir + 0.30f * nrm;
                    pen.emplace_back(p.x(), p.y(), 0.80f * static_cast<float>(j) / 4.0f);
                }
            }
            KitchenManager m; m.build(walls, tiers, tp, bp, 2.6f);
            std::vector<Eigen::Vector3f> host;   // the north run it attaches to must exist first
            push_front(host, walls[1], 0.3f, 1.7f);
            for (int it = 0; it < 40; ++it) { m.update(host, mp, tmpl); m.update_peninsula(pen, mp, tmpl); }
            const auto b = m.active_boxes();
            const auto it = std::ranges::find_if(b, [](const KitchenBox& x) { return x.wall_seg_id < 0; });
            return it == b.end() ? -1.0f : it->var_yaw;
        };
        const float v_aligned  = peninsula_axis_var(0.0f);
        const float v_tilted   = peninsula_axis_var(8.0f * std::numbers::pi_v<float> / 180.0f);
        check(v_aligned >= 0.0f and v_tilted >= 0.0f, "(i) both peninsulas activated for the axis check");
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
