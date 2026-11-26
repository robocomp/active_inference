/*
 * common/standpoint/standpoint_oracle.h — can the robot STAND and TURN here?  ONE answer, shared by the controller and every
 * agent that offers it a standpoint (K3, 2026-11-26).
 *
 * ── THE QUANTITY ─────────────────────────────────────────────────────────────────────────────────────────────────────────
 *     P(stand and turn at p) = P(every lattice cell the body covers, over the 8 headings it sweeps in a turn in place, is free)
 *                            = exp( −Σ_{c ∈ ∪_h body(p, h)} risk(c) )        and 0 if a HARD cell is under any heading.
 *   · risk(c) = −log Pr(θ_c < ½ | residual's Beta) for a residual cell with a belief behind it (residual_risk.h cell_risk_nats):
 *     the surprisal of finding the cell mostly free. Covering it is a bet on that, and the cells are independent in residual's
 *     model, so the nats add. A cell residual has collapsed (explained by a modelled object) costs 0. Unlatched ground costs 0.
 *   · HARD (P = 0): a cell outside the room (and outside every passage), inside an object polygon, a residual cell with no
 *     belief behind it, or one a live LiDAR return lands in this cycle (residual_risk.h harden_corroborated) — the last only
 *     when the caller does not ask for `persistent_only` (a cause that comes and goes).
 *     The no-belief case is a conservative default, not a belief (see "CONTINUOUS, NO THRESHOLD" below).
 *   · The body is the robot's own footprint (RobotFootprint, a robot property) grown by the caller's margin, rasterised per
 *     heading exactly as GridPlanner always has; a footprint leaving the lattice is hard ("off the map == unsafe").
 *   This is EXACTLY GridPlanner::fit_nats_turning (and, at one heading, fit_nats_some_heading / state_risk), which now calls
 *   the kernel below; the raster is built by rasterise() below, which GridPlanner::set_world now calls. So "the controller's
 *   occupancy belief" and "the oracle's" are one piece of code, not two that agree.
 *
 * ── CONTINUOUS, NO THRESHOLD ─────────────────────────────────────────────────────────────────────────────────────────────
 *   The probability has no cut-off and no constant of its own: the ½ in risk(c) is the Bernoulli rate's symmetric decision
 *   boundary (residual_risk.h), the footprint is the robot's, and the margin and lattice are the CALLER's (the controller passes
 *   its own; an agent that cannot read them passes margin 0 and its data's own resolution — see "Who calls it").
 *   HARD cells give exactly 0, and they are of TWO kinds, which must not be confused:
 *     · BELIEF-BACKED hard: a cell a live LiDAR return lands in this cycle (a measurement of contact), an object polygon (a modelled
 *       object, treated as certain), outside the room polygon (the room's wall). Here 0 is a statement the model makes.
 *     · NO BELIEF: a residual cell with no usable Beta behind it (the field is absent, does not cover the cell, or its moments are
 *       unusable). Here 0 is NOT a statement of the belief — it is the ABSENCE of one, mapped to hard by the conservative
 *       convention inherited from residual_risk.h ("keep the old behaviour exactly where we have nothing better to say").
 *       ⚠ Consequence: if the consumer cannot read residual's field (rc::read_residual_field fails, e.g. the grid node lacks
 *       grid_occupancy_prob / _var / _meta for a cycle), EVERY latched residual cell becomes hard, and every standpoint with a
 *       latched cell anywhere under its turn gets P = 0 — in the controller (hand-back) and in the countables (offer × 0) alike.
 *
 * ── WHERE A BINARY DECISION IS NEEDED, THE CONSUMER MAKES IT ──────────────────────────────────────────────────────────────
 *   The controller hands a standpoint back iff P = 0 (standpoint_fit.h verdict: isfinite(nats)); pose_free / can_turn_here are
 *   P > 0 at one / all headings. Those decisions are the controller's and are unchanged (controller/tests/
 *   test_standpoint_oracle_parity.cpp: golden recorded at BASE aea54896). An object agent multiplies its offer by P (G6).
 *
 * ── WHO CALLS IT ─────────────────────────────────────────────────────────────────────────────────────────────────────────
 *   · controller: GridPlanner (rasterise + the kernel on its own members), cell = Planner.CellSize, margin =
 *     Controller.FootprintSafetyMarginM, plus passages, live LiDAR hardening, its obstacle polygons.
 *   · countables (G6 p_stand): Oracle below, on the room polygon, their obstacle boxes and residual's published cells priced by
 *     the published field — the same attribute and the same pricing function the controller reads. Cell = residual's
 *     grid_cell_size, margin 0 (the controller's margin is its config, not readable by another agent). The differences are
 *     listed in the K3 report and in the countable call site; they are inputs, not a second model.
 *
 * Pure Eigen/std + robot_footprint. Tested in common/standpoint/tests/standpoint_oracle_test.cpp.
 */
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "../robot_footprint/robot_footprint.h"
#include "residual_risk.h"

namespace rc
{

// residual_concept's occupancy as the planner takes it: cell centres in the room frame at residual's own resolution.
// (Moved here from controller/src/grid_planner.h in K3, unchanged.)
struct OccupiedCells
{
    std::vector<Eigen::Vector2f> centres;
    float cell_size_m = 0.f;                    // 0 ⇒ no grid supplied, nothing marked
    // ── HOW MUCH EACH CELL COSTS, FROM RESIDUAL'S OWN BELIEF (2026-10-19, try96 leg 3) ─────────────
    // One entry per centre, in NATS: −log Pr(cell mostly free | residual's Beta), see residual_risk.h.
    // A finite entry makes the cell a TOLL — the A* pays residual_risk_weight per nat the body covers —
    // instead of a wall; +inf keeps it HARD (no belief behind it, or a live LiDAR return in it).
    // ★EMPTY (or the wrong length) ⇒ EVERY cell is hard, exactly as before: a caller that has no belief
    // to offer gets the conservative planner, never a permissive one.
    std::vector<float> risk_nats;
    // ── Task 3c fix round 1: WHICH HARD CELLS COME AND GO ─────────────────────────────────────────────
    // One entry per centre (or empty): 1 = the cell is hard ONLY because a live return lands in it this cycle
    // (residual_risk.h harden_corroborated; it had a finite price before). GridPlanner keeps such cells out of its
    // PERSISTENT occupancy (fit_nats_* with persistent_only), so a standpoint verdict can tell a wall from a person
    // walking past. occ_ — routing, pose_free, the distance map — is unchanged: they stay hard there.
    std::vector<std::uint8_t> live_hard;
};

namespace standpoint
{

inline constexpr int kHeadings = 8;                      // GridPlanner's heading buckets: one per 45°
using Offsets = std::vector<std::vector<Eigen::Vector2i>>;   // kHeadings entries: the body's covered cells per heading

namespace detail
{
inline bool point_in_polygon(std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& p)
{
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        if (((poly[i].y() > p.y()) != (poly[j].y() > p.y())) and
            (p.x() < (poly[j].x() - poly[i].x()) * (p.y() - poly[i].y()) / (poly[j].y() - poly[i].y()) + poly[i].x()))
            inside = not inside;
    return inside;
}
}   // namespace detail

// Room YAW (forward = (cos, sin)) → heading bucket, as GridPlanner::heading_bucket.
inline int heading_bucket(float yaw)
{
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    float t = std::fmod(yaw, two_pi);
    if (t < 0) t += two_pi;
    return static_cast<int>(std::lround(t / two_pi * kHeadings)) % kHeadings;
}

// The body's covered-cell offsets at each heading, for a lattice of `cell` m, the footprint grown by `margin` m. The quarter
// turn from a room yaw to the footprint's own theta is applied once, inside RobotFootprint::cell_offsets_yaw.
inline Offsets body_offsets(RobotFootprint body, float cell, float margin)
{
    body.set_safety_margin(margin);
    Offsets o(kHeadings);
    for (int h = 0; h < kHeadings; ++h)
        o[static_cast<std::size_t>(h)] = body.cell_offsets_yaw(cell, 2.0f * static_cast<float>(M_PI) * h / kHeadings);
    return o;
}

// A residual cell's source square (true extent), kept for the exact-geometry turn sweep.
struct SourceSquare { Eigen::Vector2f c; float half = 0.f; };

// ── THE RASTER ───────────────────────────────────────────────────────────────────────────────────────────────────────────
struct Raster
{
    float xmin = 0.f, ymin = 0.f, cell = 0.1f;
    int   w = 0, h = 0;
    std::vector<std::uint8_t> occ;           // 1 = hard
    std::vector<std::uint8_t> occ_nocells;   // occ before the residual stamp (room mask + polygons), for the sweep
    std::vector<std::uint8_t> live_only;     // 1 = hard only through a live-hard residual cell (or empty: none)
    std::vector<float>        risk;          // soft nats per cell (or empty: none)
    std::vector<SourceSquare> src_squares;   // hard residual cells at their true extent
    int  soft_in = 0, hard_in = 0;           // residual cells handed in as soft / hard
    bool room_mask_usable = true;            // false ⇒ the room polygon was discarded (marked ≥ 95 % outside)
    long outside = 0;                        // cells the room mask marked outside (before any discard)
    bool valid() const { return w > 0 and h > 0; }
};

// Read-only view: what the kernel needs. GridPlanner builds one over its own members; Raster gives one of itself.
struct RasterView
{
    float xmin = 0.f, ymin = 0.f, cell = 0.1f;
    int   w = 0, h = 0;
    std::span<const std::uint8_t> occ, live_only;
    std::span<const float>        risk;

    bool in_bounds(int ix, int iy) const { return ix >= 0 and ix < w and iy >= 0 and iy < h; }
    int  idx(int ix, int iy) const { return iy * w + ix; }
    bool world_to_cell(const Eigen::Vector2f& p, int& ix, int& iy) const
    {
        ix = static_cast<int>(std::floor((p.x() - xmin) / cell));
        iy = static_cast<int>(std::floor((p.y() - ymin) / cell));
        return in_bounds(ix, iy);
    }
};
inline RasterView view(const Raster& r) { return {r.xmin, r.ymin, r.cell, r.w, r.h, r.occ, r.live_only, r.risk}; }

inline Eigen::Vector2f cell_centre(float xmin, float ymin, float cell, int ix, int iy)
{
    return {xmin + (static_cast<float>(ix) + 0.5f) * cell, ymin + (static_cast<float>(iy) + 0.5f) * cell};
}

inline bool inside_room_or_passage(std::span<const Eigen::Vector2f> room, std::span<const std::vector<Eigen::Vector2f>> passages,
                                   const Eigen::Vector2f& p)
{
    if (detail::point_in_polygon(room, p)) return true;
    return std::ranges::any_of(passages, [&](const auto& pass) { return pass.size() >= 3 and detail::point_in_polygon(pass, p); });
}

// Rasterise the world: OUTSIDE the room (and every passage) is hard; obstacle polygons at their true extent (a cell whose centre
// is inside); residual cells — hard ones stamped over every lattice cell their square touches (outward), soft ones adding their
// nats to the one lattice cell holding their centre, live-hard ones stamped hard and flagged live-only where nothing persistent
// is. `room` must have ≥ 3 vertices. `pad_m` grows the lattice beyond the room/passage bbox so a body near the boundary is
// representable (GridPlanner: circumscribed radius + margin + cell). This is GridPlanner::set_world's rasterisation, moved.
inline Raster rasterise(std::span<const Eigen::Vector2f> room, std::span<const std::vector<Eigen::Vector2f>> passages,
                        std::span<const std::vector<Eigen::Vector2f>> obstacles, const OccupiedCells& cells, float cell, float pad_m)
{
    Raster r;
    r.cell = cell;
    if (room.size() < 3) return r;
    float xmn = room[0].x(), xmx = xmn, ymn = room[0].y(), ymx = ymn;
    for (const auto& p : room)
    { xmn = std::min(xmn, p.x()); xmx = std::max(xmx, p.x()); ymn = std::min(ymn, p.y()); ymx = std::max(ymx, p.y()); }
    for (const auto& pass : passages)
        for (const auto& p : pass)
        { xmn = std::min(xmn, p.x()); xmx = std::max(xmx, p.x()); ymn = std::min(ymn, p.y()); ymx = std::max(ymx, p.y()); }
    r.xmin = xmn - pad_m; r.ymin = ymn - pad_m;
    r.w = std::max(1, static_cast<int>(std::ceil((xmx + pad_m - r.xmin) / cell)));
    r.h = std::max(1, static_cast<int>(std::ceil((ymx + pad_m - r.ymin) / cell)));
    const int W = r.w, H = r.h;
    r.occ.assign(static_cast<std::size_t>(W) * H, 0);
    const RasterView v{r.xmin, r.ymin, r.cell, W, H, {}, {}, {}};
    const auto to_cell = [&](const Eigen::Vector2f& p, int& ix, int& iy) { return v.world_to_cell(p, ix, iy); };
    const auto centre = [&](int ix, int iy) { return cell_centre(r.xmin, r.ymin, r.cell, ix, iy); };
    const auto at = [&](int ix, int iy) -> std::uint8_t& { return r.occ[static_cast<std::size_t>(iy * W + ix)]; };

    for (int iy = 0; iy < H; ++iy)
        for (int ix = 0; ix < W; ++ix)
            if (not inside_room_or_passage(room, passages, centre(ix, iy)))
            { at(ix, iy) = 1; ++r.outside; }
    // ⚠ THRESHOLD (pre-existing, moved unchanged from GridPlanner::set_world in K3; flagged, not justified by a model term).
    //   What it guards: a room polygon that marks >= 95 % of the padded lattice outside is taken to be broken (live: 27018 of 27018
    //   cells once) and the mask is DISCARDED — planning continues on obstacles alone instead of every pose reading infeasible and
    //   the robot never moving again. Why kept for now: it is the controller's degraded-mode guard and K3 must not change the
    //   controller's decisions (golden parity); the model-level replacement would be a belief over the room polygon's validity
    //   (room_concept's own fit quality), which no consumer has yet. Note it now applies to every Oracle user too.
    r.room_mask_usable = r.outside < static_cast<long>(0.95 * W * H);
    if (not r.room_mask_usable) std::fill(r.occ.begin(), r.occ.end(), 0);

    for (const auto& poly : obstacles)
    {
        if (poly.size() < 3) continue;
        float pxn = poly[0].x(), pxx = pxn, pyn = poly[0].y(), pyx = pyn;
        for (const auto& p : poly)
        { pxn = std::min(pxn, p.x()); pxx = std::max(pxx, p.x()); pyn = std::min(pyn, p.y()); pyx = std::max(pyx, p.y()); }
        int ix0, iy0, ix1, iy1;
        to_cell({pxn, pyn}, ix0, iy0);
        to_cell({pxx, pyx}, ix1, iy1);
        for (int iy = std::max(0, iy0); iy <= std::min(H - 1, iy1); ++iy)
            for (int ix = std::max(0, ix0); ix <= std::min(W - 1, ix1); ++ix)
                if (not at(ix, iy) and detail::point_in_polygon(poly, centre(ix, iy)))
                    at(ix, iy) = 1;
    }
    r.occ_nocells = r.occ;
    if (cells.cell_size_m > 0.f and not cells.centres.empty())
    {
        const float hh = 0.5f * cells.cell_size_m;
        const bool priced = cells.risk_nats.size() == cells.centres.size();
        const bool flagged = cells.live_hard.size() == cells.centres.size();
        std::vector<std::size_t> live_cells;
        for (std::size_t k = 0; k < cells.centres.size(); ++k)
        {
            const auto& c = cells.centres[k];
            if (flagged and cells.live_hard[k] and not (priced and std::isfinite(cells.risk_nats[k])))
            { live_cells.push_back(k); continue; }
            if (priced and std::isfinite(cells.risk_nats[k]))
            {
                ++r.soft_in;
                int ix, iy;
                if (cells.risk_nats[k] <= 0.f or not to_cell(c, ix, iy)) continue;
                if (r.risk.empty()) r.risk.assign(static_cast<std::size_t>(W) * H, 0.f);
                r.risk[static_cast<std::size_t>(iy * W + ix)] += cells.risk_nats[k];
                continue;
            }
            ++r.hard_in;
            r.src_squares.push_back({c, hh});
            int ix0, iy0, ix1, iy1;
            to_cell({c.x() - hh, c.y() - hh}, ix0, iy0);
            to_cell({c.x() + hh, c.y() + hh}, ix1, iy1);
            for (int iy = std::max(0, iy0); iy <= std::min(H - 1, iy1); ++iy)
                for (int ix = std::max(0, ix0); ix <= std::min(W - 1, ix1); ++ix)
                    at(ix, iy) = 1;
        }
        if (not live_cells.empty()) r.live_only.assign(static_cast<std::size_t>(W) * H, 0);
        for (const std::size_t k : live_cells)
        {
            ++r.hard_in;
            const auto& c = cells.centres[k];
            r.src_squares.push_back({c, hh});
            int ix0, iy0, ix1, iy1;
            to_cell({c.x() - hh, c.y() - hh}, ix0, iy0);
            to_cell({c.x() + hh, c.y() + hh}, ix1, iy1);
            for (int iy = std::max(0, iy0); iy <= std::min(H - 1, iy1); ++iy)
                for (int ix = std::max(0, ix0); ix <= std::min(W - 1, ix1); ++ix)
                    if (not at(ix, iy)) { at(ix, iy) = 1; r.live_only[static_cast<std::size_t>(iy * W + ix)] = 1; }
        }
    }
    return r;
}

// ── THE KERNEL ───────────────────────────────────────────────────────────────────────────────────────────────────────────
// Is the body at heading h, centred on lattice cell (ix, iy), clear of every hard cell and inside the lattice? persistent_only:
// a cell hard only through a live return does not count (GridPlanner::cell_free_persistent; otherwise cell_free).
inline bool body_free(const RasterView& v, const Offsets& off, int ix, int iy, int h, bool persistent_only)
{
    if (off.size() != kHeadings) return false;
    for (const auto& o : off[static_cast<std::size_t>(h)])
    {
        const int nx = ix + o.x(), ny = iy + o.y();
        if (not v.in_bounds(nx, ny)) return false;           // footprint off the map == unsafe
        const auto i = static_cast<std::size_t>(v.idx(nx, ny));
        if (v.occ[i] and not (persistent_only and i < v.live_only.size() and v.live_only[i])) return false;
    }
    return true;
}

// Soft nats under the body at heading h (GridPlanner::state_risk). 0 with no soft layer.
inline float body_nats(const RasterView& v, const Offsets& off, int ix, int iy, int h)
{
    if (v.risk.empty() or off.size() != kHeadings) return 0.f;
    float r = 0.f;
    for (const auto& o : off[static_cast<std::size_t>(h)])
    {
        const int nx = ix + o.x(), ny = iy + o.y();
        if (v.in_bounds(nx, ny)) r += v.risk[static_cast<std::size_t>(v.idx(nx, ny))];
    }
    return r;
}

// −log P(the body fits standing at ONE heading, the cheapest hard-free one). +inf when none is hard-free or p is off the lattice.
inline float fit_nats_some_heading(const RasterView& v, const Offsets& off, const Eigen::Vector2f& p, bool persistent_only)
{
    constexpr float kInf = std::numeric_limits<float>::infinity();
    int ix, iy;
    if (not v.world_to_cell(p, ix, iy)) return kInf;
    float best = kInf;
    for (int h = 0; h < kHeadings; ++h)
        if (body_free(v, off, ix, iy, h, persistent_only)) best = std::min(best, body_nats(v, off, ix, iy, h));
    return best;
}

// −log P(the body fits TURNING IN PLACE through every heading): +inf unless hard-free at all eight, else the soft nats of the
// UNION of the eight footprints, each cell counted once (a turn covers it once).
inline float fit_nats_turning(const RasterView& v, const Offsets& off, const Eigen::Vector2f& p, bool persistent_only)
{
    constexpr float kInf = std::numeric_limits<float>::infinity();
    int ix, iy;
    if (not v.world_to_cell(p, ix, iy)) return kInf;
    for (int h = 0; h < kHeadings; ++h)
        if (not body_free(v, off, ix, iy, h, persistent_only)) return kInf;
    if (v.risk.empty()) return 0.f;
    std::unordered_set<int> covered;
    float r = 0.f;
    for (int h = 0; h < kHeadings; ++h)
        for (const auto& o : off[static_cast<std::size_t>(h)])
        {
            const int nx = ix + o.x(), ny = iy + o.y();
            if (v.in_bounds(nx, ny) and covered.insert(v.idx(nx, ny)).second) r += v.risk[static_cast<std::size_t>(v.idx(nx, ny))];
        }
    return r;
}

// ── THE ORACLE, FOR AN AGENT THAT HOLDS NO PLANNER ────────────────────────────────────────────────────────────────────────
// The body, the lattice and the margin are fixed at construction; set_world rasterises (skipped when its inputs are bit-identical
// to the last call's). Before any world (or with a room of < 3 vertices) there is no belief, and p_* answer 1: no veto — the
// controller's "no belief yet: no verdict".
class Oracle
{
public:
    Oracle() = default;
    Oracle(const RobotFootprint& body, float cell_m, float margin_m) { configure(body, cell_m, margin_m); }

    void configure(const RobotFootprint& body, float cell_m, float margin_m)
    {
        body_ = body;
        body_.set_safety_margin(margin_m);
        margin_ = std::max(0.f, margin_m);
        cell_ = cell_m;
        off_ = cell_m > 0.f ? body_offsets(body_, cell_m, margin_) : Offsets{};
        key_ = 0;
        raster_ = {};
    }
    bool configured() const { return cell_ > 0.f and off_.size() == kHeadings; }
    float cell() const { return cell_; }
    float margin() const { return margin_; }

    // The lattice pad is GridPlanner's: the body's circumscribed radius (margin included) + the margin + one cell.
    void set_world(std::span<const Eigen::Vector2f> room, std::span<const std::vector<Eigen::Vector2f>> passages,
                   std::span<const std::vector<Eigen::Vector2f>> obstacles, const OccupiedCells& cells)
    {
        if (not configured()) return;
        std::size_t k = 1469598103934665603ull;
        const auto mix = [&k](std::uint32_t x) { k ^= x; k *= 1099511628211ull; };
        const auto mixf = [&](float f) { mix(std::bit_cast<std::uint32_t>(f)); };
        const auto mixpoly = [&](std::span<const Eigen::Vector2f> poly)
        { mix(static_cast<std::uint32_t>(poly.size())); for (const auto& p : poly) { mixf(p.x()); mixf(p.y()); } };
        mixpoly(room);
        mix(static_cast<std::uint32_t>(passages.size())); for (const auto& p : passages) mixpoly(p);
        mix(static_cast<std::uint32_t>(obstacles.size())); for (const auto& p : obstacles) mixpoly(p);
        mixf(cells.cell_size_m); mixpoly(cells.centres);
        mix(static_cast<std::uint32_t>(cells.risk_nats.size())); for (const float r : cells.risk_nats) mixf(r);
        mix(static_cast<std::uint32_t>(cells.live_hard.size())); for (const auto x : cells.live_hard) mix(x);
        if (k == key_) return;
        key_ = k;
        raster_ = rasterise(room, passages, obstacles, cells, cell_, body_.circumscribed_radius() + margin_ + cell_);
    }
    bool has_world() const { return raster_.valid(); }
    const Raster& raster() const { return raster_; }
    const Offsets& offsets() const { return off_; }

    float nats_turning(const Eigen::Vector2f& p, bool persistent_only = true) const
    { return has_world() ? fit_nats_turning(view(raster_), off_, p, persistent_only) : 0.f; }
    float nats_at(const Eigen::Vector2f& p, float yaw, bool persistent_only = true) const
    {
        if (not has_world()) return 0.f;
        const RasterView v = view(raster_);
        int ix, iy;
        if (not v.world_to_cell(p, ix, iy)) return std::numeric_limits<float>::infinity();
        const int h = heading_bucket(yaw);
        return body_free(v, off_, ix, iy, h, persistent_only) ? body_nats(v, off_, ix, iy, h) : std::numeric_limits<float>::infinity();
    }
    // P(stand at p and turn in place through every heading) — the G6 factor for an offer that ends in a turn.
    double p_stand_and_turn(const Eigen::Vector2f& p, bool persistent_only = true) const
    { return std::exp(-static_cast<double>(nats_turning(p, persistent_only))); }
    // P(stand at p facing yaw), one heading.
    double p_stand(const Eigen::Vector2f& p, float yaw, bool persistent_only = true) const
    { return std::exp(-static_cast<double>(nats_at(p, yaw, persistent_only))); }

private:
    RobotFootprint body_ = RobotFootprint::shadow();
    float cell_ = 0.f, margin_ = 0.f;
    Offsets off_;
    Raster raster_;
    std::size_t key_ = 0;
};

// An oriented box (centre, extent w along R(yaw)·x̂ and h along R(yaw)·ŷ) as the polygon the rasteriser fills — the object boxes
// an agent reads from the graph (rc::nbv::Obstacle), at their true extent like every other obstacle.
inline std::vector<Eigen::Vector2f> box_polygon(float cx, float cy, float w, float h, float yaw)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    std::vector<Eigen::Vector2f> p;
    p.reserve(4);
    for (const auto& [u, v] : {std::pair{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}})
        p.emplace_back(cx + c * u * w - s * v * h, cy + s * u * w + c * v * h);
    return p;
}

// residual's published cells (grid_occupied_cells: flat [x,y,z] triples) priced by its published belief field with the
// controller's own function (residual_risk::risks_from_field). An invalid field leaves every cell hard, as in the controller.
inline OccupiedCells cells_from_xyz(std::span<const float> xyz, float cell_m, const residual_risk::BeliefField& field)
{
    OccupiedCells c;
    c.cell_size_m = cell_m > 0.f ? cell_m : 0.f;
    c.centres.reserve(xyz.size() / 3);
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) c.centres.emplace_back(xyz[i], xyz[i + 1]);
    c.risk_nats = residual_risk::risks_from_field(c.centres, field);
    return c;
}

}   // namespace standpoint
}   // namespace rc
