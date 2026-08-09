/*
 * cabinet_kitchen_cells.h — Stage 0 of the kitchen-of-runs model (Fable design): the finite (wall_id, tier)
 * CELL TABLE + soft per-point ROUTING. INSTRUMENT-ONLY shadow — it does NOT touch the belief/fit; it just
 * builds the cells from the trusted room polygon and softly assigns each mask point to a cell so we can see
 * the 3 base + 1 upper cells light up before any behaviour changes (the cheap-abort validation point).
 *
 * A cabinet run is an INTERVAL on a wall, not a free box. A cell = one wall segment × one tier {base, upper}.
 * Routing = a soft responsibility per point from: lateral distance INTO the carcass footprint [0, depth] off
 * the wall, the tier z-band, and the along-wall span [0, W] — all as soft boxes (Gaussian tails), normalised
 * across cells + a clutter component. Cells are keyed by a GEOMETRIC SIGNATURE (quantised wall midpoint +
 * direction), stable if the polygon republishes with a different vertex order.
 *
 * Pure geometry, no DSR. Validated by kitchen_cells_self_test().
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace rc {

// One collinear-merged room wall (the caller builds these from CabinetFitter::rebuild_wall_ids/build_wall_ref).
struct KitchenWall
{
    Eigen::Vector2f a{0, 0}, b{0, 0};   // segment corners (room frame)
    Eigen::Vector2f u{1, 0}, n{0, 1};   // unit direction a→b, INWARD unit normal (into the room)
    float           W = 0.0f;           // length
    int             seg_id = -1;        // canonical wall id (diagnostic)
};

// A carcass tier: its z-band and standard depth off the wall (tight-prior values, used only for routing here).
struct KitchenTier { float z_lo, z_hi, depth; const char* name; };

struct KitchenCell
{
    std::string     id;                 // geometric signature (vertex-order-stable)
    int             wall_seg_id = -1;
    int             tier = 0;           // 0 = base, 1 = upper
    Eigen::Vector2f a{0, 0}, u{1, 0}, n{0, 1};
    float           W = 0.0f;
    float           z_lo = 0.0f, z_hi = 0.0f, depth = 0.0f;
    double          mass = 0.0;         // routed evidence mass this frame (shadow diagnostic)
};

struct KitchenRouting
{
    std::vector<KitchenCell> cells;
    double                   clutter_mass = 0.0;

    // Vertex-order-stable id: wall midpoint quantised to 10 cm + direction to ~5°, plus the tier.
    static std::string signature(const KitchenWall& w, int tier)
    {
        const Eigen::Vector2f m = 0.5f * (w.a + w.b);
        const auto q = [](float x) { return std::lround(x / 0.10f); };
        int dq = static_cast<int>(std::lround(std::atan2(w.u.y(), w.u.x()) / 0.0873f));   // ~5° bins
        dq = ((dq % 72) + 72) % 72;                                                       // fold to [0,72)
        // a wall and its reverse are the SAME wall ⇒ fold direction modulo π (36 bins) so the sig is invariant.
        dq %= 36;
        char buf[80];
        std::snprintf(buf, sizeof buf, "w%ld_%ld_d%d_t%d", q(m.x()), q(m.y()), dq, tier);
        return buf;
    }

    void build(const std::vector<KitchenWall>& walls, const KitchenTier tiers[2])
    {
        cells.clear();
        cells.reserve(walls.size() * 2);
        for (const auto& w : walls)
            for (int t = 0; t < 2; ++t)
            {
                KitchenCell c;
                c.id = signature(w, t); c.wall_seg_id = w.seg_id; c.tier = t;
                c.a = w.a; c.u = w.u; c.n = w.n; c.W = w.W;
                c.z_lo = tiers[t].z_lo; c.z_hi = tiers[t].z_hi; c.depth = tiers[t].depth;
                cells.push_back(c);
            }
    }

    // Soft-route points into the cells. sigma_lat/sigma_z ≈ mask coarseness + localization (fold chain σ in
    // when live). Fills each cell.mass and clutter_mass with the normalised responsibility sum.
    void route(const std::vector<Eigen::Vector3f>& pts, float sigma_lat, float sigma_z, float clutter)
    {
        for (auto& c : cells) c.mass = 0.0;
        clutter_mass = 0.0;
        const float il2 = 1.0f / std::max(1e-6f, sigma_lat * sigma_lat);
        const float iz2 = 1.0f / std::max(1e-6f, sigma_z * sigma_z);
        std::vector<double> w(cells.size());
        for (const auto& p : pts)
        {
            const Eigen::Vector2f q(p.x(), p.y());
            double sum = clutter;
            for (std::size_t k = 0; k < cells.size(); ++k)
            {
                const auto& c = cells[k];
                const float t   = (q - c.a).dot(c.u);           // along the wall
                const float lat = (q - c.a).dot(c.n);           // inward from the wall line
                const float da  = std::max({0.0f, -t, t - c.W});           // outside [0, W]
                const float dl  = std::max({0.0f, -lat, lat - c.depth});   // outside the footprint [0, depth]
                const float dz  = std::max({0.0f, c.z_lo - p.z(), p.z() - c.z_hi});
                const double wt = std::exp(-0.5 * ((da * da + dl * dl) * il2 + (dz * dz) * iz2));
                w[k] = wt; sum += wt;
            }
            const double inv = 1.0 / std::max(1e-12, sum);
            for (std::size_t k = 0; k < cells.size(); ++k) cells[k].mass += w[k] * inv;
            clutter_mass += clutter * inv;
        }
    }

    double tier_mass(int tier) const
    { double s = 0.0; for (const auto& c : cells) if (c.tier == tier) s += c.mass; return s; }
    int active_cells(int tier, double min_mass) const
    { int n = 0; for (const auto& c : cells) if (c.tier == tier and c.mass >= min_mass) ++n; return n; }
};

inline KitchenWall make_kitchen_wall(Eigen::Vector2f a, Eigen::Vector2f b, Eigen::Vector2f interior, int id)
{
    KitchenWall w; w.a = a; w.b = b; w.W = (b - a).norm();
    w.u = (b - a).normalized();
    Eigen::Vector2f nrm(-w.u.y(), w.u.x());
    if (nrm.dot(interior - 0.5f * (a + b)) < 0.0f) nrm = -nrm;   // point INTO the room
    w.n = nrm; w.seg_id = id; return w;
}

inline bool kitchen_cells_self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* what)
    { if (not c) { std::printf("[kitchen_cells::self_test] FAIL: %s\n", what); ok = false; } };

    // A U-kitchen: west, north, east walls; interior toward the centre.
    const Eigen::Vector2f I(1.0f, 1.5f);
    const std::vector<KitchenWall> walls = {
        make_kitchen_wall({0, 0}, {0, 3}, I, 0),    // west  (u=+y, n=+x)
        make_kitchen_wall({0, 3}, {2, 3}, I, 1),    // north (u=+x, n=-y)
        make_kitchen_wall({2, 3}, {2, 0}, I, 2),    // east  (u=-y, n=-x)
    };
    const KitchenTier tiers[2] = { {0.0f, 0.87f, 0.55f, "base"}, {1.40f, 2.10f, 0.35f, "upper"} };
    KitchenRouting R; R.build(walls, tiers);
    check(R.cells.size() == 6, "6 cells (3 walls × 2 tiers)");

    // Front-face point clouds for the three BASE runs (a run's footprint spans lat ∈ [0, depth]; sample the
    // front face at lat ≈ depth, over the wall span, up the base z-band).
    std::vector<Eigen::Vector3f> pts;
    const auto front = [&](const KitchenWall& w, float depth) {
        for (int i = 0; i <= 20; ++i) { const float t = w.W * i / 20.0f;
            const Eigen::Vector2f p = w.a + t * w.u + depth * w.n;
            for (int j = 0; j <= 4; ++j) pts.emplace_back(p.x(), p.y(), 0.80f * j / 4.0f); } };
    front(walls[0], 0.55f); front(walls[1], 0.55f); front(walls[2], 0.55f);
    const std::size_t N = pts.size();

    R.route(pts, 0.10f, 0.15f, 0.05f);
    check(R.tier_mass(0) / std::max(1.0, R.tier_mass(0) + R.tier_mass(1) + R.clutter_mass) > 0.90,
          "≥90% of base front-face mass routes to BASE cells");
    check(R.tier_mass(1) < 0.05 * N, "almost no mass leaks to UPPER cells");
    check(R.active_cells(0, 0.12 * (N / 3.0)) == 3, "all 3 base walls light up");

    // Corner wedge: a point near the NW corner (west∩north) must SPLIT between the two adjacent base cells.
    {
        KitchenRouting Rc; Rc.build(walls, tiers);
        Rc.route({{0.30f, 2.70f, 0.4f}}, 0.10f, 0.15f, 0.05f);
        double west = 0, north = 0;
        for (const auto& c : Rc.cells) if (c.tier == 0) { if (c.wall_seg_id == 0) west = c.mass; if (c.wall_seg_id == 1) north = c.mass; }
        check(west > 0.15 and north > 0.15, "a corner-wedge point SPLITS between the two adjacent base cells");
    }

    // Mislocalization: shift ALL points 20 cm in +x → routing softens (more clutter) but does NOT collapse.
    {
        std::vector<Eigen::Vector3f> shifted;
        shifted.reserve(pts.size());
        for (const auto& p : pts) shifted.emplace_back(p.x() + 0.20f, p.y(), p.z());
        KitchenRouting Rm; Rm.build(walls, tiers);
        Rm.route(shifted, 0.10f, 0.15f, 0.05f);
        check(Rm.tier_mass(0) / std::max(1.0, static_cast<double>(N)) > 0.55,
              "under 20 cm mislocalization the base routing softens but is not lost (>55%)");
    }

    if (ok) std::printf("[kitchen_cells::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
