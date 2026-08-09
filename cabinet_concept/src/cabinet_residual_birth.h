/*
 * cabinet_residual_birth.h — residual-cluster → new-run birth candidate (header-only, pure).
 *
 * A single SAM mask that wraps a corner (an L-shaped cloud) is fitted by ONE axis-aligned run (the
 * dominant arm); the perpendicular arm then falls off-surface as residual (see the seed_room_axis note
 * in cabinet_belief.h). Nothing downstream models that arm, because the tracker only births from mask
 * DETECTIONS and one slice is assigned to one instance. This module closes the loop: it clusters the
 * pooled residual points, isolates the dominant coherent arm that is NOT already covered by a believed
 * run, and returns a ROOM-AXIS seed for it. The worker debounces the candidate over a few cycles and
 * births a new "cabinet_N" pre-seeded from it (see SpecificWorker::birth_from_residual).
 *
 * Pure geometry: no DSR, no state. Unit-tested by self_test().
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "cabinet_belief.h"     // rc::CabinetBelief::seed_from_points, rc::RunSeed
#include "cabinet_model.h"      // rc::CabinetState

namespace rc {

// Local copy of rc::geom::point_in_footprint (cabinet_geometry.h drags in DSR/Qt; this header stays
// pure so it is unit-testable in isolation). Is p inside the run's oriented footprint + margin?
inline bool in_footprint(const rc::CabinetState& s, const Eigen::Vector2f& p, float margin)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f d(p.x() - s.cx, p.y() - s.cy);
    const float along = d.x() * c + d.y() * sn;
    const float lat   = -d.x() * sn + d.y() * c;
    return std::abs(along) <= 0.5f * s.L + margin and std::abs(lat) <= 0.5f * s.d + margin;
}

struct ResidualBirthParams
{
    float cell_m              = 0.20f;   // grid cell for connected-component clustering (m)
    int   min_cluster_pts     = 600;     // mass below which a residual blob is ignored
    float min_length_m        = 0.50f;   // min along-axis extent of the arm (m)
    float min_aniso           = 0.30f;   // elongation: a run, not a square patch of clutter
    float sep_m               = 0.60f;   // min centre separation from every existing run (m)
    float footprint_margin_m  = 0.15f;   // residual points inside an existing footprint+margin are dropped
    float z_margin_m          = 0.10f;   // vertical-band slack: a run only "covers"/"is near" a residual point
                                         // if their z-bands overlap — a base run UNDER a wall run is distinct
};

struct ResidualBirthCandidate
{
    bool    ok  = false;
    RunSeed seed;                        // room-axis seed of the arm (cx,cy,yaw,L,z0,z1)
    int     pts = 0;                     // largest-component mass (0 if nothing to cluster)
    // ── diagnostics (always filled, for the per-cycle tuning log) ──────────────────────────────────
    int         pool        = 0;         // residual points NOT already on a believed footprint
    float       nearest_sep = -1.0f;     // centre gap to the nearest existing run (m); -1 if not evaluated
    const char* reason      = "no-input";// why ok is false ("ok" on success)
};

// Cluster `residual_pts` (room frame) and return the dominant coherent arm not explained by `existing`
// runs. Deterministic (grid + BFS), allocation-light.
inline ResidualBirthCandidate find_residual_birth(const std::vector<Eigen::Vector3f>& residual_pts,
                                                  const std::vector<rc::CabinetState>& existing,
                                                  const ResidualBirthParams& p)
{
    ResidualBirthCandidate out;
    if (residual_pts.size() < static_cast<std::size_t>(p.min_cluster_pts))
    { out.reason = "pool<min"; return out; }

    const float inv = 1.0f / std::max(1e-3f, p.cell_m);
    const auto key  = [&](float x, float y) -> std::int64_t
    {
        const std::int32_t ix = static_cast<std::int32_t>(std::floor(x * inv));
        const std::int32_t iy = static_cast<std::int32_t>(std::floor(y * inv));
        return (static_cast<std::int64_t>(ix) << 32) ^ (static_cast<std::uint32_t>(iy));
    };

    // Bin the points into grid cells, skipping any that already lie on a believed run's footprint.
    std::unordered_map<std::int64_t, std::vector<int>> cells;
    cells.reserve(residual_pts.size());
    for (int i = 0; i < static_cast<int>(residual_pts.size()); ++i)
    {
        const Eigen::Vector2f q(residual_pts[i].x(), residual_pts[i].y());
        const float qz = residual_pts[i].z();
        bool covered = false;
        for (const auto& e : existing)   // covered only if XY-inside AND vertically overlapping (base≠wall)
            if (in_footprint(e, q, p.footprint_margin_m)
                and qz >= e.z0 - p.z_margin_m and qz <= e.z1 + p.z_margin_m) { covered = true; break; }
        if (not covered)
        { cells[key(q.x(), q.y())].push_back(i); ++out.pool; }
    }
    if (cells.empty())
    { out.reason = "all-covered"; return out; }

    // Largest 4-connected component of occupied cells (BFS over the hash grid).
    std::unordered_map<std::int64_t, char> seen;
    seen.reserve(cells.size());
    std::vector<int> best;
    std::vector<std::int64_t> stack;
    for (const auto& [k0, _] : cells)
    {
        if (seen.count(k0)) continue;
        std::vector<int> comp;
        stack.clear(); stack.push_back(k0); seen[k0] = 1;
        while (not stack.empty())
        {
            const std::int64_t k = stack.back(); stack.pop_back();
            const auto it = cells.find(k);
            if (it == cells.end()) continue;
            comp.insert(comp.end(), it->second.begin(), it->second.end());
            const std::int32_t ix = static_cast<std::int32_t>(k >> 32);
            const std::int32_t iy = static_cast<std::int32_t>(static_cast<std::uint32_t>(k & 0xffffffff));
            const std::int64_t nb[4] = {
                (static_cast<std::int64_t>(ix + 1) << 32) ^ static_cast<std::uint32_t>(iy),
                (static_cast<std::int64_t>(ix - 1) << 32) ^ static_cast<std::uint32_t>(iy),
                (static_cast<std::int64_t>(ix) << 32) ^ static_cast<std::uint32_t>(iy + 1),
                (static_cast<std::int64_t>(ix) << 32) ^ static_cast<std::uint32_t>(iy - 1) };
            for (const std::int64_t n : nb)
                if (cells.count(n) and not seen.count(n)) { seen[n] = 1; stack.push_back(n); }
        }
        if (comp.size() > best.size()) best.swap(comp);
    }
    out.pts = static_cast<int>(best.size());
    if (out.pts < p.min_cluster_pts)
    { out.reason = "cluster<min"; return out; }

    // Room-axis seed of the arm. Reuses the same dominant-room-axis logic the birth path uses, so the
    // new run is born axis-aligned (never on the cluster's PCA diagonal).
    std::vector<Eigen::Vector3f> cluster;
    cluster.reserve(best.size());
    for (const int i : best) cluster.push_back(residual_pts[i]);
    out.seed = CabinetBelief::seed_from_points(cluster, /*room_axis_snap=*/true);
    if (not out.seed.ok)                    { out.reason = "seed-bad"; return out; }
    if (out.seed.L     < p.min_length_m)    { out.reason = "short";    return out; }
    if (out.seed.aniso < p.min_aniso)       { out.reason = "blob";     return out; }

    // The arm must be genuinely SEPARATE from every existing run — but only runs that share its VERTICAL
    // band can be "the same run". A base run directly under a wall run overlaps in XY yet is a distinct run
    // (the frontal base cabinet under the wall cabinet), so z-disjoint runs are skipped here. This is the
    // "an L-mask ⇒ two runs" invariant: don't let a vertically-separate neighbour veto the second leg.
    out.nearest_sep = std::numeric_limits<float>::max();
    for (const auto& e : existing)
    {
        const bool z_overlap = std::min(out.seed.z1, e.z1) >= std::max(out.seed.z0, e.z0) - p.z_margin_m;
        if (z_overlap)
            out.nearest_sep = std::min(out.nearest_sep, std::hypot(out.seed.cx - e.cx, out.seed.cy - e.cy));
    }
    if (out.nearest_sep < p.sep_m)
    { out.reason = "near-run"; return out; }

    out.ok = true; out.reason = "ok";
    return out;
}

// Synthesise an L-shaped cloud (long +x arm modelled by an existing run + a perpendicular +y arm as
// residual) and confirm the perpendicular arm is recovered axis-aligned and separated. Returns true on pass.
inline bool residual_birth_self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* what)
    { if (not c) { std::printf("[residual_birth::self_test] FAIL: %s\n", what); ok = false; } };

    // Existing run: the +x arm along y≈0, centred at (1.5,0), L=3, d=0.6.
    rc::CabinetState armA; armA.cx = 1.5f; armA.cy = 0.0f; armA.yaw = 0.0f; armA.L = 3.0f; armA.d = 0.6f;
    armA.z0 = 0.0f; armA.z1 = 0.9f;

    // Residual cloud = the perpendicular +y arm at x≈0 (a peninsula), PLUS a little clutter that sits on
    // armA's footprint (must be filtered out and never seed a birth on top of the existing run).
    std::vector<Eigen::Vector3f> res;
    for (int j = 0; j <= 40; ++j)
    {
        const float y = 0.2f + 1.5f * static_cast<float>(j) / 40.0f;   // y ∈ [0.2, 1.7]
        for (int w = -2; w <= 2; ++w)
            res.emplace_back(0.10f * static_cast<float>(w), y, 0.45f);
    }
    for (int i = 0; i < 40; ++i)                                        // clutter ON armA (to be excluded)
        res.emplace_back(0.5f + 0.05f * i, 0.0f, 0.45f);

    ResidualBirthParams p; p.min_cluster_pts = 100; p.min_aniso = 0.20f;
    const auto cand = find_residual_birth(res, {armA}, p);
    check(cand.ok, "an unexplained perpendicular arm yields a birth candidate");
    if (cand.ok)
    {
        constexpr float kQ = static_cast<float>(M_PI) / 2.0f;
        const float a = std::fmod(std::abs(cand.seed.yaw), kQ);
        const float axis_resid = std::min(a, kQ - a);
        check(axis_resid < 0.05f, "residual-born arm is axis-aligned (room axis, not diagonal)");
        check(std::abs(std::sin(cand.seed.yaw)) > 0.98f, "residual-born arm runs along +y (the peninsula)");
        check(std::hypot(cand.seed.cx - armA.cx, cand.seed.cy - armA.cy) > p.sep_m,
              "residual-born arm is separated from the existing run");
    }

    // Negative: no residual beyond the existing run ⇒ no birth (clutter on the run only).
    std::vector<Eigen::Vector3f> only_clutter;
    for (int i = 0; i < 200; ++i) only_clutter.emplace_back(0.5f + 0.01f * i, 0.0f, 0.45f);
    check(not find_residual_birth(only_clutter, {armA}, p).ok, "clutter on an existing run does NOT birth");

    if (ok) std::printf("[residual_birth::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
