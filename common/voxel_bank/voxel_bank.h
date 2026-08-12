/*
 * voxel_bank.h — the per-instance accumulated point bank: ownership gate · dedup · cap. SHARED, header-only.
 *
 * Extracted 2026-08-12 from four byte-similar copies (cabinet · hood · refrigerator · table), measured at
 * 75.5% identical lines with the object's name normalised away — the highest duplication in the fleet.
 *
 * ★WHAT THE COPIES HAD ALREADY DRIFTED ON, which is the argument for extracting rather than tidying:
 *
 *   1. THE DEDUP KEY EXISTED THREE TIMES. refrigerator called the shared rc::voxel_key; table and cabinet
 *      each inlined their own FNV-1a. They agree today, and nothing whatsoever would have said so if they
 *      stopped: two definitions quantise onto two grids, and a bank seeded from a pre-birth burst then
 *      double-counts its own points against a bank that keyed them differently. That is the exact hazard
 *      common/birth_fragment's voxel_key was written to close, closed for one agent out of four.
 *   2. THE VERTICAL BAND WAS FLOOR-REFERENCED IN THREE OF THEM. `z_min = -0.05` with `z_max = height +
 *      margin` is a floor-anchored statement; hood_concept HANGS, and its band admitted the hob, the
 *      worktop and the backsplash into the instance's own bank as if they were hood surface. Same defect
 *      class as the LiDAR select box, different file, found a day apart.
 *
 * So the seam is the OBJECT's two answers, and nothing else:
 *
 *      Footprint  — the XY half-diagonal. A box says hypot(w,h)/2, a run says hypot(L,d)/2, a cylinder
 *                   says its radius. That is a genuine per-object accessor, not ceremony.
 *      Band       — [z0, z1], which since the manifest became authoritative is derived from `support` and
 *                   never re-stated. An agent whose support is `resolved` (bottle, cabinet) owns it per
 *                   instance and passes what it resolved.
 *
 * Everything after those two numbers — the radius floor, the margins, the dedup, the cap, the rejection
 * counters and the throttled readout — is identical across the fleet and lives here.
 *
 * Pure: Eigen + the standard library. No DSR, no config type, no instance type — the caller passes the
 * numbers its own state already carries, so this stays unit-testable and a new agent adopts it by supplying
 * two accessors rather than by copying a file.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "../birth_fragment/birth_fragment.h"   // rc::voxel_key — ONE grid, shared with the pre-birth burst

namespace rc::voxel_bank
{

// Where the object is, in the only two terms the bank needs. Built by the agent from its own state.
struct Extent
{
    float cx = 0.0f, cy = 0.0f;   // footprint centre (room frame)
    float half_diag_m = 0.0f;     // XY half-diagonal: box hypot(w,h)/2 · run hypot(L,d)/2 · cylinder r
    float z0_m = 0.0f;            // underside of the body — from the manifest's `support`, never re-derived
    float z1_m = 0.0f;            // top of the body
};

// The two margins + the cap + the quantisation. Identical in all four agents; kept as one struct so an
// agent passes its config once rather than threading four floats.
struct Params
{
    float radius_margin_m = 0.50f;   // XY slack beyond the half-diagonal
    float height_margin_m = 0.25f;   // slack ABOVE z1
    // ★BELOW IS NOT THE SAME QUANTITY AS ABOVE, and the four copies had already noticed without saying so:
    // three wrote a bare `-0.05` under a comment about "grazing returns just below the floor" while the
    // margin above was 0.25. That is not an inconsistency to tidy — it is two different physics. When the
    // underside is a HARD BOUND (the floor, for a floor-anchored or leg-supported object) the only slack
    // it needs is sensor noise, ~5 cm. When the underside is ESTIMATED (a hood's z0, a cabinet run's z0
    // DOF) the slack is model uncertainty and belongs at the same scale as the margin above. The agent
    // says which it has; conflating them is how hood's bank swallowed the worktop.
    float below_m         = 0.05f;   // slack BELOW z0 — sensor noise at a hard bound, model slack otherwise
    float quantization_m  = 0.02f;   // dedup grid
    int   max_points      = 4000;    // hard cap on the bank
};

// The bank itself. Two members, because that is all four agents carried.
struct Bank
{
    std::vector<Eigen::Vector3f>      pts;
    std::unordered_set<std::uint64_t> keys;
};

// The dedup key. ONE definition, shared with the pre-birth burst so a seeded bank cannot double-count.
inline std::uint64_t key(const Eigen::Vector3f& p, float quantization_m)
{
    return rc::voxel_key(p, quantization_m);
}

// Does this point belong to THIS instance's bank? An XY radius (half-diagonal + margin, floored at 1 m so a
// tiny or newborn model still owns nearby returns) and the BODY's own vertical band widened by its margin.
//
// ★The band is the body's, never the floor's. Three of the four copies wrote `z_min = -0.05`, which is a
// floor-anchored claim; for a hanging object it admits the entire column beneath it.
inline bool owns(const Extent& e, const Eigen::Vector3f& p, const Params& prm)
{
    const float gate_radius = std::max(1.0f, e.half_diag_m + prm.radius_margin_m);
    if (std::hypot(p.x() - e.cx, p.y() - e.cy) > gate_radius)
        return false;
    return p.z() >= e.z0_m - prm.below_m and p.z() <= e.z1_m + prm.height_margin_m;
}

struct IngestReport { std::size_t inserted = 0, rejected_foreign = 0, capped = 0; };

// Insert this cycle's points into the bank: ownership-gated, de-duplicated by voxel key, hard-capped.
// The caller decides what to log and when — a shared module must not own an agent's log cadence.
inline IngestReport ingest(Bank& bank, const Extent& e, const Params& prm,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           const std::vector<Eigen::Vector3f>& residual_pts)
{
    IngestReport r;
    const auto cap = static_cast<std::size_t>(std::max(1, prm.max_points));
    const auto add = [&](const std::vector<Eigen::Vector3f>& src)
    {
        for (const auto& p : src)
        {
            if (bank.pts.size() >= cap) { ++r.capped; continue; }
            if (not owns(e, p, prm))    { ++r.rejected_foreign; continue; }
            if (bank.keys.insert(key(p, prm.quantization_m)).second)
            {
                bank.pts.push_back(p);
                ++r.inserted;
            }
        }
    };
    add(candidate_pts);
    add(residual_pts);
    return r;
}

// The throttled readout the four copies each wrote out by hand, in the wording they had agreed on.
inline void log_ingest(std::string_view node_name, std::size_t total, const Params& prm,
                       const IngestReport& r, bool do_log)
{
    if (r.inserted > 0 and do_log)
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   node_name, r.inserted, total, prm.max_points, r.rejected_foreign);
}

}  // namespace rc::voxel_bank
