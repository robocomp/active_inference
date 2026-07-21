/*
 * table_voxel_bank.h — the table-owned voxel memory (extracted from TableFitter).
 *
 * Header-only free functions in rc::voxel_bank that maintain each instance's persistent point bank: key()
 * (FNV-1a quantised voxel key), owned_by_table() (XY-radius + height ownership gate), and ingest() (insert
 * this cycle's candidate/residual points, cap-bounded, ownership-gated, de-duplicated by key). Depends only
 * on rc::TableInstance and rc::TableConfig, so it never pulls in TableFitter's nested observation type.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <print>
#include <vector>

#include <Eigen/Dense>

#include "table_config.h"     // rc::TableConfig
#include "table_instance.h"   // rc::TableInstance

namespace rc::voxel_bank {

// FNV-1a hash of a point quantised to a quantization_m grid — the voxel bank's O(1) dedup key.
inline std::uint64_t key(const Eigen::Vector3f& point, float quantization_m)
{
    const float q = std::max(1e-4f, quantization_m);
    const int ix = static_cast<int>(std::floor(point.x() / q));
    const int iy = static_cast<int>(std::floor(point.y() / q));
    const int iz = static_cast<int>(std::floor(point.z() / q));

    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    mix(static_cast<std::uint64_t>(ix));
    mix(static_cast<std::uint64_t>(iy));
    mix(static_cast<std::uint64_t>(iz));
    return h;
}

// True iff a point belongs to THIS table's voxel bank (ownership gate rejecting foreign/floor points).
//
// Two THRESHOLDS, both physical, keep a neighbouring table's or the floor's points out of the bank:
// an XY radius (table half-diagonal + config margin, floored at 1 m so a tiny model still owns nearby
// returns) and a vertical band [z_min, table_height + config margin].
inline bool owned_by_table(const TableInstance& inst, const Eigen::Vector3f& point, const TableConfig& cfg)
{
    const auto& s = inst.model.state();

    // XY ownership gate: table-centered radius with a configurable margin.
    const float half_diag = 0.5f * std::sqrt(s.w * s.w + s.h * s.h);
    const float gate_radius = std::max(1.0f, half_diag + cfg.voxel_select_radius_margin_m);
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate to reject floor / distant clutter points in mixed scenes.
    const float z_min = -0.05f;   // threshold: a little below the room floor (z≈0) to keep grazing returns
    const float z_max = s.table_height + cfg.voxel_select_height_margin_m;
    return point.z() >= z_min and point.z() <= z_max;
}

// Insert this cycle's candidate + residual points into the instance's voxel bank: ownership-gated,
// de-duplicated by voxel key, and hard-capped at cfg.voxel_bank_max_points. Logs on the log period.
inline void ingest(TableInstance& inst,
                   const std::vector<Eigen::Vector3f>& candidate_pts,
                   const std::vector<Eigen::Vector3f>& residual_pts,
                   const TableConfig& cfg)
{
    std::size_t inserted = 0;
    std::size_t rejected_foreign = 0;
    const auto max_points = static_cast<std::size_t>(std::max(1, cfg.voxel_bank_max_points));

    auto ingest_pts = [&](const std::vector<Eigen::Vector3f>& src)
    {
        for (const auto& p : src)
        {
            if (inst.voxel_bank_pts.size() >= max_points)
                break;
            if (not owned_by_table(inst, p, cfg))
            {
                ++rejected_foreign;
                continue;
            }
            const auto k = key(p, cfg.voxel_bank_quantization_m);
            if (inst.voxel_bank_keys.insert(k).second)
            {
                inst.voxel_bank_pts.push_back(p);
                ++inserted;
            }
        }
    };

    ingest_pts(candidate_pts);
    ingest_pts(residual_pts);

    const int period = std::max(1, cfg.table_log_period_frames);
    const bool do_log = (inst.processed_cycles % period) == 0;
    if (inserted > 0 and do_log)
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   inst.node_name, inserted, inst.voxel_bank_pts.size(), max_points, rejected_foreign);
}

}  // namespace rc::voxel_bank
