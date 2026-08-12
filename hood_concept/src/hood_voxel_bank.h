/*
 * hood_voxel_bank.h — this agent's ADAPTER onto common/voxel_bank.
 *
 * The bank itself (ownership gate, dedup grid, cap, counters, readout) is shared and lives in
 * common/voxel_bank/voxel_bank.h. What stays here is the only part that is genuinely about a hood: how to
 * read its EXTENT out of its own state, and which of the two vertical allowances its underside needs.
 */

#pragma once

#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../../common/voxel_bank/voxel_bank.h"   // rc::voxel_bank:: (SHARED)
#include "hood_config.h"
#include "hood_instance.h"

namespace rc::voxel_bank {

// This hood's extent + the allowances its geometry earns. The ONLY per-object part of the bank.
inline std::pair<Extent, Params> extent_of(const HoodInstance& inst, const HoodConfig& cfg)
{
    const auto& s = inst.model.state();
    Extent e;
    Params prm;
    prm.radius_margin_m = cfg.voxel_select_radius_margin_m;
    prm.height_margin_m = cfg.voxel_select_height_margin_m;
    prm.quantization_m  = cfg.voxel_bank_quantization_m;
    prm.max_points      = cfg.voxel_bank_max_points;
    // A hood HANGS: its band is the body's own [z0, z1] and BOTH ends are estimated, so the slack below is
    // model uncertainty at the same scale as above. This copy was the one already corrected — the other
    // three wrote a floor-referenced band, which for a hanging body admits the entire column beneath it.
    e.cx = s.cx; e.cy = s.cy;
    e.half_diag_m = 0.5f * std::sqrt(s.w * s.w + s.h * s.h);
    e.z0_m = s.z0();
    e.z1_m = s.z1();
    prm.below_m = cfg.voxel_select_height_margin_m;
    return {e, prm};
}

// Ingest this cycle's points, then log on the agent's own period.
inline void ingest(HoodInstance& inst, const std::vector<Eigen::Vector3f>& candidate_pts,
                   const std::vector<Eigen::Vector3f>& residual_pts, const HoodConfig& cfg)
{
    const auto [e, prm] = extent_of(inst, cfg);
    Bank bank{std::move(inst.voxel_bank_pts), std::move(inst.voxel_bank_keys)};
    const auto r = rc::voxel_bank::ingest(bank, e, prm, candidate_pts, residual_pts);
    inst.voxel_bank_pts  = std::move(bank.pts);
    inst.voxel_bank_keys = std::move(bank.keys);
    const int period = std::max(1, cfg.hood_log_period_frames);
    log_ingest(inst.node_name, inst.voxel_bank_pts.size(), prm, r, (inst.processed_cycles % period) == 0);
}

}  // namespace rc::voxel_bank
