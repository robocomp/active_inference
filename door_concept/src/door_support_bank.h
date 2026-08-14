/*
 * door_support_bank.h — this agent's ADAPTER onto common/support_bank.
 *
 * The bank itself (ownership gate, dedup grid, cap, counters, readout) is shared and lives in
 * common/support_bank/support_bank.h. What stays here is the only part that is genuinely about a door: how
 * to read its EXTENT out of its own state, and which of the two vertical allowances its underside needs.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../../common/support_bank/support_bank.h"   // rc::support_bank:: (SHARED)
#include "door_config.h"
#include "door_instance.h"

namespace rc::support_bank {

// This door's extent + the allowances its geometry earns. The ONLY per-object part of the bank.
inline std::pair<Extent, Params> extent_of(const DoorInstance& inst, const DoorConfig& cfg)
{
    const auto& s = inst.model.state();
    Extent e;
    Params prm;
    prm.radius_margin_m = cfg.support_select_radius_margin_m;
    prm.height_margin_m = cfg.support_select_height_margin_m;
    prm.quantization_m  = cfg.support_bank_quantization_m;
    prm.max_points      = cfg.support_bank_max_points;
    // A door is FLOOR-ANCHORED: the bank spans the whole occupied column [0, H] and its underside is a
    // HARD bound - the floor - so it needs only the sensor-noise allowance, not model slack.
    e.cx = s.cx; e.cy = s.cy;
    e.half_diag_m = 0.5f * std::sqrt((s.w) * (s.w) + (s.thickness) * (s.thickness));
    e.z0_m = 0.0f;
    e.z1_m = s.cz + s.h;
    prm.below_m = 0.05f;
    return {e, prm};
}

// Ingest this cycle's points, then log on the agent's own period.
inline void ingest(DoorInstance& inst, const std::vector<Eigen::Vector3f>& candidate_pts,
                   const std::vector<Eigen::Vector3f>& residual_pts, const DoorConfig& cfg)
{
    const auto [e, prm] = extent_of(inst, cfg);
    Bank bank{std::move(inst.support_bank_pts), std::move(inst.support_bank_keys)};
    const auto r = rc::support_bank::ingest(bank, e, prm, candidate_pts, residual_pts);
    inst.support_bank_pts  = std::move(bank.pts);
    inst.support_bank_keys = std::move(bank.keys);
    const int period = std::max(1, cfg.door_log_period_frames);
    log_ingest(inst.node_name, inst.support_bank_pts.size(), prm, r, (inst.processed_cycles % period) == 0);
}

}  // namespace rc::support_bank
