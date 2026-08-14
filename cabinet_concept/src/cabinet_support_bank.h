/*
 * cabinet_support_bank.h — this agent's ADAPTER onto common/support_bank.
 *
 * The bank itself (ownership gate, dedup grid, cap, counters, readout) is shared and lives in
 * common/support_bank/support_bank.h. What stays here is the only part that is genuinely about a cabinet: how to
 * read its EXTENT out of its own state, and which of the two vertical allowances its underside needs.
 */

#pragma once

#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../../common/support_bank/support_bank.h"   // rc::support_bank:: (SHARED)
#include "cabinet_config.h"
#include "cabinet_instance.h"

namespace rc::support_bank {

// This cabinet's extent + the allowances its geometry earns. The ONLY per-object part of the bank.
inline std::pair<Extent, Params> extent_of(const CabinetInstance& inst, const CabinetConfig& cfg)
{
    const auto& s = inst.model.state();
    Extent e;
    Params prm;
    prm.radius_margin_m = cfg.support_select_radius_margin_m;
    prm.height_margin_m = cfg.support_select_height_margin_m;
    prm.quantization_m  = cfg.support_bank_quantization_m;
    prm.max_points      = cfg.support_bank_max_points;
    // ★A RUN'S SUPPORT IS PER-INSTANCE (the manifest declares `resolved`): a BASE tier sits on the floor, a
    // WALL tier hangs at ~[1.40, 2.10]. This copy used a floor-referenced z_min = -0.05 and ignored z0
    // entirely — for a base run that is the same answer, and for a WALL run it is hood's defect exactly,
    // swallowing the whole column beneath the unit. z0 is a real estimated DOF here, so the slack below it
    // is model uncertainty rather than sensor noise at a hard bound.
    e.cx = s.cx; e.cy = s.cy;
    e.half_diag_m = 0.5f * std::sqrt(s.L * s.L + s.d * s.d);
    e.z0_m = s.z0;
    e.z1_m = s.z1;
    prm.below_m = cfg.support_select_height_margin_m;
    return {e, prm};
}

// Ingest this cycle's points, then log on the agent's own period.
inline void ingest(CabinetInstance& inst, const std::vector<Eigen::Vector3f>& candidate_pts,
                   const std::vector<Eigen::Vector3f>& residual_pts, const CabinetConfig& cfg)
{
    const auto [e, prm] = extent_of(inst, cfg);
    Bank bank{std::move(inst.support_bank_pts), std::move(inst.support_bank_keys)};
    const auto r = rc::support_bank::ingest(bank, e, prm, candidate_pts, residual_pts);
    inst.support_bank_pts  = std::move(bank.pts);
    inst.support_bank_keys = std::move(bank.keys);
    const int period = std::max(1, cfg.cabinet_log_period_frames);
    log_ingest(inst.node_name, inst.support_bank_pts.size(), prm, r, (inst.processed_cycles % period) == 0);
}

}  // namespace rc::support_bank
