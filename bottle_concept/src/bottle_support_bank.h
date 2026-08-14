/*
 * bottle_support_bank.h — this agent's ADAPTER onto common/support_bank.
 *
 * The bank itself (ownership gate, dedup grid, cap, counters, readout) is shared. What stays here is the only
 * part that is genuinely about a bottle: how to read its extent out of its own state, and the two places a
 * bottle's answer differs from the furniture agents'.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../../common/support_bank/support_bank.h"   // rc::support_bank:: (SHARED)
#include "bottle_config.h"
#include "bottle_instance.h"

namespace rc::support_bank {

// This bottle's extent + the allowances its geometry earns. The ONLY per-object part of the bank.
inline std::pair<Extent, Params> extent_of(const BottleInstance& inst, const BottleConfig& cfg)
{
    const auto& s = inst.model.state();
    Extent e;
    Params prm;
    prm.radius_margin_m = cfg.support_select_radius_margin_m;
    prm.height_margin_m = cfg.support_select_height_margin_m;
    prm.quantization_m  = cfg.support_bank_quantization_m;
    prm.max_points      = cfg.support_bank_max_points;

    e.cx = s.cx; e.cy = s.cy;

    // ★THE XY GATE IS SIZED BY THE FIXED PRIOR RADIUS, NEVER THE FITTED ONE, and carries NO floor.
    // Gating on s.radius is a feedback loop: depth points just outside the cylinder get admitted -> they
    // support a larger radius -> the gate widens next frame -> the bottle "invents" radius out of
    // edge-pixel depth noise. A fixed gate caps how far depth alone can grow it; the mask silhouette owns
    // the actual radius. The shared 1 m floor exists for furniture and would more than double this gate,
    // undoing exactly that guard — so this agent passes 0 and means it.
    e.half_diag_m    = cfg.prior_radius;
    prm.min_radius_m = 0.0f;

    // The body's own span about the fitted centre.
    e.z0_m = s.cz - 0.5f * s.height;
    e.z1_m = s.cz + 0.5f * s.height;
    prm.below_m = cfg.support_select_height_margin_m;   // model slack: the underside is ESTIMATED

    // ★A BOTTLE'S SUPPORT IS RESOLVED PER INSTANCE, so when it is known the underside stops being an
    // estimate and becomes a HARD BOUND. Points at or below the table top are table-surface deprojections
    // the mask depth-gate let in; they drag the depth/centroid and inflate the lateral spread. +1 cm keeps
    // the base ring. A hard bound needs no model slack, which is what below_m = 0 says.
    if (std::isfinite(inst.table_top_z) and inst.table_top_z + 0.01f > e.z0_m - prm.below_m)
    {
        e.z0_m      = inst.table_top_z + 0.01f;
        prm.below_m = 0.0f;
    }
    return {e, prm};
}

// Ingest this cycle's points, then log on the agent's own period.
inline void ingest(BottleInstance& inst, const std::vector<Eigen::Vector3f>& candidate_pts,
                   const std::vector<Eigen::Vector3f>& residual_pts, const BottleConfig& cfg)
{
    const auto [e, prm] = extent_of(inst, cfg);
    Bank bank{std::move(inst.support_bank_pts), std::move(inst.support_bank_keys)};
    const auto r = rc::support_bank::ingest(bank, e, prm, candidate_pts, residual_pts);
    inst.support_bank_pts  = std::move(bank.pts);
    inst.support_bank_keys = std::move(bank.keys);
    const int period = std::max(1, cfg.log_period_frames);
    log_ingest(inst.node_name, inst.support_bank_pts.size(), prm, r, (inst.processed_cycles % period) == 0);
}

}  // namespace rc::support_bank
