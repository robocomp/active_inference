/*
 * table_lidar_range_channel.cpp — the YOLO-independent LiDAR range channel (extracted from TableFitter).
 *
 * Implements feed(): select this cycle's staged sweep returns that land on ONE table and stage them on the
 * frame's LiDAR channel for the first-hit range factor / free-space VACATE term, with the z-calibration probe
 * and the two continuous informativeness down-weights (ray-count + angular coverage) on the precision. Holds
 * the TableConfig by reference; the sweep is staged/cleared per cycle by TableFitter (from TableLidarIngestor).
 */

#include "table_lidar_range_channel.h"

#include "../../common/lidar_select/lidar_select.h"   // rc::lidar_select:: (SHARED)

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace rc {

// ─── YOLO-independent LiDAR range channel ─────────────────────────────────────────────────────────

// Select returns from `sweep` (sensor at `origin`) landing on THIS table into `out`, applying the ray-count and
// angular-coverage down-weights to `precision_base`. Returns the staged ray count; *cov_ang gets the angular
// weight. Same selection geometry as the primary helios path — shared by the extra per-device (bpearl) set.
// Birth-footprint select box (rotation-agnostic radius + floor-referenced z-band). See table_lidar_range_channel.h.
TableLidarRangeChannel::SelectBox TableLidarRangeChannel::select_box() const
{
    const float m = cfg_.lidar_select_margin_m;
    return { 0.5f * std::sqrt(cfg_.tracker_birth_width_m * cfg_.tracker_birth_width_m
                            + cfg_.tracker_birth_depth_m * cfg_.tracker_birth_depth_m) + m,
             -m, cfg_.tracker_birth_height_m + m };
}

int TableLidarRangeChannel::select_into(const TableInstance& inst, const Eigen::Vector3f& centroid,
                                        const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                                        float precision_base, rc::ai::LidarRays& out, float* cov_ang) const
{
    // The selection loop and the two down-weights are SHARED (common/lidar_select) — they were byte-identical
    // between refrigerator and table and 98% across all four. What stays per-object is select_box(): WHERE
    // THIS BODY IS. That is the half that produced the live defects, so it is the half kept in sight.
    const auto [rxy, z_lo, z_hi] = select_box();
    const rc::lidar_select::Box box{rxy, z_lo, z_hi};
    const rc::lidar_select::Weights w{cfg_.lidar_coverage_n0, cfg_.lidar_coverage_ang_power, cfg_.lidar_robust_c_m};
    return rc::lidar_select::select_into(sweep, origin, {centroid.x(), centroid.y()}, box,
                                         precision_base, w, out, cov_ang);
}

// Select this cycle's returns landing on THIS table and stage them on frame.lidar (range factor / VACATE).
//
// The select box is anchored on the FRESH mask-cloud centroid (XY) — not the fitted state, which a diverging
// fit would drag into empty space, starving LiDAR exactly when it is most needed — and sized from the BIRTH
// footprint (not the fitted w/h, so a blown-up extent can't explode the region). The vertical band is
// floor-referenced [−m, birth_H+m] so it deliberately spans the LEGS and the tabletop RIM: the unbiased,
// segmentation-independent surfaces that attack the mask-erosion under-size. Final membership is the factor's
// own sphere-trace hit test (a ray must actually cross the model SDF), so this box is only a work bound +
// neighbour reject. Its radius/height margins (m = cfg.lidar_select_margin_m) are the SELECT-BOX thresholds.
void TableLidarRangeChannel::feed(TableInstance& inst, TableFrame& frame) const
{
    inst.dbg_lidar_rays = 0;
    inst.dbg_lidar_raw  = 0;
    inst.dbg_lidar_resid_m = -1.0f;
    inst.dbg_lidar_bpearl_rays = 0;
    frame.lidar_extra.clear();
    // Which planes have a FRESH sweep this cycle? helios feeds the range factor AND the free-space VACATE; bpearl
    // is an EXTRA per-device ray-set (range factor only). A cycle with masks (frame.points) but no LiDAR is a no-op.
    const bool helios_on = (cfg_.lidar_precision > 0.0f or cfg_.free_space_precision > 0.0f)
                           and lidar_have_sweep_ and not lidar_sweep_room_.empty();
    const bool bpearl_on = cfg_.lidar_bpearl_precision > 0.0f and lidar_have_bpearl_ and not lidar_sweep_bpearl_.empty();
    if ((not helios_on and not bpearl_on) or frame.points.empty())
        return;

    // Anchor XY on this cycle's fresh mask-cloud centroid (shared by both devices; NOT the fitted state, which a
    // diverging fit would drag into empty space, starving LiDAR exactly when it is most needed). Footprint select
    // box + z-band are the BIRTH dims (floor-referenced band spans the LEGS + rim) — see select_into().
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : frame.points) c += p;
    c /= static_cast<float>(frame.points.size());
    const auto& s = inst.ai2_belief.state();

    // ── Primary: helios (high 360) → frame.lidar (range factor AND free-space VACATE) + z-cal / coverage diag ──
    if (helios_on)
    {
        float cov_ang = 1.0f;
        const int n = select_into(inst, c, lidar_sweep_room_, lidar_origin_room_, cfg_.lidar_precision,
                                  frame.lidar, &cov_ang);
        inst.dbg_lidar_rays    = n;
        inst.dbg_lidar_cov_ang = cov_ang;

        // Generous "near?" raw count (1.5× box): is a return anywhere near this table? (below the birth
        // min-separation so it can't grab a neighbour's returns).
        const auto [rxy2_, zlo_, zhi_] = select_box();
        inst.dbg_lidar_raw = rc::lidar_select::raw_count(lidar_sweep_room_, {c.x(), c.y()},
                                                         {rxy2_, zlo_, zhi_});

        if (n > 0)
        {
            // Residual + z-calibration probe, SHARED. `sdf` is the agent's own surface, which is why it is a
            // callback: a box with legs uses the compound SDF, a cabinet run a plain box.
            rc::lidar_select::Diag d;
            rc::lidar_select::probe(frame.lidar, [&](const Eigen::Vector3f& p)
                                    { return inst.ai2_belief.sdf_compound(p, s); }, d);
            inst.dbg_lidar_resid_m  = d.resid_m;
            inst.dbg_lidar_meanz_m  = d.meanz;
            inst.dbg_lidar_topz_m   = d.topz;
            inst.dbg_lidar_floorz_m = d.floorz;
        }
    }

    // ── Extra: bpearl (low plane) → frame.lidar_extra, occlusion-aware from its OWN origin. Range factor only
    // (the VACATE stays on the helios sweep). Its low mounting strikes the LEGS the high helios grazes over. Runs
    // INDEPENDENTLY of helios (a table helios misses but bpearl sees still gets staged).
    if (bpearl_on)
    {
        rc::ai::LidarRays bp;
        const int nbp = select_into(inst, c, lidar_sweep_bpearl_, lidar_origin_bpearl_, cfg_.lidar_bpearl_precision, bp);
        if (nbp > 0) frame.lidar_extra.push_back(std::move(bp));
        inst.dbg_lidar_bpearl_rays = nbp;
    }
}

}  // namespace rc
