/*
 * table_lidar_range_channel.cpp — the YOLO-independent LiDAR range channel (extracted from TableFitter).
 *
 * Implements feed(): select this cycle's staged sweep returns that land on ONE table and stage them on the
 * frame's LiDAR channel for the first-hit range factor / free-space VACATE term, with the z-calibration probe
 * and the two continuous informativeness down-weights (ray-count + angular coverage) on the precision. Holds
 * the TableConfig by reference; the sweep is staged/cleared per cycle by TableFitter (from TableLidarIngestor).
 */

#include "table_lidar_range_channel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace rc {

// ─── YOLO-independent LiDAR range channel ─────────────────────────────────────────────────────────

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
    // Stage the sweep for EITHER LiDAR consumer: the first-hit range factor (LidarPrecision) or the free-space
    // VACATE term (FreeSpacePrecision, EXISTENCE_BELIEF_PLAN.md Step 4). Both read frame.lidar.endpoints/origin.
    if ((cfg_.lidar_precision <= 0.0f and cfg_.free_space_precision <= 0.0f)
        or not lidar_have_sweep_ or lidar_sweep_room_.empty() or frame.points.empty())
        return;

    // Anchor XY on this cycle's fresh mask-cloud centroid.
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : frame.points) c += p;
    c /= static_cast<float>(frame.points.size());

    const auto& s = inst.ai2_belief.state();
    // Fixed footprint from the BIRTH dims (never the fitted w/h). Circumscribed horizontal radius so the box
    // is rotation-agnostic (yaw need not be resolved to select). Vertical band is floor-referenced.
    const float m    = cfg_.lidar_select_margin_m;
    const float rxy  = 0.5f * std::sqrt(cfg_.tracker_birth_width_m * cfg_.tracker_birth_width_m
                                      + cfg_.tracker_birth_depth_m * cfg_.tracker_birth_depth_m) + m;
    const float rxy2 = rxy * rxy;
    const float z_lo = -m;                                   // floor (room z≈0), minus a little slack
    const float z_hi = cfg_.tracker_birth_height_m + m;      // tabletop, plus margin
    // Generous diagnostic box (1.5× horizontal) — "is a return anywhere near this table?" — kept below the
    // birth min-separation so it can't grab a neighbouring table's returns.
    const float rraw2 = (1.5f * rxy) * (1.5f * rxy);
    int raw = 0;

    frame.lidar.endpoints.clear();
    frame.lidar.endpoints.reserve(256);
    double resid_sum = 0.0;
    double bear_c = 0.0, bear_s = 0.0;    // Σcos φ, Σsin φ of return bearings about the centre (angular coverage)
    for (const auto& p : lidar_sweep_room_)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        const float dh2 = dx * dx + dy * dy;
        if (dh2 <= rraw2 and p.z() >= z_lo and p.z() <= z_hi) ++raw;   // generous "near?" count
        if (dh2 > rxy2) continue;
        if (p.z() < z_lo or p.z() > z_hi) continue;
        frame.lidar.endpoints.push_back(p);
        resid_sum += std::abs(inst.ai2_belief.sdf_compound(p, s));     // |dist to CURRENT model surface|
        const float phi = std::atan2(dy, dx);                          // bearing about the centre (room XY)
        bear_c += std::cos(phi); bear_s += std::sin(phi);
    }
    inst.dbg_lidar_raw  = raw;
    inst.dbg_lidar_rays = static_cast<int>(frame.lidar.endpoints.size());
    if (inst.dbg_lidar_rays > 0)
        inst.dbg_lidar_resid_m = static_cast<float>(resid_sum / inst.dbg_lidar_rays);
    if (frame.lidar.endpoints.empty())
        return;

    // z-calibration probe: mean z of ALL selected returns + mean z of the highest 20% (≈ the observed tabletop
    // surface). Compare dbg_lidar_topz_m to the fitted H — a persistent gap ⇒ a lidar3D→room z-offset, not fit.
    {
        std::vector<float> zs; zs.reserve(frame.lidar.endpoints.size());
        double zsum = 0.0;
        for (const auto& p : frame.lidar.endpoints) { zs.push_back(p.z()); zsum += p.z(); }
        std::sort(zs.begin(), zs.end());
        const std::size_t k = std::max<std::size_t>(1, zs.size() / 5);   // top/bottom 20%
        double topsum = 0.0; for (std::size_t i = zs.size() - k; i < zs.size(); ++i) topsum += zs[i];
        double botsum = 0.0; for (std::size_t i = 0; i < k; ++i) botsum += zs[i];   // lowest returns ≈ floor
        inst.dbg_lidar_meanz_m = static_cast<float>(zsum / zs.size());
        inst.dbg_lidar_topz_m  = static_cast<float>(topsum / k);
        inst.dbg_lidar_floorz_m = static_cast<float>(botsum / k);   // should read ~0 if room z=0=floor + calib OK
    }

    // Precision = base, down-weighted by TWO informativeness factors (both continuous, no gate):
    //  (1) sparse RAY-COUNT coverage — a handful of noisy returns must not swing extent.
    //  (2) ANGULAR coverage — the circular variance (1−R) of the return bearings about the centre. R is the
    //      mean-resultant length: R→1 when all returns share one bearing (a ONE-SIDED sweep, blind to the far
    //      face and to which axis is which → near-zero orientation/extent info on a near-square table), R→0
    //      when returns wrap the object. Multiplying by (1−R)^p makes a one-sided frame contribute almost
    //      nothing to the ambiguous DOFs — killing the w↔h mode THRASH seen from far, degenerate viewpoints —
    //      while the RECURSIVE belief still accumulates coverage across an orbit (each frame is one-sided, but
    //      from DIFFERENT bearings that add up in Σ). Principled: precision ∝ the frame's actual angular info.
    const float coverage = std::min(1.0f, static_cast<float>(inst.dbg_lidar_rays)
                                        / std::max(1.0f, cfg_.lidar_coverage_n0));
    const float R        = static_cast<float>(std::hypot(bear_c, bear_s) / inst.dbg_lidar_rays);  // ∈[0,1]
    const float cov_ang  = std::pow(std::max(0.0f, 1.0f - R), cfg_.lidar_coverage_ang_power);
    inst.dbg_lidar_cov_ang = cov_ang;
    frame.lidar.origin     = lidar_origin_room_;
    frame.lidar.precision  = cfg_.lidar_precision * coverage * cov_ang;
    frame.lidar.robust_c_m = cfg_.lidar_robust_c_m;
}

}  // namespace rc
