/*
 * hood_lidar_range_channel.cpp — the YOLO-independent LiDAR range channel (extracted from HoodFitter).
 *
 * Implements feed(): select this cycle's staged sweep returns that land on ONE hood and stage them on the
 * frame's LiDAR channel for the first-hit range factor / free-space VACATE term, with the z-calibration probe
 * and the two continuous informativeness down-weights (ray-count + angular coverage) on the precision. Holds
 * the HoodConfig by reference; the sweep is staged/cleared per cycle by HoodFitter (from HoodLidarIngestor).
 */

#include "hood_lidar_range_channel.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace rc {

// ─── YOLO-independent LiDAR range channel ─────────────────────────────────────────────────────────

// Select returns from `sweep` (sensor at `origin`) landing on THIS hood into `out`, applying the ray-count and
// angular-coverage down-weights to `precision_base`. Returns the staged ray count; *cov_ang gets the angular
// weight. Same selection geometry as the primary helios path — shared by the extra per-device (bpearl) set.
// Birth-footprint select box (rotation-agnostic radius + the BODY's own z-band). See the header.
//
// ★THE Z-BAND IS THE BODY'S, NOT THE FLOOR'S. It was `[−m, BirthHeightM + m]` — the fridge's
// floor-referenced band, inherited by the clone and never re-derived. For a hood that is [−0.10, 0.85] m
// against a body at [1.79, 2.29] m: THE TWO DO NOT OVERLAP, so this function selected the floor and the
// worktop as "returns landing on this hood" and excluded every real hood return before the range factor
// could see one. Measured live 2026-08-11 (etc/ai2_log.csv, every cycle): 109 returns selected, mean SDF
// residual 1.33 m, selected z spanning 0.19–0.78 while the fitted H was 2.285 — and coverage still read 1.0
// (109 rays against LidarCoverageN0 = 60), so the channel reported FULL confidence in points that could not
// possibly be on the object. The geometric fit was silently mask-only, at obliquity_cos 0.29, which is
// where the 9°-off yaw (80.9° vs ~90°) and with it the visible tilt and the corner in the wall came from.
// cfg_.body_z0_m/body_z1_m are derived ONCE from the manifest's declared `support` — see hood_config.cpp.
//
// ★THE BAND FOLLOWS THE FITTED TOP, CLAMPED TO THE PRIOR'S OWN CREDIBLE INTERVAL. Anchoring it on the prior
// alone is not enough: the live fit sat at H = 2.285 against a prior of 2.05, so a prior-anchored
// [1.45, 2.15] would still have cut off the top 0.135 m of the body — the very surface H is estimated from.
// Anchoring it on the RAW fitted H is the failure the XY anchor already guards against (a diverging fit
// drags the selection into empty space and starves LiDAR exactly when it is needed). The prior states how
// far H may legitimately have moved, so ±2σ_H is the honest bound — a credible interval, not a new gate.
HoodLidarRangeChannel::SelectBox HoodLidarRangeChannel::select_box(float fitted_z_top_m) const
{
    const float m     = cfg_.lidar_select_margin_m;
    const float sigma = std::max(0.0f, cfg_.ai2_prior_height_std);
    const float z_top = std::isfinite(fitted_z_top_m)
                      ? std::clamp(fitted_z_top_m, cfg_.body_z1_m - 2.0f * sigma, cfg_.body_z1_m + 2.0f * sigma)
                      : cfg_.body_z1_m;
    const float extent = cfg_.body_z1_m - cfg_.body_z0_m;   // the body's own vertical extent (a parameter)
    return { 0.5f * std::sqrt(cfg_.tracker_birth_width_m * cfg_.tracker_birth_width_m
                            + cfg_.tracker_birth_depth_m * cfg_.tracker_birth_depth_m) + m,
             z_top - extent - m, z_top + m };
}

int HoodLidarRangeChannel::select_into(const HoodInstance& inst, const Eigen::Vector3f& centroid,
                                        const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                                        float precision_base, rc::ai::LidarRays& out, float* cov_ang) const
{
    const auto [rxy, z_lo, z_hi] = select_box(inst.ai2_belief.state().H);
    const float rxy2 = rxy * rxy;
    out.endpoints.clear();
    out.endpoints.reserve(256);
    double bear_c = 0.0, bear_s = 0.0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - centroid.x(), dy = p.y() - centroid.y();
        const float dh2 = dx * dx + dy * dy;
        if (dh2 > rxy2 or p.z() < z_lo or p.z() > z_hi) continue;
        out.endpoints.push_back(p);
        const float phi = std::atan2(dy, dx);
        bear_c += std::cos(phi); bear_s += std::sin(phi);
    }
    const int n = static_cast<int>(out.endpoints.size());
    float ca = 1.0f;
    if (n > 0)
    {
        const float coverage = std::min(1.0f, static_cast<float>(n) / std::max(1.0f, cfg_.lidar_coverage_n0));
        const float R        = static_cast<float>(std::hypot(bear_c, bear_s) / n);
        ca = std::pow(std::max(0.0f, 1.0f - R), cfg_.lidar_coverage_ang_power);
        out.origin     = origin;
        out.precision  = precision_base * coverage * ca;
        out.robust_c_m = cfg_.lidar_robust_c_m;
    }
    if (cov_ang) *cov_ang = ca;
    return n;
}

// Select this cycle's returns landing on THIS hood and stage them on frame.lidar (range factor / VACATE).
//
// The select box is anchored on the FRESH mask-cloud centroid (XY) — not the fitted state, which a diverging
// fit would drag into empty space, starving LiDAR exactly when it is most needed — and sized from the BIRTH
// footprint (not the fitted w/h, so a blown-up extent can't explode the region). The vertical band is the
// BODY's own extent about its fitted top, clamped to the height prior's ±2σ — for a hood that is the canopy
// in mid-air, and it must never be floor-referenced (the inherited [−m, birth_H+m] selected the floor and
// saw no hood at all — see select_box()). Final membership is the factor's own sphere-trace hit test (a
// ray must actually cross the model SDF), so this box is only a work bound + neighbour reject. Its
// radius/height margins (m = cfg.lidar_select_margin_m) are the SELECT-BOX thresholds.
void HoodLidarRangeChannel::feed(HoodInstance& inst, HoodFrame& frame) const
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
    // box is the BIRTH footprint; the z-band is the BODY's declared span, never the floor — see select_box().
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

        // Generous "near?" raw count (1.5× box): is a return anywhere near this hood? (below the birth
        // min-separation so it can't grab a neighbour's returns).
        const auto [rxy, z_lo, z_hi] = select_box(inst.ai2_belief.state().H);
        const float rraw2 = (1.5f * rxy) * (1.5f * rxy);
        int raw = 0;
        for (const auto& p : lidar_sweep_room_)
        {
            const float dx = p.x() - c.x(), dy = p.y() - c.y();
            if (dx * dx + dy * dy <= rraw2 and p.z() >= z_lo and p.z() <= z_hi) ++raw;
        }
        inst.dbg_lidar_raw = raw;

        if (n > 0)
        {
            double resid_sum = 0.0;
            for (const auto& p : frame.lidar.endpoints) resid_sum += std::abs(inst.ai2_belief.sdf_compound(p, s));
            inst.dbg_lidar_resid_m = static_cast<float>(resid_sum / n);
            // z-calibration probe: mean z of ALL selected returns + the highest/lowest 20% (≈ observed hoodtop /
            // floor). A persistent dbg_lidar_topz_m vs fitted H gap ⇒ a lidar3D→room z-offset, not a fit error.
            std::vector<float> zs; zs.reserve(frame.lidar.endpoints.size());
            double zsum = 0.0;
            for (const auto& p : frame.lidar.endpoints) { zs.push_back(p.z()); zsum += p.z(); }
            std::sort(zs.begin(), zs.end());
            const std::size_t k = std::max<std::size_t>(1, zs.size() / 5);
            double topsum = 0.0; for (std::size_t i = zs.size() - k; i < zs.size(); ++i) topsum += zs[i];
            double botsum = 0.0; for (std::size_t i = 0; i < k; ++i) botsum += zs[i];
            inst.dbg_lidar_meanz_m  = static_cast<float>(zsum / zs.size());
            inst.dbg_lidar_topz_m   = static_cast<float>(topsum / k);
            inst.dbg_lidar_floorz_m = static_cast<float>(botsum / k);
        }
    }

    // ── Extra: bpearl (low plane) → frame.lidar_extra, occlusion-aware from its OWN origin. Range factor only
    // (the VACATE stays on the helios sweep). Its low mounting strikes the LEGS the high helios grazes over. Runs
    // INDEPENDENTLY of helios (a hood helios misses but bpearl sees still gets staged).
    if (bpearl_on)
    {
        rc::ai::LidarRays bp;
        const int nbp = select_into(inst, c, lidar_sweep_bpearl_, lidar_origin_bpearl_, cfg_.lidar_bpearl_precision, bp);
        if (nbp > 0) frame.lidar_extra.push_back(std::move(bp));
        inst.dbg_lidar_bpearl_rays = nbp;
    }
}

}  // namespace rc
