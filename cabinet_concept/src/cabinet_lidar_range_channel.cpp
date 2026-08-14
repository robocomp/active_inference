/*
 * cabinet_lidar_range_channel.cpp — the YOLO-independent LiDAR range channel (extracted from CabinetFitter).
 *
 * Implements feed(): select this cycle's staged sweep returns that land on ONE cabinet and stage them on the
 * frame's LiDAR channel for the first-hit range factor / free-space VACATE term, with the z-calibration probe
 * and the two continuous informativeness down-weights (ray-count + angular coverage) on the precision. Holds
 * the CabinetConfig by reference; the sweep is staged/cleared per cycle by CabinetFitter (from CabinetLidarIngestor).
 */

#include "cabinet_lidar_range_channel.h"

#include "../../common/lidar_select/lidar_select.h"   // rc::lidar_select:: (SHARED)

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rc {

// ─── YOLO-independent LiDAR range channel ─────────────────────────────────────────────────────────

// Select returns from `sweep` (sensor at `origin`) landing on THIS cabinet into `out`, applying the ray-count and
// angular-coverage down-weights to `precision_base`. Returns the staged ray count; *cov_ang gets the angular
// weight. Same selection geometry as the primary helios path — shared by the extra per-device (bpearl) set.
// Birth-footprint select box (rotation-agnostic radius + floor-referenced z-band). See cabinet_lidar_range_channel.h.
CabinetLidarRangeChannel::SelectBox CabinetLidarRangeChannel::select_box() const
{
    const float m = cfg_.lidar_select_margin_m;
    return { 0.5f * std::sqrt(cfg_.tracker_birth_width_m * cfg_.tracker_birth_width_m
                            + cfg_.tracker_birth_depth_m * cfg_.tracker_birth_depth_m) + m,
             -m, cfg_.tracker_birth_height_m + m };
}

int CabinetLidarRangeChannel::select_into(const CabinetInstance& inst, const Eigen::Vector3f& centroid,
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

// Select this cycle's returns landing on THIS cabinet and stage them on frame.lidar (range factor / VACATE).
//
// The select box is anchored on the FRESH mask-cloud centroid (XY) — not the fitted state, which a diverging
// fit would drag into empty space, starving LiDAR exactly when it is most needed — and sized from the BIRTH
// footprint (not the fitted w/h, so a blown-up extent can't explode the region). The vertical band is
// floor-referenced [−m, birth_H+m] so it deliberately spans the LEGS and the tabletop RIM: the unbiased,
// segmentation-independent surfaces that attack the mask-erosion under-size. Final membership is the factor's
// own sphere-trace hit test (a ray must actually cross the model SDF), so this box is only a work bound +
// neighbour reject. Its radius/height margins (m = cfg.lidar_select_margin_m) are the SELECT-BOX thresholds.
void CabinetLidarRangeChannel::feed(CabinetInstance& inst, CabinetFrame& frame) const
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

        // Generous "near?" raw count (1.5× box): is a return anywhere near this cabinet? (below the birth
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
                                    { return inst.ai2_belief.sdf_box(p, s); }, d);
            inst.dbg_lidar_resid_m  = d.resid_m;
            inst.dbg_lidar_meanz_m  = d.meanz;
            inst.dbg_lidar_topz_m   = d.topz;
            inst.dbg_lidar_floorz_m = d.floorz;
        }
    }

    // ── Free-space VACATE beam selection ─────────────────────────────────────────────────────────
    // Select beams whose SEGMENT (helios origin → return) crosses the CURRENT fitted box (inflated) —
    // INCLUDING through-beams whose endpoint lands BEYOND it. Those are exactly the beams the mask-centroid
    // range selection above filters out (an overgrown end's carving beams return far from the mask), so this
    // is a SEPARATE selection anchored on the fitted box (Fable's fix). The vacate factor then decides
    // membership continuously via p_through. Anchoring on the fitted box is correct here (not the risk the
    // range channel guards against): if the box sits in empty space we WANT the beams through it, to carve.
    if (helios_on and cfg_.free_space_precision > 0.0f and s.L > 0.0f)
    {
        const float m = cfg_.lidar_select_margin_m;
        const float cyaw = std::cos(-s.yaw), syaw = std::sin(-s.yaw);
        const float hL = 0.5f * s.L + m, hd = 0.5f * s.d + m;        // inflated fitted footprint
        const float lo[3] = {-hL, -hd, s.z0 - m};
        const float hi[3] = { hL,  hd, s.z1 + m};
        const Eigen::Vector3f& O = lidar_origin_room_;
        const float oxr = O.x() - s.cx, oyr = O.y() - s.cy;
        const Eigen::Vector3f Ol(oxr * cyaw - oyr * syaw, oxr * syaw + oyr * cyaw, O.z());
        frame.lidar_freespace.origin = O;
        frame.lidar_freespace.endpoints.clear();
        frame.lidar_freespace.endpoints.reserve(256);
        for (const auto& p : lidar_sweep_room_)
        {
            const float exr = p.x() - s.cx, eyr = p.y() - s.cy;
            const Eigen::Vector3f El(exr * cyaw - eyr * syaw, exr * syaw + eyr * cyaw, p.z());
            const Eigen::Vector3f dloc = El - Ol;
            float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
            bool miss = false;
            for (int a = 0; a < 3 and not miss; ++a)
            {
                if (std::abs(dloc(a)) < 1e-6f) { if (Ol(a) < lo[a] or Ol(a) > hi[a]) miss = true; }
                else
                {
                    float t1 = (lo[a] - Ol(a)) / dloc(a), t2 = (hi[a] - Ol(a)) / dloc(a);
                    if (t1 > t2) std::swap(t1, t2);
                    t_near = std::max(t_near, t1);
                    t_far  = std::min(t_far,  t2);
                }
            }
            if (miss or t_near > t_far or t_far < 0.0f or t_near > 1.0f) continue;   // segment crosses the box
            frame.lidar_freespace.endpoints.push_back(p);
        }
    }

    // ── Extra: bpearl (low plane) → frame.lidar_extra, occlusion-aware from its OWN origin. Range factor only
    // (the VACATE stays on the helios sweep). Its low mounting strikes the LEGS the high helios grazes over. Runs
    // INDEPENDENTLY of helios (a cabinet helios misses but bpearl sees still gets staged).
    if (bpearl_on)
    {
        rc::ai::LidarRays bp;
        const int nbp = select_into(inst, c, lidar_sweep_bpearl_, lidar_origin_bpearl_, cfg_.lidar_bpearl_precision, bp);
        if (nbp > 0) frame.lidar_extra.push_back(std::move(bp));
        inst.dbg_lidar_bpearl_rays = nbp;
    }
}

}  // namespace rc
