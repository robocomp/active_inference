/*
 * countable_lidar_range_channel.cpp — the YOLO-independent LiDAR range channel (extracted from CountableFitter).
 *
 * Implements feed(): select this cycle's staged sweep returns that land on ONE bed and stage them on the
 * frame's LiDAR channel for the first-hit range factor / free-space VACATE term, with the z-calibration probe
 * and the two continuous informativeness down-weights (ray-count + angular coverage) on the precision. Holds
 * the CountableConfig by reference; the sweep is staged/cleared per cycle by CountableFitter (from ConceptLidarIngestor).
 */

#include "countable_lidar_range_channel.h"

#include "../../common/lidar_select/lidar_select.h"   // rc::lidar_select:: (SHARED)

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace rc {

// ─── YOLO-independent LiDAR range channel ─────────────────────────────────────────────────────────

// Select returns from `sweep` (sensor at `origin`) landing on THIS bed into `out`, applying the ray-count and
// angular-coverage down-weights to `precision_base`. Returns the staged ray count; *cov_ang gets the angular
// weight. Same selection geometry as the primary helios path — shared by the extra per-device (bpearl) set.
// Birth-footprint select box (rotation-agnostic radius + floor-referenced z-band). See countable_lidar_range_channel.h.
CountableLidarRangeChannel::SelectBox CountableLidarRangeChannel::select_box() const
{
    // Floor-anchored: [0, birth height]. Hanging (the hood): the class body [H − e, H] at the height prior — the floor band
    // would select the worktop and the hob under it and never the body (the hood_concept clone's disjoint band).
    const rc::countable::VerticalSpan span{cfg_.vertical_extent_m};
    const auto b = span.hangs()
        ? countable_select_box(cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, cfg_.ai2_prior_height_m,
                               cfg_.lidar_select_margin_m, span.z0(cfg_.ai2_prior_height_m))
        : countable_select_box(cfg_.tracker_birth_width_m, cfg_.tracker_birth_depth_m, cfg_.tracker_birth_height_m,
                               cfg_.lidar_select_margin_m);
    return {b.radius_xy, b.z_lo, b.z_hi};
}

Eigen::Vector3f CountableLidarRangeChannel::cloud_centroid(std::span<const Eigen::Vector3f> cloud)
{
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : cloud) c += p;
    c /= static_cast<float>(cloud.size());
    return c;
}

std::vector<CountableLidarRangeChannel::DumpReturn> CountableLidarRangeChannel::helios_returns_in_select_box(
    std::span<const Eigen::Vector3f> anchor_cloud, float margin_m) const
{
    std::vector<DumpReturn> out;
    if (not lidar_have_sweep_ or anchor_cloud.empty())
        return out;
    const Eigen::Vector3f c = cloud_centroid(anchor_cloud);
    const auto [rxy, z_lo, z_hi] = select_box();
    const float rg = rxy + margin_m, r2 = rg * rg;
    const float zlo_g = z_lo - margin_m, zhi_g = z_hi + margin_m;
    for (const auto& p : lidar_sweep_room_)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        const float d2 = dx * dx + dy * dy;
        if (d2 > r2 or p.z() < zlo_g or p.z() > zhi_g) continue;   // select_into's box test, grown by margin_m
        out.push_back({p, std::max(0.f, std::sqrt(d2) - rxy)});
    }
    return out;
}

int CountableLidarRangeChannel::select_into(const CountableInstance& inst, const Eigen::Vector3f& centroid,
                                        const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                                        float precision_base, rc::ai::LidarRays& out, float* cov_ang,
                                        const std::function<float(const Eigen::Vector3f&)>* membership,
                                        float* m_sum) const
{
    // The selection loop and the two down-weights are SHARED (common/lidar_select) — they were byte-identical
    // between bed and table and 98% across all four. What stays per-object is select_box(): WHERE
    // THIS BODY IS. That is the half that produced the live defects, so it is the half kept in sight.
    const auto [rxy, z_lo, z_hi] = select_box();
    const rc::lidar_select::Box box{rxy, z_lo, z_hi};
    const rc::lidar_select::Weights w{cfg_.lidar_coverage_n0, cfg_.lidar_coverage_ang_power, cfg_.lidar_robust_c_m};
    if (membership and *membership)   // camera attributes (spec lidar-measures G2): each return weighted by its m
        return rc::lidar_select::select_into_weighted(sweep, origin, {centroid.x(), centroid.y()}, box,
                                                      precision_base, w, *membership, out, cov_ang, m_sum);
    return rc::lidar_select::select_into(sweep, origin, {centroid.x(), centroid.y()}, box,
                                         precision_base, w, out, cov_ang);
}

// Select this cycle's returns landing on THIS bed and stage them on frame.lidar (range factor / VACATE).
//
// The select box is anchored on the FRESH mask-cloud centroid (XY) — not the fitted state, which a diverging
// fit would drag into empty space, starving LiDAR exactly when it is most needed — and sized from the BIRTH
// footprint (not the fitted w/h, so a blown-up extent can't explode the region). The vertical band is
// floor-referenced [−m, birth_H+m] so it deliberately spans the LEGS and the bedtop RIM: the unbiased,
// segmentation-independent surfaces that attack the mask-erosion under-size. Final membership is the factor's
// own sphere-trace hit test (a ray must actually cross the model SDF), so this box is only a work bound +
// neighbour reject. Its radius/height margins (m = cfg.lidar_select_margin_m) are the SELECT-BOX thresholds.
void CountableLidarRangeChannel::feed(CountableInstance& inst, CountableFrame& frame,
                                      std::span<const Eigen::Vector3f> anchor_cloud,
                                      const std::function<float(const Eigen::Vector3f&)>& membership) const
{
    const auto* mem = membership ? &membership : nullptr;   // empty ⇒ today's unweighted selection
    inst.dbg_lidar_msum = 0.0f;
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
    if ((not helios_on and not bpearl_on) or anchor_cloud.empty())
        return;

    // Anchor XY on this cycle's fresh mask-cloud centroid (shared by both devices; NOT the fitted state, which a
    // diverging fit would drag into empty space, starving LiDAR exactly when it is most needed). Footprint select
    // box + z-band are the BIRTH dims (floor-referenced band spans the LEGS + rim) — see select_into().
    const Eigen::Vector3f c = cloud_centroid(anchor_cloud);
    const auto& s = inst.ai2_belief.state();

    // ── Primary: helios (high 360) → frame.lidar (range factor AND free-space VACATE) + z-cal / coverage diag ──
    if (helios_on)
    {
        float cov_ang = 1.0f;
        const int n = select_into(inst, c, lidar_sweep_room_, lidar_origin_room_, cfg_.lidar_precision,
                                  frame.lidar, &cov_ang, mem, &inst.dbg_lidar_msum);
        inst.dbg_lidar_rays    = n;
        inst.dbg_lidar_cov_ang = cov_ang;

        // Generous "near?" raw count (1.5× box): is a return anywhere near this bed? (below the birth
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
    // INDEPENDENTLY of helios (a bed helios misses but bpearl sees still gets staged).
    if (bpearl_on)
    {
        rc::ai::LidarRays bp;
        // The same camera membership weights the bpearl set (off for nightstand/bed/toilet, spec §9).
        const int nbp = select_into(inst, c, lidar_sweep_bpearl_, lidar_origin_bpearl_, cfg_.lidar_bpearl_precision, bp,
                                    nullptr, mem);
        if (nbp > 0) frame.lidar_extra.push_back(std::move(bp));
        inst.dbg_lidar_bpearl_rays = nbp;
    }
}

// The persistent path's ray-set (spec 7c.8 a0): see the header. The staging itself is rc::stage_beam_cycle, shared with the tests.
void CountableLidarRangeChannel::feed_beam(CountableInstance& inst, CountableFrame& frame,
                                           std::span<const Eigen::Vector3f> anchor_cloud,
                                           const std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>& label) const
{
    const auto [rxy, z_lo, z_hi] = select_box();
    BeamStaging st;
    st.box = {rxy, z_lo, z_hi};
    st.w = {cfg_.lidar_coverage_n0, cfg_.lidar_coverage_ang_power, cfg_.lidar_robust_c_m};
    st.rho_max = static_cast<double>(rc::ai::LidarRays{}.max_range_m);   // ρ_max = the ray factor's max_range_m (6 m), spec (a)
    st.sigma_s = static_cast<double>(cfg_.model_sigma_m);
    st.room.floor = true;
    st.room.floor_z = 0.0;
    st.room.floor_sigma = cfg_.floor_sigma_m;
    st.room.walls = room_walls_;
    // F1.4 (option A, mean-field): the staged wall is where the wall latent's current mean puts it, not the raw outline edge.
    if (frame.wall.ok and frame.wall.edge >= 0 and static_cast<std::size_t>(frame.wall.edge) < st.room.walls.size())
    {
        const Eigen::Vector2d shift = double(inst.ai2_belief.wall_offset_mean(frame.wall)) * frame.wall.n.cast<double>();
        st.room.walls[static_cast<std::size_t>(frame.wall.edge)].a += shift;
        st.room.walls[static_cast<std::size_t>(frame.wall.edge)].b += shift;
    }
    st.room.neighbours = frame.foreign_boxes;   // G3: the other objects' published boxes are known surfaces of p_f too
    if (not beam_st_)
    {
        frame.beam = rc::ai::beam::BeamRays{};   // no beam density: the fitter did not build a BeamConfig (flag off)
        return;
    }
    st.st = *beam_st_;
    const bool sweep = lidar_have_sweep_ and not lidar_sweep_room_.empty();
    stage_beam_cycle(sweep ? &lidar_sweep_room_ : nullptr, lidar_origin_room_, anchor_cloud, st, label, frame.beam, inst.beam_log);
}

}  // namespace rc
