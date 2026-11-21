/*
 * countable_beam_staging.h — one cycle's ray-set for the persistent path's smooth beam model, as the fitter stages it (spec
 * 2026-11-15 lidar-measures §7c.8 a0, rev 3.3.7/3.3.8 known room surfaces). Pure: Eigen + common/lidar_select + common/ai_belief;
 * no Qt, no DSR, no instance type, so the live-staging tests run the very function CountableLidarRangeChannel::feed_beam runs.
 *
 * What it does, in order:
 *   - no fresh Helios sweep          ⇒ nothing staged, logged as nomem_cycle = kNoSweep (not counted);
 *   - no membership (an empty label: no RGB frame staged, camera unbound, no focal length, empty slice; owner 11-18)
 *                                    ⇒ nothing staged, nomem_cycle = kNoMembership and the cumulative counter counts it,
 *                                      whether or not a mask cloud exists (an empty slice has none);
 *   - a membership but no mask cloud ⇒ nothing staged (no select box exists);
 *   - otherwise select_rays_beam stages every return whose RAY crosses the select box (ray geometry, not endpoint-in-box), each
 *     with its persistent label m (a return with no camera information is left out and counted as nolabel), κ = coverage·ca from
 *     the endpoint-in-box returns; σ_s = model σ (set here, BeamRays' own default is not used); p_f per ray against the room's
 *     known surfaces (floor z = 0 with Model.FloorSigmaM, the room polygon's walls with their σ).
 */
#pragma once

#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "../../common/ai_belief/beam_ray.h"
#include "../../common/ai_belief/beam_room.h"
#include "../../common/lidar_select/lidar_select.h"
#include "countable_mask_membership.h"   // rc::PersistentLabel

namespace rc {

// The countable select box: rotation-agnostic circumscribed radius of the BIRTH footprint + margin, floor-referenced z-band
// [−margin, birth_H + margin] (spans the legs and the rim). CountableLidarRangeChannel::select_box reads this.
inline rc::lidar_select::Box countable_select_box(float birth_w, float birth_d, float birth_H, float margin)
{
    return {0.5f * std::sqrt(birth_w * birth_w + birth_d * birth_d) + margin, -margin, birth_H + margin};
}

// The label select_rays_beam consumes: {m, q} per return, nullopt = no camera information.
using BeamLabel = std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>;

// make_persistent_label's result in select_rays_beam's form. An empty function stays empty (no membership this cycle).
inline BeamLabel beam_label(std::function<std::optional<PersistentLabel>(const Eigen::Vector3f&)> pl)
{
    if (not pl) return {};
    return [pl = std::move(pl)](const Eigen::Vector3f& p) -> std::optional<std::pair<double, float>>
    {
        const auto l = pl(p);
        if (not l) return std::nullopt;
        return std::pair<double, float>{l->m, l->q};
    };
}

// One cycle's beam-staging log (ai2_log columns beam_staged, beam_nolabel, beam_kappa, nomem_cycle, gap_rays) and the
// instance's cumulative count of cycles without membership.
struct BeamLog
{
    enum Nomem : int { kMembership = 0, kNoMembership = 1, kNoSweep = 2 };
    int   staged = 0;          // rays handed to the beam model
    int   nolabel = 0;         // rays crossing the select box left out for lack of camera information
    float kappa = 0.0f;        // the sweep weight coverage·ca
    int   nomem_cycle = kMembership;
    int   gap_rays = 0;        // the belief's count of rays with π > 0.99 and e < −5c after this cycle's update (stated limit 1)
    long  nomem_cycles = 0;    // cumulative: cycles with a sweep and a mask cloud but no membership
    void begin_cycle() { staged = 0; nolabel = 0; kappa = 0.0f; nomem_cycle = kMembership; gap_rays = 0; }
};

// What the staging reads besides the sweep: the select box and weights (the flag-off selection's, so κ is today's), ρ_max,
// the model σ_s, the room's known surfaces and the Student-t of the beam config (for p_f).
struct BeamStaging
{
    rc::lidar_select::Box     box;
    rc::lidar_select::Weights w;
    double rho_max = 6.0;
    double sigma_s = 0.01;
    rc::ai::beam::RoomSurfaces room;
    rc::ai::beam::StudentT     st;
};

// Stage one cycle's ray-set into `out` (cleared first) and log it (see the file header). `sweep` null or empty ⇒ no sweep.
// The select box is anchored on the XY centroid of `anchor_cloud` (the slice's mask cloud).
inline void stage_beam_cycle(const std::vector<Eigen::Vector3f>* sweep, const Eigen::Vector3f& origin,
                             std::span<const Eigen::Vector3f> anchor_cloud, const BeamStaging& cfg, const BeamLabel& label,
                             rc::ai::beam::BeamRays& out, BeamLog& log)
{
    out = rc::ai::beam::BeamRays{};
    out.rho_max = cfg.rho_max;
    out.sigma_s = cfg.sigma_s;
    if (sweep == nullptr or sweep->empty()) { log.nomem_cycle = BeamLog::kNoSweep; return; }
    // No membership is checked before the mask cloud: an empty slice has neither, and it is a no-membership cycle (spec a0).
    if (not label) { log.nomem_cycle = BeamLog::kNoMembership; ++log.nomem_cycles; return; }
    if (anchor_cloud.empty()) return;
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    for (const auto& p : anchor_cloud) c += p.head<2>();
    c /= static_cast<float>(anchor_cloud.size());
    rc::lidar_select::BeamStageStats st;
    rc::lidar_select::select_rays_beam(*sweep, origin, c, cfg.box, cfg.w, cfg.rho_max, label, out, &st);
    out.sigma_s = cfg.sigma_s;   // select_rays_beam leaves BeamRays' default; the model σ is the caller's (Task 18)
    rc::ai::beam::stage_room_foreign_density(out, cfg.room, cfg.st);
    log.nomem_cycle = BeamLog::kMembership;
    log.staged  = st.staged;
    log.nolabel = st.no_label;
    log.kappa   = st.kappa;
}

}  // namespace rc
