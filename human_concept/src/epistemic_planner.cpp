/*
 * epistemic_planner.cpp — reduce-occlusion next-best-view for human_concept.
 */

#include "epistemic_planner.h"

#include <algorithm>
#include <cmath>

namespace rc {

EpistemicPlanner::EpistemicPlanner(float d_obs, float view_info)
    : d_obs_(d_obs), view_info_(view_info)
{}

EpistemicProposal EpistemicPlanner::compute(const Eigen::Vector2f& person_xy,
                                            const Eigen::Vector2f& camera_xy,
                                            float worst_info,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<rc::nbv::Obstacle>& obstacles) const
{
    // The camera geometry arrives from the caller (read fresh from the graph each cycle); the
    // detector envelope is ours. Merging here keeps the two concerns in their right owners.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;

    EpistemicProposal p;

    // ★Refuse on an incomplete camera model rather than guess. This path does NOT go through plan_faces, so
    // it does not inherit that choke point: without it, a missing vfov or mount height silently collapses the
    // fill model to horizontal-only and returns a confident, far-too-close viewpoint. See Sensor::complete().
    if (not sensor.complete())
        return p;

    const Eigen::Vector2f ray = person_xy - camera_xy;          // camera → person (horizontal)
    if (not std::isfinite(ray.x()) or not std::isfinite(ray.y()) or ray.norm() < 1e-3f)
        return p;                                                // degenerate: can't resolve the hidden side

    const Eigen::Vector2f dir = ray.normalized();

    // ── The person as a target for the DETECTOR model ──────────────────────────────────────────────────
    // NOMINAL standing-adult box: this planner is handed only the pelvis XY, not a fitted extent. That is
    // more defensible here than anywhere else in the fleet — a human silhouette is the most standardised
    // target we have, and it is precisely what YOLO's `person` class was trained on. If a fitted stature
    // ever becomes available at the call site, pass it instead of these constants.
    // No sigma_* is set: with a nominal box the extent uncertainty would be invented, and inventing it would
    // widen the framing on the strength of a number nobody measured.
    constexpr float kPersonWidthM = 0.50f, kPersonDepthM = 0.35f, kPersonHeightM = 1.75f;
    rc::nbv::Target tgt;
    tgt.cx = person_xy.x(); tgt.cy = person_xy.y(); tgt.yaw = std::atan2(dir.y(), dir.x());
    tgt.w  = kPersonWidthM; tgt.h = kPersonDepthM;
    tgt.z0 = 0.0f;          tgt.z1 = kPersonHeightM;

    // Origin is the person CENTRE (the far-side viewpoint is centre-relative), so the geometric floor must
    // clear the person's own half-depth as well as the robot's radius.
    const auto band = rc::nbv::standoff_band(tgt, sensor, person_xy, dir,
                                             robot_radius_m_ + 0.5f * kPersonDepthM);
    const Eigen::Vector2f target = person_xy + dir * band.best;   // far side, beyond the person
    p.epistemic_target_x_m     = target.x();
    p.epistemic_target_y_m     = target.y();
    p.epistemic_target_yaw_rad = std::atan2(person_xy.y() - target.y(),
                                            person_xy.x() - target.x());   // look back at the person

    // Expected entropy reduction in the worst-constrained DOF: ΔH = ½·log(1 + I_view/Y_worst).
    // ΔH→0 as the skeleton is well-observed (Y_worst grows), so a well-seen person yields a low gain.
    const float Y_worst = std::max(worst_info, 1e-3f);
    const float raw_gain = 0.5f * std::log(1.0f + view_info_ / Y_worst);

    // ── Weight by the probability of ACTUALLY getting a detection ──────────────────────────────────────
    // E[ΔH] = P(detect)·ΔH is the EFE-correct quantity: raw_gain is the information a skeleton observed from
    // the far side would yield, which only materialises if the detector fires there.
    rc::nbv::Candidate cand;
    cand.pos      = target;
    cand.yaw      = p.epistemic_target_yaw_rad;
    cand.raw_gain = raw_gain;
    p.epistemic_gain = rc::nbv::score(cand, tgt, sensor, obstacles, robot_radius_m_).expected_gain;

    p.valid = p.is_finite();
    return p;
}

}  // namespace rc
