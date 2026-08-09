/*
 * epistemic_planner.cpp — hidden-face next-best-view for bottle_concept.
 */

#include "epistemic_planner.h"
#include <print>

#include <cmath>
#include <vector>

namespace rc {

namespace
{
// Synthetic points on the bottle's HIDDEN surface arc (the half whose outward normal points away
// from the camera). Evaluating the SDF-likelihood Fisher on these predicts how much a back-side view
// would constrain the radius — on the SAME scale as the accumulated posterior precision Y.
std::vector<Eigen::Vector3f> sample_hidden_arc(float cx, float cy, float cz, float radius, float height,
                                               const Eigen::Vector2f& dir)
{
    constexpr int kArc = 7, kVert = 5;
    const float dir_ang = std::atan2(dir.y(), dir.x());   // camera→bottle; the far surface faces this way
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(kArc * kVert);
    for (int i = 0; i < kArc; ++i)
    {
        const float a  = dir_ang + (-M_PI_2f + M_PIf * static_cast<float>(i) / (kArc - 1));   // ±90° = hidden half
        const float px = cx + radius * std::cos(a);
        const float py = cy + radius * std::sin(a);
        for (int k = 0; k < kVert; ++k)
        {
            const float z = (cz - 0.5f * height) + height * static_cast<float>(k) / (kVert - 1);
            pts.emplace_back(px, py, z);
        }
    }
    return pts;
}
}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs, float view_info)
    : d_obs_(d_obs), view_info_(view_info)
{}

EpistemicProposal EpistemicPlanner::compute(const BottleBelief& belief,
                                            const Eigen::Vector2f& camera_xy,
                                            float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<rc::nbv::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon) const
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

    const auto& s = belief.state();
    const Eigen::Vector2f bottle_xy(s.cx, s.cy);
    const Eigen::Vector2f ray = bottle_xy - camera_xy;          // camera → bottle (horizontal)
    if (not std::isfinite(ray.x()) or not std::isfinite(ray.y()) or ray.norm() < 1e-3f)
        return p;                                                // degenerate: can't resolve the hidden side

    const Eigen::Vector2f dir = ray.normalized();

    // ── Where to stand: the argmax of the DETECTOR's model, not the d_obs_ constant ─────────────────────
    // A bottle is small, so the old fixed 0.9 m was in roughly the right place by luck; but "roughly right
    // for a 7 cm bottle" is not a model, and it says nothing about a 3 L one. rc::nbv scans P(detect) along
    // the hidden-side ray and returns where it peaks. The bottle is a CYLINDER: a square 2r × 2r footprint
    // presents the same silhouette from every bearing, which is exactly what a cylinder does.
    rc::nbv::Target tgt;
    tgt.cx = s.cx; tgt.cy = s.cy; tgt.yaw = 0.0f;
    tgt.w  = 2.0f * s.radius; tgt.h = 2.0f * s.radius;
    tgt.z0 = s.cz - 0.5f * s.height; tgt.z1 = s.cz + 0.5f * s.height;
    tgt.sigma_pos_m    = std::sqrt(std::max({0.0f, belief.covariance()(0, 0), belief.covariance()(1, 1)}));
    tgt.sigma_extent_m = 2.0f * std::sqrt(std::max(0.0f, belief.covariance()(3, 3)));   // radius → diameter

    // Origin is the bottle CENTRE (the hidden-side viewpoint is centre-relative, unlike the four-face
    // convention), so the geometric floor must clear the bottle's own radius as well as the robot's.
    const auto band = rc::nbv::standoff_band(tgt, sensor, bottle_xy, dir,
                                             robot_radius_m_ + s.radius);
    const float standoff = band.best;
    const Eigen::Vector2f target = bottle_xy + dir * standoff;   // far side, beyond the bottle
    p.epistemic_target_x_m     = target.x();
    p.epistemic_target_y_m     = target.y();
    p.epistemic_target_yaw_rad = std::atan2(bottle_xy.y() - target.y(),
                                            bottle_xy.x() - target.x());   // look back at the bottle

    // D-optimal expected entropy reduction on the belief's FULL Σ (mirrors table/chair AI2 NBV):
    // ΔH = ½·log det(I₅ + Σ·ΔI), ΔI = predicted Fisher info of a back-side view on synthetic hidden-arc
    // points. ΔH→0 once the belief tightens, so a well-seen bottle publishes a low gain the controller's
    // EFE selection won't pick (belief→knowledge governor).
    const auto  back_pts = sample_hidden_arc(s.cx, s.cy, s.cz, s.radius, s.height, dir);
    const float R  = std::max(1e-6f, sigma_base * sigma_base);
    const auto  dI = belief.predicted_information(back_pts, R);
    const auto& Sigma = belief.covariance();
    const float det = (Eigen::Matrix<float, 5, 5>::Identity() + Sigma * dI).determinant();
    const float raw_gain = 0.5f * std::log(std::max(1e-9f, det));

    // ── Weight by the probability of ACTUALLY getting a detection ──────────────────────────────────────
    // raw_gain is the information a mask from the far side would yield. Publishing it unweighted asks the
    // controller to drive around the bottle for a gain that only materialises if YOLO fires there — and for
    // a small object at the wrong range it does not. E[ΔH] = P(detect)·ΔH is the EFE-correct quantity, and
    // it is what makes an unreachable-in-practice viewpoint lose to the travel cost instead of winning it.
    rc::nbv::Candidate cand;
    cand.pos      = target;
    cand.yaw      = p.epistemic_target_yaw_rad;
    cand.raw_gain = raw_gain;
    const rc::nbv::Score sc = rc::nbv::score(cand, tgt, sensor, obstacles, robot_radius_m_);

    // ★CAN THE ROBOT BE THERE? score() answers "is this pose inside an obstacle" (stands_inside) and "how
    // much of the object does it see", but NOT "am I even in the room" — that check lives in plan_faces,
    // which this single-candidate path deliberately does not use. A bottle stands on a table that may sit
    // against a wall, so its far-side viewpoint is exactly the one that lands outside the room; and nothing
    // downstream refuses such a pose, because the controller REPAIRS an unroutable standpoint with
    // nearest_reachable, measured FROM the robot, which snaps the goal to the floor at the object.
    // Empty polygon ⇒ rc::nbv::is_reachable imposes no constraint (it refuses to guess), i.e. prior behaviour.
    if (not rc::nbv::is_reachable(cand.pos, room_polygon, robot_radius_m_))
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("bottle_concept: [NBV] far-side viewpoint is outside the room — REFUSING "
                       "(publishing it would be repaired onto the bottle).\n");
        return EpistemicProposal{};
    }
    p.epistemic_gain = sc.expected_gain;

    p.valid = p.is_finite();
    return p;
}

}  // namespace rc
