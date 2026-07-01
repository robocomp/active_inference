/*
 * epistemic_planner.cpp — hidden-face next-best-view for bottle_concept.
 */

#include "epistemic_planner.h"

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
std::vector<Eigen::Vector3f> sample_hidden_arc(const BottleState& s, const Eigen::Vector2f& dir)
{ return sample_hidden_arc(s.cx, s.cy, s.cz, s.radius, s.height, dir); }
}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs, float view_info)
    : d_obs_(d_obs), view_info_(view_info)
{}

EpistemicProposal EpistemicPlanner::compute(const BottleModel& model,
                                            const Eigen::Vector2f& camera_xy,
                                            const std::array<float, 5>& posterior_info) const
{
    EpistemicProposal p;

    const auto& s = model.state();
    const Eigen::Vector2f bottle_xy(s.cx, s.cy);
    const Eigen::Vector2f ray = bottle_xy - camera_xy;          // camera → bottle (horizontal)
    if (not std::isfinite(ray.x()) or not std::isfinite(ray.y()) or ray.norm() < 1e-3f)
        return p;                                                // degenerate: can't resolve the hidden side

    const Eigen::Vector2f dir    = ray.normalized();
    const Eigen::Vector2f target = bottle_xy + dir * d_obs_;     // far side, beyond the bottle
    p.epistemic_target_x_m     = target.x();
    p.epistemic_target_y_m     = target.y();
    p.epistemic_target_yaw_rad = std::atan2(bottle_xy.y() - target.y(),
                                            bottle_xy.x() - target.x());   // look back at the bottle

    // Expected entropy reduction in the depth-degenerate radius DOF (index 3): ΔH = ½·log(1 + I_pred/Y).
    // I_pred = the radius Fisher information a back-side view would add, predicted on the SAME scale as
    // Y by evaluating observation_information on synthetic hidden-arc points (NOT a hand-set constant on
    // a different scale, which collapsed the ratio → gain≈0). ΔH→0 once the radius is resolved.
    const auto  back_pts = sample_hidden_arc(s, dir);
    const auto  I_pred   = model.observation_information(back_pts, {});
    const float Y_radius = std::max(posterior_info[3], 1e-3f);
    p.epistemic_gain = 0.5f * std::log(1.0f + I_pred[3] / Y_radius);

    p.valid = p.is_finite();
    return p;
}

EpistemicProposal EpistemicPlanner::compute(const BottleBelief& belief,
                                            const Eigen::Vector2f& camera_xy,
                                            float sigma_base) const
{
    EpistemicProposal p;

    const auto& s = belief.state();
    const Eigen::Vector2f bottle_xy(s.cx, s.cy);
    const Eigen::Vector2f ray = bottle_xy - camera_xy;          // camera → bottle (horizontal)
    if (not std::isfinite(ray.x()) or not std::isfinite(ray.y()) or ray.norm() < 1e-3f)
        return p;                                                // degenerate: can't resolve the hidden side

    const Eigen::Vector2f dir    = ray.normalized();
    const Eigen::Vector2f target = bottle_xy + dir * d_obs_;     // far side, beyond the bottle
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
    p.epistemic_gain = 0.5f * std::log(std::max(1e-9f, det));

    p.valid = p.is_finite();
    return p;
}

}  // namespace rc
