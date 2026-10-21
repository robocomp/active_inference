// doorway_hypotheses.h — Task 3e (rtD1h leg 5): WHICH LINE IS THE DOOR, and HAS THE ANCHOR BEEN RE-INITIALISED. Pure (Eigen + std
// + relative_pose.h); tests/test_doorway_hypotheses.cpp.
//
// ★ (1) THE GOAL LINE IS THE MAP OF TWO HYPOTHESES, NEVER THEIR AVERAGE (3b review I-2). During a cross the controller holds two
//   measurements of the one opening: door_concept's ANCHOR (its tracked belief, robot frame, with covariance) and the LiDAR's
//   measured GAP (with its score). 3b made the goal line the anchor's always — right in rtD1c (a 0.03-0.3-score gap locked on the
//   open leaf), wrong in rtD1h leg 5: the anchor jumped onto an alias ~1.9 m / 65-80° off the door while a 0.99-score gap kept
//   measuring the true opening 0.6 m ahead, and the anchor's line pulled the robot back out of the doorway. Joint hypotheses over
//   (anchor right?, gap right?), with priors from each measurement's OWN credibility,
//       π_A = P(the anchor's line is the door) = (1 − λ) · c_A      λ = P(the anchor is lost) — the anchor's own belief
//       π_G = P(the gap's line is the door)    = s · c_G             s = the gap detector's score
//   c = P(|lateral error| ≤ slack): the line's lateral sigma against the clearance the body has in that opening (½ width − the
//   body's half width) — the COVARIANCE term: a line known only to ±30 cm is not the door to a body with ±18 cm to spare.
//   The disagreement ν = gap − anchor (robot frame, yaw wrapped) scores the joint hypotheses: both right ⇒ ν ~ N(0, S), S = P_A + P_G;
//   any one wrong ⇒ ν ~ Cauchy(0, S + Σ_L), Σ_L = the alias scale (one opening width along and across the wall, a quarter turn
//   in yaw — door_concept's belief_lost, door_anchor_open.h, the same scale and tail). The posterior marginals
//       P(A right | ν) ∝ BR + GP,   P(G right | ν) ∝ BR + AL
//   differ only by AL − GP, and AL, GP share their likelihood: the disagreement says that ONE of them is wrong, never WHICH. So
//   the MAP line is the more credible measurement's — gap iff π_G > π_A — and ν is reported (both-right posterior), not used to
//   choose. A tie (no gap; equal credibility) keeps the anchor (the door's owner). Nothing here is a cutoff: s, λ and the sigmas
//   decide.
//
// ★ (2) A RE-INITIALISED ANCHOR IS A NEW BELIEF, NOT THE ONE DRIVEN ON. rtD1h attempt 2: the controller drove one tick on a parked
//   prediction (that sets `engaged`: "driven on a sighting in this attempt — stop and wait for it to be seen again"); door_concept
//   then re-initialised the anchor 2.17 m away and published it as a wait pose; the engaged branch stood still for 40 s instead of
//   turning to look. A newer block is scored against the block last driven on (both in the room frame, through the localiser's
//   pose): continuation ⇒ ν ~ N(0, S + Σ_robot); re-initialisation ⇒ ν ~ Cauchy(0, S + Σ_L), prior kAnchorLostPrior — the same
//   two-hypothesis model door_concept's own belief_lost uses. P(re-init) > 1/2 (the MAP of those two) ends `engaged`.
//
// ⚑ λ HOOK (Task D-B, door_concept): the owner will publish its anchor's P(lost); until then λ is the controller's own read —
//   the shared flagged prior, raised by re-initialisations seen this claim (anchor_lost_after_reinit). Pass the owner's value in
//   DoorwayLine::p_lost when it exists; nothing else changes.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>

#include "../../../common/interaction_anchor/relative_pose.h"

namespace rc::policy
{
// ⚑ The SAME flagged prior mass door_concept's belief_lost scores with (door_concept/src/door_anchor_open.h kAnchorLostPrior):
//   P(an anchor belief is lost) before any evidence. Replaced by the owner's published P(lost) once Task D-B lands.
inline constexpr double kDoorwayAnchorLostPrior = 0.2;

struct DoorwayLine
{
    rc::interaction::RelPose pose;        // robot frame: (x right, y forward, ψ); the opening's normal is (−sin ψ, cos ψ)
    float width = 0.f;                    // its clear width (m)
    double credibility = 0.0;             // anchor: 1 − λ (its own P(not lost)); gap: the detector's score s
};
struct DoorwayChoice
{
    bool gap = false;                     // true = the measured gap's line is the MAP door
    double pi_anchor = 0.0, pi_gap = 0.0; // the priors the choice compares (credibility × lateral coverage)
    double c_anchor = 0.0, c_gap = 0.0;   // the lateral coverage terms
    double p_both_right = 0.0;            // posterior that the two describe the same opening (reported, not used to choose)
    double d2 = 0.0;                      // Mahalanobis² of their disagreement under S
};

// The line's lateral sigma: the opening's along-wall axis u = (cos ψ, sin ψ) through its position covariance.
[[nodiscard]] inline double lateral_sigma(const rc::interaction::RelPose& r)
{
    const Eigen::Vector2d u(std::cos(double(r.x.z())), std::sin(double(r.x.z())));
    const Eigen::Matrix2d P = r.P.topLeftCorner<2, 2>().cast<double>();
    return std::sqrt(std::max(0.0, u.dot(P * u)));
}
// P(|e| ≤ slack), e ~ N(0, σ²): the probability that the body's centre, put on this line, is inside the clearance it has.
[[nodiscard]] inline double lateral_coverage(double sigma, double slack)
{
    if (not (slack > 0.0)) return 0.0;
    if (not (sigma > 0.0)) return 1.0;
    return std::erf(slack / (std::sqrt(2.0) * sigma));
}

// Log density of the two disagreement models (3-D): Gaussian N(0,S) and Cauchy-t(0, S+Σ_L). nullopt on a degenerate covariance.
struct DisagreementLogLik { double d2 = 0.0, l_same = 0.0, l_other = 0.0; };
[[nodiscard]] inline std::optional<DisagreementLogLik> disagreement_loglik(const Eigen::Vector3d& nu, const Eigen::Matrix3d& S,
                                                                           double width)
{
    const double wl = std::max(1e-3, width);
    const Eigen::Matrix3d SL = S + Eigen::Vector3d(wl * wl, wl * wl, std::numbers::pi * std::numbers::pi / 4.0).asDiagonal().toDenseMatrix();
    const Eigen::LLT<Eigen::Matrix3d> llt(S), lltL(SL);
    if (llt.info() != Eigen::Success or lltL.info() != Eigen::Success or not nu.allFinite()) return std::nullopt;
    const auto logdet = [](const Eigen::LLT<Eigen::Matrix3d>& c)
    { double l = 0; for (int i = 0; i < 3; ++i) l += 2.0 * std::log(c.matrixL()(i, i)); return l; };
    DisagreementLogLik o;
    o.d2 = nu.dot(llt.solve(nu));
    const double dL2 = nu.dot(lltL.solve(nu));
    const double k = 3.0, v = 1.0;   // Cauchy: "no scale of its own", door_concept's kAnchorLostDof
    o.l_same = -0.5 * o.d2 - 0.5 * k * std::log(2.0 * std::numbers::pi) - 0.5 * logdet(llt);
    o.l_other = std::lgamma(0.5 * (v + k)) - std::lgamma(0.5 * v) - 0.5 * k * std::log(v * std::numbers::pi) - 0.5 * logdet(lltL)
              - 0.5 * (v + k) * std::log1p(dL2 / v);
    return o;
}

// (1) The MAP doorway line. `body_half_width` = the bare body's lateral half width (its support radius along body ±x).
[[nodiscard]] inline DoorwayChoice choose_doorway_line(const DoorwayLine& anchor, const std::optional<DoorwayLine>& gap,
                                                       float body_half_width)
{
    DoorwayChoice c;
    const auto pi_of = [&](const DoorwayLine& l, double& cov)
    {
        cov = lateral_coverage(lateral_sigma(l.pose), 0.5 * double(l.width) - double(body_half_width));
        return std::clamp(l.credibility, 0.0, 1.0) * cov;
    };
    c.pi_anchor = pi_of(anchor, c.c_anchor);
    if (not gap.has_value()) { c.p_both_right = 1.0; return c; }
    c.pi_gap = pi_of(*gap, c.c_gap);
    c.gap = c.pi_gap > c.pi_anchor;
    Eigen::Vector3d nu = (gap->pose.x - anchor.pose.x).cast<double>();
    nu.z() = std::remainder(nu.z(), 2.0 * std::numbers::pi);
    const Eigen::Matrix3d S = (anchor.pose.P + gap->pose.P).cast<double>();
    if (const auto ll = disagreement_loglik(nu, S, std::max(anchor.width, gap->width)); ll.has_value())
    {
        c.d2 = ll->d2;
        const double a = c.pi_anchor, g = c.pi_gap;
        const double br = a * g, one_wrong = a * (1.0 - g) + (1.0 - a) * g + (1.0 - a) * (1.0 - g);
        const double m = std::max(ll->l_same, ll->l_other);
        const double wb = br * std::exp(ll->l_same - m), wo = one_wrong * std::exp(ll->l_other - m);
        c.p_both_right = wb + wo > 0.0 ? wb / (wb + wo) : 0.0;
    }
    return c;
}

// An anchor block in the ROOM frame through the robot's room pose (pos, θ: robot frame x right, y forward, rotated by θ), and its
// covariance rotated with it.
struct RoomAnchor { Eigen::Vector3d x = Eigen::Vector3d::Zero(); Eigen::Matrix3d P = Eigen::Matrix3d::Identity(); std::int64_t stamp_ms = 0; };
[[nodiscard]] inline RoomAnchor anchor_in_room(const rc::interaction::RelPose& r, const Eigen::Vector2f& robot_pos, float robot_theta)
{
    RoomAnchor o;
    const Eigen::Rotation2Dd R(static_cast<double>(robot_theta));
    o.x.head<2>() = robot_pos.cast<double>() + R * r.x.head<2>().cast<double>();
    o.x.z() = std::remainder(double(r.x.z()) + double(robot_theta), 2.0 * std::numbers::pi);
    Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
    J.topLeftCorner<2, 2>() = R.toRotationMatrix();
    o.P = J * r.P.cast<double>() * J.transpose();
    o.stamp_ms = r.stamp_ms;
    return o;
}

// (2) P(the newer block is a RE-INITIALISATION, not the continuation of the one driven on). Σ_robot = the localiser's room-frame
// pose covariance (x, y, θ; its θ variance also swings the anchor by its range — added as range²·var_θ on the position).
[[nodiscard]] inline double p_reinitialised(const RoomAnchor& driven, const RoomAnchor& now, double width, const Eigen::Matrix3d& robot_cov,
                                            double range_m)
{
    Eigen::Vector3d nu = now.x - driven.x;
    nu.z() = std::remainder(nu.z(), 2.0 * std::numbers::pi);
    Eigen::Matrix3d Sr = robot_cov;
    Sr(0, 0) += range_m * range_m * robot_cov(2, 2);
    Sr(1, 1) += range_m * range_m * robot_cov(2, 2);
    const auto ll = disagreement_loglik(nu, driven.P + now.P + Sr, width);
    if (not ll.has_value()) return 0.0;
    const double a = std::log(kDoorwayAnchorLostPrior) + ll->l_other, b = std::log(1.0 - kDoorwayAnchorLostPrior) + ll->l_same;
    return 1.0 / (1.0 + std::exp(b - a));
}

// λ after a re-initialisation seen this claim with probability p: the new belief and the one it replaced disagree, so ONE was lost —
// by symmetry of their (equal) priors, either; the current one is lost with ½ of that mass. No re-init (p = 0) leaves λ as it was.
[[nodiscard]] inline double anchor_lost_after_reinit(double lambda, double p_reinit)
{
    const double p = std::clamp(p_reinit, 0.0, 1.0);
    return (1.0 - p) * std::clamp(lambda, 0.0, 1.0) + p * 0.5;
}
}  // namespace rc::policy
