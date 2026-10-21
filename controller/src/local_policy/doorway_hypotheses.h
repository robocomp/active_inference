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
//   pose): continuation ⇒ ν ~ N(0, S + Σ_robot); re-initialisation ⇒ ν ~ Cauchy(0, S + Σ_L), prior = the anchor's CURRENT P(lost)
//   (AnchorLostBelief) — the two-hypothesis model door_concept's own belief_lost uses. P(re-init) > 1/2 (the MAP) ends `engaged`.
//
// ★ FIX ROUND 1 (review of 142d8d3a): no hard zero for a stale block (I-1: aged_anchor — its covariance grows with the motion
//   over its age), and INERTIA from accumulated evidence, not a hysteresis band (I-2: DoorwayBelief, AnchorLostBelief).
//
// ⚑ λ HOOK (Task D-B, door_concept): the owner will publish its anchor's P(lost); until then λ is the controller's own read —
//   AnchorLostBelief (fix round 1: recursive — falls on confirmed continuations, rises on a jump). Pass the owner's value as the
//   anchor's credibility (1 − P(lost)) when it exists; nothing else changes.
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
                                            double range_m, double prior_lost = kDoorwayAnchorLostPrior)
{
    Eigen::Vector3d nu = now.x - driven.x;
    nu.z() = std::remainder(nu.z(), 2.0 * std::numbers::pi);
    Eigen::Matrix3d Sr = robot_cov;
    Sr(0, 0) += range_m * range_m * robot_cov(2, 2);
    Sr(1, 1) += range_m * range_m * robot_cov(2, 2);
    const auto ll = disagreement_loglik(nu, driven.P + now.P + Sr, width);
    if (not ll.has_value()) return 0.0;
    const double pr = std::clamp(prior_lost, 1e-12, 1.0 - 1e-12);   // log of an exact 0/1 prior: numerics, not a model level
    const double a = std::log(pr) + ll->l_other, b = std::log(1.0 - pr) + ll->l_same;
    return 1.0 / (1.0 + std::exp(b - a));
}

// ─── FIX ROUND 1 (review of 142d8d3a) ───────────────────────────────────────────────────────────────────────────────────────────
// ⚑ τ_view: door_concept's PREDICTION HORIZON (kAnchorPredictMs, 3 s), reused as the time over which consecutive frames
//   decorrelate — a stated level, NOT a measured decorrelation time. Fix round 2 (a): it sets the WEIGHT of one look,
//   w = min(1, dt/τ_view): 20 Hz frames of one scene are not 20 independent looks (summed correlated evidence manufactures
//   confidence — the fractional-count lesson), and one decorrelated look is at most ONE observation however long the silence
//   before it. Elapsed time drives only the switching hazard.
inline constexpr double kDoorwayViewTauS = 3.0;
[[nodiscard]] inline double observation_weight(double dt_s)
{
    return dt_s > 0.0 ? std::min(1.0, dt_s / kDoorwayViewTauS) : 0.0;
}
// ⚑ T_switch: the mean time over which WHICH LINE IS THE DOOR may change within a claim (an anchor captured by an alias, a gap
//   locking onto a leaf). Stated, not measured. The switching hazard h = 1 − exp(−dt/T) mixes each belief toward its prior before
//   every update, so no verdict is final: it bounds the inertia (a reversal is answered within a few τ_view), never sets a
//   decision level.
inline constexpr double kDoorwaySwitchS = 30.0;

// I-1 / fix round 2 (c): A BLOCK'S INFORMATION DECAYS WITH THE MOTION SINCE ITS STAMP. A block is the opening relative to the
// robot AT ITS STAMP; it is used relative to the robot NOW, so the error it carries is the motion the robot ACTUALLY made in
// between — read off the localiser (pose at the stamp vs now: moved_m along, turned_rad in heading, which swings a point at range
// r by r·turned). That is added to the block's covariance and the coverage term falls with it. A robot that has not moved keeps
// the block's covariance: its age alone says nothing about the geometry (its lostness is λ's — observe_not_seen).
[[nodiscard]] inline rc::interaction::RelPose aged_anchor(const rc::interaction::RelPose& r, double moved_m, double turned_rad)
{
    rc::interaction::RelPose o = r;
    const double dv = std::abs(moved_m), dw = std::abs(turned_rad);
    const double range = double(r.x.head<2>().norm());
    const double vxy = dv * dv + range * range * dw * dw;
    o.P(0, 0) += float(vxy);
    o.P(1, 1) += float(vxy);
    o.P(2, 2) += float(dw * dw);
    return o;
}

// One frame's log-ratio for "this measurement's line is the door", from its credibility π ∈ [0,1] — door_concept's bounded view
// evidence (view_lost_llr_step): right ~ ½U + ½Beta(2,1), not ~ ½U + ½Beta(1,2), so log[(1 + 2π)/(1 + 2(1 − π))] ∈ [−log 3, log 3]
// and no single frame is decisive.
[[nodiscard]] inline double frame_llr(double pi)
{
    const double p = std::clamp(pi, 0.0, 1.0);
    return std::log((1.0 + 2.0 * p) / (1.0 + 2.0 * (1.0 - p)));
}

// I-2: WHICH LINE IS THE DOOR, as a recursive two-state belief (log-odds L = log P(gap)/P(anchor)): each cycle the switching
// hazard over the elapsed time first, then this cycle's ONE look — the gap's frame log-ratio for, the anchor's against —
// weighted by observation_weight(dt) ≤ 1. The MAP is L > 0; a tie (no evidence yet) keeps the anchor, the door's owner.
struct DoorwayBelief
{
    double L = 0.0;
    void predict(double dt_s)   // the switching hazard alone (a cycle with no gap held: no evidence about it)
    {
        if (not (dt_s > 0.0)) return;
        // a two-state chain switching at rate 1/T: P(in the other state after dt) = ½(1 − e^(−2dt/T)) — never past ½, so a long
        // silence relaxes the belief TO ½, it cannot invert it
        const double h = 0.5 * (1.0 - std::exp(-2.0 * dt_s / kDoorwaySwitchS));
        const double p = p_gap();
        const double q = (1.0 - h) * p + h * (1.0 - p);
        L = std::log(q / (1.0 - q));
    }
    void step(double pi_anchor, double pi_gap, double dt_s)
    {
        if (not (dt_s > 0.0)) return;
        predict(dt_s);
        L += (frame_llr(pi_gap) - frame_llr(pi_anchor)) * observation_weight(dt_s);
    }
    [[nodiscard]] bool gap() const { return L > 0.0; }
    [[nodiscard]] double p_gap() const { return 1.0 / (1.0 + std::exp(-L)); }
};

// I-2 (λ): P(the anchor is lost), recursive per published block. The hazard over the elapsed time relaxes it toward the shared
// prior; then the block, as ONE observation (weight ≤ 1):
//   • SEEN (the owner saw the door this frame, drivable, this claim): its transition from the last block scored under "a tracked
//     belief moves continuously" (Gaussian) vs "a lost belief continues (½) or wanders at the alias scale (½, Cauchy)" — it falls
//     on confirmed continuations (≤ log 2 per observation) and rises on a jump;
//   • PREDICTED (drivable, unseen_time > 0): odometry carries it smoothly BY CONSTRUCTION — a continuation says nothing about
//     lostness: hazard only (fix round 2 (b));
//   • NOT SEEN as the owner says so (a not-drivable wait pose, or a block parked by a claim that ended): observe_not_seen —
//     door_concept's own view evidence for a frame that did not see the door (p_view 0 ⇒ log 3 toward lost), weighted by the span
//     it stands for (fix round 2 (c)).
// observe() returns the single-transition P(re-initialised) under the CURRENT λ (the MAP test that ends `engaged`), computed
// before λ moves.
struct AnchorLostBelief
{
    double lambda = kDoorwayAnchorLostPrior;
    void predict(double dt_s)
    {
        if (not (dt_s > 0.0)) return;
        const double h = 1.0 - std::exp(-dt_s / kDoorwaySwitchS);
        lambda = (1.0 - h) * lambda + h * kDoorwayAnchorLostPrior;
    }
    double observe(const RoomAnchor& prev, const RoomAnchor& now, double width, const Eigen::Matrix3d& robot_cov, double range_m,
                   double dt_s, bool seen)
    {
        predict(dt_s);
        Eigen::Vector3d nu = now.x - prev.x;
        nu.z() = std::remainder(nu.z(), 2.0 * std::numbers::pi);
        Eigen::Matrix3d Sr = robot_cov;
        Sr(0, 0) += range_m * range_m * robot_cov(2, 2);
        Sr(1, 1) += range_m * range_m * robot_cov(2, 2);
        const auto ll = disagreement_loglik(nu, prev.P + now.P + Sr, width);
        if (not ll.has_value()) return 0.0;
        // ★ TWO DIFFERENT QUESTIONS, TWO LIKELIHOOD PAIRS. p_re answers "is THIS transition a re-initialisation?" — a jump vs a
        //   continuation: the Gaussian-vs-Cauchy pair (door_concept's belief_lost). The λ update below answers "is the BELIEF lost?"
        //   — and a lost belief also continues (it tracks its alias), so its likelihood is the ½ continue + ½ wander mixture.
        const double p_re = p_reinitialised(prev, now, width, robot_cov, range_m, lambda);
        // A LOST belief is not a random walk: door_concept keeps TRACKING whatever it has locked onto (an alias continues as
        // smoothly as the door). So under "lost" a transition continues with ½ and wanders at the alias scale with ½ — the
        // mixture door_concept's own view model uses (½U + ½Beta) — and a continuation can lower λ by at most log 2 per
        // observation, while a jump the Gaussian cannot explain still raises it decisively.
        if (seen)
        {
            const double m = std::max(ll->l_same, ll->l_other);
            const double l_lost = m + std::log(0.5 * std::exp(ll->l_same - m) + 0.5 * std::exp(ll->l_other - m));
            add_llr((l_lost - ll->l_same) * observation_weight(dt_s));
        }
        return p_re;
    }
    // dt_s: time since the previous update (the hazard); span_s: the time this block stands for unseen (its weight)
    void observe_not_seen(double dt_s, double span_s)
    {
        predict(dt_s);
        add_llr(std::log(3.0) * observation_weight(span_s));
    }

private:
    void add_llr(double d)
    {
        const double lam = std::clamp(lambda, 1e-12, 1.0 - 1e-12);   // log of an exact 0/1: numerics, not a model level
        const double logit = std::log(lam / (1.0 - lam)) + d;
        lambda = 1.0 / (1.0 + std::exp(-logit));
    }
};
}  // namespace rc::policy
