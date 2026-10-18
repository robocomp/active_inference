/*
 * door_anchor_update.h — the door's measurement update of the INTERACTION ANCHOR, as a PURE function: the
 * opening's pose relative to the robot (predicted by odometry, common/interaction_anchor/relative_pose.h) is
 * corrected by the robot's OWN LiDAR rays and the camera's door-mask points, all in the robot frame. No room
 * frame is read here, so a room-frame door fit that is 0.45 m off (live: the robot drove into the wall beside
 * the doorway) cannot reach the crossing.
 *
 * ─── FRAMES ──────────────────────────────────────────────────────────────────────────────────────────
 * Robot frame: forward = +y, side = +x (ROBOT_GEOMETRY.md). The anchor is the OPENING: its origin is the
 * opening's centre on the wall's mid-plane, its x-axis runs ALONG the wall, its y-axis is the wall normal.
 * RelPose::x = (x, y, yaw) places it in the robot frame: p_robot = R(yaw) p_anchor + (x, y). When approaching,
 * the robot is on the anchor's −y side, so its rays travel toward +y in anchor coordinates. (The depth below
 * is signed along the ray's travel, so a robot already on the +y side is handled by the same model.)
 *
 * ─── THE GENERATIVE MODEL ─────────────────────────────────────────────────────────────────────────────
 * Each ray (origin o, endpoint e, projected to the floor plane) is judged where its LINE crosses the wall line:
 *   u = the crossing's abscissa along the wall,  d = the endpoint's signed depth past the mid-plane.
 * A ray crossing INSIDE the opening must end beyond the far face; OUTSIDE, it must end on the wall:
 *   L_through(d) = logistic((d − t/2)/σ) / R      (R = the scan's max range: a return beyond the plane is
 *                                                   equally likely anywhere out to it — makes L a DENSITY,
 *                                                   commensurable with L_wall)
 *   L_wall(d)    = N(d; −t/2, σ²)                 (a beam stops on the wall's NEAR face, the one facing it)
 *   L_leaf(d|δ)  = N(d; δ, σ²)                     (a shut leaf at depth δ ∈ [−t/2, t/2]: flush with either
 *                                                   face or between)
 *   p(ray | x, open)    = m(u) L_through(d)   + (1 − m(u)) L_wall(d),   m(u) = logistic((w/2 − |u|_τ)/τ)
 *   p(ray | x, shut, δ) = m(u) L_leaf(d | δ)  + (1 − m(u)) L_wall(d)   (the leaf closes the opening)
 *   every branch is then mixed with two CLUTTER components (ε = 0.05, the one flagged constant):
 *     p_branch(ray) = (1 − 2ε)·[the above] + ε·U(d; d_origin, −t/2) + ε/R
 *   — an OCCLUDER in front of the plane (uniform depth between the beam's origin and the near face: the leaf swung
 *   toward the robot, a person) and a BACKGROUND return anywhere in range (a second doorway in the same wall, the
 *   wall's end, glass). They bound every ray's pull: a far misfit costs at most −log ε, not a quadratic.
 *   p(rays | x)         = π Π p(ray | x, open) + (1 − π) E_δ[ Π p(ray | x, shut, δ) ],   π = 0.5, δ uniform
 * The open/shut latent AND the leaf depth δ are per FRAME (one leaf, one state, one depth for every ray) and are
 * marginalised, not decided. δ is a grid over [−t/2, t/2] at step ≤ σ/2, fine enough that the sum is the integral.
 * ★δ must be marginalised per frame, not per ray: spread per ray (N(d; 0, σ² + (t/2)²)) it charged each ray on a
 * leaf flush with the near face ~0.6 nats for the width, the opening was again pushed off the rays, and the
 * Hessian at the end was indefinite (test shut_door: the covariance fell back to the prior's).
 * ★Without the shut branch a shut door's wall returns inside the hypothesised opening could only mean "no
 * opening here", and the anchor ran off the end of the wall to where no ray landed (test shut_door: x = 4.49 m).
 * With it, a flat wall is explained equally at every lateral shift and says nothing along itself (a leaf
 * RECESSED from the near face still says a little: L_leaf and L_wall then differ, which is real evidence).
 * ★The wall term is centred on the NEAR face, not the mid-plane: centred at 0 it put the posterior's depth on
 * the face the robot sees, 5.5 cm (t/2) short of the opening's centre, on every frame.
 * σ = 0.05 m is the beam's range noise (the same value select_doorway_rays in door_phi_posterior.h uses);
 * τ = σ is the lateral blur of where a beam crosses the jamb line — a 5 cm range error at the grazing angles
 * that define a jamb is a 5 cm error along the wall. |u|_τ = sqrt(u² + τ²) is |u| with the same blur, so the
 * membership is smooth at the centre too. The mixture is the PROPER log-sum (not m·log L_t + (1−m)·log L_w):
 * a through-ray that falls just outside the hypothesised opening is explained only by m, so it PULLS the
 * opening toward it; a wall-ray just inside pushes it away. That is how the jambs constrain the anchor along
 * the wall, while the wall's rays constrain its depth and yaw — and a flat (shut) wall, whose rays look the
 * same at every lateral shift, constrains nothing along itself.
 *
 * Mask: the along-wall centroid ū of the camera's door-mask points (projected into anchor coordinates) is a
 * Student-t (ν = 3) observation of 0, the opening's centre, with σ_m = spread(u)/√min(N, 30) — the 30 caps the
 * √N gain because neighbouring mask pixels are correlated (one segmentation, one boundary error). A misplaced
 * mask loses its pull through the t's tail, not through a gate; camera_nis = ū²/σ_m² at the optimum.
 *
 * Free energy: F(x) = ½ (x − x̄)ᵀ P̄⁻¹ (x − x̄) − log p(rays | x) − log p(mask | x). NO RAY IS EVER REMOVED FROM F.
 *   - PREDICTIVE SPREAD. Each ray's depth σ and jamb blur τ are widened by the spread the current belief P
 *     predicts for that ray (J P Jᵀ, the Kalman S = HPHᵀ + R); the mask's σ likewise. Two stages: the prior's P,
 *     then the Laplace P stage 1 converged to. This is what gives the update its reach: against the sensor's 5 cm
 *     alone, wall returns from a prior 0.3 m off in depth are outliers the occluder absorbs, and a 0.45 m lateral
 *     error is beyond the jamb's pull.
 *   - OPTIMISER. Per stage, at most 5 accepted Levenberg-Marquardt steps in the prior's metric, accepted on the
 *     gain ratio. Derivatives are central finite differences of F in double precision (F is smooth by
 *     construction); the curvature used is P̄⁻¹ + [I_L]₊, the data's information projected onto its PSD part.
 *   - LAPLACE + COMMON MODE. At the sensor's own σ: I_L from the exact finite-difference Hessian (a Gauss-
 *     Newton/Fisher one would count each jamb-adjacent wall ray as lateral information on a featureless shut wall),
 *     then the frame's common offset (σ_c = 0.02 m on x, y: one LiDAR extrinsic, one wall that is not a perfect
 *     plane) marginalised by Woodbury, I_eff = (I + I_L C)⁻¹ I_L — so ~900 correlated rays cannot claim a 2 mm σ.
 *
 * COST: ~29 ms on a full scan (469 returns), ~15 ms on the 221 within 2 m of the anchor (-O2). The caller
 * should pass the DOORWAY NEIGHBOURHOOD's rays (Task 4 preselects); rays far from the opening add cost, not
 * information.
 *
 * ─── THE ONLY DECISION LEVELS (deliberate, not model terms) ──────────────────────────────────────────
 *   Both are COUNTS at the posterior mean, for the log and for p_open. Neither feeds back into the pose.
 *   1. SHORT rays — endpoint d < −(3σ + t/2): ended in front of the near face beyond range noise (the same level
 *      as select_doorway_rays). ★This used to DROP the ray from F at the current pose. With a prior 0.15 m+ beyond
 *      the true near face every wall return was "short", nothing pulled the depth back, and the anchor ran 1.1 m
 *      off in depth / 1.3 m along the wall (review, fix round 1). The occluder component does the job continuously.
 *   2. p_open — rays crossing inside the span (|u| < w/2) are THROUGH if d > 3σ + t/2, BLOCKED if |d| ≤ 3σ + t/2;
 *      p_open = (n_through + 1)/(n_through + n_blocked + 2), a Beta(1,1) posterior on "rays pass".
 *
 * The caller selects the rays (the wall band in z: above the floor, below the lintel); this file uses x, y.
 * Header-only, no DSR / Qt / OpenCV of its own (relative_pose.h brings rt_twist.h).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "interaction_anchor/relative_pose.h"

namespace rc::door
{

struct AnchorEvidence
{
    int n_through = 0, n_blocked = 0, n_short = 0;   // ray classes against the predicted opening
    float p_open = 0.5f;                              // (n_through+1)/(n_through+n_blocked+2), rays inside the span
    float camera_nis = -1.f;                          // mask centroid's normalised innovation; -1 = no mask
};

namespace anchor_detail
{
constexpr double kRangeSigma  = 0.05;   // beam range noise (m), as in door_phi_posterior.h
constexpr double kMaxRange    = 8.0;    // the scan's max range (m), as in select_doorway_rays
constexpr double kMaskNu      = 3.0;    // Student-t dof of the mask centroid
constexpr int    kMaskNCap    = 30;     // correlated mask pixels: √N gain capped here
constexpr int    kMaxIter     = 5;
constexpr double kPriorOpen   = 0.5;    // P(door open) before this frame: uninformed (the interface carries no leaf belief)
// ⚑ FLAGGED CONSTANT — the clutter fraction: the share of beams explained by something that is not this doorway's
//   wall/opening/leaf. It weights BOTH extra components: an occluder in front of the plane (the leaf swung toward the
//   robot, a person) and a background return anywhere in range (another doorway in the same wall, the wall's end,
//   glass). No model term sets it from data yet; 0.05 is "one beam in twenty".
constexpr double kClutter     = 0.05;
// The per-frame COMMON MODE: every ray of one frame shares the LiDAR's extrinsic error and the wall's departure
// from a plane, so their evidence about the anchor's lateral/depth position is correlated at this σ (m).
constexpr double kCommonSigma = 0.02;

using V3d = Eigen::Vector3d;
using M3d = Eigen::Matrix3d;

inline double log_sigmoid(double z) { return z >= 0 ? -std::log1p(std::exp(-z)) : z - std::log1p(std::exp(z)); }
inline double log_add_exp(double a, double b)
{
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    const double m = std::max(a, b);
    return m + std::log1p(std::exp(-std::abs(a - b)));
}
inline double log_normal(double r, double var) { return -0.5 * r * r / var - 0.5 * std::log(2.0 * M_PI * var); }

// A 2-D ray in ROBOT coordinates.
struct Ray2 { Eigen::Vector2d o, e; };

// The ray in anchor coordinates: along-wall crossing u, the endpoint's signed depth d and the ORIGIN's signed
// depth d_o (both along the ray's travel; d_o < 0 while the beam starts in front of the plane).
struct UD { double u, d, d_o; };
inline UD to_anchor(const Ray2& r, const V3d& x)
{
    const double c = std::cos(x.z()), s = std::sin(x.z());
    const Eigen::Vector2d t(x.x(), x.y());
    auto inv = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - t; return Eigen::Vector2d(c * q.x() + s * q.y(), -s * q.x() + c * q.y()); };
    const Eigen::Vector2d oa = inv(r.o), ea = inv(r.e);
    const double sgn = oa.y() <= 0 ? 1.0 : -1.0;   // which side the beam starts on: its travel is "past the plane"
    const double dy = ea.y() - oa.y();
    // Where the ray's LINE crosses y = 0. A ray parallel to the wall never crosses it: put its crossing far along
    // the wall (1e9 m, finite so differences stay finite), where m = 0 and only the depth terms speak.
    const double u = std::abs(dy) > 1e-12 ? oa.x() - oa.y() * (ea.x() - oa.x()) / dy : std::copysign(1e9, ea.x() - oa.x());
    return {u, sgn * ea.y(), sgn * oa.y()};
}

// The PREDICTIVE spread of one ray's (u, d) under the current pose belief P: J P Jᵀ, J by finite differences.
// The likelihood is evaluated against the innovation's spread, not the sensor's alone — the Kalman S = HPHᵀ + R.
struct Spread { double su2 = 0, sd2 = 0; };
inline Spread predictive_spread(const Ray2& r, const V3d& x, const M3d& P)
{
    Eigen::Matrix<double, 2, 3> J;
    for (int i = 0; i < 3; ++i)
    {
        V3d e = V3d::Zero(); e(i) = 1e-4;
        const UD p = to_anchor(r, x + e), m = to_anchor(r, x - e);
        J(0, i) = (p.u - m.u) / 2e-4;
        J(1, i) = (p.d - m.d) / 2e-4;
    }
    const Eigen::Matrix2d S = J * P * J.transpose();
    Spread sp{S(0, 0), S(1, 1)};
    if (not std::isfinite(sp.su2)) sp.su2 = 0;   // a ray parallel to the wall: its crossing is "far" either way
    if (not std::isfinite(sp.sd2)) sp.sd2 = 0;
    return sp;
}

struct Model
{
    double w, t;                       // opening width, wall thickness (m)
    std::vector<double> leaf_depths;   // the δ grid
    double short_level, count_level;   // both 3σ + t/2 (see the header: counting only)
    explicit Model(double w_, double t_)
        : w(std::max(0.0, w_)), t(std::max(0.0, t_)),
          short_level(3.0 * kRangeSigma + 0.5 * t), count_level(3.0 * kRangeSigma + 0.5 * t)
    {
        const int n = 1 + static_cast<int>(std::ceil(t / (0.5 * kRangeSigma)));
        for (int k = 0; k < n; ++k)
            leaf_depths.push_back(n == 1 ? 0.0 : -0.5 * t + t * double(k) / double(n - 1));
    }

    // Per-ray quantities shared by every branch (open, shut at each δ).
    struct Terms { double log_in, log_out, log_wall, log_through, log_clutter, sd2; };
    Terms terms(const UD& a, const Spread& sp) const
    {
        const double sd2 = kRangeSigma * kRangeSigma + sp.sd2;               // depth: range noise + pose spread
        const double sd  = std::sqrt(sd2);
        const double tau = std::sqrt(kRangeSigma * kRangeSigma + sp.su2);    // jamb blur: beam + pose spread
        const double z   = (0.5 * w - std::hypot(a.u, tau)) / tau;           // |u| blurred by τ
        // Occluder: the return is anywhere between the origin and the near face, uniform in depth (soft edges
        // at the depth spread). Background: anywhere in range.
        const double near = -0.5 * t;
        const double log_occ = log_sigmoid((near - a.d) / sd) + log_sigmoid((a.d - a.d_o) / sd)
                               - std::log(std::hypot(near - a.d_o, sd));
        const double log_bg  = -std::log(kMaxRange);
        return {log_sigmoid(z), log_sigmoid(-z),
                log_normal(a.d - near, sd2),                                  // stops on the NEAR face
                log_sigmoid((a.d - 0.5 * t) / sd) - std::log(kMaxRange),     // beyond the far face
                log_add_exp(std::log(kClutter) + log_occ, std::log(kClutter) + log_bg),
                sd2};
    }
    // Every branch: (1 − 2ε)·[m·L_inside + (1 − m)·L_wall] + ε·L_occluder + ε·L_background.
    static double mix(const Terms& k, double log_inside)
    {
        const double core = log_add_exp(k.log_in + log_inside, k.log_out + k.log_wall);
        return log_add_exp(std::log(1.0 - 2.0 * kClutter) + core, k.log_clutter);
    }
    double loglik_open(const Terms& k) const { return mix(k, k.log_through); }
    double loglik_shut(const Terms& k, const UD& a, double delta) const { return mix(k, log_normal(a.d - delta, k.sd2)); }
};

struct Mask
{
    std::vector<Eigen::Vector2d> pts;   // robot frame, x-y
    double sigma2 = 0;                  // σ_m², fixed at the prior (the spread is a property of the mask, not of x)
    bool   valid = false;
    double centroid_u(const V3d& x) const
    {
        const double c = std::cos(x.z()), s = std::sin(x.z());
        double acc = 0;
        for (const auto& p : pts) acc += c * (p.x() - x.x()) + s * (p.y() - x.y());
        return acc / double(pts.size());
    }
    double predictive_var(const V3d& x, const M3d& P) const   // σ_m² + J P Jᵀ
    {
        Eigen::RowVector3d J;
        for (int i = 0; i < 3; ++i) { V3d e = V3d::Zero(); e(i) = 1e-4; J(i) = (centroid_u(x + e) - centroid_u(x - e)) / 2e-4; }
        return sigma2 + J * P * J.transpose();
    }
    // −log Student-t(ū; 0, √var, ν), up to a constant
    static double neg_loglik(double r, double var) { return 0.5 * (kMaskNu + 1.0) * std::log1p(r * r / (kMaskNu * var)); }
};

struct Problem
{
    V3d xbar; M3d Pinv;
    Model model;
    std::vector<Ray2> rays;            // ALL rays: none is ever removed from F
    const Mask* mask = nullptr;
    std::vector<Spread> spread;        // per-ray predictive spread, fixed for one iteration (empty = sensor only)
    double mask_var = 0;

    // Fix the predictive spreads at the linearisation point x under belief P (P = 0: the measurement model alone).
    void linearise(const V3d& x, const M3d& P)
    {
        spread.assign(rays.size(), Spread{});
        if (not P.isZero())
            for (std::size_t i = 0; i < rays.size(); ++i) spread[i] = predictive_spread(rays[i], x, P);
        if (mask != nullptr and mask->valid) mask_var = mask->predictive_var(x, P);
    }

    double F(const V3d& x) const
    {
        V3d dx = x - xbar;
        dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
        double f = 0.5 * dx.dot(Pinv * dx);
        const std::size_t nd = model.leaf_depths.size();
        double l_open = 0;
        std::vector<double> l_leaf(nd, 0.0);
        for (std::size_t i = 0; i < rays.size(); ++i)
        {
            const UD a = to_anchor(rays[i], x);
            const auto k = model.terms(a, spread[i]);
            l_open += model.loglik_open(k);
            for (std::size_t j = 0; j < nd; ++j) l_leaf[j] += model.loglik_shut(k, a, model.leaf_depths[j]);
        }
        double l_shut = -std::numeric_limits<double>::infinity();   // E_δ over the uniform grid
        for (const double l : l_leaf) l_shut = log_add_exp(l_shut, l - std::log(double(nd)));
        // The door is open or shut for the WHOLE frame: marginalise that latent at the frame level.
        if (not rays.empty())
            f -= log_add_exp(std::log(kPriorOpen) + l_open, std::log(1.0 - kPriorOpen) + l_shut);
        if (mask != nullptr and mask->valid) f += Mask::neg_loglik(mask->centroid_u(x), mask_var);
        return f;
    }
    // Central finite differences. Steps: 1 mm / 1 mrad — well below τ = 5 cm, the shortest scale in F.
    void derivatives(const V3d& x, V3d& g, M3d& H) const
    {
        const V3d h(1e-3, 1e-3, 1e-3);
        const double f0 = F(x);
        for (int i = 0; i < 3; ++i)
        {
            V3d ei = V3d::Zero(); ei(i) = h(i);
            const double fp = F(x + ei), fm = F(x - ei);
            g(i) = (fp - fm) / (2.0 * h(i));
            H(i, i) = (fp - 2.0 * f0 + fm) / (h(i) * h(i));
            for (int j = 0; j < i; ++j)
            {
                V3d ej = V3d::Zero(); ej(j) = h(j);
                H(i, j) = H(j, i) = (F(x + ei + ej) - F(x + ei - ej) - F(x - ei + ej) + F(x - ei - ej)) / (4.0 * h(i) * h(j));
            }
        }
    }
};
// The data's information I_L = H − P̄⁻¹, projected onto its positive-semidefinite part. Along a flat (shut) wall the
// exact curvature of the data term is ~0 and can come out slightly NEGATIVE (the soft jamb edges); taken at face
// value that makes the Newton step a jump of metres and a "belief" WIDER than the prior (σ_lat 0.5 → 2.2 m), which
// then widened the next stage's predictive spread. A Gaussian measurement cannot add uncertainty: negative
// directions carry no information. (This is the projection that keeps the Laplace belief a proper Gaussian, not a
// tuned floor.)
inline M3d psd_information(const M3d& H, const M3d& Pinv)
{
    Eigen::SelfAdjointEigenSolver<M3d> es(0.5 * ((H - Pinv) + (H - Pinv).transpose()));
    return es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() * es.eigenvectors().transpose();
}
}  // namespace anchor_detail

// One free-energy minimisation: prior = the predicted RelPose; likelihood = rays (+ mask). Levenberg-Marquardt on
// (x, y, yaw) against the predictive spread (two stages); Laplace covariance at the sensor's σ with the frame's
// common mode marginalised (Woodbury).
inline std::pair<rc::interaction::RelPose, AnchorEvidence>
update_door_anchor(const rc::interaction::RelPose& prior, float width, float thickness,
                   const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                   const std::vector<Eigen::Vector3f>& mask_pts_robot)
{
    using namespace anchor_detail;
    AnchorEvidence ev;

    // Mask: the spread needs two points; one point has no measured σ and claims nothing.
    Mask mask;
    if (mask_pts_robot.size() >= 2)
    {
        for (const auto& p : mask_pts_robot) mask.pts.push_back(p.head<2>().cast<double>());
        const V3d x0 = prior.x.cast<double>();
        const double c = std::cos(x0.z()), s = std::sin(x0.z());
        double mu = 0, m2 = 0;
        for (const auto& q : mask.pts) mu += c * (q.x() - x0.x()) + s * (q.y() - x0.y());
        mu /= double(mask.pts.size());
        for (const auto& q : mask.pts) { const double du = c * (q.x() - x0.x()) + s * (q.y() - x0.y()) - mu; m2 += du * du; }
        const double spread2 = m2 / double(mask.pts.size() - 1);
        mask.sigma2 = spread2 / double(std::min<std::size_t>(mask.pts.size(), kMaskNCap));
        mask.valid = mask.sigma2 > 0;   // all points at one abscissa: no spread ⇒ no measured σ
    }

    Problem pb{prior.x.cast<double>(), prior.P.cast<double>().inverse(), Model(width, thickness), {}, &mask, {}, 0};
    const Eigen::Vector2d o = lidar_origin_robot.head<2>().cast<double>();
    pb.rays.reserve(rays_robot.size());
    for (const auto& p : rays_robot)
        if (const Eigen::Vector2d e = p.head<2>().cast<double>(); (e - o).squaredNorm() > 1e-12)   // a zero-length ray has no direction
            pb.rays.push_back({o, e});

    if (pb.rays.empty() and not mask.valid)
        return {prior, ev};   // nothing measured: the prior IS the posterior, bit for bit

    V3d x = pb.xbar;
    M3d P = prior.P.cast<double>();
    M3d H = pb.Pinv;
    V3d g;
    // ─── two STAGES of predictive spread, then the sensor alone ─────────────────────────────────────────────
    // Each stage scores the rays against the spread a BELIEF predicts for them (the Kalman S = HPHᵀ + R): stage 1
    // the prior's, stage 2 the Laplace belief stage 1 converged to. From a prior 0.35 m off in depth the wall
    // returns are then a broad pull rather than 7σ outliers the occluder component would absorb, and a 0.45 m
    // lateral error sits inside the jamb's blurred reach. The spread is updated only at a stage's OPTIMUM — the
    // curvature at a point still 0.5 m off is not a belief, and using it collapsed the reach mid-way (dy = +0.35:
    // σ_lat 0.5 → 0.12 after one step, anchor stranded 0.66 m along the wall). The final Laplace (below) is at the
    // sensor's own σ. Per stage: at most kMaxIter accepted Levenberg-Marquardt steps.
    for (int stage = 0; stage < 2; ++stage)
    {
        pb.linearise(x, P);
        // Levenberg-Marquardt in the PRIOR's metric (damping μ P̄⁻¹), steps accepted on the gain ratio ρ = actual /
        // predicted decrease (Nielsen's μ update). ★A plain Newton step + Armijo line search failed here: along a
        // shut wall the lateral direction has only the prior's curvature, so the step's lateral part was metres long,
        // and the large depth/yaw gain in the same step made the line search ACCEPT it (dy = +0.35: the anchor was
        // dragged 1.3 m along the wall). The gain ratio checks the quadratic model's promise per step, so a
        // direction the model mispredicts is shrunk instead of carried along. (ρ > 0, ×2, ⅓ are the textbook
        // optimiser constants, not model levels.)
        double mu = 1e-2, nu = 2.0;
        for (int it = 0; it < kMaxIter; ++it)
        {
            pb.derivatives(x, g, H);
            const M3d Hp = pb.Pinv + psd_information(H, pb.Pinv);   // positive definite by construction
            const double f0 = pb.F(x);
            bool moved = false;
            for (int k = 0; k < 30 and not moved; ++k)
            {
                const V3d step = -(Hp + mu * pb.Pinv).llt().solve(g);
                const double predicted = -(g.dot(step) + 0.5 * step.dot(Hp * step));
                const double actual    = f0 - pb.F(x + step);
                if (predicted > 0 and actual > 0)
                {
                    const double rho = actual / predicted;
                    x += step;
                    mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                    nu = 2.0;
                    moved = true;
                }
                else { mu *= nu; nu *= 2.0; }
            }
            x.z() = std::remainder(x.z(), 2.0 * M_PI);
            if (not moved) break;   // no step decreases F: at this stage's optimum (to the optimiser's resolution)
        }
        pb.derivatives(x, g, H);
        P = (pb.Pinv + psd_information(H, pb.Pinv)).llt().solve(M3d::Identity());
    }

    // ─── Laplace at the measurement model's own σ, with the frame's common mode marginalised ─────────────────
    // H = P̄⁻¹ + I_L (I_L = the data's curvature, PSD part). The rays share one LiDAR extrinsic and one wall, so their
    // lateral/depth evidence carries a common offset b ~ N(0, C), C = σ_c² on (x, y): the data then inform the
    // anchor through I_eff = (I_L⁻¹ + C)⁻¹ = (I + I_L C)⁻¹ I_L (Woodbury: no inverse of I_L, which is singular
    // along a shut wall). Without it a frame of ~900 rays claimed a 2 mm depth σ. The mean is the same Gaussian
    // algebra: x = x̄ + P_post (I + I_L C)⁻¹ I_L (x_L − x̄), with I_L (x_L − x̄) = H (x − x̄) − g (one Newton step
    // on the true F, which also absorbs the iteration's residual gradient).
    pb.linearise(x, M3d::Zero());
    pb.derivatives(x, g, H);
    V3d dx = x - pb.xbar;
    dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
    const M3d I_L = psd_information(H, pb.Pinv);
    H = pb.Pinv + I_L;
    if (pb.F(x + H.llt().solve(-g)) > pb.F(x))
        g.setZero();   // the true-F Newton step would not descend: x already is its optimum to our resolution
    const M3d C   = V3d(kCommonSigma * kCommonSigma, kCommonSigma * kCommonSigma, 0.0).asDiagonal();
    const M3d A   = M3d::Identity() + I_L * C;
    const Eigen::PartialPivLU<M3d> Alu(A);
    M3d I_eff = Alu.solve(I_L);
    I_eff = 0.5 * (I_eff + I_eff.transpose());

    rc::interaction::RelPose post = prior;
    if (Eigen::LLT<M3d> llt(pb.Pinv + I_eff); llt.info() == Eigen::Success)
    {
        const M3d S = llt.solve(M3d::Identity());
        V3d xp = pb.xbar + S * Alu.solve(H * dx - g);
        xp.z() = std::remainder(xp.z(), 2.0 * M_PI);
        post.x = xp.cast<float>();
        post.P = (0.5 * (S + S.transpose())).cast<float>();
    }
    else
    {
        // Not positive definite: the data did not localise the anchor here — keep the last iteration's belief.
        post.x = x.cast<float>();
        post.P = P.cast<float>();
    }
    x = post.x.cast<double>();

    // ─── Evidence at the posterior mean (COUNTING ONLY — see the header) ─────────────────────────────────────
    for (const auto& r : pb.rays)
    {
        const UD a = to_anchor(r, x);
        if (a.d < -pb.model.short_level) { ++ev.n_short; continue; }   // ended in front of the near face
        if (std::abs(a.u) >= 0.5 * pb.model.w) continue;               // crosses outside the span
        if (a.d > pb.model.count_level) ++ev.n_through;
        else ++ev.n_blocked;                                            // |d| ≤ 3σ + t/2
    }
    ev.p_open = float(ev.n_through + 1) / float(ev.n_through + ev.n_blocked + 2);
    if (mask.valid)
    {
        const double r = mask.centroid_u(x);
        ev.camera_nis = float(r * r / mask.sigma2);
    }
    return {post, ev};
}

}  // namespace rc::door
