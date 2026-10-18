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
 * Free energy: F(x) = ½ (x − x̄)ᵀ P̄⁻¹ (x − x̄) − log p(rays | x) − log p(mask | x). Newton on (x, y, yaw), at
 * most 5 iterations, with a line search over halvings of the step (the logistic tails are almost flat, so
 * the raw Newton step from 0.45 m off overshoots by metres). The derivatives are central finite differences
 * of F in double precision: F is smooth by construction, and the EXACT Hessian is what the Laplace covariance
 * needs — a Gauss-Newton/Fisher approximation counts every jamb-adjacent wall ray as lateral information and
 * would make a shut, featureless wall look like it localises the door along itself.
 * Covariance = (∇²F at the optimum)⁻¹.
 *
 * ─── THE ONLY DECISION LEVELS (deliberate, not model terms) ──────────────────────────────────────────
 *   1. SHORT rays — endpoint d < −(3σ + t/2), i.e. ended in front of the wall's near face beyond range noise:
 *      they hit something in front of the doorway (the leaf swung toward the robot, a person) and tested
 *      nothing about the opening. Same rule and level as select_doorway_rays (2026-10-18). Dropped, n_short.
 *   2. COUNTING for p_open — rays crossing inside the span (|u| < w/2) at the optimum are THROUGH if
 *      d > 3σ + t/2, BLOCKED if |d| ≤ 3σ + t/2. This is a report (a Beta(1,1) posterior on "rays pass"), it
 *      never feeds back into the pose.
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
constexpr double kRangeSigma = 0.05;   // beam range noise (m), as in door_phi_posterior.h
constexpr double kMaxRange   = 8.0;    // the scan's max range (m), as in select_doorway_rays
constexpr double kMaskNu     = 3.0;    // Student-t dof of the mask centroid
constexpr int    kMaskNCap   = 30;     // correlated mask pixels: √N gain capped here
constexpr int    kMaxIter    = 5;
constexpr double kPriorOpen  = 0.5;    // P(door open) before this frame: uninformed (the interface carries no leaf belief)

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

// A 2-D ray in ROBOT coordinates.
struct Ray2 { Eigen::Vector2d o, e; };

// The ray in anchor coordinates: along-wall crossing u, signed depth d (along the ray's travel).
struct UD { double u, d; };
inline UD to_anchor(const Ray2& r, const V3d& x)
{
    const double c = std::cos(x.z()), s = std::sin(x.z());
    const Eigen::Vector2d t(x.x(), x.y());
    auto inv = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - t; return Eigen::Vector2d(c * q.x() + s * q.y(), -s * q.x() + c * q.y()); };
    const Eigen::Vector2d oa = inv(r.o), ea = inv(r.e);
    const double sgn = oa.y() <= 0 ? 1.0 : -1.0;   // which side the beam starts on: its travel is "past the plane"
    const double dy = ea.y() - oa.y();
    // Where the ray's LINE crosses y = 0. A ray parallel to the wall never crosses it: put its crossing at
    // infinity along the wall, where m = 0 and only the wall term (depth) speaks.
    const double u = std::abs(dy) > 1e-12 ? oa.x() - oa.y() * (ea.x() - oa.x()) / dy
                                          : std::copysign(std::numeric_limits<double>::max(), ea.x() - oa.x());
    return {u, sgn * ea.y()};
}

struct Model
{
    double w, t;                       // opening width, wall thickness (m)
    double sw2;                        // wall-term variance: σ²
    std::vector<double> leaf_depths;   // the δ grid
    double short_level, count_level;   // both 3σ + t/2 (see the header: the only decision levels)
    explicit Model(double w_, double t_)
        : w(std::max(0.0, w_)), t(std::max(0.0, t_)),
          sw2(kRangeSigma * kRangeSigma),
          short_level(3.0 * kRangeSigma + 0.5 * t), count_level(3.0 * kRangeSigma + 0.5 * t)
    {
        const int n = 1 + static_cast<int>(std::ceil(t / (0.5 * kRangeSigma)));
        for (int k = 0; k < n; ++k)
            leaf_depths.push_back(n == 1 ? 0.0 : -0.5 * t + t * double(k) / double(n - 1));
    }

    double log_m(const UD& a, bool inside) const
    {
        const double tau = kRangeSigma;
        const double z   = (0.5 * w - std::hypot(a.u, tau)) / tau;    // |u| blurred by τ
        return inside ? log_sigmoid(z) : log_sigmoid(-z);
    }
    double log_wall(const UD& a) const
    {
        const double r = a.d + 0.5 * t;                                // from the near face
        return -0.5 * r * r / sw2 - 0.5 * std::log(2.0 * M_PI * sw2);
    }
    // log p(ray | x, door OPEN), up to constants common to every x
    double ray_loglik_open(const UD& a) const
    {
        const double log_through = log_sigmoid((a.d - 0.5 * t) / kRangeSigma) - std::log(kMaxRange);
        return log_add_exp(log_m(a, true) + log_through, log_m(a, false) + log_wall(a));
    }
    // log p(ray | x, door SHUT, leaf at depth delta)
    double ray_loglik_shut(const UD& a, double delta) const
    {
        const double r = a.d - delta;
        const double log_leaf = -0.5 * r * r / sw2 - 0.5 * std::log(2.0 * M_PI * sw2);
        return log_add_exp(log_m(a, true) + log_leaf, log_m(a, false) + log_wall(a));
    }
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
    // −log Student-t(ū; 0, σ_m, ν), up to a constant
    double neg_loglik(const V3d& x) const
    {
        const double r = centroid_u(x);
        return 0.5 * (kMaskNu + 1.0) * std::log1p(r * r / (kMaskNu * sigma2));
    }
};

struct Problem
{
    V3d xbar; M3d Pinv;
    Model model;
    std::vector<Ray2> rays;            // the informative set, chosen at the current linearisation point
    const Mask* mask = nullptr;

    double F(const V3d& x) const
    {
        V3d dx = x - xbar;
        dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
        double f = 0.5 * dx.dot(Pinv * dx);
        const std::size_t nd = model.leaf_depths.size();
        double l_open = 0;
        std::vector<double> l_leaf(nd, 0.0);
        for (const auto& r : rays)
        {
            const UD a = to_anchor(r, x);
            l_open += model.ray_loglik_open(a);
            for (std::size_t k = 0; k < nd; ++k) l_leaf[k] += model.ray_loglik_shut(a, model.leaf_depths[k]);
        }
        double l_shut = -std::numeric_limits<double>::infinity();   // E_δ over the uniform grid
        for (const double l : l_leaf) l_shut = log_add_exp(l_shut, l - std::log(double(nd)));
        // The door is open or shut for the WHOLE frame: marginalise that latent at the frame level.
        f -= log_add_exp(std::log(kPriorOpen) + l_open, std::log(1.0 - kPriorOpen) + l_shut);
        if (mask != nullptr and mask->valid) f += mask->neg_loglik(x);
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

// Partition all rays at x: informative (kept) vs short.
inline int select_informative(const std::vector<Ray2>& all, const V3d& x, const Model& m, std::vector<Ray2>& kept)
{
    kept.clear();
    int n_short = 0;
    for (const auto& r : all)
    {
        // DECISION LEVEL 1 (see the header): ended in front of the near face, beyond range noise.
        if (to_anchor(r, x).d < -m.short_level) { ++n_short; continue; }
        kept.push_back(r);
    }
    return n_short;
}
}  // namespace anchor_detail

// One free-energy minimisation: prior = the predicted RelPose; likelihood = rays (+ mask). Newton on (x, y, yaw)
// with exact (finite-difference) curvature, Laplace covariance.
inline std::pair<rc::interaction::RelPose, AnchorEvidence>
update_door_anchor(const rc::interaction::RelPose& prior, float width, float thickness,
                   const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                   const std::vector<Eigen::Vector3f>& mask_pts_robot)
{
    using namespace anchor_detail;
    AnchorEvidence ev;

    std::vector<Ray2> all;
    all.reserve(rays_robot.size());
    const Eigen::Vector2d o = lidar_origin_robot.head<2>().cast<double>();
    for (const auto& p : rays_robot)
        if (const Eigen::Vector2d e = p.head<2>().cast<double>(); (e - o).squaredNorm() > 1e-12)   // a zero-length ray has no direction
            all.push_back({o, e});

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

    Problem pb{prior.x.cast<double>(), prior.P.cast<double>().inverse(), Model(width, thickness), {}, &mask};
    V3d x = pb.xbar;
    ev.n_short = select_informative(all, x, pb.model, pb.rays);

    if (pb.rays.empty() and not mask.valid)
        return {prior, ev};   // nothing measured: the prior IS the posterior, bit for bit

    M3d H = pb.Pinv;
    for (int it = 0; it < kMaxIter; ++it)
    {
        V3d g; pb.derivatives(x, g, H);
        // Newton direction; where F is locally concave (a logistic tail, the t's tail) fall back to the prior's
        // curvature — the step is then a preconditioned gradient step, and the line search sizes it.
        Eigen::LLT<M3d> llt(H);
        const V3d dir = llt.info() == Eigen::Success ? V3d(-llt.solve(g)) : V3d(-pb.Pinv.ldlt().solve(g));
        // Line search over halvings: the best of {1, 1/2, …, 2^-30} × dir (F is cheap; the tails make the full
        // step overshoot by metres from a 0.45 m error).
        const double f0 = pb.F(x);
        double best_f = f0; V3d best_x = x;
        for (int k = 0; k <= 30; ++k)
        {
            const V3d xt = x + std::ldexp(1.0, -k) * dir;
            if (const double ft = pb.F(xt); ft < best_f) { best_f = ft; best_x = xt; }
        }
        if (best_x == x) break;   // no descent along the direction: at the optimum (to the line search's resolution)
        x = best_x;
        x.z() = std::remainder(x.z(), 2.0 * M_PI);
        ev.n_short = select_informative(all, x, pb.model, pb.rays);
    }
    V3d g; pb.derivatives(x, g, H);

    rc::interaction::RelPose post = prior;
    post.x = x.cast<float>();
    // Laplace: Σ = (∇²F)⁻¹. At a minimum ∇²F is positive definite; if the iteration budget ended on a
    // non-convex patch it is not, and the honest statement is that the data did not localise the anchor
    // there — keep the prior's covariance.
    if (Eigen::LLT<M3d> llt(0.5 * (H + H.transpose())); llt.info() == Eigen::Success)
    {
        M3d S = llt.solve(M3d::Identity());
        post.P = (0.5 * (S + S.transpose())).cast<float>();
    }

    // Evidence at the optimum.
    for (const auto& r : pb.rays)
    {
        const UD a = to_anchor(r, x);
        if (std::abs(a.u) >= 0.5 * pb.model.w) continue;            // crosses outside the span
        // DECISION LEVEL 2 (see the header): counting only.
        if (a.d > pb.model.count_level) ++ev.n_through;
        else if (std::abs(a.d) <= pb.model.count_level) ++ev.n_blocked;
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
