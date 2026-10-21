// near_field_clearance.cpp — see the header for the model. Values carry their exact gradient (forward-mode duals), so the
// gradient handed to the optimiser IS the derivative of the value it scores — no finite differences, no second model.
#include "near_field_clearance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace rc
{
namespace
{
// A value and its gradient with respect to the query point.
struct Dual
{
    double v = 0.0;
    Eigen::Vector2d g = Eigen::Vector2d::Zero();
};
Dual operator+(const Dual &a, const Dual &b) { return {a.v + b.v, a.g + b.g}; }
Dual operator-(const Dual &a, const Dual &b) { return {a.v - b.v, a.g - b.g}; }
Dual operator*(const Dual &a, const Dual &b) { return {a.v * b.v, a.g * b.v + b.g * a.v}; }
Dual operator*(double k, const Dual &a) { return {k * a.v, k * a.g}; }
Dual operator+(double k, const Dual &a) { return {k + a.v, a.g}; }
Dual operator/(const Dual &a, const Dual &b) { return {a.v / b.v, (a.g * b.v - b.g * a.v) / (b.v * b.v)}; }
Dual dsqrt(const Dual &a)
{
    const double s = std::sqrt(a.v);
    return {s, s > 0.0 ? Eigen::Vector2d(a.g / (2.0 * s)) : Eigen::Vector2d::Zero()};
}
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
double pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }
// Phi(x), accurate in both tails
Dual dPhi(const Dual &x) { return {0.5 * std::erfc(-x.v * kInvSqrt2), pdf(x.v) * x.g}; }
Dual dphi(const Dual &x) { const double f = pdf(x.v); return {f, -x.v * f * x.g}; }

// D_j at p: the distance to the element's segment (mean) and its variance along the clearance direction
struct ElementMoments { Dual mu, s2; };
ElementMoments element_moments(const NearFieldElement &e, const Dual &px, const Dual &py)
{
    const Eigen::Vector2d ab = e.b - e.a;
    const double L2 = ab.squaredNorm();
    Dual t{0.0, Eigen::Vector2d::Zero()};
    if (L2 > 0.0)
    {
        const double tv = ((Eigen::Vector2d(px.v, py.v) - e.a).dot(ab)) / L2;
        if (tv >= 1.0) t = {1.0, Eigen::Vector2d::Zero()};
        else if (tv > 0.0) t = {tv, ab / L2};
    }
    const Dual dx = px - Dual{e.a.x(), Eigen::Vector2d::Zero()} - ab.x() * t;
    const Dual dy = py - Dual{e.a.y(), Eigen::Vector2d::Zero()} - ab.y() * t;
    const Dual mu = dsqrt(dx * dx + dy * dy);
    if (not (mu.v > 0.0))   // p ON the nominal surface: no clearance direction — the variance's isotropic part
        return {{0.0, Eigen::Vector2d::Zero()}, {0.5 * e.S.trace(), Eigen::Vector2d::Zero()}};
    const Dual nx = dx / mu, ny = dy / mu;
    const Dual s2 = e.S(0, 0) * (nx * nx) + 2.0 * e.S(0, 1) * (nx * ny) + e.S(1, 1) * (ny * ny);
    return {mu, s2};
}
}  // namespace

struct NearFieldClearance::Moments
{
    bool any = false;   // false: no certain surface folded yet (E[M] = +inf)
    Dual m, v;
};

namespace
{
// Clark's moment-matched min of the running M ~ N(m1, v1) and D ~ N(m2, v2), independent; with existence p the result is
// the mixture p min(M, D) + (1 - p) M.
void clark_fold(Dual &m1, Dual &v1, const Dual &m2, const Dual &v2, double p)
{
    const Dual th = dsqrt(v1 + v2);
    if (not (th.v > 0.0))   // two certain distances: the plain min (only reachable with a zero-sigma model)
    {
        if (m2.v < m1.v and p >= 1.0) { m1 = m2; v1 = v2; }
        return;
    }
    const Dual a = (m1 - m2) / th;
    // ★ NUMERICAL, not a model cut: at a <= -8.5 the new element is below the running minimum with probability
    //   Phi(-8.5) ~ 1e-17 — its fold changes nothing at double precision.
    if (a.v <= -8.5) return;
    const Dual Pa = dPhi(a), Pna = dPhi(-1.0 * a), ph = dphi(a);
    const Dual e1 = m1 * Pna + m2 * Pa - th * ph;
    const Dual e2 = (m1 * m1 + v1) * Pna + (m2 * m2 + v2) * Pa - (m1 + m2) * th * ph;
    Dual nm = e1, n2 = e2;
    if (p < 1.0)
    {
        nm = p * e1 + (1.0 - p) * m1;
        n2 = p * e2 + (1.0 - p) * (m1 * m1 + v1);
    }
    Dual nv = n2 - nm * nm;
    if (not (nv.v > 0.0)) nv = {0.0, Eigen::Vector2d::Zero()};   // rounding of a difference of near-equal moments
    m1 = nm;
    v1 = nv;
}
}  // namespace

NearFieldClearance::NearFieldClearance(const Eigen::Vector2f &centre, float near_radius_m, float far_sigma_m,
                                       DistFn far_distance, GradFn far_gradient)
    : centre_(centre.cast<double>()), R_(near_radius_m), sigma_far_(far_sigma_m),
      far_d_(std::move(far_distance)), far_g_(std::move(far_gradient))
{}

NearFieldElement NearFieldClearance::element_of(const policy::Return &r)
{
    NearFieldElement e;
    e.a = (r.p - r.seg).cast<double>();
    e.b = (r.p + r.seg).cast<double>();
    const double s2 = double(r.sigma) * r.sigma;
    e.S << s2 + r.sxx, r.sxy, r.sxy, s2 + r.syy;
    e.p_exist = r.p_exist;
    return e;
}

void NearFieldClearance::set_surfaces(std::span<const policy::Return> denoised)
{
    memo_valid_ = false;
    surfaces_.clear();
    surfaces_.reserve(denoised.size());
    std::vector<NearFieldElement> uncertain;
    for (const auto &r : denoised)
    {
        if (not r.p.allFinite() or not (r.p_exist > 0.f)) continue;
        NearFieldElement e = element_of(r);
        // in the coverage disc iff the segment's nearest point to the centre is
        const Eigen::Vector2d ab = e.b - e.a;
        const double L2 = ab.squaredNorm();
        const double t = L2 > 0.0 ? std::clamp((centre_ - e.a).dot(ab) / L2, 0.0, 1.0) : 0.0;
        if ((e.a + t * ab - centre_).norm() > R_) continue;
        (e.p_exist >= 1.0 ? surfaces_ : uncertain).push_back(e);
    }
    // certain surfaces first: an uncertain element enters as a mixture ON the running minimum
    surfaces_.insert(surfaces_.end(), uncertain.begin(), uncertain.end());
}

std::vector<NearFieldElement> NearFieldClearance::jamb_elements(const policy::DoorwayJambs &j)
{
    const Eigen::Vector2d n = j.gap.normal.cast<double>().normalized(), lat(-n.y(), n.x());
    const Eigen::Vector2d c = j.gap.centre.cast<double>();
    const double hw = 0.5 * j.gap.width, L = j.placement_m;
    const double s_lat2 = double(j.sigma_lat_m) * j.sigma_lat_m + double(j.sigma_s_m) * j.sigma_s_m
                        + double(j.sigma_half_width_m) * j.sigma_half_width_m;
    const double s_n2 = double(j.sigma_n_m) * j.sigma_n_m + double(j.sigma_s_m) * j.sigma_s_m;
    const Eigen::Matrix2d S = s_lat2 * lat * lat.transpose() + s_n2 * n * n.transpose();
    NearFieldElement left, right;
    left.a = c + hw * lat;  left.b = c + (hw + L) * lat;  left.S = S;
    right.a = c - hw * lat; right.b = c - (hw + L) * lat; right.S = S;
    return {left, right};
}

void NearFieldClearance::set_doorway(const policy::DoorwayRisk &risk)
{
    memo_valid_ = false;
    anchor_jambs_ = jamb_elements(risk.anchor);
    gap_jambs_.clear();
    p_gap_ = 0.0;
    if (risk.gap.has_value() and risk.p_gap > 0.f)
    {
        gap_jambs_ = jamb_elements(*risk.gap);
        p_gap_ = std::clamp(double(risk.p_gap), 0.0, 1.0);
    }
}

NearFieldEval NearFieldClearance::evaluate_hypothesis(const Eigen::Vector2d &p, std::span<const NearFieldElement> extra,
                                                      double far_v, const Eigen::Vector2d &far_g) const
{
    const Dual px{p.x(), Eigen::Vector2d(1.0, 0.0)}, py{p.y(), Eigen::Vector2d(0.0, 1.0)};
    Moments M;
    auto fold = [&](const NearFieldElement &e)
    {
        const auto [mu, s2] = element_moments(e, px, py);
        if (not M.any)
        {
            if (e.p_exist < 1.0) return;   // an uncertain element on an infinite minimum: E[M] stays +inf (w = 0)
            M = {true, mu, s2};
            return;
        }
        clark_fold(M.m, M.v, mu, s2, e.p_exist);
    };
    // jambs (certain) before the surfaces' uncertain tail, so every uncertain element folds onto a running minimum
    for (const auto &e : extra) fold(e);
    for (const auto &e : surfaces_) fold(e);

    NearFieldEval out;
    out.value = far_v;
    out.gradient = far_g;
    if (not M.any) return out;
    out.near_mean = M.m.v;
    out.near_sigma = std::sqrt(M.v.v);

    const Eigen::Vector2d d = p - centre_;
    const double rn = d.norm();
    const Dual rho{rn, rn > 0.0 ? Eigen::Vector2d(d / rn) : Eigen::Vector2d::Zero()};
    const Dual margin = R_ + (-1.0 * rho) - M.m;
    const Dual sw = dsqrt(M.v + Dual{sigma_far_ * sigma_far_, Eigen::Vector2d::Zero()});
    const Dual w = dPhi(margin / sw);
    const Dual far{far_v, far_g};
    const Dual V = w * M.m + (1.0 + (-1.0 * w)) * far;
    out.value = V.v;
    out.gradient = V.g;
    out.near_weight = w.v;
    return out;
}

NearFieldEval NearFieldClearance::evaluate(const Eigen::Vector2f &pf) const
{
    const Eigen::Vector2d p = pf.cast<double>();
    const double fv = far_d_ ? double(far_d_(pf)) : std::numeric_limits<double>::infinity();
    const Eigen::Vector2d fg = far_g_ ? Eigen::Vector2d(far_g_(pf).cast<double>()) : Eigen::Vector2d::Zero();
    if (p_gap_ > 0.0)
    {
        const NearFieldEval a = evaluate_hypothesis(p, anchor_jambs_, fv, fg);
        const NearFieldEval g = evaluate_hypothesis(p, gap_jambs_, fv, fg);
        NearFieldEval out;
        out.value = p_gap_ * g.value + (1.0 - p_gap_) * a.value;
        out.gradient = p_gap_ * g.gradient + (1.0 - p_gap_) * a.gradient;
        out.near_weight = p_gap_ * g.near_weight + (1.0 - p_gap_) * a.near_weight;
        out.near_mean = a.near_mean;
        out.near_sigma = a.near_sigma;
        return out;
    }
    return evaluate_hypothesis(p, anchor_jambs_, fv, fg);
}
}  // namespace rc
