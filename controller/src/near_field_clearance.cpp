// near_field_clearance.cpp — see the header for the model. Values carry their exact gradient (forward-mode duals), so the
// gradient handed to the optimiser IS the derivative of the value it scores — no finite differences, no second model.
#include "near_field_clearance.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <limits>
#include <optional>
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

namespace
{
double segment_distance(const NearFieldElement &e, const Eigen::Vector2d &p)
{
    const Eigen::Vector2d ab = e.b - e.a;
    const double L2 = ab.squaredNorm();
    const double t = L2 > 0.0 ? std::clamp((p - e.a).dot(ab) / L2, 0.0, 1.0) : 0.0;
    return (e.a + t * ab - p).norm();
}
// the largest eigenvalue of the (symmetric) position covariance: the widest the element's distance can spread
double s2_max(const Eigen::Matrix2d &S)
{
    const double h = 0.5 * (S(0, 0) - S(1, 1));
    return 0.5 * S.trace() + std::sqrt(h * h + S(0, 1) * S(0, 1));
}
}  // namespace

bool NearFieldClearance::covered(const NearFieldElement &e) const { return segment_distance(e, centre_) <= R_; }

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
        if (not covered(e)) continue;   // the coverage disc defines the near model
        (e.p_exist >= 1.0 ? surfaces_ : uncertain).push_back(e);
    }
    // certain surfaces first: an uncertain element enters as a mixture ON the running minimum
    surfaces_.insert(surfaces_.end(), uncertain.begin(), uncertain.end());
}

void NearFieldClearance::set_model(std::span<const NearFieldElement> model)
{
    memo_valid_ = false;
    model_.clear();
    for (const auto &e : model)
        if (e.p_exist >= 1.0 and covered(e)) model_.push_back(e);
}

std::vector<NearFieldElement> NearFieldClearance::polygon_elements(std::span<const Eigen::Vector2f> poly, double sigma,
                                                                   float spacing, const std::function<bool(const Eigen::Vector2f &)> &keep)
{
    std::vector<NearFieldElement> out;
    const int n = static_cast<int>(poly.size());
    if (n < 2 or not (spacing > 0.f)) return out;
    Eigen::Matrix2d S = sigma * sigma * Eigen::Matrix2d::Identity();
    auto emit = [&](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
    {
        NearFieldElement e;
        e.a = a.cast<double>(); e.b = b.cast<double>(); e.S = S;
        out.push_back(e);
    };
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f &a = poly[i];
        const Eigen::Vector2f &b = poly[(i + 1) % n];
        // the samples TrajectoryController puts in the ESDF (set_room_boundary / set_static_obstacles)
        const int num = std::max(1, static_cast<int>((b - a).norm() / spacing));
        std::optional<Eigen::Vector2f> run_start, run_end;
        for (int k = 0; k <= num; ++k)
        {
            const Eigen::Vector2f q = a + (static_cast<float>(k) / static_cast<float>(num)) * (b - a);
            if (not keep or keep(q))
            {
                if (not run_start) run_start = q;
                run_end = q;
            }
            else if (run_start)
            {
                emit(*run_start, *run_end);
                run_start.reset();
            }
        }
        if (run_start) emit(*run_start, *run_end);
    }
    return out;
}

std::vector<NearFieldElement> NearFieldClearance::jamb_elements(const policy::DoorwayJambs &j, double wall_half_depth_m)
{
    const Eigen::Vector2d n = j.gap.normal.cast<double>().normalized(), lat(-n.y(), n.x());
    const Eigen::Vector2d c = j.gap.centre.cast<double>();
    const double hw = 0.5 * j.gap.width, L = j.placement_m, t = std::max(0.0, wall_half_depth_m);
    const double s_lat2 = double(j.sigma_lat_m) * j.sigma_lat_m + double(j.sigma_s_m) * j.sigma_s_m
                        + double(j.sigma_half_width_m) * j.sigma_half_width_m;
    const double s_n2 = double(j.sigma_n_m) * j.sigma_n_m + double(j.sigma_s_m) * j.sigma_s_m;
    const Eigen::Matrix2d S = s_lat2 * lat * lat.transpose() + s_n2 * n * n.transpose();
    std::vector<NearFieldElement> out;
    auto seg = [&](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { NearFieldElement e; e.a = a; e.b = b; e.S = S; out.push_back(e); };
    for (const double side : {1.0, -1.0})
    {
        const Eigen::Vector2d edge = c + side * hw * lat, outer = c + side * (hw + L) * lat;
        if (t > 0.0)
        {
            seg(edge - t * n, edge + t * n);              // the jamb's FACE, along the normal, through the wall's depth
            seg(edge - t * n, outer - t * n);             // the wall's face on the near side
            seg(edge + t * n, outer + t * n);             // and on the far side
        }
        else seg(edge, outer);                            // a thin wall: its line
    }
    return out;
}

void NearFieldClearance::set_doorway(const policy::DoorwayRisk &risk, double wall_half_depth_m)
{
    memo_valid_ = false;
    anchor_jambs_ = jamb_elements(risk.anchor, wall_half_depth_m);
    gap_jambs_.clear();
    p_gap_ = 0.0;
    if (risk.gap.has_value() and risk.p_gap > 0.f)
    {
        gap_jambs_ = jamb_elements(*risk.gap, wall_half_depth_m);
        p_gap_ = std::clamp(double(risk.p_gap), 0.0, 1.0);
    }
}

NearFieldEval NearFieldClearance::evaluate_hypothesis(const Eigen::Vector2d &p, std::span<const NearFieldElement> extra,
                                                      double far_v, const Eigen::Vector2d &far_g) const
{
    const Dual px{p.x(), Eigen::Vector2d(1.0, 0.0)}, py{p.y(), Eigen::Vector2d(0.0, 1.0)};
    // ★ PREFILTER (numerical, not a model cut). Plain-double distances first: an element whose distance exceeds the nearest
    //   certain one's by more than 8.5 sqrt(2) s_max (s_max^2 = the widest variance any element here has) is below the
    //   minimum with probability < Phi(-8.5) ~ 1e-17 — its fold cannot move it. The dual work is done only for the elements
    //   that can matter. The fold ORDER stays the input order (Clark's sequential approximation is order-dependent: folding
    //   nearest-first made the value jump where the nearest element changes — measured, |grad - FD| 0.012).
    const std::array<std::span<const NearFieldElement>, 3> sets{extra, std::span<const NearFieldElement>(model_),
                                                                std::span<const NearFieldElement>(surfaces_)};
    double mu_min = std::numeric_limits<double>::infinity(), s2max = 0.0;
    for (const auto &set : sets)
        for (const auto &e : set)
        {
            s2max = std::max(s2max, s2_max(e.S));
            if (e.p_exist < 1.0) continue;
            mu_min = std::min(mu_min, segment_distance(e, p));
        }
    const double reach = mu_min + 8.5 * std::sqrt(2.0 * s2max);
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
    // jambs and model walls (certain) before the surfaces, whose uncertain tail folds onto a running minimum
    for (const auto &set : sets)
        for (const auto &e : set)
            if (segment_distance(e, p) <= reach) fold(e);

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
