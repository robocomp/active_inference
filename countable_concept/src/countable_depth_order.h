/*
 * countable_depth_order.h — the depth ordering of a LiDAR return against the object along the camera's line of sight (F1 round 7 B).
 * Pure (Eigen only): the fitter's persistent label uses it (countable_mask_membership), and the replay tool re-labels a dump with it.
 */
#pragma once

#include <cmath>
#include <numbers>

#include <Eigen/Dense>

namespace rc {

// The object's box as the camera's line of sight sees it (F1 round 7 B): the belief's box (w along the local x axis, h along y,
// standing on z = 0, as CountableBelief::beam_part 0) and the variances of its pose and size.
struct LosBox
{
    double cx = 0, cy = 0, H = 1, w = 1, h = 1, yaw = 0;
    double var_cx = 0, var_cy = 0, var_H = 0, var_w = 0, var_h = 0;
};

// P(the LiDAR return p lies BEHIND the object along the camera's line of sight C → p) (F1 round 7 B). The mask's label q is the
// probability that the camera's line of sight at that pixel meets the object; the object then occupies that line from the
// surface the camera sees to the far side, and a return beyond it is a surface the object hides (the wall behind a toilet, P1
// p1_toilet2: 14–21 % of the LiDAR label mass, beside the narrow bowl and inside the blurred mask edge). The object's far side
// along the line is where the line leaves the belief's box, t_e (or, when the line passes beside the box, its closest approach
// t*: the same point in the limit, so the probability is continuous), and
//   P(behind) = Φ((|p − C| − t_e)/σ),   σ² = sigma_m² + Var(face at t_e)
// σ_m the return's and the two sensors' relative position error, Var(face) the belief's uncertainty of the face the line leaves
// by (centre ± half size), blended over the faces near the exit point (see below). No gate: a return at the far face reads ½, one on a visible face ≈ 0, the wall 0.4 m behind ≈ 1.
namespace detail_los {
// the box's signed distance and outward gradient at x (room frame)
struct BoxSdf { double d; Eigen::Vector3d n_local, n_room; };
inline BoxSdf box_sdf(const LosBox& b, const Eigen::Vector3d& x)
{
    const double c = std::cos(b.yaw), s = std::sin(b.yaw);
    const double dx = x.x() - b.cx, dy = x.y() - b.cy;
    const Eigen::Vector3d q(c * dx + s * dy, -s * dx + c * dy, x.z() - 0.5 * b.H);
    const Eigen::Vector3d half(0.5 * b.w, 0.5 * b.h, 0.5 * b.H);
    const Eigen::Vector3d e = q.cwiseAbs() - half;
    BoxSdf o;
    const Eigen::Vector3d out = e.cwiseMax(0.0);
    if (out.norm() > 0.0)
    {
        o.d = out.norm();
        o.n_local = (out / o.d).cwiseProduct(q.cwiseSign());
    }
    else
    {
        int k = 0; e.maxCoeff(&k);
        o.d = e(k);
        o.n_local.setZero(); o.n_local(k) = q(k) >= 0.0 ? 1.0 : -1.0;
    }
    o.n_room = Eigen::Vector3d(c * o.n_local.x() - s * o.n_local.y(), s * o.n_local.x() + c * o.n_local.y(), o.n_local.z());
    return o;
}
inline double Phi(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }
}  // namespace detail_los

// The terms of P(behind): the return's range r along the line, the exit t_e, σ and P itself (the test reads them).
struct BehindTerms { double r = 0, te = 0, sigma = 0, p = 0; };
inline BehindTerms behind_object_terms(const Eigen::Vector3d& C, const Eigen::Vector3d& p, const LosBox& box, double sigma_m)
{
    const Eigen::Vector3d dp = p - C;
    const double r = dp.norm();
    if (not (r > 0.0)) return {};
    const Eigen::Vector3d u = dp / r;
    // the SDF along the ray is convex in t: golden section for its minimum over the whole ray past the box (t ∈ [0, t_max],
    // t_max = the camera's distance to the box centre plus the box's diagonal, beyond which the SDF only grows)
    const auto f = [&](double t) { return detail_los::box_sdf(box, C + t * u).d; };
    const double diag = std::sqrt(box.w * box.w + box.h * box.h + box.H * box.H);
    constexpr double g = 0.6180339887498949;
    double a = 0.0, bnd = (C - Eigen::Vector3d(box.cx, box.cy, 0.5 * box.H)).norm() + diag;
    double t1 = bnd - g * (bnd - a), t2 = a + g * (bnd - a), f1 = f(t1), f2 = f(t2);
    for (int it = 0; it < 80; ++it)
    {
        if (f1 <= f2) { bnd = t2; t2 = t1; f2 = f1; t1 = bnd - g * (bnd - a); f1 = f(t1); }
        else          { a = t1; t1 = t2; f1 = f2; t2 = a + g * (bnd - a); f2 = f(t2); }
    }
    const double ts = 0.5 * (a + bnd), ds = f(ts);
    // where the line leaves the box: the root of the SDF beyond the minimum (bisection; the SDF is monotone there), or the
    // closest-approach point when the line misses
    double te = ts;
    if (ds < 0.0)
    {
        double lo = ts, hi = ts;
        const double step = 0.5 * diag + 1e-3;
        while (f(hi) < 0.0) hi += step;
        for (int it = 0; it < 80; ++it) { const double mid = 0.5 * (lo + hi); (f(mid) < 0.0 ? lo : hi) = mid; }
        te = 0.5 * (lo + hi);
    }
    // The variance of the face the line leaves by, BLENDED over the faces (fix round 1, I1): which face that is is itself known
    // only to σ_m, so face k gets the weight w_k = exp(−e_k²/2σ_m²) of its own plane's offset e_k = |q_k| − half_k at the exit
    // point (0 on the face the point lies on; at an edge two faces share it equally). Continuous in the exit point, hence in p;
    // a hard switch by the exit normal made σ, and P, step at every edge.
    //   x faces (±½w): Var = (cos ψ, sin ψ)ᵀΣ_c(cos ψ, sin ψ) + ¼ var_w;  y faces (±½h): the same with (−sin ψ, cos ψ), + ¼ var_h;
    //   top (z = H): var_H;  bottom (z = 0, the floor the box stands on): 0.
    const Eigen::Vector3d xe = C + te * u;
    const double c = std::cos(box.yaw), sn = std::sin(box.yaw);
    const double dx = xe.x() - box.cx, dy = xe.y() - box.cy;
    const Eigen::Vector3d ql(c * dx + sn * dy, -sn * dx + c * dy, xe.z() - 0.5 * box.H);
    const Eigen::Vector3d half(0.5 * box.w, 0.5 * box.h, 0.5 * box.H);
    const double v[3] = {c * c * box.var_cx + sn * sn * box.var_cy + 0.25 * box.var_w,
                         sn * sn * box.var_cx + c * c * box.var_cy + 0.25 * box.var_h,
                         ql.z() > 0.0 ? box.var_H : 0.0};
    double W = 0.0, V = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        const double e = std::abs(ql(k)) - half(k), w = std::exp(-0.5 * e * e / (sigma_m * sigma_m));
        W += w; V += w * v[k];
    }
    const double var_face = W > 0.0 ? V / W : 0.0;
    const double sig = std::sqrt(sigma_m * sigma_m + var_face);
    return {r, te, sig, detail_los::Phi((r - te) / sig)};
}
inline double behind_object_probability(const Eigen::Vector3d& C, const Eigen::Vector3d& p, const LosBox& box, double sigma_m)
{ return behind_object_terms(C, p, box, sigma_m).p; }

// The persistent label with the depth ordering: the camera's label q applies to the return only as far as the return is NOT
// behind the object along the camera ray, q_eff = q·(1 − P(behind)), m = ε + (1 − 2ε)·q_eff.
inline double depth_ordered_label(double q, double p_behind, double eps_px)
{ return eps_px + (1.0 - 2.0 * eps_px) * q * (1.0 - p_behind); }

}  // namespace rc
