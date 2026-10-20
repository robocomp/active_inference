#include "contact.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace rc::policy
{
namespace
{
constexpr float kHalfPi = 1.57079633f;

// Integral of a speed profile "hold x for H, then decelerate linearly to 0 at rate d" up to time t.
// ★ Integrating the profile (not sampling it) is what makes the end pose exact whatever dt is.
float travelled(float x, float H, float d, float t)
{
    if (t <= H) return x * t;
    const float tb_total = d > 0.f ? std::abs(x) / d : 0.f;       // d <= 0: instant stop, no braking distance
    const float tb = std::min(t - H, tb_total);
    return x * H + std::copysign(std::abs(x) * tb - 0.5f * d * tb * tb, x);
}

// Unsigned distance from p to segment ab, and the signed side (>0 left of a->b) for the CCW inside test.
float seg_dist(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f ab = b - a;
    const float l2 = ab.squaredNorm();
    const float t = l2 > 0.f ? std::clamp((p - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
    return (p - (a + t * ab)).norm();
}

// Signed distance to the margin-grown polygon given the point already in the ROBOT frame.
float sd_local(const std::vector<Eigen::Vector2f>& poly, float margin, const Eigen::Vector2f& q)
{
    float dmin = 1e30f;
    bool inside = true;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        dmin = std::min(dmin, seg_dist(q, a, b));
        const Eigen::Vector2f ab = b - a, aq = q - a;
        if (ab.x() * aq.y() - ab.y() * aq.x() < 0.f) inside = false;   // right of a CCW edge => outside
    }
    return (inside ? -dmin : dmin) - margin;
}

Eigen::Vector2f to_robot(const Pose2& pose, const Eigen::Vector2f& q)
{
    // ★ same quarter turn as RobotFootprint::contains_yaw: body forward is +y, room yaw forward is (cos, sin)
    const float th = pose.yaw - kHalfPi, c = std::cos(th), s = std::sin(th);
    const Eigen::Vector2f d = q - pose.p;
    return {c * d.x() + s * d.y(), -s * d.x() + c * d.y()};
}

}  // namespace

std::vector<Pose2> rollout(const Pose2& start, float v, float w, float hold_s, const BaseLimits& b, float dt)
{
    const float H = std::max(0.f, hold_s);
    const float Tv = b.d_lin > 0.f ? std::abs(v) / b.d_lin : 0.f;
    const float Tw = b.d_rot > 0.f ? std::abs(w) / b.d_rot : 0.f;
    const float T = H + std::max(Tv, Tw);
    std::vector<Pose2> out{start};
    Pose2 cur = start;
    float t = 0.f, sv = 0.f, sw = 0.f;       // sv, sw: distance / angle integrated so far
    while (t < T - 1e-6f)
    {
        t = std::min(t + dt, T);
        const float nv = travelled(v, H, b.d_lin, t), nw = travelled(w, H, b.d_rot, t);
        const float dist = nv - sv, dth = nw - sw;
        sv = nv; sw = nw;
        // exact arc: chord = dist * sinc(dth/2) along the mean heading; the straight limit falls out as dth -> 0
        const float h = 0.5f * dth;
        const float chord = std::abs(h) < 1e-5f ? dist : dist * std::sin(h) / h;
        const float mid = cur.yaw + h;
        cur.p += chord * Eigen::Vector2f(std::cos(mid), std::sin(mid));
        cur.yaw += dth;
        out.push_back(cur);
    }
    return out;
}

float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q)
{
    return sd_local(body.polygon(), body.safety_margin(), to_robot(pose, q));
}

float hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q)
{
    return sd_local(body.polygon(), 0.f, to_robot(pose, q));
}

std::vector<Return> surface_cells(std::span<const Return> r, float cell_m)
{
    struct Acc { double wx = 0, wy = 0, w = 0, log_absent = 0; };
    std::unordered_map<std::int64_t, Acc> acc;
    acc.reserve(r.size());
    std::vector<std::int64_t> order;      // first-seen order: deterministic output
    order.reserve(r.size());
    const double inv_c = 1.0 / std::max(cell_m, 1e-4f);
    for (const Return& q : r)
    {
        if (q.p_exist <= 0.f) continue;
        const auto ci = static_cast<std::int64_t>(std::floor(q.p.x() * inv_c)), cj = static_cast<std::int64_t>(std::floor(q.p.y() * inv_c));
        const std::int64_t key = (ci << 32) ^ (cj & 0xffffffff);
        auto [it, fresh] = acc.try_emplace(key);
        if (fresh) order.push_back(key);
        // precision-weighted mean of the hits: n equal hits -> sigma_r / sqrt(n); one hit keeps its own sigma
        const double w = 1.0 / std::max(double(q.sigma) * q.sigma, 1e-12);
        it->second.wx += w * q.p.x(); it->second.wy += w * q.p.y(); it->second.w += w;
        it->second.log_absent += q.p_exist >= 1.f ? -std::numeric_limits<double>::infinity() : std::log1p(-double(q.p_exist));
    }
    std::vector<Return> out;
    out.reserve(order.size());
    for (const std::int64_t key : order)
    {
        const Acc& a = acc[key];
        out.push_back({{float(a.wx / a.w), float(a.wy / a.w)}, float(std::sqrt(1.0 / a.w)), float(-std::expm1(a.log_absent))});
    }
    return out;
}

namespace
{
// log Phi(x), stable for every x: 0 above +7 (Phi(7) = 1 - 1.3e-12, below working precision of the product), erfc
// down to -30 (erfc(21) ~ 1e-193, still a normal double), the asymptotic Mills-ratio series below that.
double log_Phi(double x)
{
    if (x > 7.0) return 0.0;
    if (x > -30.0) return std::log(0.5 * std::erfc(-x / 1.4142135623730951));
    const double x2 = x * x;
    return -0.5 * x2 - std::log(-x) - 0.9189385332046727 + std::log1p(-1.0 / x2 + 3.0 / (x2 * x2));
}
// log of the per-cell "no contact" factor 1 - p_exist*(1 - Phi(x)) = (1 - p) + p*Phi(x), in log space.
double log_free(double x, double p_exist)
{
    const double lphi = log_Phi(x);
    if (p_exist >= 1.0) return lphi;
    if (p_exist <= 0.0) return 0.0;
    const double a = std::log1p(-p_exist), b = std::log(p_exist) + lphi;     // log-add-exp
    return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}
double Phi(double u) { return 0.5 * std::erfc(-u / 1.4142135623730951); }
double phi(double u) { return 0.3989422804014327 * std::exp(-0.5 * u * u); }

// log G_num(u), log G_den(u): the two products at eps = sigma_s * u.
void log_products(std::span<const ContactTerm> t, double sigma_s, double u, double& ln, double& ld)
{
    ln = ld = 0.0;
    const double e = sigma_s * u;
    for (const ContactTerm& c : t)
    {
        const double f0 = log_free((c.sd0 + e) / c.sigma, c.p_exist);
        ld += f0;
        ln += c.sd_min < c.sd0 ? log_free((c.sd_min + e) / c.sigma, c.p_exist) : f0;
    }
}

// ★ THE eps INTEGRAL (controller ruling, fix round 4: accurate to 0.005 in P over sigma_j/sigma_s in [0.05, 5]).
//   Every factor F_j rises with eps, so G(u) = prod_j F_j is MONOTONE in u = eps/sigma_s and bounded in [0, 1]: a soft
//   step whose width is the smallest sigma_j/sigma_s — down to 0.05, far too sharp for a fixed Gauss-Hermite rule
//   (round 3 measured 7 points reading 0.072 for a true 0.035). Here G is sampled on a 0.5-wide grid over +-6, taken
//   LINEAR between nodes, and phi(u) * linear is integrated in closed form (so the Gaussian itself costs nothing);
//   an interval is bisected while (its jump in G) x (its Gaussian mass) exceeds kRefineTol, down to a width of
//   kMinWidth. Monotonicity bounds the error of a linear interpolant by jump x mass, so the refined sum is accurate
//   to ~kRefineTol per interval and to kMinWidth x phi(0) ~ 0.002 at a hard step — both tested against the
//   4001-node reference. Both products share the nodes (one pass of log_free per node and cell).
constexpr double kSpanU = 6.0, kCoarseU = 0.5, kRefineTol = 1e-4, kMinWidth = 1.0 / 128.0;

struct Node { double u, gn, gd; };   // G_num, G_den relative to the maximum of G_den (at u = +6, G_den is monotone)

struct Integrator
{
    std::span<const ContactTerm> t; double sigma_s, shift; double in = 0, id = 0;
    Node eval(double u) const
    {
        double ln, ld; log_products(t, sigma_s, u, ln, ld);
        return {u, std::exp(ln - shift), std::exp(ld - shift)};
    }
    // exact integral of phi(u) * (linear through (a, Ga), (b, Gb)) over [a, b]
    static double piece(double a, double b, double Ga, double Gb)
    {
        const double W = Phi(b) - Phi(a);
        const double J = phi(a) - phi(b) - a * W;            // int_a^b (u - a) phi(u) du
        return Ga * W + (Gb - Ga) / (b - a) * J;
    }
    void add(const Node& A, const Node& B)
    {
        const double W = Phi(B.u) - Phi(A.u);
        const double jump = std::max(std::abs(B.gn - A.gn), std::abs(B.gd - A.gd));
        if (jump * W > kRefineTol and B.u - A.u > kMinWidth)
        {
            const Node M = eval(0.5 * (A.u + B.u));
            add(A, M); add(M, B);
            return;
        }
        in += piece(A.u, B.u, A.gn, B.gn);
        id += piece(A.u, B.u, A.gd, B.gd);
    }
};
}  // namespace

double log_ratio_shared_offset(std::span<const ContactTerm> t, double sigma_s, int n_ref)
{
    if (t.empty()) return 0.0;
    // the shift keeps exp() in range: G_den is monotone in u, so its value at +6 is its maximum
    double ln_top, ld_top; log_products(t, sigma_s, kSpanU, ln_top, ld_top);
    if (not std::isfinite(ld_top)) return -std::numeric_limits<double>::infinity();
    Integrator I{t, sigma_s, ld_top};
    if (n_ref > 1)
    {   // brute-force trapezoid reference: n_ref equally spaced nodes over +-6
        const double h = 2.0 * kSpanU / double(n_ref - 1);
        double sn = 0, sd = 0;
        for (int k = 0; k < n_ref; ++k)
        {
            const Node n = I.eval(-kSpanU + h * k);
            const double w = (k == 0 or k == n_ref - 1 ? 0.5 : 1.0) * h * phi(n.u);
            sn += w * n.gn; sd += w * n.gd;
        }
        I.in = sn; I.id = sd;
    }
    else
    {
        Node prev = I.eval(-kSpanU);
        for (double u = -kSpanU + kCoarseU; u <= kSpanU + 1e-9; u += kCoarseU)
        {
            const Node cur = I.eval(std::min(u, kSpanU));
            I.add(prev, cur);
            prev = cur;
        }
    }
    if (not (I.id > 0.0)) return -std::numeric_limits<double>::infinity();
    // ★ numeric clamp, not a model gate: the ratio is <= 1 term by term (sd_min <= sd0); rounding can push it a hair over
    return std::min(0.0, std::log(I.in) - std::log(I.id));
}

float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m)
{
    return log_p_no_contact_cells(body, sweep, surface_cells(r), sigma_pose_xy, sigma_pose_yaw, body_geometry_sigma_m);
}

float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m)
{
    if (sweep.empty()) return 0.f;
    const auto& poly = body.polygon();
    // ★ CORRELATED ERROR, INTEGRATED, OVER SURFACE CELLS (controller rulings, fix rounds 3-4). Distances are to the
    //   BARE hull (the planner's safety margin is a route preference, not part of this probability). The error
    //   splits in two:
    //     SHARED   eps ~ N(0, sigma_s^2), sigma_s^2 = sigma_xy^2 + g^2 — one offset of every clearance at once (the
    //              pose translation and the body-model residual move all wall points together; + eps = more room);
    //     PER-CELL sigma_j^2 = sigma_cell^2 + (rho_j * sigma_yaw)^2 (the cell's surface estimate; yaw lever arm).
    //   P(no contact | none now) = E_eps[ prod_j F_j(sd_min_j + eps) ] / E_eps[ prod_j F_j(sd0_j + eps) ],
    //   F_j(x) = 1 - p_exist_j * (1 - Phi(x / sigma_j)). Conditioning on the present makes a motion that brings no
    //   cell closer than now cost nothing, continuously across the surface; the shared eps stops N wall points from
    //   counting one common error N times; the cells stop N hits of one wall from counting as N obstacles.
    const double ss = std::sqrt(double(sigma_pose_xy) * sigma_pose_xy + double(body_geometry_sigma_m) * body_geometry_sigma_m);
    float path = 0.f;
    for (std::size_t i = 1; i < sweep.size(); ++i) path += (sweep[i].p - sweep[i - 1].p).norm();
    // ★ PRUNING, two stages, both "the factor is 1 to working precision at the worst node eps = -6 sigma_s"
    //   (Phi(7) = 1 - 1.3e-12): by range before any geometry (circumscribed reach + path), then by the swept minimum.
    constexpr double kOne = 7.0;
    const float reach = body.circumscribed_radius() + path;
    const Pose2& s0 = sweep.front();
    std::vector<ContactTerm> terms;
    terms.reserve(cells.size());
    for (const Return& q : cells)
    {
        const float rho = (q.p - s0.p).norm();
        const double sj = std::sqrt(double(q.sigma) * q.sigma + double(rho) * rho * sigma_pose_yaw * sigma_pose_yaw);
        if (double(rho) - reach > kSpanU * ss + kOne * sj) continue;
        const float sd0 = sd_local(poly, 0.f, to_robot(s0, q.p));
        float sd = sd0;
        for (std::size_t i = 1; i < sweep.size(); ++i) sd = std::min(sd, sd_local(poly, 0.f, to_robot(sweep[i], q.p)));
        if (double(sd) - kSpanU * ss > kOne * sj) continue;
        terms.push_back({sd0, sd, sj, q.p_exist});
    }
    const double lr = log_ratio_shared_offset(terms, ss);
    // A denominator of -inf (every eps puts the robot inside a cell) has no information: report -inf.
    if (not std::isfinite(lr)) return -std::numeric_limits<float>::infinity();
    return float(lr);
}
}  // namespace rc::policy
