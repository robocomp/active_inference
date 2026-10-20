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

// Piecewise-linear speed profile of one axis: from x0, ramp to the command x (rate a when |speed| grows, d when it
// shrinks, through 0 on a sign change), apply the command until H, then brake to rest at d. Breakpoints (t, s).
// ★ Integrating the profile (not sampling it) is what makes the end pose exact whatever dt is.
struct Profile
{
    float t[6] = {}, s[6] = {}; int n = 0;
    void add(float tt, float ss) { t[n] = tt; s[n] = ss; ++n; }
    float end() const { return t[n - 1]; }
    // speed at time tt (linear between breakpoints)
    float speed(float tt) const
    {
        if (tt <= t[0]) return s[0];
        for (int i = 1; i < n; ++i)
            if (tt <= t[i]) { const float dt = t[i] - t[i - 1]; return dt > 0.f ? s[i - 1] + (s[i] - s[i - 1]) * (tt - t[i - 1]) / dt : s[i]; }
        return s[n - 1];
    }
    // distance covered up to tt (exact: trapezoids)
    float travelled(float tt) const
    {
        float acc = 0.f;
        for (int i = 1; i < n; ++i)
        {
            if (tt <= t[i - 1]) break;
            const float te = std::min(tt, t[i]);
            acc += 0.5f * (s[i - 1] + speed(te)) * (te - t[i - 1]);
        }
        return acc;
    }
};

Profile profile(float x0, float x, float H, float a, float d)
{
    Profile P;
    P.add(0.f, x0);
    auto ramp_time = [](float ds, float rate) { return rate > 0.f ? ds / rate : 0.f; };   // rate <= 0: instantaneous
    float tcur = 0.f, scur = x0;
    auto ramp_to = [&](float target)
    {
        if (tcur >= H) return;
        const float ds = std::abs(target - scur);
        const bool shrinking = std::abs(target) < std::abs(scur);
        const float dur = ramp_time(ds, shrinking ? d : a);
        if (tcur + dur <= H) { tcur += dur; scur = target; }
        else { scur = scur + (target - scur) * (H - tcur) / dur; tcur = H; }   // the command is not reached within H
        P.add(tcur, scur);
    };
    if (x0 * x < 0.f) ramp_to(0.f);
    ramp_to(x);
    if (tcur < H) { tcur = H; P.add(H, scur); }
    P.add(H + ramp_time(std::abs(scur), d), 0.f);
    return P;
}

Eigen::Vector2f closest_on_seg(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f ab = b - a;
    const float l2 = ab.squaredNorm();
    const float t = l2 > 0.f ? std::clamp((p - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
    return a + t * ab;
}

// Signed distance to the margin-grown polygon given the point already in the ROBOT frame; `dir`: unit direction
// from q to its nearest boundary point (the clearance direction).
float sd_local(const std::vector<Eigen::Vector2f>& poly, float margin, const Eigen::Vector2f& q, Eigen::Vector2f* dir = nullptr)
{
    float dmin = 1e30f;
    Eigen::Vector2f best = q;
    bool inside = true;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        const Eigen::Vector2f c = closest_on_seg(q, a, b);
        const float dd = (q - c).norm();
        if (dd < dmin) { dmin = dd; best = c; }
        const Eigen::Vector2f ab = b - a, aq = q - a;
        if (ab.x() * aq.y() - ab.y() * aq.x() < 0.f) inside = false;   // right of a CCW edge => outside
    }
    if (dir != nullptr) *dir = dmin > 1e-9f ? ((best - q) / dmin).eval() : Eigen::Vector2f(0.f, 0.f);
    return (inside ? -dmin : dmin) - margin;
}

// A sweep pose as a frame: q_robot = R^T (q - p) with the quarter turn of RobotFootprint::contains_yaw (body
// forward is +y, room yaw forward is (cos, sin)). ★ trig once per pose, not once per (pose, cell).
struct Frame
{
    float c, s; Eigen::Vector2f p;
    explicit Frame(const Pose2& pose) : c(std::cos(pose.yaw - kHalfPi)), s(std::sin(pose.yaw - kHalfPi)), p(pose.p) {}
    Eigen::Vector2f to_robot(const Eigen::Vector2f& q) const { const Eigen::Vector2f d = q - p; return {c * d.x() + s * d.y(), -s * d.x() + c * d.y()}; }
    Eigen::Vector2f to_room_dir(const Eigen::Vector2f& v) const { return {c * v.x() - s * v.y(), s * v.x() + c * v.y()}; }
};

// The hull's edge lines (outward unit normals + offsets, robot frame): max_i (n_i . q - o_i) is EXACT inside a convex
// polygon and a LOWER BOUND outside (the polygon lies inside every edge's half-plane), so a point whose bound is not
// below the running minimum cannot lower it and skips the full distance.
struct EdgeLines
{
    std::vector<Eigen::Vector2f> n; std::vector<float> o;
    explicit EdgeLines(const std::vector<Eigen::Vector2f>& poly)
    {
        const std::size_t m = poly.size();
        for (std::size_t i = 0; i < m; ++i)
        {
            const Eigen::Vector2f a = poly[i], b = poly[(i + 1) % m], ab = b - a;
            const float l = ab.norm();
            const Eigen::Vector2f nn = l > 0.f ? Eigen::Vector2f(ab.y() / l, -ab.x() / l) : Eigen::Vector2f(0.f, 0.f);   // outward for CCW
            n.push_back(nn); o.push_back(nn.dot(a));
        }
    }
    float lower_bound(const Eigen::Vector2f& q) const
    {
        float m = -1e30f;
        for (std::size_t i = 0; i < n.size(); ++i) m = std::max(m, n[i].dot(q) - o[i]);
        return m;
    }
};
}  // namespace

std::vector<Pose2> rollout(const Pose2& start, float v0, float w0, float v, float w, float hold_s, const BaseLimits& b, float dt)
{
    const float H = std::max(0.f, hold_s);
    const Profile pv = profile(v0, v, H, b.a_lin, b.d_lin), pw = profile(w0, w, H, b.a_rot, b.d_rot);
    const float T = std::max(pv.end(), pw.end());
    std::vector<Pose2> out{start};
    Pose2 cur = start;
    float t = 0.f, sv = 0.f, sw = 0.f;       // sv, sw: distance / angle integrated so far
    while (t < T - 1e-6f)
    {
        t = std::min(t + dt, T);
        const float nv = pv.travelled(t), nw = pw.travelled(t);
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

Pose2 advance(const Pose2& start, float v, float w, float dt)
{
    const float dist = v * dt, dth = w * dt, h = 0.5f * dth;
    const float chord = std::abs(h) < 1e-5f ? dist : dist * std::sin(h) / h;
    const float mid = start.yaw + h;
    return {start.p + chord * Eigen::Vector2f(std::cos(mid), std::sin(mid)), start.yaw + dth};
}

float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q)
{
    return sd_local(body.polygon(), body.safety_margin(), Frame(pose).to_robot(q));
}

float hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q)
{
    return sd_local(body.polygon(), 0.f, Frame(pose).to_robot(q));
}

std::vector<Return> surface_cells(std::span<const Return> r, float cell_m)
{
    struct Acc { double wx = 0, wy = 0, w = 0, wxx = 0, wxy = 0, wyy = 0, log_absent = 0; };
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
        // precision-weighted mean of the hits: n equal hits -> sigma_r / sqrt(n); one hit keeps its own sigma.
        // Second moments carry the hit's own scatter (within) plus its offset (between): merging is idempotent.
        const double w = 1.0 / std::max(double(q.sigma) * q.sigma, 1e-12);
        const double x = q.p.x(), y = q.p.y();
        Acc& a = it->second;
        a.wx += w * x; a.wy += w * y; a.w += w;
        a.wxx += w * (x * x + q.sxx); a.wxy += w * (x * y + q.sxy); a.wyy += w * (y * y + q.syy);
        a.log_absent += q.p_exist >= 1.f ? -std::numeric_limits<double>::infinity() : std::log1p(-double(q.p_exist));
    }
    std::vector<Return> out;
    out.reserve(order.size());
    for (const std::int64_t key : order)
    {
        const Acc& a = acc[key];
        const double mx = a.wx / a.w, my = a.wy / a.w;
        Return c{{float(mx), float(my)}, float(std::sqrt(1.0 / a.w)), float(-std::expm1(a.log_absent))};
        c.sxx = float(std::max(0.0, a.wxx / a.w - mx * mx));
        c.syy = float(std::max(0.0, a.wyy / a.w - my * my));
        c.sxy = float(a.wxy / a.w - mx * my);
        out.push_back(c);
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
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m, float lps_floor)
{
    return log_p_no_contact_cells(body, sweep, surface_cells(r), sigma_pose_xy, sigma_pose_yaw, body_geometry_sigma_m, lps_floor);
}

float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m, float lps_floor)
{
    if (sweep.empty()) return 0.f;
    const auto& poly = body.polygon();
    // ★ CORRELATED ERROR, INTEGRATED, OVER SURFACE CELLS (controller rulings, fix rounds 3-4). Distances are to the
    //   BARE hull (the planner's safety margin is a route preference, not part of this probability). The error
    //   splits in two:
    //     SHARED   eps ~ N(0, sigma_s^2), sigma_s^2 = sigma_xy^2 + g^2 — one offset of every clearance at once (the
    //              pose translation and the body-model residual move all wall points together; + eps = more room);
    //     PER-CELL sigma_j^2 = sigma_cell^2 + (rho_j * sigma_yaw)^2 + lambda_2 (n . a_perp)^2 (the cell's surface
    //              estimate; yaw lever arm; the scatter the support segment does not capture, projected on the
    //              clearance direction n — see contact.h).
    //   P(no contact | none now) = E_eps[ prod_j F_j(sd_min_j + eps) ] / E_eps[ prod_j F_j(sd0_j + eps) ],
    //   F_j(x) = 1 - p_exist_j * (1 - Phi(x / sigma_j)). Conditioning on the present makes a motion that brings no
    //   cell closer than now cost nothing, continuously across the surface; the shared eps stops N wall points from
    //   counting one common error N times; the cells stop N hits of one wall from counting as N obstacles.
    const double ss = std::sqrt(double(sigma_pose_xy) * sigma_pose_xy + double(body_geometry_sigma_m) * body_geometry_sigma_m);
    float path = 0.f;
    for (std::size_t i = 1; i < sweep.size(); ++i) path += (sweep[i].p - sweep[i - 1].p).norm();
    std::vector<Frame> frames; frames.reserve(sweep.size());
    for (const Pose2& q : sweep) frames.emplace_back(q);
    const EdgeLines lines(poly);
    // ★ PRUNING, two stages, both "the factor is 1 to working precision at the worst node eps = -6 sigma_s"
    //   (Phi(7) = 1 - 1.3e-12): by range before any geometry (circumscribed reach + path + the cell's extent), then
    //   by the swept minimum.
    // ★ PROVABLE EARLY STOP (review fix: the budget). Every factor rises with eps and the integral spans
    //   eps <= +6 sigma_s, so for any deepened cell P(no contact) <= F_j(sd_min_j + 6 sigma_s) whatever the other
    //   cells do (the ratio of expectations is bounded by the supremum of that cell's own factor ratio, itself
    //   <= its numerator factor). Once that bound falls below exp(lps_floor) the exact value cannot matter to the
    //   caller and the bound is returned: a candidate sweep stops as soon as it is certainly not releasable, a
    //   continuation as soon as it is certainly not eligible. lps_floor = -inf keeps the sweep exact.
    constexpr double kOne = 7.0;
    const float reach = body.circumscribed_radius() + path;
    const Frame& f0 = frames.front();
    std::vector<ContactTerm> terms;
    terms.reserve(cells.size());
    for (const Return& q : cells)
    {
        const float rho = (q.p - f0.p).norm();
        const double sj0 = std::sqrt(double(q.sigma) * q.sigma + double(rho) * rho * sigma_pose_yaw * sigma_pose_yaw);
        // the cell's support: principal axis of its scatter, half-extent of the uniform spread with that variance
        const float tr = q.sxx + q.syy, det = q.sxx * q.syy - q.sxy * q.sxy;
        const float disc = std::sqrt(std::max(0.f, 0.25f * tr * tr - det));
        const float l1 = 0.5f * tr + disc, l2 = std::max(0.f, 0.5f * tr - disc);
        const float h = std::sqrt(3.f * std::max(0.f, l1));
        Eigen::Vector2f axis(1.f, 0.f);
        if (h > 1e-4f)
        {
            axis = std::abs(q.sxy) > 1e-12f ? Eigen::Vector2f(l1 - q.syy, q.sxy) : (q.sxx >= q.syy ? Eigen::Vector2f(1.f, 0.f) : Eigen::Vector2f(0.f, 1.f));
            axis.normalize();
        }
        const float sdev = std::sqrt(std::max(0.f, l1));
        if (double(rho) - reach - h > kSpanU * ss + kOne * (sj0 + sdev)) continue;
        const int npts = h > 1e-4f ? 3 : 1;
        const Eigen::Vector2f pts[3] = {q.p, q.p + h * axis, q.p - h * axis};
        // a cell can only prove the floor once it is inside by more than 6 sigma_s (then its bound starts to fall);
        // the bound uses sj0 (sigma_j >= sj0 after the residual scatter), which only makes it looser: still valid
        const double sd_floor = std::isfinite(lps_floor) ? -kSpanU * ss : -std::numeric_limits<double>::infinity();
        // start pose: exact at every support point; then the running minimum over the sweep with the bound filter
        float sd0 = 1e30f;
        for (int k = 0; k < npts; ++k) sd0 = std::min(sd0, sd_local(poly, 0.f, f0.to_robot(pts[k])));
        float sd = sd0;
        std::size_t at = 0;
        bool proven = false;
        for (std::size_t i = 1; i < frames.size() and not proven; ++i)
            for (int k = 0; k < npts; ++k)
            {
                const Eigen::Vector2f ql = frames[i].to_robot(pts[k]);
                if (lines.lower_bound(ql) >= sd) continue;
                const float v = sd_local(poly, 0.f, ql);
                if (v < sd)
                {
                    sd = v; at = i;
                    if (sd < sd0 and sd <= sd_floor and log_free((double(sd) + kSpanU * ss) / sj0, q.p_exist) <= lps_floor) { proven = true; break; }
                }
            }
        if (proven) return float(log_free((double(sd) + kSpanU * ss) / sj0, q.p_exist));
        // residual scatter (what the segment does not capture: the minor axis) projected on the clearance direction
        // at the pose of the minimum: from the cell's mean to its nearest hull point (room frame)
        double sj = sj0;
        if (l2 > 1e-10f)
        {
            Eigen::Vector2f dir;
            sd_local(poly, 0.f, frames[at].to_robot(q.p), &dir);
            const Eigen::Vector2f n = frames[at].to_room_dir(dir);
            const float along = n.x() * axis.x() + n.y() * axis.y();
            sj = std::sqrt(sj0 * sj0 + double(l2) * std::max(0.f, 1.f - along * along));
        }
        if (double(sd) - kSpanU * ss > kOne * sj) continue;
        terms.push_back({sd0, sd, sj, q.p_exist});
    }
    const double lr = log_ratio_shared_offset(terms, ss);
    // A denominator of -inf (every eps puts the robot inside a cell) has no information: report -inf.
    if (not std::isfinite(lr)) return -std::numeric_limits<float>::infinity();
    return float(lr);
}
}  // namespace rc::policy
