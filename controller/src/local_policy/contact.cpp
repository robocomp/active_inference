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
// Signed distance from a SEGMENT [a, b] (robot frame) to the polygon grown by `margin`; `dir` (optional): unit direction
// from the segment's nearest point to the boundary. Separated: the exact distance (endpoints vs edges, vertices vs the
// segment). Overlapping (an endpoint inside, or the segment crossing an edge): the deepest of 9 points along it.
float sd_segment(const std::vector<Eigen::Vector2f>& poly, float margin, const Eigen::Vector2f& a, const Eigen::Vector2f& b,
                 Eigen::Vector2f* dir = nullptr)
{
    if ((b - a).squaredNorm() < 1e-12f) return sd_local(poly, margin, a, dir);
    Eigen::Vector2f da_dir, db_dir;
    const float da = sd_local(poly, margin, a, &da_dir), db = sd_local(poly, margin, b, &db_dir);
    bool crosses = da < 0.f or db < 0.f;
    const std::size_t n = poly.size();
    auto cross2 = [](const Eigen::Vector2f& u, const Eigen::Vector2f& v) { return u.x() * v.y() - u.y() * v.x(); };
    for (std::size_t i = 0; i < n and not crosses; ++i)
    {
        const Eigen::Vector2f p = poly[i], q = poly[(i + 1) % n], r = q - p, sseg = b - a;
        const float den = cross2(r, sseg);
        if (std::abs(den) < 1e-12f) continue;
        const float t = cross2(a - p, sseg) / den, u = cross2(a - p, r) / den;
        if (t >= 0.f and t <= 1.f and u >= 0.f and u <= 1.f) crosses = true;
    }
    if (crosses)
    {
        float best = 1e30f; Eigen::Vector2f bd = Eigen::Vector2f::Zero();
        for (int k = 0; k <= 8; ++k)
        {
            Eigen::Vector2f d; const float v = sd_local(poly, margin, a + (b - a) * (float(k) / 8.f), &d);
            if (v < best) { best = v; bd = d; }
        }
        if (dir) *dir = bd;
        return std::min(best, 0.f);
    }
    float best = da; Eigen::Vector2f bd = da_dir;
    if (db < best) { best = db; bd = db_dir; }
    for (std::size_t i = 0; i < n; ++i)
    {
        // a vertex of the (grown) polygon against the segment: distance minus the margin
        const Eigen::Vector2f c = closest_on_seg(poly[i], a, b);
        const float v = (poly[i] - c).norm() - margin;
        if (v < best) { best = v; bd = (poly[i] - c).normalized(); }
    }
    if (dir) *dir = bd;
    return best;
}
float sd_obstacle(const std::vector<Eigen::Vector2f>& poly, const Frame& f, const Return& q, Eigen::Vector2f* dir = nullptr)
{
    if (q.seg.squaredNorm() < 1e-12f) return sd_local(poly, 0.f, f.to_robot(q.p), dir);
    return sd_segment(poly, 0.f, f.to_robot(q.p - q.seg), f.to_robot(q.p + q.seg), dir);
}
}  // namespace

float obstacle_hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Return& q)
{ return sd_obstacle(body.polygon(), Frame(pose), q); }

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

std::vector<Return> denoise_returns(std::span<const Return> r, float radius_m)
{
    // contact.h: local line fits -> SURFACE SAMPLES (segments at spacing ~c) + the surfaces' ENDS + isolated hits.
    std::vector<Return> out;
    std::vector<const Return*> L;
    L.reserve(r.size());
    for (const Return& q : r)
    {
        if (q.p_exist <= 0.f) continue;
        if (not q.aggregate) { out.push_back(q); continue; }     // another source: its own obstacle, untouched
        L.push_back(&q);
    }
    const double c = std::max(double(radius_m), 1e-4), inv = 1.0 / c;
    auto cell_of = [&](double v) { return static_cast<std::int64_t>(std::floor(v * inv)); };
    auto key = [](std::int64_t i, std::int64_t j) { return (i << 32) ^ (j & 0xffffffff); };
    std::unordered_map<std::int64_t, std::vector<int>> grid;
    grid.reserve(L.size());
    for (int i = 0; i < int(L.size()); ++i) grid[key(cell_of(L[i]->p.x()), cell_of(L[i]->p.y()))].push_back(i);
    auto neighbours = [&](int i, std::vector<int>& nb)
    {
        nb.clear();
        const auto ci = cell_of(L[i]->p.x()), cj = cell_of(L[i]->p.y());
        for (std::int64_t di = -1; di <= 1; ++di)
            for (std::int64_t dj = -1; dj <= 1; ++dj)
                if (const auto it = grid.find(key(ci + di, cj + dj)); it != grid.end())
                    for (int j : it->second)
                        if ((L[j]->p - L[i]->p).squaredNorm() <= c * c) nb.push_back(j);
    };
    struct Fit { Eigen::Vector2d m, t, n; double lt, s2, s2r; int k; };
    auto fit = [&](const std::vector<int>& nb)
    {
        Fit f; f.k = int(nb.size());
        f.m.setZero(); f.s2r = 0;
        for (int j : nb) { f.m += L[j]->p.cast<double>(); f.s2r += double(L[j]->sigma) * L[j]->sigma; }
        f.m /= f.k; f.s2r /= f.k;
        double cxx = 0, cxy = 0, cyy = 0;
        for (int j : nb) { const Eigen::Vector2d d = L[j]->p.cast<double>() - f.m; cxx += d.x() * d.x(); cxy += d.x() * d.y(); cyy += d.y() * d.y(); }
        cxx /= f.k; cxy /= f.k; cyy /= f.k;
        const double tr = cxx + cyy, disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - (cxx * cyy - cxy * cxy)));
        f.lt = 0.5 * tr + disc;
        const double ln = std::max(0.0, 0.5 * tr - disc);
        f.t = std::abs(cxy) > 1e-15 ? Eigen::Vector2d(f.lt - cyy, cxy) : (cxx >= cyy ? Eigen::Vector2d(1, 0) : Eigen::Vector2d(0, 1));
        f.t.normalize();
        f.n = Eigen::Vector2d(-f.t.y(), f.t.x());
        f.s2 = f.k > 2 ? ln * f.k / double(f.k - 2) : ln;
        return f;
    };
    // the line's prediction variance at along-coordinate a (from the centroid): regression + misfit
    auto v_line = [](const Fit& f, double a)
    { return std::max(f.s2, f.s2r) * (1.0 / f.k + a * a / std::max(f.lt * f.k, 1e-12)) + std::max(0.0, f.s2 - f.s2r); };

    std::vector<char> covered(L.size(), 0), end_done(L.size(), 0);
    std::vector<int> nb, nb2;
    for (int i = 0; i < int(L.size()); ++i)
    {
        if (covered[i]) continue;
        neighbours(i, nb);
        if (int(nb.size()) < kDenoiseMinNeighbours) { covered[i] = 1; Return o = *L[i]; o.sxx = o.sxy = o.syy = 0.f; o.seg.setZero(); out.push_back(o); continue; }
        const Fit f = fit(nb);
        const Eigen::Vector2d pi = L[i]->p.cast<double>();
        const double ai = (pi - f.m).dot(f.t);
        double alo = 1e30, ahi = -1e30; int jlo = i, jhi = i;
        for (int j : nb)
        {
            const double aj = (L[j]->p.cast<double>() - f.m).dot(f.t);
            if (aj < alo) { alo = aj; jlo = j; }
            if (aj > ahi) { ahi = aj; jhi = j; }
            if ((L[j]->p - L[i]->p).squaredNorm() <= 0.25 * c * c) covered[j] = 1;   // within c/2: this sample's span
        }
        // ★ the SAMPLE: the fitted line over [ai - c/2, ai + c/2], clipped to where the data are; sigma = the line's at
        //   the farther end of that span (never claims more than the fit supports there)
        const double lo = std::max(ai - 0.5 * c, alo), hi = std::min(ai + 0.5 * c, ahi);
        const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
        const double vs = std::max(v_line(f, lo), v_line(f, hi));
        Return smp;
        const Eigen::Vector2d pc = f.m + mid * f.t;          // ON the line (its fused position IS the line here)
        smp.p = pc.cast<float>(); smp.seg = (half * f.t).cast<float>();
        smp.sigma = float(std::sqrt(std::min(vs, f.s2r)));   // never wider than one raw hit
        double pe = 0; for (int j : nb) pe += L[j]->p_exist >= 1.f ? -1e300 : std::log1p(-double(L[j]->p_exist));
        smp.p_exist = float(-std::expm1(std::max(pe, -700.0)));
        out.push_back(smp);
        // ★ the surface's ENDS inside this span: an extreme hit with no neighbour beyond it along t is where the surface
        //   stops — kept as its own point obstacle, fused onto the line along the normal only, with its raw sigma along t
        for (const auto& [ja, aa, sgn] : {std::tuple{jlo, alo, -1.0}, std::tuple{jhi, ahi, +1.0}})
        {
            if (end_done[ja] or std::abs(aa - ai) > 0.5 * c) continue;   // beyond this span: a later sample owns it
            neighbours(ja, nb2);
            bool beyond = false;
            for (int j : nb2) if (sgn * ((L[j]->p.cast<double>() - f.m).dot(f.t) - aa) > 0.0) { beyond = true; break; }
            if (beyond) continue;
            end_done[ja] = 1;
            const Return& q = *L[ja];
            const double vr = double(q.sigma) * q.sigma, vl = v_line(f, aa);
            const double across = (q.p.cast<double>() - f.m).dot(f.n), wl = vr / (vr + vl), sp2 = vr * vl / (vr + vl);
            Return e = q;
            e.p = (q.p.cast<double>() - wl * across * f.n).cast<float>();
            e.sigma = float(std::sqrt(sp2));
            e.seg.setZero();
            const double ex = std::max(0.0, vr - sp2);
            e.sxx = float(ex * f.t.x() * f.t.x()); e.sxy = float(ex * f.t.x() * f.t.y()); e.syy = float(ex * f.t.y() * f.t.y());
            out.push_back(e);
        }
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

// log G(u) = sum_j [log F_j(sd_min_j + eps) - log F_j(sd0_j + eps)] at eps = sigma_s * u: the product of the per-cell
// ratios, each <= 1 (sd_min <= sd0); a cell never brought closer contributes exactly 0.
double log_G(std::span<const ContactTerm> t, double sigma_s, double u)
{
    double g = 0.0;
    const double e = sigma_s * u;
    for (const ContactTerm& c : t)
        if (c.sd_min < c.sd0) g += log_free((c.sd_min + e) / c.sigma, c.p_exist) - log_free((c.sd0 + e) / c.sigma, c.p_exist);
    return g;
}

// ★ THE eps INTEGRAL (controller rulings, fix round 4 + review fix round 2: accurate to 0.005 in P over
//   sigma_j/sigma_s in [0.05, 5]). "No contact now" is ONE fact about the shared offset: eps ~ N(0, sigma_s^2)
//   TRUNCATED to eps > -sd0* (sd0* = the closest cell's clearance now; per-cell noise does not enter it — a product
//   of 56 per-cell conditions over one surface counted that fact 56 times and pushed eps ~+2 cm, the optimistic
//   wedge Release of review round 1). P(no contact | none now) = E_{eps | eps > -sd0*}[ G(eps) ],
//   G = prod_j F_j(sd_min_j + eps) / F_j(sd0_j + eps) in [0, 1].
//   For p_exist = 1 every ratio RISES with eps (log Phi is concave: the inverse Mills ratio decreases), so G is
//   monotone: a soft step whose width is the smallest sigma_j/sigma_s — down to 0.05, far too sharp for a fixed
//   Gauss-Hermite rule (round 3 measured 7 points reading 0.072 for a true 0.035). G is sampled on a 0.5-wide grid
//   from u_lo = -sd0*/sigma_s to +6, taken LINEAR between nodes, and phi(u) * linear is integrated in closed form (so
//   the Gaussian itself costs nothing); an interval is bisected while (its jump in G) x (its Gaussian mass) exceeds
//   kRefineTol, down to a width of kMinWidth. Monotonicity bounds the error of a linear interpolant by jump x mass,
//   so the refined sum is accurate to ~kRefineTol per interval and to kMinWidth x phi(0) ~ 0.002 at a hard step —
//   tested against the 4001-node reference. p_exist < 1 ratios are 1 at both ends with a dip between: their dip
//   edges are seeded as nodes (below), so the refinement always sees the walls of a dip.
constexpr double kSpanU = 6.0, kCoarseU = 0.25, kRefineTol = 1e-4, kMinWidth = 1.0 / 128.0;

struct Node { double u, g; };

struct Integrator
{
    std::span<const ContactTerm> t; double sigma_s; double integral = 0;
    Node eval(double u) const { return {u, std::exp(log_G(t, sigma_s, u))}; }
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
        if (std::abs(B.g - A.g) * W > kRefineTol and B.u - A.u > kMinWidth)
        {
            const Node M = eval(0.5 * (A.u + B.u));
            add(A, M); add(M, B);
            return;
        }
        integral += piece(A.u, B.u, A.g, B.g);
    }
};
}  // namespace

double log_ratio_shared_offset(std::span<const ContactTerm> t, double sigma_s, double sd0_star, int n_ref)
{
    const double u_lo = std::max(-kSpanU, -double(sd0_star) / sigma_s);
    // ★ a truncation beyond +6 sigma_s leaves no offset that explains "none touching now": no information, -inf
    if (not (u_lo < kSpanU)) return -std::numeric_limits<double>::infinity();
    if (t.empty()) return 0.0;
    const double mass = Phi(kSpanU) - Phi(u_lo);
    Integrator I{t, sigma_s};
    if (n_ref > 1)
    {   // brute-force trapezoid reference: n_ref equally spaced nodes over [u_lo, +6]
        const double h = (kSpanU - u_lo) / double(n_ref - 1);
        for (int k = 0; k < n_ref; ++k)
        {
            const Node n = I.eval(u_lo + h * k);
            I.integral += (k == 0 or k == n_ref - 1 ? 0.5 : 1.0) * h * phi(n.u) * n.g;
        }
    }
    else
    {
        // ★ nodes: the coarse grid plus, for every p_exist < 1 term, the walls and the bottom of its dip (its ratio
        //   is 1 on both sides of [-sd0_j, -sd_min_j] and below 1 between: with equal values at two grid nodes the
        //   jump criterion would never look inside — measured: a dip enclosed in one 0.25 interval read 0.0057
        //   high). With a node at the bottom the dip is between nodes whose values differ, and refinement follows.
        std::vector<double> us;
        for (double u = u_lo; u < kSpanU; u += kCoarseU) us.push_back(u);
        us.push_back(kSpanU);
        for (const ContactTerm& c : t)
            if (c.p_exist < 1.0 and c.sd_min < c.sd0)
                for (double e : {-c.sd0, -0.5 * (c.sd0 + c.sd_min), -c.sd_min})     // both walls and the bottom
                    if (const double u = e / sigma_s; u > u_lo and u < kSpanU) us.push_back(u);
        std::sort(us.begin(), us.end());
        Node prev = I.eval(us.front());
        for (std::size_t k = 1; k < us.size(); ++k)
        {
            if (us[k] - prev.u < 1e-9) continue;
            const Node cur = I.eval(us[k]);
            I.add(prev, cur);
            prev = cur;
        }
    }
    if (not (I.integral > 0.0)) return -std::numeric_limits<double>::infinity();
    // ★ numeric clamp, not a model gate: G <= 1 term by term; rounding can push the average a hair over 1
    return std::min(0.0, std::log(I.integral) - std::log(mass));
}

float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m, float lps_floor)
{
    return log_p_no_contact_cells(body, sweep, denoise_returns(r), sigma_pose_xy, sigma_pose_yaw, body_geometry_sigma_m, lps_floor);
}

float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m, float lps_floor)
{
    if (sweep.empty()) return 0.f;
    const auto& poly = body.polygon();
    // ★ CORRELATED ERROR, INTEGRATED, OVER SURFACE CELLS (controller rulings, fix rounds 3-4, review round 2).
    //   Distances are to the BARE hull (the planner's safety margin is a route preference, not part of this
    //   probability). The error splits in two:
    //     SHARED   eps ~ N(0, sigma_s^2), sigma_s^2 = sigma_xy^2 + g^2 — one offset of every clearance at once (the
    //              pose translation and the body-model residual move all wall points together; + eps = more room),
    //              TRUNCATED to eps > -sd0* by the one fact "nothing touches now";
    //     PER-CELL sigma_j^2 = sigma_cell^2 + (rho_j * sigma_yaw)^2 + lambda_2 (n . a_perp)^2 (the cell's surface
    //              estimate; yaw lever arm; the scatter the support segment does not capture, projected on the
    //              clearance direction n — see contact.h).
    //   P(no contact | none now) = E_{eps > -sd0*}[ prod_j F_j(sd_min_j + eps) / F_j(sd0_j + eps) ],
    //   F_j(x) = 1 - p_exist_j * (1 - Phi(x / sigma_j)). A motion that brings no cell closer than now costs nothing,
    //   continuously across the surface; the shared eps stops N wall points from counting one common error N times;
    //   the cells stop N hits of one wall from counting as N obstacles.
    const double ss = std::sqrt(double(sigma_pose_xy) * sigma_pose_xy + double(body_geometry_sigma_m) * body_geometry_sigma_m);
    float path = 0.f, turn = 0.f;
    for (std::size_t i = 1; i < sweep.size(); ++i) { path += (sweep[i].p - sweep[i - 1].p).norm(); turn += std::abs(sweep[i].yaw - sweep[i - 1].yaw); }
    std::vector<Frame> frames; frames.reserve(sweep.size());
    for (const Pose2& q : sweep) frames.emplace_back(q);
    const EdgeLines lines(poly);
    constexpr double kOne = 7.0;
    const float r_circ = body.circumscribed_radius(), reach = r_circ + path;
    const Frame& f0 = frames.front();

    // ---- pass 1: every cell within range gets its support and its clearance NOW; sd0* is the closest of them.
    // ★ PRUNING by range before any geometry (circumscribed reach + path + the cell's extent): such a cell is more
    //   than 6 sigma_s + 7 sigma_j from the body at every pose, its ratio is 1 to working precision at the lowest
    //   node, and it cannot be the closest cell now either (sd0 > 6 sigma_s puts the truncation below the span).
    struct Live { const Return* q; double sj0; float sd0; };
    std::vector<Live> live;
    live.reserve(cells.size());
    float sd0_star = std::numeric_limits<float>::infinity();
    for (const Return& q : cells)
    {
        const float rho = (q.p - f0.p).norm();
        const double sj0 = std::sqrt(double(q.sigma) * q.sigma + double(rho) * rho * sigma_pose_yaw * sigma_pose_yaw);
        const double sext = std::sqrt(std::max(0.f, q.sxx + q.syy));      // the largest the anisotropic part can add
        if (double(rho) - reach - q.seg.norm() > kSpanU * ss + kOne * (sj0 + sext)) continue;
        const float sd0 = sd_obstacle(poly, f0, q);
        sd0_star = std::min(sd0_star, sd0);
        live.push_back({&q, sj0, sd0});
    }
    const double eps_lo = std::max(-double(sd0_star), -kSpanU * ss);   // the lowest offset still in range
    if (not (eps_lo < kSpanU * ss)) return -std::numeric_limits<float>::infinity();

    // ---- pass 2: the swept minimum of every live cell that the motion can bring close enough to matter.
    // ★ PROVABLE EARLY STOP (review fix: the budget). For p_exist = 1 the ratio rises with eps and the integral
    //   spans eps <= +6 sigma_s, so P(no contact) <= F_j(sd_min_j + 6 sigma_s) / F_j(sd0_j + 6 sigma_s) for any
    //   deepened cell whatever the other cells do. Once that bound falls below exp(lps_floor) the exact value cannot
    //   matter to the caller and the bound is returned: a candidate sweep stops as soon as it is certainly not
    //   releasable, a continuation as soon as it is certainly not eligible. lps_floor = -inf keeps the sweep exact.
    std::vector<ContactTerm> terms;
    terms.reserve(live.size());
    for (const Live& L : live)
    {
        const Return& q = *L.q;
        // the body's nearest point moves at most path + r_circ * turn: a cell that even then stays 7 sigma_j clear
        // at the lowest offset keeps a ratio of 1 and never enters the product
        // sj_max: the largest sigma the anisotropic covariance can give in any clearance direction — the prune and the
        // early-stop proof must hold for it (the ratio is closer to 1 for a larger sigma)
        const double sj_max = std::sqrt(L.sj0 * L.sj0 + std::max(0.f, q.sxx + q.syy));
        if (double(L.sd0) - path - r_circ * turn + eps_lo >= kOne * sj_max) continue;
        const bool can_prove = std::isfinite(lps_floor) and q.p_exist >= 1.f;
        const double top = kSpanU * ss, lf0 = can_prove ? log_free((double(L.sd0) + top) / sj_max, 1.0) : 0.0;
        float sd = L.sd0;
        std::size_t at = 0;
        bool proven = false;
        for (std::size_t i = 1; i < frames.size(); ++i)
        {
            // lower bound: the edge-line bound is 1-Lipschitz, so a segment's points are no closer than the bound at its
            // centre minus its half-length
            if (lines.lower_bound(frames[i].to_robot(q.p)) - q.seg.norm() >= sd) continue;
            const float v = sd_obstacle(poly, frames[i], q);
            if (v < sd)
            {
                sd = v; at = i;
                if (can_prove and sd < L.sd0 and double(sd) + top < 0.0 and log_free((double(sd) + top) / sj_max, 1.0) - lf0 <= lps_floor) { proven = true; break; }
            }
        }
        if (proven) return float(log_free((double(sd) + top) / sj_max, 1.0) - lf0);
        if (not (sd < L.sd0)) continue;                                      // never closer than now: ratio exactly 1
        // the obstacle's ANISOTROPIC covariance (denoise_returns: tight across the fitted surface, the raw sigma along it)
        // projected on the clearance direction at the pose of the minimum (obstacle -> nearest hull point, room frame)
        double sj = L.sj0;
        if (q.sxx + q.syy > 1e-12f)
        {
            Eigen::Vector2f dir;
            sd_obstacle(poly, frames[at], q, &dir);
            const Eigen::Vector2f n = frames[at].to_room_dir(dir);
            sj = std::sqrt(L.sj0 * L.sj0 + double(n.x() * n.x() * q.sxx + 2.f * n.x() * n.y() * q.sxy + n.y() * n.y() * q.syy));
        }
        if (double(sd) + eps_lo > kOne * sj) continue;                       // 1 to working precision at the lowest node
        terms.push_back({L.sd0, sd, sj, q.p_exist});
    }
    const double lr = log_ratio_shared_offset(terms, ss, sd0_star);
    if (not std::isfinite(lr)) return -std::numeric_limits<float>::infinity();
    return float(lr);
}
}  // namespace rc::policy
