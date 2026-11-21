/*
 * beam_ray.h  —  the smooth beam likelihood of one LiDAR ray (spec 2026-11-15 lidar-measures §7c.8 a)
 *
 * SHARED, header-only, pure (no Eigen, no model): the persistent ray likelihood of the countable furniture belief.
 * A ray either crosses the model dilated by an uncertain surface offset δ ~ N(0, σ_s²) or misses it; its return is
 * either this object's (member, probability m) or foreign. The offset is MARGINALISED, not linearised:
 *
 *     p(ρ | θ) = (1 − π)(1 − m) p_pass + π · E_{δ | δ > d}[ m St(e(δ)) + (1 − m) p_pass G(e(δ)) ]
 *     π = Φ(−d/σ_s),   e(δ) = t(δ) − ρ,   p_pass = 1/ρ_max
 *     E = −log[ p(ρ | θ) / ((1 − m) p_pass) ]                    (0 for a ray far from the model)
 *
 * St is the Student-t whose −log is ½p₀c²·log(1 + e²/c²) (normalised: ν = p₀c² − 1, scale² = c²/ν), G its CDF. The
 * conditional mean over δ is a quadrature at conditional quantiles of the upper tail (quantile_nodes), so neither tail
 * cancels. This file holds the 1-D part (Task 16): densities, Φ/Φ⁻¹, Gauss–Legendre, the per-ray likelihood with its
 * derivatives in the node crossings t_k and in π, and the node placement. The ray geometry (closest approach d, crossings
 * t(δ) on the model's convex parts and their θ-derivatives) is built on top of it (Task 17, appended below; that part
 * uses Eigen for the parts' Jacobians and the 6×6 curvature, the 1-D part above does not).
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace rc::ai::beam
{

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
//  1-D MATH: densities, normal quantiles, quadrature, the per-ray likelihood
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════

namespace detail
{
    // Continued fraction of the regularised incomplete beta (modified Lentz). Converges fast for x < (a+1)/(a+b+2).
    inline double betacf(double a, double b, double x)
    {
        constexpr double tiny = 1e-300;
        constexpr double eps  = 1e-16;
        const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (std::abs(d) < tiny) d = tiny;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 500; ++m)
        {
            const double m2 = 2.0 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
            d = 1.0 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            const double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < eps) break;
        }
        return h;
    }

    // log Γ(a + b) − log Γ(a) − log Γ(b), in the association inc_beta has always used (bit-identical when cached).
    inline double lbeta_neg(double a, double b) { return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b); }
    // Regularised incomplete beta I_x(a, b), with y = 1 − x passed in separately so neither end cancels. lgab = lbeta_neg(a, b)
    // when the caller has it (the Student-t's a, b are fixed at load: its three lgamma calls were 12 % of a beam build).
    inline double inc_beta(double a, double b, double x, double y, double lgab)
    {
        if (x <= 0.0) return 0.0;
        if (y <= 0.0) return 1.0;
        const double lbt = lgab + a * std::log(x) + b * std::log(y);
        const double bt  = std::exp(lbt);
        if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
        return 1.0 - bt * betacf(b, a, y) / b;
    }
    inline double inc_beta(double a, double b, double x, double y)
    {
        if (x <= 0.0) return 0.0;
        if (y <= 0.0) return 1.0;
        const double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(y);
        const double bt  = std::exp(lbt);
        if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
        return 1.0 - bt * betacf(b, a, y) / b;
    }

    template <std::size_t N>
    inline double horner(const double (&c)[N], double r)
    {
        double v = c[N - 1];
        for (std::size_t i = N - 1; i-- > 0;) v = v * r + c[i];
        return v;
    }

    // Φ⁻¹ by AS241 (Wichura 1988, PPND16), given log p AND the complement q = 1 − p computed accurately by the caller.
    // The lower tail needs only log p, so a level far below the smallest double (π·u_k for a ray ~38σ_s from the model)
    // is handled exactly and its quantile stays finite.
    inline double phi_inv_lpq(double log_p, double q)
    {
        static constexpr double a[8] = {3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
                                        1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
                                        3.3430575583588128105e+4, 2.5090809287301226727e+3};
        static constexpr double b[8] = {1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
                                        2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
                                        5.2264952788528545610e+3};
        static constexpr double c[8] = {1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
                                        3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
                                        2.27238449892691845833e-2, 7.74545014278341407640e-4};
        static constexpr double d[8] = {1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
                                        1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
                                        1.05075007164441684324e-9};
        static constexpr double e[8] = {6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
                                        2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                        2.71155556874348757815e-5, 2.01033439929228813265e-7};
        static constexpr double f[8] = {1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
                                        7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
                                        2.04426310338993978564e-15};
        const double p  = std::exp(log_p);
        const double qq = p - 0.5;
        if (std::abs(qq) <= 0.425)
        {
            const double r = 0.180625 - qq * qq;
            return qq * horner(a, r) / horner(b, r);
        }
        const double s = (qq < 0.0) ? -log_p : -std::log(q);   // −log of the tail probability
        if (s == std::numeric_limits<double>::infinity())      // p = 0 or q = 0 exactly: the quantile is infinite
            return (qq < 0.0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        double r = std::sqrt(s);
        double v;
        if (r <= 5.0) { r -= 1.6; v = horner(c, r) / horner(d, r); }
        else          { r -= 5.0; v = horner(e, r) / horner(f, r); }
        return (qq < 0.0) ? -v : v;
    }
    inline double phi_inv_pq(double p, double q) { return phi_inv_lpq(std::log(p), q); }

    // log P(Z > x) without underflow: direct for x ≤ 5; beyond, −x²/2 − ½log 2π + log R(x) with the Mills ratio R by its
    // continued fraction R = 1/(x + 1/(x + 2/(x + 3/(x + …)))) (modified Lentz; converges fast for x > 5). The split is a
    // choice of numerical method, both branches agree to ~1e-15 there.
    inline double log_Phi_upper(double x)
    {
        if (not (x > 5.0)) return std::log(0.5 * std::erfc(x / std::numbers::sqrt2));
        if (x == std::numeric_limits<double>::infinity()) return -std::numeric_limits<double>::infinity();
        double f = x, C = x, D = 0.0;
        for (int k = 1; k <= 500; ++k)
        {
            D = 1.0 / (x + k * D);
            C = x + k / C;
            const double delta = C * D;
            f *= delta;
            if (std::abs(delta - 1.0) < 1e-16) break;
        }
        return -0.5 * x * x - 0.5 * std::log(2.0 * std::numbers::pi) - std::log(f);
    }
}  // namespace detail

// ─── Standard normal ───────────────────────────────────────────────────────────────────────────────────────────
inline double Phi(double x)       { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }   // P(Z ≤ x)
inline double Phi_upper(double x) { return 0.5 * std::erfc(x / std::numbers::sqrt2); }    // P(Z > x), no cancellation
inline double Phi_inv(double p)   { return detail::phi_inv_pq(p, 1.0 - p); }              // AS241, rel. ~1e-16
inline double log_Phi_upper(double x) { return detail::log_Phi_upper(x); }                 // log P(Z > x), any x

// ─── The range density: Student-t at the SENSOR's precision (spec "Range densities") ─────────────────────────────
// −log pdf(e) = const + ½p₀c²·log(1 + e²/c²);  ν = p₀c² − 1,  scale² = c²/ν.  cdf exact (regularised incomplete beta).
struct StudentT
{
    double p0 = 0.0, c = 0.0, nu = 0.0, scale2 = 0.0;
    double half_p0c2 = 0.0;   // ½p₀c² = (ν + 1)/2
    double log_norm  = 0.0;   // log pdf(0)
    double lgab      = 0.0;   // lbeta_neg(ν/2, ½), the cdf's constant (cached: it does not change after make)

    // nullopt when p₀c² ≤ 1: the density does not exist (ν ≤ 0). The agent refuses to start on that (load check).
    static std::optional<StudentT> make(double p0, double c)
    {
        const double p0c2 = p0 * c * c;
        if (not (p0c2 > 1.0) or not (c > 0.0)) return std::nullopt;
        StudentT s;
        s.p0 = p0; s.c = c;
        s.nu        = p0c2 - 1.0;
        s.scale2    = c * c / s.nu;
        s.half_p0c2 = 0.5 * p0c2;
        // pdf(0) = Γ((ν+1)/2) / (Γ(ν/2) √(νπ) scale), and √ν·scale = c.
        s.log_norm  = std::lgamma(0.5 * (s.nu + 1.0)) - std::lgamma(0.5 * s.nu) - 0.5 * std::log(std::numbers::pi) - std::log(c);
        s.lgab      = detail::lbeta_neg(0.5 * s.nu, 0.5);
        return s;
    }
    double logpdf(double e) const { return log_norm - half_p0c2 * std::log1p(e * e / (c * c)); }
    double pdf(double e) const { return std::exp(logpdf(e)); }
    // d/de log pdf = −p₀c²·e/(c² + e²), written as −p₀c²/(e + c²/e) for |e| > c so it is −0 (not NaN) at |e| = inf.
    double dlogpdf(double e) const
    {
        return (std::abs(e) > c) ? -2.0 * half_p0c2 / (e + c * c / e) : -2.0 * half_p0c2 * e / (c * c + e * e);
    }
    double cdf(double e) const
    {
        // P(T > |z|) = ½ I_{ν/(ν+z²)}(ν/2, ½), z = e/scale; both x and 1 − x formed without cancellation.
        const double z2 = e * e / scale2;
        const double x  = nu / (nu + z2);          // 0 at |e| = inf
        const double y  = 1.0 / (1.0 + nu / z2);   // z²/(ν + z²): 0 at e = 0, 1 at |e| = inf, never inf/inf
        const double half_tail = 0.5 * detail::inc_beta(0.5 * nu, 0.5, x, y, lgab);
        return (e < 0.0) ? half_tail : 1.0 - half_tail;
    }
};

// The Student-t smeared by a known surface's σ_r along the ray (spec 7c.8 rev 3.3.8, moment match of the scale): scale² + σ_r²
// at the SAME ν = p₀c² − 1. In (p₀, c) form: c'² = c² + ν σ_r², p₀' = (ν + 1)/c'².
inline StudentT smeared(const StudentT& st, double sigma_r)
{
    if (not (sigma_r > 0.0)) return st;
    const double c2 = st.c * st.c + st.nu * sigma_r * sigma_r;
    return *StudentT::make((st.nu + 1.0) / c2, std::sqrt(c2));
}

// ─── Gauss–Legendre on (0, 1): nodes u (ascending) and weights w (Σw = 1) ─────────────────────────────────────────
struct GaussLegendre01
{
    std::vector<double> u, w;
    static GaussLegendre01 make(int K)
    {
        assert(K >= 1);
        GaussLegendre01 g;
        g.u.assign(K, 0.0); g.w.assign(K, 0.0);
        for (int i = 0; i < (K + 1) / 2; ++i)
        {
            double x = std::cos(std::numbers::pi * (i + 0.75) / (K + 0.5));   // i-th largest root of P_K, initial guess
            double dp = 0.0;
            for (int it = 0; it < 100; ++it)
            {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 1; j <= K; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j - 1.0) * x * p2 - (j - 1.0) * p3) / j; }
                dp = K * (x * p1 - p2) / (x * x - 1.0);
                const double dx = p1 / dp;
                x -= dx;
                if (std::abs(dx) <= 1e-16 * std::abs(x) or dx == 0.0) break;
            }
            {   // final derivative at the converged root, for the weight
                double p1 = 1.0, p2 = 0.0;
                for (int j = 1; j <= K; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j - 1.0) * x * p2 - (j - 1.0) * p3) / j; }
                dp = K * (x * p1 - p2) / (x * x - 1.0);
            }
            const double wi = 1.0 / ((1.0 - x * x) * dp * dp);   // weight on (−1, 1) is 2/((1−x²)P'²); halved for (0, 1)
            g.u[K - 1 - i] = 0.5 * (1.0 + x); g.w[K - 1 - i] = wi;
            g.u[i]         = 0.5 * (1.0 - x); g.w[i]         = wi;
        }
        return g;
    }
};

// ─── The per-ray likelihood (spec "The likelihood", "Energy") ────────────────────────────────────────────────────
struct RayLik
{
    double E        = 0.0;   // −log[p(ρ|θ) / ((1 − m) p_pass)]
    double r_member = 0.0;   // posterior share of the member term in p: π·m·S̄ / p
    double o_front  = 0.0;   // (1 − m)·p_pass·Ḡ / (m·S̄ + (1 − m)·p_pass)
    double S_bar = 0.0, G_bar = 0.0;   // Σ w_k St(e_k), Σ w_k G(e_k)
    double t_bar = 0.0, var_t = 0.0;   // weighted mean / variance of the node crossings t_k
};

// t_nodes: predicted first-crossing ranges at the quadrature nodes (e_k = t_k − ρ); w_nodes: their weights, Σ = 1 (the
// conditional mean given a hit). Outputs, the partials of E for the caller's chain rule:
//   dE_dt[k] = ∂E/∂t_k (same size as t_nodes), with π and the weights held fixed;
//   *dE_dpi  = ∂E/∂π (if not null), with the nodes AND the weights held fixed (π enters p only as the hit probability;
//              its effect through the node levels and panel weights is the caller's, via dE_dt and dE_dw);
//   dE_dw[k] = ∂E/∂w_k = −π[m St(e_k) + (1 − m)p_pass G(e_k)]/p (optional: empty ⇒ not computed), weights taken as free.
// m ≤ 1 − ε_px < 1 by construction (the label floor of (a0)); m = 1 would make the reference density (1 − m)p_pass zero.
// π = 0 gives E = 0 exactly.
inline RayLik ray_likelihood(const StudentT& st, double rho, double m, double pi, double rho_max,
                             std::span<const double> t_nodes, std::span<const double> w_nodes,
                             std::span<double> dE_dt, double* dE_dpi, std::span<double> dE_dw = {})
{
    assert(t_nodes.size() == w_nodes.size() and dE_dt.size() == t_nodes.size());
    assert(dE_dw.empty() or dE_dw.size() == t_nodes.size());
    assert(m >= 0.0 and m < 1.0);
    const double p_pass = 1.0 / rho_max;
    const double a      = (1.0 - m) * p_pass;   // the miss/foreign reference density: E = 0 when p equals it
    RayLik r;
    for (std::size_t k = 0; k < t_nodes.size(); ++k)
    {
        const double e = t_nodes[k] - rho;
        r.S_bar += w_nodes[k] * st.pdf(e);
        r.G_bar += w_nodes[k] * st.cdf(e);
        r.t_bar += w_nodes[k] * t_nodes[k];
    }
    for (std::size_t k = 0; k < t_nodes.size(); ++k)
        r.var_t += w_nodes[k] * (t_nodes[k] - r.t_bar) * (t_nodes[k] - r.t_bar);

    const double H = m * r.S_bar + a * r.G_bar;          // E_δ[m St + (1 − m) p_pass G]
    const double p = (1.0 - pi) * a + pi * H;
    r.E        = 0.0 - std::log1p(pi * (H - a) / a);     // p/a = 1 + π(H − a)/a; +0 exactly at π = 0
    r.r_member = pi * m * r.S_bar / p;
    r.o_front  = a * r.G_bar / (m * r.S_bar + a);

    // ∂E/∂t_k = −π w_k [m St′(e_k) + (1 − m) p_pass St(e_k)] / p   (G′ = St);  ∂E/∂w_k = −π [m St(e_k) + a G(e_k)] / p
    for (std::size_t k = 0; k < t_nodes.size(); ++k)
    {
        const double e  = t_nodes[k] - rho;
        const double sk = st.pdf(e);
        dE_dt[k] = -pi * w_nodes[k] * (m * sk * st.dlogpdf(e) + a * sk) / p;
        if (not dE_dw.empty()) dE_dw[k] = -pi * (m * sk + a * st.cdf(e)) / p;
    }
    if (dE_dpi) *dE_dpi = -(H - a) / p;
    return r;
}

// ─── Node placement at conditional quantiles of the upper tail (spec "Quadrature") ───────────────────────────────
// Panel limits are upper-tail probabilities of δ/σ_s, b_0 = 0 < b_1 < … < b_n = π = Phi_upper(d/σ_s). Panel j has nodes
// δ_jk = −σ_s Φ⁻¹(ℓ_jk), ℓ_jk = b_{j−1} + (b_j − b_{j−1}) u_k, and weights (b_j − b_{j−1})/π · ω_k (Σ = 1).
// Everything is evaluated in LOG space from each limit's log b_j and its complement 1 − b_j:
//   log ℓ = log b_j + log(r + (1 − r) u_k), r = b_{j−1}/b_j;   weight = (b_j/π)(1 − r) ω_k;   1 − ℓ = (1 − b_j) + (b_j − b_{j−1})(1 − u_k)
// so neither tail cancels: no node is ever ±inf, however far the ray is from the model (π below the smallest double),
// and the nodes are fixed as d → −∞.
namespace detail
{
    // log_b[j], b_c[j] = log b_j and 1 − b_j for j = 0..n (log_b[0] = −inf, b_c[0] = 1).
    inline void quantile_nodes_log(double sigma_s, std::span<const double> log_b, std::span<const double> b_c,
                                   const GaussLegendre01& gl, std::vector<double>& delta, std::vector<double>& w)
    {
        const std::size_t n = log_b.size() - 1, K = gl.u.size();
        const double log_pi = log_b[n];
        delta.resize(n * K); w.resize(n * K);
        for (std::size_t j = 1; j <= n; ++j)
        {
            const double log_hi = log_b[j];
            const double r      = (log_b[j - 1] == -std::numeric_limits<double>::infinity()) ? 0.0 : std::exp(log_b[j - 1] - log_hi);
            const double width  = std::exp(log_hi) * (1.0 - r);                // b_j − b_{j−1} (only used where not tiny)
            const double scale  = std::exp(log_hi - log_pi) * (1.0 - r);       // (b_j − b_{j−1})/π
            for (std::size_t k = 0; k < K; ++k)
            {
                const double log_level = log_hi + std::log(r + (1.0 - r) * gl.u[k]);
                const double level_c   = b_c[j] + width * (1.0 - gl.u[k]);     // 1 − ℓ, without cancellation
                delta[(j - 1) * K + k] = -sigma_s * phi_inv_lpq(log_level, level_c);
                w[(j - 1) * K + k]     = scale * gl.w[k];
            }
        }
    }
}  // namespace detail

// Limits as probabilities b (b.front() = 0, b.back() = π). The last limit is taken from d (log π = log Phi_upper(d/σ_s),
// 1 − π = Φ(d/σ_s)), so it is exact even where π underflows; the inner limits come from b, with 1 − b_j formed by
// subtraction (cancels when b_j ≈ 1: prefer the z-score form below when the limits are known as levels).
inline void quantile_nodes(double d, double sigma_s, std::span<const double> b, const GaussLegendre01& gl,
                           std::vector<double>& delta, std::vector<double>& w)
{
    assert(b.size() >= 2 and b.front() == 0.0);
    const std::size_t n = b.size() - 1;
    std::vector<double> log_b(n + 1), b_c(n + 1);
    log_b[0] = -std::numeric_limits<double>::infinity(); b_c[0] = 1.0;
    for (std::size_t j = 1; j < n; ++j) { assert(b[j] > b[j - 1]); log_b[j] = std::log(b[j]); b_c[j] = 1.0 - b[j]; }
    log_b[n] = log_Phi_upper(d / sigma_s);
    b_c[n]   = Phi(d / sigma_s);
    detail::quantile_nodes_log(sigma_s, log_b, b_c, gl, delta, w);
}

// Limits as z-scores of the panel LEVELS, z_j = δ-level/σ_s with b_j = Phi_upper(z_j): z = {+inf, z_1, …, z_n = d/σ_s},
// strictly decreasing. Both log b_j and 1 − b_j = Φ(z_j) are exact, so no limit cancels at either end (Task 17's switch
// levels and part minima are levels in δ already).
inline void quantile_nodes_z(double sigma_s, std::span<const double> z, const GaussLegendre01& gl,
                             std::vector<double>& delta, std::vector<double>& w)
{
    assert(z.size() >= 2 and z.front() == std::numeric_limits<double>::infinity());
    const std::size_t n = z.size() - 1;
    std::vector<double> log_b(n + 1), b_c(n + 1);
    for (std::size_t j = 0; j <= n; ++j)
    {
        assert(j == 0 or z[j] < z[j - 1]);
        log_b[j] = log_Phi_upper(z[j]);
        b_c[j]   = Phi(z[j]);
    }
    detail::quantile_nodes_log(sigma_s, log_b, b_c, gl, delta, w);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
//  RAY GEOMETRY ON ORIENTED-BOX PARTS, THE RAY-SET ENERGY, ITS GRADIENT AND CURVATURE
//  (Task 17; spec §7c.8 a, b, c as amended by rev 3.3.6 "the likelihood is integrated in t on a composite rule")
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
//
// Every convex part of a countable model is an oriented box (plan rev 5): the main box and the bed's headboard panel.
// Nothing here calls a model SDF; every Jacobian is the part's analytic J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ.
//
// Along a ray o + t·u, in a part's frame q(t) = q0 + t·v, a_i(t) = |q_i(t)| − h_i, s(t) = ‖max(a,0)‖ + min(max_i a_i, 0):
// inside the max of six LINEAR pieces ±q_i − h_i, outside the root of a piecewise quadratic whose breakpoints are the t
// where some |q_i| = h_i. Everything below is closed form on that structure.
//
// Closest approach d_p = min_t s_p (t ∈ [0, ρ_max]); t*_p is the SMALLEST minimiser. Crossing (d ≤ 0): the vertex of
// the max of the six lines, found exactly; ∇d = (s₊J₋ + s₋J₊)/(s₋ + s₊) on the exact pair of faces meeting there.
// Miss: the minimum of the piecewise quadratic, ∇d = ∂s/∂θ at t* (envelope). One-sided at t = 0 / ρ_max.
//
// The likelihood, unconditional, in t (rev 3.3.6). δ ~ N(0, σ_s²) is pushed forward through the running minimum
// M(t) = min_p s_p(min(t, t*_p)): every δ ≥ s(0) = min over ALL parts of s_p(0) maps to t = 0 (an atom of mass F(s(0)),
// F the upper tail), and on a panel where part p is first hit (descending and equal to M) the density is
// φ_σ(s_p(t))·|∂_t s_p|. With St and G the range density and its CDF:
//     A = St(−ρ)F(s(0)) + Σ_panels ∫ St(t − ρ) φ_σ(s_p)|∂_t s_p| dt,   B = the same with G,
//     p = (1 − π)(1 − m)p_pass + m·A + (1 − m)p_pass·B,   E = −log[p/((1 − m)p_pass)].
// Panels come from a sweep in t: a part is first hit from t = 0, from a plateau start (s_p(t) = the level reached by
// the part before it, or s(0)), or from a switch root (s_p = s_q, bisected to machine precision); it stays first hit
// until its own t*_p or the next switch.
// Each panel is integrated by a composite Gauss–Legendre rule, n nodes per subinterval, between the breakpoints
// (clipped to the panel): t = ρ + G⁻¹(Φ(z_j)) and the crossings t_p(δ_j), δ_j = −σ_s z_j, z_j = −z_max … +z_max at
// spacing 0.6; the face-change points inside the part (for a face value > −8σ_s) and the region boundaries outside.
//
// The gradient is that of the DISCRETISED energy: through s and ∂_t s at every node (∂²s/∂t∂θ analytic), the atom, π,
// and every θ-dependent breakpoint, which moves its subinterval's nodes (∂_t of the integrand): level crossings
// (−∂_θs/∂_ts), t*_p (tangent: −∂_θ(∂_ts)/∂²_ts; crease: the intersection of the two face lines), plateau starts
// ((∇level − ∂_θs_p)/∂_ts_p), switch roots (−(∂_θs_p − ∂_θs_q)/(∂_ts_p − ∂_ts_q)), face changes and region boundaries.
// A zero-width subinterval keeps its motion term dL·h(b); a breakpoint at an exact tangency (∂_ts = 0) adds none.
// Breakpoints that appear, clip or reorder are the stated kink (iv).
//
// Curvature (spec (b)), PSD, × κ: r·J̄J̄ᵀ/V + (1 − o)·I_d·∇d∇dᵀ, r = m·A/p, o = (1 − m)p_pass·B/(m·A + (1 − m)p_pass·π).
// J̄, t̄ and Var t are moments on the δ-quantile nodes (K_m) at fixed δ; Var t is scaled by
// Var_exact(δ | δ > d)/Var_nodes(δ), Var_exact = σ_s²(1 + αλ − λ²), α = d/σ_s, λ the Mills ratio.
//
// Numerical floor (not a model gate): a ray with d > 8σ_s adds exactly 0; a part with d_p > 8σ_s opens no panel (its
// s_p(0) still enters s(0)).

inline constexpr int    kBeamN         = 4;     // Gauss–Legendre nodes per subinterval (the Quadrature item: n = 3 misses, 4 passes)
inline constexpr double kBeamZMax      = 3.0;   // extent of the quantile levels (set by the Quadrature item)
inline constexpr double kBeamZStep     = 0.6;   // spacing of the quantile levels
inline constexpr int    kBeamKm        = 8;     // δ-quantile nodes of the curvature's moments (set by the Reduction item)
inline constexpr double kBeamFarSigmas = 8.0;   // the 8σ_s numerical floor (ray and part)

struct BeamPart
{
    Eigen::Vector3d c    = Eigen::Vector3d::Zero();               // centre, world
    double          yaw  = 0.0;                                   // about +z; local = R(−yaw)(p − c)
    Eigen::Vector3d half = Eigen::Vector3d::Zero();               // half extents along local x, y, z
    Eigen::Matrix<double, 7, 6> J = Eigen::Matrix<double, 7, 6>::Zero();   // ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ
};

// The persistent ray-set (ranges ≥ ρ_max already clamped by staging). m_i ∈ [ε_px, 1 − ε_px].
struct BeamRays
{
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> endpoints;
    std::vector<double> m;
    double kappa = 0;
    double rho_max = 6.0;
    double sigma_s = 0.01;
    // Per-ray foreign density p_f,i (spec 7c.8 rev 3.3.7, known room surfaces; beam_room.h fills it): it replaces
    // p_pass = 1/ρ_max in every formula of the ray (p, r, o, the normaliser (1 − m)p_f, the atom, the gradient) as a constant
    // per ray. EMPTY ⇒ p_pass for every ray, bit for bit (the Task 17/18 fixtures).
    std::vector<double> p_f;
    // The room crossing of each ray (spec 7c.8 rev 3.3.8; beam_room.h stages them with p_f): t_r (+inf: none in (0, ρ_max]) and
    // σ_r = σ_surface/|n̂·u| at t_r. They enter the miss mass Q_f(t̄) of I_d. Empty with p_f (or empty alone: no room term).
    std::vector<double> t_r, sigma_r;
};

// Build with BeamConfig::make (the quantile levels of G are tabulated once, ν is fixed at load).
struct BeamConfig
{
    StudentT st;
    GaussLegendre01 gl   = GaussLegendre01::make(kBeamKm);   // the moments' δ-quantile nodes (K_m)
    GaussLegendre01 gl_t = GaussLegendre01::make(kBeamN);    // the composite rule in t (n per subinterval)
    std::vector<double> z_levels;                            // z_j = −z_max … +z_max, spacing 0.6
    std::vector<double> e_levels;                            // G⁻¹(Φ(z_j)): the member/foreign density's quantiles in e

    static BeamConfig make(const StudentT& st, int n = kBeamN, double z_max = kBeamZMax, int K_m = kBeamKm)
    {
        BeamConfig c;
        c.st = st;
        c.gl = GaussLegendre01::make(K_m);
        c.gl_t = GaussLegendre01::make(n);
        const int L = static_cast<int>(std::lround(2.0 * z_max / kBeamZStep)) + 1;
        for (int j = 0; j < L; ++j)
        {
            const double z = -z_max + j * kBeamZStep, q = Phi(z);
            double lo = -100.0, hi = 100.0;   // G⁻¹(q) by bisection to machine precision (G strictly increasing)
            for (int it = 0; it < 400; ++it)
            {
                const double mid = 0.5 * (lo + hi);
                if (not (mid > lo and mid < hi)) break;
                if (st.cdf(mid) < q) lo = mid; else hi = mid;
            }
            c.z_levels.push_back(z);
            c.e_levels.push_back(0.5 * (lo + hi));
        }
        return c;
    }
};

namespace detail
{
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
inline constexpr double kInf        = std::numeric_limits<double>::infinity();
inline constexpr double kHalfLog2Pi = 0.918938533204672741780329736406;
inline double log_phi(double z) { return -0.5 * z * z - kHalfLog2Pi; }

// One part seen along one ray: the local origin/direction, the six inside pieces α + βt, and the outside breakpoints.
struct PartRay
{
    double cy = 1.0, sy = 0.0, rho_max = 0.0;
    double q0[3]{}, v[3]{}, h[3]{};
    double alpha[6]{}, beta[6]{};   // piece m: axis m/2, sign + for even m: σq_i − h_i
    double bk[6]{};                 // sorted t ∈ (0, ρ_max) where some |q_i| = h_i
    int    bk_axis[6]{};            // its axis i and sign (bk_sign) of q_i there
    double bk_sign[6]{};
    int    nbk = 0;
    double s0  = 0.0;               // s(0)
};

inline double piece_at(const PartRay& r, int m, double t) { return r.alpha[m] + r.beta[m] * t; }
inline double gmax(const PartRay& r, double t)
{
    double g = -kInf;
    for (int m = 0; m < 6; ++m) g = std::max(g, piece_at(r, m, t));
    return g;
}
inline double sdf_at(const PartRay& r, double t)
{
    double out2 = 0.0, in = -kInf;
    for (int i = 0; i < 3; ++i)
    {
        const double a = std::abs(r.q0[i] + r.v[i] * t) - r.h[i];
        in = std::max(in, a);
        if (a > 0.0) out2 += a * a;
    }
    return std::sqrt(out2) + std::min(in, 0.0);
}

inline void make_part_ray(PartRay& r, const BeamPart& p, double cy, double sy, const Eigen::Vector3d& o,
                          const Eigen::Vector3d& u, double rho_max)
{
    r.cy = cy; r.sy = sy; r.rho_max = rho_max;
    const double dx = o.x() - p.c.x(), dy = o.y() - p.c.y();
    r.q0[0] = cy * dx + sy * dy;       r.q0[1] = -sy * dx + cy * dy;       r.q0[2] = o.z() - p.c.z();
    r.v[0]  = cy * u.x() + sy * u.y(); r.v[1]  = -sy * u.x() + cy * u.y(); r.v[2]  = u.z();
    r.nbk = 0;
    for (int i = 0; i < 3; ++i)
    {
        r.h[i] = p.half(i);
        r.alpha[2 * i]     =  r.q0[i] - r.h[i]; r.beta[2 * i]     =  r.v[i];
        r.alpha[2 * i + 1] = -r.q0[i] - r.h[i]; r.beta[2 * i + 1] = -r.v[i];
        if (r.v[i] != 0.0)
            for (const double sgn : {1.0, -1.0})
            {
                const double t = (sgn * r.h[i] - r.q0[i]) / r.v[i];
                if (t > 0.0 and t < rho_max) { r.bk[r.nbk] = t; r.bk_axis[r.nbk] = i; r.bk_sign[r.nbk] = sgn; ++r.nbk; }
            }
    }
    for (int a = 1; a < r.nbk; ++a)   // insertion sort, carrying the axis/sign
        for (int b = a; b > 0 and r.bk[b] < r.bk[b - 1]; --b)
        {
            std::swap(r.bk[b], r.bk[b - 1]); std::swap(r.bk_axis[b], r.bk_axis[b - 1]); std::swap(r.bk_sign[b], r.bk_sign[b - 1]);
        }
    r.s0 = sdf_at(r, 0.0);
}

// scale · ∂(σ q_i(t) − h_i)/∂φ added into g, φ = (cx, cy, cz, yaw, hx, hy, hz). q = R(−yaw)(p − c):
// ∂q_x/∂(cx, cy) = (−c, −s), ∂q_y/∂(cx, cy) = (s, −c), ∂q_z/∂cz = −1, ∂q_x/∂yaw = q_y, ∂q_y/∂yaw = −q_x.
inline void add_axis_grad(const PartRay& r, int i, double sigma, double t, double scale, Vec7& g)
{
    const double ss = scale * sigma;
    if (i == 0)      { g(0) -= ss * r.cy; g(1) -= ss * r.sy; g(3) += ss * (r.q0[1] + r.v[1] * t); g(4) -= scale; }
    else if (i == 1) { g(0) += ss * r.sy; g(1) -= ss * r.cy; g(3) -= ss * (r.q0[0] + r.v[0] * t); g(5) -= scale; }
    else             { g(2) -= ss; g(6) -= scale; }
}
inline void add_piece_grad(const PartRay& r, int m, double t, double scale, Vec7& g)
{
    add_axis_grad(r, m / 2, (m % 2 == 0) ? 1.0 : -1.0, t, scale, g);
}
// scale · ∂(σ v_i)/∂φ: only the yaw moves the local direction (∂v_x/∂yaw = v_y, ∂v_y/∂yaw = −v_x).
inline void add_axis_slope_grad(const PartRay& r, int i, double sigma, double scale, Vec7& g)
{
    if (i == 0) g(3) += scale * sigma * r.v[1];
    else if (i == 1) g(3) -= scale * sigma * r.v[0];
}

// s, ∂_t s, ∂²_t s at t and (if D) ∂s/∂φ, ∂(∂_t s)/∂φ. Inside: the largest piece; outside: the norm of the positive a_i.
struct SEval
{
    double s = 0.0, st = 0.0, stt = 0.0;
    Vec7 ds = Vec7::Zero(), dst = Vec7::Zero();
    int piece = -1;   // inside piece 0..5, 6 = outside
};
template <bool D>
inline SEval seval(const PartRay& r, double t)
{
    SEval e;
    double a[3], sg[3], out2 = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        const double q = r.q0[i] + r.v[i] * t;
        sg[i] = (q >= 0.0) ? 1.0 : -1.0;
        a[i]  = sg[i] * q - r.h[i];
        if (a[i] > 0.0) out2 += a[i] * a[i];
    }
    if (out2 > 0.0)
    {
        e.s = std::sqrt(out2);
        e.piece = 6;
        double vv = 0.0;
        for (int i = 0; i < 3; ++i)
            if (a[i] > 0.0) { e.st += a[i] * sg[i] * r.v[i] / e.s; vv += r.v[i] * r.v[i]; }
        e.stt = (vv - e.st * e.st) / e.s;
        if constexpr (D)
        {
            for (int i = 0; i < 3; ++i)
                if (a[i] > 0.0)
                {
                    add_axis_grad(r, i, sg[i], t, a[i] / e.s, e.ds);
                    add_axis_grad(r, i, sg[i], t, sg[i] * r.v[i] / e.s, e.dst);
                    add_axis_slope_grad(r, i, sg[i], a[i] / e.s, e.dst);
                }
            e.dst -= (e.st / e.s) * e.ds;
        }
        return e;
    }
    int k = 0;
    for (int m = 1; m < 6; ++m) if (piece_at(r, m, t) > piece_at(r, k, t) or (piece_at(r, m, t) == piece_at(r, k, t) and r.beta[m] < r.beta[k])) k = m;
    e.s = piece_at(r, k, t); e.st = r.beta[k]; e.stt = 0.0; e.piece = k;
    if constexpr (D)
    {
        add_piece_grad(r, k, t, 1.0, e.ds);
        add_axis_slope_grad(r, k / 2, (k % 2 == 0) ? 1.0 : -1.0, 1.0, e.dst);
    }
    return e;
}

struct Approach
{
    double d = kInf, t = 0.0;
    Vec7 dphi = Vec7::Zero();    // ∂d/∂φ
    Vec7 dtphi = Vec7::Zero();   // ∂t*/∂φ
    int code = -1;               // −1 miss; L·6 + R interior crease; 36 + piece one-sided at an end (a signature)
};

// The derivative of the intersection of two lines ℓ_a(t) = ℓ_b(t) (a face pair): dt = −(∂ℓ_a − ∂ℓ_b)/(β_a − β_b).
inline void add_pair_root_grad(const PartRay& r, int ma, int mb, double t, Vec7& g)
{
    const double den = r.beta[ma] - r.beta[mb];
    if (den == 0.0) return;
    add_piece_grad(r, ma, t, -1.0 / den, g);
    add_piece_grad(r, mb, t, 1.0 / den, g);
}

template <bool Grad>
inline Approach approach(const PartRay& r)
{
    Approach A;
    const double T = r.rho_max;
    double best_t = 0.0, best = gmax(r, 0.0);
    int best_rank = -1;     // the tie rank of the kept pair (below)
    int pm = -1, pn = -1;   // the pair of pieces whose intersection is the minimum (the V's two sides)
    if (const double gT = gmax(r, T); gT < best) { best = gT; best_t = T; }
    for (int m = 0; m < 6; ++m)
        for (int n = m + 1; n < 6; ++n)
        {
            const double db = r.beta[m] - r.beta[n];
            if (db == 0.0) continue;
            const double t = (r.alpha[n] - r.alpha[m]) / db;
            if (not (t > 0.0 and t < T)) continue;
            // The smallest minimiser: a flat bottom has two equal vertices, keep the earlier. At the SAME vertex (g and t tie
            // exactly; spec 7c.8 rev 3.3.8 tie rule) the pair is ranked STRUCTURALLY, never by an ulp-level difference:
            //   rank 2 = both pieces on the upper envelope AND their multipliers bracket the vertex (β_L ≤ 0 ≤ β_R: a V);
            //   rank 1 = on the envelope, same-slope (not a V: its weights would extrapolate);
            //   rank 0 = a piece below the envelope (a square part, hx == hy, puts x − hx and y − hy on the crossing of
            //            −x − hx and −y − hy; their Jacobians are not the derivative of d — Task 19 review D1).
            // "On the envelope" is equality with g up to the rounding of the two line evaluations (a numerical tolerance, not a
            // model threshold). Among equal ranks the first pair in index order is kept (deterministic). Exact ties only:
            // every other choice is unchanged.
            const double g = gmax(r, t);
            const double tol = 64.0 * std::numeric_limits<double>::epsilon() * (1.0 + std::abs(g) + std::abs(t));
            const bool on_env = g - piece_at(r, m, t) <= tol and g - piece_at(r, n, t) <= tol;
            // a V: the multipliers bracket the vertex, β_L ≤ 0 ≤ β_R (a flat side, β = 0, is a valid side: its weight takes the
            // whole derivative); same-slope pairs (both descending or both ascending) are not
            const bool vee = not ((r.beta[m] < 0.0 and r.beta[n] < 0.0) or (r.beta[m] > 0.0 and r.beta[n] > 0.0));
            const int rank = on_env ? (vee ? 2 : 1) : 0;
            // the same vertex to rounding: the line evaluations of two pairs crossing at one point differ by ulps, so "the same
            // (g, t)" is equality up to that rounding, and the structural rank decides; otherwise the smaller g, then the earlier t
            // (a flat bottom is the same g to rounding over an interval: the earliest vertex of it, whatever the ulps)
            const bool same_g = std::abs(g - best) <= tol, same_vertex = pm >= 0 and same_g and std::abs(t - best_t) <= tol;
            const bool better = same_vertex ? rank > best_rank : same_g ? t < best_t : g < best;
            if (better) { best = g; best_t = t; pm = m; pn = n; best_rank = rank; }
        }
    // the ray crosses the part: d = the deepest point, a vertex of the max of six lines. A ray through an edge or a vertex
    // (d = 0 to the rounding of the line evaluations) is a tie between this branch and the miss branch; it takes this one — a
    // V of the envelope, the crossing side's one-sided derivative (stated kink (i)) — and never the miss branch's degenerate
    // ‖max(a, 0)‖ = 0 gradient (spec 7c.8 rev 3.3.8 tie rule; the lattice fuzz in beam_ray_test).
    if (best <= 64.0 * std::numeric_limits<double>::epsilon() * (1.0 + std::abs(best_t)))
    {
        A.d = best; A.t = best_t;
        if (pm < 0)   // one-sided at an end: d = s there, ∇d its Jacobian, t* fixed
        {
            int k = 0;
            const bool right = best_t == 0.0;
            for (int m = 1; m < 6; ++m)
            {
                const double vm = piece_at(r, m, best_t), vk = piece_at(r, k, best_t);
                if (vm > vk or (vm == vk and (right ? r.beta[m] > r.beta[k] : r.beta[m] < r.beta[k]))) k = m;
            }
            A.code = 36 + k;
            if constexpr (Grad) add_piece_grad(r, k, best_t, 1.0, A.dphi);
            return A;
        }
        // the exact pair that meets at the minimum (spec rev 3.3.6 "The crease": no t* ± ε tangents)
        const int L = (r.beta[pm] < r.beta[pn]) ? pm : pn, R = (L == pm) ? pn : pm;
        A.code = L * 6 + R;
        if constexpr (Grad)
        {
            const double sm = -r.beta[L], sp = r.beta[R];
            if (sm + sp > 0.0)
            {
                add_piece_grad(r, L, best_t, sp / (sm + sp), A.dphi);
                add_piece_grad(r, R, best_t, sm / (sm + sp), A.dphi);
            }
            else add_piece_grad(r, L, best_t, 1.0, A.dphi);
            add_pair_root_grad(r, L, R, best_t, A.dtphi);
        }
        return A;
    }
    // A miss: minimise ‖max(a,0)‖² interval by interval (a quadratic on each), evaluate s exactly at the minimiser.
    double ta = 0.0, bd = kInf, bt = 0.0;
    int bt_bk = -1;   // t* sits on breakpoint bk[bt_bk] (a flat bottom starting at a region boundary)
    for (int j = 0; j <= r.nbk; ++j)
    {
        const double tb = (j < r.nbk) ? r.bk[j] : T, tm = 0.5 * (ta + tb);
        double A2 = 0.0, B = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            const double qm = r.q0[i] + r.v[i] * tm, sg = (qm >= 0.0) ? 1.0 : -1.0;
            if (sg * qm - r.h[i] > 0.0) { A2 += r.v[i] * r.v[i]; B += 2.0 * sg * r.v[i] * (sg * r.q0[i] - r.h[i]); }
        }
        const double ts = (A2 > 0.0) ? std::clamp(-B / (2.0 * A2), ta, tb) : ta;
        if (const double s = sdf_at(r, ts); s < bd)
        {
            bd = s; bt = ts;
            bt_bk = (ts == ta and j > 0) ? j - 1 : (ts == tb and j < r.nbk) ? j : -1;
        }
        ta = tb;
    }
    A.d = bd; A.t = bt; A.code = -1;
    if constexpr (Grad)
    {
        const SEval e = seval<true>(r, bt);
        A.dphi = e.ds;
        if (bt == 0.0 or bt == T) {}
        else if (bt_bk >= 0)   // a region boundary: |q_i(t)| = h_i  ⇒  dt = −∂a_i/∂φ / (σ v_i)
        {
            const int i = r.bk_axis[bt_bk];
            const double sg = r.bk_sign[bt_bk];
            add_axis_grad(r, i, sg, bt, -1.0 / (sg * r.v[i]), A.dtphi);
        }
        else if (e.stt > 0.0) A.dtphi = -e.dst / e.stt;   // tangent: ∂_t s(t*) = 0  ⇒  dt* = −∂_θ(∂_t s)/∂²_t s
    }
    return A;
}

struct Crossing
{
    double t = 0.0;
    bool clamped = true;   // t = 0: the level is at or above s(0)
};

// First crossing of the level δ (δ > d of this part) on the part's descending branch, closed form.
inline Crossing cross(const PartRay& r, double delta)
{
    Crossing c;
    if (delta <= 0.0)   // the box shrunk by |δ|: slab entry, the latest of the entering pieces
    {
        double tin = 0.0;
        int pc = -1;
        for (int i = 0; i < 3; ++i)
        {
            double te;
            int m;
            if (r.v[i] > 0.0)      { te = (-r.q0[i] - r.h[i] - delta) / r.v[i]; m = 2 * i + 1; }
            else if (r.v[i] < 0.0) { te = (r.h[i] + delta - r.q0[i]) / r.v[i];  m = 2 * i; }
            else continue;
            if (te > tin) { tin = te; pc = m; }
        }
        if (pc < 0) return c;
        c.t = tin; c.clamped = false;
        return c;
    }
    if (r.s0 <= delta) return c;
    // Walk the intervals between breakpoints from t = 0: the level is first reached in the first interval whose
    // minimum of ‖max(a,0)‖² = Σ(α_i + β_i τ)² is ≤ δ², at the smaller root.
    double ta = 0.0;
    for (int j = 0; j <= r.nbk; ++j)
    {
        const double tb = (j < r.nbk) ? r.bk[j] : r.rho_max, tm = 0.5 * (ta + tb), len = tb - ta;
        double A2 = 0.0, B = 0.0, C = -delta * delta;
        int nact = 0;
        for (int i = 0; i < 3; ++i)
        {
            const double qm = r.q0[i] + r.v[i] * tm, sg = (qm >= 0.0) ? 1.0 : -1.0;
            if (sg * qm - r.h[i] > 0.0)
            {
                const double al = sg * (r.q0[i] + r.v[i] * ta) - r.h[i], be = sg * r.v[i];
                A2 += be * be; B += 2.0 * al * be; C += al * al;
                ++nact;
            }
        }
        double tau;
        if (nact == 0) tau = 0.0;
        else
        {
            const double tmin = (A2 > 0.0) ? std::clamp(-B / (2.0 * A2), 0.0, len) : 0.0;
            if ((A2 * tmin + B) * tmin + C > 0.0 and j < r.nbk) { ta = tb; continue; }
            const double den = -B + std::sqrt(std::max(B * B - 4.0 * A2 * C, 0.0));
            tau = (den > 0.0) ? std::clamp(2.0 * C / den, 0.0, tmin) : tmin;
        }
        c.t = ta + tau; c.clamped = false;
        return c;
    }
    return c;
}

// A breakpoint of the composite rule: its t and dt/dθ (kind: 0 panel end, 1 ρ + G⁻¹(q_j), 2 crossing of δ_j,
// 3 region boundary, 4 face change; j its index, for the structure signature).
struct BeamBP
{
    double t;
    int kind, j;
    Vec6 dt;
};
struct BeamPanel
{
    int p;                 // the first-hit part
    double t0, t1;
    Vec6 dt0, dt1;
    int kind0, kind1;      // 0 t = 0, 1 plateau start, 2 switch root, 3 t*_p
};
struct BeamRayOut
{
    double E = 0.0;
    Vec6 g = Vec6::Zero();
    Mat6 W = Mat6::Zero();
};
// Per-ray diagnostics for the tests (mass of each panel by the rule and exactly, the atom, π).
struct BeamRayDiag
{
    double pi = 0.0, atom = 0.0, A = 0.0, B = 0.0;
    Vec6 dA = Vec6::Zero(), dB = Vec6::Zero();   // filled on the derivative path
    std::vector<double> panel_mass_rule, panel_mass_exact;
};
struct BeamScratch
{
    std::vector<PartRay> pr;
    std::vector<Approach> ap;
    std::vector<Vec6> dd, dts;
    std::vector<int> present;
    std::vector<std::vector<BeamBP>> geo;   // per part: region boundaries and face changes on [0, t*_p]
    std::vector<BeamPanel> panels;
    std::vector<BeamBP> bps;
    std::vector<double> samp, PA, QA, PB, QB;
    std::vector<double> mdelta, mw, mtks;   // the curvature's moment nodes (reused across rays: no allocation per ray)
};

// Geometric breakpoints of part p on its descending branch (0, t*_p): region boundaries outside, face changes inside.
template <bool D>
inline void part_geometry(const PartRay& r, const Approach& A, const BeamPart& part, double sigma, std::vector<BeamBP>& out)
{
    out.clear();
    const auto add = [&](double t, int kind, int j, const Vec7& dphi)
    {
        BeamBP b{t, kind, j, Vec6::Zero()};
        if constexpr (D) b.dt = part.J.transpose() * dphi;
        out.push_back(b);
    };
    for (int k = 0; k < r.nbk; ++k)
        if (r.bk[k] < A.t)
        {
            Vec7 g = Vec7::Zero();
            if constexpr (D)
            {
                const int i = r.bk_axis[k];
                const double sg = r.bk_sign[k];
                add_axis_grad(r, i, sg, r.bk[k], -1.0 / (sg * r.v[i]), g);
            }
            add(r.bk[k], 3, k, g);
        }
    if (not (A.d < 0.0)) return;
    // inside: walk the upper envelope of the six lines from the entry (s = 0) to t*
    double t = cross(r, 0.0).t;
    int cur = 0;
    for (int m = 1; m < 6; ++m)
        if (piece_at(r, m, t) > piece_at(r, cur, t) or (piece_at(r, m, t) == piece_at(r, cur, t) and r.beta[m] > r.beta[cur])) cur = m;
    for (int guard = 0; guard < 6; ++guard)
    {
        int nxt = -1;
        double tx_best = kInf;
        for (int m = 0; m < 6; ++m)
        {
            if (not (r.beta[m] > r.beta[cur])) continue;
            const double tx = (r.alpha[m] - r.alpha[cur]) / (r.beta[cur] - r.beta[m]);
            if (tx > t and tx < tx_best) { tx_best = tx; nxt = m; }
        }
        if (nxt < 0 or not (tx_best < A.t) or not (piece_at(r, cur, tx_best) > -kBeamFarSigmas * sigma)) break;
        Vec7 g = Vec7::Zero();
        if constexpr (D) add_pair_root_grad(r, cur, nxt, tx_best, g);
        add(tx_best, 4, cur * 6 + nxt, g);
        t = tx_best; cur = nxt;
    }
}

// BeamRays::p_f is either empty (p_pass for every ray) or one value per staged ray; anything else is a staging error and the
// ray-set is REFUSED (contributes nothing; asserts in debug). Task 19 review D2.
inline bool p_f_consistent(const BeamRays& rays)
{
    const std::size_t n = rays.endpoints.size();
    const auto one_per_ray = [n](const std::vector<double>& v) { return v.empty() or v.size() == n; };
    return one_per_ray(rays.p_f) and one_per_ray(rays.t_r) and rays.t_r.size() == rays.sigma_r.size()
           and rays.m.size() == n;   // N2: m too
}

// The term of ray i: E_i, its gradient and curvature (unscaled by κ). false ⇔ the ray contributes exactly 0 (d > 8σ_s).
// force_sweep: run the several-part panel sweep (switch search) even with one part present (the one-part check).
template <bool Deriv>
inline bool beam_ray_term(std::span<const BeamPart> parts, std::span<const double> cs, std::span<const double> sn,
                          const BeamRays& rays, std::size_t i, const BeamConfig& cfg, bool force_sweep, BeamScratch& S,
                          BeamRayOut& out, std::vector<long>* sig = nullptr, BeamRayDiag* diag = nullptr)
{
    out = BeamRayOut{};
    if (not p_f_consistent(rays) or i >= rays.endpoints.size()) return false;   // refused (D2): never read another ray's p_f
    const Eigen::Vector3d& o = rays.origin;
    const Eigen::Vector3d du = rays.endpoints[i] - o;
    const double rho = du.norm();
    if (not (rho > 0.0)) return false;
    const Eigen::Vector3d u = du / rho;
    const double sigma = rays.sigma_s, rho_max = rays.rho_max, m = rays.m[i];
    // (1 − m)·p_f: E = 0 when p equals it; p_f = p_pass (the historic division, bit-exact) unless rays.p_f is staged
    const double a_ref = rays.p_f.empty() ? (1.0 - m) / rho_max : (1.0 - m) * rays.p_f[i];
    const StudentT& st = cfg.st;
    const std::size_t P = parts.size();
    S.pr.resize(P); S.ap.resize(P); S.dd.resize(P); S.dts.resize(P); S.geo.resize(P);
    std::size_t i0 = 0, ps0 = 0;
    for (std::size_t p = 0; p < P; ++p)
    {
        make_part_ray(S.pr[p], parts[p], cs[p], sn[p], o, u, rho_max);
        S.ap[p] = approach<Deriv>(S.pr[p]);
        if (S.ap[p].d < S.ap[i0].d) i0 = p;
        if (S.pr[p].s0 < S.pr[ps0].s0) ps0 = p;
    }
    const double d = S.ap[i0].d, floor = kBeamFarSigmas * sigma, s0 = S.pr[ps0].s0;
    if (not (d <= floor)) return false;   // the 8σ_s numerical floor: exactly 0
    S.present.clear();
    for (std::size_t p = 0; p < P; ++p)
        if (S.ap[p].d <= floor)
        {
            S.present.push_back(static_cast<int>(p));
            if constexpr (Deriv)
            {
                S.dd[p]  = parts[p].J.transpose() * S.ap[p].dphi;
                S.dts[p] = parts[p].J.transpose() * S.ap[p].dtphi;
            }
            part_geometry<Deriv>(S.pr[p], S.ap[p], parts[p], sigma, S.geo[p]);
        }
    if (sig)
    {
        sig->push_back(static_cast<long>(i0) * 2 + (d > 0.0));
        for (int p : S.present) sig->push_back(S.ap[p].code);
    }
    const auto phi_s = [&](double x) { return std::exp(log_phi(x / sigma)) / sigma; };   // N(0, σ_s²) density
    const auto dt_of_level = [&](int p, double t) -> Vec6   // a level crossing moves by −∂_θs/∂_ts (0 at a tangency)
    {
        const SEval e = seval<true>(S.pr[p], t);
        if (e.st == 0.0) return Vec6::Zero();
        return parts[p].J.transpose() * (-e.ds / e.st);
    };

    // ── the atom: every δ ≥ s(0) maps to t = 0 ──
    const double zd = d / sigma, pi = Phi_upper(zd), F0 = Phi_upper(s0 / sigma);
    const double St0 = st.pdf(-rho), G0 = st.cdf(-rho);
    double A = St0 * F0, B = G0 * F0;
    Vec6 dA = Vec6::Zero(), dB = Vec6::Zero();
    if constexpr (Deriv)
    {
        const Vec6 ds0 = parts[ps0].J.transpose() * seval<true>(S.pr[ps0], 0.0).ds;
        dA -= phi_s(s0) * St0 * ds0;
        dB -= phi_s(s0) * G0 * ds0;
    }
    if (diag) { *diag = BeamRayDiag{}; diag->pi = pi; diag->atom = F0; }

    // ── the panels: a sweep in t over the running minimum ──
    S.panels.clear();
    {
        double t = 0.0, level = s0;
        Vec6 level_grad = Vec6::Zero(), dt_start = Vec6::Zero();
        if constexpr (Deriv) level_grad = parts[ps0].J.transpose() * seval<true>(S.pr[ps0], 0.0).ds;
        int active = -1, kind_start = 0;
        for (int p : S.present)   // first hit from t = 0: the present part with s_p(0) = s(0), still descending
            if (S.pr[p].s0 == s0 and S.ap[p].t > 0.0)
                if (active < 0 or seval<false>(S.pr[p], 0.0).st < seval<false>(S.pr[active], 0.0).st) active = p;
        const bool search = S.present.size() >= 2 or force_sweep;
        for (int guard = 0; guard < 8 * static_cast<int>(P) + 8; ++guard)
        {
            if (active < 0)
            {   // plateau at `level`: the next part to reach it on its descending branch
                double best = kInf;
                for (int q : S.present)
                {
                    if (not (S.ap[q].d < level) or not (S.ap[q].t > t)) continue;
                    const double tr = std::max(cross(S.pr[q], level).t, t);
                    if (tr < best) { best = tr; active = q; }
                }
                if (active < 0) break;
                t = best; kind_start = 1;
                if constexpr (Deriv)
                {
                    const SEval e = seval<true>(S.pr[active], t);
                    dt_start = (e.st == 0.0) ? Vec6::Zero() : Vec6((level_grad - parts[active].J.transpose() * e.ds) / e.st);
                }
            }
            const int a = active;
            double t_end = S.ap[a].t;
            int next = -1;
            if (search)
                for (int q : S.present)
                {
                    if (q == a or not (S.ap[q].t > t)) continue;
                    const double tm = std::min(t_end, S.ap[q].t);
                    if (not (tm > t)) continue;
                    // sample t_q − t_a's sign at both parts' breakpoints in (t, tm], then bisect the first change
                    S.samp.clear();
                    for (int pp : {a, q})
                    {
                        for (const BeamBP& b : S.geo[pp]) if (b.t > t and b.t < tm) S.samp.push_back(b.t);
                        for (double z : cfg.z_levels)
                        {
                            const double dl = -sigma * z;
                            if (dl > S.ap[pp].d and dl < S.pr[pp].s0)
                                if (const double tc = cross(S.pr[pp], dl).t; tc > t and tc < tm) S.samp.push_back(tc);
                        }
                    }
                    for (double e : cfg.e_levels) if (rho + e > t and rho + e < tm) S.samp.push_back(rho + e);
                    S.samp.push_back(tm);
                    std::ranges::sort(S.samp);
                    double prev = t;
                    for (double ts : S.samp)
                    {
                        if (not (ts > prev)) continue;
                        if (sdf_at(S.pr[q], ts) - sdf_at(S.pr[a], ts) < 0.0)
                        {
                            double lo = prev, hi = ts;
                            for (int it = 0; it < 200; ++it)
                            {
                                const double mid = 0.5 * (lo + hi);
                                if (not (mid > lo and mid < hi)) break;
                                if (sdf_at(S.pr[q], mid) - sdf_at(S.pr[a], mid) < 0.0) hi = mid; else lo = mid;
                            }
                            if (hi < t_end) { t_end = hi; next = q; }
                            break;
                        }
                        prev = ts;
                    }
                }
            BeamPanel pan{a, t, t_end, dt_start, Vec6::Zero(), kind_start, next >= 0 ? 2 : 3};
            if constexpr (Deriv)
            {
                if (next >= 0)
                {   // switch root: dt = −(∂_θs_q − ∂_θs_a)/(∂_ts_q − ∂_ts_a)
                    const SEval ea = seval<true>(S.pr[a], t_end), eq = seval<true>(S.pr[next], t_end);
                    const double den = eq.st - ea.st;
                    if (den != 0.0) pan.dt1 = -(parts[next].J.transpose() * eq.ds - parts[a].J.transpose() * ea.ds) / den;
                }
                else pan.dt1 = S.dts[a];
            }
            if (t_end > t) S.panels.push_back(pan);
            if (next >= 0) { active = next; t = t_end; dt_start = pan.dt1; kind_start = 2; }
            else
            {
                level = S.ap[a].d; t = S.ap[a].t; active = -1;
                if constexpr (Deriv) level_grad = S.dd[a];
            }
        }
    }
    if (sig) sig->push_back(1000 + static_cast<long>(S.panels.size()));

    // ── each panel on the composite Gauss–Legendre rule in t ──
    const std::size_t n = cfg.gl_t.u.size();
    for (const BeamPanel& pan : S.panels)
    {
        const int a = pan.p;
        const PartRay& r = S.pr[a];
        const double s_hi = sdf_at(r, pan.t0), s_lo = sdf_at(r, pan.t1);
        S.bps.clear();
        S.bps.push_back({pan.t0, 0, pan.kind0, pan.dt0});
        S.bps.push_back({pan.t1, 0, 10 + pan.kind1, pan.dt1});
        for (std::size_t j = 0; j < cfg.e_levels.size(); ++j)
            if (const double tb = rho + cfg.e_levels[j]; tb > pan.t0 and tb < pan.t1) S.bps.push_back({tb, 1, static_cast<int>(j), Vec6::Zero()});
        for (std::size_t j = 0; j < cfg.z_levels.size(); ++j)
        {
            const double dl = -sigma * cfg.z_levels[j];
            if (not (dl < s_hi and dl > s_lo)) continue;
            const double tc = cross(r, dl).t;
            if (not (tc > pan.t0 and tc < pan.t1)) continue;
            BeamBP b{tc, 2, static_cast<int>(j), Vec6::Zero()};
            if constexpr (Deriv) b.dt = dt_of_level(a, tc);
            S.bps.push_back(b);
        }
        for (const BeamBP& b : S.geo[a]) if (b.t > pan.t0 and b.t < pan.t1) S.bps.push_back(b);
        std::ranges::sort(S.bps, [](const BeamBP& x, const BeamBP& y) { return x.t < y.t; });
        if (sig) { sig->push_back(-a - 1); for (const BeamBP& b : S.bps) sig->push_back(b.kind * 100 + b.j); }
        // The rule on the subintervals between consecutive breakpoints. The motion of breakpoint b enters linearly:
        // d(A) ∋ dt_b·(P_prev + Q_next), with P_k = Σ_q w_q[h_q + L_k ∂_th_q u_q] (b is subinterval k's right end) and
        // Q_k = Σ_q w_q[−h_q + L_k ∂_th_q (1 − u_q)] (its left end). A zero-width subinterval (two breakpoints coincide)
        // adds nothing to A, B but has P = h(b), Q = −h(b): its motion term dL·h(b) is kept (skipping it gave ∂E/∂H
        // 2.9e5 for 3.6e3 on bed rays whose return sits exactly on the surface).
        double mass = 0.0;
        const std::size_t nb = S.bps.size();
        S.PA.assign(nb, 0.0); S.QA.assign(nb, 0.0); S.PB.assign(nb, 0.0); S.QB.assign(nb, 0.0);
        for (std::size_t k = 0; k + 1 < nb; ++k)
        {
            const double b0 = S.bps[k].t, L = S.bps[k + 1].t - b0;
            for (std::size_t q = 0; q < n; ++q)
            {
                const double uq = cfg.gl_t.u[q], wq = cfg.gl_t.w[q], t = b0 + L * uq;
                const SEval e = seval<Deriv>(r, t);
                const double ph = phi_s(e.s);
                if (sig) sig->push_back(e.piece);
                if (ph == 0.0) continue;   // underflow: the term and all its derivatives are exactly 0
                const double mu = -e.st, x = t - rho, fS = st.pdf(x), fG = st.cdf(x);
                const double hM = ph * mu, hS = fS * hM, hG = fG * hM;
                A += L * wq * hS; B += L * wq * hG; mass += L * wq * hM;
                if constexpr (Deriv)
                {
                    const Vec6 dsT = parts[a].J.transpose() * e.ds, dstT = parts[a].J.transpose() * e.dst;
                    const double dph = -e.s / (sigma * sigma) * ph;            // φ_σ′(s)
                    const Vec6 dM_th = dph * mu * dsT - ph * dstT;             // ∂_θ(φ_σ(s)·|∂_ts|) at fixed t
                    const double dM_t = dph * e.st * mu - ph * e.stt;          // ∂_t(φ_σ(s)·|∂_ts|)
                    const double dhS_t = fS * st.dlogpdf(x) * hM + fS * dM_t, dhG_t = fS * hM + fG * dM_t;
                    dA += (wq * L * fS) * dM_th;
                    dB += (wq * L * fG) * dM_th;
                    S.PA[k] += wq * (hS + L * dhS_t * uq);  S.QA[k] += wq * (-hS + L * dhS_t * (1.0 - uq));
                    S.PB[k] += wq * (hG + L * dhG_t * uq);  S.QB[k] += wq * (-hG + L * dhG_t * (1.0 - uq));
                }
            }
        }
        if constexpr (Deriv)
        {   // Breakpoint motion, group by group. Coincident breakpoints (stated kink iv) give the same energy in either
            // order but not the same gradient: per component the group is ordered by its motion in that component, so
            // the value is that component's right-hand derivative. Inside a group the zero-width coefficients telescope:
            // only the first (smallest dt) and the last (largest dt) breakpoint carry the neighbours' P and Q.
            for (std::size_t g0 = 0; g0 < nb;)
            {
                std::size_t g1 = g0;
                while (g1 + 1 < nb and S.bps[g1 + 1].t == S.bps[g0].t) ++g1;
                const double PA_l = g0 > 0 ? S.PA[g0 - 1] : 0.0, PB_l = g0 > 0 ? S.PB[g0 - 1] : 0.0;
                const double QA_r = g1 + 1 < nb ? S.QA[g1] : 0.0, QB_r = g1 + 1 < nb ? S.QB[g1] : 0.0;
                if (g1 == g0) { dA += (PA_l + QA_r) * S.bps[g0].dt; dB += (PB_l + QB_r) * S.bps[g0].dt; }
                else
                {
                    const double hA = S.PA[g0], hB = S.PB[g0];   // a zero-width subinterval: P = h(b)
                    for (int c = 0; c < 6; ++c)
                    {
                        double lo = kInf, hi = -kInf;
                        for (std::size_t k = g0; k <= g1; ++k) { lo = std::min(lo, S.bps[k].dt(c)); hi = std::max(hi, S.bps[k].dt(c)); }
                        dA(c) += (PA_l - hA) * lo + (hA + QA_r) * hi;
                        dB(c) += (PB_l - hB) * lo + (hB + QB_r) * hi;
                    }
                }
                g0 = g1 + 1;
            }
        }
        if (diag)
        {
            diag->panel_mass_rule.push_back(mass);
            diag->panel_mass_exact.push_back(Phi_upper(s_lo / sigma) - Phi_upper(s_hi / sigma));
        }
    }
    if (diag) { diag->A = A; diag->B = B; diag->dA = dA; diag->dB = dB; }

    // ── the energy: p/((1 − m)p_pass) = (1 − π) + (m/((1 − m)p_pass))·A + B ──
    const double pa = Phi(zd) + (m / a_ref) * A + B;
    out.E = (pi < 0.5) ? 0.0 - std::log1p(-pi + (m / a_ref) * A + B) : -std::log(pa);
    if constexpr (not Deriv) return true;

    // ── gradient: −d(p/a)/(p/a), d(1 − π) = φ(z_d)/σ_s·∇d ──
    const Vec6 dpa = (std::exp(log_phi(zd)) / sigma) * S.dd[i0] + (m / a_ref) * dA + dB;
    out.g = -dpa / pa;

    // ── curvature: moments on the δ-quantile nodes (K_m), Var t scaled by the truncated normal's exact variance ──
    {
        const std::size_t K = cfg.gl.u.size();
        const std::array<double, 2> lb{-kInf, log_Phi_upper(zd)}, bc{1.0, Phi(zd)};
        std::vector<double>& delta = S.mdelta;
        std::vector<double>& w = S.mw;
        quantile_nodes_log(sigma, lb, bc, cfg.gl, delta, w);
        double tbar = 0.0, dbar = 0.0;
        Vec6 Jbar = Vec6::Zero();
        std::vector<double>& tks = S.mtks;
        tks.assign(K, 0.0);
        for (std::size_t k = 0; k < K; ++k)
        {
            double tb = 0.0;
            Vec6 Tk = Vec6::Zero();
            if (delta[k] < s0)
            {
                int bp = -1;
                for (int p : S.present)
                {
                    if (not (S.ap[p].d < delta[k])) continue;
                    const double tc = cross(S.pr[p], delta[k]).t;
                    if (bp < 0 or tc < tb) { tb = tc; bp = p; }
                }
                if (bp >= 0 and tb > 0.0) Tk = dt_of_level(bp, tb);
            }
            tks[k] = tb;
            tbar += w[k] * tb; dbar += w[k] * delta[k]; Jbar += w[k] * Tk;
        }
        double vt = 0.0, vd = 0.0;
        for (std::size_t k = 0; k < K; ++k) { vt += w[k] * (tks[k] - tbar) * (tks[k] - tbar); vd += w[k] * (delta[k] - dbar) * (delta[k] - dbar); }
        const double lam = std::exp(log_phi(zd) - log_Phi_upper(zd));   // Mills ratio, log space
        const double v_exact = sigma * sigma * (1.0 + zd * lam - lam * lam);
        const double var_t = (vd > 0.0) ? vt * v_exact / vd : vt;
        const double eb = tbar - rho, c2 = st.c * st.c;
        const double V = (1.0 + eb * eb / c2) / st.p0 + var_t;
        const double p = a_ref * pa;
        // the miss mass: the foreign return's probability of lying beyond t̄ (spec 7c.8 rev 3.3.8, gap S1). No room crossing:
        // 1 − t̄/ρ_max (the historic expression, bit-exact); with one at t_r: the p_f tail
        // Q_f(t̄) = 1 − p_pass·min(t̄, t_r) − (1 − p_pass·t_r)·G_r(t̄ − t_r), G_r the CDF of the smeared Student-t.
        double Q_f = 1.0 - tbar / rho_max;
        if (not rays.t_r.empty() and std::isfinite(rays.t_r[i]))
        {
            const double tr = rays.t_r[i], pp = 1.0 / rho_max;
            Q_f = 1.0 - pp * std::min(tbar, tr) - (1.0 - pp * tr) * smeared(st, rays.sigma_r[i]).cdf(tbar - tr);
        }
        const double r_member = m * A / p, o_front = a_ref * B / (m * A + a_ref * pi);
        const double lphn = log_phi(zd);
        const double Id = (m * std::exp(2.0 * lphn - log_Phi_upper(zd))
                           + (1.0 - m) * Q_f * std::exp(2.0 * lphn - log_Phi_upper(-zd))) / (sigma * sigma);
        out.W = (V > 0.0 ? r_member / V : 0.0) * (Jbar * Jbar.transpose())
                + ((1.0 - o_front) * Id) * (S.dd[i0] * S.dd[i0].transpose());
    }
    return true;
}

inline void part_trig(std::span<const BeamPart> parts, std::vector<double>& cs, std::vector<double>& sn)
{
    cs.resize(parts.size()); sn.resize(parts.size());
    for (std::size_t p = 0; p < parts.size(); ++p) { cs[p] = std::cos(parts[p].yaw); sn[p] = std::sin(parts[p].yaw); }
}
}  // namespace detail

// d_ip and ∇_θ d_ip of one part along the ray o + t·u, t ∈ [0, ρ_max] (u unit). See the section comment.
inline double closest_approach(const BeamPart& part, const Eigen::Vector3d& o, const Eigen::Vector3d& u, double rho_max,
                               Eigen::Matrix<double, 6, 1>* grad)
{
    detail::PartRay r;
    detail::make_part_ray(r, part, std::cos(part.yaw), std::sin(part.yaw), o, u, rho_max);
    if (not grad) return detail::approach<false>(r).d;
    const detail::Approach A = detail::approach<true>(r);
    *grad = part.J.transpose() * A.dphi;
    return A.d;
}

// κ·Σ_i E_i, its exact gradient and the PSD curvature, ADDED into W, g and *E (if not null).
template <int N>
inline void accumulate_beam_rays(std::span<const BeamPart> parts, const BeamRays& rays, const BeamConfig& cfg,
                                 Eigen::Matrix<double, N, N>& W, Eigen::Matrix<double, N, 1>& g, double* E)
{
    static_assert(N == 6, "BeamPart::J maps the 6 countable parameters (cx, cy, H, w, h, yaw)");
    assert(rays.m.size() == rays.endpoints.size());
    assert(detail::p_f_consistent(rays));
    assert(not cfg.e_levels.empty());   // built by BeamConfig::make
    if (rays.kappa == 0.0 or parts.empty()) return;
    std::vector<double> cs, sn;
    detail::part_trig(parts, cs, sn);   // cos/sin of each part's yaw, once per build
    detail::BeamScratch S;
    detail::BeamRayOut ro;
    double Esum = 0.0;
    detail::Vec6 gs = detail::Vec6::Zero();
    detail::Mat6 Ws = detail::Mat6::Zero();
    for (std::size_t i = 0; i < rays.endpoints.size(); ++i)
        if (detail::beam_ray_term<true>(parts, cs, sn, rays, i, cfg, false, S, ro))
        {
            Esum += ro.E; gs += ro.g; Ws += ro.W;
        }
    W += rays.kappa * Ws;
    g += rays.kappa * gs;
    if (E) *E += rays.kappa * Esum;
}

// The same κ·Σ_i E_i, no derivatives (the damped step's trial energy).
inline double beam_rays_energy(std::span<const BeamPart> parts, const BeamRays& rays, const BeamConfig& cfg)
{
    assert(not cfg.e_levels.empty());
    assert(detail::p_f_consistent(rays));
    if (rays.kappa == 0.0 or parts.empty()) return 0.0;
    std::vector<double> cs, sn;
    detail::part_trig(parts, cs, sn);
    detail::BeamScratch S;
    detail::BeamRayOut ro;
    double Esum = 0.0;
    for (std::size_t i = 0; i < rays.endpoints.size(); ++i)
        if (detail::beam_ray_term<false>(parts, cs, sn, rays, i, cfg, false, S, ro)) Esum += ro.E;
    return rays.kappa * Esum;
}

}  // namespace rc::ai::beam
