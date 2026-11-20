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

    // Regularised incomplete beta I_x(a, b), with y = 1 − x passed in separately so neither end cancels.
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
        const double half_tail = 0.5 * detail::inc_beta(0.5 * nu, 0.5, x, y);
        return (e < 0.0) ? half_tail : 1.0 - half_tail;
    }
};

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
//  RAY GEOMETRY ON ORIENTED-BOX PARTS, THE RAY-SET ENERGY, ITS GRADIENT AND CURVATURE (Task 17; spec §7c.8 a, b, c)
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
//
// Every convex part of a countable model is an oriented box (plan rev 5, "parts are oriented boxes"): the main box and
// the bed's headboard panel. The model SDF along a ray is the minimum over parts; nothing here calls a model SDF, every
// Jacobian is the part's analytic J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ (no fd_eps, no jac_slope_clamp on this path).
//
// Along a ray o + t·u, in a part's frame q(t) = q0 + t·v, a_i(t) = |q_i(t)| − h_i, and the box SDF is
//     s(t) = ‖max(a, 0)‖ + min(max_i a_i, 0).
// Inside (s ≤ 0) it is the max of six LINEAR pieces ±q_i(t) − h_i; outside it is the norm of the positive a_i, a
// piecewise quadratic under the root whose breakpoints are the t where some |q_i| = h_i. Both are used in closed form:
//  - closest approach d = min_{t∈[0,ρ_max]} s(t). Crossing (d ≤ 0): the minimum of the max of six lines, found exactly by
//    enumerating the 15 pairwise intersections and the two ends; ∇d = (s₊J₋ + s₋J₊)/(s₋ + s₊) with J₋, J₊ the two pieces
//    whose intersection IS the minimum (the spec's two tangents, taken exactly: on a max of lines the tangents are the
//    pieces, so d has no staircase; taking them at t* ∓ ε instead picks a wrong piece when a third face lies within ε of
//    the vertex, which grazing faces make common), Jacobians at t* (the exact derivative of the vertex of the V).
//    One-sided at t = 0 / ρ_max. Miss (d > 0):
//    the minimum of the piecewise quadratic, interval by interval (the closed-form segment–box distance), and
//    ∇d = ∂s/∂θ at t* (envelope theorem; s is C¹ outside the box).
//  - first crossing of the level δ: for δ ≤ 0 the slab entry of the box shrunk by |δ| (max over the entering pieces);
//    for δ > 0 the first root of ‖max(a,0)‖² = δ² walking the breakpoints from t = 0 (one quadratic per interval).
//    ∂t/∂θ = −(∂s/∂θ)/(∂s/∂t), ∂t/∂δ = 1/(∂s/∂t). t = 0 (sensor inside the dilated part) is a constant, no special case.
//    This replaces the spec's "one monotone march per part" (c) by its closed form: no march, no bisection, so the 1e-4 m
//    bracket is not needed for crossings (BeamConfig::bracket_m is kept for the interface, unused).
//
// Panels (spec "Quadrature"): levels are d, each part's own minimum d_p > d, and every switch level where the first-hit
// part changes (t_p(δ) = t_q(δ)), found from the sign of t_p − t_q at the provisional panel ends and nodes and then
// bisected TO MACHINE PRECISION in δ (the discretised energy depends on the switch level through the node placement, so a
// root found to 1e-4 m would leave a 1e-4 m-scale noise in E that a finite-difference gradient check sees). Nodes and
// weights come from detail::quantile_nodes_log with the levels as z-scores (no limit cancels).
//
// The gradient is that of the DISCRETISED κ·Σ E_i: through π (via d), every node level (each node δ_k moves with the two
// limits of its panel: dδ_k = [(1 − u_k)φ(z_{j−1})dλ_{j−1} + u_k φ(z_j)dλ_j]/φ(z_k)), every weight
// (w_k = (b_j − b_{j−1})/π·ω_k), and every crossing (∂t/∂θ at fixed δ plus ∂t/∂δ·dδ_k). A limit's dλ/dθ is ∇d_p for a part
// minimum and, for a switch, the implicit derivative dλ = −(∂_θt_p − ∂_θt_q)/(∂_δt_p − ∂_δt_q). Ratios of φ and of the
// tail probabilities are formed in log space.
//
// Curvature (spec (b)), PSD, × κ: r·J̄J̄ᵀ/V + (1 − o)·I_d·∇d∇dᵀ with J̄ the weighted mean of ∂t/∂θ at fixed δ (the range
// score; the dependence of the nodes on d is the edge score's), V = (1 + ē²/c²)/p₀ + Var t, and I_d with the Mills ratio
// in both tails. The cross term is dropped.
//
// Numerical floor (spec (c), flagged, not a model gate): a ray with d > 8σ_s stops after the closest approach and adds
// exactly 0; a PART with d_p > 8σ_s is left out of the ray's panels for the same reason (its panel has upper-tail mass
// below Φ̄(8) = 6e-16, so it moves E by less than the rounding of the sum).

inline constexpr int    kBeamK          = 8;      // K, chosen by beam_ray_test quadrature_K (the smallest K that passes)
inline constexpr double kBeamEpsTangent = 1e-4;   // ε of the one-sided tangents at a crease (m)
inline constexpr double kBeamFarSigmas  = 8.0;    // the 8σ_s numerical floor

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
};

struct BeamConfig
{
    StudentT st;
    GaussLegendre01 gl = GaussLegendre01::make(kBeamK);
    double eps_tangent = kBeamEpsTangent, bracket_m = 1e-4;   // both unused (exact crease pair, closed-form crossings)
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
    double bk[6]{};
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
                if (t > 0.0 and t < rho_max) r.bk[r.nbk++] = t;
            }
    }
    std::sort(r.bk, r.bk + r.nbk);
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

// Outside (s > 0): returns ∂s/∂t at t and, if dphi, adds ∂s/∂φ.
inline double outside_grad(const PartRay& r, double t, Vec7* dphi)
{
    double a[3], sg[3], s2 = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        const double q = r.q0[i] + r.v[i] * t;
        sg[i] = (q >= 0.0) ? 1.0 : -1.0;
        a[i]  = sg[i] * q - r.h[i];
        if (a[i] > 0.0) s2 += a[i] * a[i];
    }
    const double s = std::sqrt(s2);
    if (not (s > 0.0))
    {   // on the surface to rounding (a level δ within a few ulp of 0): the limit δ → 0⁺ is the face crossing, i.e. the
        // entering piece, the largest at t with the most negative slope
        int k = -1;
        for (int m = 0; m < 6; ++m)
            if (k < 0 or piece_at(r, m, t) > piece_at(r, k, t) or (piece_at(r, m, t) == piece_at(r, k, t) and r.beta[m] < r.beta[k])) k = m;
        if (dphi) add_piece_grad(r, k, t, 1.0, *dphi);
        return r.beta[k];
    }
    double ds_dt = 0.0;
    for (int i = 0; i < 3; ++i)
        if (a[i] > 0.0)
        {
            ds_dt += a[i] * sg[i] * r.v[i] / s;
            if (dphi) add_axis_grad(r, i, sg[i], t, a[i] / s, *dphi);
        }
    return ds_dt;
}

struct Approach
{
    double d = kInf, t = 0.0;
    Vec7 dphi = Vec7::Zero();
    int code = -1;   // −1 miss; L·6 + R interior crease; 36 + piece one-sided at an end (a diagnostic signature)
};

template <bool Grad>
inline Approach approach(const PartRay& r, double eps)
{
    Approach A;
    const double T = r.rho_max;
    double best_t = 0.0, best = gmax(r, 0.0);
    int pm = -1, pn = -1;   // the pair of pieces whose intersection is the minimum (the V's two sides)
    if (const double gT = gmax(r, T); gT < best) { best = gT; best_t = T; }
    for (int m = 0; m < 6; ++m)
        for (int n = m + 1; n < 6; ++n)
        {
            const double db = r.beta[m] - r.beta[n];
            if (db == 0.0) continue;
            const double t = (r.alpha[n] - r.alpha[m]) / db;
            if (not (t > 0.0 and t < T)) continue;
            if (const double g = gmax(r, t); g < best) { best = g; best_t = t; pm = m; pn = n; }
        }
    if (best <= 0.0)   // the ray crosses the part: d = the deepest point, a vertex of the max of six lines
    {
        A.d = best; A.t = best_t;
        // the piece active at t, ties broken toward the side the ray continues on (larger slope going right)
        const auto arg_at = [&](double t, bool right)
        {
            int k = 0;
            double vk = piece_at(r, 0, t);
            for (int m = 1; m < 6; ++m)
            {
                const double vm = piece_at(r, m, t);
                if (vm > vk or (vm == vk and (right ? r.beta[m] > r.beta[k] : r.beta[m] < r.beta[k]))) { k = m; vk = vm; }
            }
            return k;
        };
        if (best_t == 0.0 or best_t == T)   // one-sided: d = s there, ∇d its Jacobian
        {
            const int k = arg_at(best_t, best_t == 0.0);
            A.code = 36 + k;
            if constexpr (Grad) add_piece_grad(r, k, best_t, 1.0, A.dphi);
            return A;
        }
        // The two sides of the V are the pair that produced the minimum (exact: no ε). The spec's tangents at t* ∓ ε pick
        // the same pieces unless a third face lies within ε of the vertex, which a grazing face (slope ≪ 1) makes common:
        // measured on the bed fixture (a ray at −1° over the mattress top), the ε rule gave ∂d/∂cy = 0.64 against 0.034.
        (void)eps;
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
            else add_piece_grad(r, L, best_t, 1.0, A.dphi);   // flat bottom (ray parallel to the deepest face)
        }
        return A;
    }
    // A miss: minimise ‖max(a,0)‖² interval by interval (a quadratic on each), evaluate s exactly at the minimiser.
    double ta = 0.0, bd = kInf, bt = 0.0;
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
        if (const double s = sdf_at(r, ts); s < bd) { bd = s; bt = ts; }
        ta = tb;
    }
    A.d = bd; A.t = bt; A.code = -1;
    if constexpr (Grad) outside_grad(r, bt, &A.dphi);
    return A;
}

struct Crossing
{
    double t = 0.0, ds_dt = 0.0;
    Vec7 dphi = Vec7::Zero();
    int  piece   = -1;     // entering piece (0..5), 6 = the rounded outside, −1 = clamped at t = 0
    bool clamped = true;   // t = 0: the sensor is inside the dilated part (t constant in θ and δ)
};

// First crossing of the level δ (δ > d of this part).
template <bool Grad>
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
        c.t = tin; c.clamped = false; c.piece = pc; c.ds_dt = r.beta[pc];
        if constexpr (Grad) add_piece_grad(r, pc, tin, 1.0, c.dphi);
        return c;
    }
    if (r.s0 <= delta) return c;
    // Walk the intervals between breakpoints from t = 0. On each, ‖max(a,0)‖² = Σ(α_i + β_i τ)² over the axes outside
    // their slab (τ = t − ta); the level is first reached in the first interval whose minimum is ≤ δ², at the smaller root.
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
        if (nact == 0) tau = 0.0;   // the interval lies inside the box: s ≤ 0 < δ from its start
        else
        {
            const double tmin = (A2 > 0.0) ? std::clamp(-B / (2.0 * A2), 0.0, len) : 0.0;
            if ((A2 * tmin + B) * tmin + C > 0.0 and j < r.nbk) { ta = tb; continue; }
            const double den = -B + std::sqrt(std::max(B * B - 4.0 * A2 * C, 0.0));   // the smaller root, no cancellation
            tau = (den > 0.0) ? std::clamp(2.0 * C / den, 0.0, tmin) : tmin;
        }
        c.t = ta + tau;
        c.clamped = false; c.piece = 6;
        c.ds_dt = outside_grad(r, c.t, Grad ? &c.dphi : nullptr);
        return c;
    }
    return c;
}

struct BeamRayOut
{
    double E = 0.0;
    Vec6 g = Vec6::Zero();
    Mat6 W = Mat6::Zero();
};

// kind 0: d; 1: a part's own minimum d_p; 2: switch p↔q (the first-hit part changes)
struct BeamLevel { double lam; int kind, p, q; };

struct BeamScratch
{
    std::vector<PartRay> pr;
    std::vector<Approach> ap;
    std::vector<Vec6> dd;
    std::vector<int> present, pj, owner;
    std::vector<BeamLevel> lev;
    std::vector<double> z, lphz, log_b, b_c, delta, w, t, dEdt, dEdw, tau, tp, pts, clam;
    std::vector<Vec6> T, dlam;
};

inline void levels_to_nodes(const std::vector<BeamLevel>& lev, double sigma_s, const GaussLegendre01& gl, BeamScratch& S)
{
    const std::size_t n = lev.size();
    S.z.resize(n + 1); S.log_b.resize(n + 1); S.b_c.resize(n + 1);
    S.z[0] = kInf; S.log_b[0] = -kInf; S.b_c[0] = 1.0;
    for (std::size_t j = 1; j <= n; ++j)
    {
        S.z[j]     = lev[n - j].lam / sigma_s;
        S.log_b[j] = log_Phi_upper(S.z[j]);
        S.b_c[j]   = Phi(S.z[j]);
    }
    quantile_nodes_log(sigma_s, S.log_b, S.b_c, gl, S.delta, S.w);
}

inline void sort_levels(std::vector<BeamLevel>& lev)
{
    std::ranges::sort(lev, [](const BeamLevel& a, const BeamLevel& b) { return a.lam < b.lam or (a.lam == b.lam and a.kind < b.kind); });
    auto last = std::unique(lev.begin(), lev.end(), [](const BeamLevel& a, const BeamLevel& b) { return a.lam == b.lam; });
    lev.erase(last, lev.end());
}

// The term of ray i: E_i, its gradient and curvature (unscaled by κ). false ⇔ the ray contributes exactly 0 (d > 8σ_s).
// force_panels: run the several-part (panel) code even when one part is present (the R7-I2 check).
template <bool Deriv>
inline bool beam_ray_term(std::span<const BeamPart> parts, std::span<const double> cs, std::span<const double> sn,
                          const BeamRays& rays, std::size_t i, const BeamConfig& cfg, bool force_panels, BeamScratch& S,
                          BeamRayOut& out, std::vector<long>* sig = nullptr)
{
    out = BeamRayOut{};
    const Eigen::Vector3d& o = rays.origin;
    const Eigen::Vector3d du = rays.endpoints[i] - o;
    const double rho = du.norm();
    if (not (rho > 0.0)) return false;
    const Eigen::Vector3d u = du / rho;
    const double sigma = rays.sigma_s, rho_max = rays.rho_max, m = rays.m[i];
    const std::size_t P = parts.size(), K = cfg.gl.u.size();
    S.pr.resize(P); S.ap.resize(P); S.dd.resize(P);
    std::size_t i0 = 0;
    for (std::size_t p = 0; p < P; ++p)
    {
        make_part_ray(S.pr[p], parts[p], cs[p], sn[p], o, u, rho_max);
        S.ap[p] = approach<Deriv>(S.pr[p], cfg.eps_tangent);
        if (S.ap[p].d < S.ap[i0].d) i0 = p;
    }
    const double d = S.ap[i0].d, floor = kBeamFarSigmas * sigma;
    if (not (d <= floor)) return false;   // the 8σ_s numerical floor: exactly 0
    S.present.clear();
    for (std::size_t p = 0; p < P; ++p)
        if (S.ap[p].d <= floor)
        {
            S.present.push_back(static_cast<int>(p));
            if constexpr (Deriv) S.dd[p] = parts[p].J.transpose() * S.ap[p].dphi;
        }
    if (sig)
    {
        sig->push_back(static_cast<long>(i0) * 2 + (d > 0.0));
        for (int p : S.present) sig->push_back(S.ap[p].code);
    }
    const double zn = d / sigma, pi = Phi_upper(zn), log_pi = log_Phi_upper(zn), lphn = log_phi(zn);
    const auto cross_T = [&](std::size_t p, const Crossing& c, Vec6& Tk, double& tk)
    {
        // t constant (clamped at 0), or a level crossed at an exact tangency (ds/dt = 0, which happens only at δ = d_p to
        // rounding, i.e. inside a zero-width panel whose weight is 0): no derivative term.
        if (c.clamped or c.ds_dt == 0.0) { Tk.setZero(); tk = 0.0; return; }
        tk = 1.0 / c.ds_dt;
        Tk = parts[p].J.transpose() * (-c.dphi * tk);
    };
    S.lev.clear();
    for (int p : S.present) S.lev.push_back({S.ap[p].d, p == static_cast<int>(i0) ? 0 : 1, p, -1});
    sort_levels(S.lev);
    const bool one = S.lev.size() == 1 and not force_panels;   // one part, one panel: Task 16's one-part node set
    if (one)
    {   // one part: δ_k = −σ_s Φ⁻¹(π u_k), weights ω_k (Task 16's one-part node set)
        const std::array<double, 2> lb{-kInf, log_pi}, bc{1.0, Phi(zn)};
        quantile_nodes_log(sigma, lb, bc, cfg.gl, S.delta, S.w);
    }
    else
    {
        levels_to_nodes(S.lev, sigma, cfg.gl, S);
        // switch detection: the sign of t_p − t_q at each panel's ends and nodes, then bisection to machine precision
        if (S.present.size() >= 2)
        {
            const std::size_t n = S.lev.size();
            const std::size_t nlev0 = n;
            for (std::size_t j = 1; j <= n; ++j)
            {
                const double lam_lo = S.lev[n - j].lam, lam_hi = (j == 1) ? kInf : S.lev[n - j + 1].lam;
                S.pj.clear();
                for (int p : S.present) if (S.ap[p].d <= lam_lo) S.pj.push_back(p);
                if (S.pj.size() < 2) continue;
                S.pts.clear();
                S.pts.push_back(lam_lo);
                for (std::size_t k = K; k-- > 0;) S.pts.push_back(S.delta[(j - 1) * K + k]);
                if (std::isfinite(lam_hi)) S.pts.push_back(lam_hi);
                const std::size_t np = S.pj.size(), nq = S.pts.size();
                S.tp.resize(np * nq);
                for (std::size_t a = 0; a < nq; ++a)
                    for (std::size_t b = 0; b < np; ++b)
                    {
                        const int p = S.pj[b];
                        S.tp[a * np + b] = (S.pts[a] == S.ap[p].d) ? S.ap[p].t : cross<false>(S.pr[p], S.pts[a]).t;
                    }
                for (std::size_t b1 = 0; b1 < np; ++b1)
                    for (std::size_t b2 = b1 + 1; b2 < np; ++b2)
                        for (std::size_t a = 0; a + 1 < nq; ++a)
                        {
                            const double D1 = S.tp[a * np + b1] - S.tp[a * np + b2], D2 = S.tp[(a + 1) * np + b1] - S.tp[(a + 1) * np + b2];
                            if (not ((D1 < 0.0 and D2 > 0.0) or (D1 > 0.0 and D2 < 0.0))) continue;
                            const PartRay &r1 = S.pr[S.pj[b1]], &r2 = S.pr[S.pj[b2]];
                            double lo = S.pts[a], hi = S.pts[a + 1];
                            const bool neg_lo = D1 < 0.0;
                            for (int it = 0; it < 200; ++it)
                            {
                                const double mid = 0.5 * (lo + hi);
                                if (not (mid > lo and mid < hi)) break;
                                const double Dm = cross<false>(r1, mid).t - cross<false>(r2, mid).t;
                                if (Dm == 0.0) { lo = hi = mid; break; }
                                if ((Dm < 0.0) == neg_lo) lo = mid; else hi = mid;
                            }
                            S.lev.push_back({0.5 * (lo + hi), 2, S.pj[b1], S.pj[b2]});
                        }
            }
            if (S.lev.size() != nlev0)
            {
                sort_levels(S.lev);
                levels_to_nodes(S.lev, sigma, cfg.gl, S);
            }
        }
    }

    // The node crossings: the first-hit part among those present in the node's panel.
    const std::size_t n = one ? 1 : S.lev.size(), NK = n * K;
    S.t.resize(NK); S.dEdt.resize(NK); S.dEdw.resize(NK); S.tau.resize(NK); S.T.resize(NK); S.owner.resize(NK);
    if (sig) sig->push_back(static_cast<long>(n) * 1000 + (one ? 0 : 1));
    if (sig and not one) for (const auto& L : S.lev) sig->push_back(L.kind * 100 + L.p * 10 + L.q + 1);
    for (std::size_t j = 1; j <= n; ++j)
    {
        const double lam_lo = one ? d : S.lev[n - j].lam;
        for (std::size_t k = 0; k < K; ++k)
        {
            const std::size_t idx = (j - 1) * K + k;
            Crossing best;
            int bp = -1;
            for (int p : S.present)
            {
                if (not (S.ap[p].d <= lam_lo)) continue;
                const Crossing c = cross<Deriv>(S.pr[p], S.delta[idx]);
                if (bp < 0 or c.t < best.t) { best = c; bp = p; }
            }
            S.t[idx] = best.t; S.owner[idx] = bp;
            if constexpr (Deriv) cross_T(static_cast<std::size_t>(bp), best, S.T[idx], S.tau[idx]);
            if (sig) sig->push_back(bp * 16 + best.piece + 1);
        }
    }
    double dEdpi = 0.0;
    const RayLik L = ray_likelihood(cfg.st, rho, m, pi, rho_max, std::span<const double>(S.t.data(), NK),
                                    std::span<const double>(S.w.data(), NK), std::span<double>(S.dEdt.data(), NK),
                                    Deriv ? &dEdpi : nullptr,
                                    (Deriv and not one) ? std::span<double>(S.dEdw.data(), NK) : std::span<double>());
    out.E = L.E;
    if constexpr (not Deriv) return true;

    // ── gradient of the discretised E ──
    Vec6& g = out.g;
    if (one)
    {
        double cd = -dEdpi * std::exp(lphn) / sigma;   // dπ = −φ(z_n)/σ_s · ∇d
        for (std::size_t k = 0; k < K; ++k)
        {
            g += S.dEdt[k] * S.T[k];
            cd += S.dEdt[k] * S.tau[k] * cfg.gl.u[k] * std::exp(lphn - log_phi(S.delta[k] / sigma));
        }
        g += cd * S.dd[i0];
    }
    else
    {
        S.lphz.resize(n + 1); S.clam.assign(n + 1, 0.0); S.dlam.resize(n + 1);
        S.lphz[0] = -kInf; S.dlam[0].setZero();
        for (std::size_t j = 1; j <= n; ++j)
        {
            S.lphz[j] = log_phi(S.z[j]);
            const BeamLevel& Lv = S.lev[n - j];
            if (Lv.kind < 2) S.dlam[j] = S.dd[Lv.p];
            else
            {   // switch level: t_p(λ) = t_q(λ)  ⇒  dλ = −(∂_θt_p − ∂_θt_q)/(∂_δt_p − ∂_δt_q)
                Vec6 Tp, Tq;
                double tp, tq;
                cross_T(Lv.p, cross<true>(S.pr[Lv.p], Lv.lam), Tp, tp);
                cross_T(Lv.q, cross<true>(S.pr[Lv.q], Lv.lam), Tq, tq);
                S.dlam[j] = -(Tp - Tq) / (tp - tq);
                if (not S.dlam[j].allFinite()) S.dlam[j].setZero();   // a tangent switch: its panel has zero width
            }
        }
        double sum_wdw = 0.0;
        for (std::size_t j = 1; j <= n; ++j)
            for (std::size_t k = 0; k < K; ++k)
            {
                const std::size_t idx = (j - 1) * K + k;
                const double lzk = log_phi(S.delta[idx] / sigma), uk = cfg.gl.u[k], ok = cfg.gl.w[k];
                g += S.dEdt[idx] * S.T[idx];
                const double c = S.dEdt[idx] * S.tau[idx];
                if (j > 1)
                {
                    S.clam[j - 1] += c * (1.0 - uk) * std::exp(S.lphz[j - 1] - lzk);
                    S.clam[j - 1] += S.dEdw[idx] * ok * std::exp(S.lphz[j - 1] - log_pi) / sigma;
                }
                S.clam[j] += c * uk * std::exp(S.lphz[j] - lzk);
                S.clam[j] -= S.dEdw[idx] * ok * std::exp(S.lphz[j] - log_pi) / sigma;
                sum_wdw += S.dEdw[idx] * S.w[idx];
            }
        S.clam[n] += -dEdpi * std::exp(lphn) / sigma + sum_wdw * std::exp(lphn - log_pi) / sigma;
        for (std::size_t j = 1; j <= n; ++j) g += S.clam[j] * S.dlam[j];
    }

    // ── curvature: range part r·J̄J̄ᵀ/V + edge part (1 − o)·I_d·∇d∇dᵀ ──
    Vec6 Jbar = Vec6::Zero();
    for (std::size_t k = 0; k < NK; ++k) Jbar += S.w[k] * S.T[k];
    const double eb = L.t_bar - rho, c2 = cfg.st.c * cfg.st.c;
    const double V  = (1.0 + eb * eb / c2) / cfg.st.p0 + L.var_t;
    // I_d = (φ(z)/σ)²·[m/π + (1 − m)(1 − t̄/ρ_max)/(1 − π)], both ratios in log space (Mills ratio in either tail)
    const double Id = (m * std::exp(2.0 * lphn - log_pi)
                       + (1.0 - m) * (1.0 - L.t_bar / rho_max) * std::exp(2.0 * lphn - log_Phi_upper(-zn))) / (sigma * sigma);
    out.W = (L.r_member / V) * (Jbar * Jbar.transpose()) + ((1.0 - L.o_front) * Id) * (S.dd[i0] * S.dd[i0].transpose());
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
    if (not grad) return detail::approach<false>(r, kBeamEpsTangent).d;
    const detail::Approach A = detail::approach<true>(r, kBeamEpsTangent);
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
