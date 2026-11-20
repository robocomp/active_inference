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
 * t(δ) on the model's convex parts and their θ-derivatives) is built on top of it (Task 17, appended below).
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

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

}  // namespace rc::ai::beam
