/*
 * view_atlas.h — per-object appearance atlas (spec: VIEW_ATLAS.md, same directory).
 *
 * For one object: 12 azimuth bins of DINOv3 shape in the OBJECT frame + one colour prototype. Evidence is
 * counted in VIEWS, not pixels, and a repeat look from the same place adds ~nothing (update()).
 * Header-only, pure Eigen: no DSR/DDS/OpenCV/ONNX, so every function here is unit-testable in isolation.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace rc::atlas
{

inline constexpr float kPi = std::numbers::pi_v<float>;

/// Wrap an angle to (-pi, pi].
inline float wrap_pi(float a)
{
    float r = std::remainder(a, 2.f * kPi);
    if (r <= -kPi) r += 2.f * kPi;
    return r;
}

/// Model constants. Defaults are the 2026-10-17 spike's measurements; view_atlas_constants.txt (fitted,
/// Task 6) overrides them via atlas_io.h::load_params. FLAGGED (not fittable, see VIEW_ATLAS.md §3.5/§4):
/// n0, lambda_nats, max_rate_hz, pi_move, quality_px_scale.
struct AtlasParams
{
    int   n_bins          = 12;
    float ell_rad         = 0.90f;   // shape relaxation kernel kappa(D) = exp(-D^2 / 2 ell^2)
    float repeat_ell_rad  = 0.10f;   // views closer than this in azimuth are repeats...
    float repeat_ell_logr = 0.20f;   // ...and closer than this in log-range
    int   poses_per_bin   = 4;       // stored view poses per bin for the repeat discount
    float n0              = 1.0f;    // FLAGGED: prior pseudo-count, "worth one view"
    Eigen::Vector2f mu_own{0.96f, 0.97f};      // (s_shape at gap 0, s_colour) for "it is this object"
    Eigen::Vector2f mu_bg {0.87f, 0.50f};      // same for "another object of this class"
    Eigen::Matrix2f cov_own = Eigen::Vector2f(9e-4f, 9e-4f).asDiagonal();
    Eigen::Matrix2f cov_bg  = Eigen::Vector2f(2.5e-3f, 6.25e-2f).asDiagonal();
    Eigen::Matrix2f cov_n   = Eigen::Vector2f(9e-4f, 9e-4f).asDiagonal();   // per-view noise
    float lambda_nats      = 0.05f;  // FLAGGED: price of one encoding (CPU cost, outside the model)
    float max_rate_hz      = 4.0f;   // FLAGGED: machine guard, not part of the model
    float pi_move          = 0.2f;   // FLAGGED: prior probability an archived object was moved
    float quality_px_scale = 2000.f; // FLAGGED: mask pixel count at which resolution quality is 1 - 1/e
};

struct Bin
{
    Eigen::VectorXf             mean;          // unit vector, or zeros while n == 0
    float                       n = 0.f;       // effective view count
    float                       bearing = 0.f; // circular mean bearing of the views it holds (object frame)
    std::vector<Eigen::Vector2f> poses;        // last poses (bearing, log range), newest last
    std::uint64_t               stamp_ms = 0;
};

struct AtlasState
{
    int              shape_dim = 0, colour_dim = 0;
    std::vector<Bin> bins;
    Eigen::VectorXf  colour;                   // unit vector, or zeros while n_c == 0
    float            n_c = 0.f;
};

/// One encoded view, already reduced to the object frame by the caller.
struct ViewObs
{
    Eigen::VectorXf shape, colour;
    float bearing_rad = 0.f;   // object-frame azimuth of the camera seen from the object
    float range_m = 1.f;
    float quality = 1.f;       // view_quality(): 0 = no evidence, 1 = full view
    std::uint64_t stamp_ms = 0;
};

/// Continuous view quality in [0,1]: untruncated fraction x visible fraction x resolution factor.
inline float view_quality(float trunc_frac, float visible_frac, int mask_px, float px_scale)
{
    if (not (mask_px > 0) or not (px_scale > 0.f)) return 0.f;
    const float untrunc = std::clamp(1.f - trunc_frac, 0.f, 1.f);     // fractions are bounded by definition
    const float vis     = std::clamp(visible_frac, 0.f, 1.f);
    const float res     = 1.f - std::exp(-float(mask_px) / px_scale);
    return untrunc * vis * res;
}

struct Gauss2 { Eigen::Vector2f mu = Eigen::Vector2f::Zero(); Eigen::Matrix2f cov = Eigen::Matrix2f::Identity(); };

inline float logpdf(const Eigen::Vector2f& x, const Gauss2& g)
{
    const Eigen::Vector2f d = x - g.mu;
    const Eigen::LLT<Eigen::Matrix2f> llt(g.cov);
    const Eigen::Vector2f z = llt.matrixL().solve(d);
    const float logdet = 2.f * (std::log(llt.matrixL()(0, 0)) + std::log(llt.matrixL()(1, 1)));
    return -0.5f * z.squaredNorm() - 0.5f * logdet - std::log(2.f * kPi);
}

/// Observation-quality noise added to BOTH hypotheses: q -> 0 makes them indistinguishable (ratio -> 0).
inline Eigen::Matrix2f quality_cov(const AtlasParams& p, float quality)
{
    return p.cov_n * ((1.f - quality) / quality);
}

/// "Another object of this class" (H_bg).
inline Gauss2 background(const AtlasParams& p, float quality)
{
    return {p.mu_bg, p.cov_bg + quality_cov(p, quality)};
}

class ViewAtlas
{
public:
    ViewAtlas(const AtlasParams& p, int shape_dim, int colour_dim) : p_(p)
    {
        s_.shape_dim = shape_dim; s_.colour_dim = colour_dim;
        s_.bins.resize(std::size_t(p_.n_bins));
        for (auto& b : s_.bins) b.mean = Eigen::VectorXf::Zero(shape_dim);
        s_.colour = Eigen::VectorXf::Zero(colour_dim);
    }
    static ViewAtlas from_state(const AtlasParams& p, AtlasState s)
    {
        ViewAtlas a(p, s.shape_dim, s.colour_dim);
        a.s_ = std::move(s);
        return a;
    }
    [[nodiscard]] const AtlasState&  state()  const noexcept { return s_; }
    [[nodiscard]] const AtlasParams& params() const noexcept { return p_; }

    [[nodiscard]] int bin_of(float bearing) const
    {
        const float u = (wrap_pi(bearing) + kPi) / (2.f * kPi);        // (0, 1]
        return std::clamp(int(std::floor(u * float(p_.n_bins))), 0, p_.n_bins - 1);   // index safety only
    }
    [[nodiscard]] float kernel(float d) const
    {
        if (std::isinf(d)) return 0.f;
        return std::exp(-0.5f * d * d / (p_.ell_rad * p_.ell_rad));
    }
    /// N(beta) = n0 + sum_b n_b kappa(beta - bearing_b): how well the object is known from beta.
    [[nodiscard]] float effective_count(float bearing) const
    {
        float N = p_.n0;
        for (const auto& b : s_.bins)
            if (b.n > 0.f) N += b.n * kernel(wrap_pi(bearing - b.bearing));
        return N;
    }
    /// Information (nats) one view of this quality at this bearing would add: 1/2 log((N + w) / N).
    [[nodiscard]] float learning_value(float bearing, float quality) const
    {
        if (not (quality > 0.f)) return 0.f;
        const float N = effective_count(bearing);
        return 0.5f * std::log((N + quality) / N);
    }
    /// Angular distance to the nearest populated bin; +inf when the atlas holds no shape at all.
    [[nodiscard]] float gap(float bearing) const
    {
        float g = std::numeric_limits<float>::infinity();
        for (const auto& b : s_.bins)
            if (b.n > 0.f) g = std::min(g, std::fabs(wrap_pi(bearing - b.bearing)));
        return g;
    }

    struct Prediction { Eigen::VectorXf shape; float gap_rad = 0.f; bool has_shape = false; };
    [[nodiscard]] Prediction predict(float bearing) const
    {
        Prediction pr;
        pr.shape = Eigen::VectorXf::Zero(s_.shape_dim);
        for (const auto& b : s_.bins)
            if (b.n > 0.f) pr.shape += b.n * kernel(wrap_pi(bearing - b.bearing)) * b.mean;
        pr.gap_rad = gap(bearing);
        const float nrm = pr.shape.norm();
        pr.has_shape = nrm > 0.f;
        if (pr.has_shape) pr.shape /= nrm;
        return pr;
    }

    /// Fold one encoded view in. Weight = quality x (1 - similarity to the bin's stored view poses), so a
    /// repeat look from the same place adds ~nothing. No process noise: appearance does not drift.
    void update(const ViewObs& o)
    {
        if (not (o.quality > 0.f) or not (o.range_m > 0.f)) return;
        if (o.shape.size() != s_.shape_dim or o.colour.size() != s_.colour_dim) return;
        const float sn = o.shape.norm(), cn = o.colour.norm();
        if (not (sn > 0.f) or not std::isfinite(sn) or not std::isfinite(cn)) return;

        const float b  = wrap_pi(o.bearing_rad);
        const float lr = std::log(o.range_m);
        Bin& bin = s_.bins[std::size_t(bin_of(b))];
        const float w = o.quality * (1.f - repeat_similarity(bin, b, lr));

        if (w > 0.f)
        {
            Eigen::VectorXf m = bin.n * bin.mean + w * (o.shape / sn);
            bin.mean = m / m.norm();
            const float sx = bin.n * std::sin(bin.bearing) + w * std::sin(b);
            const float cx = bin.n * std::cos(bin.bearing) + w * std::cos(b);
            bin.bearing = std::atan2(sx, cx);
            bin.n += w;
            if (cn > 0.f)
            {
                Eigen::VectorXf c = s_.n_c * s_.colour + w * (o.colour / cn);
                s_.colour = c / c.norm();
                s_.n_c += w;
            }
        }
        bin.poses.emplace_back(b, lr);
        if (int(bin.poses.size()) > p_.poses_per_bin)
            bin.poses.erase(bin.poses.begin() + most_redundant(bin.poses));
        bin.stamp_ms = o.stamp_ms;
    }

    /// (s_shape, s_colour): the view against the shape predicted at `bearing` and against the colour prototype.
    /// A channel with no stored evidence reads 0 — its hypothesis has relaxed to the background there.
    [[nodiscard]] Eigen::Vector2f similarities(const ViewObs& o, float bearing) const
    {
        const auto pr = predict(bearing);
        const float sn = o.shape.norm(), cn = o.colour.norm();
        const float sd = (pr.has_shape and sn > 0.f) ? o.shape.dot(pr.shape) / sn : 0.f;
        const float sc = (s_.n_c > 0.f and cn > 0.f) ? o.colour.dot(s_.colour) / cn : 0.f;
        return {sd, sc};
    }

    /// H_k at this bearing: relaxes from (mu_own, cov_own) to the background as the angular gap grows
    /// (shape) or as evidence vanishes (both). Cholesky factors are blended, so the covariance is always
    /// PSD and equals cov_bg EXACTLY when there is no evidence — that is what makes the ratio exactly 0.
    [[nodiscard]] Gauss2 hypothesis(float bearing, float quality) const
    {
        const float Ns = effective_count(bearing) - p_.n0;
        const float r_d = kernel(gap(bearing)) * Ns / (Ns + p_.n0);
        const float r_c = s_.n_c / (s_.n_c + p_.n0);
        const Eigen::Matrix2f R = Eigen::Vector2f(r_d, r_c).asDiagonal();
        const Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        const Eigen::Matrix2f Lb = Eigen::LLT<Eigen::Matrix2f>(p_.cov_bg).matrixL();
        const Eigen::Matrix2f Lo = Eigen::LLT<Eigen::Matrix2f>(p_.cov_own).matrixL();
        const Eigen::Matrix2f L = (I - R) * Lb + R * Lo;
        Gauss2 g;
        g.mu  = p_.mu_bg + R * (p_.mu_own - p_.mu_bg);
        g.cov = L * L.transpose() + quality_cov(p_, quality);
        return g;
    }

    /// Lambda = log N(s; H_k) - log N(s; H_bg). Exactly 0 when the atlas holds no evidence.
    [[nodiscard]] float log_ratio(const ViewObs& o, float bearing) const
    {
        if (not (o.quality > 0.f)) return 0.f;
        const Eigen::Vector2f s = similarities(o, bearing);
        return logpdf(s, hypothesis(bearing, o.quality)) - logpdf(s, background(p_, o.quality));
    }

    /// Yaw unknown or uncertain: marginalise the cyclic shift, log sum_k w_k exp(Lambda(bearing + k*bin)).
    [[nodiscard]] float log_ratio_marginal(const ViewObs& o, float bearing, const std::vector<float>& shift_logw) const
    {
        const float step = 2.f * kPi / float(p_.n_bins);
        float m = -std::numeric_limits<float>::infinity();
        std::vector<float> t(shift_logw.size());
        for (std::size_t k = 0; k < shift_logw.size(); ++k)
        {
            t[k] = std::isinf(shift_logw[k]) ? shift_logw[k]
                                             : shift_logw[k] + log_ratio(o, bearing + step * float(k));
            m = std::max(m, t[k]);
        }
        if (std::isinf(m)) return 0.f;
        float acc = 0.f;
        for (float v : t) acc += std::exp(v - m);
        return m + std::log(acc);
    }

private:
    /// Repeat-kernel similarity of two poses (bearing, log range): 1 = the same place, 0 = far apart.
    [[nodiscard]] float pose_similarity(const Eigen::Vector2f& a, const Eigen::Vector2f& b) const
    {
        const float da = wrap_pi(a.x() - b.x()) / p_.repeat_ell_rad;
        const float dr = (a.y() - b.y()) / p_.repeat_ell_logr;
        return std::exp(-0.5f * (da * da + dr * dr));
    }
    [[nodiscard]] float repeat_similarity(const Bin& bin, float b, float lr) const
    {
        const Eigen::Vector2f view(b, lr);
        float k = 0.f;
        for (const auto& ps : bin.poses) k = std::max(k, pose_similarity(view, ps));
        return k;
    }
    /// Index of the stored pose most similar to any OTHER stored pose (ties: oldest). Redundancy-based
    /// eviction keeps the distinct views; recency eviction let a run of repeat looks flush them.
    [[nodiscard]] std::ptrdiff_t most_redundant(const std::vector<Eigen::Vector2f>& poses) const
    {
        std::ptrdiff_t arg = 0;
        float best = -1.f;
        for (std::size_t i = 0; i < poses.size(); ++i)
        {
            float k = 0.f;
            for (std::size_t j = 0; j < poses.size(); ++j)
                if (j != i) k = std::max(k, pose_similarity(poses[i], poses[j]));
            if (k > best) { best = k; arg = std::ptrdiff_t(i); }
        }
        return arg;
    }

    AtlasParams p_;
    AtlasState  s_;
};

/// Log-weights over the n_bins cyclic shifts from a wrapped-Gaussian yaw posterior of sd sigma.
/// sigma <= 0: all weight on shift 0. Normalised (log-sum-exp == 0).
inline std::vector<float> shift_log_weights(int n_bins, float yaw_sigma_rad)
{
    std::vector<float> w(std::size_t(n_bins), -std::numeric_limits<float>::infinity());
    if (not (yaw_sigma_rad > 0.f)) { w[0] = 0.f; return w; }
    const float step = 2.f * kPi / float(n_bins);
    float m = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < n_bins; ++k)
    {
        const float d = wrap_pi(step * float(k)) / yaw_sigma_rad;
        w[std::size_t(k)] = -0.5f * d * d;
        m = std::max(m, w[std::size_t(k)]);
    }
    float acc = 0.f;
    for (float v : w) acc += std::exp(v - m);
    const float lse = m + std::log(acc);
    for (float& v : w) v -= lse;
    return w;
}

/// Where was it? Mixture of "stayed" (Gaussian at the archived pose) and "moved" (uniform over the room).
inline float location_log_lik(const Eigen::Vector2f& x, const Eigen::Vector2f& mu, const Eigen::Matrix2f& cov,
                              float pi_move, float room_area_m2)
{
    const float stay = std::exp(logpdf(x, {mu, cov}));
    return std::log((1.f - pi_move) * stay + pi_move / room_area_m2);
}

struct ReidCandidate { float log_ratio = 0.f, log_loc = 0.f, log_prior = 0.f; };

/// Posterior over {candidate 0..n-1, new}; the last element is "new".
inline std::vector<float> reid_posterior(const std::vector<ReidCandidate>& c, float log_prior_new, float log_loc_new)
{
    std::vector<float> lw;
    for (const auto& k : c) lw.push_back(k.log_prior + k.log_loc + k.log_ratio);
    lw.push_back(log_prior_new + log_loc_new);
    const float m = *std::max_element(lw.begin(), lw.end());
    float acc = 0.f;
    for (float& v : lw) { v = std::exp(v - m); acc += v; }
    for (float& v : lw) v /= acc;
    return lw;
}

inline float entropy(const std::vector<float>& p)
{
    float h = 0.f;
    for (float v : p) if (v > 0.f) h -= v * std::log(v);
    return h;
}

struct IdHyp { Gauss2 own, bg; float log_prior = 0.f, log_loc = 0.f; };

/// Expected entropy drop over {candidates, new} from encoding one view (nats), by Monte Carlo: under
/// "it is j", candidate j's similarities follow its own hypothesis and every other candidate's follow
/// that candidate's background; under "new", all follow their backgrounds.
inline float identity_value(const std::vector<IdHyp>& c, float log_prior_new, float log_loc_new,
                            int n_samples, std::uint32_t seed)
{
    if (c.empty() or n_samples <= 0) return 0.f;
    std::vector<ReidCandidate> prior;
    for (const auto& h : c) prior.push_back({0.f, h.log_loc, h.log_prior});
    const auto p0 = reid_posterior(prior, log_prior_new, log_loc_new);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    auto draw = [&](const Gauss2& h) {
        const Eigen::Matrix2f L = Eigen::LLT<Eigen::Matrix2f>(h.cov).matrixL();
        return Eigen::Vector2f(h.mu + L * Eigen::Vector2f(g(rng), g(rng)));
    };
    float expected_h = 0.f;
    for (std::size_t t = 0; t <= c.size(); ++t)            // t == c.size() is "new"
    {
        float acc = 0.f;
        for (int s = 0; s < n_samples; ++s)
        {
            std::vector<ReidCandidate> obs = prior;
            for (std::size_t j = 0; j < c.size(); ++j)
            {
                const Eigen::Vector2f x = draw(t == j ? c[j].own : c[j].bg);
                obs[j].log_ratio = logpdf(x, c[j].own) - logpdf(x, c[j].bg);
            }
            acc += entropy(reid_posterior(obs, log_prior_new, log_loc_new));
        }
        expected_h += p0[t] * acc / float(n_samples);
    }
    return std::max(0.f, entropy(p0) - expected_h);   // MC noise only; mutual information is >= 0
}

} // namespace rc::atlas
