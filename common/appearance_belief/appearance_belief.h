/*
 * appearance_belief.h  —  shared per-instance APPEARANCE (albedo chromaticity) belief
 *
 * SHARED, header-only across the concept agents (table/chair/refrigerator/cabinet/…): one 3-DOF Gaussian
 * per instance over the object's chromaticity, fed by the retina's per-mask colour summary
 * (MaskColor.*, see retina/src/graph_publisher.cpp) and read only by the DISPLAY path — the agent
 * publishes its MAP as `mesh_color_rgb` and the retina's 3D viewer tints the instance's mesh with it.
 *
 * SCOPE — DISPLAY ONLY. Nothing here feeds any geometric fit, association gate, or existence belief. That
 * is a deliberate containment decision, not an oversight: a colour channel derived from tens of thousands
 * of highly correlated mask pixels is exactly the kind of evidence that, wired into the metric belief
 * naively, would out-precision every geometric channel and poison the pose. The producer's grid-cell
 * aggregation already keeps its reported variance honest (see mask_color_summary), but until that is
 * validated live the safe move is that no inference reads it. Widening the scope later is a deliberate
 * act, not a side effect.
 *
 * WHY CHROMATICITY, NOT RGB — the observed pixel is albedo x irradiance + specular, and the viewer applies
 * its own ambient+diffuse shading on top. Publishing observed RGB would bake the room's lighting into the
 * asset and then shade it a second time. Normalising by (R+G+B) is invariant to the per-frame illumination
 * gain by construction: the dominant nuisance removed without inferring a light model, and without a
 * threshold anywhere.
 *
 * WHY NO OPTIMISER — with I_pred = albedo (x) shading, the residual is LINEAR in albedo, so the posterior
 * is conjugate Gaussian and available in closed form. Gradient descent on a linear-Gaussian problem is
 * strictly worse: slower, needs a step size, and does not hand back the covariance. One update() call per
 * frame per instance replaces the whole thing.
 *
 * Header-only (like existence_belief.h / lidar_ray_factor.h): pure Eigen, no DSR/DDS/OpenCV.
 */

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace rc::appearance
{

// Tuning for the appearance belief. These are scale choices, not gates — every one of them
// stays finite and continuous, and none of them switches behaviour on or off.
struct AppearanceParams
{
    // Prior: chromaticity is confined to the simplex (three components in [0,1] summing to 1), so the
    // widest honest prior is roughly uniform over it. sigma0 = 0.29 is the sd of a uniform on [0,1] —
    // deliberately wider than anything the data will produce, so the prior never fights the evidence.
    float prior_sigma = 0.29f;
    // Process noise per second. Real appearance drifts slowly: the lighting changes as the robot moves
    // through the room, and a surface can genuinely be repainted or re-covered. Without this the belief
    // saturates and freezes on whatever the first confident view happened to see. This is the same
    // rationale as inflate_for_age in common/ai_belief/recursive_laplace.h.
    float drift_sigma_per_s = 0.02f;
    // Cap on the number of grid cells one frame may contribute. A single frame's cells share the
    // illumination, the viewing angle and the exposure, so they are NOT independent samples of the
    // underlying albedo — only of the pixel noise. Letting a 400-cell mask count as 400 independent
    // observations would collapse the posterior in one frame and make the estimate hostage to that
    // frame's lighting. Saturating the per-frame information is the same common-mode discipline the
    // metric belief applies via Woodbury and the existence belief applies via tanh.
    float max_cells_per_frame = 16.0f;
    // Prior probability that a frame's mask is NOT this object: a wall, floor or table pixel run credited to it
    // (mask bleed, a wrong mask-to-instance association). Measured 2026-11-11 on the chat_e2e8/e2e9 traces:
    // 1-3 % of chair_35's frames were a neutral surface (chromaticity 0.346/0.334/0.320) with hundreds of cells,
    // and each one renamed the yellow chair "grey"/"white". Not a gate: observe() weighs every frame by its
    // posterior probability of being the object, continuously. Replaying the traces, any value from 0.005 to
    // 0.2 removes every one of those flips and keeps every true name, so its exact value hardly matters.
    float p_foreign_mask = 0.05f;
};

// 3-DOF diagonal Gaussian over chromaticity (r,g,b). Diagonal rather than full covariance because the
// producer reports per-channel scatter and the consumer (a display tint) has no use for the correlations;
// the simplex constraint that couples the channels is enforced by renormalising the mean, not by Sigma.
class AppearanceBelief
{
public:
    AppearanceBelief() = default;
    explicit AppearanceBelief(const AppearanceParams& p) : params_(p) {}

    // Fold in one frame's mask colour summary. chroma/var/n_cells come straight off a MaskSlice
    // (color_chroma / color_var / color_neff). n_cells <= 0 or a non-finite summary contributes nothing —
    // there is no state to guard against, so this is an early return, not a gate on the evidence.
    // `weight` in [0,1] scales the frame's precision: observe() passes its probability of being this object.
    void update(const Eigen::Vector3f& chroma, const Eigen::Vector3f& var, float n_cells, float weight = 1.0f)
    {
        if (!(n_cells > 0.0f) or !chroma.allFinite() or !var.allFinite())
            return;

        // Saturate the per-frame sample count (see max_cells_per_frame): one frame is close to ONE
        // independent look at the albedo however many cells it contains.
        const float n_eff = params_.max_cells_per_frame * n_cells
                          / (params_.max_cells_per_frame + n_cells);

        if (!initialized_)
        {
            mean_ = chroma;
            sigma_ = Eigen::Vector3f::Constant(params_.prior_sigma * params_.prior_sigma);
            initialized_ = true;
        }

        // Conjugate Gaussian update, per channel: precision adds, mean is the precision-weighted average.
        // The measurement variance of the frame's MEAN is the between-cell scatter divided by n_eff.
        for (int k = 0; k < 3; ++k)
        {
            const float r = std::max(var[k], kMinVar) / n_eff;   // variance of this frame's estimate
            const float prior_prec = 1.0f / std::max(sigma_[k], kMinVar);
            const float obs_prec   = weight / r;
            const float post_prec  = prior_prec + obs_prec;
            mean_[k]  = (prior_prec * mean_[k] + obs_prec * chroma[k]) / post_prec;
            sigma_[k] = 1.0f / post_prec;
        }
        renormalize();
    }

    // Grow the covariance to reflect elapsed time (drift). Call once per cycle with the wall dt, whether
    // or not the instance was observed — an unobserved object's colour becomes less certain, not more.
    void inflate_for_age(float dt_s)
    {
        if (not (dt_s > 0.0f))
            return;
        const float q = params_.drift_sigma_per_s * params_.drift_sigma_per_s * dt_s;
        const float cap = params_.prior_sigma * params_.prior_sigma;
        if (initialized_)
        {
            sigma_ += Eigen::Vector3f::Constant(q);
            sigma_ = sigma_.cwiseMin(Eigen::Vector3f::Constant(cap));   // never worse than the prior
        }
        if (lum_initialized_)   // its own flag: a pure-black object has brightness and no chromaticity
            lum_var_ = std::min(lum_var_ + q, cap);
    }

    // BRIGHTNESS channel (added 2026-11-11 for colour NAMES): chromaticity cannot tell black, grey and white
    // apart, brightness can. lum/var/n_cells come off a MaskSlice (color_lum / color_lum_var / color_neff),
    // and the same per-frame cap and the same drift apply. Unlike chromaticity, brightness is NOT invariant
    // to the light; in Webots the light is fixed, on the real robot the drift and the palette's width carry
    // it. Kept as its own initialised flag: a retina that publishes no brightness leaves it unknown, which
    // the naming reads as "no brightness evidence", never as dark.
    void update_lum(float lum, float var, float n_cells, float weight = 1.0f)
    {
        if (not (n_cells > 0.0f) or not std::isfinite(lum) or not std::isfinite(var))
            return;
        const float n_eff = params_.max_cells_per_frame * n_cells / (params_.max_cells_per_frame + n_cells);
        if (not lum_initialized_)
        {
            lum_ = lum;
            lum_var_ = params_.prior_sigma * params_.prior_sigma;
            lum_initialized_ = true;
        }
        const float r = std::max(var, kMinVar) / n_eff;
        const float prior_prec = 1.0f / std::max(lum_var_, kMinVar);
        const float obs_prec = weight / r;
        const float post_prec = prior_prec + obs_prec;
        lum_ = std::clamp((prior_prec * lum_ + obs_prec * lum) / post_prec, 0.0f, 1.0f);
        lum_var_ = 1.0f / post_prec;
    }
    // ONE FRAME, BOTH CHANNELS, ROBUST TO A FOREIGN MASK (2026-11-11). The frame is explained by one of two
    // hypotheses: it shows THIS object (its predictive density under the current belief: belief variance plus
    // the frame's own), or it shows something else (p_foreign_mask; density uniform over the chromaticity
    // simplex, 2 in (r,g), and uniform over brightness [0,1]). The frame then counts with its posterior
    // probability of being the object — a mixture likelihood, no cutoff. A lasting change still wins: the
    // drift widens the belief until the new colour is the likelier explanation (about 3 s at 10 Hz).
    // A channel the belief has not initialised yet is equally likely under both and is left out of the ratio.
    void observe(const Eigen::Vector3f& chroma, const Eigen::Vector3f& var, float n_cells,
                 float lum, float lum_var, float lum_n)
    {
        const bool has_c = n_cells > 0.0f and chroma.allFinite() and var.allFinite();
        const bool has_l = lum_n > 0.0f and std::isfinite(lum) and std::isfinite(lum_var);
        double log_in = 0.0, log_out = 0.0;
        bool compared = false;
        const auto gauss = [](double x, double mu, double v)
        { return -0.5 * ((x - mu) * (x - mu) / v + std::log(2.0 * 3.14159265358979323846 * v)); };
        if (has_c and initialized_)
        {
            const float n_eff = params_.max_cells_per_frame * n_cells / (params_.max_cells_per_frame + n_cells);
            for (int k = 0; k < 2; ++k)   // b = 1 - r - g carries nothing more
                log_in += gauss(chroma[k], mean_[k], sigma_[k] + std::max(var[k], kMinVar) / n_eff);
            log_out += std::log(2.0);
            compared = true;
        }
        if (has_l and lum_initialized_)
        {
            const float n_eff = params_.max_cells_per_frame * lum_n / (params_.max_cells_per_frame + lum_n);
            log_in += gauss(lum, lum_, lum_var_ + std::max(lum_var, kMinVar) / n_eff);
            compared = true;   // uniform on [0,1]: log density 0
        }
        last_weight_ = 1.0f;
        if (compared)
        {
            const double a = std::log1p(-double(params_.p_foreign_mask)) + log_in;
            const double b = std::log(double(params_.p_foreign_mask)) + log_out;
            last_weight_ = static_cast<float>(1.0 / (1.0 + std::exp(b - a)));
        }
        if (has_c) update(chroma, var, n_cells, last_weight_);
        if (has_l) update_lum(lum, lum_var, lum_n, last_weight_);
    }
    /// The last observe()'s probability that its frame showed this object (1 before any).
    [[nodiscard]] float last_weight() const noexcept { return last_weight_; }

    [[nodiscard]] bool lum_initialized() const noexcept { return lum_initialized_; }
    [[nodiscard]] float lum() const noexcept { return lum_; }              // MAP brightness in [0,1]
    [[nodiscard]] float lum_variance() const noexcept { return lum_var_; }

    [[nodiscard]] bool initialized() const noexcept { return initialized_; }
    // MAP chromaticity, normalised to sum to 1. Meaningless before the first update — check initialized().
    [[nodiscard]] const Eigen::Vector3f& map() const noexcept { return mean_; }
    [[nodiscard]] const Eigen::Vector3f& variance() const noexcept { return sigma_; }

    // Confidence in [0,1]: 0 while the belief is as uncertain as the prior, approaching 1 as the posterior
    // sharpens. The viewer multiplies the tint by this so the colour fades IN as evidence accumulates
    // rather than popping to a wrong value on the first frame. Continuous — deliberately not a "confident
    // enough yet?" boolean.
    [[nodiscard]] float confidence() const noexcept
    {
        if (!initialized_)
            return 0.0f;
        const float prior_var = params_.prior_sigma * params_.prior_sigma;
        const float mean_var  = sigma_.mean();
        return std::clamp(1.0f - mean_var / prior_var, 0.0f, 1.0f);
    }

private:
    // Project the mean back onto the chromaticity simplex. The three channels are estimated independently
    // but are constrained to sum to 1; renormalising is the cheapest correct projection and keeps the
    // published value meaningful as a chromaticity.
    void renormalize()
    {
        mean_ = mean_.cwiseMax(0.0f);
        const float s = mean_.sum();
        if (s > 1e-6f)
            mean_ /= s;
    }

    static constexpr float kMinVar = 1e-9f;   // numerical guard only (divide-by-zero), not a modelling floor

    AppearanceParams params_{};
    Eigen::Vector3f  mean_  = Eigen::Vector3f::Constant(1.0f / 3.0f);   // neutral grey chromaticity
    Eigen::Vector3f  sigma_ = Eigen::Vector3f::Constant(0.29f * 0.29f);
    bool             initialized_ = false;
    float            lum_ = 0.5f;
    float            lum_var_ = 0.29f * 0.29f;
    bool             lum_initialized_ = false;
    float            last_weight_ = 1.0f;
};

}   // namespace rc::appearance
