/*
 * appearance_belief.h  —  shared per-instance APPEARANCE (albedo chromaticity) belief
 *
 * SHARED, header-only across the concept agents (table/chair/refrigerator/cabinet/…): one 3-DOF Gaussian
 * per instance over the object's chromaticity, fed by the voxelizer's per-mask colour summary
 * (MaskColor.*, see voxelizer/src/graph_publisher.cpp) and read only by the DISPLAY path — the agent
 * publishes its MAP as `mesh_color_rgb` and the voxelizer's 3D viewer tints the instance's mesh with it.
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
    void update(const Eigen::Vector3f& chroma, const Eigen::Vector3f& var, float n_cells)
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
            const float obs_prec   = 1.0f / r;
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
        if (!initialized_ or !(dt_s > 0.0f))
            return;
        const float q = params_.drift_sigma_per_s * params_.drift_sigma_per_s * dt_s;
        sigma_ += Eigen::Vector3f::Constant(q);
        const float cap = params_.prior_sigma * params_.prior_sigma;
        sigma_ = sigma_.cwiseMin(Eigen::Vector3f::Constant(cap));   // never worse than the prior
    }

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
};

}   // namespace rc::appearance
