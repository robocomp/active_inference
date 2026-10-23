/*
 * door_body_evidence.h — the ROBOT'S OWN BODY IN THE APERTURE as evidence about the leaf (Task C, 2026-10-23).
 *
 * gold3 leg 4: the robot drove through DOOR_2 (the actuator had opened it) while door_concept's P(open) for the same door
 * stayed 0.65-0.79 — the beams from outside the wall never settled the leaf, and nothing counted the strongest evidence of
 * all: the body itself went through. A leaf that blocks the aperture cannot be passed without contact, so a body whose
 * CENTRE lies inside the opening (within its span, within the wall's thickness) is incompatible with a leaf whose clear
 * span is narrower than the body. That is a likelihood over the SAME φ grid estimate_phi scores (door_phi_posterior.h
 * phi_grid), added to the frame's φ likelihood once (door_fit.cpp), so it rides the leaf channel's own pooling: each frame
 * is ε-contaminated (|Δ| ≤ log 1/ε) and the frames of one viewpoint are one observation (door_leaf_channel.h).
 *
 *   p_in = P(centre in span) · P(centre inside the wall thickness), under the robot's pose covariance projected on the
 *          opening's axes (a pose that is wrong by more than the opening's size says almost nothing — the covariance
 *          widens p_in's spread, not a gate).
 *   L(φ) = (1 − p_in) + p_in · [ε_c + (1 − ε_c)·1(clear_width(w, φ) ≥ body width)]
 *
 * Beside the door (centre outside the span/band): p_in → 0 ⇒ L ≡ 1, log L ≡ 0 EXACTLY — no change. φ ≥ the clear angle is
 * a geometric fact (door_passability.h clear_width), not a tuned cutoff.
 *
 * ⚑ MODEL CONSTANT, flagged: kBodyThroughLeafEps = ε_c = 1e-3, P(the centre reads inside the opening although the leaf
 *   blocks it) beyond what the pose covariance already prices: the base is stopped by a leaf (contact), so the only way
 *   is a pushed leaf swinging aside, which is the open case anyway. A model constant, no measurement; the per-frame
 *   ε = 0.03 contamination of the leaf channel bounds what any single frame can claim regardless.
 *
 * Header-only, no DSR / Qt.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "door_passability.h"

namespace rc::door
{

inline constexpr double kBodyThroughLeafEps = 1e-3;

namespace body_detail
{
[[nodiscard]] inline double ncdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
}

/// P(the body centre lies inside the opening): (u, y) = the centre in the opening frame (u along the wall from the
/// aperture centre, y across it), `cov` its 2x2 covariance in the SAME axes, half_w the aperture half-width, half_t the
/// wall's half-thickness. The 1e-8 is numerical (a zero covariance is a legal frozen/synthetic input).
[[nodiscard]] inline float body_in_aperture_prob(float u, float y, const Eigen::Matrix2f& cov_uy, float half_w, float half_t)
{
    using body_detail::ncdf;
    const double su = std::sqrt(double(std::max(0.0f, cov_uy(0, 0))) + 1e-8);
    const double sy = std::sqrt(double(std::max(0.0f, cov_uy(1, 1))) + 1e-8);
    const double pu = ncdf((half_w - u) / su) * ncdf((half_w + u) / su);
    const double py = ncdf((half_t - y) / sy) * ncdf((half_t + y) / sy);
    return float(std::clamp(pu * py, 0.0, 1.0));
}

/// The log-likelihood of "the body is at the opening" over the φ grid [lo, hi] (n points). Empty when p_in is exactly 0.
[[nodiscard]] inline std::vector<float> body_phi_loglik(float p_in, float aperture_w, float body_width,
                                                        float lo, float hi, int n)
{
    std::vector<float> ll;
    if (not (p_in > 0.0f) or n < 2) return ll;
    ll.resize(std::size_t(n));
    const double p = std::clamp(double(p_in), 0.0, 1.0);
    for (int k = 0; k < n; ++k)
    {
        const float phi = lo + (hi - lo) * float(k) / float(n - 1);
        const double fits = clear_width(aperture_w, phi) >= body_width ? 1.0 : 0.0;
        const double like = (1.0 - p) + p * (kBodyThroughLeafEps + (1.0 - kBodyThroughLeafEps) * fits);
        ll[std::size_t(k)] = float(std::log(like));
    }
    return ll;
}

}  // namespace rc::door
