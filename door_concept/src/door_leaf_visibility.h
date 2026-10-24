/*
 * door_leaf_visibility.h — what the leaf's opening angle does to a WALL-REFERENCED detector (the 360 camera), and the
 * existence credit of having walked through the doorway. DSR-free, header-only (Task D2, gold12).
 *
 * THE DEFECT (gold12, DOOR_1): the robot crossed a door whose leaf stood open at ~92 deg. The ricoh detector looks for a
 * door as a panel IN the wall; an open leaf is edge-on and a leaf's width off the wall, so it is not detected — and the
 * ricoh absence channel read that as "looked, door not there": L fell 3.5 nats and the crossed door was removed. The ZED
 * silhouette already marginalises phi; the ricoh channel did not. These are the same marginalisation, for the ricoh.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace rc::door
{

/// E_phi[ cos^2(phi) ] over the leaf-angle posterior: how much of the leaf still looks like a panel IN the wall plane.
/// 1 for a shut leaf, 0 for a leaf at 90 deg (edge-on: a wall-referenced detector sees a jamb, not a door), smooth
/// in between — a continuous covariate, NOT an open/closed cutoff. (Beyond 90 deg the leaf swings out the other side,
/// edge-on again then flat against the far wall: cos^2 is symmetric about 90 deg, which is the honest answer for a
/// detector that does not know which side the hinge is on.) `phi_curve` is (phi, unnormalised weight) as the phi
/// posterior leaves it; empty/zero-weight curve ⇒ the point estimate `phi_est` (a degenerate posterior).
[[nodiscard]] inline float leaf_closedness(const std::vector<std::pair<float, float>>& phi_curve, float phi_est)
{
    double num = 0.0, den = 0.0;
    for (const auto& [phi, w] : phi_curve)
    {
        if (not std::isfinite(phi) or not std::isfinite(w) or w <= 0.0f) continue;
        const double c = std::cos(static_cast<double>(phi));
        num += static_cast<double>(w) * c * c;
        den += static_cast<double>(w);
    }
    if (den > 0.0) return static_cast<float>(num / den);
    if (not std::isfinite(phi_est)) return 1.0f;           // no angle at all = the old behaviour (a shut leaf)
    const float c = std::cos(phi_est);
    return c * c;
}

/// The factor on a closed-leaf P(detect): P(closed-looking)·1 + P(open-looking)·open_rel. `open_rel` is a MODEL
/// PARAMETER (Ricoh.OpenDetectRel): the ricoh's detection rate of an open leaf RELATIVE to a shut one.
[[nodiscard]] inline float ricoh_detect_scale(float closedness, float open_rel)
{
    const float c = std::clamp(closedness, 0.0f, 1.0f);   // (cos^2 averages are already in [0,1]; guards float dust)
    return c + (1.0f - c) * std::clamp(open_rel, 0.0f, 1.0f);
}

/// The ricoh's detection rate of an OPEN leaf RELATIVE to a shut one, LEARNED where the ZED confirms the door: `open_hits` /
/// `open_trials` are the trials weighted by (1 - closedness) (the same Bernoulli trials the shut-leaf hit rate is learned from,
/// each counted in proportion to how open the leaf looked), `hit_shut` the shut-leaf hit rate. A Beta prior of `n_prior`
/// pseudo-trials around `rel_prior` (Ricoh.OpenDetectRel) stands in until there are open trials: no open trials ⇒ exactly
/// rel_prior; many ⇒ the data. Capped at 1 (an open leaf is not detected BETTER than a shut one).
[[nodiscard]] inline float learned_open_rel(float open_hits, float open_trials, float hit_shut, float rel_prior, float n_prior = 2.0f)
{
    const float open_hit = (open_hits + n_prior * rel_prior * hit_shut) / (open_trials + n_prior);
    return std::clamp(open_hit / std::max(hit_shut, 1e-3f), 0.0f, 1.0f);
}

/// log-likelihood ratio of "the robot's body passed through this aperture" for the doorway EXISTING versus being a wall:
///   P(crossed | doorway) = P(passable) (a closed doorway is not crossed),   P(crossed | wall) = leak,
/// leak = the probability a localisation error alone reports the body through a wall (Existence.CrossLeak, a model
/// parameter, not a gate). A crossing that HAPPENED can never count against the doorway: a P(passable) below the leak
/// is the stale part (the body is through), and such a crossing contributes 0, not a negative. Pass 1 when no
/// passability was measured (nothing contradicts the crossing).
[[nodiscard]] inline float crossing_logodds(float p_passable, float leak)
{
    const float l = std::clamp(leak, 1e-4f, 1.0f);
    const float p = std::clamp(p_passable, l, 1.0f);
    return std::log(p / l);
}

}  // namespace rc::door
