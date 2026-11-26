/*
 * countable_size_std.h — the published size std (`size_std_m`, shared-engine Task S1), both halves, header-only and pure.
 *
 * Publisher: the per-axis 1-σ of the box extents the node PUBLISHES, ordered like them {width_m, depth_m, height_m}.
 *   width_m = s.w, depth_m = s.h, height_m = VerticalSpan.height(H) — so σ_w = √Σ(3,3), σ_d = √Σ(4,4) (state order cx, cy, H, w, h, yaw),
 *   and σ_height = √Σ(2,2) when the body is floor-anchored (its extent IS H) and 0 when it hangs (its extent is the class constant e,
 *   a parameter, not a DOF). Σ is the belief's θθ block: with the persistent latents on it is the joint's θθ marginal, so the shape
 *   and wall latents are already integrated in — the published extents are θ's mean (no latent added to the mean), so this is
 *   exactly their marginal. A c2v class publishes the state's own w, h and so the std of those same two entries.
 * MEANING / SCALE: σ is the posterior at the LAST COVARIANCE PUBLISH (it moves with rt_covariance, not with the extents' 3 mm
 * dead-band). It is the RAW √Σ — rt_covariance carries cfg.rt_cov_scale·Σ (a controller-governor inflation), size_std_m does not.
 * HOOD: height σ is 0 (height_m is the constant e) while rt_covariance z carries H's variance (z = H − e, ∂z/∂H = 1): the two are
 * different quantities by design. LATENTS: Σ is the θθ block the belief carries. With params.persistent on it is the joint's θθ
 * marginal (latents integrated); with it off it is the per-frame Σ with its common-mode floor folded in (countable_belief.h) —
 * the same Σ rt_covariance publishes, so the two agree either way; nothing is guarded because nothing is wrong, only differently
 * derived.
 * Reader: a neighbour's footprint axes are its width_m / depth_m, so {σ_w, σ_h} = the attribute's first two entries. An absent,
 * malformed or non-finite attribute (a legacy agent) is nullopt and the caller keeps the manifest interim (countable_neighbour_size.h).
 */
#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <vector>
#include <Eigen/Core>

#include "countable_vertical_span.h"

namespace rc
{

inline std::vector<float> published_size_std(const Eigen::Matrix<float, 6, 6>& S, rc::countable::VerticalSpan span)
{
    const auto sd = [&](int i) { return std::sqrt(std::max(S(i, i), 0.0f)); };   // a variance is ≥ 0: a rounding negative is 0, not NaN
    return {sd(3), sd(4), span.hangs() ? 0.0f : sd(2)};
}

// WHEN it is published (S1 fix round 1): with rt_covariance, on the rtcov::publish gate (`cov_published` = its return), and once
// when the belief is first seeded (`ever` false ⇒ due, then set). Pure so the policy is testable without a graph.
inline std::optional<std::vector<float>> size_std_to_publish(bool cov_published, bool& ever, const Eigen::Matrix<float, 6, 6>& S,
                                                             rc::countable::VerticalSpan span)
{
    if (not (cov_published or not ever)) return std::nullopt;
    ever = true;
    return published_size_std(S, span);
}

// {σ_w, σ_h} of a neighbour's footprint from its published size_std_m; nullopt when it is not a usable {w, d, height} std.
inline std::optional<std::array<float, 2>> neighbour_footprint_std(const std::vector<float>& v)
{
    if (v.size() != 3) return std::nullopt;
    for (const float x : v) if (not (std::isfinite(x) and x >= 0.0f)) return std::nullopt;
    return std::array<float, 2>{v[0], v[1]};
}

}  // namespace rc
