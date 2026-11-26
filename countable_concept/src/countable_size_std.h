/*
 * countable_size_std.h — the published size std (`size_std_m`, shared-engine Task S1), both halves, header-only and pure.
 *
 * Publisher: the per-axis 1-σ of the box extents the node PUBLISHES, ordered like them {width_m, depth_m, height_m}.
 *   width_m = s.w, depth_m = s.h, height_m = VerticalSpan.height(H) — so σ_w = √Σ(3,3), σ_d = √Σ(4,4) (state order cx, cy, H, w, h, yaw),
 *   and σ_height = √Σ(2,2) when the body is floor-anchored (its extent IS H) and 0 when it hangs (its extent is the class constant e,
 *   a parameter, not a DOF). Σ is the belief's θθ block: with the persistent latents on it is the joint's θθ marginal, so the shape
 *   and wall latents are already integrated in — the published extents are θ's mean (no latent added to the mean), so this is
 *   exactly their marginal. A c2v class publishes the state's own w, h and so the std of those same two entries.
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

// {σ_w, σ_h} of a neighbour's footprint from its published size_std_m; nullopt when it is not a usable {w, d, height} std.
inline std::optional<std::array<float, 2>> neighbour_footprint_std(const std::vector<float>& v)
{
    if (v.size() != 3) return std::nullopt;
    for (const float x : v) if (not (std::isfinite(x) and x >= 0.0f)) return std::nullopt;
    return std::array<float, 2>{v[0], v[1]};
}

}  // namespace rc
