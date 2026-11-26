/*
 * S1 DONE: the published `size_std_m` attribute (countable_size_std.h) is now the primary source; this file stays as the FALLBACK for a
 * neighbour that does not publish it (a legacy agent), logged once per class. The text below describes that fallback.
 *
 * countable_neighbour_size.h — the interim size std of a neighbour's published box (plan rev 2 F1.7; Fable order review §2.6).
 *
 * A neighbour's belief enters the update as a foreign density (G3, foreign_box.h) with a std per DOF. Position and yaw come from
 * the neighbour's rt_covariance; the SIZE has no published std yet (the cortex attribute is Task S1), and it was 0 — a
 * known-false input: it under-explains the leaked returns at the neighbour's edges, exactly where the leak is.
 * Interim: the neighbour CLASS's layer spread from its manifest, Δ/√12 per footprint axis (w, h) — the std of a layer lying
 * uniformly over Δ, the same marginal the shape latent carries. Class knowledge compiled into every agent (common/
 * concept_manifest), not a file another agent writes at run time; still flagged against "agents talk only through the graph",
 * and replaced by the published attribute in S1.
 * The neighbour's class is its node name without the trailing _<n> ("nightstand_2" → "nightstand"). No manifest, no
 * [shape].layer_spread_m, or a malformed one ⇒ {0, 0} (the previous behaviour) and the caller logs it once.
 */
#pragma once

#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "../../common/concept_manifest/layer_spread.h"

namespace rc
{

inline std::string node_class(std::string_view name)
{
    std::size_t end = name.size();
    while (end > 0 and std::isdigit(static_cast<unsigned char>(name[end - 1]))) --end;
    if (end < name.size() and end > 0 and name[end - 1] == '_') return std::string(name.substr(0, end - 1));
    return std::string(name);
}

// {σ_w, σ_h} = {Δ_w, Δ_h}/√12 from the manifest at `path` (state order cx, cy, H, w, h, yaw); nullopt when it declares none.
inline std::optional<std::array<float, 2>> neighbour_size_std(const std::string& path)
{
    if (not std::filesystem::exists(path)) return std::nullopt;
    try
    {
        const auto ls = rc::manifest::layer_spread(path, 6);
        if (ls.delta_m.size() != 6) return std::nullopt;
        const float k = 1.0f / std::sqrt(12.0f);
        return std::array<float, 2>{ls.delta_m[3] * k, ls.delta_m[4] * k};
    }
    catch (const std::exception&) { return std::nullopt; }
}

}  // namespace rc
