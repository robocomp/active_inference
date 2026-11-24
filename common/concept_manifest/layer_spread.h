/*
 * layer_spread.h — the [shape] block's layer spread of a <concept>.concept.toml (spec 2026-11-23 shared-engine G4).
 *
 *   [shape]
 *   layer_spread_m      = [Δ_0, …, Δ_{N−1}]   # one per state DOF, in [model].state order, metres (radians on an angle)
 *   layer_spread_source = "…"                 # MANDATORY with the array: the dimension convention each Δ comes from
 *
 * Δ is CLASS knowledge: the range over which the layer a sensor sees may lie on a typical real object of the class (a bed's
 * mattress vs its frame, a toilet's bowl vs its tank), never a Webots mesh. The engine's shape latent is uniform over it
 * (marginal Δ²/12, ρ = 1). Δ = 0 on a DOF with no layering.
 *
 * Read through rc::manifest::parse_manifest (toml++, manifest_toml.h), the same parser as load_geometry; toml++ parses
 * numbers locale-independently (manifest_load_test checks it under es_ES).
 */
#pragma once

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include "manifest_toml.h"

namespace rc::manifest {

struct LayerSpread
{
    std::vector<float> delta_m;   // Δ per DOF; empty ⇔ the manifest declares none
    std::string        source;    // where the values come from (non-empty whenever delta_m is)
};

// Absent [shape].layer_spread_m and layer_spread_source ⇒ {} (no layering declared). Malformed ⇒ throws std::runtime_error
// naming the file and the fault: unreadable or invalid TOML, an array without its source (or a blank source, or a source
// without its array), a non-array, a length other than n, an element that is not a finite non-negative number.
inline LayerSpread layer_spread(const std::string& path, int n)
{
    const toml::table m = parse_manifest(path);
    const toml::node_view arr = m.at_path("shape.layer_spread_m");
    const toml::node_view src = m.at_path("shape.layer_spread_source");
    if (not arr and not src) return {};
    if (not arr)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_source without layer_spread_m", path));
    const auto source = src.value<std::string>();
    if (not source or source->find_first_not_of(" \t") == std::string::npos)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_m needs its layer_spread_source (a non-blank string: "
                                             "where each Δ comes from)", path));
    const toml::array* a = arr.as_array();
    if (a == nullptr) throw std::runtime_error(std::format("{}: [shape].layer_spread_m is not an array", path));
    LayerSpread out;
    out.source = *source;
    for (std::size_t k = 0; k < a->size(); ++k)
    {
        const auto x = (*a)[k].value<double>();
        if (not x or not std::isfinite(*x) or *x < 0.0)
            throw std::runtime_error(std::format("{}: [shape].layer_spread_m element {} is not a finite non-negative number",
                                                 path, k));
        out.delta_m.push_back(static_cast<float>(*x));
    }
    if (static_cast<int>(out.delta_m.size()) != n)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_m has {} elements, the state has {}", path,
                                             out.delta_m.size(), n));
    return out;
}

}  // namespace rc::manifest
