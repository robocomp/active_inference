/*
 * wall_standoff.h — the [prior.attachment] stand-off of a <concept>.concept.toml (plan rev 2 F1.4).
 *
 *   [prior.attachment]
 *   standoff_m      = 0.03    # how far the back face of a real object of the class typically stands off its wall (m)
 *   standoff_std_m  = 0.02    # its spread over real placements (m)
 *   standoff_source = "…"     # MANDATORY with the values: the placement convention they come from, or "uncited convention: …"
 *
 * CLASS knowledge from typical real placement, never from a Webots world. The persistent path enters it ONCE, as the wall
 * latent's flush prior: gap − δ ~ N(standoff_m, standoff_std_m²).
 * Read through rc::manifest::parse_manifest (toml++: locale-independent numbers).
 */
#pragma once

#include <cmath>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

#include "manifest_toml.h"

namespace rc::manifest {

struct WallStandoff
{
    float mean_m = 0.0f, std_m = 0.0f;
    std::string source;
};

// Absent (none of the three keys) ⇒ nullopt. Malformed ⇒ throws std::runtime_error naming the file and the fault: a value
// without the other or without its source, a blank source, a non-finite or negative mean, a std that is not > 0.
inline std::optional<WallStandoff> wall_standoff(const std::string& path)
{
    const toml::table m = parse_manifest(path);
    const auto mean = m.at_path("prior.attachment.standoff_m").value<double>();
    const auto sd   = m.at_path("prior.attachment.standoff_std_m").value<double>();
    const auto src  = m.at_path("prior.attachment.standoff_source").value<std::string>();
    if (not mean and not sd and not src) return std::nullopt;
    if (not mean or not sd)
        throw std::runtime_error(std::format("{}: [prior.attachment] needs both standoff_m and standoff_std_m", path));
    if (not src or src->find_first_not_of(" \t") == std::string::npos)
        throw std::runtime_error(std::format("{}: [prior.attachment].standoff_m needs its standoff_source (a non-blank string: "
                                             "the placement convention, or \"uncited convention: …\")", path));
    if (not std::isfinite(*mean) or *mean < 0.0)
        throw std::runtime_error(std::format("{}: [prior.attachment].standoff_m is not a finite non-negative number", path));
    if (not std::isfinite(*sd) or not (*sd > 0.0))
        throw std::runtime_error(std::format("{}: [prior.attachment].standoff_std_m is not a finite positive number", path));
    return WallStandoff{static_cast<float>(*mean), static_cast<float>(*sd), *src};
}

// [prior.placement] (chair): the placement MIXTURE's weight of "against a wall", p_against_wall / (p_free + p_against_wall) — the
// two weights are normalised by the reader, as the manifest says. Absent (neither key) ⇒ nullopt: a class with a flush prior or
// a stand-off and no placement mixture stands against its wall with p = 1 implied. Malformed (one key without the other, a
// non-finite or negative weight, both zero) ⇒ throws std::runtime_error naming the file.
inline std::optional<float> placement_p_against_wall(const std::string& path)
{
    const toml::table m = parse_manifest(path);
    const auto pf = m.at_path("prior.placement.p_free").value<double>();
    const auto pw = m.at_path("prior.placement.p_against_wall").value<double>();
    if (not pf and not pw) return std::nullopt;
    if (not pf or not pw)
        throw std::runtime_error(std::format("{}: [prior.placement] needs both p_free and p_against_wall", path));
    if (not std::isfinite(*pf) or not std::isfinite(*pw) or *pf < 0.0 or *pw < 0.0 or not (*pf + *pw > 0.0))
        throw std::runtime_error(std::format("{}: [prior.placement] p_free / p_against_wall must be finite, non-negative, not both 0", path));
    return static_cast<float>(*pw / (*pf + *pw));
}

}  // namespace rc::manifest
