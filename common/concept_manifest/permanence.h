/*
 * permanence.h — the [prior.permanence] block of a <concept>.concept.toml (plan rev 2 F1, round 6; owner ruling 11-25).
 *
 *   [prior.permanence]
 *   move_rate_per_day = 0.0027   # how often a real object of the class is moved (moves per day)
 *   move_scale_m      = 0.5      # the typical displacement of such a move (m, on the footprint centre)
 *   move_scale_rad    = 0.5      # its typical rotation (rad)
 *   source            = "…"      # MANDATORY: where the numbers come from, or "uncited convention: …"
 *
 * CLASS knowledge, never a Webots world. It is the volatility's hyperprior (a spike-and-slab process noise): Q is inferred
 * around q₀ = rate·dt·scale². toml++ parses numbers locale-independently.
 */
#pragma once

#include <cmath>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

#include "manifest_toml.h"

namespace rc::manifest {

struct Permanence
{
    float move_rate_per_day = 0.0f, move_scale_m = 0.0f, move_scale_rad = 0.0f;
    std::string source;
};

// Absent block ⇒ nullopt. Malformed ⇒ throws std::runtime_error naming the file and the fault (a missing key, a blank source, a
// non-finite or negative number).
inline std::optional<Permanence> permanence(const std::string& path)
{
    const toml::table m = parse_manifest(path);
    if (not m.at_path("prior.permanence")) return std::nullopt;
    const auto num = [&](const char* k) -> float
    {
        const auto v = m.at_path(std::string("prior.permanence.") + k).value<double>();
        if (not v or not std::isfinite(*v) or *v < 0.0)
            throw std::runtime_error(std::format("{}: [prior.permanence].{} is missing or not a finite non-negative number", path, k));
        return static_cast<float>(*v);
    };
    Permanence p{num("move_rate_per_day"), num("move_scale_m"), num("move_scale_rad"), {}};
    const auto src = m.at_path("prior.permanence.source").value<std::string>();
    if (not src or src->find_first_not_of(" \t") == std::string::npos)
        throw std::runtime_error(std::format("{}: [prior.permanence] needs its source (a non-blank string, or \"uncited convention: …\")", path));
    p.source = *src;
    return p;
}

}  // namespace rc::manifest
