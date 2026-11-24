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
 * Textual like shape_parts.h / back_panel.h (no Qt, no DSR); numbers through std::from_chars only (LANG=es_ES.UTF-8 safe).
 * The array may span lines, with a comment after each element (the per-DOF justification).
 */
#pragma once

#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "back_panel.h"

namespace rc::manifest {

struct LayerSpread
{
    std::vector<float> delta_m;   // Δ per DOF; empty ⇔ the manifest declares none
    std::string        source;    // where the values come from (non-empty whenever delta_m is)
};

// Absent [shape].layer_spread_m ⇒ {} (no layering declared). Malformed ⇒ throws std::runtime_error naming the file and the
// fault: unreadable file, an array without its source (or a source without its array), a length other than n, a
// non-numeric, non-finite or negative element.
inline LayerSpread layer_spread(const std::string& path, int n)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [shape].layer_spread_m", path));
    // The value text of a key in [shape]: a string up to its closing quote, an array (possibly over several lines) with
    // each line's comment dropped.
    std::string arr, source, block, line;
    bool has_arr = false, has_source = false, in_arr = false;
    const auto strip_comment = [](std::string_view v)
    {
        const auto c = v.find('#');
        return detail::trim(c == std::string_view::npos ? v : v.substr(0, c));
    };
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (in_arr)
        {
            const std::string_view v = strip_comment(l);
            arr += ' ';
            arr += v;
            if (v.find(']') != std::string_view::npos) in_arr = false;
            continue;
        }
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            const auto e = l.find(']');
            block = std::string(detail::trim(l.substr(1, e == std::string_view::npos ? std::string_view::npos : e - 1)));
            continue;
        }
        if (block != "shape") continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos) continue;
        const auto key = detail::trim(l.substr(0, eq));
        const auto val = detail::trim(l.substr(eq + 1));
        if (key == "layer_spread_m")
        {
            if (has_arr) throw std::runtime_error(std::format("{}: [shape] declares layer_spread_m twice", path));
            has_arr = true;
            arr = std::string(strip_comment(val));
            in_arr = arr.find(']') == std::string::npos;
        }
        else if (key == "layer_spread_source")
        {
            if (has_source) throw std::runtime_error(std::format("{}: [shape] declares layer_spread_source twice", path));
            has_source = true;
            if (val.empty() or val.front() != '"')
                throw std::runtime_error(std::format("{}: [shape].layer_spread_source must be a string", path));
            const auto q = val.find('"', 1);
            source = std::string(val.substr(1, q == std::string_view::npos ? std::string_view::npos : q - 1));
        }
    }
    if (not has_arr and not has_source) return {};
    if (not has_arr)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_source without layer_spread_m", path));
    if (not has_source or detail::trim(source).empty())
        throw std::runtime_error(std::format("{}: [shape].layer_spread_m needs its layer_spread_source (where each Δ comes "
                                             "from)", path));
    const auto open = arr.find('['), close = arr.find(']');
    if (open == std::string::npos or close == std::string::npos or close < open)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_m is not an array ('{}')", path, arr));
    LayerSpread out;
    out.source = source;
    std::string_view body = std::string_view(arr).substr(open + 1, close - open - 1);
    while (not detail::trim(body).empty())
    {
        const auto comma = body.find(',');
        const std::string_view item = detail::trim(body.substr(0, comma));
        body = comma == std::string_view::npos ? std::string_view{} : body.substr(comma + 1);
        if (item.empty() and detail::trim(body).empty()) break;   // a trailing comma
        const auto x = detail::parse_float(item);
        if (not x or not std::isfinite(*x) or *x < 0.0f)
            throw std::runtime_error(std::format("{}: [shape].layer_spread_m element {} is not a non-negative number ('{}')",
                                                 path, out.delta_m.size(), item));
        out.delta_m.push_back(*x);
    }
    if (int(out.delta_m.size()) != n)
        throw std::runtime_error(std::format("{}: [shape].layer_spread_m has {} elements, the state has {}", path,
                                             out.delta_m.size(), n));
    return out;
}

}  // namespace rc::manifest
