// chair_prior.h -- the chair's SIZE PRIOR, read from the concept manifest (common/concept_manifest/chair.concept.toml,
// [prior.template]). The manifest is the single source of these numbers: the birth template, the perception card and
// every reader of the chair's nominal size take them from a ChairPrior, never from etc/config.toml.
// Header-only and framework-free (no Qt / DSR), so the unit tests build it. Numbers are parsed with std::from_chars:
// the agents run under LANG=es_ES.UTF-8, where strtof/atof silently read "0.60" as 0 (see CLAUDE.md).
#pragma once

#include <charconv>
#include <expected>
#include <fstream>
#include <string>
#include <string_view>

namespace rc
{
struct ChairPrior
{
    float seat_w = 0.0f, seat_d = 0.0f, seat_h = 0.0f, back_h = 0.0f, seat_thick = 0.0f;
    float size_std = 0.0f;                                  // std of the size prior (m)
    float top() const { return seat_h + back_h; }           // floor-anchored chair: z in [0, seat_h + back_h]
};

namespace chair_prior_detail
{
inline std::string_view trim(std::string_view s)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t' or s.front() == '\r')) s.remove_prefix(1);
    while (not s.empty() and (s.back() == ' ' or s.back() == '\t' or s.back() == '\r')) s.remove_suffix(1);
    return s;
}
}  // namespace chair_prior_detail

// Reads the six keys of [prior.template]. Any missing or unparsable key is an error naming it.
inline std::expected<ChairPrior, std::string> load_chair_prior(const std::string& manifest_path)
{
    using chair_prior_detail::trim;
    std::ifstream f(manifest_path);
    if (not f) return std::unexpected("cannot open manifest " + manifest_path);

    struct Key { const char* name; float ChairPrior::* dst; bool seen; };
    Key keys[] = {{"seat_w_m", &ChairPrior::seat_w, false},         {"seat_d_m", &ChairPrior::seat_d, false},
                  {"seat_h_m", &ChairPrior::seat_h, false},         {"back_h_m", &ChairPrior::back_h, false},
                  {"seat_thick_m", &ChairPrior::seat_thick, false}, {"size_std_m", &ChairPrior::size_std, false}};
    ChairPrior out;
    bool in_section = false;
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view v = trim(line);
        if (v.empty() or v.front() == '#') continue;
        if (v.front() == '[') { in_section = (trim(v.substr(0, v.find(']') + 1)) == "[prior.template]"); continue; }
        if (not in_section) continue;
        const auto eq = v.find('=');
        if (eq == std::string_view::npos) continue;
        const auto name = trim(v.substr(0, eq));
        auto val = trim(v.substr(eq + 1));
        for (auto& k : keys)
        {
            if (name != k.name) continue;
            float x = 0.0f;
            const auto r = std::from_chars(val.data(), val.data() + val.size(), x);
            if (r.ec != std::errc{})
                return std::unexpected(std::string("[prior.template] ") + k.name + " is not a number in " + manifest_path);
            out.*(k.dst) = x;
            k.seen = true;
        }
    }
    for (const auto& k : keys)
        if (not k.seen) return std::unexpected(std::string("[prior.template] ") + k.name + " missing in " + manifest_path);
    return out;
}
}  // namespace rc
