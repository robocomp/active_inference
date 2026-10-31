// chair_prior.h -- the chair's SIZE PRIOR, read from the concept manifest (common/concept_manifest/chair.concept.toml,
// [prior.template]). The manifest is the single source of these numbers: the birth template, the perception card and
// every reader of the chair's nominal size take them from a ChairPrior, never from etc/config.toml.
// Header-only and framework-free (no Qt / DSR), so the unit tests build it. Numbers are parsed with std::from_chars:
// the agents run under LANG=es_ES.UTF-8, where strtof/atof silently read "0.60" as 0 (see CLAUDE.md).
#pragma once

#include <charconv>
#include <cmath>
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
    // [prior.placement]: where a chair stands. Weights of the two contexts (unnormalised) and the gap to the wall
    // when it stands against one.
    float p_free = 0.0f, p_against_wall = 0.0f, wall_gap_mean = 0.0f, wall_gap_sigma = 0.0f;
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

// Reads the six keys of [prior.template] and the four of [prior.placement]. STRICT, because this parser is the startup gate for the size prior:
//  - the whole numeric token must be consumed (only whitespace or a `#` comment may follow);
//  - every value must be finite and > 0 (a size of 0, a NaN or an Inf is a mistake, not a prior);
//  - a duplicate key in the section is an error;
//  - the contents of `"""` multi-line strings (the `why` prose) are skipped, so a line in prose can neither be
//    read as a key nor open/close a section.
// Hand-written scanner instead of the manifest's ConfigLoader: ConfigLoader comes with genericworker.h (Qt / DSR),
// which the framework-free unit tests cannot link, and it would also hide the strictness rules above.
inline std::expected<ChairPrior, std::string> load_chair_prior(const std::string& manifest_path)
{
    using chair_prior_detail::trim;
    std::ifstream f(manifest_path);
    if (not f) return std::unexpected("cannot open manifest " + manifest_path);

    struct Key { const char* section; const char* name; float ChairPrior::* dst; bool seen; };
    Key keys[] = {{"[prior.template]", "seat_w_m", &ChairPrior::seat_w, false},
                  {"[prior.template]", "seat_d_m", &ChairPrior::seat_d, false},
                  {"[prior.template]", "seat_h_m", &ChairPrior::seat_h, false},
                  {"[prior.template]", "back_h_m", &ChairPrior::back_h, false},
                  {"[prior.template]", "seat_thick_m", &ChairPrior::seat_thick, false},
                  {"[prior.template]", "size_std_m", &ChairPrior::size_std, false},
                  {"[prior.placement]", "p_free", &ChairPrior::p_free, false},
                  {"[prior.placement]", "p_against_wall", &ChairPrior::p_against_wall, false},
                  {"[prior.placement]", "wall_gap_mean_m", &ChairPrior::wall_gap_mean, false},
                  {"[prior.placement]", "wall_gap_sigma_m", &ChairPrior::wall_gap_sigma, false}};
    ChairPrior out;
    std::string section;                                    // the header of the section the scanner is in
    bool in_string = false;
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view v = trim(line);
        // a line holding an odd number of `"""` toggles the multi-line string; text inside it is prose
        std::size_t quotes = 0;
        for (auto at = v.find("\"\"\""); at != std::string_view::npos; at = v.find("\"\"\"", at + 3)) ++quotes;
        const bool was_in_string = in_string;
        if (quotes % 2 == 1) in_string = not in_string;
        if (was_in_string) continue;
        if (v.empty() or v.front() == '#') continue;
        if (v.front() == '[') { section = std::string(trim(v.substr(0, v.find(']') + 1))); continue; }
        const auto eq = v.find('=');
        if (eq == std::string_view::npos) continue;
        const auto name = trim(v.substr(0, eq));
        const auto val = trim(v.substr(eq + 1));
        for (auto& k : keys)
        {
            if (section != k.section or name != k.name) continue;
            const std::string where = std::string(k.section) + ' ' + k.name + " in " + manifest_path;
            if (k.seen) return std::unexpected("duplicate " + where);
            float x = 0.0f;
            const auto r = std::from_chars(val.data(), val.data() + val.size(), x);
            if (r.ec != std::errc{}) return std::unexpected(where + " is not a number");
            const auto rest = trim(std::string_view(r.ptr, val.data() + val.size() - r.ptr));
            if (not rest.empty() and rest.front() != '#') return std::unexpected(where + " has trailing garbage");
            if (not std::isfinite(x) or not (x > 0.0f)) return std::unexpected(where + " must be finite and > 0");
            out.*(k.dst) = x;
            k.seen = true;
        }
    }
    for (const auto& k : keys)
        if (not k.seen) return std::unexpected(std::string(k.section) + ' ' + k.name + " missing in " + manifest_path);
    return out;
}
}  // namespace rc
