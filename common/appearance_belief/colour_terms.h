// colour_terms.h — the fixed colour palette's terms (names + sRGB prototypes). Eigen-free on purpose, so the
// language agent can read the names without the appearance belief: ONE list, used by colour_name.h to name an
// object's colour and by language_agent to read a colour word in a question ("the magenta chair").
#pragma once

#include <array>
#include <string_view>

namespace rc::appearance
{
struct ColourTerm
{
    std::string_view name;
    float r, g, b;   // sRGB prototype, each in [0,1]
};

/// The fixed palette (owner 2026-11-11: fixed first; person-given names later, on top).
inline constexpr std::array<ColourTerm, 12> kPalette{{
    {"red", 0.80f, 0.10f, 0.10f},    {"orange", 0.95f, 0.55f, 0.10f}, {"yellow", 0.95f, 0.90f, 0.15f},
    {"green", 0.15f, 0.60f, 0.20f},  {"blue", 0.15f, 0.30f, 0.80f},   {"purple", 0.45f, 0.20f, 0.65f},
    {"magenta", 0.90f, 0.25f, 0.90f}, {"pink", 0.98f, 0.70f, 0.78f},  {"brown", 0.45f, 0.28f, 0.15f},
    {"white", 0.95f, 0.95f, 0.95f},  {"grey", 0.55f, 0.55f, 0.55f},   {"black", 0.10f, 0.10f, 0.10f},
}};

/// The term named `w` (exact, lower case), or nullptr.
[[nodiscard]] constexpr const ColourTerm* find_term(std::string_view w)
{
    for (const auto& t : kPalette)
        if (t.name == w) return &t;
    return nullptr;
}
}   // namespace rc::appearance
