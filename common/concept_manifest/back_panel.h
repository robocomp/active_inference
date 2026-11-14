/*
 * back_panel.h — the [prior.headboard] block of a <concept>.concept.toml: a FIXED PANEL on the object's back face
 * (a bed's headboard). Absent block ⇒ the class has no panel (nightstand, toilet): a plain box.
 *
 * Its own header, included by concept_manifest.h, because concept_manifest.h drags in genericworker.h (Qt +
 * ConfigLoader) and so cannot be compiled by a standalone test. This one is textual: no Qt, no DSR. Numbers go
 * through std::from_chars — locale-independent by definition (the agents run under LANG=es_ES.UTF-8, where
 * strtof would read "1.25" as 1).
 */
#pragma once

#include <charconv>
#include <fstream>
#include <optional>
#include <print>
#include <string>
#include <string_view>

namespace rc::manifest {

struct BackPanel { float height_m; float thickness_m; };

namespace detail {
inline std::string_view trim(std::string_view s)
{
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
}
inline std::optional<float> parse_float(std::string_view v)
{
    v = trim(v);
    float x = 0.0f;
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), x);
    if (ec != std::errc{} or p != v.data() + v.size()) return std::nullopt;
    return x;
}
}  // namespace detail

// Read [prior.headboard]. nullopt = no panel declared (block absent, quietly) or an INVALID declaration (printed
// loudly: any kind other than "fixed_template", any face other than "back", a missing or non-numeric size).
// `opened` (optional out-param) tells "the manifest could not be opened" (false, and an error naming the path is
// printed) apart from "opened, no panel declared" (true): a missing manifest must not silently become a plain box.
inline std::optional<BackPanel> load_back_panel(const std::string& path, bool* opened = nullptr)
{
    std::ifstream f(path);
    if (opened) *opened = static_cast<bool>(f);
    if (not f)
    {
        std::print("[manifest] ★★CANNOT OPEN {} — the back panel cannot be read (the path is relative to the agent's CWD)\n", path);
        return std::nullopt;
    }
    std::string line, block, kind, face, hs, ts;
    bool seen = false;
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            const auto e = l.find(']');
            block = std::string(detail::trim(l.substr(1, e == std::string_view::npos ? std::string_view::npos : e - 1)));
            if (block == "prior.headboard") seen = true;
            continue;
        }
        if (block != "prior.headboard") continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos) continue;
        const auto key = detail::trim(l.substr(0, eq));
        auto val = detail::trim(l.substr(eq + 1));
        if (not val.empty() and val.front() == '"')                 // a string: up to the closing quote
        {
            const auto q = val.find('"', 1);
            val = val.substr(1, q == std::string_view::npos ? std::string_view::npos : q - 1);
        }
        else if (const auto c = val.find('#'); c != std::string_view::npos) val = detail::trim(val.substr(0, c));
        if (key == "kind") kind = val;
        else if (key == "face") face = val;
        else if (key == "height_m") hs = val;
        else if (key == "thickness_m") ts = val;
    }
    if (not seen) return std::nullopt;
    if (kind != "fixed_template" or face != "back")
    {
        std::print("[manifest] ★{}: [prior.headboard] must be kind=\"fixed_template\" face=\"back\" "
                   "(got kind=\"{}\" face=\"{}\") — no back panel adopted\n", path, kind, face);
        return std::nullopt;
    }
    const auto h = detail::parse_float(hs), t = detail::parse_float(ts);
    if (not h or not t)
    {
        std::print("[manifest] ★{}: [prior.headboard] needs numeric height_m and thickness_m — no back panel adopted\n", path);
        return std::nullopt;
    }
    return BackPanel{*h, *t};
}

}  // namespace rc::manifest
