/*
 * shape_parts.h — the [[shape.part]] tables of a <concept>.concept.toml: the class's solid PARTS as fractions of the
 * fitted box (w, h, H). They feed the 3-D LiDAR existence carve, and, when [shape] solid = "parts", the model's solid too (the
 * camera SDF and the beam parts; common/ai_belief/parts_geometry.h). Absent table ⇒ the whole box is one part.
 * Textual like back_panel.h (no Qt, no DSR); numbers through std::from_chars only (LANG=es_ES.UTF-8 safe).
 * Local axes: x across the width w, +y toward the BACK face.
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

struct ShapePart { std::string name; float x_f, y_f, w_f, h_f, z0_f, z1_f; };   // fractions of the fitted (w, h, H)

// absent table ⇒ {} ; malformed ⇒ throws std::runtime_error naming the file, the part and the key
inline std::vector<ShapePart> load_shape_parts(const std::string& path)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [[shape.part]]", path));
    struct Raw { std::string name; std::string v[6]; bool has[6] = {}; int index = 0; };
    static constexpr std::string_view keys[6] = {"x_f", "y_f", "w_f", "h_f", "z0_f", "z1_f"};
    std::vector<Raw> raws;
    bool in_part = false;
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            in_part = l.starts_with("[[") and detail::trim(l.substr(2, l.find(']') - 2)) == "shape.part";
            if (in_part) { raws.emplace_back(); raws.back().index = static_cast<int>(raws.size()) - 1; }
            continue;
        }
        if (not in_part) continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos) continue;
        const auto key = detail::trim(l.substr(0, eq));
        auto val = detail::trim(l.substr(eq + 1));
        if (not val.empty() and val.front() == '"')
        {
            const auto q = val.find('"', 1);
            val = val.substr(1, q == std::string_view::npos ? std::string_view::npos : q - 1);
        }
        else if (const auto c = val.find('#'); c != std::string_view::npos) val = detail::trim(val.substr(0, c));
        Raw& r = raws.back();
        if (key == "name") { r.name = val; continue; }
        for (int i = 0; i < 6; ++i)
            if (key == keys[i])
            {
                if (r.has[i])
                    throw std::runtime_error(std::format("{}: [[shape.part]] '{}': duplicate key {}",
                                                         path, r.name.empty() ? std::format("#{}", r.index) : r.name, keys[i]));
                r.v[i] = val; r.has[i] = true;
            }
    }
    std::vector<ShapePart> out;
    for (const Raw& r : raws)
    {
        const std::string who = r.name.empty() ? std::format("#{}", r.index) : r.name;
        float x[6];
        for (int i = 0; i < 6; ++i)
        {
            if (not r.has[i])
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': missing key {}", path, who, keys[i]));
            const auto p = detail::parse_float(r.v[i]);
            if (not p or not std::isfinite(*p))
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': key {} is not a number ('{}')", path, who, keys[i], r.v[i]));
            x[i] = *p;
        }
        if (not (x[2] > 0.0f)) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': w_f must be > 0", path, who));
        if (not (x[3] > 0.0f)) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': h_f must be > 0", path, who));
        if (not (x[5] > x[4])) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': z1_f must exceed z0_f", path, who));
        out.push_back({r.name, x[0], x[1], x[2], x[3], x[4], x[5]});
    }
    return out;
}

// [shape] solid: what the class MODEL's solid is (common/ai_belief/parts_geometry.h). "box" (or absent) ⇒ the whole fitted box (the
// [[shape.part]] rows then describe the carve only); "parts" ⇒ the union of the [[shape.part]] boxes, which must then exist.
// Anything else, or "parts" without a [[shape.part]] row, throws std::runtime_error naming the file.
inline bool load_solid_is_parts(const std::string& path)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [shape].solid", path));
    bool in_shape = false, parts = false, seen = false;
    std::string val, line;
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            if (l.starts_with("[[") and detail::trim(l.substr(2, l.find(']') - 2)) == "shape.part") parts = true;
            in_shape = l == "[shape]";
            continue;
        }
        if (not in_shape) continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos or detail::trim(l.substr(0, eq)) != "solid") continue;
        auto v = detail::trim(l.substr(eq + 1));
        if (v.empty() or v.front() != '"' or v.find('"', 1) == std::string_view::npos)
            throw std::runtime_error(std::format("{}: [shape].solid must be a quoted string (\"box\" or \"parts\")", path));
        val = std::string(v.substr(1, v.find('"', 1) - 1));
        seen = true;
    }
    if (not seen or val == "box") return false;
    if (val != "parts") throw std::runtime_error(std::format("{}: [shape].solid = \"{}\": not \"box\" or \"parts\"", path, val));
    if (not parts) throw std::runtime_error(std::format("{}: [shape].solid = \"parts\" without any [[shape.part]] row", path));
    return true;
}

}  // namespace rc::manifest
