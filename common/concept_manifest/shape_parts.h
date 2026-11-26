/*
 * shape_parts.h — the [[shape.part]] tables of a <concept>.concept.toml: the class's solid PARTS as fractions of the
 * fitted box (w, h, H). They feed the 3-D LiDAR existence carve, and, when [shape] solid = "parts", the model's solid too (the
 * camera SDF and the beam parts; common/ai_belief/parts_geometry.h). Absent table ⇒ the whole box is one part.
 * Textual like back_panel.h (no Qt, no DSR); numbers through std::from_chars only (LANG=es_ES.UTF-8 safe).
 * Local axes: x across the width w, +y toward the BACK face.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "back_panel.h"

namespace rc::manifest {

// Fractions of the fitted (w, h, H). z0_ref / z1_ref ≥ 0: that boundary is the class PROPORTION of that index (K2a-L,
// [[shape.proportion]] below) plus the fixed fraction z0_off / z1_off (K2b, the manifest's z0_off_f / z1_off_f: a chair's seat
// underside is a slab thickness below its seat top), and z0_f / z1_f hold the proportion's midpoint + that offset; −1: a fixed
// fraction (offset 0).
struct ShapePart { std::string name; float x_f, y_f, w_f, h_f, z0_f, z1_f; int z0_ref = -1, z1_ref = -1; float z0_off = 0.0f, z1_off = 0.0f; };

// A [[shape.proportion]]: a fraction whose real value varies across the class, over [lo, hi] (class knowledge, `source`
// mandatory). The model carries it as a ρ = 1 latent, mean (lo + hi)/2, variance (hi − lo)²/12 (spec G4). K2a-L.
struct ShapeProportion { std::string name; float lo = 0.0f, hi = 0.0f; std::string source; };

// absent ⇒ {}; malformed (missing / non-numeric / non-finite lo or hi, lo ≥ hi, outside [0, 1], blank source, a duplicate name)
// ⇒ throws std::runtime_error naming the file and the proportion
inline std::vector<ShapeProportion> load_shape_proportions(const std::string& path)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [[shape.proportion]]", path));
    struct Raw { std::string name, lo, hi, source; bool has_lo = false, has_hi = false; };
    std::vector<Raw> raws;
    bool in = false;
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            in = l.starts_with("[[") and detail::trim(l.substr(2, l.find(']') - 2)) == "shape.proportion";
            if (in) raws.emplace_back();
            continue;
        }
        if (not in) continue;
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
        if (key == "name") r.name = val;
        else if (key == "lo") { r.lo = val; r.has_lo = true; }
        else if (key == "hi") { r.hi = val; r.has_hi = true; }
        else if (key == "source") r.source = val;
    }
    std::vector<ShapeProportion> out;
    for (std::size_t i = 0; i < raws.size(); ++i)
    {
        const Raw& r = raws[i];
        const std::string who = r.name.empty() ? std::format("#{}", i) : r.name;
        if (r.name.empty()) throw std::runtime_error(std::format("{}: [[shape.proportion]] {} has no name", path, who));
        const auto lo = r.has_lo ? detail::parse_float(r.lo) : std::nullopt, hi = r.has_hi ? detail::parse_float(r.hi) : std::nullopt;
        if (not lo or not hi or not std::isfinite(*lo) or not std::isfinite(*hi))
            throw std::runtime_error(std::format("{}: [[shape.proportion]] '{}': lo and hi must be numbers", path, who));
        if (not (*lo < *hi) or *lo < 0.0f or *hi > 1.0f)
            throw std::runtime_error(std::format("{}: [[shape.proportion]] '{}': needs 0 <= lo < hi <= 1 (a fraction of the box)", path, who));
        if (detail::trim(r.source).empty())
            throw std::runtime_error(std::format("{}: [[shape.proportion]] '{}': a source is mandatory (class knowledge)", path, who));
        for (const auto& o : out)
            if (o.name == r.name) throw std::runtime_error(std::format("{}: [[shape.proportion]] '{}' declared twice", path, who));
        out.push_back({r.name, *lo, *hi, r.source});
    }
    return out;
}

// absent table ⇒ {} ; malformed ⇒ throws std::runtime_error naming the file, the part and the key
inline std::vector<ShapePart> load_shape_parts(const std::string& path)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [[shape.part]]", path));
    const std::vector<ShapeProportion> props = load_shape_proportions(path);
    struct Raw { std::string name; std::string v[6], off[2]; bool has[6] = {}, quoted[6] = {}, has_off[2] = {}; int index = 0; };
    static constexpr std::string_view keys[6] = {"x_f", "y_f", "w_f", "h_f", "z0_f", "z1_f"};
    static constexpr std::string_view off_keys[2] = {"z0_off_f", "z1_off_f"};
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
        const bool quoted = not val.empty() and val.front() == '"';
        if (quoted)
        {
            const auto q = val.find('"', 1);
            val = val.substr(1, q == std::string_view::npos ? std::string_view::npos : q - 1);
        }
        else if (const auto c = val.find('#'); c != std::string_view::npos) val = detail::trim(val.substr(0, c));
        Raw& r = raws.back();
        if (key == "name") { r.name = val; continue; }
        for (int j = 0; j < 2; ++j)
            if (key == off_keys[j])
            {
                if (r.has_off[j])
                    throw std::runtime_error(std::format("{}: [[shape.part]] '{}': duplicate key {}",
                                                         path, r.name.empty() ? std::format("#{}", r.index) : r.name, off_keys[j]));
                r.off[j] = val; r.has_off[j] = true;
            }
        for (int i = 0; i < 6; ++i)
            if (key == keys[i])
            {
                if (r.has[i])
                    throw std::runtime_error(std::format("{}: [[shape.part]] '{}': duplicate key {}",
                                                         path, r.name.empty() ? std::format("#{}", r.index) : r.name, keys[i]));
                r.v[i] = val; r.has[i] = true; r.quoted[i] = quoted;
            }
    }
    std::vector<ShapePart> out;
    for (const Raw& r : raws)
    {
        const std::string who = r.name.empty() ? std::format("#{}", r.index) : r.name;
        float x[6];
        int ref[6] = {-1, -1, -1, -1, -1, -1};
        float lo[6], hi[6], off[2] = {0.0f, 0.0f};
        for (int j = 0; j < 2; ++j)
        {   // an offset from a proportion (K2b): a number, and only on a boundary that names one
            if (not r.has_off[j]) continue;
            if (not r.quoted[4 + j])
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': {} needs {} to name a proportion", path, who, off_keys[j], keys[4 + j]));
            const auto p = detail::parse_float(r.off[j]);
            if (not p or not std::isfinite(*p))
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': key {} is not a number ('{}')", path, who, off_keys[j], r.off[j]));
            off[j] = *p;
        }
        for (int i = 0; i < 6; ++i)
        {
            if (not r.has[i])
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': missing key {}", path, who, keys[i]));
            if (r.quoted[i])
            {   // a quoted value names a [[shape.proportion]]: only the vertical boundaries (z0_f, z1_f) may
                if (i < 4) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': {} cannot name a proportion (only z0_f, z1_f)", path, who, keys[i]));
                const auto it = std::ranges::find(props, r.v[i], &ShapeProportion::name);
                if (it == props.end())
                    throw std::runtime_error(std::format("{}: [[shape.part]] '{}': {} names an undeclared proportion '{}'", path, who, keys[i], r.v[i]));
                ref[i] = int(it - props.begin());
                const float o = off[i - 4];
                x[i] = 0.5f * (it->lo + it->hi) + o; lo[i] = it->lo + o; hi[i] = it->hi + o;
                continue;
            }
            const auto p = detail::parse_float(r.v[i]);
            if (not p or not std::isfinite(*p))
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': key {} is not a number ('{}')", path, who, keys[i], r.v[i]));
            x[i] = *p; lo[i] = hi[i] = *p;
        }
        if (not (x[2] > 0.0f)) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': w_f must be > 0", path, who));
        if (not (x[3] > 0.0f)) throw std::runtime_error(std::format("{}: [[shape.part]] '{}': h_f must be > 0", path, who));
        // over the whole range of a proportion: z1 above z0 for every value it may take. Both on the SAME proportion, they move
        // together, so the offsets alone decide (K2b: a slab of fixed thickness under the seat top)
        const bool same = ref[4] >= 0 and ref[4] == ref[5];
        if (not (same ? off[1] > off[0] : lo[5] > hi[4]))
            throw std::runtime_error(std::format("{}: [[shape.part]] '{}': z1_f must exceed z0_f", path, who));
        // a boundary shifted off a proportion stays a fraction of the box over the proportion's whole range
        for (int i = 4; i < 6; ++i)
            if (ref[i] >= 0 and (lo[i] < 0.0f or hi[i] > 1.0f))
                throw std::runtime_error(std::format("{}: [[shape.part]] '{}': {} + {} leaves [0, 1] over the proportion's range", path, who,
                                                     keys[i], off_keys[i - 4]));
        out.push_back({r.name, x[0], x[1], x[2], x[3], x[4], x[5], ref[4], ref[5], off[0], off[1]});
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
            const auto c = l.find('#');   // a header may carry a trailing comment: "[shape]   # ..."
            in_shape = detail::trim(c == std::string_view::npos ? l : l.substr(0, c)) == "[shape]";
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

// [shape] symmetry (K2c): "none" (or absent) — the class has a front, θ's (w, h, yaw) names one body; "c2v" — the class has no
// front and no labelled sides (a table), so (w, h, yaw), (w, h, yaw + π) and (h, w, yaw ± π/2) are the SAME body. The group acts
// on the part fractions as r_π: (x, y, w, h) → (−x, −y, w, h) and swap∘r_π/2: (x, y, w, h) → (y, −x, h, w) (z unchanged); the
// solid is invariant iff each generator maps every part onto a part of the set (exact equality of the declared numbers: a
// declared −0.441 parses to the exact negation of 0.441).
enum class Symmetry { none, c2v };

inline bool parts_invariant_c2v(std::span<const ShapePart> parts)
{
    const auto has = [&](const ShapePart& img)
    {
        return std::ranges::any_of(parts, [&](const ShapePart& p)
        {
            return p.x_f == img.x_f and p.y_f == img.y_f and p.w_f == img.w_f and p.h_f == img.h_f and p.z0_f == img.z0_f
                   and p.z1_f == img.z1_f and p.z0_ref == img.z0_ref and p.z1_ref == img.z1_ref and p.z0_off == img.z0_off
                   and p.z1_off == img.z1_off;
        });
    };
    return std::ranges::all_of(parts, [&](const ShapePart& p)
    {
        ShapePart r = p;  r.x_f = -p.x_f; r.y_f = -p.y_f;                                   // r_π
        ShapePart s = p;  s.x_f = p.y_f;  s.y_f = -p.x_f; s.w_f = p.h_f; s.h_f = p.w_f;    // swap∘r_π/2
        return has(r) and has(s);
    });
}

// absent / "none" ⇒ none; "c2v" ⇒ c2v, and the [[shape.part]] rows must then be invariant (above). Anything else, or a c2v class
// whose parts are not invariant, throws std::runtime_error naming the file.
inline Symmetry load_shape_symmetry(const std::string& path)
{
    std::ifstream f(path);
    if (not f) throw std::runtime_error(std::format("{}: cannot open the manifest to read [shape].symmetry", path));
    bool in_shape = false, seen = false;
    std::string val, line;
    while (std::getline(f, line))
    {
        std::string_view l = detail::trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[')
        {
            const auto c = l.find('#');
            in_shape = detail::trim(c == std::string_view::npos ? l : l.substr(0, c)) == "[shape]";
            continue;
        }
        if (not in_shape) continue;
        const auto eq = l.find('=');
        if (eq == std::string_view::npos or detail::trim(l.substr(0, eq)) != "symmetry") continue;
        auto v = detail::trim(l.substr(eq + 1));
        if (v.empty() or v.front() != '"' or v.find('"', 1) == std::string_view::npos)
            throw std::runtime_error(std::format("{}: [shape].symmetry must be a quoted string (\"none\" or \"c2v\")", path));
        val = std::string(v.substr(1, v.find('"', 1) - 1));
        seen = true;
    }
    if (not seen or val == "none") return Symmetry::none;
    if (val != "c2v") throw std::runtime_error(std::format("{}: [shape].symmetry = \"{}\": not \"none\" or \"c2v\"", path, val));
    if (const auto parts = load_shape_parts(path); not parts_invariant_c2v(parts))
        throw std::runtime_error(std::format("{}: [shape].symmetry = \"c2v\" but the [[shape.part]] rows are not invariant under "
                                             "(x,y,w,h) -> (-x,-y,w,h) and (y,-x,h,w)", path));
    return Symmetry::c2v;
}

}  // namespace rc::manifest
