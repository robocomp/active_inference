// card.h -- the PERCEPTION CARD: one concept agent's published description of its own detection model.
// Header-only. The card is DATA (which sensors see the concept, at what detectability, which closed-list
// terms apply); the term functions themselves live elsewhere and are looked up by the names below.
//
// JSON: the same flat-JSON facility the media descriptors use (media_transport/flat_json.h), so no second
// parser. A card is nested, so it is written as a FLAT object with dotted keys ("subtype.0.env.min_fill"),
// string lists as a ';'-joined quoted string. Numbers go through to_chars/from_chars only (es_ES safe).
#pragma once

#include "detectability/detectability.h"
#include "media_transport/flat_json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace rc::card
{
namespace flat = rc::media::flat;
}
namespace rc::card
{
enum class Level { Mass, Countable, Articulated, Animate };
enum class Role  { Identity, Occupancy, Both };

struct Channel
{
    std::string sensor;     // graph node name: zed | helios | bpearl | ricoh
    Role        role = Role::Both;
    bool        modelled = false;
};

struct TermSpec
{
    std::string                  name;     // one of kTermNames
    std::map<std::string, float> params;
};

struct SubtypeParams
{
    rc::detect::DetectorEnvelope env;
    float                        q_return = 1.0f;   // per-beam LiDAR return probability
    std::vector<TermSpec>        terms;
};

struct GeometryPrior { float w = 0, d = 0, h = 0, z0 = 0, z1 = 0; };

struct Card
{
    int version = 1;
    std::string concept_, graph_type;     // "concept" is a C++20 keyword
    Level level = Level::Countable;
    std::vector<Channel> channels;
    GeometryPrior geometry;
    std::map<std::string, SubtypeParams> subtypes;   // key "" = default
    std::vector<std::string> instance_terms;
    std::vector<std::string> blind_spots;
    std::string calibration;                          // "prior" | "fitted" | "blind_spot_map"
    std::string calibration_date;
    std::string blind_spot_map_path;
};

// The closed list of term names. Task 2 implements one function per name.
inline constexpr std::array kTermNames{"envelope", "in_fov", "central", "range_conf", "leaf_obliquity",
                                       "resolvability", "kappa", "coverage", "motion"};

inline bool is_term_name(std::string_view n)
{
    return std::ranges::any_of(kTermNames, [&](std::string_view k) { return k == n; });
}

inline const SubtypeParams& params_for(const Card& c, std::string_view subtype)
{
    static const SubtypeParams empty{};
    if (const auto it = c.subtypes.find(std::string(subtype)); it != c.subtypes.end()) return it->second;
    if (const auto it = c.subtypes.find(std::string{}); it != c.subtypes.end()) return it->second;
    return empty;
}

namespace detail
{
inline constexpr std::array<std::string_view, 4> kLevels{"mass", "countable", "articulated", "animate"};
inline constexpr std::array<std::string_view, 3> kRoles{"identity", "occupancy", "both"};

inline std::string q(std::string_view s)    // quoted, escaped for parse_flat_json's \" \\ \n \t
{
    std::string r = "\"";
    for (const char c : s)
    {
        if (c == '"' or c == '\\') { r += '\\'; r += c; }
        else if (c == '\n') r += "\\n";
        else if (c == '\t') r += "\\t";
        else r += c;
    }
    return r + '"';
}
inline std::string join(const std::vector<std::string>& v)
{
    std::string r;
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) r += ';'; r += v[i]; }
    return r;
}
inline std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    if (s.empty()) return out;
    std::size_t i = 0;
    while (true)
    {
        const auto j = s.find(';', i);
        out.push_back(s.substr(i, j == std::string::npos ? j : j - i));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}
} // namespace detail

inline std::string to_json(const Card& c)
{
    using rc::media::flat::fmt_float;
    using detail::q;
    std::string o = "{";
    const auto kv  = [&](const std::string& k, const std::string& raw) { o += q(k) + ':' + raw + ','; };
    const auto kvs = [&](const std::string& k, std::string_view v) { kv(k, q(v)); };
    const auto kvf = [&](const std::string& k, float v) { kv(k, fmt_float(v)); };
    kv("version", std::to_string(c.version));
    kvs("concept", c.concept_);
    kvs("graph_type", c.graph_type);
    kvs("level", detail::kLevels[static_cast<std::size_t>(c.level)]);
    kv("channels", std::to_string(c.channels.size()));
    for (std::size_t i = 0; i < c.channels.size(); ++i)
    {
        const auto p = "channel." + std::to_string(i) + '.';
        kvs(p + "sensor", c.channels[i].sensor);
        kvs(p + "role", detail::kRoles[static_cast<std::size_t>(c.channels[i].role)]);
        kv(p + "modelled", c.channels[i].modelled ? "true" : "false");
    }
    kvf("geom.w", c.geometry.w); kvf("geom.d", c.geometry.d); kvf("geom.h", c.geometry.h);
    kvf("geom.z0", c.geometry.z0); kvf("geom.z1", c.geometry.z1);
    kv("subtypes", std::to_string(c.subtypes.size()));
    std::size_t si = 0;
    for (const auto& [name, sp] : c.subtypes)
    {
        const auto p = "subtype." + std::to_string(si++) + '.';
        kvs(p + "name", name);
        kvf(p + "env.min_fill", sp.env.min_fill);
        kvf(p + "env.max_fill", sp.env.max_fill);
        kvf(p + "env.soft", sp.env.soft);
        kvf(p + "q_return", sp.q_return);
        kv(p + "terms", std::to_string(sp.terms.size()));
        for (std::size_t t = 0; t < sp.terms.size(); ++t)
        {
            const auto tp = p + "term." + std::to_string(t) + '.';
            kvs(tp + "name", sp.terms[t].name);
            for (const auto& [pn, pv] : sp.terms[t].params) kvf(tp + "p." + pn, pv);
        }
    }
    kvs("instance_terms", detail::join(c.instance_terms));
    kvs("blind_spots", detail::join(c.blind_spots));
    kvs("calibration", c.calibration);
    kvs("calibration_date", c.calibration_date);
    kvs("blind_spot_map_path", c.blind_spot_map_path);
    o.back() = '}';
    return o;
}

inline std::expected<Card, std::string> card_from_json(std::string_view json)
{
    using Err = std::unexpected<std::string>;
    const auto kv = flat::parse_flat_json(std::string(json));
    const auto str = [&](const std::string& k) -> const std::string*
    { const auto it = kv.find(k); return it == kv.end() ? nullptr : &it->second; };
    const auto num = [&](const std::string& k) -> std::expected<float, std::string>
    {
        const auto* v = str(k);
        if (not v) return std::unexpected("missing number '" + k + "'");
        const float f = flat::as_float(*v, std::nanf(""));
        if (std::isnan(f)) return std::unexpected("bad number '" + k + "'");
        return f;
    };
    const auto cnt = [&](const std::string& k) -> std::expected<std::size_t, std::string>
    {
        const auto f = num(k);
        if (not f) return std::unexpected(f.error());
        if (*f < 0 or *f > 4096 or *f != std::floor(*f)) return std::unexpected("bad count '" + k + "'");
        return static_cast<std::size_t>(*f);
    };
    const auto text = [&](const std::string& k) { const auto* v = str(k); return v ? *v : std::string{}; };
    const auto index_of = [](const auto& names, const std::string& s) -> int
    {
        for (std::size_t i = 0; i < names.size(); ++i) if (names[i] == s) return static_cast<int>(i);
        return -1;
    };

    Card c;
    if (const auto v = num("version")) c.version = static_cast<int>(*v);
    c.concept_ = text("concept");
    c.graph_type = text("graph_type");
    const auto* lv = str("level");
    if (not lv) return Err("card: missing 'level'");
    const int li = index_of(detail::kLevels, *lv);
    if (li < 0) return Err("card: unknown level '" + *lv + "'");
    c.level = static_cast<Level>(li);

    const auto nch = cnt("channels");
    if (not nch) return Err("card: " + nch.error());
    for (std::size_t i = 0; i < *nch; ++i)
    {
        const auto p = "channel." + std::to_string(i) + '.';
        Channel ch;
        ch.sensor = text(p + "sensor");
        const int ri = index_of(detail::kRoles, text(p + "role"));
        if (ri < 0) return Err("card: unknown role '" + text(p + "role") + "' on channel " + std::to_string(i));
        ch.role = static_cast<Role>(ri);
        ch.modelled = text(p + "modelled") == "true";
        c.channels.push_back(std::move(ch));
    }
    for (const auto& [key, dst] : std::initializer_list<std::pair<const char*, float*>>{
             {"geom.w", &c.geometry.w}, {"geom.d", &c.geometry.d}, {"geom.h", &c.geometry.h},
             {"geom.z0", &c.geometry.z0}, {"geom.z1", &c.geometry.z1}})
    {
        const auto f = num(key);
        if (not f) return Err("card: " + f.error());
        *dst = *f;
    }

    const auto nst = cnt("subtypes");
    if (not nst) return Err("card: " + nst.error());
    for (std::size_t i = 0; i < *nst; ++i)
    {
        const auto p = "subtype." + std::to_string(i) + '.';
        SubtypeParams sp;
        for (const auto& [key, dst] : std::initializer_list<std::pair<const char*, float*>>{
                 {"env.min_fill", &sp.env.min_fill}, {"env.max_fill", &sp.env.max_fill},
                 {"env.soft", &sp.env.soft}, {"q_return", &sp.q_return}})
        {
            const auto f = num(p + key);
            if (not f) return Err("card: " + f.error());
            *dst = *f;
        }
        const auto nt = cnt(p + "terms");
        if (not nt) return Err("card: " + nt.error());
        for (std::size_t t = 0; t < *nt; ++t)
        {
            const auto tp = p + "term." + std::to_string(t) + '.';
            TermSpec ts;
            ts.name = text(tp + "name");
            if (not is_term_name(ts.name)) return Err("card: unknown term '" + ts.name + "'");
            const std::string pp = tp + "p.";
            for (const auto& [k, v] : kv)
                if (k.starts_with(pp))
                {
                    const auto f = num(k);
                    if (not f) return Err("card: " + f.error());
                    ts.params[k.substr(pp.size())] = *f;
                }
            sp.terms.push_back(std::move(ts));
        }
        c.subtypes[text(p + "name")] = std::move(sp);
    }
    c.instance_terms = detail::split(text("instance_terms"));
    for (const auto& t : c.instance_terms)
        if (not is_term_name(t)) return Err("card: unknown term '" + t + "'");
    c.blind_spots = detail::split(text("blind_spots"));
    c.calibration = text("calibration");
    c.calibration_date = text("calibration_date");
    c.blind_spot_map_path = text("blind_spot_map_path");
    return c;
}
} // namespace rc::card
