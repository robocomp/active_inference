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
#include "perception_card/term_spec.h"   // TermSpec (split out so terms.h stays free of the JSON / media_transport includes)

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <map>
#include <optional>
#include <span>
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

struct SubtypeParams
{
    rc::detect::DetectorEnvelope env;
    float                        q_return = 1.0f;   // per-beam LiDAR return probability
    std::vector<TermSpec>        terms;
};

struct GeometryPrior { float w = 0, d = 0, h = 0, z0 = 0, z1 = 0; };

// Level 1 (spec §3): how the agent's own line-of-sight test treats the ROOM WALLS, as the agent actually runs it.
//   own_wall_skip_m   a wall segment closer than this to the object never occludes it (rc::occlusion::walls_block's
//                     radius form: the object's OWN wall). 0 = no radius skip.
//   own_wall_plane_m  >= 0: the own wall is given by PLANE instead (walls_block's explicit-edge form, door_concept): a
//                     polygon edge whose two ends both lie within +-own_wall_plane_m of the object's face plane (the
//                     plane through its centre, normal = its local y axis) is its own wall; every other wall occludes
//                     however close it passes. < 0 = not used.
//   wall_thickness_m  the depth of the wall boxes the agent tests against. 0 = a LINE (walls_block tests polygon
//                     edges as lines); the evaluator represents a line as a 1 mm box, as door_look.h does.
struct WallOcclusion
{
    float own_wall_skip_m  = 0.0f;
    float own_wall_plane_m = -1.0f;
    float wall_thickness_m = 0.0f;
};

// ── Generative block (imagined observations): what the concept looks like, as DATA ────────────────────────────
// A shape program of CLOSED primitives, a distribution per state variable, and a placement prior. All lengths in m,
// all in the object's LOCAL frame (+y forward, as ChairState), a floor-anchored object's z = 0 at the floor.
// Conventions (so a sampler and a renderer read the same thing):
//   Slab, Box, Panel : x,y = CENTRE of the footprint, z = BASE (lowest z); w along x, d (Panel: t) along y, h along z.
//   Cylinder         : x,y = centre of the axis, z = base; r radius, h height.
//   LegSet           : n legs of square half-side `half`, one at each corner of the fw x fd footprint, inset so the
//                      leg's outer face is flush with the footprint (centre at +-(fw/2 - half), +-(fd/2 - half));
//                      z0..z1 is the leg's z span.
// A dimension is a Bind: a*state[var] + b (var empty: the constant b).
enum class Prim { Slab, Panel, LegSet, Box, Cylinder };

struct Bind { std::string var; float a = 0.0f, b = 0.0f; };

struct Primitive
{
    Prim type = Prim::Slab;
    std::map<std::string, Bind> dims;     // keys from prim_dim_names(type), all required
};

struct VarDist { std::string var; float mean = 0, sigma = 0, lo = 0, hi = 0; bool log_normal = false; };

struct PlacementPrior
{
    std::string support;                          // "floor"
    std::map<std::string, float> contexts;        // keys "free" | "against_wall" -> weight
    // Gap to the wall when against_wall (m): the distance from the BACKREST's OUTER face (the object's local -y edge) to the
    // wall line (the polygon edge), drawn from this normal truncated at 0. The backrest faces the wall.
    float wall_gap_mean = 0.0f, wall_gap_sigma = 0.0f;
};

struct Generative
{
    std::vector<Primitive> shape;
    std::vector<VarDist> vars;
    PlacementPrior placement;
    std::string bounds_source;                    // where lo/hi come from (a human-readable provenance)
};

struct Card
{
    int version = 1;
    std::string concept_, graph_type;     // "concept" is a C++20 keyword
    Level level = Level::Countable;
    std::vector<Channel> channels;
    GeometryPrior geometry;
    std::optional<WallOcclusion> walls;               // level 1; nullopt = the card does not declare it (old card)
    std::map<std::string, SubtypeParams> subtypes;   // key "" = default
    std::vector<std::string> instance_terms;          // names from kInstanceTermNames (per-INSTANCE state, not hypothesis terms)
    std::map<std::string, float> instance_prior;       // value a hypothesis (no instance history) takes for an instance term: "p_clear"
    std::vector<std::string> blind_spots;
    std::string calibration;                          // "prior" | "fitted" | "blind_spot_map"
    std::string calibration_date;
    std::string blind_spot_map_path;
    std::optional<Generative> generative;             // nullopt = the card does not describe its own appearance
};

// The closed list of term names. Task 2 implements one function per name.
inline constexpr std::array kTermNames{"envelope", "in_fov", "central", "range_conf", "leaf_obliquity",
                                       "resolvability", "kappa", "coverage", "motion"};

// Instance terms are NOT hypothesis terms: they describe per-instance state (a score history, a learned map).
// Their own closed list; the evaluator marginalises each at its prior for a hypothetical object.
inline constexpr std::array kInstanceTermNames{"p_clear", "blind_spot_map"};

inline bool is_instance_term_name(std::string_view n)
{
    return std::ranges::any_of(kInstanceTermNames, [&](std::string_view k) { return k == n; });
}

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

// Closed vocabularies of the generative block.
inline constexpr std::array<std::string_view, 5> kPrimNames{"slab", "panel", "legset", "box", "cylinder"};
inline constexpr std::array<std::string_view, 2> kContextNames{"free", "against_wall"};
inline constexpr std::array<std::string_view, 1> kSupportNames{"floor"};

inline std::span<const std::string_view> prim_dim_names(Prim p)
{
    static constexpr std::array<std::string_view, 6> slab{"w", "d", "h", "x", "y", "z"};
    static constexpr std::array<std::string_view, 6> panel{"w", "t", "h", "x", "y", "z"};
    static constexpr std::array<std::string_view, 6> legs{"n", "half", "fw", "fd", "z0", "z1"};
    static constexpr std::array<std::string_view, 5> cyl{"r", "h", "x", "y", "z"};
    switch (p)
    {
        case Prim::Slab: case Prim::Box: return slab;
        case Prim::Panel: return panel;
        case Prim::LegSet: return legs;
        case Prim::Cylinder: return cyl;
    }
    return {};
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
    if (c.walls)
    {
        kvf("wall.own_skip_m", c.walls->own_wall_skip_m);
        kvf("wall.own_plane_m", c.walls->own_wall_plane_m);
        kvf("wall.thickness_m", c.walls->wall_thickness_m);
    }
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
    for (const auto& [k, v] : c.instance_prior) kvf("instance_prior." + k, v);
    kvs("instance_terms", detail::join(c.instance_terms));
    kvs("blind_spots", detail::join(c.blind_spots));
    kvs("calibration", c.calibration);
    kvs("calibration_date", c.calibration_date);
    kvs("blind_spot_map_path", c.blind_spot_map_path);
    if (c.generative)
    {
        const Generative& g = *c.generative;
        kv("gen.shapes", std::to_string(g.shape.size()));
        for (std::size_t i = 0; i < g.shape.size(); ++i)
        {
            const auto p = "gen.shape." + std::to_string(i) + '.';
            kvs(p + "type", kPrimNames[static_cast<std::size_t>(g.shape[i].type)]);
            for (const auto& [dn, b] : g.shape[i].dims)
            {
                kvs(p + "dim." + dn + ".var", b.var);
                kvf(p + "dim." + dn + ".a", b.a);
                kvf(p + "dim." + dn + ".b", b.b);
            }
        }
        kv("gen.vars", std::to_string(g.vars.size()));
        for (std::size_t i = 0; i < g.vars.size(); ++i)
        {
            const auto p = "gen.var." + std::to_string(i) + '.';
            kvs(p + "name", g.vars[i].var);
            kvf(p + "mean", g.vars[i].mean); kvf(p + "sigma", g.vars[i].sigma);
            kvf(p + "lo", g.vars[i].lo);     kvf(p + "hi", g.vars[i].hi);
            kv(p + "log_normal", g.vars[i].log_normal ? "true" : "false");
        }
        kvs("gen.placement.support", g.placement.support);
        for (const auto& [k, w] : g.placement.contexts) kvf("gen.placement.ctx." + k, w);
        kvf("gen.placement.wall_gap_mean", g.placement.wall_gap_mean);
        kvf("gen.placement.wall_gap_sigma", g.placement.wall_gap_sigma);
        kvs("gen.bounds_source", g.bounds_source);
    }
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

    // Level 1: optional as a block (a card written before it existed still loads, as "not declared"); once
    // "wall.own_skip_m" is present the other two must be too.
    if (str("wall.own_skip_m"))
    {
        WallOcclusion wo;
        for (const auto& [key, dst] : std::initializer_list<std::pair<const char*, float*>>{
                 {"wall.own_skip_m", &wo.own_wall_skip_m}, {"wall.own_plane_m", &wo.own_wall_plane_m},
                 {"wall.thickness_m", &wo.wall_thickness_m}})
        {
            const auto f = num(key);
            if (not f) return Err("card: " + f.error());
            *dst = *f;
        }
        c.walls = wo;
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
        if (not is_instance_term_name(t)) return Err("card: unknown instance term '" + t + "'");
    for (const auto& [k, v] : kv)
        if (k.starts_with("instance_prior."))
        {
            const auto f = num(k);
            if (not f) return Err("card: " + f.error());
            c.instance_prior[k.substr(15)] = *f;
        }
    c.blind_spots = detail::split(text("blind_spots"));
    c.calibration = text("calibration");
    c.calibration_date = text("calibration_date");
    c.blind_spot_map_path = text("blind_spot_map_path");

    // Generative block: optional as a whole (a card written before it existed still loads), strict once "gen.shapes"
    // is present: closed primitive / dim / context / support names, every dim of a primitive declared, every Bind
    // naming a declared variable.
    if (str("gen.shapes"))
    {
        Generative g;
        const auto ns = cnt("gen.shapes");
        if (not ns) return Err("card: " + ns.error());
        const auto nv = cnt("gen.vars");
        if (not nv) return Err("card: " + nv.error());
        for (std::size_t i = 0; i < *nv; ++i)
        {
            const auto p = "gen.var." + std::to_string(i) + '.';
            VarDist v;
            v.var = text(p + "name");
            if (v.var.empty()) return Err("card: gen variable " + std::to_string(i) + " has no name");
            for (const auto& [key, dst] : std::initializer_list<std::pair<const char*, float*>>{
                     {"mean", &v.mean}, {"sigma", &v.sigma}, {"lo", &v.lo}, {"hi", &v.hi}})
            {
                const auto f = num(p + key);
                if (not f) return Err("card: " + f.error());
                *dst = *f;
            }
            v.log_normal = text(p + "log_normal") == "true";
            g.vars.push_back(std::move(v));
        }
        const auto declared = [&](const std::string& n)
        { return std::ranges::any_of(g.vars, [&](const VarDist& v) { return v.var == n; }); };
        for (std::size_t i = 0; i < *ns; ++i)
        {
            const auto p = "gen.shape." + std::to_string(i) + '.';
            Primitive pr;
            const int pi = index_of(kPrimNames, text(p + "type"));
            if (pi < 0) return Err("card: unknown primitive '" + text(p + "type") + "'");
            pr.type = static_cast<Prim>(pi);
            const auto names = prim_dim_names(pr.type);
            const std::string dp = p + "dim.";
            for (const auto& [k, v] : kv)     // reject any dim key outside the closed list for this type
                if (k.starts_with(dp))
                {
                    const std::string dn = k.substr(dp.size(), k.find('.', dp.size()) - dp.size());
                    if (not std::ranges::any_of(names, [&](std::string_view n) { return n == dn; }))
                        return Err("card: unknown dim '" + dn + "' on primitive " + std::to_string(i));
                }
            for (const auto dn : names)
            {
                const std::string kp = dp + std::string(dn) + '.';
                if (not str(kp + "var")) return Err("card: primitive " + std::to_string(i) + " lacks dim '" + std::string(dn) + "'");
                Bind b;
                b.var = text(kp + "var");
                if (not b.var.empty() and not declared(b.var)) return Err("card: dim '" + std::string(dn) + "' binds undeclared variable '" + b.var + "'");
                const auto fa = num(kp + "a"), fb = num(kp + "b");
                if (not fa) return Err("card: " + fa.error());
                if (not fb) return Err("card: " + fb.error());
                b.a = *fa; b.b = *fb;
                pr.dims[std::string(dn)] = std::move(b);
            }
            g.shape.push_back(std::move(pr));
        }
        g.placement.support = text("gen.placement.support");
        if (index_of(kSupportNames, g.placement.support) < 0) return Err("card: unknown support '" + g.placement.support + "'");
        for (const auto& [k, v] : kv)
            if (k.starts_with("gen.placement.ctx."))
            {
                const std::string cn = k.substr(18);
                if (index_of(kContextNames, cn) < 0) return Err("card: unknown placement context '" + cn + "'");
                const auto f = num(k);
                if (not f) return Err("card: " + f.error());
                g.placement.contexts[cn] = *f;
            }
        const auto gm = num("gen.placement.wall_gap_mean"), gs = num("gen.placement.wall_gap_sigma");
        if (not gm) return Err("card: " + gm.error());
        if (not gs) return Err("card: " + gs.error());
        g.placement.wall_gap_mean = *gm; g.placement.wall_gap_sigma = *gs;
        g.bounds_source = text("gen.bounds_source");
        c.generative = std::move(g);
    }
    return c;
}
} // namespace rc::card
