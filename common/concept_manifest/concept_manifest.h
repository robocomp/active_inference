/*
 * concept_manifest.h — read the GEOMETRY block of a <concept>.concept.toml. SHARED, header-only.
 *
 * Step 2B of the declarative-priors experiment (step 1 was the read-only cross-check in each agent's
 * <obj>_config.cpp). This is the first part of a manifest that is AUTHORITATIVE rather than advisory: the
 * agent asks the manifest how its object is supported and how tall it is, instead of inheriting whatever
 * its template happened to say.
 *
 * ★WHY GEOMETRY FIRST, AND WHY `support` IN PARTICULAR. Cloning refrigerator into hood_concept on
 * 2026-08-11 produced an agent whose box spanned z ∈ [0, H] — floor-anchored — because a fridge rests on
 * the floor and a hood hangs. Nothing objected: it compiled, the audit stayed green, and the defect was
 * only visible by sweeping the detector envelope, where the cloned box reported p_detect = 0.000 at every
 * range out to 3 m and put its stand-off at 6.4 m instead of 3.3 m. Fixing it meant hand-editing four
 * sites (the SDF, the point-admission band, the NBV Target, the LiDAR carve) that each re-stated the same
 * assumption in their own words.
 *
 * `support` states it ONCE, as a fact about the object rather than an arithmetic detail repeated four
 * times, and z_span() derives the rest. A clone that inherits `floor_anchored` for a hanging object is
 * then a declaration that is WRONG rather than an omission that is invisible — and wrong declarations can
 * be checked.
 *
 * ★AND EVERY VALUE CARRIES ITS PROVENANCE. `from` is not documentation: `inherited` means "this number
 * arrived by a rename and nobody has chosen it", which is precisely the state hood's geometry was in, and
 * precisely what no display could distinguish from a considered choice. It is machine-checkable.
 *
 * SCOPE: world facts only. Gates, debounces and frame counts are LIFECYCLE and belong to the shared
 * modules — if tuning knobs migrate in here, this becomes etc/config.toml under a new name.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <print>
#include <vector>
#include <string>
#include <string_view>

#include <genericworker.h>   // ConfigLoader

namespace rc::manifest {

// How the object is held up. This is the single statement from which the vertical span follows.
enum class Support
{
    unknown,          // not declared — the agent must not guess
    floor_anchored,   // rests on the floor:      z ∈ [0, z_top]                    (fridge, cabinet, chair)
    leg_supported,    // slab on legs:            z ∈ [z_top − extent, z_top]       (table: the solid band is
                      //                          a lossy abstraction of mostly-empty space — see hollow_delta)
    hangs,            // fixed to a wall, air beneath: z ∈ [z_top − extent, z_top]  (range hood, wall cabinet)
    counter_top       // stands on a worktop:     z ∈ [z_base, z_base + extent]     (microwave, kettle)
};

inline Support support_from(std::string_view s)
{
    if (s == "floor_anchored") return Support::floor_anchored;
    if (s == "leg_supported")  return Support::leg_supported;
    if (s == "hangs")          return Support::hangs;
    if (s == "counter_top")    return Support::counter_top;
    return Support::unknown;
}

inline const char* support_name(Support s)
{
    switch (s)
    {
        case Support::floor_anchored: return "floor_anchored";
        case Support::leg_supported:  return "leg_supported";
        case Support::hangs:          return "hangs";
        case Support::counter_top:    return "counter_top";
        default:                      return "unknown";
    }
}

// Provenance of a declared value. The ORDER is the point: `inherited` is worse than absent, because an
// absent value is asked about and an inherited one is trusted.
enum class From { unknown, inherited, nominal, fitted, measured };

inline From from_of(std::string_view s)
{
    if (s == "measured")  return From::measured;
    if (s == "fitted")    return From::fitted;
    if (s == "nominal")   return From::nominal;
    if (s == "inherited") return From::inherited;
    return From::unknown;
}
inline const char* from_name(From f)
{
    switch (f)
    {
        case From::measured:  return "measured";
        case From::fitted:    return "fitted";
        case From::nominal:   return "nominal";
        case From::inherited: return "inherited";
        default:              return "unknown";
    }
}

struct Geometry
{
    Support support   = Support::unknown;
    float   z_top_m   = 0.0f;   // top of the body above the FLOOR (floor_anchored/leg_supported/hangs)
    float   z_base_m  = 0.0f;   // underside above the floor (counter_top only)
    float   extent_m  = 0.0f;   // vertical extent of the body
    From    from      = From::unknown;
    bool    valid     = false;  // false ⇒ nothing declared; the agent keeps its own defaults and says so

    // The vertical span the whole agent must agree on — the SDF, the point-admission band, the NBV
    // Target's z0/z1 and the LiDAR carve's z_min/z_max. Derived here so those four cannot drift apart.
    void z_span(float& z0, float& z1) const
    {
        switch (support)
        {
            case Support::floor_anchored: z0 = 0.0f;                 z1 = z_top_m;             break;
            case Support::leg_supported:
            case Support::hangs:          z0 = z_top_m - extent_m;   z1 = z_top_m;             break;
            case Support::counter_top:    z0 = z_base_m;             z1 = z_base_m + extent_m; break;
            default:                      z0 = 0.0f;                 z1 = z_top_m;             break;
        }
        if (z1 < z0) std::swap(z0, z1);
    }
};

// Load the [model.geometry] block. Missing file or missing block ⇒ valid = false, never a guess: an agent
// that cannot read its manifest must fall back to its own defaults LOUDLY, not silently adopt zeros.
inline Geometry load_geometry(const std::string& path, std::string_view concept_name = "")
{
    Geometry g;
    ConfigLoader m;
    try { m.load(path); }
    catch (...)
    {
        // ★NOT a benign note. This is the declarative layer being INERT: the agent silently keeps whatever
        // its template left behind, which for a clone is the template's geometry — the exact failure the
        // manifest exists to prevent. It read as harmless for a week: refrigerator's step-1 cross-check
        // was written 2026-08-03 with a path relative to the SOURCE tree ("../../common/…", correct for
        // an #include from <agent>/src/) rather than to the agent's CWD ("../common/…", where it actually
        // runs), so it printed this line and skipped, every start, for a week. Same string, right in one
        // place and wrong in the other, which is precisely why nobody looked twice.
        std::print("[manifest] ★★{} MANIFEST NOT LOADED ({}) — THE DECLARATIVE GEOMETRY IS INERT and the\n"
                   "[manifest]   agent is running on its own defaults, which for a CLONE are its template's.\n"
                   "[manifest]   The path is relative to the agent's CWD, not to src/.\n",
                   concept_name, path);
        return g;
    }
    const auto getf = [&](const char* k, float d) { return m.exists(k) ? static_cast<float>(m.get<double>(k)) : d; };
    const auto gets = [&](const char* k) { return m.exists(k) ? m.get<std::string>(k) : std::string{}; };

    g.support  = support_from(gets("model.geometry.support"));
    g.z_top_m  = getf("model.geometry.z_top_m",  0.0f);
    g.z_base_m = getf("model.geometry.z_base_m", 0.0f);
    g.extent_m = getf("model.geometry.extent_m", 0.0f);
    g.from     = from_of(gets("model.geometry.from"));
    g.valid    = (g.support != Support::unknown) and (g.extent_m > 0.0f or g.support == Support::floor_anchored);

    float z0 = 0.0f, z1 = 0.0f; g.z_span(z0, z1);
    std::print("[manifest] {} geometry: support={} span=[{:.2f},{:.2f}] m  from={}{}\n",
               concept_name, support_name(g.support), z0, z1, from_name(g.from),
               g.from == From::inherited
                   ? "  ★INHERITED — this number arrived by a rename and nobody has chosen it"
                   : "");
    if (not g.valid)
        std::print("[manifest] {} geometry INCOMPLETE — declare model.geometry.support and extent_m\n",
                   concept_name);
    return g;
}

// ─── PROVENANCE: an INHERITED world fact is FATAL ────────────────────────────────────────────────
//
// ★★"INHERITED IS WORSE THAN ABSENT — AN ABSENT VALUE GETS ASKED ABOUT, AN INHERITED ONE GETS TRUSTED."
// That was written as a note when the `from` field was introduced. A note cannot stop anything, and the
// week that followed proved it: hood_concept was cloned from refrigerator and shipped TEN inherited
// defects — a height prior stated in four places with three values (the winner being the fridge's), a
// detector envelope that was a 2296-row measurement OF A DIFFERENT OBJECT, a square-footprint shape prior
// scoring a correct hood at 0.027, an identity prior voting on a hypothesis it does not model, and a sigma*
// set the audit scored 6/6 green because the numbers were present. Every one of them was DECLARED as
// inherited, in writing, in the manifest. Nobody was stopped.
//
// So the declaration is now the enforcement. A manifest block whose numbers arrived by a rename is not a
// manifest, it is a copy, and an agent running on a copy is the exact failure this file exists to prevent.
// The agent REFUSES TO START and names the block.
//
// The escape is not a flag — it is doing the work. Either the value is genuinely a considered class-level
// prior, in which case say `nominal` and write the `why` that makes it one; or it was measured, in which
// case say `measured`; or nobody has chosen it for this object, in which case that is what must change.
// ⚠Relabelling `inherited` to `nominal` to silence this is the one way to defeat it, and it is visible in
// the diff — the `why` is what makes a nominal honest.
struct Provenance
{
    std::string block;      // the TOML table it was declared in, e.g. "sigma_star"
    From        from = From::unknown;
};

// Scan the manifest TEXT for every [block] and the `from` it declares. Textual on purpose: ConfigLoader
// cannot enumerate keys, and this must find blocks nobody wrote a C++ accessor for — those are exactly the
// ones that rot. No numbers are parsed here, so it is locale-proof by construction (see CLAUDE.md).
inline std::vector<Provenance> scan_provenance(const std::string& path)
{
    std::vector<Provenance> out;
    std::ifstream f(path);
    if (not f) return out;
    std::string line, block;
    while (std::getline(f, line))
    {
        const auto b = line.find_first_not_of(" \t");
        if (b == std::string::npos or line[b] == '#') continue;
        if (line[b] == '[')
        {
            const auto e = line.find(']', b);
            if (e != std::string::npos) { block = line.substr(b + 1, e - b - 1); out.push_back({block, From::unknown}); }
            continue;
        }
        if (block.empty() or line.compare(b, 4, "from") != 0) continue;
        const auto q1 = line.find('"');
        const auto q2 = (q1 == std::string::npos) ? std::string::npos : line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        out.back().from = from_of(line.substr(q1 + 1, q2 - q1 - 1));
    }
    return out;
}

// THE RULE. Returns true if every declared block has an honest provenance. On an INHERITED block it prints
// the banner and returns false — the caller must not continue. `blocks_without_from` are reported too but
// do NOT fail: a block that never claimed a provenance is the "absent" case, which gets asked about.
inline bool provenance_ok(const std::string& path, std::string_view concept_name)
{
    const auto ps = scan_provenance(path);
    if (ps.empty()) return true;                       // no manifest / unreadable — load_geometry says so
    std::vector<std::string> inherited, undeclared;
    for (const auto& p : ps)
        if (p.from == From::inherited)      inherited.push_back(p.block);
        else if (p.from == From::unknown)   undeclared.push_back(p.block);

    if (not undeclared.empty())
    {
        std::string list;
        for (const auto& b : undeclared) { if (not list.empty()) list += ", "; list += b; }
        std::print("[manifest] {} — {} block(s) declare no `from`: {}\n"
                   "[manifest]   Not fatal (an absent provenance gets asked about), but a value nobody has\n"
                   "[manifest]   claimed responsibility for is one step from an inherited one.\n",
                   concept_name, undeclared.size(), list);
    }
    if (inherited.empty()) return true;

    std::print("\n"
               "[manifest] ══════════════════════════════════════════════════════════════════════════════\n"
               "[manifest] ★★ {} REFUSES TO START: {} world fact(s) are declared INHERITED.\n",
               concept_name, inherited.size());
    for (const auto& b : inherited)
        std::print("[manifest]      [{}]  — arrived by a rename; nobody has chosen it for THIS object\n", b);
    std::print("[manifest]\n"
               "[manifest]   An inherited value is worse than a missing one: it looks decided. Every one of\n"
               "[manifest]   hood_concept's ten cloned defects was declared exactly like this and shipped.\n"
               "[manifest]   Fix the block in {}:\n"
               "[manifest]     · measured — you took the measurement. Say what of, and when.\n"
               "[manifest]     · fitted   — a fit produced it, from this agent's own data.\n"
               "[manifest]     · nominal  — a considered CLASS-level prior. The `why` is what makes it one.\n"
               "[manifest]   Or delete the block, and the agent will fall back to its own default LOUDLY.\n"
               "[manifest] ══════════════════════════════════════════════════════════════════════════════\n\n",
               path);
    return false;
}

// ─── AUTHORITY: manifest first, config only as an explicit override ──────────────────────────────
//
// The precedence hood proved out, collapsed into one call so it stops being re-typed per value: MANIFEST
// (the world fact) → config.toml (an explicit override, which PRINTS when it contradicts) → the agent's own
// default only if neither speaks. That is the reverse of the order that let hood run on the refrigerator's
// height while two corrected files said otherwise.
inline float resolve(const ConfigLoader& cfg, const char* cfg_key,
                     const ConfigLoader& man, const char* man_key,
                     float agent_default, std::string_view what)
{
    const bool has_man = man.exists(man_key);
    const bool has_cfg = cfg.exists(cfg_key);
    const float m = has_man ? static_cast<float>(man.get<double>(man_key)) : 0.0f;
    const float c = has_cfg ? static_cast<float>(cfg.get<double>(cfg_key)) : 0.0f;
    if (has_man and has_cfg and std::abs(m - c) > 1e-4f)
        std::print("[manifest] OVERRIDE {}: manifest={:.4g} config={:.4g} — the config wins, but a world "
                   "fact is being contradicted; say why in the manifest\n", what, m, c);
    if (has_cfg) return c;
    if (has_man) return m;
    return agent_default;
}

// ─── BAND COHERENCE: does this z-band actually contain the body? ─────────────────────────────────
//
// ★THE CHEAPEST CHECK IN THE FLEET, AND IT WOULD HAVE CAUGHT A WEEK OF WORK. An agent derives several
// vertical bands — which LiDAR returns to select, which voxels it owns, where to carve free space, where to
// sample the silhouette — and each was written as its own arithmetic over "the height". When hood_concept was
// cloned from a floor-anchored parent, those bands kept measuring from the floor while the body hung at
// [1.55, 2.05] m. The LiDAR selection band came out as [−0.10, 0.85]: **disjoint from the body**, 109 returns
// per cycle selected off the floor at a mean 1.33 m from the model, every real hood return excluded, and the
// channel still reporting full coverage. Nothing failed. The audit was green. It took days and a log dig.
//
// It is a contradiction between two numbers the agent already holds, available before a single frame arrives.
// Call this on every derived band at startup. A band that does not intersect the body is never a tuning
// question — it is a statement that cannot be true — so say so loudly and name the band.
inline bool band_contains_body(std::string_view who, std::string_view band_name,
                               float band_lo, float band_hi, const Geometry& g)
{
    if (not g.valid) return true;                     // nothing declared to check against
    float z0 = 0.0f, z1 = 0.0f; g.z_span(z0, z1);
    const float lo = std::min(band_lo, band_hi), hi = std::max(band_lo, band_hi);
    const float overlap = std::min(hi, z1) - std::max(lo, z0);
    const float body    = std::max(1e-3f, z1 - z0);
    const float frac    = std::max(0.0f, overlap) / body;
    if (frac >= 0.999f) return true;
    std::print("[manifest] {}{} band '{}' = [{:.2f},{:.2f}] m covers only {:.0f}% of the declared body "
               "[{:.2f},{:.2f}] — a band that does not contain the body cannot be measuring it\n",
               frac <= 0.0f ? "★DISJOINT: " : "", who, band_name, lo, hi, 100.0f * frac, z0, z1);
    return false;
}

}  // namespace rc::manifest
