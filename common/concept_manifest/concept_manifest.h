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
#include <print>
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

}  // namespace rc::manifest
