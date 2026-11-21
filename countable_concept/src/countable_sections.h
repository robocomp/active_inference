/*
 * countable_sections.h — refuse a config written for one of the pre-countable clones.
 *
 * nightstand_concept / bed_concept / toilet_concept each had their own [<X>Model] / [<X>Concept] sections. The
 * countable reader reads only [Model] / [Concept], and every key is optional, so an unmigrated config would start
 * with EVERY tuned value silently replaced by a code default. A section name is not a belief: there is no model term
 * for it, so this is a plain structural check — it names the section and points at the key map.
 *
 * Needs only RoboComp's ConfigLoader (no Qt / DSR), so a standalone test can include it.
 */
#pragma once

#include <ConfigLoader/ConfigLoader.h>

#include "countable_config_error.h"   // rc::CountableConfigError
#include "countable_beam_config.h"    // rc::beam_config_problem

#include <algorithm>
#include <cmath>
#include <optional>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace rc {

// The first (alphabetically, so the message is deterministic) top-level section whose name ends in "Model" or
// "Concept" and is not exactly [Model] / [Concept]; nullopt when there is none.
inline std::optional<std::string> pre_countable_section(const ConfigLoader& cfg)
{
    std::vector<std::string> found;
    for (const std::string_view k : cfg.getKeys())
    {
        const std::string_view sec = k.substr(0, k.find('.'));
        if (sec.size() == k.size()) continue;   // a key outside any section
        if ((sec.ends_with("Model") or sec.ends_with("Concept")) and sec != "Model" and sec != "Concept")
            found.emplace_back(sec);
    }
    if (found.empty()) return std::nullopt;
    return *std::ranges::min_element(found);
}

// Throw rc::CountableConfigError (main prints it and exits non-zero) when the config still uses a pre-countable section.
inline void refuse_pre_countable_config(const ConfigLoader& cfg)
{
    if (const auto sec = pre_countable_section(cfg))
        throw CountableConfigError(std::format("config uses pre-countable section [{}] — see countable_concept/KEYMAP.md\n", *sec));
}

// beam_config_problem (the smooth-beam config validation) lives in countable_beam_config.h (pure, shared with the belief).

// The same check on a loaded config (sigma_sensor is not known at load: it comes from the graph later; the caller checks
// again with it). Does nothing unless Model.LidarMeasuresOnGated is true. A missing MaskPixelErrorRate reads as 0 => refused.
inline void refuse_bad_beam_config(const ConfigLoader& cfg)
{
    const auto get = [&](const char* k, double def) {
        try { return cfg.exists(k) ? cfg.get<double>(k) : def; } catch (...) { return def; }
    };
    bool on = false;
    try { on = cfg.exists("Model.LidarMeasuresOnGated") and cfg.get<bool>("Model.LidarMeasuresOnGated"); } catch (...) {}
    if (not on) return;
    if (const auto why = beam_config_problem(get("Model.MaskPixelErrorRate", 0.0), get("Model.LidarPrecision", 0.0),
                                             get("Model.LidarRobustCM", 0.05)))
        throw CountableConfigError(std::format("config: {} — refusing to start\n", *why));
    if (const auto why = floor_sigma_problem(get("Model.FloorSigmaM", 0.0)))   // missing ⇒ 0 ⇒ refused (no default)
        throw CountableConfigError(std::format("config: {} — refusing to start\n", *why));
}

}  // namespace rc
