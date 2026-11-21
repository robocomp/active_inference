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

// Why a smooth-beam configuration cannot run, or nullopt (spec 7c.8): the persistent label needs 0 < eps_px < 1/2 (the camera's
// pixel-error rate; a rate of 0 or 1/2 makes a label certain or uninformative), and the Student-t range density needs
// nu = p0 c^2 - 1 > 0 with 1/p0 = 1/LidarPrecision + sigma_sensor^2 (sigma_sensor = 0 when the SensorModel is absent). Pure.
inline std::optional<std::string> beam_config_problem(double eps_px, double lidar_precision, double robust_c_m,
                                                      double sigma_sensor_m = 0.0)
{
    if (not (eps_px > 0.0 and eps_px < 0.5))
        return std::format("Model.MaskPixelErrorRate = {} must lie in the open interval (0, 1/2) when Model.LidarMeasuresOnGated is on "
                           "(it has no default: measure it, see countable_concept/tools/eps_px_measure.py)", eps_px);
    if (not (lidar_precision > 0.0))
        return std::format("Model.LidarPrecision = {} must be > 0 when Model.LidarMeasuresOnGated is on", lidar_precision);
    const double p0 = 1.0 / (1.0 / lidar_precision + sigma_sensor_m * sigma_sensor_m);
    const double p0c2 = p0 * robust_c_m * robust_c_m;
    if (not (p0c2 > 1.0))
        return std::format("Model.LidarPrecision {} with Model.LidarRobustCM {} (sigma_sensor {} m) gives p0*c^2 = {} <= 1: the "
                           "Student-t range density of the beam model does not exist (nu = p0*c^2 - 1 must be > 0)",
                           lidar_precision, robust_c_m, sigma_sensor_m, p0c2);
    return std::nullopt;
}

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
}

}  // namespace rc
