/*
 * countable_beam_config.h — the smooth beam model's configuration checks and its BeamConfig (spec 2026-11-15
 * lidar-measures §7c.8 a, a0). Pure: no Qt, no DSR, no ConfigLoader, so the belief tests and the fitter share it.
 *
 * beam_config_problem is the load-time validation (countable_sections.h calls it with sigma_sensor = 0, the SensorModel is
 * not on the graph yet). make_beam_config is the ONE place a BeamConfig for the countable belief is built: it repeats the
 * check with the Helios SensorModel's sigma_sensor (1/p0 = 1/LidarPrecision + sigma_sensor^2) and returns the reason
 * instead of a config when the Student-t range density does not exist (nu = p0 c^2 - 1 <= 0). A default BeamConfig{} has
 * no quantile levels; CountableBelief refuses to use one (CountableBelief::beam_ready).
 */
#pragma once

#include <cmath>
#include <expected>
#include <format>
#include <optional>
#include <string>

#include "../../common/ai_belief/beam_ray.h"

namespace rc {

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

// The floor's known-surface σ (spec 7c.8 rev 3.3.8, Model.FloorSigmaM): no default when the flag is on; it must be > 0 (a σ of
// 0 makes the floor's smear the sensor's alone, which the rule does not claim) and finite.
inline std::optional<std::string> floor_sigma_problem(double floor_sigma_m)
{
    if (not (floor_sigma_m > 0.0 and std::isfinite(floor_sigma_m)))
        return std::format("Model.FloorSigmaM = {} must be > 0 when Model.LidarMeasuresOnGated is on (the floor's known-surface sigma, "
                           "spec 7c.8 rev 3.3.8; it has no default)", floor_sigma_m);
    return std::nullopt;
}

// The countable belief's BeamConfig (CountableBeliefParams::beam), at the SENSOR's precision p0 (spec "Range densities"):
// 1/p0 = 1/LidarPrecision + sigma_sensor^2, c = LidarRobustCM, nu = p0 c^2 - 1. Built with BeamConfig::make at the composite
// rule chosen by the Quadrature item (n, z_max, K_m) = (4, 3.0, 8), L = 11. The caller (the fitter, once the Helios
// SensorModel's sigma_sensor is known) refuses or logs on the returned reason exactly as the loader does.
inline std::expected<rc::ai::beam::BeamConfig, std::string>
make_beam_config(double eps_px, double lidar_precision, double robust_c_m, double sigma_sensor_m)
{
    if (const auto why = beam_config_problem(eps_px, lidar_precision, robust_c_m, sigma_sensor_m))
        return std::unexpected(*why);
    const double p0 = 1.0 / (1.0 / lidar_precision + sigma_sensor_m * sigma_sensor_m);
    const auto st = rc::ai::beam::StudentT::make(p0, robust_c_m);
    if (not st) return std::unexpected(std::string("StudentT::make refused p0 c^2 (unreachable after beam_config_problem)"));
    return rc::ai::beam::BeamConfig::make(*st);
}

}  // namespace rc
