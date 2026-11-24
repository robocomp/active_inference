/*
 * common/ai_belief/sensor_var.h — variances a sensor channel feeds the shared estimation engine (header-only, pure).
 *
 *   camera_shared_var : the shared (common-mode) variance of a camera edge, in metres²: a pixel error σ_px at range z
 *                       with focal length f is σ_px·z/f metres across the optical axis, and 1/cos(obliquity) longer
 *                       along a surface seen at that obliquity.
 *   fresh_var         : the variance a moving platform adds between two samples, (v·τ)²: speed times the staleness.
 */
#pragma once

#include <algorithm>

namespace rc::ai::sensor {

// (σ_px · z / f / cos_obl)².
inline double camera_shared_var(double sigma_px, double z_m, double f_px, double cos_obliquity)
{
    // max(cos, 1e-3) is a numerical guard against dividing by zero at a grazing view, not a tuning constant.
    const double s = sigma_px * z_m / f_px / std::max(cos_obliquity, 1e-3);
    return s * s;
}

// (v · τ)².
inline double fresh_var(double speed_mps, double tau_s)
{
    const double d = speed_mps * tau_s;
    return d * d;
}

}  // namespace rc::ai::sensor
