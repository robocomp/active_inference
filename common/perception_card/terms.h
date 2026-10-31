// terms.h -- the CLOSED LIBRARY of P(detect) factor functions a perception card may name (card.h kTermNames).
// Header-only, pure, no Eigen / OpenCV / DSR: every function takes plain scalars.
//
// ★THESE ARE MOVED, NOT WRITTEN. Each body is the formula door_concept / cabinet_concept computed privately until
// 2026-10-31; those agents now CALL these functions, so the agent and the shared evaluator (card -> P(detect) for
// the planner and the dashboard) are the same code by construction and cannot drift. What stays in the agent is the
// MEASUREMENT (the silhouette counting, the pose covariance); what is here is the formula applied to it.
// (It includes term_spec.h, not card.h: the door core is framework-free and test_door_core_tracker forbids any
// media_transport include under src/core, which card.h's JSON codec would bring.)
// Pinned to the agents' pre-move outputs by tests/terms_test.cpp (golden tables, 1e-6).
//
//   term              moved from                                              agent still measures
//   envelope          rc::detect::p_detect (detectability.h), both overloads   roi_fill, visible fraction
//   in_fov            identity on the fraction (see below)                     DoorSilhouette / CabinetSilhouette::in_fov_frac
//   central           identity on the fraction (see below)                     ...::central_frac (centroid vs principal point)
//   range_conf        cabinet_existence.cpp absence_range_conf                 mean range of the detectable samples
//   leaf_obliquity    door_existence.cpp door_view_obliquity                   leaf pose, camera pose
//   resolvability     door_existence.h DoorSilhouette::resolvability           n_cells, n_detectable
//   kappa             door_existence.cpp pose_overlap_factor (closed form)     RT covariance -> var_b, range
//   coverage          the visible fraction, passed through
//   motion            Task 3 (instance-level, not a view factor)
#pragma once

#include "../detectability/detectability.h"
#include "term_spec.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rc::card
{
// P(detect | geometry) from the detector envelope. Both overloads of rc::detect::p_detect, unchanged.
inline float term_envelope(float fill_max, float fill_min, float visible, const rc::detect::DetectorEnvelope& e)
{
    return rc::detect::p_detect(fill_max, fill_min, visible, e);
}
inline float term_envelope(float fill, float visible, const rc::detect::DetectorEnvelope& e)
{
    return rc::detect::p_detect(fill, visible, e);
}

// Both agents use the silhouette fractions DIRECTLY as P(detect) factors, with no further formula, so these two
// terms are the identity on the fraction. The fraction itself is the agent's measurement (it folds the real
// frustum + occluders for in_fov; the centroid-vs-principal-point radius for central) and stays there.
inline float term_in_fov(float in_fov_frac) { return in_fov_frac; }
inline float term_central(float central_frac) { return central_frac; }

// Cabinet's absence confidence vs range: min(1, (ref/range)^power), 1 up close and ->0 far. ref<=0 or power<=0
// disables it (constant 1).
inline float term_range_conf(float range_m, float ref_m, float power)
{
    if (ref_m <= 0.0f or power <= 0.0f) return 1.0f;
    return std::min(1.0f, std::pow(ref_m / std::max(range_m, 1e-3f), power));
}

// |cos| of the camera->leaf ray against the LEAF's face normal. 1 = square-on, 0 = grazing. (rx, ry) = leaf centre
// minus camera, in the plan; (nx, ny) = the leaf's unit face normal. Coincident camera and leaf -> 1 (cannot judge,
// do not suppress). The caller decides when there is no pose / no fitted leaf (it passes nothing and keeps 1).
inline float term_leaf_obliquity(float rx, float ry, float nx, float ny)
{
    const float z = rx * rx + ry * ry;
    if (std::sqrt(z) < 1e-6f) return 1.0f;
    const float n = std::sqrt(z);
    rx /= n; ry /= n;
    return std::clamp(std::abs(rx * nx + ry * ny), 0.0f, 1.0f);
}

// Resolvability in (0,1]: the fraction of the sampling budget landing in DISTINCT image cells. 0 if nothing was
// detectable.
inline float term_resolvability(int n_cells, int n_detectable)
{
    return n_detectable > 0 ? static_cast<float>(n_cells) / n_detectable : 0.0f;
}

// Expected overlap of the predicted image window with the true door under the robot-pose uncertainty (door
// pose_overlap_factor). var_b = bearing variance (sigma_yaw^2 + sigma_xy^2/range^2), range = camera->door distance,
// half_diag_m = the door's half-diagonal. Closed form (2Phi(a/sb)-1) - (2 sb/(a sqrt(2pi)))(1-exp(-a^2/(2 sb^2))),
// a = 2 atan(half_diag/range): 1 for a well-localised robot, -> 0 as the pose spreads. Requires var_b > 0.
inline float term_kappa(double var_b, double range, double half_diag_m)
{
    const double sb = std::sqrt(var_b);
    const double a = 2.0 * std::atan(std::max(1e-3, half_diag_m) / range);
    const double z = a / sb;
    const double k = std::erf(z / std::sqrt(2.0))
                   - (2.0 / (z * std::sqrt(2.0 * std::numbers::pi))) * (1.0 - std::exp(-0.5 * z * z));
    return static_cast<float>(std::clamp(k, 0.0, 1.0));
}

// The visible fraction, passed through.
inline float term_coverage(float visible) { return visible; }

// The intermediate quantities a view produces; eval_term picks what each term needs.
struct TermInputs
{
    float fill_max = 0, fill_min = 0;   // projected bbox fill, larger / smaller axis
    float visible  = 1;                 // fraction left visible by frustum + occluders
    float in_fov   = 1, central = 1;    // the silhouette fractions
    float range    = 0;                 // camera -> object, m
    float phi      = 0;                 // articulation angle (reserved for Task 3 / instance terms)
    float obliquity = 1;                // cos of the ray vs the face normal (sign irrelevant)
    float resolvability = 1;            // distinct-cell fraction (a measurement the evaluator must supply)
    float kappa    = 1;                 // pose-overlap factor (likewise)
};

// Dispatch by name. Parameters come from spec.params: envelope {min_fill,max_fill,soft}; range_conf {ref_m,power}.
// A parameter that is absent takes the DetectorEnvelope default / disables range_conf. motion is Task 3's: until
// then it (and any unknown name) is neutral, 1.
inline float eval_term(const TermSpec& s, const TermInputs& in)
{
    const auto p = [&](const char* k, float d) { const auto it = s.params.find(k); return it != s.params.end() ? it->second : d; };
    if (s.name == "envelope")
    {
        const rc::detect::DetectorEnvelope def;
        const rc::detect::DetectorEnvelope e{p("min_fill", def.min_fill), p("max_fill", def.max_fill), p("soft", def.soft)};
        return term_envelope(in.fill_max, in.fill_min, in.visible, e);
    }
    if (s.name == "in_fov")         return term_in_fov(in.in_fov);
    if (s.name == "central")        return term_central(in.central);
    if (s.name == "range_conf")     return term_range_conf(in.range, p("ref_m", 0.0f), p("power", 0.0f));
    if (s.name == "leaf_obliquity") return std::clamp(std::abs(in.obliquity), 0.0f, 1.0f);
    if (s.name == "resolvability")  return in.resolvability;
    if (s.name == "kappa")          return in.kappa;
    if (s.name == "coverage")       return term_coverage(in.visible);
    return 1.0f;
}
}  // namespace rc::card
