/*
 * door_wall_support.h — THE APERTURE LIES INSIDE THE WALL THAT CARRIES IT (2026-11-11, clean50 door_3).
 *
 * A doorway is a hole cut IN a wall, so its along-wall extent [s, s + w] (wall frame: s is the near jamb, m from
 * Aperture::wall_O along wall_u) lies inside the wall segment [a, b] that holds it. The segment's ends are corners of the
 * room outline, each placed with the outline's own positional σ; so the prior of the aperture given its wall is
 *
 *     P(s, w | wall)  ∝  Φ((s − a)/σ) · Φ((b − s − w)/σ)
 *
 * — a door whose jamb overhangs a wall end is improbable in proportion to how far it overhangs, measured in the outline's
 * own uncertainty; a door well inside pays nothing (log Φ → 0). No gate, no clamp: the support is a smooth, log-concave
 * factor, so the fit (DoorBelief::apply_wall_support) can carry it as a Laplace factor and the existence prior
 * (place_prior.h, PlacePrior::extent) can score it as a marginal likelihood.
 *
 * clean50 door_3 (the bathroom's DOOR_0, 5 cm of wall between its west post and the west wall): the fit held the centre
 * 0.21 m west of the truth, so the west jamb sat 0.17 m past the end of the wall — "a door that crosses a wall into the air".
 *
 * WHERE [a, b] COMES FROM: rc::door::door_carrying_wall (door_phi_posterior.h) = door_wall_extent — the in-band polygon edges
 * of the doorway's plane merged into runs (a doorway notch's short sides are in band, so a notch does not split the wall), the
 * run holding the centre or the two runs flanking a gap — and NaN when no run carries the centre (door_wall_extent's
 * "nearest run" fallback names a wall the door is not on). NaN ⇒ that end is unknown ⇒ no factor at that end.
 * NON-WALL EDGES: room_concept publishes no per-edge kind (delimiting_polygon_x/y only). The only non-wall outline it
 * publishes is a PROTO room's (a free-space frontier ring, rc::room::is_proto): a frontier is no evidence of where a wall
 * ends, so the callers pass NaN for a proto room — the same rule along_wall_update's association term already follows.
 * σ: Existence.BoundarySigmaM (door registration ⊕ the published corner σ 0.06), the σ the wall band and the field's
 * room-wall offset already read. ⚑ It is one σ for both corners: room_concept does not publish a per-corner σ on the graph.
 *
 * Header-only, Eigen-free, DSR-free.
 */
#pragma once
#include <cmath>
#include <limits>
#include <numbers>

namespace rc::door
{

/// The wall segment that carries a door, along its wall (m, same frame as DoorBeliefState::s), and the outline's σ there.
struct WallSupport
{
    float lo = std::numeric_limits<float>::quiet_NaN();   // a: the near end (NaN = unknown)
    float hi = std::numeric_limits<float>::quiet_NaN();   // b: the far end  (NaN = unknown)
    float sigma = 0.0f;                                   // the outline's positional σ at the corners (m)
    [[nodiscard]] bool active() const { return (std::isfinite(lo) or std::isfinite(hi)) and sigma > 0.0f; }
};

namespace wall_support_detail
{
/// log Φ(z), stable for very negative z (erfc's tail, then its asymptote — a floating-point domain guard, not a model cutoff).
[[nodiscard]] inline double log_Phi(double z)
{
    if (z > -30.0)
        return std::log(0.5 * std::erfc(-z / std::numbers::sqrt2));
    return -0.5 * z * z - std::log(-z) - 0.5 * std::log(2.0 * std::numbers::pi);
}
/// λ(z) = φ(z)/Φ(z), the inverse Mills ratio: d log Φ/dz. Same tail treatment.
[[nodiscard]] inline double mills(double z)
{
    if (z > -30.0)
        return std::exp(-0.5 * z * z - 0.5 * std::log(2.0 * std::numbers::pi) - log_Phi(z));
    return -z - 1.0 / z;   // asymptote of φ/Φ as z → −∞
}
}   // namespace wall_support_detail

/// The log-support at (s, w) and its derivatives in (s, w): f, ∇f (gs, gw) and the NEGATIVE Hessian (hss, hsw, hww), which is
/// positive semi-definite (log Φ is concave). An inactive support returns all zeros.
struct WallSupportEval
{
    double f = 0.0, gs = 0.0, gw = 0.0, hss = 0.0, hsw = 0.0, hww = 0.0;
};
[[nodiscard]] inline WallSupportEval eval_wall_support(double s, double w, const WallSupport& ws)
{
    using namespace wall_support_detail;
    WallSupportEval e;
    if (not ws.active())
        return e;
    const double sg = ws.sigma;
    if (std::isfinite(ws.lo))
    {   // z = (s − a)/σ : ∂z/∂s = 1/σ
        const double z = (s - static_cast<double>(ws.lo)) / sg;
        const double l = mills(z), c = l * (l + z);   // −d² log Φ / dz²  ∈ (0, 1)
        e.f += log_Phi(z);
        e.gs += l / sg;
        e.hss += c / (sg * sg);
    }
    if (std::isfinite(ws.hi))
    {   // z = (b − s − w)/σ : ∂z/∂s = ∂z/∂w = −1/σ
        const double z = (static_cast<double>(ws.hi) - s - w) / sg;
        const double l = mills(z), c = l * (l + z);
        const double k = c / (sg * sg);
        e.f += log_Phi(z);
        e.gs -= l / sg;
        e.gw -= l / sg;
        e.hss += k; e.hsw += k; e.hww += k;
    }
    return e;
}

/// The support as a MARGINAL LIKELIHOOD over the fit's own uncertainty (existence prior, place_prior.h): each jamb's position
/// is known to sd_lo / sd_hi, so its overhang is scored under σ² + sd². 0 deep inside, → −∞ for a jamb metres past the end.
[[nodiscard]] inline float wall_support_log_marginal(float s, float w, const WallSupport& ws, float sd_lo, float sd_hi)
{
    using namespace wall_support_detail;
    if (not ws.active())
        return 0.0f;
    double t = 0.0;
    if (std::isfinite(ws.lo))
        t += log_Phi((static_cast<double>(s) - ws.lo) / std::hypot(static_cast<double>(ws.sigma), static_cast<double>(sd_lo)));
    if (std::isfinite(ws.hi))
        t += log_Phi((static_cast<double>(ws.hi) - s - w) / std::hypot(static_cast<double>(ws.sigma), static_cast<double>(sd_hi)));
    return static_cast<float>(t);
}

}   // namespace rc::door
