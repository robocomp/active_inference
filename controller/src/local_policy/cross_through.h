// cross_through.h — Task 3f (a) (tour3A leg 1): `through` is a POSTERIOR, not a fixed line.
//
// ── WHY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The door passage held `clearing` until s ≥ AnchorCrossStandoffM (2.0 m) along the held gap's normal. The 2 m only ever
// stood for "inside enough to localise in the new room" (Task 12 Fix E, user 10-18). In tour3A the robot was 0.5 m past the
// plane and room_concept had ALREADY localised it in room_2 (the switch), but the 2-m line lay beyond the newborn room's
// outline: unreachable, and with no other exit the passage waited 9 minutes.
//
// ── THE MODEL ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
// through = the body wholly past the door plane (s_true > h, h = the body's extent behind its centre along the normal) AND
// localised in the room beyond. Two independent pieces of evidence about s_true:
//   · the pose driven on (the measured gap, carried by odometry, or the anchor): s_true ~ N(s, σ_s²), σ_s its own σ along
//     the normal;
//   · the ROOM SWITCH into the target room: room_concept hands the robot over when its centre is past the door line, judged
//     on its own pose (R1b: the centre's side of the line with the pre-clamp covariance). Its likelihood is
//     P(switch | s_true) = Φ(s_true / σ_r), σ_r = room_concept's published pose σ — and the same σ_r is how much the switch
//     says about "localised": a switch made on a vague pose moves the posterior little.
// P(through) = ∫_{h}^{∞} N(x; s, σ_s²) Φ(x/σ_r) dx / ∫ N(x; s, σ_s²) Φ(x/σ_r) dx, and the passage ends where it is the MAP
// (> ½: the decision rule rc::anchor_cross_end already applies to its two outcomes). No switch ⇒ no evidence of being
// localised in the new room: this posterior is not the exit (the s ≥ standoff rule stands, unchanged).
// Pure std, no Eigen: tests/test_field_extent.cpp.
#pragma once

#include <algorithm>
#include <cmath>

namespace rc::policy
{
[[nodiscard]] inline double std_normal_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

// P(body wholly past the plane | s on the driven pose with σ_s, the room switched into the target room judged with σ_r).
// σ's ≤ 0 or non-finite are taken as exact (a vanishing σ, the limit of the same integral). Non-finite s judges nothing (0).
[[nodiscard]] inline double p_through_given_switch(double s, double sigma_s, double h, double sigma_r)
{
    if (not std::isfinite(s) or not std::isfinite(h)) return 0.0;
    constexpr double kTiny = 1e-6;
    const double ss = std::isfinite(sigma_s) and sigma_s > kTiny ? sigma_s : kTiny;
    const double sr = std::isfinite(sigma_r) and sigma_r > kTiny ? sigma_r : kTiny;
    // trapezoid over ±8 σ_s around s (the Gaussian's mass beyond is < 1e-15)
    constexpr int kN = 1600;
    const double a = s - 8.0 * ss, b = s + 8.0 * ss, dx = (b - a) / kN;
    double num = 0.0, den = 0.0;
    for (int i = 0; i <= kN; ++i)
    {
        const double x = a + i * dx, z = (x - s) / ss;
        const double w = (i == 0 or i == kN ? 0.5 : 1.0) * std::exp(-0.5 * z * z) * std_normal_cdf(x / sr);
        den += w;
        if (x > h) num += w;
    }
    if (not (den > 0.0))   // the switch is incompatible with every s_true the pose allows: trust the pose (no switch evidence)
        return std_normal_cdf((s - h) / ss);
    return std::clamp(num / den, 0.0, 1.0);
}
}  // namespace rc::policy
