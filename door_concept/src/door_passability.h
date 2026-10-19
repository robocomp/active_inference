/*
 * door_passability.h — P(passable) MARGINALISED over the leaf-angle posterior, and the geometry it asks.
 * Split out of door_pragmatics.h (which pulls the affordance protocol / DSR) so offline harnesses can call
 * the exact function the agent publishes door_open_prob with. No behaviour change.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace rc::door
{

// Along-wall clear span left by a leaf at opening angle phi. The leaf is hinged on one vertical aperture
// edge and its projection back onto the wall line is w·cos phi, measured FROM THE HINGE INTO THE APERTURE,
// so the gap is what remains of w. phi = 0 (flush) ⇒ 0; phi >= pi/2 ⇒ the whole aperture.
// ★ cos, NOT |cos| (fixed 2026-10-12). phi runs to DoorModel.PhiMaxRad = 120 deg; past 90 deg the leaf's
//   projection lies BEHIND the hinge, outside the aperture, so it blocks nothing along the wall. |cos| folded
//   120 deg onto 60 deg: live try15, the crossed door's mirror door_29 settled at phi = 2.09 rad (the bound),
//   clear span 0.50 m < 0.70 m of body ⇒ door_open_prob = 0 for a door the robot had just driven through —
//   and room_concept's aperture filter, reading that as "certainly shut", kept every through-door return.
// ⚠This is the ALONG-WALL span only. A leaf swung into the corridor also eats space ACROSS the wall,
// which this does not model — so it is an UPPER bound on passability and `passage_margin_m` is what
// stands in for the rest until the swung leaf is a planner obstacle in its own right.
inline float clear_width(float aperture_w, float phi)
{
    return std::max(0.0f, aperture_w * (1.0f - std::max(0.0f, std::cos(phi))));
}

// P(passable), MARGINALISED over the leaf-angle posterior. `phi_curve` is (phi, unnormalised weight) as
// DoorFitter::estimate_phi leaves it — image support x the persistence/command prior. Returns nullopt when
// the curve is empty or carries no weight: that is "not measured", never "not passable".
inline std::optional<float> passable_prob(const std::vector<std::pair<float, float>>& phi_curve,
                                          float aperture_w, float needed_m)
{
    double num = 0.0, den = 0.0;
    for (const auto& [phi, w] : phi_curve)
    {
        if (not std::isfinite(phi) or not std::isfinite(w) or w <= 0.0f)
            continue;
        den += w;
        if (clear_width(aperture_w, phi) >= needed_m)
            num += w;
    }
    if (den <= 0.0)
        return std::nullopt;
    return static_cast<float>(num / den);
}

// ★ THE SAME QUESTION IN LOG SPACE (Task 11 fix round 1). passable_prob sums FLOAT weights: a posterior a few
// hundredths of a radian wide about phi = 0 underflows every passable angle to exactly 0, the ratio is exactly 0, and
// no later evidence fused with it can move it — "certainly shut" was a rounding artefact, not a belief (try69: P(open)
// 0.000 while the anchor's beams said open at +50..157 nats a frame). `log_curve` is (phi, log weight) — the posterior
// step's own lp, never exponentiated — and the result is the LOG-ODDS log P(passable) − log P(not), by log-sum-exp on
// each side. Finite whenever both sides have a grid point (the grid spans phi = 0 and phi_max). nullopt = no curve.
inline std::optional<double> passable_logit(const std::vector<std::pair<float, double>>& log_curve,
                                            float aperture_w, float needed_m)
{
    constexpr double ninf = -std::numeric_limits<double>::infinity();
    double a = ninf, b = ninf;
    const auto lae = [](double x, double y)
    {
        if (x == -std::numeric_limits<double>::infinity()) return y;
        if (y == -std::numeric_limits<double>::infinity()) return x;
        const double m = std::max(x, y);
        return m + std::log1p(std::exp(-std::abs(x - y)));
    };
    bool any = false;
    for (const auto& [phi, lw] : log_curve)
    {
        if (not std::isfinite(phi) or std::isnan(lw)) continue;
        any = true;
        if (clear_width(aperture_w, phi) >= needed_m) a = lae(a, lw);
        else                                          b = lae(b, lw);
    }
    if (not any or (a == ninf and b == ninf)) return std::nullopt;
    if (a == ninf) return -std::numeric_limits<double>::infinity();
    if (b == ninf) return std::numeric_limits<double>::infinity();
    return a - b;
}
// σ(logit), without overflow at either end.
[[nodiscard]] inline double sigmoid(double l)
{
    return l >= 0 ? 1.0 / (1.0 + std::exp(-l)) : std::exp(l) / (1.0 + std::exp(l));
}

// P(d <= R) for a distance d known to +/- sigma. Normal CDF; sigma is floored by the caller so this can
// never degenerate into a step function.
inline float prob_within(float d, float R, float sigma)
{
    const float s = std::max(sigma, 1e-3f);
    return 0.5f * std::erfc(-(R - d) / (s * std::numbers::sqrt2_v<float>));
}

// ─── cross's precondition: P(passable) × P(at the door) (try102, 2026-10-19) ─────────────────────────────────────────────
// `cross` was offered on P(passable) alone — from ANYWHERE in the room. Its consumer reads an offered cross as "the door is
// within reach" (controller_door_passage.h: `open`/`cross` offered ⇒ reached_door_, the approach pre-empted), and the cross
// is now driven on the passage anchor, a robot-relative measurement of the doorway that this agent hangs at the claim and
// that only the robot's own sweep near the door can refresh. try102 leg 5: after 60 s of exploration the robot stood
// beside chair_2, 5.5 m out; `cross` was on offer (P(passable) 1.00), the controller pre-empted the approach, the anchor
// was hung from 5.5 m, was never once seen from there (no "RESUMED publishing" in door_concept.log between the claims at
// l.10287 and l.11080), and the robot waited facing the opening for a fresh anchor, 40 s, twice. The third attempt crossed
// because the approach had moved the robot first. `open` already has the same two halves ("offering it from across the
// room is a request the consumer would have to chain by itself") — so does `cross`: the joint probability that the door is
// passable AND the robot is at it (P(d ≤ reach), the open door's half of p_reach), under the two independent beliefs. No
// new level: the reach is the actuation band's far edge and its σ the range's own (compute_interaction_state).
[[nodiscard]] inline float cross_offer_prob(float p_passable, float p_within_reach)
{
    return std::clamp(p_passable, 0.0f, 1.0f) * std::clamp(p_within_reach, 0.0f, 1.0f);
}

}  // namespace rc::door
