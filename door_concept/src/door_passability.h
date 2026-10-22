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

// The along-wall clear span MARGINALISED over the leaf-angle posterior: E[clear_width(w, phi)] under `log_curve` ((phi, log
// weight), the posterior step's own lp). The span a log line or a planner quotes must come from the whole posterior, not from
// its modal angle: a posterior straddling a shut and an open leaf has a mean span between the two, and the modal one is a lie
// about it. nullopt-like 0 when the curve is empty.
inline float clear_width_expected(const std::vector<std::pair<float, double>>& log_curve, float aperture_w)
{
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& [phi, lw] : log_curve) if (std::isfinite(phi) and not std::isnan(lw)) m = std::max(m, lw);
    if (not std::isfinite(m)) return 0.0f;
    double num = 0.0, den = 0.0;
    for (const auto& [phi, lw] : log_curve)
    {
        if (not std::isfinite(phi) or std::isnan(lw)) continue;
        const double w = std::exp(lw - m);
        num += w * clear_width(aperture_w, phi);
        den += w;
    }
    return den > 0.0 ? static_cast<float>(num / den) : 0.0f;
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
// ★Task E: `sigma_w` marginalises the aperture width too. Under each leaf angle the clear span is w·(1 − cos φ) with w ~
// N(ŵ, σ_w²), so P(span ≥ needed | φ) = Φ((ŵ(1 − cos φ) − needed)/(σ_w(1 − cos φ))) instead of an indicator; a shut leaf
// (cos φ ≥ 1) still has no span whatever the width. σ_w = 0 reproduces the indicator exactly.
inline std::optional<double> passable_logit(const std::vector<std::pair<float, double>>& log_curve,
                                            float aperture_w, float needed_m, float sigma_w = 0.0f)
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
        const double open_frac = std::max(0.0, 1.0 - std::cos(static_cast<double>(phi)));   // clear span per metre of width
        if (not (sigma_w > 0.0f) or not (open_frac > 0.0))
        {
            if (clear_width(aperture_w, phi) >= needed_m) a = lae(a, lw);
            else                                          b = lae(b, lw);
            continue;
        }
        const double z = (static_cast<double>(aperture_w) * open_frac - needed_m) / (static_cast<double>(sigma_w) * open_frac);
        const double p_ok = 0.5 * std::erfc(-z / std::numbers::sqrt2);
        const double tiny = std::numeric_limits<double>::min();   // log(0) guard, not a decision
        a = lae(a, lw + std::log(std::max(p_ok, tiny)));
        b = lae(b, lw + std::log(std::max(1.0 - p_ok, tiny)));
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

// ─── AT THE DOOR INCLUDES THE VIEWING ANGLE (tourP8, 2026-10-21) ─────────────────────────────────────────────────────────
// tourP8 leg 1 opened and crossed door_2 from door_4's standpoint: 1.95 m from the aperture centre — inside the actuation
// band, so P(in band) and P(d ≤ reach) were high — but 58° off the door's normal and 0.69 m from its wall plane. From there
// the anchor never saw the doorway (P(door in view) never reached 0.5), the φ estimate collapsed at grazing incidence, and
// the cross timed out. Range alone does not say the robot is AT a door: the doorway must also be IN FRONT of it.
// The view of an opening of width w from a point at angle θ off its normal is its apparent (projected) span w·|cos θ| —
// the span the scanner's beams can pass through and the camera's two jambs subtend, which is what the anchor's P(door in
// view) is built from (lidar_view's opening cover, camera_view's jambs). The robot is "at the door" for open / cross when
// that apparent opening admits its body: P(w·|cos θ| ≥ needed) under the range σ — the same body width the passability
// uses and the same σ the band uses. Continuous; no angle threshold: 0° ⇒ the full aperture, 90° ⇒ a sliver.
// `cos_theta` = (robot − centre)/|robot − centre| · normal (its sign — which face — does not change how much is seen).
[[nodiscard]] inline float facing_cos(float rx, float ry, float nx, float ny)
{
    const float r = std::hypot(rx, ry), n = std::hypot(nx, ny);
    return (r > 1e-4f and n > 1e-6f) ? (rx * nx + ry * ny) / (r * n) : 1.0f;   // on the centre: nothing is oblique
}
[[nodiscard]] inline float prob_facing_opening(float cos_theta, float aperture_w, float needed_w, float sigma)
{
    const float apparent = std::max(0.0f, aperture_w) * std::min(1.0f, std::abs(cos_theta));
    return prob_within(needed_w, apparent, sigma);   // P(needed ≤ apparent)
}

// ─── NO PASSABLE ON PREDICTION ALONE (tourP8 D5) ────────────────────────────────────────────────────────────────────────
// The actuator-command anticipation moves φ toward the commanded angle so the agent is not surprised by its own action —
// a PRIOR. tourP8 asserted `transitable` at φ = 89.5° with measured = 0 on every row, i.e. on the request echoed back.
// The streak toward ASSERTING advances only on a cycle a leaf / aperture likelihood contributed to (DoorInstance::
// phi_measured); an unmeasured cycle HOLDS it (neither advances nor resets). Retraction — the safe direction — still
// follows any known cycle, as before.
struct TransitableStreaks { int yes = 0, no = 0; };
struct TransitableDecision { bool assert_now = false, retract_now = false; };
[[nodiscard]] inline TransitableDecision transitable_step(TransitableStreaks& s, bool asserted, float p_passable, bool measured,
                                                          float yes_prob, float no_prob, int stable_cycles)
{
    const bool yes = p_passable >= yes_prob;
    const bool no  = p_passable <  no_prob;
    if (measured) s.yes = yes ? s.yes + 1 : 0;   // evidence only
    s.no = no ? s.no + 1 : 0;
    TransitableDecision d;
    d.assert_now  = not asserted and measured and s.yes >= stable_cycles;
    d.retract_now = asserted and s.no >= stable_cycles;
    return d;
}

[[nodiscard]] inline bool near_equal(float a, float b, float tol = 1e-6f) { return std::abs(a - b) <= tol; }

}  // namespace rc::door
