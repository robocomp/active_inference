#pragma once
/*
 * door_close_clear.h — P(the robot is clear of the leaf's swing), the precondition of `close` (2026-11-01).
 *
 * The old rule was the arc alone: P(range > arc), the arc = the leaf's width + the body, on BOTH sides of the wall, because
 * the swing side was never fitted. It is now (the leaf channel's swing log-odds, Task 12), so the arc only matters on the
 * side the leaf actually sweeps. warm10: four closes never offered — after each crossing the robot stood 1.0-1.3 m from a
 * door whose leaf opens into the OTHER room, inside the symmetric 1.5 m arc.
 *
 *   P(clear) = 1 − P(the leaf sweeps a side the body is on) · P(range < arc)
 *   P(the leaf sweeps the body's side) = Σ_side P(swing = side) · P(the body reaches that side)
 *   P(the body reaches side s) = Φ((s · robot_side + body_radius) / σ)   — a body straddling the plane is on both sides
 *
 * No gate: every factor is a probability. Pure (Eigen-free), so the offer and its test share it.
 */
#include <algorithm>
#include <cmath>

namespace rc::door
{
/// The acceptable P(the leaf touches the robot) for a close: the motion policy's own chance constraint (controller
/// [Policy] RiskEps = 0.01, local_policy.h PolicyParams::risk_eps — "acceptable P(contact) per decision"). Swinging a door
/// is an action that can hit the robot like any motion. warm18 leg 3 (2026-11-01): gated at kApproachCompleteProb (0.90),
/// a close went out at P(clear) 0.94 with the robot 0.48 m past the plane, under a WRONG swing belief (llr +4.2), and the
/// leaf closed onto the robot. Not a tuning knob: the same number as the policy's, kept a copy here (pure header).
inline constexpr float kCloseContactRiskMax = 0.01f;

/// Where a robot that comes to rest NEAR its goal has stopped: the controller's place goal is reached within this many metres
/// (controller local_policy.h Goal::tol_xy = 0.25 — a copy, this header stays pure).
inline constexpr float kPlaceArrivalTolM = 0.25f;

/// THE CLOSE STANDPOINT (warm24 leg 1, 2026-11-01): door_concept put it at the band's middle (1.75 m) against a 1.5 m swept
/// arc; the controller counts a place reached within kPlaceArrivalTolM, stopped at ~1.5-1.6 m, P(clear) stayed below the 0.99
/// the request needs, and the close timed out. The standpoint is where P(clear) >= 1 - kCloseContactRiskMax holds wherever the
/// robot ACTUALLY stops: the arc plus z(0.99) times the stop's scatter (the range sigma and the arrival tolerance, combined).
/// Already clear here: stay (`range`).
[[nodiscard]] inline float close_standoff(float range, float arc, float range_sigma, float arrival_tol = kPlaceArrivalTolM)
{
    constexpr float z99 = 2.3263479f;   // Phi^-1(1 - kCloseContactRiskMax)
    return std::max(range, arc + z99 * std::hypot(range_sigma, arrival_tol));
}

/// `range` = |robot − aperture centre|, `robot_side` = its signed distance to the aperture plane along across_u, `arc` = the
/// leaf's swept radius + the body (safe_standoff), `sigma` = the range's σ, `swing_map` = the MAP swing side (±1, the leaf
/// opens toward across_u·swing), `swing_llr_map` = log-odds of that MAP side, `body_radius` = the robot's.
[[nodiscard]] inline float prob_clear_of_swing(float range, float robot_side, float arc, float sigma, float swing_map,
                                               float swing_llr_map, float body_radius)
{
    const float sg = std::max(sigma, 1e-3f);   // 1e-3: numerical (a zero σ is a legal input)
    const auto Phi = [](float z) { return 0.5f * std::erfc(-z / std::sqrt(2.f)); };
    const float p_map = 1.f / (1.f + std::exp(-swing_llr_map));
    const float s_map = swing_map >= 0.f ? 1.f : -1.f;
    const float reach_map   = Phi((s_map * robot_side + body_radius) / sg);    // the body reaches the MAP side
    const float reach_other = Phi((-s_map * robot_side + body_radius) / sg);   // … the other side
    const float p_swept_body_side = p_map * reach_map + (1.f - p_map) * reach_other;
    const float p_in_arc = Phi((arc - range) / sg);
    return std::clamp(1.f - p_swept_body_side * p_in_arc, 0.f, 1.f);
}
}  // namespace rc::door
