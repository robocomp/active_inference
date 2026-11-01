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
