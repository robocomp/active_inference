#pragma once
/*
 * door_cross_aif.h — the door cross as ACTIVE-INFERENCE ACTION CONTROL in the robot frame
 * (spec docs/superpowers/specs/2026-10-31-door-cross-aif-servo-design.md).
 *
 * The goal is an OBSERVATION, not a pose: "the door frame behind me" — the body's front on the opening's cross line,
 * past the plane. The action is the one that makes that predicted observation least surprising: it descends the free
 * energy of the observation predicted one horizon ahead, and the derivative it needs, ∂o/∂a, is the robot-frame
 * interaction matrix of the opening. Nothing here reads a room frame.
 *
 * GEOMETRY (robot frame, forward = +y; anchor as controller_anchor_servo.h): the opening's centre A, wall direction a,
 * normal n through it from the robot's side. Shadow is DIFFERENTIAL (v forward, ω), so the axle midpoint cannot move
 * sideways and a gradient on it stalls beside the axis. The CONTROLLED POINT is the body's front, P = (0, d), d = the
 * body's reach along +y (the part that threads the opening first). Its velocity is
 *     Ṗ = J a,   J = [[0, −d], [1, 0]],   a = (v, ω),   det J = d ≠ 0   ⇒ P is fully actuated.
 * Observations of P in the anchor frame:  e = a·(P − A) − line  (off the cross line, the clear gap's centre),
 *                                         s = n·(P − A)          (progress through the plane).
 * Interaction matrix:  L = [aᵀ; nᵀ] J   (a, n fixed in the world; first order in the control period).
 *
 * GENERATIVE MODEL / PRIORS (no gains):
 *   goal on s:  s* = standoff + d (the axle `standoff` past the plane when aligned), precision Π_s = 1/σ_s², σ_s the
 *               opening's own σ along its normal (how well the robot knows where the plane is);
 *   goal on e:  e* = 0, precision Π_e = 1/(r² + σ_lat²), r = the clear gap's half width minus the body's half width —
 *               the room the body has on each side (a tight doorway is a stiff preference, a wide one a loose one);
 *   PROGRESS IS EXPECTED ONLY AS FAR AS IT IS POSSIBLE: the progress the step expects is P(fit)·(s* − s), P(fit) =
 *               P(the body fits where it will enter the opening) (anchor_p_fit_at_entry, the old law's own evidence) —
 *               an expectation over {fits, does not}, not a precision (as a precision it saturated: a precise plane
 *               drove full speed whatever P(fit), 1/75 contacts at 5 cm / 3 deg anchor noise). Without it the front reached the line with
 *               the body still off and angled and a corner clipped the jamb (harness: 6/75 contacts from 0.5 m off, 0.8 m
 *               out). A heading goal at the plane was tried instead and made it worse (23/75): a unicycle corrects a
 *               lateral offset BY heading off the normal, so the two preferences fight;
 *   action prior a ~ N(0, diag(v_max², w_max²)): the base's rates are the scale of a plausible action.
 * Each observation is predicted at ITS horizon: e must be null by the time the front reaches the plane,
 * T_e = max(−s, 0)/v_max + dt, s by the time it reaches its goal, T_s = max(s* − s, 0)/v_max + dt (dt = control period:
 * once there, the next step). Minimising F(a) = ½ Σ_i Π_i (o_i + L_i a T_i − o_i*)² + ½ aᵀ Π_a a gives the closed form
 *     a* = (Σ_i Π_i T_i² L_iᵀ L_i + Π_a)⁻¹ Σ_i Π_i T_i L_iᵀ (o_i* − o_i).
 * v ≥ 0 (the cross does not reverse; a negative optimum is replaced by the optimum on ω alone, v = 0), and the pair is
 * scaled — not clipped per axis — to the base's rates. Contact with the jambs is NOT modelled here: the caller's guard
 * chain (step_anchor_cross) caps the command as for the old law (spec §3.3, V1).
 *
 * Header-only, pure Eigen + std (+ controller_anchor_servo.h for the shared anchor geometry).
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#include <Eigen/Dense>

#include "controller_anchor_servo.h"

namespace rc
{
struct DoorCrossAifBody
{
    float half_width = 0.30f;   ///< lateral half width (m)
    float front = 0.33f;        ///< reach along +y: the controlled point's distance ahead of the axle (m)
};

/// Diagnostics of one step (what the law saw and chose); the command is in `cmd`.
struct DoorCrossAifStep
{
    AnchorCommand cmd;
    float e = 0.f, s = 0.f;       ///< the controlled point's observations (m)
    float s_goal = 0.f;           ///< s*
    float pi_e = 0.f, pi_s = 0.f; ///< the goal precisions
    float p_fit = 1.f;            ///< P(body fits at entry): the weight of the progress preference
    float t_e = 0.f, t_s = 0.f;   ///< the horizons (s)
};

[[nodiscard]] inline DoorCrossAifStep door_cross_aif(const rc::interaction::RelPose& r, float width,
                                                     std::span<const rc::interaction::AnchorLeaf> leaves,
                                                     const DoorCrossAifBody& body, float standoff, float v_max, float w_max,
                                                     float dt, float free_advance = std::numeric_limits<float>::infinity())
{
    DoorCrossAifStep out;
    if (not r.x.allFinite() or not r.P.allFinite() or not (v_max > 0.f) or not (w_max > 0.f) or not (dt > 0.f)) return out;
    const auto g = anchor_cross_geometry(r);
    out.cmd.through = g.s >= standoff;
    if (out.cmd.through) return out;

    const float d = body.front;
    const Eigen::Vector2f A = r.x.head<2>();
    const Eigen::Vector2f P(0.f, d);
    const float line = anchor_cross_line(width, leaves, body.half_width, g.sigma_lat);
    out.e = g.a.dot(P - A) - line;
    out.s = g.n.dot(P - A);
    out.s_goal = standoff + d;

    // the room each side: the clear gap beside any leaf, less the body
    float clear_half = 0.5f * width;
    for (const auto& l : leaves) clear_half = std::min(clear_half, 0.5f * anchor_clear_gap(width, &l).width);
    // the body as it stands now: angled by ψ it spans half_w·|cos ψ| + d·|sin ψ| across the opening
    const float span = body.half_width * std::abs(std::cos(g.heading_err)) + d * std::abs(std::sin(g.heading_err));
    const float room = std::max(0.f, clear_half - span);
    const float sigma_n2 = std::max(1e-6f, g.n.dot(r.P.topLeftCorner<2, 2>() * g.n));
    out.pi_e = 1.f / std::max(1e-6f, room * room + g.sigma_lat * g.sigma_lat);
    out.p_fit = anchor_p_fit_at_entry(width, leaves, body.half_width, g.e, line, g.s, d, v_max, g.sigma_lat, free_advance);
    out.pi_s = 1.f / sigma_n2;
    out.t_e = std::max(-out.s, 0.f) / v_max + dt;
    out.t_s = std::max(out.s_goal - out.s, 0.f) / v_max + dt;

    Eigen::Matrix2f J;
    J << 0.f, -d, 1.f, 0.f;
    const Eigen::RowVector2f Le = g.a.transpose() * J;
    const Eigen::RowVector2f Ls = g.n.transpose() * J;
    const Eigen::Matrix2f Pa = Eigen::Vector2f(1.f / (v_max * v_max), 1.f / (w_max * w_max)).asDiagonal();
    const float we = out.pi_e * out.t_e, ws = out.pi_s * out.t_s;
    const Eigen::Matrix2f H = we * out.t_e * Le.transpose() * Le + ws * out.t_s * Ls.transpose() * Ls + Pa;
    const Eigen::Vector2f b = we * Le.transpose() * (0.f - out.e) + ws * Ls.transpose() * (out.p_fit * (out.s_goal - out.s));
    Eigen::Vector2f a = H.ldlt().solve(b);
    if (a.x() < 0.f)   // no reversing: the optimum over ω alone at v = 0
        a = Eigen::Vector2f(0.f, b.y() / H(1, 1));
    const float scale = std::max({1.f, std::abs(a.x()) / v_max, std::abs(a.y()) / w_max});
    out.cmd.adv = a.x() / scale;
    out.cmd.rot = a.y() / scale;
    out.cmd.side = 0.f;
    return out;
}
}  // namespace rc
