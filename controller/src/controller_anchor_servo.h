#pragma once
/*
 * controller_anchor_servo.h — cross a doorway by servoing on its INTERACTION ANCHOR, in the robot frame (2026-10-18).
 *
 * While door_concept executes `cross`, it hangs a shallow copy of the door from the robot (`door_N@robot`); the RT
 * edge robot→copy is the opening's pose IN THE ROBOT FRAME with its covariance (interaction_anchor.h). Anchor frame:
 * origin = the opening's centre on the wall plane, x along the wall, y = the wall normal, robot on −y before crossing.
 * Room-frame door and robot poses were 0.4–0.5 m wrong and drove the robot into the wall; this law never touches a
 * room frame, so a relocalisation or a room switch cannot move the target under it.
 *
 * GEOMETRY. The anchor is at A = (ax, ay) with yaw ψ, all in the robot frame (forward = +y, side = +x).
 *   a = (cos ψ, sin ψ)    the wall direction (anchor +x) in the robot frame
 *   n = (−sin ψ, cos ψ)   the wall normal    (anchor +y) in the robot frame — the direction of travel
 * The robot origin in the anchor frame is Rᵀ(0 − A) = (−a·A, −n·A), so
 *   e = −a·A   the robot's signed offset ALONG THE WALL from the opening's normal line (0 = centred)
 *   s = −n·A   progress along the normal: < 0 before the opening, 0 on the wall plane, > 0 past it
 *   heading error = ψ: n is the robot's forward +y rotated by ψ CCW, so a positive ψ wants a CCW (positive) turn.
 *
 * LAW (omnidirectional base, robot-frame velocity v = v_n·n + v_a·a):
 *   v_a = −v_max·tanh(e / (v_max·τ_lat))                  nulls e (moves the robot back onto the normal line)
 *   v_n =  v_max·P_fit·tanh((standoff − s)/(v_max·τ_s))   advances, tapering to 0 at the standoff
 *   rot =  w_max·tanh(ψ / (w_max·τ_rot))                  aligns forward with n
 * tanh is a smooth saturation with unit slope at 0, so each term is a first-order loop with time constant τ that
 * never exceeds its rate limit. The translation is then scaled (not clipped per axis) to |v| ≤ v_max.
 *
 * ★ HOW FAST = HOW LIKELY THE BODY FITS. P_fit = Φ((w/2 − body_half_width − |e|) / σ_lat), σ_lat² = aᵀ P_xy a: the
 *   variance of the opening's position along the wall, from the anchor's own covariance. Off-centre or unsure ⇒
 *   slow (the lateral loop keeps working, so it re-centres first); centred and sure ⇒ full speed. No gate.
 *
 * `through` = s ≥ standoff: the opening is `standoff` behind along the normal. FLAGGED decision level: it ends the
 * action (the robot has nothing left to do here), not a state of belief.
 *
 * Header-only, pure Eigen + std (relative_pose.h for the type).
 */

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "../../common/interaction_anchor/relative_pose.h"

namespace rc
{

struct AnchorCommand { float adv = 0, side = 0, rot = 0; bool through = false; };

struct AnchorCrossGeometry
{
    float e = 0.f;              // signed along-wall offset of the robot from the opening's normal line (m)
    float s = 0.f;              // progress along the normal: < 0 before the opening (m)
    float heading_err = 0.f;    // angle from robot forward to the normal, CCW positive (rad)
    float sigma_lat = 0.f;      // sqrt(aᵀ P_xy a): the opening's along-wall position sigma (m)
    Eigen::Vector2f a{1.f, 0.f}, n{0.f, 1.f};   // wall direction and normal, robot frame
};

[[nodiscard]] inline AnchorCrossGeometry anchor_cross_geometry(const rc::interaction::RelPose& r)
{
    AnchorCrossGeometry g;
    const float psi = rc::interaction::wrap_pi(r.x.z());
    g.a = {std::cos(psi), std::sin(psi)};
    g.n = {-std::sin(psi), std::cos(psi)};
    const Eigen::Vector2f A = r.x.head<2>();
    g.e = -g.a.dot(A);
    g.s = -g.n.dot(A);
    g.heading_err = psi;
    g.sigma_lat = std::sqrt(std::max(0.f, g.a.dot(r.P.topLeftCorner<2, 2>() * g.a)));
    return g;
}

// Φ, the standard normal CDF.
[[nodiscard]] inline float anchor_std_normal_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

// P(the body fits through the opening at the current lateral offset). σ_lat = 0 (a certain opening) is the limit
// of the expression: a step at slack 0.
[[nodiscard]] inline float anchor_p_fit(float width, float body_half_width, float e, float sigma_lat)
{
    const float slack = 0.5f * width - body_half_width - std::abs(e);
    if (sigma_lat <= 0.f) return slack > 0.f ? 1.f : (slack < 0.f ? 0.f : 0.5f);
    return anchor_std_normal_cdf(slack / sigma_lat);
}

// Loop time constants (s). Control-loop gains, not decision thresholds.
inline constexpr float kAnchorTauLat = 1.0f;   // lateral re-centring
inline constexpr float kAnchorTauS   = 1.0f;   // approach to the standoff
inline constexpr float kAnchorTauRot = 0.5f;   // heading alignment

// side nulls the lateral offset e to the opening's normal line, rot aligns the heading with it, adv scales with
// P(body fits) = Φ((w/2 − body_half_width − |e|)/σ_lat). `through` once the opening is `standoff` behind along it.
// Signs: adv = forward (+y) m/s, side = lateral (+x) m/s, rot = rad/s CCW positive (the value handed to the base).
[[nodiscard]] inline AnchorCommand anchor_cross_command(const rc::interaction::RelPose& r, float width,
                                                        float body_half_width, float standoff, float v_max,
                                                        float w_max)
{
    AnchorCommand c;
    if (not r.x.allFinite() or not r.P.allFinite() or not (v_max > 0.f) or not (w_max > 0.f)) return c;
    const auto g = anchor_cross_geometry(r);
    if (g.s >= standoff) { c.through = true; return c; }

    const float p_fit = anchor_p_fit(width, body_half_width, g.e, g.sigma_lat);
    const float v_a = -v_max * std::tanh(g.e / (v_max * kAnchorTauLat));
    const float v_n = v_max * p_fit * std::tanh((standoff - g.s) / (v_max * kAnchorTauS));
    Eigen::Vector2f v = v_n * g.n + v_a * g.a;
    if (const float norm = v.norm(); norm > v_max) v *= v_max / norm;
    c.side = v.x();
    c.adv = v.y();
    c.rot = w_max * std::tanh(g.heading_err / (w_max * kAnchorTauRot));
    return c;
}

}  // namespace rc
