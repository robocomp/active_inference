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
 * DESIRED VELOCITY (robot frame), the same for both bases:  d = v_n·n + v_a·a
 *   v_a = −v_max·tanh(e / (v_max·τ_lat))                  nulls e (moves the robot back onto the normal line)
 *   v_n =  v_max·P_fit·tanh((standoff − s)/(v_max·τ_s))   advances, tapering to 0 at the standoff
 * tanh is a smooth saturation with unit slope at 0, so each term is a first-order loop with time constant τ that
 * never exceeds its rate limit. |d| is then scaled (not clipped per axis) to ≤ v_max.
 *
 * DIFFERENTIAL BASE (Shadow: the bridge drops `side`, rc::BaseCapability::holonomic = false). The base can only
 * move along its forward axis, so it TURNS TOWARD d and advances by d's forward projection:
 *   ε   = atan2(−d.x, d.y)                 angle from forward (+y) to d, CCW positive (ε = 0: d straight ahead)
 *   rot = w_max·tanh(ε / (w_max·τ_rot)),  adv = max(0, |d|·cos ε),  side = 0
 * As e → 0, d → n, so the heading converges to the wall normal on its own; with nothing to do (d = 0) it aligns
 * with the normal (ε = ψ).
 *
 * HOLONOMIC BASE. side = d.x, adv = d.y, rot = w_max·tanh(ψ/(w_max·τ_rot)).
 *   ⚠ THE SIGN OF +side ON A REAL BASE IS UNVERIFIED. This law takes +side = body +x (right), as the robot-frame
 *   convention says; trajectory_controller.h (clearance_at's comment) says the OmniRobot command frame has
 *   +side = LEFT, while webots-bridge odometry reports side = −velocity_local(0), i.e. body +x. No controller path
 *   has ever commanded a non-zero side. Verify on the first holonomic run (base_commands.jsonl `side` vs
 *   robot_current_side_speed) before trusting this branch.
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

// Loop time constants (s). Control-loop gains, not decision thresholds. Chosen on the differential base, where the
// heading must turn toward d and back while e closes: the lateral loop must be slow against the rate-limited turn
// (w_max = 1 rad/s) or the robot arrives on the line still pointing across it and overshoots. On a kinematic
// unicycle from e = 0.3 m (tests/test_anchor_servo.cpp): τ_lat 1.0 / τ_rot 0.5 overshoots 7 cm; 2.0 / 0.2
// overshoots 2 mm and is centred (< 5 cm) 1.8 m before the opening.
inline constexpr float kAnchorTauLat = 2.0f;   // lateral re-centring
inline constexpr float kAnchorTauS   = 1.0f;   // approach to the standoff
inline constexpr float kAnchorTauRot = 0.2f;   // heading alignment

// d nulls the lateral offset e to the opening's normal line and advances along it at a speed scaled by
// P(body fits) = Φ((w/2 − body_half_width − |e|)/σ_lat); `through` once the opening is `standoff` behind along it.
// holonomic: side/adv = d, rot aligns the heading with n. Differential: side = 0, rot turns toward d, adv = d's
// forward projection. Signs: adv = forward (+y) m/s, side = lateral (+x) m/s, rot = rad/s CCW positive (the value
// handed to the base).
[[nodiscard]] inline AnchorCommand anchor_cross_command(const rc::interaction::RelPose& r, float width,
                                                        float body_half_width, float standoff, float v_max,
                                                        float w_max, bool holonomic)
{
    AnchorCommand c;
    if (not r.x.allFinite() or not r.P.allFinite() or not (v_max > 0.f) or not (w_max > 0.f)) return c;
    const auto g = anchor_cross_geometry(r);
    if (g.s >= standoff) { c.through = true; return c; }

    const float p_fit = anchor_p_fit(width, body_half_width, g.e, g.sigma_lat);
    const float v_a = -v_max * std::tanh(g.e / (v_max * kAnchorTauLat));
    const float v_n = v_max * p_fit * std::tanh((standoff - g.s) / (v_max * kAnchorTauS));
    Eigen::Vector2f d = v_n * g.n + v_a * g.a;
    float speed = d.norm();
    if (speed > v_max) { d *= v_max / speed; speed = v_max; }
    if (holonomic)
    {
        c.side = d.x();
        c.adv = d.y();
        c.rot = w_max * std::tanh(g.heading_err / (w_max * kAnchorTauRot));
        return c;
    }
    const float eps = speed > 0.f ? std::atan2(-d.x(), d.y()) : g.heading_err;
    c.side = 0.f;
    c.adv = std::max(0.f, speed * std::cos(eps));
    c.rot = w_max * std::tanh(eps / (w_max * kAnchorTauRot));
    return c;
}

// ── THE JAMBS AS OBSTACLES THE MODEL ALREADY KNOWS ────────────────────────────────────────────────────────────
// The live field cannot stop the robot at a jamb: the self-filter clears every return within ~0.4 m of the body
// centre, and the LiDAR ceiling is floored at 0.05 m/s. The anchor knows where the jambs are: J± = A ± (w/2)·a in
// the robot frame, each uncertain along the wall by σ_lat (the opening's along-wall sigma — the width is carried as
// known), so each is a disc of radius σ_lat. Moving along the unit robot-frame direction u, the body sweeps a
// corridor of half-width support(±p) (p ⟂ u): a jamb inside it is met after travelling along − σ_lat − support(u),
// and the base must be able to stop before that: v ≤ sqrt(2·a_dec·max(0, along − σ_lat − support(u))). A jamb
// behind the body (along < −support(−u)) or beside the corridor limits nothing. No floor: this ceiling may be 0.
// support(dir) = the body's reach along a robot-frame direction (RobotFootprint::support_radius at heading 0).
template <class Support>
[[nodiscard]] inline float anchor_jamb_speed_limit(const rc::interaction::RelPose& r, float width,
                                                   const Eigen::Vector2f& u, const Support& support, float a_dec,
                                                   float v_cap)
{
    if (not r.x.allFinite() or not (u.squaredNorm() > 0.f) or not (width > 0.f)) return v_cap;
    const auto g = anchor_cross_geometry(r);
    const Eigen::Vector2f un = u.normalized();
    const Eigen::Vector2f p(-un.y(), un.x());
    const Eigen::Vector2f A = r.x.head<2>();
    const float rho = g.sigma_lat;
    float v = v_cap;
    for (const float sgn : {-1.f, 1.f})
    {
        const Eigen::Vector2f J = A + sgn * 0.5f * width * g.a;
        const float along = J.dot(un), lat = J.dot(p);
        if (along + rho < -support(Eigen::Vector2f(-un))) continue;                     // behind the body
        const Eigen::Vector2f side_dir = lat >= 0.f ? p : Eigen::Vector2f(-p);
        if (std::abs(lat) - rho >= support(side_dir)) continue;                           // beside the corridor
        const float gap = std::max(0.f, along - rho - support(un));
        v = std::min(v, std::sqrt(2.f * std::max(0.f, a_dec) * gap));
    }
    return v;
}

}  // namespace rc
