#pragma once
/*
 * door_approach_law.h — THE DOOR APPROACH AS A STATE-FEEDBACK LAW (2026-11-02; user: "IT IS SIMPLE, make the robot face the
 * door first. We have to fix the door_approach to make it thermostat robust." / "You are hitting the corners again").
 *
 * warm29-31: the approach followed a CURVED 4.6 m route and turned where the route ended — beside the walls, 50-63 deg off
 * the door's normal, the turn refused by live returns, "MAKING ROOM" nudged 5 cm for minutes. The law below has NO latched
 * sub-steps: every cycle it reads the CURRENT pose and returns the command, so a push, an overshoot or a slip is corrected by
 * the same law on the next cycle.
 *
 *   V        the vantage (on the door's normal at the standoff, the room side the robot is on)
 *   A        the aim: the route's carrot, `lookahead` of arc ahead of the robot, never past V. The route's last segment is the
 *            door's normal (door_route.h lead point), so A reaches V along the normal and the robot ARRIVES facing the door.
 *   d        |V - p|            w = e / (e + rho*yaw_tol), e = max(0, d - rho)  (the aim until the arrival disc, the door inside it)            rho = the arrival disc (goal tolerance)
 *   h        w * u(A - p) + (1 - w) * u(c - p)     the direction to face: the aim while far, the door's centre c at V.
 *            One continuous blend — no "far / near" switch.
 *   psi      heading error to h (CCW positive, forward = theta + pi/2)
 *   rot      = w_max * tanh(psi / (w_max * tau))                                   (first-order loop, tau = kAnchorTauS)
 *   adv      = v_max * tanh(d_route / (v_max * tau)) * max(0, cos psi_A) * exp(-(|A-p| sin psi_A)^2 / (2 rho^2))
 *            The last factor is the PRECISION on the heading: translating at heading error psi misses the aim by
 *            |A-p| sin psi; the robot translates in proportion to the probability that miss is within the arrival disc.
 *            Far from the aim that is a narrow cone, so it FACES FIRST; near it (and at V) it is wide. No cos^k, no gate.
 *   BACK OFF when the live returns refuse the turn (the sweep's cap w_cap < |rot|): the robot reverses at v_max * (1 - w_cap /
 *            |rot|) — the refused share of the turn — steering onto the reverse of the poses it arrived on (the trail: free
 *            by construction) until the turn fits. Replaces the 5 cm "MAKING ROOM" nudge for this drive.
 *   DONE     d <= rho and |psi_door| <= align tolerance: in the band AND facing — what `open` needs.
 *
 * ⚑ CONSTANTS KEPT: tau = kAnchorTauS (1 s, the servo's own approach time constant), rho = path_controller.goal_threshold()
 * (the existing arrival tolerance), lookahead = route_smoothing_m (the route's own bend scale), done tolerance =
 * align_yaw_tol_rad. No new constant.
 *
 * Pure Eigen/STL (tests/test_door_approach_law.cpp).
 */

#include <algorithm>
#include <cmath>
#include <span>
#include <string_view>

#include <Eigen/Dense>

namespace rc::door_approach
{

[[nodiscard]] inline float wrap_pi(float a)
{
    while (a > static_cast<float>(M_PI)) a -= 2.f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.f * static_cast<float>(M_PI);
    return a;
}

/// Heading error (CCW positive) to face room point q from pose (p, theta); the body's forward is theta + pi/2.
[[nodiscard]] inline float heading_error(const Eigen::Vector2f& p, float theta, const Eigen::Vector2f& q)
{
    const Eigen::Vector2f d = q - p;
    return wrap_pi(std::atan2(d.y(), d.x()) - (theta + static_cast<float>(M_PI_2)));
}

enum class Mode { Rotate, Drive, BackOff, Hold };
[[nodiscard]] inline std::string_view mode_name(Mode m)
{
    switch (m) { case Mode::Rotate: return "rotate"; case Mode::Drive: return "drive"; case Mode::BackOff: return "back-off"; default: return "hold"; }
}

struct State
{
    Eigen::Vector2f p = Eigen::Vector2f::Zero();     ///< robot, room frame
    float theta = 0.f;                               ///< body angle (forward = theta + pi/2)
    Eigen::Vector2f vantage = Eigen::Vector2f::Zero();
    Eigen::Vector2f aim = Eigen::Vector2f::Zero();   ///< the route's carrot (never past V)
    Eigen::Vector2f door = Eigen::Vector2f::Zero();  ///< the aperture centre
    float d_route = 0.f;                             ///< arc length left to V
    float rho = 0.25f;                               ///< arrival disc
    float v_max = 0.5f, w_max = 0.7f, tau = 1.f;
    float w_cap = 1e9f;                              ///< the turn rate the live returns allow in the direction asked
    float yaw_tol = 0.06f;
    std::span<const Eigen::Vector2f> trail;          ///< the poses the robot arrived through, oldest first
};

struct Command
{
    float adv = 0.f, rot = 0.f;   ///< forward m/s, CCW rad/s
    Mode mode = Mode::Hold;
    bool done = false;
    float d = 0.f, psi_V = 0.f, psi_door = 0.f, lateral = 0.f;
};

// There is NO arrival verdict here. The approach leg ENDS when the NEXT affordance (`open` while the door reads shut, `cross` once
// open) is eligible by door_concept's own measure -- DoorPassage::next_eligible(leg key), per installed route (warm32 leg 3: a
// verdict carried from the previous door). `Command::done` below is the law's own trace label only.

[[nodiscard]] inline Command law(const State& s)
{
    Command c;
    const Eigen::Vector2f to_v = s.vantage - s.p;
    c.d = to_v.norm();
    c.psi_V = c.d > 1e-4f ? heading_error(s.p, s.theta, s.vantage) : 0.f;
    c.psi_door = heading_error(s.p, s.theta, s.door);
    const Eigen::Vector2f to_door = s.door - s.vantage;   // the normal, pointing into the door
    if (to_door.norm() > 1e-4f)
    {
        const Eigen::Vector2f t(-to_door.y(), to_door.x());
        c.lateral = (s.p - s.vantage).dot(t.normalized());
    }
    c.done = c.d <= s.rho and std::abs(c.psi_door) <= s.yaw_tol;

    auto unit = [](Eigen::Vector2f v, const Eigen::Vector2f& fallback) { return v.norm() > 1e-4f ? Eigen::Vector2f(v.normalized()) : fallback; };
    const Eigen::Vector2f u_door = unit(s.door - s.p, Eigen::Vector2f::Zero());
    const Eigen::Vector2f u_aim = unit(s.aim - s.p, u_door);
    const float excess = std::max(0.f, c.d - s.rho);   // how far OUTSIDE the arrival disc
    const float band = s.rho * s.yaw_tol;   // the lateral miss one yaw tolerance subtends at the disc's radius (1.5 cm)
    const float w = excess / (excess + band);
    Eigen::Vector2f h = w * u_aim + (1.f - w) * u_door;
    if (h.norm() < 1e-3f) h = u_door;
    const float psi = std::atan2(h.y(), h.x()) - (s.theta + static_cast<float>(M_PI_2));
    const float psi_h = wrap_pi(psi);

    const float rot = s.w_max * std::tanh(psi_h / (s.w_max * s.tau));
    // Translation is toward the AIM (never toward the door: that would walk the robot off V), and is judged on the heading
    // error to the aim, not to the blended facing direction.
    const float d_aim = (s.aim - s.p).norm();
    const float psi_a = d_aim > 1e-4f ? heading_error(s.p, s.theta, s.aim) : 0.f;
    const float miss = d_aim * std::sin(psi_a);
    const float precision = std::exp(-0.5f * miss * miss / (s.rho * s.rho));
    const float adv = s.v_max * std::tanh(s.d_route / (s.v_max * s.tau)) * std::max(0.f, std::cos(psi_a)) * precision;

    if (std::abs(rot) > s.w_cap)   // the turn is refused: give it room by retracing the way in
    {
        const float share = 1.f - s.w_cap / std::abs(rot);
        c.adv = -s.v_max * share;
        c.rot = std::copysign(s.w_cap, rot);
        c.mode = Mode::BackOff;
        // steer the reverse onto the trail: the newest pose at least rho behind
        for (auto it = s.trail.rbegin(); it != s.trail.rend(); ++it)
            if ((*it - s.p).norm() >= s.rho)
            {
                const float back = wrap_pi(std::atan2(it->y() - s.p.y(), it->x() - s.p.x()) - (s.theta + static_cast<float>(M_PI_2) + static_cast<float>(M_PI)));
                c.rot = std::clamp(s.w_max * std::tanh(back / (s.w_max * s.tau)), -s.w_cap, s.w_cap);
                break;
            }
        return c;
    }
    c.adv = adv;
    c.rot = rot;
    c.mode = (adv < 0.1f * s.v_max and std::abs(rot) > 1e-3f) ? Mode::Rotate : (adv > 0.f ? Mode::Drive : Mode::Hold);   // trace label only
    return c;
}

}   // namespace rc::door_approach
