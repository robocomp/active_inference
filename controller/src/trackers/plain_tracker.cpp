#include "plain_tracker.h"

#include "../route_spline.h"

#include <algorithm>
#include <cmath>

namespace rc
{
namespace
{
float wrap_pi(float a) { return std::remainder(a, 2.f * static_cast<float>(M_PI)); }

// Reference speed at which the FEEDBACK's rotation authority is evaluated when the robot is crawling,
// so that authority never reaches zero at the end-of-route taper.
constexpr float kSteerFloorMps = 0.15f;
}   // namespace

// PLAIN TRACKER — curvature feedforward + critically damped Frenet feedback. Follows the route and does
// nothing else: no obstacle data of any kind reaches this file, which is why it takes a PathWorld and
// not a FieldWorld. Safety belongs to the planner's footprint predicate, to the band, and to
// detect_path_blockage -> replan, all of which live outside it.
//
//   omega = g_dc * [ v*kappa_bar(s + v*T_lag)  -  v_steer*( (2/L)e_psi + (1/L^2)e_y ) ]
//
// The feedforward is the kinematic relation omega = v*kappa, previewed by the identified actuator lag.
// It is the only term that reads the route, so the feedback sees a zero-mean residual and cannot
// double-count the nominal turn. The feedback is critically damped in the ARC-LENGTH domain, so an error
// decays over L metres at any speed and no gain is right at one speed while ringing at another.
ControlOutput& PlainTracker::compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p)
{
    const RouteSpline* spline = world_.route_spline();
    if (spline == nullptr or not spline->valid())
    {
        out.adv = 0.f; out.side = 0.f; out.rot = 0.f;
        return out;
    }
    const RouteSpline& sp = *spline;

    const Eigen::Vector2f pos{in.robot_pose.translation().x(), in.robot_pose.translation().y()};
    const float theta_pose = std::atan2(in.robot_pose.linear()(1, 0), in.robot_pose.linear()(0, 0));
    const float theta_fwd = wrap_pi(theta_pose + static_cast<float>(M_PI_2));   // FRAME conversion (1/2)

    // Monotone-forward projection: the tour crosses itself, so a global nearest-point search would snap
    // onto a branch driven ten metres ago.
    // ★reacquire_ is set by reset(), which TrajectoryController::stop() calls. A monotone projection is
    // only meaningful within ONE traversal — across a mission stop, or a repaired curve, it is stale,
    // and stale arc length pointed the tracker at the route's END POINT after a restart (route_s pinned
    // at 0.20 m, robot driving into a wall). Re-acquiring searches the whole route once and adopts the
    // nearest point: the route start after a stop, the robot's own position after a repair.
    if (reacquire_) { s_hint_ = sp.project(pos, 0.f, sp.length()); reacquire_ = false; }
    else            { s_hint_ = sp.project(pos, s_hint_, 2.0f); }
    const float s = s_hint_;

    const Eigen::Vector2f r = sp.position_at(s);
    const float psi = sp.heading_at(s);
    // Cross-track, LEFT POSITIVE — the session's convention, deliberately NOT compute_pd's opposite one.
    const float e_y   = -std::sin(psi) * (pos.x() - r.x()) + std::cos(psi) * (pos.y() - r.y());
    const float e_psi = wrap_pi(theta_fwd - psi);

    // Speed: the route's own geometric profile (curvature and lateral-acceleration limits are already
    // folded into max_adv by route_speed_limit) plus a stop taper on the remaining arc. No field query.
    // ★sqrt(2*a*d) reaches zero exactly at the route end. It was sqrt(0.15^2 + 2*a*d), and that constant
    // under the root is a FLOOR: at zero remaining arc it still commanded 0.15 m/s, so the robot crept
    // past the end of its own route — measured, route_s pinned at 38.42 for the last ten cycles with
    // cmd_adv at 0.146 — and drifted out of RouteFollower::finished()'s 1 m end-reach check.
    const float a_dec = std::max(0.1f, p.cbf_max_decel);
    const float s_remaining = std::max(0.f, sp.length() - s);
    const float v_stop = std::sqrt(2.f * a_dec * s_remaining);
    const float v_cmd = std::clamp(std::min(p.max_adv, v_stop), 0.f, p.max_adv);

    // kappa_avg is CENTRED, so this preview is the one and only lookahead in the law.
    const float k_ff = sp.kappa_avg(s + v_cmd * p.plain_T_lag, p.plain_W);
    const float L = std::max(0.05f, p.plain_L);
    const float v_steer = std::max(v_cmd, kSteerFloorMps);
    const float omega_ccw = std::clamp(
        p.plain_g_dc * (v_cmd * k_ff + v_steer * (-(2.f / L) * e_psi - (1.f / (L * L)) * e_y)),
        -p.max_rot, p.max_rot);

    out.adv  = v_cmd;
    out.side = 0.f;
    out.rot  = -omega_ccw;                        // FRAME conversion (2/2)
    out.cross_track_m = e_y;
    out.carrot_bearing_rad = e_psi;               // the heading error, not a bearing
    out.carrot_dist_m = v_cmd * p.plain_T_lag;    // the preview distance, for the overlay
    out.dist_to_goal = s_remaining;
    out.gate_speed_scale = p.max_adv > 1e-3f ? v_cmd / p.max_adv : 1.f;
    out.safety_guard_triggered = false;           // nothing in this mode can trigger; keep it honest
    return out;
}

}   // namespace rc
