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
    // ★An EMPTY s_hint_ means re-acquire; reset() empties it, and TrajectoryController::stop() and
    // set_route (on a curve change) both call reset(). A monotone projection is
    // only meaningful within ONE traversal — across a mission stop, or a repaired curve, it is stale,
    // and stale arc length pointed the tracker at the route's END POINT after a restart (route_s pinned
    // at 0.20 m, robot driving into a wall). Re-acquiring searches the whole route once and adopts the
    // nearest point: the route start after a stop, the robot's own position after a repair.
    s_hint_ = s_hint_ ? sp.project(pos, *s_hint_, std::max(0.05f, p.plain_proj_window))
                      : sp.project(pos, 0.f, sp.length());
    const float s = *s_hint_;

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
    // ── ⚠OPEN, 2026-08-09: THIS TRACKER CANNOT STOP ON A POINT TARGET ────────────────────────────
    // Reproduce: tools/tracker_sim <short route> --stop-test. On a 2.5 m hop PLAIN NEVER STOPS: it runs
    // past the end of its own curve until the bench gives up (8 m past, still doing 0.235 m/s, 10.6 m
    // driven of a 2.5 m route, 564 mm rms cross-track). PD stops 0.124 m short having never passed it.
    // ★It is not "overshoots by X" — there is no terminal behaviour in this tracker at all. On a MISSION
    // that is invisible, because RouteFollower::finished() ends the traversal from OUTSIDE; nothing
    // inside the tracker ever brings the robot to rest on the curve's end.
    // On the robot this is "jumps over the target and moves forward until it hits an obstacle", and it
    // is why point targets are NOT wired to PLAIN — see controller_session.cpp's set_route(nullptr).
    // ★THE TAPER BELOW IS NOT THE DEFECT. It is starved: s_remaining never shrinks, so v_stop never
    // falls. Proven by arming a stop guard inside the braking distance v^2/(2a) ~ 0.5 m — it NEVER FIRED
    // while the robot was metres beyond the end.
    // ★AND THE PROJECTION WINDOW IS NOT THE DEFECT EITHER — tried and reverted: bounding the forward
    // search by recent progress (5-cycle history, window = 3x the arc actually covered, so a closed turn
    // cannot offer a nearer point across the chord). Terminal behaviour did not change AT ALL, and the
    // tour got worse: rms 41.2 -> 46.6 mm, TV(v)/m 0.581 -> 1.048, TV(w)/m 2.068 -> 3.177, 80 -> 89 s.
    // Whatever is wrong, it is not the width of the search.
    // ★THREE FIXES TRIED HERE AND ALL REVERTED, each broken by the same fact — A TOUR ENDS WHERE IT
    // BEGAN, so the endpoint sits beside the start:
    //   • bound the taper by the EUCLIDEAN distance to the endpoint -> brakes at the START of a loop.
    //     Measured on the 37 m tour: 88 s -> 400 s, TV(w)/m 2.07 -> 16.34.
    //   • zero the taper once PAST the endpoint's heading plane -> true on cycle one of a loop, same
    //     crawl.
    //   • arm that guard only when s_remaining is inside the braking distance -> tour restored exactly,
    //     but inert in the failing case, because s_remaining is the very thing that is wrong.
    // Anything attempted next must be measured on BOTH routes (loop and short hop) before it ships.
    const float a_dec = std::max(0.1f, p.cbf_max_decel);
    const float s_remaining = std::max(0.f, sp.length() - s);
    const float v_stop = std::sqrt(2.f * a_dec * s_remaining);
    const float v_profile = std::clamp(std::min(p.max_adv, v_stop), 0.f, p.max_adv);

    // kappa_avg is CENTRED, so this preview is the one and only lookahead in the law.
    const float k_ff = sp.kappa_avg(s + v_profile * p.plain_T_lag, p.plain_W);
    const float L = std::max(0.05f, p.plain_L);
    const float k_fb = -(2.f / L) * e_psi - (1.f / (L * L)) * e_y;

    // ── ADVANCE IS BOUND BY THE TURN THE ROBOT CAN ACTUALLY DELIVER ─────────────────────────────
    // omega = g_dc*v*(k_ff + k_fb), so requiring |omega| <= max_rot inverts to a speed limit. Without
    // it omega simply CLIPS at max_rot while v carries on: the robot cannot come round, overshoots,
    // and the error that caused the demand persists.
    // ★MEASURED 20:54, the lap that hit the counter at the hairpin: |cmd_rot| sat at the 0.8 rad/s cap
    // on 1614 of 3039 cycles — 53.1% — with mean cmd_adv 0.677 m/s. It drove 63.9 m of a 37.6 m route
    // (ratio 1.70) with cross-track max 2.24 m. Mean |kappa| at those cycles was only 0.65, so v*kappa
    // = 0.44 and the FEEDFORWARD alone never saturates: it is the FEEDBACK, (1/L^2)*e_y = 2.78 per
    // metre, so 0.3 m off-route demands 0.83 rad/s by itself. Off-route -> saturate -> cannot converge
    // -> stay off-route.
    // ★The rotation is computed FIRST, at the speed the route would allow; then the advance is scaled by
    // exactly the factor the rotation had to be clipped by. That factor preserves the commanded ARC:
    // kappa = omega/v is unchanged, so a saturated turn becomes a slower turn of the SAME shape rather
    // than a wider one. Clipping omega alone silently changes the arc, which is the failure above.
    // ★No new parameter: max_rot and g_dc already exist, and this is omega = v*kappa read backwards.
    const float omega_want = p.plain_g_dc * (v_profile * k_ff
                                             + std::max(v_profile, kSteerFloorMps) * k_fb);
    const float omega_ccw = std::clamp(omega_want, -p.max_rot, p.max_rot);
    const float scale = std::abs(omega_want) > 1e-6f
                      ? std::abs(omega_ccw) / std::abs(omega_want) : 1.f;
    // ★EXPONENTIAL BRAKE on the turn rate, times the saturation ratio's job — BOTH, not either:
    //   brake — "slow down as the turn demand grows". Acts BEFORE saturation, which is the approach to
    //           every hairpin: at omega just under max_rot the ratio is still 1 and the robot would
    //           otherwise drive flat out through the turn.
    //   scale — "do not demand a turn the robot cannot deliver". Acts only ONCE omega saturates,
    //           decaying as 1/demand.
    // ★THE EXPONENT USES THE DELIVERABLE RATE (omega_ccw, clamped), NOT THE DEMAND. With omega_want
    // unbounded the brake goes to ZERO and that DEADLOCKS: e_y can only be reduced by MOVING, so a
    // robot at v = 0 pivots at max rate forever and never approaches the path. Observed driving a
    // single-shot target, whose plan starts off-path: cmd_adv = 7.5e-19, cmd_rot pinned at -0.800,
    // demand ~10 rad/s (at L = 0.50 the feedback alone gives 4 rad/s per metre of cross-track).
    // ⚠CONSEQUENCE: r_om is pinned at 1 past the cap, so the brake is CONSTANT there at exp(-k) —
    // it no longer "keeps falling past saturation", and scale is what acts beyond the cap. That is why
    // both terms are kept. It also means the k = 0.25 optimum was measured on the UNBOUNDED exponent
    // (3 laps, 2026-08-05, before this bound) and has not been re-measured since.
    // At k = 0 this reduces to the saturation ratio alone.
    const float r_om = omega_ccw / std::max(0.05f, p.max_rot);
    const float brake = std::exp(-std::max(0.f, p.plain_brake_k) * r_om * r_om);
    const float v_cmd = std::clamp(v_profile * std::min(scale, brake), 0.f, p.max_adv);

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
