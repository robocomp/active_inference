#include "pd_tracker.h"

#include <algorithm>
#include <cmath>

namespace rc
{
namespace
{
// Same transform as TrajectoryController::room_to_robot: the controller's heading is measured from +y
// toward +x (clockwise), which is why this is not a plain inverse-rotation.
Eigen::Vector2f room_to_robot(const Eigen::Vector2f& p_room, const Eigen::Affine2f& robot_pose)
{
    return robot_pose.inverse() * p_room;
}
}   // namespace


ControlOutput& PdTracker::compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p)
{
    // Angle to carrot in robot frame (Y+ = forward, X+ = right)
    const float carrot_angle = std::atan2(in.carrot_robot.x(), in.carrot_robot.y());
    const float carrot_dist = in.carrot_robot.norm();

    // ── CROSS-TRACK FEEDBACK (Stanley's second term) ──────────────────────────────────────────────
    // Steering at the carrot ALONE is pure pursuit, and pure pursuit converges to the carrot's
    // DIRECTION, not to the path: on a curve it settles at a standing offset that grows with the SQUARE
    // of the lookahead, i.e. it cuts every corner. That is invisible while the MPPI drives (its lateral
    // term pulls the body back onto the centre) and it is the whole reason the tracker wanders off the
    // optimised route the band worked to produce.
    // The fix is the term pure pursuit lacks: the signed distance to the path itself. Written in
    // Stanley's form (Thrun et al. 2006) — atan(k*e / (v + k_soft)) — because it is BOUNDED (never more
    // than a quarter turn, so a large excursion cannot demand an impossible rate) and SPEED-NORMALISED:
    // the same lateral error justifies a sharper angular correction when moving slowly, which is exactly
    // when there is room to make it. k_soft keeps it finite at a standstill.
    // ★Sign: proj_robot.x > 0 means the PATH lies to the robot's right, so the robot has drifted LEFT
    // and must steer right — the same sign convention as carrot_angle, so the two simply add.
    float cross_track = 0.f;
    float cross_term = 0.f;
    if (in.path_proj_valid and p.pd_cross_track_gain > 0.f)
    {
        const Eigen::Vector2f proj_robot = room_to_robot(in.path_proj_room, in.robot_pose);
        cross_track = proj_robot.x();
        const float v_ref = std::max(p.pd_cross_track_soft_mps,
                                     has_prev_vel_ ? std::abs(smoothed_vel_[0]) : 0.f);
        cross_term = std::atan2(p.pd_cross_track_gain * cross_track, v_ref);
    }
    out.cross_track_m = cross_track;

    // ── LATERAL BUMPER ────────────────────────────────────────────────────────────────────────────
    // Probe the LIVE field either side of the body and push away from whichever side is tight. See
    // Params::pd_bumper_gain. Gaps are measured from the BODY: query_esdf is distance from the probe
    // point to the nearest obstacle, and the probe already sits at the body's lateral extent, so what
    // is left is the gap the body actually has. Probed slightly AHEAD (half the carrot's reach or a
    // body length, whichever is smaller) because steering now affects where the body will be, not
    // where it is.
    float bumper_term = 0.f;
    if (p.pd_bumper_gain > 0.f and p.pd_bumper_dist_m > 1e-3f)
    {
        const float probe = std::max(0.05f, p.lateral_probe_offset);
        const float ahead = std::min(0.5f * std::max(0.f, carrot_dist), path_.body_extent_max());
        const float gap_l = world_.esdf_at(-probe, ahead);
        const float gap_r = world_.esdf_at(+probe, ahead);
        const float d_b = p.pd_bumper_dist_m;
        const float def_l = std::max(0.f, d_b - gap_l);
        const float def_r = std::max(0.f, d_b - gap_r);
        const float push = std::clamp((def_l - def_r) / d_b, -1.f, 1.f);   // + ⇒ steer right
        out.pd_bumper_push = push;
        out.pd_gap_left_m = gap_l;
        out.pd_gap_right_m = gap_r;
        const float v_ref = std::max(p.pd_cross_track_soft_mps,
                                     has_prev_vel_ ? std::abs(smoothed_vel_[0]) : 0.f);
        bumper_term = std::atan2(p.pd_bumper_gain * push, v_ref / std::max(0.05f, d_b));
    }

    // One angular error, then one PD on it — the cross-track and bumper terms are both ANGLES, so they
    // belong inside the error the controller regulates, not bolted onto the rate afterwards.
    const float angle_err = carrot_angle + cross_term + bumper_term;
    const float d_err = angle_err - prev_angle_err_;
    prev_angle_err_ = angle_err;

    float cmd_rot = p.pd_Kp_rot * angle_err
                  + p.pd_Kd_rot * d_err;
    cmd_rot = std::clamp(cmd_rot, -p.max_rot, p.max_rot);

    // Forward speed: proportional to alignment, reduced near goal
    // Speed shaping stays on the CARROT bearing, deliberately not on the corrected error: it asks "how
    // aligned am I with where I am going", which is a statement about the route ahead. Folding the
    // cross-track correction in here would make the robot slow down BECAUSE it is correcting, which is
    // the opposite of useful — and it would couple a steering fix to the speed profile, so a lap could
    // not say which of the two changed the result.
    const float alignment = std::pow(std::max(0.f, std::cos(carrot_angle)),
                                     p.pd_speed_cos_power);
    float dist_factor = std::min(1.f, carrot_dist / std::max(p.nominal_goal_dist_scale, 0.1f));
    float cmd_adv = p.max_adv * alignment * dist_factor;
    cmd_adv = std::clamp(cmd_adv, p.min_adv_cmd, p.max_adv);

    // Smoothing + Gaussian brake (same as MPPI path)
    Eigen::Vector3f raw(cmd_adv, 0.f, cmd_rot);
    if (has_prev_vel_)
        smoothed_vel_ = p.velocity_smoothing * smoothed_vel_
                      + (1.f - p.velocity_smoothing) * raw;
    else { smoothed_vel_ = raw; has_prev_vel_ = true; }

    const float rot_ratio = smoothed_vel_[2] / p.max_rot;
    const float brake = std::exp(-p.gauss_k * rot_ratio * rot_ratio);

    out.adv  = smoothed_vel_[0] * brake;
    out.side = smoothed_vel_[1];
    out.rot  = smoothed_vel_[2];

    // Safety gate (same ESDF-based forward prediction as MPPI mode)
    {
        // ── HOW FAR AHEAD THE GATE MUST LOOK ──
        // In MPPI mode this gate is a backstop behind a controller that scores its own rollouts. In PD
        // mode it is the LAST line of defence, so a fixed 0.30 s is not defensible: at 0.376 m/s that is
        // 0.11 m of lookahead, less than the body's own reach, and it can only report a collision the
        // robot can no longer avoid. The horizon is therefore the time to STOP plus one control period
        // of reaction — derived from the braking model already used by the CBF, not chosen.
        // ★DIRECTIONAL BY CONSTRUCTION, and that matters: an earlier attempt inflated the OMNIDIRECTIONAL
        // field by the stopping distance and made the robot crawl, because it demanded braking room from
        // walls the robot drives PARALLEL to. eval_risk integrates along the COMMANDED ARC, so extending
        // its horizon asks only for room along the path actually being taken.
        const float gate_a_dec = std::max(0.1f, p.cbf_max_decel);
        const float gate_horizon_s = std::clamp(std::max(0.f, out.adv) / gate_a_dec + 0.15f, 0.30f, 1.50f);
        constexpr float gate_inflate_m = 0.03f;
        constexpr float gate_hard_margin_m = 0.01f;
        constexpr int gate_soft_consecutive_needed = 5;
        const float gate_dt = std::max(0.03f, p.trajectory_dt);

        auto eval_risk = [&](float adv, float rot, float horizon_s)
        {
            struct R { bool trigger = false; bool hard_collision = false; float min_esdf = 1e9f; };
            R r;
            float x = 0.f, y = 0.f, theta = 0.f;
            const int steps = std::max(1, static_cast<int>(std::ceil(horizon_s / gate_dt)));
            // Thresholds are per-STEP, below: the body's reach depends on the heading the rollout has turned
            // to and on where the obstacle is, and this gate exists precisely to catch tight passages —
            // exactly where a heading-blind disc is most wrong.
            int soft_consecutive = 0;
            for (int i = 0; i < steps; ++i)
            {
                x += adv * std::sin(theta) * gate_dt;
                y += adv * std::cos(theta) * gate_dt;
                theta += rot * gate_dt;
                const float d = world_.esdf_at(x, y);
                r.min_esdf = std::min(r.min_esdf, d);
                const float body_r = world_.body_extent_toward_obstacle(x, y, theta);
                if (d < body_r + gate_hard_margin_m) { r.trigger = true; r.hard_collision = true; return r; }
                if (d < body_r + gate_inflate_m) { if (++soft_consecutive >= gate_soft_consecutive_needed) { r.trigger = true; return r; } }
                else soft_consecutive = 0;
            }
            return r;
        };

        auto risk = eval_risk(out.adv, out.rot, gate_horizon_s);
        // Recorded whether or not it triggers: a gate that never fires is evidence too, and it is the
        // only way to tell "the route was clear" from "the gate was not looking".
        out.gate_horizon_s = gate_horizon_s;
        out.gate_min_esdf = risk.min_esdf;
        out.gate_hard_collision = risk.hard_collision;
        if (risk.trigger)
        {
            out.safety_guard_triggered = true;
            // ── LARGEST ADMISSIBLE SPEED, not the first of three guesses ──
            // {0.5, 0.25, 0} quantises the response: a situation needing 0.9 gets 0.5, and one needing
            // 0.45 gets 0.25. In MPPI mode that coarseness is hidden because the sampler is already
            // slowing down via its obstacle cost; in PD mode this IS the speed control near obstacles,
            // and quantising it is what "crawl or slam" feels like. Six bisections resolve the scale to
            // ~1.6% using the SAME predicate, so the gate stays exactly as conservative as it was.
            if (not eval_risk(0.f, out.rot, gate_horizon_s).trigger)
            {
                float lo = 0.f, hi = 1.f;      // lo verified safe, hi known to trigger
                for (int i = 0; i < 6; ++i)
                {
                    const float mid = 0.5f * (lo + hi);
                    if (eval_risk(out.adv * mid, out.rot, gate_horizon_s).trigger) hi = mid; else lo = mid;
                }
                out.adv *= lo;
                out.gate_speed_scale = lo;
                return out;
            }
            out.gate_hard_stop = true;      // even a full stop was unsafe; rotating away below
            out.gate_speed_scale = 0.f;
            // Full stop + rotate away from closest obstacle
            out.adv = 0.f;
            float sign = (out.rot >= 0.f) ? 1.f : -1.f;
            out.rot = sign * 0.5f * p.max_rot;
        }
    }

    // No per-cycle print: this path populates the same ControlOutput the session logs to mppi_diag.csv
    // every cycle, so a terminal line would be a second, worse copy of a record that already exists —
    // and one nobody can compare across runs.
    return out;
}

// ============================================================================
// Nominal control: simple proportional control toward carrot
// Used as the base for MPPI warm-start blending
// ============================================================================

}   // namespace rc
