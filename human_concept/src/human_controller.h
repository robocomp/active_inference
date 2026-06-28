/*
 * human_controller.h
 *
 * Output-side motion model for human_concept. The estimator produces the per-frame target joint
 * angles theta* (a clean Bayesian fit with anatomical priors); this drives the PUBLISHED command
 * angles theta_cmd toward theta* under per-DOF velocity + acceleration limits, so the rendered model
 * moves naturally and one-frame YOLO glitches are filtered out (the body can't teleport). Temporal
 * plausibility lives HERE, not inside the fit.
 */

#pragma once

#include <cmath>
#include <utility>

#include "human_kinematic_model.h"   // rc::human::Vec11, DOF

namespace rc::human
{

struct RateLimits
{
    float omega_max;   // max angular speed     (rad/s)   — angle DOFs
    float alpha_max;   // max angular accel     (rad/s^2)
    float vlin_max;    // max linear speed      (m/s)     — lower-body translation DOFs (lb_x, lb_z)
    float alin_max;    // max linear accel      (m/s^2)
};

// Velocity- and acceleration-limited per-DOF tracker. Trapezoidal "decel-to-target" profile: the
// commanded speed is capped both by v_max and by sqrt(2·a_max·|error|) so it brakes to land on the
// target without overshoot, and the per-step speed change is clamped to a_max·dt. Advances cmd/vel
// in place; returns {#DOFs at the speed limit, #DOFs at the accel limit} for diagnostics.
inline std::pair<int, int> track_angles(Vec11& cmd, Vec11& vel, const Vec11& target,
                                        float dt, const RateLimits& lim)
{
    if (dt <= 0.f)
        return {0, 0};
    int vsat = 0, asat = 0;
    for (int k = 0; k < DOF; ++k)
    {
        const bool trans = (k == 8 or k == 9);   // lb_x, lb_z are metric; the rest are angles
        const float vmax = trans ? lim.vlin_max : lim.omega_max;
        const float amax = trans ? lim.alin_max : lim.alpha_max;

        const float e = target(k) - cmd(k);
        const float s = (e > 0.f) ? 1.f : ((e < 0.f) ? -1.f : 0.f);
        const float v_brake = std::sqrt(2.f * amax * std::abs(e));   // speed that still stops at target
        float v_des = s * std::min(vmax, v_brake);
        if (std::abs(v_des) >= vmax - 1e-6f)
            ++vsat;

        float dv = v_des - vel(k);
        const float dv_max = amax * dt;
        if (std::abs(dv) > dv_max)
        {
            dv = (dv > 0.f) ? dv_max : -dv_max;
            ++asat;
        }
        vel(k) += dv;

        // Anti-overshoot: dt is coarse (~0.1 s) and a full Euler step vel*dt can blow PAST the target,
        // which then reverses next step → ring. Never step beyond the target: on arrival snap to it
        // and zero the velocity so a static target makes the joint actually STOP.
        const float step = vel(k) * dt;
        if (std::abs(step) >= std::abs(e))
        {
            cmd(k) = target(k);
            vel(k) = 0.f;
        }
        else
            cmd(k) += step;
    }
    return {vsat, asat};
}

}  // namespace rc::human
