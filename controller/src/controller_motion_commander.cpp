#include <print>

#include "controller_motion_commander.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "controller_world_model.h"

void ControllerMotionCommander::set_params(const ControllerParams *params)
{
    params_ = params;
}

void ControllerMotionCommander::set_dependencies(std::shared_ptr<DSR::DSRGraph> graph,
                                                 const ControllerWorldModel *world_model,
                                                 RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy,
                                                 int agent_id,
                                                 CommandTextSink command_text_sink)
{
    graph_ = std::move(graph);
    world_model_ = world_model;
    omnirobot_proxy_ = std::move(omnirobot_proxy);
    agent_id_ = agent_id;
    command_text_sink_ = std::move(command_text_sink);

    // Start the fixed-rate actuation loop once the proxy exists (see the header for why this is a thread and
    // not a QTimer). Guarded so a second set_dependencies call cannot spawn a second loop.
    if (omnirobot_proxy_ and not output_thread_.joinable())
    {
        const float period_ms = params_ ? std::max(5.f, params_->velocity_output_period_ms) : 50.f;
        output_thread_ = std::jthread([this](std::stop_token s) { output_loop(std::move(s)); });
        // Report whether params_ was available, not just the period. The period is read ONCE at
        // thread start, so a set_params() that lands afterwards silently leaves the built-in 50 ms in
        // force and NO config value can ever take effect — and the old line, printing only the number,
        // could not distinguish "configured 50" from "defaulted to 50 because params_ was null".
        // (The generic handler no longer swallows qInfo — it suppresses qDebug only — so either would
        // reach the terminal; std::println just keeps this off the Qt path entirely.)
        std::println("[vel-out] output loop started at {:.1f} ms  (params {})",
                     period_ms, params_ ? "OK" : "NULL -> using the 50 ms built-in default");
    }
}

float ControllerMotionCommander::ramp_uncertainty_scale(float value,
                                                        float slow_threshold,
                                                        float stop_threshold,
                                                        float min_scale)
{
    if (stop_threshold <= slow_threshold)
        return value <= slow_threshold ? 1.f : std::clamp(min_scale, 0.f, 1.f);

    const float clamped_min_scale = std::clamp(min_scale, 0.f, 1.f);
    const float alpha = std::clamp((value - slow_threshold) / (stop_threshold - slow_threshold), 0.f, 1.f);
    return 1.f - alpha * (1.f - clamped_min_scale);
}

float ControllerMotionCommander::preserve_sign_clamp(float value, float max_abs)
{
    return std::copysign(std::min(std::abs(value), std::max(0.f, max_abs)), value);
}

std::uint64_t ControllerMotionCommander::current_time_ms()
{
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

void ControllerMotionCommander::apply_uncertainty_speed_limit(float &adv_mps, float &side_mps, float &rot_rps) const
{
    // Both early exits mean NO limiting happened. Record that explicitly rather than leaving the previous
    // cycle's numbers standing, or the diagnostic would report a throttle that is not being applied.
    uncertainty_diag_ = UncertaintyDiag{};
    if (!world_model_ || !params_)
        return;

    const auto uncertainty = world_model_->read_pose_uncertainty();
    if (!uncertainty.has_value())
        return;

    const float current_trans_speed_mps = std::hypot(adv_mps, side_mps);
    const float current_rot_speed_rps = std::abs(rot_rps);
    const float forward_ratio = (current_trans_speed_mps > 1e-3f)
        ? std::clamp(std::abs(adv_mps) / current_trans_speed_mps, 0.f, 1.f)
        : 0.f;
    const float lateral_ratio = (current_trans_speed_mps > 1e-3f)
        ? std::clamp(std::abs(side_mps) / current_trans_speed_mps, 0.f, 1.f)
        : 0.f;
    const float turning_ratio = std::clamp(current_rot_speed_rps / std::max(0.12f, 0.35f * params_->max_rot_speed_rps), 0.f, 1.f);
    const float straight_motion_ratio = std::clamp(forward_ratio * (1.f - lateral_ratio) * (1.f - turning_ratio), 0.f, 1.f);

    const float effective_xy_slow_m = params_->pose_xy_std_slow_m * (1.f + 1.0f * straight_motion_ratio);
    const float effective_xy_stop_m = params_->pose_xy_std_stop_m * (1.f + 0.6f * straight_motion_ratio);

    float adv_scale = ramp_uncertainty_scale(uncertainty->xy_std_m,
                                             effective_xy_slow_m,
                                             effective_xy_stop_m,
                                             params_->min_adv_speed_scale);
    float rot_scale = ramp_uncertainty_scale(uncertainty->theta_std_rad,
                                             params_->pose_theta_std_slow_rad,
                                             params_->pose_theta_std_stop_rad,
                                             params_->min_rot_speed_scale);

    const float horizon_s = std::max(1e-3f, params_->uncertainty_prediction_horizon_s);
    const float xy_growth = std::max(1e-3f,
                                     params_->pose_xy_std_growth_per_mps / (1.f + 1.5f * straight_motion_ratio));
    const float theta_growth = std::max(1e-3f, params_->pose_theta_std_growth_per_rps);

    if (current_trans_speed_mps > 1e-3f)
    {
        const float xy_margin = std::max(0.f, effective_xy_stop_m - uncertainty->xy_std_m);
        const float max_predicted_trans_speed_mps = xy_margin / (horizon_s * xy_growth);
        const float predictive_adv_scale = std::clamp(max_predicted_trans_speed_mps / current_trans_speed_mps,
                                                      params_->min_adv_speed_scale,
                                                      1.f);
        adv_scale = std::min(adv_scale, predictive_adv_scale);
    }

    if (current_rot_speed_rps > 1e-3f)
    {
        const float theta_margin = std::max(0.f, params_->pose_theta_std_stop_rad - uncertainty->theta_std_rad);
        const float max_predicted_rot_speed_rps = theta_margin / (horizon_s * theta_growth);
        const float predictive_rot_scale = std::clamp(max_predicted_rot_speed_rps / current_rot_speed_rps,
                                                      params_->min_rot_speed_scale,
                                                      1.f);
        rot_scale = std::min(rot_scale, predictive_rot_scale);
    }

    const float coupled_adv_scale = std::pow(std::max(rot_scale, 0.f),
                                             std::max(0.f, params_->adv_rotation_coupling_exponent) * turning_ratio);
    adv_scale = std::min(adv_scale, coupled_adv_scale);

    // ── HOW HARD LOCALISATION UNCERTAINTY IS ALLOWED TO BITE ─────────────────────────────────────
    // An A/B lever over everything above, applied to the RESULT rather than to any of the knees, so it
    // cannot silently re-tune one term relative to another: 1 = exactly the behaviour without it, 0 =
    // the limiter computes and REPORTS but never restrains, 0.5 = half the restraint. The point is to
    // answer "is the pose covariance what is slowing the robot" by measurement instead of by argument —
    // drive the same route at 1.0 and at 0.0 and compare, which no amount of reading the code can do.
    // ★A DIAGNOSTIC LEVER, NOT A CURE. Turning it down does not make the robot better localised; it
    // makes it drive fast on a pose it has less reason to trust. If a run at 0 is the good one, the fix
    // belongs in whatever inflated sigma — not here.
    // ★The diagnostic below records the SCALE ACTUALLY APPLIED, so a run at 0 reports 1.000 throughout
    // and profile.csv keeps meaning "what the robot obeyed". The raw ramp is still visible as the gap
    // between cmd_adv (mppi_diag) and adv_mps (profile).
    if (params_->pose_uncertainty_coupling < 1.f)
    {
        const float k = std::clamp(params_->pose_uncertainty_coupling, 0.f, 1.f);
        adv_scale = 1.f - k * (1.f - adv_scale);
        rot_scale = 1.f - k * (1.f - rot_scale);
    }

    // Recorded AFTER the rotation coupling, so adv_scale is the multiplier actually applied — the coupling
    // is frequently the binding one while turning, and reporting the pre-coupling ramp would hide that.
    uncertainty_diag_ = UncertaintyDiag{.valid = true,
                                        .xy_std_m = uncertainty->xy_std_m,
                                        .theta_std_rad = uncertainty->theta_std_rad,
                                        .adv_scale = adv_scale,
                                        .rot_scale = rot_scale};

    adv_mps *= adv_scale;
    side_mps *= adv_scale;
    rot_rps = preserve_sign_clamp(rot_rps, current_rot_speed_rps * rot_scale);
}

void ControllerMotionCommander::publish_robot_reference_speed(float adv_mps,
                                                              float side_mps,
                                                              float rot_rps,
                                                              std::uint64_t timestamp_ms)
{
    if (!graph_ || !world_model_ || world_model_->graph_state().robot_id == 0)
        return;

    auto robot_node_opt = graph_->get_node(world_model_->graph_state().robot_id);
    if (!robot_node_opt.has_value())
        return;

    auto robot_node = robot_node_opt.value();
    auto &attrs = robot_node.attrs();
    attrs[kRobotRefAdvSpeedAttr] = DSR::Attribute{adv_mps, timestamp_ms, static_cast<std::uint32_t>(agent_id_)};
    attrs[kRobotRefSideSpeedAttr] = DSR::Attribute{side_mps, timestamp_ms, static_cast<std::uint32_t>(agent_id_)};
    attrs[kRobotRefRotSpeedAttr] = DSR::Attribute{rot_rps, timestamp_ms, static_cast<std::uint32_t>(agent_id_)};
    attrs[kRobotRefSpeedTimestampAttr] = DSR::Attribute{timestamp_ms, timestamp_ms, static_cast<std::uint32_t>(agent_id_)};
    if (!graph_->update_node(robot_node))
        qWarning() << "Controller failed to publish robot reference speed attrs to DSR";
}

void ControllerMotionCommander::send_speed_command(float adv_mps, float side_mps, float rot_rps)
{
    const float adv_mm_s = adv_mps * 1000.f;
    const float side_mm_s = side_mps * 1000.f;

    if (command_text_sink_)
    {
        command_text_sink_(adv_mm_s, side_mm_s, rot_rps);
    }

    if (!omnirobot_proxy_)
        return;

    // ── A SWALLOWED COMMAND MUST NOT BE A SILENT ONE ─────────────────────────────────────────────
    // When the output is disarmed this class does exactly what it should — nothing — but it did it
    // WITHOUT SAYING SO, and a robot that plans, computes a speed and never moves looks like a broken
    // controller from every angle: the cycle runs at 20 Hz, the plan is valid, the tracker is on the
    // path, and the base does not turn a wheel. That cost hours of chasing the planner and the tracker.
    // Said once per disarm, so re-arming makes it sayable again.
    {
        const std::scoped_lock lock(mutex_);
        if (not output_enabled_
            and (std::abs(adv_mps) > 1e-3f or std::abs(side_mps) > 1e-3f or std::abs(rot_rps) > 1e-3f))
        {
            if (not disarmed_notice_given_)
            {
                disarmed_notice_given_ = true;
                std::println("[vel-out] ⚠ BASE OUTPUT IS DISARMED — commands are being computed "
                             "({:.2f} m/s, {:.2f} rad/s) and DISCARDED. Stop disarms it; press Run, or "
                             "click a target, to re-arm.", adv_mps, rot_rps);
            }
        }
    }

    const auto timestamp_ms = current_time_ms();

    // Hand the command to the fixed-rate output thread. The Ice call itself is NOT made here any more: it is
    // what ties the base's command rate to however long this cycle took, and it is a synchronous RPC that can
    // itself block the loop. Timestamped so the output loop can tell how stale it is getting.
    {
        const std::scoped_lock lock(mutex_);
        pending_ = PendingCommand{.adv_mps = adv_mps, .side_mps = side_mps, .rot_rps = rot_rps,
                                  .stamp_ms = timestamp_ms, .stop_requested = false};
    }
    cv_.notify_all();

    // The DSR publish and the dedup stay on THIS (main) thread: update_node erases attributes absent from the
    // submitted copy, so the robot node must keep a single writer. Deduping it also avoids pointless graph
    // churn — but note the dedup no longer gates the BASE command, only the graph mirror of it.
    const Eigen::Vector3f cmd(adv_mps, side_mps, rot_rps);
    if (has_last_speed_command_ && (cmd - last_speed_command_).cwiseAbs().maxCoeff() < 1e-4f)
        return;
    publish_robot_reference_speed(adv_mps, side_mps, rot_rps, timestamp_ms);
    last_speed_command_ = cmd;
    has_last_speed_command_ = true;
    stop_command_latched_ = false;
}

ControllerMotionCommander::~ControllerMotionCommander()
{
    output_thread_.request_stop();
    cv_.notify_all();     // don't make the destructor wait out a full tick
}

void ControllerMotionCommander::set_output_enabled(bool enabled)
{
    const std::scoped_lock lock(mutex_);
    if (output_enabled_ == enabled) return;
    output_enabled_ = enabled;
    disarmed_notice_given_ = false;   // each disarm gets its own notice
    if (not enabled)
    {
        // Stop MEANS stop: zero the pending command and queue a burst of explicit zeros so the base
        // cannot be left holding the last non-zero value if one packet is dropped. After the burst the
        // loop goes quiet and stays quiet until this is re-armed.
        pending_.stop_requested = true;
        quiet_burst_left_ = 5;        // 5 ticks at 20 Hz = 250 ms of "definitely stop"
        applied_adv_ = applied_side_ = applied_rot_ = 0.f;
    }
}

bool ControllerMotionCommander::output_enabled() const
{
    const std::scoped_lock lock(mutex_);
    return output_enabled_;
}

void ControllerMotionCommander::output_loop(std::stop_token stop)
{
    using namespace std::chrono;
    const float period_ms = params_ ? std::max(5.f, params_->velocity_output_period_ms) : 50.f;
    const float tau_ms    = params_ ? std::max(1.f, params_->command_freshness_tau_ms) : 500.f;
    auto next = steady_clock::now();
    auto last_send = steady_clock::time_point{};

    while (not stop.stop_requested())
    {
        next += milliseconds(static_cast<int>(period_ms));
        {
            std::unique_lock lock(mutex_);
            if (cv_.wait_until(lock, stop, next, [&] { return stop.stop_requested(); }))
                break;
        }
        if (stop.stop_requested()) break;
        // A late wake-up must not make the loop chase a backlog of missed ticks.
        if (const auto now = steady_clock::now(); next < now) next = now;

        PendingCommand cmd;
        bool armed = true;
        {
            const std::scoped_lock lock(mutex_);
            cmd = pending_;
            // Disarmed: send the remaining zero burst, then nothing at all. The tick still happens, so
            // the loop's own timing statistics do not develop a hole that looks like a stall.
            if (not output_enabled_)
            {
                if (quiet_burst_left_ > 0) --quiet_burst_left_;
                else armed = false;
            }
        }
        if (not armed) continue;

        float adv = 0.f, side = 0.f, rot = 0.f, scale = 1.f, age_ms = 0.f;
        if (not cmd.stop_requested)
        {
            const auto now_ms = current_time_ms();
            age_ms = now_ms > cmd.stamp_ms ? static_cast<float>(now_ms - cmd.stamp_ms) : 0.f;
            // FRESHNESS AS PRECISION. Re-issuing a stale command forever would drive the robot blind through
            // exactly the multi-second stalls this loop exists to survive — but a hard age cutoff would chop
            // the motion at normal cadence. So the command's authority falls off as 1/(1+(age/τ)²): the same
            // form the occupancy grid uses for range precision. At the healthy ~108 ms cycle that is ≈0.95 (no
            // practical penalty); by 1 s it is 0.2 and by 3 s effectively zero, so a stalled planner coasts
            // smoothly to a halt instead of either lurching or ploughing on.
            const float r = age_ms / tau_ms;
            scale = 1.f / (1.f + r * r);
            adv = cmd.adv_mps * scale; side = cmd.side_mps * scale; rot = cmd.rot_rps * scale;
        }

        // SLEW LIMIT toward the target, at the exact tick dt. Bounds what the base is asked to do to what it
        // can physically achieve; also smooths the freshness scale's own recovery, which is otherwise a step
        // (age resets to ~0 on a fresh command, so the factor would jump straight back to 1 after a stall).
        {
            const float dt_s = period_ms * 1e-3f;
            const auto slew = [dt_s](float current, float target, float accel, float decel)
            {
                // Speeding up (or reversing through zero) is limited by accel; reducing magnitude by decel.
                const float limit = (std::abs(target) >= std::abs(current) ? accel : decel) * dt_s;
                return current + std::clamp(target - current, -limit, limit);
            };
            const float la = params_ ? params_->max_lin_accel_mps2 : 1.5f;
            const float ld = params_ ? params_->max_lin_decel_mps2 : 3.0f;
            const float ra = params_ ? params_->max_rot_accel_rps2 : 4.0f;
            const float rd = params_ ? params_->max_rot_decel_rps2 : 8.0f;
            adv  = slew(applied_adv_,  adv,  la, ld);
            side = slew(applied_side_, side, la, ld);
            rot  = slew(applied_rot_,  rot,  ra, rd);
            applied_adv_ = adv; applied_side_ = side; applied_rot_ = rot;
        }

        // Profile tap. AFTER the limiter, so it records the actuation the world receives — the command
        // the MPPI emitted is already visible per-cycle in the trajectory stats, and it is the difference
        // between the two that shows how much the limiter is absorbing.
        if (profile_sink_)
            profile_sink_(current_time_ms(), adv, side, rot, scale);

        // setSpeedBase is a SYNCHRONOUS Ice RPC to the Webots bridge, so its duration is a hard floor on this
        // loop's period: no amount of scheduling here can beat a bridge that blocks. Time it, so if the output
        // period is still not stable we can tell immediately whether the cause is us or the bridge.
        const auto ice_t0 = steady_clock::now();
        try
        {
            // Sent EVERY tick, unchanged or not — a constant cadence is the whole point, and a base watchdog
            // needs to keep hearing from us.
            omnirobot_proxy_->setSpeedBase(side * 1000.f, adv * 1000.f, rot);
        }
        catch (const Ice::Exception &) { /* transient bridge hiccup: keep the cadence, try again next tick */ }

        const auto now = steady_clock::now();
        const float ice_ms = duration<float, std::milli>(now - ice_t0).count();
        const std::scoped_lock lock(mutex_);
        if (last_send.time_since_epoch().count() != 0)
        {
            const float gap = duration<float, std::milli>(now - last_send).count();
            stat_period_sum_ms_ += gap;
            stat_period_max_ms_ = std::max(stat_period_max_ms_, gap);
            ++stat_ticks_;
        }
        last_send = now;
        stat_cmd_age_max_ms_ = std::max(stat_cmd_age_max_ms_, age_ms);
        stat_scale_min_ = std::min(stat_scale_min_, scale);
        stat_ice_max_ms_ = std::max(stat_ice_max_ms_, ice_ms);
        stat_ice_sum_ms_ += ice_ms;
    }
}

ControllerMotionCommander::OutputRateStats ControllerMotionCommander::take_output_rate_stats()
{
    const std::scoped_lock lock(mutex_);
    OutputRateStats s;
    s.ticks = stat_ticks_;
    s.period_mean_ms = stat_ticks_ > 0 ? stat_period_sum_ms_ / static_cast<float>(stat_ticks_) : 0.f;
    s.period_max_ms = stat_period_max_ms_;
    s.cmd_age_max_ms = stat_cmd_age_max_ms_;
    s.scale_min = stat_scale_min_;
    s.ice_mean_ms = stat_ticks_ > 0 ? stat_ice_sum_ms_ / static_cast<float>(stat_ticks_) : 0.f;
    s.ice_max_ms = stat_ice_max_ms_;
    stat_ticks_ = 0; stat_period_sum_ms_ = 0.f; stat_period_max_ms_ = 0.f;
    stat_cmd_age_max_ms_ = 0.f; stat_scale_min_ = 1.f;
    stat_ice_max_ms_ = 0.f; stat_ice_sum_ms_ = 0.f;
    last_rate_stats_ = s;      // so the per-cycle CSV can read it without stealing the window
    return s;
}

ControllerMotionCommander::OutputRateStats ControllerMotionCommander::last_output_rate_stats() const
{
    const std::scoped_lock lock(mutex_);
    return last_rate_stats_;
}

void ControllerMotionCommander::stop_robot()
{
    if (command_text_sink_)
        command_text_sink_(0.f, 0.f, 0.f);

    // Latch the hold for the output thread FIRST and unconditionally: even when the graph-side dedup below
    // decides there is nothing new to publish, the base must start receiving zeros on the very next tick.
    {
        const std::scoped_lock lock(mutex_);
        pending_ = PendingCommand{.stamp_ms = current_time_ms(), .stop_requested = true};
    }
    cv_.notify_all();

    if (stop_command_latched_)
        return;
    if (!omnirobot_proxy_)
        return;

    publish_robot_reference_speed(0.f, 0.f, 0.f, current_time_ms());

    try
    {
        omnirobot_proxy_->stopBase();
        stop_command_latched_ = true;
        has_last_speed_command_ = false;
    }
    catch (const Ice::Exception &) {}
}