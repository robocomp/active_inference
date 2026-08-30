#pragma once
#include <source_location>

#include <genericworker.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "controller_runtime_types.h"

class ControllerWorldModel;

class ControllerMotionCommander
{
public:
    // Numbers, not a sentence: the UI shows them on LCDs, and formatting here only to re-parse there
    // would put the display's precision decision inside the motion path.
    using CommandTextSink = std::function<void(float adv_mm_s, float side_mm_s, float rot_rps)>;
    // Sampled once per OUTPUT tick, after the freshness scale and the slew limiter — i.e. what the base
    // is actually told, not what the planner wished for. That is the signal a smoothness metric must
    // measure, and the output thread is the only fixed-rate clock in the agent (compute() is ragged).
    using ProfileSink = std::function<void(std::uint64_t t_ms, float adv, float side, float rot, float freshness)>;
    void set_profile_sink(ProfileSink sink) { profile_sink_ = std::move(sink); }

    void set_params(const ControllerParams *params);
    void set_dependencies(std::shared_ptr<DSR::DSRGraph> graph,
                          const ControllerWorldModel *world_model,
                          RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy,
                          int agent_id,
                          CommandTextSink command_text_sink);

    ~ControllerMotionCommander();

    void apply_uncertainty_speed_limit(float &adv_mps, float &side_mps, float &rot_rps) const;

    // What apply_uncertainty_speed_limit ACTUALLY did on its last call. The limiter is the only place the
    // robot's own localisation covariance is allowed to bound its speed, and none of it was observable:
    // a lap could not distinguish "sigma is large and this is throttling hard" from "no covariance on the
    // RT edge at all, so the whole mechanism is inert". Both look like a robot that will not go fast.
    // valid=false means read_pose_uncertainty() returned nullopt ⇒ NO limiting was applied.
    struct UncertaintyDiag
    {
        bool  valid = false;
        float xy_std_m = -1.f;      // -1 = not available (never a real sigma)
        float theta_std_rad = -1.f;
        float adv_scale = 1.f;      // multiplier actually applied to adv/side
        float rot_scale = 1.f;      // multiplier actually applied to rot
    };
    UncertaintyDiag last_uncertainty_diag() const { return uncertainty_diag_; }
    // ★★★WHO COMMANDED THE BASE. Measured 2026-08-19: 16 s in which cmd_rot alternated between
    // -0.800 and +0.800 (the full cap) while track_s stayed frozen at 0.07 and cross-track error grew
    // 0.46 -> 0.89 m. The alternation was perfectly correlated with adv: (0.106, -0.800) one cycle,
    // (0.000, +0.800) the next. That is TWO command sources taking turns, neither ever winning — the
    // robot pirouettes off its path while making no progress along it, and the stuck detector stays
    // silent because it IS moving. std::source_location makes the code name its own caller, so both
    // authorities are identified in one run instead of by reading the tracker.
    void send_speed_command(float adv_mps, float side_mps, float rot_rps,
                            std::source_location loc = std::source_location::current());
    void stop_robot();

    /// ARM / DISARM the base output entirely.
    ///
    /// stop_robot() only makes the command ZERO — the output loop keeps sending setSpeedBase(0,0,0)
    /// at the fixed cadence forever, because a constant cadence is what a base watchdog expects. That
    /// is right while the robot is under control and wrong after an ABORT: the user pressed Stop and
    /// expects the controller to stop talking to the base until they press Run again.
    ///
    /// Disarming sends a short BURST of explicit zeros first (a single dropped packet must not leave
    /// the base coasting at the last non-zero command) and then goes silent. Arming resumes the
    /// cadence immediately. The loop keeps ticking either way, so timing statistics stay continuous.
    void set_output_enabled(bool enabled);
    bool output_enabled() const;

    // Achieved output cadence since the last call (diagnostic; resets the accumulators).
    struct OutputRateStats
    {
        int   ticks = 0;              // LOOP iterations — counted whether or not anything was sent
        int   sends = 0;              // of those, how many actually reached the base
        float period_mean_ms = 0.f;
        float period_max_ms = 0.f;    // worst gap between consecutive loop TICKS
        float cmd_age_max_ms = 0.f;   // oldest command ever pushed to the base
        float scale_min = 1.f;        // strongest freshness attenuation applied
        float ice_mean_ms = 0.f;      // setSpeedBase RPC duration — a hard floor on the achievable period
        float ice_max_ms = 0.f;
    };
    OutputRateStats take_output_rate_stats();
    /// The last stats take_output_rate_stats() computed, WITHOUT consuming the accumulators. The taking
    /// call resets them, so a second consumer would silently steal the first one's window — this lets the
    /// per-cycle CSV record the same numbers the 5 s [vel-out] line prints, one control cycle old.
    OutputRateStats last_output_rate_stats() const;

private:
    // Written by apply_uncertainty_speed_limit (const, hence mutable) and read on the same thread by the
    // control cycle that just called it. Never touched by the output loop, so it needs no lock.
    mutable UncertaintyDiag uncertainty_diag_{};
    OutputRateStats last_rate_stats_{};   // cached copy of the last take (see last_output_rate_stats)

    // ── Fixed-rate output loop ──
    // The base used to be commanded from inside compute(), which meant the command rate WAS the perception
    // rate: measured median 108 ms but mean 212, p99 1.26 s and a worst case of 9.5 s, with 16% of cycles over
    // twice the configured 100 ms period. On top of that, send_speed_command() deduplicated — an unchanged
    // command was never sent at all — so the base could go arbitrarily long with no message even when the loop
    // was healthy. Both together are what makes the motion stutter.
    //
    // So the actuation path is now its own thread ticking at a fixed period, publishing the latest command
    // regardless of what compute() is doing and WITHOUT dedup (a base watchdog needs to keep hearing from us).
    // It deliberately touches nothing but the Ice proxy and a mutex-guarded POD: no DSR (the graph publish and
    // the UI sink stay on the control thread, which also avoids racing another writer on the robot node's
    // attrs), no Qt, no cv::Mat.
    //
    // Why a thread and not a QTimer: compute() already runs on its own control thread (the presence hook
    // on_operating_loop only wakes it), so a GUI-thread timer would not be starved by a slow compute() — but
    // it WOULD still be at the mercy of GUI work, and more to the point it would not help at all, because the
    // problem is not only WHEN the command is sent but that setSpeedBase is a synchronous RPC issued from
    // whichever thread produced the command. Owning the cadence in one place, decoupled from both, is what
    // actually makes the period constant.
    void output_loop(std::stop_token stop);

    struct PendingCommand
    {
        float adv_mps = 0.f, side_mps = 0.f, rot_rps = 0.f;
        std::uint64_t stamp_ms = 0;
        bool stop_requested = true;   // start held: never drive until compute() has actually asked for it
    };
    static float ramp_uncertainty_scale(float value, float slow_threshold, float stop_threshold, float min_scale);
    static float preserve_sign_clamp(float value, float max_abs);
    static std::uint64_t current_time_ms();

    void publish_robot_reference_speed(float adv_mps, float side_mps, float rot_rps, std::uint64_t timestamp_ms);

    static constexpr const char *kRobotRefAdvSpeedAttr = "robot_ref_adv_speed";
    static constexpr const char *kRobotRefRotSpeedAttr = "robot_ref_rot_speed";
    static constexpr const char *kRobotRefSideSpeedAttr = "robot_ref_side_speed";
    static constexpr const char *kRobotRefSpeedTimestampAttr = "robot_ref_speed_timestamp";

    const ControllerParams *params_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    const ControllerWorldModel *world_model_ = nullptr;
    RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy_;
    int agent_id_ = 0;
    CommandTextSink command_text_sink_;
    ProfileSink profile_sink_;
    bool stop_command_latched_ = false;
    bool has_last_speed_command_ = false;
    Eigen::Vector3f last_speed_command_ = Eigen::Vector3f::Zero();

    // Shared with the output thread. `mutex_` guards pending_ and the stats accumulators; `cv_` lets the
    // destructor wake the loop immediately instead of waiting out a tick.
    mutable std::mutex mutex_;
    bool output_enabled_ = true;      // false ⇒ the loop stays silent (see set_output_enabled)
    bool disarmed_notice_given_ = false;   // the "commands are being discarded" line, once per disarm
    // How many explicit zeros go out before the loop goes quiet. Long enough that a dropped packet
    // cannot leave the base holding the last non-zero value; short enough to hand the bus back fast.
    static constexpr int kQuietBurstTicks = 5;   // 5 ticks at 20 Hz = 250 ms of "definitely stop"
    int  quiet_burst_left_ = 0;       // zero commands still to send before going silent

    // ── WHEN THERE IS NOTHING TO DRIVE, LET GO OF THE BASE ────────────────────────────────────────
    // ★★★THE DEFECT. The loop sent setSpeedBase EVERY tick "unchanged or not", which is right while
    // driving and wrong the moment the controller has nothing to say: after a tour completes, after an
    // affordance finishes, during the post-affordance dwell, while waiting for a pose or a room
    // polygon, and before Run is ever pressed. In all of those the controller was streaming
    // setSpeedBase(0,0,0) at 20 Hz for ever, which is not harmless — it is a second writer on the
    // base, and it BLOCKS THE JOYSTICK. Stop already knew this ("the output loop otherwise keeps
    // sending setSpeedBase(0,0,0) at 20 Hz forever, which is right under control and wrong after the
    // user has said stop" -- specificworker.cpp) and disarmed explicitly; the same reasoning was
    // simply never applied to the controller finishing on its own.
    //
    // ★THE CONDITION IS THE COMMAND, NOT A MAGNITUDE. `stop_requested` is a latched STATEMENT that
    // the controller is holding the robot, and every idle path already sets it: the halt branch calls
    // stop_robot(), and all three of build_planning_step's nullopt returns call
    // ControllerSession::stop() — including "idle — no eligible affordance". So no |v| < eps test is
    // needed, and none is wanted: a freshness-decayed command coasts asymptotically toward zero
    // without ever reaching it, so a magnitude test would either cut that coast short or never fire.
    // PendingCommand::stop_requested also defaults TRUE, so a freshly constructed commander is quiet
    // from the start rather than hogging the base before Run.
    //
    // ★IT COSTS NOTHING TO RESUME. The tick still happens; only the RPC is skipped. send_speed_command
    // clears stop_requested, so motion resumes on the very next tick (<= one output period).
    //
    // ★SILENCE IS EQUIVALENT TO ZERO, once the burst has landed. The base is a velocity servo holding
    // the last command — which is zero — so ceasing to repeat it changes nothing about the robot and
    // everything about who else can talk to it.
    int  idle_quiet_left_ = kQuietBurstTicks;
    bool idle_quiet_notice_given_ = false;   // said once per idle episode, not per tick
    std::condition_variable_any cv_;
    PendingCommand pending_;
    // Last values actually sent to the base — the state the slew limiter integrates from. Output-thread only.
    float applied_adv_ = 0.f, applied_side_ = 0.f, applied_rot_ = 0.f;
    int   stat_ticks_ = 0;
    int   stat_sends_ = 0;
    float stat_period_sum_ms_ = 0.f;
    float stat_period_max_ms_ = 0.f;
    float stat_cmd_age_max_ms_ = 0.f;
    float stat_scale_min_ = 1.f;
    float stat_ice_max_ms_ = 0.f;
    float stat_ice_sum_ms_ = 0.f;
    // Declared LAST so it is destroyed FIRST: the thread must stop before the Ice proxy it uses goes away.
    std::jthread output_thread_;
};