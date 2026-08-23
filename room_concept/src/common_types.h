#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

// Minimal shared types required by room_model/room_concept code.
// Keep this header lightweight and dependency-free.

namespace rc
{
    enum class RoomState
    {
        MAPPING = 0,
        LOCALIZED = 1
    };

    struct VelocityCommand
    {
        float adv_x = 0.0f;  // lateral velocity, m/s (robot frame)
        float adv_y = 0.0f;  // forward velocity, m/s (robot frame)
        float rot = 0.0f;    // angular velocity, rad/s
        std::int64_t source_ts_ms = 0;   // timestamp from the source (sensor/planner epoch-ms)
        std::int64_t recv_ts_ms   = 0;   // local reception timestamp (epoch-ms)
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp; // kept for compat
        std::int64_t effective_ts_ms() const { return source_ts_ms > 0 ? source_ts_ms : recv_ts_ms; }
        VelocityCommand() = default;
        VelocityCommand(float x, float z, float r)
            : adv_x(x), adv_y(z), rot(r)
            , timestamp(std::chrono::high_resolution_clock::now())
        {}
    };

    /// Measured odometry reading from encoders/IMU (robot frame velocities)
    /// Received from the active odometry ingestion path, currently DSR attrs.
    ///
    /// TWO CLOCKS, and which to use depends on what you are doing. A simulator reports velocities
    /// per SIMULATION second and runs behind real time, so integrating those rates over wall-clock
    /// intervals under-counts by exactly the sim/wall ratio (~6% on this setup). `sim_ts_ms` is the
    /// instant this sample refers to on the producer's own clock and is the one to INTEGRATE
    /// against; `source_ts_ms` stays wall-clock and remains right for latency, staleness and
    /// ordering against anything else on this host. On real hardware the two coincide and
    /// `simulated` is false, so integration_ts_ms() needs no special case either way.
    struct OdometryReading
    {
        float adv = 0.0f;    // forward velocity, m/s (robot frame +Y)
        float side = 0.0f;   // lateral velocity, m/s (robot frame +X)
        float rot = 0.0f;    // angular velocity, rad/s (CCW+)
        std::int64_t source_ts_ms = 0;   // timestamp from the sensor (epoch-ms, WALL)
        std::int64_t sim_ts_ms    = 0;   // the same sample on the producer's SIM clock; 0 when real
        bool         simulated    = false;
        std::int64_t recv_ts_ms   = 0;   // local reception timestamp (epoch-ms)
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp; // kept for compat
        std::int64_t effective_ts_ms() const { return source_ts_ms > 0 ? source_ts_ms : recv_ts_ms; }
        /// The stamp to integrate rates against: the producer's own clock where there is one.
        std::int64_t integration_ts_ms() const
        { return (simulated and sim_ts_ms > 0) ? sim_ts_ms : effective_ts_ms(); }
    };

    /// One inertial sample. This is the FAST channel -- ~125 Hz against odometry's 10 Hz -- and it
    /// exists separately because yaw is the channel wheel odometry gets worst. A differential base
    /// can only turn by scrubbing its wheels sideways, so it over-reports rotation (measured 8.2% on
    /// this robot) while translation stays exact to 0.1%. Integrating the gyro over the same
    /// interval replaces an error the wheels cannot fix with one they do not have.
    struct ImuReading
    {
        float gyro_z   = 0.0f;   // angular rate about robot +Z, rad/s, CCW+
        float gyro_var = -1.0f;  // (rad/s)^2 per-sample variance; <0 means the producer said "unknown"
        // Proper acceleration in the BODY frame, m/s^2, GRAVITY INCLUDED (a real accelerometer reads
        // ~9.81 on the up axis at rest, and Webots models that faithfully). The horizontal pair is
        // usable as horizontal acceleration only to the extent the mount is level: a tilt error eps
        // leaks g*sin(eps) into them, which at 0.5 deg is 0.086 m/s^2 and dominates the sensor's own
        // noise by an order of magnitude. That leak is a slowly varying bias, not noise, and must be
        // treated as a state rather than averaged away.
        //
        // WHAT THIS IS FOR: velocity CHANGE over one short interval, never velocity itself. Over the
        // ~50 ms between predictions the noise contributes 0.1-0.4 mm/s against the wheels' 67 mm/s
        // per sample, so the increment is ~15x better than the wheels. Chained across intervals the
        // same 0.5 deg tilt gives 0.086 m/s after 1 s and 5.1 m/s after 60 s -- which is why this
        // codebase refused the double integration for absolute motion and still should.
        float acc_x    = 0.0f;   // lateral  (body +X)  -- VERIFY the axis mapping empirically
        float acc_y    = 0.0f;   // forward  (body +Y)
        float acc_var  = -1.0f;  // (m/s^2)^2; <0 = unknown. The media-plane ImuFrame IDL carries NO
                                 // acc_var field (only gyro_var), so this is unknown until the IDL
                                 // gains one -- a real gap for any precision-weighted use.
        std::int64_t source_ts_ms = 0;   // WALL epoch-ms
        std::int64_t sim_ts_ms    = 0;   // producer's SIM clock; 0 when real
        bool         simulated    = false;
        std::int64_t recv_ts_ms   = 0;
        std::int64_t effective_ts_ms() const { return source_ts_ms > 0 ? source_ts_ms : recv_ts_ms; }
        std::int64_t integration_ts_ms() const
        { return (simulated and sim_ts_ms > 0) ? sim_ts_ms : effective_ts_ms(); }
    };

    /// Maps the producer's SIM clock onto the local WALL clock, fitted from the (wall, sim) pairs
    /// that arrive together on every odometry sample.
    ///
    /// It exists because the two ends of an integration interval come from different places: the
    /// interval bounds are lidar sweep stamps, which are wall-clock, while the rates being
    /// integrated are per-sim-second. Converting the BOUNDS is the cheap direction -- it keeps the
    /// change inside the integrator instead of re-clocking the whole localizer.
    ///
    /// A plain offset would drift, because the sim clock ticks slower than the wall clock; this
    /// tracks the RATE too, over a trailing window, and falls back to identity until it has a real
    /// span to divide. Identity is exactly right for real hardware, where the clocks coincide.
    class SimClockMap
    {
    public:
        void observe(std::int64_t wall_ms, std::int64_t sim_ms)
        {
            if (wall_ms <= 0 or sim_ms <= 0) return;
            if (first_wall_ == 0) { first_wall_ = wall_ms; first_sim_ = sim_ms; }
            last_wall_ = wall_ms; last_sim_ = sim_ms;
            // Re-anchor on a long span so the rate stays a trailing estimate rather than a lifetime
            // average, and so a paused or restarted simulator cannot poison it forever.
            if (wall_ms - first_wall_ > 30000) { first_wall_ = last_wall_; first_sim_ = last_sim_; }
            valid_ = true;
        }
        bool valid() const { return valid_; }
        /// Wall epoch-ms -> the producer's sim clock. Identity when unmapped or on real hardware.
        std::int64_t to_sim(std::int64_t wall_ms) const
        {
            if (not valid_) return wall_ms;
            const std::int64_t dw = last_wall_ - first_wall_;
            const std::int64_t ds = last_sim_ - first_sim_;
            const double rate = (dw > 500 and ds > 0) ? static_cast<double>(ds) / static_cast<double>(dw) : 1.0;
            return last_sim_ + static_cast<std::int64_t>(std::llround((wall_ms - last_wall_) * rate));
        }
    private:
        std::int64_t first_wall_ = 0, first_sim_ = 0, last_wall_ = 0, last_sim_ = 0;
        bool valid_ = false;
    };

    using VelocityHistory = std::vector<VelocityCommand>;
    using TimeStamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
}
