// sensor_model.h -- the SensorModel struct, factored out of media_transport.h UNCHANGED so readers that only
// need the sensor physics (perception_card evaluator) do not pull FastDDS / the IDL types. Pure data, DDS-free.
// media_transport.h includes this, so its users see no change.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rc::media
{
// ---------------------------------------------------------------------------
// Sensor physics, carried alongside the transport descriptor
// ---------------------------------------------------------------------------
// WHY HERE. A driver is the only party that knows its own geometry and noise;
// every consumer today re-asserts them as its own config constants (room_concept
// has PreintOdomSigma*, residual_concept infers beam geometry). Two sources of
// truth for one physical fact, and nothing makes them disagree loudly. This
// rides the descriptor because that channel already exists end to end --
// MediaPlaneDDS::getMediaDescriptor() returns a string, robot_concept relays it
// VERBATIM onto the sensor node, and it re-relays only when the string changes.
// So no .idsl edit, no Ice stub regeneration, no attribute churn.
//
// ★ ABSENT MEANS UNKNOWN, AND THAT IS A REAL ANSWER. Every field is optional. A
// producer that has not measured its noise must leave the field out rather than
// advertise a plausible default: a nominal constant published as fact is exactly
// what left this channel inert from 2026-08-25 to 08-26. A consumer that finds a
// field absent keeps its own constant.
//
// ★ QUADRATURE, NEVER REPLACEMENT. What a consumer already holds is lumped MODEL
// error (slip, timing, unmodelled dynamics); this is SENSOR noise. They ADD.
// Replacing was tried 2026-08-23 and made the heading prior 135x too confident.
//
// ★ EXTRINSICS ARE NOT HERE ON PURPOSE. Where a sensor sits is the RT tree, which
// is authoritative, timestamped and interpolatable. `frame` only NAMES the frame
// these intrinsics are expressed in; it never restates the pose.
//
// ★ ONE NOISE IDIOM FLEET-WIDE: sigma(x) = hypot(floor, k * x). It is what
// p3bot-bridge already publishes for the wheels, and it degrades correctly -- at
// x = 0 a channel reports its floor, not zero, and not a flat density that would
// inject motion into a parked robot.
struct SensorModel
{
    // ── Provenance. Without it a number is an assertion, not a measurement. ──
    int          version  = 0;      // 0 = no model advertised
    std::string  source;            // "measured"|"calibrated"|"datasheet"|"nominal"|"simulated"
    std::string  ref;               // config file, calibration id, datasheet part
    std::int64_t stamp_ms = 0;      // when this model was produced/calibrated
    std::string  frame;             // DSR frame the intrinsics are expressed in

    // ── Intrinsic geometry (LiDAR). Extrinsics live in the RT tree. ──
    std::optional<int>   rings;
    std::vector<float>   ring_elev_deg;      // per-ring elevation; ';'-separated on the wire
    std::optional<float> azimuth_step_deg;
    std::optional<float> range_min_m, range_max_m;
    std::optional<float> fov_start_deg, fov_end_deg;
    std::optional<float> rate_hz;

    // ── Noise: range channel. sigma_r(r) = hypot(floor, k_rel*r), grown by incidence. ──
    std::optional<float> range_sigma_floor_m;
    std::optional<float> range_k_rel;        // per metre of range
    std::optional<float> range_k_incidence;  // per unit tan(incidence), optional

    // ── Noise: IMU. ──
    std::optional<float> gyro_sigma;         // rad/sqrt(s)
    std::optional<float> gyro_bias;          // rad/s
    std::optional<float> gyro_bias_rw;       // rad/s/sqrt(s), bias random walk
    std::optional<float> acc_sigma;          // m/s^2/sqrt(Hz)

    // ── Noise: wheels. Present for completeness; on P3Bot the wheel model is
    // already recoverable from the per-sample velCov that arrives at 50 Hz
    // (regressing var on speed^2 recovers k_slip to within 4-10%), so a wheel
    // producer need not fill these in. ──
    std::optional<float> v_floor, k_slip_v;  // m/sqrt(s), per (m/s)
    std::optional<float> w_floor, k_slip_w;  // rad/sqrt(s), per (rad/s)

    [[nodiscard]] bool empty() const noexcept { return version == 0; }
};
}  // namespace rc::media
