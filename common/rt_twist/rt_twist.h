/*
 * rt_twist.h — read the body twist that rides in an RT edge's history ring. SHARED, header-only.
 *
 * ★WHY THIS IS SHARED RATHER THAN FOUR LINES IN EACH CONSUMER. The twist is not a loose attribute; it
 * is packed one 3-vector per ring slot alongside the pose it was measured with, so reading it means
 * finding the newest slot in rt_timestamps and indexing rt_twist_linear at slot*3. Every consumer that
 * open-codes that walk is one more place to get the slot arithmetic or the axis convention wrong, and
 * the fleet has already paid that bill: the pair this replaces was decoded by hand in four agents and
 * THREE of them put the advance rate on the wrong axis, rotating every prediction by 90 degrees. The
 * bug was found and fixed four separate times, on four separate dates, because the decoding was
 * duplicated rather than shared. This file is the cure for the producer-side half of that class.
 *
 * ★AXIS ORDER IS THE WHOLE POINT, so it is stated once here and nowhere else. rt_twist_linear is
 * [vx, vy, vz] and rt_twist_angular is [wx, wy, wz], both in the CHILD FRAME'S OWN AXES. That is what
 * lets a consumer write dp = R * v * dt without knowing which way the robot faces. It is NOT the
 * layout of the deprecated rt_translation_velocity, which is ARRAY order [adv, side, _]; on a
 * +Y-forward robot the first two numbers therefore appear SWAPPED between the two attributes. If you
 * are porting a call site off the old pair, that swap is the change — unless you only ever took
 * magnitudes (hypot of the first two, |wz|), which are identical in both layouts.
 *
 * ★THE STAMP COMES BACK WITH IT, and that is the second reason the ring exists. A twist read as a
 * loose attribute could not be paired with a pose or told apart from a dead producer's last value.
 * Here it carries the timestamp of the block it was measured with, so a caller can age it.
 *
 * A static edge (a camera mount, root->robot) has no ring and no twist: this returns nullopt for it,
 * which is the correct answer rather than a failure — a parameter does not move.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>

namespace rc::rt
{

struct Twist
{
    Eigen::Vector3f linear{Eigen::Vector3f::Zero()};   // [vx,vy,vz] m/s,   CHILD axes
    Eigen::Vector3f angular{Eigen::Vector3f::Zero()};   // [wx,wy,wz] rad/s, CHILD axes
    std::uint64_t   stamp_ms = 0;                       // the block this twist was measured with

    // The two quantities almost every consumer actually wants, named so nobody indexes for them.
    [[nodiscard]] float speed()    const { return std::hypot(linear.x(), linear.y()); }
    [[nodiscard]] float yaw_rate() const { return angular.z(); }
};

// Newest twist on an RT edge, or nullopt if the edge carries no twist ring (static edges, and any
// producer that has not been updated to write one).
[[nodiscard]] inline std::optional<Twist> newest_twist(DSR::DSRGraph &G, const DSR::Edge &edge)
{
    const auto ts  = G.get_attrib_by_name<rt_timestamps_att>(edge);
    const auto twl = G.get_attrib_by_name<rt_twist_linear_att>(edge);
    const auto twa = G.get_attrib_by_name<rt_twist_angular_att>(edge);
    if (not ts.has_value() or not twl.has_value() or not twa.has_value())
        return std::nullopt;

    // The ring is not in time order — a slot is overwritten in place — so the newest block is found
    // by scanning the stamps, never by taking the last element or trusting rt_head_index to be in
    // range. An out-of-order write (a late pose) lands wherever it belongs, not at the end.
    std::uint64_t newest = 0;
    std::size_t   slot   = 0;
    for (std::size_t k = 0; k < ts->get().size(); ++k)
        if (ts->get()[k] > newest) { newest = ts->get()[k]; slot = k; }
    if (newest == 0)
        return std::nullopt;

    const std::size_t base = slot * 3;
    if (twl->get().size() < base + 3 or twa->get().size() < base + 3)
        return std::nullopt;   // ring and twist packs disagree: report absence, never a partial read

    Twist t;
    t.linear  = {twl->get()[base], twl->get()[base + 1], twl->get()[base + 2]};
    t.angular = {twa->get()[base], twa->get()[base + 1], twa->get()[base + 2]};
    t.stamp_ms = newest;
    return t;
}

// Convenience for the usual shape: the twist of `child` relative to `parent`.
[[nodiscard]] inline std::optional<Twist> newest_twist(DSR::DSRGraph &G, DSR::RT_API *rt_api,
                                                       const DSR::Node &parent, std::uint64_t child_id)
{
    if (rt_api == nullptr)
        return std::nullopt;
    const auto e = rt_api->get_edge_RT(parent, child_id);
    return e.has_value() ? newest_twist(G, e.value()) : std::nullopt;
}


// ── THE PRODUCER'S OWN EGO-MOTION CHANNEL, AND THE FIRST PLACE TO LOOK ─────────────────────────
// ★IF THE QUESTION IS "HOW FAST IS THE ROBOT MOVING?", DO NOT READ THE RING FOR IT. The robot node
// carries robot_current_advance_speed / _side_speed / _angular_speed, published by robot_concept off
// the FullPoseEstimation stream (odometry fused with the IMU upstream in the base bridge, which is
// where the two are actually combined). That channel is strictly better for ego-motion than anything
// derived from an RT edge:
//   - it is the producer's own measurement, not a quantity recovered through a frame conversion;
//   - it is STAMPED (robot_current_speed_timestamp), so staleness is checkable;
//   - it carries a per-sample VARIANCE (robot_current_speed_variance, [var_adv, var_side, var_rot]);
//   - it does not depend on a room existing, or on which way the RT edge happens to be anchored.
//
// ★AND THE RING IS NOT A SUBSTITUTE, because on this fleet the localisation edge is anchored
// parent=ROBOT child=ROOM, so its twist is the ROOM's apparent motion. That differs from the robot's
// by the SE(2) adjoint, and the lever-arm term is not small: measured live 2026-09-10, a near-parked
// robot wobbling at 0.02 rad/s put 0.038 m/s in the ring against a true body twist of 0.0003 m/s.
// Two orders of magnitude. Use newest_twist() only when the ring itself is what you are asking about
// -- diagnosing an extrapolation, say -- and this function whenever you mean the robot's own motion.
[[nodiscard]] inline std::optional<Twist> robot_body_twist(DSR::DSRGraph &G, const DSR::Node &robot)
{
    const auto adv  = G.get_attrib_by_name<robot_current_advance_speed_att>(robot);
    const auto side = G.get_attrib_by_name<robot_current_side_speed_att>(robot);
    const auto rot  = G.get_attrib_by_name<robot_current_angular_speed_att>(robot);
    if (not adv.has_value() or not side.has_value() or not rot.has_value())
        return std::nullopt;

    Twist t;
    // ★The attributes are named SEMANTICALLY (advance / side); Twist is AXIS order. On this
    // +Y-forward robot the advance rate is the y component and the lateral rate the x one. This is
    // the same crossover that was mis-encoded in three consumers when it was left to each of them.
    t.linear  = {side.value(), adv.value(), 0.f};
    t.angular = {0.f, 0.f, rot.value()};
    if (const auto ts = G.get_attrib_by_name<robot_current_speed_timestamp_att>(robot); ts.has_value())
        t.stamp_ms = static_cast<std::uint64_t>(ts.value());
    return t;
}

// ── THE TWIST'S OWN CLOCK (2026-10-19, Task 9) ───────────────────────────────────────────────────
// ★A SIMULATOR'S VELOCITIES ARE PER SIMULATION SECOND. robot_concept forwards them as the producer states them, with
// the producer's sim stamp (robot_current_speed_sim_timestamp) and robot_current_speed_simulated beside the wall stamp
// (robot_current_speed_timestamp). A consumer that integrates the twist over WALL time over-counts every displacement
// by wall/sim — ★measured try66 (door_concept's interaction anchor, predicted by this twist over the wall clock): the
// sim ran at ~0.94× real time (commanded 0.35 m/s, the robot's ground truth 0.33 m/s per wall second) and the anchor's
// predicted translation came out 3–9 % long and its rotation 14–15 % long (the clock is ~6 % of that; the rest is the
// rotation's own over-report, which the anchor's evidence absorbs once the clock is right), so a 290° spin left it 15°
// and 1.1 m off, with σ 0.3° / 7 mm.
// room_concept's SimClockMap has integrated over the sim clock since 2026-08 (room_concept/src/common_types.h); this is
// the same estimator, shared, so no further consumer has to rediscover it.
// rate() = Δsim / Δwall over a trailing span, re-anchored after kSimClockSpanMs so a paused or restarted simulator cannot
// poison it for ever; 1 until a span of kSimClockMinSpanMs exists, and on real hardware (never observed). ⚑ Both spans
// are room_concept's SimClockMap's (30 s, 0.5 s): an estimation window, not a decision level.
inline constexpr std::int64_t kSimClockSpanMs = 30000;
inline constexpr std::int64_t kSimClockMinSpanMs = 500;
class SimClockRate
{
public:
    void observe(std::int64_t wall_ms, std::int64_t sim_ms)
    {
        if (wall_ms <= 0 or sim_ms <= 0 or wall_ms <= last_wall_) return;   // the same sample again, or out of order
        if (first_wall_ == 0 or sim_ms < last_sim_) { first_wall_ = wall_ms; first_sim_ = sim_ms; }   // first, or a restart
        last_wall_ = wall_ms; last_sim_ = sim_ms;
        if (last_wall_ - first_wall_ > kSimClockSpanMs) { first_wall_ = last_wall_; first_sim_ = last_sim_; }
        else if (const std::int64_t dw = last_wall_ - first_wall_, ds = last_sim_ - first_sim_; dw > kSimClockMinSpanMs and ds > 0)
            rate_ = double(ds) / double(dw);
    }
    // Δsim / Δwall: the factor that turns a wall interval into the interval a sim-second velocity must be integrated over.
    [[nodiscard]] double rate() const { return rate_; }
private:
    std::int64_t first_wall_ = 0, first_sim_ = 0, last_wall_ = 0, last_sim_ = 0;
    double rate_ = 1.0;
};
// The robot node's (wall, sim) stamp pair of its current twist sample — only when the producer says it is SIMULATED.
[[nodiscard]] inline std::optional<std::pair<std::int64_t, std::int64_t>> robot_twist_sim_stamps(DSR::DSRGraph &G, const DSR::Node &robot)
{
    const auto sim = G.get_attrib_by_name<robot_current_speed_simulated_att>(robot);
    if (not sim.has_value() or not sim.value()) return std::nullopt;
    const auto wall = G.get_attrib_by_name<robot_current_speed_timestamp_att>(robot);
    const auto sts  = G.get_attrib_by_name<robot_current_speed_sim_timestamp_att>(robot);
    if (not wall.has_value() or not sts.has_value()) return std::nullopt;
    return std::pair{static_cast<std::int64_t>(wall.value()), static_cast<std::int64_t>(sts.value())};
}

}   // namespace rc::rt
