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

#include <algorithm>
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
// the same wall→sim mapping, shared, so no further consumer has to rediscover it — by CONSECUTIVE samples:
// sim_at(wall) maps a WALL instant onto the producer's sim clock by the last two samples: sim_last + (wall − wall_last)·r,
// r = Δsim/Δwall between those two CONSECUTIVE samples (Task 9 fix round 1: a trailing 30 s rate kept integrating at
// 0.94 through a PAUSED simulator, whose last twist sample still says the robot is moving; between consecutive samples a
// pause is Δsim = 0, so nothing is integrated). A consumer integrates a twist over sim_at(t1) − sim_at(t0). Unobserved
// (real hardware) ⇒ the identity. A restart (sim stepping back) re-anchors.
class SimClockRate
{
public:
    void observe(std::int64_t wall_ms, std::int64_t sim_ms)
    {
        if (wall_ms <= 0 or sim_ms <= 0 or (n_ > 0 and wall_ms <= wall_[1])) return;   // the same sample again, or out of order
        if (n_ > 0 and sim_ms < sim_[1]) n_ = 0;                                         // the simulator restarted
        wall_[0] = wall_[1]; sim_[0] = sim_[1];
        wall_[1] = wall_ms; sim_[1] = sim_ms;
        n_ = std::min(n_ + 1, 2);
    }
    // Δsim/Δwall between the last two samples; 1 until two exist (and on real hardware).
    [[nodiscard]] double rate() const
    {
        return n_ < 2 ? 1.0 : double(sim_[1] - sim_[0]) / double(wall_[1] - wall_[0]);
    }
    // The producer's sim clock at a wall instant (ms). Identity when unobserved.
    [[nodiscard]] double sim_at(std::int64_t wall_ms) const
    {
        if (n_ == 0) return double(wall_ms);
        return double(sim_[1]) + double(wall_ms - wall_[1]) * rate();
    }
    // The producer-time interval (s) between two WALL instants, both mapped by the CURRENT samples — so it is never
    // negative and never frozen by when the earlier instant was mapped (fix round 2, try68: an anchor created before the
    // first sample stored its stamp mapped by the identity, on the WALL epoch, and every later interval against the sim
    // clock came out 0 — the anchor was never predicted).
    [[nodiscard]] double interval_s(std::int64_t wall_then, std::int64_t wall_now) const
    {
        return std::max(0.0, sim_at(wall_now) - sim_at(wall_then)) * 1e-3;
    }
private:
    std::int64_t wall_[2] = {0, 0}, sim_[2] = {0, 0};
    int n_ = 0;
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
