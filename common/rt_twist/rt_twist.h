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

}   // namespace rc::rt
