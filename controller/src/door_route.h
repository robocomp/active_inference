#pragma once
/*
 * door_route.h — THE WAYPOINTS OF A DOOR ROUTE (2026-10-24, user: "we need a path that goes to a point in front of the door
 * and then into the other room, so the robot can follow it instead of wandering").
 *
 * door_concept owns the geometry: aff_<door>_approach publishes a vantage ON the door axis in front of the door
 * (door_approach_pose.h), aff_<door>_cross a point just past the aperture (DoorAffordance.CrossTargetM). The controller only
 * plans and follows: RouteFollower::build plans every hop with the grid planner and fits one curve. This header turns the
 * two published targets (+ the door's aperture centre, its RT translation) into that waypoint list:
 *
 *   approach leg (door shut, or no cross target yet):   [P_pre]
 *   through leg  (door open):                           [P_pre]?  →  P_ap  →  P_post
 *
 * P_ap (the aperture centre, projected onto the axis) keeps the through-segment straight on the axis.
 *
 * ★THE HOLE THE ROBOT PERCEIVES (2026-10-30, user: "the hole is already in robot (local) coordinates so the hole that is
 *  perceived has to be the right one"). When door_concept's interaction anchor (door_N@robot) is up, its opening — measured in
 *  the robot frame, composed with the robot's pose — IS the axis: the room-frame door (recog_proto11/12: re-fitted 0.45 m along
 *  the wall seconds after the crossing) only places the route before an anchor exists.
 * ★P_pre NEVER LIES FARTHER FROM THE HOLE THAN THE ROBOT. Its standoff is min(the published one, the robot's distance to the
 *  aperture centre): from far away it is the published vantage; from close by it is the point of the axis at the robot's own
 *  distance — reached by turning onto the axis, never by driving away from the door. This replaced a corridor gate (drop P_pre
 *  when |lateral| ≤ the body's reach), which with the door 0.45 m off sent the robot back to a vantage behind it, where it spun
 *  in place for minutes (recog_proto12 leg 2).
 *
 * ★THE AXIS IS THE ONLY FREEDOM. A published target whose footprint is NOT free on the controller's grid (door_concept does
 *  not see the controller's occupancy — a bed foot) is moved ALONG THE AXIS, never sideways: P_pre toward the door, down to
 *  the body's reach (closer and the footprint enters the doorway); P_post outward, by at most the corridor's forward extent
 *  (past that the planner has no free space anyway). This is a FREE-SPACE SEARCH over a 1-D parameter on the grid, at the
 *  grid's own cell step, taking the first free pose from the published one — not a threshold: no distance decides anything,
 *  the grid's own occupancy does. No free pose on the axis ⇒ nullopt, and the caller reports the route unbuildable.
 *
 * Pure Eigen/STL — no DSR, no Qt (tests/test_door_route.cpp).
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace rc::door_route
{

struct Target { Eigen::Vector2f p = Eigen::Vector2f::Zero(); float yaw = 0.f; };   ///< a published affordance target

struct Axis
{
    Eigen::Vector2f c = Eigen::Vector2f::Zero();   ///< the aperture centre, on the axis
    Eigen::Vector2f n = Eigen::Vector2f::UnitX();  ///< unit normal INTO the current room (the approach side)
};

/// The door axis from the approach target (its yaw faces the aperture: door_concept front_pose) and the aperture centre,
/// projected onto the approach target's line so the axis passes through the published standpoint. No centre: the cross
/// target's projection is the far end and the centre is unknown — nullopt (the caller drives the targets as published).
[[nodiscard]] inline std::optional<Axis> axis_from(const Target& approach, const std::optional<Eigen::Vector2f>& door_centre)
{
    if (not door_centre.has_value() or not door_centre->allFinite() or not approach.p.allFinite() or not std::isfinite(approach.yaw))
        return std::nullopt;
    const Eigen::Vector2f to_door(std::cos(approach.yaw), std::sin(approach.yaw));
    Axis a;
    a.n = -to_door;
    a.c = approach.p + to_door * (*door_centre - approach.p).dot(to_door);
    return a;
}

/// First standoff s (from s_from toward s_to, one `step` at a time, both ends included) whose pose c + s·n at `heading` is free.
[[nodiscard]] inline std::optional<float> first_free_standoff(const Axis& ax, float s_from, float s_to, float step, float heading,
                                                              const std::function<bool(const Eigen::Vector2f&, float)>& is_free)
{
    const float st = std::max(step, 1e-3f);
    const float dir = s_to >= s_from ? 1.f : -1.f;
    const int n = static_cast<int>(std::floor(std::abs(s_to - s_from) / st + 1e-4f));
    for (int k = 0; k <= n; ++k)
    {
        const float s = s_from + dir * st * static_cast<float>(k);
        if (is_free(ax.c + s * ax.n, heading)) return s;
    }
    if (is_free(ax.c + s_to * ax.n, heading)) return s_to;   // the end itself when the step does not land on it
    return std::nullopt;
}

struct Plan
{
    std::vector<Eigen::Vector2f> waypoints;   ///< what RouteFollower::build is given, in order
    Eigen::Vector2f pre = Eigen::Vector2f::Zero(), ap = Eigen::Vector2f::Zero(), post = Eigen::Vector2f::Zero();
    float s_pre = 0.f, s_post = 0.f;          ///< standoffs along the axis (s_post measured PAST the aperture)
    float pre_yaw = 0.f, post_yaw = 0.f;      ///< the final headings: facing the door at P_pre, the travel direction at P_post
    bool through = false, pre_moved = false, post_moved = false, has_axis = false;
    bool pre_omitted = false;                 ///< the robot stands in the doorway: no free vantage between it and the hole
    bool perceived = false;                   ///< the axis is the interaction anchor's opening (else the published targets')
    std::string why;                          ///< non-empty on failure
};

struct Params
{
    float body_reach_m = 0.33f;   ///< nearest P_pre may come to the aperture plane (the body's reach toward the door)
    float step_m = 0.05f;         ///< the grid's cell
    float post_slide_m = 0.8f;    ///< how far past the published cross target P_post may move (the corridor's forward extent)
};

/// The door route's waypoints. through = the door is open and a cross target is published.
/// `perceived`: the axis of the opening the robot measures (the interaction anchor, in the room frame) — preferred over the
/// one built from the published targets.
[[nodiscard]] inline Plan plan(const Target& approach, const std::optional<Target>& cross, const std::optional<Eigen::Vector2f>& door_centre,
                               const Eigen::Vector2f& robot, bool through, const Params& prm,
                               const std::function<bool(const Eigen::Vector2f&, float)>& is_free,
                               const std::optional<Axis>& perceived = std::nullopt)
{
    Plan out;
    out.through = through and cross.has_value();
    const auto ax = perceived.has_value() ? perceived : axis_from(approach, door_centre);
    out.perceived = perceived.has_value();
    out.has_axis = ax.has_value();
    out.pre = approach.p;
    out.pre_yaw = approach.yaw;
    if (ax.has_value())
    {
        const float face = std::atan2(-ax->n.y(), -ax->n.x());   // facing the door = the travel direction through it
        // the published standoff (along this axis), capped by the robot's own distance to the hole
        const float s_robot = (robot - ax->c).norm();
        const float s_pub = (approach.p - ax->c).dot(ax->n);
        out.s_pre = std::min(s_pub, s_robot);
        const auto s = first_free_standoff(*ax, out.s_pre, prm.body_reach_m, prm.step_m, face, is_free);
        // No free pose between the robot and the body's reach, the robot being NEARER the hole than the published vantage: it
        // stands in the doorway, and no vantage lies in front of it — the route starts at the aperture (recog_proto13 leg 2:
        // 0.21 m from the hole, the jambs inside every footprint). Not a tolerance: the grid says no such pose exists.
        out.pre_omitted = not s.has_value() and s_robot < s_pub and out.through;
        if (not s.has_value() and not out.pre_omitted)
        {
            out.why = std::format("no free standpoint on the door axis between {:.2f} m (published) and {:.2f} m (the body's reach)",
                                  out.s_pre, prm.body_reach_m);
            return out;
        }
        if (s.has_value())
        {
            out.pre_moved = std::abs(*s - out.s_pre) > 1e-4f;
            out.s_pre = *s;
        }
        out.pre = ax->c + out.s_pre * ax->n;
        out.pre_yaw = face;
        out.ap = ax->c;
        if (out.through)
        {
            out.s_post = (cross->p - ax->c).dot(-ax->n);
            const auto sp = first_free_standoff(*ax, -out.s_post, -(out.s_post + prm.post_slide_m), prm.step_m, face, is_free);
            if (not sp.has_value())
            {
                out.why = std::format("no free point past the door on its axis between {:.2f} and {:.2f} m", out.s_post,
                                      out.s_post + prm.post_slide_m);
                return out;
            }
            out.post_moved = std::abs(-*sp - out.s_post) > 1e-4f;
            out.s_post = -*sp;
            out.post = ax->c + *sp * ax->n;
            out.post_yaw = face;
            // P_pre always: at most as far from the hole as the robot (above), so it is never behind the robot; a robot beside
            // the doorway still gets the axis before the jambs (recog_try16).
            if (not out.pre_omitted) out.waypoints.push_back(out.pre);
            out.waypoints.push_back(out.ap);
            out.waypoints.push_back(out.post);
            return out;
        }
        out.waypoints.push_back(out.pre);
        return out;
    }
    // no aperture centre: the targets as published (logged by the caller)
    out.waypoints.push_back(approach.p);
    if (out.through)
    {
        out.post = cross->p;
        out.post_yaw = cross->yaw;
        out.waypoints.push_back(cross->p);
    }
    return out;
}

}   // namespace rc::door_route
