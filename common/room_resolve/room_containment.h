/*
 * common/room_resolve/room_containment.h — THE BIRTH RULE for countable concept agents, as a pure function.
 *
 * Is a candidate object's centre inside the room it would be parented to? Pure (Eigen only) so it is pinned by
 * room_containment_test.cpp without a graph. Shared by chair/table/bottle/hood/refrigerator/cabinet.
 *
 *   Admit    the centre is inside the room outline, or within `tolerance_m` of its boundary;
 *   Suppress the centre is outside by more than that — a thing seen through a door belongs to the other room;
 *   Defer    there is no usable outline yet (WAIT: never guess, never fall back to another room's polygon).
 *
 * `tolerance_m` is the agent's PHYSICAL footprint half-depth, not a tuning knob: a wall-mounted object's centre
 * may sit just outside the wall line by half its depth.
 */
#pragma once

#include <span>

#include <Eigen/Core>

#include "../nbv/viewpoint_score.h"   // rc::nbv::detail::point_in_polygon / distance_to_boundary
#include "room_resolve_rule.h"        // rc::room::left_behind

namespace rc::room
{

enum class BirthVerdict { Admit, Suppress, Defer };

/// A closed outline needs at least three vertices.
inline bool outline_usable(std::span<const Eigen::Vector2f> outline)
{
    return outline.size() >= 3;
}

inline BirthVerdict birth_verdict(std::span<const Eigen::Vector2f> outline, const Eigen::Vector2f& centre, float tolerance_m)
{
    if (not outline_usable(outline))
        return BirthVerdict::Defer;
    if (rc::nbv::detail::point_in_polygon(centre, outline) or
        rc::nbv::detail::distance_to_boundary(centre, outline) <= tolerance_m)
        return BirthVerdict::Admit;
    return BirthVerdict::Suppress;
}

/// The existence-channel term for an object outside its room (0 when inside).
inline float out_of_room_logodds(bool inside, float gain)
{
    return inside ? 0.f : -gain;
}

}   // namespace rc::room
