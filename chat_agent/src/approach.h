#pragma once
/*
 * approach.h — "approach the table": which object, where to stand, and the grounded mission that drives there
 * (2026-10-18). Pure (no DSR, no Qt beyond the mission IR); tested in tests/test_approach.cpp.
 *
 * Before this the chat had no way to go to an object: the model sent a straight `move` at it (refused — the object
 * itself blocks the way) or claimed to be moving without calling anything. The controller already drives to a POSE in
 * the current room with its planner, around obstacles: the mission language's go_to (controller
 * specificworker_mission.cpp). So the chat picks the object, puts a standpoint short of it, and sends a one-step
 * mission; the controller mirrors its progress into controller_mission (the Plan window) and chat_intention.
 *
 * WHERE TO STAND: on the line from the object's centre towards the robot, at the object's footprint radius (half
 * its footprint diagonal; 0 when its size is unknown) plus `standoff` = robot radius + clearance ([Chat] config) —
 * geometry, not a tuning knob: the body must not overlap the object, and the clearance is room to turn. The
 * approach is from the side the robot is on; if that spot is taken the controller moves the standpoint itself
 * (its "[approach] … Standpoint moved" rule).
 */

#include <algorithm>
#include <cmath>
#include <expected>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "../../common/mission/mission_ir.h"
#include "../../common/mission/mission_json.h"
#include "object_facts.h"

namespace rc::chat
{

struct Xy
{
    double x = 0, y = 0;
};

struct ApproachPoint
{
    double x = 0, y = 0, yaw = 0;   ///< room frame; yaw faces the object
};

/// The object meant by `name_raw`: an exact name, else the nearest of that class — in the CURRENT room only, since the
/// controller's go_to drives within the room the robot is in. The error says why not, in words for the model.
[[nodiscard]] inline std::expected<ObjectFact, std::string> pick_object(const std::vector<ObjectFact>& objects,
                                                                        const std::string& name_raw)
{
    std::string name = name_raw;
    std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not name.empty() and name.back() == ' ') name.pop_back();
    if (name.starts_with("the ")) name = name.substr(4);
    const std::string want = normalize_subtype(name);
    std::vector<ObjectFact> here, elsewhere;
    for (const auto& o : objects)
        if (o.name == name or (not want.empty() and o.subtype == want))
            (o.in_current_room ? here : elsewhere).push_back(o);
    if (const auto exact = std::ranges::find(here, name, &ObjectFact::name); exact != here.end()) return *exact;
    if (not here.empty())
    {
        sort_by_distance(here);
        return here.front();
    }
    if (not elsewhere.empty())
        return std::unexpected("'" + name_raw + "' is in " +
                               (elsewhere.front().room.empty() ? std::string("another room") : elsewhere.front().room) +
                               ", not in the room the robot is in: cross into that room first");
    return std::unexpected("'" + name_raw + "' is not perceived in the current room");
}

/// Half the footprint diagonal: the radius of the disc the object covers on the floor. 0 when the size is unknown.
[[nodiscard]] inline double footprint_radius(const ObjectFact& o)
{
    if (not o.width_m.has_value() or not o.depth_m.has_value()) return 0.0;
    return 0.5 * std::hypot(double(*o.width_m), double(*o.depth_m));
}

/// The standpoint `radius + standoff` from the object's centre, towards the robot, facing the object. nullopt when
/// the robot is already that close (nothing to drive).
[[nodiscard]] inline std::optional<ApproachPoint> approach_point(Xy object, Xy robot, double radius, double standoff)
{
    const double dx = robot.x - object.x, dy = robot.y - object.y;
    const double d = std::hypot(dx, dy);
    const double r = radius + standoff;
    if (d <= r) return std::nullopt;
    const double ux = dx / d, uy = dy / d;
    return ApproachPoint{.x = object.x + ux * r, .y = object.y + uy * r, .yaw = std::atan2(-uy, -ux)};
}

/// One go_to step to the standpoint, in the current room.
[[nodiscard]] inline rc::mission::Mission approach_mission(const std::string& object, const std::string& room,
                                                           const ApproachPoint& p)
{
    using namespace rc::mission;
    Ref to{.kind = Ref::Kind::Pose, .pose = Pose{.room = room, .x = p.x, .y = p.y, .yaw = p.yaw}};
    Step s{.id = "1", .op = "go_to", .args = {{"to", to}}};
    return Mission{.title = "approach " + object, .steps = {s}};
}

/// A command as people read it: a mission payload ('{' — chat_intention carries both) by its title, else as written.
[[nodiscard]] inline std::string display_command(const std::string& command)
{
    if (not command.starts_with('{')) return command;
    const auto m = rc::mission::decode_json(command);
    return m.has_value() ? "mission \"" + m->title + "\"" : std::string("mission");
}

}   // namespace rc::chat
