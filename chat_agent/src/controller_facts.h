#pragma once
/*
 * controller_facts.h — what the controller is doing now, as STATE lines (2026-10-18). Pure apart from the mission
 * progress type (Qt JSON); tested in tests/test_controller_facts.cpp.
 *
 * The controller drives on its own (exploring, door steps, missions), so the chat's last command is not what the
 * robot is doing: asked "where are you going?" while moving, the model invented a destination. Two sources, both
 * already in the graph:
 *   · the affordance it EXECUTES: the affordance node whose protocol reads Executing (active and pending), with the
 *     goal pose of the controller's `executing` claim edge, or the producer's proposed target when the claim carries
 *     none (a producer without proposal epochs, e.g. door_concept, gets no edge) — world_digest.cpp reads them;
 *   · the MISSION it runs: controller_mission's progress (mission_watch.h), the step being driven.
 */

#include <format>
#include <optional>
#include <string>

#include "../../common/mission/mission_progress.h"
#include "object_facts.h"

namespace rc::chat
{

/// An affordance's name in words. aff_<object>_<step> (door_concept's passage steps, an epistemic look) → "cross door_1",
/// "look at table_1"; a bare aff_<object> (door_concept's viewpoint to see the door better) → "look at door_1";
/// afford_room (room_concept's next viewpoint) → "explore the room". Any other name as it is.
[[nodiscard]] inline std::string affordance_words(const std::string& name)
{
    if (name == "afford_room") return "explore the room";
    if (not name.starts_with("aff_") or name.size() <= 4) return name;
    const std::string rest = name.substr(4);
    for (const char* step : {"approach", "open", "cross", "epistemic"})
        if (const std::string tail = std::string("_") + step; rest.ends_with(tail) and rest.size() > tail.size())
        {
            const std::string object = rest.substr(0, rest.size() - tail.size());
            return (std::string(step) == "epistemic" ? std::string("look at") : std::string(step)) + " " + object;
        }
    return "look at " + rest;
}

struct TargetFact
{
    std::string affordance;               ///< the executed affordance node's name
    std::optional<double> dist_m{};       ///< its goal from the robot (body frame), when the pose resolves
    std::optional<double> bearing_deg{};  ///< 0 = ahead, + = left
};

[[nodiscard]] inline std::string target_line(const std::optional<TargetFact>& t)
{
    if (not t.has_value()) return "controller target now: none (it is executing no affordance)";
    std::string s = std::format("controller target now: {} ({})", affordance_words(t->affordance), t->affordance);
    if (t->dist_m.has_value() and t->bearing_deg.has_value())
        s += std::format(", its goal {:.1f} m away, {}", *t->dist_m, bearing_words(*t->bearing_deg));
    return s;
}

/// A RUNNING mission's line; nullopt otherwise (a finished one is reported by the last-command line).
[[nodiscard]] inline std::optional<std::string> mission_now_line(const rc::mission::Progress& p)
{
    if (p.status != "in_progress") return std::nullopt;
    std::string s = std::format("controller mission now: \"{}\"", p.title);
    if (not p.source.empty()) s += " (from " + p.source + ")";
    if (not p.line.empty()) s += ", running: " + p.line;
    return s;
}

}   // namespace rc::chat
