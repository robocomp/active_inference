// remembered_goto.h -- "go to chair_34" for an object that is only REMEMBERED (long-term object memory, domain 5): the
// mission is built HERE, the missions skill's recipe word for word, and shown as a card for the person's yes. No model.
//
// Live 2026-11-10 (chat_e2e1), every step left to the model failed: the planner wrote a bare `approach chair_34` (the
// controller refused it: a remembered object offers no approach until it is seen again), and asked again as "go to where
// chair_34 was last seen" the chat model answered that it could only go to objects it perceives, calling no tool.
//
//   go_to_object_name     the person's words → the object's name, when the whole message is "go to <name>"
//   remembered_goto_text  the recipe: the legs into its room (topology.h legs_to), go_to its last pose, approach, report
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numbers>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "object_facts.h"

namespace rc::chat
{
/// The name in "go to chair_34", "go back to the chair_34", "go to where chair_34 was (last seen)", "ve a chair_34",
/// "vuelve donde estaba chair_34". Only a WHOLE message of that shape with ONE word for the object: anything more
/// ("go to the kitchen and look for a chair") is the model's to read. The caller decides whether the word names a
/// remembered object; nothing is matched against English inside a tool's result.
[[nodiscard]] inline std::optional<std::string> go_to_object_name(std::string_view text)
{
    std::string s(text);
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not s.empty() and (s.back() == ' ' or s.back() == '.' or s.back() == '!' or s.back() == '?')) s.pop_back();
    while (not s.empty() and s.front() == ' ') s.erase(s.begin());
    static constexpr std::array<std::string_view, 8> heads{
        "go to where ", "go back to where ", "go to ", "go back to ",
        "vuelve donde estaba ", "ve donde estaba ", "vuelve a ", "ve a "};
    static constexpr std::array<std::string_view, 3> tails{" was last seen", " was", " is"};
    for (const auto h : heads)
    {
        if (not s.starts_with(h)) continue;
        std::string name = s.substr(h.size());
        for (const auto t : tails)
            if (name.ends_with(t)) { name.resize(name.size() - t.size()); break; }
        if (name.starts_with("the ")) name = name.substr(4);
        if (name.empty() or name.contains(' ')) return std::nullopt;
        return name;
    }
    return std::nullopt;
}

/// The turn that faces a point given in the ROBOT frame (world_digest's convention: y ahead, x right, + = to the left),
/// whole degrees in (-180, 180].
[[nodiscard]] inline int turn_to_face_deg(double x_robot, double y_robot)
{
    int d = static_cast<int>(std::lround(std::atan2(-x_robot, y_robot) * 180.0 / std::numbers::pi));
    return d == -180 ? 180 : d;
}

/// The steps that bring a remembered object into view. In the robot's own room (turn_deg known): TURN to it and wait
/// `wait_s` for the tracker to see it again — chat_e2e2: it was behind the robot, so nothing could see it, and a go_to
/// onto its own centre had no route (the policy backed up 2 m). In another room: go_to its last pose (on_fail=skip).
/// ★ wait_s is a fixed time until the mission language has `wait until=seen(NODE)` (agreed 2026-11-10, owned here).
[[nodiscard]] inline std::vector<std::string> view_steps(const RememberedObject& o, double wait_s)
{
    if (o.turn_deg.has_value())
        return {std::format("move turn_deg={:.0f}deg", *o.turn_deg), std::format("wait s={:g}s", wait_s)};
    return {std::format("go_to pose:{}({:.2f}, {:.2f}) on_fail=skip", o.room, o.x, o.y)};
}

/// The missions skill's recipe for a remembered object: each leg into its room with retry=2, the steps that bring it
/// into view (view_steps), approach it (on_fail=skip: it may have moved), then report. `legs` are topology.h legs_to's
/// steps; empty when the object's room is the current one.
[[nodiscard]] inline std::string remembered_goto_text(const RememberedObject& o, const std::vector<std::string>& legs,
                                                      double wait_s)
{
    std::string t = std::format("mission \"go to where {} was last seen\"\n", o.name);
    for (const auto& l : legs) t += "  " + l + " retry=2\n";
    for (const auto& v : view_steps(o, wait_s)) t += "  " + v + "\n";
    t += std::format("  approach {} on_fail=skip\n", o.name);
    t += std::format("  report \"whether {} is still here\"\n", o.name);
    return t;
}
}   // namespace rc::chat
