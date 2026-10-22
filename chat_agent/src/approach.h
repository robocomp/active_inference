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

// ── an object's approach SIDES (2026-10-18) ─────────────────────────────────────────────────────────────────────
// Every object offers ONE approach affordance, aff_<object>_approach, listing the sides it can be approached from
// (aff_option_names + aff_option_x_m/_y_m/_yaw_rad, the default first; common/pragmatic_affordance/object_approach.h).
// A fronted object names them as its OWN sides (front, left, back, right: a chair's left as one sits in it); a
// frontless one by room axis (xp, yp, xn, yn), which mean nothing to a person. So each side also gets a label AS SEEN
// FROM THE ROBOT NOW (near / far / left / right), and the person's words are resolved against those — the model never
// does the geometry. Fronted names win for left/right/front/back ("the chair's right"); near/far are always the view.

struct ApproachSide
{
    std::string name;            ///< the option, as the object lists it
    double x = 0, y = 0, yaw = 0;   ///< its standpoint, room frame
    std::string view;            ///< near · far · left · right, as seen from the robot (label_sides)
};

/// Where `standpoint` lies around `object` as seen from `robot`: the side facing the robot is "near".
[[nodiscard]] inline std::string view_of(Xy object, Xy robot, Xy standpoint)
{
    double vx = object.x - robot.x, vy = object.y - robot.y;   // the viewing axis
    const double n = std::hypot(vx, vy);
    if (n < 1e-9) return "near";
    vx /= n;
    vy /= n;
    const double dx = standpoint.x - object.x, dy = standpoint.y - object.y;
    const double along = dx * vx + dy * vy, across = -dx * vy + dy * vx;   // across > 0 = the robot's left
    if (std::abs(along) >= std::abs(across)) return along < 0 ? "near" : "far";
    return across > 0 ? "left" : "right";
}

/// The options as the node lists them; an error when the parallel lists disagree (not an offer).
[[nodiscard]] inline std::expected<std::vector<ApproachSide>, std::string>
parse_sides(const std::string& names_csv, const std::vector<float>& xs, const std::vector<float>& ys,
            const std::vector<float>& yaws)
{
    std::vector<ApproachSide> out;
    std::size_t a = 0;
    while (a < names_csv.size())
    {
        const auto b = std::min(names_csv.find(',', a), names_csv.size());
        out.push_back({.name = names_csv.substr(a, b - a)});
        a = b + 1;
    }
    if (out.size() != xs.size() or out.size() != ys.size() or out.size() != yaws.size())
        return std::unexpected(std::string("the approach node's option lists have different lengths"));
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out[i].x = xs[i];
        out[i].y = ys[i];
        out[i].yaw = yaws[i];
    }
    return out;
}

inline void label_sides(std::vector<ApproachSide>& sides, Xy object, Xy robot)
{
    for (auto& s : sides) s.view = view_of(object, robot, {s.x, s.y});
}

namespace detail
{
inline bool is_own_side(const std::string& n) { return n == "front" or n == "back" or n == "left" or n == "right"; }
inline std::string lower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
inline bool has_any(const std::string& s, std::initializer_list<const char*> words)
{
    return std::ranges::any_of(words, [&](const char* w) { return s.find(w) != std::string::npos; });
}
}   // namespace detail

/// "xn (near side, as seen from here), yp (left side, as seen from here), …" — a fronted object's own sides say so.
[[nodiscard]] inline std::string sides_line(const std::vector<ApproachSide>& sides)
{
    std::string out;
    for (const auto& s : sides)
    {
        if (not out.empty()) out += ", ";
        out += detail::is_own_side(s.name) ? s.name + " (its own " + s.name + "; " + s.view + " side as seen from here)"
                                           : s.name + " (" + s.view + " side, as seen from here)";
    }
    return out;
}

/// The option the person's words mean; "" = no side asked for (the object's default). English and Spanish.
[[nodiscard]] inline std::expected<std::string, std::string> resolve_side(const std::vector<ApproachSide>& sides,
                                                                          const std::string& words_raw)
{
    std::string w = detail::lower(words_raw);
    while (not w.empty() and w.back() == ' ') w.pop_back();
    while (not w.empty() and w.front() == ' ') w.erase(w.begin());
    if (w.empty() or w == "default" or w == "any") return std::string{};
    for (const auto& s : sides)
        if (w == s.name) return s.name;
    const bool own = std::ranges::any_of(sides, [](const auto& s) { return detail::is_own_side(s.name); });
    const auto by_name = [&](const char* n) -> std::optional<std::string>
    {
        for (const auto& s : sides)
            if (s.name == n) return s.name;
        return std::nullopt;
    };
    const auto by_view = [&](const char* v) -> std::optional<std::string>
    {
        for (const auto& s : sides)
            if (s.view == v) return s.name;
        return std::nullopt;
    };
    using detail::has_any;
    std::optional<std::string> pick;
    bool understood = true;
    if (has_any(w, {"far", "other side", "opposite", "lado opuesto", "otro lado"})) pick = by_view("far");
    else if (has_any(w, {"near", "this side", "closest", "este lado", "cerca"})) pick = by_view("near");
    else if (has_any(w, {"right", "derecha"})) pick = own ? by_name("right") : by_view("right");
    else if (has_any(w, {"left", "izquierda"})) pick = own ? by_name("left") : by_view("left");
    else if (has_any(w, {"behind", "back", "detrás", "detras", "atrás", "atras"})) pick = own ? by_name("back") : by_view("far");
    else if (has_any(w, {"front", "frente", "delante"})) pick = own ? by_name("front") : by_view("near");
    else understood = false;
    if (pick.has_value()) return *pick;
    const std::string offered = sides.empty() ? std::string("its default side only") : sides_line(sides);
    return std::unexpected(understood ? "it cannot be approached from '" + words_raw + "' now; it offers: " + offered
                                      : "'" + words_raw + "' is not a side; it offers: " + offered);
}

// ── "go around the table" (2026-10-19) ───────────────────────────────────────────────────────────────────────────
// A full loop: one mission of go_to steps on a ring around the object, each facing it, starting from the stop
// nearest the robot and ending back there. The stops are the object's own approach sides when it offers at least
// three (they are standpoints it vouches for); otherwise kRingStops points at its footprint radius + standoff. A stop
// the controller cannot reach (against a wall, taken) is SKIPPED, not fatal: a loop with one side blocked is still the
// loop the person asked for, and the controller's planner drives each leg around the object itself.
// `left` = set off to the robot's left as it faces the object, i.e. clockwise seen from above.

inline constexpr int kRingStops = 6;

[[nodiscard]] inline rc::mission::Mission go_around_mission(const std::string& object, const std::string& room,
                                                            Xy centre, Xy robot, const std::vector<ApproachSide>& sides,
                                                            double ring_radius, bool left)
{
    using namespace rc::mission;
    std::vector<ApproachPoint> stops;
    if (sides.size() >= 3)
        for (const auto& s : sides) stops.push_back({s.x, s.y, s.yaw});
    else
        for (int i = 0; i < kRingStops; ++i)
        {
            const double a = 2.0 * std::numbers::pi * i / kRingStops;
            stops.push_back({centre.x + ring_radius * std::cos(a), centre.y + ring_radius * std::sin(a), a + std::numbers::pi});
        }
    const auto angle = [&](const ApproachPoint& p) { return std::atan2(p.y - centre.y, p.x - centre.x); };
    std::ranges::sort(stops, std::less{}, angle);   // counter-clockwise
    if (left) std::ranges::reverse(stops);           // clockwise
    // Start at the stop nearest the robot, so the loop does not begin by crossing to the far side.
    const auto first = std::ranges::min_element(stops, std::less{}, [&](const ApproachPoint& p)
                                                { return std::hypot(p.x - robot.x, p.y - robot.y); });
    std::ranges::rotate(stops, first);
    stops.push_back(stops.front());   // back to where it set off
    Mission m{.title = "go around " + object};
    for (std::size_t i = 0; i < stops.size(); ++i)
    {
        Ref to{.kind = Ref::Kind::Pose, .pose = Pose{.room = room, .x = stops[i].x, .y = stops[i].y, .yaw = stops[i].yaw}};
        m.steps.push_back(Step{.id = std::to_string(i + 1), .op = "go_to", .args = {{"to", to}},
                               .on_fail = OnFail{.kind = OnFail::Kind::Skip}});
    }
    return m;
}

/// "left" / "right" (English and Spanish) → set off to the robot's left; "" = left. nullopt = not a direction.
[[nodiscard]] inline std::optional<bool> parse_around_direction(const std::string& words_raw)
{
    const std::string w = detail::lower(words_raw);
    if (w.find_first_not_of(' ') == std::string::npos) return true;
    using detail::has_any;
    if (has_any(w, {"right", "derecha", "counterclockwise", "counter-clockwise", "anticlockwise", "antihorario"}))
        return false;
    if (has_any(w, {"left", "izquierda", "clockwise", "horario"})) return true;
    return std::nullopt;
}

/// A command as people read it: a mission payload ('{' — chat_intention carries both) by its title, else as written.
[[nodiscard]] inline std::string display_command(const std::string& command)
{
    if (not command.starts_with('{')) return command;
    const auto m = rc::mission::decode_json(command);
    return m.has_value() ? "mission \"" + m->title + "\"" : std::string("mission");
}

}   // namespace rc::chat
