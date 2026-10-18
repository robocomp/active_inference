#pragma once
/*
 * command_gate.h — what the chat may ask the robot to do, decided in C++, never by the language model.
 *
 * The LLM proposes; this gate disposes. Every tool call and every typed command goes through validate() against the
 * facts the world digest read from the graph THIS cycle:
 *
 *   cross_door(door)   door ∈ doors of the current room, or "nearest" resolvable; controller present.
 *                      Resolved to an EXPLICIT door name, so the controller cannot pick a different "nearest" door
 *                      than the one the person meant.
 *   stop()             always accepted (stopping is the safe direction).
 *   cancel_passage()   "uncross"; controller present. Ends a passage, does not move the robot.
 *   move_relative(distance_m, turn_deg)
 *                      robot-relative step: turn first (+ = left), then drive straight (+ = forward, − = back).
 *                      → controller grammar "turn <deg> move <m>" (either part omitted when 0). Controller and room
 *                      known; |distance| ≤ max_move_m, |turn| ≤ 360, not both 0. The controller re-checks the swept
 *                      footprint against its own occupancy grid and refuses with a reason (controller_relative_motion.h).
 *
 * NO CONFIRMATION (user decision 2026-10-18): an accepted command is sent at once. The person can stop the robot at
 * any time ("stop" / "para", without the model). The card the model kept promising and the "door controller
 * confirmation" it invented were what stalled crossings (live 2026-10-18).
 *
 * A rejected call carries a reason, which goes back to the model as a role:"tool" message so it can tell the user.
 *
 * Plus the two deterministic fast paths that bypass the model entirely:
 *   is_stop(text)       ^(stop|para|detente|alto)\b   → `stop` immediately
 *   parse_direct(text)  text already in the controller grammar (cross door / cross door_N / uncross /
 *                       move <m> / turn <deg> / turn <deg> move <m>) → the same validation as the tool
 *
 * Pure std (no Qt, no DSR) — tested in tests/test_gate.cpp. Callers pass text already lower-cased (UTF-8 aware,
 * QString::toLower) and trimmed; ASCII is lower-cased again here so the tests need no Qt.
 */

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <format>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace rc::chat
{

/// The graph facts a decision may depend on (filled by WorldDigest).
struct GateFacts
{
    std::string room;                          ///< current room name, "" if unknown
    std::vector<std::string> doors;            ///< doors of the current room
    std::optional<std::string> nearest_door;   ///< nearest of them to the robot
    bool controller_present = false;
};

struct ToolCall
{
    std::string name;   ///< cross_door | stop | cancel_passage | move_relative
    std::string door{};   ///< cross_door's argument ("nearest" or a door name)
    double distance_m = 0.0;   ///< move_relative: + forward, − backward
    double turn_deg = 0.0;     ///< move_relative: + left (CCW), − right; executed BEFORE the distance
};

/// [Chat] limits for move_relative (etc/config.toml).
struct GateLimits
{
    double max_move_m = 3.0;           ///< longer is refused (the controller's own cap is 3 m)
};

/// "turn left 90 deg, then move forward 0.30 m"
inline std::string describe_relative(double distance_m, double turn_deg)
{
    std::string s;
    if (turn_deg != 0.0) s = std::format("turn {} {:.0f} deg", turn_deg > 0 ? "left" : "right", std::abs(turn_deg));
    if (distance_m != 0.0)
        s += std::format("{}move {} {:.2f} m", s.empty() ? "" : ", then ", distance_m > 0 ? "forward" : "backward",
                         std::abs(distance_m));
    return s;
}

struct GateResult
{
    bool ok = false;
    std::string command;   ///< the controller grammar text to send (cross door_19 | stop | uncross)
    std::string reason;    ///< why it was rejected
};

inline std::string ascii_lower_trim(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n.!");
    return s.substr(a, b - a + 1);
}

inline GateResult validate(const ToolCall& call, const GateFacts& f, const GateLimits& lim = {})
{
    GateResult r;
    if (call.name == "move_relative")
    {
        if (not f.controller_present) { r.reason = "the controller is not running, so the robot cannot move"; return r; }
        if (f.room.empty()) { r.reason = "the current room is not known yet, so the way cannot be checked"; return r; }
        if (not std::isfinite(call.distance_m) or not std::isfinite(call.turn_deg))
        { r.reason = "distance and angle must be numbers"; return r; }
        // The grammar's resolution: whole degrees, centimetres.
        const double turn = std::round(call.turn_deg);
        const double dist = std::round(call.distance_m * 100.0) / 100.0;
        if (std::abs(dist) > lim.max_move_m)
        {
            r.reason = std::format("{:.2f} m is more than a relative step (at most {:.1f} m at a time)", std::abs(dist),
                                   lim.max_move_m);
            return r;
        }
        if (std::abs(turn) > 360.0) { r.reason = "a turn of more than 360 degrees makes no sense"; return r; }
        if (turn == 0.0 and dist == 0.0) { r.reason = "nothing to do: distance and angle are both zero"; return r; }
        std::string cmd;
        if (turn != 0.0) cmd = std::format("turn {:.0f}", turn);
        if (dist != 0.0) cmd += std::format("{}move {:.2f}", cmd.empty() ? "" : " ", dist);
        r.ok = true;
        r.command = cmd;
        return r;
    }
    if (call.name == "stop")
    {
        r.ok = true;
        r.command = "stop";
        if (not f.controller_present) r.reason = "warning: the controller is not present; the stop is written anyway";
        return r;
    }
    if (call.name == "cancel_passage")
    {
        if (not f.controller_present) { r.reason = "the controller is not running"; return r; }
        r.ok = true;
        r.command = "uncross";
        return r;
    }
    if (call.name == "cross_door")
    {
        if (not f.controller_present) { r.reason = "the controller is not running, so the robot cannot move"; return r; }
        if (f.room.empty()) { r.reason = "the current room is not known yet"; return r; }
        std::string door = ascii_lower_trim(call.door);
        if (door.empty() or door == "nearest" or door == "door" or door == "the door")
        {
            if (not f.nearest_door.has_value())
            { r.reason = "no door of room '" + f.room + "' is known, so there is no nearest door"; return r; }
            // Same rule as the controller's "cross door": an unnamed door is only unambiguous in a one-door room.
            if (f.doors.size() > 1)
            {
                std::string known;
                for (const auto& d : f.doors) known += (known.empty() ? "" : ", ") + d;
                r.reason = "room '" + f.room + "' has " + std::to_string(f.doors.size()) + " doors (" + known +
                           "); ask the person which one";
                return r;
            }
            door = *f.nearest_door;
        }
        if (std::ranges::find(f.doors, door) == f.doors.end())
        {
            std::string known;
            for (const auto& d : f.doors) known += (known.empty() ? "" : ", ") + d;
            r.reason = "'" + door + "' is not a door of the current room '" + f.room + "'" +
                       (known.empty() ? std::string(" (it has no known doors)") : " (known: " + known + ")");
            return r;
        }
        r.ok = true;
        r.command = "cross " + door;
        return r;
    }
    r.reason = "unknown tool '" + call.name + "'";
    return r;
}

/// ^(stop|para|detente|alto)\b — the stop fast path. Sent at once, without the model.
inline bool is_stop(const std::string& text)
{
    static const std::regex re(R"(^(stop|para|detente|alto)\b)", std::regex::icase);
    return std::regex_search(ascii_lower_trim(text), re);
}

/// Locale-proof number (CLAUDE.md: never strtod/stod under es_ES).
inline std::optional<double> parse_number(const std::string& s)
{
    const char* a = s.data();
    const char* b = s.data() + s.size();
    if (a != b and *a == '+') ++a;
    double v = 0.0;
    const auto [p, ec] = std::from_chars(a, b, v);
    if (ec != std::errc{} or p != b) return std::nullopt;
    return v;
}

inline std::optional<ToolCall> parse_direct(const std::string& text);

/// "move -0.30" → "move -0.30 (move backward 0.30 m)" — the STATE's last-move line, so "a bit more" keeps the sign.
inline std::string explain_relative_command(const std::string& command)
{
    const auto c = parse_direct(command);
    if (not c.has_value() or c->name != "move_relative") return command;
    return command + " (" + describe_relative(c->distance_m, c->turn_deg) + ")";
}

/// Text already in the controller's grammar. English plus the obvious Spanish phrasing.
inline std::optional<ToolCall> parse_direct(const std::string& text)
{
    const std::string t = ascii_lower_trim(text);
    // move <m> | turn <deg> | turn <deg> [then|and] move <m> — exactly the controller's grammar, nothing looser:
    // "move a bit" is a sentence for the model, not a command.
    static const std::regex move_re(R"(^(?:move|advance) ([+-]?[0-9]*\.?[0-9]+) ?(?:m|metres|meters)?$)");
    static const std::regex turn_re(R"(^(?:turn|rotate) ([+-]?[0-9]*\.?[0-9]+) ?(?:deg|degrees)?$)");
    static const std::regex turn_move_re(
        R"(^(?:turn|rotate) ([+-]?[0-9]*\.?[0-9]+) ?(?:deg|degrees)?,?(?: then| and)? (?:move|advance) ([+-]?[0-9]*\.?[0-9]+) ?(?:m|metres|meters)?$)");
    {
        std::smatch mm;
        if (std::regex_match(t, mm, move_re))
            if (const auto d = parse_number(mm[1].str())) return ToolCall{.name = "move_relative", .distance_m = *d};
        if (std::regex_match(t, mm, turn_re))
            if (const auto a = parse_number(mm[1].str())) return ToolCall{.name = "move_relative", .turn_deg = *a};
        if (std::regex_match(t, mm, turn_move_re))
        {
            const auto a = parse_number(mm[1].str());
            const auto d = parse_number(mm[2].str());
            if (a and d) return ToolCall{.name = "move_relative", .distance_m = *d, .turn_deg = *a};
        }
    }
    static const std::regex cross_re(R"(^(cross|cruza|atraviesa)( the| la| a)? (door|puerta)$)");
    static const std::regex cross_n_re(R"(^(cross|cruza|atraviesa)( the| la)? (door_[0-9]+)$)");
    static const std::regex uncross_re(R"(^(uncross|cancel passage|cancel crossing)$)");
    std::smatch m;
    if (std::regex_match(t, m, cross_n_re)) return ToolCall{.name = "cross_door", .door = m[3].str()};
    if (std::regex_match(t, cross_re)) return ToolCall{.name = "cross_door", .door = "nearest"};
    if (std::regex_match(t, uncross_re)) return ToolCall{.name = "cancel_passage"};
    return std::nullopt;
}

/// Doors in `before` that are not in `after` (the current room's doors at the last request vs now). Deleted from the
/// map — by the person, or by door_concept's existence model — so the STATE says so: otherwise the model goes on
/// describing a door from its own earlier reply (live 2026-10-18).
inline std::vector<std::string> doors_gone(const std::vector<std::string>& before, const std::vector<std::string>& after)
{
    std::vector<std::string> out;
    for (const auto& d : before)
        if (std::ranges::find(after, d) == after.end()) out.push_back(d);
    return out;
}

/// "cross door_3" while door_3 is not among `doors` → "door_3": the command aims at a door that is no longer there.
inline std::optional<std::string> command_target_gone(const std::string& command, const std::vector<std::string>& doors)
{
    if (not command.starts_with("cross ")) return std::nullopt;
    const std::string door = command.substr(6);
    if (std::ranges::find(doors, door) != doors.end()) return std::nullopt;
    return door;
}

/// The model's line when a command ends (the robot_status tool message). state is the intention's "failed|<reason>",
/// "completed", …; event the BT's last event, often the same reason again (said once here). A FAILED relative move
/// says what it does not mean: the model otherwise read "a wall blocks the way" as blocking the door it was heading to
/// and refused every later crossing (live 2026-10-18).
inline std::string robot_status_text(const std::string& command, const std::string& state, const std::string& event)
{
    const auto bar = state.find('|');
    const std::string outcome = state.substr(0, bar);
    std::string reason = bar == std::string::npos ? "" : state.substr(bar + 1);
    if (reason.empty()) reason = event;
    // explain_relative_command gives "<command> (<plain words>)" for a relative move, the bare command otherwise
    std::string out = "The command '" + command + "'" + explain_relative_command(command).substr(command.size())
                    + " ended: " + outcome;
    if (not reason.empty()) out += " — " + reason;
    out += ".";
    const auto c = parse_direct(command);
    if (outcome == "failed" and c.has_value() and c->name == "move_relative")
        out += " That blocked only this straight-line move; it says nothing about a door: cross_door approaches a door "
               "along its own path and opens it if it is shut.";
    return out;
}

}   // namespace rc::chat
