#pragma once
/*
 * command_gate.h — what the chat may ask the robot to do, decided in C++, never by the language model.
 *
 * The LLM proposes; this gate disposes. Every tool call and every typed command goes through validate() against the
 * facts the world digest read from the graph THIS cycle:
 *
 *   cross_door(door)   door ∈ doors of the current room, or "nearest" resolvable; controller present.
 *                      Resolved to an EXPLICIT door name, so the controller cannot pick a different "nearest" door
 *                      than the one the user confirmed. Moves the robot ⇒ needs_confirmation.
 *   stop()             always accepted, never needs confirmation (stopping is the safe direction).
 *   cancel_passage()   "uncross"; controller present. Ends a passage, does not move the robot ⇒ no confirmation.
 *
 * A rejected call carries a reason, which goes back to the model as a role:"tool" message so it can tell the user.
 *
 * Plus the three deterministic fast paths that bypass the model entirely:
 *   is_stop(text)       ^(stop|para|detente|alto)\b   → `stop` immediately
 *   is_yes / is_no      a typed answer to a pending confirmation card (the model can never confirm)
 *   parse_direct(text)  text already in the controller grammar (cross door / cross door_N / uncross) → a card
 *
 * Pure std (no Qt, no DSR) — tested in tests/test_gate.cpp. Callers pass text already lower-cased (UTF-8 aware,
 * QString::toLower) and trimmed; ASCII is lower-cased again here so the tests need no Qt.
 */

#include <algorithm>
#include <cctype>
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
    std::string name;   ///< cross_door | stop | cancel_passage
    std::string door{};   ///< cross_door's argument ("nearest" or a door name)
};

struct GateResult
{
    bool ok = false;
    bool needs_confirmation = false;
    std::string command;   ///< the controller grammar text to send (cross door_19 | stop | uncross)
    std::string reason;    ///< why it was rejected
    std::string summary;   ///< one line for the confirmation card / transcript
};

inline std::string ascii_lower_trim(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n.!");
    return s.substr(a, b - a + 1);
}

inline GateResult validate(const ToolCall& call, const GateFacts& f)
{
    GateResult r;
    if (call.name == "stop")
    {
        r.ok = true;
        r.command = "stop";
        r.summary = "stop the robot";
        if (not f.controller_present) r.reason = "warning: the controller is not present; the stop is written anyway";
        return r;
    }
    if (call.name == "cancel_passage")
    {
        if (not f.controller_present) { r.reason = "the controller is not running"; return r; }
        r.ok = true;
        r.command = "uncross";
        r.summary = "cancel the door passage";
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
        r.needs_confirmation = true;
        r.command = "cross " + door;
        r.summary = "cross " + door + " (approach, open if shut, go through)";
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

/// A typed answer to a pending confirmation card. The whole message must be the answer.
inline bool is_yes(const std::string& text)
{
    static const std::regex re(R"(^(yes|y|yep|ok|okay|confirm|confirmed|go|go ahead|do it|sí|si|vale|adelante|confirmo|hazlo|dale)$)");
    return std::regex_match(ascii_lower_trim(text), re);
}
inline bool is_no(const std::string& text)
{
    static const std::regex re(R"(^(no|n|nope|cancel|abort|don't|dont|cancela|cancelar|mejor no)$)");
    return std::regex_match(ascii_lower_trim(text), re);
}

/// Text already in the controller's grammar. English plus the obvious Spanish phrasing.
inline std::optional<ToolCall> parse_direct(const std::string& text)
{
    const std::string t = ascii_lower_trim(text);
    static const std::regex cross_re(R"(^(cross|cruza|atraviesa)( the| la| a)? (door|puerta)$)");
    static const std::regex cross_n_re(R"(^(cross|cruza|atraviesa)( the| la)? (door_[0-9]+)$)");
    static const std::regex uncross_re(R"(^(uncross|cancel passage|cancel crossing)$)");
    std::smatch m;
    if (std::regex_match(t, m, cross_n_re)) return ToolCall{.name = "cross_door", .door = m[3].str()};
    if (std::regex_match(t, cross_re)) return ToolCall{.name = "cross_door", .door = "nearest"};
    if (std::regex_match(t, uncross_re)) return ToolCall{.name = "cancel_passage"};
    return std::nullopt;
}

}   // namespace rc::chat
