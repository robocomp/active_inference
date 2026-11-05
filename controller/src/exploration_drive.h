#pragma once
// exploration_drive.h — the EXPLORATION DRIVE's decision core (spec docs/superpowers/specs/2026-11-05-exploration-tour-design.md).
// Pure: no DSR. SpecificWorker builds the inputs each cycle and acts through start_door_passage.
//
// Doors are the curiosity (door_concept's looks find them); crossing is the intrinsic motivation. In a room:
//   explore (the open contest: afford_room + door looks) until the room is PROMOTED, or MaxRoomS has passed (the one
//   deliberate time limit, decision 5 — every leave says which); then cross the NEAREST door that is found and not yet
//   crossed; if there is none, return through the door the room was FIRST entered by (a depth-first search whose root is
//   the start room, which has no entry door); in the start room with nothing left: Done.
// A door whose passage failed is skipped for the rest of that visit (re-entering the room clears it).
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace rc::explore
{
struct DoorCand
{
    std::string door;
    bool found = false;     ///< door_concept offers aff_<door>_approach
    bool crossed = false;   ///< a crossing fact names it, or the LTSM holds a passage through it with crossings >= 1
    double dist_m = std::numeric_limits<double>::infinity();
};

enum class Cause { None, Promoted, MaxTime, Return, Done };

[[nodiscard]] inline const char* to_string(Cause c)
{
    switch (c)
    {
        case Cause::None: return "none";
        case Cause::Promoted: return "promoted";
        case Cause::MaxTime: return "max-time";
        case Cause::Return: return "return";
        case Cause::Done: return "done";
    }
    return "?";
}

struct Decision
{
    Cause cause = Cause::None;
    std::optional<std::string> door;
    bool is_return = false;
};

class Drive
{
public:
    explicit Drive(double max_room_s) : max_room_s_(max_room_s) {}

    Decision step(const std::string& room, bool promoted, const std::vector<DoorCand>& doors, bool passage_active, double now_s)
    {
        if (not start_room_.has_value()) start_room_ = room;
        if (not entered_s_.contains(room)) entered_s_[room] = now_s;
        if (passage_active) return {};
        const bool timed_out = now_s - entered_s_.at(room) > max_room_s_;
        if (not promoted and not timed_out) return {};
        const auto entry = entry_door(room);
        const auto& failed = failed_[room];
        const DoorCand* best = nullptr;
        for (const auto& d : doors)
        {
            if (not d.found or d.crossed or crossed_.contains(d.door) or (entry.has_value() and d.door == *entry) or failed.contains(d.door)) continue;
            if (best == nullptr or d.dist_m < best->dist_m) best = &d;
        }
        if (best != nullptr) return {.cause = promoted ? Cause::Promoted : Cause::MaxTime, .door = best->door, .is_return = false};
        if (entry.has_value()) return {.cause = Cause::Return, .door = *entry, .is_return = true};
        return {.cause = Cause::Done, .door = std::nullopt, .is_return = false};
    }

    /// After a crossing into `room` through `via_door` (the door of `room` the robot came in by).
    void note_entered(const std::string& room, const std::string& via_door, double now_s)
    {
        if (room != start_room_ and not entry_.contains(room)) entry_[room] = via_door;   // the FIRST entry: the DFS parent
        crossed_.insert(via_door);   // the face just come in by is a crossed doorway in EVERY room, the start room included
        entered_s_[room] = now_s;
        failed_[room].clear();
        current_ = room;
    }

    /// The drive itself crossed `door` (its run memory; the graph's crossing fact / the LTSM may name other faces).
    void note_crossed(const std::string& door) { crossed_.insert(door); }

    /// The passage through `door` failed in the room the robot is in.
    void note_failed(const std::string& door) { failed_[current_.value_or(start_room_.value_or(""))].insert(door); }

    [[nodiscard]] std::optional<std::string> entry_door(const std::string& room) const
    {
        if (const auto it = entry_.find(room); it != entry_.end()) return it->second;
        return std::nullopt;
    }

private:
    double max_room_s_;
    std::optional<std::string> start_room_, current_;
    std::map<std::string, std::string> entry_;
    std::map<std::string, double> entered_s_;
    std::map<std::string, std::set<std::string>> failed_;
    std::set<std::string> crossed_;   ///< faces this drive crossed or entered by (explore_tour1: a re-entered face looked new)
};
}  // namespace rc::explore
