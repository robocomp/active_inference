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
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <algorithm>
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
    bool believed = true;   ///< P(exists) >= 1/2 (door_believed)
};

enum class Cause { None, Promoted, MaxTime, Return, Done, Stuck };

[[nodiscard]] inline const char* to_string(Cause c)
{
    switch (c)
    {
        case Cause::None: return "none";
        case Cause::Promoted: return "promoted";
        case Cause::MaxTime: return "max-time";
        case Cause::Return: return "return";
        case Cause::Done: return "done";
        case Cause::Stuck: return "stuck";
    }
    return "?";
}

struct Decision
{
    Cause cause = Cause::None;
    std::optional<std::string> door;
    bool is_return = false;
};

/// Recognition's posterior over [known rooms..., new] (ltsm_agent room_recognizer.cpp, `new` last): is a remembered room its best explanation?
/// A comparison between hypotheses, not a level. Before any candidates exist (fewer than 2 entries) there is nothing to lean on.
[[nodiscard]] inline bool recognition_leans_known(std::span<const float> posterior)
{
    if (posterior.size() < 2) return false;
    return *std::ranges::max_element(posterior.first(posterior.size() - 1)) >= posterior.back();
}

/// Is this door believed (P > 1/2)? A door with no published belief (an older door agent, a door restored this cycle) is taken as believed.
[[nodiscard]] inline bool door_believed(std::optional<float> p_exists) { return not p_exists.has_value() or p_exists.value() >= 0.5f; }

/// door_concept's statement (node `door_questions`): the room it speaks about and "open" | "settled".
struct DoorQuestions { std::uint64_t room_id = 0; std::string state; };

/// Has door_concept no open question about this room? A statement about another room is no answer (promotion and re-entry race it);
/// no statement at all (an older door agent) does not hold the drive. Spec 2026-11-09-undecided-candidate-look.
[[nodiscard]] inline bool doors_settled(const std::optional<DoorQuestions>& q, std::uint64_t room_id)
{
    return not q.has_value() or (q->room_id == room_id and q->state == "settled");
}

class Drive
{
public:
    explicit Drive(double max_room_s) : max_room_s_(max_room_s) {}

    // `remembered_uncrossed`: passages of this room the LTSM remembers, never crossed, whose door is not in view yet (a
    // re-entered room shows only the face it was entered by at first — final review C1). While there is one, and the room's
    // clock has not run out, the drive waits: the room's looks re-find it. No new limit: MaxRoomS bounds the wait.
    Decision step(const std::string& room, bool promoted, const std::vector<DoorCand>& doors, bool passage_active, double now_s,
                  int remembered_uncrossed = 0, bool doors_are_settled = true)
    {
        if (not start_room_.has_value()) start_room_ = room;
        if (not entered_s_.contains(room)) entered_s_[room] = now_s;
        if (passage_active) return {};
        const bool timed_out = now_s - entered_s_.at(room) > max_room_s_;
        if (not promoted and not timed_out) return {};
        const auto entry = entry_door(room);
        const auto& failed = failed_[room];
        if (const DoorCand* best = best_door(room, doors); best != nullptr)
            return {.cause = promoted ? Cause::Promoted : Cause::MaxTime, .door = best->door, .is_return = false};
        // a believed door whose approach is not offered yet is a door about to be found (clean28: born, Done, offered 0.3 s later)
        const bool door_coming = std::ranges::any_of(doors, [&](const DoorCand& d)
                                 { return d.believed and not d.found and not d.crossed and not crossed_.contains(d.door)
                                          and d.door != entry.value_or("") and not failed.contains(d.door); });
        if ((remembered_uncrossed > 0 or not doors_are_settled or door_coming) and not timed_out) return {};   // an open door question: leaving for good waits, MaxRoomS bounds it
        // a Return that already failed is not retried (final review C2): the drive stops and says so
        if (entry.has_value() and failed.contains(*entry)) return {.cause = Cause::Stuck, .door = std::nullopt, .is_return = false};
        if (entry.has_value()) return {.cause = Cause::Return, .door = *entry, .is_return = true};
        return {.cause = Cause::Done, .door = std::nullopt, .is_return = false};
    }

    /// The room's clock has not run out (a room not entered yet has not started it).
    [[nodiscard]] bool within_room_time(const std::string& room, double now_s) const
    {
        const auto it = entered_s_.find(room);
        return it == entered_s_.end() or now_s - it->second <= max_room_s_;
    }

    /// A Return is a conclusion drawn from the doors KNOWN when it was decided; a door found while the robot is still in the
    /// room is new evidence against it (explore_tour11: room_3 promoted, the drive returned 0.2 s later, DOOR_0 was born
    /// 2.6 s after that, while the robot was still driving to the entry door). The caller asks this while its Return
    /// passage has not yet committed to the doorway; a found, uncrossed, not-failed door supersedes it, nearest first.
    [[nodiscard]] std::optional<Decision> reconsider_return(const std::string& room, const std::vector<DoorCand>& doors)
    {
        if (const DoorCand* best = best_door(room, doors); best != nullptr)
            return Decision{.cause = Cause::Promoted, .door = best->door, .is_return = false};
        return std::nullopt;
    }

    /// After a crossing into `room` through `via_door` (the door of `room` the robot came in by).
    void note_entered(const std::string& room, const std::string& via_door, double now_s)
    {
        if (not via_door.empty())   // never an empty entry (final review C2): a later real one is still the first REAL entry
        {
            if (room != start_room_ and not entry_.contains(room)) entry_[room] = via_door;   // the FIRST entry: the DFS parent
            crossed_.insert(via_door);
        }   // the face just come in by is a crossed doorway in EVERY room, the start room included
        entered_s_[room] = now_s;
        failed_[room].clear();
        current_ = room;
    }

    /// The drive itself crossed `door` (its run memory; the graph's crossing fact / the LTSM may name other faces).
    void note_crossed(const std::string& door) { crossed_.insert(door); }

    /// A crossing that recognition withdrew (clean16): `door` of `from_room` was gone through, `into_room` turned out to be `from_room`. The
    /// crossing did not happen: the room the robot is in is `from_room`, the false room leaves no trace, the door is failed there.
    void note_crossing_withdrawn(const std::string& door, const std::string& entry_face, const std::string& into_room, const std::string& from_room)
    {
        crossed_.erase(door);
        crossed_.erase(entry_face);
        entry_.erase(into_room);
        entered_s_.erase(into_room);
        failed_.erase(into_room);
        failed_[from_room].insert(door);
        current_ = from_room;
    }

    /// The passage through `door` failed in the room the robot is in.
    void note_failed(const std::string& door) { failed_[current_.value_or(start_room_.value_or(""))].insert(door); }

    [[nodiscard]] std::optional<std::string> entry_door(const std::string& room) const
    {
        if (const auto it = entry_.find(room); it != entry_.end()) return it->second;
        return std::nullopt;
    }

private:
    /// The nearest door of `room` that is found, not crossed, not the room's entry and not failed this visit (nullptr = none).
    [[nodiscard]] const DoorCand* best_door(const std::string& room, const std::vector<DoorCand>& doors)
    {
        const auto entry = entry_door(room);
        const auto& failed = failed_[room];
        const DoorCand* best = nullptr;
        for (const auto& d : doors)
        {
            if (not d.found or d.crossed or crossed_.contains(d.door) or (entry.has_value() and d.door == *entry) or failed.contains(d.door)) continue;
            if (best == nullptr or d.dist_m < best->dist_m) best = &d;
        }
        return best;
    }

    double max_room_s_;
    std::optional<std::string> start_room_, current_;
    std::map<std::string, std::string> entry_;
    std::map<std::string, double> entered_s_;
    std::map<std::string, std::set<std::string>> failed_;
    std::set<std::string> crossed_;   ///< faces this drive crossed or entered by (explore_tour1: a re-entered face looked new)
};
}  // namespace rc::explore
