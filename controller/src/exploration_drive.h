#pragma once
// exploration_drive.h — the EXPLORATION DRIVE's decision core (spec docs/superpowers/specs/2026-11-05-exploration-tour-design.md).
// Pure: no DSR. SpecificWorker builds the inputs each cycle and acts through start_door_passage.
//
// Doors are the curiosity (door_concept's looks find them); crossing is the intrinsic motivation. In a room:
//   explore (the open contest: afford_room + door looks) until the room is PROMOTED, or MaxRoomS has passed (the one
//   deliberate time limit, decision 5 — every leave says which); then cross the NEAREST door that is found and not yet
//   crossed; if there is none, return through the door the room was FIRST entered by (a depth-first search whose root is
//   the start room, which has no entry door); in the start room with nothing left: Done.
// A door whose passage failed is skipped for the rest of that visit (re-entering the room clears it) — UNLESS there is NEW
// EVIDENCE about it (DoorEvidence below): a model event, never a timer (clean58, 2026-11-12).
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace rc::explore
{
/// ★ WHAT THE DRIVE KNOWS OF A DOOR THAT CAN CHANGE ITS MIND ABOUT A FAILED PASSAGE (clean58, 2026-11-12: the robot parked 26+ min at a
///   phantom door_3 whose cross was never offered; owner: "the robot should do something if it is not allowed to cross"). A failed
///   passage skips the door; it may be retried only once this differs from what it was at the failure — every field a DISCRETE model
///   event, no level of our own:
///     node_id      the door node re-born (door_concept deleted it and made another under the same name: a new hypothesis);
///     exists_map   its MAP on existence flipped (door_believed: P(exists) >= 1/2);
///     step_offered door_concept now offers a step it did not (`open` or `cross` on offer: the very thing a failed passage waited for).
///   NOT evidence here: a geometry refit (model_generation — door_concept bumps it every 3 mm, it would retry the door at once) and
///   the open/shut MAP (clean58's door_3 sat at P(passable) 0.54: noise around 1/2 would retake the door on every flicker). Both
///   reach the drive anyway through the step door_concept offers or withdraws.
struct DoorEvidence
{
    std::uint64_t node_id = 0;
    bool exists_map = true;
    bool step_offered = false;
    bool operator==(const DoorEvidence&) const = default;
};

struct DoorCand
{
    std::string door;
    bool found = false;     ///< door_concept offers aff_<door>_approach
    bool crossed = false;   ///< a crossing fact names it, or the LTSM holds a passage through it with crossings >= 1
    double dist_m = std::numeric_limits<double>::infinity();
    bool believed = true;   ///< P(exists) >= 1/2 (door_believed)
    DoorEvidence evidence{};///< the door's discrete belief now (a failed door is retried once this changes)
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
        if (const DoorCand* best = best_door(room, doors); best != nullptr)
            return {.cause = promoted ? Cause::Promoted : Cause::MaxTime, .door = best->door, .is_return = false};
        // a believed door whose approach is not offered yet is a door about to be found (clean28: born, Done, offered 0.3 s later)
        const bool door_coming = std::ranges::any_of(doors, [&](const DoorCand& d)
                                 { return d.believed and not d.found and not d.crossed and not crossed_.contains(d.door)
                                          and d.door != entry.value_or("") and not still_failed(room, d); });
        if ((remembered_uncrossed > 0 or not doors_are_settled or door_coming) and not timed_out) return {};   // an open door question: leaving for good waits, MaxRoomS bounds it
        // ★ THE START ROOM WITH NO DOOR AT ALL is not evidence of a house with one room (clean43, 2026-11-11: the robot started in
        //   room_2, a small room whose shape converged — promoted — before it moved; every door was shut at the start and none was
        //   born yet; the drive read "no uncrossed door anywhere" and ended the tour DONE 5 s in). Every other room has at least the
        //   door it was entered by; the start room has none until one is found. So no door known there is an open door question,
        //   bounded by the same MaxRoomS — the room's own exploration goes on and finds its doors.
        if (not entry.has_value() and doors.empty() and not timed_out) return {};
        // a Return that already failed is not retried (final review C2) until there is new evidence about its door: Stuck says so
        if (entry.has_value() and still_failed(room, *entry, doors)) return {.cause = Cause::Stuck, .door = std::nullopt, .is_return = false};
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
        failed_[from_room].insert_or_assign(door, DoorEvidence{});
        current_ = from_room;
    }

    /// The passage through `door` failed in the room the robot is in; `evidence` = the door's discrete belief at the failure (it is
    /// skipped until that changes — DoorEvidence). Default: a door with no evidence to read, skipped for the visit.
    void note_failed(const std::string& door, DoorEvidence evidence = {})
    {
        failed_[current_.value_or(start_room_.value_or(""))].insert_or_assign(door, evidence);
    }
    /// Doors whose failure was lifted by new evidence since the last call (the caller logs them).
    [[nodiscard]] std::vector<std::string> take_retried() { return std::exchange(retried_, {}); }
    /// Is `door` failed in `room` now (its failure not lifted by new evidence)?
    [[nodiscard]] bool is_failed(const std::string& room, const std::string& door) const
    {
        const auto r = failed_.find(room);
        return r != failed_.end() and r->second.contains(door);
    }

    [[nodiscard]] std::optional<std::string> entry_door(const std::string& room) const
    {
        if (const auto it = entry_.find(room); it != entry_.end()) return it->second;
        return std::nullopt;
    }

private:
    /// The nearest door of `room` that is found, believed, not crossed, not the room's entry and not failed this visit (nullptr = none).
    /// ★ BELIEVED is read HERE, each decision, not frozen into `failed_` (clean46, 2026-11-11: door_7 = DOOR_0 dipped to P 0.41 for a
    ///   few seconds, was marked failed for the visit, recovered to 0.97 and was never taken again — the tour ended DONE at 2/3 doors).
    ///   A door below 1/2 is skipped while it stays there, and comes back when its belief does.
    [[nodiscard]] const DoorCand* best_door(const std::string& room, const std::vector<DoorCand>& doors)
    {
        const auto entry = entry_door(room);
        const DoorCand* best = nullptr;
        for (const auto& d : doors)
        {
            if (not d.found or not d.believed or d.crossed or crossed_.contains(d.door) or (entry.has_value() and d.door == *entry) or still_failed(room, d)) continue;
            if (best == nullptr or d.dist_m < best->dist_m) best = &d;
        }
        return best;
    }

    /// Is this door's failure in `room` still standing? NEW EVIDENCE (its DoorEvidence now differs from the one at the failure) lifts
    /// it, once: the record is erased, so the door is a candidate again and a second failure records the evidence anew.
    bool still_failed(const std::string& room, const DoorCand& d)
    {
        auto& failed = failed_[room];
        const auto it = failed.find(d.door);
        if (it == failed.end()) return false;
        if (it->second == d.evidence) return true;
        failed.erase(it);
        retried_.push_back(d.door);
        return false;
    }
    /// The same for a door named only (the entry door): with no candidate for it this cycle there is nothing new — still failed.
    bool still_failed(const std::string& room, const std::string& door, const std::vector<DoorCand>& doors)
    {
        const auto c = std::ranges::find(doors, door, &DoorCand::door);
        if (c != doors.end()) return still_failed(room, *c);
        return failed_[room].contains(door);
    }

    double max_room_s_;
    std::optional<std::string> start_room_, current_;
    std::map<std::string, std::string> entry_;
    std::map<std::string, double> entered_s_;
    std::map<std::string, std::map<std::string, DoorEvidence>> failed_;   ///< room -> door -> its evidence at the failure
    std::vector<std::string> retried_;   ///< failures lifted by new evidence (take_retried)
    std::set<std::string> crossed_;   ///< faces this drive crossed or entered by (explore_tour1: a re-entered face looked new)
};
}  // namespace rc::explore
