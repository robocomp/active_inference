#pragma once
// DoorRelation — pick a door by what it LEADS TO, not by its name (door-identity-design.md, part 3).
//
// A `door_N` name is a graph slot, not an identity: door nodes are re-made on every crossing. What survives is the relation:
//   cross <room>   the live door of the current room that has a `recall_<room>` child (ltsm posts it under the door that
//                  leads to that remembered room);
//   cross new      a live door with NO recall child and NO passage `match` edge: a door nothing has been learnt about yet.
//                  Several: the one a door_concept look has targeted (aff_door_look_<name> exists or existed), else the
//                  nearest to the robot (ties by name, so the pick is deterministic).
// Resolution is LAZY: the recall (or the new door) may appear late, so the step holds (RelationHold) and re-resolves each
// poll. Pure: the controller gathers DoorFacts from the graph; no graph, no Qt here.
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../../common/mission/mission_ir.h"

namespace rc
{
struct DoorFacts
{
    std::string name{};
    std::set<std::string> recall_rooms{};   ///< rooms R for which the door has a child `recall_R`
    bool has_match = false;                 ///< a passage `match` edge points at the door
    bool looked = false;                    ///< an aff_door_look_<name> node exists or existed
    double dist = std::numeric_limits<double>::infinity();   ///< metres from the robot (inf = unknown)
};

enum class DoorRelationKind { None, Room, New };

/// A Node-ref name that is a relation rather than a door: "new", or a room ("room_2").
[[nodiscard]] inline DoorRelationKind relation_of(const mission::Ref &r)
{
    if (r.kind != mission::Ref::Kind::Node) return DoorRelationKind::None;
    if (r.name == "new") return DoorRelationKind::New;
    if (r.name.starts_with("room")) return DoorRelationKind::Room;
    return DoorRelationKind::None;
}

/// The door the relation names among the live doors of the current room, if any resolves now.
[[nodiscard]] inline std::optional<std::string> pick_door(const mission::Ref &r, const std::vector<DoorFacts> &doors)
{
    const auto kind = relation_of(r);
    const DoorFacts *best = nullptr;
    for (const auto &d : doors)
    {
        if (kind == DoorRelationKind::Room)
        {
            if (not d.recall_rooms.contains(r.name)) continue;
            if (best == nullptr or d.name < best->name) best = &d;
        }
        else if (kind == DoorRelationKind::New)
        {
            if (not d.recall_rooms.empty() or d.has_match) continue;
            const bool better = best == nullptr or (d.looked != best->looked ? d.looked
                                                    : d.dist != best->dist   ? d.dist < best->dist
                                                                             : d.name < best->name);
            if (better) best = &d;
        }
    }
    return best != nullptr ? std::optional<std::string>(best->name) : std::nullopt;
}

enum class RoomEvent { None, Renamed, Changed };

/// The room the robot is in, tracked by node id. The id says WHICH room; the name is owned by room_concept and can change
/// under the same id (a recognised room adopts its remembered name in place: room_4 -> room_2). A rename is not a room
/// change: prev_name stays the name the room had when the robot LEFT it (so ltsm's recall_<prev> is found by that name).
inline RoomEvent track_room(std::uint64_t &id, std::string &name, std::string &prev_name, std::uint64_t new_id, const std::string &new_name)
{
    if (new_id != id)
    {
        if (not name.empty()) prev_name = name;
        id = new_id; name = new_name;
        return RoomEvent::Changed;
    }
    if (name != new_name) { name = new_name; return RoomEvent::Renamed; }
    return RoomEvent::None;
}

/// The step waits for its door to resolve; it has a budget so it cannot hang (like RoomHold).
class RelationHold
{
public:
    /// ⚑ FLAGGED CONSTANT: a MISSION-LEVEL timeout, not an ordering guess. The release is the FACT (pick_door resolves: ltsm
    /// marked recall_<room>, or a new door was seen); this only bounds how long the mission waits for a fact that may never
    /// come (`cross new` is gated on ltsm posting the way-back recall; ltsm down, door never seen) so the step fails as 'door_unresolved'. The step's own budget_s wins when smaller.
    static constexpr std::uint64_t kHoldBudgetMs = 120'000;

    enum class Verdict { Waiting, Ready, TimedOut };

    void hold(const mission::Step &s, std::uint64_t now_ms) { pending_ = s; since_ = now_ms; }
    bool holding() const { return pending_.has_value(); }
    void clear() { pending_.reset(); }

    /// `door` = what pick_door gives now. Ready releases the hold.
    Verdict check(const std::optional<std::string> &door, std::uint64_t now_ms)
    {
        if (door.has_value()) { pending_.reset(); return Verdict::Ready; }
        const double b = pending_.has_value() ? pending_->number("budget_s").value_or(0.0) : 0.0;
        const std::uint64_t budget = b > 0 ? std::min<std::uint64_t>(kHoldBudgetMs, std::uint64_t(b * 1000.0)) : kHoldBudgetMs;
        if (now_ms - since_ >= budget) { pending_.reset(); return Verdict::TimedOut; }
        return Verdict::Waiting;
    }

private:
    std::optional<mission::Step> pending_;
    std::uint64_t since_ = 0;
};
}   // namespace rc
